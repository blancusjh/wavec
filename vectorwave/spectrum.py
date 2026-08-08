"""The angular spectrum — the common currency of every propagator here.

Whatever produced the light (an aplanatic pupil, refraction at a curved
interface, a field on a plane), it is reduced to a vector angular spectrum
``A(kx, ky)``.  Propagation is then a phase, and evaluating the field anywhere
is one transform.  This keeps a single, well-tested path from many sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import Field
from .grids import Grid

__all__ = ["AngularSpectrum"]


@dataclass
class AngularSpectrum:
    """Vector angular spectrum ``A`` of shape ``(3, Ny, Nx)`` on ``grid``'s k-mesh.

    ``sigma`` is the propagation sense (+1 toward +z, -1 toward -z).
    """

    A: np.ndarray
    grid: Grid
    wavelength: float = 1.0
    n: float = 1.0
    sigma: int = +1

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=complex)
        if self.A.shape[0] != 3:
            raise ValueError("A must have three vector components")

    # ------------------------------------------------------------------ basics
    @property
    def k(self) -> float:
        return 2 * np.pi * self.n / self.wavelength

    @property
    def kz(self) -> np.ndarray:
        return self.grid.kz(self.k)

    def transversality_residual(self) -> float:
        """``max |k.A| / (k max|A|)`` — zero for a physical (divergence-free) field."""
        KX, KY = self.grid.KXY
        mask = self.grid.propagating(self.k)
        kdotA = KX * self.A[0] + KY * self.A[1] + self.sigma * self.kz * self.A[2]
        denom = np.abs(self.A[:, mask]).max() * self.k
        if denom == 0:
            return 0.0
        return float(np.abs(kdotA[mask]).max() / denom)

    def impose_transversality(self) -> "AngularSpectrum":
        """Rebuild ``A_z`` from the transverse pair so that ``k.A = 0`` exactly."""
        KX, KY = self.grid.KXY
        kz = np.maximum(self.kz, 1e-12 * self.k)
        Az = -(KX * self.A[0] + KY * self.A[1]) / (self.sigma * kz)
        A = self.A.copy()
        A[2] = np.where(self.grid.propagating(self.k), Az, 0.0)
        return AngularSpectrum(A, self.grid, self.wavelength, self.n, self.sigma)

    def bandlimit(self, fraction: float = 0.999) -> "AngularSpectrum":
        """Zero the evanescent sector (and anything beyond ``fraction * k``)."""
        A = self.A * self.grid.propagating(self.k, fraction)[None]
        return AngularSpectrum(A, self.grid, self.wavelength, self.n, self.sigma)

    def apodize_horizon(self, start: float = 0.90, stop: float = 0.985) -> "AngularSpectrum":
        """Cosine roll-off toward the light cone.

        Near-grazing components are the ones the ``1/kz`` factors amplify; a
        smooth taper there removes that numerical noise without touching the
        physical band.
        """
        kr = self.grid.KR / self.k
        w = np.ones_like(kr)
        ramp = np.clip((kr - start) / max(stop - start, 1e-12), 0.0, 1.0)
        w = np.where(kr >= start, 0.5 * (1 + np.cos(np.pi * ramp)), w)
        w[kr >= stop] = 0.0
        return AngularSpectrum(self.A * w[None], self.grid, self.wavelength,
                               self.n, self.sigma)

    # ------------------------------------------------------------- propagation
    def propagated(self, z: float) -> np.ndarray:
        """The spectrum advanced by ``z`` (evanescent orders removed)."""
        mask = self.grid.propagating(self.k)
        return self.A * (mask * np.exp(1j * self.sigma * self.kz * z))[None]

    def field(self, z: float = 0.0) -> Field:
        """Synthesise the field on the spectrum's own grid at height ``z``."""
        Az = self.propagated(z)
        E = np.fft.ifft2(Az, axes=(-2, -1))
        E = np.fft.fftshift(E, axes=(-2, -1)) / (self.grid.dx * self.grid.dy)
        return Field(E[0], E[1], self.grid, self.wavelength, self.n, Ez=E[2], z=z)

    def field_on(self, x: np.ndarray, y: np.ndarray, z: float = 0.0) -> Field:
        """Exact band-limited synthesis on an arbitrary fine grid (zoom DFT).

        No interpolation: the transform is evaluated directly at the requested
        coordinates, so the sampling of the output is decoupled from the grid
        that carries the spectrum.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        Az = self.propagated(z)
        kx = 2 * np.pi * np.fft.fftfreq(self.grid.x.size, d=self.grid.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.grid.y.size, d=self.grid.dy)
        Fx = np.exp(1j * np.outer(x, kx))
        Fy = np.exp(1j * np.outer(y, ky))
        scale = 1.0 / ((self.grid.x.size * self.grid.dx) *
                       (self.grid.y.size * self.grid.dy))
        out = np.empty((3, y.size, x.size), dtype=complex)
        for c in range(3):
            out[c] = (Fy @ Az[c] @ Fx.T) * scale
        return Field(out[0], out[1], Grid(x, y), self.wavelength, self.n,
                     Ez=out[2], z=z)

    def meridional(self, x: np.ndarray, z: np.ndarray, y0: float = 0.0):
        """Field in the ``(x, z)`` plane; returns ``(components, x, z)``.

        ``components`` has shape ``(3, len(z), len(x))``.
        """
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        out = np.empty((3, z.size, x.size), dtype=complex)
        for i, zz in enumerate(z):
            f = self.field_on(x, np.array([y0]), zz)
            out[0, i], out[1, i], out[2, i] = f.Ex[0], f.Ey[0], f.components[2][0]
        return out, x, z

    def focus_scan(self, z: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
        """On-axis (or small-box) intensity versus ``z`` — used to locate a focus."""
        pts = np.array([0.0]) if x is None else np.asarray(x, dtype=float)
        return np.array([self.field_on(pts, pts, zz).intensity.sum() for zz in z])

    def best_focus(self, z: np.ndarray) -> float:
        """The ``z`` in the scan with the largest on-axis intensity."""
        return float(np.asarray(z)[np.argmax(self.focus_scan(np.asarray(z)))])
