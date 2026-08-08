"""Pupils and aplanatic (Richards-Wolf / Debye) focusing.

A :class:`Pupil` is an exit-pupil description: a numerical aperture, an
aberration wavefront, an apodization and an input polarization.  Asking it for
a spectrum applies the aplanatic polarization projection — the rotation each
ray's field undergoes as the pupil bends it toward the focus — and hands back
an :class:`~vectorwave.spectrum.AngularSpectrum` you can propagate anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .grids import Grid
from .spectrum import AngularSpectrum

__all__ = ["Pupil", "POLARIZATIONS"]


def _pol_linear(angle: float):
    def f(rho, phi):
        return (np.full_like(rho, np.cos(angle), dtype=complex),
                np.full_like(rho, np.sin(angle), dtype=complex))
    return f


def _pol_circular(handedness: int = +1):
    def f(rho, phi):
        return (np.ones_like(rho, dtype=complex),
                np.full_like(rho, 1j * handedness, dtype=complex))
    return f


def _pol_radial(rho, phi):
    return np.cos(phi).astype(complex), np.sin(phi).astype(complex)


def _pol_azimuthal(rho, phi):
    return (-np.sin(phi)).astype(complex), np.cos(phi).astype(complex)


def _pol_vortex(charge: int, base: str = "x"):
    inner = _pol_linear(0.0) if base == "x" else _pol_circular()

    def f(rho, phi):
        ex, ey = inner(rho, phi)
        ramp = np.exp(1j * charge * phi)
        return ex * ramp, ey * ramp
    return f


#: Ready-made input polarizations. Any ``(rho, phi) -> (Ex, Ey)`` callable works too.
POLARIZATIONS: dict[str, Callable] = {
    "x": _pol_linear(0.0),
    "y": _pol_linear(np.pi / 2),
    "45": _pol_linear(np.pi / 4),
    "circular": _pol_circular(+1),
    "circular-left": _pol_circular(-1),
    "radial": _pol_radial,
    "azimuthal": _pol_azimuthal,
    "vortex1": _pol_vortex(1),
    "vortex2": _pol_vortex(2),
}


@dataclass
class Pupil:
    """An exit pupil ready to be focused.

    Parameters
    ----------
    na, n, wavelength
        Image-space numerical aperture, refractive index and vacuum wavelength.
    wavefront
        ``W(u, v) -> waves`` on normalized pupil coordinates, or ``None`` for a
        perfect pupil.
    amplitude
        ``T(rho, phi) -> amplitude`` transmission (``None`` for uniform).
    polarization
        Key of :data:`POLARIZATIONS` or a ``(rho, phi) -> (Ex, Ey)`` callable.
    apodization
        ``"aplanatic"`` applies the ``sqrt(cos theta)`` sine-condition factor
        (correct for an objective obeying the Abbe sine condition);
        ``"uniform"`` applies none.
    """

    na: float
    n: float = 1.0
    wavelength: float = 1.0
    wavefront: Callable | None = None
    amplitude: Callable | None = None
    polarization: str | Callable = "x"
    apodization: str = "aplanatic"

    # ------------------------------------------------------------------ basics
    @property
    def k(self) -> float:
        return 2 * np.pi * self.n / self.wavelength

    @property
    def sin_theta_max(self) -> float:
        return float(self.na / self.n)

    @property
    def theta_max(self) -> float:
        return float(np.arcsin(np.clip(self.sin_theta_max, 0, 1)))

    def _polarization_fn(self) -> Callable:
        if callable(self.polarization):
            return self.polarization
        try:
            return POLARIZATIONS[self.polarization]
        except KeyError as exc:  # pragma: no cover - user error path
            raise KeyError(f"unknown polarization {self.polarization!r}; "
                           f"choose from {sorted(POLARIZATIONS)}") from exc

    # ------------------------------------------------------------------ physics
    def spectrum(self, grid: Grid, sigma: int = +1,
                 edge_softness: float = 0.02) -> AngularSpectrum:
        """Aplanatic vector spectrum of this pupil on ``grid``'s k-mesh.

        The Richards-Wolf projection is applied: the incident transverse field
        is split into azimuthal (s) and meridional (p) parts, the p part is
        rotated by the ray angle and sheds a longitudinal component, and the
        result is weighted by the apodization, the aberration phase and the
        Debye measure.
        """
        k = self.k
        KX, KY = grid.KXY
        kr = np.hypot(KX, KY)
        sin_t = np.clip(kr / k, 0.0, 1.0)
        cos_t = np.sqrt(np.clip(1.0 - sin_t**2, 0.0, None))
        phi = np.arctan2(KY, KX)
        rho = sin_t / max(self.sin_theta_max, 1e-12)          # normalized radius

        # soft-edged aperture: a hard rim radiates a spurious edge wave
        w = np.clip((1.0 - rho) / max(edge_softness, 1e-9), 0.0, 1.0)
        aperture = np.where(rho <= 1.0, 0.5 * (1 - np.cos(np.pi * w)), 0.0)

        ex0, ey0 = self._polarization_fn()(rho, phi)
        ex0 = np.asarray(ex0, dtype=complex)
        ey0 = np.asarray(ey0, dtype=complex)

        # s/p decomposition on the pupil, then the aplanatic bend
        a_p = ex0 * np.cos(phi) + ey0 * np.sin(phi)
        a_s = -ex0 * np.sin(phi) + ey0 * np.cos(phi)
        Ex = a_p * cos_t * np.cos(phi) - a_s * np.sin(phi)
        Ey = a_p * cos_t * np.sin(phi) + a_s * np.cos(phi)
        Ez = -a_p * sin_t

        weight = aperture.astype(complex)
        if self.apodization == "aplanatic":
            weight = weight * np.sqrt(cos_t)
        if self.amplitude is not None:
            weight = weight * np.asarray(self.amplitude(rho, phi), dtype=complex)
        if self.wavefront is not None:
            u, v = rho * np.cos(phi), rho * np.sin(phi)
            weight = weight * np.exp(2j * np.pi * np.asarray(self.wavefront(u, v),
                                                             dtype=float))
        # Debye measure: d^2k = k^2 sin cos dtheta dphi  ->  weight by 1/kz
        kz = np.maximum(k * cos_t, 1e-12 * k)
        weight = weight / kz

        A = np.stack([Ex * weight, Ey * weight, Ez * weight])
        return AngularSpectrum(A, grid, self.wavelength, self.n, sigma)

    # ---------------------------------------------------------------- shortcuts
    def focus(self, grid: Grid, x: np.ndarray | None = None,
              y: np.ndarray | None = None, z: float = 0.0):
        """Field at (or near) the focus; ``x``/``y`` allow a zoomed window."""
        spec = self.spectrum(grid)
        if x is None:
            return spec.field(z)
        y = x if y is None else y
        return spec.field_on(x, y, z)

    def airy_fwhm(self) -> float:
        """Diffraction-limited intensity FWHM (``~0.514 lambda / NA``)."""
        return 0.5144 * self.wavelength / self.na

    def depth_of_focus(self) -> float:
        """Rayleigh depth of focus ``lambda n / NA^2``."""
        return self.wavelength * self.n / self.na**2
