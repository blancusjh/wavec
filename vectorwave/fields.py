"""Sampled vector fields and their polarization diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import Grid

__all__ = ["Field", "Polarization"]


@dataclass
class Polarization:
    """Point-wise polarization state derived from the transverse components."""

    s0: np.ndarray          # total transverse intensity
    s1: np.ndarray
    s2: np.ndarray
    s3: np.ndarray

    @property
    def dop(self) -> np.ndarray:
        """Degree of polarization (1 for a fully coherent field)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(np.sqrt(self.s1**2 + self.s2**2 + self.s3**2) / self.s0)

    @property
    def orientation(self) -> np.ndarray:
        """Major-axis azimuth in radians, in ``[-pi/2, pi/2]``."""
        return 0.5 * np.arctan2(self.s2, self.s1)

    @property
    def ellipticity(self) -> np.ndarray:
        """Ellipticity angle in radians; 0 linear, +-pi/4 circular."""
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num(self.s3 / np.sqrt(self.s1**2 + self.s2**2 + self.s3**2))
        return 0.5 * np.arcsin(np.clip(r, -1, 1))


@dataclass
class Field:
    """A vector field sampled on a :class:`~vectorwave.grids.Grid`.

    ``Ex``, ``Ey`` and (optionally) ``Ez`` are complex arrays of the grid's
    shape.  ``wavelength`` is the vacuum wavelength and ``n`` the refractive
    index of the medium the samples live in.
    """

    Ex: np.ndarray
    Ey: np.ndarray
    grid: Grid
    wavelength: float = 1.0
    n: float = 1.0
    Ez: np.ndarray | None = None
    z: float = 0.0

    # ------------------------------------------------------------------ basics
    def __post_init__(self):
        self.Ex = np.asarray(self.Ex, dtype=complex)
        self.Ey = np.asarray(self.Ey, dtype=complex)
        if self.Ez is not None:
            self.Ez = np.asarray(self.Ez, dtype=complex)

    @property
    def k(self) -> float:
        """Wavenumber in the medium."""
        return 2 * np.pi * self.n / self.wavelength

    @property
    def components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zero = np.zeros_like(self.Ex)
        return (self.Ex, self.Ey, self.Ez if self.Ez is not None else zero)

    @property
    def intensity(self) -> np.ndarray:
        """``|Ex|^2 + |Ey|^2 + |Ez|^2``."""
        return sum(np.abs(c) ** 2 for c in self.components)

    @property
    def amplitude(self) -> np.ndarray:
        return np.sqrt(self.intensity)

    @property
    def energy(self) -> float:
        return float(self.intensity.sum() * self.grid.dx * self.grid.dy)

    def component_fractions(self) -> dict[str, float]:
        """Share of the total energy carried by each Cartesian component."""
        tot = self.intensity.sum()
        if tot == 0:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        return {name: float((np.abs(c) ** 2).sum() / tot)
                for name, c in zip("xyz", self.components)}

    # ---------------------------------------------------------- polarization
    def polarization(self) -> Polarization:
        """Stokes parameters of the transverse part."""
        ex, ey = self.Ex, self.Ey
        return Polarization(
            s0=np.abs(ex) ** 2 + np.abs(ey) ** 2,
            s1=np.abs(ex) ** 2 - np.abs(ey) ** 2,
            s2=2 * np.real(ex * np.conj(ey)),
            s3=-2 * np.imag(ex * np.conj(ey)),
        )

    # -------------------------------------------------------------- utilities
    def cut(self, axis: str = "x"):
        """Central 1-D cut of the intensity: returns ``(coordinate, values)``."""
        I = self.intensity
        if axis == "x":
            return self.grid.x, I[I.shape[0] // 2, :]
        return self.grid.y, I[:, I.shape[1] // 2]

    def fwhm(self, axis: str = "x") -> float:
        """Full width at half maximum of the intensity along a central cut."""
        c, v = self.cut(axis)
        v = v / v.max()
        above = np.flatnonzero(v >= 0.5)
        if above.size < 2:
            return float("nan")
        return float(c[above[-1]] - c[above[0]])

    def to_vecdiff(self):
        """Wrap as a ``vecdiff.fields.Field`` for its polarization plots."""
        from vecdiff.fields import Field as VField      # noqa: PLC0415
        from vecdiff.grid import Grid as VGrid          # noqa: PLC0415
        X, Y = self.grid.XY
        return VField(self.Ex, self.Ey, VGrid.from_cartesian(X, Y),
                      symmetry=None, Ez=self.Ez)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        f = self.component_fractions()
        return (f"Field({self.grid.shape[0]}x{self.grid.shape[1]}, z={self.z:.4g}, "
                f"Ex/Ey/Ez = {f['x']:.3f}/{f['y']:.3f}/{f['z']:.3f})")
