"""Sampling grids in real space and in the angular (k) domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Grid"]


@dataclass(frozen=True)
class Grid:
    """A centred, uniform Cartesian grid.

    ``x`` and ``y`` are 1-D axes in physical units (the same units used for the
    wavelength everywhere else in the package).  The grid is the common
    substrate for :class:`~vectorwave.fields.Field` samples and for the
    reciprocal (k-space) representation used by the propagators.
    """

    x: np.ndarray
    y: np.ndarray

    # ---------------------------------------------------------------- builders
    @classmethod
    def square(cls, half_width: float, n: int) -> "Grid":
        """``n x n`` samples spanning ``[-half_width, +half_width]``."""
        ax = np.linspace(-half_width, half_width, int(n))
        return cls(ax, ax)

    @classmethod
    def from_spacing(cls, spacing: float, n: int) -> "Grid":
        """``n x n`` samples with the given pitch, centred on the origin."""
        ax = (np.arange(int(n)) - int(n) // 2) * float(spacing)
        return cls(ax, ax)

    # -------------------------------------------------------------- properties
    @property
    def shape(self) -> tuple[int, int]:
        return (self.y.size, self.x.size)

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float:
        return float(self.y[1] - self.y[0])

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` — ready for ``imshow(extent=...)``."""
        return (float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1]))

    @property
    def XY(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, indexing="xy")

    @property
    def R(self) -> np.ndarray:
        X, Y = self.XY
        return np.hypot(X, Y)

    @property
    def PHI(self) -> np.ndarray:
        X, Y = self.XY
        return np.arctan2(Y, X)

    # ------------------------------------------------------------------ k-space
    @property
    def KXY(self) -> tuple[np.ndarray, np.ndarray]:
        """Angular frequencies in FFT order (rad / length)."""
        kx = 2 * np.pi * np.fft.fftfreq(self.x.size, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.y.size, d=self.dy)
        return np.meshgrid(kx, ky, indexing="xy")

    @property
    def KR(self) -> np.ndarray:
        KX, KY = self.KXY
        return np.hypot(KX, KY)

    def kz(self, k: float) -> np.ndarray:
        """Longitudinal wavenumber; real inside the light cone, 0 outside."""
        KR = self.KR
        return np.sqrt(np.maximum(k**2 - KR**2, 0.0))

    def propagating(self, k: float, margin: float = 0.999) -> np.ndarray:
        """Boolean mask of the propagating (non-evanescent) sector."""
        return self.KR <= margin * k

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Grid({self.shape[0]}x{self.shape[1]}, dx={self.dx:.4g})"
