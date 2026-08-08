"""Exit-pupil wavefronts.

A wavefront is anything callable as ``W(u, v) -> waves`` on normalized pupil
coordinates.  :class:`WavefrontMap` wraps sampled data (from a ray trace, an
interferogram, a Zernike sum) in that interface, so the pupil and imaging code
never needs to know where the aberration came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WavefrontMap", "zernike", "ZernikeWavefront"]


@dataclass
class WavefrontMap:
    """Wavefront sampled on a regular ``[-1, 1]^2`` grid, in waves."""

    values: np.ndarray
    mask: np.ndarray | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    def __call__(self, u, v):
        """Bilinear interpolation; zero outside the unit disc."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        n = self.n
        fu = (u + 1.0) * 0.5 * (n - 1)
        fv = (v + 1.0) * 0.5 * (n - 1)
        i0 = np.clip(np.floor(fu).astype(int), 0, n - 2)
        j0 = np.clip(np.floor(fv).astype(int), 0, n - 2)
        du, dv = fu - i0, fv - j0
        V = self.values
        out = (V[j0, i0] * (1 - du) * (1 - dv) + V[j0, i0 + 1] * du * (1 - dv) +
               V[j0 + 1, i0] * (1 - du) * dv + V[j0 + 1, i0 + 1] * du * dv)
        return np.where(u**2 + v**2 <= 1.0, out, 0.0)

    # ------------------------------------------------------------------ stats
    @property
    def _inside(self) -> np.ndarray:
        ax = np.linspace(-1, 1, self.n)
        U, V = np.meshgrid(ax, ax)
        return U**2 + V**2 <= 1.0

    @property
    def rms_waves(self) -> float:
        v = self.values[self._inside]
        return float(np.std(v))

    @property
    def pv_waves(self) -> float:
        v = self.values[self._inside]
        return float(np.ptp(v))

    def rms_nm(self, wavelength_nm: float) -> float:
        return self.rms_waves * float(wavelength_nm)

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_samples(cls, u, v, waves, n: int = 129,
                     method: str = "cubic") -> "WavefrontMap":
        """Grid scattered pupil samples (e.g. from a ray trace)."""
        from scipy.interpolate import griddata
        ax = np.linspace(-1, 1, int(n))
        U, V = np.meshgrid(ax, ax)
        vals = griddata(np.column_stack([np.asarray(u), np.asarray(v)]),
                        np.asarray(waves, dtype=float), (U, V),
                        method=method, fill_value=0.0)
        vals = np.nan_to_num(vals)
        return cls(vals, mask=(U**2 + V**2 <= 1.0))

    @classmethod
    def from_callable(cls, fn, n: int = 129) -> "WavefrontMap":
        ax = np.linspace(-1, 1, int(n))
        U, V = np.meshgrid(ax, ax)
        return cls(np.nan_to_num(np.asarray(fn(U, V), dtype=float)))

    @classmethod
    def flat(cls, n: int = 33) -> "WavefrontMap":
        return cls(np.zeros((int(n), int(n))))

    def to_dict(self) -> dict:
        return {"n": self.n, "values_waves": self.values.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "WavefrontMap":
        return cls(np.asarray(d["values_waves"], dtype=float))


# ------------------------------------------------------------------- Zernike
def zernike(j: int, u, v) -> np.ndarray:
    """Noll-indexed Zernike polynomial ``Z_j`` on the unit disc."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    r = np.hypot(u, v)
    th = np.arctan2(v, u)
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * (j - 1))) / 2))
    m_candidates = [m for m in range(0, n + 1) if (n - m) % 2 == 0]
    idx = j - (n * (n + 1)) // 2 - 1
    m = m_candidates[min(idx // 2 if n % 2 == 0 else (idx + 1) // 2, len(m_candidates) - 1)]
    R = np.zeros_like(r)
    for s in range((n - m) // 2 + 1):
        from math import factorial
        c = ((-1) ** s * factorial(n - s) /
             (factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s)))
        R = R + c * r ** (n - 2 * s)
    if m == 0:
        return np.sqrt(n + 1) * R
    even = (j % 2 == 0)
    ang = np.cos(m * th) if even else np.sin(m * th)
    return np.sqrt(2 * (n + 1)) * R * ang


@dataclass
class ZernikeWavefront:
    """Wavefront as a Noll-indexed Zernike sum, coefficients in waves."""

    coefficients: dict[int, float]

    def __call__(self, u, v):
        out = np.zeros_like(np.asarray(u, dtype=float))
        for j, c in self.coefficients.items():
            out = out + float(c) * zernike(int(j), u, v)
        r2 = np.asarray(u, dtype=float) ** 2 + np.asarray(v, dtype=float) ** 2
        return np.where(r2 <= 1.0, out, 0.0)
