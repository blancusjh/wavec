"""Refracting / reflecting surface shapes.

Every surface exposes the same three things the physics needs — a sag, its
slope, and an outward unit normal — so the propagators never care whether they
were handed a sphere, a conic, a polynomial asphere or a measured profile.

Sign convention (the usual optical one): the surface is a graph ``z = sag(rho)``
with its vertex at the origin, and a **positive radius puts the centre of
curvature at +z**, so the surface curves toward +z and is concave as seen from
+z.  A beam travelling toward +z is therefore converged by a *negative* radius:
for the classic glass-to-air focusing cap use ``Conic(radius=-R, conic=...)``.

The outward normal always carries a positive radial component in the sense of
the sag gradient, so it is continuous through the vertex.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Surface",
    "Plane",
    "Sphere",
    "Conic",
    "EvenAsphere",
    "Freeform",
    "Freeform2D",
    "stigmatic_conic_constant",
]


class Surface(ABC):
    """Abstract rotationally-symmetric surface ``z = sag(rho)``."""

    #: surfaces of revolution admit the fast azimuthal-harmonic propagators
    rotationally_symmetric: bool = True

    @abstractmethod
    def sag(self, rho: np.ndarray) -> np.ndarray:
        """Axial displacement of the surface at transverse radius ``rho``."""

    @abstractmethod
    def dsag(self, rho: np.ndarray) -> np.ndarray:
        """Slope ``d(sag)/d(rho)``."""

    # ------------------------------------------------------------------ derived
    @property
    def max_radius(self) -> float:
        """Largest usable transverse radius (``inf`` when unbounded)."""
        return np.inf

    def normal(self, rho: np.ndarray, phi: np.ndarray | None = None) -> np.ndarray:
        """Outward unit normal.

        With ``phi`` given, returns a ``(3, ...)`` Cartesian normal; without it,
        the meridional ``(n_rho, n_z)`` pair.
        """
        rho = np.asarray(rho, dtype=float)
        s = self.dsag(rho)
        norm = np.sqrt(1.0 + s**2)
        n_rho, n_z = s / norm, 1.0 / norm
        if phi is None:
            return np.stack([n_rho, n_z])
        return np.stack([n_rho * np.cos(phi), n_rho * np.sin(phi),
                         np.broadcast_to(n_z, np.shape(phi))])

    def area_element(self, rho: np.ndarray) -> np.ndarray:
        """``dS / (drho dphi)`` = ``rho * sqrt(1 + sag'^2)``."""
        rho = np.asarray(rho, dtype=float)
        return rho * np.sqrt(1.0 + self.dsag(rho) ** 2)

    def profile(self, n: int = 256, r_max: float | None = None):
        """``(rho, sag)`` samples, handy for drawing the surface."""
        rmax = r_max if r_max is not None else (
            self.max_radius if np.isfinite(self.max_radius) else 1.0)
        rho = np.linspace(0.0, 0.999 * rmax, int(n))
        return rho, self.sag(rho)


# --------------------------------------------------------------------- shapes
@dataclass(frozen=True)
class Plane(Surface):
    """Flat surface."""

    def sag(self, rho):
        return np.zeros_like(np.asarray(rho, dtype=float))

    def dsag(self, rho):
        return np.zeros_like(np.asarray(rho, dtype=float))


@dataclass(frozen=True)
class Conic(Surface):
    """Conic of vertex radius ``radius`` and conic constant ``conic``.

    ``conic`` 0 is a sphere, -1 a paraboloid, < -1 a hyperboloid and
    -1 < k < 0 a prolate ellipsoid.
    """

    radius: float
    conic: float = 0.0

    @property
    def max_radius(self) -> float:
        one_plus_k = 1.0 + self.conic
        if one_plus_k <= 0:
            return np.inf                      # paraboloid / hyperboloid: unbounded
        return abs(self.radius) / np.sqrt(one_plus_k)

    def _root(self, rho):
        rho = np.asarray(rho, dtype=float)
        arg = 1.0 - (1.0 + self.conic) * (rho / self.radius) ** 2
        return np.sqrt(np.maximum(arg, 1e-12))

    def sag(self, rho):
        rho = np.asarray(rho, dtype=float)
        return rho**2 / (self.radius * (1.0 + self._root(rho)))

    def dsag(self, rho):
        rho = np.asarray(rho, dtype=float)
        return rho / (self.radius * self._root(rho))


@dataclass(frozen=True)
class Sphere(Conic):
    """Spherical surface (a conic with ``conic = 0``)."""

    conic: float = 0.0


@dataclass(frozen=True)
class EvenAsphere(Surface):
    """Conic base plus even polynomial terms.

    ``coefficients[i]`` multiplies ``rho**(2*i + 4)`` — the standard optical
    convention in which the first coefficient is the rho^4 term.
    """

    radius: float
    conic: float = 0.0
    coefficients: tuple[float, ...] = ()

    @property
    def _base(self) -> Conic:
        return Conic(self.radius, self.conic)

    @property
    def max_radius(self) -> float:
        return self._base.max_radius

    def sag(self, rho):
        rho = np.asarray(rho, dtype=float)
        z = self._base.sag(rho)
        for i, c in enumerate(self.coefficients):
            z = z + c * rho ** (2 * i + 4)
        return z

    def dsag(self, rho):
        rho = np.asarray(rho, dtype=float)
        d = self._base.dsag(rho)
        for i, c in enumerate(self.coefficients):
            d = d + c * (2 * i + 4) * rho ** (2 * i + 3)
        return d


@dataclass(frozen=True)
class Freeform(Surface):
    """Rotationally-symmetric surface from an arbitrary sag callable or samples.

    Pass either ``sag_fn`` (any smooth ``rho -> z``) or tabulated
    ``(rho_samples, sag_samples)``, which are splined.  Slopes are taken
    analytically when possible and by central differences otherwise.
    """

    sag_fn: object = None
    rho_samples: np.ndarray | None = None
    sag_samples: np.ndarray | None = None
    step: float = 1e-6
    _spline: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.sag_fn is None:
            if self.rho_samples is None or self.sag_samples is None:
                raise ValueError("provide sag_fn or (rho_samples, sag_samples)")
            from scipy.interpolate import CubicSpline
            object.__setattr__(self, "_spline",
                               CubicSpline(np.asarray(self.rho_samples, float),
                                           np.asarray(self.sag_samples, float)))

    @property
    def max_radius(self) -> float:
        if self.rho_samples is not None:
            return float(np.max(self.rho_samples))
        return np.inf

    def sag(self, rho):
        rho = np.asarray(rho, dtype=float)
        if self._spline is not None:
            return self._spline(rho)
        return np.asarray(self.sag_fn(rho), dtype=float)

    def dsag(self, rho):
        rho = np.asarray(rho, dtype=float)
        if self._spline is not None:
            return self._spline(rho, 1)
        h = self.step
        return (self.sag(rho + h) - self.sag(rho - h)) / (2 * h)


@dataclass(frozen=True)
class Freeform2D(Surface):
    """A genuinely non-axisymmetric surface ``z = sag(x, y)``.

    Pass any smooth callable ``sag_fn(x, y) -> z`` and an aperture ``radius``.
    Such a surface forgoes the azimuthal Bessel kernel of the surfaces of
    revolution; its operator is carried by the general NUFFT surface transform
    (:func:`vectorwave.surface_transform`).  The operator is identical --- only
    its cost changes.

    ``sag`` / ``dsag`` (the rotationally-symmetric interface) raise, because a
    freeform surface has no single-argument profile; use ``sag_xy`` and
    ``normal_xy`` instead, which the general propagator calls.
    """

    rotationally_symmetric: bool = False
    sag_fn: object = None
    radius: float = np.inf
    step: float = 1e-6

    def __post_init__(self):
        if self.sag_fn is None:
            raise ValueError("Freeform2D needs a sag_fn(x, y)")

    # the rotationally-symmetric interface does not apply
    def sag(self, rho):        # pragma: no cover - guard
        raise NotImplementedError("Freeform2D is not axisymmetric; use sag_xy")

    def dsag(self, rho):       # pragma: no cover - guard
        raise NotImplementedError("Freeform2D is not axisymmetric; use normal_xy")

    @property
    def max_radius(self) -> float:
        return float(self.radius)

    def sag_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Axial displacement at Cartesian ``(x, y)``."""
        return np.asarray(self.sag_fn(np.asarray(x, float), np.asarray(y, float)),
                          dtype=float)

    def normal_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Outward unit normal ``(3, ...)`` at Cartesian ``(x, y)`` (with a
        positive ``z`` component), by central differences of the sag."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        h = self.step
        zx = (self.sag_xy(x + h, y) - self.sag_xy(x - h, y)) / (2 * h)
        zy = (self.sag_xy(x, y + h) - self.sag_xy(x, y - h)) / (2 * h)
        n = np.stack([-zx, -zy, np.ones_like(zx)])
        return n / np.linalg.norm(n, axis=0, keepdims=True)


# ------------------------------------------------------------------ utilities
def stigmatic_conic_constant(n_incident: float, n_transmitted: float) -> float:
    """Conic constant of the Cartesian oval that is stigmatic for a plane wave.

    ``kappa = -(n_incident / n_transmitted)**2`` — the surface then focuses a
    collimated beam with no spherical aberration at any order.
    """
    return -(float(n_incident) / float(n_transmitted)) ** 2
