"""Composable operators on the angular spectrum.

The paper's spine: in the intrinsic representation an interface maps the space
of angular-spectrum states to itself, so interfaces *compose*.  This module
makes that literal.

* :class:`FreeSpace` -- the trivial diagonal operator (a propagation phase).
* :class:`InterfaceOperator` -- a curved dielectric interface as an operator
  that takes an incident spectrum to an outgoing one, via the same local
  tangent-plane model as :func:`vectorwave.surface_spectrum`, but for a
  *general* incident field and (optionally) a *general* surface.
* :class:`System` -- an ordered product of operators; an objective is a word in
  this algebra, and a closed body is the same product with one surface repeated.

Each operator is callable and returns an :class:`~vectorwave.spectrum.AngularSpectrum`,
so ``System([...])(spectrum)`` reads exactly like the mathematics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .grids import Grid
from .interfaces import reflect_field, transmit_field
from .propagation import (_axisym_samples, _freeform_samples,
                          _rim_tir_apodization, _return_integral_nufft,
                          _return_integral_polar, incident_on_surface,
                          raised_cosine, spectrum_of)
from .spectrum import AngularSpectrum
from .surfaces import Surface

__all__ = ["Operator", "FreeSpace", "InterfaceOperator", "System",
           "plane_wave_spectrum", "point_source_spectrum"]


class Operator(ABC):
    """A map from an angular spectrum to an angular spectrum."""

    @abstractmethod
    def apply(self, spec: AngularSpectrum) -> AngularSpectrum:
        ...

    def __call__(self, spec: AngularSpectrum) -> AngularSpectrum:
        return self.apply(spec)


# --------------------------------------------------------------------------
class FreeSpace(Operator):
    """Propagation through a homogeneous gap of length ``distance``.

    Diagonal on the angular spectrum: every direction is multiplied by its own
    phase and none is mixed.  The index is inherited from the incoming
    spectrum (free space does not change the medium)."""

    def __init__(self, distance: float):
        self.distance = float(distance)

    def apply(self, spec: AngularSpectrum) -> AngularSpectrum:
        mask = spec.grid.propagating(spec.k)
        phase = mask * np.exp(1j * spec.sigma * spec.kz * self.distance)
        return AngularSpectrum(spec.A * phase[None], spec.grid, spec.wavelength,
                               spec.n, spec.sigma)

    def __repr__(self):  # pragma: no cover - cosmetic
        return f"FreeSpace({self.distance:g})"


# --------------------------------------------------------------------------
class InterfaceOperator(Operator):
    """A curved dielectric interface, as an operator on the angular spectrum.

    ``apply`` samples the incident spectrum on the surface, refracts
    (``mode='t'``) or reflects (``mode='r'``) it with the full vector Fresnel
    operator, and returns the outgoing spectrum by the surface transform.  For
    a surface of revolution the azimuthal Bessel kernel is used
    (``method='polar'``); a :class:`~vectorwave.surfaces.Freeform2D` surface, or
    ``method='nufft'``, uses the general NUFFT transform.

    Parameters mirror :func:`vectorwave.surface_spectrum`.  Because it consumes
    and produces a spectrum, ``InterfaceOperator`` composes with others.
    """

    def __init__(self, surface: Surface, *, n1: float, n2: float,
                 mode: str = "t", aperture: float | None = None,
                 wavelength: float = 1.0, m_max: int = 6,
                 n_rho: int = 600, n_phi: int = 64, n_kr: int = 512,
                 n_free: int = 220, edge_softness: float = 0.25,
                 tir_margin: float = 0.04, method: str = "auto"):
        self.surface = surface
        self.n1, self.n2, self.mode = float(n1), float(n2), mode
        self.wavelength = float(wavelength)
        self.m_max, self.n_rho, self.n_phi = m_max, n_rho, n_phi
        self.n_kr, self.n_free = n_kr, n_free
        self.edge_softness, self.tir_margin = edge_softness, tir_margin
        self.method = method
        if aperture is None:
            aperture = (surface.max_radius if np.isfinite(surface.max_radius)
                        else 10.0 * self.wavelength)
        self.aperture = float(aperture)

    @property
    def out_index(self) -> float:
        return self.n2 if self.mode == "t" else self.n1

    def _local(self, points, nhat, khat, E_in):
        if self.mode == "t":
            E_out, _, coeffs = transmit_field(E_in, khat, nhat, self.n1, self.n2)
        else:
            E_out, _, coeffs = reflect_field(E_in, khat, nhat, self.n1, self.n2)
        return E_out, coeffs

    def apply(self, spec: AngularSpectrum) -> AngularSpectrum:
        k_out = 2 * np.pi * self.out_index / self.wavelength
        axisym = getattr(self.surface, "rotationally_symmetric", True)
        method = self.method
        if method == "auto":
            method = "polar" if axisym else "nufft"
        if not axisym and method == "polar":
            raise ValueError("polar kernel needs an axisymmetric surface")

        if axisym:
            smp = _axisym_samples(self.surface, self.aperture, self.n_rho, self.n_phi)
            pts, nhat = smp["points"], smp["nhat"]
            E_in, khat = incident_on_surface(spec, pts[0], pts[1], pts[2])
            E_out, coeffs = self._local(pts, nhat, khat, E_in)
            rho, phi = smp["rho"], smp["phi"]
            vis = _rim_tir_apodization(rho, len(phi), self.aperture,
                                       self.edge_softness, coeffs, self.mode,
                                       self.n1, self.n2, self.tir_margin)
            visg = np.broadcast_to(vis[:, None], smp["RHO"].shape).ravel()
            E_out = E_out * visg[None, :]
            if method == "polar":
                datum = E_out.reshape(3, len(rho), len(phi))
                return _return_integral_polar(datum, rho, smp["sag"],
                                              self.surface.dsag(rho), k_out,
                                              spec.grid, self.m_max, self.n_kr,
                                              spec.sigma, self.wavelength,
                                              self.out_index)
            drho, dphi = rho[1] - rho[0], 2 * np.pi / len(phi)
            area = (rho * np.sqrt(1.0 + self.surface.dsag(rho) ** 2) * drho * dphi)
            dS = np.broadcast_to(area[:, None], smp["RHO"].shape).ravel()
            return _return_integral_nufft(pts, E_out, dS, k_out, spec.grid,
                                          spec.sigma, self.wavelength,
                                          self.out_index)

        # ---- general freeform surface -----------------------------------
        smp = _freeform_samples(self.surface, self.aperture, self.n_free)
        pts, nhat, dS = smp["points"], smp["nhat"], smp["dS"]
        E_in, khat = incident_on_surface(spec, pts[0], pts[1], pts[2])
        E_out, coeffs = self._local(pts, nhat, khat, E_in)
        r = np.hypot(pts[0], pts[1])
        vis = raised_cosine(r, self.aperture * (1.0 - self.edge_softness),
                            self.aperture)
        E_out = E_out * vis[None, :]
        return _return_integral_nufft(pts, E_out, dS * vis, k_out, spec.grid,
                                      spec.sigma, self.wavelength, self.out_index)

    def __repr__(self):  # pragma: no cover - cosmetic
        return (f"InterfaceOperator({type(self.surface).__name__}, "
                f"n1={self.n1:g}, n2={self.n2:g}, mode={self.mode!r})")


# --------------------------------------------------------------------------
class System(Operator):
    """An ordered product of operators: ``System([A, B, C])`` applies A, then B,
    then C.  A multi-element objective and a closed body are both words in this
    algebra."""

    def __init__(self, operators):
        self.operators = list(operators)

    def apply(self, spec: AngularSpectrum) -> AngularSpectrum:
        for op in self.operators:
            spec = op.apply(spec)
        return spec

    def __mul__(self, other: "Operator") -> "System":
        """``self * other`` = apply ``other`` first, then ``self`` (operator order)."""
        right = other.operators if isinstance(other, System) else [other]
        return System(list(right) + self.operators)

    def __len__(self):
        return len(self.operators)

    def __repr__(self):  # pragma: no cover - cosmetic
        return "System([" + ", ".join(repr(o) for o in self.operators) + "])"


# --------------------------------------------------------------------------
#  Source spectra to drive a system
# --------------------------------------------------------------------------
def plane_wave_spectrum(grid: Grid, *, wavelength: float = 1.0, n: float = 1.0,
                        polarization: str = "x", amplitude: float = 1.0,
                        sigma: int = +1) -> AngularSpectrum:
    """A unit on-axis plane wave as an angular spectrum (a single k=0 mode)."""
    from .pupil import POLARIZATIONS
    pol = polarization if callable(polarization) else POLARIZATIONS[polarization]
    ex, ey = pol(np.zeros(()), np.zeros(()))
    area = (grid.x.size * grid.dx) * (grid.y.size * grid.dy)
    A = np.zeros((3, *grid.shape), dtype=complex)
    A[0, 0, 0] = amplitude * complex(ex) * area
    A[1, 0, 0] = amplitude * complex(ey) * area
    return AngularSpectrum(A, grid, wavelength, n, sigma).impose_transversality()


def point_source_spectrum(grid: Grid, *, distance: float, wavelength: float = 1.0,
                          n: float = 1.0, polarization: str = "x",
                          sigma: int = +1) -> AngularSpectrum:
    """A diverging spherical wave from a point a distance ``distance`` before the
    plane ``z = 0``, reduced to its angular spectrum."""
    from .fields import Field
    from .pupil import POLARIZATIONS
    pol = polarization if callable(polarization) else POLARIZATIONS[polarization]
    X, Y = grid.XY
    R = np.hypot(np.hypot(X, Y), distance)
    k = 2 * np.pi * n / wavelength
    env = np.exp(1j * k * R) * (distance / R)
    ex, ey = pol(grid.R / max(grid.R.max(), 1e-12), grid.PHI)
    field = Field(np.asarray(ex) * env, np.asarray(ey) * env, grid, wavelength, n)
    return spectrum_of(field, sigma).impose_transversality()
