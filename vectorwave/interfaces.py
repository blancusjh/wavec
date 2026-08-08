"""Fresnel physics at a dielectric boundary, done vectorially.

Everything here is point-wise and local: given an incident direction, a surface
normal and the two indices, it returns the refracted/reflected direction and the
field amplitudes in the local s/p frame.  The propagators call it once per
surface sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FresnelCoefficients", "fresnel", "refract_direction",
           "reflect_direction", "transmit_field", "reflect_field",
           "critical_angle"]


@dataclass
class FresnelCoefficients:
    """Amplitude coefficients and the geometry that produced them."""

    ts: np.ndarray
    tp: np.ndarray
    rs: np.ndarray
    rp: np.ndarray
    cos_i: np.ndarray
    cos_t: np.ndarray
    tir: np.ndarray

    @property
    def transmittance(self) -> np.ndarray:
        """Unpolarized intensity transmittance (for diagnostics)."""
        return 0.5 * (np.abs(self.ts) ** 2 + np.abs(self.tp) ** 2)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=0, keepdims=True)


def _oriented(khat: np.ndarray, nhat: np.ndarray):
    """Flip the normal so it opposes the incident direction; return cos_i > 0."""
    c = np.sum(khat * nhat, axis=0)
    sgn = np.where(c >= 0, 1.0, -1.0)
    return nhat * sgn, np.abs(c)


def critical_angle(n1: float, n2: float) -> float:
    """Critical angle in radians (``nan`` when there is none)."""
    if n2 >= n1:
        return float("nan")
    return float(np.arcsin(n2 / n1))


def fresnel(khat: np.ndarray, nhat: np.ndarray, n1: float, n2: float) -> FresnelCoefficients:
    """Fresnel amplitude coefficients for each incident ray."""
    _, cos_i = _oriented(khat, nhat)
    mu = n1 / n2
    sin_t2 = (mu**2) * (1.0 - cos_i**2)
    tir = sin_t2 > 1.0
    cos_t = np.sqrt(np.clip(1.0 - sin_t2, 0.0, None))
    denom_s = n1 * cos_i + n2 * cos_t
    denom_p = n2 * cos_i + n1 * cos_t
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(tir, 0.0, 2 * n1 * cos_i / denom_s)
        tp = np.where(tir, 0.0, 2 * n1 * cos_i / denom_p)
        rs = np.where(tir, 1.0, (n1 * cos_i - n2 * cos_t) / denom_s)
        rp = np.where(tir, 1.0, (n2 * cos_i - n1 * cos_t) / denom_p)
    return FresnelCoefficients(np.nan_to_num(ts), np.nan_to_num(tp),
                               np.nan_to_num(rs), np.nan_to_num(rp),
                               cos_i, cos_t, tir)


def refract_direction(khat: np.ndarray, nhat: np.ndarray, n1: float, n2: float) -> np.ndarray:
    """Snell's law in vector form; TIR rays are returned as zeros."""
    ns, cos_i = _oriented(khat, nhat)
    mu = n1 / n2
    sin_t2 = (mu**2) * (1.0 - cos_i**2)
    cos_t = np.sqrt(np.clip(1.0 - sin_t2, 0.0, None))
    kt = mu * khat + (mu * cos_i - cos_t)[None] * (-ns)
    kt = np.where(sin_t2[None] > 1.0, 0.0, kt)
    norm = np.linalg.norm(kt, axis=0, keepdims=True)
    return np.divide(kt, norm, out=np.zeros_like(kt), where=norm > 0)


def reflect_direction(khat: np.ndarray, nhat: np.ndarray) -> np.ndarray:
    """Specular reflection direction."""
    c = np.sum(khat * nhat, axis=0)
    return khat - 2.0 * c[None] * nhat


def _sp_frame(khat: np.ndarray, nhat: np.ndarray):
    """Local ``(s_hat, p_hat_in)`` frame; ``s`` is perpendicular to the plane of
    incidence, ``p`` completes the right-handed triad with the incident ray."""
    s = np.cross(khat, nhat, axis=0)
    mag = np.linalg.norm(s, axis=0, keepdims=True)
    # at normal incidence the plane of incidence is degenerate: pick any s
    fallback = np.zeros_like(s)
    fallback[0] = 1.0
    s = np.where(mag > 1e-12, np.divide(s, np.where(mag > 0, mag, 1.0)), fallback)
    p_in = np.cross(khat, s, axis=0)
    return s, p_in


def transmit_field(E: np.ndarray, khat: np.ndarray, nhat: np.ndarray,
                   n1: float, n2: float):
    """Refract a vector field through the boundary.

    Returns ``(E_t, khat_t, coefficients)``.  The transmitted field is built in
    the local s/p frame so it stays transverse to its own new direction.
    """
    coeffs = fresnel(khat, nhat, n1, n2)
    kt = refract_direction(khat, nhat, n1, n2)
    s, p_in = _sp_frame(khat, nhat)
    p_out = np.cross(kt, s, axis=0)
    Es = np.sum(E * s, axis=0)
    Ep = np.sum(E * p_in, axis=0)
    Et = coeffs.ts * Es * s + coeffs.tp * Ep * p_out
    Et = np.where(coeffs.tir[None], 0.0, Et)
    return Et, kt, coeffs


def reflect_field(E: np.ndarray, khat: np.ndarray, nhat: np.ndarray,
                  n1: float, n2: float):
    """Reflect a vector field at the boundary. Returns ``(E_r, khat_r, coeffs)``."""
    coeffs = fresnel(khat, nhat, n1, n2)
    kr = reflect_direction(khat, nhat)
    s, p_in = _sp_frame(khat, nhat)
    p_out = np.cross(kr, s, axis=0)
    Es = np.sum(E * s, axis=0)
    Ep = np.sum(E * p_in, axis=0)
    Er = coeffs.rs * Es * s + coeffs.rp * Ep * p_out
    return Er, kr, coeffs
