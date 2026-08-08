"""Optional bridges to neighbouring tools.

Nothing in the core imports these; they are only pulled in when you call them,
so ``vectorwave`` installs and tests with no extra dependencies.

* :func:`system_from_raytracer` turns a traced objective into a
  :class:`~vectorwave.systems.LithoSystem` (prescription -> exit-pupil wavefront).
* :func:`to_vecdiff_field` hands a :class:`~vectorwave.fields.Field` to
  ``vecdiff`` for its polarization-ellipse plots.
"""

from __future__ import annotations

import numpy as np

from .fields import Field
from .systems import LithoSystem
from .wavefront import WavefrontMap

__all__ = ["system_from_raytracer", "wavefront_from_pupil_trace",
           "to_vecdiff_field", "plot_polarization"]


def wavefront_from_pupil_trace(pupil_trace, *, na_image: float, n_image: float,
                               wavelength_mm: float, max_order: int = 8,
                               n: int = 129, refocus: bool = True) -> WavefrontMap:
    """Exit-pupil wavefront from a ``raytracer`` pupil trace, as a map in waves.

    Uses the transverse-ray (``fit_transverse``) reconstruction, which is the
    numerically robust path, and removes tilt (and defocus when ``refocus``).

    ``ZernikeExpansion.wavefront`` returns optical path in **mm**; it is
    converted to waves here, which is what the pupil and imaging code expects.
    """
    from raytracer.analysis import fit_transverse           # noqa: PLC0415
    exp_ = fit_transverse(pupil_trace, na_image=na_image, n_image=n_image,
                          wavelength_mm=wavelength_mm, max_order=max_order)
    ax = np.linspace(-1.0, 1.0, int(n))
    U, V = np.meshgrid(ax, ax)
    inside = U**2 + V**2 <= 1.0
    W = np.zeros_like(U)
    W[inside] = np.asarray(exp_.wavefront(U[inside], V[inside],
                                          remove_tilt=True, refocus=refocus), dtype=float)
    W = np.nan_to_num(W) / float(wavelength_mm)            # mm -> waves
    return WavefrontMap(W, mask=inside)


def system_from_raytracer(tracer, *, name: str, na_image: float,
                          na_object_sine: float, n_image: float = 1.0,
                          reduction: float = 4.0, field_y: float = 0.0,
                          stop_index: int | None = None,
                          chief_bracket: tuple[float, float] | None = None,
                          slope_model: str = "tangent",
                          radial: int = 16, azimuth: int = 96,
                          wavefront_samples: int = 129,
                          meta: dict | None = None) -> LithoSystem:
    """Build a :class:`LithoSystem` by tracing a ``raytracer`` objective.

    ``chief_bracket`` is needed for ring-field systems whose chief ray sits in a
    narrow slice of slope space (EUV-style objectives).
    """
    from raytracer.propagation import (FieldPoint, PupilSampling,   # noqa: PLC0415
                                       chief_ray_slope, trace_pupil)
    lam_mm = tracer.system.wavelength_um * 1e-3
    kwargs = {}
    if chief_bracket is not None:
        kwargs["chief_slope"] = chief_ray_slope(
            tracer, FieldPoint(y=field_y), stop_index=stop_index,
            bracket=chief_bracket, scan=71)
    pupil_trace = trace_pupil(tracer, FieldPoint(y=field_y),
                              na_object_sine=na_object_sine,
                              stop_index=stop_index, slope_model=slope_model,
                              sampling=PupilSampling(kind="rings", radial=radial,
                                                     azimuth=azimuth),
                              **kwargs)
    wf = wavefront_from_pupil_trace(pupil_trace, na_image=na_image,
                                    n_image=n_image, wavelength_mm=lam_mm,
                                    n=wavefront_samples)
    return LithoSystem(name=name, wavelength=lam_mm, na=na_image,
                       n_image=n_image, reduction=reduction,
                       field_heights=(field_y,), wavefront=wf, meta=meta or {})


# ------------------------------------------------------------------- vecdiff
def to_vecdiff_field(field: Field):
    """Wrap a :class:`Field` as a ``vecdiff`` field (same samples, no copy of physics)."""
    return field.to_vecdiff()


def plot_polarization(field: Field, ax=None, **kwargs):
    """Draw polarization ellipses over the intensity using ``vecdiff``.

    Sensible defaults for a focal spot; every keyword is forwarded, so
    ``stride``, ``scale``, ``sampling`` and friends work as documented there.
    """
    from vecdiff.polarization_visualization import plot_field_polarization  # noqa: PLC0415
    kwargs.setdefault("background", "intensity")
    kwargs.setdefault("intensity_gamma", 0.45)
    kwargs.setdefault("glyph", "ellipse")
    return plot_field_polarization(field.to_vecdiff(), ax=ax, **kwargs)
