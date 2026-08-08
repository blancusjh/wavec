"""Regenerate the packaged systems from their ray-traced prescriptions.

Run from the repo root with ``geometrical-raytracer`` importable::

    python tools/build_data.py

EUV  : US 7,151,592 B2 six-mirror projector (aspheres refit by damped least
       squares — the published table is truncated and is *not* diffraction
       limited as printed).
DUV  : US 7,557,996 hyper-NA immersion objective, read from its prescription CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/claude/build/geometrical-raytracer")

from raytracer.design import OpticalSystem, SurfaceRow                # noqa: E402
from raytracer.propagation import (FieldPoint, PupilSampling,          # noqa: E402
                                   SequentialTracer, chief_ray_slope,
                                   solve_object_plane, trace_pupil)
from vectorwave.adapters import wavefront_from_pupil_trace            # noqa: E402
from vectorwave.systems import LithoSystem                            # noqa: E402

OUT = ROOT / "vectorwave" / "data"
OUT.mkdir(parents=True, exist_ok=True)
REFIT = Path("/home/claude/spectral/euv_refit_prescription.json")


# --------------------------------------------------------------------- EUV
def build_euv() -> LithoSystem:
    spec = json.loads(REFIT.read_text())
    thick = spec["thicknesses_mm"]
    mirrors = spec["mirrors"]
    rows = [SurfaceRow.mirror(radius=mirrors[0]["radius_mm"], conic=mirrors[0]["conic"],
                              coefficients=mirrors[0]["coefficients"], thickness=thick[0]),
            SurfaceRow.stop(thickness=thick[1])]
    for i in range(1, 6):
        rows.append(SurfaceRow.mirror(radius=mirrors[i]["radius_mm"],
                                      conic=mirrors[i]["conic"],
                                      coefficients=mirrors[i]["coefficients"],
                                      thickness=thick[i + 1]))
    system = OpticalSystem(rows, wavelength_um=spec["wavelength_mm"] * 1e3,
                           name="US7151592 six-mirror EUV")
    system.object_z = spec["object_z_mm"]
    tracer = SequentialTracer(system, clip_apertures=False)

    lam = spec["wavelength_mm"]
    nao, nai = spec["na_object"], spec["na_image"]
    y = 120.0
    th = chief_ray_slope(tracer, FieldPoint(y=y), stop_index=spec["stop_index"],
                         bracket=(-0.3, 0.05), scan=71)
    pup = trace_pupil(tracer, FieldPoint(y=y), na_object_sine=nao,
                      stop_index=spec["stop_index"], slope_model="sine",
                      chief_slope=th,
                      sampling=PupilSampling(kind="rings", radial=24, azimuth=96))
    wf = wavefront_from_pupil_trace(pup, na_image=nai, n_image=1.0,
                                    wavelength_mm=lam, n=129)
    return LithoSystem(
        name="US7151592B2 six-mirror EUV projection objective",
        wavelength=lam, na=nai, n_image=1.0, reduction=4.0,
        field_heights=(116.0, 120.0, 124.0), wavefront=wf,
        meta={"source": "US 7,151,592 B2 Table 2 (Embodiment 1), aspheres refit",
              "mirrors": 6, "track_mm": 1500.0, "na_object": nao,
              "field_type": "ring", "chief_field_mm": y,
              "note": "published aspheric table is truncated; coefficients were "
                      "refit by damped least squares to recover the design"},
    )


# --------------------------------------------------------------------- DUV
def build_duv() -> LithoSystem:
    csv = Path("/home/claude/build/geometrical-raytracer/data/optical_systems/"
               "lithography/US7557996_Fig3_Table3_prescription.csv")
    system = OpticalSystem.from_prescription(csv)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)
    na, red = 1.2, 4.0
    lam = system.wavelength_um * 1e-3
    pup = trace_pupil(tracer, FieldPoint(y=62.0), na_object_sine=na / red,
                      sampling=PupilSampling(kind="rings", radial=16, azimuth=96))
    wf = wavefront_from_pupil_trace(pup, na_image=na, n_image=system.n_image,
                                    wavelength_mm=lam, n=129)
    return LithoSystem(
        name="US7557996 hyper-NA DUV immersion objective",
        wavelength=lam, na=na, n_image=float(system.n_image), reduction=red,
        field_heights=(56.0, 62.0, 67.0), wavefront=wf,
        meta={"source": "US 7,557,996 Fig. 3 / Table 3", "surfaces": len(system),
              "mirrors": 2, "catadioptric": True, "immersion": True,
              "chief_field_mm": 62.0},
    )


if __name__ == "__main__":
    for name, builder in (("euv", build_euv), ("duv", build_duv)):
        s = builder()
        s.save(OUT / f"{name}.json")
        print(f"{name}: {s}")
        print(f"     WFE {s.wavefront_rms_nm:.3f} nm  (lambda/14 = "
              f"{s.wavelength_nm/14:.3f} nm)  diffraction-limited={s.diffraction_limited}")
        print(f"     Rayleigh half-pitch {s.rayleigh_half_pitch_nm:.1f} nm, "
              f"Airy FWHM {s.airy_fwhm_nm:.1f} nm, DoF {s.depth_of_focus_nm:.0f} nm")
