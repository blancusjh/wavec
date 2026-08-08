"""Packaged optical systems.

A :class:`LithoSystem` is everything the wave code needs about a projection
objective — wavelength, image-side NA and index, reduction, and the exit-pupil
wavefront — without carrying the ray trace around.  The shipped systems store a
sampled wavefront so the package works standalone; the raytracer adapter can
regenerate one from a prescription.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .imaging import ImagingSystem
from .pupil import Pupil
from .wavefront import WavefrontMap

__all__ = ["LithoSystem", "load", "available", "DATA_DIR"]

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class LithoSystem:
    """A projection objective, reduced to what wave optics needs."""

    name: str
    wavelength: float                    # vacuum, in the package's length unit (mm here)
    na: float                            # image-side numerical aperture
    n_image: float = 1.0
    reduction: float = 4.0
    field_heights: tuple[float, ...] = ()
    wavefront: WavefrontMap | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ scales
    @property
    def wavelength_nm(self) -> float:
        return self.wavelength * 1e6      # mm -> nm

    @property
    def rayleigh_half_pitch_nm(self) -> float:
        return 0.5 * self.wavelength_nm / self.na

    @property
    def airy_fwhm_nm(self) -> float:
        return 0.5144 * self.wavelength_nm / self.na

    @property
    def depth_of_focus_nm(self) -> float:
        return self.wavelength_nm * self.n_image / self.na**2

    @property
    def wavefront_rms_nm(self) -> float:
        return 0.0 if self.wavefront is None else self.wavefront.rms_nm(self.wavelength_nm)

    @property
    def diffraction_limited(self) -> bool:
        """Marechal criterion: RMS wavefront error below lambda/14."""
        return self.wavefront_rms_nm <= self.wavelength_nm / 14.0

    # --------------------------------------------------------------- factories
    def pupil(self, polarization="x", *, ideal: bool = False,
              wavelength_units: float | None = None) -> Pupil:
        """A :class:`~vectorwave.pupil.Pupil` for focal-field calculations.

        Lengths are expressed in wavelengths (``wavelength = 1``) unless
        ``wavelength_units`` overrides it, which keeps focal-plane numbers in
        convenient units.
        """
        lam = 1.0 if wavelength_units is None else wavelength_units
        return Pupil(na=self.na, n=self.n_image, wavelength=lam,
                     wavefront=None if ideal else self.wavefront,
                     polarization=polarization)

    def imaging(self, *, pixel_nm: float = 12.0, size: int = 512,
                polarization="x", ideal: bool = False) -> ImagingSystem:
        """An :class:`~vectorwave.imaging.ImagingSystem` at wafer scale (nm)."""
        return ImagingSystem(na=self.na, wavelength=self.wavelength_nm,
                             n=self.n_image, pixel=float(pixel_nm), size=int(size),
                             wavefront=None if ideal else self.wavefront,
                             polarization=polarization)

    # ------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        d = {"name": self.name, "wavelength_mm": self.wavelength, "na_image": self.na,
             "n_image": self.n_image, "reduction": self.reduction,
             "field_heights_mm": list(self.field_heights), "meta": self.meta}
        if self.wavefront is not None:
            d["wavefront"] = self.wavefront.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LithoSystem":
        wf = WavefrontMap.from_dict(d["wavefront"]) if "wavefront" in d else None
        return cls(name=d["name"], wavelength=d["wavelength_mm"], na=d["na_image"],
                   n_image=d.get("n_image", 1.0), reduction=d.get("reduction", 4.0),
                   field_heights=tuple(d.get("field_heights_mm", ())),
                   wavefront=wf, meta=d.get("meta", {}))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"LithoSystem({self.name!r}, lambda={self.wavelength_nm:.1f} nm, "
                f"NA={self.na:g}, n={self.n_image:g}, "
                f"WFE={self.wavefront_rms_nm:.2f} nm)")


# --------------------------------------------------------------------- registry
def available() -> list[str]:
    """Names of the packaged systems."""
    return sorted(p.stem for p in DATA_DIR.glob("*.json"))


def load(name: str) -> LithoSystem:
    """Load a packaged system, e.g. ``load("euv")`` or ``load("duv")``."""
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"unknown system {name!r}; available: {available()}")
    return LithoSystem.from_dict(json.loads(path.read_text()))
