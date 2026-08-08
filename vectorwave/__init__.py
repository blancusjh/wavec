"""vectorwave — vectorial wave optics for curved interfaces and projection optics.

The package is built around four abstractions:

``Surface``   a shape (sphere, conic, asphere, freeform) that knows its sag,
              slope and normal;
``Field``     a sampled vector field with polarization diagnostics;
``AngularSpectrum``
              the propagation currency — every source reduces to one, and
              propagating or synthesising a field is then a single transform;
``Pupil`` / ``ImagingSystem``
              an exit pupil for focal fields, and its imaging counterpart.

Typical use::

    import vectorwave as vw

    sys_ = vw.load("duv")                       # a packaged objective
    pup = sys_.pupil(polarization="x")
    grid = vw.Grid.from_spacing(0.25, 256)
    field = pup.spectrum(grid).field_on(x, x, z=0)
    print(field.component_fractions())

    interface = vw.Conic(radius=20.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    spec = vw.surface_spectrum(interface, grid, n1=1.5, n2=1.0, aperture=12.0)
"""

from .fields import Field, Polarization
from .grids import Grid
from .imaging import AerialImage, ImagingSystem, Mask
from .interfaces import (critical_angle, fresnel, reflect_field,
                         refract_direction, transmit_field)
from .operators import (FreeSpace, InterfaceOperator, Operator, System,
                        plane_wave_spectrum, point_source_spectrum)
from .propagation import (propagate, spectrum_of, surface_spectrum,
                          surface_transform)
from .pupil import POLARIZATIONS, Pupil
from .spectrum import AngularSpectrum
from .surfaces import (Conic, EvenAsphere, Freeform, Freeform2D, Plane, Sphere,
                       Surface, stigmatic_conic_constant)
from .systems import LithoSystem, available, load
from .wavefront import WavefrontMap, ZernikeWavefront, zernike

__version__ = "0.2.0"

__all__ = [
    "Field", "Polarization", "Grid", "AngularSpectrum",
    "Surface", "Plane", "Sphere", "Conic", "EvenAsphere", "Freeform",
    "Freeform2D", "stigmatic_conic_constant",
    "Pupil", "POLARIZATIONS",
    "Operator", "FreeSpace", "InterfaceOperator", "System",
    "plane_wave_spectrum", "point_source_spectrum",
    "propagate", "spectrum_of", "surface_spectrum", "surface_transform",
    "fresnel", "transmit_field", "reflect_field", "refract_direction",
    "critical_angle",
    "Mask", "ImagingSystem", "AerialImage",
    "LithoSystem", "load", "available",
    "WavefrontMap", "ZernikeWavefront", "zernike",
    "__version__",
]
