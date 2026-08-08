# vectorwave

Vectorial wave optics for curved interfaces and projection optics.

`vectorwave` computes the **vector** electromagnetic field of focusing and
imaging systems: the field at a focus, the field transmitted by a curved
dielectric interface, and the aerial image of a mask — with the longitudinal
and cross-polarized components that a scalar model cannot represent.

## Why

At low numerical aperture a scalar wave is a fine description of a focus.  At
NA 1.2 in immersion it is not: the rays meet at 49°, the polarization is
projected as the pupil bends them, and roughly a seventh of the focal energy
ends up in the longitudinal component.  `vectorwave` keeps the vector field
throughout, and stays cheap for interfaces that are hundreds of wavelengths
across by working in the angular spectrum rather than on a volume mesh.

## Install

```bash
pip install -e .
pytest            # 33 physics/API tests
```

`numpy`, `scipy` and `finufft` are required (`finufft` powers the general
surface transform, the freeform path, and operator composition). `matplotlib`
is needed for the notebook; the adapters to `geometrical-raytracer` and
`vecdiff` are optional and imported lazily.

## The four abstractions

| | |
|---|---|
| `Surface` | a shape that knows its sag, slope and normal — `Sphere`, `Conic`, `EvenAsphere`, `Freeform` |
| `Field` | sampled vector field, with intensity, component fractions and polarization (Stokes, ellipticity, orientation) |
| `AngularSpectrum` | the propagation currency: every source reduces to one, and propagating or synthesising is a single transform |
| `Operator` / `System` | interfaces and free space as composable operators on the spectrum — `InterfaceOperator`, `FreeSpace`, chained by `System([...])` |
| `Pupil` / `ImagingSystem` | an exit pupil for focal fields, and its imaging counterpart for masks |

## Quick start

Focus a hyper-NA pupil and look at where the energy went:

```python
import numpy as np, vectorwave as vw

system = vw.load("duv")                       # US7557996, NA 1.2 immersion
pupil  = system.pupil(polarization="x")       # carries the traced wavefront
grid   = vw.Grid.from_spacing(0.25, 256)

x = np.arange(-1.5, 1.5, 1/64)
field = pupil.spectrum(grid).field_on(x, x, z=0.0)

print(field.component_fractions())     # {'x': 0.86, 'y': 0.004, 'z': 0.14}
print(field.fwhm("x"), pupil.airy_fwhm())
```

Refract a plane wave through a stigmatic interface (the spectral operator):

```python
kappa = vw.stigmatic_conic_constant(1.5, 1.0)          # -2.25
surface = vw.Conic(radius=-8.0, conic=kappa)            # R<0 converges toward +z
spec = vw.surface_spectrum(surface, grid, n1=1.5, n2=1.0, aperture=9.9)
z = spec.best_focus(np.linspace(8, 22, 73))             # 14.5 lambda (paraxial 16)
```

## Composition: systems as products of operators

An interface is an operator on the angular spectrum, so interfaces *compose* —
a system of many surfaces is their ordered product interleaved with free
propagation, exactly as in the theory. The same algebra describes a closed body
(one surface encountered repeatedly).

```python
pw    = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0, polarization="x")
front = vw.InterfaceOperator(vw.Conic(radius=+6.0, conic=vw.stigmatic_conic_constant(1.0, 1.5)),
                             n1=1.0, n2=1.5, aperture=5.5)
back  = vw.InterfaceOperator(vw.Plane(), n1=1.5, n2=1.0, aperture=5.5)

system = vw.System([front, vw.FreeSpace(8.0), back])    # a word in the operator algebra
out    = system(pw)                                     # incident spectrum -> focal spectrum
```

`InterfaceOperator` accepts *any* incident spectrum (it reads the local ray
direction from the field itself), so the output of one element feeds the next.
`FreeSpace` is the exact diagonal propagator.

## Any smooth surface

Surfaces of revolution use the fast azimuthal Bessel kernel. A genuinely
non-axisymmetric surface uses the same operator through a general surface
transform — only the cost changes:

```python
lens = vw.Freeform2D(sag_fn=lambda x, y: (x**2 + 1.3*y**2) / (2*R), radius=6.0)
out  = vw.InterfaceOperator(lens, n1=1.0, n2=1.5, aperture=6.0)(pw)   # NUFFT path
```

The general transform (`vw.surface_transform`, a type-3 NUFFT of the surface
currents) is also the fast path for large axisymmetric interfaces: it is
near-linear in the number of samples where the Bessel kernel grows with the
size parameter (≈10× faster by aperture 16λ), and both agree on the field.

Image a mask, vectorially:

```python
im   = system.imaging(pixel_nm=12.0, size=512, polarization="x")
mask = vw.Mask.from_image("circuit.png", pixel=12.0, size=512)
img  = im.aerial_image(mask, sigma=0.6, vector=True)
print(img.fractions())                 # longitudinal share of the aerial image
```

## Packaged systems

`vw.load(name)` ships two real objectives, each carrying its ray-traced
exit-pupil wavefront so the package needs no tracer at runtime:

| name | system | λ | NA | wavefront |
|---|---|---|---|---|
| `euv` | US 7,151,592 six-mirror EUV projector | 13.4 nm | 0.22 | 0.16 nm RMS |
| `duv` | US 7,557,996 hyper-NA immersion objective | 193.4 nm | 1.2 (n=1.60) | 0.40 nm RMS |

Both are diffraction-limited.  The EUV aspheric table as published is truncated
and is *not* — `tools/build_data.py` refits it (69.5 µm → 2.9 nm transverse
residual) before extracting the wavefront.

## Notebooks

Three self-contained demonstration notebooks under `notebooks/` run end to end on
the shipped package alone (no ray tracer needed) — regenerate them with
`python tools/build_demos.py`:

| notebook | what it shows |
|---|---|
| `01_vector_focusing.ipynb` | focal fields, the `NA²/4n²` longitudinal law, structured polarization, through-focus |
| `02_curved_interfaces.ipynb` | surfaces, the MEEP-validated stigmatic focus, the operator algebra (`InterfaceOperator`, `FreeSpace`, `System`), and a freeform |
| `03_projection_imaging.ipynb` | the two packaged objectives: PSFs, a vectorially imaged circuit layout, TE/TM/scalar contrast |

`lithography.ipynb` and `lithography_expanded_v2.ipynb` are the fuller studies;
their last sections also draw the traced hardware, which needs the optional
`geometrical-raytracer` and `vecdiff` packages.

## Sign convention

Surfaces are graphs `z = sag(rho)` with the vertex at the origin and a
**positive radius placing the centre of curvature at +z**.  A beam travelling
toward +z is therefore converged by a *negative* radius.

## Validation

The physics is pinned by tests rather than asserted:

* transversality `k·E = 0` to ~1e-16 for every spectrum;
* the longitudinal energy share follows `NA²/4n²` in the low-NA limit;
* focal spot width matches the Airy FWHM to 10%;
* the stigmatic hyperboloid focus reproduces an independent full-wave FDTD
  (MEEP) result — 14.5 λ where the paraxial prediction is 16 λ;
* scalar and vector imaging agree at low NA, and diverge at hyper-NA in the
  documented way.
