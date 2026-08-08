"""Generate notebooks/lithography.ipynb."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C, M = [], []
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# Projection lithography with `vectorwave`

Two real objectives, taken from their patents and traced, then propagated as
**vector** fields:

| | system | λ | NA | reduction |
|---|---|---|---|---|
| **EUV** | US 7,151,592 six-mirror projector | 13.4 nm | 0.22 | 4× |
| **DUV** | US 7,557,996 hyper-NA immersion objective | 193.4 nm | 1.2 (n = 1.60) | 4× |

We compute point-spread functions, look at what the vector nature of light does
to them, and then image an actual circuit layout through both.

The contrast between the two systems is the point of the notebook. EUV buys
resolution with a short wavelength at modest aperture, so its focus is
essentially scalar. DUV buys it with an enormous aperture in immersion, so its
focus is strongly vectorial — a seventh of the energy is longitudinal, and the
polarization of the illumination changes what prints.
""")

code(r"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import vectorwave as vw

plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.titlesize": 10, "figure.facecolor": "white"})
print("vectorwave", vw.__version__, "| packaged systems:", vw.available())
""")

md(r"""
## 1 · The systems

Each packaged system carries its ray-traced exit-pupil wavefront, so no ray
tracer is needed at run time. Both are diffraction-limited (Marechal: RMS
wavefront error below λ/14).
""")

code(r"""
euv, duv = vw.load("euv"), vw.load("duv")

rows = []
for s in (euv, duv):
    rows.append([s.name.split()[0], f"{s.wavelength_nm:.1f} nm", f"{s.na:g}",
                 f"{s.n_image:.3f}", f"{s.wavefront_rms_nm:.3f} nm",
                 f"{s.wavelength_nm/14:.3f} nm", "yes" if s.diffraction_limited else "NO",
                 f"{s.rayleigh_half_pitch_nm:.1f} nm", f"{s.airy_fwhm_nm:.1f} nm",
                 f"{s.depth_of_focus_nm:.0f} nm"])

hdr = ["system", "lambda", "NA", "n", "WFE rms", "lambda/14", "diff.ltd",
       "Rayleigh hp", "Airy FWHM", "DoF"]
w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(hdr)]
print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
print("-+-".join("-" * x for x in w))
for r in rows:
    print(" | ".join(c.ljust(w[i]) for i, c in enumerate(r)))
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a, s in zip(ax, (euv, duv)):
    W = s.wavefront.values * 1000.0                     # milliwaves
    m = np.abs(W).max()
    im = a.imshow(np.where(s.wavefront._inside, W, np.nan),
                  extent=[-1, 1, -1, 1], origin="lower", cmap="RdBu_r",
                  vmin=-m, vmax=m)
    a.set_title(f"{s.name.split()[0]} exit-pupil wavefront\n"
                f"{s.wavefront.rms_waves*1000:.1f} m$\\lambda$ rms, "
                f"{s.wavefront.pv_waves*1000:.0f} m$\\lambda$ PV")
    a.set_xlabel("u"); a.set_ylabel("v")
    plt.colorbar(im, ax=a, shrink=0.85, label="m$\\lambda$")
fig.tight_layout()
""")

md(r"""
## 2 · Point-spread functions

`Pupil.spectrum(grid)` applies the aplanatic (Richards–Wolf) polarization
projection and returns an `AngularSpectrum`; `field_on` then synthesises the
field on any grid we like, exactly — the focal sampling is decoupled from the
grid carrying the spectrum.

Lengths here are in wavelengths, so a spot width of 1 means 1 λ.
""")

code(r"""
def psf(system, half=2.0, step=1/48, polarization="x", ideal=False, z=0.0, n_grid=256):
    # focal field of a system, synthesised on a fine zoomed window
    pupil = system.pupil(polarization=polarization, ideal=ideal)
    grid = vw.Grid.from_spacing(0.25, n_grid)
    x = np.arange(-half, half + 1e-9, step)
    return pupil, pupil.spectrum(grid).field_on(x, x, z), x

pupil_euv, f_euv, x_euv = psf(euv)
pupil_duv, f_duv, x_duv = psf(duv)

for tag, s, p, f in (("EUV", euv, pupil_euv, f_euv), ("DUV", duv, pupil_duv, f_duv)):
    fr = f.component_fractions()
    print(f"{tag}: FWHM = {f.fwhm('x'):.3f} lambda (Airy {p.airy_fwhm():.3f})"
          f"  ->  {f.fwhm('x')*s.wavelength_nm:5.1f} nm  [Airy {s.airy_fwhm_nm:.1f} nm]")
    print(f"      energy  |Ex|^2={fr['x']:.4f}  |Ey|^2={fr['y']:.5f}  |Ez|^2={fr['z']:.4f}")
""")

md(r"""
The EUV focus is scalar for all practical purposes: at NA 0.22 the longitudinal
share is a fraction of a percent. The DUV focus is not — about 14 % of its
energy is in $E_z$, and that component carries no image information, only
background. This is the whole reason hyper-NA lithography controls polarization.
""")

code(r"""
fig, ax = plt.subplots(2, 4, figsize=(13, 6.2))
for row, (tag, s, f, x) in enumerate((("EUV", euv, f_euv, x_euv),
                                      ("DUV", duv, f_duv, x_duv))):
    ext = [x[0], x[-1], x[0], x[-1]]
    comps = [("total $|E|^2$", f.intensity),
             ("$|E_x|^2$", np.abs(f.Ex)**2),
             ("$|E_y|^2$  (cross)", np.abs(f.Ey)**2),
             ("$|E_z|^2$  (longitudinal)", np.abs(f.Ez)**2)]
    vmax = f.intensity.max()
    for col, (title, I) in enumerate(comps):
        a = ax[row, col]
        a.imshow(I, extent=ext, origin="lower", cmap="inferno",
                 norm=PowerNorm(0.5, 0, vmax if col == 0 else I.max()))
        share = I.sum() / f.intensity.sum()
        a.set_title(f"{tag} · {title}" + ("" if col == 0 else f"  ({share*100:.2f} %)"))
        a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5)
        a.set_xlabel("x / $\\lambda$")
        if col == 0:
            a.set_ylabel("y / $\\lambda$")
fig.tight_layout()
""")

code(r"""
# radial profiles against the ideal Airy pattern
from scipy.special import j1

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
for a, (tag, s, p, f, x) in zip(ax, (("EUV", euv, pupil_euv, f_euv, x_euv),
                                     ("DUV", duv, pupil_duv, f_duv, x_duv))):
    I = f.intensity; I = I / I.max()
    cut = I[I.shape[0] // 2]
    r = np.linspace(1e-3, 2.5, 400)
    v = 2 * np.pi * s.na * r
    airy = (2 * j1(v) / v) ** 2
    a.plot(x, cut, lw=2, label=f"{tag} vector PSF")
    a.plot(r, airy, "k--", lw=1.1, label="ideal Airy")
    a.set_xlim(0, 2.5); a.set_xlabel("r / $\\lambda$"); a.set_ylabel("$|E|^2$")
    a.set_title(f"{tag}: FWHM {f.fwhm('x')*s.wavelength_nm:.1f} nm "
                f"(Airy {s.airy_fwhm_nm:.1f} nm)")
    a.legend(fontsize=8)
fig.tight_layout()
""")

md(r"""
### Polarization of the focal field

The transverse field is no longer uniformly polarized once the pupil bends the
rays: the ellipse orientation rotates away from the input direction wherever the
projection mixes the $s$ and $p$ components.
""")

code(r"""
pol = f_duv.polarization()
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
ext = [x_duv[0], x_duv[-1], x_duv[0], x_duv[-1]]
bright = f_duv.intensity > 0.02 * f_duv.intensity.max()

ax[0].imshow(f_duv.intensity, extent=ext, origin="lower", cmap="inferno",
             norm=PowerNorm(0.5))
ax[0].set_title("DUV focal $|E|^2$")
im = ax[1].imshow(np.where(bright, np.degrees(pol.orientation), np.nan),
                  extent=ext, origin="lower", cmap="twilight", vmin=-90, vmax=90)
ax[1].set_title("major-axis orientation"); plt.colorbar(im, ax=ax[1], shrink=0.85, label="deg")
im = ax[2].imshow(np.where(bright, np.degrees(pol.ellipticity), np.nan),
                  extent=ext, origin="lower", cmap="RdBu_r", vmin=-45, vmax=45)
ax[2].set_title("ellipticity"); plt.colorbar(im, ax=ax[2], shrink=0.85, label="deg")
for a in ax:
    a.set_xlim(-1.2, 1.2); a.set_ylim(-1.2, 1.2)
    a.set_xlabel("x / $\\lambda$")
fig.tight_layout()
""")

md(r"""
### Through focus

The same spectrum evaluated over a range of $z$ gives the longitudinal
structure. The DUV depth of focus is short — the price of the aperture that
buys the resolution.
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, (tag, s) in zip(ax, (("EUV", euv), ("DUV", duv))):
    grid = vw.Grid.from_spacing(0.25, 256)
    spec = s.pupil().spectrum(grid)
    dof = s.depth_of_focus_nm / s.wavelength_nm
    xs = np.arange(-2.0, 2.0, 1/32)
    zs = np.linspace(-1.6 * dof, 1.6 * dof, 121)
    comp, _, _ = spec.meridional(xs, zs)
    I = np.sum(np.abs(comp) ** 2, axis=0)
    a.imshow(I.T, extent=[zs[0], zs[-1], xs[0], xs[-1]], origin="lower",
             aspect="auto", cmap="inferno", norm=PowerNorm(0.5))
    a.set_title(f"{tag} through focus  (DoF $\\approx$ {s.depth_of_focus_nm:.0f} nm)")
    a.set_xlabel("z / $\\lambda$"); a.set_ylabel("x / $\\lambda$")
    a.set_ylim(-1.5, 1.5)
fig.tight_layout()
""")

md(r"""
## 3 · Imaging a circuit

Now an extended object with no symmetry: a Manhattan routing layout, imaged at
wafer scale through each objective with partially coherent illumination
(conventional disc source, $\sigma = 0.6$).

The mask is given at wafer scale — on the reticle every feature is 4× larger.
""")

code(r"""
CIRCUIT = "assets/circuit_pattern.png"

# sample each system near its own resolution limit
euv_px, duv_px = 4.0, 12.0          # nm per pixel
N = 512
mask_euv = vw.Mask.from_image(CIRCUIT, pixel=euv_px, size=N)
mask_duv = vw.Mask.from_image(CIRCUIT, pixel=duv_px, size=N)

im_euv = euv.imaging(pixel_nm=euv_px, size=N, polarization="x")
im_duv = duv.imaging(pixel_nm=duv_px, size=N, polarization="x")

img_euv = im_euv.aerial_image(mask_euv, sigma=0.6, source_points=9, vector=True)
img_duv = im_duv.aerial_image(mask_duv, sigma=0.6, source_points=9, vector=True)

for tag, s, im, img in (("EUV", euv, im_euv, img_euv), ("DUV", duv, im_duv, img_duv)):
    fr = img.fractions()
    print(f"{tag}: field {N*im.pixel/1000:.2f} um, Rayleigh half-pitch "
          f"{im.rayleigh_half_pitch:.1f} nm, contrast {img.contrast():.3f}")
    print(f"      aerial-image energy  x={fr['x']:.4f}  y={fr['y']:.5f}  z={fr['z']:.4f}")
""")

code(r"""
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for row, (tag, mask, img) in enumerate((("EUV", mask_euv, img_euv),
                                        ("DUV", mask_duv, img_duv))):
    ext = np.array(mask.grid.extent) / 1000.0            # microns
    ax[row, 0].imshow(mask.data, extent=ext, origin="lower", cmap="gray")
    ax[row, 0].set_title(f"{tag} · object (wafer scale)")
    ax[row, 1].imshow(img.normalized, extent=ext, origin="lower", cmap="inferno")
    ax[row, 1].set_title(f"{tag} · vector aerial image  (contrast {img.contrast():.2f})")
    Iz = img.Iz / img.total.max()
    ax[row, 2].imshow(Iz, extent=ext, origin="lower", cmap="viridis")
    ax[row, 2].set_title(f"{tag} · longitudinal $|E_z|^2$  ({img.fractions()['z']*100:.2f} %)")
    for a in ax[row]:
        a.set_xlabel("x (um)"); a.set_ylabel("y (um)")
fig.tight_layout()
""")

md(r"""
The longitudinal panel is the interesting one. In EUV it is a rounding error.
In DUV it sits on the feature edges — it adds intensity where the image should
be dark, so it eats contrast without carrying information.

### What the vector model changes
""")

code(r"""
scal = im_duv.aerial_image(mask_duv, sigma=0.6, source_points=9, vector=False)
vect = img_duv
d = vect.normalized - scal.normalized

fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
ext = np.array(mask_duv.grid.extent) / 1000.0
ax[0].imshow(scal.normalized, extent=ext, origin="lower", cmap="inferno")
ax[0].set_title(f"DUV scalar  (contrast {scal.contrast():.3f})")
ax[1].imshow(vect.normalized, extent=ext, origin="lower", cmap="inferno")
ax[1].set_title(f"DUV vector  (contrast {vect.contrast():.3f})")
m = np.abs(d).max()
im = ax[2].imshow(d, extent=ext, origin="lower", cmap="RdBu_r", vmin=-m, vmax=m)
ax[2].set_title("vector - scalar")
plt.colorbar(im, ax=ax[2], shrink=0.85)
for a in ax:
    a.set_xlabel("x (um)"); a.set_ylabel("y (um)")
fig.tight_layout()
print(f"contrast: scalar {scal.contrast():.4f}  vector {vect.contrast():.4f}")
""")

md(r"""
## 4 · Resolution

Contrast of equal line/space gratings against half pitch. The coherent cutoff
sits at $\lambda/4\mathrm{NA}$; useful printing stops well before it.
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, (tag, s, px) in zip(ax, (("EUV", euv, 2.0), ("DUV", duv, 6.0))):
    imsys = s.imaging(pixel_nm=px, size=256, polarization="x")
    cutoff = imsys.coherent_cutoff_half_pitch
    hp = np.linspace(3.2 * cutoff, 0.9 * cutoff, 14)
    c = imsys.contrast_vs_pitch(hp, sigma=0.6, source_points=5, vector=True)
    a.plot(hp, c, "o-", lw=1.6)
    a.axhline(0.6, color="grey", ls=":", lw=1)
    a.axvline(cutoff, color="crimson", ls="--", lw=1, label=f"cutoff {cutoff:.0f} nm")
    a.set_xlabel("half pitch (nm)"); a.set_ylabel("contrast")
    a.set_title(f"{tag}  (NA {s.na:g}, $\\lambda$ {s.wavelength_nm:.1f} nm)")
    a.legend(fontsize=8); a.grid(alpha=0.25)
    ok = hp[c >= 0.6]
    k1 = ok.min() * s.na / s.wavelength_nm if ok.size else float("nan")
    msg = f"{ok.min():.0f} nm  (k1 = {k1:.2f})" if ok.size else "not reached in range"
    print(f"{tag}: contrast >= 0.6 down to half pitch {msg}")
fig.tight_layout()
""")

md(r"""
## 5 · Why polarization is a knob at hyper-NA

Dense lines interfere with two beams that meet at a large angle. If the light is
polarized **along** the lines (TE), the two beams' fields stay parallel and
interfere fully. Polarized **across** them (TM), the fields are tilted with
respect to each other and the fringe contrast collapses.

The scalar model cannot see this at all — it gives one answer for both.
""")

code(r"""
imsys_y = duv.imaging(pixel_nm=6.0, size=256, polarization="y")   # along the lines (TE)
imsys_x = duv.imaging(pixel_nm=6.0, size=256, polarization="x")   # across them (TM)
hp = np.linspace(140, 45, 12)

c_te = imsys_y.contrast_vs_pitch(hp, sigma=0.4, source_points=5, vector=True)
c_tm = imsys_x.contrast_vs_pitch(hp, sigma=0.4, source_points=5, vector=True)
c_sc = imsys_x.contrast_vs_pitch(hp, sigma=0.4, source_points=5, vector=False)

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(hp, c_te, "o-", label="TE — polarized along the lines")
ax.plot(hp, c_tm, "s-", label="TM — polarized across the lines")
ax.plot(hp, c_sc, "k--", lw=1.2, label="scalar model (blind to polarization)")
ax.set_xlabel("half pitch (nm)"); ax.set_ylabel("contrast")
ax.set_title("DUV NA 1.2: vertical line/space gratings")
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()

i = np.argmin(np.abs(hp - 60))
print(f"at half pitch {hp[i]:.0f} nm:  TE {c_te[i]:.3f}   TM {c_tm[i]:.3f}   "
      f"scalar {c_sc[i]:.3f}")
""")

md(r"""
## 6 · What the systems actually look like

Everything so far used only the packaged wavefronts. Here we import the ray
tracer to draw the hardware they came from — the folded catadioptric DUV
objective and the six-mirror EUV projector — with real traced ray fans.

This is the division of labour that makes the whole thing tractable: the ray
tracer carries the field through metres of glass and mirrors to the exit pupil,
and `vectorwave` does wave optics on the last few microns. A full-wave solve of
the EUV train would be ~10^8 wavelengths long; the focus is ~30.
""")

code(r"""
import sys
sys.path.insert(0, "/home/claude/build/geometrical-raytracer")

from raytracer.design import OpticalSystem, SurfaceRow
from raytracer.propagation import (FieldPoint, SequentialTracer, chief_ray_slope,
                                   solve_object_plane, trace_from_object)
from raytracer.viz.sag_drawing import sample_profile_curve, surface_semidiameter as _semi

import json
from vectorwave.systems import DATA_DIR
PRESC = DATA_DIR / "prescriptions"


def build_duv():
    s = OpticalSystem.from_prescription(PRESC / "duv_prescription.csv")
    t = SequentialTracer(s); solve_object_plane(t)
    return s, t, dict(fields=[56.0, 62.0, 67.0], fan=0.305, chief={}, )


def build_euv():
    spec = json.loads((PRESC / "euv_prescription.json").read_text())
    th, mir = spec["thicknesses_mm"], spec["mirrors"]
    rows = [SurfaceRow.mirror(radius=mir[0]["radius_mm"], conic=mir[0]["conic"],
                              coefficients=mir[0]["coefficients"], thickness=th[0]),
            SurfaceRow.stop(thickness=th[1])]
    for i in range(1, 6):
        rows.append(SurfaceRow.mirror(radius=mir[i]["radius_mm"], conic=mir[i]["conic"],
                                      coefficients=mir[i]["coefficients"],
                                      thickness=th[i + 1]))
    s = OpticalSystem(rows, wavelength_um=spec["wavelength_mm"] * 1e3, name="EUV")
    s.object_z = spec["object_z_mm"]
    t = SequentialTracer(s, clip_apertures=False)
    return s, t, dict(fields=spec["field_heights_mm"], fan=spec["na_object"],
                      chief=dict(stop_index=spec["stop_index"],
                                 bracket=(-0.3, 0.05), scan=71))


def draw(system, tracer, cfg, ax, title, n_rays=31):
    colors = [(0.30, 0.65, 1.0), (0.30, 0.90, 0.50), (1.0, 0.55, 0.25)]
    ytop = 0.0
    for fy, rgb in zip(cfg["fields"], colors):
        sy0 = chief_ray_slope(tracer, FieldPoint(y=fy), **cfg["chief"])
        for d in np.linspace(-cfg["fan"], cfg["fan"], n_rays):
            res = trace_from_object(tracer, (0.0, fy), (0.0, sy0 + d), keep_path=True)
            if res.path is None:
                continue
            p = res.path[~np.isnan(res.path[:, 2])]
            if len(p) > 1:
                ax.plot(p[:, 2], p[:, 1], color=rgb, lw=0.35, alpha=0.55)
                ytop = max(ytop, np.abs(p[:, 1]).max())
    mirrors = set(system.mirror_indices)
    for i in range(len(system)):
        try:
            z, h = sample_profile_curve(system, i, samples=180)
        except Exception:
            continue
        ax.plot(z, h, color="#ffd24a" if i in mirrors else "#9fb6c9",
                lw=2.4 if i in mirrors else 0.7, alpha=0.95)
        ytop = max(ytop, np.nanmax(np.abs(h)) if len(h) else 0.0)
    ax.axvline(system.object_z, color="#888", ls=":", lw=1)
    ax.axvline(system.image_z, color="w", lw=1.2)
    ymax = max(ytop, max((_semi(r, 0.0) for r in system.rows), default=0.0)) * 1.12
    ax.set_ylim(-ymax, ymax); ax.set_facecolor("black")
    ax.set_xlabel("z (mm)"); ax.set_ylabel("y (mm)"); ax.set_title(title, color="w")
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_color("#555")


duv_sys, duv_tr, duv_cfg = build_duv()
euv_sys, euv_tr, euv_cfg = build_euv()

fig, ax = plt.subplots(2, 1, figsize=(13, 7.5), facecolor="black")
draw(duv_sys, duv_tr, duv_cfg, ax[0],
     f"DUV US7557996 — {len(duv_sys)} surfaces, 2 mirrors, NA 1.2 immersion")
draw(euv_sys, euv_tr, euv_cfg, ax[1],
     "EUV US7151592 — 6 aspheric mirrors, NA 0.22, 1500 mm track")
fig.tight_layout()
print(f"DUV: object z={duv_sys.object_z:.1f} mm, image z={duv_sys.image_z:.1f} mm, "
      f"n_image={duv_sys.n_image:.3f}")
print(f"EUV: object z={euv_sys.object_z:.1f} mm, image z={euv_sys.image_z:.1f} mm")
""")

md(r"""
## 7 · A projector the vector effect ruins

The two systems above are production designs: they are diffraction-limited and
their illumination is chosen so polarization never becomes fatal. To see what
the vector nature of light can actually *do* to an image, take the hyper-NA
immersion diopter from the `vecdiff` study — the last element of a projector,
**LuAG (n = 2.14) into water (n = 1.437)**, a stigmatic Cartesian oval worked
right up to its grazing-incidence aperture limit.

Two things make it brutal:

1. **The aperture is enormous** — NA ≈ 0.88 in water, so at the coherent cutoff
   the $\pm 1$ orders leave at $\pm\theta$ with $\sin\theta = \mathrm{NA}/n$.
   In TM their electric fields project onto each other as $\cos 2\theta$.
2. **It is a real refracting surface, not an ideal lens.** Its transfer
   operator carries the Fresnel $t_s$, $t_p$ and the flux factor $A(Q)$, which
   diverge from each other badly at grazing incidence.

We import `vecdiff` for exactly that operator and inject it into a
`vectorwave` `ImagingSystem` through the `transfer` hook.
""")

code(r"""
sys.path.insert(0, "/home/claude/build/vecdiff")
from vecdiff import CartesianSurface
from vecdiff.propagation import transfer_weights_on_grid

SURF = CartesianSurface(n0=2.14, ni=1.437, z0=-42.0, zi=2.0)
APERTURE = SURF.aperture_limit
SIN_MAX = float(SURF.ray_geometry(np.array([APERTURE * (1 - 1e-6)])).sin_ai[0])
N_WATER = 1.437
NA_BAD = N_WATER * SIN_MAX

# invert the pupil map r <-> sin(alpha_i) once
_r = np.linspace(1e-6, APERTURE * (1 - 1e-9), 4000)
_sin = np.asarray(SURF.ray_geometry(_r).sin_ai, dtype=float)


def diopter_transfer(rho, phi, inside):
    # vecdiff's interface eigenvalues (radial, azimuthal, longitudinal)
    r = np.interp(np.clip(rho * SIN_MAX, 0, _sin.max()), _sin, _r)
    return transfer_weights_on_grid(r, SURF, support=inside)


theta = np.degrees(np.arcsin(SIN_MAX))
print(f"LuAG -> water stigmatic diopter: aperture {APERTURE:.3f} mm, "
      f"sin(alpha_i)max = {SIN_MAX:.4f}")
print(f"NA = {NA_BAD:.4f}  (theta = {theta:.1f} deg,  cos 2theta = "
      f"{np.cos(2*np.radians(theta)):+.3f}  <- the TM projection at the cutoff)")
for tag, na, nn in (("EUV", euv.na, euv.n_image), ("DUV", duv.na, duv.n_image),
                    ("bad diopter", NA_BAD, N_WATER)):
    t = np.arcsin(na / nn)
    print(f"   {tag:12s} NA {na:4.2f}  n {nn:5.3f}  theta {np.degrees(t):5.1f} deg  "
          f"cos 2theta {np.cos(2*t):+.3f}")
""")

code(r"""
# the transfer eigenvalues across the pupil: an ideal lens would be flat at 1
rho = np.linspace(0, 1, 300)
lr, lp, lz = diopter_transfer(rho, np.zeros_like(rho), np.ones_like(rho, bool))
sin_t = rho * SIN_MAX
cos_t = np.sqrt(1 - sin_t**2)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(rho, lr, label=r"$\lambda_r$  (meridional, $t_p$)")
ax[0].plot(rho, lp, label=r"$\lambda_\varphi$  (azimuthal, $t_s$)")
ax[0].plot(rho, lz, label=r"$\lambda_z$  (longitudinal)")
ax[0].set_title("real diopter transfer eigenvalues"); ax[0].legend(fontsize=8)
ax[1].plot(rho, cos_t, label=r"$\cos\theta$  (ideal aplanatic $\lambda_r$)")
ax[1].plot(rho, np.ones_like(rho), label=r"$1$  (ideal $\lambda_\varphi$)")
ax[1].plot(rho, sin_t, label=r"$\sin\theta$  (ideal $\lambda_z$)")
ax[1].set_title("ideal aplanatic lens, for comparison"); ax[1].legend(fontsize=8)
for a in ax:
    a.set_xlabel(r"normalized pupil radius $\rho$"); a.grid(alpha=0.25)
fig.tight_layout()
""")

md(r"""
### Contrast at the resolution limit

Equal line/space gratings, illumination $\sigma = 0.2$, polarized **along** the
lines (TE) and **across** them (TM).
""")

code(r"""
lam_bad = 193.0
px_bad, N_bad = 8.0, 384   # wide window -> fine pitch quantization
hp_cut = 0.5 * lam_bad / NA_BAD   # half pitch where the +-1 orders hit the pupil rim


def snapped(hp, window):
    # nearest half pitch fitting an integer number of periods (avoids FFT leakage)
    return window / max(round(window / (2 * hp)), 1) / 2


window = px_bad * N_bad
hps = np.array([snapped(h, window) for h in np.linspace(2.4, 0.95, 16) * hp_cut])
hps = np.unique(hps)[::-1]

curves = {}
for tag, pol, transfer in (("scalar", "x", None),
                           ("vector TE (ideal lens)", "y", None),
                           ("vector TM (ideal lens)", "x", None),
                           ("vector TE (real diopter)", "y", diopter_transfer),
                           ("vector TM (real diopter)", "x", diopter_transfer)):
    imsys = vw.ImagingSystem(na=NA_BAD, wavelength=lam_bad, n=N_WATER,
                             pixel=px_bad, size=N_bad, polarization=pol,
                             transfer=transfer)
    curves[tag] = imsys.contrast_vs_pitch(hps, sigma=0.2, source_points=7,
                                          vector=(tag != "scalar"))

fig, ax = plt.subplots(figsize=(7, 4.2))
styles = {"scalar": dict(color="k", ls="--", lw=1.6),
          "vector TE (ideal lens)": dict(color="tab:blue", ls=":", lw=1.4),
          "vector TM (ideal lens)": dict(color="tab:red", ls=":", lw=1.4),
          "vector TE (real diopter)": dict(color="tab:blue", lw=2.2, marker="o", ms=4),
          "vector TM (real diopter)": dict(color="tab:red", lw=2.2, marker="s", ms=4)}
for tag, c in curves.items():
    ax.plot(hps, c, label=tag, **styles[tag])
ax.axvline(hp_cut, color="grey", ls="--", lw=1, label="$\\pm1$ orders at the rim")
ax.set_xlabel("half pitch (nm)"); ax.set_ylabel("contrast")
ax.set_title(f"LuAG$\\to$water diopter, NA {NA_BAD:.2f}, $\\sigma$ = 0.2")
ax.legend(fontsize=7.5); ax.grid(alpha=0.25); ax.set_ylim(0, 1.02)
fig.tight_layout()

i = int(np.argmin(np.abs(hps - 1.02 * hp_cut)))
print(f"at half pitch {hps[i]:.0f} nm  (1.02x the +-1-order cutoff {hp_cut:.0f} nm):")
for tag, c in curves.items():
    print(f"   {tag:26s} {c[i]:.3f}")
""")

md(r"""
That is the point of the section. The scalar model promises a printable image.
The ideal-lens vector model already shaves the TM contrast. The **real
diopter** — the same surface, with its own Fresnel and flux transfer — drops TM
by roughly a factor of three below TE, at a pitch where the scalar model still
says everything is fine.

A projector like this cannot be run with unpolarized light. Either you polarize
along the features, or the layout simply does not print.

### The same thing on the circuit
""")

code(r"""
mask_bad = vw.Mask.from_image(CIRCUIT, pixel=px_bad, size=N_bad)

variants = [("scalar model", "x", None, False),
            ("vector, TE (along lines)", "y", diopter_transfer, True),
            ("vector, TM (across lines)", "x", diopter_transfer, True)]
imgs = []
for title, pol, tr, vec in variants:
    imsys = vw.ImagingSystem(na=NA_BAD, wavelength=lam_bad, n=N_WATER,
                             pixel=px_bad, size=N_bad, polarization=pol, transfer=tr)
    imgs.append((title, imsys.aerial_image(mask_bad, sigma=0.2, source_points=7,
                                           vector=vec)))

fig, ax = plt.subplots(1, 4, figsize=(15, 3.9))
ext = np.array(mask_bad.grid.extent) / 1000.0
ax[0].imshow(mask_bad.data, extent=ext, origin="lower", cmap="gray")
ax[0].set_title("object")
for a, (title, img) in zip(ax[1:], imgs):
    a.imshow(img.normalized, extent=ext, origin="lower", cmap="inferno", vmin=0, vmax=1)
    a.set_title(f"{title}\ncontrast {img.contrast():.2f}, "
                f"$|E_z|^2$ {img.fractions()['z']*100:.1f} %")
for a in ax:
    a.set_xlabel("x (um)"); a.set_ylabel("y (um)")
fig.tight_layout()
""")

md(r"""
## 8 · Polarization maps of the three focal fields

Finally, the focal fields themselves, drawn with `vecdiff`'s polarization
plotter through the adapter. The ellipse field is uniform and linear at EUV,
visibly structured at DUV, and strongly disturbed for the diopter.
""")

code(r"""
from vectorwave.adapters import plot_polarization

def focal_field(na, n, transfer=None, polarization="x", half=1.6, step=1/48,
                wavefront=None):
    p = vw.Pupil(na=na, n=n, wavelength=1.0, polarization=polarization,
                 wavefront=wavefront)
    g = vw.Grid.from_spacing(0.25, 256)
    x = np.arange(-half, half + 1e-9, step)
    return p.spectrum(g).field_on(x, x, 0.0)

# each panel spans ~1.6 Airy widths of its own system, so the spots are comparable
spans = {na: 1.6 * 0.5144 / na for na in (euv.na, duv.na, NA_BAD)}
fields = [("EUV  NA 0.22", euv.na,
           focal_field(euv.na, euv.n_image, half=spans[euv.na],
                       step=spans[euv.na]/90, wavefront=euv.wavefront)),
          ("DUV  NA 1.20", duv.na,
           focal_field(duv.na, duv.n_image, half=spans[duv.na],
                       step=spans[duv.na]/90, wavefront=duv.wavefront)),
          (f"diopter  NA {NA_BAD:.2f}", NA_BAD,
           focal_field(NA_BAD, N_WATER, half=spans[NA_BAD], step=spans[NA_BAD]/90))]

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
for a, (tag, na, f) in zip(ax, fields):
    span = spans[na]
    try:
        plot_polarization(f, ax=a, half_size=0.92 * span, n_img=340,
                          sampling="cartesian", stride=13, scale=0.052 * span)
    except Exception as exc:                      # vecdiff not installed
        a.imshow(f.intensity, extent=[-span, span, -span, span], origin="lower",
                 cmap="inferno", norm=PowerNorm(0.5))
        print("vecdiff plot unavailable:", exc)
    fr = f.component_fractions()
    a.set_title(f"{tag}\n$|E_z|^2$ = {fr['z']*100:.1f} %,  "
                f"$|E_y|^2$ = {fr['y']*100:.2f} %")
    a.set_xlabel("x / $\\lambda$"); a.set_ylabel("y / $\\lambda$")
fig.tight_layout()

for tag, na, f in fields:
    fr = f.component_fractions()
    print(f"{tag:18s} Ex {fr['x']:.4f}  Ey {fr['y']:.5f}  Ez {fr['z']:.4f}")
""")

md(r"""
## Summary

* Both objectives are diffraction-limited, and their PSFs land on the Airy
  width — EUV at 31 nm FWHM, DUV at 83 nm.
* At NA 0.22 the focus is scalar; at NA 1.2 it is not — 14 % of the focal energy
  is longitudinal, and the aerial image inherits an edge-localized $E_z$
  background that costs contrast.
* Because of that, illumination polarization is a real design knob at hyper-NA
  and a non-issue at EUV — which is exactly why the two technologies make
  opposite engineering choices.

Everything above is `vectorwave` plus the two packaged systems; the ray tracing
happened once, offline, and is baked into the shipped wavefronts.
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                             "name": "python3"}
out = ROOT / "notebooks" / "lithography.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out)
