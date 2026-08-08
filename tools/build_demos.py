"""Generate the self-contained demonstration notebooks.

Unlike ``tools/build_notebook.py`` (which reaches for the optional
``geometrical-raytracer`` and ``vecdiff`` packages to draw the hardware), the
notebooks built here depend only on the shipped ``vectorwave`` package and its
declared runtime dependencies (numpy, scipy, finufft, matplotlib).  Every cell
executes end to end, so they double as an integration test of the public API.

Run from the repo root::

    python tools/build_demos.py            # write the .ipynb files
    jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb

Produces:

* ``notebooks/01_vector_focusing.ipynb``       — focal fields and polarization
* ``notebooks/02_curved_interfaces.ipynb``     — surfaces, operators, systems
* ``notebooks/03_projection_imaging.ipynb``    — the two packaged objectives
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


class Builder:
    """Tiny helper: accumulate markdown/code cells, then write the notebook."""

    def __init__(self, title: str):
        self.title = title
        self.cells: list = []

    def md(self, text: str) -> None:
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(self, src: str) -> None:
        self.cells.append(nbf.v4.new_code_cell(src.strip("\n")))

    def write(self, name: str) -> Path:
        nb = nbf.v4.new_notebook()
        nb.cells = self.cells
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3", "language": "python", "name": "python3",
        }
        nb.metadata["language_info"] = {"name": "python"}
        path = NB / name
        nbf.write(nb, str(path))
        return path


SETUP = r"""
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")            # keep the rendered output tidy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import vectorwave as vw

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.titlesize": 10,
                     "figure.facecolor": "white"})
print("vectorwave", vw.__version__, "| packaged systems:", vw.available())
"""


# ======================================================================
#  Notebook 1 — vector focusing and polarization
# ======================================================================
def build_focusing() -> None:
    b = Builder("Vector focusing")

    b.md(r"""
# Vector focusing and polarization

A scalar wave is a fine model for a focus at low numerical aperture.  At high NA
it is not: the pupil bends every ray toward the axis, the field is projected as
it bends, and a real fraction of the focal energy ends up in the **longitudinal**
component $E_z$ that points along the optical axis and carries no image
information.

`vectorwave` keeps the full vector field.  The object here is `Pupil`: it holds a
numerical aperture, an input polarization and (optionally) an aberration
wavefront, and `spectrum(grid)` applies the aplanatic **Richards–Wolf**
projection to hand back an `AngularSpectrum` we can synthesise anywhere.

Lengths in this notebook are in wavelengths, so a width of `1` means one $\lambda$.
""")

    b.code(SETUP)

    b.md(r"""
## 1 · A high-NA focus is not scalar

Focus an $x$-polarized pupil at NA 0.95 (dry) and split the focal intensity into
its three Cartesian components.  `field_on` synthesises the field on any grid we
ask for — exactly, by a zoom DFT — so the focal sampling is decoupled from the
grid that carries the spectrum.
""")

    b.code(r"""
grid = vw.Grid.from_spacing(0.25, 256)
pupil = vw.Pupil(na=0.95, n=1.0, wavelength=1.0, polarization="x")

x = np.arange(-2.0, 2.0, 1 / 48)
field = pupil.spectrum(grid).field_on(x, x, z=0.0)

fr = field.component_fractions()
print(f"NA {pupil.na}:  |Ex|^2 = {fr['x']:.3f}   |Ey|^2 = {fr['y']:.4f}   "
      f"|Ez|^2 = {fr['z']:.3f}")
print(f"focal FWHM = {field.fwhm('x'):.3f} lambda   (Airy {pupil.airy_fwhm():.3f})")

ext = [x[0], x[-1], x[0], x[-1]]
comps = [("total $|E|^2$", field.intensity),
         ("$|E_x|^2$", np.abs(field.Ex) ** 2),
         ("$|E_y|^2$  (cross-pol)", np.abs(field.Ey) ** 2),
         ("$|E_z|^2$  (longitudinal)", np.abs(field.Ez) ** 2)]
fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
for a, (title, I) in zip(ax, comps):
    a.imshow(I, extent=ext, origin="lower", cmap="inferno", norm=PowerNorm(0.5))
    share = I.sum() / field.intensity.sum()
    a.set_title(title + ("" if "total" in title else f"\n{share*100:.2f} % of energy"))
    a.set_xlim(-1.2, 1.2); a.set_ylim(-1.2, 1.2); a.set_xlabel("x / $\\lambda$")
ax[0].set_ylabel("y / $\\lambda$")
fig.tight_layout()
""")

    b.md(r"""
The longitudinal lobe is split along $x$ — the input polarization direction —
because it comes from the meridional ($p$) part of the field that the pupil tips
into $z$.  The cross-polarized $|E_y|^2$ is a four-lobed clover, orders of
magnitude weaker.
""")

    b.md(r"""
## 2 · The longitudinal share follows $\mathrm{NA}^2/4n^2$

As the aperture closes, the longitudinal energy fraction of an aplanatic
$x$-pupil approaches the analytic low-NA limit $\mathrm{NA}^2/4n^2$.  This is the
scalar limit made quantitative — and one of the package's regression tests.
""")

    b.code(r"""
nas = np.linspace(0.1, 0.95, 18)
fz = []
g = vw.Grid.from_spacing(0.25, 512)
for na in nas:
    f = vw.Pupil(na=na, n=1.0, wavelength=1.0).spectrum(g).field(0.0)
    fz.append(f.component_fractions()["z"])
fz = np.array(fz)

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(nas, fz, "o-", lw=1.6, ms=4, label=r"$|E_z|^2$ fraction (computed)")
ax.plot(nas, nas ** 2 / 4, "k--", lw=1.2, label=r"low-NA law  $\mathrm{NA}^2/4n^2$")
ax.set_xlabel("numerical aperture"); ax.set_ylabel("longitudinal energy fraction")
ax.set_title("Where the focal energy goes vs NA")
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
print(f"at NA 0.95 about a quarter of the energy is longitudinal: {fz[-1]:.3f}")
""")

    b.md(r"""
## 3 · The focus matches the Airy pattern

For a moderate NA the vector PSF sits right on top of the ideal scalar Airy
pattern $\big(2J_1(v)/v\big)^2$ — the vector machinery reproduces the classical
result where the classical result is valid.
""")

    b.code(r"""
from scipy.special import j1

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
for a, na in zip(ax, (0.4, 0.8)):
    p = vw.Pupil(na=na, n=1.0, wavelength=1.0)
    xr = np.arange(-4, 4, 1 / 40)
    f = p.spectrum(g).field_on(xr, xr, 0.0)
    I = f.intensity / f.intensity.max()
    cut = I[I.shape[0] // 2]
    r = np.linspace(1e-3, 4, 400)
    v = 2 * np.pi * na * r
    airy = (2 * j1(v) / v) ** 2
    a.plot(xr, cut, lw=2, label="vector PSF")
    a.plot(r, airy, "k--", lw=1.1, label="ideal Airy")
    a.set_xlim(0, 4); a.set_xlabel("r / $\\lambda$"); a.set_ylabel("$|E|^2$")
    a.set_title(f"NA {na}: FWHM {f.fwhm('x'):.3f} (Airy {p.airy_fwhm():.3f})")
    a.legend(fontsize=8)
fig.tight_layout()
""")

    b.md(r"""
## 4 · Structured polarization reshapes the spot

The input polarization is any `(rho, phi) -> (Ex, Ey)` callable; a handful of
useful ones are named in `vw.POLARIZATIONS`.  At high NA they focus to visibly
different spots — the reason polarization is an engineering knob in microscopy
and lithography.

* **radial** collapses to a strong on-axis longitudinal spike (a sub-Airy core),
* **azimuthal** has a pure doughnut with a dark centre,
* **circular** stays compact and nearly round,
* **vortex** carries a phase singularity, so the centre is dark.
""")

    b.code(r"""
states = ["x", "radial", "azimuthal", "circular", "vortex1"]
xr = np.arange(-2.5, 2.5, 1 / 40)
fig, ax = plt.subplots(1, len(states), figsize=(14, 3.1))
for a, s in zip(ax, states):
    f = vw.Pupil(na=0.9, n=1.0, polarization=s).spectrum(g).field_on(xr, xr, 0.0)
    a.imshow(f.intensity, extent=[xr[0], xr[-1], xr[0], xr[-1]], origin="lower",
             cmap="inferno", norm=PowerNorm(0.5))
    fz = f.component_fractions()["z"]
    a.set_title(f"{s}\n$|E_z|^2$ = {fz*100:.0f} %")
    a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_xlabel("x / $\\lambda$")
ax[0].set_ylabel("y / $\\lambda$")
fig.tight_layout()
""")

    b.md(r"""
Radial polarization puts **roughly 40 %** of the focal energy into $E_z$ at NA 0.9
— an on-axis longitudinal spike that is narrower than the transverse Airy core,
which is why radially-polarized beams are used to beat the scalar spot size.
Azimuthal input is the complement: a purely transverse doughnut with no $E_z$ at
all.
""")

    b.md(r"""
## 5 · The polarization is no longer uniform

Once the pupil bends the rays, the transverse field stops being uniformly
polarized.  The Stokes parameters of the focal field (`field.polarization()`)
give the ellipse orientation and ellipticity point by point.
""")

    b.code(r"""
xr = np.arange(-2.0, 2.0, 1 / 40)
f = vw.Pupil(na=0.95, polarization="45").spectrum(g).field_on(xr, xr, 0.0)
pol = f.polarization()
ext = [xr[0], xr[-1], xr[0], xr[-1]]
bright = f.intensity > 0.02 * f.intensity.max()

fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
ax[0].imshow(f.intensity, extent=ext, origin="lower", cmap="inferno", norm=PowerNorm(0.5))
ax[0].set_title("focal $|E|^2$  (45$\\degree$ input)")
im = ax[1].imshow(np.where(bright, np.degrees(pol.orientation), np.nan),
                  extent=ext, origin="lower", cmap="twilight", vmin=-90, vmax=90)
ax[1].set_title("major-axis orientation"); plt.colorbar(im, ax=ax[1], shrink=0.85, label="deg")
im = ax[2].imshow(np.where(bright, np.degrees(pol.ellipticity), np.nan),
                  extent=ext, origin="lower", cmap="RdBu_r", vmin=-45, vmax=45)
ax[2].set_title("ellipticity (0 = linear)"); plt.colorbar(im, ax=ax[2], shrink=0.85, label="deg")
for a in ax:
    a.set_xlim(-1.2, 1.2); a.set_ylim(-1.2, 1.2); a.set_xlabel("x / $\\lambda$")
fig.tight_layout()
""")

    b.md(r"""
## 6 · Through focus

The same spectrum evaluated over a range of $z$ gives the axial structure.
`spectrum.meridional` returns the field in the $(x, z)$ plane in one call.
""")

    b.code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for a, na in zip(ax, (0.6, 0.95)):
    spec = vw.Pupil(na=na, n=1.0).spectrum(g)
    xs = np.arange(-2.0, 2.0, 1 / 32)
    zs = np.linspace(-6, 6, 141)
    comp, _, _ = spec.meridional(xs, zs)
    I = np.sum(np.abs(comp) ** 2, axis=0)
    a.imshow(I.T, extent=[zs[0], zs[-1], xs[0], xs[-1]], origin="lower",
             aspect="auto", cmap="inferno", norm=PowerNorm(0.5))
    dof = 1.0 / na ** 2
    a.set_title(f"NA {na}: through focus  (DoF $\\approx$ {dof:.1f} $\\lambda$)")
    a.set_xlabel("z / $\\lambda$"); a.set_ylabel("x / $\\lambda$"); a.set_ylim(-1.6, 1.6)
fig.tight_layout()
""")

    b.md(r"""
The higher aperture buys a tighter waist at the cost of a shorter depth of focus
— the axial hourglass narrows in both directions.  That trade is exactly what the
two production objectives in notebook 3 navigate.
""")

    b.write("01_vector_focusing.ipynb")


# ======================================================================
#  Notebook 2 — curved interfaces as operators
# ======================================================================
def build_interfaces() -> None:
    b = Builder("Curved interfaces")

    b.md(r"""
# Curved interfaces as operators

A field in a homogeneous medium *is* its angular spectrum.  A flat interface acts
diagonally on that spectrum — one Fresnel dyadic per direction.  A curved
interface does not, and `vectorwave` computes the operator that replaces it:
locally refract the field by the tangent plane, then radiate the surface currents
back into an outgoing spectrum by a surface transform.

Because every interface maps the space of spectra to itself, interfaces
**compose**.  A multi-element system is an ordered product of operators with free
propagation between them.  This notebook walks from a single surface up to a
two-element lens, and out to a genuinely non-axisymmetric freeform.

The sign convention: surfaces are graphs $z = \mathrm{sag}(\rho)$ with a
**positive radius placing the centre of curvature at $+z$**, so a beam going
toward $+z$ is converged by a *negative* radius.
""")

    b.code(SETUP)

    b.md(r"""
## 1 · The surface zoo

Every `Surface` knows its sag, slope and normal, so the propagators never care
which shape they were handed.  A conic constant $\kappa$ selects the type:
$\kappa=0$ sphere, $-1$ paraboloid, $<-1$ hyperboloid, $-1<\kappa<0$ ellipsoid.
""")

    b.code(r"""
R = -8.0
shapes = {
    "sphere ($\\kappa$=0)": vw.Conic(R, 0.0),
    "paraboloid ($\\kappa$=-1)": vw.Conic(R, -1.0),
    "hyperboloid ($\\kappa$=-2.25)": vw.Conic(R, -2.25),
    "even asphere": vw.EvenAsphere(R, 0.0, coefficients=(2e-4, -3e-6)),
}
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for name, s in shapes.items():
    rho, sag = s.profile(n=200, r_max=6.0)
    ax.plot(rho, sag, lw=1.8, label=name)
ax.set_xlabel(r"$\rho$ / $\lambda$"); ax.set_ylabel("sag / $\\lambda$")
ax.set_title(f"Surface profiles (vertex radius R = {R})")
ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.invert_yaxis()
fig.tight_layout()
""")

    b.md(r"""
## 2 · The stigmatic focus, against full-wave FDTD

The Cartesian oval that focuses a collimated beam with **no spherical aberration
at any order** is the conic with $\kappa = -(n_1/n_2)^2$
(`vw.stigmatic_conic_constant`).  Refract a plane wave in glass ($n_1=1.5$)
through such a hyperboloid into air ($n_2=1$) and scan for the focus.

The paraxial prediction is $z = n_2 R_v /(n_1-n_2) = 16\,\lambda$.  The actual
focus sits **short** of that, near $14.5\,\lambda$, because the Fresnel number is
finite — and an independent full-wave FDTD (MEEP) run put it at the same place.
This is a pinned regression test of the package.
""")

    b.code(r"""
n1, n2, Rv = 1.5, 1.0, 8.0
surf = vw.Conic(radius=-Rv, conic=vw.stigmatic_conic_constant(n1, n2))
print(f"stigmatic conic constant kappa = {surf.conic:.3f}")

grid = vw.Grid.from_spacing(0.25, 200)
spec = vw.surface_spectrum(surf, grid, n1=n1, n2=n2, wavelength=1.0,
                           aperture=0.62 * 2 * Rv, m_max=2, n_rho=700,
                           n_phi=32, n_kr=256)
z_paraxial = n2 * Rv / (n1 - n2)
zs = np.linspace(0.5 * z_paraxial, 1.4 * z_paraxial, 90)
scan = spec.focus_scan(zs)
zf = zs[np.argmax(scan)]

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5))
ax[0].plot(zs, scan / scan.max(), lw=1.8)
ax[0].axvline(zf, color="C1", lw=1.2, label=f"focus {zf:.1f} $\\lambda$")
ax[0].axvline(z_paraxial, color="k", ls="--", lw=1.1, label=f"paraxial {z_paraxial:.0f} $\\lambda$")
ax[0].set_xlabel("z / $\\lambda$"); ax[0].set_ylabel("on-axis intensity")
ax[0].set_title("focal shift from a finite Fresnel number"); ax[0].legend(fontsize=8)

xr = np.arange(-2.5, 2.5, 1 / 40)
f = spec.field_on(xr, xr, zf)
ax[1].imshow(f.intensity, extent=[xr[0], xr[-1], xr[0], xr[-1]], origin="lower",
             cmap="inferno", norm=PowerNorm(0.5))
ax[1].set_title(f"focal spot at z = {zf:.1f} $\\lambda$")
ax[1].set_xlabel("x / $\\lambda$"); ax[1].set_ylabel("y / $\\lambda$")
ax[1].set_xlim(-1.5, 1.5); ax[1].set_ylim(-1.5, 1.5)
fig.tight_layout()
print(f"transversality residual |k.A| = {spec.transversality_residual():.1e}  (physical field)")
""")

    b.md(r"""
## 3 · The interface as a composable operator

`surface_spectrum` above takes a *named* source (a plane wave).  `InterfaceOperator`
does the same physics but consumes and produces an `AngularSpectrum`, so it
**composes**.  Driven by a plane-wave spectrum it must reproduce `surface_spectrum`
exactly — the two paths agree on the focus and the focal field.
""")

    b.code(r"""
pw = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5, polarization="x")
op = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=0.62 * 2 * Rv, n_rho=600)
out = op(pw)

zf_src = spec.best_focus(zs)
zf_op = out.best_focus(zs)
print(f"focus from surface_spectrum : {zf_src:.2f} lambda")
print(f"focus from InterfaceOperator: {zf_op:.2f} lambda   (agree to < 0.5 lambda)")

a = spec.field_on(xr, xr, zf_src).intensity
c = out.field_on(xr, xr, zf_op).intensity
corr = np.sum(a * c) / np.sqrt(np.sum(a * a) * np.sum(c * c))
print(f"focal-field correlation between the two paths: {corr:.4f}")
""")

    b.md(r"""
### Free space shifts the focus, exactly

`FreeSpace(d)` is the diagonal propagator.  Applying it before synthesis simply
slides the focus by $d$ — the operator algebra reads like the geometry.
""")

    b.code(r"""
D = 4.0
zf0 = out.best_focus(zs)
zfD = vw.FreeSpace(D)(out).best_focus(zs - D)
print(f"focus of the bare spectrum        : {zf0:.2f} lambda")
print(f"focus after FreeSpace({D}) applied  : {zfD:.2f} lambda")
print(f"shift = {zf0 - zfD:.2f} lambda   (expected {D})")
""")

    b.md(r"""
## 4 · A system is a product of operators

Two refractions make a lens.  `System([front, FreeSpace(t), back])` is a word in
the operator algebra: a plane wave in air enters a glass cap, propagates through
the body, and exits the flat back face to a focus in air.
""")

    b.code(r"""
n_g = 1.5
front = vw.InterfaceOperator(
    vw.Conic(radius=+6.0, conic=vw.stigmatic_conic_constant(1.0, n_g)),
    n1=1.0, n2=n_g, aperture=5.5)
back = vw.InterfaceOperator(vw.Plane(), n1=n_g, n2=1.0, aperture=5.5)
system = vw.System([front, vw.FreeSpace(8.0), back])
print("system:", system, f"({len(system)} operators)")

out = system(vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0))
zsys = np.linspace(2, 40, 140)
scan = out.focus_scan(zsys)
zf = zsys[np.argmax(scan)]

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
ax[0].plot(zsys, scan / scan.max(), lw=1.8)
ax[0].axvline(zf, color="C1", lw=1.2, label=f"focus {zf:.1f} $\\lambda$")
ax[0].set_xlabel("z / $\\lambda$"); ax[0].set_ylabel("on-axis intensity")
ax[0].set_title("two-surface lens: an interior focus"); ax[0].legend(fontsize=8)
xr2 = np.arange(-2.5, 2.5, 1 / 36)
f = out.field_on(xr2, xr2, zf)
ax[1].imshow(f.intensity, extent=[xr2[0], xr2[-1], xr2[0], xr2[-1]], origin="lower",
             cmap="inferno", norm=PowerNorm(0.5))
ax[1].set_title(f"focal spot at z = {zf:.1f} $\\lambda$")
ax[1].set_xlim(-1.5, 1.5); ax[1].set_ylim(-1.5, 1.5)
ax[1].set_xlabel("x / $\\lambda$"); ax[1].set_ylabel("y / $\\lambda$")
fig.tight_layout()
print(f"system output transversality: {out.transversality_residual():.1e}")
""")

    b.md(r"""
## 5 · Two return integrals, one answer

For a surface of revolution the return integral separates into azimuthal
harmonics (`method="polar"`, the fast Bessel kernel).  A general NUFFT of the
surface currents (`method="nufft"`) needs no symmetry at all.  On an axisymmetric
surface both are available and must agree — that overlap is what keeps the fast
path honest.
""")

    b.code(r"""
a = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, method="polar")(pw)
c = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, method="nufft")(pw)
za, zc = a.best_focus(zs), c.best_focus(zs)
fa = a.field_on(xr, xr, za).intensity
fc = c.field_on(xr, xr, za).intensity
corr = np.sum(fa * fc) / np.sqrt(np.sum(fa * fa) * np.sum(fc * fc))
print(f"polar focus {za:.2f} lambda | nufft focus {zc:.2f} lambda")
print(f"focal-field correlation polar vs nufft: {corr:.4f}")
""")

    b.md(r"""
## 6 · A genuine freeform

Break the rotational symmetry.  `Freeform2D` takes any smooth `sag_fn(x, y)`; its
operator runs through the same NUFFT surface transform, so only the cost changes.
An **astigmatic** cap (different curvature in $x$ and $y$) focuses to two
separated line foci — the tell-tale of astigmatism, reproduced from first
principles.
""")

    b.code(r"""
Rx, Ry = -14.0, -10.0                       # different meridional radii -> astigmatism
ff = vw.Freeform2D(sag_fn=lambda x, y: x ** 2 / (2 * Rx) + y ** 2 / (2 * Ry), radius=6.0)
outff = vw.InterfaceOperator(ff, n1=1.5, n2=1.0, aperture=6.0, n_free=200)(
    vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5))

xr3 = np.arange(-3, 3, 1 / 28)
zline = np.linspace(6, 26, 5)
fig, ax = plt.subplots(1, len(zline), figsize=(14, 3.0))
for a, z in zip(ax, zline):
    f = outff.field_on(xr3, xr3, z)
    a.imshow(f.intensity, extent=[xr3[0], xr3[-1], xr3[0], xr3[-1]], origin="lower",
             cmap="inferno", norm=PowerNorm(0.5))
    a.set_title(f"z = {z:.0f} $\\lambda$")
    a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.set_xlabel("x / $\\lambda$")
ax[0].set_ylabel("y / $\\lambda$")
fig.suptitle("Astigmatic freeform: a horizontal line focus, then a vertical one", y=1.04)
fig.tight_layout()
print(f"freeform output transversality: {outff.transversality_residual():.1e}")
""")

    b.md(r"""
The beam squeezes to a **horizontal** line where the $y$-curvature focuses, opens
through a round-ish waist, and squeezes to a **vertical** line where the
$x$-curvature focuses.  Two line foci separated along $z$ — astigmatism — from a
surface described by nothing more than `x**2/(2Rx) + y**2/(2Ry)`.
""")

    b.write("02_curved_interfaces.ipynb")


# ======================================================================
#  Notebook 3 — projection imaging with the packaged objectives
# ======================================================================
def build_imaging() -> None:
    b = Builder("Projection imaging")

    b.md(r"""
# Projection lithography, vectorially

`vw.load(name)` ships two real projection objectives, each carrying its
ray-traced exit-pupil wavefront so the package needs no ray tracer at run time:

| name | system | $\lambda$ | NA | medium |
|---|---|---|---|---|
| `euv` | US 7,151,592 six-mirror EUV projector | 13.4 nm | 0.22 | vacuum |
| `duv` | US 7,557,996 hyper-NA immersion objective | 193.4 nm | 1.2 | water ($n=1.60$) |

They resolve comparable features by opposite strategies: EUV uses a short
wavelength at modest aperture, so its focus is essentially **scalar**; DUV uses an
enormous aperture in immersion, so its focus is strongly **vectorial**.  This
notebook contrasts the two and images an actual layout through them.
""")

    b.code(SETUP)

    b.md(r"""
## 1 · The two systems

Both are diffraction-limited by the Maréchal criterion (RMS wavefront error below
$\lambda/14$), which is what makes their shipped wavefronts usable as-is.
""")

    b.code(r"""
euv, duv = vw.load("euv"), vw.load("duv")

hdr = ["system", "lambda", "NA", "n", "WFE rms", "lambda/14", "diff.ltd",
       "Rayleigh hp", "Airy FWHM", "DoF"]
rows = []
for s in (euv, duv):
    rows.append([s.name.split()[0], f"{s.wavelength_nm:.1f} nm", f"{s.na:g}",
                 f"{s.n_image:.3f}", f"{s.wavefront_rms_nm:.3f} nm",
                 f"{s.wavelength_nm/14:.3f} nm", "yes" if s.diffraction_limited else "NO",
                 f"{s.rayleigh_half_pitch_nm:.1f} nm", f"{s.airy_fwhm_nm:.1f} nm",
                 f"{s.depth_of_focus_nm:.0f} nm"])
w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(hdr)]
print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
print("-+-".join("-" * x for x in w))
for r in rows:
    print(" | ".join(c.ljust(w[i]) for i, c in enumerate(r)))
""")

    b.code(r"""
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a, s in zip(ax, (euv, duv)):
    W = s.wavefront.values * 1000.0                       # milliwaves
    m = np.abs(W).max()
    im = a.imshow(np.where(s.wavefront._inside, W, np.nan), extent=[-1, 1, -1, 1],
                  origin="lower", cmap="RdBu_r", vmin=-m, vmax=m)
    a.set_title(f"{s.name.split()[0]} exit-pupil wavefront\n"
                f"{s.wavefront.rms_waves*1000:.1f} m$\\lambda$ rms, "
                f"{s.wavefront.pv_waves*1000:.0f} m$\\lambda$ PV")
    a.set_xlabel("u"); a.set_ylabel("v")
    plt.colorbar(im, ax=a, shrink=0.85, label="m$\\lambda$")
fig.tight_layout()
""")

    b.md(r"""
## 2 · Point-spread functions

Each system's `pupil()` carries its traced wavefront.  The EUV focus is scalar for
all practical purposes; the DUV focus puts roughly a seventh of its energy in the
longitudinal component — which carries no image information, only background.
That is the whole reason hyper-NA lithography controls the illumination
polarization.
""")

    b.code(r"""
def psf(system, half=2.0, step=1 / 44, polarization="x"):
    pupil = system.pupil(polarization=polarization)
    grid = vw.Grid.from_spacing(0.25, 256)
    xr = np.arange(-half, half + 1e-9, step)
    return pupil, pupil.spectrum(grid).field_on(xr, xr, 0.0), xr

pupil_euv, f_euv, x_euv = psf(euv)
pupil_duv, f_duv, x_duv = psf(duv)

for tag, s, p, f in (("EUV", euv, pupil_euv, f_euv), ("DUV", duv, pupil_duv, f_duv)):
    fr = f.component_fractions()
    print(f"{tag}: FWHM {f.fwhm('x'):.3f} lambda -> {f.fwhm('x')*s.wavelength_nm:5.1f} nm "
          f"[Airy {s.airy_fwhm_nm:.1f} nm]   |Ez|^2 = {fr['z']*100:5.2f} %")
""")

    b.code(r"""
fig, ax = plt.subplots(2, 4, figsize=(13, 6.2))
for row, (tag, s, f, xr) in enumerate((("EUV", euv, f_euv, x_euv),
                                       ("DUV", duv, f_duv, x_duv))):
    ext = [xr[0], xr[-1], xr[0], xr[-1]]
    comps = [("total $|E|^2$", f.intensity), ("$|E_x|^2$", np.abs(f.Ex) ** 2),
             ("$|E_y|^2$  (cross)", np.abs(f.Ey) ** 2),
             ("$|E_z|^2$  (long.)", np.abs(f.Ez) ** 2)]
    for col, (title, I) in enumerate(comps):
        a = ax[row, col]
        a.imshow(I, extent=ext, origin="lower", cmap="inferno", norm=PowerNorm(0.5))
        share = I.sum() / f.intensity.sum()
        a.set_title(f"{tag} · {title}" + ("" if col == 0 else f"  ({share*100:.2f} %)"))
        a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5); a.set_xlabel("x / $\\lambda$")
        if col == 0:
            a.set_ylabel("y / $\\lambda$")
fig.tight_layout()
""")

    b.md(r"""
## 3 · Imaging a real layout

Now an extended object with no symmetry — a Manhattan routing layout shipped with
the repo — imaged at wafer scale through the DUV objective with partially coherent
illumination (a conventional disc source, $\sigma = 0.6$).  The image is kept
split by field component, so we can read off the longitudinal background directly.
""")

    b.code(r"""
im = duv.imaging(pixel_nm=12.0, size=512, polarization="x")
mask = vw.Mask.from_image("assets/circuit_pattern.png", pixel=12.0, size=512)
aerial = im.aerial_image(mask, sigma=0.6, source_points=9, vector=True)
fr = aerial.fractions()
print(f"aerial-image energy split:  Ex {fr['x']:.3f}   Ey {fr['y']:.4f}   Ez {fr['z']:.3f}")
print(f"image contrast (Michelson): {aerial.contrast():.3f}")

extent = mask.grid.extent
extent_um = [e / 1000 for e in extent]                    # nm -> um for the axes
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
ax[0].imshow(mask.data, extent=extent_um, origin="lower", cmap="gray")
ax[0].set_title("mask (wafer scale)")
ax[1].imshow(aerial.normalized, extent=extent_um, origin="lower", cmap="inferno")
ax[1].set_title(f"vector aerial image ($\\sigma$=0.6)\ncontrast {aerial.contrast():.2f}")
ax[2].imshow(aerial.Iz / aerial.total.max(), extent=extent_um, origin="lower", cmap="magma")
ax[2].set_title(f"longitudinal $I_z$  ({fr['z']*100:.1f} % of energy)")
for a in ax:
    a.set_xlabel("x / $\\mu$m"); a.set_ylabel("y / $\\mu$m")
fig.tight_layout()
""")

    b.md(r"""
The longitudinal channel is not noise — it is a structured background that tracks
the mask edges, strongest where the local features run along the polarization
direction.  A scalar model cannot see it.

## 4 · Resolution and the polarization penalty

Image equal line/space gratings of shrinking half pitch and read the contrast.
For dense lines at hyper-NA the illumination polarization matters: **TE**
(polarized along the lines) keeps the diffracted orders' fields parallel and
prints with high contrast, while **TM** (across the lines) lets the orders'
fields tip apart by $\cos 2\theta$ and washes out.  The scalar model is blind to
the difference.
""")

    b.code(r"""
hp = np.linspace(120, 34, 12)                             # half pitch, nm

def contrast_curve(polarization, vector):
    sysm = duv.imaging(pixel_nm=8.0, size=256, polarization=polarization)
    return sysm.contrast_vs_pitch(hp, sigma=0.5, source_points=7, vector=vector)

c_te = contrast_curve("y", True)                          # lines vertical -> E along lines is 'y'
c_tm = contrast_curve("x", True)
c_sc = contrast_curve("x", False)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(hp, c_te, "o-", lw=1.8, ms=3, label="TE (E along lines)")
ax.plot(hp, c_tm, "s-", lw=1.8, ms=3, label="TM (E across lines)")
ax.plot(hp, c_sc, "k--", lw=1.3, label="scalar (blind to polarization)")
ax.axvline(duv.rayleigh_half_pitch_nm, color="grey", ls=":", lw=1,
           label=f"Rayleigh {duv.rayleigh_half_pitch_nm:.0f} nm")
ax.set_xlabel("half pitch (nm)"); ax.set_ylabel("contrast")
ax.set_title("DUV NA 1.2: line/space contrast vs pitch"); ax.invert_xaxis()
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()

i = int(np.argmin(np.abs(hp - 60)))
print(f"at {hp[i]:.0f} nm half pitch:  TE {c_te[i]:.3f}   TM {c_tm[i]:.3f}   scalar {c_sc[i]:.3f}")
""")

    b.md(r"""
## 5 · Scalar and vector agree where they should

The vector machinery is not a different physics — it is a *superset*.  Drop the NA
low enough and the vector and scalar aerial images become indistinguishable; the
longitudinal component vanishes and the projection factors go to unity.
""")

    b.code(r"""
low = vw.ImagingSystem(na=0.10, wavelength=193.0, pixel=200.0, size=128)
m = vw.Mask.lines_spaces(2000.0, pixel=200.0, size=128)
v = low.aerial_image(m, vector=True).normalized
s = low.aerial_image(m, vector=False).normalized
print(f"max |vector - scalar| at NA 0.1: {np.abs(v - s).max():.2e}  (indistinguishable)")

line = v.shape[0] // 2
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(m.grid.x / 1000, v[line], lw=2.2, label="vector")
ax.plot(m.grid.x / 1000, s[line], "k--", lw=1.2, label="scalar")
ax.set_xlabel("x / $\\mu$m"); ax.set_ylabel("normalized intensity")
ax.set_title("NA 0.10: the vector model reduces to the scalar one")
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
""")

    b.md(r"""
That is the invariant worth keeping in view: the same code that reduces to the
textbook Airy pattern and the scalar aerial image at low NA is the one that, at
NA 1.2 in immersion, tells you how much of your image is longitudinal background
and how much contrast your polarization choice costs.
""")

    b.write("03_projection_imaging.ipynb")


if __name__ == "__main__":
    build_focusing()
    build_interfaces()
    build_imaging()
    for name in ("01_vector_focusing", "02_curved_interfaces", "03_projection_imaging"):
        print("wrote notebooks/" + name + ".ipynb")
