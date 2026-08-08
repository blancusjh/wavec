"""Physics and API tests.

The assertions are deliberately physical: transversality, energy, known
analytic limits (Airy, scalar low-NA), and the stigmatic focus condition.
A refactor that breaks the optics will fail these; a cosmetic one will not.
"""

import numpy as np
import pytest

import vectorwave as vw


# ------------------------------------------------------------------- grids
def test_grid_geometry():
    g = vw.Grid.from_spacing(0.25, 64)
    assert g.shape == (64, 64)
    assert g.dx == pytest.approx(0.25)
    assert g.KR.shape == g.shape
    k = 2 * np.pi
    assert g.propagating(k).any()
    assert np.all(g.kz(k)[~g.propagating(k)] == 0)


# ---------------------------------------------------------------- surfaces
def test_conic_sag_and_normal():
    s = vw.Conic(radius=10.0, conic=0.0)          # sphere
    rho = np.array([0.0, 1.0, 3.0])
    # sphere sag: R - sqrt(R^2 - rho^2)
    assert np.allclose(s.sag(rho), 10.0 - np.sqrt(100.0 - rho**2))
    n = s.normal(np.array([3.0]))
    assert np.allclose(np.hypot(n[0], n[1]), 1.0)
    assert n[0] > 0 and n[1] > 0                   # outward: +rho and +z


def test_sphere_bounded_hyperboloid_unbounded():
    assert np.isfinite(vw.Conic(10.0, 0.0).max_radius)
    assert not np.isfinite(vw.Conic(10.0, -2.25).max_radius)


def test_even_asphere_reduces_to_conic():
    base = vw.Conic(20.0, -1.0)
    asph = vw.EvenAsphere(20.0, -1.0, coefficients=())
    rho = np.linspace(0, 5, 11)
    assert np.allclose(base.sag(rho), asph.sag(rho))
    assert np.allclose(base.dsag(rho), asph.dsag(rho))


def test_freeform_matches_its_callable():
    f = vw.Freeform(sag_fn=lambda r: 0.01 * r**2)
    rho = np.linspace(0.5, 4, 7)
    assert np.allclose(f.sag(rho), 0.01 * rho**2)
    assert np.allclose(f.dsag(rho), 0.02 * rho, rtol=1e-5)


def test_stigmatic_constant_sign():
    assert vw.stigmatic_conic_constant(1.5, 1.0) == pytest.approx(-2.25)
    assert vw.stigmatic_conic_constant(1.0, 1.5) == pytest.approx(-4 / 9)


# --------------------------------------------------------------- interfaces
def test_fresnel_normal_incidence():
    khat = np.array([[0.0], [0.0], [1.0]])
    nhat = np.array([[0.0], [0.0], [1.0]])
    c = vw.fresnel(khat, nhat, 1.0, 1.5)
    assert c.ts[0] == pytest.approx(2 * 1.0 / (1.0 + 1.5))
    assert c.rs[0] == pytest.approx((1.0 - 1.5) / (1.0 + 1.5))


def test_total_internal_reflection_onset():
    theta_c = vw.critical_angle(1.5, 1.0)
    assert theta_c == pytest.approx(np.arcsin(1 / 1.5))
    for th, expect_tir in ((theta_c - 0.05, False), (theta_c + 0.05, True)):
        khat = np.array([[np.sin(th)], [0.0], [np.cos(th)]])
        nhat = np.array([[0.0], [0.0], [1.0]])
        assert bool(vw.fresnel(khat, nhat, 1.5, 1.0).tir[0]) is expect_tir


def test_transmitted_field_is_transverse():
    rng = np.random.default_rng(0)
    th = np.linspace(0.05, 0.6, 12)
    khat = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)])
    nhat = np.tile(np.array([[0.0], [0.0], [1.0]]), (1, th.size))
    E = np.zeros((3, th.size), dtype=complex)
    E[0] = np.cos(th)            # transverse to khat in the x-z plane
    E[2] = -np.sin(th)
    E[1] = rng.normal(size=th.size)
    Et, kt, _ = vw.transmit_field(E, khat, nhat, 1.0, 1.5)
    assert np.abs(np.sum(Et * kt, axis=0)).max() < 1e-12


# ----------------------------------------------------------------- spectrum
def test_pupil_spectrum_is_transverse():
    g = vw.Grid.from_spacing(0.25, 96)
    spec = vw.Pupil(na=0.95, n=1.0, wavelength=1.0).spectrum(g)
    assert spec.transversality_residual() < 1e-12


def test_propagation_round_trip():
    g = vw.Grid.from_spacing(0.25, 96)
    spec = vw.Pupil(na=0.6, n=1.0, wavelength=1.0).spectrum(g)
    f0 = spec.field(0.0)
    there_and_back = vw.propagate(vw.propagate(f0, 3.0), -3.0)
    num = np.linalg.norm(there_and_back.Ex - f0.Ex)
    assert num / np.linalg.norm(f0.Ex) < 1e-8


def test_zoom_synthesis_matches_grid_synthesis():
    g = vw.Grid.from_spacing(0.25, 96)
    spec = vw.Pupil(na=0.8, n=1.0, wavelength=1.0).spectrum(g)
    on_grid = spec.field(0.0)
    zoomed = spec.field_on(g.x, g.y, 0.0)
    assert np.allclose(on_grid.Ex, zoomed.Ex, atol=1e-10 * np.abs(on_grid.Ex).max())


# -------------------------------------------------------------------- pupil
def test_longitudinal_energy_follows_low_na_law():
    """For an aplanatic x-polarized pupil the longitudinal share tends to
    ``NA^2 / 4 n^2`` as the aperture closes — the scalar limit, quantitatively."""
    g = vw.Grid.from_spacing(0.25, 512)
    fz = []
    for na in (0.1, 0.2, 0.4):
        f = vw.Pupil(na=na, n=1.0, wavelength=1.0).spectrum(g).field(0.0)
        fz.append(f.component_fractions()["z"])
    for na, got in zip((0.1, 0.2, 0.4), fz):
        assert got == pytest.approx(na**2 / 4, rel=0.10)
    assert all(b > a for a, b in zip(fz, fz[1:]))   # and grows with NA


def test_focal_spot_matches_airy_width():
    g = vw.Grid.from_spacing(0.2, 192)
    p = vw.Pupil(na=0.5, n=1.0, wavelength=1.0)
    x = np.arange(-4, 4, 1 / 32)
    f = p.spectrum(g).field_on(x, x, 0.0)
    assert f.fwhm("x") == pytest.approx(p.airy_fwhm(), rel=0.10)


def test_polarization_states():
    g = vw.Grid.from_spacing(0.25, 96)
    x = np.arange(-2, 2, 1 / 16)
    circ = vw.Pupil(na=0.7, polarization="circular").spectrum(g).field_on(x, x, 0.0)
    pol = circ.polarization()
    centre = tuple(s // 2 for s in circ.intensity.shape)
    assert abs(pol.ellipticity[centre]) > 0.6           # near-circular on axis
    radial = vw.Pupil(na=0.9, polarization="radial").spectrum(g).field_on(x, x, 0.0)
    # radial polarization concentrates energy in the longitudinal component
    assert radial.component_fractions()["z"] > 0.3


def test_vortex_has_dark_core():
    g = vw.Grid.from_spacing(0.25, 128)
    x = np.arange(-3, 3, 1 / 16)
    f = vw.Pupil(na=0.6, polarization="vortex1").spectrum(g).field_on(x, x, 0.0)
    I = f.intensity
    c = tuple(s // 2 for s in I.shape)
    assert I[c] < 0.2 * I.max()


# ------------------------------------------------------- surface propagation
def test_stigmatic_hyperboloid_focuses_where_predicted():
    """Regression against the MEEP-validated reference.

    A stigmatic hyperboloid (kappa = -(n1/n2)^2) with R_v = 8 lambda focuses a
    plane wave at 14.5 lambda from the vertex — short of the 16 lambda paraxial
    prediction because the Fresnel number is finite.  Independent full-wave FDTD
    put this focus at 14.5-14.6 lambda.
    """
    n1, n2, Rv = 1.5, 1.0, 8.0
    surf = vw.Conic(radius=-Rv, conic=vw.stigmatic_conic_constant(n1, n2))
    g = vw.Grid.from_spacing(0.25, 192)
    spec = vw.surface_spectrum(surf, g, n1=n1, n2=n2, wavelength=1.0,
                               aperture=0.62 * 2 * Rv, m_max=2, n_rho=700,
                               n_phi=32, n_kr=256)
    z_paraxial = n2 * Rv / (n1 - n2)                     # = 16 lambda
    zf = spec.best_focus(np.linspace(0.5 * z_paraxial, 1.4 * z_paraxial, 73))
    assert zf == pytest.approx(14.5, abs=0.6)
    assert zf < z_paraxial                               # focal shift toward the lens


def test_surface_spectrum_transversality():
    surf = vw.Conic(radius=-10.0, conic=-2.25)
    g = vw.Grid.from_spacing(0.25, 128)
    spec = vw.surface_spectrum(surf, g, n1=1.5, n2=1.0, aperture=8.0,
                               m_max=2, n_rho=400, n_phi=32, n_kr=192)
    assert spec.transversality_residual() < 1e-9


# ------------------------------------------------------------------ imaging
def test_uniform_mask_images_flat():
    sysm = vw.ImagingSystem(na=0.5, wavelength=193.0, pixel=20.0, size=64)
    mask = vw.Mask(np.ones((64, 64)), 20.0)
    img = sysm.aerial_image(mask, vector=False).total
    assert img.std() / img.mean() < 1e-6


def test_contrast_falls_with_pitch():
    sysm = vw.ImagingSystem(na=1.2, wavelength=193.0, n=1.6, pixel=8.0, size=128)
    hp = np.array([120.0, 80.0, 50.0, 35.0])
    c = sysm.contrast_vs_pitch(hp, sigma=0.5, source_points=5, vector=False)
    assert c[0] > c[-1]
    assert c[0] > 0.5


def test_vector_imaging_produces_longitudinal_component():
    sysm = vw.ImagingSystem(na=1.2, wavelength=193.0, n=1.6, pixel=10.0, size=128)
    mask = vw.Mask.lines_spaces(100.0, pixel=10.0, size=128)
    img = sysm.aerial_image(mask, sigma=0.0, vector=True)
    fr = img.fractions()
    assert fr["z"] > 0.02                       # real longitudinal content at NA 1.2
    assert fr["x"] > fr["z"]


def test_scalar_and_vector_agree_at_low_na():
    sysm = vw.ImagingSystem(na=0.1, wavelength=193.0, pixel=200.0, size=64)
    mask = vw.Mask.lines_spaces(2000.0, pixel=200.0, size=64)
    v = sysm.aerial_image(mask, vector=True).normalized
    s = sysm.aerial_image(mask, vector=False).normalized
    assert np.abs(v - s).max() < 0.02


# ------------------------------------------------------------------ systems
@pytest.mark.parametrize("name", ["euv", "duv"])
def test_packaged_systems_load_and_are_diffraction_limited(name):
    s = vw.load(name)
    assert s.na > 0 and s.wavelength_nm > 0
    assert s.wavefront is not None
    assert s.diffraction_limited
    assert s.rayleigh_half_pitch_nm > 0


def test_system_builds_pupil_and_imaging():
    s = vw.load("duv")
    p = s.pupil(polarization="x")
    assert p.na == s.na
    im = s.imaging(pixel_nm=12.0, size=64)
    assert im.size == 64 and im.wavelength == pytest.approx(s.wavelength_nm)
