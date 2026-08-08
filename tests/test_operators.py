"""Tests for the composable operator algebra, the general (NUFFT) surface
transform, and freeform 2D surfaces (vectorwave 0.2)."""
import numpy as np
import pytest

import vectorwave as vw


def _corr(a, b):
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


@pytest.fixture(scope="module")
def grid():
    return vw.Grid.from_spacing(0.28, 200)


# --------------------------------------------------------------- FreeSpace
def test_freespace_is_exact(grid):
    spec = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0)
    # a tilted-ish real spectrum: refract a plane wave first
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    s = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, n_rho=300)(
        vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5))
    x = np.linspace(-3, 3, 61)
    D = 4.0
    lhs = vw.FreeSpace(D)(s).field_on(x, x, 0.0).intensity
    rhs = s.field_on(x, x, D).intensity
    assert np.abs(lhs - rhs).max() / rhs.max() < 1e-9


def test_freespace_group_law(grid):
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    s = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, n_rho=300)(
        vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5))
    x = np.linspace(-3, 3, 61)
    g1 = vw.FreeSpace(1.5)(vw.FreeSpace(2.5)(s)).field_on(x, x, 0.0).intensity
    g2 = vw.FreeSpace(4.0)(s).field_on(x, x, 0.0).intensity
    assert np.abs(g1 - g2).max() / g2.max() < 1e-9


# ------------------------------------------- interface operator == source path
def test_interface_operator_matches_source(grid):
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    ref = vw.surface_spectrum(surf, grid, n1=1.5, n2=1.0, aperture=9.0,
                              polarization="x")
    op = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, n_rho=600)
    out = op(vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5))
    zs = np.linspace(8, 22, 43)
    assert abs(ref.best_focus(zs) - out.best_focus(zs)) < 0.5
    x = np.linspace(-3, 3, 81)
    zf = ref.best_focus(zs)
    assert _corr(ref.field_on(x, x, zf).intensity,
                 out.field_on(x, x, zf).intensity) > 0.99


def test_interface_freespace_shifts_focus(grid):
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    out = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0)(
        vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5))
    zs = np.linspace(8, 22, 57)
    zf = out.best_focus(zs)
    D = 4.0
    zfD = vw.FreeSpace(D)(out).best_focus(zs - D)
    assert abs(zfD - (zf - D)) < 0.4


# --------------------------------------------------- NUFFT == polar (axisym)
def test_nufft_matches_polar(grid):
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    pw = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5)
    a = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, method="polar")(pw)
    b = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, method="nufft")(pw)
    zs = np.linspace(8, 22, 43)
    za = a.best_focus(zs)
    assert abs(za - b.best_focus(zs)) < 0.5
    x = np.linspace(-3, 3, 81)
    assert _corr(a.field_on(x, x, za).intensity,
                 b.field_on(x, x, za).intensity) > 0.99


# ------------------------------------------------------- Freeform2D == polar
def test_freeform2d_matches_axisym(grid):
    conic = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    ff = vw.Freeform2D(sag_fn=lambda x, y: conic.sag(np.hypot(x, y)), radius=9.0)
    pw = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5)
    a = vw.InterfaceOperator(conic, n1=1.5, n2=1.0, aperture=9.0, method="polar")(pw)
    b = vw.InterfaceOperator(ff, n1=1.5, n2=1.0, aperture=9.0, n_free=220)(pw)
    zs = np.linspace(8, 22, 43)
    za = a.best_focus(zs)
    assert abs(za - b.best_focus(zs)) < 0.6
    x = np.linspace(-3, 3, 81)
    assert _corr(a.field_on(x, x, za).intensity,
                 b.field_on(x, x, za).intensity) > 0.99


# ------------------------------------------------------------- composition
def test_two_surface_system_focuses(grid):
    n_g = 1.5
    front = vw.InterfaceOperator(
        vw.Conic(radius=+6.0, conic=vw.stigmatic_conic_constant(1.0, n_g)),
        n1=1.0, n2=n_g, aperture=5.5)
    back = vw.InterfaceOperator(vw.Plane(), n1=n_g, n2=1.0, aperture=5.5)
    system = vw.System([front, vw.FreeSpace(8.0), back])
    assert len(system) == 3
    out = system(vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0))
    z = np.linspace(2, 40, 120)
    k = int(np.argmax(out.focus_scan(z)))
    assert 0 < k < len(z) - 1                      # an interior focus
    assert out.transversality_residual() < 1e-6


def test_plane_wave_spectrum_polarization(grid):
    f = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0,
                               polarization="x").field(0.0)
    fr = f.component_fractions()
    assert fr["x"] > 0.999 and fr["y"] < 1e-6
