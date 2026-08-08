"""Projection imaging: masks, pupils, and scalar or vector aerial images.

The scalar path is the classical Hopkins/Abbe model.  The vector path uses the
same aplanatic projection as :mod:`vectorwave.pupil`, so at high NA the aerial
image carries the longitudinal and cross-polarized components that the scalar
model cannot represent — the effect behind polarized illumination in
hyper-NA lithography.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import Field
from .grids import Grid

__all__ = ["Mask", "ImagingSystem", "AerialImage"]


# --------------------------------------------------------------------- masks
@dataclass
class Mask:
    """Binary (or grey) amplitude mask on a square pixel grid, at wafer scale."""

    data: np.ndarray
    pixel: float

    @property
    def size(self) -> int:
        return int(self.data.shape[0])

    @property
    def grid(self) -> Grid:
        return Grid.from_spacing(self.pixel, self.size)

    @classmethod
    def from_image(cls, path, *, pixel: float, size: int = 512,
                   threshold: float = 0.5, invert: bool = False) -> "Mask":
        """Load an image file, grey it, resample to ``size`` and binarize."""
        import matplotlib.image as mpimg
        from scipy import ndimage
        raw = mpimg.imread(path)
        if raw.ndim == 3:
            raw = raw[..., :3].mean(axis=2)
        raw = np.asarray(raw, dtype=float)
        if raw.max() > 1.0:
            raw = raw / 255.0
        zoom = (size / raw.shape[0], size / raw.shape[1])
        img = ndimage.zoom(raw, zoom, order=1)[:size, :size]
        out = np.zeros((size, size))
        out[: img.shape[0], : img.shape[1]] = img
        binary = (out < threshold) if invert else (out > threshold)
        return cls(np.flipud(binary.astype(float)), float(pixel))

    @classmethod
    def lines_spaces(cls, half_pitch: float, *, pixel: float, size: int = 512,
                     vertical: bool = True) -> "Mask":
        """Equal line/space grating of the given half pitch."""
        ax = (np.arange(size) - size // 2) * pixel
        line = (np.mod(ax, 2 * half_pitch) < half_pitch).astype(float)
        data = np.tile(line, (size, 1))
        if not vertical:
            data = data.T
        return cls(data, float(pixel))

    @classmethod
    def contacts(cls, pitch: float, width: float, *, pixel: float,
                 size: int = 512) -> "Mask":
        """Square array of contact holes."""
        ax = (np.arange(size) - size // 2) * pixel
        X, Y = np.meshgrid(ax, ax)
        inside = ((np.abs(np.mod(X + pitch / 2, pitch) - pitch / 2) < width / 2) &
                  (np.abs(np.mod(Y + pitch / 2, pitch) - pitch / 2) < width / 2))
        return cls(inside.astype(float), float(pixel))


# ------------------------------------------------------------- aerial images
@dataclass
class AerialImage:
    """Intensity of an imaged mask, kept split by field component."""

    Ix: np.ndarray
    Iy: np.ndarray
    Iz: np.ndarray
    grid: Grid
    vector: bool = True

    @property
    def total(self) -> np.ndarray:
        return self.Ix + self.Iy + self.Iz

    @property
    def normalized(self) -> np.ndarray:
        t = self.total
        return t / t.max() if t.max() > 0 else t

    def fractions(self) -> dict[str, float]:
        tot = self.total.sum()
        if tot == 0:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        return {"x": float(self.Ix.sum() / tot),
                "y": float(self.Iy.sum() / tot),
                "z": float(self.Iz.sum() / tot)}

    def contrast(self) -> float:
        """Michelson contrast of the (normalized) image."""
        t = self.normalized
        hi, lo = np.percentile(t, 99.0), np.percentile(t, 1.0)
        return float((hi - lo) / (hi + lo)) if (hi + lo) > 0 else 0.0


@dataclass
class ImagingSystem:
    """A projection pupil that images masks.

    ``na`` and ``n`` are image-side; ``wavelength`` is in vacuum; ``pixel`` is
    the wafer-plane sampling that the mask is given on.
    """

    na: float
    wavelength: float
    n: float = 1.0
    pixel: float = 1.0
    size: int = 512
    wavefront: object | None = None
    polarization: str | object = "x"
    apodization: str = "aplanatic"
    #: Optional interface transfer, as the three eigenvalues of the pupil
    #: operator in the local ``(radial, azimuthal, longitudinal)`` frame:
    #: ``f(rho, phi, inside) -> (lam_r, lam_phi, lam_z)``.  ``None`` uses the
    #: ideal aplanatic values ``(cos theta, 1, sin theta)``.  Supplying the real
    #: Fresnel/flux eigenvalues of a refracting surface is what turns an ideal
    #: lens into a physical diopter.
    transfer: object | None = None

    # ------------------------------------------------------------- pupil setup
    def _pupil(self):
        f = np.fft.fftfreq(self.size, d=self.pixel)
        FX, FY = np.meshgrid(f, f)
        rho = np.hypot(FX, FY) * self.wavelength / self.na
        phi = np.arctan2(FY, FX)
        inside = rho <= 1.0
        P = inside.astype(complex)
        if self.wavefront is not None:
            u, v = rho * np.cos(phi), rho * np.sin(phi)
            with np.errstate(invalid="ignore"):
                W = np.asarray(self.wavefront(np.where(inside, u, 0.0),
                                              np.where(inside, v, 0.0)), dtype=float)
            P = P * np.exp(2j * np.pi * np.nan_to_num(W))
        return P, rho, phi, inside

    def _projection(self, rho, phi, inside):
        """Aplanatic polarization factors ``(px, py, pz)`` on the pupil."""
        from .pupil import POLARIZATIONS
        sin_t = np.clip(rho * (self.na / self.n), 0.0, 0.999999)
        cos_t = np.sqrt(1.0 - sin_t**2)
        pol = self.polarization if callable(self.polarization) else POLARIZATIONS[self.polarization]
        ex0, ey0 = pol(rho, phi)
        ex0 = np.asarray(ex0, dtype=complex)
        ey0 = np.asarray(ey0, dtype=complex)
        a_p = ex0 * np.cos(phi) + ey0 * np.sin(phi)      # meridional (radial)
        a_s = -ex0 * np.sin(phi) + ey0 * np.cos(phi)     # azimuthal
        if self.transfer is None:
            lam_r, lam_phi, lam_z = cos_t, np.ones_like(cos_t), sin_t
            apod = np.sqrt(cos_t) if self.apodization == "aplanatic" else np.ones_like(cos_t)
        else:
            lam_r, lam_phi, lam_z = self.transfer(rho, phi, inside)
            apod = np.ones_like(cos_t)                   # the transfer carries its own flux factor
        Er, Ephi = a_p * lam_r, a_s * lam_phi
        px = Er * np.cos(phi) - Ephi * np.sin(phi)
        py = Er * np.sin(phi) + Ephi * np.cos(phi)
        pz = -a_p * lam_z
        return (px * apod * inside, py * apod * inside, pz * apod * inside)

    # ---------------------------------------------------------------- imaging
    def coherent_field(self, mask: Mask, vector: bool = True) -> Field:
        """Coherent (on-axis illumination) image as a vector :class:`Field`."""
        P, rho, phi, inside = self._pupil()
        M = np.fft.fft2(np.asarray(mask.data, dtype=float))
        if vector:
            px, py, pz = self._projection(rho, phi, inside)
        else:
            px, py, pz = inside.astype(complex), np.zeros_like(P), np.zeros_like(P)
        Ex = np.fft.ifft2(M * P * px)
        Ey = np.fft.ifft2(M * P * py)
        Ez = np.fft.ifft2(M * P * pz)
        return Field(Ex, Ey, mask.grid, self.wavelength, self.n, Ez=Ez)

    def aerial_image(self, mask: Mask, *, sigma: float = 0.0,
                     source_points: int = 11, vector: bool = True) -> AerialImage:
        """Partially coherent (Abbe) aerial image.

        ``sigma`` is the illumination coherence factor; ``sigma = 0`` reduces to
        the coherent case.  A conventional disc source is integrated by shifting
        the pupil in frequency.
        """
        P, rho, phi, inside = self._pupil()
        if vector:
            comps = self._projection(rho, phi, inside)
        else:
            comps = (inside.astype(complex), np.zeros_like(P), np.zeros_like(P))
        M = np.fft.fft2(np.asarray(mask.data, dtype=float))
        f = np.fft.fftfreq(self.size, d=self.pixel)
        df = f[1] - f[0]
        acc = [np.zeros(mask.data.shape) for _ in range(3)]

        if sigma <= 0:
            offsets = [(0.0, 0.0)]
        else:
            g = np.linspace(-sigma, sigma, int(source_points))
            offsets = [(sx, sy) for sx in g for sy in g if sx * sx + sy * sy <= sigma * sigma]

        for sx, sy in offsets:
            shx = int(round(sx * self.na / self.wavelength / df))
            shy = int(round(sy * self.na / self.wavelength / df))
            Ms = np.roll(np.roll(M, -shy, axis=0), -shx, axis=1)
            for c in range(3):
                acc[c] += np.abs(np.fft.ifft2(Ms * P * comps[c])) ** 2
        return AerialImage(acc[0], acc[1], acc[2], mask.grid, vector)

    # ------------------------------------------------------------- resolution
    def contrast_vs_pitch(self, half_pitches, *, sigma: float = 0.6,
                          source_points: int = 9, vector: bool = True) -> np.ndarray:
        """Image contrast of equal line/space gratings versus half pitch."""
        out = []
        for hp in np.asarray(half_pitches, dtype=float):
            m = Mask.lines_spaces(hp, pixel=self.pixel, size=self.size)
            img = self.aerial_image(m, sigma=sigma, source_points=source_points,
                                    vector=vector).total
            line = img[img.shape[0] // 2]
            hi, lo = line.max(), line.min()
            out.append((hi - lo) / (hi + lo) if (hi + lo) > 0 else 0.0)
        return np.array(out)

    @property
    def rayleigh_half_pitch(self) -> float:
        """``0.5 lambda / NA`` — the classical resolution scale."""
        return 0.5 * self.wavelength / self.na

    @property
    def coherent_cutoff_half_pitch(self) -> float:
        """``0.25 lambda / NA`` — the hard coherent diffraction cutoff."""
        return 0.25 * self.wavelength / self.na
