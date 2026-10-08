#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seestar Double Star Analyzer Basic v0.6

Az asztrometriai mérési mag közvetlenül az IsoTool PhotoAnalyzer v1.3 bevált
kétkomponensű PSF-illesztéséből származik. A Delta mag külön apertúrás
fotometriából készül a stacked FIT-en.

Basic:
- 1 Seestar stacked FIT
- 1 megadott WDS/discoverer pár
- sep >= 10"
- Mag1, Mag2 <= 14.0
- kimenet: PA, Separation, abs(Delta mag)
"""

from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

from photutils.detection import DAOStarFinder


APP_NAME = "Seestar Double Star Analyzer Basic"
VERSION = "1.1a"

MIN_BASIC_SEPARATION_ARCSEC = 10.0
MAX_BASIC_MAG = 14.0
WIDE_PAIR_THRESHOLD_ARCSEC = 30.0
MAX_WDS_DMAG_DIFFERENCE = 3.0

# EREDETI PHOTOANALYZER MÉRÉSI KONSTANSOK
MIN_MEASURABLE_SEPARATION_ARCSEC = 1.0
MIN_CUTOUT_HALF_SIZE = 14
MAX_POSITION_SHIFT_PIX = 3.0

FITS_EXTENSIONS = {".fit", ".fits", ".fts"}


@dataclass(frozen=True)
class WDSRow:
    wds_id: str
    disc: str
    comp: str
    pa_last: float
    sep_last: float
    mag_a: Optional[float]
    mag_b: Optional[float]
    coord: SkyCoord

    @property
    def pair_name(self) -> str:
        return f"{self.disc} {self.comp}".strip()


def safe_float(value):
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def normalize_designation(text: str) -> str:
    return re.sub(r"\s+", "", str(text).upper().strip())


def load_basic_catalog(path: Path) -> list[WDSRow]:
    if not path.exists():
        raise FileNotFoundError(f"Katalógus nem található: {path}")

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "WDS", "Discoverer", "Components",
            "PA_Last", "Sep_Last", "Mag1", "Mag2",
            "RA_deg", "Dec_deg"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Hiányzó katalógusoszlop(ok): " + ", ".join(sorted(missing)))

        for rec in reader:
            pa = safe_float(rec.get("PA_Last"))
            sep = safe_float(rec.get("Sep_Last"))
            m1 = safe_float(rec.get("Mag1"))
            m2 = safe_float(rec.get("Mag2"))
            ra = safe_float(rec.get("RA_deg"))
            dec = safe_float(rec.get("Dec_deg"))
            if None in (pa, sep, m1, m2, ra, dec):
                continue

            # A katalogus eleve szűrt, de a program is védi a Basic határokat.
            if sep < MIN_BASIC_SEPARATION_ARCSEC or m1 > MAX_BASIC_MAG or m2 > MAX_BASIC_MAG:
                continue

            rows.append(
                WDSRow(
                    wds_id=str(rec.get("WDS", "")).strip(),
                    disc=str(rec.get("Discoverer", "")).strip(),
                    comp=str(rec.get("Components", "")).strip(),
                    pa_last=float(pa),
                    sep_last=float(sep),
                    mag_a=float(m1),
                    mag_b=float(m2),
                    coord=SkyCoord(float(ra)*u.deg, float(dec)*u.deg, frame="icrs"),
                )
            )
    return rows


def find_matches(rows: list[WDSRow], query: str) -> list[WDSRow]:
    q = normalize_designation(query)
    out = []
    for row in rows:
        keys = {
            normalize_designation(row.wds_id),
            normalize_designation(row.disc),
            normalize_designation(row.disc + row.comp),
            normalize_designation(row.disc + " " + row.comp),
        }
        if q in keys:
            out.append(row)
    return out


def choose_match(matches: list[WDSRow]) -> WDSRow:
    if not matches:
        raise LookupError("Nincs ilyen kettős a Seestar katalógusban.")

    if len(matches) == 1:
        return matches[0]

    print("\nTöbb találat van. Válassz sorszámot:")
    for i, row in enumerate(matches, 1):
        comp = row.comp or "-"
        print(
            f"  {i}. {row.disc} {comp} | WDS {row.wds_id} | "
            f"PA={row.pa_last:.0f}° | sep={row.sep_last:.1f}\" | "
            f"{row.mag_a:.2f}/{row.mag_b:.2f} mag"
        )
    while True:
        try:
            n = int(input("Sorszám: ").strip())
            if 1 <= n <= len(matches):
                return matches[n-1]
        except Exception:
            pass
        print("Érvénytelen választás.")


def observation_time_text(header: fits.Header) -> str:
    date_obs = header.get("DATE-OBS") or header.get("DATE")
    if not date_obs:
        return "unknown"
    try:
        start = Time(str(date_obs), scale="utc")
        exptime = safe_float(header.get("EXPTIME")) or 0.0
        mid = start + TimeDelta(exptime / 2.0, format="sec")
        return mid.utc.isot.replace("T", " ")
    except Exception:
        return str(date_obs)


# ---------------------------------------------------------------------------
# EREDETI PHOTOANALYZER v1.3 MÉRÉSI MAG
# Az asztrometria változatlan. A végső Delta magot külön apertúrás fotometria adja.
# ---------------------------------------------------------------------------

def normalize_image_data(data: np.ndarray) -> np.ndarray:
    """A FITS-adatot mérésre alkalmas, kétdimenziós float tömbbé alakítja."""
    arr = np.asarray(data)

    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3:
        # Többcsatornás adat esetén robusztus medián-kombináció.
        arr = np.nanmedian(arr.astype(float), axis=0)

    if arr.ndim != 2:
        raise ValueError(f"Nem támogatott FITS dimenzió: {arr.shape}")

    arr = np.asarray(arr, dtype=float)
    if not np.any(np.isfinite(arr)):
        raise ValueError("A FITS-kép nem tartalmaz véges pixelértékeket.")

    return arr


def estimate_fwhm_and_sources(data: np.ndarray) -> tuple[float, object]:
    """
    Automatikus forrásdetektálás több FWHM-próbával.

    A DAO itt csak elődetektálásra és kezdőpozíciókhoz szolgál.
    A végleges mérés nem a DAO nyers centroidjaiból készül.
    """
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Nem becsülhető a kép háttérzaja.")

    best_sources = None
    best_count = -1
    best_fwhm = 3.0

    # A képből automatikusan választunk használható kezdő FWHM-et.
    for fwhm in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        finder = DAOStarFinder(
            threshold=6.0 * std,
            fwhm=fwhm,
            exclude_border=True,
        )
        sources = finder(data - median)
        count = 0 if sources is None else len(sources)
        if count > best_count:
            best_sources = sources
            best_count = count
            best_fwhm = fwhm

    if best_sources is None or len(best_sources) == 0:
        raise RuntimeError("DAOStarFinder nem talált használható csillagokat.")

    # A sharpness alapján csak ésszerű forrásokat tartunk meg.
    if "sharpness" in best_sources.colnames:
        good = (
            np.isfinite(best_sources["sharpness"])
            & (best_sources["sharpness"] > 0.1)
            & (best_sources["sharpness"] < 1.5)
        )
        if np.any(good):
            best_sources = best_sources[good]

    return float(best_fwhm), best_sources


def gaussian_pair_model(
    params: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """
    Két elliptikus, közös alakú Gauss-profil + konstans háttér.

    Paraméterek:
    x1, y1, amp1, x2, y2, amp2, sigma_x, sigma_y, theta_rad, background
    """
    x1, y1, a1, x2, y2, a2, sx, sy, theta, background = params

    ct = np.cos(theta)
    st = np.sin(theta)

    def component(x0: float, y0: float, amp: float) -> np.ndarray:
        dx = xx - x0
        dy = yy - y0
        xr = ct * dx + st * dy
        yr = -st * dx + ct * dy
        return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

    return background + component(x1, y1, a1) + component(x2, y2, a2)


def nearest_dao_position(
    sources: object,
    expected_x: float,
    expected_y: float,
    max_distance: float,
) -> Optional[tuple[float, float]]:
    if sources is None or len(sources) == 0:
        return None

    xs = np.asarray(sources["xcentroid"], dtype=float)
    ys = np.asarray(sources["ycentroid"], dtype=float)
    dist = np.hypot(xs - expected_x, ys - expected_y)
    idx = int(np.nanargmin(dist))

    if dist[idx] <= max_distance:
        return float(xs[idx]), float(ys[idx])
    return None



def require_two_independent_sources(
    sources,
    x_a_exp: float,
    y_a_exp: float,
    x_b_exp: float,
    y_b_exp: float,
    fwhm_guess: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Basic biztonsági ellenőrzés:
    csak akkor engedjük a mérést, ha A és B várható helyén
    két külön DAOStarFinder-forrás ténylegesen detektálható.
    """
    search_radius = max(2.5 * float(fwhm_guess), 5.0)

    dao_a = nearest_dao_position(sources, x_a_exp, y_a_exp, search_radius)
    dao_b = nearest_dao_position(sources, x_b_exp, y_b_exp, search_radius)

    if dao_a is None:
        raise ValueError(
            "Az A komponens helyén nincs önállóan detektálható csillagforrás."
        )

    if dao_b is None:
        raise ValueError(
            "A B komponens helyén nincs önállóan detektálható csillagforrás."
        )

    measured_sep_pix = math.hypot(
        dao_a[0] - dao_b[0],
        dao_a[1] - dao_b[1],
    )

    min_independent_sep = max(1.5, 0.75 * float(fwhm_guess))
    if measured_sep_pix < min_independent_sep:
        raise ValueError(
            "A két komponens nem különül el önálló csillagforrásként a stacked FIT-en."
        )

    expected_sep_pix = math.hypot(
        x_a_exp - x_b_exp,
        y_a_exp - y_b_exp,
    )

    if expected_sep_pix > 0:
        ratio = measured_sep_pix / expected_sep_pix
        if not (0.70 <= ratio <= 1.30):
            raise ValueError(
                "A detektált két csillag távolsága túlzottan eltér "
                "a WDS alapján várt komponens-távolságtól."
            )

    return dao_a, dao_b



def fit_pair_on_frame(
    data: np.ndarray,
    wcs: WCS,
    row: WDSRow,
    sources: object,
    fwhm_guess: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Egy WDS-pár kétkomponensű PSF-illesztése egyetlen képen.

    Visszatérés:
    pa, rho, dmag, xA, yA, xB, yB, fit_rms, fitted_fwhm_pix,
    raA, decA, raB, decB
    """
    if row.sep_last < MIN_MEASURABLE_SEPARATION_ARCSEC:
        raise ValueError("A pár az 1 ívmásodperces mérési határ alatt van.")

    expected_a = row.coord
    expected_b = expected_a.directional_offset_by(
        position_angle=row.pa_last * u.deg,
        separation=row.sep_last * u.arcsec,
    )

    x_a_exp, y_a_exp = skycoord_to_pixel(expected_a, wcs, origin=0, mode="all")
    x_b_exp, y_b_exp = skycoord_to_pixel(expected_b, wcs, origin=0, mode="all")

    if not all(np.isfinite([x_a_exp, y_a_exp, x_b_exp, y_b_exp])):
        raise ValueError("A várható komponenspozíció nem alakítható pixelkoordinátává.")

    # Basic módban csak akkor mérünk, ha mindkét komponens
    # külön, önálló csillagforrásként ténylegesen detektálható.
    dao_a, dao_b = require_two_independent_sources(
        sources=sources,
        x_a_exp=x_a_exp,
        y_a_exp=y_a_exp,
        x_b_exp=x_b_exp,
        y_b_exp=y_b_exp,
        fwhm_guess=fwhm_guess,
    )

    x_a0, y_a0 = dao_a
    x_b0, y_b0 = dao_b

    expected_sep_pix = math.hypot(x_b_exp - x_a_exp, y_b_exp - y_a_exp)
    half = max(
        MIN_CUTOUT_HALF_SIZE,
        int(math.ceil(expected_sep_pix + 4.0 * fwhm_guess)),
    )

    xc = 0.5 * (x_a0 + x_b0)
    yc = 0.5 * (y_a0 + y_b0)

    ny, nx = data.shape

    # A teljes PSF-illesztési kivágásnak a képen belül kell maradnia.
    # Nem vágjuk le automatikusan a képszélen, mert abból hamis centroid
    # és értelmetlen asztrometriai eredmény születhetne.
    x_min_raw = int(math.floor(xc - half))
    x_max_raw = int(math.ceil(xc + half + 1))
    y_min_raw = int(math.floor(yc - half))
    y_max_raw = int(math.ceil(yc + half + 1))

    if (
        x_min_raw < 0
        or y_min_raw < 0
        or x_max_raw > nx
        or y_max_raw > ny
    ):
        raise ValueError(
            "A pár vagy a teljes PSF-kivágás a képszélen kívülre esik."
        )

    x_min = x_min_raw
    x_max = x_max_raw
    y_min = y_min_raw
    y_max = y_max_raw

    cut = data[y_min:y_max, x_min:x_max]
    if cut.shape[0] < 9 or cut.shape[1] < 9:
        raise ValueError("A pár túl közel van a képszélhez.")

    _, background, noise = sigma_clipped_stats(cut, sigma=3.0)
    if not np.isfinite(noise) or noise <= 0:
        noise = max(float(np.nanstd(cut)), 1.0)

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    def local_peak(x: float, y: float) -> float:
        ix = int(np.clip(round(x), 0, nx - 1))
        iy = int(np.clip(round(y), 0, ny - 1))
        r = max(2, int(round(fwhm_guess)))
        sub = data[
            max(0, iy - r):min(ny, iy + r + 1),
            max(0, ix - r):min(nx, ix + r + 1),
        ]
        return max(float(np.nanmax(sub) - background), noise)

    amp_a0 = local_peak(x_a0, y_a0)
    amp_b0 = local_peak(x_b0, y_b0)
    sigma0 = max(fwhm_guess / 2.35482, 0.7)

    p0 = np.array([
        x_a0, y_a0, amp_a0,
        x_b0, y_b0, amp_b0,
        sigma0, sigma0, 0.0, background,
    ], dtype=float)

    lower = np.array([
        x_a0 - MAX_POSITION_SHIFT_PIX,
        y_a0 - MAX_POSITION_SHIFT_PIX,
        0.0,
        x_b0 - MAX_POSITION_SHIFT_PIX,
        y_b0 - MAX_POSITION_SHIFT_PIX,
        0.0,
        0.45,
        0.45,
        -math.pi / 2.0,
        float(np.nanmin(cut)) - 5.0 * noise,
    ])

    upper = np.array([
        x_a0 + MAX_POSITION_SHIFT_PIX,
        y_a0 + MAX_POSITION_SHIFT_PIX,
        max(float(np.nanmax(cut) - background) * 5.0, amp_a0 * 5.0),
        x_b0 + MAX_POSITION_SHIFT_PIX,
        y_b0 + MAX_POSITION_SHIFT_PIX,
        max(float(np.nanmax(cut) - background) * 5.0, amp_b0 * 5.0),
        max(8.0, 2.0 * fwhm_guess),
        max(8.0, 2.0 * fwhm_guess),
        math.pi / 2.0,
        float(np.nanmax(cut)),
    ])

    # A zajjal skálázott maradék és soft_l1 veszteség mérsékli
    # a kozmikus sugarak, hot pixelek és enyhe modellhiba hatását.
    def residuals(params: np.ndarray) -> np.ndarray:
        model = gaussian_pair_model(params, xx, yy)
        return ((model - cut) / noise).ravel()

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=3000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    if not result.success:
        raise RuntimeError(f"A PSF-illesztés nem konvergált: {result.message}")

    x_a, y_a, amp_a, x_b, y_b, amp_b, sx, sy, _, _ = result.x

    # Az A/B sorrendet a WDS-várakozáshoz legközelebbi hozzárendeléssel tartjuk.
    direct = (
        math.hypot(x_a - x_a_exp, y_a - y_a_exp)
        + math.hypot(x_b - x_b_exp, y_b - y_b_exp)
    )
    swapped = (
        math.hypot(x_b - x_a_exp, y_b - y_a_exp)
        + math.hypot(x_a - x_b_exp, y_a - y_b_exp)
    )
    if swapped < direct:
        x_a, x_b = x_b, x_a
        y_a, y_b = y_b, y_a
        amp_a, amp_b = amp_b, amp_a

    if amp_a <= 0 or amp_b <= 0:
        raise ValueError("Nem pozitív illesztett komponensfluxus.")

    coord_a = pixel_to_skycoord(x_a, y_a, wcs, origin=0, mode="all")
    coord_b = pixel_to_skycoord(x_b, y_b, wcs, origin=0, mode="all")

    pa = float(coord_a.position_angle(coord_b).to_value(u.deg) % 360.0)
    rho = float(coord_a.separation(coord_b).to_value(u.arcsec))

    # Közös PSF-alak mellett az integrált fluxusarány az amplitúdóarány.
    dmag = abs(float(-2.5 * math.log10(amp_b / amp_a)))

    model = gaussian_pair_model(result.x, xx, yy)
    fit_rms = float(np.sqrt(np.nanmean((cut - model) ** 2)))
    fitted_fwhm = float(2.35482 * math.sqrt(abs(sx * sy)))

    if rho < MIN_MEASURABLE_SEPARATION_ARCSEC:
        raise ValueError(
            f"Az illesztett szeparáció {rho:.3f}\", a {MIN_MEASURABLE_SEPARATION_ARCSEC:.1f}\"-es határ alatt van."
        )

    return (
        pa, rho, dmag,
        x_a, y_a, x_b, y_b,
        fit_rms, fitted_fwhm,
        float(coord_a.ra.deg), float(coord_a.dec.deg),
        float(coord_b.ra.deg), float(coord_b.dec.deg),
    )


def read_header_and_data(path: Path) -> tuple[fits.Header, np.ndarray, WCS]:
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[0]
        header = hdu.header.copy()
        data = normalize_image_data(hdu.data)

    wcs = WCS(header, naxis=2)
    if not wcs.has_celestial:
        raise ValueError(f"Nincs használható égi WCS: {path.name}")

    return header, data, wcs.celestial





def fit_single_star_centroid(
    data: np.ndarray,
    x0: float,
    y0: float,
    fwhm_guess: float,
    nearby_sources=None,
) -> tuple[float, float, float]:
    """Adaptive, masked single-star PSF fit for WIDE pairs only.

    Excludes the cores of nearby DAO sources and saturated pixels, rather
    than shrinking an aperture until almost no stellar profile remains.
    Returns a centroid only when sufficient usable pixels and a stable fit
    remain. Original two-star PSF algorithm is untouched.
    """
    x0, y0 = float(x0), float(y0)
    ny, nx = data.shape
    fwhm = max(1.5, float(fwhm_guess))
    half = max(8, int(math.ceil(3.2 * fwhm)))
    xmin, xmax = int(math.floor(x0-half)), int(math.ceil(x0+half+1))
    ymin, ymax = int(math.floor(y0-half)), int(math.ceil(y0+half+1))
    if xmin < 0 or ymin < 0 or xmax > nx or ymax > ny:
        raise ValueError('A csillag túl közel van a képszélhez.')
    cut = np.asarray(data[ymin:ymax, xmin:xmax], dtype=float)
    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    radius = np.hypot(xx-x0, yy-y0)
    usable = np.isfinite(cut) & (radius <= 2.8*fwhm)

    # Genuine 16-bit ADC saturation: never fit the clipped core as a Gaussian.
    # The threshold is deliberately high to avoid discarding normal bright pixels.
    saturated = bool(np.any(cut[radius <= 2*fwhm] >= 65000))
    if saturated:
        usable &= (cut < 65000)

    # Nearby, *distinct* DAO detections.  Restrict to the local PSF footprint;
    # exclude their cores without excluding the target's own pixels.
    neighbours = []
    if nearby_sources is not None and len(nearby_sources):
        xs = np.asarray(nearby_sources['xcentroid'], dtype=float)
        ys = np.asarray(nearby_sources['ycentroid'], dtype=float)
        dist = np.hypot(xs-x0, ys-y0)
        for sx, sy, d in zip(xs, ys, dist):
            if np.isfinite(d) and 2.0 <= d < half + 2*fwhm:
                neighbours.append((float(sx), float(sy), float(d)))
                exclusion = max(1.8, 0.85*fwhm)
                usable &= np.hypot(xx-sx, yy-sy) > exclusion

    if np.count_nonzero(usable) < 25:
        raise ValueError('Nincs elegendő ép pixel az adaptív PSF-illesztéshez.')
    # Estimate background from *unmasked* distant pixels.
    sky = cut[usable & (radius >= 1.8*fwhm)]
    if sky.size < 12:
        sky = cut[usable]
    background = float(np.median(sky))
    mad = float(np.median(np.abs(sky-background)))
    noise = max(1.4826*mad, 1.0)

    fit_values = cut[usable]
    fit_x, fit_y = xx[usable], yy[usable]
    peak = max(float(np.max(fit_values)-background), noise)
    sigma0 = max(fwhm / 2.35482, 0.7)
    p0 = np.array([x0, y0, peak, sigma0, background], dtype=float)
    lower = [x0-MAX_POSITION_SHIFT_PIX, y0-MAX_POSITION_SHIFT_PIX,
             0.0, 0.45, float(np.min(fit_values))-5*noise]
    upper = [x0+MAX_POSITION_SHIFT_PIX, y0+MAX_POSITION_SHIFT_PIX,
             max(peak*15.0, 1.0), max(8.0, 2*fwhm),
             max(float(np.max(fit_values)), background+noise)]
    p0 = np.clip(p0, np.asarray(lower)+1e-6, np.asarray(upper)-1e-6)

    def residuals(params):
        x, y, amp, sig, bg = params
        model = bg + amp*np.exp(-0.5*((fit_x-x)**2+(fit_y-y)**2)/sig**2)
        return (model-fit_values)/noise

    result = least_squares(residuals, p0, bounds=(lower, upper),
                           loss='soft_l1', f_scale=1.0, max_nfev=1500,
                           xtol=1e-8, ftol=1e-8, gtol=1e-8)
    if not result.success:
        raise RuntimeError(f'Adaptív PSF-illesztés nem konvergált: {result.message}')
    x, y, amp, sig, bg = (float(v) for v in result.x)
    if amp <= 0 or not np.all(np.isfinite(result.x)):
        raise ValueError('Nem megbízható PSF-paraméterek.')
    if math.hypot(x-x0, y-y0) > MAX_POSITION_SHIFT_PIX:
        raise ValueError('Az illesztett centroid túl távol került a detektált csillagtól.')
    # Reject fits stuck on the allowed sigma or centroid limits.
    if sig >= max(8.0, 2*fwhm)*0.99 or sig <= 0.46:
        raise ValueError('Az illesztett PSF szélessége határértéken van.')
    print(f'Adaptive PSF: centroid=({x:.3f},{y:.3f}), '
          f'pixels={usable.sum()}, saturation={saturated}, '
          f'nearby_DAO={len(neighbours)}, nfev={result.nfev}')
    return x, y, amp


def locate_wide_component(data, sources, expected_x, expected_y, fwhm_guess, label):
    """A wide-mode component must have a local, significant peak near WDS position.

    Use DAO when possible; if saturated/weak sources are missed, use a tightly
    bounded peak search. This never substitutes the other component and does
    not bypass subsequent PSF fitting or WDS/photometric quality checks.
    """
    # Astropy WCS may return a zero-dimensional ndarray rather than a Python float.
    # Convert scalar coordinates before round(), scalar masks and distance checks.
    expected_x = float(np.asarray(expected_x).item())
    expected_y = float(np.asarray(expected_y).item())
    ny, nx = data.shape
    radius = 3.0  # identical to the PSF centroid's MAX_POSITION_SHIFT_PIX
    cx, cy = int(round(expected_x)), int(round(expected_y))
    margin = 11
    if cx - margin < 0 or cy - margin < 0 or cx + margin >= nx or cy + margin >= ny:
        raise ValueError(f"{label}: nincs elég képfelület a csillag helyi vizsgálatához.")

    yy, xx = np.mgrid[cy-margin:cy+margin+1, cx-margin:cx+margin+1]
    tile = data[cy-margin:cy+margin+1, cx-margin:cx+margin+1]
    rad = np.hypot(xx - expected_x, yy - expected_y)
    sky = tile[(rad >= 7.0) & (rad <= 11.0) & np.isfinite(tile)]
    if sky.size < 25:
        raise ValueError(f"{label}: nem mérhető a lokális háttér.")
    bg = float(np.median(sky))
    mad = float(np.median(np.abs(sky-bg)))
    noise = max(1.4826 * mad, float(np.std(sky)), 1.0)
    local_mask = (rad <= radius) & np.isfinite(tile)
    if not np.any(local_mask):
        raise ValueError(f"{label}: nincs érvényes pixel a várt pozícióban.")
    peak = float(np.max(tile[local_mask]))
    snr = (peak-bg)/noise
    if snr < 6.0:
        raise ValueError(f"{label}: nem detektálható helyi fényességcsúcs (SNR={snr:.1f}).")
    candidates = np.where(local_mask & (tile == peak))
    idx = int(np.argmin(rad[candidates]))
    py = float(yy[candidates][idx]); px = float(xx[candidates][idx])

    # Prefer DAO source when it is also tightly associated with the local peak.
    if sources is not None and len(sources):
        sx = np.asarray(sources['xcentroid'], dtype=float)
        sy = np.asarray(sources['ycentroid'], dtype=float)
        dist = np.hypot(sx-px, sy-py)
        j = int(np.argmin(dist))
        if dist[j] <= 2.0 and math.hypot(sx[j]-expected_x, sy[j]-expected_y) <= radius:
            return float(sx[j]), float(sy[j]), 'DAO'
    return px, py, 'local peak'


def measure_wide_pair(
    data: np.ndarray,
    wcs: WCS,
    row: WDSRow,
    sources,
    fwhm_guess: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Széles párok (> WIDE_PAIR_THRESHOLD_ARCSEC) mérése:
    a két komponenst külön kis kivágáson mérjük.
    """
    expected_a = row.coord
    expected_b = expected_a.directional_offset_by(
        position_angle=row.pa_last * u.deg,
        separation=row.sep_last * u.arcsec,
    )

    x_a_exp, y_a_exp = skycoord_to_pixel(expected_a, wcs, origin=0, mode="all")
    x_b_exp, y_b_exp = skycoord_to_pixel(expected_b, wcs, origin=0, mode="all")

    if not all(np.isfinite([x_a_exp, y_a_exp, x_b_exp, y_b_exp])):
        raise ValueError("A várható komponenspozíció nem alakítható pixelkoordinátává.")

    dao_ax, dao_ay, method_a = locate_wide_component(
        data, sources, x_a_exp, y_a_exp, fwhm_guess, 'A'
    )
    dao_bx, dao_by, method_b = locate_wide_component(
        data, sources, x_b_exp, y_b_exp, fwhm_guess, 'B/C'
    )
    sep_detected = math.hypot(dao_ax - dao_bx, dao_ay - dao_by)
    sep_expected = math.hypot(x_a_exp - x_b_exp, y_a_exp - y_b_exp)
    if sep_detected < max(2.0, 0.7 * sep_expected):
        raise ValueError('Széles pár: ugyanaz a forrás vagy hibás komponens-azonosítás.')

    print(f'Wide identification: A={method_a} ({dao_ax:.2f}, {dao_ay:.2f}); '
          f'B/C={method_b} ({dao_bx:.2f}, {dao_by:.2f})')
    # Azonosítsuk pontosan, melyik komponens illesztése hibázik.
    try:
        x_a, y_a, amp_a = fit_single_star_centroid(
            data, dao_ax, dao_ay, fwhm_guess, nearby_sources=sources
        )
    except Exception as exc:
        raise RuntimeError(f"Széles pár, A komponens PSF-illesztés: {exc}") from exc
    try:
        x_b, y_b, amp_b = fit_single_star_centroid(
            data, dao_bx, dao_by, fwhm_guess, nearby_sources=sources
        )
    except Exception as exc:
        raise RuntimeError(f"Széles pár, társkomponens PSF-illesztés: {exc}") from exc
    if (math.hypot(x_a-x_a_exp, y_a-y_a_exp) > 3.0 or
            math.hypot(x_b-x_b_exp, y_b-y_b_exp) > 3.0):
        raise ValueError('Széles pár: az illesztett csillagpozíció eltér a WDS alapján várttól.')

    coord_a = pixel_to_skycoord(x_a, y_a, wcs, origin=0, mode="all")
    coord_b = pixel_to_skycoord(x_b, y_b, wcs, origin=0, mode="all")

    pa = float(coord_a.position_angle(coord_b).to_value(u.deg) % 360.0)
    rho = float(coord_a.separation(coord_b).to_value(u.arcsec))
    dmag = abs(float(-2.5 * math.log10(amp_b / amp_a)))

    # wide mode-ban nincs közös fit RMS; NaN-ként adjuk vissza
    fit_rms = float("nan")
    fitted_fwhm = float(fwhm_guess)

    return (
        pa, rho, dmag,
        x_a, y_a, x_b, y_b,
        fit_rms, fitted_fwhm,
        float(coord_a.ra.deg), float(coord_a.dec.deg),
        float(coord_b.ra.deg), float(coord_b.dec.deg),
    )



def aperture_delta_mag(
    data: np.ndarray,
    x_a: float,
    y_a: float,
    x_b: float,
    y_b: float,
    fitted_fwhm: float,
) -> tuple[float, float, float, float]:
    """
    AIJ-szerű változó apertúrás fotometria.

    - apertúra sugara: kb. 1.3 * FWHM
    - közeli pároknál az apertúrát korlátozzuk, hogy a két komponens
      ne fedje erősen egymást
    - a háttérgyűrűt távolabb tesszük a csillagprofil szárnyaitól

    Visszatérés:
    abs(delta_mag), flux_a, flux_b, aperture_radius
    """
    sep_pix = math.hypot(x_b - x_a, y_b - y_a)
    if not np.isfinite(sep_pix) or sep_pix <= 1.0:
        raise ValueError("Apertúrás fotometriához túl kicsi komponens-távolság.")

    fwhm = max(float(fitted_fwhm), 1.0)

    # AIJ-szerű változó apertúra.
    r_ap = 1.60 * fwhm

    # Közeli pároknál ne nyúljon erősen bele a társ csillagba.
    r_ap = min(r_ap, 0.40 * sep_pix)
    r_ap = max(r_ap, 1.5)

    # A sky annulus legyen kellően távol a PSF szárnyaitól.
    r_in = max(3.0 * fwhm, r_ap + 3.0)
    r_out = max(5.0 * fwhm, r_in + 4.0)

    ny, nx = data.shape
    margin = int(math.ceil(r_out + 2.0))

    def one_flux(x0: float, y0: float, xo: float, yo: float) -> float:
        xmin = max(0, int(math.floor(x0 - margin)))
        xmax = min(nx, int(math.ceil(x0 + margin + 1)))
        ymin = max(0, int(math.floor(y0 - margin)))
        ymax = min(ny, int(math.ceil(y0 + margin + 1)))

        if xmax - xmin < 7 or ymax - ymin < 7:
            raise ValueError("Nincs elég hely az apertúrás fotometriához.")

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        sub = data[ymin:ymax, xmin:xmax]

        rr = np.hypot(xx - x0, yy - y0)
        rr_other = np.hypot(xx - xo, yy - yo)

        aper_mask = rr <= r_ap
        ann_mask = (rr >= r_in) & (rr <= r_out)

        # A társ csillag és közvetlen környezete ne kerüljön a sky annulusba.
        companion_exclusion = max(1.5 * r_ap, 1.5 * fwhm)
        ann_mask &= rr_other > companion_exclusion

        aper_vals = sub[aper_mask]
        ann_vals = sub[ann_mask]

        aper_vals = aper_vals[np.isfinite(aper_vals)]
        ann_vals = ann_vals[np.isfinite(ann_vals)]

        if len(aper_vals) < 8 or len(ann_vals) < 20:
            raise ValueError("Nincs elég használható pixel az apertúrás fotometriához.")

        # Robusztus háttér, AIJ-szerű sky annulus logikával.
        bg = float(np.median(ann_vals))
        mad = float(np.median(np.abs(ann_vals - bg)))
        if mad > 0:
            sigma = 1.4826 * mad
            good = np.abs(ann_vals - bg) <= 3.0 * sigma
            if np.any(good):
                bg = float(np.median(ann_vals[good]))

        flux = float(np.sum(aper_vals - bg))
        return flux

    flux_a = one_flux(x_a, y_a, x_b, y_b)
    flux_b = one_flux(x_b, y_b, x_a, y_a)

    if flux_a <= 0 or flux_b <= 0:
        raise ValueError(
            f"Nem pozitív apertúrás fluxus (A={flux_a:.3f}, B={flux_b:.3f})."
        )

    dmag = abs(float(-2.5 * math.log10(flux_b / flux_a)))
    return dmag, flux_a, flux_b, float(r_ap)


def ask_fit_path() -> Path:
    while True:
        raw = input("Seestar stacked FIT útvonala: ").strip().strip('"')
        p = Path(raw)
        if p.is_file() and p.suffix.lower() in FITS_EXTENSIONS:
            return p
        print("Nem található használható FIT/FITS fájl.")


def write_result_txt(fit_path: Path, row: WDSRow, obs_time: str,
                     pa: float, rho: float, dmag: float) -> Path:
    name = re.sub(r'[^A-Za-z0-9_+\-]+', "_", row.pair_name).strip("_")
    out = fit_path.with_name(f"{name}_measurement.txt")
    out.write_text(
        f"{APP_NAME} v{VERSION}\n\n"
        f"Object: {row.pair_name}\n"
        f"WDS: {row.wds_id}\n"
        f"Date: {obs_time} UTC\n\n"
        f"PA: {pa:.2f} deg\n"
        f"Separation: {rho:.2f} arcsec\n"
        f"Delta mag: {abs(dmag):.2f} mag\n\n"
        f"Quality: GOOD\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    print("=" * 68)
    print(f"{APP_NAME} v{VERSION}")
    print("=" * 68)
    print("Basic mód: 1 stacked Seestar FIT + 1 megadott kettős")
    print(
        f'Ajánlott tartomány: sep >= {MIN_BASIC_SEPARATION_ARCSEC:.0f}", '
        f'mindkét komponens <= {MAX_BASIC_MAG:.1f} mag'
    )
    print("Mérési motor: IsoTool PhotoAnalyzer v1.3")
    print(f"Fényességi védelem: max. {MAX_WDS_DMAG_DIFFERENCE:.1f} mag eltérés a WDS Δmag-tól")
    print(
        f'Módváltás: <= {WIDE_PAIR_THRESHOLD_ARCSEC:.0f}" közös PSF-fit, '
        f'> {WIDE_PAIR_THRESHOLD_ARCSEC:.0f}" külön komponensmérés'
    )
    print()

    try:
        catalog = load_basic_catalog(get_app_dir() / "Seestar_wds.csv")
        print(f"Katalógus betöltve: {len(catalog)} sor")
    except Exception as exc:
        print("\nHIBA")
        print("-" * 40)
        print(str(exc))
        input("\nEnter a kilépéshez...")
        return 1

    while True:
        try:
            fit_path = ask_fit_path()
            query = input("Kettős azonosító (pl. HJ 1362): ").strip()
            row = choose_match(find_matches(catalog, query))

            print("\nKiválasztott pár:")
            print(f"  {row.pair_name} | WDS {row.wds_id}")
            print(f'  WDS last: PA={row.pa_last:.0f}°, sep={row.sep_last:.1f}"')
            print(f"  Mag: {row.mag_a:.2f} / {row.mag_b:.2f}")

            print("\nMérés folyamatban...")
            header, data, wcs = read_header_and_data(fit_path)
            fwhm_guess, sources = estimate_fwhm_and_sources(data)
            print(f"DAO elődetektálás: {len(sources)} forrás, FWHM~{fwhm_guess:.1f} px")

            if row.sep_last > WIDE_PAIR_THRESHOLD_ARCSEC:
                print(
                    f"Széles pár mód: külön komponensmérés "
                    f'(WDS sep > {WIDE_PAIR_THRESHOLD_ARCSEC:.0f}").'
                )
                (
                    pa, rho, dmag_psf,
                    x_a, y_a, x_b, y_b,
                    fit_rms, fitted_fwhm,
                    ra_a, dec_a, ra_b, dec_b,
                ) = measure_wide_pair(
                    data=data,
                    wcs=wcs,
                    row=row,
                    sources=sources,
                    fwhm_guess=fwhm_guess,
                )
            else:
                print(
                    f"Közeli pár mód: közös kétkomponensű PSF-fit "
                    f'(WDS sep <= {WIDE_PAIR_THRESHOLD_ARCSEC:.0f}").'
                )
                (
                    pa, rho, dmag_psf,
                    x_a, y_a, x_b, y_b,
                    fit_rms, fitted_fwhm,
                    ra_a, dec_a, ra_b, dec_b,
                ) = fit_pair_on_frame(
                    data=data,
                    wcs=wcs,
                    row=row,
                    sources=sources,
                    fwhm_guess=fwhm_guess,
                )

            sep_ratio = rho / row.sep_last if row.sep_last > 0 else np.nan
            if not np.isfinite(sep_ratio) or not (0.75 <= sep_ratio <= 1.25):
                raise ValueError(
                    f'A mért szeparáció ({rho:.2f}") túlzottan eltér '
                    f'a WDS referenciaértéktől ({row.sep_last:.2f}").'
                )

            dmag_psf = abs(float(dmag_psf))

            dmag, flux_a, flux_b, r_ap = aperture_delta_mag(
                data=data,
                x_a=x_a, y_a=y_a,
                x_b=x_b, y_b=y_b,
                fitted_fwhm=fitted_fwhm,
            )

            # Fényességi plausibility-check:
            # ne fogadjunk el egy véletlen, jóval halványabb/fényesebb csillagot
            # pusztán azért, mert a várt pozíció közelében van.
            wds_dmag = abs(float(row.mag_b) - float(row.mag_a))
            dmag_diff = abs(float(dmag) - wds_dmag)

            print(
                f"Fotometria: PSF Δmag={dmag_psf:.2f}, "
                f"AIJ-szerű apertúrás Δmag={dmag:.2f}, r={r_ap:.2f} px"
            )
            print(
                f"Fényességi ellenőrzés: WDS Δmag={wds_dmag:.2f}, "
                f"eltérés={dmag_diff:.2f} mag"
            )

            if dmag_diff > MAX_WDS_DMAG_DIFFERENCE:
                raise ValueError(
                    f"A detektált társ fényessége nem egyezik eléggé a WDS-adattal "
                    f"(WDS Δmag={wds_dmag:.2f}, mért Δmag={dmag:.2f}, "
                    f"eltérés={dmag_diff:.2f} mag)."
                )

            obs_time = observation_time_text(header)
            out = write_result_txt(fit_path, row, obs_time, pa, rho, dmag)

            print("\nEREDMÉNY")
            print("-" * 40)
            print(f"Object:      {row.pair_name}")
            print(f"WDS:         {row.wds_id}")
            print(f"PA:          {pa:.2f} deg")
            print(f"Separation:  {rho:.2f} arcsec")
            print(f"Delta mag:   {dmag:.2f} mag")
            print("Quality:     GOOD")
            print(f"\nTXT mentve: {out}")
            # Optional image export: a failure here cannot invalidate a valid measurement.
            try:
                from duostar_annotation import create_annotated_image
                png = create_annotated_image(
                    data, wcs, row, x_a, y_a, x_b, y_b,
                    out.with_name(out.stem.replace("_measurement", "_annotated") + ".png"),
                )
                print(f"PNG mentve: {png}")
            except Exception as image_exc:
                print(f"PNG nem készült: {image_exc}")

        except Exception as exc:
            print("\nMÉRÉS ELUTASÍTVA")
            print("-" * 40)
            print(str(exc))

        while True:
            print("\nMit szeretnél?")
            print("  1 - Új mérés")
            print("  2 - Kilépés")
            choice = input("Választás: ").strip()
            if choice == "1":
                print()
                break
            if choice == "2":
                return 0
            print("Érvénytelen választás.")



if __name__ == "__main__":
    raise SystemExit(main())
