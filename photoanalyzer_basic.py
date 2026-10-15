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
VERSION = "1.1e"
LAST_WIDE_DIAGNOSTICS = None
LAST_WING_TESTS = None

MIN_BASIC_SEPARATION_ARCSEC = 10.0
MAX_BASIC_MAG = 14.0
WIDE_PAIR_THRESHOLD_ARCSEC = 30.0

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
            raise ValueError("Missing catalog column(s): " + ", ".join(sorted(missing)))

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
        raise LookupError("This double star is not in the Seestar catalog.")

    if len(matches) == 1:
        return matches[0]

    print("\nMultiple matches found. Select a number:")
    for i, row in enumerate(matches, 1):
        comp = row.comp or "-"
        print(
            f"  {i}. {row.disc} {comp} | WDS {row.wds_id} | "
            f"PA={row.pa_last:.0f}° | sep={row.sep_last:.1f}\" | "
            f"{row.mag_a:.2f}/{row.mag_b:.2f} mag"
        )
    while True:
        try:
            n = int(input("Number: ").strip())
            if 1 <= n <= len(matches):
                return matches[n-1]
        except Exception:
            pass
        print("Invalid selection.")


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
        raise ValueError("The FITS image contains no finite pixel values.")

    return arr


def estimate_fwhm_and_sources(data: np.ndarray) -> tuple[float, object]:
    """
    Automatikus forrásdetektálás több FWHM-próbával.

    A DAO itt csak elődetektálásra és kezdőpozíciókhoz szolgál.
    A végleges mérés nem a DAO nyers centroidjaiból készül.
    """
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Unable to estimate the image background noise.")

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
        raise RuntimeError("DAOStarFinder found no usable stars.")

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
            "No independently detectable stellar source at the first component position."
        )

    if dao_b is None:
        raise ValueError(
            "No independently detectable stellar source at the second component position."
        )

    measured_sep_pix = math.hypot(
        dao_a[0] - dao_b[0],
        dao_a[1] - dao_b[1],
    )

    min_independent_sep = max(1.5, 0.75 * float(fwhm_guess))
    if measured_sep_pix < min_independent_sep:
        raise ValueError(
            "The two components cannot be resolved as separate stars in the stacked FITS image."
        )

    expected_sep_pix = math.hypot(
        x_a_exp - x_b_exp,
        y_a_exp - y_b_exp,
    )

    if expected_sep_pix > 0:
        ratio = measured_sep_pix / expected_sep_pix
        if not (0.70 <= ratio <= 1.30):
            raise ValueError(
                "The separation of the detected stars differs too much "
                "from the WDS reference separation."
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
        raise ValueError("The pair is below the 1-arcsecond measurement limit.")

    expected_a = row.coord
    expected_b = expected_a.directional_offset_by(
        position_angle=row.pa_last * u.deg,
        separation=row.sep_last * u.arcsec,
    )

    x_a_exp, y_a_exp = skycoord_to_pixel(expected_a, wcs, origin=0, mode="all")
    x_b_exp, y_b_exp = skycoord_to_pixel(expected_b, wcs, origin=0, mode="all")

    if not all(np.isfinite([x_a_exp, y_a_exp, x_b_exp, y_b_exp])):
        raise ValueError("Cannot transform the expected component position into pixel coordinates.")

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
            "The pair or full PSF cutout extends beyond the image boundary."
        )

    x_min = x_min_raw
    x_max = x_max_raw
    y_min = y_min_raw
    y_max = y_max_raw

    cut = data[y_min:y_max, x_min:x_max]
    if cut.shape[0] < 9 or cut.shape[1] < 9:
        raise ValueError("The pair is too close to the image edge.")

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
        raise ValueError("Non-positive fitted component flux.")

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
            f"The fitted separation ({rho:.3f}\") is below the {MIN_MEASURABLE_SEPARATION_ARCSEC:.1f}\" measurement limit."
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
        raise ValueError('The star is too close to the image edge.')
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
        raise ValueError('Insufficient valid pixels for adaptive PSF fitting.')
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
        raise ValueError('Unreliable PSF parameters.')
    if math.hypot(x-x0, y-y0) > MAX_POSITION_SHIFT_PIX:
        raise ValueError('The fitted centroid moved too far from the detected star.')
    # Reject fits stuck on the allowed sigma or centroid limits.
    if sig >= max(8.0, 2*fwhm)*0.99 or sig <= 0.46:
        raise ValueError('The fitted PSF width reached a limit.')
    print(f'Adaptive PSF: centroid=({x:.3f},{y:.3f}), '
          f'pixels={usable.sum()}, saturation={saturated}, '
          f'nearby_DAO={len(neighbours)}, nfev={result.nfev}')
    return x, y, amp




def fit_saturated_star_wings(
    data: np.ndarray,
    x0: float,
    y0: float,
    fwhm_guess: float,
    nearby_sources=None,
) -> tuple[float, float, float]:
    """Tartalék centroid a levágott csillagprofil *nem telített szárnyaiból*.

    Csak akkor alkalmazható, ha a normál illesztés meghiúsult ÉS a csillag
    magjában valóban vannak telített pixelek. A koordinátákat ellenőrzi;
    a fluxust továbbra is az eredeti apertúrás fotometria számítja.
    """
    x0, y0 = float(x0), float(y0)
    fwhm = max(1.5, float(fwhm_guess))
    ny, nx = data.shape
    half = max(10, int(math.ceil(3.5 * fwhm)))
    xmin, xmax = int(math.floor(x0-half)), int(math.ceil(x0+half+1))
    ymin, ymax = int(math.floor(y0-half)), int(math.ceil(y0+half+1))
    if xmin < 0 or ymin < 0 or xmax > nx or ymax > ny:
        raise ValueError('Saturated PSF: star too close to the image edge.')
    cut = np.asarray(data[ymin:ymax, xmin:xmax], dtype=float)
    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    radius = np.hypot(xx-x0, yy-y0)
    saturated_core = (radius <= max(3.0, 1.5*fwhm)) & (cut >= 65000)
    if np.count_nonzero(saturated_core) == 0:
        raise ValueError('Saturated PSF fallback unavailable: no saturated core.')

    usable = np.isfinite(cut) & (cut < 65000) & (radius <= 3.0*fwhm)
    if nearby_sources is not None and len(nearby_sources):
        xs = np.asarray(nearby_sources['xcentroid'], dtype=float)
        ys = np.asarray(nearby_sources['ycentroid'], dtype=float)
        for sx, sy in zip(xs, ys):
            dist = math.hypot(float(sx)-x0, float(sy)-y0)
            if 2.0 <= dist < half+fwhm:
                usable &= np.hypot(xx-sx, yy-sy) > max(1.8, 0.85*fwhm)
    if np.count_nonzero(usable) < 35:
        raise ValueError('Saturated PSF: insufficient usable wing pixels.')

    # A szárnyak több irányból kell, hogy megmaradjanak, különben a
    # szomszédos csillag fénye felé eltolódhatna a centroid.
    ang = (np.arctan2(yy-y0, xx-x0) + 2*np.pi) % (2*np.pi)
    sectors = set(np.floor(ang[usable] / (np.pi/4)).astype(int).tolist())
    if len(sectors) < 5:
        raise ValueError('Saturated PSF: usable profile does not cover enough directions.')

    bg_pixels = cut[usable & (radius >= 2.2*fwhm)]
    if bg_pixels.size < 15:
        bg_pixels = cut[usable & (radius >= 1.7*fwhm)]
    if bg_pixels.size < 15:
        raise ValueError('Saturated PSF: background cannot be estimated.')
    bg0 = float(np.median(bg_pixels))
    noise = max(1.4826*float(np.median(np.abs(bg_pixels-bg0))), 1.0)
    vx = xx[usable].astype(float)
    vy = yy[usable].astype(float)
    vals = cut[usable]
    sigma0 = max(0.7, fwhm/2.35482)
    amplitude0 = max(float(np.max(vals)-bg0), noise*6.0)
    lower = np.array([x0-2.0, y0-2.0, 0.0, 0.55, bg0-5*noise])
    upper = np.array([x0+2.0, y0+2.0,
                      max(amplitude0*30, 70000.0),
                      max(7.0, 2.0*fwhm), bg0+5*noise])

    def residuals(p):
        x, y, amplitude, sigma, bg = p
        model = bg + amplitude*np.exp(-0.5*((vx-x)**2+(vy-y)**2)/sigma**2)
        return (model-vals)/noise

    candidates = []
    for sigma_scale in (0.8, 1.0, 1.3, 1.7):
        p0 = np.array([x0, y0, max(amplitude0*1.5, 65000-bg0),
                       sigma0*sigma_scale, bg0])
        p0 = np.maximum(lower+1e-5, np.minimum(upper-1e-5, p0))
        try:
            result = least_squares(residuals, p0, bounds=(lower, upper),
                                   loss='soft_l1', f_scale=2.0,
                                   max_nfev=5000)
        except (ValueError, FloatingPointError):
            continue
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        x, y, amp, sig, bg = result.x
        if amp <= 0 or sig < 0.6 or sig >= upper[3]*0.98:
            continue
        if math.hypot(x-x0, y-y0) > 1.8:
            continue
        if np.any(np.isclose(result.x[:2], lower[:2], atol=0.03)) or \
           np.any(np.isclose(result.x[:2], upper[:2], atol=0.03)):
            continue
        error = float(np.median(np.abs(residuals(result.x))))
        if np.isfinite(error):
            candidates.append((error, float(x), float(y), float(amp)))

    if not candidates:
        raise ValueError('Saturated PSF: wing fit is unstable.')
    candidates.sort()
    error, x, y, amp = candidates[0]
    if error > 12.0:
        raise ValueError(f'Saturated PSF: excessive wing residual ({error:.1f} sigma).')
    print(f'Telített PSF-tartalék: centroid=({x:.3f},{y:.3f}), '
          f'használható pixelek={int(np.count_nonzero(usable))}, '
          f'illesztési maradék={error:.2f} sigma')
    return x, y, amp


def independent_wing_centroid(data, x0, y0, fwhm_guess, nearby_sources=None):
    """Független, konzervatív szárny-centroid teszt TELÍTETT csillagokra.

    Nem módosítja a mérést: két eltérő belső maszkkal becsli a súlyozott
    középpontot, és elutasítja az instabil/aszimmetrikus eredményt.
    """
    x0, y0 = float(x0), float(y0)
    fwhm = max(1.5, float(fwhm_guess))
    ny, nx = data.shape
    half = int(math.ceil(3.5 * fwhm))
    xmin, xmax = int(math.floor(x0-half)), int(math.ceil(x0+half+1))
    ymin, ymax = int(math.floor(y0-half)), int(math.ceil(y0+half+1))
    if xmin < 0 or ymin < 0 or xmax > nx or ymax > ny:
        raise ValueError('Wing centroid: image edge.')
    cut = np.asarray(data[ymin:ymax, xmin:xmax], dtype=float)
    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    rr = np.hypot(xx-x0, yy-y0)
    sat_count = int(np.count_nonzero((rr < 1.6*fwhm) & (cut >= 65000)))
    if sat_count == 0:
        raise ValueError('Wing centroid: no saturated core.')
    sky = cut[(rr >= 2.8*fwhm) & (rr <= 3.4*fwhm) & np.isfinite(cut)]
    if sky.size < 20:
        raise ValueError('Wing centroid: insufficient background pixels.')
    bg = float(np.median(sky))
    noise = max(1.0, 1.4826 * float(np.median(np.abs(sky-bg))))
    exclusion = np.zeros_like(cut, dtype=bool)
    if nearby_sources is not None and len(nearby_sources):
        for sx, sy in zip(nearby_sources['xcentroid'], nearby_sources['ycentroid']):
            d = math.hypot(float(sx)-x0, float(sy)-y0)
            if 2.0 <= d < 3.5*fwhm:
                exclusion |= np.hypot(xx-float(sx), yy-float(sy)) < max(1.8, .85*fwhm)

    estimates = []
    for inner in (0.65*fwhm, 0.9*fwhm):
        mask = (np.isfinite(cut) & (cut < 65000) & ~exclusion
                & (rr >= inner) & (rr <= 2.2*fwhm))
        weights = np.maximum(cut-bg-2.0*noise, 0.0)
        mask &= weights > 0
        if np.count_nonzero(mask) < 25 or weights[mask].sum() <= 0:
            raise ValueError('Wing centroid: insufficient usable wing pixels.')
        angles = (np.arctan2(yy[mask]-y0, xx[mask]-x0) + 2*np.pi) % (2*np.pi)
        if len(np.unique(np.floor(angles / (np.pi/4)).astype(int))) < 6:
            raise ValueError('Wing centroid: asymmetric profile.')
        cx = float(np.average(xx[mask], weights=weights[mask]))
        cy = float(np.average(yy[mask], weights=weights[mask]))
        if math.hypot(cx-x0, cy-y0) > 1.5:
            raise ValueError('Wing centroid: excessive displacement.')
        estimates.append((cx, cy))
    stability = math.hypot(estimates[0][0]-estimates[1][0],
                           estimates[0][1]-estimates[1][1])
    if stability > 0.35:
        raise ValueError(f'Szárnycentroid: instabil két maszk ({stability:.3f} px).')
    return (0.5*(estimates[0][0]+estimates[1][0]),
            0.5*(estimates[0][1]+estimates[1][1]), stability, sat_count)


def locate_wide_component(data, sources, expected_x, expected_y, fwhm_guess, label, search_radius=3.0):
    """A wide-mode component must have a local, significant peak near WDS position.

    Use DAO when possible; if saturated/weak sources are missed, use a tightly
    bounded peak search. This never substitutes the other component and does
    not bypass subsequent PSF fitting or astrometric quality checks.
    """
    # Astropy WCS may return a zero-dimensional ndarray rather than a Python float.
    # Convert scalar coordinates before round(), scalar masks and distance checks.
    expected_x = float(np.asarray(expected_x).item())
    expected_y = float(np.asarray(expected_y).item())
    ny, nx = data.shape
    radius = float(search_radius)  # WDS-eltérés megengedett keresési sugara
    cx, cy = int(round(expected_x)), int(round(expected_y))
    margin = max(11, int(math.ceil(radius + 9.0)))
    if cx - margin < 0 or cy - margin < 0 or cx + margin >= nx or cy + margin >= ny:
        raise ValueError(f"{label}: insufficient image area for local star inspection.")

    yy, xx = np.mgrid[cy-margin:cy+margin+1, cx-margin:cx+margin+1]
    tile = data[cy-margin:cy+margin+1, cx-margin:cx+margin+1]
    rad = np.hypot(xx - expected_x, yy - expected_y)
    sky = tile[(rad >= radius + 4.0) & (rad <= radius + 8.0) & np.isfinite(tile)]
    if sky.size < 25:
        raise ValueError(f"{label}: local background cannot be estimated.")
    bg = float(np.median(sky))
    mad = float(np.median(np.abs(sky-bg)))
    noise = max(1.4826 * mad, float(np.std(sky)), 1.0)
    local_mask = (rad <= radius) & np.isfinite(tile)
    if not np.any(local_mask):
        raise ValueError(f"{label}: no valid pixels at the expected position.")
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
        raise ValueError("Cannot transform the expected component position into pixel coordinates.")

    # Régi WDS-adatoknál a társ pozíciója több pixellel is elmozdulhat.
    # A nagyobb keresési sugarat a PA/szeparáció ellenőrzése ellensúlyozza.
    expected_sep_pix = math.hypot(x_a_exp - x_b_exp, y_a_exp - y_b_exp)
    companion_search = min(12.0, max(3.0, 0.25 * expected_sep_pix))
    primary_search = 6.0  # az A komponens WDS-koordinátája is lehet elavult

    dao_ax, dao_ay, method_a = locate_wide_component(
        data, sources, x_a_exp, y_a_exp, fwhm_guess, 'A', search_radius=primary_search
    )
    dao_bx, dao_by, method_b = locate_wide_component(
        data, sources, x_b_exp, y_b_exp, fwhm_guess, 'B/C', search_radius=companion_search
    )
    sep_detected = math.hypot(dao_ax - dao_bx, dao_ay - dao_by)
    sep_expected = math.hypot(x_a_exp - x_b_exp, y_a_exp - y_b_exp)
    if sep_detected < max(2.0, 0.7 * sep_expected):
        raise ValueError('Wide pair: same source detected twice or incorrect component identification.')

    print(f'Wide identification: A={method_a} ({dao_ax:.2f}, {dao_ay:.2f}); '
          f'B/C={method_b} ({dao_bx:.2f}, {dao_by:.2f})')
    # Azonosítsuk pontosan, melyik komponens illesztése hibázik.
    psf_a = "normal"
    psf_b = "normal"
    try:
        x_a, y_a, amp_a = fit_single_star_centroid(
            data, dao_ax, dao_ay, fwhm_guess, nearby_sources=sources
        )
    except Exception as exc:
        psf_a = 'saturated fallback'
        print(f'First component: standard PSF fit failed ({exc}); trying saturated-star fallback.')
        try:
            x_a, y_a, amp_a = fit_saturated_star_wings(
                data, dao_ax, dao_ay, fwhm_guess, nearby_sources=sources
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Wide pair, first component PSF fit: {exc}; "
                f"telített tartalék: {fallback_exc}"
            ) from fallback_exc
    try:
        x_b, y_b, amp_b = fit_single_star_centroid(
            data, dao_bx, dao_by, fwhm_guess, nearby_sources=sources
        )
    except Exception as exc:
        psf_b = 'saturated fallback'
        print(f'Second component: standard PSF fit failed ({exc}); trying saturated-star fallback.')
        try:
            x_b, y_b, amp_b = fit_saturated_star_wings(
                data, dao_bx, dao_by, fwhm_guess, nearby_sources=sources
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Wide pair, second component PSF fit: {exc}; "
                f"telített tartalék: {fallback_exc}"
            ) from fallback_exc
    # Független szárny-centroidok csak összehasonlításra. A mért PA/rho
    # változatlanul a v1.1c illesztett koordinátákból számolódik.
    wing_tests = {}
    for label, dx, dy, fx, fy in (
        ('A', dao_ax, dao_ay, x_a, y_a),
        ('B', dao_bx, dao_by, x_b, y_b),
    ):
        try:
            wx, wy, stability, sat_count = independent_wing_centroid(
                data, dx, dy, fwhm_guess, nearby_sources=sources)
            wing_tests[label] = (wx, wy, stability, sat_count, None)
            print(f'{label}: wing centroid=({wx:.4f}, {wy:.4f}), '
                  f'PSF difference={math.hypot(wx-fx,wy-fy):.4f} px, '
                  f'stabilitás={stability:.4f} px')
        except Exception as exc:
            wing_tests[label] = (None, None, None, None, str(exc))
            print(f'{label}: wing centroid unavailable: {exc}')
    global LAST_WING_TESTS
    LAST_WING_TESTS = wing_tests
    # Diagnosztikai adatok: nincs hatásuk a mérési algoritmusra.
    # A naplót csak a sikeres komponensillesztés után írjuk.
    global LAST_WIDE_DIAGNOSTICS
    diagnostics = {
        "A": (x_a_exp, y_a_exp, dao_ax, dao_ay, x_a, y_a, method_a, psf_a),
        "B": (x_b_exp, y_b_exp, dao_bx, dao_by, x_b, y_b, method_b, psf_b),
    }
    LAST_WIDE_DIAGNOSTICS = diagnostics
    print("\n--- DUOSTAR CENTROID DIAGNOSTICS ---")
    for label, (ex, ey, dx, dy, fx, fy, detection, fitter) in diagnostics.items():
        print(f"{label}: WDS=({float(ex):.4f}, {float(ey):.4f}) "
              f"detected=({dx:.4f}, {dy:.4f}) "
              f"fitted=({fx:.4f}, {fy:.4f}) "
              f"WDS offset={math.hypot(fx-ex,fy-ey):.4f}px "
              f"detektált-fitted={math.hypot(fx-dx,fy-dy):.4f}px "
              f"detection={detection} PSF={fitter}")
    print("--- END OF DIAGNOSTICS ---\n")
    offset_a = math.hypot(x_a-x_a_exp, y_a-y_a_exp)
    offset_b = math.hypot(x_b-x_b_exp, y_b-y_b_exp)
    primary_limit = primary_search + MAX_POSITION_SHIFT_PIX
    print(f'Wide WDS offsets: A={offset_a:.2f} px (limit={primary_limit:.2f}); '
          f'B/C={offset_b:.2f} px (limit={companion_search + MAX_POSITION_SHIFT_PIX:.2f})')
    if offset_a > primary_limit or offset_b > companion_search + MAX_POSITION_SHIFT_PIX:
        failed = []
        if offset_a > primary_limit:
            failed.append(f'A: {offset_a:.2f} px (limit {primary_limit:.2f} px)')
        if offset_b > companion_search + MAX_POSITION_SHIFT_PIX:
            failed.append(f'B/C: {offset_b:.2f} px (limit {companion_search + MAX_POSITION_SHIFT_PIX:.2f} px)')
        raise ValueError('Wide pair: WDS position check failed: ' + '; '.join(failed))

    coord_a = pixel_to_skycoord(x_a, y_a, wcs, origin=0, mode="all")
    coord_b = pixel_to_skycoord(x_b, y_b, wcs, origin=0, mode="all")

    pa = float(coord_a.position_angle(coord_b).to_value(u.deg) % 360.0)
    rho = float(coord_a.separation(coord_b).to_value(u.arcsec))
    pa_diff = abs((pa - row.pa_last + 180.0) % 360.0 - 180.0)
    if pa_diff > 8.0:
        raise ValueError(f'Wide pair: PA difference is too large ({pa_diff:.2f}° > 8°); possible misidentification.')
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
    signed(delta_mag), flux_a, flux_b, aperture_radius
    """
    sep_pix = math.hypot(x_b - x_a, y_b - y_a)
    if not np.isfinite(sep_pix) or sep_pix <= 1.0:
        raise ValueError("Component separation is too small for aperture photometry.")

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
            raise ValueError("Insufficient image area for aperture photometry.")

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
            raise ValueError("Insufficient usable pixels for aperture photometry.")

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

    dmag = float(-2.5 * math.log10(flux_b / flux_a))
    return dmag, flux_a, flux_b, float(r_ap)


def ask_fit_path() -> Path:
    while True:
        raw = input("Seestar stacked FITS path: ").strip().strip('"')
        p = Path(raw)
        if p.is_file() and p.suffix.lower() in FITS_EXTENSIONS:
            return p
        print("No usable FIT/FITS file found.")


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
        f"Mag1 (WDS): {row.mag_a:.2f} mag\n"
        f"Mag2 (calculated): {row.mag_a + dmag:.2f} mag\n"
        f"Delta mag: {abs(dmag):.2f} mag\n\n"
        f"Quality: GOOD\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    global LAST_WIDE_DIAGNOSTICS, LAST_WING_TESTS
    print("=" * 68)
    print(f"{APP_NAME} v{VERSION}")
    print("=" * 68)
    print("Basic mode: 1 stacked Seestar FITS + 1 selected double star")
    print(
        f'Recommended range: sep >= {MIN_BASIC_SEPARATION_ARCSEC:.0f}", '
        f'both components <= {MAX_BASIC_MAG:.1f} mag'
    )
    print("Measurement engine: IsoTool PhotoAnalyzer v1.3")
    print(
        f'Mode selection: <= {WIDE_PAIR_THRESHOLD_ARCSEC:.0f}" joint PSF fit, '
        f'> {WIDE_PAIR_THRESHOLD_ARCSEC:.0f}" independent component fitting'
    )
    print()

    try:
        catalog = load_basic_catalog(get_app_dir() / "Seestar_wds.csv")
        print(f"Catalog loaded: {len(catalog)} entries")
    except Exception as exc:
        print("\nHIBA")
        print("-" * 40)
        print(str(exc))
        input("\nPress Enter to exit...")
        return 1

    while True:
        try:
            fit_path = ask_fit_path()
            query = input("Double star designation (e.g. HJ 1362): ").strip()
            row = choose_match(find_matches(catalog, query))

            print("\nSelected pair:")
            print(f"  {row.pair_name} | WDS {row.wds_id}")
            print(f'  WDS last: PA={row.pa_last:.0f}°, sep={row.sep_last:.1f}"')
            print(f"  Mag: {row.mag_a:.2f} / {row.mag_b:.2f}")

            print("\nMeasurement in progress...")
            LAST_WIDE_DIAGNOSTICS = None
            LAST_WING_TESTS = None
            header, data, wcs = read_header_and_data(fit_path)
            fwhm_guess, sources = estimate_fwhm_and_sources(data)
            print(f"DAO elődetektálás: {len(sources)} forrás, FWHM~{fwhm_guess:.1f} px")

            if row.sep_last > WIDE_PAIR_THRESHOLD_ARCSEC:
                print(
                    f"Wide-pair mode: independent component fitting "
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
                    f"Close-pair mode: joint two-component PSF fit "
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

            # Egyszerű, reprodukálható asztrometriai napló a FITS mellett.
            # A mérés eredményét és a Quality minősítést nem módosítja.
            diag_path = fit_path.with_name(
                re.sub(r'[^A-Za-z0-9_+\-]+', '_', row.pair_name).strip('_')
                + '_centroid_diagnostic.txt'
            )
            try:
                expected_a = row.coord
                expected_b = expected_a.directional_offset_by(
                    position_angle=row.pa_last * u.deg,
                    separation=row.sep_last * u.arcsec,
                )
                exa, eya = skycoord_to_pixel(expected_a, wcs, origin=0, mode='all')
                exb, eyb = skycoord_to_pixel(expected_b, wcs, origin=0, mode='all')
                lines = [f"DuoStar centroid diagnostics v{VERSION}",
                         f"FITS: {fit_path.name}", f"Object: {row.pair_name}",
                         f"WDS: {row.wds_id}",
                         f"Mode: {'wide' if row.sep_last > WIDE_PAIR_THRESHOLD_ARCSEC else 'close'}",
                         f"DAO FWHM guess: {fwhm_guess:.4f} px", ""]
                for label, ex, ey, fx, fy, ra, dec in (
                    ('A', exa, eya, x_a, y_a, ra_a, dec_a),
                    ('B', exb, eyb, x_b, y_b, ra_b, dec_b),
                ):
                    lines.extend([
                        f"{label}: expected_WDS_pixel = ({float(ex):.5f}, {float(ey):.5f})",
                        f"{label}: fitted_pixel = ({float(fx):.5f}, {float(fy):.5f})",
                        f"{label}: fitted_RA_Dec_deg = ({ra:.9f}, {dec:.9f})",
                        f"{label}: WDS_offset_px = {math.hypot(fx-ex, fy-ey):.5f}",
                    ])
                if LAST_WIDE_DIAGNOSTICS is not None:
                    lines.append("")
                    for label, (ex, ey, dx, dy, fx, fy, det, psf) in LAST_WIDE_DIAGNOSTICS.items():
                        lines.extend([
                            f"{label}: detected_pixel = ({dx:.5f}, {dy:.5f})",
                            f"{label}: detection_method = {det}",
                            f"{label}: psf_method = {psf}",
                            f"{label}: detected_to_fitted_px = {math.hypot(fx-dx,fy-dy):.5f}",
                        ])
                if LAST_WING_TESTS is not None:
                    lines.append('')
                    lines.append('INDEPENDENT WING CENTROID (diagnostic only, not used in PA/rho):')
                    for label, (wx, wy, stability, sat_count, error) in LAST_WING_TESTS.items():
                        if error is not None:
                            lines.append(f'{label}: wing_status = rejected: {error}')
                        else:
                            fx, fy = (x_a, y_a) if label == 'A' else (x_b, y_b)
                            lines.extend([
                                f'{label}: wing_pixel = ({wx:.5f}, {wy:.5f})',
                                f'{label}: wing_mask_stability_px = {stability:.5f}',
                                f'{label}: saturated_core_pixel_count = {sat_count}',
                                f'{label}: wing_to_psf_px = {math.hypot(wx-fx,wy-fy):.5f}',
                            ])
                    if all(LAST_WING_TESTS[k][4] is None for k in ('A','B')):
                        wa, wb = LAST_WING_TESTS['A'], LAST_WING_TESTS['B']
                        ca = pixel_to_skycoord(wa[0], wa[1], wcs, origin=0, mode='all')
                        cb = pixel_to_skycoord(wb[0], wb[1], wcs, origin=0, mode='all')
                        lines.extend([
                            f'WING_ONLY_PA_deg = {float(ca.position_angle(cb).to_value(u.deg)%360):.5f}',
                            f'WING_ONLY_Separation_arcsec = {float(ca.separation(cb).to_value(u.arcsec)):.5f}',
                        ])
                    elif LAST_WING_TESTS['B'][4] is None:
                        wb = LAST_WING_TESTS['B']
                        ca = pixel_to_skycoord(x_a, y_a, wcs, origin=0, mode='all')
                        cb = pixel_to_skycoord(wb[0], wb[1], wcs, origin=0, mode='all')
                        lines.extend([
                            f'A_PSF_B_WING_PA_deg = {float(ca.position_angle(cb).to_value(u.deg)%360):.5f}',
                            f'A_PSF_B_WING_Separation_arcsec = {float(ca.separation(cb).to_value(u.arcsec)):.5f}',
                        ])
                lines.extend(["", f"PA_deg = {pa:.5f}", f"Separation_arcsec = {rho:.5f}"])
                diag_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                print(f"Centroid diagnostics saved: {diag_path}")
            except Exception as diag_exc:
                print(f"Centroid diagnostics not created: {diag_exc}")

            sep_ratio = rho / row.sep_last if row.sep_last > 0 else np.nan
            if not np.isfinite(sep_ratio) or not (0.75 <= sep_ratio <= 1.25):
                raise ValueError(
                    f'The measured separation ({rho:.2f}") differs too much '
                    f'from the WDS reference value ({row.sep_last:.2f}").'
                )

            dmag_psf = abs(float(dmag_psf))

            dmag, flux_a, flux_b, r_ap = aperture_delta_mag(
                data=data,
                x_a=x_a, y_a=y_a,
                x_b=x_b, y_b=y_b,
                fitted_fwhm=fitted_fwhm,
            )

            # A WDS magnitúdók nem szolgálnak kizárási feltételként.
            # A ténylegesen mért (előjeles) Delta mag marad a kimenet alapja.
            print(
                f"Photometry: PSF Δmag={dmag_psf:.2f}, "
                f"AIJ-style aperture Δmag={dmag:.2f}, r={r_ap:.2f} px"
            )

            obs_time = observation_time_text(header)
            out = write_result_txt(fit_path, row, obs_time, pa, rho, dmag)

            print("\nRESULT")
            print("-" * 40)
            print(f"Object:      {row.pair_name}")
            print(f"WDS:         {row.wds_id}")
            print(f"PA:          {pa:.2f} deg")
            print(f"Separation:  {rho:.2f} arcsec")
            print(f"Mag1 (WDS): {row.mag_a:.2f} mag")
            print(f"Mag2 (calc): {row.mag_a + dmag:.2f} mag")
            print(f"Delta mag:   {abs(dmag):.2f} mag")
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
            print("\nMEASUREMENT REJECTED")
            print("-" * 40)
            print(str(exc))

        while True:
            print("\nWhat would you like to do?")
            print("  1 - New measurement")
            print("  2 - Exit")
            choice = input("Selection: ").strip()
            if choice == "1":
                print()
                break
            if choice == "2":
                return 0
            print("Invalid selection.")



if __name__ == "__main__":
    raise SystemExit(main())
