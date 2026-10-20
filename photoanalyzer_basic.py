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
VERSION = "1.2i"
LAST_QUALITY = "GOOD"
LAST_WARNINGS = []
LAST_WIDE_DIAGNOSTICS = None
LAST_WING_TESTS = None

MIN_BASIC_SEPARATION_ARCSEC = 10.0
MAX_BASIC_MAG = 14.0
DAO_SEPARATION_FWHM_LIMIT = 3.0  # actual centroid separation / FWHM
# Legacy GUI compatibility only; NOT used to select the astrometric method.
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



def identify_component_at_wds(data, sources, expected_x, expected_y,
                              fwhm_guess, label, search_radius,
                              min_snr=6.0):
    """Shared component identification for close and wide systems.

    DAO is preferred when its centroid is genuinely near the expected
    component. If DAO misses the star, inspect a local peak. This is an
    *identification* only, never a measurement. Return source plus diagnostic.
    """
    x = float(np.asarray(expected_x).item())
    y = float(np.asarray(expected_y).item())
    ny, nx = data.shape
    radius = float(search_radius)
    cx, cy = int(round(x)), int(round(y))
    margin = max(11, int(math.ceil(radius + 9.0)))
    if cx-margin < 0 or cy-margin < 0 or cx+margin >= nx or cy+margin >= ny:
        raise ValueError(f"{label}: too close to FITS boundary")
    tile = data[cy-margin:cy+margin+1, cx-margin:cx+margin+1]
    yy, xx = np.mgrid[cy-margin:cy+margin+1, cx-margin:cx+margin+1]
    rr = np.hypot(xx-x, yy-y)
    sky = tile[(rr >= radius+4.0) & (rr <= radius+8.0) & np.isfinite(tile)]
    if sky.size < 25:
        raise ValueError(f"{label}: not enough sky pixels")
    bg = float(np.median(sky))
    mad = float(np.median(np.abs(sky-bg)))
    # Bright/saturated nearby stars can contaminate the background annulus.
    # Standard deviation then measures stellar contamination, not sky noise.
    # Use the robust MAD estimate, provided enough uncontaminated pixels remain.
    robust_noise = max(1.4826 * mad, 1.0)
    uncontaminated = sky[np.abs(sky-bg) <= 3.0*robust_noise]
    if uncontaminated.size < max(25, int(0.30 * sky.size)):
        raise ValueError(f"{label}: local background is too contaminated")
    bg = float(np.median(uncontaminated))
    noise = max(1.4826 * float(np.median(np.abs(uncontaminated-bg))), 1.0)
    region = (rr <= radius) & np.isfinite(tile)
    if not np.any(region):
        raise ValueError(f"{label}: no valid pixels at expected position")
    peak = float(np.max(tile[region]))
    snr = (peak-bg)/noise
    if snr < min_snr:
        raise ValueError(f"{label}: local source SNR={snr:.2f} (< {min_snr:.1f})")
    inds = np.where(region & (tile == peak))
    k = int(np.argmin(rr[inds]))
    px, py = float(xx[inds][k]), float(yy[inds][k])
    if sources is not None and len(sources):
        sx = np.asarray(sources['xcentroid'],dtype=float)
        sy = np.asarray(sources['ycentroid'],dtype=float)
        dist = np.hypot(sx-px,sy-py)
        j = int(np.argmin(dist))
        if dist[j] <= 2.0 and math.hypot(sx[j]-x,sy[j]-y) <= radius:
            return float(sx[j]), float(sy[j]), 'DAO', snr
    return px, py, 'local peak', snr


def require_two_independent_sources(
    sources,
    x_a_exp: float,
    y_a_exp: float,
    x_b_exp: float,
    y_b_exp: float,
    fwhm_guess: float,
    data=None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find two real peaks before joint PSF fitting.

    A global DAO omission alone is not a rejection. When necessary, inspect
    local peaks close to each WDS position using the wide-mode SNR check.
    Keep the distinct-source and catalog-separation sanity checks.
    """
    expected_sep_pix = math.hypot(x_a_exp - x_b_exp, y_a_exp - y_b_exp)
    if expected_sep_pix <= 0:
        raise ValueError("Invalid expected component separation.")

    # Avoid excessively large DAO radii taking a neighbouring star as a
    # component, especially in crowded open-cluster fields.
    radius = min(max(2.0, 0.30 * expected_sep_pix), 5.0)
    dao_a = nearest_dao_position(sources, x_a_exp, y_a_exp, radius)
    dao_b = nearest_dao_position(sources, x_b_exp, y_b_exp, radius)

    def distinct(a, b):
        if a is None or b is None:
            return False
        sep = math.hypot(a[0] - b[0], a[1] - b[1])
        return sep >= max(1.5, 0.60 * float(fwhm_guess))

    if not distinct(dao_a, dao_b):
        if data is None:
            raise ValueError("Two independent components were not detected, and local search is unavailable.")
        # In the fallback, inspect the *same* WDS positions, not arbitrary
        # nearby detections. A local maximum still needs SNR >= 6 (the
        # existing wide-pair criterion), and must be a distinct source.
        local_radius = min(3.0, max(1.5, 0.30 * expected_sep_pix))
        try:
            la = locate_wide_component(data, sources, x_a_exp, y_a_exp,
                                       fwhm_guess, "A", search_radius=local_radius)
            lb = locate_wide_component(data, sources, x_b_exp, y_b_exp,
                                       fwhm_guess, "B", search_radius=local_radius)
        except ValueError as exc:
            raise ValueError(f"Independent source detection failed; local check: {exc}") from exc
        dao_a, dao_b = la[:2], lb[:2]
        print(f"Close pair: local-source fallback A={la[2]}, B={lb[2]}")

    if not distinct(dao_a, dao_b):
        raise ValueError("The two components cannot be resolved as separate stars in the stacked FITS image.")

    measured_sep_pix = math.hypot(dao_a[0] - dao_b[0], dao_a[1] - dao_b[1])
    ratio = measured_sep_pix / expected_sep_pix
    # Coarse pre-fit guard only. The definitive +/-25% comparison occurs
    # after PSF fitting and is deliberately unchanged.
    if not (0.60 <= ratio <= 1.40):
        raise ValueError("Detected component separation is inconsistent with the WDS positions.")

    return dao_a, dao_b


def component_search_diagnostic(data, wcs, row, sources, fwhm_guess):
    """Non-invasive component diagnostics; never changes accepted measurements."""
    expected_a = row.coord
    expected_b = expected_a.directional_offset_by(
        position_angle=row.pa_last*u.deg, separation=row.sep_last*u.arcsec)
    lines = []
    sep = float(row.sep_last)
    for label, coord in [('first',expected_a),('second',expected_b)]:
        x,y = skycoord_to_pixel(coord, wcs, origin=0, mode='all')
        lines.append(f"{label}: WDS predicted pixel=({float(x):.3f}, {float(y):.3f})")
        for r in (3.0, 6.0):
            try:
                px,py,method,snr=identify_component_at_wds(
                    data,sources,x,y,fwhm_guess,label,r,min_snr=-1e9)
                lines.append(f"  radius={r:.1f}px peak=({px:.2f},{py:.2f}) "
                             f"method={method} SNR={snr:.2f}")
            except Exception as exc:
                lines.append(f"  radius={r:.1f}px failed: {exc}")
    return '\n'.join(lines)


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

    # Elsőként az eredeti, két független csillagcsúcsra épülő módszer.
    # Összeérő PSF esetén a B csúcsa elmaradhat: ekkor szigorúan
    # WDS-közeli kezdőpozícióból végzünk KÉT komponensű illesztést.
    blended_fallback = False
    legacy_detected = False
    # Regression-first: preserve the original v1.1e DAO association if it
    # really identifies TWO different stars with a plausible separation.
    # This path is intentionally not affected by the newer wing heuristic.
    legacy_radius = max(2.5 * float(fwhm_guess), 5.0)
    legacy_a = nearest_dao_position(sources, x_a_exp, y_a_exp, legacy_radius)
    legacy_b = nearest_dao_position(sources, x_b_exp, y_b_exp, legacy_radius)
    if legacy_a is not None and legacy_b is not None:
        legacy_gap = math.hypot(legacy_a[0]-legacy_b[0],
                                legacy_a[1]-legacy_b[1])
        wds_gap = math.hypot(x_a_exp-x_b_exp, y_a_exp-y_b_exp)
        if legacy_gap >= max(1.5, 0.75*float(fwhm_guess)) and wds_gap > 0 \
                and 0.70 <= legacy_gap/wds_gap <= 1.30:
            legacy_detected = True
            dao_a, dao_b = legacy_a, legacy_b
            print('Close pair: original v1.1e independent-DAO path selected.')
    if not legacy_detected:
      try:
        dao_a, dao_b = require_two_independent_sources(
            data=data, sources=sources,
            x_a_exp=x_a_exp, y_a_exp=y_a_exp,
            x_b_exp=x_b_exp, y_b_exp=y_b_exp,
            fwhm_guess=fwhm_guess,
        )
      except ValueError as exc:
          blended_fallback = True
          print(f"Blended pair: independent peaks unavailable ({exc}); "
                "trying WDS-guided joint PSF fit.")
          # A DAO A-komponens pozícióját csak nagyon közeli egyezésnél használjuk.
          # A B-t nem helyettesítjük az A csúcsával.
          a_radius = min(3.0, max(1.5, 0.25 * math.hypot(
              x_b_exp - x_a_exp, y_b_exp - y_a_exp)))
          dao_a = nearest_dao_position(sources, x_a_exp, y_a_exp, a_radius)
          if dao_a is None:
              dao_a = (float(x_a_exp), float(y_a_exp))
          dao_b = (float(x_b_exp), float(y_b_exp))

    # Regression guard: for a tight pair the bright primary's wing can be
    # misidentified as the secondary even when the local SNR is huge.
    # Prefer catalogue-guided initialisation to a demonstrably wrong peak.
    # This changes only starting coordinates, never the reported measurement.
    predicted_gap = math.hypot(x_b_exp-x_a_exp, y_b_exp-y_a_exp)
    detected_gap = math.hypot(dao_b[0]-dao_a[0], dao_b[1]-dao_a[1])
    b_offset = math.hypot(dao_b[0]-x_b_exp, dao_b[1]-y_b_exp)
    if (not legacy_detected and not blended_fallback and predicted_gap <= 2.5*float(fwhm_guess)
            and (detected_gap < 0.80*predicted_gap
                 or b_offset > min(1.5, 0.25*predicted_gap))):
        print('Close pair: suspect B peak in primary PSF wing; '
              'using WDS-guided joint fit.')
        blended_fallback = True
        dao_b = (float(x_b_exp), float(y_b_exp))

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

    # WDS-vezérelt módban a B csillag nem vándorolhat át egy másik csúcsra.
    shift_b = min(MAX_POSITION_SHIFT_PIX, 1.5) if blended_fallback else MAX_POSITION_SHIFT_PIX
    lower = np.array([
        x_a0 - MAX_POSITION_SHIFT_PIX,
        y_a0 - MAX_POSITION_SHIFT_PIX,
        0.0,
        x_b0 - shift_b,
        y_b0 - shift_b,
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
        x_b0 + shift_b,
        y_b0 + shift_b,
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

    # A 10-paraméteres közös PSF illesztés zajos Seestar-képeken
    # nem mindig teljesíti az indokolatlanul szigorú 1e-10 toleranciát.
    # Először a bevált kezdőértékekkel próbálunk, majd szükség esetén
    # eltérő kezdő PSF-szélességekkel. Sikertelen fitet NEM fogadunk el.
    attempts = []
    for width_scale in (1.0, 0.75, 1.4):
        initial = p0.copy()
        initial[6:8] = np.clip(initial[6:8] * width_scale,
                               lower[6:8] + 1e-5, upper[6:8] - 1e-5)
        try:
            candidate = least_squares(
                residuals, initial, bounds=(lower, upper),
                loss="soft_l1", f_scale=1.0,
                max_nfev=4500, xtol=1e-7, ftol=1e-7, gtol=1e-7,
            )
        except (ValueError, FloatingPointError) as exc:
            attempts.append(str(exc))
            continue
        if candidate.success and np.all(np.isfinite(candidate.x)):
            result = candidate
            break
        attempts.append(str(candidate.message))
    else:
        raise RuntimeError("A PSF-illesztés több kezdőértékkel sem konvergált: "
                           + "; ".join(attempts))

    # A két komponens egymásra csúszását ne próbáljuk WDS-értékre
    # kényszerített szeparációval javítani. Csak az első fit összeomlása
    # esetén próbálunk új, önálló kezdőpozíciókhoz kötött illesztéseket.
    fitted_gap = math.hypot(result.x[3]-result.x[0], result.x[4]-result.x[1])
    independent_gap = math.hypot(x_b0-x_a0, y_b0-y_a0)
    collapse_limit = max(1.0, 0.45 * independent_gap)
    if fitted_gap < collapse_limit and independent_gap > 2.0:
        print(f'Close pair: collapsed initial PSF solution '
              f'({fitted_gap:.2f} px); attempting independent-centroid refits.')
        # A kezdeti koordináták körüli külön korlátok nem engedik, hogy a
        # két pozíció ugyanabba a lokális maximumba vándoroljon. A mérés
        # továbbra is szabad PSF-centroidokból történik, nem WDS-ből.
        max_safe_shift = min(1.15, 0.25 * independent_gap)
        tight_low = lower.copy()
        tight_high = upper.copy()
        for k, start in ((0,x_a0),(1,y_a0),(3,x_b0),(4,y_b0)):
            tight_low[k] = start - max_safe_shift
            tight_high[k] = start + max_safe_shift
        candidates = []
        for width_scale in (0.75, 1.0, 1.4):
            for secondary_scale in (0.4, 1.0, 1.8):
                start = p0.copy()
                start[6:8] = np.clip(sigma0*width_scale,
                                      tight_low[6:8]+1e-5, tight_high[6:8]-1e-5)
                start[5] = np.clip(amp_b0*secondary_scale,
                                    tight_low[5]+1e-5, tight_high[5]-1e-5)
                start = np.clip(start, tight_low+1e-5, tight_high-1e-5)
                try:
                    trial = least_squares(
                        residuals, start, bounds=(tight_low,tight_high),
                        loss='soft_l1', f_scale=1.0, max_nfev=4500,
                        xtol=1e-7, ftol=1e-7, gtol=1e-7)
                except (ValueError, FloatingPointError):
                    continue
                if not trial.success or not np.all(np.isfinite(trial.x)):
                    continue
                gap = math.hypot(trial.x[3]-trial.x[0], trial.x[4]-trial.x[1])
                if gap < collapse_limit or trial.x[2] <= 0 or trial.x[5] <= 0:
                    continue
                # Határon megtapadó centroid nem megbízható mérés.
                if any(min(trial.x[k]-tight_low[k],tight_high[k]-trial.x[k]) < 0.03
                       for k in (0,1,3,4)):
                    continue
                r = residuals(trial.x)
                score = float(np.sum(2.0*(np.sqrt(1.0+r*r)-1.0)))
                candidates.append((score,trial))
        if not candidates:
            raise ValueError('The two fitted centroids collapsed; constrained refits '
                             'did not yield a stable, separated solution.')
        candidates.sort(key=lambda item:item[0])
        result = candidates[0][1]
        print(f'Close pair: separated PSF refit found '
              f'({math.hypot(result.x[3]-result.x[0],result.x[4]-result.x[1]):.2f} px).')

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

    if blended_fallback:
        # A két-Gauss illesztés matematikailag konvergálhat egyetlen csillagra
        # is. Önálló egy-Gauss modellt is illesztünk UGYANAZOKRA a pixelekre;
        # csak a kimutathatóan jobb kettős modellt fogadjuk el.
        # A teszt nem módosítja a PSF-alapú PA/rho értékeket.
        def single_residual(params):
            x, y, amp, sig_x, sig_y, theta, bg = params
            ct, st = np.cos(theta), np.sin(theta)
            dx, dy = xx - x, yy - y
            xr, yr = ct*dx + st*dy, -st*dx + ct*dy
            model = bg + amp*np.exp(-0.5*((xr/sig_x)**2 + (yr/sig_y)**2))
            return ((model-cut)/noise).ravel()

        single_start = np.array([x_a0, y_a0, amp_a0,
                                 sigma0, sigma0, 0.0, background], dtype=float)
        single_low = np.array([x_a0-4, y_a0-4, 0, .45, .45,
                               -math.pi/2, float(np.nanmin(cut))-5*noise])
        single_high = np.array([x_a0+4, y_a0+4,
                                max(float(np.nanmax(cut)-background)*5, amp_a0*5),
                                max(8.,2*fwhm_guess), max(8.,2*fwhm_guess),
                                math.pi/2, float(np.nanmax(cut))])
        single_start = np.clip(single_start, single_low+1e-5, single_high-1e-5)
        single_fit = least_squares(single_residual, single_start,
                                   bounds=(single_low, single_high),
                                   loss='soft_l1', f_scale=1.0,
                                   max_nfev=4500, xtol=1e-7, ftol=1e-7, gtol=1e-7)
        if not single_fit.success or not np.all(np.isfinite(single_fit.x)):
            raise ValueError('Blended pair: single-star comparison fit failed.')
        # Only assess residuals in the two-star vicinity, not empty sky pixels.
        neighbourhood = ((xx-x_a_exp)**2+(yy-y_a_exp)**2 <= (2.2*fwhm_guess)**2) | \
                        ((xx-x_b_exp)**2+(yy-y_b_exp)**2 <= (2.2*fwhm_guess)**2)
        resid_single = single_residual(single_fit.x).reshape(cut.shape)[neighbourhood]
        resid_double = residuals(result.x).reshape(cut.shape)[neighbourhood]
        sse_single = float(np.mean(np.minimum(resid_single**2, 100.)))
        sse_double = float(np.mean(np.minimum(resid_double**2, 100.)))
        improvement = (sse_single-sse_double) / max(sse_single, 1e-9)
        b_snr = float(amp_b/noise)
        print(f'Blended pair validation: improvement={improvement:.1%}, '
              f'B peak/noise={b_snr:.1f}')
        if improvement < 0.10 or b_snr < 5.0:
            raise ValueError('Blended pair: second component not confirmed by the '
                             f'two-star model (improvement {improvement:.1%}, '
                             f'B peak/noise {b_snr:.1f}).')
        if math.hypot(x_b-x_b_exp, y_b-y_b_exp) > 2.0:
            raise ValueError('Blended pair: fitted B position too far from WDS.')

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


def locate_wide_component(data, sources, expected_x, expected_y, fwhm_guess,
                          label, search_radius=3.0):
    """Wide mode uses the same WDS-guided source identifier as close mode."""
    x, y, method, snr = identify_component_at_wds(
        data, sources, expected_x, expected_y, fwhm_guess,
        label, search_radius, min_snr=6.0)
    print(f"{label} identification: {method}; local peak SNR={snr:.2f}")
    return x, y, method


def _dao_candidates(sources, x, y, radius, forbidden=None):
    """Valid DAO centroids in a bounded WDS neighbourhood."""
    if sources is None or len(sources) == 0:
        return []
    xs = np.asarray(sources['xcentroid'], dtype=float)
    ys = np.asarray(sources['ycentroid'], dtype=float)
    flux_col = 'flux' if 'flux' in sources.colnames else None
    out = []
    for i, (sx, sy) in enumerate(zip(xs, ys)):
        if not np.isfinite(sx) or not np.isfinite(sy):
            continue
        d = math.hypot(sx-x, sy-y)
        if d > radius:
            continue
        if forbidden is not None and math.hypot(sx-forbidden[0], sy-forbidden[1]) < 2.0:
            continue
        flux = float(sources[flux_col][i]) if flux_col else float('nan')
        if flux_col and (not np.isfinite(flux) or flux <= 0):
            continue
        out.append((float(sx), float(sy), d, flux))
    return sorted(out, key=lambda v: v[2])


def _choose_dao_candidate(candidates, label):
    """Avoid automatic false identification in crowded fields."""
    if not candidates:
        raise ValueError(f'{label}: no DAO centroid in the search area.')
    chosen = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        # Equally plausible geometrical matches must not silently become GOOD.
        if second[2] - chosen[2] < max(1.5, 0.15*max(chosen[2], 1.0)):
            raise ValueError(f'{label}: ambiguous DAO identification; two candidate stars at '
                             f'{chosen[2]:.2f} and {second[2]:.2f} px from WDS.')
    return chosen


def _local_dao_candidates(data, expected, radius, fwhm, forbidden=None):
    """Second DAO search in a local cutout; output stays DAO centroids only.

    Used exclusively when full-frame DAO supplied no candidate. Re-estimate
    local sky/noise and try a few sensible widths; never use a raw pixel peak
    or a geometric/PSF fit as an astrometric measurement.
    """
    x, y = map(float, expected)
    ny, nx = data.shape
    half = int(math.ceil(max(radius + 3.0 * fwhm, 5.0 * fwhm, 16.0)))
    xmin = int(round(x)) - half
    xmax = int(round(x)) + half + 1
    ymin = int(round(y)) - half
    ymax = int(round(y)) + half + 1
    if xmin < 0 or ymin < 0 or xmax > nx or ymax > ny:
        raise ValueError('local DAO cutout extends beyond image boundary')
    tile = np.asarray(data[ymin:ymax, xmin:xmax], dtype=float)
    if not np.all(np.isfinite(tile)):
        raise ValueError('local DAO cutout contains invalid pixels')
    # Estimate sky away from the expected stellar core. This prevents a bright
    # star from inflating the source threshold and hiding the intended source.
    yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
    outer = np.hypot(xx - x, yy - y) >= max(2.5 * fwhm, radius + fwhm)
    sky = tile[outer]
    if sky.size < 40:
        sky = tile.ravel()
    bg = float(np.median(sky))
    mad = float(np.median(np.abs(sky - bg)))
    noise = max(1.4826 * mad, 1.0)
    found = []
    # Try the full-frame FWHM first, then a modestly narrower/broader value.
    for width in dict.fromkeys([float(fwhm), max(2., .75 * float(fwhm)),
                                min(8., 1.25 * float(fwhm))]):
        finder = DAOStarFinder(fwhm=width, threshold=6.0 * noise,
                               exclude_border=False)
        local = finder(tile - bg)
        if local is None or len(local) == 0:
            continue
        for row in local:
            sx = float(row['xcentroid']) + xmin
            sy = float(row['ycentroid']) + ymin
            dist = math.hypot(sx - x, sy - y)
            if not np.isfinite(dist) or dist > radius:
                continue
            if forbidden is not None and math.hypot(sx - forbidden[0], sy - forbidden[1]) < max(2.0, 1.2 * fwhm):
                continue
            flux = float(row['flux']) if 'flux' in local.colnames else float('nan')
            if 'flux' in local.colnames and (not np.isfinite(flux) or flux <= 0):
                continue
            # Same star found with multiple FWHM settings: keep its first
            # detection rather than treating it as a competing physical star.
            if any(math.hypot(sx - a[0], sy - a[1]) < 1.0 for a in found):
                continue
            found.append((sx, sy, dist, flux))
    if not found:
        # Saturated or asymmetric profiles can fail DAO's default sharpness
        # and roundness cuts. Retry only on the SAME local image, with wider
        # morphology limits. Astrometry still uses DAO's x/y centroids.
        for width in (max(2.0, float(fwhm)), max(2.0, .75*float(fwhm))):
            for sigmas in (6.0, 3.0):
                finder = DAOStarFinder(
                    fwhm=width, threshold=sigmas*noise,
                    sharplo=-2.0, sharphi=2.0,
                    roundlo=-2.0, roundhi=2.0,
                    exclude_border=False,
                )
                local = finder(tile-bg)
                if local is None:
                    continue
                for row in local:
                    sx = float(row['xcentroid']) + xmin
                    sy = float(row['ycentroid']) + ymin
                    dist = math.hypot(sx-x, sy-y)
                    if not np.isfinite(dist) or dist > radius:
                        continue
                    if forbidden is not None and math.hypot(
                            sx-forbidden[0], sy-forbidden[1]) < max(2.0, 1.2*fwhm):
                        continue
                    flux = float(row['flux']) if 'flux' in local.colnames else float('nan')
                    if 'flux' in local.colnames and (not np.isfinite(flux) or flux <= 0):
                        continue
                    if any(math.hypot(sx-a[0], sy-a[1]) < 1.0 for a in found):
                        continue
                    found.append((sx, sy, dist, flux))
            if found:
                print(f'Local DAO: extended morphology search found {len(found)} candidate(s)')
                break
    if not found:
        print(f'Local DAO: zero accepted centroids at ({x:.2f},{y:.2f}), '
              f'radius={radius:.1f}px, local background={bg:.1f}, noise={noise:.2f}')
    return sorted(found, key=lambda item: item[2])


def _wide_measure_component(data, sources, expected, radius, fwhm, label, other=None):
    """Wide pair astrometry: DAO centroid only; WDS is a search hint.

    Nearby local pixels may help disambiguate DAO detections, but they are
    never used as the reported centroid. Saturation is not an exclusion.
    """
    x, y = (float(v) for v in expected)
    candidates = _dao_candidates(sources, x, y, radius, forbidden=other)
    origin = 'global DAO'
    if not candidates:
        candidates = _local_dao_candidates(data, (x, y), radius, fwhm,
                                           forbidden=other)
        origin = 'local DAO'
        if candidates:
            print(f'{label}: local DAO found {len(candidates)} candidate(s)')
    if not candidates:
        raise ValueError(f'{label}: no DAO centroid inside {radius:.1f}px search region (global and local DAO failed)')

    chosen = candidates[0]
    if len(candidates) > 1:
        rival = candidates[1]
        if rival[2] - chosen[2] < max(1.5, 0.15*max(chosen[2], 1.0)):
            # Look for a real core peak near the predicted position only to
            # identify which DAO centroid corresponds to the intended star.
            ny, nx = data.shape
            r = min(3.0, max(2.0, 0.75*float(fwhm)))
            xm, ym = int(round(x)), int(round(y))
            reach = int(math.ceil(r))
            if xm-reach < 0 or ym-reach < 0 or xm+reach >= nx or ym+reach >= ny:
                raise ValueError(f'{label}: too close to FITS boundary')
            yy, xx = np.mgrid[ym-reach:ym+reach+1, xm-reach:xm+reach+1]
            tile = data[ym-reach:ym+reach+1, xm-reach:xm+reach+1]
            disk = (xx-x)**2+(yy-y)**2 <= r*r
            disk &= np.isfinite(tile)
            if not np.any(disk):
                raise ValueError(f'{label}: ambiguous DAO identification (no local pixels)')
            max_value = np.max(tile[disk])
            ys, xs = np.where(disk & (tile == max_value))
            k = int(np.argmin((xx[ys,xs]-x)**2+(yy[ys,xs]-y)**2))
            anchor = (float(xx[ys[k],xs[k]]), float(yy[ys[k],xs[k]]))
            sorted_by_anchor = sorted(candidates, key=lambda c:math.hypot(c[0]-anchor[0],c[1]-anchor[1]))
            # Do not force a false GOOD identification if several DAO peaks
            # remain similarly plausible relative to the same local source.
            da = math.hypot(sorted_by_anchor[0][0]-anchor[0],sorted_by_anchor[0][1]-anchor[1])
            db = math.hypot(sorted_by_anchor[1][0]-anchor[0],sorted_by_anchor[1][1]-anchor[1])
            if db-da < 0.5:
                raise ValueError(f'{label}: ambiguous DAO identification; nearest WDS offsets '
                                 f'{chosen[2]:.2f}/{rival[2]:.2f}px')
            chosen = sorted_by_anchor[0]
            print(f'{label}: DAO association checked against local peak (no centroid fallback)')

    dx, dy, offset = chosen[:3]
    if other is not None and math.hypot(dx-other[0],dy-other[1]) < max(2.0,1.2*fwhm):
        raise ValueError(f'{label}: DAO centroid belongs to the other component')
    print(f'{label}: {origin} centroid=({dx:.3f}, {dy:.3f}); WDS offset={offset:.2f}px')
    return dx, dy, ('DAO-local' if origin == 'local DAO' else 'DAO'), offset


def measure_wide_pair(data: np.ndarray, wcs: WCS, row: WDSRow, sources,
                      fwhm_guess: float) -> tuple:
    """Wide pairs: only DAO centroid astrometry (no PSF or geometry)."""
    global LAST_WIDE_DIAGNOSTICS, LAST_QUALITY, LAST_WARNINGS, LAST_WING_TESTS
    LAST_WING_TESTS=None
    expected_a=row.coord
    expected_b=expected_a.directional_offset_by(
        position_angle=row.pa_last*u.deg,separation=row.sep_last*u.arcsec)
    # Astropy can return 0-D or length-one NumPy arrays from a scalar SkyCoord.
    # Convert to genuine Python floats before passing coordinates to round().
    xa_raw, ya_raw = skycoord_to_pixel(expected_a, wcs, origin=0, mode='all')
    xb_raw, yb_raw = skycoord_to_pixel(expected_b, wcs, origin=0, mode='all')
    def scalar_pixel(value):
        arr = np.asarray(value, dtype=float)
        if arr.size != 1:
            raise ValueError(f'Expected a scalar WDS pixel coordinate, got shape {arr.shape}')
        return float(arr.item())
    xa, ya = scalar_pixel(xa_raw), scalar_pixel(ya_raw)
    xb, yb = scalar_pixel(xb_raw), scalar_pixel(yb_raw)
    if not all(np.isfinite([xa,ya,xb,yb])):
        raise ValueError('Invalid WDS pixel coordinates')
    ny,nx=data.shape
    pred_gap=math.hypot(xa-xb,ya-yb)
    rad_a=12.0  # Allow moderate A-component proper-motion offsets too.
    rad_b=min(30.,max(12.,.35*pred_gap))
    margin=max(11,int(math.ceil(2.8*max(1.5,fwhm_guess))))
    for label,x,y in (('A',xa,ya),('B/C',xb,yb)):
        if x-margin<0 or y-margin<0 or x+margin>=nx or y+margin>=ny:
            raise ValueError(f'{label}: too close to FITS boundary')
    x_a,y_a,method_a,offset_a=_wide_measure_component(
        data,sources,(xa,ya),rad_a,fwhm_guess,'A')
    x_b,y_b,method_b,offset_b=_wide_measure_component(
        data,sources,(xb,yb),rad_b,fwhm_guess,'B/C',other=(x_a,y_a))
    if math.hypot(x_b-x_a,y_b-y_a) < max(2.,1.2*fwhm_guess):
        raise ValueError('Wide pair: components not independently resolved')
    ca=pixel_to_skycoord(x_a,y_a,wcs,origin=0,mode='all')
    cb=pixel_to_skycoord(x_b,y_b,wcs,origin=0,mode='all')
    pa=float(ca.position_angle(cb).to_value(u.deg)%360.)
    rho=float(ca.separation(cb).to_value(u.arcsec))
    pa_diff=abs((pa-row.pa_last+180.)%360.-180.)
    sep_diff=100.*(rho/row.sep_last-1.) if row.sep_last>0 else float('nan')
    if (method_a!='DAO' or method_b!='DAO' or offset_a>6 or offset_b>12
        or pa_diff>8 or abs(sep_diff)>25):
        LAST_QUALITY='CHECK'
        LAST_WARNINGS.append(f'Method A={method_a}, B={method_b}; WDS offsets '
                             f'{offset_a:.2f}/{offset_b:.2f}px; '
                             f'dPA={pa_diff:.2f}deg; dSep={sep_diff:+.1f}%')
    LAST_WIDE_DIAGNOSTICS={
        'A':(xa,ya,x_a,y_a,x_a,y_a,method_a,method_a),
        'B':(xb,yb,x_b,y_b,x_b,y_b,method_b,method_b)}
    print(f'Wide centroids: A={method_a} ({x_a:.3f},{y_a:.3f}); '
          f'B={method_b} ({x_b:.3f},{y_b:.3f})')
    return (pa,rho,0.,x_a,y_a,x_b,y_b,float('nan'),float(fwhm_guess),
            float(ca.ra.deg),float(ca.dec.deg),float(cb.ra.deg),float(cb.dec.deg))


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

        # Szubpixel apertúra: 1.5 px sugárnál a 8 teljes pixeles minimum
        # geometriailag nem mindig teljesíthető. A pixel lefedettségét
        # 5x5 mintaponttal közelítjük, azonos súlyozással A és B esetén.
        samples = 5
        aper_weight = np.zeros_like(rr, dtype=float)
        for oy in (np.arange(samples) + 0.5) / samples - 0.5:
            for ox in (np.arange(samples) + 0.5) / samples - 0.5:
                aper_weight += (np.hypot(xx + ox - x0, yy + oy - y0) <= r_ap)
        aper_weight /= samples * samples

        ann_mask = (rr >= r_in) & (rr <= r_out) & np.isfinite(sub)
        companion_exclusion = max(1.5 * r_ap, 1.5 * fwhm)
        ann_mask &= rr_other > companion_exclusion
        ann_vals = sub[ann_mask]
        effective_area = float(np.sum(aper_weight[np.isfinite(sub)]))
        if effective_area < 5.0 or len(ann_vals) < 20:
            raise ValueError(
                f"Insufficient usable aperture/sky pixels "
                f"(area={effective_area:.2f}, sky={len(ann_vals)})."
            )
        if np.any((aper_weight > 0) & ~np.isfinite(sub)):
            raise ValueError("Non-finite pixel inside photometric aperture.")

        bg = float(np.median(ann_vals))
        mad = float(np.median(np.abs(ann_vals - bg)))
        if mad > 0:
            sigma = 1.4826 * mad
            good = np.abs(ann_vals - bg) <= 3.0 * sigma
            if np.any(good):
                bg = float(np.median(ann_vals[good]))

        flux = float(np.sum(np.where(aper_weight > 0, aper_weight * (sub - bg), 0.0)))
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
                     pa: float, rho: float, dmag: float,
                     quality: str = "GOOD", warnings=None) -> Path:
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
        f"Quality: {quality}\n"
        + ("\n".join("Warning: " + item for item in (warnings or [])) + "\n" if warnings else ""),
        encoding="utf-8",
    )
    return out


def main() -> int:
    global LAST_WIDE_DIAGNOSTICS, LAST_WING_TESTS, LAST_QUALITY, LAST_WARNINGS
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
        f'Mode selection: DAO when separation / FWHM >= {DAO_SEPARATION_FWHM_LIMIT:.1f}; '
        'otherwise original joint PSF fit'
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
            LAST_QUALITY = "GOOD"
            LAST_WARNINGS = []
            header, data, wcs = read_header_and_data(fit_path)
            fwhm_guess, sources = estimate_fwhm_and_sources(data)
            print(f"DAO elődetektálás: {len(sources)} forrás, FWHM~{fwhm_guess:.1f} px")

            # Choose the astrometric method from the *observed* DAO separation,
            # not the historical WDS separation. DAO never uses PSF/geometry.
            # PSF remains the original two-star model for genuinely close stars.
            selected_mode = 'DAO'
            dao_error = None
            try:
                dao_result = measure_wide_pair(
                    data=data, wcs=wcs, row=row, sources=sources,
                    fwhm_guess=fwhm_guess,
                )
                dao_gap = math.hypot(dao_result[5]-dao_result[3],
                                     dao_result[6]-dao_result[4])
                dao_k = dao_gap / max(float(fwhm_guess), 1e-6)
                print(f'DAO separation={dao_gap:.3f} px; FWHM={fwhm_guess:.3f} px; K={dao_k:.3f}')
                if dao_k >= DAO_SEPARATION_FWHM_LIMIT:
                    print('Measurement mode: DAO centroid (independent stars).')
                    result_values = dao_result
                else:
                    selected_mode = 'PSF'
                    print(f'K={dao_k:.3f} < {DAO_SEPARATION_FWHM_LIMIT:.1f}: '
                          'switching to original joint PSF fit.')
            except ValueError as exc:
                dao_error = str(exc)
                # An absent/ambiguous DAO source does not mean the star is blended.
                # Only attempt PSF if the expected pixel spacing supports a
                # genuinely close system. WDS provides search positions, not PA/rho.
                expected_b = row.coord.directional_offset_by(
                    position_angle=row.pa_last*u.deg,
                    separation=row.sep_last*u.arcsec)
                exa, eya = skycoord_to_pixel(row.coord, wcs, origin=0, mode='all')
                exb, eyb = skycoord_to_pixel(expected_b, wcs, origin=0, mode='all')
                expected_gap = math.hypot(float(np.asarray(exa).item())-float(np.asarray(exb).item()),
                                          float(np.asarray(eya).item())-float(np.asarray(eyb).item()))
                if expected_gap / max(float(fwhm_guess), 1e-6) >= DAO_SEPARATION_FWHM_LIMIT:
                    raise ValueError('DAO identification failed for an apparently separated pair: '
                                     + dao_error) from exc
                selected_mode = 'PSF'
                print('DAO identification unavailable for a close pair: '
                      f'{dao_error}; attempting original joint PSF fit.')
            if selected_mode == 'PSF':
                # The existing PSF engine is unchanged.
                LAST_WIDE_DIAGNOSTICS = None
                LAST_WING_TESTS = None
                LAST_QUALITY = 'GOOD'
                LAST_WARNINGS = []
                result_values = fit_pair_on_frame(
                    data=data, wcs=wcs, row=row, sources=sources,
                    fwhm_guess=fwhm_guess,
                )
            (
                pa, rho, dmag_psf,
                x_a, y_a, x_b, y_b,
                fit_rms, fitted_fwhm,
                ra_a, dec_a, ra_b, dec_b,
            ) = result_values

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
                         f"Mode: {selected_mode}",
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
                            f"{label}: measurement_method = {psf}",
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
                lines.extend(["", f"PA_deg = {pa:.5f}", f"Separation_arcsec = {rho:.5f}",
                              f"Quality = {LAST_QUALITY}"] +
                             ["Warning: " + w for w in LAST_WARNINGS])
                diag_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                print(f"Centroid diagnostics saved: {diag_path}")
            except Exception as diag_exc:
                print(f"Centroid diagnostics not created: {diag_exc}")

            # WDS reference values are advisory; never force the measured PA/rho.
            sep_ratio = rho / row.sep_last if row.sep_last > 0 else np.nan
            if np.isfinite(sep_ratio) and not (0.75 <= sep_ratio <= 1.25):
                LAST_QUALITY = 'CHECK'
                msg = f'Measured separation {rho:.2f} arcsec vs WDS {row.sep_last:.2f} arcsec'
                if msg not in LAST_WARNINGS:
                    LAST_WARNINGS.append(msg)
            pa_delta = abs((pa - row.pa_last + 180.0)%360.0 - 180.0)
            if pa_delta > 8.0:
                LAST_QUALITY = 'CHECK'
                msg = f'Position angle differs from WDS by {pa_delta:.2f} deg'
                if msg not in LAST_WARNINGS:
                    LAST_WARNINGS.append(msg)

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
                f"Photometry: position-model Δmag={dmag_psf:.2f}, "
                f"AIJ-style aperture Δmag={dmag:.2f}, r={r_ap:.2f} px"
            )

            obs_time = observation_time_text(header)
            out = write_result_txt(fit_path, row, obs_time, pa, rho, dmag,
                                   quality=LAST_QUALITY, warnings=LAST_WARNINGS)

            print("\nRESULT")
            print("-" * 40)
            print(f"Object:      {row.pair_name}")
            print(f"WDS:         {row.wds_id}")
            print(f"PA:          {pa:.2f} deg")
            print(f"Separation:  {rho:.2f} arcsec")
            print(f"Mag1 (WDS): {row.mag_a:.2f} mag")
            print(f"Mag2 (calc): {row.mag_a + dmag:.2f} mag")
            print(f"Delta mag:   {abs(dmag):.2f} mag")
            print(f"Quality:     {LAST_QUALITY}")
            for warning in LAST_WARNINGS:
                print(f"Warning:     {warning}")
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
