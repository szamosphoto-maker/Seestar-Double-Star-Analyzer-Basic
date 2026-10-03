#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seestar Sunspot Counter 6.1.1

Pipeline:
1) Read Seestar white-light solar image
2) Detect solar disk
3) Solar orientation correction (observer location + parallactic angle + P-angle)
4) Limb-darkening / background normalization
5) Detect sunspots
6) Group sunspots into daily groups G1, G2, ...
7) Calculate detection quality on a 1-5 scale
8) Estimate facular area near the limb
9) Calculate raw Wolf number: R_raw = 10*g + s
10) Save to WORKDIR by default:
    - annotated image
    - spots.csv
    - groups.csv
    - daily_summary.csv

Dependencies:
    pip install numpy pandas opencv-python scipy

Notes:
- This script deliberately uses only the Seestar image for detection.
- No NOAA / SolarMonitor / KSO / SILSO information is used in detection.
- The script automatically finds the newest Seestar solar JPG/PNG in WORKDIR.
- The filename time is interpreted as local time in Europe/Budapest and converted to UTC.
- Alt-az field rotation is corrected from the observer coordinates and parallactic angle.
- A small fixed camera/mount offset can still be supplied with --camera-angle-deg if needed.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


# ---------------------------------------------------------------------
# User-editable work directory
# ---------------------------------------------------------------------

WORKDIR = Path(r"D:\Astro\Isotool\output")
LOCAL_TIMEZONE = "Europe/Budapest"

# Observer location in decimal degrees.
# North latitude and east longitude are positive.
# Change these once to your own observing site.
OBSERVER_LAT_DEG = 47.80
OBSERVER_LON_DEG = 18.76

# Input image and all output files are expected/written here by default.
# Change only this path later if you want to move the working folder.


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class Config:
    # Solar disk / normalization
    disk_threshold: int = 20
    inner_fraction: float = 0.99
    background_sigma_px: float = 45.0
    darkness_smooth_sigma_px: float = 1.2

    # Spot detection
    spot_region_threshold: float = 0.055
    spot_core_threshold: float = 0.080
    min_region_area_px: int = 7
    peak_window_px: int = 5
    min_peak_separation_px: float = 4.5

    # Grouping
    # 5.8.0: reduced from 0.13 to 0.10 to prevent neighbouring
    # active regions from merging through single-linkage chaining.
    group_distance_rsun: float = 0.10

    # Q>=3 detections are secure. Selected Q2 detections may later be
    # retained only when they are genuine group companions.
    count_quality_min: int = 3

    # 5.9.1: physical sunspot activity belt. No longitude restriction.
    max_abs_spot_lat_deg: float = 55.0

    # Only dedicated group/core candidates may be retained as Q2.
    q2_group_attach_min_px: float = 4.0
    q2_group_attach_max_px: float = 25.0

    # 5.9.2: second-look core search ONLY inside already established groups.
    # This does not lower the ordinary whole-disk sensitivity.
    group_core_search_radius_px: float = 9.0
    group_core_peak_threshold: float = 0.038
    group_core_prominence: float = 0.006
    group_core_min_sep_px: float = 2.0
    group_core_max_new_per_anchor: int = 2

    # Dedicated limb recovery pass
    limb_recovery_inner_rsun: float = 0.88
    limb_recovery_outer_rsun: float = 1.005
    limb_local_sigma_px: float = 9.0
    limb_core_threshold: float = 0.060
    limb_min_area_px: int = 3
    limb_merge_distance_px: float = 4.0


    # V4: radial limb-profile recovery ("bright limb sandwich")
    limb_profile_min_rsun: float = 0.97
    limb_profile_max_rsun: float = 1.005
    limb_profile_inner_px: int = 10
    limb_profile_outer_px: int = 10
    limb_profile_dark_threshold: float = 0.045
    limb_profile_bright_margin: float = 0.012

    # V5: extra strong-core splitter inside already detected limb regions.
    # This is meant to separate several real dark umbral/pore cores that
    # sit very close together without lowering the global spot threshold.
    strong_split_min_rsun: float = 0.72
    strong_split_peak_threshold: float = 0.064
    strong_split_prominence: float = 0.008
    strong_split_window_px: int = 3
    strong_split_background_sigma_px: float = 3.0
    strong_split_min_sep_px: float = 1.25

    # 5.1: recover faint but coherent companions close to an already
    # confirmed group. This does NOT scan the whole disk for weak dots.
    companion_search_radius_rsun: float = 0.085
    companion_region_threshold: float = 0.040
    companion_peak_threshold: float = 0.045
    companion_min_area_px: int = 3
    companion_min_sep_px: float = 1.5

    # 5.3: isolated faint-spot recovery.
    # This is deliberately restricted to the inner disk and requires
    # a compact, locally prominent feature that survives two smoothing scales.
    faint_single_max_rsun: float = 0.90
    faint_single_region_threshold: float = 0.036
    faint_single_peak_threshold: float = 0.046
    faint_single_prominence: float = 0.010
    faint_single_min_area_px: int = 3
    faint_single_max_area_px: int = 80
    faint_single_min_sep_px: float = 13.0
    faint_single_sigma_small_px: float = 1.0
    faint_single_sigma_large_px: float = 4.0
    faint_single_compactness_min: float = 0.12

    # -------------------------------------------------------------
    # 6.0.2 dedicated extreme-limb detector
    # -------------------------------------------------------------
    # This pass is independent of the inner-disk umbra detector.
    limb6_inner_rsun: float = 0.86
    limb6_outer_rsun: float = 0.999

    # Tangential comparison at nearly identical solar radius. This removes
    # ordinary limb darkening without comparing against the black sky.
    limb6_tangent_offsets_px: tuple[int, ...] = (6, 10, 14)
    limb6_min_local_contrast: float = 0.035
    limb6_min_area_px: int = 3
    limb6_max_area_px: int = 120
    limb6_min_compactness: float = 0.12
    limb6_min_sep_px: float = 5.0

    # 6.0.5 multi-scale tangential + radial-profile confirmation.
    # A weak candidate must remain dark at several tangential offsets.
    limb6_scale_contrast_min: float = 0.025
    limb6_min_confirmed_scales: int = 2

    # 6.0.5 radial-profile guard. Tangential contrast alone is too easily
    # triggered by faculae/granulation near the limb. For each candidate we
    # estimate the typical photospheric intensity at almost the same radius
    # from a broad azimuthal neighbourhood, then require a real dark deficit.
    limb6_radial_angle_halfwidth_deg: float = 10.0
    limb6_radial_exclusion_deg: float = 1.8
    limb6_radial_min_samples: int = 80
    # The acceptance test is made on a small 5x5 core average, not one
    # single pixel. This is the key 6.0.5 granulation guard: real limb spots
    # remain dark across several adjacent pixels, while most texture minima do not.
    limb6_radial_patch_radius_px: int = 2
    limb6_radial_patch_sigma_min: float = 1.40
    limb6_radial_patch_contrast_min: float = 0.050
    limb6_radial_annulus_halfwidth_px: float = 3.0

    # Very elongated components are typically limb/facular texture rather
    # than compact umbrae. Keep this conservative so true foreshortened spots
    # can still pass.
    limb6_max_aspect_ratio: float = 4.5

    # Dedicated limb quality. Unlike the ordinary Q score, this does NOT
    # depend on global darkness_peak, which is unreliable at the limb.
    limb6_quality_min: int = 3
    
    # 6.0.6: final validation of independent single limb detections.
    # Applied ONLY to a one-member LIMB_x object after the normal
    # limb-to-existing-group attachment check.
    limb_single_confidence_min: float = 0.50
    limb_single_radial_z_min: float = 3.0
    limb_single_area_min_px: int = 8
    limb_single_min_checks: int = 2

    # 5.7: faint isolated detections may only be promoted to counted
    # members in the outer limb zone. This prevents photospheric texture
    # inside ordinary groups from being promoted as extra spots.
    faint_group_promote_min_rsun: float = 0.84

    # -----------------------------------------------------------------
    # 6.0.0 two-stage umbra/pore detector
    # -----------------------------------------------------------------

    # Image-adaptive calibration is measured on the central photosphere.
    calibration_max_rsun: float = 0.70
    calibration_sigma_clip: float = 3.5
    strong_umbra_sigma: float = 6.0
    strong_umbra_threshold_floor: float = 0.060
    strong_umbra_threshold_ceiling: float = 0.105
    strong_umbra_min_area_px: int = 4
    strong_umbra_min_compactness: float = 0.08

    # Only Q4-Q5 primary umbrae seed the secure S/G pattern.
    strong_seed_quality_min: int = 3

    # Second pass: only around already-established groups, never around S.
    faint_group_search_radius_rsun: float = 0.085
    faint_local_sigma_small_px: float = 2.0
    faint_local_sigma_large_px: float = 6.0
    faint_local_noise_sigma: float = 3.2
    faint_local_threshold_floor: float = 0.014
    faint_local_threshold_ceiling: float = 0.050
    faint_local_min_area_px: int = 2
    faint_local_max_area_px: int = 90
    faint_local_min_compactness: float = 0.10
    faint_local_min_prominence: float = 0.006
    faint_local_min_sep_px: float = 3.0

    # 6.0.1: local contrast alone admits granulation. A faint candidate
    # must ALSO be globally darker than the normalized photosphere by this
    # minimum amount. This is still relative/normalized, not raw brightness.
    faint_main_darkness_min: float = 0.040

    # A single secure Q4-Q5 umbra may be upgraded from S to G if the local
    # pass finds at least this many accepted faint umbral companions.
    faint_single_seed_min_companions: int = 1

    # Faint local components have already passed morphology + local-intensity
    # tests and occur inside a secure group; Q2 is therefore the minimum final
    # quality that may still contribute. Q1 always fails.
    faint_group_quality_min: int = 2

    # Facular fields (white light, limb only)
    # Faculae are searched only near the visible limb, where their contrast
    # against the photosphere is physically meaningful.
    facula_inner_rsun: float = 0.88
    facula_outer_rsun: float = 0.985
    facula_local_sigma_px: float = 12.0
    facula_brightness_threshold: float = 0.040
    facula_min_area_px: int = 25
    facula_max_area_px: int = 5000
    facula_min_radial_width_px: int = 2
    facula_min_tangential_width_px: int = 4
    facula_label_min_separation_px: float = 24.0
    facula_max_abs_lat_deg: float = 45.0

    # Nearby bright fragments of the same extended facular field are merged
    # after the conservative detection pass. The distance is expressed in
    # solar-radius units so it scales with image size.
    facula_merge_distance_rsun: float = 0.08
    facula_merge_max_lat_diff_deg: float = 12.0

    # Annotation
    jpeg_quality: int = 95


CFG = Config()



def find_latest_solar_image(workdir: Path) -> Path:
    """
    Automatically find the newest Seestar solar image in WORKDIR.

    Priority:
      1) Files whose names contain YYYY-MM-DD-HHMMSS
      2) Newest file modification time

    Ignored:
      - *_annotated.*
      - files beginning with output-like names
      - non-JPG/PNG images
    """
    if not workdir.exists():
        raise FileNotFoundError(f"WORKDIR does not exist: {workdir}")

    candidates = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        candidates.extend(workdir.glob(pattern))

    ignored_tokens = (
        "_annotated",
        "spot_detection",
        "solar_north_up",
    )
    candidates = [
        p for p in candidates
        if not any(tok.lower() in p.name.lower() for tok in ignored_tokens)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No Seestar JPG/PNG image found in WORKDIR: {workdir}"
        )

    def sort_key(path: Path):
        m = re.search(
            r"(20\d{2})-(\d{2})-(\d{2})-(\d{2})(\d{2})(\d{2})",
            path.name
        )
        if m:
            y, mo, d, hh, mm, ss = map(int, m.groups())
            return (1, y, mo, d, hh, mm, ss, path.stat().st_mtime)
        return (0, 0, 0, 0, 0, 0, 0, path.stat().st_mtime)

    return max(candidates, key=sort_key)


# ---------------------------------------------------------------------
# Time / ephemeris
# ---------------------------------------------------------------------

def parse_datetime_from_filename(path: Path) -> datetime:
    """
    Looks for YYYY-MM-DD-HHMMSS in the filename.

    The Seestar filename timestamp is interpreted as Hungarian local time
    (Europe/Budapest). Daylight-saving time is handled automatically:
      - CET  = UTC+1 in winter
      - CEST = UTC+2 in summer

    The returned datetime is always UTC and is used for the solar ephemeris.
    """
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})-(\d{2})(\d{2})(\d{2})", path.name)
    if not m:
        raise ValueError(
            "Could not find YYYY-MM-DD-HHMMSS in the filename. "
            "Use a filename such as 2026-06-07-150043-....jpg"
        )

    y, mo, d, hh, mm, ss = map(int, m.groups())

    try:
        local_tz = ZoneInfo(LOCAL_TIMEZONE)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(
            f"Timezone database not found for {LOCAL_TIMEZONE}. "
            "On Windows, install it once with: pip install tzdata"
        ) from exc

    dt_local = datetime(y, mo, d, hh, mm, ss, tzinfo=local_tz)
    return dt_local.astimezone(timezone.utc)


def julian_date(dt_utc: datetime) -> float:
    return dt_utc.timestamp() / 86400.0 + 2440587.5


def solar_ephemeris_approx(dt_utc: datetime) -> dict:
    """
    Approximate solar ephemeris used by the program.

    Returns:
      P_deg       - position angle of the solar north pole
      B0_deg      - heliographic latitude of disk center
      sun_ra_deg  - approximate apparent solar right ascension
      sun_dec_deg - approximate apparent solar declination

    The accuracy is sufficient for image orientation and approximate
    heliographic coordinates in this project.
    """
    jd = julian_date(dt_utc)
    D = jd - 2451545.0

    g = math.radians((357.529 + 0.98560028 * D) % 360.0)
    q = (280.459 + 0.98564736 * D) % 360.0
    L = (q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360.0
    eps = 23.439 - 0.00000036 * D

    I = 7.25
    K = 73.6667 + 1.3958333 * (jd - 2396758.0) / 36525.0

    x = math.degrees(
        math.atan(-math.cos(math.radians(L - K)) * math.tan(math.radians(I)))
    )
    y = math.degrees(
        math.atan(-math.cos(math.radians(L)) * math.tan(math.radians(eps)))
    )
    P = x + y

    B0 = math.degrees(
        math.asin(math.sin(math.radians(L - K)) * math.sin(math.radians(I)))
    )

    # Approximate apparent equatorial coordinates of the Sun.
    Lr = math.radians(L)
    epsr = math.radians(eps)
    ra = math.atan2(
        math.cos(epsr) * math.sin(Lr),
        math.cos(Lr),
    )
    dec = math.asin(
        math.sin(epsr) * math.sin(Lr)
    )

    return {
        "P_deg": P,
        "B0_deg": B0,
        "sun_ra_deg": math.degrees(ra) % 360.0,
        "sun_dec_deg": math.degrees(dec),
    }


def greenwich_sidereal_time_deg(dt_utc: datetime) -> float:
    """Approximate Greenwich mean sidereal time in degrees."""
    jd = julian_date(dt_utc)
    T = (jd - 2451545.0) / 36525.0

    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - (T * T * T) / 38710000.0
    )
    return gmst % 360.0


def parallactic_angle_deg(
    dt_utc: datetime,
    ra_deg: float,
    dec_deg: float,
    observer_lat_deg: float,
    observer_lon_deg: float,
) -> tuple[float, float]:
    """
    Calculate the parallactic angle for an alt-az mounted camera.

    Longitude convention:
      east positive.

    Returns:
      (q_deg, hour_angle_deg)

    q is the sky rotation between celestial north and the local vertical.
    It is used together with the solar P-angle to place solar north up.
    """
    lst_deg = (
        greenwich_sidereal_time_deg(dt_utc) + observer_lon_deg
    ) % 360.0

    hour_angle_deg = (lst_deg - ra_deg + 180.0) % 360.0 - 180.0

    H = math.radians(hour_angle_deg)
    dec = math.radians(dec_deg)
    lat = math.radians(observer_lat_deg)

    q = math.atan2(
        math.sin(H),
        math.tan(lat) * math.cos(dec) - math.sin(dec) * math.cos(H),
    )

    return math.degrees(q), hour_angle_deg


# ---------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------

def detect_solar_disk(gray: np.ndarray) -> tuple[float, float, float]:
    _, thr = cv2.threshold(gray, CFG.disk_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("Solar disk could not be detected.")

    contour = max(contours, key=cv2.contourArea)
    (xc, yc), radius = cv2.minEnclosingCircle(contour)
    return float(xc), float(yc), float(radius)


def rotate_image(img: np.ndarray, angle_deg_ccw: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg_ccw, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def orient_solar_north_up(
    img: np.ndarray,
    dt_utc: datetime,
    camera_angle_deg: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Correct an unrotated Seestar alt-az image to solar north-up.

    The correction combines:
      1) parallactic angle q, calculated from observer position and time;
      2) solar P-angle;
      3) optional fixed camera/mount offset.

    Applied image rotation (OpenCV CCW-positive):
        q - P + camera_angle_deg

    The default camera_angle_deg=0 assumes the Seestar image axes are fixed
    to the alt-az frame without an additional constant rotation. If a
    consistent residual offset is found, it can be calibrated once.
    """
    eph = solar_ephemeris_approx(dt_utc)

    q_deg, hour_angle_deg = parallactic_angle_deg(
        dt_utc=dt_utc,
        ra_deg=eph["sun_ra_deg"],
        dec_deg=eph["sun_dec_deg"],
        observer_lat_deg=OBSERVER_LAT_DEG,
        observer_lon_deg=OBSERVER_LON_DEG,
    )

    rotation = q_deg - eph["P_deg"] + camera_angle_deg
    rotated = rotate_image(img, rotation)

    eph["parallactic_angle_deg"] = q_deg
    eph["solar_hour_angle_deg"] = hour_angle_deg
    eph["observer_lat_deg"] = OBSERVER_LAT_DEG
    eph["observer_lon_deg"] = OBSERVER_LON_DEG
    eph["camera_angle_deg"] = camera_angle_deg
    eph["applied_rotation_deg_ccw"] = rotation

    return rotated, eph


def normalized_maps(gray: np.ndarray, xc: float, yc: float, radius: float):
    yy, xx = np.indices(gray.shape)
    rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
    rnorm = rr / radius

    disk = rnorm <= 1.0
    inner = rnorm <= CFG.inner_fraction

    smooth = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=CFG.background_sigma_px, sigmaY=CFG.background_sigma_px
    ).astype(np.float32)

    g = gray.astype(np.float32)

    darkness = np.zeros_like(g, dtype=np.float32)
    brightness = np.zeros_like(g, dtype=np.float32)

    valid = smooth > 1.0
    darkness[valid] = (smooth[valid] - g[valid]) / smooth[valid]
    brightness[valid] = (g[valid] - smooth[valid]) / smooth[valid]

    darkness[~inner] = 0.0
    brightness[~disk] = 0.0

    darkness = cv2.GaussianBlur(
        darkness, (0, 0),
        sigmaX=CFG.darkness_smooth_sigma_px,
        sigmaY=CFG.darkness_smooth_sigma_px,
    )

    return rr, rnorm, disk, inner, darkness, brightness


def heliographic_activity_band_mask(
    shape: tuple[int, int],
    xc: float,
    yc: float,
    radius: float,
    B0_deg: float,
    max_abs_lat_deg: float,
) -> np.ndarray:
    """Boolean S55..N55 heliographic latitude mask.

    This is a latitude filter only. There is no heliographic-longitude
    restriction, so the east and west limbs remain searchable.
    """
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    x = (xx - float(xc)) / float(radius)
    y = (yy - float(yc)) / float(radius)
    rho2 = x * x + y * y
    disk = rho2 <= 1.0
    z = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))

    B0 = math.radians(B0_deg)
    y_solar = -y
    sin_lat = y_solar * math.cos(B0) + z * math.sin(B0)
    lat = np.degrees(np.arcsin(np.clip(sin_lat, -1.0, 1.0)))
    return disk & (np.abs(lat) <= float(max_abs_lat_deg))


# ---------------------------------------------------------------------
# Sunspot detection
# ---------------------------------------------------------------------

def clean_binary_mask(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            out[labels == i] = 255

    return out


def detect_spots(
    darkness: np.ndarray,
    inner_mask: np.ndarray,
    rnorm: np.ndarray,
) -> pd.DataFrame:
    """
    Detect broad dark regions, then resolve individual darkness peaks.

    Limb handling:
    - detection is allowed out to 0.99 R_sun;
    - thresholds become moderately more sensitive toward the limb;
    - minimum peak separation is reduced toward the limb, because projection
      compresses distinct umbrae/pore cores.

    The function remains deliberately conservative; weak detections are kept
    for the CSV and later receive quality 1-2, but they do not contribute to
    the Wolf count.
    """
    # Radially adaptive thresholds.
    # From r=0.78 outward, gradually reduce thresholds to 72% at the limb.
    limb_t = np.clip((rnorm - 0.78) / (0.99 - 0.78), 0.0, 1.0)
    region_thr_map = CFG.spot_region_threshold * (1.0 - 0.28 * limb_t)
    core_thr_map = CFG.spot_core_threshold * (1.0 - 0.25 * limb_t)

    region_mask = (
        (darkness > region_thr_map) & inner_mask
    ).astype(np.uint8) * 255

    kernel3 = np.ones((3, 3), np.uint8)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel3)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel3)
    region_mask = clean_binary_mask(region_mask, CFG.min_region_area_px)

    nreg, labels, stats, centroids = cv2.connectedComponentsWithStats(
        region_mask, connectivity=8
    )

    local_bg = cv2.GaussianBlur(
        darkness,
        (0, 0),
        sigmaX=5.0,
        sigmaY=5.0,
    )

    local_contrast = darkness - local_bg

    maxf = ndi.maximum_filter(
        darkness,
        size=CFG.peak_window_px,
        mode="nearest",
    )
    local_peak = darkness == maxf

    records = []
    sid = 1

    for reg in range(1, nreg):
        area = int(stats[reg, cv2.CC_STAT_AREA])
        region_pixels = labels == reg
        if area < CFG.min_region_area_px:
            continue

        # Candidate peaks must exceed the locally adjusted core threshold.
        yp, xp = np.where(
            region_pixels
            & local_peak
            & (darkness >= core_thr_map)
        )

        order = np.argsort(darkness[yp, xp])[::-1] if len(xp) else []
        selected: list[tuple[int, int]] = []

        for idx in order:
            x = int(xp[idx])
            y = int(yp[idx])

            # Reduce required separation toward the limb:
            # ~4.5 px near center, ~2.2 px at r=0.99.
            rp = float(rnorm[y, x])
            lt = float(np.clip((rp - 0.72) / (0.99 - 0.72), 0.0, 1.0))
            local_min_sep = CFG.min_peak_separation_px * (1.0 - 0.62 * lt)

            if all(
                math.hypot(x - ex, y - ey) >= local_min_sep
                for ex, ey in selected
            ):
                selected.append((x, y))

        # At least one candidate per valid dark region.
        if not selected:
            x = int(round(float(centroids[reg][0])))
            y = int(round(float(centroids[reg][1])))
            selected = [(x, y)]

        region_u8 = region_pixels.astype(np.uint8)
        contours, _ = cv2.findContours(
            region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        compactness = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0 else 0.0
        )

        for x, y in selected:
            records.append(
                {
                    "spot_id": f"S{sid}",
                    "region_id": reg,
                    "cx_px": float(x),
                    "cy_px": float(y),
                    "darkness_peak": float(darkness[y, x]),
                    "region_area_px": area,
                    "compactness": float(np.clip(compactness, 0.0, 1.0)),
                    "detection_source": "main",
                    "limb_confirmed": False,
                }
            )
            sid += 1

    return pd.DataFrame(records), region_mask



def radial_limb_confirmation(
    gray: np.ndarray,
    xc: float,
    yc: float,
    x: float,
    y: float,
    radius: float,
) -> bool:
    """
    Confirm a very limb-near dark candidate using the radial intensity pattern:

        disk interior -> dark spot -> brighter limb layer -> black sky

    This rejects many edge artefacts while preserving real sunspots that
    still have a thin bright solar rim outside them.
    """
    dx = x - xc
    dy = y - yc
    r = math.hypot(dx, dy)

    if r <= 0:
        return False

    rn = r / radius
    if rn < CFG.limb_profile_min_rsun or rn > CFG.limb_profile_max_rsun:
        return False

    ux = dx / r
    uy = dy / r

    def sample_at(offset_px: float):
        sx = int(round(x + ux * offset_px))
        sy = int(round(y + uy * offset_px))
        if sx < 0 or sy < 0 or sx >= gray.shape[1] or sy >= gray.shape[0]:
            return None
        return float(gray[sy, sx])

    center_val = sample_at(0)
    if center_val is None:
        return False

    # Samples slightly inward and outward.
    inward_vals = []
    outward_vals = []

    for k in range(2, CFG.limb_profile_inner_px + 1, 2):
        v = sample_at(-k)
        if v is not None:
            inward_vals.append(v)

    for k in range(2, CFG.limb_profile_outer_px + 1, 2):
        v = sample_at(k)
        if v is not None:
            outward_vals.append(v)

    if not inward_vals or not outward_vals:
        return False

    inner_med = float(np.median(inward_vals))
    outer_max = float(np.max(outward_vals))
    outer_min = float(np.min(outward_vals))

    # Candidate should be distinctly darker than nearby photosphere inward.
    dark_enough = center_val < inner_med * (1.0 - CFG.limb_profile_dark_threshold)

    # There should still be a brighter solar-rim sample outward from the spot.
    bright_rim = outer_max > center_val * (1.0 + CFG.limb_profile_bright_margin)

    # And farther outward, some sample should approach the black sky.
    sky_reached = outer_min < max(15.0, 0.35 * inner_med)

    return bool(dark_enough and bright_rim and sky_reached)


def recover_limb_spots(
    gray: np.ndarray,
    rnorm: np.ndarray,
    existing_spots: pd.DataFrame,
    xc: float,
    yc: float,
    radius: float,
) -> pd.DataFrame:
    """
    Second-pass detector dedicated to the extreme limb.

    Uses a much smaller local background scale than the main detector,
    so the black sky outside the solar disk does not suppress a real
    spot close to the limb.

    New detections are appended only if they are not already represented
    by a main-pass spot.
    """
    g = gray.astype(np.float32)

    local_bg = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=CFG.limb_local_sigma_px,
        sigmaY=CFG.limb_local_sigma_px,
    ).astype(np.float32)

    local_dark = np.zeros_like(g, dtype=np.float32)
    valid = local_bg > 5
    local_dark[valid] = (local_bg[valid] - g[valid]) / local_bg[valid]

    annulus = (
        (rnorm >= CFG.limb_recovery_inner_rsun)
        & (rnorm <= CFG.limb_recovery_outer_rsun)
    )

    # 5.8.0: radially adaptive limb threshold.
    # Keep the inner limb conservative, but become more sensitive only
    # toward the extreme edge. Radial-profile confirmation below remains
    # the safeguard against ordinary limb/sky artefacts.
    limb_t = np.clip(
        (rnorm - CFG.limb_recovery_inner_rsun)
        / max(
            1e-6,
            CFG.limb_recovery_outer_rsun - CFG.limb_recovery_inner_rsun
        ),
        0.0,
        1.0,
    )
    limb_thr_map = CFG.limb_core_threshold * (1.0 - 0.50 * limb_t)

    mask = (
        annulus
        & (local_dark >= limb_thr_map)
    ).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = clean_binary_mask(mask, CFG.limb_min_area_px)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    additions = []
    next_id = 1 if existing_spots.empty else len(existing_spots) + 1

    existing_xy = []
    if not existing_spots.empty:
        existing_xy = list(
            zip(existing_spots["cx_px"].astype(float), existing_spots["cy_px"].astype(float))
        )

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < CFG.limb_min_area_px:
            continue

        region = labels == i
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue

        # Strongest local-dark pixel as the recovered spot core.
        vals = local_dark[ys, xs]
        k = int(np.argmax(vals))
        x = float(xs[k])
        y = float(ys[k])
        peak = float(vals[k])

        # For the extreme limb, require the radial pattern:
        # dark spot with a thin brighter solar rim still outside it.
        rn = float(rnorm[int(round(y)), int(round(x))])
        limb_confirmed = False
        if rn >= CFG.limb_profile_min_rsun:
            if not radial_limb_confirmation(gray, xc, yc, x, y, radius):
                continue
            limb_confirmed = True

        # Do not duplicate a spot already found by the main detector.
        if any(
            math.hypot(x - ex, y - ey) <= CFG.limb_merge_distance_px
            for ex, ey in existing_xy
        ):
            continue

        # Shape information for quality calculation.
        region_u8 = region.astype(np.uint8)
        contours, _ = cv2.findContours(
            region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        compactness = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0 else 0.0
        )

        additions.append(
            {
                "spot_id": f"S{next_id}",
                "region_id": 10000 + i,
                "cx_px": x,
                "cy_px": y,
                # Map the small-scale limb contrast onto the same rough
                # contrast scale used by the main detector.
                "darkness_peak": peak,
                "region_area_px": area,
                "compactness": float(np.clip(compactness, 0.0, 1.0)),
                "detection_source": "limb_recovery",
                "limb_confirmed": bool(limb_confirmed),
            }
        )
        next_id += 1
        existing_xy.append((x, y))

    if not additions:
        return existing_spots

    extra = pd.DataFrame(additions)
    if existing_spots.empty:
        return extra.reset_index(drop=True)

    return pd.concat([existing_spots, extra], ignore_index=True)


def split_strong_limb_cores(
    spots: pd.DataFrame,
    darkness: np.ndarray,
    region_mask: np.ndarray,
    rnorm: np.ndarray,
) -> pd.DataFrame:
    """
    V5 second-stage splitter.

    It does NOT search the whole disk for new weak spots. Instead it only
    revisits dark regions that the main detector already accepted, and only
    toward the limb. Within those confirmed regions it looks for several
    strong, locally prominent darkness maxima.

    This is designed for cases where four clearly dark cores are visually
    resolved but the first pass merges them into two.
    """
    if spots.empty:
        return spots

    nreg, labels, stats, centroids = cv2.connectedComponentsWithStats(
        region_mask, connectivity=8
    )

    # Small-scale local baseline; a true core must rise above this in the
    # darkness map by a minimum prominence.
    local_base = cv2.GaussianBlur(
        darkness,
        (0, 0),
        sigmaX=CFG.strong_split_background_sigma_px,
        sigmaY=CFG.strong_split_background_sigma_px,
    )

    maxf = ndi.maximum_filter(
        darkness,
        size=CFG.strong_split_window_px,
        mode="nearest",
    )
    local_max = darkness == maxf

    out_records = spots.to_dict("records")

    # Existing coordinates keyed by region.
    existing_by_region = {}
    for _, row in spots.iterrows():
        reg = int(row["region_id"])
        existing_by_region.setdefault(reg, []).append(
            (float(row["cx_px"]), float(row["cy_px"]))
        )

    next_id = len(out_records) + 1

    for reg in range(1, nreg):
        region_pixels = labels == reg
        if not np.any(region_pixels):
            continue

        ys, xs = np.where(region_pixels)
        mean_r = float(np.mean(rnorm[ys, xs]))
        if mean_r < CFG.strong_split_min_rsun:
            continue

        # Only strong + locally prominent cores.
        prominence = darkness - local_base
        cand = (
            region_pixels
            & local_max
            & (darkness >= CFG.strong_split_peak_threshold)
            & (prominence >= CFG.strong_split_prominence)
        )

        yp, xp = np.where(cand)
        if len(xp) == 0:
            continue

        order = np.argsort(darkness[yp, xp])[::-1]
        existing = list(existing_by_region.get(reg, []))
        selected_new = []

        for idx in order:
            x = float(xp[idx])
            y = float(yp[idx])

            # Keep close but genuinely distinct cores. We only reject
            # candidates almost on top of an already selected core.
            if any(
                math.hypot(x - ex, y - ey) < CFG.strong_split_min_sep_px
                for ex, ey in (existing + selected_new)
            ):
                continue

            selected_new.append((x, y))

        if not selected_new:
            continue

        area = int(stats[reg, cv2.CC_STAT_AREA])

        region_u8 = region_pixels.astype(np.uint8)
        contours, _ = cv2.findContours(
            region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        compactness = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0 else 0.0
        )

        for x, y in selected_new:
            out_records.append(
                {
                    "spot_id": f"S{next_id}",
                    "region_id": reg,
                    "cx_px": x,
                    "cy_px": y,
                    "darkness_peak": float(darkness[int(round(y)), int(round(x))]),
                    "region_area_px": area,
                    "compactness": float(np.clip(compactness, 0.0, 1.0)),
                    "detection_source": "strong_split",
                    "limb_confirmed": False,
                }
            )
            next_id += 1

    return pd.DataFrame(out_records)


def recover_faint_group_companions(
    spots: pd.DataFrame,
    darkness: np.ndarray,
    inner_mask: np.ndarray,
    radius: float,
) -> pd.DataFrame:
    """
    Recover faint but coherent companions only near already detected spots.

    Important 5.5 correction:
    Every local darkness maximum inside a connected candidate region is
    examined. Previously only the strongest maximum of the region was used,
    which could hide a faint companion next to a stronger already-known spot.
    """
    if spots.empty:
        return spots

    # Local contrast map used only by the faint-companion search.
    # This must be defined INSIDE this function.
    local_bg = cv2.GaussianBlur(
        darkness,
        (0, 0),
        sigmaX=5.0,
        sigmaY=5.0,
    )
    local_contrast = darkness - local_bg

    yy, xx = np.indices(darkness.shape)

    # Search only around already detected spots/groups.
    near_existing = np.zeros_like(inner_mask, dtype=bool)
    search_r = CFG.companion_search_radius_rsun * radius

    for _, row in spots.iterrows():
        cx = float(row["cx_px"])
        cy = float(row["cy_px"])

        # Csak peremhez közeli meglévő foltok körül keressünk halvány társakat
        r = math.hypot(
            cx - darkness.shape[1] / 2,
            cy - darkness.shape[0] / 2
        )
        r_norm = r / radius

        if r_norm < 0.70:
            continue

        near_existing |= ((xx - cx) ** 2 + (yy - cy) ** 2) <= search_r ** 2

    candidate_mask = (
        inner_mask
        & near_existing
        & (darkness >= CFG.companion_region_threshold)
    ).astype(np.uint8) * 255

    # Minimal cleanup.
    kernel = np.ones((2, 2), np.uint8)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
    candidate_mask = clean_binary_mask(
        candidate_mask,
        CFG.companion_min_area_px
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8
    )

    existing_xy = list(
        zip(
            spots["cx_px"].astype(float),
            spots["cy_px"].astype(float)
        )
    )

    records = spots.to_dict("records")
    next_id = len(records) + 1

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < CFG.companion_min_area_px:
            continue

        region = labels == i
        ys, xs = np.where(region)

        if len(xs) == 0:
            continue

        # Find ALL local darkness maxima within this connected region.
        maxf = ndi.maximum_filter(
            local_contrast,
            size=3,
            mode="nearest",
        )

        local_max = (
            region
            & (local_contrast == maxf)
            & (local_contrast >= 0.015)
        )

        yp, xp = np.where(local_max)

        if len(xp) == 0:
            continue

        # Calculate region morphology once.
        region_u8 = region.astype(np.uint8)
        contours, _ = cv2.findContours(
            region_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0

        compactness = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )

        # Strongest local maxima first.
        order = np.argsort(darkness[yp, xp])[::-1]

        for idx in order:
            x = float(xp[idx])
            y = float(yp[idx])
            peak = float(darkness[int(y), int(x)])

            # 5.7.9: enforce the configured companion peak threshold.
            # Earlier versions defined companion_peak_threshold but never
            # actually used it here.
            if peak < CFG.companion_peak_threshold:
                continue

            # If this peak is already represented, skip ONLY this peak.
            # Do not discard the whole connected region.
            if any(
                math.hypot(x - ex, y - ey) < CFG.companion_min_sep_px
                for ex, ey in existing_xy
            ):
                continue

            records.append(
                {
                    "spot_id": f"S{next_id}",
                    "region_id": 20000 + i,
                    "cx_px": x,
                    "cy_px": y,
                    "darkness_peak": peak,
                    "region_area_px": area,
                    "compactness": float(
                        np.clip(compactness, 0.0, 1.0)
                    ),
                    "detection_source": "companion",
                    "limb_confirmed": False,
                }
            )

            existing_xy.append((x, y))
            next_id += 1

    return pd.DataFrame(records)


def recover_isolated_faint_spots(
    spots: pd.DataFrame,
    gray: np.ndarray,
    darkness: np.ndarray,
    inner_mask: np.ndarray,
    rnorm: np.ndarray,
) -> pd.DataFrame:
    """
    Version 5.3:
    Search for isolated faint but coherent sunspots on the inner disk.

    Safeguards:
    - only r < faint_single_max_rsun
    - must be locally dark on two smoothing scales
    - must have measurable local prominence
    - must form a compact connected component
    - must not lie too close to an already detected spot

    The goal is to recover real faint single spots without turning
    photospheric texture, dust, or sharpening artefacts into sunspots.
    """
    if spots.empty:
        existing_xy = []
        records = []
        next_id = 1
    else:
        existing_xy = list(
            zip(spots["cx_px"].astype(float), spots["cy_px"].astype(float))
        )
        records = spots.to_dict("records")
        next_id = len(records) + 1

    g = gray.astype(np.float32)

    small_bg = cv2.GaussianBlur(
        gray, (0, 0),
        sigmaX=CFG.faint_single_sigma_small_px,
        sigmaY=CFG.faint_single_sigma_small_px,
    ).astype(np.float32)

    large_bg = cv2.GaussianBlur(
        gray, (0, 0),
        sigmaX=CFG.faint_single_sigma_large_px,
        sigmaY=CFG.faint_single_sigma_large_px,
    ).astype(np.float32)

    dark_small = np.zeros_like(g, dtype=np.float32)
    dark_large = np.zeros_like(g, dtype=np.float32)

    valid_small = small_bg > 5
    valid_large = large_bg > 5

    dark_small[valid_small] = (small_bg[valid_small] - g[valid_small]) / small_bg[valid_small]
    dark_large[valid_large] = (large_bg[valid_large] - g[valid_large]) / large_bg[valid_large]

    search_zone = (
        inner_mask
        & (rnorm <= CFG.faint_single_max_rsun)
    )

    # A real faint spot should appear dark relative to both a small- and
    # a larger-scale local photospheric background.
    candidate = (
        search_zone
        & (dark_small >= CFG.faint_single_region_threshold)
        & (dark_large >= CFG.faint_single_region_threshold)
    ).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < CFG.faint_single_min_area_px:
            continue
        if area > CFG.faint_single_max_area_px:
            continue

        region = labels == i
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue

        # Peak from the large-scale darkness map.
        vals = dark_large[ys, xs]
        k = int(np.argmax(vals))
        peak = float(vals[k])

        if peak < CFG.faint_single_peak_threshold:
            continue

        x = float(xs[k])
        y = float(ys[k])

        # Must be isolated from already known detections.
        if any(
            math.hypot(x - ex, y - ey) < CFG.faint_single_min_sep_px
            for ex, ey in existing_xy
        ):
            continue

        # Local prominence: candidate should be darker than the
        # surrounding larger-scale local background.
        y0 = max(0, int(round(y)) - 8)
        y1 = min(gray.shape[0], int(round(y)) + 9)
        x0 = max(0, int(round(x)) - 8)
        x1 = min(gray.shape[1], int(round(x)) + 9)

        local_patch = dark_large[y0:y1, x0:x1]
        if local_patch.size == 0:
            continue

        local_med = float(np.median(local_patch))
        prominence = peak - local_med
        if prominence < CFG.faint_single_prominence:
            continue

        # Compactness filter.
        region_u8 = region.astype(np.uint8)
        contours, _ = cv2.findContours(
            region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        compactness = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0 else 0.0
        )

        if compactness < CFG.faint_single_compactness_min:
            continue

        # Use the main darkness map for the stored detection metric so
        # quality scoring remains comparable with the rest of the pipeline.
        iy = int(round(y))
        ix = int(round(x))
        stored_peak = float(darkness[iy, ix])

        records.append(
            {
                "spot_id": f"S{next_id}",
                "region_id": 30000 + i,
                "cx_px": x,
                "cy_px": y,
                "darkness_peak": stored_peak,
                "region_area_px": area,
                "compactness": float(np.clip(compactness, 0.0, 1.0)),
                "detection_source": "isolated_faint",
                "limb_confirmed": False,
            }
        )
        existing_xy.append((x, y))
        next_id += 1

    return pd.DataFrame(records)


def deduplicate_spots(
    spots: pd.DataFrame,
    min_distance_px: float = 2.0,
) -> pd.DataFrame:
    """
    Remove duplicate detections produced by overlapping recovery passes.

    If two detections are closer than min_distance_px, keep only the stronger
    one (higher darkness_peak; if equal, keep the larger region).
    """
    if spots.empty:
        return spots

    work = spots.copy()

    # Stronger detections get priority.
    work = work.sort_values(
        ["darkness_peak", "region_area_px"],
        ascending=[False, False],
    ).reset_index(drop=True)

    keep_rows = []
    kept_xy = []

    for _, row in work.iterrows():
        x = float(row["cx_px"])
        y = float(row["cy_px"])

        duplicate = any(
            math.hypot(x - kx, y - ky) < min_distance_px
            for kx, ky in kept_xy
        )

        if duplicate:
            continue

        keep_rows.append(row.to_dict())
        kept_xy.append((x, y))

    out = pd.DataFrame(keep_rows)

    if out.empty:
        return out

    # Stable spatial-ish order is useful before grouping/renumbering.
    out = out.sort_values(
        ["cy_px", "cx_px"]
    ).reset_index(drop=True)

    return out



# ---------------------------------------------------------------------
# 6.0.0 two-stage umbra / pore detection
# ---------------------------------------------------------------------

def robust_photosphere_calibration(
    darkness: np.ndarray,
    rnorm: np.ndarray,
    activity_band: np.ndarray,
) -> dict:
    """
    Estimate the image's photospheric darkness noise robustly.

    Only the central disk is used, avoiding the limb. Several clipping
    iterations remove real spots and extreme processing artefacts. Thresholds
    therefore follow the actual contrast/sharpening of each Seestar image
    rather than being tuned to one frame.
    """
    sample_mask = (
        activity_band
        & (rnorm <= CFG.calibration_max_rsun)
        & np.isfinite(darkness)
    )

    vals = darkness[sample_mask].astype(np.float64)
    if vals.size < 100:
        raise RuntimeError("Too few central-photosphere pixels for calibration.")

    work = vals.copy()
    for _ in range(4):
        med = float(np.median(work))
        mad = float(np.median(np.abs(work - med)))
        sigma = max(1e-6, 1.4826 * mad)
        keep = np.abs(work - med) <= CFG.calibration_sigma_clip * sigma
        if keep.sum() < 100 or keep.sum() == work.size:
            break
        work = work[keep]

    med = float(np.median(work))
    mad = float(np.median(np.abs(work - med)))
    sigma = max(1e-6, 1.4826 * mad)

    strong_thr = float(np.clip(
        med + CFG.strong_umbra_sigma * sigma,
        CFG.strong_umbra_threshold_floor,
        CFG.strong_umbra_threshold_ceiling,
    ))

    return {
        "photosphere_dark_median": med,
        "photosphere_dark_sigma": sigma,
        "strong_umbra_threshold": strong_thr,
        "calibration_pixels": int(work.size),
    }


def _component_shape(region: np.ndarray, area: int) -> float:
    contours, _ = cv2.findContours(
        region.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
    if perimeter <= 0:
        return 0.0
    return float(np.clip(
        4.0 * math.pi * area / (perimeter * perimeter),
        0.0,
        1.0,
    ))


def detect_strong_umbrae(
    darkness: np.ndarray,
    active_inner: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Primary 6.0 detector.

    One connected dark component is one umbra/pore, regardless of how many
    local darkness maxima exist inside it. Irregular, elongated and lobed
    umbrae therefore remain single counted objects.
    """
    mask = (
        active_inner
        & (darkness >= threshold)
    ).astype(np.uint8) * 255

    # Minimal cleanup only. Avoid closing that could merge neighbouring umbrae.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = clean_binary_mask(mask, CFG.strong_umbra_min_area_px)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    rows = []
    sid = 1

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < CFG.strong_umbra_min_area_px:
            continue

        region = labels == i
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue

        vals = darkness[ys, xs]
        k = int(np.argmax(vals))
        x = float(xs[k])
        y = float(ys[k])
        peak = float(vals[k])

        compactness = _component_shape(region, area)
        if compactness < CFG.strong_umbra_min_compactness:
            continue

        rows.append({
            "spot_id": f"S{sid}",
            "region_id": i,
            "cx_px": x,
            "cy_px": y,
            "darkness_peak": peak,
            "region_area_px": area,
            "compactness": compactness,
            "detection_source": "strong_umbra",
            "local_contrast": np.nan,
            "local_prominence": np.nan,
            "counted": False,
        })
        sid += 1

    return pd.DataFrame(rows), mask


def _local_darkness_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Two local relative-intensity maps.

    A real small pore should remain darker than its surroundings on both a
    small and a larger background scale. Granulation is much less stable
    between the two scales.
    """
    g = gray.astype(np.float32)

    bg_small = cv2.GaussianBlur(
        gray, (0, 0),
        sigmaX=CFG.faint_local_sigma_small_px,
        sigmaY=CFG.faint_local_sigma_small_px,
    ).astype(np.float32)

    bg_large = cv2.GaussianBlur(
        gray, (0, 0),
        sigmaX=CFG.faint_local_sigma_large_px,
        sigmaY=CFG.faint_local_sigma_large_px,
    ).astype(np.float32)

    dsmall = np.zeros_like(g, dtype=np.float32)
    dlarge = np.zeros_like(g, dtype=np.float32)

    ok1 = bg_small > 5.0
    ok2 = bg_large > 5.0

    dsmall[ok1] = (bg_small[ok1] - g[ok1]) / bg_small[ok1]
    dlarge[ok2] = (bg_large[ok2] - g[ok2]) / bg_large[ok2]

    return dsmall, dlarge


def detect_faint_umbrae_in_groups(
    gray: np.ndarray,
    darkness: np.ndarray,
    active_inner: np.ndarray,
    strong_mask: np.ndarray,
    secure_spots: pd.DataFrame,
    radius: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Second-pass local-intensity search.

    Search around every secure Q4-Q5 primary umbra/group. An isolated S may
    be upgraded to G only if this pass finds at least one accepted companion.

    Faint candidates must:
      * be a connected component on TWO local-intensity scales;
      * exceed the local robust noise level;
      * have finite area and compactness;
      * show measurable local prominence;
      * be distinct from primary strong umbrae.

    They are assigned directly to an existing group and never create/bridge
    groups by themselves.
    """
    empty_stats = {
        "faint_local_threshold": np.nan,
        "faint_local_noise_sigma": np.nan,
    }

    if secure_spots.empty or "group_id" not in secure_spots.columns:
        return pd.DataFrame(), empty_stats

    # 6.0.1: search around every secure Q4-Q5 seed, including an isolated S.
    # An isolated seed remains S unless at least one accepted faint companion
    # is found nearby.
    group_sizes = secure_spots.groupby("group_id").size().to_dict()
    valid_groups = set(group_sizes.keys())
    if not valid_groups:
        return pd.DataFrame(), empty_stats

    dsmall, dlarge = _local_darkness_maps(gray)

    yy, xx = np.indices(gray.shape)
    search_zone = np.zeros_like(active_inner, dtype=bool)
    search_r = CFG.faint_group_search_radius_rsun * radius

    anchors_by_group: dict[str, list[tuple[float, float]]] = {}
    for gid in valid_groups:
        gdf = secure_spots[secure_spots["group_id"] == gid]
        anchors = list(zip(
            gdf["cx_px"].astype(float),
            gdf["cy_px"].astype(float),
        ))
        anchors_by_group[gid] = anchors
        for ax, ay in anchors:
            search_zone |= ((xx - ax) ** 2 + (yy - ay) ** 2) <= search_r ** 2

    search_zone &= active_inner

    # Exclude primary umbra pixels plus a tiny one-pixel guard rim, otherwise
    # the faint pass could rediscover the edge of a large primary umbra.
    strong_guard = cv2.dilate(
        (strong_mask > 0).astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1,
    ).astype(bool)
    search_zone &= ~strong_guard

    noise_vals = dlarge[search_zone]
    if noise_vals.size < 30:
        return pd.DataFrame(), empty_stats

    # Robust local-granulation statistics.
    med = float(np.median(noise_vals))
    mad = float(np.median(np.abs(noise_vals - med)))
    sigma = max(1e-6, 1.4826 * mad)

    local_thr = float(np.clip(
        med + CFG.faint_local_noise_sigma * sigma,
        CFG.faint_local_threshold_floor,
        CFG.faint_local_threshold_ceiling,
    ))

    # Require coherence on two smoothing/background scales.
    candidate = (
        search_zone
        & (dlarge >= local_thr)
        & (dsmall >= 0.60 * local_thr)
    ).astype(np.uint8) * 255

    # No opening: true faint pores may only span 2-3 pixels.
    candidate = clean_binary_mask(
        candidate,
        CFG.faint_local_min_area_px,
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )

    existing_xy = list(zip(
        secure_spots["cx_px"].astype(float),
        secure_spots["cy_px"].astype(float),
    ))

    rows = []
    sid = 1

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < CFG.faint_local_min_area_px:
            continue
        if area > CFG.faint_local_max_area_px:
            continue

        region = labels == i
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue

        vals = dlarge[ys, xs]
        k = int(np.argmax(vals))
        x = float(xs[k])
        y = float(ys[k])
        peak_local = float(vals[k])

        # Separate from existing secure umbra centroids.
        if any(
            math.hypot(x - ex, y - ey) < CFG.faint_local_min_sep_px
            for ex, ey in existing_xy
        ):
            continue

        compactness = _component_shape(region, area)
        if compactness < CFG.faint_local_min_compactness:
            continue

        # Local prominence against an annular-ish 11x11 neighbourhood median.
        iy, ix = int(round(y)), int(round(x))
        y0, y1 = max(0, iy - 5), min(gray.shape[0], iy + 6)
        x0, x1 = max(0, ix - 5), min(gray.shape[1], ix + 6)

        patch = dlarge[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        local_med = float(np.median(patch))
        prominence = peak_local - local_med
        if prominence < CFG.faint_local_min_prominence:
            continue

        # Assign to nearest SECURE anchor, not to another faint candidate.
        best_gid = None
        best_dist = float("inf")
        for gid, anchors in anchors_by_group.items():
            for ax, ay in anchors:
                d = math.hypot(x - ax, y - ay)
                if d < best_dist:
                    best_dist = d
                    best_gid = gid

        if best_gid is None or best_dist > search_r:
            continue

        # Store main normalized darkness for ordinary Q compatibility.
        main_peak = float(darkness[iy, ix])

        # 6.0.1 granululation guard:
        # local contrast must be backed up by a real depression in the
        # globally normalized darkness map. This rejects most granulation
        # elements that look locally dark but are ordinary photosphere.
        if main_peak < CFG.faint_main_darkness_min:
            continue

        rows.append({
            "spot_id": f"F{sid}",
            "region_id": 50000 + i,
            "cx_px": x,
            "cy_px": y,
            "darkness_peak": main_peak,
            "region_area_px": area,
            "compactness": compactness,
            "detection_source": "faint_umbra_local",
            "local_contrast": peak_local,
            "local_prominence": prominence,
            "counted": False,
            "group_id": best_gid,
        })
        sid += 1

    return pd.DataFrame(rows), {
        "faint_local_threshold": local_thr,
        "faint_local_noise_sigma": sigma,
    }



def detect_extreme_limb_umbrae(
    gray: np.ndarray,
    darkness: np.ndarray,
    rnorm: np.ndarray,
    activity_band: np.ndarray,
    existing_spots: pd.DataFrame,
    xc: float,
    yc: float,
    radius: float,
    B0_deg: float,
) -> pd.DataFrame:
    """
    6.0.5 conservative extreme-limb detector.

    Stage 1: multi-scale tangential contrast creates CANDIDATES only.
    Stage 2: each connected candidate must also be significantly darker than
             the photosphere at almost the same solar radius in a broader
             azimuthal neighbourhood.

    This keeps the limb-darkening compensation of 6.0.3 while rejecting the
    large number of facular/granulation minima that previously passed on
    tangential contrast alone.

    The inner-disk detector and its groups are untouched. Limb objects are
    returned independently and are grouped only after this final filtering.
    """
    g = gray.astype(np.float32)
    h, w = gray.shape

    yy, xx = np.indices(gray.shape, dtype=np.float32)
    dx = xx - float(xc)
    dy = yy - float(yc)
    rr = np.sqrt(dx * dx + dy * dy)

    annulus = (
        activity_band
        & (rnorm >= CFG.limb6_inner_rsun)
        & (rnorm <= CFG.limb6_outer_rsun)
        & (rr > 1.0)
    )

    # Polar angle around the disk centre, used by the radial-profile guard.
    theta = np.arctan2(dy, dx)

    tx = np.zeros_like(g, dtype=np.float32)
    ty = np.zeros_like(g, dtype=np.float32)
    tx[annulus] = -dy[annulus] / rr[annulus]
    ty[annulus] =  dx[annulus] / rr[annulus]

    # ---------- Stage 1: tangential candidate generation ----------
    scale_contrast_maps = []

    for off in CFG.limb6_tangent_offsets_px:
        pair_samples = []

        for sign in (-1.0, 1.0):
            sx = np.rint(xx + sign * float(off) * tx).astype(np.int32)
            sy = np.rint(yy + sign * float(off) * ty).astype(np.int32)

            valid = (
                annulus
                & (sx >= 0) & (sx < w)
                & (sy >= 0) & (sy < h)
            )

            sample = np.full_like(g, np.nan, dtype=np.float32)
            sample[valid] = g[sy[valid], sx[valid]]
            pair_samples.append(sample)

        pair_stack = np.stack(pair_samples, axis=0)
        with np.errstate(invalid="ignore"):
            pair_ref = np.nanmedian(pair_stack, axis=0)

        c = np.zeros_like(g, dtype=np.float32)
        valid_ref = annulus & np.isfinite(pair_ref) & (pair_ref > 5.0)
        c[valid_ref] = (pair_ref[valid_ref] - g[valid_ref]) / pair_ref[valid_ref]

        c = cv2.GaussianBlur(c, (0, 0), sigmaX=0.7, sigmaY=0.7)
        scale_contrast_maps.append(c)

    contrast_stack = np.stack(scale_contrast_maps, axis=0)
    limb_contrast = np.median(contrast_stack, axis=0)
    confirmed_scales = np.sum(
        contrast_stack >= CFG.limb6_scale_contrast_min,
        axis=0,
    ).astype(np.int16)

    mask = (
        annulus
        & (limb_contrast >= CFG.limb6_min_local_contrast)
        & (confirmed_scales >= CFG.limb6_min_confirmed_scales)
    ).astype(np.uint8) * 255

    # No morphology that could connect neighbouring texture into arcs.
    mask = clean_binary_mask(mask, CFG.limb6_min_area_px)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    existing_xy = []
    if not existing_spots.empty:
        existing_xy = list(zip(
            existing_spots["cx_px"].astype(float),
            existing_spots["cy_px"].astype(float),
        ))

    # Helper: wrapped angular difference in radians.
    def angle_diff(a: np.ndarray, b: float) -> np.ndarray:
        return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

    rows = []

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < CFG.limb6_min_area_px or area > CFG.limb6_max_area_px:
            continue

        region = labels == i
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue

        vals = limb_contrast[ys, xs]
        k = int(np.argmax(vals))
        x = float(xs[k])
        y = float(ys[k])
        iy = int(round(y))
        ix = int(round(x))

        local_peak = float(limb_contrast[iy, ix])
        n_scales = int(confirmed_scales[iy, ix])
        scale_vals = [float(m[iy, ix]) for m in scale_contrast_maps]
        min_scale_contrast = float(min(scale_vals))
        median_scale_contrast = float(np.median(scale_vals))

        if any(
            math.hypot(x - ex, y - ey) < CFG.limb6_min_sep_px
            for ex, ey in existing_xy
        ):
            continue

        compactness = _component_shape(region, area)
        if compactness < CFG.limb6_min_compactness:
            continue

        # Reject extremely elongated arcs/threads before the expensive test.
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        short_side = max(1, min(ww, hh))
        aspect_ratio = max(ww, hh) / short_side
        if aspect_ratio > CFG.limb6_max_aspect_ratio:
            continue

        # ---------- Stage 2: same-radius radial-profile confirmation ----------
        cand_r = float(rr[iy, ix])
        cand_theta = float(theta[iy, ix])

        radial_band = np.abs(rr - cand_r) <= CFG.limb6_radial_annulus_halfwidth_px
        dtheta = angle_diff(theta, cand_theta)

        halfwidth = math.radians(CFG.limb6_radial_angle_halfwidth_deg)
        exclusion = math.radians(CFG.limb6_radial_exclusion_deg)

        reference_zone = (
            activity_band
            & radial_band
            & (dtheta <= halfwidth)
            & (dtheta >= exclusion)
            & (g > 5.0)
        )

        ref_vals = g[reference_zone].astype(np.float64)
        if ref_vals.size < CFG.limb6_radial_min_samples:
            continue

        # Robust clipping prevents another real spot/facula from dominating
        # the local same-radius reference.
        work = ref_vals.copy()
        for _ in range(3):
            med = float(np.median(work))
            mad = float(np.median(np.abs(work - med)))
            sig = max(1e-6, 1.4826 * mad)
            keep = np.abs(work - med) <= 3.5 * sig
            if keep.sum() < CFG.limb6_radial_min_samples or keep.sum() == work.size:
                break
            work = work[keep]

        radial_med = float(np.median(work))
        radial_mad = float(np.median(np.abs(work - radial_med)))
        radial_sigma = max(1e-6, 1.4826 * radial_mad)

        # 6.0.5 coherence test: use a small core average instead of a
        # one-pixel minimum. A real pore/umbra produces a coherent dark patch;
        # granulation and sharpening minima usually disappear in this average.
        pr = int(CFG.limb6_radial_patch_radius_px)
        py0, py1 = max(0, iy - pr), min(h, iy + pr + 1)
        px0, px1 = max(0, ix - pr), min(w, ix + pr + 1)
        core_patch = g[py0:py1, px0:px1]
        if core_patch.size == 0:
            continue
        core_mean = float(np.mean(core_patch))

        radial_deficit = radial_med - core_mean
        radial_contrast = radial_deficit / max(radial_med, 1.0)
        radial_z = radial_deficit / radial_sigma

        if radial_contrast < CFG.limb6_radial_patch_contrast_min:
            continue
        if radial_z < CFG.limb6_radial_patch_sigma_min:
            continue

        # Quality is now based on BOTH independent confirmations.
        tangential_score = float(np.clip(
            (median_scale_contrast - 0.025) / 0.18, 0.0, 1.0
        ))
        radial_score = float(np.clip(
            (radial_contrast - CFG.limb6_radial_patch_contrast_min) / 0.12,
            0.0, 1.0
        ))
        z_score = float(np.clip(
            (radial_z - CFG.limb6_radial_patch_sigma_min) / 4.0,
            0.0, 1.0
        ))
        persistence_score = float(np.clip(
            n_scales / max(1, len(CFG.limb6_tangent_offsets_px)), 0.0, 1.0
        ))
        area_score = float(np.clip(
            math.log1p(area) / math.log1p(60.0), 0.0, 1.0
        ))
        shape_score = float(np.clip(compactness / 0.65, 0.0, 1.0))

        limb_confidence = (
            0.25 * tangential_score
            + 0.30 * radial_score
            + 0.20 * z_score
            + 0.10 * persistence_score
            + 0.08 * area_score
            + 0.07 * shape_score
        )

        if limb_confidence >= 0.76:
            limb_q = 5
        elif limb_confidence >= 0.56:
            limb_q = 4
        elif limb_confidence >= 0.36:
            limb_q = 3
        elif limb_confidence >= 0.22:
            limb_q = 2
        else:
            limb_q = 1

        rows.append({
            "spot_id": f"L{i}",
            "region_id": 60000 + i,
            "cx_px": x,
            "cy_px": y,
            "darkness_peak": float(darkness[iy, ix]),
            "region_area_px": area,
            "compactness": compactness,
            "limb_aspect_ratio": aspect_ratio,
            "detection_source": "limb_umbra_local",
            "local_contrast": local_peak,
            "local_prominence": local_peak,
            "limb_scale_contrast_min": min_scale_contrast,
            "limb_scale_contrast_median": median_scale_contrast,
            "limb_confirmed_scales": n_scales,
            "limb_radial_reference": radial_med,
            "limb_radial_sigma": radial_sigma,
            "limb_radial_contrast": radial_contrast,
            "limb_radial_z": radial_z,
            "limb_confidence": limb_confidence,
            "detection_quality": limb_q,
            "counted": False,
        })

    if not rows:
        return pd.DataFrame()

    limb = pd.DataFrame(rows)

    # Geometry/heliographic coordinates. Dedicated limb Q is preserved.
    xnorm = (limb["cx_px"] - xc) / radius
    ynorm = (limb["cy_px"] - yc) / radius
    limb["x_norm"] = xnorm
    limb["y_norm"] = ynorm
    limb["r_norm"] = np.sqrt(xnorm**2 + ynorm**2)

    B0 = math.radians(B0_deg)
    lat_list = []
    lon_list = []

    for x, y in zip(xnorm, ynorm):
        rho2 = min(float(x*x + y*y), 0.999999)
        z = math.sqrt(max(0.0, 1.0 - rho2))
        y_solar = -float(y)
        x_solar = float(x)

        lat = math.asin(y_solar * math.cos(B0) + z * math.sin(B0))
        lon = math.atan2(
            x_solar,
            z * math.cos(B0) - y_solar * math.sin(B0)
        )
        lat_list.append(math.degrees(lat))
        lon_list.append(math.degrees(lon))

    limb["heliographic_lat_deg"] = lat_list
    limb["heliographic_lon_deg"] = lon_list
    limb["hemisphere"] = np.where(limb["heliographic_lat_deg"] >= 0, "N", "S")
    limb["inside_activity_latitude_band"] = (
        limb["heliographic_lat_deg"].abs() <= CFG.max_abs_spot_lat_deg
    )

    limb["counted"] = (
        (limb["detection_quality"] >= CFG.limb6_quality_min)
        & limb["inside_activity_latitude_band"]
        & (limb["r_norm"] >= CFG.limb6_inner_rsun)
        & (limb["r_norm"] <= CFG.limb6_outer_rsun)
    )

    return limb


def build_umbra_pipeline(
    gray: np.ndarray,
    darkness: np.ndarray,
    rnorm: np.ndarray,
    active_inner: np.ndarray,
    activity_band: np.ndarray,
    xc: float,
    yc: float,
    radius: float,
    B0_deg: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Complete 6.0 detection pipeline.

    Stage A:
      adaptive whole-disk strong-umbra components -> Q -> retain Q4-Q5 ->
      first grouping into secure S and G.

    Stage B:
      secure Q4-Q5 seeds/G regions are searched locally for faint components ->
      Q final filter -> Q2+ retained and attached directly to their G.

    S detections from Stage A are never expanded by the faint search.
    """
    calib = robust_photosphere_calibration(
        darkness,
        rnorm,
        activity_band,
    )

    strong_all, strong_mask = detect_strong_umbrae(
        darkness,
        active_inner,
        calib["strong_umbra_threshold"],
    )

    if strong_all.empty:
        return strong_all, strong_all.copy(), calib

    strong_all = add_coordinates_and_quality(
        strong_all,
        darkness,
        xc,
        yc,
        radius,
        B0_deg,
    )

    # Only Q4-Q5 primary umbrae seed the secure pattern.
        # 6.1.2: very small but coherent inner-disk strong umbrae may score only Q2
    # because their area is tiny. Accept these only well inside the disk.
    inner_q2_strong = (
        (strong_all["detection_source"] == "strong_umbra")
        & (strong_all["detection_quality"] == 2)
        & strong_all["inside_activity_latitude_band"]
        & (strong_all["r_norm"] <= 0.70)
    )

    secure = strong_all[
        (
            strong_all["detection_quality"] >= CFG.strong_seed_quality_min
        )
        | inner_q2_strong
    ].copy().reset_index(drop=True)
    
    # secure = strong_all[
        # (strong_all["detection_quality"] >= CFG.strong_seed_quality_min)
        # & strong_all["inside_activity_latitude_band"]
    # ].copy().reset_index(drop=True)

    if secure.empty:
        candidates = strong_all.copy()
        return secure, candidates, calib

    secure["counted"] = True
    secure = group_spots(secure, radius)

    faint, faint_stats = detect_faint_umbrae_in_groups(
        gray,
        darkness,
        active_inner,
        strong_mask,
        secure,
        radius,
    )
    calib.update(faint_stats)
    calib["faint_main_darkness_min"] = CFG.faint_main_darkness_min

    candidate_parts = [strong_all.copy()]

    if not faint.empty:
        faint = add_coordinates_and_quality(
            faint,
            darkness,
            xc,
            yc,
            radius,
            B0_deg,
        )

        # Preserve group assignments through the quality calculation.
        faint["counted"] = (
            (faint["detection_quality"] >= CFG.faint_group_quality_min)
            & faint["inside_activity_latitude_band"]
        )

        candidate_parts.append(faint.copy())

        faint_keep = faint[faint["counted"]].copy()

        if not faint_keep.empty:
            # Align columns while preserving the secure group IDs.
            for col in secure.columns:
                if col not in faint_keep.columns:
                    faint_keep[col] = np.nan
            for col in faint_keep.columns:
                if col not in secure.columns:
                    secure[col] = np.nan

            secure = pd.concat(
                [secure[faint_keep.columns], faint_keep],
                ignore_index=True,
                sort=False,
            )

    # 6.0.2 independent extreme-limb pass. The inner-disk detector above
    # remains unchanged.
    limb = detect_extreme_limb_umbrae(
        gray=gray,
        darkness=darkness,
        rnorm=rnorm,
        activity_band=activity_band,
        existing_spots=secure,
        xc=xc,
        yc=yc,
        radius=radius,
        B0_deg=B0_deg,
    )

    if not limb.empty:
        candidate_parts.append(limb.copy())

        limb_keep = limb[limb["counted"]].copy()
        if not limb_keep.empty:
            # 6.0.3: group limb detections ONLY among themselves.
            # Established inner-disk group IDs are frozen and never recalculated.
            limb_keep = group_spots(limb_keep, radius)

            # Remap limb group IDs to unique IDs appended after all inner groups.
            limb_gid_order = list(dict.fromkeys(limb_keep["group_id"].tolist()))
            limb_gid_map = {
                old_gid: f"LIMB_{i+1}"
                for i, old_gid in enumerate(limb_gid_order)
            }
            limb_keep["group_id"] = limb_keep["group_id"].map(limb_gid_map)

            # Final 6.0.5 limb-only group diagnostic:
            # do NOT regroup the already established inner pattern.
            # Only decide whether each accepted limb group belongs to one
            # existing inner group or must remain an independent LIMB object.
            limb_keep = attach_limb_groups_to_existing_groups(
                inner_spots=secure,
                limb_spots=limb_keep,
                radius=radius,
            )
            
            # 6.0.6: final check ONLY for independent one-spot limb objects.
            limb_keep = validate_independent_limb_singles(limb_keep)

            for col in secure.columns:
                if col not in limb_keep.columns:
                    limb_keep[col] = np.nan
            for col in limb_keep.columns:
                if col not in secure.columns:
                    secure[col] = np.nan

            # Append limb objects last. make_group_table(sort=False) therefore
            # gives them the highest final G/S display numbers.
            secure = pd.concat(
                [secure[limb_keep.columns], limb_keep],
                ignore_index=True,
                sort=False,
            )

    candidates = pd.concat(
        candidate_parts,
        ignore_index=True,
        sort=False,
    )

    # IMPORTANT: no whole-disk regrouping here.
    return secure.reset_index(drop=True), candidates, calib


# ---------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------


def attach_limb_groups_to_existing_groups(
    inner_spots: pd.DataFrame,
    limb_spots: pd.DataFrame,
    radius: float,
) -> pd.DataFrame:
    """
    Final limb-only group diagnostic.

    IMPORTANT:
    - Existing inner-disk groups are never regrouped or modified.
    - Only accepted detections produced by the dedicated limb detector are
      examined.
    - A limb group is attached to an already established inner group if any
      of its members lies within the normal group-linking distance of any
      member of that inner group.
    - Otherwise the limb group keeps its own LIMB_x group_id and will become
      either a final G or S only during the normal final numbering step.

    This is deliberately a post-limb safety check, not a new detection pass.
    """
    if limb_spots.empty or inner_spots.empty:
        return limb_spots

    if "group_id" not in limb_spots.columns or "group_id" not in inner_spots.columns:
        return limb_spots

    out = limb_spots.copy()
    max_dist = CFG.group_distance_rsun * radius

    # Freeze the already-established inner groups.
    inner_groups = {}
    for gid, gdf in inner_spots.groupby("group_id", sort=False):
        inner_groups[gid] = list(zip(
            gdf["cx_px"].astype(float),
            gdf["cy_px"].astype(float),
        ))

    # Diagnose only the limb detector's own groups.
    for limb_gid, lgdf in out.groupby("group_id", sort=False):
        limb_xy = list(zip(
            lgdf["cx_px"].astype(float),
            lgdf["cy_px"].astype(float),
        ))

        best_gid = None
        best_dist = float("inf")

        for inner_gid, inner_xy in inner_groups.items():
            dmin = min(
                math.hypot(lx - ix, ly - iy)
                for lx, ly in limb_xy
                for ix, iy in inner_xy
            )

            if dmin < best_dist:
                best_dist = dmin
                best_gid = inner_gid

        if best_gid is not None and best_dist <= max_dist:
            out.loc[out["group_id"] == limb_gid, "group_id"] = best_gid

    return out
    
def validate_independent_limb_singles(
    limb_spots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final safety check for independent single limb detections.

    IMPORTANT:
    - Only objects that still have a LIMB_x group_id are examined.
    - Only one-member LIMB_x groups are examined.
    - Limb detections already attached to an established inner group
      are untouched.
    - Multi-spot independent limb groups are untouched.

    A single independent limb candidate must satisfy at least
    `limb_single_min_checks` of:
        1) limb_confidence >= threshold
        2) limb_radial_z >= threshold
        3) region_area_px >= threshold
    """
    if limb_spots.empty or "group_id" not in limb_spots.columns:
        return limb_spots

    out = limb_spots.copy()

    group_sizes = out.groupby("group_id").size().to_dict()
    drop_indices = []

    for idx, row in out.iterrows():
        gid = str(row["group_id"])

        # Already attached to an established inner group -> leave untouched.
        if not gid.startswith("LIMB_"):
            continue

        # Only independent single limb objects are checked.
        if int(group_sizes.get(gid, 0)) != 1:
            continue

        checks = [
            float(row.get("limb_confidence", 0.0))
            >= CFG.limb_single_confidence_min,

            float(row.get("limb_radial_z", 0.0))
            >= CFG.limb_single_radial_z_min,

            int(row.get("region_area_px", 0))
            >= CFG.limb_single_area_min_px,
        ]

        if sum(checks) < CFG.limb_single_min_checks:
            drop_indices.append(idx)

    if drop_indices:
        out = out.drop(index=drop_indices)

    return out.reset_index(drop=True)


def group_spots(spots: pd.DataFrame, radius: float) -> pd.DataFrame:
    """
    Group nearby counted spots.

    5.8.0 uses a 0.10 R_sun linking distance rather than 0.13 R_sun.
    This prevents distinct active regions from merging through one
    intermediate bridging spot.
    """
    if spots.empty:
        spots["group_id"] = []
        return spots

    n = len(spots)
    parent = list(range(n))
    max_dist = CFG.group_distance_rsun * radius

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    xy = spots[["cx_px", "cy_px"]].to_numpy()

    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
            if d <= max_dist:
                union(i, j)

    roots = {}
    gid = 1
    group_ids = []

    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = f"G{gid}"
            gid += 1
        group_ids.append(roots[r])

    spots = spots.copy()
    spots["group_id"] = group_ids
    return spots


# ---------------------------------------------------------------------
# Coordinates and quality
# ---------------------------------------------------------------------

def add_coordinates_and_quality(
    spots: pd.DataFrame,
    darkness: np.ndarray,
    xc: float,
    yc: float,
    radius: float,
    B0_deg: float,
) -> pd.DataFrame:

    if spots.empty:
        return spots

    out = spots.copy()

    xnorm = (out["cx_px"] - xc) / radius
    ynorm = (out["cy_px"] - yc) / radius

    out["x_norm"] = xnorm
    out["y_norm"] = ynorm
    out["r_norm"] = np.sqrt(xnorm**2 + ynorm**2)

    # After solar north-up orientation:
    # y image increases downward, therefore northern solar latitude uses -y.
    B0 = math.radians(B0_deg)

    lat_list = []
    lon_list = []

    for x, y in zip(xnorm, ynorm):
        rho2 = x*x + y*y
        rho2 = min(float(rho2), 0.999999)
        z = math.sqrt(max(0.0, 1.0 - rho2))

        # Stonyhurst heliographic approximation:
        # x positive to image right = solar west after north-up/E-left orientation
        y_solar = -float(y)
        x_solar = float(x)

        lat = math.asin(
            y_solar * math.cos(B0) + z * math.sin(B0)
        )

        lon = math.atan2(
            x_solar,
            z * math.cos(B0) - y_solar * math.sin(B0)
        )

        lat_list.append(math.degrees(lat))
        lon_list.append(math.degrees(lon))

    out["heliographic_lat_deg"] = lat_list
    out["heliographic_lon_deg"] = lon_list
    out["hemisphere"] = np.where(out["heliographic_lat_deg"] >= 0, "N", "S")

    # Detection quality (1-5):
    # image-derived only, no external reference.
    # Components:
    #  - contrast
    #  - region area
    #  - limb penalty
    #  - morphology / compactness
    #
    # This is a score for detection confidence, not astrophysical importance.
    q_values = []

    for _, r in out.iterrows():
        contrast = float(r["darkness_peak"])
        area = float(r["region_area_px"])
        rnorm = float(r["r_norm"])
        compact = float(r["compactness"])

        # Normalized sub-scores 0..1
        contrast_score = np.clip((contrast - 0.055) / 0.16, 0, 1)
        area_score = np.clip(math.log1p(area) / math.log1p(120.0), 0, 1)
        # Real sunspots close to the limb can still be reliable if their
        # contrast/area are strong. Use only a mild limb penalty.
        limb_score = np.clip((1.00 - rnorm) / 0.35, 0, 1)
        shape_score = np.clip(compact / 0.65, 0, 1)

        confidence = (
            0.50 * contrast_score
            + 0.28 * area_score
            + 0.12 * limb_score
            + 0.10 * shape_score
        )

        # Convert to 1-5
        if confidence >= 0.80:
            q = 5
        elif confidence >= 0.62:
            q = 4
        elif confidence >= 0.44:
            q = 3
        elif confidence >= 0.26:
            q = 2
        else:
            q = 1

        q_values.append(q)

    out["detection_quality"] = q_values

    # 5.9.1 hard physical activity belt for all detection sources.
    out["inside_activity_latitude_band"] = (
        out["heliographic_lat_deg"].abs() <= CFG.max_abs_spot_lat_deg
    )
    out["counted"] = (
        (out["detection_quality"] >= CFG.count_quality_min)
        & out["inside_activity_latitude_band"]
    )

    return out




def promote_targeted_q2_candidates(
    spots: pd.DataFrame,
    radius: float,
) -> pd.DataFrame:
    """
    5.9.1 safe Q2 promotion.

    Only Q2 detections from `companion` or `strong_split` may be retained,
    and only when they are 4..25 px from a secure Q>=3 detection.

    `isolated_faint` Q2 detections are never promoted; this prevents
    photospheric granulation around real active regions from inflating s.

    One secure spot plus one genuine companion Q2 is allowed to form a
    two-spot group. Promoted Q2 detections never act as anchors for others.
    """
    if spots.empty:
        return spots

    out = spots.copy()
    if "q2_group_promoted" not in out.columns:
        out["q2_group_promoted"] = False

    in_band = (
        out["heliographic_lat_deg"].abs()
        <= CFG.max_abs_spot_lat_deg
    )

    secure = out[
        (out["detection_quality"] >= CFG.count_quality_min)
        & in_band
    ].copy()

    if secure.empty:
        return out

    secure_xy = list(zip(
        secure["cx_px"].astype(float),
        secure["cy_px"].astype(float),
    ))

    for idx, row in out.iterrows():
        if int(row["detection_quality"]) != 2:
            continue

        if abs(float(row["heliographic_lat_deg"])) > CFG.max_abs_spot_lat_deg:
            continue

        source = str(row.get("detection_source", ""))
        if source not in {"companion", "strong_split"}:
            continue

        x = float(row["cx_px"])
        y = float(row["cy_px"])

        nearest = min(
            math.hypot(x - sx, y - sy)
            for sx, sy in secure_xy
        )

        if nearest < CFG.q2_group_attach_min_px:
            continue
        if nearest > CFG.q2_group_attach_max_px:
            continue

        out.at[idx, "counted"] = True
        out.at[idx, "q2_group_promoted"] = True

    return out

def promote_faint_group_members(
    spots: pd.DataFrame,
    radius: float,
) -> pd.DataFrame:
    """
    5.9.1 compatibility no-op.

    Isolated-faint Q1/Q2 detections are deliberately not promoted anymore.
    """
    return spots


def refine_existing_group_cores(
    spots: pd.DataFrame,
    darkness: np.ndarray,
    active_inner: np.ndarray,
    xc: float,
    yc: float,
    radius: float,
    B0_deg: float,
) -> pd.DataFrame:
    """
    5.9.2 post-group core refinement.

    This pass runs ONLY after the normal detector, Q2 companion promotion,
    rejection, and final grouping have already produced real groups.

    Only groups with >=2 counted members are revisited. Around each existing
    group member, a very small neighbourhood is searched for additional
    compact local darkness maxima. These new cores:
      - cannot create a new group;
      - cannot bridge two groups;
      - are attached directly to the existing group;
      - are never searched around isolated S detections;
      - must satisfy an independent local-prominence condition.

    The purpose is to recover visually resolved faint pores/umbrae inside an
    already certain active region without making the whole disk more sensitive.
    """
    if spots.empty or "group_id" not in spots.columns:
        return spots

    out = spots.copy()
    records = out.to_dict("records")

    # Only established groups (at least two already counted detections).
    group_sizes = out.groupby("group_id").size().to_dict()
    established = {
        gid for gid, n in group_sizes.items()
        if int(n) >= 2
    }
    if not established:
        return out

    # Small-scale baseline and prominence map.
    local_base = cv2.GaussianBlur(
        darkness,
        (0, 0),
        sigmaX=3.0,
        sigmaY=3.0,
    )
    prominence = darkness - local_base

    maxf = ndi.maximum_filter(
        darkness,
        size=3,
        mode="nearest",
    )
    local_max = darkness == maxf

    existing_xy = [
        (float(r["cx_px"]), float(r["cy_px"]))
        for r in records
    ]

    next_id = len(records) + 1

    for gid in established:
        gdf = out[out["group_id"] == gid]

        # Use only the ORIGINAL accepted group members as anchors.
        # Newly recovered cores never become anchors during this pass.
        anchors = list(zip(
            gdf["cx_px"].astype(float),
            gdf["cy_px"].astype(float),
        ))

        for ax, ay in anchors:
            yy, xx = np.ogrid[:darkness.shape[0], :darkness.shape[1]]
            neighbourhood = (
                ((xx - ax) ** 2 + (yy - ay) ** 2)
                <= CFG.group_core_search_radius_px ** 2
            )

            cand = (
                neighbourhood
                & active_inner
                & local_max
                & (darkness >= CFG.group_core_peak_threshold)
                & (prominence >= CFG.group_core_prominence)
            )

            yp, xp = np.where(cand)
            if len(xp) == 0:
                continue

            order = np.argsort(darkness[yp, xp])[::-1]
            added_for_anchor = 0

            for k in order:
                x = float(xp[k])
                y = float(yp[k])

                # Must be a genuinely distinct core.
                if any(
                    math.hypot(x - ex, y - ey)
                    < CFG.group_core_min_sep_px
                    for ex, ey in existing_xy
                ):
                    continue

                peak = float(darkness[int(round(y)), int(round(x))])

                # Build a tiny component around this peak only, so area and
                # compactness do not inherit a large neighbouring sunspot.
                iy = int(round(y))
                ix = int(round(x))
                y0 = max(0, iy - 3)
                y1 = min(darkness.shape[0], iy + 4)
                x0 = max(0, ix - 3)
                x1 = min(darkness.shape[1], ix + 4)

                patch = darkness[y0:y1, x0:x1]
                pmask = (
                    patch >= max(
                        CFG.group_core_peak_threshold * 0.80,
                        peak * 0.55,
                    )
                ).astype(np.uint8) * 255

                n, labels, stats, _ = cv2.connectedComponentsWithStats(
                    pmask, connectivity=8
                )

                py = iy - y0
                px = ix - x0
                lab = int(labels[py, px]) if (
                    0 <= py < labels.shape[0]
                    and 0 <= px < labels.shape[1]
                ) else 0

                if lab <= 0:
                    continue

                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < 2:
                    continue

                region = labels == lab
                contours, _ = cv2.findContours(
                    region.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                perimeter = cv2.arcLength(
                    contours[0], True
                ) if contours else 0.0
                compactness = (
                    4.0 * math.pi * area / (perimeter * perimeter)
                    if perimeter > 0 else 0.0
                )

                records.append(
                    {
                        "spot_id": f"S{next_id}",
                        "region_id": 40000 + next_id,
                        "cx_px": x,
                        "cy_px": y,
                        "darkness_peak": peak,
                        "region_area_px": area,
                        "compactness": float(
                            np.clip(compactness, 0.0, 1.0)
                        ),
                        "detection_source": "group_core_refine",
                        "limb_confirmed": False,
                        "counted": True,
                        "group_id": gid,
                        "q2_group_promoted": False,
                        "group_core_recovered": True,
                    }
                )

                existing_xy.append((x, y))
                next_id += 1
                added_for_anchor += 1

                if (
                    added_for_anchor
                    >= CFG.group_core_max_new_per_anchor
                ):
                    break

    refined = pd.DataFrame(records)

    # Recalculate coordinates and quality for all rows, then explicitly keep
    # the new group-core detections counted if they lie inside the activity
    # latitude band. Their Q remains diagnostic; group membership is the
    # reason they are accepted.
    refined = add_coordinates_and_quality(
        refined,
        darkness,
        xc,
        yc,
        radius,
        B0_deg,
    )

    if "group_core_recovered" not in refined.columns:
        refined["group_core_recovered"] = False
    else:
        refined["group_core_recovered"] = (
            refined["group_core_recovered"]
            .fillna(False)
            .astype(bool)
        )

    recovered = refined["group_core_recovered"]
    refined.loc[
        recovered
        & refined["inside_activity_latitude_band"],
        "counted"
    ] = True

    # Preserve direct group assignment for the recovered cores. Existing rows
    # already retain their group_id from the input dataframe.
    return refined


def make_group_table(spots: pd.DataFrame) -> pd.DataFrame:
    if spots.empty:
        return pd.DataFrame(
            columns=[
                "group_id",
                "display_id",
                "n_spots_detected",
                "n_spots_counted",
                "included_in_R",
                "group_detection_quality",
                "mean_x_px",
                "mean_y_px",
                "mean_x_norm",
                "mean_y_norm",
                "heliographic_lat_deg",
                "heliographic_lon_deg",
                "hemisphere",
            ]
        )

    group_rows = []

    for group_id, gdf in spots.groupby("group_id", sort=False):
        lat = float(gdf["heliographic_lat_deg"].mean())
        lon = float(gdf["heliographic_lon_deg"].mean())

        group_quality = int(
            np.clip(round(float(gdf["detection_quality"].median())), 1, 5)
        )

        n_detected = int(len(gdf))
        n_counted = int(gdf["counted"].sum())
        included = bool(n_counted > 0)

        group_rows.append(
            {
                "group_id": group_id,
                "display_id": "",
                "n_spots_detected": n_detected,
                "n_spots_counted": n_counted,
                "included_in_R": included,
                "group_detection_quality": group_quality,
                "mean_x_px": float(gdf["cx_px"].mean()),
                "mean_y_px": float(gdf["cy_px"].mean()),
                "mean_x_norm": float(gdf["x_norm"].mean()),
                "mean_y_norm": float(gdf["y_norm"].mean()),
                "heliographic_lat_deg": lat,
                "heliographic_lon_deg": lon,
                "hemisphere": "N" if lat >= 0 else "S",
            }
        )

    groups = pd.DataFrame(group_rows)

    # Renumber ONLY groups that actually contribute to R.
    # Multi-spot groups: G1, G2, ...
    # Isolated single spots: S1, S2, ...
    gnum = 1
    snum = 1
    for idx, row in groups.iterrows():
        if not bool(row["included_in_R"]):
            continue
        if int(row["n_spots_counted"]) == 1:
            groups.at[idx, "display_id"] = f"S{snum}"
            snum += 1
        else:
            groups.at[idx, "display_id"] = f"G{gnum}"
            gnum += 1

    return groups


# ---------------------------------------------------------------------
# Faculae
# ---------------------------------------------------------------------

def merge_nearby_facula_fields(
    fields: pd.DataFrame,
    xc: float,
    yc: float,
    radius: float,
    B0_deg: float,
) -> pd.DataFrame:
    """Merge nearby detected facular fragments into extended fields.

    The initial detector deliberately keeps the thresholding conservative,
    which can split one physically extended facular field into several
    disconnected bright islands. This post-processing step merges only
    nearby fragments that
      * are on the same east/west side of the solar disk,
      * are within ``facula_merge_distance_rsun`` of one another, and
      * have similar heliographic latitude.

    Single-linkage is intentional here: several small islands can form one
    larger field even when the first and last islands are farther apart.
    """
    if fields.empty or len(fields) == 1:
        out = fields.copy()
        if not out.empty:
            out["component_count"] = 1
            out["member_ids"] = out["facula_id"].astype(str)
            out["facula_id"] = [f"FAC{i+1}" for i in range(len(out))]
        return out

    work = fields.reset_index(drop=True).copy()
    n = len(work)
    parent = list(range(n))
    max_dist = CFG.facula_merge_distance_rsun * radius

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        xi = float(work.at[i, "cx_px"])
        yi = float(work.at[i, "cy_px"])
        lati = float(work.at[i, "heliographic_lat_deg"])
        side_i = -1 if xi < xc else 1

        for j in range(i + 1, n):
            xj = float(work.at[j, "cx_px"])
            yj = float(work.at[j, "cy_px"])
            latj = float(work.at[j, "heliographic_lat_deg"])
            side_j = -1 if xj < xc else 1

            if side_i != side_j:
                continue
            if abs(lati - latj) > CFG.facula_merge_max_lat_diff_deg:
                continue
            if math.hypot(xi - xj, yi - yj) <= max_dist:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    B0 = math.radians(B0_deg)
    rows = []

    # Keep a stable order based on the earliest original component.
    grouped_indices = sorted(groups.values(), key=lambda ids: min(ids))

    for new_id, ids in enumerate(grouped_indices, start=1):
        sub = work.iloc[ids]
        areas = sub["area_px"].astype(float).to_numpy()
        weights = np.maximum(areas, 1.0)

        cx = float(np.average(sub["cx_px"].astype(float), weights=weights))
        cy = float(np.average(sub["cy_px"].astype(float), weights=weights))
        area_total = int(round(float(areas.sum())))
        mean_bright = float(np.average(
            sub["mean_local_brightness"].astype(float),
            weights=weights,
        ))
        peak_bright = float(sub["peak_local_brightness"].astype(float).max())

        xnorm = (cx - xc) / radius
        ynorm = (cy - yc) / radius
        r_here = float(math.hypot(xnorm, ynorm))
        rho2 = min(float(xnorm * xnorm + ynorm * ynorm), 0.999999)
        z = math.sqrt(max(0.0, 1.0 - rho2))
        y_solar = -ynorm
        x_solar = xnorm

        lat = math.degrees(math.asin(
            y_solar * math.cos(B0) + z * math.sin(B0)
        ))
        lon = math.degrees(math.atan2(
            x_solar,
            z * math.cos(B0) - y_solar * math.sin(B0)
        ))

        rows.append({
            "facula_id": f"FAC{new_id}",
            "cx_px": cx,
            "cy_px": cy,
            "r_norm": r_here,
            "area_px": area_total,
            "mean_local_brightness": mean_bright,
            "peak_local_brightness": peak_bright,
            "heliographic_lat_deg": lat,
            "heliographic_lon_deg": lon,
            "hemisphere": "N" if lat >= 0 else "S",
            "component_count": int(len(ids)),
            "member_ids": ";".join(sub["facula_id"].astype(str).tolist()),
        })

    return pd.DataFrame(rows)


def detect_facula_fields(
    gray: np.ndarray,
    rnorm: np.ndarray,
    xc: float,
    yc: float,
    radius: float,
    B0_deg: float,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Detect white-light facular fields only near the solar limb.

    The purpose is not to measure a total facular area over the whole disk.
    Instead, the detector finds coherent bright regions in a limb annulus and
    returns one labelled object per connected facular field. This mirrors the
    H-alpha AR-style annotation: the field is identified by its centroid and
    label, without drawing a contour around it.

    2026-09-21 refinement:
    facular fields are restricted not only to the limb annulus but also to
    the usual solar activity belt, because white-light faculae are normally
    associated with active regions and are not expected all around the limb.
    This removes many false detections near the polar limb.
    """
    g = gray.astype(np.float32)

    annulus = (
        (rnorm >= CFG.facula_inner_rsun)
        & (rnorm <= CFG.facula_outer_rsun)
    )

    facula_lat_band = heliographic_activity_band_mask(
        gray.shape,
        xc,
        yc,
        radius,
        B0_deg,
        CFG.facula_max_abs_lat_deg,
    )

    # Local background at the same general limb zone. This is intentionally
    # much smaller than the whole-disk limb-darkening normalization scale.
    local_bg = cv2.GaussianBlur(
        gray, (0, 0),
        sigmaX=CFG.facula_local_sigma_px,
        sigmaY=CFG.facula_local_sigma_px,
    ).astype(np.float32)

    local_bright = np.zeros_like(g, dtype=np.float32)
    valid = local_bg > 5.0
    local_bright[valid] = (g[valid] - local_bg[valid]) / local_bg[valid]

    mask = (
        annulus
        & facula_lat_band
        & (local_bright >= CFG.facula_brightness_threshold)
    ).astype(np.uint8) * 255

    # Remove isolated bright granulation while keeping extended facular
    # structures. Closing reconnects small fragmented patches.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    rows = []
    clean = np.zeros_like(mask)
    fid = 1
    B0 = math.radians(B0_deg)

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < CFG.facula_min_area_px or area > CFG.facula_max_area_px:
            continue

        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        if min(width, height) < CFG.facula_min_radial_width_px:
            continue
        if max(width, height) < CFG.facula_min_tangential_width_px:
            continue

        region = labels == i
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue

        vals = local_bright[ys, xs]
        peak_k = int(np.argmax(vals))
        peak = float(vals[peak_k])
        mean_bright = float(np.mean(vals))

        cx = float(centroids[i][0])
        cy = float(centroids[i][1])

        ix = int(np.clip(round(cx), 0, rnorm.shape[1] - 1))
        iy = int(np.clip(round(cy), 0, rnorm.shape[0] - 1))
        r_here = float(rnorm[iy, ix])

        xnorm = (cx - xc) / radius
        ynorm = (cy - yc) / radius
        rho2 = min(float(xnorm * xnorm + ynorm * ynorm), 0.999999)
        z = math.sqrt(max(0.0, 1.0 - rho2))
        y_solar = -float(ynorm)
        x_solar = float(xnorm)
        lat = math.degrees(math.asin(
            y_solar * math.cos(B0) + z * math.sin(B0)
        ))
        lon = math.degrees(math.atan2(
            x_solar,
            z * math.cos(B0) - y_solar * math.sin(B0)
        ))

        rows.append({
            "facula_id": f"FAC{fid}",
            "cx_px": cx,
            "cy_px": cy,
            "r_norm": r_here,
            "area_px": area,
            "mean_local_brightness": mean_bright,
            "peak_local_brightness": peak,
            "heliographic_lat_deg": lat,
            "heliographic_lon_deg": lon,
            "hemisphere": "N" if lat >= 0 else "S",
        })
        clean[region] = 255
        fid += 1

    fields = pd.DataFrame(rows)

    # A real white-light facular field is usually larger than one thresholded
    # bright island. Merge nearby fragments only after all conservative
    # detection filters have passed, so the detection sensitivity itself is
    # unchanged.
    fields = merge_nearby_facula_fields(
        fields=fields,
        xc=xc,
        yc=yc,
        radius=radius,
        B0_deg=B0_deg,
    )

    return fields, clean, {
        "facula_field_count": int(len(fields)),
    }


# ---------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------

GROUP_COLORS = [
    (0, 0, 255),
    (0, 200, 0),
    (255, 0, 0),
    (255, 0, 255),
    (0, 200, 255),
    (255, 180, 0),
    (160, 255, 0),
    (0, 150, 255),
    (220, 0, 120),
    (120, 120, 255),
    (255, 255, 0),
    (180, 0, 255),
]


def draw_direction_markers(
    img: np.ndarray,
    xc: float,
    yc: float,
    radius: float,
) -> np.ndarray:
    """Draw simple solar-direction labels on the already oriented image.

    After orient_solar_north_up(), solar north is up and solar west is right,
    so only fixed label placement is needed here.
    """
    out = img
    h, w = out.shape[:2]

    def clamp_pos(x: int, y: int) -> tuple[int, int]:
        return (
            max(0, min(w - 20, x)),
            max(20, min(h - 5, y)),
        )

    n_pos = clamp_pos(
        int(round(xc - 10)),
        int(round(yc - radius - 90)),
    )
    w_pos = clamp_pos(
        int(round(xc + radius + 90)),
        int(round(yc + 8)),
    )

    cv2.putText(
        out,
        "N",
        n_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "W",
        w_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def draw_facula_field_labels(
    img: np.ndarray,
    facula_fields: pd.DataFrame,
) -> np.ndarray:
    """Label detected facular fields without drawing their outlines."""
    if facula_fields.empty:
        return img

    out = img
    placed: list[tuple[float, float]] = []

    for _, row in facula_fields.iterrows():
        x = float(row["cx_px"])
        y = float(row["cy_px"])

        # Avoid unreadable stacks of labels for fragmented neighbouring fields.
        if any(
            math.hypot(x - px, y - py) < CFG.facula_label_min_separation_px
            for px, py in placed
        ):
            continue

        label = str(row["facula_id"])
        cv2.putText(
            out,
            label,
            (int(round(x)) + 8, int(round(y)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        placed.append((x, y))

    return out


def annotate_image(
    img: np.ndarray,
    spots: pd.DataFrame,
    groups: pd.DataFrame,
    xc: float,
    yc: float,
    radius: float,
    g: int,
    s: int,
    R_raw: int,
    P_deg: float,
    B0_deg: float,
    date_str: str,
    time_ut_str: str,
    observer_place: str,
    facula_fields: pd.DataFrame,
) -> np.ndarray:

    out = img.copy()

    cv2.circle(
        out,
        (int(round(xc)), int(round(yc))),
        int(round(radius)),
        (220, 220, 220),
        1,
    )

    out = draw_direction_markers(out, xc, yc, radius)
    out = draw_facula_field_labels(out, facula_fields)

    visible_groups = groups[groups["included_in_R"] == True].copy()

    group_color = {}
    for i, gid in enumerate(visible_groups["group_id"].tolist()):
        group_color[gid] = GROUP_COLORS[i % len(GROUP_COLORS)]

    # Zero-spot safety: on a completely spotless image the accepted-spots
    # DataFrame can be empty and may not contain the "counted" column.
    # In that case keep an empty counted_spots table instead of raising KeyError.
    if spots.empty or "counted" not in spots.columns:
        counted_spots = spots.iloc[0:0].copy()
    else:
        counted_spots = spots[spots["counted"] == True]

    for _, row in counted_spots.iterrows():
        gid = row["group_id"]
        if gid not in group_color:
            continue
        x = int(round(row["cx_px"]))
        y = int(round(row["cy_px"]))
        c = group_color[gid]
        cv2.circle(out, (x, y), 5, c, 2)

    # Display labels:
    #   multi-spot groups -> G1 (4)
    #   isolated single spots -> S1
    for _, row in visible_groups.iterrows():
        gid = row["group_id"]
        gspots = counted_spots[counted_spots["group_id"] == gid]
        if gspots.empty:
            continue

        gx = int(round(gspots["cx_px"].mean()))
        gy = int(round(gspots["cy_px"].mean()))
        n_spots = int(row["n_spots_counted"])
        display_id = str(row["display_id"])
        c = group_color[gid]

        label = display_id if n_spots == 1 else f"{display_id} ({n_spots})"

        cv2.putText(
            out,
            label,
            (gx + 10, gy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            c,
            2,
            cv2.LINE_AA,
        )

    lines = [
        f"Date (UT) = {date_str} {time_ut_str}",
        f"Place = {observer_place}",
        f"Groups g = {g}",
        f"Spots s = {s}",
        f"R_raw = {R_raw}",
        f"P = {P_deg:+.2f} deg",
        f"B0 = {B0_deg:+.2f} deg",
    ]

    y0 = 28
    for line in lines:
        cv2.putText(
            out,
            line,
            (18, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y0 += 24

    return out


# ---------------------------------------------------------------------
# CSV handling
# ---------------------------------------------------------------------

def append_or_replace_daily_summary(path: Path, row: dict):
    new = pd.DataFrame([row])

    if path.exists():
        old = pd.read_csv(path)

        # Replace same date+time row if already present
        if {"date", "time_UT"}.issubset(old.columns):
            keep = ~(
                (old["date"].astype(str) == str(row["date"]))
                & (old["time_UT"].astype(str) == str(row["time_UT"]))
            )
            old = old[keep]

        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    out = out.sort_values(["date", "time_UT"]).reset_index(drop=True)
    out.to_csv(path, index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def process_image(
    image_path: Path,
    outdir: Path,
    camera_angle_deg: float,
):
    outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    dt_utc = parse_datetime_from_filename(image_path)

    # 1) Orient solar north up
    oriented, eph = orient_solar_north_up(
        img,
        dt_utc,
        camera_angle_deg=camera_angle_deg,
    )

    # 2) Disk detection after orientation
    gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
    xc, yc, radius = detect_solar_disk(gray)

    rr, rnorm, disk, inner, darkness, brightness = normalized_maps(
        gray, xc, yc, radius
    )

    # 5.9.1 physical activity belt: S55..N55 only.
    # This is latitude-only; no east/west longitude cut is applied.
    activity_band = heliographic_activity_band_mask(
        gray.shape,
        xc,
        yc,
        radius,
        eph["B0_deg"],
        CFG.max_abs_spot_lat_deg,
    )
    active_inner = inner & activity_band

    # 3) 6.0.0 two-stage umbra/pore detection.
    # The orientation/time/ephemeris pipeline above is unchanged.
    spots, spot_candidates, calibration = build_umbra_pipeline(
        gray=gray,
        darkness=darkness,
        rnorm=rnorm,
        active_inner=active_inner,
        activity_band=activity_band,
        xc=xc,
        yc=yc,
        radius=radius,
        B0_deg=eph["B0_deg"],
    )

    # All objects reaching this point are accepted counted umbra/pore
    # components. Faint components already carry the secure group's group_id.
    if not spots.empty:
        spots["counted"] = True

    # Make spot IDs group-relative for CSV readability
    if not spots.empty:
        relative_ids = []
        counters = {}
        for gid in spots["group_id"]:
            counters.setdefault(gid, 0)
            counters[gid] += 1
            relative_ids.append(f"{gid}-S{counters[gid]}")
        spots["spot_id"] = relative_ids

    groups = make_group_table(spots)

    # Add the final visible label (G1/G2... or S1/S2...) to spots.csv.
    if not spots.empty and not groups.empty:
        display_map = dict(zip(groups["group_id"], groups["display_id"]))
        spots["display_id"] = spots["group_id"].map(display_map)

    # Conservative Wolf counting:
    # only quality 3-5 spots contribute. A group contributes only if it has
    # at least one counted spot.
    s = int(spots["counted"].sum()) if not spots.empty else 0
    g = int(groups["included_in_R"].sum()) if not groups.empty else 0
    R_raw = int(10 * g + s)

    # 4) Facular fields: limb-only, labelled like H-alpha AR detections.
    facula_fields, fac_mask, fac_stats = detect_facula_fields(
        gray=gray,
        rnorm=rnorm,
        xc=xc,
        yc=yc,
        radius=radius,
        B0_deg=eph["B0_deg"],
    )

    # 5) Date/time columns
    date_str = dt_utc.strftime("%Y-%m-%d")
    time_str = dt_utc.strftime("%H:%M:%S")
    dt_local = dt_utc.astimezone(ZoneInfo(LOCAL_TIMEZONE))
    local_date_str = dt_local.strftime("%Y-%m-%d")
    local_time_str = dt_local.strftime("%H:%M:%S")
    local_tz_name = dt_local.tzname() or LOCAL_TIMEZONE

    spots.insert(0, "date", date_str)
    spots.insert(1, "time_UT", time_str)
    if not spots.empty:
        spots["group_spot_count_detected"] = spots.groupby("group_id")["spot_id"].transform("count")
        spots["group_spot_count_counted"] = spots.groupby("group_id")["counted"].transform("sum").astype(int)

    groups.insert(0, "date", date_str)
    groups.insert(1, "time_UT", time_str)

    # 6) Annotated image
    annotated = annotate_image(
        oriented,
        spots,
        groups,
        xc,
        yc,
        radius,
        g,
        s,
        R_raw,
        eph["P_deg"],
        eph["B0_deg"],
        date_str,
        time_str,
        "Esztergom",
        facula_fields,
    )

    stem = image_path.stem

    annotated_path = outdir / f"{stem}_annotated.jpg"
    spots_path = outdir / "spots.csv"
    candidates_path = outdir / "spot_candidates.csv"
    groups_path = outdir / "groups.csv"
    faculae_path = outdir / "facula_fields.csv"
    daily_path = outdir / "daily_summary.csv"

    cv2.imwrite(
        str(annotated_path),
        annotated,
        [int(cv2.IMWRITE_JPEG_QUALITY), CFG.jpeg_quality],
    )

    spots.to_csv(spots_path, index=False)
    spot_candidates.to_csv(candidates_path, index=False)
    groups.to_csv(groups_path, index=False)
    facula_fields.to_csv(faculae_path, index=False)

    daily_row = {
        "date": date_str,
        "time_UT": time_str,
        "groups_g": g,
        "spots_s": s,
        "R_raw": R_raw,
        "facula_field_count": fac_stats["facula_field_count"],
        "solar_P_deg": eph["P_deg"],
        "solar_B0_deg": eph["B0_deg"],
        "parallactic_angle_deg": eph["parallactic_angle_deg"],
        "solar_hour_angle_deg": eph["solar_hour_angle_deg"],
        "observer_lat_deg": eph["observer_lat_deg"],
        "observer_lon_deg": eph["observer_lon_deg"],
        "camera_angle_deg": eph["camera_angle_deg"],
        "applied_rotation_deg_ccw": eph["applied_rotation_deg_ccw"],
        "solar_center_x_px": xc,
        "solar_center_y_px": yc,
        "solar_radius_px": radius,
        "photosphere_dark_median": calibration.get("photosphere_dark_median"),
        "photosphere_dark_sigma": calibration.get("photosphere_dark_sigma"),
        "strong_umbra_threshold": calibration.get("strong_umbra_threshold"),
        "faint_local_threshold": calibration.get("faint_local_threshold"),
        "faint_local_noise_sigma": calibration.get("faint_local_noise_sigma"),
        "faint_main_darkness_min": calibration.get("faint_main_darkness_min"),
        "limb6_inner_rsun": CFG.limb6_inner_rsun,
        "limb6_outer_rsun": CFG.limb6_outer_rsun,
        "limb6_min_local_contrast": CFG.limb6_min_local_contrast,
        "limb6_scale_contrast_min": CFG.limb6_scale_contrast_min,
        "limb6_min_confirmed_scales": CFG.limb6_min_confirmed_scales,
        "limb6_radial_patch_sigma_min": CFG.limb6_radial_patch_sigma_min,
        "limb6_radial_patch_contrast_min": CFG.limb6_radial_patch_contrast_min,
        "image_name": image_path.name,
    }

    append_or_replace_daily_summary(daily_path, daily_row)

    print()
    print("Seestar Sunspot Counter 6.1.1")
    print("-----------------------")
    print(f"Image:               {image_path.name}")
    print(f"Local image time:    {local_date_str} {local_time_str} {local_tz_name}")
    print(f"UTC used for eph.:   {date_str} {time_str}")
    print(f"P-angle:             {eph['P_deg']:.3f} deg")
    print(f"B0:                  {eph['B0_deg']:.3f} deg")
    print(f"Parallactic angle:   {eph['parallactic_angle_deg']:.3f} deg")
    print(f"Observer:            {eph['observer_lat_deg']:.5f}, {eph['observer_lon_deg']:.5f}")
    print(f"Applied rotation:    {eph['applied_rotation_deg_ccw']:.3f} deg CCW")
    print(f"Photosphere sigma:   {calibration.get('photosphere_dark_sigma', float('nan')):.5f}")
    print(f"Strong umbra thr.:   {calibration.get('strong_umbra_threshold', float('nan')):.5f}")
    flt = calibration.get("faint_local_threshold", float("nan"))
    print(f"Faint local thr.:    {flt:.5f}" if pd.notna(flt) else "Faint local thr.:    n/a")
    print(f"Groups g:            {g}")
    print(f"Spots s:             {s}")
    print(f"R_raw:               {R_raw}")
    print(f"Facular fields:      {fac_stats['facula_field_count']}")
    print()
    print("Outputs:")
    print(f"  {annotated_path}")
    print(f"  {spots_path}")
    print(f"  {candidates_path}")
    print(f"  {groups_path}")
    print(f"  {faculae_path}")
    print(f"  {daily_path}")
    print()

    return {
        "annotated": annotated_path,
        "spots": spots_path,
        "candidates": candidates_path,
        "groups": groups_path,
        "faculae": faculae_path,
        "daily": daily_path,
        "g": g,
        "s": s,
        "R_raw": R_raw,
    }


def build_argparser():
    p = argparse.ArgumentParser(
        description="Seestar white-light sunspot counter - automatically processes the newest image in WORKDIR"
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=WORKDIR,
        help=f"Output directory (default: {WORKDIR})",
    )
    p.add_argument(
        "--camera-angle-deg",
        type=float,
        default=0.0,
        help=(
            "Optional fixed residual Seestar camera/mount orientation offset "
            "in degrees, applied after parallactic-angle correction. Default 0."
        ),
    )
    return p


def main():
    args = build_argparser().parse_args()

    image_path = find_latest_solar_image(WORKDIR)

    outdir = args.outdir
    if not outdir.is_absolute():
        outdir = WORKDIR / outdir

    print(f"Automatically selected image: {image_path.name}")

    process_image(
        image_path=image_path,
        outdir=outdir,
        camera_angle_deg=args.camera_angle_deg,
    )


if __name__ == "__main__":
    main()
