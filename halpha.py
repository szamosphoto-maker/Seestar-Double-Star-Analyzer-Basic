# -*- coding: utf-8 -*-
r"""
halpha_v17_4_fixed_csv_names.py
---------------------

H-alfa teljes napkorong elemző – filamentumok + kétágú AR detektor.

Munkakönyvtár:
    D:\Astro\Isotool\output

Elv:
- az eredeti (nem tükrözött) H-alfa képből dolgozik
- a daily_summary.csv-ből a solar_P_deg és solar_B0_deg értékeket olvassa
- a forgatás:
      rotation_deg = ACUTER_CAMERA_ANGLE_DEG - solar_P_deg

Újdonság a v12-höz képest:
- filamentumrész változatlan
- AR/plage rész kétágú:
    1) belső zóna   : r/R <= 0.90
    2) peremzóna    : 0.90 < r/R <= 0.99
- a peremzóna külön plage-osztályozót kap
- a peremzóna külön AR-csoportosítót kap
- a végén a két zóna AR-jeit közös listába rendezi
- a plage/fáklyamező detektálás csak belső segédlépés az AR-ekhez;
  külön fáklyamező-számot nem ír ki és nem ment végső eredményként
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import re
import math
from datetime import datetime, timedelta

WORK_DIR = Path(r"D:\Astro\Isotool\output")
ALGORITHM_VERSION = "17.4"

# ---------- Geometria ----------
ACUTER_CAMERA_ANGLE_DEG = 27.83
HALPHA_FILENAME_UTC_OFFSET_HOURS = 2

# ---------- Filamentumok ----------
FILAMENT_INNER_R = 0.95
FILAMENT_DARK_RATIO = 0.888
FILAMENT_MIN_AREA = 50
FILAMENT_MIN_ELONGATION = 2.1
FILAMENT_MIN_SKELETON = 18
FILAMENT_BIG_AREA = 240

THIN_FILAMENT_DARK_RATIO = 0.925
THIN_FILAMENT_MIN_AREA = 22
THIN_FILAMENT_MIN_ELONGATION = 4.0
THIN_FILAMENT_MIN_SKELETON = 18
THIN_FILAMENT_MAX_WIDTH = 18

FILAMENT_LINK_GAP_FRAC = 0.040
FILAMENT_LINK_TIGHT_FRAC = 0.018
FILAMENT_LINK_MAX_ANGLE_DEG = 38.0
NORMAL_SEED_MIN_Q = 4

# ---------- Plage / AR ----------
PLAGE_OUTER_R = 0.99
AR_INNER_ZONE_R = 0.90

PRIMARY_PLAGE_RATIO = 1.135
PLAGE_MIN_AREA = 55

MAX_AR_LAT_DEG = 55.0
MAX_AR_LIMB_LAT_DEG = 50.0
SOFT_AR_LAT_DEG = 40.0
AR_MID_LAT_DEG = 50.0

# Belső AR
AR_GROUP_FRAC_INNER = 0.050
AR_SINGLE_STRONG_MIN_AREA = 220
AR_SINGLE_STRONG_MIN_RATIO = 1.18

# Perem AR
AR_GROUP_FRAC_LIMB = 0.040
AR_LIMB_SINGLE_MIN_AREA = 80
AR_LIMB_SINGLE_MIN_RATIO = 1.155
AR_LIMB_MIN_AREA = 40
AR_LIMB_MIN_RATIO = 1.150
AR_VERY_LIMB_START_R = 0.95
AR_VERY_LIMB_MAX_LAT_DEG = 50.0
AR_VERY_LIMB_MIN_AREA = 90
AR_VERY_LIMB_MIN_RATIO = 1.185

# A 35 fok folotti peremi singleton csak akkor maradhat meg, ha eleg eros.
# Ez a magasabban fekvo, szemcses hatterbol kialakulo hamis AR-eket szuri.
AR_LIMB_HIGH_LAT_START_DEG = 35.0
AR_LIMB_HIGH_LAT_MIN_AREA = 130
AR_LIMB_HIGH_LAT_MIN_RATIO = 1.18
AR_LIMB_HIGH_LAT_COMPACT_MIN_AREA = 80
AR_LIMB_HIGH_LAT_COMPACT_MIN_RATIO = 1.235

# A filamentum szelen keletkezo vilagos halo ne valjon kulon AR-re.
AR_FILAMENT_REJECT_DIST_FRAC = 0.018
AR_INNER_FILAMENT_REJECT_MAX_AREA = 350
AR_LIMB_FILAMENT_REJECT_MAX_AREA = 180

# Masodik, csak alacsony szelessegu peremteruleten mukodo mentesi ag.
# A vetuletben osszenyomott, diffuz AR-ek gyengebb atlagos kontraszt mellett
# is visszajuthatnak, de csak akkor, ha eros fenyessegi magjuk is van.
AR_LIMB_RESCUE_INNER_R = 0.86
AR_LIMB_RESCUE_RATIO = 1.10
AR_LIMB_RESCUE_MIN_AREA = 35
AR_LIMB_RESCUE_MIN_PEAK_RATIO = 1.18
AR_LIMB_RESCUE_MAX_LAT_DEG = 35.0
AR_LIMB_RESCUE_SMOOTH_SIGMA = 1.5
AR_LIMB_RESCUE_COMPANION_DIST_FRAC = 0.30
AR_LIMB_RESCUE_COMPANION_LAT_DELTA_DEG = 6.0
AR_LIMB_RESCUE_MIN_OUTWARD_DELTA_R = 0.12
AR_LIMB_RESCUE_EXISTING_AR_DIST_FRAC = 0.05
AR_LIMB_RESCUE_FINAL_MERGE_FRAC = 0.085

# Végső AR összevonás belső/perem között
AR_FINAL_MERGE_FRAC = 0.045

LOCAL_BLUR_FRAC = 0.045

ANNOTATED_SUFFIX = "_halpha.jpg"
FEATURES_FILENAME = "halpha_features.csv"
SUMMARY_FILENAME = "halpha_summary.csv"


# ============================================================
# Dátum / fájlválasztás
# ============================================================

def parse_halpha_datetime(name: str):
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', name)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    tail = name[m.end():]
    digits = re.findall(r'\d+', tail)
    if not digits:
        return datetime(y, mo, d)

    first = digits[0]
    hh = mm = ss = 0
    try:
        if len(first) >= 6:
            hh, mm, ss = int(first[:2]), int(first[2:4]), int(first[4:6])
        elif len(first) == 4:
            hh, mm = int(first[:2]), int(first[2:4])
            if len(digits) >= 2 and len(digits[1]) <= 2:
                ss = int(digits[1])
        elif len(first) == 2 and len(digits) >= 2:
            hh = int(first)
            mm = int(digits[1])
            if len(digits) >= 3:
                ss = int(digits[2])
        return datetime(y, mo, d, hh, mm, ss)
    except Exception:
        return datetime(y, mo, d)


def choose_latest_halpha_image(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files = [
        p for p in files
        if "_halpha" not in p.stem.lower()
        and "_annotated" not in p.stem.lower()
        and "solar-raw" not in p.name.lower()
        and "solar_raw" not in p.name.lower()
    ]
    if not files:
        raise FileNotFoundError("Nem található H-alfa bemeneti kép.")

    def score(p):
        s = 0
        n = p.name.lower()
        if "pano" in n:
            s += 10
        if "sun" in n:
            s += 5
        if "surface" in n:
            s += 2
        return (s, p.stat().st_mtime)

    return sorted(files, key=score, reverse=True)[0]


def find_latest_daily_summary(folder: Path):
    files = list(folder.glob("daily_summary*.csv"))
    if not files:
        raise FileNotFoundError("Nem található daily_summary*.csv a munkakönyvtárban.")
    return max(files, key=lambda p: p.stat().st_mtime)


def read_solar_geometry(folder: Path, ha_file: Path):
    summary_path = find_latest_daily_summary(folder)
    df = pd.read_csv(summary_path, sep=None, engine="python")

    required = {"date", "time_UT", "solar_P_deg", "solar_B0_deg"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError("Hiányzó oszlop(ok): " + ", ".join(sorted(missing)))

    ha_local = parse_halpha_datetime(ha_file.name)
    if ha_local is None:
        raise RuntimeError("A H-alfa fájlnévből nem olvasható ki a dátum.")

    ha_utc = ha_local - timedelta(hours=HALPHA_FILENAME_UTC_OFFSET_HOURS)

    rows = []
    for idx, row in df.iterrows():
        try:
            dt = datetime.fromisoformat(f"{row['date']}T{row['time_UT']}")
            delta = abs((dt - ha_utc).total_seconds())
            rows.append((delta, idx, dt))
        except Exception:
            continue

    if not rows:
        raise RuntimeError("A daily_summary.csv időadatai nem értelmezhetők.")

    same_day = [x for x in rows if x[2].date() == ha_utc.date()]
    if same_day:
        rows = same_day

    rows.sort(key=lambda x: x[0])
    delta_s, idx, ref_dt = rows[0]
    row = df.loc[idx]

    return {
        "summary_path": summary_path,
        "reference_utc": ref_dt,
        "delta_seconds": float(delta_s),
        "P_deg": float(row["solar_P_deg"]),
        "B0_deg": float(row["solar_B0_deg"]),
        "image_name": str(row["image_name"]) if "image_name" in df.columns else "",
        "groups_g": int(row["groups_g"]) if "groups_g" in df.columns else None,
        "spots_s": int(row["spots_s"]) if "spots_s" in df.columns else None,
        "R_raw": int(row["R_raw"]) if "R_raw" in df.columns else None,
    }


# ============================================================
# Kép / korong / geometria
# ============================================================

def detect_disk(gray):
    nz = gray[gray > 0]
    if nz.size == 0:
        raise RuntimeError("Üres kép.")
    threshold = max(8, int(np.percentile(nz, 8) * 0.28))
    bw = (gray > threshold).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        raise RuntimeError("Nem található Napkorong.")
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = (labels == idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)

    if len(cnt) >= 5:
        (cx, cy), (a, b), _ = cv2.fitEllipse(cnt)
        r = 0.25 * (a + b)
    else:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return float(cx), float(cy), float(r)


def rotate_image(img, angle_deg, center):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def robust_cleanup(mask, open_k=3, close_k=5):
    img = mask.astype(np.uint8) * 255
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    return img > 0


def component_table(binary, gray, ratio, prefix, min_area):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
    rows = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = map(float, centroids[i])
        mask = labels == i

        pts_yx = np.column_stack(np.where(mask))
        elong = 1.0
        if len(pts_yx) >= 5:
            pts_xy = pts_yx[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
            try:
                ell = cv2.fitEllipse(pts_xy)
                aa, bb = ell[1]
                if min(aa, bb) > 0:
                    elong = float(max(aa, bb) / min(aa, bb))
            except cv2.error:
                pass

        rows.append({
            "id": f"{prefix}{len(rows)+1}",
            "area_px": area,
            "x": cx,
            "y": cy,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "elongation": float(elong),
            "mean_gray": float(gray[mask].mean()),
            "mean_ratio": float(ratio[mask].mean()),
            "max_ratio": float(ratio[mask].max()),
            "_mask": mask,
        })
    return rows


def skeleton_length(mask):
    img = mask.astype(np.uint8) * 255
    skel = np.zeros_like(img)
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, el)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = cv2.erode(img, el)
    return int(cv2.countNonZero(skel))


def heliographic_latitude_deg(x, y, cx, cy, r, b0_deg):
    X = (x - cx) / r
    Y = -(y - cy) / r
    rho = math.hypot(X, Y)

    if rho > 1.0:
        return np.nan
    if rho < 1e-12:
        return float(b0_deg)

    c = math.asin(min(1.0, rho))
    b0 = math.radians(b0_deg)
    sin_phi = math.cos(c) * math.sin(b0) + (Y * math.sin(c) * math.cos(b0) / rho)
    sin_phi = max(-1.0, min(1.0, sin_phi))
    return math.degrees(math.asin(sin_phi))


# ============================================================
# Q osztályozás
# ============================================================

def q_from_score(score):
    if score <= 0:
        return 1
    if score == 1:
        return 2
    if score == 2:
        return 3
    if score == 3:
        return 4
    return 5


def classify_filament(obj):
    score = 0
    if obj["mean_ratio"] < 0.84:
        score += 2
    elif obj["mean_ratio"] < 0.88:
        score += 1

    if obj["elongation"] >= 4.0:
        score += 2
    elif obj["elongation"] >= 2.7:
        score += 1

    if obj["length_px"] >= 65:
        score += 1
    elif obj["length_px"] < FILAMENT_MIN_SKELETON:
        score -= 2

    if obj["area_px"] >= 260:
        score += 1

    return q_from_score(score), score


def classify_thin_filament(obj):
    score = 0

    if obj["mean_ratio"] < 0.88:
        score += 2
    elif obj["mean_ratio"] < 0.915:
        score += 1

    if obj["elongation"] >= 6.0:
        score += 2
    elif obj["elongation"] >= THIN_FILAMENT_MIN_ELONGATION:
        score += 1

    if obj["length_px"] >= 35:
        score += 1
    elif obj["length_px"] < THIN_FILAMENT_MIN_SKELETON:
        score -= 2

    min_side = min(obj["bbox_w"], obj["bbox_h"])
    if min_side <= THIN_FILAMENT_MAX_WIDTH:
        score += 1
    else:
        score -= 1

    return q_from_score(score), score


def classify_plage_inner(obj, lat):
    if np.isnan(lat) or abs(lat) > MAX_AR_LAT_DEG:
        return 1, -10

    alat = abs(lat)
    score = 0

    if alat <= SOFT_AR_LAT_DEG:
        score += 2
    elif alat <= AR_MID_LAT_DEG:
        score += 1
    else:
        if obj["mean_ratio"] < 1.18 and obj["area_px"] < 180:
            score -= 2

    if obj["mean_ratio"] >= 1.18:
        score += 2
    elif obj["mean_ratio"] >= 1.15:
        score += 1

    if obj["area_px"] >= 160:
        score += 1
    elif obj["area_px"] < 80:
        score -= 1

    return q_from_score(score), score


def classify_plage_limb(obj, lat, radial_frac):
    """
    Külön peremdetektor.

    Finomhangolás:
    - a peremzóna belső részén (0.90-0.95 R) kicsit engedékenyebb,
      hogy a valódi, vetületben összenyomott plage-ok visszajöjjenek
    - a legkülső gyűrűben (>0.95 R) viszont szigorúbb a szűrés,
      hogy a limb menti hamis AR-ek (pl. túl magas északi jelölt) eltűnjenek
    """
    if np.isnan(lat):
        return 1, -10

    alat = abs(lat)
    if alat > MAX_AR_LIMB_LAT_DEG:
        return 1, -10

    if radial_frac <= AR_INNER_ZONE_R or radial_frac > PLAGE_OUTER_R:
        return 1, -10

    # Legkülső peremgyűrű: itt sok a hamis jelölt
    if radial_frac > AR_VERY_LIMB_START_R:
        if alat > AR_VERY_LIMB_MAX_LAT_DEG:
            return 1, -10
        if obj["mean_ratio"] < AR_VERY_LIMB_MIN_RATIO:
            return 1, -10
        if obj["area_px"] < AR_VERY_LIMB_MIN_AREA:
            return 1, -10

        score = 0
        score += 2  # ha idáig eljut, eleve erős jelölt
        if obj["mean_ratio"] >= 1.20:
            score += 1
        if obj["area_px"] >= 140:
            score += 1
        return q_from_score(score), score

    # 0.90-0.95R között engedékenyebb peremdetektor
    score = 0

    if obj["mean_ratio"] >= 1.19:
        score += 3
    elif obj["mean_ratio"] >= 1.165:
        score += 2
    elif obj["mean_ratio"] >= AR_LIMB_MIN_RATIO:
        score += 1
    else:
        score -= 2

    if obj["area_px"] >= 120:
        score += 2
    elif obj["area_px"] >= 65:
        score += 1
    elif obj["area_px"] < AR_LIMB_MIN_AREA:
        score -= 1

    # szélesség itt csak puha prior
    if alat <= 45:
        score += 1
    elif alat <= 60:
        score += 0
    else:
        if obj["mean_ratio"] >= 1.175 and obj["area_px"] >= 75:
            score += 0
        else:
            score -= 2

    return q_from_score(score), score


# ============================================================
# Filamentum segédek
# ============================================================

def _mask_orientation_deg(mask):
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    ang = math.degrees(math.atan2(v[1], v[0])) % 180.0
    return float(ang)


def _angle_diff_180(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _min_mask_distance(mask_a, mask_b):
    inv = (~mask_a).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    vals = dist[mask_b]
    if vals.size == 0:
        return 1e9
    return float(vals.min())


def merge_filaments_graph(normal_seed, thin_seed, gray, ratio, r):
    candidates = []

    for obj in normal_seed:
        obj = dict(obj)
        obj["source"] = "normal"
        obj["orientation_deg"] = _mask_orientation_deg(obj["_mask"])
        candidates.append(obj)

    for obj in thin_seed:
        obj = dict(obj)
        obj["source"] = "thin"
        obj["orientation_deg"] = _mask_orientation_deg(obj["_mask"])
        candidates.append(obj)

    n = len(candidates)
    if n == 0:
        return []

    gap = max(5.0, FILAMENT_LINK_GAP_FRAC * r)
    tight = max(3.0, FILAMENT_LINK_TIGHT_FRAC * r)
    adj = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            a = candidates[i]
            b = candidates[j]
            d = _min_mask_distance(a["_mask"], b["_mask"])
            da = _angle_diff_180(a["orientation_deg"], b["orientation_deg"])
            connected = (d <= tight) or (d <= gap and da <= FILAMENT_LINK_MAX_ANGLE_DEG)
            if connected:
                adj[i].append(j)
                adj[j].append(i)

    seen = set()
    groups = []

    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        group = []
        while stack:
            k = stack.pop()
            group.append(k)
            for nb in adj[k]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        groups.append(group)

    merged = []
    for group in groups:
        members = [candidates[i] for i in group]
        has_normal = any(m["source"] == "normal" for m in members)
        thin_count = sum(m["source"] == "thin" for m in members)

        if not has_normal and thin_count < 2:
            continue

        final_mask = np.zeros_like(gray, dtype=bool)
        for m in members:
            final_mask |= m["_mask"]

        area = int(np.count_nonzero(final_mask))
        ys, xs = np.where(final_mask)
        if area == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx = float(xs.mean())
        cy = float(ys.mean())

        elong = 1.0
        pts_xy = np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2)
        if len(pts_xy) >= 5:
            try:
                ell = cv2.fitEllipse(pts_xy)
                aa, bb = ell[1]
                if min(aa, bb) > 0:
                    elong = float(max(aa, bb) / min(aa, bb))
            except cv2.error:
                pass

        length_px = skeleton_length(final_mask)

        obj = {
            "id": "",
            "area_px": area,
            "x": cx,
            "y": cy,
            "bbox_x": int(x0),
            "bbox_y": int(y0),
            "bbox_w": int(x1 - x0 + 1),
            "bbox_h": int(y1 - y0 + 1),
            "elongation": elong,
            "mean_gray": float(gray[final_mask].mean()),
            "mean_ratio": float(ratio[final_mask].mean()),
            "_mask": final_mask,
            "member_seed_count": len(members),
            "member_normal_count": sum(m["source"] == "normal" for m in members),
            "member_thin_count": thin_count,
            "length_px": length_px,
            "orientation_deg": _mask_orientation_deg(final_mask),
        }

        obj["Q"] = max(int(m.get("Q", 1)) for m in members)
        obj["Q_score"] = max(int(m.get("Q_score", 0)) for m in members)

        if not has_normal and (elong < 3.0 or length_px < 28):
            continue

        merged.append(obj)

    for i, obj in enumerate(merged, 1):
        obj["id"] = f"F{i}"

    return merged


# ============================================================
# AR segédek
# ============================================================

def group_plage_into_ars_zone(plage_objs, gray_shape, radius_px, zone="inner", filaments=None):
    if not plage_objs:
        return []

    h, w = gray_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for obj in plage_objs:
        mask[obj["_mask"]] = 255

    frac = AR_GROUP_FRAC_INNER if zone == "inner" else AR_GROUP_FRAC_LIMB
    d = int(max(5, round(frac * radius_px)))
    if d % 2 == 0:
        d += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    grouped = cv2.dilate(mask, kernel, iterations=1)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8))
    groups = []

    for comp in range(1, n):
        comp_mask = labels == comp
        members = []
        for pf in plage_objs:
            x = int(round(pf["x"]))
            y = int(round(pf["y"]))
            if 0 <= y < comp_mask.shape[0] and 0 <= x < comp_mask.shape[1] and comp_mask[y, x]:
                members.append(pf)

        if not members:
            continue

        max_ratio = max(m["mean_ratio"] for m in members)
        max_peak_ratio = max(m.get("max_ratio", m["mean_ratio"]) for m in members)
        max_area = max(m["area_px"] for m in members)
        mean_ratio = float(np.mean([m["mean_ratio"] for m in members]))

        # zóna-specifikus singleton szabály
        if len(members) == 1:
            if zone == "inner":
                valid_single = (
                    max_area >= AR_SINGLE_STRONG_MIN_AREA
                    and max_ratio >= AR_SINGLE_STRONG_MIN_RATIO
                )
            else:
                valid_single = (
                    max_area >= AR_LIMB_SINGLE_MIN_AREA
                    and max_ratio >= AR_LIMB_SINGLE_MIN_RATIO
                )

                m = members[0]
                alat = abs(m.get("lat_deg", np.nan))

                # Alacsony szelessegu, diffuz peremi AR mentesi ag.
                if m.get("limb_rescue", False):
                    valid_single = (
                        max_area >= AR_LIMB_RESCUE_MIN_AREA
                        and max_peak_ratio >= AR_LIMB_RESCUE_MIN_PEAK_RATIO
                        and not np.isnan(alat)
                        and alat <= AR_LIMB_RESCUE_MAX_LAT_DEG
                    )

                # Magasabb szelessegen a peremi szemcsezet csak eros,
                # eleg nagy jeloltkent fogadhato el.
                if (
                    valid_single
                    and not np.isnan(alat)
                    and alat > AR_LIMB_HIGH_LAT_START_DEG
                ):
                    valid_single = (
                        (
                            max_area >= AR_LIMB_HIGH_LAT_MIN_AREA
                            and max_ratio >= AR_LIMB_HIGH_LAT_MIN_RATIO
                        )
                        or (
                            max_area >= AR_LIMB_HIGH_LAT_COMPACT_MIN_AREA
                            and max_ratio >= AR_LIMB_HIGH_LAT_COMPACT_MIN_RATIO
                        )
                    )

            # A filamentum konturjanak vilagos haloibol ne legyen singleton AR.
            if (
                valid_single
                and filaments
                and not members[0].get("limb_rescue", False)
            ):
                min_filament_dist = min(
                    _min_mask_distance(members[0]["_mask"], f["_mask"])
                    for f in filaments
                )
                reject_max_area = (
                    AR_INNER_FILAMENT_REJECT_MAX_AREA
                    if zone == "inner"
                    else AR_LIMB_FILAMENT_REJECT_MAX_AREA
                )
                if (
                    min_filament_dist <= AR_FILAMENT_REJECT_DIST_FRAC * radius_px
                    and max_area < reject_max_area
                ):
                    valid_single = False
            if not valid_single:
                continue

        weights = np.array([max(0.001, m["mean_ratio"] - 1.0) for m in members], dtype=float)
        xs = np.array([m["x"] for m in members], dtype=float)
        ys = np.array([m["y"] for m in members], dtype=float)

        if weights.sum() <= 0:
            cx = float(xs.mean())
            cy = float(ys.mean())
        else:
            cx = float((xs * weights).sum() / weights.sum())
            cy = float((ys * weights).sum() / weights.sum())

        groups.append({
            "zone": zone,
            "id": "",
            "x": cx,
            "y": cy,
            "member_count": len(members),
            "max_member_ratio": float(max_ratio),
            "max_member_peak_ratio": float(max_peak_ratio),
            "max_member_area_px": int(max_area),
            "mean_member_ratio": mean_ratio,
            "member_plage_ids": ",".join(m["id"] for m in members),
            "max_member_radial_frac": float(max(m.get("radial_frac", 0.0) for m in members)),
            "has_limb_rescue": bool(any(m.get("limb_rescue", False) for m in members)),
        })

    return groups


def merge_ar_groups(ar_groups, radius_px):
    """
    Belső és perem AR-csoportok végső összevonása.
    Ha két csoport középpontja nagyon közel van, egy AR marad belőle.
    """
    if not ar_groups:
        return []

    max_dist = max(8.0, AR_FINAL_MERGE_FRAC * radius_px)
    groups = [dict(g) for g in ar_groups]

    n = len(groups)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = groups[i]["x"] - groups[j]["x"]
            dy = groups[i]["y"] - groups[j]["y"]
            d = math.hypot(dx, dy)
            pair_max_dist = max_dist
            if groups[i].get("has_limb_rescue") and groups[j].get("has_limb_rescue"):
                pair_max_dist = max(
                    pair_max_dist,
                    AR_LIMB_RESCUE_FINAL_MERGE_FRAC * radius_px,
                )
            if d <= pair_max_dist:
                adj[i].append(j)
                adj[j].append(i)

    seen = set()
    merged = []
    idx = 1

    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp = []
        while stack:
            k = stack.pop()
            comp.append(groups[k])
            for nb in adj[k]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)

        weights = np.array([max(0.001, g["mean_member_ratio"] - 1.0) for g in comp], dtype=float)
        xs = np.array([g["x"] for g in comp], dtype=float)
        ys = np.array([g["y"] for g in comp], dtype=float)
        if weights.sum() <= 0:
            cx = float(xs.mean())
            cy = float(ys.mean())
        else:
            cx = float((xs * weights).sum() / weights.sum())
            cy = float((ys * weights).sum() / weights.sum())

        merged.append({
            "id": f"AR{idx}",
            "x": cx,
            "y": cy,
            "member_count": int(sum(g["member_count"] for g in comp)),
            "max_member_ratio": float(max(g["max_member_ratio"] for g in comp)),
            "max_member_peak_ratio": float(max(g["max_member_peak_ratio"] for g in comp)),
            "max_member_area_px": int(max(g["max_member_area_px"] for g in comp)),
            "mean_member_ratio": float(np.mean([g["mean_member_ratio"] for g in comp])),
            "member_plage_ids": ",".join(g["member_plage_ids"] for g in comp if g["member_plage_ids"]),
            "source_zones": ",".join(sorted(set(g["zone"] for g in comp))),
            "has_limb_rescue": bool(any(g.get("has_limb_rescue", False) for g in comp)),
        })
        idx += 1

    return merged


# ============================================================
# Elemzés
# ============================================================

def analyze_halpha(img, b0_deg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx, cy, r = detect_disk(gray)

    yy, xx = np.indices(gray.shape)
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    rfrac = rr / r

    filament_zone = rfrac <= FILAMENT_INNER_R
    plage_zone = rfrac <= PLAGE_OUTER_R
    inner_zone = rfrac <= AR_INNER_ZONE_R
    limb_zone = (rfrac > AR_INNER_ZONE_R) & (rfrac <= PLAGE_OUTER_R)

    sigma = max(7.0, LOCAL_BLUR_FRAC * r)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    ratio = gray.astype(np.float32) / np.maximum(blur.astype(np.float32), 1.0)

    # ----- Filamentumok -----
    fil_mask = robust_cleanup((ratio < FILAMENT_DARK_RATIO) & filament_zone, open_k=3, close_k=5)
    fil_raw = component_table(fil_mask, gray, ratio, "NF", FILAMENT_MIN_AREA)

    normal_all = []
    for obj in fil_raw:
        if obj["elongation"] < FILAMENT_MIN_ELONGATION and obj["area_px"] < FILAMENT_BIG_AREA:
            continue
        obj["length_px"] = skeleton_length(obj["_mask"])
        if obj["length_px"] < FILAMENT_MIN_SKELETON and obj["area_px"] < FILAMENT_BIG_AREA:
            continue
        obj["lat_deg"] = heliographic_latitude_deg(obj["x"], obj["y"], cx, cy, r, b0_deg)
        obj["Q"], obj["Q_score"] = classify_filament(obj)
        normal_all.append(obj)

    normal_seed = [
        x for x in normal_all
        if x["Q"] >= NORMAL_SEED_MIN_Q or x["area_px"] >= 350 or x["length_px"] >= 75
    ]

    thin_mask = robust_cleanup((ratio < THIN_FILAMENT_DARK_RATIO) & filament_zone, open_k=1, close_k=3)
    thin_raw = component_table(thin_mask, gray, ratio, "TF", THIN_FILAMENT_MIN_AREA)

    thin_all = []
    for obj in thin_raw:
        if obj["elongation"] < THIN_FILAMENT_MIN_ELONGATION:
            continue
        obj["length_px"] = skeleton_length(obj["_mask"])
        if obj["length_px"] < THIN_FILAMENT_MIN_SKELETON:
            continue
        if min(obj["bbox_w"], obj["bbox_h"]) > THIN_FILAMENT_MAX_WIDTH * 1.8:
            continue
        obj["lat_deg"] = heliographic_latitude_deg(obj["x"], obj["y"], cx, cy, r, b0_deg)
        obj["Q"], obj["Q_score"] = classify_thin_filament(obj)
        thin_all.append(obj)

    thin_seed = [x for x in thin_all if x["Q"] >= 3]

    filaments = merge_filaments_graph(normal_seed, thin_seed, gray, ratio, r)
    for obj in filaments:
        obj["lat_deg"] = heliographic_latitude_deg(obj["x"], obj["y"], cx, cy, r, b0_deg)

    # ----- Plage / AR: közös nyers keresés 0.99R-ig -----
    plage_mask = robust_cleanup((ratio > PRIMARY_PLAGE_RATIO) & plage_zone, open_k=3, close_k=7)
    plage_raw = component_table(plage_mask, gray, ratio, "PF", PLAGE_MIN_AREA)

    plage_all = []
    inner_plage = []
    limb_plage = []

    for obj in plage_raw:
        obj["lat_deg"] = heliographic_latitude_deg(obj["x"], obj["y"], cx, cy, r, b0_deg)
        obj["radial_frac"] = math.hypot((obj["x"] - cx) / r, (obj["y"] - cy) / r)

        if obj["radial_frac"] <= AR_INNER_ZONE_R:
            obj["Q"], obj["Q_score"] = classify_plage_inner(obj, obj["lat_deg"])
            obj["zone"] = "inner"
            if obj["Q"] >= 3:
                inner_plage.append(obj)
        elif obj["radial_frac"] <= PLAGE_OUTER_R:
            obj["Q"], obj["Q_score"] = classify_plage_limb(obj, obj["lat_deg"], obj["radial_frac"])
            obj["zone"] = "limb"
            if obj["Q"] >= 3:
                limb_plage.append(obj)
        else:
            obj["Q"], obj["Q_score"] = 1, -10
            obj["zone"] = "outer"

        plage_all.append(obj)

    # ----- Perem-AR mentesi ag -----
    # A nyers kep enyhe simitasa megszunteti az apro szemcseket, mikozben a
    # gyengebb, vetuletben osszenyomott vilagos teruletek eros magja megmarad.
    rescue_gray = cv2.GaussianBlur(
        gray, (0, 0), AR_LIMB_RESCUE_SMOOTH_SIGMA
    )
    rescue_ratio = rescue_gray.astype(np.float32) / np.maximum(
        blur.astype(np.float32), 1.0
    )
    rescue_zone = (
        (rfrac > AR_LIMB_RESCUE_INNER_R)
        & (rfrac <= PLAGE_OUTER_R)
    )
    rescue_mask = robust_cleanup(
        (rescue_ratio > AR_LIMB_RESCUE_RATIO) & rescue_zone,
        open_k=3,
        close_k=5,
    )
    rescue_raw = component_table(
        rescue_mask,
        gray,
        rescue_ratio,
        "RF",
        AR_LIMB_RESCUE_MIN_AREA,
    )

    # A mentesi jelolt csak egy mar biztosan elfogadott AR peremi
    # folytatasakent johet vissza. Ez megakadalyozza, hogy elszigetelt
    # peremszerkezeti zajokbol uj AR keletkezzen.
    preliminary_groups = (
        group_plage_into_ars_zone(
            inner_plage, gray.shape, r, zone="inner", filaments=filaments
        )
        + group_plage_into_ars_zone(
            limb_plage, gray.shape, r, zone="limb", filaments=filaments
        )
    )
    for group in preliminary_groups:
        group["lat_deg"] = heliographic_latitude_deg(
            group["x"], group["y"], cx, cy, r, b0_deg
        )
        group["radial_frac"] = math.hypot(
            (group["x"] - cx) / r,
            (group["y"] - cy) / r,
        )

    existing_plage = inner_plage + limb_plage

    for obj in rescue_raw:
        obj["lat_deg"] = heliographic_latitude_deg(
            obj["x"], obj["y"], cx, cy, r, b0_deg
        )
        obj["radial_frac"] = math.hypot(
            (obj["x"] - cx) / r,
            (obj["y"] - cy) / r,
        )
        alat = abs(obj["lat_deg"])

        if (
            np.isnan(alat)
            or alat > AR_LIMB_RESCUE_MAX_LAT_DEG
            or obj["radial_frac"] <= AR_LIMB_RESCUE_INNER_R
            or obj["radial_frac"] > PLAGE_OUTER_R
            or obj["max_ratio"] < AR_LIMB_RESCUE_MIN_PEAK_RATIO
        ):
            continue

        outward_companions = [
            g for g in preliminary_groups
            if (
                math.hypot(obj["x"] - g["x"], obj["y"] - g["y"])
                <= AR_LIMB_RESCUE_COMPANION_DIST_FRAC * r
                and not np.isnan(g["lat_deg"])
                and abs(obj["lat_deg"] - g["lat_deg"])
                <= AR_LIMB_RESCUE_COMPANION_LAT_DELTA_DEG
                and obj["radial_frac"] - g["radial_frac"]
                >= AR_LIMB_RESCUE_MIN_OUTWARD_DELTA_R
            )
        ]
        if not outward_companions:
            continue

        # Ha mar van AR gyakorlatilag ugyanezen a helyen, nincs uj cimke.
        # A gyenge, AR-re nem kepes plage-jeloltek viszont nem blokkolnak.
        if preliminary_groups:
            nearest_ar_center = min(
                math.hypot(obj["x"] - g["x"], obj["y"] - g["y"])
                for g in preliminary_groups
            )
            if nearest_ar_center <= AR_LIMB_RESCUE_EXISTING_AR_DIST_FRAC * r:
                continue

        obj["Q"], obj["Q_score"] = 3, 2
        obj["zone"] = "limb"
        obj["limb_rescue"] = True
        limb_plage.append(obj)
        plage_all.append(obj)
        existing_plage.append(obj)

    plage = inner_plage + limb_plage
    for i, obj in enumerate(plage, 1):
        obj["id"] = f"PF{i}"

    # ----- AR-csoportok külön zónákban -----
    ar_inner = group_plage_into_ars_zone(
        inner_plage, gray.shape, r, zone="inner", filaments=filaments
    )
    ar_limb = group_plage_into_ars_zone(
        limb_plage, gray.shape, r, zone="limb", filaments=filaments
    )
    ars = merge_ar_groups(ar_inner + ar_limb, r)

    return {
        "center_x": cx,
        "center_y": cy,
        "radius_px": r,
        "normal_filaments_all": normal_all,
        "thin_filaments_all": thin_all,
        "normal_seed_count": len(normal_seed),
        "thin_seed_count": len(thin_seed),
        "filaments": filaments,
        "plage_all": plage_all,
        "plage": plage,
        "inner_plage": inner_plage,
        "limb_plage": limb_plage,
        "ar_inner_premerge": ar_inner,
        "ar_limb_premerge": ar_limb,
        "ars": ars,
    }


# ============================================================
# Annotálás
# ============================================================

def draw_cross(img, x, y, color, size=6, thickness=1):
    x = int(round(x))
    y = int(round(y))
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def annotate(img, analysis, geom, rotation_deg):
    out = img.copy()
    cx = analysis["center_x"]
    cy = analysis["center_y"]
    r = analysis["radius_px"]

    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), (190, 190, 190), 1)

    for obj in analysis["filaments"]:
        mask = obj["_mask"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
        cv2.putText(out, obj["id"], (int(obj["x"]) + 4, int(obj["y"]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

    for ar in analysis["ars"]:
        draw_cross(out, ar["x"], ar["y"], (255, 255, 0), size=6, thickness=1)
        cv2.putText(out, ar["id"], (int(ar["x"]) + 7, int(ar["y"]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 1, cv2.LINE_AA)

    # Egtajak a Napkorongon kivul.
    cv2.putText(out, "N", (int(cx) - 7, max(24, int(cy - r) - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "W", (min(out.shape[1] - 28, int(cx + r) + 10), int(cy) + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    lines = [
        f"Algorithm: v{ALGORITHM_VERSION}",
        f"Filaments: {len(analysis['filaments'])}",
        f"Active regions: {len(analysis['ars'])}",
    ]

    x0, y0 = 10, 10
    line_h = 22
    box_w = 315
    box_h = 12 + line_h * len(lines)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        cv2.putText(out, line, (x0 + 10, y0 + 21 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# ============================================================
# CSV
# ============================================================

def build_tables(analysis, geom, rotation_deg, ha_file):
    rows = []

    for obj in analysis["normal_filaments_all"]:
        row = {k: v for k, v in obj.items() if k != "_mask"}
        row["type"] = "normal_filament_candidate"
        row["accepted_as_seed"] = bool(
            obj["Q"] >= NORMAL_SEED_MIN_Q
            or obj["area_px"] >= 350
            or obj["length_px"] >= 75
        )
        rows.append(row)

    for obj in analysis["thin_filaments_all"]:
        row = {k: v for k, v in obj.items() if k != "_mask"}
        row["type"] = "thin_filament_candidate"
        row["accepted_as_seed"] = bool(obj["Q"] >= 3)
        rows.append(row)

    for obj in analysis["filaments"]:
        row = {k: v for k, v in obj.items() if k != "_mask"}
        row["type"] = "filament_final"
        row["accepted_as_seed"] = True
        rows.append(row)

    for obj in analysis["ars"]:
        row = dict(obj)
        row["type"] = "active_region_label"
        row["accepted_as_seed"] = True
        rows.append(row)

    features = pd.DataFrame(rows)

    summary = pd.DataFrame([{
        "algorithm_version": ALGORITHM_VERSION,
        "input_halpha_file": ha_file.name,
        "daily_summary_file": geom["summary_path"].name,
        "seestar_reference_image": geom["image_name"],
        "reference_utc": geom["reference_utc"].isoformat(),
        "reference_time_delta_seconds": geom["delta_seconds"],
        "solar_P_deg": geom["P_deg"],
        "solar_B0_deg": geom["B0_deg"],
        "acuter_camera_angle_deg": ACUTER_CAMERA_ANGLE_DEG,
        "applied_rotation_deg_ccw": rotation_deg,
        "normal_filament_candidate_count": len(analysis["normal_filaments_all"]),
        "normal_filament_seed_count": analysis["normal_seed_count"],
        "thin_filament_candidate_count": len(analysis["thin_filaments_all"]),
        "thin_filament_seed_count": analysis["thin_seed_count"],
        "filament_count_final": len(analysis["filaments"]),
        "active_region_count_final": len(analysis["ars"]),
        "active_region_count_inner_premerge": len(analysis["ar_inner_premerge"]),
        "active_region_count_limb_premerge": len(analysis["ar_limb_premerge"]),
        "sesuco_groups_g": geom["groups_g"],
        "sesuco_spots_s": geom["spots_s"],
        "sesuco_R_raw": geom["R_raw"],
    }])

    return features, summary


# ============================================================
# MAIN
# ============================================================

def main():
    folder = WORK_DIR
    if not folder.exists():
        raise FileNotFoundError(f"Nem létezik a munkakönyvtár: {folder}")

    print("=" * 84)
    print(f"H-alfa filamentumok + sugariranyu peremi AR detektor v{ALGORITHM_VERSION}")
    print("=" * 84)
    print(f"Munkakönyvtár: {folder}")

    ha_file = choose_latest_halpha_image(folder)
    print(f"H-alfa kép: {ha_file.name}")

    geom = read_solar_geometry(folder, ha_file)
    print(f"daily_summary: {geom['summary_path'].name}")
    print(f"Seestar referencia: {geom['image_name']}")
    print(f"P = {geom['P_deg']:+.3f} deg")
    print(f"B0 = {geom['B0_deg']:+.3f} deg")
    print(f"Időkülönbség = {geom['delta_seconds']/60.0:.1f} perc")

    rotation_deg = ACUTER_CAMERA_ANGLE_DEG - geom["P_deg"]
    print(f"Acuter kamera offset = {ACUTER_CAMERA_ANGLE_DEG:+.3f} deg")
    print(f"Alkalmazott forgatás = {rotation_deg:+.3f} deg")

    img = cv2.imread(str(ha_file), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Nem olvasható: {ha_file}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx, cy, _ = detect_disk(gray)
    rotated = rotate_image(img, rotation_deg, (cx, cy))

    analysis = analyze_halpha(rotated, geom["B0_deg"])
    annotated = annotate(rotated, analysis, geom, rotation_deg)
    features, summary = build_tables(analysis, geom, rotation_deg, ha_file)

    out_img = folder / f"{ha_file.stem}{ANNOTATED_SUFFIX}"
    out_features = folder / FEATURES_FILENAME
    out_summary = folder / SUMMARY_FILENAME

    cv2.imwrite(str(out_img), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    features.to_csv(out_features, index=False, encoding="utf-8-sig")
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print()
    print(f"Filamentumok: {len(analysis['filaments'])}")
    print(f"Aktív területek: {len(analysis['ars'])}")
    print()
    print(f"Annotált kép: {out_img.name}")
    print(f"Részletes CSV: {out_features.name}")
    print(f"Összesítő CSV: {out_summary.name}")


if __name__ == "__main__":
    main()
