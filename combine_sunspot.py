import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table
from matplotlib.ticker import AutoMinorLocator

# --- Beállítások ---
SOLAR_DIR = r"D:\Csillagászat\Seestar\Solar"

HONAPOK = {
    1:  ("Január", "januar"),
    2:  ("Február", "februar"),
    3:  ("Március", "marcius"),
    4:  ("Április", "aprilis"),
    5:  ("Május", "majus"),
    6:  ("Június", "junius"),
    7:  ("Július", "julius"),
    8:  ("Augusztus", "augusztus"),
    9:  ("Szeptember", "szeptember"),
    10: ("Október", "oktober"),
    11: ("November", "november"),
    12: ("December", "december"),
}

HONAP_NEVEK = {
    "január": 1, "januar": 1,
    "február": 2, "februar": 2,
    "március": 3, "marcius": 3,
    "április": 4, "aprilis": 4,
    "május": 5, "majus": 5,
    "június": 6, "junius": 6,
    "július": 7, "julius": 7,
    "augusztus": 8,
    "szeptember": 9,
    "október": 10, "oktober": 10,
    "november": 11,
    "december": 12
}


def honap_bekerese():
    user_input = input("Add meg a hónapot (pl. 9 vagy szeptember): ").strip().lower()

    if user_input.isdigit():
        honap_szam = int(user_input)
        if honap_szam in HONAPOK:
            return honap_szam
    elif user_input in HONAP_NEVEK:
        return HONAP_NEVEK[user_input]

    raise ValueError("Érvénytelen hónap. Adj meg 1-12 közötti számot vagy magyar hónapnevet.")


def find_daily_summary_files(base_dir):
    matches = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            lower = fname.lower()
            if lower.startswith("daily_summary") and lower.endswith(".csv"):
                matches.append(os.path.join(root, fname))
    return matches


def first_existing_value(row, candidates):
    for col in candidates:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                return value
    return None


def infer_date_from_path(path):
    # yyyy-mm-dd keresése az elérési útban
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    if m:
        return pd.to_datetime(m.group(1), errors="coerce")

    # yyyy_mm_dd vagy yyyy.mm.dd
    m = re.search(r"(\d{4})[-_.](\d{2})[-_.](\d{2})", path)
    if m:
        s = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return pd.to_datetime(s, errors="coerce")

    return pd.NaT


def read_summary_file(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        print(f"Hiba beolvasáskor: {path}")
        print(e)
        return None

    if df.empty:
        return None

    # oszlopnevek normalizálása
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    row = df.iloc[0]

    # Dátum/UTC keresése
    date_candidates = []
    for candidate in ["date", "utc", "datetime", "obs_date", "observation_date"]:
        if candidate in lower_map:
            date_candidates.append(lower_map[candidate])

    date_value = first_existing_value(row, date_candidates)

    if date_value is not None:
        date_parsed = pd.to_datetime(date_value, errors="coerce")
    else:
        date_parsed = infer_date_from_path(path)

    if pd.isna(date_parsed):
        date_parsed = infer_date_from_path(path)

    # g, s, R_raw keresése
    g_candidates = []
    for candidate in ["g", "groups", "group_count"]:
        if candidate in lower_map:
            g_candidates.append(lower_map[candidate])

    s_candidates = []
    for candidate in ["s", "spots", "spot_count"]:
        if candidate in lower_map:
            s_candidates.append(lower_map[candidate])

    r_candidates = []
    for candidate in ["r_raw", "r", "relative_sunspot_number"]:
        if candidate in lower_map:
            r_candidates.append(lower_map[candidate])

    g_val = first_existing_value(row, g_candidates)
    s_val = first_existing_value(row, s_candidates)
    r_val = first_existing_value(row, r_candidates)

    result = {
        "date": date_parsed,
        "g": pd.to_numeric(g_val, errors="coerce"),
        "s": pd.to_numeric(s_val, errors="coerce"),
        "R_raw": pd.to_numeric(r_val, errors="coerce"),
        "source_file": path
    }

    return result


def create_monthly_plot(df, honap_cim, output_file):
    # LaTeX-szerű megjelenés
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    df = df.copy()
    df["day"] = pd.to_datetime(df["date"]).dt.day
    df = df.sort_values("date")

    fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(
        df["day"],
        df["R_raw"],
        marker="o",
        linewidth=1.5,
        markersize=5
    )

    ax.set_xlabel("Nap")
    ax.set_ylabel(r"Relatív napfoltszám ($R$)")
    ax.set_title(f"{honap_cim} havi relatív napfoltszám")
      
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.0
    )
    
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(
        axis="y",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8
    )

    ax.set_xlim(1, max(df["day"]))
    ax.set_xticks(range(1, max(df["day"]) + 1))

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        honap_szam = honap_bekerese()
    except ValueError as e:
        print(e)
        return

    honap_cim, honap_fajlnev = HONAPOK[honap_szam]

    print(f"\nKeresés a következő hónapra: {honap_cim}")

    files = find_daily_summary_files(SOLAR_DIR)

    if not files:
        print("Nem találtam daily_summary CSV fájlokat.")
        return

    records = []
    for path in files:
        rec = read_summary_file(path)
        if rec is None:
            continue
        if pd.isna(rec["date"]):
            continue
        if rec["date"].month == honap_szam:
            records.append(rec)

    if not records:
        print(f"Nincs találat erre a hónapra: {honap_cim}")
        return

    monthly_df = pd.DataFrame(records)

    # csak a lényeges oszlopok
    monthly_df = monthly_df[["date", "g", "s", "R_raw", "source_file"]]

    # rendezés dátum szerint
    monthly_df = monthly_df.sort_values("date")

    # duplikált dátumok esetén az utolsót tartjuk meg
    monthly_df = monthly_df.drop_duplicates(subset=["date"], keep="last")

    # nap oszlop a .vo fájlhoz is
    monthly_df["day"] = monthly_df["date"].dt.day

    # dátum stringgé alakítás a VO fájlhoz
    monthly_df["date"] = monthly_df["date"].dt.strftime("%Y-%m-%d")

    # oszloprend
    monthly_df = monthly_df[["date", "day", "g", "s", "R_raw", "source_file"]]

    # --- VO fájl mentése ---
    vo_path = os.path.join(SOLAR_DIR, f"{honap_fajlnev}.vo")
    table = Table.from_pandas(monthly_df)
    table.write(vo_path, format="votable", overwrite=True)

    # --- Grafikon mentése ---
    plot_path = os.path.join(SOLAR_DIR, f"{honap_fajlnev}_napfoltszam.png")
    create_monthly_plot(monthly_df, honap_cim, plot_path)

    print("\nKész!")
    print(f"VO fájl: {vo_path}")
    print(f"Grafikon: {plot_path}")
    print(f"Talált napok száma: {len(monthly_df)}")


if __name__ == "__main__":
    main()