#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seestar Double Star Analyzer Basic GUI v1.0

Grafikus kezelőfelület a rögzített Basic v1.1a mérési motorhoz.
A mérési algoritmust ez a fájl nem módosítja.
"""

from __future__ import annotations

import math
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

import photoanalyzer_basic as engine


APP_TITLE = "Seestar Double Star Analyzer Basic"
GUI_VERSION = "1.0.2"


def external_app_dir() -> Path:
    """
    Forrásból futva: a .py fájl mappája.
    PyInstaller one-file EXE-ből futva: az EXE mappája.

    A Seestar_wds.csv szándékosan külső fájl marad az EXE mellett,
    így később külön is frissíthető.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


class MatchDialog(tk.Toplevel):
    def __init__(self, parent, matches):
        super().__init__(parent)
        self.title("Komponenspár kiválasztása")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result = None
        self.matches = matches
        self.var = tk.IntVar(value=0)

        outer = ttk.Frame(self, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            outer,
            text="Több megfelelő komponenspár található.",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        ttk.Label(
            outer,
            text="Válaszd ki, melyiket szeretnéd megmérni:",
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        for i, row in enumerate(matches):
            comp = row.comp or "-"
            label = (
                f"{row.disc} {comp}    "
                f"WDS {row.wds_id}    "
                f"PA {row.pa_last:.0f}°    "
                f"Sep {row.sep_last:.1f}\"    "
                f"{row.mag_a:.2f}/{row.mag_b:.2f} mag"
            )
            ttk.Radiobutton(
                outer,
                text=label,
                variable=self.var,
                value=i,
            ).grid(row=2 + i, column=0, sticky="w", pady=3)

        buttons = ttk.Frame(outer)
        buttons.grid(row=3 + len(matches), column=0, sticky="e", pady=(14, 0))

        ttk.Button(buttons, text="Mégse", command=self.cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Kiválasztás", command=self.accept).grid(row=0, column=1)

        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Return>", lambda _e: self.accept())
        self.bind("<Escape>", lambda _e: self.cancel())

        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def accept(self):
        self.result = self.matches[self.var.get()]
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


class AnalyzerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} — GUI v{GUI_VERSION}")
        self.geometry("760x585")
        self.minsize(720, 520)

        self.catalog = None
        self.fit_path = tk.StringVar()
        self.object_name = tk.StringVar()
        self.status_text = tk.StringVar(value="Katalógus betöltése...")
        self.current_row = None

        self._build_ui()
        self.after(100, self._load_catalog_thread)

    def _build_ui(self):
        try:
            style = ttk.Style(self)
            if "vista" in style.theme_names():
                style.theme_use("vista")
        except Exception:
            pass

        main = ttk.Frame(self, padding=22)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)

        title = ttk.Label(
            main,
            text=APP_TITLE,
            font=("Segoe UI", 18, "bold"),
        )
        title.grid(row=0, column=0, columnspan=3, sticky="w")

        subtitle = ttk.Label(
            main,
            text="Egyszerű kettőscsillag-mérés Seestar stacked FIT képből",
            font=("Segoe UI", 10),
        )
        subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 22))

        ttk.Label(main, text="Seestar stacked FIT:").grid(row=2, column=0, sticky="w", padx=(0, 12))
        self.fit_entry = ttk.Entry(main, textvariable=self.fit_path)
        self.fit_entry.grid(row=2, column=1, sticky="ew")
        ttk.Button(main, text="Tallózás...", command=self.browse_fit).grid(row=2, column=2, padx=(10, 0))

        ttk.Label(main, text="Kettős azonosító:").grid(row=3, column=0, sticky="w", padx=(0, 12), pady=(14, 0))
        self.object_entry = ttk.Entry(main, textvariable=self.object_name, width=25)
        self.object_entry.grid(row=3, column=1, sticky="w", pady=(14, 0))
        self.object_entry.bind("<Return>", lambda _e: self.start_measurement())

        self.measure_button = ttk.Button(main, text="MÉRÉS", command=self.start_measurement)
        self.measure_button.grid(row=4, column=0, columnspan=3, pady=(22, 18), ipadx=35, ipady=7)

        ttk.Separator(main).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 18))

        result_frame = ttk.LabelFrame(main, text=" Eredmény ", padding=16)
        result_frame.grid(row=6, column=0, columnspan=3, sticky="nsew")
        main.rowconfigure(6, weight=1)
        result_frame.columnconfigure(1, weight=1)

        self.result_vars = {}
        fields = [
            ("Object", "Objektum"),
            ("WDS", "WDS"),
            ("PA", "PA"),
            ("Separation", "Szeparáció"),
            ("Delta mag", "Δmag"),
            ("Quality", "Minőség"),
        ]
        for r, (key, label) in enumerate(fields):
            ttk.Label(result_frame, text=f"{label}:").grid(row=r, column=0, sticky="w", padx=(0, 18), pady=5)
            var = tk.StringVar(value="—")
            self.result_vars[key] = var
            font = ("Segoe UI", 11, "bold") if key in ("PA", "Separation", "Delta mag") else ("Segoe UI", 10)
            ttk.Label(result_frame, textvariable=var, font=font).grid(row=r, column=1, sticky="w", pady=5)

        self.saved_var = tk.StringVar(value="")
        ttk.Label(
            result_frame,
            textvariable=self.saved_var,
            font=("Segoe UI", 9),
            wraplength=620,
        ).grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(12, 0))

        bottom = ttk.Frame(main)
        bottom.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(18, 0))
        bottom.columnconfigure(1, weight=1)

        self.new_button = ttk.Button(bottom, text="Új mérés", command=self.new_measurement)
        self.new_button.grid(row=0, column=0)
        ttk.Label(bottom, textvariable=self.status_text).grid(row=0, column=1, padx=18, sticky="w")
        ttk.Button(bottom, text="Kilépés", command=self.destroy).grid(row=0, column=2)

    def _load_catalog_thread(self):
        self.measure_button.configure(state="disabled")

        def worker():
            try:
                catalog_path = external_app_dir() / "Seestar_wds.csv"
                catalog = engine.load_basic_catalog(catalog_path)
                self.after(0, lambda: self._catalog_loaded(catalog))
            except Exception as exc:
                self.after(0, lambda msg=str(exc): self._catalog_failed(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _catalog_loaded(self, catalog):
        self.catalog = catalog
        self.status_text.set(f"Katalógus betöltve: {len(catalog)} pár")
        self.measure_button.configure(state="normal")

    def _catalog_failed(self, msg):
        self.status_text.set("Katalógushiba")
        messagebox.showerror("Hiba", f"A katalógus nem tölthető be:\n\n{msg}")

    def browse_fit(self):
        filename = filedialog.askopenfilename(
            title="Seestar stacked FIT kiválasztása",
            filetypes=[
                ("FITS fájlok", "*.fit *.fits *.fts"),
                ("Minden fájl", "*.*"),
            ],
        )
        if filename:
            self.fit_path.set(filename)

    def clear_result(self):
        for var in self.result_vars.values():
            var.set("—")
        self.saved_var.set("")

    def new_measurement(self):
        # A FIT útvonalát szándékosan megtartjuk, mert ugyanarról a képről
        # gyakran több kettőst is egymás után mérünk.
        self.object_name.set("")
        self.current_row = None
        self.clear_result()
        self.status_text.set("Új mérésre kész.")
        self.object_entry.focus_set()

    def start_measurement(self):
        if self.catalog is None:
            messagebox.showinfo("Kis türelmet", "A katalógus még betöltés alatt van.")
            return

        raw_path = self.fit_path.get().strip().strip('"')
        query = self.object_name.get().strip()

        if not raw_path:
            messagebox.showwarning("Hiányzó FIT", "Válassz ki egy Seestar stacked FIT fájlt.")
            return
        fit_path = Path(raw_path)
        if not fit_path.is_file() or fit_path.suffix.lower() not in engine.FITS_EXTENSIONS:
            messagebox.showerror("Hibás FIT", "A megadott FIT/FITS fájl nem található vagy nem használható.")
            return
        if not query:
            messagebox.showwarning("Hiányzó azonosító", "Írd be a kettős azonosítóját, például: HJ 1362")
            return

        matches = engine.find_matches(self.catalog, query)
        if not matches:
            messagebox.showinfo(
                "Nincs találat",
                "Nincs ilyen kettős a Seestar katalógusban.",
            )
            return

        if len(matches) > 1:
            dialog = MatchDialog(self, matches)
            self.wait_window(dialog)
            row = dialog.result
            if row is None:
                return
        else:
            row = matches[0]

        self.current_row = row
        self.clear_result()
        self.measure_button.configure(state="disabled")
        self.new_button.configure(state="disabled")
        self.status_text.set("Mérés folyamatban...")
        self.update_idletasks()

        threading.Thread(
            target=self._measurement_worker,
            args=(fit_path, row),
            daemon=True,
        ).start()

    def _measurement_worker(self, fit_path, row):
        try:
            result = self._measure(fit_path, row)
            self.after(0, lambda: self._measurement_success(result))
        except Exception as exc:
            self.after(0, lambda msg=str(exc): self._measurement_failed(msg))

    def _measure(self, fit_path: Path, row):
        header, data, wcs = engine.read_header_and_data(fit_path)
        fwhm_guess, sources = engine.estimate_fwhm_and_sources(data)

        if row.sep_last > engine.WIDE_PAIR_THRESHOLD_ARCSEC:
            (
                pa, rho, dmag_psf,
                x_a, y_a, x_b, y_b,
                fit_rms, fitted_fwhm,
                ra_a, dec_a, ra_b, dec_b,
            ) = engine.measure_wide_pair(
                data=data,
                wcs=wcs,
                row=row,
                sources=sources,
                fwhm_guess=fwhm_guess,
            )
        else:
            (
                pa, rho, dmag_psf,
                x_a, y_a, x_b, y_b,
                fit_rms, fitted_fwhm,
                ra_a, dec_a, ra_b, dec_b,
            ) = engine.fit_pair_on_frame(
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

        dmag, flux_a, flux_b, r_ap = engine.aperture_delta_mag(
            data=data,
            x_a=x_a, y_a=y_a,
            x_b=x_b, y_b=y_b,
            fitted_fwhm=fitted_fwhm,
        )

        wds_dmag = abs(float(row.mag_b) - float(row.mag_a))
        dmag_diff = abs(float(dmag) - wds_dmag)
        if dmag_diff > engine.MAX_WDS_DMAG_DIFFERENCE:
            raise ValueError(
                "A detektált társ fényessége nem egyezik eléggé a WDS-adattal."
            )

        obs_time = engine.observation_time_text(header)
        out = engine.write_result_txt(
            fit_path, row, obs_time, pa, rho, dmag
        )

        png = None
        image_error = None
        try:
            from duostar_annotation import create_annotated_image
            png = create_annotated_image(
                data, wcs, row, x_a, y_a, x_b, y_b,
                out.with_name(out.stem.replace("_measurement", "_annotated") + ".png"),
            )
        except Exception as exc:
            image_error = str(exc)

        return {
            "object": row.pair_name,
            "wds": row.wds_id,
            "pa": pa,
            "rho": rho,
            "dmag": abs(float(dmag)),
            "quality": "GOOD",
            "saved": out,
            "png": png,
            "image_error": image_error,
        }

    def _measurement_success(self, result):
        self.result_vars["Object"].set(result["object"])
        self.result_vars["WDS"].set(result["wds"])
        self.result_vars["PA"].set(f'{result["pa"]:.2f}°')
        self.result_vars["Separation"].set(f'{result["rho"]:.2f}"')
        self.result_vars["Delta mag"].set(f'{result["dmag"]:.2f} mag')
        self.result_vars["Quality"].set(result["quality"])
        saved_text = f'TXT mentve: {result["saved"]}'
        if result["png"]:
            saved_text += f'\nPNG mentve: {result["png"]}'
        else:
            saved_text += f'\nPNG nem készült: {result["image_error"]}'
        self.saved_var.set(saved_text)
        self.status_text.set("Mérés kész.")
        self.measure_button.configure(state="normal")
        self.new_button.configure(state="normal")

    def _measurement_failed(self, msg):
        self.status_text.set("A mérés nem végezhető el megbízhatóan.")
        self.measure_button.configure(state="normal")
        self.new_button.configure(state="normal")

        # Kezdőbarát üzenetek: a technikai részleteket nem tesszük az arcába.
        friendly = msg
        if "Széles pár:" in msg or "nem detektálható helyi fényességcsúcs" in msg:
            friendly = msg
        elif "nem különül el" in msg:
            friendly = "A két komponens ezen a képen nem különül el megbízhatóan."
        elif "nincs önállóan detektálható" in msg:
            friendly = "Az egyik komponens ezen a képen nem detektálható önálló csillagként."
        elif "fényessége nem egyezik" in msg:
            friendly = "A megtalált csillag fényessége túlzottan eltér a WDS alapján várttól."
        elif "szeparáció" in msg and "eltér" in msg:
            friendly = "A megtalált komponenspár helyzete túlzottan eltér a WDS alapján várttól."
        elif "konvergált" in msg or "konvergált" in msg.lower():
            friendly = "A csillagprofil illesztése ezen a képen nem volt megbízható."

        messagebox.showwarning(
            "Mérés elutasítva",
            friendly,
        )


if __name__ == "__main__":
    app = AnalyzerGUI()
    app.mainloop()
