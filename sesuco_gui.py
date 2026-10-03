#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SeSuCo GUI

English Tkinter front-end for:
  - Seestar Sunspot Counter (sunspot*.py)
  - H-alpha Analyzer (halpha*.py)
  - Monthly Sunspot Summary (combine_sunspot.py)

Keep this file in the same directory as the three processing scripts.
The scientific processing remains in those scripts; this file only provides
an easy Windows-style interface around them.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import os
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_TITLE = "SeSuCo – Solar Observation Tools"
COPYRIGHT_TEXT = "© Zsolt Szamosvari — szamos.photo@gmail.com"

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def resource_dirs() -> list[Path]:
    """Directories that may contain bundled SeSuCo resources/modules."""
    dirs: list[Path] = []
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent)
        if hasattr(sys, "_MEIPASS"):
            dirs.append(Path(sys._MEIPASS))  # type: ignore[attr-defined]
    else:
        dirs.append(Path(__file__).resolve().parent)

    # Preserve order while removing duplicates.
    unique: list[Path] = []
    for d in dirs:
        if d not in unique:
            unique.append(d)
    return unique


def app_dir() -> Path:
    return resource_dirs()[0]


def resource_path(name: str) -> Path | None:
    for base in resource_dirs():
        p = base / name
        if p.exists():
            return p
    return None


def find_module_file(kind: str) -> Path:
    """Find one of the processing scripts beside the GUI or inside PyInstaller."""

    if kind == "sunspot":
        exact_names = ["sunspot.py"]
        patterns = ["sunspot*.py"]
    elif kind == "halpha":
        exact_names = ["halpha.py"]
        patterns = ["halpha*.py"]
    elif kind == "combine":
        exact_names = ["combine_sunspot.py"]
        patterns = ["combine_sunspot*.py"]
    else:
        raise ValueError(kind)

    searched: list[Path] = []
    candidates: list[Path] = []
    for base in resource_dirs():
        searched.append(base)
        exact = [base / name for name in exact_names]
        for p in exact:
            if p.exists():
                return p
        for pattern in patterns:
            candidates.extend(base.glob(pattern))

    candidates = [
        p for p in candidates
        if p.name.lower() != Path(__file__).name.lower()
        and "gui" not in p.stem.lower()
    ]

    if not candidates:
        places = "\n".join(str(x) for x in searched)
        raise FileNotFoundError(
            f"Could not find the {kind} Python module. Searched:\n{places}"
        )

    # Prefer the newest matching file, useful while versions are being updated.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_module(kind: str):
    # In a PyInstaller executable always use the module compiled into the EXE.
    # Never prefer a stray/older .py file sitting beside SeSuCo.exe.
    if getattr(sys, "frozen", False):
        module_names = {
            "sunspot": "sunspot",
            "halpha": "halpha",
            "combine": "combine_sunspot",
        }
        if kind not in module_names:
            raise ValueError(kind)
        module = importlib.import_module(module_names[kind])
        enable_unicode_cv2_io(module)
        module_path = Path(getattr(module, "__file__", module_names[kind]))
        return module, module_path

    # Normal Python/development mode: load the processing script from disk.
    path = find_module_file(kind)
    module_name = f"sesuco_{kind}_module"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    enable_unicode_cv2_io(module)
    return module, path



def enable_unicode_cv2_io(module):
    """Make OpenCV image I/O safe for Windows paths containing accents/Unicode.

    OpenCV's cv2.imread/cv2.imwrite can fail on some Windows installations when
    the path contains non-ASCII characters (for example: Csillagászat).
    Reading through numpy.fromfile + cv2.imdecode and writing through
    cv2.imencode + ndarray.tofile avoids that limitation.
    """
    if not hasattr(module, "cv2") or not hasattr(module, "np"):
        return

    cv2 = module.cv2
    np = module.np

    # Avoid wrapping the same shared cv2 module more than once.
    if getattr(cv2, "_sesuco_unicode_io", False):
        return

    original_imread = cv2.imread
    original_imwrite = cv2.imwrite

    def unicode_imread(filename, flags=cv2.IMREAD_COLOR):
        try:
            data = np.fromfile(str(filename), dtype=np.uint8)
            if data.size:
                img = cv2.imdecode(data, flags)
                if img is not None:
                    return img
        except Exception:
            pass
        return original_imread(str(filename), flags)

    def unicode_imwrite(filename, img, params=None):
        try:
            path = Path(filename)
            ext = path.suffix or ".png"
            encode_params = [] if params is None else list(params)
            ok, encoded = cv2.imencode(ext, img, encode_params)
            if ok:
                encoded.tofile(str(path))
                return True
        except Exception:
            pass

        if params is None:
            return original_imwrite(str(filename), img)
        return original_imwrite(str(filename), img, params)

    cv2.imread = unicode_imread
    cv2.imwrite = unicode_imwrite
    cv2._sesuco_unicode_io = True

def open_folder(path: Path):
    path = Path(path)
    if not path.exists():
        return
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        messagebox.showerror("Open folder", str(exc))


class SeSuCoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        icon = resource_path("sesuco.ico")
        if icon is not None:
            try:
                self.iconbitmap(default=str(icon))
            except tk.TclError:
                pass
        self.geometry("760x650")
        self.minsize(720, 610)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self._setup_style()

        self.container = ttk.Frame(self, padding=18)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for cls in (HomeFrame, SunspotFrame, HAlphaFrame, MonthlyFrame):
            frame = cls(self.container, self)
            self.frames[cls.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)
        self.show_frame("HomeFrame")

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Result.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 12, "bold"), padding=(14, 12))

    def show_frame(self, name: str):
        frame = self.frames[name]
        frame.tkraise()

    def run_threaded(self, owner, worker):
        """Run processing without freezing the GUI."""
        owner.set_busy(True)

        def runner():
            try:
                result = worker()
            except Exception as exc:
                details = traceback.format_exc()
                self.after(0, lambda: owner.on_error(exc, details))
            else:
                self.after(0, lambda: owner.on_success(result))
            finally:
                self.after(0, lambda: owner.set_busy(False))

        threading.Thread(target=runner, daemon=True).start()


class BaseToolFrame(ttk.Frame):
    def __init__(self, parent, app: SeSuCoApp):
        super().__init__(parent)
        self.app = app
        self.status_var = tk.StringVar(value="Ready")
        self.output_dir: Path | None = None

    def header(self, title: str, subtitle: str):
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 14))
        ttk.Label(top, text=title, style="Title.TLabel").pack(side="left", padx=18)
        ttk.Label(self, text=subtitle, style="Subtitle.TLabel", wraplength=690).pack(anchor="w", pady=(0, 12))

    def add_status_bar(self):
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", pady=(12, 0))

        ttk.Button(
            bottom,
            text="← Back to main menu",
            command=lambda: self.app.show_frame("HomeFrame"),
        ).pack(side="left")

        ttk.Label(bottom, text="Status:").pack(side="left", padx=(14, 0))
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", padx=(6, 0))

        ttk.Label(
            bottom,
            text=COPYRIGHT_TEXT,
            font=("Segoe UI", 8),
        ).pack(side="right")

    def set_busy(self, busy: bool):
        self.status_var.set("Processing…" if busy else self.status_var.get())
        state = "disabled" if busy else "normal"
        for widget in getattr(self, "action_widgets", []):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def on_error(self, exc: Exception, details: str):
        self.status_var.set("Error")
        messagebox.showerror("SeSuCo error", f"{exc}\n\nDetails were printed to the Python console/log.")
        print(details)

    def new_run(self):
        pass


class HomeFrame(ttk.Frame):
    def __init__(self, parent, app: SeSuCoApp):
        super().__init__(parent)
        self.app = app

        ttk.Label(self, text="🌞  SeSuCo", style="Title.TLabel").pack(pady=(25, 4))
        ttk.Label(self, text="Solar Observation Tools", font=("Segoe UI", 13)).pack(pady=(0, 25))

        panel = ttk.Frame(self)
        panel.pack(fill="x", padx=85)

        ttk.Button(
            panel, text="Sunspot Counter", style="Big.TButton",
            command=lambda: app.show_frame("SunspotFrame")
        ).pack(fill="x", pady=8)
        ttk.Button(
            panel, text="H-alpha Analyzer", style="Big.TButton",
            command=lambda: app.show_frame("HAlphaFrame")
        ).pack(fill="x", pady=8)
        ttk.Button(
            panel, text="Monthly Sunspot Summary", style="Big.TButton",
            command=lambda: app.show_frame("MonthlyFrame")
        ).pack(fill="x", pady=8)

        ttk.Separator(self).pack(fill="x", padx=85, pady=25)
        ttk.Button(self, text="Exit", command=app.destroy).pack()

        footer = ttk.Frame(self)
        footer.pack(side="bottom", fill="x", pady=(20, 0))
        ttk.Separator(footer).pack(fill="x", pady=(0, 8))
        ttk.Label(
            footer,
            text=COPYRIGHT_TEXT,
            font=("Segoe UI", 8),
        ).pack(side="right")


class SunspotFrame(BaseToolFrame):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.image_path: Path | None = None
        self.file_var = tk.StringVar(value="No image selected")

        self.header("Sunspot Counter", "White-light Seestar sunspot and facular-field analysis.")

        file_box = ttk.LabelFrame(self, text="Input image", padding=12)
        file_box.pack(fill="x")
        ttk.Label(file_box, textvariable=self.file_var, wraplength=570).pack(side="left", fill="x", expand=True)
        browse = ttk.Button(file_box, text="Browse…", command=self.browse)
        browse.pack(side="right", padx=(10, 0))

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=14)
        self.run_btn = ttk.Button(buttons, text="Run", style="Big.TButton", command=self.run)
        self.run_btn.pack(side="left")
        self.new_btn = ttk.Button(buttons, text="New run", command=self.new_run)
        self.new_btn.pack(side="left", padx=8)
        self.open_btn = ttk.Button(buttons, text="Open output folder", command=self.open_output, state="disabled")
        self.open_btn.pack(side="left")

        results = ttk.LabelFrame(self, text="Results", padding=12)
        results.pack(fill="x", pady=(0, 12))
        self.result_vars = {name: tk.StringVar(value="–") for name in ("Groups", "Spots", "R_raw", "Facular fields")}
        for i, (name, var) in enumerate(self.result_vars.items()):
            ttk.Label(results, text=f"{name}:").grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=(0, 8), pady=5)
            ttk.Label(results, textvariable=var, style="Result.TLabel").grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=(0, 45), pady=5)

        outputs = ttk.LabelFrame(self, text="Created files", padding=8)
        outputs.pack(fill="both", expand=True)
        self.output_list = tk.Listbox(outputs, height=8)
        self.output_list.pack(fill="both", expand=True)

        self.action_widgets = [browse, self.run_btn, self.new_btn]
        self.add_status_bar()

    def browse(self):
        name = filedialog.askopenfilename(
            title="Select white-light solar image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if name:
            self.image_path = Path(name)
            self.file_var.set(str(self.image_path))
            self.status_var.set("Ready")

    def run(self):
        if not self.image_path:
            messagebox.showwarning("Sunspot Counter", "Select an image first.")
            return
        self.app.run_threaded(self, self._worker)

    def _worker(self):
        module, module_path = load_module("sunspot")
        outdir = self.image_path.parent
        result = module.process_image(self.image_path, outdir, 0.0)

        facula_count = None
        daily = Path(result["daily"])
        if daily.exists():
            try:
                with daily.open("r", encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
                matching = [r for r in rows if r.get("image_name") == self.image_path.name]
                row = matching[-1] if matching else (rows[-1] if rows else None)
                if row:
                    facula_count = row.get("facula_field_count")
            except Exception:
                pass

        return {
            "module": module_path.name,
            "g": result["g"],
            "s": result["s"],
            "R_raw": result["R_raw"],
            "facula": facula_count if facula_count not in (None, "") else "–",
            "files": [str(v) for k, v in result.items() if k in {"annotated", "spots", "candidates", "groups", "faculae", "daily"}],
            "outdir": outdir,
        }

    def on_success(self, result):
        self.result_vars["Groups"].set(str(result["g"]))
        self.result_vars["Spots"].set(str(result["s"]))
        self.result_vars["R_raw"].set(str(result["R_raw"]))
        self.result_vars["Facular fields"].set(str(result["facula"]))
        self.output_list.delete(0, tk.END)
        for p in result["files"]:
            self.output_list.insert(tk.END, Path(p).name)
        self.output_dir = Path(result["outdir"])
        self.open_btn.configure(state="normal")
        self.status_var.set("Finished")

    def new_run(self):
        self.image_path = None
        self.file_var.set("No image selected")
        for var in self.result_vars.values():
            var.set("–")
        self.output_list.delete(0, tk.END)
        self.output_dir = None
        self.open_btn.configure(state="disabled")
        self.status_var.set("Ready")

    def open_output(self):
        if self.output_dir:
            open_folder(self.output_dir)


class HAlphaFrame(BaseToolFrame):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.image_path: Path | None = None
        self.file_var = tk.StringVar(value="No image selected")

        self.header(
            "H-alpha Analyzer",
            "H-alpha filament and active-region analysis. A matching daily_summary*.csv must be in the same folder as the selected image.",
        )

        file_box = ttk.LabelFrame(self, text="Input image", padding=12)
        file_box.pack(fill="x")
        ttk.Label(file_box, textvariable=self.file_var, wraplength=570).pack(side="left", fill="x", expand=True)
        browse = ttk.Button(file_box, text="Browse…", command=self.browse)
        browse.pack(side="right", padx=(10, 0))

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=14)
        self.run_btn = ttk.Button(buttons, text="Run", style="Big.TButton", command=self.run)
        self.run_btn.pack(side="left")
        self.new_btn = ttk.Button(buttons, text="New run", command=self.new_run)
        self.new_btn.pack(side="left", padx=8)
        self.open_btn = ttk.Button(buttons, text="Open output folder", command=self.open_output, state="disabled")
        self.open_btn.pack(side="left")

        results = ttk.LabelFrame(self, text="Results", padding=12)
        results.pack(fill="x", pady=(0, 12))
        self.fil_var = tk.StringVar(value="–")
        self.ar_var = tk.StringVar(value="–")
        ttk.Label(results, text="Filaments:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Label(results, textvariable=self.fil_var, style="Result.TLabel").grid(row=0, column=1, sticky="w", padx=(0, 45))
        ttk.Label(results, text="Active regions:").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=5)
        ttk.Label(results, textvariable=self.ar_var, style="Result.TLabel").grid(row=0, column=3, sticky="w")

        outputs = ttk.LabelFrame(self, text="Created files", padding=8)
        outputs.pack(fill="both", expand=True)
        self.output_list = tk.Listbox(outputs, height=8)
        self.output_list.pack(fill="both", expand=True)

        self.action_widgets = [browse, self.run_btn, self.new_btn]
        self.add_status_bar()

    def browse(self):
        name = filedialog.askopenfilename(
            title="Select H-alpha image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")],
        )
        if name:
            self.image_path = Path(name)
            self.file_var.set(str(self.image_path))
            self.status_var.set("Ready")

    def run(self):
        if not self.image_path:
            messagebox.showwarning("H-alpha Analyzer", "Select an image first.")
            return
        self.app.run_threaded(self, self._worker)

    def _worker(self):
        module, module_path = load_module("halpha")
        folder = self.image_path.parent
        ha_file = self.image_path

        geom = module.read_solar_geometry(folder, ha_file)
        rotation_deg = module.ACUTER_CAMERA_ANGLE_DEG - geom["P_deg"]

        img = module.cv2.imread(str(ha_file), module.cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {ha_file}")

        gray = module.cv2.cvtColor(img, module.cv2.COLOR_BGR2GRAY)
        cx, cy, _ = module.detect_disk(gray)
        rotated = module.rotate_image(img, rotation_deg, (cx, cy))

        analysis = module.analyze_halpha(rotated, geom["B0_deg"])
        annotated = module.annotate(rotated, analysis, geom, rotation_deg)
        features, summary = module.build_tables(analysis, geom, rotation_deg, ha_file)

        out_img = folder / f"{ha_file.stem}{module.ANNOTATED_SUFFIX}"
        out_features = folder / module.FEATURES_FILENAME
        out_summary = folder / module.SUMMARY_FILENAME

        module.cv2.imwrite(str(out_img), annotated, [module.cv2.IMWRITE_JPEG_QUALITY, 95])
        features.to_csv(out_features, index=False, encoding="utf-8-sig")
        summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

        return {
            "module": module_path.name,
            "filaments": len(analysis["filaments"]),
            "ars": len(analysis["ars"]),
            "files": [str(out_img), str(out_features), str(out_summary)],
            "outdir": folder,
        }

    def on_success(self, result):
        self.fil_var.set(str(result["filaments"]))
        self.ar_var.set(str(result["ars"]))
        self.output_list.delete(0, tk.END)
        for p in result["files"]:
            self.output_list.insert(tk.END, Path(p).name)
        self.output_dir = Path(result["outdir"])
        self.open_btn.configure(state="normal")
        self.status_var.set("Finished")

    def new_run(self):
        self.image_path = None
        self.file_var.set("No image selected")
        self.fil_var.set("–")
        self.ar_var.set("–")
        self.output_list.delete(0, tk.END)
        self.output_dir = None
        self.open_btn.configure(state="disabled")
        self.status_var.set("Ready")

    def open_output(self):
        if self.output_dir:
            open_folder(self.output_dir)


class MonthlyFrame(BaseToolFrame):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.base_dir: Path | None = None
        self.folder_var = tk.StringVar(value="No Solar folder selected")
        self.month_var = tk.StringVar(value=MONTHS[0])
        self.days_var = tk.StringVar(value="–")

        self.header(
            "Monthly Sunspot Summary",
            "Collect daily_summary CSV files for one month and create the monthly VOTable and plot.",
        )

        folder_box = ttk.LabelFrame(self, text="Solar base folder", padding=12)
        folder_box.pack(fill="x")
        ttk.Label(folder_box, textvariable=self.folder_var, wraplength=570).pack(side="left", fill="x", expand=True)
        browse = ttk.Button(folder_box, text="Browse…", command=self.browse_folder)
        browse.pack(side="right", padx=(10, 0))

        select = ttk.Frame(self)
        select.pack(fill="x", pady=14)
        ttk.Label(select, text="Month:").pack(side="left")
        month_box = ttk.Combobox(select, textvariable=self.month_var, values=MONTHS, state="readonly", width=18)
        month_box.pack(side="left", padx=8)
        self.run_btn = ttk.Button(select, text="Run", style="Big.TButton", command=self.run)
        self.run_btn.pack(side="left", padx=8)
        self.new_btn = ttk.Button(select, text="New run", command=self.new_run)
        self.new_btn.pack(side="left")
        self.open_btn = ttk.Button(select, text="Open output folder", command=self.open_output, state="disabled")
        self.open_btn.pack(side="left", padx=8)

        result = ttk.LabelFrame(self, text="Result", padding=12)
        result.pack(fill="x", pady=(0, 12))
        ttk.Label(result, text="Observation days found:").pack(side="left")
        ttk.Label(result, textvariable=self.days_var, style="Result.TLabel").pack(side="left", padx=8)

        outputs = ttk.LabelFrame(self, text="Created files", padding=8)
        outputs.pack(fill="both", expand=True)
        self.output_list = tk.Listbox(outputs, height=8)
        self.output_list.pack(fill="both", expand=True)

        self.action_widgets = [browse, month_box, self.run_btn, self.new_btn]
        self.add_status_bar()

    def browse_folder(self):
        name = filedialog.askdirectory(title="Select Solar base folder")
        if name:
            self.base_dir = Path(name)
            self.folder_var.set(str(self.base_dir))
            self.status_var.set("Ready")

    def run(self):
        if not self.base_dir:
            messagebox.showwarning("Monthly Sunspot Summary", "Select the Solar base folder first.")
            return
        self.app.run_threaded(self, self._worker)

    def _worker(self):
        module, module_path = load_module("combine")
        month_num = MONTHS.index(self.month_var.get()) + 1
        honap_cim, honap_fajlnev = module.HONAPOK[month_num]

        files = module.find_daily_summary_files(self.base_dir)
        if not files:
            raise RuntimeError("No daily_summary CSV files were found.")

        records = []
        for path in files:
            rec = module.read_summary_file(path)
            if rec is None:
                continue
            if module.pd.isna(rec["date"]):
                continue
            if rec["date"].month == month_num:
                records.append(rec)

        if not records:
            raise RuntimeError(f"No observations were found for {self.month_var.get()}.")

        monthly_df = module.pd.DataFrame(records)
        monthly_df = monthly_df[["date", "g", "s", "R_raw", "source_file"]]
        monthly_df = monthly_df.sort_values("date")
        monthly_df = monthly_df.drop_duplicates(subset=["date"], keep="last")
        monthly_df["day"] = monthly_df["date"].dt.day
        monthly_df["date"] = monthly_df["date"].dt.strftime("%Y-%m-%d")
        monthly_df = monthly_df[["date", "day", "g", "s", "R_raw", "source_file"]]

        vo_path = self.base_dir / f"{honap_fajlnev}.vo"
        table = module.Table.from_pandas(monthly_df)
        table.write(vo_path, format="votable", overwrite=True)

        plot_path = self.base_dir / f"{honap_fajlnev}_napfoltszam.png"
        module.create_monthly_plot(monthly_df, honap_cim, plot_path)

        return {
            "module": module_path.name,
            "days": len(monthly_df),
            "files": [str(vo_path), str(plot_path)],
            "outdir": self.base_dir,
        }

    def on_success(self, result):
        self.days_var.set(str(result["days"]))
        self.output_list.delete(0, tk.END)
        for p in result["files"]:
            self.output_list.insert(tk.END, Path(p).name)
        self.output_dir = Path(result["outdir"])
        self.open_btn.configure(state="normal")
        self.status_var.set("Finished")

    def new_run(self):
        self.days_var.set("–")
        self.output_list.delete(0, tk.END)
        self.output_dir = None
        self.open_btn.configure(state="disabled")
        self.status_var.set("Ready")

    def open_output(self):
        if self.output_dir:
            open_folder(self.output_dir)


if __name__ == "__main__":
    app = SeSuCoApp()
    app.mainloop()
