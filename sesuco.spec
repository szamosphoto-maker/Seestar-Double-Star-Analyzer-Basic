# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# The GUI loads the three processing modules dynamically from bundled .py files.
# Listing them as hidden imports makes PyInstaller ANALYZE their imports too,
# so dependencies such as zoneinfo are not omitted from the executable.
hiddenimports = [
    "sunspot",
    "halpha",
    "combine_sunspot",
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "scipy.ndimage",
    "astropy",
    "astropy.table",
    "matplotlib",
    "matplotlib.pyplot",
    "zoneinfo",
    "zoneinfo._common",
    "zoneinfo._tzpath",
    "tzdata",
]

hiddenimports += collect_submodules("scipy.ndimage")
hiddenimports += collect_submodules("astropy.table")
hiddenimports += collect_submodules("zoneinfo")

# Windows has no system IANA timezone database.  ZoneInfo therefore needs the
# data shipped by the tzdata package (e.g. Europe/Budapest).
datas = collect_data_files("tzdata")
datas += [
    ("sunspot.py", "."),
    ("halpha.py", "."),
    ("combine_sunspot.py", "."),
    ("sesuco.ico", "."),
]

a = Analysis(
    ["sesuco_gui.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SeSuCo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["sesuco.ico"],
)
