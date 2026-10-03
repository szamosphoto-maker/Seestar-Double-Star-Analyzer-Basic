# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# The processing modules are imported from the PyInstaller bundle when frozen.
# Hidden imports force PyInstaller to analyse and package them and their dependencies.
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

# Windows has no system IANA timezone database, therefore bundle tzdata.
# Only the icon is needed as a loose data file; processing .py files are compiled
# into the executable through hiddenimports above.
datas = collect_data_files("tzdata")
datas += [
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
