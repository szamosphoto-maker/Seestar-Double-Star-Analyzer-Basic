# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "scipy.ndimage",
    "astropy",
    "astropy.table",
    "matplotlib",
    "matplotlib.pyplot",
    "tzdata",
]

# Dynamic processing modules import parts of these packages, so include their
# submodules explicitly for a robust one-file build.
hiddenimports += collect_submodules("scipy.ndimage")
hiddenimports += collect_submodules("astropy.table")

a = Analysis(
    ["sesuco_gui.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("sunspot.py", "."),
        ("halpha.py", "."),
        ("combine_sunspot.py", "."),
        ("sesuco.ico", "."),
    ],
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
