# -*- mode: python ; coding: utf-8 -*-
import os
import vispy
import freetype
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs


block_cipher = None

datas = []
binaries = []
hiddenimports = []

# Vispy: bundle full package folder to avoid font path issues
vispy_path = os.path.dirname(vispy.__file__)
datas.append((vispy_path, "vispy"))

# Freetype DLLs
binaries += collect_dynamic_libs("freetype")

# Package data/hidden imports
packages_to_collect = [
    "napari",
    "napari_builtins",
    "imageio",
    "scipy",
    "PIL",
    "magicgui",
    "dask",
    "jinja2",
    "napari_svg",
    "numpy",
]

for package in packages_to_collect:
    try:
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception:
        pass

# Default config
datas += [("blemish_config.json", ".")]

a = Analysis(
    ["dqe_v16.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6",
        "PySide6-Addons",
        "PySide6-Essentials",
        "shiboken6",
        "PyQt6",
        "tkinter",
        "scipy._lib.array_api_compat.torch",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dqe_v16",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["program_icon.ico"],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="dqe_v16",
)
