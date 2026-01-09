# -*- mode: python ; coding: utf-8 -*-
import sys
import os
import vispy
import freetype
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs


block_cipher = None

datas = []
binaries = []
hiddenimports = []

# -------------------------------------------------------------------------
# [핵심 해결책] Vispy 폴더 강제 매핑 (경로 문제 원천 봉쇄)
# -------------------------------------------------------------------------
# Vispy가 설치된 실제 경로를 찾습니다.
vispy_path = os.path.dirname(vispy.__file__)

# Vispy 폴더 전체를 실행 파일 내부의 'vispy' 폴더로 그대로 복사합니다.
# 이렇게 하면 코드가 폰트를 찾을 때 경로가 꼬일 일이 없습니다.
datas.append((vispy_path, 'vispy'))

# -------------------------------------------------------------------------
# [핵심 해결책] Freetype 바이너리 강제 수집
# -------------------------------------------------------------------------
# Freetype DLL을 확실하게 챙깁니다.
binaries += collect_dynamic_libs('freetype')

# -------------------------------------------------------------------------
# 나머지 패키지 수집
# -------------------------------------------------------------------------
packages_to_collect = [
    'napari',
    'napari_builtins',
    'imageio',
    'scipy',
    'superqt',
    'PIL',
    'magicgui',
    'dask',
    'jinja2',
    'napari_svg',
    'numpy'
]

for package in packages_to_collect:
    try:
        # Vispy는 위에서 수동으로 처리했으므로 제외하고 나머지만 수집
        if package == 'vispy': 
            continue
            
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception:
        pass

a = Analysis(
    ['dqe_v1_patched.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 충돌 방지
    excludes=['PySide6', 'shiboken6', 'PyQt6', 'tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='XrayInspector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # 에러 확인용 (성공 시 False 변경)
    icon='program_icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='XrayInspector',
)