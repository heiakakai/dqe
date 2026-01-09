"""
X-ray Spot Tracer & Normalizer (napari edition)

- 기존 Tkinter+matplotlib 기반(dqe_13.py)의 "Raw/Result 2-pane + ROI 기반 라인노이즈 보정 + Blemish 검출/저장" 흐름을
  napari(=Qt+vispy 기반 GPU 렌더링 뷰어)로 재구성한 스크립트입니다.

주의 / 범위
- 이 스크립트는 "napari 공식 예제(Multiple viewer widget)" 방식으로 1개 napari 윈도우 안에
  여러 Viewer(정확히는 ViewerModel+QtViewer)를 QSplitter로 배치합니다.
- 따라서 QtViewer, ViewerModel 등을 사용합니다. 이는 napari가 공식 문서/예제에서 사용하는 경로이긴 하지만,
  napari 버전 변화에 따라 일부 API가 바뀔 수 있습니다(완전히 배제 불가).

실행
    python xray_spot_tracer_napari.py

필수 패키지(예시)
    pip install "napari[all]" scipy superqt

    - napari[all]  : napari + Qt + vispy + image IO 등
    - scipy        : uniform_filter, label, find_objects, binary_erosion
    - superqt      : (선택) qthrottled 유틸. 없으면 내부 타이머로 대체해도 됨.
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
from scipy.ndimage import uniform_filter, label as nd_label, find_objects, binary_erosion, binary_dilation

# Text overlay rendering (robust across napari versions)
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
from PIL import ImageFont as PILImageFont

from qtpy.QtCore import Qt, QTimer, Signal, QObject, QThread, QEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# -----------------------------------------------------------------------------
# Runtime fixes for frozen builds (PyInstaller) on Windows
#
# Symptom: freetype.ft_errors.FT_Exception: (cannot open resource)
# Root cause (highly likely): VisPy's default TextVisual face is 'OpenSans', which
# is loaded from a file path that can include non-ASCII characters when bundled.
# On some Windows setups, freetype-py cannot open font paths that contain
# non-ASCII characters, causing VisPy/Napari overlays (e.g. scale bar) to crash
# at startup.
#
# Fix strategy:
#  1) Force VisPy text visuals to use a system font (Arial/Segoe UI) instead of
#     the bundled 'OpenSans' font, avoiding problematic file paths.
#  2) Best effort: set VisPy's data dir to an ASCII-only temp folder.
# -----------------------------------------------------------------------------
def _patch_vispy_font_for_frozen_app() -> None:
    try:
        import sys, os, tempfile

        # 2) Use ASCII-only VisPy data dir (best effort; does not hurt on dev runs)
        try:
            from vispy.util.config import set_data_dir  # type: ignore
            vispy_data_dir = os.path.join(tempfile.gettempdir(), "vispy-data")
            set_data_dir(vispy_data_dir, create=True)
        except Exception:
            pass

        # 1) Monkeypatch VisPy TextVisual default font face
        try:
            from vispy.util.fonts import list_fonts  # type: ignore
            available = {f.lower() for f in list_fonts()}
        except Exception:
            available = set()

        preferred_faces = ["Arial", "Segoe UI", "DejaVu Sans", "Liberation Sans"]
        fallback_face = None
        for f in preferred_faces:
            if not available or f.lower() in available:
                fallback_face = f
                break
        if fallback_face is None:
            fallback_face = "Arial"

        import vispy.visuals.text.text as _vispy_text  # type: ignore

        _orig_init = _vispy_text.TextVisual.__init__

        def _patched_init(self, *args, **kwargs):
            # If caller doesn't specify face, or uses VisPy default 'OpenSans',
            # switch to a system font (ASCII path under Windows Fonts dir).
            if "face" not in kwargs or kwargs.get("face") == "OpenSans":
                kwargs["face"] = fallback_face
            return _orig_init(self, *args, **kwargs)

        _vispy_text.TextVisual.__init__ = _patched_init  # type: ignore[attr-defined]
    except Exception:
        # If anything goes wrong here, do not block startup.
        return


_patch_vispy_font_for_frozen_app()
import napari
from napari.components.viewer_model import ViewerModel
from napari.layers import Image as NapariImageLayer
from napari.layers import Shapes as NapariShapesLayer
from napari.layers import Points as NapariPointsLayer
from napari.qt import QtViewer  # napari 공식 예제에서 사용


def _remove_layers_by_prefix(viewer: ViewerModel, prefix: str) -> None:
    """viewer.layers에서 name이 prefix와 같거나 prefix로 시작하는 레이어를 모두 제거합니다.

    napari 버전에 따라 LayerList에 dict-like `get()`이 없어서,
    동일 이름 레이어가 누적되며 자동으로 '[1]' suffix가 붙는 문제가 생길 수 있습니다.
    이를 방지하기 위해 prefix 매칭으로 일괄 정리합니다.
    """
    try:
        layers = list(getattr(viewer, "layers", []))
    except Exception:
        return

    to_remove = []
    for ly in layers:
        try:
            nm = getattr(ly, "name", "")
        except Exception:
            nm = ""
        if isinstance(nm, str) and (nm == prefix or nm.startswith(prefix)):
            to_remove.append(ly)

    for ly in to_remove:
        try:
            viewer.layers.remove(ly)
        except Exception:
            pass


def _trace_binary_boundary(mask: np.ndarray) -> Optional[np.ndarray]:
    """2D binary mask에서 외곽선을 (N,2) 좌표(y,x)로 반환합니다.

    skimage/cv2 없이 동작하도록 Moore-Neighbor tracing을 구현했습니다.
    - 입력: mask (H,W) bool/0-1
    - 출력: (N,2) float64, 각 row는 (y,x)

    제한:
    - 매우 복잡한 경계에서 완벽한 단일 루프가 아닐 수 있습니다.
    - 그럼에도 napari Shapes의 'path' 오버레이로 "매우 얇은 선" 표시 목적에는 충분합니다.
    """
    try:
        m = (mask.astype(bool)).copy()
        if m.ndim != 2 or m.size == 0 or not m.any():
            return None

        # 패딩을 두어 경계 처리 단순화
        H, W = m.shape
        pm = np.zeros((H + 2, W + 2), dtype=bool)
        pm[1:-1, 1:-1] = m

        # 시작점: top-most, left-most boundary pixel
        ys, xs = np.nonzero(pm)
        if ys.size == 0:
            return None
        # boundary 후보로 제한(이웃 중 하나라도 배경이면 boundary)
        # (빠르게 찾기 위해 전체를 스캔하지 않고, 후보 중에서 검사)
        best = None
        for y, x in zip(ys, xs):
            nb = pm[y-1:y+2, x-1:x+2]
            if nb.size == 9 and (not nb.all()):
                if best is None or (y < best[0]) or (y == best[0] and x < best[1]):
                    best = (y, x)
        if best is None:
            # 전부 꽉 찬 경우(희귀) -> 테두리는 이미지 가장자리
            best = (int(ys.min()), int(xs[ys.argmin()]))

        sy, sx = best

        # Moore-Neighbor (8방향) 순서: E,SE,S,SW,W,NW,N,NE
        dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        # "이전 방향"을 관리해 경계를 따라감
        # 시작은 W 방향에서 들어왔다고 가정(그래야 첫 탐색이 N쪽부터 시작되는 경향)
        prev_dir = 4  # W
        y, x = sy, sx
        out = [(y, x)]

        # 무한루프 방지
        max_steps = int(pm.size * 2)
        steps = 0
        while steps < max_steps:
            steps += 1
            # 이전 방향의 "다음"부터 시계방향으로 탐색
            found = False
            for k in range(8):
                d = (prev_dir + 1 + k) % 8
                dy, dx = dirs[d]
                ny, nx = y + dy, x + dx
                if pm[ny, nx]:
                    # 다음 점으로 이동
                    y, x = ny, nx
                    out.append((y, x))
                    # 다음 탐색은 반대 방향의 왼쪽부터(표준 Moore)
                    prev_dir = (d + 4) % 8
                    found = True
                    break
            if not found:
                break
            # 시작점으로 돌아오면 종료(최소 길이 조건)
            if len(out) > 10 and y == sy and x == sx:
                break

        # 패딩 제거 좌표로 보정
        arr = np.asarray(out, dtype=np.float64)
        arr[:, 0] -= 1.0
        arr[:, 1] -= 1.0
        # 클리핑
        arr[:, 0] = np.clip(arr[:, 0], 0, H - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, W - 1)
        return arr
    except Exception:
        return None


# ---------------------------
# Model Definitions (원본 코드와 동일한 모델/타입/크기 맵)
# ---------------------------

MODEL_DIMENSIONS: Dict[str, Dict[str, Tuple[Optional[int], Optional[int]]]] = {
    "4343": {"Probe": (3072, 3072), "PT": (3072, 3072), "MAP": (3072, 3072)},
    "3643": {"Probe": (3072, 3072), "PT": (3072, 2560), "MAP": (3072, 2560)},
    "2430": {"Probe": (3840, 3072), "PT": (3840, 3072), "MAP": (3840, 3072)},
    "1824": {"Probe": (3840, 3072), "PT": (3840, 3072), "MAP": (3840, 3072)},
    "1616": {"Probe": (3840, 3072), "PT": (1628, 1628), "MAP": (1628, 1628)},
    "1624": {"Probe": (3840, 3072), "PT": (3840, 3072), "MAP": (3840, 3072)},
}

# ---------------------------
# Fixed ROI presets (요청: 로딩 시 자동 ROI 생성)
# - 좌상단(start) / 우하단(end) 좌표는 (x, y) 기준
# - napari Shapes 좌표는 (y, x) 이므로 내부에서 변환합니다.
# ---------------------------
ROI_PRESETS_XY: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "4343": ((1, 1), (3072, 3072)),
    "3643": ((1, 1), (3065, 2556)),
    "2430": ((1, 1), (3840, 3072)),
    "1616": ((0, 548), (1650, 2191)),
    "1824": ((1, 1), (2303, 3072)),
    "1624": ((0, 548), (2477, 2191)),
}


# ---------------------------
# Result red-line box (요청: Result 이미지에 모델별 빨간 박스)
# - 좌상단(start) / 우하단(end) 좌표는 (x, y) 기준
# - napari Shapes 좌표는 (y, x)
# ---------------------------
RESULT_REDBOX_XY: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "4343": ((21, 21), (3052, 3052)),
    "3643": ((21, 21), (3045, 2536)),
    "2430": ((11, 11), (3830, 3072)),
    "1824": ((11, 11), (2303, 3062)),
    "1616": ((17, 556), (1642, 2183)),
    "1624": ((24, 556), (2455, 2183)),
}

def estimate_pixel_pitch_cm(
    model: str,
    *,
    img_w_px: int,
    img_h_px: int,
    orig_w_px: Optional[int] = None,
    orig_h_px: Optional[int] = None,
) -> Optional[float]:
    """모델명(예: '4343')과 활성 영역 픽셀 수로부터 픽셀 피치(cm)를 추정합니다.

    가정(확실하지 않음):
    - 모델 문자열이 4자리 숫자일 때, 앞 2자리/뒤 2자리가 물리 크기(cm)를 의미한다고 가정합니다.
      (예: 3643 -> 36cm x 43cm)

    구현:
    - 가능하면 RESULT_REDBOX_XY의 활성 영역(빨간 박스) 크기를 사용합니다.
    - ORIENT_ROTATE_LEFT_FLIP_H_MODELS 모델은 redbox 좌표를 동일한 변환으로 보정합니다.
    - (가로,세로) 매핑은 두 경우(앞2자리=가로 vs 뒤2자리=가로)를 모두 계산 후,
      x/y 피치가 가장 비슷해지는 쪽을 선택합니다.

    반환:
    - pixel_pitch_cm (cm/pixel) 또는 추정 불가 시 None
    """
    try:
        if not (isinstance(model, str) and len(model) == 4 and model.isdigit()):
            return None
        a_cm = int(model[:2])
        b_cm = int(model[2:])
        if a_cm <= 0 or b_cm <= 0:
            return None
    except Exception:
        return None

    # active area pixels (prefer result red box if available)
    aw = int(img_w_px)
    ah = int(img_h_px)
    preset = RESULT_REDBOX_XY.get(model)
    if preset is not None:
        try:
            (x1, y1), (x2, y2) = preset
            # orientation-corrected models: map presets into current displayed coordinates
            try:
                x1, y1, x2, y2 = orient_transform_box_xy(model, x1, y1, x2, y2, orig_w_px, orig_h_px)
            except Exception:
                pass
            aw = abs(int(x2) - int(x1))
            ah = abs(int(y2) - int(y1))
        except Exception:
            aw, ah = int(img_w_px), int(img_h_px)

    if aw < 2 or ah < 2:
        aw, ah = int(img_w_px), int(img_h_px)
    if aw < 2 or ah < 2:
        return None

    def _rel_diff(p1: float, p2: float) -> float:
        den = max(p1, p2, 1e-12)
        return abs(p1 - p2) / den

    # Option A: (a_cm -> width), (b_cm -> height)
    pxA = a_cm / aw
    pyA = b_cm / ah
    dA = _rel_diff(pxA, pyA)

    # Option B: swapped
    pxB = b_cm / aw
    pyB = a_cm / ah
    dB = _rel_diff(pxB, pyB)

    px, py = (pxA, pyA) if dA <= dB else (pxB, pyB)
    pitch = (float(px) + float(py)) / 2.0
    if not np.isfinite(pitch) or pitch <= 0:
        return None
    return float(pitch)




# ---------------------------
# Orientation correction models (요청)
# - 파일 로드 후: rotate 90° left(CCW) -> flip horizontally 순으로 적용
# - 표시/크롭/저장 모두 동일한 좌표계로 동작하도록 raw 자체를 변환
# ---------------------------
ORIENT_ROTATE_LEFT_FLIP_H_MODELS = {"1824", "1616", "1624"}


def orient_transform_box_xy(
    model: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    orig_w: Optional[int],
    orig_h: Optional[int],
) -> Tuple[int, int, int, int]:
    """Apply orientation correction to a box in (x,y) coords.

    For models in ORIENT_ROTATE_LEFT_FLIP_H_MODELS, raw is transformed at load:
      rotate 90° left (CCW) -> flip horizontally

    This helper transforms a *boundary* box (x1,y1)-(x2,y2) where (x2,y2) is treated as
    an exclusive upper bound (slice-style), matching how ROI/red-box presets are used.

    Mapping for boundary coordinates with original width W and height H:
      (x, y) -> (H - y, W - x)
    """
    if model not in ORIENT_ROTATE_LEFT_FLIP_H_MODELS:
        return int(x1), int(y1), int(x2), int(y2)
    if orig_w is None or orig_h is None:
        return int(x1), int(y1), int(x2), int(y2)

    W = int(orig_w)
    H = int(orig_h)

    nx1 = H - int(y2)
    nx2 = H - int(y1)
    ny1 = W - int(x2)
    ny2 = W - int(x1)

    return int(nx1), int(ny1), int(nx2), int(ny2)



def orient_inverse_image(model: str, img: np.ndarray) -> np.ndarray:
    """Inverse orientation transform for images.

    Forward transform used at load for ORIENT_ROTATE_LEFT_FLIP_H_MODELS:
      img_fwd = fliplr(rot90(img, k=1))   # rotate 90° left (CCW) then flip horizontally

    Inverse:
      img = rot90(fliplr(img_fwd), k=3)  # flip back then rotate 90° right (CW)
    """
    if model not in ORIENT_ROTATE_LEFT_FLIP_H_MODELS:
        return img
    try:
        return np.rot90(np.fliplr(img), k=3)
    except Exception:
        return img


# ---------------------------
# Blemish Parameters (원본과 호환되는 JSON 저장)
# ---------------------------

def default_blemish_params() -> Dict[str, Any]:
    return {
        "analysis_scope": "ROI",  # "ROI" or "Full"
        "use_roi_as_reference": True,
        "bg_window": 51,
        "measure_window": 1,        "threshold_ref_percent_u": 2.0,
        "threshold_ref_percent_l": 2.0,
        # NOTE: CropΔ threshold band (abs 기준)
        # - abs(CropΔ%) < L : 제외
        # - L <= abs(CropΔ%) < U : list에 표시 + 확인 팝업
        # - abs(CropΔ%) >= U : list에 표시 (팝업 없음)
        # legacy 호환: threshold_ref_percent는 U와 동일하게 유지
        "threshold_ref_percent": 2.0,
        "threshold_crop_percent": 2.0,
        "min_area": 5,
        "max_area": 5000,
        "border_margin": 30,
        "max_detections": 20,
        "min_dn": 1,
        "min_circularity": 0.0,
        "max_eccentricity": 1.0,
        "overlay_bottom_margin_px": 5,
        "fixed_line_x": [],
        "fixed_line_y": [],

    }


# ---------------------------
# Utility
# ---------------------------

def percentile_clim(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    roi_bounds: Optional["RoiBounds"] = None,
) -> Tuple[float, float]:
    """Auto L/W(contrast limits)를 퍼센타일로 계산합니다.

    - roi_bounds가 유효하면 ROI 영역만 사용합니다.
    - roi_bounds가 없거나 비정상이면 전체 이미지로 계산합니다.
    """
    if img.size == 0:
        return (0.0, 1.0)

    view = img
    try:
        if roi_bounds is not None and getattr(roi_bounds, "valid", False) and img.ndim >= 2:
            y1 = clamp_int(roi_bounds.y1, 0, img.shape[0])
            y2 = clamp_int(roi_bounds.y2, 0, img.shape[0])
            x1 = clamp_int(roi_bounds.x1, 0, img.shape[1])
            x2 = clamp_int(roi_bounds.x2, 0, img.shape[1])
            if (y2 - y1) >= 2 and (x2 - x1) >= 2:
                view = img[y1:y2, x1:x2]
    except Exception:
        view = img

    if view.size == 0:
        view = img

    vmin = float(np.percentile(view, p_low))
    vmax = float(np.percentile(view, p_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        try:
            vmin = float(np.nanmin(view))
            vmax = float(np.nanmax(view))
        except Exception:
            vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return (vmin, vmax)


def clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def show_err(parent: QWidget, title: str, msg: str) -> None:
    QMessageBox.critical(parent, title, msg)


def show_info(parent: QWidget, title: str, msg: str) -> None:
    QMessageBox.information(parent, title, msg)



# ---------------------------
# napari 버전 호환용: ViewerModel에 add_image/add_shapes/add_points가 없거나,
# 내부 API가 바뀐 경우를 대비한 래퍼.
# - 핵심: LayerList(model.layers)에는 add_image 같은 메서드가 없습니다.
#   반드시 viewer/viewermodel의 add_* 를 사용해야 합니다.
# ---------------------------

def add_image_to_model(
    model: ViewerModel,
    data: np.ndarray,
    *,
    name: str,
    colormap: str = "gray",
    contrast_limits: Tuple[float, float] = (0.0, 1.0),
    interpolation: str = "nearest",
) -> NapariImageLayer:
    """가능한 경우 model.add_image를 쓰고, 실패 시 레이어를 생성해 layers에 append합니다."""
    try:
        return model.add_image(
            data,
            name=name,
            colormap=colormap,
            contrast_limits=contrast_limits,
            interpolation=interpolation,
        )
    except Exception:
        layer = NapariImageLayer(data)
        layer.name = name
        try:
            layer.colormap = colormap
        except Exception:
            pass
        try:
            layer.contrast_limits = contrast_limits
        except Exception:
            pass
        try:
            layer.interpolation2d = interpolation
        except Exception:
            pass
        model.layers.append(layer)
        return layer


def add_shapes_to_model(
    model: ViewerModel,
    data=None,
    *,
    name: str,
    shape_type: str = "rectangle",
    edge_color: str = "yellow",
    face_color=(0, 0, 0, 0),
    edge_width: int = 2,
) -> NapariShapesLayer:
    try:
        return model.add_shapes(
            data=([] if data is None else data),
            name=name,
            shape_type=shape_type,
            edge_color=edge_color,
            face_color=face_color,
            edge_width=edge_width,
        )
    except Exception:
        layer = NapariShapesLayer(
            data=([] if data is None else data),
            shape_type=shape_type,
            edge_color=edge_color,
            face_color=face_color,
            edge_width=edge_width,
            name=name,
        )
        model.layers.append(layer)
        return layer


def add_points_to_model(
    model: ViewerModel,
    data: np.ndarray,
    *,
    name: str,
    size: float = 8.0,
    face_color: str = "red",
    # NOTE: 일부 napari 버전에서는 Points에 edge_color 인자가 없습니다.
    # (예: Points.__init__() got an unexpected keyword argument 'edge_color')
    # 따라서 호환성을 위해 인자는 받되 실제 생성에는 사용하지 않습니다.
    edge_color: str = "white",
    opacity: float = 0.9,
) -> NapariPointsLayer:
    try:
        return model.add_points(
            data,
            name=name,
            size=size,
            face_color=face_color,
            opacity=opacity,
        )
    except Exception:
        layer = NapariPointsLayer(data, name=name, size=size, face_color=face_color, opacity=opacity)
        model.layers.append(layer)
        return layer


# ---------------------------
# ROI extraction from napari Shapes rectangle
# napari shapes는 (row=y, col=x) 좌표계를 사용합니다.
# rectangle은 보통 4개 vertex가 들어오므로 min/max로 bbox를 뽑습니다.
# ---------------------------

@dataclass
class RoiBounds:
    x1: int
    x2: int
    y1: int
    y2: int

    @property
    def valid(self) -> bool:
        return (self.x2 - self.x1) >= 2 and (self.y2 - self.y1) >= 2


def roi_from_shapes_layer(layer: NapariShapesLayer, width: int, height: int) -> Optional[RoiBounds]:
    if layer is None:
        return None
    if layer.data is None or len(layer.data) == 0:
        return None

    # 여기서는 "가장 마지막(최근) ROI 1개"만 사용 (원본도 1개 ROI만 관리)
    rect = layer.data[-1]  # shape: (N, 2) in (y, x)
    if rect is None or len(rect) < 2:
        return None

    ys = rect[:, 0]
    xs = rect[:, 1]
    y1 = int(max(0, min(height - 1, float(ys.min()))))
    y2 = int(max(0, min(height, float(ys.max()))))
    x1 = int(max(0, min(width - 1, float(xs.min()))))
    x2 = int(max(0, min(width, float(xs.max()))))

    rb = RoiBounds(x1=x1, x2=x2, y1=y1, y2=y2)
    return rb if rb.valid else None


# ---------------------------
# Line artifact detection (요청)
# - ROI 안에서 특정 column(x 고정) 또는 row(y 고정)가 DN 0~100 범위를 "꾸준히" 유지하는지 탐지
# - TFT 문제/IC 접촉 문제로 발생하는 세로/가로 라인(저DN streak) 가정
# ---------------------------

def _runs_from_sorted_indices(idxs: np.ndarray) -> List[Tuple[int, int]]:
    """정렬된 정수 인덱스 배열을 (start, end) 연속 구간 리스트로 변환 (둘 다 inclusive)."""
    out: List[Tuple[int, int]] = []
    if idxs.size == 0:
        return out
    s = int(idxs[0]); e = int(idxs[0])
    for v in idxs[1:]:
        v = int(v)
        if v == e + 1:
            e = v
        else:
            out.append((s, e))
            s = v; e = v
    out.append((s, e))
    return out


def _parse_fixed_line_list(value: Any) -> List[int]:
    """고정 라인 좌표 입력(문자열/리스트)을 정수 리스트로 정규화합니다."""
    if value is None:
        return []
    items: List[Any]
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        items = list(value)
    else:
        return []

    out: List[int] = []
    for item in items:
        try:
            out.append(int(str(item).strip()))
        except Exception:
            continue
    return sorted(set(out))


def _format_fixed_line_list(value: Any) -> str:
    vals = _parse_fixed_line_list(value)
    return ",".join(str(v) for v in vals)


def detect_low_value_lines_in_roi(
    img: np.ndarray,
    roi: Optional[RoiBounds],
    *,
    dn_lo: float = 0.0,
    dn_hi: float = 100.0,
    frac_threshold: float = 0.95,
    ignore_border_px: int = 8,
    fixed_line_x: Optional[List[int]] = None,
    fixed_line_y: Optional[List[int]] = None,
    broken_line_min_low_frac: float = 0.80,
    broken_line_min_normal_frac: float = 0.10,
) -> Tuple[List[int], List[int], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """ROI 내에서 '저DN(0~100)'이 거의 전체 길이로 유지되는 column/row 라인을 탐지합니다.

    반환:
      bad_cols_x: ROI 기준이 아닌 '전체 이미지 좌표계'의 x 인덱스 리스트
      bad_rows_y: 전체 이미지 좌표계의 y 인덱스 리스트
      col_runs: bad_cols_x를 연속 구간으로 묶은 (start_x, end_x) 리스트 (inclusive)
      row_runs: bad_rows_y를 연속 구간으로 묶은 (start_y, end_y) 리스트 (inclusive)
      broken_points_xy: 끊긴 라인이 의심되는 지점의 (x, y) 좌표 리스트

    주의:
      - '꾸준히'의 해석은 (해당 축 방향 픽셀 중 dn_lo~dn_hi 비율 >= frac_threshold) 입니다.
      - ROI 가장자리의 0값(프레임/마진) 오탐을 줄이기 위해 ignore_border_px 만큼 ROI를 안쪽으로 줄입니다.
    """
    if img is None:
        return [], [], [], [], []
    try:
        if img.ndim != 2 or img.size == 0:
            return [], [], [], [], []
        H, W = int(img.shape[0]), int(img.shape[1])

        if roi is not None and getattr(roi, 'valid', False):
            x1, x2, y1, y2 = int(roi.x1), int(roi.x2), int(roi.y1), int(roi.y2)
        else:
            x1, x2, y1, y2 = 0, W, 0, H

        m = max(0, int(ignore_border_px))
        x1i = max(0, min(W, x1 + m))
        x2i = max(0, min(W, x2 - m))
        y1i = max(0, min(H, y1 + m))
        y2i = max(0, min(H, y2 - m))

        # ROI가 너무 작아지면 원래 ROI 사용
        if (x2i - x1i) < 10 or (y2i - y1i) < 10:
            x1i, x2i, y1i, y2i = x1, x2, y1, y2

        if (x2i - x1i) < 2 or (y2i - y1i) < 2:
            return [], [], [], [], []

        view = img[y1i:y2i, x1i:x2i]
        # bool mask: DN in [dn_lo, dn_hi]
        mask = (view >= float(dn_lo)) & (view <= float(dn_hi))

        # 비율 계산 (column/row)
        col_frac = mask.mean(axis=0)  # (roi_w,)
        row_frac = mask.mean(axis=1)  # (roi_h,)

        fixed_x = set(_parse_fixed_line_list(fixed_line_x))
        fixed_y = set(_parse_fixed_line_list(fixed_line_y))

        bad_cols_rel = np.where(col_frac >= float(frac_threshold))[0].astype(np.int64)
        bad_rows_rel = np.where(row_frac >= float(frac_threshold))[0].astype(np.int64)

        if fixed_x:
            bad_cols_rel = np.array([i for i in bad_cols_rel if (int(i) + int(x1i)) not in fixed_x], dtype=np.int64)
        if fixed_y:
            bad_rows_rel = np.array([i for i in bad_rows_rel if (int(i) + int(y1i)) not in fixed_y], dtype=np.int64)

        bad_cols = (bad_cols_rel + int(x1i)).tolist() if bad_cols_rel.size else []
        bad_rows = (bad_rows_rel + int(y1i)).tolist() if bad_rows_rel.size else []

        col_runs = _runs_from_sorted_indices(bad_cols_rel + int(x1i)) if bad_cols_rel.size else []
        row_runs = _runs_from_sorted_indices(bad_rows_rel + int(y1i)) if bad_rows_rel.size else []

        broken_points_xy: List[Tuple[int, int]] = []

        if float(broken_line_min_low_frac) > 0 and float(broken_line_min_normal_frac) > 0:
            low_min = float(broken_line_min_low_frac)
            normal_min = float(broken_line_min_normal_frac)

            for idx, low_frac in enumerate(col_frac):
                x = int(x1i + idx)
                if fixed_x and x in fixed_x:
                    continue
                normal_frac = 1.0 - float(low_frac)
                if float(low_frac) >= low_min and normal_frac >= normal_min:
                    normal_idx = np.where(~mask[:, idx])[0]
                    if normal_idx.size > 0:
                        y = int(y1i + normal_idx[int(normal_idx.size // 2)])
                        broken_points_xy.append((x, y))

            for idx, low_frac in enumerate(row_frac):
                y = int(y1i + idx)
                if fixed_y and y in fixed_y:
                    continue
                normal_frac = 1.0 - float(low_frac)
                if float(low_frac) >= low_min and normal_frac >= normal_min:
                    normal_idx = np.where(~mask[idx, :])[0]
                    if normal_idx.size > 0:
                        x = int(x1i + normal_idx[int(normal_idx.size // 2)])
                        broken_points_xy.append((x, y))

        return bad_cols, bad_rows, col_runs, row_runs, broken_points_xy
    except Exception:
        return [], [], [], [], []

def apply_line_noise_correction(raw: np.ndarray, roi: Optional[RoiBounds]) -> np.ndarray:
    """
    원본 코드의 apply_process() 로직:
      - target = ROI median (있으면) else full median
      - col_prof = mean(ROI rows, all cols)
      - img -= (col_prof - target)
      - row_prof = mean(all rows, ROI cols)
      - img -= (row_prof - target)
    """
    img = raw.astype(np.float32, copy=True)
    h, w = img.shape

    if roi is not None and roi.valid:
        x1, x2, y1, y2 = roi.x1, roi.x2, roi.y1, roi.y2
        target = float(np.median(img[y1:y2, x1:x2]))
    else:
        x1, x2, y1, y2 = 0, w, 0, h
        target = float(np.median(img))

    col_prof = np.mean(img[y1:y2, :], axis=0)  # shape (w,)
    img = img - (col_prof - target).reshape(1, -1)

    row_prof = np.mean(img[:, x1:x2], axis=1)  # shape (h,)
    img = img - (row_prof - target).reshape(-1, 1)

    return img


# ---------------------------
# Blemish detection - 원본 로직 유지
# ---------------------------

def compute_shape_metrics(comp_mask: np.ndarray) -> Tuple[float, float, float]:
    """
    원본 코드의 _compute_shape_metrics()
    - circularity = 4*pi*area / perim^2
    - eccentricity from covariance eigenvalues
    - perim estimated by erosion boundary count
    """
    area = int(np.sum(comp_mask))
    if area <= 0:
        return 0.0, 0.0, 0.0

    try:
        er = binary_erosion(comp_mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        boundary = comp_mask & (~er)
        perim = float(np.sum(boundary))
    except Exception:
        perim = 0.0

    if perim <= 0:
        circularity = 0.0
    else:
        circularity = float(4.0 * np.pi * area / (perim * perim))
        circularity = float(np.clip(circularity, 0.0, 1.0))

    coords = np.argwhere(comp_mask)
    if coords.shape[0] < 3:
        eccentricity = 0.0
    else:
        yy = coords[:, 0].astype(np.float64)
        xx = coords[:, 1].astype(np.float64)
        yy -= yy.mean()
        xx -= xx.mean()
        cov_xx = np.mean(xx * xx)
        cov_yy = np.mean(yy * yy)
        cov_xy = np.mean(xx * yy)
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
        try:
            eig = np.linalg.eigvalsh(cov)
            eig = np.sort(eig)[::-1]
            lam1 = float(eig[0])
            lam2 = float(eig[1]) if len(eig) > 1 else 0.0
            if lam1 <= 0:
                eccentricity = 0.0
            else:
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (lam2 / lam1))))
        except Exception:
            eccentricity = 0.0

    return circularity, float(np.clip(eccentricity, 0.0, 1.0)), perim


@dataclass
class BlemishCandidate:
    x: int
    y: int
    crop_diff_percent: float
    # Thr metric (%): blemish 경계(inside boundary)에서의 drop 값을 대표(기본: median)로 표시
    # - 설정 Threshold와 1:1 매칭 값이 아니라, "검출된 영역에서 측정된" drop 값입니다.
    drop_percent: float = 0.0
    # 참고용: 컴포넌트 내부 최대 drop(peak)
    drop_peak_percent: float = 0.0
    area_px: int = 0  # detected blemish area (pixels)
    needs_pit_check: bool = False  # L <= abs(CropΔ%) < U
    mask: Optional[np.ndarray] = None
    mask_origin: Optional[Tuple[int, int]] = None  # (x0, y0) of mask in full coords

# ---------------------------
# Background worker (Qt thread) for blemish detection
# ---------------------------

class _DetectBlemishWorker(QObject):
    finished = Signal(object)  # payload: (cand, ref_mean, scope_desc, token, cache_key)
    error = Signal(object)     # payload: (error_message, token)

    def __init__(self, img: np.ndarray, params: Dict[str, Any], roi: Optional[RoiBounds], token: int, cache_key: Any):
        super().__init__()
        self._img = img
        self._params = params
        self._roi = roi
        self._token = int(token)
        self._cache_key = cache_key

    def run(self) -> None:
        try:
            cand, ref_mean, scope_desc = detect_blemish_candidates(self._img, self._params, self._roi)
            self.finished.emit((cand, ref_mean, scope_desc, self._token, self._cache_key))
        except Exception as e:
            self.error.emit((str(e), self._token))





def extract_crop(img: np.ndarray, x: int, y: int, size: int = 50) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img.shape
    x0 = int(round(x) - size // 2)
    y0 = int(round(y) - size // 2)
    x1 = x0 + size
    y1 = y0 + size

    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(w, x1)
    y1c = min(h, y1)
    return img[y0c:y1c, x0c:x1c], (x0c, y0c)


def detect_blemish_candidates(
    img: np.ndarray,
    params: Dict[str, Any],
    roi: Optional[RoiBounds],
) -> Tuple[List[BlemishCandidate], float, str]:
    """
    반환:
      - candidates: crop_diff_percent(작을수록 더 'drop' 큰 방향으로 정렬되는 원본 코드 스타일) 기준 정렬
      - ref_mean: ROI mean (옵션) 또는 full mean
      - scope_desc: "ROI" or "Full"
    """
    p = params
    h, w = img.shape

    # analysis scope
    if p.get("analysis_scope") == "ROI" and roi is not None and roi.valid:
        x1, x2, y1, y2 = roi.x1, roi.x2, roi.y1, roi.y2
        work = img[y1:y2, x1:x2]
        off_x, off_y = x1, y1
        scope_desc = "ROI"
    else:
        work = img
        off_x, off_y = 0, 0
        scope_desc = "Full"

    # reference mean
    if p.get("use_roi_as_reference") and roi is not None and roi.valid:
        ref_mean = float(np.mean(img[roi.y1:roi.y2, roi.x1:roi.x2]))
    else:
        ref_mean = float(np.mean(img))

    # core algorithm
    mw = int(p["measure_window"])
    if mw <= 1:
        # uniform_filter(size=1)은 항등(결과 동일)이라 계산을 생략하여 속도를 개선합니다.
        w_m = work
    else:
        w_m = uniform_filter(work, mw)
    bg = uniform_filter(w_m, int(p["bg_window"]))
    drop = np.abs(1 - w_m / (bg + 1e-6)) * 100.0

    mask = (drop >= float(p["threshold_crop_percent"])) & (work > int(p["min_dn"]))

    m = int(p["border_margin"])
    if m > 0:
        mask[:m, :] = 0
        mask[-m:, :] = 0
        mask[:, :m] = 0
        mask[:, -m:] = 0

    lbl, num = nd_label(mask)
    candidates: List[BlemishCandidate] = []

    if num > 0:
        objs = find_objects(lbl)
        for i in range(num):
            sl = objs[i]
            if sl is None:
                continue

            comp = (lbl[sl] == (i + 1))
            area = int(np.sum(comp))
            if not (int(p["min_area"]) <= area <= int(p["max_area"])):
                continue

            circ, ecc, _perim = compute_shape_metrics(comp)
            if circ < float(p["min_circularity"]):
                continue
            if ecc > float(p["max_eccentricity"]):
                continue

            c_drop = drop[sl]
            rr, cc = np.unravel_index(np.argmax(c_drop * comp), c_drop.shape)

            yc, xc = sl[0].start + rr, sl[1].start + cc


            # Thr(%) 표시용 drop 값 계산

            # - peak: 컴포넌트 내부 최대 drop

            # - boundary: 경계(inside boundary)에서의 drop 대표값(기본: median)

            d_peak = float(c_drop[rr, cc])

            d_thr = d_peak

            try:

                er = binary_erosion(comp, structure=np.ones((3, 3), dtype=bool), border_value=0)

                boundary = comp & (~er)

                bd = c_drop[boundary]

                if bd.size:

                    d_thr = float(np.median(bd))

            except Exception:

                d_thr = d_peak
            # local crop diff (원본 코드 유지)
            cx0, cy0 = max(0, xc - 25), max(0, yc - 25)
            cx1, cy1 = min(work.shape[1], xc + 25), min(work.shape[0], yc + 25)
            crop_local = work[cy0:cy1, cx0:cx1]
            bg_mean = float(np.mean(crop_local)) if crop_local.size else 0.0
            val_pix = float(work[yc, xc])
            d_crp = (val_pix - bg_mean) / (bg_mean + 1e-6) * 100.0

            # 밝거나 어두운 blemish 모두 탐지: CropΔ threshold band (abs 기준)
            # - abs(CropΔ%) < L : 제외
            # - L <= abs(CropΔ%) < U : list 표시 + 확인 팝업
            # - abs(CropΔ%) >= U : list 표시 (팝업 없음)
            try:
                cropdiff_u = float(p.get("threshold_ref_percent_u", p.get("threshold_ref_percent", 0.0)))
            except Exception:
                cropdiff_u = float(p.get("threshold_ref_percent", 0.0) or 0.0)
            try:
                cropdiff_l = float(p.get("threshold_ref_percent_l", cropdiff_u))
            except Exception:
                cropdiff_l = cropdiff_u
            # 안전장치: U < L이면 swap
            if cropdiff_u < cropdiff_l:
                cropdiff_u, cropdiff_l = cropdiff_l, cropdiff_u

            abs_crp = float(np.abs(d_crp))
            if abs_crp < cropdiff_l:
                continue
            needs_pit_check = (cropdiff_u > cropdiff_l) and (cropdiff_l <= abs_crp < cropdiff_u)

            candidates.append(
                BlemishCandidate(
                    x=int(off_x + xc),
                    y=int(off_y + yc),
                    crop_diff_percent=float(d_crp),
                    drop_percent=float(d_thr),
                    drop_peak_percent=float(d_peak),
                    area_px=int(area),
                    needs_pit_check=bool(needs_pit_check),
                    mask=comp.astype(np.uint8),
                    mask_origin=(int(off_x + sl[1].start), int(off_y + sl[0].start)),
                )
            )

    # 밝거나 어두운 blemish 모두: |crop_diff_percent| 큰 순서로 정렬
    candidates.sort(key=lambda c: abs(float(c.crop_diff_percent)), reverse=True)
    max_det = int(p.get("max_detections", 20))
    return candidates[:max_det], ref_mean, scope_desc


# ---------------------------
# Parameter dialog (Qt)
# ---------------------------

class BlemishParamDialog(QWidget):
    """원본의 Toplevel 파라미터 창을 Qt로 구현(적용 버튼)"""

    applied = Signal(dict)

    def __init__(self, params: Dict[str, Any], model: str = "", typ: str = "", saved_at: Optional[str] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Blemish Param")
        self._params = dict(params)

        layout = QVBoxLayout(self)

        # profile info (model/type + saved date)
        self.lbl_profile = QLabel()
        self.lbl_profile.setWordWrap(True)
        self.lbl_profile.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.lbl_profile)
        self.set_profile_info(model, typ, saved_at)

        form = QFormLayout()
        layout.addLayout(form)

        # widgets
        self.cb_scope = QComboBox()
        self.cb_scope.addItems(["ROI", "Full"])
        self.cb_scope.setCurrentText(str(self._params.get("analysis_scope", "ROI")))

        self.chk_use_ref = QCheckBox("Use ROI Mean Ref")
        self.chk_use_ref.setChecked(bool(self._params.get("use_roi_as_reference", True)))

        self.sp_bg = QSpinBox(); self.sp_bg.setRange(1, 9999); self.sp_bg.setValue(int(self._params.get("bg_window", 51)))
        self.sp_meas = QSpinBox(); self.sp_meas.setRange(1, 9999); self.sp_meas.setValue(int(self._params.get("measure_window", 1)))

        self.db_ref_thr_u = QDoubleSpinBox(); self.db_ref_thr_u.setRange(0, 100); self.db_ref_thr_u.setDecimals(3); self.db_ref_thr_u.setValue(float(self._params.get("threshold_ref_percent_u", self._params.get("threshold_ref_percent", 2.0))))

        self.db_ref_thr_l = QDoubleSpinBox(); self.db_ref_thr_l.setRange(0, 100); self.db_ref_thr_l.setDecimals(3); self.db_ref_thr_l.setValue(float(self._params.get("threshold_ref_percent_l", self._params.get("threshold_ref_percent_u", self._params.get("threshold_ref_percent", 2.0)))))
        self.db_crop_thr = QDoubleSpinBox(); self.db_crop_thr.setRange(0, 100); self.db_crop_thr.setDecimals(3); self.db_crop_thr.setValue(float(self._params.get("threshold_crop_percent", 2.0)))

        self.sp_min_area = QSpinBox(); self.sp_min_area.setRange(0, 1_000_000); self.sp_min_area.setValue(int(self._params.get("min_area", 5)))
        self.sp_max_area = QSpinBox(); self.sp_max_area.setRange(0, 1_000_000); self.sp_max_area.setValue(int(self._params.get("max_area", 5000)))

        self.sp_margin = QSpinBox(); self.sp_margin.setRange(0, 10_000); self.sp_margin.setValue(int(self._params.get("border_margin", 30)))
        self.sp_overlay_margin = QSpinBox(); self.sp_overlay_margin.setRange(0, 300); self.sp_overlay_margin.setValue(int(self._params.get("overlay_bottom_margin_px", 10)))
        self.sp_max_det = QSpinBox(); self.sp_max_det.setRange(1, 10_000); self.sp_max_det.setValue(int(self._params.get("max_detections", 20)))

        self.sp_min_dn = QSpinBox(); self.sp_min_dn.setRange(0, 65535); self.sp_min_dn.setValue(int(self._params.get("min_dn", 1)))

        self.db_min_circ = QDoubleSpinBox(); self.db_min_circ.setRange(0, 1); self.db_min_circ.setDecimals(4); self.db_min_circ.setValue(float(self._params.get("min_circularity", 0.0)))
        self.db_max_ecc = QDoubleSpinBox(); self.db_max_ecc.setRange(0, 1); self.db_max_ecc.setDecimals(4); self.db_max_ecc.setValue(float(self._params.get("max_eccentricity", 1.0)))

        self.le_fixed_line_x = QLineEdit(); self.le_fixed_line_x.setPlaceholderText("ex) 235,2304")
        self.le_fixed_line_x.setText(_format_fixed_line_list(self._params.get("fixed_line_x", "")))
        self.le_fixed_line_y = QLineEdit(); self.le_fixed_line_y.setPlaceholderText("ex) 1304,3043")
        self.le_fixed_line_y.setText(_format_fixed_line_list(self._params.get("fixed_line_y", "")))

        # form rows
        form.addRow("Scope", self.cb_scope)
        form.addRow("", self.chk_use_ref)
        form.addRow("BG Window", self.sp_bg)
        form.addRow("Meas Window", self.sp_meas)
        form.addRow("CropΔ Thr.U(%)", self.db_ref_thr_u)
        form.addRow("CropΔ Thr.L(%)", self.db_ref_thr_l)
        form.addRow("Crop Thr(%)", self.db_crop_thr)
        form.addRow("Min Area", self.sp_min_area)
        form.addRow("Max Area", self.sp_max_area)
        form.addRow("Margin", self.sp_margin)
        form.addRow("Label Offset(px)", self.sp_overlay_margin)
        form.addRow("Max Det", self.sp_max_det)
        form.addRow("Min DN", self.sp_min_dn)
        form.addRow("Min Circ", self.db_min_circ)
        form.addRow("Max Ecc", self.db_max_ecc)
        form.addRow("Fixed Line X", self.le_fixed_line_x)
        form.addRow("Fixed Line Y", self.le_fixed_line_y)

        btns = QHBoxLayout()
        layout.addLayout(btns)
        btn_apply = QPushButton("Apply")
        btn_close = QPushButton("Close")
        btns.addWidget(btn_apply)
        btns.addWidget(btn_close)
        btns.addStretch(1)

        btn_apply.clicked.connect(self._on_apply)
        btn_close.clicked.connect(self.close)


    def set_profile_info(self, model: str, typ: str, saved_at: Optional[str]) -> None:
        """Param UI 상단에 현재 프로필(model/type)과 마지막 저장 시간을 표시합니다."""
        def _fmt(ts: Optional[str]) -> str:
            if not ts:
                return "Not saved yet"
            try:
                dt = datetime.fromisoformat(str(ts))
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return str(ts)

        try:
            m = str(model or "").strip()
            t = str(typ or "").strip()
            head = f"Profile: {m} / {t}"
            tail = f"Saved: {_fmt(saved_at)}"
            self.lbl_profile.setText(f"{head}   |   {tail}")
        except Exception:
            pass


    def _on_apply(self) -> None:
        p = {
            "analysis_scope": self.cb_scope.currentText(),
            "use_roi_as_reference": self.chk_use_ref.isChecked(),
            "bg_window": int(self.sp_bg.value()),
            "measure_window": int(self.sp_meas.value()),
            "threshold_ref_percent_u": float(self.db_ref_thr_u.value()),
            "threshold_ref_percent_l": float(self.db_ref_thr_l.value()),
            # legacy 호환: threshold_ref_percent는 U와 동일
            "threshold_ref_percent": float(self.db_ref_thr_u.value()),
            "threshold_crop_percent": float(self.db_crop_thr.value()),
            "min_area": int(self.sp_min_area.value()),
            "max_area": int(self.sp_max_area.value()),
            "border_margin": int(self.sp_margin.value()),
            "overlay_bottom_margin_px": int(self.sp_overlay_margin.value()),
            "max_detections": int(self.sp_max_det.value()),
            "min_dn": int(self.sp_min_dn.value()),
            "min_circularity": float(self.db_min_circ.value()),
            "max_eccentricity": float(self.db_max_ecc.value()),
            "fixed_line_x": _parse_fixed_line_list(self.le_fixed_line_x.text()),
            "fixed_line_y": _parse_fixed_line_list(self.le_fixed_line_y.text()),
        }
        self.applied.emit(p)


# ---------------------------
# Main widget: two viewers + right panel (blemish list/crop)
# ---------------------------



class _QtPanLwFilter(QObject):
    """Qt 레벨에서 마우스 제스처를 강제로 보장하는 이벤트 필터.

    목표:
    - 좌클릭 드래그(Shift 없음): Pan
    - 우클릭 드래그(Shift 없음): L/W 조절
    - 더블클릭(좌): Auto L/W
    - Shift+좌드래그: ROI (napari 쪽 콜백에 위임)
    - Shift+우클릭: ROI 해제 등 기존 매핑 위임
    """

    def __init__(self, owner: "XrayNapariWidget", which: str):
        super().__init__(owner)
        self._owner = owner
        self._which = which
        self._drag_right = False
        self._drag_left = False
        self._start_pos = None
        self._start_clim = None
        self._start_center = None
        self._start_zoom = 1.0

    def eventFilter(self, obj, event):  # noqa: N802
        try:
            et = event.type()
        except Exception:
            return False

        # 더블클릭: Auto L/W (Shift 없이 좌)
        if et == QEvent.Type.MouseButtonDblClick:
            try:
                if event.button() == Qt.MouseButton.LeftButton and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                    self._owner._auto_lw_for_view(self._which)
                    return True
            except Exception:
                return False
            return False

        # 마우스 프레스: Pan 시작(좌) / L/W 시작(우)
        if et == QEvent.Type.MouseButtonPress:
            try:
                mods = event.modifiers()
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    # Shift 제스처는 napari 콜백에 위임
                    return False

                if event.button() == Qt.MouseButton.RightButton:
                    layer = self._owner._layer_for_view(self._which)
                    if layer is None:
                        return False
                    try:
                        cmin, cmax = layer.contrast_limits
                        self._start_clim = (float(cmin), float(cmax))
                    except Exception:
                        return False
                    self._start_pos = event.pos()
                    self._drag_right = True
                    return True

                if event.button() == Qt.MouseButton.LeftButton:
                    cam = self._owner._camera_for_view(self._which)
                    if cam is None:
                        return False
                    self._start_pos = event.pos()
                    try:
                        self._start_center = tuple(cam.center)
                    except Exception:
                        self._start_center = None
                    try:
                        self._start_zoom = float(getattr(cam, "zoom", 1.0) or 1.0)
                    except Exception:
                        self._start_zoom = 1.0
                    self._drag_left = True
                    return True
            except Exception:
                return False
            return False

        # 마우스 이동: Pan / L/W 업데이트
        if et == QEvent.Type.MouseMove:
            try:
                mods = event.modifiers()
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    return False

                if self._drag_right and self._start_pos is not None and self._start_clim is not None:
                    dx = float(event.pos().x() - self._start_pos.x())
                    dy = float(event.pos().y() - self._start_pos.y())

                    cmin, cmax = self._start_clim
                    level0 = (cmin + cmax) / 2.0
                    width0 = max(1e-6, (cmax - cmin))
                    scale_w = width0 / 300.0
                    scale_l = width0 / 300.0

                    width = max(1e-6, width0 + dx * scale_w)
                    level = level0 - dy * scale_l  # 위로 드래그 -> 밝게

                    new_min = level - width / 2.0
                    new_max = level + width / 2.0
                    self._owner._set_view_contrast_limits(self._which, (new_min, new_max))
                    return True

                if self._drag_left and self._start_pos is not None and self._start_center is not None:
                    cam = self._owner._camera_for_view(self._which)
                    if cam is None:
                        return False
                    dx = float(event.pos().x() - self._start_pos.x())
                    dy = float(event.pos().y() - self._start_pos.y())

                    # napari camera.center는 (y,x) 또는 (z,y,x) 등일 수 있어 뒤에서부터 적용
                    try:
                        z = float(getattr(cam, "zoom", None) or self._start_zoom or 1.0)
                    except Exception:
                        z = 1.0
                    z = max(1e-6, z)

                    cen = list(self._start_center)
                    if len(cen) >= 2:
                        cen[-1] = cen[-1] - dx / z  # x
                        cen[-2] = cen[-2] - dy / z  # y
                        cam.center = tuple(cen)
                        return True
            except Exception:
                return False
            return False

        # 릴리즈: 드래그 종료
        if et == QEvent.Type.MouseButtonRelease:
            try:
                if self._drag_right and event.button() == Qt.MouseButton.RightButton:
                    self._drag_right = False
                    return True
                if self._drag_left and event.button() == Qt.MouseButton.LeftButton:
                    self._drag_left = False
                    return True
            except Exception:
                return False
            return False

        return False
class XrayNapariWidget(QSplitter):
    """
    메인 레이아웃:
      [좌] Raw viewer
      [중] Result viewer (synced camera)
      [우] Control/blemish panel (table + crop viewer)
    """

    def __init__(self, host_viewer: napari.Viewer) -> None:
        super().__init__()
        self.setOrientation(Qt.Orientation.Horizontal)

        self.host_viewer = host_viewer  # napari 기본 viewer (dock 위젯 호스트용)

        # ---------------------------
        # data state
        # ---------------------------
        self.raw_data: Optional[np.ndarray] = None
        self.processed_data: Optional[np.ndarray] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.loaded_file_path: Optional[str] = None
        self.blemish_save_dir: Optional[str] = None

        # ---------------------------
        # background detect (QThread) + result cache
        # ---------------------------
        self._detect_token: int = 0
        self._detect_running: bool = False
        self._detect_queue: Optional[Tuple[str, np.ndarray, Dict[str, Any], Optional[RoiBounds], Any]] = None
        self._detect_cache: "OrderedDict[Any, Tuple[List[BlemishCandidate], float, str]]" = OrderedDict()
        self._detect_cache_max: int = 4
        self._detect_thread: Optional[QThread] = None
        self._detect_worker: Optional[_DetectBlemishWorker] = None
        self._detect_cursor_set: bool = False


        # 현재 선택/로드된 모델 (Result red box 및 orientation 등에 사용)
        self.current_model: Optional[str] = None

        # ROI
        self.roi_bounds: Optional[RoiBounds] = None

        # ROI: Shift+좌클릭 드래그로 1개만 그리도록 제어
        self._roi_internal_update = False

        # Blemish
        self.config_file = "blemish_config.json"
        self.blemish_config_store: Dict[str, Any] = self._load_config_store()
        self.blemish_params: Dict[str, Any] = default_blemish_params()
        self.blemish_profile_key: Optional[str] = None
        self.blemish_profile_saved_at: Optional[str] = None
        self.blemish_last_img_full: Optional[np.ndarray] = None
        self.blemish_last_candidates: List[BlemishCandidate] = []
        self.blemish_last_ref_mean: Optional[float] = None

        # pixel pitch (cm/pixel): legacy (area now displayed in px^2)
        self.pixel_pitch_cm: Optional[float] = None

        # ---------------------------
        # viewer models (2-pane + crop)
        # ---------------------------
        self.model_raw = ViewerModel(title="Raw")
        self.model_result = ViewerModel(title="Result")
        self.model_crop = ViewerModel(title="Crop")

        self.qt_raw = QtViewer(self.model_raw)
        self.qt_result = QtViewer(self.model_result)
        self.qt_crop = QtViewer(self.model_crop)


        # 각 QtViewer 내부의 기본 좌측 패널(레이어 리스트/컨트롤 등)을 숨김
        for _qt in (self.qt_raw, self.qt_result, self.qt_crop):
            self._strip_qtviewer_ui(_qt)
        # Qt 레벨 이벤트 필터: Pan / L/W / Auto L/W를 플랫폼 차이 없이 보장
        self._install_qt_mouse_filters()
        # raw/result은 좌우로, 오른쪽 패널은 별도 QWidget
        self.addWidget(self.qt_raw)
        self.addWidget(self.qt_result)

        self.right_panel = self._build_right_panel()
        # 표/라벨이 가로로 잘리지 않게 최소 폭 확보
        try:
            self.right_panel.setMinimumWidth(520)
        except Exception:
            pass

        self.addWidget(self.right_panel)

        # load blemish params profile for current model/type selection
        self._load_profile_params_for_current_selection()

        # 초기 splitter 비율
        self.setStretchFactor(0, 6)
        self.setStretchFactor(1, 6)
        self.setStretchFactor(2, 3)

        # ---------------------------
        # layers
        # ---------------------------
        self.layer_raw: Optional[NapariImageLayer] = None
        self.layer_result: Optional[NapariImageLayer] = None
        self.layer_crop: Optional[NapariImageLayer] = None
        self.layer_roi: Optional[NapariShapesLayer] = None
        self.layer_blemish_points: Optional[Any] = None  # napari Points layer
        self.layer_blemish_boxes: Optional[Any] = None   # napari Shapes layer (rectangle boxes)

        # ---------------------------
        # camera sync
        # ---------------------------
        self._camera_block = False
        # 첫 로딩 시 뷰를 화면에 꽉 차게 맞추기 위한 플래그
        self._fit_after_load = False
        self._install_camera_sync()

        # ---------------------------
        # cursor/status throttling
        # ---------------------------
        self._last_status_t = 0.0
        self._status_interval = 1.0 / 30.0  # 30Hz
        self._install_status_update()
        self._install_mouse_controls()


    def apply_default_splitter_sizes(self) -> None:
        """초기 레이아웃: 우측 패널을 더 줄이고, Raw/Result를 더 크게."""
        try:
            total = max(1, self.width())
            # [Raw, Result, Right] 비율: 45% / 45% / 10%
            self.setSizes([int(total*0.45), int(total*0.45), int(total*0.10)])
        except Exception:
            pass

    # ---------------------------
    # Config I/O
    # ---------------------------

    def _profile_key(self, model: str, typ: str) -> str:
        return f"{model}|{typ}"

    def _now_kst_iso(self) -> str:
        try:
            tz = timezone(timedelta(hours=9))
            return datetime.now(tz).isoformat(timespec="seconds")
        except Exception:
            return datetime.now().isoformat(timespec="seconds")

    def _normalize_cropdelta_thresholds(self, p: Dict[str, Any]) -> None:
        """CropΔ threshold(U/L) 키를 정규화(레거시 포함)합니다.

        - 레거시(threshold_ref_percent)만 있는 경우: U=L=threshold_ref_percent
        - U/L이 있으면 U>=L이 되도록 swap
        - threshold_ref_percent는 항상 U로 mirror
        """
        if not isinstance(p, dict):
            return

        legacy = p.get("threshold_ref_percent", None)

        try:
            u = float(p.get("threshold_ref_percent_u", legacy if legacy is not None else 0.0))
        except Exception:
            try:
                u = float(legacy or 0.0)
            except Exception:
                u = 0.0

        try:
            l = float(p.get("threshold_ref_percent_l", u))
        except Exception:
            l = u

        if u < l:
            u, l = l, u

        p["threshold_ref_percent_u"] = float(u)
        p["threshold_ref_percent_l"] = float(l)
        # legacy mirror
        p["threshold_ref_percent"] = float(u)

    def _load_config_store(self) -> Dict[str, Any]:
        """Blemish 파라미터를 model/type 별로 저장하는 config를 로드합니다.

        파일 형식:
          {
            "version": 2,
            "profiles": {
              "4343|Probe": {"saved_at": "...", "params": {...}},
              ...
            },
            "last_profile": "4343|Probe"
          }

        레거시(단일 dict params) 형식이면 "*|*" 프로필로 이관합니다.
        """
        store: Dict[str, Any] = {"version": 2, "profiles": {}, "last_profile": None}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict) and isinstance(saved.get("profiles"), dict):
                    store.update(saved)
                elif isinstance(saved, dict):
                    # legacy single-param dict
                    store["profiles"]["*|*"] = {"saved_at": None, "params": saved}
            except Exception:
                pass
        if not isinstance(store.get("profiles"), dict):
            store["profiles"] = {}
        return store

    def _save_config_store(self) -> None:
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.blemish_config_store, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    def _get_profile_entry(self, model: str, typ: str) -> Tuple[Dict[str, Any], Optional[str]]:
        profiles = {}
        try:
            profiles = self.blemish_config_store.get("profiles", {})
        except Exception:
            profiles = {}
        if not isinstance(profiles, dict):
            profiles = {}

        key = self._profile_key(model, typ)
        entry = profiles.get(key)

        # fallback: legacy/default
        if not isinstance(entry, dict):
            entry = profiles.get("*|*")

        if isinstance(entry, dict):
            params = entry.get("params")
            saved_at = entry.get("saved_at")
            if isinstance(params, dict):
                return dict(params), (str(saved_at) if saved_at else None)

        return {}, None

    def _load_profile_params_for_current_selection(self) -> None:
        """현재 콤보박스(model/type)에 해당하는 blemish param을 로드합니다."""
        try:
            model = self.cb_model.currentText()
            typ = self.cb_type.currentText()
        except Exception:
            return
        if not model or not typ:
            return

        params, saved_at = self._get_profile_entry(model, typ)
        p = default_blemish_params()
        try:
            p.update(params)
        except Exception:
            pass
        self._normalize_cropdelta_thresholds(p)
        self.blemish_params = p
        self.blemish_profile_key = self._profile_key(model, typ)
        self.blemish_profile_saved_at = saved_at

    def _save_profile_params_for_current_selection(self) -> Optional[str]:
        """현재 콤보박스(model/type)에 blemish param을 저장하고 saved_at을 갱신합니다."""
        try:
            model = self.cb_model.currentText()
            typ = self.cb_type.currentText()
        except Exception:
            return None
        if not model or not typ:
            return None

        key = self._profile_key(model, typ)
        saved_at = self._now_kst_iso()

        # normalize CropΔ threshold (U/L) before saving
        self._normalize_cropdelta_thresholds(self.blemish_params)

        try:
            if not isinstance(self.blemish_config_store.get("profiles"), dict):
                self.blemish_config_store["profiles"] = {}
            self.blemish_config_store["profiles"][key] = {
                "saved_at": saved_at,
                "params": dict(self.blemish_params),
            }
            self.blemish_config_store["last_profile"] = key
        except Exception:
            return None

        self.blemish_profile_key = key
        self.blemish_profile_saved_at = saved_at
        self._save_config_store()
        return saved_at



    def _set_status(self, msg: str) -> None:
        """패널 내부 상태 라벨 + (가능하면) host_viewer.status 동시 업데이트"""
        try:
            if hasattr(self, 'lbl_status') and self.lbl_status is not None:
                self.lbl_status.setText(msg)
        except Exception:
            pass

    
    
    def _update_blemish_text_overlay(self, clear_only: bool = False) -> None:
        """Result(=Corrected) 뷰에 blemish 텍스트 라벨을 '실제 위치 근처'에 표시.

        v4_13까지는 napari Points.text 기반으로 구현했지만, 환경/버전 차이로
        텍스트가 아예 렌더링되지 않는 경우가 있어(사용자 보고) 더 확실한 방식으로 변경했습니다.

        구현:
        - Result 뷰에 "Blemish Text Overlay" RGBA 이미지 레이어를 추가
        - PIL로 투명 캔버스에 검정 그림자 + 노란 텍스트를 직접 래스터화
        - 각 blemish의 (x,y) 위치에서 아래로 offset(px)만큼 이동한 곳에 표기
        """

        # 기존 오버레이 레이어 정리
        _remove_layers_by_prefix(self.model_result, "Blemish Text Overlay")

        if clear_only:
            return

        if self.processed_data is None:
            return

        cand = list(self.blemish_last_candidates) if self.blemish_last_candidates else []
        if not cand:
            return

        H, W = int(self.processed_data.shape[0]), int(self.processed_data.shape[1])

        try:
            offset = int(self.blemish_params.get("overlay_bottom_margin_px", 5))
        except Exception:
            offset = 5

        try:
            tsize = int(self.blemish_params.get("overlay_text_size", 18))
        except Exception:
            tsize = 18

        # 투명 RGBA 캔버스
        try:
            canvas = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
            draw = PILImageDraw.Draw(canvas)

            # 폰트: DejaVuSans가 있으면 size 적용, 없으면 기본
            font = None
            try:
                font = PILImageFont.truetype("DejaVuSans.ttf", size=int(tsize))
            except Exception:
                try:
                    font = PILImageFont.load_default()
                except Exception:
                    font = None

            for c in cand:
                try:
                    x = int(round(float(c.x)))
                    y = int(round(float(c.y)))
                    thr_val = float(getattr(c, "drop_percent", 0.0) or 0.0)
                    txt = f"({int(c.x)},{int(c.y)})  Thr {thr_val:.2f}%"
                except Exception:
                    continue

                ty = y + int(offset)
                # 화면 밖으로 나가지 않게 클램프
                tx = max(0, min(W - 1, x))
                ty = max(0, min(H - 1, ty))

                # 그림자(검정) -> 본문(노랑)
                try:
                    draw.text((tx + 1, ty + 1), txt, fill=(0, 0, 0, 255), font=font)
                except Exception:
                    draw.text((tx + 1, ty + 1), txt, fill=(0, 0, 0, 255))
                try:
                    draw.text((tx, ty), txt, fill=(255, 255, 0, 255), font=font)
                except Exception:
                    draw.text((tx, ty), txt, fill=(255, 255, 0, 255))

            rgba = np.asarray(canvas, dtype=np.uint8)

            # napari에 overlay 레이어 추가
            try:
                lay = self.model_result.add_image(
                    rgba,
                    name="Blemish Text Overlay",
                    rgb=True,
                    blending="additive",
                    opacity=1.0,
                )
            except Exception:
                lay = self.model_result.add_image(
                    rgba,
                    name="Blemish Text Overlay",
                    rgb=True,
                    opacity=1.0,
                )
            try:
                lay.interactive = False
            except Exception:
                pass

        except Exception:
            # 어떤 이유로든 실패하면(폰트/캔버스 등) 조용히 넘어가되, 기존 흐름은 유지
            return
    
    # ---------------------------
    # UI build (right panel)
    # ---------------------------


    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(6, 6, 6, 6)

        # ---- Load group (model/type)
        gb_load = QGroupBox("1. Load")
        gl = QVBoxLayout(gb_load)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Model"))
        self.cb_model = QComboBox()
        self.cb_model.addItems(list(MODEL_DIMENSIONS.keys()))
        row1.addWidget(self.cb_model)

        row1.addWidget(QLabel("Type"))
        self.cb_type = QComboBox()
        row1.addWidget(self.cb_type)
        gl.addLayout(row1)

        self.btn_load = QPushButton("Load Raw (.IMG/.raw)")
        gl.addWidget(self.btn_load)

        self.cb_model.currentTextChanged.connect(self._on_model_changed)
        self.cb_type.currentTextChanged.connect(self._on_type_changed)
        self.btn_load.clicked.connect(self._on_load_clicked)

        # init type list
        self._on_model_changed(self.cb_model.currentText())

        # ---- ROI UI 제거 (요청)
        # ROI는 Raw 뷰에서 Shift+좌클릭 드래그로 그립니다.

        # ---- Blemish group (Process UI 제거 -> 이 영역을 더 크게)
        gb_blm = QGroupBox("2. Blemish")
        bl = QVBoxLayout(gb_blm)

        rowt = QHBoxLayout()
        rowt.addWidget(QLabel("Detect Target"))
        self.cb_target = QComboBox()
        self.cb_target.addItems(["Raw", "Corrected"])
        rowt.addWidget(self.cb_target)
        bl.addLayout(rowt)

        rowb = QHBoxLayout()
        self.btn_detect = QPushButton("Detect")
        self.btn_save_crop = QPushButton("Save Crop")
        self.btn_param = QPushButton("Param")
        rowb.addWidget(self.btn_detect)
        rowb.addWidget(self.btn_save_crop)
        rowb.addWidget(self.btn_param)
        bl.addLayout(rowb)

        self.btn_detect.clicked.connect(self._on_detect)
        self.btn_save_crop.clicked.connect(self._on_save_crop)
        self.btn_param.clicked.connect(self._on_param)

        # ---- table (요청: (X,Y)/Thr(%)/CropΔ%/Area(px^2))
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["(X,Y)", "Thr(%)", "CropΔ%", "Area(px^2)"])
        self.tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.tbl.cellClicked.connect(self._on_table_clicked)
        self.tbl.setMinimumHeight(280)
        # 가로 잘림 방지(패널 폭이 좁아질 때 최소 폭 확보)
        try:
            self.tbl.setMinimumWidth(460)
        except Exception:
            pass

        try:
            hdr = self.tbl.horizontalHeader()
            hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            hdr.resizeSection(0, 120)
            hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
            hdr.resizeSection(1, 80)
            hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
            hdr.resizeSection(2, 90)
            hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        except Exception:
            pass

        bl.addWidget(self.tbl, stretch=1)

        # ---- crop viewer
        gb_crop = QGroupBox("3. Crop Preview")
        cr = QVBoxLayout(gb_crop)
        cr.addWidget(self.qt_crop)
        self.qt_crop.setMinimumHeight(220)

        # ---- Save result
        gb_save = QGroupBox("4. Save")
        sv = QVBoxLayout(gb_save)
        self.btn_save_result = QPushButton("Save Result (.IMG, uint16)")
        sv.addWidget(self.btn_save_result)
        self.btn_save_result.clicked.connect(self._on_save_result)

        # ---- Status (메인 napari 창을 숨겨도 보이도록 패널 내부에 표시)
        self.lbl_status = QLabel("Ready | ROI: Shift+좌클릭 드래그 / Clear: Shift+우클릭")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.lbl_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # ---- assemble with stretch: Blemish가 남는 높이를 최대한 가져가도록
        outer.addWidget(gb_load, 0)
        outer.addWidget(gb_blm, 1)
        outer.addWidget(gb_crop, 0)
        outer.addWidget(gb_save, 0)
        outer.addWidget(self.lbl_status, 0)

        return w

    # ---------------------------
    # Model/type combobox logic
    # ---------------------------

    def _on_model_changed(self, model: str) -> None:
        self.cb_type.blockSignals(True)
        self.cb_type.clear()
        self.cb_type.addItems(list(MODEL_DIMENSIONS.get(model, {}).keys()))
        self.cb_type.blockSignals(False)
        # select first
        if self.cb_type.count():
            self.cb_type.setCurrentIndex(0)
        self._on_type_changed(self.cb_type.currentText())

    def _on_type_changed(self, typ: str) -> None:
        model = self.cb_model.currentText()
        w, h = MODEL_DIMENSIONS.get(model, {}).get(typ, (None, None))
        ok = (w is not None) and (h is not None)
        self.btn_load.setEnabled(bool(ok))

        # model/type 변경 시 해당 프로필 파라미터 자동 로드
        try:
            self._load_profile_params_for_current_selection()
        except Exception:
            pass

    # ---------------------------
    # Viewer helpers
    # ---------------------------

    def _ensure_layers(self) -> None:
        """raw/result/crop/roi 레이어가 없으면 생성"""
        if self.layer_raw is None:
            self.layer_raw = add_image_to_model(self.model_raw,
                np.zeros((10, 10), np.float32),
                name="Raw",
                colormap="gray_r",
                contrast_limits=(0, 1),
                interpolation="nearest",
            )
        if self.layer_result is None:
            self.layer_result = add_image_to_model(self.model_result,
                np.zeros((10, 10), np.float32),
                name="Result",
                colormap="gray_r",
                contrast_limits=(0, 1),
                interpolation="nearest",
            )
        if self.layer_crop is None:
            self.layer_crop = add_image_to_model(self.model_crop,
                np.zeros((10, 10), np.float32),
                name="Crop",
                colormap="gray_r",
                contrast_limits=(0, 1),
                interpolation="nearest",
            )
        if self.layer_roi is None:
            # ROI는 raw viewer에 shapes로 생성
            self.layer_roi = add_shapes_to_model(self.model_raw,
                name="ROI",
                shape_type="rectangle",
                edge_color="yellow",
                face_color=[0, 0, 0, 0],  # transparent
                edge_width=2,
            )
            self.layer_roi.mode = "select"
            # 기본 편집 기능은 끄고(드래그/핸들 조작 등), Shift+드래그로만 갱신
            try:
                self.layer_roi.editable = False
            except Exception:
                pass
            # ROI 레이어 자체를 비대화형으로 만들어(선택/이동) Raw 좌클릭 드래그가 Pan으로 동작하도록 함
            try:
                self.layer_roi.interactive = False
            except Exception:
                pass
            self.layer_roi.events.data.connect(self._on_roi_changed)

    def _layer_for_view(self, which: str):
        self._ensure_layers()
        if which == "raw":
            return self.layer_raw
        if which == "result":
            return self.layer_result
        if which == "crop":
            return self.layer_crop
        return None

    def _camera_for_view(self, which: str):
        try:
            if which == "raw":
                return self.model_raw.camera
            if which == "result":
                return self.model_result.camera
            if which == "crop":
                return self.model_crop.camera
        except Exception:
            return None
        return None

    def _set_view_contrast_limits(self, which: str, clim: Tuple[float, float]) -> None:
        layer = self._layer_for_view(which)
        if layer is None:
            return
        try:
            layer.contrast_limits = (float(clim[0]), float(clim[1]))
        except Exception:
            pass

    def _auto_lw_for_view(self, which: str) -> None:
        """더블클릭 Auto L/W: Raw/Result는 현재 ROI 기반으로 계산."""
        layer = self._layer_for_view(which)
        if layer is None:
            return
        try:
            data = np.asarray(layer.data)
        except Exception:
            return
        if data.size == 0:
            return

        # Raw/Result는 ROI 기반, Crop은 전체
        roi = self.roi_bounds if which in ("raw", "result") else None
        clim = percentile_clim(data, roi_bounds=roi)
        self._set_view_contrast_limits(which, clim)

    def _install_qt_mouse_filters(self) -> None:
        """QtViewer 캔버스에 이벤트 필터를 설치해 제스처를 '확실하게' 보장."""
        self._qt_filters = []
        for which, qt_viewer in (("raw", self.qt_raw), ("result", self.qt_result), ("crop", self.qt_crop)):
            try:
                cnv = getattr(qt_viewer, "canvas", None)
                native = getattr(cnv, "native", None) if cnv is not None else None
                if native is None:
                    continue
                f = _QtPanLwFilter(self, which)
                native.installEventFilter(f)
                self._qt_filters.append(f)
            except Exception:
                continue

    def _reset_view_both(self) -> None:
        try:
            self.model_raw.reset_view()
            self.model_result.reset_view()
        except Exception:
            # 버전에 따라 reset_view가 ViewerModel 메서드로 존재(문서에 있음)하나, 혹시 없으면 무시
            pass

    # ---------------------------
    # Camera sync (pan/zoom sync)
    # ---------------------------

    def _install_camera_sync(self) -> None:
        """
        napari Camera는 center/zoom/angles 등의 속성이 있고(2D에선 center[-2:], zoom이 핵심),
        이벤트는 camera.events.<field> 형태로 노출되는 경우가 많습니다.

        여기서는:
          raw 카메라 변경 -> result 카메라에 복사
          result 카메라 변경 -> raw 카메라에 복사

        루프 방지를 위해 self._camera_block 사용.
        """
        cam_r = self.model_raw.camera
        cam_o = self.model_result.camera

        def sync_from_raw(_event=None):
            if self._camera_block:
                return
            try:
                self._camera_block = True
                cam_o.center = tuple(cam_r.center)
                cam_o.zoom = float(cam_r.zoom)
                cam_o.angles = tuple(cam_r.angles)
            finally:
                self._camera_block = False

        def sync_from_res(_event=None):
            if self._camera_block:
                return
            try:
                self._camera_block = True
                cam_r.center = tuple(cam_o.center)
                cam_r.zoom = float(cam_o.zoom)
                cam_r.angles = tuple(cam_o.angles)
            finally:
                self._camera_block = False

        # 이벤트 연결 (center/zoom/angles)
        # napari 버전에 따라 events 구조가 조금 달라질 수 있어 try/except로 방어.
        for name in ("center", "zoom", "angles"):
            try:
                getattr(cam_r.events, name).connect(sync_from_raw)
            except Exception:
                pass
            try:
                getattr(cam_o.events, name).connect(sync_from_res)
            except Exception:
                pass

        # 폴링 백업: 어떤 이유로 이벤트가 안 불리는 환경 대비(드문 케이스)
        self._cam_poll_timer = QTimer(self)
        self._cam_poll_timer.setInterval(33)  # ~30Hz
        self._last_cam_state = None

        def poll():
            try:
                state = (tuple(cam_r.center), float(cam_r.zoom), tuple(cam_r.angles))
                if state != self._last_cam_state:
                    self._last_cam_state = state
                    sync_from_raw()
            except Exception:
                return

        self._cam_poll_timer.timeout.connect(poll)
        self._cam_poll_timer.start()

    # ---------------------------
    # Status bar: cursor position + DN (throttle)
    # ---------------------------

    def _strip_qtviewer_ui(self, qt_viewer: 'QtViewer') -> None:
        """QtViewer 기본 UI(레이어 리스트/레이어 컨트롤/콘솔 등)를 숨깁니다.

        - napari.qt.QtViewer는 기본적으로 dockLayerList, dockLayerControls, dockConsole 등을 갖습니다.
          (속성명은 napari 버전에 따라 달라질 수 있어 hasattr로 방어합니다.)
        """
        # 좌측 2개(레이어 리스트/컨트롤)
        for attr in ("dockLayerList", "dockLayerControls"):
            dw = getattr(qt_viewer, attr, None)
            if dw is None:
                continue
            try:
                dw.hide()
            except Exception:
                try:
                    dw.setVisible(False)
                except Exception:
                    pass

        # 콘솔(있으면 숨김)
        dw = getattr(qt_viewer, "dockConsole", None)
        if dw is not None:
            try:
                dw.hide()
            except Exception:
                try:
                    dw.setVisible(False)
                except Exception:
                    pass

        # 캔버스 우클릭 컨텍스트 메뉴/기본 상호작용(가능한 범위) 비활성화
        try:
            cnv = getattr(qt_viewer, 'canvas', None)
            native = getattr(cnv, 'native', None) if cnv is not None else None
            if native is not None:
                native.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        except Exception:
            pass

        # 일부 버전에서 남는 버튼/컨트롤 영역이 있으면 숨김(속성명은 버전별로 다를 수 있음)
        for attr in ('viewerButtons', '_viewerButtons', '_qt_viewer_controls', 'qt_viewer_controls'):
            w = getattr(qt_viewer, attr, None)
            if w is None:
                continue
            try:
                w.hide()
            except Exception:
                try:
                    w.setVisible(False)
                except Exception:
                    pass


    def _fit_raw_result_views(self) -> None:
        """Raw/Result 뷰를 현재 레이어 범위에 맞춰 '가운데 + 꽉 차게' 리셋합니다."""
        # ViewerModel.reset_view는 레이어 전체 extent를 기준으로 center/zoom을 맞춥니다.
        try:
            self.model_raw.reset_view(margin=0.0)
        except TypeError:
            # 구버전 시그니처 호환
            try:
                self.model_raw.reset_view()
            except Exception:
                pass

        # raw 기준 카메라를 result에 복사해서 두 뷰가 동일하게 보이도록 함
        try:
            cam_r = self.model_raw.camera
            cam_o = self.model_result.camera
            self._camera_block = True
            cam_o.center = tuple(cam_r.center)
            cam_o.zoom = float(cam_r.zoom)
            cam_o.angles = tuple(cam_r.angles)
        except Exception:
            pass
        finally:
            self._camera_block = False



    def _install_status_update(self) -> None:
        # raw / result 각각의 mouse move에 반응하도록 callbacks를 설치
        def make_mouse_move_cb(which: str):
            def cb(viewer, event):
                now = time.time()
                if now - self._last_status_t < self._status_interval:
                    return
                self._last_status_t = now

                if self.raw_data is None:
                    self._set_status("Ready")
                    return

                pos = getattr(event, "position", None)
                if pos is None or len(pos) < 2:
                    self._set_status("Ready")
                    return

                y = int(round(pos[-2]))
                x = int(round(pos[-1]))

                if self.width is None or self.height is None:
                    self._set_status(f"{which} | X:{x} Y:{y}")
                    return

                if not (0 <= x < self.width and 0 <= y < self.height):
                    self._set_status(f"{which} | X:{x} Y:{y}")
                    return

                if which == "Raw" and self.raw_data is not None:
                    dn = float(self.raw_data[y, x])
                    self._set_status(f"Raw | X:{x} Y:{y} | DN:{dn:.1f}")
                elif which == "Result" and self.processed_data is not None:
                    dn = float(self.processed_data[y, x])
                    self._set_status(f"Result | X:{x} Y:{y} | DN:{dn:.1f}")
                else:
                    self._set_status(f"{which} | X:{x} Y:{y}")

            return cb

        try:
            self.model_raw.mouse_move_callbacks.append(make_mouse_move_cb("Raw"))
            self.model_result.mouse_move_callbacks.append(make_mouse_move_cb("Result"))
        except Exception:
            pass

    def _install_mouse_controls(self) -> None:
        """요구된 마우스 매핑을 설치합니다.
        - 우클릭 드래그: 상/하 = L(레벨) 조절, 좌/우 = W(윈도우) 조절
        - 더블클릭: Auto L/W
        - 휠 클릭: View(줌/팬) 초기화
        """

        # napari 기본 마우스/단축키 바인딩과 충돌을 피하기 위해
        #  - mouse_*_callbacks 를 '전체 초기화' 후 커스텀만 다시 등록
        #  - camera.mouse_zoom/mouse_pan도 끕니다(우클릭 드래그가 줌으로 먹는 케이스 방지)
        #  - keymap(예: a 등)도 비웁니다
        for _m in (getattr(self, 'model_raw', None), getattr(self, 'model_result', None), getattr(self, 'model_crop', None)):
            if _m is None:
                continue
            try:
                _m.mouse_drag_callbacks[:] = []
            except Exception:
                pass
            try:
                _m.mouse_double_click_callbacks[:] = []
            except Exception:
                pass
            try:
                _m.mouse_wheel_callbacks[:] = []
            except Exception:
                pass
            try:
                _m.mouse_click_callbacks[:] = []
            except Exception:
                pass
            try:
                _m.camera.mouse_zoom = False
            except Exception:
                pass
            try:
                _m.camera.mouse_pan = False
            except Exception:
                pass
            # 단축키 비활성화(혼선 방지)
            try:
                km = getattr(_m, 'keymap', None)
                if isinstance(km, dict):
                    km.clear()
            except Exception:
                pass
            try:
                if hasattr(_m, '_keymap') and isinstance(_m._keymap, dict):
                    _m._keymap.clear()
            except Exception:
                pass

        # host_viewer(숨겨진 메인 viewer) 쪽 키맵도 비워 단축키 혼선 방지
        try:
            km = getattr(self.host_viewer, 'keymap', None)
            if isinstance(km, dict):
                km.clear()
        except Exception:
            pass
        try:
            if hasattr(self.host_viewer, '_keymap') and isinstance(self.host_viewer._keymap, dict):
                self.host_viewer._keymap.clear()
        except Exception:
            pass


        def _get_main_layer(model: ViewerModel) -> Optional[napari.layers.Image]:
            # Raw/Result/Crop에서 가장 앞의 Image 레이어를 대상으로 함
            for lyr in list(getattr(model, 'layers', [])):
                try:
                    from napari.layers import Image
                    if isinstance(lyr, Image):
                        return lyr
                except Exception:
                    # 타입 체크 실패 시 이름으로 fallback
                    if getattr(lyr, 'name', '') in ('Raw', 'Result', 'Crop'):
                        return lyr
            return None

        def _set_clim(which: str, clim):
            """요청: Raw에서 Auto L/W 시 Result는 건드리지 않음. (Raw/Result 각각 독립)"""
            try:
                if which == 'raw' and self.layer_raw is not None:
                    self.layer_raw.contrast_limits = clim
                elif which == 'result' and self.layer_result is not None:
                    self.layer_result.contrast_limits = clim
                elif which == 'crop' and self.layer_crop is not None:
                    self.layer_crop.contrast_limits = clim
            except Exception:
                pass

        def _auto_clim_from_layer(layer):
            try:
                data = np.asarray(layer.data)
                if data.size == 0:
                    return None
                ds = data[::4, ::4] if data.ndim == 2 else data
                lo = float(np.percentile(ds, 1))
                hi = float(np.percentile(ds, 99))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo = float(np.nanmin(ds)); hi = float(np.nanmax(ds))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    return None
                return (lo, hi)
            except Exception:
                return None

        def _reset_all(which: str):
            """이미지 초기화: 위치/줌 + L/W(contrast)까지 초기화 (요청: Raw/Result 각각)"""
            # 1) view reset (zoom/pan)
            try:
                if which == 'raw':
                    self.model_raw.reset_view(margin=0.0)
                elif which == 'result':
                    self.model_result.reset_view(margin=0.0)
                else:
                    self.model_crop.reset_view(margin=0.0)
            except Exception:
                try:
                    if which == 'raw':
                        self.model_raw.reset_view()
                    elif which == 'result':
                        self.model_result.reset_view()
                    else:
                        self.model_crop.reset_view()
                except Exception:
                    pass

            # 2) L/W reset (auto)
            self._ensure_layers()
            layer = self.layer_raw if which == 'raw' else (self.layer_result if which == 'result' else self.layer_crop)
            if layer is not None:
                clim = _auto_clim_from_layer(layer)
                if clim is not None:
                    _set_clim(which, clim)

        # -----------------------
        # 1) 휠 클릭: view 초기화
        # -----------------------
        def make_middle_click_cb(which: str):
            def cb(viewer, event):
                try:
                    btn = getattr(event, 'button', None)
                    # 숫자 코드에 의존하면 OS/드라이버별로 뒤틀릴 수 있어서 Qt 상수 우선
                    is_mid = False
                    try:
                        is_mid = (btn == Qt.MouseButton.MiddleButton)
                    except Exception:
                        is_mid = False
                    if not is_mid:
                        return
                except Exception:
                    return

                _reset_all(which)

                try:
                    event.handled = True
                except Exception:
                    pass
                yield
            return cb

        # -----------------------
        # 2) 우클릭 드래그: L/W
        # -----------------------
        def make_right_drag_cb(which: str):
            def drag_cb(viewer, event):
                if not _right_button(event):
                    return

                try:
                    event.handled = True
                except Exception:
                    pass

                # 기준 레이어 및 초기 값
                self._ensure_layers()
                layer = self.layer_crop if which == 'crop' else (self.layer_raw if which == 'raw' else self.layer_result)
                if layer is None:
                    return
                try:
                    cmin, cmax = layer.contrast_limits
                    cmin = float(cmin); cmax = float(cmax)
                except Exception:
                    return
                level0 = (cmin + cmax) / 2.0
                width0 = max(1e-6, (cmax - cmin))

                # 드래그 시작 위치(캔버스 픽셀)
                p0 = getattr(event, 'pos', None)
                if p0 is None:
                    p0 = getattr(event, 'position', None)
                if p0 is None:
                    return
                x0, y0 = float(p0[0]), float(p0[1])

                # 감도: 화면 300px 이동 시 width가 2배 정도 바뀌는 느낌
                # (너무 민감하면 scale만 조절)
                scale_w = width0 / 300.0
                scale_l = width0 / 300.0

                while True:
                    ev = yield
                    p = getattr(ev, 'pos', None)
                    if p is None:
                        p = getattr(ev, 'position', None)
                    if p is None:
                        continue
                    x, y = float(p[0]), float(p[1])
                    dx = x - x0
                    dy = y - y0

                    # 좌/우: W, 상/하: L (상으로 올리면 밝게? 요청은 상하는 L조절만이므로 방향은 직관적으로)
                    width = max(1e-6, width0 + dx * scale_w)
                    level = level0 - dy * scale_l  # 위로 드래그 -> level 증가(더 밝게)

                    new_min = level - width / 2.0
                    new_max = level + width / 2.0
                    clim = (new_min, new_max)

                    _set_clim(which, clim)

                    try:
                        ev.handled = True
                    except Exception:
                        pass
            return drag_cb

        # -----------------------
        # 3) 더블클릭: Auto L/W
        # -----------------------
        def make_double_click_cb(which: str):
            def cb(viewer, event):
                # 좌클릭 더블클릭만 Auto L/W
                try:
                    btn = getattr(event, 'button', None)
                    is_left = (btn == 1)
                    try:
                        is_left = is_left or (btn == Qt.MouseButton.LeftButton)
                    except Exception:
                        pass
                    if not is_left:
                        return
                except Exception:
                    return

                self._ensure_layers()
                layer = self.layer_crop if which == 'crop' else (self.layer_raw if which == 'raw' else self.layer_result)
                if layer is None:
                    return
                try:
                    data = np.asarray(layer.data)
                except Exception:
                    return
                if data.size == 0:
                    return

                # 빠른 auto: downsample 후 1~99 percentile
                try:
                    roi = self.roi_bounds if which in ('raw','result') else None
                    lo, hi = percentile_clim(data, roi_bounds=roi)
                except Exception:
                    return

                clim = (lo, hi)
                _set_clim(which, clim)

                try:
                    event.handled = True
                except Exception:
                    pass
            return cb

        # -----------------------
        # 4) 휠: 커스텀 줌 (줌 인/아웃)
        # -----------------------
        def make_wheel_zoom_cb(which: str):
            """마우스 휠로 camera.zoom을 조절합니다.

            - napari/vispy 환경에 따라 wheel delta 스케일이 (1) 또는 (120) 등으로 달라질 수 있어
              정규화해서 처리합니다.
            - 기본 바인딩을 꺼둔 상태에서도 동작하도록, 우리가 직접 cam.zoom을 변경합니다.
            """
            def cb(viewer, event):
                dy = None

                # delta 확보 (가장 흔한 케이스)
                try:
                    delta = getattr(event, 'delta', None)
                    if isinstance(delta, (tuple, list)) and len(delta) >= 2:
                        dy = delta[1]
                    elif isinstance(delta, (int, float)):
                        dy = delta
                except Exception:
                    dy = None

                # fallback
                if dy is None:
                    try:
                        dy = getattr(event, 'dy', None)
                    except Exception:
                        dy = None

                if dy is None:
                    return

                try:
                    dyf = float(dy)
                except Exception:
                    return

                # 큰 값(예: 120 단위)은 120으로 나눠 steps로 환산
                steps = dyf
                if abs(steps) > 10:
                    steps = steps / 120.0

                # wheel up(+): zoom in, wheel down(-): zoom out
                base = 1.15
                factor = base ** steps

                try:
                    if which == 'raw':
                        cam = self.model_raw.camera
                    elif which == 'result':
                        cam = self.model_result.camera
                    else:
                        cam = self.model_crop.camera
                except Exception:
                    cam = None

                if cam is None:
                    return

                try:
                    z = float(getattr(cam, 'zoom', 1.0))
                    newz = z * factor
                    if not np.isfinite(newz):
                        return
                    newz = max(0.02, min(200.0, newz))
                    cam.zoom = newz
                except Exception:
                    pass

                try:
                    event.handled = True
                except Exception:
                    pass
                return

            return cb

        # -----------------------
        # 5) ROI: Shift + 좌클릭 드래그로 Rectangle 1개 그리기
        #    ROI 삭제: Shift + 우클릭
        # -----------------------
        def _shift_pressed(ev) -> bool:
            try:
                mods = getattr(ev, 'modifiers', None)
                if isinstance(mods, (list, tuple, set)):
                    for m in mods:
                        if str(m).lower().startswith('shift'):
                            return True
            except Exception:
                pass
            try:
                return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
            except Exception:
                return False

        def _left_button(ev) -> bool:
            btn = getattr(ev, 'button', None)
            try:
                if btn == Qt.MouseButton.LeftButton:
                    return True
            except Exception:
                pass
            return btn == 1

        def _right_button(ev) -> bool:
            btn = getattr(ev, 'button', None)
            if btn is None:
                return False
            # napari/Qt 버전/플랫폼에 따라 button 값이 int(2) 또는 Qt enum 또는 'right' 문자열로 올 수 있음
            if isinstance(btn, str):
                b = btn.lower()
                return ('right' in b) or ('secondary' in b)
            try:
                if btn == Qt.MouseButton.RightButton:
                    return True
            except Exception:
                pass
            # 일부 환경에서는 enum이 int로 비교되어야 함
            try:
                if int(btn) == int(Qt.MouseButton.RightButton):
                    return True
            except Exception:
                pass
            return btn == 2

        def _rect_from_two_points(y0: float, x0: float, y1: float, x1: float) -> np.ndarray:
            ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
            xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
            # 최소 1px 이상 보이도록 미세 보정
            if abs(yb - ya) < 1e-3:
                yb = ya + 1.0
            if abs(xb - xa) < 1e-3:
                xb = xa + 1.0
            return np.array([[ya, xa], [ya, xb], [yb, xb], [yb, xa]], dtype=np.float32)

        def make_shift_roi_drag_cb():
            def drag_cb(viewer, event):
                if self.raw_data is None:
                    return
                if not (_shift_pressed(event) and _left_button(event)):
                    return

                self._ensure_layers()
                if self.layer_roi is None:
                    return

                # 시작점 (data coords: (y,x))
                p0 = getattr(event, 'position', None)
                if p0 is None or len(p0) < 2:
                    return
                y0 = float(p0[-2]); x0 = float(p0[-1])

                try:
                    event.handled = True
                except Exception:
                    pass

                try:
                    self.layer_roi.mode = 'select'
                except Exception:
                    pass

                rect_last: Optional[np.ndarray] = None

                # 드래그 중 실시간 업데이트(레이어 이벤트는 막고, 종료 시 1회만 처리)
                try:
                    while True:
                        ev = yield
                        p = getattr(ev, 'position', None)
                        if p is None or len(p) < 2:
                            continue
                        y1 = float(p[-2]); x1 = float(p[-1])

                        # 이미지 범위 클리핑(가능한 경우)
                        try:
                            if self.width is not None and self.height is not None:
                                y0c = float(np.clip(y0, 0, self.height - 1))
                                x0c = float(np.clip(x0, 0, self.width - 1))
                                y1c = float(np.clip(y1, 0, self.height - 1))
                                x1c = float(np.clip(x1, 0, self.width - 1))
                            else:
                                y0c, x0c, y1c, x1c = y0, x0, y1, x1
                        except Exception:
                            y0c, x0c, y1c, x1c = y0, x0, y1, x1

                        rect_last = _rect_from_two_points(y0c, x0c, y1c, x1c)
                        try:
                            self._roi_internal_update = True
                            self.layer_roi.data = [rect_last]
                        except Exception:
                            pass
                        finally:
                            self._roi_internal_update = False

                        try:
                            ev.handled = True
                        except Exception:
                            pass
                        # 일부 napari 버전에서는 drag generator가 'mouse_release'를 보내거나,
                        # is_dragging 플래그를 False로 바꿔 종료를 알립니다. 종료 신호가 오면 루프를 빠져나갑니다.
                        try:
                            et = getattr(ev, 'type', None) or getattr(ev, 'event_type', None)
                            dragging = getattr(ev, 'is_dragging', None)
                            if dragging is False:
                                break
                            if isinstance(et, str) and ('release' in et.lower() or 'up' == et.lower()):
                                break
                        except Exception:
                            pass
                finally:
                    # 드래그 종료 시 ROI bounds 업데이트 + 1회만 보정/표시 갱신
                    try:
                        if self.width is not None and self.height is not None and self.layer_roi is not None:
                            self.roi_bounds = roi_from_shapes_layer(self.layer_roi, self.width, self.height)
                    except Exception:
                        pass
                    try:
                        self._apply_process_and_refresh()
                    except Exception:
                        pass
                    try:
                        self._set_status('ROI set (Shift+좌클릭 드래그). Clear: Shift+우클릭')
                    except Exception:
                        pass
            return drag_cb

        def make_shift_right_click_clear_roi_cb():
            def cb(viewer, event):
                if self.raw_data is None:
                    return
                if not (_shift_pressed(event) and _right_button(event)):
                    return
                self._on_roi_clear()
                try:
                    event.handled = True
                except Exception:
                    pass
            return cb

        # Raw/Result/Crop에 모두 설치
        for model, which in (
            (self.model_raw, 'raw'),
            (self.model_result, 'result'),
            (self.model_crop, 'crop'),
        ):
            try:
                model.mouse_click_callbacks.append(make_middle_click_cb(which))
            except Exception:
                pass
            try:
                model.mouse_drag_callbacks.append(make_right_drag_cb(which))
            except Exception:
                pass
            try:
                model.mouse_double_click_callbacks.append(make_double_click_cb(which))
            except Exception:
                pass
            try:
                model.mouse_wheel_callbacks.append(make_wheel_zoom_cb(which))
            except Exception:
                pass

        # ROI 바인딩은 Raw에만 설치
        try:
            self.model_raw.mouse_drag_callbacks.append(make_shift_roi_drag_cb())
        except Exception:
            pass
        try:
            self.model_raw.mouse_click_callbacks.append(make_shift_right_click_clear_roi_cb())
        except Exception:
            pass

        # host_viewer(숨김) 쪽의 기본 단축키도 비활성화
        try:
            hm = getattr(self.host_viewer, 'viewer_model', None)
            if hm is not None:
                km = getattr(hm, 'keymap', None)
                if isinstance(km, dict):
                    km.clear()
        except Exception:
            pass

    def _on_load_clicked(self) -> None:
        model = self.cb_model.currentText()
        typ = self.cb_type.currentText()
        # ensure blemish params match selected model/type profile
        self._load_profile_params_for_current_selection()
        w, h = MODEL_DIMENSIONS.get(model, {}).get(typ, (None, None))
        if w is None or h is None:
            show_err(self, "Error", "This model/type has no fixed dimensions.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Raw", "", "IMG/RAW (*.IMG *.img *.raw *.RAW);;All (*.*)"
        )
        if not file_path:
            return

        try:
            size_bytes = int(w) * int(h) * 2
            fsz = os.path.getsize(file_path)
            header = fsz - size_bytes
            if header < 0:
                raise ValueError("File too small for expected dimensions")

            with open(file_path, "rb") as fp:
                fp.seek(header)
                data = np.fromfile(fp, dtype=np.uint16, count=int(w) * int(h))

            if data.size != int(w) * int(h):
                raise ValueError("Data size mismatch")

            raw = data.reshape((int(h), int(w))).astype(np.float32)

            # keep original dimensions (before orientation correction)
            self._orig_w = int(w)
            self._orig_h = int(h)

            # 요청: 특정 모델은 로드 후 orientation 보정(rotate left -> flip horizontal)
            # - 표시/크롭/저장 모두 동일하게 되도록 raw 자체를 변환
            if model in ORIENT_ROTATE_LEFT_FLIP_H_MODELS:
                try:
                    raw = np.rot90(raw, k=1)  # 90 deg left (CCW)
                    raw = np.fliplr(raw)      # flip horizontally
                except Exception:
                    pass


            # 연산(필터/라벨링) 속도 최적화: orientation 보정 후 비연속(strided) 배열이 될 수 있으므로 연속 메모리로 맞춥니다.
            try:
                raw = np.ascontiguousarray(raw)
            except Exception:
                pass

            self.raw_data = raw
            self.processed_data = None
            # 변환 후 실제 크기를 기준으로 width/height 갱신
            try:
                self.height = int(raw.shape[0])
                self.width = int(raw.shape[1])
            except Exception:
                self.width = int(w)
                self.height = int(h)
            self.current_model = model
            # pixel pitch: legacy (area now displayed in px^2)
            try:
                self.pixel_pitch_cm = estimate_pixel_pitch_cm(
                    model,
                    img_w_px=int(self.width or 0),
                    img_h_px=int(self.height or 0),
                    orig_w_px=getattr(self, '_orig_w', None),
                    orig_h_px=getattr(self, '_orig_h', None),
                )
            except Exception:
                self.pixel_pitch_cm = None
            self.loaded_file_path = file_path
            self._fit_after_load = True  # 새 파일 로딩 시 1회 reset_view
            self.blemish_save_dir = os.path.dirname(file_path) if os.path.isdir(os.path.dirname(file_path)) else None

            self._ensure_layers()
            self._clear_roi_and_blemish_layers()
            self._refresh_raw_layer()
            # 고정 ROI 프리셋을 자동 적용 (요청)
            self._apply_fixed_roi_preset_on_load(model)
            self._apply_process_and_refresh()

            # 로딩 직후 ROI 내 저DN 라인(0~100) 탐지 팝업 (요청)
            self._check_low_dn_line_and_popup()

            # 로딩 후 자동 blemish detect (Raw)
            try:
                self.cb_target.setCurrentText("Raw")
            except Exception:
                pass
            self._on_detect()

            roi_desc = ""
            if self.roi_bounds is not None and getattr(self.roi_bounds, "valid", False):
                roi_desc = f" | ROI=({self.roi_bounds.x1},{self.roi_bounds.y1})-({self.roi_bounds.x2},{self.roi_bounds.y2})"
            # status에는 실제 표시 크기를 표기
            self._set_status(f"Loaded: {os.path.basename(file_path)} ({self.width}x{self.height}){roi_desc}")
        except Exception as e:
            show_err(self, "Load Error", str(e))



    # ---------------------------
    # Line artifact popup (요청)
    # ---------------------------

    def _check_low_dn_line_and_popup(self) -> None:
        """ROI 안에서 DN 0~100 라인이 있으면 로딩 직후 팝업으로 알립니다."""
        try:
            if self.raw_data is None:
                return
            roi = self.roi_bounds if (self.roi_bounds is not None and getattr(self.roi_bounds, 'valid', False)) else None
            fixed_x = _parse_fixed_line_list(self.blemish_params.get("fixed_line_x"))
            fixed_y = _parse_fixed_line_list(self.blemish_params.get("fixed_line_y"))

            bad_cols, bad_rows, col_runs, row_runs, broken_points = detect_low_value_lines_in_roi(
                self.raw_data,
                roi,
                dn_lo=0.0,
                dn_hi=100.0,
                frac_threshold=0.95,
                ignore_border_px=8,
                fixed_line_x=fixed_x,
                fixed_line_y=fixed_y,
            )

            total_lines = int(len(bad_cols) + len(bad_rows))
            max_run = 0
            for (s, e) in (col_runs + row_runs):
                try:
                    max_run = max(max_run, int(e) - int(s) + 1)
                except Exception:
                    pass

            if broken_points:
                coord_text = ", ".join(f"({x},{y})" for x, y in broken_points)
                show_info(self, "끊긴 라인 발견", f"끊긴 라인 발견! {coord_text}")

            # 3개 이상(집합/다발) 발견 시 강한 경고
            if total_lines >= 3 or max_run >= 3:
                show_info(self, "3LINE 발견", "3LINE 발견, 확인필요!")
            elif total_lines >= 1:
                show_info(self, "LINE 발견", "LINE 발견, Raw -> Collected blemish 검사 추천!")
        except Exception:
            # 팝업 실패/예외는 동작을 막지 않음
            return
    # ---------------------------
    # ROI
    # ---------------------------

    def _apply_fixed_roi_preset_on_load(self, model: str) -> None:
        """요청된 고정 ROI를 로딩 시 자동 적용.
        - ROI_PRESETS_XY는 (x,y) 기준이므로, napari shapes (y,x)로 변환
        - self.roi_bounds도 즉시 갱신
        """
        if self.layer_roi is None or self.width is None or self.height is None:
            return
        preset = ROI_PRESETS_XY.get(model)
        if preset is None:
            return

        (x1, y1), (x2, y2) = preset

        # orientation correction models: preset is given in original coords
        try:
            x1, y1, x2, y2 = orient_transform_box_xy(model, x1, y1, x2, y2, getattr(self, '_orig_w', None), getattr(self, '_orig_h', None))
        except Exception:
            pass
        w, h = int(self.width), int(self.height)

        # clamp + 최소 크기 보장 (slice upper bound는 exclusive)
        x1 = max(0, min(w - 2, int(x1)))
        y1 = max(0, min(h - 2, int(y1)))
        x2 = max(x1 + 2, min(w, int(x2)))
        y2 = max(y1 + 2, min(h, int(y2)))

        rect = np.asarray([[y1, x1], [y1, x2], [y2, x2], [y2, x1]], dtype=np.float32)

        try:
            self._roi_internal_update = True
            self.layer_roi.data = [rect]
            try:
                self.layer_roi.mode = 'select'
            except Exception:
                pass
        finally:
            self._roi_internal_update = False

        self.roi_bounds = RoiBounds(x1=x1, x2=x2, y1=y1, y2=y2)

    def _clear_roi_and_blemish_layers(self) -> None:
        # clear ROI shapes
        if self.layer_roi is not None:
            try:
                self.layer_roi.data = []
            except Exception:
                pass
        self.roi_bounds = None

        # clear blemish points
        if self.layer_blemish_points is not None:
            try:
                # remove from both viewers if present
                if self.layer_blemish_points in self.model_raw.layers:
                    self.model_raw.layers.remove(self.layer_blemish_points)
            except Exception:
                pass
            try:
                if self.layer_blemish_points in self.model_result.layers:
                    self.model_result.layers.remove(self.layer_blemish_points)
            except Exception:
                pass
            self.layer_blemish_points = None

        # clear table + crop viewer
        self.tbl.setRowCount(0)
        self.blemish_last_candidates = []
        self.blemish_last_img_full = None
        self.blemish_last_ref_mean = None
        self._update_blemish_text_overlay(clear_only=True)
        if self.layer_crop is not None:
            self.layer_crop.data = np.zeros((10, 10), np.float32)
            self.layer_crop.contrast_limits = (0, 1)

    def _on_roi_clear(self) -> None:
        if self.layer_roi is None:
            return
        try:
            self._roi_internal_update = True
            self.layer_roi.data = []
        except Exception:
            pass
        finally:
            self._roi_internal_update = False
        self.roi_bounds = None
        try:
            self.layer_roi.mode = 'select'
        except Exception:
            pass
        self._apply_process_and_refresh()


    def _on_roi_changed(self, _event=None) -> None:
        if self._roi_internal_update:
            return
        if self.raw_data is None or self.width is None or self.height is None:
            return
        if self.layer_roi is None:
            return

        try:
            data = list(self.layer_roi.data) if self.layer_roi.data is not None else []
        except Exception:
            data = []

        # 언제나 ROI는 1개만 유지 (여러개가 생기면 마지막 것만 남김)
        if len(data) > 1:
            try:
                self._roi_internal_update = True
                self.layer_roi.data = [data[-1]]
                data = [data[-1]]
            finally:
                self._roi_internal_update = False

        # ROI는 (있으면) 항상 보정/검출에 사용합니다. (우측 ROI UI 제거 요청)
        self.roi_bounds = roi_from_shapes_layer(self.layer_roi, self.width, self.height)

        self._apply_process_and_refresh()

    # ---------------------------
    # Refresh layers
    # ---------------------------

    def _refresh_raw_layer(self) -> None:
        if self.raw_data is None:
            return
        self._ensure_layers()
        self.layer_raw.data = self.raw_data
        self.layer_raw.contrast_limits = percentile_clim(self.raw_data, roi_bounds=self.roi_bounds)

    def _refresh_result_layer(self) -> None:
        if self.processed_data is None:
            return
        self._ensure_layers()
        self.layer_result.data = self.processed_data
        self.layer_result.contrast_limits = percentile_clim(self.processed_data, roi_bounds=self.roi_bounds)

        # 모델별 Result red box 갱신
        try:
            self._update_result_red_box()
        except Exception:
            pass


    def _update_result_red_box(self) -> None:
        """Result 이미지에 모델별 빨간 박스를 표시합니다."""
        # 기존 박스 제거
        _remove_layers_by_prefix(self.model_result, "Result Red Box")

        if self.processed_data is None:
            return
        model = self.current_model or self.cb_model.currentText()
        preset = RESULT_REDBOX_XY.get(model)
        if preset is None:
            return

        (x1, y1), (x2, y2) = preset

        # orientation correction models: preset is given in original coords
        try:
            x1, y1, x2, y2 = orient_transform_box_xy(model, x1, y1, x2, y2, getattr(self, '_orig_w', None), getattr(self, '_orig_h', None))
        except Exception:
            pass
        H, W = int(self.processed_data.shape[0]), int(self.processed_data.shape[1])

        # clamp
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(0, min(W, int(x2)))
        y2 = max(0, min(H, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return

        rect = np.asarray(
            [[y1, x1], [y1, x2], [y2, x2], [y2, x1]],
            dtype=np.float32,
        )

        try:
            lay = self.model_result.add_shapes(
                [rect],
                shape_type="polygon",
                name="Result Red Box",
                edge_color="red",
                face_color="transparent",
                edge_width=1,
                opacity=1.0,
            )
        except Exception:
            lay = self.model_result.add_shapes(
                [rect],
                shape_type="polygon",
                name="Result Red Box",
                edge_color="red",
                edge_width=1,
                opacity=1.0,
            )
        try:
            lay.interactive = False
        except Exception:
            pass

    def _apply_process_and_refresh(self) -> None:
        if self.raw_data is None:
            return
        self._ensure_layers()

        self.processed_data = apply_line_noise_correction(self.raw_data, self.roi_bounds)

        self._refresh_raw_layer()
        self._refresh_result_layer()


        # 새 파일 로딩 직후에는 항상 화면 중앙에 꽉 차게 맞춤
        if self._fit_after_load:
            self._fit_after_load = False
            self._fit_raw_result_views()
    # ---------------------------
    # Blemish
    # ---------------------------

    def _on_param(self) -> None:
        try:
            model = self.cb_model.currentText()
            typ = self.cb_type.currentText()
        except Exception:
            model, typ = "", ""

        params, saved_at = ({}, None)
        try:
            if model and typ:
                params, saved_at = self._get_profile_entry(model, typ)
        except Exception:
            pass

        # ensure current params reflect selected profile
        try:
            self._load_profile_params_for_current_selection()
        except Exception:
            pass

        dlg = BlemishParamDialog(self.blemish_params, model=model, typ=typ, saved_at=saved_at, parent=self)

        def _apply_and_refresh(p: Dict[str, Any]) -> None:
            self._on_param_applied(p)
            try:
                dlg.set_profile_info(self.cb_model.currentText(), self.cb_type.currentText(), self.blemish_profile_saved_at)
            except Exception:
                pass

        dlg.applied.connect(_apply_and_refresh)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _on_param_applied(self, new_params: Dict[str, Any]) -> None:
        self.blemish_params.update(new_params)
        # normalize CropΔ threshold keys (U/L + legacy mirror)
        self._normalize_cropdelta_thresholds(self.blemish_params)
        saved_at = self._save_profile_params_for_current_selection()
        try:
            model = self.cb_model.currentText()
            typ = self.cb_type.currentText()
        except Exception:
            model, typ = "", ""
        if saved_at:
            self._set_status(f"Blemish params saved: {model}/{typ} | {saved_at}")
        else:
            self._set_status("Blemish params updated.")

    def _freeze_detect_value(self, v: Any) -> Any:
        """detect cache key용: 해시 가능한 형태로 최대한 안정적으로 변환"""
        if v is None or isinstance(v, (int, float, str, bool)):
            return v
        try:
            return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            try:
                return repr(v)
            except Exception:
                return str(type(v))

    def _make_detect_cache_key(self, img: np.ndarray, tgt: str) -> Any:
        # 이미지 태그: 파일경로/mtime + 배열 identity(주소)까지 포함 (안전하게 stale 방지)
        fp = self.loaded_file_path or ""
        try:
            mtime = os.path.getmtime(fp) if fp else None
        except Exception:
            mtime = None
        try:
            img_tag = (fp, mtime, int(id(img)), tuple(int(x) for x in img.shape), str(img.dtype))
        except Exception:
            img_tag = (fp, mtime, int(id(img)))

        # ROI 태그
        roi_tag = None
        try:
            rb = self.roi_bounds
            if rb is not None and getattr(rb, "valid", False):
                roi_tag = (int(rb.x1), int(rb.x2), int(rb.y1), int(rb.y2))
        except Exception:
            roi_tag = None

        # detect에 영향을 주는 핵심 파라미터만 키에 포함
        p = self.blemish_params or {}
        detect_keys = [
            "analysis_scope",
            "use_roi_as_reference",
            "measure_window",
            "bg_window",
            "threshold_crop_percent",
            "min_dn",
            "min_area",
            "max_area",
            "min_circularity",
            "max_eccentricity",
            "border_margin",
            "max_detections",
            "threshold_ref_percent",
            "threshold_ref_percent_l",
            "threshold_ref_percent_u",
        ]
        p_tag = tuple((k, self._freeze_detect_value(p.get(k))) for k in detect_keys)
        return (str(tgt), img_tag, roi_tag, p_tag)

    def _detect_cache_get(self, key: Any) -> Optional[Tuple[List[BlemishCandidate], float, str]]:
        try:
            v = self._detect_cache.get(key)
            if v is not None:
                try:
                    self._detect_cache.move_to_end(key)
                except Exception:
                    pass
            return v
        except Exception:
            return None

    def _detect_cache_put(self, key: Any, value: Tuple[List[BlemishCandidate], float, str]) -> None:
        try:
            self._detect_cache[key] = value
            try:
                self._detect_cache.move_to_end(key)
            except Exception:
                pass
            while len(self._detect_cache) > int(self._detect_cache_max):
                try:
                    self._detect_cache.popitem(last=False)
                except Exception:
                    break
        except Exception:
            pass

    def _start_detect_job(
        self,
        img: np.ndarray,
        tgt: str,
        params_copy: Dict[str, Any],
        roi_copy: Optional[RoiBounds],
        cache_key: Any,
    ) -> None:
        # token update
        self._detect_token = int(self._detect_token) + 1
        token = int(self._detect_token)

        # active job context
        self._detect_active_img = img
        self._detect_active_tgt = str(tgt)
        self._detect_active_cache_key = cache_key

        self._detect_running = True
        try:
            if hasattr(self, "btn_detect") and self.btn_detect is not None:
                self.btn_detect.setEnabled(False)
        except Exception:
            pass

        # busy cursor
        try:
            cursor = None
            try:
                cursor = getattr(getattr(Qt, "CursorShape", Qt), "WaitCursor")
            except Exception:
                cursor = getattr(Qt, "WaitCursor", None)
            if cursor is not None:
                QApplication.setOverrideCursor(cursor)
                self._detect_cursor_set = True
        except Exception:
            self._detect_cursor_set = False

        self._set_status(f"Detecting... | target={tgt}")

        thread = QThread()
        worker = _DetectBlemishWorker(img, params_copy, roi_copy, token, cache_key)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_detect_worker_done)
        worker.error.connect(self._on_detect_worker_error)

        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)

        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # keep references
        self._detect_thread = thread
        self._detect_worker = worker

        thread.start()

    def _on_detect_worker_done(self, payload: Any) -> None:
        # payload: (cand, ref_mean, scope_desc, token, cache_key)
        try:
            cand, ref_mean, scope_desc, token, cache_key = payload
        except Exception:
            # unexpected payload
            self._detect_running = False
            try:
                show_err(self, "Detect Error", "Internal payload error")
            except Exception:
                pass
            return

        # cache first (even if UI apply is skipped)
        try:
            self._detect_cache_put(cache_key, (cand, float(ref_mean), str(scope_desc)))
        except Exception:
            pass

        # restore cursor
        try:
            if self._detect_cursor_set:
                QApplication.restoreOverrideCursor()
        except Exception:
            pass
        self._detect_cursor_set = False

        self._detect_running = False
        try:
            if hasattr(self, "btn_detect") and self.btn_detect is not None:
                self.btn_detect.setEnabled(True)
        except Exception:
            pass

        # stale guard: current selection/image가 바뀌었으면 UI 적용은 건너뜀
        apply_ok = True
        try:
            if int(token) != int(getattr(self, "_detect_token", -1)):
                apply_ok = False
        except Exception:
            pass
        try:
            cur_tgt = self.cb_target.currentText()
            if str(cur_tgt) != str(getattr(self, "_detect_active_tgt", "")):
                apply_ok = False
        except Exception:
            pass
        try:
            # 현재 target의 이미지 객체가 동일할 때만 적용
            cur_tgt = self.cb_target.currentText()
            cur_img = self.processed_data if (cur_tgt == "Corrected" and self.processed_data is not None) else self.raw_data
            if cur_img is None or int(id(cur_img)) != int(id(getattr(self, "_detect_active_img", None))):
                apply_ok = False
        except Exception:
            apply_ok = False

        if apply_ok:
            self._apply_detect_result(
                img=getattr(self, "_detect_active_img", None),
                tgt=getattr(self, "_detect_active_tgt", ""),
                cand=cand,
                ref_mean=float(ref_mean),
                scope_desc=str(scope_desc),
            )

        # run queued request if exists
        try:
            if self._detect_queue is not None:
                tgt2, img2, params2, roi2, key2 = self._detect_queue
                self._detect_queue = None
                # queued detect 시작 전에 이전 오버레이를 정리(중복/잔상 방지)
                self._update_blemish_text_overlay(clear_only=True)
                # cache hit일 수도 있으니 재확인
                cached = self._detect_cache_get(key2)
                if cached is not None:
                    c2, rm2, sd2 = cached
                    self._apply_detect_result(img=img2, tgt=tgt2, cand=c2, ref_mean=float(rm2), scope_desc=str(sd2))
                else:
                    self._start_detect_job(img2, tgt2, params2, roi2, key2)
        except Exception:
            pass

    def _on_detect_worker_error(self, payload: Any) -> None:
        try:
            msg, token = payload
        except Exception:
            msg, token = "Unknown error", -1

        # restore cursor
        try:
            if self._detect_cursor_set:
                QApplication.restoreOverrideCursor()
        except Exception:
            pass
        self._detect_cursor_set = False

        self._detect_running = False
        try:
            if hasattr(self, "btn_detect") and self.btn_detect is not None:
                self.btn_detect.setEnabled(True)
        except Exception:
            pass

        try:
            show_err(self, "Detect Error", str(msg))
        except Exception:
            pass

        # run queued request if exists
        try:
            if self._detect_queue is not None:
                tgt2, img2, params2, roi2, key2 = self._detect_queue
                self._detect_queue = None
                # queued detect 시작 전에 이전 오버레이를 정리(중복/잔상 방지)
                self._update_blemish_text_overlay(clear_only=True)
                cached = self._detect_cache_get(key2)
                if cached is not None:
                    c2, rm2, sd2 = cached
                    self._apply_detect_result(img=img2, tgt=tgt2, cand=c2, ref_mean=float(rm2), scope_desc=str(sd2))
                else:
                    self._start_detect_job(img2, tgt2, params2, roi2, key2)
        except Exception:
            pass

    def _apply_detect_result(
        self,
        img: Optional[np.ndarray],
        tgt: str,
        cand: List[BlemishCandidate],
        ref_mean: float,
        scope_desc: str,
    ) -> None:
        if img is None:
            return
        try:
            # 기존 동작 유지
            self.blemish_last_img_full = img
            self.blemish_last_candidates = cand
            self.blemish_last_ref_mean = ref_mean

            self._populate_table()
            self._render_points_layer(cand, tgt)

            # CropΔ threshold band (L <= abs(CropΔ%) < U)인 경우: list 표시 + 확인 팝업
            # 단, abs(CropΔ%) >= Thr.U 인 blemish가 하나라도 있으면(=PASS 불가) 확인 팝업은 띄우지 않음.
            try:
                # Thr.U (CropΔ%) 읽기
                try:
                    cropdiff_u = float(
                        self.blemish_params.get(
                            "threshold_ref_percent_u",
                            self.blemish_params.get("threshold_ref_percent", 0.0),
                        )
                    )
                except Exception:
                    cropdiff_u = float(self.blemish_params.get("threshold_ref_percent", 0.0) or 0.0)

                any_over_u = False
                if cropdiff_u > 0:
                    any_over_u = any(
                        abs(float(getattr(c, "crop_diff_percent", 0.0) or 0.0)) >= cropdiff_u
                        for c in (cand or [])
                    )

                if (not any_over_u) and any(getattr(c, "needs_pit_check", False) for c in (cand or [])):
                    show_info(self, "확인 필요", "패임/돌출 확인 필요! 없으면 PASS 가능한 blemish!")
            except Exception:
                pass

            if not cand:
                show_info(self, "Info", "No blemish detected")
            else:
                self._render_candidate_crop(0)

            self._set_status(f"Blemish: {len(cand)} found | target={tgt} | scope={scope_desc}")
        except Exception as e:
            show_err(self, "Detect Error", str(e))

    def _on_detect(self) -> None:
        if self.raw_data is None:
            return
        tgt = self.cb_target.currentText()
        img = self.processed_data if (tgt == "Corrected" and self.processed_data is not None) else self.raw_data
        if img is None:
            return

        # cache check (cache hit이면 즉시 UI 반영)
        cache_key = self._make_detect_cache_key(img, tgt)
        cached = self._detect_cache_get(cache_key)
        if cached is not None:
            self._update_blemish_text_overlay(clear_only=True)
            cand, ref_mean, scope_desc = cached
            self._apply_detect_result(img=img, tgt=tgt, cand=cand, ref_mean=float(ref_mean), scope_desc=str(scope_desc))
            return

        # snapshot params/roi for this run (thread-safe)
        params_copy = dict(self.blemish_params) if isinstance(self.blemish_params, dict) else {}
        roi_copy = None
        try:
            rb = self.roi_bounds
            if rb is not None and getattr(rb, "valid", False):
                roi_copy = RoiBounds(int(rb.x1), int(rb.x2), int(rb.y1), int(rb.y2))
        except Exception:
            roi_copy = None

        # if running, queue the latest request (keep current view until new result is ready)
        if self._detect_running:
            self._detect_queue = (str(tgt), img, params_copy, roi_copy, cache_key)
            self._set_status(f"Detect queued... | target={tgt}")
            return

        # 수동 Detect 시작 시 기존 텍스트 오버레이를 먼저 정리(중복 방지)
        self._update_blemish_text_overlay(clear_only=True)
        self._start_detect_job(img, str(tgt), params_copy, roi_copy, cache_key)
    def _populate_table(self) -> None:
        self.tbl.setRowCount(0)

        # area unit: px^2 (pixel count)

        for c in self.blemish_last_candidates:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)

            self.tbl.setItem(r, 0, QTableWidgetItem(f"({c.x},{c.y})"))

            # Thr(%) = DROP metric (blemish boundary 기반 대표 drop[%])
            try:
                thr_val = float(getattr(c, "drop_percent", 0.0) or 0.0)
            except Exception:
                thr_val = 0.0
            self.tbl.setItem(r, 1, QTableWidgetItem(f"{thr_val:.1f}%"))

            # CropΔ% = (pixel - local mean)/local mean * 100
            self.tbl.setItem(r, 2, QTableWidgetItem(f"{int(round(float(c.crop_diff_percent)))}%"))

            area_px = int(getattr(c, "area_px", 0) or 0)
            self.tbl.setItem(r, 3, QTableWidgetItem(f"{area_px}"))

        if self.tbl.rowCount() > 0:
            self.tbl.selectRow(0)

    def _on_table_clicked(self, row: int, _col: int) -> None:
        self._render_candidate_crop(row)

    def _render_candidate_crop(self, idx: int) -> None:
        if self.blemish_last_img_full is None:
            return
        if not (0 <= idx < len(self.blemish_last_candidates)):
            return

        c = self.blemish_last_candidates[idx]
        crop, crop_origin = extract_crop(self.blemish_last_img_full, c.x, c.y, size=50)

        self._ensure_layers()
        self.layer_crop.data = crop.astype(np.float32, copy=False) if crop.size else np.zeros((10, 10), np.float32)
        # crop은 개별적으로 L/W를 쓰는 경우가 많아서 여기서는 auto만 갱신
        self.layer_crop.contrast_limits = percentile_clim(self.layer_crop.data)

        # ---- Crop preview outline (disabled)
        # Crop preview에서 붉은 테두리(Outline)는 요청에 따라 완전히 제거합니다.
        # 혹시 이전 실행에서 남아있는 'Crop Outline' 레이어가 있다면 정리만 합니다.
        _remove_layers_by_prefix(self.model_crop, "Crop Outline")

        # ---- Crop preview blemish area (요청: 테두리 X, 실제 blemish 영역 픽셀만 빨강/50% 투명도)
        # 이전 버전 레이어 정리
        _remove_layers_by_prefix(self.model_crop, "Crop Blemish Border")
        _remove_layers_by_prefix(self.model_crop, "Crop Blemish Area")

        try:
            if crop.size and c.mask is not None and c.mask_origin is not None:
                ch, cw = int(crop.shape[0]), int(crop.shape[1])
                x0c, y0c = int(crop_origin[0]), int(crop_origin[1])

                mx0, my0 = int(c.mask_origin[0]), int(c.mask_origin[1])
                m = (c.mask.astype(bool)).copy()
                mh, mw = int(m.shape[0]), int(m.shape[1])

                # overlap (full coords)
                ox0 = max(mx0, x0c)
                oy0 = max(my0, y0c)
                ox1 = min(mx0 + mw, x0c + cw)
                oy1 = min(my0 + mh, y0c + ch)

                if ox1 > ox0 and oy1 > oy0:
                    mask_crop = np.zeros((ch, cw), dtype=bool)
                    mask_crop[(oy0 - y0c):(oy1 - y0c), (ox0 - x0c):(ox1 - x0c)] = m[(oy0 - my0):(oy1 - my0), (ox0 - mx0):(ox1 - mx0)]
                    labels = np.zeros((ch, cw), dtype=np.uint8)
                    labels[mask_crop] = 1  # 0=transparent, 1=blemish

                    try:
                        lay = self.model_crop.add_labels(
                            labels,
                            name="Crop Blemish Area",
                            opacity=0.5,  # 50%
                            color={1: "red"},
                        )
                    except Exception:
                        lay = self.model_crop.add_labels(labels, name="Crop Blemish Area", opacity=0.5)
                        try:
                            lay.color = {1: "red"}
                        except Exception:
                            pass

                    try:
                        lay.interactive = False
                    except Exception:
                        pass

                    # 레이어가 뒤로 깔리는 케이스 방지: 항상 최상단으로 이동
                    try:
                        self.model_crop.layers.move(self.model_crop.layers.index(lay), len(self.model_crop.layers) - 1)
                    except Exception:
                        pass
        except Exception:
            pass

        # ---- Crop preview text label (selected blemish only)
        # 가독성을 위해 검정 그림자 + 노란 글자 2겹으로 표시합니다.
        _remove_layers_by_prefix(self.model_crop, "Crop Blemish Label")
        _remove_layers_by_prefix(self.model_crop, "Crop Blemish Label Shadow")
        
        try:
            offset = int(self.blemish_params.get("overlay_bottom_margin_px", 5))
        except Exception:
            offset = 5
        
        try:
            cx = float(c.x - crop_origin[0])
            cy = float(c.y - crop_origin[1])
            pos = np.array([[cy + float(offset), cx]], dtype=np.float32)  # (y,x)
            shadow = pos.copy()
            shadow[:, 0] += 1.0
            shadow[:, 1] += 1.0
            thr_val = float(getattr(c, "drop_percent", 0.0) or 0.0)
            label = f"({int(c.x)},{int(c.y)})  Thr {thr_val:.2f}%"
        
            try:
                tsize = int(self.blemish_params.get("overlay_text_size", 18))
            except Exception:
                tsize = 18
        
            def _add_crop_text(name: str, pts: np.ndarray, color: str) -> None:
                try:
                    ll = self.model_crop.add_points(
                        pts,
                        name=name,
                        size=1,
                        face_color=np.array([[0, 0, 0, 0]], dtype=np.float32),
                        edge_color=np.array([[0, 0, 0, 0]], dtype=np.float32),
                        opacity=1.0,
                    )
                except Exception:
                    ll = self.model_crop.add_points(
                        pts,
                        name=name,
                        size=1,
                        face_color="transparent",
                        edge_color="transparent",
                        opacity=1.0,
                    )
                try:
                    ll.interactive = False
                except Exception:
                    pass
                try:
                    ll.features = {"label": [label]}
                except Exception:
                    try:
                        import pandas as pd
                        ll.features = pd.DataFrame({"label": [label]})
                    except Exception:
                        pass
                try:
                    ll.text = {"string": "{label}", "color": color, "size": int(tsize), "anchor": "upper_left", "visible": True}
                    try:
                        ll.text.visible = True
                    except Exception:
                        pass
                except Exception:
                    try:
                        ll.text.values = [label]
                        ll.text.color = color
                        ll.text.size = int(tsize)
                        ll.text.visible = True
                    except Exception:
                        pass
        
            _add_crop_text("Crop Blemish Label Shadow", shadow, "black")
            _add_crop_text("Crop Blemish Label", pos, "yellow")
        except Exception:
            pass
        
        # Crop도 항상 캔버스에 꽉 차게
        try:
            self.model_crop.reset_view(margin=0.0)
        except Exception:
            try:
                self.model_crop.reset_view()
            except Exception:
                pass

    def _render_points_layer(self, candidates: List[BlemishCandidate], target: str) -> None:
        """요청사항 반영:
        - Raw에는 붉은 테두리 표시하지 않음
        - Result에만 "아주 얇은 빨간 선"으로 blemish area의 외곽선(path) 표시
        """

        # 1) Raw는 항상 오버레이 제거
        _remove_layers_by_prefix(self.model_raw, "Blemish Outline")

        # 2) Result의 기존 오버레이 제거
        _remove_layers_by_prefix(self.model_result, "Blemish Outline")

        if not candidates:
            self._update_blemish_text_overlay(clear_only=True)
            return

        paths: List[np.ndarray] = []

        for c in candidates:
            if c.mask is not None and c.mask_origin is not None:
                contour = _trace_binary_boundary(c.mask)
                if contour is not None and contour.shape[0] >= 4:
                    # full-image 좌표로 오프셋 (y,x)
                    contour = contour.copy()
                    contour[:, 0] += float(c.mask_origin[1])
                    contour[:, 1] += float(c.mask_origin[0])
                    paths.append(contour.astype(np.float32))
                    continue

            # fallback: mask가 없으면 50x50 bbox
            x0, x1 = float(c.x - 25), float(c.x + 25)
            y0, y1 = float(c.y - 25), float(c.y + 25)
            paths.append(np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0], [y0, x0]], dtype=np.float32))
        # (요청에 따라) Blemish Outline 오버레이는 생성하지 않음

        # 대신: 위치 기반 텍스트 라벨/마커 표시
        self._update_blemish_text_overlay(clear_only=False)

    def _on_save_crop(self) -> None:
        if not self.blemish_last_candidates or self.blemish_last_img_full is None:
            return

        # 선택된 row들
        sel = self.tbl.selectionModel().selectedRows()
        if not sel:
            return
        idxs = sorted({s.row() for s in sel})

        # 저장 경로
        default_dir = self.blemish_save_dir or os.getcwd()
        out_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory", default_dir)
        if not out_dir:
            return

        # 이미지 저장: napari[all] 환경이면 imageio가 함께 오는 경우가 많지만, 없을 수도 있음.
        # 그러면 np.save 같은 대체가 필요합니다.
        try:
            import imageio.v3 as iio
        except Exception as e:
            show_err(self, "Missing dependency", f"imageio가 필요합니다. (napari[all] 설치 권장)\n{e}")
            return

        cnt = 0
        for idx in idxs:
            if not (0 <= idx < len(self.blemish_last_candidates)):
                continue
            c = self.blemish_last_candidates[idx]
            crop, _ = extract_crop(self.blemish_last_img_full, c.x, c.y, size=50)
            if crop.size == 0:
                continue

            up = np.repeat(np.repeat(crop, 5, axis=0), 5, axis=1)
            # 8-bit jpg 저장을 위해 정규화
            vmin, vmax = percentile_clim(up)
            img8 = np.clip((up - vmin) / (vmax - vmin + 1e-6) * 255.0, 0, 255).astype(np.uint8)
            # 요청: value 낮은 부분=white, 높은 부분=black으로 저장 (grayscale invert)
            img8 = (255 - img8).astype(np.uint8)

            fn = os.path.join(out_dir, f"Blemish_{int(round(c.crop_diff_percent))}%_({c.x},{c.y}).jpg")
            iio.imwrite(fn, img8)
            cnt += 1

        show_info(self, "Saved", f"{cnt} images saved")

    # ---------------------------
    # Save result
    # ---------------------------

    def _on_save_result(self) -> None:
        if self.processed_data is None:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Result", "", "IMG (*.IMG *.img);;All (*.*)")
        if not out_path:
            return
        try:
            img_to_save = self.processed_data
            # 요청: 1616/1624/1824는 로딩 시 rotate left + flip horizontal로 표시/처리를 진행함.
            # Save Result는 그 이전(원본 좌표계) 상태로 저장해야 하므로 역변환 후 저장.
            if (self.current_model or "") in ORIENT_ROTATE_LEFT_FLIP_H_MODELS:
                try:
                    img_to_save = orient_inverse_image(self.current_model or "", img_to_save)
                except Exception:
                    pass
            np.clip(img_to_save, 0, 65535).astype(np.uint16).tofile(out_path)
            show_info(self, "Saved", "File saved!")
        except Exception as e:
            show_err(self, "Save Error", str(e))


# ---------------------------
# App runner (napari dock widget)
# ---------------------------

def main() -> None:
    # Multiple viewers를 한 윈도우에 넣을 때 OpenGL context 공유가 필요할 수 있습니다.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    # napari가 사용하는 QApplication을 가져오거나 생성
    try:
        from napari.qt import get_qapp
        app = get_qapp()
    except Exception:
        app = QApplication.instance() or QApplication([])

    
    # --- Theme (요청: 초기의 '푸른계열' napari dark 테마로 복귀) ---
    # 일부 napari 버전에서는 napari.Viewer(theme=...) 인자가 없습니다.
    # 그 경우 settings를 통해 테마를 먼저 'dark'로 지정합니다.
    try:
        # napari 0.4.x~0.6.x에서 제공되는 경우가 많음
        from napari.utils.theme import set_theme  # type: ignore
        set_theme('dark')
    except Exception:
        pass
    try:
        from napari.settings import get_settings  # type: ignore

        s = get_settings()
        # napari 0.4.x~0.6.x 계열에서 appearance.theme 형태가 흔합니다.
        if hasattr(s, 'appearance') and hasattr(s.appearance, 'theme'):
            s.appearance.theme = 'dark'
        # (구버전/다른 구조) 대비
        elif hasattr(s, 'application') and hasattr(s.application, 'theme'):
            s.application.theme = 'dark'
    except Exception:
        pass

    # host napari viewer (dock 호스트로만 사용) - 화면에는 숨김
    host = napari.Viewer()
    try:
        # 기본 도킹/메뉴 제거(불필요 UI)
        host.window.remove_dock_widget('all')
    except Exception:
        pass
    try:
        host.window.main_menu.setVisible(False)
    except Exception:
        pass
    try:
        mw = getattr(host.window, '_qt_window', None)
        if mw is not None:
            mw.hide()
    except Exception:
        pass

    # 작업표시줄에 남는 '진짜' 최상위 창
    from qtpy.QtGui import QGuiApplication
    from qtpy.QtWidgets import QMainWindow

    win = QMainWindow()
    win.setWindowTitle("이미지 판정 프로그램 ver1.0")

    # 사용자 해상도 가로폭(작업표시줄 제외)과 동일하게 시작
    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        geo = screen.availableGeometry()
        win.resize(geo.width(), int(geo.height() * 0.92))
        win.move(geo.x(), geo.y())
    else:
        win.resize(1600, 900)

    widget = XrayNapariWidget(host)
    widget.apply_default_splitter_sizes()
    win.setCentralWidget(widget)
    win.show()

    # 이벤트 루프 실행
    try:
        app.exec()
    except AttributeError:
        app.exec_()


if __name__ == "__main__":
    main()
