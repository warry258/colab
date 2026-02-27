"""
头颅 CTA 减影-whw 版 V3.3.1 (分段减影版) - 定制修改版
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[定制] 移除了 血管缩放 功能
[定制] 移除了 MIP预览 功能
[终极优化] 增加分段减影模式 - 彻底解决 32位系统 600层+ 内存溢出
[优化] int16 存储体数据 - 节省约 300MB
[优化] 钙化检测改 2.5D 滑动窗口 - 避免 600MB 峰值
[优化] 交互标注降采样显示 - 节省约 300MB
[优化] top_hat 按需逐层计算 - 节省约 300MB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import re
import sys
import json
import gc
import warnings
import ctypes
import numpy as np
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set
import time
from datetime import datetime
from collections import defaultdict
from scipy import ndimage
from dataclasses import dataclass, field
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    import SimpleITK as sitk
except ImportError:
    print("错误：找不到 SimpleITK 库。请运行 'pip install SimpleITK'")
    sys.exit(1)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QProgressBar,
    QTextEdit, QGroupBox, QMessageBox, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QSlider, QRadioButton, QFrame, QDialog,
    QButtonGroup, QSizePolicy, QDesktopWidget, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt5.QtGui import (
    QFont, QImage, QPixmap, QIcon, QPainter, QPen, QBrush, QColor, QPolygonF
)
import pydicom
from pydicom.uid import generate_uid
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("提示：pip install shapely 可启用多边形融合")
print("头颅 CTA 减影-whw 版 V4.3.1 ")

# ════════════════════════════════════════════════════════════════════
# 交互标注数据结构
# ════════════════════════════════════════════════════════════════════
@dataclass
class PolygonROI:
    """多边形 ROI"""
    points: List[QPointF] = field(default_factory=list)
    closed: bool = False
    roi_type: str = "local"
    slice_index: int = 0

    def copy(self) -> 'PolygonROI':
        return PolygonROI(
            points=[QPointF(p.x(), p.y()) for p in self.points],
            closed=self.closed,
            roi_type=self.roi_type,
            slice_index=self.slice_index
        )

    def scale(self, factor: float) -> 'PolygonROI':
        """缩放 ROI 坐标"""
        return PolygonROI(
            points=[QPointF(p.x() * factor, p.y() * factor) for p in self.points],
            closed=self.closed,
            roi_type=self.roi_type,
            slice_index=self.slice_index
        )

    def to_shapely(self):
        if not HAS_SHAPELY or len(self.points) < 3:
            return None
        coords = [(p.x(), p.y()) for p in self.points]
        try:
            poly = ShapelyPolygon(coords)
            if not poly.is_valid:
                poly = make_valid(poly)
            return poly if poly.is_valid and not poly.is_empty else None
        except:
            return None

    @staticmethod
    def from_shapely(shapely_poly, roi_type: str, slice_index: int) -> Optional['PolygonROI']:
        if shapely_poly is None or shapely_poly.is_empty:
            return None
        try:
            if shapely_poly.geom_type != 'Polygon':
                return None 
            coords = list(shapely_poly.exterior.coords)[:-1]
            if len(coords) < 3:
                return None
            return PolygonROI(
                points=[QPointF(x, y) for x, y in coords],
                closed=True, roi_type=roi_type, slice_index=slice_index
            )
        except:
            return None

@dataclass
class DrawingState:
    """跨层绘制状态"""
    is_drawing: bool = False
    start_slice: int = 0
    current_slice: int = 0
    visited_slices: Set[int] = field(default_factory=set)
    polygon: Optional[PolygonROI] = None

    def reset(self):
        self.is_drawing = False
        self.start_slice = 0
        self.current_slice = 0
        self.visited_slices.clear()
        self.polygon = None

    def get_slice_range(self) -> Tuple[int, int]:
        if not self.visited_slices:
            return (self.start_slice, self.start_slice)
        return (min(self.visited_slices), max(self.visited_slices))

    def get_slice_count(self) -> int:
        min_z, max_z = self.get_slice_range()
        return max_z - min_z + 1

@dataclass
class AnnotationData:
    """标注数据"""
    global_rois: List[PolygonROI] = field(default_factory=list)
    local_rois: Dict[int, List[PolygonROI]] = field(default_factory=lambda: defaultdict(list))
    inverse_global_rois: List[PolygonROI] = field(default_factory=list)
    inverse_local_rois: Dict[int, List[PolygonROI]] = field(default_factory=lambda: defaultdict(list))
    z_range_start: int = 0
    z_range_end: int = -1
    series_uid: str = ""
    scale_factor: float = 1.0

    def get_scaled_rois(self, target_scale: float) -> 'AnnotationData':
        """获取缩放后的 ROI 用于实际处理"""
        if abs(self.scale_factor - target_scale) < 0.001:
            return self
        
        ratio = target_scale / self.scale_factor
        scaled = AnnotationData(
            z_range_start=self.z_range_start,
            z_range_end=self.z_range_end,
            series_uid=self.series_uid,
            scale_factor=target_scale
        )
        
        for roi in self.global_rois:
            scaled.global_rois.append(roi.scale(ratio))
        for z, rois in self.local_rois.items():
            for roi in rois:
                scaled.local_rois[z].append(roi.scale(ratio))

        for roi in self.inverse_global_rois:
            scaled.inverse_global_rois.append(roi.scale(ratio))
        for z, rois in self.inverse_local_rois.items():
            for roi in rois:
                scaled.inverse_local_rois[z].append(roi.scale(ratio))

        return scaled

# ════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════
def get_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def force_gc():
    gc.collect()
    gc.collect()

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_next_output_dir(base_dir, prefix):
    idx = 1
    while True:
        dir_name = f"{prefix}{idx:03d}"
        full_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(full_path):
            return full_path
        idx += 1
        if idx > 999:
            return os.path.join(base_dir, f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# ════════════════════════════════════════════════════════════════════
# 多边形工具函数
# ════════════════════════════════════════════════════════════════════
def point_to_line_distance(p: QPointF, a: QPointF, b: QPointF) -> Tuple[float, float]:
    dx, dy = b.x() - a.x(), b.y() - a.y()
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        return math.sqrt((p.x() - a.x())**2 + (p.y() - a.y())**2), 0.0
    t = max(0, min(1, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / length_sq))
    proj_x, proj_y = a.x() + t * dx, a.y() + t * dy
    return math.sqrt((p.x() - proj_x)**2 + (p.y() - proj_y)**2), t

def find_nearest_vertex(point: QPointF, polygon: PolygonROI, threshold: float = 10.0) -> int:
    min_dist, best_idx = float('inf'), -1
    for i, p in enumerate(polygon.points):
        dist = math.sqrt((point.x() - p.x())**2 + (point.y() - p.y())**2)
        if dist < min_dist:
            min_dist, best_idx = dist, i
    return best_idx if min_dist <= threshold else -1

def find_nearest_edge(point: QPointF, polygon: PolygonROI) -> Tuple[int, float]:
    if len(polygon.points) < 2:
        return -1, 0
    min_dist, best_edge, best_t = float('inf'), -1, 0
    n = len(polygon.points)
    for i in range(n):
        a, b = polygon.points[i], polygon.points[(i + 1) % n]
        dist, t = point_to_line_distance(point, a, b)
        if dist < min_dist:
            min_dist, best_edge, best_t = dist, i, t
    return best_edge, best_t

def merge_overlapping_polygons(rois: List[PolygonROI], roi_type: str, slice_index: int) -> List[PolygonROI]:
    if not HAS_SHAPELY or len(rois) <= 1:
        return rois
    shapely_polys =[]
    for roi in rois:
        sp = roi.to_shapely()
        if sp is not None:
            shapely_polys.append(sp)

    if len(shapely_polys) <= 1:
        return rois

    try:
        merged = unary_union(shapely_polys)
    except Exception as e:
        print(f"融合错误：{e}")
        return rois

    result =[]
    if merged.is_empty:
        return rois

    if merged.geom_type == 'Polygon':
        roi = PolygonROI.from_shapely(merged, roi_type, slice_index)
        if roi:
            result.append(roi)
    elif merged.geom_type == 'MultiPolygon':
        for poly in merged.geoms:
            roi = PolygonROI.from_shapely(poly, roi_type, slice_index)
            if roi:
                result.append(roi)
    elif merged.geom_type == 'GeometryCollection':
        for geom in merged.geoms:
            if geom.geom_type == 'Polygon':
                roi = PolygonROI.from_shapely(geom, roi_type, slice_index)
                if roi:
                    result.append(roi)

    return result if result else rois

# ════════════════════════════════════════════════════════════════════
# 优化级别与预设
# ════════════════════════════════════════════════════════════════════
OPTIMIZATION_LEVELS = {
    "none": { "name": "无优化", "description": "原生模式" },
    "light": { "name": "轻量级", "description": "局部自适应 + ICA 精确保护" },
    "standard": { "name": "标准级", "description": "+ 亚像素精修 + 静脉窦" },
    "deep": { "name": "深度级", "description": "+ Top-Hat 血管增强" },
}
PRESET_FILE = Path(__file__).parent / "device_presets.json"
_MANUFACTURER_MAP = {
    ("siemens",): "Siemens SOMATOM",
    ("ge", "ge", "general electric"): "GE Revolution/Discovery",
    ("philips",): "Philips Brilliance/IQon",
    ("canon", "toshiba"): "Canon Aquilion",
    ("united imaging", "联影", "uih"): "联影 uCT",
}
BUILTIN_PRESETS = {
    "通用默认": {
        "bone_strength": 1.2, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.7, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400, "preserve_calcification": True
    },
    "Siemens SOMATOM": {
        "bone_strength": 1.2, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400, "preserve_calcification": True
    },
    "GE Revolution/Discovery": {
        "bone_strength": 1.4, "vessel_sensitivity": 0.9, "vessel_enhance": 1.8,
        "smooth_sigma": 0.9, "clean_bone_edges": True, "min_vessel_size": 6,
        "wc": 220, "ww": 420, "preserve_calcification": True
    },
    "Philips Brilliance/IQon": {
        "bone_strength": 1.0, "vessel_sensitivity": 1.1, "vessel_enhance": 2.2,
        "smooth_sigma": 0.5, "clean_bone_edges": True, "min_vessel_size": 4,
        "wc": 180, "ww": 380, "preserve_calcification": True
    },
    "Canon Aquilion": {
        "bone_strength": 1.1, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": False, "min_vessel_size": 5,
        "wc": 200, "ww": 400, "preserve_calcification": True
    },
    "联影 uCT": {
        "bone_strength": 1.3, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.8, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400, "preserve_calcification": True
    },
}

def load_all_presets() -> dict:
    merged = dict(BUILTIN_PRESETS)
    if PRESET_FILE.exists():
        try:
            with open(PRESET_FILE, "r", encoding="utf-8") as f:
                merged.update(json.load(f))
        except:
            pass
    return merged

def detect_manufacturer_preset(files: list) -> Optional[str]:
    if not files:
        return None
    try:
        ds = pydicom.dcmread(files[0][1], stop_before_pixels=True)
        combined = (str(getattr(ds, "Manufacturer", "")) + " " + str(getattr(ds, "ManufacturerModelName", ""))).lower()
        for keywords, preset_name in _MANUFACTURER_MAP.items():
            if any(kw in combined for kw in keywords):
                return preset_name
    except:
        pass
    return None

# ════════════════════════════════════════════════════════════════════
# 序列扫描与分析
# ════════════════════════════════════════════════════════════════════
class SeriesInfo:
    def __init__(self):
        self.series_uid, self.series_number, self.series_description = "", 0, ""
        self.study_description, self.modality, self.body_part, self.protocol_name = "", "", "", ""
        self.acquisition_time, self.file_count, self.files = None, 0,[]
        self.slice_thickness, self.image_shape, self.manufacturer = 0, (0, 0), ""
        self._contrast_status, self._contrast_cached = None, False

    @property
    def contrast_status(self):
        if not self._contrast_cached:
            desc = self.series_description.upper()
            pos =[r'\bC\+', r'\bC\s*\+', r'CE\b', r'CONTRAST', r'ENHANCED', r'POST', r'ARTERIAL', r'增强', r'动脉期']
            neg =[r'\bC-', r'\bC\s*-', r'\bNC\b', r'NON-CONTRAST', r'PLAIN', r'PRE\b', r'WITHOUT', r'平扫', r'非增强']
            self._contrast_status = True if any(re.search(p, desc) for p in pos) else False if any(re.search(p, desc) for p in neg) else None
            self._contrast_cached = True
        return self._contrast_status

def analyze_dicom_file(filepath):
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        time_str = str(getattr(ds, 'AcquisitionTime', '')).split('.')[0]
        acq_time = datetime.strptime(time_str[:6], "%H%M%S") if len(time_str) >= 6 else None
        return {
            'filepath': filepath, 'series_uid': str(getattr(ds, 'SeriesInstanceUID', '')),
            'series_number': int(getattr(ds, 'SeriesNumber', 0)), 'series_description': str(getattr(ds, 'SeriesDescription', '')),
            'study_description': str(getattr(ds, 'StudyDescription', '')), 'modality': str(getattr(ds, 'Modality', '')),
            'body_part': str(getattr(ds, 'BodyPartExamined', '')), 'protocol_name': str(getattr(ds, 'ProtocolName', '')),
            'instance_number': int(getattr(ds, 'InstanceNumber', 0)), 'slice_thickness': float(getattr(ds, 'SliceThickness', 0)),
            'rows': int(getattr(ds, 'Rows', 0)), 'columns': int(getattr(ds, 'Columns', 0)),
            'acquisition_time': acq_time, 'manufacturer': str(getattr(ds, 'Manufacturer', '')),
        }
    except:
        return None

def scan_directory_for_series(directory, progress_callback=None, log_callback=None):
    all_files = list(Path(directory).rglob('*'))
    dicom_files =[]
    for f in all_files:
        if f.is_file():
            try:
                with open(f, 'rb') as fp:
                    fp.seek(128)
                    if fp.read(4) == b'DICM':
                        dicom_files.append(f)
            except:
                pass
    if not dicom_files:
        return {}
    series_dict = defaultdict(SeriesInfo)
    for i, f in enumerate(dicom_files):
        info = analyze_dicom_file(str(f))
        if info and info['series_uid']:
            s = series_dict[info['series_uid']]
            if s.file_count == 0:
                s.series_uid, s.series_number, s.series_description = info['series_uid'], info['series_number'], info['series_description']
                s.study_description, s.modality, s.body_part = info['study_description'], info['modality'], info['body_part']
                s.protocol_name, s.acquisition_time = info['protocol_name'], info['acquisition_time']
                s.slice_thickness, s.image_shape, s.manufacturer = info['slice_thickness'], (info['rows'], info['columns']), info['manufacturer']
            s.files.append((info['instance_number'], str(f)))
            s.file_count += 1
            if info['acquisition_time'] and (s.acquisition_time is None or info['acquisition_time'] < s.acquisition_time):
                s.acquisition_time = info['acquisition_time']
            if progress_callback:
                progress_callback(int((i + 1) / len(dicom_files) * 100))
    for s in series_dict.values():
        s.files.sort(key=lambda x: x[0])
    return dict(series_dict)

def is_head_cta_series(series):
    desc = f"{series.series_description} {series.study_description} {series.protocol_name} {series.body_part}".upper()
    if any(kw in desc for kw in['SCOUT', 'LOCALIZER', 'TOPOGRAM', '定位', 'LUNG', 'CHEST', '肺', '胸', 'CARDIAC', 'HEART', '心', 'ABDOMEN', 'LIVER', '腹']):
        return False
    if series.modality != 'CT' or series.file_count < 50:
        return False
    has_head = any(kw in desc for kw in['HEAD', 'BRAIN', 'CEREBR', 'CRANIAL', '头', '颅', '脑', 'CAROTID'])
    has_cta = any(kw in desc for kw in['CTA', 'ANGIO', 'C+', 'C-', '血管', '动脉'])
    return (has_head and has_cta) or (has_head and series.file_count >= 100)

def find_cta_pairs(series_dict):
    cta_series =[s for s in series_dict.values() if is_head_cta_series(s)]
    if len(cta_series) < 2:
        return[], cta_series
    groups = defaultdict(list)
    for s in cta_series:
        groups[(s.file_count, round(s.slice_thickness, 1))].append(s)
    pairs, used =[], set()
    for group in groups.values():
        enhanced, plain, unknown = [], [],[]
        for s in group:
            st = s.contrast_status
            if st is True:
                enhanced.append(s)
            elif st is False:
                plain.append(s)
            else:
                unknown.append(s)
        for pre in plain:
            for post in enhanced:
                if pre.series_uid not in used and post.series_uid not in used:
                    pairs.append((pre, post))
                    used.update([pre.series_uid, post.series_uid])
                    break
            if pre.series_uid in used:
                break
    if not pairs and len(unknown) >= 2:
        unknown.sort(key=lambda s: (s.acquisition_time or datetime.max, s.series_number))
        pairs.append((unknown[0], unknown[1]))
    return pairs, cta_series

class SeriesScanThread(QThread):
    progress, log, finished_signal = pyqtSignal(int), pyqtSignal(str), pyqtSignal(dict, list, list)
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def run(self):
        try:
            self.log.emit("=" * 55 + f"\n扫描：{self.directory}\n" + "=" * 55)
            series_dict = scan_directory_for_series(self.directory, self.progress.emit, self.log.emit)
            self.log.emit(f"\n找到 {len(series_dict)} 个序列\n")
            for s in sorted(series_dict.values(), key=lambda s: s.series_number):
                t = s.acquisition_time.strftime("%H:%M:%S") if s.acquisition_time else "--:--:--"
                m = "C+" if s.contrast_status is True else "C-" if s.contrast_status is False else "★" if is_head_cta_series(s) else "   "
                self.log.emit(f"{m} #{s.series_number:3d} | {s.file_count:4d}张 | {t} | {s.series_description[:35]}")
            pairs, cta_series = find_cta_pairs(series_dict)
            if pairs:
                self.log.emit(f"\n★ 自动配对:\n  平扫：#{pairs[0][0].series_number}\n  增强：#{pairs[0][1].series_number}")
            self.progress.emit(100)
            self.finished_signal.emit(series_dict, pairs, cta_series)
        except Exception as e:
            self.log.emit(f"[错误] {e}")
            self.finished_signal.emit({}, [],[])

# ════════════════════════════════════════════════════════════════════
# 配准相关算法
# ════════════════════════════════════════════════════════════════════
def fft_phase_correlation_2d(fixed, moving, max_shift=15):
    from numpy.fft import fft2, ifft2, fftshift
    margin = max(fixed.shape[0] // 6, 30)
    f_roi = fixed[margin:-margin, margin:-margin].astype(np.float32)
    m_roi = moving[margin:-margin, margin:-margin].astype(np.float32)
    window = np.outer(np.hanning(f_roi.shape[0]), np.hanning(f_roi.shape[1])).astype(np.float32)
    f1, f2 = fft2(f_roi * window), fft2(m_roi * window)
    correlation = np.real(fftshift(ifft2((f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10))))
    py, px = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy, dx = float(py - correlation.shape[0] // 2), float(px - correlation.shape[1] // 2)
    return float(np.clip(dy, -max_shift, max_shift)), float(np.clip(dx, -max_shift, max_shift))

def shift_image_2d(image, dy, dx):
    return ndimage.shift(image.astype(np.float32), [dy, dx], order=1, mode='constant', cval=0)

def trim_to_common_z_range(files_a, files_b, log_cb=None):
    if len(files_a) == len(files_b):
        return files_a, files_b

    n_a, n_b = len(files_a), len(files_b)
    n = min(n_a, n_b)          

    if n_a > n_b:
        trimmed_a = files_a[-n:]
        trimmed_b = files_b
        if log_cb:
            log_cb(f"  [层数对齐] 平扫 ({n_a}层) > 增强 ({n_b}层)，取末尾 {n} 层")
    else:
        trimmed_a = files_a
        trimmed_b = files_b[-n:]
        if log_cb:
            log_cb(f"  [层数对齐] 增强 ({n_b}层) > 平扫 ({n_a}层)，取末尾 {n} 层")

    return trimmed_a, trimmed_b

def build_z_matched_file_pairs(pre_files_tuples, post_files_tuples, log_cb=None):
    pre_dict_inst = {i: p for i, p in pre_files_tuples}
    post_dict_inst = {i: p for i, p in post_files_tuples}
    common_inst = sorted(set(pre_dict_inst) & set(post_dict_inst))
    if common_inst:
        return common_inst, pre_dict_inst, post_dict_inst

    if log_cb:
        log_cb("  [参数分析] instance number 无重叠，改用 Z 坐标匹配采样层")

    def get_z(path):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp:
                return float(ipp[2])
            sl = getattr(ds, 'SliceLocation', None)
            return float(sl) if sl is not None else None
        except Exception:
            return None

    pre_z_list =[(get_z(p), i, p) for i, p in pre_files_tuples]
    post_z_list =[(get_z(p), i, p) for i, p in post_files_tuples]
    pre_z_list =[(z, i, p) for z, i, p in pre_z_list if z is not None]
    post_z_list =[(z, i, p) for z, i, p in post_z_list if z is not None]

    if not pre_z_list or not post_z_list:
        return[], {}, {}

    pre_z_list.sort(key=lambda x: x[0])
    post_z_list.sort(key=lambda x: x[0])

    z_min = max(pre_z_list[0][0], post_z_list[0][0])
    z_max = min(pre_z_list[-1][0], post_z_list[-1][0])

    if z_min > z_max:
        return [], {}, {}

    sample_z =[z_min + (z_max - z_min) * t for t in[0.25, 0.40, 0.50, 0.60, 0.75]]
    pre_arr = [z for z, _, _ in pre_z_list]
    post_arr =[z for z, _, _ in post_z_list]

    synthetic_common =[]
    new_pre_dict, new_post_dict = {}, {}
    key = 10000 

    for sz in sample_z:
        pi = int(np.argmin(np.abs(np.array(pre_arr) - sz)))
        qi = int(np.argmin(np.abs(np.array(post_arr) - sz)))
        new_pre_dict[key] = pre_z_list[pi][2]
        new_post_dict[key] = post_z_list[qi][2]
        synthetic_common.append(key)
        key += 1

    return synthetic_common, new_pre_dict, new_post_dict

class ParamAnalyzeThread(QThread):
    progress, log, finished_signal = pyqtSignal(int), pyqtSignal(str), pyqtSignal(dict)
    def __init__(self, pre_files, post_files):
        super().__init__()
        self.pre_files, self.post_files = pre_files, post_files

    def run(self):
        try:
            self.log.emit("\n" + "=" * 55 + "\n智能参数分析 (多维图像质量评估)\n" + "=" * 55)

            common, pre_dict, post_dict = build_z_matched_file_pairs(
                self.pre_files, self.post_files, self.log.emit
            )
            if not common:
                return self.finished_signal.emit({'error': '序列不匹配（无重叠 Z 范围）'})

            n_slices = len(common)
            indices =[int(n_slices * p) for p in[0.25, 0.40, 0.50, 0.60, 0.75]]
            indices =[min(i, n_slices - 1) for i in indices]
            samples = [common[i] for i in indices]
            
            all_chars = {'shift': [], 'noise': [], 'vessel':[], 'bone_mismatch':[]}
            
            for i, inst in enumerate(samples):
                pre_d = pydicom.dcmread(pre_dict[inst]).pixel_array.astype(np.float32)
                post_d = pydicom.dcmread(post_dict[inst]).pixel_array.astype(np.float32)
                 
                dy, dx = fft_phase_correlation_2d(pre_d, post_d, max_shift=10)
                shift_mag = np.sqrt(dy**2 + dx**2)
                all_chars['shift'].append(shift_mag)
                
                post_aligned = shift_image_2d(post_d, dy, dx)
                diff = np.clip(post_aligned - pre_d, 0, None)
                
                brain_mask = ndimage.binary_erosion((pre_d > 20) & (pre_d < 80), iterations=2)
                noise = float(pre_d[brain_mask].std()) if brain_mask.sum() > 500 else 8.0
                all_chars['noise'].append(noise)
                
                skull_mask = ndimage.binary_fill_holes(pre_d > 100)
                intracranial = skull_mask & ~ndimage.binary_dilation(pre_d > 100, iterations=4)
                if intracranial.sum() > 1000:
                    vessel_enhancement = float(np.percentile(diff[intracranial], 99.5))
                else:
                    vessel_enhancement = float(np.percentile(diff, 99.9))
                all_chars['vessel'].append(vessel_enhancement)
                
                bone_edge = ndimage.binary_dilation(pre_d > 150, iterations=1) ^ (pre_d > 150)
                bone_mismatch = float(diff[bone_edge].mean()) if bone_edge.sum() > 100 else 0.0
                all_chars['bone_mismatch'].append(bone_mismatch)
                
                self.progress.emit(int((i + 1) / len(samples) * 100))
            
            avg_shift = float(np.mean(all_chars['shift']))
            avg_noise = float(np.mean(all_chars['noise']))
            avg_vessel = float(np.mean(all_chars['vessel']))
            avg_bone = float(np.mean(all_chars['bone_mismatch']))
            
            score = 100.0
            score -= max(0, (avg_shift - 0.3) * 25)
            score -= max(0, (avg_noise - 6.0) * 3.0)
            score -= max(0, (150.0 - avg_vessel) * 0.4)
            if avg_bone > 20.0:
                score -= ((avg_bone - 20.0) ** 1.2) * 1.5
            
            score = max(0.0, min(100.0, score)) / 100.0
            
            rec = {
                'bone_strength': 1.4 if avg_bone > 35 else 1.2,
                'vessel_sensitivity': 0.8 if avg_noise > 12 else 1.0,
                'vessel_enhance': 2.5 if avg_vessel < 100 else (2.2 if avg_vessel < 150 else 1.8),
                'clean_bone_edges': avg_bone > 20,
                'min_vessel_size': 7 if avg_noise > 10 else 5,
                'smooth_sigma': 1.0 if avg_noise > 10 else 0.7,
                'wc': 200, 'ww': 400,
                'quality_score': score,
                'preserve_calcification': True,
                'details': (f"位移偏差：{avg_shift:.2f} px\n"
                            f"实质噪声：{avg_noise:.1f} HU\n"
                            f"强化峰值：{avg_vessel:.0f} HU\n"
                            f"骨边误差：{avg_bone:.0f} HU")
            }
            
            if score >= 0.85 and avg_shift < 0.6:
                rec['recommended_mode'] = 'fast'
                rec['recommended_opt'] = 'light'
                quality_str = "优良 (自动配置：快速 + 轻量级)"
            elif score >= 0.70:
                rec['recommended_mode'] = 'quality'
                rec['recommended_opt'] = 'standard'
                quality_str = "一般 (自动配置：精细 + 标准级)"
            else:
                rec['recommended_mode'] = 'quality'
                rec['recommended_opt'] = 'deep'
                quality_str = "较差 (自动配置：精细 + 深度级)"
                
            self.log.emit(f"综合质量：{score * 100:.0f} 分")
            self.log.emit(rec['details'])
            self.log.emit(f"系统建议：{quality_str}")
            
            self.finished_signal.emit(rec)
        except Exception as e:
            import traceback
            self.log.emit(f"[分析错误] {traceback.format_exc()}")
            self.finished_signal.emit({'error': str(e)})

def compute_3d_registration_transform(fixed_shrink, moving_shrink, log_cb=None, progress_cb=None):
    if log_cb:
        log_cb("  [配准升级] 启动：物理降采样 + SITK 纯 C++ 剥离扫描床 + 四元数旋转")
    
    body_raw = sitk.BinaryThreshold(fixed_shrink, lowerThreshold=-400.0, upperThreshold=3000.0, insideValue=1, outsideValue=0)
    cc = sitk.ConnectedComponent(body_raw)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    body_only = sitk.BinaryThreshold(relabeled, lowerThreshold=1.0, upperThreshold=1.0, insideValue=1, outsideValue=0)
    
    bone_raw = sitk.BinaryThreshold(fixed_shrink, lowerThreshold=150.0, upperThreshold=3000.0, insideValue=1, outsideValue=0)
    patient_bone = sitk.And(bone_raw, body_only)
    bone_mask = sitk.BinaryDilate(patient_bone, [1, 1, 1], sitk.sitkBall)
    
    del body_raw, cc, relabeled, body_only, bone_raw, patient_bone
    force_gc()
    
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricFixedMask(bone_mask)
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_shrink, moving_shrink,
        sitk.VersorRigid3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R.SetInitialTransform(initial_transform, inPlace=False)
    
    R.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=0.001, numberOfIterations=50, estimateLearningRate=R.Once
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInterpolator(sitk.sitkLinear)
    
    if progress_cb:
        R.AddCommand(sitk.sitkIterationEvent, lambda: progress_cb(int((R.GetOptimizerIteration() / 50.0) * 20)))
        
    final_transform = R.Execute(fixed_shrink, moving_shrink)
    
    if log_cb:
        params =[round(x, 4) for x in final_transform.GetParameters()]
        log_cb(f"  [配准参数] 四元数及平移向量：{params}")
        
    del bone_mask, R
    force_gc()
    return final_transform

# ════════════════════════════════════════════════════════════════════
# 基础掩膜与特征提取工具
# ════════════════════════════════════════════════════════════════════
def create_body_mask(image):
    img_f = image.astype(np.float32) if image.dtype != np.float32 else image
    body = ndimage.binary_fill_holes(ndimage.binary_closing(img_f > -400, iterations=3))
    labeled, num = ndimage.label(body)
    if num > 0:
        sizes = ndimage.sum(body, labeled, range(1, num + 1))
        body = labeled == (np.argmax(sizes) + 1)
    return ndimage.binary_erosion(body, iterations=3)

def detect_thin_bone(pre_image):
    img_f = pre_image.astype(np.float32) if pre_image.dtype != np.float32 else pre_image
    bone = img_f > 150
    thin_bone = bone & ~ndimage.binary_erosion(bone, iterations=2)
    edges = np.abs(ndimage.sobel(img_f))
    ref_edges = edges[bone] if bone.sum() > 0 else edges
    high_edge = edges > np.percentile(ref_edges, 70)
    return thin_bone | (bone & high_edge)

def detect_equipment(pre, post, body=None):
    pre_f = pre.astype(np.float32) if pre.dtype != np.float32 else pre
    post_f = post.astype(np.float32) if post.dtype != np.float32 else post
    if body is None:
        body = create_body_mask(pre_f)
    high_both = (pre_f > 150) & (post_f > 150)
    stable = np.abs(post_f - pre_f) < 30
    body_dilated = ndimage.binary_dilation(body, iterations=8)
    return ndimage.binary_dilation(high_both & stable & ~body_dilated, iterations=5)

class SliceCache:
    def __init__(self):
        self.body, self.scalp, self.air_bone, self.petrous = None, None, None, None
        self.thin_bone, self.venous, self.petrous_edge = None, None, None
        self.bone_edge, self.ab_edge, self.ica_protect = None, None, None

def precompute_slice_cache(pre_s, diff_pos, quality_mode):
    pre_f = pre_s.astype(np.float32) if pre_s.dtype != np.float32 else pre_s
    cache = SliceCache()
    cache.body = create_body_mask(pre_f)
    bone = pre_f > 150
    grad_y, grad_x = ndimage.sobel(pre_f, axis=0), ndimage.sobel(pre_f, axis=1)
    gradient = np.sqrt(grad_y ** 2 + grad_x ** 2)
    cache.thin_bone = (bone & ~ndimage.binary_erosion(bone, iterations=2)) | (bone & (gradient > np.percentile(gradient[bone] if bone.sum() > 0 else gradient.ravel(), 70)))
    cache.bone_edge = ndimage.binary_dilation(bone, iterations=2) & ~ndimage.binary_erosion(bone, iterations=2)

    if not quality_mode:
        return cache

    cache.scalp = ndimage.binary_dilation((cache.body & ~ndimage.binary_erosion(cache.body, iterations=8)) & ((pre_f > -50) & (pre_f < 100)), iterations=2)
    air_d4, bone_d4 = ndimage.binary_dilation(pre_f < -200, iterations=4), ndimage.binary_dilation(bone, iterations=4)
    cache.air_bone = air_d4 & bone_d4 & ndimage.binary_dilation(bone, iterations=6)
    cache.ab_edge = air_d4 & bone_d4

    petrous = ndimage.binary_dilation((pre_f > 400) & ndimage.binary_dilation(gradient > 100, iterations=2), iterations=3)
    cache.petrous, cache.petrous_edge = petrous, petrous & ~ndimage.binary_erosion(petrous, iterations=2)

    ica_core = petrous & (pre_f < 150) & (diff_pos > 50)
    cache.ica_protect = ndimage.binary_dilation(ica_core, iterations=2) if ica_core.sum() > 0 else np.zeros_like(pre_s, dtype=bool)

    brain_dilated = ndimage.binary_dilation(ndimage.binary_erosion((pre_f > 20) & (pre_f < 60), iterations=3), iterations=6)
    cache.venous = (brain_dilated & ~ndimage.binary_erosion((pre_f > 20) & (pre_f < 60), iterations=3)) & ((diff_pos > 20) & (diff_pos < 80))
    return cache

# ════════════════════════════════════════════════════════════════════
# 解剖学分区与问题骨区检测
# ════════════════════════════════════════════════════════════════════
def get_anatomical_zone(z_index, total_slices):
    z_ratio = z_index / total_slices
    if z_ratio < 0.25:
        return 'skull_base'
    elif z_ratio < 0.40:
        return 'willis_circle'
    else:
        return 'convexity'

def detect_problematic_bone_zones(pre_slice, z_ratio):
    pre_f = pre_slice.astype(np.float32) if pre_slice.dtype != np.float32 else pre_slice
    gradient = np.sqrt(ndimage.sobel(pre_f, axis=0)**2 + ndimage.sobel(pre_f, axis=1)**2)
    petrous = np.zeros_like(pre_slice, dtype=bool)
    if z_ratio < 0.35:
        high_density = pre_f > 400
        air_adjacent = ndimage.binary_dilation(pre_f < -200, iterations=5)
        petrous = high_density & air_adjacent

    skull_base = np.zeros_like(pre_slice, dtype=bool)
    if z_ratio < 0.30:
        skull_base = (pre_f > 250) & (pre_f < 1000)

    mandible = np.zeros_like(pre_slice, dtype=bool)
    if z_ratio < 0.18:
        mandible = pre_f > 200

    sharp_edge = (pre_f > 150) & (gradient > 120)

    return {'petrous': petrous, 'skull_base': skull_base, 'mandible': mandible, 'sharp_edge': sharp_edge}

def get_zone_adjusted_params(zone, bone_strength, vessel_sensitivity):
    if zone == 'skull_base': return bone_strength * 1.3, vessel_sensitivity * 1.1
    elif zone == 'willis_circle': return bone_strength * 0.85, vessel_sensitivity * 1.2
    else: return bone_strength, vessel_sensitivity

# ════════════════════════════════════════════════════════════════════
# 减影算法核心分级
# ════════════════════════════════════════════════════════════════════
def fast_subtraction(pre, post_aligned, bone_strength=1.0, vessel_sensitivity=1.0, body=None):
    pre_f = pre.astype(np.float32) if pre.dtype != np.float32 else pre
    post_f = post_aligned.astype(np.float32) if post_aligned.dtype != np.float32 else post_aligned
    if body is None: body = create_body_mask(pre_f)
    diff_pos = np.clip(post_f - pre_f, 0, None)
    gain = np.ones_like(pre_f, dtype=np.float32)

    gain[~body] = 0
    gain[pre_f < -500] = 0
    gain[pre_f > 500] = 0

    high_bone = (pre_f > 300) & (pre_f <= 500)
    med_bone = (pre_f > 180) & (pre_f <= 300)

    gain[high_bone & (diff_pos < 120 * bone_strength)] = 0
    gain[high_bone & (diff_pos >= 120 * bone_strength)] = 0.1 / bone_strength
    gain[med_bone & (diff_pos < 60 * bone_strength)] = 0
    gain[med_bone & (diff_pos >= 60 * bone_strength) & (diff_pos < 120 * bone_strength)] = 0.15 / bone_strength
    gain[med_bone & (diff_pos >= 120 * bone_strength)] = 0.3 / bone_strength

    thin_bone = detect_thin_bone(pre_f)
    gain[thin_bone & (diff_pos < 80 * bone_strength)] = 0
    gain[thin_bone & (diff_pos >= 80 * bone_strength)] = 0.1 / bone_strength

    soft = (pre_f > -100) & (pre_f <= 100)
    shifted_bone_veto = (pre_f < 150) & (post_f > 350)
    gain[soft & (diff_pos < 20 / vessel_sensitivity)] = 0.1

    vessel_mask = (pre_f > -100) & (pre_f < 60) & (diff_pos > 50 * vessel_sensitivity) & ~shifted_bone_veto
    gain[vessel_mask] = 1.0

    return diff_pos * np.clip(gain, 0, 1.5)

def quality_subtraction_cached(pre, post_aligned, bone_strength, vessel_sensitivity, cache):
    pre_f = pre.astype(np.float32) if pre.dtype != np.float32 else pre
    post_f = post_aligned.astype(np.float32) if post_aligned.dtype != np.float32 else post_aligned
    diff_pos = np.clip(post_f - pre_f, 0, None)
    gain = np.ones_like(diff_pos, dtype=np.float32)
    gain[~cache.body] = 0
    gain[pre_f < -500] = 0
    gain[pre_f > 500] = 0

    if cache.petrous is not None:
        th_p = 150 * bone_strength
        ica = cache.ica_protect if cache.ica_protect is not None else np.zeros_like(pre, dtype=bool)
        petrous_no_ica = cache.petrous & ~ica
        gain[petrous_no_ica & (diff_pos < th_p)] = 0
        gain[petrous_no_ica & (diff_pos >= th_p)] = 0.05 / bone_strength
        gain[ica & (diff_pos > 150)] = 0.8
        gain[ica & (diff_pos > 80) & (diff_pos <= 150)] = 0.6

    if cache.air_bone is not None:
        th_a = 100 * bone_strength
        gain[cache.air_bone & (diff_pos < th_a)] = 0
        gain[cache.air_bone & (diff_pos >= th_a) & (diff_pos < th_a * 1.5)] = 0.1 / bone_strength

    if cache.scalp is not None:
        th_s = 60 * bone_strength
        gain[cache.scalp & (diff_pos < th_s)] = 0
        gain[cache.scalp & (diff_pos >= th_s) & (diff_pos < th_s * 2)] = 0.2

    if cache.venous is not None:
        venous_active = cache.venous & (gain > 0)
        gain[venous_active] = np.maximum(gain[venous_active], 0.3)

    exclude = np.zeros_like(pre, dtype=bool)
    if cache.petrous is not None: exclude |= cache.petrous
    if cache.air_bone is not None: exclude |= cache.air_bone
    med_bone = (pre_f > 180) & (pre_f <= 300) & ~exclude
    gain[med_bone & (diff_pos < 60 * bone_strength)] = 0
    gain[med_bone & (diff_pos >= 60 * bone_strength) & (diff_pos < 120 * bone_strength)] = 0.15 / bone_strength
    gain[med_bone & (diff_pos >= 120 * bone_strength)] = 0.3 / bone_strength

    thin_only = cache.thin_bone & ~exclude
    th_thin = 80 * bone_strength
    gain[thin_only & (diff_pos < th_thin)] = 0
    gain[thin_only & (diff_pos >= th_thin)] = 0.1 / bone_strength

    shifted_bone_veto = (pre_f < 150) & (post_f > 350)
    scalp_mask = cache.scalp if cache.scalp is not None else np.zeros_like(pre, dtype=bool)
    ab_mask = cache.air_bone if cache.air_bone is not None else np.zeros_like(pre, dtype=bool)
    vessel_mask = (pre_f > -100) & (pre_f < 60) & (diff_pos > 50 * vessel_sensitivity) & ~scalp_mask & ~ab_mask & ~shifted_bone_veto
    gain[vessel_mask] = 1.0

    return diff_pos * np.clip(gain, 0, 1.5)

def detect_ica_protection_zone_precise(pre_s, diff_s, body_mask):
    pre_f = pre_s.astype(np.float32) if pre_s.dtype != np.float32 else pre_s
    dense_bone = pre_f > 400
    if dense_bone.sum() < 50: return np.zeros_like(pre_s, dtype=bool)
    ica_candidate = ndimage.binary_dilation(dense_bone, iterations=5) & (diff_s > 80) & body_mask
    ica_candidate = ica_candidate & ndimage.binary_erosion(body_mask, iterations=10)
    return ndimage.binary_dilation(ica_candidate, iterations=2) if ica_candidate.sum() > 0 else np.zeros_like(pre_s, dtype=bool)

def compute_adaptive_threshold(pre_s, diff_s, bone_mask, bone_strength, window_size=31):
    b_diff = np.where(bone_mask, diff_s, 0).astype(np.float32)
    b_float = bone_mask.astype(np.float32)
    local_count_safe = np.maximum(ndimage.uniform_filter(b_float, size=window_size), 1e-6)
    local_mean = ndimage.uniform_filter(b_diff, size=window_size) / local_count_safe
    local_std = np.sqrt(np.maximum(ndimage.uniform_filter(b_diff ** 2, size=window_size) / local_count_safe - local_mean ** 2, 0))
    adaptive_threshold = np.maximum(local_mean + (1.5 / bone_strength) * local_std, 50 * bone_strength)
    return np.where(bone_mask, adaptive_threshold, 1000)

def optimized_subtraction_light_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache):
    pre_f = pre.astype(np.float32) if pre.dtype != np.float32 else pre
    post_f = post_aligned.astype(np.float32) if post_aligned.dtype != np.float32 else post_aligned
    diff_pos = np.clip(post_f - pre_f, 0, None)
    bone_mask = pre_f > 150
    gain = np.ones_like(diff_pos, dtype=np.float32)
    gain[~cache.body] = 0
    gain[pre_f < -500] = 0
    gain[pre_f > 500] = 0

    ica_protect = detect_ica_protection_zone_precise(pre_f, diff_pos, cache.body)
    adaptive_thresh = compute_adaptive_threshold(pre_f, diff_pos, bone_mask, bone_strength)
    gain[bone_mask & (diff_pos < adaptive_thresh) & ~ica_protect] = 0
    gain[bone_mask & (diff_pos >= adaptive_thresh) & (diff_pos < adaptive_thresh * 1.5) & ~ica_protect] = 0.2 / bone_strength
    gain[bone_mask & (diff_pos >= adaptive_thresh * 1.5) & ~ica_protect] = 0.4 / bone_strength
    shifted_bone_veto = (pre_f < 150) & (post_f > 350)

    vessel_mask = (pre_f > -100) & (pre_f < 60) & (diff_pos > 50 * vessel_sensitivity) & cache.body & ~shifted_bone_veto
    gain[vessel_mask] = 1.0
    gain[ica_protect & (diff_pos > 60)] = 0.85
    gain[~cache.body] = 0

    return diff_pos * np.clip(gain, 0, 1.5)

def optimized_subtraction_standard_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy):
    pre_f = pre.astype(np.float32) if pre.dtype != np.float32 else pre
    post_f = post_aligned.astype(np.float32) if post_aligned.dtype != np.float32 else post_aligned
    result = optimized_subtraction_light_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache)
    diff_pos = np.clip(post_f - pre_f, 0, None)
    bone = pre_f > 200

    if bone.sum() > 100:
        sinus = (ndimage.binary_dilation(bone & ~ndimage.binary_erosion(bone, iterations=3), iterations=4) & ~bone & cache.body) & ((diff_pos > 35) & (diff_pos < 90))
        if sinus.sum() > 30:
            result[sinus] = np.maximum(result[sinus], diff_pos[sinus] * 0.35)
    return result

# ════════════════════════════════════════════════════════════════════
# 形态学 Top-Hat 变换
# ════════════════════════════════════════════════════════════════════
def morphological_top_hat_2d(image, disk_radius=3):
    y, x = np.ogrid[-disk_radius:disk_radius+1, -disk_radius:disk_radius+1]
    disk = (x*x + y*y <= disk_radius*disk_radius).astype(np.uint8)
    opened = ndimage.grey_opening(image.astype(np.float32), footprint=disk)
    top_hat = np.maximum(image - opened, 0)
    return top_hat.astype(np.float32)

def multi_scale_top_hat_2d(image, radii=[2, 3, 5]):
    result = np.zeros_like(image, dtype=np.float32)
    for r in radii:
        th = morphological_top_hat_2d(image, disk_radius=r)
        result = np.maximum(result, th)
    return result

def compute_top_hat_for_slice(pre_s, post_s, spacing_zyx):
    pre_f = pre_s.astype(np.float32) if pre_s.dtype != np.float32 else pre_s
    post_f = post_s.astype(np.float32) if post_s.dtype != np.float32 else post_s
    slice_diff = np.clip(post_f - pre_f, 0, None)
    if slice_diff.max() <= 15:
        return None

    min_sp = min(spacing_zyx[1], spacing_zyx[2])
    radii =[max(1, int(round(r / min_sp))) for r in[0.8, 1.5, 2.5]]
    radii = list(set(radii))
    radii =[r for r in radii if 1 <= r <= 6]
    if not radii: radii =[2, 3]

    top_hat = multi_scale_top_hat_2d(slice_diff, radii=radii)
    vmax = top_hat.max()
    if vmax > 0: top_hat /= vmax
    return top_hat

def optimized_subtraction_deep_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy, spacing_zyx):
    pre_f = pre.astype(np.float32) if pre.dtype != np.float32 else pre
    post_f = post_aligned.astype(np.float32) if post_aligned.dtype != np.float32 else post_aligned
    result = optimized_subtraction_standard_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy)

    top_hat_map = compute_top_hat_for_slice(pre_f, post_f, spacing_zyx)
    if top_hat_map is not None and top_hat_map.max() > 0:
        high_response = top_hat_map > 0.2
        boost = 1.0 + 0.25 * top_hat_map
        result = result * np.where(high_response & cache.body, boost, 1.0)
    return result

def enhanced_bone_suppression_v299(pre_s, diff_pos, gain, bone_zones, bone_strength):
    petrous_thresh = 180 * bone_strength
    petrous = bone_zones['petrous']
    if np.any(petrous):
        gain[petrous & (diff_pos < petrous_thresh)] = 0
        gain[petrous & (diff_pos >= petrous_thresh) & (diff_pos < petrous_thresh * 1.5)] = 0.03
        gain[petrous & (diff_pos >= petrous_thresh * 1.5)] = 0.08

    base_thresh = 120 * bone_strength
    skull_base = bone_zones['skull_base']
    if np.any(skull_base):
        gain[skull_base & (diff_pos < base_thresh)] = 0
        gain[skull_base & (diff_pos >= base_thresh) & (diff_pos < base_thresh * 1.5)] = 0.05
        gain[skull_base & (diff_pos >= base_thresh * 1.5)] = 0.1

    mandible = bone_zones['mandible']
    if np.any(mandible):
        gain[mandible & (diff_pos < 200 * bone_strength)] = 0
        gain[mandible & (diff_pos >= 200 * bone_strength)] = 0.05

    sharp_edge = bone_zones['sharp_edge']
    if np.any(sharp_edge):
        gain[sharp_edge & (diff_pos < 100 * bone_strength)] = 0
        gain[sharp_edge & (diff_pos >= 100 * bone_strength) & (diff_pos < 150 * bone_strength)] = 0.1
    return gain

def detect_calcification_2d_slice(pre_s, diff_pos, body_mask, calc_threshold, vessel_sensitivity, prev_calc_labels=None):
    pre_f = pre_s.astype(np.float32) if pre_s.dtype != np.float32 else pre_s
    main_bone = ndimage.binary_opening(pre_f > 250, iterations=1)
    bone_forbidden_zone = ndimage.binary_dilation(main_bone, iterations=4)
    calc_candidate = (pre_f >= calc_threshold) & (pre_f <= 2500) & body_mask & (~bone_forbidden_zone)
    labeled_calc, n_calc = ndimage.label(calc_candidate)

    if n_calc == 0: return np.zeros_like(pre_s, dtype=bool), None
        
    sizes = np.bincount(labeled_calc.ravel(), minlength=n_calc + 1)
    valid_mask = np.zeros(n_calc + 1, dtype=bool)
    valid_mask[(sizes >= 2) & (sizes <= 120)] = True  
    valid_mask[0] = False
    isolated_small = valid_mask[labeled_calc]

    if not np.any(isolated_small): return np.zeros_like(pre_s, dtype=bool), None
        
    vessel_vicinity = ndimage.binary_dilation((diff_pos > 40 * vessel_sensitivity) & body_mask, iterations=2)
    final_calc_mask = isolated_small & vessel_vicinity
    return final_calc_mask, labeled_calc

def get_safe_percentile_nonzero(volume, percentile, max_bin=4096):
    hist = np.zeros(max_bin, dtype=np.int64)
    for z in range(volume.shape[0]):
        v = volume[z]
        mask = v > 0
        if np.any(mask):
            hist += np.bincount(np.clip(v[mask], 0, max_bin - 1).astype(np.int32), minlength=max_bin)
    total = hist.sum()
    if total == 0: return 1.0
    target = total * (percentile / 100.0)
    cum = 0
    for i in range(max_bin):
        cum += hist[i]
        if cum >= target: return float(i)
    return float(max_bin - 1)

def apply_3d_cleanup(volume, min_area=15, log_cb=None):
    if log_cb: log_cb("  [+] 2.5D 切片连通清理...")
    fg_thresh = max(10.0, get_safe_percentile_nonzero(volume, 10, max_bin=4096))
    depth = volume.shape[0]
    mask_prev = None
    mask_curr = volume[0] > fg_thresh if depth > 0 else None
    for z in range(depth):
        mask_next = volume[z+1] > fg_thresh if z < depth - 1 else np.zeros_like(mask_curr)
        if mask_curr is not None and np.any(mask_curr):
            labeled, n = ndimage.label(mask_curr)
            if n > 0:
                overlap_mask = mask_next.copy()
                if mask_prev is not None: overlap_mask |= mask_prev
                sizes = np.bincount(labeled.ravel(), minlength=n+1)
                overlap_counts = np.bincount(labeled[overlap_mask], minlength=n+1)
                keep = (sizes >= min_area) | (overlap_counts > 0)
                keep[0] = True
                noise = ~keep[labeled] & (labeled > 0)
                if np.any(noise):
                    chunk_vol = volume[z]
                    chunk_vol[noise & (chunk_vol < 60)] = 0
                    mask_to_dim = noise & (chunk_vol >= 60)
                    chunk_vol[mask_to_dim] = (chunk_vol[mask_to_dim] * 0.3).astype(np.int16)
                    mask_curr[noise] = False
        mask_prev = mask_curr
        mask_curr = mask_next
    force_gc()
    return volume

def shape_analysis_fast(volume, spacing, log_cb=None):
    if log_cb: log_cb("  [形状分析] 2.5D 滑动连通分析...")
    voxel_area = spacing[1] * spacing[2]
    depth = volume.shape[0]
    mask_prev = None
    mask_curr = volume[0] > 12 if depth > 0 else None
    for z in range(depth):
        mask_next = volume[z+1] > 12 if z < depth - 1 else np.zeros_like(mask_curr)
        if mask_curr is not None and np.any(mask_curr):
            labeled, n = ndimage.label(mask_curr)
            if n > 0:
                overlap_mask = mask_next.copy()
                if mask_prev is not None: overlap_mask |= mask_prev
                sizes = np.bincount(labeled.ravel(), minlength=n+1)
                overlap_counts = np.bincount(labeled[overlap_mask], minlength=n+1)
                sums = np.bincount(labeled.ravel(), weights=volume[z].ravel(), minlength=n+1)
                areas = sizes * voxel_area
                isolated = overlap_counts == 0
                action1 = isolated & (areas < 1.0)
                means = np.zeros_like(sums)
                np.divide(sums, sizes, out=means, where=sizes > 0)
                action2 = isolated & (areas < 3.0) & (means < 25) & ~action1
                act_z = np.zeros(n+1, dtype=np.uint8)
                act_z[action1] = 1
                act_z[action2] = 2
                actions_mapped = act_z[labeled]
                if np.any(actions_mapped > 0):
                    res_z = volume[z]
                    res_z[actions_mapped == 1] = 0
                    mask_to_dim = actions_mapped == 2
                    res_z[mask_to_dim] = (res_z[mask_to_dim] * 0.4).astype(np.int16)
                    mask_curr[actions_mapped == 1] = False
        mask_prev = mask_curr
        mask_curr = mask_next
    force_gc()
    return volume

def clean_bone_edges_cached(image, pre_image, cache, mode='fast'):
    result = image.copy()
    edge = cache.bone_edge
    result[edge & (image < 40)] = 0
    result[edge & (image >= 40) & (image < 80)] = result[edge & (image >= 40) & (image < 80)] * 0.3
    if mode == 'quality':
        if cache.ab_edge is not None:
            result[cache.ab_edge & (image < 60)] = 0
            result[cache.ab_edge & (image >= 60) & (image < 100)] = result[cache.ab_edge & (image >= 60) & (image < 100)] * 0.2
        if cache.petrous_edge is not None:
            pe_no_ica = cache.petrous_edge & ~(cache.ica_protect if cache.ica_protect is not None else np.zeros_like(image, dtype=bool))
            result[pe_no_ica & (image < 80)] = 0
            result[pe_no_ica & (image >= 80) & (image < 120)] = result[pe_no_ica & (image >= 80) & (image < 120)] * 0.15
    return result

def edge_preserving_smooth(image, pre_image, sigma=0.7):
    pre_f = pre_image.astype(np.float32) if pre_image.dtype != np.float32 else pre_image
    edges = np.abs(ndimage.sobel(pre_f))
    edge_norm = edges / (edges.max() + 1e-6)
    return ndimage.gaussian_filter(image, sigma * 1.5) * (1 - edge_norm) + ndimage.gaussian_filter(image, sigma * 0.3) * edge_norm

# ════════════════════════════════════════════════════════════════════
# 生成 ROI Mask
# ════════════════════════════════════════════════════════════════════
def generate_roi_mask_from_polygon(roi: PolygonROI, shape: Tuple[int, int]) -> np.ndarray:
    if not roi.closed or len(roi.points) < 3: return np.zeros(shape, dtype=bool)
    h, w = shape
    qimg = QImage(w, h, QImage.Format_Grayscale8)
    qimg.fill(0)
    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.Antialiasing, False)
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    painter.setPen(Qt.NoPen)
    painter.drawPolygon(QPolygonF(roi.points))
    painter.end()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w))
    return arr > 127

def apply_roi_masks_to_volume(volume: np.ndarray, annotations: AnnotationData, z_start: int, z_end: int, log_cb=None) -> np.ndarray:
    if log_cb: log_cb("  [ROI 应用] 生成掩膜中 (原地修改)...")
    result = volume 
    depth, h, w = volume.shape
    global_masks =[generate_roi_mask_from_polygon(roi, (h, w)) for roi in annotations.global_rois] 
    combined_global = np.any(np.stack(global_masks), axis=0) if global_masks else None
    
    inv_global_masks =[generate_roi_mask_from_polygon(roi, (h, w)) for roi in annotations.inverse_global_rois]
    combined_inv_global = np.any(np.stack(inv_global_masks), axis=0) if inv_global_masks else None

    for z in range(z_start, min(z_end + 1, depth)):
        if combined_inv_global is not None: result[z][~combined_inv_global] = 0
        if z in annotations.inverse_local_rois and annotations.inverse_local_rois[z]:
            inv_local_masks =[generate_roi_mask_from_polygon(roi, (h, w)) for roi in annotations.inverse_local_rois[z]]
            if inv_local_masks:
                result[z][~np.any(np.stack(inv_local_masks), axis=0)] = 0
        if combined_global is not None: result[z][combined_global] = 0
        if z in annotations.local_rois and annotations.local_rois[z]:
            local_masks =[generate_roi_mask_from_polygon(roi, (h, w)) for roi in annotations.local_rois[z]]
            if local_masks:
                result[z][np.any(np.stack(local_masks), axis=0)] = 0
    return result

# ════════════════════════════════════════════════════════════════════
# 交互式图像画布（优化：支持降采样显示）
# ════════════════════════════════════════════════════════════════════
class ImageCanvas(QWidget):
    """
    ★ 优化：支持降采样 volume 显示 ★
    - display_volume: 降采样后的显示数据
    - display_scale: 降采样比例（用于坐标还原）
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#1a1a1a;")
        self.setMinimumSize(512, 512)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.display_volume: Optional[np.ndarray] = None
        self.display_scale: float = 1.0  
        
        self.current_z = 0
        self.depth = 0
        self.img_width = 0
        self.img_height = 0
        
        self.wc, self.ww = 200, 600
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)
        
        self.mode = "navigate"
        self.is_panning = False
        self.is_zooming = False
        self.is_windowing = False
        self.is_moving_vertex = False
        self.last_mouse_pos = QPointF()
        self.current_mouse_pos = QPointF()
        
        self.selected_roi: Optional[PolygonROI] = None
        self.selected_vertex_idx: int = -1
        self.hover_vertex_idx: int = -1
        self.hover_roi: Optional[PolygonROI] = None
        
        self.drawing_state = DrawingState()
        self.current_polygon: Optional[PolygonROI] = None
        self.annotations: Optional[AnnotationData] = None
        self.display_pixmap: Optional[QPixmap] = None
        
        self.on_slice_changed = None
        self.on_annotation_changed = None
        self.on_window_changed = None
        self.on_status_message = None
        self.on_drawing_state_changed = None

    def set_volume(self, volume: np.ndarray, scale: float = 1.0):
        self.display_volume = volume
        self.display_scale = scale
        self.depth, self.img_height, self.img_width = volume.shape
        self.current_z = self.depth // 2
        self._fit_to_window()
        self._update_pixmap()
        self.update()

    def _fit_to_window(self):
        if self.display_volume is None: return
        w, h = self.width() - 20, self.height() - 20
        self.zoom = min(w / self.img_width, h / self.img_height, 2.0)
        self.pan_offset = QPointF((w - self.img_width * self.zoom) / 2 + 10, (h - self.img_height * self.zoom) / 2 + 10)

    def set_slice(self, z: int):
        if self.display_volume is None: return
        z = max(0, min(z, self.depth - 1))
        if z != self.current_z:
            self.current_z = z
            if self.drawing_state.is_drawing:
                self.drawing_state.current_slice = z
                self.drawing_state.visited_slices.add(z)
                if self.on_drawing_state_changed:
                    self.on_drawing_state_changed(self.drawing_state)
            else:
                self._clear_edit_state()
            self._update_pixmap()
            self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        self.current_polygon = None
        self.drawing_state.reset()
        self._clear_edit_state()
        self.update()
        cursors = {"navigate": Qt.OpenHandCursor, "window": Qt.SizeAllCursor,
                    "move_vertex": Qt.PointingHandCursor, "delete_vertex": Qt.ForbiddenCursor}
        self.setCursor(cursors.get(mode, Qt.CrossCursor))    

    def _clear_edit_state(self):
        self.selected_roi = None
        self.selected_vertex_idx = -1
        self.hover_vertex_idx = -1
        self.hover_roi = None
        self.is_moving_vertex = False

    def _update_pixmap(self):
        if self.display_volume is None: return
        data = self.display_volume[self.current_z]
        data_f = data.astype(np.float32) if data.dtype != np.float32 else data
        vmin, vmax = self.wc - self.ww / 2, self.wc + self.ww / 2
        norm = np.clip((data_f - vmin) / (vmax - vmin), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1).copy()
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
        self.display_pixmap = QPixmap.fromImage(qimg.copy())

    def _get_visible_rois(self) -> List[PolygonROI]:
        if not self.annotations: return []
        rois =[]
        if self.mode not in ("local_roi", "local_inverse_roi"):
            rois.extend(self.annotations.global_rois)
            rois.extend(self.annotations.inverse_global_rois)
        if self.current_z in self.annotations.local_rois:
            rois.extend(self.annotations.local_rois[self.current_z])
        if self.current_z in self.annotations.inverse_local_rois:
            rois.extend(self.annotations.inverse_local_rois[self.current_z])
        return rois

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(26, 26, 26))
        
        if not self.display_pixmap:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "加载中...")
            return
        
        painter.translate(self.pan_offset)
        painter.scale(self.zoom, self.zoom)
        painter.drawPixmap(0, 0, self.display_pixmap)
        
        if self.annotations:
            if self.mode not in ("local_roi", "local_inverse_roi"):
                for roi in self.annotations.global_rois:
                    self._draw_polygon(painter, roi, QColor(255, 80, 80, 200), roi == self.selected_roi, roi == self.hover_roi)
                for roi in self.annotations.inverse_global_rois:
                    self._draw_polygon(painter, roi, QColor(0, 220, 220, 220), roi == self.selected_roi, roi == self.hover_roi, inverse=True)
            if self.current_z in self.annotations.local_rois:
                for roi in self.annotations.local_rois[self.current_z]:
                    self._draw_polygon(painter, roi, QColor(80, 255, 80, 200), roi == self.selected_roi, roi == self.hover_roi)
            if self.current_z in self.annotations.inverse_local_rois:
                for roi in self.annotations.inverse_local_rois[self.current_z]:
                    self._draw_polygon(painter, roi, QColor(0, 200, 255, 220), roi == self.selected_roi, roi == self.hover_roi, inverse=True)
        
        if self.current_polygon and self.current_polygon.points:
            color = QColor(255, 180, 0, 220) if (self.drawing_state.is_drawing and self.drawing_state.get_slice_count() > 1) else QColor(255, 255, 0, 220)
            self._draw_polygon(painter, self.current_polygon, color, drawing=True)
        
        painter.resetTransform()
        
        if self.mode == "window":
            painter.setPen(QColor(255, 255, 255, 180))
            painter.setFont(QFont("Microsoft YaHei", 10))
            painter.drawText(20, 30, f"窗位:{self.wc} 窗宽:{self.ww}")
        
        if self.drawing_state.is_drawing:
            self._draw_cross_slice_indicator(painter)

    def _draw_cross_slice_indicator(self, painter: QPainter):
        state = self.drawing_state
        min_z, max_z = state.get_slice_range()
        slice_count = state.get_slice_count()
        
        rect_width = 220
        rect_height = 60 if slice_count > 1 else 40
        rect_x, rect_y = self.width() - rect_width - 20, 20
        
        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(QPen(QColor(255, 180, 0) if slice_count > 1 else QColor(255, 255, 0), 2))
        painter.drawRoundedRect(rect_x, rect_y, rect_width, rect_height, 8, 8)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        
        if slice_count > 1:
            painter.drawText(rect_x + 10, rect_y + 22, f"跨层绘制中")
            painter.setFont(QFont("Microsoft YaHei", 9))
            painter.setPen(QColor(255, 180, 0))
            painter.drawText(rect_x + 10, rect_y + 42, f"层 {min_z + 1} → {max_z + 1} (共 {slice_count} 层)")
        else:
            painter.drawText(rect_x + 10, rect_y + 25, f"绘制中 - 第 {state.start_slice + 1} 层")

    def _draw_polygon(self, painter: QPainter, roi: PolygonROI, color: QColor, selected=False, hover=False, drawing=False, inverse=False):
        if not roi.points: return
        if selected: color = QColor(255, 200, 50, 230)
        elif hover and self.mode in["add_vertex", "move_vertex", "delete_vertex"]: color = QColor(color.red(), color.green(), color.blue(), 240)
        
        pen_w = 3.0 / self.zoom if selected else 2.0 / self.zoom
        pen_style = Qt.DashLine if inverse else Qt.SolidLine
        painter.setPen(QPen(color, pen_w, pen_style))
        
        pts = roi.points
        for i in range(len(pts) - 1): painter.drawLine(pts[i], pts[i + 1])
        
        if drawing:
            painter.setPen(QPen(color, pen_w, Qt.DashLine))
            painter.drawLine(pts[-1], self.current_mouse_pos)
            if len(pts) >= 2: painter.drawLine(self.current_mouse_pos, pts[0])
        
        if roi.closed and len(pts) >= 3:
            painter.setPen(QPen(color, pen_w, pen_style))
            painter.drawLine(pts[-1], pts[0])
            fill_alpha = 30 if inverse else 40
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), fill_alpha)))
            painter.drawPolygon(QPolygonF(pts))
        
        vs = 5.0 / self.zoom if selected else 4.0 / self.zoom
        for i, p in enumerate(pts):
            is_hover = (roi == self.hover_roi and i == self.hover_vertex_idx)
            is_sel = (roi == self.selected_roi and i == self.selected_vertex_idx)
            if is_sel:
                painter.setBrush(QColor(255, 255, 255))
                painter.drawEllipse(p, vs * 1.5, vs * 1.5)
            elif is_hover:
                c = QColor(255, 100, 100) if self.mode == "delete_vertex" else QColor(255, 255, 100)
                painter.setBrush(c)
                painter.drawEllipse(p, vs * 1.3, vs * 1.3)
            elif i == 0:
                painter.setBrush(color)
                painter.fillRect(p.x() - vs, p.y() - vs, vs * 2, vs * 2, color)
            else:
                painter.setBrush(color)
                painter.drawEllipse(p, vs, vs)
        
        if drawing:
            painter.setBrush(QColor(255, 255, 255, 150))
            painter.drawEllipse(self.current_mouse_pos, vs, vs)

    def _screen_to_image(self, pos: QPointF) -> QPointF:
        return QPointF((pos.x() - self.pan_offset.x()) / self.zoom, (pos.y() - self.pan_offset.y()) / self.zoom)

    def _is_in_image(self, p: QPointF) -> bool:
        return 0 <= p.x() < self.img_width and 0 <= p.y() < self.img_height

    def _find_vertex_at(self, img_pos: QPointF) -> Tuple[Optional[PolygonROI], int]:
        th = 8.0 / self.zoom
        for roi in self._get_visible_rois():
            if roi.closed:
                idx = find_nearest_vertex(img_pos, roi, th) 
                if idx >= 0: return roi, idx
        return None, -1

    def _find_edge_at(self, img_pos: QPointF) -> Optional[PolygonROI]:
        th = 6.0 / self.zoom
        for roi in self._get_visible_rois():
            if roi.closed and len(roi.points) >= 3:
                edge_idx, _ = find_nearest_edge(img_pos, roi)
                if edge_idx >= 0:
                    a, b = roi.points[edge_idx], roi.points[(edge_idx + 1) % len(roi.points)]
                    dist, _ = point_to_line_distance(img_pos, a, b)
                    if dist <= th: return roi
        return None

    def mousePressEvent(self, event):
        spos = QPointF(event.pos())
        ipos = self._screen_to_image(spos)
        self.last_mouse_pos = spos
        
        if self.mode == "navigate":
            if event.button() == Qt.LeftButton:
                self.is_panning = True
                self.setCursor(Qt.ClosedHandCursor)
            elif event.button() == Qt.RightButton:
                self.is_zooming = True
        elif self.mode == "window" and event.button() == Qt.LeftButton:
            self.is_windowing = True
        elif self.mode == "move_vertex" and event.button() == Qt.LeftButton:
            roi, idx = self._find_vertex_at(ipos)
            if roi and idx >= 0:
                self.selected_roi, self.selected_vertex_idx = roi, idx
                self.is_moving_vertex = True
                self.setCursor(Qt.ClosedHandCursor)
                self.update()
        elif self.mode == "delete_vertex" and event.button() == Qt.LeftButton:
            roi, idx = self._find_vertex_at(ipos)
            if roi and idx >= 0 and len(roi.points) > 3:
                roi.points.pop(idx)
                self._clear_edit_state()
                self.update()
                if self.on_annotation_changed: self.on_annotation_changed()
            elif roi and len(roi.points) <= 3 and self.on_status_message:
                self.on_status_message("多边形至少需要 3 个顶点")
        elif self.mode == "add_vertex" and event.button() == Qt.LeftButton:
            roi = self._find_edge_at(ipos)
            if roi and roi.closed:
                edge_idx, _ = find_nearest_edge(ipos, roi)
                if edge_idx >= 0:
                    roi.points.insert(edge_idx + 1, ipos)
                    self.update()
                    if self.on_annotation_changed: self.on_annotation_changed()
        elif self.mode in["global_roi", "local_roi", "global_inverse_roi", "local_inverse_roi"]:
            if event.button() == Qt.LeftButton and self._is_in_image(ipos):
                if not self.current_polygon:
                    roi_type = "global" if self.mode == "global_roi" else "global_inverse" if self.mode == "global_inverse_roi" else "local_inverse" if self.mode == "local_inverse_roi" else "local"
                    self.current_polygon = PolygonROI(points=[], closed=False, roi_type=roi_type, slice_index=self.current_z)
                    self.drawing_state.is_drawing = True
                    self.drawing_state.start_slice = self.current_z
                    self.drawing_state.current_slice = self.current_z
                    self.drawing_state.visited_slices = {self.current_z}
                    self.drawing_state.polygon = self.current_polygon
                    if self.on_drawing_state_changed: self.on_drawing_state_changed(self.drawing_state)
                self.current_polygon.points.append(ipos)
                self.update()
            elif event.button() == Qt.RightButton:
                if self.current_polygon and len(self.current_polygon.points) >= 3:
                    self.current_polygon.closed = True
                    self._save_current_polygon_cross_slice()

    def mouseMoveEvent(self, event):
        spos = QPointF(event.pos())
        ipos = self._screen_to_image(spos)
        self.current_mouse_pos = ipos
        
        if self.mode == "navigate":
            if self.is_panning:
                self.pan_offset += spos - self.last_mouse_pos
                self.last_mouse_pos = spos
                self.update()
            elif self.is_zooming:
                dy = spos.y() - self.last_mouse_pos.y()
                old_ipos = self._screen_to_image(spos)
                self.zoom = max(0.2, min(10.0, self.zoom * (1.0 - dy * 0.005)))
                new_spos = QPointF(old_ipos.x() * self.zoom + self.pan_offset.x(), old_ipos.y() * self.zoom + self.pan_offset.y())
                self.pan_offset += spos - new_spos
                self.last_mouse_pos = spos
                self.update()
        elif self.mode == "window" and self.is_windowing:
            dx, dy = spos.x() - self.last_mouse_pos.x(), spos.y() - self.last_mouse_pos.y()
            self.wc = int(max(-1000, min(3000, self.wc - dy * 2)))
            self.ww = int(max(1, min(4000, self.ww + dx * 2)))
            self.last_mouse_pos = spos
            self._update_pixmap()
            self.update()
            if self.on_window_changed: self.on_window_changed(self.wc, self.ww)
        elif self.mode == "move_vertex":
            if self.is_moving_vertex and self.selected_roi and self.selected_vertex_idx >= 0:
                self.selected_roi.points[self.selected_vertex_idx] = ipos
                self.update()
            else:
                roi, idx = self._find_vertex_at(ipos)
                if roi != self.hover_roi or idx != self.hover_vertex_idx:
                    self.hover_roi, self.hover_vertex_idx = roi, idx
                    self.update()
        elif self.mode == "delete_vertex":
            roi, idx = self._find_vertex_at(ipos)
            if roi != self.hover_roi or idx != self.hover_vertex_idx:
                self.hover_roi, self.hover_vertex_idx = roi, idx
                self.update()
        elif self.mode == "add_vertex":
            roi = self._find_edge_at(ipos)
            if roi != self.hover_roi:
                self.hover_roi = roi
                self.update()
        elif self.mode in["global_roi", "local_roi", "global_inverse_roi", "local_inverse_roi"]:
            if self.current_polygon and self.current_polygon.points:
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_moving_vertex:
                self.is_moving_vertex = False
                self.setCursor(Qt.PointingHandCursor)
                self._try_merge_rois()
                if self.on_annotation_changed: self.on_annotation_changed()
            self.is_panning = False
            self.is_windowing = False
            if self.mode == "navigate": self.setCursor(Qt.OpenHandCursor)
        elif event.button() == Qt.RightButton:
            self.is_zooming = False

    def wheelEvent(self, event):
        if self.display_volume is None: return
        dz = -1 if event.angleDelta().y() > 0 else 1
        new_z = max(0, min(self.depth - 1, self.current_z + dz))
        if new_z != self.current_z:
            self.current_z = new_z
            if self.drawing_state.is_drawing:
                self.drawing_state.current_slice = new_z
                self.drawing_state.visited_slices.add(new_z)
                if self.on_drawing_state_changed: self.on_drawing_state_changed(self.drawing_state)
            else:
                self._clear_edit_state()
            self._update_pixmap()
            self.update()
            if self.on_slice_changed: self.on_slice_changed(self.current_z)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.display_volume is not None:
            self._fit_to_window()
            self.update()

    def _save_current_polygon_cross_slice(self):
        if not self.current_polygon or not self.annotations: return
        state = self.drawing_state
        min_z, max_z = state.get_slice_range()
        slice_count = state.get_slice_count()
        roi_type = self.current_polygon.roi_type  

        if roi_type == "global":
            self.annotations.global_rois.append(self.current_polygon)
        elif roi_type == "global_inverse":
            self.annotations.inverse_global_rois.append(self.current_polygon)
        elif roi_type == "local":
            for z in range(min_z, max_z + 1):
                roi_copy = self.current_polygon.copy()
                roi_copy.slice_index = z
                self.annotations.local_rois[z].append(roi_copy)
        elif roi_type == "local_inverse":
            for z in range(min_z, max_z + 1):
                roi_copy = self.current_polygon.copy()
                roi_copy.slice_index = z
                self.annotations.inverse_local_rois[z].append(roi_copy)

        self.current_polygon = None
        self.drawing_state.reset()
        if self.on_drawing_state_changed: self.on_drawing_state_changed(self.drawing_state)
        self._try_merge_rois()
        self.update()
        if self.on_annotation_changed: self.on_annotation_changed()

    def _try_merge_rois(self):
        if not self.annotations: return
        merged_count = 0
        if len(self.annotations.global_rois) > 1:
            old_count = len(self.annotations.global_rois)
            self.annotations.global_rois = merge_overlapping_polygons(self.annotations.global_rois, "global", 0)
            merged_count += old_count - len(self.annotations.global_rois)
        if len(self.annotations.inverse_global_rois) > 1:
            old_count = len(self.annotations.inverse_global_rois)
            self.annotations.inverse_global_rois = merge_overlapping_polygons(self.annotations.inverse_global_rois, "global_inverse", 0)
            merged_count += old_count - len(self.annotations.inverse_global_rois)
        if self.current_z in self.annotations.local_rois:
            rois = self.annotations.local_rois[self.current_z]
            if len(rois) > 1:
                old_count = len(rois)
                self.annotations.local_rois[self.current_z] = merge_overlapping_polygons(rois, "local", self.current_z)
                merged_count += old_count - len(self.annotations.local_rois[self.current_z])
        if self.current_z in self.annotations.inverse_local_rois:
            rois = self.annotations.inverse_local_rois[self.current_z]
            if len(rois) > 1:
                old_count = len(rois)
                self.annotations.inverse_local_rois[self.current_z] = merge_overlapping_polygons(rois, "local_inverse", self.current_z)
                merged_count += old_count - len(self.annotations.inverse_local_rois[self.current_z])
        if merged_count > 0:
            self._clear_edit_state()

    def undo_last_point(self):
        if self.current_polygon and self.current_polygon.points:
            self.current_polygon.points.pop()
            if not self.current_polygon.points:
                self.current_polygon = None
                self.drawing_state.reset()
                if self.on_drawing_state_changed: self.on_drawing_state_changed(self.drawing_state)
            self.update()

    def cancel_current(self):
        self.current_polygon = None
        self.drawing_state.reset()
        self._clear_edit_state()
        if self.on_drawing_state_changed: self.on_drawing_state_changed(self.drawing_state)
        self.update()

    def delete_last_roi(self):
        if not self.annotations: return
        if self.current_z in self.annotations.inverse_local_rois and self.annotations.inverse_local_rois[self.current_z]:
            self.annotations.inverse_local_rois[self.current_z].pop()
        elif self.current_z in self.annotations.local_rois and self.annotations.local_rois[self.current_z]:
            self.annotations.local_rois[self.current_z].pop()
        elif self.annotations.inverse_global_rois:
            self.annotations.inverse_global_rois.pop()
        elif self.annotations.global_rois:
            self.annotations.global_rois.pop()
        self._clear_edit_state()
        self.update()
        if self.on_annotation_changed: self.on_annotation_changed()

    def update_window(self, wc: int, ww: int):
        self.wc, self.ww = wc, ww
        self._update_pixmap()
        self.update()

# ════════════════════════════════════════════════════════════════════
# 交互标注对话框
# ════════════════════════════════════════════════════════════════════
class InteractiveAnnotationDialog(QDialog):
    def __init__(self, aligned_volume: np.ndarray, parent=None, max_display_size: int = 384):
        super().__init__(parent)
        self.original_shape = aligned_volume.shape 
        h, w = aligned_volume.shape[1], aligned_volume.shape[2]
        if max(h, w) > max_display_size:
            self.display_scale = max_display_size / max(h, w)
        else:
            self.display_scale = 1.0
        
        if self.display_scale < 1.0:
            new_h = int(h * self.display_scale)
            new_w = int(w * self.display_scale)
            depth = aligned_volume.shape[0]
            self.display_volume = np.empty((depth, new_h, new_w), dtype=np.int16)
            zoom_factors = (self.display_scale, self.display_scale)
            for z in range(depth):
                slice_data = aligned_volume[z].astype(np.float32)
                downsampled = ndimage.zoom(slice_data, zoom_factors, order=1)
                self.display_volume[z] = np.clip(downsampled, -32768, 32767).astype(np.int16)
        else:
            self.display_volume = aligned_volume
            self.display_scale = 1.0
        
        self.annotations = AnnotationData(
            z_range_start=0, z_range_end=aligned_volume.shape[0] - 1, scale_factor=self.display_scale
        )
        self.result_accepted = False
        self.setWindowTitle("交互式 ROI 标注")
        self.setMinimumSize(1200, 900)
        self.setStyleSheet("QDialog{background:#2d2d2d;} QLabel{color:white;}")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        toolbar = QWidget()
        toolbar.setFixedWidth(160)
        toolbar.setStyleSheet("""
            QWidget { background: #2d2d2d; border-radius: 6px; }
            QLabel { color: #888; font-size: 9pt; }
            QPushButton { background: #3d3d3d; color: white; border: 1px solid #555; border-radius: 4px; padding: 8px; }
            QPushButton:hover { background: #4d4d4d; }
            QPushButton:checked { background: #1a6fbf; border-color: #2a7fcf; }
        """)
        tl = QVBoxLayout(toolbar)
        
        if self.display_scale < 1.0:
            scale_info = QLabel(f"显示：{self.display_scale:.0%}")
            scale_info.setStyleSheet("color: #f39c12; font-size: 8pt;")
            tl.addWidget(scale_info)
        
        tl.addWidget(QLabel("配准后数据"))
        tl.addWidget(self._sep())
        tl.addWidget(self._section_label("交互模式"))
        
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for text, mode in[("漫游", "navigate"), ("调节窗位", "window"), ("清除挡板 - 全局", "global_roi"), ("清除骨质 - 单层", "local_roi")]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, m=mode: self.set_mode(m))
            self.mode_group.addButton(btn)
            tl.addWidget(btn)
            self.mode_buttons[mode] = btn
        self.mode_buttons["navigate"].setChecked(True)
        
        tl.addWidget(self._sep())
        tl.addWidget(self._section_label("★ 反向选择（保留框内）"))
        for text, mode in[("保留区域 - 全局", "global_inverse_roi"), ("保留区域 - 单层", "local_inverse_roi")]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton { background: #1a4a4a; color: #00dddd; border: 1px solid #00aaaa; }
                QPushButton:hover { background: #225555; }
                QPushButton:checked { background: #007777; border-color: #00dddd; color: white; }
            """)
            btn.clicked.connect(lambda _, m=mode: self.set_mode(m))
            self.mode_group.addButton(btn)
            tl.addWidget(btn)
            self.mode_buttons[mode] = btn
        
        tl.addWidget(self._sep())
        tl.addWidget(self._section_label("顶点编辑"))
        for text, mode in[("增加顶点", "add_vertex"), ("移动顶点", "move_vertex"), ("删除顶点", "delete_vertex")]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, m=mode: self.set_mode(m))
            self.mode_group.addButton(btn)
            tl.addWidget(btn)
            self.mode_buttons[mode] = btn
        
        tl.addWidget(self._sep())
        tl.addWidget(self._section_label("编辑操作"))
        for text, slot in[("取消绘制", self.cancel), ("删除上一个", self.delete_last), ("清除当前层", self.clear_slice)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            tl.addWidget(btn)
        
        tl.addStretch()
        stats_box = QGroupBox()
        stats_box.setStyleSheet("QGroupBox { background: #252525; border: 1px solid #444; border-radius: 4px; padding: 8px; }")
        sl = QVBoxLayout(stats_box)
        self.stats_label = QLabel("全局：0\n局部：0")
        self.stats_label.setStyleSheet("color: #88ff88; font-size: 9pt;")
        sl.addWidget(self.stats_label)
        tl.addWidget(stats_box)
        main_layout.addWidget(toolbar)
        
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = ImageCanvas()
        self.canvas.annotations = self.annotations
        self.canvas.on_slice_changed = self.on_slice_changed
        self.canvas.on_annotation_changed = self.update_stats
        self.canvas.on_window_changed = lambda wc, ww: None
        self.canvas.on_status_message = lambda m: None
        self.canvas.on_drawing_state_changed = self.on_drawing_state_changed
        self.canvas.set_volume(self.display_volume, self.display_scale)
        cl.addWidget(self.canvas, 1)
        
        bottom = QWidget()
        bottom.setStyleSheet("background: #2d2d2d; border-radius: 6px;")
        bottom.setFixedHeight(115)
        bl = QVBoxLayout(bottom)
        
        r1 = QHBoxLayout()
        lbl1 = QLabel("层面：")
        lbl1.setStyleSheet("color: white; font-size: 11pt; font-weight: bold;")
        r1.addWidget(lbl1)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 14px; background: #333; border-radius: 7px; }
            QSlider::handle:horizontal { background: #1a6fbf; width: 24px; margin: -5px 0; border-radius: 12px; }
        """)
        self.slice_slider.setRange(0, self.display_volume.shape[0] - 1)
        self.slice_slider.setValue(self.display_volume.shape[0] // 2)
        self.slice_slider.valueChanged.connect(self.on_slider)
        r1.addWidget(self.slice_slider, 1)
        self.slice_info = QLabel(f"{self.display_volume.shape[0] // 2 + 1} / {self.display_volume.shape[0]}")
        self.slice_info.setStyleSheet("color: white; font-weight: bold; min-width: 100px; font-size: 13pt;")
        r1.addWidget(self.slice_info)
        self.drawing_indicator = QLabel("")
        self.drawing_indicator.setStyleSheet("color: #ffb400; font-weight: bold; font-size: 10pt;")
        self.drawing_indicator.setMinimumWidth(150)
        r1.addWidget(self.drawing_indicator)
        bl.addLayout(r1)
        
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("处理范围："))
        self.range_start = QSpinBox()
        self.range_start.setStyleSheet("QSpinBox { background: #3d3d3d; color: white; padding: 6px; }")
        self.range_start.setRange(1, self.display_volume.shape[0])
        self.range_start.setValue(1)
        r2.addWidget(self.range_start)
        btn_s = QPushButton("设为起点")
        btn_s.setStyleSheet("QPushButton { background: #3d3d3d; color: white; padding: 6px 12px; }")
        btn_s.clicked.connect(lambda: self.range_start.setValue(self.canvas.current_z + 1))
        r2.addWidget(btn_s)
        r2.addWidget(QLabel("  -->   "))
        self.range_end = QSpinBox()
        self.range_end.setStyleSheet("QSpinBox { background: #3d3d3d; color: white; padding: 6px; }")
        self.range_end.setRange(1, self.display_volume.shape[0])
        self.range_end.setValue(self.display_volume.shape[0])
        r2.addWidget(self.range_end)
        btn_e = QPushButton("设为终点")
        btn_e.setStyleSheet("QPushButton { background: #3d3d3d; color: white; padding: 6px 12px; }")
        btn_e.clicked.connect(lambda: self.range_end.setValue(self.canvas.current_z + 1))
        r2.addWidget(btn_e)
        r2.addStretch()
        bl.addLayout(r2)
        cl.addWidget(bottom)
        main_layout.addWidget(center, 1)
        
        right = QWidget()
        right.setFixedWidth(160)
        right.setStyleSheet("""
            QWidget { background: #2d2d2d; border-radius: 6px; }
            QLabel { color: #888; font-size: 9pt; }
            QPushButton { background: #3d3d3d; color: white; border: 1px solid #555; padding: 8px; }
            QPushButton:hover { background: #4d4d4d; }
        """)
        rl = QVBoxLayout(right)
        rl.addWidget(self._section_label("操作"))
        confirm_btn = QPushButton("确认标注")
        confirm_btn.setStyleSheet("QPushButton { background: #27ae60; color: white; font-weight: bold; border: none; }")
        confirm_btn.clicked.connect(self.accept_annotation)
        rl.addWidget(confirm_btn)
        skip_btn = QPushButton("放弃标注")
        skip_btn.setStyleSheet("QPushButton { background: #e74c3c; color: white; font-weight: bold; border: none; }")
        skip_btn.clicked.connect(self.skip_annotation)
        rl.addWidget(skip_btn)
        
        rl.addWidget(self._sep())
        rl.addWidget(self._section_label("窗宽窗位"))
        preset_combo = QComboBox()
        preset_combo.addItems(["血管 CTA", "软组织", "骨窗"])
        preset_combo.setStyleSheet("QComboBox { background: #3d3d3d; color: white; padding: 5px; }")
        preset_combo.currentTextChanged.connect(self.apply_window_preset)
        rl.addWidget(preset_combo)
        
        rl.addWidget(self._sep())
        rl.addWidget(self._section_label("快捷键"))
        help_label = QLabel("1-4 切换模式\n5-7 顶点编辑\n8-9 反向保留\nEsc 取消\nDel 删除\nCtrl+Z 撤销")
        help_label.setStyleSheet("color: #aaa; font-size: 8pt;")
        rl.addWidget(help_label)
        rl.addStretch()
        main_layout.addWidget(right)
        self.update_stats()

    def _sep(self):
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFixedHeight(2)
        f.setStyleSheet("background: #444;")
        return f

    def _section_label(self, text):
        lbl = QLabel(f"-- {text} --")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #666; font-size: 9pt;")
        return lbl

    def set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        if mode in self.mode_buttons: self.mode_buttons[mode].setChecked(True)

    def on_slice_changed(self, z: int):
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(z)
        self.slice_slider.blockSignals(False)
        self.slice_info.setText(f"{z + 1} / {self.canvas.depth}")

    def on_slider(self, v: int):
        self.canvas.set_slice(v)
        self.slice_info.setText(f"{v + 1} / {self.canvas.depth}")

    def on_drawing_state_changed(self, state: DrawingState):
        if state.is_drawing:
            slice_count = state.get_slice_count()
            if slice_count > 1:
                min_z, max_z = state.get_slice_range()
                self.drawing_indicator.setText(f"跨层：{min_z + 1}→{max_z + 1} ({slice_count}层)")
            else:
                self.drawing_indicator.setText(f"绘制中：第{state.start_slice + 1}层")
        else:
            self.drawing_indicator.setText("")

    def apply_window_preset(self, name: str):
        presets = {"血管 CTA": (200, 600), "软组织": (40, 400), "骨窗": (400, 1800)}
        if name in presets:
            wc, ww = presets[name]
            self.canvas.wc, self.canvas.ww = wc, ww
            self.canvas._update_pixmap()
            self.canvas.update()

    def cancel(self): self.canvas.cancel_current()
    def delete_last(self): self.canvas.delete_last_roi()

    def clear_slice(self):
        z = self.canvas.current_z
        if (z in self.annotations.local_rois and self.annotations.local_rois[z]) or (z in self.annotations.inverse_local_rois and self.annotations.inverse_local_rois[z]):
            if QMessageBox.question(self, "确认", f"清除第 {z + 1} 层的所有 ROI?") == QMessageBox.Yes:
                if z in self.annotations.local_rois: self.annotations.local_rois[z].clear()
                if z in self.annotations.inverse_local_rois: self.annotations.inverse_local_rois[z].clear()
                self.canvas._clear_edit_state()
                self.canvas.update()
                self.update_stats()

    def update_stats(self):
        gc = len(self.annotations.global_rois)
        ls = len([z for z, r in self.annotations.local_rois.items() if r])
        lc = sum(len(r) for r in self.annotations.local_rois.values())
        igc = len(self.annotations.inverse_global_rois)
        ils = len([z for z, r in self.annotations.inverse_local_rois.items() if r])
        ilc = sum(len(r) for r in self.annotations.inverse_local_rois.values())
        text = f"清除 - 全局：{gc}\n清除 - 局部：{lc} ({ls}层)"
        if igc or ilc: text += f"\n保留 - 全局：{igc}\n保留 - 局部：{ilc} ({ils}层)"
        self.stats_label.setText(text)

    def accept_annotation(self):
        self.annotations.z_range_start = self.range_start.value() - 1
        self.annotations.z_range_end = self.range_end.value() - 1
        self.result_accepted = True
        self.accept()
 
    def skip_annotation(self):
        self.annotations.global_rois.clear()
        self.annotations.local_rois.clear()
        self.annotations.z_range_start = 0
        self.annotations.z_range_end = self.original_shape[0] - 1
        self.result_accepted = True
        self.accept()

    def get_scaled_annotations(self) -> AnnotationData:
        if abs(self.display_scale - 1.0) < 0.001: return self.annotations
        return self.annotations.get_scaled_rois(1.0)

    def keyPressEvent(self, event):
        mode_keys = {Qt.Key_1: "navigate", Qt.Key_2: "window", Qt.Key_3: "global_roi", Qt.Key_4: "local_roi",
                     Qt.Key_5: "add_vertex", Qt.Key_6: "move_vertex", Qt.Key_7: "delete_vertex",
                     Qt.Key_8: "global_inverse_roi", Qt.Key_9: "local_inverse_roi"}
        if event.key() in mode_keys: self.set_mode(mode_keys[event.key()])
        elif event.key() == Qt.Key_Escape: self.canvas.cancel_current()
        elif event.key() == Qt.Key_Delete: self.canvas.delete_last_roi()
        elif event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier: self.canvas.undo_last_point()
        else: super().keyPressEvent(event)

    def closeEvent(self, event):
        self.canvas.display_volume = None
        self.display_volume = None
        force_gc()
        super().closeEvent(event)

# ════════════════════════════════════════════════════════════════════
# 主处理线程 (★★★分段减影防溢出版★★★)
# ════════════════════════════════════════════════════════════════════
class ProcessThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    request_annotation = pyqtSignal(np.ndarray, str)
    
    def __init__(self, pre_series, post_series, output_dir, options):
        super().__init__()
        self.pre_series = pre_series
        self.post_series = post_series
        self.output_dir = output_dir
        self.options = options
        self.cancelled = False
        
        self.enable_annotation = options.get('enable_annotation', False)
        self.annotation_result: Optional[AnnotationData] = None
        self.annotation_ready = False

    def cancel(self):
        self.cancelled = True

    def set_annotation_result(self, annotations: AnnotationData):
        self.annotation_result = annotations
        self.annotation_ready = True

    def run(self):
        try:
            t0 = time.time()
            opt = self.options
            opt_level = opt.get('optimization_level', 'none')
            opt_name = OPTIMIZATION_LEVELS.get(opt_level, {}).get('name', '无')
            preserve_calc = opt.get('preserve_calcification', False)
            calc_threshold = opt.get('calcification_threshold', 600)
            num_chunks = opt.get('num_chunks', 1)

            self.log.emit("=" * 55)
            self.log.emit(f"CTA 减影 V4.3.1 (分段减影版) | 优化:{opt_name}")
            self.log.emit(f"  [分段] 设定分为 {num_chunks} 段处理")
            self.log.emit(f"  [内存] 初始：{get_memory_mb():.0f} MB")
            if self.enable_annotation: self.log.emit("[交互标注] 已启用 (每段分别弹出)")
            if preserve_calc: self.log.emit(f"  [钙化保留] 已启用 (阈值：{calc_threshold} HU, 2.5D 模式)")
            self.log.emit("=" * 55)

            reader = sitk.ImageSeriesReader()
            def get_files(series):
                s_dir = os.path.dirname(series.files[0][1])
                f_list = reader.GetGDCMSeriesFileNames(s_dir, series.series_uid)
                if f_list and len(f_list) == series.file_count: return list(f_list)
                return [f[1] for f in series.files]

            pre_files = get_files(self.pre_series)
            post_files = get_files(self.post_series)
            
            if not pre_files or not post_files:
                return self.finished_signal.emit(False, "读取失败：文件列表为空")

            if len(pre_files) != len(post_files):
                self.log.emit(f"\n  ⚠ 层数不等：平扫 {len(pre_files)} 层，增强 {len(post_files)} 层")
                pre_files, post_files = trim_to_common_z_range(pre_files, post_files, self.log.emit)
                if not pre_files or not post_files:
                    return self.finished_signal.emit(False, "层数对齐失败：两序列无有效重叠范围")

            total_slices = len(pre_files)
            chunk_size = math.ceil(total_slices / num_chunks)
            
            global_output_count = 0
            # 保证整个序列所有分段合并后拥有唯一的 SeriesUID，以便看图软件将其视为一个完整序列
            series_uid = generate_uid() 
            
            calcification_slice_count = 0
            calcification_pixel_count = 0
            annotation_global_rois_count = 0

            # ★★★ 核心分段大循环 ★★★
            for chunk_idx in range(num_chunks):
                if self.cancelled: return
                
                start_idx = chunk_idx * chunk_size
                end_idx = min(total_slices, (chunk_idx + 1) * chunk_size)
                if start_idx >= total_slices: break
                
                chunk_pre_files = pre_files[start_idx:end_idx]
                chunk_post_files = post_files[start_idx:end_idx]
                chunk_depth = len(chunk_pre_files)
                
                def update_prog(p):
                    self.progress.emit(int((chunk_idx * 100 + p) / num_chunks))
                    
                if num_chunks > 1:
                    self.log.emit(f"\n{'=' * 40}")
                    self.log.emit(f"▶ 开始处理分段 {chunk_idx+1}/{num_chunks} (源序列第 {start_idx+1} ~ {end_idx} 层)")
                    self.log.emit(f"{'=' * 40}")
                    
                # 1. 读取当前分段
                self.log.emit("\n[1/7] 读取平扫序列 (当前段)...")
                update_prog(5)
                pre_img = sitk.ReadImage(chunk_pre_files, sitk.sitkInt16)
                spacing_zyx = (pre_img.GetSpacing()[2], pre_img.GetSpacing()[1], pre_img.GetSpacing()[0])
                h, w = pre_img.GetHeight(), pre_img.GetWidth()

                ref_size = pre_img.GetSize()
                ref_spacing = pre_img.GetSpacing()
                ref_origin = pre_img.GetOrigin()
                ref_direction = pre_img.GetDirection()

                pre_vol = np.empty((chunk_depth, h, w), dtype=np.int16)
                for z in range(chunk_depth): pre_vol[z] = sitk.GetArrayFromImage(pre_img[:, :, z])

                shrink_factors =[
                    4 if ref_size[0] >= 512 else 2,
                    4 if ref_size[1] >= 512 else 2,
                    2 if ref_size[2] >= 100 else 1
                ]
                fixed_shrink = sitk.Cast(sitk.Shrink(pre_img, shrink_factors), sitk.sitkFloat32)

                del pre_img
                force_gc()
                self.log.emit(f"  [极限释放] 当前段平扫 C++ 对象已销毁，内存：{get_memory_mb():.0f} MB")
                update_prog(10)

                # 2. 配准当前分段
                self.log.emit("\n[2/7] 读取增强序列与 3D 旋转配准 (当前段)...")
                post_img = sitk.ReadImage(chunk_post_files, sitk.sitkInt16)
                moving_shrink = sitk.Cast(sitk.Shrink(post_img, shrink_factors), sitk.sitkFloat32)
                
                final_transform = compute_3d_registration_transform(fixed_shrink, moving_shrink, self.log.emit, lambda v: update_prog(10 + v))
                del fixed_shrink, moving_shrink
                force_gc()
                if self.cancelled: return

                # 3. 空间重采样
                self.log.emit("\n[3/7] 应用空间重采样...")
                ref_img = sitk.Image(ref_size, sitk.sitkInt16)
                ref_img.SetOrigin(ref_origin)
                ref_img.SetSpacing(ref_spacing)
                ref_img.SetDirection(ref_direction)

                aligned_img = sitk.Resample(post_img, ref_img, final_transform, sitk.sitkLinear, -1000.0, sitk.sitkInt16)
                del post_img, ref_img
                force_gc()
                
                aligned_vol = np.empty((chunk_depth, h, w), dtype=np.int16)
                for z in range(chunk_depth): aligned_vol[z] = sitk.GetArrayFromImage(aligned_img[:, :, z])
                del aligned_img
                force_gc()
                update_prog(35)
                
                # 4. 交互标注
                annotation_data = None
                z_start, z_end = 0, chunk_depth - 1
                
                if self.enable_annotation:
                    self.log.emit("\n[4/7] 等待交互标注...")
                    update_prog(40)
                    
                    self.annotation_ready = False
                    self.annotation_result = None
                    chunk_title = f" (分段 {chunk_idx+1}/{num_chunks})" if num_chunks > 1 else ""
                    self.request_annotation.emit(aligned_vol, chunk_title)
                    
                    while not self.annotation_ready and not self.cancelled: self.msleep(100)
                    if self.cancelled: return
                    
                    if self.annotation_result:
                        annotation_data = self.annotation_result
                        z_start = annotation_data.z_range_start
                        z_end = annotation_data.z_range_end
                        annotation_global_rois_count += len(annotation_data.global_rois)
                        self.log.emit(f"[标注完成] 本段范围：第 {z_start + 1} ~ {z_end + 1} 层")
                else:
                    self.log.emit("\n[4/7] 跳过交互标注")
                update_prog(45)

                # 5. 减影
                self.log.emit(f"\n[5/7] 减影计算...")
                spacing_xy = (spacing_zyx[1], spacing_zyx[2])
                prev_calc_labels = None
                 
                for z in range(z_start, z_end + 1):
                    if self.cancelled: return
                    pre_s = pre_vol[z].astype(np.float32)
                    post_s = aligned_vol[z].astype(np.float32)
                    
                    if pre_s.max() > -500:
                        diff_pos = np.clip(post_s - pre_s, 0, None)
                        cache = precompute_slice_cache(pre_s, diff_pos, opt['quality_mode'])
                        equip = detect_equipment(pre_s, post_s, body=cache.body)
                        
                        # 换算全局 Z 比例，以获取正确的区域解剖位置
                        global_z = start_idx + z
                        zone = get_anatomical_zone(global_z, total_slices)
                        adj_bone, adj_vessel = get_zone_adjusted_params(zone, opt['bone_strength'], opt['vessel_sensitivity'])
                        z_ratio = global_z / total_slices
                        bone_zones = detect_problematic_bone_zones(pre_s, z_ratio)
                        
                        if opt_level == 'none':
                            res = quality_subtraction_cached(pre_s, post_s, adj_bone, adj_vessel, cache) if opt['quality_mode'] else fast_subtraction(pre_s, post_s, adj_bone, adj_vessel, body=cache.body)
                        elif opt_level == 'light':
                            res = optimized_subtraction_light_v3(pre_s, post_s, adj_bone, adj_vessel, cache)
                        elif opt_level == 'standard':
                            res = optimized_subtraction_standard_v3(pre_s, post_s, adj_bone, adj_vessel, cache, spacing_xy)
                        else:
                            res = optimized_subtraction_deep_v3(pre_s, post_s, adj_bone, adj_vessel, cache, spacing_xy, spacing_zyx)

                        if zone == 'skull_base':
                            gain = np.ones_like(res, dtype=np.float32)
                            gain = enhanced_bone_suppression_v299(pre_s, diff_pos, gain, bone_zones, adj_bone)
                            res = res * gain

                        res[equip] = 0
                        res[~cache.body] = 0
                        
                        if opt['clean_bone_edges']: res = clean_bone_edges_cached(res, pre_s, cache, 'quality' if opt['quality_mode'] else 'fast')
                        if opt['smooth_sigma'] > 0: res = edge_preserving_smooth(res, pre_s, opt['smooth_sigma'])
                        
                        if preserve_calc:
                            calc_mask, prev_calc_labels = detect_calcification_2d_slice(pre_s, diff_pos, cache.body, calc_threshold, adj_vessel, prev_calc_labels)
                            if calc_mask is not None and calc_mask.sum() > 0:
                                calcification_slice_count += 1
                                calcification_pixel_count += calc_mask.sum()
                                preserved_value = np.maximum(post_s - 50.0, 0) * opt['vessel_enhance']
                                res[calc_mask] = preserved_value[calc_mask]
                                     
                        res_scaled = res * opt['vessel_enhance']
                        aligned_vol[z] = np.clip(res_scaled, -32768, 32767).astype(np.int16)
                    else:
                        aligned_vol[z] = 0
                        
                    if z % 50 == 0: update_prog(45 + int(((z - z_start) / max(1, z_end - z_start + 1)) * 30))

                del pre_vol
                force_gc()
                result_vol = aligned_vol
                
                if annotation_data and (annotation_data.global_rois or any(annotation_data.local_rois.values()) or annotation_data.inverse_global_rois or any(annotation_data.inverse_local_rois.values())):
                    self.log.emit("  [应用 ROI] 挖除标记区域...")
                    result_vol = apply_roi_masks_to_volume(result_vol, annotation_data, z_start, z_end, self.log.emit)
                    force_gc()

                # 6. 后处理
                self.log.emit("\n[6/7] 后处理...")
                min_area_2d = opt['min_vessel_size'] * 3
                result_vol = apply_3d_cleanup(result_vol, min_area=min_area_2d, log_cb=self.log.emit)
                
                if opt_level == 'deep': result_vol = shape_analysis_fast(result_vol, spacing_zyx, self.log.emit)
                    
                update_prog(80)
                force_gc()

                # 7. 写出本段结果
                self.log.emit(f"\n[7/7] 写入磁盘...")
                os.makedirs(self.output_dir, exist_ok=True)
                
                range_vol = result_vol[z_start:z_end + 1]
                g_min, g_max = float(range_vol.min()), float(range_vol.max())
                g_slope = (g_max - g_min) / 4095.0 if g_max > g_min else 1.0

                for z in range(z_start, z_end + 1):
                    if self.cancelled: return
                    
                    ds = pydicom.dcmread(chunk_pre_files[z])
                    ds.SpecificCharacterSet = 'ISO_IR 192'
                     
                    pix = result_vol[z]
                    pix_int = ((pix - g_min) / (g_max - g_min) * 4095).astype(np.int16) if g_max > g_min else np.zeros_like(pix, dtype=np.int16)
                     
                    ds.PixelData = pix_int.tobytes()
                    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
                    ds.PixelRepresentation, ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 1, 'MONOCHROME2'
                    
                    for tag in['LossyImageCompression', 'LossyImageCompressionRatio', 'LossyImageCompressionMethod']:
                        if hasattr(ds, tag): delattr(ds, tag)
                    
                    ds.RescaleSlope, ds.RescaleIntercept = g_slope, g_min
                    ds.WindowCenter, ds.WindowWidth = opt['wc'], opt['ww']
                    
                    calc_tag = "+Calc" if preserve_calc else ""
                    ann_tag = "+ROI" if self.enable_annotation and annotation_data else ""
                    ds.SeriesDescription = f"CTA Sub V3.3.1 [{opt_name}]{calc_tag}{ann_tag}"
                    ds.SeriesInstanceUID = series_uid # 保持跨段相同
                    ds.SOPInstanceUID = generate_uid()
                    
                    # 递增全局实例号，让合并后的序列完美连贯
                    ds.InstanceNumber = global_output_count + 1
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                    ds.save_as(os.path.join(self.output_dir, f"SUB_{global_output_count + 1:04d}.dcm"), write_like_original=False)
                    
                    global_output_count += 1
                    if z % 50 == 0: update_prog(80 + int(((z - z_start) / max(1, z_end - z_start + 1)) * 15))
                
                del result_vol
                force_gc()
            # ▲▲▲ 分段大循环结束 ▲▲▲

            self.progress.emit(95)
            force_gc()
            self.progress.emit(100)
            elapsed = time.time() - t0
            
            finish_msg = f"处理完成!\n输出：{global_output_count} 层\n耗时：{elapsed:.1f}秒\n目录：{self.output_dir}"
            if num_chunks > 1: finish_msg += f"\n分段信息：成功分为 {num_chunks} 段处理"
            if preserve_calc and calcification_slice_count > 0: finish_msg += f"\n钙化保留：{calcification_slice_count} 层"
            if self.enable_annotation and annotation_global_rois_count > 0: finish_msg += f"\n应用 ROI: 全局{annotation_global_rois_count}个"
            
            self.log.emit(f"\n{'=' * 55}")
            self.log.emit(f"✅ 完成！耗时：{elapsed:.1f}s")
            self.log.emit(f"[内存] 最终：{get_memory_mb():.0f} MB")
            self.finished_signal.emit(True, finish_msg)

        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))

# ════════════════════════════════════════════════════════════════════
# 主界面 UI
# ════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.all_series, self.selected_pre, self.selected_post, self.recommendations = {}, None, None, None
        self.proc_thread, self._all_presets = None, load_all_presets()
        self.current_data_dir = ""
        self.annotation_dialog: Optional[InteractiveAnnotationDialog] = None
        self.init_ui()
    def init_ui(self):
        self.setWindowTitle("头颅 CTA 减影-whw 版 V4.3.1 (分版减影版)")
        icon_path = resource_path("whw.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(1000, 950)
        self.resize(1100, 1000)
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.setStyleSheet("""
        QMainWindow,QWidget{background:#F4F5F7;font-family:"Microsoft YaHei";font-size:9pt;}
        QGroupBox{background:#fff;border:1px solid #D0D5DE;border-radius:6px;margin-top:10px;padding:10px;font-weight:600;}
        QGroupBox::title{subcontrol-origin:margin;left:10px;top:0;padding:0 4px;background:#fff;color:#1A6FBF;}
        QTextEdit{background:#1C1E26;color:#C8D0E0;border-radius:4px;font-family:Consolas;font-size:8.5pt;}
        QProgressBar{background:#D0D5DE;border:none;border-radius:4px;height:18px;text-align:center;color:white;font-weight:bold;}
        QProgressBar::chunk{background:#1A6FBF;border-radius:4px;text-align:center;}
        QCheckBox{color:#2c3e50;font-size:9pt;}
        QCheckBox::indicator{width:16px;height:16px;}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)
        
        dir_grp = QGroupBox("1 数据目录")
        dir_lay = QHBoxLayout(dir_grp)
        self.data_dir_edit = QLineEdit()
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_data_dir)
        self.scan_btn = QPushButton("扫描")
        self.scan_btn.setStyleSheet("background:#1A6FBF;color:white;border-radius:4px;padding:6px 16px;")
        self.scan_btn.clicked.connect(self.scan_directory)
        dir_lay.addWidget(self.data_dir_edit)
        dir_lay.addWidget(browse_btn)
        dir_lay.addWidget(self.scan_btn)
        dir_grp.setFixedHeight(95)
        root.addWidget(dir_grp)

        ser_grp = QGroupBox("2 序列配对")
        ser_lay = QVBoxLayout(ser_grp)
        r1 = QHBoxLayout()
        self.pre_combo, self.post_combo = QComboBox(), QComboBox()
        r1.addWidget(QLabel("平扫："))
        r1.addWidget(self.pre_combo, 1)
        r1.addWidget(QLabel("增强："))
        r1.addWidget(self.post_combo, 1)
        ser_lay.addLayout(r1)
        r2 = QHBoxLayout()
        self.analyze_btn = QPushButton("分析特征")
        self.analyze_btn.setStyleSheet("background:#1A6FBF;color:white;border-radius:4px;padding:6px 16px;")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_params)
        r2.addWidget(self.analyze_btn)
        r2.addStretch()
        ser_lay.addLayout(r2) 
        ser_grp.setFixedHeight(135)
        root.addWidget(ser_grp)

        out_grp = QGroupBox("3 输出目录 (自动生成，自动编号)")
        out_lay = QHBoxLayout(out_grp)
        self.out_edit = QLineEdit()
        self.out_edit.setReadOnly(True)
        self.out_edit.setStyleSheet("background:#E8EAED;")
        out_lay.addWidget(self.out_edit)
        out_grp.setFixedHeight(90)
        root.addWidget(out_grp)

        mode_grp = QGroupBox("4 处理模式与优化级别")
        mode_lay = QVBoxLayout(mode_grp)
        mr = QHBoxLayout()
        self.fast_radio = QRadioButton("⚡ 快速（常规，适用扫描质量好）")
        self.quality_radio = QRadioButton("⚡⚡ 精细（适用扫描质量较差）")
        self.fast_radio.setChecked(True)
        mr.addWidget(self.fast_radio)
        mr.addWidget(self.quality_radio)
        mr.addStretch()
        mode_lay.addLayout(mr)
        self.mode_label = QLabel("")
        mode_lay.addWidget(self.mode_label)
        mode_lay.addWidget(QFrame(frameShape=QFrame.HLine))
        
        self.opt_group = QButtonGroup(self)
        self.opt_none = QRadioButton("无优化（原生模式）")
        self.opt_none.setChecked(True)
        self.opt_light = QRadioButton("轻量级 - 自适应阈值+ICA 保护")
        self.opt_standard = QRadioButton("标准级 - +亚像素 + 静脉窦保护")
        self.opt_deep = QRadioButton("深度级 - +Top-Hat 血管增强")
        self.opt_group.addButton(self.opt_none)
        self.opt_group.addButton(self.opt_light)
        self.opt_group.addButton(self.opt_standard)
        self.opt_group.addButton(self.opt_deep)
        mode_lay.addWidget(self.opt_none)
        mode_lay.addWidget(self.opt_light)
        mode_lay.addWidget(self.opt_standard)
        mode_lay.addWidget(self.opt_deep)
        mode_grp.setMinimumHeight(230)
        mode_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root.addWidget(mode_grp)

        param_grp = QGroupBox("5 参数调节")
        param_lay = QVBoxLayout(param_grp)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        for name in self._all_presets: self.preset_combo.addItem(name, self._all_presets[name])
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self.apply_preset)
        self.preset_status = QLabel("")
        self.preset_status.setStyleSheet("color:#27AE60;")
        preset_row.addWidget(QLabel("预设："))
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(apply_btn)
        preset_row.addWidget(self.preset_status)
        preset_row.addStretch()
        param_lay.addLayout(preset_row)

        # ★★★ 分段处理选项 ★★★
        chunk_lay = QHBoxLayout()
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 10)
        self.chunk_spin.setValue(1)
        self.chunk_spin.setStyleSheet("QSpinBox { font-weight: bold; font-size: 10pt; width: 50px; }")
        chunk_lay.addWidget(QLabel("分段减影 (防32位系统内存溢出):"))
        chunk_lay.addWidget(self.chunk_spin)
        chunk_label = QLabel("段 (如层数超过400层，建议选 2 或 3)")
        chunk_label.setStyleSheet("color:#e74c3c; font-weight:bold;")
        chunk_lay.addWidget(chunk_label)
        chunk_lay.addStretch()
        param_lay.addLayout(chunk_lay)

        p_lay = QHBoxLayout()
        self.bone_slider = QSlider(Qt.Horizontal)
        self.bone_slider.setRange(5, 25)
        self.bone_slider.setValue(12)
        self.bone_label = QLabel("1.2")
        self.bone_slider.valueChanged.connect(lambda v: self.bone_label.setText(f"{v/10:.1f}"))
        self.enhance, self.smooth = QDoubleSpinBox(), QDoubleSpinBox()
        self.enhance.setRange(0.5, 5.0)
        self.enhance.setValue(2.0)
        self.smooth.setRange(0, 1.5)
        self.smooth.setValue(0.7)
        self.clean_check = QCheckBox("去骨边")
        self.clean_check.setChecked(True)
        self.wc, self.ww = QSpinBox(), QSpinBox()
        self.wc.setRange(-500, 1000)
        self.wc.setValue(200)
        self.ww.setRange(10, 2000)
        self.ww.setValue(400)
        p_lay.addWidget(QLabel("骨骼抑制："))
        p_lay.addWidget(self.bone_slider)
        p_lay.addWidget(self.bone_label)
        p_lay.addWidget(QLabel("增强："))
        p_lay.addWidget(self.enhance)
        p_lay.addWidget(self.clean_check)
        p_lay.addWidget(QLabel("降噪："))
        p_lay.addWidget(self.smooth)
        p_lay.addWidget(QLabel("窗："))
        p_lay.addWidget(self.wc)
        p_lay.addWidget(self.ww)
        p_lay.addStretch()
        param_lay.addLayout(p_lay)

        calc_lay = QHBoxLayout()
        self.calcification_check = QCheckBox("保留血管壁钙化")
        self.calcification_check.setChecked(True)
        self.calc_threshold_label = QLabel("阈值 (HU):")
        self.calc_threshold_spin = QSpinBox()
        self.calc_threshold_spin.setRange(400, 1000)
        self.calc_threshold_spin.setValue(600)
        calc_lay.addWidget(self.calcification_check)
        calc_lay.addWidget(self.calc_threshold_label)
        calc_lay.addWidget(self.calc_threshold_spin)
        
        calc_lay.addStretch()
        param_lay.addLayout(calc_lay)

        ann_lay = QHBoxLayout()
        self.annotation_check = QCheckBox("启用交互标注 (配准后弹出标注窗口)")
        self.annotation_check.setStyleSheet("color:#e67e22;font-weight:bold;")
        ann_lay.addWidget(self.annotation_check)       
        ann_lay.addStretch()
        param_lay.addLayout(ann_lay)

        param_grp.setMinimumHeight(200)
        param_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root.addWidget(param_grp)

        act_lay = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始")
        self.start_btn.setStyleSheet("background:#1A6FBF;color:white;border-radius:5px;font-weight:bold;padding:0 24px;height:34px;")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        act_lay.addWidget(self.start_btn)
        act_lay.addWidget(self.progress, 1)
        act_lay.addWidget(self.cancel_btn)
        root.addLayout(act_lay)

        log_grp = QGroupBox("日志")
        log_lay = QVBoxLayout(log_grp)
        log_lay.setContentsMargins(6, 6, 6, 6)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)
        log_lay.addWidget(self.log_text)
        root.addWidget(log_grp, 1)

        self.log("头颅 CTA 减影-whw 版 V4.3.1 (分段减影版)")
        self.log("=" * 50)
        self.log("V4.3.1 分段减影版说明:")
        self.log("增加【分段减影】控制参数，可指定切分段数处理长序列")
        self.log("600多层的大序列切分 2 段，内存可无压力")
        self.log("  [保证] 输出图像将拥有连续的实例号(InstanceNumber)，三维重建无缝合并！")
        self.log("=" * 50)

    def log(self, msg):
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        QApplication.processEvents()

    def browse_data_dir(self):
        d = QFileDialog.getExistingDirectory(self)
        if d:
            self.data_dir_edit.setText(d)
            self.current_data_dir = d
            self.update_output_dir_preview()

    def update_output_dir_preview(self):
        if self.current_data_dir:
            preview_dir = get_next_output_dir(self.current_data_dir, "CTA_SUB")
            self.out_edit.setText(preview_dir)

    def _get_opt_level(self):
        if self.opt_light.isChecked(): return 'light'
        if self.opt_standard.isChecked(): return 'standard'
        if self.opt_deep.isChecked(): return 'deep'
        return 'none'

    def apply_preset(self):
        cfg = self.preset_combo.currentData()
        if cfg:
            self.bone_slider.setValue(int(cfg["bone_strength"] * 10))
            self.enhance.setValue(cfg["vessel_enhance"])
            self.smooth.setValue(cfg["smooth_sigma"])
            self.clean_check.setChecked(cfg["clean_bone_edges"])
            self.wc.setValue(cfg["wc"])
            self.ww.setValue(cfg["ww"])
            if 'preserve_calcification' in cfg: self.calcification_check.setChecked(cfg['preserve_calcification'])
            if self.recommendations is None: self.recommendations = {}
            self.recommendations['vessel_sensitivity'] = cfg.get('vessel_sensitivity', 1.0)
            self.recommendations['min_vessel_size'] = cfg.get('min_vessel_size', 5)
            self.preset_status.setText(f"✓ 已应用：{self.preset_combo.currentText()}")
            self.log(f"✅ 应用预设：{self.preset_combo.currentText()}")

    def scan_directory(self):
        d = self.data_dir_edit.text()
        if not d: return
        self.current_data_dir = d
        self.pre_combo.clear()
        self.post_combo.clear()
        self.scan_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.scan_thread = SeriesScanThread(d)
        self.scan_thread.progress.connect(self.progress.setValue)
        self.scan_thread.log.connect(self.log)
        self.scan_thread.finished_signal.connect(self.on_scan_finished)
        self.scan_thread.start()

    def on_scan_finished(self, all_series, pairs, cta_series):
        self.scan_btn.setEnabled(True)
        for s in sorted(all_series.values(), key=lambda x: x.series_number):
            txt = f"#{s.series_number:03d} | {s.file_count}张 | {'[C+]' if s.contrast_status is True else '[C-]' if s.contrast_status is False else ''} {s.series_description[:40]}"
            self.pre_combo.addItem(txt, s)
            self.post_combo.addItem(txt, s)
        if pairs:
            self.pre_combo.setCurrentIndex(self.pre_combo.findData(pairs[0][0]))
            self.post_combo.setCurrentIndex(self.post_combo.findData(pairs[0][1]))
            # 智能建议分段数
            total_slices = min(pairs[0][0].file_count, pairs[0][1].file_count)
            if total_slices > 400:
                self.chunk_spin.setValue(2)
                self.log(f"🔔 智能提示：序列长达 {total_slices} 层，已自动建议分为 2 段处理。")
        self.analyze_btn.setEnabled(True)
        self.update_output_dir_preview()
        if pairs or cta_series:
            preset = detect_manufacturer_preset((pairs[0][0] if pairs else cta_series[0]).files)
            if preset and self.preset_combo.findText(preset) >= 0:
                self.preset_combo.setCurrentIndex(self.preset_combo.findText(preset))
                self.apply_preset()
                self.log(f"🔍 检测设备：{preset}")

    def analyze_params(self):
        if not self.pre_combo.currentData() or not self.post_combo.currentData() or self.pre_combo.currentData() == self.post_combo.currentData():
            return QMessageBox.warning(self, "提示", "请选择不同序列")
        self.selected_pre, self.selected_post = self.pre_combo.currentData(), self.post_combo.currentData()
        self.analyze_btn.setEnabled(False)
        self.param_thread = ParamAnalyzeThread(self.selected_pre.files, self.selected_post.files)
        self.param_thread.progress.connect(self.progress.setValue)
        self.param_thread.log.connect(self.log)
        self.param_thread.finished_signal.connect(self.on_analyze_finished)
        self.param_thread.start()

    def on_analyze_finished(self, res):
        self.analyze_btn.setEnabled(True)
        if 'error' in res: return QMessageBox.warning(self, "错误", res['error'])
        if self.recommendations is None: self.recommendations = {}
        self.recommendations.update(res)
        self.bone_slider.setValue(int(res.get('bone_strength', 1.2) * 10))
        self.enhance.setValue(res.get('vessel_enhance', 2.0))
        self.clean_check.setChecked(res.get('clean_bone_edges', True))
        self.smooth.setValue(res.get('smooth_sigma', 0.7))
        self.wc.setValue(res.get('wc', 200))
        self.ww.setValue(res.get('ww', 400))
        self.calcification_check.setChecked(res.get('preserve_calcification', True))
        if res.get('recommended_mode', 'fast') == 'fast': self.fast_radio.setChecked(True)
        else: self.quality_radio.setChecked(True)
        opt_level = res.get('recommended_opt', 'light')
        if opt_level == 'none': self.opt_none.setChecked(True)
        elif opt_level == 'light': self.opt_light.setChecked(True)
        elif opt_level == 'standard': self.opt_standard.setChecked(True)
        elif opt_level == 'deep': self.opt_deep.setChecked(True)
        self.mode_label.setText(f"智能评分：{res.get('quality_score', 0) * 100:.0f}分")
        self.start_btn.setEnabled(True)

    def start_processing(self):
        if not self.current_data_dir: return QMessageBox.warning(self, "提示", "请先选择数据目录")
        output_dir = get_next_output_dir(self.current_data_dir, "CTA_SUB")
        self.out_edit.setText(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        opt = {
            'quality_mode': self.quality_radio.isChecked(), 
            'bone_strength': self.bone_slider.value() / 10.0,
            'vessel_sensitivity': (self.recommendations or {}).get('vessel_sensitivity', 1.0),
            'vessel_enhance': self.enhance.value(), 
            'clean_bone_edges': self.clean_check.isChecked(),
            'min_vessel_size': (self.recommendations or {}).get('min_vessel_size', 5), 
            'smooth_sigma': self.smooth.value(),
            'wc': self.wc.value(), 
            'ww': self.ww.value(),
            'optimization_level': self._get_opt_level(),
            'preserve_calcification': self.calcification_check.isChecked(),
            'calcification_threshold': self.calc_threshold_spin.value(),
            'enable_annotation': self.annotation_check.isChecked(),
            'num_chunks': self.chunk_spin.value() # ★ 传递分段数 ★
        }
        
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.proc_thread = ProcessThread(self.selected_pre, self.selected_post, output_dir, opt)
        self.proc_thread.progress.connect(self.progress.setValue)
        self.proc_thread.log.connect(self.log)
        self.proc_thread.finished_signal.connect(self.on_finished)
        self.proc_thread.request_annotation.connect(self.on_annotation_requested)
        self.proc_thread.start()

    def on_annotation_requested(self, aligned_volume: np.ndarray, chunk_title: str):
        self.log(f"⏸ 打开交互标注窗口{chunk_title} (降采样显示)...")
        self.annotation_dialog = InteractiveAnnotationDialog(aligned_volume, self)
        if chunk_title:
            self.annotation_dialog.setWindowTitle(f"交互式 ROI 标注 - 配准后数据 {chunk_title}")
            
        result = self.annotation_dialog.exec_()
        if result == QDialog.Accepted and self.annotation_dialog.result_accepted:
            scaled_annotations = self.annotation_dialog.get_scaled_annotations()
            self.proc_thread.set_annotation_result(scaled_annotations)
            self.log("✅ 标注完成，继续减影...")
        else:
            self.proc_thread.cancel()
            self.log("❌ 用户取消标注")
        
        self.annotation_dialog.canvas.display_volume = None
        self.annotation_dialog.display_volume = None
        self.annotation_dialog = None
        force_gc()
        self.log(f"  [内存] 标注窗口关闭后：{get_memory_mb():.0f} MB")

    def cancel(self):
        if self.proc_thread:
            self.proc_thread.cancel()
            self.log("用户取消处理")

    def on_finished(self, success, msg):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.update_output_dir_preview()
        if success:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "错误", msg)

# ════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if os.name == 'nt':
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("whw.cta.subtraction.331")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    palette = app.palette()
    palette.setColor(palette.Window, QColor(244, 245, 247))
    palette.setColor(palette.WindowText, QColor(44, 62, 80))
    palette.setColor(palette.Base, QColor(255, 255, 255))
    palette.setColor(palette.AlternateBase, QColor(248, 249, 250))
    palette.setColor(palette.Text, QColor(44, 62, 80))
    palette.setColor(palette.Button, QColor(255, 255, 255))
    palette.setColor(palette.ButtonText, QColor(44, 62, 80))
    palette.setColor(palette.Highlight, QColor(26, 111, 191))
    palette.setColor(palette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())