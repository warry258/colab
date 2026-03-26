"""
头颅 CTA 减影-whw 版 V5.0 (三维形态学终极破局版)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[V5.0 新增] 微弱血管非线性提亮（AChA / 脉络膜支 / 深穿支）
  减影后 50~150 HU 的低亮度血管信号经余弦平滑曲线提升至 VR 显示门槛，
  最大增益 ×1.35（位于区间中心约 100 HU），两端渐变至 ×1.0，无硬边界。
  提亮前经 2D 连通性过滤（≥3 像素），排除孤立噪点，保留空间连续血管。
[V5.0 修复] 解剖三区分区坐标系重构
  旧版以 z/total_slices 比例划分，头颈CTA/分段/选段时分区严重错位。
  新版以距顶点绝对层数（slices_from_top，0.625mm/层）为唯一分区坐标：
    颅盖区  convexity    :   0~119层 ( 0~ 74mm)
    精细分区 willis_circle: 102~185层 (63.8~115.6mm，眶顶板消失→鞍底/岩尖上界)
    颅底区  skull_base   : ≥186层   (≥116.3mm，鞍底/岩尖上界+8层余量)
  各边界两侧 ±8 层线性渐变，消除相邻层亮度阶梯。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[定制] 移除了 血管缩放 功能
[定制] 移除了 MIP预览 功能
[终极优化] 增加分段减影模式 - 彻底解决 32位系统 600层+ 内存溢出[修复] 金属保护区 UI状态、Z轴分段映射
[核心重构] 三维立体打击逻辑：
  1. 形态学开运算识别厚实颅底骨，正常执行骨抑制（去骨完美）
  2. 提取细长放射状金属伪影，在最终图像输出前精准剔除（消除星芒伪影）
  3. 提取 >1700 HU 金属本体，输出时强制赋值 50 HU（供后期三维重建完美上色）
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
print("头颅 CTA 减影-whw 版 V5.0")

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
        coords =[(p.x(), p.y()) for p in self.points]
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
    vertex_slices: Set[int] = field(default_factory=set)  # 实际添加了顶点的层
    polygon: Optional[PolygonROI] = None

    def reset(self):
        self.is_drawing = False
        self.start_slice = 0
        self.current_slice = 0
        self.visited_slices.clear()
        self.vertex_slices.clear()
        self.polygon = None

    def get_slice_range(self) -> Tuple[int, int]:
        # 用有顶点的层决定范围，而不是浏览过的层
        ref = self.vertex_slices if self.vertex_slices else self.visited_slices
        if not ref:
            return (self.start_slice, self.start_slice)
        return (min(ref), max(ref))

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
    metal_rois: Dict[int, List[PolygonROI]] = field(default_factory=lambda: defaultdict(list))
    clip_rois: Dict[int, List[PolygonROI]] = field(default_factory=lambda: defaultdict(list))  # 瘤夹保护区
    z_range_start: int = 0
    z_range_end: int = -1
    series_uid: str = ""
    scale_factor: float = 1.0
    # Willis环手动标定（0-based，对应交互窗口 display_volume 的层索引）
    willis_layer_top: Optional[int] = None
    willis_layer_bot: Optional[int] = None

    def get_scaled_rois(self, target_scale: float) -> 'AnnotationData':
        if abs(self.scale_factor - target_scale) < 0.001:
            return self
        
        ratio = target_scale / self.scale_factor
        scaled = AnnotationData(
            z_range_start=self.z_range_start,
            z_range_end=self.z_range_end,
            series_uid=self.series_uid,
            scale_factor=target_scale,
            willis_layer_top=self.willis_layer_top,
            willis_layer_bot=self.willis_layer_bot,
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

        for z, rois in self.metal_rois.items():
            for roi in rois:
                scaled.metal_rois[z].append(roi.scale(ratio))

        for z, rois in self.clip_rois.items():
            for roi in rois:
                scaled.clip_rois[z].append(roi.scale(ratio))

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
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def _get_short_path(path: str) -> str:
    if os.name != 'nt':
        return path
    try:
        if not os.path.exists(path):
            return path
        buf = ctypes.create_unicode_buffer(1024)
        ret = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024)
        if ret == 0:
            return path
        short = buf.value
        if short and os.path.exists(short):
            return short
        return path
    except Exception:
        return path

def _sitk_read_image_safe(file_list: list, pixel_type=None) -> 'sitk.Image':
    if pixel_type is None:
        pixel_type = sitk.sitkInt16

    safe_files =[_get_short_path(f) for f in file_list]
    try:
        return sitk.ReadImage(safe_files, pixel_type)
    except Exception as e1:
        pass

    try:
        import pydicom as _pydicom

        slices =[]
        ds0 = None
        for fp in file_list:
            ds = _pydicom.dcmread(fp)
            arr = ds.pixel_array.astype(np.float32)
            slope     = float(getattr(ds, 'RescaleSlope',     1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            if slope != 1.0 or intercept != 0.0:
                arr = arr * slope + intercept
            slices.append(arr.astype(np.int16))
            if ds0 is None:
                ds0 = ds

        vol = np.stack(slices, axis=0)          
        img = sitk.GetImageFromArray(vol)        

        try:
            px_spacing = ds0.PixelSpacing          
            row_sp = float(px_spacing[0])
            col_sp = float(px_spacing[1])
            z_sp = float(getattr(ds0, 'SpacingBetweenSlices',
                         getattr(ds0, 'SliceThickness', 1.0)))
            img.SetSpacing((col_sp, row_sp, z_sp))   
        except Exception:
            pass  

        try:
            iop = ds0.ImageOrientationPatient       
            ipp = ds0.ImagePositionPatient          
            F =[float(v) for v in iop]
            row_dir    = F[0:3]   
            col_dir    = F[3:6]   
            z_dir = [
                row_dir[1]*col_dir[2] - row_dir[2]*col_dir[1],
                row_dir[2]*col_dir[0] - row_dir[0]*col_dir[2],
                row_dir[0]*col_dir[1] - row_dir[1]*col_dir[0],
            ]
            direction = tuple(row_dir + col_dir + z_dir)
            img.SetDirection(direction)
            origin = tuple(float(v) for v in ipp)
            img.SetOrigin(origin)
        except Exception:
            pass  

        return img

    except Exception as e2:
        raise RuntimeError(
            f"sitk.ReadImage 失败（短路径降级无效）且 pydicom 回退也失败。\n"
            f"SimpleITK 错误：{e1}\npydicom 错误：{e2}"
        )

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
    "standard": { "name": "标准级", "description": "+ 局部自适应骨抑制" },
    "deep": { "name": "深度级", "description": "+ Top-Hat 血管增强" },
}
def _get_preset_file_path() -> Path:
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
        user_preset = exe_dir / "device_presets.json"
        if user_preset.exists():
            return user_preset
        return user_preset
    else:
        return Path(os.path.dirname(os.path.abspath(__file__))) / "device_presets.json"

PRESET_FILE = _get_preset_file_path()
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
        "wc": 200, "ww": 400
    },
    "Siemens SOMATOM": {
        "bone_strength": 1.2, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400
    },
    "GE Revolution/Discovery": {
        "bone_strength": 1.4, "vessel_sensitivity": 0.9, "vessel_enhance": 1.8,
        "smooth_sigma": 0.9, "clean_bone_edges": True, "min_vessel_size": 6,
        "wc": 220, "ww": 420
    },
    "Philips Brilliance/IQon": {
        "bone_strength": 1.0, "vessel_sensitivity": 1.1, "vessel_enhance": 2.2,
        "smooth_sigma": 0.5, "clean_bone_edges": True, "min_vessel_size": 4,
        "wc": 180, "ww": 380
    },
    "Canon Aquilion": {
        "bone_strength": 1.1, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": False, "min_vessel_size": 5,
        "wc": 200, "ww": 400
    },
    "联影 uCT": {
        "bone_strength": 1.3, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.8, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400
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
            self.finished_signal.emit({},[],[])

# ════════════════════════════════════════════════════════════════════
# 预标注线程
# ════════════════════════════════════════════════════════════════════
class PreAnnotationThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    ready = pyqtSignal(np.ndarray, list)
    error = pyqtSignal(str)

    def __init__(self, post_series):
        super().__init__()
        self.post_series = post_series

    def run(self):
        post_vol = None
        try:
            self.log.emit("=" * 55)
            self.log.emit("[预标注] 读取造影序列（不配准）...")
            reader = sitk.ImageSeriesReader()
            s_dir = os.path.dirname(self.post_series.files[0][1])
            s_dir_safe = _get_short_path(s_dir)
            try:
                f_list = reader.GetGDCMSeriesFileNames(s_dir_safe, self.post_series.series_uid)
            except Exception:
                f_list =[]
            if not f_list or len(f_list) != self.post_series.file_count:
                f_list = [f[1] for f in self.post_series.files]
            if not f_list:
                self.error.emit("造影序列文件列表为空")
                return

            total = len(f_list)
            self.log.emit(f"[预标注] 共 {total} 层，逐层读取中...")
            ds0 = pydicom.dcmread(f_list[0])
            h, w = ds0.pixel_array.shape
            post_vol = np.empty((total, h, w), dtype=np.int16)
            for z, fp in enumerate(f_list):
                ds = ds0 if z == 0 else pydicom.dcmread(fp)
                arr = ds.pixel_array.astype(np.int16)
                slope = float(getattr(ds, 'RescaleSlope', 1))
                intercept = float(getattr(ds, 'RescaleIntercept', 0))
                if slope != 1.0 or intercept != 0.0:
                    arr = (arr.astype(np.float32) * slope + intercept).astype(np.int16)
                post_vol[z] = arr
                if z == 0: ds0 = None  
                if z % 50 == 0: self.progress.emit(int(z / total * 100))
            self.progress.emit(100)
            self.log.emit(f"[预标注] 读取完成，内存：{get_memory_mb():.0f} MB")
            self.ready.emit(post_vol, list(f_list))
        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.error.emit(str(e))

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
    return ndimage.shift(image.astype(np.float32),[dy, dx], order=1, mode='constant', cval=0)

def trim_to_common_z_range(files_a, files_b, log_cb=None):
    if len(files_a) == len(files_b): return files_a, files_b
    n_a, n_b = len(files_a), len(files_b)
    n = min(n_a, n_b)          
    if n_a > n_b:
        if log_cb: log_cb(f"  [层数对齐] 平扫 ({n_a}层) > 增强 ({n_b}层)，取末尾 {n} 层")
        return files_a[-n:], files_b
    else:
        if log_cb: log_cb(f"[层数对齐] 增强 ({n_b}层) > 平扫 ({n_a}层)，取末尾 {n} 层")
        return files_a, files_b[-n:]

def build_z_matched_file_pairs(pre_files_tuples, post_files_tuples, log_cb=None):
    pre_dict_inst = {i: p for i, p in pre_files_tuples}
    post_dict_inst = {i: p for i, p in post_files_tuples}
    common_inst = sorted(set(pre_dict_inst) & set(post_dict_inst))
    if common_inst: return common_inst, pre_dict_inst, post_dict_inst

    if log_cb: log_cb("  [参数分析] instance number 无重叠，改用 Z 坐标匹配采样层")
    def get_z(path):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp: return float(ipp[2])
            sl = getattr(ds, 'SliceLocation', None)
            return float(sl) if sl is not None else None
        except Exception:
            return None

    pre_z_list =[(get_z(p), i, p) for i, p in pre_files_tuples if get_z(p) is not None]
    post_z_list =[(get_z(p), i, p) for i, p in post_files_tuples if get_z(p) is not None]

    if not pre_z_list or not post_z_list: return[], {}, {}
    pre_z_list.sort(key=lambda x: x[0])
    post_z_list.sort(key=lambda x: x[0])
    z_min = max(pre_z_list[0][0], post_z_list[0][0])
    z_max = min(pre_z_list[-1][0], post_z_list[-1][0])

    if z_min > z_max: return[], {}, {}

    sample_z =[z_min + (z_max - z_min) * t for t in[0.25, 0.40, 0.50, 0.60, 0.75]]
    pre_arr =[z for z, _, _ in pre_z_list]
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
            common, pre_dict, post_dict = build_z_matched_file_pairs(self.pre_files, self.post_files, self.log.emit)
            if not common: return self.finished_signal.emit({'error': '序列不匹配（无重叠 Z 范围）'})

            n_slices = len(common)
            indices =[min(int(n_slices * p), n_slices - 1) for p in[0.25, 0.40, 0.50, 0.60, 0.75]]
            samples = [common[i] for i in indices]
            all_chars = {'shift':[], 'noise':[], 'vessel':[], 'bone_mismatch':[]}
            
            for i, inst in enumerate(samples):
                pre_d = pydicom.dcmread(pre_dict[inst]).pixel_array.astype(np.float32)
                post_d = pydicom.dcmread(post_dict[inst]).pixel_array.astype(np.float32)
                 
                dy, dx = fft_phase_correlation_2d(pre_d, post_d, max_shift=10)
                all_chars['shift'].append(np.sqrt(dy**2 + dx**2))
                
                post_aligned = shift_image_2d(post_d, dy, dx)
                diff = np.clip(post_aligned - pre_d, 0, None)
                
                brain_mask = ndimage.binary_erosion((pre_d > 20) & (pre_d < 80), iterations=2)
                all_chars['noise'].append(float(pre_d[brain_mask].std()) if brain_mask.sum() > 500 else 8.0)
                
                skull_mask = ndimage.binary_fill_holes(pre_d > 100)
                intracranial = skull_mask & ~ndimage.binary_dilation(pre_d > 100, iterations=4)
                vessel_enhancement = float(np.percentile(diff[intracranial], 99.5)) if intracranial.sum() > 1000 else float(np.percentile(diff, 99.9))
                all_chars['vessel'].append(vessel_enhancement)
                
                bone_edge = ndimage.binary_dilation(pre_d > 150, iterations=1) ^ (pre_d > 150)
                all_chars['bone_mismatch'].append(float(diff[bone_edge].mean()) if bone_edge.sum() > 100 else 0.0)
                self.progress.emit(int((i + 1) / len(samples) * 100))
            
            avg_shift, avg_noise = float(np.mean(all_chars['shift'])), float(np.mean(all_chars['noise']))
            avg_vessel, avg_bone = float(np.mean(all_chars['vessel'])), float(np.mean(all_chars['bone_mismatch']))
            
            score = 100.0 - max(0, (avg_shift - 0.3) * 25) - max(0, (avg_noise - 6.0) * 3.0) - max(0, (150.0 - avg_vessel) * 0.4)
            if avg_bone > 20.0: score -= ((avg_bone - 20.0) ** 1.2) * 1.5
            score = max(0.0, min(100.0, score)) / 100.0
            
            rec = {
                'bone_strength': 1.4 if avg_bone > 35 else 1.2,
                'vessel_sensitivity': 0.8 if avg_noise > 12 else 1.0,
                'vessel_enhance': 2.5 if avg_vessel < 100 else (2.2 if avg_vessel < 150 else 1.8),
                'clean_bone_edges': avg_bone > 20,
                'min_vessel_size': 7 if avg_noise > 10 else 5,
                'smooth_sigma': 1.0 if avg_noise > 10 else 0.7,
                'wc': 200, 'ww': 400, 'quality_score': score,
                'details': (f"位移偏差：{avg_shift:.2f} px\n实质噪声：{avg_noise:.1f} HU\n"
                            f"强化峰值：{avg_vessel:.0f} HU\n骨边误差：{avg_bone:.0f} HU")
            }
            
            if score >= 0.85 and avg_shift < 0.6: rec['recommended_mode'], rec['recommended_opt'], quality_str = 'fast', 'light', "优良 (自动配置：快速 + 轻量级)"
            elif score >= 0.70: rec['recommended_mode'], rec['recommended_opt'], quality_str = 'quality', 'standard', "一般 (自动配置：精细 + 标准级)"
            else: rec['recommended_mode'], rec['recommended_opt'], quality_str = 'quality', 'deep', "较差 (自动配置：精细 + 深度级)"
                
            self.log.emit(f"综合质量：{score * 100:.0f} 分\n{rec['details']}\n系统建议：{quality_str}")
            self.finished_signal.emit(rec)
        except Exception as e:
            self.log.emit(f"[分析错误] {traceback.format_exc()}")
            self.finished_signal.emit({'error': str(e)})

def compute_3d_registration_transform(fixed_shrink, moving_shrink, log_cb=None, progress_cb=None):
    if log_cb: log_cb("[配准升级] 启动：物理降采样 + SITK 纯 C++ 剥离扫描床 + 四元数旋转")
    
    body_raw = sitk.BinaryThreshold(fixed_shrink, lowerThreshold=-400.0, upperThreshold=3000.0, insideValue=1, outsideValue=0)
    cc = sitk.ConnectedComponent(body_raw)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    body_only = sitk.BinaryThreshold(relabeled, lowerThreshold=1.0, upperThreshold=1.0, insideValue=1, outsideValue=0)
    
    bone_raw = sitk.BinaryThreshold(fixed_shrink, lowerThreshold=150.0, upperThreshold=3000.0, insideValue=1, outsideValue=0)
    patient_bone = sitk.And(bone_raw, body_only)
    bone_mask = sitk.BinaryDilate(patient_bone,[1, 1, 1], sitk.sitkBall)
    
    del body_raw, cc, relabeled, body_only, bone_raw, patient_bone
    force_gc()
    
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricFixedMask(bone_mask)
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_shrink, moving_shrink, sitk.VersorRigid3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initial_transform, inPlace=False)
    
    R.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.001, numberOfIterations=50, estimateLearningRate=R.Once)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInterpolator(sitk.sitkLinear)
    
    if progress_cb: R.AddCommand(sitk.sitkIterationEvent, lambda: progress_cb(int((R.GetOptimizerIteration() / 50.0) * 20)))
    final_transform = R.Execute(fixed_shrink, moving_shrink)
    
    if log_cb: log_cb(f"[配准参数] 四元数及平移向量：{[round(x, 4) for x in final_transform.GetParameters()]}")
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
    if body is None: body = create_body_mask(pre_f)
    high_both = (pre_f > 150) & (post_f > 150)
    stable = np.abs(post_f - pre_f) < 30
    body_dilated = ndimage.binary_dilation(body, iterations=8)
    return ndimage.binary_dilation(high_both & stable & ~body_dilated, iterations=5)

class SliceCache:
    def __init__(self):
        self.body, self.scalp, self.air_bone, self.petrous = None, None, None, None
        self.thin_bone, self.petrous_edge = None, None
        self.bone_edge, self.ab_edge, self.ica_protect = None, None, None
        # 骨边缘血管保护带：骨mask向内erosion 3px 之前与之后的差值区域
        # 该带子内贴骨血管不做强抑制，避免ICA/眼动脉等被误删
        self.bone_vessel_edge = None
        # z_ratio：保留供内部兜底使用，不再用于解剖分区
        self.z_ratio = 0.5
        # slices_from_top：距顶点（vertex）的绝对层数（0.625mm/层）
        self.slices_from_top = 0
        # eff_conv_end：颅盖/Willis有效边界（由调用方注入，默认程序常量）
        # 供 estimate_k_by_zone / detect_problematic_bone_zones_v2 使用，
        # 保证底层骨遮罩与主分区完全同步。
        self.eff_conv_end = ZONE_CONVEXITY_END

def precompute_slice_cache(pre_s, diff_pos, quality_mode):
    pre_f = pre_s.astype(np.float32) if pre_s.dtype != np.float32 else pre_s
    cache = SliceCache()
    cache.body = create_body_mask(pre_f)
    bone = pre_f > 150
    grad_y, grad_x = ndimage.sobel(pre_f, axis=0), ndimage.sobel(pre_f, axis=1)
    gradient = np.sqrt(grad_y ** 2 + grad_x ** 2)
    cache.thin_bone = (bone & ~ndimage.binary_erosion(bone, iterations=2)) | (bone & (gradient > np.percentile(gradient[bone] if bone.sum() > 0 else gradient.ravel(), 70)))
    cache.bone_edge = ndimage.binary_dilation(bone, iterations=2) & ~ndimage.binary_erosion(bone, iterations=2)

    # ── 骨边缘血管保护带 ──────────────────────────────────────────────
    # bone 向内 erosion 3px，erosion前后之差 = 骨边缘3px宽的环状区域。
    # 颅底/Willis环贴骨血管（ICA岩骨段、眼动脉、ACoA）落在此区域内。
    # 后续在该带内不做零增益强抑制，改用软化增益保留血管信号。
    bone_eroded3 = ndimage.binary_erosion(bone, iterations=3)
    cache.bone_vessel_edge = bone & ~bone_eroded3   # 3px宽骨内侧边缘带

    if not quality_mode: return cache

    cache.scalp = ndimage.binary_dilation((cache.body & ~ndimage.binary_erosion(cache.body, iterations=8)) & ((pre_f > -50) & (pre_f < 100)), iterations=2)
    air_d4, bone_d4 = ndimage.binary_dilation(pre_f < -200, iterations=4), ndimage.binary_dilation(bone, iterations=4)
    cache.air_bone = air_d4 & bone_d4 & ndimage.binary_dilation(bone, iterations=6)
    cache.ab_edge = air_d4 & bone_d4
    petrous = ndimage.binary_dilation((pre_f > 400) & ndimage.binary_dilation(gradient > 100, iterations=2), iterations=3)
    cache.petrous, cache.petrous_edge = petrous, petrous & ~ndimage.binary_erosion(petrous, iterations=2)
    ica_core = petrous & (pre_f < 150) & (diff_pos > 50)
    cache.ica_protect = ndimage.binary_dilation(ica_core, iterations=2) if ica_core.sum() > 0 else np.zeros_like(pre_s, dtype=bool)
    return cache

# ════════════════════════════════════════════════════════════════════
# 解剖学分区与问题骨区检测
# ════════════════════════════════════════════════════════════════════
#
# ── 三区绝对边界（0.625 mm/层，GE 256排，OM线基准）────────────────
#   坐标原点：颅顶 vertex（slices_from_top = 0），向颅底/颈部递增。
#   与扫描总层数、医师裁切范围、分段处理方式完全无关。
#
#   颅盖骨区  convexity   : slices_from_top   0 ~ 101   (0  ~ 63.1mm)
#   精细分区  willis_circle: slices_from_top 102 ~ 185  (63.8 ~ 115.6mm)
#     上界解剖标志：眶顶板消失层（实测顶下~110层），含-8层过渡余量 → 102
#     下界解剖标志：鞍底/岩尖上界（实测顶下~178层），含+8层过渡余量 → 186
#   颅底区    skull_base  : slices_from_top ≥ 186       (≥ 116.3mm)
#     （头颈CTA时颈椎段也落于此区，接受强骨抑制，符合预期）
#
#   边界过渡带：各边界两侧 ±8层线性渐变，消除相邻层亮度阶梯。
# ─────────────────────────────────────────────────────────────────
ZONE_CONVEXITY_END   = 102   # 颅盖区结束（含），眶顶板消失层-8过渡余量
ZONE_WILLIS_END      = 186   # 精细分区结束（含），鞍底/岩尖上界+8过渡余量
ZONE_TRANSITION      = 8     # 各边界两侧过渡带宽度（层）

def get_slices_from_top(original_global_z: int, original_total_slices: int) -> int:
    """
    将原始序列坐标（0=最底层/IPP.z最小）转换为距顶点层数（0=颅顶vertex）。
    original_total_slices 必须是裁切前的完整序列层数，调用方保证不变。
    """
    return max(0, (original_total_slices - 1) - original_global_z)

def get_anatomical_zone_v2(slices_from_top: int,
                            conv_end: int = ZONE_CONVEXITY_END,
                            willis_end: int = ZONE_WILLIS_END) -> str:
    """
    基于距顶点绝对层数的三区判定。
    conv_end / willis_end 可由医生标定覆盖，默认使用程序常量。
    """
    if slices_from_top < conv_end:
        return 'convexity'
    elif slices_from_top < willis_end:
        return 'willis_circle'
    else:
        return 'skull_base'

def get_zone_adjusted_params_v2(slices_from_top: int,
                                 bone_strength: float,
                                 vessel_sensitivity: float,
                                 conv_end: int = ZONE_CONVEXITY_END,
                                 willis_end: int = ZONE_WILLIS_END):
    """
    分区参数调整，各边界两侧 ±ZONE_TRANSITION 层线性渐变。
    conv_end / willis_end 可由医生标定覆盖。
    """
    def lerp(t, a, b):
        return a + (b - a) * max(0.0, min(1.0, t))

    C  = (1.00, 1.00)
    W  = (0.75, 1.60)
    S  = (1.30, 1.30)
    TW = ZONE_TRANSITION
    B1 = conv_end
    B2 = willis_end

    if slices_from_top < B1 - TW:
        bm, vm = C
    elif slices_from_top < B1 + TW:
        t = (slices_from_top - (B1 - TW)) / (2.0 * TW)
        bm = lerp(t, C[0], W[0])
        vm = lerp(t, C[1], W[1])
    elif slices_from_top < B2 - TW:
        bm, vm = W
    elif slices_from_top < B2 + TW:
        t = (slices_from_top - (B2 - TW)) / (2.0 * TW)
        bm = lerp(t, W[0], S[0])
        vm = lerp(t, W[1], S[1])
    else:
        bm, vm = S

    return bone_strength * bm, vessel_sensitivity * vm

def detect_problematic_bone_zones_v2(pre_slice, slices_from_top: int,
                                      eff_conv_end: int = ZONE_CONVEXITY_END):
    """
    各骨区激活边界由 eff_conv_end（颅盖/Willis边界）派生，
    与医生手动标定完全同步，不写死绝对层数。

    各骨结构相对于颅盖/Willis边界的偏移量（0.625mm/层）：
      颅顶薄骨  : slices_from_top < eff_conv_end + 8    （边界上方，5mm缓冲）
      颅底/前床突: slices_from_top > eff_conv_end        （进入Willis区即激活）
      眶板      : eff_conv_end+8  < sft < eff_conv_end+144（约深5~90mm处）
      岩骨      : slices_from_top > eff_conv_end + 40    （比边界深25mm）
      下颌骨    : slices_from_top > eff_conv_end + 90    （比边界深56mm）
    """
    pre_f = pre_slice.astype(np.float32) if pre_slice.dtype != np.float32 else pre_slice
    gradient = np.sqrt(ndimage.sobel(pre_f, axis=0)**2 + ndimage.sobel(pre_f, axis=1)**2)

    # ── 各骨区激活阈值（全部由 eff_conv_end 派生）────────────────────
    th_skull   = eff_conv_end           # 颅底/前床突激活下限
    th_orb_lo  = eff_conv_end + 8       # 眶板激活上限（浅界）
    th_orb_hi  = eff_conv_end + 144     # 眶板激活下限（深界）
    th_petrous = eff_conv_end + 40      # 岩骨激活下限
    th_mandible= eff_conv_end + 90      # 下颌骨激活下限
    th_conv    = eff_conv_end + 8       # 颅顶薄骨激活上限

    # ── 岩骨：高密度 + 紧邻气腔（中耳、乳突气房）──
    petrous = np.zeros_like(pre_slice, dtype=bool)
    if slices_from_top > th_petrous:
        high_density = pre_f > 400
        air_adjacent = ndimage.binary_dilation(pre_f < -200, iterations=5)
        petrous = high_density & air_adjacent

    # ── 颅底骨/前床突 ──
    skull_base = ((pre_f > 250) & (pre_f < 1000)
                  if slices_from_top > th_skull
                  else np.zeros_like(pre_slice, dtype=bool))

    # ── 下颌骨 ──
    mandible = (pre_f > 200
                if slices_from_top > th_mandible
                else np.zeros_like(pre_slice, dtype=bool))

    # ── 锐利骨边缘（高梯度骨质，全层有效）──
    sharp_edge = (pre_f > 150) & (gradient > 120)

    # ── 眶板：薄骨 + 紧邻眼眶脂肪/气腔 ──────────────────────────────
    orbital_plate = np.zeros_like(pre_slice, dtype=bool)
    if th_orb_lo < slices_from_top < th_orb_hi:
        orbit_air    = ndimage.binary_dilation(pre_f < -50, iterations=4)
        orbital_bone = (pre_f > 150) & (pre_f < 700)
        orbital_plate = orbital_bone & orbit_air
        if orbital_plate.sum() > 20:
            orbital_plate = ndimage.binary_opening(orbital_plate, iterations=1)

    # ── 颅顶/穹隆薄骨（erosion 3px 覆盖外板）──────────────────────
    convexity_thin = np.zeros_like(pre_slice, dtype=bool)
    if slices_from_top < th_conv:
        bone = pre_f > 150
        bone_eroded = ndimage.binary_erosion(bone, iterations=3)
        convexity_thin = bone & ~bone_eroded

    return {
        'petrous': petrous,
        'skull_base': skull_base,
        'mandible': mandible,
        'sharp_edge': sharp_edge,
        'orbital_plate': orbital_plate,
        'convexity_thin': convexity_thin,
    }

# ── 旧接口保留（向后兼容，内部转发到 v2）────────────────────────────
def get_anatomical_zone(z_index, total_slices):
    sft = get_slices_from_top(z_index, total_slices)
    return get_anatomical_zone_v2(sft)

def get_zone_adjusted_params(zone, bone_strength, vessel_sensitivity):
    # 旧的 zone 字符串接口，映射到区中心层数再调用 v2
    _center = {'convexity': 60, 'willis_circle': 144, 'skull_base': 200}
    sft = _center.get(zone, 60)
    return get_zone_adjusted_params_v2(sft, bone_strength, vessel_sensitivity)

# ════════════════════════════════════════════════════════════════════
# 减影算法核心分级
# ════════════════════════════════════════════════════════════════════
def fast_subtraction(pre, post_aligned, bone_strength=1.0, vessel_sensitivity=1.0, body=None, cache=None):
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

    # ── 骨边缘血管保护 ───────────────────────────────────────────────
    # 在3px骨边缘带内，若diff_pos超过血管阈值，说明是贴骨血管，
    # 覆盖前面的骨抑制增益，改用0.6的软化增益保留血管信号。
    if cache is not None and cache.bone_vessel_edge is not None:
        bve = cache.bone_vessel_edge & ~shifted_bone_veto
        gain[bve & (diff_pos > 60 * vessel_sensitivity)] = 0.6

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

    # ── 骨边缘血管保护 ───────────────────────────────────────────────
    # 3px骨边缘带内，diff_pos超血管阈值则为贴骨血管（ICA/眼动脉/ACoA），
    # 覆盖前面的骨抑制，用0.6软化增益透出血管信号。
    # ICA保护区已在上方单独处理，此处排除避免双重覆盖。
    if cache.bone_vessel_edge is not None:
        ica = cache.ica_protect if cache.ica_protect is not None else np.zeros_like(pre, dtype=bool)
        bve = cache.bone_vessel_edge & ~ica & ~shifted_bone_veto & ~scalp_mask
        gain[bve & (diff_pos > 60 * vessel_sensitivity)] = 0.6

    return diff_pos * np.clip(gain, 0, 1.5)

def detect_ica_protection_zone_precise(pre_s, diff_s, body_mask):
    pre_f = pre_s.astype(np.float32) if pre_s.dtype != np.float32 else pre_s
    dense_bone = pre_f > 400
    if dense_bone.sum() < 50: return np.zeros_like(pre_s, dtype=bool)
    ica_candidate = ndimage.binary_dilation(dense_bone, iterations=5) & (diff_s > 80) & body_mask
    ica_candidate = ica_candidate & ndimage.binary_erosion(body_mask, iterations=10)
    return ndimage.binary_dilation(ica_candidate, iterations=2) if ica_candidate.sum() > 0 else np.zeros_like(pre_s, dtype=bool)

def estimate_k_by_zone(pre_f, post_f, bone_mask, slices_from_top,
                        eff_conv_end: int = ZONE_CONVEXITY_END):
    """
    分区域估计beam hardening系数k：post_bone ≈ k * pre_bone
    分区边界与主分区同步，使用 eff_conv_end 而非写死的常量。
      颅底/Willis区（slices_from_top > eff_conv_end）：致密骨与中密度骨分别回归
      颅盖区（slices_from_top ≤ eff_conv_end）：凸面薄骨统一回归
    避免颅底高HU骨污染凸面薄骨的k估计。
    返回每个像素对应的k值图（浮点，骨mask外填1.0）。
    """
    # 只用HU 200~1200的确定骨质做回归
    # 上限1200：排除金属体素（瘤夹/弹簧圈/颅骨板钉等，HU>1200）
    # 金属HU极高且post/pre比值不稳定，纳入回归会严重污染k估计
    solid_bone = bone_mask & (pre_f > 200) & (pre_f < 1200)
    k_map = np.ones_like(pre_f, dtype=np.float32)

    zones = []
    if slices_from_top > eff_conv_end:
        zones.append(('skull_base',     solid_bone & (pre_f > 400)))
        zones.append(('skull_base_med', solid_bone & (pre_f <= 400)))
    else:
        zones.append(('convexity', solid_bone))

    k_global = 1.0
    global_pre = pre_f[solid_bone]
    global_post = post_f[solid_bone]
    if len(global_pre) >= 20:
        denom = float(np.dot(global_pre, global_pre))
        if denom > 0:
            k_global = float(np.dot(global_pre, global_post)) / denom
            # k合理范围：0.90~1.15，超出说明配准严重失败或非骨组织污染
            k_global = float(np.clip(k_global, 0.90, 1.15))

    for zone_name, zmask in zones:
        pre_z = pre_f[zmask]
        post_z = post_f[zmask]
        if len(pre_z) < 20:
            # 像素太少，用全局k
            k_map[zmask] = k_global
            continue
        denom = float(np.dot(pre_z, pre_z))
        if denom <= 0:
            k_map[zmask] = k_global
            continue
        k_zone = float(np.dot(pre_z, post_z)) / denom
        k_zone = float(np.clip(k_zone, 0.90, 1.15))
        k_map[zmask] = k_zone

    # 骨mask外不参与k修正，保持1.0
    k_map[~bone_mask] = 1.0
    return k_map, k_global


def compute_adaptive_threshold(pre_s, diff_s, bone_mask, bone_strength,
                                window_size=31, k_map=None):
    """
    k增强版自适应阈值：
    threshold = expected_drift + sigma_term
    expected_drift = max((k-1)*pre_f, 0)  ← 物理预期的beam hardening漂移
    sigma_term     = local_std / bone_strength  ← 统计波动项
    
    相比原版（纯统计local_mean+1.5σ）：
    - expected_drift 把高HU骨的大漂移从统计均值中分离出来
    - 邻域统计不再被颅底骨污染，凸面阈值更干净
    - 贴骨血管像素(pre_f<200)的expected_drift≈0，阈值仅由local_std决定，
      不被骨的漂移拉高，血管信号得以保留
    """
    pre_f = pre_s.astype(np.float32) if pre_s.dtype != np.float32 else pre_s
    b_diff = np.where(bone_mask, diff_s, 0).astype(np.float32)
    b_float = bone_mask.astype(np.float32)
    local_count_safe = np.maximum(ndimage.uniform_filter(b_float, size=window_size), 1e-6)
    local_std = np.sqrt(np.maximum(
        ndimage.uniform_filter(b_diff ** 2, size=window_size) / local_count_safe
        - (ndimage.uniform_filter(b_diff, size=window_size) / local_count_safe) ** 2,
        0))

    if k_map is not None:
        # k估计路径：物理预期漂移 + 统计波动
        expected_drift = np.maximum((k_map - 1.0) * pre_f, 0)
        sigma_term = local_std / bone_strength
        # 保底值：至少30HU，防止k≈1.0区域阈值过低误删贴骨血管
        adaptive_threshold = np.maximum(expected_drift + sigma_term, 30 * bone_strength)
    else:
        # 原始路径（兜底，保持向后兼容）
        local_mean = ndimage.uniform_filter(b_diff, size=window_size) / local_count_safe
        adaptive_threshold = np.maximum(local_mean + (1.5 / bone_strength) * local_std,
                                         50 * bone_strength)

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

    # ── k估计：分区域估计beam hardening系数，用于自适应阈值物理校正 ──
    sft = getattr(cache, 'slices_from_top', 0)
    ece = getattr(cache, 'eff_conv_end', ZONE_CONVEXITY_END)
    k_map, k_global = estimate_k_by_zone(pre_f, post_f, bone_mask, sft, ece)

    ica_protect = detect_ica_protection_zone_precise(pre_f, diff_pos, cache.body)
    adaptive_thresh = compute_adaptive_threshold(pre_f, diff_pos, bone_mask, bone_strength,
                                                  k_map=k_map)
    gain[bone_mask & (diff_pos < adaptive_thresh) & ~ica_protect] = 0
    gain[bone_mask & (diff_pos >= adaptive_thresh) & (diff_pos < adaptive_thresh * 1.5) & ~ica_protect] = 0.2 / bone_strength
    gain[bone_mask & (diff_pos >= adaptive_thresh * 1.5) & ~ica_protect] = 0.4 / bone_strength
    shifted_bone_veto = (pre_f < 150) & (post_f > 350)

    vessel_mask = (pre_f > -100) & (pre_f < 60) & (diff_pos > 50 * vessel_sensitivity) & cache.body & ~shifted_bone_veto
    gain[vessel_mask] = 1.0
    gain[ica_protect & (diff_pos > 60)] = 0.85
    gain[~cache.body] = 0

    # ── 骨边缘血管保护 ───────────────────────────────────────────────
    # 3px骨边缘带内，adaptive_thresh已经把骨抑制到0，
    # 但贴骨血管的diff_pos会超过血管阈值。
    # 此处在ICA保护区之外再加一层保护，gain改为0.6透出血管。
    if cache.bone_vessel_edge is not None:
        bve = cache.bone_vessel_edge & ~ica_protect & ~shifted_bone_veto
        gain[bve & (diff_pos > 60 * vessel_sensitivity)] = 0.6
        gain[~cache.body] = 0   # 边界再次清零，防止体外溢出

    return diff_pos * np.clip(gain, 0, 1.5)

def optimized_subtraction_standard_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy, z_ratio):
    return optimized_subtraction_light_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache)

def morphological_top_hat_2d(image, disk_radius=3):
    y, x = np.ogrid[-disk_radius:disk_radius+1, -disk_radius:disk_radius+1]
    disk = (x*x + y*y <= disk_radius*disk_radius).astype(np.uint8)
    opened = ndimage.grey_opening(image.astype(np.float32), footprint=disk)
    return np.maximum(image - opened, 0).astype(np.float32)

def multi_scale_top_hat_2d(image, radii=[2, 3, 5]):
    result = np.zeros_like(image, dtype=np.float32)
    for r in radii: result = np.maximum(result, morphological_top_hat_2d(image, disk_radius=r))
    return result

def compute_top_hat_for_slice(pre_s, post_s, spacing_zyx):
    pre_f = pre_s.astype(np.float32) if pre_s.dtype != np.float32 else pre_s
    post_f = post_s.astype(np.float32) if post_s.dtype != np.float32 else post_s
    slice_diff = np.clip(post_f - pre_f, 0, None)
    if slice_diff.max() <= 15: return None

    min_sp = min(spacing_zyx[1], spacing_zyx[2])
    radii = list(set([max(1, int(round(r / min_sp))) for r in[0.8, 1.5, 2.5]]))
    radii =[r for r in radii if 1 <= r <= 6]
    if not radii: radii =[2, 3]

    top_hat = multi_scale_top_hat_2d(slice_diff, radii=radii)
    vmax = top_hat.max()
    if vmax > 0: top_hat /= vmax
    return top_hat

def optimized_subtraction_deep_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy, spacing_zyx, z_ratio):
    pre_f = pre.astype(np.float32) if pre.dtype != np.float32 else pre
    post_f = post_aligned.astype(np.float32) if post_aligned.dtype != np.float32 else post_aligned
    result = optimized_subtraction_standard_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy, z_ratio)

    top_hat_map = compute_top_hat_for_slice(pre_f, post_f, spacing_zyx)
    if top_hat_map is not None and top_hat_map.max() > 0:
        high_response = top_hat_map > 0.2
        boost = 1.0 + 0.25 * top_hat_map
        result = result * np.where(high_response & cache.body, boost, 1.0)
    return result

def enhanced_bone_suppression_v299(pre_s, diff_pos, gain, bone_zones, bone_strength, vessel_sensitivity=1.0):
    # 血管豁免条件：diff_pos大 且 pre_f在软组织侧（<200HU）
    # pre_f < 200 这个条件是关键：骨内像素（>200HU）的beam hardening差异
    # 也可能达到80~100HU，单凭diff_pos阈值无法区分。
    # 加上pre_f < 200，确保豁免只发生在骨旁软组织/血管腔内，
    # 前床突/颅底骨质像素本身（pre_f>300）不会被误救。
    vessel_exempt_thresh = 110 * vessel_sensitivity   # 提高到110HU，远离beam hardening区间
    soft_tissue_side = pre_s < 200                    # 软组织/血管腔侧，骨质本身不在此

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
        # 血管豁免修复：
        # 原逻辑 skull_base(pre_f>250) & soft_tissue_side(pre_s<200) 恒为空集，豁免从未触发。
        # 修正：将 skull_base 向外膨胀 3px，形成骨旁软组织侧缓冲带，
        # 再限定 soft_tissue_side，使 ICA/眼动脉等贴骨血管腔能落入该区间。
        # 颅底骨质本身（HU>250）被 soft_tissue_side(<200) 排除，不会被误救。
        skull_base_halo = ndimage.binary_dilation(skull_base, iterations=3)
        skull_vessel = skull_base_halo & soft_tissue_side & (diff_pos >= vessel_exempt_thresh)
        gain[skull_vessel] = np.maximum(gain[skull_vessel], 0.75)

    mandible = bone_zones['mandible']
    if np.any(mandible):
        gain[mandible & (diff_pos < 200 * bone_strength)] = 0
        gain[mandible & (diff_pos >= 200 * bone_strength)] = 0.05

    sharp_edge = bone_zones['sharp_edge']
    if np.any(sharp_edge):
        gain[sharp_edge & (diff_pos < 100 * bone_strength)] = 0
        gain[sharp_edge & (diff_pos >= 100 * bone_strength) & (diff_pos < 150 * bone_strength)] = 0.1

    # ── 眶板 ──────────────────────────────────────────────────────────
    orbital_plate = bone_zones.get('orbital_plate')
    if orbital_plate is not None and np.any(orbital_plate):
        orb_thresh = 40 * bone_strength
        gain[orbital_plate & (diff_pos < orb_thresh)] = 0
        gain[orbital_plate & (diff_pos >= orb_thresh)] = 0.04
        # 血管保护优先于骨抑制：眼动脉/眶内血管 diff_pos 通常 >90HU，
        # 豁免阈值从110降到90，增益从0.6提到0.8，确保血管不被误削。
        # 眶板残留可手工三维减除，血管断裂则丢失诊断信息，代价不对等。
        orb_vessel = orbital_plate & soft_tissue_side & (diff_pos >= 90 * vessel_sensitivity)
        gain[orb_vessel] = np.maximum(gain[orb_vessel], 0.8)

    # ── 颅顶穹隆薄骨 ──────────────────────────────────────────────────
    convexity_thin = bone_zones.get('convexity_thin')
    if convexity_thin is not None and np.any(convexity_thin):
        conv_thresh = 50 * bone_strength
        gain[convexity_thin & (diff_pos < conv_thresh)] = 0
        gain[convexity_thin & (diff_pos >= conv_thresh) & (diff_pos < conv_thresh * 2)] = 0.05
        gain[convexity_thin & (diff_pos >= conv_thresh * 2)] = 0.12
        # 皮层静脉/上矢状窦在软组织侧走行，同样限定 pre_f<200
        conv_vessel = convexity_thin & soft_tissue_side & (diff_pos >= vessel_exempt_thresh)
        gain[conv_vessel] = np.maximum(gain[conv_vessel], 0.75)

    return gain

def get_safe_percentile_nonzero(volume, percentile, max_bin=4096):
    hist = np.zeros(max_bin, dtype=np.int64)
    for z in range(volume.shape[0]):
        v = volume[z]
        mask = v > 0
        if np.any(mask): hist += np.bincount(np.clip(v[mask], 0, max_bin - 1).astype(np.int32), minlength=max_bin)
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
    if log_cb: log_cb("[形状分析] 2.5D 滑动连通分析...")
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
# 弹簧圈星芒伪影三维扩展
# ════════════════════════════════════════════════════════════════════
def expand_coil_artifact_3d(pre_vol: np.ndarray, seed_mask_3d: np.ndarray,
                             spacing_zyx: tuple, log_cb=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从魔棒取核的弹簧圈seed_mask出发，三维识别星芒伪影范围。
    【32位低内存版】所有形态学操作严格限制在seed包围盒+搜索半径的局部crop内，
    避免在全量volume（300×512×512）上做重型3D膨胀，防止32位Win7内存/CPU爆满。
    返回：(伪影区, 光环搜索区, 核心区)，均为与pre_vol同shape的全量bool数组。
    """
    if log_cb: log_cb("  [弹簧圈] 三维星芒伪影扩展识别与血管救援准备中（低内存局部模式）...")

    vol_shape = pre_vol.shape

    # ── 计算搜索半径（体素数） ──────────────────────────────────────
    r_xy = max(3, int(np.ceil(10.0 / min(spacing_zyx[1], spacing_zyx[2]))))
    r_z  = max(2, int(np.ceil(8.0  / spacing_zyx[0])))

    # ── 找seed包围盒，加搜索半径padding，裁剪到局部crop ────────────
    seed_idx = np.where(seed_mask_3d)
    if len(seed_idx[0]) == 0:
        return (np.zeros(vol_shape, dtype=bool),
                np.zeros(vol_shape, dtype=bool),
                seed_mask_3d)

    sz0 = max(0, int(seed_idx[0].min()) - r_z  - 2)
    sz1 = min(vol_shape[0] - 1, int(seed_idx[0].max()) + r_z  + 2)
    sy0 = max(0, int(seed_idx[1].min()) - r_xy - 2)
    sy1 = min(vol_shape[1] - 1, int(seed_idx[1].max()) + r_xy + 2)
    sx0 = max(0, int(seed_idx[2].min()) - r_xy - 2)
    sx1 = min(vol_shape[2] - 1, int(seed_idx[2].max()) + r_xy + 2)

    # 局部crop（全部操作在这个小块上完成）
    local_seed = seed_mask_3d[sz0:sz1+1, sy0:sy1+1, sx0:sx1+1]
    local_pre  = pre_vol[sz0:sz1+1, sy0:sy1+1, sx0:sx1+1].astype(np.float32)

    lz, ly, lx = local_seed.shape
    if log_cb:
        log_cb(f"  [弹簧圈] 局部crop：{lz}×{ly}×{lx} 体素（原{vol_shape[0]}×{vol_shape[1]}×{vol_shape[2]}）")

    # ── 步骤1：在局部空间做椭球膨胀，得到搜索区 ─────────────────────
    # 仅在crop内膨胀，避免全量操作
    zz, yy, xx = np.ogrid[-r_z:r_z+1, -r_xy:r_xy+1, -r_xy:r_xy+1]
    struct_ellipsoid = ((zz / r_z) ** 2 + (yy / r_xy) ** 2 + (xx / r_xy) ** 2) <= 1.0
    local_search = ndimage.binary_dilation(local_seed, structure=struct_ellipsoid)
    local_search_ring = local_search & ~local_seed   # 去掉核心本身

    # ── 步骤2：候选高HU区 ────────────────────────────────────────────
    candidate = local_search_ring & (local_pre > 150)
    if not candidate.any():
        if log_cb: log_cb("  [弹簧圈] 搜索区内无高HU候选，跳过伪影扩展")
        # 仍需把search_zone写回全量
        search_full = np.zeros(vol_shape, dtype=bool)
        search_full[sz0:sz1+1, sy0:sy1+1, sx0:sx1+1] = local_search_ring
        return np.zeros(vol_shape, dtype=bool), search_full, seed_mask_3d

    # ── 步骤3：局部骨区开运算，识别厚实骨质（仅在crop内） ──────────
    local_bone = local_pre > 150
    struct_open = ndimage.generate_binary_structure(3, 1)   # 3×3×3
    solid_bone = ndimage.binary_opening(local_bone, structure=struct_open, iterations=2)
    solid_bone = ndimage.binary_dilation(solid_bone, structure=struct_open, iterations=1)

    # 细长伪影 = 候选区内、不是厚实骨质的高HU像素
    thin_artifact = candidate & ~solid_bone

    # ── 步骤4：暗带伪影（负HU条纹，与核心相邻） ─────────────────────
    dark_artifact = local_search_ring & (local_pre < -20)

    local_artifact = thin_artifact | dark_artifact

    # ── 写回全量输出数组（除crop外全为False，不分配额外全量中间数组） ──
    artifact_full = np.zeros(vol_shape, dtype=bool)
    artifact_full[sz0:sz1+1, sy0:sy1+1, sx0:sx1+1] = local_artifact

    search_full = np.zeros(vol_shape, dtype=bool)
    search_full[sz0:sz1+1, sy0:sy1+1, sx0:sx1+1] = local_search_ring

    if log_cb:
        log_cb(f"  [弹簧圈] 核心 {int(seed_mask_3d.sum())} 体素 | "
               f"伪影靶区 {int(local_artifact.sum())} 体素 | 血管救援光环已就绪")

    return artifact_full, search_full, seed_mask_3d

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
        inv_parts =[]
        if combined_inv_global is not None:
            inv_parts.append(combined_inv_global)
        if z in annotations.inverse_local_rois and annotations.inverse_local_rois[z]:
            inv_local_masks =[generate_roi_mask_from_polygon(roi, (h, w)) for roi in annotations.inverse_local_rois[z]]
            if inv_local_masks:
                inv_parts.append(np.any(np.stack(inv_local_masks), axis=0))
        if inv_parts:
            combined_inv = np.any(np.stack(inv_parts), axis=0)
            result[z][~combined_inv] = 0
        if combined_global is not None: result[z][combined_global] = 0
        if z in annotations.local_rois and annotations.local_rois[z]:
            local_masks =[generate_roi_mask_from_polygon(roi, (h, w)) for roi in annotations.local_rois[z]]
            if local_masks: result[z][np.any(np.stack(local_masks), axis=0)] = 0
    return result

# ════════════════════════════════════════════════════════════════════
# 交互式图像画布
# ════════════════════════════════════════════════════════════════════
class ImageCanvas(QWidget):
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
        if self.current_z in self.annotations.local_rois: rois.extend(self.annotations.local_rois[self.current_z])
        if self.current_z in self.annotations.inverse_local_rois: rois.extend(self.annotations.inverse_local_rois[self.current_z])
        if self.current_z in self.annotations.metal_rois: rois.extend(self.annotations.metal_rois[self.current_z])
        if self.current_z in self.annotations.clip_rois: rois.extend(self.annotations.clip_rois[self.current_z])
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
                for roi in self.annotations.global_rois: self._draw_polygon(painter, roi, QColor(255, 80, 80, 200), roi == self.selected_roi, roi == self.hover_roi)
                for roi in self.annotations.inverse_global_rois: self._draw_polygon(painter, roi, QColor(0, 220, 220, 220), roi == self.selected_roi, roi == self.hover_roi, inverse=True)
                for roi in self.annotations.metal_rois.get(self.current_z,[]): self._draw_polygon(painter, roi, QColor(255, 140, 0, 220), roi == self.selected_roi, roi == self.hover_roi)
                for roi in self.annotations.clip_rois.get(self.current_z,[]): self._draw_polygon(painter, roi, QColor(0, 200, 255, 220), roi == self.selected_roi, roi == self.hover_roi)
            if self.current_z in self.annotations.local_rois:
                for roi in self.annotations.local_rois[self.current_z]: self._draw_polygon(painter, roi, QColor(80, 255, 80, 200), roi == self.selected_roi, roi == self.hover_roi)
            if self.current_z in self.annotations.inverse_local_rois:
                for roi in self.annotations.inverse_local_rois[self.current_z]: self._draw_polygon(painter, roi, QColor(0, 200, 255, 220), roi == self.selected_roi, roi == self.hover_roi, inverse=True)

        if self.current_polygon and self.current_polygon.points:
            color = QColor(255, 180, 0, 220) if (self.drawing_state.is_drawing and self.drawing_state.get_slice_count() > 1) else QColor(255, 255, 0, 220)
            self._draw_polygon(painter, self.current_polygon, color, drawing=True)
        
        painter.resetTransform()
        
        if self.mode == "window":
            painter.setPen(QColor(255, 255, 255, 180))
            painter.setFont(QFont("Microsoft YaHei", 10))
            painter.drawText(20, 30, f"窗位:{self.wc} 窗宽:{self.ww}")
        
        if self.drawing_state.is_drawing: self._draw_cross_slice_indicator(painter)

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
        elif self.mode in["global_roi", "local_roi", "global_inverse_roi", "local_inverse_roi", "metal_roi", "clip_roi"]:
            if event.button() == Qt.LeftButton and self._is_in_image(ipos):
                if not self.current_polygon:
                    roi_type = "global" if self.mode == "global_roi" else "global_inverse" if self.mode == "global_inverse_roi" else "local_inverse" if self.mode == "local_inverse_roi" else "metal" if self.mode == "metal_roi" else "clip" if self.mode == "clip_roi" else "local"
                    self.current_polygon = PolygonROI(points=[], closed=False, roi_type=roi_type, slice_index=self.current_z)
                    self.drawing_state.is_drawing = True
                    self.drawing_state.start_slice = self.current_z
                    self.drawing_state.current_slice = self.current_z
                    self.drawing_state.visited_slices = {self.current_z}
                    self.drawing_state.vertex_slices = {self.current_z}
                    self.drawing_state.polygon = self.current_polygon
                    if self.on_drawing_state_changed: self.on_drawing_state_changed(self.drawing_state)
                self.current_polygon.points.append(ipos)
                self.drawing_state.vertex_slices.add(self.current_z)  # 记录实际有顶点的层
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
        elif self.mode in["global_roi", "local_roi", "global_inverse_roi", "local_inverse_roi", "metal_roi", "clip_roi"]:
            if self.current_polygon and self.current_polygon.points:
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_moving_vertex:
                self.is_moving_vertex = False
                self.setCursor(Qt.PointingHandCursor)
                self._try_merge_rois()
                self.update()
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
        elif roi_type == "metal":
            # 共享同一对象引用：顶点编辑自动同步到所有层
            shared = self.current_polygon.copy()
            shared.slice_index = min_z
            for z in range(min_z, max_z + 1):
                self.annotations.metal_rois[z].append(shared)
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
        elif roi_type == "clip":
            # 共享同一对象引用：顶点编辑自动同步到所有层
            shared = self.current_polygon.copy()
            shared.slice_index = min_z
            for z in range(min_z, max_z + 1):
                self.annotations.clip_rois[z].append(shared)

        self.current_polygon = None
        self.drawing_state.reset()
        if self.on_drawing_state_changed: self.on_drawing_state_changed(self.drawing_state)
        self._try_merge_rois()
        self.update()
        if self.on_annotation_changed: self.on_annotation_changed()

    def _try_merge_rois(self):
        """
        对所有类型的ROI做融合：
        - global/inverse_global：全局融合一次
        - local/inverse_local/metal/clip：遍历所有有数据的层，
          每层独立融合，确保部分层重叠的两个跨层选区在每个共同层都完成融合
        """
        if not self.annotations: return
        merged_count = 0

        # 全局ROI
        if len(self.annotations.global_rois) > 1:
            old_count = len(self.annotations.global_rois)
            self.annotations.global_rois = merge_overlapping_polygons(self.annotations.global_rois, "global", 0)
            merged_count += old_count - len(self.annotations.global_rois)
        if len(self.annotations.inverse_global_rois) > 1:
            old_count = len(self.annotations.inverse_global_rois)
            self.annotations.inverse_global_rois = merge_overlapping_polygons(self.annotations.inverse_global_rois, "global_inverse", 0)
            merged_count += old_count - len(self.annotations.inverse_global_rois)

        # 逐层ROI：遍历所有有数据的层，而非仅当前层
        for _dict, _type in [
            (self.annotations.local_rois,         "local"),
            (self.annotations.inverse_local_rois, "local_inverse"),
            (self.annotations.metal_rois,         "metal"),
            (self.annotations.clip_rois,          "clip"),
        ]:
            for z in list(_dict.keys()):
                rois = _dict[z]
                if len(rois) > 1:
                    old_count = len(rois)
                    _dict[z] = merge_overlapping_polygons(rois, _type, z)
                    merged_count += old_count - len(_dict[z])

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
        if self.current_z in self.annotations.clip_rois and self.annotations.clip_rois[self.current_z]:
            self.annotations.clip_rois[self.current_z].pop()
        elif self.current_z in self.annotations.metal_rois and self.annotations.metal_rois[self.current_z]:
            self.annotations.metal_rois[self.current_z].pop()
        elif self.current_z in self.annotations.inverse_local_rois and self.annotations.inverse_local_rois[self.current_z]:
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
        
        toolbar_widget = QWidget()
        toolbar_widget.setFixedWidth(200)
        toolbar_widget.setStyleSheet("""
            QWidget { background: #2d2d2d; border-radius: 6px; }
            QLabel { color: #888; font-size: 9pt; }
            QPushButton { background: #3d3d3d; color: white; border: 1px solid #555; border-radius: 4px; padding: 8px; }
            QPushButton:hover { background: #4d4d4d; }
            QPushButton:checked { background: #1a6fbf; border-color: #2a7fcf; }
        """)
        tl = QVBoxLayout(toolbar_widget)
        tl.setContentsMargins(4, 4, 4, 4)
        tl.setSpacing(4)

        from PyQt5.QtWidgets import QScrollArea
        toolbar_scroll = QScrollArea()
        toolbar_scroll.setWidget(toolbar_widget)
        toolbar_scroll.setWidgetResizable(True)
        toolbar_scroll.setFixedWidth(218)
        toolbar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        toolbar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        toolbar_scroll.setStyleSheet("""
            QScrollArea { background: #2d2d2d; border: none; }
            QScrollBar:vertical { background: #2d2d2d; width: 8px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 4px; }
        """)
        toolbar = toolbar_widget
        
        if self.display_scale < 1.0:
            scale_info = QLabel(f"显示：{self.display_scale:.0%}")
            scale_info.setStyleSheet("color: #f39c12; font-size: 8pt;")
            tl.addWidget(scale_info)
        
        tl.addWidget(QLabel("配准后数据"))
        tl.addWidget(self._sep())
        tl.addWidget(self._section_label("交互模式"))
        
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for text, mode in[("漫游", "navigate"), ("调节窗位", "window"), ("清除区域 - 全局", "global_roi"), ("清除区域 - 单层", "local_roi")]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, m=mode: self.set_mode(m))
            self.mode_group.addButton(btn)
            tl.addWidget(btn)
            self.mode_buttons[mode] = btn
        self.mode_buttons["navigate"].setChecked(True)
        
        tl.addWidget(self._sep())
        tl.addWidget(self._section_label("反向选择（保留框内）"))
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
        tl.addWidget(self._section_label("金属保护区（三维形态学）"))
        btn_metal = QPushButton("金属保护 - 跨层")
        btn_metal.setCheckable(True)
        btn_metal.setStyleSheet("""
            QPushButton { background: #4a2e00; color: #ffaa00; border: 1px solid #ff8800; border-radius: 4px; padding: 8px; }
            QPushButton:hover { background: #5a3a00; }
            QPushButton:checked { background: #cc6600; border-color: #ffaa00; color: white; }
        """)
        btn_metal.clicked.connect(lambda _, m="metal_roi": self.set_mode(m))
        self.mode_group.addButton(btn_metal)
        tl.addWidget(btn_metal)
        self.mode_buttons["metal_roi"] = btn_metal

        tl.addWidget(self._sep())
        tl.addWidget(self._section_label("瘤夹保护区（搬平扫原始HU）"))
        btn_clip = QPushButton("瘤夹保护 - 跨层")
        btn_clip.setCheckable(True)
        btn_clip.setStyleSheet("""
            QPushButton { background: #001a33; color: #00eeff; border: 1px solid #00aacc; border-radius: 4px; padding: 8px; }
            QPushButton:hover { background: #002244; }
            QPushButton:checked { background: #005577; border-color: #00eeff; color: white; }
        """)
        btn_clip.clicked.connect(lambda _, m="clip_roi": self.set_mode(m))
        self.mode_group.addButton(btn_clip)
        tl.addWidget(btn_clip)
        self.mode_buttons["clip_roi"] = btn_clip
        
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
        main_layout.addWidget(toolbar_scroll)
        
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = ImageCanvas()
        self.canvas.annotations = self.annotations
        self.canvas.on_slice_changed = self.on_slice_changed
        self.canvas.on_annotation_changed = None
        self.canvas.on_window_changed = lambda wc, ww: None
        self.canvas.on_drawing_state_changed = self.on_drawing_state_changed
        self.canvas.set_volume(self.display_volume, self.display_scale)
        cl.addWidget(self.canvas, 1)
        
        bottom = QWidget()
        bottom.setStyleSheet("background: #2d2d2d; border-radius: 6px;")
        bottom.setMinimumHeight(150)   # 不再 setFixedHeight，内容自动撑开
        bl = QVBoxLayout(bottom)
        bl.setSpacing(6)
        bl.setContentsMargins(6, 6, 6, 6)
        
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
        self.canvas.on_status_message = lambda m: self.drawing_indicator.setText(m)
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

        # ── 精细分区手动标定（眶顶板消失层→鞍底/岩尖上界层）──────────
        # 上界：眶顶板消失那层；下界：鞍底或岩尖上界那层。
        # 必须先点击"设为上界"或"设为下界"才生效，否则使用程序默认常量。
        self._willis_user_set = False   # 标志：医生是否真正标定过
        r3 = QHBoxLayout()

        lbl_w = QLabel("精细分区：")
        lbl_w.setStyleSheet("color: white; font-size: 9pt;")
        r3.addWidget(lbl_w)

        n_total = self.display_volume.shape[0]
        n_mid   = max(1, n_total // 2)   # 初始值设为中间层，top==bot → 不满足 top>bot → 不会误生效

        _sb  = "QSpinBox { background: #3d3d3d; color: white; padding: 6px; }"
        _btn = "QPushButton { background: #3d3d3d; color: white; padding: 6px 12px; }"

        self.willis_top = QSpinBox()
        self.willis_top.setStyleSheet(_sb)
        self.willis_top.setRange(1, n_total)
        self.willis_top.setValue(n_mid)
        r3.addWidget(self.willis_top)

        self._willis_btn_top = QPushButton("设为上界（眶顶板消失层）")
        self._willis_btn_top.setStyleSheet(_btn)
        def _set_top():
            self.willis_top.setValue(self.canvas.current_z + 1)
            self._willis_user_set = True
        self._willis_btn_top.clicked.connect(_set_top)
        r3.addWidget(self._willis_btn_top)

        r3.addWidget(QLabel("  ↓   "))

        self.willis_bot = QSpinBox()
        self.willis_bot.setStyleSheet(_sb)
        self.willis_bot.setRange(1, n_total)
        self.willis_bot.setValue(n_mid)
        r3.addWidget(self.willis_bot)

        self._willis_btn_bot = QPushButton("设为下界（鞍底/岩尖上界）")
        self._willis_btn_bot.setStyleSheet(_btn)
        def _set_bot():
            self.willis_bot.setValue(self.canvas.current_z + 1)
            self._willis_user_set = True
        self._willis_btn_bot.clicked.connect(_set_bot)
        r3.addWidget(self._willis_btn_bot)

        # willis_check 仍保留兼容性，隐藏不显示
        self.willis_check = QCheckBox("")
        self.willis_check.setVisible(False)

        r3.addStretch()
        bl.addLayout(r3)
        cl.addWidget(bottom)
        main_layout.addWidget(center, 1)
        
        right = QWidget()
        right.setFixedWidth(190)
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
        help_label = QLabel("1-4 切换模式\n5-7 顶点编辑\n8-9 反向保留\n0 金属保护\n= 瘤夹保护\nEsc 取消\nDel 删除\nCtrl+Z 撤销")
        help_label.setStyleSheet("color: #aaa; font-size: 8pt;")
        rl.addWidget(help_label)
        rl.addStretch()
        main_layout.addWidget(right)

    def _sep(self):
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFixedHeight(2)
        f.setStyleSheet("background: #444;")
        return f

    def _section_label(self, text):
        lbl = QLabel(f"-- {text} --")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #666; font-size: 8.5pt;")
        lbl.setWordWrap(True)
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
        has_local = (z in self.annotations.local_rois and self.annotations.local_rois[z])
        has_inv = (z in self.annotations.inverse_local_rois and self.annotations.inverse_local_rois[z])
        has_metal = (z in self.annotations.metal_rois and self.annotations.metal_rois[z])
        has_clip = (z in self.annotations.clip_rois and self.annotations.clip_rois[z])

        if has_local or has_inv or has_metal or has_clip:
            mb = QMessageBox(self)
            mb.setWindowTitle("确认")
            mb.setText(f"清除第 {z + 1} 层的所有 ROI?")
            mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            mb.setStyleSheet("QMessageBox { background:#f0f0f0; } QLabel { color:#1a1a1a; font-size:10pt; } QPushButton { background:#ddd; color:#1a1a1a; padding:6px 18px; border-radius:3px; }")
            if mb.exec_() == QMessageBox.Yes:
                if z in self.annotations.local_rois: self.annotations.local_rois[z].clear()
                if z in self.annotations.inverse_local_rois: self.annotations.inverse_local_rois[z].clear()
                if z in self.annotations.metal_rois: self.annotations.metal_rois[z].clear()
                if z in self.annotations.clip_rois: self.annotations.clip_rois[z].clear()
                self.canvas._clear_edit_state()
                self.canvas.update()

    def accept_annotation(self):
        self.annotations.z_range_start = self.range_start.value() - 1
        self.annotations.z_range_end   = self.range_end.value() - 1
        # 精细分区标定：必须医生主动点击了"设为上/下界"才生效
        # 仅凭 SpinBox 默认值满足 top>bot 不算标定，防止误覆盖程序常量
        if self._willis_user_set:
            top_val = self.willis_top.value() - 1
            bot_val = self.willis_bot.value() - 1
            if top_val > bot_val:
                self.annotations.willis_layer_top = top_val
                self.annotations.willis_layer_bot = bot_val
            else:
                self.annotations.willis_layer_top = None
                self.annotations.willis_layer_bot = None
        else:
            # 医生未标定，保持 None，使用程序默认常量
            self.annotations.willis_layer_top = None
            self.annotations.willis_layer_bot = None
        self.result_accepted = True
        self.accept()
 
    def skip_annotation(self):
        self.annotations.global_rois.clear()
        self.annotations.local_rois.clear()
        self.annotations.inverse_global_rois.clear()
        self.annotations.inverse_local_rois.clear()
        self.annotations.metal_rois.clear()
        self.annotations.clip_rois.clear()
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
                     Qt.Key_8: "global_inverse_roi", Qt.Key_9: "local_inverse_roi", Qt.Key_0: "metal_roi",
                     Qt.Key_Equal: "clip_roi"}
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

# 主处理线程 (★★★分段减影防溢出版★★★)
# ════════════════════════════════════════════════════════════════════
class ProcessThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, pre_series, post_series, output_dir, options, pre_annotation=None):
        super().__init__()
        self.pre_series = pre_series
        self.post_series = post_series
        self.output_dir = output_dir
        self.options = options
        self.cancelled = False
        self.pre_annotation: Optional[AnnotationData] = pre_annotation

    def cancel(self):
        self.cancelled = True

    def run(self):
        pre_vol = None
        aligned_vol = None
        result_vol = None
        success = False
        finish_msg = ""

        try:
            t0 = time.time()
            opt = self.options
            opt_level = opt.get('optimization_level', 'none')
            opt_name = OPTIMIZATION_LEVELS.get(opt_level, {}).get('name', '无')
            num_chunks = opt.get('num_chunks', 1)

            self.log.emit("=" * 55)
            self.log.emit(f"CTA 减影 V5.0 (三维形态学终极破局版) | 优化:{opt_name}")
            self.log.emit(f"[分段] 设定分为 {num_chunks} 段处理")
            self.log.emit(f"  [内存] 初始：{get_memory_mb():.0f} MB")
            if self.pre_annotation: self.log.emit("[预标注] 已接收标注数据，将裁切处理范围")
            self.log.emit("=" * 55)

            reader = sitk.ImageSeriesReader()
            def _get_ipp_z(filepath):
                try:
                    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                    ipp = getattr(ds, 'ImagePositionPatient', None)
                    if ipp is not None:
                        return float(ipp[2])
                    sl = getattr(ds, 'SliceLocation', None)
                    if sl is not None:
                        return float(sl)
                except Exception:
                    pass
                return 0.0

            def get_files(series):
                s_dir = os.path.dirname(series.files[0][1])
                s_dir_safe = _get_short_path(s_dir)
                try:
                    f_list = reader.GetGDCMSeriesFileNames(s_dir_safe, series.series_uid)
                except Exception:
                    f_list =[]
                if f_list and len(f_list) == series.file_count:
                    return list(f_list)
                fallback = [f[1] for f in series.files]
                try:
                    fallback.sort(key=_get_ipp_z)
                except Exception:
                    pass
                return fallback

            pre_files = get_files(self.pre_series)
            post_files = get_files(self.post_series)
            
            if not pre_files or not post_files:
                finish_msg = "读取失败：文件列表为空"
                return

            # ann_total_slices：标注窗口（post序列）原始长度，用于Willis标定层号→sft转换
            # original_total_slices：trim后pre序列长度，用于逐层sft计算，两者可能不同
            ann_total_slices = len(post_files)   # 医生看到的层数，trim前

            if len(pre_files) != len(post_files):
                _trim_offset = abs(len(post_files) - len(pre_files))
                self.log.emit(f"\n  ⚠ 层数不等：平扫 {len(pre_files)} 层，增强 {len(post_files)} 层")
                pre_files, post_files = trim_to_common_z_range(pre_files, post_files, self.log.emit)
                if not pre_files or not post_files:
                    finish_msg = "层数对齐失败：两序列无有效重叠范围"
                    return
                self.log.emit(f"  [坐标] Willis转换用post原始长度 {ann_total_slices} 层，"
                              f"逐层sft用trim后长度 {len(pre_files)} 层，偏差已分离")

            total_slices = len(pre_files)
            original_total_slices = total_slices  # pre序列长度，逐层sft专用

            ann_rois: Optional[AnnotationData] = None
            az0 = 0

            if self.pre_annotation:
                az0 = self.pre_annotation.z_range_start
                az1 = self.pre_annotation.z_range_end
                az0 = max(0, min(az0, total_slices - 1))
                az1 = max(az0, min(az1, total_slices - 1))
                pre_files  = pre_files[az0: az1 + 1]
                post_files = post_files[az0: az1 + 1]
                total_slices = len(pre_files)
                
                ann_rois = AnnotationData(
                    global_rois=self.pre_annotation.global_rois,
                    local_rois=self.pre_annotation.local_rois,
                    inverse_global_rois=self.pre_annotation.inverse_global_rois,
                    inverse_local_rois=self.pre_annotation.inverse_local_rois,
                    metal_rois=self.pre_annotation.metal_rois,
                    clip_rois=self.pre_annotation.clip_rois,
                    z_range_start=0,
                    z_range_end=total_slices - 1,
                    scale_factor=self.pre_annotation.scale_factor,
                    willis_layer_top=self.pre_annotation.willis_layer_top,
                    willis_layer_bot=self.pre_annotation.willis_layer_bot,
                )
                self.log.emit(f"[预标注] 裁切后处理范围：原序列第 {az0+1}~{az1+1} 层，共 {total_slices} 层")

            chunk_size = math.ceil(total_slices / num_chunks)
            global_output_count = 0
            series_uid = generate_uid()

            for chunk_idx in range(num_chunks):
                if self.cancelled: return
                
                start_idx = chunk_idx * chunk_size
                end_idx = min(total_slices, (chunk_idx + 1) * chunk_size)
                if start_idx >= total_slices: break
                
                chunk_pre_files = pre_files[start_idx:end_idx]
                chunk_post_files = post_files[start_idx:end_idx]
                chunk_depth = len(chunk_pre_files)
                
                def update_prog(p, _cidx=chunk_idx):
                    self.progress.emit(int((_cidx * 100 + p) / num_chunks))
                    
                if num_chunks > 1:
                    self.log.emit(f"\n{'=' * 40}")
                    self.log.emit(f"▶ 开始处理分段 {chunk_idx+1}/{num_chunks} (源序列第 {start_idx+1} ~ {end_idx} 层)")
                    self.log.emit(f"{'=' * 40}")
                    
                self.log.emit("\n[1/7] 读取平扫序列 (当前段)...")
                update_prog(5)
                pre_img = _sitk_read_image_safe(chunk_pre_files, sitk.sitkInt16)
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
                self.log.emit(f"[极限释放] 当前段平扫对象已销毁，内存：{get_memory_mb():.0f} MB")
                update_prog(10)

                self.log.emit("\n[2/7] 读取增强序列与 3D 旋转配准 (当前段)...")
                post_img = _sitk_read_image_safe(chunk_post_files, sitk.sitkInt16)
                moving_shrink = sitk.Cast(sitk.Shrink(post_img, shrink_factors), sitk.sitkFloat32)
                
                final_transform = compute_3d_registration_transform(fixed_shrink, moving_shrink, self.log.emit, lambda v: update_prog(10 + v))
                del fixed_shrink, moving_shrink
                force_gc()
                if self.cancelled: return

                self.log.emit("\n[3/7] 应用空间重采样...")
                ref_img = sitk.Image(ref_size, sitk.sitkInt16)
                ref_img.SetOrigin(ref_origin)
                ref_img.SetSpacing(ref_spacing)
                ref_img.SetDirection(ref_direction)

                _post_img_origin    = post_img.GetOrigin()
                _post_img_spacing   = post_img.GetSpacing()
                _post_img_direction = post_img.GetDirection()

                aligned_img = sitk.Resample(post_img, ref_img, final_transform, sitk.sitkLinear, -1000.0, sitk.sitkInt16)
                del post_img, ref_img
                force_gc()
                
                aligned_vol = np.empty((chunk_depth, h, w), dtype=np.int16)
                for z in range(chunk_depth): aligned_vol[z] = sitk.GetArrayFromImage(aligned_img[:, :, z])
                del aligned_img
                force_gc()
                update_prog(35)
                
                annotation_data = None
                z_start, z_end = 0, chunk_depth - 1

                if ann_rois:
                    seg_ann = AnnotationData(
                        z_range_start=0, z_range_end=chunk_depth - 1,
                        scale_factor=ann_rois.scale_factor,
                        willis_layer_top=ann_rois.willis_layer_top,
                        willis_layer_bot=ann_rois.willis_layer_bot,
                    )
                    seg_ann.global_rois = ann_rois.global_rois
                    seg_ann.inverse_global_rois = ann_rois.inverse_global_rois
                    for orig_z, rois in ann_rois.local_rois.items():
                        local_z = orig_z - az0 - start_idx
                        if 0 <= local_z < chunk_depth: seg_ann.local_rois[local_z] = rois
                    for orig_z, rois in ann_rois.inverse_local_rois.items():
                        local_z = orig_z - az0 - start_idx
                        if 0 <= local_z < chunk_depth: seg_ann.inverse_local_rois[local_z] = rois
                    for orig_z, rois in ann_rois.metal_rois.items():
                        local_z = orig_z - az0 - start_idx
                        if 0 <= local_z < chunk_depth: seg_ann.metal_rois[local_z] = rois
                    for orig_z, rois in ann_rois.clip_rois.items():
                        local_z = orig_z - az0 - start_idx
                        if 0 <= local_z < chunk_depth: seg_ann.clip_rois[local_z] = rois
                    annotation_data = seg_ann
                    self.log.emit(f"\n[4/7] 应用预标注 ROI（本段）")
                else:
                    self.log.emit("\n[4/7] 无预标注，全段处理")
                update_prog(45)

                self.log.emit(f"\n[5/7] 减影计算...")
                spacing_xy = (spacing_zyx[1], spacing_zyx[2])

                has_metal = annotation_data is not None and any(annotation_data.metal_rois.values())
                if has_metal:
                    self.log.emit(f"[金属保护] 已启用！(覆盖 {len(annotation_data.metal_rois)} 层)")
                has_clip = annotation_data is not None and any(annotation_data.clip_rois.values())
                if has_clip:
                    self.log.emit(f"[瘤夹保护] 已启用！(覆盖 {len(annotation_data.clip_rois)} 层，搬平扫原始HU)")

                eff_conv_end  = ZONE_CONVEXITY_END
                eff_willis_end = ZONE_WILLIS_END
                if (annotation_data is not None and
                        annotation_data.willis_layer_top is not None and
                        annotation_data.willis_layer_bot is not None):
                    og_top = annotation_data.willis_layer_top
                    og_bot = annotation_data.willis_layer_bot
                    # 用 ann_total_slices（post原始长度）转换，与医生标注坐标系一致
                    eff_conv_end  = max(0, (ann_total_slices - 1) - og_top)
                    eff_willis_end = max(eff_conv_end + 1,
                                        (ann_total_slices - 1) - og_bot)
                    self.log.emit(
                        f"[Willis] 医生标定：上界第{annotation_data.willis_layer_top+1}层"
                        f"→ sft={eff_conv_end}，"
                        f"下界第{annotation_data.willis_layer_bot+1}层"
                        f"→ sft={eff_willis_end}"
                        f"（参考序列 {ann_total_slices} 层，程序默认 {ZONE_CONVEXITY_END}/{ZONE_WILLIS_END}）")
                else:
                    self.log.emit(
                        f"[Willis] 使用程序默认边界"
                        f"（颅盖/Willis: {ZONE_CONVEXITY_END}层，"
                        f"Willis/颅底: {ZONE_WILLIS_END}层）")

                # [瘤夹处理] 已由逐层"瘤夹保护区"取代，3D机械流程已移除

                for z in range(z_start, z_end + 1):
                    if self.cancelled: return
                    pre_s = pre_vol[z].astype(np.float32)
                    post_s = aligned_vol[z].astype(np.float32)

                    if pre_s.max() > -500:
                        real_diff = np.clip(post_s - pre_s, 0, None)
                        rescue_threshold = 45 * opt['vessel_sensitivity']

                        # ── 金属保护区：减影前预处理（4.5.5工作流）───────────────
                        # 在工作副本上完成血管救援，喂给减影引擎的数据已干净，
                        # 引擎不受金属伪影污染，不会产生假骨抑制或误杀邻近血管。
                        pre_s_work  = pre_s.copy()
                        post_s_work = post_s.copy()
                        m_core, m_artifact = None, None

                        if has_metal and z in annotation_data.metal_rois:
                            masks = [generate_roi_mask_from_polygon(roi, (h, w)) for roi in annotation_data.metal_rois[z]]
                            if masks:
                                metal_mask_z = np.any(np.stack(masks), axis=0)
                                raw_metal_core = metal_mask_z & (pre_s > 2800)  # 铂合金真实HU>3000
                                m_core = ndimage.binary_dilation(raw_metal_core, iterations=2) & metal_mask_z
                                strict_bone = pre_s > 400
                                strict_solid_bone = ndimage.binary_dilation(
                                    ndimage.binary_opening(strict_bone, iterations=2), iterations=2) & strict_bone
                                bright_artifact = metal_mask_z & (pre_s > 120) & (~strict_solid_bone) & (~m_core)
                                dark_artifact   = metal_mask_z & (pre_s < -20)
                                raw_artifact    = bright_artifact | dark_artifact
                                m_halo          = metal_mask_z & (~m_core)

                                # 血管救援：ROI框内有真实造影增强→假扮软组织喂给引擎
                                rescue_mask_m = m_halo & (real_diff > rescue_threshold)
                                pre_s_work[rescue_mask_m]  = 40.0
                                post_s_work[rescue_mask_m] = 40.0 + real_diff[rescue_mask_m]

                                # 排除已救援像素，剩余为纯伪影
                                m_artifact = raw_artifact & ~rescue_mask_m

                        diff_pos_work = np.clip(post_s_work - pre_s_work, 0, None)
                        cache = precompute_slice_cache(pre_s_work, diff_pos_work, opt['quality_mode'])
                        equip = detect_equipment(pre_s, post_s, body=cache.body)  # 设备检测用原始数据
                        
                        global_z = start_idx + z
                        original_global_z = az0 + global_z
                        slices_from_top = get_slices_from_top(original_global_z, original_total_slices)
                        zone = get_anatomical_zone_v2(slices_from_top, eff_conv_end, eff_willis_end)
                        adj_bone, adj_vessel = get_zone_adjusted_params_v2(
                            slices_from_top, opt['bone_strength'], opt['vessel_sensitivity'],
                            eff_conv_end, eff_willis_end)
                        cache.slices_from_top = slices_from_top
                        cache.eff_conv_end = eff_conv_end
                        bone_zones = detect_problematic_bone_zones_v2(pre_s_work, slices_from_top, eff_conv_end)

                        if opt_level == 'none':
                            res = quality_subtraction_cached(pre_s_work, post_s_work, adj_bone, adj_vessel, cache) if opt['quality_mode'] else fast_subtraction(pre_s_work, post_s_work, adj_bone, adj_vessel, body=cache.body, cache=cache)
                        elif opt_level == 'light':
                            res = optimized_subtraction_light_v3(pre_s_work, post_s_work, adj_bone, adj_vessel, cache)
                        elif opt_level == 'standard':
                            res = optimized_subtraction_standard_v3(pre_s_work, post_s_work, adj_bone, adj_vessel, cache, spacing_xy, 0.5)
                        else:
                            res = optimized_subtraction_deep_v3(pre_s_work, post_s_work, adj_bone, adj_vessel, cache, spacing_xy, spacing_zyx, 0.5)

                        gain = np.ones_like(res, dtype=np.float32)
                        gain = enhanced_bone_suppression_v299(pre_s_work, diff_pos_work, gain, bone_zones, adj_bone, adj_vessel)
                        res = res * gain

                        res[equip] = 0
                        res[~cache.body] = 0
                        
                        if opt['clean_bone_edges']: res = clean_bone_edges_cached(res, pre_s_work, cache, 'quality' if opt['quality_mode'] else 'fast')
                        if opt['smooth_sigma'] > 0: res = edge_preserving_smooth(res, pre_s_work, opt['smooth_sigma'])
                        
                        res_scaled = res * opt['vessel_enhance']

                        _WV_LO, _WV_HI, _WV_GAIN = 50.0, 150.0, 1.35
                        _wv_mask = (res_scaled > _WV_LO) & (res_scaled < _WV_HI) & cache.body
                        if _wv_mask.any():
                            _labeled, _n = ndimage.label(_wv_mask)
                            if _n > 0:
                                _sizes = np.bincount(_labeled.ravel(), minlength=_n + 1)
                                _valid_labels = np.where(_sizes >= 3)[0]
                                _valid_labels = _valid_labels[_valid_labels > 0]
                                if len(_valid_labels) > 0:
                                    _valid_mask = np.isin(_labeled, _valid_labels)
                                    _t = ((res_scaled[_valid_mask] - _WV_LO) / (_WV_HI - _WV_LO))
                                    res_scaled[_valid_mask] = res_scaled[_valid_mask] * (1.0 + (_WV_GAIN - 1.0) * 0.5 * (1.0 - np.cos(_t * np.pi)))

                        # ── 金属保护区：减影后后处理 ─────────────────────────────
                        # 血管救援已在减影前完成，此处仅做两件事：
                        # 1. 点杀纯伪影（亮条纹/暗带）
                        # 2. 金属本体幽灵化：随机噪声填充 → VR毛玻璃效果 / MIP镂空轮廓
                        if m_artifact is not None and m_artifact.any():
                            res_scaled[m_artifact] = 0.0

                        if m_core is not None and m_core.any():
                            n_metal = int(m_core.sum())
                            res_scaled[m_core] = np.random.normal(
                                200, 60, size=n_metal).clip(50, 400).astype(np.float32)

                        # 瘤夹保护区：直接搬平扫原始HU，免疫减影
                        if has_clip and z in annotation_data.clip_rois:
                            _clip_masks = [generate_roi_mask_from_polygon(roi, (h, w))
                                           for roi in annotation_data.clip_rois[z]]
                            if _clip_masks:
                                _clip_zone = np.any(np.stack(_clip_masks), axis=0)
                                # 用高阈值找实心金属核（>1200 HU，避开骨质上限~700 HU）
                                # 再向外膨胀2像素，把CT原生的部分容积过渡带纳入
                                # 这样既保留clip边缘的平滑过渡，又不会延伸到血管
                                # ── 瘤夹保护区三个可调参数 ──────────────────────────
                                _CLIP_HU_THR   = 1200.0  # 金属核阈值(HU)：高→核小，低→核大
                                _CLIP_DILATE   = 2       # 膨胀像素数：覆盖过渡带，不影响轮廓外形
                                _CLIP_HU_SCALE = 0.3     # HU缩放比：1.0=原始，0.5/0.3=降低过渡带
                                _CLIP_VESSEL_THR = 400.0 # 二次筛选下限：低于此值为软组织/血管，不操作
                                # ──────────────────────────────────────────────────
                                _metal_core = _clip_zone & (pre_s > _CLIP_HU_THR)
                                if _metal_core.any():
                                    _metal_zone = ndimage.binary_dilation(_metal_core, iterations=_CLIP_DILATE)
                                    _metal_zone &= _clip_zone        # 严格限制在画框内
                                    _metal_zone &= (pre_s > _CLIP_VESSEL_THR)  # 排除软组织/血管体素
                                    res_scaled[_metal_zone] = pre_s[_metal_zone] * _CLIP_HU_SCALE

                        aligned_vol[z] = np.clip(res_scaled, -32768, 32767).astype(np.int16)
                    else:
                        aligned_vol[z] = 0
                        
                    if z % 50 == 0: update_prog(45 + int(((z - z_start) / max(1, z_end - z_start + 1)) * 30))

                del pre_vol
                pre_vol = None
                force_gc()
                result_vol = aligned_vol
                aligned_vol = None
                
                if annotation_data and (annotation_data.global_rois or any(annotation_data.local_rois.values()) or annotation_data.inverse_global_rois or any(annotation_data.inverse_local_rois.values())):
                    self.log.emit("  [应用 ROI] 挖除标记区域...")
                    result_vol = apply_roi_masks_to_volume(result_vol, annotation_data, z_start, z_end, self.log.emit)
                    force_gc()

                self.log.emit("\n[6/7] 后处理...")
                min_area_2d = opt['min_vessel_size'] * 3

                # 1. 先做常规的三维降噪清理和孤立点抹除
                result_vol = apply_3d_cleanup(result_vol, min_area=min_area_2d, log_cb=self.log.emit)
                if opt_level == 'deep': 
                    result_vol = shape_analysis_fast(result_vol, spacing_zyx, self.log.emit)

                update_prog(80)
                force_gc()

                self.log.emit(f"\n[7/7] 写入磁盘...")
                os.makedirs(self.output_dir, exist_ok=True)
                safe_output_dir = _get_short_path(self.output_dir)
                
                range_vol = result_vol[z_start:z_end + 1]
                g_min, g_max = float(range_vol.min()), float(range_vol.max())
                g_slope = (g_max - g_min) / 4095.0 if g_max > g_min else 1.0

                new_frame_of_reference_uid = generate_uid()

                for z in range(z_start, z_end + 1):
                    if self.cancelled: return
                    
                    ds = pydicom.dcmread(chunk_pre_files[z])
                    ds.SpecificCharacterSet = 'ISO_IR 192'
                     
                    pix = result_vol[z].astype(np.float32)

                    # 【金属本体最终覆写：写盘时再注入一次随机噪声】
                    # apply_3d_cleanup 可能把幽灵化后的小连通域当孤立点清掉，
                    # 在所有后处理结束后、写盘前再覆写一次，确保毛玻璃效果入盘。
                    if has_metal and annotation_data is not None:
                        if z in annotation_data.metal_rois and annotation_data.metal_rois[z]:
                            _m_masks = [generate_roi_mask_from_polygon(roi, (h, w))
                                        for roi in annotation_data.metal_rois[z]]
                            if _m_masks:
                                _metal_z   = np.any(np.stack(_m_masks), axis=0)
                                _raw_core  = _metal_z & (ds.pixel_array.astype(np.float32)
                                             * float(getattr(ds, 'RescaleSlope', 1.0))
                                             + float(getattr(ds, 'RescaleIntercept', 0.0)) > 2800)
                                _m_core_z  = ndimage.binary_dilation(_raw_core, iterations=2) & _metal_z
                                _n = int(_m_core_z.sum())
                                if _n > 0:
                                    pix[_m_core_z] = np.random.uniform(50, 300, size=_n).astype(np.float32)
                    
                    pix_int = ((pix - g_min) / (g_max - g_min) * 4095).astype(np.int16) if g_max > g_min else np.zeros_like(pix, dtype=np.int16)
                     
                    ds.PixelData = pix_int.tobytes()
                    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
                    ds.PixelRepresentation, ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 1, 'MONOCHROME2'
                    
                    for tag in['LossyImageCompression', 'LossyImageCompressionRatio', 'LossyImageCompressionMethod']:
                        if hasattr(ds, tag): delattr(ds, tag)
                    
                    ds.RescaleSlope, ds.RescaleIntercept = g_slope, g_min
                    ds.WindowCenter, ds.WindowWidth = opt['wc'], opt['ww']
                    
                    ann_tag = "+ROI" if annotation_data else ""
                    ds.SeriesDescription = f"CTA Sub V5.0[{opt_name}]{ann_tag}"
                    ds.SeriesInstanceUID = series_uid
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'AXIAL']

                    new_sop_uid = generate_uid()
                    ds.SOPInstanceUID = new_sop_uid
                    ds.file_meta.MediaStorageSOPInstanceUID = new_sop_uid

                    ds.InstanceNumber = global_output_count + 1
                    ds.FrameOfReferenceUID = new_frame_of_reference_uid
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                    ds.save_as(os.path.join(safe_output_dir, f"SUB_{global_output_count + 1:04d}.dcm"), write_like_original=False)
                    
                    global_output_count += 1
                    if z % 50 == 0: update_prog(80 + int(((z - z_start) / max(1, z_end - z_start + 1)) * 15))
                
                del result_vol
                result_vol = None
                force_gc()

            elapsed = time.time() - t0
            success = True
            finish_msg = f"处理完成!\n输出：{global_output_count} 层\n耗时：{elapsed:.1f}秒\n目录：{self.output_dir}"
            if num_chunks > 1: finish_msg += f"\n分段信息：成功分为 {num_chunks} 段处理"
            if ann_rois and (ann_rois.global_rois or any(ann_rois.local_rois.values())):
                finish_msg += f"\n预标注 ROI 已应用"
            self.log.emit(f"\n{'=' * 55}")
            self.log.emit(f"✅ 完成！耗时：{elapsed:.1f}s")
            self.log.emit(f"[内存] 最终：{get_memory_mb():.0f} MB")

        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            finish_msg = str(e)

        finally:
            if pre_vol is not None: del pre_vol
            if aligned_vol is not None: del aligned_vol
            if result_vol is not None: del result_vol
            force_gc()
            self.log.emit(f"[内存] 退出后：{get_memory_mb():.0f} MB")
            if self.cancelled:
                self.finished_signal.emit(False, "⛔ 用户已取消，内存已释放")
            else:
                self.finished_signal.emit(success, finish_msg)

# ════════════════════════════════════════════════════════════════════
# 颈椎专用减影参数（独立于头颅全局参数，便于单独调试）
# ════════════════════════════════════════════════════════════════════
#
# 颈椎段解剖特点（与颅内血管的主要区别）：
#   · 颈总动脉 / 颈内动脉颈段 / 椎动脉：管径 4~8 mm，增强峰值 200~400 HU
#   · 椎体骨质：皮质约 300~600 HU，松质约 150~300 HU，比颅底薄骨 HU 低
#   · 颈部肌肉/脂肪软组织背景：-80~80 HU
#   · 无颅底岩骨等超高HU结构（除非有骨刺/钙化，但一般不超过 800 HU）
#
# 参数设定依据：
#   CERV_BONE_STRENGTH  = 1.5   ← 略强于头颅标准(1.2)：颈椎椎体骨质厚实，
#                                  beam-hardening 漂移比颅盖骨更大，需要更强抑制
#   CERV_VESSEL_SENS    = 1.2   ← 略高于头颅标准(1.0)：颈部大血管HU高、管径大，
#                                  可以用更高灵敏度，而不用担心小血管假阳性
#   CERV_VESSEL_ENHANCE = 1.8   ← 低于头颅标准(2.0)：颈部血管本身HU已够高，
#                                  过度增强会使椎旁静脉丛假阳性
#   CERV_SMOOTH_SIGMA   = 0.8   ← 略大于头颅标准(0.7)：颈椎位移噪声略大，
#                                  稍多平滑抑制逐层配准残差
#   CERV_MIN_VESSEL_PX  = 4     ← 轻微降低(头颅5)：颈部小血管较少，
#                                  但保留穿支不能太激进过滤
#   CERV_Z_SMOOTH_SIGMA = 1.5   ← Z轴位移序列高斯平滑：防止逐层配准的
#                                  随机噪声引起血管在Z方向折线断开
#                                  sigma=1.5 对应约 ±1 层的平滑半径，
#                                  足以消除1层估计误差，不会掩盖真实颈椎运动
#   CERV_MAX_SHIFT_PX   = 30    ← 允许的最大层间位移（像素）：
#                                  颈椎前后屈伸可达20~30mm（0.625mm/px → ~32~48px），
#                                  取30作为安全上限，超出说明序列质量太差
# ─────────────────────────────────────────────────────────────────
CERV_BONE_STRENGTH  = 1.5
CERV_VESSEL_SENS    = 1.2
CERV_VESSEL_ENHANCE = 1.8
CERV_SMOOTH_SIGMA   = 0.8
CERV_MIN_VESSEL_PX  = 4
CERV_Z_SMOOTH_SIGMA = 1.5
CERV_MAX_SHIFT_PX   = 30


# ════════════════════════════════════════════════════════════════════
# 颈椎专用减影线程（2.5D配准：逐层FFT平移 + Z轴柔性约束）
# ════════════════════════════════════════════════════════════════════
class CervicalProcessThread(QThread):
    """
    颈椎段专用减影处理线程。

    配准策略：2.5D（Slice-by-Slice + Z轴高斯平滑）
    ─────────────────────────────────────────────
    头颅 ProcessThread 用的是 SimpleITK VersorRigid3DTransform（全局3D旋转）。
    这对刚性颅骨有效，但颈椎是多节段准刚性体，前后屈伸时C3和C7的运动完全独立，
    全局一个旋转变换追不上，结果必然是"上段对了下段错"。

    本线程改为：
      Step-1  逐层独立 FFT 相位相关（已有 fft_phase_correlation_2d）估计 (dy, dx)
      Step-2  对整列位移序列做 Z 轴高斯平滑（sigma=CERV_Z_SMOOTH_SIGMA）：
              消除单层估计噪声，保证颈部血管在 Z 方向不出现台阶断开，
              同时保留跨多层的真实缓变位移（颈椎屈伸是渐变的，不是突跳的）
      Step-3  用 ndimage.shift 逐层应用平滑后位移
      Step-4  调用已有的减影/骨抑制/后处理函数，骨抑制参数固定为
              颈椎专用常量（CERV_* 系列），不走解剖分区逻辑，
              全程等效于 skull_base 区（强骨抑制）
    """
    progress = pyqtSignal(int)
    log      = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, pre_series, post_series, output_dir, options, pre_annotation=None):
        super().__init__()
        self.pre_series    = pre_series
        self.post_series   = post_series
        self.output_dir    = output_dir
        self.options       = options        # 仅用 wc/ww/clean_bone_edges，其余用 CERV_* 常量
        self.cancelled     = False
        self.pre_annotation: Optional[AnnotationData] = pre_annotation

    def cancel(self):
        self.cancelled = True

    # ── 内部工具：读取序列文件列表（与 ProcessThread.get_files 逻辑相同）──
    @staticmethod
    def _get_files(series):
        reader = sitk.ImageSeriesReader()
        s_dir = os.path.dirname(series.files[0][1])
        s_dir_safe = _get_short_path(s_dir)
        try:
            f_list = reader.GetGDCMSeriesFileNames(s_dir_safe, series.series_uid)
        except Exception:
            f_list = []
        if f_list and len(f_list) == series.file_count:
            return list(f_list)

        def _get_ipp_z(fp):
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True)
                ipp = getattr(ds, 'ImagePositionPatient', None)
                if ipp is not None:
                    return float(ipp[2])
                sl = getattr(ds, 'SliceLocation', None)
                return float(sl) if sl is not None else 0.0
            except Exception:
                return 0.0

        fallback = [f[1] for f in series.files]
        try:
            fallback.sort(key=_get_ipp_z)
        except Exception:
            pass
        return fallback

    def run(self):
        pre_vol    = None
        result_vol = None
        success    = False
        finish_msg = ""

        try:
            t0  = time.time()
            opt = self.options

            self.log.emit("=" * 55)
            self.log.emit("颈椎专用减影 V5.0 | 2.5D逐层配准 + Z轴柔性约束")
            self.log.emit(f"  骨抑制强度  : {CERV_BONE_STRENGTH:.1f}  (颈椎专用)")
            self.log.emit(f"  血管敏感性  : {CERV_VESSEL_SENS:.1f}  (颈椎专用)")
            self.log.emit(f"  血管增强    : {CERV_VESSEL_ENHANCE:.1f}  (颈椎专用)")
            self.log.emit(f"  Z轴平滑σ   : {CERV_Z_SMOOTH_SIGMA:.1f} 层")
            self.log.emit(f"  最大允许位移: {CERV_MAX_SHIFT_PX} px")
            self.log.emit(f"  [内存] 初始：{get_memory_mb():.0f} MB")
            self.log.emit("=" * 55)

            # ── 1. 获取文件列表 ──────────────────────────────────────
            pre_files  = self._get_files(self.pre_series)
            post_files = self._get_files(self.post_series)
            if not pre_files or not post_files:
                finish_msg = "读取失败：文件列表为空"
                return

            ann_total_slices = len(post_files)

            if len(pre_files) != len(post_files):
                self.log.emit(f"  ⚠ 层数不等：平扫 {len(pre_files)} 层，增强 {len(post_files)} 层")
                pre_files, post_files = trim_to_common_z_range(pre_files, post_files, self.log.emit)
                if not pre_files or not post_files:
                    finish_msg = "层数对齐失败"
                    return

            # ── 2. 应用预标注 Z 轴裁切 ──────────────────────────────
            az0 = 0
            ann_rois: Optional[AnnotationData] = None

            if self.pre_annotation:
                az0 = max(0, self.pre_annotation.z_range_start)
                az1 = min(len(pre_files) - 1, self.pre_annotation.z_range_end)
                pre_files  = pre_files[az0: az1 + 1]
                post_files = post_files[az0: az1 + 1]

                def _remap_rois(roi_dict, offset):
                    """将 {abs_z: rois} 重映射为 {abs_z - offset: rois}，过滤裁切范围外的层"""
                    return defaultdict(list,
                        {z - offset: rois for z, rois in roi_dict.items() if z >= offset})

                ann_rois = AnnotationData(
                    global_rois=self.pre_annotation.global_rois,
                    local_rois=_remap_rois(self.pre_annotation.local_rois, az0),
                    inverse_global_rois=self.pre_annotation.inverse_global_rois,
                    inverse_local_rois=_remap_rois(self.pre_annotation.inverse_local_rois, az0),
                    metal_rois=_remap_rois(self.pre_annotation.metal_rois, az0),
                    clip_rois=_remap_rois(self.pre_annotation.clip_rois, az0),
                    z_range_start=0,
                    z_range_end=len(pre_files) - 1,
                    scale_factor=self.pre_annotation.scale_factor,
                )
                self.log.emit(f"[裁切] 处理范围：原序列第 {az0+1}~{az1+1} 层，共 {len(pre_files)} 层")

            total_slices = len(pre_files)
            if total_slices == 0:
                finish_msg = "裁切后无可处理层"
                return

            self.log.emit(f"\n[1/6] 逐层读取平扫序列（共 {total_slices} 层）...")
            self.progress.emit(5)

            # 读取第一帧获取几何参数
            ds0       = pydicom.dcmread(pre_files[0])
            h, w      = ds0.pixel_array.shape
            px_sp     = getattr(ds0, 'PixelSpacing', [0.625, 0.625])
            sp_xy     = (float(px_sp[0]), float(px_sp[1]))
            sp_z      = float(getattr(ds0, 'SpacingBetweenSlices',
                              getattr(ds0, 'SliceThickness', 0.625)))
            spacing_zyx = (sp_z, sp_xy[0], sp_xy[1])

            # 整块读入（颈椎段一般 50~120 层，约 50~120 MB，32位下完全安全）
            pre_vol  = np.empty((total_slices, h, w), dtype=np.int16)
            post_vol = np.empty((total_slices, h, w), dtype=np.int16)

            for z, fp in enumerate(pre_files):
                ds = ds0 if z == 0 else pydicom.dcmread(fp)
                arr = ds.pixel_array.astype(np.float32)
                slope     = float(getattr(ds, 'RescaleSlope',     1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                if slope != 1.0 or intercept != 0.0:
                    arr = arr * slope + intercept
                pre_vol[z] = np.clip(arr, -32768, 32767).astype(np.int16)
                if z % 30 == 0:
                    self.progress.emit(5 + int(z / total_slices * 10))
            ds0 = None

            self.log.emit(f"  [内存] 平扫读完：{get_memory_mb():.0f} MB")
            self.log.emit(f"\n[2/6] 逐层读取增强序列...")
            self.progress.emit(15)

            for z, fp in enumerate(post_files):
                ds = pydicom.dcmread(fp)
                arr = ds.pixel_array.astype(np.float32)
                slope     = float(getattr(ds, 'RescaleSlope',     1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                if slope != 1.0 or intercept != 0.0:
                    arr = arr * slope + intercept
                post_vol[z] = np.clip(arr, -32768, 32767).astype(np.int16)
                if z % 30 == 0:
                    self.progress.emit(15 + int(z / total_slices * 10))

            self.log.emit(f"  [内存] 增强读完：{get_memory_mb():.0f} MB")
            if self.cancelled:
                return

            # ── 3. 逐层 FFT 配准：估计位移序列 ───────────────────────
            self.log.emit(f"\n[3/6] 2.5D逐层相位相关配准（{total_slices} 层）...")
            self.progress.emit(25)

            dy_arr = np.zeros(total_slices, dtype=np.float32)
            dx_arr = np.zeros(total_slices, dtype=np.float32)

            for z in range(total_slices):
                if self.cancelled:
                    return
                pre_f  = pre_vol[z].astype(np.float32)
                post_f = post_vol[z].astype(np.float32)
                dy, dx = fft_phase_correlation_2d(pre_f, post_f,
                                                   max_shift=CERV_MAX_SHIFT_PX)
                dy_arr[z] = dy
                dx_arr[z] = dx
                if z % 20 == 0:
                    self.progress.emit(25 + int(z / total_slices * 15))

            # ── 4. Z 轴高斯平滑位移序列（核心约束）────────────────────
            #
            # 直觉：颈椎屈伸是一段连续的曲线运动，相邻3~5层之间位移变化应该平滑。
            # 单层 FFT 估计受噪声影响，可能出现 ±3~5 px 的随机跳变。
            # sigma=1.5 层的高斯核 FWHM ≈ 3.5 层，
            # 足以把单层估计误差压平，同时保留跨10层以上的真实位移趋势。
            #
            # 如果颈椎某段真实存在突变（很少见，除非手术），
            # 平滑后该段位移会被"缓化"，最坏结果是该层配准不完美，
            # 而不是整段垮掉——代价可控。

            self.log.emit(f"  [Z平滑] 对位移序列做高斯平滑（σ={CERV_Z_SMOOTH_SIGMA:.1f} 层）...")
            from scipy.ndimage import gaussian_filter1d
            dy_smooth = gaussian_filter1d(dy_arr, sigma=CERV_Z_SMOOTH_SIGMA)
            dx_smooth = gaussian_filter1d(dx_arr, sigma=CERV_Z_SMOOTH_SIGMA)

            # 记录位移统计，便于医生判断配准质量
            shift_mag = np.sqrt(dy_arr**2 + dx_arr**2)
            self.log.emit(f"  [位移统计] 原始：均值={shift_mag.mean():.1f}px，"
                          f"最大={shift_mag.max():.1f}px，"
                          f">10px层数={int((shift_mag > 10).sum())}/{total_slices}")
            shift_mag_s = np.sqrt(dy_smooth**2 + dx_smooth**2)
            self.log.emit(f"  [位移统计] 平滑后：均值={shift_mag_s.mean():.1f}px，"
                          f"最大={shift_mag_s.max():.1f}px")

            # ── 5. 逐层应用平移 + 减影计算 ───────────────────────────
            self.log.emit(f"\n[4/6] 逐层配准对齐 + 减影计算（颈椎专用参数）...")
            self.progress.emit(40)

            result_vol = np.zeros((total_slices, h, w), dtype=np.int16)

            for z in range(total_slices):
                if self.cancelled:
                    return

                pre_s  = pre_vol[z].astype(np.float32)
                post_s = post_vol[z].astype(np.float32)

                # ── 4a. 应用平滑后的2D平移 ──
                post_aligned = shift_image_2d(post_s,
                                               dy_smooth[z],
                                               dx_smooth[z])

                if pre_s.max() <= -500:
                    # 空气层，直接跳过
                    result_vol[z] = 0
                else:
                    real_diff = np.clip(post_aligned - pre_s, 0, None)

                    # ── 4b. 计算slice cache（质量模式使用精细cache）──
                    # 颈椎段不走解剖分区，固定使用强骨抑制参数
                    # quality_mode 继承自主界面选项
                    cache = precompute_slice_cache(pre_s, real_diff,
                                                   opt.get('quality_mode', False))

                    # ── 4c. 骨抑制：直接调用已有的减影核函数 ──────────
                    # 注意：颈椎专用参数不走 get_zone_adjusted_params_v2，
                    # 直接传入 CERV_BONE_STRENGTH / CERV_VESSEL_SENS，
                    # 等效于 skull_base 区的参数体系。
                    if opt.get('optimization_level', 'none') == 'none':
                        if opt.get('quality_mode', False):
                            res = quality_subtraction_cached(
                                pre_s, post_aligned,
                                CERV_BONE_STRENGTH, CERV_VESSEL_SENS, cache)
                        else:
                            res = fast_subtraction(
                                pre_s, post_aligned,
                                CERV_BONE_STRENGTH, CERV_VESSEL_SENS,
                                body=cache.body, cache=cache)
                    elif opt.get('optimization_level') == 'light':
                        res = optimized_subtraction_light_v3(
                            pre_s, post_aligned,
                            CERV_BONE_STRENGTH, CERV_VESSEL_SENS, cache)
                    elif opt.get('optimization_level') == 'standard':
                        res = optimized_subtraction_standard_v3(
                            pre_s, post_aligned,
                            CERV_BONE_STRENGTH, CERV_VESSEL_SENS, cache,
                            sp_xy, 0.5)
                    else:  # deep
                        res = optimized_subtraction_deep_v3(
                            pre_s, post_aligned,
                            CERV_BONE_STRENGTH, CERV_VESSEL_SENS, cache,
                            sp_xy, spacing_zyx, 0.5)

                    # ── 4d. 颈椎专用增强型骨区抑制 ────────────────────
                    # 颈椎段解剖标志：
                    #   · 无 Willis 环，无岩骨，无眶板
                    #   · 椎体皮质骨（300~600 HU）+ 椎间盘（60~120 HU）
                    #   · 横突孔内椎动脉（贴骨走行，pre_f < 200 但 diff 高）
                    # 颈椎专用骨区字典：只启用 skull_base + mandible + sharp_edge，
                    # 其余（petrous/orbital_plate/convexity_thin）全部留空，
                    # 避免颅底骨区逻辑在颈椎层面误触发。
                    pre_f_local = pre_s  # float32
                    gradient = np.sqrt(
                        ndimage.sobel(pre_f_local, axis=0)**2 +
                        ndimage.sobel(pre_f_local, axis=1)**2
                    )
                    cervical_bone_zones = {
                        'petrous':        np.zeros_like(pre_s, dtype=bool),
                        'skull_base':     (pre_f_local > 250) & (pre_f_local < 1000),
                        'mandible':       pre_f_local > 200,
                        'sharp_edge':     (pre_f_local > 150) & (gradient > 120),
                        'orbital_plate':  np.zeros_like(pre_s, dtype=bool),
                        'convexity_thin': np.zeros_like(pre_s, dtype=bool),
                    }
                    gain = np.ones_like(res, dtype=np.float32)
                    gain = enhanced_bone_suppression_v299(
                        pre_s, real_diff, gain, cervical_bone_zones,
                        CERV_BONE_STRENGTH, CERV_VESSEL_SENS)
                    res = res * gain

                    # ── 4e. 去除体外 + 设备区 ──
                    equip = detect_equipment(pre_s, post_aligned, body=cache.body)
                    res[equip]      = 0
                    res[~cache.body] = 0

                    # ── 4f. 可选后处理 ──
                    if opt.get('clean_bone_edges', True):
                        res = clean_bone_edges_cached(
                            res, pre_s, cache,
                            'quality' if opt.get('quality_mode', False) else 'fast')
                    if CERV_SMOOTH_SIGMA > 0:
                        res = edge_preserving_smooth(res, pre_s, CERV_SMOOTH_SIGMA)

                    # ── 4g. 血管增强 + 微弱血管非线性提亮 ────────────
                    res_scaled = res * CERV_VESSEL_ENHANCE

                    # 微弱血管提亮（与头颅版相同逻辑，区间和增益不变）
                    _WV_LO, _WV_HI, _WV_GAIN = 50.0, 150.0, 1.35
                    _wv_mask = (res_scaled > _WV_LO) & (res_scaled < _WV_HI) & cache.body
                    if _wv_mask.any():
                        _labeled, _n = ndimage.label(_wv_mask)
                        if _n > 0:
                            _sizes = np.bincount(_labeled.ravel(), minlength=_n + 1)
                            _valid = np.where(_sizes >= CERV_MIN_VESSEL_PX)[0]
                            _valid = _valid[_valid > 0]
                            if len(_valid) > 0:
                                _vmask = np.isin(_labeled, _valid)
                                _t = ((res_scaled[_vmask] - _WV_LO) /
                                      (_WV_HI - _WV_LO))
                                res_scaled[_vmask] *= (
                                    1.0 + (_WV_GAIN - 1.0) * 0.5 *
                                    (1.0 - np.cos(_t * np.pi)))

                    result_vol[z] = np.clip(res_scaled, -32768, 32767).astype(np.int16)

                if z % 20 == 0:
                    self.progress.emit(40 + int(z / total_slices * 30))

            del pre_vol, post_vol
            pre_vol = None
            force_gc()
            self.log.emit(f"  [内存] 减影完成：{get_memory_mb():.0f} MB")

            # ── 5. 应用 ROI 掩膜 ─────────────────────────────────────
            if (ann_rois and (
                    ann_rois.global_rois or
                    any(ann_rois.local_rois.values()) or
                    ann_rois.inverse_global_rois or
                    any(ann_rois.inverse_local_rois.values()))):
                self.log.emit("\n[5/6] 应用 ROI 掩膜...")
                result_vol = apply_roi_masks_to_volume(
                    result_vol, ann_rois, 0, total_slices - 1, self.log.emit)
                force_gc()
            else:
                self.log.emit("\n[5/6] 无 ROI，跳过掩膜步骤")

            # 3D 连通清理
            self.log.emit("  [后处理] 3D连通清理...")
            min_area_2d = CERV_MIN_VESSEL_PX * 3
            result_vol = apply_3d_cleanup(result_vol,
                                          min_area=min_area_2d,
                                          log_cb=self.log.emit)
            self.progress.emit(75)
            force_gc()

            # ── 6. 写入 DICOM ────────────────────────────────────────
            self.log.emit(f"\n[6/6] 写入磁盘（{total_slices} 层）...")
            os.makedirs(self.output_dir, exist_ok=True)
            safe_out = _get_short_path(self.output_dir)

            g_min  = float(result_vol.min())
            g_max  = float(result_vol.max())
            g_slope = (g_max - g_min) / 4095.0 if g_max > g_min else 1.0
            series_uid = generate_uid()
            new_frame_uid = generate_uid()

            for z in range(total_slices):
                if self.cancelled:
                    return

                ds = pydicom.dcmread(pre_files[z])
                ds.SpecificCharacterSet = 'ISO_IR 192'

                pix = result_vol[z].astype(np.float32)
                pix_int = (
                    ((pix - g_min) / (g_max - g_min) * 4095)
                    .astype(np.int16)
                    if g_max > g_min
                    else np.zeros_like(pix, dtype=np.int16)
                )
                ds.PixelData = pix_int.tobytes()
                ds.BitsAllocated        = 16
                ds.BitsStored           = 16
                ds.HighBit              = 15
                ds.PixelRepresentation  = 1
                ds.SamplesPerPixel      = 1
                ds.PhotometricInterpretation = 'MONOCHROME2'

                for tag in ['LossyImageCompression',
                             'LossyImageCompressionRatio',
                             'LossyImageCompressionMethod']:
                    if hasattr(ds, tag):
                        delattr(ds, tag)

                ds.RescaleSlope         = g_slope
                ds.RescaleIntercept     = g_min
                ds.WindowCenter         = opt.get('wc', 200)
                ds.WindowWidth          = opt.get('ww', 400)
                ds.SeriesDescription    = "CTA Sub V5.0[颈椎2.5D]"
                ds.SeriesInstanceUID    = series_uid
                ds.ImageType            = ['DERIVED', 'PRIMARY', 'AXIAL']
                ds.FrameOfReferenceUID  = new_frame_uid

                new_sop = generate_uid()
                ds.SOPInstanceUID = new_sop
                ds.file_meta.MediaStorageSOPInstanceUID = new_sop
                ds.InstanceNumber = z + 1
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                ds.save_as(
                    os.path.join(safe_out, f"CERV_SUB_{z+1:04d}.dcm"),
                    write_like_original=False
                )
                if z % 30 == 0:
                    self.progress.emit(75 + int(z / total_slices * 20))

            elapsed = time.time() - t0
            success = True
            finish_msg = (
                f"颈椎减影完成！\n"
                f"输出：{total_slices} 层\n"
                f"耗时：{elapsed:.1f} 秒\n"
                f"目录：{self.output_dir}\n"
                f"配准：2.5D逐层FFT + Z轴σ={CERV_Z_SMOOTH_SIGMA}平滑\n"
                f"最大位移：{shift_mag.max():.1f} px"
            )
            self.log.emit(f"\n{'=' * 55}")
            self.log.emit(f"✅ 颈椎减影完成！耗时：{elapsed:.1f}s")
            self.log.emit(f"[内存] 最终：{get_memory_mb():.0f} MB")

        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            finish_msg = str(e)

        finally:
            if pre_vol is not None:
                del pre_vol
            if result_vol is not None:
                del result_vol
            force_gc()
            self.log.emit(f"[内存] 退出后：{get_memory_mb():.0f} MB")
            if self.cancelled:
                self.finished_signal.emit(False, "⛔ 用户已取消，内存已释放")
            else:
                self.finished_signal.emit(success, finish_msg)


# ════════════════════════════════════════════════════════════════════
# 主界面 UI (未变动内容)
# ════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.all_series, self.selected_pre, self.selected_post, self.recommendations = {}, None, None, None
        self.proc_thread, self._all_presets = None, load_all_presets()
        self.cerv_thread = None          # 颈椎专用减影线程
        self.current_data_dir = ""
        self.annotation_dialog: Optional[InteractiveAnnotationDialog] = None
        self.pre_ann_thread = None
        self.pre_annotation_result: Optional[AnnotationData] = None  
        self.init_ui()
    def init_ui(self):
        self.setWindowTitle("头颅 CTA 减影-whw 版 V5.0")
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
        QComboBox{background:#fff;color:#2c3e50;border:1px solid #ccc;border-radius:3px;padding:3px 6px;}
        QComboBox QAbstractItemView{background:#fff;color:#2c3e50;selection-background-color:#1A6FBF;selection-color:#fff;}
        QSpinBox,QDoubleSpinBox{background:#fff;color:#2c3e50;border:1px solid #ccc;border-radius:3px;padding:2px 4px;}
        QLineEdit{background:#fff;color:#2c3e50;border:1px solid #ccc;border-radius:3px;padding:3px 6px;}
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
        self.copy_data_btn = QPushButton("拷贝数据")
        self.copy_data_btn.setStyleSheet("background:#16a085;color:white;border-radius:4px;padding:6px 16px;")
        self.copy_data_btn.setEnabled(False)
        self.copy_data_btn.setToolTip("将当前配对的平扫+增强序列文件拷贝到指定目录")
        self.copy_data_btn.clicked.connect(self.copy_series_data)
        r2.addWidget(self.copy_data_btn)
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
        self.opt_standard = QRadioButton("标准级 - +局部自适应骨抑制")
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
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(apply_btn)
        preset_row.addWidget(self.preset_status)
        preset_row.addStretch()
        param_lay.addLayout(preset_row)

        chunk_lay = QHBoxLayout()
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 10)
        self.chunk_spin.setValue(1)
        self.chunk_spin.setStyleSheet("QSpinBox { width: 50px; }")
        chunk_lay.addWidget(QLabel("分段减影 (防32位系统内存溢出):"))
        chunk_lay.addWidget(self.chunk_spin)
        chunk_lay.addWidget(QLabel("段"))
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

        # ── 预标注 + 颈椎优化 并排 ──────────────────────────────────
        ann_cerv_lay = QHBoxLayout()
        self.pre_ann_btn = QPushButton("预标注")
        self.pre_ann_btn.setStyleSheet("background:#16a085;color:white;border-radius:4px;padding:6px 16px;")
        self.pre_ann_btn.setEnabled(False)
        self.pre_ann_btn.clicked.connect(self.start_pre_annotation)
        self.pre_ann_status = QLabel("")
        self.pre_ann_status.setStyleSheet("color:#888;font-size:9pt;")
        self.cerv_check = QCheckBox("颈椎优化")
        self.cerv_check.setToolTip(
            "勾选后将启用颈椎专用2.5D配准算法（逐层FFT平移 + Z轴柔性约束），\n"
            "使用颈椎独立参数（骨抑制/血管敏感性），不走头颅解剖分区逻辑。\n"
            "建议配合「预标注」先用Z轴裁切只保留颈椎段再运行。"
        )
        ann_cerv_lay.addWidget(self.pre_ann_btn)
        ann_cerv_lay.addWidget(self.pre_ann_status)
        ann_cerv_lay.addSpacing(24)
        ann_cerv_lay.addWidget(self.cerv_check)
        ann_cerv_lay.addStretch()
        param_lay.addLayout(ann_cerv_lay)

        param_grp.setMinimumHeight(230)
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

        self.log("头颅 CTA 减影-whw 版 V5.0 操作步骤")
        self.log("=" * 50)             
        self.log("1. 扫描目录自动配对")
        self.log("2. 分析文件特征，拷贝数据（可选）")
        self.log("3. 预标注（可选，选层/划ROI，金属伪影画保护区）")
        self.log("4. 开始减影")
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
        self.pre_ann_btn.setEnabled(False)
        self.copy_data_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.pre_annotation_result = None
        self.pre_ann_status.setText("")
        self.pre_ann_status.setStyleSheet("color:#888;font-size:9pt;")
        self.scan_thread = SeriesScanThread(d)
        self.scan_thread.progress.connect(self.progress.setValue)
        self.scan_thread.log.connect(self.log)
        self.scan_thread.finished_signal.connect(self.on_scan_finished)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
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
            total_slices = min(pairs[0][0].file_count, pairs[0][1].file_count)
            if total_slices > 400:
                self.chunk_spin.setValue(2)
                self.log(f"🔔 智能提示：序列长达 {total_slices} 层，已自动建议分为 2 段处理。")
        self.analyze_btn.setEnabled(True)
        self.pre_ann_btn.setEnabled(True)
        self.copy_data_btn.setEnabled(True)
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
        self.param_thread.finished.connect(self.param_thread.deleteLater)
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
            'num_chunks': self.chunk_spin.value(),
        }
        
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        # ── 颈椎专用路由 ──────────────────────────────────────────
        if self.cerv_check.isChecked():
            self.log("🦴 颈椎专用减影模式启动（2.5D逐层FFT + Z轴柔性约束）")
            self.cerv_thread = CervicalProcessThread(
                self.selected_pre, self.selected_post, output_dir, opt,
                pre_annotation=self.pre_annotation_result
            )
            self.cerv_thread.progress.connect(self.progress.setValue)
            self.cerv_thread.log.connect(self.log)
            self.cerv_thread.finished_signal.connect(self.on_finished)
            self.cerv_thread.finished.connect(self.cerv_thread.deleteLater)
            self.cerv_thread.start()
            return

        # ── 常规头颅减影路由（原有逻辑不变）──────────────────────
        self.proc_thread = ProcessThread(
            self.selected_pre, self.selected_post, output_dir, opt,
            pre_annotation=self.pre_annotation_result
        )
        self.proc_thread.progress.connect(self.progress.setValue)
        self.proc_thread.log.connect(self.log)
        self.proc_thread.finished_signal.connect(self.on_finished)
        self.proc_thread.finished.connect(self.proc_thread.deleteLater)
        self.proc_thread.start()

    def start_pre_annotation(self):
        if not self.post_combo.currentData():
            return QMessageBox.warning(self, "提示", "请先选择增强序列")
        self.selected_pre  = self.pre_combo.currentData()
        self.selected_post = self.post_combo.currentData()
        self.pre_ann_btn.setEnabled(False)
        self.pre_ann_status.setText("读取造影序列中...")
        self.pre_ann_status.setStyleSheet("color:#f39c12;font-size:9pt;")
        self.pre_ann_thread = PreAnnotationThread(self.selected_post)
        self.pre_ann_thread.progress.connect(self.progress.setValue)
        self.pre_ann_thread.log.connect(self.log)
        self.pre_ann_thread.ready.connect(self.on_pre_ann_ready)
        self.pre_ann_thread.error.connect(self.on_pre_ann_error)
        self.pre_ann_thread.finished.connect(self.pre_ann_thread.deleteLater)
        self.pre_ann_thread.start()

    def on_pre_ann_ready(self, post_vol: np.ndarray, post_files: list):
        self.progress.setValue(0)
        self.log(f"⏸ 打开预标注窗口（造影原片，共 {post_vol.shape[0]} 层）...")
        dlg = InteractiveAnnotationDialog(post_vol, self)
        dlg.setWindowTitle("预标注 - 造影序列（选择减影范围 / 可选划ROI）")
        result = dlg.exec_()

        dlg.canvas.display_volume = None
        dlg.display_volume = None
        del post_vol
        force_gc()
        self.log(f"  [内存] 预标注窗口关闭后：{get_memory_mb():.0f} MB")

        if result == QDialog.Accepted and dlg.result_accepted:
            self.pre_annotation_result = dlg.get_scaled_annotations()
            z0 = self.pre_annotation_result.z_range_start + 1
            z1 = self.pre_annotation_result.z_range_end   + 1
            n  = z1 - z0 + 1
            has_roi = bool(self.pre_annotation_result.global_rois or
                           any(self.pre_annotation_result.local_rois.values()) or
                           self.pre_annotation_result.inverse_global_rois or
                           any(self.pre_annotation_result.inverse_local_rois.values()) or
                           any(self.pre_annotation_result.metal_rois.values()))
            roi_tip = "含ROI" if has_roi else "无ROI"
            wt = self.pre_annotation_result.willis_layer_top
            wb = self.pre_annotation_result.willis_layer_bot
            willis_tip = (f"，Willis第{wb+1}~{wt+1}层"
                          if wt is not None and wb is not None else "")
            self.pre_ann_status.setText(
                f"✓ 已标注：第{z0}~{z1}层（{n}层，{roi_tip}{willis_tip}）")
            self.pre_ann_status.setStyleSheet("color:#27ae60;font-size:9pt;font-weight:bold;")
            self.log(f"✅ 预标注完成：选定第 {z0}~{z1} 层，共 {n} 层，{roi_tip}{willis_tip}")
            self.start_btn.setEnabled(True)
        else:
            self.pre_annotation_result = None
            self.pre_ann_btn.blockSignals(True)
            self.pre_ann_btn.setChecked(False)
            self.pre_ann_btn.blockSignals(False)
            self.pre_ann_status.setText("")
            self.pre_ann_status.setStyleSheet("color:#888;font-size:9pt;")
            self.log("预标注已取消")
            self.start_btn.setEnabled(True)

        self.pre_ann_btn.setEnabled(True)
        dlg.deleteLater()

    def on_pre_ann_error(self, msg: str):
        self.progress.setValue(0)
        self.pre_ann_btn.setEnabled(True)
        self.pre_ann_status.setText("读取失败")
        self.pre_ann_status.setStyleSheet("color:#e74c3c;font-size:9pt;")
        QMessageBox.warning(self, "错误", f"预标注读取失败：{msg}")

    def copy_series_data(self):
        pre_s  = self.pre_combo.currentData()
        post_s = self.post_combo.currentData()
        if not pre_s or not post_s:
            QMessageBox.warning(self, "提示", "请先扫描并选择序列")
            return
        if pre_s == post_s:
            QMessageBox.warning(self, "提示", "平扫和增强序列相同，请重新选择")
            return

        dest_dir = QFileDialog.getExistingDirectory(self, "选择拷贝目标目录")
        if not dest_dir:
            return

        def safe_name(s):
            raw = f"S{s.series_number:03d}_{s.series_description[:30]}"
            return re.sub(r'[\\/:*?"<>|]', '_', raw).strip()

        pre_dest  = os.path.join(dest_dir, safe_name(pre_s))
        post_dest = os.path.join(dest_dir, safe_name(post_s))

        pre_files  =[f[1] for f in pre_s.files]
        post_files = [f[1] for f in post_s.files]
        total = len(pre_files) + len(post_files)

        self.copy_data_btn.setEnabled(False)
        self.log(f"📋 开始拷贝：共 {total} 个文件 → {dest_dir}")

        import shutil

        class CopyThread(QThread):
            progress_sig = pyqtSignal(int)
            log_sig      = pyqtSignal(str)
            done_sig     = pyqtSignal(bool, str)

            def __init__(self, jobs):
                super().__init__()
                self.jobs = jobs

            def run(self):
                done = 0
                try:
                    for src, dst_dir in self.jobs:
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.copy2(src, dst_dir)
                        done += 1
                        if done % 30 == 0:
                            self.progress_sig.emit(int(done / len(self.jobs) * 100))
                    self.progress_sig.emit(100)
                    self.done_sig.emit(True, f"拷贝完成：{done} 个文件")
                except Exception as e:
                    self.done_sig.emit(False, f"拷贝失败：{e}")

        jobs =[(f, pre_dest) for f in pre_files] +[(f, post_dest) for f in post_files]
        self._copy_thread = CopyThread(jobs)
        self._copy_thread.progress_sig.connect(self.progress.setValue)
        self._copy_thread.log_sig.connect(self.log)

        def on_copy_done(success, msg):
            self.progress.setValue(0)
            self.copy_data_btn.setEnabled(True)
            self.log(("✅ " if success else "❌ ") + msg)
            if success:
                try:
                    import subprocess
                    subprocess.Popen(["explorer", os.path.normpath(dest_dir)])
                except Exception:
                    pass
            else:
                QMessageBox.warning(self, "拷贝失败", msg)

        self._copy_thread.done_sig.connect(on_copy_done)
        self._copy_thread.finished.connect(self._copy_thread.deleteLater)
        self._copy_thread.start()

    def cancel(self):
        if self.proc_thread:
            self.proc_thread.cancel()
            self.log("⛔ 取消指令已发出，等待当前步骤结束后释放内存...")
            self.cancel_btn.setEnabled(False)
        if self.cerv_thread:
            self.cerv_thread.cancel()
            self.log("⛔ 颈椎减影取消指令已发出...")
            self.cancel_btn.setEnabled(False)

    def on_finished(self, success, msg):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(0)
        # 不直接置 None（会立即触发 C++ 析构），改为 deleteLater()，
        # 让 Qt 在下一个事件循环周期安全销毁，此时线程必然已完全退出。
        if self.proc_thread is not None:
            self.proc_thread = None
        if self.cerv_thread is not None:
            self.cerv_thread = None
        self.update_output_dir_preview()
        if msg.startswith("⛔"):
            self.log(msg)
        elif success:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "错误", msg)

# ════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    if hasattr(sys.stderr, "reconfigure"):
        try: sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    if getattr(sys, 'frozen', False):
        import tempfile
        _safe_cwd = tempfile.gettempdir()
        try:
            os.chdir(_safe_cwd)
        except Exception:
            pass
        os.environ.setdefault("ITK_GLOBAL_DEFAULT_THREADER", "Platform")

    if os.name == 'nt':
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("whw.cta.subtraction.331")
        except Exception:
            pass
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
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