"""
头颅 CT 血肿体积自动计算工具 (极限优化 + 临床多田公式融合版 + 图层与时序完美修复版)
特点：
1. 包含魔棒、层数防截断、防漏墙、橡皮擦(减)、画笔(增) 五大全链路微调工具。
2. 后台自动提取并计算多田公式 (ABC/2)，零交互同步比对。
3. 强制通过 ImagePositionPatient 绝对空间坐标抓取真实层厚。
4. 【修复】层面截断时体积闪烁的 Bug。
5. 【修复】橡皮擦抗泛洪失效的 Bug，橡皮图层现可完美阻挡后续魔棒。
"""
import os
import sys
import math
import numpy as np
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
import pydicom
import ctypes
import scipy.ndimage as ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

# ==========================================
# 引入 Shapely 库 (用于多边形融合)
# ==========================================
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("提示：未检测到 shapely 库。pip install shapely 可启用多边形智能融合功能。")

def _get_short_path(path: str) -> str:
    if os.name != 'nt': return path
    try:
        if not os.path.exists(path): return path
        buf = ctypes.create_unicode_buffer(1024)
        ret = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024)
        if ret == 0: return path
        short = buf.value
        if short and os.path.exists(short): return short
        return path
    except Exception: return path

def _sitk_read_image_safe(file_list: list, pixel_type=None):
    import SimpleITK as sitk
    if pixel_type is None: pixel_type = sitk.sitkInt16
    safe_files =[_get_short_path(f) for f in file_list]
    try:
        return sitk.ReadImage(safe_files, pixel_type)
    except Exception as e1:
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
                if ds0 is None: ds0 = ds
            vol = np.stack(slices, axis=0)          
            img = sitk.GetImageFromArray(vol)        
            try:
                px_spacing = ds0.PixelSpacing          
                row_sp = float(px_spacing[0])
                col_sp = float(px_spacing[1])
                z_raw = getattr(ds0, 'SpacingBetweenSlices', getattr(ds0, 'SliceThickness', None))
                z_sp = float(z_raw) if z_raw is not None and str(z_raw).strip() != "" else 1.0
                img.SetSpacing((col_sp, row_sp, z_sp))   
            except Exception: pass  
            return img
        except Exception as e2:
            raise RuntimeError(f"SimpleITK和pydicom均无法读取该序列。\n错误信息: {e2}")

try:
    import SimpleITK as sitk
except ImportError:
    print("请安装 SimpleITK: pip install SimpleITK")
    sys.exit(1)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QGroupBox, QMessageBox, QSpinBox, QSlider, QButtonGroup, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QPolygonF, QFont

# ==========================================
# 几何计算工具
# ==========================================
def point_to_line_distance(p: QPointF, a: QPointF, b: QPointF) -> Tuple[float, float]:
    dx, dy = b.x() - a.x(), b.y() - a.y()
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        return math.sqrt((p.x() - a.x())**2 + (p.y() - a.y())**2), 0.0
    t = max(0, min(1, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / length_sq))
    proj_x, proj_y = a.x() + t * dx, a.y() + t * dy
    return math.sqrt((p.x() - proj_x)**2 + (p.y() - proj_y)**2), t

# ==========================================
# ROI 数据结构
# ==========================================
@dataclass
class PolygonROI:
    points: List[QPointF] = field(default_factory=list)
    closed: bool = False

    def copy(self) -> 'PolygonROI':
        return PolygonROI(points=[QPointF(p.x(), p.y()) for p in self.points], closed=self.closed)

    def to_shapely(self):
        if not HAS_SHAPELY or len(self.points) < 3: return None
        coords =[(p.x(), p.y()) for p in self.points]
        try:
            poly = ShapelyPolygon(coords)
            if not poly.is_valid: poly = make_valid(poly)
            return poly if poly.is_valid and not poly.is_empty else None
        except: return None

    @staticmethod
    def from_shapely(shapely_poly) -> Optional['PolygonROI']:
        if shapely_poly is None or shapely_poly.is_empty: return None
        try:
            if shapely_poly.geom_type != 'Polygon': return None 
            coords = list(shapely_poly.exterior.coords)[:-1]
            if len(coords) < 3: return None
            return PolygonROI(points=[QPointF(x, y) for x, y in coords], closed=True)
        except: return None

def merge_overlapping_polygons(rois: List[PolygonROI]) -> List[PolygonROI]:
    if not HAS_SHAPELY or len(rois) <= 1: return rois
    shapely_polys =[]
    for roi in rois:
        sp = roi.to_shapely()
        if sp is not None: shapely_polys.append(sp)

    if len(shapely_polys) <= 1: return rois
    try:
        merged = unary_union(shapely_polys)
    except Exception:
        return rois

    result =[]
    if merged.is_empty: return rois
    if merged.geom_type == 'Polygon':
        roi = PolygonROI.from_shapely(merged)
        if roi: result.append(roi)
    elif merged.geom_type == 'MultiPolygon':
        for poly in merged.geoms:
            roi = PolygonROI.from_shapely(poly)
            if roi: result.append(roi)
    return result if result else rois

@dataclass
class DrawingState:
    is_drawing: bool = False
    start_slice: int = 0
    current_slice: int = 0
    visited_slices: Set[int] = field(default_factory=set)
    vertex_slices: Set[int] = field(default_factory=set)
    polygon: Optional[PolygonROI] = None

    def reset(self):
        self.is_drawing = False
        self.start_slice = 0
        self.current_slice = 0
        self.visited_slices.clear()
        self.vertex_slices.clear()
        self.polygon = None

    def get_slice_range(self) -> Tuple[int, int]:
        ref = self.vertex_slices if self.vertex_slices else self.visited_slices
        if not ref: return (self.start_slice, self.start_slice)
        return (min(ref), max(ref))

    def get_slice_count(self) -> int:
        min_z, max_z = self.get_slice_range()
        return max_z - min_z + 1

def generate_roi_mask_from_polygon(roi: PolygonROI, shape: Tuple[int, int]) -> np.ndarray:
    if not roi.closed or len(roi.points) < 3: return np.zeros(shape, dtype=bool)
    h, w = shape
    qimg = QImage(w, h, QImage.Format_Grayscale8)
    qimg.fill(0)
    painter = QPainter(qimg)
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    painter.setPen(Qt.NoPen)
    painter.drawPolygon(QPolygonF(roi.points))
    painter.end()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w))
    return arr > 127

# ==========================================
# DICOM 扫描工具
# ==========================================
class SeriesInfo:
    def __init__(self):
        self.series_uid = ""
        self.series_description = ""
        self.series_number = 0
        self.file_count = 0
        self.files =[]

def analyze_dicom_file(filepath):
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        return {
            'filepath': filepath,
            'series_uid': str(getattr(ds, 'SeriesInstanceUID', '')),
            'series_description': str(getattr(ds, 'SeriesDescription', '')),
            'series_number': int(getattr(ds, 'SeriesNumber', 0)),
            'instance_number': int(getattr(ds, 'InstanceNumber', 0)),
        }
    except: return None

class ScanThread(QThread):
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def run(self):
        dicom_files =[]
        for root, _, files in os.walk(self.directory):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as fp:
                        fp.seek(128)
                        if fp.read(4) == b'DICM': dicom_files.append(filepath)
                except: pass
        
        series_dict = defaultdict(SeriesInfo)
        total = len(dicom_files)
        if total == 0:
            self.finished_signal.emit({})
            return

        for i, f in enumerate(dicom_files):
            info = analyze_dicom_file(f)
            if info and info['series_uid']:
                s = series_dict[info['series_uid']]
                if s.file_count == 0:
                    s.series_uid, s.series_description, s.series_number = info['series_uid'], info['series_description'], info['series_number']
                s.files.append((info['instance_number'], f))
                s.file_count += 1
            if i % 10 == 0: self.progress.emit(int((i / total) * 100))
        
        for s in series_dict.values(): s.files.sort(key=lambda x: x[0])
        self.progress.emit(100)
        self.finished_signal.emit(series_dict)

# ==========================================
# 交互式双屏画布
# ==========================================
class HematomaCanvas(QWidget):
    volume_changed = pyqtSignal(float, float, int, float, float, float)
    status_msg = pyqtSignal(str)
    slice_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#1a1a1a;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.volume = None
        self.mask = None
        self.spacing_zyx = (1.0, 1.0, 1.0)
        
        # 多层独立遮罩
        self.base_mask = None
        self.current_wand_mask = None
        self.erase_mask = None
        
        self.undo_mask = None
        self.undo_base_mask = None
        self.undo_current_wand_mask = None
        self.undo_erase_mask = None
        
        self.current_z = 0
        self.depth, self.img_height, self.img_width = 0, 0, 0
        self.wc, self.ww = 40, 120
        
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)
        
        self.mode = "navigate"
        self.is_panning = False
        self.is_zooming = False 
        self.is_windowing = False
        self.last_mouse_pos = QPointF()
        self.current_mouse_pos = QPointF()
        
        self.is_erasing = False
        self.is_brushing = False
        self.eraser_radius = 5
        self.last_erase_pos = None
        self.last_brush_pos = None

        self.rois = defaultdict(list)
        self.drawing_state = DrawingState()
        self.current_polygon = None
        
        self.selected_roi = None
        self.selected_vertex_idx = -1
        self.hover_vertex_idx = -1
        self.hover_roi = None
        self.is_moving_vertex = False
        
        self.display_pixmap_masked = None
        self.display_pixmap_raw = None
        
        self.magic_wand_lower = 40
        self.magic_wand_upper = 95
        self.last_seed = None 
        self.undo_last_seed = None
        
        # 层面范围限制
        self.limit_z_min = 0
        self.limit_z_max = 9999

    def enterEvent(self, event):
        self.setFocus()
        super().enterEvent(event)

    def set_volume(self, volume_array, spacing_zyx):
        self.volume = volume_array
        self.spacing_zyx = spacing_zyx
        self.depth, self.img_height, self.img_width = self.volume.shape
        
        shape = self.volume.shape
        self.mask = np.zeros(shape, dtype=bool)
        
        self.base_mask = np.zeros(shape, dtype=bool)
        self.current_wand_mask = np.zeros(shape, dtype=bool)
        self.erase_mask = np.zeros(shape, dtype=bool)
        
        self.undo_mask = np.zeros(shape, dtype=bool)
        self.undo_base_mask = np.zeros(shape, dtype=bool)
        self.undo_current_wand_mask = np.zeros(shape, dtype=bool)
        self.undo_erase_mask = np.zeros(shape, dtype=bool)
        
        self.rois.clear()
        self.drawing_state.reset()
        self._clear_edit_state()
        
        self.last_seed = None
        self.undo_last_seed = None
        
        self.limit_z_min = 0
        self.limit_z_max = self.depth - 1
        self.current_z = self.depth // 2
        
        self._fit_to_window()
        self._update_pixmap()
        self.update()

    def _save_undo_state(self):
        if self.mask is not None:
            np.copyto(self.undo_mask, self.mask)
            np.copyto(self.undo_base_mask, self.base_mask)
            np.copyto(self.undo_current_wand_mask, self.current_wand_mask)
            np.copyto(self.undo_erase_mask, self.erase_mask)
            self.undo_last_seed = self.last_seed

    def _fit_to_window(self):
        if self.volume is None: return
        w, h = self.width() / 2.0, self.height()
        if w > 0 and h > 0:
            self.zoom = min(w / self.img_width, h / self.img_height, 2.0)
            self.pan_offset = QPointF((w - self.img_width * self.zoom) / 2, (h - self.img_height * self.zoom) / 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_to_window()
        self.update()

    def set_slice(self, z):
        if self.volume is None: return
        new_z = max(0, min(z, self.depth - 1))
        
        if new_z != self.current_z:
            self.current_z = new_z
            if self.drawing_state.is_drawing:
                self.drawing_state.current_slice = self.current_z
                self.drawing_state.visited_slices.add(self.current_z)
                self._emit_drawing_status()
            else:
                self._clear_edit_state()
                
            self._update_pixmap()
            self.update()
            self.slice_changed.emit(self.current_z)

    def set_mode(self, mode):
        self.mode = mode
        cursors = {
            "navigate": Qt.OpenHandCursor,
            "window": Qt.SizeAllCursor,
            "magic_wand": Qt.CrossCursor,
            "draw_roi": Qt.CrossCursor,
            "move_vertex": Qt.PointingHandCursor,
            "add_vertex": Qt.CrossCursor,
            "delete_vertex": Qt.ForbiddenCursor,
            "eraser": Qt.BlankCursor,
            "brush": Qt.BlankCursor 
        }
        self.setCursor(cursors.get(mode, Qt.ArrowCursor))
        
        if mode not in["draw_roi", "move_vertex", "add_vertex", "delete_vertex"]:
            self.cancel_drawing()
            self._clear_edit_state()

    def _clear_edit_state(self):
        self.selected_roi = None
        self.selected_vertex_idx = -1
        self.hover_vertex_idx = -1
        self.hover_roi = None
        self.is_moving_vertex = False

    def _update_pixmap(self):
        if self.volume is None: return
        data = self.volume[self.current_z].astype(np.float32)
        vmin, vmax = self.wc - self.ww / 2, self.wc + self.ww / 2
        norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        
        rgb_raw = np.empty((self.img_height, self.img_width, 3), dtype=np.uint8)
        rgb_raw[..., 0] = gray
        rgb_raw[..., 1] = gray
        rgb_raw[..., 2] = gray
        qimg_raw = QImage(rgb_raw.tobytes(), self.img_width, self.img_height, self.img_width * 3, QImage.Format_RGB888)
        self.display_pixmap_raw = QPixmap.fromImage(qimg_raw)
        
        rgb_masked = rgb_raw.copy()
        
        # 截断优先级隔离显示
        if self.limit_z_min <= self.current_z <= self.limit_z_max:
            slice_mask = self.mask[self.current_z]
            if slice_mask.any():
                rgb_masked[slice_mask, 0] = np.clip(rgb_masked[slice_mask, 0] * 0.5 + 255 * 0.5, 0, 255).astype(np.uint8)
                rgb_masked[slice_mask, 1] = np.clip(rgb_masked[slice_mask, 1] * 0.5, 0, 255).astype(np.uint8)
                rgb_masked[slice_mask, 2] = np.clip(rgb_masked[slice_mask, 2] * 0.5, 0, 255).astype(np.uint8)

        qimg_masked = QImage(rgb_masked.tobytes(), self.img_width, self.img_height, self.img_width * 3, QImage.Format_RGB888)
        self.display_pixmap_masked = QPixmap.fromImage(qimg_masked)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform) 
        painter.fillRect(self.rect(), QColor(26, 26, 26))
        
        if not self.display_pixmap_masked: 
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待加载序列...")
            return
            
        w_half = self.width() / 2.0
        
        painter.save()
        painter.setClipRect(0, 0, int(w_half), self.height())
        painter.translate(self.pan_offset)
        painter.scale(self.zoom, self.zoom)
        painter.drawPixmap(0, 0, self.display_pixmap_masked)
        self._draw_overlays(painter, is_reference=False)
        painter.restore()

        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.drawLine(int(w_half), 0, int(w_half), self.height())

        painter.save()
        painter.setClipRect(int(w_half), 0, int(w_half), self.height())
        painter.translate(w_half, 0)
        painter.translate(self.pan_offset)
        painter.scale(self.zoom, self.zoom)
        painter.drawPixmap(0, 0, self.display_pixmap_raw)
        self._draw_overlays(painter, is_reference=True)
        painter.restore()

        if self.mode == "window":
            painter.setPen(QColor(255, 255, 255, 180))
            painter.drawText(20, 30, f"窗位:{self.wc} 窗宽:{self.ww}")
            painter.drawText(int(w_half) + 20, 30, f"窗位:{self.wc} 窗宽:{self.ww}")
            
        if not (self.limit_z_min <= self.current_z <= self.limit_z_max):
            painter.setPen(QColor(231, 76, 60, 220))
            painter.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            painter.drawText(20, self.height() - 35, "⚠ 越界：此层位于血肿截断范围外，已排除提取与计算")
            painter.drawText(int(w_half) + 20, self.height() - 35, "⚠ 越界：此层位于血肿截断范围外，已排除提取与计算")
            
        if self.drawing_state.is_drawing:
            self._draw_cross_slice_indicator(painter)
            
    def _draw_overlays(self, painter, is_reference=False):
        if self.current_z in self.rois:
            for roi in self.rois[self.current_z]:
                is_selected_roi = (roi == self.selected_roi)
                is_hover_roi = (roi == self.hover_roi)
                alpha = 80 if is_reference else 200
                self._draw_polygon(painter, roi, QColor(80, 255, 80, alpha), selected=is_selected_roi, hover=is_hover_roi, is_reference=is_reference)
                
        if self.current_polygon and self.current_polygon.points:
            color = QColor(255, 180, 0, 220) if self.drawing_state.get_slice_count() > 1 else QColor(255, 255, 0, 220)
            self._draw_polygon(painter, self.current_polygon, color, drawing=True, is_reference=is_reference)

        if self.mode in ("eraser", "brush") and self.current_mouse_pos:
            r = self.eraser_radius
            if self.mode == "eraser":
                painter.setPen(QPen(QColor(255, 50, 50, 200), 1.5 / self.zoom, Qt.DashLine))
                painter.setBrush(QBrush(QColor(255, 50, 50, 40)))
            else:
                painter.setPen(QPen(QColor(50, 255, 50, 200), 1.5 / self.zoom, Qt.DashLine))
                painter.setBrush(QBrush(QColor(50, 255, 50, 40)))
            painter.drawEllipse(self.current_mouse_pos, r, r)

        if self.last_seed is not None and self.last_seed[2] == self.current_z:
            sx, sy = self.last_seed[0], self.last_seed[1]
            painter.setPen(QPen(QColor(0, 255, 255), 2.0 / self.zoom))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(sx, sy), 3.0 / self.zoom, 3.0 / self.zoom)
            painter.drawLine(QPointF(sx - 6.0 / self.zoom, sy), QPointF(sx + 6.0 / self.zoom, sy))
            painter.drawLine(QPointF(sx, sy - 6.0 / self.zoom), QPointF(sx, sy + 6.0 / self.zoom))

    def _draw_polygon(self, painter, roi, color, selected=False, hover=False, drawing=False, is_reference=False):
        if not roi.points: return
        if selected: color = QColor(255, 200, 50, 230)
        elif hover and self.mode in["add_vertex", "move_vertex", "delete_vertex"]: color = QColor(color.red(), color.green(), color.blue(), 240)
        
        pen_w = 3.0 / self.zoom if selected else 2.0 / self.zoom
        painter.setPen(QPen(color, pen_w, Qt.SolidLine))
        
        pts = roi.points
        for i in range(len(pts) - 1): painter.drawLine(pts[i], pts[i + 1])
        
        if drawing:
            painter.setPen(QPen(color, pen_w, Qt.DashLine))
            painter.drawLine(pts[-1], self.current_mouse_pos)
            if len(pts) >= 2: painter.drawLine(self.current_mouse_pos, pts[0])
            
        if roi.closed and len(pts) >= 3:
            painter.setPen(QPen(color, pen_w, Qt.SolidLine))
            painter.drawLine(pts[-1], pts[0])
            fill_alpha = min(30, color.alpha())
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), fill_alpha)))
            painter.drawPolygon(QPolygonF(pts))

        if not is_reference and not drawing and self.mode in["move_vertex", "add_vertex", "delete_vertex", "draw_roi"]:
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
                else:
                    painter.setBrush(color)
                    painter.drawEllipse(p, vs, vs)

    def _draw_cross_slice_indicator(self, painter):
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
            painter.drawText(rect_x + 10, rect_y + 22, "跨层框选绘制中")
            painter.setFont(QFont("Microsoft YaHei", 9))
            painter.setPen(QColor(255, 180, 0))
            painter.drawText(rect_x + 10, rect_y + 42, f"层 {min_z + 1} → {max_z + 1} (共 {slice_count} 层)")
        else:
            painter.drawText(rect_x + 10, rect_y + 25, f"绘制中 - 第 {state.start_slice + 1} 层")

    def _get_image_pos(self, screen_pos: QPointF) -> QPointF:
        x = screen_pos.x()
        if x >= self.width() / 2.0:
            x -= self.width() / 2.0
        return QPointF((x - self.pan_offset.x()) / self.zoom, (screen_pos.y() - self.pan_offset.y()) / self.zoom)

    def _find_vertex_at(self, img_pos: QPointF) -> Tuple[Optional[PolygonROI], int]:
        th = 8.0 / self.zoom
        for roi in self.rois.get(self.current_z,[]):
            if roi.closed:
                min_dist, best_idx = float('inf'), -1
                for i, p in enumerate(roi.points):
                    dist = math.sqrt((img_pos.x() - p.x())**2 + (img_pos.y() - p.y())**2)
                    if dist < min_dist:
                        min_dist, best_idx = dist, i
                if min_dist <= th: return roi, best_idx
        return None, -1

    def _find_edge_at(self, img_pos: QPointF) -> Optional[PolygonROI]:
        th = 6.0 / self.zoom
        for roi in self.rois.get(self.current_z,[]):
            if roi.closed and len(roi.points) >= 3:
                min_dist, best_edge = float('inf'), -1
                n = len(roi.points)
                for i in range(n):
                    a, b = roi.points[i], roi.points[(i + 1) % n]
                    dist, _ = point_to_line_distance(img_pos, a, b)
                    if dist < min_dist:
                        min_dist, best_edge = dist, i
                if min_dist <= th: return roi, best_edge
        return None, -1

    def mousePressEvent(self, event):
        if not self.volume is None:
            spos = QPointF(event.pos())
            qpos = self._get_image_pos(spos)
            ix, iy = int(qpos.x()), int(qpos.y())
            
            if self.mode == "navigate":
                if event.button() == Qt.LeftButton:
                    self.is_panning = True
                    self.last_mouse_pos = spos  
                    self.setCursor(Qt.ClosedHandCursor)
                elif event.button() == Qt.RightButton: 
                    self.is_zooming = True
                    self.last_mouse_pos = spos
                    self.setCursor(Qt.SizeVerCursor)
            
            elif self.mode == "window" and event.button() == Qt.LeftButton:
                self.is_windowing = True
                self.last_mouse_pos = spos
            
            elif self.mode == "magic_wand" and event.button() == Qt.LeftButton:
                if 0 <= ix < self.img_width and 0 <= iy < self.img_height:
                    self._execute_magic_wand(ix, iy, self.current_z, is_update=False)
            
            elif self.mode == "eraser" and event.button() == Qt.LeftButton:
                self._save_undo_state()
                self.is_erasing = True
                self.last_erase_pos = qpos
                self._apply_eraser_brush(qpos)
                
            elif self.mode == "brush" and event.button() == Qt.LeftButton:
                self._save_undo_state()
                self.is_brushing = True
                self.last_brush_pos = qpos
                self._apply_brush_action(qpos)
                    
            elif self.mode == "draw_roi":
                if event.button() == Qt.LeftButton and (0 <= ix < self.img_width and 0 <= iy < self.img_height):
                    if not self.current_polygon:
                        self.current_polygon = PolygonROI()
                        self.drawing_state.is_drawing = True
                        self.drawing_state.start_slice = self.current_z
                        self.drawing_state.current_slice = self.current_z
                        self.drawing_state.visited_slices = {self.current_z}
                        self.drawing_state.vertex_slices = {self.current_z}
                        self.drawing_state.polygon = self.current_polygon
                        self._emit_drawing_status()
                    
                    self.current_polygon.points.append(qpos)
                    self.drawing_state.vertex_slices.add(self.current_z)
                    self.update()
                    
                elif event.button() == Qt.RightButton:
                    if self.current_polygon and len(self.current_polygon.points) >= 3:
                        self._save_current_polygon_cross_slice()

            elif self.mode == "move_vertex" and event.button() == Qt.LeftButton:
                roi, idx = self._find_vertex_at(qpos)
                if roi and idx >= 0:
                    self.selected_roi, self.selected_vertex_idx = roi, idx
                    self.is_moving_vertex = True
                    self.setCursor(Qt.ClosedHandCursor)
                    self.update()

            elif self.mode == "delete_vertex" and event.button() == Qt.LeftButton:
                roi, idx = self._find_vertex_at(qpos)
                if roi and idx >= 0:
                    if len(roi.points) > 3:
                        roi.points.pop(idx)
                        self._clear_edit_state()
                        self._try_merge_rois()
                        self._recalculate_last_seed() 
                        self.update()
                    else:
                        self.status_msg.emit("提示：多边形至少需要 3 个顶点")

            elif self.mode == "add_vertex" and event.button() == Qt.LeftButton:
                roi, edge_idx = self._find_edge_at(qpos)
                if roi and edge_idx >= 0:
                    roi.points.insert(edge_idx + 1, qpos)
                    self._try_merge_rois()
                    self._recalculate_last_seed() 
                    self.update()

    def mouseMoveEvent(self, event):
        spos = QPointF(event.pos())
        self.current_mouse_pos = self._get_image_pos(spos)
        qpos = self.current_mouse_pos
        
        if getattr(self, 'is_panning', False):
            self.pan_offset += spos - self.last_mouse_pos
            self.last_mouse_pos = spos
            self.update()
            
        elif getattr(self, 'is_zooming', False): 
            dy = spos.y() - self.last_mouse_pos.y()
            zoom_factor = math.exp(-dy * 0.01) 
            new_zoom = max(0.1, min(self.zoom * zoom_factor, 20.0))
            
            img_pos = self._get_image_pos(self.last_mouse_pos)
            screen_x_relative = self.last_mouse_pos.x()
            if screen_x_relative >= self.width() / 2.0:
                screen_x_relative -= self.width() / 2.0
                
            self.zoom = new_zoom
            self.pan_offset.setX(screen_x_relative - img_pos.x() * self.zoom)
            self.pan_offset.setY(self.last_mouse_pos.y() - img_pos.y() * self.zoom)
            
            self.last_mouse_pos = spos
            self.update()
            
        elif getattr(self, 'is_windowing', False):
            dx, dy = spos.x() - self.last_mouse_pos.x(), spos.y() - self.last_mouse_pos.y()
            self.wc = int(max(-100, min(500, self.wc - dy)))
            self.ww = int(max(1, min(1000, self.ww + dx)))
            self.last_mouse_pos = spos
            self._update_pixmap()
            self.update()
            
        elif self.mode == "eraser":
            if getattr(self, 'is_erasing', False):
                self._apply_eraser_line(self.last_erase_pos, qpos)
                self.last_erase_pos = qpos
            self.update()
            
        elif self.mode == "brush":
            if getattr(self, 'is_brushing', False):
                self._apply_brush_line(self.last_brush_pos, qpos)
                self.last_brush_pos = qpos
            self.update()
            
        elif self.mode == "draw_roi" and getattr(self, 'current_polygon', None):
            self.update()
            
        elif self.mode == "move_vertex":
            if getattr(self, 'is_moving_vertex', False) and self.selected_roi and self.selected_vertex_idx >= 0:
                self.selected_roi.points[self.selected_vertex_idx] = qpos
                self.update()
            else:
                roi, idx = self._find_vertex_at(qpos)
                if roi != self.hover_roi or idx != self.hover_vertex_idx:
                    self.hover_roi, self.hover_vertex_idx = roi, idx
                    self.update()
        elif self.mode == "delete_vertex":
            roi, idx = self._find_vertex_at(qpos)
            if roi != getattr(self, 'hover_roi', None) or idx != getattr(self, 'hover_vertex_idx', -1):
                self.hover_roi, self.hover_vertex_idx = roi, idx
                self.update()
        elif self.mode == "add_vertex":
            roi, _ = self._find_edge_at(qpos)
            if roi != getattr(self, 'hover_roi', None):
                self.hover_roi = roi
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if getattr(self, 'is_erasing', False):
                self.is_erasing = False
                self._calculate_volume() 
                
            if getattr(self, 'is_brushing', False):
                self.is_brushing = False
                self._calculate_volume()
                
            if getattr(self, 'is_moving_vertex', False):
                self.is_moving_vertex = False
                self.setCursor(Qt.PointingHandCursor)
                self._try_merge_rois()
                self._recalculate_last_seed() 
                self.update()
                
            self.is_panning = False
            self.is_windowing = False
            if self.mode == "navigate": self.setCursor(Qt.OpenHandCursor)

        elif event.button() == Qt.RightButton: 
            if getattr(self, 'is_zooming', False):
                self.is_zooming = False
                if self.mode == "navigate": self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event):
        if self.volume is None: return
        dz = -1 if event.angleDelta().y() > 0 else 1
        self.set_slice(self.current_z + dz)

    def keyPressEvent(self, event):
        if self.mode == "draw_roi":
            if event.key() == Qt.Key_Escape:
                self.cancel_drawing()
            elif event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
                self.undo_last_point()

    def _emit_drawing_status(self):
        state = self.drawing_state
        if state.is_drawing:
            slice_count = state.get_slice_count()
            min_z, max_z = state.get_slice_range()
            if slice_count > 1:
                self.status_msg.emit(f"跨层绘制中：{min_z + 1}→{max_z + 1}")
            else:
                self.status_msg.emit(f"绘制中：第{state.start_slice + 1}层")
        else:
            self.status_msg.emit("")

    def _save_current_polygon_cross_slice(self):
        if not self.current_polygon: return
        self.current_polygon.closed = True
        
        min_z, max_z = self.drawing_state.get_slice_range()
        for z in range(min_z, max_z + 1):
            roi_copy = self.current_polygon.copy()
            self.rois[z].append(roi_copy)
            
        self.current_polygon = None
        self.drawing_state.reset()
        self._emit_drawing_status()
        self._try_merge_rois()
        self._recalculate_last_seed() 
        self.update()

    def _apply_eraser_brush(self, pos: QPointF):
        ix, iy = int(pos.x()), int(pos.y())
        r = self.eraser_radius
        
        y_min, y_max = max(0, iy - r), min(self.img_height, iy + r + 1)
        x_min, x_max = max(0, ix - r), min(self.img_width, ix + r + 1)
        
        y, x = np.ogrid[y_min - iy : y_max - iy, x_min - ix : x_max - ix]
        disk = x**2 + y**2 <= r**2
        
        self.erase_mask[self.current_z, y_min:y_max, x_min:x_max][disk] = True
        self.mask[self.current_z, y_min:y_max, x_min:x_max][disk] = False
        
        self._update_pixmap()
        self.update()

    def _apply_eraser_line(self, p1: QPointF, p2: QPointF):
        dist = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
        steps = max(1, int(dist / (max(1, self.eraser_radius / 2))))
        for i in range(steps + 1):
            t = i / steps
            cx = p1.x() + t * (p2.x() - p1.x())
            cy = p1.y() + t * (p2.y() - p1.y())
            self._apply_eraser_brush(QPointF(cx, cy))

    def _apply_brush_action(self, pos: QPointF):
        ix, iy = int(pos.x()), int(pos.y())
        r = self.eraser_radius
        
        y_min, y_max = max(0, iy - r), min(self.img_height, iy + r + 1)
        x_min, x_max = max(0, ix - r), min(self.img_width, ix + r + 1)
        
        y, x = np.ogrid[y_min - iy : y_max - iy, x_min - ix : x_max - ix]
        disk = x**2 + y**2 <= r**2
        
        self.base_mask[self.current_z, y_min:y_max, x_min:x_max][disk] = True
        self.erase_mask[self.current_z, y_min:y_max, x_min:x_max][disk] = False
        self.mask[self.current_z, y_min:y_max, x_min:x_max][disk] = True
        
        self._update_pixmap()
        self.update()

    def _apply_brush_line(self, p1: QPointF, p2: QPointF):
        dist = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
        steps = max(1, int(dist / (max(1, self.eraser_radius / 2))))
        for i in range(steps + 1):
            t = i / steps
            cx = p1.x() + t * (p2.x() - p1.x())
            cy = p1.y() + t * (p2.y() - p1.y())
            self._apply_brush_action(QPointF(cx, cy))

    def _try_merge_rois(self):
        if not HAS_SHAPELY: return
        merged_count = 0
        for z in list(self.rois.keys()):
            rois_list = self.rois[z]
            if len(rois_list) > 1:
                old_count = len(rois_list)
                self.rois[z] = merge_overlapping_polygons(rois_list)
                merged_count += old_count - len(self.rois[z])
        if merged_count > 0:
            self._clear_edit_state()

    def undo_last_point(self):
        if self.current_polygon and self.current_polygon.points:
            self.current_polygon.points.pop()
            if not self.current_polygon.points:
                self.current_polygon = None
                self.drawing_state.reset()
                self._emit_drawing_status()
            self.update()

    def cancel_drawing(self):
        self.current_polygon = None
        self.drawing_state.reset()
        self._emit_drawing_status()
        self.update()

    def clear_all_rois(self):
        self.rois.clear()
        self._clear_edit_state()
        self._recalculate_last_seed()
        self.update()

    def _recalculate_last_seed(self):
        if self.last_seed is not None:
            self._execute_magic_wand(*self.last_seed, is_update=True)
        else:
            self._calculate_volume()
            self._update_pixmap()
            self.update()

    def _execute_magic_wand(self, x, y, z, is_update=False):
        if not is_update:
            self._save_undo_state()
            np.copyto(self.base_mask, self.mask)
            self.current_wand_mask.fill(False)
            # 【修复 BUG ②】: 不再清空 erase_mask，橡皮擦图层将完美约束接下来的魔棒泛洪
            self.last_seed = (x, y, z)

        if not (self.limit_z_min <= z <= self.limit_z_max):
            self.status_msg.emit(f"提示：该点击层({z+1})不在您设定的提取界限[{self.limit_z_min+1} - {self.limit_z_max+1}] 内！")
            self.current_wand_mask.fill(False)
            np.logical_or(self.base_mask, self.current_wand_mask, out=self.mask)
            self.mask[self.erase_mask] = False
            self._calculate_volume()
            self._update_pixmap()
            self.update()
            return

        seed_value = self.volume[z, int(y), int(x)]
        if not (self.magic_wand_lower <= seed_value <= self.magic_wand_upper):
            self.status_msg.emit(f"种子点 HU={seed_value}，不在[{self.magic_wand_lower}, {self.magic_wand_upper}] 范围内！")
            self.current_wand_mask.fill(False)
            np.logical_or(self.base_mask, self.current_wand_mask, out=self.mask)
            self.mask[self.erase_mask] = False
            self._calculate_volume()
            self._update_pixmap()
            self.update()
            return

        has_roi = any(len(rois) > 0 for rois in self.rois.values())
        restore_backups =[]
        
        if has_roi:
            for zi, roi_list in self.rois.items():
                if not roi_list: continue
                if not (self.limit_z_min <= zi <= self.limit_z_max): continue
                
                slice_mask = np.zeros((self.img_height, self.img_width), dtype=bool)
                for roi in roi_list:
                    poly_mask = generate_roi_mask_from_polygon(roi, (self.img_height, self.img_width))
                    slice_mask = slice_mask | poly_mask
                
                if zi == z and not slice_mask[int(y), int(x)]:
                    self.status_msg.emit("提示：当前种子点已被您绘制的防漏墙隔离，泛洪取消。")
                    self.current_wand_mask.fill(False)
                    np.logical_or(self.base_mask, self.current_wand_mask, out=self.mask)
                    self.mask[self.erase_mask] = False
                    self._calculate_volume()
                    self._update_pixmap()
                    self.update()
                    return
                
                outside_mask = ~slice_mask
                restore_backups.append((zi, outside_mask, self.volume[zi][outside_mask].copy()))
                self.volume[zi][outside_mask] = -2000

        seed_list =[(int(x), int(y), int(z - self.limit_z_min))]
        
        try:
            sitk_img = sitk.GetImageFromArray(self.volume[self.limit_z_min : self.limit_z_max + 1])
            seg_img = sitk.ConnectedThreshold(
                image1=sitk_img,
                seedList=seed_list,
                lower=float(self.magic_wand_lower),
                upper=float(self.magic_wand_upper),
                replaceValue=1
            )
            sub_mask = sitk.GetArrayFromImage(seg_img) > 0
            del sitk_img, seg_img
        except Exception as e:
            self.status_msg.emit("提示：运算异常，可能是系统内存碎片化严重。")
            sub_mask = None
        finally:
            for zi, outside_mask, orig_pixels in restore_backups:
                self.volume[zi][outside_mask] = orig_pixels

        if sub_mask is None:
            self.current_wand_mask.fill(False)
            np.logical_or(self.base_mask, self.current_wand_mask, out=self.mask)
            self.mask[self.erase_mask] = False
            self._calculate_volume()
            self._update_pixmap()
            self.update()
            return
        
        self.current_wand_mask.fill(False)
        self.current_wand_mask[self.limit_z_min : self.limit_z_max + 1] = sub_mask
            
        z_indices = np.where(self.current_wand_mask.any(axis=(1, 2)))[0]
        for zi in z_indices:
            self.current_wand_mask[zi] = ndimage.binary_fill_holes(self.current_wand_mask[zi])
            
        np.logical_or(self.base_mask, self.current_wand_mask, out=self.mask)
        self.mask[self.erase_mask] = False
        
        self.status_msg.emit(f"🪄魔棒提取完毕 | 种子层:{z+1} HU:{seed_value}")
        self._calculate_volume()
        self._update_pixmap()
        self.update()

    def _calculate_volume(self):
        voxel_volume_mm3 = self.spacing_zyx[0] * self.spacing_zyx[1] * self.spacing_zyx[2]
        
        if self.limit_z_min <= self.limit_z_max:
            valid_mask = self.mask[self.limit_z_min : self.limit_z_max + 1]
            voxel_count = np.sum(valid_mask)
        else:
            valid_mask = np.zeros_like(self.mask)
            voxel_count = 0
            
        volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0
        
        # ---------------------------------------------------------
        # 【全自动后台计算 - 多田公式 Tada Formula ABC/2】
        # ---------------------------------------------------------
        tada_ml, z_max_idx, A_cm, B_cm, C_cm = 0.0, 0, 0.0, 0.0, 0.0
        if voxel_count > 0:
            z_counts = valid_mask.sum(axis=(1, 2))
            z_max_rel = np.argmax(z_counts)
            z_max_idx = self.limit_z_min + z_max_rel
            
            C_cm = np.count_nonzero(z_counts) * self.spacing_zyx[0] / 10.0
            
            slice_mask = valid_mask[z_max_rel]
            y_coords, x_coords = np.nonzero(slice_mask)
            points = np.column_stack((x_coords * self.spacing_zyx[2], y_coords * self.spacing_zyx[1]))
            
            if len(points) >= 2:
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    
                    dist_matrix = cdist(hull_points, hull_points)
                    i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
                    max_dist = dist_matrix[i, j]
                    A_cm = max_dist / 10.0
                    
                    p1, p2 = hull_points[i], hull_points[j]
                    vec_A = p2 - p1
                    len_A = np.linalg.norm(vec_A)
                    if len_A > 0:
                        dir_A = vec_A / len_A
                        dir_B = np.array([-dir_A[1], dir_A[0]])
                        projections = points @ dir_B
                        B_cm = (np.max(projections) - np.min(projections)) / 10.0
                except BaseException:
                    min_pt = np.min(points, axis=0)
                    max_pt = np.max(points, axis=0)
                    A_cm = np.linalg.norm(max_pt - min_pt) / 10.0
                    B_cm = 0.0
                    
            tada_ml = (A_cm * B_cm * C_cm) / 2.0

        self.volume_changed.emit(volume_ml, tada_ml, z_max_idx, A_cm, B_cm, C_cm)

    def undo(self):
        if self.undo_mask is not None:
            np.copyto(self.mask, self.undo_mask)
            np.copyto(self.base_mask, self.undo_base_mask)
            np.copyto(self.current_wand_mask, self.undo_current_wand_mask)
            np.copyto(self.erase_mask, self.undo_erase_mask)
            self.last_seed = self.undo_last_seed
            self._calculate_volume()
            self._update_pixmap()
            self.update()

    def clear_mask(self):
        if self.mask is not None:
            self._save_undo_state()
            self.mask.fill(False)
            self.base_mask.fill(False)
            self.current_wand_mask.fill(False)
            self.erase_mask.fill(False)
            self.last_seed = None
            self._calculate_volume()
            self._update_pixmap()
            self.update()

# ==========================================
# 主窗口 
# ==========================================
class HematomaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能血肿体积计算工具 (自动化多田公式评估版)")
        self.resize(1700, 850)
        
        self.setStyleSheet("""
            QMainWindow, QWidget { 
                background: #2d2d2d; 
                color: white; 
                font-family: "Microsoft YaHei"; 
                font-size: 11pt; 
            }
            QPushButton { 
                background: #3d3d3d; 
                border: 1px solid #555; 
                padding: 8px; 
                border-radius: 4px; 
                font-size: 11pt;
            }
            QPushButton:hover { background: #4d4d4d; }
            QPushButton:checked { background: #1a6fbf; border: 1px solid #2a7fcf; font-weight: bold; }
            QSpinBox, QSlider { background: #1c1e26; color: white; border: 1px solid #555; padding: 6px; font-size: 11pt; }
            QComboBox { background: #1c1e26; color: white; border: 1px solid #555; padding: 6px; font-size: 11pt; }
            QComboBox QAbstractItemView { background: #1c1e26; color: white; selection-background-color: #1a6fbf; }
            
            QGroupBox { 
                border: 1px solid #555; 
                border-radius: 6px; 
                margin-top: 28px;   
                padding-top: 18px;  
                padding-bottom: 8px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left;
                left: 12px; 
                top: 0px;           
                padding: 0 8px; 
                color: #ddd; 
                font-size: 11pt; 
                font-weight: bold; 
            }
            
            #ResGroup { border: 2px solid #16a085; }
            #ResGroup::title { color: #1abc9c; font-weight: bold; font-size: 12pt; }
        """)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        left_panel = QWidget()
        left_panel.setFixedWidth(420) 
        lp_layout = QVBoxLayout(left_panel)
        lp_layout.setSpacing(12) 
        lp_layout.setContentsMargins(10, 10, 10, 20)
        
        # --- 1. 数据加载区 ---
        grp_data = QGroupBox("1. 数据加载")
        lay_data = QVBoxLayout(grp_data)
        self.btn_open = QPushButton("📂 选择文件夹并扫描")
        self.btn_open.setStyleSheet("background: #2980b9; font-weight: bold; font-size: 12pt;")
        self.btn_open.clicked.connect(self.open_dir)
        lay_data.addWidget(self.btn_open)
        
        self.progress = QProgressBar()
        self.progress.setFixedHeight(8)
        self.progress.setTextVisible(False)
        lay_data.addWidget(self.progress)
        
        self.combo_series = QComboBox()
        self.combo_series.setToolTip("选择包含平扫的序列")
        lay_data.addWidget(self.combo_series)
        
        self.btn_load = QPushButton("⬇️ 载入选定序列")
        self.btn_load.setStyleSheet("background: #16a085; font-weight: bold; font-size: 12pt;")
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self.load_selected_series)
        lay_data.addWidget(self.btn_load)
        lp_layout.addWidget(grp_data)
        
        # --- 2. 交互模式区 ---
        grp_mode = QGroupBox("2. 交互与提取模式")
        lay_mode = QVBoxLayout(grp_mode)
        self.mode_group = QButtonGroup(self)
        
        btn_nav = QPushButton("🖐 漫游")
        btn_nav.setCheckable(True)
        btn_nav.setChecked(True)
        btn_nav.setToolTip("左键拖拽平移，右键上下拖拽缩放")
        btn_nav.clicked.connect(lambda: self.canvas.set_mode("navigate"))
        
        btn_win = QPushButton("🌓 调窗")
        btn_win.setCheckable(True)
        btn_win.clicked.connect(lambda: self.canvas.set_mode("window"))
        
        btn_wand = QPushButton("🪄 3D 魔棒提取")
        btn_wand.setCheckable(True)
        btn_wand.setStyleSheet("QPushButton:checked { background: #c0392b; border-color: #e74c3c; font-size: 12pt; }")
        btn_wand.clicked.connect(lambda: self.canvas.set_mode("magic_wand"))
        
        btn_roi = QPushButton("📐 防漏墙(跨层)")
        btn_roi.setCheckable(True)
        btn_roi.setStyleSheet("QPushButton:checked { background: #d35400; border-color: #e67e22; }")
        btn_roi.clicked.connect(lambda: self.canvas.set_mode("draw_roi"))
        
        btn_erase = QPushButton("🧽 橡皮(减)")
        btn_erase.setToolTip("按住鼠标左键涂抹，擦掉多余血肿")
        btn_erase.setCheckable(True)
        btn_erase.setStyleSheet("QPushButton:checked { background: #8e44ad; border-color: #9b59b6; }")
        btn_erase.clicked.connect(lambda: self.canvas.set_mode("eraser"))
        
        btn_brush = QPushButton("🖌️ 画笔(增)")
        btn_brush.setToolTip("按住鼠标左键涂抹，手动填补漏选血肿")
        btn_brush.setCheckable(True)
        btn_brush.setStyleSheet("QPushButton:checked { background: #27ae60; border-color: #2ecc71; }")
        btn_brush.clicked.connect(lambda: self.canvas.set_mode("brush"))
        
        self.mode_group.addButton(btn_nav)
        self.mode_group.addButton(btn_win)
        self.mode_group.addButton(btn_wand)
        self.mode_group.addButton(btn_roi)
        self.mode_group.addButton(btn_erase)
        self.mode_group.addButton(btn_brush)
        
        grid1 = QHBoxLayout()
        grid1.addWidget(btn_nav)
        grid1.addWidget(btn_win)
        
        grid2 = QHBoxLayout()
        grid2.addWidget(btn_roi)
        grid2.addWidget(btn_wand)
        
        grid3 = QHBoxLayout()
        grid3.addWidget(btn_erase)
        grid3.addWidget(btn_brush)
        
        lay_mode.addLayout(grid1)
        lay_mode.addLayout(grid2)
        lay_mode.addLayout(grid3)
        
        lay_brush = QHBoxLayout()
        lay_brush.addWidget(QLabel("画笔/橡皮: "))
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(2, 40)
        self.slider_brush.setValue(10)
        self.slider_brush.valueChanged.connect(self.update_brush_size)
        lay_brush.addWidget(self.slider_brush)
        self.lbl_brush_size = QLabel("10 px")
        lay_brush.addWidget(self.lbl_brush_size)
        lay_mode.addLayout(lay_brush)
        
        lp_layout.addWidget(grp_mode)

        # --- 防漏墙微调编辑 ---
        grp_edit = QGroupBox("防漏墙顶点微调")
        lay_edit = QHBoxLayout(grp_edit)
        
        btn_move_v = QPushButton("移动")
        btn_move_v.setCheckable(True)
        btn_move_v.clicked.connect(lambda: self.canvas.set_mode("move_vertex"))
        
        btn_add_v = QPushButton("添加")
        btn_add_v.setCheckable(True)
        btn_add_v.clicked.connect(lambda: self.canvas.set_mode("add_vertex"))
        
        btn_del_v = QPushButton("删除")
        btn_del_v.setCheckable(True)
        btn_del_v.clicked.connect(lambda: self.canvas.set_mode("delete_vertex"))
        
        self.mode_group.addButton(btn_move_v)
        self.mode_group.addButton(btn_add_v)
        self.mode_group.addButton(btn_del_v)
        
        lay_edit.addWidget(btn_move_v)
        lay_edit.addWidget(btn_add_v)
        lay_edit.addWidget(btn_del_v)
        lp_layout.addWidget(grp_edit)
        
        # --- 3. 魔棒阈值 ---
        grp_th = QGroupBox("3. 魔棒阈值 (HU)")
        lay_th = QHBoxLayout(grp_th)
        self.spin_lower = QSpinBox()
        self.spin_lower.setRange(0, 100)
        self.spin_lower.setValue(40)
        self.spin_lower.valueChanged.connect(self.update_thresholds)
        
        self.spin_upper = QSpinBox()
        self.spin_upper.setRange(50, 200)
        self.spin_upper.setValue(90)
        self.spin_upper.valueChanged.connect(self.update_thresholds)
        
        lay_th.addWidget(QLabel("下限: "))
        lay_th.addWidget(self.spin_lower)
        lay_th.addWidget(QLabel("上限: "))
        lay_th.addWidget(self.spin_upper)
        lp_layout.addWidget(grp_th)

        # --- 4. 清理操作 ---
        grp_clear = QGroupBox("4. 全局清理操作")
        lay_clear = QVBoxLayout(grp_clear)
        
        row_clear1 = QHBoxLayout()
        btn_undo = QPushButton("撤销上步魔棒/画笔/橡皮")
        btn_undo.clicked.connect(lambda: self.canvas.undo())
        btn_clear_mask = QPushButton("清空全部血肿")
        btn_clear_mask.clicked.connect(lambda: self.canvas.clear_mask())
        row_clear1.addWidget(btn_undo)
        row_clear1.addWidget(btn_clear_mask)
        lay_clear.addLayout(row_clear1)
        
        btn_clear_roi = QPushButton("🗑 清除屏幕所有防漏墙")
        btn_clear_roi.setStyleSheet("color: #e67e22; border-color: #d35400; ")
        btn_clear_roi.clicked.connect(lambda: self.canvas.clear_all_rois())
        lay_clear.addWidget(btn_clear_roi)
        lp_layout.addWidget(grp_clear)
        
        self.lbl_status = QLabel(" ")
        self.lbl_status.setStyleSheet("color: #f39c12; font-weight: bold; padding: 0 5px; font-size: 11pt;")
        self.lbl_status.setWordWrap(True) 
        lp_layout.addWidget(self.lbl_status)
        
        lp_layout.addStretch()
        
        # --- 结果显示区 (多田公式展示在此) ---
        res_group = QGroupBox("体积计算结果")
        res_group.setObjectName("ResGroup") 
        res_layout = QVBoxLayout(res_group)
        
        self.lbl_volume = QLabel("0.00 mL")
        self.lbl_volume.setStyleSheet("color: #e74c3c; font-size: 34pt; font-weight: bold; ")
        self.lbl_volume.setContentsMargins(0, 5, 0, 0) 
        self.lbl_volume.setAlignment(Qt.AlignCenter)
        
        self.lbl_tada = QLabel("自动多田估算 (ABC/2): 0.00 mL\n(等待提取...)")
        self.lbl_tada.setStyleSheet("color: #95a5a6; font-size: 10pt; font-weight: normal;")
        self.lbl_tada.setContentsMargins(0, 0, 0, 10)
        self.lbl_tada.setAlignment(Qt.AlignCenter)
        
        res_layout.addWidget(self.lbl_volume)
        res_layout.addWidget(self.lbl_tada)
        lp_layout.addWidget(res_group)

        # 延时重算计时器
        self.calc_timer = QTimer(self)
        self.calc_timer.setSingleShot(True)
        self.calc_timer.timeout.connect(self.do_recalculate)

        # ==========================================
        # 右侧画布区 (双屏)
        # ==========================================
        right_panel = QWidget()
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(0,0,0,0)
         
        self.canvas = HematomaCanvas()
        self.canvas.eraser_radius = self.slider_brush.value()
        self.canvas.volume_changed.connect(self.on_volume_changed)
        self.canvas.status_msg.connect(self.lbl_status.setText)
        rp_layout.addWidget(self.canvas, 1)

        limit_layout = QHBoxLayout()
        self.btn_limit_min = QPushButton("↑ 设置当前为: 血肿第一层 (-)")
        self.btn_limit_max = QPushButton("↓ 设置当前为: 血肿最后一层 (-)")
        self.btn_limit_min.setStyleSheet("background: #d35400; color: white; font-weight: bold; font-size: 11.5pt;")
        self.btn_limit_max.setStyleSheet("background: #27ae60; color: white; font-weight: bold; font-size: 11.5pt;")
        self.btn_limit_min.clicked.connect(self.set_current_as_first_layer)
        self.btn_limit_max.clicked.connect(self.set_current_as_last_layer)
        limit_layout.addWidget(self.btn_limit_min)
        limit_layout.addWidget(self.btn_limit_max)
        rp_layout.addLayout(limit_layout)
        
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.canvas.set_slice)
        self.canvas.slice_changed.connect(self.sync_slider) 
        
        self.lbl_slice = QLabel("0/0")
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.lbl_slice)
        rp_layout.addLayout(slider_layout)

        layout.addWidget(left_panel)
        layout.addWidget(right_panel, 1)

    def set_current_as_first_layer(self):
        if self.canvas.volume is None: return
        curr = self.canvas.current_z
        if curr > self.canvas.limit_z_max:
            self.canvas.limit_z_max = curr
        self.canvas.limit_z_min = curr
        
        self.update_limit_buttons_text()
        # 【修复 BUG ①】移除立刻旧状态计算与图像渲染，统一交给定时器 150ms 延后刷新，防止闪烁
        self.canvas.update() # 仅立即更新文字警告 UI
        self.calc_timer.start(150) 

    def set_current_as_last_layer(self):
        if self.canvas.volume is None: return
        curr = self.canvas.current_z
        if curr < self.canvas.limit_z_min:
            self.canvas.limit_z_min = curr
        self.canvas.limit_z_max = curr
        
        self.update_limit_buttons_text()
        self.canvas.update() 
        self.calc_timer.start(150)

    def update_limit_buttons_text(self):
        self.btn_limit_min.setText(f"↑ 设置当前为: 血肿第一层 (已设第 {self.canvas.limit_z_min + 1} 层)")
        self.btn_limit_max.setText(f"↓ 设置当前为: 血肿最后一层 (已设第 {self.canvas.limit_z_max + 1} 层)")

    def sync_slider(self, z):
        self.slider.blockSignals(True)
        self.slider.setValue(z)
        self.slider.blockSignals(False)
        self.lbl_slice.setText(f"{z + 1}/{self.canvas.depth}")

    def update_brush_size(self, val):
        self.canvas.eraser_radius = val
        self.lbl_brush_size.setText(f"{val} px")
        if self.canvas.mode in ("eraser", "brush"):
            self.canvas.update()

    def open_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择平扫 DICOM 文件夹")
        if d:
            self.combo_series.clear()
            self.btn_load.setEnabled(False)
            self.scan_thread = ScanThread(d)
            self.scan_thread.progress.connect(self.progress.setValue)
            self.scan_thread.finished_signal.connect(self.on_scan_finished)
            self.scan_thread.start()

    def on_scan_finished(self, series_dict):
        if not series_dict:
            QMessageBox.warning(self, "提示", "未找到 DICOM 序列")
            return
            
        for s in sorted(series_dict.values(), key=lambda x: x.series_number):
            display_text = f"#{s.series_number:03d} | {s.file_count}张 | {s.series_description[:20]}"
            self.combo_series.addItem(display_text, s)
            
        self.btn_load.setEnabled(True)
        for i in range(self.combo_series.count()):
            s = self.combo_series.itemData(i)
            desc = s.series_description.upper()
            if "C-" in desc or "PLAIN" in desc or "平扫" in desc:
                self.combo_series.setCurrentIndex(i)
                break

    def load_selected_series(self):
        series = self.combo_series.currentData()
        if not series: return
            
        files = [f[1] for f in series.files]
        if not files: return
            
        self.lbl_volume.setText("读取中...")
        self.btn_load.setEnabled(False)
        QApplication.processEvents()
        
        try:
            img = _sitk_read_image_safe(files, sitk.sitkInt16)
            volume = sitk.GetArrayFromImage(img)
            depth, h, w = volume.shape
            
            sp = img.GetSpacing()
            spacing_zyx = list((sp[2], sp[1], sp[0]))
            
            if len(files) > 1:
                try:
                    ds1 = pydicom.dcmread(files[0], stop_before_pixels=True)
                    ds2 = pydicom.dcmread(files[-1], stop_before_pixels=True)
                    
                    if hasattr(ds1, 'ImagePositionPatient') and hasattr(ds2, 'ImagePositionPatient'):
                        p1 = np.array([float(x) for x in ds1.ImagePositionPatient])
                        p2 = np.array([float(x) for x in ds2.ImagePositionPatient])
                        
                        calc_z = -1
                        if hasattr(ds1, 'ImageOrientationPatient'):
                            iop = [float(x) for x in ds1.ImageOrientationPatient]
                            if len(iop) == 6:
                                row_dir = np.array(iop[:3])
                                col_dir = np.array(iop[3:])
                                normal = np.cross(row_dir, col_dir)
                                n_norm = np.linalg.norm(normal)
                                if n_norm > 1e-5:
                                    normal = normal / n_norm
                                    calc_z = abs(np.dot(p1 - p2, normal)) / (len(files) - 1)
                        
                        if calc_z <= 0:
                            calc_z = np.linalg.norm(p1 - p2) / (len(files) - 1)
                            
                        if calc_z > 0:
                            spacing_zyx[0] = calc_z
                except Exception as e:
                    pass
            
            self.canvas.set_volume(volume, tuple(spacing_zyx))
            
            self.slider.blockSignals(True)
            self.slider.setRange(0, depth - 1)
            self.slider.setValue(depth // 2)
            self.slider.blockSignals(False) 
            self.lbl_slice.setText(f"{depth//2 + 1}/{depth}")
            
            self.canvas.limit_z_min = 0
            self.canvas.limit_z_max = depth - 1
            self.update_limit_buttons_text()
            
            self.lbl_volume.setText("0.00 mL")
            self.lbl_tada.setText("自动多田估算 (ABC/2): 0.00 mL\n(等待提取...)")
            self.lbl_status.setText(" ")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "读取错误", f"序列读取失败：\n{str(e)}")
            self.lbl_volume.setText("0.00 mL")
        finally:
            self.btn_load.setEnabled(True)

    def update_thresholds(self):
        self.canvas.magic_wand_lower = self.spin_lower.value()
        self.canvas.magic_wand_upper = self.spin_upper.value()
        self.calc_timer.start(150)

    def do_recalculate(self):
        self.canvas._recalculate_last_seed()

    def on_volume_changed(self, vol_ml, tada_ml, z_idx, A, B, C):
        self.lbl_volume.setText(f"{vol_ml:.2f} mL")
        if vol_ml > 0:
            self.lbl_tada.setText(f"多田估算: {tada_ml:.2f} mL (寻至第 {z_idx + 1} 层)\n[ A: {A:.2f} cm | B: {B:.2f} cm | C: {C:.2f} cm ]")
        else:
            self.lbl_tada.setText("自动多田估算 (ABC/2): 0.00 mL\n(等待提取...)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = HematomaApp()
    win.show()
    sys.exit(app.exec_())