"""
v2.9.2 修复版 - 血管缩放增强
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[修复] 彻底修复“血管缩放”模块中血管无法变细/消失、逻辑混乱的问题
[优化] 支持因子取0时完全剔除目标血管群，支持任意反向操作
[增强] <1.0的缩放系数增加连贯的形态学变细和变暗效果
[修复] 消除边缘"挡板"伪影与ICA区域优化（继承自v2.9.1）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import re
import sys
import json
import gc
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import time
from datetime import datetime
from collections import defaultdict
from scipy import ndimage

try:
    import SimpleITK as sitk
except ImportError:
    print("错误: 找不到 SimpleITK 库。请运行 'pip install SimpleITK'")
    sys.exit(1)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QProgressBar,
    QTextEdit, QGroupBox, QMessageBox, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QSlider, QRadioButton, QFrame, QDialog,
    QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import pydicom
from pydicom.uid import generate_uid

print("头颅CTA减影-whw 版 v2.9.2")


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


# ════════════════════════════════════════════════════════════════════
# 优化级别
# ════════════════════════════════════════════════════════════════════
OPTIMIZATION_LEVELS = {
    "light": {"name": "轻量级", "description": "局部自适应 + ICA精确保护"},
    "standard": {"name": "标准级", "description": "+ 亚像素精修 + 静脉窦"},
    "deep": {"name": "深度级", "description": "+ Frangi增强"},
}


# ════════════════════════════════════════════════════════════════════
# 设备预设 - 修复：确保所有参数都有
# ════════════════════════════════════════════════════════════════════
PRESET_FILE = Path(__file__).parent / "device_presets.json"

_MANUFACTURER_MAP = {
    ("siemens",): "Siemens SOMATOM",
    ("ge ", "ge", "general electric"): "GE Revolution/Discovery",
    ("philips",): "Philips Brilliance/IQon",
    ("canon", "toshiba"): "Canon Aquilion",
    ("united imaging", "联影", "uih"): "联影 uCT",
}

BUILTIN_PRESETS = {
    "通用默认": {
        "bone_strength": 1.2, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.7, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400, "description": "适合未知设备",
    },
    "Siemens SOMATOM": {
        "bone_strength": 1.2, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400, "description": "SOMATOM系列",
    },
    "GE Revolution/Discovery": {
        "bone_strength": 1.4, "vessel_sensitivity": 0.9, "vessel_enhance": 1.8,
        "smooth_sigma": 0.9, "clean_bone_edges": True, "min_vessel_size": 6,
        "wc": 220, "ww": 420, "description": "GE CT系列",
    },
    "Philips Brilliance/IQon": {
        "bone_strength": 1.0, "vessel_sensitivity": 1.1, "vessel_enhance": 2.2,
        "smooth_sigma": 0.5, "clean_bone_edges": True, "min_vessel_size": 4,
        "wc": 180, "ww": 380, "description": "Philips CT系列",
    },
    "Canon Aquilion": {
        "bone_strength": 1.1, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": False, "min_vessel_size": 5,
        "wc": 200, "ww": 400, "description": "Canon/Toshiba系列",
    },
    "联影 uCT": {
        "bone_strength": 1.3, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.8, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400, "description": "联影uCT系列",
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
        combined = (str(getattr(ds, "Manufacturer", "")) + " " +
                    str(getattr(ds, "ManufacturerModelName", ""))).lower()
        for keywords, preset_name in _MANUFACTURER_MAP.items():
            if any(kw in combined for kw in keywords):
                return preset_name
    except:
        pass
    return None


# ════════════════════════════════════════════════════════════════════
# 序列扫描
# ════════════════════════════════════════════════════════════════════
class SeriesInfo:
    def __init__(self):
        self.series_uid = ""
        self.series_number = 0
        self.series_description = ""
        self.study_description = ""
        self.modality = ""
        self.body_part = ""
        self.protocol_name = ""
        self.acquisition_time = None
        self.file_count = 0
        self.files = []
        self.slice_thickness = 0
        self.image_shape = (0, 0)
        self.manufacturer = ""
        self._contrast_status = None
        self._contrast_cached = False

    @property
    def contrast_status(self):
        if not self._contrast_cached:
            self._contrast_status = _calc_contrast_status(self)
            self._contrast_cached = True
        return self._contrast_status


def _calc_contrast_status(series):
    desc = series.series_description.upper()
    pos = [r'\bC\+', r'\bC\s*\+', r'CE\b', r'CONTRAST', r'ENHANCED', r'POST', r'ARTERIAL', r'增强', r'动脉期']
    neg = [r'\bC-', r'\bC\s*-', r'\bNC\b', r'NON-CONTRAST', r'PLAIN', r'PRE\b', r'WITHOUT', r'平扫', r'非增强']
    if any(re.search(p, desc) for p in pos):
        return True
    if any(re.search(p, desc) for p in neg):
        return False
    return None


def parse_time(time_str):
    if not time_str:
        return None
    try:
        time_str = str(time_str).split('.')[0]
        if len(time_str) >= 6:
            return datetime.strptime(time_str[:6], "%H%M%S")
    except:
        pass
    return None


def analyze_dicom_file(filepath):
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        return {
            'filepath': filepath,
            'series_uid': str(getattr(ds, 'SeriesInstanceUID', '')),
            'series_number': int(getattr(ds, 'SeriesNumber', 0)),
            'series_description': str(getattr(ds, 'SeriesDescription', '')),
            'study_description': str(getattr(ds, 'StudyDescription', '')),
            'modality': str(getattr(ds, 'Modality', '')),
            'body_part': str(getattr(ds, 'BodyPartExamined', '')),
            'protocol_name': str(getattr(ds, 'ProtocolName', '')),
            'instance_number': int(getattr(ds, 'InstanceNumber', 0)),
            'slice_thickness': float(getattr(ds, 'SliceThickness', 0)),
            'rows': int(getattr(ds, 'Rows', 0)),
            'columns': int(getattr(ds, 'Columns', 0)),
            'acquisition_time': parse_time(getattr(ds, 'AcquisitionTime', '')),
            'manufacturer': str(getattr(ds, 'Manufacturer', '')),
        }
    except:
        return None


def scan_directory_for_series(directory, progress_callback=None, log_callback=None):
    path = Path(directory)
    all_files = list(path.rglob('*'))
    dicom_files = []
    for f in all_files:
        if not f.is_file():
            continue
        try:
            with open(f, 'rb') as fp:
                fp.seek(128)
                if fp.read(4) == b'DICM':
                    dicom_files.append(f)
        except:
            pass
    if log_callback:
        log_callback(f"找到 {len(dicom_files)} 个DICOM文件")
    if not dicom_files:
        return {}

    series_dict = defaultdict(SeriesInfo)
    for i, f in enumerate(dicom_files):
        info = analyze_dicom_file(str(f))
        if info and info['series_uid']:
            uid = info['series_uid']
            series = series_dict[uid]
            if series.file_count == 0:
                series.series_uid = uid
                series.series_number = info['series_number']
                series.series_description = info['series_description']
                series.study_description = info['study_description']
                series.modality = info['modality']
                series.body_part = info['body_part']
                series.protocol_name = info['protocol_name']
                series.acquisition_time = info['acquisition_time']
                series.slice_thickness = info['slice_thickness']
                series.image_shape = (info['rows'], info['columns'])
                series.manufacturer = info['manufacturer']
            series.files.append((info['instance_number'], str(f)))
            series.file_count += 1
            if info['acquisition_time']:
                if series.acquisition_time is None or info['acquisition_time'] < series.acquisition_time:
                    series.acquisition_time = info['acquisition_time']
        if progress_callback:
            progress_callback(int((i + 1) / len(dicom_files) * 100))
    for series in series_dict.values():
        series.files.sort(key=lambda x: x[0])
    return dict(series_dict)


def is_head_cta_series(series):
    desc = f"{series.series_description} {series.study_description} {series.protocol_name} {series.body_part}".upper()
    exclude = ['SCOUT', 'LOCALIZER', 'TOPOGRAM', '定位', 'LUNG', 'CHEST', '肺', '胸', 'CARDIAC', 'HEART', '心', 'ABDOMEN', 'LIVER', '腹']
    if any(kw in desc for kw in exclude):
        return False
    if series.modality != 'CT' or series.file_count < 50:
        return False
    has_head = any(kw in desc for kw in ['HEAD', 'BRAIN', 'CEREBR', 'CRANIAL', '头', '颅', '脑', 'CAROTID'])
    has_cta = any(kw in desc for kw in ['CTA', 'ANGIO', 'C+', 'C-', '血管', '动脉'])
    return (has_head and has_cta) or (has_head and series.file_count >= 100)


def find_cta_pairs(series_dict):
    cta_series = [s for s in series_dict.values() if is_head_cta_series(s)]
    if len(cta_series) < 2:
        return [], cta_series
    groups = defaultdict(list)
    for s in cta_series:
        groups[(s.file_count, round(s.slice_thickness, 1))].append(s)
    pairs, used = [], set()
    for group in groups.values():
        if len(group) < 2:
            continue
        enhanced, plain, unknown = [], [], []
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
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(dict, list, list)

    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def run(self):
        try:
            self.log.emit("=" * 55)
            self.log.emit(f"扫描: {self.directory}")
            self.log.emit("=" * 55)
            series_dict = scan_directory_for_series(self.directory, self.progress.emit, self.log.emit)
            self.log.emit(f"\n找到 {len(series_dict)} 个序列\n")
            for s in sorted(series_dict.values(), key=lambda s: s.series_number):
                time_str = s.acquisition_time.strftime("%H:%M:%S") if s.acquisition_time else "--:--:--"
                st = s.contrast_status
                marker = "C+" if st is True else "C-" if st is False else "★ " if is_head_cta_series(s) else "  "
                desc = s.series_description[:35] or "(无描述)"
                self.log.emit(f"{marker} #{s.series_number:3d} | {s.file_count:4d}张 | {time_str} | {desc}")
            pairs, cta_series = find_cta_pairs(series_dict)
            if pairs:
                pre, post = pairs[0]
                self.log.emit(f"\n★ 自动配对:\n  平扫: #{pre.series_number} {pre.series_description[:30]}\n  增强: #{post.series_number} {post.series_description[:30]}")
            self.progress.emit(100)
            self.finished_signal.emit(series_dict, pairs, cta_series)
        except Exception as e:
            self.log.emit(f"[错误] {e}")
            self.finished_signal.emit({}, [], [])


# ════════════════════════════════════════════════════════════════════
# 配准工具
# ════════════════════════════════════════════════════════════════════
def fft_phase_correlation_2d(fixed, moving, max_shift=15):
    from numpy.fft import fft2, ifft2, fftshift
    h, w = fixed.shape
    margin = max(h // 6, 30)
    f_roi = fixed[margin:-margin, margin:-margin].astype(np.float32)
    m_roi = moving[margin:-margin, margin:-margin].astype(np.float32)
    wy = np.hanning(f_roi.shape[0]).astype(np.float32)
    wx = np.hanning(f_roi.shape[1]).astype(np.float32)
    window = np.outer(wy, wx)
    f1 = fft2(f_roi * window)
    f2 = fft2(m_roi * window)
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    correlation = np.real(fftshift(ifft2(cross_power)))
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    py, px = peak_idx
    dy = float(py - correlation.shape[0] // 2)
    dx = float(px - correlation.shape[1] // 2)
    return float(np.clip(dy, -max_shift, max_shift)), float(np.clip(dx, -max_shift, max_shift))


def shift_image_2d(image, dy, dx):
    return ndimage.shift(image.astype(np.float32), [dy, dx], order=1, mode='constant', cval=0)


class ParamAnalyzeThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, pre_files, post_files):
        super().__init__()
        self.pre_files = pre_files
        self.post_files = post_files

    def run(self):
        try:
            self.log.emit("\n" + "=" * 55 + "\n智能参数分析\n" + "=" * 55)
            pre_dict = {inst: path for inst, path in self.pre_files}
            post_dict = {inst: path for inst, path in self.post_files}
            common = sorted(set(pre_dict.keys()) & set(post_dict.keys()))
            if not common:
                self.finished_signal.emit({'error': '序列不匹配'})
                return
            total = len(common)
            sample_indices = [common[total // 2]] if total <= 10 else [common[total // 2 - 5], common[total // 2], common[total // 2 + 5]]
            all_chars = []
            for i, inst in enumerate(sample_indices):
                pre_d = pydicom.dcmread(pre_dict[inst]).pixel_array.astype(np.float32)
                post_d = pydicom.dcmread(post_dict[inst]).pixel_array.astype(np.float32)
                dy, dx = fft_phase_correlation_2d(pre_d, post_d)
                post_aligned = shift_image_2d(post_d, dy, dx)
                diff = np.clip(post_aligned - pre_d, 0, None)
                bone = pre_d > 150
                air = pre_d < -800
                noise = float(pre_d[air].std()) if air.sum() > 100 else 15.0
                strong = diff > 50
                vessel = float(diff[strong].mean()) if strong.sum() > 0 else 0.0
                all_chars.append({'bone': float(bone.sum() / bone.size), 'noise': noise, 'vessel_signal': vessel})
                self.progress.emit(int((i + 1) / len(sample_indices) * 100))
            
            avg = {k: float(np.mean([c[k] for c in all_chars])) for k in all_chars[0]}
            score = 1.0
            if avg['noise'] > 20:
                score -= 0.2
            if avg['vessel_signal'] < 40:
                score -= 0.2
            score = max(0, min(1, score))
            
            rec = {
                'bone_strength': 1.2,
                'vessel_sensitivity': 1.0,
                'vessel_enhance': 2.0 if avg['vessel_signal'] < 60 else 1.8,
                'clean_bone_edges': True,
                'min_vessel_size': 5,
                'smooth_sigma': 0.7,
                'wc': 200, 'ww': 400,
                'quality_score': score,
                'recommended_mode': 'fast' if score >= 0.7 else 'quality'
            }
            self.log.emit(f"图像质量: {'优良' if score >= 0.7 else '一般'}")
            self.finished_signal.emit(rec)
        except Exception as e:
            self.finished_signal.emit({'error': str(e)})


def execute_3d_registration(fixed_img, moving_img, log_cb=None, progress_cb=None):
    fixed_size = fixed_img.GetSize()
    shrink_factors = [1, 1, 1]
    if fixed_size[0] >= 400:
        shrink_factors[0] = 2
        shrink_factors[1] = 2
    if fixed_size[2] >= 100:
        shrink_factors[2] = 2

    if log_cb:
        log_cb(f"  缩放因子: {shrink_factors}")

    fixed_reg = sitk.Shrink(fixed_img, shrink_factors)
    moving_reg = sitk.Shrink(moving_img, shrink_factors)
    fixed_reg = sitk.Cast(fixed_reg, sitk.sitkFloat32)
    moving_reg = sitk.Cast(moving_reg, sitk.sitkFloat32)

    bone_mask = sitk.BinaryThreshold(fixed_reg, 200.0, 3000.0, 1, 0)
    bone_mask = sitk.Cast(bone_mask, sitk.sitkUInt8)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricFixedMask(bone_mask)

    reg_size = fixed_reg.GetSize()
    center_index = [(s - 1) / 2.0 for s in reg_size]
    center_physical = fixed_reg.TransformContinuousIndexToPhysicalPoint(center_index)

    initial_transform = sitk.Euler3DTransform()
    initial_transform.SetCenter(center_physical)
    R.SetInitialTransform(initial_transform, inPlace=False)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.001, numberOfIterations=60, estimateLearningRate=R.Never)
    R.SetInterpolator(sitk.sitkLinear)

    def _on_iter(method):
        it = method.GetOptimizerIteration()
        if progress_cb:
            progress_cb(int((it / 60.0) * 25))
        if log_cb and it % 15 == 0:
            log_cb(f"  迭代 {it:02d} | MSE: {method.GetMetricValue():.2f}")

    R.AddCommand(sitk.sitkIterationEvent, lambda: _on_iter(R))
    if log_cb:
        log_cb("  3D配准...")

    final_transform = R.Execute(fixed_reg, moving_reg)

    if log_cb:
        log_cb(f"  配准完成. MSE: {R.GetMetricValue():.2f}")
        params = final_transform.GetParameters()
        if len(params) == 6:
            log_cb(f"  旋转: {np.degrees(params[0]):.2f}°, {np.degrees(params[1]):.2f}°, {np.degrees(params[2]):.2f}°")
            log_cb(f"  平移: {params[3]:.2f}, {params[4]:.2f}, {params[5]:.2f} mm")

    del fixed_reg, moving_reg, bone_mask, R
    gc.collect()

    aligned_img = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
    return aligned_img


# ════════════════════════════════════════════════════════════════════
# 基础工具
# ════════════════════════════════════════════════════════════════════
def create_body_mask(image):
    """创建身体掩膜 - 修复边缘问题"""
    body = image > -400
    body = ndimage.binary_closing(body, iterations=3)
    body = ndimage.binary_fill_holes(body)
    labeled, num = ndimage.label(body)
    if num > 0:
        sizes = ndimage.sum(body, labeled, range(1, num + 1))
        body = labeled == (np.argmax(sizes) + 1)
    
    # ★ 关键修复：腐蚀边缘，避免边缘伪影 ★
    body = ndimage.binary_erosion(body, iterations=3)
    
    return body


def detect_thin_bone(pre_image):
    bone = pre_image > 150
    thin_bone = bone & ~ndimage.binary_erosion(bone, iterations=2)
    edges = np.abs(ndimage.sobel(pre_image.astype(np.float32)))
    ref_edges = edges[bone] if bone.sum() > 0 else edges
    high_edge = edges > np.percentile(ref_edges, 70)
    return thin_bone | (bone & high_edge)


def detect_equipment(pre, post, body=None):
    if body is None:
        body = create_body_mask(pre)
    high_both = (pre > 150) & (post > 150)
    stable = np.abs(post - pre) < 30
    body_dilated = ndimage.binary_dilation(body, iterations=8)
    return ndimage.binary_dilation(high_both & stable & ~body_dilated, iterations=5)


class SliceCache:
    def __init__(self):
        self.body = None
        self.scalp = None
        self.air_bone = None
        self.petrous = None
        self.thin_bone = None
        self.venous = None
        self.petrous_edge = None
        self.bone_edge = None
        self.ab_edge = None
        self.ica_protect = None


def precompute_slice_cache(pre_s, diff_pos, quality_mode):
    cache = SliceCache()
    cache.body = create_body_mask(pre_s)
    bone = pre_s > 150

    grad_y = ndimage.sobel(pre_s.astype(np.float32), axis=0)
    grad_x = ndimage.sobel(pre_s.astype(np.float32), axis=1)
    gradient = np.sqrt(grad_y ** 2 + grad_x ** 2)

    thin_bone = bone & ~ndimage.binary_erosion(bone, iterations=2)
    ref_edges = gradient[bone] if bone.sum() > 0 else gradient.ravel()
    high_edge = gradient > np.percentile(ref_edges, 70)
    cache.thin_bone = thin_bone | (bone & high_edge)

    bone_dilated_2 = ndimage.binary_dilation(bone, iterations=2)
    bone_eroded_2 = ndimage.binary_erosion(bone, iterations=2)
    cache.bone_edge = bone_dilated_2 & ~bone_eroded_2

    if not quality_mode:
        return cache

    body_eroded_8 = ndimage.binary_erosion(cache.body, iterations=8)
    scalp_zone = cache.body & ~body_eroded_8
    soft_tissue = (pre_s > -50) & (pre_s < 100)
    cache.scalp = ndimage.binary_dilation(scalp_zone & soft_tissue, iterations=2)

    air = pre_s < -200
    air_dilated_4 = ndimage.binary_dilation(air, iterations=4)
    bone_dilated_4 = ndimage.binary_dilation(bone, iterations=4)
    bone_dilated_6 = ndimage.binary_dilation(bone, iterations=6)
    cache.air_bone = air_dilated_4 & bone_dilated_4 & bone_dilated_6
    cache.ab_edge = air_dilated_4 & bone_dilated_4

    dense_bone = pre_s > 400
    high_gradient = gradient > 100
    petrous = dense_bone & ndimage.binary_dilation(high_gradient, iterations=2)
    petrous = ndimage.binary_dilation(petrous, iterations=3)
    cache.petrous = petrous
    cache.petrous_edge = petrous & ~ndimage.binary_erosion(petrous, iterations=2)

    # ICA保护区：岩骨内低密度+高增强
    ica_core = petrous & (pre_s < 150) & (diff_pos > 50)
    if ica_core.sum() > 0:
        cache.ica_protect = ndimage.binary_dilation(ica_core, iterations=2)
    else:
        cache.ica_protect = np.zeros_like(pre_s, dtype=bool)

    brain = (pre_s > 20) & (pre_s < 60)
    brain = ndimage.binary_erosion(brain, iterations=3)
    brain_dilated = ndimage.binary_dilation(brain, iterations=6)
    brain_edge = brain_dilated & ~brain
    medium_signal = (diff_pos > 20) & (diff_pos < 80)
    cache.venous = brain_edge & medium_signal

    return cache


# ════════════════════════════════════════════════════════════════════
# ★★★ 原始减影算法（保持不变作为基准）★★★
# ════════════════════════════════════════════════════════════════════
def fast_subtraction(pre, post_aligned, bone_strength=1.0, vessel_sensitivity=1.0, body=None):
    if body is None:
        body = create_body_mask(pre)
    diff_pos = np.clip(post_aligned - pre, 0, None)
    gain = np.ones_like(diff_pos, dtype=np.float32)
    
    # ★ 修复：边缘清零 ★
    gain[~body] = 0
    gain[pre < -500] = 0
    gain[pre > 500] = 0

    th_high = 120 * bone_strength
    th_med1 = 60 * bone_strength
    th_med2 = 120 * bone_strength
    high_bone = (pre > 300) & (pre <= 500)
    med_bone = (pre > 180) & (pre <= 300)

    gain[high_bone & (diff_pos < th_high)] = 0
    gain[high_bone & (diff_pos >= th_high)] = 0.1 / bone_strength
    gain[med_bone & (diff_pos < th_med1)] = 0
    gain[med_bone & (diff_pos >= th_med1) & (diff_pos < th_med2)] = 0.15 / bone_strength
    gain[med_bone & (diff_pos >= th_med2)] = 0.3 / bone_strength

    thin_bone = detect_thin_bone(pre)
    th_thin = 80 * bone_strength
    gain[thin_bone & (diff_pos < th_thin)] = 0
    gain[thin_bone & (diff_pos >= th_thin)] = 0.1 / bone_strength

    soft = (pre > -100) & (pre <= 100)
    gain[soft & (diff_pos < 20 / vessel_sensitivity)] = 0.1
    gain[(pre > -100) & (pre < 60) & (diff_pos > 50 * vessel_sensitivity)] = 1.0

    return diff_pos * np.clip(gain, 0, 1.5)


def quality_subtraction_cached(pre, post_aligned, bone_strength, vessel_sensitivity, cache):
    diff_pos = np.clip(post_aligned - pre, 0, None)
    gain = np.ones_like(diff_pos, dtype=np.float32)
    
    # ★ 修复：边缘清零 ★
    gain[~cache.body] = 0
    gain[pre < -500] = 0
    gain[pre > 500] = 0

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
    if cache.petrous is not None:
        exclude |= cache.petrous
    if cache.air_bone is not None:
        exclude |= cache.air_bone
    med_bone = (pre > 180) & (pre <= 300) & ~exclude
    gain[med_bone & (diff_pos < 60 * bone_strength)] = 0
    gain[med_bone & (diff_pos >= 60 * bone_strength) & (diff_pos < 120 * bone_strength)] = 0.15 / bone_strength
    gain[med_bone & (diff_pos >= 120 * bone_strength)] = 0.3 / bone_strength

    thin_only = cache.thin_bone & ~exclude
    th_thin = 80 * bone_strength
    gain[thin_only & (diff_pos < th_thin)] = 0
    gain[thin_only & (diff_pos >= th_thin)] = 0.1 / bone_strength

    scalp_mask = cache.scalp if cache.scalp is not None else np.zeros_like(pre, dtype=bool)
    ab_mask = cache.air_bone if cache.air_bone is not None else np.zeros_like(pre, dtype=bool)
    vessel_mask = (pre > -100) & (pre < 60) & (diff_pos > 50 * vessel_sensitivity) & ~scalp_mask & ~ab_mask
    gain[vessel_mask] = 1.0

    return diff_pos * np.clip(gain, 0, 1.5)


# ════════════════════════════════════════════════════════════════════
# ★★★ 轻量级优化 ★★★
# ════════════════════════════════════════════════════════════════════
def detect_ica_protection_zone_precise(pre_s, diff_s, body_mask):
    """
    ★ 修复版ICA保护区检测 ★
    - 必须在body内
    - 岩骨周围的高信号
    """
    dense_bone = pre_s > 400
    if dense_bone.sum() < 50:
        return np.zeros_like(pre_s, dtype=bool)
    
    petrous = ndimage.binary_dilation(dense_bone, iterations=5)
    ica_candidate = petrous & (diff_s > 80) & body_mask
    
    body_eroded = ndimage.binary_erosion(body_mask, iterations=10)
    ica_candidate = ica_candidate & body_eroded
    
    if ica_candidate.sum() > 0:
        return ndimage.binary_dilation(ica_candidate, iterations=2)
    
    return np.zeros_like(pre_s, dtype=bool)


def compute_adaptive_threshold(pre_s, diff_s, bone_mask, bone_strength, window_size=31):
    """计算局部自适应阈值"""
    bone_diff = np.where(bone_mask, diff_s, 0).astype(np.float32)
    bone_float = bone_mask.astype(np.float32)

    local_sum = ndimage.uniform_filter(bone_diff, size=window_size)
    local_count = ndimage.uniform_filter(bone_float, size=window_size)
    local_count_safe = np.maximum(local_count, 1e-6)
    local_mean = local_sum / local_count_safe

    local_sq = ndimage.uniform_filter(bone_diff ** 2, size=window_size)
    local_var = local_sq / local_count_safe - local_mean ** 2
    local_std = np.sqrt(np.maximum(local_var, 0))

    k = 1.5 / bone_strength
    adaptive_threshold = local_mean + k * local_std
    
    min_thresh = 50 * bone_strength
    adaptive_threshold = np.maximum(adaptive_threshold, min_thresh)
    
    return np.where(bone_mask, adaptive_threshold, 1000)


def optimized_subtraction_light_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache):
    """★ 轻量级优化 v3 ★"""
    diff_pos = np.clip(post_aligned - pre, 0, None)
    bone_mask = pre > 150
    gain = np.ones_like(diff_pos, dtype=np.float32)
    
    gain[~cache.body] = 0
    gain[pre < -500] = 0
    gain[pre > 500] = 0
    
    ica_protect = detect_ica_protection_zone_precise(pre, diff_pos, cache.body)
    adaptive_thresh = compute_adaptive_threshold(pre, diff_pos, bone_mask, bone_strength)
    
    bone_low = bone_mask & (diff_pos < adaptive_thresh) & ~ica_protect
    gain[bone_low] = 0
    
    bone_medium = bone_mask & (diff_pos >= adaptive_thresh) & (diff_pos < adaptive_thresh * 1.5) & ~ica_protect
    gain[bone_medium] = 0.2 / bone_strength
    
    bone_high = bone_mask & (diff_pos >= adaptive_thresh * 1.5) & ~ica_protect
    gain[bone_high] = 0.4 / bone_strength
    
    soft_vessel = (pre > -100) & (pre < 60) & (diff_pos > 50 * vessel_sensitivity) & cache.body
    gain[soft_vessel] = 1.0
    
    gain[ica_protect & (diff_pos > 60)] = 0.85
    gain[~cache.body] = 0
    
    return diff_pos * np.clip(gain, 0, 1.5)


# ════════════════════════════════════════════════════════════════════
# ★★★ 标准级优化 ★★★
# ════════════════════════════════════════════════════════════════════
def safe_subpixel_refinement_v2(pre_vol, aligned_vol, log_cb=None, progress_cb=None, base_progress=0):
    total_slices = pre_vol.shape[0]
    refined_count = 0
    
    sample_errors = []
    for z in range(total_slices // 4, total_slices * 3 // 4, 15):
        pre_s = pre_vol[z]
        mov_s = aligned_vol[z]
        bone_mask = pre_s > 150
        if bone_mask.sum() > 500:
            mse = np.mean((pre_s[bone_mask] - mov_s[bone_mask]) ** 2)
            sample_errors.append(mse)
    
    if not sample_errors:
        return aligned_vol
    
    mean_error = np.mean(sample_errors)
    
    if mean_error < 1000:
        if log_cb:
            log_cb(f"  [亚像素] 配准良好(MSE={mean_error:.0f})，跳过")
        return aligned_vol
    
    if log_cb:
        log_cb(f"  [亚像素] MSE={mean_error:.0f}，精修中...")

    for z in range(total_slices):
        pre_s = pre_vol[z]
        mov_s = aligned_vol[z]
        bone_mask = pre_s > 150

        if bone_mask.sum() < 500:
            continue

        dy, dx = fft_phase_correlation_2d(pre_s, mov_s, max_shift=2)
        dy = np.clip(dy, -0.6, 0.6)
        dx = np.clip(dx, -0.6, 0.6)

        if abs(dy) > 0.15 or abs(dx) > 0.15:
            aligned_vol[z] = shift_image_2d(mov_s, dy, dx)
            refined_count += 1

        if progress_cb and z % 50 == 0:
            progress_cb(base_progress + int((z / total_slices) * 8))

    if log_cb:
        log_cb(f"  [亚像素] 校正 {refined_count} 层")

    return aligned_vol


def optimized_subtraction_standard_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy):
    result = optimized_subtraction_light_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache)
    
    diff_pos = np.clip(post_aligned - pre, 0, None)
    bone = pre > 200
    if bone.sum() > 100:
        bone_inner = ndimage.binary_erosion(bone, iterations=3)
        inner_surface = bone & ~bone_inner
        near_inner = ndimage.binary_dilation(inner_surface, iterations=4) & ~bone & cache.body
        moderate_enhance = (diff_pos > 35) & (diff_pos < 90)
        sinus = near_inner & moderate_enhance
        
        if sinus.sum() > 30:
            result[sinus] = np.maximum(result[sinus], diff_pos[sinus] * 0.35)
    
    return result


# ════════════════════════════════════════════════════════════════════
# ★★★ 深度级优化 ★★★
# ════════════════════════════════════════════════════════════════════
def frangi_2d_fast(image, sigmas):
    vesselness = np.zeros_like(image, dtype=np.float32)
    for sigma in sigmas:
        smoothed = ndimage.gaussian_filter(image.astype(np.float64), sigma=sigma)
        Hxx = ndimage.gaussian_filter(smoothed, sigma=sigma, order=[0, 2])
        Hyy = ndimage.gaussian_filter(smoothed, sigma=sigma, order=[2, 0])
        Hxy = ndimage.gaussian_filter(smoothed, sigma=sigma, order=[1, 1])
        tmp = np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy ** 2)
        l1 = 0.5 * (Hxx + Hyy - tmp)
        l2 = 0.5 * (Hxx + Hyy + tmp)
        swap = np.abs(l1) > np.abs(l2)
        l1_s = np.where(swap, l2, l1)
        l2_s = np.where(swap, l1, l2)
        valid = l2_s < 0
        Rb = np.abs(l1_s) / (np.abs(l2_s) + 1e-10)
        S = np.sqrt(l1_s ** 2 + l2_s ** 2)
        c = S.max() * 0.5 if S.max() > 0 else 1.0
        v = np.exp(-Rb ** 2 / 0.5) * (1 - np.exp(-S ** 2 / (2 * c ** 2)))
        v *= sigma ** 2
        v[~valid] = 0
        vesselness = np.maximum(vesselness, v.astype(np.float32))
    return vesselness


def frangi_3d_fast(diff_vol, spacing, log_cb=None, progress_cb=None, base_progress=0):
    min_sp = min(spacing[1], spacing[2])
    sigmas = [0.8 / min_sp, 1.4 / min_sp]
    sigmas = [max(0.5, min(2.0, s)) for s in sigmas]
    if log_cb:
        log_cb(f"  [Frangi] 尺度: {[f'{s:.1f}' for s in sigmas]}")

    result = np.zeros_like(diff_vol, dtype=np.float32)
    for z in range(diff_vol.shape[0]):
        if diff_vol[z].max() > 15:
            result[z] = frangi_2d_fast(diff_vol[z], sigmas)
        if progress_cb and z % 50 == 0:
            progress_cb(base_progress + int((z / diff_vol.shape[0]) * 10))

    vmax = result.max()
    if vmax > 0:
        result /= vmax
    if log_cb:
        log_cb(f"  [Frangi] 完成")
    return result


def shape_analysis_fast(result_vol, spacing, log_cb=None):
    if log_cb:
        log_cb("  [形状分析] 快速模式...")
    binary = result_vol > 12
    labeled, n = ndimage.label(binary)
    if n == 0:
        return result_vol
    if log_cb:
        log_cb(f"  [形状分析] {n} 个连通域")

    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    
    rejected = 0
    for i in range(1, n + 1):
        vol = sizes[i - 1] * voxel_vol
        if vol < 2:
            region = labeled == i
            result_vol[region] = 0
            rejected += 1
        elif vol < 8:
            region = labeled == i
            if result_vol[region].mean() < 25:
                result_vol[region] *= 0.4
                rejected += 1

    if log_cb:
        log_cb(f"  [形状分析] 处理 {rejected} 个")
    return result_vol


def optimized_subtraction_deep_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy, frangi_map=None):
    result = optimized_subtraction_standard_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy)

    if frangi_map is not None and frangi_map.max() > 0:
        frangi_norm = frangi_map / (frangi_map.max() + 1e-6)
        high_frangi = frangi_norm > 0.3
        boost = 1 + 0.2 * frangi_norm
        result = result * np.where(high_frangi & cache.body, boost, 1.0)

    return result


# ════════════════════════════════════════════════════════════════════
# 后处理与血管缩放（★ 彻底重构修复版本 ★）
# ════════════════════════════════════════════════════════════════════
def clean_bone_edges_cached(image, pre_image, cache, mode='fast'):
    result = image.copy()
    edge = cache.bone_edge
    result[edge & (image < 40)] = 0
    result[edge & (image >= 40) & (image < 80)] *= 0.3

    if mode == 'quality':
        if cache.ab_edge is not None:
            result[cache.ab_edge & (image < 60)] = 0
            result[cache.ab_edge & (image >= 60) & (image < 100)] *= 0.2
        if cache.petrous_edge is not None:
            ica = cache.ica_protect if cache.ica_protect is not None else np.zeros_like(image, dtype=bool)
            pe_no_ica = cache.petrous_edge & ~ica
            result[pe_no_ica & (image < 80)] = 0
            result[pe_no_ica & (image >= 80) & (image < 120)] *= 0.15
    return result


def edge_preserving_smooth(image, pre_image, sigma=0.7):
    edges = np.abs(ndimage.sobel(pre_image.astype(np.float32)))
    edge_norm = edges / (edges.max() + 1e-6)
    smooth_h = ndimage.gaussian_filter(image, sigma * 1.5)
    smooth_l = ndimage.gaussian_filter(image, sigma * 0.3)
    return smooth_h * (1 - edge_norm) + smooth_l * edge_norm


def apply_3d_cleanup(volume, min_volume=100, log_cb=None):
    pos_vox = volume[volume > 0]
    if pos_vox.size == 0:
        return volume
    fg_thresh = max(10.0, float(np.percentile(pos_vox, 10)))
    del pos_vox
    force_gc()

    mask = volume > fg_thresh
    labeled, n = ndimage.label(mask)
    labeled = labeled.astype(np.int32)
    del mask
    force_gc()

    if n == 0:
        return volume
    if log_cb:
        log_cb(f"  连通域: {n}")

    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes[1:] >= min_volume
    del sizes
    force_gc()

    for z in range(volume.shape[0]):
        chunk_labels = labeled[z]
        chunk_vol = volume[z]
        noise = ~keep[chunk_labels] & (chunk_labels > 0)
        chunk_vol[noise & (chunk_vol < 60)] = 0
        chunk_vol[noise & (chunk_vol >= 60)] *= 0.3

    del labeled, keep
    force_gc()
    return volume


def apply_vessel_scaling(volume, spacing, split_mm, small_factor, large_factor, log_cb=None):
    """
    ★ 修复版血管缩放模块 ★
    解决原版无法缩小/移除血管的问题，支持极限操作(取0时彻底剔除)，让缩放变换顺滑连贯。
    """
    if small_factor == 1.0 and large_factor == 1.0:
        return volume

    # 1. 提取所有血管的初始 mask
    mask = volume > 20
    if not np.any(mask):
        return volume

    if log_cb:
        log_cb(f"  [血管调节] 界限: {split_mm}mm, 小血管: {small_factor}x, 大血管: {large_factor}x")

    # 2. 区分大、小血管
    r_thresh = split_mm / 2.0
    min_sp = min(spacing) if min(spacing) > 0 else 1.0
    erode_iters = max(1, int(np.ceil(r_thresh / min_sp)))

    struct = ndimage.generate_binary_structure(3, 1)
    large_core = mask.copy()
    for _ in range(erode_iters):
        large_core = ndimage.binary_erosion(large_core, structure=struct)

    # 还原大血管边界（稍微限制膨胀次数，防侵占相连小血管太多）
    large_mask = large_core.copy()
    del large_core
    for _ in range(erode_iters + 1):
        next_mask = ndimage.binary_dilation(large_mask, structure=struct) & mask
        if np.array_equal(next_mask, large_mask):
            break
        large_mask = next_mask

    small_mask = mask & ~large_mask
    del mask
    force_gc()

    # 安全的局部区域生长
    def safe_grow(vol, base_mask, new_region):
        for z in range(0, vol.shape[0], 32):
            z1, z2 = max(0, z - 1), min(vol.shape[0], z + 33)
            sub_vol = vol[z1:z2] * base_mask[z1:z2]
            max_f = ndimage.maximum_filter(sub_vol, size=3)
            valid_z1 = 1 if z > 0 else 0
            z_len = min(32, vol.shape[0] - z)
            target = new_region[z:z + z_len]
            chunk = max_f[valid_z1:valid_z1 + z_len]
            vol[z:z + z_len][target] = chunk[target]

    def scale_vessels(vol, target_mask, factor):
        if factor == 1.0:
            return
            
        if factor == 0.0:
            # ★ 彻底消除目标血管 ★
            vol[target_mask] = 0
            return
            
        if factor < 1.0:
            # ★ 缩小（腐蚀/细化）★
            iters = max(1, int(round((1.0 - factor) * 3)))
            shrunk = target_mask.copy()
            for _ in range(iters):
                shrunk = ndimage.binary_erosion(shrunk, structure=struct)
            
            removed = target_mask & ~shrunk
            # 将被去掉的外层清零，实现物理变细
            vol[removed] = 0
            
            # 伴随轻微信号衰减，让极小因子(如0.3)具有慢慢消失的视觉效果
            if factor < 0.8:
                vol[shrunk] *= max(0.3, factor)
            return
            
        if factor > 1.0:
            # ★ 放大（膨胀）★
            iters = max(1, int(round((factor - 1.0) * 2)))
            grow = target_mask.copy()
            for _ in range(iters):
                grow = ndimage.binary_dilation(grow, structure=struct)
            new_region = grow & ~target_mask
            if np.any(new_region):
                safe_grow(vol, target_mask, new_region)
            return

    # 3. 分别处理小血管和大血管 (先小后大，避免放大后的大血管被小血管处理覆盖)
    if small_factor != 1.0 and np.any(small_mask):
        scale_vessels(volume, small_mask, small_factor)

    if large_factor != 1.0 and np.any(large_mask):
        scale_vessels(volume, large_mask, large_factor)

    del small_mask, large_mask
    force_gc()
    return volume


# ════════════════════════════════════════════════════════════════════
# 处理线程
# ════════════════════════════════════════════════════════════════════
class ProcessThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, data_dir, pre_uid, post_uid, output_dir, options):
        super().__init__()
        self.data_dir = data_dir
        self.pre_uid = pre_uid
        self.post_uid = post_uid
        self.output_dir = output_dir
        self.options = options
        self.cancelled = False
        self.result_vol = None
        self.pre_vol = None
        self._pre_files = None

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            t0 = time.time()
            opt = self.options
            mode_name = "精细" if opt['quality_mode'] else "快速"
            opt_level = opt.get('optimization_level', 'none')
            opt_name = OPTIMIZATION_LEVELS.get(opt_level, {}).get('name', '无')

            self.log.emit("=" * 55)
            self.log.emit(f"CTA减影 v2.9.2 | {mode_name}模式 | 优化:{opt_name}")
            self.log.emit("=" * 55)
            
            self.log.emit(f"  骨骼抑制: {opt['bone_strength']:.1f}")
            self.log.emit(f"  血管敏感度: {opt['vessel_sensitivity']:.1f}")
            self.log.emit(f"  血管增强: {opt['vessel_enhance']:.1f}")
            self.log.emit(f"  最小血管: {opt['min_vessel_size']}")
            self.log.emit(f"  内存: {get_memory_mb():.0f} MB")

            # 1. 读取
            self.log.emit("\n[1/5] 读取...")
            reader = sitk.ImageSeriesReader()
            pre_files = reader.GetGDCMSeriesFileNames(self.data_dir, self.pre_uid)
            post_files = reader.GetGDCMSeriesFileNames(self.data_dir, self.post_uid)
            if not pre_files or not post_files:
                self.finished_signal.emit(False, "读取失败")
                return

            self._pre_files = list(pre_files)
            pre_img = sitk.ReadImage(pre_files, sitk.sitkInt16)
            post_img = sitk.ReadImage(post_files, sitk.sitkInt16)
            spacing = pre_img.GetSpacing()
            spacing_zyx = (spacing[2], spacing[1], spacing[0])

            self.log.emit(f"  尺寸: {pre_img.GetSize()}")
            self.progress.emit(10)

            # 2. 配准
            self.log.emit("\n[2/5] 3D配准...")
            aligned_img = execute_3d_registration(pre_img, post_img, self.log.emit, self.progress.emit)
            del post_img
            force_gc()
            if self.cancelled:
                return

            # 3. 优化配准
            self.log.emit("\n[3/5] 配准优化...")
            pre_vol = sitk.GetArrayFromImage(pre_img).astype(np.float32)
            del pre_img
            force_gc()
            aligned_vol = sitk.GetArrayFromImage(aligned_img).astype(np.float32)
            del aligned_img
            force_gc()

            if opt_level in ['standard', 'deep']:
                aligned_vol = safe_subpixel_refinement_v2(pre_vol, aligned_vol, self.log.emit, self.progress.emit, 30)
                force_gc()

            self.progress.emit(40)

            # Frangi
            frangi_vol = None
            if opt_level == 'deep':
                self.log.emit("  [深度] Frangi增强...")
                temp_diff = np.clip(aligned_vol - pre_vol, 0, None)
                frangi_vol = frangi_3d_fast(temp_diff, spacing_zyx, self.log.emit, self.progress.emit, 40)
                del temp_diff
                force_gc()

            self.progress.emit(50)
            if self.cancelled:
                return

            # 4. 减影
            self.log.emit("\n[4/5] 减影...")
            total = pre_vol.shape[0]
            os.makedirs(self.output_dir, exist_ok=True)
            series_uid = generate_uid()
            mode_str = 'quality' if opt['quality_mode'] else 'fast'
            spacing_xy = (spacing_zyx[1], spacing_zyx[2])

            for z in range(total):
                if self.cancelled:
                    return
                pre_s = pre_vol[z]
                post_s = aligned_vol[z]

                if pre_s.max() > -500:
                    diff_pos = np.clip(post_s - pre_s, 0, None)
                    cache = precompute_slice_cache(pre_s, diff_pos, opt['quality_mode'])
                    equip = detect_equipment(pre_s, post_s, body=cache.body)

                    if opt_level == 'none':
                        if opt['quality_mode']:
                            res = quality_subtraction_cached(pre_s, post_s, opt['bone_strength'], opt['vessel_sensitivity'], cache)
                        else:
                            res = fast_subtraction(pre_s, post_s, opt['bone_strength'], opt['vessel_sensitivity'], body=cache.body)
                    elif opt_level == 'light':
                        res = optimized_subtraction_light_v3(pre_s, post_s, opt['bone_strength'], opt['vessel_sensitivity'], cache)
                    elif opt_level == 'standard':
                        res = optimized_subtraction_standard_v3(pre_s, post_s, opt['bone_strength'], opt['vessel_sensitivity'], cache, spacing_xy)
                    else:
                        frangi_s = frangi_vol[z] if frangi_vol is not None else None
                        res = optimized_subtraction_deep_v3(pre_s, post_s, opt['bone_strength'], opt['vessel_sensitivity'], cache, spacing_xy, frangi_s)

                    res[equip] = 0
                    res[~cache.body] = 0

                    if opt['clean_bone_edges']:
                        res = clean_bone_edges_cached(res, pre_s, cache, mode_str)
                    if opt['smooth_sigma'] > 0:
                        res = edge_preserving_smooth(res, pre_s, opt['smooth_sigma'])

                    aligned_vol[z] = res * opt['vessel_enhance']
                else:
                    aligned_vol[z] = 0

                if z % 50 == 0:
                    self.progress.emit(50 + int((z / total) * 25))

            if frangi_vol is not None:
                del frangi_vol
                force_gc()

            result_vol = aligned_vol

            # 3D后处理
            self.log.emit("\n[*] 3D后处理...")
            min_vol = opt['min_vessel_size'] * 20
            result_vol = apply_3d_cleanup(result_vol, min_volume=min_vol, log_cb=self.log.emit)
            self.progress.emit(80)

            if opt_level == 'deep':
                result_vol = shape_analysis_fast(result_vol, spacing_zyx, self.log.emit)
                force_gc()

            if opt.get('vessel_scale_enable', False):
                self.log.emit("  血管调节...")
                result_vol = apply_vessel_scaling(result_vol, spacing_zyx, opt['vessel_split_mm'], opt['small_vessel_factor'], opt['large_vessel_factor'], self.log.emit)

            self.progress.emit(85)
            self.result_vol = result_vol
            self.pre_vol = pre_vol

            # 5. 写出
            self.log.emit("\n[5/5] 写入...")
            g_min = float(result_vol.min())
            g_max = float(result_vol.max())
            g_slope = (g_max - g_min) / 4095.0 if g_max > g_min else 1.0

            for z, fp in enumerate(self._pre_files[:total]):
                if self.cancelled:
                    return
                ds = pydicom.dcmread(fp)
                pix = result_vol[z]
                inst = ds.get('InstanceNumber', z + 1)
                out_path = os.path.join(self.output_dir, f"SUB_{inst:04d}.dcm")

                if g_max > g_min:
                    norm = (pix - g_min) / (g_max - g_min)
                    pix_int = (norm * 4095).astype(np.int16)
                else:
                    pix_int = np.zeros_like(pix, dtype=np.int16)

                new_ds = ds.copy()
                new_ds.PixelData = pix_int.tobytes()
                new_ds.BitsAllocated = 16
                new_ds.BitsStored = 16
                new_ds.HighBit = 15
                new_ds.PixelRepresentation = 1
                new_ds.SamplesPerPixel = 1
                new_ds.PhotometricInterpretation = 'MONOCHROME2'
                for tag in ['LossyImageCompression', 'LossyImageCompressionRatio', 'LossyImageCompressionMethod']:
                    if hasattr(new_ds, tag):
                        delattr(new_ds, tag)
                new_ds.RescaleSlope = g_slope
                new_ds.RescaleIntercept = g_min
                new_ds.WindowCenter = opt['wc']
                new_ds.WindowWidth = opt['ww']
                new_ds.SeriesDescription = f"CTA Sub v2.9.2 [{opt_name}]"
                new_ds.SeriesInstanceUID = series_uid
                new_ds.SOPInstanceUID = generate_uid()
                new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                new_ds.save_as(out_path, write_like_original=False)

                del ds, new_ds, pix_int
                if z % 50 == 0:
                    self.progress.emit(85 + int((z / total) * 15))

            self.progress.emit(100)
            elapsed = time.time() - t0
            self.log.emit(f"\n{'=' * 55}")
            self.log.emit(f"✅ 完成! 耗时: {elapsed:.1f}s | 内存: {get_memory_mb():.0f} MB")
            self.log.emit(f"保存: {self.output_dir}")
            self.finished_signal.emit(True, f"完成!\n耗时: {elapsed:.1f}秒\n优化: {opt_name}")

        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))


# ════════════════════════════════════════════════════════════════════
# 主界面
# ════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.all_series = {}
        self.selected_pre = None
        self.selected_post = None
        self.recommendations = None
        self.proc_thread = None
        self._all_presets = load_all_presets()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("头颅CTA减影-whw 版 v2.9.2")
        self.setMinimumSize(1000, 900)
        self.setStyleSheet("""
        QMainWindow,QWidget{background:#F4F5F7;font-family:"Microsoft YaHei";font-size:9pt;}
        QGroupBox{background:#fff;border:1px solid #D0D5DE;border-radius:6px;margin-top:10px;padding:10px;font-weight:600;}
        QGroupBox::title{subcontrol-origin:margin;left:10px;top:0;padding:0 4px;background:#fff;color:#1A6FBF;}
        QTextEdit{background:#1C1E26;color:#C8D0E0;border-radius:4px;font-family:Consolas;font-size:8.5pt;}
        QProgressBar{
            background:#D0D5DE;
            border:none;
            border-radius:4px;
            height:18px;
            text-align:center;
            color:white;
            font-weight:bold;
        }
        QProgressBar::chunk{
            background:#1A6FBF;
            border-radius:4px;
            text-align:center;
        }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)

        # 标题
        header = QWidget()
        header.setStyleSheet("background:#1A6FBF;border-radius:6px;")
        header.setMinimumHeight(44)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(12, 6, 12, 6)
        title = QLabel("头颅CTA减影-whw 版 V2.9.2")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setStyleSheet("color:white;background:transparent;")
        h_lay.addWidget(title)
        h_lay.addStretch()
        root.addWidget(header)

        # 1. 目录
        dir_grp = QGroupBox("1 数据目录")
        dir_lay = QHBoxLayout(dir_grp)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("选择DICOM文件夹...")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(lambda: self.data_dir_edit.setText(QFileDialog.getExistingDirectory(self) or self.data_dir_edit.text()))
        self.scan_btn = QPushButton("扫描")
        self.scan_btn.setStyleSheet("background:#1A6FBF;color:white;border:none;border-radius:4px;padding:6px 16px;")
        self.scan_btn.clicked.connect(self.scan_directory)
        dir_lay.addWidget(self.data_dir_edit)
        dir_lay.addWidget(browse_btn)
        dir_lay.addWidget(self.scan_btn)
        root.addWidget(dir_grp)

        # 2. 配对
        ser_grp = QGroupBox("2 序列配对")
        ser_lay = QVBoxLayout(ser_grp)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("平扫:"))
        self.pre_combo = QComboBox()
        r1.addWidget(self.pre_combo, 1)
        r1.addWidget(QLabel("增强:"))
        self.post_combo = QComboBox()
        r1.addWidget(self.post_combo, 1)
        ser_lay.addLayout(r1)
        r2 = QHBoxLayout()
        self.analyze_btn = QPushButton("🔍 分析特征")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_params)
        r2.addWidget(self.analyze_btn)
        r2.addStretch()
        ser_lay.addLayout(r2)
        root.addWidget(ser_grp)

        # 3. 输出
        out_grp = QGroupBox("3 输出目录")
        out_lay = QHBoxLayout(out_grp)
        self.out_edit = QLineEdit()
        out_btn = QPushButton("...")
        out_btn.clicked.connect(lambda: self.out_edit.setText(QFileDialog.getExistingDirectory(self) or self.out_edit.text()))
        out_lay.addWidget(self.out_edit)
        out_lay.addWidget(out_btn)
        root.addWidget(out_grp)

        # 4. 模式
        mode_grp = QGroupBox("4 处理模式与优化级别")
        mode_lay = QVBoxLayout(mode_grp)

        mr = QHBoxLayout()
        self.fast_radio = QRadioButton("⚡ 快速（适合普通扫描）")
        self.quality_radio = QRadioButton("✨ 精细（适合质量差的扫描）")
        self.fast_radio.setChecked(True)
        mr.addWidget(self.fast_radio)
        mr.addWidget(self.quality_radio)
        mr.addStretch()
        mode_lay.addLayout(mr)

        self.mode_label = QLabel("")
        mode_lay.addWidget(self.mode_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        mode_lay.addWidget(line)

        self.opt_group = QButtonGroup(self)
        self.opt_none = QRadioButton("无优化（正常）")
        self.opt_none.setChecked(True)
        self.opt_none.setStyleSheet("color:#222222;font-weight:normal;")
        self.opt_group.addButton(self.opt_none)
        mode_lay.addWidget(self.opt_none)

        self.opt_light = QRadioButton("轻量级 - 自适应阈值+ICA保护（速度延长很少）")
        self.opt_light.setStyleSheet("color:#222222;font-weight:normal;")
        self.opt_group.addButton(self.opt_light)
        mode_lay.addWidget(self.opt_light)

        self.opt_standard = QRadioButton("标准级 - +亚像素+静脉窦（速度有一定减慢）")
        self.opt_standard.setStyleSheet("color:#222222;font-weight:normal;")
        self.opt_group.addButton(self.opt_standard)
        mode_lay.addWidget(self.opt_standard)

        self.opt_deep = QRadioButton("深度级 - +Frangi+形状分析（速度慢，吃内存，低配慎用）")
        self.opt_deep.setStyleSheet("color:#222222;font-weight:normal;")
        self.opt_group.addButton(self.opt_deep)
        mode_lay.addWidget(self.opt_deep)

        root.addWidget(mode_grp)

        # 5. 参数
        param_grp = QGroupBox("5 参数")
        param_lay = QVBoxLayout(param_grp)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("预设:"))
        self.preset_combo = QComboBox()
        for name in self._all_presets:
            self.preset_combo.addItem(name, self._all_presets[name])
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self.apply_preset)
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(apply_btn)
        self.preset_status = QLabel("")
        self.preset_status.setStyleSheet("color:#27AE60;")
        preset_row.addWidget(self.preset_status)
        preset_row.addStretch()
        param_lay.addLayout(preset_row)

        p_lay = QHBoxLayout()
        p_lay.addWidget(QLabel("骨骼抑制:"))
        self.bone_slider = QSlider(Qt.Horizontal)
        self.bone_slider.setRange(5, 25)
        self.bone_slider.setValue(12)
        self.bone_label = QLabel("1.2")
        self.bone_slider.valueChanged.connect(lambda v: self.bone_label.setText(f"{v/10:.1f}"))
        p_lay.addWidget(self.bone_slider)
        p_lay.addWidget(self.bone_label)

        p_lay.addWidget(QLabel(" 血管增强:"))
        self.enhance = QDoubleSpinBox()
        self.enhance.setRange(0.5, 5.0)
        self.enhance.setValue(2.0)
        p_lay.addWidget(self.enhance)

        self.clean_check = QCheckBox("去骨边")
        self.clean_check.setChecked(True)
        p_lay.addWidget(self.clean_check)

        p_lay.addWidget(QLabel(" 降噪:"))
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 1.5)
        self.smooth.setValue(0.7)
        p_lay.addWidget(self.smooth)

        p_lay.addWidget(QLabel(" 窗:"))
        self.wc = QSpinBox()
        self.wc.setRange(-500, 1000)
        self.wc.setValue(200)
        self.ww = QSpinBox()
        self.ww.setRange(10, 2000)
        self.ww.setValue(400)
        p_lay.addWidget(self.wc)
        p_lay.addWidget(self.ww)
        p_lay.addStretch()
        param_lay.addLayout(p_lay)

        vs_lay = QHBoxLayout()
        self.scale_check = QCheckBox("血管缩放")
        vs_lay.addWidget(self.scale_check)
        vs_lay.addWidget(QLabel("界限（mm）:"))
        self.split_mm = QDoubleSpinBox()
        self.split_mm.setRange(0.5, 5.0)
        self.split_mm.setValue(1.5)
        vs_lay.addWidget(self.split_mm)
        vs_lay.addWidget(QLabel("小:"))
        self.small_factor = QDoubleSpinBox()
        self.small_factor.setRange(0.0, 5.0)
        self.small_factor.setValue(1.0)
        vs_lay.addWidget(self.small_factor)
        vs_lay.addWidget(QLabel("大:"))
        self.large_factor = QDoubleSpinBox()
        self.large_factor.setRange(0.0, 5.0)
        self.large_factor.setValue(1.0)
        vs_lay.addWidget(self.large_factor)
        vs_lay.addStretch()
        param_lay.addLayout(vs_lay)
        root.addWidget(param_grp)

        # 操作
        act_lay = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始")
        self.start_btn.setStyleSheet("background:#1A6FBF;color:white;border-radius:5px;font-weight:700;font-size:11pt;padding:0 24px;height:34px;")
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

        # 日志
        log_grp = QGroupBox("日志")
        log_lay = QVBoxLayout(log_grp)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_lay.addWidget(self.log_text)
        root.addWidget(log_grp)

        self.log("头颅CTA减影-whw 版 v2.9.2")
        self.log("=" * 50)
        self.log("⚡ 快速模式: 扫描质量好时使用，速度快")
        self.log("✨ 精细模式: 复杂情况使用，减少残留和噪点")
        self.log("=" * 50)
        self.log("")
        self.log("使用步骤:")
        self.log("  1. 选择DICOM目录，点击「扫描」")
        self.log("  2. 确认序列配对")
        self.log("  3. 点击「分析特征」（自动推荐模式和参数）")
        self.log("  4. 选择处理模式和优化级别，点击「开始」")
        self.log("  5. 血管缩放：界限以上、以下的血管缩放倍数，取0时血管彻底消失")

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        QApplication.processEvents()

    def _get_opt_level(self):
        if self.opt_light.isChecked():
            return 'light'
        elif self.opt_standard.isChecked():
            return 'standard'
        elif self.opt_deep.isChecked():
            return 'deep'
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
            
            if self.recommendations is None:
                self.recommendations = {}
            self.recommendations['vessel_sensitivity'] = cfg.get('vessel_sensitivity', 1.0)
            self.recommendations['min_vessel_size'] = cfg.get('min_vessel_size', 5)
            
            name = self.preset_combo.currentText()
            self.preset_status.setText(f"✓ 已应用: {name}")
            self.log(f"✅ 应用预设: {name}")
            self.log(f"   vessel_sensitivity={cfg.get('vessel_sensitivity', 1.0)}")
            self.log(f"   min_vessel_size={cfg.get('min_vessel_size', 5)}")

    def scan_directory(self):
        d = self.data_dir_edit.text()
        if not d:
            return
        if self.proc_thread:
            self.proc_thread.result_vol = None
            self.proc_thread.pre_vol = None
            self.proc_thread = None
            force_gc()
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
        self.all_series = all_series
        for s in sorted(all_series.values(), key=lambda x: x.series_number):
            st = s.contrast_status
            tag = '[C+]' if st is True else '[C-]' if st is False else ''
            txt = f"#{s.series_number:03d} | {s.file_count}张 | {tag} {s.series_description[:40]}"
            self.pre_combo.addItem(txt, s)
            self.post_combo.addItem(txt, s)
        if pairs:
            self.pre_combo.setCurrentIndex(self.pre_combo.findData(pairs[0][0]))
            self.post_combo.setCurrentIndex(self.post_combo.findData(pairs[0][1]))
        self.analyze_btn.setEnabled(True)
        if not self.out_edit.text():
            self.out_edit.setText(os.path.join(self.data_dir_edit.text(), "CTA_SUB_v292"))
        ref = pairs[0][0] if pairs else (cta_series[0] if cta_series else None)
        if ref:
            preset = detect_manufacturer_preset(ref.files)
            if preset:
                idx = self.preset_combo.findText(preset)
                if idx >= 0:
                    self.preset_combo.setCurrentIndex(idx)
                    self.apply_preset()
                self.log(f"🔍 检测设备: {preset}")

    def analyze_params(self):
        pre = self.pre_combo.currentData()
        post = self.post_combo.currentData()
        if not pre or not post or pre == post:
            return QMessageBox.warning(self, "提示", "请选择不同序列")
        self.selected_pre = pre
        self.selected_post = post
        self.analyze_btn.setEnabled(False)
        self.param_thread = ParamAnalyzeThread(pre.files, post.files)
        self.param_thread.progress.connect(self.progress.setValue)
        self.param_thread.log.connect(self.log)
        self.param_thread.finished_signal.connect(self.on_analyze_finished)
        self.param_thread.start()

    def on_analyze_finished(self, res):
        self.analyze_btn.setEnabled(True)
        if 'error' in res:
            return QMessageBox.warning(self, "错误", res['error'])
        
        if self.recommendations is None:
            self.recommendations = {}
        self.recommendations.update(res)
        
        self.bone_slider.setValue(int(res.get('bone_strength', 1.2) * 10))
        self.enhance.setValue(res.get('vessel_enhance', 2.0))
        self.clean_check.setChecked(res.get('clean_bone_edges', True))
        self.smooth.setValue(res.get('smooth_sigma', 0.7))
        self.wc.setValue(res.get('wc', 200))
        self.ww.setValue(res.get('ww', 400))
        
        mode = res.get('recommended_mode', 'fast')
        if mode == 'fast':
            self.fast_radio.setChecked(True)
        else:
            self.quality_radio.setChecked(True)
        self.mode_label.setText(f"质量: {res.get('quality_score', 0) * 100:.0f}%")
        self.start_btn.setEnabled(True)

    def start_processing(self):
        if not self.out_edit.text():
            return QMessageBox.warning(self, "提示", "输出目录为空")
        if self.proc_thread:
            self.proc_thread.result_vol = None
            self.proc_thread.pre_vol = None
            self.proc_thread = None
            force_gc()

        rec = self.recommendations or {}
        opt = {
            'quality_mode': self.quality_radio.isChecked(),
            'bone_strength': self.bone_slider.value() / 10.0,
            'vessel_sensitivity': rec.get('vessel_sensitivity', 1.0),
            'vessel_enhance': self.enhance.value(),
            'clean_bone_edges': self.clean_check.isChecked(),
            'min_vessel_size': rec.get('min_vessel_size', 5),
            'smooth_sigma': self.smooth.value(),
            'wc': self.wc.value(),
            'ww': self.ww.value(),
            'vessel_scale_enable': self.scale_check.isChecked(),
            'vessel_split_mm': self.split_mm.value(),
            'small_vessel_factor': self.small_factor.value(),
            'large_vessel_factor': self.large_factor.value(),
            'optimization_level': self._get_opt_level(),
        }

        opt_name = OPTIMIZATION_LEVELS.get(opt['optimization_level'], {}).get('name', '无')
        self.log(f"\n🚀 开始 | 优化: {opt_name}")

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.proc_thread = ProcessThread(
            self.data_dir_edit.text(),
            self.selected_pre.series_uid,
            self.selected_post.series_uid,
            self.out_edit.text(),
            opt)
        self.proc_thread.progress.connect(self.progress.setValue)
        self.proc_thread.log.connect(self.log)
        self.proc_thread.finished_signal.connect(self.on_finished)
        self.proc_thread.start()

    def cancel(self):
        if self.proc_thread:
            self.proc_thread.cancel()

    def on_finished(self, success, msg):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if success:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "错误", msg)


# ════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
