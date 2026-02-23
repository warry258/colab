"""
v2.9.8 32位极限防溢出 & 独立图标修复版
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[修复] 彻底消灭 scipy.ndimage.label 导致 32位系统 OOM 的崩溃问题
[修复] 增加 Windows 任务栏 AppUserModelID 声明，修复任务栏图标不显示问题
[修复] 增加 sys._MEIPASS 解析，使得 PyInstaller 打包后的内嵌图标能正常挂载到标题栏
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
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import time
from datetime import datetime
from collections import defaultdict
from scipy import ndimage

# 屏蔽 PyQt5 的底层弃用警告，保持控制台干净
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    QButtonGroup, QScrollArea, QSizePolicy, QDesktopWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap, QIcon

import pydicom
from pydicom.uid import generate_uid

print("头颅CTA减影-whw 版 v2.9.8")


# ════════════════════════════════════════════════════════════════════
# 工具函数 (包含 PyInstaller 资源路径解析)
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
    """获取资源的绝对路径，兼容开发环境与 PyInstaller 打包环境"""
    try:
        # PyInstaller 创建临时文件夹，将路径存入 _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ════════════════════════════════════════════════════════════════════
# 优化级别与预设
# ════════════════════════════════════════════════════════════════════
OPTIMIZATION_LEVELS = {
    "none": {"name": "无优化", "description": "原生模式"},
    "light": {"name": "轻量级", "description": "局部自适应 + ICA精确保护"},
    "standard": {"name": "标准级", "description": "+ 亚像素精修 + 静脉窦"},
    "deep": {"name": "深度级", "description": "+ Frangi增强"},
}

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
        "smooth_sigma": 0.7, "clean_bone_edges": True, "min_vessel_size": 5, "wc": 200, "ww": 400
    },
    "Siemens SOMATOM": {
        "bone_strength": 1.2, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": True, "min_vessel_size": 5, "wc": 200, "ww": 400
    },
    "GE Revolution/Discovery": {
        "bone_strength": 1.4, "vessel_sensitivity": 0.9, "vessel_enhance": 1.8,
        "smooth_sigma": 0.9, "clean_bone_edges": True, "min_vessel_size": 6, "wc": 220, "ww": 420
    },
    "Philips Brilliance/IQon": {
        "bone_strength": 1.0, "vessel_sensitivity": 1.1, "vessel_enhance": 2.2,
        "smooth_sigma": 0.5, "clean_bone_edges": True, "min_vessel_size": 4, "wc": 180, "ww": 380
    },
    "Canon Aquilion": {
        "bone_strength": 1.1, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": False, "min_vessel_size": 5, "wc": 200, "ww": 400
    },
    "联影 uCT": {
        "bone_strength": 1.3, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.8, "clean_bone_edges": True, "min_vessel_size": 5, "wc": 200, "ww": 400
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
    if not files: return None
    try:
        ds = pydicom.dcmread(files[0][1], stop_before_pixels=True)
        combined = (str(getattr(ds, "Manufacturer", "")) + " " + str(getattr(ds, "ManufacturerModelName", ""))).lower()
        for keywords, preset_name in _MANUFACTURER_MAP.items():
            if any(kw in combined for kw in keywords): return preset_name
    except: pass
    return None


# ════════════════════════════════════════════════════════════════════
# 序列扫描与分析
# ════════════════════════════════════════════════════════════════════
class SeriesInfo:
    def __init__(self):
        self.series_uid, self.series_number, self.series_description = "", 0, ""
        self.study_description, self.modality, self.body_part, self.protocol_name = "", "", "", ""
        self.acquisition_time, self.file_count, self.files = None, 0, []
        self.slice_thickness, self.image_shape, self.manufacturer = 0, (0, 0), ""
        self._contrast_status, self._contrast_cached = None, False

    @property
    def contrast_status(self):
        if not self._contrast_cached:
            desc = self.series_description.upper()
            pos = [r'\bC\+', r'\bC\s*\+', r'CE\b', r'CONTRAST', r'ENHANCED', r'POST', r'ARTERIAL', r'增强', r'动脉期']
            neg = [r'\bC-', r'\bC\s*-', r'\bNC\b', r'NON-CONTRAST', r'PLAIN', r'PRE\b', r'WITHOUT', r'平扫', r'非增强']
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
    except: return None

def scan_directory_for_series(directory, progress_callback=None, log_callback=None):
    all_files = list(Path(directory).rglob('*'))
    dicom_files = []
    for f in all_files:
        if f.is_file():
            try:
                with open(f, 'rb') as fp:
                    fp.seek(128)
                    if fp.read(4) == b'DICM': dicom_files.append(f)
            except: pass
    if not dicom_files: return {}
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
        if progress_callback: progress_callback(int((i + 1) / len(dicom_files) * 100))
    for s in series_dict.values(): s.files.sort(key=lambda x: x[0])
    return dict(series_dict)

def is_head_cta_series(series):
    desc = f"{series.series_description} {series.study_description} {series.protocol_name} {series.body_part}".upper()
    if any(kw in desc for kw in ['SCOUT', 'LOCALIZER', 'TOPOGRAM', '定位', 'LUNG', 'CHEST', '肺', '胸', 'CARDIAC', 'HEART', '心', 'ABDOMEN', 'LIVER', '腹']): return False
    if series.modality != 'CT' or series.file_count < 50: return False
    has_head = any(kw in desc for kw in ['HEAD', 'BRAIN', 'CEREBR', 'CRANIAL', '头', '颅', '脑', 'CAROTID'])
    has_cta = any(kw in desc for kw in ['CTA', 'ANGIO', 'C+', 'C-', '血管', '动脉'])
    return (has_head and has_cta) or (has_head and series.file_count >= 100)

def find_cta_pairs(series_dict):
    cta_series = [s for s in series_dict.values() if is_head_cta_series(s)]
    if len(cta_series) < 2: return [], cta_series
    groups = defaultdict(list)
    for s in cta_series: groups[(s.file_count, round(s.slice_thickness, 1))].append(s)
    pairs, used = [], set()
    for group in groups.values():
        enhanced, plain, unknown = [], [], []
        for s in group:
            st = s.contrast_status
            if st is True: enhanced.append(s)
            elif st is False: plain.append(s)
            else: unknown.append(s)
        for pre in plain:
            for post in enhanced:
                if pre.series_uid not in used and post.series_uid not in used:
                    pairs.append((pre, post))
                    used.update([pre.series_uid, post.series_uid])
                    break
            if pre.series_uid in used: break
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
            self.log.emit("=" * 55 + f"\n扫描: {self.directory}\n" + "=" * 55)
            series_dict = scan_directory_for_series(self.directory, self.progress.emit, self.log.emit)
            self.log.emit(f"\n找到 {len(series_dict)} 个序列\n")
            for s in sorted(series_dict.values(), key=lambda s: s.series_number):
                t = s.acquisition_time.strftime("%H:%M:%S") if s.acquisition_time else "--:--:--"
                m = "C+" if s.contrast_status is True else "C-" if s.contrast_status is False else "★ " if is_head_cta_series(s) else "  "
                self.log.emit(f"{m} #{s.series_number:3d} | {s.file_count:4d}张 | {t} | {s.series_description[:35]}")
            pairs, cta_series = find_cta_pairs(series_dict)
            if pairs:
                self.log.emit(f"\n★ 自动配对:\n  平扫: #{pairs[0][0].series_number} \n  增强: #{pairs[0][1].series_number}")
            self.progress.emit(100)
            self.finished_signal.emit(series_dict, pairs, cta_series)
        except Exception as e:
            self.log.emit(f"[错误] {e}")
            self.finished_signal.emit({}, [], [])


# ════════════════════════════════════════════════════════════════════
# 配准相关算法 
# ════════════════════════════════════════════════════════════════════
def fft_phase_correlation_2d(fixed, moving, max_shift=15):
    from numpy.fft import fft2, ifft2, fftshift
    margin = max(fixed.shape[0] // 6, 30)
    f_roi, m_roi = fixed[margin:-margin, margin:-margin].astype(np.float32), moving[margin:-margin, margin:-margin].astype(np.float32)
    window = np.outer(np.hanning(f_roi.shape[0]), np.hanning(f_roi.shape[1])).astype(np.float32)
    f1, f2 = fft2(f_roi * window), fft2(m_roi * window)
    correlation = np.real(fftshift(ifft2((f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10))))
    py, px = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy, dx = float(py - correlation.shape[0] // 2), float(px - correlation.shape[1] // 2)
    return float(np.clip(dy, -max_shift, max_shift)), float(np.clip(dx, -max_shift, max_shift))

def shift_image_2d(image, dy, dx):
    return ndimage.shift(image.astype(np.float32), [dy, dx], order=1, mode='constant', cval=0)

class ParamAnalyzeThread(QThread):
    progress, log, finished_signal = pyqtSignal(int), pyqtSignal(str), pyqtSignal(dict)
    def __init__(self, pre_files, post_files):
        super().__init__()
        self.pre_files, self.post_files = pre_files, post_files
    def run(self):
        try:
            self.log.emit("\n" + "=" * 55 + "\n智能参数分析\n" + "=" * 55)
            common = sorted(set(inst for inst, _ in self.pre_files) & set(inst for inst, _ in self.post_files))
            if not common:
                return self.finished_signal.emit({'error': '序列不匹配'})
            pre_dict = {i: p for i, p in self.pre_files}
            post_dict = {i: p for i, p in self.post_files}
            samples = [common[len(common) // 2]] if len(common) <= 10 else [common[len(common)//2 - 5], common[len(common)//2], common[len(common)//2 + 5]]
            all_chars = []
            for i, inst in enumerate(samples):
                pre_d = pydicom.dcmread(pre_dict[inst]).pixel_array.astype(np.float32)
                post_d = pydicom.dcmread(post_dict[inst]).pixel_array.astype(np.float32)
                dy, dx = fft_phase_correlation_2d(pre_d, post_d)
                post_aligned = shift_image_2d(post_d, dy, dx)
                diff = np.clip(post_aligned - pre_d, 0, None)
                bone, air = pre_d > 150, pre_d < -800
                noise = float(pre_d[air].std()) if air.sum() > 100 else 15.0
                strong = diff > 50
                vessel = float(diff[strong].mean()) if strong.sum() > 0 else 0.0
                all_chars.append({'noise': noise, 'vessel_signal': vessel})
                self.progress.emit(int((i + 1) / len(samples) * 100))
            
            avg = {k: float(np.mean([c[k] for c in all_chars])) for k in all_chars[0]}
            score = max(0, min(1, 1.0 - (0.2 if avg['noise'] > 20 else 0) - (0.2 if avg['vessel_signal'] < 40 else 0)))
            rec = {
                'bone_strength': 1.2, 'vessel_sensitivity': 1.0,
                'vessel_enhance': 2.0 if avg['vessel_signal'] < 60 else 1.8,
                'clean_bone_edges': True, 'min_vessel_size': 5, 'smooth_sigma': 0.7,
                'wc': 200, 'ww': 400, 'quality_score': score, 'recommended_mode': 'fast' if score >= 0.7 else 'quality'
            }
            self.log.emit(f"图像质量: {'优良' if score >= 0.7 else '一般'}")
            self.finished_signal.emit(rec)
        except Exception as e:
            self.finished_signal.emit({'error': str(e)})

def execute_3d_registration(fixed_img, moving_img, log_cb=None, progress_cb=None):
    fixed_size = fixed_img.GetSize()
    shrink_factors = [2 if fixed_size[0] >= 400 else 1, 2 if fixed_size[0] >= 400 else 1, 2 if fixed_size[2] >= 100 else 1]
    fixed_reg = sitk.Cast(sitk.Shrink(fixed_img, shrink_factors), sitk.sitkFloat32)
    moving_reg = sitk.Cast(sitk.Shrink(moving_img, shrink_factors), sitk.sitkFloat32)
    bone_mask = sitk.Cast(sitk.BinaryThreshold(fixed_reg, 200.0, 3000.0, 1, 0), sitk.sitkUInt8)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricFixedMask(bone_mask)
    R.SetInitialTransform(sitk.Euler3DTransform(sitk.Euler3DTransform().GetCenter()), inPlace=False)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.001, numberOfIterations=60, estimateLearningRate=R.Never)
    R.SetInterpolator(sitk.sitkLinear)
    if progress_cb: R.AddCommand(sitk.sitkIterationEvent, lambda: progress_cb(int((R.GetOptimizerIteration() / 60.0) * 20)))
    final_transform = R.Execute(fixed_reg, moving_reg)
    del fixed_reg, moving_reg, bone_mask, R
    force_gc()
    return sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, -1000.0, sitk.sitkFloat32)


# ════════════════════════════════════════════════════════════════════
# 基础掩膜与特征提取工具 
# ════════════════════════════════════════════════════════════════════
def create_body_mask(image):
    body = ndimage.binary_fill_holes(ndimage.binary_closing(image > -400, iterations=3))
    labeled, num = ndimage.label(body)
    if num > 0:
        sizes = ndimage.sum(body, labeled, range(1, num + 1))
        body = labeled == (np.argmax(sizes) + 1)
    return ndimage.binary_erosion(body, iterations=3)

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
        self.body, self.scalp, self.air_bone, self.petrous = None, None, None, None
        self.thin_bone, self.venous, self.petrous_edge = None, None, None
        self.bone_edge, self.ab_edge, self.ica_protect = None, None, None

def precompute_slice_cache(pre_s, diff_pos, quality_mode):
    cache = SliceCache()
    cache.body = create_body_mask(pre_s)
    bone = pre_s > 150
    grad_y, grad_x = ndimage.sobel(pre_s.astype(np.float32), axis=0), ndimage.sobel(pre_s.astype(np.float32), axis=1)
    gradient = np.sqrt(grad_y ** 2 + grad_x ** 2)
    cache.thin_bone = (bone & ~ndimage.binary_erosion(bone, iterations=2)) | (bone & (gradient > np.percentile(gradient[bone] if bone.sum() > 0 else gradient.ravel(), 70)))
    cache.bone_edge = ndimage.binary_dilation(bone, iterations=2) & ~ndimage.binary_erosion(bone, iterations=2)
    if not quality_mode: return cache

    cache.scalp = ndimage.binary_dilation((cache.body & ~ndimage.binary_erosion(cache.body, iterations=8)) & ((pre_s > -50) & (pre_s < 100)), iterations=2)
    air_d4, bone_d4 = ndimage.binary_dilation(pre_s < -200, iterations=4), ndimage.binary_dilation(bone, iterations=4)
    cache.air_bone = air_d4 & bone_d4 & ndimage.binary_dilation(bone, iterations=6)
    cache.ab_edge = air_d4 & bone_d4

    petrous = ndimage.binary_dilation((pre_s > 400) & ndimage.binary_dilation(gradient > 100, iterations=2), iterations=3)
    cache.petrous, cache.petrous_edge = petrous, petrous & ~ndimage.binary_erosion(petrous, iterations=2)
    
    ica_core = petrous & (pre_s < 150) & (diff_pos > 50)
    cache.ica_protect = ndimage.binary_dilation(ica_core, iterations=2) if ica_core.sum() > 0 else np.zeros_like(pre_s, dtype=bool)
    
    brain_dilated = ndimage.binary_dilation(ndimage.binary_erosion((pre_s > 20) & (pre_s < 60), iterations=3), iterations=6)
    cache.venous = (brain_dilated & ~ndimage.binary_erosion((pre_s > 20) & (pre_s < 60), iterations=3)) & ((diff_pos > 20) & (diff_pos < 80))
    return cache


# ════════════════════════════════════════════════════════════════════
# 减影算法核心分级
# ════════════════════════════════════════════════════════════════════
def fast_subtraction(pre, post_aligned, bone_strength=1.0, vessel_sensitivity=1.0, body=None):
    if body is None: body = create_body_mask(pre)
    diff_pos, gain = np.clip(post_aligned - pre, 0, None), np.ones_like(pre, dtype=np.float32)
    gain[~body], gain[pre < -500], gain[pre > 500] = 0, 0, 0
    high_bone, med_bone = (pre > 300) & (pre <= 500), (pre > 180) & (pre <= 300)
    gain[high_bone & (diff_pos < 120 * bone_strength)] = 0
    gain[high_bone & (diff_pos >= 120 * bone_strength)] = 0.1 / bone_strength
    gain[med_bone & (diff_pos < 60 * bone_strength)] = 0
    gain[med_bone & (diff_pos >= 60 * bone_strength) & (diff_pos < 120 * bone_strength)] = 0.15 / bone_strength
    gain[med_bone & (diff_pos >= 120 * bone_strength)] = 0.3 / bone_strength
    thin_bone = detect_thin_bone(pre)
    gain[thin_bone & (diff_pos < 80 * bone_strength)] = 0
    gain[thin_bone & (diff_pos >= 80 * bone_strength)] = 0.1 / bone_strength
    soft = (pre > -100) & (pre <= 100)
    gain[soft & (diff_pos < 20 / vessel_sensitivity)] = 0.1
    gain[(pre > -100) & (pre < 60) & (diff_pos > 50 * vessel_sensitivity)] = 1.0
    return diff_pos * np.clip(gain, 0, 1.5)

def quality_subtraction_cached(pre, post_aligned, bone_strength, vessel_sensitivity, cache):
    diff_pos = np.clip(post_aligned - pre, 0, None)
    gain = np.ones_like(diff_pos, dtype=np.float32)
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

def detect_ica_protection_zone_precise(pre_s, diff_s, body_mask):
    dense_bone = pre_s > 400
    if dense_bone.sum() < 50: return np.zeros_like(pre_s, dtype=bool)
    ica_candidate = ndimage.binary_dilation(dense_bone, iterations=5) & (diff_s > 80) & body_mask
    ica_candidate = ica_candidate & ndimage.binary_erosion(body_mask, iterations=10)
    return ndimage.binary_dilation(ica_candidate, iterations=2) if ica_candidate.sum() > 0 else np.zeros_like(pre_s, dtype=bool)

def compute_adaptive_threshold(pre_s, diff_s, bone_mask, bone_strength, window_size=31):
    b_diff, b_float = np.where(bone_mask, diff_s, 0).astype(np.float32), bone_mask.astype(np.float32)
    local_count_safe = np.maximum(ndimage.uniform_filter(b_float, size=window_size), 1e-6)
    local_mean = ndimage.uniform_filter(b_diff, size=window_size) / local_count_safe
    local_std = np.sqrt(np.maximum(ndimage.uniform_filter(b_diff ** 2, size=window_size) / local_count_safe - local_mean ** 2, 0))
    adaptive_threshold = np.maximum(local_mean + (1.5 / bone_strength) * local_std, 50 * bone_strength)
    return np.where(bone_mask, adaptive_threshold, 1000)

def optimized_subtraction_light_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache):
    diff_pos, bone_mask = np.clip(post_aligned - pre, 0, None), pre > 150
    gain = np.ones_like(diff_pos, dtype=np.float32)
    gain[~cache.body], gain[pre < -500], gain[pre > 500] = 0, 0, 0
    ica_protect = detect_ica_protection_zone_precise(pre, diff_pos, cache.body)
    adaptive_thresh = compute_adaptive_threshold(pre, diff_pos, bone_mask, bone_strength)
    gain[bone_mask & (diff_pos < adaptive_thresh) & ~ica_protect] = 0
    gain[bone_mask & (diff_pos >= adaptive_thresh) & (diff_pos < adaptive_thresh * 1.5) & ~ica_protect] = 0.2 / bone_strength
    gain[bone_mask & (diff_pos >= adaptive_thresh * 1.5) & ~ica_protect] = 0.4 / bone_strength
    gain[(pre > -100) & (pre < 60) & (diff_pos > 50 * vessel_sensitivity) & cache.body] = 1.0
    gain[ica_protect & (diff_pos > 60)] = 0.85
    gain[~cache.body] = 0
    return diff_pos * np.clip(gain, 0, 1.5)

def optimized_subtraction_standard_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy):
    result = optimized_subtraction_light_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache)
    diff_pos, bone = np.clip(post_aligned - pre, 0, None), pre > 200
    if bone.sum() > 100:
        sinus = (ndimage.binary_dilation(bone & ~ndimage.binary_erosion(bone, iterations=3), iterations=4) & ~bone & cache.body) & ((diff_pos > 35) & (diff_pos < 90))
        if sinus.sum() > 30: result[sinus] = np.maximum(result[sinus], diff_pos[sinus] * 0.35)
    return result

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
    if not sample_errors: return aligned_vol
    mean_error = np.mean(sample_errors)
    if mean_error < 1000:
        if log_cb: log_cb(f"  [亚像素] 配准良好(MSE={mean_error:.0f})，跳过")
        return aligned_vol
    if log_cb: log_cb(f"  [亚像素] MSE={mean_error:.0f}，精修中...")

    for z in range(total_slices):
        pre_s = pre_vol[z]
        mov_s = aligned_vol[z]
        bone_mask = pre_s > 150
        if bone_mask.sum() < 500: continue
        dy, dx = fft_phase_correlation_2d(pre_s, mov_s, max_shift=2)
        dy, dx = np.clip(dy, -0.6, 0.6), np.clip(dx, -0.6, 0.6)
        if abs(dy) > 0.15 or abs(dx) > 0.15:
            aligned_vol[z] = shift_image_2d(mov_s, dy, dx)
            refined_count += 1
        if progress_cb and z % 50 == 0: progress_cb(base_progress + int((z / total_slices) * 8))

    if log_cb: log_cb(f"  [亚像素] 校正 {refined_count} 层")
    return aligned_vol


# ════════════════════════════════════════════════════════════════════
# 深度级优化算法与内存安全分层统计工具
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

def optimized_subtraction_deep_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy, frangi_map=None):
    result = optimized_subtraction_standard_v3(pre, post_aligned, bone_strength, vessel_sensitivity, cache, spacing_xy)
    if frangi_map is not None and frangi_map.max() > 0:
        frangi_norm = frangi_map / (frangi_map.max() + 1e-6)
        high_frangi = frangi_norm > 0.3
        boost = 1 + 0.2 * frangi_norm
        result = result * np.where(high_frangi & cache.body, boost, 1.0)
    return result


# ════════════════════════════════════════════════════════════════════
# 2.5D 切片级滑窗降维算法 (32位彻底消灭 OOM)
# ════════════════════════════════════════════════════════════════════
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
    if log_cb: log_cb("  [+] 2.5D 切片连通清理 (32位0拷贝安全机制)...")
    
    fg_thresh = max(10.0, get_safe_percentile_nonzero(volume, 10, max_bin=4096))
    depth = volume.shape[0]
    
    # 滑动窗口掩膜：只需 3 张二值图 (约占 < 1 MB)
    mask_prev = None
    mask_curr = volume[0] > fg_thresh if depth > 0 else None
    
    for z in range(depth):
        mask_next = volume[z+1] > fg_thresh if z < depth - 1 else np.zeros_like(mask_curr)
        
        if mask_curr is not None and np.any(mask_curr):
            # 仅在单个切片做连通分析
            labeled, n = ndimage.label(mask_curr)
            if n > 0:
                overlap_mask = mask_next.copy()
                if mask_prev is not None:
                    overlap_mask |= mask_prev
                    
                sizes = np.bincount(labeled.ravel(), minlength=n+1)
                overlap_counts = np.bincount(labeled[overlap_mask], minlength=n+1)
                
                # 保留条件：2D连通面积达标，或者在Z轴与相邻层有血管连接
                keep = (sizes >= min_area) | (overlap_counts > 0)
                keep[0] = True
                
                noise = ~keep[labeled] & (labeled > 0)
                if np.any(noise):
                    chunk_vol = volume[z]
                    chunk_vol[noise & (chunk_vol < 60)] = 0
                    chunk_vol[noise & (chunk_vol >= 60)] *= 0.3
                    
                    # 实时更新当前层掩膜，防连带假阳性
                    mask_curr[noise] = False
        
        # 滑动窗口步进
        mask_prev = mask_curr
        mask_curr = mask_next
        
    force_gc()
    return volume

def shape_analysis_fast(volume, spacing, log_cb=None):
    if log_cb: log_cb("  [形状分析] 2.5D 滑动连通分析 (32位0拷贝安全机制)...")
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
                if mask_prev is not None:
                    overlap_mask |= mask_prev
                    
                sizes = np.bincount(labeled.ravel(), minlength=n+1)
                overlap_counts = np.bincount(labeled[overlap_mask], minlength=n+1)
                sums = np.bincount(labeled.ravel(), weights=volume[z].ravel(), minlength=n+1)
                
                areas = sizes * voxel_area
                isolated = overlap_counts == 0
                
                # 若为孤立噪点，且面积过小，则予以惩罚
                action1 = isolated & (areas < 1.0)
                
                means = np.zeros_like(sums)
                np.divide(sums, sizes, out=means, where=sizes>0)
                action2 = isolated & (areas < 3.0) & (means < 25) & ~action1
                
                act_z = np.zeros(n+1, dtype=np.uint8)
                act_z[action1] = 1
                act_z[action2] = 2
                
                actions_mapped = act_z[labeled]
                if np.any(actions_mapped > 0):
                    res_z = volume[z]
                    res_z[actions_mapped == 1] = 0
                    res_z[actions_mapped == 2] *= 0.4
                    mask_curr[actions_mapped == 1] = False
        
        mask_prev = mask_curr
        mask_curr = mask_next
        
    force_gc()
    return volume

def apply_vessel_scaling(volume, spacing, split_mm, small_factor, large_factor, log_cb=None):
    if small_factor == 1.0 and large_factor == 1.0: return volume
    if log_cb: log_cb("  [血管缩放] 2D切片级缩放 (消除3D矩阵复制崩溃风险)...")
    
    erode_iters = max(1, int(np.ceil((split_mm / 2.0) / (min(spacing) if min(spacing) > 0 else 1.0))))
    struct_2d = ndimage.generate_binary_structure(2, 1)
    
    # 彻底弃用3D形态学缩放，改为2D切片安全级
    for z in range(volume.shape[0]):
        chunk = volume[z]
        mask = chunk > 20
        if not np.any(mask): continue
        
        large_mask = ndimage.binary_erosion(mask, structure=struct_2d, iterations=erode_iters)
        large_mask = ndimage.binary_dilation(large_mask, structure=struct_2d, iterations=erode_iters + 1, mask=mask)
        small_mask = mask & ~large_mask
        
        # 小血管操作
        if small_factor != 1.0 and np.any(small_mask):
            if small_factor == 0.0:
                chunk[small_mask] = 0
            elif small_factor < 1.0:
                shrink_iters = max(1, int(round((1.0 - small_factor) * 3)))
                shrunk = ndimage.binary_erosion(small_mask, structure=struct_2d, iterations=shrink_iters)
                chunk[small_mask & ~shrunk] = 0
                if small_factor < 0.8: chunk[shrunk] *= max(0.3, small_factor)
            else:
                grow_iters = max(1, int(round((small_factor - 1.0) * 2)))
                grow = ndimage.binary_dilation(small_mask, structure=struct_2d, iterations=grow_iters)
                new_region = grow & ~small_mask
                if np.any(new_region):
                    max_f = ndimage.maximum_filter(chunk, size=3)
                    chunk[new_region] = max_f[new_region]
        
        # 大血管操作
        if large_factor != 1.0 and np.any(large_mask):
            if large_factor == 0.0:
                chunk[large_mask] = 0
            elif large_factor < 1.0:
                shrink_iters = max(1, int(round((1.0 - large_factor) * 3)))
                shrunk = ndimage.binary_erosion(large_mask, structure=struct_2d, iterations=shrink_iters)
                chunk[large_mask & ~shrunk] = 0
                if large_factor < 0.8: chunk[shrunk] *= max(0.3, large_factor)
            else:
                grow_iters = max(1, int(round((large_factor - 1.0) * 2)))
                grow = ndimage.binary_dilation(large_mask, structure=struct_2d, iterations=grow_iters)
                new_region = grow & ~large_mask
                if np.any(new_region):
                    max_f = ndimage.maximum_filter(chunk, size=3)
                    chunk[new_region] = max_f[new_region]
                    
    force_gc()
    return volume

def clean_bone_edges_cached(image, pre_image, cache, mode='fast'):
    result = image.copy()
    edge = cache.bone_edge
    result[edge & (image < 40)], result[edge & (image >= 40) & (image < 80)] = 0, result[edge & (image >= 40) & (image < 80)] * 0.3
    if mode == 'quality':
        if cache.ab_edge is not None:
            result[cache.ab_edge & (image < 60)], result[cache.ab_edge & (image >= 60) & (image < 100)] = 0, result[cache.ab_edge & (image >= 60) & (image < 100)] * 0.2
        if cache.petrous_edge is not None:
            pe_no_ica = cache.petrous_edge & ~(cache.ica_protect if cache.ica_protect is not None else np.zeros_like(image, dtype=bool))
            result[pe_no_ica & (image < 80)], result[pe_no_ica & (image >= 80) & (image < 120)] = 0, result[pe_no_ica & (image >= 80) & (image < 120)] * 0.15
    return result

def edge_preserving_smooth(image, pre_image, sigma=0.7):
    edges = np.abs(ndimage.sobel(pre_image.astype(np.float32)))
    edge_norm = edges / (edges.max() + 1e-6)
    return ndimage.gaussian_filter(image, sigma * 1.5) * (1 - edge_norm) + ndimage.gaussian_filter(image, sigma * 0.3) * edge_norm


# ════════════════════════════════════════════════════════════════════
# 主处理线程
# ════════════════════════════════════════════════════════════════════
class ProcessThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, pre_series, post_series, output_dir, options):
        super().__init__()
        self.pre_series = pre_series
        self.post_series = post_series
        self.output_dir = output_dir
        self.options = options
        self.cancelled = False
        self.preview_data = None 

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            t0 = time.time()
            opt = self.options
            opt_level = opt.get('optimization_level', 'none')
            opt_name = OPTIMIZATION_LEVELS.get(opt_level, {}).get('name', '无')

            self.log.emit("=" * 55)
            self.log.emit(f"CTA减影 v2.9.8 | 优化:{opt_name}")
            self.log.emit("=" * 55)

            # 1. 读取
            self.log.emit("\n[1/6] 读取...")
            reader = sitk.ImageSeriesReader()
            
            def get_files(series):
                s_dir = os.path.dirname(series.files[0][1])
                f_list = reader.GetGDCMSeriesFileNames(s_dir, series.series_uid)
                if f_list and len(f_list) == series.file_count:
                    return list(f_list)
                return [f[1] for f in series.files]

            pre_files = get_files(self.pre_series)
            post_files = get_files(self.post_series)
            
            if not pre_files or not post_files:
                self.finished_signal.emit(False, "读取失败：文件列表为空")
                return
            
            pre_img = sitk.ReadImage(pre_files, sitk.sitkInt16)
            post_img = sitk.ReadImage(post_files, sitk.sitkInt16)
            
            spacing = pre_img.GetSpacing()
            spacing_zyx = (spacing[2], spacing[1], spacing[0])
            self.progress.emit(10)

            # 2. 配准
            self.log.emit("\n[2/6] 3D配准...")
            aligned_img = execute_3d_registration(pre_img, post_img, self.log.emit, lambda v: self.progress.emit(10 + v))
            del post_img
            force_gc()
            if self.cancelled: return

            # 3. 优化配准 (32位系统防OOM极简内存模式)
            self.log.emit("\n[3/6] 配准优化 (32位低内存模式)...")
            
            w, h, depth = pre_img.GetWidth(), pre_img.GetHeight(), pre_img.GetDepth()
            
            pre_vol = np.empty((depth, h, w), dtype=np.float32)
            for z in range(depth):
                pre_vol[z] = sitk.GetArrayFromImage(pre_img[:, :, z]).astype(np.float32)
            del pre_img
            force_gc()

            if self.cancelled: return

            aligned_vol = np.empty((depth, h, w), dtype=np.float32)
            for z in range(depth):
                aligned_vol[z] = sitk.GetArrayFromImage(aligned_img[:, :, z])
            del aligned_img
            force_gc()

            if opt_level in ['standard', 'deep']:
                aligned_vol = safe_subpixel_refinement_v2(pre_vol, aligned_vol, self.log.emit, lambda v: self.progress.emit(25 + int(v/100*10)), 0)
                force_gc()

            self.progress.emit(35)
            
            frangi_vol = None
            if opt_level == 'deep':
                self.log.emit("  [深度] Frangi增强 (分层降显存版)...")
                frangi_vol = np.zeros_like(aligned_vol, dtype=np.float32)
                min_sp = min(spacing_zyx[1], spacing_zyx[2])
                sigmas = [max(0.5, min(2.0, s)) for s in [0.8 / min_sp, 1.4 / min_sp]]
                
                for z in range(depth):
                    if self.cancelled: return
                    slice_diff = np.clip(aligned_vol[z] - pre_vol[z], 0, None)
                    if slice_diff.max() > 15:
                        frangi_vol[z] = frangi_2d_fast(slice_diff, sigmas)
                    if z % 50 == 0:
                        self.progress.emit(35 + int((z / depth) * 10))
                
                vmax = frangi_vol.max()
                if vmax > 0: frangi_vol /= vmax
                self.log.emit("  [Frangi] 完成")
                force_gc()

            if self.cancelled: return

            # 4. 减影
            self.log.emit("\n[4/6] 减影...")
            total = pre_vol.shape[0]
            series_uid = generate_uid()
            spacing_xy = (spacing_zyx[1], spacing_zyx[2])
            
            for z in range(total):
                if self.cancelled: return
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
                        res = clean_bone_edges_cached(res, pre_s, cache, 'quality' if opt['quality_mode'] else 'fast')
                    if opt['smooth_sigma'] > 0:
                        res = edge_preserving_smooth(res, pre_s, opt['smooth_sigma'])
                        
                    aligned_vol[z] = res * opt['vessel_enhance']
                else:
                    aligned_vol[z] = 0
                    
                if z % 50 == 0:
                    self.progress.emit(45 + int((z / total) * 20))

            if frangi_vol is not None:
                del frangi_vol
                force_gc()

            result_vol = aligned_vol
            
            # 3D后处理 - 彻底修改为安全级连通判断
            min_area_2d = opt['min_vessel_size'] * 3
            result_vol = apply_3d_cleanup(result_vol, min_area=min_area_2d, log_cb=self.log.emit)
            
            if opt_level == 'deep':
                result_vol = shape_analysis_fast(result_vol, spacing_zyx, self.log.emit)
                
            if opt.get('vessel_scale_enable', False):
                result_vol = apply_vessel_scaling(result_vol, spacing_zyx, opt['vessel_split_mm'], opt['small_vessel_factor'], opt['large_vessel_factor'], self.log.emit)
                
            self.progress.emit(70)

            # 5. 写出
            self.log.emit("\n[5/6] 写入磁盘...")
            os.makedirs(self.output_dir, exist_ok=True)
            g_min, g_max = float(result_vol.min()), float(result_vol.max())
            g_slope = (g_max - g_min) / 4095.0 if g_max > g_min else 1.0

            for z, fp in enumerate(pre_files[:total]):
                if self.cancelled: return
                ds = pydicom.dcmread(fp)
                pix = result_vol[z]
                pix_int = ((pix - g_min) / (g_max - g_min) * 4095).astype(np.int16) if g_max > g_min else np.zeros_like(pix, dtype=np.int16)
                
                ds.PixelData = pix_int.tobytes()
                ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
                ds.PixelRepresentation, ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 1, 'MONOCHROME2'
                for tag in ['LossyImageCompression', 'LossyImageCompressionRatio', 'LossyImageCompressionMethod']:
                    if hasattr(ds, tag): delattr(ds, tag)
                
                ds.RescaleSlope, ds.RescaleIntercept = g_slope, g_min
                ds.WindowCenter, ds.WindowWidth = opt['wc'], opt['ww']
                ds.SeriesDescription = f"CTA Sub v2.9.8 [{opt_name}]"
                ds.SeriesInstanceUID = series_uid
                ds.SOPInstanceUID = generate_uid()
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                ds.save_as(os.path.join(self.output_dir, f"SUB_{ds.get('InstanceNumber', z + 1):04d}.dcm"), write_like_original=False)
                
                if z % 50 == 0:
                    self.progress.emit(70 + int((z / total) * 20))
                    
            self.progress.emit(90)

            # 6. 生成极速 3D 预览 (可选)
            if opt.get('generate_mip', False):
                self.log.emit("\n[6/6] 生成MIP 3D 预览动画...")
                try:
                    step = max(1, result_vol.shape[1] // 256)
                    vol_sm = result_vol[:, ::step, ::step]
                    
                    v_max_sm = max(1.0, get_safe_percentile_nonzero(vol_sm, 99.9, max_bin=4096))
                    
                    vol_sm = np.clip(vol_sm / v_max_sm * 255.0, 0, 255).astype(np.uint8)
                    
                    angles = range(0, 360, 15)
                    frames = []
                    for i, a in enumerate(angles):
                        if self.cancelled: break
                        rot = ndimage.rotate(vol_sm, a, axes=(1, 2), reshape=False, order=1)
                        mip = np.max(rot, axis=1) 
                        mip = np.flipud(mip)
                        frames.append(np.ascontiguousarray(mip))
                        self.progress.emit(90 + int((i / len(angles)) * 10))
                    
                    aspect_ratio = spacing_zyx[0] / (spacing_zyx[2] * step)
                    self.preview_data = (frames, aspect_ratio)
                    del vol_sm
                except Exception as e:
                    self.log.emit(f"  [预览生成跳过] {e}")
            else:
                self.log.emit("\n[6/6] 跳过MIP预览生成")

            del result_vol, pre_vol, aligned_vol
            force_gc()

            self.progress.emit(100)
            elapsed = time.time() - t0
            finish_msg = f"处理完成!\n耗时: {elapsed:.1f}秒"
            if opt.get('generate_mip', False):
                finish_msg += "\n您现在可以点击「MIP 预览」查看动态3D了"
            self.log.emit(f"\n{'=' * 55}\n✅ 完成! 耗时: {elapsed:.1f}s | 内存: {get_memory_mb():.0f} MB")
            self.finished_signal.emit(True, finish_msg)

        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))


# ════════════════════════════════════════════════════════════════════
# MIP 预览窗口 
# ════════════════════════════════════════════════════════════════════
class PreviewDialog(QDialog):
    def __init__(self, frames, aspect, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D MIP 预览 - 左右拖拽图像旋转")
        
        # --- 给预览窗口也加上图标 ---
        icon_path = resource_path("whw.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.setStyleSheet("background-color: #111; color: white;")
        self.setMinimumSize(1000, 1000)
        
        self.pixmaps = []
        for f in frames:
            h, w = f.shape
            qimg = QImage(f.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
            target_h = max(1, int(h * aspect))
            scale_factor = 600.0 / target_h if target_h > 0 else 1.0
            pix = pix.scaled(int(w * scale_factor), int(target_h * scale_factor), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.pixmaps.append(pix)
            
        self.current_idx = 0
        self.last_x = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setPixmap(self.pixmaps[self.current_idx])
        self.img_label.installEventFilter(self)
        layout.addWidget(self.img_label, 1)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(self.pixmaps) - 1)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #444; height: 8px; background: #333; border-radius: 4px; }
            QSlider::handle:horizontal { background: #1A6FBF; width: 18px; margin: -5px 0; border-radius: 9px; }
        """)
        self.slider.valueChanged.connect(self.on_slider)
        layout.addWidget(self.slider)

    def on_slider(self, val):
        self.current_idx = val
        self.img_label.setPixmap(self.pixmaps[val])

    def eventFilter(self, source, event):
        if source == self.img_label:
            if event.type() == event.MouseButtonPress:
                self.last_x = event.x()
                return True
            elif event.type() == event.MouseMove:
                dx = event.x() - self.last_x
                if abs(dx) > 8:  
                    steps = dx // 8
                    new_idx = (self.current_idx - steps) % len(self.pixmaps)
                    self.slider.setValue(new_idx)
                    self.last_x = event.x()
                return True
        return super().eventFilter(source, event)


# ════════════════════════════════════════════════════════════════════
# 主界面 UI
# ════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.all_series, self.selected_pre, self.selected_post, self.recommendations = {}, None, None, None
        self.proc_thread, self._all_presets = None, load_all_presets()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("头颅CTA减影-whw 版 v2.9.8")
        
        # --- 核心：挂载图标到主窗口的标题栏 ---
        icon_path = resource_path("whw.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.setMinimumSize(1000, 1000)
        self.resize(1100, 960)
        # 居中显示
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
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # 标题栏
        header = QWidget()
        header.setStyleSheet("background:#1A6FBF;border-radius:6px;")
        h_lay = QHBoxLayout(header)
        title = QLabel("头颅CTA减影-whw 版 V2.9.8")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setStyleSheet("color:white;")
        h_lay.addWidget(title)
        root.addWidget(header)
        # 1. 目录
        dir_grp = QGroupBox("1 数据目录")
        dir_lay = QHBoxLayout(dir_grp)
        self.data_dir_edit = QLineEdit()
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(lambda: self.data_dir_edit.setText(QFileDialog.getExistingDirectory(self) or self.data_dir_edit.text()))
        self.scan_btn = QPushButton("扫描")
        self.scan_btn.setStyleSheet("background:#1A6FBF;color:white;border-radius:4px;padding:6px 16px;")
        self.scan_btn.clicked.connect(self.scan_directory)
        dir_lay.addWidget(self.data_dir_edit)
        dir_lay.addWidget(browse_btn)
        dir_lay.addWidget(self.scan_btn)
        dir_grp.setFixedHeight(95)
        root.addWidget(dir_grp)

        # 2. 配对
        ser_grp = QGroupBox("2 序列配对")
        ser_lay = QVBoxLayout(ser_grp)
        r1 = QHBoxLayout()
        self.pre_combo, self.post_combo = QComboBox(), QComboBox()
        r1.addWidget(QLabel("平扫:")); r1.addWidget(self.pre_combo, 1)
        r1.addWidget(QLabel("增强:")); r1.addWidget(self.post_combo, 1)
        ser_lay.addLayout(r1)
        r2 = QHBoxLayout()
        self.analyze_btn = QPushButton("分析特征")
        self.analyze_btn.setStyleSheet("background:#1A6FBF;color:white;border-radius:4px;padding:6px 16px;")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_params)
        r2.addWidget(self.analyze_btn); r2.addStretch()
        ser_lay.addLayout(r2)
        ser_grp.setFixedHeight(135)
        root.addWidget(ser_grp)

        # 3. 输出
        out_grp = QGroupBox("3 输出目录")
        out_lay = QHBoxLayout(out_grp)
        self.out_edit = QLineEdit()
        out_btn = QPushButton("...")
        out_btn.clicked.connect(lambda: self.out_edit.setText(QFileDialog.getExistingDirectory(self) or self.out_edit.text()))
        out_lay.addWidget(self.out_edit); out_lay.addWidget(out_btn)
        out_grp.setFixedHeight(90)
        root.addWidget(out_grp)

        # 4. 模式
        mode_grp = QGroupBox("4 处理模式与优化级别")
        mode_lay = QVBoxLayout(mode_grp)
        mr = QHBoxLayout()
        self.fast_radio, self.quality_radio = QRadioButton("⚡ 快速（常规，适用扫描质量好）"), QRadioButton("⚡ ⚡  精细（适用扫描质量较差）")
        self.fast_radio.setChecked(True)
        mr.addWidget(self.fast_radio); mr.addWidget(self.quality_radio); mr.addStretch()
        mode_lay.addLayout(mr)
        self.mode_label = QLabel("")
        mode_lay.addWidget(self.mode_label)
        mode_lay.addWidget(QFrame(frameShape=QFrame.HLine))
        
        self.opt_group = QButtonGroup(self)
        self.opt_none = QRadioButton("无优化（原生模式）"); self.opt_none.setChecked(True)
        self.opt_light = QRadioButton("轻量级 - 自适应阈值+ICA保护（时间稍有增加）")
        self.opt_standard = QRadioButton("标准级 - +亚像素+静脉窦保护（时间有增加）")
        self.opt_deep = QRadioButton("深度级 - +Frangi血管增强 (低内存慎用，可能内存溢出)")
        
        self.opt_group.addButton(self.opt_none); self.opt_group.addButton(self.opt_light)
        self.opt_group.addButton(self.opt_standard); self.opt_group.addButton(self.opt_deep)
        
        mode_lay.addWidget(self.opt_none); mode_lay.addWidget(self.opt_light)
        mode_lay.addWidget(self.opt_standard); mode_lay.addWidget(self.opt_deep)
        mode_grp.setMinimumHeight(245)
        mode_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root.addWidget(mode_grp)

        # 5. 参数
        param_grp = QGroupBox("5 参数调节")
        param_lay = QVBoxLayout(param_grp)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        for name in self._all_presets: self.preset_combo.addItem(name, self._all_presets[name])
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self.apply_preset)
        self.preset_status = QLabel("")
        self.preset_status.setStyleSheet("color:#27AE60;")
        preset_row.addWidget(QLabel("预设:")); preset_row.addWidget(self.preset_combo); preset_row.addWidget(apply_btn); preset_row.addWidget(self.preset_status); preset_row.addStretch()
        param_lay.addLayout(preset_row)

        p_lay = QHBoxLayout()
        self.bone_slider = QSlider(Qt.Horizontal); self.bone_slider.setRange(5, 25); self.bone_slider.setValue(12)
        self.bone_label = QLabel("1.2"); self.bone_slider.valueChanged.connect(lambda v: self.bone_label.setText(f"{v/10:.1f}"))
        self.enhance, self.smooth = QDoubleSpinBox(), QDoubleSpinBox()
        self.enhance.setRange(0.5, 5.0); self.enhance.setValue(2.0)
        self.smooth.setRange(0, 1.5); self.smooth.setValue(0.7)
        self.clean_check = QCheckBox("去骨边"); self.clean_check.setChecked(True)
        self.wc, self.ww = QSpinBox(), QSpinBox()
        self.wc.setRange(-500, 1000); self.wc.setValue(200)
        self.ww.setRange(10, 2000); self.ww.setValue(400)
        p_lay.addWidget(QLabel("骨骼抑制:")); p_lay.addWidget(self.bone_slider); p_lay.addWidget(self.bone_label)
        p_lay.addWidget(QLabel("增强:")); p_lay.addWidget(self.enhance)
        p_lay.addWidget(self.clean_check); p_lay.addWidget(QLabel("降噪:")); p_lay.addWidget(self.smooth)
        p_lay.addWidget(QLabel("窗:")); p_lay.addWidget(self.wc); p_lay.addWidget(self.ww); p_lay.addStretch()
        param_lay.addLayout(p_lay)

        vs_lay = QHBoxLayout()
        self.scale_check = QCheckBox("血管缩放")
        self.split_mm, self.small_factor, self.large_factor = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        self.split_mm.setRange(0.5, 5.0); self.split_mm.setValue(1.5)
        self.small_factor.setRange(0.0, 5.0); self.small_factor.setValue(1.0)
        self.large_factor.setRange(0.0, 5.0); self.large_factor.setValue(1.0)
        self.mip_check = QCheckBox("生成MIP预览")
        vs_lay.addWidget(self.scale_check); vs_lay.addWidget(QLabel("界限(mm):")); vs_lay.addWidget(self.split_mm)
        vs_lay.addWidget(QLabel("小:")); vs_lay.addWidget(self.small_factor); vs_lay.addWidget(QLabel("大:")); vs_lay.addWidget(self.large_factor)
        vs_lay.addWidget(self.mip_check); vs_lay.addStretch()
        param_lay.addLayout(vs_lay)
        param_grp.setMinimumHeight(175)
        param_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root.addWidget(param_grp)

        # 操作栏
        act_lay = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始")
        self.start_btn.setStyleSheet("background:#1A6FBF;color:white;border-radius:5px;font-weight:bold;padding:0 24px;height:34px;")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        
        self.preview_btn = QPushButton("MIP 预览")
        self.preview_btn.setStyleSheet("background:#27AE60;color:white;border-radius:5px;font-weight:bold;padding:0 20px;height:34px;")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self.show_preview)
        
        self.progress = QProgressBar()
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        act_lay.addWidget(self.start_btn); act_lay.addWidget(self.preview_btn); act_lay.addWidget(self.progress, 1); act_lay.addWidget(self.cancel_btn)
        root.addLayout(act_lay)

        # 日志：stretch=1 让其随窗口垂直扩张
        log_grp = QGroupBox("日志")
        log_lay = QVBoxLayout(log_grp)
        log_lay.setContentsMargins(6, 6, 6, 6)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)
        log_lay.addWidget(self.log_text)
        root.addWidget(log_grp, 1)


        self.log("头颅CTA减影-whw 版 v2.9.8")
        self.log("=" * 50)
        self.log("⚡ 快速模式: 扫描质量好时使用，速度快")
        self.log("⚡⚡ 精细模式: 复杂情况使用，减少残留和噪点")
        self.log("=" * 50)
        self.log("")
        self.log("使用步骤:")
        self.log("  1. 选择DICOM目录，点击「扫描」")
        self.log("  2. 确认序列配对")
        self.log("  3. 点击「分析特征」（自动推荐模式和参数）")
        self.log("  4. 选择处理模式和优化级别，点击「开始」")
        self.log("  5. 血管缩放：界限以上、以下的血管缩放倍数，取0时血管彻底消失")
        self.log("  6. 勾选「生成MIP预览」后运算结束可点击「MIP预览」，初步观察大血管有无混乱、断裂")

    def log(self, msg):
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        QApplication.processEvents()

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
            self.wc.setValue(cfg["wc"]); self.ww.setValue(cfg["ww"])
            if self.recommendations is None: self.recommendations = {}
            self.recommendations['vessel_sensitivity'], self.recommendations['min_vessel_size'] = cfg.get('vessel_sensitivity', 1.0), cfg.get('min_vessel_size', 5)
            self.preset_status.setText(f"✓ 已应用: {self.preset_combo.currentText()}")
            self.log(f"✅ 应用预设: {self.preset_combo.currentText()}")

    def scan_directory(self):
        d = self.data_dir_edit.text()
        if not d: return
        self.preview_btn.setEnabled(False)
        self.pre_combo.clear(); self.post_combo.clear()
        self.scan_btn.setEnabled(False); self.analyze_btn.setEnabled(False)
        self.scan_thread = SeriesScanThread(d)
        self.scan_thread.progress.connect(self.progress.setValue)
        self.scan_thread.log.connect(self.log)
        self.scan_thread.finished_signal.connect(self.on_scan_finished)
        self.scan_thread.start()

    def on_scan_finished(self, all_series, pairs, cta_series):
        self.scan_btn.setEnabled(True)
        for s in sorted(all_series.values(), key=lambda x: x.series_number):
            txt = f"#{s.series_number:03d} | {s.file_count}张 | {'[C+]' if s.contrast_status is True else '[C-]' if s.contrast_status is False else ''} {s.series_description[:40]}"
            self.pre_combo.addItem(txt, s); self.post_combo.addItem(txt, s)
        if pairs:
            self.pre_combo.setCurrentIndex(self.pre_combo.findData(pairs[0][0])); self.post_combo.setCurrentIndex(self.post_combo.findData(pairs[0][1]))
        self.analyze_btn.setEnabled(True)
        if not self.out_edit.text(): self.out_edit.setText(os.path.join(self.data_dir_edit.text(), "CTA_SUB_v298"))
        if pairs or cta_series:
            preset = detect_manufacturer_preset((pairs[0][0] if pairs else cta_series[0]).files)
            if preset and self.preset_combo.findText(preset) >= 0:
                self.preset_combo.setCurrentIndex(self.preset_combo.findText(preset)); self.apply_preset()
                self.log(f"🔍 检测设备: {preset}")

    def analyze_params(self):
        if not self.pre_combo.currentData() or not self.post_combo.currentData() or self.pre_combo.currentData() == self.post_combo.currentData(): return QMessageBox.warning(self, "提示", "请选择不同序列")
        self.selected_pre, self.selected_post = self.pre_combo.currentData(), self.post_combo.currentData()
        self.analyze_btn.setEnabled(False)
        self.param_thread = ParamAnalyzeThread(self.selected_pre.files, self.selected_post.files)
        self.param_thread.progress.connect(self.progress.setValue); self.param_thread.log.connect(self.log); self.param_thread.finished_signal.connect(self.on_analyze_finished)
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
        self.wc.setValue(res.get('wc', 200)); self.ww.setValue(res.get('ww', 400))
        (self.fast_radio if res.get('recommended_mode', 'fast') == 'fast' else self.quality_radio).setChecked(True)
        self.mode_label.setText(f"质量: {res.get('quality_score', 0) * 100:.0f}%"); self.start_btn.setEnabled(True)

    def start_processing(self):
        if not self.out_edit.text(): return QMessageBox.warning(self, "提示", "输出目录为空")
        self.preview_btn.setEnabled(False)
        opt = {
            'quality_mode': self.quality_radio.isChecked(), 'bone_strength': self.bone_slider.value() / 10.0,
            'vessel_sensitivity': (self.recommendations or {}).get('vessel_sensitivity', 1.0),
            'vessel_enhance': self.enhance.value(), 'clean_bone_edges': self.clean_check.isChecked(),
            'min_vessel_size': (self.recommendations or {}).get('min_vessel_size', 5), 'smooth_sigma': self.smooth.value(),
            'wc': self.wc.value(), 'ww': self.ww.value(),
            'vessel_scale_enable': self.scale_check.isChecked(), 'vessel_split_mm': self.split_mm.value(),
            'small_vessel_factor': self.small_factor.value(), 'large_vessel_factor': self.large_factor.value(),
            'optimization_level': self._get_opt_level(),
            'generate_mip': self.mip_check.isChecked(),
        }
        self.start_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.proc_thread = ProcessThread(self.selected_pre, self.selected_post, self.out_edit.text(), opt)
        self.proc_thread.progress.connect(self.progress.setValue); self.proc_thread.log.connect(self.log); self.proc_thread.finished_signal.connect(self.on_finished)
        self.proc_thread.start()

    def cancel(self):
        if self.proc_thread: self.proc_thread.cancel()

    def on_finished(self, success, msg):
        self.start_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        if success:
            if getattr(self.proc_thread, 'preview_data', None):
                self.preview_btn.setEnabled(True)
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "错误", msg)

    def show_preview(self):
        if hasattr(self.proc_thread, 'preview_data') and self.proc_thread.preview_data:
            frames, aspect = self.proc_thread.preview_data
            dlg = PreviewDialog(frames, aspect, self)
            dlg.exec_()
        else:
            QMessageBox.warning(self, "提示", "未找到预览数据")


# ════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    
    # --- 核心修复：欺骗 Windows 任务栏，声明这是一个独立应用，不再使用 Python 默认图标 ---
    if os.name == 'nt':
        try:
            # 设定唯一的应用ID (AppUserModelID)
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("whw.cta.subtraction.298")
        except Exception:
            pass
            
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())