"""
v2.8.6 极限内存优化版 - 极简自动化版
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[优化] 3D配准阶段引入多分辨率降采样策略，解决3G内存狂飙问题
[优化] 彻底重写"血管自适应调节"算法，采用Z轴分块和布尔腐蚀，消灭3GB瞬时尖峰
[交互] 移除手动释放按钮，引入"智能隐式垃圾回收"，下一次任务触发时自动释放内存，界面更纯粹
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
    QSpinBox, QDoubleSpinBox, QSlider, QRadioButton, QFrame, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import pydicom
from pydicom.uid import generate_uid

print("头颅CTA减影 v2.8.6 - WHW 版 (自动化内存管理)")


# ════════════════════════════════════════════════════════════════════
# 内存监控工具
# ════════════════════════════════════════════════════════════════════
def get_memory_mb() -> float:
    """获取当前进程内存占用(MB)"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def force_gc():
    """强制垃圾回收"""
    gc.collect()
    gc.collect()


# ════════════════════════════════════════════════════════════════════
# 设备预设
# ════════════════════════════════════════════════════════════════════
PRESET_FILE = Path(__file__).parent / "device_presets.json"

_MANUFACTURER_MAP = {
    ("siemens",):                        "Siemens SOMATOM",
    ("ge ", "ge", "general electric"):   "GE Revolution/Discovery",
    ("philips",):                        "Philips Brilliance/IQon",
    ("canon", "toshiba"):                "Canon Aquilion",
    ("united imaging", "联影", "uih"):    "联影 uCT",
}

BUILTIN_PRESETS = {
    "通用默认": {
        "bone_strength": 1.2, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.7, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400,
        "description": "适合未知设备或多厂商混合环境",
    },
    "Siemens SOMATOM": {
        "bone_strength": 1.2, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400,
        "description": "SOMATOM Definition/Force/go",
    },
    "GE Revolution/Discovery": {
        "bone_strength": 1.4, "vessel_sensitivity": 0.9, "vessel_enhance": 1.8,
        "smooth_sigma": 0.9, "clean_bone_edges": True, "min_vessel_size": 6,
        "wc": 220, "ww": 420,
        "description": "Revolution CT/Discovery CT750",
    },
    "Philips Brilliance/IQon": {
        "bone_strength": 1.0, "vessel_sensitivity": 1.1, "vessel_enhance": 2.2,
        "smooth_sigma": 0.5, "clean_bone_edges": True, "min_vessel_size": 4,
        "wc": 180, "ww": 380,
        "description": "Brilliance iCT/IQon Spectral",
    },
    "Canon Aquilion": {
        "bone_strength": 1.1, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": False, "min_vessel_size": 5,
        "wc": 200, "ww": 400,
        "description": "Aquilion ONE/GENESIS",
    },
    "联影 uCT": {
        "bone_strength": 1.3, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.8, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400,
        "description": "uCT 960/T+",
    },
}


def load_all_presets() -> dict:
    merged = dict(BUILTIN_PRESETS)
    if PRESET_FILE.exists():
        try:
            with open(PRESET_FILE, "r", encoding="utf-8") as f:
                user = json.load(f)
            merged.update(user)
        except Exception as e:
            print(f"[警告] 读取本地预设失败: {e}")
    return merged


def save_custom_preset(name: str, params: dict):
    user = {}
    if PRESET_FILE.exists():
        try:
            with open(PRESET_FILE, "r", encoding="utf-8") as f:
                user = json.load(f)
        except Exception:
            pass
    user[name] = params
    with open(PRESET_FILE, "w", encoding="utf-8") as f:
        json.dump(user, f, ensure_ascii=False, indent=2)


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
    except Exception as e:
        print(f"[警告] 读取厂商信息失败: {e}")
    return None


# ════════════════════════════════════════════════════════════════════
# 序列扫描与分析
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
        self._contrast_status: Optional[bool] = None
        self._contrast_cached: bool = False

    @property
    def contrast_status(self) -> Optional[bool]:
        if not self._contrast_cached:
            self._contrast_status = _calc_contrast_status(self)
            self._contrast_cached = True
        return self._contrast_status


def _calc_contrast_status(series: "SeriesInfo") -> Optional[bool]:
    desc = series.series_description.upper()
    pos_patterns = [r'\bC\+', r'\bC\s*\+', r'CE\b', r'CONTRAST', r'ENHANCED',
                    r'POST', r'ARTERIAL', r'增强', r'动脉期', r'A期']
    neg_patterns = [r'\bC-', r'\bC\s*-', r'\bNC\b', r'NON-CONTRAST',
                    r'PLAIN', r'PRE\b', r'WITHOUT', r'平扫', r'非增强']
    if any(re.search(p, desc) for p in pos_patterns):
        return True
    if any(re.search(p, desc) for p in neg_patterns):
        return False
    return None


def parse_time(time_str):
    if not time_str:
        return None
    try:
        time_str = str(time_str).split('.')[0]
        if len(time_str) >= 6:
            return datetime.strptime(time_str[:6], "%H%M%S")
    except ValueError:
        pass
    return None


def analyze_dicom_file(filepath: str) -> Optional[dict]:
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
    except Exception as e:
        print(f"  [警告] 跳过文件 {filepath}: {e}")
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
        except OSError as e:
            if log_callback:
                log_callback(f"  [警告] 无法读取 {f.name}: {e}")
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
                if (series.acquisition_time is None or
                        info['acquisition_time'] < series.acquisition_time):
                    series.acquisition_time = info['acquisition_time']
        if progress_callback:
            progress_callback(int((i + 1) / len(dicom_files) * 100))
    for series in series_dict.values():
        series.files.sort(key=lambda x: x[0])
    return dict(series_dict)


def is_head_cta_series(series: SeriesInfo) -> bool:
    desc = (f"{series.series_description} {series.study_description} "
            f"{series.protocol_name} {series.body_part}").upper()
    exclude = ['SCOUT', 'LOCALIZER', 'TOPOGRAM', '定位',
               'LUNG', 'CHEST', '肺', '胸', 'CARDIAC', 'HEART', '心',
               'ABDOMEN', 'LIVER', '腹']
    if any(kw in desc for kw in exclude):
        return False
    if series.modality != 'CT' or series.file_count < 50:
        return False
    has_head = any(kw in desc for kw in
                   ['HEAD', 'BRAIN', 'CEREBR', 'CRANIAL', '头', '颅', '脑', 'CAROTID'])
    has_cta = any(kw in desc for kw in
                  ['CTA', 'ANGIO', 'C+', 'C-', '血管', '动脉'])
    return (has_head and has_cta) or (has_head and series.file_count >= 100)


def find_cta_pairs(series_dict: dict):
    cta_series = [s for s in series_dict.values() if is_head_cta_series(s)]
    if len(cta_series) < 2:
        return [], cta_series
    groups = defaultdict(list)
    for s in cta_series:
        groups[(s.file_count, round(s.slice_thickness, 1))].append(s)
    pairs, used = [], set()
    for _key, group in groups.items():
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
            unknown.sort(key=lambda s: (s.acquisition_time or datetime.max,
                                        s.series_number))
            pairs.append((unknown[0], unknown[1]))
    return pairs, cta_series


# ════════════════════════════════════════════════════════════════════
# 扫描线程
# ════════════════════════════════════════════════════════════════════
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
            series_dict = scan_directory_for_series(
                self.directory, self.progress.emit, self.log.emit)
            self.log.emit(f"\n找到 {len(series_dict)} 个序列\n")
            for s in sorted(series_dict.values(), key=lambda s: s.series_number):
                time_str = (s.acquisition_time.strftime("%H:%M:%S")
                            if s.acquisition_time else "--:--:--")
                st = s.contrast_status
                marker = ("C+" if st is True else
                          "C-" if st is False else
                          "★ " if is_head_cta_series(s) else "  ")
                desc = s.series_description[:35] or "(无描述)"
                self.log.emit(
                    f"{marker} #{s.series_number:3d} | {s.file_count:4d}张 "
                    f"| {time_str} | {desc}")
            pairs, cta_series = find_cta_pairs(series_dict)
            if pairs:
                pre, post = pairs[0]
                self.log.emit(
                    f"\n★ 自动配对:\n"
                    f"  平扫: #{pre.series_number} {pre.series_description[:30]}\n"
                    f"  增强: #{post.series_number} {post.series_description[:30]}")
            self.progress.emit(100)
            self.finished_signal.emit(series_dict, pairs, cta_series)
        except Exception as e:
            self.log.emit(f"[错误] 扫描失败: {e}")
            self.finished_signal.emit({}, [], [])


# ════════════════════════════════════════════════════════════════════
# FFT 相位相关配准
# ════════════════════════════════════════════════════════════════════
def fft_phase_correlation_2d(fixed: np.ndarray, moving: np.ndarray,
                             max_shift: int = 15) -> Tuple[float, float]:
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
    if 1 <= py < correlation.shape[0] - 1:
        y_vals = correlation[py - 1:py + 2, px]
        if y_vals[1] > y_vals[0] and y_vals[1] > y_vals[2]:
            denom = 2.0 * (y_vals[0] + y_vals[2] - 2.0 * y_vals[1])
            if abs(denom) > 1e-6:
                dy += (y_vals[0] - y_vals[2]) / denom
    if 1 <= px < correlation.shape[1] - 1:
        x_vals = correlation[py, px - 1:px + 2]
        if x_vals[1] > x_vals[0] and x_vals[1] > x_vals[2]:
            denom = 2.0 * (x_vals[0] + x_vals[2] - 2.0 * x_vals[1])
            if abs(denom) > 1e-6:
                dx += (x_vals[0] - x_vals[2]) / denom
    return (float(np.clip(dy, -max_shift, max_shift)),
            float(np.clip(dx, -max_shift, max_shift)))


def shift_image_2d(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    return ndimage.shift(image.astype(np.float32), [dy, dx],
                         order=1, mode='constant', cval=0)


# ════════════════════════════════════════════════════════════════════
# 智能参数分析线程
# ════════════════════════════════════════════════════════════════════
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
            self.log.emit("\n" + "=" * 55 +
                          "\n智能参数分析 (基于图像特征)\n" + "=" * 55)
            pre_dict = {inst: path for inst, path in self.pre_files}
            post_dict = {inst: path for inst, path in self.post_files}
            common = sorted(set(pre_dict.keys()) & set(post_dict.keys()))
            if not common:
                self.finished_signal.emit({'error': '序列不匹配，无共同层号'})
                return
            total = len(common)
            if total > 10:
                sample_indices = [common[total // 2 - 5],
                                  common[total // 2],
                                  common[total // 2 + 5]]
            else:
                sample_indices = [common[0]]
            all_chars = []
            for i, inst in enumerate(sample_indices):
                pre_d = pydicom.dcmread(pre_dict[inst]).pixel_array.astype(np.float32)
                post_d = pydicom.dcmread(post_dict[inst]).pixel_array.astype(np.float32)
                dy, dx = fft_phase_correlation_2d(pre_d, post_d)
                post_aligned = shift_image_2d(post_d, dy, dx)
                all_chars.append(self._analyze_image(pre_d, post_aligned))
                self.progress.emit(int((i + 1) / len(sample_indices) * 100))
            avg = {k: float(np.mean([c[k] for c in all_chars]))
                   for k in all_chars[0]}
            rec = self._compute_recommendations(avg)
            quality_score = self._assess_quality(avg)
            rec['quality_score'] = quality_score
            rec['recommended_mode'] = 'fast' if quality_score >= 0.7 else 'quality'
            self.log.emit(
                f"图像质量: {'优良 (建议快速模式)' if quality_score >= 0.7 else '一般 (建议精细模式)'}")
            self.finished_signal.emit(rec)
        except Exception as e:
            self.finished_signal.emit({'error': str(e)})

    def _analyze_image(self, pre, post):
        diff = np.clip(post - pre, 0, None)
        bone = pre > 150
        thin = detect_thin_bone(pre)
        air = pre < -800
        noise = float(pre[air].std()) if air.sum() > 100 else 15.0
        strong = diff > 50
        vessel = float(diff[strong].mean()) if strong.sum() > 0 else 0.0
        air_dil = ndimage.binary_dilation(pre < -200, iterations=3)
        bone_dil = ndimage.binary_dilation(pre > 100, iterations=3)
        air_bone = air_dil & bone_dil
        return {
            'bone': float(bone.sum() / bone.size),
            'thin_bone': float(thin.sum() / thin.size),
            'noise': noise,
            'vessel_signal': vessel,
            'air_bone_interface': float(air_bone.sum() / air_bone.size),
        }

    def _assess_quality(self, chars) -> float:
        score = 1.0
        if chars['noise'] > 20:
            score -= 0.2
        elif chars['noise'] > 15:
            score -= 0.1
        if chars['vessel_signal'] < 40:
            score -= 0.2
        elif chars['vessel_signal'] < 60:
            score -= 0.1
        if chars.get('air_bone_interface', 0) > 0.015:
            score -= 0.15
        elif chars.get('air_bone_interface', 0) > 0.01:
            score -= 0.08
        if chars['thin_bone'] > 0.008:
            score -= 0.1
        return float(np.clip(score, 0, 1))

    def _compute_recommendations(self, chars) -> dict:
        rec = {}
        base = 1.0
        if chars['thin_bone'] > 0.005:
            base += 0.3
        if chars['bone'] > 0.03:
            base += 0.2
        if chars.get('air_bone_interface', 0) > 0.01:
            base += 0.2
        rec['bone_strength'] = round(min(base, 2.0), 1)
        rec['vessel_sensitivity'] = (0.8 if chars['vessel_signal'] > 80 else
                                     1.0 if chars['vessel_signal'] > 40 else 1.2)
        rec['vessel_enhance'] = (1.5 if chars['vessel_signal'] > 80 else
                                 2.0 if chars['vessel_signal'] > 50 else 2.5)
        rec['clean_bone_edges'] = (chars['thin_bone'] > 0.003 or
                                   chars.get('air_bone_interface', 0) > 0.008)
        rec['min_vessel_size'] = (8 if chars['noise'] > 20 else
                                  5 if chars['noise'] > 12 else 3)
        rec['smooth_sigma'] = 0.9 if chars['noise'] > 15 else 0.6
        rec['wc'], rec['ww'] = 200, 400
        return rec


# ════════════════════════════════════════════════════════════════════
# SimpleITK 3D 掩膜配准引擎 (内存极速优化版)
# ════════════════════════════════════════════════════════════════════
def execute_3d_registration(fixed_img, moving_img, log_cb=None, progress_cb=None):
    fixed_size = fixed_img.GetSize()
    shrink_factors = [1, 1, 1]
    if fixed_size[0] >= 400: 
        shrink_factors[0] = 2
        shrink_factors[1] = 2
    if fixed_size[2] >= 100: 
        shrink_factors[2] = 2

    if log_cb:
        log_cb(f"  [内存优化] 启动低分辨率配准, 缩放因子: {shrink_factors}")

    fixed_reg = sitk.Shrink(fixed_img, shrink_factors)
    moving_reg = sitk.Shrink(moving_img, shrink_factors)

    fixed_reg = sitk.Cast(fixed_reg, sitk.sitkFloat32)
    moving_reg = sitk.Cast(moving_reg, sitk.sitkFloat32)

    if log_cb:
        log_cb("  正在生成 3D 骨骼掩膜 (HU > 200)...")
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
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=0.001, numberOfIterations=60,
        estimateLearningRate=R.Never)
    R.SetInterpolator(sitk.sitkLinear)

    def _on_iteration(method):
        it = method.GetOptimizerIteration()
        if progress_cb:
            progress_cb(int((it / 60.0) * 30))
        if log_cb and it % 15 == 0:
            log_cb(f"  迭代 {it:02d} | MSE: {method.GetMetricValue():.2f}")

    R.AddCommand(sitk.sitkIterationEvent, lambda: _on_iteration(R))
    
    if log_cb:
        log_cb("  开始 3D 空间微调优化...")
        
    final_transform = R.Execute(fixed_reg, moving_reg)
    
    if log_cb:
        log_cb(f"  3D配准完成. 最终MSE: {R.GetMetricValue():.2f}")
        params = final_transform.GetParameters()
        if len(params) == 6:
            angles = np.degrees(params[:3])
            trans = params[3:]
            log_cb(f"  [监测] 旋转偏角: X={angles[0]:.2f}° Y={angles[1]:.2f}° Z={angles[2]:.2f}°")
            log_cb(f"  [监测] 平移距离: X={trans[0]:.2f} Y={trans[1]:.2f} Z={trans[2]:.2f} mm")
            
    del fixed_reg
    del moving_reg
    del bone_mask
    del R
    gc.collect()

    if log_cb:
        log_cb("  正在将增强序列重采样至平扫的绝对物理网格...")
        
    aligned_img = sitk.Resample(
        moving_img, fixed_img, final_transform,
        sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
        
    return aligned_img


# ════════════════════════════════════════════════════════════════════
# 基础形态学工具
# ════════════════════════════════════════════════════════════════════
def create_body_mask(image: np.ndarray) -> np.ndarray:
    body = image > -400
    body = ndimage.binary_closing(body, iterations=3)
    body = ndimage.binary_fill_holes(body)
    labeled, num = ndimage.label(body)
    if num > 0:
        sizes = ndimage.sum(body, labeled, range(1, num + 1))
        body = labeled == (np.argmax(sizes) + 1)
    return body


def detect_thin_bone(pre_image: np.ndarray) -> np.ndarray:
    bone = pre_image > 150
    thin_bone = bone & ~ndimage.binary_erosion(bone, iterations=2)
    edges = np.abs(ndimage.sobel(pre_image.astype(np.float32)))
    ref_edges = edges[bone] if bone.sum() > 0 else edges
    high_edge = edges > np.percentile(ref_edges, 70)
    return thin_bone | (bone & high_edge)


def detect_equipment(pre: np.ndarray, post: np.ndarray,
                     body: Optional[np.ndarray] = None) -> np.ndarray:
    if body is None:
        body = create_body_mask(pre)
    high_both = (pre > 150) & (post > 150)
    stable = np.abs(post - pre) < 30
    body_dilated = ndimage.binary_dilation(body, iterations=8)
    return ndimage.binary_dilation(
        high_both & stable & ~body_dilated, iterations=5)


# ════════════════════════════════════════════════════════════════════
# 预计算缓存架构
# ════════════════════════════════════════════════════════════════════
class SliceCache:
    def __init__(self):
        self.body: Optional[np.ndarray] = None
        self.scalp: Optional[np.ndarray] = None
        self.air_bone: Optional[np.ndarray] = None
        self.petrous: Optional[np.ndarray] = None
        self.thin_bone: Optional[np.ndarray] = None
        self.venous: Optional[np.ndarray] = None
        self.petrous_edge: Optional[np.ndarray] = None
        self.bone_edge: Optional[np.ndarray] = None
        self.ab_edge: Optional[np.ndarray] = None
        self.ica_protect: Optional[np.ndarray] = None


def precompute_slice_cache(pre_s: np.ndarray,
                           diff_pos: np.ndarray,
                           quality_mode: bool) -> SliceCache:
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
    cache.scalp = ndimage.binary_dilation(
        scalp_zone & soft_tissue, iterations=2)

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
# 减影算法
# ════════════════════════════════════════════════════════════════════
def fast_subtraction(pre: np.ndarray, post_aligned: np.ndarray,
                     bone_strength: float = 1.0, vessel_sensitivity: float = 1.0,
                     body: Optional[np.ndarray] = None) -> np.ndarray:
    if body is None:
        body = create_body_mask(pre)
    diff_pos = np.clip(post_aligned - pre, 0, None)
    gain = np.ones_like(diff_pos, dtype=np.float32)
    gain[pre < -500] = 0
    gain[~body] = 0
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


def quality_subtraction_cached(pre: np.ndarray, post_aligned: np.ndarray,
                               bone_strength: float, vessel_sensitivity: float,
                               cache: SliceCache) -> np.ndarray:
    diff_pos = np.clip(post_aligned - pre, 0, None)
    gain = np.ones_like(diff_pos, dtype=np.float32)
    gain[~cache.body | (pre < -500) | (pre > 500)] = 0

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
    gain[med_bone & (diff_pos >= 60 * bone_strength) &
         (diff_pos < 120 * bone_strength)] = 0.15 / bone_strength
    gain[med_bone & (diff_pos >= 120 * bone_strength)] = 0.3 / bone_strength

    thin_only = cache.thin_bone & ~exclude
    th_thin = 80 * bone_strength
    gain[thin_only & (diff_pos < th_thin)] = 0
    gain[thin_only & (diff_pos >= th_thin)] = 0.1 / bone_strength

    scalp_mask = cache.scalp if cache.scalp is not None else np.zeros_like(pre, dtype=bool)
    ab_mask = cache.air_bone if cache.air_bone is not None else np.zeros_like(pre, dtype=bool)
    vessel_mask = ((pre > -100) & (pre < 60) &
                   (diff_pos > 50 * vessel_sensitivity) &
                   ~scalp_mask & ~ab_mask)
    gain[vessel_mask] = 1.0

    return diff_pos * np.clip(gain, 0, 1.5)


def clean_bone_edges_cached(image: np.ndarray, pre_image: np.ndarray,
                            cache: SliceCache, mode: str = 'fast') -> np.ndarray:
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


def edge_preserving_smooth(image: np.ndarray, pre_image: np.ndarray,
                           sigma: float = 0.7) -> np.ndarray:
    edges = np.abs(ndimage.sobel(pre_image.astype(np.float32)))
    edge_norm = edges / (edges.max() + 1e-6)
    smooth_h = ndimage.gaussian_filter(image, sigma * 1.5)
    smooth_l = ndimage.gaussian_filter(image, sigma * 0.3)
    return smooth_h * (1 - edge_norm) + smooth_l * edge_norm


# ════════════════════════════════════════════════════════════════════
# 3D 拓扑清理 - 简化版（减少内存）
# ════════════════════════════════════════════════════════════════════
def apply_3d_morphological_cleanup_simple(volume: np.ndarray,
                                          min_volume: int = 100,
                                          log_cb=None) -> np.ndarray:
    pos_vox = volume[volume > 0]
    if pos_vox.size == 0:
        return volume
    
    fg_threshold = max(10.0, float(np.percentile(pos_vox, 10)))
    del pos_vox
    force_gc()
    
    mask = volume > fg_threshold
    if log_cb:
        log_cb(f"  [内存] 清理前: {get_memory_mb():.0f} MB")
    
    labeled_vol, num_features = ndimage.label(mask)
    labeled_vol = labeled_vol.astype(np.int32)
    
    del mask
    force_gc()
    
    if num_features == 0:
        del labeled_vol
        return volume
    
    if log_cb:
        log_cb(f"  连通域数量: {num_features}")
    
    sizes = np.bincount(labeled_vol.ravel(), minlength=num_features + 1)
    
    keep_labels = np.zeros(num_features + 1, dtype=bool)
    keep_labels[0] = False
    keep_labels[1:] = sizes[1:] >= min_volume
    
    del sizes
    force_gc()
    
    chunk_size = max(1, volume.shape[0] // 4)
    for z_start in range(0, volume.shape[0], chunk_size):
        z_end = min(z_start + chunk_size, volume.shape[0])
        chunk_labels = labeled_vol[z_start:z_end]
        chunk_vol = volume[z_start:z_end]
        
        noise_mask = ~keep_labels[chunk_labels] & (chunk_labels > 0)
        chunk_vol[noise_mask & (chunk_vol < 60)] = 0
        chunk_vol[noise_mask & (chunk_vol >= 60)] *= 0.3
    
    del labeled_vol, keep_labels
    force_gc()
    
    if log_cb:
        log_cb(f"  [内存] 清理后: {get_memory_mb():.0f} MB")
    
    return volume


# ════════════════════════════════════════════════════════════════════
# 血管自适应膨胀缩放 - ★ V2.8.5 解决3G内存尖峰核心 ★
# ════════════════════════════════════════════════════════════════════
def apply_vessel_scaling_simple(volume: np.ndarray, spacing: tuple,
                                split_mm: float, small_factor: float, large_factor: float,
                                log_cb=None) -> np.ndarray:
    """血管自适应缩放 - 彻底消灭3GB内存尖峰的最终版"""
    if small_factor == 1.0 and large_factor == 1.0:
        return volume
    
    if log_cb:
        log_cb(f"  [内存] 缩放前: {get_memory_mb():.0f} MB")
    
    mask = volume > 20
    if not np.any(mask):
        return volume
    
    r_thresh = split_mm / 2.0
    min_sp = min(spacing) if min(spacing) > 0 else 1.0
    erode_iters = int(np.ceil(r_thresh / min_sp))
    
    large_core = mask.copy()
    struct = ndimage.generate_binary_structure(3, 1)
    
    for _ in range(erode_iters):
        large_core = ndimage.binary_erosion(large_core, structure=struct)
    
    force_gc()
    if log_cb:
        log_cb(f"  [内存] 布尔腐蚀定位后: {get_memory_mb():.0f} MB")
    
    # 形态学重建大血管边界
    large_mask = large_core.copy()
    del large_core
    
    max_iters = erode_iters + 3
    for _ in range(max_iters):
        next_mask = ndimage.binary_dilation(large_mask, structure=struct) & mask
        if np.array_equal(next_mask, large_mask):
            break
        large_mask = next_mask
    
    small_mask = mask & ~large_mask
    del mask
    force_gc()
    
    def safe_grow(vol, base_mask, new_region):
        chunk_size = 32
        for z in range(0, vol.shape[0], chunk_size):
            z1 = max(0, z - 1)
            z2 = min(vol.shape[0], z + chunk_size + 1)
            
            sub_vol = vol[z1:z2] * base_mask[z1:z2]
            max_filtered = ndimage.maximum_filter(sub_vol, size=3)
            
            valid_z1 = 1 if z > 0 else 0
            z_len = min(chunk_size, vol.shape[0] - z)
            
            target_new = new_region[z:z+z_len]
            chunk_max = max_filtered[valid_z1:valid_z1+z_len]
            
            vol_chunk = vol[z:z+z_len]
            vol_chunk[target_new] = chunk_max[target_new]

    # 处理小血管
    if small_factor != 1.0 and np.any(small_mask):
        iters = max(1, int(round(abs(small_factor - 1.0) * 3)))
        if small_factor > 1.0:
            grow_mask = small_mask.copy()
            for _ in range(iters):
                grow_mask = ndimage.binary_dilation(grow_mask, structure=struct)
            new_region = grow_mask & ~small_mask
            if np.any(new_region):
                safe_grow(volume, small_mask, new_region)
            del grow_mask
        else:
            eroded = small_mask.copy()
            for _ in range(iters):
                eroded = ndimage.binary_erosion(eroded, structure=struct)
            removed = small_mask & ~eroded
            volume[removed] = 0
            del eroded
    
    del small_mask
    force_gc()
    
    # 处理大血管
    if large_factor != 1.0 and np.any(large_mask):
        iters = max(1, int(round(abs(large_factor - 1.0) * 3)))
        if large_factor > 1.0:
            grow_mask = large_mask.copy()
            for _ in range(iters):
                grow_mask = ndimage.binary_dilation(grow_mask, structure=struct)
            new_region = grow_mask & ~large_mask
            if np.any(new_region):
                safe_grow(volume, large_mask, new_region)
            del grow_mask
        else:
            eroded = large_mask.copy()
            for _ in range(iters):
                eroded = ndimage.binary_erosion(eroded, structure=struct)
            removed = large_mask & ~eroded
            volume[removed] = 0
            del eroded
    
    del large_mask
    force_gc()
    
    if log_cb:
        log_cb(f"  [内存] 分块滤波缩放后: {get_memory_mb():.0f} MB")
    
    return volume


# ════════════════════════════════════════════════════════════════════
# 3D 容积处理主线程
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
        self.result_vol: Optional[np.ndarray] = None
        self.pre_vol: Optional[np.ndarray] = None
        self._pre_files: Optional[list] = None

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            t0 = time.time()
            opt = self.options
            mode_name = "精细模式" if opt['quality_mode'] else "快速模式"

            self.log.emit("=" * 50)
            self.log.emit(f"CTA减影处理 [v2.8.6] - {mode_name}")
            self.log.emit("=" * 50)
            
            mem_start = get_memory_mb()
            self.log.emit(f"  初始内存: {mem_start:.0f} MB")

            # ── 1. 读取 ──────────────────────────────────────────
            self.log.emit("\n[1/4] 读取序列...")
            reader = sitk.ImageSeriesReader()
            pre_files = reader.GetGDCMSeriesFileNames(self.data_dir, self.pre_uid)
            post_files = reader.GetGDCMSeriesFileNames(self.data_dir, self.post_uid)
            if not pre_files or not post_files:
                self.finished_signal.emit(False, "未能读取到完整的物理序列数据。")
                return
            
            self._pre_files = list(pre_files)
            
            pre_img = sitk.ReadImage(pre_files, sitk.sitkInt16)
            post_img = sitk.ReadImage(post_files, sitk.sitkInt16)
            self.log.emit(f"  平扫尺寸: {pre_img.GetSize()}")
            self.log.emit(f"  增强尺寸: {post_img.GetSize()}")
            self.log.emit(f"  [内存] 读取后: {get_memory_mb():.0f} MB")

            # ── 2. 3D配准 ────────────────────────────────────────
            self.log.emit("\n[2/4] 3D配准...")
            aligned_img = execute_3d_registration(
                pre_img, post_img, self.log.emit, self.progress.emit)
            
            del post_img
            force_gc()
            self.log.emit(f"  [内存] 释放post_img后: {get_memory_mb():.0f} MB")
            
            if self.cancelled:
                return

            # ── 3. 提取数组 ────────────────────────────────────────
            self.log.emit("\n[3/4] 减影计算...")
            
            pre_vol = sitk.GetArrayFromImage(pre_img).astype(np.float32)
            
            output_spacing = pre_img.GetSpacing()
            del pre_img
            force_gc()
            
            aligned_vol = sitk.GetArrayFromImage(aligned_img)
            
            del aligned_img
            force_gc()
            
            self.log.emit(f"  [内存] 提取数组后: {get_memory_mb():.0f} MB")
            
            total_slices = pre_vol.shape[0]
            os.makedirs(self.output_dir, exist_ok=True)
            series_uid_out = generate_uid()
            mode_str = 'quality' if opt['quality_mode'] else 'fast'

            if opt['quality_mode']:
                self.log.emit("  精细模式: ICA颅底段保护已启用")

            for z in range(total_slices):
                if self.cancelled:
                    return
                pre_s = pre_vol[z]
                post_s = aligned_vol[z]

                if pre_s.max() > -500:
                    diff_pos = np.clip(post_s - pre_s, 0, None)
                    cache = precompute_slice_cache(pre_s, diff_pos, opt['quality_mode'])
                    equip = detect_equipment(pre_s, post_s, body=cache.body)

                    if opt['quality_mode']:
                        res = quality_subtraction_cached(
                            pre_s, post_s, opt['bone_strength'],
                            opt['vessel_sensitivity'], cache)
                    else:
                        res = fast_subtraction(
                            pre_s, post_s, opt['bone_strength'],
                            opt['vessel_sensitivity'], body=cache.body)

                    res[equip] = 0
                    if opt['clean_bone_edges']:
                        res = clean_bone_edges_cached(res, pre_s, cache, mode_str)
                    if opt['smooth_sigma'] > 0:
                        res = edge_preserving_smooth(res, pre_s, opt['smooth_sigma'])

                    aligned_vol[z] = res * opt['vessel_enhance']
                else:
                    aligned_vol[z] = 0

                if z % 50 == 0:
                    self.progress.emit(30 + int((z + 1) / total_slices * 40))

            self.log.emit(f"  [内存] 逐层处理后: {get_memory_mb():.0f} MB")

            result_vol = aligned_vol

            # ── 3D清理 ──────────────────────────────────────────
            self.log.emit("\n[*] 3D拓扑清理...")
            min_vol_3d = opt['min_vessel_size'] * 20
            result_vol = apply_3d_morphological_cleanup_simple(
                result_vol, min_volume=min_vol_3d, log_cb=self.log.emit)
            self.progress.emit(75)

            # 血管自适应缩放 (v2.8.5 内存安全版)
            if opt.get('vessel_scale_enable', False):
                self.log.emit("\n[*] 血管自适应调节...")
                spacing = output_spacing
                spacing_zyx = (spacing[2], spacing[1], spacing[0])
                result_vol = apply_vessel_scaling_simple(
                    result_vol, spacing_zyx,
                    opt['vessel_split_mm'],
                    opt['small_vessel_factor'],
                    opt['large_vessel_factor'],
                    log_cb=self.log.emit)
            self.progress.emit(85)

            self.result_vol = result_vol
            self.pre_vol = pre_vol

            # ── 4. 写出DICOM ─────────────────────────────────────
            self.log.emit("\n[4/4] 写入DICOM...")
            global_min = float(result_vol.min())
            global_max = float(result_vol.max())
            global_slope = ((global_max - global_min) / 4095.0
                            if global_max > global_min else 1.0)

            pre_files_list = self._pre_files
            if len(pre_files_list) != total_slices:
                self.log.emit(f"  [警告] 文件数与层数不符")

            for z, filepath in enumerate(pre_files_list[:total_slices]):
                if self.cancelled:
                    return
                template_ds = pydicom.dcmread(filepath)
                pixel_data = result_vol[z]
                orig_inst = template_ds.get('InstanceNumber', z + 1)
                out_path = os.path.join(self.output_dir, f"SUB_{orig_inst:04d}.dcm")
                
                if global_max > global_min:
                    normalized = (pixel_data - global_min) / (global_max - global_min)
                    pixel_int = (normalized * 4095).astype(np.int16)
                else:
                    pixel_int = np.zeros_like(pixel_data, dtype=np.int16)

                new_ds = template_ds.copy()
                new_ds.PixelData = pixel_int.tobytes()
                new_ds.BitsAllocated = 16
                new_ds.BitsStored = 16
                new_ds.HighBit = 15
                new_ds.PixelRepresentation = 1
                new_ds.SamplesPerPixel = 1
                new_ds.PhotometricInterpretation = 'MONOCHROME2'
                for tag in ['LossyImageCompression', 'LossyImageCompressionRatio',
                            'LossyImageCompressionMethod']:
                    if hasattr(new_ds, tag):
                        delattr(new_ds, tag)
                new_ds.RescaleSlope = global_slope
                new_ds.RescaleIntercept = global_min
                new_ds.WindowCenter = opt['wc']
                new_ds.WindowWidth = opt['ww']
                new_ds.SeriesDescription = "CTA Subtraction v2.8.6"
                new_ds.SeriesInstanceUID = series_uid_out
                new_ds.SOPInstanceUID = generate_uid()
                new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                new_ds.save_as(out_path, write_like_original=False)
                
                del template_ds, new_ds, pixel_int
                
                if z % 50 == 0:
                    self.progress.emit(85 + int((z + 1) / total_slices * 15))

            elapsed = time.time() - t0
            mem_end = get_memory_mb()
            self.log.emit(f"\n完成! 耗时: {elapsed:.1f}s")
            self.log.emit(f"当前内存: {mem_end:.0f} MB")
            self.log.emit(f"保存: {self.output_dir}")
            self.finished_signal.emit(True, f"处理完成!\n耗时: {elapsed:.1f}秒")

        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))


# ════════════════════════════════════════════════════════════════════
# 校准助手
# ════════════════════════════════════════════════════════════════════
def _measure_performance(result_vol: np.ndarray, pre_vol: np.ndarray) -> dict:
    report = {}
    bone_mask = pre_vol > 300
    if bone_mask.sum() > 0:
        r = float((result_vol[bone_mask] > 30).sum()) / bone_mask.sum()
        report['bone_residue_ratio'] = r
        report['bone_residue_pct'] = f"{r * 100:.2f}%"
        report['bone_grade'] = "优" if r < 0.02 else "良" if r < 0.05 else "差"
    else:
        report.update(bone_residue_ratio=0.0, bone_residue_pct="N/A", bone_grade="N/A")

    soft_mask = (pre_vol > -100) & (pre_vol < 80)
    vessel_mask = soft_mask & (result_vol > 20)
    bg_mask = soft_mask & (result_vol >= 0) & (result_vol <= 20)
    if vessel_mask.sum() > 100 and bg_mask.sum() > 100:
        v_mean = float(result_vol[vessel_mask].mean())
        b_std = float(result_vol[bg_mask].std()) + 1e-6
        snr = v_mean / b_std
        report['vessel_snr'] = round(snr, 2)
        report['vessel_mean_hu'] = round(v_mean, 1)
        report['bg_noise_std'] = round(b_std, 1)
        report['vessel_grade'] = "优" if snr >= 4.0 else "良" if snr >= 2.5 else "差"
    else:
        report.update(vessel_snr=0.0, vessel_mean_hu=0.0, bg_noise_std=0.0, vessel_grade="差")

    mid_z = pre_vol.shape[0] // 2
    bone_2d = pre_vol[mid_z] > 150
    edge_2d = (ndimage.binary_dilation(bone_2d, iterations=2) &
               ~ndimage.binary_erosion(bone_2d, iterations=2))
    if edge_2d.sum() > 0:
        er = float(result_vol[mid_z][edge_2d].mean())
        report['edge_residue_mean'] = round(er, 1)
        report['edge_grade'] = "优" if er < 15 else "良" if er < 35 else "差"
    else:
        report.update(edge_residue_mean=0.0, edge_grade="N/A")

    gmap = {"优": 100, "良": 70, "差": 30, "N/A": 70}
    report['total_score'] = round(
        gmap[report['bone_grade']] * 0.4 +
        gmap[report['vessel_grade']] * 0.4 +
        gmap[report['edge_grade']] * 0.2, 1)
    return report


def _generate_advice(report: dict, params: dict) -> list:
    advice = []
    bone_str = params.get('bone_strength', 1.2)
    smooth = params.get('smooth_sigma', 0.7)
    enhance = params.get('vessel_enhance', 2.0)
    clean = params.get('clean_bone_edges', True)

    if report['bone_grade'] == "差":
        advice.append(dict(
            param='bone_strength', label='骨骼抑制强度',
            current=bone_str, suggested=round(min(bone_str + 0.2, 2.5), 1),
            reason=f"骨骼残留率 {report['bone_residue_pct']} 偏高",
            direction='up'))
    if report['vessel_grade'] == "差":
        advice.append(dict(
            param='vessel_enhance', label='血管增强系数',
            current=enhance, suggested=round(min(enhance + 0.3, 4.0), 1),
            reason=f"血管SNR={report['vessel_snr']:.1f} 偏低",
            direction='up'))
    if not advice:
        advice.append(dict(
            param=None, label='✅ 当前参数已达最优',
            current=None, suggested=None,
            reason=f"综合评分 {report['total_score']}/100",
            direction='ok'))
    return advice


class _CalibThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished_sig = pyqtSignal(dict, list)

    def __init__(self, result_vol, pre_vol, params):
        super().__init__()
        self.result_vol = result_vol
        self.pre_vol = pre_vol
        self.params = params

    def run(self):
        try:
            self.log.emit("正在分析...")
            self.progress.emit(50)
            report = _measure_performance(self.result_vol, self.pre_vol)
            self.progress.emit(80)
            advice = _generate_advice(report, self.params)
            self.progress.emit(100)
            self.finished_sig.emit(report, advice)
        except Exception as e:
            self.log.emit(f"分析失败: {e}")
            self.finished_sig.emit({}, [])


class CalibrationDialog(QDialog):
    def __init__(self, parent, proc_thread, current_params, apply_cb=None):
        super().__init__(parent)
        self.proc_thread = proc_thread
        self.current_params = current_params.copy()
        self.apply_cb = apply_cb
        self._adjusted = current_params.copy()
        self.report = {}
        self.advice = []
        self.setWindowTitle("参数校准助手")
        self.setMinimumSize(600, 500)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        
        hdr = QLabel("🔬 参数校准助手")
        hdr.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        root.addWidget(hdr)

        ctrl = QHBoxLayout()
        self.run_btn = QPushButton("▶ 开始分析")
        self.run_btn.clicked.connect(self._start)
        self.prog = QProgressBar()
        ctrl.addWidget(self.run_btn)
        ctrl.addWidget(self.prog, 1)
        root.addLayout(ctrl)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(60)
        root.addWidget(self.log_text)

        self.score_lbl = QLabel("尚未分析")
        self.score_lbl.setWordWrap(True)
        root.addWidget(self.score_lbl)

        self.adv_lay = QVBoxLayout()
        root.addLayout(self.adv_lay)

        br = QHBoxLayout()
        self.apply_btn = QPushButton("✅ 应用建议")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        br.addWidget(self.apply_btn)
        br.addStretch()
        br.addWidget(close_btn)
        root.addLayout(br)

    def _start(self):
        self.run_btn.setEnabled(False)
        self.log_text.clear()
        self._thread = _CalibThread(
            self.proc_thread.result_vol, self.proc_thread.pre_vol, self.current_params)
        self._thread.log.connect(lambda m: self.log_text.append(m))
        self._thread.progress.connect(self.prog.setValue)
        self._thread.finished_sig.connect(self._on_done)
        self._thread.start()

    def _on_done(self, report, advice):
        self.run_btn.setEnabled(True)
        if not report:
            self.score_lbl.setText("❌ 分析失败")
            return
        self.report = report
        self.advice = advice
        
        html = f"<b>综合评分: {report['total_score']}/100</b><br>"
        html += f"骨骼残留: {report['bone_residue_pct']} ({report['bone_grade']})<br>"
        html += f"血管SNR: {report['vessel_snr']:.2f} ({report['vessel_grade']})"
        self.score_lbl.setText(html)
        
        while self.adv_lay.count():
            item = self.adv_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for item in advice:
            lbl = QLabel(f"• {item['label']}: {item['reason']}")
            lbl.setWordWrap(True)
            self.adv_lay.addWidget(lbl)
        
        self.apply_btn.setEnabled(True)

    def _apply(self):
        if self.apply_cb:
            self.apply_cb(self._adjusted)
            QMessageBox.information(self, "已应用", "参数已更新")


# ════════════════════════════════════════════════════════════════════
# 主界面
# ════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.all_series = {}
        self.cta_pairs = []
        self.selected_pre = None
        self.selected_post = None
        self.recommendations: Optional[dict] = None
        self.proc_thread: Optional[ProcessThread] = None
        self._all_presets = load_all_presets()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("头颅CTA减影 v2.8.6 - WHW")
        self.setMinimumSize(1000, 900)
        self.setStyleSheet("""
        QMainWindow,QWidget{background:#F4F5F7;color:#1A1D23;font-family:"Microsoft YaHei",sans-serif;font-size:9pt;}
        QGroupBox{background:#fff;border:1px solid #D0D5DE;border-radius:6px;margin-top:10px;padding:10px 12px;font-weight:600;}
        QGroupBox::title{subcontrol-origin:margin;left:10px;top:0;padding:0 4px;background:#fff;color:#1A6FBF;}
        QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:#fff;border:1px solid #D0D5DE;border-radius:4px;padding:5px 8px;}
        QTextEdit{background:#1C1E26;color:#C8D0E0;border:1px solid #D0D5DE;border-radius:4px;font-family:Consolas,monospace;font-size:8.5pt;}
        QProgressBar{background:#D0D5DE;border:none;border-radius:4px;height:8px;text-align:center;color:transparent;}
        QProgressBar::chunk{background:#1A6FBF;border-radius:4px;}
        """)

        def btn_primary(t):
            b = QPushButton(t)
            b.setStyleSheet(
                "QPushButton{background:#1A6FBF;color:white;border:none;border-radius:4px;padding:6px 16px;font-weight:600;}"
                "QPushButton:disabled{background:#B0BEC5;}")
            return b

        def btn_secondary(t):
            b = QPushButton(t)
            b.setStyleSheet("QPushButton{background:#fff;border:1px solid #D0D5DE;border-radius:4px;padding:5px 12px;}")
            return b

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        header = QWidget()
        header.setStyleSheet("background:#1A6FBF;border-radius:6px;")
        header.setFixedHeight(44)
        h_lay = QHBoxLayout(header)
        title_lbl = QLabel("头颅CTA减影 v2.8.6 WHW")
        title_lbl.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_lbl.setStyleSheet("color:white;")
        sub_lbl = QLabel("自动化内存管理版")
        sub_lbl.setStyleSheet("color:rgba(255,255,255,0.8);font-size:8.5pt;")
        h_lay.addWidget(title_lbl)
        h_lay.addStretch()
        h_lay.addWidget(sub_lbl)
        root.addWidget(header)

        # 1. 数据目录
        dir_group = QGroupBox("1 数据目录")
        dir_lay = QHBoxLayout(dir_group)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("选择DICOM文件夹…")
        browse_btn = btn_secondary("浏览…")
        browse_btn.clicked.connect(
            lambda: self.data_dir_edit.setText(
                QFileDialog.getExistingDirectory(self) or self.data_dir_edit.text()))
        self.scan_btn = btn_primary("扫描")
        self.scan_btn.clicked.connect(self.scan_directory)
        dir_lay.addWidget(QLabel("目录:"))
        dir_lay.addWidget(self.data_dir_edit)
        dir_lay.addWidget(browse_btn)
        dir_lay.addWidget(self.scan_btn)
        root.addWidget(dir_group)

        # 2. 序列配对
        ser_group = QGroupBox("2 序列配对")
        ser_lay = QVBoxLayout(ser_group)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("平扫(C-):"))
        self.pre_combo = QComboBox()
        r1.addWidget(self.pre_combo, 1)
        r1.addWidget(QLabel(" 增强(C+):"))
        self.post_combo = QComboBox()
        r1.addWidget(self.post_combo, 1)
        ser_lay.addLayout(r1)
        r2 = QHBoxLayout()
        self.analyze_btn = btn_primary("🔍 分析图像特征")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_params)
        r2.addWidget(self.analyze_btn)
        r2.addStretch()
        ser_lay.addLayout(r2)
        root.addWidget(ser_group)

        # 3. 输出目录
        out_group = QGroupBox("3 输出目录")
        out_lay = QHBoxLayout(out_group)
        self.out_edit = QLineEdit()
        out_btn = btn_secondary("…")
        out_btn.clicked.connect(
            lambda: self.out_edit.setText(
                QFileDialog.getExistingDirectory(self) or self.out_edit.text()))
        out_lay.addWidget(self.out_edit)
        out_lay.addWidget(out_btn)
        root.addWidget(out_group)

        # 4. 处理模式
        mode_group = QGroupBox("4 处理模式")
        mode_lay = QVBoxLayout(mode_group)
        self.fast_radio = QRadioButton("⚡ 快速模式")
        self.quality_radio = QRadioButton("✨ 精细模式 (ICA保护)")
        self.fast_radio.setChecked(True)
        mr = QHBoxLayout()
        mr.addWidget(self.fast_radio)
        mr.addWidget(self.quality_radio)
        mr.addStretch()
        mode_lay.addLayout(mr)
        self.mode_recommend_label = QLabel("")
        mode_lay.addWidget(self.mode_recommend_label)
        root.addWidget(mode_group)

        # 5. 减影参数
        param_group = QGroupBox("5 减影参数")
        param_vlay = QVBoxLayout(param_group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("设备预设:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(210)
        for name in self._all_presets:
            self.preset_combo.addItem(name, self._all_presets[name])
        self.preset_desc_lbl = QLabel("")
        self.preset_desc_lbl.setStyleSheet("color:#666;font-size:8pt;")
        apply_preset_btn = btn_secondary("应用预设")
        apply_preset_btn.clicked.connect(self.apply_device_preset)
        self.preset_combo.currentIndexChanged.connect(self._update_preset_desc)
        self._update_preset_desc()
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(apply_preset_btn)
        preset_row.addWidget(self.preset_desc_lbl, 1)
        param_vlay.addLayout(preset_row)

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

        self.clean_check = QCheckBox("强力去骨边")
        self.clean_check.setChecked(True)
        p_lay.addWidget(self.clean_check)

        p_lay.addWidget(QLabel(" 降噪:"))
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 1.5)
        self.smooth.setValue(0.7)
        p_lay.addWidget(self.smooth)

        p_lay.addWidget(QLabel(" 窗位/宽:"))
        self.wc = QSpinBox()
        self.wc.setRange(-500, 1000)
        self.wc.setValue(200)
        self.ww = QSpinBox()
        self.ww.setRange(10, 2000)
        self.ww.setValue(400)
        p_lay.addWidget(self.wc)
        p_lay.addWidget(self.ww)
        p_lay.addStretch()
        param_vlay.addLayout(p_lay)

        # 血管自适应调节
        vs_lay = QHBoxLayout()
        self.scale_check = QCheckBox("血管自适应调节(3D)")
        self.scale_check.setChecked(False)
        self.scale_check.setStyleSheet("color:#1A6FBF;font-weight:bold;")
        vs_lay.addWidget(self.scale_check)

        vs_lay.addWidget(QLabel(" 界限(mm):"))
        self.split_mm = QDoubleSpinBox()
        self.split_mm.setRange(0.5, 5.0)
        self.split_mm.setSingleStep(0.1)
        self.split_mm.setValue(1.5)
        vs_lay.addWidget(self.split_mm)

        vs_lay.addWidget(QLabel(" 小血管:"))
        self.small_factor = QDoubleSpinBox()
        self.small_factor.setRange(0.0, 5.0)
        self.small_factor.setSingleStep(0.1)
        self.small_factor.setValue(1.5)
        vs_lay.addWidget(self.small_factor)

        vs_lay.addWidget(QLabel(" 大血管:"))
        self.large_factor = QDoubleSpinBox()
        self.large_factor.setRange(0.0, 5.0)
        self.large_factor.setSingleStep(0.1)
        self.large_factor.setValue(1.0)
        vs_lay.addWidget(self.large_factor)

        vs_lay.addStretch()
        param_vlay.addLayout(vs_lay)
        root.addWidget(param_group)

        # 操作行
        act_lay = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始减影")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#C0392B;color:white;border-radius:5px;font-weight:700;font-size:11pt;padding:0 24px;height:36px;}"
            "QPushButton:disabled{background:#B0BEC5;}")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.cancel_btn = btn_secondary("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.calib_btn = QPushButton("🔬 校准")
        self.calib_btn.setEnabled(False)
        self.calib_btn.setStyleSheet(
            "QPushButton{background:#8E44AD;color:white;border:none;border-radius:4px;padding:6px 14px;font-weight:600;}"
            "QPushButton:disabled{background:#B0BEC5;}")
        self.calib_btn.clicked.connect(self.open_calibration)
        
        act_lay.addWidget(self.start_btn)
        act_lay.addWidget(self.progress, 1)
        act_lay.addWidget(self.cancel_btn)
        act_lay.addWidget(self.calib_btn)
        root.addLayout(act_lay)

        # 日志
        log_group = QGroupBox("运行日志")
        log_lay = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_lay.addWidget(self.log_text)
        root.addWidget(log_group)

        self.log("头颅CTA减影 v2.8.6 - 极限内存安全版")
        self.log("=" * 50)

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())
        QApplication.processEvents()

    def _update_preset_desc(self):
        cfg = self.preset_combo.currentData()
        if cfg:
            self.preset_desc_lbl.setText(cfg.get("description", ""))

    def apply_device_preset(self):
        cfg = self.preset_combo.currentData()
        if not cfg:
            return
        self.bone_slider.setValue(int(cfg["bone_strength"] * 10))
        self.enhance.setValue(cfg["vessel_enhance"])
        self.smooth.setValue(cfg["smooth_sigma"])
        self.clean_check.setChecked(cfg["clean_bone_edges"])
        self.wc.setValue(cfg["wc"])
        self.ww.setValue(cfg["ww"])
        if self.recommendations is None:
            self.recommendations = {}
        self.recommendations["vessel_sensitivity"] = cfg.get("vessel_sensitivity", 1.0)
        self.recommendations["min_vessel_size"] = cfg.get("min_vessel_size", 5)
        self.log(f"✅ 已应用预设: {self.preset_combo.currentText()}")

    def scan_directory(self):
        d = self.data_dir_edit.text()
        if not d:
            return
            
        # [智能释放] 在开始新一轮扫描时，自动彻底清理上一次的缓存
        if self.proc_thread:
            self.proc_thread.result_vol = None
            self.proc_thread.pre_vol = None
            self.proc_thread = None
            self.calib_btn.setEnabled(False)
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
            tag = ('[C+]' if st is True else '[C-]' if st is False else '')
            txt = f"#{s.series_number:03d} | {s.file_count}张 | {tag} {s.series_description[:40]}"
            self.pre_combo.addItem(txt, s)
            self.post_combo.addItem(txt, s)
        if pairs:
            self.pre_combo.setCurrentIndex(self.pre_combo.findData(pairs[0][0]))
            self.post_combo.setCurrentIndex(self.post_combo.findData(pairs[0][1]))
        self.analyze_btn.setEnabled(True)
        if not self.out_edit.text():
            self.out_edit.setText(os.path.join(self.data_dir_edit.text(), "CTA_SUB_v286"))
        ref_series = pairs[0][0] if pairs else (cta_series[0] if cta_series else None)
        if ref_series:
            preset_name = detect_manufacturer_preset(ref_series.files)
            if preset_name:
                idx = self.preset_combo.findText(preset_name)
                if idx >= 0:
                    self.preset_combo.setCurrentIndex(idx)
                self.log(f"🔍 检测到设备: {preset_name}")

    def analyze_params(self):
        pre = self.pre_combo.currentData()
        post = self.post_combo.currentData()
        if not pre or not post or pre == post:
            return QMessageBox.warning(self, "提示", "请选择两个不同的序列")
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
        self.recommendations = res
        
        # ==== 以下为自动回填参数到界面的部分 ====
        self.bone_slider.setValue(int(res.get('bone_strength', 1.2) * 10))
        self.enhance.setValue(res.get('vessel_enhance', 2.0))
        self.clean_check.setChecked(res.get('clean_bone_edges', True))
        self.smooth.setValue(res.get('smooth_sigma', 0.7))
        
        # [新增] 补上窗位和窗宽的自动回填
        if 'wc' in res:
            self.wc.setValue(int(res['wc']))
        if 'ww' in res:
            self.ww.setValue(int(res['ww']))
        # =======================================

        if res.get('recommended_mode', 'fast') == 'fast':
            self.fast_radio.setChecked(True)
            self.mode_recommend_label.setText(f"评分: {res['quality_score']*100:.0f}% → 推荐快速模式")
        else:
            self.quality_radio.setChecked(True)
            self.mode_recommend_label.setText(f"评分: {res['quality_score']*100:.0f}% → 推荐精细模式")
        self.start_btn.setEnabled(True)

    def start_processing(self):
        if not self.out_edit.text():
            return QMessageBox.warning(self, "提示", "输出目录为空")
            
        # [智能释放] 在开始新一轮减影计算时，自动彻底清理上一次的缓存
        if self.proc_thread:
            self.proc_thread.result_vol = None
            self.proc_thread.pre_vol = None
            self.proc_thread = None
            self.calib_btn.setEnabled(False)
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
        }
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.calib_btn.setEnabled(False)
        self.proc_thread = ProcessThread(
            self.data_dir_edit.text(),
            self.selected_pre.series_uid,
            self.selected_post.series_uid,
            self.out_edit.text(),
            opt)
        self.proc_thread.progress.connect(self.progress.setValue)
        self.proc_thread.log.connect(self.log)
        self.proc_thread.finished_signal.connect(self.on_process_finished)
        self.proc_thread.start()

    def cancel(self):
        if self.proc_thread:
            self.proc_thread.cancel()

    def on_process_finished(self, success: bool, msg: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if success:
            self.calib_btn.setEnabled(True)
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "错误", msg)

    def open_calibration(self):
        if self.proc_thread is None or self.proc_thread.result_vol is None:
            return QMessageBox.warning(self, "提示", "请先完成减影处理")
        dlg = CalibrationDialog(
            parent=self, proc_thread=self.proc_thread,
            current_params=self.proc_thread.options,
            apply_cb=self.apply_calibrated_params)
        dlg.exec_()

    def apply_calibrated_params(self, params: dict):
        self.bone_slider.setValue(int(params.get('bone_strength', 1.2) * 10))
        self.enhance.setValue(params.get('vessel_enhance', 2.0))
        self.smooth.setValue(params.get('smooth_sigma', 0.7))
        self.clean_check.setChecked(params.get('clean_bone_edges', True))
        self.wc.setValue(params.get('wc', 200))
        self.ww.setValue(params.get('ww', 400))
        if self.recommendations is None:
            self.recommendations = {}
        self.recommendations['vessel_sensitivity'] = params.get('vessel_sensitivity', 1.0)
        self.recommendations['min_vessel_size'] = params.get('min_vessel_size', 5)
        self.log("🔬 参数已应用")


# ════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())