"""
v2.8 新增功能：血管自适应膨胀系数模块
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[新增] 血管管径自适应调节 (3D形态学)
· 针对小动脉（如眼动脉等约1.5mm管径）狭窄常被误判为堵塞的问题
· 允许基于距离变换（Distance Transform）精准区分大/小血管
· 提供独立的膨胀/腐蚀系数：
  - 小血管系数 > 1 膨胀，防止三维重建时小血管断线误判
  - 大血管/小血管系数 < 1 腐蚀，隐藏无关分支，突显主干动脉瘤等病变
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v2.7 精细模式性能优化版
v2.7.1 补丁：ICA颅底段保护
"""

import os
import re
import sys
import json
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

print("头颅CTA减影 v2.8 - WHW 版 (新增自适应血管膨胀系数)")


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
        "description": "SOMATOM Definition/Force/go，HU校准标准，噪声均匀",
    },
    "GE Revolution/Discovery": {
        "bone_strength": 1.4, "vessel_sensitivity": 0.9, "vessel_enhance": 1.8,
        "smooth_sigma": 0.9, "clean_bone_edges": True, "min_vessel_size": 6,
        "wc": 220, "ww": 420,
        "description": "Revolution CT/Discovery CT750，HU偏高15，噪声颗粒较粗",
    },
    "Philips Brilliance/IQon": {
        "bone_strength": 1.0, "vessel_sensitivity": 1.1, "vessel_enhance": 2.2,
        "smooth_sigma": 0.5, "clean_bone_edges": True, "min_vessel_size": 4,
        "wc": 180, "ww": 380,
        "description": "Brilliance iCT/IQon Spectral，HU偏低10，薄骨保留好",
    },
    "Canon Aquilion": {
        "bone_strength": 1.1, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.6, "clean_bone_edges": False, "min_vessel_size": 5,
        "wc": 200, "ww": 400,
        "description": "Aquilion ONE/GENESIS，颅底伪影较少，骨骼边缘过渡平滑",
    },
    "联影 uCT": {
        "bone_strength": 1.3, "vessel_sensitivity": 1.0, "vessel_enhance": 2.0,
        "smooth_sigma": 0.8, "clean_bone_edges": True, "min_vessel_size": 5,
        "wc": 200, "ww": 400,
        "description": "uCT 960/T+，重建核较锐利，骨骼边缘HU梯度陡，需稍强抑制",
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
    f_roi = fixed[margin:-margin, margin:-margin].astype(np.float64)
    m_roi = moving[margin:-margin, margin:-margin].astype(np.float64)
    wy = np.hanning(f_roi.shape[0])
    wx = np.hanning(f_roi.shape[1])
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
# SimpleITK 3D 掩膜配准引擎
# ════════════════════════════════════════════════════════════════════
def execute_3d_registration(fixed_img, moving_img, log_cb=None, progress_cb=None):
    fixed_img = sitk.Cast(fixed_img, sitk.sitkFloat32)
    moving_img = sitk.Cast(moving_img, sitk.sitkFloat32)
    if log_cb:
        log_cb("  正在生成 3D 骨骼掩膜 (HU > 200)...")
    bone_mask = sitk.BinaryThreshold(fixed_img, 200.0, 3000.0, 1, 0)
    bone_mask = sitk.Cast(bone_mask, sitk.sitkUInt8)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricFixedMask(bone_mask)
    fixed_size = fixed_img.GetSize()
    center_index = [(s - 1) / 2.0 for s in fixed_size]
    center_physical = fixed_img.TransformContinuousIndexToPhysicalPoint(center_index)
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
        log_cb("  开始 3D 空间微调优化 (防翻滚保护已启动)...")
    final_transform = R.Execute(fixed_img, moving_img)
    if log_cb:
        log_cb(f"  3D配准完成. 最终MSE: {R.GetMetricValue():.2f}")
        params = final_transform.GetParameters()
        if len(params) == 6:
            angles = np.degrees(params[:3])
            trans = params[3:]
            log_cb(f"  [监测] 旋转偏角: X={angles[0]:.2f}° Y={angles[1]:.2f}° Z={angles[2]:.2f}°")
            log_cb(f"  [监测] 平移距离: X={trans[0]:.2f} Y={trans[1]:.2f} Z={trans[2]:.2f} mm")
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

    # ── 公共计算 ──────────────────────────────────────────────
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

    # ── 精细模式专用 ──────────────────────────────────────────
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
# 3D 拓扑清理 & 血管自适应膨胀缩放 [v2.8 新增]
# ════════════════════════════════════════════════════════════════════
def apply_3d_morphological_cleanup(volume: np.ndarray,
                                   pre_vol: np.ndarray,
                                   min_volume: int = 100) -> np.ndarray:
    pos_vox = volume[volume > 0]
    if pos_vox.size == 0:
        return volume
    fg_threshold = max(10.0, float(np.percentile(pos_vox, 10)))
    mask = volume > fg_threshold
    labeled_vol, num_features = ndimage.label(mask)
    if num_features == 0:
        return volume

    mid_z = pre_vol.shape[0] // 2
    mid_slice = pre_vol[mid_z]
    dense_bone_2d = mid_slice > 400
    petrous_2d = ndimage.binary_dilation(dense_bone_2d, iterations=3)
    body_2d = create_body_mask(mid_slice)
    body_eroded_2d = ndimage.binary_erosion(body_2d, iterations=8)
    scalp_2d = body_2d & ~body_eroded_2d

    multiplier_2d = np.ones(mid_slice.shape, dtype=np.uint8)
    multiplier_2d[scalp_2d] = 2
    multiplier_2d[petrous_2d] = 3

    sizes = np.bincount(labeled_vol.ravel())
    sizes[0] = 0

    mult_3d = np.broadcast_to(multiplier_2d[np.newaxis, :, :],
                               volume.shape).copy()
    label_max_mult = ndimage.maximum(mult_3d, labeled_vol,
                                     range(1, num_features + 1))
    label_max_mult = np.array(label_max_mult, dtype=np.float32)
    effective_thresholds = min_volume * label_max_mult

    label_sizes = sizes[1:num_features + 1].astype(np.float32)
    keep_labels = np.zeros(num_features + 1, dtype=bool)
    keep_labels[0] = False
    keep_labels[1:] = label_sizes >= effective_thresholds
    keep_mask = keep_labels[labeled_vol]

    noise_region = mask & ~keep_mask
    volume[noise_region & (volume < 60)] = 0
    volume[noise_region & (volume >= 60)] *= 0.3

    del mult_3d
    return volume


def apply_vessel_scaling(volume: np.ndarray, spacing: tuple,
                         split_mm: float, small_factor: float, large_factor: float) -> np.ndarray:
    """
    [v2.8 新增] 自适应分离大小血管，并分别予以膨胀或腐蚀。
    基于欧氏距离变换与形态学重建剥离大小血管，实现互不干扰的缩放。
    """
    if small_factor == 1.0 and large_factor == 1.0:
        return volume

    # 提取显著的血管掩膜（阈值可根据实际增强强度稍微放宽）
    mask = volume > 20
    if not np.any(mask):
        return volume

    # 1. 计算三维距离变换，提取不同管径的核心轴线
    # spacing 必须是 (dz, dy, dx) 顺序
    dt = ndimage.distance_transform_edt(mask, sampling=spacing)
    r_thresh = split_mm / 2.0  # 半径阈值

    # 超过指定半径阈值的部分被视为大血管核心
    large_core = dt > r_thresh
    large_mask = large_core.copy()

    # 形态学重建：将被大血管核心覆盖的区域复原成完整的大血管树枝
    max_iters = int(np.ceil(r_thresh / min(spacing))) + 3
    struct = ndimage.generate_binary_structure(3, 1)
    for _ in range(max_iters):
        next_mask = ndimage.binary_dilation(large_mask, structure=struct) & mask
        if np.array_equal(next_mask, large_mask):
            break
        large_mask = next_mask

    # 剩余的即为小血管
    small_mask = mask & ~large_mask

    # 2. 根据用户系数，分别对掩膜应用形态学映射
    def process_mask(target_mask, factor):
        if factor == 1.0:
            return volume * target_mask
        if factor == 0.0:
            return np.zeros_like(volume)

        # 估算迭代次数：以 1.0 为基准
        iters = int(round(abs(factor - 1.0) * 3))
        if iters == 0 and factor != 1.0:
            iters = 1

        if factor > 1.0:
            # 膨胀：采用最大极值滤波保留CT高亮信息
            vol = volume * target_mask
            grow_mask = target_mask.copy()
            for _ in range(iters):
                grow_mask = ndimage.binary_dilation(grow_mask, structure=struct)
                vol = ndimage.maximum_filter(vol, footprint=struct) * grow_mask
            return vol
        else:
            # 腐蚀：逐渐收缩并截断细枝
            eroded = target_mask.copy()
            for _ in range(iters):
                eroded = ndimage.binary_erosion(eroded, structure=struct)
            return volume * eroded

    small_vol = process_mask(small_mask, small_factor)
    large_vol = process_mask(large_mask, large_factor)

    # 3. 三维无缝拼合
    new_vessel_mask = (small_vol > 0) | (large_vol > 0)
    final_vol = volume.copy()
    # 先清理掉原本的血管痕迹，避免腐蚀后依然有残留
    final_vol[mask] = 0  

    # 解决膨胀重叠问题：取最大值
    combined_new = np.maximum(small_vol, large_vol)
    final_vol[new_vessel_mask] = combined_new[new_vessel_mask]

    return final_vol


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

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            t0 = time.time()
            opt = self.options
            mode_name = "精细模式" if opt['quality_mode'] else "快速模式"

            self.log.emit("=" * 50)
            self.log.emit(f"CTA减影处理 [v2.8] - {mode_name}")
            self.log.emit("=" * 50)

            # ── 1. 读取 ──────────────────────────────────────────
            self.log.emit("\n[1/4] 读取并解析绝对物理坐标 3D 容积...")
            reader = sitk.ImageSeriesReader()
            pre_files = reader.GetGDCMSeriesFileNames(self.data_dir, self.pre_uid)
            post_files = reader.GetGDCMSeriesFileNames(self.data_dir, self.post_uid)
            if not pre_files or not post_files:
                self.finished_signal.emit(False, "未能读取到完整的物理序列数据。")
                return
            pre_img = sitk.ReadImage(pre_files)
            post_img = sitk.ReadImage(post_files)
            self.log.emit(f"  平扫空间尺寸: {pre_img.GetSize()}")
            self.log.emit(f"  增强空间尺寸: {post_img.GetSize()}")

            # ── 2. 3D配准 ────────────────────────────────────────
            self.log.emit("\n[2/4] 进行 3D 骨骼掩膜锚定配准...")
            aligned_img = execute_3d_registration(
                pre_img, post_img, self.log.emit, self.progress.emit)
            if self.cancelled:
                return

            # ── 3. 逐层减影 ──────────────────────────────────────
            self.log.emit("\n[3/4] 开始智能减影计算...")
            pre_vol = sitk.GetArrayFromImage(pre_img).astype(np.float32)
            aligned_vol = sitk.GetArrayFromImage(aligned_img).astype(np.float32)
            result_vol = np.zeros_like(pre_vol)
            total_slices = pre_vol.shape[0]
            os.makedirs(self.output_dir, exist_ok=True)
            series_uid_out = generate_uid()
            mode_str = 'quality' if opt['quality_mode'] else 'fast'

            if opt['quality_mode']:
                self.log.emit("  [v2.7.1+] 精细模式: 预计算缓存 + ICA颅底段保护")

            for z in range(total_slices):
                if self.cancelled:
                    return
                pre_s = pre_vol[z]
                post_s = aligned_vol[z]

                if pre_s.max() > -500:
                    diff_pos = np.clip(post_s - pre_s, 0, None)
                    cache = precompute_slice_cache(
                        pre_s, diff_pos, opt['quality_mode'])

                    equip = detect_equipment(pre_s, post_s, body=cache.body)

                    if opt['quality_mode']:
                        res = quality_subtraction_cached(
                            pre_s, post_s,
                            opt['bone_strength'], opt['vessel_sensitivity'],
                            cache)
                    else:
                        res = fast_subtraction(
                            pre_s, post_s,
                            opt['bone_strength'], opt['vessel_sensitivity'],
                            body=cache.body)

                    res[equip] = 0
                    if opt['clean_bone_edges']:
                        res = clean_bone_edges_cached(res, pre_s, cache, mode_str)
                    if opt['smooth_sigma'] > 0:
                        res = edge_preserving_smooth(res, pre_s, opt['smooth_sigma'])

                    result_vol[z] = res * opt['vessel_enhance']

                self.progress.emit(30 + int((z + 1) / total_slices * 40))

            # ── 3D清理与形态学调整 ──────────────────────────────────
            self.log.emit("\n[*] 3D 全局拓扑连续性分析...")
            min_vol_3d = opt['min_vessel_size'] * 20
            result_vol = apply_3d_morphological_cleanup(
                result_vol, pre_vol, min_volume=min_vol_3d)
            self.progress.emit(75)

            # [v2.8] 血管自适应膨胀/收缩模块
            if opt.get('vessel_scale_enable', False):
                self.log.emit("\n[*] 正在应用血管自适应管径调节(3D Distance Transform)...")
                spacing = pre_img.GetSpacing()  # (dx, dy, dz)
                spacing_zyx = (spacing[2], spacing[1], spacing[0])
                result_vol = apply_vessel_scaling(
                    result_vol, spacing_zyx,
                    opt['vessel_split_mm'],
                    opt['small_vessel_factor'],
                    opt['large_vessel_factor']
                )
                self.log.emit("  自适应调节完成。")
            self.progress.emit(85)

            self.result_vol = result_vol
            self.pre_vol = pre_vol

            # ── 4. 写出DICOM ─────────────────────────────────────
            self.log.emit("\n[4/4] 保持原生几何拓扑并写入 DICOM...")
            global_min = float(result_vol.min())
            global_max = float(result_vol.max())
            global_slope = ((global_max - global_min) / 4095.0
                            if global_max > global_min else 1.0)
            if len(pre_files) != total_slices:
                self.log.emit(
                    f"  [警告] DICOM文件数({len(pre_files)})与"
                    f"体积层数({total_slices})不符，将按实际层数截断")

            for z, filepath in enumerate(pre_files[:total_slices]):
                if self.cancelled:
                    return
                template_ds = pydicom.dcmread(filepath)
                pixel_data = result_vol[z]
                orig_inst = template_ds.get('InstanceNumber', z + 1)
                out_path = os.path.join(self.output_dir,
                                        f"SUB_{orig_inst:04d}.dcm")
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
                for tag in ['LossyImageCompression',
                            'LossyImageCompressionRatio',
                            'LossyImageCompressionMethod']:
                    if hasattr(new_ds, tag):
                        delattr(new_ds, tag)
                new_ds.RescaleSlope = global_slope
                new_ds.RescaleIntercept = global_min
                new_ds.WindowCenter = opt['wc']
                new_ds.WindowWidth = opt['ww']
                new_ds.SeriesDescription = "CTA Subtraction v2.8"
                new_ds.SeriesInstanceUID = series_uid_out
                new_ds.SOPInstanceUID = generate_uid()
                new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                new_ds.save_as(out_path, write_like_original=False)
                self.progress.emit(85 + int((z + 1) / total_slices * 15))

            elapsed = time.time() - t0
            self.log.emit(
                f"\n全部完成! 耗时: {elapsed:.1f} 秒\n文件保存在:\n{self.output_dir}")
            self.finished_signal.emit(
                True,
                f"处理完成!\n耗时: {elapsed:.1f} 秒\n保存目录: {self.output_dir}")

        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))


# ════════════════════════════════════════════════════════════════════
# 校准助手
# ════════════════════════════════════════════════════════════════════
def _measure_performance(result_vol: np.ndarray,
                         pre_vol: np.ndarray) -> dict:
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
        report.update(vessel_snr=0.0, vessel_mean_hu=0.0,
                      bg_noise_std=0.0, vessel_grade="差")

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
            reason=f"骨骼残留率 {report['bone_residue_pct']} 偏高（目标<2%）",
            direction='up'))
        if not clean:
            advice.append(dict(
                param='clean_bone_edges', label='强力去骨边',
                current=False, suggested=True,
                reason="骨骼残留明显，建议开启强力去骨边",
                direction='up'))
    if report['bone_grade'] == "优" and report['vessel_grade'] == "差":
        advice.append(dict(
            param='bone_strength', label='骨骼抑制强度',
            current=bone_str, suggested=round(max(bone_str - 0.1, 0.8), 1),
            reason="抑制过强可能误压血管，建议小幅降低",
            direction='down'))
    if report['vessel_grade'] == "差":
        advice.append(dict(
            param='vessel_enhance', label='血管增强系数',
            current=enhance, suggested=round(min(enhance + 0.3, 4.0), 1),
            reason=f"血管SNR={report['vessel_snr']:.1f} 偏低（目标>4.0）",
            direction='up'))
    if report.get('bg_noise_std', 0) > 25:
        advice.append(dict(
            param='smooth_sigma', label='降噪平滑',
            current=smooth, suggested=round(min(smooth + 0.2, 1.5), 1),
            reason=f"背景噪声σ={report['bg_noise_std']:.1f} 偏大",
            direction='up'))
    if report['edge_grade'] == "差":
        advice.append(dict(
            param='clean_bone_edges', label='强力去骨边',
            current=clean, suggested=True,
            reason=f"骨缘残影均值={report['edge_residue_mean']:.1f} HU（目标<15）",
            direction='up'))
    if not advice:
        advice.append(dict(
            param=None, label='✅ 当前参数已达最优',
            current=None, suggested=None,
            reason=f"综合评分 {report['total_score']}/100，各指标均在优良范围",
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
            self.log.emit("正在分析骨骼残留率...")
            self.progress.emit(25)
            self.log.emit("正在计算血管信噪比...")
            self.progress.emit(55)
            report = _measure_performance(self.result_vol, self.pre_vol)
            self.progress.emit(80)
            self.log.emit("正在生成参数调整建议...")
            advice = _generate_advice(report, self.params)
            self.progress.emit(100)
            self.finished_sig.emit(report, advice)
        except Exception as e:
            self.log.emit(f"分析失败: {e}")
            self.finished_sig.emit({}, [])


class CalibrationDialog(QDialog):
    def __init__(self, parent, proc_thread: "ProcessThread",
                 current_params: dict, apply_cb=None):
        super().__init__(parent)
        self.proc_thread = proc_thread
        self.current_params = current_params.copy()
        self.apply_cb = apply_cb
        self._adjusted = current_params.copy()
        self.report = {}
        self.advice = []

        self.setWindowTitle("参数校准助手 v2.8")
        self.setMinimumSize(700, 740)
        self.setStyleSheet("""
            QDialog,QWidget{background:#F4F5F7;
                font-family:"Microsoft YaHei",sans-serif;font-size:9pt;}
            QGroupBox{background:#fff;border:1px solid #D0D5DE;
                border-radius:6px;margin-top:10px;padding:10px 12px;font-weight:600;}
            QGroupBox::title{subcontrol-origin:margin;left:10px;top:0;
                padding:0 4px;background:#fff;color:#1A6FBF;}
            QTextEdit{background:#1C1E26;color:#C8D0E0;border-radius:4px;
                font-family:Consolas,monospace;font-size:8.5pt;}
            QProgressBar{background:#D0D5DE;border:none;border-radius:4px;
                height:8px;color:transparent;}
            QProgressBar::chunk{background:#1A6FBF;border-radius:4px;}
        """)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        hdr = QLabel("🔬  参数校准助手")
        hdr.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        hdr.setStyleSheet("color:#1A6FBF;padding:4px 0;")
        root.addWidget(hdr)

        sub = QLabel("对本次减影结果做定量评估，给出骨骼残留率 / 血管SNR / 骨缘残影三项指标及参数建议")
        sub.setStyleSheet("color:#666;font-size:8.5pt;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        ctrl = QHBoxLayout()
        self.run_btn = QPushButton("▶  开始分析")
        self.run_btn.setStyleSheet(
            "QPushButton{background:#1A6FBF;color:white;border:none;"
            "border-radius:4px;padding:6px 20px;font-weight:600;}"
            "QPushButton:disabled{background:#B0BEC5;}")
        self.run_btn.clicked.connect(self._start)
        self.prog = QProgressBar()
        ctrl.addWidget(self.run_btn)
        ctrl.addWidget(self.prog, 1)
        root.addLayout(ctrl)

        log_grp = QGroupBox("分析日志")
        ll = QVBoxLayout(log_grp)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(70)
        ll.addWidget(self.log_text)
        root.addWidget(log_grp)

        score_grp = QGroupBox("定量评估报告")
        sl = QVBoxLayout(score_grp)
        self.score_lbl = QLabel("尚未分析")
        self.score_lbl.setWordWrap(True)
        sl.addWidget(self.score_lbl)
        root.addWidget(score_grp)

        adv_grp = QGroupBox("参数调整建议（可直接微调后应用）")
        self.adv_lay = QVBoxLayout(adv_grp)
        self.adv_ph = QLabel("分析完成后自动显示建议")
        self.adv_ph.setStyleSheet("color:#aaa;")
        self.adv_lay.addWidget(self.adv_ph)
        root.addWidget(adv_grp)

        save_grp = QGroupBox("保存为设备预设")
        sv = QHBoxLayout(save_grp)
        sv.addWidget(QLabel("预设名称:"))
        self.preset_name = QLineEdit()
        self.preset_name.setPlaceholderText("例如：联影uCT960-急诊方案")
        sv.addWidget(self.preset_name, 1)
        self.save_btn = QPushButton("💾  保存")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(
            "QPushButton{background:#27AE60;color:white;border:none;"
            "border-radius:4px;padding:5px 14px;}"
            "QPushButton:disabled{background:#B0BEC5;}")
        self.save_btn.clicked.connect(self._save)
        sv.addWidget(self.save_btn)
        root.addWidget(save_grp)

        br = QHBoxLayout()
        self.apply_btn = QPushButton("✅  应用建议参数到主界面")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet(
            "QPushButton{background:#C0392B;color:white;border:none;"
            "border-radius:4px;padding:7px 20px;font-weight:700;}"
            "QPushButton:disabled{background:#B0BEC5;}")
        self.apply_btn.clicked.connect(self._apply)
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet(
            "QPushButton{background:#fff;border:1px solid #D0D5DE;"
            "border-radius:4px;padding:6px 16px;}")
        close_btn.clicked.connect(self.close)
        br.addWidget(self.apply_btn)
        br.addStretch()
        br.addWidget(close_btn)
        root.addLayout(br)

    def _start(self):
        self.run_btn.setEnabled(False)
        self.log_text.clear()
        self._thread = _CalibThread(
            self.proc_thread.result_vol,
            self.proc_thread.pre_vol,
            self.current_params)
        self._thread.log.connect(lambda m: self.log_text.append(m))
        self._thread.progress.connect(self.prog.setValue)
        self._thread.finished_sig.connect(self._on_done)
        self._thread.start()

    def _on_done(self, report, advice):
        self.run_btn.setEnabled(True)
        if not report:
            self.score_lbl.setText("❌ 分析失败，请查看日志")
            return
        self.report = report
        self.advice = advice

        gc = {"优": "#27AE60", "良": "#F39C12", "差": "#E74C3C", "N/A": "#888"}

        def colored(g):
            return f'<span style="color:{gc.get(g, "#333")};font-weight:700;">{g}</span>'

        html = f"""
        <table cellspacing="5" style="font-size:9pt;">
          <tr><td width="160"><b>综合评分</b></td>
              <td><b style="font-size:13pt;color:#1A6FBF;">{report['total_score']}/100</b></td></tr>
          <tr><td>骨骼残留率</td>
              <td>{report['bone_residue_pct']} &nbsp; {colored(report['bone_grade'])}</td></tr>
          <tr><td>血管信噪比</td>
              <td>{report['vessel_snr']:.2f} &nbsp; {colored(report['vessel_grade'])}
              <span style="color:#999;font-size:8pt;">
                （均值={report['vessel_mean_hu']:.0f} / 背景σ={report['bg_noise_std']:.1f}）
              </span></td></tr>
          <tr><td>骨缘残影</td>
              <td>{report['edge_residue_mean']:.1f} HU &nbsp; {colored(report['edge_grade'])}</td></tr>
        </table>"""
        self.score_lbl.setText(html)
        self._render_advice(advice)
        self.apply_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def _render_advice(self, advice):
        while self.adv_lay.count():
            item = self.adv_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._spinboxes = {}
        for item in advice:
            frame = QFrame()
            frame.setStyleSheet(
                "QFrame{background:#F8F9FA;border:1px solid #E0E4EB;border-radius:5px;}")
            fl = QHBoxLayout(frame)
            fl.setContentsMargins(10, 6, 10, 6)
            txt_col = QVBoxLayout()
            txt_col.addWidget(QLabel(f"<b>{item['label']}</b>"))
            rl = QLabel(item['reason'])
            rl.setStyleSheet("color:#555;font-size:8.5pt;")
            rl.setWordWrap(True)
            txt_col.addWidget(rl)
            fl.addLayout(txt_col, 1)
            param = item['param']
            if param and item['suggested'] is not None:
                if isinstance(item['suggested'], bool):
                    cb = QCheckBox()
                    cb.setChecked(item['suggested'])
                    cb.stateChanged.connect(
                        lambda state, p=param:
                        self._adjusted.__setitem__(p, state == Qt.Checked))
                    self._adjusted[param] = item['suggested']
                    fl.addWidget(QLabel(
                        f"{'开' if item['current'] else '关'} → "))
                    fl.addWidget(cb)
                    self._spinboxes[param] = cb
                else:
                    arrow = "↑" if item['direction'] == 'up' else "↓"
                    fl.addWidget(QLabel(f"{item['current']}  {arrow}"))
                    spn = QDoubleSpinBox()
                    spn.setRange(0.1, 5.0)
                    spn.setSingleStep(0.1)
                    spn.setValue(item['suggested'])
                    spn.setFixedWidth(80)
                    spn.valueChanged.connect(
                        lambda v, p=param:
                        self._adjusted.__setitem__(p, round(v, 2)))
                    self._adjusted[param] = item['suggested']
                    fl.addWidget(spn)
                    self._spinboxes[param] = spn
            self.adv_lay.addWidget(frame)

    def _apply(self):
        if self.apply_cb:
            merged = self.current_params.copy()
            merged.update(self._adjusted)
            self.apply_cb(merged)
            QMessageBox.information(
                self, "已应用",
                "建议参数已更新到主界面。\n可直接点击「开始减影」验证效果。")

    def _save(self):
        name = self.preset_name.text().strip()
        if not name:
            return QMessageBox.warning(self, "提示", "请输入预设名称")
        merged = self.current_params.copy()
        merged.update(self._adjusted)
        merged['description'] = (
            f"评分{self.report.get('total_score', '?')}/100"
            f"（用户校准 {datetime.now().strftime('%Y-%m-%d')}）")
        save_custom_preset(name, merged)
        QMessageBox.information(
            self, "已保存",
            f"预设「{name}」已保存到:\n{PRESET_FILE}\n\n"
            f"重启程序后可在「设备预设」下拉框中选择。")


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
        self.setWindowTitle("头颅CTA减影 v2.8 - WHW 版")
        self.setMinimumSize(1000, 1020)
        self.setStyleSheet("""
        QMainWindow,QWidget{background:#F4F5F7;color:#1A1D23;
            font-family:"Microsoft YaHei",sans-serif;font-size:9pt;}
        QGroupBox{background:#fff;border:1px solid #D0D5DE;border-radius:6px;
            margin-top:10px;padding:10px 12px;font-weight:600;}
        QGroupBox::title{subcontrol-origin:margin;left:10px;top:0;
            padding:0 4px;background:#fff;color:#1A6FBF;}
        QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:#fff;
            border:1px solid #D0D5DE;border-radius:4px;padding:5px 8px;}
        QTextEdit{background:#1C1E26;color:#C8D0E0;border:1px solid #D0D5DE;
            border-radius:4px;font-family:Consolas,monospace;font-size:8.5pt;}
        QProgressBar{background:#D0D5DE;border:none;border-radius:4px;
            height:8px;text-align:center;color:transparent;}
        QProgressBar::chunk{background:#1A6FBF;border-radius:4px;}
        """)

        def btn_primary(t):
            b = QPushButton(t)
            b.setStyleSheet(
                "QPushButton{background:#1A6FBF;color:white;border:none;"
                "border-radius:4px;padding:6px 16px;font-weight:600;}"
                "QPushButton:disabled{background:#B0BEC5;}")
            return b

        def btn_secondary(t):
            b = QPushButton(t)
            b.setStyleSheet(
                "QPushButton{background:#fff;border:1px solid #D0D5DE;"
                "border-radius:4px;padding:5px 12px;}")
            return b

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        header = QWidget()
        header.setStyleSheet("background:#1A6FBF;border-radius:6px;")
        header.setFixedHeight(44)
        h_lay = QHBoxLayout(header)
        title_lbl = QLabel("头颅CTA 数字减影  v2.8 WHW 版")
        title_lbl.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_lbl.setStyleSheet("color:white;")
        sub_lbl = QLabel("自适应管径缩放 · ICA保护 · 预计算缓存 · 设备预设 · 校准助手")
        sub_lbl.setStyleSheet("color:rgba(255,255,255,0.8);font-size:8.5pt;")
        h_lay.addWidget(title_lbl)
        h_lay.addStretch()
        h_lay.addWidget(sub_lbl)
        root.addWidget(header)

        # 1. 数据目录
        dir_group = QGroupBox("1   数据目录")
        dir_lay = QHBoxLayout(dir_group)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("选择 DICOM 文件夹…")
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
        ser_group = QGroupBox("2   序列配对")
        ser_lay = QVBoxLayout(ser_group)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("平扫 (C-):"))
        self.pre_combo = QComboBox()
        r1.addWidget(self.pre_combo, 1)
        r1.addWidget(QLabel("  增强 (C+):"))
        self.post_combo = QComboBox()
        r1.addWidget(self.post_combo, 1)
        ser_lay.addLayout(r1)
        r2 = QHBoxLayout()
        self.analyze_btn = btn_primary("🔍  分析图像特征")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_params)
        r2.addWidget(self.analyze_btn)
        r2.addStretch()
        ser_lay.addLayout(r2)
        root.addWidget(ser_group)

        # 3. 输出目录
        out_group = QGroupBox("3   输出目录")
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
        mode_group = QGroupBox("4   处理模式")
        mode_lay = QVBoxLayout(mode_group)
        self.fast_radio = QRadioButton("⚡ 快速模式 (适合常规质量)(建议首选试用)")
        self.quality_radio = QRadioButton("✨ 精细模式 (颅底伪影抑制 · ICA保护 · 静脉窦保护)")
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
        param_group = QGroupBox("5   减影参数")
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
        self.bone_slider.valueChanged.connect(
            lambda v: self.bone_label.setText(f"{v / 10:.1f}"))
        p_lay.addWidget(self.bone_slider)
        p_lay.addWidget(self.bone_label)

        p_lay.addWidget(QLabel("  血管增强:"))
        self.enhance = QDoubleSpinBox()
        self.enhance.setRange(0.5, 5.0)
        self.enhance.setValue(2.0)
        p_lay.addWidget(self.enhance)

        self.clean_check = QCheckBox("强力去骨边")
        self.clean_check.setChecked(True)
        p_lay.addWidget(self.clean_check)

        p_lay.addWidget(QLabel("  降噪平滑:"))
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 1.5)
        self.smooth.setValue(0.7)
        p_lay.addWidget(self.smooth)

        p_lay.addWidget(QLabel("  窗位/宽:"))
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

        # [v2.8] 新增模块：自适应管径膨胀
        vs_lay = QHBoxLayout()
        self.scale_check = QCheckBox("血管自适应调节(3D形态学)")
        self.scale_check.setChecked(False) # 默认关闭，供特定需要时开启
        self.scale_check.setToolTip("开启后，可基于物理管径独立缩放大小血管。如针对闭塞疑团开启小血管膨胀。")
        self.scale_check.setStyleSheet("color:#1A6FBF; font-weight:bold;")
        vs_lay.addWidget(self.scale_check)

        vs_lay.addWidget(QLabel("  大小血管界限(mm):"))
        self.split_mm = QDoubleSpinBox()
        self.split_mm.setRange(0.5, 5.0)
        self.split_mm.setSingleStep(0.1)
        self.split_mm.setValue(1.5)
        self.split_mm.setToolTip("管径小于该值的将被视为小血管")
        vs_lay.addWidget(self.split_mm)

        vs_lay.addWidget(QLabel("  小血管系数:"))
        self.small_factor = QDoubleSpinBox()
        self.small_factor.setRange(0.0, 5.0)
        self.small_factor.setSingleStep(0.1)
        self.small_factor.setValue(1.5)
        self.small_factor.setToolTip("大于1产生膨胀以防断线；小于1产生腐蚀。")
        vs_lay.addWidget(self.small_factor)

        vs_lay.addWidget(QLabel("  大血管系数:"))
        self.large_factor = QDoubleSpinBox()
        self.large_factor.setRange(0.0, 5.0)
        self.large_factor.setSingleStep(0.1)
        self.large_factor.setValue(1.0)
        self.large_factor.setToolTip("例如设为0.5腐蚀掉非主干血管，突出大动脉瘤。")
        vs_lay.addWidget(self.large_factor)

        vs_lay.addStretch()
        param_vlay.addLayout(vs_lay)

        root.addWidget(param_group)

        # 操作行
        act_lay = QHBoxLayout()
        self.start_btn = QPushButton("▶  开始减影")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#C0392B;color:white;border-radius:5px;"
            "font-weight:700;font-size:11pt;padding:0 24px;height:36px;}"
            "QPushButton:disabled{background:#B0BEC5;}")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.cancel_btn = btn_secondary("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.calib_btn = QPushButton("🔬 校准助手")
        self.calib_btn.setEnabled(False)
        self.calib_btn.setStyleSheet(
            "QPushButton{background:#8E44AD;color:white;border:none;"
            "border-radius:4px;padding:6px 14px;font-weight:600;}"
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

        self.log("头颅CTA减影 v2.8 - WHW 版")
        self.log("=" * 50)
        self.log("⚡ 快速模式: 扫描质量好时使用，速度快")
        self.log("✨ 精细模式: 预计算缓存 + ICA颅底段保护 + 静脉窦保护")
        self.log("🌀 血管膨胀: 按设定毫米级区分大小血管管径，自适应膨胀/收缩")
        self.log("=" * 50)
        self.log("使用步骤:")
        self.log("  1. 选择DICOM目录 → 扫描")
        self.log("  2. 确认序列配对")
        self.log("  3. 分析图像特征 → 自动推荐模式")
        self.log("  4. 开始减影")
        self.log("  5. 若VR小血管易断线，勾选「血管自适应调节」，调高小血管系数。")

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
        self.log(f"✅ 已应用设备预设: {self.preset_combo.currentText()}")

    def scan_directory(self):
        d = self.data_dir_edit.text()
        if not d:
            return
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
            self.out_edit.setText(
                os.path.join(self.data_dir_edit.text(), "CTA_SUB_v28"))
        ref_series = pairs[0][0] if pairs else (cta_series[0] if cta_series else None)
        if ref_series:
            preset_name = detect_manufacturer_preset(ref_series.files)
            if preset_name:
                idx = self.preset_combo.findText(preset_name)
                if idx >= 0:
                    self.preset_combo.setCurrentIndex(idx)
                self.log(f"🔍 检测到设备厂商，已自动匹配预设: {preset_name}")

    def analyze_params(self):
        pre = self.pre_combo.currentData()
        post = self.post_combo.currentData()
        if not pre or not post or pre == post:
            return QMessageBox.warning(self, "提示", "请选择两个不同的有效序列")
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
        self.bone_slider.setValue(int(res.get('bone_strength', 1.2) * 10))
        self.enhance.setValue(res.get('vessel_enhance', 2.0))
        self.clean_check.setChecked(res.get('clean_bone_edges', True))
        self.smooth.setValue(res.get('smooth_sigma', 0.7))
        if res.get('recommended_mode', 'fast') == 'fast':
            self.fast_radio.setChecked(True)
            self.mode_recommend_label.setText(
                f"图像评分: {res['quality_score'] * 100:.0f}% → 推荐「快速模式」")
        else:
            self.quality_radio.setChecked(True)
            self.mode_recommend_label.setText(
                f"图像评分: {res['quality_score'] * 100:.0f}% → 推荐「精细模式」")
        self.start_btn.setEnabled(True)

    def start_processing(self):
        if not self.out_edit.text():
            return QMessageBox.warning(self, "提示", "输出目录为空")
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
            # [v2.8 新增]
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
            return QMessageBox.warning(
                self, "提示", "请先完成一次减影处理再使用校准助手")
        dlg = CalibrationDialog(
            parent=self,
            proc_thread=self.proc_thread,
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
        self.log("🔬 校准助手：建议参数已应用，可重新减影验证效果")
        self._all_presets = load_all_presets()
        current_text = self.preset_combo.currentText()
        self.preset_combo.clear()
        for name, cfg in self._all_presets.items():
            self.preset_combo.addItem(name, cfg)
        idx = self.preset_combo.findText(current_text)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)


# ════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
