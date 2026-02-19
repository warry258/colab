"""
头颅CTA DCM 减影处理程序 v1.0
智能序列识别 + 智能参数推荐
"""

import os
import re
import sys
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from collections import defaultdict
from scipy import ndimage

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QProgressBar,
    QTextEdit, QGroupBox, QMessageBox, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QSlider, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import pydicom
from pydicom.uid import generate_uid

print("头颅CTA减影 v1.0 - 智能序列识别 + 智能参数")


# ============================================================
# 序列分析
# ============================================================

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
        }
    except:
        return None


def scan_directory_for_series(directory, progress_callback=None, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
    
    path = Path(directory)
    all_files = list(path.rglob('*'))
    
    dicom_files = []
    for f in all_files:
        if f.is_file():
            try:
                with open(f, 'rb') as fp:
                    fp.seek(128)
                    if fp.read(4) == b'DICM':
                        dicom_files.append(f)
            except:
                pass
    
    log("找到 {} 个DICOM文件".format(len(dicom_files)))
    
    if not dicom_files:
        return {}
    
    series_dict = defaultdict(lambda: SeriesInfo())
    
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
            
            series.files.append((info['instance_number'], str(f)))
            series.file_count += 1
            
            if info['acquisition_time']:
                if series.acquisition_time is None or info['acquisition_time'] < series.acquisition_time:
                    series.acquisition_time = info['acquisition_time']
        
        if progress_callback and i % 50 == 0:
            progress_callback(int((i + 1) / len(dicom_files) * 50))
    
    for series in series_dict.values():
        series.files.sort(key=lambda x: x[0])
    
    return dict(series_dict)


def is_head_cta_series(series):
    """判断是否是头颅CTA序列"""
    desc = (series.series_description + ' ' + series.study_description + ' ' + 
            series.protocol_name + ' ' + series.body_part).upper()
    
    # 排除
    exclude = ['SCOUT', 'LOCALIZER', 'TOPOGRAM', '定位',
               'LUNG', 'CHEST', 'PULMONARY', '肺', '胸',
               'CARDIAC', 'HEART', 'CORONARY', '心', '冠脉',
               'ABDOMEN', 'LIVER', '腹', '肝']
    
    for kw in exclude:
        if kw in desc:
            return False
    
    if series.modality != 'CT':
        return False
    
    if series.file_count < 50:
        return False
    
    # 头颅关键词
    head_kw = ['HEAD', 'BRAIN', 'CEREBR', 'CRANIAL', '头', '颅', '脑', 'CAROTID']
    has_head = any(kw in desc for kw in head_kw)
    
    # CTA关键词
    cta_kw = ['CTA', 'ANGIO', 'C+', 'C-', '血管', '动脉']
    has_cta = any(kw in desc for kw in cta_kw)
    
    if has_head and has_cta:
        return True
    
    if has_head and series.file_count >= 100:
        return True
    
    return False


def is_contrast_enhanced(series):
    """
    判断是否是增强序列（造影后）
    基于描述信息中的C+/C-等标识
    """
    desc = series.series_description.upper()
    
    # 增强标识（C+优先级最高）
    enhanced_patterns = [
        r'\bC\+',           # C+
        r'\bC\s*\+',        # C +
        r'CE\b',            # CE
        r'CONTRAST',        # contrast
        r'ENHANCED',        # enhanced
        r'POST',            # post
        r'ARTERIAL',        # arterial
        r'增强',
        r'动脉期',
        r'A期',
    ]
    
    # 平扫标识
    plain_patterns = [
        r'\bC\-',           # C-
        r'\bC\s*\-',        # C -
        r'\bNC\b',          # NC (non-contrast)
        r'NON-CONTRAST',
        r'NONCONTRAST', 
        r'PLAIN',
        r'PRE\b',
        r'WITHOUT',
        r'平扫',
        r'非增强',
    ]
    
    # 检查增强标识
    for pattern in enhanced_patterns:
        if re.search(pattern, desc):
            return True
    
    # 检查平扫标识
    for pattern in plain_patterns:
        if re.search(pattern, desc):
            return False
    
    # 无法确定
    return None


def find_cta_pairs(series_dict):
    """找出CTA配对"""
    cta_series = [s for s in series_dict.values() if is_head_cta_series(s)]
    
    if len(cta_series) < 2:
        return [], cta_series
    
    # 按文件数量和层厚分组
    groups = defaultdict(list)
    for s in cta_series:
        key = (s.file_count, round(s.slice_thickness, 1))
        groups[key].append(s)
    
    pairs = []
    used = set()
    
    for key, group in groups.items():
        if len(group) < 2:
            continue
        
        # 分类
        enhanced = []
        plain = []
        unknown = []
        
        for s in group:
            status = is_contrast_enhanced(s)
            if status is True:
                enhanced.append(s)
            elif status is False:
                plain.append(s)
            else:
                unknown.append(s)
        
        # 配对：平扫 + 增强
        for pre in plain:
            for post in enhanced:
                if pre.series_uid not in used and post.series_uid not in used:
                    pairs.append((pre, post))
                    used.add(pre.series_uid)
                    used.add(post.series_uid)
                    break
            if pre.series_uid in used:
                break
        
        # 如果没找到明确配对，尝试用unknown
        if not pairs and len(unknown) >= 2:
            # 按时间排序
            unknown.sort(key=lambda s: (s.acquisition_time or datetime.max, s.series_number))
            if len(unknown) >= 2:
                pairs.append((unknown[0], unknown[1]))
    
    return pairs, cta_series


# ============================================================
# 序列扫描线程
# ============================================================

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
            self.log.emit("扫描: {}".format(self.directory))
            self.log.emit("=" * 55)
            
            series_dict = scan_directory_for_series(
                self.directory,
                progress_callback=self.progress.emit,
                log_callback=self.log.emit
            )
            
            self.log.emit("找到 {} 个序列".format(len(series_dict)))
            self.log.emit("")
            
            sorted_series = sorted(series_dict.values(), key=lambda s: s.series_number)
            for s in sorted_series:
                time_str = s.acquisition_time.strftime("%H:%M:%S") if s.acquisition_time else "--:--:--"
                is_cta = is_head_cta_series(s)
                contrast = is_contrast_enhanced(s)
                
                marker = "  "
                if is_cta:
                    if contrast is True:
                        marker = "C+"
                    elif contrast is False:
                        marker = "C-"
                    else:
                        marker = "★ "
                
                self.log.emit("{} #{:3d} | {:4d}张 | {} | {}".format(
                    marker, s.series_number, s.file_count, time_str,
                    s.series_description[:35] if s.series_description else "(无描述)"
                ))
            
            pairs, cta_series = find_cta_pairs(series_dict)
            
            self.log.emit("")
            if pairs:
                pre, post = pairs[0]
                self.log.emit("★ 自动配对:")
                self.log.emit("  平扫(C-): #{} {}".format(pre.series_number, pre.series_description[:30]))
                self.log.emit("  增强(C+): #{} {}".format(post.series_number, post.series_description[:30]))
            else:
                self.log.emit("未找到自动配对，请手动选择")
            
            self.progress.emit(100)
            self.finished_signal.emit(series_dict, pairs, cta_series)
            
        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit({}, [], [])


# ============================================================
# 智能参数分析线程
# ============================================================

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
            self.log.emit("")
            self.log.emit("=" * 55)
            self.log.emit("智能参数分析")
            self.log.emit("=" * 55)
            
            # 建立映射
            pre_dict = {inst: path for inst, path in self.pre_files}
            post_dict = {inst: path for inst, path in self.post_files}
            common = sorted(set(pre_dict.keys()) & set(post_dict.keys()))
            
            if not common:
                self.finished_signal.emit({'error': '序列不匹配'})
                return
            
            total = len(common)
            sample_n = min(8, total)
            indices = [common[i * (total - 1) // (sample_n - 1)] for i in range(sample_n)] if sample_n > 1 else [common[0]]
            
            self.log.emit("采样 {} 层分析...".format(sample_n))
            
            all_shifts = []
            all_chars = []
            
            for i, inst in enumerate(indices):
                try:
                    _, pre_d = load_dicom(pre_dict[inst])
                    _, post_d = load_dicom(post_dict[inst])
                    
                    dy, dx, angle, score = robust_registration(pre_d, post_d, 20, 3.0)
                    all_shifts.append((dy, dx, angle))
                    
                    # 分析特征
                    chars = self._analyze_image(pre_d, post_d)
                    all_chars.append(chars)
                    
                    self.log.emit("  #{}: dy={:.2f} dx={:.2f} rot={:.3f}°".format(inst, dy, dx, angle))
                except Exception as e:
                    self.log.emit("  #{}: 错误".format(inst))
                
                self.progress.emit(int((i + 1) / sample_n * 100))
            
            if not all_shifts:
                self.finished_signal.emit({'error': '分析失败'})
                return
            
            shifts = np.array(all_shifts)
            is_rigid = shifts[:, 0].std() < 0.8 and shifts[:, 1].std() < 0.8
            
            avg = {k: np.mean([c[k] for c in all_chars]) for k in all_chars[0]}
            
            # 计算推荐参数
            rec = self._compute_recommendations(shifts, avg, is_rigid, total)
            
            self.log.emit("")
            self.log.emit("分析结果:")
            self.log.emit("  位移一致性: {}".format("高(全局配准)" if is_rigid else "中等(逐层配准)"))
            self.log.emit("  薄骨占比: {:.2f}%".format(avg['thin_bone'] * 100))
            self.log.emit("  噪声水平: {:.1f} HU".format(avg['noise']))
            self.log.emit("  血管信号: {:.1f} HU".format(avg['vessel_signal']))
            self.log.emit("")
            self.log.emit("推荐参数:")
            self.log.emit("  骨骼抑制: {:.1f}".format(rec['bone_strength']))
            self.log.emit("  血管增强: {:.1f}x".format(rec['vessel_enhance']))
            self.log.emit("  清理边缘: {}".format("是" if rec['clean_bone_edges'] else "否"))
            self.log.emit("=" * 55)
            
            self.finished_signal.emit(rec)
            
        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit({'error': str(e)})
    
    def _analyze_image(self, pre, post):
        diff_pos = np.clip(post - pre, 0, None)
        
        bone = pre > 150
        thin = detect_thin_bone(pre)
        
        air = pre < -800
        noise = float(pre[air].std()) if air.sum() > 100 else 15.0
        
        strong = diff_pos > 50
        vessel = float(diff_pos[strong].mean()) if strong.sum() > 0 else 0
        
        return {
            'bone': float(bone.sum() / bone.size),
            'thin_bone': float(thin.sum() / thin.size),
            'noise': noise,
            'vessel_signal': vessel
        }
    
    def _compute_recommendations(self, shifts, chars, is_rigid, file_count):
        rec = {
            'global_mode': is_rigid,
            'max_shift': max(8, min(int(np.ceil(np.abs(shifts[:, :2]).max() * 1.5)) + 5, 25)),
            'max_angle': max(1.0, min(round(np.abs(shifts[:, 2]).max() * 2 + 0.5, 1), 4.0)),
        }
        
        # 骨骼抑制
        base = 1.0
        if chars['thin_bone'] > 0.005:
            base += 0.3
        if chars['bone'] > 0.03:
            base += 0.2
        rec['bone_strength'] = round(min(base, 2.0), 1)
        
        # 血管敏感度
        if chars['vessel_signal'] > 80:
            rec['vessel_sensitivity'] = 0.8
        elif chars['vessel_signal'] > 40:
            rec['vessel_sensitivity'] = 1.0
        else:
            rec['vessel_sensitivity'] = 1.2
        
        # 血管增强
        if chars['vessel_signal'] > 80:
            rec['vessel_enhance'] = 1.5
        elif chars['vessel_signal'] > 50:
            rec['vessel_enhance'] = 2.0
        else:
            rec['vessel_enhance'] = 2.5
        
        # 清理
        rec['clean_bone_edges'] = chars['thin_bone'] > 0.003
        rec['min_vessel_size'] = 8 if chars['noise'] > 20 else 5 if chars['noise'] > 12 else 3
        rec['smooth_sigma'] = 0.9 if chars['noise'] > 15 else 0.6
        
        rec['wc'] = 200
        rec['ww'] = 400
        
        return rec


# ============================================================
# 配准和减影算法
# ============================================================

def compute_ncc(fixed, moving):
    f = fixed.ravel().astype(np.float64)
    m = moving.ravel().astype(np.float64)
    f = f - f.mean()
    m = m - m.mean()
    f_std, m_std = f.std(), m.std()
    if f_std < 1e-6 or m_std < 1e-6:
        return 0.0
    return np.dot(f, m) / (len(f) * f_std * m_std)


def shift_image(image, dy, dx):
    if abs(dy) < 0.001 and abs(dx) < 0.001:
        return image.copy()
    return ndimage.shift(image.astype(np.float64), [dy, dx], order=1, mode='constant', cval=0)


def rotate_image(image, angle_deg):
    if abs(angle_deg) < 0.001:
        return image.copy()
    return ndimage.rotate(image.astype(np.float64), angle_deg, reshape=False, order=1, mode='constant', cval=0)


def apply_transform(image, dy, dx, angle):
    result = image.copy()
    if abs(angle) > 0.001:
        result = rotate_image(result, angle)
    if abs(dy) > 0.001 or abs(dx) > 0.001:
        result = shift_image(result, dy, dx)
    return result


def fft_phase_correlation(fixed, moving, max_shift=15):
    from numpy.fft import fft2, ifft2, fftshift
    
    h, w = fixed.shape
    margin = h // 4
    
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
    dy = peak_idx[0] - correlation.shape[0] // 2
    dx = peak_idx[1] - correlation.shape[1] // 2
    
    py, px = peak_idx
    if 1 <= py < correlation.shape[0] - 1:
        y_vals = correlation[py-1:py+2, px]
        if y_vals[1] > y_vals[0] and y_vals[1] > y_vals[2]:
            denom = 2 * (y_vals[0] + y_vals[2] - 2 * y_vals[1])
            if abs(denom) > 1e-6:
                dy += (y_vals[0] - y_vals[2]) / denom
    
    if 1 <= px < correlation.shape[1] - 1:
        x_vals = correlation[py, px-1:px+2]
        if x_vals[1] > x_vals[0] and x_vals[1] > x_vals[2]:
            denom = 2 * (x_vals[0] + x_vals[2] - 2 * x_vals[1])
            if abs(denom) > 1e-6:
                dx += (x_vals[0] - x_vals[2]) / denom
    
    return float(np.clip(dy, -max_shift, max_shift)), float(np.clip(dx, -max_shift, max_shift))


def robust_registration(fixed, moving, max_shift=15, max_angle=3.0):
    h, w = fixed.shape
    margin = max(h // 6, 30)
    
    def get_roi(img):
        return img[margin:-margin, margin:-margin]
    
    def evaluate(dy, dx, angle):
        transformed = apply_transform(moving, dy, dx, angle)
        return compute_ncc(get_roi(fixed), get_roi(transformed))
    
    dy, dx = fft_phase_correlation(fixed, moving, max_shift)
    
    best_angle = 0.0
    best_score = evaluate(dy, dx, 0.0)
    
    for angle in np.arange(-max_angle, max_angle + 0.5, 0.5):
        score = evaluate(dy, dx, angle)
        if score > best_score:
            best_score = score
            best_angle = angle
    
    for angle in np.arange(best_angle - 0.5, best_angle + 0.55, 0.1):
        score = evaluate(dy, dx, angle)
        if score > best_score:
            best_score = score
            best_angle = angle
    
    for angle in np.arange(best_angle - 0.1, best_angle + 0.12, 0.02):
        score = evaluate(dy, dx, angle)
        if score > best_score:
            best_score = score
            best_angle = angle
    
    if abs(best_angle) > 0.05:
        rotated = rotate_image(moving, best_angle)
        dy2, dx2 = fft_phase_correlation(fixed, rotated, 5)
        if evaluate(dy + dy2, dx + dx2, best_angle) > best_score:
            dy += dy2
            dx += dx2
    
    for ddy in [-0.3, -0.15, 0, 0.15, 0.3]:
        for ddx in [-0.3, -0.15, 0, 0.15, 0.3]:
            score = evaluate(dy + ddy, dx + ddx, best_angle)
            if score > best_score:
                best_score = score
                dy += ddy
                dx += ddx
    
    return dy, dx, best_angle, best_score


def create_body_mask(image):
    body = image > -400
    body = ndimage.binary_closing(body, iterations=3)
    body = ndimage.binary_fill_holes(body)
    body = ndimage.binary_opening(body, iterations=2)
    
    labeled, num = ndimage.label(body)
    if num > 0:
        sizes = ndimage.sum(body, labeled, range(1, num + 1))
        body = labeled == (np.argmax(sizes) + 1)
    
    return body


def detect_equipment(pre, post):
    high_both = (pre > 150) & (post > 150)
    stable = np.abs(post - pre) < 30
    
    body = create_body_mask(pre)
    body_dilated = ndimage.binary_dilation(body, iterations=10)
    
    equipment = high_both & stable & ~body_dilated
    equipment = ndimage.binary_dilation(equipment, iterations=5)
    
    return equipment


def detect_thin_bone(pre_image):
    bone = pre_image > 150
    eroded = ndimage.binary_erosion(bone, iterations=2)
    thin_bone = bone & ~eroded
    
    edges = np.abs(ndimage.sobel(pre_image.astype(np.float64)))
    high_edge = edges > np.percentile(edges[bone] if bone.sum() > 0 else edges, 70)
    
    return thin_bone | (bone & high_edge)


def advanced_subtraction(pre, post_aligned, bone_strength=1.0, vessel_sensitivity=1.0):
    diff = post_aligned - pre
    diff_pos = np.clip(diff, 0, None)
    
    gain = np.ones_like(diff_pos, dtype=np.float64)
    
    body = create_body_mask(pre)
    gain[pre < -500] = 0
    gain[~body] = 0
    
    gain[pre > 500] = 0
    
    high_bone = (pre > 300) & (pre <= 500)
    th_high = 120 * bone_strength
    gain[high_bone & (diff_pos < th_high)] = 0
    gain[high_bone & (diff_pos >= th_high)] = 0.1 / bone_strength
    
    med_bone = (pre > 180) & (pre <= 300)
    th_med1 = 60 * bone_strength
    th_med2 = 120 * bone_strength
    gain[med_bone & (diff_pos < th_med1)] = 0
    gain[med_bone & (diff_pos >= th_med1) & (diff_pos < th_med2)] = 0.15 / bone_strength
    gain[med_bone & (diff_pos >= th_med2)] = 0.3 / bone_strength
    
    thin_bone = detect_thin_bone(pre)
    th_thin = 80 * bone_strength
    gain[thin_bone & (diff_pos < th_thin)] = 0
    gain[thin_bone & (diff_pos >= th_thin)] = 0.1 / bone_strength
    
    low_bone = (pre > 100) & (pre <= 180)
    th_low = 40 * bone_strength
    gain[low_bone & (diff_pos < th_low)] = 0.05
    gain[low_bone & (diff_pos >= th_low) & (diff_pos < th_low * 2)] = 0.3
    
    soft = (pre > -100) & (pre <= 100)
    weak_th = 20 / vessel_sensitivity
    gain[soft & (diff_pos < weak_th)] = 0.1
    
    vessel = (pre > -100) & (pre < 60) & (diff_pos > 50 * vessel_sensitivity)
    gain[vessel] = 1.0
    
    return diff_pos * np.clip(gain, 0, 1.5)


def clean_bone_edges(image, pre_image, edge_width=2):
    bone = pre_image > 150
    
    bone_dilated = ndimage.binary_dilation(bone, iterations=edge_width)
    bone_eroded = ndimage.binary_erosion(bone, iterations=edge_width)
    edge_region = bone_dilated & ~bone_eroded
    
    result = image.copy()
    result[edge_region & (image < 40)] = 0
    medium = edge_region & (image >= 40) & (image < 80)
    result[medium] *= 0.3
    
    return result


def morphological_cleanup(image, min_size=5):
    mask = image > 10
    mask = ndimage.binary_opening(mask, iterations=1)
    
    labeled, num = ndimage.label(mask)
    if num > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        small = np.isin(labeled, np.where(np.array(sizes) < min_size)[0] + 1)
        mask[small] = False
    
    result = image.copy()
    result[~mask & (image < 30)] = 0
    return result


def edge_preserving_smooth(image, pre_image, sigma=0.7):
    edges = np.abs(ndimage.sobel(pre_image.astype(np.float64)))
    edge_norm = edges / (edges.max() + 1e-6)
    
    smooth_heavy = ndimage.gaussian_filter(image, sigma * 1.5)
    smooth_light = ndimage.gaussian_filter(image, sigma * 0.3)
    
    return smooth_heavy * (1 - edge_norm) + smooth_light * edge_norm


# ============================================================
# DICOM处理
# ============================================================

def load_dicom(filepath):
    ds = pydicom.dcmread(filepath)
    try:
        pixels = ds.pixel_array
    except:
        ds.decompress()
        pixels = ds.pixel_array
    
    pixels = pixels.astype(np.float64)
    if hasattr(ds, 'RescaleSlope'):
        pixels = pixels * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return ds, pixels


def save_dicom(template_ds, pixel_data, output_path, wc=200, ww=400):
    new_ds = template_ds.copy()
    
    data_min, data_max = float(pixel_data.min()), float(pixel_data.max())
    
    if data_max > data_min:
        normalized = (pixel_data - data_min) / (data_max - data_min)
        pixel_int = (normalized * 4095).astype(np.int16)
    else:
        pixel_int = np.zeros_like(pixel_data, dtype=np.int16)
    
    new_ds.PixelData = pixel_int.tobytes()
    new_ds.BitsAllocated = 16
    new_ds.BitsStored = 16
    new_ds.HighBit = 15
    new_ds.PixelRepresentation = 1
    new_ds.SamplesPerPixel = 1
    new_ds.PhotometricInterpretation = 'MONOCHROME2'
    
    for tag in ['LossyImageCompression', 'LossyImageCompressionRatio', 'LossyImageCompressionMethod']:
        if hasattr(new_ds, tag):
            delattr(new_ds, tag)
    
    new_ds.RescaleSlope = (data_max - data_min) / 4095 if data_max > data_min else 1
    new_ds.RescaleIntercept = data_min if data_max > data_min else 0
    new_ds.WindowCenter = wc
    new_ds.WindowWidth = ww
    new_ds.SeriesDescription = "CTA Subtraction"
    new_ds.SeriesInstanceUID = generate_uid()
    new_ds.SOPInstanceUID = generate_uid()
    new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
    new_ds.save_as(output_path, write_like_original=False)


# ============================================================
# 处理线程
# ============================================================

class ProcessThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, pre_files, post_files, output_dir, options):
        super().__init__()
        self.pre_files = pre_files
        self.post_files = post_files
        self.output_dir = output_dir
        self.options = options
        self.cancelled = False
    
    def cancel(self):
        self.cancelled = True
    
    def run(self):
        try:
            t0 = time.time()
            opt = self.options
            
            self.log.emit("=" * 50)
            self.log.emit("CTA减影处理")
            self.log.emit("=" * 50)
            
            pre_dict = {inst: path for inst, path in self.pre_files}
            post_dict = {inst: path for inst, path in self.post_files}
            common = sorted(set(pre_dict.keys()) & set(post_dict.keys()))
            
            if not common:
                self.finished_signal.emit(False, "序列不匹配")
                return
            
            total = len(common)
            self.log.emit("匹配: {} 对".format(total))
            
            global_params = None
            if opt.get('global_mode', True):
                mid = common[total // 2]
                self.log.emit("计算全局参数...")
                _, pre_d = load_dicom(pre_dict[mid])
                _, post_d = load_dicom(post_dict[mid])
                dy, dx, ang, _ = robust_registration(pre_d, post_d, opt['max_shift'], opt['max_angle'])
                global_params = (dy, dx, ang)
                self.log.emit("  dy={:.2f} dx={:.2f} rot={:.3f}°".format(dy, dx, ang))
            
            os.makedirs(self.output_dir, exist_ok=True)
            series_uid = generate_uid()
            
            done = 0
            times = []
            
            self.log.emit("处理中...")
            
            for i, inst in enumerate(common):
                if self.cancelled:
                    break
                
                try:
                    t1 = time.time()
                    
                    pre_ds, pre_d = load_dicom(pre_dict[inst])
                    post_ds, post_d = load_dicom(post_dict[inst])
                    
                    if pre_d.shape != post_d.shape:
                        continue
                    
                    if global_params:
                        dy, dx, ang = global_params
                    else:
                        dy, dx, ang, _ = robust_registration(pre_d, post_d, opt['max_shift'], opt['max_angle'])
                    
                    aligned = apply_transform(post_d, dy, dx, ang)
                    
                    equip = detect_equipment(pre_d, aligned)
                    
                    result = advanced_subtraction(pre_d, aligned, opt['bone_strength'], opt['vessel_sensitivity'])
                    result[equip] = 0
                    
                    if opt.get('clean_bone_edges', True):
                        result = clean_bone_edges(result, pre_d)
                    
                    result = morphological_cleanup(result, opt['min_vessel_size'])
                    
                    if opt.get('smooth_sigma', 0) > 0:
                        result = edge_preserving_smooth(result, pre_d, opt['smooth_sigma'])
                    
                    result = result * opt['vessel_enhance']
                    
                    out_path = os.path.join(self.output_dir, "SUB_{:04d}.dcm".format(inst))
                    save_dicom(post_ds, result, out_path, opt['wc'], opt['ww'])
                    
                    ds = pydicom.dcmread(out_path)
                    ds.SeriesInstanceUID = series_uid
                    ds.InstanceNumber = i + 1
                    ds.save_as(out_path)
                    
                    done += 1
                    times.append(time.time() - t1)
                    
                    if done % 25 == 0:
                        avg = np.mean(times[-25:])
                        remain = avg * (total - i - 1)
                        self.log.emit("  {}/{} ({:.2f}s/张, 剩余{:.0f}s)".format(done, total, avg, remain))
                
                except:
                    pass
                
                self.progress.emit(int((i + 1) / total * 100))
            
            elapsed = time.time() - t0
            self.log.emit("")
            self.log.emit("完成: {} 张, {:.1f}秒".format(done, elapsed))
            
            self.finished_signal.emit(True, "完成!\n\n处理: {} 张\n耗时: {:.1f}秒\n\n输出:\n{}".format(
                done, elapsed, self.output_dir))
            
        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))


# ============================================================
# GUI
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.all_series = {}
        self.cta_pairs = []
        self.selected_pre = None
        self.selected_post = None
        self.recommendations = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("头颅CTA减影 v1.0-WHW版")
        self.setMinimumSize(900, 950)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(6)
        
        # 标题
        title = QLabel("头颅CTA 数字减影-whw版 v1.0")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 数据目录
        dir_group = QGroupBox("1. 选择数据目录")
        dir_layout = QHBoxLayout(dir_group)
        dir_layout.addWidget(QLabel("DICOM目录:"))
        self.data_dir_edit = QLineEdit()
        dir_layout.addWidget(self.data_dir_edit)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(lambda: self.data_dir_edit.setText(
            QFileDialog.getExistingDirectory(self, "选择目录") or self.data_dir_edit.text()))
        dir_layout.addWidget(browse_btn)
        self.scan_btn = QPushButton("扫描")
        self.scan_btn.setStyleSheet("background:#9C27B0;color:white;font-weight:bold;")
        self.scan_btn.clicked.connect(self.scan_directory)
        dir_layout.addWidget(self.scan_btn)
        layout.addWidget(dir_group)
        
        # 序列选择
        series_group = QGroupBox("2. 选择序列配对")
        series_layout = QVBoxLayout(series_group)
        
        # 下拉框选择
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("平扫(C-):"))
        self.pre_combo = QComboBox()
        self.pre_combo.setMinimumWidth(300)
        select_layout.addWidget(self.pre_combo)
        select_layout.addSpacing(20)
        select_layout.addWidget(QLabel("增强(C+):"))
        self.post_combo = QComboBox()
        self.post_combo.setMinimumWidth(300)
        select_layout.addWidget(self.post_combo)
        select_layout.addStretch()
        series_layout.addLayout(select_layout)
        
        # 分析按钮
        analyze_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("🔍 分析序列 && 推荐参数")
        self.analyze_btn.setStyleSheet("background:#2196F3;color:white;font-weight:bold;padding:8px;")
        self.analyze_btn.clicked.connect(self.analyze_params)
        self.analyze_btn.setEnabled(False)
        analyze_layout.addWidget(self.analyze_btn)
        analyze_layout.addStretch()
        series_layout.addLayout(analyze_layout)
        
        layout.addWidget(series_group)
        
        # 输出目录
        out_group = QGroupBox("3. 输出目录")
        out_layout = QHBoxLayout(out_group)
        self.out_edit = QLineEdit()
        out_layout.addWidget(self.out_edit)
        out_btn = QPushButton("...")
        out_btn.setFixedWidth(30)
        out_btn.clicked.connect(lambda: self.out_edit.setText(
            QFileDialog.getExistingDirectory(self, "选择输出目录") or self.out_edit.text()))
        out_layout.addWidget(out_btn)
        layout.addWidget(out_group)
        
        # 参数
        param_group = QGroupBox("4. 参数设置")
        param_layout = QVBoxLayout(param_group)
        
        # 第一行
        row1 = QHBoxLayout()
        self.global_check = QCheckBox("全局配准")
        self.global_check.setChecked(True)
        row1.addWidget(self.global_check)
        row1.addSpacing(20)
        row1.addWidget(QLabel("最大位移:"))
        self.max_shift = QSpinBox()
        self.max_shift.setRange(5, 30)
        self.max_shift.setValue(15)
        row1.addWidget(self.max_shift)
        row1.addWidget(QLabel("px"))
        row1.addSpacing(10)
        row1.addWidget(QLabel("最大旋转:"))
        self.max_angle = QDoubleSpinBox()
        self.max_angle.setRange(0.5, 5.0)
        self.max_angle.setValue(3.0)
        row1.addWidget(self.max_angle)
        row1.addWidget(QLabel("°"))
        row1.addStretch()
        param_layout.addLayout(row1)
        
        # 第二行
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("骨骼抑制:"))
        self.bone_slider = QSlider(Qt.Horizontal)
        self.bone_slider.setRange(5, 25)
        self.bone_slider.setValue(12)
        self.bone_slider.setMaximumWidth(150)
        self.bone_slider.valueChanged.connect(lambda v: self.bone_label.setText("{:.1f}".format(v/10)))
        row2.addWidget(self.bone_slider)
        self.bone_label = QLabel("1.2")
        self.bone_label.setFixedWidth(30)
        row2.addWidget(self.bone_label)
        row2.addSpacing(20)
        row2.addWidget(QLabel("血管增强:"))
        self.enhance = QDoubleSpinBox()
        self.enhance.setRange(0.5, 5.0)
        self.enhance.setValue(2.0)
        row2.addWidget(self.enhance)
        row2.addSpacing(20)
        self.clean_check = QCheckBox("清理骨骼边缘")
        self.clean_check.setChecked(True)
        row2.addWidget(self.clean_check)
        row2.addStretch()
        param_layout.addLayout(row2)
        
        # 第三行
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("最小血管:"))
        self.min_size = QSpinBox()
        self.min_size.setRange(1, 15)
        self.min_size.setValue(5)
        row3.addWidget(self.min_size)
        row3.addWidget(QLabel("px"))
        row3.addSpacing(20)
        row3.addWidget(QLabel("平滑:"))
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 1.5)
        self.smooth.setValue(0.7)
        row3.addWidget(self.smooth)
        row3.addSpacing(20)
        row3.addWidget(QLabel("窗位:"))
        self.wc = QSpinBox()
        self.wc.setRange(0, 2000)
        self.wc.setValue(200)
        row3.addWidget(self.wc)
        row3.addWidget(QLabel("窗宽:"))
        self.ww = QSpinBox()
        self.ww.setRange(1, 2000)
        self.ww.setValue(400)
        row3.addWidget(self.ww)
        row3.addStretch()
        param_layout.addLayout(row3)
        
        layout.addWidget(param_group)
        
        # 开始按钮
        self.start_btn = QPushButton("▶ 开始处理")
        self.start_btn.setStyleSheet("background:#E91E63;color:white;font-weight:bold;padding:12px;font-size:14px;")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn)
        
        # 进度
        prog_layout = QHBoxLayout()
        self.progress = QProgressBar()
        prog_layout.addWidget(self.progress)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setFixedWidth(50)
        self.cancel_btn.clicked.connect(self.cancel)
        prog_layout.addWidget(self.cancel_btn)
        layout.addLayout(prog_layout)
        
        # 日志
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setStyleSheet("font-family:Consolas;font-size:9pt;")
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)
        
        self.log("头颅CTA减影-whw版 v1.0")
        self.log("使用说明:")
        self.log("  1. 选择包含DICOM的目录，点击「扫描」")
        self.log("  2. 确认序列配对（C-=平扫，C+=增强）")
        self.log("  3. 点击「分析序列」获取推荐参数")
        self.log("  4. 点击「开始处理」")
        self.log("  5. 处理完毕，在医三维中打开输出目录，进行三维重建")
        self.log("  6. 扫描做得烂，程序无法挽救，如出现静脉窦，动脉自然会差")
        self.log("  7. Enjoy it！")

    
    def log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        QApplication.processEvents()
    
    def scan_directory(self):
        data_dir = self.data_dir_edit.text()
        if not data_dir:
            QMessageBox.warning(self, "提示", "请先选择目录")
            return
        
        self.log_text.clear()
        self.progress.setValue(0)
        self.pre_combo.clear()
        self.post_combo.clear()
        self.scan_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        
        self.scan_thread = SeriesScanThread(data_dir)
        self.scan_thread.progress.connect(self.progress.setValue)
        self.scan_thread.log.connect(self.log)
        self.scan_thread.finished_signal.connect(self.on_scan_finished)
        self.scan_thread.start()
    
    def on_scan_finished(self, all_series, pairs, cta_series):
        self.scan_btn.setEnabled(True)
        self.all_series = all_series
        self.cta_pairs = pairs
        
        # 填充下拉框
        sorted_series = sorted(all_series.values(), key=lambda s: s.series_number)
        
        for s in sorted_series:
            contrast = is_contrast_enhanced(s)
            marker = ""
            if contrast is True:
                marker = "[C+] "
            elif contrast is False:
                marker = "[C-] "
            
            text = "#{:03d} | {}张 | {}{}".format(
                s.series_number, s.file_count, marker, 
                s.series_description[:40] if s.series_description else "(无描述)"
            )
            
            self.pre_combo.addItem(text, s)
            self.post_combo.addItem(text, s)
        
        # 如果有自动配对，预选
        if pairs:
            pre, post = pairs[0]
            for i in range(self.pre_combo.count()):
                if self.pre_combo.itemData(i) == pre:
                    self.pre_combo.setCurrentIndex(i)
                    break
            for i in range(self.post_combo.count()):
                if self.post_combo.itemData(i) == post:
                    self.post_combo.setCurrentIndex(i)
                    break
        
        self.analyze_btn.setEnabled(True)
        
        # 默认输出目录
        if not self.out_edit.text():
            self.out_edit.setText(os.path.join(self.data_dir_edit.text(), "CTA_Subtraction"))
    
    def analyze_params(self):
        pre_series = self.pre_combo.currentData()
        post_series = self.post_combo.currentData()
        
        if not pre_series or not post_series:
            QMessageBox.warning(self, "提示", "请选择序列")
            return
        
        if pre_series == post_series:
            QMessageBox.warning(self, "提示", "请选择不同的序列")
            return
        
        self.selected_pre = pre_series
        self.selected_post = post_series
        
        self.progress.setValue(0)
        self.scan_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        
        self.param_thread = ParamAnalyzeThread(pre_series.files, post_series.files)
        self.param_thread.progress.connect(self.progress.setValue)
        self.param_thread.log.connect(self.log)
        self.param_thread.finished_signal.connect(self.on_analyze_finished)
        self.param_thread.start()
    
    def on_analyze_finished(self, result):
        self.scan_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        
        if 'error' in result:
            QMessageBox.warning(self, "错误", result['error'])
            return
        
        self.recommendations = result
        
        # 应用推荐参数
        self.global_check.setChecked(result.get('global_mode', True))
        self.max_shift.setValue(result.get('max_shift', 15))
        self.max_angle.setValue(result.get('max_angle', 3.0))
        self.bone_slider.setValue(int(result.get('bone_strength', 1.2) * 10))
        self.enhance.setValue(result.get('vessel_enhance', 2.0))
        self.clean_check.setChecked(result.get('clean_bone_edges', True))
        self.min_size.setValue(result.get('min_vessel_size', 5))
        self.smooth.setValue(result.get('smooth_sigma', 0.7))
        self.wc.setValue(result.get('wc', 200))
        self.ww.setValue(result.get('ww', 400))
        
        self.start_btn.setEnabled(True)
        self.log("")
        self.log("✓ 参数已自动设置，点击「开始处理」运行")
    
    def start_processing(self):
        if not self.selected_pre or not self.selected_post:
            QMessageBox.warning(self, "提示", "请先分析序列")
            return
        
        out_dir = self.out_edit.text()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请设置输出目录")
            return
        
        options = {
            'global_mode': self.global_check.isChecked(),
            'max_shift': self.max_shift.value(),
            'max_angle': self.max_angle.value(),
            'bone_strength': self.bone_slider.value() / 10.0,
            'vessel_sensitivity': self.recommendations.get('vessel_sensitivity', 1.0) if self.recommendations else 1.0,
            'vessel_enhance': self.enhance.value(),
            'clean_bone_edges': self.clean_check.isChecked(),
            'min_vessel_size': self.min_size.value(),
            'smooth_sigma': self.smooth.value(),
            'wc': self.wc.value(),
            'ww': self.ww.value()
        }
        
        self.progress.setValue(0)
        self.scan_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        self.log("")
        self.log("平扫: #{} {}".format(self.selected_pre.series_number, self.selected_pre.series_description[:30]))
        self.log("增强: #{} {}".format(self.selected_post.series_number, self.selected_post.series_description[:30]))
        
        self.proc_thread = ProcessThread(
            self.selected_pre.files, self.selected_post.files, out_dir, options)
        self.proc_thread.progress.connect(self.progress.setValue)
        self.proc_thread.log.connect(self.log)
        self.proc_thread.finished_signal.connect(self.on_process_finished)
        self.proc_thread.start()
    
    def cancel(self):
        if hasattr(self, 'proc_thread') and self.proc_thread:
            self.proc_thread.cancel()
    
    def on_process_finished(self, success, msg):
        self.scan_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(self, "错误", msg)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()