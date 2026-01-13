# 手动实现车头的检测    将车头区域拿出来 进行检测#
"""
套牌车/换挂识别 GUI（持久化存储版本）

新增功能：
1. 输出文件管理功能
2. 修改CSV结构支持新旧格式
3. 优化界面显示
4. 完善删除和保存操作
5. 重构批量检测
"""

import sys
import os
import re
import shutil
from typing import Tuple, Optional, List
from datetime import datetime
from dateutil import parser

import cv2
import numpy as np
import cx_Oracle
import pandas as pd
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox, QTextEdit, QDialog, QScrollArea,
    QProgressDialog, QComboBox, QDoubleSpinBox, QGridLayout,
    QDateEdit, QDialogButtonBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QDate
from PySide6.QtGui import QPixmap, QImage, QAction
from paddleocr import PaddleOCR
from ultralytics import YOLO

from siamese import Siamese

parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_chuli.cropper import VehicleCropper


class PlateRecognizer:
    """车牌识别器：在车辆图像的车头区域上做 OCR，返回车牌号。"""

    def __init__(self, seg_model_path: str = r"D:\project\yolo11n-seg.pt"):
        self.seg_model = YOLO(seg_model_path)
        # head/tail 模型：用于在整车区域中裁出车头
        self.headtail_model = YOLO(r"D:\data2\runs\detect\train\weights\best.pt")
        self.ocr = PaddleOCR()
        self.province_prefix = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼港澳")
        self.special_suffix = "挂警学领港澳"

    def extract_vehicle_mask_crop(self, image_path: str) -> np.ndarray:
        """利用分割模型提取车辆区域，返回 BGR 裁剪图。"""
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"无法读取图像: {image_path}")

        results = self.seg_model(image_path, verbose=False)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            raise RuntimeError("分割模型未检测到车辆")
        if result.masks is None or len(result.masks) == 0:
            raise RuntimeError("分割模型未返回掩膜")

        masks = result.masks.data.cpu().numpy()
        areas = masks.sum(axis=(1, 2))
        largest_idx = int(np.argmax(areas))
        mask = masks[largest_idx]

        h, w = image.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError("掩膜为空")
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        masked = cv2.bitwise_and(image, image, mask=mask)
        crop = masked[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            raise RuntimeError("掩膜裁剪结果为空")
        return crop

    def is_valid_plate(self, text: str) -> bool:
        text = str(text).strip().upper()
        text = re.sub(r"[·•∙.]", "", text)
        pattern = rf"^[\u4E00-\u9FA5][A-Z][A-Z0-9]{{4,5}}[A-Z0-9{self.special_suffix}]$"
        return bool(re.match(pattern, text)) and text[0] in self.province_prefix

    def _crop_head_from_vehicle_bgr(self, vehicle_bgr: np.ndarray) -> np.ndarray:
        """在车辆 BGR 图上用 head/tail 模型裁出车头区域；失败则返回原图。"""
        if vehicle_bgr is None or vehicle_bgr.size == 0:
            return vehicle_bgr

        results = self.headtail_model(vehicle_bgr, conf=0.25, verbose=False)
        if not results:
            return vehicle_bgr
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return vehicle_bgr

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        best_idx = None
        best_score = -1.0
        for i, (cls_id, score) in enumerate(zip(classes, scores)):
            if int(cls_id) != 0:
                continue
            if float(score) > best_score:
                best_score = float(score)
                best_idx = i
        if best_idx is None:
            return vehicle_bgr

        x1, y1, x2, y2 = boxes[int(best_idx)]
        h, w = vehicle_bgr.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return vehicle_bgr

        head_bgr = vehicle_bgr[y1:y2, x1:x2].copy()
        if head_bgr.size == 0:
            return vehicle_bgr
        return head_bgr

    def recognize_plate(self, image_path: str) -> Tuple[bool, Optional[str], str]:
        """在车头区域上做 OCR，返回 (是否成功, 车牌号, 错误信息)。"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, None, f"无法读取图片: {image_path}"

            try:
                vehicle_crop = self.extract_vehicle_mask_crop(image_path)
            except Exception as e:
                print(f"车辆分割失败: {e}")
                vehicle_crop = image

            try:
                head_crop = self._crop_head_from_vehicle_bgr(vehicle_crop)
            except Exception as e:
                print(f"车头裁剪失败: {e}")
                head_crop = vehicle_crop

            ocr_input = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
            result = self.ocr.predict(input=ocr_input)
            if result is not None and len(result) > 0:
                texts = [line["rec_texts"] for line in result][0]
                for text in texts:
                    raw = str(text).strip().upper()
                    t = re.sub(r"[·•∙.]", "", raw)
                    if re.match(rf"^[\u4E00-\u9FA5][A-Z][A-Z0-9]{{4,5}}[A-Z0-9{self.special_suffix}]$", t):
                        if t[0] in self.province_prefix:
                            return True, t, ""
            return False, None, "未找到符合格式的车牌号"
        except Exception as e:
            import traceback
            print(f"识别过程中出错: {e}\n{traceback.format_exc()}")
            return False, None, f"识别过程中出错: {e}"


class BatchDetectWorker(QThread):
    """批量检测工作线程"""
    progress_updated = Signal(int, int)  # current, total
    result_ready = Signal(list)  # suspicious_pairs
    error_occurred = Signal(str)

    def __init__(self, df, gui_instance):
        super().__init__()
        self.df = df
        self.gui = gui_instance
        self.should_stop = False

    def run(self):
        try:
            last_record_by_plate = {}
            max_task_id_seen = None
            suspicious_pairs = []

            total_rows = len(self.df)

            for idx, row in self.df.iterrows():
                if self.should_stop:
                    break

                self.progress_updated.emit(idx + 1, total_rows)

                plate = row['TRUCK_ID']
                if plate is None:
                    continue
                plate_str = str(plate).strip()
                if not plate_str:
                    continue

                # 记录本次遍历中遇到的最大 TASK_ID
                try:
                    curr_tid_int = int(row['TASK_ID'])
                    if max_task_id_seen is None or curr_tid_int > max_task_id_seen:
                        max_task_id_seen = curr_tid_int
                except Exception:
                    pass

                prev_row = last_record_by_plate.get(plate_str)
                if prev_row is not None:
                    # 对皮重、毛重分别比较
                    for key, prev_key, tare_or_gross in [
                        ('TARE_IMAGE_PATH1', 'TARE_IMAGE_PATH1', 'tare'),
                        ('GROSS_IMAGE_PATH1', 'GROSS_IMAGE_PATH1', 'gross'),
                    ]:
                        curr_path = row.get(key)
                        prev_path = prev_row.get(prev_key)
                        if not curr_path or not prev_path:
                            continue

                        head_prob = self.gui.compare_head(curr_path, prev_path)
                        tail_prob = self.gui.compare_tail(curr_path, prev_path)

                        # 识别车牌
                        success1, plate1, _ = self.gui.plate_recognizer.recognize_plate(curr_path)
                        success2, plate2, _ = self.gui.plate_recognizer.recognize_plate(prev_path)
                        plate_same = success1 and success2 and (plate1 == plate2)

                        case_type = None

                        # 只有 head_prob 或 tail_prob 为 None（检测/对比异常）时才记为 abnormal
                        if head_prob is None or tail_prob is None:
                            case_type = 'abnormal'
                        elif plate_same:
                            # 疑似套牌：车牌相同 + 头部相似度低
                            if head_prob < self.gui.HEAD_LOW_TH:
                                case_type = 'fake_plate'
                            # 疑似换挂：车牌相同 + 头部>0.3(没换头) + 尾部<=0.3(换尾)
                            elif head_prob > self.gui.HEAD_SAME_TH and tail_prob <= self.gui.TAIL_LOW_TH:
                                case_type = 'change_trailer'

                        if case_type is not None:
                            if case_type == 'abnormal':
                                continue

                            output_folder = self.gui._create_result_folder(str(row['TASK_ID']), case_type)
                            image_paths = self.gui._save_result_images(curr_path, prev_path, output_folder)
                            if not image_paths:
                                self.gui._delete_result_folder(output_folder)
                                continue

                            suspicious_pairs.append({
                                'tare_or_gross': tare_or_gross,
                                'case_type': case_type,
                                'task_id': row['TASK_ID'],
                                'prev_task_id': prev_row.get('TASK_ID'),
                                'truck_id': plate_str,
                                'curr_path': curr_path,
                                'prev_path': prev_path,
                                'head_prob': head_prob,
                                'tail_prob': tail_prob,
                                'plate_curr': plate1 if success1 else None,
                                'plate_prev': plate2 if success2 else None,
                                'output_folder': output_folder,
                                'image_paths': image_paths,
                                'format_type': 'new'
                            })

                last_record_by_plate[plate_str] = row

            self.result_ready.emit(suspicious_pairs)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.should_stop = True


class LocalCompareWorker(QThread):
    head_ready = Signal(int, object)
    tail_ready = Signal(int, object)
    probs_ready = Signal(object, object)
    error_occurred = Signal(str)

    def __init__(self, gui_instance, path1: str, path2: str):
        super().__init__()
        self.gui = gui_instance
        self.path1 = path1
        self.path2 = path2

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            head1 = self.gui._get_processed_head_image(self.path1)
            self.head_ready.emit(1, head1)
            if self.isInterruptionRequested():
                return
            head2 = self.gui._get_processed_head_image(self.path2)
            self.head_ready.emit(2, head2)
            if self.isInterruptionRequested():
                return

            tail1 = self.gui._get_processed_tail_image(self.path1)
            self.tail_ready.emit(1, tail1)
            if self.isInterruptionRequested():
                return
            tail2 = self.gui._get_processed_tail_image(self.path2)
            self.tail_ready.emit(2, tail2)
            if self.isInterruptionRequested():
                return

            head_prob = self.gui.compare_head(self.path1, self.path2)
            tail_prob = self.gui.compare_tail(self.path1, self.path2)
            self.probs_ready.emit(head_prob, tail_prob)
        except Exception as e:
            self.error_occurred.emit(str(e))


class LocalCompareDialog(QDialog):
    def __init__(self, gui_instance):
        super().__init__(gui_instance)
        self.gui = gui_instance
        self.path1 = None
        self.path2 = None
        self.head_prob = None
        self.tail_prob = None
        self.worker = None

        self.setWindowTitle('本地两图比对')
        self.setMinimumSize(1100, 900)

        self._init_ui()

    def closeEvent(self, event):
        try:
            if self.worker is not None and self.worker.isRunning():
                self.worker.requestInterruption()
                self.worker.wait(1500)
        except Exception:
            pass
        super().closeEvent(event)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        pick_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        self.pick1_btn = QPushButton('选择图片1')
        self.path1_label = QLabel('未选择')
        self.path1_label.setStyleSheet('color: #7f8c8d;')
        row1.addWidget(self.pick1_btn)
        row1.addWidget(self.path1_label, 1)
        pick_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.pick2_btn = QPushButton('选择图片2')
        self.path2_label = QLabel('未选择')
        self.path2_label.setStyleSheet('color: #7f8c8d;')
        row2.addWidget(self.pick2_btn)
        row2.addWidget(self.path2_label, 1)
        pick_layout.addLayout(row2)

        top_layout = QHBoxLayout()
        top_layout.addLayout(pick_layout, 1)
        self.start_btn = QPushButton('开始比对')
        top_layout.addWidget(self.start_btn)
        main_layout.addLayout(top_layout)

        self.pick1_btn.clicked.connect(self._pick_image1)
        self.pick2_btn.clicked.connect(self._pick_image2)
        self.start_btn.clicked.connect(self._start_compare)

        th_layout = QHBoxLayout()
        self.head_low_spin = QDoubleSpinBox()
        self.head_low_spin.setRange(0.0, 1.0)
        self.head_low_spin.setSingleStep(0.05)
        self.head_low_spin.setDecimals(2)
        self.head_low_spin.setValue(0.8)

        self.head_same_spin = QDoubleSpinBox()
        self.head_same_spin.setRange(0.0, 1.0)
        self.head_same_spin.setSingleStep(0.05)
        self.head_same_spin.setDecimals(2)
        self.head_same_spin.setValue(0.3)

        self.tail_low_spin = QDoubleSpinBox()
        self.tail_low_spin.setRange(0.0, 1.0)
        self.tail_low_spin.setSingleStep(0.05)
        self.tail_low_spin.setDecimals(2)
        self.tail_low_spin.setValue(0.3)

        self.head_low_spin.valueChanged.connect(self._refresh_judgement)
        self.head_same_spin.valueChanged.connect(self._refresh_judgement)
        self.tail_low_spin.valueChanged.connect(self._refresh_judgement)

        th_layout.addWidget(QLabel('头部低阈值:'))
        th_layout.addWidget(self.head_low_spin)
        th_layout.addWidget(QLabel('头部同阈值:'))
        th_layout.addWidget(self.head_same_spin)
        th_layout.addWidget(QLabel('尾部阈值:'))
        th_layout.addWidget(self.tail_low_spin)
        th_layout.addStretch()
        main_layout.addLayout(th_layout)

        images_scroll = QScrollArea()
        images_scroll.setWidgetResizable(True)
        images_widget = QWidget()
        grid = QGridLayout(images_widget)

        def _make_image_cell(title: str):
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet('font-weight: bold; color: #2c3e50;')
            image_label = QLabel('未处理')
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(360, 230)
            image_label.setStyleSheet('border: 2px dashed #bdc3c7; background-color: #ecf0f1;')
            cell = QVBoxLayout()
            cell.addWidget(title_label)
            cell.addWidget(image_label)
            w = QWidget()
            w.setLayout(cell)
            return w, image_label

        w_o1, self.original1_label = _make_image_cell('原图1')
        w_o2, self.original2_label = _make_image_cell('原图2')
        w_h1, self.head1_label = _make_image_cell('车头1（打码后）')
        w_h2, self.head2_label = _make_image_cell('车头2（打码后）')
        w_t1, self.tail1_label = _make_image_cell('车尾1')
        w_t2, self.tail2_label = _make_image_cell('车尾2')

        grid.addWidget(w_o1, 0, 0)
        grid.addWidget(w_o2, 0, 1)
        grid.addWidget(w_h1, 1, 0)
        grid.addWidget(w_h2, 1, 1)
        grid.addWidget(w_t1, 2, 0)
        grid.addWidget(w_t2, 2, 1)

        images_scroll.setWidget(images_widget)
        main_layout.addWidget(images_scroll, 1)

        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(170)
        self.result_text.setReadOnly(True)
        main_layout.addWidget(self.result_text)

        self._set_status('请选择两张图片后点击“开始比对”。')

    def _set_status(self, text: str) -> None:
        self.result_text.setText(text)

    def _pick_image1(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择图片1', '', '图片文件 (*.jpg *.jpeg *.png *.bmp)')
        if not path:
            return
        self.path1 = path
        self.path1_label.setText(path)
        self.gui.display_image(self.path1, self.original1_label)

    def _pick_image2(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择图片2', '', '图片文件 (*.jpg *.jpeg *.png *.bmp)')
        if not path:
            return
        self.path2 = path
        self.path2_label.setText(path)
        self.gui.display_image(self.path2, self.original2_label)

    def _reset_outputs(self):
        for label in [self.head1_label, self.head2_label, self.tail1_label, self.tail2_label]:
            label.setText('未处理')
            label.setPixmap(QPixmap())
        self.head_prob = None
        self.tail_prob = None

    def _start_compare(self):
        if not self.path1 or not os.path.exists(self.path1):
            QMessageBox.warning(self, '提示', '请选择有效的图片1。')
            return
        if not self.path2 or not os.path.exists(self.path2):
            QMessageBox.warning(self, '提示', '请选择有效的图片2。')
            return

        self._reset_outputs()
        self.gui.display_image(self.path1, self.original1_label)
        self.gui.display_image(self.path2, self.original2_label)

        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(1500)

        self._set_status('开始比对：正在处理车头/车尾，请稍候...')
        self.start_btn.setEnabled(False)

        self.worker = LocalCompareWorker(self.gui, self.path1, self.path2)
        self.worker.head_ready.connect(self._on_head_ready)
        self.worker.tail_ready.connect(self._on_tail_ready)
        self.worker.probs_ready.connect(self._on_probs_ready)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.finished.connect(lambda: self.start_btn.setEnabled(True))
        self.worker.start()

    def _on_head_ready(self, idx: int, img_obj: object):
        if idx == 1:
            self.gui._display_pil_image(img_obj, self.head1_label, '车头1处理失败')
        else:
            self.gui._display_pil_image(img_obj, self.head2_label, '车头2处理失败')

    def _on_tail_ready(self, idx: int, img_obj: object):
        if idx == 1:
            self.gui._display_pil_image(img_obj, self.tail1_label, '车尾1处理失败')
        else:
            self.gui._display_pil_image(img_obj, self.tail2_label, '车尾2处理失败')

    def _on_probs_ready(self, head_prob_obj: object, tail_prob_obj: object):
        self.head_prob = head_prob_obj
        self.tail_prob = tail_prob_obj
        self._refresh_judgement()

    def _on_worker_error(self, msg: str):
        self._set_status(f'比对失败: {msg}')

    def _refresh_judgement(self):
        head_prob = self.head_prob
        tail_prob = self.tail_prob

        if head_prob is None or tail_prob is None:
            result_line = '结果: 异常（无法完成比对）'
        else:
            try:
                hp = float(head_prob)
                tp = float(tail_prob)
            except Exception:
                result_line = '结果: 异常（无法完成比对）'
            else:
                head_low = float(self.head_low_spin.value())
                head_same = float(self.head_same_spin.value())
                tail_low = float(self.tail_low_spin.value())

                if hp < head_low:
                    result_line = '结果: 疑似套牌'
                elif hp > head_same and tp <= tail_low:
                    result_line = '结果: 疑似换挂'
                else:
                    result_line = '结果: 正常'

        info_lines = [
            result_line,
            f"head_prob: {head_prob if head_prob is not None else 'N/A'}",
            f"tail_prob: {tail_prob if tail_prob is not None else 'N/A'}",
            f"图片1: {self.path1}",
            f"图片2: {self.path2}",
            f"阈值: head_low={self.head_low_spin.value():.2f}, head_same={self.head_same_spin.value():.2f}, tail_low={self.tail_low_spin.value():.2f}",
        ]
        self._set_status("\n".join(info_lines))


class CarPlateRecognitionGUI(QMainWindow):
    """车头 + 车尾双路对比的批量检测 GUI。"""

    def __init__(self):
        super().__init__()
        # 输出文件管理配置
        self.OUTPUT_BASE_DIR = r"D:\output"
        # 车头 Siamese 模型（车牌打码后对比）
        self.head_model = Siamese(
            model_path=r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\logs\head\1211\best_epoch_weights.pth"
        )
        # 车尾 Siamese 模型（不打码）
        self.tail_model = Siamese(
            model_path=r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\logs\weibu\1211\best_epoch_weights.pth"
        )

        self.cropper = VehicleCropper()
        self.plate_recognizer = PlateRecognizer()
        # 头尾检测模型：用于从整车图中裁头或裁尾
        self.headtail_model = YOLO(r"D:\data2\runs\detect\train\weights\best.pt")

        # 阈值：<=0.3 认为头部相似度低；>0.3 头没变
        self.HEAD_LOW_TH = 0.8
        self.HEAD_SAME_TH = 0.3
        # 尾部相似度阈值，默认 0.6，可在界面中调整用于换挂判定与结果筛选
        self.TAIL_LOW_TH = 0.3

        self.image1_path = None
        self.image2_path = None
        self.suspicious_pairs: List[dict] = []   # 全部疑似记录
        self.filtered_pairs: List[dict] = []     # 按 CASE_TYPE 过滤后的记录
        self.current_pair_index = -1

        # 确保输出目录存在
        self._ensure_output_dir()

        self.init_ui()
        # 初始化时刷新一次 last_task_id 显示
        self.update_last_task_id_label()

        # 默认结果CSV路径
        self.DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), 'suspected_fake_or_change_20251218_124640.csv')
        # 启动时自动加载默认CSV（若存在）
        self._load_default_csv_on_start()

        self._batch_force_overwrite = False
        self._batch_last_task_id_candidate = None

    # -------------- 输出文件管理 --------------
    def _ensure_output_dir(self) -> None:
        """确保输出目录存在"""
        try:
            os.makedirs(self.OUTPUT_BASE_DIR, exist_ok=True)
        except Exception as e:
            print(f"创建输出目录失败: {e}")

    def _create_result_folder(self, task_id: str, case_type: str) -> str:
        """为检测结果创建专用文件夹"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f"{task_id}_{case_type}_{timestamp}"
        folder_path = os.path.join(self.OUTPUT_BASE_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def _save_result_images(self, curr_path: str, prev_path: str, output_folder: str) -> dict:
        """保存6张处理图片到指定文件夹，返回图片路径字典"""
        image_paths = {}

        try:
            if curr_path and (not os.path.exists(curr_path)):
                print(f"原图不存在: {curr_path}")
            if prev_path and (not os.path.exists(prev_path)):
                print(f"原图不存在: {prev_path}")

            # 保存原图
            if curr_path and os.path.exists(curr_path):
                curr_original_path = os.path.join(output_folder, "original_curr.jpg")
                shutil.copy2(curr_path, curr_original_path)
                image_paths['original_curr'] = curr_original_path

            if prev_path and os.path.exists(prev_path):
                prev_original_path = os.path.join(output_folder, "original_prev.jpg")
                shutil.copy2(prev_path, prev_original_path)
                image_paths['original_prev'] = prev_original_path

            # 处理并保存车头图片（打码后）
            if curr_path and os.path.exists(curr_path):
                head_curr_img = self._get_processed_head_image(curr_path)
                if head_curr_img:
                    head_curr_path = os.path.join(output_folder, "head_curr.jpg")
                    head_curr_img.save(head_curr_path, 'JPEG')
                    image_paths['head_curr'] = head_curr_path

            if prev_path and os.path.exists(prev_path):
                head_prev_img = self._get_processed_head_image(prev_path)
                if head_prev_img:
                    head_prev_path = os.path.join(output_folder, "head_prev.jpg")
                    head_prev_img.save(head_prev_path, 'JPEG')
                    image_paths['head_prev'] = head_prev_path

            # 处理并保存车尾图片
            if curr_path and os.path.exists(curr_path):
                tail_curr_img = self._get_processed_tail_image(curr_path)
                if tail_curr_img:
                    tail_curr_path = os.path.join(output_folder, "tail_curr.jpg")
                    tail_curr_img.save(tail_curr_path, 'JPEG')
                    image_paths['tail_curr'] = tail_curr_path

            if prev_path and os.path.exists(prev_path):
                tail_prev_img = self._get_processed_tail_image(prev_path)
                if tail_prev_img:
                    tail_prev_path = os.path.join(output_folder, "tail_prev.jpg")
                    tail_prev_img.save(tail_prev_path, 'JPEG')
                    image_paths['tail_prev'] = tail_prev_path

        except Exception as e:
            print(f"保存处理图片失败: {e}")

        return image_paths

    def _clear_all_output(self) -> None:
        """清空所有输出文件"""
        try:
            if os.path.exists(self.OUTPUT_BASE_DIR):
                shutil.rmtree(self.OUTPUT_BASE_DIR)
            self._ensure_output_dir()
        except Exception as e:
            print(f"清空输出目录失败: {e}")

    def _delete_result_folder(self, output_folder: str) -> None:
        """删除指定结果文件夹"""
        try:
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
        except Exception as e:
            print(f"删除结果文件夹失败: {e}")

    def _find_image_in_folder(self, folder_path: str, image_type: str) -> Optional[str]:
        """在输出文件夹中查找指定类型的图片"""
        if not os.path.exists(folder_path):
            return None
        image_file = f"{image_type}.jpg"
        image_path = os.path.join(folder_path, image_file)
        if os.path.exists(image_path):
            return image_path
        return None

    # -------------- CSV格式兼容处理 --------------
    def _load_old_format_csv(self, csv_path: str) -> List[dict]:
        """加载旧格式CSV：TASK_ID, CURR_IMAGE_PATH, PREV_IMAGE_PATH, CASE_TYPE, HEAD_PROB, TAIL_PROB"""
        pairs = []
        try:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='gbk')
            required_old_cols = ['TASK_ID', 'CURR_IMAGE_PATH', 'PREV_IMAGE_PATH', 'CASE_TYPE', 'HEAD_PROB', 'TAIL_PROB']
            if all(col in df.columns for col in required_old_cols):
                for _, row in df.iterrows():
                    pairs.append({
                        'task_id': row['TASK_ID'],
                        'curr_path': row['CURR_IMAGE_PATH'],
                        'prev_path': row['PREV_IMAGE_PATH'],
                        'case_type': row['CASE_TYPE'],
                        'head_prob': row['HEAD_PROB'],
                        'tail_prob': row['TAIL_PROB'],
                        'output_folder': None,  # 旧格式没有输出文件夹
                        'format_type': 'old'
                    })
        except Exception as e:
            print(f"加载旧格式CSV失败: {e}")
        return pairs

    def _load_new_format_csv(self, csv_path: str) -> List[dict]:
        """加载新格式CSV：TASK_ID, OUTPUT_FOLDER, CASE_TYPE, HEAD_PROB, TAIL_PROB"""
        pairs = []
        try:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='gbk')
            required_new_cols = ['TASK_ID', 'OUTPUT_FOLDER', 'CASE_TYPE', 'HEAD_PROB', 'TAIL_PROB']
            if all(col in df.columns for col in required_new_cols):
                for _, row in df.iterrows():
                    output_folder = row['OUTPUT_FOLDER']
                    # 从输出文件夹获取图片路径
                    curr_path = self._find_image_in_folder(output_folder, 'original_curr')
                    prev_path = self._find_image_in_folder(output_folder, 'original_prev')
                    pairs.append({
                        'task_id': row['TASK_ID'],
                        'curr_path': curr_path,
                        'prev_path': prev_path,
                        'case_type': row['CASE_TYPE'],
                        'head_prob': row['HEAD_PROB'],
                        'tail_prob': row['TAIL_PROB'],
                        'output_folder': output_folder,
                        'format_type': 'new',
                        'image_paths': {
                            'original_curr': curr_path,
                            'original_prev': prev_path,
                            'head_curr': self._find_image_in_folder(output_folder, 'head_curr'),
                            'head_prev': self._find_image_in_folder(output_folder, 'head_prev'),
                            'tail_curr': self._find_image_in_folder(output_folder, 'tail_curr'),
                            'tail_prev': self._find_image_in_folder(output_folder, 'tail_prev')
                        }
                    })
        except Exception as e:
            print(f"加载新格式CSV失败: {e}")
        return pairs
    # ---------------- UI -----------------
    def init_ui(self):
        self.setWindowTitle('柳钢套牌车识别系统v3.0')
        self.setGeometry(100, 100, 1200, 900)

        tools_menu = self.menuBar().addMenu('工具')
        local_compare_action = QAction('本地两图比对...', self)
        local_compare_action.triggered.connect(self.open_local_compare_dialog)
        tools_menu.addAction(local_compare_action)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部：标题 + 类型筛选
        header_layout = QHBoxLayout()
        title_label = QLabel('套牌车智能检测系统')
        title_label.setStyleSheet('font-size: 24px; font-weight: bold; color: #2c3e50;')
        title_label.setAlignment(Qt.AlignCenter)

        header_layout.addStretch()
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        self.case_filter_combo = QComboBox()
        self.case_filter_combo.addItem('全部', userData=None)
        self.case_filter_combo.addItem('疑似套牌', userData='fake_plate')
        self.case_filter_combo.addItem('疑似换挂', userData='change_trailer')
        self.case_filter_combo.currentIndexChanged.connect(self.apply_case_filter)
        header_layout.addWidget(QLabel('筛选:'))
        header_layout.addWidget(self.case_filter_combo)

        # 尾部相似度阈值调节（用于换挂判定和结果筛选）
        self.tail_th_spin = QDoubleSpinBox()
        self.tail_th_spin.setRange(0.0, 1.0)
        self.tail_th_spin.setSingleStep(0.05)
        self.tail_th_spin.setDecimals(2)
        self.tail_th_spin.setValue(self.TAIL_LOW_TH)
        self.tail_th_spin.valueChanged.connect(self.on_tail_threshold_changed)
        header_layout.addWidget(QLabel('尾部阈值:'))
        header_layout.addWidget(self.tail_th_spin)

        # 显示当前记录的 last_task_id
        self.last_task_id_label = QLabel('last_task_id: 无记录')
        self.last_task_id_label.setStyleSheet('color: #7f8c8d;')
        header_layout.addWidget(self.last_task_id_label)

        main_layout.addLayout(header_layout)

        # 图片区域 - 3行2列布局
        images_scroll = QScrollArea()
        images_scroll.setWidgetResizable(True)
        images_scroll.setMinimumHeight(600)
        images_widget = QWidget()
        images_main_layout = QVBoxLayout(images_widget)

        # 第一行：原图1、原图2
        row1_layout = QHBoxLayout()
        row1_left = QVBoxLayout()
        row1_left_label = QLabel('原图1（当前）')
        row1_left_label.setAlignment(Qt.AlignCenter)
        row1_left_label.setStyleSheet('font-weight: bold; color: #2c3e50;')
        row1_left.addWidget(row1_left_label)
        self.original1_label = QLabel('未选择图片')
        self.original1_label.setAlignment(Qt.AlignCenter)
        self.original1_label.setMinimumSize(300, 200)
        self.original1_label.setStyleSheet('border: 2px dashed #bdc3c7; background-color: #ecf0f1;')
        row1_left.addWidget(self.original1_label)

        row1_right = QVBoxLayout()
        row1_right_label = QLabel('原图2（历史）')
        row1_right_label.setAlignment(Qt.AlignCenter)
        row1_right_label.setStyleSheet('font-weight: bold; color: #2c3e50;')
        row1_right.addWidget(row1_right_label)
        self.original2_label = QLabel('未选择图片')
        self.original2_label.setAlignment(Qt.AlignCenter)
        self.original2_label.setMinimumSize(300, 200)
        self.original2_label.setStyleSheet('border: 2px dashed #bdc3c7; background-color: #ecf0f1;')
        row1_right.addWidget(self.original2_label)

        row1_layout.addLayout(row1_left)
        row1_layout.addLayout(row1_right)
        images_main_layout.addLayout(row1_layout)

        # 第二行：车头1(打码)、车头2(打码)
        row2_layout = QHBoxLayout()
        row2_left = QVBoxLayout()
        row2_left_label = QLabel('车头1（打码后）')
        row2_left_label.setAlignment(Qt.AlignCenter)
        row2_left_label.setStyleSheet('font-weight: bold; color: #e67e22;')
        row2_left.addWidget(row2_left_label)
        self.head1_label = QLabel('未处理')
        self.head1_label.setAlignment(Qt.AlignCenter)
        self.head1_label.setMinimumSize(300, 200)
        self.head1_label.setStyleSheet('border: 2px dashed #bdc3c7; background-color: #ecf0f1;')
        row2_left.addWidget(self.head1_label)

        row2_right = QVBoxLayout()
        row2_right_label = QLabel('车头2（打码后）')
        row2_right_label.setAlignment(Qt.AlignCenter)
        row2_right_label.setStyleSheet('font-weight: bold; color: #e67e22;')
        row2_right.addWidget(row2_right_label)
        self.head2_label = QLabel('未处理')
        self.head2_label.setAlignment(Qt.AlignCenter)
        self.head2_label.setMinimumSize(300, 200)
        self.head2_label.setStyleSheet('border: 2px dashed #bdc3c7; background-color: #ecf0f1;')
        row2_right.addWidget(self.head2_label)

        row2_layout.addLayout(row2_left)
        row2_layout.addLayout(row2_right)
        images_main_layout.addLayout(row2_layout)

        # 第三行：车尾1、车尾2
        row3_layout = QHBoxLayout()
        row3_left = QVBoxLayout()
        row3_left_label = QLabel('车尾1')
        row3_left_label.setAlignment(Qt.AlignCenter)
        row3_left_label.setStyleSheet('font-weight: bold; color: #27ae60;')
        row3_left.addWidget(row3_left_label)
        self.tail1_label = QLabel('未处理')
        self.tail1_label.setAlignment(Qt.AlignCenter)
        self.tail1_label.setMinimumSize(300, 200)
        self.tail1_label.setStyleSheet('border: 2px dashed #bdc3c7; background-color: #ecf0f1;')
        row3_left.addWidget(self.tail1_label)

        row3_right = QVBoxLayout()
        row3_right_label = QLabel('车尾2')
        row3_right_label.setAlignment(Qt.AlignCenter)
        row3_right_label.setStyleSheet('font-weight: bold; color: #27ae60;')
        row3_right.addWidget(row3_right_label)
        self.tail2_label = QLabel('未处理')
        self.tail2_label.setAlignment(Qt.AlignCenter)
        self.tail2_label.setMinimumSize(300, 200)
        self.tail2_label.setStyleSheet('border: 2px dashed #bdc3c7; background-color: #ecf0f1;')
        row3_right.addWidget(self.tail2_label)

        row3_layout.addLayout(row3_left)
        row3_layout.addLayout(row3_right)
        images_main_layout.addLayout(row3_layout)

        images_scroll.setWidget(images_widget)
        main_layout.addWidget(images_scroll)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.batch_btn = QPushButton('从数据库批量检测')
        self.batch_btn.clicked.connect(self.run_batch_check_from_gui)
        btn_layout.addWidget(self.batch_btn)

        self.load_csv_btn = QPushButton('从CSV加载结果')
        self.load_csv_btn.clicked.connect(self.load_suspicious_from_csv)
        btn_layout.addWidget(self.load_csv_btn)

        self.prev_pair_btn = QPushButton('上一对')
        self.next_pair_btn = QPushButton('下一对')
        self.prev_pair_btn.clicked.connect(self.show_prev_suspicious_pair)
        self.next_pair_btn.clicked.connect(self.show_next_suspicious_pair)
        self.prev_pair_btn.setEnabled(False)
        self.next_pair_btn.setEnabled(False)
        btn_layout.addWidget(self.prev_pair_btn)
        btn_layout.addWidget(self.next_pair_btn)

        self.save_pair_btn = QPushButton('保存原图...')
        self.save_pair_btn.clicked.connect(self.save_current_originals)
        self.save_pair_btn.setEnabled(False)
        btn_layout.addWidget(self.save_pair_btn)

        self.delete_pair_btn = QPushButton('删除当前记录')
        self.delete_pair_btn.clicked.connect(self.delete_current_pair)
        self.delete_pair_btn.setEnabled(False)
        self.delete_pair_btn.setStyleSheet('''
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        ''')
        btn_layout.addWidget(self.delete_pair_btn)

        main_layout.addLayout(btn_layout)

        # 结果文本
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(160)
        self.result_text.setReadOnly(True)
        main_layout.addWidget(self.result_text)

    def open_local_compare_dialog(self):
        dlg = LocalCompareDialog(self)
        dlg.exec()

    # -------------- 工具方法 --------------
    def display_image(self, path: Optional[str], label: QLabel):
        if not path or not os.path.exists(path):
            label.setText('图片不存在')
            label.setPixmap(QPixmap())
            return
        try:
            pixmap = QPixmap(path)
            scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
        except Exception:
            label.setText('加载失败')

    def _last_task_id_file_path(self) -> str:
        """返回记录上次最大 TASK_ID 的本地文件路径。"""
        return os.path.join(os.path.dirname(__file__), 'last_task_id.txt')

    def _load_last_task_id(self) -> Optional[int]:
        """从本地文件读取上次处理到的最大 TASK_ID，没有则返回 None。"""
        path = self._last_task_id_file_path()
        try:
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text:
                return None
            return int(text)
        except Exception as e:
            print(f"读取 last_task_id 失败: {e}")
            return None

    def _save_last_task_id(self, task_id: int) -> None:
        """将本次处理到的最大 TASK_ID 写入本地文件。"""
        try:
            path = self._last_task_id_file_path()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(str(int(task_id)))
        except Exception as e:
            print(f"保存 last_task_id 失败: {e}")

    def update_last_task_id_label(self) -> None:
        """根据本地记录刷新界面上的 last_task_id 显示。"""
        try:
            last_tid = self._load_last_task_id()
            if last_tid is None:
                text = 'last_task_id: 无记录'
            else:
                text = f'last_task_id: {last_tid}'
            if hasattr(self, 'last_task_id_label') and self.last_task_id_label is not None:
                self.last_task_id_label.setText(text)
        except Exception as e:
            print(f"更新 last_task_id 显示失败: {e}")

    def _load_default_csv_on_start(self):
        """启动时自动加载默认CSV（若存在）"""
        if os.path.exists(self.DEFAULT_CSV_PATH):
            try:
                self._load_suspicious_from_csv_path(self.DEFAULT_CSV_PATH)
                print(f"启动时自动加载了默认CSV: {self.DEFAULT_CSV_PATH}")
            except Exception as e:
                print(f"启动时加载默认CSV失败: {e}")

    # -------------- 头尾裁剪与对比 --------------
    def _crop_head_from_vehicle_pil(self, vehicle_image: Image.Image) -> Image.Image:
        """在整车 PIL 图上用 head/tail 模型裁车头；失败则返回原图。"""
        try:
            if vehicle_image is None:
                return vehicle_image
            rgb = vehicle_image.convert('RGB')
            img_np = np.array(rgb)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            results = self.headtail_model(img_bgr, conf=0.25, verbose=False)
            if not results:
                return vehicle_image
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                return vehicle_image

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            best_idx = None
            best_score = -1.0
            for i, (cls_id, score) in enumerate(zip(classes, scores)):
                if int(cls_id) != 0:
                    continue
                if float(score) > best_score:
                    best_score = float(score)
                    best_idx = i
            if best_idx is None:
                return vehicle_image

            x1, y1, x2, y2 = boxes[int(best_idx)]
            h, w = img_bgr.shape[:2]
            x1 = max(0, min(int(x1), w - 1))
            x2 = max(0, min(int(x2), w))
            y1 = max(0, min(int(y1), h - 1))
            y2 = max(0, min(int(y2), h))
            if x2 <= x1 or y2 <= y1:
                return vehicle_image

            head_bgr = img_bgr[y1:y2, x1:x2].copy()
            if head_bgr.size == 0:
                return vehicle_image
            head_rgb = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(head_rgb)
        except Exception:
            return vehicle_image

    def _crop_tail_from_vehicle_pil(self, vehicle_image: Image.Image) -> Image.Image:
        """在整车 PIL 图上用 head/tail 模型裁车尾（cls==1）；失败则返回原图。"""
        try:
            if vehicle_image is None:
                return vehicle_image
            rgb = vehicle_image.convert('RGB')
            img_np = np.array(rgb)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            results = self.headtail_model(img_bgr, conf=0.25, verbose=False)
            if not results:
                return vehicle_image
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                return vehicle_image

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            best_idx = None
            best_score = -1.0
            for i, (cls_id, score) in enumerate(zip(classes, scores)):
                if int(cls_id) != 1:
                    continue
                if float(score) > best_score:
                    best_score = float(score)
                    best_idx = i
            if best_idx is None:
                return vehicle_image

            x1, y1, x2, y2 = boxes[int(best_idx)]
            h, w = img_bgr.shape[:2]
            x1 = max(0, min(int(x1), w - 1))
            x2 = max(0, min(int(x2), w))
            y1 = max(0, min(int(y1), h - 1))
            y2 = max(0, min(int(y2), h))
            if x2 <= x1 or y2 <= y1:
                return vehicle_image

            tail_bgr = img_bgr[y1:y2, x1:x2].copy()
            if tail_bgr.size == 0:
                return vehicle_image
            tail_rgb = cv2.cvtColor(tail_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(tail_rgb)
        except Exception:
            return vehicle_image

    def _mask_plate_region(self, head_image: Image.Image) -> Image.Image:
        """在车头图中对车牌区域涂黑。失败则返回原图。"""
        try:
            if head_image is None:
                return head_image
            rgb = head_image.convert('RGB')
            img_np = np.array(rgb)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            ocr = self.plate_recognizer.ocr
            result = ocr.ocr(img_bgr, cls=False)
            if not result or not result[0]:
                return head_image

            best_box = None
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                conf = float(line[1][1]) if line[1][1] is not None else 0.0
                if conf < 0.7:
                    continue
                if not self.plate_recognizer.is_valid_plate(text):
                    continue
                best_box = box
                break
            if best_box is None:
                return head_image

            xs = [p[0] for p in best_box]
            ys = [p[1] for p in best_box]
            x1, x2 = int(max(0, min(xs))), int(max(xs))
            y1, y2 = int(max(0, min(ys))), int(max(ys))

            h, w = img_bgr.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                return head_image

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
            masked_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(masked_rgb)
        except Exception:
            return head_image

    def compare_head(self, path1: str, path2: str) -> Optional[float]:
        """车头相似度：整车 -> 车头 -> 打码 -> head_model 对比。"""
        try:
            if (not path1) or (not path2):
                return None
            if (not os.path.exists(path1)) or (not os.path.exists(path2)):
                return None

            img1 = Image.open(path1)
            img2 = Image.open(path2)
            img1 = self.cropper.process_pil(img1)
            img2 = self.cropper.process_pil(img2)

            img1 = self._crop_head_from_vehicle_pil(img1)
            img2 = self._crop_head_from_vehicle_pil(img2)

            img1 = self._mask_plate_region(img1)
            img2 = self._mask_plate_region(img2)

            prob = self.head_model.detect_image(img1, img2)
            prob = prob.item() if hasattr(prob, 'item') else float(prob)
            return prob
        except Exception as e:
            print(f"compare_head 出错: {e}")
            return None

    def compare_tail(self, path1: str, path2: str) -> Optional[float]:
        """车尾相似度：整车 -> 车尾 -> tail_model 对比（不打码）。"""
        try:
            if (not path1) or (not path2):
                return None
            if (not os.path.exists(path1)) or (not os.path.exists(path2)):
                return None

            img1 = Image.open(path1)
            img2 = Image.open(path2)
            img1 = self.cropper.process_pil(img1)
            img2 = self.cropper.process_pil(img2)

            img1 = self._crop_tail_from_vehicle_pil(img1)
            img2 = self._crop_tail_from_vehicle_pil(img2)

            prob = self.tail_model.detect_image(img1, img2)
            prob = prob.item() if hasattr(prob, 'item') else float(prob)
            return prob
        except Exception as e:
            print(f"compare_tail 出错: {e}")
            return None

    # -------------- 数据库读写 --------------
    def connect_to_oracle(self):
        try:
            os.environ["PATH"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0" + ";" + os.environ.get("PATH", "")
            os.environ["TNS_ADMIN"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0\\network\\admin"
            dsn_tns = cx_Oracle.makedsn('10.100.2.229', '1521', service_name='JLYXZ')
            conn = cx_Oracle.connect(user='identify', password='123456', dsn=dsn_tns)
            print("成功连接到Oracle数据库")
            return conn
        except Exception as e:
            print(f"连接数据库时出错: {e}")
            return None

    def read_pic_matchtask_by_gross_time(self, connection):
        try:
            query = """
            SELECT *
            FROM jlyxz.PIC_MATCHTASK
            WHERE GROSS_WEIGH_TIME IS NOT NULL
            ORDER BY GROSS_WEIGH_TIME ASC
            """
            df = pd.read_sql(query, connection)
            return df
        except Exception as e:
            print(f"读取PIC_MATCHTASK表数据时出错: {e}")
            return None
    # -------------- 批量检测主流程 --------------
    def run_batch_check_from_gui(self):
        """从数据库批量检测，支持从头检测和接着检测两种模式"""
        self.result_text.setText('开始从数据库读取数据并批量检测，请稍候...')
        QApplication.processEvents()

        conn = self.connect_to_oracle()
        if conn is None:
            QMessageBox.critical(self, '错误', '无法连接到Oracle数据库，请检查配置。')
            return

        try:
            df = self.read_pic_matchtask_by_gross_time(conn)
        finally:
            conn.close()

        if df is None or df.empty:
            QMessageBox.information(self, '结果', '未从数据库中读取到任何PIC_MATCHTASK数据。')
            self.result_text.setText('数据库中没有可用的数据。')
            return

        required_cols = ['TASK_ID', 'TRUCK_ID', 'GROSS_WEIGH_TIME', 'TARE_IMAGE_PATH1', 'GROSS_IMAGE_PATH1']
        for col in required_cols:
            if col not in df.columns:
                QMessageBox.critical(self, '错误', f'数据表缺少必要字段: {col}')
                self.result_text.setText(f'数据表缺少必要字段: {col}')
                return

        df = df[df['GROSS_WEIGH_TIME'].notna()].copy()
        if df.empty:
            QMessageBox.information(self, '结果', '没有包含GROSS_WEIGH_TIME的数据记录。')
            self.result_text.setText('没有包含GROSS_WEIGH_TIME的数据记录。')
            return

        df = df.sort_values('GROSS_WEIGH_TIME')

        # 选择检测模式
        last_task_id = self._load_last_task_id()
        append_mode = False

        self._batch_force_overwrite = False
        self._batch_last_task_id_candidate = None

        def _pick_start_date() -> int:
            dlg = QDialog(self)
            dlg.setWindowTitle('选择起始日期')
            layout = QVBoxLayout(dlg)
            date_edit = QDateEdit(dlg)
            date_edit.setCalendarPopup(True)
            date_edit.setDisplayFormat('yyyy-MM-dd')
            date_edit.setDate(QDate.currentDate())
            layout.addWidget(date_edit)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
            layout.addWidget(btns)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            if dlg.exec() != QDialog.Accepted:
                return -1
            qd = date_edit.date()
            return (qd.year() % 100) * 10000 + qd.month() * 100 + qd.day()

        def _cleanup_output_before_yymmdd(start_yymmdd_int: int) -> None:
            try:
                base = os.path.abspath(self.OUTPUT_BASE_DIR)
                if not os.path.exists(base):
                    return
                for name in os.listdir(base):
                    folder_path = os.path.join(base, name)
                    if not os.path.isdir(folder_path):
                        continue
                    try:
                        task_id_part = str(name).split('_', 1)[0]
                        if len(task_id_part) < 6:
                            continue
                        yymmdd = task_id_part[:6]
                        if not yymmdd.isdigit():
                            continue
                        if int(yymmdd) < int(start_yymmdd_int):
                            abs_folder = os.path.abspath(folder_path)
                            if abs_folder.startswith(base):
                                shutil.rmtree(abs_folder)
                    except Exception:
                        continue
            except Exception as e:
                print(f"清理旧输出文件夹失败: {e}")

        def _write_empty_default_csv() -> None:
            try:
                empty_df = pd.DataFrame(columns=['TASK_ID', 'OUTPUT_FOLDER', 'CASE_TYPE', 'HEAD_PROB', 'TAIL_PROB'])
                os.makedirs(os.path.dirname(self.DEFAULT_CSV_PATH), exist_ok=True)
                empty_df.to_csv(self.DEFAULT_CSV_PATH, index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"写入空CSV失败: {e}")

        if last_task_id is not None:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('选择检测模式')
            msg_box.setText(
                f"检测到上次已处理到 TASK_ID = {last_task_id}。\n\n"
                "请选择检测模式：\n\n"
                "从头检测：清空所有现有结果文件和CSV，重新检测所有数据\n"
                "接着检测：只检测新数据，追加到现有结果中\n"
                "从指定日期开始：删除起始日期之前的输出文件夹，覆盖CSV并重新检测（包含当天）"
            )
            restart_btn = msg_box.addButton('从头检测', QMessageBox.YesRole)
            continue_btn = msg_box.addButton('接着检测', QMessageBox.NoRole)
            date_btn = msg_box.addButton('从指定日期开始...', QMessageBox.ActionRole)
            cancel_btn = msg_box.addButton('取消', QMessageBox.RejectRole)
            msg_box.exec()

            clicked = msg_box.clickedButton()
            if clicked == cancel_btn:
                return
            elif clicked == restart_btn:
                # 从头检测：清空所有输出文件和CSV
                self._clear_all_output()
                if os.path.exists(self.DEFAULT_CSV_PATH):
                    os.remove(self.DEFAULT_CSV_PATH)
                append_mode = False
            elif clicked == continue_btn:
                # 接着检测：只处理新数据
                try:
                    df['TASK_ID_INT'] = pd.to_numeric(df['TASK_ID'], errors='coerce')
                    df = df[df['TASK_ID_INT'].notna()]
                    df = df[df['TASK_ID_INT'] > int(last_task_id)]
                except Exception as e:
                    print(f"按 last_task_id 过滤数据失败: {e}")
                if df.empty:
                    QMessageBox.information(self, '结果', '没有比上次更新的任务记录，本次无需检测。')
                    self.result_text.setText('没有比上次更新的任务记录，本次无需检测。')
                    return
                append_mode = True
            elif clicked == date_btn:
                start_yymmdd_int = _pick_start_date()
                if start_yymmdd_int < 0:
                    return
                reply = QMessageBox.question(
                    self,
                    '确认操作',
                    f"将从 {start_yymmdd_int:06d} (yyMMdd) 开始检测（包含当天）。\n\n"
                    f"这将删除 {self.OUTPUT_BASE_DIR} 中起始日期之前的结果文件夹，并覆盖写入CSV。\n\n"
                    "是否继续？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

                _cleanup_output_before_yymmdd(start_yymmdd_int)
                if os.path.exists(self.DEFAULT_CSV_PATH):
                    try:
                        os.remove(self.DEFAULT_CSV_PATH)
                    except Exception:
                        pass

                try:
                    df['TASK_ID_STR'] = df['TASK_ID'].astype(str).str.strip()
                    df['TASK_YYMMDD_INT'] = pd.to_numeric(df['TASK_ID_STR'].str.slice(0, 6), errors='coerce')
                    df = df[df['TASK_YYMMDD_INT'].notna()]
                    df = df[df['TASK_YYMMDD_INT'] >= int(start_yymmdd_int)]
                except Exception as e:
                    print(f"按起始日期过滤数据失败: {e}")

                if df.empty:
                    self.suspicious_pairs = []
                    self.filtered_pairs = []
                    self.current_pair_index = -1
                    self.prev_pair_btn.setEnabled(False)
                    self.next_pair_btn.setEnabled(False)
                    self.delete_pair_btn.setEnabled(False)
                    self.save_pair_btn.setEnabled(False)
                    _write_empty_default_csv()
                    QMessageBox.information(self, '结果', '起始日期之后没有可检测的数据记录。已清理旧结果并覆盖CSV。')
                    self.result_text.setText('起始日期之后没有可检测的数据记录。已清理旧结果并覆盖CSV。')
                    return

                append_mode = False
                self._batch_force_overwrite = True
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle('选择检测模式')
            msg_box.setText(
                "未检测到 last_task_id。\n\n"
                "请选择检测模式：\n\n"
                "从头检测：清空所有现有结果文件和CSV，重新检测所有数据\n"
                "从指定日期开始：删除起始日期之前的输出文件夹，覆盖CSV并重新检测（包含当天）"
            )
            restart_btn = msg_box.addButton('从头检测', QMessageBox.YesRole)
            date_btn = msg_box.addButton('从指定日期开始...', QMessageBox.ActionRole)
            cancel_btn = msg_box.addButton('取消', QMessageBox.RejectRole)
            msg_box.exec()

            clicked = msg_box.clickedButton()
            if clicked == cancel_btn:
                return
            elif clicked == restart_btn:
                self._clear_all_output()
                if os.path.exists(self.DEFAULT_CSV_PATH):
                    try:
                        os.remove(self.DEFAULT_CSV_PATH)
                    except Exception:
                        pass
                append_mode = False
            elif clicked == date_btn:
                start_yymmdd_int = _pick_start_date()
                if start_yymmdd_int < 0:
                    return
                reply = QMessageBox.question(
                    self,
                    '确认操作',
                    f"将从 {start_yymmdd_int:06d} (yyMMdd) 开始检测（包含当天）。\n\n"
                    f"这将删除 {self.OUTPUT_BASE_DIR} 中起始日期之前的结果文件夹，并覆盖写入CSV。\n\n"
                    "是否继续？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

                _cleanup_output_before_yymmdd(start_yymmdd_int)
                if os.path.exists(self.DEFAULT_CSV_PATH):
                    try:
                        os.remove(self.DEFAULT_CSV_PATH)
                    except Exception:
                        pass

                try:
                    df['TASK_ID_STR'] = df['TASK_ID'].astype(str).str.strip()
                    df['TASK_YYMMDD_INT'] = pd.to_numeric(df['TASK_ID_STR'].str.slice(0, 6), errors='coerce')
                    df = df[df['TASK_YYMMDD_INT'].notna()]
                    df = df[df['TASK_YYMMDD_INT'] >= int(start_yymmdd_int)]
                except Exception as e:
                    print(f"按起始日期过滤数据失败: {e}")

                if df.empty:
                    self.suspicious_pairs = []
                    self.filtered_pairs = []
                    self.current_pair_index = -1
                    self.prev_pair_btn.setEnabled(False)
                    self.next_pair_btn.setEnabled(False)
                    self.delete_pair_btn.setEnabled(False)
                    self.save_pair_btn.setEnabled(False)
                    _write_empty_default_csv()
                    QMessageBox.information(self, '结果', '起始日期之后没有可检测的数据记录。已清理旧结果并覆盖CSV。')
                    self.result_text.setText('起始日期之后没有可检测的数据记录。已清理旧结果并覆盖CSV。')
                    return

                append_mode = False
                self._batch_force_overwrite = True

        try:
            df['_TASK_ID_INT_FOR_MAX'] = pd.to_numeric(df['TASK_ID'], errors='coerce')
            df_tmp = df[df['_TASK_ID_INT_FOR_MAX'].notna()]
            if not df_tmp.empty:
                self._batch_last_task_id_candidate = int(df_tmp['_TASK_ID_INT_FOR_MAX'].max())
        except Exception:
            self._batch_last_task_id_candidate = None

        # 启动批量检测工作线程
        self.batch_worker = BatchDetectWorker(df, self)
        self.batch_worker.progress_updated.connect(self._on_batch_progress)
        self.batch_worker.result_ready.connect(lambda pairs: self._on_batch_done(pairs, append_mode))
        self.batch_worker.error_occurred.connect(self._on_batch_error)

        # 显示进度对话框
        self.progress_dialog = QProgressDialog("正在批量检测...", "取消", 0, len(df), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.canceled.connect(self.batch_worker.stop)
        self.progress_dialog.show()

        self.batch_worker.start()

    def _on_batch_progress(self, current: int, total: int):
        """批量检测进度更新"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(current)
            self.progress_dialog.setLabelText(f"正在处理第 {current}/{total} 条记录...")

    def _on_batch_done(self, suspicious_pairs: List[dict], append_mode: bool):
        """批量检测完成回调"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        self.suspicious_pairs = suspicious_pairs

        # 导出CSV
        if self.suspicious_pairs:
            export_rows = []
            for pair in self.suspicious_pairs:
                if pair.get('format_type') == 'new':
                    # 新格式：TASK_ID, OUTPUT_FOLDER, CASE_TYPE, HEAD_PROB, TAIL_PROB
                    export_rows.append({
                        'TASK_ID': pair.get('task_id'),
                        'OUTPUT_FOLDER': pair.get('output_folder'),
                        'CASE_TYPE': pair.get('case_type'),
                        'HEAD_PROB': pair.get('head_prob'),
                        'TAIL_PROB': pair.get('tail_prob'),
                    })
                else:
                    # 兼容旧格式
                    export_rows.append({
                        'TASK_ID': pair.get('task_id'),
                        'CURR_IMAGE_PATH': pair.get('curr_path'),
                        'PREV_IMAGE_PATH': pair.get('prev_path'),
                        'CASE_TYPE': pair.get('case_type'),
                        'HEAD_PROB': pair.get('head_prob'),
                        'TAIL_PROB': pair.get('tail_prob'),
                    })

            out_df = pd.DataFrame(export_rows)
            try:
                os.makedirs(os.path.dirname(self.DEFAULT_CSV_PATH), exist_ok=True)
                if append_mode and os.path.exists(self.DEFAULT_CSV_PATH):
                    # 追加写入，不写表头
                    out_df.to_csv(self.DEFAULT_CSV_PATH, index=False, encoding='utf-8-sig', mode='a', header=False)
                    action = '已追加到'
                else:
                    # 覆盖写入
                    out_df.to_csv(self.DEFAULT_CSV_PATH, index=False, encoding='utf-8-sig')
                    action = '已保存到'

                msg = (
                    f"检测完成，共发现 {len(self.suspicious_pairs)} 条疑似记录，"
                    f"{action}:\n{self.DEFAULT_CSV_PATH}"
                )
                QMessageBox.information(self, '检测完成', msg)
                self.result_text.setText(msg)

                # 重新从默认CSV加载，确保界面显示与文件一致
                self._load_suspicious_from_csv_path(self.DEFAULT_CSV_PATH)

                # 记录最大TASK_ID
                if self._batch_last_task_id_candidate:
                    self._save_last_task_id(self._batch_last_task_id_candidate)
                    self.update_last_task_id_label()

            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存CSV文件时出错: {e}')
                self.result_text.setText(f'保存CSV文件时出错: {e}')
        else:
            if getattr(self, '_batch_force_overwrite', False):
                try:
                    empty_df = pd.DataFrame(columns=['TASK_ID', 'OUTPUT_FOLDER', 'CASE_TYPE', 'HEAD_PROB', 'TAIL_PROB'])
                    os.makedirs(os.path.dirname(self.DEFAULT_CSV_PATH), exist_ok=True)
                    empty_df.to_csv(self.DEFAULT_CSV_PATH, index=False, encoding='utf-8-sig')
                except Exception as e:
                    print(f"写入空CSV失败: {e}")
                QMessageBox.information(self, '检测完成', '检测完成，未发现疑似记录。已覆盖CSV（空结果）。')
                self.result_text.setText('检测完成，未发现疑似记录。已覆盖CSV（空结果）。')
                if self._batch_last_task_id_candidate:
                    self._save_last_task_id(self._batch_last_task_id_candidate)
                    self.update_last_task_id_label()
            else:
                QMessageBox.information(self, '检测完成', '检测完成，未发现疑似记录。')
                self.result_text.setText('检测完成，未发现疑似记录。')

        self._batch_force_overwrite = False
        self._batch_last_task_id_candidate = None

        # 默认显示全部
        self.apply_case_filter()
        if self.filtered_pairs:
            self.current_pair_index = 0
            self.prev_pair_btn.setEnabled(True)
            self.next_pair_btn.setEnabled(True)
            self.delete_pair_btn.setEnabled(True)
            self.save_pair_btn.setEnabled(True)
            self.show_current_suspicious_pair()

    def _on_batch_error(self, error_msg: str):
        """批量检测错误回调"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        QMessageBox.critical(self, '检测错误', f'批量检测过程中出错: {error_msg}')
        self.result_text.setText(f'批量检测过程中出错: {error_msg}')

    # -------------- CSV加载 --------------
    def load_suspicious_from_csv(self):
        """从CSV文件加载疑似记录"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择CSV文件', '', 'CSV文件 (*.csv)'
        )
        if not file_path:
            return
        
        self._load_suspicious_from_csv_path(file_path)

    def _load_suspicious_from_csv_path(self, csv_path: str):
        """从指定路径加载CSV文件"""
        try:
            # 尝试加载新格式
            new_pairs = self._load_new_format_csv(csv_path)
            # 尝试加载旧格式
            old_pairs = self._load_old_format_csv(csv_path)
            
            # 合并结果，优先使用新格式
            if new_pairs:
                self.suspicious_pairs = new_pairs
                format_type = "新格式"
            elif old_pairs:
                self.suspicious_pairs = old_pairs
                format_type = "旧格式"
            else:
                QMessageBox.warning(self, '警告', 'CSV文件格式不正确或为空')
                return
            
            self.apply_case_filter()
            
            if self.filtered_pairs:
                self.current_pair_index = 0
                self.prev_pair_btn.setEnabled(True)
                self.next_pair_btn.setEnabled(True)
                self.delete_pair_btn.setEnabled(True)
                self.save_pair_btn.setEnabled(True)
                self.show_current_suspicious_pair()
                
                QMessageBox.information(
                    self, '加载成功', 
                    f'成功加载 {len(self.suspicious_pairs)} 条记录（{format_type}）'
                )
            else:
                QMessageBox.information(self, '结果', 'CSV文件中没有符合筛选条件的记录')
                
        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载CSV文件时出错: {e}')

    # -------------- 浏览疑似图片对 --------------
    def apply_case_filter(self):
        user_data = self.case_filter_combo.currentData() if self.case_filter_combo is not None else None
        filtered = []
        for p in self.suspicious_pairs:
            # 界面中不展示异常数据
            if p.get('case_type') == 'abnormal':
                continue
            # 先按 CASE_TYPE 过滤
            if user_data is not None and p.get('case_type') != user_data:
                continue
            # 再按尾部阈值过滤换挂案例：tail_prob 必须 <= 当前阈值
            if p.get('case_type') == 'change_trailer':
                tail_prob = p.get('tail_prob')
                if tail_prob is not None and tail_prob > self.TAIL_LOW_TH:
                    continue
            filtered.append(p)

        self.filtered_pairs = filtered

        if not self.filtered_pairs:
            self.current_pair_index = -1
            self._clear_image_displays()
            self.result_text.clear()
            self.prev_pair_btn.setEnabled(False)
            self.next_pair_btn.setEnabled(False)
            self.delete_pair_btn.setEnabled(False)
            self.save_pair_btn.setEnabled(False)
        else:
            self.current_pair_index = 0
            self.prev_pair_btn.setEnabled(True)
            self.next_pair_btn.setEnabled(True)
            self.delete_pair_btn.setEnabled(True)
            self.save_pair_btn.setEnabled(True)

    def _clear_image_displays(self):
        """清空所有图片显示"""
        for label in [self.original1_label, self.original2_label, 
                     self.head1_label, self.head2_label,
                     self.tail1_label, self.tail2_label]:
            label.setText('无记录')
            label.setPixmap(QPixmap())

    def on_tail_threshold_changed(self, value: float):
        """更新尾部相似度阈值，并重新应用筛选。"""
        self.TAIL_LOW_TH = float(value)
        self.apply_case_filter()

    def _get_processed_head_image(self, image_path: str) -> Optional[Image.Image]:
        """获取处理后的车头图片（车牌打码后）。"""
        try:
            if not image_path or not os.path.exists(image_path):
                return None
            img = Image.open(image_path)
            img = self.cropper.process_pil(img)
            img = self._crop_head_from_vehicle_pil(img)
            img = self._mask_plate_region(img)
            return img
        except Exception as e:
            print(f"处理车头图片失败 {image_path}: {e}")
            return None

    def _get_processed_tail_image(self, image_path: str) -> Optional[Image.Image]:
        """获取处理后的车尾图片。"""
        try:
            if not image_path or not os.path.exists(image_path):
                return None
            img = Image.open(image_path)
            img = self.cropper.process_pil(img)
            img = self._crop_tail_from_vehicle_pil(img)
            return img
        except Exception as e:
            print(f"处理车尾图片失败 {image_path}: {e}")
            return None

    def _display_pil_image(self, pil_image: Optional[Image.Image], label: QLabel, placeholder: str = "处理失败"):
        """在QLabel中显示PIL图片。"""
        if pil_image is None:
            label.setText(placeholder)
            label.setPixmap(QPixmap())
            return
        try:
            # PIL Image -> QPixmap
            img_array = np.array(pil_image.convert('RGB'))
            h, w, ch = img_array.shape
            bytes_per_line = ch * w
            q_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
        except Exception as e:
            print(f"显示PIL图片失败: {e}")
            label.setText(placeholder)
            label.setPixmap(QPixmap())

    def show_current_suspicious_pair(self):
        """显示当前疑似图片对，支持新旧格式"""
        if not self.filtered_pairs or self.current_pair_index < 0:
            return
        
        pair = self.filtered_pairs[self.current_pair_index]
        curr_path = pair.get('curr_path')
        prev_path = pair.get('prev_path')

        # 显示原图
        self.display_image(curr_path, self.original1_label)
        self.display_image(prev_path, self.original2_label)

        # 根据格式类型显示处理图片
        if pair.get('format_type') == 'new' and pair.get('image_paths'):
            # 新格式：直接加载已保存的处理图片，显示速度大幅提升
            image_paths = pair.get('image_paths', {})
            self.display_image(image_paths.get('head_curr'), self.head1_label)
            self.display_image(image_paths.get('head_prev'), self.head2_label)
            self.display_image(image_paths.get('tail_curr'), self.tail1_label)
            self.display_image(image_paths.get('tail_prev'), self.tail2_label)
        else:
            # 旧格式：保持原有异步处理逻辑，向后兼容
            self.result_text.setText("正在处理图片，请稍候...")
            QApplication.processEvents()
            
            head1_img = self._get_processed_head_image(curr_path)
            head2_img = self._get_processed_head_image(prev_path)
            self._display_pil_image(head1_img, self.head1_label, "车头1处理失败")
            self._display_pil_image(head2_img, self.head2_label, "车头2处理失败")

            tail1_img = self._get_processed_tail_image(curr_path)
            tail2_img = self._get_processed_tail_image(prev_path)
            self._display_pil_image(tail1_img, self.tail1_label, "车尾1处理失败")
            self._display_pil_image(tail2_img, self.tail2_label, "车尾2处理失败")

        # 显示详细信息
        info_lines = [
            f"当前第 {self.current_pair_index + 1} 对 / 共 {len(self.filtered_pairs)} 对",
            f"类型: {pair.get('tare_or_gross')}  案例类型: {pair.get('case_type')}",
            f"车牌: {pair.get('truck_id')}",
            f"当前TASK_ID: {pair.get('task_id')}, 对比TASK_ID: {pair.get('prev_task_id')}",
            f"车头相似度(head_prob): {pair.get('head_prob') if pair.get('head_prob') is not None else 'N/A'}",
            f"车尾相似度(tail_prob): {pair.get('tail_prob') if pair.get('tail_prob') is not None else 'N/A'}",
            f"当前图片路径: {curr_path}",
            f"历史图片路径: {prev_path}",
        ]
        
        if pair.get('format_type') == 'new':
            info_lines.append(f"输出文件夹: {pair.get('output_folder')}")
        
        plate_curr = pair.get('plate_curr')
        plate_prev = pair.get('plate_prev')
        info_lines.append(f"当前车牌: {plate_curr if plate_curr else '识别失败'}")
        info_lines.append(f"历史车牌: {plate_prev if plate_prev else '识别失败'}")

        self.result_text.setText("\n".join(info_lines))

    def show_prev_suspicious_pair(self):
        if not self.filtered_pairs:
            return
        self.current_pair_index = (self.current_pair_index - 1) % len(self.filtered_pairs)
        self.show_current_suspicious_pair()

    def show_next_suspicious_pair(self):
        if not self.filtered_pairs:
            return
        self.current_pair_index = (self.current_pair_index + 1) % len(self.filtered_pairs)
        self.show_current_suspicious_pair()

    def delete_current_pair(self):
        """删除当前显示的疑似图片对，同步删除输出文件夹和CSV记录"""
        if not self.filtered_pairs or self.current_pair_index < 0:
            return

        pair_to_delete = self.filtered_pairs[self.current_pair_index]
        folders_to_delete = []
        try:
            output_folder = pair_to_delete.get('output_folder')
            if isinstance(output_folder, str) and output_folder.strip() and os.path.exists(output_folder):
                folders_to_delete = [output_folder]
            else:
                image_paths = pair_to_delete.get('image_paths') or {}
                for _, p in (image_paths.items() if isinstance(image_paths, dict) else []):
                    if isinstance(p, str) and p.strip() and os.path.exists(p):
                        candidate = os.path.dirname(p)
                        if candidate:
                            folders_to_delete = [candidate]
                            break

                if not folders_to_delete:
                    task_id = pair_to_delete.get('task_id')
                    case_type = pair_to_delete.get('case_type')
                    if task_id is not None and case_type:
                        prefix = f"{task_id}_{case_type}_"
                        try:
                            for name in os.listdir(self.OUTPUT_BASE_DIR):
                                full = os.path.join(self.OUTPUT_BASE_DIR, name)
                                if os.path.isdir(full) and name.startswith(prefix):
                                    folders_to_delete.append(full)
                        except Exception:
                            folders_to_delete = []

            safe_folders = []
            base_abs = os.path.abspath(self.OUTPUT_BASE_DIR)
            for f in folders_to_delete:
                if not isinstance(f, str):
                    continue
                f_abs = os.path.abspath(f)
                try:
                    if os.path.commonpath([base_abs, f_abs]) == base_abs:
                        safe_folders.append(f)
                except Exception:
                    continue
            folders_to_delete = safe_folders
        except Exception:
            folders_to_delete = []
        
        # 确认对话框
        msg = f'确定要删除当前记录吗？\n\n'
        msg += f'TASK_ID: {pair_to_delete.get("task_id")}\n'
        msg += f'案例类型: {pair_to_delete.get("case_type")}\n'
        if folders_to_delete:
            msg += '\n注意：将同时删除相关图片文件夹:\n' + "\n".join(folders_to_delete)
        
        reply = QMessageBox.question(
            self, '确认删除', msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 删除输出文件夹（如果存在）
        for folder in folders_to_delete:
            self._delete_result_folder(folder)

        # 从列表中删除
        del self.filtered_pairs[self.current_pair_index]

        # 从 suspicious_pairs 中删除对应的记录
        task_id = pair_to_delete.get('task_id')
        curr_path = pair_to_delete.get('curr_path')
        prev_path = pair_to_delete.get('prev_path')

        for i, pair in enumerate(self.suspicious_pairs):
            if (pair.get('task_id') == task_id and
                pair.get('curr_path') == curr_path and
                pair.get('prev_path') == prev_path):
                del self.suspicious_pairs[i]
                break

        # 同步更新CSV文件
        self._update_csv_after_deletion(pair_to_delete)

        # 更新索引和显示
        if not self.filtered_pairs:
            self.current_pair_index = -1
            self._clear_image_displays()
            self.result_text.setText('已删除所有记录')
            self.prev_pair_btn.setEnabled(False)
            self.next_pair_btn.setEnabled(False)
            self.delete_pair_btn.setEnabled(False)
            self.save_pair_btn.setEnabled(False)
        else:
            if self.current_pair_index >= len(self.filtered_pairs):
                self.current_pair_index = len(self.filtered_pairs) - 1
            elif self.current_pair_index < 0:
                self.current_pair_index = 0
            self.show_current_suspicious_pair()

        QMessageBox.information(
            self, '删除成功',
            f'已删除该记录。\n剩余记录数: {len(self.filtered_pairs)}'
        )

    def _update_csv_after_deletion(self, deleted_pair: dict):
        """删除记录后更新CSV文件"""
        try:
            if not hasattr(self, 'DEFAULT_CSV_PATH') or not os.path.exists(self.DEFAULT_CSV_PATH):
                return
                
            try:
                df_csv = pd.read_csv(self.DEFAULT_CSV_PATH, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df_csv = pd.read_csv(self.DEFAULT_CSV_PATH, encoding='gbk')
            
            task_id = deleted_pair.get('task_id')
            
            # 支持新旧格式的删除
            if 'OUTPUT_FOLDER' in df_csv.columns:
                # 新格式
                output_folder = deleted_pair.get('output_folder')
                if output_folder is None or (isinstance(output_folder, float) and pd.isna(output_folder)):
                    case_type = deleted_pair.get('case_type')
                    mask = ~((df_csv['TASK_ID'].astype(str) == str(task_id)) &
                            (df_csv['CASE_TYPE'].astype(str) == str(case_type)))
                else:
                    mask = ~((df_csv['TASK_ID'].astype(str) == str(task_id)) & 
                            (df_csv['OUTPUT_FOLDER'].astype(str) == str(output_folder)))
            else:
                # 旧格式
                curr_path = deleted_pair.get('curr_path')
                prev_path = deleted_pair.get('prev_path')
                mask = ~((df_csv['TASK_ID'].astype(str) == str(task_id)) & 
                        (df_csv['CURR_IMAGE_PATH'].astype(str) == str(curr_path)) &
                        (df_csv['PREV_IMAGE_PATH'].astype(str) == str(prev_path)))
            
            new_df = df_csv[mask]
            new_df.to_csv(self.DEFAULT_CSV_PATH, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            print(f"更新CSV文件失败: {e}")

    def save_current_originals(self):
        """保存当前记录的原图到用户选择的文件夹，支持新旧格式"""
        if not self.filtered_pairs or self.current_pair_index < 0:
            QMessageBox.information(self, '提示', '当前没有可保存的记录。')
            return
            
        pair = self.filtered_pairs[self.current_pair_index]
        
        # 根据格式获取图片路径
        if pair.get('format_type') == 'new' and pair.get('image_paths'):
            curr_path = pair.get('image_paths', {}).get('original_curr')
            prev_path = pair.get('image_paths', {}).get('original_prev')
        else:
            curr_path = pair.get('curr_path')
            prev_path = pair.get('prev_path')
        
        if (not curr_path or not os.path.exists(curr_path)) and (not prev_path or not os.path.exists(prev_path)):
            QMessageBox.warning(self, '警告', '当前记录的原图路径无效，无法保存。')
            return

        target_dir = QFileDialog.getExistingDirectory(self, '选择保存文件夹')
        if not target_dir:
            return

        saved = []
        errors = []

        def _save_one(src_path: str, suffix: str):
            try:
                if not src_path or not os.path.exists(src_path):
                    return
                base = os.path.basename(src_path)
                name, ext = os.path.splitext(base)
                task_id = pair.get('task_id')
                case_type = str(pair.get('case_type') or 'unknown')
                safe_case = re.sub(r'[^\w\-]+', '_', case_type)
                out_name = f"{task_id}_{safe_case}_{suffix}{ext}"
                out_path = os.path.join(target_dir, out_name)
                
                if os.path.exists(out_path):
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    out_name = f"{task_id}_{safe_case}_{suffix}_{ts}{ext}"
                    out_path = os.path.join(target_dir, out_name)
                
                shutil.copy2(src_path, out_path)
                saved.append(out_name)
            except Exception as e:
                errors.append(f"{suffix}: {e}")

        _save_one(curr_path, "curr")
        _save_one(prev_path, "prev")

        if saved:
            msg = f"成功保存 {len(saved)} 个文件到:\n{target_dir}\n\n文件:\n" + "\n".join(saved)
            if errors:
                msg += f"\n\n错误:\n" + "\n".join(errors)
            QMessageBox.information(self, '保存完成', msg)
        else:
            msg = "没有文件被保存"
            if errors:
                msg += f"\n\n错误:\n" + "\n".join(errors)
            QMessageBox.warning(self, '保存失败', msg)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CarPlateRecognitionGUI()
    window.show()
    sys.exit(app.exec())