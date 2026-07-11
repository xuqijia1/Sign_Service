#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理引擎模块 - Sign_Service 标志牌检测服务
参照 ChuanDaiJianCa 的处理方式：
- 检测区域用于过滤检测框，而非裁剪图片
- 整图 resize 到 640x640 进行推理
- 检测区域根据图片分辨率自适应缩放
- 使用中心点判断检测框是否在检测区域内
"""

import cv2
import numpy as np
import time
import os
import logging

logger = logging.getLogger(__name__)

# ===================== 工具函数 =====================

def get_adaptive_recog_area(recog_area_config, config_resolution, current_resolution):
    """
    根据当前分辨率自适应检测区域

    Args:
        recog_area_config: 配置的检测区域 [x1, y1, x2, y2]
        config_resolution: 配置的分辨率 (width, height)
        current_resolution: 当前图片分辨率 (width, height)

    Returns:
        自适应后的检测区域 [x1, y1, x2, y2]
    """
    config_w, config_h = config_resolution
    current_w, current_h = current_resolution

    scale_x = current_w / config_w
    scale_y = current_h / config_h

    x1 = int(recog_area_config[0] * scale_x)
    y1 = int(recog_area_config[1] * scale_y)
    x2 = int(recog_area_config[2] * scale_x)
    y2 = int(recog_area_config[3] * scale_y)

    return [x1, y1, x2, y2]


def nms(boxes, scores, iou_threshold=0.45):
    """
    非极大值抑制 (NMS)

    Args:
        boxes: 检测框数组 (N, 4) - [x1, y1, x2, y2]
        scores: 置信度数组 (N,)
        iou_threshold: IoU 阈值

    Returns:
        保留的索引列表
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # 计算 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def is_box_in_recog_area(x1, y1, x2, y2, recog_area):
    """
    判断检测框中心点是否在检测区域内

    Args:
        x1, y1, x2, y2: 检测框坐标
        recog_area: 检测区域 [x1, y1, x2, y2]

    Returns:
        True 如果中心点在检测区域内
    """
    recog_x1, recog_y1, recog_x2, recog_y2 = recog_area
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return not (center_x < recog_x1 or center_y < recog_y1 or center_x > recog_x2 or center_y > recog_y2)


def load_class_names(classes_file=None):
    """
    加载类别名称

    Args:
        classes_file: 类别文件路径，默认为当前目录下的 classes.txt

    Returns:
        类别名称列表
    """
    if classes_file is None:
        classes_file = os.path.join(os.path.dirname(__file__), 'classes.txt')

    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return []


# ===================== 推理引擎基类 =====================

class BaseInferenceEngine:
    """推理引擎基类"""

    # 配置分辨率（检测区域配置基于此分辨率）
    CONFIG_RESOLUTION = (1920, 1080)

    def __init__(self, model_path, conf_threshold=0.5, recog_area=None, names=None):
        """
        初始化推理引擎

        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            recog_area: 检测区域 [x1, y1, x2, y2]（基于配置分辨率）
            names: 类别名称列表
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.recog_area = recog_area
        self.names = names or load_class_names()
        self._infer_count = 0

    def infer(self, image):
        """
        执行推理

        Args:
            image: BGR 格式图片 (numpy array)

        Returns:
            list: 检测框列表，每个元素为字典:
                {
                    'x': int,          # 原图 x 坐标
                    'y': int,          # 原图 y 坐标
                    'width': int,      # 宽度
                    'height': int,     # 高度
                    'label': str,      # 类别名称
                    'confidence': float # 置信度
                }
        """
        raise NotImplementedError

    def set_recog_area(self, recog_area):
        """设置检测区域"""
        self.recog_area = recog_area

    def set_conf_threshold(self, conf_threshold):
        """设置置信度阈值"""
        self.conf_threshold = conf_threshold

    def _get_adaptive_recog_area(self, image_shape):
        """获取自适应检测区域"""
        h, w = image_shape[:2]
        current_resolution = (w, h)
        if self.recog_area:
            return get_adaptive_recog_area(self.recog_area, self.CONFIG_RESOLUTION, current_resolution)
        return [0, 0, w, h]

    def _get_label(self, class_id):
        """获取类别名称"""
        if class_id < len(self.names):
            return self.names[class_id]
        return f"class_{class_id}"

    def release(self):
        """释放资源"""
        pass


# ===================== CUDA 推理引擎 =====================

class CUDAInferenceEngine(BaseInferenceEngine):
    """CUDA/CPU 推理引擎（使用 YOLOv8 检测 + YOLOv8s-cls 分类 二级串联方案）"""

    def __init__(self, model_path, conf_threshold=0.5, recog_area=None, names=None, device='cuda',
                 cls_model_path=None, cls_conf_threshold=0.5):
        """
        初始化 CUDA 推理引擎

        Args:
            model_path: YOLOv8 检测模型路径 (.pt)
            conf_threshold: 检测置信度阈值
            recog_area: 检测区域
            names: 类别名称列表
            device: 设备 ('cuda' 或 'cpu')
            cls_model_path: YOLOv8s-cls 分类模型路径 (.pt)，二级串联分类
            cls_conf_threshold: 分类置信度阈值
        """
        super().__init__(model_path, conf_threshold, recog_area, names)
        self.device = device
        self.cls_model_path = cls_model_path
        self.cls_conf_threshold = cls_conf_threshold
        self.cls_model = None
        self.cls_names = []
        self._init_model()
        if self.cls_model_path:
            self._init_cls_model()

    def _init_model(self):
        """初始化模型"""
        try:
            import torch
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)

            # 检测 CUDA 可用性
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning(f"[CUDAInferenceEngine] CUDA 不可用，切换到 CPU")
                self.device = 'cpu'

            # 预热模型
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=640, conf=self.conf_threshold, device=self.device, verbose=False)

            # 如果未提供 names，使用模型的 names
            if not self.names and hasattr(self.model, 'names'):
                self.names = [self.model.names[i] for i in sorted(self.model.names.keys())]

            logger.info(f"[CUDAInferenceEngine] 模型加载成功: {self.model_path} | 设备: {self.device}")
        except Exception as e:
            raise RuntimeError(f"[CUDAInferenceEngine] 模型加载失败: {e}")

    def _init_cls_model(self):
        """初始化分类模型（二级串联）"""
        try:
            from ultralytics import YOLO

            cls_path = self.cls_model_path
            if not os.path.isabs(cls_path):
                cls_path = os.path.join(os.path.dirname(self.model_path), cls_path)

            if not os.path.exists(cls_path):
                logger.warning(f"[CUDAInferenceEngine] 分类模型不存在，跳过: {cls_path}")
                self.cls_model_path = None
                return

            self.cls_model = YOLO(cls_path)

            # 预热
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            self.cls_model.predict(dummy, device=self.device, verbose=False)

            # 获取分类模型的类别名称
            if hasattr(self.cls_model, 'names'):
                self.cls_names = [self.cls_model.names[i] for i in sorted(self.cls_model.names.keys())]

            logger.info(f"[CUDAInferenceEngine] 分类模型加载成功: {cls_path} | 类别数: {len(self.cls_names)} | 设备: {self.device}")
        except Exception as e:
            logger.warning(f"[CUDAInferenceEngine] 分类模型加载失败，将仅使用检测模型: {e}")
            self.cls_model = None
            self.cls_model_path = None

    def infer(self, image):
        """执行推理（检测 + 分类二级串联）"""
        t0 = time.time()

        # 获取图片尺寸
        orig_h, orig_w = image.shape[:2]

        # 自适应检测区域
        recog_area = self._get_adaptive_recog_area(image.shape)
        recog_x1, recog_y1, recog_x2, recog_y2 = recog_area

        # 整图 resize 到 640x640
        img_640 = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)

        # 第一级：检测
        results = self.model.predict(img_640, imgsz=640, conf=self.conf_threshold, device=self.device, verbose=False)

        # 解析检测结果
        boxes = []
        if results and len(results) > 0 and results[0].boxes is not None:
            det_boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy()

            # 坐标还原比例
            scale_x = orig_w / 640
            scale_y = orig_h / 640

            for box, conf, label in zip(det_boxes, confs, labels):
                bx1, by1, bx2, by2 = map(int, box)

                # 坐标还原：从 640x640 到原图
                orig_x1 = int(bx1 * scale_x)
                orig_y1 = int(by1 * scale_y)
                orig_x2 = int(bx2 * scale_x)
                orig_y2 = int(by2 * scale_y)

                # 检测区域过滤（中心点判断）
                center_x = (orig_x1 + orig_x2) / 2
                center_y = (orig_y1 + orig_y2) / 2
                if center_x < recog_x1 or center_y < recog_y1 or center_x > recog_x2 or center_y > recog_y2:
                    continue

                # 第二级：分类（裁剪检测区域送入分类模型）
                final_label = self._get_label(int(label))
                final_conf = float(conf)

                if self.cls_model is not None:
                    # 裁剪检测区域，加适当padding
                    pad = 5
                    crop_x1 = max(0, orig_x1 - pad)
                    crop_y1 = max(0, orig_y1 - pad)
                    crop_x2 = min(orig_w, orig_x2 + pad)
                    crop_y2 = min(orig_h, orig_y2 + pad)

                    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop_img.size > 0:
                        cls_result = self.cls_model.predict(crop_img, device=self.device, verbose=False)
                        if cls_result and len(cls_result) > 0 and cls_result[0].probs is not None:
                            probs = cls_result[0].probs.data.cpu().numpy()
                            cls_idx = int(np.argmax(probs))
                            cls_conf = float(probs[cls_idx])

                            if cls_conf >= self.cls_conf_threshold and cls_idx < len(self.cls_names):
                                final_label = self.cls_names[cls_idx]
                                final_conf = cls_conf

                boxes.append({
                    'x': orig_x1,
                    'y': orig_y1,
                    'width': orig_x2 - orig_x1,
                    'height': orig_y2 - orig_y1,
                    'label': final_label,
                    'confidence': final_conf
                })

        self._infer_count += 1
        if self._infer_count % 1000 == 0:
            elapsed = (time.time() - t0) * 1000
            logger.info(f"[CUDAInferenceEngine] 推理统计: {self._infer_count}帧 | 耗时: {elapsed:.1f}ms | 检测: {len(boxes)}个")

        return boxes


# ===================== Ascend 推理引擎 =====================

class AscendInferenceEngine(BaseInferenceEngine):
    """Ascend NPU 推理引擎"""

    def __init__(self, model_path, conf_threshold=0.5, recog_area=None, names=None, device_id=0):
        """
        初始化 Ascend 推理引擎

        Args:
            model_path: OM 模型路径
            conf_threshold: 置信度阈值
            recog_area: 检测区域
            names: 类别名称列表
            device_id: Ascend 设备 ID
        """
        super().__init__(model_path, conf_threshold, recog_area, names)
        self.device_id = device_id
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            from ais_bench.infer.interface import InferSession

            self.session = InferSession(self.device_id, self.model_path)

            # 预分配输入缓冲区
            self._input_buffer = np.zeros((1, 3, 640, 640), dtype=np.float32)

            logger.info(f"[AscendInferenceEngine] 模型加载成功: {self.model_path} | 设备: {self.device_id}")
        except ImportError:
            raise RuntimeError("[AscendInferenceEngine] ais_bench 未安装，无法使用 Ascend 模式")
        except Exception as e:
            raise RuntimeError(f"[AscendInferenceEngine] 模型加载失败: {e}")

    def _preprocess(self, image):
        """预处理：整图 resize 到 640x640 + 归一化"""
        img_640 = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) * 0.00392156862745098  # 1/255
        self._input_buffer[0] = img_float.transpose(2, 0, 1)
        return self._input_buffer

    def infer(self, image):
        """执行推理"""
        t0 = time.time()

        # 获取图片尺寸
        orig_h, orig_w = image.shape[:2]

        # 自适应检测区域
        recog_area = self._get_adaptive_recog_area(image.shape)
        recog_x1, recog_y1, recog_x2, recog_y2 = recog_area

        # 预处理
        input_data = self._preprocess(image)

        # 推理
        outputs = self.session.infer([input_data])

        # 后处理
        boxes = self._postprocess(outputs[0], orig_w, orig_h, recog_area)

        self._infer_count += 1
        if self._infer_count % 1000 == 0:
            elapsed = (time.time() - t0) * 1000
            logger.info(f"[AscendInferenceEngine] 推理统计: {self._infer_count}帧 | 耗时: {elapsed:.1f}ms | 检测: {len(boxes)}个")

        return boxes

    def _postprocess(self, output, orig_w, orig_h, recog_area):
        """后处理：坐标还原 + NMS + 检测区域过滤"""
        predictions = output[0]

        # YOLOv8 输出格式: (num_classes+4, num_boxes) 或 (batch, num_boxes, num_classes+4)
        if len(predictions.shape) == 3:
            predictions = predictions[0]

        # 判断是否需要转置
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose(1, 0)

        det_boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        # sigmoid（如果需要）
        if scores.max() > 1.0:
            scores = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # 阈值过滤
        mask = max_scores >= self.conf_threshold
        if not np.any(mask):
            return []

        filtered_boxes = det_boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]

        # xywh -> xyxy（在 640x640 坐标系）
        cx, cy, bw, bh = filtered_boxes.T
        x1_640 = cx - bw / 2
        y1_640 = cy - bh / 2
        x2_640 = cx + bw / 2
        y2_640 = cy + bh / 2

        # 坐标从 640x640 还原到原图
        x1_orig = x1_640 * orig_w / 640
        y1_orig = y1_640 * orig_h / 640
        x2_orig = x2_640 * orig_w / 640
        y2_orig = y2_640 * orig_h / 640

        # NMS：按类别分组
        unique_classes = np.unique(filtered_classes)
        keep_indices = []

        for cls_id in unique_classes:
            cls_mask = filtered_classes == cls_id
            cls_boxes = np.column_stack((x1_orig[cls_mask], y1_orig[cls_mask], x2_orig[cls_mask], y2_orig[cls_mask]))
            cls_scores = filtered_scores[cls_mask]

            cls_keep = nms(cls_boxes, cls_scores, iou_threshold=0.45)
            global_indices = np.where(cls_mask)[0][cls_keep]
            keep_indices.extend(global_indices)

        if not keep_indices:
            return []

        # 构建结果
        recog_x1, recog_y1, recog_x2, recog_y2 = recog_area
        boxes = []

        for idx in keep_indices:
            orig_x1 = int(x1_orig[idx])
            orig_y1 = int(y1_orig[idx])
            orig_x2 = int(x2_orig[idx])
            orig_y2 = int(y2_orig[idx])

            # 检测区域过滤（中心点判断）
            center_x = (orig_x1 + orig_x2) / 2
            center_y = (orig_y1 + orig_y2) / 2
            if center_x < recog_x1 or center_y < recog_y1 or center_x > recog_x2 or center_y > recog_y2:
                continue

            label_idx = int(filtered_classes[idx])
            label_name = self._get_label(label_idx)

            boxes.append({
                'x': orig_x1,
                'y': orig_y1,
                'width': orig_x2 - orig_x1,
                'height': orig_y2 - orig_y1,
                'label': label_name,
                'confidence': float(filtered_scores[idx])
            })

        return boxes

    def release(self):
        """释放资源"""
        if self.session is not None:
            del self.session
            self.session = None
        logger.info("[AscendInferenceEngine] 资源已释放")


# ===================== 工厂函数 =====================

def create_inference_engine(backend, model_path, conf_threshold=0.5, recog_area=None, names=None, device_id=0,
                           cls_model_path=None, cls_conf_threshold=0.5):
    """
    创建推理引擎

    Args:
        backend: 后端类型 ('cuda'/'gpu' 或 'ascend')
        model_path: 模型路径
        conf_threshold: 置信度阈值
        recog_area: 检测区域
        names: 类别名称列表
        device_id: 设备 ID
        cls_model_path: 分类模型路径（二级串联）
        cls_conf_threshold: 分类置信度阈值

    Returns:
        推理引擎实例
    """
    backend_lower = backend.lower()
    if backend_lower == 'ascend':
        return AscendInferenceEngine(model_path, conf_threshold, recog_area, names, device_id)
    else:
        return CUDAInferenceEngine(model_path, conf_threshold, recog_area, names, device='cuda',
                                   cls_model_path=cls_model_path, cls_conf_threshold=cls_conf_threshold)
