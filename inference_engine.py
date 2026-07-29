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

    def infer(self, image, orig_size=None):
        """
        执行推理

        Args:
            image: BGR 格式图片 (numpy array)
            orig_size: (orig_h, orig_w) 原始分辨率，None 时用 image.shape[:2]

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

    def _get_adaptive_recog_area(self, orig_size):
        """获取自适应检测区域

        Args:
            orig_size: (orig_h, orig_w) 原始分辨率元组
        """
        orig_h, orig_w = orig_size
        current_resolution = (orig_w, orig_h)
        if self.recog_area:
            return get_adaptive_recog_area(self.recog_area, self.CONFIG_RESOLUTION, current_resolution)
        return [0, 0, orig_w, orig_h]

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

    def infer(self, image, orig_size=None):
        """执行推理（检测 + 分类二级串联）

        Args:
            image: BGR numpy 数组
            orig_size: (orig_h, orig_w) 原始分辨率，None 时用 image.shape[:2]
        """
        t0 = time.time()

        # 获取原始分辨率（DVPP 硬解时 image 已是 640x640，需用 orig_size）
        if orig_size is not None:
            orig_h, orig_w = orig_size
        else:
            orig_h, orig_w = image.shape[:2]

        # 自适应检测区域
        recog_area = self._get_adaptive_recog_area((orig_h, orig_w))
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
    """Ascend NPU 推理引擎（检测 + 分类二级串联）"""

    def __init__(self, model_path, conf_threshold=0.5, recog_area=None, names=None, device_id=0,
                 cls_model_path=None, cls_conf_threshold=0.5, aipp=False):
        """
        初始化 Ascend 推理引擎

        Args:
            model_path: OM 模型路径
            conf_threshold: 置信度阈值
            recog_area: 检测区域
            names: 类别名称列表
            device_id: Ascend 设备 ID
            cls_model_path: 分类模型路径 (.om)，二级串联分类
            cls_conf_threshold: 分类置信度阈值
            aipp: True=模型已插入 AIPP，输入为 NV12 device buffer（零拷贝）
        """
        super().__init__(model_path, conf_threshold, recog_area, names)
        self.device_id = device_id
        self.aipp = aipp
        self.cls_model_path = cls_model_path
        self.cls_conf_threshold = cls_conf_threshold
        self.cls_session = None
        self.cls_names = []
        self.cls_input_size = 224
        self._init_model()
        if self.cls_model_path:
            self._init_cls_model()

    def _init_model(self):
        """初始化模型"""
        try:
            if self.aipp:
                import acl
                ret = acl.init()
                if ret != 0 and ret != 1 and ret != 100002:
                    raise RuntimeError(f"acl.init 失败: ret={ret}")
                acl.rt.set_device(self.device_id)
                self._model_id, ret = acl.mdl.load_from_file(self.model_path)
                if ret != 0:
                    raise RuntimeError(f"acl.mdl.load_from_file 失败: ret={ret}")
                self._model_desc = acl.mdl.create_desc()
                acl.mdl.get_desc(self._model_desc, self._model_id)

                output_count = acl.mdl.get_num_outputs(self._model_desc)
                self._output_dataset = acl.mdl.create_dataset()
                self._output_sizes = []
                self._output_dev_bufs = []
                for i in range(output_count):
                    buf_size = acl.mdl.get_output_size_by_index(self._model_desc, i)
                    self._output_sizes.append(buf_size)
                    dev_buf, ret = acl.rt.malloc(buf_size, 2)
                    if ret != 0:
                        raise RuntimeError(f"输出 buffer malloc 失败: ret={ret}")
                    self._output_dev_bufs.append(dev_buf)
                    data_buf = acl.create_data_buffer(dev_buf, buf_size)
                    acl.mdl.add_dataset_buffer(self._output_dataset, data_buf)

                # 获取输出 shape（兼容不同 ACL 版本）
                self._output_dims = []
                for i in range(output_count):
                    try:
                        dims, ret = acl.mdl.get_cur_output_dims(self._model_desc, i)
                        if ret == 0 and isinstance(dims, dict) and 'dims' in dims:
                            self._output_dims.append(dims['dims'])
                        else:
                            raise TypeError("ret != 0")
                    except (AttributeError, TypeError):
                        # 回退：从 output_size 和 float32 推算
                        num_floats = self._output_sizes[i] // 4
                        self._output_dims.append([1, num_floats])

                # 读取模型输入信息（兼容不同 ACL 版本）
                # AIPP 模型输入是展平的 NV12: size = H * W * 1.5
                # dims 可能是 [1, H*W*1.5]（一维），不能直接取 d[1]d[2] 作为 H/W
                input_count = acl.mdl.get_num_inputs(self._model_desc)
                self._expected_input_size = 0
                self._model_input_h = 640
                self._model_input_w = 640
                for i in range(input_count):
                    in_size = acl.mdl.get_input_size_by_index(self._model_desc, i)
                    self._expected_input_size = in_size
                    # 从 input_size 反推 NV12 分辨率: size = H * W * 1.5
                    # 假设正方形输入: H * H * 1.5 = size → H = sqrt(size / 1.5)
                    import math
                    if in_size > 0:
                        est_h = int(round(math.sqrt(in_size / 1.5)))
                        for candidate in [640, 416, 320, 224, 1280]:
                            if abs(est_h - candidate) <= 2:
                                est_h = candidate
                                break
                        self._model_input_h = est_h
                        self._model_input_w = est_h
                    # 仅用于日志，不依赖 dims 解析 H/W
                    try:
                        dims, ret = acl.mdl.get_cur_input_dims(self._model_desc, i)
                        logger.info(f"[AIPP] 模型输入[{i}]: size={in_size}, dims={dims['dims'] if ret == 0 and isinstance(dims, dict) else 'N/A'}")
                    except (AttributeError, TypeError):
                        logger.info(f"[AIPP] 模型输入[{i}]: size={in_size}, 估算维度: {self._model_input_h}x{self._model_input_w}")

                logger.info(f"[AscendInferenceEngine/AIPP] 零拷贝模型加载成功: {self.model_path} | "
                            f"设备: {self.device_id} | 输入: {self._model_input_h}x{self._model_input_w} "
                            f"(size={self._expected_input_size})")
            else:
                from ais_bench.infer.interface import InferSession
                self.session = InferSession(self.device_id, self.model_path)
                self._input_buffer = np.zeros((1, 3, 640, 640), dtype=np.float32)
                logger.info(f"[AscendInferenceEngine] 模型加载成功: {self.model_path} | 设备: {self.device_id}")
        except ImportError:
            raise RuntimeError("[AscendInferenceEngine] ais_bench 未安装，无法使用 Ascend 模式")
        except Exception as e:
            raise RuntimeError(f"[AscendInferenceEngine] 模型加载失败: {e}")

    def _init_cls_model(self):
        """初始化分类模型（二级串联）

        AIPP 模式下用 acl.mdl.execute 原生推理，预分配输入输出 buffer，
        避免 ais_bench 运行时临时 malloc device memory 导致显存不足。
        非 AIPP 模式继续用 ais_bench。
        """
        try:
            cls_path = self.cls_model_path
            if not os.path.isabs(cls_path):
                cls_path = os.path.join(os.path.dirname(self.model_path), cls_path)

            if not os.path.exists(cls_path):
                logger.warning(f"[AscendInferenceEngine] 分类模型不存在，跳过: {cls_path}")
                self.cls_model_path = None
                return

            # 从模型文件名推断输入尺寸
            if '224' in os.path.basename(cls_path):
                self.cls_input_size = 224
            elif '416' in os.path.basename(cls_path):
                self.cls_input_size = 416

            # 获取分类模型的类别名称（从 classes_cls.txt 或默认）
            cls_names_file = os.path.join(os.path.dirname(cls_path), 'classes_cls.txt')
            if os.path.exists(cls_names_file):
                with open(cls_names_file, 'r', encoding='utf-8') as f:
                    self.cls_names = [line.strip() for line in f if line.strip()]
            else:
                self.cls_names = [f"sign_{i}" for i in range(24)]

            if self.aipp:
                # AIPP 模式：用原生 acl API 加载分类模型，预分配 buffer
                import acl
                acl.rt.set_device(self.device_id)
                self._cls_model_id, ret = acl.mdl.load_from_file(cls_path)
                if ret != 0:
                    raise RuntimeError(f"分类模型加载失败: ret={ret}")
                self._cls_model_desc = acl.mdl.create_desc()
                acl.mdl.get_desc(self._cls_model_desc, self._cls_model_id)

                # 预分配输入 buffer（float32 NCHW，分类模型无 AIPP）
                cls_input_size = acl.mdl.get_input_size_by_index(self._cls_model_desc, 0)
                self._cls_input_dev, ret = acl.rt.malloc(cls_input_size, 2)
                if ret != 0:
                    raise RuntimeError(f"分类输入 buffer malloc 失败: ret={ret}")
                self._cls_input_size = cls_input_size
                self._cls_input_dataset = acl.mdl.create_dataset()
                cls_input_buf = acl.create_data_buffer(self._cls_input_dev, cls_input_size)
                acl.mdl.add_dataset_buffer(self._cls_input_dataset, cls_input_buf)

                # 预分配输出 dataset
                cls_output_count = acl.mdl.get_num_outputs(self._cls_model_desc)
                self._cls_output_dataset = acl.mdl.create_dataset()
                self._cls_output_sizes = []
                self._cls_output_dev_bufs = []
                self._cls_output_dims = []
                for i in range(cls_output_count):
                    buf_size = acl.mdl.get_output_size_by_index(self._cls_model_desc, i)
                    self._cls_output_sizes.append(buf_size)
                    dev_buf, ret = acl.rt.malloc(buf_size, 2)
                    if ret != 0:
                        raise RuntimeError(f"分类输出 buffer malloc 失败: ret={ret}")
                    self._cls_output_dev_bufs.append(dev_buf)
                    data_buf = acl.create_data_buffer(dev_buf, buf_size)
                    acl.mdl.add_dataset_buffer(self._cls_output_dataset, data_buf)
                    # 获取输出 dims
                    try:
                        dims, r = acl.mdl.get_cur_output_dims(self._cls_model_desc, i)
                        if r == 0 and isinstance(dims, dict) and 'dims' in dims:
                            self._cls_output_dims.append(dims['dims'])
                        else:
                            raise TypeError("ret != 0")
                    except (AttributeError, TypeError):
                        self._cls_output_dims.append([1, buf_size // 4])

                # 预分配 host 输入 buffer（CPU 预处理写入，H2D 拷贝到 device）
                self._cls_input_buffer = np.zeros((1, 3, self.cls_input_size, self.cls_input_size), dtype=np.float32)

                logger.info(f"[AscendInferenceEngine/AIPP] 分类模型加载成功(原生ACL): {cls_path} | "
                            f"类别数: {len(self.cls_names)} | 输入: {self.cls_input_size}x{self.cls_input_size} | "
                            f"输入size: {cls_input_size}")
            else:
                # 非 AIPP：继续用 ais_bench
                from ais_bench.infer.interface import InferSession
                self.cls_session = InferSession(self.device_id, cls_path)
                self.cls_input_buffer = np.zeros((1, 3, self.cls_input_size, self.cls_input_size), dtype=np.float32)

                logger.info(f"[AscendInferenceEngine] 分类模型加载成功(ais_bench): {cls_path} | "
                            f"类别数: {len(self.cls_names)} | 输入: {self.cls_input_size}x{self.cls_input_size}")
        except Exception as e:
            logger.warning(f"[AscendInferenceEngine] 分类模型加载失败，将仅使用检测模型: {e}")
            self.cls_session = None
            self.cls_model_path = None

    def _classify_crop(self, crop_img):
        """对裁剪区域执行分类推理

        AIPP 模式下用预分配的 acl buffer 推理，避免运行时 malloc device memory。
        非 AIPP 模式用 ais_bench。

        Args:
            crop_img: BGR 裁剪图片 (numpy array)

        Returns:
            (cls_name, cls_conf) 或 None
        """
        if self.aipp and hasattr(self, '_cls_model_id'):
            return self._classify_crop_acl(crop_img)

        if self.cls_session is None:
            return None

        try:
            img_resized = cv2.resize(crop_img, (self.cls_input_size, self.cls_input_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) * 0.00392156862745098
            self.cls_input_buffer[0] = img_float.transpose(2, 0, 1)

            outputs = self.cls_session.infer([self.cls_input_buffer])

            if outputs and len(outputs) > 0:
                probs = np.asarray(outputs[0])
                if probs.ndim == 3:
                    probs = probs[0]
                if probs.ndim == 2:
                    probs = probs[0]
                cls_idx = int(np.argmax(probs))
                cls_conf = float(probs[cls_idx])

                if cls_idx < len(self.cls_names):
                    return self.cls_names[cls_idx], cls_conf
            return None
        except Exception as e:
            logger.warning(f"[AscendInferenceEngine] 分类推理失败: {e}")
            return None

    def _classify_crop_acl(self, crop_img):
        """AIPP 模式下分类推理：用预分配 acl buffer，零临时 device malloc

        Args:
            crop_img: BGR 裁剪图片 (numpy array)

        Returns:
            (cls_name, cls_conf) 或 None
        """
        import acl
        try:
            # CPU 预处理：resize + BGR→RGB + /255 + NCHW
            img_resized = cv2.resize(crop_img, (self.cls_input_size, self.cls_input_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) * 0.00392156862745098
            self._cls_input_buffer[0] = img_float.transpose(2, 0, 1)

            # H2D：host buffer → 预分配 device input buffer
            input_nbytes = self._cls_input_buffer.nbytes
            acl.rt.memcpy(int(self._cls_input_dev), input_nbytes,
                          int(self._cls_input_buffer.ctypes.data), input_nbytes, 1)

            # 执行推理
            ret = acl.mdl.execute(self._cls_model_id, self._cls_input_dataset, self._cls_output_dataset)
            if ret != 0:
                logger.warning(f"[AIPP-CLS] execute 失败 ret={ret}")
                return None

            # D2H 输出
            buf = acl.mdl.get_dataset_buffer(self._cls_output_dataset, 0)
            out_ptr = acl.get_data_buffer_addr(buf)
            out_size = acl.get_data_buffer_size_v2(buf)
            host_ptr, _ = acl.rt.malloc_host(out_size)
            acl.rt.memcpy(host_ptr, out_size, int(out_ptr), out_size, 2)
            bytes_data = acl.util.ptr_to_bytes(host_ptr, out_size)
            raw = np.frombuffer(bytes_data, dtype=np.float32)
            try:
                probs = raw.reshape(tuple(self._cls_output_dims[0]))
            except Exception:
                probs = raw
            if probs.ndim == 3:
                probs = probs[0]
            if probs.ndim == 2:
                probs = probs[0]
            acl.rt.free_host(host_ptr)

            cls_idx = int(np.argmax(probs))
            cls_conf = float(probs[cls_idx])

            if cls_idx < len(self.cls_names):
                return self.cls_names[cls_idx], cls_conf
            return None
        except Exception as e:
            logger.warning(f"[AscendInferenceEngine] AIPP分类推理失败: {e}")
            return None

    def _execute_zero_copy(self, dev_ptr, dev_size):
        """零拷贝推理：直接用 device NV12 buffer 作为输入"""
        import acl
        acl.rt.set_device(self.device_id)

        # 输入大小校验
        if hasattr(self, '_expected_input_size') and self._expected_input_size > 0:
            if dev_size != self._expected_input_size:
                # AIPP 模型输入 NV12: size = H*W*1.5 (如 640*640*1.5=614400)
                # 非 AIPP 模型输入 float32 NCHW: size = 1*3*H*W*4 (如 1*3*640*640*4=4915200)
                # 如果 expected_input_size 是 dev_size 的 8 倍，说明模型不是 AIPP 模型
                if self._expected_input_size == dev_size * 8:
                    logger.error(f"[AIPP] 模型输入 size={self._expected_input_size} 是 float32 NCHW 格式，"
                                 f"不是 NV12 AIPP 模型！请检查: 1) model_path 是否指向 _aipp.om; "
                                 f"2) ATC 转换时是否加了 --insert_op_conf=aipp.cfg")
                else:
                    logger.warning(f"[AIPP] 输入大小不匹配: dev_size={dev_size}, 期望={self._expected_input_size} "
                                   f"(模型输入: {self._model_input_h}x{self._model_input_w})")
                return None

        input_dataset = acl.mdl.create_dataset()
        input_buf = acl.create_data_buffer(dev_ptr, dev_size)
        _, ret = acl.mdl.add_dataset_buffer(input_dataset, input_buf)
        if ret != 0:
            acl.destroy_data_buffer(input_buf)
            acl.mdl.destroy_dataset(input_dataset)
            return None

        ret = acl.mdl.execute(self._model_id, input_dataset, self._output_dataset)

        num_bufs = acl.mdl.get_dataset_num_buffers(input_dataset)
        for i in range(num_bufs):
            buf = acl.mdl.get_dataset_buffer(input_dataset, i)
            acl.destroy_data_buffer(buf)
        acl.mdl.destroy_dataset(input_dataset)

        if ret != 0:
            return None

        outputs = []
        for i in range(len(self._output_sizes)):
            buf = acl.mdl.get_dataset_buffer(self._output_dataset, i)
            out_ptr = acl.get_data_buffer_addr(buf)
            out_size = acl.get_data_buffer_size_v2(buf)
            host_ptr, _ = acl.rt.malloc_host(out_size)
            acl.rt.memcpy(host_ptr, out_size, int(out_ptr), out_size, 2)
            bytes_data = acl.util.ptr_to_bytes(host_ptr, out_size)
            raw = np.frombuffer(bytes_data, dtype=np.float32)
            try:
                result = raw.reshape(tuple(self._output_dims[i]))
            except Exception:
                result = raw
            if result.ndim == 2:
                result = result[np.newaxis, :]
            outputs.append(result)
            acl.rt.free_host(host_ptr)
        return outputs

    def cleanup(self):
        """释放 AIPP 模式的 acl 资源"""
        if not self.aipp:
            return
        try:
            import acl
            if hasattr(self, '_output_dataset'):
                num = acl.mdl.get_dataset_num_buffers(self._output_dataset)
                for i in range(num):
                    buf = acl.mdl.get_dataset_buffer(self._output_dataset, i)
                    acl.destroy_data_buffer(buf)
                acl.mdl.destroy_dataset(self._output_dataset)
            for dev_buf in getattr(self, '_output_dev_bufs', []):
                acl.rt.free(dev_buf)
            if hasattr(self, '_model_id'):
                acl.mdl.unload(self._model_id)
            if hasattr(self, '_model_desc'):
                acl.mdl.destroy_desc(self._model_desc)
        except Exception as e:
            logger.warning(f"[AscendInferenceEngine] cleanup 警告: {e}")

    def _preprocess(self, image):
        """预处理：整图 resize 到 640x640 + 归一化"""
        img_640 = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) * 0.00392156862745098  # 1/255
        self._input_buffer[0] = img_float.transpose(2, 0, 1)
        return self._input_buffer

    def infer(self, image, orig_size=None, bgr_image=None):
        """执行推理（检测 + 分类二级串联）

        Args:
            image: BGR numpy 数组，或 AIPP 模式下的 device buffer dict
                   AIPP dict: {'buffer': dev_nv12_ptr, 'size': int}
            orig_size: (orig_h, orig_w) 原始分辨率，None 时用 image.shape[:2]
            bgr_image: AIPP 模式下传入的 BGR 原图（用于分类裁剪），
                       None 时非 AIPP 模式用 image 作为 BGR 原图
        """
        t0 = time.time()

        # AIPP 零拷贝路径
        if self.aipp and isinstance(image, dict):
            if orig_size is None:
                orig_h, orig_w = 1080, 1920
            else:
                orig_h, orig_w = orig_size

            recog_area = self._get_adaptive_recog_area((orig_h, orig_w))
            outputs = self._execute_zero_copy(image['buffer'], image['size'])
            if outputs is None:
                return []
            boxes = self._postprocess(outputs[0], orig_w, orig_h, recog_area)

            # 第二级：分类（用 bgr_image 裁剪检测区域）
            if self.cls_model_path and boxes and bgr_image is not None:
                boxes = self._classify_boxes(boxes, bgr_image, orig_h, orig_w)

            self._infer_count += 1
            return boxes

        # 获取原始分辨率（DVPP 硬解时 image 已是 640x640，需用 orig_size）
        if orig_size is not None:
            orig_h, orig_w = orig_size
        else:
            orig_h, orig_w = image.shape[:2]

        # 自适应检测区域
        recog_area = self._get_adaptive_recog_area((orig_h, orig_w))

        # 预处理
        input_data = self._preprocess(image)

        # 推理
        outputs = self.session.infer([input_data])

        # 后处理
        boxes = self._postprocess(outputs[0], orig_w, orig_h, recog_area)

        # 第二级：分类（用原始 image 裁剪检测区域）
        if self.cls_model_path and boxes:
            boxes = self._classify_boxes(boxes, image, orig_h, orig_w)

        self._infer_count += 1
        if self._infer_count % 1000 == 0:
            elapsed = (time.time() - t0) * 1000
            logger.info(f"[AscendInferenceEngine] 推理统计: {self._infer_count}帧 | 耗时: {elapsed:.1f}ms | 检测: {len(boxes)}个")

        return boxes

    def _classify_boxes(self, boxes, bgr_image, orig_h, orig_w):
        """对检测框执行二级分类

        Args:
            boxes: 检测结果列表
            bgr_image: BGR 原图（用于裁剪）
            orig_h, orig_w: 原始分辨率

        Returns:
            更新标签后的检测结果列表
        """
        img_h, img_w = bgr_image.shape[:2]
        # 检测框坐标是 orig_w x orig_h，bgr_image 可能是 640x640（AIPP）或原图尺寸
        scale_x = img_w / orig_w
        scale_y = img_h / orig_h

        for box in boxes:
            # 将坐标从 orig 分辨率缩放到 bgr_image 实际尺寸
            x1 = max(0, int((box['x'] - 5) * scale_x))
            y1 = max(0, int((box['y'] - 5) * scale_y))
            x2 = min(img_w, int((box['x'] + box['width'] + 5) * scale_x))
            y2 = min(img_h, int((box['y'] + box['height'] + 5) * scale_y))

            crop_img = bgr_image[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            cls_result = self._classify_crop(crop_img)
            if cls_result is not None:
                cls_name, cls_conf = cls_result
                if cls_conf >= self.cls_conf_threshold:
                    box['label'] = cls_name
                    box['confidence'] = cls_conf

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
        if self.aipp:
            self.cleanup()
            # 释放 AIPP 分类模型资源
            self._cleanup_cls_model()
        else:
            if self.session is not None:
                del self.session
                self.session = None
            if self.cls_session is not None:
                del self.cls_session
                self.cls_session = None
        logger.info("[AscendInferenceEngine] 资源已释放")

    def _cleanup_cls_model(self):
        """释放 AIPP 分类模型的 acl 资源"""
        if not self.aipp:
            return
        try:
            import acl
            if hasattr(self, '_cls_output_dataset'):
                num = acl.mdl.get_dataset_num_buffers(self._cls_output_dataset)
                for i in range(num):
                    buf = acl.mdl.get_dataset_buffer(self._cls_output_dataset, i)
                    acl.destroy_data_buffer(buf)
                acl.mdl.destroy_dataset(self._cls_output_dataset)
            for dev_buf in getattr(self, '_cls_output_dev_bufs', []):
                acl.rt.free(dev_buf)
            if hasattr(self, '_cls_input_dev'):
                acl.rt.free(self._cls_input_dev)
            if hasattr(self, '_cls_input_dataset'):
                num = acl.mdl.get_dataset_num_buffers(self._cls_input_dataset)
                for i in range(num):
                    buf = acl.mdl.get_dataset_buffer(self._cls_input_dataset, i)
                    acl.destroy_data_buffer(buf)
                acl.mdl.destroy_dataset(self._cls_input_dataset)
            if hasattr(self, '_cls_model_id'):
                acl.mdl.unload(self._cls_model_id)
            if hasattr(self, '_cls_model_desc'):
                acl.mdl.destroy_desc(self._cls_model_desc)
        except Exception as e:
            logger.warning(f"[AscendInferenceEngine] 分类模型 cleanup 警告: {e}")


# ===================== 工厂函数 =====================

def create_inference_engine(backend, model_path, conf_threshold=0.5, recog_area=None, names=None, device_id=0,
                           cls_model_path=None, cls_conf_threshold=0.5, aipp=False):
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
        aipp: True=Ascend 模型已插入 AIPP，零拷贝推理

    Returns:
        推理引擎实例
    """
    backend_lower = backend.lower()
    if backend_lower == 'ascend':
        return AscendInferenceEngine(model_path, conf_threshold, recog_area, names, device_id,
                                     cls_model_path=cls_model_path, cls_conf_threshold=cls_conf_threshold, aipp=aipp)
    else:
        return CUDAInferenceEngine(model_path, conf_threshold, recog_area, names, device='cuda',
                                   cls_model_path=cls_model_path, cls_conf_threshold=cls_conf_threshold)
