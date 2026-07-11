# SignModel.py - 标志牌检测模型
import os
import cv2
import numpy as np
import logging
from typing import List, Optional

# 导入新的推理引擎模块
from inference_engine import create_inference_engine, load_class_names

logger = logging.getLogger(__name__)

class SignModel:
    """标志牌检测模型封装 - 使用统一推理引擎（检测+分类二级串联）"""

    def __init__(self, model_path: str, device_type: str = "gpu", cuda_device: int = 0,
                 confidence_threshold: float = 0.5, recog_area=None,
                 cls_model_path: str = None, cls_conf_threshold: float = 0.5):
        self.model_path = model_path
        self.device_type = device_type.lower()
        self.cuda_device = cuda_device
        self.confidence_threshold = confidence_threshold
        self.recog_area = recog_area
        self.cls_model_path = cls_model_path
        self.cls_conf_threshold = cls_conf_threshold

        # 加载类别名称
        self.class_names = load_class_names()
        logger.info(f"加载类别名称: {len(self.class_names)} 个")

        # 确定模型路径
        if not os.path.exists(self.model_path):
            self.model_path = os.path.join(os.path.dirname(__file__), self.model_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # Ascend 模式需要 .om 模型
        if self.device_type == "ascend":
            if not self.model_path.endswith('.om'):
                om_path = self.model_path.replace('.pt', '.om').replace('.onnx', '.om')
                if os.path.exists(om_path):
                    self.model_path = om_path
                    logger.info(f"使用昇腾模型: {om_path}")
                else:
                    raise FileNotFoundError(f"昇腾NPU需要.om模型，未找到: {om_path}")

        # 创建推理引擎
        try:
            self.engine = create_inference_engine(
                backend=self.device_type,
                model_path=self.model_path,
                conf_threshold=self.confidence_threshold,
                recog_area=self.recog_area,
                names=self.class_names,
                device_id=self.cuda_device,
                cls_model_path=self.cls_model_path,
                cls_conf_threshold=self.cls_conf_threshold
            )
            cls_info = f" + 分类模型: {self.cls_model_path}" if self.cls_model_path else ""
            logger.info(f"推理引擎创建成功: {self.model_path}{cls_info}, 设备: {self.device_type}")
        except Exception as e:
            logger.error(f"推理引擎创建失败: {e}")
            raise

    def detect(self, frame) -> List[dict]:
        """执行检测"""
        try:
            return self.engine.infer(frame)
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return []

    def get_class_names(self) -> List[str]:
        """获取所有类别名称"""
        return self.class_names

    def set_recog_area(self, recog_area):
        """设置检测区域"""
        self.recog_area = recog_area
        if self.engine:
            self.engine.set_recog_area(recog_area)

    def set_confidence_threshold(self, threshold):
        """设置置信度阈值"""
        self.confidence_threshold = threshold
        if self.engine:
            self.engine.set_conf_threshold(threshold)

    def release(self):
        """释放资源"""
        if self.engine:
            self.engine.release()
