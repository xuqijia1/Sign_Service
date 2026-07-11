# SharedData.py - 全局共享数据模块
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

@dataclass
class DetectionBox:
    """检测框数据"""
    X: int = 0
    Y: int = 0
    Width: int = 0
    Height: int = 0
    Label: str = ""
    Confidence: float = 0.0

@dataclass
class SignResult:
    """标志牌识别结果"""
    SignType: str = ""
    SignName: str = ""
    IsCorrect: bool = False
    Confidence: float = 0.0
    Boxes: List[DetectionBox] = field(default_factory=list)
    Timestamp: float = 0.0

# 标志牌类型映射
SIGN_TYPE_MAP = {
    "prohibit": "禁止",
    "warning": "警告",
    "mandatory": "指令",
    "info": "提示"
}

def get_sign_type(label: str) -> str:
    """根据标签获取标志牌类型"""
    label_lower = label.lower()
    for prefix, sign_type in SIGN_TYPE_MAP.items():
        if label_lower.startswith(prefix):
            return sign_type
    return "未知"

class SharedData:
    """全局共享数据类"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 当前用户ID
        self.current_user_id: str = ""

        # 是否正在运行
        self.is_running: bool = False

        # 是否正在录制
        self.is_recording: bool = False

        # 最新检测结果
        self.latest_boxes: List[DetectionBox] = []
        self.latest_result: Optional[SignResult] = None

        # 已识别的标志牌（用于去重）
        self.recognized_signs: Dict[str, bool] = {}

        # 视频录制器引用
        self.video_recorder = None

        # 视频帧尺寸
        self.frame_width: int = 1920
        self.frame_height: int = 1080
        self.frame_fps: float = 25.0

        # 推送客户端URL
        self.client_url: str = "http://127.0.0.1:8090"

        # 推送间隔
        self.push_interval: float = 0.1

        # 数据锁
        self.data_lock = threading.Lock()

        # 帧计数
        self.frame_count: int = 0

        # 开始时间
        self.start_time: float = 0.0

    def reset(self):
        """重置状态"""
        with self.data_lock:
            self.current_user_id = ""
            self.is_running = False
            self.is_recording = False
            self.latest_boxes = []
            self.latest_result = None
            self.recognized_signs = {}
            self.frame_count = 0
            self.start_time = 0.0

    def set_running(self, running: bool):
        """设置运行状态"""
        with self.data_lock:
            self.is_running = running
            if running:
                self.start_time = time.time()

    def update_detections(self, boxes: List[DetectionBox]):
        """更新检测结果"""
        with self.data_lock:
            self.latest_boxes = boxes
            self.frame_count += 1

    def update_result(self, result: SignResult):
        """更新识别结果"""
        with self.data_lock:
            self.latest_result = result
            # 记录已识别的标志牌
            if result.SignName:
                self.recognized_signs[result.SignName] = True

    def get_boxes(self) -> List[DetectionBox]:
        """获取最新检测框"""
        with self.data_lock:
            return self.latest_boxes.copy()

    def get_result(self) -> Optional[SignResult]:
        """获取最新识别结果"""
        with self.data_lock:
            return self.latest_result

    def get_all_results(self) -> Dict:
        """获取所有识别结果"""
        with self.data_lock:
            return {
                "recognized_signs": list(self.recognized_signs.keys()),
                "frame_count": self.frame_count,
                "is_running": self.is_running,
                "user_id": self.current_user_id,
                "elapsed_time": time.time() - self.start_time if self.start_time > 0 else 0
            }

# 全局单例
shared_data = SharedData()
