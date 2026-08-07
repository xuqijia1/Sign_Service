# SharedData.py - 全局共享数据模块
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime


# ===================== 服务保活：启动锁超时兜底 =====================
# _handle_start 锁等待超时：若上一次 /start 卡在锁内（reader 自愈 force_reconnect），
# 新请求不会无限挂起。AIPP 路径 /start 毫秒级（非阻塞查 is_healthy），10s 仅兜底。
START_LOCK_TIMEOUT = 10.0


class _LockTimeout(Exception):
    """带超时锁获取失败时抛出，供 _handle_start 捕获后快速返回。"""
    pass


class _TimedLock:
    """带超时的 threading.Lock 上下文管理器：超时未获锁抛 _LockTimeout 而非死等。"""

    def __init__(self, lock, timeout, name="lock"):
        self._lock = lock
        self._timeout = timeout
        self._name = name
        self._acquired = False

    def __enter__(self):
        self._acquired = self._lock.acquire(timeout=self._timeout)
        if not self._acquired:
            raise _LockTimeout(
                f"系统繁忙，上一次启动未结束（等待 {self._timeout:.0f}s 未获得 {self._name}），请稍后重试")
        return self

    def __exit__(self, *exc):
        if self._acquired:
            self._lock.release()
        return False


# /start 启动串行锁：防止多个 /start 并发 force_reconnect 同一 DVPP 解码器（507018/损坏）
START_LOCK = threading.Lock()


class ExamState:
    """考试状态机：统一管理读帧/推理的生命周期

    IDLE      待考：读帧线程睡眠，不推理
    STARTING  /start 进行中：读帧线程读帧供校验，但跳过推理
    RUNNING   考试中：完整推理
    """
    IDLE = 0
    STARTING = 1
    RUNNING = 2


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

        # 考试状态机
        self.exam_state: int = ExamState.IDLE

        # 最新检测结果
        self.latest_boxes: List[DetectionBox] = []
        self.latest_result: Optional[SignResult] = None

        # 已识别的标志牌（用于去重）
        self.recognized_signs: Dict[str, bool] = {}

        # VideoStream 引用（HTTP 层通过它访问 dvpp_decoder；不再做服务端录制）
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
            self.exam_state = ExamState.IDLE
            self.latest_boxes = []
            self.latest_result = None
            self.recognized_signs = {}
            self.frame_count = 0
            self.start_time = 0.0

    def set_exam_state(self, state: int):
        """设置考试状态（替代 set_running）"""
        with self.data_lock:
            self.exam_state = state
            if state == ExamState.RUNNING:
                self.start_time = time.time()

    def update_detections(self, boxes: List[DetectionBox]):
        """更新检测结果（frame_count 由读帧线程递增，此处不重复）"""
        with self.data_lock:
            self.latest_boxes = boxes

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
                "exam_state": self.exam_state,
                "is_running": self.exam_state != ExamState.IDLE,
                "user_id": self.current_user_id,
                "elapsed_time": time.time() - self.start_time if self.start_time > 0 else 0
            }

# 全局单例
shared_data = SharedData()
