# SharedData.py - 全局共享数据模块
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


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

        # 已识别的标志牌（用于去重，仅收录多帧稳定确认后的标签）
        self.recognized_signs: Dict[str, bool] = {}

        # 多帧确认：各标签连续命中帧计数（本帧未出现即清零）
        self.label_streaks: Dict[str, int] = {}

        # 确认阈值：连续命中该帧数才写入 recognized_signs（挡单帧闪现误检）
        self.confirm_frames: int = 5

        # VideoStream 引用（HTTP 层通过它访问 dvpp_decoder；不再做服务端录制）
        self.video_recorder = None

        # 视频帧尺寸
        self.frame_width: int = 1920
        self.frame_height: int = 1080
        self.frame_fps: float = 25.0

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
            self.label_streaks = {}
            self.frame_count = 0
            self.start_time = 0.0

    def set_exam_state(self, state: int):
        """设置考试状态（替代 set_running）"""
        with self.data_lock:
            self.exam_state = state
            if state == ExamState.RUNNING:
                self.start_time = time.time()

    def update_detections(self, boxes: List[DetectionBox]):
        """更新检测结果（frame_count 由读帧线程递增，此处不重复）。
        多帧确认：各标签连续命中计数，本帧未出现即清零；连续命中达 confirm_frames
        才写入 recognized_signs——单帧/短暂闪现的误检（反光、遮挡、快速挥动）
        进不了累计集，客户端不再一帧定对错。仅在 RUNNING 期间被调用。"""
        with self.data_lock:
            self.latest_boxes = boxes

            # 同帧同标签只计1次（画面里多块同款牌不能一帧凑满阈值）
            present = {b.Label for b in boxes if b.Label}
            for label in present:
                self.label_streaks[label] = self.label_streaks.get(label, 0) + 1
                if self.label_streaks[label] >= self.confirm_frames and label not in self.recognized_signs:
                    self.recognized_signs[label] = True
                    logger.info(f"标志牌多帧确认: {label} 连续{self.label_streaks[label]}帧命中，写入累计集")

            # 上一帧在、本帧不在的标签：连续性打断，计数清零
            for label in list(self.label_streaks.keys()):
                if label not in present:
                    del self.label_streaks[label]

    def update_result(self, result: SignResult):
        """更新识别结果（仅刷新 latest_result 供轮询展示；
        recognized_signs 的写入统一由 update_detections 的多帧确认负责）"""
        with self.data_lock:
            self.latest_result = result

    def clear_results(self):
        """清空识别结果（/api/Stop 当场调用：两场之间轮询接口不再返回上一考生残留；
        frame_count/start_time 不动，/start 仍有自己的全量清理）"""
        with self.data_lock:
            self.latest_boxes = []
            self.latest_result = None
            self.recognized_signs = {}
            self.label_streaks = {}

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
