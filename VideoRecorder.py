# VideoRecorder.py - 视频录制器
import os
import cv2
import time
import threading
import platform
from datetime import datetime
from typing import Optional

class VideoRecorder:
    """视频录制器"""

    def __init__(self, save_dir: str = "./videos", max_duration_minutes: int = 45):
        self.save_dir = save_dir
        self.max_duration = max_duration_minutes * 60
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.start_time = 0.0
        self.current_file = ""
        self.lock = threading.Lock()
        self.min_frame_interval = 0.04  # 25fps
        self.last_frame_time = 0.0

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

    def _get_codec_config(self):
        """根据操作系统返回编码器配置"""
        system = platform.system()
        if system == "Windows":
            return [
                ('XVID', '.avi'),
                ('MJPG', '.avi'),
                ('mp4v', '.mp4')
            ]
        elif system == "Linux":
            return [
                ('X264', '.mp4'),
                ('mp4v', '.mp4'),
                ('XVID', '.avi')
            ]
        else:
            return [('mp4v', '.mp4')]

    def _test_codec(self, fourcc_str: str, width: int = 1920, height: int = 1080) -> bool:
        """测试编码器是否可用"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            test_file = os.path.join(self.save_dir, "codec_test.tmp")
            writer = cv2.VideoWriter(test_file, fourcc, 25.0, (width, height))
            if writer.isOpened():
                writer.release()
                if os.path.exists(test_file):
                    os.remove(test_file)
                return True
        except Exception:
            pass
        return False

    def start_recording(self, user_id: str = "", width: int = None, height: int = None, fps: float = 25.0) -> str:
        """开始录制"""
        with self.lock:
            if self.is_recording:
                self._stop_recording_internal()

            # 如果没有指定尺寸，使用默认值
            if width is None or height is None:
                width, height = 1920, 1080

            # 创建日期/用户子目录
            date_dir = datetime.now().strftime("%Y%m%d")
            full_dir = os.path.join(self.save_dir, date_dir, user_id) if user_id else os.path.join(self.save_dir, date_dir)
            os.makedirs(full_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%H%M%S")
            user_suffix = f"_{user_id}" if user_id else ""
            base_name = f"sign_record{user_suffix}_{timestamp}"

            # 尝试不同的编码器
            for fourcc_str, ext in self._get_codec_config():
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    file_path = os.path.join(full_dir, base_name + ext)
                    self.video_writer = cv2.VideoWriter(file_path, fourcc, fps, (int(width), int(height)))

                    if self.video_writer.isOpened():
                        self.current_file = file_path
                        self.is_recording = True
                        self.start_time = time.time()
                        self.last_frame_time = 0.0
                        self.frame_size = (int(width), int(height))
                        self.fps = fps
                        return file_path
                    else:
                        self.video_writer.release()
                except Exception as e:
                    continue

            return ""

    def write_frame(self, frame) -> bool:
        """写入帧"""
        with self.lock:
            if not self.is_recording or self.video_writer is None:
                return False

            # 控制帧率
            current_time = time.time()
            if current_time - self.last_frame_time < self.min_frame_interval:
                return True
            self.last_frame_time = current_time

            # 检查时长限制
            if current_time - self.start_time >= self.max_duration:
                self._stop_recording_internal()
                return False

            try:
                self.video_writer.write(frame)
                return True
            except Exception:
                self._stop_recording_internal()
                return False

    def stop_recording(self) -> str:
        """停止录制"""
        with self.lock:
            return self._stop_recording_internal()

    def _stop_recording_internal(self) -> str:
        """内部停止录制方法（不加锁）"""
        video_path = self.current_file

        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        self.is_recording = False
        self.current_file = ""
        self.start_time = 0.0

        return video_path

    def get_status(self) -> dict:
        """获取录制状态"""
        return {
            "is_recording": self.is_recording,
            "current_file": self.current_file,
            "duration": time.time() - self.start_time if self.is_recording else 0
        }
