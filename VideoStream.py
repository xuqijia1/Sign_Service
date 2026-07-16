# VideoStream.py - 视频流处理
import os
import cv2
import time
import shutil
import threading
import subprocess
import logging
import requests
import json
from datetime import datetime
from typing import Optional, List
from SharedData import shared_data, DetectionBox, SignResult, get_sign_type
from SignModel import SignModel

try:
    from dvpp_decoder import create_dvpp_decoder
    _HAS_DVPP = True
except ImportError:
    _HAS_DVPP = False

logger = logging.getLogger(__name__)


class JpegFrameRecorder:
    """JPEG 帧录制器 — 录制期间存 JPEG，分段流式合成视频"""

    CHUNK_SIZE = 500

    def __init__(self, output_path, fps, width, height,
                 max_duration_minutes=0, jpeg_quality=85, chunk_size=500):
        self._output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._max_duration = max_duration_minutes * 60 if max_duration_minutes > 0 else 0
        self._jpeg_quality = jpeg_quality
        self._chunk_size = chunk_size
        self._tmp_dir = output_path + ".frames"
        self._frame_idx = 0
        self._chunk_idx = 0
        self._chunk_frames = 0
        self._closed = False
        self._segments = []
        self._has_ffmpeg = None
        self._min_frame_interval = 1.0 / fps if fps > 0 else 0
        self._last_frame_time = 0.0
        self._lock = threading.Lock()
        self._start_time = time.time()
        os.makedirs(self._tmp_dir, exist_ok=True)

    @property
    def output_path(self):
        return self._output_path

    @property
    def frame_count(self):
        return self._frame_idx

    def write(self, frame) -> bool:
        with self._lock:
            if self._closed:
                return False
            now = time.time()
            if self._min_frame_interval > 0 and now - self._last_frame_time < self._min_frame_interval:
                return True
            self._last_frame_time = now
            if self._max_duration > 0 and now - self._start_time >= self._max_duration:
                self._close_internal()
                return False
            path = os.path.join(self._tmp_dir, f"{self._frame_idx:07d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            self._frame_idx += 1
            self._chunk_frames += 1
            if self._chunk_frames >= self._chunk_size:
                self._flush_chunk()
            return True

    def isOpened(self) -> bool:
        return not self._closed

    def release(self):
        with self._lock:
            self._close_internal()

    def _close_internal(self):
        if self._closed:
            return
        self._closed = True
        if self._frame_idx == 0:
            self._cleanup()
            return
        t = threading.Thread(target=self._finalize, daemon=True)
        t.start()
        logger.info(f"录制停止，后台合成视频中... ({self._frame_idx}帧, {len(self._segments)}段)")

    def _finalize(self):
        try:
            if self._chunk_frames > 0:
                self._flush_chunk()
            if self._segments:
                self._concat_segments()
            self._cleanup()
        except Exception as e:
            logger.error(f"后台合成异常: {e}")

    def _check_ffmpeg(self):
        if self._has_ffmpeg is None:
            self._has_ffmpeg = shutil.which('ffmpeg') is not None
        return self._has_ffmpeg

    def _flush_chunk(self):
        if self._chunk_frames == 0:
            return
        if not self._check_ffmpeg():
            self._chunk_frames = 0
            return
        seg_path = os.path.join(self._tmp_dir, f"seg_{self._chunk_idx:04d}.mp4")
        start_idx = self._frame_idx - self._chunk_frames
        if self._mux_jpeg_to_mp4(start_idx, self._chunk_frames, seg_path):
            self._segments.append(seg_path)
            for i in range(start_idx, self._frame_idx):
                try:
                    os.remove(os.path.join(self._tmp_dir, f"{i:07d}.jpg"))
                except OSError:
                    pass
        self._chunk_idx += 1
        self._chunk_frames = 0

    def _mux_jpeg_to_mp4(self, start_idx, count, output):
        try:
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(self.fps),
                '-start_number', str(start_idx),
                '-i', os.path.join(self._tmp_dir, '%07d.jpg'),
                '-frames:v', str(count),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
                return True
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(self.fps),
                '-start_number', str(start_idx),
                '-i', os.path.join(self._tmp_dir, '%07d.jpg'),
                '-frames:v', str(count),
                '-c:v', 'mpeg4', '-q:v', '5',
                output
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._has_ffmpeg = False
        except Exception as e:
            logger.error(f"ffmpeg 合成失败: {e}")
        return False

    def _concat_segments(self):
        if len(self._segments) == 1:
            try:
                os.rename(self._segments[0], self._output_path)
                size = os.path.getsize(self._output_path)
                logger.info(f"视频合成完成: {self._output_path} ({size:,} bytes, {self._frame_idx}帧)")
            except OSError:
                self._try_copy_segment()
        else:
            concat_file = os.path.join(self._tmp_dir, "concat.txt")
            try:
                with open(concat_file, 'w') as f:
                    for seg in self._segments:
                        f.write(f"file '{seg}'\n")
                cmd = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', concat_file, '-c', 'copy', self._output_path
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode == 0 and os.path.exists(self._output_path):
                    size = os.path.getsize(self._output_path)
                    logger.info(f"视频拼接完成: {self._output_path} ({size:,} bytes, {self._frame_idx}帧)")
                else:
                    cmd = [
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', concat_file,
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        self._output_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, timeout=120)
                    if result.returncode == 0 and os.path.exists(self._output_path):
                        size = os.path.getsize(self._output_path)
                        logger.info(f"视频重编码拼接完成: {self._output_path} ({size:,} bytes)")
                    else:
                        self._try_copy_segment()
            except Exception as e:
                logger.error(f"视频拼接失败: {e}")
                self._try_copy_segment()

    def _try_copy_segment(self):
        if self._segments and os.path.exists(self._segments[0]):
            try:
                shutil.copy2(self._segments[0], self._output_path)
                size = os.path.getsize(self._output_path)
                logger.info(f"保留第一段视频: {self._output_path} ({size:,} bytes)")
            except Exception:
                logger.error(f"视频保存失败，JPEG 帧序列保留在: {self._tmp_dir}/")

    def _cleanup(self):
        try:
            if os.path.isdir(self._tmp_dir):
                shutil.rmtree(self._tmp_dir)
        except Exception:
            pass

class VideoStream:
    """视频流处理类"""

    def __init__(self, config: dict):
        self.config = config
        self.camera_url = config.get('camera_url', '')
        self.model_path = config.get('model_path', './sign.pt')
        self.cls_model_path = config.get('cls_model_path', None)
        self.cls_conf_threshold = config.get('cls_conf_threshold', 0.5)
        self.device_type = config.get('DeviceType', 'gpu')
        self.cuda_device = config.get('CudaDevice', 0)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.client_url = config.get('client_url', 'http://127.0.0.1:8090')
        self.push_interval = config.get('push_interval', 0.1)

        # 视频源
        self.cap: Optional[cv2.VideoCapture] = None
        self.dvpp_decoder = None
        self.use_dvpp = False
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_width = 1920
        self.frame_height = 1080
        self.frame_fps = 25.0

        # 模型
        self.model: Optional[SignModel] = None

        # 视频录制器（由 custom_http_server 创建和管理）
        self.video_recorder = None
        video_dir = config.get('video_save_dir', '/home/kickpi/ExamVideos/Sign')
        if not os.path.isabs(video_dir):
            video_dir = os.path.join(os.path.dirname(__file__), video_dir)
        self._video_save_dir = video_dir
        shared_data.video_recorder = self

        # 推送控制
        self.last_push_time = 0.0
        self.push_thread: Optional[threading.Thread] = None

    def start(self):
        """启动视频流处理"""
        if self.is_running:
            return

        # 加载模型
        try:
            self.model = SignModel(
                model_path=self.model_path,
                device_type=self.device_type,
                cuda_device=self.cuda_device,
                confidence_threshold=self.confidence_threshold,
                cls_model_path=self.cls_model_path,
                cls_conf_threshold=self.cls_conf_threshold
            )
            logger.info("标志牌检测模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return

        # 尝试 DVPP 硬解码
        if _HAS_DVPP and self.camera_url.startswith('rtsp://'):
            try:
                self.dvpp_decoder = create_dvpp_decoder(
                    rtsp_url=self.camera_url,
                    device_id=int(self.cuda_device) if self.device_type == 'ascend' else 0,
                    en_type="H264",
                )
                if self.dvpp_decoder is not None:
                    self.use_dvpp = True
                    self.frame_width = self.dvpp_decoder.src_width or 1920
                    self.frame_height = self.dvpp_decoder.src_height or 1080
                    self.frame_fps = self.dvpp_decoder.fps or 25.0
                    shared_data.frame_width = self.frame_width
                    shared_data.frame_height = self.frame_height
                    shared_data.frame_fps = self.frame_fps
                    logger.info(f"DVPP 硬解码已启动: {self.camera_url} | "
                                f"{self.frame_width}x{self.frame_height} @ {self.frame_fps:.1f}fps")
            except Exception as e:
                logger.warning(f"DVPP 硬解码启动失败，回退 cv2: {e}")
                self.dvpp_decoder = None
                self.use_dvpp = False

        # cv2 回退
        if not self.use_dvpp:
            self._open_video_source()
            if self.cap is None or not self.cap.isOpened():
                logger.error("无法打开视频源")
                return

        self.is_running = True

        # 启动处理线程
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

        # 启动推送线程
        self.push_thread = threading.Thread(target=self._push_loop, daemon=True)
        self.push_thread.start()

        logger.info("视频流处理已启动")

    def stop(self):
        """停止视频流处理"""
        self.is_running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

        if self.push_thread is not None:
            self.push_thread.join(timeout=2.0)
            self.push_thread = None

        if self.dvpp_decoder is not None:
            self.dvpp_decoder.release()
            self.dvpp_decoder = None
            self.use_dvpp = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.video_recorder is not None:
            self.video_recorder.release()
            self.video_recorder = None

        logger.info("视频流处理已停止")

    def _open_video_source(self):
        """打开视频源"""
        if not self.camera_url:
            logger.warning("摄像头URL为空")
            return

        # 判断是RTSP还是本地文件
        if self.camera_url.startswith('rtsp://') or self.camera_url.startswith('rtmp://'):
            # RTSP流 - 设置超时和TCP传输
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'
            self.cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
        elif self.camera_url.startswith('http://') or self.camera_url.startswith('https://'):
            # HTTP流
            self.cap = cv2.VideoCapture(self.camera_url)
        elif os.path.exists(self.camera_url):
            # 本地文件
            self.cap = cv2.VideoCapture(self.camera_url)
        else:
            # 尝试作为摄像头索引
            try:
                index = int(self.camera_url)
                self.cap = cv2.VideoCapture(index)
            except ValueError:
                self.cap = cv2.VideoCapture(self.camera_url)

        if self.cap is not None and self.cap.isOpened():
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_fps = max(1.0, float(self.cap.get(cv2.CAP_PROP_FPS)))
            # 更新共享数据的帧尺寸
            shared_data.frame_width = self.frame_width
            shared_data.frame_height = self.frame_height
            shared_data.frame_fps = self.frame_fps
            logger.info(f"视频源已打开: {self.frame_width}x{self.frame_height}, {self.frame_fps}fps")
        else:
            logger.error(f"无法打开视频源: {self.camera_url}")

    def _process_loop(self):
        """处理循环"""
        frame_count = 0

        # DVPP 后台线程需要设置 ACL context
        if self.use_dvpp and self.dvpp_decoder and hasattr(self.dvpp_decoder, '_ctx'):
            try:
                import acl
                acl.rt.set_device(self.dvpp_decoder.device_id)
                if self.dvpp_decoder._ctx is not None:
                    acl.rt.set_context(self.dvpp_decoder._ctx)
                logger.info("后台线程 ACL context 已设置")
            except Exception as e:
                logger.warning(f"ACL context 设置失败: {e}")

        while self.is_running:
            try:
                frame = None

                if self.use_dvpp and self.dvpp_decoder:
                    frame = self.dvpp_decoder.read_frame()
                    # read_frame 切换了 ACL context 到 DVPP 的，
                    # 推理前需恢复默认 context
                    if frame is not None:
                        try:
                            import acl
                            acl.rt.set_device(self.dvpp_decoder.device_id)
                        except Exception:
                            pass
                    if frame is None:
                        frame_count += 1
                        if frame_count % 100 == 0:
                            logger.warning("DVPP 读取帧失败")
                        time.sleep(0.05)
                        continue
                    if frame_count < 5:
                        logger.info(f"DVPP 读帧成功: shape={frame.shape}")
                else:
                    if self.cap is None or not self.cap.isOpened():
                        logger.warning("视频源断开，尝试重连...")
                        self._open_video_source()
                        time.sleep(1.0)
                        continue

                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        if self.camera_url and not self.camera_url.startswith('rtsp'):
                            logger.info("视频播放完毕，重新开始...")
                            self.cap.release()
                            self._open_video_source()
                            continue
                        logger.warning("RTSP读取帧失败，尝试重连...")
                        self.cap.release()
                        self.cap = None
                        time.sleep(1.0)
                        continue

                frame_count += 1

                # 录制视频
                if shared_data.is_recording and self.video_recorder is not None:
                    self.video_recorder.write(frame)

                # 执行检测
                if shared_data.is_running:
                    detections = self.model.detect(frame)

                    # 转换为检测框列表
                    boxes = []
                    for det in detections:
                        box = DetectionBox(
                            X=det['x'],
                            Y=det['y'],
                            Width=det['width'],
                            Height=det['height'],
                            Label=det['label'],
                            Confidence=det['confidence']
                        )
                        boxes.append(box)

                    # 更新共享数据
                    shared_data.update_detections(boxes)

                    # 如果有检测结果，更新识别结果
                    if boxes:
                        # 取置信度最高的
                        best_box = max(boxes, key=lambda b: b.Confidence)
                        result = SignResult(
                            SignType=get_sign_type(best_box.Label),
                            SignName=best_box.Label,
                            IsCorrect=True,
                            Confidence=best_box.Confidence,
                            Boxes=boxes,
                            Timestamp=time.time()
                        )
                        shared_data.update_result(result)

                # 显示窗口（可选）
                if self.config.get('show_window', False):
                    # 绘制检测框
                    display_frame = frame.copy()
                    if shared_data.is_running and boxes:
                        display_frame = self._draw_boxes(display_frame, boxes)
                    cv2.imshow('Sign Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            except Exception as e:
                logger.error(f"处理帧异常: {e}")
                time.sleep(0.1)

    def _push_loop(self):
        """推送结果到客户端"""
        while self.is_running:
            try:
                if not shared_data.is_running:
                    time.sleep(0.1)
                    continue

                current_time = time.time()
                if current_time - self.last_push_time < self.push_interval:
                    time.sleep(0.01)
                    continue

                self.last_push_time = current_time

                # 获取最新结果
                result = shared_data.get_result()
                if result is None:
                    continue

                # 推送到客户端
                self._push_to_client(result)

            except Exception as e:
                logger.error(f"推送结果异常: {e}")
                time.sleep(0.1)

    def _push_to_client(self, result: SignResult):
        """推送结果到客户端"""
        try:
            url = f"{self.client_url}/sign_return"

            data = {
                "SignType": result.SignType,
                "SignName": result.SignName,
                "IsCorrect": result.IsCorrect,
                "Confidence": result.Confidence,
                "Boxes": [
                    {
                        "X": b.X,
                        "Y": b.Y,
                        "Width": b.Width,
                        "Height": b.Height,
                        "Label": b.Label,
                        "Confidence": b.Confidence
                    }
                    for b in result.Boxes
                ]
            }

            response = requests.post(url, json=data, timeout=1.0)
            if response.status_code == 200:
                # 同类型标志牌只记录一次推送日志
                if not hasattr(self, '_last_pushed_type') or self._last_pushed_type != result.SignType:
                    self._last_pushed_type = result.SignType
                    logger.info(f"推送成功: {result.SignType} - {result.SignName}")
            else:
                # 推送失败只记录一次
                if not hasattr(self, '_push_fail_logged') or not self._push_fail_logged:
                    self._push_fail_logged = True
                    logger.warning(f"推送失败: {response.status_code} (后续不再重复提示)")

        except requests.exceptions.RequestException as e:
            if not hasattr(self, '_push_fail_logged') or not self._push_fail_logged:
                self._push_fail_logged = True
                logger.warning(f"推送请求失败: {e} (后续不再重复提示)")
        except Exception as e:
            logger.error(f"推送异常: {e}")

    def _draw_boxes(self, frame, boxes: List[DetectionBox]):
        """在帧上绘制检测框"""
        # 标志牌类型颜色映射
        color_map = {
            'prohibit': (0, 0, 255),      # 禁止类 - 红色
            'warning': (0, 255, 255),     # 警告类 - 黄色
            'mandatory': (255, 0, 0),     # 指令类 - 蓝色
            'info': (0, 255, 0)           # 提示类 - 绿色
        }

        # 显示名称映射
        display_names = {
            'prohibit_switch_on': '禁止合闸',
            'prohibit_start': '禁止启动',
            'prohibit_approach': '禁止靠近',
            'prohibit_climb': '禁止攀爬',
            'prohibit_touch': '禁止触摸',
            'prohibit_enter': '禁止入内',
            'warning_attention': '注意危险',
            'warning_electric': '当心触电',
            'warning_cable': '当心电缆',
            'warning_fire': '当心火灾',
            'warning_auto_start': '当心自动启动',
            'warning_fall': '当心坠落',
            'mandatory_grounding': '必须接地',
            'mandatory_helmet': '必须戴安全帽',
            'mandatory_clothing': '必须穿防护服',
            'mandatory_shoes': '必须穿防护鞋',
            'mandatory_gloves': '必须戴防护手套',
            'mandatory_belt': '必须系安全带',
            'info_three_phase': '三相电提示',
            'info_work_here': '在此工作',
            'info_emergency_stop': '紧急停止',
            'info_overvoltage': '过电压提示',
            'info_live_work': '带电作业',
            'info_anti_interference': '抗干扰标识'
        }

        for box in boxes:
            # 根据标签类型确定颜色
            label_lower = box.Label.lower()
            color = (0, 255, 0)  # 默认绿色
            for prefix, c in color_map.items():
                if label_lower.startswith(prefix):
                    color = c
                    break

            # 绘制矩形框
            x, y, w, h = box.X, box.Y, box.Width, box.Height
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 获取显示名称
            display_name = display_names.get(box.Label, box.Label)
            text = f"{display_name} ({box.Confidence:.0%})"

            # 绘制文本背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # 文本位置（框上方）
            text_y = y - text_h - 5
            if text_y < 0:
                text_y = y + h + 5

            cv2.rectangle(frame, (x, text_y - 2), (x + text_w, text_y + text_h + 2), color, -1)
            cv2.putText(frame, text, (x, text_y + text_h), font, font_scale, (255, 255, 255), thickness)

        return frame
