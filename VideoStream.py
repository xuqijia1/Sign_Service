# VideoStream.py - 视频流处理
import os
import cv2
import time
import threading
import logging
import requests
import json
from datetime import datetime
from typing import Optional, List
from SharedData import shared_data, DetectionBox, SignResult, get_sign_type, ExamState
from SignModel import SignModel

try:
    from dvpp_decoder import create_dvpp_decoder
    _HAS_DVPP = True
except ImportError:
    _HAS_DVPP = False

logger = logging.getLogger(__name__)


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
        self.orig_size = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_width = 1920
        self.frame_height = 1080
        self.frame_fps = 25.0

        # 模型
        self.model: Optional[SignModel] = None

        # VideoStream 引用，供 HTTP 层访问 dvpp_decoder（不再做服务端录制）
        shared_data.video_recorder = self

        # 推送控制
        self.last_push_time = 0.0
        self.push_thread: Optional[threading.Thread] = None

    def start(self):
        """启动视频流处理"""
        if self.is_running:
            return

        # 加载模型
        # 加载模型（根据设备类型和 AIPP 模式选择模型路径）
        use_aipp = self.config.get('AscendAipp', False)
        if self.device_type == 'ascend':
            # Ascend 模式：选择 OM 模型
            if use_aipp:
                model_path = self.config.get('ascend_aipp_model_path', './sign_aipp.om')
                cls_model_path = self.config.get('ascend_aipp_cls_model_path', './sign_cls_aipp.om')
            else:
                model_path = self.config.get('ascend_model_path', './sign.om')
                cls_model_path = self.config.get('ascend_cls_model_path', './sign_cls.om')
        else:
            model_path = self.model_path
            cls_model_path = self.cls_model_path

        try:
            self.model = SignModel(
                model_path=model_path,
                device_type=self.device_type,
                cuda_device=self.cuda_device,
                confidence_threshold=self.confidence_threshold,
                cls_model_path=cls_model_path,
                cls_conf_threshold=self.cls_conf_threshold,
                aipp=use_aipp
            )
            cls_info = f" + 分类模型: {cls_model_path}" if cls_model_path else ""
            logger.info(f"标志牌检测模型加载成功: {model_path}{cls_info}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return

        # Ascend 环境 DVPP 硬解码：摄像头未开/不可达时不崩服务，留 dvpp_decoder=None 由 _process_loop 后台重连自愈
        if _HAS_DVPP and self.camera_url.startswith('rtsp://'):
            self.use_dvpp = True  # 先置位，避免创建失败时误掉 cv2 分支（cap=None 必崩）
            try:
                self.dvpp_decoder = create_dvpp_decoder(
                    rtsp_url=self.camera_url,
                    device_id=int(self.cuda_device) if self.device_type == 'ascend' else 0,
                    en_type="H264",
                )
                self.frame_width = self.dvpp_decoder.src_width or 1920
                self.frame_height = self.dvpp_decoder.src_height or 1080
                self.frame_fps = self.dvpp_decoder.fps or 25.0
                self.orig_size = (self.dvpp_decoder.src_height, self.dvpp_decoder.src_width)
                shared_data.frame_width = self.frame_width
                shared_data.frame_height = self.frame_height
                shared_data.frame_fps = self.frame_fps
                logger.info(f"DVPP 硬解码已启动: {self.camera_url} | "
                            f"{self.frame_width}x{self.frame_height} @ {self.frame_fps:.1f}fps")
            except Exception as e:
                # 摄像头未开/不可达：不崩服务，留 dvpp_decoder=None，由 _process_loop IDLE 触发 force_reconnect 后台重连
                # （与 ydpt 一致：服务能启动，/start 查 is_healthy 非阻塞拒绝）
                self.dvpp_decoder = None
                logger.error(f"DVPP 硬解码启动失败（不回退 cv2，_process_loop 将后台重连）: {e}")

        # 非 Ascend 环境：使用 cv2 软解码作为兼容方案
        if not self.use_dvpp:
            self._open_video_source()
            if self.cap is None or not self.cap.isOpened():
                logger.error("无法打开视频源")
                return

        self.is_running = True
        self._consecutive_failures = 0
        self._last_dvpp_warn_ts = 0
        self._force_reconnecting = False  # force_reconnect 期间置 True，阻止 _process_loop 干扰
        self.is_healthy = False  # 流健康标志（IDLE 时由 _process_loop 探测维护，/start 非阻塞检查）
        self._last_probe_time = 0.0  # 上次健康探测时间
        self._probe_fail_count = 0  # 连续探测失败次数

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
            self.orig_size = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # 释放模型（NPU/GPU 资源）
        if self.model is not None:
            try:
                if hasattr(self.model, 'engine') and self.model.engine is not None:
                    if hasattr(self.model.engine, 'release'):
                        self.model.engine.release()
                self.model = None
            except Exception as e:
                logger.warning(f"模型释放异常: {e}")

        logger.info("视频流处理已停止")

    def force_reconnect(self):
        """强制 DVPP 完整重连（由 /start 在帧获取失败时调用）

        soft_reset 不足以恢复长时间空闲后的死连接，需完整重建解码器。
        设置 _force_reconnecting 标志防止 _process_loop 干扰。
        VDEC 通道释放是异步的，507018 时需重试（递增等待）。
        """
        if not self.use_dvpp or not _HAS_DVPP:
            return False
        logger.warning("[FORCE-RECONNECT] 强制重建 DVPP 解码器...")
        self._force_reconnecting = True
        try:
            # 1. 释放旧解码器
            if self.dvpp_decoder is not None:
                try:
                    self.dvpp_decoder.release()
                except Exception:
                    pass
                self.dvpp_decoder = None
            # 2. 创建新解码器（507018 时重试，递增等待 VDEC 通道释放）
            # Ascend 环境 create_dvpp_decoder 失败直接抛 RuntimeError，不回退 cv2
            new_decoder = None
            max_attempts = 5
            wait_schedule = [2.0, 5.0, 8.0, 15.0, 30.0]  # 总计 60s
            for attempt in range(max_attempts):
                wait_s = wait_schedule[attempt]
                logger.info(f"[FORCE-RECONNECT] 等待 {wait_s:.0f}s 后创建新解码器 (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait_s)
                try:
                    new_decoder = create_dvpp_decoder(
                        rtsp_url=self.camera_url,
                        device_id=int(self.cuda_device) if self.device_type == 'ascend' else 0,
                        en_type="H264",
                    )
                    # 成功（force_reconnect 仅在 _HAS_DVPP=True 即 Ascend 环境调用，
                    # create_dvpp_decoder 必返回 AclVdecDecoder 或抛 RuntimeError）
                    if new_decoder is not None:
                        break
                except RuntimeError as e:
                    logger.warning(f"[FORCE-RECONNECT] attempt {attempt+1}/{max_attempts} 失败: {e}")
                    new_decoder = None
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        logger.error(f"[FORCE-RECONNECT] DVPP 重建 {max_attempts} 次均失败（最后尝试: {e}）")
                        return False
            if new_decoder is None:
                logger.error("[FORCE-RECONNECT] DVPP 重建失败：解码器为 None")
                return False
            self.dvpp_decoder = new_decoder
            self.frame_width = self.dvpp_decoder.src_width or 1920
            self.frame_height = self.dvpp_decoder.src_height or 1080
            self.frame_fps = self.dvpp_decoder.fps or 25.0
            self.orig_size = (self.dvpp_decoder.src_height, self.dvpp_decoder.src_width)
            shared_data.frame_width = self.frame_width
            shared_data.frame_height = self.frame_height
            shared_data.frame_fps = self.frame_fps
            self._consecutive_failures = 0
            logger.info("[FORCE-RECONNECT] DVPP 解码器重建成功")
            return True
        except Exception as e:
            logger.error(f"[FORCE-RECONNECT] DVPP 重建失败: {e}")
            self.dvpp_decoder = None
            return False
        finally:
            self._force_reconnecting = False

    def probe_health(self):
        """流健康探测：拉一帧验证流可解，维护 is_healthy。

        IDLE 时由 _process_loop 周期调用，/start 非阻塞查 is_healthy。
        AIPP 走 read_frame_aipp（内部已 dvpp_free(vdec_buffer)，返回的 nv12_dict['buffer']
        是复用的 _dev_rsz_nv12 不能 free），非 AIPP 走 read_frame。probe 零清理。
        """
        if not (self.use_dvpp and self.dvpp_decoder and self.dvpp_decoder.is_started):
            self.is_healthy = False
            return False
        try:
            use_aipp = self.config.get('AscendAipp', False)
            if use_aipp:
                nv12_info, _ = self.dvpp_decoder.read_frame_aipp()
                ok = nv12_info is not None
            else:
                frame = self.dvpp_decoder.read_frame()
                ok = frame is not None
            if ok:
                self.is_healthy = True
                self._probe_fail_count = 0
                return True
        except Exception as e:
            logger.warning(f"健康探测异常: {e}")
        self._probe_fail_count += 1
        self.is_healthy = False
        return False

    def is_streaming(self, max_wait=10.0):
        """校验视频流是否在产出帧（通过 frame_count 增长判断）

        用于 /start 校验视频源可用性。
        """
        start_count = shared_data.frame_count
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if shared_data.frame_count > start_count:
                return True
            time.sleep(0.2)
        return False

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
                # force_reconnect 期间跳过，避免 _process_loop 抢占 dvpp_decoder
                if self._force_reconnecting:
                    time.sleep(0.1)
                    continue

                # 待考状态（IDLE）：不读帧、不解码、不推理、不录屏，降低 CPU/NPU 占用
                # demux 线程仍在后台拉流保活，RTSP 不会断
                if shared_data.exam_state == ExamState.IDLE:
                    # 解码器未建（启动时摄像头未开）：尝试创建，成功后下个周期开始健康探测
                    if self.use_dvpp and self.dvpp_decoder is None:
                        self.force_reconnect()
                    # IDLE 健康探测：周期验证流可解，维护 is_healthy 供 /start 非阻塞检查。
                    # 连续 3 次失败触发 force_reconnect 自愈（reader 是 daemon 线程，
                    # 卡死只影响该路自愈，不影响 HTTP 响应）。
                    elif self.use_dvpp and self.dvpp_decoder:
                        _now = time.time()
                        if _now - self._last_probe_time >= 2.0:
                            self._last_probe_time = _now
                            _was_healthy = self.is_healthy
                            self.probe_health()
                            if not self.is_healthy:
                                if self._probe_fail_count >= 3 and self._probe_fail_count % 3 == 0:
                                    logger.warning(f"IDLE 健康探测失败 {self._probe_fail_count} 次，触发重连")
                                    if self.force_reconnect():
                                        self.is_healthy = True
                                        self._probe_fail_count = 0
                            elif not _was_healthy:
                                logger.info("IDLE 健康探测恢复")
                    time.sleep(0.5)
                    continue

                frame = None
                use_aipp = self.config.get('AscendAipp', False)

                if self.use_dvpp and self.dvpp_decoder:
                    if use_aipp:
                        # AIPP 零拷贝路径：同时获取 NV12 device buffer 和 BGR 帧
                        nv12_info, bgr_frame = self.dvpp_decoder.read_frame_aipp()
                        if nv12_info is not None:
                            try:
                                import acl
                                acl.rt.set_device(self.dvpp_decoder.device_id)
                            except Exception:
                                pass
                            self._consecutive_failures = 0
                            frame = bgr_frame  # BGR 帧用于显示

                            # AIPP 零拷贝推理：直接传 device buffer，同时传 BGR 帧用于分类裁剪
                            # 仅 RUNNING 状态推理，STARTING 期间只读帧供三级校验
                            if shared_data.exam_state == ExamState.RUNNING:
                                detections = self.model.detect(nv12_info, orig_size=self.orig_size, bgr_image=bgr_frame)
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
                                shared_data.update_detections(boxes)
                                if boxes:
                                    if frame_count <= 5 or frame_count % 100 == 0:
                                        labels = [b.Label for b in boxes]
                                        logger.info(f"[AIPP] 检测: {len(boxes)}个 | 标签: {labels}")
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
                        else:
                            self._consecutive_failures += 1
                            if self._consecutive_failures > 10:
                                now_ts = time.time()
                                if now_ts - self._last_dvpp_warn_ts > 30:
                                    logger.warning(f"DVPP 连续读取失败{self._consecutive_failures}次，执行软复位清空缓存")
                                    self._last_dvpp_warn_ts = now_ts
                                self.dvpp_decoder.soft_reset()
                                self._consecutive_failures = 0
                            time.sleep(0.05)
                            continue
                    else:
                        # 非 AIPP 路径：读取 BGR 帧
                        frame = self.dvpp_decoder.read_frame()
                        if frame is not None:
                            try:
                                import acl
                                acl.rt.set_device(self.dvpp_decoder.device_id)
                            except Exception:
                                pass
                            self._consecutive_failures = 0
                            if frame_count < 5:
                                logger.info(f"DVPP 读帧成功: shape={frame.shape}")
                        else:
                            self._consecutive_failures += 1
                            if self._consecutive_failures > 10:
                                now_ts = time.time()
                                if now_ts - self._last_dvpp_warn_ts > 30:
                                    logger.warning(f"DVPP 连续读取失败{self._consecutive_failures}次，执行软复位清空缓存")
                                    self._last_dvpp_warn_ts = now_ts
                                self.dvpp_decoder.soft_reset()
                                self._consecutive_failures = 0
                            time.sleep(0.05)
                            continue
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
                # 读帧计数（STARTING/RUNNING 都递增），供 is_streaming 校验帧产出
                with shared_data.data_lock:
                    shared_data.frame_count += 1

                # 执行检测（AIPP 模式已在上面处理，跳过）
                if not use_aipp and shared_data.exam_state == ExamState.RUNNING:
                    detections = self.model.detect(frame, orig_size=self.orig_size)

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
                        if frame_count <= 5 or frame_count % 100 == 0:
                            labels = [b.Label for b in boxes]
                            logger.info(f"检测: {len(boxes)}个 | 标签: {labels}")
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
                    if shared_data.exam_state == ExamState.RUNNING and boxes:
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
                if shared_data.exam_state == ExamState.IDLE:
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
            'info_three_phase': '动火作业区',
            'info_live_work': '紧急破窗',
            'info_emergency_phone': '紧急电话',
            'info_assembly_point': '紧急集合点',
            'info_emergency_exit': '紧急出口',
            'info_first_aid': '急救'
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
