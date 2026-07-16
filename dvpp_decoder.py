#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ascend-FFmpeg 硬解码 + VPC 硬件色彩转换模块

NPU 全流程：VDEC 硬解码(H.264/H.265) → VPC 硬件色彩转换(NV12→BGR) + resize
替代 CPU 的 cv2.cvtColor + cv2.resize，大幅降低 CPU 占用。

使用方式：
    decoder = create_dvpp_decoder(rtsp_url="rtsp://...", en_type="H264")
    frame = decoder.read_frame()           # BGR numpy (VPC 硬件转换)
    bgr640 = decoder.read_frame_resized(640, 640)  # BGR 640x640 (VPC 一步完成)
    decoder.release()
"""

import subprocess
import os
import time
import queue
import threading
import numpy as np

try:
    import acl
    _HAS_ACL = True
except ImportError:
    _HAS_ACL = False

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

# VPC 格式常量
FMT_NV12 = 1   # PIXEL_FORMAT_YUV_SEMIPLANAR_420
FMT_BGR  = 13  # PIXEL_FORMAT_BGR_888
ACL_H2D = 1
ACL_D2H = 2

# VDEC 通道文件锁路径（同容器内多服务自动分配 channel_id）
_CHANNEL_LOCK_FILE = "/tmp/ascend_vdec_channels.lock"

# Ascend-FFmpeg 解码器映射
CODEC_MAP = {
    "H264": "h264_ascend", "H265": "h265_ascend",
    "h264": "h264_ascend", "h265": "h265_ascend",
    "HEVC": "h265_ascend", "AVC": "h264_ascend",
}

# RTSP 流 codec_name → 标准编码类型
CODEC_NAME_TO_EN_TYPE = {
    "h264": "H264", "avc": "H264", "H264": "H264",
    "hevc": "H265", "h265": "H265", "H265": "H265",
    "libx264": "H264", "libx265": "H265",
}


def _align_up(val, alignment):
    return ((val + alignment - 1) // alignment) * alignment


def _allocate_channel_id() -> int:
    """自动分配 VDEC channel_id（同容器内唯一，文件锁保证安全）

    优先级：环境变量 DVPP_CHANNEL_ID > 文件锁自动分配 > 回退 0
    """
    env_ch = os.environ.get('DVPP_CHANNEL_ID')
    if env_ch is not None:
        try:
            return int(env_ch)
        except ValueError:
            pass

    if not _HAS_FCNTL:
        return 0

    try:
        fd = open(_CHANNEL_LOCK_FILE, 'a+')
        fcntl.flock(fd, fcntl.LOCK_EX)
        fd.seek(0)
        used = set()
        for line in fd:
            line = line.strip()
            if line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        used.add(int(parts[1]))
                    except ValueError:
                        pass
        for ch in range(256):
            if ch not in used:
                fd.write(f"{os.getpid()}:{ch}\n")
                fd.flush()
                fcntl.flock(fd, fcntl.LOCK_UN)
                fd.close()
                return ch
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
        raise RuntimeError("无可用 VDEC 通道 (0~255 已满)")
    except (OSError, IOError):
        return 0


def _release_channel_id(channel_id: int):
    """释放自动分配的 VDEC 通道"""
    if not _HAS_FCNTL:
        return
    try:
        fd = open(_CHANNEL_LOCK_FILE, 'r+')
        fcntl.flock(fd, fcntl.LOCK_EX)
        lines = fd.readlines()
        fd.seek(0)
        fd.truncate()
        pid_ch = f"{os.getpid()}:{channel_id}"
        for line in lines:
            if line.strip() != pid_ch:
                fd.write(line)
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
    except (OSError, IOError):
        pass


# ffmpeg/ffprobe 搜索路径
_FFMPEG_SEARCH_PATHS = [
    "/usr/local/ascend_ffmpeg/bin",   # Ascend-FFmpeg 默认安装路径
    "/usr/local/bin",
    "/usr/bin",
]


def _find_ffmpeg():
    """查找 Ascend-FFmpeg 可执行文件路径"""
    for name in ('ffmpeg',):
        for dir_path in os.environ.get('PATH', '').split(os.pathsep):
            full = os.path.join(dir_path, name)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                return os.path.dirname(full)

    for dir_path in _FFMPEG_SEARCH_PATHS:
        ffmpeg_path = os.path.join(dir_path, 'ffmpeg')
        if os.path.isfile(ffmpeg_path) and os.access(ffmpeg_path, os.X_OK):
            return dir_path

    return None


def _get_ffmpeg_bin():
    ffmpeg_dir = _find_ffmpeg()
    if ffmpeg_dir:
        return os.path.join(ffmpeg_dir, 'ffmpeg')
    return 'ffmpeg'


def _get_ffprobe_bin():
    ffmpeg_dir = _find_ffmpeg()
    if ffmpeg_dir:
        ffprobe = os.path.join(ffmpeg_dir, 'ffprobe')
        if os.path.isfile(ffprobe):
            return ffprobe
    return 'ffprobe'


def _check_decoder_available(decoder_name):
    try:
        env = os.environ.copy()
        ffmpeg_dir = _find_ffmpeg()
        if ffmpeg_dir:
            lib_dir = os.path.join(os.path.dirname(ffmpeg_dir), 'lib')
            if os.path.isdir(lib_dir):
                env['LD_LIBRARY_PATH'] = lib_dir + ':' + env.get('LD_LIBRARY_PATH', '')
        result = subprocess.run(
            [_get_ffmpeg_bin(), '-decoders'],
            capture_output=True, text=True, timeout=5, env=env
        )
        return decoder_name in result.stdout
    except Exception:
        return False


def _probe_video_info(rtsp_url):
    """用 ffprobe 探测视频流信息：宽高、帧率、编码格式"""
    try:
        cmd = [
            _get_ffprobe_bin(), '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,codec_name',
            '-of', 'json',
            '-rtsp_transport', 'tcp',
            rtsp_url
        ]
        env = os.environ.copy()
        ffmpeg_dir = _find_ffmpeg()
        if ffmpeg_dir:
            lib_dir = os.path.join(os.path.dirname(ffmpeg_dir), 'lib')
            if os.path.isdir(lib_dir):
                env['LD_LIBRARY_PATH'] = lib_dir + ':' + env.get('LD_LIBRARY_PATH', '')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, env=env)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        import json
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        if not streams:
            return None
        s = streams[0]
        width = int(s.get('width', 0))
        height = int(s.get('height', 0))
        codec_name = s.get('codec_name', '')
        fps_str = s.get('r_frame_rate', '25/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) > 0 else 25.0
        else:
            fps = float(fps_str) if fps_str else 25.0
        if width <= 0 or height <= 0:
            return None
        return {'width': width, 'height': height, 'fps': fps, 'codec_name': codec_name}
    except Exception as e:
        print(f"[Ascend-FFmpeg] ffprobe 失败: {e}")
    return None


def _nv12_to_bgr_cpu(nv12_data, width, height):
    """NV12 → BGR CPU 转换（回退用）"""
    import cv2
    nv12 = np.frombuffer(nv12_data, dtype=np.uint8).reshape(height * 3 // 2, width)
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


def nv12_to_bgr(nv12_data, width, height):
    """NV12 → BGR（公开接口，CPU 回退）"""
    return _nv12_to_bgr_cpu(nv12_data, width, height)


# ==================== 抽象基类 ====================

class BaseVideoDecoder:
    """视频解码器抽象基类"""
    def start(self) -> bool:
        raise NotImplementedError
    def read_frame(self):
        """返回 BGR numpy，或 None"""
        raise NotImplementedError
    def release(self):
        raise NotImplementedError

    @property
    def width(self):
        return 0

    @property
    def height(self):
        return 0

    @property
    def fps(self):
        return 25.0

    @property
    def is_started(self):
        return False


# ==================== Ascend-FFmpeg + VPC 解码器 ====================

class AscendFFmpegDecoder(BaseVideoDecoder):
    """Ascend-FFmpeg 硬解码 + VPC 硬件色彩转换

    VDEC 硬解码输出 NV12 → VPC 硬件转换 NV12→BGR（可选 + resize）
    全程 NPU 处理，CPU 零色彩转换开销。

    use_vpc=True（默认）: read_frame() 返回 BGR numpy（VPC 硬件转换）
    use_vpc=False: read_frame() 返回 NV12 dict，由调用方自行转换
    """

    # 进程级缓存：RTSP URL → 上次成功检测的编码格式
    _detected_codec_cache = {}

    def __init__(self, rtsp_url, device_id=0, channel_id=None, en_type="H264",
                 auto_detect_codec=True, use_vpc=True):
        self.rtsp_url = rtsp_url
        self.device_id = device_id
        self._channel_id = channel_id
        self._allocated_channel = None
        self._en_type = en_type
        self._auto_detect_codec = auto_detect_codec
        self._use_vpc = use_vpc and _HAS_ACL

        self._process = None
        self._width = 0
        self._height = 0
        self._fps = 25.0
        self._started = False
        self._frame_count = 0
        self._frame_size = 0
        self._first_frame = None

        # VPC 资源（懒初始化）
        self._vpc_stream = None
        self._vpc_desc = None
        self._vpc_inited = False

        # VPC 缓存 device buffer + pic_desc
        self._dev_nv12 = None
        self._dev_nv12_size = 0
        self._nv12_desc = None

        self._dev_bgr = None
        self._dev_bgr_size = 0
        self._bgr_desc = None
        self._bgr_w_stride = 0

        self._dev_bgr_rsz = None
        self._dev_bgr_rsz_size = 0
        self._bgr_rsz_desc = None
        self._bgr_rsz_w_stride = 0

        self._resize_cfg = None
        self._roi = None

        # 缓存尺寸标记
        self._cached_nv12_w = 0
        self._cached_nv12_h = 0
        self._cached_bgr_w = 0
        self._cached_bgr_h = 0
        self._cached_rsz_w = 0
        self._cached_rsz_h = 0

    # ==================== VPC 内部方法 ====================

    def _vpc_init(self):
        """初始化 VPC 通道和 stream"""
        if self._vpc_inited:
            return True
        try:
            acl.rt.set_device(self.device_id)
            self._vpc_stream, ret = acl.rt.create_stream()
            if ret != 0:
                print(f"[VPC] create_stream failed: ret={ret}")
                return False
            self._vpc_desc = acl.media.dvpp_create_channel_desc()
            ret = acl.media.dvpp_create_channel(self._vpc_desc)
            if ret != 0:
                print(f"[VPC] create_channel failed: ret={ret}")
                return False
            self._vpc_inited = True
            print(f"[VPC] 初始化成功 (device={self.device_id})")
            return True
        except Exception as e:
            print(f"[VPC] 初始化异常: {e}")
            return False

    def _vpc_ensure_nv12(self, width, height):
        """确保 NV12 device buffer 就绪"""
        if self._cached_nv12_w == width and self._cached_nv12_h == height:
            return True
        w_stride = _align_up(width, 16)
        h_stride = _align_up(height, 2)
        size = w_stride * h_stride * 3 // 2

        if self._dev_nv12 is not None:
            acl.media.dvpp_free(self._dev_nv12)
        if self._nv12_desc is not None:
            acl.media.dvpp_destroy_pic_desc(self._nv12_desc)

        self._dev_nv12, ret = acl.media.dvpp_malloc(size)
        if ret != 0:
            return False

        self._nv12_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(self._nv12_desc, self._dev_nv12)
        acl.media.dvpp_set_pic_desc_format(self._nv12_desc, FMT_NV12)
        acl.media.dvpp_set_pic_desc_width(self._nv12_desc, width)
        acl.media.dvpp_set_pic_desc_height(self._nv12_desc, height)
        acl.media.dvpp_set_pic_desc_width_stride(self._nv12_desc, w_stride)
        acl.media.dvpp_set_pic_desc_height_stride(self._nv12_desc, h_stride)
        acl.media.dvpp_set_pic_desc_size(self._nv12_desc, size)

        self._dev_nv12_size = size
        self._cached_nv12_w = width
        self._cached_nv12_h = height
        return True

    def _vpc_ensure_bgr(self, width, height):
        """确保 BGR device buffer 就绪"""
        if self._cached_bgr_w == width and self._cached_bgr_h == height:
            return True
        w_stride = _align_up(width, 16) * 3
        h_stride = _align_up(height, 2)
        size = w_stride * h_stride

        if self._dev_bgr is not None:
            acl.media.dvpp_free(self._dev_bgr)
        if self._bgr_desc is not None:
            acl.media.dvpp_destroy_pic_desc(self._bgr_desc)

        self._dev_bgr, ret = acl.media.dvpp_malloc(size)
        if ret != 0:
            return False

        self._bgr_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(self._bgr_desc, self._dev_bgr)
        acl.media.dvpp_set_pic_desc_format(self._bgr_desc, FMT_BGR)
        acl.media.dvpp_set_pic_desc_width(self._bgr_desc, width)
        acl.media.dvpp_set_pic_desc_height(self._bgr_desc, height)
        acl.media.dvpp_set_pic_desc_width_stride(self._bgr_desc, w_stride)
        acl.media.dvpp_set_pic_desc_height_stride(self._bgr_desc, h_stride)
        acl.media.dvpp_set_pic_desc_size(self._bgr_desc, size)

        self._dev_bgr_size = size
        self._bgr_w_stride = w_stride
        self._cached_bgr_w = width
        self._cached_bgr_h = height
        return True

    def _vpc_ensure_bgr_rsz(self, dst_w, dst_h):
        """确保 BGR resize 输出 device buffer 就绪"""
        if self._cached_rsz_w == dst_w and self._cached_rsz_h == dst_h:
            return True
        w_stride = _align_up(dst_w, 16) * 3
        h_stride = _align_up(dst_h, 2)
        size = w_stride * h_stride

        if self._dev_bgr_rsz is not None:
            acl.media.dvpp_free(self._dev_bgr_rsz)
        if self._bgr_rsz_desc is not None:
            acl.media.dvpp_destroy_pic_desc(self._bgr_rsz_desc)

        self._dev_bgr_rsz, ret = acl.media.dvpp_malloc(size)
        if ret != 0:
            return False

        self._bgr_rsz_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(self._bgr_rsz_desc, self._dev_bgr_rsz)
        acl.media.dvpp_set_pic_desc_format(self._bgr_rsz_desc, FMT_BGR)
        acl.media.dvpp_set_pic_desc_width(self._bgr_rsz_desc, dst_w)
        acl.media.dvpp_set_pic_desc_height(self._bgr_rsz_desc, dst_h)
        acl.media.dvpp_set_pic_desc_width_stride(self._bgr_rsz_desc, w_stride)
        acl.media.dvpp_set_pic_desc_height_stride(self._bgr_rsz_desc, h_stride)
        acl.media.dvpp_set_pic_desc_size(self._bgr_rsz_desc, size)

        self._dev_bgr_rsz_size = size
        self._bgr_rsz_w_stride = w_stride
        self._cached_rsz_w = dst_w
        self._cached_rsz_h = dst_h
        return True

    def _vpc_upload_nv12(self, nv12_data, width, height):
        """NV12 host 数据上传到 device buffer"""
        if isinstance(nv12_data, bytes):
            nv12_np = np.frombuffer(nv12_data, dtype=np.uint8)
        else:
            nv12_np = nv12_data

        raw_size = width * height * 3 // 2
        w_stride = _align_up(width, 16)
        h_stride = _align_up(height, 2)
        stride_size = w_stride * h_stride * 3 // 2

        if stride_size == raw_size and len(nv12_np) == raw_size:
            acl.rt.memcpy(int(self._dev_nv12), stride_size,
                          int(nv12_np.ctypes.data), raw_size, ACL_H2D)
        else:
            # 向量化 stride 对齐，零 Python 循环
            aligned = np.zeros(stride_size, dtype=np.uint8)
            y_plane = nv12_np[:height * width].reshape(height, width)
            uv_plane = nv12_np[height * width:].reshape(height // 2, width)
            aligned[:height * w_stride].reshape(height, w_stride)[:, :width] = y_plane
            uv_offset = h_stride * w_stride
            aligned[uv_offset:uv_offset + (height // 2) * w_stride].reshape(height // 2, w_stride)[:, :width] = uv_plane
            acl.rt.memcpy(int(self._dev_nv12), stride_size,
                          int(aligned.ctypes.data), stride_size, ACL_H2D)

    def _vpc_download_bgr(self, dev_bgr, width, height, bgr_size, w_stride):
        """从 device 下载 BGR 数据到 host numpy（向量化，零 Python 循环）"""
        buf = np.zeros(bgr_size, dtype=np.uint8)
        acl.rt.memcpy(int(buf.ctypes.data), bgr_size, int(dev_bgr), bgr_size, ACL_D2H)
        # reshape 利用 stride 对齐: w_stride = align_up(width, 16) * 3
        # 切片 [:, :width, :] 去掉每行 padding
        return buf[:height * w_stride].reshape(height, w_stride // 3, 3)[:, :width, :].copy()

    def _vpc_nv12_to_bgr(self, nv12_data, width, height):
        """VPC 硬件 NV12→BGR"""
        if not self._vpc_init():
            return None
        if not self._vpc_ensure_nv12(width, height):
            return None
        if not self._vpc_ensure_bgr(width, height):
            return None

        self._vpc_upload_nv12(nv12_data, width, height)
        acl.rt.memset(int(self._dev_bgr), self._dev_bgr_size, 0, self._dev_bgr_size)

        ret = acl.media.dvpp_vpc_convert_color_async(
            self._vpc_desc, self._nv12_desc, self._bgr_desc, self._vpc_stream)
        if ret != 0:
            return None
        acl.rt.synchronize_stream(self._vpc_stream)
        return self._vpc_download_bgr(self._dev_bgr, width, height,
                                       self._dev_bgr_size, self._bgr_w_stride)

    def _vpc_nv12_to_bgr_resized(self, nv12_data, src_w, src_h, dst_w, dst_h):
        """VPC 硬件 NV12→BGR + resize 一步完成"""
        if not self._vpc_init():
            return None
        if not self._vpc_ensure_nv12(src_w, src_h):
            return None
        if not self._vpc_ensure_bgr_rsz(dst_w, dst_h):
            return None

        self._vpc_upload_nv12(nv12_data, src_w, src_h)
        acl.rt.memset(int(self._dev_bgr_rsz), self._dev_bgr_rsz_size, 0, self._dev_bgr_rsz_size)

        if self._roi is not None:
            acl.media.dvpp_destroy_roi_config(self._roi)
        self._roi = acl.media.dvpp_create_roi_config(0, src_w - 1, 0, src_h - 1)

        if self._resize_cfg is None:
            self._resize_cfg = acl.media.dvpp_create_resize_config()
            acl.media.dvpp_set_resize_config_interpolation(self._resize_cfg, 0)

        ret = acl.media.dvpp_vpc_crop_resize_async(
            self._vpc_desc, self._nv12_desc, self._bgr_rsz_desc,
            self._roi, self._resize_cfg, self._vpc_stream)
        if ret != 0:
            return None
        acl.rt.synchronize_stream(self._vpc_stream)
        return self._vpc_download_bgr(self._dev_bgr_rsz, dst_w, dst_h,
                                       self._dev_bgr_rsz_size, self._bgr_rsz_w_stride)

    def _vpc_cleanup(self):
        """释放 VPC 资源"""
        if not self._vpc_inited:
            return
        for desc in (self._nv12_desc, self._bgr_desc, self._bgr_rsz_desc):
            if desc is not None:
                acl.media.dvpp_destroy_pic_desc(desc)
        for dev in (self._dev_nv12, self._dev_bgr, self._dev_bgr_rsz):
            if dev is not None:
                acl.media.dvpp_free(dev)
        if self._resize_cfg is not None:
            acl.media.dvpp_destroy_resize_config(self._resize_cfg)
        if self._roi is not None:
            acl.media.dvpp_destroy_roi_config(self._roi)
        if self._vpc_desc is not None:
            acl.media.dvpp_destroy_channel(self._vpc_desc)
        if self._vpc_stream is not None:
            acl.rt.destroy_stream(self._vpc_stream)

        self._vpc_inited = False
        self._vpc_stream = None
        self._vpc_desc = None
        self._dev_nv12 = None
        self._dev_bgr = None
        self._dev_bgr_rsz = None
        self._nv12_desc = None
        self._bgr_desc = None
        self._bgr_rsz_desc = None
        self._resize_cfg = None
        self._roi = None
        self._cached_nv12_w = 0
        self._cached_nv12_h = 0
        self._cached_bgr_w = 0
        self._cached_bgr_h = 0
        self._cached_rsz_w = 0
        self._cached_rsz_h = 0

    # ==================== VDEC 方法 ====================

    def start(self) -> bool:
        """启动 ffmpeg 硬解码管道"""
        if self._started:
            return True

        # Step 1: 探测视频信息
        print(f"[Ascend-FFmpeg] 探测视频源: {self.rtsp_url}")
        info = _probe_video_info(self.rtsp_url)

        if info:
            self._width = info['width']
            self._height = info['height']
            self._fps = info['fps']

            if self._auto_detect_codec:
                detected = CODEC_NAME_TO_EN_TYPE.get(info['codec_name'])
                if detected and detected != self._en_type:
                    print(f"[Ascend-FFmpeg] 自动检测编码: {info['codec_name']} → {detected} (配置: {self._en_type})")
                    self._en_type = detected
                AscendFFmpegDecoder._detected_codec_cache[self.rtsp_url] = self._en_type
            print(f"[Ascend-FFmpeg] 视频信息: {self._width}x{self._height} @ {self._fps:.1f}fps, codec={info.get('codec_name', 'unknown')}")
        else:
            cached_en_type = AscendFFmpegDecoder._detected_codec_cache.get(self.rtsp_url)
            if cached_en_type:
                print(f"[Ascend-FFmpeg] ffprobe 失败，使用缓存的编码: {cached_en_type}")
                self._en_type = cached_en_type
            else:
                print("[Ascend-FFmpeg] ffprobe 失败，使用配置默认值")
            self._width = 1920
            self._height = 1080
            self._fps = 25.0

        # Step 2: NV12 帧大小
        self._frame_size = self._width * self._height * 3 // 2

        # Step 3: 确定 channel_id
        if self._channel_id is not None:
            ch_id = self._channel_id
        else:
            ch_id = _allocate_channel_id()
            self._allocated_channel = ch_id
        print(f"[Ascend-FFmpeg] VDEC channel_id={ch_id}"
              f"{' (自动分配)' if self._allocated_channel is not None else ''}")

        # Step 4: 检查解码器可用性
        ascend_codec = CODEC_MAP.get(self._en_type, "h264_ascend")
        if not _check_decoder_available(ascend_codec):
            print(f"[Ascend-FFmpeg] 解码器 '{ascend_codec}' 不可用，硬解码不支持 {self._en_type}")
            return False

        # Step 5: 构建 ffmpeg 命令（固定 NV12 输出）
        cmd = [_get_ffmpeg_bin()]

        if self.rtsp_url.startswith('rtsp://'):
            cmd.extend([
                '-rtsp_transport', 'tcp',
                '-stimeout', '5000000',
                '-fflags', '+genpts+discardcorrupt',
            ])

        cmd.extend([
            '-hwaccel', 'ascend',
            '-c:v', ascend_codec,
            '-device_id', str(self.device_id),
            '-channel_id', str(ch_id),
        ])

        cmd.append('-i')
        cmd.append(self.rtsp_url)

        cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'nv12',
            '-loglevel', 'error',
            'pipe:1'
        ])

        vpc_label = "VPC硬件转换" if self._use_vpc else "CPU转换"
        print(f"[Ascend-FFmpeg] 启动硬解码管道: {ascend_codec}, pix_fmt=nv12, {vpc_label}")

        # Step 6: 启动 ffmpeg 子进程
        env = os.environ.copy()
        ffmpeg_dir = _find_ffmpeg()
        if ffmpeg_dir:
            lib_dir = os.path.join(os.path.dirname(ffmpeg_dir), 'lib')
            if os.path.isdir(lib_dir):
                env['LD_LIBRARY_PATH'] = lib_dir + ':' + env.get('LD_LIBRARY_PATH', '')
            env['PATH'] = ffmpeg_dir + ':' + env.get('PATH', '')

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 2,
                env=env,
            )
        except FileNotFoundError:
            print("[Ascend-FFmpeg] ffmpeg 未安装!")
            return False
        except Exception as e:
            print(f"[Ascend-FFmpeg] 启动 ffmpeg 失败: {e}")
            return False

        # Step 7: 等待 ffmpeg 初始化并读取首帧验证
        import time as _time
        _time.sleep(0.5)

        if self._process.poll() is not None:
            stderr_output = ""
            try:
                stderr_output = self._process.stderr.read().decode('utf-8', errors='replace')
            except:
                pass
            print(f"[Ascend-FFmpeg] ffmpeg 启动后立即退出 (ret={self._process.returncode})")
            if stderr_output:
                print(f"[Ascend-FFmpeg] stderr: {stderr_output[:500]}")
            self._process = None
            return False

        try:
            first_frame_data = self._process.stdout.read(self._frame_size)
            if len(first_frame_data) != self._frame_size:
                stderr_output = ""
                try:
                    stderr_output = self._process.stderr.read().decode('utf-8', errors='replace')
                except:
                    pass
                print(f"[Ascend-FFmpeg] 首帧读取失败 (got {len(first_frame_data)}/{self._frame_size} bytes)")
                if stderr_output:
                    print(f"[Ascend-FFmpeg] ffmpeg stderr: {stderr_output[:500]}")
                self._process.kill()
                self._process.wait(timeout=3)
                self._process = None
                return False

            self._first_frame = first_frame_data
        except Exception as e:
            print(f"[Ascend-FFmpeg] 首帧读取异常: {e}")
            try:
                self._process.kill()
                self._process.wait(timeout=3)
            except:
                pass
            self._process = None
            return False

        self._started = True
        print(f"[Ascend-FFmpeg] 硬解码管道就绪: {self._width}x{self._height} @ {self._fps:.1f}fps, "
              f"编码={self._en_type}, 色彩转换={vpc_label}")
        return True

    def _read_raw_nv12(self):
        """读取一帧 NV12 raw bytes（内部用）"""
        if not self._started:
            return None

        if self._first_frame is not None:
            raw = self._first_frame
            self._first_frame = None
        else:
            try:
                raw = self._process.stdout.read(self._frame_size)
            except Exception:
                return None

            if len(raw) != self._frame_size:
                return None

        self._frame_count += 1
        return raw

    def read_frame(self):
        """读取一帧 BGR 图像

        use_vpc=True: VPC 硬件 NV12→BGR（零 CPU 色彩转换）
        use_vpc=False: CPU cv2.cvtColor NV12→BGR
        返回 BGR numpy (H, W, 3)，或 None
        """
        raw = self._read_raw_nv12()
        if raw is None:
            return None

        if self._use_vpc:
            bgr = self._vpc_nv12_to_bgr(raw, self._width, self._height)
            if bgr is not None:
                return bgr
            # VPC 失败，回退 CPU
            if self._frame_count <= 2:
                print("[Ascend-FFmpeg] VPC 转换失败，回退 CPU")
            self._use_vpc = False
            self._vpc_cleanup()

        return _nv12_to_bgr_cpu(raw, self._width, self._height)

    def read_frame_bgr(self):
        """读取一帧 BGR 图像（兼容旧接口，等同 read_frame()）"""
        return self.read_frame()

    def read_frame_resized(self, dst_w, dst_h):
        """读取一帧 BGR 图像并 resize（VPC 一步完成 NV12→BGR+resize）

        仅 use_vpc=True 时有效，否则回退 read_frame() + CPU resize。
        返回 BGR numpy (dst_h, dst_w, 3)，或 None
        """
        raw = self._read_raw_nv12()
        if raw is None:
            return None

        if self._use_vpc:
            bgr = self._vpc_nv12_to_bgr_resized(raw, self._width, self._height, dst_w, dst_h)
            if bgr is not None:
                return bgr
            if self._frame_count <= 2:
                print("[Ascend-FFmpeg] VPC crop_resize 失败，回退 CPU")
            self._use_vpc = False
            self._vpc_cleanup()

        # CPU 回退
        import cv2
        bgr = _nv12_to_bgr_cpu(raw, self._width, self._height)
        return cv2.resize(bgr, (dst_w, dst_h))

    def read_frame_nv12(self):
        """读取一帧 NV12 原始数据（跳过色彩转换）

        返回 dict: {'data': bytes, 'width': int, 'height': int, 'format': 'nv12'}
        或 None
        """
        raw = self._read_raw_nv12()
        if raw is None:
            return None
        return {'data': raw, 'width': self._width, 'height': self._height, 'format': 'nv12'}

    def release(self):
        """释放所有资源（VPC + VDEC）"""
        self._started = False
        self._vpc_cleanup()
        if self._process:
            try:
                self._process.stdout.close()
                self._process.stderr.close()
                self._process.kill()
                self._process.wait(timeout=3)
            except Exception:
                pass
            self._process = None
        if self._allocated_channel is not None:
            _release_channel_id(self._allocated_channel)
            self._allocated_channel = None
        self._first_frame = None
        print(f"[Ascend-FFmpeg] 资源已释放 (解码帧数: {self._frame_count}, VPC={'已用' if not self._use_vpc or self._vpc_inited else '未用'})")

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def fps(self):
        return self._fps

    @property
    def is_started(self):
        return self._started and self._process is not None


# ==================== acl 原生 VDEC + VPC 解码器 ====================

try:
    import av as _av
    _HAS_PYAV = True
except ImportError:
    _HAS_PYAV = False

# VDEC 编码类型常量
ENTYPE_H265_MAIN = 0
ENTYPE_H264_BASE = 1
ENTYPE_H264_MAIN = 2
ENTYPE_H264_HIGH = 3

# en_type 字符串 → 整数映射
_EN_TYPE_MAP = {
    "H265": ENTYPE_H265_MAIN, "H264": ENTYPE_H264_MAIN,
    "h265": ENTYPE_H265_MAIN, "h264": ENTYPE_H264_MAIN,
    "HEVC": ENTYPE_H265_MAIN, "AVC": ENTYPE_H264_MAIN,
    0: 0, 1: 1, 2: 2, 3: 3,
}


def _get_h265_nal_type(data):
    """获取 H265 NAL unit type"""
    if len(data) < 5:
        return -1
    if data[:4] == b"\x00\x00\x00\x01":
        return (data[4] >> 1) & 0x3F
    elif data[:3] == b"\x00\x00\x01":
        return (data[3] >> 1) & 0x3F
    return -1


class AclVdecDecoder(BaseVideoDecoder):
    """acl 原生 VDEC + VPC 全 NPU 解码器

    PyAV 拉 RTSP → acl VDEC 解码(device NV12) → VPC crop_resize+convert_color(device BGR)

    read_frame():       返回 BGR numpy (VPC 硬件转换 + D2H，用于录制/MJPEG)
    read_frame_device(): 返回 (dev_nv12_ptr, size) (VPC resize 后的 NV12，零拷贝给推理)
    """

    def __init__(self, rtsp_url, device_id=0, channel_id=None, en_type="H264",
                 target_w=640, target_h=640, auto_detect_codec=True):
        self.rtsp_url = rtsp_url
        self.device_id = device_id
        self._channel_id = channel_id if channel_id is not None else 0
        self._en_type = _EN_TYPE_MAP.get(en_type, ENTYPE_H264_MAIN)
        self._auto_detect_codec = auto_detect_codec
        self._target_w = target_w
        self._target_h = target_h

        self._width = 0
        self._height = 0
        self._fps = 25.0
        self._started = False
        self._frame_count = 0

        # ACL 资源
        self._ctx = None
        self._stream = None
        self._acl_inited = False

        # VDEC 资源
        self._vdec_ch_desc = None
        self._frame_cfg = None
        self._input_buf = None
        self._cb_tid = None
        self._vdec_run = True
        self._vdec_cb_count = 0

        # VPC 资源
        self._vpc_stream = None
        self._vpc_ch = None
        self._vpc_inited = False

        # VPC 预分配输出 buffer
        self._dev_rsz_nv12 = None
        self._rsz_nv12_desc = None
        self._rsz_nv12_size = 0
        self._dev_bgr = None
        self._bgr_desc = None
        self._bgr_size = 0
        self._bgr_w_stride = 0

        # VPC 配置
        self._roi = None
        self._resize_cfg = None

        # PyAV demux 线程（独立线程拥有 container，避免跨线程阻塞）
        self._container = None
        self._video_stream = None
        self._is_h265 = False
        self._demux_thread = None
        self._demux_running = False
        self._nal_queue = queue.Queue(maxsize=32)  # NAL 包队列

        # 帧队列（回调 → 主线程）
        self._frame_queue = queue.Queue(maxsize=4)
        # 延迟销毁队列（回调存入 desc，主线程 destroy，避免回调线程 double free）
        self._desc_queue = queue.Queue(maxsize=64)

    # ==================== VDEC 回调 ====================

    def _vdec_callback(self, stream_desc, pic_desc, user_data):
        """VDEC 回调 — 延迟销毁 desc（主线程统一 destroy，避免回调线程 double free）"""
        self._vdec_cb_count += 1
        try:
            # 延迟销毁：存到 desc 队列，主线程统一 destroy
            try:
                self._desc_queue.put_nowait((stream_desc, pic_desc))
            except queue.Full:
                # desc 队列满，直接在回调中 destroy（有 double free 风险但比泄漏好）
                if stream_desc is not None:
                    acl.media.dvpp_destroy_stream_desc(stream_desc)
                if pic_desc is not None:
                    acl.media.dvpp_destroy_pic_desc(pic_desc)

            if pic_desc is not None:
                ret_code = acl.media.dvpp_get_pic_desc_ret_code(pic_desc)
                pic_data = acl.media.dvpp_get_pic_desc_data(pic_desc)
                pic_size = acl.media.dvpp_get_pic_desc_size(pic_desc)

                if ret_code == 0 and pic_data is not None and pic_size > 0 and user_data is not None and user_data >= 0:
                    # 有效图像帧 — 入队列，主线程用完 dvpp_free
                    try:
                        self._frame_queue.put_nowait({"buffer": pic_data, "size": pic_size})
                    except queue.Full:
                        # 队列满，丢弃最旧帧
                        try:
                            old = self._frame_queue.get_nowait()
                            acl.media.dvpp_free(old["buffer"])
                        except queue.Empty:
                            pass
                        self._frame_queue.put_nowait({"buffer": pic_data, "size": pic_size})
                else:
                    # 序列头空帧 / 解码失败 — 释放 device buffer
                    if pic_data is not None:
                        acl.media.dvpp_free(pic_data)
        except Exception as e:
            print(f"[AclVdec] 回调异常: {e}")

    def _cb_thread_func(self, args):
        """VDEC 回调线程"""
        timeout = args[0] if args else 100
        acl.rt.set_context(self._ctx)
        while self._vdec_run:
            acl.rt.process_report(timeout)

    # ==================== 启动 ====================

    def start(self) -> bool:
        if self._started:
            return True

        try:
            # Step 1: 探测视频信息
            if not self._probe_video():
                return False

            # Step 2: ACL 初始化
            if not self._init_acl():
                return False

            # Step 3: VDEC 通道
            if not self._init_vdec():
                return False

            # Step 4: VPC 通道 + 预分配 buffer
            if not self._init_vpc():
                return False

            # Step 5: 启动 demux 线程
            if not self._open_rtsp():
                return False

            # 预喂入几个 NAL 包，让 VDEC 开始解码
            try:
                _pre_sent = self._demux_and_send(max_frames=5)
                print(f"[AclVdec] 预喂入 {_pre_sent} 个 NAL 包")
            except Exception as e:
                print(f"[AclVdec] 预喂入异常: {e}")

            self._started = True
            print(f"[AclVdec] 全 NPU 解码器就绪: {self._width}x{self._height} → "
                  f"{self._target_w}x{self._target_h}, "
                  f"codec={'H265' if self._is_h265 else 'H264'}")
            return True
        except Exception as e:
            print(f"[AclVdec] 启动失败: {e}")
            self.release()
            return False

    def _probe_video(self):
        """用 PyAV 探测视频信息"""
        try:
            container = _av.open(self.rtsp_url, options={
                'rtsp_transport': 'tcp', 'stimeout': '5000000'})
            stream = container.streams.video[0]
            self._width = stream.codec_context.width
            self._height = stream.codec_context.height
            codec_name = stream.codec_context.name
            self._is_h265 = codec_name in ('hevc', 'h265')

            if self._auto_detect_codec:
                detected = ENTYPE_H265_MAIN if self._is_h265 else ENTYPE_H264_MAIN
                if detected != self._en_type:
                    print(f"[AclVdec] 自动检测编码: {codec_name} → "
                          f"{'H265' if detected == 0 else 'H264'} (配置: {self._en_type})")
                    self._en_type = detected

            try:
                fps_frac = stream.average_rate
                if fps_frac:
                    self._fps = float(fps_frac)
            except Exception:
                pass

            container.close()
            print(f"[AclVdec] 视频信息: {self._width}x{self._height} @ {self._fps:.1f}fps, "
                  f"codec={codec_name}")
            return True
        except Exception as e:
            print(f"[AclVdec] PyAV 探测失败: {e}")
            return False

    def _init_acl(self):
        """初始化 ACL 运行时"""
        try:
            ret = acl.init()
            if ret != 0 and ret != 1 and ret != 100002:
                # 0=成功, 1=已初始化(旧版), 100002=ACL_ERROR_REPEAT_INITIALIZE
                print(f"[AclVdec] acl.init 失败: ret={ret}")
                return False
            self._acl_inited = (ret == 0)  # 仅首次初始化成功才标记

            acl.rt.set_device(self.device_id)
            self._ctx, ret = acl.rt.create_context(self.device_id)
            if ret != 0:
                print(f"[AclVdec] create_context 失败: ret={ret}")
                return False

            self._stream, ret = acl.rt.create_stream()
            if ret != 0:
                print(f"[AclVdec] create_stream 失败: ret={ret}")
                return False

            return True
        except Exception as e:
            print(f"[AclVdec] ACL 初始化异常: {e}")
            return False

    def _init_vdec(self):
        """初始化 VDEC 通道"""
        try:
            # 启动回调线程（保持引用防 GC）
            self._cb_thread_ref = self._cb_thread_func
            self._cb_tid, ret = acl.util.start_thread(self._cb_thread_ref, [100])
            if ret != 0:
                print(f"[AclVdec] start_thread 失败: ret={ret}")
                return False

            # subscribe_report
            ret = acl.rt.subscribe_report(self._cb_tid, self._stream)
            if ret != 0:
                print(f"[AclVdec] subscribe_report 失败: ret={ret}")
                return False
            time.sleep(0.3)

            # 创建 VDEC 通道
            ch_desc = acl.media.vdec_create_channel_desc()
            acl.media.vdec_set_channel_desc_channel_id(ch_desc, self._channel_id)
            acl.media.vdec_set_channel_desc_thread_id(ch_desc, self._cb_tid)
            # 保持回调引用，防止 GC 回收绑定方法导致回调失效
            self._vdec_cb_ref = self._vdec_callback
            acl.media.vdec_set_channel_desc_callback(ch_desc, self._vdec_cb_ref)
            acl.media.vdec_set_channel_desc_entype(ch_desc, self._en_type)
            acl.media.vdec_set_channel_desc_out_pic_format(ch_desc, FMT_NV12)

            out_mode = acl.media.vdec_get_channel_desc_out_mode(ch_desc)
            acl.media.vdec_set_channel_desc_out_mode(ch_desc, out_mode)

            ret = acl.media.vdec_create_channel(ch_desc)
            if ret != 0:
                # 尝试其他编码类型
                for alt in [ENTYPE_H264_HIGH, ENTYPE_H264_MAIN, ENTYPE_H265_MAIN]:
                    if alt == self._en_type:
                        continue
                    acl.media.vdec_set_channel_desc_entype(ch_desc, alt)
                    ret = acl.media.vdec_create_channel(ch_desc)
                    if ret == 0:
                        self._en_type = alt
                        break
            if ret != 0:
                print(f"[AclVdec] vdec_create_channel 失败: ret={ret}")
                return False

            self._vdec_ch_desc = ch_desc
            self._frame_cfg = acl.media.vdec_create_frame_config()

            # 预分配 input buffer（全局复用）
            vdec_w = _align_up(self._width, 16)
            vdec_h = _align_up(self._height, 2)
            vdec_size = vdec_w * vdec_h * 3 // 2
            # input buffer 需要足够大容纳最大 NAL 包
            max_input = max(vdec_size, 512 * 1024)  # 至少 512KB
            self._input_buf, ret = acl.media.dvpp_malloc(max_input)
            if ret != 0:
                print(f"[AclVdec] dvpp_malloc input_buf 失败: ret={ret}")
                return False
            self._vdec_out_size = vdec_size

            return True
        except Exception as e:
            print(f"[AclVdec] VDEC 初始化异常: {e}")
            return False

    def _init_vpc(self):
        """初始化 VPC 通道 + 预分配输出 buffer"""
        try:
            self._vpc_stream, ret = acl.rt.create_stream()
            if ret != 0:
                return False
            self._vpc_ch = acl.media.dvpp_create_channel_desc()
            ret = acl.media.dvpp_create_channel(self._vpc_ch)
            if ret != 0:
                print(f"[AclVdec] VPC create_channel 失败: ret={ret}")
                return False
            self._vpc_inited = True

            # 预分配 resize 输出 NV12 buffer (640x640)
            rsz_w = _align_up(self._target_w, 16)
            rsz_h = _align_up(self._target_h, 2)
            self._rsz_nv12_size = rsz_w * rsz_h * 3 // 2
            self._dev_rsz_nv12, ret = acl.media.dvpp_malloc(self._rsz_nv12_size)
            if ret != 0:
                return False
            self._rsz_nv12_desc = acl.media.dvpp_create_pic_desc()
            acl.media.dvpp_set_pic_desc_data(self._rsz_nv12_desc, self._dev_rsz_nv12)
            acl.media.dvpp_set_pic_desc_format(self._rsz_nv12_desc, FMT_NV12)
            acl.media.dvpp_set_pic_desc_width(self._rsz_nv12_desc, self._target_w)
            acl.media.dvpp_set_pic_desc_height(self._rsz_nv12_desc, self._target_h)
            acl.media.dvpp_set_pic_desc_width_stride(self._rsz_nv12_desc, rsz_w)
            acl.media.dvpp_set_pic_desc_height_stride(self._rsz_nv12_desc, rsz_h)
            acl.media.dvpp_set_pic_desc_size(self._rsz_nv12_desc, self._rsz_nv12_size)

            # 预分配 BGR 输出 buffer (640x640)
            bgr_w = _align_up(self._target_w, 16) * 3
            bgr_h = _align_up(self._target_h, 2)
            self._bgr_size = bgr_w * bgr_h
            self._bgr_w_stride = bgr_w
            self._dev_bgr, ret = acl.media.dvpp_malloc(self._bgr_size)
            if ret != 0:
                return False
            self._bgr_desc = acl.media.dvpp_create_pic_desc()
            acl.media.dvpp_set_pic_desc_data(self._bgr_desc, self._dev_bgr)
            acl.media.dvpp_set_pic_desc_format(self._bgr_desc, FMT_BGR)
            acl.media.dvpp_set_pic_desc_width(self._bgr_desc, self._target_w)
            acl.media.dvpp_set_pic_desc_height(self._bgr_desc, self._target_h)
            acl.media.dvpp_set_pic_desc_width_stride(self._bgr_desc, bgr_w)
            acl.media.dvpp_set_pic_desc_height_stride(self._bgr_desc, bgr_h)
            acl.media.dvpp_set_pic_desc_size(self._bgr_desc, self._bgr_size)

            # ROI + resize config
            self._roi = acl.media.dvpp_create_roi_config(
                0, self._width - 1, 0, self._height - 1)
            self._resize_cfg = acl.media.dvpp_create_resize_config()
            acl.media.dvpp_set_resize_config_interpolation(self._resize_cfg, 0)

            return True
        except Exception as e:
            print(f"[AclVdec] VPC 初始化异常: {e}")
            return False

    def _open_rtsp(self):
        """启动独立 demux 线程拉 RTSP 流

        关键：PyAV container 必须在同一线程中打开和使用，
        跨线程调用 demux 会阻塞。所以 demux 线程自己拥有 container，
        通过 NAL 队列传递数据给 read_frame 线程。
        """
        self._demux_running = True
        self._demux_thread = threading.Thread(target=self._demux_loop, daemon=True)
        self._demux_thread.start()

        # 等待 demux 线程产出首批 NAL 包（最多 5 秒）
        for _ in range(50):
            if not self._nal_queue.empty():
                break
            if not self._demux_running:
                return False
            time.sleep(0.1)

        if self._nal_queue.empty():
            print(f"[AclVdec] demux 线程启动后无 NAL 数据")
            return False

        print(f"[AclVdec] demux 线程已启动，NAL 队列有数据")
        return True

    def _demux_loop(self):
        """demux 线程主循环：PyAV demux → NAL 队列

        此线程独占 PyAV container，不与其他线程共享。
        """
        container = None
        try:
            container = _av.open(self.rtsp_url, options={
                'rtsp_transport': 'tcp', 'stimeout': '5000000'})
            video_stream = container.streams.video[0]

            for packet in container.demux([video_stream]):
                if not self._demux_running:
                    break
                nal = bytes(packet)
                if len(nal) == 0:
                    continue
                # 队列满时阻塞等待（跟流速率，避免空转丢包）
                try:
                    self._nal_queue.put(nal, timeout=0.5)
                except queue.Full:
                    # 超时仍未消费，丢弃当前包
                    pass

        except _av.error.EOFError:
            print(f"[AclVdec] demux 线程: RTSP 流 EOF")
        except Exception as e:
            print(f"[AclVdec] demux 线程异常: {e}")
        finally:
            self._demux_running = False
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

    def _stop_demux(self):
        """停止 demux 线程"""
        self._demux_running = False
        if self._demux_thread is not None:
            self._demux_thread.join(timeout=3)
            self._demux_thread = None
        # 清空 NAL 队列
        while not self._nal_queue.empty():
            try:
                self._nal_queue.get_nowait()
            except queue.Empty:
                break

    # ==================== 帧读取 ====================

    def _send_nal_to_vdec(self, nal_data, user_data):
        """发送 NAL 包到 VDEC 解码

        关键：vdec_send_frame 后不 destroy stream_desc/pic_desc，
        VDEC 异步获取所有权，回调中负责 destroy。
        """
        nal_np = np.frombuffer(nal_data, dtype=np.uint8)
        nal_len = nal_np.nbytes
        if nal_len == 0:
            return False

        # memcpy H2D
        acl.rt.memcpy(int(self._input_buf), nal_len,
                      int(nal_np.ctypes.data), nal_len, ACL_H2D)

        # 创建 stream_desc
        sd = acl.media.dvpp_create_stream_desc()
        acl.media.dvpp_set_stream_desc_data(sd, self._input_buf)
        acl.media.dvpp_set_stream_desc_size(sd, nal_len)

        # 创建 output pic_desc
        dev_out, ret = acl.media.dvpp_malloc(self._vdec_out_size)
        if ret != 0:
            acl.media.dvpp_destroy_stream_desc(sd)
            return False
        pd = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(pd, dev_out)
        acl.media.dvpp_set_pic_desc_size(pd, self._vdec_out_size)
        acl.media.dvpp_set_pic_desc_format(pd, FMT_NV12)

        # vdec_send_frame — VDEC 获取 sd/pd 所有权，回调中 destroy
        ret = acl.media.vdec_send_frame(
            self._vdec_ch_desc, sd, pd, self._frame_cfg, user_data)
        if ret != 0:
            # send_frame 失败，VDEC 未获取所有权，需手动释放
            acl.media.dvpp_destroy_stream_desc(sd)
            acl.media.dvpp_destroy_pic_desc(pd)
            acl.media.dvpp_free(dev_out)
            return False
        return True

    def _flush_desc_queue(self):
        """主线程中统一销毁回调存入的 stream_desc/pic_desc"""
        while not self._desc_queue.empty():
            try:
                sd, pd = self._desc_queue.get_nowait()
                if sd is not None:
                    acl.media.dvpp_destroy_stream_desc(sd)
                if pd is not None:
                    acl.media.dvpp_destroy_pic_desc(pd)
            except queue.Empty:
                break

    def _demux_and_send(self, max_frames=1):
        """从 NAL 队列取包并发送到 VDEC

        demux 线程持续往 NAL 队列放包，此方法从队列取包送 VDEC。
        """
        sent = 0
        try:
            while sent < max_frames:
                try:
                    nal = self._nal_queue.get(timeout=0.1)
                except queue.Empty:
                    break

                if len(nal) == 0:
                    continue

                # H265 序列头用负 user_data 标记
                if self._is_h265:
                    nt = _get_h265_nal_type(nal)
                    if nt in (32, 33, 34):
                        self._send_nal_to_vdec(nal, -(nt + 1))
                        continue

                if self._send_nal_to_vdec(nal, self._frame_count):
                    self._frame_count += 1
                    sent += 1

        except Exception as e:
            if self._frame_count > 0:
                print(f"[AclVdec] NAL 队列读取异常: {e}")
        return sent

    def _get_decoded_frame(self, timeout=1.0):
        """从帧队列获取已解码的 NV12 device buffer"""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _vpc_process(self, nv12_dev_buf, nv12_size):
        """VPC crop_resize NV12→640x640 NV12 + convert_color→BGR

        Args:
            nv12_dev_buf: VDEC 输出的 NV12 device buffer 指针
            nv12_size: buffer 大小

        Returns:
            BGR numpy (target_h, target_w, 3)，或 None
        """
        # 构造 VPC 输入 pic_desc（VDEC 输出）
        vpc_in = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(vpc_in, nv12_dev_buf)
        acl.media.dvpp_set_pic_desc_format(vpc_in, FMT_NV12)
        acl.media.dvpp_set_pic_desc_width(vpc_in, self._width)
        acl.media.dvpp_set_pic_desc_height(vpc_in, self._height)
        vdec_w = _align_up(self._width, 16)
        vdec_h = _align_up(self._height, 2)
        acl.media.dvpp_set_pic_desc_width_stride(vpc_in, vdec_w)
        acl.media.dvpp_set_pic_desc_height_stride(vpc_in, vdec_h)
        acl.media.dvpp_set_pic_desc_size(vpc_in, nv12_size)

        # Step 1: crop_resize NV12→640x640 NV12
        acl.rt.memset(int(self._dev_rsz_nv12), self._rsz_nv12_size,
                      0, self._rsz_nv12_size)
        ret = acl.media.dvpp_vpc_crop_resize_async(
            self._vpc_ch, vpc_in, self._rsz_nv12_desc,
            self._roi, self._resize_cfg, self._vpc_stream)
        acl.rt.synchronize_stream(self._vpc_stream)
        acl.media.dvpp_destroy_pic_desc(vpc_in)

        if ret != 0:
            print(f"[AclVdec] VPC crop_resize 失败: ret={ret}")
            return None

        # Step 2: convert_color NV12→BGR
        acl.rt.memset(int(self._dev_bgr), self._bgr_size,
                      0, self._bgr_size)
        ret = acl.media.dvpp_vpc_convert_color_async(
            self._vpc_ch, self._rsz_nv12_desc, self._bgr_desc, self._vpc_stream)
        acl.rt.synchronize_stream(self._vpc_stream)

        if ret != 0:
            print(f"[AclVdec] VPC convert_color 失败: ret={ret}")
            return None

        # Step 3: D2H
        bgr_buf = np.zeros(self._bgr_size, dtype=np.uint8)
        acl.rt.memcpy(int(bgr_buf.ctypes.data), self._bgr_size,
                      int(self._dev_bgr), self._bgr_size, ACL_D2H)

        # 去除 stride padding
        return bgr_buf[:self._target_h * self._bgr_w_stride].reshape(
            self._target_h, self._bgr_w_stride // 3, 3)[:, :self._target_w, :].copy()

    def _vpc_resize_nv12(self, nv12_dev_buf, nv12_size):
        """VPC crop_resize NV12→640x640 NV12（不转 BGR，用于推理零拷贝）

        Returns:
            (dev_rsz_nv12_ptr, rsz_nv12_size)，调用方负责 dvpp_free 原始 nv12_dev_buf
        """
        vpc_in = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(vpc_in, nv12_dev_buf)
        acl.media.dvpp_set_pic_desc_format(vpc_in, FMT_NV12)
        acl.media.dvpp_set_pic_desc_width(vpc_in, self._width)
        acl.media.dvpp_set_pic_desc_height(vpc_in, self._height)
        vdec_w = _align_up(self._width, 16)
        vdec_h = _align_up(self._height, 2)
        acl.media.dvpp_set_pic_desc_width_stride(vpc_in, vdec_w)
        acl.media.dvpp_set_pic_desc_height_stride(vpc_in, vdec_h)
        acl.media.dvpp_set_pic_desc_size(vpc_in, nv12_size)

        acl.rt.memset(int(self._dev_rsz_nv12), self._rsz_nv12_size,
                      0, self._rsz_nv12_size)
        ret = acl.media.dvpp_vpc_crop_resize_async(
            self._vpc_ch, vpc_in, self._rsz_nv12_desc,
            self._roi, self._resize_cfg, self._vpc_stream)
        acl.rt.synchronize_stream(self._vpc_stream)
        acl.media.dvpp_destroy_pic_desc(vpc_in)

        if ret != 0:
            return None, 0
        return self._dev_rsz_nv12, self._rsz_nv12_size

    def read_frame(self):
        """读取一帧 BGR 图像 (VPC 硬件转换 + D2H)

        Returns:
            BGR numpy (target_h, target_w, 3)，或 None（连续失败时）
        """
        if not self._started:
            return None

        # 调用方需确保当前线程有正确的 ACL context
        # 使用 set_context 切换到 DVPP 的 context（VPC 操作需要）
        # 调用方在推理前需恢复自己的 context
        if self._ctx is not None:
            acl.rt.set_context(self._ctx)
        else:
            acl.rt.set_device(self.device_id)

        _t0 = time.time()
        for attempt in range(3):
            sent = self._demux_and_send(max_frames=2)
            self._flush_desc_queue()

            frame_info = self._get_decoded_frame(timeout=0.5)
            _t3 = time.time()
            if frame_info is not None:
                bgr = self._vpc_process(frame_info["buffer"], frame_info["size"])
                acl.media.dvpp_free(frame_info["buffer"])

                if bgr is not None:
                    if self._frame_count <= 3:
                        print(f"[AclVdec] read_frame OK: total={(_t3-_t0)*1000:.0f}ms")
                    return bgr
                if attempt == 0:
                    print(f"[AclVdec] VPC 处理失败，重试...")
            else:
                if attempt == 0 and self._frame_count <= 5:
                    print(f"[AclVdec] 超时: sent={sent} cb={self._vdec_cb_count} nal_q={self._nal_queue.qsize()} demux_running={self._demux_running}")

        return None

    def read_frame_device(self):
        """读取一帧 NV12 device buffer (VPC resize 后，零拷贝给推理)

        Returns:
            dict: {'buffer': dev_nv12_ptr, 'size': int, 'format': 'nv12',
                   'width': int, 'height': int, 'vdec_buffer': ptr}
            或 None。调用方用完后需 dvpp_free(result['vdec_buffer'])
        """
        if not self._started:
            return None

        if self._ctx is not None:
            acl.rt.set_context(self._ctx)
        else:
            acl.rt.set_device(self.device_id)

        self._demux_and_send(max_frames=2)
        self._flush_desc_queue()

        frame_info = self._get_decoded_frame(timeout=0.5)
        if frame_info is None:
            return None

        dev_nv12, nv12_size = self._vpc_resize_nv12(
            frame_info["buffer"], frame_info["size"])

        if dev_nv12 is None:
            acl.media.dvpp_free(frame_info["buffer"])
            return None

        return {
            'buffer': dev_nv12,
            'size': nv12_size,
            'format': 'nv12',
            'width': self._target_w,
            'height': self._target_h,
            'vdec_buffer': frame_info["buffer"],  # 主线程用完 dvpp_free
        }

    def release(self):
        """释放所有资源"""
        self._started = False

        # 先停 demux 线程，不再有新 NAL 数据
        self._stop_demux()

        # 释放帧队列中残留的 VDEC buffer
        while not self._frame_queue.empty():
            try:
                f = self._frame_queue.get_nowait()
                acl.media.dvpp_free(f["buffer"])
            except queue.Empty:
                break

        # 销毁延迟队列中的 desc
        self._flush_desc_queue()

        # VDEC
        self._vdec_run = False
        time.sleep(0.5)
        if self._vdec_ch_desc is not None:
            acl.media.vdec_destroy_channel(self._vdec_ch_desc)
            self._vdec_ch_desc = None
        if self._frame_cfg is not None:
            acl.media.vdec_destroy_frame_config(self._frame_cfg)
            self._frame_cfg = None
        if self._cb_tid is not None:
            try:
                acl.util.stop_thread(self._cb_tid)
            except Exception:
                pass
            self._cb_tid = None
        if self._input_buf is not None:
            acl.media.dvpp_free(self._input_buf)
            self._input_buf = None

        # VPC
        if self._dev_rsz_nv12 is not None:
            acl.media.dvpp_free(self._dev_rsz_nv12)
            self._dev_rsz_nv12 = None
        if self._dev_bgr is not None:
            acl.media.dvpp_free(self._dev_bgr)
            self._dev_bgr = None
        if self._rsz_nv12_desc is not None:
            acl.media.dvpp_destroy_pic_desc(self._rsz_nv12_desc)
            self._rsz_nv12_desc = None
        if self._bgr_desc is not None:
            acl.media.dvpp_destroy_pic_desc(self._bgr_desc)
            self._bgr_desc = None
        if self._resize_cfg is not None:
            acl.media.dvpp_destroy_resize_config(self._resize_cfg)
            self._resize_cfg = None
        if self._roi is not None:
            acl.media.dvpp_destroy_roi_config(self._roi)
            self._roi = None
        if self._vpc_ch is not None and self._vpc_inited:
            acl.media.dvpp_destroy_channel(self._vpc_ch)
            self._vpc_inited = False
        if self._vpc_stream is not None:
            acl.rt.destroy_stream(self._vpc_stream)
            self._vpc_stream = None

        # ACL
        if self._stream is not None:
            acl.rt.destroy_stream(self._stream)
            self._stream = None
        if self._ctx is not None:
            acl.rt.destroy_context(self._ctx)
            self._ctx = None
        if self._acl_inited:
            # 只有自己初始化的 ACL 才 reset/finalize
            acl.rt.reset_device(self.device_id)
            acl.finalize()
            self._acl_inited = False

        print(f"[AclVdec] 资源已释放 (解码帧数: {self._frame_count})")

    @property
    def width(self):
        return self._target_w

    @property
    def height(self):
        return self._target_h

    @property
    def src_width(self):
        return self._width

    @property
    def src_height(self):
        return self._height

    @property
    def fps(self):
        return self._fps

    @property
    def is_started(self):
        return self._started


# ==================== 工厂函数 ====================

def create_dvpp_decoder(rtsp_url, device_id=0, channel_id=None, en_type="H264",
                         auto_detect_codec=True, use_vpc=True):
    """
    创建硬解码器（优先 acl 原生 VDEC，回退 Ascend-FFmpeg）

    Args:
        channel_id: VDEC 通道 ID。None=自动分配，int=指定通道。
                    Docker 多服务部署建议通过环境变量 DVPP_CHANNEL_ID 指定。
        use_vpc: True=VPC 硬件 NV12→BGR（推荐，零 CPU 色彩转换）
                 False=CPU cv2.cvtColor 转换

    Returns:
        AclVdecDecoder 或 AscendFFmpegDecoder 实例，或 None（启动失败）
    """
    # 优先 acl 原生 VDEC（全 NPU 链路，零 CPU 图像搬运）
    if _HAS_ACL and _HAS_PYAV:
        try:
            decoder = AclVdecDecoder(
                rtsp_url=rtsp_url,
                device_id=device_id,
                channel_id=channel_id,
                en_type=en_type,
                auto_detect_codec=auto_detect_codec,
            )
            if decoder.start():
                return decoder
            print("[create_dvpp_decoder] AclVdecDecoder 启动失败，回退 Ascend-FFmpeg")
            decoder.release()
        except Exception as e:
            print(f"[create_dvpp_decoder] AclVdecDecoder 异常: {e}，回退 Ascend-FFmpeg")

    # 回退 Ascend-FFmpeg
    decoder = AscendFFmpegDecoder(
        rtsp_url=rtsp_url,
        device_id=device_id,
        channel_id=channel_id,
        en_type=en_type,
        auto_detect_codec=auto_detect_codec,
        use_vpc=use_vpc,
    )
    if decoder.start():
        return decoder
    return None
