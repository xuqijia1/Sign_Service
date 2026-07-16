# Sign_Service - 标志牌识别服务
import os
import sys
import json
import logging
import platform
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局配置
CONFIG = None
VIDEO_STREAM = None
HTTP_SERVER = None

# ===================== CPU 亲和性设置（多服务部署时避免争抢CPU） =====================
def set_cpu_affinity(cpu_list=None):
    """
    设置进程的 CPU 亲和性，绑定到指定的 CPU 核心

    Args:
        cpu_list: CPU 核心列表，如 [52, 53]。如果为 None，从环境变量 CPU_CORES 读取

    环境变量:
        CPU_CORES: 逗号分隔的 CPU 核心列表，如 "52,53" 或 "52"
        CPU_START: 起始 CPU 核心号（用于多容器部署，自动分配）
    """
    try:
        import psutil

        # 获取要绑定的 CPU 核心
        if cpu_list is None:
            # 优先从环境变量读取
            cpu_cores_env = os.environ.get('CPU_CORES', '')
            cpu_start_env = os.environ.get('CPU_START', '')

            if cpu_cores_env:
                cpu_list = [int(x.strip()) for x in cpu_cores_env.split(',')]
            elif cpu_start_env:
                # 只指定起始核心，默认使用 2 个核心
                start = int(cpu_start_env)
                cpu_list = [start, start + 1]
            else:
                # 默认不设置
                return False

        # 设置 CPU 亲和性
        p = psutil.Process()
        p.cpu_affinity(cpu_list)

        # 同时设置主线程的调度亲和性（Linux）
        if platform.system() == 'Linux':
            try:
                os.sched_setaffinity(0, cpu_list)
            except Exception:
                pass

        logger.info(f"CPU 亲和性已设置: 核心 {cpu_list}")
        return True
    except ImportError:
        logger.warning("psutil 未安装，无法设置 CPU 亲和性")
        return False
    except Exception as e:
        logger.warning(f"设置 CPU 亲和性失败: {e}")
        return False

def load_config():
    """加载配置文件"""
    global CONFIG
    config_path = os.path.join(os.path.dirname(__file__), 'system_config.json')

    default_config = {
        "httpServerPort": 8090,
        "DeviceType": "gpu",
        "camera_url": "rtsp://admin:password@192.168.10.100/ch1/mian/av_stream",
        "model_path": "./sign.pt",
        "cls_model_path": "./sign_cls.pt",
        "cls_conf_threshold": 0.5,
        "CudaDevice": 0,
        "confidence_threshold": 0.5,
        "video_save_dir": "./videos",
        "client_url": "http://127.0.0.1:8090",
        "push_interval": 0.1,
        "cpu_cores": None
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
                logger.info(f"配置文件加载成功: {config_path}")
        except Exception as e:
            logger.error(f"配置文件加载失败，使用默认配置: {e}")
    else:
        logger.warning(f"配置文件不存在，使用默认配置: {config_path}")

    CONFIG = default_config
    return CONFIG

def main():
    """主入口"""
    global VIDEO_STREAM, HTTP_SERVER

    logger.info("=" * 50)
    logger.info("Sign_Service 标志牌识别服务启动")
    logger.info("=" * 50)

    # 加载配置
    config = load_config()

    # CPU 亲和性设置（从配置文件读取）
    cpu_cores = config.get('cpu_cores', None)
    if cpu_cores:
        set_cpu_affinity(cpu_cores)

    # 创建视频保存目录
    video_dir = config.get('video_save_dir', './videos')
    if not os.path.isabs(video_dir):
        video_dir = os.path.join(os.path.dirname(__file__), video_dir)
    os.makedirs(video_dir, exist_ok=True)

    # 启动视频流处理
    from VideoStream import VideoStream
    VIDEO_STREAM = VideoStream(config)
    VIDEO_STREAM.start()

    # 启动HTTP服务器
    from custom_http_server import run_server
    port = config.get('httpServerPort', 8090)
    run_server(port=port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("服务被用户中断")
    except Exception as e:
        logger.error(f"服务异常: {e}", exc_info=True)
    finally:
        if VIDEO_STREAM:
            VIDEO_STREAM.stop()
        logger.info("服务已停止")