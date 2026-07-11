# Sign_Service 标志牌识别服务

## 功能说明

K11考试系统第4题（标志牌识别）的AI服务端，基于sign.pt模型实现标志牌检测和识别。

## 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/Start?userid=xxx` | GET | 开始识别/录制 |
| `/api/Stop?userid=xxx` | GET | 停止识别/录制，返回视频路径 |
| `/api/sign_boxes` | GET | 获取当前检测框数据 |
| `/api/sign_results` | GET | 获取识别结果 |
| `/api/status` | GET | 获取服务状态 |

## 配置文件 (system_config.json)

```json
{
    "httpServerPort": 8530,           // HTTP服务端口
    "DeviceType": "gpu",              // 推理设备: gpu/cpu/ascend
    "camera_url": "rtsp://...",       // 摄像头地址
    "model_path": "./sign.pt",        // 模型路径
    "CudaDevice": 0,                  // CUDA设备编号
    "confidence_threshold": 0.5,      // 置信度阈值
    "video_save_dir": "./videos",     // 视频保存目录
    "client_url": "http://127.0.0.1:8090",  // 客户端推送地址
    "push_interval": 0.1              // 推送间隔(秒)
}
```

## 标志牌类别 (24种)

- **禁止类**: prohibit_switch_on, prohibit_start, prohibit_approach, prohibit_climb, prohibit_touch, prohibit_enter
- **警告类**: warning_attention, warning_electric, warning_cable, warning_fire, warning_auto_start, warning_fall
- **指令类**: mandatory_grounding, mandatory_helmet, mandatory_clothing, mandatory_shoes, mandatory_gloves, mandatory_belt
- **提示类**: info_three_phase, info_work_here, info_emergency_stop, info_overvoltage, info_live_work, info_anti_interference

## 启动方式

```bash
# Windows
start.bat

# 或直接运行
python main.py
```

## 依赖

- Python 3.8+
- ultralytics
- opencv-python
- requests