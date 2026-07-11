#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量转换 PT 模型为 ONNX 格式（全精度FP32，适配昇腾NPU）
鲲鹏CPU上FP16运算极慢，使用FP32预处理更快
"""

import os
import glob
from ultralytics import YOLO


def convert_pt_to_onnx(pt_path, imgsz=640, half=False, device=0):
    """转换单个 PT 模型为 ONNX（默认FP32）"""
    print(f"🔄 正在转换: {pt_path}")

    try:
        model = YOLO(pt_path)
        onnx_path = model.export(
            format='onnx',
            imgsz=imgsz,
            batch=1,
            opset=12,
            simplify=True,
            half=half,  # 默认False，使用FP32
            device=device,
            dynamic=False
        )
        print(f"✅ 导出成功: {onnx_path} (FP32)")
        return True
    except Exception as e:
        print(f"❌ 导出失败: {pt_path}, 错误: {e}")
        return False


def find_and_convert_all(root_dir=".", imgsz=640, half=False, device=0):
    """查找并转换所有 PT 模型"""
    # 查找所有 .pt 文件
    pt_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pt') and not file.startswith('.'):
                pt_files.append(os.path.join(root, file))

    if not pt_files:
        print("⚠️ 未找到任何 .pt 模型文件")
        return

    print(f"==================== 找到 {len(pt_files)} 个 PT 模型 ====================")

    success_count = 0
    for pt_path in pt_files:
        if convert_pt_to_onnx(pt_path, imgsz, half, device):
            success_count += 1

    print(f"==================== 转换完成: {success_count}/{len(pt_files)} ====================")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='批量转换 PT 模型为 ONNX 格式（FP32）')
    parser.add_argument('--dir', type=str, default='.', help='搜索目录（默认当前目录）')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸（默认640）')
    parser.add_argument('--half', action='store_true', default=False, help='启用FP16半精度（默认禁用，使用FP32）')
    parser.add_argument('--device', type=str, default='0', help='推理设备（默认GPU:0，CPU用"cpu"）')

    args = parser.parse_args()

    # 处理参数
    device = args.device if args.device in ['cpu', '0', '1', '2', '3'] else int(args.device)

    print(f"配置: imgsz={args.imgsz}, half={args.half}, device={device}")
    print("注意: 默认使用FP32全精度，因为鲲鹏CPU上FP16运算极慢")

    find_and_convert_all(args.dir, args.imgsz, args.half, device)
