#!/bin/bash
set -e

# Sign_Service ATC 模型转换脚本
# 检测模型 (640x640) 生成普通OM和AIPP OM；分类模型 (224x224) 只生成普通OM
# （分类链路是 CPU float32 预处理，AIPP/NV12 输入的分类模型与运行时不兼容）

SOC_VERSION="Ascend310P3"
FRAMEWORK=5

echo "===================== Sign_Service ONNX → OM 转换 ====================="

# ===== 检测模型 sign.onnx (640x640) =====
DET_ONNX="./sign.onnx"
DET_AIPP_CFG="./aipp_det.cfg"
DET_INPUT_SHAPE="images:1,3,640,640"

if [ -f "${DET_ONNX}" ]; then
    # 普通模型
    echo ""
    echo "----- 检测模型 (普通) -----"
    atc \
        --model="${DET_ONNX}" \
        --output="./sign" \
        --framework=${FRAMEWORK} \
        --input_shape="${DET_INPUT_SHAPE}" \
        --soc_version=${SOC_VERSION} \
        --output_type=FP32

    # AIPP 模型
    if [ -f "${DET_AIPP_CFG}" ]; then
        echo ""
        echo "----- 检测模型 (AIPP零拷贝) -----"
        atc \
            --model="${DET_ONNX}" \
            --output="./sign_aipp" \
            --framework=${FRAMEWORK} \
            --input_shape="${DET_INPUT_SHAPE}" \
            --insert_op_conf="${DET_AIPP_CFG}" \
            --soc_version=${SOC_VERSION} \
            --output_type=FP32
    else
        echo "WARNING: 找不到 AIPP 配置 ${DET_AIPP_CFG}，跳过 AIPP 模型转换"
    fi
else
    echo "WARNING: 找不到 ${DET_ONNX}，跳过检测模型转换"
fi

# ===== 分类模型 sign_cls.onnx (224x224) =====
# 注意：分类只转普通 OM。运行时分类走 CPU float32 预处理（resize+BGR2RGB+/255+NCHW）
# 喂给原生 ACL 加载的模型，若转成 AIPP(NV12) 输入模型会导致 memcpy 越界 + 分类输出噪声
CLS_ONNX="./sign_cls.onnx"
CLS_INPUT_SHAPE="images:1,3,224,224"

if [ -f "${CLS_ONNX}" ]; then
    echo ""
    echo "----- 分类模型 (普通，AIPP 模式也用这份) -----"
    atc \
        --model="${CLS_ONNX}" \
        --output="./sign_cls" \
        --framework=${FRAMEWORK} \
        --input_shape="${CLS_INPUT_SHAPE}" \
        --soc_version=${SOC_VERSION} \
        --output_type=FP32
else
    echo "WARNING: 找不到 ${CLS_ONNX}，跳过分类模型转换"
fi

echo ""
echo "===================== 转换完毕 ====================="
echo "普通模型: sign.om, sign_cls.om"
echo "AIPP模型: sign_aipp.om (仅检测零拷贝)"
echo ""
echo "使用方式:"
echo "  AscendAipp=false → model_path 指向 sign.om，分类用 sign_cls.om"
echo "  AscendAipp=true  → model_path 指向 sign_aipp.om，分类仍用 sign_cls.om"
