#!/bin/bash
set -e

# Sign_Service ATC 模型转换脚本
# 检测模型 (640x640) + 分类模型 (224x224) 各生成普通OM和AIPP OM

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
CLS_ONNX="./sign_cls.onnx"
CLS_AIPP_CFG="./aipp_cls.cfg"
CLS_INPUT_SHAPE="images:1,3,224,224"

if [ -f "${CLS_ONNX}" ]; then
    # 普通模型
    echo ""
    echo "----- 分类模型 (普通) -----"
    atc \
        --model="${CLS_ONNX}" \
        --output="./sign_cls" \
        --framework=${FRAMEWORK} \
        --input_shape="${CLS_INPUT_SHAPE}" \
        --soc_version=${SOC_VERSION} \
        --output_type=FP32

    # AIPP 模型
    if [ -f "${CLS_AIPP_CFG}" ]; then
        echo ""
        echo "----- 分类模型 (AIPP零拷贝) -----"
        atc \
            --model="${CLS_ONNX}" \
            --output="./sign_cls_aipp" \
            --framework=${FRAMEWORK} \
            --input_shape="${CLS_INPUT_SHAPE}" \
            --insert_op_conf="${CLS_AIPP_CFG}" \
            --soc_version=${SOC_VERSION} \
            --output_type=FP32
    else
        echo "WARNING: 找不到 AIPP 配置 ${CLS_AIPP_CFG}，跳过 AIPP 模型转换"
    fi
else
    echo "WARNING: 找不到 ${CLS_ONNX}，跳过分类模型转换"
fi

echo ""
echo "===================== 转换完毕 ====================="
echo "普通模型: sign.om, sign_cls.om (用于 ais_bench 推理)"
echo "AIPP模型: sign_aipp.om, sign_cls_aipp.om (用于零拷贝推理)"
echo ""
echo "使用方式:"
echo "  AscendAipp=false → model_path 指向 sign.om"
echo "  AscendAipp=true  → model_path 指向 sign_aipp.om"
