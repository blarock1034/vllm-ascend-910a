#!/bin/bash
# 用法:
#   conda create -y -n ascend910 python=3.10
#   conda activate ascend910
#   bash deploy/setup_env_online.sh
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "错误: 请先创建并激活 conda 环境再运行本脚本:" >&2
    echo "  conda create -y -n ascend910 python=3.10 && conda activate ascend910" >&2
    exit 1
fi
PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "$PYVER" = "3.10" ] || echo "警告: 当前 Python 为 $PYVER，验证过的版本是3.10"
PIP="python -m pip"   # 始终作用于当前激活环境

echo "[1/5] 安装Python依赖"
$PIP install "numpy==1.26.4" pyyaml psutil decorator attrs cloudpickle tornado ml-dtypes \
    absl-py scipy pandas matplotlib openpyxl loguru pydantic \
    "opentelemetry-exporter-otlp-proto-grpc==1.33.1" "opentelemetry-exporter-otlp-proto-http==1.33.1" \
    "setuptools<81" wheel packaging cmake pybind11 setuptools-scm

echo "[2/5] 安装torch 2.5.1 CPU"
$PIP install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

echo "[3/5] 安装torch-npu"
$PIP install torch-npu==2.5.1.post1.dev20250619 --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi --pre \
  || $PIP install torch-npu==2.5.1.post1

echo "[4/5] 安装vllm 0.9.1"
$PIP install vllm==0.9.1 --no-deps
$PIP install $(grep -vE '^#' "$REPO/deploy/vllm_deps_list.txt" | tr '\n' ' ')
$PIP install "transformers==4.52.4" "tokenizers==0.21.1" "compressed-tensors==0.10.1" "numpy==1.26.4"
$PIP uninstall -y torchaudio torchvision 2>/dev/null || true

echo "[5/5] 安装补丁"
SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
cd "$SP/vllm/model_executor/layers"
for f in $(grep -rl 'list\[int\]\|list\[str\]' fused_moe/ quantization/utils/); do
    sed -i 's/\blist\[int\]/List[int]/g; s/\blist\[str\]/List[str]/g' "$f"
    if grep -q '^from typing import' "$f" && ! grep -qE '^from typing import.*\bList\b' "$f"; then
        sed -i 's/^from typing import /from typing import List, /' "$f"
    fi
done
cd "$REPO"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
COMPILE_CUSTOM_KERNELS=0 $PIP install -e "$REPO/vllm-ascend" --no-deps --no-build-isolation

echo "环境就绪（当前激活环境: $CONDA_DEFAULT_ENV）"
echo "下一步: bash $REPO/deploy/build_ascend_ops.sh   # 编译自定义算子"
