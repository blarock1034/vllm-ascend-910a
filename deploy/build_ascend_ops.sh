#!/bin/bash
#   conda activate ascend910
#   bash deploy/build_ascend_ops.sh
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "错误: 请先激活 conda 环境（conda activate ascend910）再运行本脚本" >&2
    exit 1
fi
CE="$(python -c 'import sys; print(sys.prefix)')"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=Ascend910B1
ATB0=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0

PYBIND_DIR=$(python -m pybind11 --cmakedir)
TORCH_DIR=$(python -c 'import os,torch; print(os.path.join(os.path.dirname(torch.__file__),"share","cmake","Torch"))')
TNPU_DIR=$(python -c 'import os,torch_npu; print(os.path.dirname(torch_npu.__file__))')
PREFIX="$REPO/ascend-ops/.install"; mkdir -p "$PREFIX"   # .run 安装器不建多级父目录

cd "$REPO/ascend-ops" && rm -rf .build && mkdir .build && cd .build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -Dpybind11_DIR=$PYBIND_DIR -DTorch_DIR=$TORCH_DIR \
  -DSOC_VERSION=Ascend910B1 -DTORCH_NPU_PATH=$TNPU_DIR \
  -DASCEND_HOME_PATH=$ASCEND_HOME_PATH -DATB_HOME_PATH=$ATB0 \
  -DPYBIND11_FINDPYTHON=ON -DPython3_FIND_STRATEGY=LOCATION \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build . --target ascend910a_extras_C -j"$(nproc)"
cmake --install .

cd "$REPO/ascend-ops"
cd .install/ascend910a_extras && for f in *.so; do cp -f "$f" ../../ascend910a_extras/; done
cd "$REPO/ascend-ops" && rm -rf ascend910a_extras/opp_install
cp -a .install/ascend910a_extras/opp_install ascend910a_extras/opp_install
python -c 'import site; print(site.getsitepackages()[0])' | xargs -I{} \
    bash -c "echo '$REPO/ascend-ops' > '{}/ascend910a_extras.pth"

echo "算子编译安装完成。验证:"
export LD_LIBRARY_PATH=$ATB0/lib:$LD_LIBRARY_PATH
python "$REPO/deploy/smoke_ops.py"
