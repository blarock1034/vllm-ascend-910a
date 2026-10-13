#!/bin/bash
set -e
export NPU="${NPU:-${1:-0}}"
export MODEL_DIR="${MODEL_DIR:-${2:-$HOME/models/Qwen3-4B}}"
export PORT="${PORT:-${3:-8040}}"
export ASCEND_RT_VISIBLE_DEVICES=$NPU

if ! command -v vllm >/dev/null 2>&1; then
    export PATH="${CONDA_ENV:-/root/anaconda3/envs/ascend910}/bin:$PATH"
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
exec vllm serve "$MODEL_DIR" \
  --tensor-parallel-size 1 --dtype float16 \
  --max-num-seqs 32 --max-model-len 4096 \
  --generation-config vllm \
  --host 0.0.0.0 --port "$PORT"
