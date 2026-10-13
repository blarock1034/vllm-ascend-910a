# Deploying vLLM on Ascend 910A Series NPUs

English | [简体中文](README.zh-CN.md)

vllm-ascend only supports Ascend 910B and later NPUs; the original 910A series is not supported.

Currently, the only support for 910A is the community container-image project [vllm-ascend-910a](https://github.com/shirwy/vllm-ascend-910a), which starts the vLLM service through a container and has only been adapted for Qwen3-8B and Qwen3-30B-A3B. Its approach is to fill in the Qwen-required operators missing on Ascend 910A using Ascend C, and to bypass the vllm-ascend installation self-check by modifying the NPU name. Inference still runs in eager mode.

This project attempts to adapt this approach to a bare-metal Ascend 910A environment.

## 1. Environment


| Item | Value |
|---|---|
| Machine | Atlas 800 (9010) training server |
| OS | openEuler 24.03 LTS SP3 (x86) |
| NPU | Ascend 910A series (d801) |
| Firmware | 7.8.0.7.220 |
| Driver | 25.5.2 |
| CANN Toolkit | 9.0.0 |


## 2. Quick Start

```bash
# 0 Prerequisite: install the Ascend driver, firmware and CANN packages following the official documentation, and install Conda for environment isolation
conda create -y -n ascend910 python=3.10
conda activate ascend910

# 1 Run the setup script inside the conda environment
bash deploy/setup_env_online.sh

# 2 Run the script to build the custom operators
bash deploy/build_ascend_ops.sh

# 3 Download the Qwen3-4B model
pip install modelscope                    
modelscope download --model Qwen/Qwen3-4B --local_dir ~/models/Qwen3-4B

# 4 Test — remember to change the model path in the script
ASCEND_RT_VISIBLE_DEVICES=0 python deploy/llm_smoke.py

# 5 Start the service — remember to change the model path in the script
bash deploy/serve_qwen.sh                   

# 6 Test the service
python client/stream_test_openai.py "介绍一下华为昇腾"
```

## Measured Performance

| Model | Config | Speed |
|---|---|---|
| Qwen3-4B | 1 card | 89 ms/token |
| Qwen3-8B | 1 card | 127 ms/token |  
| Qwen3-30B-A3B | 4 cards | 210 ms/token| 
| GLM-4-9B-Chat | 4 cards | 150 ms/token |


Cold start: the first prompt of each new length triggers TBE shape-specific compilation

## Repository Structure

```
├── vllm-ascend/          # vllm_ascend adaptation fork   
├── ascend-ops/           # custom AscendC operator package
├── deploy/               # deployment scripts
├── client/               # test scripts                
```

## Acknowledgements

Thanks to the [vllm-ascend](https://gitee.com/ascend/vllm-ascend) project for providing the foundation of vLLM adaptation on the Ascend platform

Thanks to the community adaptation fork [vllm-ascend-910a](https://github.com/shirwy/vllm-ascend-910a) for the pioneering work that made vLLM run on the older 910A
