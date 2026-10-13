# Ascend 910A系列NPU部署vllm

vllm-ascend只支持Ascend 910B及以上系列NPU，对于最初版本的910A系列均不进行支持

目前对910A的支持仅有社区适配镜像方案[vllm-ascend-910a](https://github.com/shirwy/vllm-ascend-910a),通过容器的方式启动vllm服务，且只对Qwen3-8B和Qwen3-30B-A3B进行了适配，其思路是基于Ascend C补齐Ascend 910A中缺少的Qwen所需算子，并通过修改NPU名称的方式绕过vllm-ascend安装自检，推理仍为eager模式

本项目尝试在Ascend 910A裸机环境中适配此方案

## 一、环境信息


| 项目 | 参数 |
|---|---|
| 机型 | Atlas 800 (9010) 训练服务器 |
| 操作系统 | openEuler 24.03 LTS SP3（x86） |
| NPU | Ascend 910A系列 (d801) |
| 昇腾固件 | 7.8.0.7.220 |
| 昇腾驱动 | 25.5.2 |
| CANN Toolkit | 9.0.0 |


## 二、快速开始

```bash
# 0 前提：根据官方文档安装昇腾环境驱动、固件和CANN软件包，安装Conda进行环境隔离
conda create -y -n ascend910 python=3.10
conda activate ascend910

# 1 执行脚本，在conda环境中执行脚本进行配置
bash deploy/setup_env_online.sh

# 2 执行脚本，编译自定义算子
bash deploy/build_ascend_ops.sh

# 3 下载Qwen3-4B模型
pip install modelscope                    
modelscope download --model Qwen/Qwen3-4B --local_dir ~/models/Qwen3-4B

# 4 测试，注意修改脚本里的模型路径
ASCEND_RT_VISIBLE_DEVICES=0 python deploy/llm_smoke.py

# 5 拉起服务，注意修改脚本里的模型路径
bash deploy/serve_qwen.sh                   

# 6 测试服务
python client/stream_test_openai.py "介绍一下华为昇腾"
```

## 实测速度

| 模型 | 配置 | 速度 |
|---|---|---|
| Qwen3-4B | 单卡 | 89 ms/token |
| Qwen3-8B | 单卡 | 127 ms/token |  
| Qwen3-30B-A3B | 四卡 | 210 ms/token| 
| GLM-4-9B-Chat | 四卡 | 150 ms/token |


冷启动：每个新prompt长度首次会触发TBE按形状编译

## 目录结构

```
├── vllm-ascend/          # vllm_ascend适配fork   
├── ascend-ops/           # 自定义 AscendC 算子包
├── deploy/               # 部署脚本
├── client/               # 测试脚本                
```

## 致谢

感谢 [vllm-ascend](https://gitee.com/ascend/vllm-ascend) 项目提供了昇腾平台vLLM适配的基础

感谢社区适配分支 [vllm-ascend-910a](https://github.com/shirwy/vllm-ascend-910a) 的先行工作，使老款910A得以运行vLLM

