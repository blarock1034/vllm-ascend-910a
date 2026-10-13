"""端到端冒烟：vLLM 离线推理两条中英文 prompt。
用法: ASCEND_RT_VISIBLE_DEVICES=1 python llm_smoke.py [模型目录]"""
import os
import time
from vllm import LLM, SamplingParams

model = os.sys.argv[1] if len(os.sys.argv) > 1 else os.path.expanduser("~/models/Qwen3-4B")
llm = LLM(model=model, max_model_len=2048, tensor_parallel_size=1,
          dtype="float16", max_num_seqs=32)
prompts = ["The capital of France is", "介绍一下华为昇腾"]
t0 = time.time()
for o in llm.generate(prompts, SamplingParams(max_tokens=48, temperature=0.0)):
    print("PROMPT:", o.prompt, "\nOUTPUT:", o.outputs[0].text[:120].replace("\n", " "), "\n---")
print(f"[wall {time.time() - t0:.1f}s] SMOKE PASS")
