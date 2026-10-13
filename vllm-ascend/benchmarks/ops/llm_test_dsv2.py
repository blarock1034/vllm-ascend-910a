# DeepSeek-V2-Lite 4卡冒烟测试（910A/d801, fp16）
# 采样参数保持贪心，绕开全词表 Sort 慢路径
import time

from vllm import LLM, SamplingParams

llm = LLM(
    model="/data/models/DeepSeek-V2-Lite",
    max_model_len=1024,
    tensor_parallel_size=4,
    dtype="float16",
    max_num_seqs=32,
    trust_remote_code=True,
)

prompts = [
    "请用一句话介绍你自己。",
    "Write a python function to compute fibonacci numbers.",
    "简述机器学习与深度学习的区别。",
]
sampling = SamplingParams(temperature=0.0, max_tokens=64)

t0 = time.time()
outputs = llm.generate(prompts, sampling)
elapsed = time.time() - t0

for out in outputs:
    prompt = out.prompt
    text = out.outputs[0].text
    n = len(out.outputs[0].token_ids)
    print("=" * 60)
    print("PROMPT:", prompt)
    print(f"OUTPUT({n} tok):", text)

total_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
print("=" * 60)
print(f"[smoke] total {total_tok} tokens in {elapsed:.1f}s")
