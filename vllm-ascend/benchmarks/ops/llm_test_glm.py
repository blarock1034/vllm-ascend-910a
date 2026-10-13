# GLM-4-9B-Chat-HF 冒烟测试（910A/d801, fp16）
# 先单卡（TP1, 18.9GB 权重单卡放得下），贪心采样规避全词表 Sort 慢路径
import time

from vllm import LLM, SamplingParams

TP = int(__import__("os").environ.get("GLM_TP", "1"))

llm = LLM(
    model="/data/models/GLM-4-9B-Chat-HF",
    max_model_len=2048,
    tensor_parallel_size=TP,
    dtype="float16",
    max_num_seqs=32,
)

tok = llm.get_tokenizer()
prompts = []
for q in ["用两句话介绍人工智能。", "Write a haiku about autumn leaves."]:
    msgs = [{"role": "user", "content": q}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True))

t0 = time.time()
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=64))
elapsed = time.time() - t0

for o in outs:
    print("=" * 56)
    print("OUTPUT:", repr(o.outputs[0].text))

total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
print("=" * 56)
print(f"[glm smoke tp={TP}] {total_tok} tokens in {elapsed:.1f}s")
