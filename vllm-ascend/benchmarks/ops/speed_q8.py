# Qwen3-8B 速度测试（910A/d801, fp16）
# 用法: Q8_TP=1 或 4 决定张量并行度
# 预热1条(TBE编译) -> 3条顺序(单并发延迟) -> 4条并发(聚合吞吐)
import os
import time

from vllm import LLM, SamplingParams

TP = int(os.environ.get("Q8_TP", "1"))

llm = LLM(model="/data/models/Qwen3-8B", max_model_len=2048,
          tensor_parallel_size=TP, dtype="float16", max_num_seqs=32)

tok = llm.get_tokenizer()


def mk(q):
    return tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)


sp = SamplingParams(temperature=0.0, max_tokens=128)

t0 = time.time()
llm.generate([mk("你好")], SamplingParams(temperature=0.0, max_tokens=8))
print(f"[warmup] {time.time()-t0:.1f}s")

lat = []
for i in range(3):
    t0 = time.time()
    o = llm.generate([mk(f"从{i*10}开始数数")] , sp)[0]
    dt = time.time() - t0
    n = len(o.outputs[0].token_ids)
    lat.append(dt / n * 1000)
    print(f"[seq{i}] {n} tok / {dt:.1f}s -> {dt/n*1000:.0f} ms/tok | {o.outputs[0].text[:40]!r}")

ps = [mk(f"写一段描写季节的短文，编号{i}") for i in range(4)]
for r in range(2):
    t0 = time.time()
    outs = llm.generate(ps, sp)
    dt = time.time() - t0
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[4way-r{r}] {tot} tok / {dt:.1f}s -> {tot/dt:.2f} tok/s aggregate")
print(f"[summary tp={TP}] seq best {min(lat):.0f} ms/tok")
