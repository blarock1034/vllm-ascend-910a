import time
from vllm import LLM, SamplingParams

llm = LLM(model="/data/models/DeepSeek-V2-Lite-Chat", max_model_len=2048,
          tensor_parallel_size=4, dtype="float16", max_num_seqs=32,
          trust_remote_code=True, enforce_eager=True)
tok = llm.get_tokenizer()
msgs = [{"role": "user", "content": "用两句话介绍一下杭州。"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
t0 = time.time()
outs = llm.generate([prompt] * 2, SamplingParams(temperature=0.0, max_tokens=64))
dt = time.time() - t0
for o in outs:
    print("=" * 50)
    print(repr(o.outputs[0].text))
n = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"[chat smoke] {n} tokens in {dt:.1f}s -> {dt/max(n,1)*1000:.0f} ms/tok(avg, 2并发)")
