import time
from vllm import LLM, SamplingParams

llm = LLM(model="/data/models/DeepSeek-V2-Lite", max_model_len=1024,
          tensor_parallel_size=4, dtype="float16", max_num_seqs=32,
          trust_remote_code=True)
sampling = SamplingParams(temperature=0.0, max_tokens=12)
out = llm.generate(["The capital of France is"], sampling)[0]
o = out.outputs[0]
print("TOKEN_IDS:", list(o.token_ids))
print("FINISH:", o.finish_reason, "| TEXT_REPR:", repr(o.text))
