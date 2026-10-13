# -*- coding: utf-8 -*-
"""requests 流式测试（无 openai 包依赖）。
用法: python stream_test_requests.py "你的问题" [BASE] [MODEL]"""
import json
import os
import sys
import time

import requests

BASE = os.environ.get("BASE", sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8040/v1")
MODEL = os.environ.get("MODEL", sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/models/Qwen3-4B"))
prompt = sys.argv[1] if len(sys.argv) > 1 else "用两百字介绍大语言模型的工作原理"

payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
           "max_tokens": 256, "stream": True,
           "stream_options": {"include_usage": True},
           "chat_template_kwargs": {"enable_thinking": False}}
t0 = ttft = None
usage = n_char = 0
with requests.post(f"{BASE}/chat/completions", json=payload, stream=True,
                   timeout=(5, 600)) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not (line or "").startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]["completion_tokens"]
        delta = chunk["choices"][0]["delta"].get("content")
        if delta:
            ttft = ttft if ttft is not None else time.time() - t0
            n_char += len(delta)
            print(delta, end="", flush=True)
dt = time.time() - t0
toks = usage or n_char
print(f"\nTTFT: {ttft:.2f}s | 总耗时: {dt:.1f}s | tokens: {toks} | "
      f"生成速度: {toks / max(dt - ttft, 1e-6):.1f} tok/s")
