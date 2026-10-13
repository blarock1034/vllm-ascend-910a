# -*- coding: utf-8 -*-
"""OpenAI SDK 流式速度测试。
用法: python stream_test_openai.py "你的问题" [BASE] [MODEL]
默认 BASE=http://127.0.0.1:8040/v1  MODEL=~/models/Qwen3-4B"""
import os
import sys
import time

from openai import OpenAI

BASE = os.environ.get("BASE", sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8040/v1")
MODEL = os.environ.get("MODEL", sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/models/Qwen3-4B"))
prompt = sys.argv[1] if len(sys.argv) > 1 else "用两百字介绍大语言模型的工作原理"

client = OpenAI(base_url=BASE, api_key="EMPTY")
t0 = time.time()
ttft = None
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256, stream=True,
    stream_options={"include_usage": True},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # Qwen3 关闭思考模式
)
n_char = usage = 0
for chunk in stream:
    if chunk.usage and chunk.usage.completion_tokens:
        usage = chunk.usage.completion_tokens
    delta = chunk.choices[0].delta.content if chunk.choices else None
    if delta:
        ttft = ttft or time.time() - t0
        n_char += len(delta)
        print(delta, end="", flush=True)
dt = time.time() - t0
toks = usage or n_char
print(f"\n{'=' * 50}\nTTFT: {ttft:.2f}s | 总耗时: {dt:.1f}s | tokens: {toks} | "
      f"生成速度: {toks / max(dt - ttft, 1e-6):.1f} tok/s")
