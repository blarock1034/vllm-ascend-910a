"""自定义算子单测：swiglu(数值对齐) / add_rms_norm / grouped_matmul。
依赖 build_ascend_ops.sh 已执行。用法: ASCEND_RT_VISIBLE_DEVICES=N python smoke_ops.py"""
import os
import torch
import torch_npu
import ascend910a_extras.ops as ops

m, n, G, K, N_out = 128, 512, 8, 512, 1024
x = torch.randn(m, 2 * n, dtype=torch.float16).npu()
y = ops.swiglu(x); torch.npu.synchronize()
gate, up = x.float().chunk(2, dim=-1)
ref = torch.nn.functional.silu(gate) * up
assert (y.float() - ref).abs().max().item() < 0.05, "swiglu 数值偏差过大"
print("[1/3] swiglu OK（数值对齐）")

h = torch.randn(m, n, dtype=torch.float16).npu()
r = torch.randn(m, n, dtype=torch.float16).npu()
w = torch.randn(n, dtype=torch.float16).npu()
o, rn = ops.add_rms_norm(h, r, w, 1e-5); torch.npu.synchronize()
assert o.shape == h.shape and rn.shape == h.shape
print("[2/3] add_rms_norm OK")

xs = [torch.randn(32 + i * 8, K, dtype=torch.float16).npu() for i in range(G)]
wgt = torch.randn(G, N_out, K, dtype=torch.float16).npu()
wt = wgt.transpose(1, 2).contiguous().transpose(1, 2)   # K-major 视图
out = ops.grouped_matmul(torch.cat(xs, 0), wt,
                         torch.tensor([t.shape[0] for t in xs], dtype=torch.int64).npu())
torch.npu.synchronize()
assert out.shape == (sum(t.shape[0] for t in xs), N_out)
print("[3/3] grouped_matmul OK")
print("ALL OPS PASS")
