#!/usr/bin/env python3
"""ceiling_p6.py -- what the A40 can do on P6's trunk shapes (bf16), to bound any recipe-preserving speedup.

ViT-L/16 at 512x960: 1,925 tokens/image (1,920 patches + cls + 4 registers), 16 images per micro-batch (batch 4 x 4 cameras).
Times (CUDA events, median of reps): the four frozen-trunk GEMMs forward and their input-grad GEMMs, and SDPA forward+backward
per backend. Prints achieved TFLOPS and the implied lower bound for one micro-batch's trunk fwd+bwd.
"""
import json
import sys

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
NIMG = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N, D, H, DH, L = 1925, 1024, 16, 64, 24
R = NIMG * N


def t(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) / 1e3)
    ts.sort()
    return ts[len(ts) // 2]


out = {"torch": torch.__version__, "n_img": NIMG, "rows": R}
x = torch.randn(R, D, device=dev, dtype=torch.bfloat16)
x4 = torch.randn(R, 4 * D, device=dev, dtype=torch.bfloat16)
gemms = {"qkv_sep(3x D->D)": (x, [torch.randn(D, D, device=dev, dtype=torch.bfloat16) for _ in range(3)]),
         "qkv_fused(D->3D)": (x, [torch.randn(3 * D, D, device=dev, dtype=torch.bfloat16)]),
         "proj(D->D)": (x, [torch.randn(D, D, device=dev, dtype=torch.bfloat16)]),
         "fc1(D->4D)": (x, [torch.randn(4 * D, D, device=dev, dtype=torch.bfloat16)]),
         "fc2(4D->D)": (x4, [torch.randn(D, 4 * D, device=dev, dtype=torch.bfloat16)])}
g_tot = 0.0
for name, (inp, ws) in gemms.items():
    fl = sum(2 * R * w.shape[0] * w.shape[1] for w in ws)
    s = t(lambda: [F.linear(inp, w) for w in ws])
    out[f"gemm {name}"] = {"s": s, "tflops": fl / s / 1e12}
    if name != "qkv_fused(D->3D)":
        g_tot += s
# input-grad GEMMs have the transposed shapes and the same FLOPs: time fc1's and fc2's backward-input forms
gy4 = torch.randn(R, 4 * D, device=dev, dtype=torch.bfloat16)
w1 = torch.randn(4 * D, D, device=dev, dtype=torch.bfloat16)
s = t(lambda: gy4 @ w1)
out["gemm fc1 dgrad"] = {"s": s, "tflops": 2 * R * 4 * D * D / s / 1e12}
out["trunk_linear_fwd_per_layer_s"] = g_tot
out["trunk_linear_fwd+dgrad_24L_s"] = 2 * g_tot * L

q = torch.randn(NIMG, H, N, DH, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(NIMG, H, N, DH, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(NIMG, H, N, DH, device=dev, dtype=torch.bfloat16, requires_grad=True)
go = torch.randn(NIMG, H, N, DH, device=dev, dtype=torch.bfloat16)
from torch.nn.attention import SDPBackend, sdpa_kernel
fl_f = 4 * NIMG * H * N * N * DH
for nm, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION)):
    try:
        with sdpa_kernel(be):
            sf = t(lambda: F.scaled_dot_product_attention(q, k, v))
            def fb():
                o = F.scaled_dot_product_attention(q, k, v)
                o.backward(go)
            sfb = t(fb)
        out[f"sdpa {nm}"] = {"fwd_s": sf, "fwd_tflops": fl_f / sf / 1e12, "fwd+bwd_s": sfb, "bwd_s": sfb - sf,
                             "x24_fwd+bwd_s": 24 * sfb}
    except Exception as e:  # noqa: BLE001
        out[f"sdpa {nm}"] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
print(json.dumps(out, indent=1))
