"""Throughput proxy for PREREG_V8F_LORA on a new GPU (no download, random weights).
ViT-L/16 shape of DINOv3-L (24 blocks, d 1024, 16 heads, MLP 4096, 640 patches + CLS + 4 registers at 256x640),
frozen weights in bf16, rank-32 LoRA in fp32 on q and v, per-block gradient checkpointing, encoder calls of exactly 8 frames
(Amendment 3). Prints s/frame for forward+backward and for forward without grad, and the peak memory.
A proxy: DINOv3's RoPE and layer-scale are left out, so the real smoke on this GPU is the number of record."""
import json
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

D, H, MLP, NB, R = 1024, 16, 4096, 24, 32
dev = "cuda"


class LoRA(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dout, din, dtype=torch.bfloat16, device=dev) * 0.02, requires_grad=False)
        self.a = nn.Parameter(torch.randn(R, din, device=dev) * 0.01)
        self.b = nn.Parameter(torch.randn(dout, R, device=dev) * 0.01)

    def forward(self, x):
        return F.linear(x, self.w) + (F.linear(F.linear(x.float(), self.a), self.b)).to(x.dtype)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D).to(dev, torch.bfloat16), nn.LayerNorm(D).to(dev, torch.bfloat16)
        self.q, self.v = LoRA(D, D), LoRA(D, D)
        self.k = nn.Linear(D, D).to(dev, torch.bfloat16)
        self.o = nn.Linear(D, D).to(dev, torch.bfloat16)
        self.f1, self.f2 = nn.Linear(D, MLP).to(dev, torch.bfloat16), nn.Linear(MLP, D).to(dev, torch.bfloat16)
        for m in (self.n1, self.n2, self.k, self.o, self.f1, self.f2):
            for p in m.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        h = self.n1(x)
        B, T, _ = h.shape
        q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (self.q(h), self.k(h), self.v(h)))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(a)
        return x + self.f2(F.gelu(self.f1(self.n2(x))))


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = nn.Conv2d(3, D, 16, 16).to(dev, torch.bfloat16).requires_grad_(False)
        self.tok = nn.Parameter(torch.zeros(1, 5, D, dtype=torch.bfloat16, device=dev), requires_grad=False)
        self.blocks = nn.ModuleList(Block() for _ in range(NB))

    def forward(self, img, ckpt):
        x = self.pe(img).flatten(2).transpose(1, 2)
        x = torch.cat([self.tok.expand(x.shape[0], -1, -1), x], 1)
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if ckpt else b(x)
        return x[:, 5:].float()


def timeit(fn, n):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    torch.manual_seed(0)
    e = Enc()
    img = torch.randn(8, 3, 256, 640, device=dev, dtype=torch.bfloat16)
    ntr = sum(p.numel() for p in e.parameters() if p.requires_grad)

    def fb():
        y = e(img, True)
        y.pow(2).mean().backward()

    def fb_nockpt():
        y = e(img, False)
        y.pow(2).mean().backward()

    def fwd():
        with torch.no_grad():
            e(img, False)

    out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "trainable": ntr, "chunk": 8}
    torch.cuda.reset_peak_memory_stats()
    out["s_per_frame_fwd_bwd_ckpt"] = timeit(fb, 20) / 8
    out["peak_gib_ckpt"] = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    out["s_per_frame_fwd_bwd_nockpt"] = timeit(fb_nockpt, 20) / 8
    out["peak_gib_nockpt"] = torch.cuda.max_memory_allocated() / 2**30
    out["s_per_frame_fwd_nograd"] = timeit(fwd, 20) / 8
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
