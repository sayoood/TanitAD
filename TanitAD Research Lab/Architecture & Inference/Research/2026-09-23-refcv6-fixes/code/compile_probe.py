"""Thor: does torch.compile (Inductor, Triton 3.7.1) speed up the refcv6 backbone in its LAUNCH
configuration -- resnet101.a1_in1k at 416x1024, frozen + FOLDED BN, bf16 + NHWC, chunked
checkpointing in slices of 8 images -- and is it the same function?

What compile can buy here: the convolutions stay cuDNN calls; what Inductor can fuse is the
elementwise work around them (ReLU, residual add + ReLU, the fold's per-step weight scaling) and
their backward. Committed before running: compile is taken forward ONLY if the backbone fwd+bwd
is >= 10 % faster AND its outputs match eager within the bf16 path's own scale (relative error
vs eager <= 1 %); otherwise it is recorded as measured-and-declined.

24 images (3 checkpoint slices of 8), 3 warm-up + 6 timed fwd+bwd; compile time reported
separately; torch.cuda.max_memory_allocated() only.
"""
import json
import sys
import time

import torch

from tanitad.models import timm_trunk as TT


def build():
    torch.manual_seed(0)
    t = TT.build_timm_trunk(in_channels=3, image_hw=(416, 1024),
                            model_name="resnet101.a1_in1k", pretrained=True, frozen_bn=True,
                            fold_bn=True, bf16=True, channels_last=True, chunk_ckpt=8).cuda()
    t.train()
    return t


def step(t, x):
    out = t._backbone(x)
    sum(o.float().pow(2).mean() for o in out).backward()


def timed(t, x, n=6, warm=3):
    for i in range(warm + n):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
        step(t, x)
    torch.cuda.synchronize()
    return (time.time() - t0) / n


g = torch.Generator(device="cuda").manual_seed(1)
x = torch.rand(24, 3, 416, 1024, device="cuda", generator=g)
res = {"_what": __doc__.split("\n")[0], "_evidence_class": "MEASURED (ours), Thor",
       "torch": torch.__version__}

eager = build()
xn = eager.normalise(x)
with torch.no_grad():
    ref = [o.clone() for o in eager._backbone(xn[:8])]
torch.cuda.reset_peak_memory_stats()
res["eager_s_fwd_bwd_24img"] = round(timed(eager, xn), 4)
res["eager_max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
del eager
torch.cuda.empty_cache()

comp = build()
cnet = torch.compile(comp.net)
_real = TT._chunked_backbone


def _chunked_compiled(net, xx, chunk):
    return _real(cnet if net is comp.net else net, xx, chunk)


TT._chunked_backbone = _chunked_compiled
try:
    torch.cuda.synchronize()
    t0 = time.time()
    step(comp, xn)
    torch.cuda.synchronize()
    res["compile_first_step_s"] = round(time.time() - t0, 1)
    with torch.no_grad():
        got = comp._backbone(xn[:8])
    res["rel_err_vs_eager_per_level"] = [round(float((a - b).norm() / b.norm()), 6)
                                         for a, b in zip(got, ref)]
    torch.cuda.reset_peak_memory_stats()
    res["compiled_s_fwd_bwd_24img"] = round(timed(comp, xn), 4)
    res["compiled_max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
finally:
    TT._chunked_backbone = _real
res["speedup"] = round(res["eager_s_fwd_bwd_24img"] / res["compiled_s_fwd_bwd_24img"], 3)
res["take_forward"] = bool(res["speedup"] >= 1.10
                           and max(res["rel_err_vs_eager_per_level"]) <= 0.01)
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("ZZCOMPILEPROBE-DONEZZ")
