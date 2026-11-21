"""Thor: does arm C (the fold with its bias added in fp32 after a bias-free conv -- MEASURED more
accurate than both the unfolded and the shipped fold, `fold_bias_mechanism.py`) keep the fold's
speed? Committed before running: C is adopted only if its backbone fwd+bwd is within 5 % of B's;
otherwise B (the shipped fold) stays and C's precision gain is recorded as priced, not taken.

Backbone forward+backward, 8 images at 416x1024, bf16 + channels_last, frozen BN, pretrained
resnet101.a1_in1k; 3 warm-up + 8 timed iterations; torch.cuda.max_memory_allocated() only.
"""
import json
import sys
import time

import torch

from tanitad.models import timm_trunk as TT


def build(**kw):
    torch.manual_seed(0)
    t = TT.build_timm_trunk(in_channels=3, image_hw=(416, 1024),
                            model_name="resnet101.a1_in1k", pretrained=True,
                            frozen_bn=True, bf16=True, channels_last=True, **kw).cuda()
    t.train()
    return t


def fp32_bias_fold_(net):
    for conv, bn in TT._conv_bn_pairs(net):
        def _fwd(x, _c=conv, _b=bn):
            s = _b.weight / torch.sqrt(_b.running_var + _b.eps)
            w = _c.weight * s.reshape(-1, 1, 1, 1)
            bias = _b.bias - _b.running_mean * s
            y = _c._conv_forward(x, w, None)
            return (y.float() + bias.reshape(1, -1, 1, 1)).to(y.dtype)
        conv.forward = _fwd
        bn.forward = (lambda x: x)


def fwd_bwd(t, x, n=8, warm=3):
    for i in range(warm + n):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
        out = t._backbone(x)
        sum(o.float().pow(2).mean() for o in out).backward()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


g = torch.Generator(device="cuda").manual_seed(1)
x = torch.rand(8, 3, 416, 1024, device="cuda", generator=g)
res = {"_what": __doc__.split("\n")[0], "_evidence_class": "MEASURED (ours), Thor",
       "torch": torch.__version__}
for tag in ("A_unfolded", "B_folded_shipped", "C_folded_fp32_bias"):
    t = build(fold_bn=(tag == "B_folded_shipped"))
    if tag == "C_folded_fp32_bias":
        fp32_bias_fold_(t.net)
    xn = t.normalise(x)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    s = fwd_bwd(t, xn)
    res[tag] = {"s_fwd_bwd_8img": round(s, 4),
                "max_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(tag, res[tag], flush=True)
    del t
    torch.cuda.empty_cache()
res["C_within_5pct_of_B"] = res["C_folded_fp32_bias"]["s_fwd_bwd_8img"] <= 1.05 * res["B_folded_shipped"]["s_fwd_bwd_8img"]
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("ZZFOLDSPEED-DONEZZ")
