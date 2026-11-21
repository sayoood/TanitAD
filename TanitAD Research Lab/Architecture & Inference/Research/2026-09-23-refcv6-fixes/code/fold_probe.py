"""Thor probe: the frozen-BN fold on the REAL resnet101 (pretrained BN statistics), on CUDA.

Two questions, each answered from the tensors, not the config:
  (1) is the folded trunk the same function on this device, in fp32 AND in bf16+NHWC?
  (2) what does the fold save on a backbone forward+backward of 8 images at 416x1024?
Only torch.cuda.max_memory_allocated() is quoted for memory (unified memory on Thor).
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
                            frozen_bn=True, **kw).cuda()
    t.train()
    return t


def rel(a, b):
    return float(((a.float() - b.float()).norm() / b.float().norm()).detach())


def fwd_bwd_time(t, x, n=8, warm=3):
    for p in t.parameters():
        p.grad = None
    for i in range(warm + n):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
        out = t._backbone(x)
        sum(o.float().pow(2).mean() for o in out).backward()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


res = {"_what": "fold probe, resnet101.a1_in1k pretrained, 416x1024, CUDA (Thor)",
       "_evidence_class": "MEASURED (ours), Thor", "torch": torch.__version__}
g = torch.Generator(device="cuda").manual_seed(1)
x = torch.rand(8, 3, 416, 1024, device="cuda", generator=g)

for tag, kw in (("fp32", {}), ("bf16_cl", {"bf16": True, "channels_last": True})):
    a = build(**kw)
    b = build(fold_bn=True, **kw)
    xn = a.normalise(x)
    with torch.no_grad():
        oa, ob = a._backbone(xn[:2]), b._backbone(xn[:2])
    res[tag] = {"bn_folded": b.memory_levers.get("bn_folded"),
                "rel_err_per_level": [rel(u, v) for u, v in zip(ob, oa)]}
    del oa, ob
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ta = fwd_bwd_time(a, xn)
    ma = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tb = fwd_bwd_time(b, xn)
    mb = torch.cuda.max_memory_allocated() / 2**30
    res[tag].update({"s_fwd_bwd_8img_unfolded": round(ta, 4), "s_fwd_bwd_8img_folded": round(tb, 4),
                     "speedup": round(ta / tb, 3),
                     "max_mem_gib_unfolded": round(ma, 2), "max_mem_gib_folded": round(mb, 2)})
    del a, b
    torch.cuda.empty_cache()
    print(tag, json.dumps(res[tag]), flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=1)
print("ZZFOLDPROBE-DONEZZ")
