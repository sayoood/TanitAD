"""Is the GPU gap between folded and unfolded ROUNDING or a DEFECT? Discriminated against a
strict-fp32 reference (TF32 OFF) -- the question is whether folding makes the error WORSE
than the unfolded path at the same precision, not whether two rounded paths agree.

Each arm's output is compared to REF = unfolded, fp32, TF32 disabled. Prediction if the fold
is exact algebra: folded-strict ~1e-6 vs REF; folded-TF32 ~ unfolded-TF32; folded-bf16 <=
unfolded-bf16 (the fold subtracts the running mean inside the conv's fp32 epilogue instead of
after a bf16 rounding of the conv output).
"""
import json
import sys

import torch

from tanitad.models import timm_trunk as TT


def build(**kw):
    torch.manual_seed(0)
    t = TT.build_timm_trunk(in_channels=3, image_hw=(416, 1024),
                            model_name="resnet101.a1_in1k", pretrained=True,
                            frozen_bn=True, **kw).cuda()
    t.train()
    return t


def run(t, xn, tf32):
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    with torch.no_grad():
        return [o.float() for o in t._backbone(xn)]


def rel(a, b):
    return [float((u - v).norm() / v.norm()) for u, v in zip(a, b)]


g = torch.Generator(device="cuda").manual_seed(1)
x = torch.rand(2, 3, 416, 1024, device="cuda", generator=g)
plain, fold = build(), build(fold_bn=True)
plain_bc, fold_bc = build(bf16=True, channels_last=True), build(fold_bn=True, bf16=True,
                                                                  channels_last=True)
xn = plain.normalise(x)
ref = run(plain, xn, tf32=False)
res = {"_what": "fold precision vs strict-fp32 REF (unfolded, TF32 off), resnet101 416x1024",
       "_evidence_class": "MEASURED (ours), Thor", "torch": torch.__version__,
       "default_cudnn_allow_tf32": True,
       "rel_err_vs_REF_per_level": {
           "folded_fp32_strict": rel(run(fold, xn, False), ref),
           "unfolded_fp32_tf32": rel(run(plain, xn, True), ref),
           "folded_fp32_tf32": rel(run(fold, xn, True), ref),
           "unfolded_bf16_cl": rel(run(plain_bc, xn, True), ref),
           "folded_bf16_cl": rel(run(fold_bc, xn, True), ref)}}
torch.backends.cudnn.allow_tf32 = True
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("ZZFOLDPREC-DONEZZ")
