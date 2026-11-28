"""Thor: is the COMPILED bf16 backbone any further from the truth than the EAGER bf16 backbone?

`compile_probe.py` (MEASURED 1.53x on the backbone fwd+bwd) failed the precision bar it committed
before running: relative difference vs EAGER <= 1 %; it read 2.05 % / 5.46 %. That bar compared two
bf16 computations with each other, and bf16 alone sits 1.9 % / 4.7 % from fp32 -- so it could not
separate "compile is wrong" from "two bf16 roundings differ". The cheapest discriminating test,
COMMITTED BEFORE RUNNING (a new pre-registration, not a re-read of the old bar):

    REF = unfolded fp32 with TF32 off (strict). E = the eager launch backbone (bf16 + NHWC + frozen
    + folded BN). C = the same trunk under torch.compile. Compile is admissible on precision iff
    rel_err(C, REF) <= 1.10 x rel_err(E, REF) at BOTH feature levels; otherwise it is declined.

resnet101.a1_in1k pretrained, 416x1024, 8 images, chunk 8, no_grad forward.
"""
import json
import sys

import torch

from tanitad.models import timm_trunk as TT


def build(**kw):
    torch.manual_seed(0)
    t = TT.build_timm_trunk(in_channels=3, image_hw=(416, 1024),
                            model_name="resnet101.a1_in1k", pretrained=True, frozen_bn=True,
                            **kw).cuda()
    t.train()
    return t


def rel(a, b):
    return [round(float((u.float() - v.float()).norm() / v.float().norm()), 6)
            for u, v in zip(a, b)]


g = torch.Generator(device="cuda").manual_seed(1)
x = torch.rand(8, 3, 416, 1024, device="cuda", generator=g)
ref_t = build()
xn = ref_t.normalise(x)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
with torch.no_grad():
    ref = [o.clone() for o in ref_t._backbone(xn)]
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
del ref_t
torch.cuda.empty_cache()

lev = dict(fold_bn=True, bf16=True, channels_last=True, chunk_ckpt=8)
e = build(**lev)
with torch.no_grad():
    e_out = e._backbone(xn)
err_e = rel(e_out, ref)
del e, e_out
torch.cuda.empty_cache()

c = build(**lev)
cnet = torch.compile(c.net)
_real = TT._chunked_backbone


def _chunked_compiled(net, xx, chunk):
    return _real(cnet if net is c.net else net, xx, chunk)


TT._chunked_backbone = _chunked_compiled
try:
    with torch.no_grad():
        c_out = c._backbone(xn)
finally:
    TT._chunked_backbone = _real
err_c = rel(c_out, ref)
res = {"_what": __doc__.split("\n")[0], "_evidence_class": "MEASURED (ours), Thor",
       "torch": torch.__version__,
       "rel_err_vs_strict_fp32": {"E_eager_bf16_fold": err_e, "C_compiled_bf16_fold": err_c},
       "ratio_C_over_E": [round(c_ / e_, 3) for c_, e_ in zip(err_c, err_e)]}
res["admissible_on_precision"] = all(c_ <= 1.10 * e_ for c_, e_ in zip(err_c, err_e))
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("ZZCOMPILEPREC-DONEZZ")
