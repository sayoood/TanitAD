"""Why is the FOLDED bf16 trunk a little further from strict fp32 than the UNFOLDED bf16 trunk?

HYPOTHESIS under test: under autocast the conv receives its folded bias (beta - mu*s) in bf16,
so each channel's whole offset carries one bf16 rounding -- an error COHERENT over the feature
map -- where the unfolded path rounds the conv output per element and applies the BN shift in
fp32 inside cuDNN's BN kernel.

Arm C keeps everything of the fold EXCEPT that the bias is added in fp32 after a bias-free conv
(one rounding at the output, like the unfolded BN output). Committed before running:
  * C's error ~ A's (unfolded) and < B's (folded)  => the bias rounding IS the mechanism;
  * C's error ~ B's                                 => it is NOT, and the doc says so.
Run on the dev-box RTX 4060 (cached resnet101.a1_in1k weights), 2 images at 416x1024, no_grad.
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


def fp32_bias_fold_(net):
    """Arm C: the same fold, but the bias is added in fp32 AFTER a bias-free conv."""
    for conv, bn in TT._conv_bn_pairs(net):
        def _fwd(x, _c=conv, _b=bn):
            s = _b.weight / torch.sqrt(_b.running_var + _b.eps)
            w = _c.weight * s.reshape(-1, 1, 1, 1)
            bias = _b.bias - _b.running_mean * s
            y = _c._conv_forward(x, w, None)
            return (y.float() + bias.reshape(1, -1, 1, 1)).to(y.dtype)
        conv.forward = _fwd
        bn.forward = (lambda x: x)


def run(t, xn, strict):
    torch.backends.cudnn.allow_tf32 = not strict
    torch.backends.cuda.matmul.allow_tf32 = not strict
    with torch.no_grad():
        return [o.float() for o in t._backbone(xn)]


def rel(a, b):
    return [round(float((u - v).norm() / v.norm()), 5) for u, v in zip(a, b)]


g = torch.Generator(device="cuda").manual_seed(1)
x = torch.rand(2, 3, 416, 1024, device="cuda", generator=g)
ref_t = build()
xn = ref_t.normalise(x)
ref = run(ref_t, xn, strict=True)
del ref_t
res = {"_what": __doc__.split("\n")[0], "_evidence_class": "MEASURED (ours), dev-box RTX 4060",
       "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
arms = {}
a = build(bf16=True, channels_last=True)
arms["A_unfolded_bf16_cl"] = rel(run(a, xn, False), ref); del a
b = build(bf16=True, channels_last=True, fold_bn=True)
arms["B_folded_bf16_cl"] = rel(run(b, xn, False), ref); del b
c = build(bf16=True, channels_last=True)
fp32_bias_fold_(c.net)
arms["C_folded_fp32_bias_bf16_cl"] = rel(run(c, xn, False), ref); del c
torch.backends.cudnn.allow_tf32 = True
res["rel_err_vs_strict_fp32_per_level"] = arms
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
