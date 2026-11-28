"""How far is the 1e-5 bar from rounding, and from a WRONG fold? Measured, not argued."""
import sys, json
import os; _R = os.environ["FOLD_ROOT"]; sys.path.insert(0, _R + "/stack"); sys.path.insert(0, _R + "/stack/tests")
import torch
import test_trunk_speed_levers as T
from tanitad.models import timm_trunk as TT

def rel_err(fold_fn=None):
    ref = T._randomise_bn(T._trunk(frozen_bn=True))
    if fold_fn is None:
        fold = T._randomise_bn(T._trunk(frozen_bn=True, fold_bn=True))
    else:
        fold = T._randomise_bn(T._trunk(frozen_bn=True))
        fold_fn(fold.net)
    x = ref.normalise(T._x())
    with torch.no_grad():
        return max(float((a - b).norm() / b.norm()) for a, b in zip(ref._backbone(x), fold._backbone(x)))

def mk(variant):
    def f(net):
        for conv, bn in TT._conv_bn_pairs(net):
            def _fwd(x, _c=conv, _b=bn):
                s = _b.weight / torch.sqrt(_b.running_var + (0.0 if variant == "no_eps" else _b.eps))
                w = _c.weight * s.reshape(-1, 1, 1, 1)
                bias = _b.bias - (0 if variant == "no_mean" else _b.running_mean * s)
                return _c._conv_forward(x, w, bias)
            conv.forward = _fwd
            if variant != "bn_kept":
                bn.forward = (lambda x: x)
    return f

out = {"correct_fold": rel_err()}
for v in ("no_mean", "no_eps", "bn_kept"):
    out[v] = rel_err(mk(v))
out["_what"] = "relative error of the folded vs unfolded trunk (resnet18, randomised BN) -- correct fold vs three wrong ones"
out["_evidence_class"] = "MEASURED (ours), CPU"
print(json.dumps(out, indent=1))
open(sys.argv[1], "w").write(json.dumps(out, indent=1))
