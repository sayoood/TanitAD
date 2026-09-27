"""Verify the A19 MAIN-only map spec on the FROZEN harness itself (blob 9bec9e88), CPU, no data:
main() with `--arms healthy` passes every spec check (load_spec, the gated-arm check, the lr_decay
check) and reaches the trunk build (stubbed to raise a sentinel there) -- while the A18 spec with
the same argv is REFUSED at the gated-arm check (the red arm). Then verdict() on a synthetic
MAIN-PASS result with no must-fail rows reads G_MAP_OVERFIT PASS, and one missed bar reads FAIL.
Usage: a19_spec_accept_check.py <tree> <a19 spec> <a18 spec>"""
import hashlib
import sys
from pathlib import Path

tree, a19, a18 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(tree / "stack"), str(tree / "stack" / "scripts")]
import tanitad  # noqa: E402

assert str(tree).replace("\\", "/").lower() in tanitad.__file__.replace("\\", "/").lower()
import map_hires_overfit as O  # noqa: E402
from tanitad.models import timm_trunk as TT  # noqa: E402

hb = Path(O.__file__).read_bytes()
blob = hashlib.sha1(b"blob %d\0" % len(hb) + hb).hexdigest()
assert blob == "9bec9e8844d522550eee5e5b4d1769fb09147d76", blob
PKG = Path(r"C:/Users/Admin/nb2_tree_2301/_pkg/raw")
W = PKG / "map_hires_class_weights_train_100x30.json"          # the TRAIN sqrt_mf launch file


class Reached(Exception):
    pass


def _stop(*_a, **_k):
    raise Reached()


TT.TimmResNetTrunk = _stop
argv = lambda spec: ["--spec", str(spec), "--class-weights", str(W), "--decision-rule",  # noqa: E731
                     "prior_corrected", "--near-lift-m", "20", "--near-refine-blocks", "1",
                     "--v2-cache", "V", "--gt-root", "G", "--extrinsics", "E", "--out", "o",
                     "--arms", "healthy", "--device", "cpu"]
try:
    O.main(argv(a19))
    raise SystemExit("A19: main() returned without building a trunk?")
except Reached:
    print("A19 MAIN-only spec + --arms healthy: every spec check PASSED, reached the trunk build")
try:
    O.main(argv(a18))
    raise AssertionError("the A18 spec was NOT refused")
except SystemExit as e:
    assert "is not in --arms" in str(e), e
    print("A18 spec + --arms healthy: REFUSED as expected:", str(e))
except Reached:
    raise AssertionError("the A18 spec reached the trunk build -- the refusal is gone?")

# the harness's own verdict on a MAIN-only result under the A19 spec
from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7  # noqa: E402

spec = O.load_spec(a19, class_weights=str(W), decision_rule="prior_corrected",
                   band_keys=EXTENT_REFCV7.band_keys)
bars = spec["thresholds"]["iou"]
main = {"final": {"n": {c: 5000 for c in bars}, "iou": {c: b + 0.05 for c, b in bars.items()},
                  "ce_mean": {c: 0.1 for c in bars}},
        "step0": {"ce_mean": {c: 1.0 for c in bars}}, "loss_finite_every_step": True}
ctrl = {"C1_constant_drivable": {"reproduced": True, "iou": {"drivable": 0.45}},
        "C2_gt_as_logits": {"reproduced": True}, "C3_rule_identity_w_ones": {"reproduced": True}}
guard = {"x": {"kind": "time_1ms"}}
v = O.verdict({"healthy": main}, spec, ctrl, guard)
print("verdict(MAIN pass):", v["G_MAP_OVERFIT"], "| regression rows:", v["regression_arms"])
assert v["G_MAP_OVERFIT"] == "PASS" and v["regression_arms"] == {}
main["final"]["iou"]["edge"] = 0.49
v = O.verdict({"healthy": main}, spec, ctrl, guard)
print("verdict(edge .49):", v["G_MAP_OVERFIT"], v["MAIN"]["verdict"])
assert v["G_MAP_OVERFIT"] == "FAIL"
print("A19 SPEC ACCEPT-CHECK OK")
