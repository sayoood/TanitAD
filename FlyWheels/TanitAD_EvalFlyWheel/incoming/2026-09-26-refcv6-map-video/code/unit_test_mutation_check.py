"""Test-power check for taniteval/tests/test_render_refcv6_map_video.py: the unit tests must go RED on
the real defects they exist for, not merely be green on the correct code.

Three mutations are swapped into the tool under test, in memory, and every test is re-run:

* ``strict_threshold``: ``> 0.5`` instead of the trainer's ``>= 0.5``, on both sides;
* ``argmax_prediction``: the prediction taken by ARGMAX against a thresholded GT. The trainer's
  comment names this "two rules" (refc_v3_train.py:4501-4503);
* ``no_orientation_flip``: the grid is drawn without the row/col flip, so the ego lands at the top
  left and the right side of the road shows on the left.

Each mutation must turn at least one test RED, and the unmutated tool must be all green. The output
is banked in raw/logs/unit_test_mutation_check.log.
"""
import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location(
    "t", str(REPO / "taniteval" / "tests" / "test_render_refcv6_map_video.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
M = T.M
orig_rule, orig_g2i = M.drivable_iou_rule, M.grid_to_image


def _row(map_logits, gt, pr, sn):
    inter = float((gt & pr).sum())
    union = float((gt | pr).sum())
    n = float(sn.sum())
    return {"iou": inter / union if union else 0.0, "inter": inter, "union": union, "n_scored": n,
            "pred_drivable_frac": float(pr.sum()) / n, "gt_drivable_frac": float(gt.sum()) / n,
            "pred_drivable_prob_mean": 0.0, "_scored": sn, "_gt": gt, "_pr": pr}


def strict_rule(map_logits, map_frac, map_seen, map_valid=None, *, drivable_ch=1):
    sn = map_seen & map_valid if map_valid is not None else map_seen
    prob = map_logits.softmax(dim=1)[:, drivable_ch]
    return _row(map_logits, (map_frac[:, drivable_ch] > 0.5) & sn, (prob > 0.5) & sn, sn)


def argmax_rule(map_logits, map_frac, map_seen, map_valid=None, *, drivable_ch=1):
    sn = map_seen & map_valid if map_valid is not None else map_seen
    return _row(map_logits, (map_frac[:, drivable_ch] >= 0.5) & sn,
                (map_logits.argmax(1) == drivable_ch) & sn, sn)


def no_flip(a):
    a = np.asarray(a)
    return np.repeat(np.repeat(a, M.S, axis=0), M.S, axis=1)


tests = [t for t in dir(T) if t.startswith("test_")]
res = {}
for name, (attr, fn) in (("strict_threshold", ("drivable_iou_rule", strict_rule)),
                         ("argmax_prediction", ("drivable_iou_rule", argmax_rule)),
                         ("no_orientation_flip", ("grid_to_image", no_flip))):
    setattr(M, attr, fn)
    red = []
    for tn in tests:
        try:
            getattr(T, tn)()
        except AssertionError:
            red.append(tn)
    res[name] = red
    M.drivable_iou_rule, M.grid_to_image = orig_rule, orig_g2i
for k, v in res.items():
    print(f"{k}: {len(v)} test(s) RED -> {v}")
for tn in tests:
    getattr(T, tn)()
print("unmutated: all", len(tests), "green")
if not all(res.values()):
    raise SystemExit("a mutation stayed GREEN -- the tests have no power against it")
