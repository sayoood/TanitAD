"""test_d9_accept.py -- the D9 metric-cache exactness gate (build_metric_cache.exactness + d9_accept), with stub scoring (no NAVSIM, no DB).

    python test_d9_accept.py        -> ZZTEST_PASS <n>

Pins the 2026-10-16 fix: the old rule `regression_equal == 0` refused a bit-identical cache because a 5 m ego shift cannot move a SATURATED set
(every proposal's PDMS exactly 0).  The new rule: identity on every set, the regression arm moves every set that CAN move, and it moves at least half.
Includes the real pod measurement (raw/2026-10-16-p6-d9-regression-arm/d9_regression_diag.json summary) and deliberate-regression arms that must FAIL."""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_metric_cache as B  # noqa: E402

n = 0


def check(cond, what):
    global n
    if not cond:
        print("FAIL:", what)
        sys.exit(1)
    n += 1


def rec(**kw):
    r = {"n_tokens": 24, "n_sets": 96, "n_equal": 96, "regression_sets": 96, "regression_equal": 0, "regression_saturated_sets": 0,
         "regression_saturated_undetected": 0, "regression_nonsaturated_undetected": 0, "tokens_with_a_detected_set": 24}
    r.update(kw)
    return r


# ---- d9_accept on records
check(B.d9_accept(rec())[0], "all detected, identity exact -> PASS")
pod = rec(regression_equal=22, regression_saturated_sets=76, regression_saturated_undetected=22, regression_nonsaturated_undetected=0, tokens_with_a_detected_set=21)
check(B.d9_accept(pod)[0], "the MEASURED pod sample (22 undetected, all saturated; 74 / 96 moved) -> PASS")
check(not B.d9_accept(rec(n_equal=95))[0], "REGRESSION: one cached != fresh set -> FAIL (identity)")
check("IDENTITY" in B.d9_accept(rec(n_equal=95))[1], "the identity failure is named")
check(not B.d9_accept(rec(regression_equal=1, regression_nonsaturated_undetected=1))[0], "REGRESSION: a NON-saturated set left equal -> FAIL (sensitivity)")
check(not B.d9_accept(rec(regression_equal=96, regression_saturated_sets=96, regression_saturated_undetected=96))[0],
      "REGRESSION: everything saturated and undetected (a blind comparator) -> FAIL (non-vacuity)")
check(not B.d9_accept(rec(regression_equal=49, regression_saturated_sets=49, regression_saturated_undetected=49))[0], "47 of 96 moved (< half) -> FAIL")
check(B.d9_accept(rec(regression_equal=48, regression_saturated_sets=48, regression_saturated_undetected=48))[0], "exactly half moved -> PASS")
check(not B.d9_accept(rec(n_sets=0, n_equal=0, regression_sets=0))[0], "no set scored -> FAIL")


# ---- exactness() end to end with stubbed scoring: the per-set classification is what the rule reads
class Ego:
    def __init__(self, x):
        self.rear_axle = types.SimpleNamespace(x=x)


def make_pt(mode):
    """mode: 'live' = scores depend on the ego x (moves); 'saturated' = all-zero PDMS that ignore the ego; 'blind' = non-zero PDMS that ignore the ego."""
    def score_plans(parts, plans8):
        x = parts[1].rear_axle.x
        m = len(plans8)
        if mode == "saturated":
            return {"pdms": np.zeros(m), "dac": np.zeros(m)}
        if mode == "blind":
            return {"pdms": np.full(m, 0.7), "dac": np.ones(m)}
        return {"pdms": np.full(m, 0.5) + 0.001 * x, "dac": np.ones(m)}
    return types.SimpleNamespace(metric_cache_parts=lambda sc: (None, Ego(0.0), None, None, None, None),
                                 executed_plans8=lambda props, repair=True: props, score_plans=score_plans)


def run(mode, cached_x=0.0):
    sys.modules["navsim_pdm_targets"] = make_pt(mode)
    B._scenarios = lambda db, mp_, log, toks: {t: object() for t in toks}
    B.OL = types.SimpleNamespace(load_parts=lambda p: (None, Ego(cached_x), None, None, None, None), parts_path=lambda c, log, tok: "x")
    return B.exactness("cache", "db", "map", [("L1", "t1"), ("L2", "t2")], n_sets=3)


r = run("live")
check(r["n_equal"] == 6 and r["regression_equal"] == 0 and B.d9_accept(r)[0], "live scoring: identity 6 / 6, all moved -> PASS")
r = run("saturated")
check(r["regression_saturated_undetected"] == 6 and r["regression_nonsaturated_undetected"] == 0, "saturated sets are classified saturated")
check(not B.d9_accept(r)[0], "all-saturated sample -> FAIL (non-vacuity), never a pass on saturation alone")
r = run("blind")
check(r["regression_nonsaturated_undetected"] == 6 and not B.d9_accept(r)[0], "REGRESSION: a comparator blind to the ego on non-zero sets -> FAIL")
r = run("live", cached_x=1.0)
check(r["n_equal"] == 0 and not B.d9_accept(r)[0], "REGRESSION: a cache whose ego differs from fresh -> FAIL (identity)")
check(r["tokens_with_a_detected_set"] == 2, "per-token detection is counted")

# ---- mutation: the OLD rule must refuse the measured pod sample (this is the defect the fix removes)
old_rule = lambda r: r["n_sets"] > 0 and r["n_equal"] == r["n_sets"] and r["regression_equal"] == 0  # noqa: E731
check(not old_rule(pod), "the old rule refuses the measured bit-identical pod sample (the defect, reproduced)")
print(f"ZZTEST_PASS {n}")
