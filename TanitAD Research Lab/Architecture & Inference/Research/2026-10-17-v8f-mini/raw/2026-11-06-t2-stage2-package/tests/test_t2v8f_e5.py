"""Gates T3 and T4 (CPU). Every `check_*` takes `mut=None`; `code/run_mutations_e5.py` runs each with a deliberate regression that must raise AssertionError (RED).
T3: (a) ZERO == P10 exactly (policy level: the ZERO wrapper reads the prior from the model forward, P10 is arm 'p10'); (b) P10 attributed against itself reads exactly 0.
T4: the scorer reproduces the reader's per-window arrays (a_k arm, b_k P10 floor) on the banked RIC0_s0 dump, bit for bit, on every row it computes (16 of 19).
The heavy readers bind their own tree, so (b) and T4 run in a subprocess (`t2v8f_score.run_sub`). Run from t2v8f/: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_e5.py -q"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import t2v8f_kit as K  # noqa: E402
import t2v8f_score as S  # noqa: E402
import test_t2v8f_gates as G  # noqa: E402

ART = S.ART
RIC0_ARR = f"{ART}/dagland/motion_c/ri_read/raw/arrays/RIC0_s0.npz"
RIC0_DUMP = "F:/TanitAD-scratch/motion_c/ri/out/RIC0_s0/dump_RIC0_s0"
FLOOR_DUMP = f"{ART}/v8f-dagoacc-t1/dump_DAGOACC_s0_m10_causal_heads"
N_COMPUTED = 16


def check_t3_zero(mut=None, n=4):
    """ZERO == P10 exactly (max |d| == 0.0) on n windows; the model arm differs from both (the control must not be vacuous)."""
    for win in G.pick(n):
        p10, _, _ = G.make_policy(win, arm="p10")
        zero = K.ZeroPolicy(G.make_policy(win, arm="model")[0], skew=1e-6 if mut == "zero_eps" else 0.0)
        if mut == "zero_is_model":
            zero.plan = lambda frames, intr, v0, nav_cmd, nav_known=None, _z=zero: _z.inner.plan(frames, intr, v0, nav_cmd)
        v0 = float(p10.track[-1, 3])
        a, _ = p10.plan(None, None, v0, 0)
        b, _ = zero.plan(None, None, v0, 0)
        d = float(np.abs(np.asarray(a) - np.asarray(b)).max())
        assert d == 0.0, f"T3: ZERO differs from P10 (max |d| {d:.3g} m)"
        m, _ = G.make_policy(win, arm="model")[0].plan(None, None, v0, 0)
        assert float(np.abs(np.asarray(m) - np.asarray(a)).max()) > 1e-3, "T3: the model arm equals the prior; the control is vacuous"
    return n


def check_t3_attr(mut=None):
    """P10 attributed against itself: delta == lo == hi == 0.0 on every computed row."""
    r = S.run_sub("t3_attr", floor_dump=FLOOR_DUMP, mut=mut)
    assert r["n_rows"] >= N_COMPUTED, f"T3: only {r['n_rows']} rows computed"
    assert r["max_abs_delta_lo_hi"] == 0.0, f"T3: P10 against itself reads {r['max_abs_delta_lo_hi']:.3g}, not exactly 0"
    return {"rows": r["n_rows"], "max_abs": r["max_abs_delta_lo_hi"]}


def check_t4(mut=None):
    """The scorer's rows on the banked RIC0_s0 dump equal the reader's arrays bit for bit, both sides, on the 16 computed rows; the 3 S-POS rows are reported NOT COMPUTED."""
    r = S.run_sub("t4", arr=RIC0_ARR, dump=RIC0_DUMP, floor_dump=FLOOR_DUMP, mut=mut)
    comp = {k: v for k, v in r["rows"].items() if isinstance(v, dict)}
    bad = {k: (v["max_abs_dev_a"], v["max_abs_dev_b"]) for k, v in comp.items() if not (v["a_bit_identical"] and v["b_bit_identical"])}
    assert len(comp) == N_COMPUTED, f"T4: {len(comp)} rows computed, expected {N_COMPUTED}"
    assert not bad, f"T4: rows not bit-identical to the reader: {bad}"
    assert r["n_windows"] == 2399
    return {"rows_bit_identical": len(comp), "n_windows": r["n_windows"], "M1_pooled_share_a": r["m1_share_pooled_a"],
            "not_computed": sorted(k for k, v in r["rows"].items() if not isinstance(v, dict))}


def test_t3_zero():
    print(check_t3_zero())


def test_t3_attr():
    print(check_t3_attr())


def test_t4():
    print(check_t4())
