"""Tests of the PURE layer of ``speedsub_t1_read.py`` (PREREG_V8F_SPEEDSUB_T1 sections 3-7): synthetic rows reach EVERY outcome row, every guard, every control and the floor rule; the boundaries are
tested on both sides. No heavy import: runs in seconds.

Run: python -m pytest -q test_speedsub_t1_read.py        (SPEEDSUB_READER=<mutant.py> points the whole file at a mutated copy; see mutation_speedsub_reader.py)

Controls that must read as stated:
  * BAR       a point of exactly -0.06035 with the CI upper end below 0 is a HIT (inclusive); -0.06034 is not; an upper end of exactly 0.0 is not.
  * GUARDS    G-WORSE needs >= 3 runs (2 is not enough); G-NEW has two readings and the coded one is the stricter; G-ADE fires at ONE run; G-VIOL needs >= 3 runs and is inert when the amendment switch is off.
  * TABLE     REFUSED before HARMS before PASS; all four hits with one guard firing is HARMS; PARTIAL needs both CTRL seeds; NULL-ONE / NULL-FLAT are labels only.
  * READ A    D1 (<= -2F = -0.2415), D2, D3, crawl rows; DOMINANT only with all of them.
  * FLOOR     COLLAPSE only if g_id > 0 and g_c <= 0.5 g_id on ALL THREE rows.
  * CONTROLS  K-ID at 1e-4, K-ZERO exactly 0.0, K-REG at all four runs, K-REACH 99 % / 95 %, K-SHUF registered, K-NONSPEED exactly 0.0, K-SHA per field, K-REC 1e-5, K-FLOOR bit for bit.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
READER = os.environ.get("SPEEDSUB_READER") or str(HERE / "speedsub_t1_read.py")
spec = importlib.util.spec_from_file_location("speedsub_t1_read_under_test", READER)
R = importlib.util.module_from_spec(spec)
sys.modules["speedsub_t1_read_under_test"] = R
spec.loader.exec_module(R)
RUNS, CTRL = R.RUNS, R.CTRL_RUNS


def row(delta, lo=None, hi=None, sep=None):
    lo = delta - 0.05 if lo is None else lo
    hi = delta + 0.05 if hi is None else hi
    return {"delta": delta, "lo": lo, "hi": hi, "separated": (lo > 0 or hi < 0) if sep is None else sep, "n": "2399/141"}


def flat():
    """A row that is neither better nor worse."""
    return row(0.0, -0.1, 0.1)


def table(over=None):
    t = {l: flat() for l in R.LABELS}
    t[R.VIOL] = flat()
    t.update(over or {})
    return t


def by_run(over_by_run=None, default=None):
    over_by_run = over_by_run or {}
    return {r: table(over_by_run.get(r, default)) for r in RUNS}


GOOD_ADE = row(-0.10, -0.15, -0.05)                      # a hit
NO_GUARD = {"G-WORSE": False, "G-NEW": False, "G-ADE": False, "G-VIOL": False}


def hits(*runs):
    return {r: r in runs for r in RUNS}


# ------------------------------------------------------------------------------------------------ the bar and a HIT
def test_the_bar_is_half_f():
    assert R.BAR == 0.1207 / 2.0 and abs(R.BAR - 0.06035) < 1e-12 and R.TWO_F == 0.2415


@pytest.mark.parametrize("delta,hi,want", [(-0.06035, -0.01, True), (-0.06034, -0.01, False), (-0.1, -1e-9, True), (-0.1, 0.0, False), (-0.1, 0.01, False), (0.2, 0.3, False), (-0.06036, -0.02, True)])
def test_a_hit_is_separated_better_and_at_least_f_over_two_inclusive(delta, hi, want):
    assert R.is_hit(row(delta, delta - 0.1, hi)) is want


def test_a_row_not_computed_is_never_a_hit_or_a_worsening():
    nc = {"status": "NOT COMPUTED"}
    assert not R.is_hit(nc) and not R.sep_worse("ADE6", nc) and not R.sep_better("ADE6", nc)
    with pytest.raises(R.RowNotComputed):
        R.worsening("ADE6", nc)


def test_direction_of_agreement_rows():
    assert R.sep_worse("TAC-LAT", row(-0.05, -0.08, -0.02)) and not R.sep_worse("TAC-LAT", row(0.05, 0.02, 0.08)) and R.sep_better("TAC-LAT", row(0.05, 0.02, 0.08))
    assert R.sep_worse("ADE6", row(0.05, 0.02, 0.08)) and R.sep_better("ADE6", row(-0.05, -0.08, -0.02))


# ------------------------------------------------------------------------------------------------ guards
def test_g_ade_fires_at_one_run():
    ade = {r: GOOD_ADE for r in RUNS}
    assert R.g_ade(ade) == (False, [])
    ade["REP_s1"] = row(0.05, 0.02, 0.08)
    assert R.g_ade(ade) == (True, ["REP_s1"])
    ade["REP_s1"] = row(0.05, -0.02, 0.08)                      # worse point, not separated
    assert R.g_ade(ade)[0] is False


def test_g_worse_needs_three_runs_and_names_the_row():
    w = row(0.3, 0.1, 0.5)
    for n, want in ((2, False), (3, True), (4, True)):
        rows = by_run({r: {"curvature MAE": w} for r in RUNS[:n]})
        fires, per = R.g_worse(rows)
        assert fires is want and per == {"curvature MAE": list(RUNS[:n])}
    rows = by_run({r: {"TAC-LON": row(-0.2, -0.3, -0.1)} for r in RUNS[:3]})                     # an agreement row worsens downward
    assert R.g_worse(rows)[0] is True
    rows = by_run({r: {"TAC-LON": row(0.2, 0.1, 0.3)} for r in RUNS})
    assert R.g_worse(rows)[0] is False
    rows = by_run({r: {"ADE6": w} for r in RUNS[:3]})                                              # ADE6 is among the rows
    assert R.g_worse(rows) == (True, {"ADE6": list(RUNS[:3])})


def test_g_worse_ignores_rows_not_computed():
    rows = by_run({r: {"S-POS@30": {"status": "NOT COMPUTED"}} for r in RUNS})
    assert R.g_worse(rows) == (False, {})


def test_g_new_stricter_reading_and_the_other_beside():
    worse = row(0.3, 0.1, 0.5)
    ep = by_run({r: {"yaw-rate MAE": worse} for r in CTRL} | {r: {} for r in RUNS[2:]})
    ip_clear = by_run()
    fires, d = R.g_new(ep, ip_clear)
    assert fires and d["rows"] == ["yaw-rate MAE"] and d["fires_if_identity_clear_at_both"] == ["yaw-rate MAE"]
    ip_one = by_run({"CTRL_s0": {"yaw-rate MAE": worse}})                                           # identity was already worse at ONE CTRL seed
    fires, d = R.g_new(ep, ip_one)
    assert fires and d["fires_if_identity_clear_at_both"] == []
    ip_both = by_run({r: {"yaw-rate MAE": worse} for r in CTRL})                                    # identity was worse at both: not new
    assert R.g_new(ep, ip_both)[0] is False
    ep_one = by_run({"CTRL_s0": {"yaw-rate MAE": worse}})                                           # egoc worse at one CTRL seed only: does not fire
    assert R.g_new(ep_one, ip_clear)[0] is False
    ep_rep = by_run({r: {"yaw-rate MAE": worse} for r in RUNS[2:]})                                  # REP runs do not enter G-NEW
    assert R.g_new(ep_rep, ip_clear)[0] is False


def test_g_viol_three_of_four_and_the_switch():
    w = row(0.04, 0.01, 0.07)
    for n, want in ((2, False), (3, True), (4, True)):
        v = {r: (w if i < n else flat()) for i, r in enumerate(RUNS)}
        assert R.g_viol(v) == (want, list(RUNS[:n]))
    v = {r: row(-0.04, -0.07, -0.01) for r in RUNS}                                                  # a DROP in violations is not a harm
    assert R.g_viol(v) == (False, [])
    v = {r: w for r in RUNS}
    old = R.AMENDMENT_1_GVIOL
    try:
        R.AMENDMENT_1_GVIOL = False
        assert R.g_viol(v)[0] is False
    finally:
        R.AMENDMENT_1_GVIOL = old
    assert R.g_viol(v)[0] is True
    assert R.g_viol({r: {"status": "NOT COMPUTED"} for r in RUNS}) == (False, [])


# ------------------------------------------------------------------------------------------------ the decision table
def D(controls_ok=True, present=True, hit=(), guards=None):
    return R.decide_speedsub(controls_ok, present, hits(*hit), {**NO_GUARD, **(guards or {})})


def test_refused_first():
    assert D(controls_ok=False, hit=RUNS, guards={"G-ADE": True})["OUTCOME"] == "REFUSED"
    assert D(present=False, hit=RUNS)["OUTCOME"] == "REFUSED"


@pytest.mark.parametrize("g", ["G-WORSE", "G-NEW", "G-ADE", "G-VIOL"])
def test_harms_beats_pass_for_every_guard(g):
    d = D(hit=RUNS, guards={g: True})
    assert d["OUTCOME"] == "HARMS" and d["flags"]["guards_fired"] == [g]


def test_pass_needs_all_four_hits_and_no_guard():
    assert D(hit=RUNS)["OUTCOME"] == "PASS"
    for miss in RUNS:
        assert D(hit=[r for r in RUNS if r != miss])["OUTCOME"] in ("PARTIAL", "NULL")


def test_partial_needs_both_ctrl_seeds_and_not_all_four():
    assert D(hit=CTRL)["OUTCOME"] == "PARTIAL"
    assert D(hit=CTRL + ("REP_s0",))["OUTCOME"] == "PARTIAL"
    assert D(hit=("CTRL_s0", "CTRL_s1", "REP_s1"))["OUTCOME"] == "PARTIAL"


def test_null_and_its_descriptive_labels():
    one = D(hit=("CTRL_s0",))
    assert one["OUTCOME"] == "NULL" and one["label"] == "NULL-ONE"
    assert D(hit=("REP_s0", "REP_s1"))["OUTCOME"] == "NULL" and D(hit=("REP_s0", "REP_s1", "CTRL_s1"))["label"] == "NULL-ONE"
    flatd = D()
    assert flatd["OUTCOME"] == "NULL" and flatd["label"] == "NULL-FLAT"
    assert D(hit=CTRL)["label"] is None and D(hit=RUNS)["label"] is None


def test_every_outcome_name_is_reachable_and_has_a_committed_next():
    seen = {D(controls_ok=False)["OUTCOME"], D(hit=RUNS, guards={"G-ADE": True})["OUTCOME"], D(hit=RUNS)["OUTCOME"], D(hit=CTRL)["OUTCOME"], D()["OUTCOME"]}
    assert seen == set(R.OUTCOME_NAMES) and set(R.COMMITTED_NEXT) == set(R.OUTCOME_NAMES)
    assert "section 8 (b)" in R.COMMITTED_NEXT["HARMS"] and "section 8 (a)" in R.COMMITTED_NEXT["NULL"] and "section 8 (a)" in R.COMMITTED_NEXT["PARTIAL"]


def test_hits_by_run_reads_the_ade6_row_of_each_run():
    ade = {"CTRL_s0": GOOD_ADE, "CTRL_s1": GOOD_ADE, "REP_s0": row(-0.05, -0.1, -0.01), "REP_s1": row(-0.2, -0.3, 0.0)}
    assert R.hits_by_run(ade) == {"CTRL_s0": True, "CTRL_s1": True, "REP_s0": False, "REP_s1": False}


# ------------------------------------------------------------------------------------------------ 6.1 Read A and 6.2 the floor rule
def readA_rows(d1=True, worse=(), fam_missing=(), m1_worse=False):
    t = {}
    for l in R.LABELS:
        fam = R.FAMILY[l]
        better = row(-0.3, -0.4, -0.2) if R.DIRECTION[l] == "lower" else row(0.3, 0.2, 0.4)
        t[l] = flat() if fam in fam_missing else better
    t["ADE6"] = row(-0.30, -0.40, -0.20) if d1 else row(-0.10, -0.2, -0.01)
    for l in worse:
        t[l] = row(0.3, 0.1, 0.5) if R.DIRECTION[l] == "lower" else row(-0.3, -0.5, -0.1)
    if m1_worse:
        t["M1 crawl share"] = row(0.05, 0.03, 0.07)
    return t


def A(**kw):
    return R.read_a({r: readA_rows(**kw) for r in CTRL})


def test_read_a_outcomes():
    assert A()["outcome"] == "DOMINANT" and A()["unamended_label"] == "DOMINANT"
    assert A(d1=False)["outcome"] == "NOT DOMINANT: NO MARGIN"
    assert A(worse=("speed MAE",))["outcome"] == "NOT DOMINANT: MARGIN, NOT DOMINANT"
    assert A(fam_missing=("STRATEGIC",))["unamended_label"] == "MARGIN, NOT DOMINANT" and A(fam_missing=("STRATEGIC",))["D3_both"] is False
    m1 = A(m1_worse=True)
    assert m1["unamended_label"] == "MARGIN, NOT DOMINANT" and m1["D2_both"] is False                  # M1 is a row of the table: separated-worse fails D2 itself
    one = R.read_a({"CTRL_s0": readA_rows(), "CTRL_s1": readA_rows(d1=False)})
    assert one["outcome"] == "NOT DOMINANT: NO MARGIN" and one["per_seed"]["CTRL_s0"]["D1"] and not one["per_seed"]["CTRL_s1"]["D1"]
    d3 = R.read_a({"CTRL_s0": readA_rows(), "CTRL_s1": readA_rows(fam_missing=("STRATEGIC",))})                     # D3 at ONE seed is not enough
    assert d3["D3_both"] is False and d3["unamended_label"] == "MARGIN, NOT DOMINANT" and d3["per_seed"]["CTRL_s0"]["D3"] and not d3["per_seed"]["CTRL_s1"]["D3"]
    d2 = R.read_a({"CTRL_s0": readA_rows(), "CTRL_s1": readA_rows(worse=("speed MAE",))})
    assert d2["D2_both"] is False and d2["outcome"] == "NOT DOMINANT: MARGIN, NOT DOMINANT"


def test_read_a_d1_is_two_f_not_f_over_two():
    rows = {r: readA_rows() for r in CTRL}
    for r in CTRL:
        rows[r]["ADE6"] = row(-0.2414, -0.3, -0.1)                                                       # -0.2414 > -0.2415: D1 fails
    assert R.read_a(rows)["D1_both"] is False
    for r in CTRL:
        rows[r]["ADE6"] = row(-0.2415, -0.3, -0.1)
    assert R.read_a(rows)["D1_both"] is True


def test_read_a_crawl_rows_gate_dominant_only_via_the_a1_clause():
    rows = {r: readA_rows() for r in CTRL}
    out = R.read_a(rows)
    assert out["crawl_rows_pass"] is True
    rows["CTRL_s1"]["L2 curvature MAE (GT>=5.2)"] = row(0.3, 0.1, 0.5)
    out = R.read_a(rows)
    assert out["crawl_rows_pass"] is False and out["outcome"].startswith("NOT DOMINANT") and out["per_seed"]["CTRL_s1"]["L2_sep_worse"] is True


def test_floor_rule_collapse_only_when_all_three_rows_collapse():
    gid = {"ADE6": 0.40, "yaw-rate MAE": 15.86, "curvature MAE": 0.1324}
    ok = {k: 0.5 * v for k, v in gid.items()}                                                              # exactly half: <= 0.5 g_id, inclusive
    assert R.floor_rule(gid, ok)["verdict"] == "COLLAPSE"
    for k in gid:
        bad = dict(ok)
        bad[k] = 0.5 * gid[k] + 1e-6
        r = R.floor_rule(gid, bad)
        assert r["verdict"] == "NOT COLLAPSE" and r["per_row"][k] is False
    neg = dict(gid)
    neg["ADE6"] = -0.01                                                                                    # g_id <= 0 on a row: nothing to collapse
    assert R.floor_rule(neg, {k: -1.0 for k in gid})["verdict"] == "NOT COLLAPSE"
    zero = dict(gid)
    zero["curvature MAE"] = 0.0
    assert R.floor_rule(zero, {k: -1.0 for k in gid})["verdict"] == "NOT COLLAPSE"
    assert R.floor_rule(gid, {k: v for k, v in gid.items()})["verdict"] == "NOT COLLAPSE"


# ------------------------------------------------------------------------------------------------ controls
def test_k_zero_is_exact():
    z = {l: {"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": False} for l in R.LABELS}
    assert R.is_exact_zero(z)[0] is True
    for k in ("delta", "lo", "hi"):
        bad = {l: dict(v) for l, v in z.items()}
        bad["ADE6"][k] = 1e-12
        ok, det = R.is_exact_zero(bad)
        assert ok is False and "ADE6" in det
    assert R.is_exact_zero({"S-POS@30": {"status": "NOT COMPUTED"}})[0] is False                       # nothing computed is not a zero


def test_k_reg_needs_all_four_runs_separated_worse():
    w = row(0.5, 0.3, 0.7)
    assert R.k_reg({r: w for r in RUNS})[0] is True
    for r in RUNS:
        t = {q: w for q in RUNS}
        t[r] = row(0.5, -0.1, 0.7)
        assert R.k_reg(t)[0] is False
    t = {q: w for q in RUNS}
    t["REP_s0"] = row(-0.5, -0.7, -0.3)
    assert R.k_reg(t)[0] is False


def test_k_id_tolerance():
    g = np.zeros((5, 30, 2))
    cl = np.random.default_rng(0).normal(size=(5, 30, 2))
    cl[..., 0] = np.abs(cl[..., 0]) + 1.0      # x > 0, y = 0: a shift in x moves the per-window ADE6 by exactly the shift
    cl[..., 1] = 0.0
    assert R.k_id(cl, cl.copy(), g)[0] is True
    bump = cl.copy()
    bump[2, :, 0] += 5e-5
    ok, det = R.k_id(bump, cl, g)
    assert ok is True and 0 < det["max_abs_ade6_diff_m"] < 1e-4
    bump[2, :, 0] += 1.5e-4
    assert R.k_id(bump, cl, g)[0] is False
    ok4, det4 = R.k_id(cl[:4], cl, g)
    assert ok4 is False and "shape_mismatch" in det4
    assert R.k_id(cl[:4], cl[:4], g)[0] is False       # g and the banked rollout disagree on the window count


def test_per_window_ade6_is_the_mean_over_knots():
    cl = np.zeros((1, 30, 2))
    cl[0, :15, 0] = 2.0
    assert abs(R.per_window_ade6(cl, np.zeros((1, 30, 2)))[0] - 1.0) < 1e-12


def rec_of(n=36, moved=1.0, seed=0):
    rng = np.random.default_rng(seed)
    own = rng.normal(size=(n, 5)).astype(np.float32)
    o = own.copy()
    k = int(round(moved * n))
    o[:k] += 1.0
    return {"own_std": own, "o_std": o, "used_std": o.copy(), "nonspeed_dev": np.zeros(n, np.float32)}


def test_k_reach_thresholds():
    c = rec_of(100, 1.0)
    s = {"o_std": c["o_std"] + 1.0}
    assert R.k_reach(c, s)[0] is True
    assert R.k_reach(rec_of(100, 0.98), s)[0] is False
    assert R.k_reach(rec_of(100, 0.99), s)[0] is True
    part = c["o_std"].copy()
    part[:6] += 1.0
    assert R.k_reach(c, {"o_std": c["o_std"].copy() * 0 + np.where(np.arange(100)[:, None] < 94, c["o_std"] + 1.0, c["o_std"])})[0] is False       # 94 % differ
    assert R.k_reach(c, {"o_std": np.where(np.arange(100)[:, None] < 95, c["o_std"] + 1.0, c["o_std"])})[0] is True


def shuf_rec(**over):
    perm = np.random.default_rng(20261029).permutation(2399)
    win = np.repeat(np.arange(2399), 3)
    meta = {"perm_sha256": R.REG_PERM_SHA, "perm_seed": 20261029, "perm_n": 2399}
    meta.update(over.pop("meta", {}))
    r = {"meta": meta, "win": win, "src_win": perm[win]}
    r.update(over)
    return r


def test_k_shuf_registered():
    assert R.perm_sha256(np.random.default_rng(20261029).permutation(2399)) == R.REG_PERM_SHA
    ok, det = R.k_shuf(shuf_rec())
    assert ok is True and det["n_fixed"] == 0 and det["rows_use_perm_of_window"]
    assert R.k_shuf(shuf_rec(meta={"perm_sha256": "0" * 64}))[0] is False
    assert R.k_shuf(shuf_rec(meta={"perm_seed": 20261030}))[0] is False
    assert R.k_shuf(shuf_rec(meta={"perm_n": 12}))[0] is False
    bad = shuf_rec()
    bad["src_win"] = bad["win"].copy()
    assert R.k_shuf(bad)[0] is False
    off = shuf_rec()
    off["src_win"] = np.roll(off["src_win"], 1)
    assert R.k_shuf(off)[0] is False


def test_the_fixed_point_rule_of_the_shuffle():
    assert R.perm_fixed_ok(np.arange(100)) == (False, 100)
    q = np.roll(np.arange(100), 1)
    q[:4] = np.arange(4)
    assert R.perm_fixed_ok(q)[1] == 4 and R.perm_fixed_ok(q)[0] is False                              # n = 100: 3 is the allowance, 4 is over
    q[:3] = np.arange(3)
    q[3] = 99
    assert R.perm_fixed_ok(q)[0] is True
    r2 = np.roll(np.arange(2399), 1)
    r2[:24] = np.arange(24)
    assert R.perm_fixed_ok(r2)[1] == 24 and R.perm_fixed_ok(r2)[0] is False                           # n = 2399: 1 % = 23.99, so 24 is over
    r2[:24] = np.roll(np.arange(24), 1)
    r2[:23] = np.arange(23)
    assert R.perm_fixed_ok(r2)[0] is True and R.perm_fixed_ok(r2)[1] == 23


def test_k_nonspeed_is_exactly_zero():
    z = {"a": {"nonspeed_dev": np.zeros(10, np.float32)}, "b": {"nonspeed_dev": np.zeros(4, np.float32)}}
    assert R.k_nonspeed(z)[0] is True
    z["b"]["nonspeed_dev"][2] = 1e-9
    assert R.k_nonspeed(z)[0] is False
    assert R.k_nonspeed({})[0] is False


def pins():
    return {"ckpts": {r: {"sha256": R.CKPT_PREFIX[r] + "0" * 56} for r in RUNS}, "tool_sha256": "t" * 64, "wrapper_sha256": "w" * 64, "instrument_sha256": "i" * 64,
            "readout_c": {"sha256": "c" * 64}, "readout_banked": {"sha256": "b" * 64}}


def meta_of(run, mode):
    p = pins()
    m = {"mode": mode, "ckpt_sha256": p["ckpts"][run]["sha256"], "tool_sha256": "t" * 64, "wrapper_sha256": "w" * 64, "instrument_sha256": "i" * 64}
    if mode in ("egoc", "egoshufc"):
        m["readout_sha256"] = "c" * 64
    if mode == "egol":
        m["readout_sha256"] = "b" * 64
    return m


@pytest.mark.parametrize("mode", ["identity", "egoc", "egoshufc", "egol"])
def test_k_sha_every_field(mode):
    assert R.k_sha(meta_of("CTRL_s0", mode), "CTRL_s0", pins())[0] is True
    fields = ["ckpt_sha256", "tool_sha256", "wrapper_sha256", "instrument_sha256"] + (["readout_sha256"] if mode != "identity" else [])
    for f in fields:
        m = meta_of("CTRL_s0", mode)
        m[f] = "f" + m[f][1:]
        ok, bad = R.k_sha(m, "CTRL_s0", pins())
        assert ok is False and f in bad, f
    m = meta_of("CTRL_s0", mode)
    p = pins()
    p["ckpts"]["CTRL_s0"]["sha256"] = "deadbeef" + "0" * 56                                               # a pin that is not the PREREG's printed prefix
    m["ckpt_sha256"] = p["ckpts"]["CTRL_s0"]["sha256"]
    assert R.k_sha(m, "CTRL_s0", p)[0] is False


def test_k_sha_uses_the_banked_readout_for_the_logged_lane_only():
    m = meta_of("CTRL_s1", "egol")
    m["readout_sha256"] = "c" * 64
    assert R.k_sha(m, "CTRL_s1", pins())[0] is False
    m = meta_of("CTRL_s1", "egoc")
    m["readout_sha256"] = "b" * 64
    assert R.k_sha(m, "CTRL_s1", pins())[0] is False


def test_k_rec_tolerance():
    rng = np.random.default_rng(1)
    ego = rng.normal(size=(30, 22)).astype(np.float32)
    mu, sd = np.array([1., 2., 3., 4., 5.]), np.array([2., 2., 3., 3., 4.])
    pred = lambda e: e[:, :5].astype(np.float64) * 3.0 + 7.0          # noqa: E731
    want = ((pred(ego) - mu) / sd).astype(np.float32)
    rec = {"meta": {"cons_mu": mu.tolist(), "cons_sd": sd.tolist()}, "ego": ego, "used_std": want.copy(), "o_std": want.copy()}
    assert R.k_rec(rec, pred)[0] is True
    rec["o_std"] = want + 2e-5
    assert R.k_rec(rec, pred)[0] is False
    rec["o_std"], rec["used_std"] = want + 5e-6, want + 5e-6
    assert R.k_rec(rec, pred)[0] is True
    rec["meta"]["cons_sd"] = (sd * 1.01).tolist()
    assert R.k_rec(rec, pred)[0] is False


def test_k_floor_is_bit_for_bit():
    a = np.random.default_rng(2).normal(size=(10, 30, 2))
    assert R.k_floor(a, a.copy())[0] is True
    b = a.copy()
    b[3, 4, 1] = np.nextafter(b[3, 4, 1], 9)
    assert R.k_floor(a, b)[0] is False
    assert R.k_floor(a, a[:9])[0] is False


# ------------------------------------------------------------------------------------------------ the four families, the estimator row, the PREREG pin
def test_families_report_covers_the_four_families_and_states_what_is_not_computed():
    tabs = {"egoc-identity": by_run({r: {"S-POS@30": {"status": "NOT COMPUTED"}} for r in RUNS}), "egoc-P10": by_run()}
    rep = R.families_report(tabs)
    assert set(rep) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC", "RULES"}
    assert rep["STRATEGIC"]["rows"]["S-POS@30"]["egoc-identity"]["CTRL_s0"] == "NOT COMPUTED"
    assert "NOT COMPUTED" in rep["STRATEGIC"]["note"] and "NOT COMPUTABLE" in rep["TACTICAL"]["note"] and "NOT BUILT" in rep["LONGITUDINAL"]["note"]
    assert set(rep["LATERAL"]["rows"]) == {"heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)"}
    assert set(rep["LONGITUDINAL"]["rows"]) == {"speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"}
    assert set(rep["TACTICAL"]["rows"]) == {"TAC-LAT", "TAC-LON"} and set(rep["RULES"]["rows"]) == {R.VIOL, R.VIOL_OL}


def test_the_bar_and_runs_are_the_preregs():
    assert R.RUNS == ("CTRL_s0", "CTRL_s1", "REP_s0", "REP_s1") and R.CTRL_RUNS == ("CTRL_s0", "CTRL_s1") and R.GUARD_RUNS_MIN == 3
    assert (R.N_WINDOWS, R.N_EPISODES, R.N_TICKS, R.N_BOOT, R.SHUF_SEED, R.SHUF_N) == (2399, 141, 3, 2000, 20261029, 2399)
    assert R.CODE_IDX == [46, 56, 57, 58, 59] and R.TOL_K_ID == 1e-4 and R.TOL_K_REC == 1e-5 and (R.REACH_EGOC_MIN, R.REACH_SHUF_MIN) == (0.99, 0.95)
    assert len(R.LABELS) == 19 and R.FLOOR_ROWS == ("ADE6", "yaw-rate MAE", "curvature MAE")


def test_the_prereg_pin_and_its_sentences(tmp_path):
    src = os.environ.get("SPEEDSUB_PREREG")
    if not src or not Path(src).is_file():
        pytest.skip("SPEEDSUB_PREREG=<PREREG_V8F_SPEEDSUB_T1.md at 842ec2ac> not given")
    assert R.prereg_problems(src) == []
    txt = Path(src).read_bytes().replace(b"\r\n", b"\n")
    p = tmp_path / "edited.md"
    p.write_bytes(txt.replace(b"d_X \xe2\x89\xa4 \xe2\x88\x920.06035", b"d_X \xe2\x89\xa4 \xe2\x88\x920.06034"))
    assert len(R.prereg_problems(str(p))) >= 2
    q = tmp_path / "crlf.md"
    q.write_bytes(txt.replace(b"\n", b"\r\n"))
    assert R.prereg_problems(str(q)) == []                                                                  # LF-normalised: a Windows checkout reads as the same text


# ------------------------------------------------------------------------------------------------ the light layer: --preflight / --smoke-check / --make-manifest, driven through main()
import json                                                                                                  # noqa: E402

MU5, SD5 = np.array([1., 2., 3., 4., 5.]), np.array([2., 2., 3., 3., 4.])
NW = 2399
NROW = NW * 3
FAKE_PRED = lambda e: e[:, :5].astype(np.float64) * 3.0 + 7.0                                                  # noqa: E731


class FakeRO:
    def predict_raw(self, _x, ego):
        return FAKE_PRED(ego)


def _base_windows():
    rng = np.random.default_rng(11)
    return {"g": rng.normal(size=(6, 30, 2)), "ws": np.arange(6) * 3, "eid": np.array([0, 0, 1, 1, 2, 2]), "clip_index": np.arange(6) + 100, "v0": rng.uniform(1, 9, 6),
            "fl": rng.normal(size=(6, 30, 2)), "ol": rng.normal(size=(6, 30, 2)), "ha0": rng.normal(size=(6, 30, 2)), "cl": rng.normal(size=(6, 30, 2))}


def write_dump(d, base, cl=None, fl=None):
    os.makedirs(d, exist_ok=True)
    for i, sl in enumerate((slice(0, 2), slice(2, 4), slice(4, 6))):
        z = {k: v[sl] for k, v in base.items()}
        if cl is not None:
            z["cl"] = cl[sl]
        if fl is not None:
            z["fl"] = fl[sl]
        np.savez(os.path.join(d, f"ep{i:04d}.npz"), **z)


def write_rec(path, run, mode, **over):
    rng = np.random.default_rng({"identity": 0, "egoc": 1, "egoshufc": 2}[mode])
    ego = rng.normal(size=(NROW, 22)).astype(np.float32)
    own = rng.normal(size=(NROW, 5)).astype(np.float32)
    egoc_std = ((FAKE_PRED(ego) - MU5) / SD5).astype(np.float32)
    o = {"identity": own.copy(), "egoc": egoc_std, "egoshufc": egoc_std + 1.0}[mode]
    meta = meta_of(run, mode)
    meta.update({"aligned": True, "tool_rc": 0, "alignment_with_tool_ticks": "ok", "cons_mu": MU5.tolist(), "cons_sd": SD5.tolist()})
    win = np.repeat(np.arange(NW), 3)
    z = {"ego": ego, "own_std": own, "o_std": o, "used_std": o.copy(), "nonspeed_dev": np.zeros(NROW, np.float32), "win": win, "src_win": win.copy()}
    if mode == "egoshufc":
        perm = np.random.default_rng(20261029).permutation(2399)
        z["src_win"] = perm[win]
        meta.update({"perm_sha256": R.REG_PERM_SHA, "perm_seed": 20261029, "perm_n": 2399})
    meta.update(over.pop("meta", {}))
    z.update(over)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, meta_json=np.array(json.dumps(meta)), **z)


def light_world(tmp_path, monkeypatch, kind="preflight", run="CTRL_s0"):
    """Writes the rollouts' light outputs for ``run`` in the layout the pin script renders; the banked dump of ``run`` is redirected to a synthetic one."""
    out_root = tmp_path / "out"
    base = _base_windows()
    banked = tmp_path / "banked"
    write_dump(str(banked), base)
    monkeypatch.setitem(R.BANKED_DUMP, run, str(banked))
    monkeypatch.setattr(R, "sr_load_readout", lambda heavy, path, key, instr: FakeRO())
    pf = tmp_path / "pins.json"
    p = pins()
    p["readout_c"]["path"] = "unused"
    pf.write_text(json.dumps(p))
    modes = ("identity", "egoc", "egoshufc") if kind == "preflight" else ("identity", "egoc")
    root = out_root / ("SMOKE" if kind == "smoke" else "") / run
    for m in modes:
        tag = f"SST1_{'SMOKE_' if kind == 'smoke' else ''}{run}_{m}"
        write_rec(str(root / m / "ticks" / f"speedsub_{tag}.npz"), run, m)
        if kind == "preflight":
            write_dump(str(root / m / f"dump_{tag}"), base, cl=base["cl"] + (0.5 if m != "identity" else 0.0))
    return {"out_root": str(out_root), "pins": str(pf), "base": base, "root": root, "run": run, "tmp": tmp_path}


def run_light(w, kind, out_name="light.json"):
    out = str(w["tmp"] / out_name)
    flag = "--preflight" if kind == "preflight" else "--smoke-check"
    rc = R.main([flag, "--pins", w["pins"], "--out-root", w["out_root"], "--run", w["run"], "--out", out])
    return rc, (json.load(open(out, encoding="utf-8")) if os.path.exists(out) else None)


def test_preflight_passes_on_a_clean_world_and_reads_every_light_control(tmp_path, monkeypatch, capsys):
    w = light_world(tmp_path, monkeypatch)
    rc, out = run_light(w, "preflight")
    assert rc == 0 and out["controls_PASS"] is True
    names = set(out["controls"])
    for n in ("K-NONSPEED", "K-REC", "K-REACH", "K-SHUF", "K-ID", "K-SHA:identity", "K-SHA:egoc", "K-SHA:egoshufc", "K-FLOOR:egoc", "K-WIN:egoshufc", "K-ALIGN:identity"):
        assert n in names and out["controls"][n]["ok"] is True, n
    assert "ZZSPEEDSUB-T1-PREFLIGHT-OK" in capsys.readouterr().out


@pytest.mark.parametrize("breaker,control", [
    ("id_drift", "K-ID"), ("gt_differs", "K-WIN:egoc"), ("floor_ulp", "K-FLOOR:egoshufc"), ("nonspeed", "K-NONSPEED"), ("no_reach", "K-REACH"), ("wrong_ckpt", "K-SHA:egoc"), ("windows", "K-WIN:egoc"),
    ("tool_rc", "K-ALIGN:identity"), ("rec_off", "K-REC"), ("shuf_perm", "K-SHUF")])
def test_each_preflight_control_can_stop_the_chain(tmp_path, monkeypatch, capsys, breaker, control):
    w = light_world(tmp_path, monkeypatch)
    run, root, base = w["run"], w["root"], w["base"]
    tag = lambda m: f"SST1_{run}_{m}"                                                                           # noqa: E731
    rec = lambda m: str(root / m / "ticks" / f"speedsub_{tag(m)}.npz")                                           # noqa: E731
    dump = lambda m: str(root / m / f"dump_{tag(m)}")                                                            # noqa: E731
    if breaker == "id_drift":
        rad = base["cl"] - base["g"]
        rad = rad / np.linalg.norm(rad, axis=-1, keepdims=True)
        write_dump(dump("identity"), base, cl=base["cl"] + 2e-4 * rad)                                      # radial: every knot moves its error norm by exactly 2e-4
    elif breaker == "floor_ulp":
        fl = base["fl"].copy()
        fl[3, 4, 1] = np.nextafter(fl[3, 4, 1], 9)
        write_dump(dump("egoshufc"), base, cl=base["cl"] + 0.5, fl=fl)
    elif breaker == "nonspeed":
        nd = np.zeros(NROW, np.float32)
        nd[5] = 1e-9
        write_rec(rec("egoc"), run, "egoc", nonspeed_dev=nd)
    elif breaker == "no_reach":
        z = np.load(rec("egoc"))
        own = z["own_std"]
        write_rec(rec("egoc"), run, "egoc", o_std=own.copy(), used_std=own.copy())
    elif breaker == "wrong_ckpt":
        write_rec(rec("egoc"), run, "egoc", meta={"ckpt_sha256": "f" * 64})
    elif breaker == "windows":
        b2 = dict(base)
        b2["ws"] = base["ws"] + 1
        write_dump(dump("egoc"), b2, cl=base["cl"] + 0.5)
    elif breaker == "gt_differs":
        b3 = dict(base)
        b3["g"] = base["g"] + 1e-9
        write_dump(dump("egoc"), b3, cl=base["cl"] + 0.5)
    elif breaker == "tool_rc":
        write_rec(rec("identity"), run, "identity", meta={"tool_rc": 3})
    elif breaker == "rec_off":
        z = np.load(rec("egoc"))
        write_rec(rec("egoc"), run, "egoc", used_std=z["used_std"] + 3e-5, o_std=z["o_std"] + 3e-5)
    elif breaker == "shuf_perm":
        z = np.load(rec("egoshufc"))
        write_rec(rec("egoshufc"), run, "egoshufc", src_win=np.roll(z["src_win"], 1))
    rc, out = run_light(w, "preflight")
    assert rc == 1 and out["controls_PASS"] is False and out["controls"][control]["ok"] is False, (control, {k: v["ok"] for k, v in out["controls"].items()})
    assert "ZZSPEEDSUB-T1-PREFLIGHT-STOP" in capsys.readouterr().out


def test_preflight_stops_with_rc2_when_a_rollout_is_missing(tmp_path, monkeypatch, capsys):
    w = light_world(tmp_path, monkeypatch)
    os.remove(str(w["root"] / "egoshufc" / "ticks" / f"speedsub_SST1_{w['run']}_egoshufc.npz"))
    rc, out = run_light(w, "preflight")
    assert rc == 2 and out is None and "ZZSPEEDSUB-T1-PREFLIGHT-STOP missing" in capsys.readouterr().out


def test_smoke_check_reads_only_the_light_controls_and_never_the_banked_ones(tmp_path, monkeypatch, capsys):
    w = light_world(tmp_path, monkeypatch, kind="smoke")
    rc, out = run_light(w, "smoke")
    assert rc == 0 and out["controls_PASS"] is True and out["kind"] == "smoke"
    assert not any(n.startswith(("K-ID", "K-FLOOR", "K-WIN", "K-SHUF")) for n in out["controls"])
    assert out["controls"]["K-REACH"]["detail"]["note"].startswith("egoshufc not rolled")
    assert "ZZSPEEDSUB-T1-SMOKE-OK" in capsys.readouterr().out
    # a smoke whose egoc did not move the cells stops the chain
    p = str(w["root"] / "egoc" / "ticks" / f"speedsub_SST1_SMOKE_{w['run']}_egoc.npz")
    z = np.load(p)
    own = z["own_std"]
    write_rec(p, w["run"], "egoc", o_std=own.copy(), used_std=own.copy())
    rc, out = run_light(w, "smoke", "light2.json")
    assert rc == 1 and out["controls"]["K-REACH"]["ok"] is False


def test_load_dump_light_concatenates_episodes_in_sorted_order(tmp_path):
    base = _base_windows()
    write_dump(str(tmp_path / "d"), base)
    d = R.load_dump_light(str(tmp_path / "d"))
    assert np.array_equal(d["eid"], base["eid"]) and np.array_equal(d["cl"], base["cl"]) and d["g"].shape == (6, 30, 2)


def test_make_manifest_is_what_the_read_loads(tmp_path):
    out = str(tmp_path / "m.json")
    rc = R.main(["--make-manifest", "--pins", "P.json", "--out-root", str(tmp_path / "o").replace(chr(92), "/"), "--out", out])
    man = json.load(open(out, encoding="utf-8"))
    assert rc == 0 and set(man["runs"]) == set(RUNS) and man["pins"] == "P.json"
    for r in RUNS:
        modes = set(man["runs"][r]["modes"])
        assert modes == set(R.MODES) | (set(R.BESIDE_MODES) if r.startswith("CTRL") else set()), r
        assert man["runs"][r]["banked"]["dump_dir"] == R.BANKED_DUMP[r]
    m, missing = R.load_manifest(out)
    assert len(missing) == len(RUNS) * len(R.MODES) * 4                                                          # nothing on disk: every dump_dir / ticks / json / rec is named as missing
    for r in RUNS:                                                                                              # the chain's files make it load clean
        for mode in R.MODES:
            e = man["runs"][r]["modes"][mode]
            os.makedirs(e["dump_dir"], exist_ok=True)
            for k in ("ticks", "json", "rec"):
                os.makedirs(os.path.dirname(e[k]), exist_ok=True)
                open(e[k], "wb").close()
    assert R.load_manifest(out)[1] == []


def test_the_manifest_tags_are_the_pin_scripts_tags(tmp_path):
    name = "v8f_speedsub_t1_pin.py"
    cands = [Path(os.environ["SPEEDSUB_PIN"])] if os.environ.get("SPEEDSUB_PIN") else []
    cands += [HERE.parent / "stack" / "scripts" / name] + [q / "stack" / "scripts" / name for q in HERE.parents]       # the build layout, then any repo checkout above this file
    found = next((c for c in cands if c.is_file()), None)
    if found is None:
        pytest.skip("v8f_speedsub_t1_pin.py not found beside or above this test (set SPEEDSUB_PIN)")
    spec = importlib.util.spec_from_file_location("pin_under_test", str(found))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    out = str(tmp_path / "m.json")
    R.main(["--make-manifest", "--pins", "P.json", "--out-root", "X:/o", "--out", out])
    man = json.load(open(out, encoding="utf-8"))
    for r in RUNS:
        for mode, e in man["runs"][r]["modes"].items():
            assert e["tag"] == P.tag_of(r, mode), (r, mode)
            assert e["out_dir"] == f"X:/o/{r}/{mode}"


def test_cli_flags_and_their_requirements():
    ap = R.build_parser()
    a = ap.parse_args(["--smoke-check", "--run", "REP_s1", "--pins", "p"])
    assert a.smoke_check and a.run == "REP_s1" and not a.preflight and not a.beside
    with pytest.raises(SystemExit):
        ap.parse_args(["--preflight", "--run", "CTRL_s9"])
    with pytest.raises(SystemExit):
        R.main(["--preflight"])                                                                                  # --pins is required
    with pytest.raises(SystemExit):
        R.main(["--beside"])                                                                                     # --manifest is required


def test_the_manifest_paths_are_where_the_pin_scripts_rollouts_write(tmp_path):
    """manifest -> the files the tool writes under `--out-dir D --dump-ticks D/ticks --tag T`: D/dump_T, D/t1v8f_T.json, D/ticks/ticks_T.npz, and the wrapper's D/ticks/speedsub_T.npz; the preflight reads the same."""
    root = str(tmp_path / "o").replace(chr(92), "/")
    out = str(tmp_path / "m.json")
    R.main(["--make-manifest", "--pins", "P.json", "--out-root", root, "--out", out])
    man = json.load(open(out, encoding="utf-8"))
    for r in RUNS:
        for mode, e in man["runs"][r]["modes"].items():
            t, d = f"SST1_{r}_{mode}", f"{root}/{r}/{mode}"
            assert (e["dump_dir"], e["json"], e["ticks"], e["rec"]) == (f"{d}/dump_{t}", f"{d}/t1v8f_{t}.json", f"{d}/ticks/ticks_{t}.npz", f"{d}/ticks/speedsub_{t}.npz"), (r, mode)


class _FakeHeavy:
    calls = []

    def __init__(self, orc_dir, work_dir, n_boot):
        import types
        self.A = types.SimpleNamespace(dump=lambda d: {"dir": d})

    def tables(self, dumps, packs, x, y, ref):
        _FakeHeavy.calls.append({"x": x, "y": y, "dumps": {s: {m: v["dir"] for m, v in dm.items()} for s, dm in dumps.items()}, "ref": {s: v["dir"] for s, v in ref.items()}})
        return {s: {"ADE6": {"delta": -0.1 * (s + 1), "lo": -0.2, "hi": -0.01, "separated": True, "mean_arm": 1.0, "mean_ref": 1.1, "status": None, "extra": 5}, "_hidden": {"delta": 9.0}} for s in (0, 1)}


def test_the_beside_lane_reads_egol_minus_identity_l_on_ctrl_only_and_decides_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "Heavy", _FakeHeavy)
    _FakeHeavy.calls.clear()
    root = str(tmp_path / "o").replace(chr(92), "/")
    man_p, out_p = str(tmp_path / "m.json"), str(tmp_path / "beside.json")
    R.main(["--make-manifest", "--pins", "P.json", "--out-root", root, "--out", man_p])
    man = json.load(open(man_p, encoding="utf-8"))
    rc = R.main(["--beside", "--manifest", man_p, "--out", out_p])
    out = json.load(open(out_p, encoding="utf-8"))
    assert rc == 0 and len(_FakeHeavy.calls) == 1
    c = _FakeHeavy.calls[0]
    assert (c["x"], c["y"]) == ("egol", "identity_l")
    for s in (0, 1):
        assert c["dumps"][s] == {m: man["runs"][f"CTRL_s{s}"]["modes"][m]["dump_dir"] for m in R.BESIDE_MODES}
        assert c["ref"][s] == man["runs"][f"CTRL_s{s}"]["modes"]["identity_l"]["dump_dir"]
    t = out["egol_minus_identity_l"]
    assert set(t) == {"CTRL_s0", "CTRL_s1"} and set(t["CTRL_s0"]) == {"ADE6"} and t["CTRL_s1"]["ADE6"]["delta"] == -0.2 and "extra" not in t["CTRL_s0"]["ADE6"]
    assert "OPEN" in out["lane"] and "D-SPEEDBAND-EGO-ECHO-1" in out["lane"] and "never a capability row" in out["role"] and "decides nothing" in out["role"]
    assert "OUTCOME" not in out and out["tier"] == R.TIER_T1


# ------------------------------------------------------------------------------------------------ main_read end to end on a FAKE heavy layer: the wiring from the tables to the decision
import types                                                                                                  # noqa: E402


def _e2e_base():
    rng = np.random.default_rng(5)
    n = R.N_WINDOWS
    return {"g": rng.normal(size=(n, 30, 2)), "cl": rng.normal(size=(n, 30, 2)), "ol": rng.normal(size=(n, 30, 2)), "fl": rng.normal(size=(n, 30, 2)), "ha0": rng.normal(size=(n, 30, 2)),
            "v0": rng.uniform(1, 9, n), "ws": np.arange(n) * 3, "eid": np.arange(n) % R.N_EPISODES, "clip_index": np.arange(n) + 7}


class FakeRig:
    """`scen(kind, x, y, run, label) -> row`; kind in {"tab", "viol", "viol_ol"}. Dumps carry the run and mode they were built for."""

    def __init__(self, scen, floor):
        self.scen, self.floor, self.dumps = scen, floor, {}
        self.A = types.SimpleNamespace(dump=lambda d: self.dumps[d], tacdk_context=lambda ref: {"ref_run": ref.get("run")})
        self.rig = types.SimpleNamespace(backend=types.SimpleNamespace(analyze_t1=self.analyze_t1), pair_table=self.pair_table)
        self.tab_calls = []

    def analyze_t1(self, entry, label, seed):
        return types.SimpleNamespace(controls={"K-HEADS": (True, "fake")}, extra={"tag": entry["tag"]})

    def tables(self, dumps, packs, x, y, ref):
        runs = {s: dumps[s][x]["run"] for s in (0, 1)}
        self.tab_calls.append((x, y, runs))
        return {s: {l: self.scen("tab", x, y, runs[s], l) for l in R.LABELS} for s in (0, 1)}

    def viol_row(self, dx, dy, arm="cl"):
        return self.scen("viol_ol" if arm == "ol" else "viol", dx["mode"], dy["mode"], dx["run"], R.VIOL)

    def pair_table(self, sd, cache, x, y, ref):
        return {lab: {"delta": self.floor(x, sd[x]["run"], lab)} for lab in R.FLOOR_ROWS}, None


def _flat_row():
    return {"delta": 0.0, "lo": -0.05, "hi": 0.05, "separated": False, "n": "2399/141"}


def _zero_row():
    return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": False, "n": "2399/141"}


def default_scen(over=None):
    over = over or {}

    def scen(kind, x, y, run, label):
        if (kind, x, y, run, label) in over:
            return over[(kind, x, y, run, label)]
        if (kind, x, y, None, label) in over:
            return over[(kind, x, y, None, label)]
        if x == y == "identity":
            return _zero_row()
        if x == "egoshufc" and y == "identity" and label == "ADE6":
            return row(0.5, 0.3, 0.7)                                                                          # the deliberate regression, separated-worse at every run (K-REG)
        return _flat_row()
    return scen


def e2e_world(tmp_path, monkeypatch, over=None, floor=None, bad_rec=None, bad_dump=None):
    root = str(tmp_path / "o").replace(chr(92), "/")
    man_p, pins_p = str(tmp_path / "m.json"), str(tmp_path / "pins.json")
    p = pins()
    p["readout_c"]["path"] = "unused"
    open(pins_p, "w").write(json.dumps(p))
    R.main(["--make-manifest", "--pins", pins_p, "--out-root", root, "--out", man_p])
    man = json.load(open(man_p, encoding="utf-8"))
    base = _e2e_base()
    fr = FakeRig(default_scen(over), floor or (lambda x, run, lab: 0.1 if x == "REP_id" else 0.09))
    for r in RUNS:
        fr.dumps[R.BANKED_DUMP[r]] = {**base, "run": r, "mode": "banked"}
        for mode in R.MODES:
            e = man["runs"][r]["modes"][mode]
            cl = {"identity": base["cl"], "egoc": base["cl"] + 0.01, "egoshufc": base["cl"] + 0.05}[mode]
            d = {**base, "cl": cl, "run": r, "mode": mode}
            if bad_dump and bad_dump[0] == (r, mode):
                d = bad_dump[1](d)
            fr.dumps[e["dump_dir"]] = d
            os.makedirs(os.path.dirname(e["rec"]), exist_ok=True)
            for k in ("ticks", "json"):
                os.makedirs(os.path.dirname(e[k]), exist_ok=True)
                open(e[k], "wb").close()
            os.makedirs(e["dump_dir"], exist_ok=True)
            kw = {}
            if bad_rec and bad_rec[0] == (r, mode):
                kw = bad_rec[1]
            write_rec(e["rec"], r, mode, **kw)
    monkeypatch.setattr(R, "Heavy", lambda orc_dir, work_dir, n_boot: fr)
    monkeypatch.setattr(R, "prereg_problems", lambda path, pin=None: [])
    monkeypatch.setattr(R, "sr_load_readout", lambda heavy, path, key, instr: FakeRO())
    return {"man": man_p, "out": str(tmp_path / "read.json"), "fr": fr}


def _egoc_std():
    rng = np.random.default_rng(1)                                                                        # write_rec's egoc stream
    ego = rng.normal(size=(NROW, 22)).astype(np.float32)
    return ((FAKE_PRED(ego) - MU5) / SD5).astype(np.float32)


def sub(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def read_e2e(w):
    rc = R.main(["--manifest", w["man"], "--prereg", "unused.md", "--out", w["out"]])
    return rc, json.load(open(w["out"], encoding="utf-8"))


def hit_rows(runs):
    return {("tab", "egoc", "identity", r, "ADE6"): row(-0.30, -0.40, -0.20) for r in runs}


def test_e2e_every_outcome_through_main_read(tmp_path, monkeypatch):
    # a clean flat world: all four runs null
    w = e2e_world(tmp_path, monkeypatch)
    rc, out = read_e2e(w)
    assert rc == 0 and out["controls_PASS"] is True and out["OUTCOME"] == "NULL" and out["label"] == "NULL-FLAT"
    assert out["tier"] == R.TIER_T1 and out["claim_row"] == "H-V8F-SPEEDSUB-T1-1" and out["guards"]["G-VIOL"]["active"] is True
    assert "read_A_on_CTRL" not in out and all(out["controls"].values())
    assert set(out["four_families"]) >= {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"} and "floor_rule_H_V8F_SPEEDSUB_T1_2" in out
    # PASS: a hit at all four runs and no guard; Read A is then attached
    good = {("tab", "egoc", "P10", r, l): v for r in CTRL for l, v in readA_rows().items()}                  # Read A reads `egoc - P10`: DOMINANT there, flat on `egoc - identity`
    rc, out = read_e2e(e2e_world(sub(tmp_path, "a"), monkeypatch, over={**hit_rows(RUNS), **good}))
    assert out["OUTCOME"] == "PASS" and out["hits"] == {r: True for r in RUNS} and out["read_A_on_CTRL"]["outcome"] == "DOMINANT"
    # PARTIAL: both CTRL seeds hit, REP does not
    rc, out = read_e2e(e2e_world(sub(tmp_path, "b"), monkeypatch, over=hit_rows(CTRL)))
    assert out["OUTCOME"] == "PARTIAL"
    # NULL-ONE: only one CTRL run hits
    rc, out = read_e2e(e2e_world(sub(tmp_path, "c"), monkeypatch, over=hit_rows(["CTRL_s0"])))
    assert out["OUTCOME"] == "NULL" and out["label"] == "NULL-ONE"


def test_e2e_the_guards_read_the_rows_they_are_named_for(tmp_path, monkeypatch):
    allhit = hit_rows(RUNS)
    worse = row(0.5, 0.3, 0.7)
    # G-VIOL reads the cl-arm violation row, at >= 3 of 4 runs
    viol = lambda rr: {("viol", "egoc", "identity", r, R.VIOL): worse for r in rr}                             # noqa: E731
    cases = {
        "gviol3": ({**allhit, **viol(("CTRL_s0", "CTRL_s1", "REP_s0"))}, "HARMS", "G-VIOL"),
        "gviol2": ({**allhit, **viol(("CTRL_s0", "REP_s1"))}, "PASS", None),
        "gviol_ol_row_only": ({**allhit, **{("viol_ol", "egoc", "identity", r, R.VIOL): worse for r in RUNS}}, "PASS", None),
        "gworse3": ({**allhit, **{("tab", "egoc", "identity", r, "speed MAE"): worse for r in ("CTRL_s0", "REP_s0", "REP_s1")}}, "HARMS", "G-WORSE"),
        "gade1": ({**allhit, ("tab", "egoc", "identity", "REP_s1", "ADE6"): worse}, "HARMS", "G-ADE"),
        "gnew": ({**allhit, **{("tab", "egoc", "P10", r, "speed MAE"): worse for r in CTRL}, **{("tab", "identity", "P10", r, "speed MAE"): worse for r in CTRL}}, "PASS", None),
    }
    for i, (name, (over, outcome, fired)) in enumerate(cases.items()):
        rc, out = read_e2e(e2e_world(sub(tmp_path, name), monkeypatch, over=over))
        assert out["OUTCOME"] == outcome, (name, out["OUTCOME"], out["flags"])
        assert (out["flags"]["guards_fired"] if outcome == "HARMS" else []) == ([fired] if fired else []), (name, out["flags"])
    # G-NEW (the stricter reading): worse than P10 for egoc at both CTRL seeds while identity is NOT worse than P10 there
    over = {**allhit, **{("tab", "egoc", "P10", r, "speed MAE"): worse for r in CTRL}}
    rc, out = read_e2e(e2e_world(sub(tmp_path, "gnew2"), monkeypatch, over=over))
    assert out["OUTCOME"] == "HARMS" and out["flags"]["guards_fired"] == ["G-NEW"]


def test_e2e_a_failed_control_refuses_and_names_it(tmp_path, monkeypatch):
    cases = {
        "k_zero": dict(over={("tab", "identity", "identity", "REP_s1", "ADE6"): {"delta": 1e-9, "lo": 0.0, "hi": 0.0, "separated": False}}, key="K-ZERO:REP_s1"),
        "k_reg": dict(over={("tab", "egoshufc", "identity", "CTRL_s1", "ADE6"): _flat_row()}, key="K-REG"),
        "k_sha": dict(bad_rec=(("REP_s0", "egoc"), {"meta": {"tool_sha256": "x" * 64}}), key="K-SHA:REP_s0:egoc"),
        "k_reach_plan": dict(bad_dump=(("CTRL_s1", "egoc"), lambda d: {**d, "cl": d["cl"] - 0.01}), key="K-REACH-PLAN:CTRL_s1"),
        "k_floor": dict(bad_dump=(("REP_s1", "egoshufc"), lambda d: {**d, "fl": d["fl"] + 1e-12}), key="K-FLOOR:REP_s1:egoshufc"),
        "k_id": dict(bad_dump=(("CTRL_s0", "identity"), lambda d: {**d, "cl": d["cl"] + 1e-3}), key="K-ID:CTRL_s0"),
        "k_win": dict(bad_dump=(("REP_s0", "identity"), lambda d: {**d, "eid": d["eid"][::-1].copy()}), key="K-WIN:REP_s0:identity"),
        "k_rec": dict(bad_rec=(("CTRL_s1", "egoc"), {"used_std": np.zeros((NROW, 5), np.float32)}), key="K-REC:CTRL_s1"),
        "k_nonspeed": dict(bad_rec=(("REP_s1", "identity"), {"nonspeed_dev": np.full(NROW, 1e-9, np.float32)}), key="K-NONSPEED"),
        "k_align_rc": dict(bad_rec=(("CTRL_s0", "egoshufc"), {"meta": {"tool_rc": 3}}), key="K-ALIGN:CTRL_s0:egoshufc"),
        "k_reach": dict(bad_rec=(("REP_s0", "egoshufc"), {"o_std": _egoc_std(), "used_std": _egoc_std()}), key="K-REACH:REP_s0"),                 # the shuffle arm reads the egoc cells: it moved nothing
        "k_shuf": dict(bad_rec=(("CTRL_s1", "egoshufc"), {"src_win": np.zeros(NROW, np.int64)}), key="K-SHUF:CTRL_s1"),
    }
    for name, c in cases.items():
        w = e2e_world(sub(tmp_path, name), monkeypatch, over={**hit_rows(RUNS), **c.get("over", {})}, bad_rec=c.get("bad_rec"), bad_dump=c.get("bad_dump"))
        rc, out = read_e2e(w)
        assert rc == 2 and out["OUTCOME"] == "REFUSED" and out["controls_PASS"] is False and out["controls"][c["key"]] is False, (name, [k for k, v in out["controls"].items() if not v])


def test_e2e_the_floor_rule_is_beside_and_reads_rep_minus_ctrl(tmp_path, monkeypatch):
    # seed 0: REP^identity - CTRL^identity = +0.40 and REP^egoc - CTRL^egoc = +0.15 (<= 0.5 * 0.40) on all three rows: COLLAPSE; seed 1: +0.30 (> 0.20): NOT COLLAPSE
    floor = lambda x, run, lab: 0.40 if x == "REP_id" else (0.15 if run.endswith("s0") else 0.30)                # noqa: E731
    rc, out = read_e2e(e2e_world(tmp_path, monkeypatch, floor=floor, over=hit_rows(RUNS)))
    f = out["floor_rule_H_V8F_SPEEDSUB_T1_2"]
    s0, s1 = f["s0 (decisive, beside)"], f["s1 (reported)"]
    assert s0["verdict"] == "COLLAPSE" and s0["g_id"]["ADE6"] == 0.40 and s0["g_c"]["ADE6"] == 0.15 and s0["g_c"]["yaw-rate MAE"] == 0.15
    assert s1["verdict"] == "NOT COLLAPSE" and s1["g_c"]["curvature MAE"] == 0.30 and s1["per_row"] == {r: False for r in R.FLOOR_ROWS}
    assert out["OUTCOME"] == "PASS"                                                                              # the floor rule is beside: it never changes the outcome
    rc, out = read_e2e(e2e_world(sub(tmp_path, "n"), monkeypatch, floor=lambda x, run, lab: 0.40 if x == "REP_id" else 0.30))
    assert out["floor_rule_H_V8F_SPEEDSUB_T1_2"]["s0 (decisive, beside)"]["verdict"] == "NOT COLLAPSE" and out["OUTCOME"] == "NULL"


def test_e2e_the_tables_asked_for_are_the_registered_contrasts(tmp_path, monkeypatch):
    w = e2e_world(tmp_path, monkeypatch)
    read_e2e(w)
    asked = {(x, y) for x, y, _ in w["fr"].tab_calls}
    assert asked == {("egoc", "identity"), ("egoc", "P10"), ("identity", "P10"), ("egoshufc", "identity"), ("identity", "identity")}
    fams = {(x, y, tuple(runs[s] for s in (0, 1))) for x, y, runs in w["fr"].tab_calls}
    assert ("egoc", "identity", ("CTRL_s0", "CTRL_s1")) in fams and ("egoc", "identity", ("REP_s0", "REP_s1")) in fams


def test_e2e_a_missing_run_and_a_bad_prereg_write_nothing_decisive(tmp_path, monkeypatch, capsys):
    w = e2e_world(tmp_path, monkeypatch)
    man = json.load(open(w["man"], encoding="utf-8"))
    os.remove(man["runs"]["REP_s1"]["modes"]["egoshufc"]["rec"])
    rc = R.main(["--manifest", w["man"], "--prereg", "unused.md", "--out", w["out"]])
    assert rc == 4 and not os.path.exists(w["out"]) and "REP_s1/egoshufc/rec" in capsys.readouterr().out
    w = e2e_world(sub(tmp_path, "p"), monkeypatch)
    monkeypatch.setattr(R, "prereg_problems", lambda path, pin=None: ["sentence moved"])
    rc = R.main(["--manifest", w["man"], "--prereg", "unused.md", "--out", w["out"]])
    assert rc == 2 and not os.path.exists(w["out"]) and "sentence moved" in capsys.readouterr().err
    rc = R.main(["--manifest", w["man"], "--out", w["out"]])                                                    # no --prereg at all
    assert rc == 2 and not os.path.exists(w["out"])


def test_e2e_the_heads_pipelines_own_controls_are_controls(tmp_path, monkeypatch):
    w = e2e_world(tmp_path, monkeypatch, over=hit_rows(RUNS))
    fr = w["fr"]
    fr.rig.backend.analyze_t1 = lambda entry, label, seed: types.SimpleNamespace(controls={"K-HEADS": (entry["tag"].endswith("REP_s1_egoc"), "fake")}, extra={})
    rc, out = read_e2e(w)
    assert rc == 2 and out["OUTCOME"] == "REFUSED" and [k for k, v in out["controls"].items() if not v] == [f"K-HEADS:{r}:{m}" for r in RUNS for m in R.MODES if (r, m) != ("REP_s1", "egoc")]
