"""Unit tests of the NAVRB2 read on SYNTHETIC data (no dumps, no model forward).  Run: python -m pytest -q -p no:cacheprovider test_navrb2_read.py
Mutations (mutate_navrb2.py) must turn named tests RED.  CUDA_VISIBLE_DEVICES=-1."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb2_lib as NL  # noqa: E402
import navrb2_read as R  # noqa: E402
import navrb2_dumps as DU  # noqa: E402
import navrb2_t0 as T0  # noqa: E402
import navrb2_view as VW  # noqa: E402
import navrb2_strategic as ST  # noqa: E402
import navrb2_ctx_nofog as NF  # noqa: E402

CI = R.L.load_ci(NL.TREE)
CELLS = R.CELLS


def cells_of(v):
    return {c: np.asarray(v, np.float64).copy() for c in CELLS}


def eps(n_ep, per):
    return np.repeat(np.arange(n_ep), per)


# ------------------------------------------------------------------------------------------------------ strata
def test_window_state_now_and_history_rule():
    x = np.zeros(20, np.float32)
    x[5] = 1.0                                            # clip 0 spans rows 0..9; clip 1 rows 10..19
    off = np.array([0, 10])
    now, anyh = NL.window_state(x, off, np.array([0, 0, 0, 1]), np.array([5, 7, 2, 3]), hmax=4)
    assert now.tolist() == [True, False, False, False]
    assert anyh.tolist() == [True, True, False, False]    # t=7: rows 4..7 include row 5; t=2: rows 0..2 do not; clip 1 never reads clip 0's row 5 (clipped at the clip start)


def test_window_state_history_does_not_cross_clips():
    x = np.zeros(20, np.float32)
    x[9] = 1.0                                            # the last row of clip 0
    now, anyh = NL.window_state(x, np.array([0, 10]), np.array([1]), np.array([1]), hmax=4)     # row 11, rows 8..11 unclipped would include clip 0's row 9
    assert not anyh[0]


def test_card_counts_on_the_real_table():
    w = NL.rb_windows()
    st = NL.strata(w, 16)
    assert NL.count_stratum(st["rb"], w["eid"]) == {"n_windows": NL.RB_EXPECT_WINDOWS, "n_episodes": NL.RB_EXPECT_EPISODES}
    assert not (st["rb"] & st["zero_hist"]).any()          # a window with the state on at NOW is never a zero-history window


def test_rb_npz_pin_refuses_a_changed_file(monkeypatch, tmp_path):
    p = tmp_path / "t.npz"
    p.write_bytes(b"x")
    monkeypatch.setattr(NL, "RB_NPZ", str(p))
    with pytest.raises(RuntimeError):
        NL.rb_npz_checked()


# ------------------------------------------------------------------------------------------------------ Z-EVAL / Z-USE
def test_z_eval_identical_passes_and_names_the_n():
    a = cells_of(np.arange(10.0))
    r = R.z_eval(a, {c: v.copy() for c, v in a.items()}, np.ones(10, bool))
    assert r["pass"] and r["n_windows"] == 10


def test_z_eval_one_window_in_mask_fails():
    a = cells_of(np.arange(10.0))
    b = {c: v.copy() for c, v in a.items()}
    b[CELLS[3]][4] += 1e-9
    assert not R.z_eval(a, b, np.ones(10, bool))["pass"]


def test_z_eval_difference_outside_mask_is_not_read():
    a = cells_of(np.arange(10.0))
    b = {c: v.copy() for c, v in a.items()}
    b[CELLS[3]][4] += 1.0
    m = np.ones(10, bool)
    m[4] = False
    assert R.z_eval(a, b, m)["pass"]


def test_z_eval_nan_mismatch_fails_and_nan_equal_passes():
    a = cells_of(np.arange(10.0))
    a[CELLS[0]][2] = np.nan
    b = {c: v.copy() for c, v in a.items()}
    assert R.z_eval(a, b, np.ones(10, bool))["pass"]
    b[CELLS[0]][2] = 1.0
    assert not R.z_eval(a, b, np.ones(10, bool))["pass"]


def test_z_eval_empty_mask_is_not_a_pass():
    a = cells_of(np.arange(10.0))
    assert not R.z_eval(a, a, np.zeros(10, bool))["pass"]


def test_z_use_identical_not_separated_and_shifted_separated():
    rng = np.random.default_rng(0)
    base = rng.normal(size=400)
    eid = eps(20, 20)
    rb = np.ones(400, bool)
    real, zero = cells_of(base), cells_of(base)
    assert not R.z_use(CI, real, zero, eid, rb, 200)["separated"]
    real[CELLS[2]] = base + 5.0
    zu = R.z_use(CI, real, zero, eid, rb, 200)
    assert zu["separated"] and CELLS[2] in zu["separated_cells"]


def test_z_use_reads_only_the_rb_windows():
    base = np.random.default_rng(1).normal(size=400)
    eid = eps(20, 20)
    real, zero = cells_of(base), cells_of(base)
    real[CELLS[2]] = base + 5.0
    rb = np.zeros(400, bool)
    rb[100:140] = True
    real[CELLS[2]][:100] = base[:100]                      # the shift lives only on the RB windows
    assert R.z_use(CI, real, zero, eid, rb, 200)["separated"]
    real2 = cells_of(base)
    real2[CELLS[2]][300:] += 5.0                           # a shift OUTSIDE the RB windows: not read
    assert not R.z_use(CI, real2, zero, eid, rb, 200)["separated"]


# ------------------------------------------------------------------------------------------------------ D1 / D2 / floors
def _acc_cells(vals):
    c = cells_of(np.zeros(len(vals)))
    c[R.DECIDING] = np.asarray(vals, np.float64)
    return c


def test_d1_gain_needs_separation_and_the_floor():
    eid = eps(16, 15)
    n = len(eid)
    ref = _acc_cells(np.zeros(n))
    arm = _acc_cells(np.ones(n))                           # +1.0 everywhere
    mask = np.ones(n, bool)
    r = R.d1_row(CI, arm, ref, eid, mask, 0.05, 300)
    assert r["gain"] and not r["cost"] and r["above_floor"]
    r2 = R.d1_row(CI, arm, ref, eid, mask, 1.5, 300)       # the floor above |delta|: neither a gain nor a cost
    assert not r2["gain"] and not r2["cost"]
    r3 = R.d1_row(CI, ref, arm, eid, mask, 0.05, 300)      # swapped: a cost
    assert r3["cost"] and not r3["gain"]


def test_d1_reads_only_the_stratum():
    eid = eps(16, 15)
    n = len(eid)
    arm, ref = _acc_cells(np.zeros(n)), _acc_cells(np.zeros(n))
    mask = np.zeros(n, bool)
    mask[:60] = True
    arm[R.DECIDING][100:] = 1.0                            # outside the stratum
    assert R.d1_row(CI, arm, ref, eid, mask, 0.0, 200)["delta"] == 0.0


def test_d2_turn_recall_and_false_turn_share():
    n = 40
    eid = eps(8, 5)
    rb = np.zeros(n, bool)
    rb[:20] = True
    turn = np.zeros(n, bool)
    turn[:8] = True
    arm = _acc_cells(np.r_[np.ones(8), np.zeros(32)])      # recall 1.0 on the turn windows
    ref = _acc_cells(np.zeros(n))
    arm_pred = np.r_[np.zeros(8), np.ones(4), np.zeros(28)]  # 4 of the 12 non-turn RB windows predicted a turn
    ref_pred = np.zeros(n)
    d = R.d2_rows(CI, arm, arm_pred, ref, ref_pred, eid, turn, rb, 100)
    assert d["turn"]["n_windows"] == 8 and d["non_turn_rb"]["n_windows"] == 12
    assert d["turn_recall"]["arm"] == 1.0 and d["turn_recall"]["ref"] == 0.0
    assert abs(d["false_turn_share"]["arm"] - 4 / 12) < 1e-12 and d["false_turn_share"]["ref"] == 0.0


def test_floor2_and_rb_floors():
    F = {c: 0.01 for c in CELLS}
    F[CELLS[1]] = None
    Fr = {c: 0.03 for c in CELLS}
    f2 = R.floor2(F, Fr)
    assert f2[CELLS[0]] == 0.03 and f2[CELLS[1]] is None
    assert R.floor2(F, None)[CELLS[0]] == 0.01
    n = 10
    mk = lambda v: _acc_cells(np.full(n, v))  # noqa: E731
    cells = {"C0_rep": mk(0.5), "C0_s0": mk(0.4), "C0_s10r": mk(0.9), "C0_s10": mk(0.6), "C0_s11r": mk(0.3), "C0_s11": mk(0.35)}
    f, frep, f2b = R.rb_floors(cells, np.ones(n, bool))
    assert abs(f - 0.1) < 1e-12 and abs(frep - 0.3) < 1e-12 and abs(f2b - 0.3) < 1e-12


# ------------------------------------------------------------------------------------------------------ outcome labels (card section 7)
def row(gain=False, cost=False):
    return {"status": "OK", "gain": gain, "cost": cost}


def pl(pair, gains=(True, True), costs=(False, False), guard=(set(), set()), zuse=(True, True), route=False):
    return R.pair_labels(pair, {0: row(gains[0], costs[0]), 1: row(gains[1], costs[1])}, {0: set(guard[0]), 1: set(guard[1])}, {0: zuse[0], 1: zuse[1]}, route)


def test_label_rb_closes():
    assert pl("S1R-S1")[0] == ["RB-CLOSES"]


def test_label_closes_needs_the_guard_zuse_and_the_s1_pair():
    assert "RB-CLOSES" not in pl("S1R-S1", guard=({"ade_m"}, set()))[0]
    assert "RB-CLOSES" not in pl("S1R-S1", zuse=(True, False))[0]
    assert "RB-CLOSES" not in pl("S1R-S1", route=True)[0]
    assert "RB-CLOSES" not in pl("N1R-N1")[0]
    assert "RB-CLOSES" not in pl("S1R-S1", gains=(True, False))[0]


def test_label_seeds_disagree_on_a_one_seed_gain_or_cost():
    assert "RB-SEEDS-DISAGREE" in pl("S1R-S1", gains=(True, False))[0]
    assert "RB-SEEDS-DISAGREE" in pl("S1R-S1", gains=(False, False), guard=({CELLS[0]}, set()))[0]
    assert "RB-SEEDS-DISAGREE" not in pl("S1R-S1")[0]


def test_label_read_no_gain_and_unread():
    assert "RB-READ-NO-GAIN" in pl("S1R-S1", gains=(False, False))[0]
    assert "RB-UNREAD" in pl("S1R-S1", gains=(False, False), zuse=(False, False))[0]
    assert "RB-READ-NO-GAIN" not in pl("S1R-S1", gains=(False, False), zuse=(False, False))[0]
    assert "RB-UNREAD" not in pl("S1R-S1", gains=(False, False), zuse=(True, False))[0]     # reading r3: needs BOTH seeds unread


def test_label_costs_d1_both_seeds_or_a_guard_cell_at_both_seeds():
    assert "RB-COSTS" in pl("S1R-S1", gains=(False, False), costs=(True, True))[0]
    assert "RB-COSTS" in pl("N1R-N1", gains=(False, False), guard=({CELLS[2]}, {CELLS[2]}))[0]
    assert "RB-COSTS" not in pl("N1R-N1", gains=(False, False), costs=(True, False))[0]
    assert "RB-COSTS" not in pl("N1R-N1", gains=(False, False), guard=({CELLS[2]}, {CELLS[3]}))[0]


def test_label_not_yet_on_a_missing_seed():
    assert R.pair_labels("S1R-S1", {0: row(True)}, {0: set()}, {0: True})[0] == ["NOT-YET"]


def test_cross_pair_nav_only():
    n = pl("N1R-N1")
    s = pl("S1R-S1", gains=(False, False))
    assert R.outcomes({"N1R-N1": n, "S1R-S1": s})["cross_pair"] == ["RB-NAV-ONLY"]
    assert R.outcomes({"N1R-N1": n, "S1R-S1": pl("S1R-S1")})["cross_pair"] == []
    assert R.outcomes({"N1R-N1": pl("N1R-N1", gains=(True, False)), "S1R-S1": s})["cross_pair"] == []


def test_every_label_has_a_next_lever():
    for k in ("RB-CLOSES", "RB-NAV-ONLY", "RB-SEEDS-DISAGREE", "RB-READ-NO-GAIN", "RB-UNREAD", "RB-COSTS"):
        assert R.NEXT[k]


# ------------------------------------------------------------------------------------------------------ K-RE
def test_k_re_equal_changed_and_vacuous():
    row_ = {"delta": -0.0158, "lo": -0.0329, "hi": 0.0012, "separated": False}
    banked = {"families": {"TACTICAL": {"cells": {"TAC-LAT": {"N1-C0_s0": dict(row_)}}}}}
    ok, bad, n = R.k_re({"N1-C0_s0": {"TAC-LAT": dict(row_)}}, banked)
    assert ok and n == 1 and not bad
    ok, bad, n = R.k_re({"N1-C0_s0": {"TAC-LAT": dict(row_, delta=-0.0159)}}, banked)
    assert not ok and bad
    ok, bad, n = R.k_re({"S1-C0_s0": {"TAC-LAT": dict(row_)}}, banked)
    assert not ok and n == 0                                # a vacuous compare is a failure


# ------------------------------------------------------------------------------------------------------ refusals, argv, view
def _fake_bank(tmp_path, tag="N1R_s0", commit=None, tag_in_manifest=None, sha_ok=True):
    d = tmp_path / tag
    d.mkdir(parents=True)
    (d / "ckpt_last.pt").write_bytes(b"ckpt-bytes")
    sha = NL.sha256_file(str(d / "ckpt_last.pt"))
    json.dump({"tag": tag_in_manifest or tag, "commit": commit or NL.COMMIT, "files": {"ckpt_last.pt": {"sha256": sha if sha_ok else "0" * 64}}}, open(d / "BANK_MANIFEST.json", "w"))
    return str(tmp_path)


def test_refusal_accepts_a_good_bank_and_refuses_each_defect(tmp_path):
    assert NL.refusal("N1R_s0", _fake_bank(tmp_path / "a")) is None
    assert "commit" in NL.refusal("N1R_s0", _fake_bank(tmp_path / "b", commit="deadbeef"))
    assert "names tag" in NL.refusal("N1R_s0", _fake_bank(tmp_path / "c", tag_in_manifest="N1R_s1"))
    assert "sha256" in NL.refusal("N1R_s0", _fake_bank(tmp_path / "d", sha_ok=False))
    assert "missing" in NL.refusal("N1R_s0", str(tmp_path / "nowhere"))
    assert NL.refusal("N1_s0", str(tmp_path)) is not None           # a banked arm is not an R arm


def test_refusal_of_the_zero_twin_checks_the_same_bank(tmp_path):
    root = _fake_bank(tmp_path)
    assert NL.refusal("N1R_s0" + NL.Z, root) is None


def test_t1_argv_follow_and_zero_twin():
    a = DU.extra_args("NAVRB2-N1R_s0")
    z = DU.extra_args("NAVRB2-N1R_s0Z")
    for x in (a, z):
        assert x[x.index("--nav-rb") + 1] == "on" and x[x.index("--nav-rb-sha256") + 1] == NL.RB_SHA256 and "--dump-nav" in x and "--nav-text-npz" in x
    assert a[a.index("--nav-rb-variant") + 1] == "follow" and z[z.index("--nav-rb-variant") + 1] == "zero"
    assert DU.variant_of("NAVRB2-S1R_s1Z") == "zero" and DU.variant_of("NAVRB2-S1R_s1") == "follow"


def test_t1_shim_swaps_the_script_and_pins_the_tree():
    got = {}

    def fake_run(argv, **kw):
        got["argv"], got["env"] = argv, kw["env"]

    old = DU.subprocess
    DU.subprocess = type("S", (), {"run": staticmethod(fake_run), "PIPE": 0, "STDOUT": 0})
    DU._Shim.cur = "NAVRB2-N1R_s0Z"
    DU.OUT = os.path.join(os.environ.get("TMP", "."), "navrb2_shim_test")
    try:
        DU._Shim.run(["py", NL.TREE + "/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", "x"], env={"PYTHONPATH": "bad"})
    finally:
        DU.subprocess = old
    assert got["argv"][1].endswith("v8f_t1_eval_rb.py") and "--nav-rb-variant" in got["argv"]
    assert NL.TREE.replace("/", "\\").lower().replace("\\", "/") in got["env"]["PYTHONPATH"].replace("\\", "/").lower()


def test_t0_variants_donor_holds_channel_zero_and_zero_is_withheld(monkeypatch):
    monkeypatch.setattr(T0.L, "pinned", lambda n, root=None, check=True: "/x/" + n)
    real, donor, zero, k4 = (T0.argv_for("S1R_s0", v, "/o") for v in ("real", "donor", "zero", "k4"))
    assert real[1].endswith("v8f_mini_eval_rb.py") and real[real.index("--nav-rb-variant") + 1] == "follow" and "--nav-shuffle" not in real
    assert "--nav-shuffle" in donor and donor[donor.index("--nav-rb-variant") + 1] == "zero"
    assert "--nav-withheld" in zero and "--nav-rb-variant" not in zero
    assert k4[1].endswith("navrb2_k4_eval.py")
    assert all(x[x.index("--nav-rb") + 1] == "on" and x[x.index("--ages") + 1] == "fresh" for x in (real, donor, zero, k4))


def test_view_plan_names_and_refuses_missing(tmp_path):
    p = VW.plan()
    names = [d for _, d in p]
    assert "t0/eval_STRAT1-S1_s0_k4.npz" in names and "t0/eval_STRAT1-N1_s1_donor.npz" in names and "t1/ticks/ticks_STRAT1-C0_rep_m10_causal_heads.npz" in names
    assert not any("N1_s0_k4" in n for n in names)                  # N1 has no K4
    assert all(("NAVRB2-" in s) for s, d in p if "S1_s" in d or "N1_s" in d)   # S1 / N1 names in the view come from R arms only
    with pytest.raises(FileNotFoundError):
        VW.build(str(tmp_path / "view"), t0=str(tmp_path / "none"))


# ------------------------------------------------------------------------------------------------------ STRATEGIC contrast + nofog
def test_route_cost_cells_needs_both_seeds():
    c = ST.SS.RC_CELLS[0]
    rows = lambda cost: {k: {"cost": (k == c and cost)} for k in ST.SS.RC_CELLS}  # noqa: E731
    assert ST.route_cost_cells({0: rows(True), 1: rows(True)}) == [c]
    assert ST.route_cost_cells({0: rows(True), 1: rows(False)}) == []
    assert ST.route_cost_cells({0: rows(True)}) == []


def test_nofog_masks_labels_and_logits_and_baselines():
    tab = {}
    for sp in ("train", "eval"):
        tab[f"{sp}_weather_id"] = np.array([0, 1, 2, 3, 1, 0])
        tab[f"{sp}_weather_valid"] = np.ones(6, bool)
    m = NF.mask_table(tab)
    assert m["eval_weather_valid"].tolist() == [True, False, True, True, False, True] and tab["eval_weather_valid"].all()   # the input table is not edited
    saved = (NF.C.label_rows, NF.C.split_logits, NF.C.table_baselines)
    try:
        NF.patch()
        lg = np.zeros((3, sum(NF.C.CTX_N)))
        parts = NF.C.split_logits(lg)
        assert np.isneginf(parts[NF.W][:, NF.FOG_ID]).all() and np.isfinite(parts[NF.W][:, 0]).all()
        t2 = dict(tab, eval_flag=np.zeros(6, np.uint8), eval_off=np.array([0]), eval_n=np.array([6]))
        for v in NF.C.CTX_VARS:
            t2.setdefault(f"eval_{v}_id", np.zeros(6, np.int64))
            t2.setdefault(f"eval_{v}_valid", np.ones(6, bool))
        y = NF.C.label_rows(t2, np.arange(6))
        assert y[:, NF.W].tolist() == [0, -1, 2, 3, -1, 0]
    finally:
        NF.C.label_rows, NF.C.split_logits, NF.C.table_baselines = saved


def test_reading_choices_are_written_and_card_sha_is_the_frozen_one():
    assert set(NL.READING) == {"r1_Z-USE_separated", "r2_guard", "r3_not_separated_both", "r4_T0_donor_channel", "r5_zeval_mask"}
    sha_file = "F:/TanitAD-scratch/strat1/navrb2/PREREG_V8F_NAVRB2.sha256"
    assert open(sha_file, encoding="utf-8").read().split()[0] == NL.CARD_SHA256
    assert NL.sha256_file("F:/TanitAD-scratch/strat1/navrb2/PREREG_V8F_NAVRB2.md") == NL.CARD_SHA256


def test_tree_pins_pinned_evaluators_byte_equal():
    assert NL.tree_problem() is None
