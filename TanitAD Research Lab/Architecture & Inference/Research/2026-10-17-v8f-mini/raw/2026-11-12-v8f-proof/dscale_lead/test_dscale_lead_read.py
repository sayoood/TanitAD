"""Tests of dscale_lead_read.py and pod_dscale_lead_run.sh (card PREREG_V8F_PROOF_ADD45 with note N1). numpy only, synthetic head outputs, tiny worlds (no torch, no real data). Evidence class: MEASURED (synthetic).
NOT RUN by the builder; the lead runs it on the pod mirror (where the pilot reader chain imports, as for finegrid_r).

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider test_dscale_lead_read.py --basetemp <NEW folder two levels deep>
The reader under test is dscale_lead_read.py next to this file, or the file named by DSCALE_LEAD_READ_FILE (run_mutations_u.py points it at a mutated copy; set FINEGRID_BASE_DIR too). The kit script is
pod_dscale_lead_run.sh next to this file, or DSCALE_POD_SCRIPT. Every test holds at least one arm that must FAIL (a planted wrong value) through `must_fail`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
RFILE = Path(os.environ.get("DSCALE_LEAD_READ_FILE", str(HERE / "dscale_lead_read.py")))
spec = importlib.util.spec_from_file_location("dscale_lead_read_under_test", str(RFILE))
R = importlib.util.module_from_spec(spec)
sys.modules["dscale_lead_read_under_test"] = R
spec.loader.exec_module(R)
FR, N, RP, O, S0F = R.FR, R.N, R.RP, R.O, R.S0F
SCRIPT = Path(os.environ.get("DSCALE_POD_SCRIPT", str(HERE / "pod_dscale_lead_run.sh")))


def must_fail(fn):
    """The planted-wrong arm: ``fn`` states something false and must raise AssertionError."""
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError("a planted wrong statement was accepted: the check is blind")


def _assert(x):
    assert x


def make_ctx(n_ep=40, per=10, seed=0, B=400):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), per)
    n = len(ep)
    G15 = np.full((n, 15), np.nan)
    G15[:, S0F] = rng.uniform(10, 60, n)
    return {"valid": np.ones(n, bool), "c_lab": rng.normal(0, 2.0, n), "G15": G15, "ep10": ep, "n10": n_ep, "C10": RP.boot_counts(n_ep, B, O.SEED)}


@pytest.fixture(scope="module")
def ctx():
    return make_ctx()


def scenario(ctx, c_sd, u_sd, base=100):
    """P16C and P16U arms (four seed indices) with independent gap noise of sd c_sd and u_sd, plus P16U2_s0 with the sd of U."""
    y = ctx["G15"][:, S0F]
    n = len(y)
    outs = {}
    for kind, sd in (("C", c_sd), ("U", u_sd)):
        for k in R.SEEDS:
            r = np.random.default_rng(base + 1000 * (kind == "U") + k)
            outs[R.lab(kind, k)] = {"gap_m": y + r.normal(0, sd, n)}
    outs["P16U2_s0"] = {"gap_m": y + np.random.default_rng(base + 77).normal(0, u_sd, n)}
    return outs


# ------------------------------------------------------------------------------------------------ the arm map
def test_arm_map_is_the_card_map():
    assert R.ARM_MAP["P16C_s2"][:2] == ("runC", "TK16_s2") and R.ARM_MAP["P16U_s3"][:2] == ("runU", "TK16_s3") and R.ARM_MAP["P16U2_s0"][:2] == ("runU2", "TK16_s0")
    assert R.ARM_MAP["P16CSH_s0"][:2] == ("runCSH", "TK16_s0") and R.ARM_MAP["P16USH_s0"][:2] == ("runUSH", "TK16_s0")
    assert R.ARM_MAP["P16U2_s0"][2] == 38328 and R.ARM_MAP["P16U_s0"][2] == 19164 and R.ARM_MAP["P16CSH_s0"][3] is True and R.ARM_MAP["P16U_s0"][3] is False
    assert R.ARM_MAP["P16C_s0"][4] is True and R.ARM_MAP["P16U_s0"][4] is False and len(R.ARM_MAP) == 11
    must_fail(lambda: _assert(R.ARM_MAP["P16C_s0"][0] == "runU"))


# ------------------------------------------------------------------------------------------------ the two-level estimator
def test_boot_diff_is_the_pilot_readers_paired_row_for_m2_and_the_gap_error(ctx):
    o = scenario(ctx, 5.0, 2.0)
    a, b = o["P16U_s1"], o["P16C_s1"]
    for measure, theirs in (("M2", FR.pair_r2(ctx, ctx["G15"][:, S0F], a["gap_m"], b["gap_m"])), ("MAE", FR.pair_mae(ctx, a["gap_m"], b["gap_m"]))):
        pt, vec = R.boot_diff(ctx, measure, a, b)
        assert pt == theirs["diff"] and RP.ci(vec) == theirs["ci95"], measure
    pt, vec = R.boot_diff(ctx, "M2", a, b)
    must_fail(lambda: _assert(pt == FR.pair_r2(ctx, ctx["G15"][:, S0F], b["gap_m"], a["gap_m"])["diff"]))


def test_the_seed_level_is_resampled_the_interval_is_wider_than_episodes_alone(ctx):
    r = np.random.default_rng(1)
    pts = np.array([0.02, 0.10, 0.18, 0.06])
    D = pts[:, None] + r.normal(0, 0.003, size=(4, 400))
    tl = R.two_level(pts, D)
    ep_only = RP.ci(D.mean(0))
    assert tl["ci95"][1] - tl["ci95"][0] > 3 * (ep_only[1] - ep_only[0])
    assert tl["mean"] == pts.mean() and tl["n_pos"] == 4 and tl["per_seed"] == [float(x) for x in pts] and tl["n_seeds"] == 4
    assert R.two_level(pts, D) == tl                                                  # rng 0: the same call gives the same interval
    must_fail(lambda: _assert(tl["ci95"][1] - tl["ci95"][0] <= 3 * (ep_only[1] - ep_only[0])))


def test_control_d7_planted_gain_every_seed_one_of_four_not_identical_exactly_zero_swap_negative(ctx):
    c = R.control_d7(ctx)
    assert c["pass"], c
    assert c["gain_at_every_seed"]["ci95"][0] > 0 and c["gain_at_every_seed"]["n_pos"] == 4
    assert c["gain_at_one_of_four"]["n_pos"] == 1 and not R._above(c["gain_at_one_of_four"])
    assert c["identical_pairs"]["mean"] == 0.0 and c["identical_pairs"]["ci95"] == [0.0, 0.0]
    assert c["swapped_order"]["mean"] < 0 and c["swapped_order"]["ci95"][1] < 0 and abs(c["swapped_order"]["mean"] + c["gain_at_every_seed"]["mean"]) < 1e-9
    must_fail(lambda: _assert(c["gain_at_one_of_four"]["ci95"][0] > 0 and c["gain_at_one_of_four"]["n_pos"] >= 3))
    must_fail(lambda: _assert(c["swapped_order"]["mean"] > 0))


# ------------------------------------------------------------------------------------------------ names (card section 4 with note N1)
def test_a_planted_better_union_arm_reads_scales_gap(ctx):
    oc = R.compute_outcomes_u(ctx, scenario(ctx, 5.0, 1.5))
    n = oc["names"]
    assert n["name"] == "LEAD-READ-SCALES-GAP" and n["mae_clause_met"] and "cannot be given" in n["closing_name"], n
    assert oc["two_level"]["M2"]["n_pos"] == 4 and oc["two_level"]["MAE"]["mean"] < 0
    assert set(oc["beside"]["std_over_seeds_of_M2"]) == {"C", "U"} and "P16U2_s0" in oc["beside"]
    must_fail(lambda: _assert(n["name"] == "LEAD-READ-FLAT-AT-SCALE"))


def test_equal_arms_read_flat_with_every_per_seed_difference_zero(ctx):
    one = scenario(ctx, 4.0, 4.0)["P16C_s0"]
    outs = {R.lab(t, k): dict(one) for t in ("C", "U") for k in R.SEEDS}
    outs["P16U2_s0"] = dict(one)
    oc = R.compute_outcomes_u(ctx, outs)
    assert oc["names"]["name"] == "LEAD-READ-FLAT-AT-SCALE"
    assert oc["two_level"]["M2"]["mean"] == 0.0 and oc["two_level"]["M2"]["ci95"] == [0.0, 0.0] and oc["two_level"]["M2"]["per_seed"] == [0.0] * 4
    assert R.compute_outcomes_u(ctx, scenario(ctx, 4.0, 4.0))["names"]["name"] == "LEAD-READ-FLAT-AT-SCALE"
    must_fail(lambda: _assert(oc["names"]["name"] == "LEAD-READ-SCALES-GAP"))


def test_a_planted_worse_union_arm_reads_worse_at_scale(ctx):
    oc = R.compute_outcomes_u(ctx, scenario(ctx, 4.0, 9.0))
    assert oc["names"]["name"] == "LEAD-READ-WORSE-AT-SCALE" and oc["two_level"]["M2"]["ci95"][1] < 0
    must_fail(lambda: _assert(oc["names"]["name"] == "LEAD-READ-SCALES-GAP"))


def _tl(mean=0.05, ci=(0.01, 0.09), n_pos=4):
    return {"mean": mean, "ci95": list(ci), "n_pos": n_pos, "per_seed": [mean] * 4, "n_seeds": 4}


def test_the_error_clause_the_three_of_four_clause_and_the_touching_interval_each_withhold_the_gap_name():
    assert R.decide_u(_tl(), 10.0, 8.9)["name"] == "LEAD-READ-SCALES-GAP"
    assert R.decide_u(_tl(), 10.0, 9.2)["name"] == "LEAD-READ-FLAT-AT-SCALE"             # the error clause: 9.2 > 0.9 x 10
    assert R.decide_u(_tl(n_pos=3), 10.0, 8.9)["name"] == "LEAD-READ-SCALES-GAP"
    assert R.decide_u(_tl(n_pos=2), 10.0, 8.9)["name"] == "LEAD-READ-FLAT-AT-SCALE"
    assert R.decide_u(_tl(ci=(0.0, 0.09)), 10.0, 8.9)["name"] == "LEAD-READ-FLAT-AT-SCALE"  # an interval that touches 0 does not separate
    assert R.decide_u(_tl(mean=-0.05, ci=(-0.09, -0.01), n_pos=0), 10.0, 12.0)["name"] == "LEAD-READ-WORSE-AT-SCALE"
    assert R.decide_u(_tl(mean=-0.05, ci=(-0.09, 0.0), n_pos=0), 10.0, 12.0)["name"] == "LEAD-READ-FLAT-AT-SCALE"
    d = R.decide_u(_tl(), 10.0, 8.9)
    assert d["closing_name"].startswith("cannot be given") and abs(d["mae_ratio_U_to_C"] - 0.89) < 1e-12
    must_fail(lambda: _assert(R.decide_u(_tl(n_pos=2), 10.0, 8.9)["name"] == "LEAD-READ-SCALES-GAP"))


def test_the_pairing_is_the_same_seed_index_and_union_minus_corpus(ctx):
    o = scenario(ctx, 5.0, 2.0)
    t = R.two_level_pair(ctx, o, "M2")
    for i, k in enumerate(R.SEEDS):
        want, _ = R.boot_diff(ctx, "M2", o[R.lab("U", k)], o[R.lab("C", k)])
        assert t["per_seed"][i] == want
    assert t["mean"] > 0                                                                # U is the better arm in this scenario
    mis = [R.boot_diff(ctx, "M2", o[R.lab("U", k)], o[R.lab("C", (k + 1) % 4)])[0] for k in R.SEEDS]
    must_fail(lambda: _assert(t["per_seed"] == mis))


# ------------------------------------------------------------------------------------------------ controls from the precheck
def good_precheck():
    grp = lambda: {"clips": 200, "clips_absent_from_join": 0, "frames": 1200, "presence_frames": 700, "gap_label_frames": 650, "clips_with_a_gap_label": 190, "pass": True}
    return {"pass": True, "checks": {"counts": {"pass": True}, "d1_no_leak": {"pass": True, "planted": False}, "d2_labels_one_clock": {"pass": True}, "d2j_join_matches_lead_file": {"pass": True},
                                     "d2n_join_matches_lead_file_new_clips": {"pass": True, "sampled": 200, "rows_mismatching": 0},
                                     "d2g_join_labels_per_group": {"pass": True, "corpus": grp(), "new": grp()}}}


def good_planted():
    return {"pass": False, "checks": {"d1_no_leak": {"pass": False, "planted": True}}}


def test_precheck_controls_all_pass_on_the_good_record():
    c = R.precheck_controls(good_precheck(), good_planted())
    assert all(v["pass"] for v in c.values()), c
    assert set(c) >= {"d1_no_leak_precheck", "d2_labels_one_clock_precheck", "d2j_join_labels_precheck", "d2n_new_clips_join_matches_lead_file", "d2g_new_clips_carry_gap_labels", "d1_planted_shared_id_fails"}


def test_precheck_controls_each_planted_defect_fails_its_control():
    def one(edit, key, planted=None):
        p = good_precheck()
        edit(p)
        return R.precheck_controls(p, good_planted() if planted is None else planted)[key]["pass"]
    assert not one(lambda p: p["checks"]["d1_no_leak"].update(pass_=0) or p["checks"]["d1_no_leak"].update({"pass": False}), "d1_no_leak_precheck")
    assert not one(lambda p: p["checks"]["d2_labels_one_clock"].update({"pass": False}), "d2_labels_one_clock_precheck")
    assert not one(lambda p: p["checks"]["d2j_join_matches_lead_file"].update({"pass": False}), "d2j_join_labels_precheck")
    assert not one(lambda p: p["checks"].pop("d2n_join_matches_lead_file_new_clips"), "d2n_new_clips_join_matches_lead_file")            # absent = fail, not skipped
    assert not one(lambda p: p["checks"]["d2n_join_matches_lead_file_new_clips"].update({"rows_mismatching": 3}), "d2n_new_clips_join_matches_lead_file")
    assert not one(lambda p: p["checks"]["d2n_join_matches_lead_file_new_clips"].update({"sampled": 150}), "d2n_new_clips_join_matches_lead_file")
    assert not one(lambda p: p["checks"]["d2g_join_labels_per_group"]["new"].update({"gap_label_frames": 0}), "d2g_new_clips_carry_gap_labels")
    assert not one(lambda p: p["checks"]["d2g_join_labels_per_group"]["new"].update({"presence_frames": 0}), "d2g_new_clips_carry_gap_labels")
    assert not one(lambda p: p["checks"].pop("d2g_join_labels_per_group"), "d2g_new_clips_carry_gap_labels")
    assert not one(lambda p: p.update({"pass": False}), "d2g_new_clips_carry_gap_labels")                                                   # the whole record must pass
    assert not one(lambda p: None, "d1_planted_shared_id_fails", planted={"pass": True, "checks": {"d1_no_leak": {"pass": True, "planted": True}}})      # a planted id that d1 did not catch
    assert not R.precheck_controls(None, None)["d1_no_leak_precheck"]["pass"] and not R.precheck_controls(None, None)["d1_planted_shared_id_fails"]["pass"]
    must_fail(lambda: _assert(one(lambda p: p["checks"]["d2g_join_labels_per_group"]["new"].update({"gap_label_frames": 0}), "d2g_new_clips_carry_gap_labels")))


def good_clock(plant=0):
    grp = lambda: {"pass": plant == 0, "argmin_shift": -plant, "ratio_shift0_to_nearest": 0.04 if plant == 0 else 22.0, "clips_read": 200, "mean_abs_diff_mps": {"0": 0.005, "1": 0.1, "-1": 0.1}}
    return {"pass": plant == 0, "plant_shift": plant, "n_clips": 200, "groups": {"corpus": grp(), "new": grp()}}


def test_clock_controls_pass_on_the_good_records_and_each_planted_defect_fails():
    c = R.clock_controls(good_clock(), good_clock(1))
    assert set(c) == {"d2c_lead_rows_on_the_token_clock", "d2c_planted_shift_fails"} and all(v["pass"] for v in c.values()), c

    def one(edit, key, planted=None):
        k = good_clock()
        edit(k)
        return R.clock_controls(k, good_clock(1) if planted is None else planted)[key]["pass"]
    a, b = "d2c_lead_rows_on_the_token_clock", "d2c_planted_shift_fails"
    assert not one(lambda k: k.update({"pass": False}), a)
    assert not one(lambda k: k["groups"]["new"].update({"argmin_shift": 1}), a)                 # the new clips one token frame off
    assert not one(lambda k: k["groups"]["new"].update({"pass": False}), a)
    assert not one(lambda k: k["groups"].pop("new"), a)                                           # a group that was not read
    assert not one(lambda k: k["groups"]["corpus"].update({"clips_read": 150}), a)
    assert not one(lambda k: k.update({"plant_shift": 1}), a)                                     # a planted record handed in as the real one
    assert not one(lambda k: None, b, planted=good_clock())                                       # the planted run that passed
    wrong = good_clock(1)
    wrong["groups"]["new"]["argmin_shift"] = 0                                                    # the planted shift not found where it was put
    assert not one(lambda k: None, b, planted=wrong)
    none = R.clock_controls(None, None)
    assert not none[a]["pass"] and not none[b]["pass"]                                            # absent = fail, not skipped
    fl = R.gated(R.FR.flatten(none))
    assert fl[a] is False and fl[b] is False                                                      # both gate
    must_fail(lambda: _assert(one(lambda k: k["groups"]["new"].update({"argmin_shift": 1}), a)))


def good_evalset(planted=False):
    return {"pass": not planted, "d2e_eval_set_is_the_study_set": {"pass": not planted, "n_here": 141, "n_study": 141, "same_ids_same_order": not planted, "planted_swap": planted},
            "d3t_tokens_same": {"gates": False}}


def test_evalset_controls_pass_on_the_good_records_and_each_planted_defect_fails():
    a, b = "d2e_eval_set_is_the_study_set", "d2e_planted_swap_fails"
    c = R.evalset_controls(good_evalset(), good_evalset(True))
    assert set(c) == {a, b} and c[a]["pass"] and c[b]["pass"], c

    def one(edit, key, planted=None):
        e = good_evalset()
        edit(e)
        return R.evalset_controls(e, good_evalset(True) if planted is None else planted)[key]["pass"]
    assert not one(lambda e: e[a].update({"same_ids_same_order": False}), a)             # another order would misalign every window
    assert not one(lambda e: e[a].update({"n_here": 140}), a)
    assert not one(lambda e: e.update({"pass": False}), a)
    assert not one(lambda e: e[a].update({"planted_swap": True}), a)                      # a planted record handed in as the real one
    assert not one(lambda e: None, b, planted=good_evalset())                             # the planted swap that passed
    none = R.evalset_controls(None, None)
    assert not none[a]["pass"] and not none[b]["pass"]                                    # absent = fail, not skipped
    fl = R.gated(R.FR.flatten(none))
    assert fl[a] is False and fl[b] is False                                              # both gate
    must_fail(lambda: _assert(one(lambda e: e[a].update({"same_ids_same_order": False}), a)))


def head(r2_ci, auroc):
    return {"read": True, "M2": {"head": {"r2": r2_ci[1], "r2_ci95": [r2_ci[0], r2_ci[2]]}}, "M4": {"auroc": auroc}}


def test_d3_corpus_arm_is_a_live_gap_probe_and_the_row97_range_is_beside_only():
    ok = R.d3_corpus_live(head((0.55, 0.70, 0.80), 0.90), "P16C_s0")
    assert ok["pass"] and ok["beside_inside_row97_range"] and ok["beside_row97_range"] == [0.513, 0.791]
    outside = R.d3_corpus_live(head((0.80, 0.85, 0.90), 0.90), "P16C_s1")                 # above the row 97 range: printed, gates nothing
    assert outside["pass"] and outside["beside_inside_row97_range"] is False
    assert not R.d3_corpus_live(head((-0.01, 0.30, 0.50), 0.90), "x")["pass"]              # interval reaches 0
    assert not R.d3_corpus_live(head((0.55, 0.70, 0.80), 0.84), "x")["pass"]               # presence AUROC under 0.85
    must_fail(lambda: _assert(R.d3_corpus_live(head((0.55, 0.70, 0.80), 0.84), "x")["pass"]))


def test_d5_permuted_gap_arm_must_read_no_gap_above_zero_and_keep_presence_live():
    assert R.d5_permuted_dead(head((-0.10, 0.01, 0.12), 0.90), "P16CSH_s0")["pass"]          # interval includes 0
    assert R.d5_permuted_dead(head((-0.30, -0.20, -0.10), 0.90), "P16USH_s0")["pass"]        # entirely below 0
    assert not R.d5_permuted_dead(head((0.05, 0.20, 0.30), 0.90), "x")["pass"]               # the permuted arm still reads a gap: the labels were not permuted
    assert not R.d5_permuted_dead(head((-0.10, 0.01, 0.12), 0.79), "x")["pass"]              # presence AUROC under 0.80: the arm is not live
    must_fail(lambda: _assert(R.d5_permuted_dead(head((0.05, 0.20, 0.30), 0.90), "x")["pass"]))


# ------------------------------------------------------------------------------------------------ train logs (d4), records, d6
def wlog(path, *, total=19164, sh=False, w_clos=0.0, kept=None, guard=True, pres=None, gap=None, n=60, done=True, ngap=10.0):
    down, up = np.linspace(1.0, 0.5, n), np.linspace(0.5, 1.0, n)
    pres = down if pres is None else pres
    gap = down if gap is None else gap
    hdr = {"header": True, "arm": "TK16_s0", "total": total, "w_clos": w_clos, "shuffle_gap": ({"n_shuffled": 9} if sh else None), "clip_list": ({"n_kept": kept, "n_listed": kept} if kept is not None else None),
           "eval_guard": ({"n_shared": 0, "n_eval_ids": 48} if guard else None)}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(hdr) + "\n")
        for i in range(n):
            last = (i == n - 1) and done
            fh.write(json.dumps({"step": (total if last else 50 * (i + 1)), "first": False, "lead_sup_mean": pres[i] + gap[i], "lead_sup_bce_mean": pres[i], "lead_sup_gap_mean": gap[i], "lead_sup_clos_mean": 0.0,
                                 "n_gap": ngap, "n_gap_min_interval": ngap}) + "\n")


def test_p5_u_header_levers_and_the_lead_term_fall(tmp_path):
    p = str(tmp_path / "t.jsonl")
    wlog(p, kept=4512)
    ok = R.p5_u(p, False, 19164, 4512)
    assert ok["pass"] and ok["last_mean"] < ok["first_mean"] and all(ok["header_checks"].values())
    wlog(p, kept=None)
    assert R.p5_u(p, False, 19164, None)["pass"]                                                        # a union arm: no clip list recorded
    for kw, exp in ((dict(total=19000, kept=4512), (19164, 4512)), (dict(w_clos=1.0, kept=4512), (19164, 4512)), (dict(sh=True, kept=4512), (19164, 4512)), (dict(guard=False, kept=4512), (19164, 4512)),
                    (dict(kept=4000), (19164, 4512)), (dict(kept=None), (19164, 4512)), (dict(kept=4512), (19164, None)), (dict(kept=4512, done=False), (19164, 4512)), (dict(kept=4512, ngap=0.0), (19164, 4512)),
                    (dict(kept=4512, gap=np.linspace(0.5, 1.0, 60), pres=np.linspace(0.5, 1.0, 60)), (19164, 4512))):
        wlog(p, **kw)
        assert not R.p5_u(p, False, exp[0], exp[1])["pass"], kw
    wlog(p, kept=4512)
    must_fail(lambda: _assert(not R.p5_u(p, False, 19164, 4512)["pass"]))


def test_p5_u_a_permuted_gap_arm_is_read_on_its_presence_part_only(tmp_path):
    p = str(tmp_path / "t.jsonl")
    down, up = np.linspace(1.0, 0.5, 60), np.linspace(0.5, 1.0, 60)
    wlog(p, sh=True, kept=4512, pres=down, gap=up)                                                      # presence falls, the permuted gap part rises
    d = R.p5_u(p, True, 19164, 4512)
    assert d["pass"] and d["form"].startswith("presence part only")
    wlog(p, sh=False, kept=4512, pres=down, gap=up)                                                     # the same numbers on an ordinary arm: presence + gap does not fall
    assert not R.p5_u(p, False, 19164, 4512)["pass"]
    wlog(p, sh=True, kept=4512, pres=up, gap=down)                                                      # presence rises: a permuted-gap arm that does not learn presence fails
    assert not R.p5_u(p, True, 19164, 4512)["pass"]
    wlog(p, sh=False, kept=4512, pres=down, gap=up)                                                     # the arm is expected to be permuted but the header says it is not
    assert not R.p5_u(p, True, 19164, 4512)["pass"]
    wlog(p, sh=True, kept=4512, pres=down, gap=up)
    must_fail(lambda: _assert(not R.p5_u(p, True, 19164, 4512)["pass"]))


def test_records_probe_form(tmp_path):
    def rec(**kw):
        base = {"grid": "pool4x10", "n_tok": 40, "n_tok_seen_in_training": 40, "W": 16, "D": 640, "input_norm": "ln", "step": 19164, "arm": "TK16_s0"}
        base.update(kw)
        (tmp_path / "TK16_s0_eval.npz.json").write_text(json.dumps(base), encoding="utf-8")
        return {"eval_states": str(tmp_path / "TK16_s0_eval.npz")}
    assert R.check_records_u(rec(), "P16C_s0", 19164, "TK16_s0")["pass"]
    assert R.check_records_u(rec(step=38328), "P16U2_s0", 38328, "TK16_s0")["pass"]
    for kw in (dict(step=19000), dict(grid="pool8x20"), dict(n_tok_seen_in_training=160), dict(W=64, D=2560), dict(input_norm="none"), dict(arm="TK64_s0")):
        assert not R.check_records_u(rec(**kw), "x", 19164, "TK16_s0")["pass"], kw
    assert not R.check_records_u({"eval_states": str(tmp_path / "none.npz")}, "x", 19164, "TK16_s0")["pass"]
    must_fail(lambda: _assert(R.check_records_u(rec(step=1), "x", 19164, "TK16_s0")["pass"]))


def test_d6_is_a_limit_not_a_pass_and_gates_nothing(tmp_path):
    for f, ch in (("runC", 48), ("runU", 48)):
        (tmp_path / f).mkdir()
        (tmp_path / f / "r1_batches.json").write_text(json.dumps({"sampler_chunk": ch}), encoding="utf-8")
    d = R.d6_limit(str(tmp_path))
    assert d["gates"] is False and d["statement"].startswith("LIMIT") and d["same_sampler_chunk"] is True and d["sampler_chunk"] == {"runC": 48, "runU": 48}
    (tmp_path / "runU" / "r1_batches.json").write_text(json.dumps({"sampler_chunk": 41}), encoding="utf-8")
    assert R.d6_limit(str(tmp_path))["same_sampler_chunk"] is False
    assert R.d6_limit(str(tmp_path / "none"))["same_sampler_chunk"] is False
    must_fail(lambda: _assert(R.d6_limit(str(tmp_path))["same_sampler_chunk"]))


def test_closing_controls_gate_nothing_every_other_control_gates():
    flat = {"k3_closing|P16C_s0": False, "k3_slot_n1|P16C_s0": False, "k4_no_future_input|P16C_s0": True, "d1_no_leak_precheck": True}
    assert R.gated(flat) == {"k4_no_future_input|P16C_s0": True, "d1_no_leak_precheck": True}
    assert "d3_corpus_arm_live|P16C_s0" in R.gated({"d3_corpus_arm_live|P16C_s0": False})
    must_fail(lambda: _assert("k3_closing|P16C_s0" in R.gated(flat)))


# ------------------------------------------------------------------------------------------------ the verdict text
def fake_res(withheld, ctx):
    ctrl = {"d1_no_leak_precheck": {"pass": True}, "d4_train_log|P16C_s0": {"pass": not withheld}}
    flat = FR.flatten(ctrl)
    res = {"tier": R.TIER_LINE, "families": list(R.FAMILY_LINES), "card": "x", "B": 2000, "evidence_class": "MEASURED once run", "controls": ctrl, "controls_flat": flat,
           "controls_failed": [k for k, v in flat.items() if not v], "controls_pass": not withheld, "limits": {"d6": {"statement": "LIMIT: x", "sampler_chunk": {"runC": 48, "runU": 48}, "same_sampler_chunk": True}},
           "arms": {"P16C_s0": {"M2": {"r2": 0.7}, "M4": {"auroc": 0.9}}}}
    if withheld:
        res.update({"names": None, "outcome_differences": "withheld: a control did not pass: d4_train_log|P16C_s0"})
    else:
        oc = R.compute_outcomes_u(ctx, scenario(ctx, 5.0, 1.5))
        res["arms"]["P16C_s0"]["gap_mae"] = oc["summary"]["P16C_s0"]["MAE"]
        res.update({"names": oc["names"], "two_level": oc["two_level"], "beside": oc["beside"]})
    return res


def test_verdict_text_tier_line_family_lines_controls_first_and_names_withheld_on_a_failed_control(tmp_path, ctx):
    exact = "TIER: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance"
    assert R.TIER_LINE == exact
    R.write_verdict(str(tmp_path / "v_ok.md"), fake_res(False, ctx))
    t = (tmp_path / "v_ok.md").read_text(encoding="utf-8")
    assert exact in t and "STRATEGIC: NOT COMPUTED (n = 0" in t and "LATERAL: not measured" in t and "TACTICAL: not measured" in t and "LONGITUDINAL (distance keeping): not measured" in t
    assert t.index("## Controls") < t.index("## Name") and "**LEAD-READ-SCALES-GAP**" in t and "cannot be given" in t
    R.write_verdict(str(tmp_path / "v_bad.md"), fake_res(True, ctx))
    b = (tmp_path / "v_bad.md").read_text(encoding="utf-8")
    assert exact in b and "No name is given and no outcome difference is printed" in b and "LEAD-READ-SCALES-GAP" not in b and "LEAD-READ-FLAT-AT-SCALE" not in b and "## Two-level reads" not in b
    must_fail(lambda: _assert("LEAD-READ-SCALES-GAP" in b))


def test_main_refuses_when_a_states_folder_is_missing(tmp_path):
    (tmp_path / "runC").mkdir()
    assert R.main(["--in-dir", str(tmp_path), "--run-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")]) == 3
    must_fail(lambda: _assert(R.main(["--in-dir", str(tmp_path), "--run-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")]) == 0))


# ------------------------------------------------------------------------------------------------ the kit script (DRY=1, bash only)
def dry(stage, **env):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash on PATH")
    e = dict(os.environ)
    e.update({"DRY": "1", "TREE": "/dummy/tree", "OUT": "/dummy/out", "P10": "/dummy/p10.pt", "TRACKS": "/dummy/tracks.pt"})
    e.update(env)
    r = subprocess.run([bash, str(SCRIPT).replace("\\", "/"), stage], env=e, capture_output=True, text=True, encoding="utf-8", timeout=120)
    return r.returncode, r.stdout


def test_kit_run_stage_dry_flags_per_folder_matched_steps_and_no_exclude():
    rc, out = dry("run")
    assert rc == 0
    cmds = {m.group(1): m.group(0) for m in re.finditer(r"--out /dummy/out/(run[A-Z0-9a-z]+) .*", out)}
    assert set(cmds) == {"runC", "runU", "runU2", "runCSH", "runUSH"}
    for f, c in cmds.items():
        assert "--w-clos 0.0" in c and "--eval-guard" in c and "--expect-steps" in c and "--exclude" not in c, f
        assert ("--steps 38328" in c and "--expect-steps 38328" in c) if f == "runU2" else ("--steps 19164" in c and "--expect-steps 19164" in c), f
        assert ("--clip-list" in c) == (f in ("runC", "runCSH")), f
        assert ("--shuffle-gap-labels" in c) == (f in ("runCSH", "runUSH")), f
        assert ("TK16_s0\\,TK16_s1\\,TK16_s2\\,TK16_s3" in c) == (f in ("runC", "runU")), f
    must_fail(lambda: _assert("--clip-list" in cmds["runU"]))


def test_kit_other_stages_dry_and_the_script_text_never_removes_anything():
    for stage in ("precheck", "smoke", "extract", "checklog"):
        rc, out = dry(stage)
        assert rc == 0 and "DRY:" in out, stage
    rc, out = dry("precheck")
    assert out.count("dscale_precheck.py") == 2 and "--plant" in out and "--join-union" in out
    rc, out = dry("extract")
    assert out.count("--splits eval") == 11 and "states/runUSH" in out
    assert re.search(r"(^|[;&|( ])rm( |$)", SCRIPT.read_text(encoding="utf-8"), re.M) is None
    rc, out = dry("run", OUT="SET-ME")
    assert rc == 3
    rc, out = dry("run", P10="")
    assert rc == 3 and "REFUSED" in out
    must_fail(lambda: _assert(re.search(r"(^|[;&|( ])rm( |$)", "x; rm -f y") is None))
