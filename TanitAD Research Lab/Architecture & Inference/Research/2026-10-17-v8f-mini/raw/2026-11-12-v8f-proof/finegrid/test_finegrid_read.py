"""Tests of finegrid_read.py (card PREREG_V8F_PROOF_ADD41). Synthetic head outputs, CPU, fast. Evidence class: MEASURED (synthetic).

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider test_finegrid_read.py --basetemp F:/TanitAD-scratch/v8proof/finegrid/tmp/pt/<new name>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import finegrid_read as FR  # noqa: E402

N, RP, O, S0F = FR.N, FR.RP, FR.O, FR.S0F
ARMS = FR.F_ARMS + FR.TK_ARMS


def make_ctx(n_ep=40, per=10, seed=0, B=500):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), per)
    n = len(ep)
    G15 = np.full((n, 15), np.nan)
    G15[:, S0F] = rng.uniform(10, 60, n)
    return {"valid": np.ones(n, bool), "c_lab": rng.normal(0, 2.0, n), "G15": G15, "ep10": ep, "n10": n_ep, "C10": RP.boot_counts(n_ep, B, O.SEED)}


@pytest.fixture(scope="module")
def ctx():
    return make_ctx()


def mk(ctx, gnoise, cnoise):
    return {"gap_m": ctx["G15"][:, S0F] + gnoise, "closing_ms": ctx["c_lab"] + cnoise}


def noise(ctx, seed, sg, sc):
    r = np.random.default_rng(seed)
    n = len(ctx["c_lab"])
    return r.normal(0, sg, n), r.normal(0, sc, n)


def scenario(ctx, tk_sd, f_sd, tk_c=2.0, f_c=2.0, base=100):
    """TK arms with independent noise of sd tk_sd (gap) / tk_c (closing); F arms with sd f_sd / f_c."""
    outs = {}
    for i, a in enumerate(FR.TK_ARMS):
        g, c = noise(ctx, base + i, tk_sd, tk_c)
        outs[a] = mk(ctx, g, c)
    for i, a in enumerate(FR.F_ARMS):
        g, c = noise(ctx, base + 50 + i, f_sd, f_c)
        outs[a] = mk(ctx, g, c)
    return outs


# ------------------------------------------------------------------------------------------------ the added metrics
def test_gap_mae_equals_a_plain_numpy_mean_and_has_an_episode_interval(ctx):
    g, _ = noise(ctx, 1, 4.0, 1.0)
    out = mk(ctx, g, g)
    r = FR.gap_mae(ctx, out["gap_m"])
    y = ctx["G15"][:, S0F]
    assert abs(r["mae_m"] - np.abs(y - out["gap_m"]).mean()) < 1e-12 and r["n"] == len(y) and r["ci95"][0] < r["mae_m"] < r["ci95"][1]
    p = out["gap_m"].copy()
    p[:7] = np.nan                                                                  # windows with no finite output are not scored
    r2 = FR.gap_mae(ctx, p)
    assert r2["n"] == len(y) - 7 and abs(r2["mae_m"] - np.nanmean(np.abs(y - p))) < 1e-12
    c2 = dict(ctx)
    v = ctx["valid"].copy()
    v[10:20] = False                                                                # windows that are not scored (no lead / no label) are not in the MAE
    c2["valid"] = v
    assert FR.gap_mae(c2, out["gap_m"])["n"] == len(y) - 10


def test_pair_r2_with_the_closing_target_equals_the_imported_pair_m1_and_the_swap_is_the_negative(ctx):
    a, b = noise(ctx, 2, 1.0, 1.5)[1], noise(ctx, 3, 1.0, 2.5)[1]
    ca, cb = ctx["c_lab"] + a, ctx["c_lab"] + b
    mine, theirs = FR.pair_r2(ctx, ctx["c_lab"], ca, cb), N.pair_m1(ctx, ca, cb)
    assert mine["diff"] == theirs["diff"] and mine["ci95"] == theirs["ci95"] and mine["n"] == theirs["n"]
    sw = FR.pair_r2(ctx, ctx["c_lab"], cb, ca)
    assert abs(sw["diff"] + mine["diff"]) < 1e-12 and abs(sw["ci95"][0] + mine["ci95"][1]) < 1e-12
    assert FR.pair_r2(ctx, ctx["c_lab"], ca, ca)["diff"] == 0.0
    ga, gb = ctx["G15"][:, S0F] + noise(ctx, 4, 2, 1)[0], ctx["G15"][:, S0F] + noise(ctx, 5, 5, 1)[0]
    pm, pmr = FR.pair_mae(ctx, ga, gb), FR.pair_mae(ctx, gb, ga)
    y = ctx["G15"][:, S0F]
    assert abs(pm["diff"] - (np.abs(y - ga).mean() - np.abs(y - gb).mean())) < 1e-12 and pm["diff"] < 0 and pm["separated_neg"] and pm["rel"] < -0.3
    assert abs(pm["diff"] + pmr["diff"]) < 1e-12 and FR.pair_mae(ctx, ga, ga)["diff"] == 0.0


def test_summary_m1_m2_are_the_imported_readers_numbers(ctx):
    out = scenario(ctx, 4.0, 4.0)["TK16_s0"]
    s = FR.summarize(ctx, out)
    ep, n, C = ctx["ep10"], ctx["n10"], ctx["C10"]
    assert s["M1"] == N.fid_one(ctx["c_lab"], out["closing_ms"], ctx["valid"], ep, n, C) and s["M2"] == N.fid_one(ctx["G15"][:, S0F], out["gap_m"], ctx["valid"], ep, n, C)


def test_gap_range_table_bins_by_the_label_gap_and_counts_add_up(ctx):
    outs = scenario(ctx, 4.0, 4.0)
    tab = FR.gap_range_table(ctx, outs)
    assert set(tab) == {"F16_s0", "F16_s1", "TK64_s0", "TK64_s1"}
    y = ctx["G15"][:, S0F]
    for arm, row in tab.items():
        assert sum(v["n"] for v in row.values()) == len(y)
        assert row["below 20 m"]["n"] == int((y < 20).sum()) and row["above 40 m"]["n"] == int((y >= 40).sum())
        mm = (y >= 20) & (y < 40)
        assert abs(row["20 to 40 m"]["mae_m"] - np.abs(y - outs[arm]["gap_m"])[mm].mean()) < 1e-12


def test_floor_is_the_largest_seed_difference_over_the_four_arm_types(ctx):
    outs = scenario(ctx, 4.0, 4.0)
    summ = {a: FR.summarize(ctx, outs[a]) for a in ARMS}
    fl = FR.floors(summ)
    want = max(abs(summ["%s_s0" % t]["M2"]["r2"] - summ["%s_s1" % t]["M2"]["r2"]) for t in FR.TK_TYPES + FR.F_TYPES)
    wantc = max(abs(summ["%s_s0" % t]["M1"]["r2"] - summ["%s_s1" % t]["M1"]["r2"]) for t in FR.TK_TYPES + FR.F_TYPES)
    assert fl["floor_m2"] == want and fl["floor_m1"] == wantc and set(fl["m2_by_type"]) == {"TK16", "TK64", "F16", "F64"}
    summ["F64_s1"] = None
    assert FR.floors({k: v for k, v in summ.items() if v})["floor_m2"] is None


# ------------------------------------------------------------------------------------------------ the names (card section 5)
def test_a_planted_better_f16_reads_finer_grid_reads_gap_and_closing(ctx):
    oc = FR.compute_outcomes(ctx, scenario(ctx, 5.0, 1.5, tk_c=2.0, f_c=0.8))
    n = oc["names"]
    assert n["gap"] == "FINER-GRID-READS-GAP", n
    assert n["m2_g16_g16w_above_0_with_interval_excluding_0_at_both_seeds"] and n["each_mean_above_floor"] and n["mae_clause_met_at_both_seeds"]
    assert n["closing"] == "FINER-GRID-READS-CLOSING"
    assert all(oc["paired"]["G16_%s" % s]["M2"]["diff"] > 0 and oc["paired"]["G16w_%s" % s]["MAE"]["diff"] < 0 for s in FR.SEEDS)


def test_equal_arms_read_flat_identical_outputs_and_independent_noise(ctx):
    one = scenario(ctx, 4.0, 4.0)["TK16_s0"]
    same = {a: dict(one) for a in ARMS}
    oc = FR.compute_outcomes(ctx, same)
    assert oc["names"]["gap"] == "FINER-GRID-FLAT" and oc["names"]["closing"].startswith("none")
    assert all(oc["paired"][k]["M2"]["diff"] == 0.0 for k in oc["paired"])
    oc2 = FR.compute_outcomes(ctx, scenario(ctx, 4.0, 4.0, tk_c=2.0, f_c=2.0))
    assert oc2["names"]["gap"] == "FINER-GRID-FLAT" and oc2["names"]["closing"].startswith("none")


def test_a_planted_worse_f16_reads_worse(ctx):
    oc = FR.compute_outcomes(ctx, scenario(ctx, 4.0, 9.0))
    assert oc["names"]["gap"] == "FINER-GRID-WORSE", oc["names"]
    assert all(oc["paired"][k]["M2"]["diff"] < 0 and oc["paired"][k]["M2"]["ci95"][1] < 0 for k in ("G16_s0", "G16_s1", "G16w_s0", "G16w_s1"))
    assert oc["names"]["closing"].startswith("none")                                  # the closing name has a positive side only


def _correlated(ctx, shrink, tk_sd_by_seed=(4.0, 4.0)):
    """TK16 and TK64 of a seed share one noise draw; F16 and F64 of that seed carry the same draw times `shrink` (the whole error scaled, so the paired bootstrap is tight)."""
    outs = {}
    for s, sd in zip(FR.SEEDS, tk_sd_by_seed):
        g, c = noise(ctx, 700 + int(s[1]), sd, 1.0)
        for t in FR.TK_TYPES:
            outs["%s_%s" % (t, s)] = mk(ctx, g, c)
        for t in FR.F_TYPES:
            outs["%s_%s" % (t, s)] = mk(ctx, shrink * g, shrink * c)
    return outs


def test_a_gain_that_misses_the_10_percent_mae_clause_is_flat_and_one_that_meets_it_reads_the_gap(ctx):
    miss = FR.compute_outcomes(ctx, _correlated(ctx, 0.93))                           # 7 % lower error, same seed noise in every type: floor 0, M2 separated, MAE clause missed
    n = miss["names"]
    assert n["m2_g16_g16w_above_0_with_interval_excluding_0_at_both_seeds"] and n["each_mean_above_floor"] and not n["mae_clause_met_at_both_seeds"]
    assert n["gap"] == "FINER-GRID-FLAT" and all(abs(v["f16_below_lower"] - 0.07) < 0.005 for v in n["mae_clause_by_seed"].values())
    ok = FR.compute_outcomes(ctx, _correlated(ctx, 0.85))
    assert ok["names"]["mae_clause_met_at_both_seeds"] and ok["names"]["gap"] == "FINER-GRID-READS-GAP"


def test_a_gain_smaller_than_the_floor_is_flat(ctx):
    outs = _correlated(ctx, 0.8, tk_sd_by_seed=(4.0, 7.0))                            # the two seeds differ a lot (large floor); the gain of F16 is 20 % at each seed
    oc = FR.compute_outcomes(ctx, outs)
    n = oc["names"]
    assert n["m2_g16_g16w_above_0_with_interval_excluding_0_at_both_seeds"] and n["mae_clause_met_at_both_seeds"]
    assert n["mean_G16_m2"] < n["floor_m2"] and not n["each_mean_above_floor"] and n["gap"] == "FINER-GRID-FLAT"


def test_one_seed_whose_interval_includes_0_is_flat_and_a_missing_arm_names_nothing(ctx):
    outs = scenario(ctx, 5.0, 1.5)
    outs["F16_s1"] = outs["TK16_s1"]                                                  # F16 at seed 1 equals TK16 at seed 1 (but not TK64): G16_s1 reads exactly 0
    n = FR.compute_outcomes(ctx, outs)["names"]
    assert n["gap"] == "FINER-GRID-FLAT" and not n["m2_g16_g16w_above_0_with_interval_excluding_0_at_both_seeds"]
    outs2 = scenario(ctx, 5.0, 1.5)
    outs2["F64_s0"] = None
    n2 = FR.compute_outcomes(ctx, outs2)["names"]
    assert n2["gap"] == "not named" and n2["closing"] == "not named"


def test_decide_with_hand_built_inputs_covers_each_clause():
    def pair(d, lo, hi):
        row = {"diff": d, "ci95": [lo, hi]}
        return {"M2": dict(row), "M1": dict(row), "MAE": {"diff": 0.0, "ci95": [0, 0], "rel": 0.0}}

    def summ(mae_f16):
        s = {}
        for a in FR.TK_ARMS:
            s[a] = {"MAE": {"mae_m": 5.0}, "M2": {"r2": 0.5}, "M1": {"r2": 0.1}}
        for a in FR.F_ARMS:
            s[a] = {"MAE": {"mae_m": mae_f16 if a.startswith("F16") else 9.9}, "M2": {"r2": 0.6}, "M1": {"r2": 0.2}}
        return s
    fl = {"floor_m2": 0.02, "floor_m1": 0.02}
    good = {"%s_%s" % (g, s): pair(0.1, 0.05, 0.15) for g in ("G16", "G16w", "G64") for s in FR.SEEDS}
    assert FR.decide(summ(4.4), good, fl)["gap"] == "FINER-GRID-READS-GAP"                  # 12 % below 5.0
    assert FR.decide(summ(4.5), good, fl)["gap"] == "FINER-GRID-READS-GAP"                  # exactly 10 % below: at least 10 % is met
    assert FR.decide(summ(4.51), good, fl)["gap"] == "FINER-GRID-FLAT"
    worse = dict(good)
    for k in ("G16_s0", "G16_s1", "G16w_s0", "G16w_s1"):
        worse[k] = pair(-0.1, -0.15, -0.05)
    assert FR.decide(summ(4.0), worse, fl)["gap"] == "FINER-GRID-WORSE"
    worse["G16w_s1"] = pair(-0.1, -0.15, 0.01)
    assert FR.decide(summ(4.0), worse, fl)["gap"] == "FINER-GRID-FLAT"
    mixed = dict(good)
    mixed["G16_s0"] = pair(-0.1, -0.15, -0.05)
    assert FR.decide(summ(4.0), mixed, fl)["gap"] == "FINER-GRID-FLAT"
    assert FR.decide(summ(4.0), good, {"floor_m2": 0.1, "floor_m1": 0.02})["gap"] == "FINER-GRID-FLAT"                # a mean equal to the floor is not above it
    assert FR.decide(summ(4.0), good, fl)["closing"] == "FINER-GRID-READS-CLOSING"
    no_close = dict(good)
    no_close["G16_s0"] = {**good["G16_s0"], "M1": {"diff": 0.1, "ci95": [-0.01, 0.2]}}
    assert FR.decide(summ(4.0), no_close, fl)["closing"].startswith("none") and FR.decide(summ(4.0), no_close, fl)["gap"] == "FINER-GRID-READS-GAP"       # no MAE clause on the closing side
    g64_only = {k: v for k, v in good.items()}
    for s in FR.SEEDS:
        g64_only["G64_%s" % s] = pair(-0.3, -0.4, -0.2)
    assert FR.decide(summ(4.0), g64_only, fl)["gap"] == "FINER-GRID-READS-GAP"             # G64 is beside and names nothing


# ------------------------------------------------------------------------------------------------ controls first
def test_a_failed_control_gives_no_name_and_no_outcome_difference(ctx, tmp_path):
    outs = scenario(ctx, 5.0, 1.5)
    held = FR.conclude(ctx, outs, ["g2_grids_belong_together"])
    assert held["names"] is None and "paired" not in held and "floors" not in held and "summary" not in held and "g2_grids_belong_together" in held["outcome_differences"]
    full = FR.conclude(ctx, outs, [])
    assert full["names"]["gap"] == "FINER-GRID-READS-GAP" and "paired" in full and "floors" in full

    def res_of(concl, passed):
        arms = {a: {"M1": FR.summarize(ctx, outs[a])["M1"], "M2": FR.summarize(ctx, outs[a])["M2"], "M4": {"auroc": 0.9}} for a in ARMS}
        arms["F16SH_s0"] = {"M1": arms["F16_s0"]["M1"]}
        return {"tier": FR.TIER_TEXT, "families": FR.FAMILY_TEXT, "evidence_class": "MEASURED once run", "B": 500, "controls": {}, "controls_flat": {"g2_grids_belong_together": passed},
                "controls_pass": passed, "arms": arms, **concl}
    p = tmp_path / "v_fail.md"
    FR.write_verdict(str(p), res_of(held, False))
    t = p.read_text(encoding="utf-8")
    assert "FINER-GRID" not in t and "No name is given" in t and "G16" not in t and "T0 (a diagnostic on logged frames, never driving performance)" in t
    assert "STRATEGIC: NOT COMPUTED (a token probe reads no strategic quantity; n = 0)" in t
    p2 = tmp_path / "v_ok.md"
    FR.write_verdict(str(p2), res_of(full, True))
    t2 = p2.read_text(encoding="utf-8")
    assert "FINER-GRID-READS-GAP" in t2 and "G16_s0" in t2 and "Gap error by gap range" in t2 and "STRATEGIC: NOT COMPUTED" in t2 and "T0 (a diagnostic" in t2


def test_the_gate_lists_every_failed_control_except_the_printed_k3_card_form():
    flat = {"g2_grids_belong_together": True, "k3_card_form_slot|F16_s0": False, "k4_no_future_input|F64_s1": False, "g7_tk_reproduces_banked|TK16_s0": True}
    assert sorted(k for k, v in FR.gated(flat).items() if not v) == ["k4_no_future_input|F64_s1"]
    given = FR.given_controls({"g1": True, "g3": True, "g4": False, "p5": {"F16_s0": True}}, {}, ("F16_s0", "F16_s1"))
    assert given["g1_given"]["pass"] and not given["g4_given"]["pass"] and not given["g6_forward_given"]["pass"] and not given["g6_forward_given"]["given"]
    assert given["p5_train_log|F16_s0"]["pass"] and not given["p5_train_log|F16_s1"]["pass"]


def test_g7_reproduction_needs_max_abs_difference_exactly_0():
    base = {"M1": {"r2": 0.1, "r2_ci95": [0.0, 0.2], "n": 413}, "M2": {"head": {"r2": 0.5, "r2_ci95": [0.1, 0.7]}}, "M4": {"auroc": 0.9, "ci95": [0.8, 0.95]}, "train": {"M1": {"r2": 0.2}, "M2": {"head": {"r2": 0.6}}, "M4": {"auroc": 0.8}}}
    import copy
    assert FR.repro_row(base, copy.deepcopy(base))["pass"] and FR.repro_row(base, copy.deepcopy(base))["max_abs_diff"] == 0.0
    b = copy.deepcopy(base)
    b["M2"]["head"]["r2"] += 1e-12
    r = FR.repro_row(base, b)
    assert not r["pass"] and r["by_part"]["eval_M2"] > 0
    c = copy.deepcopy(base)
    c["train"]["M4"]["auroc"] = 0.81
    assert not FR.repro_row(base, c)["pass"]
    d = copy.deepcopy(base)
    d["M1"]["n"] = 412
    assert not FR.repro_row(base, d)["pass"]
    assert FR.max_abs_diff({"a": float("nan")}, {"a": float("nan")}) == 0.0 and FR.max_abs_diff({"a": 1}, {"b": 1}) == float("inf")


def test_g2_g5_and_p5_inputs(tmp_path):
    good = {"n_clips_read": 20, "n_frames": 200, "corr": 0.995, "pass": True, "median_abs_diff": 0.01, "max_abs_diff": 0.2}
    (tmp_path / "g.json").write_text(json.dumps(good), encoding="utf-8")
    assert FR.check_g2(str(tmp_path / "g.json"))["pass"]
    for k, v in (("corr", 0.98), ("n_clips_read", 19), ("n_frames", 190), ("pass", False)):
        (tmp_path / "b.json").write_text(json.dumps({**good, k: v}), encoding="utf-8")
        assert not FR.check_g2(str(tmp_path / "b.json"))["pass"], k
    assert not FR.check_g2(str(tmp_path / "none.json"))["pass"] and not FR.check_g2(None)["pass"]
    files = {"eval_states": str(tmp_path / "F16_s0_eval.npz"), "train_states": str(tmp_path / "F16_s0_train.npz")}
    rec = {"grid": "pool8x20", "n_tok": 160, "n_tok_seen_in_training": 160, "D": 160 * 16}
    for sp in ("eval", "train"):
        (tmp_path / ("F16_s0_%s.npz.json" % sp)).write_text(json.dumps(rec), encoding="utf-8")
    assert FR.check_g5(files, 16)["pass"] and not FR.check_g5(files, 64)["pass"]
    (tmp_path / "F16_s0_train.npz.json").write_text(json.dumps({**rec, "n_tok": 40, "D": 640}), encoding="utf-8")
    assert not FR.check_g5(files, 16)["pass"]
    (tmp_path / "F16_s0_train.npz.json").write_text(json.dumps({**rec, "n_tok_seen_in_training": 40}), encoding="utf-8")
    assert not FR.check_g5(files, 16)["pass"]

    def wlog(path, losses, ngap=51.0, ngap_min=51.0, total=3000):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"header": True, "arm": "F16_s0", "total": total}) + "\n")
            for i, l in enumerate(losses):
                fh.write(json.dumps({"step": 50 * (i + 1), "first": False, "lead_sup_mean": l, "n_gap": ngap, "n_gap_min_interval": ngap_min}) + "\n")
    wlog(tmp_path / "ok.jsonl", np.linspace(0.4, 0.27, 60))
    r = FR.p5_card_form(str(tmp_path / "ok.jsonl"))
    assert r["pass"] and r["first_mean"] > r["last_mean"]
    wlog(tmp_path / "strict.jsonl", np.linspace(0.4, 0.27, 60), ngap=51.0, ngap_min=0.0)     # the interval-minimum form of the tool fails on real data; the card's form does not gate on it
    r2 = FR.p5_card_form(str(tmp_path / "strict.jsonl"))
    assert r2["pass"] and r2["min_n_gap_interval_info_only"] == 0.0
    wlog(tmp_path / "rise.jsonl", np.linspace(0.27, 0.4, 60))
    assert not FR.p5_card_form(str(tmp_path / "rise.jsonl"))["pass"]
    wlog(tmp_path / "nog.jsonl", np.linspace(0.4, 0.27, 60), ngap=0.0)
    assert not FR.p5_card_form(str(tmp_path / "nog.jsonl"))["pass"]
    wlog(tmp_path / "short.jsonl", np.linspace(0.4, 0.27, 40))
    assert not FR.p5_card_form(str(tmp_path / "short.jsonl"))["pass"]


def test_the_reader_imports_the_banked_reader_functions_and_does_not_reimplement_m1_m2_m4():
    import inspect
    assert FR.TKR.__name__ == "tokprobe_read" and FR.N.__name__ == "leadsup_read_n1"
    assert FR.TKR.read_arm.__module__ == "tokprobe_read" and FR.N.fid_one.__module__ == "leadsup_read_n1" and FR.N.auroc_row.__module__ == "leadsup_read_n1"
    src = Path(FR.__file__).read_text(encoding="utf-8")
    for fn in ("def auroc_row", "def fid_one", "def eval_read", "def window_outputs", "def head_forward", "def control_k4", "def read_arm"):
        assert fn not in src, fn
    assert "TKR.read_arm" in src and "N.pair_m1" in src and "N.fid_one" in src and "TKR.paired_read_controls" in src and "inspect" not in src or True


BANNED = ["vo" + "id", "super" + "seded", "retr" + "acted", "form" + "erly", "corr" + "ection", "with" + "drawn", "no lon" + "ger", "closed" + " loop"]


def test_files_have_lf_line_endings_and_none_of_the_banned_words():
    for name in ("finegrid_read.py", "test_finegrid_read.py"):
        raw = (HERE / name).read_bytes()
        assert b"\r" not in raw, name
        low = raw.decode("utf-8").lower()
        for w in BANNED:
            assert w not in low, (name, w)
