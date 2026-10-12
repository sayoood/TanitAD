#!/usr/bin/env python3
"""Self-test of eval/rl_val_eval.py (RL-S1 priority 3).  Run: python eval/test_rl_val_eval.py -> ZZTEST_PASS <n> (prints the mutation table).

Expected values are derived BY HAND from the fixture tables below with plain Python arithmetic (never from the module under test).  Each
mutation edits the module's SOURCE (the snippet must occur exactly once) and the suite must then report >= 1 failing check.
Fixture: 5 proposals per set, 12 sets in 6 logs, and the arms
    X picks P0 on sets 0-8, P1 on 9-11     B picks P1 on 0-7, P4 on 8-11     S picks P4 everywhere     T picks P3 everywhere     Z picks P2 everywhere
proposal  (NC DAC EP  TTC C DDC)   PDMS = NC*DAC*(5EP+5TTC+2C)/12       D    path
P0        (1  1   1.0 1   1 1)     12/12        = 1.0                   0.0  50
P1        (1  1   0.5 1   1 1)     (2.5+5+2)/12 = 9.5/12                0.2  40
P2        (0  1   0.0 1   1 1)     0                                    1.0  30
P3        (1  1   0.2 0   1 1)     (1+0+2)/12   = 0.25                  0.4  20
P4        (1  1   0.1 1   1 1)     (0.5+5+2)/12 = 7.5/12                0.1  10
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import types
from fractions import Fraction as F

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "rl_val_eval.py")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


TG = np.array([[1, 1, 1.0, 1, 1, 1], [1, 1, 0.5, 1, 1, 1], [0, 1, 0.0, 1, 1, 1], [1, 1, 0.2, 0, 1, 1], [1, 1, 0.1, 1, 1, 1]], float)
TGF = [[F(1), F(1), F(1), F(1), F(1), F(1)], [F(1), F(1), F(1, 2), F(1), F(1), F(1)], [F(0), F(1), F(0), F(1), F(1), F(1)],
       [F(1), F(1), F(1, 5), F(0), F(1), F(1)], [F(1), F(1), F(1, 10), F(1), F(1), F(1)]]
PDMSF = [r[0] * r[1] * (5 * r[2] + 5 * r[3] + 2 * r[4]) / 12 for r in TGF]
DF = [F(0), F(1, 5), F(1), F(2, 5), F(1, 10)]
LEN = [50, 40, 30, 20, 10]
M = 5
SETS = [(f"LOG{i // 2}", f"tok{i:02d}", 0, 0) for i in range(12)]
CK = 100
PICKS = {"X": [0] * 9 + [1] * 3, "B": [1] * 8 + [4] * 4, "S": [4] * 12, "T": [3] * 12, "Z": [2] * 12}
SCENES = {k[1]: k[0] for k in SETS}


def mk_scores(picks, lane_logit=8.0):
    out = {}
    for k, pk in zip(SETS, picks):
        s = np.full((M, 7), -8.0)
        s[pk, :6] = 8.0
        s[:, 6] = lane_logit
        out[k] = {"ckpt": CK, "scores": s, "space": "logit"}
    return out


def mk_pdm(ckpt=CK, drop=()):
    return {k: {"ckpt": ckpt, "tg": TG.copy()} for i, k in enumerate(SETS) if i not in drop}


def mk_terms(ckpt=CK, drop=(), stale=()):
    return {k: {"ckpt": (ckpt - 1 if i in stale else ckpt), "d": np.array([float(x) for x in DF]), "k": np.zeros(M), "lc": np.zeros(M, bool),
                "fires": False, "dither": np.zeros(M)} for i, k in enumerate(SETS) if i not in drop}


def mk_paths():
    L = np.array(LEN, float)
    return {k: L.copy() for k in SETS}


def mean(xs):
    return sum(xs) / len(xs)


def suite(E) -> list:
    """The checks against module `E`; returns the FAILING check names."""
    bad: list = []

    def t(name, fn):
        try:
            ok = bool(fn())
        except Exception:                                                                       # noqa: BLE001 -- an exception IS a failed check
            ok = False
        if not ok:
            bad.append(name)

    def raises(fn, exc=ValueError):
        try:
            fn()
        except exc:
            return True
        return False

    near = lambda a, b, tol=1e-9: abs(float(a) - float(b)) <= tol                              # noqa: E731
    R = E.rl_reward
    cfg0 = R.RewardCfg(lam_d=0.0, guard=False)                      # R == PDMS (analytic identity)
    cfgD = R.RewardCfg(lam_d=0.15, guard=False)                     # R == PDMS - 0.15 D
    arms = {n: mk_scores(p) for n, p in PICKS.items()}
    run = lambda names=("X", "B"), cfg=cfg0, **kw: E.run({n: arms[n] for n in names}, kw.pop("pdm", mk_pdm()), kw.pop("terms", mk_terms()),  # noqa: E731
                                                       kw.pop("paths", mk_paths()), "navsim_v1_lane", cfg, n_boot=300, **kw)
    res = run(("X", "B", "S", "T", "Z"), cfgD, base="B")
    A = res["arms"]
    # ---- hand-derived arm means (exact fractions)
    def exp_mean(picks, f):
        return mean([f(p) for p in picks])
    for n in ("X", "B", "S", "T", "Z"):
        p = PICKS[n]
        t(f"{n}_pdms", lambda n=n, p=p: near(A[n]["PDMS_x100"], 100 * float(exp_mean(p, lambda i: PDMSF[i]))))
        t(f"{n}_subscores", lambda n=n, p=p: all(near(A[n]["subscores_x100"][s], 100 * float(exp_mean(p, lambda i, c=c: TGF[i][c])))
                                                for c, s in enumerate(("NC", "DAC", "EP", "TTC", "C", "DDC"))))
        t(f"{n}_R_is_pdms_minus_lamD", lambda n=n, p=p: near(A[n]["R_pick_x100"], 100 * float(exp_mean(p, lambda i: PDMSF[i] - F(15, 100) * DF[i]))))
        t(f"{n}_D_at_pick", lambda n=n, p=p: near(A[n]["D_at_pick_mean"], float(exp_mean(p, lambda i: DF[i]))))
    rd, bs = float(mean(PDMSF)), float(max(PDMSF))
    t("fan_random_and_best", lambda: near(A["X"]["fan_random_x100"], 100 * rd) and near(A["X"]["fan_best_x100"], 100 * bs))
    xk = float(exp_mean(PICKS["X"], lambda i: PDMSF[i]))
    t("skill", lambda: near(A["X"]["skill"], (xk - rd) / (bs - rd)))
    t("R_fan_mean_is_fan_mean", lambda: near(A["X"]["R_fan_mean_x100"], 100 * float(mean([PDMSF[i] - F(15, 100) * DF[i] for i in range(M)]))))
    t("zero_picks_counted", lambda: A["Z"]["zero_picks"] == 12 and A["X"]["zero_picks"] == 0 and A["Z"]["unsafe_picks"] == 12 and A["X"]["unsafe_picks"] == 0)
    t("ep_and_ttc_not_swapped", lambda: near(A["T"]["subscores_x100"]["EP"], 20.0) and near(A["T"]["subscores_x100"]["TTC"], 0.0)
      and near(A["Z"]["subscores_x100"]["NC"], 0.0) and near(A["Z"]["subscores_x100"]["DAC"], 100.0))
    # ---- the R terms are wired to rl_reward (default config, guard ON) and the identity at lam 0
    cfgG = R.RewardCfg()
    rg = E.run({"X": arms["X"]}, mk_pdm(), mk_terms(), None, "navsim_v1_lane", cfgG, n_boot=50)
    exp = mean([R.reward(TG, cfgG, d=np.array([float(x) for x in DF]))["R"][i] for i in PICKS["X"]])
    t("R_wired_to_rl_reward_guard_on", lambda: near(rg["arms"]["X"]["R_pick_x100"], 100 * exp))
    r0 = E.run({"X": arms["X"]}, mk_pdm(), None, None, "navsim_v1_lane", cfg0, n_boot=50)
    t("lam0_R_equals_PDMS_exactly", lambda: r0["arms"]["X"]["R_pick_x100"] == r0["arms"]["X"]["PDMS_x100"] and "D_at_pick_mean" not in r0["arms"]["X"])
    # ---- slow-pick telemetry (hand-derived above the table: fan mean 30, median 30)
    tx, tb = A["X"]["slow_pick_telemetry"], A["B"]["slow_pick_telemetry"]
    t("tele_X", lambda: near(tx["pick_over_fan_mean_path"], 19 / 12) and tx["median_rank_by_length_0_longest"] == 0.0
      and tx["share_pick_shorter_than_fan_median"] == 0.0 and near(tx["longest_bin_share"], 0.75) and tx["shortest_bin_share"] == 0.0
      and np.allclose(tx["pick_length_quintile_shares_shortest_first"], [0, 0, 0, 0.25, 0.75]) and near(tx["mean_pick_path_m"], 47.5) and near(tx["mean_fan_path_m"], 30.0))
    t("tele_B", lambda: near(tb["pick_over_fan_mean_path"], 1.0) and tb["median_rank_by_length_0_longest"] == 1.0
      and near(tb["share_pick_shorter_than_fan_median"], 4 / 12) and near(tb["shortest_bin_share"], 4 / 12) and tb["longest_bin_share"] == 0.0
      and np.allclose(tb["pick_length_quintile_shares_shortest_first"], [1 / 3, 0, 0, 2 / 3, 0]) and near(tb["mean_pick_path_m"], 30.0))
    g = {n: A[n]["g3"] for n in ("X", "B", "S")}
    t("g3_flags", lambda: g["B"]["pick_ok"] and g["B"]["share_ok"] and g["X"]["pick_ok"] and not g["S"]["pick_ok"] and not g["S"]["share_ok"])
    t("g3_bounds_are_the_reviews", lambda: near(g["B"]["min_allowed"], 0.894) and near(g["B"]["max_allowed"], 0.55))
    t("path_len_two_terms", lambda: near(E.path_len(np.array([[[25.0, 0.0], [50.0, 0.0]]]))[0], 50.0)
      and near(E.path_len(np.array([[[3.0, 4.0], [3.0, 10.0]]]))[0], 11.0))                       # |(3,4)| = 5, then 6 more
    t("tele_bins_M64_width13", lambda: E.pick_telemetry(np.arange(64.0)[::-1] + 1, 0)["bin"] == 0 and E.pick_telemetry(np.arange(64.0)[::-1] + 1, 63)["bin"] == 4
      and E.pick_telemetry(np.arange(64.0)[::-1] + 1, 51)["bin"] == 3 and E.pick_telemetry(np.arange(64.0)[::-1] + 1, 52)["bin"] == 4)
    # ---- pairing: hand-derived deltas  X-B: sets 0-7 (1 - 9.5/12), set 8 (1 - 7.5/12), sets 9-11 (9.5/12 - 7.5/12)   x100
    d = [F(100) * (PDMSF[0] - PDMSF[1])] * 8 + [F(100) * (PDMSF[0] - PDMSF[4])] + [F(100) * (PDMSF[1] - PDMSF[4])] * 3
    pr = res["paired_vs_base"]["X"]
    pp = pr["primary"]
    t("pair_mean_hand_derived", lambda: near(pp["mean"], float(mean(d))) and near(pp["mean"], 21.180555555555557, 1e-9))
    t("pair_counts", lambda: (pp["better"], pp["worse"], pp["tied"], pp["n"], pp["n_logs"]) == (12, 0, 0, 12, 6))
    t("pair_arm_base_means", lambda: near(pp["arm_mean_x100"], 94.79166666666667) and near(pp["base_mean_x100"], 73.61111111111111))
    t("pair_ci_contains_mean", lambda: pp["ci95"][0] <= pp["mean"] <= pp["ci95"][1])
    dd = {E.kstr(k): float(100 * (PDMSF[PICKS["X"][i]] - PDMSF[PICKS["B"][i]])) for i, k in enumerate(SETS)}
    mu, lo, hi = E.PE.boot(dd, {E.kstr(k): k[0] for k in SETS}, n=300, seed=E.SEED)
    t("pair_is_PE_boot_by_log", lambda: near(pp["ci95"][0], lo) and near(pp["ci95"][1], hi) and near(pp["mean"], mu))
    perset = {n: {k: E.eval_set(arms[n][k], {"tg": TG, "terms": None, "L": None}, "navsim_v1_lane", cfg0) for k in SETS} for n in ("X", "B")}
    big = E.read_one(perset["X"], perset["B"], sorted(SETS), "PDMS", n=10000, seed=20260927)
    mu2, lo2, hi2 = E.PE.boot(dd, {E.kstr(k): k[0] for k in SETS}, n=10000, seed=20260927)
    t("registered_estimator_defaults", lambda: near(big["ci95"][0], lo2) and near(big["ci95"][1], hi2) and E.N_BOOT == 10000 and E.SEED == 20260927)
    t("pair_subscore_deltas", lambda: near(pr["quantities"]["EP"]["mean"], 100 * float(mean([TGF[PICKS["X"][i]][2] - TGF[PICKS["B"][i]][2] for i in range(12)]))))
    t("pair_D_delta", lambda: near(pr["quantities"]["D"]["mean"], 100 * float(mean([DF[PICKS["X"][i]] - DF[PICKS["B"][i]] for i in range(12)]))))
    t("pair_adverse_flags", lambda: pr["adverse"] == [] and "PDMS" in res["paired_vs_base"]["Z"]["adverse"] and res["paired_vs_base"]["Z"]["K4"]["breach"]
      and not pr["K4"]["breach"] and not res["paired_vs_base"]["Z"]["g5"]["NC_fall_ok"] and pr["g5"]["NC_fall_ok"])
    # ---- the self-checks: control GREEN, deliberate regression RED
    t("selfpair_exactly_zero", lambda: res["selfcheck"]["selfpair"]["ok"] and res["selfcheck"]["swap"]["ok"])
    fw = E.read_pair(perset["X"], perset["B"], 300)
    rv = E.read_pair(perset["B"], perset["X"], 300)
    t("swap_flips_sign_bitwise", lambda: fw["primary"]["mean"] == -rv["primary"]["mean"] and rv["primary"]["better"] == 0 and rv["primary"]["worse"] == 12)
    t("swap_check_sees_a_non_swap", lambda: E.check_swap(fw, rv)["ok"] and not E.check_swap(fw, fw)["ok"])
    t("selfpair_check_sees_a_non_selfpair", lambda: E.check_selfpair(E.read_pair(perset["X"], perset["X"], 300))["ok"] and not E.check_selfpair(fw)["ok"])
    sub = {k: perset["B"][k] for k in SETS[:9]}
    t("pair_uses_only_common_sets", lambda: E.read_pair(perset["X"], sub, 100)["n_sets"] == 9)
    # ---- selection rule
    two = np.array([[4.0] * 6 + [-2.0], [3.0] * 6 + [2.0]])
    t("select_v1_vs_lane", lambda: E.select(two, "navsim_v1")[0] == 0 and E.select(two, "navsim_v1_lane")[0] == 1)
    sig = np.array([[3.0] * 6, [-4.0, -4.0, 3.0, 3.0, 3.0, 3.0]])                               # raw product picks row 1, the sigmoid picks row 0
    t("select_applies_sigmoid", lambda: E.select(sig, "navsim_v1")[0] == 0)
    wt = np.array([[4.0, 4.0, 4.0, 4.0, -4.0, 4.0], [4.0, 4.0, 4.0, -4.0, 4.0, 4.0]])           # TTC (weight 5) beats C (weight 2)
    t("select_weights_5_5_2", lambda: E.select(wt, "navsim_v1")[0] == 0)
    t("select_tie_lowest_index", lambda: E.select(np.zeros((4, 6)), "navsim_v1")[0] == 0)
    t("select_lane_needs_7", lambda: raises(lambda: E.select(np.zeros((3, 6)), "navsim_v1_lane")) and raises(lambda: E.select(np.zeros((3, 6)), "bogus")))
    t("select_prob_space_not_sigmoided", lambda: near(E.select(np.array([[0.9] * 6, [0.5] * 6]), "navsim_v1", "prob")[1][0], 0.729))
    # ---- eligibility: stale, missing, foreign, coverage
    rs = run(("X", "B"), cfgD, pdm=mk_pdm(ckpt=CK), terms=mk_terms(drop=(5,), stale=(6,)))
    t("terms_missing_and_stale_dropped", lambda: rs["eligibility"]["missing_terms"] == 1 and rs["eligibility"]["stale_or_mismatched_ckpt"] == 1 and rs["eligibility"]["sets_used"] == 10)
    pd = mk_pdm(drop=(4,))
    pd[SETS[3]]["ckpt"] = CK - 1
    re_ = run(("X", "B"), cfg0, pdm=pd)
    t("pdm_missing_and_stale_dropped", lambda: re_["eligibility"]["missing_pdm"] == 1 and re_["eligibility"]["stale_or_mismatched_ckpt"] == 1 and re_["eligibility"]["sets_used"] == 10)
    short = dict(arms["B"])
    short.pop(SETS[7])
    ru = E.run({"X": arms["X"], "B": short}, mk_pdm(), None, None, "navsim_v1_lane", cfg0, n_boot=50)
    t("unequal_coverage_intersected", lambda: ru["eligibility"]["sets_scored_by_every_arm"] == 11 and ru["eligibility"]["sets_used"] == 11)
    foreign = dict(SCENES)
    foreign.pop("tok03")
    t("foreign_scene_refused", lambda: raises(lambda: run(("X", "B"), cfg0, scenes=foreign)))
    extra = dict(SCENES)
    extra["tokZZ"] = "LOGZ"
    rc = run(("X", "B"), cfg0, scenes=extra)
    t("coverage_reported", lambda: near(rc["eligibility"]["rl_val_scene_coverage"], 12 / 13) and rc["eligibility"]["rl_val_scenes"] == 13 and rc["eligibility"]["rl_val_logs"] == 7)
    t("require_full_refuses_a_gap", lambda: raises(lambda: run(("X", "B"), cfg0, scenes=extra, require_full=True)))
    t("require_full_passes_when_full", lambda: run(("X", "B"), cfg0, scenes=dict(SCENES), require_full=True)["eligibility"]["rl_val_scene_coverage"] == 1.0)
    t("missing_path_counted_and_row_none", lambda: (lambda r: r["eligibility"]["path_lengths_missing"] == 1 and r["arms"]["X"]["slow_pick_telemetry"] is None)(
        run(("X", "B"), cfg0, paths={k: v for k, v in mk_paths().items() if k != SETS[2]})))
    # ---- I/O round trip and the CLI
    def io_cli():
        out = {}
        with tempfile.TemporaryDirectory() as d:
            keys = SETS
            for n in ("X", "B"):
                E.write_scores_npz(os.path.join(d, n + ".npz"), keys, [CK] * 12, np.stack([arms[n][k]["scores"] for k in keys]))
            back = E.load_scores(os.path.join(d, "X.npz"))
            out["npz_roundtrip"] = all(np.array_equal(back[k]["scores"], arms["X"][k]["scores"]) and back[k]["ckpt"] == CK and back[k]["space"] == "logit" for k in keys)
            E.write_scores_npz(os.path.join(d, "dup.npz"), keys[:1] * 2, [CK, CK], np.zeros((2, M, 7)))
            out["duplicate_scores_refused"] = raises(lambda: E.load_scores(os.path.join(d, "dup.npz")))
            with open(os.path.join(d, "x.jsonl"), "w") as f:
                for k in keys:
                    f.write(json.dumps({"kind": "rl_scores", "key": list(k), "ckpt_step": CK, "scores": arms["X"][k]["scores"].tolist(), "space": "logit"}) + "\n")
            out["jsonl_roundtrip"] = len(E.load_scores(os.path.join(d, "x.jsonl"))) == 12
            os.makedirs(os.path.join(d, "pdm"))
            os.makedirs(os.path.join(d, "terms"))
            os.makedirs(os.path.join(d, "sets"))
            names = ("nc", "dac", "ep", "ttc", "c", "ddc")
            with open(os.path.join(d, "pdm", "pdm_a.jsonl"), "w") as f:
                for k in keys:
                    rec = {"kind": "pdm_targets", "key": list(k), "ckpt_step": CK, "pdms": [float(x) for x in PDMSF]}
                    rec.update({nm: [float(TG[i, c]) for i in range(M)] for c, nm in enumerate(names)})
                    f.write(json.dumps(rec) + "\n")
            with open(os.path.join(d, "terms", "rl_terms_a.jsonl"), "w") as f:
                for k in keys:
                    f.write(json.dumps({"kind": "rl_terms", "key": list(k), "ckpt_step": CK, "d": [float(x) for x in DF], "k": [0.0] * M,
                                        "lc_executed": [False] * M, "label_fires": False, "dither": [0.0] * M}) + "\n")
            with open(os.path.join(d, "sets", "onpolicy_a.jsonl"), "w") as f:
                for k in keys:
                    f.write(json.dumps({"kind": "onpolicy_set", "log_name": k[0], "token": k[1], "step": k[2], "rank": k[3], "ckpt_step": CK,
                                        "traj": [[[L / 2, 0.0], [float(L), 0.0]] for L in LEN]}) + "\n")
            with open(os.path.join(d, "val.tsv"), "w") as f:
                f.write("token\tlog\n" + "".join(f"{k[1]}\t{k[0]}\n" for k in keys))
            base = ["--scores", f"X={d}/X.npz", f"B={d}/B.npz", "--base", "B", "--pdm", d + "/pdm", "--terms", d + "/terms", "--sets", d + "/sets",
                    "--rl-val-scenes", d + "/val.tsv", "--rule", "navsim_v1_lane", "--n-boot", "200", "--require-full"]
            with contextlib.redirect_stdout(io.StringIO()):
                rc1 = E.main(base + ["--out", d + "/r.json"])
                rc2 = E.main([a for a in base if a != "--terms" and a != d + "/terms"] + ["--out", d + "/r2.json"])
                rc3 = E.main([a for a in base if a != "--terms" and a != d + "/terms"] + ["--lam-d", "0", "--out", d + "/r3.json"])
            r = json.load(open(d + "/r.json"))
            out["cli_rc_ok"] = rc1 == 0
            out["cli_pdms"] = near(r["arms"]["X"]["PDMS_x100"], 94.79166666666667) and near(r["arms"]["B"]["PDMS_x100"], 73.61111111111111)
            out["cli_paired"] = near(r["paired_vs_base"]["X"]["primary"]["mean"], 21.180555555555557)
            out["cli_telemetry_from_sets"] = near(r["arms"]["B"]["slow_pick_telemetry"]["pick_over_fan_mean_path"], 1.0)
            out["cli_input_hashes"] = len(r["inputs"]["scores_sha256"]["X"]) == 64 and r["rl_val_scenes_checked"] is True
            out["cli_selfcheck_in_output"] = r["selfcheck"]["selfpair"]["ok"] and r["selfcheck"]["swap"]["ok"]
            out["cli_needs_terms_unless_lam_d_0_named"] = rc2 == 4 and rc3 == 0
        return out
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ioc = io_cli()
    except Exception as e:                                                                       # noqa: BLE001
        ioc = {f"raised_{type(e).__name__}": False}
    for k, ok in ioc.items():
        t("io_" + k, lambda ok=ok: ok)
    return bad


# (name, class, old snippet, new snippet) -- each snippet must occur EXACTLY once in rl_val_eval.py
MUTATIONS = [
    ("pick_is_argmin", "select", "return int(np.argmax(sel)), sel", "return int(np.argmin(sel)), sel"),
    ("lane_factor_dropped", "select", "sel = sel * p[:, 6]", "sel = sel * 1.0"),
    ("lane_factor_always_on", "select", 'if rule == "navsim_v1_lane":\n        if s.shape[-1] < 7:', 'if True:\n        if s.shape[-1] < 7:'),
    ("sigmoid_skipped", "select", 'p = sigmoid(s) if space == "logit" else s', "p = s"),
    ("weights_ttc_c_swapped", "select", "(5.0 * p[:, 2] + 5.0 * p[:, 3] + 2.0 * p[:, 4]) / 12.0", "(5.0 * p[:, 2] + 2.0 * p[:, 3] + 5.0 * p[:, 4]) / 12.0"),
    ("ep_ttc_columns_swapped", "lookup", '"EP": float(s["ep"][i]), "TTC": float(s["ttc"][i])', '"EP": float(s["ttc"][i]), "TTC": float(s["ep"][i])'),
    ("zero_flag_wrong", "lookup", 'float(pdms[i] == 0.0)', 'float(pdms[i] < 0.0)'),
    ("skill_denominator_flipped", "lookup", "(pk - rd) / max(bs - rd, 1e-9)", "(pk - rd) / max(rd - bs, 1e-9)"),
    ("R_taken_at_fan_mean", "lookup", '"R": float(res["R"][i])', '"R": float(res["R"].mean())'),
    ("D_taken_at_index_0", "lookup", 'out["D"] = float(np.asarray(d)[i])', 'out["D"] = float(np.asarray(d)[0])'),
    ("tele_rank_direction", "telemetry", "rk = int((-L).argsort().argsort()[i])", "rk = int((L).argsort().argsort()[i])"),
    ("tele_shorter_flag_flipped", "telemetry", 'L[i] < np.median(L)', 'L[i] > np.median(L)'),
    ("tele_bin_width_fixed_13", "telemetry", "width = int(math.ceil(M / 5.0))", "width = 13"),
    ("tele_ratio_inverted", "telemetry", 'float(L[i] / L.mean())', 'float(L.mean() / L[i])'),
    ("g3_floor_removed", "bounds", "G3_PICK_FAN_MEAN_MIN = 0.914 - 0.02", "G3_PICK_FAN_MEAN_MIN = 0.0"),
    ("K4_breach_sign", "bounds", 'bool(res["PDMS"]["mean"] < -K4_PDMS_FALL_MAX)', 'bool(res["PDMS"]["mean"] > -K4_PDMS_FALL_MAX)'),
    ("delta_sign_flipped", "pair", "scale * (arm[k][key] - base[k][key])", "scale * (base[k][key] - arm[k][key])"),
    ("delta_biased", "pair", "scale * (arm[k][key] - base[k][key])", "scale * (arm[k][key] - base[k][key]) + 1e-9"),
    ("pair_on_arm_sets_only", "pair", "ks = sorted(set(arm) & set(base))", "ks = sorted(set(arm))"),
    ("cluster_unit_is_the_set", "pair", "tl = {kstr(k): k[0] for k in ks}", "tl = {kstr(k): kstr(k) for k in ks}"),
    ("stale_ckpt_accepted", "eligibility", 'if cks != {p["ckpt"]}:', "if False:"),
    ("terms_stale_accepted", "eligibility", 't_ok = t is not None and t["ckpt"] == p["ckpt"]', "t_ok = t is not None"),
    ("foreign_scene_accepted", "eligibility", "if foreign:", "if False:"),
    ("coverage_hardcoded", "eligibility", 'elig["rl_val_scene_coverage"] = len(covered & set(scenes)) / max(len(scenes), 1)', 'elig["rl_val_scene_coverage"] = 1.0'),
    ("require_full_ignored", "eligibility", "if require_full and", "if False and"),
    ("swap_check_blind", "selfcheck", 'return {"ok": all(per.values()), "per_quantity": per, "tolerance": SWAP_TOL}', 'return {"ok": True, "per_quantity": per, "tolerance": SWAP_TOL}'),
    ("selfpair_check_blind", "selfcheck", 'return {"ok": bool(ok), "mean": p["mean"]', 'return {"ok": True, "mean": p["mean"]'),
    ("duplicate_scores_silent", "io", 'if k in out:\n            raise ValueError(f"duplicate scores', 'if False:\n            raise ValueError(f"duplicate scores'),
    ("cli_terms_guard_removed", "io", "if (cfg.lam_d != 0.0 or cfg.use_k or cfg.use_l) and terms is None:", "if False:"),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    n = 0
    real = suite(load(src, "rl_val_eval_real"))
    assert not real, f"the unmutated module FAILS its own checks: {real}"
    n += 1
    print(f"{'mutation':32s} {'class':12s} caught-by (failing checks)")
    for i, (name, cls, old, new) in enumerate(MUTATIONS):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        try:
            mod = load(src.replace(old, new), f"rl_val_eval_mut{i}")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        try:
            failing = suite(mod)
        except Exception as e:                                                                   # noqa: BLE001
            failing = [f"suite raised {type(e).__name__}: {str(e)[:60]}"]
        assert failing, f"mutation {name} ({cls}) stayed GREEN: the suite cannot see it"
        print(f"{name:32s} {cls:12s} {', '.join(failing[:4])}")
        n += 1
    need = {"select", "lookup", "telemetry", "bounds", "pair", "eligibility", "selfcheck", "io"}
    assert {c for _, c, _, _ in MUTATIONS} >= need, "a required mutation class is missing"
    n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
