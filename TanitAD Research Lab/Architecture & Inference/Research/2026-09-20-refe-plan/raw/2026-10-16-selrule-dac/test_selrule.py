#!/usr/bin/env python3
"""Tests of selrule.py: synthetic sets with KNOWN answers for every rule and oracle, the registered grid and tie rule, the estimator pinned to
proxy_eval.boot, the committed outcome table, the identity with rl_val_eval, and DELIBERATE-REGRESSION arms (each mutation of selrule.py must turn a
check RED):  M1 veto fallback removed;  M2 k applied to EP instead of the gates;  M3 tie-break inverted;  M4 oracle uses the PREDICTED gate.

    python test_selrule.py        -> ZZTEST_PASS <checks> (<functions> tests, <mutations> mutations RED)
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "selrule.py")
sys.path.insert(0, HERE)
import selrule as SR  # noqa: E402

PK = SR.PK
import proxy_eval as PE  # noqa: E402
import rl_reward as RR  # noqa: E402
import rl_val_eval as RV  # noqa: E402

N_CHECKS = 0


def chk(cond, msg):
    global N_CHECKS
    N_CHECKS += 1
    if not cond:
        raise AssertionError(msg)


def mk(nc, dac, f, lane=None):
    """One synthetic set, probabilities [M, 6 or 7]: EP = TTC = C = f (so the weighted term (5EP + 5TTC + 2C)/12 = f); DDC = 1."""
    nc, dac, f = (np.asarray(x, np.float64) for x in (nc, dac, f))
    cols = [nc, dac, f, f, f, np.ones_like(f)]
    if lane is not None:
        cols.append(np.asarray(lane, np.float64))
    return np.stack(cols, -1)


def truth(nc, dac):
    """true sub-scores [M, 6] with EP = TTC = C = DDC = 1 (only the two gates matter to the oracles)."""
    nc, dac = np.asarray(nc, np.float64), np.asarray(dac, np.float64)
    return np.stack([nc, dac, np.ones_like(nc), np.ones_like(nc), np.ones_like(nc), np.ones_like(nc)], -1)


# S1: B = 0 (agg .55), the DAC veto and the sharpening both move off it.   agg = [.55, .475, .4, .3]
S1 = mk([1, 1, 1, 1], [.55, .95, .8, .3], [1, .5, .5, 1])
S1_TRUE_NC, S1_TRUE_DAC = [0, 1, 1, 1], [0, 1, 1, 1]
# S2: no proposal clears any veto (every p_DAC < 0.5): B = 2 and V(tau) must FALL BACK to it
S2 = mk([1, 1, 1, 1], [.1, .4, .3, .2], [.2, .4, 1, .9])
# S3: the NC gate alone: agg = [.9, .85]; K(2) -> NC^2 = .81 < .85 -> proposal 1;  an EP-power instead gives proposal 0
S3 = mk([.9, 1.0], [1, 1], [1, .85])


# ----------------------------------------------------------------------------------------------------------------- behaviour of a module
def behaviour_checks(M):
    """-> list of failed check names for module M (selrule or a mutant of it).  [] = every known answer reproduced."""
    bad = []

    def ok(cond, nm):
        if not cond:
            bad.append(nm)

    try:
        ok(M.pick_B(S1) == 0 and M.pick_B(S2) == 2 and M.pick_B(S3) == 0, "B_known")
        # V: tau .5 keeps B (p_DAC .55 >= .5); .6 vetoes proposal 0 -> best qualified = 1 (agg .475); .9 -> only proposal 1 qualifies
        ok([M.pick_V(S1, t) for t in (0.5, 0.6, 0.7, 0.8, 0.9)] == [0, 1, 1, 1, 1], "V_known")
        ok(M.pick_V(S1, 0.0) == M.pick_B(S1), "V0_is_B")
        # the fallback: nobody qualifies -> B (2), for every registered tau
        ok(all(M.pick_V(S2, t) == 2 for t in M.GRID_TAU), "V_fallback_is_B")
        ok(M.pick_K(S1, 1) == M.pick_B(S1) and M.pick_K(S3, 1) == 0, "K1_is_B")
        ok(M.pick_K(S1, 2) == 1, "K_gates_known_S1")                     # (.55^2=.3025) < (.95^2 * .5=.451)
        ok(M.pick_K(S3, 2) == 1, "K_gates_known_S3")                     # NC^2: .81 < .85
        ok(M.pick_K(S1, 8) == 1, "K8_known")
        ok(M.pick_O(S1, truth(S1_TRUE_NC, S1_TRUE_DAC), "DAC") == 3, "O_DAC_known")   # DAC [0,1,1,1] -> agg [0, .5, .5, 1.0]
        ok(M.pick_O(S1, truth(S1_TRUE_NC, S1_TRUE_DAC), "NC") == 1, "O_NC_known")     # NC [0,1,1,1]  -> agg [0, .475, .4, .3]
        ok(M.pick_O(S1, truth(S1_TRUE_NC, S1_TRUE_DAC), "both") == 3, "O_both_known")
        # an oracle whose "true" gate equals the predicted one must be B
        ok(M.pick_O(S1, np.asarray(S1)[:, :6], "both") == M.pick_B(S1), "O_with_true_eq_pred_is_B")
        # the order of the registered choice
        ok(M.choose({"V(0.5)": 89.0, "V(0.6)": 89.0, "V(0.7)": 90.5, "V(0.8)": 90.0, "V(0.9)": 89.0, "K(2)": 90.0, "K(3)": 90.0, "K(5)": 89.0, "K(8)": 88.0})[0] == "V(0.7)", "choose_clear_max")
        base = {"V(0.5)": 80.0, "V(0.6)": 80.0, "V(0.7)": 80.0, "V(0.8)": 90.46, "V(0.9)": 80.0, "K(2)": 80.0, "K(3)": 90.50, "K(5)": 80.0, "K(8)": 80.0}
        ok(M.choose(base)[0] == "V(0.8)", "tie_goes_to_smaller_change_V_before_K")      # K(3) is the max, V(0.8) is within .04
        base2 = dict(base, **{"V(0.8)": 90.43})
        ok(M.choose(base2)[0] == "K(3)", "no_tie_beyond_tolerance")                       # .07 apart
        base3 = {n: 80.0 for n in base}
        base3.update({"K(2)": 91.00, "K(5)": 90.99, "V(0.9)": 90.97})
        ok(M.choose(base3)[0] == "V(0.9)" and M.choose(base3)[1] == ["V(0.9)", "K(2)", "K(5)"], "tie_three_way_V_first")
        base4 = {n: 80.0 for n in base}
        base4.update({"K(3)": 91.0, "K(2)": 90.97})
        ok(M.choose(base4)[0] == "K(2)", "tie_lower_k")
        base5 = {n: 80.0 for n in base}
        base5.update({"V(0.6)": 90.00, "K(2)": 90.05})
        ok(M.choose(base5)[0] == "V(0.6)", "tie_boundary_inclusive")
    except Exception as e:  # a mutant that crashes is RED too
        bad.append(f"EXC:{type(e).__name__}:{e}")
    return bad


def load_mutant(old, new):
    src = open(SRC, encoding="utf-8").read()
    if src.count(old) != 1:
        raise AssertionError(f"mutation anchor not found exactly once ({src.count(old)}): {old!r}")
    mod = types.ModuleType("selrule_mut")
    mod.__file__ = SRC
    exec(compile(src.replace(old, new), SRC, "exec"), mod.__dict__)
    return mod


# ----------------------------------------------------------------------------------------------------------------- tests
def test_registered_constants():
    chk(SR.B_BOOT == 10000 and SR.SEED == 20260927 and SR.TIE_TOL == 0.05, "bootstrap / seed / tie tolerance as registered")
    chk(SR.GRID_TAU == (0.5, 0.6, 0.7, 0.8, 0.9) and SR.GRID_K == (2, 3, 5, 8), "the grid as registered")
    chk([SR.name(c) for c in SR.CANDIDATES] == ["V(0.5)", "V(0.6)", "V(0.7)", "V(0.8)", "V(0.9)", "K(2)", "K(3)", "K(5)", "K(8)"], "registered order, V before K, 9 candidates, B not among them")
    chk(SR.COL == {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "C": 4, "DDC": 5, "LANE": 6}, "column order = rl_reward.COMPONENTS (+ lane)")
    chk(tuple(c.upper() for c in RR.COMPONENTS) == SR.SUBS, "sub-score order")
    chk(SR.GATE_COLS == (0, 1), "the gates are NC and DAC")
    chk(SR.SEL_CEILING_PTS == 3.5376954888978354, "the +3.54 ceiling of P2_MEASURED.json")


def test_behaviour_of_the_real_module():
    bad = behaviour_checks(SR)
    chk(bad == [], f"known answers failed: {bad}")


def test_tie_break_of_argmax_is_lowest_index():
    s = mk([1, 1, 1], [.9, .9, .9], [.8, .8, .8])
    chk(SR.pick_B(s) == 0 and RV.select(s, "navsim_v1", space="prob")[0] == 0, "identical proposals -> lowest index, as rl_val_eval")
    chk(SR.pick_V(s, 0.5) == 0 and SR.pick_K(s, 3) == 0, "V and K keep the lowest-index tie rule")


def test_lane_factor_only_in_the_secondary():
    s = mk([1, 1], [1, 1], [.9, .8], lane=[.1, .9])
    chk(SR.pick_B(s, lane=False) == 0, "primary B ignores column 6")
    chk(SR.pick_B(s, lane=True) == 1 and RV.select(s, "navsim_v1_lane", space="prob")[0] == 1, "secondary B = navsim_v1_lane")
    chk(SR.pick_V(s, 0.5, lane=True) == 1 and SR.pick_K(s, 2, lane=True) == 1, "V and K carry the lane flag")


def test_oracle_with_nc_half_is_not_binarised():
    # a true NC of 0.5 (static collision) multiplies the proposal's agg by .5, it is not rounded to 0 or 1
    s = mk([1, 1], [1, 1], [.6, 1.0])
    t = truth([1.0, 0.5], [1, 1])
    chk(SR.pick_O(s, t, "NC") == 0, "NC .5 -> agg [.6, .5] -> proposal 0 (a binarised-to-0 or -to-1 oracle would give 1)")
    t2 = truth([1.0, 0.7], [1, 1])
    chk(SR.pick_O(s, t2, "NC") == 1, "control: NC .7 -> agg [.6, .7] -> proposal 1")


def test_identity_with_rl_val_eval_on_random_sets():
    rng = np.random.default_rng(7)
    S = rng.normal(0.0, 2.5, size=(60, 64, 7))
    P = RV.sigmoid(S)
    pk = SR.picks(("B",), P)
    pl = SR.picks(("B",), P, lane=True)
    chk(all(int(RV.select(S[i], "navsim_v1")[0]) == int(pk[i]) for i in range(60)), "B == RV.select(navsim_v1) on logits, 60 random sets")
    chk(all(int(RV.select(S[i], "navsim_v1_lane")[0]) == int(pl[i]) for i in range(60)), "B_lane == RV.select(navsim_v1_lane)")
    # an independent vectorised aggregate (not RV's): the same argmax
    a = P[..., 0] * P[..., 1] * (5 * P[..., 2] + 5 * P[..., 3] + 2 * P[..., 4]) / 12.0
    chk(np.array_equal(a.argmax(-1), pk), "B == an independent numpy aggregate")
    chk(np.array_equal(SR.picks(("K", 1), P), pk) and np.array_equal(SR.picks(("V", 0.0), P), pk), "K(1) == B and V(0) == B on 60 sets")
    tg = np.clip(P[..., :6], 0, 1)
    chk(np.array_equal(SR.picks(("O", "both"), P, tg), pk), "oracle with true == predicted is B on 60 sets")
    chk(not np.array_equal(SR.picks(("K", 8), P), pk), "control: K(8) is not B on random sets (the check can fail)")


def synth_rlval(n_sets=40, M=16, n_logs=5, seed=3):
    rng = np.random.default_rng(seed)
    keys = [(f"log{i % n_logs:02d}", f"tok{i:04d}", 0, 0) for i in range(n_sets)]
    S = rng.normal(0.5, 2.0, size=(n_sets, M, 7))
    tg = np.stack([rng.choice([0.0, 0.5, 1.0], size=(n_sets, M), p=[.1, .05, .85]), rng.choice([0.0, 1.0], size=(n_sets, M), p=[.1, .9]),
                   rng.uniform(0, 1, size=(n_sets, M)), rng.choice([0.0, 1.0], size=(n_sets, M)), rng.choice([0.0, 1.0], size=(n_sets, M), p=[.05, .95]),
                   rng.choice([0.0, .5, 1.0], size=(n_sets, M))], -1)
    return keys, S, tg


def test_curves_and_identity_block_on_a_synthetic_rlval():
    keys, S, tg = synth_rlval()
    pdms = SR.labels_pdms(tg)
    P = RV.sigmoid(S)
    boot = SR.LogBoot([k[0] for k in keys], 300, SR.SEED)
    cur, pk = SR.curves(P, tg, pdms, boot, lane=False)
    chk(set(cur) == {"B"} | {SR.name(c) for c in SR.CANDIDATES} | {"O-DAC", "O-NC", "O-both"}, "curves() covers B, 9 candidates, 3 oracles")
    chk(cur["B"]["PDMS_x100"] == float(100 * pdms[np.arange(len(pdms)), pk["B"]].mean()), "B mean PDMS is the label PDMS at the pick")
    chk(all("delta_vs_B" in cur[n] for n in cur if n != "B") and "delta_vs_B" not in cur["B"], "delta vs B on everything except B")
    cur2, pk2 = SR.curves(P, tg, pdms, boot, lane=False)
    chk(all(cur[n]["picks_sha256"] == cur2[n]["picks_sha256"] for n in cur), "curves() is deterministic (picks_sha256 identical on re-run)")
    # identity block against rl_val_eval.run on the very same dicts
    sc = {k: {"ckpt": 1, "scores": S[i], "space": "logit"} for i, k in enumerate(keys)}
    pdm = {k: {"ckpt": 1, "tg": tg[i]} for i, k in enumerate(keys)}
    _, pk_l = SR.curves(P, tg, pdms, boot, lane=True)
    ib = SR.identity_block(keys, S, pdms, pk["B"], pk_l["B"], pdm, sc, None)
    chk(ib["ok"] and ib["B_pick_mismatches_vs_RV_select_navsim_v1"] == 0, "identity block OK on a faithful B")
    wrong = (pk["B"] + 1) % S.shape[1]
    ib2 = SR.identity_block(keys, S, pdms, wrong, pk_l["B"], pdm, sc, None)
    chk(not ib2["ok"] and ib2["B_pick_mismatches_vs_RV_select_navsim_v1"] == len(keys), "DELIBERATE REGRESSION: a shifted B turns the identity RED on every set")


def test_logboot_equals_proxy_eval_boot():
    rng = np.random.default_rng(11)
    logs = [f"L{rng.integers(0, 9):02d}" for _ in range(200)]
    vals = rng.normal(0, 1, 200)
    toks = [f"t{i:04d}" for i in range(200)]
    mu, lo, hi = PE.boot(dict(zip(toks, vals)), dict(zip(toks, logs)), n=500, seed=SR.SEED)
    r = SR.LogBoot(logs, 500, SR.SEED).ratio(vals)
    chk(abs(r["mean"] - mu) < 1e-12 and abs(r["ci95"][0] - lo) < 1e-9 and abs(r["ci95"][1] - hi) < 1e-9, f"LogBoot == proxy_eval.boot: {r} vs {(mu, lo, hi)}")
    b = SR.LogBoot(logs, 500, SR.SEED)
    p = SR.paired(vals + 1.0, vals, b, scale=1.0)
    chk(abs(p["mean"] - 1.0) < 1e-12 and abs(p["ci95"][0] - 1.0) < 1e-9 and abs(p["ci95"][1] - 1.0) < 1e-9, "paired delta of a constant +1 is +1 with a degenerate interval")
    chk(p["better"] == 200 and p["worse"] == 0 and not p["adverse"] and p["separated"], "better / worse / adverse / separated flags")
    q = SR.paired(vals - 1.0, vals, b, scale=1.0)
    chk(q["adverse"] and q["worse"] == 200, "a constant -1 is adverse")
    z = SR.paired(vals, vals, b)
    chk(z["mean"] == 0.0 and z["ci95"] == [0.0, 0.0] and z["tied"] == 200 and not z["separated"], "self-pair is exactly 0 [0, 0]")


def test_ratio_for_command_splits():
    logs = ["a", "a", "b", "b", "c", "c"]
    num = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    den = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    b = SR.LogBoot(logs, 200, SR.SEED)
    r = b.ratio(num, den)
    chk(abs(r["mean"] - 3.0 / 4.0) < 1e-12, "ratio of sums over the masked tokens")


def test_outcome_table():
    o = SR.outcome
    chk(o([-0.5, -0.1], [])["outcome"] == "REFUTED", "upper bound < 0 -> REFUTED")
    chk(o([-0.5, 0.2], [])["outcome"] == "NOT PROVEN", "straddle -> NOT PROVEN")
    chk(o([0.0, 0.3], [])["outcome"] == "NOT PROVEN", "lower bound == 0 straddles (strict >)")
    chk(o([-0.3, 0.0], [])["outcome"] == "NOT PROVEN", "upper bound == 0 straddles (strict <)")
    chk(o([0.1, 0.4], [], [], True)["outcome"] == "ADOPT", "lo > 0, no adverse sub-score, no adverse family -> ADOPT")
    r = o([0.1, 0.4], ["DAC"], [], True)
    chk(r["outcome"].startswith("NOT ADOPTED") and "DAC" in r["why"], "lo > 0 with an adverse sub-score is the A7 cell, named")
    r = o([0.1, 0.4], [], ["longitudinal.along_mae_m"], True)
    chk(r["outcome"].startswith("NOT ADOPTED") and r["adopt_if_subscore_leg_only"], "lo > 0 with an adverse family component is the A7 cell; the sub-score-only reading is shown beside it")
    r = o([0.1, 0.4], [], None, False)
    chk(r["outcome"].startswith("UNRESOLVED") and r["adopt_if_subscore_leg_only"], "families not computed -> UNRESOLVED, never ADOPT")


def test_compare_known_answers():
    # 4 tokens x 3 proposals; a = picks (1,1,1,1), b = picks (0,0,0,0); pdms chosen so b has two zero tokens that a rescues, and a has one new zero
    pdms = np.array([[0.0, 0.8, 0.5], [0.0, 0.6, 0.5], [0.7, 0.0, 0.5], [0.9, 0.9, 0.5]])
    sub = np.zeros((4, 3, 6))
    sub[..., 0] = 1.0
    sub[..., 1] = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], float)
    boot = SR.LogBoot(["x", "x", "y", "y"], 200, SR.SEED)
    c = SR.compare(pdms, sub, np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), boot)
    chk(c["zero_tokens"] == {"b_zero": 2, "a_zero": 1, "rescued_b0_to_a_nonzero": 2, "new_zero_b_nonzero_to_a0": 1}, f"zero-token movements: {c['zero_tokens']}")
    chk(c["dac0"] == {"b": 2, "a": 1} and c["nc0"] == {"b": 0, "a": 0}, "DAC = 0 / NC = 0 counts at the picks")
    chk(abs(c["PDMS"]["mean"] - 100 * (0.8 + 0.6 + 0.0 + 0.9 - 0.0 - 0.0 - 0.7 - 0.9) / 4) < 1e-9, "DeltaPDMS x100")
    chk(c["picks_changed"] == 4, "picks changed")
    sq = SR.selection_quality(pdms, np.array([1, 1, 1, 1]))
    chk(abs(sq["actual_x100"] - 100 * (0.8 + 0.6 + 0.0 + 0.9) / 4) < 1e-9 and abs(sq["best_of_64_x100"] - 100 * (0.8 + 0.6 + 0.7 + 0.9) / 4) < 1e-9, "random / actual / best-of-N")
    sm = SR.summarise_pick(pdms, sub, np.array([0, 0, 0, 0]))
    chk(sm["zero_picks"] == 2 and sm["dac0_picks"] == 2, "summarise_pick counts")


def test_confirm_compute_on_a_synthetic_table():
    rng = np.random.default_rng(5)
    N, M = 30, 16
    logs = np.array([f"log{i % 3}" for i in range(N)])
    cmd = np.array([i % 3 for i in range(N)])
    logits = rng.normal(0.5, 2.0, size=(N, M, 6))
    sub = np.stack([rng.choice([0.0, 0.5, 1.0], size=(N, M), p=[.1, .05, .85]), rng.choice([0.0, 1.0], size=(N, M), p=[.15, .85]), rng.uniform(0, 1, size=(N, M)),
                    rng.choice([0.0, 1.0], size=(N, M)), rng.choice([0.0, 1.0], size=(N, M), p=[.05, .95]), np.ones((N, M))], -1)
    pdms = RR.pdms_v1(sub[..., 0], sub[..., 1], sub[..., 2], sub[..., 3], sub[..., 4])
    P = RV.sigmoid(logits)
    table_pick = SR.picks(("B",), P)
    res, B, C = SR.confirm_compute(logs, cmd, pdms, sub, logits, ("V", 0.8), table_pick, n_boot=300)
    chk(res["n_tokens"] == N and res["n_logs"] == 3, "shape bookkeeping")
    chk(res["B_vs_table_pick"]["equal"] == N, "B reproduces the table pick when the table pick is B")
    chk(np.array_equal(C, SR.picks(("V", 0.8), P)), "the frozen rule is the one applied")
    chk(res["compare"]["picks_changed"] == int((B != C).sum()), "picks_changed bookkeeping")
    chk(abs(res["compare"]["PDMS"]["mean"] - 100 * (SR.at_pick(pdms, C) - SR.at_pick(pdms, B)).mean()) < 1e-9, "DeltaPDMS x100")
    chk(res["per_command"]["available"] and res["per_command"]["tokens_covered"] == N and set(res["per_command"]["by_command"]) == {"left", "straight", "right"}, "per-command splits cover every token")
    wsum = sum(v["tokens"] * v["delta_PDMS_x100"]["mean"] for v in res["per_command"]["by_command"].values()) / N
    chk(abs(wsum - res["compare"]["PDMS"]["mean"]) < 1e-9, "token-weighted per-command deltas sum to the overall delta")
    o = res["oracles"]["O-both"]
    chk(abs(o["share_of_SEL_ceiling_3.54"] - o["delta_vs_B"]["mean"] / 3.5376954888978354) < 1e-12, "oracle share of the +3.54 ceiling")
    chk(res["selection_quality"]["B"]["best_of_64_x100"] >= res["selection_quality"]["B"]["actual_x100"] - 1e-9, "best-of-N >= actual")
    res2, _, _ = SR.confirm_compute(logs, None, pdms, sub, logits, ("K", 3), None, n_boot=300)
    chk(res2["per_command"]["available"] is False and "B_vs_table_pick" not in res2, "no command field -> NOT COMPUTED, said so")
    resB, _, CB = SR.confirm_compute(logs, cmd, pdms, sub, logits, ("V", 0.0), None, n_boot=300)
    chk(resB["compare"]["PDMS"]["mean"] == 0.0 and resB["compare"]["PDMS"]["ci95"] == [0.0, 0.0] and resB["compare"]["picks_changed"] == 0, "SELF-PAIR: V(0) is B -> delta exactly 0 [0, 0], 0 picks changed")


MUTANTS = [
    ("M1 veto fallback removed", "        return int(b)  # VETO-FALLBACK", "        pass  # VETO-FALLBACK", ("V_fallback_is_B",)),
    ("M2 k applied to EP, not the gates", "    for c in GATE_COLS:\n        q[:, c] = q[:, c] ** k  # K-GATES", "    for c in (COL[\"EP\"],):\n        q[:, c] = q[:, c] ** k  # K-GATES", ("K_gates_known_S3", "K_gates_known_S1")),
    ("M3 tie-break inverted", "    return tied[0], tied  # TIE-RULE", "    return tied[-1], tied  # TIE-RULE", ("tie_goes_to_smaller_change_V_before_K", "tie_three_way_V_first")),
    ("M4 oracle uses the PREDICTED gate", "        q[:, c] = ts[:, c]  # ORACLE-TRUE", "        q[:, c] = q[:, c]  # ORACLE-TRUE", ("O_DAC_known", "O_NC_known", "O_both_known")),
]


def test_deliberate_regression_arms():
    for label, old, new, expect in MUTANTS:
        bad = behaviour_checks(load_mutant(old, new))
        chk(bad != [], f"{label}: the mutant must turn a check RED but every known answer still passed")
        chk(any(e in bad for e in expect), f"{label}: expected one of {expect} RED, got {bad}")
    # control: the unmutated module through the same loader is GREEN (the loader itself does not break anything)
    mod = types.ModuleType("selrule_ctrl")
    mod.__file__ = SRC
    exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
    chk(behaviour_checks(mod) == [], "control: the same loader on the UNMUTATED source is GREEN")


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"  ok  {n}", flush=True)
    print(f"ZZTEST_PASS {N_CHECKS} checks ({len(tests)} tests, {len(MUTANTS)} deliberate-regression arms RED)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
