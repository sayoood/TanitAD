#!/usr/bin/env python3
"""Tests for eval/hn_screen_read.py, hn_screen_offsets_score.py and hn_screen_families.py (P6-HN-S checklist items 5, 6, 7; PREREG_P6HN_SCREEN.md sections 5-11).
Run:  PYTHONIOENCODING=utf-8 python eval/test_hn_screen_read.py   ->   ZZTEST_PASS <n>   (prints the mutation table).   CPU only, SYNTHETIC fixtures only: no model, no checkpoint, no real scene.

THE FIXTURE.  400 sets in 20 logs, 7 proposals with known labels (good / alt / DAC-zero / NC-zero / NC-half / TTC-poor / EP-poor); an arm's behaviour is a pick per set (forced by its logits), so every statistic
the read computes has a hand-checkable value.  Executed plans make proposals 0 and 2 near-twins (DAC differs), so M2 is defined; a 200-scene bank (anchor + 8 offsets, 5 DAC = 1 members vs 4 DAC = 0 offsets) defines M3.
WHAT IS PROVEN:
  O  EVERY section-11 row is reached end to end (0 INVALID, 1 SPARSE, 2 UNSAFE, 3 GO, 3 via rule E, 4 NET-ZERO, 5 SEED-FRAGILE, 6 REVERSED, 7 HEAD-MOVES-NO-ZEROS, 8 SYNTHETIC-ONLY, 9 HEAD-BLIND, 8-or-9 unresolved,
     PENDING rule E), a mixed-sign effect is NOT 'reversed', a missing gate is a failed gate, and the boundaries hold: -dz-bar exactly at F_z is NOT ZEROS-FALL (one pick more is), 2 of 3 adverse seeds fail the co-gate (1 does not),
     a Delta PDMS of exactly 0 is NET-ZERO, SY exactly 0.10 is not sparse, Z2 needs EVERY seed, rule E needs 4 of 5.
  J  ONE log draw per resample is applied jointly: identical per-seed differences give a seed-mean interval EQUAL to each seed's; a single-quantity interval equals proxy_eval.boot's on the same sets; the draws are seeded 20260927.
  T  ties go to the LOWEST index (rl_val_eval's rule); the lookup's PDMS / zero / split are the hand values.
  I  IC: an arm minus itself reads exactly 0 with CI [0, 0] through the same functions (and a biased interval turns it RED).
  S  scoring (item 5): the offsets are scored by the SAME masked call the training term uses, equal to scoring each slot alone, deterministic, refused on a cache of another trunk.
  F  families (item 6), with the REAL taniteval adapter on synthetic windows: all four families present or stated unavailable with reason and n, STRATEGIC stamped PAUSED, watch-items by the landed fam_adverse rule.
  C  the CLI: files in, a result json out (the GO scenario), with the pause stamp and the estimator named; and the same with the families' per-set export (published only when paired and IC holds).
  P  the trajectory-level families, PAIRED (section 10 + amendment B4): on a fixture whose between-log variation swamps the arm difference, a shift the single-arm (unpaired) intervals cannot see is sharp on the paired
     draws; an arm minus itself reads exactly 0 with CI [0, 0] for every component; the landed direction table is read on the paired interval (bias: |B| - |A|; progress ratio: |B - 1| - |A - 1|); the masked
     components are ratios of sums, rms and median and the tactical accuracy / kappa are re-taken on each resample (equal to the EXPANDED resample, drawn by hand); kappa undefined is never 1.0; the export is paired by
     (log, token) and a shuffled export reads identically, a different set / a duplicate / a missing arm is refused loudly; watch-items are reported and never move the outcome row; families are published only when
     paired and IC holds.  F (items 5-6 file) also covers the REAL adapter's per-set export: 21 components per arm, a lossless npz, blocks identical with the export on and off.
Every mutation below edits a SOURCE file (each snippet must occur EXACTLY once) and must turn >= 1 expected check RED.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
for _p in (HERE, os.path.join(PKG, "refe")):
    sys.path.insert(0, _p)
import rl_val_eval as RV  # noqa: E402
import proxy_eval as PE  # noqa: E402

SRC_READ = os.path.join(HERE, "hn_screen_read.py")
NB = 2000                                                              # bootstrap resamples in the fixtures (the production default is 10,000; the J checks run 10,000)
N, NLOG, PERLOG, M = 400, 20, 20, 7
PROP = np.array([[1, 1, .9, 1, 1, 1],       # 0 good           PDMS .958333
                 [1, 1, .8, 1, 1, 1],       # 1 alt            PDMS .916667
                 [1, 0, .9, 1, 1, 1],       # 2 DAC-zero       PDMS 0
                 [0, 1, .9, 1, 1, 1],       # 3 NC-zero        PDMS 0
                 [.5, 1, .9, 1, 1, 1],      # 4 NC-half        PDMS .479167
                 [1, 1, .9, 0, 1, 1],       # 5 TTC-poor       PDMS .541667
                 [1, 1, 0, 1, 1, 1]], float)  # 6 EP-poor      PDMS .583333
GATES_OK = {k: True for k in ("I1", "I2", "I3", "I4", "I5", "I6", "D1", "G14", "G15", "launch_gate")}
SY_OK = {"share_with_hn": 0.3, "sparse": False}


def load_read(src, tag):
    mod = types.ModuleType(f"hn_screen_read_{tag}")
    mod.__file__ = SRC_READ
    exec(compile(src, f"hn_screen_read_{tag}.py", "exec"), mod.__dict__)    # noqa: S102 -- the module under test, possibly mutated
    return mod


# ------------------------------------------------------------------------------------------------ fixture builders
def spread(n, exclude=()):
    """n set indices spread round-robin across the 20 logs (nested: spread(k) is a prefix of spread(n) for k < n)"""
    seq = [(j % NLOG) * PERLOG + (j // NLOG) for j in range(N)]
    ex = set(exclude)
    return [i for i in seq if i not in ex][:n]


def scores_from(picks, dac_over=None):
    s = np.full((N, M, 7), -10.0)
    s[np.arange(N), picks, :] = 10.0
    for prop, vec in (dac_over or {}).items():
        for i in range(N):
            if picks[i] != prop:
                s[i, prop, 1] = vec[i] if hasattr(vec, "__len__") else vec
    return s


def picks_with(base=0, **moves):
    p = np.full(N, base)
    for prop, idx in moves.items():
        p[list(idx)] = int(prop[1:])
    return p


def plans():
    P = np.zeros((N, M, 8, 3))
    for j in range(M):
        P[:, j, :, 1] = 10.0 * j
        P[:, j, :, 0] = np.linspace(2, 16, 8)
    P[:, 2, :, 1] = 0.2                                               # proposal 2 is a near-twin of proposal 0 (0.2 m) with DAC differing
    return P


def bank(n_q=200):
    dac9 = np.zeros((n_q, 9))
    dac9[:, :5] = 1.0                                                  # anchor + offsets 1-4: DAC = 1
    logs = [f"log{i % NLOG}" for i in range(n_q)]
    return {"dac9": dac9, "status": np.zeros(n_q, np.int8), "logs": logs}


def scores9(n_q, informative):
    s = np.zeros((n_q, 9, 7))
    if informative:
        s[:, :5, 1] = 2.0
        s[:, 5:, 1] = -2.0
    return s


def build(H, arm_picks, dac_over=None, with_plans=True, bank_qual=None, b_informative=None, a_informative=False):
    """arm_picks {name: picks [N]}; names base, A0.., B0.. (and C0)"""
    keys = [(f"log{i // PERLOG}", f"t{i}", 0, 0) for i in range(N)]
    arms = {n: scores_from(p, (dac_over or {}).get(n)) for n, p in arm_picks.items()}
    tg = np.tile(PROP, (N, 1, 1))
    bk = arms9 = None
    if bank_qual is not None:
        bk = bank(200)
        if bank_qual < 200:
            bk["dac9"][bank_qual:, :] = 1.0                            # no DAC = 0 offset: the scene has no pair
        arms9 = {n: scores9(200, (n[0] == "B" and b_informative) or (n[0] == "A" and a_informative)) for n in arm_picks if n[0] in "AB"}
    return H.make_data(keys, [k[0] for k in keys], tg, arms, plans() if with_plans else None, bk, arms9)


def run(H, **kw):
    D = build(H, kw.pop("picks"), **{k: kw.pop(k) for k in ("dac_over", "with_plans", "bank_qual", "b_informative", "a_informative") if k in kw})
    return H.read(D, "navsim_v1_lane", kw.pop("gates", GATES_OK), sy=kw.pop("sy", SY_OK), n_boot=NB)


def zero_picks(nA, nB, seeds=(0, 1, 2), extra_b=None, extra_a=None, base_other=0):
    """A_s picks the DAC-zero proposal on spread(nA[s]) sets, B_s on spread(nB[s]); everything else picks prop0.  extra_b/extra_a: {s: (prop, idx)} further picks"""
    p = {"base": np.zeros(N, int)}
    for k, s in enumerate(seeds):
        p[f"A{s}"] = picks_with(0, p2=spread(nA[k]))
        p[f"B{s}"] = picks_with(0, p2=spread(nB[k]))
        if extra_b:
            prop, idx = extra_b
            p[f"B{s}"][list(idx)] = prop
        if extra_a:
            prop, idx = extra_a
            p[f"A{s}"][list(idx)] = prop
    return p


def all_prop1(seeds=(0, 1, 2)):
    p = {"base": np.full(N, 1)}
    for s in seeds:
        p[f"A{s}"] = np.full(N, 1)
        p[f"B{s}"] = np.full(N, 1)
    return p


def twin_logits(b_better):
    """DAC logits of the non-picked props 0 and 2: A ties, B ranks prop 0 above prop 2 (when b_better)"""
    d = {}
    for s in (0, 1, 2):
        d[f"A{s}"] = {0: 1.0, 2: 1.0}
        d[f"B{s}"] = {0: 1.0, 2: -1.0 if b_better else 1.0}
    return d


# ================================================================================================ the suite
def suite(H, tag, only=None):
    bad: list = []
    sel = (lambda name: only is None or name in only)

    def check(name, fn):
        if not sel(name):
            return
        try:
            ok = bool(fn())
        except Exception as e:                                         # noqa: BLE001 -- an exception IS a failed check
            ok = False
            if os.environ.get("HSR_DEBUG"):
                print("  [check raised]", name, type(e).__name__, str(e)[:300])
        if not ok:
            bad.append(name)

    row = lambda r: r["outcome"]["row"]                                # noqa: E731
    z0 = lambda n: spread(n)                                           # noqa: E731
    # ------------------------------------------------------------------ O: every section-11 row
    GO_P = zero_picks((40, 40, 40), (0, 0, 0))
    check("O_row3_GO", lambda: row(run(H, picks=GO_P)) == 3 and run(H, picks=GO_P)["zeros"]["state"] == "ZEROS-FALL")
    check("O_row0_INVALID_gate_fails", lambda: row(run(H, picks=GO_P, gates={**GATES_OK, "I1": False})) == 0)
    check("O_row0_a_missing_gate_is_a_failed_gate", lambda: row(run(H, picks=GO_P, gates={k: v for k, v in GATES_OK.items() if k != "G15"})) == 0)
    check("O_row1_SPARSE", lambda: row(run(H, picks=GO_P, sy={"share_with_hn": 0.05, "sparse": True})) == 1)
    check("O_SY_exactly_0.10_is_not_sparse", lambda: row(run(H, picks=GO_P, sy={"share_with_hn": 0.10, "sparse": False})) == 3)
    unsafe = zero_picks((40, 40, 40), (0, 0, 0), extra_b=(5, spread(60, exclude=spread(40))))
    check("O_row2_UNSAFE_even_when_zeros_fall", lambda: (lambda r: row(r) == 2 and r["zeros"]["zeros_fall"] and r["cogate"]["TTC"]["ci95"][1] < 0)(run(H, picks=unsafe)))
    netzero = zero_picks((40, 40, 40), (0, 0, 0), extra_b=(6, spread(150, exclude=spread(40))))
    check("O_row4_NET_ZERO", lambda: (lambda r: row(r) == 4 and r["delta_pdms"]["mean"] < 0 and r["zeros"]["zeros_fall"])(run(H, picks=netzero)))
    nA5, nB_fragile = (60, 40, 60, 50, 60), (10, 5, 60, 50, 10)            # Z1 holds at seeds 0-2; seed 2 has delta 0 (Z2 fails); 3 of 5 negative after the extension (Z2' needs 4); Z3 holds
    frag = zero_picks(nA5, nB_fragile, seeds=(0, 1, 2, 3, 4))
    check("O_row5_SEED_FRAGILE_rule_E_fired_and_fails", lambda: (lambda r: row(r) == 5 and r["zeros"]["rule_e_fired"] and r["zeros"]["primary_5seed"]["Z3"] and not r["zeros"]["primary_5seed"]["Z2_prime_ge4of5"])(run(H, picks=frag)))
    nB_ok = (10, 5, 60, 5, 10)                                            # 4 of 5 negative, -dsum = 50+35+0+45+50 = 180 > 5 x 20
    ok5 = zero_picks(nA5, nB_ok, seeds=(0, 1, 2, 3, 4))
    check("O_row3_via_rule_E", lambda: (lambda r: row(r) == 3 and r["zeros"]["rule_e_fired"] and r["zeros"]["state"] == "ZEROS-FALL" and len(r["zeros"]["seeds_used"]) == 5)(run(H, picks=ok5)))
    pend = zero_picks(nA5[:3], nB_fragile[:3])
    check("O_pending_rule_E_when_seeds_3_4_are_missing", lambda: (lambda r: row(r) is None and r["zeros"]["rule_e_required"])(run(H, picks=pend)))
    check("O_rule_E_seeds_ignored_when_not_triggered", lambda: (lambda r: r["zeros"]["extension_ignored"] and not r["zeros"]["rule_e_fired"] and len(r["zeros"]["seeds_used"]) == 3)(
        run(H, picks={**zero_picks((40, 40, 40), (0, 0, 0)), "A3": np.zeros(N, int), "B3": np.zeros(N, int), "A4": np.zeros(N, int), "B4": np.zeros(N, int)})))
    rev = zero_picks((10, 10, 10), (60, 60, 60))
    check("O_row6_REVERSED", lambda: row(run(H, picks=rev)) == 6)
    mixed = {"base": np.zeros(N, int), **{f"{a}{s}": np.zeros(N, int) for s in (0, 1, 2) for a in "AB"}}
    for s in (0, 1, 2):
        mixed[f"A{s}"] = picks_with(0, p2=[i for i in range(N) if i // PERLOG == 0][:10])        # A's zeros in log 0
        mixed[f"B{s}"] = picks_with(0, p2=[i for i in range(N) if i // PERLOG == 1][:10])        # B's zeros in log 1 (mean difference 0, signs opposite)
    check("O_mixed_sign_effect_is_not_reversed", lambda: (lambda r: row(r) != 6 and not r["zeros"]["final"]["LB_positive"] and r["zeros"]["final"]["delta_z_bar"]["ci95"][1] > 0)(
        run(H, picks=mixed, with_plans=True, bank_qual=200, b_informative=False)))
    check("O_row7_HEAD_MOVES_NO_ZEROS", lambda: (lambda r: row(r) == 7 and r["M2"]["HEAD_MOVES"] and r["zeros"]["final"]["delta_z_bar"]["mean"] == 0.0)(
        run(H, picks=all_prop1(), dac_over=twin_logits(True), bank_qual=200)))
    check("O_row8_SYNTHETIC_ONLY", lambda: (lambda r: row(r) == 8 and not r["M2"]["HEAD_MOVES"] and r["M3"]["LEARNS_BANK"])(
        run(H, picks=all_prop1(), dac_over=twin_logits(False), bank_qual=200, b_informative=True)))
    check("O_row9_HEAD_BLIND", lambda: (lambda r: row(r) == 9 and not r["M2"]["HEAD_MOVES"] and not r["M3"]["LEARNS_BANK"])(
        run(H, picks=all_prop1(), dac_over=twin_logits(False), bank_qual=200, b_informative=False)))
    check("O_row8or9_unresolved_when_M3_not_computable", lambda: (lambda r: row(r) == "8or9" and r["M3"]["status"] == "NOT COMPUTABLE" and r["M3"]["n_scenes_qualifying"] == 60)(
        run(H, picks=all_prop1(), dac_over=twin_logits(False), bank_qual=60, b_informative=True)))
    one_log = {"base": np.zeros(N, int)}
    for s in (0, 1, 2):
        one_log[f"A{s}"] = picks_with(0, p2=list(range(20)))             # the whole effect sits in ONE log
        one_log[f"B{s}"] = np.zeros(N, int)
    check("O_Z1_fails_when_the_effect_sits_in_one_log_no_extension", lambda: (lambda r: r["zeros"]["state"] == "Z1-FAILS-NO-EXTENSION" and not r["zeros"]["rule_e_required"] and row(r) in (7, 8, 9))(
        run(H, picks=one_log, bank_qual=200)))
    # boundaries (counts): -dsum == S x spread is NOT ZEROS-FALL; one more pick is
    at = zero_picks((40, 30, 50), (20, 10, 30))
    over = zero_picks((40, 30, 50), (19, 10, 30))
    check("O_boundary_dz_exactly_at_Fz_is_not_ZEROS_FALL", lambda: (lambda r: (not r["zeros"]["primary_3seed"]["Z3"]) and r["zeros"]["primary_3seed"]["Z1"] and r["zeros"]["rule_e_required"])(run(H, picks=at)))
    check("O_boundary_one_pick_more_is_ZEROS_FALL", lambda: (lambda r: r["zeros"]["primary_3seed"]["Z3"] and r["zeros"]["zeros_fall"])(run(H, picks=over)))
    z2 = zero_picks((40, 40, 40), (0, 0, 40))
    check("O_Z2_needs_every_seed", lambda: (lambda r: r["zeros"]["primary_3seed"]["Z1"] and not r["zeros"]["primary_3seed"]["Z2"] and not r["zeros"]["zeros_fall"])(run(H, picks=z2)))
    # pure classifier / co-gate boundaries
    cg = H.cogate_decision
    check("O_cogate_2_of_3_adverse_seeds_fail_1_does_not", lambda: cg(0.1, [-1, -1, 1])["fails"] and not cg(0.1, [-1, 1, 1])["fails"] and cg(-1e-9, [1, 1, 1])["fails"] and not cg(0.0, [1, 1, 1])["fails"])
    zf = {"zeros_fall": True, "rule_e_required": False, "rule_e_fired": False, "state": "ZEROS-FALL"}
    base_in = {"gates": {**GATES_OK, "IC": True}, "sy_sparse": False, "cogate_fails": False, "zeros": zf, "dpdms_point": 0.0, "lb_dz_positive": False, "head_moves": False, "learns_bank": None}
    check("O_dPDMS_exactly_zero_is_NET_ZERO", lambda: H.classify(base_in) == 4 and H.classify({**base_in, "dpdms_point": 1e-12}) == 3)
    check("O_first_matching_row_wins", lambda: H.classify({**base_in, "gates": {**base_in["gates"], "G14": False}, "sy_sparse": True, "cogate_fails": True}) == 0
          and H.classify({**base_in, "sy_sparse": True, "cogate_fails": True}) == 1 and H.classify({**base_in, "cogate_fails": True}) == 2)
    # ------------------------------------------------------------------ J: the joint draws
    def joint():
        logs = [f"log{i // PERLOG}" for i in range(N)]
        JB = H.JointBoot(logs, 10000, 20260927)
        rng = np.random.default_rng(1)
        v = rng.normal(size=N)
        pe = PE.boot({str(i): float(v[i]) for i in range(N)}, {str(i): logs[i] for i in range(N)}, n=10000, seed=20260927)
        mine = JB.mean_interval(v)
        same_pe = abs(mine["mean"] - pe[0]) < 1e-12 and abs(mine["ci95"][0] - pe[1]) < 1e-9 and abs(mine["ci95"][1] - pe[2]) < 1e-9
        d = rng.normal(size=N)
        sm = H.seedmean_diff(JB, [d + 0.0] * 3, [np.zeros(N)] * 3)                   # three seeds with IDENTICAL differences
        one = H.seedmean_diff(JB, [d], [np.zeros(N)])
        close = lambda a, b: bool(np.allclose(a, b, rtol=0, atol=1e-12))                      # noqa: E731 -- (x + x + x) / 3 need not be x to the last bit
        return same_pe and close(sm["ci95"], one["ci95"]) and all(close(c, one["ci95"]) for c in sm["per_seed_ci95"]) and H.SEED == 20260927 and H.N_BOOT == 10000

    check("J_joint_draws_equal_proxy_eval_boot_and_perfectly_correlated_seeds_keep_their_interval", joint)

    # ------------------------------------------------------------------ T: ties to the lowest index, the lookup's hand values
    def ties():
        D = H.make_data([("log0", "t0", 0, 0)], ["log0"], PROP[None], {"A0": np.zeros((1, M, 7))})
        lk = H.lookup(D, "A0", "navsim_v1_lane")
        D2 = H.make_data([("log0", "t0", 0, 0)], ["log0"], PROP[None], {"A0": scores_from(np.array([2] * N), None)[:1]})
        l2 = H.lookup(D2, "A0", "navsim_v1_lane")
        return (int(lk["pick"][0]) == 0 and abs(lk["PDMS"][0] - 11.5 / 12) < 1e-12 and l2["zero"][0] == 1.0 and l2["zero_dac_only"][0] == 1.0 and l2["zero_nc_only"][0] == 0.0
                and abs(lk["rand"][0] - np.mean([11.5 / 12, 11 / 12, 0, 0, .5 * 11.5 / 12, 6.5 / 12, 7 / 12])) < 1e-12 and abs(lk["best"][0] - 11.5 / 12) < 1e-12)

    check("T_ties_go_to_the_lowest_index_and_lookup_hand_values", ties)

    # ------------------------------------------------------------------ I: IC
    check("I_arm_minus_itself_reads_exactly_zero", lambda: (lambda r: r["IC"]["ok"] and r["IC"]["n_statistics"] >= 9)(run(H, picks=GO_P, dac_over=twin_logits(True), bank_qual=200, b_informative=True)))
    check("I_ic_failure_makes_the_screen_invalid", lambda: row(run(H, picks=GO_P)) == 3)       # the positive control; the mutation makes the interval biased -> IC false -> row 0
    return bad


# ================================================================================================ P: the trajectory-level families, PAIRED (section 10 + amendment B4)
PSEEDS = (0, 1, 2)
P_SHIFT = 0.15


def _fx():
    """the per-set values of the paired-families fixture, in CANONICAL set order (t0..t399; log = i // 20).  Between-LOG variation is large (sd 3) and the arm-to-arm differences are tiny: the case where the
    single-arm intervals overlap (an unpaired read sees nothing) while the paired difference is sharp."""
    rng = np.random.default_rng(7)
    lo = np.repeat(rng.normal(0, 3, NLOG), PERLOG)
    base = lo + rng.normal(0, 0.1, N)
    cnt = rng.integers(0, 20, N).astype(float)
    cnt[:5] = 0.0                                                        # sets with no valid step in the masked component
    member = rng.random(N) < 0.8                                         # sets that carry the progress components
    gtc = rng.integers(0, 3, N)
    fx = {"lo": lo, "base": base, "cnt": cnt, "member": member, "gt3": gtc, "gt5": rng.integers(0, 5, N), "flipA": rng.random((len(PSEEDS), N)), "flipB": rng.random((len(PSEEDS), N)),
          "noise": {(a, s): rng.normal(0, 0.02, (4, N)) for a in "AB" for s in PSEEDS}}
    return fx


def _perm_rows(perm):
    return perm if perm is not None else np.arange(N)


def pexport(fx, arm, perm=None, shifts=None, drop_token=None):
    """the per-set export of one arm (the dict shape `taniteval.per_window.export` returns), rows in the order `perm` (canonical index of each exported row).  shifts: {component: additive B shift}."""
    a, s = arm[0], int(arm[1:])
    perm = _perm_rows(perm)
    sh = {"cross_mae_m": P_SHIFT, "target_speed_acc.within_1.0_mps": -0.1, "along_bias_m": 0.3, "cross_bias_m": -0.3, "ego_progress.progress_ratio_mean": 0.1,
          "ego_progress.progress_ratio_median": -0.1, "heading_mae_deg": 0.5, "speed_rmse_mps": 0.0, "goal_point_error_m": 0.2}
    sh.setdefault("curvature_bias_1pm", -0.3)
    sh.update(shifts or {})
    isB = a == "B"
    nz = fx["noise"][(a, s)]
    b = fx["base"]
    lo = fx["lo"]

    def vals(name, center, spread):                                      # an arm's per-set values: a centre, a log-level spread, independent per-arm noise, B's shift
        return center + spread * lo + nz[sum(map(ord, name)) % 4] + (sh[name] if isB else 0.0)

    mean = lambda name, center, spread: {"kind": "mean", "dp": 4, "value": vals(name, center, spread)[perm]}   # noqa: E731
    ratio_den = np.where(fx["member"], 1.0, 0.0)
    prog = 1.0 + 0.002 * lo + nz[0] * 0.2
    comp_lon = {
        "speed_rmse_mps": {"kind": "rms", "dp": 4, "value": ((b + 6.0 + fx["noise"][("A", s)][1]) ** 2)[perm]},                       # identical for A and B: the no-effect control (lower is better)
        "target_speed_acc.within_1.0_mps": mean("target_speed_acc.within_1.0_mps", 5.0, 1.0),                           # higher is better; B is 0.1 LOWER -> adverse
        "along_bias_m": mean("along_bias_m", 0.5, 0.02),                                                                # bias; A is ~0.5, B is ~0.3 further from 0 -> adverse
        "ego_progress.progress_ratio_mean": {"kind": "ratio", "dp": 4, "scale": 1.0, "den": ratio_den[perm],
                                             "num": (np.where(fx["member"], prog + (sh["ego_progress.progress_ratio_mean"] if isB else 0.0), 0.0))[perm]},   # closer to 1; A ~1.0, B ~1.1 -> adverse
        "ego_progress.progress_ratio_median": {"kind": "median", "dp": 4, "rows": np.flatnonzero(fx["member"][perm]),
                                               "value": (prog + (sh["ego_progress.progress_ratio_median"] if isB else 0.0))[perm][fx["member"][perm]]},          # A ~1.0, B ~0.9 -> adverse
        "ego_progress.gt_progress_mean_m": {"kind": "ratio", "dp": 4, "scale": 1.0, "human_only": True, "den": ratio_den[perm], "num": (ratio_den * 25.0)[perm]},      # a property of the human: identical, excluded
    }
    comp_lat = {
        "cross_mae_m": mean("cross_mae_m", 4.0, 1.0),                                                                  # lower is better; B is 0.15 HIGHER -> adverse
        "curvature_bias_1pm": mean("curvature_bias_1pm", -0.5, 0.02),                                              # bias A ~ -0.5, B ~ -0.8: FARTHER from zero though the signed shift is NEGATIVE -> adverse by |B| - |A| only
        "cross_bias_m": mean("cross_bias_m", 0.5, 0.02),                                                                # bias moving TOWARD zero (0.5 -> 0.2): |B| < |A| -> NOT adverse although the signed shift separates
        "heading_mae_deg": {"kind": "ratio", "dp": 4, "scale": 1.0, "den": fx["cnt"][perm], "num": (fx["cnt"] * (np.abs(b) + 1.0 + (sh["heading_mae_deg"] if isB else 0.0)))[perm]},
    }
    flip = (fx["flipB"] if isB else fx["flipA"])[s]
    thr = 0.30 if isB else 0.20

    def dec(gt, k, pred=None):
        p = np.where(flip < thr, (gt + 1) % k, gt) if pred is None else pred
        return {"classes": [f"c{j}" for j in range(k)], "gt": gt[perm], "pred": p[perm]}

    const = np.ones(N, int)
    tact = {"status": "OK", "lateral_decision": dec(fx["gt3"], 3), "longitudinal_decision": dec(const, 3, pred=const), "maneuver_5way_collapsed": dec(fx["gt5"], 5),
            "goal_setting": {"goal_point_error_m": {"kind": "mean", "dp": 4, "value": vals("goal_point_error_m", 2.0, 1.0)[perm]}}}
    toks = np.array([f"t{i}" for i in range(N)])[perm]
    logs = np.array([f"log{i // PERLOG}" for i in range(N)])[perm]
    if drop_token is not None:
        toks = toks.copy()
        toks[drop_token] = "t_renamed"
    return {"schema": "taniteval.per_window/1", "n": N, "status": "OK", "grid": "dense", "dt_s": 0.5, "tokens": [str(t) for t in toks], "eid": [str(x) for x in logs], "longitudinal": comp_lon,
            "lateral": comp_lat, "tactical": tact, "unavailable": {"longitudinal.distance_keeping.mean_headway_min_m": {"status": "UNAVAILABLE", "reason": "no lead", "n": N}},
            "strategic": {"status": "PAUSED"}}


def pexports(fx, perm=None, **kw):
    return {f"{a}{s}": pexport(fx, f"{a}{s}", perm, **kw) for s in PSEEDS for a in "AB"}


def brute_draw(JB, b, v):
    """the mean of the EXPANDED resample of draw b (each log drawn counts[b, g] times)"""
    w = JB.counts[b][JB.li]
    return float((w * v).sum() / w.sum())


def suite_paired(H, tag, only=None):
    bad: list = []
    sel = (lambda name: only is None or name in only)

    def check(name, fn):
        if not sel(name):
            return
        try:
            ok = bool(fn())
        except Exception as e:                                         # noqa: BLE001
            ok = False
            if os.environ.get("HSR_DEBUG"):
                print("  [check raised]", name, type(e).__name__, str(e)[:300])
        if not ok:
            bad.append(name)

    import proxy_eval as PE_

    def safe(fn):                                                      # a mutated module may raise inside the call itself: that is a RED check, not a crashed suite
        try:
            return fn()
        except Exception as e:                                         # noqa: BLE001
            return e

    fx = _fx()
    GO_P = zero_picks((40, 40, 40), (0, 0, 0))
    D = build(H, GO_P)
    JB = H.JointBoot(D["logs"], NB, 20260927)
    ex = pexports(fx)
    FP = safe(lambda: H.families_paired(JB, D, ex, PSEEDS))
    L, T = (FP.get("LONGITUDINAL", {}), FP.get("LATERAL", {})) if isinstance(FP, dict) else ({}, {})

    def fx_arm_values(fx_, arm, comp):                                                             # the exported value array of an arm's component, canonical order
        e = ex[arm]
        return e["lateral"][comp]["value"] if comp in e["lateral"] else e["longitudinal"][comp]["value"]

    # ---- the pairing is REAL: a shift the unpaired single-arm intervals cannot see is sharp on the paired draws
    def pairing_real():
        a_mean = np.mean([fx_arm_values(fx, f"A{s}", "cross_mae_m") for s in PSEEDS], axis=0)
        b_mean = np.mean([fx_arm_values(fx, f"B{s}", "cross_mae_m") for s in PSEEDS], axis=0)
        ia, ib = JB.mean_interval(a_mean), JB.mean_interval(b_mean)
        unpaired_sees = ib["ci95"][0] > ia["ci95"][1]                                               # the landed unpaired rule for a lower-is-better component: B's interval entirely above A's
        r = T["cross_mae_m"]
        width_u = ia["ci95"][1] - ia["ci95"][0]
        return (not unpaired_sees) and r["adverse"] and r["ci95"][0] > 0 and (r["ci95"][1] - r["ci95"][0]) < 0.05 * width_u and abs(r["mean"] - P_SHIFT) < 0.01

    check("P_pairing_is_real_a_shift_the_unpaired_intervals_miss_is_sharp_on_the_paired_draws", pairing_real)
    # ---- identity
    check("P_family_IC_an_arm_minus_itself_is_exactly_zero_with_CI_0_0_for_every_component", lambda: FP["IC"]["ok"] and FP["IC"]["n_statistics"] >= 12)
    # ---- directions (the landed proxy_eval.fam_direction table, read on the paired interval)
    def directions():
        return (L["target_speed_acc.within_1.0_mps"]["direction"] == "higher" and L["target_speed_acc.within_1.0_mps"]["adverse"] and L["target_speed_acc.within_1.0_mps"]["ci95"][1] < 0
                and T["cross_mae_m"]["direction"] == "lower" and T["cross_mae_m"]["adverse"]
                and not L["speed_rmse_mps"]["adverse"] and L["speed_rmse_mps"]["mean"] == 0.0
                and all(PE_.fam_direction(k) == v["direction"] for d in (L, T) for k, v in d.items()))

    check("P_directions_follow_the_landed_table_higher_UB_below_0_lower_LB_above_0", directions)
    check("P_bias_is_adverse_only_when_B_is_farther_from_zero_and_progress_ratio_when_farther_from_1",
          lambda: L["along_bias_m"]["adverse"] and L["along_bias_m"]["adverse_stat"]["mean"] > 0.2 and T["curvature_bias_1pm"]["adverse"] and T["curvature_bias_1pm"]["ci95"][1] < 0 and T["curvature_bias_1pm"]["adverse_stat"]["ci95"][0] > 0 and not T["cross_bias_m"]["adverse"] and T["cross_bias_m"]["ci95"][1] < 0 and T["cross_bias_m"]["adverse_stat"]["ci95"][1] < 0
          and L["ego_progress.progress_ratio_mean"]["adverse"] and L["ego_progress.progress_ratio_median"]["adverse"])
    check("P_masked_ratio_components_and_empty_denominator_sets_are_ratios_of_sums", lambda: T["heading_mae_deg"]["adverse"] and abs(T["heading_mae_deg"]["mean"] - 0.5) < 0.02 and (fx["cnt"] == 0).sum() >= 5)

    def ratio_draws():
        c = ex["A0"]["lateral"]["heading_mae_deg"]
        pt, dr = H.comp_draws(JB, c)
        ok = abs(pt - c["num"].sum() / c["den"].sum()) < 1e-12
        for b in (0, 17, 1999):
            w = JB.counts[b][JB.li]
            ok = ok and abs(dr[b] - (w * c["num"]).sum() / (w * c["den"]).sum()) < 1e-12
        return ok

    check("P_masked_ratio_draws_equal_the_expanded_resample_ratio", ratio_draws)

    def rms_draws():
        c = ex["A0"]["longitudinal"]["speed_rmse_mps"]
        pt, dr = H.comp_draws(JB, c)
        ok = abs(pt - np.sqrt(c["value"].mean())) < 1e-12
        for b in (0, 5, 1999):
            ok = ok and abs(dr[b] - np.sqrt(brute_draw(JB, b, c["value"]))) < 1e-12
        return ok

    check("P_rms_draws_equal_the_root_of_the_expanded_mean_square", rms_draws)

    def median_draws():
        c = ex["A0"]["longitudinal"]["ego_progress.progress_ratio_median"]
        pt, dr = H.comp_draws(JB, c)
        ok = abs(pt - np.median(c["value"])) < 1e-12
        li_rows = JB.li[c["rows"]]
        for b in tuple(range(12)) + (500, 1999):
            exp_ = np.repeat(c["value"], JB.counts[b][li_rows].astype(int))
            ok = ok and abs(dr[b] - np.median(exp_)) < 1e-12
        return ok

    check("P_median_draws_equal_the_median_of_the_expanded_resample", median_draws)
    # ---- tactical: accuracy and kappa from the draw's confusion table
    def kappa_draws():
        e = ex["A0"]["tactical"]["lateral_decision"]
        (a0, ad), (k0, kd) = H.decision_draws(JB, e["gt"], e["pred"], 3)

        def cohen(g, p, k=3):
            po = float((g == p).mean())
            pe = float(sum((g == c).mean() * (p == c).mean() for c in range(k)))
            return po, (po - pe) / (1 - pe)

        po, ka = cohen(e["gt"], e["pred"])
        ok = abs(a0 - po) < 1e-12 and abs(k0 - ka) < 1e-12
        for b in (0, 9, 1234):
            w = JB.counts[b][JB.li].astype(int)
            g, p = np.repeat(e["gt"], w), np.repeat(e["pred"], w)
            pb, kb = cohen(g, p)
            ok = ok and abs(ad[b] - pb) < 1e-12 and abs(kd[b] - kb) < 1e-12
        return ok

    check("P_tactical_accuracy_and_kappa_draws_equal_the_expanded_resample", kappa_draws)
    tac = FP.get("TACTICAL", {}) if isinstance(FP, dict) else {}
    check("P_kappa_undefined_is_not_computable_never_1", lambda: tac["longitudinal_decision"]["kappa"]["status"] == "NOT COMPUTABLE" and tac["longitudinal_decision"]["accuracy"]["mean"] == 0.0
          and not np.isfinite(H.decision_draws(JB, np.ones(N, int), np.ones(N, int), 3)[1][0]))
    check("P_tactical_adverse_is_reported_beside_and_is_not_a_watch_item",
          lambda: "lateral_decision.accuracy" in tac["adverse_reported"] and "goal_setting.goal_point_error_m" in tac["adverse_reported"] and not any("decision" in w for w in FP["watch_items"]))
    # ---- the watch-items, human-only exclusion, the count
    exp_watch = ["LATERAL.cross_mae_m", "LATERAL.curvature_bias_1pm", "LATERAL.heading_mae_deg", "LONGITUDINAL.along_bias_m", "LONGITUDINAL.ego_progress.progress_ratio_mean", "LONGITUDINAL.ego_progress.progress_ratio_median",
                 "LONGITUDINAL.target_speed_acc.within_1.0_mps"]
    check("P_watch_items_are_the_adverse_longitudinal_and_lateral_components", lambda: FP["watch_items"] == exp_watch and FP["n_components_compared"] == 9)
    check("P_human_only_components_excluded", lambda: FP["excluded_human_only"] == ["longitudinal.ego_progress.gt_progress_mean_m"] and "ego_progress.gt_progress_mean_m" not in L)
    check("P_unavailable_components_are_listed_with_reason_and_n", lambda: FP["unavailable"]["longitudinal.distance_keeping.mean_headway_min_m"]["n"] == N and FP["STRATEGIC"]["status"] == "PAUSED")
    # ---- alignment by (log, token), never by position
    perm = np.random.default_rng(3).permutation(N)
    FPs = safe(lambda: H.families_paired(JB, D, pexports(fx, perm=perm), PSEEDS))
    same = lambda x, y: x["mean"] == y["mean"] and x["ci95"] == y["ci95"]                              # noqa: E731
    check("P_shuffled_export_pairs_identically", lambda: FPs["status"] == "OK" and all(same(FPs[f][k], FP[f][k]) for f in ("LONGITUDINAL", "LATERAL") for k in FP[f]) and FPs["watch_items"] == FP["watch_items"]
          and all(same(FPs["TACTICAL"][d][q], tac[d][q]) for d in ("lateral_decision", "maneuver_5way_collapsed") for q in ("accuracy", "kappa")))
    r_ren = safe(lambda: H.families_paired(JB, D, pexports(fx, drop_token=7), PSEEDS))
    dup = pexports(fx)
    dup["A0"]["tokens"], dup["A0"]["eid"] = list(dup["A0"]["tokens"]), list(dup["A0"]["eid"])
    dup["A0"]["tokens"][-1], dup["A0"]["eid"][-1] = dup["A0"]["tokens"][-2], dup["A0"]["eid"][-2]
    r_dup = safe(lambda: H.families_paired(JB, D, dup, PSEEDS))
    short = pexports(fx)
    r_short = safe(lambda: H.families_paired(JB, D, {k: v for k, v in short.items() if k != "B2"}, PSEEDS))
    check("P_different_sets_duplicates_and_missing_arms_are_refused_loudly", lambda: r_ren["status"] == "NOT PAIRABLE" and "not the same sets" in r_ren["reason"] and r_dup["status"] == "NOT PAIRABLE"
          and "duplicate" in r_dup["reason"] and r_short["status"] == "NOT COMPUTABLE" and "B2" in r_short["reason"])
    # ---- reported, not gating; published only when paired and IC holds
    def not_gating():
        r0 = H.read(D, "navsim_v1_lane", GATES_OK, sy=SY_OK, n_boot=NB)
        r1 = H.read(D, "navsim_v1_lane", GATES_OK, sy=SY_OK, n_boot=NB, per_set=ex)
        return (r0["outcome"]["row"] == r1["outcome"]["row"] == 3 and r1["outcome"] == r0["outcome"] and r1["watch_items_trajectory"] == exp_watch
                and set(r1["watch_items"]) == set(r1["watch_items_pdm"]) | set(exp_watch) and r1["families_paired"]["status"] == "OK" and "families_paired" not in r0 and r0["watch_items_trajectory"] is None)

    check("P_watch_items_reported_not_gating_the_outcome_row_is_unchanged", not_gating)
    def same_draws_as_pdm():
        r = H.read(D, "navsim_v1_lane", GATES_OK, sy=SY_OK, n_boot=NB, per_set=ex)
        fam_ = r["families_paired"]["LATERAL"]["cross_mae_m"]
        JBt = H.JointBoot(D["logs"], NB, 20260927)                                                     # the PDM sub-scores' own draws, built independently of the read
        want = H.seedmean_diff(JBt, [ex[f"B{s}"]["lateral"]["cross_mae_m"]["value"] for s in PSEEDS], [ex[f"A{s}"]["lateral"]["cross_mae_m"]["value"] for s in PSEEDS], scale=1.0)
        return (abs(fam_["mean"] - want["mean"]) < 1e-12 and np.allclose(fam_["ci95"], want["ci95"], rtol=0, atol=1e-12) and np.allclose(fam_["per_seed_ci95"], want["per_seed_ci95"], rtol=0, atol=1e-12)
                and np.allclose(fam_["per_seed_point"], want["per_seed_point"], rtol=0, atol=1e-12) and want["ci95"][0] != want["ci95"][1])

    check("P_family_components_are_read_on_the_SAME_joint_draws_as_the_PDM_sub_scores", same_draws_as_pdm)
    famjson = {"families_present": {k: {"status": "OK"} for k in ("longitudinal", "lateral", "tactical")} | {"strategic": {"status": "PAUSED"}}}

    def published():
        ok_ = {"status": "OK", "IC": {"ok": True}}
        return (H.families_status(famjson, ok_)["published"] is True and H.families_status(famjson, None)["published"] is False and "B4" in H.families_status(famjson, None)["reason"]
                and H.families_status(famjson, {"status": "OK", "IC": {"ok": False}})["published"] is False and H.families_status(famjson, {"status": "NOT PAIRABLE", "reason": "x"})["published"] is False
                and H.families_status(None, ok_)["published"] is False)

    check("P_families_published_only_when_paired_and_IC_holds", published)
    return bad


# ================================================================================================ items 5 and 6 (checked on the real modules, not mutated)
def suite_items56():
    bad = []

    def check(name, fn):
        try:
            ok = bool(fn())
        except Exception as e:                                         # noqa: BLE001
            ok = False
            if os.environ.get("HSR_DEBUG"):
                print("  [check raised]", name, type(e).__name__, str(e)[:300])
        if not ok:
            bad.append(name)

    import torch
    import hn_bank as HB
    import ctx_cache as CCH
    import hn_screen_offsets_score as OS
    import test_hn_bank as THB
    import test_ctx_cache as TCC
    tmp = tempfile.mkdtemp(prefix="hsr_")
    rows = THB.anchors_for(2, 4)
    bank_p = os.path.join(tmp, "val.hnbank")
    bk = HB.mint_bank(rows, bank_p, partition_id="val", fan_fingerprint="f", label_fn=THB.fake_label_fn)
    keys = [(r["log_name"], r["token"], 0, 0) for r in rows]
    units = [(f"log{li}", [k for k in keys if k[0] == f"log{li}"]) for li in range(2)]
    fm = TCC.Fake().eval()
    d = os.path.join(tmp, "cache")
    fp = CCH.trunk_fingerprint(fm.state_dict())
    CCH.build_units(fm, units, TCC.load_batch, TCC.make_forward(fm), d, trunk_fp=fp, log=lambda *_: None)
    CCH.finalize(d, units, trunk_fp=fp, calib_sha256="c")
    cache = CCH.CtxCache(d, expect_trunk_fingerprint=fp, verify=True)
    sc = THB.mk(0, w=8, heads=2, depth=2)
    gfan = torch.Generator().manual_seed(5)
    fan_cache = {k: torch.randn(64, 20, 3, generator=gfan) for k in keys}
    fan_provider = lambda ks: torch.stack([fan_cache[k] for k in ks])   # noqa: E731
    ks, s9 = OS.score_bank_offsets(sc, bk, cache, fan_provider, batch=3)
    n_lab = int((bk.status == HB.STATUS_LABELLED).sum())
    check("S_shape_labelled_scenes_only", lambda: s9.shape == (n_lab, 9, 7) and len(ks) == n_lab and n_lab == 6)
    ks2, s9b = OS.score_bank_offsets(sc, bk, cache, fan_provider, batch=1, autocast=False)
    ks3, s9c = OS.score_bank_offsets(sc, bk, cache, fan_provider, batch=3, autocast=False)
    check("S_deterministic_and_batch_independent", lambda: np.array_equal(s9, OS.score_bank_offsets(sc, bk, cache, fan_provider, batch=3)[1]) and float(np.abs(s9c - s9b).max()) < 1e-5)

    def equals_training_call():
        k0 = ks[0]
        i0 = [tuple(x[:4]) for x in bk.keys].index(k0)
        with torch.no_grad():
            direct = HB.hn_scores(sc.eval(), fan_provider([k0]), torch.from_numpy(bk.traj9[[i0]]), torch.ones(1, 9, dtype=torch.bool), cache.get_ctx([k0]).float())
            alone = torch.stack([HB.hn_scores(sc, fan_provider([k0]), torch.from_numpy(bk.traj9[[i0], j:j + 1]), torch.ones(1, 1, dtype=torch.bool), cache.get_ctx([k0]).float())[0, 0] for j in range(9)])
        return (float(np.abs(s9c[0] - direct[0].numpy()).max()) < 1e-5 and float((direct[0] - alone).abs().max()) < 1e-5            # fp32: the module's scores ARE the training call's; each slot equals scoring it alone
                and float(np.abs(s9[0] - direct[0].numpy()).max()) < 0.15)                                                         # bf16 autocast (the trainer's dtype) agrees to bf16 noise

    check("S_same_masked_call_as_training_and_each_slot_equals_scoring_it_alone", equals_training_call)

    def stale_refused():
        try:
            OS.score_bank_offsets(sc, bk, cache, fan_provider, expect_trunk_fingerprint="0" * 64)
            return False
        except CCH.StaleCache:
            return True

    check("S_cache_of_another_trunk_refused", stale_refused)
    p9 = os.path.join(tmp, "s9.npz")
    OS.write_scores9(p9, bk, ks, s9)
    b9, sx = OS.bank9_from(p9)
    check("S_scores9_file_round_trip_carries_dac_labels", lambda: b9["dac9"].shape == (n_lab, 9) and np.array_equal(sx, s9.astype(np.float64)) and b9["status"].sum() == 0 and len(b9["logs"]) == n_lab)

    # ------------------------------------------------------------------ F: families, with the real adapter
    import hn_screen_families as HF
    td = os.environ.get("TANITEVAL_DIR") or os.path.join(HF.repo_root(), "taniteval")
    if not os.path.isdir(os.path.join(td, "adapters")):
        return bad + [f"F_adapter_dir_missing ({td}; set TANITEVAL_DIR to a directory holding adapters/ and taniteval/)"]
    ad = HF.load_adapter(td)
    rng = np.random.default_rng(0)
    n = 60
    t = 0.5 * np.arange(1, 9)
    human = np.zeros((n, 8, 3))
    human[:, :, 0] = (8 * t)[None] + rng.normal(0, 0.2, (n, 8))
    plans8 = np.stack([human + rng.normal(0, 0.2, human.shape) for _ in range(3)], 1)
    plans8[:, 2, :, 1] += 1.5                                           # proposal 2 is laterally far from the human
    inputs = {"tokens": [f"t{i}" for i in range(n)], "logs": [f"log{i % 6}" for i in range(n)], "v0": np.full(n, 8.0), "human": human, "plans8": plans8}
    picks = {"base": np.zeros(n, int)}
    for s in (0, 1, 2):
        picks[f"A{s}"] = np.zeros(n, int)
        picks[f"B{s}"] = np.full(n, 2)                                  # B's plan is 1.5 m laterally off: an adverse LATERAL separation
    out = HF.build(ad, inputs, picks, n_boot=60)
    check("F_four_families_present_strategic_paused_and_unavailable_ones_have_reason_and_n", lambda: out["strategic"]["status"] == "PAUSED" and out["families_present"]["strategic"]["status"] == "PAUSED"
          and all(out["families_present"][k]["status"] == "OK" for k in ("longitudinal", "lateral", "tactical")) and out["not_computable"]["lead_distance_keeping"]["n"] == n
          and "reason" in out["not_computable"]["command_adherence_of_pick"] and set(out["blocks"]) == set(picks))
    check("F_adverse_lateral_watch_item_by_the_landed_rule_and_not_for_identical_arms", lambda: any(w.startswith("lateral.") for w in out["watch_items"]) and out["seeds"] == [0, 1, 2]
          and not HF.build(ad, inputs, {**picks, **{f"B{s}": np.zeros(n, int) for s in (0, 1, 2)}}, n_boot=60)["watch_items"])
    check("F_adapter_is_imported_not_reimplemented", lambda: "def scenes_to_win" not in open(os.path.join(HERE, "hn_screen_families.py"), encoding="utf-8").read() and hasattr(ad, "four_families_block"))
    check("F_block_is_the_seed_mean_of_its_leaves", lambda: abs(HF.seed_mean_block([{"a": {"x": 1.0}, "s": "k"}, {"a": {"x": 3.0}, "s": "k"}])["a"]["x"] - 2.0) < 1e-12)

    # ------------------------------------------------------------------ F: the per-set export (amendment B4), with the real adapter, the real writer / reader and the real paired read
    import hn_screen_read as HR
    from taniteval import per_window as PW
    dump = lambda o: json.dumps(o, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))   # noqa: E731
    outp = HF.build(ad, inputs, picks, n_boot=60, per_set=True)
    exps = outp.pop("_per_set_export")
    off = HF.build(ad, inputs, picks, n_boot=60, per_set=False)
    check("F_per_set_blocks_equal_the_default_blocks_and_the_unpaired_reading_is_not_computed_when_paired",
          lambda: dump(outp["blocks"]) == dump(off["blocks"]) and "watch_items" not in outp and "watch_items" in off and outp["adverse_reading"]["paired"] is True and off["adverse_reading"]["paired"] is False
          and set(exps) == set(picks) and outp["per_set"]["requested"] is True)
    n21 = lambda e: (len([c for c in e["longitudinal"].values() if not c.get("human_only")]), len([c for c in e["lateral"].values() if not c.get("extra_to_the_four_family_block")]))   # noqa: E731
    check("F_per_set_export_carries_the_21_components_of_section_10_for_every_arm_plus_the_declared_extras",
          lambda: all(e["status"] == "OK" and n21(e) == (14, 7) and "last_pose_heading_err_deg" in e["lateral"] and sum(c.get("human_only", False) for c in e["longitudinal"].values()) == 2 for e in exps.values())
          and all(sorted(e["unavailable"]) == ["longitudinal.distance_keeping.mean_headway_min_m", "longitudinal.distance_keeping.mean_min_ttc_s", "longitudinal.distance_keeping.mean_time_gap_min_s"] for e in exps.values()))
    pth = os.path.join(tmp, "fam_per_set.npz")
    HF.write_per_set(pth, exps)
    back = HF.read_per_set(pth)

    def same_export(x, y):
        for f in ("longitudinal", "lateral"):
            if set(x[f]) != set(y[f]):
                return False
            for k, c in x[f].items():
                for kk, v in c.items():
                    w = y[f][k][kk]
                    if isinstance(v, np.ndarray):
                        if not np.array_equal(v, w):
                            return False
                    elif v != w:
                        return False
        tx, ty = x["tactical"], y["tactical"]
        return (x["tokens"] == y["tokens"] and x["eid"] == y["eid"] and all(np.array_equal(tx[d][q], ty[d][q]) and tx[d]["classes"] == ty[d]["classes"] for d in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed")
                                                                             for q in ("gt", "pred")) and set(tx["goal_setting"]) == set(ty["goal_setting"]) and x["unavailable"] == y["unavailable"])

    check("F_per_set_npz_round_trips_exactly", lambda: set(back) == set(exps) and all(same_export(exps[a], back[a]) for a in exps))
    keys_n = [(lg, tk, 0, 0) for lg, tk in zip(inputs["logs"], inputs["tokens"])]

    def sc_n(pk):
        s = np.full((n, M, 7), -10.0)
        s[np.arange(n), pk, :] = 10.0
        return s

    Dn = HR.make_data(keys_n, inputs["logs"], np.tile(PROP, (n, 1, 1)), {nm: sc_n(pk) for nm, pk in picks.items()})
    JBn = HR.JointBoot(Dn["logs"], 400, 20260927)
    FPn = HR.families_paired(JBn, Dn, back, (0, 1, 2))
    check("F_paired_read_of_the_real_export_reads_21_components_and_flags_the_laterally_off_arm", lambda: FPn["status"] == "OK" and FPn["n_components_compared_in_the_21"] == 21 and FPn["IC"]["ok"]
          and "LATERAL.cross_mae_m" in FPn["watch_items"] and FPn["STRATEGIC"]["status"] == "PAUSED" and FPn["excluded_human_only"] == ["longitudinal.ego_progress.gt_progress_mean_m", "longitudinal.ego_progress.t0_axis_gt_self_ratio"])
    same_picks = {**picks, **{f"B{s}": picks[f"A{s}"] for s in (0, 1, 2)}}
    o_id = HF.build(ad, inputs, same_picks, n_boot=60, per_set=True)
    FPi = HR.families_paired(JBn, Dn, o_id.pop("_per_set_export"), (0, 1, 2))
    comps_i = [r for f in ("LONGITUDINAL", "LATERAL") for r in FPi[f].values()] + [FPi["TACTICAL"][d][q] for d in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed") for q in ("accuracy", "kappa")
                                                                                       if FPi["TACTICAL"][d][q]["status"] == "OK"]
    check("F_identical_arms_read_exactly_zero_with_CI_0_0_and_no_watch_item_on_the_real_export", lambda: FPi["watch_items"] == [] and len(comps_i) >= 21 and all(r["mean"] == 0.0 and r["ci95"] == [0.0, 0.0] for r in comps_i))
    cmp_ok = []
    for a, e in back.items():
        for f in ("longitudinal", "lateral"):
            for k, c in e[f].items():
                p_reader, p_inst = HR.comp_draws(JBn, c)[0], PW.point(c)
                cmp_ok.append(abs(p_reader - p_inst) < 1e-12 or (np.isnan(p_reader) and np.isnan(p_inst)))
    check("F_the_readers_point_equals_the_instruments_own_reduction_for_every_component", lambda: len(cmp_ok) >= 6 * 22 and all(cmp_ok))
    # the families CLI end to end: json without the arrays, the per-set npz beside it
    fin, fpk, fout, fps = (os.path.join(tmp, x) for x in ("fam_in.npz", "fam_picks.npz", "fam.json", "fam_ps.npz"))
    np.savez(fin, tokens=np.asarray(inputs["tokens"]), logs=np.asarray(inputs["logs"]), v0=inputs["v0"], human=inputs["human"], plans8=inputs["plans8"])
    np.savez(fpk, **picks)
    rc_f = HF.main(["--inputs", fin, "--picks", fpk, "--out", fout, "--per-set-out", fps, "--taniteval-dir", td, "--n-boot", "40"])
    fj = json.load(open(fout, encoding="utf-8"))
    bk2 = HF.read_per_set(fps)
    check("F_cli_writes_the_families_json_without_the_arrays_and_the_per_set_npz", lambda: rc_f == 0 and fj["per_set"]["file"] == "fam_ps.npz" and "_per_set_export" not in fj and "watch_items" not in fj
          and fj["adverse_reading"]["paired"] is True and set(bk2) == set(picks) and same_export(exps["B0"], bk2["B0"]))
    return bad


# ================================================================================================ C: the CLI end to end
def suite_cli(H):
    bad = []
    tmp = tempfile.mkdtemp(prefix="hsr_cli_")
    keys = [(f"log{i // PERLOG}", f"t{i}", 0, 0) for i in range(N)]
    picks = GO_PICKS = zero_picks((40, 40, 40), (0, 0, 0))
    for n, p in picks.items():
        RV.write_scores_npz(os.path.join(tmp, f"{n}.npz"), keys, [10075] * N, scores_from(p))
    os.makedirs(os.path.join(tmp, "pdm"))
    with open(os.path.join(tmp, "pdm", "pdm_all.jsonl"), "w", encoding="utf-8") as f:
        pdms = (PROP[:, 0] * PROP[:, 1] * (5 * PROP[:, 2] + 5 * PROP[:, 3] + 2 * PROP[:, 4]) / 12)
        for k in keys:
            rec = {"kind": "pdm_targets", "key": list(k), "ckpt_step": 10075, **{c: PROP[:, j].tolist() for j, c in enumerate(("nc", "dac", "ep", "ttc", "c", "ddc"))}, "pdms": pdms.tolist()}
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(tmp, "scenes.tsv"), "w", encoding="utf-8") as f:
        f.write("token\tlog\n" + "".join(f"{k[1]}\t{k[0]}\n" for k in keys))
    json.dump(GATES_OK, open(os.path.join(tmp, "gates.json"), "w"))
    json.dump(SY_OK, open(os.path.join(tmp, "sy.json"), "w"))
    chain_root, chain_key = make_chain_root(tmp, "chain", (0, 1, 2), "content")                  # HNSCREEN-FP-COST: the reader's cache-binding gate needs the passes' tokens
    out = os.path.join(tmp, "result.json")
    rc = H.main(["--scores"] + [f"{n}={os.path.join(tmp, n + '.npz')}" for n in picks] + ["--pdm", os.path.join(tmp, "pdm"), "--rl-val-scenes", os.path.join(tmp, "scenes.tsv"),
                 "--gates", os.path.join(tmp, "gates.json"), "--sy", os.path.join(tmp, "sy.json"), "--rule", "navsim_v1_lane", "--n-boot", str(NB), "--out", out,
                 "--chain-root", chain_root, "--key-file", chain_key])
    res = json.load(open(out, encoding="utf-8"))
    ok = (rc == 0 and res["outcome"]["row"] == 3 and res["strategic"]["status"] == "PAUSED" and "log-cluster bootstrap" in res["estimator"] and res["M2"]["status"] == "NOT COMPUTABLE"
          and res["families"]["published"] is False and res["IC"]["ok"] and res["n_sets"] == N)
    if not ok:
        bad.append("C_cli_files_in_result_json_out")
    # the same run with the families json and the per-set export (synthetic, canonical order): published, with the paired watch-items, and the outcome unchanged
    import hn_screen_families as HF
    fx = _fx()
    HF.write_per_set(os.path.join(tmp, "fam_per_set.npz"), pexports(fx))
    json.dump({"families_present": {k: {"status": "OK"} for k in ("longitudinal", "lateral", "tactical")} | {"strategic": {"status": "PAUSED"}}}, open(os.path.join(tmp, "fam.json"), "w"))
    out2 = os.path.join(tmp, "result2.json")
    rc2 = H.main(["--scores"] + [f"{n}={os.path.join(tmp, n + '.npz')}" for n in picks] + ["--pdm", os.path.join(tmp, "pdm"), "--rl-val-scenes", os.path.join(tmp, "scenes.tsv"),
                  "--gates", os.path.join(tmp, "gates.json"), "--sy", os.path.join(tmp, "sy.json"), "--families", os.path.join(tmp, "fam.json"), "--families-per-set", os.path.join(tmp, "fam_per_set.npz"),
                  "--rule", "navsim_v1_lane", "--n-boot", str(NB), "--out", out2, "--chain-root", chain_root, "--key-file", chain_key])
    r2 = json.load(open(out2, encoding="utf-8"))
    if not (rc2 == 0 and r2["outcome"]["row"] == 3 and r2["families"]["published"] is True and r2["families"]["paired"] is True and r2["families_paired"]["status"] == "OK"
            and r2["watch_items_trajectory"] == ["LATERAL.cross_mae_m", "LATERAL.curvature_bias_1pm", "LATERAL.heading_mae_deg", "LONGITUDINAL.along_bias_m", "LONGITUDINAL.ego_progress.progress_ratio_mean",
                                                 "LONGITUDINAL.ego_progress.progress_ratio_median", "LONGITUDINAL.target_speed_acc.within_1.0_mps"] and r2["strategic"]["status"] == "PAUSED"):
        bad.append("C_cli_families_per_set_published_paired_and_outcome_unchanged")
    return bad




# ================================================================================================ G: the CACHE-BINDING GATE of the reader (HNSCREEN-FP-COST)
def make_chain_root(tmp, name, seeds, mode="content", end="ok"):
    """A chain state dir the reader's gate can read: signed PASS tokens under passes/s<seed>/lg/ whose --ctx-cache is bound by CONTENT or by an ANCHOR, and (anchor mode) ctx_chain/anchor.json + end_verify.json written by the
    gate's own functions on a tiny sealed directory.  end = ok (END_VERIFY_OK) | void (a same-size, same-mtime edit before the end verify) | none (no end record).  -> (root, key file)"""
    import shutil
    import gate_token as GT
    keyf = os.path.join(tmp, "gate.key")
    key = GT.load_key(keyf, create=True)
    root = os.path.join(tmp, name)
    cache = os.path.join(tmp, name + "_cache")
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(os.path.join(cache, "units"))
    for i in range(3):
        with open(os.path.join(cache, "units", f"u{i}.bin"), "wb") as f:
            f.write(bytes([i]) * (64 + i))
    with open(os.path.join(cache, "manifest.json"), "w") as f:
        f.write("{}")
    GT.seal_content_manifest(cache)
    pin = GT.fp_key(GT.content_fingerprint(cache))
    if mode == "content":
        fp = {"path": cache, **GT.content_fingerprint(cache)}
    else:
        os.makedirs(os.path.join(root, "ctx_chain"), exist_ok=True)
        ap = os.path.join(root, GT.ANCHOR_REL)
        GT.mint_anchor(cache, chain_id="ab" * 16, key=key, pin=pin, out_path=ap)
        doc, _why, sha = GT.load_anchor(ap, key)
        anc = {"doc": doc, "sha256": sha}
        fp = {"path": cache, **GT.anchor_fingerprint(cache, anc)}
        if end == "void":
            p = os.path.join(cache, "units", "u0.bin")
            st = os.stat(p)
            with open(p, "r+b") as f:
                f.write(b"\xff")
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        if end in ("ok", "void"):
            GT.end_verify(cache, anc, key, os.path.join(root, GT.END_REL), seeds=seeds)
    for s in seeds:
        out = os.path.join(root, "passes", f"s{s}", "lg")
        os.makedirs(out, exist_ok=True)
        GT.mint(profile="refe_hnscreen", commit="a" * 40, tree_man={}, argv=[], trainer="scorer_finetune.py", data={"--ctx-cache": fp}, checks={}, reasons=[], key=key, out_dir=out)
    return root, keyf


def suite_chain_gate(H, tag, only=None):
    """The reader refuses (rc 6, ZZHNSCREEN_REFUSED, nothing read, no result written) unless every pass's cache binding is proven; admits a content-bound chain and an anchored chain with an END_VERIFY_OK.  Runs the REAL
    `main` of the module under test up to the first score file (a sentinel stops it there: the gate has run by then)."""
    import contextlib
    import io
    bad = []
    sel = (lambda name: only is None or name in only)
    tmp = tempfile.mkdtemp(prefix=f"hsr_gate_{tag}_")

    class Past(Exception):
        pass

    def stop(*a, **k):
        raise Past()

    def run(root, keyf, seeds=(0, 1, 2), extra=()):
        out = os.path.join(tmp, "result_gate.json")
        if os.path.exists(out):
            os.remove(out)
        buf = io.StringIO()
        real = H.RV.load_scores
        H.RV.load_scores = stop
        try:
            with contextlib.redirect_stdout(buf):
                try:
                    rc = H.main(["--scores", "base=x.npz"] + [f"{a}{s}=x.npz" for s in seeds for a in "AB"] + ["--pdm", "p", "--rl-val-scenes", "s", "--gates", "g.json", "--rule", "navsim_v1_lane", "--out", out,
                                 "--chain-root", root, "--key-file", keyf, *extra])
                except Past:
                    rc = "PAST_GATE"
        finally:
            H.RV.load_scores = real
        return rc, buf.getvalue(), os.path.exists(out)

    def refused(rc, txt, wrote, frag):
        return rc == 6 and "ZZHNSCREEN_REFUSED" in txt and frag in txt and not wrote

    if sel("G_a_content_bound_chain_is_admitted_without_an_end_record"):
        root, keyf = make_chain_root(tmp, "c_ok", (0, 1, 2), "content")
        rc, txt, wrote = run(root, keyf)
        if not (rc == "PAST_GATE" and "REFUSED" not in txt):
            bad.append("G_a_content_bound_chain_is_admitted_without_an_end_record")
    if sel("G_an_anchored_chain_with_END_VERIFY_OK_is_admitted"):
        root, keyf = make_chain_root(tmp, "a_ok", (0, 1, 2), "anchor", "ok")
        rc, txt, wrote = run(root, keyf)
        if not (rc == "PAST_GATE" and "REFUSED" not in txt):
            bad.append("G_an_anchored_chain_with_END_VERIFY_OK_is_admitted")
    if sel("G_R5_an_anchored_chain_without_an_end_record_is_refused_before_any_score_is_read"):
        root, keyf = make_chain_root(tmp, "a_none", (0, 1, 2), "anchor", "none")
        if not refused(*run(root, keyf), "no END_VERIFY_OK record"):
            bad.append("G_R5_an_anchored_chain_without_an_end_record_is_refused_before_any_score_is_read")
    if sel("G_R1_a_VOID_chain_is_refused_before_any_score_is_read"):
        root, keyf = make_chain_root(tmp, "a_void", (0, 1, 2), "anchor", "void")
        if not refused(*run(root, keyf), "VOID"):
            bad.append("G_R1_a_VOID_chain_is_refused_before_any_score_is_read")
    if sel("G_an_END_record_that_does_not_cover_every_seed_read_is_refused"):
        root, keyf = make_chain_root(tmp, "a_few", (0, 1), "anchor", "ok")
        import shutil
        shutil.copytree(os.path.join(root, "passes", "s1"), os.path.join(root, "passes", "s2"))      # a third pass the END record never covered
        if not refused(*run(root, keyf), "covers seeds"):
            bad.append("G_an_END_record_that_does_not_cover_every_seed_read_is_refused")
    if sel("G_a_chain_root_with_no_tokens_is_refused"):
        os.makedirs(os.path.join(tmp, "empty_root"), exist_ok=True)
        root, keyf = make_chain_root(tmp, "c_for_key", (0,), "content")
        if not refused(*run(os.path.join(tmp, "empty_root"), keyf), "no PASS token"):
            bad.append("G_a_chain_root_with_no_tokens_is_refused")
    if sel("G_a_missing_gate_key_is_a_refusal_never_a_pass"):
        root, keyf = make_chain_root(tmp, "c_nokey", (0, 1, 2), "content")
        rc, txt, wrote = run(root, os.path.join(tmp, "no_such.key"))
        if not (rc == 6 and "cannot be checked" in txt and not wrote):
            bad.append("G_a_missing_gate_key_is_a_refusal_never_a_pass")
    if sel("G_the_chain_root_is_a_required_argument"):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                H.main(["--scores", "base=x.npz", "--pdm", "p", "--rl-val-scenes", "s", "--gates", "g.json", "--rule", "navsim_v1_lane", "--out", os.path.join(tmp, "r.json")])
            bad.append("G_the_chain_root_is_a_required_argument")
        except SystemExit as e:
            if e.code != 2:
                bad.append("G_the_chain_root_is_a_required_argument")
    return bad


# ================================================================================================ mutations
MUTATIONS = [
    ("the per-seed draws are de-synchronised",
     [('    r = JB.mean_interval(d)\n    r["per_seed_point"]',
       '    r = {"mean": float(d.mean()), "ci95": JointBoot._ci(np.mean([JointBoot([str(x) for x in JB.li], JB.n, JB.seed + k).means(scale * (np.asarray(b, np.float64) - np.asarray(a, np.float64)))\n'
       '                                                                   for k, (b, a) in enumerate(zip(vb, va))], axis=0))}\n    r["per_seed_point"]')],
     ["J_joint_draws_equal_proxy_eval_boot_and_perfectly_correlated_seeds_keep_their_interval"]),
    ("tied picks broken by the LAST index", [("    return RV.select(scores, rule, space)[0]\n", "    sel = RV.select(scores, rule, space)[1]\n    return int(len(sel) - 1 - np.argmax(sel[::-1]))\n")],
     ["T_ties_go_to_the_lowest_index_and_lookup_hand_values"]),
    ("Z3 uses >= (the boundary)", [('"Z3": bool(-dsum > S * spread)', '"Z3": bool(-dsum >= S * spread)')], ["O_boundary_dz_exactly_at_Fz_is_not_ZEROS_FALL"]),
    ("Z2 needs only ONE seed", [('"Z2": bool(all(b - a < 0 for a, b in zip(nA, nB)))', '"Z2": bool(any(b - a < 0 for a, b in zip(nA, nB)))')], ["O_Z2_needs_every_seed", "O_row5_SEED_FRAGILE_rule_E_fired_and_fails"]),
    ("F_z read from the B arms", [("    spread = max(nA) - min(nA)\n", "    spread = max(nB) - min(nB)\n")], ["O_row3_via_rule_E"]),
    ("rule E needs 3 of 5", [('"Z2_prime_ge4of5": bool(sum(b - a < 0 for a, b in zip(nA, nB)) >= 4)', '"Z2_prime_ge4of5": bool(sum(b - a < 0 for a, b in zip(nA, nB)) >= 3)')], ["O_row5_SEED_FRAGILE_rule_E_fired_and_fails"]),
    ("the co-gate needs 3 adverse seeds", [("def cogate_decision(mean_ub, seed_ubs, k_seeds=2):", "def cogate_decision(mean_ub, seed_ubs, k_seeds=3):")], ["O_cogate_2_of_3_adverse_seeds_fail_1_does_not"]),
    ("the co-gate ignores the seed-mean interval", [('"fails": bool(mean_ub < 0 or n_adverse >= k_seeds)', '"fails": bool(n_adverse >= k_seeds)')], ["O_cogate_2_of_3_adverse_seeds_fail_1_does_not"]),
    ("Delta PDMS >= 0 counts as GO", [('return 3 if inp["dpdms_point"] > 0 else 4', 'return 3 if inp["dpdms_point"] >= 0 else 4')], ["O_dPDMS_exactly_zero_is_NET_ZERO"]),
    ("SY <= 0.10 is sparse", [('sy.get("share_with_hn", 1.0) < SY_MIN_SHARE', 'sy.get("share_with_hn", 1.0) <= SY_MIN_SHARE')], ["O_SY_exactly_0.10_is_not_sparse"]),
    ("the co-gate is judged after the zeros", [('    if inp["cogate_fails"]:\n        return 2\n', '    if inp["cogate_fails"] and not inp["zeros"]["zeros_fall"]:\n        return 2\n')],
     ["O_row2_UNSAFE_even_when_zeros_fall", "O_first_matching_row_wins"]),
    ("M3 minimum 10 scenes instead of 100", [("M3_MIN_SCENES = 100", "M3_MIN_SCENES = 10")], ["O_row8or9_unresolved_when_M3_not_computable"]),
    ("REVERSED reads the upper end", [('"LB_positive": bool(iv["ci95"][0] > 0)', '"LB_positive": bool(iv["ci95"][1] > 0)')], ["O_mixed_sign_effect_is_not_reversed"]),
    ("a missing gate is not a failure", [('for k in GATE_KEYS + ("IC",)', 'for k in ()')],
     ["O_row0_a_missing_gate_is_a_failed_gate"]),
    ("the interval is biased", [('return {"mean": float(v.mean()), "ci95": self._ci(self.means(v))}', 'return {"mean": float(v.mean()) + 1e-9, "ci95": self._ci(self.means(v))}')],
     ["I_arm_minus_itself_reads_exactly_zero", "I_ic_failure_makes_the_screen_invalid"]),
    ("the bootstrap seed", [("N_BOOT, SEED = 10000, 20260927", "N_BOOT, SEED = 10000, 20260928")], ["J_joint_draws_equal_proxy_eval_boot_and_perfectly_correlated_seeds_keep_their_interval"]),
    ("the paired B draws meet the REVERSED A draws (unpaired)", [("    drs = [b[1] - a[1] for b, a in zip(pb, pa)]\n", "    drs = [b[1] - a[1][::-1] for b, a in zip(pb, pa)]\n")], ["P_pairing_is_real_a_shift_the_unpaired_intervals_miss_is_sharp_on_the_paired_draws", "P_family_IC_an_arm_minus_itself_is_exactly_zero_with_CI_0_0_for_every_component"]),
    ("higher-is-better read as lower-is-better", [('    out["adverse"] = bool(adv["ci95"][1] < 0) if direction == "higher" else bool(adv["ci95"][0] > 0)\n', '    out["adverse"] = bool(adv["ci95"][0] > 0)\n')],
     ["P_directions_follow_the_landed_table_higher_UB_below_0_lower_LB_above_0"]),
    ("bias read as a signed difference", [('ADVERSE_TF = {"magnitude": np.abs, "closer_to_1"', 'ADVERSE_TF = {"closer_to_1"')], ["P_bias_is_adverse_only_when_B_is_farther_from_zero_and_progress_ratio_when_farther_from_1", "P_watch_items_are_the_adverse_longitudinal_and_lateral_components"]),
    ("the export is aligned by position, not by (log, token)", [("    order = np.asarray([idx[w] for w in want], np.int64)\n", "    order = np.arange(len(want), dtype=np.int64)\n")], ["P_shuffled_export_pairs_identically"]),
    ("a renamed token is refused only on a count mismatch", [("    if miss or len(idx) != len(want):\n", "    if len(idx) != len(want):\n")], ["P_different_sets_duplicates_and_missing_arms_are_refused_loudly"]),
    ("an undefined kappa reads 1.0", [("(po - pe) / (1.0 - pe), np.nan)", "(po - pe) / (1.0 - pe), 1.0)")], ["P_kappa_undefined_is_not_computable_never_1"]),
    ("the resampled median is the lower middle element", [("            i_hi = (cum <= np.floor(tot / 2)[:, None]).sum(1)\n", "            i_hi = i_lo\n")], ["P_median_draws_equal_the_median_of_the_expanded_resample"]),
    ("a masked component's draws are a mean of ratios", [('else float("nan")), sc * JB.ratios(num, den)', 'else float("nan")), sc * JB.means(np.divide(num, den, out=np.zeros_like(num), where=den > 0))')], ["P_masked_ratio_draws_equal_the_expanded_resample_ratio"]),
    ("the rms draws are the mean square (no root)", [("math.sqrt(float(v.mean())), np.sqrt(JB.means(v))", "math.sqrt(float(v.mean())), JB.means(v)")], ["P_rms_draws_equal_the_root_of_the_expanded_mean_square"]),
    ("the family bootstrap uses its own draws (another seed)", [("    fp = families_paired(JB, D, per_set, seeds) if per_set is not None else None\n", '    fp = families_paired(JointBoot(D["logs"], n_boot, seed + 1), D, per_set, seeds) if per_set is not None else None\n')], ["P_family_components_are_read_on_the_SAME_joint_draws_as_the_PDM_sub_scores"]),
    ("the watch-items gate the outcome", [('"cogate_fails": res["cogate"]["fails"],', '"cogate_fails": res["cogate"]["fails"] or bool(fp and fp.get("watch_items")),')], ["P_watch_items_reported_not_gating_the_outcome_row_is_unchanged"]),
    ("families are published without the family IC", [('paired.get("status") == "OK" and paired.get("IC", {}).get("ok"))', 'paired.get("status") == "OK")')], ["P_families_published_only_when_paired_and_IC_holds"]),
    ("the human-only components are compared", [('            if any(e[fam][nm].get("human_only") for e in Bs + As):\n', "            if False:\n")], ["P_human_only_components_excluded"]),
    ("the reader's cache-binding gate is not called", [('        why = GT.ctx_chain_refusals(a.chain_root, passes, GT.load_key(a.key_file, create=False))\n', '        why = []\n')],
     ["G_R5_an_anchored_chain_without_an_end_record_is_refused_before_any_score_is_read", "G_R1_a_VOID_chain_is_refused_before_any_score_is_read", "G_a_chain_root_with_no_tokens_is_refused"]),
    ("the reader ignores a refusal of the gate", [('    if why:\n        for r in why[:6]:', '    if False:\n        for r in why[:6]:')],
     ["G_R5_an_anchored_chain_without_an_end_record_is_refused_before_any_score_is_read", "G_R1_a_VOID_chain_is_refused_before_any_score_is_read"]),
    ("a gate that cannot run is a pass", [('        why = [f"the chain\'s cache binding cannot be checked: {type(e).__name__}: {str(e)[:200]}"]\n', '        why = []\n')],
     ["G_a_missing_gate_key_is_a_refusal_never_a_pass"]),
    ("the reader checks no seed (an empty pass list)", [('    passes = sorted({int(n[1:])', '    passes = [] and sorted({int(n[1:])')],
     ["G_R5_an_anchored_chain_without_an_end_record_is_refused_before_any_score_is_read", "G_a_chain_root_with_no_tokens_is_refused"]),
    ("--chain-root is optional", [('    ap.add_argument("--chain-root", required=True,', '    ap.add_argument("--chain-root", required=False, default=".",')],
     ["G_the_chain_root_is_a_required_argument"]),
    ("M2 reads A minus B", [('m2 = ratio_diff(JB, [tw[f"B{s}"][1] for s in seeds], [tw[f"A{s}"][1] for s in seeds], den)', 'm2 = ratio_diff(JB, [tw[f"A{s}"][1] for s in seeds], [tw[f"B{s}"][1] for s in seeds], den)')], ["O_row7_HEAD_MOVES_NO_ZEROS"]),
]


def apply_edits(src, edits, name):
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation {name!r}: snippet occurs {n} times (need exactly 1): {old[:70]!r}")
        src = src.replace(old, new)
    return src


def main() -> int:
    src = open(SRC_READ, encoding="utf-8").read()
    Hreal = load_read(src, "real")
    real = suite(Hreal, "real") + suite_items56() + suite_cli(Hreal) + suite_paired(Hreal, "real") + suite_chain_gate(Hreal, "real")
    assert not real, f"the unmutated sources FAIL their own checks: {real}"
    n, problems = 1, []
    print(f"{'mutation':50s} caught-by (failing checks)")
    for name, edits, expect in MUTATIONS:
        m_src = apply_edits(src, edits, name)
        try:
            ast.parse(m_src)
            Hm = load_read(m_src, f"m{n}")
            failing = suite(Hm, f"m{n}", only=set(expect)) + suite_paired(Hm, f"m{n}", only=set(expect)) + suite_chain_gate(Hm, f"m{n}", only=set(expect))
        except (Exception, SystemExit) as e:                           # noqa: BLE001 -- a mutated source that breaks the suite's own setup is RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:80]}"]
        if not failing:
            problems.append(f"mutation {name!r} stayed GREEN: the suite cannot see it")
            continue
        for c in expect:
            if c not in failing and not any(f.startswith("suite setup raised") for f in failing):
                problems.append(f"mutation {name!r}: the check {c} must go RED and stayed green (RED: {failing[:8]})")
        print(f"{name[:50]:50s} {', '.join(failing[:2])}", flush=True)
        n += 1
    if problems:
        print("\n".join("PROBLEM " + p for p in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite (incl. items 5, 6, CLI, paired families) + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
