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
  C  the CLI: files in, a result json out (the GO scenario), with the pause stamp and the estimator named.
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
    out = os.path.join(tmp, "result.json")
    rc = H.main(["--scores"] + [f"{n}={os.path.join(tmp, n + '.npz')}" for n in picks] + ["--pdm", os.path.join(tmp, "pdm"), "--rl-val-scenes", os.path.join(tmp, "scenes.tsv"),
                 "--gates", os.path.join(tmp, "gates.json"), "--sy", os.path.join(tmp, "sy.json"), "--rule", "navsim_v1_lane", "--n-boot", str(NB), "--out", out])
    res = json.load(open(out, encoding="utf-8"))
    ok = (rc == 0 and res["outcome"]["row"] == 3 and res["strategic"]["status"] == "PAUSED" and "log-cluster bootstrap" in res["estimator"] and res["M2"]["status"] == "NOT COMPUTABLE"
          and res["families"]["published"] is False and res["IC"]["ok"] and res["n_sets"] == N)
    if not ok:
        bad.append("C_cli_files_in_result_json_out")
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
    real = suite(Hreal, "real") + suite_items56() + suite_cli(Hreal)
    assert not real, f"the unmutated sources FAIL their own checks: {real}"
    n, problems = 1, []
    print(f"{'mutation':50s} caught-by (failing checks)")
    for name, edits, expect in MUTATIONS:
        m_src = apply_edits(src, edits, name)
        try:
            ast.parse(m_src)
            failing = suite(load_read(m_src, f"m{n}"), f"m{n}", only=set(expect))
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
    print(f"ZZTEST_PASS {n}  (1 unmutated suite (incl. items 5, 6, CLI) + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
