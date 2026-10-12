#!/usr/bin/env python3
"""Tests for refe/rl_reward.py (RL-S1 part 1).  Run: python refe/test_rl_reward.py  ->  ZZTEST_PASS <n> (prints the mutation table).

Every value below is HAND-COMPUTED (the arithmetic is in the comment next to it), never produced by running the module.
A check that cannot go RED is not a check: each term has a mutation that edits the module's SOURCE (the snippet must occur exactly once,
else the test errors -- a vacuous mutation proves nothing) and the suite must then report >= 1 failing check.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "rl_reward.py")


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    try:
        exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    finally:
        pass
    return m


def suite(R) -> list:
    """The checks, against module `R`.  Returns the list of FAILING check names (empty = all green)."""
    bad: list = []

    def t(name, fn):
        try:
            ok = bool(fn())
        except Exception:                                                     # noqa: BLE001 -- an exception IS a failed check
            ok = False
        if not ok:
            bad.append(name)

    def raises(fn, exc=ValueError):
        try:
            fn()
        except exc:
            return True
        return False

    close = lambda a, b, tol=1e-9: bool(np.allclose(a, b, rtol=0, atol=tol))                  # noqa: E731
    # ---- the scene: (nc, dac, ep, ttc, c, ddc) per proposal
    tg = np.array([[1, 1, 1.0, 1, 1, 1],        # P0 safe, full progress           PDMS = (5 + 5 + 2) / 12                 = 1.0
                   [1, 1, 0.5, 1, 1, 1],        # P1 safe, half progress          PDMS = (2.5 + 5 + 2) / 12 = 9.5 / 12    = 0.7916666667
                   [0, 1, 0.0, 1, 1, 1],        # P2 collision (EP zeroed)        PDMS = 0
                   [1, 1, 0.2, 0, 1, 1]])       # P3 TTC violation                PDMS = (1 + 0 + 2) / 12                  = 0.25
    d = np.array([0.0, 0.2, 1.0, 0.4])
    cfg0 = R.RewardCfg()
    # ---- defaults: the registered values, and K / L OFF
    t("defaults", lambda: (cfg0.lam_d, cfg0.w_prog, cfg0.guard, cfg0.lam_k, cfg0.use_k, cfg0.lam_l, cfg0.use_l) == (0.15, 0.5, True, 0.05, False, 0.05, False))
    # ---- PDMS (v1.1 weights 5/5/2 over 12)
    p = R.pdms_v1(*(tg[:, i] for i in range(5)))
    t("pdms_values", lambda: close(p, [1.0, 9.5 / 12.0, 0.0, 0.25]))
    # ---- progress guard: best safe EP = 1.0 (P0); P1 shortfall 0.5 - 1.0 = -0.5; unsafe proposals get 0
    out = R.reward(tg, cfg0, d=d)
    t("guard_values", lambda: close(out["A_prog"], [0.0, -0.5, 0.0, 0.0]))
    t("safe_mask", lambda: out["safe"].tolist() == [True, True, False, False])
    # ---- R = PDMS - 0.15 D + 0.5 A_prog:  P0 1.0 | P1 0.7916666667 - 0.03 - 0.25 = 0.5116666667 | P2 0 - 0.15 = -0.15 | P3 0.25 - 0.06 = 0.19
    t("R_values", lambda: close(out["R"], [1.0, 9.5 / 12.0 - 0.03 - 0.25, -0.15, 0.19]))
    # ---- the analytic identity: lam_d = 0 and guard OFF  =>  R == PDMS bit for bit (and the K / L flags OFF ignore their inputs)
    cid = R.RewardCfg(lam_d=0.0, guard=False)
    lane_junk = {"lc_executed": np.ones(4, bool), "label_fires": np.array(False), "dither": np.ones(4)}
    oid = R.reward(tg, cid, k=np.full(4, 0.9), lane=lane_junk)
    t("identity_R_equals_PDMS", lambda: np.array_equal(oid["R"], oid["pdms"]) and "K" not in oid and "L" not in oid)
    # ---- a term the cfg turns ON but whose input is missing RAISES (never silently zero)
    t("missing_d_raises", lambda: raises(lambda: R.reward(tg, cfg0)))
    t("missing_k_raises", lambda: raises(lambda: R.reward(tg, R.RewardCfg(lam_d=0.0, use_k=True))))
    t("missing_lane_raises", lambda: raises(lambda: R.reward(tg, R.RewardCfg(lam_d=0.0, use_l=True))))
    t("d_range_raises", lambda: raises(lambda: R.reward(tg, cfg0, d=np.array([0.0, 0.2, 1.5, 0.4]))))
    t("subscore_range_raises", lambda: raises(lambda: R.reward(tg * 2.0, cid)))
    # ---- the reference is the best SAFE proposal: an UNSAFE plan with more progress must not set it (P1 is the only safe one: shortfall 0)
    tg2 = np.array([[1, 1, 0.5, 1, 1, 1], [1, 1, 1.0, 0, 1, 1]])
    t("guard_reference_ignores_unsafe", lambda: close(R.reward(tg2, R.RewardCfg(lam_d=0.0))["A_prog"], [0.0, 0.0]))
    # ---- no safe proposal in the scene: no reference, the guard is exactly 0
    tg3 = np.array([[0, 1, 0.0, 1, 1, 1], [1, 1, 0.3, 0, 1, 1]])
    t("guard_no_safe_is_zero", lambda: close(R.reward(tg3, R.RewardCfg(lam_d=0.0))["A_prog"], [0.0, 0.0]))
    # ---- a leading batch axis changes nothing
    tgb, db = np.stack([tg, tg[::-1]]), np.stack([d, d[::-1]])
    ob = R.reward(tgb, cfg0, d=db)["R"]
    t("batched_equals_single", lambda: close(ob[0], out["R"]) and close(ob[1], out["R"][::-1]))
    # ---- P4 ENFORCED (ruling 2026-10-12).  Uncapped: safe half-speed 0.7917 - 0.25 = 0.5417 < TTC-violating full speed 7/12 = 0.5833 (the finding).
    # Capped at half the margin: 0.7917 - 0.5 * (0.7917 - 0.5833) = 0.5 * (9.5 + 7) / 12 = 16.5 / 24 = 0.6875 > 0.5833.  P0 (best safe) untouched.
    tg4 = np.array([[1, 1, 1.0, 1, 1, 1], [1, 1, 0.5, 1, 1, 1], [1, 1, 1.0, 0, 1, 1]])
    o4 = R.reward(tg4, R.RewardCfg(lam_d=0.0))
    t("guard_never_crosses_safety_boundary", lambda: o4["R"][1] > o4["R"][2] and close(o4["R"], [1.0, 16.5 / 24.0, 7.0 / 12.0]))
    # ---- ...but the guard still REORDERS INSIDE the safe set (its purpose).  P0 safe EP 1 C 0: (5+5+0)/12 = 0.8333 | P1 safe EP 0.8 C 1:
    # (4+5+2)/12 = 0.9167, shortfall -0.2 -> -0.1 -> 0.8167 < 0.8333 (no unsafe proposal below either, so no cap)
    tg5 = np.array([[1, 1, 1.0, 1, 0, 1], [1, 1, 0.8, 1, 1, 1]])
    t("guard_reorders_inside_safe", lambda: close(R.reward(tg5, R.RewardCfg(lam_d=0.0))["R"], [10.0 / 12.0, 11.0 / 12.0 - 0.1]))
    # ---- the P4 PROPERTY over 300 random 16-proposal fans with drivable excess: no safe proposal ends at or below an unsafe one it beat pre-guard
    rng = np.random.default_rng(12)
    S_, M_ = 300, 16
    tgr = np.stack([rng.random((S_, M_)) < 0.8, rng.random((S_, M_)) < 0.9, rng.random((S_, M_)), rng.random((S_, M_)) < 0.7,
                    rng.random((S_, M_)) < 0.8, np.ones((S_, M_))], -1).astype(np.float64)
    dr = rng.random((S_, M_)) * (rng.random((S_, M_)) < 0.3)
    orr = R.reward(tgr, R.RewardCfg(), d=dr)
    br, sf, rr = orr["pdms"] - 0.15 * dr, orr["safe"], orr["R"]
    pairs = sf[:, :, None] & ~sf[:, None, :] & (br[:, :, None] > br[:, None, :])
    t("P4_property_random_fans", lambda: int(pairs.sum()) > 1000 and int((pairs & ~(rr[:, :, None] > rr[:, None, :])).sum()) == 0)
    # ---- check_pdms: agrees with a matching pdms column, refuses a wrong one
    t("check_pdms_ok", lambda: R.check_pdms(tg, p) < 1e-12)
    t("check_pdms_refuses", lambda: raises(lambda: R.check_pdms(tg, p + 0.01)))
    # ---- the label-record stacking (OnPolicyBank.PDM_ORDER) is COMPONENTS order
    rec = {c: [float(x)] * 2 for c, x in zip(("nc", "dac", "ep", "ttc", "c", "ddc"), (1, 0.5, 0.25, 0.125, 0.0625, 0.03125))}
    t("record_order", lambda: close(R.tg_from_pdm_record(rec)[0], [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]))
    # ---- D: mean_t min(g / 0.5, 1):  g = [0, 0.25, 0.5, 1.0] -> [0, 0.5, 1, 1] -> 2.5 / 4 = 0.625; inside everywhere -> exactly 0
    t("D_value", lambda: close(R.drivable_excess(np.array([[0.0, 0.25, 0.5, 1.0]])), [0.625]))
    t("D_zero_inside", lambda: float(R.drivable_excess(np.zeros((3, 41)))[0]) == 0.0)
    t("D_negative_gap_raises", lambda: raises(lambda: R.drivable_excess(np.array([[-0.1, 0.0]]))))
    # ---- K.  o = [0, .25, .625, 1, 2], dpsi = [0, 10, 20, 30, 40], straddle 0.4
    o_, dp_ = np.array([[0.0, 0.25, 0.625, 1.0, 2.0]]), np.array([[0.0, 10.0, 20.0, 30.0, 40.0]])
    no_int = np.zeros((1, 5), bool)
    # k1 = mean([0, 0, .5, 1, 1]) = 0.5 | k2 (o < 1.5 only: the 2.0 m step is a lane change, not a heading violation) = mean([0, 0, .5, 1, 0]) = 0.3
    # K = 0.6 * 0.5 + 0.4 * 0.3 + 0.2 * 0.4 = 0.30 + 0.12 + 0.08 = 0.5
    t("K_value", lambda: close(R.lane_consistency(o_, dp_, no_int, np.array([0.4])), [0.5]))
    # steps 3, 4 inside an intersection are masked: k1 = mean([0, 0, .5, 0, 0]) = 0.1 | k2 = mean([0, 0, .5, 0, 0]) = 0.1 | K = 0.06 + 0.04 + 0.08 = 0.18
    t("K_intersection_masked", lambda: close(R.lane_consistency(o_, dp_, np.array([[0, 0, 0, 1, 1]], bool), np.array([0.4])), [0.18]))
    t("K_zero_when_on_centerline", lambda: float(R.lane_consistency(np.zeros((1, 5)), np.zeros((1, 5)), no_int, np.zeros(1))[0]) == 0.0)
    ck = R.RewardCfg(lam_d=0.0, guard=False, use_k=True)
    t("K_enters_reward", lambda: close(R.reward(tg, ck, k=np.array([0.5, 0.0, 1.0, 0.2]))["R"], p - 0.05 * np.array([0.5, 0.0, 1.0, 0.2])))
    # ---- L (review 3.3).  P0 lane change, safe, ep .9 | P1 stay, safe, ep .6 | P2 lane change UNSAFE, ep .95 | P3 stay, safe, ep .5, dithers
    tgl = np.array([[1, 1, 0.9, 1, 1, 1], [1, 1, 0.6, 1, 1, 1], [1, 0, 0.95, 1, 1, 1], [1, 1, 0.5, 1, 1, 1]])
    lcx, dith = np.array([True, False, True, False]), np.array([0.0, 0.0, 0.0, 1.0])
    sf = np.array([True, True, False, True])

    def L(ep_col, lab, lc=lcx, di=dith, safe=sf):
        return R.lane_change_term(lc, np.array(lab), di, np.asarray(ep_col, np.float64), safe)
    ep_l = tgl[:, 2]
    # label fires: EP_stay = max(.6, .5) = .6; P0 gain = clip((.9 - .6) / .2, 0, 1) = clip(1.5) = 1 -> +1 | P1 0 | P2 unsafe -> no credit, labelled -> no penalty -> 0 | P3 dither -> -1
    t("L_labelled", lambda: close(L(ep_l, True), [1.0, 0.0, 0.0, -1.0]))
    # NO label: every executed change is penalised -1 (P0, P2), no positive credit anywhere; P3's dither -1
    t("L_unlabelled_penalised", lambda: close(L(ep_l, False), [-1.0, 0.0, -1.0, -1.0]))
    # credit is BOUNDED by the progress gain: ep .65 vs stay .6 -> (.65 - .6) / .2 = 0.25 ; ep .55 < stay -> 0
    t("L_bounded_by_gain", lambda: close(L([0.65, 0.6, 0.95, 0.5], True)[0], 0.25) and close(L([0.55, 0.6, 0.95, 0.5], True)[0], 0.0))
    # the clip: positive 1 with a dither flag -> 0 ; an unlabelled change with a dither flag -> -2 clipped to -1
    # (a stay-in-lane reference proposal with ep .6 is present, so the change's gain is clip((.9 - .6) / .2) = 1)
    t("L_dither_and_clip", lambda: close(R.lane_change_term(np.array([True, False]), np.array(True), np.array([1.0, 0.0]), np.array([0.9, 0.6]), np.array([True, True])), [0.0, 0.0])
      and close(R.lane_change_term(np.array([True, False]), np.array(False), np.array([1.0, 0.0]), np.array([0.9, 0.6]), np.array([True, True])), [-1.0, 0.0]))
    # no safe stay-in-lane proposal => no reference => no credit
    t("L_no_stay_reference", lambda: close(R.lane_change_term(np.array([True, True]), np.array(True), np.zeros(2), np.array([0.9, 0.8]), np.array([True, True])), [0.0, 0.0]))
    cl = R.RewardCfg(lam_d=0.0, guard=False, use_l=True)
    lane = {"lc_executed": lcx, "label_fires": np.array(True), "dither": dith}
    t("L_enters_reward", lambda: close(R.reward(tgl, cl, lane=lane)["R"], R.reward(tgl, R.RewardCfg(lam_d=0.0, guard=False))["R"] + 0.05 * np.array([1.0, 0.0, 0.0, -1.0])))
    t("K_L_off_ignores_inputs", lambda: np.array_equal(R.reward(tgl, R.RewardCfg(lam_d=0.0, guard=False), k=np.ones(4), lane=lane)["R"],
                                                        R.reward(tgl, R.RewardCfg(lam_d=0.0, guard=False))["R"]))
    return bad


# (name, term, old snippet, new snippet) -- each snippet must occur EXACTLY once in rl_reward.py
MUTATIONS = [
    ("pdms_ttc_weight_5_to_4", "PDMS", "V1_W = (5.0, 5.0, 2.0)", "V1_W = (5.0, 4.0, 2.0)"),
    ("lam_d_sign_flipped", "D", "r = r - cfg.lam_d * dd", "r = r + cfg.lam_d * dd"),
    ("lam_d_default_0", "D", "lam_d: float = 0.15", "lam_d: float = 0.0"),
    ("D_saturation_removed", "D", "np.minimum(g / scale, 1.0).mean(axis=-1)", "(g / scale).mean(axis=-1)"),
    ("guard_applied_to_unsafe", "g2", "return np.where(safe & has, a, 0.0)", "return np.where(has, a, 0.0)"),
    ("guard_reference_includes_unsafe", "g2", "ref = np.where(safe, ep, -np.inf).max(axis=-1, keepdims=True)", "ref = ep.max(axis=-1, keepdims=True)"),
    ("guard_sign_flipped", "g2", "a = np.clip(ep - np.where(has, ref, 0.0), -1.0, 0.0)", "a = np.clip(np.where(has, ref, 0.0) - ep, 0.0, 1.0)"),
    ("w_prog_default_0", "g2", "w_prog: float = 0.5", "w_prog: float = 0.0"),
    ("P4_cap_removed", "g2", "return np.where(safe, -np.minimum(-raw, cap), 0.0)", "return np.where(safe, raw, 0.0)"),
    ("P4_cap_keep_1_ties", "g2", "keep: float = 0.5", "keep: float = 1.0"),
    ("P4_cap_counts_safe_rivals", "g2", "(~safe)[..., None, :] & (bj < bi)", "(bj < bi)"),
    ("guard_default_off", "g2", "guard: bool = True", "guard: bool = False"),
    ("guard_flag_ignored", "g2", "if cfg.guard:", "if True:"),
    ("safe_requires_ttc_dropped", "g2", "& (ttc >= 1.0 - tol)", ""),
    ("K_sign_flipped", "K", "r = r - cfg.lam_k * kk", "r = r + cfg.lam_k * kk"),
    ("K_intersection_mask_dropped", "K", "out = ~np.asarray(in_intersection, bool)", "out = np.ones_like(o, dtype=bool)"),
    ("K_lane_change_exemption_dropped", "K", "(out & (o < near))", "out"),
    ("K_straddle_weight_0", "K", "w=(0.6, 0.4, 0.2)", "w=(0.6, 0.4, 0.0)"),
    ("K_default_on", "K", "use_k: bool = False", "use_k: bool = True"),
    ("L_sign_flipped", "L", "r = r + cfg.lam_l * ll", "r = r - cfg.lam_l * ll"),
    ("L_dither_dropped", "L", "- np.asarray(dither, np.float64), -1.0, 1.0)", "- 0.0 * np.asarray(dither, np.float64), -1.0, 1.0)"),
    ("L_unlabelled_unpenalised", "L", "neg = (lc & ~lab) * 1.0", "neg = (lc & ~lab) * 0.0"),
    ("L_credit_without_safety", "L", "pos = (lc & safe & lab) * gain", "pos = (lc & lab) * gain"),
    ("L_credit_without_label", "L", "pos = (lc & safe & lab) * gain", "pos = (lc & safe) * gain"),
    ("L_credit_unbounded_by_gain", "L", "pos = (lc & safe & lab) * gain", "pos = (lc & safe & lab) * 1.0"),
    ("L_stay_reference_includes_changes", "L", "stay = np.where(safe & ~lc, ep, -np.inf)", "stay = np.where(safe, ep, -np.inf)"),
    ("L_clip_removed", "L", "return np.clip(pos - neg - np.asarray(dither, np.float64), -1.0, 1.0)", "return pos - neg - np.asarray(dither, np.float64)"),
    ("L_default_on", "L", "use_l: bool = False", "use_l: bool = True"),
    ("silent_zero_when_d_missing", "reach", "if d is None:", "if d is None:\n            d = np.zeros_like(p)\n        if False:"),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    n = 0
    real = suite(load(src, "rl_reward_real"))
    assert not real, f"the unmutated module FAILS its own checks: {real}"
    n += 1
    print(f"{'mutation':40s} {'term':6s} caught-by (failing checks)")
    for i, (name, term, old, new) in enumerate(MUTATIONS):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        try:
            mod = load(src.replace(old, new), f"rl_reward_mut{i}")
        except SyntaxError as e:                                    # a mutation that does not compile is not a RED test of the logic
            raise AssertionError(f"mutation {name} does not compile: {e}")
        try:
            failing = suite(mod)
        except Exception as e:                                      # noqa: BLE001 -- the mutated default makes the suite's own setup raise: RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:60]}"]
        assert failing, f"mutation {name} ({term}) stayed GREEN: the suite cannot see it"
        print(f"{name:40s} {term:6s} {', '.join(failing[:4])}")
        n += 1
    # every term has at least one mutation
    assert {t for _, t, _, _ in MUTATIONS} >= {"PDMS", "D", "g2", "K", "L", "reach"}
    n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
