"""brake_locus_reroll_analyze_r8.py -- the ANALYSIS of the banked brake-locus re-roll (numpy only; NO model, NO GPU).  Reads the chunked sidecars written by
`brake_locus_reroll_r8.py roll` and applies, WITHOUT changing any threshold, the rules sealed in PREREG_BRAKE_LOCUS.md + AMENDMENTS 1-4:
  E (section 4 of the original; `brake_locus_r8.fan_cells`) and its label (section 5); ALONG-RAND (A1 A3; `along_stats`, `along_r_diff_boot`, `along_verdict`);
  the hedge arms H1 / H2 (AMENDMENT 3); the intervention arms F2 / S0a / S0b with their gates (AMENDMENT 4); the R6 controls of the design (section 5).

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 brake_locus_reroll_analyze_r8.py [--tag full] [--selftest-only]     # writes <PKG>/raw/brake_reroll/BRAKE_REROLL.{json,md}

A section whose windows / arms are not banked yet is reported as NOT_AVAILABLE with the counts, never as a result.  PARTIAL runs are stamped PARTIAL.
Tier T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  Estimator: episode-cluster bootstrap B 2000 seed 0 (taniteval.ci draws), answers
"another draw of EPISODES" only; the inference-draw question is answered only by N2 / the second draw N_s1; NOT another training run (single seed, H-ESTIM-SEED-1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brake_locus_r8 as BL  # noqa: E402
import brake_locus_reroll_r8 as M  # noqa: E402

FL = M.FL
S6_SLOTS = M.S6_SLOTS
N_BOOT = 2000
EXCESS_R8_R7 = 0.265625 - 0.09105603448275862            # literal: r8 - r7 false-decel, seed 0 (L2_HEADROOM.json); AMENDMENT 3
HEDGE_BAR = 0.5 * EXCESS_R8_R7
W_FB250, W_CT100 = 493 / 250, 1363 / 100                  # AMENDMENT 3 weights restoring the banked GT-cruise population
W_FB, W_CT = 1.0, 1363 / 250                              # stage-A weights (all 493 FB windows, 250 CT)
SEEDS_DRAW = {"draw0": "N", "draw1": "N_s1"}


# ------------------------------------------------------------------------------------------ estimator
def boot_ctx(eid):
    import s6_lon_decomp as S6
    C, ei, n = S6.boot_counts([str(x) for x in eid], N_BOOT, 0)
    return C, ei, n


def wmean_ci(x, w, eid) -> dict:
    """weighted mean of x (ratio of resampled episode sums) + episode-cluster 95 % interval; NaN rows dropped."""
    x, w, eid = np.asarray(x, float), np.asarray(w, float), np.asarray(eid)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return {"point": None, "lo": None, "hi": None, "n": int(ok.sum()), "n_episodes": 0}
    x, w, eid = x[ok], w[ok], eid[ok]
    C, ei, n_ep = boot_ctx(eid)
    num, den = np.bincount(ei, w * x, n_ep), np.bincount(ei, w, n_ep)
    d = C @ den
    keep = d > 0
    est = (C @ num)[keep] / d[keep]
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {"point": float(num.sum() / den.sum()), "lo": float(lo), "hi": float(hi), "n": int(len(x)), "n_episodes": int(n_ep)}


def pair_ci(xa, xb, w, eid) -> dict:
    return wmean_ci(np.asarray(xa, float) - np.asarray(xb, float), w, eid)


def g_diff_boot(d1, d2, h, eid) -> dict:
    """paired episode-cluster bootstrap of g*(d1) - g*(d2) (the same shared resample matrix as along_r_diff_boot)."""
    C, ei, n_ep = boot_ctx(eid)
    a, b = BL._ep_sums(d1, h, eid), BL._ep_sums(d2, h, eid)
    ga, gb = (C @ a[0]) / (C @ a[1]), (C @ b[0]) / (C @ b[1])
    pt = a[0].sum() / a[1].sum() - b[0].sum() / b[1].sum()
    lo, hi = np.percentile(ga - gb, [2.5, 97.5])
    return {"delta_g": float(pt), "lo": float(lo), "hi": float(hi), "separated_above": bool(lo > 0)}


# ------------------------------------------------------------------------------------------ the pure rules (selftested)
def hedge_label(dfd, dmd) -> str:
    """AMENDMENT 3, committed.  dfd = CI dict of dFD = FD_N - FD_X (positive = X reduces the false brake); dmd = CI dict of dMD = MD_X - MD_N."""
    if dfd["point"] is None or dmd["point"] is None:
        return "NOT_AVAILABLE"
    if dfd["lo"] <= 0.0 <= dfd["hi"]:
        return "NO-EFFECT"
    if dfd["point"] >= HEDGE_BAR and dfd["lo"] > 0.0 and dmd["lo"] <= 0.0:
        return "HEDGE-WORKS"
    return "MIXED"


def closure_label(a) -> str:
    """AMENDMENT 4: a = CI dict of a_X = c_X - c_N2 (paired)."""
    if a["point"] is None:
        return "NOT_AVAILABLE"
    if a["point"] >= 0.50 and a["lo"] > 0.0:
        return "CLOSES"
    if a["hi"] < 0.25:
        return "NOT-CLOSES"
    return "INCONCLUSIVE"


def combine_closures(f2, s0a, s0b) -> str:
    L = {"F2": f2, "S0a": s0a, "S0b": s0b}
    if "NOT_AVAILABLE" in L.values():
        return "NOT_AVAILABLE"
    if s0a == "CLOSES":
        return "PROGRESS-CONSTRAINT-IS-THE-LOCUS"
    if s0b == "CLOSES":
        return "HEAD-COUPLED-RANKING-TERMS-OTHER-THAN-PROGRESS"
    if f2 == "CLOSES" and s0a == "NOT-CLOSES" and s0b == "NOT-CLOSES":
        return "HEAD-ACTS-OUTSIDE-RANKING"
    if f2 == s0a == s0b == "NOT-CLOSES":
        return "PICK-IS-MADE-BY-THE-CORE-RANKING"
    if "INCONCLUSIVE" in L.values():
        return "INCONCLUSIVE"
    return "MIXED"


def selftest() -> dict:
    ci = lambda p, lo, hi: {"point": p, "lo": lo, "hi": hi}
    r = {}
    r["hedge_no_effect"] = hedge_label(ci(0.10, -0.02, 0.2), ci(0.0, -0.01, 0.01)) == "NO-EFFECT"
    r["hedge_works"] = hedge_label(ci(0.10, 0.05, 0.15), ci(0.0, -0.01, 0.01)) == "HEDGE-WORKS"
    r["hedge_small_separated_is_mixed"] = hedge_label(ci(0.05, 0.02, 0.08), ci(0.0, -0.01, 0.01)) == "MIXED"
    r["hedge_missed_worse_is_mixed"] = hedge_label(ci(0.10, 0.05, 0.15), ci(0.03, 0.01, 0.05)) == "MIXED"
    r["hedge_bar_is_literal"] = abs(HEDGE_BAR - 0.08728448275862069) < 1e-12
    r["closure_closes"] = closure_label(ci(0.6, 0.3, 0.9)) == "CLOSES"
    r["closure_not"] = closure_label(ci(0.05, -0.1, 0.2)) == "NOT-CLOSES"
    r["closure_inconclusive"] = closure_label(ci(0.4, 0.1, 0.7)) == "INCONCLUSIVE"
    r["combine"] = (combine_closures("CLOSES", "NOT-CLOSES", "NOT-CLOSES") == "HEAD-ACTS-OUTSIDE-RANKING"
                    and combine_closures("NOT-CLOSES", "CLOSES", "NOT-CLOSES") == "PROGRESS-CONSTRAINT-IS-THE-LOCUS"
                    and combine_closures("NOT-CLOSES", "NOT-CLOSES", "NOT-CLOSES") == "PICK-IS-MADE-BY-THE-CORE-RANKING"
                    and combine_closures("NOT-CLOSES", "NOT-CLOSES", "CLOSES") == "HEAD-COUPLED-RANKING-TERMS-OTHER-THAN-PROGRESS"
                    and combine_closures("CLOSES", "INCONCLUSIVE", "NOT-CLOSES") == "INCONCLUSIVE")
    rng = np.random.default_rng(3)                       # estimator known values: a constant has a zero-width CI; a 0/1 split reads its share
    eid = np.repeat(np.arange(30), 10)
    c = wmean_ci(np.full(300, 0.25), np.ones(300), eid)
    r["wmean_constant_zero_width"] = bool(c["point"] == 0.25 and c["lo"] == 0.25 and c["hi"] == 0.25)
    x = (rng.random(300) < 0.4).astype(float)
    c2 = wmean_ci(x, np.ones(300), eid)
    r["wmean_share_contains_truth"] = bool(c2["lo"] < 0.4 < c2["hi"] and abs(c2["point"] - x.mean()) < 1e-12)
    cw = wmean_ci(np.r_[np.ones(100), np.zeros(200)], np.r_[np.full(100, 3.0), np.ones(200)], eid)   # weights: (100*3*1)/(100*3+200)
    r["wmean_weights"] = bool(abs(cw["point"] - 300 / 500) < 1e-12)
    gd = g_diff_boot(rng.normal(size=(300, 4)), rng.normal(size=(300, 4)), rng.normal(size=(300, 4)), eid)
    r["g_diff_runs"] = bool(np.isfinite(gd["delta_g"]))
    r["ALL_PASS"] = bool(all(r.values()))
    return r


# ------------------------------------------------------------------------------------------ data
def cls_of(P6, v0):
    import cruise_brake_retime_r8 as CB
    return CB.plan_class_thr(np.asarray(P6, np.float64), np.asarray(v0, np.float64), 1.0, 1.5)


class Data:
    """banked S6 tree + deploy npz + the plan + the re-roll bank, joined by window key."""

    def __init__(self, tag):
        self.seals = M.check_seals()
        self.plan = FL.jload(M.SCR / "plan_raw.json")
        self.W = self.plan["windows"]
        self.B = M.load_bank(tag)
        self.T = BL.load_tree(f"{BL.ARCHIVE}/{BL.TREE}", 0, True)
        self.z = np.load(M.DEP / f"DEPLOY8_POS_{M.TREE}_s0.npz", allow_pickle=True)
        T, z = self.T, self.z
        self.comp = np.asarray(z["S6_compliant"], bool)
        self.gtc = cls_of(T["g"], T["v0"])
        self.c8 = cls_of(T["os"], T["v0"])
        self.idx = {a: {str(k): i for i, k in enumerate(v["keys"])} for a, v in self.B.items()}

    def keys_of(self, setname, arm):
        return [k for k, w in self.W.items() if setname in w["sets"] and k in self.idx.get(arm, {})]

    def get(self, arm, f, keys):
        """stacked field for `keys` (NaN / -1 filled where the arm is absent) + presence mask."""
        have = np.array([k in self.idx.get(arm, {}) for k in keys])
        ref = next((self.B[arm][f][0] for k in keys if k in self.idx.get(arm, {})), None) if arm in self.B and f in self.B[arm] else None
        if ref is None:
            return None, have
        out = np.full((len(keys),) + ref.shape, np.nan if ref.dtype.kind == "f" else -1, dtype=(np.float64 if ref.dtype.kind == "f" else np.int64))
        for j, k in enumerate(keys):
            if have[j]:
                out[j] = self.B[arm][f][self.idx[arm][k]]
        return out, have

    def rows6(self, keys):
        return np.array([self.W[k]["row6"] for k in keys])

    def eid(self, keys):
        return np.array([str(self.W[k]["ep_index"]) for k in keys])


# ------------------------------------------------------------------------------------------ sections
def sec_controls(D) -> dict:
    out = {}
    bad_sel, bad_arg, n_chk = 0, 0, 0
    for a, fs in D.B.items():
        if "fan" not in fs:
            continue
        for i in range(len(fs["keys"])):
            fan, tj, si = fs["fan"][i].astype(np.float64), fs["traj"][i].astype(np.float64), int(fs["sel_idx"][i])
            n_chk += 1
            bad_sel += int(np.abs(fan[si] - tj).max() > 6e-4 * max(1.0, np.abs(tj).max()) + 1e-3)
            sc = fs["sel_score_v3"][i].astype(np.float64)
            bad_arg += int(int(np.argmax(np.where(fs["reach_keep"][i].astype(bool), sc, -np.inf))) != si)
    out["selected_equals_emitted_and_argmax"] = {"n_arm_records": n_chk, "n_selected_ne_emitted": bad_sel, "n_sel_idx_ne_argmax": bad_arg, "PASS": bool(n_chk > 0 and bad_sel == 0 and bad_arg == 0)}
    ks = [k for k in D.idx.get("S0id", {}) if k in D.idx.get("N", {})]
    if ks:
        d = [float(np.abs(D.B["S0id"]["traj"][D.idx["S0id"][k]].astype(np.float64) - D.B["N"]["traj"][D.idx["N"][k]].astype(np.float64)).max()) for k in ks]
        out["S0id_equals_N"] = {"n": len(ks), "max_abs_diff_m": max(d), "PASS": bool(max(d) <= 1e-4)}
    ks = [k for k in D.idx.get("S0x", {}) if k in D.idx.get("N", {})]
    if ks:
        mv = sum(int(D.B["S0x"]["sel_idx"][D.idx["S0x"][k]]) != int(D.B["N"]["sel_idx"][D.idx["N"][k]]) for k in ks)
        out["S0x_moved_pick"] = {"n": len(ks), "n_moved": int(mv), "PASS": bool(mv >= 1)}
    flp = FL.jload("E:/refcv8_eval_kit/force_lon_scratch/roll_full.json")["windows"]
    dd = [float(np.abs(np.asarray(flp[k]["arms"]["N"]["traj"], np.float64) - D.B["N"]["traj"][D.idx["N"][k]].astype(np.float64)).max()) for k in D.idx.get("N", {}) if k in flp]
    out["N_vs_force_lon_banked_N"] = {"n": len(dd), "max_abs_diff_m": (max(dd) if dd else None), "note": "same CPU code path and per-window seed; fp32 reduction order may differ with OMP"}
    # a_star of the re-rolled unrefined bank vs the banked a_star (the bank is model-free of the sampler: expected ~100 %)
    ks = [k for k in D.idx.get("N", {}) if D.W.get(k, {}).get("row6", -1) >= 0]
    if ks:
        eq = 0
        for k in ks:
            r6 = D.W[k]["row6"]
            gt8 = D.z["S6_pos8"][r6].astype(np.float64)
            bank = D.B["N"]["bank"][D.idx["N"][k]].astype(np.float64)
            eq += int(int(np.argmin(((bank - gt8[None]) ** 2).sum((-1, -2)))) == int(D.T["a_star"][r6]))
        out["a_star_from_rerolled_bank_equals_banked"] = {"n": len(ks), "share_equal": eq / len(ks), "note": "fp16 bank storage can flip near-ties"}
    out["ALL_PASS_gated"] = bool(all(v.get("PASS", True) for v in out.values() if isinstance(v, dict)))
    return out


def sec_E(D) -> dict:
    keys = [k for k in D.keys_of("FB", "N")] + [k for k in D.keys_of("CT", "N")]
    keys = sorted(set(keys))
    out = {"n_windows_with_N": len(keys), "n_planned": sum(1 for w in D.W.values() if "FB" in w["sets"] or "CT" in w["sets"])}
    if len(keys) < 10:
        out["status"] = "NOT_AVAILABLE"
        return out
    r6 = D.rows6(keys)
    F, _ = D.get("N", "fan", keys)
    F = F[:, :, S6_SLOTS]
    keep, _ = D.get("N", "reach_keep", keys)
    kind, _ = D.get("N", "kind", keys)
    tj, _ = D.get("N", "traj", keys)
    G, v0 = D.T["g"][r6].astype(np.float64), D.T["v0"][r6]
    ccl = cls_of(tj[:, S6_SLOTS], v0)
    gt_cruise = D.comp[r6] & (D.gtc[r6] == "cruise")
    fb = gt_cruise & (ccl == "decel")
    ctl = gt_cruise & (ccl != "decel")
    kp = keep.astype(bool)
    eid = D.eid(keys)
    res = {}
    for nm, kk, use_ade, x in (("primary_reach_FDE6_2.91m", kp, False, BL.X_FDE6_M), ("all149", None, False, BL.X_FDE6_M), ("base117_reach", kp & (kind[:, :] == 0), False, BL.X_FDE6_M),
                               ("ADE_1.21m_reach", kp, True, BL.X_ADE_M)):
        fc = BL.fan_cells(F, G, v0, fb, kk, x_m=x, use_ade=use_ade)
        fcc = BL.fan_cells(F, G, v0, ctl, kk, x_m=x, use_ade=use_ade)
        e_ci = wmean_ci(fc["E_w"].astype(float)[fb], np.ones(int(fb.sum())), eid[fb]) if fb.sum() > 1 else None
        res[nm] = {"E": fc["E"], "E_ci": e_ci, "n_fb": fc["n_fb"], "E_ctrl": fcc["E"], "n_ctrl": fcc["n_fb"]}
    head = FL.jload(M.HEAD / "BRAKE_LOCUS.json")["head_test"]["HEAD_TEST"]
    e = res["primary_reach_FDE6_2.91m"]["E"]
    out.update({"status": "OK", "n_fb_by_rerolled_N": int(fb.sum()), "n_gt_cruise_rolled": int(gt_cruise.sum()), "cells": res, "HEAD_TEST_banked": head,
                "label": BL.final_label(head == "PASS", e), "rule": f"E < {BL.E_CAND} CANDIDATE, E >= {BL.E_SEL} SELECTION (original section 4-5); label stands as written"})
    # R6: banked class vs re-rolled class
    b = D.comp[r6] & (D.gtc[r6] == "cruise") & (D.c8[r6] == "decel")
    out["flip_table_banked_vs_rerolled_on_GT_cruise"] = {"banked_FB_now_decel": int((b & fb).sum()), "banked_FB_now_not": int((b & ~fb & gt_cruise).sum()),
                                                         "banked_nonFB_now_decel": int((~b & fb).sum()), "banked_nonFB_now_not": int((~b & gt_cruise & ~fb).sum())}
    w = np.where(np.array(["FB" in D.W[k]["sets"] for k in keys]), W_FB, W_CT)
    fdn = wmean_ci(fb[gt_cruise].astype(float), w[gt_cruise], eid[gt_cruise])
    out["R6_rerolled_false_decel_weighted_stageA"] = {**fdn, "banked": BL.LIT["fd_r8"][0]}
    return out


def arc(P):
    return BL.arc_len(P)


def sec_along(D) -> dict:
    keys = sorted(k for k in D.idx.get("N_s1", {}) if k in D.idx.get("N", {}) and D.W[k]["row2"] >= 0)
    out = {"n_windows_both_draws": len(keys)}
    if len(keys) < 20:
        out["status"] = "NOT_AVAILABLE"
        return out
    r2 = np.array([D.W[k]["row2"] for k in keys])
    H, H8 = D.z["S2_pos"][r2].astype(np.float64), D.z["S2_pos8"][r2].astype(np.float64)
    eid = D.eid(keys)
    sH = arc(H)
    res = {"status": "OK", "draws": {}}
    diffs, gdiffs = {}, {}
    for dn, arm in SEEDS_DRAW.items():
        F, _ = D.get(arm, "fan", keys)
        P, _ = D.get(arm, "prior_path", keys)
        os4, _ = D.get(arm, "traj", keys)
        kp, _ = D.get(arm, "reach_keep", keys)
        kp = kp.astype(bool)
        sP = arc(P[:, :4])
        h = sH - sP
        rng = np.random.default_rng(M.ALONG_SEED)
        n = len(keys)
        Fa = F[:, :, :4]                                     # [W,149,4,2]
        arcs = np.stack([arc(Fa[:, c]) for c in range(Fa.shape[1])], 1)          # [W,149,4]
        pick_rand, pick_med, pick_orc = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        for i in range(n):
            ids = np.nonzero(kp[i])[0] if kp[i].any() else np.arange(Fa.shape[1])
            pick_rand[i] = ids[rng.integers(len(ids))]
            o = ids[np.argsort(arcs[i, ids, 3], kind="stable")]
            pick_med[i] = o[(len(o) - 1) // 2]                                      # lower median, ties to the lowest index of the stable sort
            pick_orc[i] = ids[int(np.argmin(((F[i, ids] - H8[i][None]) ** 2).sum((-1, -2))))]
        take = lambda pk: arcs[np.arange(n), pk]
        s_os = arc(os4[:, :4])
        dX = {"os": s_os - sP, "RAND": take(pick_rand) - sP, "MED": take(pick_med) - sP, "ORC": take(pick_orc) - sP}
        stats = {a: BL.along_stats(d, h, eid) for a, d in dX.items()}
        stats["os_banked_GPU_same_windows"] = BL.along_stats(arc(D.z["S2_os"][r2].astype(np.float64)) - arc(D.z["S2_prior8"][r2][:, :4].astype(np.float64)), h, eid)
        dR = BL.along_r_diff_boot(dX["os"], dX["RAND"], h, eid)
        dRm = BL.along_r_diff_boot(dX["os"], dX["MED"], h, eid)
        diffs[dn] = dR
        gdiffs[dn] = g_diff_boot(dX["os"], dX["RAND"], h, eid)
        spread = []
        ids_of = [np.nonzero(kp[i])[0] if kp[i].any() else np.arange(Fa.shape[1]) for i in range(n)]
        for rep in range(20):
            rr = np.random.default_rng(M.ALONG_SEED + 1 + rep)
            pk = np.array([ids_of[i][rr.integers(len(ids_of[i]))] for i in range(n)])
            spread.append(BL.along_stats(take(pk) - sP, h, eid)["r"])
        bok = {}
        for k_ in (1, 2, 4, 8, 16, 32, "all"):
            rr = np.random.default_rng(M.ALONG_SEED + 99)
            pk = np.zeros(n, int)
            for i in range(n):
                ids = ids_of[i]
                sub = ids if k_ == "all" else rr.choice(ids, size=min(int(k_), len(ids)), replace=False)
                pk[i] = sub[int(np.argmin(((F[i, sub] - H8[i][None]) ** 2).sum((-1, -2))))]
            bok[str(k_)] = BL.along_stats(take(pk) - sP, h, eid)["r"]
        res["draws"][dn] = {"arm": arm, "stats": stats, "dR_os_minus_RAND": dR, "dR_os_minus_MED": dRm, "dG_os_minus_RAND": gdiffs[dn], "RAND_r_spread_20_draws": [min(spread), max(spread)], "best_of_k_r": bok,
                            "prior_vs_banked_prior_max_abs_m": float(np.abs(P[:, :4] - D.z["S2_prior8"][r2][:, :4]).max())}
    res["verdict_RAND_r"] = BL.along_verdict(diffs)
    res["verdict_RAND_gstar"] = BL.along_verdict({k: {"lo": v["lo"]} for k, v in gdiffs.items()})
    res["reading"] = ("SELECTOR-ADDS-VALUE iff the CI lower bound of dR > 0 at BOTH draws; SELECTION-SUPPORTED iff not at both; else SEED-SPLIT (A1 A3). os above RAND is the EXPECTED outcome; the best-of-k curve bounds the by-construction part of ORC. "
                      "The two draws are CPU re-roll draws (offsets 0 / +1), NOT the banked GPU seeds 0 / 1.")
    return {**out, **res}


def sec_interventions(D) -> dict:
    out = {}
    keys = sorted(set(D.keys_of("FB250", "N")) | set(D.keys_of("CT100", "N")) | set(D.keys_of("DC100", "N")))
    if not keys:
        return {"status": "NOT_AVAILABLE", "reason": "no native N windows of FB250 / CT100 / DC100 banked yet"}
    r6 = D.rows6(keys)
    v0, G, r7 = D.T["v0"][r6], D.T["g"][r6].astype(np.float64), D.T["b_r7"][r6].astype(np.float64)
    eid = D.eid(keys)
    st = np.array(["FB250" if "FB250" in D.W[k]["sets"] else ("CT100" if "CT100" in D.W[k]["sets"] else "DC100") for k in keys])
    w = np.where(st == "FB250", W_FB250, np.where(st == "CT100", W_CT100, 1.0))
    gtcr = D.gtc[r6]

    def arm_cells(arm):
        tj, have = D.get(arm, "traj", keys)
        if tj is None:
            return None
        P6 = np.where(have[:, None, None], tj[:, S6_SLOTS], np.nan)
        dec = np.where(have, (cls_of(np.nan_to_num(P6), v0) == "decel").astype(float), np.nan)
        ade = np.where(have, np.linalg.norm(P6 - G, axis=-1).mean(-1), np.nan)
        return {"P6": P6, "dec": dec, "ade": ade, "have": have}
    N = arm_cells("N")
    cells = {a: arm_cells(a) for a in ("N", "N2", "H1", "H2", "F2", "Fbrake", "S0a", "S0b", "S0id", "S0x")}
    gc = np.isin(st, ("FB250", "CT100"))
    dcm = st == "DC100"
    ade7 = np.linalg.norm(r7 - G, axis=-1).mean(-1)
    out["counts"] = {a: {s: int(((st == s) & c["have"]).sum()) for s in ("FB250", "CT100", "DC100")} for a, c in cells.items() if c is not None}

    def arm_vs_n(a):
        c = cells.get(a)
        if c is None or not (c["have"] & gc).any():
            return {"status": "NOT_AVAILABLE"}
        both = c["have"] & N["have"]
        r = {"status": "OK"}
        dfd = pair_ci(np.where(both & gc, N["dec"], np.nan), np.where(both & gc, c["dec"], np.nan), w, eid)
        miss = lambda cc: np.where(cc["have"] & dcm, 1.0 - cc["dec"], np.nan)
        dmd = pair_ci(np.where(both & dcm, miss(c), np.nan), np.where(both & dcm, miss(N), np.nan), np.ones(len(keys)), eid) if (both & dcm).any() else {"point": None, "lo": None, "hi": None}
        r.update({"FD_N": wmean_ci(np.where(both & gc, N["dec"], np.nan), w, eid), "FD_X": wmean_ci(np.where(both & gc, c["dec"], np.nan), w, eid), "dFD_N_minus_X": dfd, "dMD_X_minus_N": dmd})
        for s in ("FB250", "CT100", "DC100"):
            m = both & (st == s)
            if m.any():
                chg = np.abs(c["P6"][m] - N["P6"][m]).max((-1, -2)) > 1e-4
                r[f"{s}"] = {"n": int(m.sum()), "share_plan_changed_vs_N": float(chg.mean()), "ADE_X_minus_N": pair_ci(np.where(m, c["ade"], np.nan), np.where(m, N["ade"], np.nan), np.ones(len(keys)), eid),
                             "ADE_X_minus_r7": pair_ci(np.where(m, c["ade"], np.nan), np.where(m, ade7, np.nan), np.ones(len(keys)), eid)}
        return r
    arms = {a: arm_vs_n(a) for a in ("N2", "H1", "H2")}
    floor = arms["N2"]
    for a in ("H1", "H2"):
        if arms[a].get("status") == "OK":
            arms[a]["label_AMENDMENT3"] = hedge_label(arms[a]["dFD_N_minus_X"], arms[a]["dMD_X_minus_N"])
            if floor.get("status") == "OK":
                f = floor["dFD_N_minus_X"]
                arms[a]["ABOVE_N2_FLOOR"] = bool(abs(arms[a]["dFD_N_minus_X"]["point"]) > max(abs(f["lo"]), abs(f["hi"])))
    out["hedge_AMENDMENT3"] = {"bar_dFD": HEDGE_BAR, "excess_r8_minus_r7_literal": EXCESS_R8_R7, "arms": arms,
                               "reading_caveat": "strata weights FB250 493/250, CT100 1363/100 restore the banked GT-cruise population (ESTIMATED approximation); one CPU draw; N2 floor beside"}
    # ---- AMENDMENT 4: closures on FB-N windows
    fbn = (st == "FB250") & N["have"] & (N["dec"] == 1.0)
    out["n_FB250_with_N"] = int(((st == "FB250") & N["have"]).sum())
    out["n_FB_N_decel_under_rerolled_native"] = int(fbn.sum())
    nd = lambda c: np.where(c["have"] & fbn, 1.0 - c["dec"], np.nan)
    gate = {"S0id_equals_N": D_ctl_cache.get("S0id_equals_N", {}).get("PASS"), "S0x_moved_pick": D_ctl_cache.get("S0x_moved_pick", {}).get("PASS")}
    if cells["Fbrake"] is not None and (cells["Fbrake"]["have"] & (st == "CT100")).any():
        both = cells["Fbrake"]["have"] & N["have"] & (st == "CT100")
        pc = pair_ci(np.where(both, cells["Fbrake"]["dec"], np.nan), np.where(both, N["dec"], np.nan), np.ones(len(keys)), eid)
        gate["Fbrake_raises_decel_on_CT100"] = {**pc, "PASS": bool(pc["point"] is not None and pc["lo"] > 0)}
    cl = {}
    for a in ("F2", "S0a", "S0b", "N2"):
        c = cells[a]
        if c is None or not (c["have"] & fbn).any():
            cl[a] = {"status": "NOT_AVAILABLE"}
            continue
        cl[a] = {"c_X": wmean_ci(nd(c), np.ones(len(keys)), eid)}
        if a != "N2" and cells["N2"] is not None and (cells["N2"]["have"] & c["have"] & fbn).any():
            both = c["have"] & cells["N2"]["have"] & fbn
            cl[a]["a_X_minus_N2"] = pair_ci(np.where(both, nd(c), np.nan), np.where(both, nd(cells["N2"]), np.nan), np.ones(len(keys)), eid)
            cl[a]["label"] = closure_label(cl[a]["a_X_minus_N2"])
        elif a != "N2":
            cl[a]["label"] = "NOT_AVAILABLE"
    comb = combine_closures(cl.get("F2", {}).get("label", "NOT_AVAILABLE"), cl.get("S0a", {}).get("label", "NOT_AVAILABLE"), cl.get("S0b", {}).get("label", "NOT_AVAILABLE"))
    void = not all(v is True or (isinstance(v, dict) and v.get("PASS") is True) for v in gate.values())
    out["interventions_AMENDMENT4"] = {"gate": gate, "closures": cl, "combination": ("VOID (a gate failed or is not available)" if void else comb), "combination_if_gates_pass": comb}
    return out


D_ctl_cache: dict = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--selftest-only", action="store_true")
    a = ap.parse_args()
    st = selftest()
    print("selftest", json.dumps(st))
    if not st["ALL_PASS"]:
        return 1
    if a.selftest_only:
        return 0
    D = Data(a.tag)
    plan_chunks = M.chunk_state(D.plan, a.tag)
    J = {"tool": "code/brake_locus_reroll_analyze_r8.py", "tier": M.TIER, "generated": time.strftime("%FT%T%z"), "seals": D.seals, "tag": a.tag, "selftest": st,
         "estimator": BL.ESTIMATOR, "coverage": {s: {"windows_done": sum(r["done"] for r in plan_chunks if r["stage"] == s), "windows_planned": sum(r["n"] for r in plan_chunks if r["stage"] == s)} for s in M.STAGES},
         "arms_banked": {k: int(len(v["keys"])) for k, v in D.B.items()}}
    J["PARTIAL"] = any(v["windows_done"] < v["windows_planned"] for v in J["coverage"].values())
    D_ctl = sec_controls(D)
    D_ctl_cache.update(D_ctl)
    J["controls_R6"] = D_ctl
    J["E_and_label"] = sec_E(D)
    J["ALONG_RAND"] = sec_along(D)
    J["interventions_and_hedge"] = sec_interventions(D)
    M.OUT.mkdir(parents=True, exist_ok=True)
    FL.jdump(J, M.OUT / "BRAKE_REROLL.json")
    lab = J["E_and_label"].get("label", {}).get("label", "NOT_AVAILABLE")
    iv = J["interventions_and_hedge"]
    md = [f"# BRAKE_REROLL (generated {J['generated']}; tag {a.tag}; {'PARTIAL' if J['PARTIAL'] else 'COMPLETE'})", "", f"Tier: {M.TIER}.  Estimator: {BL.ESTIMATOR}", "",
          f"* coverage: {json.dumps(J['coverage'])}", f"* controls R6 gated PASS: {D_ctl.get('ALL_PASS_gated')}", f"* label (E): {lab}  E = {J['E_and_label'].get('cells', {}).get('primary_reach_FDE6_2.91m', {}).get('E')}",
          f"* ALONG-RAND verdict (r): {J['ALONG_RAND'].get('verdict_RAND_r', 'NOT_AVAILABLE')}  (g*): {J['ALONG_RAND'].get('verdict_RAND_gstar', 'NOT_AVAILABLE')}",
          f"* hedge H1: {iv.get('hedge_AMENDMENT3', {}).get('arms', {}).get('H1', {}).get('label_AMENDMENT3', 'NOT_AVAILABLE')}; H2: {iv.get('hedge_AMENDMENT3', {}).get('arms', {}).get('H2', {}).get('label_AMENDMENT3', 'NOT_AVAILABLE')}",
          f"* F2 / S0 combination: {iv.get('interventions_AMENDMENT4', {}).get('combination', 'NOT_AVAILABLE')}", "", "Numbers live in BRAKE_REROLL.json; read the controls first."]
    (M.OUT / "BRAKE_REROLL.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("ZZBLRANALYSISZZ", "PARTIAL" if J["PARTIAL"] else "COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
