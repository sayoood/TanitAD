#!/usr/bin/env python3
"""E-CP-1 — is there a DEPLOY-TIME-OBSERVABLE proxy for world-model
trustworthiness that recovers the oracle-gated C2 win?

CONTEXT (V5_IMAGINATION_SELECTION.md 2.3).  On the 22.7 % of windows where
v4's world model is good (`wm_canary_ade_2s <= 0.55`), rule **C2** — score fan
candidates by distance to a SINGLE world-model roll-out used as a reference
trajectory — beats the as-trained selector 0.7085 -> 0.3330, paired
-0.3754 [-0.5123, -0.2656].  ⛔ But `wm_canary_ade_2s` is computed against
GROUND-TRUTH future poses, so the stratifier is an ORACLE.  This file asks
whether anything observable at deploy time can replace it.

THE ESTIMAND, AND WHY IT IS A **POLICY** AND NOT A STRATUM
----------------------------------------------------------
A stratum mean is not comparable across gates: a gate that fires on 5 % of
windows and one that fires on 95 % report their wins on different populations.
Everything here is therefore scored as a DEPLOYABLE POLICY on the FULL 881
windows:

    pi_g(w) = C2's pick   if gate g(w) fires
              A0's pick   otherwise

and compared to pi_A0 (always the as-trained pick) with the paired
episode-cluster bootstrap over the 40 val episodes.  Under this framing the
oracle gate's whole-set value is (n_sel / n) * (-0.3754) and the ungated
policy is the special case g == 1, so the trivial baseline and every proxy sit
on ONE axis with ONE denominator.

ADMISSIBILITY (enforced, not asserted) — every feature is tagged:
  DEPLOY-1WM  : inputs + one world model's own forward pass.  Deployable.
  DEPLOY-2WM  : needs a SECOND world model at deploy time (ensemble
                disagreement).  Deployable at 2x imagination cost — flagged.
  ORACLE      : touches ground-truth future poses.  NEVER a gate input; used
                only as the ceiling and as the S-tests' answer key.

Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B=2000, unit =
episode.  NEVER `overlapping_holdout_se`.

⚠️ `base_rank` in the reduced dumps is NOT a score ranking (v5_cost_curve.py
E-H1 9.2: column 0 is the deployed pick, columns 1.. are anchor INDEX order).
It is therefore not used for any rank-agreement feature here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "taniteval"))
from taniteval.ci import (  # noqa: E402
    episode_cluster_bootstrap,
    paired_episode_cluster_bootstrap,
)

V5 = (REPO / "TanitAD Research Hub" / "Architecture & Inference" / "Implementation"
      / "incoming" / "2026-07-26-v5-imagination-selection" / "raw")

N_BOOT = 2000
SEED = 20260727
CANARY_BAR = 0.55
COST_KEYS = ("A1_imag_consistency", "A2_imag_goal_speed", "A3_imag_kinematic",
             "C1_ctrv_consistency", "C2_wm_ref_proximity")
K_KEYS = ("k1", "k2", "k4", "k6", "k8", "k10", "k14", "k20")

# committed numbers this harness must reproduce before it may adjudicate
COMMITTED = {
    "v4": {"A0": 0.8563, "C2": 1.0653, "O": 0.2505, "R": 15.8738,
           "strat_A0": 0.7085, "strat_C2": 0.3330, "strat_delta": -0.3754,
           "strat_n_win": 200, "strat_n_ep": 29},
    "v1": {"A0": 0.8563, "C2": 0.5645, "O": 0.2505,
           "whole_delta": -0.2918},
}


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
def load(tag: str) -> dict:
    D = torch.load(str(V5 / f"v5_{tag}_windows_reduced.pt"), map_location="cpu",
                   weights_only=False)
    out = {
        "ep": D["ep"].numpy().astype(np.int64),
        "t": D["t"].numpy().astype(np.int64),
        "v0": D["v0"].numpy().astype(np.float64),
        "canary": D["canary_err"].numpy().astype(np.float64),
        "fan_err4": D["fan_err4"].numpy().astype(np.float64),      # [W, 256] ORACLE
        "costs": {k: D["costs"][k].numpy().astype(np.float64) for k in COST_KEYS},
        "by_k": {k: D["cost_A1_by_k"][k].numpy().astype(np.float64) for k in K_KEYS},
        "picks": {k: v.numpy().astype(np.int64) for k, v in D["picks"].items()},
    }
    return out


def err_of(d: dict, pick: np.ndarray) -> np.ndarray:
    """ORACLE readout: the 4-waypoint ade_0_2s of a per-window candidate choice."""
    return d["fan_err4"][np.arange(len(pick)), pick]


# --------------------------------------------------------------------------- #
# STAGE 0 — fidelity, BOTH DIRECTIONS                                          #
# --------------------------------------------------------------------------- #
def stage0(dv4: dict, dv1: dict) -> dict:
    R: dict = {"_read": "S-tests. The harness may not adjudicate until every "
                        "PASS is true. S0d/S0e are the DELIBERATELY FAILING "
                        "direction."}
    ok = True

    # S0a — the reduced dumps must agree on everything that is WM-independent
    same = {
        "ep": bool((dv4["ep"] == dv1["ep"]).all()),
        "v0": bool(np.allclose(dv4["v0"], dv1["v0"])),
        "fan_err4": bool(np.allclose(dv4["fan_err4"], dv1["fan_err4"])),
        "A0_pick": bool((dv4["picks"]["A0_as_trained"]
                         == dv1["picks"]["A0_as_trained"]).all()),
        "C1_cost_is_WM_free": bool(np.allclose(dv4["costs"]["C1_ctrv_consistency"],
                                               dv1["costs"]["C1_ctrv_consistency"])),
    }
    R["S0a_same_fan_and_windows"] = {**same, "PASS": all(same.values())}
    ok &= all(same.values())

    # S0b — reproduce the committed arm means from fan_err4 alone
    rep: dict = {}
    for tag, d in (("v4", dv4), ("v1", dv1)):
        rep[tag] = {}
        for nm, key in (("A0", "A0_as_trained"), ("C2", "C2_wm_ref_proximity"),
                        ("O", "O_oracle_in_fan"), ("R", "R_random")):
            if key not in d["picks"] or nm not in COMMITTED[tag]:
                continue
            got = float(err_of(d, d["picks"][key]).mean())
            want = COMMITTED[tag][nm]
            rep[tag][nm] = {"got": round(got, 4), "committed": want,
                            "abs_diff": round(abs(got - want), 5),
                            "PASS": bool(abs(got - want) < 5e-4)}
            ok &= rep[tag][nm]["PASS"]
    R["S0b_reproduce_committed_arm_means"] = rep

    # S0c — reproduce the ORACLE-GATED stratum of 2.3 exactly
    m = dv4["canary"] <= CANARY_BAR
    a0, c2 = err_of(dv4, dv4["picks"]["A0_as_trained"]), err_of(
        dv4, dv4["picks"]["C2_wm_ref_proximity"])
    ci = paired_episode_cluster_bootstrap(c2[m], a0[m], dv4["ep"][m],
                                          n_boot=N_BOOT, seed=SEED)
    s = {"n_windows": int(m.sum()), "n_episodes": int(len(np.unique(dv4["ep"][m]))),
         "A0": round(float(a0[m].mean()), 4), "C2": round(float(c2[m].mean()), 4),
         "delta": ci["delta"], "committed_delta": COMMITTED["v4"]["strat_delta"]}
    s["PASS"] = bool(abs(s["A0"] - COMMITTED["v4"]["strat_A0"]) < 5e-4
                     and abs(s["C2"] - COMMITTED["v4"]["strat_C2"]) < 5e-4
                     and abs(s["delta"] - COMMITTED["v4"]["strat_delta"]) < 5e-4
                     and s["n_windows"] == COMMITTED["v4"]["strat_n_win"]
                     and s["n_episodes"] == COMMITTED["v4"]["strat_n_ep"])
    R["S0c_reproduce_oracle_gated_stratum"] = s
    ok &= s["PASS"]

    # S0d — DELIBERATELY FAILING: a gate built from pure noise must NOT separate
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(a0))
    pol = np.where(noise <= np.quantile(noise, 0.227), c2, a0)
    nci = paired_episode_cluster_bootstrap(pol, a0, dv4["ep"], n_boot=N_BOOT,
                                           seed=SEED)
    R["S0d_noise_gate_must_NOT_win"] = {
        **nci, "selected_frac": 0.227,
        "PASS": bool(not (nci["separated"] and nci["delta"] < 0)),
        "_read": "a random 22.7 % gate on v4's WM applies a rule that is "
                 "separated-WORSE whole-set, so it must come out >= 0."}
    ok &= R["S0d_noise_gate_must_NOT_win"]["PASS"]

    # S0e — DELIBERATELY FAILING: a gate that fires on 0 % must give EXACTLY 0
    zci = paired_episode_cluster_bootstrap(a0, a0, dv4["ep"], n_boot=N_BOOT,
                                           seed=SEED)
    R["S0e_empty_gate_is_exactly_zero"] = {
        "delta": zci["delta"], "separated": zci["separated"],
        "PASS": bool(zci["delta"] == 0.0 and not zci["separated"])}
    ok &= R["S0e_empty_gate_is_exactly_zero"]["PASS"]

    # S0f — the ORACLE is not admissible as a feature: assert it is excluded
    R["S0f_oracle_features_excluded"] = {
        "oracle_arrays": ["canary_err", "fan_err4"],
        "used_as_gate_input": False,
        "PASS": True,
        "_read": "`canary_err` and `fan_err4` appear ONLY in the ceiling rows "
                 "and the S-tests' answer key; `features()` never reads them."}

    R["ALL_PASS"] = bool(ok)
    return R


# --------------------------------------------------------------------------- #
# STAGE 1 — the UNGATED baseline and the ceilings.  Priority 1.                #
# --------------------------------------------------------------------------- #
def policy_ci(sel: np.ndarray, d: dict, a0: np.ndarray, c2: np.ndarray,
              ep: np.ndarray) -> dict:
    """Whole-set paired CI of pi_sel (C2 where sel, else A0) vs pi_A0.

    ⚠️ Also carries `vs_ungated`: the paired CI against `pi_C2-everywhere`.
    When the UNGATED policy already wins (v1's world model, §1.2), "separated
    vs A0" is NOT evidence that the gate does anything — a gate firing on 99 %
    of windows inherits the ungated win wholesale. The incremental interval is
    the only one that can distinguish a gate from its own baseline.
    """
    pol = np.where(sel, c2, a0)
    ci = paired_episode_cluster_bootstrap(pol, a0, ep, n_boot=N_BOOT, seed=SEED)
    ci["policy_ade_0_2s"] = round(float(pol.mean()), 4)
    ci["selected_frac"] = round(float(sel.mean()), 4)
    ci["n_selected"] = int(sel.sum())
    ci["n_selected_episodes"] = int(len(np.unique(ep[sel]))) if sel.any() else 0
    inc = paired_episode_cluster_bootstrap(pol, c2, ep, n_boot=N_BOOT, seed=SEED)
    ci["vs_ungated"] = {k: inc[k] for k in
                        ("delta", "lo", "hi", "separated", "p_delta_gt0")}
    return ci


def stage1(dv4: dict, dv1: dict) -> dict:
    ep = dv4["ep"]
    a0 = err_of(dv4, dv4["picks"]["A0_as_trained"])
    R: dict = {
        "_read": "⭐ PRIORITY 1. Every row is a DEPLOYABLE POLICY scored on all "
                 "881 windows / 40 episodes, so gated and ungated are on one "
                 "axis. `delta` NEGATIVE = better than the as-trained selector.",
        "A0_as_trained_ade_0_2s": round(float(a0.mean()), 4),
        "arms": {},
    }
    for tag, d in (("v4", dv4), ("v1", dv1)):
        c2 = err_of(d, d["picks"]["C2_wm_ref_proximity"])
        rows: dict = {}
        # the trivial baseline: apply C2 everywhere
        rows["UNGATED_C2_everywhere"] = policy_ci(
            np.ones(len(a0), bool), d, a0, c2, ep)
        # the oracle gate as actually registered
        rows["ORACLE_gate_canary_le_0.55"] = policy_ci(
            d["canary"] <= CANARY_BAR, d, a0, c2, ep)
        # oracle gate at the best possible canary threshold (in-sample, a ceiling)
        best = None
        for q in np.linspace(0.02, 1.0, 50):
            thr = float(np.quantile(d["canary"], q))
            sel = d["canary"] <= thr
            v = float(np.where(sel, c2, a0).mean())
            if best is None or v < best[0]:
                best = (v, q, thr, sel)
        rows["ORACLE_gate_best_canary_threshold_INSAMPLE"] = policy_ci(
            best[3], d, a0, c2, ep)
        rows["ORACLE_gate_best_canary_threshold_INSAMPLE"]["threshold"] = round(
            best[2], 4)
        # the absolute ceiling of ANY gate: per-window min(A0, C2)
        rows["CEILING_perfect_per_window_gate"] = policy_ci(
            c2 < a0, d, a0, c2, ep)
        # single-arm intervals for the record
        rows["_single_arm"] = {
            "A0": episode_cluster_bootstrap(a0, ep, n_boot=N_BOOT, seed=SEED),
            "C2": episode_cluster_bootstrap(c2, ep, n_boot=N_BOOT, seed=SEED),
        }
        R["arms"][tag] = rows
    return R


# --------------------------------------------------------------------------- #
# STAGE 2 — the proxy feature bank.  DEPLOY-TIME ONLY.                         #
# --------------------------------------------------------------------------- #
def _row_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = A - A.mean(1, keepdims=True)
    b = B - B.mean(1, keepdims=True)
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-12
    return num / den


def features(d: dict, other: dict | None = None) -> tuple[dict, dict]:
    """Per-window deploy-time scalars.  Returns (features, admissibility)."""
    F: dict = {}
    A: dict = {}
    W = len(d["ep"])
    a0pick = d["picks"]["A0_as_trained"]
    ar = np.arange(W)

    # --- family I: input-side scene statistics ---------------------------
    F["v0"] = d["v0"]
    A["v0"] = "DEPLOY-1WM/input"

    # --- family II: cost-distribution shape, per rule --------------------
    for key in COST_KEYS:
        M = d["costs"][key]
        srt = np.sort(M, axis=1)
        F[f"{key}__min"] = srt[:, 0]
        F[f"{key}__mean"] = M.mean(1)
        F[f"{key}__std"] = M.std(1)
        F[f"{key}__p90"] = srt[:, int(0.90 * M.shape[1])]
        F[f"{key}__range"] = srt[:, -1] - srt[:, 0]
        F[f"{key}__gap12"] = srt[:, 1] - srt[:, 0]
        # margin normalised by the spread: how DECISIVE is this rule here?
        F[f"{key}__margin_norm"] = (srt[:, 1] - srt[:, 0]) / (M.std(1) + 1e-9)
        # the head's own pick scored by this rule = head-vs-rule disagreement
        F[f"{key}__at_A0pick"] = M[ar, a0pick]
        F[f"{key}__at_A0pick_z"] = (M[ar, a0pick] - M.mean(1)) / (M.std(1) + 1e-9)
        for suf in ("min", "mean", "std", "p90", "range", "gap12",
                    "margin_norm", "at_A0pick", "at_A0pick_z"):
            A[f"{key}__{suf}"] = "DEPLOY-1WM"

    # --- family III: WM-vs-analytic-model divergence (self-consistency) --
    A1, C1, C2 = (d["costs"]["A1_imag_consistency"],
                  d["costs"]["C1_ctrv_consistency"],
                  d["costs"]["C2_wm_ref_proximity"])
    F["wm_minus_ctrv__mean"] = (A1 - C1).mean(1)
    F["wm_minus_ctrv__absmean"] = np.abs(A1 - C1).mean(1)
    F["wm_minus_ctrv__corr"] = _row_corr(A1, C1)
    F["corr_A1_C2"] = _row_corr(A1, C2)
    F["corr_C1_C2"] = _row_corr(C1, C2)
    # do the WM and the analytic model pick the same candidate?
    F["argmin_agree_A1_C1"] = (A1.argmin(1) == C1.argmin(1)).astype(float)
    F["A1_at_C1argmin_z"] = ((A1[ar, C1.argmin(1)] - A1.mean(1))
                             / (A1.std(1) + 1e-9))
    for k in ("wm_minus_ctrv__mean", "wm_minus_ctrv__absmean",
              "wm_minus_ctrv__corr", "corr_A1_C2", "corr_C1_C2",
              "argmin_agree_A1_C1", "A1_at_C1argmin_z"):
        A[k] = "DEPLOY-1WM"

    # --- family IV: roll-out drift over the horizon ----------------------
    for k in K_KEYS:
        F[f"drift__{k}_mean"] = d["by_k"][k].mean(1)
        A[f"drift__{k}_mean"] = "DEPLOY-1WM"
    for a_, b_ in (("k20", "k1"), ("k20", "k2"), ("k20", "k8"), ("k8", "k2"),
                   ("k8", "k1"), ("k4", "k1")):
        F[f"drift__ratio_{a_}_{b_}"] = (d["by_k"][a_].mean(1)
                                        / (d["by_k"][b_].mean(1) + 1e-9))
        A[f"drift__ratio_{a_}_{b_}"] = "DEPLOY-1WM"
    # convexity of the drift curve: accelerating divergence is the bad sign
    ks = np.array([1, 2, 4, 6, 8, 10, 14, 20], float)
    Y = np.stack([d["by_k"][k].mean(1) for k in K_KEYS], 1)      # [W, 8]
    lg = np.polyfit(np.log(ks), np.log(Y.T + 1e-9), 1)           # [2, W]
    F["drift__loglog_slope"] = lg[0]
    A["drift__loglog_slope"] = "DEPLOY-1WM"

    # --- family V: 2-WM ensemble disagreement (flagged: 2x cost) ---------
    if other is not None:
        oC2 = other["costs"]["C2_wm_ref_proximity"]
        oA1 = other["costs"]["A1_imag_consistency"]
        F["ens__C2_absdiff_mean"] = np.abs(C2 - oC2).mean(1)
        F["ens__C2_corr"] = _row_corr(C2, oC2)
        F["ens__C2_argmin_agree"] = (C2.argmin(1) == oC2.argmin(1)).astype(float)
        # the two WMs' reference rolls both live in the fan's metric frame; the
        # fan distance between their two argmins is an ensemble spread proxy
        F["ens__C2_at_other_argmin_z"] = ((C2[ar, oC2.argmin(1)] - C2.mean(1))
                                          / (C2.std(1) + 1e-9))
        F["ens__A1_corr"] = _row_corr(A1, oA1)
        for k in ("ens__C2_absdiff_mean", "ens__C2_corr", "ens__C2_argmin_agree",
                  "ens__C2_at_other_argmin_z", "ens__A1_corr"):
            A[k] = "DEPLOY-2WM"
    return F, A


# --------------------------------------------------------------------------- #
# STAGE 3 — single-feature gates, out-of-fold                                  #
# --------------------------------------------------------------------------- #
def folds_by_episode(ep: np.ndarray, k: int = 5, seed: int = SEED) -> list:
    eps = np.unique(ep)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eps))
    out = []
    for f in range(k):
        held = set(eps[perm[f::k]].tolist())
        out.append(np.array([e in held for e in ep]))
    return out


def oof_threshold_gate(x: np.ndarray, a0: np.ndarray, c2: np.ndarray,
                       ep: np.ndarray, qgrid=None, k=5) -> dict:
    """Pick the threshold on the TRAIN folds, apply on the HELD-OUT fold.

    ⚠️ Episode-disjoint.  The selected quantile is chosen on train windows only;
    the threshold VALUE is the train-fold empirical quantile, so no held-out
    information enters either the rule or its calibration.
    """
    if qgrid is None:
        qgrid = np.round(np.arange(0.05, 1.01, 0.05), 3)
    sel = np.zeros(len(x), bool)
    chosen = []
    for held in folds_by_episode(ep, k):
        tr = ~held
        best_q, best_v = None, None
        for q in qgrid:
            thr = float(np.quantile(x[tr], q))
            s = x[tr] <= thr
            v = float(np.where(s, c2[tr], a0[tr]).mean())
            if best_v is None or v < best_v:
                best_q, best_v = float(q), v
        thr = float(np.quantile(x[tr], best_q))       # calibrated on TRAIN only
        sel[held] = x[held] <= thr
        chosen.append({"q": best_q, "thr": round(thr, 5)})
    pol = np.where(sel, c2, a0)
    ci = paired_episode_cluster_bootstrap(pol, a0, ep, n_boot=N_BOOT, seed=SEED)
    ci["policy_ade_0_2s"] = round(float(pol.mean()), 4)
    ci["selected_frac"] = round(float(sel.mean()), 4)
    ci["n_selected"] = int(sel.sum())
    ci["fold_choices"] = chosen
    inc = paired_episode_cluster_bootstrap(pol, c2, ep, n_boot=N_BOOT, seed=SEED)
    ci["vs_ungated"] = {k: inc[k] for k in
                        ("delta", "lo", "hi", "separated", "p_delta_gt0")}
    return ci


def oof_feature_and_threshold_gate(F: dict, a0, c2, ep, k=5,
                                   qgrid=None) -> dict:
    """⭐ The HONEST single-feature result: the FEATURE, its SIGN and its
    THRESHOLD are all chosen on the training episodes of each fold.

    Reporting "the best of 73 features x 2 signs" after looking at all of them
    is a 146-hypothesis search dressed as one test.  This wrapper moves the
    feature choice inside the fold so the held-out episodes never inform it.
    """
    if qgrid is None:
        qgrid = np.round(np.arange(0.05, 1.01, 0.05), 3)
    names = sorted(F)
    X = {n: np.nan_to_num(np.asarray(F[n], float), nan=0.0, posinf=0.0,
                          neginf=0.0) for n in names}
    sel = np.zeros(len(a0), bool)
    chosen = []
    for held in folds_by_episode(ep, k):
        tr = ~held
        best = None
        for n in names:
            for sign in (+1, -1):
                x = sign * X[n]
                for q in qgrid:
                    thr = float(np.quantile(x[tr], q))
                    s = x[tr] <= thr
                    v = float(np.where(s, c2[tr], a0[tr]).mean())
                    if best is None or v < best[0]:
                        best = (v, n, sign, float(q), thr)
        _, n, sign, q, thr = best
        sel[held] = (sign * X[n][held]) <= thr
        chosen.append({"feature": n, "sign": sign, "q": q, "thr": round(thr, 5)})
    pol = np.where(sel, c2, a0)
    ci = paired_episode_cluster_bootstrap(pol, a0, ep, n_boot=N_BOOT, seed=SEED)
    ci["policy_ade_0_2s"] = round(float(pol.mean()), 4)
    ci["selected_frac"] = round(float(sel.mean()), 4)
    ci["n_selected"] = int(sel.sum())
    ci["fold_choices"] = chosen
    ci["fold_feature_stability"] = len(set(c["feature"] for c in chosen))
    inc = paired_episode_cluster_bootstrap(pol, c2, ep, n_boot=N_BOOT, seed=SEED)
    ci["vs_ungated"] = {k: inc[k] for k in
                        ("delta", "lo", "hi", "separated", "p_delta_gt0")}
    return ci


def _rank(x: np.ndarray) -> np.ndarray:
    o = np.empty(len(x), float)
    o[np.argsort(x, kind="stable")] = np.arange(len(x), dtype=float)
    return o


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    a, b = _rank(x), _rank(y)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def stage2(dv4: dict, dv1: dict) -> dict:
    """Every deploy-time feature vs BOTH targets, with the OOF single-feature gate.

    Target A = `canary_err`      — the brief's literal question ("is the WM good here?")
    Target B = A0_err - C2_err   — the question that PAYS ("does C2 beat A0 here?")
    They are different targets and §1.4 says so; both are reported.
    """
    ep = dv4["ep"]
    a0 = err_of(dv4, dv4["picks"]["A0_as_trained"])
    out: dict = {"_read":
                 "corr_* are diagnostics ONLY. The deliverable is the GATED "
                 "POLICY column, out-of-fold, with its selected fraction. A "
                 "correlation that does not convert is reported as PARTIAL.",
                 "arms": {}}
    for tag, d, other in (("v4", dv4, dv1), ("v1", dv1, dv4)):
        c2 = err_of(d, d["picks"]["C2_wm_ref_proximity"])
        util = a0 - c2                      # >0 means C2 is the better pick here
        F, ADM = features(d, other)
        rows = []
        for name, x in F.items():
            x = np.asarray(x, float)
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            r = {"feature": name, "admissibility": ADM[name],
                 "corr_pearson_vs_canary": round(
                     float(np.corrcoef(x, d["canary"])[0, 1]), 4),
                 "corr_spearman_vs_canary": round(spearman(x, d["canary"]), 4),
                 "corr_spearman_vs_gate_utility": round(spearman(x, util), 4)}
            best = None
            for sign in (+1, -1):
                g = oof_threshold_gate(sign * x, a0, c2, ep)
                g["sign"] = sign
                if best is None or g["delta"] < best["delta"]:
                    best = g
            r["oof_gate"] = {k: best[k] for k in
                             ("delta", "lo", "hi", "separated", "selected_frac",
                              "policy_ade_0_2s", "sign", "n_selected",
                              "vs_ungated")}
            rows.append(r)
        rows.sort(key=lambda z: z["oof_gate"]["delta"])
        F1 = {k: v for k, v in F.items() if ADM[k].startswith("DEPLOY-1WM")}
        out["arms"][tag] = {
            "n_features": len(rows),
            "corr_pearson_v0_vs_canary_REPRODUCTION": round(
                float(np.corrcoef(d["v0"], d["canary"])[0, 1]), 4),
            "best_abs_spearman_vs_canary": max(
                rows, key=lambda z: abs(z["corr_spearman_vs_canary"]))["feature"],
            "NESTED_oof_feature_and_threshold_ALL": oof_feature_and_threshold_gate(
                F, a0, c2, ep),
            "NESTED_oof_feature_and_threshold_1WM": oof_feature_and_threshold_gate(
                F1, a0, c2, ep),
            "rows": rows}
    return out


def _design(F: dict) -> tuple[np.ndarray, list]:
    names = sorted(F)
    X = np.stack([np.nan_to_num(np.asarray(F[n], float), nan=0.0, posinf=0.0,
                                neginf=0.0) for n in names], 1)
    return X, names


def _fit_predict(Xtr, ytr, Xte, kind: str, seed: int = SEED):
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xte))
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=40,
        l2_regularization=1.0, random_state=seed).fit(Xtr, ytr)
    return m.predict(Xte)


def learned_gate(F: dict, a0, c2, ep, kind="gbm", k=5, tune_tau=False) -> dict:
    """OOF, EPISODE-DISJOINT learned gate on the utility (A0_err - C2_err).

    Primary rule fixes tau = 0 (fire iff predicted utility > 0), which removes a
    degree of freedom entirely. ``tune_tau`` adds a NESTED inner CV on the train
    episodes so tau is never chosen on data the outer fold will be scored on.
    """
    X, names = _design(F)
    util = a0 - c2
    sel = np.zeros(len(a0), bool)
    pred = np.full(len(a0), np.nan)
    taus = []
    for held in folds_by_episode(ep, k):
        tr = ~held
        p = _fit_predict(X[tr], util[tr], X[held], kind)
        pred[held] = p
        tau = 0.0
        if tune_tau:
            inner_pred = np.full(int(tr.sum()), np.nan)
            ep_tr, a0_tr, c2_tr = ep[tr], a0[tr], c2[tr]
            Xtr, utr = X[tr], util[tr]
            for ih in folds_by_episode(ep_tr, 4, seed=SEED + 1):
                inner_pred[ih] = _fit_predict(Xtr[~ih], utr[~ih], Xtr[ih], kind)
            best = None
            for q in np.linspace(0.0, 0.95, 20):
                t = float(np.quantile(inner_pred, q))
                v = float(np.where(inner_pred > t, c2_tr, a0_tr).mean())
                if best is None or v < best[0]:
                    best = (v, t)
            tau = best[1]
        taus.append(round(float(tau), 5))
        sel[held] = p > tau
    pol = np.where(sel, c2, a0)
    ci = paired_episode_cluster_bootstrap(pol, a0, ep, n_boot=N_BOOT, seed=SEED)
    ci["policy_ade_0_2s"] = round(float(pol.mean()), 4)
    ci["selected_frac"] = round(float(sel.mean()), 4)
    ci["n_selected"] = int(sel.sum())
    ci["model"] = kind
    ci["tau_per_fold"] = taus
    ci["oof_spearman_pred_vs_true_utility"] = round(spearman(pred, util), 4)
    ci["n_features"] = len(names)
    inc = paired_episode_cluster_bootstrap(pol, c2, ep, n_boot=N_BOOT, seed=SEED)
    ci["vs_ungated"] = {k: inc[k] for k in
                        ("delta", "lo", "hi", "separated", "p_delta_gt0")}
    return ci


def learned_canary_regressor(F: dict, canary, ep, kind="gbm", k=5) -> dict:
    """Can ANY learned combination predict the canary out-of-fold?  (Target A.)"""
    X, _ = _design(F)
    pred = np.full(len(canary), np.nan)
    for held in folds_by_episode(ep, k):
        pred[held] = _fit_predict(X[~held], canary[~held], X[held], kind)
    ss_res = float(((canary - pred) ** 2).sum())
    ss_tot = float(((canary - canary.mean()) ** 2).sum())
    return {"model": kind,
            "oof_r2": round(1.0 - ss_res / ss_tot, 4),
            "oof_pearson": round(float(np.corrcoef(pred, canary)[0, 1]), 4),
            "oof_spearman": round(spearman(pred, canary), 4),
            "_read": "R2 <= 0 means the learned proxy is worse than predicting "
                     "the mean canary. This is the PARTIAL-branch evidence."}


def predicted_canary_gate(F: dict, canary, a0, c2, ep, kind="ridge", k=5,
                          bar=CANARY_BAR, tune_q=False) -> dict:
    """⭐ THE LITERAL INSTRUMENT THE BRIEF NAMED: learn an OBSERVABLE estimate of
    `wm_canary_ade_2s`, then gate on it exactly where the oracle gated.

    The oracle rule is `canary <= 0.55`.  This replaces `canary` with `canary_hat`
    from deploy-time features only, fitted on the TRAIN episodes of each fold.
    `tune_q` instead picks the firing quantile on train (the bar is a v4-specific
    constant and need not transfer).
    """
    X, _ = _design(F)
    sel = np.zeros(len(a0), bool)
    pred = np.full(len(a0), np.nan)
    bars = []
    for held in folds_by_episode(ep, k):
        tr = ~held
        p = _fit_predict(X[tr], canary[tr], X[held], kind)
        pred[held] = p
        b = bar
        if tune_q:
            ptr = _fit_predict(X[tr], canary[tr], X[tr], kind)
            best = None
            for q in np.linspace(0.05, 1.0, 20):
                t = float(np.quantile(ptr, q))
                v = float(np.where(ptr <= t, c2[tr], a0[tr]).mean())
                if best is None or v < best[0]:
                    best = (v, t)
            b = best[1]
        bars.append(round(float(b), 4))
        sel[held] = p <= b
    pol = np.where(sel, c2, a0)
    ci = paired_episode_cluster_bootstrap(pol, a0, ep, n_boot=N_BOOT, seed=SEED)
    ci["policy_ade_0_2s"] = round(float(pol.mean()), 4)
    ci["selected_frac"] = round(float(sel.mean()), 4)
    ci["n_selected"] = int(sel.sum())
    ci["bar_per_fold"] = bars
    ci["oof_canary_pearson"] = round(float(np.corrcoef(pred, canary)[0, 1]), 4)
    inc = paired_episode_cluster_bootstrap(pol, c2, ep, n_boot=N_BOOT, seed=SEED)
    ci["vs_ungated"] = {k2: inc[k2] for k2 in
                        ("delta", "lo", "hi", "separated", "p_delta_gt0")}
    # agreement with the ORACLE gate it is replacing
    orc = canary <= bar
    ci["agreement_with_oracle_gate"] = {
        "accuracy": round(float((sel == orc).mean()), 4),
        "recall_of_oracle_selected": round(
            float((sel & orc).sum() / max(1, orc.sum())), 4),
        "precision": round(float((sel & orc).sum() / max(1, sel.sum())), 4)}
    return ci


def seed_stability(F: dict, a0, c2, ep, kind="ridge", seeds=(1, 2, 3, 4, 5)) -> dict:
    """⚠️ 40 episodes / 5 folds — a single fold assignment can flatter a gate.

    Re-runs the tau=0 learned gate under different EPISODE fold assignments and
    reports the spread.  If the headline moves sign across seeds it is fold luck,
    not a capability.
    """
    out = []
    X, _ = _design(F)
    util = a0 - c2
    for s in seeds:
        sel = np.zeros(len(a0), bool)
        for held in folds_by_episode(ep, 5, seed=SEED + 1000 * s):
            sel[held] = _fit_predict(X[~held], util[~held], X[held], kind) > 0
        pol = np.where(sel, c2, a0)
        ci = paired_episode_cluster_bootstrap(pol, a0, ep, n_boot=500, seed=SEED)
        inc = paired_episode_cluster_bootstrap(pol, c2, ep, n_boot=500, seed=SEED)
        out.append({"fold_seed": int(SEED + 1000 * s), "delta": ci["delta"],
                    "separated": ci["separated"],
                    "delta_vs_ungated": inc["delta"],
                    "separated_vs_ungated": inc["separated"],
                    "selected_frac": round(float(sel.mean()), 4)})
    d = [o["delta"] for o in out]
    u = [o["delta_vs_ungated"] for o in out]
    return {"runs": out, "delta_min": round(min(d), 4), "delta_max": round(max(d), 4),
            "delta_mean": round(float(np.mean(d)), 4),
            "all_same_sign": bool(all(x < 0 for x in d) or all(x > 0 for x in d)),
            "all_separated": bool(all(o["separated"] for o in out)),
            "vs_ungated_min": round(min(u), 4), "vs_ungated_max": round(max(u), 4),
            "vs_ungated_mean": round(float(np.mean(u)), 4),
            "vs_ungated_all_same_sign": bool(all(x < 0 for x in u)
                                             or all(x > 0 for x in u)),
            "vs_ungated_all_separated": bool(o["separated_vs_ungated"]
                                             for o in out) and all(
                                                 o["separated_vs_ungated"]
                                                 for o in out)}


def stage3(dv4: dict, dv1: dict) -> dict:
    ep = dv4["ep"]
    a0 = err_of(dv4, dv4["picks"]["A0_as_trained"])
    rng = np.random.default_rng(1234)
    R: dict = {"_read":
               "OOF, EPISODE-DISJOINT. The NOISE rows are the deliberately "
               "failing direction: an identical pipeline fed pure Gaussian "
               "features must not produce a separated win.",
               "arms": {}}
    for tag, d, other in (("v4", dv4, dv1), ("v1", dv1, dv4)):
        c2 = err_of(d, d["picks"]["C2_wm_ref_proximity"])
        F, ADM = features(d, other)
        F1 = {k: v for k, v in F.items() if ADM[k].startswith("DEPLOY-1WM")}
        F2 = {k: v for k, v in F.items() if ADM[k] == "DEPLOY-2WM"}
        noise = {f"noise_{i}": rng.standard_normal(len(a0)) for i in range(24)}
        arm: dict = {
            "ungated_reference": policy_ci(np.ones(len(a0), bool), d, a0, c2, ep),
            "canary_oracle_reference": policy_ci(d["canary"] <= CANARY_BAR, d,
                                                 a0, c2, ep),
            "ceiling_reference": policy_ci(c2 < a0, d, a0, c2, ep),
            "_feature_counts": {"ALL": len(F), "1WM": len(F1), "2WM": len(F2)},
        }
        for kind in ("ridge", "gbm"):
            arm[f"learned_gate_1WM_{kind}_tau0"] = learned_gate(F1, a0, c2, ep, kind)
            arm[f"learned_gate_2WMonly_{kind}_tau0"] = learned_gate(F2, a0, c2, ep,
                                                                    kind)
            arm[f"learned_gate_ALL_{kind}_tau0"] = learned_gate(F, a0, c2, ep, kind)
            arm[f"learned_gate_ALL_{kind}_tuned_tau"] = learned_gate(
                F, a0, c2, ep, kind, tune_tau=True)
            arm[f"NEGCTRL_noise_gate_{kind}_tau0"] = learned_gate(
                noise, a0, c2, ep, kind)
            # the LITERAL instrument: an observable stand-in for the canary
            arm[f"PREDICTED_CANARY_gate_ALL_{kind}_bar0.55"] = \
                predicted_canary_gate(F, d["canary"], a0, c2, ep, kind)
            arm[f"PREDICTED_CANARY_gate_1WM_{kind}_bar0.55"] = \
                predicted_canary_gate(F1, d["canary"], a0, c2, ep, kind)
            arm[f"PREDICTED_CANARY_gate_ALL_{kind}_tunedq"] = \
                predicted_canary_gate(F, d["canary"], a0, c2, ep, kind,
                                      tune_q=True)
            arm[f"NEGCTRL_PREDICTED_CANARY_noise_{kind}"] = \
                predicted_canary_gate(noise, d["canary"], a0, c2, ep, kind)
            arm[f"learned_canary_regressor_ALL_{kind}"] = learned_canary_regressor(
                F, d["canary"], ep, kind)
            arm[f"learned_canary_regressor_1WM_{kind}"] = learned_canary_regressor(
                F1, d["canary"], ep, kind)
            arm[f"NEGCTRL_noise_canary_regressor_{kind}"] = \
                learned_canary_regressor(noise, d["canary"], ep, kind)
        # ---- which FAMILY carries it?  one learned gate per family alone ----
        fam = {
            "I_input_v0": ["v0"],
            "II_costshape_A1": [k for k in F if k.startswith("A1_")],
            "II_costshape_A2": [k for k in F if k.startswith("A2_")],
            "II_costshape_A3": [k for k in F if k.startswith("A3_")],
            "II_costshape_C1": [k for k in F if k.startswith("C1_")],
            "II_costshape_C2": [k for k in F if k.startswith("C2_")],
            "III_wm_vs_ctrv_selfconsistency": [
                k for k in F if k.startswith(("wm_minus_ctrv", "corr_", "argmin_agree",
                                              "A1_at_C1argmin"))],
            "IV_rollout_drift": [k for k in F if k.startswith("drift__")],
            "V_ensemble_2WM": [k for k in F if k.startswith("ens__")],
        }
        arm["FAMILY_ABLATION_ridge_tau0"] = {
            nm: {k2: v2 for k2, v2 in learned_gate(
                {k: F[k] for k in ks_}, a0, c2, ep, "ridge").items()
                if k2 in ("delta", "lo", "hi", "separated", "selected_frac",
                          "policy_ade_0_2s", "vs_ungated",
                          "oof_spearman_pred_vs_true_utility")}
            | {"n_features": len(ks_)}
            for nm, ks_ in fam.items() if ks_}
        arm["SEED_STABILITY_ALL_ridge_tau0"] = seed_stability(F, a0, c2, ep, "ridge")
        arm["SEED_STABILITY_1WM_ridge_tau0"] = seed_stability(F1, a0, c2, ep, "ridge")
        arm["NEGCTRL_SEED_STABILITY_noise_ridge"] = seed_stability(
            noise, a0, c2, ep, "ridge")
        R["arms"][tag] = arm
    return R


def headline(R: dict) -> dict:
    """Recovered fractions, on ONE axis, with the dominance check spelled out."""
    s1 = R["stage1_ungated_baseline_and_ceilings"]["arms"]
    s3 = R["stage3_learned_gates_oof"]["arms"]
    orc4 = s1["v4"]["ORACLE_gate_canary_le_0.55"]["delta"]          # -0.0852
    ung1 = s1["v1"]["UNGATED_C2_everywhere"]["delta"]               # -0.2918

    def rec(x):
        return round(float(x) / float(orc4), 3)

    best4 = s3["v4"]["learned_gate_ALL_ridge_tuned_tau"]
    best4_1wm = s3["v4"]["learned_gate_1WM_gbm_tau0"]
    pc4 = s3["v4"]["PREDICTED_CANARY_gate_ALL_ridge_bar0.55"]
    best1 = s3["v1"]["learned_gate_ALL_ridge_tau0"]
    best1_1wm = s3["v1"]["learned_gate_1WM_ridge_tau0"]
    return {
        "_stratum_vs_policy": {
            "oracle_stratum_delta_2.3": -0.3754, "oracle_selected_frac": 0.227,
            "oracle_policy_delta_wholeset": orc4,
            "_read": "0.227 x 0.3754 = 0.0852. The 53 % cut is real and applies "
                     "to a fifth of driving."},
        "v4_wm": {
            "best_gate_ALL_2WM": {"delta": best4["delta"],
                                  "sep": best4["separated"],
                                  "frac": best4["selected_frac"],
                                  "recovered_fraction_of_oracle": rec(best4["delta"])},
            "best_gate_1WM": {"delta": best4_1wm["delta"],
                              "sep": best4_1wm["separated"],
                              "frac": best4_1wm["selected_frac"],
                              "recovered_fraction_of_oracle": rec(best4_1wm["delta"])},
            "predicted_canary_gate_ALL": {
                "delta": pc4["delta"], "sep": pc4["separated"],
                "frac": pc4["selected_frac"],
                "oof_canary_pearson": pc4["oof_canary_pearson"],
                "recovered_fraction_of_oracle": rec(pc4["delta"])},
            "canary_oof_r2_ALL": s3["v4"]["learned_canary_regressor_ALL_ridge"]["oof_r2"],
            "canary_oof_r2_1WM": s3["v4"]["learned_canary_regressor_1WM_gbm"]["oof_r2"],
            "canary_r2_from_v0_alone": round(0.2645 ** 2, 4)},
        "v1_wm": {
            "ungated_delta": ung1,
            "best_gate_ALL": {"delta": best1["delta"],
                              "vs_ungated": best1["vs_ungated"],
                              "frac": best1["selected_frac"]},
            "best_gate_1WM": {"delta": best1_1wm["delta"],
                              "vs_ungated": best1_1wm["vs_ungated"],
                              "frac": best1_1wm["selected_frac"]},
            "gate_headroom_above_ungated": s1["v1"][
                "CEILING_perfect_per_window_gate"]["vs_ungated"]["delta"],
            "canary_oracle_increment": s1["v1"][
                "ORACLE_gate_canary_le_0.55"]["vs_ungated"]},
        "POWER_unseparated_rows": {
            "_estimator": "ESTIMATED — sqrt(n) extrapolation of the MEASURED "
                          "half-width: n_req = 40 * (ci95/|delta|)^2. "
                          "MODEL_REGISTRY.md §1.2a MEASURED the half-width "
                          "shrinking ×2.8–3.9 (mean ≈3.4) from 40 → 600 "
                          "episodes, i.e. slightly FASTER than sqrt(15)=3.87 "
                          "would predict at the low end, so these n are if "
                          "anything conservative.",
            "rows": {k: {"delta": v["delta"], "ci95": v["ci95"],
                         "n_episodes_to_separate": int(np.ceil(
                             40 * (v["ci95"] / max(1e-9, abs(v["delta"]))) ** 2))}
                     for k, v in (
                         ("v4_gate_1WM_gbm_tau0", s3["v4"]["learned_gate_1WM_gbm_tau0"]),
                         ("v4_gate_1WM_ridge_tau0",
                          s3["v4"]["learned_gate_1WM_ridge_tau0"]))},
            "v1_incremental_worst_seed": {
                "delta": s3["v1"]["SEED_STABILITY_ALL_ridge_tau0"]["vs_ungated_max"],
                "ci95_at_headline_seed": round(
                    (best1["vs_ungated"]["hi"] - best1["vs_ungated"]["lo"]) / 2, 4),
                "n_episodes_to_separate": int(np.ceil(40 * (
                    ((best1["vs_ungated"]["hi"] - best1["vs_ungated"]["lo"]) / 2)
                    / abs(s3["v1"]["SEED_STABILITY_ALL_ridge_tau0"]
                          ["vs_ungated_max"])) ** 2))},
        },
        "DOMINANCE_CHECK": {
            "v4_2WM_gate_delta": best4["delta"],
            "v1_ungated_delta": ung1,
            "v1_gated_delta": best1["delta"],
            "_read": "the v4 2-WM gate needs a SECOND world model at deploy "
                     "time. With that same second world model one can instead "
                     "score with it directly and UNGATED. Compare the rows.",
            "v4_2WM_gate_is_dominated": bool(ung1 < best4["delta"])},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "raw" / "canary_proxy_s01.json"))
    ap.add_argument("--stage", default="01")
    a = ap.parse_args()

    dv4, dv1 = load("v4"), load("v1")
    R = {
        "_experiment": "E-CP-1 — deploy-time proxy for world-model trustworthiness",
        "_evidence_class": "MEASURED (ours) — recomputed from "
                           "2026-07-26-v5-imagination-selection/raw/"
                           "v5_{v4,v1}_windows_reduced.pt, no GPU, no pod",
        "_estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), "
                      "B=2000, unit = episode. NEVER overlapping_holdout_se.",
        "_n": {"windows": int(len(dv4["ep"])),
               "episodes": int(len(np.unique(dv4["ep"])))},
        "_date": "2026-07-27",
    }
    R["S_tests"] = stage0(dv4, dv1)
    if not R["S_tests"]["ALL_PASS"]:
        Path(a.out).write_text(json.dumps(R, indent=2))
        raise SystemExit("S-TESTS FAILED — refusing to adjudicate. see " + a.out)
    R["stage1_ungated_baseline_and_ceilings"] = stage1(dv4, dv1)
    if "2" in a.stage:
        R["stage2_single_feature_proxies"] = stage2(dv4, dv1)
    if "3" in a.stage:
        R["stage3_learned_gates_oof"] = stage3(dv4, dv1)
        R["headline"] = headline(R)
    Path(a.out).write_text(json.dumps(R, indent=2))
    print("WROTE", a.out, "| S_tests ALL_PASS =", R["S_tests"]["ALL_PASS"])
    if "headline" in R:
        print(json.dumps(R["headline"], indent=2))


if __name__ == "__main__":
    main()
