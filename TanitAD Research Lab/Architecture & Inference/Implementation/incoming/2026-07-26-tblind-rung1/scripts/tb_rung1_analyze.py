#!/usr/bin/env python3
"""RUNG 1 — the planner-action adjudication. ZERO GPU.

Reads the per-window dense ``de`` produced on pod2 and answers, in the
pre-registered order (``PRE_REGISTRATION.md`` §8), banking each stage's artifact
before the next begins:

  gates    the WINDOW-SET IDENTITY gate, the PLUMBING SELF-TEST, a fidelity gate
           against the ladder's committed headline numbers, and the DIAGNOSTIC
           VACUITY AUDIT (§6c) — nothing below is read until they pass.
  blend    the own <-> hold-last blend curve. It interpolates the two MEASURED
           endpoints (25 and 115 steps), so its SHAPE is evidence a
           best-of-N selection effect cannot manufacture.
  mech     the mechanism separation: compounding drift vs clamp saturation vs
           feedback instability vs horizon onset, each with the signature that
           would REFUTE it (§7).
  other    the clip / smooth / update-frequency families.
  verdict  the pre-registered buckets (§5) plus the CAPABILITY CAP (§5.1).

Estimator everywhere: paired episode-cluster bootstrap, ``taniteval/ci.py``,
B = 2000, seed 0, unit = episode cluster, identical windows for every arm.
``overlapping_holdout_se`` appears nowhere. The rule machinery (``t_blind``,
``paired_at``, ``separated_better_interval``, ``draws_for``) is IMPORTED from the
ladder's ``tb_rung0.py`` rather than re-implemented — two independent
re-implementations of a firewall produced nulls that were overturned.

Usage:
    python tb_rung1_analyze.py --new  perwindow/rung1_perwindow_compact.pt \
        --bi ../2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt \
        --matched ../2026-07-26-tblind-ladder/perwindow/perwindow_matched_K185.pt \
        --audit perwindow/action_audit_K185.pt --out artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_INCOMING = _HERE.parent.parent
sys.path.insert(0, str(_INCOMING / "2026-07-26-tblind-ladder" / "scripts"))
_REPO = _HERE.parents[5]
for _p in (_REPO / "taniteval", _REPO / "stack"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tb_rung0 import (DT, GRID, B_BOOT, SEED, ade_0_2s, draws_for,  # noqa: E402
                      paired_at, separated_better_interval, single_at,
                      t_blind, t_contiguous)
from taniteval import blindimag as bi                              # noqa: E402

A_OWN, B_OWN = "a_imagination__own__roSTR", "b_frozenlast__own__roSTR"
A_HOLD, B_HOLD = "a_imagination__hold__roSTR", "b_frozenlast__hold__roSTR"
ANCHOR_TOL = 1e-4              # m — Rung 0b measured 3.05e-05 between two passes

#: PRE_REGISTRATION.md §0, re-read from the ladder's raw JSON. The fidelity gate.
LADDER = {
    "T_blind_own_str_steps": 25, "T_blind_own_str_ci_s": [2.5, 3.9],
    "T_blind_hold_str_steps": 115, "T_blind_hold_str_ci_s": [11.5, 17.4],
    "de2s_own_str": 1.8165, "de2s_hold_str": 0.6718,
    "ade_0_2s_own_str": 0.8710, "ade_0_2s_hold_str": 0.3351,
    "paired_delta_2s_own_str": 0.4130,
    "T_useful_1m_own_str_s": 1.4, "T_useful_1m_hold_str_s": 2.3,
}

#: §4 — fixed BEFORE any number existed. ELIGIBLE = may be adopted / may set the
#: verdict: both fed channels stay a non-degenerate function of the model's own
#: decoded motion. Everything else is reported and can never be quoted as a fix.
ELIGIBLE = (
    [f"blend{a:g}" for a in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)]
    + [f"steerclip{s:g}" for s in (0.02, 0.005)]
    + [f"accelclip{c:g}" for c in (1.0, 0.3)]
    + [f"ema{b:g}" for b in (0.5, 0.8, 0.95)]
    + [f"every{m}" for m in (2, 5, 20)]
    + ["own_vupd"])
DIAGNOSTIC = (["chansteer", "chanaccel", "steerclip0", "accelclip0", "gtkin"]
              + [f"ownbefore{m}" for m in (5, 10, 20, 40)]
              + [f"ownafter{m}" for m in (5, 10, 20, 40)])
BARS = (1.0, 1.391, 2.0)


# --------------------------------------------------------------------------- #
def t_useful(de, bar: float) -> float:
    """Largest horizon (s) with mean ``de`` below ``bar``, contiguous from step 1.

    ⚠️ Returns **0.0** when even the first step is already above the bar — a
    reachable failing value, not a structural minimum.
    """
    m = de.mean(axis=0)
    ok = m < bar
    if not ok[0]:
        return 0.0
    bad = np.flatnonzero(~ok)
    return round(float(int(bad[0]) if bad.size else int(ok.size)) * DT, 2)


def t_blind_paired_gain(de_a, de_b, de_a0, de_b0, draws) -> dict:
    """Paired bootstrap of ``T_blind(intervention) - T_blind(baseline)``.

    Both contiguity curves are re-derived INSIDE each episode resample, so the
    comparison is paired at the cluster level rather than two marginal intervals
    put side by side.
    """
    ti, t0 = np.empty(len(draws)), np.empty(len(draws))
    for i, s in enumerate(draws):
        ti[i] = t_contiguous(de_b[s].mean(axis=0) - de_a[s].mean(axis=0))
        t0[i] = t_contiguous(de_b0[s].mean(axis=0) - de_a0[s].mean(axis=0))
    d = ti - t0
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"median_gain_steps": float(np.median(d)),
            "ci95_steps": [float(lo), float(hi)],
            "separated_better": bool(lo > 0),
            "frac_draws_gain_gt_0": round(float((d > 0).mean()), 4),
            "n_boot": B_BOOT, "estimator": "paired_episode_cluster_bootstrap"}


def arm_block(de, eid, draws, a_arm, b_arm, base_a=None, base_b=None) -> dict:
    """Everything reported for one intervention, with its matched comparator."""
    out = t_blind(de[a_arm], de[b_arm], draws, label_a=a_arm, label_b=b_arm)
    out["de_at_2s"] = single_at(de[a_arm], eid, draws, 20)
    out["de_at_6s"] = single_at(de[a_arm], eid, draws, 60)
    out["ade_0_2s"] = ade_0_2s(de[a_arm], eid)
    out["paired_delta_2s_vs_comparator"] = paired_at(de[a_arm], de[b_arm],
                                                     draws, 20)
    out["beats_cv"] = separated_better_interval(de[a_arm],
                                                de["d_constant_velocity"], draws)
    out["T_useful_s"] = {f"{b:g}m": t_useful(de[a_arm], b) for b in BARS}
    if base_a is not None:
        out["gain_vs_own_baseline"] = t_blind_paired_gain(
            de[a_arm], de[b_arm], de[base_a], de[base_b], draws)
        out["de_2s_delta_vs_own_baseline"] = paired_at(de[a_arm], de[base_a],
                                                       draws, 20)
    return out


def _log_slope(x, y):
    """Slope of log(y) on log(x) WITH its window, R^2 and n (CLAUDE.md rule)."""
    m = (np.asarray(x) > 0) & (np.asarray(y) > 0)
    lx, ly = np.log(np.asarray(x)[m]), np.log(np.asarray(y)[m])
    if lx.size < 3:
        return None
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss = 1.0 - ((ly - pred) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-12)
    return {"exponent": round(float(b), 3), "r2": round(float(ss), 3),
            "n": int(lx.size), "fit_window_steps": [int(x[0]), int(x[-1])],
            "admissible_per_CLAUDE_md": bool(ss >= 0.80)}


# =========================================================================== #
def stage_gates(new, bi_d, mat, de, eid, draws, out: Path) -> dict:
    de_bi = {k: v.double().numpy() for k, v in bi_d["dense_de_headline"].items()}
    de_mat = {k: v.double().numpy()
              for k, v in mat["dense_de_matched_arms"].items()}
    committed = dict(de_bi)
    committed.update(de_mat)

    gate = {"n_windows_new": int(de[A_OWN].shape[0]),
            "n_windows_committed": int(de_bi[A_OWN].shape[0]),
            "n_episode_clusters": len(set(eid)),
            "K_max": int(de[A_OWN].shape[1]),
            "eid_identical_vs_bi": bool([str(x) for x in bi_d["eid"]] == eid),
            "t0_identical_vs_bi": bool(list(map(int, bi_d["t0"]))
                                       == list(map(int, new["t0"]))),
            "eid_identical_vs_matched": bool([str(x) for x in mat["eid"]] == eid),
            "anchors": {}}
    for arm in (A_OWN, A_HOLD, B_OWN, B_HOLD):
        d = float(np.abs(de[arm] - committed[arm]).max())
        gate["anchors"][arm] = {"max_abs_diff_m": d,
                                "within_tol": bool(d < ANCHOR_TOL),
                                "tol_m": ANCHOR_TOL,
                                "committed_in": ("bi_perwindow_compact"
                                                 if arm in de_bi and arm not in de_mat
                                                 else "perwindow_matched_K185")}
    gate["GATE_PASS"] = bool(
        gate["eid_identical_vs_bi"] and gate["t0_identical_vs_bi"]
        and gate["eid_identical_vs_matched"]
        and gate["n_windows_new"] == gate["n_windows_committed"]
        and all(v["within_tol"] for v in gate["anchors"].values()))

    # ---- fidelity against the ladder's committed headline numbers -------- #
    own = t_blind(de[A_OWN], de[B_OWN], draws)
    hold = t_blind(de[A_HOLD], de[B_HOLD], draws)
    fid = {
        "T_blind_own_str": {"committed": LADDER["T_blind_own_str_steps"],
                            "recomputed": own["T_blind_steps"],
                            "committed_ci_s": LADDER["T_blind_own_str_ci_s"],
                            "recomputed_ci_s": own["T_blind_ci95_s"]},
        "T_blind_hold_str": {"committed": LADDER["T_blind_hold_str_steps"],
                             "recomputed": hold["T_blind_steps"],
                             "committed_ci_s": LADDER["T_blind_hold_str_ci_s"],
                             "recomputed_ci_s": hold["T_blind_ci95_s"]},
        "de2s_own_str": {"committed": LADDER["de2s_own_str"],
                         "recomputed": single_at(de[A_OWN], eid, draws, 20)["mean"]},
        "de2s_hold_str": {"committed": LADDER["de2s_hold_str"],
                          "recomputed": single_at(de[A_HOLD], eid, draws, 20)["mean"]},
        "ade_0_2s_own_str": {"committed": LADDER["ade_0_2s_own_str"],
                             "recomputed": ade_0_2s(de[A_OWN], eid)["mean"]},
        "ade_0_2s_hold_str": {"committed": LADDER["ade_0_2s_hold_str"],
                              "recomputed": ade_0_2s(de[A_HOLD], eid)["mean"]},
        "T_useful_1m_own_str_s": {"committed": LADDER["T_useful_1m_own_str_s"],
                                  "recomputed": t_useful(de[A_OWN], 1.0)},
        "T_useful_1m_hold_str_s": {"committed": LADDER["T_useful_1m_hold_str_s"],
                                   "recomputed": t_useful(de[A_HOLD], 1.0)},
    }
    # ⚠️ Split deliberately. The LEVEL agreement is BLOCKING: if `de@2s` or
    # `ade_0_2s` moved, something real changed and nothing below is readable.
    # The `T_blind` INTEGER reproduction is REPORTED, not blocking — these arms
    # come from a second encode pass (~3e-05 m of float-kernel noise, bounded by
    # the identity gate) and a step count is a threshold crossing, so a one-step
    # move would be a fact about the statistic's fragility that must be
    # published rather than a reason to discard the run. Only the pre-registered
    # gate (§2, anchors within 1e-4 m) blocks.
    fid["LEVEL_FIDELITY_PASS"] = bool(
        all(abs(v["recomputed"] - v["committed"]) < 1e-3
            for k, v in fid.items() if k.startswith(("de2s", "ade"))))
    fid["T_BLIND_EXACT_REPRODUCTION"] = bool(
        fid["T_blind_own_str"]["recomputed"] == LADDER["T_blind_own_str_steps"]
        and fid["T_blind_hold_str"]["recomputed"] == LADDER["T_blind_hold_str_steps"])
    fid["T_useful_reproduces"] = bool(
        fid["T_useful_1m_own_str_s"]["recomputed"] == LADDER["T_useful_1m_own_str_s"]
        and fid["T_useful_1m_hold_str_s"]["recomputed"] == LADDER["T_useful_1m_hold_str_s"])
    fid["FIDELITY_PASS"] = fid["LEVEL_FIDELITY_PASS"]

    # ---- §6c DIAGNOSTIC VACUITY AUDIT ------------------------------------- #
    audit = {
        "T_blind_rule": {
            "minimum_return_steps": 1, "maximum_return_steps": int(de[A_OWN].shape[1]),
            "failing_value_reachable": "MEASURED below (see reachability probe)",
        },
        "frac_draws_T_blind_at_floor_1step": {
            "admissible": True,
            "why": "both 0.000 and 1.000 are attainable; Rung 0 measured both"},
        "frac_draws_T_blind_is_zero": {
            "admissible": False,
            "why": "structurally 0 under A4 (minimum return is 1) — NOT emitted"},
        "selftest_blend0_equals_own": {
            "admissible_alone": False,
            "why": ("satisfied by a no-op implementation; admissible only "
                    "paired with the anti-no-op requirement, which is also run")},
        "t_useful": {"admissible": True,
                     "why": "returns 0.0 when step 1 is already above the bar"},
        "anti_noop": {"admissible": True,
                      "why": "a zero-difference arm is attainable and would fail"},
    }
    # reachability probe: the rule MUST return its failing value on a real arm
    probe = {}
    for name, aa, bb in (("identical_arms", A_OWN, A_OWN),
                         ("swapped_arms", B_OWN, A_OWN)):
        probe[name] = t_blind(de[aa], de[bb], draws)["T_blind_steps"]
    audit["failing_value_probe"] = {
        "identical_arms_steps": probe["identical_arms"],
        "swapped_arms_steps": probe["swapped_arms"],
        "both_must_be_1": bool(probe["identical_arms"] == 1
                               and probe["swapped_arms"] == 1)}

    res = {"window_set_identity_gate": gate,
           "plumbing_selftest": new.get("selftest", {}),
           "fidelity_vs_ladder": fid,
           "diagnostic_vacuity_audit": audit,
           "ALL_GATES_PASS": bool(
               gate["GATE_PASS"] and fid["FIDELITY_PASS"]
               and new.get("selftest", {}).get("SELFTEST_PASS", False)
               and audit["failing_value_probe"]["both_must_be_1"]),
           "meta": {"n_windows": gate["n_windows_new"],
                    "n_episode_clusters": gate["n_episode_clusters"],
                    "n_boot": B_BOOT, "seed": SEED,
                    "estimator": "paired_episode_cluster_bootstrap",
                    "arm_ckpt": new["meta"].get("arm_ckpt"),
                    "ckpt_step": new["meta"].get("ckpt_step"),
                    "n_arms": len(de)}}
    (out / "rung1_gates.json").write_text(json.dumps(res, indent=2, default=float),
                                          encoding="utf-8")
    print("GATES:", "PASS" if res["ALL_GATES_PASS"] else "FAIL")
    print(json.dumps({k: v for k, v in gate.items() if k != "anchors"}, indent=1))
    print(" anchors:", {k: round(v["max_abs_diff_m"], 8)
                        for k, v in gate["anchors"].items()})
    print(" fidelity:", json.dumps(fid, indent=1, default=float))
    return res


def stage_blend(de, eid, draws, out: Path) -> dict:
    alphas = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    curve = {}
    for a in alphas:
        if a == 0.0:
            aa, bb = A_OWN, B_OWN
        elif a == 1.0:
            aa, bb = A_HOLD, B_HOLD
        else:
            aa, bb = f"a_blend{a:g}", f"b_blend{a:g}"
        curve[f"{a:g}"] = arm_block(de, eid, draws, aa, bb, A_OWN, B_OWN)
        curve[f"{a:g}"]["eligible"] = bool(0.0 < a < 1.0)
        print("  alpha=%-6g T_blind=%3d steps (%4.1f s) CI %s  de@2s=%.4f "
              "beats_cv=%d" % (a, curve[f"{a:g}"]["T_blind_steps"],
                               curve[f"{a:g}"]["T_blind_s"],
                               curve[f"{a:g}"]["T_blind_ci95_s"],
                               curve[f"{a:g}"]["de_at_2s"]["mean"],
                               curve[f"{a:g}"]["beats_cv"]["n_steps"]))
    steps = [curve[f"{a:g}"]["T_blind_steps"] for a in alphas]
    res = {"curve": curve, "alphas": alphas, "T_blind_steps": steps,
           "monotone_nondecreasing": bool(all(steps[i] <= steps[i + 1]
                                              for i in range(len(steps) - 1))),
           "spearman_alpha_vs_T": round(float(np.corrcoef(
               np.argsort(np.argsort(alphas)),
               np.argsort(np.argsort(steps)))[0, 1]), 4),
           "note": ("alpha = 0 and alpha = 1 are the two MEASURED endpoints and "
                    "are NOT eligible interventions (alpha = 1 removes the "
                    "policy entirely); they anchor the curve.")}
    (out / "rung1_blend_curve.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def stage_mechanism(new, de, eid, draws, audit_pt, out: Path) -> dict:
    res = {}
    K = de[A_OWN].shape[1]
    grid = [1, 2, 5, 10, 20, 40, 80, 120, 185]

    # ---- (1) the penalty curve: own vs hold, comparator-free -------------- #
    m_own, m_hold = de[A_OWN].mean(axis=0), de[A_HOLD].mean(axis=0)
    pen = m_own - m_hold
    ns = np.array([n for n in grid if n <= K])
    res["penalty_own_minus_hold"] = {
        "at_steps": {f"{n}": round(float(pen[n - 1]), 4) for n in ns},
        "ratio_own_over_hold": {f"{n}": round(float(m_own[n - 1] / m_hold[n - 1]), 3)
                                for n in ns},
        "loglog_fit_full": _log_slope(np.arange(2, K + 1), pen[1:]),
        "loglog_fit_2_20": _log_slope(np.arange(2, 21), pen[1:20]),
        "loglog_fit_20_185": _log_slope(np.arange(20, K + 1), pen[19:]),
        "linear_reference": ("a penalty that is a LEVEL SHIFT has exponent ~0; "
                             "pure compounding drift is >= 1; the exponent is "
                             "quotable only at R2 >= 0.80 (CLAUDE.md)"),
    }

    # ---- (2) the ACTION statistics — saturation, magnitude, jitter -------- #
    act = {}
    for arm in new["psi"]:
        steer, accel = bi.reconstruct_kinematic_actions(
            new["psi"][arm], new["pred_speed"][arm], new["v_last"])
        s, a = steer.numpy(), accel.numpy()
        sat_s = (np.abs(s) >= bi.STEER_CLAMP - 1e-9)
        sat_a = (np.abs(a) >= bi.ACCEL_CLAMP - 1e-9)
        jit_s = np.abs(np.diff(s, axis=1))
        jit_a = np.abs(np.diff(a, axis=1))
        act[arm] = {
            "mean_abs_steer_rad": {f"{n}": round(float(np.abs(s[:, :n]).mean()), 5)
                                   for n in ns},
            "mean_abs_accel_ms2": {f"{n}": round(float(np.abs(a[:, :n]).mean()), 4)
                                   for n in ns},
            "frac_steer_at_clamp": {f"{n}": round(float(sat_s[:, :n].mean()), 4)
                                    for n in ns},
            "frac_accel_at_clamp": {f"{n}": round(float(sat_a[:, :n].mean()), 4)
                                    for n in ns},
            "frac_accel_at_clamp_by_window_of_step": {
                f"{n}": round(float(sat_a[:, max(0, n - 20):n].mean()), 4)
                for n in ns},
            "mean_step_jitter_steer": {f"{n}": round(float(jit_s[:, :max(1, n - 1)].mean()), 5)
                                       for n in ns},
            "mean_step_jitter_accel": {f"{n}": round(float(jit_a[:, :max(1, n - 1)].mean()), 4)
                                       for n in ns},
            "mean_pred_speed_ms": {f"{n}": round(float(new["pred_speed"][arm][:, :n].mean()), 3)
                                   for n in ns},
            "max_pred_speed_ms": round(float(new["pred_speed"][arm].max()), 2),
        }
    res["action_statistics_599_windows"] = act
    res["clamps"] = {"steer_rad": bi.STEER_CLAMP, "accel_ms2": bi.ACCEL_CLAMP,
                     "wheelbase_m": bi.WHEELBASE, "v_eps": bi.V_EPS,
                     "note": ("the deployed inverse ALREADY clamps; the blow-up "
                              "hypothesis testable here is SATURATION against "
                              "these bands, not unbounded actions")}

    # ---- (3) the ONSET sweep: own for m steps, then hold (and vice versa) - #
    onset = {}
    for m in (5, 10, 20, 40):
        for tag, arm in ((f"own_first_{m}_then_hold", f"a_ownbefore{m}"),
                         (f"hold_first_{m}_then_own", f"a_ownafter{m}")):
            if arm not in de:
                continue
            mm = de[arm].mean(axis=0)
            onset[tag] = {
                "de_at_steps": {f"{n}": round(float(mm[n - 1]), 4) for n in ns},
                "recovery_frac_of_own_hold_gap": {
                    f"{n}": (round(float((m_own[n - 1] - mm[n - 1])
                                         / max(m_own[n - 1] - m_hold[n - 1], 1e-9)), 4))
                    for n in ns},
            }
    res["onset_switch_sweep"] = onset
    res["onset_note"] = ("recovery = (de_own - de_arm) / (de_own - de_hold): 0 = "
                         "no better than letting the model act throughout, 1 = as "
                         "good as never letting it act at all. DIAGNOSTIC arms: "
                         "they read the horizon, so they are not deployable.")

    # ---- (4) the audit's own reconstruction gate + filtered fed actions --- #
    if audit_pt is not None:
        ad = torch.load(audit_pt, map_location="cpu", weights_only=False)
        res["reconstruction_gate_on_pod"] = ad["meta"]["reconstruction_gate"]
        fa = {}
        for name, t in ad["fed_actions"].items():
            fa[name] = {
                "mean_abs_steer_rad": round(float(t[:, :-1, 0].abs().mean()), 5),
                "mean_abs_accel_ms2": round(float(t[:, :-1, 1].abs().mean()), 4),
                "frac_steer_at_clamp": round(float(
                    (t[:, :-1, 0].abs() >= bi.STEER_CLAMP - 1e-9).float().mean()), 4),
                "frac_accel_at_clamp": round(float(
                    (t[:, :-1, 1].abs() >= bi.ACCEL_CLAMP - 1e-9).float().mean()), 4),
                "mean_step_jitter_accel": round(float(
                    t[:, :-1, 1].diff(dim=1).abs().mean()), 4),
                "mean_step_jitter_steer": round(float(
                    t[:, :-1, 0].diff(dim=1).abs().mean()), 5),
                "n_windows": int(t.shape[0])}
            fa[name]["FILTERED_dense_fed_actions"] = True
        res["fed_action_stats_audit_subset"] = fa
    (out / "rung1_mechanism.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def stage_other(de, eid, draws, out: Path) -> dict:
    fam = {
        "clip_steer": [f"steerclip{s:g}" for s in (0.02, 0.005, 0.0)],
        "clip_accel": [f"accelclip{c:g}" for c in (1.0, 0.3, 0.0)],
        "smooth_ema": [f"ema{b:g}" for b in (0.5, 0.8, 0.95)],
        "update_every": [f"every{m}" for m in (2, 5, 20)],
        "channel_decomposition": ["chansteer", "chanaccel"],
        "convention_and_speed": ["gtkin", "own_vupd"],
    }
    res = {}
    for fname, tags in fam.items():
        res[fname] = {}
        for tag in tags:
            aa, bb = f"a_{tag}", f"b_{tag}"
            if aa not in de or bb not in de:
                res[fname][tag] = {"MISSING": True}
                continue
            blk = arm_block(de, eid, draws, aa, bb, A_OWN, B_OWN)
            blk["eligible"] = tag in ELIGIBLE
            res[fname][tag] = blk
            print("  %-16s T_blind=%3d (%4.1f s) CI %s de@2s=%.4f elig=%s"
                  % (tag, blk["T_blind_steps"], blk["T_blind_s"],
                     blk["T_blind_ci95_s"], blk["de_at_2s"]["mean"],
                     blk["eligible"]))
    (out / "rung1_interventions.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def stage_verdict(de, eid, draws, blend, other, out: Path) -> dict:
    """PRE_REGISTRATION.md §5 — buckets fixed before any number existed."""
    cand = {}
    for a in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
        cand[f"blend{a:g}"] = blend["curve"][f"{a:g}"]
    for fam in other.values():
        for tag, blk in fam.items():
            if isinstance(blk, dict) and blk.get("eligible"):
                cand[tag] = blk
    assert set(cand) == set(ELIGIBLE), (sorted(set(cand) ^ set(ELIGIBLE)))
    n_elig = len(cand)
    # the baseline and ceiling AS RE-ROLLED IN THIS RUN, not as inherited
    base_t = blend["curve"]["0"]["T_blind_steps"]
    ceil_t = blend["curve"]["1"]["T_blind_steps"]
    best = max(cand.items(), key=lambda kv: kv[1]["T_blind_steps"])
    t = best[1]["T_blind_steps"]
    gain = best[1]["gain_vs_own_baseline"]
    bonf = 1.0 - 0.05 / n_elig
    bucket = ("CONFIRM" if (t >= 50 and gain["separated_better"])
              else "REFUTE" if t <= 30 else "PARTIAL")
    if t >= 50 and not gain["separated_better"]:
        bucket = "PARTIAL"
    cap = {
        "beats_cv_steps": best[1]["beats_cv"]["n_steps"],
        "beats_cv_interval_s": [best[1]["beats_cv"]["first_s"],
                                best[1]["beats_cv"]["last_s"]],
        "T_useful_s": best[1]["T_useful_s"],
        "baseline_T_useful_1m_s": t_useful(de[A_OWN], 1.0),
        "capability_verdict": None}
    cap["capability_verdict"] = (
        "CONFIRM" if (cap["beats_cv_steps"] > 0
                      or cap["T_useful_s"]["1m"] > cap["baseline_T_useful_1m_s"])
        else "NEGATIVE — extension is against a FROZEN PERCEPT only")
    res = {
        "rule": ("PRE_REGISTRATION.md §5 — CONFIRM: best ELIGIBLE T_blind >= 50 "
                 "steps AND paired gain separated; PARTIAL: 31..49 separated, or "
                 ">=50 not separated; REFUTE: <= 30 steps. Baseline 25 steps, "
                 "ceiling 115 steps."),
        "n_eligible_arms": n_elig,
        "baseline_T_blind_steps_prereg": 25, "ceiling_T_blind_steps_prereg": 115,
        "baseline_T_blind_steps_this_run": base_t,
        "ceiling_T_blind_steps_this_run": ceil_t,
        "best_eligible": best[0],
        "best_T_blind_steps": t, "best_T_blind_s": round(t * DT, 2),
        "best_T_blind_ci95_s": best[1]["T_blind_ci95_s"],
        "paired_gain_vs_own": gain,
        "bonferroni_required_frac": round(bonf, 4),
        "bonferroni_met": bool(gain["frac_draws_gain_gt_0"] >= bonf),
        "frac_of_ceiling_recovered": round((t - base_t) / max(ceil_t - base_t, 1), 4),
        "VERDICT": bucket,
        "ranking": sorted(((k, v["T_blind_steps"], v["T_blind_s"])
                           for k, v in cand.items()),
                          key=lambda r: -r[1]),
        "CAPABILITY_CAP": cap,
        "dose_response": {
            "monotone_nondecreasing": blend["monotone_nondecreasing"],
            "spearman_alpha_vs_T_blind": blend["spearman_alpha_vs_T"],
            "why_it_matters": ("a best-of-N selection effect cannot manufacture a "
                               "monotone dose-response between two independently "
                               "MEASURED endpoints")},
    }
    (out / "rung1_verdict.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\nVERDICT:", json.dumps({k: v for k, v in res.items()
                                    if k not in ("ranking",)}, indent=1,
                                   default=float))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True)
    ap.add_argument("--bi", required=True)
    ap.add_argument("--matched", required=True)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    new = torch.load(a.new, map_location="cpu", weights_only=False)
    bi_d = torch.load(a.bi, map_location="cpu", weights_only=False)
    mat = torch.load(a.matched, map_location="cpu", weights_only=False)
    de = {k: v.double().numpy() for k, v in new["dense_de"].items()}
    eid = [str(x) for x in new["eid"]]
    draws, _ = draws_for(eid)

    g = stage_gates(new, bi_d, mat, de, eid, draws, out)
    if not g["ALL_GATES_PASS"]:
        (out / "rung1_BLOCKED.json").write_text(
            json.dumps({"blocked": True, "gates": g}, indent=2, default=float),
            encoding="utf-8")
        print("\n⛔ BLOCKED — gates did not pass; nothing further is read.")
        return 2
    print("\n[2/5] the blend curve")
    blend = stage_blend(de, eid, draws, out)
    print("\n[3/5] mechanism separation")
    stage_mechanism(new, de, eid, draws, a.audit, out)
    print("\n[4/5] the other intervention families")
    other = stage_other(de, eid, draws, out)
    print("\n[5/5] verdict")
    stage_verdict(de, eid, draws, blend, other, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
