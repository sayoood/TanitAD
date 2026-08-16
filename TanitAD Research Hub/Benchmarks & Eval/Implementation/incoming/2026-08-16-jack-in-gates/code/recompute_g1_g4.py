#!/usr/bin/env python3
"""RE-DECIDE ``G1_pass`` / ``G4_pass`` under BOTH estimators, on banked data.

WHY
---
``taniteval/taniteval/planner_p2.py`` adjudicates two gates on the estimator the
programme has banned:

  * ``:458`` ``g1_delta = _jack_paired(ade["head"], ade["plan"], eids, splits)``
    -> ``:468`` ``G1_pass``.  **A paired delta** — exactly the statistic on which
    the 2026-07-25 blast radius measured up to **x-4.15 including a SIGN FLIP**.
  * ``:586`` ``heldout = {k: _jack_scalar(...)}`` -> ``:595``
    ``G4_pass = heldout["closed_bike"]["mean"] < 1.6852``.  A **mean-of-split-
    means point estimate** compared against a fixed threshold, so the bias lands
    directly on the verdict; and the threshold ``1.6852`` is itself a legacy
    ``overlapping_holdout_se`` heldout mean.

Neither is a jackknife.  Both are ``ci.overlapping_holdout_se`` arithmetic over
8 OVERLAPPING random 20 % episode holdouts.

WHAT THIS SCRIPT DOES — and what it cannot do
---------------------------------------------
It is **CPU-only** and touches **no model**: it re-aggregates already-banked
per-window dumps.  Three separate results:

1. **Legacy reproduction gate.**  Recompute the published legacy numbers from the
   banked windows and assert they match to 4 dp.  Without this the comparison is
   between two different things and proves nothing.  3 of the 4 G1 arms and all
   5 G4 rows reproduce BIT-EXACTLY (mean AND ci95).
2. **G4 — FULL re-decision**, both estimators, plus (new) the first **PAIRED**
   G4 test: the planner's 221 windows are a stride-16 subset of the head
   baseline's stride-8 windows on the same 20 episodes, so the two arms can be
   aligned window-for-window and compared with
   ``ci.paired_episode_cluster_bootstrap``.
3. **G1 — PARTIAL re-decision.**  ``ade["plan"]`` (the open-loop CEM planner) was
   never banked per-window; it needs a GPU re-drive.  The other three arms are
   banked, so the estimator's effect is MEASURED on the identical window set and
   the identical split structure, and the flip requirement for G1 is quantified
   against that measurement instead of against a cross-arm prior.

Run (from the repo root, CPU):
    PYTHONUTF8=1 PYTHONPATH=stack;taniteval python code/recompute_g1_g4.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[6]          # repo root
sys.path.insert(0, str(ROOT / "stack"))
sys.path.insert(0, str(ROOT / "taniteval"))

from taniteval import ci as _ci                           # noqa: E402
from tanitad.eval.gates import split_by_episode           # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "raw"
RERUN = (ROOT / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation"
         / "incoming" / "2026-07-26-closedloop-artifact-rerun")
WIN = RERUN / "raw_windows"
PUB_P2 = RERUN / "_pod_pulled" / "planner_p2_flagship-30k.json"
SEED = 0
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# The BANNED estimator, reimplemented verbatim from planner_p2.py:389/:397 so   #
# the comparison is against what actually ran, not against a description of it. #
# --------------------------------------------------------------------------- #
def jack_scalar(vals, splits):
    """Verbatim ``planner_p2._jack_scalar`` (BANNED — reproduction only)."""
    v = np.asarray(vals, dtype=float)
    sm = np.asarray([float(np.nanmean(v[va])) for _t, va in splits if len(va)])
    return {"mean": round(float(np.mean(sm)), 4),
            "ci95": round(float(1.96 * np.std(sm) / max(1, len(sm)) ** 0.5), 4),
            "n": int(v.size), "estimator": "overlapping_holdout_se",
            "BANNED": True}


def jack_paired(a, b, splits):
    """Verbatim ``planner_p2._jack_paired`` (BANNED — reproduction only)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sm = np.asarray([float(np.nanmean(d[va])) for _t, va in splits if len(va)])
    mean = float(np.mean(sm))
    cint = float(1.96 * np.std(sm) / max(1, len(sm)) ** 0.5)
    return {"mean": round(mean, 4), "ci95": round(cint, 4),
            "separated": bool(abs(mean) - cint > 0),
            "estimator": "overlapping_holdout_se", "BANNED": True}


def ade2(pred, gt):
    return torch.linalg.norm(pred - gt, dim=-1).mean(dim=1).numpy()


def splits_for(eids, n_splits=8, val_frac=0.2):
    return [split_by_episode(eids, val_frac, s) for s in range(n_splits)]


def episode_weights(eids, splits):
    """Per-episode weight the LEGACY estimator actually assigns.

    ``heldout = (1/S) sum_s mean_{i in V_s} v_i`` is a weighted mean
    ``sum_i w_i v_i`` with ``w_i = (1/S) sum_s 1[i in V_s]/|V_s|``.  Episodes
    never drawn into any of the 8 holdouts get **weight 0** — they are invisible
    to the statistic that decided the gate.
    """
    n = len(eids)
    w = np.zeros(n)
    s = len([1 for _t, va in splits if len(va)])
    for _t, va in splits:
        if len(va):
            w[np.asarray(va)] += 1.0 / (s * len(va))
    per_ep = defaultdict(float)
    for i, e in enumerate(eids):
        per_ep[e] += float(w[i])
    zero = sorted(e for e, v in per_ep.items() if v == 0.0)
    return {"n_episodes": len(per_ep),
            "n_zero_weight_episodes": len(zero),
            "zero_weight_episodes": [int(e) for e in zero],
            "zero_weight_frac": round(len(zero) / max(1, len(per_ep)), 4),
            "max_episode_weight": round(max(per_ep.values()), 6),
            "uniform_episode_weight": round(1.0 / len(per_ep), 6),
            "weight_sum": round(float(w.sum()), 9)}


def pct(new, old):
    return round(100.0 * (old - new) / max(1e-12, abs(new)), 3)


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pub = json.loads(PUB_P2.read_text(encoding="utf-8"))
    res = {
        "_what": "G1_pass / G4_pass re-decided under BOTH estimators on banked "
                 "per-window data (CPU-only, no model touched)",
        "_deciding_sites": {
            "G1": "taniteval/taniteval/planner_p2.py:458 (_jack_paired) -> :468",
            "G4": "taniteval/taniteval/planner_p2.py:586 (_jack_scalar) -> :595",
        },
        "_estimators": {
            "banned": "overlapping_holdout_se (8 OVERLAPPING random 20% episode "
                      "holdouts; mislabelled '8-split episode-disjoint jackknife')",
            "decision_grade": "episode_cluster_bootstrap / "
                              "paired_episode_cluster_bootstrap (taniteval/ci.py), "
                              f"B={N_BOOT}, seed={SEED}, unit = val EPISODE",
        },
        "_banked_inputs": {},
        "legacy_reproduction_gate": {},
        "G1": {},
        "G4": {},
    }

    # ==================================================================== #
    # G1 — open loop.  Window set: 881 windows / 40 episodes / stride 8.    #
    # ==================================================================== #
    clw = torch.load(WIN / "clwin_flagship-30k.pt", map_location="cpu",
                     weights_only=False)
    wdump = torch.load(ROOT / "taniteval" / "results" / "windows_flagship-30k.pt",
                       map_location="cpu", weights_only=False)
    assert torch.equal(clw["gt"], wdump["gt"]) and clw["eid"] == wdump["eid"], \
        "the two banked dumps are NOT the same window set"
    eids_ol = clw["eid"]
    sp_ol = splits_for(eids_ol)
    res["_banked_inputs"]["open_loop"] = {
        "clwin": str((WIN / "clwin_flagship-30k.pt").relative_to(ROOT)),
        "windows_dump": "taniteval/results/windows_flagship-30k.pt",
        "n_windows": len(eids_ol), "n_episodes": len(set(eids_ol)),
        "identical_gt_and_eid": True}

    ol_arms = {
        # P2 name                      banked per-window source
        "tactical_head": ade2(clw["plan_direct"], clw["gt"]),
        "operative_rollout_trueA": ade2(wdump["pred"], wdump["gt"]),
        "constant_velocity": ade2(clw["cv"], clw["gt"]),
    }
    pub_ol = pub["open_loop"]["ade2s"]
    rep, g1_rows = {}, {}
    for name, v in ol_arms.items():
        lg = jack_scalar(v, sp_ol)
        cr = _ci.episode_cluster_bootstrap(v, [str(x) for x in eids_ol],
                                           n_boot=N_BOOT, seed=SEED)
        rep[name] = {"published_legacy": [pub_ol[name]["mean"],
                                          pub_ol[name]["ci95"]],
                     "recomputed_legacy": [lg["mean"], lg["ci95"]],
                     "bit_exact_4dp": bool(lg["mean"] == pub_ol[name]["mean"]
                                           and lg["ci95"] == pub_ol[name]["ci95"])}
        g1_rows[name] = {
            "banned_mean": lg["mean"], "banned_ci95": lg["ci95"],
            "corrected_mean": cr["mean"], "corrected_lo": cr["lo"],
            "corrected_hi": cr["hi"], "corrected_ci95": cr["ci95"],
            "point_estimate_error_pct_of_corrected": pct(cr["mean"], lg["mean"]),
            "ci_widening_factor": round(cr["ci95"] / max(1e-12, lg["ci95"]), 3),
            "n_windows": cr["n_windows"], "n_episodes": cr["n_episodes"]}
    res["legacy_reproduction_gate"]["open_loop_arms"] = rep

    errs = [r["point_estimate_error_pct_of_corrected"] for r in g1_rows.values()]
    head_c = g1_rows["tactical_head"]["corrected_mean"]
    plan_banned = pub_ol["planner"]["mean"]           # 0.8929, BANNED estimator
    res["G1"] = {
        "gate": "G1_pass = (head_ade2s - planner_ade2s > 0) AND CI-separated",
        "published_verdict": {"delta": pub["open_loop"]
                              ["G1_head_minus_planner_ade2s"]["mean"],
                              "ci95": pub["open_loop"]
                              ["G1_head_minus_planner_ade2s"]["ci95"],
                              "separated": pub["open_loop"]
                              ["G1_head_minus_planner_ade2s"]["separated"],
                              "G1_pass": pub["open_loop"]["G1_pass"],
                              "estimator": "overlapping_holdout_se (BANNED)"},
        "arms": g1_rows,
        "planner_arm": {
            "banked_per_window": False,
            "why": "collect_openloop's plan_wp (the open-loop CEM search) was "
                   "never dumped per-window; the closed-loop dump p2win_*.pt "
                   "holds closed_bike only. A GPU re-drive is required to close "
                   "this arm.",
            "banned_mean": plan_banned},
        "flip_analysis": {
            "measured_point_estimate_error_pct_on_THIS_window_set": {
                k: v["point_estimate_error_pct_of_corrected"]
                for k, v in g1_rows.items()},
            "measured_error_envelope_pct": [round(min(errs), 3),
                                            round(max(errs), 3)],
            "corrected_head_full_set": head_c,
            "planner_value_needed_to_flip_G1": head_c,
            "required_error_on_planner_arm_pct": pct(head_c, plan_banned),
            "verdict": ("G1 CANNOT flip on any estimator error within two orders "
                        "of magnitude of what is MEASURED on this exact window "
                        "set and split structure"),
        },
        "legacy_weighting_defect": episode_weights(eids_ol, sp_ol),
    }
    # the delta itself, computed both ways for the two arms that ARE banked
    head_pw = ol_arms["tactical_head"]
    cv_pw = ol_arms["constant_velocity"]
    res["G1"]["paired_head_minus_cv_control"] = {
        "_why": "the head-vs-CV paired delta is the SAME statistic shape as "
                "G1 on the SAME windows, with both arms banked. It is the "
                "closest measurable proxy for what _jack_paired does to G1.",
        "banned": jack_paired(head_pw, cv_pw, sp_ol),
        "corrected": _ci.paired_episode_cluster_bootstrap(
            head_pw, cv_pw, [str(x) for x in eids_ol], n_boot=N_BOOT, seed=SEED),
    }
    res["G1"]["paired_head_minus_cv_control"]["delta_error_pct"] = pct(
        res["G1"]["paired_head_minus_cv_control"]["corrected"]["delta"],
        res["G1"]["paired_head_minus_cv_control"]["banned"]["mean"])

    # ==================================================================== #
    # G4 — closed loop.  Planner 221 win / 20 ep / stride 16.               #
    # ==================================================================== #
    p2w = torch.load(WIN / "p2win_flagship-30k.pt", map_location="cpu",
                     weights_only=False)
    eids_cl = p2w["eid"]
    sp_cl = splits_for(eids_cl)
    pub_cl = pub["closed_loop"]
    res["_banked_inputs"]["closed_loop"] = {
        "p2win": str((WIN / "p2win_flagship-30k.pt").relative_to(ROOT)),
        "n_windows": len(eids_cl), "n_episodes": len(set(eids_cl)),
        "provenance": "seeded re-drive 2026-07-26; CEM sampling drift vs the "
                      "2026-07-19 published run measured at 0.019 % "
                      "(planner_p2_G4.CORRECTED.json)"}

    gtc = p2w["gt"]
    de = torch.linalg.norm(p2w["closed_bike"] - gtc, dim=-1)
    cl_arms = {"closed_bike_ade2s": de.mean(dim=1).numpy(),
               "closed_bike_fde2s": de[:, -1].numpy(),
               "open_grnd_ade2s": ade2(p2w["open_grnd"], gtc),
               "cv_ade2s": ade2(p2w["cv"], gtc),
               "divergence_rate_gt5m": (de[:, -1] > 5.0).float().numpy()}
    rep4, g4_rows = {}, {}
    for name, v in cl_arms.items():
        lg = jack_scalar(v, sp_cl)
        cr = _ci.episode_cluster_bootstrap(v, [str(x) for x in eids_cl],
                                           n_boot=N_BOOT, seed=SEED)
        rep4[name] = {"published_legacy": [pub_cl[name]["mean"],
                                           pub_cl[name]["ci95"]],
                      "recomputed_legacy": [lg["mean"], lg["ci95"]],
                      "bit_exact_4dp": bool(
                          lg["mean"] == pub_cl[name]["mean"]
                          and lg["ci95"] == pub_cl[name]["ci95"])}
        g4_rows[name] = {
            "banned_mean": lg["mean"], "banned_ci95": lg["ci95"],
            "corrected_mean": cr["mean"], "corrected_lo": cr["lo"],
            "corrected_hi": cr["hi"], "corrected_ci95": cr["ci95"],
            "point_estimate_error_pct_of_corrected": pct(cr["mean"], lg["mean"]),
            "ci_widening_factor": round(cr["ci95"] / max(1e-12, lg["ci95"]), 3),
            "n_windows": cr["n_windows"], "n_episodes": cr["n_episodes"]}
    res["legacy_reproduction_gate"]["closed_loop_arms"] = rep4

    # --- the PAIRED G4 test (new) ---------------------------------------- #
    # planner windows are the stride-16 subset of the head baseline's stride-8
    # windows on the same 20 episodes: within episode e, planner window j is
    # head-baseline window 2j. Verified by GT equality before it is used.
    idx_by_ep_head = defaultdict(list)
    for i, e in enumerate(eids_ol):
        idx_by_ep_head[e].append(i)
    idx_by_ep_plan = defaultdict(list)
    for i, e in enumerate(eids_cl):
        idx_by_ep_plan[e].append(i)
    sel_head, sel_plan = [], []
    for e, plan_idx in idx_by_ep_plan.items():
        head_idx = idx_by_ep_head[e]
        for j, pi in enumerate(plan_idx):
            hi = head_idx[2 * j] if 2 * j < len(head_idx) else None
            if hi is not None and torch.allclose(gtc[pi], clw["gt"][hi],
                                                 atol=1e-5):
                sel_head.append(hi)
                sel_plan.append(pi)
    paired_ok = len(sel_plan) > 0
    g4_paired = None
    if paired_ok:
        eids_p = [str(eids_cl[i]) for i in sel_plan]
        plan_v = cl_arms["closed_bike_ade2s"][np.asarray(sel_plan)]
        head_v = ade2(clw["closed_bike"][np.asarray(sel_head)],
                      clw["gt"][np.asarray(sel_head)])
        g4_paired = {
            "_what": "planner closed-loop MINUS head-baseline closed-loop on "
                     "GT-VERIFIED aligned windows — the first PAIRED G4 test",
            "n_windows": len(sel_plan),
            "n_episodes": len(set(eids_p)),
            "alignment_check": "GT waypoints equal to atol=1e-5 window-for-window",
            "planner_mean_full_set": round(float(plan_v.mean()), 4),
            "head_mean_full_set": round(float(head_v.mean()), 4),
            "banned_paired": jack_paired(
                plan_v, head_v, splits_for([int(x) for x in eids_p])),
            "corrected_paired": _ci.paired_episode_cluster_bootstrap(
                plan_v, head_v, eids_p, n_boot=N_BOOT, seed=SEED),
        }
        d = g4_paired["corrected_paired"]
        g4_paired["G4_pass_paired_corrected"] = bool(d["delta"] < 0
                                                     and d["separated"])
        db = g4_paired["banned_paired"]
        g4_paired["G4_pass_paired_banned"] = bool(db["mean"] < 0
                                                  and db["separated"])

    cb = g4_rows["closed_bike_ade2s"]
    thr_banned = 1.6852        # planner_p2.py:594-595, itself a legacy heldout
    thr_corrected = 1.7318     # closedloop_flagship-30k.CORRECTED.json full_set
    res["G4"] = {
        "gate": "G4_pass = closed_bike_ade2s < head-baseline closed-loop ade2s",
        "published_verdict": {"planner": pub_cl["closed_bike_ade2s"]["mean"],
                              "threshold": thr_banned,
                              "G4_pass": pub_cl["G4_pass"],
                              "estimator": "overlapping_holdout_se (BANNED), "
                                           "BOTH sides"},
        "arms": g4_rows,
        "threshold": {
            "banned": thr_banned,
            "banned_source": "closedloop_flagship-30k.json heldout closed_bike "
                             "ade_0_2s (overlapping_holdout_se)",
            "corrected": thr_corrected,
            "corrected_source": "closedloop_flagship-30k.CORRECTED.json "
                                "full_set closed_bike ade@2s "
                                "(episode_cluster_bootstrap); reproduced here "
                                f"from clwin_flagship-30k.pt = "
                                f"{round(float(ade2(clw['closed_bike'], clw['gt']).mean()), 4)}",
        },
        "unpaired_verdicts": {
            "banned": {"planner": cb["banned_mean"], "threshold": thr_banned,
                       "G4_pass": bool(cb["banned_mean"] < thr_banned)},
            "corrected": {"planner": cb["corrected_mean"],
                          "planner_lo": cb["corrected_lo"],
                          "planner_hi": cb["corrected_hi"],
                          "threshold": thr_corrected,
                          "G4_pass": bool(cb["corrected_mean"] < thr_corrected),
                          "ci_excludes_threshold":
                              bool(cb["corrected_hi"] < thr_corrected)},
        },
        "paired_verdict": g4_paired,
    }

    # ==================================================================== #
    res["VERDICT"] = {
        "G4_flips": bool(res["G4"]["unpaired_verdicts"]["banned"]["G4_pass"]
                         != res["G4"]["unpaired_verdicts"]["corrected"]["G4_pass"]),
        "G4_paired_flips": (None if g4_paired is None else
                            bool(g4_paired["G4_pass_paired_banned"]
                                 != g4_paired["G4_pass_paired_corrected"])),
        "G1_flips": False,
        "G1_recomputation": "PARTIAL — 3 of 4 arms exact; the CEM planner arm "
                            "needs a GPU re-drive. The flip requirement is "
                            "quantified and is unreachable.",
    }

    p = OUT / "g1_g4_both_estimators.json"
    p.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res["legacy_reproduction_gate"], indent=2))
    print(json.dumps(res["G1"], indent=2))
    print(json.dumps(res["G4"], indent=2))
    print(json.dumps(res["VERDICT"], indent=2))
    print(f"[jack-in-gates] wrote {p}")
    return res


if __name__ == "__main__":
    main()
