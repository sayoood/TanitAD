#!/usr/bin/env python3
"""E-V5-1 post-hoc — the MECHANISM behind the REFUTE, plus the axis and stratum
intervals the headline needs.  CPU-only, runs off the persisted per-window dump.

"Verify before alarming" (CLAUDE.md): a scoring rule that lands near a RANDOM
pick is as likely to be a bug as a finding.  This file exists to tell those apart
by measuring WHAT the rule picks, not by reasoning about it.

Emits:
  1. per-arm paired episode-cluster CIs on the ALONG / CROSS axes separately
     (v4's regression was 100% longitudinal; an undecomposed number hides it);
  2. per-stratum paired CIs for EVERY arm, not just A1 (the main run only paired
     A1 vs A0 inside strata);
  3. the mechanism: the terminal along-track displacement of each rule's pick
     against ground truth, which says whether a bad rule picks trajectories that
     are too SHORT or too LONG;
  4. the deployability question for the canary gate, asked honestly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

WP_STEPS = (5, 10, 15, 20)
N_BOOT, SEED = 2000, 20260726


def _boot(v, ep, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    by = [np.where(ep == e)[0] for e in eps]
    return np.asarray([float(v[np.concatenate(
        [by[j] for j in rng.integers(0, len(eps), len(eps))])].mean())
        for _ in range(n_boot)])


def paired(a, b, ep):
    d = _boot(a - b, ep)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float((a - b).mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "ci95": round(float((hi - lo) / 2), 4),
            "p_delta_gt0": round(float((d > 0).mean()), 4),
            "separated": bool(lo > 0 or hi < 0),
            "n_windows": int(len(a)), "n_episodes": int(len(np.unique(ep))),
            "n_boot": n_boot_of(), "estimator": "paired_episode_cluster_bootstrap",
            "_orientation": "a - b; NEGATIVE = a is BETTER"}


def n_boot_of():
    return N_BOOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/workspace/_v5/v5_v4_windows.pt")
    ap.add_argument("--out", default="/workspace/_v5/v5_posthoc.json")
    a = ap.parse_args()
    D = torch.load(a.dump, map_location="cpu", weights_only=False)

    fan = D["fan"].float()                 # [W, N, 20, 2]
    tgt = D["tgt"].float()                 # [W, 20, 2]
    ep = D["ep"].numpy()
    can = D["canary_err"].numpy()
    picks = D["picks"]
    W, N, K, _ = fan.shape
    wp = [k - 1 for k in WP_STEPS]

    R: dict = {
        "_experiment": "E-V5-1 post-hoc: mechanism + axis/stratum intervals",
        "_evidence_class": "MEASURED (ours)",
        "_estimator": "paired_episode_cluster_bootstrap (B=2000, unit = episode "
                      "cluster). NEVER overlapping_holdout_se.",
        "_n": {"windows": W, "candidates": N, "episodes": int(len(np.unique(ep)))},
    }

    # ---- per-arm axis arrays ---------------------------------------------
    axes: dict = {}
    for nm, idx in picks.items():
        tr = fan[torch.arange(W), idx.long()]                   # [W, 20, 2]
        r = tr - tgt
        axes[nm] = {
            "ade": (tr[:, wp] - tgt[:, wp]).norm(dim=-1).mean(dim=1).numpy(),
            "along": r[..., 0].abs().mean(dim=1).numpy(),
            "cross": r[..., 1].abs().mean(dim=1).numpy(),
            "term_along": tr[:, -1, 0].numpy(),                  # 2 s along-track
            "term_cross": tr[:, -1, 1].numpy(),
        }
    gt_term_along = tgt[:, -1, 0].numpy()
    gt_term_cross = tgt[:, -1, 1].numpy()

    # ---- 1+2. paired intervals: overall, per axis, per stratum -----------
    base = axes["A0_as_trained"]
    strata = {
        "ALL": np.ones(W, bool),
        "under_canary_bar_0.55": can <= 0.55,
        "over_canary_bar_0.55": can > 0.55,
        "q1_best_canary": can <= np.quantile(can, .25),
        "q4_worst_canary": can > np.quantile(can, .75),
    }
    out: dict = {}
    for sname, m in strata.items():
        row = {"n_windows": int(m.sum()),
               "n_episodes": int(len(np.unique(ep[m])))}
        for nm in picks:
            if nm == "A0_as_trained":
                continue
            row[nm] = {ax: paired(axes[nm][ax][m], base[ax][m], ep[m])
                       for ax in ("ade", "along", "cross")}
        out[sname] = row
    R["paired_vs_as_trained_by_axis_and_stratum"] = out

    # ---- 3. THE MECHANISM: what does each rule actually pick? ------------
    mech: dict = {"_read":
                  "term_along is the picked trajectory's 2 s along-track "
                  "displacement. A rule whose picks are systematically SHORTER or "
                  "LONGER than ground truth is failing on the longitudinal axis "
                  "by CONSTRUCTION, which is a mechanism, not noise."}
    for nm in picks:
        ta = axes[nm]["term_along"]
        mech[nm] = {
            "mean_term_along_m": round(float(ta.mean()), 3),
            "gt_mean_term_along_m": round(float(gt_term_along.mean()), 3),
            "bias_m": round(float((ta - gt_term_along).mean()), 3),
            "frac_picks_shorter_than_gt": round(
                float((ta < gt_term_along).mean()), 4),
            "p05_term_along_m": round(float(np.percentile(ta, 5)), 3),
            "p95_term_along_m": round(float(np.percentile(ta, 95)), 3),
            "mean_abs_term_cross_m": round(
                float(np.abs(axes[nm]["term_cross"]).mean()), 3),
        }
    mech["_gt"] = {"mean_term_along_m": round(float(gt_term_along.mean()), 3),
                   "p05": round(float(np.percentile(gt_term_along, 5)), 3),
                   "p95": round(float(np.percentile(gt_term_along, 95)), 3),
                   "mean_abs_term_cross_m": round(
                       float(np.abs(gt_term_cross).mean()), 3)}
    # the FAN's own longitudinal spread -- how extreme CAN a candidate be?
    fan_ta = fan[:, :, -1, 0]
    mech["_fan_envelope"] = {
        "min_term_along_m": round(float(fan_ta.min()), 3),
        "max_term_along_m": round(float(fan_ta.max()), 3),
        "per_window_span_mean_m": round(
            float((fan_ta.max(1).values - fan_ta.min(1).values).mean()), 3),
        "_read": "if the fan spans a huge longitudinal range, ANY rule that does "
                 "not control the along axis will land far from GT -- the span is "
                 "the size of the mistake available to be made.",
    }
    R["mechanism_what_each_rule_picks"] = mech

    # ---- 4. is the canary gate DEPLOYABLE? -------------------------------
    # The canary needs GT future poses, so it cannot be evaluated at deploy time.
    # Ask honestly whether anything observable predicts it: v0 is the only
    # deploy-time scalar in the dump.
    v0 = D["v0"].numpy()
    c = np.corrcoef(v0, can)[0, 1]
    R["canary_gate_deployability"] = {
        "corr_v0_vs_canary_err": round(float(c), 4),
        "canary_mean_by_v0_tercile": [
            round(float(can[v0 <= np.quantile(v0, 1 / 3)].mean()), 4),
            round(float(can[(v0 > np.quantile(v0, 1 / 3))
                            & (v0 <= np.quantile(v0, 2 / 3))].mean()), 4),
            round(float(can[v0 > np.quantile(v0, 2 / 3)].mean()), 4)],
        "_read": "⚠️ `wm_canary_ade_2s` is computed against GROUND-TRUTH future "
                 "poses, so it is NOT observable at deploy time. Any 'it works "
                 "where the WM is good' claim is therefore an ORACLE-GATED claim "
                 "unless an observable proxy is established. This row reports the "
                 "only deploy-time scalar available here (v0); it is not a proxy "
                 "search and does not license one.",
    }

    Path(a.out).write_text(json.dumps(R, indent=2))
    print(json.dumps({"mechanism": {k: v for k, v in mech.items()
                                    if not k.startswith("_")},
                      "fan_envelope": mech["_fan_envelope"],
                      "gt": mech["_gt"]}, indent=2))


if __name__ == "__main__":
    main()
