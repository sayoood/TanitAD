#!/usr/bin/env python3
"""E-H3b — GoalFlow's LITERAL shadow rule, now that the cache exists.

``LAMBDA_TAU_SWEEP.md`` §8 lists this as NOT MEASURED, with the exact reason:

    "GoalFlow's OWN rule is a distance between the conditioned and
     unconditioned TRAJECTORIES. No staged artifact holds the neutral branch's
     trajectory -- only its per-window error."

``eh2_build_cache.py`` dumps ``pick_traj`` for BOTH goal modes, so the literal
rule is now computable.  This file is that one addition, run on the cache the
same pass produced.  CPU only; nothing is trained.

THE RULE (GoalFlow, Xing et al., CVPR 2025, as quoted in E-H3):
    if || produced_traj - neutral_traj || > threshold, the goal is unreliable
    -> emit the SHADOW (neutral) trajectory instead.

PRE-REGISTERED BAR, carried over from E-H3 (HIERARCHY_PRIOR_RESEARCH §6.4):
    CONFIRM if the rule beats the neutral-always arm (0.7620) on the deployable
    surface, paired and separated.  Otherwise REFUTE -- say so plainly.
The threshold is chosen LEAVE-ONE-EPISODE-OUT, never in-sample, because the
in-sample optimum of a 1-parameter rule is not a deployable number.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _c in (Path("/root/taniteval"), Path("/workspace/taniteval")):
    if (_c / "taniteval" / "ci.py").exists():
        sys.path.insert(0, str(_c))
        break
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402

B = 2000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/_eh2/eh2_cache.pt")
    ap.add_argument("--out", default="/workspace/_eh2/eh3_goalflow.json")
    ap.add_argument("--dump", default="/workspace/_eh2/eh3_goalflow_windows.pt")
    ap.add_argument("--n-thr", type=int, default=61)
    a = ap.parse_args()
    t0 = time.time()
    C = torch.load(a.cache, map_location="cpu", weights_only=False)
    need = ("pick_traj", "neutral|pick_traj", "fan_err", "neutral|fan_err",
            "ref_sel_idx", "neutral|ref_sel_idx", "ep")
    missing = [k for k in need if k not in C]
    if missing:
        raise SystemExit(f"cache lacks {missing} -- rebuild with "
                         f"eh2_build_cache.py --goal-modes produced,neutral")

    pt = C["pick_traj"].double()                    # [W, 20, 2]
    nt = C["neutral|pick_traj"].double()
    fe = C["fan_err"].double()
    nfe = C["neutral|fan_err"].double()
    ep = [str(int(x)) for x in C["ep"]]
    W = pt.shape[0]
    ade_p = fe.gather(1, C["ref_sel_idx"][:, None].long()).squeeze(1).numpy()
    ade_n = nfe.gather(1, C["neutral|ref_sel_idx"][:, None].long()
                       ).squeeze(1).numpy()
    # the DEVIATION signal, exactly GoalFlow's: distance between the two picks
    dev = (pt - nt).norm(dim=-1)                    # [W, 20]
    sig = {"mean_dense": dev.mean(1).numpy(),
           "endpoint": dev[:, -1].numpy(),
           "max_dense": dev.max(1).values.numpy()}

    R = {
        "_experiment": "E-H3b -- GoalFlow's LITERAL shadow rule on the E-H2 "
                       "cache (the neutral branch's TRAJECTORY, which no "
                       "previously staged artifact held)",
        "_evidence_class": "MEASURED (ours)",
        "_estimator": "paired_episode_cluster_bootstrap (B=2000, unit = episode "
                      "cluster). NEVER overlapping_holdout_se.",
        "_host": platform.node(), "_n_windows": int(W),
        "_n_episodes": len(set(ep)),
        "arms": {
            "produced_always": round(float(ade_p.mean()), 4),
            "neutral_always": round(float(ade_n.mean()), 4),
            "ORACLE_shadow_bound_not_deployable": round(
                float(np.minimum(ade_p, ade_n).mean()), 4),
            "frac_windows_produced_better": round(
                float((ade_p < ade_n).mean()), 4)},
        "deviation_signal_stats": {
            k: {"mean": round(float(v.mean()), 4),
                "p50": round(float(np.median(v)), 4),
                "p95": round(float(np.percentile(v, 95)), 4),
                "max": round(float(v.max()), 4)} for k, v in sig.items()},
    }

    rules = {}
    for name, s in sig.items():
        thr = np.quantile(s, np.linspace(0.0, 1.0, a.n_thr))
        # LEAVE-ONE-EPISODE-OUT threshold choice
        eps = sorted(set(ep))
        epa = np.array(ep)
        loo = np.zeros(W)
        for e in eps:
            tr = epa != e
            best_t, best_v = thr[0], np.inf
            for t in thr:
                pick = np.where(s[tr] > t, ade_n[tr], ade_p[tr])
                if pick.mean() < best_v:
                    best_v, best_t = pick.mean(), t
            m = ~tr
            loo[m] = np.where(s[m] > best_t, ade_n[m], ade_p[m])
        ins = min(np.where(s > t, ade_n, ade_p).mean() for t in thr)
        rules[name] = {
            "leave_one_episode_out_ade": round(float(loo.mean()), 4),
            "in_sample_ceiling_NOT_deployable": round(float(ins), 4),
            "vs_neutral_always": paired_episode_cluster_bootstrap(
                loo, ade_n, ep, n_boot=B, seed=0),
            "vs_produced_always": paired_episode_cluster_bootstrap(
                loo, ade_p, ep, n_boot=B, seed=0),
            "_per_window": loo}
    R["deployable_rules"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "_per_window"}
        for k, v in rules.items()}
    best = min(rules, key=lambda k: rules[k]["leave_one_episode_out_ade"])
    bv = rules[best]["vs_neutral_always"]
    R["VERDICT"] = {
        "verdict": ("CONFIRM" if (bv["separated"] and bv["delta"] < 0)
                    else "REFUTE"),
        "best_signal": best,
        "_bar": "CONFIRM iff the best deployable shadow rule is paired-and-"
                "separated BETTER than the neutral-always arm. Otherwise "
                "REFUTE -- stated plainly, never re-scoped.",
        "_note": "the ORACLE shadow bound is reported for scale only and is "
                 "NOT a deployable number."}
    torch.save({"ade_produced": torch.from_numpy(ade_p),
                "ade_neutral": torch.from_numpy(ade_n),
                "ep": C["ep"],
                **{f"signal|{k}": torch.from_numpy(v) for k, v in sig.items()},
                **{f"loo|{k}": torch.from_numpy(v["_per_window"])
                   for k, v in rules.items()},
                "_dump_IS": "per-window ade for the produced and neutral picks, "
                            "the GoalFlow deviation signals, and each rule's "
                            "leave-one-episode-out output"}, a.dump)
    R["_wallclock_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
