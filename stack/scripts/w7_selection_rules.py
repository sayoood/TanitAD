"""W7 SELECTION-RULE SWEEP — is argmin the problem, or is the cost?

W7-FULL measured a paradox: over 256 candidates the roll-cost's WITHIN-window
rank correlation is ρ≈0.45–0.50, yet `argmin` selects 26× the oracle. That is
the signature of a winner's curse — with a noisily-correlated cost, the minimum
is where the cost most UNDER-estimates, so enlarging the fan hurts the minimiser.

This re-scores the BANKED per-window arrays (no model, no GPU) under selection
rules that are robust to that noise, and reports each against the same oracle:

  argmin            the incumbent
  top-m centroid    mean waypoints of the m lowest-cost candidates
  top-m medoid      the top-m candidate closest to that centroid (a REAL plan)
  rank-blend        argmin of (cost rank + kinematic rank), ranks not values

Pre-registration: this is EXPLORATORY — it re-uses the windows W7 was scored on,
so any winning rule must be re-measured on a fresh grid before it is deployed.
Reported as a rule COMPARISON, never as a new v5.8f number.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """[N, T, 2] vs [N, T, 2] -> [N] mean displacement error."""
    return np.linalg.norm(pred - gt, axis=-1).mean(-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w7_selection_rules")
    ap.add_argument("--windows", required=True, help="w7_eval_windows.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", default="1,2,3,5,8,16,32")
    a = ap.parse_args(argv)

    import torch
    d = torch.load(a.windows, map_location="cpu", weights_only=False)
    keys = sorted(d.keys()) if isinstance(d, dict) else []
    print(f"[rules] keys: {keys}", flush=True)

    def pick(*names):
        for n in names:
            if isinstance(d, dict) and n in d:
                return d[n]
        return None

    fan = pick("fan", "fan_traj", "anchor_traj", "cand_traj")
    gt = pick("gt", "gt_dense", "target", "traj_tgt")
    cost = pick("cost", "total_cost", "w7_cost", "costs")
    fan_err = pick("fan_err_ade", "fan_err", "cand_err", "per_candidate_err")
    kin = pick("kincost", "kin_cost", "kinematic_cost", "cost_kin")

    if cost is None:
        print(json.dumps({"status": "no cost array in dump", "keys": keys}))
        return 1
    cost = np.asarray(torch.as_tensor(cost).float().numpy())      # [N, C]

    # per-candidate error: either banked directly, or computed from fan vs gt
    if fan_err is not None:
        err = np.asarray(torch.as_tensor(fan_err).float().numpy())
    elif fan is not None and gt is not None:
        f = torch.as_tensor(fan).float().numpy()                  # [N, C, T, 2]
        g = torch.as_tensor(gt).float().numpy()                   # [N, T, 2]
        err = np.linalg.norm(f - g[:, None], axis=-1).mean(-1)
    else:
        print(json.dumps({"status": "no per-candidate error available",
                          "keys": keys}))
        return 1

    N, C = err.shape
    oracle = err.min(1)
    res = {
        "n_windows": int(N), "n_candidates": int(C),
        "_class": "EXPLORATORY — re-uses the W7 scoring windows; any winning "
                  "rule must be re-measured on a fresh grid before deployment",
        "oracle_ade": float(oracle.mean()),
        "rules": {},
    }
    order = np.argsort(cost, axis=1)                              # ascending

    res["rules"]["argmin"] = {"ade": float(err[np.arange(N), order[:, 0]].mean())}

    # rule: among the m lowest-cost, take the one whose ERROR the cost would
    # predict best is not knowable — so use geometry: the medoid of the top-m
    # plans (robust to a single cost outlier) and the centroid as an upper bound
    if fan is not None and gt is not None:
        f = torch.as_tensor(fan).float().numpy()
        g = torch.as_tensor(gt).float().numpy()
        for m in [int(x) for x in a.ms.split(",") if x]:
            if m > C:
                continue
            idx = order[:, :m]                                    # [N, m]
            sub = np.take_along_axis(f, idx[:, :, None, None], axis=1)
            cen = sub.mean(1)                                     # [N, T, 2]
            res["rules"][f"top{m}_centroid"] = {"ade": float(ade(cen, g).mean())}
            dist = np.linalg.norm(sub - cen[:, None], axis=-1).mean(-1)  # [N,m]
            med = np.argmin(dist, axis=1)
            chosen = idx[np.arange(N), med]
            res["rules"][f"top{m}_medoid"] = {
                "ade": float(err[np.arange(N), chosen].mean())}
    else:
        for m in [int(x) for x in a.ms.split(",") if x]:
            if m > C:
                continue
            idx = order[:, :m]
            # without trajectories, the medoid is unavailable; report the
            # best-achievable-within-top-m as a diagnostic ceiling instead
            res["rules"][f"top{m}_ceiling"] = {
                "ade": float(np.take_along_axis(err, idx, 1).min(1).mean())}
            res["rules"][f"top{m}_mean_err"] = {
                "ade": float(np.take_along_axis(err, idx, 1).mean(1).mean())}

    if kin is not None:
        k = np.asarray(torch.as_tensor(kin).float().numpy())
        rc = np.argsort(np.argsort(cost, 1), 1)
        rk = np.argsort(np.argsort(k, 1), 1)
        blend = np.argmin(rc + rk, axis=1)
        res["rules"]["rank_blend"] = {"ade": float(err[np.arange(N), blend].mean())}

    # the winner's-curse diagnostic itself: does the cost's rank of the CHOSEN
    # candidate sit far from the error's rank?
    err_rank = np.argsort(np.argsort(err, 1), 1)
    res["winners_curse"] = {
        "mean_err_rank_of_argmin": float(err_rank[np.arange(N), order[:, 0]].mean()),
        "n_candidates": int(C),
        "read": "if the argmin's ERROR-rank is far above 0, the cost minimum is "
                "not the error minimum — the deeper the fan, the worse argmin",
    }
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:1400], flush=True)
    print("W7RULES_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
