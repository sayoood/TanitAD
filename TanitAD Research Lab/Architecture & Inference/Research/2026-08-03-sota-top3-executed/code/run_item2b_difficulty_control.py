#!/usr/bin/env python3
"""E2e — THE CONFOUND CONTROL that E2d's own output demanded. 0 GPU.

⚠️ **This control is POST-HOC and is labelled as such.** It was not in `../PRE_REGISTRATION.md`; it
was forced by E2d's result, where the **hold-v0 CV baseline's** open-loop error predicts the MODEL's
closed-loop failure *better than the model's own open-loop error does* on 3 of 4 arms. That can only
mean the open->closed correlation is carried substantially by **WINDOW DIFFICULTY** — hard windows
are hard for every arm — rather than by anything the arm's open-loop score says about the arm.

So the question that actually decides whether open-loop scoring is worth anything becomes:

    does an arm's OWN open-loop error predict its closed-loop failure ABOVE AND BEYOND
    a difficulty proxy that knows nothing about the arm?

Implemented as a **rank partial correlation**: rank-transform all three series, linearly residualise
both the predictor and the outcome on the difficulty proxy (CV's open-loop ADE on the same window),
and correlate the residuals — with the same episode-cluster bootstrap on the statistic itself.

⛔ It is a control, not a promotion path. Nothing here may be quoted as a registered outcome; it
generates the hypothesis that a REPLICATION must pre-register.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))

from taniteval import ci as CI                      # noqa: E402
from taniteval import four_families as FF           # noqa: E402
from taniteval import progress as PROG              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_item2_progress_and_predictivity import (   # noqa: E402
    ade_per_window, rankdata, spearman)

CLBANK = (REPO / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation" / "incoming"
          / "2026-07-26-closedloop-artifact-rerun" / "raw_windows")
OUT = Path(__file__).resolve().parents[1] / "raw"
B, SEED = 2000, 0


def _resid(y, x):
    """Residual of y on x, both already rank-transformed. Plain least squares."""
    x = x - x.mean()
    y = y - y.mean()
    denom = float((x * x).sum())
    beta = float((x * y).sum()) / denom if denom > 0 else 0.0
    return y - beta * x


def partial_spearman(a, b, ctrl):
    """rho(a, b | ctrl) on ranks."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(ctrl)
    return spearman(_resid(ra, rc), _resid(rb, rc))


def stat_ci(fn, arrays, eid, n_boot=B, seed=SEED):
    """Episode-cluster bootstrap on ANY statistic of aligned per-window arrays."""
    stacked = np.stack(arrays, axis=1)
    uniq, idx_by_ep = CI.episode_index(eid)
    point = fn(*[stacked[:, i] for i in range(stacked.shape[1])])
    d = []
    for sel in CI._draws(uniq, idx_by_ep, n_boot, seed):
        s = stacked[sel]
        d.append(fn(*[s[:, i] for i in range(s.shape[1])]))
    d = np.asarray(d, float)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"stat": round(float(point), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "separated_from_0": bool(lo > 0 or hi < 0),
            "n_windows": int(len(stacked)), "n_episodes": int(len(uniq)), "n_boot": n_boot,
            "estimator": "episode_cluster_bootstrap_on_the_statistic"}


def main():
    t0 = time.time()
    out = {"run": "E2e — window-difficulty confound control (POST-HOC, not pre-registered)",
           "date": "2026-08-03", "host": "dev-box CPU", "gpu_hours": 0.0,
           "evidence_class": "MEASURED (ours)",
           "difficulty_proxy": ("hold-v0 CV's own open-loop ADE@2s on the same window — it knows "
                                "NOTHING about the arm, so anything it explains is difficulty"),
           "arms": {}}
    for p in sorted(CLBANK.glob("clwin_*.pt")):
        name = p.stem.replace("clwin_", "")
        d = torch.load(p, map_location="cpu", weights_only=False)
        if "eid" not in d:
            continue
        g, eid = d["gt"], list(d["eid"])
        closed = ade_per_window(d["closed_bike"], g)
        diff = ade_per_window(d["cv"], g)                      # the difficulty proxy
        P_ = FF._seq_geometry(torch.as_tensor(d["open_grnd"]).float(), 0.5)
        G_ = FF._seq_geometry(torch.as_tensor(g).float(), 0.5)
        cand = {
            "ade_0_2s": ade_per_window(d["open_grnd"], g),
            "progress_error": PROG.progress_per_window(d["open_grnd"].numpy(), g.numpy())["error"],
            "cross_mae": (P_["cross"] - G_["cross"]).abs().mean(1).numpy(),
            "heading_mae": (P_["heading"] - G_["heading"]).abs().mean(1).numpy(),
            "yaw_rate_mae": (P_["yaw_rate"] - G_["yaw_rate"]).abs().mean(1).numpy(),
            "speed_bias_abs": (P_["speed"] - G_["speed"]).mean(1).abs().numpy(),
        }
        blk = {"n_windows": len(eid), "n_episodes": len(set(eid)),
               "difficulty_alone": stat_ci(spearman, [diff, closed], eid)}
        for k, v in cand.items():
            v = np.asarray(v, float)
            ok = ~np.isnan(v) & ~np.isnan(closed) & ~np.isnan(diff)
            e_ = [x for x, m in zip(eid, ok) if m]
            blk[k] = {
                "raw_rho": stat_ci(spearman, [v[ok], closed[ok]], e_),
                "partial_rho_given_difficulty": stat_ci(
                    partial_spearman, [v[ok], closed[ok], diff[ok]], e_),
            }
        out["arms"][name] = blk
        print(f"{name}: difficulty alone rho={blk['difficulty_alone']['stat']}  "
              f"ade raw={blk['ade_0_2s']['raw_rho']['stat']} "
              f"partial={blk['ade_0_2s']['partial_rho_given_difficulty']['stat']}"
              f"{'*' if blk['ade_0_2s']['partial_rho_given_difficulty']['separated_from_0'] else ''}"
              f"  cross raw={blk['cross_mae']['raw_rho']['stat']} "
              f"partial={blk['cross_mae']['partial_rho_given_difficulty']['stat']}"
              f"{'*' if blk['cross_mae']['partial_rho_given_difficulty']['separated_from_0'] else ''}")
    out["wall_clock_s"] = round(time.time() - t0, 1)
    (OUT / "item2b_difficulty_control.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT / "item2b_difficulty_control.json", out["wall_clock_s"], "s")


if __name__ == "__main__":
    main()
