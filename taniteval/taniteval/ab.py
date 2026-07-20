"""TanitEval — A/B comparison.

Paired, per-window comparison of two models on the SAME rollout windows:
win-rate, mean paired delta with a bootstrap CI (10k resamples), significance
(CI excludes 0), and per-stratum breakdown. Requires both runs to have been
collected over identical episodes/stride (the runner guarantees this)."""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/TanitAD/stack/scripts")
from driving_diagnostic import curvature_bucket  # noqa: E402


def _check_aligned(a, b):
    assert a["gt"].shape == b["gt"].shape, "window sets differ (shape)"
    assert torch.allclose(a["gt"], b["gt"], atol=1e-4), \
        "window sets differ (GT mismatch) — collect both models identically"


def compare(a, b, name_a="A", name_b="B", n_boot=10000, seed=0):
    """a/b: rollout window dicts. Positive delta => B better (lower ADE)."""
    _check_aligned(a, b)
    de_a = torch.linalg.norm(a["pred"] - a["gt"], dim=-1).mean(dim=1)  # ade0-2s
    de_b = torch.linalg.norm(b["pred"] - b["gt"], dim=-1).mean(dim=1)
    delta = (de_a - de_b).numpy()                 # >0 where B wins
    rng = np.random.default_rng(seed)
    boots = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    curv = [curvature_bucket(float(h)) for h in a["head_deg"]]
    strata = {}
    for lab in sorted(set(curv)):
        idx = np.array([i for i, l in enumerate(curv) if l == lab])
        strata[lab] = {"win_rate_b": round(float((delta[idx] > 0).mean()), 3),
                       "mean_delta_m": round(float(delta[idx].mean()), 4),
                       "n": int(len(idx))}
    return {
        "a": name_a, "b": name_b, "n_windows": int(len(delta)),
        "ade_a": round(float(de_a.mean()), 4),
        "ade_b": round(float(de_b.mean()), 4),
        "win_rate_b": round(float((delta > 0).mean()), 4),
        "mean_delta_m": round(float(delta.mean()), 4),
        "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "significant": bool(lo > 0 or hi < 0),
        "verdict": (name_b if lo > 0 else name_a if hi < 0 else "tie"),
        "by_curvature": strata,
    }
