"""Bootstrap / multi-seed wrapper for the D1 decode gate (Benchmarks & Eval).

Motivation (measured, 2026-07-11 power audit): the shipped `run_d1` reports
`ade@1s` from a SINGLE fixed seed=0 episode split. At the val sizes we run (4-9
episodes) that single number swings 5-7 m across split seeds on the *same*
checkpoint (audit §3B) — larger than gate-movement deltas the program tries to
read. A D1 number is only a decision-grade claim as a mean +/- CI over seeds.

This wrapper is additive: it calls the unchanged `run_d1` over `seeds` splits and
returns the sampling distribution. Proposed target: a `run_d1_bootstrap` helper
in `stack/tanitad/eval/gates.py` (next to `run_d1`), or exposed via
`evaluate_checkpoint.py --d1-seeds N`.

Standalone (this package):
    from run_d1_bootstrap import run_d1_bootstrap
    stats = run_d1_bootstrap(states, targets_xy, episode_ids, seeds=8)
"""

from __future__ import annotations

from typing import Sequence

import torch

from tanitad.eval.gates import run_d1


def run_d1_bootstrap(states: torch.Tensor, targets_xy: torch.Tensor,
                     episode_ids: Sequence[int], *, seeds: int = 8,
                     unit: str = "camera", alpha: float = 1e-3,
                     val_frac: float = 0.2, **run_d1_kwargs) -> dict:
    """Report D1 ADE@1s as mean +/- 95% CI over `seeds` episode splits.

    Every kwarg is forwarded to `run_d1` unchanged; only the split `seed` varies,
    so the wrapper cannot alter the estimator — it only characterises its variance.
    """
    if seeds < 2:
        raise ValueError("run_d1_bootstrap needs seeds >= 2; use run_d1 for a point read")
    n_val = round(len(set(int(e) for e in episode_ids)) * val_frac)
    ades, fdes = [], []
    for sd in range(seeds):
        r = run_d1(states, targets_xy, episode_ids, unit=unit, alpha=alpha,
                   val_frac=val_frac, seed=sd, **run_d1_kwargs)
        ades.append(r.metrics["ade@1s"])
        fdes.append(r.metrics["fde@1s"])
    a = torch.tensor(ades)
    lo, hi = torch.quantile(a, torch.tensor([0.025, 0.975]))
    halfwidth = float((hi - lo) / 2)
    return {
        "ade@1s_mean": round(float(a.mean()), 3),
        "ade@1s_sd": round(float(a.std()), 3),
        "ade@1s_ci95": [round(float(lo), 3), round(float(hi), 3)],
        "ade@1s_ci95_halfwidth": round(halfwidth, 3),
        "fde@1s_mean": round(float(torch.tensor(fdes).mean()), 3),
        "seeds": seeds,
        "n_val_eps_approx": n_val,
        "single_seed0": round(ades[0], 3),
        "decision_grade": bool(halfwidth < 3.17 and n_val >= 20),
        "note": ("D1/D3 are open-loop decode gates: weak claims. Report this "
                 "mean+/-CI, never a single-seed point (audit 2026-07-11). "
                 "decision_grade requires CI half-width < 3.17 m AND n_val_eps >= 20."),
    }
