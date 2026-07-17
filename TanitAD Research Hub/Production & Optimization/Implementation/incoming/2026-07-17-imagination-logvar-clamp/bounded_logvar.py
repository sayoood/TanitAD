"""Bounded log-variance for the H15 ImaginationField (models review #3, numerics).

Production & Optimization compliance review of `stack/tanitad/models/imagination.py`.
The imagination field's per-cell `logvar_head` is an **unbounded** `nn.Linear`, and
its output flows into two `exp()` sites with no clamp:

1. `imagination_nll` (`imagination.py:135`):
       nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)
   When the head goes over-confident (logvar strongly negative — common early in
   training and on OOD sectors), `exp(-logvar)` **overflows fp32 at logvar < -88.7**
   → `+inf` loss → NaN gradients. This is a live path: `imagination_nll` is called
   in `train_worldmodel.py:338` (the flagship arm), `train_flagship4b.py:164`, and
   `finetune_traj.py:217` — a single over-confident cell can NaN-kill a training run
   (the F-5/F-6/F-7 ops-fragility class this stream tracks).

2. The uncertainty EXPORT path `replay/arms.py:284`:
       (0.5 * logvar.float()).exp().mean(-1)
   feeds OKRI / LOPS / the H2 modality-steering trigger; an unbounded **positive**
   logvar overflows the std here → NaN metric / NaN trigger.

Fix (behaviour-preserving in the healthy range): bound the head output to
`[LOGVAR_MIN, LOGVAR_MAX]`. A converged, healthy head sits well inside these
bounds, so a trained checkpoint is numerically unchanged (proven by
`test_safe_matches_stack_in_normal_range`); only pathological values are trimmed.
`clamp` passes gradient through in-range and zeroes it at the bound — exactly the
desired "stop pushing logvar to ±inf" behaviour. This is the post-hoc production
guard; a tanh/softplus reparam would perturb every value and need retraining.

Bounds: exp(-LOGVAR_MIN) = exp(10) ≈ 2.2e4 (finite); exp(0.5*LOGVAR_MAX) = exp(5)
≈ 148 (finite std ceiling = "no idea"). Both far inside fp32 range.

Self-contained (torch only) so the tests run without the tanitad package.
"""

from __future__ import annotations

import torch
from torch import Tensor

# exp(-(-10)) = e^10 ≈ 2.2e4 and exp(0.5*10) = e^5 ≈ 148 — both finite in fp32,
# while a healthy trained logvar head sits well within [-10, 10].
LOGVAR_MIN: float = -10.0
LOGVAR_MAX: float = 10.0


def clamp_logvar(logvar: Tensor,
                 lo: float = LOGVAR_MIN, hi: float = LOGVAR_MAX) -> Tensor:
    """Bound a per-cell log-variance to a numerically safe range.

    No-op on values already inside [lo, hi]; gradient flows in-range and is
    zeroed at the bound (stops the head from diverging to ±inf). Apply at the
    `logvar_head` output so every consumer (nll, d9_rows, the replay/arms std
    export) receives a bounded logvar.
    """
    return logvar.clamp(lo, hi)


def imagination_nll_safe(pred: Tensor, target: Tensor, logvar: Tensor, vis: Tensor,
                         observed_weight: float = 0.1) -> Tensor:
    """Drop-in replacement for `tanitad.models.imagination.imagination_nll` that
    clamps `logvar` before `exp(-logvar)`.

    Identical to the stack function on any logvar already inside
    [LOGVAR_MIN, LOGVAR_MAX] (the healthy range) — it only prevents the fp32
    overflow when a cell becomes pathologically over-confident.
    """
    logvar = clamp_logvar(logvar)
    err2 = (pred - target).pow(2).mean(dim=-1)                 # [B, N]
    nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)
    w = (1.0 - vis) + observed_weight * vis
    return (w * nll).sum() / w.sum().clamp_min(1e-8)
