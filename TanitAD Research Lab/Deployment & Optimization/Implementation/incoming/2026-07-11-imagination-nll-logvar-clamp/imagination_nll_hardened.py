"""Fail-safe imagination NLL — Production & Optimization compliance review #3
(numerics hardening, backlog P1.7).

Self-contained drop-in for two functions in
``stack/tanitad/models/imagination.py`` (imports only torch + stdlib, so the
intake tests need no ``tanitad`` install):

  * ``imagination_nll``  — clamp ``logvar`` before ``exp(-logvar)``
  * ``d9_rows``          — accept a ``torch.Generator`` for the shuffled-cell
                           baseline (reproducible D9 gate evidence)

--------------------------------------------------------------------------- #
Defect 1 (silent NaN / F-ops-fragility) — the anchor.
--------------------------------------------------------------------------- #
Current stack (``imagination.py:135``, wired into the live trainer at
``train_worldmodel.py:338``):

    nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)     # exp(-logvar) UNCLAMPED

``logvar`` comes from an UNBOUNDED Linear head (``ImaginationField.logvar_head``,
imagination.py:110 — no output activation). ``torch.exp(-logvar)`` overflows to
``+inf`` once ``-logvar > ln(FLOAT_MAX)``:

    dtype   overflow boundary (measured == -ln(finfo.max))
    fp32    logvar < -88.72
    fp16    logvar < -11.09      <-- the deployment / autocast precision

Below the boundary the loss is non-finite, and there is NO nan/inf guard
between the loss and ``opt.step()`` (verified train_worldmodel.py:330-358):
one non-finite cell -> non-finite ``loss`` -> ``backward()`` NaNs every gradient
-> ``clip_grad_norm_`` cannot recover (NaN in -> NaN out) -> ``opt.step()`` NaNs
every parameter -> the atomic checkpoint save PERSISTS a corrupted resume point.

REACHABILITY IS MEASURED, not hypothetical (``../imagination_nll_overflow/``):
the heteroscedastic NLL's own per-cell optimum is ``logvar* = ln(err2)``; any
cell predicted better than ``err2 = 1.53e-5`` has ``logvar*`` already below the
fp16 boundary, so plain SGD toward the optimum crosses into ``+inf`` — reproduced
in **45 SGD steps** (fp16, err2=1e-7) and 356 steps (fp32, err2=1e-40). Even well
before the hard overflow the gradient ``0.5*(1 - exp(-logvar)*err2)`` explodes
(``2.4e8`` at logvar=-20, ``2.8e34`` at -80), which ``clip_grad_norm_`` then
collapses the whole optimisation step onto.

Fix: clamp ``logvar`` to ``[-logvar_clamp, +logvar_clamp]`` before the exp.
Default ``logvar_clamp = 8.0`` keeps ``exp(-logvar)`` finite in BOTH fp32 and
fp16 across ``err2`` up to ~20, and is IDENTITY inside the band (measured
in-band parity vs the original = 0.0 exactly) — so training is unchanged wherever
``logvar`` is well-conditioned; the clamp only caps the pathological tail.
8.0 also comfortably covers the realistic optimum range (``logvar* = ln(err2)``
for ``err2 >= 3.4e-4``). Symmetric so the degenerate "predict infinite
uncertainty to zero the error term" direction is bounded too.

Note (honest, P8): a hard clamp has zero gradient outside the band, so a runaway
``logvar`` can *park* just past the edge (loss stays finite and bounded). This is
the minimal, parity-preserving fix; a ``softplus`` precision reparameterisation
would keep gradient everywhere but changes the head's semantics and is a larger
change — deferred (noted in INTAKE.md) unless a run shows edge-parking hurts D9.

--------------------------------------------------------------------------- #
Defect 2 (determinism / seed discipline) — the D9 baseline.
--------------------------------------------------------------------------- #
``d9_rows`` (imagination.py:154) draws ``perm = torch.randperm(...)`` with no
generator, so the shuffled-cell chance-floor baseline (``shuffled_cosine``) is
non-reproducible run-to-run — a determinism gap on the PRODUCTION_READINESS
checklist for a value that feeds a GATE (D9). ``hidden_cosine`` and
``calibration_gap`` are unaffected (no shuffle). Fix: accept an optional
``generator`` and thread it into ``randperm``; default ``None`` preserves the
current behaviour exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def imagination_nll(pred: Tensor, target: Tensor, logvar: Tensor, vis: Tensor,
                    observed_weight: float = 0.1,
                    logvar_clamp: float = 8.0) -> Tensor:
    """Heteroscedastic Gaussian NLL, emphasis on hidden cells — fail-safe.

    Identical to ``stack/tanitad/models/imagination.py::imagination_nll`` except
    ``logvar`` is clamped to ``[-logvar_clamp, logvar_clamp]`` before the exp, so
    ``exp(-logvar)`` cannot overflow to a non-finite loss (which would silently
    NaN-corrupt the whole model via ``backward``/``opt.step``). The clamp is
    identity for well-conditioned ``logvar`` -> no behaviour change in-band.

    pred/target: [B, N, D]; logvar/vis: [B, N].
    """
    err2 = (pred - target).pow(2).mean(dim=-1)                 # [B, N]
    logvar = logvar.clamp(min=-logvar_clamp, max=logvar_clamp)  # <-- fail-safe
    nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)
    w = (1.0 - vis) + observed_weight * vis
    return (w * nll).sum() / w.sum().clamp_min(1e-8)


@torch.no_grad()
def d9_rows(pred: Tensor, target: Tensor, logvar: Tensor, vis: Tensor,
            generator: torch.Generator | None = None) -> dict:
    """Gate D9 evidence rows (with the shuffled-cell baseline as instrument).

    Identical to the stack ``d9_rows`` except the shuffle accepts an optional
    ``generator`` so the ``shuffled_cosine`` chance-floor is reproducible.
    ``generator=None`` reproduces the current (non-deterministic) behaviour.

    - hidden_cosine   : cosine(pred, target) on hidden cells
    - shuffled_cosine : same but targets shuffled across cells (chance floor)
    - calibration_gap : mean logvar(hidden) - mean logvar(visible)  (must be > 0)
    """
    hidden = vis < 0.5
    if hidden.sum() == 0:
        return {"hidden_cosine": float("nan"), "shuffled_cosine": float("nan"),
                "calibration_gap": float("nan")}
    p = F.normalize(pred[hidden], dim=-1)
    t = F.normalize(target[hidden], dim=-1)
    perm = torch.randperm(t.shape[0], generator=generator, device=t.device)
    return {
        "hidden_cosine": float((p * t).sum(-1).mean()),
        "shuffled_cosine": float((p * t[perm]).sum(-1).mean()),
        "calibration_gap": float(logvar[hidden].mean() - logvar[~hidden].mean()),
    }
