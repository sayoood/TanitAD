"""Encoder<->ego decorrelation (v2 lever B / H25) — self-contained, torch-only.

H25 (HYPOTHESIS_LEDGER 2026-07-18): the trained flagship encoder REDUNDANTLY
re-encodes the fed ego dynamics — yaw is LINEARLY decodable from the pooled
latent at R2 ~0.89 — spending capacity on a signal already available as the v0
action channel + the planner ego vector instead of on SCENE (imagination-panel
vision_use only ~12%). The ledger names encoder-ego decorrelation the
"cheapest+strongest" measure. ("Is Ego Status All You Need", Li et al. CVPR
2024, shows planning over-relies on ego velocity while image corruption barely
moves it; AdaptiveAD decouples scene perception from ego status; DINO-WM keeps
the encoder purely perceptual and puts dynamics in the transition.)

This module supplies two pieces (NO learnable parameters — both are functional):

  * :func:`ego_decorr_loss` — a LINEAR decorrelation penalty: the mean squared
    Pearson cross-correlation between the pooled latent ``z_t`` and the fed ego
    ``[v0, yr0]``. Differentiable in ``z`` (gradient flows to the encoder);
    ``ego`` is a data tensor (no grad). A no-op (0) when ego is None.

  * :func:`ego_linear_r2` — a DETACHED ridge linear-probe R2 predicting ego from
    ``z_t`` (dual/Gram form, robust when D >> batch). Lower = less re-encoding =
    the thing we want to drop. Never differentiable — a monitoring proxy only.

MECHANISM CHOICE (linear, not adversarial/HSIC) — justification:
  1. The measured pathology is LINEAR decodability (a linear in-latent probe,
     yaw R2 0.89) and the monitored proxy (:func:`ego_linear_r2`) is itself a
     LINEAR probe, so the penalty and the proxy share one functional form — no
     train/monitor mismatch, the penalty drives exactly the number we watch.
  2. Zero extra parameters, no adversarial min-max, no kernel bandwidth -> the
     STABLE pick for a 4-5 day unattended run. Gradient-reversal / DANN
     "critically depends on careful schedule and hyperparameter selection to
     avoid representational collapse" (Ganin & Lempitsky 2016); we cannot babysit
     an adversary for days. This is the cross-block analog of VICReg's covariance
     regularizer (Bardes, Ponce & LeCun 2022), which stably enforces
     decorrelation in large-scale SSL with a plain covariance penalty.
  3. O(B*D) cost (one cross-covariance matmul on a [B,2] ego and [B,~2048] z) —
     negligible next to the ViT.
  4. Correlation-normalized (per-dim std) -> bounded in [0,1], so a conservative
     weight cannot let it blow up and fight SIGReg / the shared trunk.

NONLINEAR UPGRADE PATH (documented, one-function swap in this module): if
``ego_r2`` falls but imagination-panel vision_use stalls (the encoder hid ego
NON-linearly), replace :func:`ego_decorr_loss` with a biased HSIC estimator
(RBF + median-heuristic bandwidth) or a distance-correlation (DisCo) penalty —
both still parameter-free, both O(B^2).
"""

from __future__ import annotations

import torch
from torch import Tensor


def ego_decorr_loss(z: Tensor, ego: Tensor | None, eps: float = 1e-5) -> Tensor:
    """Mean squared linear (Pearson) cross-correlation between the pooled latent
    ``z`` [B, D] and the fed ego ``ego`` [B, E].

    Returns a scalar in [0, 1]; 0 when ``ego`` is None or the batch is too small
    to estimate a correlation (B < 2). Differentiable in ``z`` (the gradient
    flows back to the encoder); ``ego`` carries no gradient (it is built from
    odometry poses). Computed in float32 for bf16-autocast stability.
    """
    if ego is None:
        return z.new_zeros(())
    b = z.shape[0]
    if b < 2:
        return z.new_zeros(())
    z = z.float()
    ego = ego.float()
    zc = z - z.mean(0, keepdim=True)
    ec = ego - ego.mean(0, keepdim=True)
    zs = zc / (zc.std(0, unbiased=False, keepdim=True) + eps)      # [B, D]
    es = ec / (ec.std(0, unbiased=False, keepdim=True) + eps)      # [B, E]
    corr = (zs.transpose(0, 1) @ es) / b                          # [D, E] Pearson r
    return corr.pow(2).mean()


@torch.no_grad()
def ego_linear_r2(z: Tensor, ego: Tensor | None, ridge: float = 1.0,
                  eps: float = 1e-5) -> Tensor:
    """DETACHED ridge linear-probe R2 predicting ego from ``z`` (the vision-
    reliance proxy). Dual/Gram form (linear kernel), O(B^2 * D + B^3), which is
    the numerically sound way to fit a linear probe when D >> B (a naive
    D-space normal-equation solve is rank-deficient for a micro-batch and would
    trivially interpolate to R2=1). ``ridge`` is a scale-adaptive Tikhonov term
    so the in-sample R2 stays a meaningful < 1 quantity whose TREND across
    training is the signal (lower = the encoder re-encodes ego less).

    Returns a detached scalar in [0, 1]; 0 when ``ego`` is None or B < 3. Never
    carries a gradient — it must not add a training signal (monitoring only).
    """
    if ego is None:
        return torch.zeros(())
    z = z.detach().float()
    ego = ego.detach().float()
    b = z.shape[0]
    if b < 3:
        return torch.zeros(())
    # Force fp32 with autocast OFF: under the trainer's bf16 autocast the Gram
    # matmul `k` would be re-cast to bf16 while the elementwise `es` stays fp32,
    # so torch.linalg.solve fails on the dtype mismatch (and bf16 solve is
    # ill-conditioned anyway). This is a DETACHED monitoring probe, so disabling
    # autocast is correct and adds no training gradient.
    with torch.autocast(device_type=("cuda" if z.is_cuda else "cpu"),
                        enabled=False):
        # Explicit .float() on every solve operand — autocast can still cast the
        # matmul output to bf16, so force fp32 at each step (bulletproof).
        zc = (z - z.mean(0, keepdim=True)).float()
        zs = (zc / (zc.std(0, unbiased=False, keepdim=True) + eps)).float()  # [B,D]
        es = (ego - ego.mean(0, keepdim=True)).float()           # [B, E] centered
        k = (zs @ zs.transpose(0, 1)).float() / zs.shape[1]      # [B, B] fp32 kernel
        lam = ridge * k.diagonal().mean().clamp_min(eps)         # scale-adaptive
        eye = torch.eye(b, device=z.device, dtype=torch.float32)
        alpha = torch.linalg.solve((k + lam * eye).float(), es)  # both fp32
        e_hat = (k @ alpha).float()                              # in-sample fit
        ss_res = (es - e_hat).pow(2).sum()
        ss_tot = es.pow(2).sum().clamp_min(eps)
        return (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
