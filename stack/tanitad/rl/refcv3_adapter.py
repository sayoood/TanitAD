"""Connect the RL library to a real ``RefCV3Model`` — importing refc surfaces.

This is the only module that knows what refcv3 IS. Everything else in
``tanitad.rl`` is model-agnostic and talks through the ``sample_fn`` seam, which
is what lets the tests run a synthetic policy on CPU. ⛔ Nothing is copied from
``refc``/``refc_v3``: the model is imported and called.

WHAT IT SAMPLES, AND THE ONE THING IT IS NOT
---------------------------------------------
refcv3's decoder emits, per anchor, a deterministic ``offset``; the fan is
``anchor_traj = anchors + offset`` (``refc.py:1365``, ``:1515``). This adapter
treats the offset as the mean of an explicit **Gaussian exploration policy** and
draws ``G`` samples per anchor::

    offset_g = offset * (1 + sigma * eps),   eps ~ N(0, I)        [multiplicative]
    offset_g = offset + sigma * eps                                [additive]

The multiplicative form is the default because DDv2's ablation measured it above
additive (90.1 vs 89.7 PDMS) — it perturbs proportionally instead of swamping
small offsets. Because this is a reparameterisation with a known density, the
log-probability is analytic and DIFFERENTIABLE w.r.t. the decoder's output, so
the score-function gradient reaches the generator's parameters.

⛔ **STATED LIMIT, NOT HIDDEN.** This is a surrogate Gaussian policy over the
EMITTED offset. It is **not** the true reverse-diffusion density of the
truncated-diffusion decoder. DDv2's exact formulation attaches the policy
gradient to the denoising step distribution itself. The surrogate is:
  * correct as a policy gradient for the sampler *as defined here*, and
  * the cheapest way to get a working, testable loop against the real model,
but any claim of parity with DDv2's estimator is UNVERIFIED and must say so.
Replacing this with the decoder's own step density is a named backlog item.

⚠️ ``offset`` can be exactly 0 for an anchor the decoder does not move. Under
the multiplicative form the sample density then collapses (zero scale), so the
scale is floored at ``min_scale`` — otherwise ``logp`` is -inf and the loss is
NaN, which surfaces as a dead run rather than as the degenerate anchor it is.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import PostTrainConfig

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def sample_offsets(offset: Tensor, cfg: PostTrainConfig, *,
                   generator: torch.Generator | None = None,
                   min_scale: float = 1e-3) -> tuple[Tensor, Tensor]:
    """``offset [B, N, S, 2]`` -> (``offset_g [B, N, G, S, 2]``, ``logp [B, N, G]``).

    ``logp`` is the summed Gaussian log-density of the drawn sample over the
    (S, 2) axes, differentiable through ``offset``.
    """
    g = int(cfg.group_size)
    if g < 2:
        raise ValueError(f"group_size must be >= 2 to have any advantage, got {g}")
    b, n, s, two = offset.shape
    mean = offset.unsqueeze(2).expand(b, n, g, s, two)

    if cfg.noise_mode == "multiplicative":
        scale = (mean.abs() * cfg.noise_scale).clamp_min(min_scale)
    else:
        scale = torch.full_like(mean, max(cfg.noise_scale, min_scale))

    eps = torch.randn(mean.shape, generator=generator, device=mean.device,
                      dtype=mean.dtype)
    sample = mean + scale * eps

    # ⛔ THE SCORE-FUNCTION GRADIENT LIVES IN (sample.detach() - mean) — NOT eps.
    #
    # CORRECTED 2026-08-29 after pilot P-RC21's first P1 arm COLLAPSED THE
    # PLANNER (fan collision 11.1 % -> 0.0 % because R3 exploded 1.97 m ->
    # 347.2 m: every candidate left the road). Root cause, MEASURED by A/B
    # (`code/estimator_ab.py`): the original wrote
    #     logp = -0.5*eps**2 - log(scale)
    # with eps the RAW DRAW — a symbolic substitution that makes the
    # (sample - mean) dependence CANCEL. d(logp)/d(mean) then flows ONLY through
    # log(scale)=log|mean·sigma|, so the update can shrink or inflate |offset|
    # by advantage sign but can NEVER move the mean toward good samples. In the
    # toy A/B the degenerate form leaves the mean at its start (+0.500 after
    # 400 steps, target +3.0) while this form reaches +2.750.
    #
    # The correct REINFORCE logp treats the drawn sample as a CONSTANT and the
    # density parameters as live: eps_eff = (sample.detach() - mean)/scale.
    # ⚠️ The old version PASSED test_sample_offsets_shapes_and_differentiability
    # — a nonzero gradient is not a correct gradient. The pinning test is now
    # DIRECTIONAL (test_score_function_gradient_points_toward_good_samples).
    eps_eff = (sample.detach() - mean) / scale
    logp = (-0.5 * eps_eff.pow(2) - torch.log(scale) - LOG_SQRT_2PI)
    return sample, logp.sum(dim=(-1, -2))


def make_refcv3_sample_fn(model, cfg: PostTrainConfig, *,
                          build_ctx=None,
                          generator: torch.Generator | None = None):
    """Return a ``sample_fn(batch, cfg) -> (traj, logp, ctx)`` for a RefCV3Model.

    ``batch`` must be a mapping carrying at least ``frames``; optional
    ``nav_cmd`` / ``v0`` / ``lan`` are forwarded. ``build_ctx(batch, out)``
    supplies the REWARD CONTEXT — scene facts only.

    ⛔ The context is never taken from the model's own outputs. The reward may
    not read a model-produced ranking, and the cheapest way to guarantee that is
    to never hand it one: ``build_ctx`` receives the batch, and anything it adds
    is audited by ``assert_selector_disjoint`` on every step.
    """
    def sample_fn(batch, cfg_in: PostTrainConfig):
        frames = batch["frames"]
        out = model(frames, batch.get("nav_cmd"), batch.get("v0"),
                    steps=int(getattr(cfg_in, "decoder_steps", 0)),
                    lan=batch.get("lan"))
        anchor_traj = out["anchor_traj"]                   # [B, N, S, 2]
        offset = out["offset"]                             # [B, N, S, 2]
        base = anchor_traj - offset                        # the anchors alone

        off_g, logp = sample_offsets(offset, cfg_in, generator=generator)
        traj = base.unsqueeze(2) + off_g                   # [B, N, G, S, 2]

        ctx = dict(build_ctx(batch, out)) if build_ctx else {}
        ctx.setdefault("dt", cfg_in.dt)
        return traj, logp, ctx

    return sample_fn


def gt_context(batch, out=None) -> dict:
    """A minimal reward context from the BATCH only — the safe default.

    Passes through the scene facts the reward components understand and
    NOTHING the model produced. ``out`` is accepted and ignored on purpose, so
    that a caller who reaches for it has to write that code deliberately.
    """
    ctx: dict = {}
    for key in ("gt_traj", "obstacles", "lead_path", "lead_len_m",
                "target_time_gap_s", "a_max", "kappa_max", "progress_ref_m",
                "v0", "dt"):
        if key in batch:
            ctx[key] = batch[key]
    return ctx
