"""grad_surgery.py — one-sided PCGrad at the planner<->WM seam (v4 coupling).

Design: `…/incoming/2026-07-23-planner-wm-gradient-coupling/DESIGN.md`.
Pre-registration: `…/PRE_REGISTRATION.md`.

WHAT THIS REPLACES
------------------
v4 couples the anchored-diffusion planner into the shared ViT trunk through ONE
seam: `states_p = grad_scale(states, lambda_plan)` (`flagship_v4.py:211`,
`metric_dynamics.grad_scale`). `grad_scale` multiplies the ENTIRE planner->trunk
gradient by a scalar in `[floor, 1]`. That scalar is the only knob we have, and it
has been mis-tuned three times (v4 hot / v4.1 starve / v4.2 degrade). A scalar
shrinks the planner gradient in EVERY direction at once — including the directions
that do NOT conflict with the world-model objective — so it trades planner signal
against WM integrity globally instead of resolving the actual gradient conflict.

THE MECHANISM (single shared activation => single-backward surgery)
-------------------------------------------------------------------
The planner reaches the trunk ONLY through `states` (imagination off in the
shipping config, `cond_imagination:false`), and the WM loss ALSO reads `states`.
So the entire WM<->planner conflict lives in two vectors of the SAME shape:
  g_wm   = dL_predict/d states   (how the WM wants the encoder output to move)
  g_plan = dL_plan  /d states    (how the planner wants it to move)
`SeamProject` forks `states` into a WM view and a planner view, and in a SINGLE
backward pass projects g_plan to remove only its component that OPPOSES g_wm,
leaving g_wm untouched (one-sided / asymmetric PCGrad, Yu et al. 2020,
arXiv:2001.06782). The encoder therefore receives, from the seam,
  g_wm + proj_deconflict(g_plan, g_wm)
so the planner may shape the trunk only in directions that do not hurt prediction.
Cost is a few elementwise ops on `states` ([B,W,2048]); NO extra encoder pass, so
grad-checkpointing and gradient accumulation are unaffected (§Cost in DESIGN.md).

This module is pure-torch (no tanitad imports) so the projection math is unit-
tested in isolation, exactly like the trainer's other correctness-critical pieces.
The integration point is `train_flagship_v4.v4_loss_step` (see DESIGN.md §4).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# The projection math — pure, testable, per-sample or global                   #
# --------------------------------------------------------------------------- #
def deconflict(g_plan: Tensor, g_ref: Tensor, *, per_sample: bool = True,
               target_cos: float = 0.0, eps: float = 1e-12
               ) -> Tuple[Tensor, dict]:
    """Remove the component of ``g_plan`` that conflicts with ``g_ref``.

    ONE-SIDED (asymmetric) PCGrad: ``g_ref`` (the WM-prediction gradient) is the
    protected reference and is NEVER modified; only ``g_plan`` (the planner
    gradient) is projected.

    * ``target_cos == 0.0`` -> pure PCGrad: if ``<g_plan, g_ref> < 0`` (the tasks
      conflict) remove the ``g_ref`` component so the result is orthogonal to
      ``g_ref``; otherwise pass ``g_plan`` through unchanged. When the tasks agree
      this is a strict no-op, so the surgery only ever *subtracts* conflict.
    * ``target_cos > 0.0`` -> GradVaccine-style (Wang et al. 2021,
      arXiv:2010.05874): raise the cosine to a small positive floor ``phi`` instead
      of only to 0. A knob, off by default.

    ``per_sample`` projects each row of the batch against its own reference
    (finer, matches per-example PCGrad); ``False`` projects the batch-flattened
    gradient once (global). Returns ``(g_plan_projected, diagnostics)`` where the
    diagnostics carry the PRE-projection cosine and the fraction of the planner
    gradient's norm that was removed — the instrument the pre-registration reads to
    tell a *directional* conflict (surgery helps) from a magnitude/one-task effect
    (surgery cannot help).
    """
    if g_plan.shape != g_ref.shape:
        raise ValueError(f"shape mismatch: g_plan {tuple(g_plan.shape)} "
                         f"g_ref {tuple(g_ref.shape)}")
    if per_sample:
        b = g_plan.shape[0]
        gp = g_plan.reshape(b, -1).float()
        gr = g_ref.reshape(b, -1).float()
        dim = 1
        keep = (b, *([1] * (g_plan.dim() - 1)))
    else:
        gp = g_plan.reshape(1, -1).float()
        gr = g_ref.reshape(1, -1).float()
        dim = 1
        keep = (1, *([1] * (g_plan.dim() - 1)))

    dot = (gp * gr).sum(dim=dim, keepdim=True)                     # <g_plan, g_ref>
    rr = (gr * gr).sum(dim=dim, keepdim=True)                      # ||g_ref||^2
    pp = (gp * gp).sum(dim=dim, keepdim=True)                      # ||g_plan||^2
    denom = (rr.sqrt() * pp.sqrt()).clamp_min(eps)
    cos = (dot / denom)                                            # pre-projection

    if target_cos <= 0.0:
        # pure PCGrad: only when conflicting (cos < 0), remove the g_ref component.
        conflict = (dot < 0).to(gp.dtype)
        coeff = conflict * (dot / rr.clamp_min(eps))
    else:
        # GradVaccine: whenever cos < phi, add just enough g_ref to reach phi.
        phi = float(target_cos)
        below = (cos < phi).to(gp.dtype)
        # solve for a s.t. cos(g_plan + a*g_ref_hat_scaled, g_ref) = phi (closed form
        # from Wang et al. eq.); implemented as the standard magnitude-preserving mix.
        gp_norm = pp.sqrt().clamp_min(eps)
        gr_norm = rr.sqrt().clamp_min(eps)
        num = gp_norm * (phi * (1.0 - cos.clamp(-1, 1) ** 2).clamp_min(0).sqrt()
                         - cos.clamp(-1, 1) * (1.0 - phi ** 2) ** 0.5)
        coeff = -below * (num / ((1.0 - phi ** 2) ** 0.5 * gr_norm))
        # `coeff` multiplies g_ref below; sign chosen so cos rises toward phi.

    gp_proj = gp - coeff * gr
    removed_norm = ((gp - gp_proj) ** 2).sum(dim=dim).sqrt()
    frac_removed = (removed_norm / pp.sqrt().squeeze(dim).clamp_min(eps))

    diag = {
        "seam_cos_mean": float(cos.mean()),
        "seam_cos_min": float(cos.min()),
        "seam_frac_conflict": float((dot < 0).float().mean()),
        "seam_frac_removed_mean": float(frac_removed.mean()),
        "per_sample": per_sample,
    }
    return gp_proj.reshape_as(g_plan).to(g_plan.dtype), diag


# --------------------------------------------------------------------------- #
# The seam — a forward-identity fork whose backward de-conflicts, one pass     #
# --------------------------------------------------------------------------- #
class _SeamProject(torch.autograd.Function):
    """Fork ``x`` into a (WM view, planner view). Forward is identity on both;
    backward receives ``(g_wm, g_plan)`` and returns ``g_wm + deconflict(g_plan,
    g_wm)`` as the single gradient w.r.t. ``x`` — so the trunk sees the WM gradient
    at full strength plus only the non-conflicting part of the planner gradient, in
    ONE backward pass. The most recent backward's diagnostics are stashed on the
    class for the trainer to log (`_SeamProject.last_diag`)."""

    last_diag: dict = {}

    @staticmethod
    def forward(ctx, x: Tensor, per_sample: bool, target_cos: float):
        ctx.per_sample = per_sample
        ctx.target_cos = target_cos
        # two distinct tensors sharing x's value, so autograd tracks two grads.
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, g_wm: Tensor, g_plan: Tensor):
        g_plan_proj, diag = deconflict(g_plan, g_wm, per_sample=ctx.per_sample,
                                       target_cos=ctx.target_cos)
        _SeamProject.last_diag = diag
        return g_wm + g_plan_proj, None, None


def seam_project(states: Tensor, *, per_sample: bool = True,
                 target_cos: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Fork the shared trunk output for the WM stack and the planner stack.

    Returns ``(states_wm, states_plan)`` — feed ``states_wm`` to ``flagship_loss``
    and ``states_plan`` to the planner head + strategic head. Backprop of the summed
    loss then de-conflicts the planner's trunk gradient against the WM's in a single
    pass. Forward-identical to ``states`` on both branches, so with ``target_cos=0``
    and no gradient conflict it is a strict no-op vs today's summed backward — the
    safety contract for an unattended multi-day run (cf. ``grad_scale`` alpha==1)."""
    return _SeamProject.apply(states, per_sample, target_cos)


# --------------------------------------------------------------------------- #
# CPU smoke — proves identity fwd, one-sidedness, no-op-when-aligned, finite   #
# --------------------------------------------------------------------------- #
def smoke() -> dict:
    torch.manual_seed(0)
    b, w, d = 4, 8, 2048

    # A toy shared "trunk": one linear layer feeding both a WM loss and a planner
    # loss, so we can read the REAL trunk gradient under the seam.
    trunk = torch.nn.Linear(d, d, bias=False)
    x = torch.randn(b, w, d)

    def trunk_grad(target_cos, use_seam):
        trunk.zero_grad(set_to_none=True)
        states = trunk(x)
        if use_seam:
            s_wm, s_plan = seam_project(states, target_cos=target_cos)
        else:
            s_wm = s_plan = states
        # WM loss pulls states toward +1; planner loss pulls toward a conflicting
        # random direction so the two gradients genuinely oppose on many samples.
        l_wm = ((s_wm - 1.0) ** 2).mean()
        gdir = torch.randn(b, w, d)
        l_plan = (s_plan * gdir).mean()
        (l_wm + l_plan).backward()
        return trunk.weight.grad.clone()

    # 1) forward identity: both seam branches equal states exactly
    states = trunk(x)
    s_wm, s_plan = seam_project(states)
    fwd_identical = bool(torch.equal(s_wm, states) and torch.equal(s_plan, states))

    # 2) one-sidedness: the WM-only gradient is byte-identical with/without the seam
    #    (the seam never touches g_wm).
    trunk.zero_grad(set_to_none=True); wm_only_seam = trunk(x)
    sw, sp = seam_project(wm_only_seam)
    ((sw - 1.0) ** 2).mean().backward(); g_wm_seam = trunk.weight.grad.clone()
    trunk.zero_grad(set_to_none=True); wm_only_plain = trunk(x)
    ((wm_only_plain - 1.0) ** 2).mean().backward(); g_wm_plain = trunk.weight.grad.clone()
    wm_grad_untouched = float((g_wm_seam - g_wm_plain).abs().max())

    # 3) no-op when the planner AGREES with the WM (cos>=0 everywhere): seam == plain
    trunk.zero_grad(set_to_none=True); a = trunk(x); sa, sb = seam_project(a)
    (((sa - 1.0) ** 2).mean() + ((sb - 1.0) ** 2).mean() * 0.3).backward()
    g_agree_seam = trunk.weight.grad.clone()
    trunk.zero_grad(set_to_none=True); a2 = trunk(x)
    (((a2 - 1.0) ** 2).mean() + ((a2 - 1.0) ** 2).mean() * 0.3).backward()
    g_agree_plain = trunk.weight.grad.clone()
    noop_when_aligned = float((g_agree_seam - g_agree_plain).abs().max())

    # 4) under real conflict the seam changes the trunk gradient and stays finite
    g_plain = trunk_grad(0.0, use_seam=False)
    g_seam = trunk_grad(0.0, use_seam=True)
    changed = float((g_seam - g_plain).norm())
    finite = bool(torch.isfinite(g_seam).all())
    diag = _SeamProject.last_diag

    out = {
        "fwd_identical": fwd_identical,
        "wm_grad_untouched_maxabs": wm_grad_untouched,     # ~0 => one-sided proven
        "noop_when_aligned_maxabs": noop_when_aligned,     # ~0 => no-op when agree
        "trunk_grad_changed_under_conflict": changed,      # >0 => surgery acts
        "trunk_grad_finite": finite,
        "seam_diag_under_conflict": diag,
    }
    return out


if __name__ == "__main__":
    import json
    r = smoke()
    print(json.dumps(r, indent=2))
    assert r["fwd_identical"], r
    assert r["wm_grad_untouched_maxabs"] < 1e-6, r      # WM gradient never modified
    assert r["noop_when_aligned_maxabs"] < 1e-6, r      # strict no-op when aligned
    assert r["trunk_grad_changed_under_conflict"] > 0.0, r
    assert r["trunk_grad_finite"], r
    print("[grad_surgery] smoke OK — one-sided, no-op-when-aligned, finite.")
