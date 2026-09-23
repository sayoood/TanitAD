"""refcv6 §3 — F1…F9, the nine ways our diffusion is not DiffusionDrive's.

⭐ **Reference is the RELEASED CODE, not the paper's prose**, and where the two
disagree the code wins. Every port below names its source line in
``TanitAD Research Lab/Architecture & Inference/Research/
2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/v1/transfuser_model_v2.py``
(V1, the NAVSIM model) or in the V2 config beside it.

============================================================================
THE NINE, AND WHAT EACH ONE IS
============================================================================

============  =========================================================
flag          what it changes  (DD source)
============  =========================================================
``F1``        **random-t training.** DD draws ``timesteps = randint(0, 50)``
              ONCE per sample and makes ONE decoder call
              (``transfuser_model_v2.py:463-476``); every cascade layer is
              then supervised (``:494-497``). Ours backprops through the
              **2-step inference chain** and supervises only the final
              output, so the network never sees a timestep outside
              ``{10, 0}``.
``F2``        **step semantics.** DD calls
              ``diffusion_scheduler.set_timesteps(1000, device)``
              (``:507``) and then steps at each ladder label, so diffusers
              computes ``prev_timestep = t - 1000 // 1000 = t - 1``. The
              10 -> 9 step keeps ~95 % of the residual, i.e. pass 2
              re-denoises nearly the same input under a different time
              label. Ours steps 10 -> 0 and keeps ~28 %.
``F3``        **cascade.** DD clones its decoder layer (``:345-347``), each
              clone emits its OWN ``(poses_reg, poses_cls)``, the loss is
              summed over layers (``:494-497``), and the trajectory handed
              to the next layer is **detached** (``:379``).
``F4``        **per-layer AdaLN.** ``ModulationLayer`` (``:229-268``) is
              applied inside EVERY layer (``:337``):
              ``x * (1 + scale) + shift`` from a Mish->Linear on the time
              embedding. Ours adds the timestep ONCE, to the query
              (``refc.py::_decode_ctrl``).
``F5``        **scoring + focal.** DD's emitted trajectory and its score
              come from the SAME pass (``:544-552``), and the
              classification loss is ``py_sigmoid_focal_loss`` with
              ``gamma=2.0, alpha=0.25`` (``multimodal_loss.py:146-157``),
              not a softmax CE.
``F6``        **one reconstruction loss.** DD's matched-anchor L1 IS its x0
              loss (``multimodal_loss.py:161``). Our ``--w-u0`` duplicates
              it in control space. F6 is ``--w-u0 0``.
``F7``        **several noise samples per anchor.** DD Tab. 6: N 20 -> 40
              is +0.1 PDMS. Ours refuses ``sampler_groups > 1`` because
              three call sites would be mis-indexed
              (``refc.py::forward``). See :func:`tile_anchor_prior` and
              :func:`candidate_to_anchor_id` — two of the three are widened
              here; the third is named and still refused.
``F8``        **flat waypoint-space noise.** DD noises WAYPOINTS under
              ``norm_odo`` (``:432-441``), giving a per-waypoint sigma that
              is the SAME at 0.5 s and at 4 s. Ours noises CONTROLS, so the
              positional noise grows with the horizon.
``F9``        **the vocabulary stays ours.** v0-conditioned, 117 anchors.
              Assert-only — :func:`assert_f9_vocabulary`.
============  =========================================================

⛔ **EVERY FLAG DEFAULTS OFF AND OFF MEANS BIT-IDENTICAL.** Nothing in this
module is constructed, called or imported on a default build; the flags live
on :class:`DiffusionFlags` whose ``any_on`` is False by default, and
``tests/test_refcv6_diffusion.py::test_all_flags_off_is_bit_identical`` pins a
64-window forward against the pre-refcv6 file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "DiffusionFlags", "focal_cls_loss", "AdaLNModulation", "CascadeHeads",
    "dd_step_pairs", "dd_norm_waypoints", "dd_denorm_waypoints",
    "DD_X_SPAN", "DD_Y_SPAN", "DD_X_OFF", "DD_Y_OFF",
    "tile_anchor_prior", "candidate_to_anchor_id", "F7_UNWIDENED_SITES",
    "assert_f9_vocabulary", "flag_stamp",
]


# ============================================================================
# The flag block
# ============================================================================

@dataclass
class DiffusionFlags:
    """F1…F9. **All nine default to the current behaviour.**

    ⛔ These are DECLARED FIELDS on a dataclass and not ad-hoc attributes,
    because the refcv5 ``sampler`` defect (a trainer writing
    ``core.decoder.sampler = "ddim"`` onto a class with no such field, which
    Python accepted, the guards read back, and ``config.json`` recorded — for
    a model containing no denoiser) is exactly the failure this whole seam is
    instrumented against.
    """

    # -- F1 -- one decoder call at t ~ U[0, t_max), every layer supervised --
    f1_random_t: bool = False
    f1_t_max: int = 50                 # DD's `randint(0, 50)` (:465-468)
    # -- F2 -- DD's `t -> t-1` step semantics -------------------------------
    f2_dd_step: bool = False
    # -- F3 -- per-layer offset heads + per-layer loss + detach -------------
    f3_per_layer: bool = False
    # -- F4 -- per-layer AdaLN timestep modulation --------------------------
    f4_adaln: bool = False
    #: DD's own ``if_zeroinit_scale`` is **False** (``:233``). We keep the code's
    #: value as the default and expose the knob, because a zero-init AdaLN is
    #: an identity at step 0 and therefore a *removable* graft — a different,
    #: also-legitimate arm. The code wins on the default; say so in the stamp.
    f4_zero_init: bool = False
    # -- F5 -- emitting pass's own confidence + focal loss -------------------
    f5_emitting_conf: bool = False
    #: ⛔ The GUARD, not the feature. Default OFF, so every banked arm is
    #: untouched. When True a sampler build REFUSES unless the ranked surface is
    #: the emitted fan's own score (``sel.refined`` or :attr:`f5_emitting_conf`).
    #: MEASURED 2026-09-22 on the default config: perturbing ONLY the sampler
    #: moved ``traj`` 12.20 m and ``sel_score`` by **exactly 0.0** — the ranked
    #: score is blind to the trajectory the model emits, which is REFe's class-C
    #: defect in a different architecture. Enforced in ``refc.py``'s WP-4
    #: preflight so it refuses BEFORE the compute.
    f5_refuse_blind_rank: bool = False
    f5_focal: bool = False
    f5_focal_gamma: float = 2.0        # `multimodal_loss.py:151`
    f5_focal_alpha: float = 0.25       # `multimodal_loss.py:152`
    # -- F6 -- DD has ONE reconstruction loss --------------------------------
    f6_w_u0_zero: bool = False
    # -- F7 -- several noise samples per anchor ------------------------------
    f7_samples_per_anchor: int = 1
    #: The operator's acknowledgement that the THIRD un-widened site (the eval
    #: harness's ``sel_idx`` join) has been handled by reading ``sel_anchor_id``
    #: instead. Without it, ``> 1`` still refuses — see :data:`F7_UNWIDENED_SITES`.
    f7_ack_eval_join: bool = False
    # -- F8 -- DD-style flat waypoint-space noise ----------------------------
    f8_flat_waypoint_noise: bool = False
    # -- F9 -- assert the v0-conditioned 117-anchor vocabulary ---------------
    f9_assert_vocab: bool = False
    f9_n_anchors: int = 117

    @property
    def any_on(self) -> bool:
        """True iff this build differs from pre-refcv6 behaviour in ANY way."""
        return bool(
            self.f1_random_t or self.f2_dd_step or self.f3_per_layer
            or self.f4_adaln or self.f5_emitting_conf or self.f5_focal
            or self.f5_refuse_blind_rank
            or self.f6_w_u0_zero or int(self.f7_samples_per_anchor) > 1
            or self.f8_flat_waypoint_noise or self.f9_assert_vocab)

    @property
    def builds_modules(self) -> bool:
        """True iff the flags add PARAMETERS.

        ⛔ Only F3 and F4 do. Everything else is a different use of the same
        weights, which is why every other flag is checkpoint-compatible and
        these two are not.
        """
        return bool(self.f3_per_layer or self.f4_adaln)

    def __post_init__(self) -> None:
        if int(self.f1_t_max) < 1:
            raise ValueError(
                f"f1_t_max must be >= 1, got {self.f1_t_max}; 1 is the "
                f"pre-registered zero-noise regression arm (PREREG P13-R), "
                f"0 would make the draw empty.")
        if int(self.f7_samples_per_anchor) < 1:
            raise ValueError(
                f"f7_samples_per_anchor must be >= 1, got "
                f"{self.f7_samples_per_anchor}")


def flag_stamp(flags: DiffusionFlags) -> dict:
    """What ``config.json`` must carry, so a reader can tell which arm ran."""
    return {
        "f1_random_t": bool(flags.f1_random_t),
        "f1_t_max": int(flags.f1_t_max),
        "f2_dd_step": bool(flags.f2_dd_step),
        "f3_per_layer": bool(flags.f3_per_layer),
        "f4_adaln": bool(flags.f4_adaln),
        "f4_zero_init": bool(flags.f4_zero_init),
        "f5_emitting_conf": bool(flags.f5_emitting_conf),
        "f5_refuse_blind_rank": bool(flags.f5_refuse_blind_rank),
        "f5_focal": bool(flags.f5_focal),
        "f6_w_u0_zero": bool(flags.f6_w_u0_zero),
        "f7_samples_per_anchor": int(flags.f7_samples_per_anchor),
        "f7_ack_eval_join": bool(flags.f7_ack_eval_join),
        "f8_flat_waypoint_noise": bool(flags.f8_flat_waypoint_noise),
        #: ⭐ Provenance for the 2026-09-23 fix: DD re-clamps the normalised
        #: sample at the top of EVERY ladder iteration
        #: (`transfuser_model_v2.py:519`); until then we clamped once, outside
        #: the loop, and `|x_n|` was MEASURED at 15.04 after one `sched.step`.
        #: A literal `True`, not a knob: there is no F8 arm to keep compatible
        #: (F8 has never been run) and two clamp semantics under one flag name
        #: is how an arm becomes unattributable. It is stamped so any future
        #: artifact says which semantics produced it.
        "f8_clamp_in_ladder": True,
        "f9_assert_vocab": bool(flags.f9_assert_vocab),
        "any_on": bool(flags.any_on),
    }


# ============================================================================
# F5 — DD's focal classification loss
# ============================================================================

def focal_cls_loss(logits: Tensor, target_idx: Tensor,
                   gamma: float = 2.0, alpha: float = 0.25,
                   reduction: str = "mean") -> Tensor:
    """DD's ``py_sigmoid_focal_loss`` on a one-hot target. -> scalar (or [B,N]).

    Ported line for line from ``multimodal_loss.py:90-98`` and its caller
    ``:140-157``:

    * ``pred_sigmoid = pred.sigmoid()``
    * ``pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)``
    * ``focal_weight = (alpha*target + (1-alpha)*(1-target)) * pt**gamma``
    * ``loss = BCE_with_logits(pred, target, reduction='none') * focal_weight``
    * reduced with ``reduction='mean'`` and ``avg_factor=None``.

    ⛔ **``reduction='mean'`` means the mean over ``B x N`` elements, not over
    ``B``.** DD's ``weight_reduce_loss`` calls ``loss.mean()`` on the full
    ``[bs, num_mode]`` tensor (``multimodal_loss.py:112-114`` -> ``:24-25``),
    so the per-window magnitude scales as ``1 / N``. With our N = 117 against
    DD's 20 that is a 5.85x smaller term at the same weight — a real
    re-weighting, and the reason this is stated rather than assumed.

    ⛔ **This is a SIGMOID loss, not a softmax one.** Every anchor gets its own
    independent Bernoulli decision. Swapping in ``cross_entropy`` reproduces
    neither the gradient nor the scale, which is what makes F5 a separable arm
    from our ``loss_cls``.
    """
    if logits.ndim != 2:
        raise ValueError(f"focal_cls_loss expects [B, N] logits, got "
                         f"{tuple(logits.shape)}")
    if target_idx.ndim != 1 or target_idx.shape[0] != logits.shape[0]:
        raise ValueError(
            f"focal_cls_loss expects one target index per row: "
            f"{tuple(target_idx.shape)} against {tuple(logits.shape)}")
    target = torch.zeros_like(logits)
    target.scatter_(1, target_idx.reshape(-1, 1).long(), 1.0)
    pred_sigmoid = logits.sigmoid()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none") * focal_weight
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"reduction {reduction!r} not in ('none','mean','sum')")


# ============================================================================
# F4 — DD's AdaLN timestep modulation
# ============================================================================

class AdaLNModulation(nn.Module):
    """``ModulationLayer`` from ``transfuser_model_v2.py:229-268``.

    ``scale, shift = Linear(Mish(cond)).chunk(2)`` then
    ``x * (1 + scale) + shift``.

    ⛔ The ``Mish`` comes FIRST and there is no norm — DD's ``scale_shift_mlp``
    is ``Sequential(nn.Mish(), nn.Linear(condition_dims, embed_dims*2))``
    (``:235-238``). Writing it the usual way round (Linear then activation) is
    a different function of the timestep and would make the arm unattributable
    to the paper.

    ``zero_init`` mirrors DD's ``if_zeroinit_scale``, which in the released code
    is **False** (``:233``). With it True the layer is the identity at step 0,
    which makes F4 a removable graft; DD's own choice is not.
    """

    def __init__(self, d: int, cond_dim: int, zero_init: bool = False):
        super().__init__()
        self.scale_shift_mlp = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, d * 2))
        if zero_init:
            nn.init.zeros_(self.scale_shift_mlp[-1].weight)
            nn.init.zeros_(self.scale_shift_mlp[-1].bias)
        self.zero_init = bool(zero_init)

    def forward(self, x: Tensor, time_embed: Tensor) -> Tensor:
        """``x`` [B, N, d] · ``time_embed`` [B, d] or [B, 1, d] -> [B, N, d]."""
        te = time_embed if time_embed.ndim == 3 else time_embed.unsqueeze(1)
        scale, shift = self.scale_shift_mlp(te).chunk(2, dim=-1)
        return x * (1 + scale) + shift


# ============================================================================
# F3 — the cascade: per-layer offset + confidence heads
# ============================================================================

class CascadeHeads(nn.Module):
    """One ``(control_head, conf_head)`` pair PER decoder layer.

    DD gets this by ``copy.deepcopy``-ing a whole decoder layer that already
    contains its own ``task_decoder`` (``:345-347`` + ``:308-312``). Our layers
    are shared cross-attention blocks with a single head at the end, so the
    faithful graft is to give each layer its own heads here and leave the
    attention stack alone — the cascade is about *per-stage supervision with a
    detached hand-off*, not about duplicating attention parameters.

    ⛔ Every ``control_head`` is ZERO-INIT, exactly like the single head it
    generalises (``refc.py:1705-1706``), so a freshly-built cascade predicts
    "the current state" at every stage and every later movement is learned.
    """

    def __init__(self, n_layers: int, d: int, n_steps: int):
        super().__init__()
        if int(n_layers) < 1:
            raise ValueError(f"cascade needs >= 1 layer, got {n_layers}")
        self.n_layers = int(n_layers)
        self.n_steps = int(n_steps)
        self.control_heads = nn.ModuleList()
        self.conf_heads = nn.ModuleList()
        for _ in range(self.n_layers):
            ch = nn.Linear(d, self.n_steps * 2)
            nn.init.zeros_(ch.weight)
            nn.init.zeros_(ch.bias)
            self.control_heads.append(ch)
            self.conf_heads.append(nn.Linear(d, 1))

    def emit(self, i: int, q: Tensor) -> tuple[Tensor, Tensor]:
        """-> ``(conf [B, N], du [B, N, S, 2])`` from layer ``i``'s own heads."""
        b, n = q.shape[:2]
        conf = self.conf_heads[i](q).squeeze(-1)
        du = self.control_heads[i](q).reshape(b, n, self.n_steps, 2)
        return conf, du


# ============================================================================
# F2 — the step ladder
# ============================================================================

def dd_step_pairs(ladder: list[int], dd_step: bool) -> list[tuple[int, int]]:
    """-> ``[(t, t_prev), …]`` for one inference ladder.

    ``dd_step=False`` (TODAY): ``t_prev`` is the NEXT LADDER ENTRY, so the
    published ``[10, 0]`` ladder steps 10 -> 0 in one move and lands ~72 % of
    the way to the prediction. Pass 2 then sees an almost fully denoised input.

    ``dd_step=True`` (DD): ``t_prev = t - 1``. DD calls
    ``set_timesteps(1000, device)`` (``transfuser_model_v2.py:507``) while
    stepping at the labels ``[10, 0]``, and diffusers derives
    ``prev_timestep = timestep - num_train_timesteps // num_inference_steps
    = t - 1``. So the 10 -> 9 step keeps ~95 % of the residual and pass 2
    re-denoises nearly the same input under a different time label. THAT is
    what the cascade is for.

    ⚠️ The TERMINAL pair is ``(0, -1)`` in diffusers, where ``alpha_prod_prev``
    becomes ``final_alpha_cumprod`` and the sample collapses onto ``x0_hat``.
    Our schedule clamps ``t_prev`` into the table, making the terminal step an
    identity. **It changes nothing**: both DD (``:555-557``, which reads
    ``poses_reg`` of the last pass) and we (``u0_hat`` of the last pass) emit
    the last pass's PREDICTION, never the stepped sample. Stated because a
    reader will otherwise think the clamp is the bug.
    """
    out: list[tuple[int, int]] = []
    for i, t in enumerate(ladder):
        if dd_step:
            out.append((int(t), int(t) - 1))
        else:
            out.append((int(t), int(ladder[i + 1]) if i + 1 < len(ladder) else 0))
    return out


# ============================================================================
# F8 — DD's flat waypoint-space normalisation
# ============================================================================

# ``norm_odo`` / ``denorm_odo``, ``transfuser_model_v2.py:432-441 / 443-453``.
# Only the (x, y) pair is ported; DD's third channel is heading, which our fan
# does not carry.
DD_X_OFF, DD_X_SPAN = 1.2, 56.9
DD_Y_OFF, DD_Y_SPAN = 20.0, 46.0


def dd_norm_waypoints(xy: Tensor) -> Tensor:
    """metres -> DD's ``[-1, 1]`` box. ``xy[..., 0]`` = x, ``xy[..., 1]`` = y.

    ⭐ **This, and not our control normaliser, is what makes DD's noise FLAT.**
    At the truncation point ``sqrt(1 - abar_8) = 0.0316``; de-normalised that is
    ``0.0316 * 56.9 / 2 = 0.899 m`` in x and ``0.0316 * 46 / 2 = 0.727 m`` in
    y — **per waypoint, identically at 0.5 s and at 4 s**. Our control-space
    noise integrates, so its positional sigma grows from ~0.016 m at 0.5 s to
    ~0.86 m at 6 s. F8 is the arm that separates the two.
    """
    x = 2 * (xy[..., 0:1] + DD_X_OFF) / DD_X_SPAN - 1
    y = 2 * (xy[..., 1:2] + DD_Y_OFF) / DD_Y_SPAN - 1
    return torch.cat([x, y], dim=-1)


def dd_denorm_waypoints(xy: Tensor) -> Tensor:
    """DD's ``[-1, 1]`` box -> metres. Inverse of :func:`dd_norm_waypoints`."""
    x = (xy[..., 0:1] + 1) / 2 * DD_X_SPAN - DD_X_OFF
    y = (xy[..., 1:2] + 1) / 2 * DD_Y_SPAN - DD_Y_OFF
    return torch.cat([x, y], dim=-1)


# ============================================================================
# F7 — widening the fan from N to G*N
# ============================================================================

#: The sites ``refc.py::forward``'s refusal named, and their status here.
F7_UNWIDENED_SITES: dict[str, str] = {
    "loss_cls_a_star":
        "WIDENED. `candidate_to_anchor_id` maps a [B, G*N] candidate index "
        "back to its anchor id, so the matched-anchor target and the "
        "anchor-accuracy read stay defined.",
    "anchor_priors":
        "WIDENED. `tile_anchor_prior` repeats every [B, N] prior across the G "
        "groups in the SAME layout `roll_bank` tiles the bank in, so a prior "
        "and its candidate never come apart.",
    "eval_sel_idx_join":
        "⛔ STILL REFUSED. `taniteval`'s window dumps join on `sel_idx` as an "
        "ANCHOR id, and that file is not ours to change. With G > 1 `sel_idx` "
        "indexes a candidate, not an anchor. The forward therefore ALSO emits "
        "`sel_anchor_id`, and `f7_ack_eval_join` is the operator's statement "
        "that the consumer reads it. Without the acknowledgement the refusal "
        "stands — a silently mis-joined eval is worse than a crash.",
}


def tile_anchor_prior(prior: Tensor, groups: int) -> Tensor:
    """``[B, N]`` -> ``[B, G*N]``, tiled to match a group-major fan.

    ⛔ **The layout is ``repeat``, not ``repeat_interleave``, and it is not a
    taste question.** The widened fan is built by repeating the WHOLE bank G
    times (group-major: ``[a0..aN-1, a0..aN-1, …]``), so candidate ``g*N + a``
    is anchor ``a`` under noise draw ``g``. ``repeat_interleave`` would produce
    ``[a0,a0,…,a1,a1,…]`` and every prior would be attached to the wrong
    candidate while every shape still checked out.
    """
    if prior.ndim != 2:
        raise ValueError(f"tile_anchor_prior expects [B, N], got "
                         f"{tuple(prior.shape)}")
    g = int(groups)
    if g < 1:
        raise ValueError(f"groups must be >= 1, got {groups}")
    return prior.repeat(1, g) if g > 1 else prior


def candidate_to_anchor_id(idx: Tensor, n_anchors: int, groups: int) -> Tensor:
    """Candidate index in ``[0, G*N)`` -> anchor id in ``[0, N)``.

    The inverse of :func:`tile_anchor_prior`'s layout: ``idx % N``.
    """
    n = int(n_anchors)
    if n < 1:
        raise ValueError(f"n_anchors must be >= 1, got {n_anchors}")
    if int(groups) < 1:
        raise ValueError(f"groups must be >= 1, got {groups}")
    return idx % n


# ============================================================================
# F9 — the vocabulary is unchanged, and the build says so
# ============================================================================

def assert_f9_vocabulary(n_anchors: int, v0_conditioned: bool,
                         expect_n: int = 117,
                         anchor_controls: Tensor | None = None) -> None:
    """F9 is a NO-CHANGE item, so it is enforced as an assertion.

    ⭐ Spec §3: *"keep the v0-conditioned 117-anchor vocabulary (no change;
    assert it)"*. An arm that quietly ran on 20 DD-style k-means anchors, or on
    a fixed-path bank, would be a different experiment wearing refcv6's name —
    and MEASURED, the v0 conditioning is what makes the anchored Gaussian mean
    anything (``refc.py``'s WP-4 preflight refuses a fixed bank for the same
    reason).

    ⛔⛔ ``anchor_controls`` IS THE HALF THIS FUNCTION DID NOT HAVE. Until
    2026-09-23 the signature was ``(n_anchors, v0_conditioned, expect_n)`` —
    **three declarations and no tensor** — so the third message below was
    unenforceable by construction. MEASURED 2026-09-22
    (`…/2026-09-22-refcv6-review` §6.1): with ``v0_conditioned=True`` and the
    ``anchor_controls`` buffer left at its registered zeros, this function
    PASSED, the decoder forward did not raise, and the rolled bank was exactly
    degenerate — spread across anchors **0.000000000 m**, one straight line
    repeated N times (control, with real controls: **7.106836 m**). The
    expectation here is the literal ``0.0``: a v0-conditioned bank whose
    controls sum to exactly zero is the "do nothing" vocabulary the message
    describes. ``None`` keeps the declaration-only behaviour for callers that
    genuinely have no tensor (a classifier build).
    """
    if int(n_anchors) != int(expect_n):
        raise ValueError(
            f"F9: the vocabulary must stay {expect_n} anchors, got "
            f"{n_anchors}. Spec §3 F9 is a no-change item; a different "
            f"vocabulary is a different arm and must be registered as one.")
    if not bool(v0_conditioned):
        raise ValueError(
            "F9: the vocabulary must stay v0-CONDITIONED. A fixed-path bank "
            "carries `anchor_controls` of all zeros, so the anchored Gaussian "
            "would be centred on 'do nothing'.")
    if anchor_controls is not None and bool(v0_conditioned):
        if float(anchor_controls.abs().sum()) == 0.0:
            raise ValueError(
                "F9: the vocabulary DECLARES v0-conditioned but "
                "`anchor_controls` is all zeros -- the exact state the "
                "declaration cannot see. Every one of the "
                f"{int(n_anchors)} candidates rolls to the SAME straight line "
                "(MEASURED spread across anchors 0.000000000 m; a controls-"
                "carrying bank reads 7.106836 m), so the anchored Gaussian is "
                "centred on 'do nothing' and the arm would be a plausible-"
                "looking WRONG experiment. Load the anchor artifact.")


# ============================================================================
# F1 — the training draw, and the residual measurement F2 is defined by
# ============================================================================

def residual_retained(sched, t: int, t_prev: int) -> float:
    """The fraction of the ``x_t - x0_hat`` gap a DDIM step leaves behind.

    ``x_prev - x0_hat = (sqrt(1-abar_prev) / sqrt(1-abar_t)) * (x_t - x0_hat)``
    once the ``sqrt(abar)`` terms are carried through with ``abar ~ 1`` near
    the truncation point. This is the number the spec quotes as "95 % vs 28 %",
    and it is MEASURED from the alpha table rather than asserted.
    """
    s_t = float(sched.sqrt_one_minus_abar(int(t)))
    s_p = float(sched.sqrt_one_minus_abar(max(int(t_prev), 0)))
    return s_p / max(s_t, 1e-12)


def sinusoidal_time_embed(t: Tensor, d: int) -> Tensor:
    """DD's ``SinusoidalPosEmb`` (``transfuser_model_v2.py``), for tests that
    need a timestep embedding without building a decoder."""
    half = d // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device)
                      / max(half - 1, 1))
    args = t.reshape(-1, 1).float() * freqs.reshape(1, -1)
    return torch.cat([args.sin(), args.cos()], dim=-1)
