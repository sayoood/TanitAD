"""``tanitad.rl.ddv2_rl`` — DiffusionDriveV2's RL stage, the RELEASED arithmetic, model-agnostic.

⭐ WHAT THIS MODULE IS. A line-by-line port of the three pieces of
``hustvl/DiffusionDriveV2@1cd12a1`` that define its RL post-training stage and that do
NOT depend on the model they are attached to:

1. the per-denoising-step policy step with its log-probability
   (``DDIMScheduler_with_logprob.step``, ``diffusiondrivev2_model_rl.py:518-676``);
2. the advantage (``forward_train_rl``, ``:885-932``) — intra-anchor GRPO over G samples,
   truncated at zero, the ≥GT positive mask, the −1 constraint branch, the γ discount;
3. the loss (``get_rlloss``, ``:1096-1119``) — REINFORCE averaged over the NON-ZERO-advantage
   samples, plus the all-chains IL L1 with its 1.0 / 0.1 per-row weight.

The spec every line below cites is ``TanitAD Research Lab/Architecture & Inference/Research/
2026-09-15-ddv2-rl-prep/SPEC_DDV2_RL_PAPER.md`` (§ numbers in brackets). The model binding
(refcv5-v2's control-space DDIM sampler) lives in :mod:`tanitad.rl.ddv2_refc_chain`; the reward
in :mod:`tanitad.rl.pdm_proxy`.

⛔⛔ WHY A NEW MODULE AND NOT A FLAG ON ``advantage.py`` / ``posttrain.py``. MEASURED
2026-09-15 by reading both files against the released code: the library's
``composite_advantage`` is **not** DDv2's advantage and cannot be made into it by a flag —

* DDv2 truncates the **intra-anchor** advantage per sample (``clamp(min=0)``, ``:893``); the
  library adds an UN-truncated intra term to a separate **across-anchor centred** term
  (``advantage.py:208-213``) that neither the paper (Eq. 10) nor the code has;
* DDv2's ≥GT mask and −1 branch act **per sample** (``:892``, ``:900-902``); the library
  applies both to the **per-anchor mean** and pins the whole anchor (``advantage.py:209-212``);
* DDv2 divides by ``std_G + 1e-4`` (``:889``); the library defaults to centre-only;
* DDv2 averages the policy loss over **non-zero-advantage samples** (``:1099-1102``); the
  library averages over all samples (``advantage.py:244``).

Every one of those library choices is documented as deliberate, and several banked arms
depend on them, so they are LEFT UNCHANGED. This module is the released arithmetic beside
them, so an arm can be DDv2-faithful without silently changing what the banked arms meant.

⛔ NOTHING HERE READS A MODEL OUTPUT OTHER THAN THE POLICY'S OWN MEAN. The reward and the
constraint flags are handed in; the selector's score is never an input (the
``rewards.FORBIDDEN_REWARD_INPUTS`` doctrine).

Tier: T0 training-side machinery. Evidence class of the mechanism: PUBLISHED-CODE (cited per
function); of this implementation: MEASURED — ``tests/test_ddv2_rl.py`` pins it against the
banked released scheduler class executed verbatim, against analytic literals, and against
deliberate regressions that must go RED.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

__all__ = [
    "DDV2", "DDV2Constants", "rollout_labels", "abar_at", "ddim_logprob_step",
    "intra_anchor_advantage", "discount_weights", "rl_loss_per_row",
    "il_row_weights", "total_loss", "Ddv2ConfigError",
    "diffusers_alphas_cumprod", "tile_groups", "truncated_start", "rollout_chain",
    "chain_step_logprob", "step_loss_weights", "per_step_loss",
]

LOG_SQRT_2PI = math.log(math.sqrt(2.0 * math.pi))


class Ddv2ConfigError(ValueError):
    """A DDv2 constant was changed without saying so, or an input breaks the released shapes."""


@dataclass(frozen=True)
class DDV2Constants:
    """Every published constant of the RL stage, with where it was read.

    ⛔ Frozen and cited so an arm that departs from one has to construct a DIFFERENT object and
    record it — the ``--w-tac-goal default=0.0`` class (a knob nobody typed) cannot happen here.
    """
    #: G, ``diffusiondrivev2_rl_config.py:37``  [SPEC §3.5]
    group_size: int = 4
    #: rollout length in training, ``rl.py:801``  [SPEC §3.2]
    rollout_steps: int = 10
    #: label span: ``step_ratio = 20 / step_num``, ``rl.py:805``  [SPEC §3.2]
    label_span: int = 20
    #: truncation timestep of the start noise, ``rl.py:819``  [SPEC §3.1]
    trunc_t: int = 8
    #: minimum exploration std, ``rl.py:639``; P suppl. §7 / Tab. 7  [SPEC §2.3]
    explore_std_floor: float = 0.04
    #: minimum likelihood std, ``rl.py:668``; P suppl. §7 / Tab. 7  [SPEC §2.4]
    likelihood_std_floor: float = 0.10
    #: denoising discount, ``rl.py:926-932``; P Tab. 7  [SPEC §4.5]
    gamma: float = 0.8
    #: eta during the RL rollout, ``diffusiondrivev2_rl_agent.py:118``  [SPEC §1.4]
    eta: float = 1.0
    #: x0_hat clamp INSIDE the step: the release builds its scheduler without ``clip_sample=``
    #: (``rl.py:703-708``), so diffusers' default ``clip_sample=True, clip_sample_range=1.0``
    #: applies (``rl.py:609-614``). MEASURED 2026-09-15 on diffusers 0.40.0 by executing the
    #: vendored class: without this clamp the port is NOT bitwise the release.  [SPEC A18]
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    #: ``std_G + 1e-4`` divisor, ``rl.py:889``  [SPEC §4.1]
    adv_std_eps: float = 1e-4
    #: ``reward > reward_gt - 1e-6``, ``rl.py:892``  [SPEC §4.3]
    bar_eps: float = 1e-6
    #: value of the constraint branch, ``rl.py:902``  [SPEC §4.4]
    veto_value: float = -1.0
    #: IL weight on rows WITH a positive advantage, ``rl.py:1117``; P Tab. 7  [SPEC §6.5]
    il_weight_with_positive: float = 0.1
    #: IL weight on rows with NO positive advantage, ``rl.py:1116``  [SPEC §6.5]
    il_weight_no_positive: float = 1.0
    #: AdamW lr / weight decay, ``diffusiondrivev2_rl_agent.yaml:17``, ``cfg.py:120``  [SPEC §7.3]
    lr: float = 2e-4
    weight_decay: float = 1e-4
    #: epochs / total batch, P §5.2, Tab. 7  [SPEC §7.5-7.6]
    epochs: int = 10
    total_batch: int = 512

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_source"] = "hustvl/DiffusionDriveV2@1cd12a1 + arXiv 2512.07745v1 (SPEC_DDV2_RL_PAPER.md)"
        return d


#: THE constants. An arm that changes one builds its own instance and records ``to_dict()``.
DDV2 = DDV2Constants()


def rollout_labels(step_num: int = DDV2.rollout_steps,
                   span: int = DDV2.label_span) -> list[int]:
    """The RL rollout's timestep LABELS, ``round(arange(step_num) * span/step_num)[::-1]``.

    ``step_num=10, span=20`` -> ``[18, 16, ..., 2, 0]`` (``rl.py:805-806``). ⚠️ Each label is a
    ONE-UNIT transition ``t -> t-1`` because the rollout calls ``set_timesteps(1000)``
    (``rl.py:804``, ``:580-582``) — the chain does not step 18 -> 16. [SPEC §3.2, A6]
    """
    if step_num < 1:
        raise Ddv2ConfigError(f"step_num must be >= 1, got {step_num}")
    ratio = span / step_num
    return [int(round(i * ratio)) for i in range(step_num)][::-1]


def abar_at(table: Tensor, t: int) -> Tensor:
    """``alphas_cumprod[t]``, and the FINAL value 1.0 for ``t < 0``.

    ⛔ Not a clamp to ``table[0]``. The released scheduler reads
    ``self.final_alpha_cumprod`` when ``prev_timestep < 0`` (``rl.py:587``), which diffusers
    sets to 1.0 under its ``set_alpha_to_one=True`` default (framework, SPEC §3.2 UNVERIFIED
    note) — so the last RL step's mean is EXACTLY the network's ``x0_hat``. ``tanitad.refs.
    refc_sampler.DDIMSchedule._gather`` clamps negative indices to 0 instead; using it here
    would silently change the step the policy gradient weights most (61 % of its weight,
    SPEC §3.4).
    """
    if t < 0:
        return torch.ones((), dtype=table.dtype, device=table.device)
    return table[int(t)]


def ddim_logprob_step(x0_hat: Tensor, sample: Tensor, t: int, t_prev: int,
                      alphas_cumprod: Tensor, *, eta: float = DDV2.eta,
                      explore_std_floor: float = DDV2.explore_std_floor,
                      likelihood_std_floor: float = DDV2.likelihood_std_floor,
                      clip_sample: bool = DDV2.clip_sample,
                      clip_sample_range: float = DDV2.clip_sample_range,
                      prev_sample: Tensor | None = None,
                      generator: torch.Generator | None = None
                      ) -> tuple[Tensor, Tensor, Tensor]:
    """One RL denoising step. -> ``(prev_sample, log_prob [..., N], prev_sample_mean)``.

    The released ``DDIMScheduler_with_logprob.step`` with ``prediction_type="sample"``,
    arithmetic in the released order (``rl.py:586-675``):

    * ``eps_hat = (x_t - sqrt(abar_t) x0_hat) / sqrt(1 - abar_t)`` — from the UNCLAMPED x0_hat
    * ``x0_hat <- clamp(x0_hat, -1, 1)`` (diffusers' ``clip_sample=True`` default, SPEC A18) —
      AFTER eps_hat, so a clamped coordinate still moves the mean through ``eps_hat`` (with the
      opposite sign) at t > 0 and not at all at the final step
    * ``sigma_t = eta * sqrt((1-abar_prev)/(1-abar_t) * (1 - abar_t/abar_prev))``, floored at 1e-10
    * ``mean = sqrt(abar_prev) x0_hat + sqrt(max(1 - abar_prev - sigma_t^2, 0)) eps_hat``
    * exploration (``eta > 0`` only): TWO scalars per trajectory, ``randn([B, N, 1, 1]) *
      max(sigma_t, 0.04) + 1`` per axis, broadcast over the ``S`` waypoints; the additive pair is
      DRAWN (so the RNG stream matches the release) and multiplied by exactly 0.0
    * ``log_prob = sum_{S,2} [ -(prev.detach() - mean)^2 / (2 s^2) - log s - log sqrt(2 pi) ]``,
      ``s = max(sigma_t, 0.1)`` — NOT the sampler's density (SPEC A4)

    ``x0_hat`` / ``sample`` are ``[B, N, S, 2]`` in the DIFFUSION STATE's own normalised units.
    ``prev_sample`` given => nothing is drawn and the log-probability is evaluated AT it (the
    grad pass, ``rl.py:1085-1091``). Gradient reaches the caller only through ``x0_hat``.
    """
    if x0_hat.shape != sample.shape or x0_hat.dim() != 4 or x0_hat.shape[-1] != 2:
        raise Ddv2ConfigError(
            f"x0_hat {tuple(x0_hat.shape)} / sample {tuple(sample.shape)} must both be "
            "[B, N, S, 2] — the released noise draws one scalar per (B, N) and broadcasts "
            "over S; any other rank would change WHAT a 'trajectory' is.")
    dtype = x0_hat.dtype
    a_t = abar_at(alphas_cumprod, t).to(dtype)
    a_p = abar_at(alphas_cumprod, t_prev).to(dtype)
    b_t = 1 - a_t
    # prediction_type == "sample" (rl.py:596-598)
    pred_original_sample = x0_hat
    pred_epsilon = (sample - a_t ** 0.5 * pred_original_sample) / b_t ** 0.5
    # clip_sample (rl.py:609-614): diffusers default True, range 1.0 — after eps_hat, as released
    if clip_sample:
        pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)
    # _get_variance (diffusers DDIMScheduler), rl.py:618
    variance = (1 - a_p) / b_t * (1 - a_t / a_p)
    std_dev_t = (eta * variance ** 0.5).clamp(min=1e-10)
    direction = (1 - a_p - std_dev_t ** 2).clamp(min=0) ** 0.5 * pred_epsilon  # rl.py:627
    prev_sample_mean = a_p ** 0.5 * pred_original_sample + direction            # rl.py:630

    if eta > 0:                                                                   # rl.py:638-643
        std_mul = torch.clip(std_dev_t, min=explore_std_floor)
        std_add = torch.zeros((), dtype=dtype, device=x0_hat.device)
    else:
        std_mul = torch.zeros((), dtype=dtype, device=x0_hat.device)
        std_add = torch.zeros((), dtype=dtype, device=x0_hat.device)
    if prev_sample is None:                                                       # rl.py:644-666
        b, n, s = x0_hat.shape[0], x0_hat.shape[1], x0_hat.shape[2]
        shp = (b, n, 1, 1)
        kw = dict(generator=generator, device=x0_hat.device, dtype=dtype)
        horizon = torch.randn(shp, **kw) * std_mul + 1.0
        vert = torch.randn(shp, **kw) * std_mul + 1.0
        mul = torch.cat((horizon, vert), dim=-1).repeat(1, 1, s, 1)
        add_x = torch.randn(shp, **kw)
        add_y = torch.randn(shp, **kw)
        add = torch.cat((add_x, add_y), dim=-1).repeat(1, 1, s, 1)
        prev_sample = prev_sample_mean * mul + std_add * add
    lik_std = torch.clip(std_dev_t, min=likelihood_std_floor)                    # rl.py:668
    # ⚠️ the constant is built exactly as the release builds it (a float32 tensor), not as a
    # Python float: the two differ in the last ulp, and the vendored-reference test is bitwise.
    log_prob = (-((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (lik_std ** 2))
                - torch.log(lik_std)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))).to(lik_std.device))
    return prev_sample.to(sample.dtype), log_prob.sum(dim=(-2, -1)), prev_sample_mean


def intra_anchor_advantage(reward: Tensor, reward_gt: Tensor, constraint_fail: Tensor, *,
                           consts: DDV2Constants = DDV2, use_gt_bar: bool = True
                           ) -> dict[str, Tensor]:
    """DDv2's per-sample advantage over ``reward [B, G, N]``. -> dict with ``advantage [B, G, N]``.

    ``rl.py:886-902``, in the released order:
      1. ``A = (r - mean_G r) / (std_G r + 1e-4)`` — per ANCHOR over its G samples (P Eq. 8);
      2. ``A = clamp(A, min=0) * [r > r_gt - 1e-6]`` — truncation + the ≥GT mask (P Eq. 10 +
         the code-only bar, SPEC A7);
      3. ``A = -1`` wherever ``constraint_fail`` (NC != 1 or DAC != 1 in the release, SPEC A8)
         — applied LAST, so it overrides the mask.

    ``reward_gt [B]`` is the logged trajectory scored under the SAME reward in the same call
    (``rl.py:867-873``). ⛔ ``use_gt_bar=False`` exists ONLY for a declared ablation; the
    returned ``frac_admitted_by_bar`` is ``None`` then, so a record cannot claim a bar it did not
    apply.
    """
    if reward.dim() != 3:
        raise Ddv2ConfigError(f"reward must be [B, G, N], got {tuple(reward.shape)}")
    b, g, n = reward.shape
    if g < 2:
        raise Ddv2ConfigError(
            f"G={g}: a group of one has std NaN (unbiased) and no relative signal; the release "
            "uses G=4 (cfg.py:37). Refusing rather than returning a silent zero/NaN advantage.")
    if constraint_fail.shape != reward.shape or constraint_fail.dtype != torch.bool:
        raise Ddv2ConfigError(
            f"constraint_fail must be a bool [B, G, N] like reward; got "
            f"{tuple(constraint_fail.shape)} {constraint_fail.dtype}")
    if tuple(reward_gt.shape) != (b,):
        raise Ddv2ConfigError(f"reward_gt must be [B]={b}, got {tuple(reward_gt.shape)}")
    mean_g = reward.mean(dim=1, keepdim=True)
    std_g = reward.std(dim=1, keepdim=True)                     # unbiased, torch default
    adv = (reward - mean_g) / (std_g + consts.adv_std_eps)
    raw_positive = adv > 0
    if use_gt_bar:
        mask_positive = reward > (reward_gt[:, None, None] - consts.bar_eps)
        adv = adv.clamp(min=0) * mask_positive.to(adv.dtype)
    else:
        mask_positive = None
        adv = adv.clamp(min=0)
    adv = torch.where(constraint_fail, torch.full_like(adv, consts.veto_value), adv)
    out = {
        "advantage": adv,
        "frac_positive_before_bar": raw_positive.float().mean(),
        "frac_admitted_by_bar": (mask_positive.float().mean() if mask_positive is not None
                                 else None),
        "frac_positive_after_bar": (adv > 0).float().mean(),
        "frac_constraint_fail": constraint_fail.float().mean(),
        "frac_nonzero": (adv != 0).float().mean(),
    }
    return out


def discount_weights(step_num: int = DDV2.rollout_steps, gamma: float = DDV2.gamma,
                     device=None, dtype=torch.float32) -> Tensor:
    """``gamma ** (step_num - i - 1)`` for rollout step ``i`` — weight 1.0 on the FINAL step.

    ``rl.py:926-931``. ⚠️ Index ``i`` is the ROLLOUT order (``i = 0`` is the noisiest label 18),
    so the discount down-weights early denoising steps, as P §4.4 states. [SPEC §4.5]
    """
    return torch.tensor([gamma ** (step_num - i - 1) for i in range(step_num)],
                        device=device, dtype=dtype)


def rl_loss_per_row(logp: Tensor, advantage: Tensor, discount: Tensor
                    ) -> dict[str, Tensor]:
    """REINFORCE per batch row, averaged over NON-ZERO-advantage samples. -> ``rl_loss_b [B]``.

    ``logp [B, M, T]`` (M = G·N chains, T rollout steps) carries gradient; ``advantage [B, M]``
    is detached here; ``discount [T]``. Released arithmetic (``rl.py:924-925``, ``:1096-1102``):

        per_token = -exp(logp - logp.detach()) * (A[..., None] * gamma_t)     # ratio == 1
        rl_b      = mean_T [ sum_M(per_token * [per_token != 0]) / max(1, count_M(per_token != 0)) ]

    ⛔ The average is over the samples that CARRY advantage, not over M (SPEC A11): a row with 3
    admitted samples weights each of them as heavily as a row with 80 does. A row with none
    contributes exactly 0.
    """
    if logp.dim() != 3:
        raise Ddv2ConfigError(f"logp must be [B, M, T], got {tuple(logp.shape)}")
    if advantage.shape != logp.shape[:2]:
        raise Ddv2ConfigError(f"advantage {tuple(advantage.shape)} must be [B, M] "
                              f"matching logp {tuple(logp.shape)}")
    if discount.shape != (logp.shape[2],):
        raise Ddv2ConfigError(f"discount {tuple(discount.shape)} must be [T]={logp.shape[2]}")
    adv_t = advantage.detach().unsqueeze(-1) * discount.to(logp.dtype)   # [B, M, T]
    per_token = -torch.exp(logp - logp.detach()) * adv_t
    mask_nz = per_token != 0
    rl_bt = ((per_token * mask_nz).sum(dim=1)
             / mask_nz.sum(dim=1).clamp_min(1))                          # [B, T]
    return {"rl_loss_b": rl_bt.mean(dim=-1), "adv_discounted": adv_t,
            "n_nonzero_per_row": mask_nz[..., -1].sum(dim=1)}


def il_row_weights(adv_discounted: Tensor, consts: DDV2Constants = DDV2) -> Tensor:
    """``1.0`` on rows with NO positive advantage, ``0.1`` otherwise (``rl.py:1113-1117``). -> [B]."""
    has_positive = (adv_discounted > 0).any(dim=2).any(dim=1)
    return torch.where(has_positive,
                       torch.full(has_positive.shape, consts.il_weight_with_positive,
                                  device=adv_discounted.device),
                       torch.full(has_positive.shape, consts.il_weight_no_positive,
                                  device=adv_discounted.device))


def total_loss(rl_loss_b: Tensor, il_scalar: Tensor, il_weight_b: Tensor) -> Tensor:
    """``mean_B(rl_b + w_b * IL)`` with ``IL`` a BATCH-GLOBAL scalar (``rl.py:1104-1119``).

    ⛔ The release's IL "per row" is ``traj_l1.mean()`` — one scalar added to every row — so the
    per-row weights only set the batch-average λ (SPEC A14). Refuses a non-scalar IL so a port
    that 'fixes' it to per-row is a visible decision, not a silent one.
    """
    if il_scalar.dim() != 0:
        raise Ddv2ConfigError(
            f"il_scalar must be a 0-dim tensor (the release's batch-global L1), got "
            f"{tuple(il_scalar.shape)}")
    if rl_loss_b.shape != il_weight_b.shape:
        raise Ddv2ConfigError(f"rl_loss_b {tuple(rl_loss_b.shape)} vs il_weight_b "
                              f"{tuple(il_weight_b.shape)}")
    return (rl_loss_b + il_weight_b.to(rl_loss_b.dtype) * il_scalar).mean()


# =========================================================================== #
# THE CHAIN — rollout (no grad) and grad pass, model-agnostic                   #
# =========================================================================== #
# Added 2026-09-15 (resume pass). The model enters ONLY through ``x0_fn(x_t, t)``
# -> ``x0_hat`` in the diffusion state's own normalised units, so the same chain
# drives the vendored-reference tests, a toy network and refcv5-v2's decoder
# (:mod:`tanitad.rl.ddv2_refc_chain`).

def diffusers_alphas_cumprod(device=None) -> Tensor:
    """diffusers' ``scaled_linear`` table in ITS dtype (float32), as the release reads it.

    ``linspace(sqrt(1e-4), sqrt(0.02), 1000, float32) ** 2`` then ``cumprod(1 - betas)``.
    ⛔ Not ``tanitad.refs.refc_sampler.DDIMSchedule()``'s default float64 table: that one is
    the more accurate of the two (refc_sampler.py:214-231) and therefore NOT the release's
    numbers in the last ulps. Pinned bit-equal to the vendored scheduler in the tests.
    """
    betas = torch.linspace(0.0001 ** 0.5, 0.02 ** 0.5, 1000, dtype=torch.float32) ** 2
    return torch.cumprod(1.0 - betas, dim=0).to(device)


def tile_groups(x: Tensor, groups: int) -> Tensor:
    """``[B, N, ...] -> [B, G*N, ...]``, GROUP-MAJOR: index ``m = g * N + n``.

    The release tiles ``plan_anchor.repeat(bs, num_groups, 1, 1, 1).view(bs, G*N, 8, 2)``
    (``rl.py:813-815``) and un-tiles rewards with ``.view(bs, num_groups, N)`` (``rl.py:886``);
    both orders must agree or every advantage lands on the wrong anchor.
    """
    if groups < 1:
        raise Ddv2ConfigError(f"groups must be >= 1, got {groups}")
    b, n = x.shape[0], x.shape[1]
    return x.unsqueeze(1).expand(b, groups, *x.shape[1:]).reshape(b, groups * n, *x.shape[2:])


def truncated_start(anchor_state: Tensor, groups: int, alphas_cumprod: Tensor, *,
                    trunc_t: int = DDV2.trunc_t,
                    generator: torch.Generator | None = None,
                    noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """The anchored-Gaussian start. ``[B, N, S, 2] -> ([B, G*N, S, 2] x_t, noise)``.

    ``sqrt(abar_8) a + sqrt(1 - abar_8) eps`` on the tiled anchors, ONE fresh ``eps`` per tiled
    copy (``rl.py:816-820``; diffusers ``add_noise``). ``noise`` may be supplied so a parity
    test can share the draw with another sampler.
    """
    x0 = tile_groups(anchor_state, groups)
    if noise is None:
        noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
    elif noise.shape != x0.shape:
        raise Ddv2ConfigError(f"noise {tuple(noise.shape)} != tiled start {tuple(x0.shape)}")
    a = alphas_cumprod[int(trunc_t)].to(device=x0.device, dtype=x0.dtype)
    return a ** 0.5 * x0 + (1 - a) ** 0.5 * noise, noise


def rollout_chain(x0_fn, x_start: Tensor, alphas_cumprod: Tensor, *,
                  labels: list[int] | None = None, eta: float = DDV2.eta,
                  consts: DDV2Constants = DDV2,
                  generator: torch.Generator | None = None,
                  keep_x0: bool = False) -> dict:
    """The RL rollout (``forward_train_rl``, ``rl.py:825-862``). Call under ``no_grad``.

    -> ``chain [B, M, S, 2, T+1]`` (the start and every sample: the release's
    ``all_diffusion_output``), ``logp [B, M, T]``, ``labels``; with ``keep_x0`` also
    ``x0 [B, M, S, 2, T]`` (the network's prediction at each step, pre-clamp).

    ⭐ Transitions are ``t -> t - 1`` (``set_timesteps(1000)``, SPEC §3.2), NOT label to next
    label; ``abar(-1) = 1.0`` at the last step.
    """
    labels = rollout_labels() if labels is None else [int(t) for t in labels]
    x = x_start
    states, logps, x0s = [x_start], [], []
    for t in labels:
        x0_hat = x0_fn(x, t)
        if x0_hat.shape != x.shape:
            raise Ddv2ConfigError(f"x0_fn returned {tuple(x0_hat.shape)} for state "
                                  f"{tuple(x.shape)}")
        prev, lp, _ = ddim_logprob_step(
            x0_hat, x, t, t - 1, alphas_cumprod, eta=eta,
            explore_std_floor=consts.explore_std_floor,
            likelihood_std_floor=consts.likelihood_std_floor,
            clip_sample=consts.clip_sample, clip_sample_range=consts.clip_sample_range,
            generator=generator)
        states.append(prev)
        logps.append(lp)
        if keep_x0:
            x0s.append(x0_hat)
        x = prev
    out = {"chain": torch.stack(states, dim=-1), "logp": torch.stack(logps, dim=-1),
           "labels": labels}
    if keep_x0:
        out["x0"] = torch.stack(x0s, dim=-1)
    return out


def chain_step_logprob(x0_fn, chain: Tensor, i: int, alphas_cumprod: Tensor, *,
                       labels: list[int], eta: float = DDV2.eta,
                       consts: DDV2Constants = DDV2) -> tuple[Tensor, Tensor]:
    """The GRAD pass at rollout step ``i`` (``get_rlloss``, ``rl.py:1047-1092``).

    Re-runs the network WITH gradient on the STORED state ``chain[..., i]`` and evaluates the
    log-probability AT the stored next state ``chain[..., i+1]``. -> ``(logp [B, M], x0_hat)``.
    """
    t = int(labels[i])
    x_t = chain[..., i]
    x0_hat = x0_fn(x_t, t)
    _, lp, _ = ddim_logprob_step(
        x0_hat, x_t, t, t - 1, alphas_cumprod, eta=eta,
        explore_std_floor=consts.explore_std_floor,
        likelihood_std_floor=consts.likelihood_std_floor,
        clip_sample=consts.clip_sample, clip_sample_range=consts.clip_sample_range,
        prev_sample=chain[..., i + 1])
    return lp, x0_hat


def step_loss_weights(advantage: Tensor, n_steps: int, *, consts: DDV2Constants = DDV2
                      ) -> dict[str, Tensor]:
    """The FIXED coefficients that make the release's loss separable over steps.

    ``advantage [B, M]`` (detached). The released loss (``rl.py:1096-1119``) is
        ``mean_B [ mean_T( sum_M(-r * A * g_t) / c_b ) + w_b * IL ]``,  ``r = exp(logp - logp.detach())``
    with ``c_b = max(1, #{m : A_bm != 0})`` (independent of ``t`` because every ``g_t > 0``)
    and ``IL = (1/T) sum_t IL_t`` batch-global. Every coefficient is known before the grad
    pass, so ``loss = sum_t loss_t`` EXACTLY and a per-step ``backward()`` accumulates the SAME
    gradient as one graph over all steps (pinned in the tests) — which is what lets a 468-chain
    grad pass fit an 8 GB card.

    -> ``coef_rl [B, M, T]`` (multiplies ``-r``), ``il_weight_b [B]``, ``il_coef`` (0-dim,
    multiplies ``IL_t``), ``n_nonzero_b [B]``, ``adv_discounted [B, M, T]``.
    """
    adv = advantage.detach()
    disc = discount_weights(n_steps, consts.gamma, device=adv.device, dtype=adv.dtype)
    adv_t = adv.unsqueeze(-1) * disc                                   # [B, M, T]
    nz = (adv_t != 0)
    c_bt = nz.sum(dim=1).clamp_min(1).to(adv.dtype)                     # [B, T]
    b = adv.shape[0]
    coef = adv_t / c_bt.unsqueeze(1) / (n_steps * b)                   # [B, M, T]
    il_w = il_row_weights(adv_t, consts)                                # [B]
    il_coef = il_w.to(adv.dtype).mean() / n_steps
    return {"coef_rl": coef, "il_weight_b": il_w, "il_coef": il_coef,
            "n_nonzero_b": nz[..., -1].sum(dim=1), "adv_discounted": adv_t}


def per_step_loss(logp_i: Tensor, il_i: Tensor, weights: dict, i: int) -> Tensor:
    """``loss_i = sum_{B,M} -exp(logp - logp.detach()) * coef[..., i] + il_coef * IL_i``.

    ``il_i`` is THIS step's batch-global mean L1 (0-dim). Summing ``loss_i`` over ``i``
    reproduces :func:`total_loss` of :func:`rl_loss_per_row` exactly (value and gradient).
    """
    if il_i.dim() != 0:
        raise Ddv2ConfigError(f"il_i must be the batch-global 0-dim L1, got {tuple(il_i.shape)}")
    r = torch.exp(logp_i - logp_i.detach())
    return (-(r * weights["coef_rl"][..., i])).sum() + weights["il_coef"] * il_i
