"""``tanitad.rl.ddv2_il`` — lever **L1 / D9**: the two changes to DiffusionDriveV2's RL stage,
and NOTHING else.

⭐ WHY THIS MODULE EXISTS AND WHY IT IS NOT IN ``ddv2_rl.py``. :mod:`tanitad.rl.ddv2_rl` is a
line-by-line port of ``hustvl/DiffusionDriveV2@1cd12a1`` and its value is that it is the
release and can be read as the release. Everything here is a **declared departure** from it, so
it lives beside it — the same reason ``ddv2_rl.py`` was built beside ``advantage.py`` rather
than as a flag on it. ``ddv2_rl.py`` is NOT edited by this lever; the departures are applied as
post-hoc transforms of the objects it returns, which is what makes the DIFF checkable.

⛔ THE MEASURED FAILURE THIS LEVER ANSWERS (``…/Research/2026-09-15-ddv2-rl-prep/RESULT.md``
§7.2, §9). The faithful port was validated and FAILED, and the cause is measured: **DDv2's
imitation term, not its RL term, does the damage.**

* the release's IL term is ``(path − gt).abs().mean()`` over **all** ``G·N = 468`` chains at all
  10 rollout steps — every chain is pulled to the single logged trajectory;
* it **collapsed refcv5-v2's 117-anchor fan by 93 %**: held-out fan endpoint spread
  **37.5 m [32.8, 42.0] → 2.45 m [1.95, 2.99]**;
* switching only the policy gradient off reproduced almost all the harm (T1 ADE vs the cold
  start: RL −0.083 m [0.056, 0.115], RL-off −0.081 m [0.051, 0.116]; RL − RL-off
  −0.0012 m [−0.0285, +0.0268], **not separated**);
* one RL seed diverged under the release's no-clipping recipe (grad norm 15,712 vs 147).

**THE TWO CHANGES, AND ONLY THESE TWO.**

1. :func:`imitation_term` — replace the all-modes L1 with the **mode-preserving matched-anchor**
   form our own trainer already uses (``stack/scripts/refc_v3_train.py:2383-2391``), so the
   imitation signal lands on the anchor nearest the GT instead of dragging all 468 chains onto
   it. The plain **λ ≈ 0.01** variant is exposed beside it as the cheaper alternative arm
   (:func:`apply_lambda_scale`), so the two can be compared rather than chosen by argument.
2. :func:`clip_gradients` — **gradient clipping** (the release clips nothing). Max-norm
   **100** — AMENDED 2026-09-16 from 1.0 **before any arm ran**, because 1.0 was MEASURED to
   bind on 600/600 steps of all three banked arms (an every-step 17-62× rescale) while 100
   binds on 1/600 of a stable arm and 276/600 of the diverged one. See :data:`GRAD_CLIP`.

Every other release element is untouched and is asserted untouched by
``…/Research/2026-09-16-refcv6-rl-l1/DIFF_L1_VS_RELEASE.md`` and by
``stack/tests/test_ddv2_il.py::test_the_release_path_is_bit_identical_when_the_lever_is_off``:
truncated start (t = 8), the 10-label chain, the two-scalar exploration with its 0.04 floor, the
0.1 likelihood floor, G = 4, intra-anchor normalisation, per-sample truncation with the ≥GT
mask, γ = 0.8, REINFORCE over non-zero samples, AdamW 2e-4 / wd 1e-4.

⛔ AMENDMENTS, all made BEFORE any arm ran (so they are pre-run, not post-hoc):
**A-1 (2026-09-16)** — the gradient-clip max-norm 1.0 → **100**, on measured evidence that 1.0
was an every-step rescale rather than a spike guard. Recorded in
``Project Steering/PREREG_DDV2_RL_VALIDATION.md`` §11 and in :data:`GRAD_CLIP`.

⛔ SCOPE LIMITS THAT CARRY FORWARD (they do not change because the objective did):
the three 2026-09-15 checkpoints are **burned** (trained on 100 of the 141 EVAL clips) and must
never be scored on the 141-clip panel; the dev-box scale is ≈ 0.4 % of the paper's optimiser
samples; the reward is a proxy with **no drivable-area term** (our maps do not yet cover the
training clips).

Tier: T0 training-side machinery. Evidence class of the mechanism: MEASURED (ours, the
2026-09-15 validation); of this implementation: MEASURED — ``stack/tests/test_ddv2_il.py`` pins
it against analytic literals, against the release form it replaces, and by a **mutation proof**
(a synthetic fan whose spread must survive the mode-preserving term and must NOT survive the
release's all-modes term).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from tanitad.rl.ddv2_rl import DDV2, DDV2Constants, Ddv2ConfigError

__all__ = [
    "IL_FORMS", "RELEASE_IL_FORM", "GRAD_CLIP", "IlSettings",
    "matched_anchor_index", "all_modes_il", "matched_anchor_il", "imitation_term",
    "apply_lambda_scale", "clip_gradients", "fan_endpoint_spread", "anchor_match_diagnostics",
]

#: the release's own IL form — the all-modes L1 of ``rl.py:1104-1112``
RELEASE_IL_FORM = "release_all_modes"

#: every IL form an arm of this lever may run
#:
#: * ``release_all_modes``    — the release, unchanged (the baseline this lever departs from)
#: * ``matched_anchor``       — CHANGE 1: the trainer's nearest-anchor L1, mode preserving
#: * ``release_all_modes_lambda`` — the cheaper alternative arm: the release's FORM at a weaker
#:   λ (``lambda_scale`` multiplies the release's advantage-derived 0.1 / 1.0 row weights)
IL_FORMS = (RELEASE_IL_FORM, "matched_anchor", "release_all_modes_lambda")

#: CHANGE 2. ``torch.nn.utils.clip_grad_norm_`` max-norm. The release clips nothing
#: (``rl.py`` has no ``clip_grad``); RL-s1 diverged at grad norm 15,712.3 against RL-s0's 147.3
#: (RESULT §7.1).
#:
#: ⭐⭐ **AMENDMENT A-1, 2026-09-16, BEFORE ANY ARM RAN: 1.0 → 100.0.**
#: The value was 1.0 (the PI's first instruction). MEASURED on the banked logs
#: (``…/2026-09-15-ddv2-rl-prep/raw/metrics_arm-*.jsonl``, 600 steps × 3 arms) and
#: INDEPENDENTLY VERIFIED by the coordinator:
#:
#: * ``> 1.0`` binds on **600/600** steps of all three arms — min norm 1.87 / 2.22 / 3.45,
#:   medians 16.67 / 17.93 / 61.96. At 1.0 the clip is a **17-62× rescale on every step**, not
#:   a guard;
#: * ``> 100`` binds on **1/600** for each stable arm and **276/600** for the diverged seed
#:   (max 15,712.3) — i.e. 100 is an **actual spike guard**, which is what C2 is for.
#:
#: 1.0 and 0 (the release) stay available as declared arms; 1.0 becomes interesting only if the
#: diverging seed also fails at 100. See ``Project Steering/PREREG_DDV2_RL_VALIDATION.md``
#: §11 AMENDMENT A-1.
GRAD_CLIP: float = 100.0


@dataclass(frozen=True)
class IlSettings:
    """The imitation objective an arm runs, recorded verbatim in ``run.json``.

    ⛔ Defaults are **the release**, so an arm that departs has to say so in a field that is
    written to the run record — the ``--w-tac-goal default=0.0`` class (a knob nobody typed)
    cannot happen here.
    """
    #: one of :data:`IL_FORMS`
    form: str = RELEASE_IL_FORM
    #: multiplies the release's per-row IL weights (``1.0`` = the release's 0.1 / 1.0).
    #: ``0.1`` gives 0.01 / 0.1 — and MEASURED D6 says essentially every cold-start row carries
    #: a positive advantage, so that row's weight is the one that binds: **λ ≈ 0.01**.
    lambda_scale: float = 1.0
    #: ``torch.nn.utils.clip_grad_norm_`` max-norm; ``None`` or ``0`` = the release (no clip).
    #: Default :data:`GRAD_CLIP` = **100** (AMENDMENT A-1, 2026-09-16, pre-run).
    grad_clip: float | None = GRAD_CLIP

    def __post_init__(self):
        if self.form not in IL_FORMS:
            raise Ddv2ConfigError(f"il form {self.form!r} not in {IL_FORMS}")
        if not (self.lambda_scale > 0):
            raise Ddv2ConfigError(
                f"lambda_scale must be > 0, got {self.lambda_scale}. A scale of 0 is not a "
                "weaker IL term — it DELETES the term, which is a different arm (and would "
                "silently also delete the release's 1.0-weight fallback on rows with no "
                "positive advantage). Run that as its own declared arm, not as a scale.")
        if self.grad_clip is not None and self.grad_clip < 0:
            raise Ddv2ConfigError(f"grad_clip must be >= 0 or None, got {self.grad_clip}")
        if self.form == "release_all_modes_lambda" and self.lambda_scale == 1.0:
            raise Ddv2ConfigError(
                "form='release_all_modes_lambda' with lambda_scale=1.0 IS the release "
                "(form='release_all_modes') under a different name. Refusing, so a run record "
                "cannot claim a lambda arm that never weakened anything.")

    @property
    def is_release(self) -> bool:
        """True only when this settings object reproduces the release exactly."""
        return (self.form == RELEASE_IL_FORM and self.lambda_scale == 1.0
                and not self.grad_clip)

    def effective_lambda(self, consts: DDV2Constants = DDV2) -> dict:
        """The λ actually multiplying the IL scalar on each row class."""
        return {"rows_with_positive_advantage": consts.il_weight_with_positive * self.lambda_scale,
                "rows_with_no_positive_advantage": consts.il_weight_no_positive * self.lambda_scale}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_release"] = self.is_release
        d["effective_lambda"] = self.effective_lambda()
        d["_lever"] = ("L1 / D9 — mode-preserving IL + grad clip 1.0; everything else is "
                       "hustvl/DiffusionDriveV2@1cd12a1")
        return d


# =========================================================================== #
# CHANGE 1 — the imitation term                                               #
# =========================================================================== #
def matched_anchor_index(bank: Tensor, gt: Tensor, valid: Tensor | None = None) -> Tensor:
    """``a_star [B]`` — the anchor of ``bank`` nearest the GT, the TRAINER's own arithmetic.

    ``stack/scripts/refc_v3_train.py:2383-2386`` verbatim::

        dist   = (((traj_tgt[:, None] - anchors) ** 2).sum(-1) * sv[:, None]).sum(-1)
        a_star = dist.argmin(dim=1)

    ``bank [B, N, S, 2]`` is the DECODED bank in metres — ``out["anchor_bank"]``, which is
    ``_sample``'s own ``bank`` argument (``refc.py:2615``), i.e. this window's v0-rolled
    vocabulary and not ``decoder.anchors``. ⛔ Matching against ``decoder.anchors`` on a
    v0-conditioned build would supervise a geometry the model never emitted — the defect the
    trainer's own comment (``:2376-2382``) exists to refuse, and it would read plausibly.

    ``gt [B, S, 2]``; ``valid [B, S]`` is the trainer's ``sv`` (all-ones here: the RL runner
    admits only windows with a full 6 s future, ``ddv2_rl_refcv5.py:_select``), kept so the two
    call sites are the same function and not a look-alike.
    """
    if bank.dim() != 4 or bank.shape[-1] != 2:
        raise Ddv2ConfigError(f"bank must be [B, N, S, 2], got {tuple(bank.shape)}")
    if gt.dim() != 3 or gt.shape[0] != bank.shape[0] or gt.shape[1:] != bank.shape[2:]:
        raise Ddv2ConfigError(f"gt {tuple(gt.shape)} must be [B, S, 2] matching bank "
                              f"{tuple(bank.shape)}")
    gt = gt.to(bank.dtype)
    sq = ((gt[:, None] - bank) ** 2).sum(-1)                        # [B, N, S]
    if valid is not None:
        if valid.shape != gt.shape[:2]:
            raise Ddv2ConfigError(f"valid {tuple(valid.shape)} must be [B, S]={tuple(gt.shape[:2])}")
        sq = sq * valid.to(sq.dtype)[:, None]
    return sq.sum(-1).argmin(dim=1)


def all_modes_il(path: Tensor, gt: Tensor) -> Tensor:
    """THE RELEASE'S IL TERM, bit for bit: ``(path − gt[:, None]).abs().mean()``. -> 0-dim.

    ``rl.py:1104-1112`` (``traj_l1.mean()``), as run on 2026-09-15
    (``ddv2_rl_refcv5.py`` at 424f9e0). ``path [B, M, S, 2]`` in metres, ``gt [B, S, 2]``.

    ⛔ THIS IS THE TERM THAT DID THE DAMAGE. It is kept, named, and tested precisely so the
    lever's arms can be compared against it instead of against a description of it.
    """
    _check_path_gt(path, gt)
    return (path - gt[:, None]).abs().mean()


def matched_anchor_il(path: Tensor, gt: Tensor, a_star: Tensor, n_anchors: int) -> Tensor:
    """CHANGE 1 — the MODE-PRESERVING L1: only the matched anchor's chains are pulled. -> 0-dim.

    ``path [B, M, S, 2]`` with ``M = G · N`` in the release's **GROUP-MAJOR** order
    (``m = g·N + n``, ``ddv2_rl.tile_groups``); ``a_star [B]`` from
    :func:`matched_anchor_index`. The G chains of anchor ``a_star`` receive the L1; the other
    ``M − G`` receive **no imitation gradient at all** and keep their own geometry — which is
    exactly what "the fan must not collapse" means.

    ⭐ NORMALISATION, AND THE ALTERNATIVE THAT WAS REJECTED. This is a ``mean`` over the
    ``G · S · 2`` matched entries, which is the TRAINER's convention
    (``refc_v3_train.py:2388-2390`` divides by ``sv.sum() * 2``, i.e. per supervised waypoint).
    The total imitation budget per batch is therefore the SAME as the release's — the sum of
    per-entry weights is 1 in both forms — it is just spent on the mode that should carry it
    instead of smeared over 468 chains. The rejected alternative, ``sum(matched) / (M·S·2)``,
    would keep each chain's weight at the release's value and shrink the batch's total IL
    budget by ``N`` = 117×; that is a λ change wearing a mode-preservation costume, and it
    would confound this lever with the λ arm it is being compared against.

    ⛔ NOT ``argmin`` OVER THE PREDICTIONS. A winner-take-all over ``path`` itself would let the
    objective choose whichever chain already sits nearest the GT — a moving target that
    supervises no fixed mode and cannot be attributed to an anchor. The match is over the
    fixed BANK, as the trainer does it.
    """
    _check_path_gt(path, gt)
    b, m = path.shape[0], path.shape[1]
    n = int(n_anchors)
    if n < 1 or m % n:
        raise Ddv2ConfigError(
            f"M={m} is not a whole number of groups of N={n} anchors; the release tiles "
            "GROUP-MAJOR (rl.py:813-815) and a wrong N would gather other anchors' chains.")
    if tuple(a_star.shape) != (b,):
        raise Ddv2ConfigError(f"a_star must be [B]={b}, got {tuple(a_star.shape)}")
    if a_star.dtype not in (torch.int16, torch.int32, torch.int64):
        raise Ddv2ConfigError(
            f"a_star must be an integer index, got {a_star.dtype}. A float 'index' would be "
            "truncated toward zero by `gather` and would silently supervise anchor floor(a).")
    if int(a_star.max()) >= n or int(a_star.min()) < 0:
        raise Ddv2ConfigError(f"a_star out of range for N={n}: [{int(a_star.min())}, "
                              f"{int(a_star.max())}]")
    g = m // n
    s = path.shape[2]
    p = path.reshape(b, g, n, s, 2)
    # ⛔ gather, not `p[arange(b), :, a_star]`: separated advanced indices silently REORDER the
    # result's dimensions, and the reordered tensor has the same shape here (B, G, S, 2) — so
    # the defect would be invisible. Pinned by `test_group_major_gather_picks_the_right_chains`.
    idx = a_star.to(torch.long).view(b, 1, 1, 1, 1).expand(b, g, 1, s, 2)
    sel = p.gather(2, idx).squeeze(2)                               # [B, G, S, 2]
    return (sel - gt.to(sel.dtype)[:, None]).abs().mean()


def imitation_term(path: Tensor, gt: Tensor, settings: IlSettings, *,
                   bank: Tensor | None = None, n_anchors: int | None = None,
                   a_star: Tensor | None = None) -> Tensor:
    """The IL scalar this arm's ``settings`` say to use. -> 0-dim, as ``per_step_loss`` requires.

    ``release_all_modes`` and ``release_all_modes_lambda`` share the release's FORM (the λ arm
    differs only in the row weights, applied by :func:`apply_lambda_scale`); ``matched_anchor``
    needs ``n_anchors`` and either ``a_star`` or ``bank``.
    """
    if settings.form in (RELEASE_IL_FORM, "release_all_modes_lambda"):
        return all_modes_il(path, gt)
    if a_star is None:
        if bank is None:
            raise Ddv2ConfigError("form='matched_anchor' needs `bank` (or a precomputed "
                                  "`a_star`) to match the GT against")
        a_star = matched_anchor_index(bank, gt)
    if n_anchors is None:
        if bank is None:
            raise Ddv2ConfigError("form='matched_anchor' needs `n_anchors` (or `bank`, whose "
                                  "dim 1 is N) — M // N decides which chains are gathered")
        n_anchors = int(bank.shape[1])
    return matched_anchor_il(path, gt, a_star, n_anchors)


def apply_lambda_scale(weights: dict, settings: IlSettings) -> dict:
    """Scale the release's IL weights by ``settings.lambda_scale``. -> a NEW dict.

    ``weights`` is :func:`tanitad.rl.ddv2_rl.step_loss_weights`'s output. Only ``il_coef`` (what
    multiplies the per-step IL scalar) and ``il_weight_b`` (the per-row weights, logged) are
    touched; ``coef_rl`` — the policy gradient — is returned **unchanged and by identity**, so a
    λ arm cannot silently move the RL term.

    ``lambda_scale == 1.0`` returns the input dict object itself, so "the lever is off" is the
    release by construction rather than by a float multiply that happens to be exact.
    """
    if settings.lambda_scale == 1.0:
        return weights
    out = dict(weights)
    s = float(settings.lambda_scale)
    out["il_coef"] = weights["il_coef"] * s
    out["il_weight_b"] = weights["il_weight_b"] * s
    out["coef_rl"] = weights["coef_rl"]          # identity, not a copy
    return out


# =========================================================================== #
# CHANGE 2 — gradient clipping                                                #
# =========================================================================== #
def clip_gradients(params, settings: IlSettings) -> dict:
    """``clip_grad_norm_(params, settings.grad_clip)``. -> the norms, before and after.

    Call ONCE, after every per-step ``backward()`` has accumulated into ``.grad`` and
    immediately before ``optimizer.step()`` — the grad pass is decomposed over the 10 rollout
    steps (``ddv2_rl.per_step_loss``), so clipping inside the loop would clip ten partial
    gradients and would not be "clip the update".

    -> ``grad_norm`` (BEFORE clipping — the diagnostic that caught RL-s1's divergence at 15,712),
    ``grad_norm_clipped`` (after), ``grad_clipped`` (whether it bound), ``grad_clip`` (the
    max-norm, ``None`` when off). With the lever off, ``grad_norm_clipped == grad_norm`` and
    nothing is written to any ``.grad``.
    """
    params = [p for p in params if p is not None and p.grad is not None]
    if not params:
        raise Ddv2ConfigError(
            "no parameter carries a gradient — clip_gradients was called before backward(), or "
            "requires_grad was never set. Refusing rather than reporting a norm of 0.0.")
    clip = settings.grad_clip
    if not clip:                                            # None or 0.0 == the release
        total = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in params))
        return {"grad_norm": float(total), "grad_norm_clipped": float(total),
                "grad_clipped": False, "grad_clip": None}
    total = torch.nn.utils.clip_grad_norm_(params, float(clip))     # returns the PRE-clip norm
    pre = float(total)
    return {"grad_norm": pre, "grad_norm_clipped": min(pre, float(clip)),
            "grad_clipped": pre > float(clip), "grad_clip": float(clip)}


# =========================================================================== #
# telemetry — the collapse must be visible DURING training, not only at the end #
# =========================================================================== #
def fan_endpoint_spread(path: Tensor, n_anchors: int, group: int = 0) -> float:
    """Mean pairwise distance between the N anchors' 6 s ENDPOINTS, within one group. -> metres.

    The same statistic the held-out read calls ``fan_endpoint_spread_m``
    (``ddv2_rl_refcv5.py:cmd_heldout``, ``torch.cdist(ends, ends).mean()``), computed on the RL
    chain so the 93 % collapse the release's IL term produced is visible at the step that
    produces it rather than only in a held-out read after 600 of them.

    ⚠️ NOT the same number as the held-out metric: this is the 10-step RL chain with exploration
    at η = 1, that one is the deployed 2-step sampler at η = 0. It is a within-run canary and
    is reported as one, never as the endpoint.
    """
    b, m = path.shape[0], path.shape[1]
    n = int(n_anchors)
    if n < 2 or m % n:
        raise Ddv2ConfigError(f"M={m} is not a whole number of groups of N={n} (need N >= 2)")
    g = m // n
    if not (0 <= int(group) < g):
        raise Ddv2ConfigError(f"group {group} out of range for G={g}")
    ends = path.detach().reshape(b, g, n, path.shape[2], 2)[:, int(group), :, -1]   # [B, N, 2]
    return float(torch.cdist(ends, ends).mean())


def anchor_match_diagnostics(a_star: Tensor, n_anchors: int) -> dict:
    """How concentrated this batch's matched anchors are — the mode-preservation canary.

    ``modal_frac`` = the fraction of the batch's windows that matched the SAME anchor. If the
    matched-anchor term still collapsed the fan, this says whether the cause is the term or a
    degenerate match (one anchor winning every window), which are different defects.
    """
    a = a_star.to(torch.long).reshape(-1)
    counts = torch.bincount(a, minlength=int(n_anchors))
    return {"n_distinct_anchors_matched": int((counts > 0).sum()),
            "modal_anchor": int(counts.argmax()),
            "modal_frac": float(counts.max()) / max(1, a.numel())}


def _check_path_gt(path: Tensor, gt: Tensor) -> None:
    if path.dim() != 4 or path.shape[-1] != 2:
        raise Ddv2ConfigError(f"path must be [B, M, S, 2], got {tuple(path.shape)}")
    if gt.dim() != 3 or gt.shape[0] != path.shape[0] or gt.shape[1:] != path.shape[2:]:
        raise Ddv2ConfigError(f"gt {tuple(gt.shape)} must be [B, S, 2] matching path "
                              f"{tuple(path.shape)}")
