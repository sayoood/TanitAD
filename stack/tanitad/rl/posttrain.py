"""The RL post-training loop — planner head only, frozen trunk, done-marker.

⛔ THREE HOUSE RULES THIS LOOP IMPLEMENTS RATHER THAN DOCUMENTS
---------------------------------------------------------------
1. **COUNTERS, ASSERTED NON-ZERO.** Every phase increments a counter, and
   `run_posttrain` REFUSES to write a success done-marker when any required
   counter is zero. *An exit 0 over an empty set is the false-green class:* a
   smoke that "passed" having scored nothing looks identical to one that
   worked. `SmokeCounters.assert_nonzero()` is the difference.
2. **DONE-MARKER IN THE SAME TURN.** `summary.json` with `done: true` is
   written by the same call that finishes the run. A supervised run whose
   marker never lands is resurrected forever (measured: 2 days of relaunches).
3. **FROZEN TRUNK BY DEFAULT.** `freeze_trunk=True` is the mandatory option.
   The parameters that move are named by prefix and COUNTED into the run
   record, so "we fine-tuned only the head" is a verifiable claim, not a
   sentence in a report.

⚠️ WHAT THIS LOOP DELIBERATELY DOES **NOT** DO
-----------------------------------------------
It does not run a real training job in this deliverable, and it does not
pretend a rule-based reward is a driving result. Any number produced here is
**T0 training-side**; a capability claim needs T1 (`EVAL_DOCTRINE.md`). The
loop's job is to be correct, counted and resumable so that when a real run is
authorised, the run record answers "what exactly ran".

THE SAMPLING CONTRACT
---------------------
`sample_fn(batch, cfg) -> (traj, logp, ctx)` where
  * ``traj`` ``[B, N, G, S, 2]`` — G exploration samples per anchor,
  * ``logp`` ``[B, N, G]`` — log-prob of each sample under the current policy,
  * ``ctx``  the reward context (scene facts only — `assert_selector_disjoint`
    is called on it every step).
Keeping sampling injectable is what lets the smoke run with a synthetic policy
on CPU while the real path plugs in refcv3's diffusion decoder unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from . import advantage as A
from . import anchor as ANC
from . import audit as AUD
from . import rewards as R
from .config import PostTrainConfig


@dataclass
class SmokeCounters:
    """Counters proving each phase did work over a NON-EMPTY set."""
    steps: int = 0
    windows_scored: int = 0
    samples_scored: int = 0
    reward_evals: int = 0
    advantage_evals: int = 0
    optimizer_steps: int = 0
    components_fired: dict[str, int] = field(default_factory=dict)

    #: counters that MUST be > 0 for a run to be called successful
    REQUIRED = ("steps", "windows_scored", "samples_scored",
                "reward_evals", "advantage_evals", "optimizer_steps")

    def assert_nonzero(self) -> None:
        zero = [k for k in self.REQUIRED if getattr(self, k) == 0]
        if zero:
            raise RuntimeError(
                f"FALSE-GREEN REFUSED: counters {zero} are ZERO — the loop "
                "completed without doing the work they count. An exit 0 over an "
                "empty set is indistinguishable from a real pass, which is "
                "exactly why this check exists. Nothing was written.")
        dead = [k for k, v in self.components_fired.items() if v == 0]
        if dead:
            # not fatal: a corpus may genuinely have no lead vehicle. But it is
            # REPORTED, never silent — a component that never fires carries no
            # signal and must not be mistaken for a safeguard.
            self.components_fired["_never_fired"] = dead  # type: ignore[assignment]

    def to_dict(self) -> dict:
        return {"steps": self.steps, "windows_scored": self.windows_scored,
                "samples_scored": self.samples_scored,
                "reward_evals": self.reward_evals,
                "advantage_evals": self.advantage_evals,
                "optimizer_steps": self.optimizer_steps,
                "components_fired": self.components_fired}


def select_trainable(model, cfg: PostTrainConfig) -> dict:
    """Freeze everything, then unfreeze only the named planner-head prefixes.

    Returns a report with the COUNTS, so the frozen-trunk claim is checkable.
    ⛔ Refuses if nothing became trainable — an optimiser over an empty
    parameter list runs happily and changes nothing.
    """
    forbidden = tuple(getattr(cfg, "forbidden_prefixes", ()) or ())
    excluded_pfx = tuple(getattr(cfg, "exclude_prefixes", ()) or ())
    total = trainable = 0
    names: list[str] = []
    leaked: list[str] = []
    excluded: list[str] = []
    for name, p in model.named_parameters():
        total += p.numel()
        want = (not cfg.freeze_trunk) or any(
            name.startswith(pfx) or f".{pfx}" in name
            for pfx in cfg.trainable_prefixes)
        # exclude runs FIRST (normal path, recorded); forbidden stays the
        # tripwire that fires only if an exclusion is removed by mistake.
        if want and any(name.startswith(e) or f".{e}" in name
                        for e in excluded_pfx):
            excluded.append(name)
            want = False
        if want and any(name.startswith(f) or f".{f}" in name for f in forbidden):
            leaked.append(name)
            want = False
        p.requires_grad_(bool(want))
        if want:
            trainable += p.numel()
            names.append(name)
    if leaked:
        raise RuntimeError(
            f"FORBIDDEN PARAMETERS WOULD HAVE BEEN TRAINED: {leaked[:8]} match "
            f"forbidden_prefixes={forbidden}. This library post-trains the "
            "GENERATOR (the diffusion decoder), never the SELECTOR — training "
            "the selector under a reward designed to avoid selector "
            "over-reliance is the defect, not the fix.")
    if trainable == 0:
        raise RuntimeError(
            "no parameters are trainable after applying "
            f"trainable_prefixes={cfg.trainable_prefixes}. An optimiser over an "
            "empty parameter list steps happily and changes NOTHING — refusing.")
    return {"total_params": total, "trainable_params": trainable,
            "frozen_params": total - trainable,
            "trainable_fraction": trainable / max(total, 1),
            "n_trainable_tensors": len(names),
            "trainable_names_head": names[:12],
            "trainable_prefixes": list(cfg.trainable_prefixes),
            "forbidden_prefixes": list(forbidden),
            "exclude_prefixes": list(excluded_pfx),
            "excluded_names": excluded,
            "freeze_trunk": cfg.freeze_trunk}


def apply_exploration_noise(offset: Tensor, cfg: PostTrainConfig,
                            gen: torch.Generator | None = None) -> Tensor:
    """DDv2-style exploration noise on the decoder's offset.

    MULTIPLICATIVE (the default) scales with the offset magnitude, so small
    offsets are perturbed proportionally instead of being swamped — DDv2's
    ablation measured 90.1 vs 89.7 PDMS for multiplicative over additive.
    """
    if cfg.noise_mode == "two_scalar":
        # V2's released sampler: ONE scalar per axis per trajectory, broadcast
        # over the waypoints (see refcv3_adapter.sample_offsets for the pin).
        shape = (*offset.shape[:-2], 1, offset.shape[-1])
        eps = torch.randn(shape, generator=gen, device=offset.device,
                          dtype=offset.dtype).expand(offset.shape)
        return offset * (1.0 + cfg.noise_scale * eps)
    eps = torch.randn(offset.shape, generator=gen, device=offset.device,
                      dtype=offset.dtype)
    if cfg.noise_mode == "multiplicative":
        return offset * (1.0 + cfg.noise_scale * eps)
    return offset + cfg.noise_scale * eps


def veto_mask(traj: Tensor, ctx: dict, cfg: PostTrainConfig) -> Tensor:
    """The hard-constraint mask over ``traj [..., S, 2]`` -> bool, EXPLICITLY keyed.

    ⛔⛔ WHY THIS FUNCTION EXISTS, AND WHY THE OLD TWO LINES WERE A DEFECT.
    The veto used to be composed as::

        if "collision" in spec.weights:            # KEY membership: TRUE at 0.0
            veto = COMPONENTS["collision"](traj, ctx) < 0
        ttc = ttc_violation(...)                   # read no config at all
        veto = ttc if veto is None else (veto | ttc)

    so a "constant reward" control with every weight at 0.0 still carried the FULL
    veto. Because `advantage.truncated_inter_anchor_advantage` pins vetoed
    candidates at `veto_value` OUTSIDE the group-relative centring, a constant
    reward yields a VETO-ONLY advantage at full strength rather than a zero one.
    MEASURED on `ctrl_const` (2026-09-05): `veto_rate_mean` **0.0897**,
    `final_loss` **-1.863**, 14 of 57 fan-safety metrics moved — a control that
    could not be a null, and a result that was read as a harness fault before the
    mechanism was found.

    => The channels are now three booleans somebody has to TYPE, and `to_dict()`
    records them, so a run's own record says whether the constraint was in force.
    Returns an all-False mask (never ``None``) when the veto is off, so the caller
    always has a `veto_rate` to log — a channel that reports 0.0 is evidence; a
    channel that reports nothing is not.

    ⭐ This also makes VETO-ONLY a first-class arm rather than an accident: zero
    reward weights with `veto_enabled=True` is DDv2's constraint mechanism with no
    ranking term at all, which is the only half of the stage that improved
    refcv3's fan feasibility.
    """
    off = torch.zeros(traj.shape[:-2], dtype=torch.bool, device=traj.device)
    if not cfg.veto_enabled:
        return off
    veto = off
    if cfg.veto_collision:
        veto = veto | (R.COMPONENTS["collision"](traj, ctx) < 0)
    if cfg.veto_ttc:
        veto = veto | R.ttc_violation(traj, {**ctx, "ttc_min_s": cfg.ttc_min_s})
    return veto


def rl_objective(traj: Tensor, logp: Tensor, ctx: dict, cfg: PostTrainConfig,
                 spec: R.RewardSpec, *, imitation_loss: Tensor | None = None,
                 anchor_pair: tuple[Tensor, Tensor] | None = None,
                 gt_bar: Tensor | None = None
                 ) -> dict[str, Tensor]:
    """One RL objective evaluation over ``traj [B, N, G, S, 2]``.

    Returns the loss and every part of it, because an aggregate that hides which
    half moved makes an ablation unattributable.

    ``gt_bar`` ``[B]`` — the GT trajectory's reward under ``spec`` on the same
    window (V2's >=GT positive mask, `advantage.truncated_inter_anchor_advantage`).
    Required iff ``cfg.use_gt_bar``; refused otherwise, in both directions.
    """
    AUD.assert_selector_disjoint(ctx)
    reward = spec(traj, ctx)                                    # [B, N, G]
    if reward.shape != logp.shape:
        raise ValueError(f"reward {tuple(reward.shape)} vs logp "
                         f"{tuple(logp.shape)} must match")
    use_bar = bool(getattr(cfg, "use_gt_bar", False))
    if use_bar and gt_bar is None:
        raise ValueError(
            "use_gt_bar=True but no gt_bar was handed to rl_objective. A bar that "
            "is configured and never applied is a silent no-op that still exits 0 "
            "— refusing. Score the GT trajectory under the same RewardSpec and pass "
            "it as gt_bar=[B].")
    if gt_bar is not None and not use_bar:
        raise ValueError("gt_bar was supplied but cfg.use_gt_bar is False — the run "
                         "record would not say the bar was in force; set use_gt_bar")
    if gt_bar is not None and tuple(gt_bar.shape) != tuple(reward.shape[:-2]):
        raise ValueError(f"gt_bar {tuple(gt_bar.shape)} must be per-window "
                         f"{tuple(reward.shape[:-2])}")
    # ⛔ THE VETO CHANNEL — constraints, not ranking terms. Collision OR
    # TTC-imminent. Applied outside the group-relative centring so a vetoed
    # candidate is PINNED, never merely ranked lower. ⭐ Keyed on the
    # CONFIG (`veto_enabled`/`veto_collision`/`veto_ttc`), NEVER on
    # `spec.weights` — see `veto_mask` and config.py's `veto_enabled` note.
    veto = veto_mask(traj, ctx, cfg)

    if cfg.method == "awr":
        # advantage-weighted regression: weight the imitation loss by exp(A/beta)
        adv = A.grpo_advantage(reward, normalize=cfg.normalize)
        w = torch.exp(adv / max(cfg.awr_beta, 1e-6)).detach()
        if imitation_loss is None:
            raise ValueError("awr requires an imitation_loss to weight")
        loss = (w * imitation_loss).mean()
        return {"loss": loss, "reward": reward, "advantage": adv,
                "awr_weight": w}

    parts = A.composite_advantage(reward, veto_ig=veto,
                                  w_intra=cfg.w_intra, w_inter=cfg.w_inter,
                                  normalize=cfg.normalize,
                                  gt_bar=(gt_bar.detach() if gt_bar is not None else None))
    pg = A.policy_gradient_loss(logp, parts["total"], kl_coef=cfg.kl_coef)
    loss = pg["loss"]
    if imitation_loss is not None:
        loss = loss + cfg.w_imitation * imitation_loss.mean()

    # ⭐ THE TRUST REGION — applied to the LOSS, never inside the advantage.
    # Same placement rule as the veto: it CONSTRAINS the policy rather than
    # ranking candidates. Inside the group-relative advantage it would become a
    # fan-collapse objective (see rewards.DEFAULT_WEIGHTS on gt_similarity).
    out = {"loss": loss, "pg": pg["pg"], "reward": reward,
           "advantage": parts["total"], "adv_intra": parts["intra"],
           "adv_inter": parts["inter"],
           # the veto channel's firing rate — how often the constraint had
           # anything to constrain. A reward whose veto never fires cannot
           # push mass away from collision (the re-scope's primary question).
           "veto_rate": veto.to(reward.dtype).mean().detach()}
    if "frac_above_bar" in parts:
        out["frac_above_bar"] = parts["frac_above_bar"]
    if cfg.w_anchor > 0.0:
        if anchor_pair is None:
            raise ValueError(
                f"w_anchor={cfg.w_anchor} requires anchor_pair=(live_mean_fan, "
                "ref_mean_fan) on the SAME inputs. Refusing to run a trust "
                "region with no reference — that is the unanchored "
                "configuration that drifted +69.5 % in P-RC21.")
        # ⛔ THE TRUST REGION CONSTRAINS THE POLICY'S MEAN, NOT ITS SAMPLES.
        # Penalising the drawn samples would also penalise the exploration
        # noise, whose scale is |offset|*sigma under the multiplicative form —
        # i.e. it would SHRINK offsets rather than HOLD POSITION. Those are
        # different objectives, and the shrinkage one is not a trust region.
        live_mean, ref_mean = anchor_pair
        anc = ANC.anchor_penalty(live_mean, ref_mean, w=cfg.w_anchor,
                                 form=cfg.anchor_form)
        out["loss"] = out["loss"] + anc["penalty"]
        out["anchor_penalty"] = anc["penalty"]
        out["anchor_divergence"] = anc["divergence"]
    return out


def run_posttrain(model, sample_fn: Callable, cfg: PostTrainConfig, *,
                  spec: R.RewardSpec | None = None,
                  batches=None, device: str = "cpu") -> dict:
    """Run the loop and write `config.json` + `summary.json` (done-marker).

    ``batches`` is any iterable of batch objects handed straight to
    ``sample_fn``; the loop is agnostic to what a batch is so the smoke can pass
    integers and the real path can pass windows.
    """
    cfg.validate()
    spec = spec or R.RewardSpec(weights=dict(cfg.reward_weights), dt=cfg.dt)
    torch.manual_seed(cfg.seed)

    # ⛔ Set the forward mode EXPLICITLY and record it. Inheriting
    # nn.Module's training=True default is TRAIN-C5.
    model.train(bool(getattr(cfg, "train_mode_forward", False)))
    freeze_report = select_trainable(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    counters = SmokeCounters()

    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        rec = {**cfg.to_dict(), "freeze_report": freeze_report,
               "reward_weights_in_force": dict(spec.weights)}
        with open(os.path.join(cfg.out_dir, "config.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)

    src = batches if batches is not None else range(cfg.steps)
    history: list[dict] = []
    for step, batch in enumerate(src, start=1):
        got = sample_fn(batch, cfg)
        # sample_fn may return (traj, logp, ctx) or (traj, logp, ctx, extras);
        # `extras` carries anchor_pair when a trust region is configured.
        traj, logp, ctx = got[0], got[1], got[2]
        extras = got[3] if len(got) > 3 else {}
        out = rl_objective(traj, logp, ctx, cfg, spec,
                           anchor_pair=extras.get("anchor_pair"),
                           gt_bar=extras.get("gt_bar"))

        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()

        counters.steps += 1
        counters.optimizer_steps += 1
        counters.reward_evals += 1
        counters.advantage_evals += 1
        counters.windows_scored += int(traj.shape[0])
        counters.samples_scored += int(traj[..., 0, 0].numel())
        for name, info in AUD.report_component_coverage(spec, traj, ctx).items():
            counters.components_fired.setdefault(name, 0)
            counters.components_fired[name] += int(bool(info["fired"]))

        h = {"step": step, "loss": float(out["loss"].detach()),
             "reward_mean": float(out["reward"].detach().mean()),
             "veto_rate": float(out["veto_rate"])}
        if "frac_above_bar" in out:
            h["frac_above_bar"] = float(out["frac_above_bar"])
        history.append(h)
        if step >= cfg.steps:
            break

    counters.assert_nonzero()

    audit = AUD.audit_reward(spec)
    summary = {
        "done": True,
        "run": cfg.run_name,
        "method": cfg.method,
        "steps": counters.steps,
        "counters": counters.to_dict(),
        "freeze_report": freeze_report,
        "train_mode_forward": bool(getattr(cfg, "train_mode_forward", False)),
        "reward_audit": {"verdict": audit.verdict, "flagged": audit.flagged,
                         "inconclusive": audit.inconclusive,
                         "dead_components": audit.dead_components,
                         "reason": audit.reason,
                         "reference": audit.reference, "scores": audit.scores},
        "final_loss": history[-1]["loss"] if history else None,
        # the two facts the RE-SCOPE (2026-09-05) needs from every run record:
        # how often the veto channel had anything to pin, and how often V2's
        # >=GT bar let a positive advantage through at all.
        "veto_rate_mean": (float(sum(h["veto_rate"] for h in history) / len(history))
                           if history else None),
        "frac_above_bar_mean": (float(sum(h["frac_above_bar"] for h in history) / len(history))
                                if history and "frac_above_bar" in history[0] else None),
        "use_gt_bar": bool(getattr(cfg, "use_gt_bar", False)),
        "noise_mode": cfg.noise_mode,
        "_tier": "T0 training-side; capability claims are T1 only",
        "_evidence_class": "MEASURED (ours)",
    }
    if cfg.out_dir:
        with open(os.path.join(cfg.out_dir, "summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
    summary["history"] = history
    return summary
