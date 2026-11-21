"""The >=GT positive mask, proven ON THE GRADIENT PATH — not merely importable.

⛔ WHY THIS FILE EXISTS, AND WHY THE EXISTING `test_rl_v2_faithful.py` WAS NOT ENOUGH
------------------------------------------------------------------------------------
`test_rl_v2_faithful.py` pins what the mask COMPUTES. It does not pin that the mask
reaches a parameter's ``.grad``. Those are different claims, and the programme has
already paid for confusing them: `tac_goal_tok_head` was **11,286 parameters that
received `grad_abs_sum` exactly 0.00000 for all 40,284 steps** — built, tested,
rollable, and never reached by the trainer's forward. ⭐ *Rollable and trained are
different claims.* So the first section here MUTATES the bar and observes the
gradient move, which is the only evidence that settles it.

TWO REAL DEFECTS ARE PINNED HERE, both MEASURED 2026-09-10:

1. ``frac_above_bar`` DISAGREED WITH THE MASK IT REPORTS ON. The admission
   predicate was written TWICE — the mask used ``bar_eps`` 1e-6 (DDv2's released
   slack, `diffusiondrivev2_model_rl.py:891-893`) and the diagnostic re-derived it
   with the module's ``EPS``, 1e-8. On a reward sitting at ``bar - 1e-7`` the mask
   admitted **1.0** of the anchors while ``frac_above_bar`` reported **0.0**.
   ⚠️ The direction is what makes it serious: that number's ONLY job is to make a
   bar that zeroed the batch VISIBLE rather than silent, and it failed toward
   "everything was zeroed" while nothing had been. Fixed by a single shared
   derivation, `advantage.gt_bar_mask`.

2. THE BAR AND THE CANDIDATES COULD BE SCORED UNDER DIFFERENT REWARDS. The bar is
   built by the caller's ``sample_fn`` from ``cfg.reward_weights``; the candidates
   are scored inside ``run_posttrain`` under ``spec``. Nothing required them to
   agree. ⚠️ EVIDENCE CLASS: **LATENT** — MEASURED that no caller in the tree
   passes ``spec=`` today — so this is a hole closed before it opened, not a live
   bug, and it is reported that way.

⛔ HOW THE EXPECTATIONS ARE WRITTEN. Every one is a **literal** or an **analytic
identity**, never an expression over the code under test. *"A check that shares the
defect it checks for is green forever"* — re-running the producer's own derivation
and finding agreement measures determinism, not correctness. The two analytic
identities used below need no reward-function knowledge at all:

  * a bar BELOW every achievable reward is the identity mask ⇒ the gradient must
    equal the no-bar gradient **exactly**;
  * a bar ABOVE every achievable reward admits nothing ⇒ with ``w_intra = 0`` the
    loss loses every dependence on ``logp`` ⇒ the gradient must be **exactly 0.0**.

Each section ships a DELIBERATE-REGRESSION arm that reintroduces the real historical
defect and asserts the discriminator fires on it — so the test is provably not
vacuous.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tanitad.rl import PostTrainConfig, RewardSpec
from tanitad.rl import advantage as A
from tanitad.rl.posttrain import rl_objective, run_posttrain


# --------------------------------------------------------------------------- #
# a policy whose sampled fan depends on ONE leaf scalar, so `.grad` is readable  #
# --------------------------------------------------------------------------- #
B, N, G, S = 2, 3, 4, 5
_SCALE = torch.tensor([0.4, 0.8, 1.2]).view(1, N, 1, 1)      # per-anchor extent
_JITTER = torch.linspace(0.9, 1.1, G).view(1, 1, G, 1)       # per-sample spread


def _objective_grad(*, use_bar: bool, bar_value: float = 0.0,
                    w_intra: float = 0.0):
    """Run one objective on a fan that depends on ``theta``; return (grad, out).

    ``w_intra = 0`` isolates the INTER-anchor half, which is the only half the
    >=GT mask touches (`_model_rl.py:891-893`). With the intra term live the
    gradient would move for reasons unrelated to the bar.
    """
    theta = torch.zeros(1, requires_grad=True)
    x = (torch.linspace(0, 8, S).view(1, 1, 1, S) * _SCALE * _JITTER)
    x = x.expand(B, N, G, S) + theta
    traj = torch.stack([x, torch.zeros_like(x)], dim=-1)      # [B,N,G,S,2]
    logp = -0.5 * traj.pow(2).sum(dim=(-1, -2))               # [B,N,G], carries theta
    ctx = {"dt": 0.5, "v0": torch.full((B, 1, 1), 4.0)}
    cfg = PostTrainConfig(method="grpo", group_size=G, normalize="none",
                          use_gt_bar=use_bar, w_intra=w_intra, w_inter=1.0,
                          reward_weights={"progress": 1.0}, dt=0.5,
                          veto_enabled=False, w_anchor=0.0)
    spec = RewardSpec(weights={"progress": 1.0}, dt=0.5)
    kw = {"gt_bar": torch.full((B,), float(bar_value))} if use_bar else {}
    out = rl_objective(traj, logp, ctx, cfg, spec, **kw)
    out["loss"].backward()
    return float(theta.grad), out


# --------------------------------------------------------------------------- #
# 1. THE GRADIENT-PATH PROOF — mutate the bar, watch `.grad` move               #
# --------------------------------------------------------------------------- #
def test_the_instrument_carries_signal_before_it_is_used_to_conclude_anything():
    """⛔ THE CONTROL THAT MUST READ A KNOWN VALUE, AND IT CAUGHT A REAL MISTAKE.

    The first draft of this rig gave every anchor the SAME forward extent, so every
    per-anchor reward was identical, the centred advantage was identically 0, and
    `.grad` read 0.0 for the bar, against the bar, and with no bar at all. A probe
    that reads the no-information value everywhere proves nothing about the thing
    it is aimed at — so the rig must be shown to carry signal FIRST.

    The literals are analytic: `progress` is along-track over ``v0 * horizon``, and
    the fan is built so anchor *i* travels exactly ``8 * scale[i]`` metres against a
    ``4.0 m/s * 2 s = 8 m`` reference ⇒ the per-anchor rewards are exactly the
    scales, 0.4 / 0.8 / 1.2.
    """
    _, out = _objective_grad(use_bar=False)
    per_anchor = out["reward"].mean(-1)[0]
    assert per_anchor.tolist() == pytest.approx([0.4, 0.8, 1.2], abs=1e-6)
    # mean 0.8 -> centred [-0.4, 0.0, +0.4] -> clamp_min(0) -> [0, 0, 0.4]
    assert out["adv_inter"][0].tolist() == pytest.approx([0.0, 0.0, 0.4], abs=1e-6)


def test_an_impossible_bar_ZEROES_the_gradient_and_a_permissive_one_does_not():
    """⭐ THE LOAD-BEARING TEST. Only the bar moves; `.grad` must move with it."""
    g_none, _ = _objective_grad(use_bar=False)
    g_all, o_all = _objective_grad(use_bar=True, bar_value=-1e9)
    g_none_admitted, o_no = _objective_grad(use_bar=True, bar_value=1e9)

    # the rig is live (guarded by the control above, restated as a literal here)
    assert g_none == pytest.approx(3.2000008, abs=1e-5)
    assert g_none != 0.0

    # ANALYTIC: a bar below every achievable reward is the identity mask.
    assert g_all == g_none, "an admit-all bar must be a no-op on the gradient"
    assert float(o_all["frac_above_bar"]) == 1.0

    # ANALYTIC: a bar above every achievable reward admits nothing; with the
    # intra half off, the loss keeps no dependence on logp at all.
    assert g_none_admitted == 0.0, (
        "a bar nothing clears must zero the policy gradient — if it does not, "
        "the mask is not on the gradient path")
    assert float(o_no["frac_above_bar"]) == 0.0


def test_the_mask_is_reached_through_the_objective_not_only_the_helper():
    """The bar must change `adv_inter` as the OBJECTIVE assembles it, so that a
    future refactor which stops threading `gt_bar` into `composite_advantage` is
    caught here rather than in a 40 k-step run."""
    _, permissive = _objective_grad(use_bar=True, bar_value=0.5)
    _, strict = _objective_grad(use_bar=True, bar_value=1.0000001)
    assert permissive["adv_inter"][0].tolist() == pytest.approx([0.0, 0.0, 0.4], abs=1e-6)
    # only the 1.2 anchor carried positive advantage, and 1.2 > 1.0000001 still
    # clears; push the bar past it and the inter half must be exactly empty
    _, above_all = _objective_grad(use_bar=True, bar_value=1.5)
    assert above_all["adv_inter"][0].tolist() == [0.0, 0.0, 0.0]


def test_DELIBERATE_REGRESSION_a_detached_advantage_would_still_pass_a_value_test():
    """⛔ THE REGRESSION ARM FOR SECTION 1.

    Reintroduce the historical failure this section exists to catch — an advantage
    that is correct in VALUE but disconnected from the gradient — and assert the
    section's discriminator fires. A value-only test (the whole of
    `test_rl_v2_faithful.py`) is GREEN against this; the gradient assertion is not.
    """
    theta = torch.zeros(1, requires_grad=True)
    x = (torch.linspace(0, 8, S).view(1, 1, 1, S) * _SCALE * _JITTER)
    x = x.expand(B, N, G, S) + theta
    traj = torch.stack([x, torch.zeros_like(x)], dim=-1)
    # ⛔ THE DEFECT: logp detached from the policy. Every advantage VALUE below is
    # bit-identical to the honest arm's; only the gradient path is gone.
    logp = (-0.5 * traj.pow(2).sum(dim=(-1, -2))).detach()
    ctx = {"dt": 0.5, "v0": torch.full((B, 1, 1), 4.0)}
    cfg = PostTrainConfig(method="grpo", group_size=G, normalize="none",
                          use_gt_bar=True, w_intra=0.0, w_inter=1.0,
                          reward_weights={"progress": 1.0}, dt=0.5,
                          veto_enabled=False, w_anchor=0.0)
    spec = RewardSpec(weights={"progress": 1.0}, dt=0.5)
    out = rl_objective(traj, logp, ctx, cfg, spec, gt_bar=torch.full((B,), -1e9))

    # the VALUE test still passes -- this is exactly why it is not sufficient
    assert out["adv_inter"][0].tolist() == pytest.approx([0.0, 0.0, 0.4], abs=1e-6)
    assert float(out["frac_above_bar"]) == 1.0
    # ...and the GRADIENT assertion catches it outright: with the intra half off
    # and logp detached, the loss keeps NO connection to the policy, so it has no
    # grad_fn at all. MEASURED: torch refuses the backward rather than returning
    # a zero — an even sharper discriminator than `grad == 0`.
    assert out["loss"].requires_grad is False
    with pytest.raises(RuntimeError, match="does not require grad"):
        out["loss"].backward()
    assert theta.grad is None


# --------------------------------------------------------------------------- #
# 2. `frac_above_bar` IS THE MASK'S OWN ADMISSION SET                            #
# --------------------------------------------------------------------------- #
#: The exact reward that sat in the two-epsilon disagreement band, MEASURED
#: 2026-09-10. Written as a literal so this file does not re-derive the bug.
_BAND_REWARD = 1.0 - 1e-7
_BAND_BAR = 1.0


def test_frac_above_bar_reports_what_the_mask_actually_admitted():
    """⭐ THE MEASURED DEFECT. A reward at ``bar - 1e-7`` clears DDv2's 1e-6 slack.

    Literals, both analytic: the mask's predicate is ``r > bar - 1e-6`` and
    ``1.0 - 1e-7 > 1.0 - 1e-6`` is TRUE, so every anchor is admitted and the
    reported fraction must be exactly 1.0. Before the fix this read 0.0.
    """
    bar = torch.tensor([_BAND_BAR], dtype=torch.float64)
    reward_ig = torch.full((1, 2, 4), _BAND_REWARD, dtype=torch.float64)
    out = A.composite_advantage(reward_ig, gt_bar=bar)

    assert float(out["frac_above_bar"]) == 1.0, (
        "frac_above_bar must report the fraction the MASK admitted; it read 0.0 "
        "while the mask admitted everything (the 2026-09-10 two-epsilon defect)")

    # and it agrees with the mask applied directly, on the same inputs
    per_anchor = out["per_anchor_reward"]
    admitted = A.gt_bar_mask(per_anchor, bar).to(torch.float64).mean()
    assert float(admitted) == 1.0


def test_DELIBERATE_REGRESSION_the_two_epsilon_form_disagrees_and_is_caught():
    """⛔ THE REGRESSION ARM FOR SECTION 2 — the historical code, verbatim.

    ``(per_anchor > (bar - EPS))`` with ``EPS = 1e-8`` is what the diagnostic used
    to compute. Asserting that it returns 0.0 on the same input the mask admits at
    1.0 proves the test above discriminates rather than restating the code.
    """
    bar = torch.tensor([_BAND_BAR], dtype=torch.float64)
    per_anchor = torch.full((1, 2), _BAND_REWARD, dtype=torch.float64)

    legacy_eps = 1e-8                                   # the defect's constant
    legacy = (per_anchor > (bar.unsqueeze(-1) - legacy_eps)).to(torch.float64).mean()
    assert float(legacy) == 0.0, "the historical form must read 0.0 here"

    current = A.gt_bar_mask(per_anchor, bar).to(torch.float64).mean()
    assert float(current) == 1.0
    assert float(legacy) != float(current), (
        "if these ever agree, this band no longer discriminates and the test "
        "above has stopped being evidence — pick a new band, do not delete this")


def test_BAR_EPS_is_the_released_constant_and_is_named_once():
    """A literal, from `diffusiondrivev2_model_rl.py:891-893`."""
    assert A.BAR_EPS == 1e-6


def test_bar_eps_threads_from_composite_into_the_mask():
    """A caller that changes the slack must change BOTH the mask and the report —
    that is what a single derivation buys, and it is asserted rather than assumed."""
    bar = torch.tensor([1.0], dtype=torch.float64)
    reward_ig = torch.full((1, 2, 4), 1.0 - 1e-7, dtype=torch.float64)
    tight = A.composite_advantage(reward_ig, gt_bar=bar, bar_eps=1e-9)
    assert float(tight["frac_above_bar"]) == 0.0     # 1e-7 below now fails the bar
    loose = A.composite_advantage(reward_ig, gt_bar=bar, bar_eps=1e-6)
    assert float(loose["frac_above_bar"]) == 1.0


# --------------------------------------------------------------------------- #
# 3. THE BAR AND THE CANDIDATES ARE SCORED UNDER THE SAME REWARD                 #
# --------------------------------------------------------------------------- #
class _Unused(nn.Module):
    """A stand-in policy shaped like RefCV3Model (`core.decoder`), so the loop's
    own `trainable_prefixes` check passes and the SPEC guard is what we measure."""

    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.decoder = nn.Linear(2, 2)


def _cfg_for_spec_guard(**kw):
    return PostTrainConfig(method="grpo", group_size=4, steps=1, lr=0.0,
                           use_gt_bar=True, reward_weights={"progress": 1.0},
                           dt=0.5, veto_enabled=False, w_anchor=0.0, **kw)


def test_run_posttrain_REFUSES_a_spec_that_disagrees_with_the_bars_config():
    """⚠️ LATENT, not live: MEASURED 2026-09-10 that no caller passes ``spec=``.

    The bar is scored from ``cfg.reward_weights`` in the caller's ``sample_fn``;
    the candidates are scored under ``spec`` here. V2's mask is a threshold on ONE
    reward function, so two different ones make it meaningless — silently, with the
    loop still exiting 0.
    """
    def never_called(batch, cfg):                      # pragma: no cover
        raise AssertionError("the guard must fire before any sampling")

    cfg = _cfg_for_spec_guard()
    foreign = RewardSpec(weights={"progress": 0.5}, dt=0.5)   # different weight
    with pytest.raises(ValueError, match="use_gt_bar=True with an explicit spec"):
        run_posttrain(_Unused(), never_called, cfg, spec=foreign, batches=[])

    foreign_dt = RewardSpec(weights={"progress": 1.0}, dt=0.25)  # different dt
    with pytest.raises(ValueError, match="use_gt_bar=True with an explicit spec"):
        run_posttrain(_Unused(), never_called, cfg, spec=foreign_dt, batches=[])


def test_the_matching_spec_is_ACCEPTED_so_the_guard_is_not_just_a_blanket_refusal():
    """⛔ THE CONTROL. A guard that refuses everything passes the test above while
    breaking every legitimate caller — so the accepting direction is asserted too."""
    cfg = _cfg_for_spec_guard()
    same = RewardSpec(weights={"progress": 1.0}, dt=0.5)
    # zero batches: the loop's own false-green guard is what stops this, NOT the
    # spec guard — so reaching that error proves the spec check let us through.
    with pytest.raises(RuntimeError, match="FALSE-GREEN REFUSED"):
        run_posttrain(_Unused(), lambda b, c: None, cfg, spec=same, batches=[])


def test_the_guard_does_not_fire_when_the_bar_is_off():
    """Without ``use_gt_bar`` there is no bar to be commensurable with, so an
    explicit spec is a legitimate override and must not be refused."""
    cfg = PostTrainConfig(method="grpo", group_size=4, steps=1, lr=0.0,
                          use_gt_bar=False, reward_weights={"progress": 1.0},
                          dt=0.5, veto_enabled=False, w_anchor=0.0)
    foreign = RewardSpec(weights={"progress": 0.5}, dt=0.25)
    with pytest.raises(RuntimeError, match="FALSE-GREEN REFUSED"):
        run_posttrain(_Unused(), lambda b, c: None, cfg, spec=foreign, batches=[])
