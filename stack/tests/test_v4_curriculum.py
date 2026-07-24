"""flagship v4 P4 curriculum primitives — tanitad/train/v4_curriculum.py.

Pins the correctness-critical, GPU-free half of the joint trainer:

* the O-17 invariant — with the default phases the G1 gate (step 10,000) falls in
  Phase C, so the arm is scored on the produced-goal / own-selected configuration
  it has actually trained under (the old ramps ended at 20,000 and put the gate
  inside the ramp: a near-certain restart for a schedule reason, cap 2);
* λ_plan's three-phase values and the ``0``/``1`` reproduction modes;
* the plan-smoothness penalty needs and rewards a DENSE plan;
* the factorised CEs mask their ``unknown`` sentinels (a masked-out logit trains
  on nothing, never a dead-parameter shortcut);
* the canary is a CONTROLLER — it only ratchets λ_plan DOWN and never kills.
"""

from __future__ import annotations

import pytest
import torch

from tanitad.train.v4_curriculum import (IGNORE_INDEX, CanaryController,
                                         CurriculumPhases, factorised_ce,
                                         gate_is_after_ramp, lambda_plan_at,
                                         plan_smoothness_loss, produced_goal_frac)

PHASES = CurriculumPhases()          # 2000 / 8000
GATE_STEP = 10000


# ------------------------------------------------- the O-17 invariant -------
def test_gate_falls_in_phase_c_not_inside_a_ramp():
    """O-17: the whole point. At the pre-registered gate the ramps are DONE."""
    assert gate_is_after_ramp(GATE_STEP, PHASES)
    assert produced_goal_frac(GATE_STEP, PHASES) == 1.0     # fully produced goal
    assert lambda_plan_at(GATE_STEP, PHASES) == 1.0         # full joint training
    # ... and it has had a real amount of training in that configuration
    assert GATE_STEP - PHASES.phase_b == 2000


def test_the_old_straddling_schedule_would_have_failed_the_invariant():
    """The regression guard: the pre-O-17 ramp ended at 20,000, i.e. AFTER the
    gate. Encoded so a future edit that re-introduces it fails loudly."""
    straddling = CurriculumPhases(phase_a=10000, phase_b=20000)
    assert not gate_is_after_ramp(GATE_STEP, straddling)
    assert produced_goal_frac(GATE_STEP, straddling) == 0.0  # zero produced-goal training


# ------------------------------------------------- λ_plan schedule ----------
def test_lambda_plan_three_phases():
    assert lambda_plan_at(0, PHASES) == 0.0                  # Phase A: LP
    assert lambda_plan_at(1999, PHASES) == 0.0
    assert lambda_plan_at(2000, PHASES) == 0.0               # ramp start
    assert lambda_plan_at(5000, PHASES) == pytest.approx(0.5)  # mid-ramp
    assert lambda_plan_at(8000, PHASES) == 1.0               # ramp end
    assert lambda_plan_at(30000, PHASES) == 1.0              # Phase C


def test_lambda_plan_reproduction_modes():
    for s in (0, 2000, 5000, 10000, 30000):
        assert lambda_plan_at(s, PHASES, mode="0") == 0.0    # frozen-trunk v1.5 regime
        assert lambda_plan_at(s, PHASES, mode="1") == 1.0    # full joint from step 0


def test_phases_reject_a_bad_boundary():
    with pytest.raises(ValueError):
        CurriculumPhases(phase_a=8000, phase_b=2000)
    with pytest.raises(ValueError):
        lambda_plan_at(0, PHASES, mode="bogus")


# ------------------------------------------------- plan smoothness ----------
def test_smoothness_rewards_a_straight_constant_speed_plan():
    b, s = 4, 20
    # a straight, constant-velocity plan has zero jerk and zero curvature-rate
    x = torch.linspace(0, 10, s)
    straight = torch.stack([x, torch.zeros(s)], dim=-1)[None].repeat(b, 1, 1)
    loss, parts = plan_smoothness_loss(straight)
    assert parts["jerk"] == pytest.approx(0.0, abs=1e-6)
    assert parts["curv_rate"] == pytest.approx(0.0, abs=1e-6)
    # a jerky plan costs strictly more
    torch.manual_seed(0)
    jerky = straight + torch.randn(b, s, 2) * 0.3
    loss_jerky, _ = plan_smoothness_loss(jerky)
    assert float(loss_jerky) > float(loss)


def test_smoothness_needs_a_dense_plan():
    with pytest.raises(ValueError, match="precondition"):
        plan_smoothness_loss(torch.zeros(2, 3, 2))          # 3 points: no 3rd diff
    plan_smoothness_loss(torch.zeros(2, 4, 2))              # 4 points: exactly one


def test_smoothness_is_differentiable():
    plan = torch.randn(2, 20, 2, requires_grad=True)
    plan_smoothness_loss(plan)[0].backward()
    assert plan.grad is not None and torch.isfinite(plan.grad).all()


# ------------------------------------------------- factorised CE ------------
def _factor_out(b=4):
    torch.manual_seed(0)
    return {"lat_logits": torch.randn(b, 8), "lon_logits": torch.randn(b, 7),
            "dist_logits": torch.randn(b, 8)}


def test_factorised_ce_masks_unknown_sentinels():
    out = _factor_out()
    # all-unknown targets contribute exactly zero and no NaN (the dead-param guard)
    z = torch.full((4,), IGNORE_INDEX)
    loss, parts = factorised_ce(out, z, z, z)
    assert float(loss) == 0.0
    assert all(v == 0.0 for v in parts.values())
    # a real target produces a positive CE
    real = torch.tensor([0, 1, 2, 3])
    loss2, parts2 = factorised_ce(out, real, real.clamp(max=6), real)
    assert float(loss2) > 0.0 and parts2["lon_ce"] > 0.0


def test_factorised_weights_zero_reproduce_the_baseline():
    """--lat/lon/dist-weight 0 -> no factorised loss (the isolate-select diff)."""
    out = _factor_out()
    real = torch.tensor([0, 1, 2, 3])
    loss, _ = factorised_ce(out, real, real.clamp(max=6), real,
                            w_lat=0.0, w_lon=0.0, w_dist=0.0)
    assert float(loss) == 0.0


def test_factorised_ce_gradient_reaches_only_unmasked_rows():
    out = {"lat_logits": torch.randn(3, 8, requires_grad=True),
           "lon_logits": torch.randn(3, 7, requires_grad=True),
           "dist_logits": torch.randn(3, 8, requires_grad=True)}
    lat_tgt = torch.tensor([2, IGNORE_INDEX, 5])            # middle row masked
    ok = torch.tensor([0, 1, 2])
    factorised_ce(out, lat_tgt, ok, ok, w_lat=1.0)[0].backward()
    g = out["lat_logits"].grad
    assert float(g[1].abs().sum()) == 0.0                   # masked row: no gradient
    assert float(g[0].abs().sum()) > 0.0 and float(g[2].abs().sum()) > 0.0


# ------------------------------------------------- canary controller --------
def test_canary_controller_is_quiet_when_healthy():
    c = CanaryController(baseline=0.452)
    mult, action = c.update(0.460)                          # +0.008, under ctrl_thresh
    assert mult == 1.0 and action == "ok"
    assert c.effective_lambda(1.0) == 1.0


def test_canary_controller_halves_lambda_on_a_soft_breach_and_never_kills():
    c = CanaryController(baseline=0.452)
    mult, action = c.update(0.452 + 0.06)                   # soft breach (> 0.05)
    assert mult == 0.5 and "halved" in action
    assert c.effective_lambda(1.0) == 0.5
    # it only ratchets DOWN — a subsequent healthy read does not restore λ
    mult2, _ = c.update(0.455)
    assert mult2 <= 0.5


def test_canary_controller_holds_at_floor_after_consecutive_hard_breaches():
    """⭐ v4.2 CAP-AND-HOLD: three hard breaches drop λ_plan to the FLOOR, NOT to 0.
    v4.1 ran the old halve-to-ZERO here and starved its planner (lam_mult -> 1.5e-5);
    the floor guarantees the planner always keeps a real gradient into the trunk."""
    c = CanaryController(baseline=0.452, alarm_evals=3, mult_floor=0.25)
    for i in range(2):
        mult, action = c.update(0.452 + 0.4)                # hard breach
        assert mult > 0.0, "must not zero before alarm_evals in a row"
    mult, action = c.update(0.452 + 0.4)                    # 3rd consecutive
    assert mult == 0.25 and action == "alarm_lambda_to_floor"
    # even a full alarm is not a KILL and NOT a starvation — λ_plan holds at the floor
    assert c.effective_lambda(1.0) == 0.25
    # and it NEVER goes below the floor no matter how many more hard breaches arrive
    for _ in range(10):
        mult, _ = c.update(0.452 + 5.0)
        assert mult == 0.25


def test_canary_controller_never_starves_the_planner():
    """The whole point of the v4.2 fix: under a relentless canary regression (the
    exact v4.1 condition — every eval breaches), the multiplier is monotone DOWN but
    is bounded below by the floor, so effective λ_plan is never ~0."""
    c = CanaryController(baseline=0.452, mult_floor=0.25)
    seen = []
    for _ in range(50):                                     # 50 straight soft breaches
        mult, _ = c.update(0.452 + 0.10)
        seen.append(mult)
    assert min(seen) >= 0.25 - 1e-12                        # never below the floor
    assert seen == sorted(seen, reverse=True)               # monotone non-increasing
    assert c.effective_lambda(1.0) >= 0.25                  # planner keeps a real gradient


def test_canary_controller_rejects_a_zero_floor():
    """A floor of 0 IS the v4.1 halve-to-zero bug — construction refuses it."""
    with pytest.raises(ValueError):
        CanaryController(baseline=0.452, mult_floor=0.0)


def test_canary_hard_streak_resets_on_a_healthy_read():
    c = CanaryController(baseline=0.452, alarm_evals=3)
    c.update(0.452 + 0.4)
    c.update(0.452 + 0.4)
    c.update(0.452)                                         # healthy -> resets streak
    mult, _ = c.update(0.452 + 0.4)                         # only 1 in the new streak
    assert mult > 0.0
