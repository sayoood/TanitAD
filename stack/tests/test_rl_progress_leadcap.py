"""`H-RL-PROGRESS-LEADCAP-1` -- the `progress` lead cap, and its coordinate discipline.

Pre-registration: `TanitAD Research Lab/Architecture & Inference/Research/
2026-09-06-refcv4b-rl-repair/PREREG.md`, committed BEFORE `rewards.py` changed.

WHAT THESE TESTS ARE FOR, beyond "the code runs":

  1. the cap is INERT where it must be (no lead / mode off / lead far), and inert
     EXACTLY -- a repair whose no-op path is only approximately a no-op silently
     re-scores every banked number;
  2. the cap is a POSITION QUERY and NOT A RATE. `M84` measured the lead's
     closing rate a clean null (+0.0061) against position's +0.4145
     [+0.2018, +0.6120]; a ranking term whose ordering rested on the rate would
     inherit a coordinate the latent does not carry. The test perturbs the
     lead's INTERMEDIATE samples with its endpoint held fixed and requires the
     term to be EXACTLY unchanged -- and it carries its own DISCRIMINATING
     CONTROL: the same perturbation MUST move `ttc_violation`'s closing rate, or
     the test proves nothing about rates;
  3. the cap is a property of the SCENE, identical for every candidate, so it
     cannot rank on anything the reward may not see;
  4. the term still RANKS after the repair -- an inert term would move a rate for
     the wrong reason, which the pre-registration calls a DEGENERATE REPAIR.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.rl import rewards as R  # noqa: E402


DT = 0.5
GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
H = (len(GRID) - 1) * DT          # 2.0 s


def straight(v0: float, decel: float = 0.0):
    """A [1, 1, 1, 5, 2] straight path at v0 with a constant deceleration."""
    xs = []
    x, v = 0.0, v0
    for i, _ in enumerate(GRID):
        if i == 0:
            xs.append(0.0)
            continue
        x += v * DT
        v = max(v - decel * DT, 0.0)
        xs.append(x)
    t = torch.tensor(xs, dtype=torch.float32)
    return torch.stack([t, torch.zeros_like(t)], dim=-1).reshape(1, 1, 1, 5, 2)


def lead_track(x0: float, v_lead: float):
    """A [1, 1, 1, 5, 2] lead track starting x0 ahead and moving at v_lead."""
    t = torch.tensor([x0 + v_lead * s for s in GRID], dtype=torch.float32)
    return torch.stack([t, torch.zeros_like(t)], dim=-1).reshape(1, 1, 1, 5, 2)


def base_ctx(v0: float, lead=None, **kw):
    ctx = {"dt": DT, "v0": torch.tensor([v0]).reshape(1, 1, 1),
           "lead_len_m": 4.5, "target_time_gap_s": 2.0}
    if lead is not None:
        ctx["lead_path"] = lead
    ctx.update(kw)
    return ctx


def legacy_progress(traj, ctx):
    """The pre-repair term, reimplemented here so the no-op claim is checked
    against an INDEPENDENT expression and not against the code under test."""
    kin = R.kinematics(traj, ctx.get("dt", R.DT_S))
    v0 = ctx["v0"]
    horizon_s = (traj.shape[-2] - 1) * float(ctx.get("dt", R.DT_S))
    ref = (v0 * horizon_s).clamp_min(float(ctx.get("progress_min_ref_m", 5.0)))
    while ref.dim() < kin.along.dim():
        ref = ref.unsqueeze(-1)
    return kin.along / ref


# --------------------------------------------------------------------------- #
# 1. INERT WHERE IT MUST BE, AND INERT EXACTLY                                  #
# --------------------------------------------------------------------------- #
def test_mode_off_is_bit_identical_to_the_pre_repair_term():
    traj = straight(12.0)
    ctx = base_ctx(12.0, lead=lead_track(15.0, 8.0), progress_lead_cap=False)
    assert torch.equal(R._progress(traj, ctx), legacy_progress(traj, ctx))


def test_no_lead_is_inert_even_with_the_cap_on():
    traj = straight(12.0)
    on = R._progress(traj, base_ctx(12.0))
    off = R._progress(traj, base_ctx(12.0, progress_lead_cap=False))
    assert torch.equal(on, off)
    assert torch.equal(on, legacy_progress(traj, base_ctx(12.0)))


def test_far_lead_does_not_bind():
    """Lead 200 m ahead: the achievable bound is far above the free reference."""
    traj = straight(12.0)
    lead = lead_track(200.0, 12.0)
    on = R._progress(traj, base_ctx(12.0, lead=lead))
    off = R._progress(traj, base_ctx(12.0, lead=lead, progress_lead_cap=False))
    assert torch.equal(on, off)


def test_absent_lead_returns_none_from_the_reference_helper():
    assert R.achievable_along_ref(base_ctx(10.0), 20.0) is None
    assert R.achievable_along_ref(
        base_ctx(10.0, lead=lead_track(20.0, 5.0), progress_lead_cap=False), 20.0) is None


# --------------------------------------------------------------------------- #
# 2. THE REPAIR ITSELF                                                          #
# --------------------------------------------------------------------------- #
def test_close_lead_removes_the_payment_to_the_constant_velocity_path():
    """The measured defect, reproduced in miniature and then repaired.

    v0 = 12 m/s. hold-v0 travels 24 m in 2 s. The 'human' brakes for a lead that
    is close and slow, travelling less. Under the pre-repair term hold-v0 scores
    STRICTLY higher; under the cap the payment is removed.
    """
    v0 = 12.0
    hold = straight(v0)
    human = straight(v0, decel=3.0)
    lead = lead_track(18.0, 6.0)          # ends at 18 + 12 = 30 m
    ctx_on = base_ctx(v0, lead=lead)
    ctx_off = base_ctx(v0, lead=lead, progress_lead_cap=False)

    gap_off = float(R._progress(hold, ctx_off) - R._progress(human, ctx_off))
    gap_on = float(R._progress(hold, ctx_on) - R._progress(human, ctx_on))
    assert gap_off > 0.0, "the pre-repair defect must be present, or this proves nothing"
    assert gap_on < gap_off
    assert gap_on >= 0.0                  # the cap removes a payment, never adds a penalty


def test_the_reference_is_the_lead_position_minus_the_standoff():
    """The value is hand-computable from the lead's END POSITION alone."""
    v0 = 10.0
    lead = lead_track(20.0, 5.0)                       # x at t=2 s is 30.0
    ref_free = torch.tensor([v0 * H]).reshape(1, 1, 1)  # 20.0
    got = R.achievable_along_ref(base_ctx(v0, lead=lead), ref_free)
    standoff = 4.5 + 2.0 * v0                          # 24.5
    expect = min(float(ref_free), 30.0 - standoff)     # min(20.0, 5.5) = 5.5
    assert float(got) == pytest.approx(expect, abs=1e-6)


def test_reference_floors_at_min_ref_when_the_lead_is_inside_the_standoff():
    v0 = 10.0
    lead = lead_track(2.0, 0.0)                        # x at t=2 s is 2.0 -> ref_lead < 0
    got = R.achievable_along_ref(base_ctx(v0, lead=lead),
                                 torch.tensor([v0 * H]).reshape(1, 1, 1))
    assert float(got) == pytest.approx(5.0, abs=1e-6)  # progress_min_ref_m


def test_mode_lead_differs_from_achievable_only_above_the_free_reference():
    """The pre-registered form carries a SECOND cap (at ref_free) that was not the
    stated intent. `mode="lead"` is the sensitivity variant; the two must agree
    wherever the candidate does not exceed the free reference."""
    v0 = 10.0
    lead = lead_track(300.0, 10.0)                     # far: ref_lead >> ref_free
    slow, fast = straight(6.0), straight(16.0)         # 12 m and 32 m vs ref_free 20 m
    for traj, same in ((slow, True), (fast, False)):
        a = R._progress(traj, base_ctx(v0, lead=lead, progress_lead_cap="achievable"))
        b = R._progress(traj, base_ctx(v0, lead=lead, progress_lead_cap="lead"))
        assert torch.equal(a, b) is same


def test_bad_mode_is_refused():
    with pytest.raises(ValueError, match="progress_lead_cap"):
        R._progress(straight(10.0), base_ctx(10.0, lead=lead_track(20.0, 5.0),
                                             progress_lead_cap="capped"))


# --------------------------------------------------------------------------- #
# 3. COORDINATE DISCIPLINE -- A POSITION QUERY, NOT A RATE                       #
# --------------------------------------------------------------------------- #
def test_cap_is_a_position_query_and_not_a_rate():
    """Perturb the lead's INTERMEDIATE samples; hold its ENDPOINT fixed.

    A reference computed from the lead's POSITION at the horizon end is EXACTLY
    unchanged. A reference that used the closing RATE could not be.

    The second half is the DISCRIMINATING CONTROL demanded by CLAUDE.md's probe
    rule: the same perturbation MUST move a genuinely rate-based quantity, or
    this test would pass for a term that ignores the lead entirely.
    """
    v0 = 12.0
    traj = straight(v0)
    smooth = lead_track(18.0, 6.0)
    wobbly = smooth.clone()
    wobbly[..., 1:-1, 0] += torch.tensor([-4.0, 5.0, -3.0])   # endpoint untouched
    assert torch.equal(smooth[..., -1, :], wobbly[..., -1, :])

    p_smooth = R._progress(traj, base_ctx(v0, lead=smooth))
    p_wobbly = R._progress(traj, base_ctx(v0, lead=wobbly))
    assert torch.equal(p_smooth, p_wobbly), "the cap read something other than the endpoint"

    # DISCRIMINATING CONTROL, two halves, both required.
    # (a) the CLOSING RATE -- the coordinate `M84` measured as a clean null, and the
    #     one a graded TTC term would rank on -- MUST move under this edit.
    def closing_rate(lead):
        gap = (lead[..., 1:, :] - traj[..., 1:, :]).norm(dim=-1)
        gap = (gap - 4.5).clamp_min(0.0)
        return (gap[..., :-1] - gap[..., 1:]) / DT
    assert not torch.equal(closing_rate(smooth), closing_rate(wobbly)), (
        "the perturbation moved no closing rate, so the first assertion is vacuous")
    # (b) what a PER-STEP reader of the lead sees must also move, or the first
    #     assertion would pass for a cap that ignores the lead entirely.
    per_step = lambda lead: (lead[..., 1:, :] - traj[..., 1:, :]).norm(dim=-1)
    assert not torch.equal(per_step(smooth), per_step(wobbly)), (
        "the perturbation moved no per-step lead reading, so the first assertion "
        "is vacuous")


def test_headway_is_blind_to_this_perturbation_and_that_is_the_next_candidate():
    """MEASURED SIDE-FINDING, and it is the pre-registered next lever.

    `_headway` reduces the per-step time gap with ``amin`` -- a MIN-OVER-N ORDER
    STATISTIC. On the fixture above the worst step is the LAST one, which the
    perturbation does not touch, so `headway` reads **identically** while three of
    the five lead samples moved by up to 5 m. That is the same family of defect as
    `fan_floor@k` read alone: a min-over-N quantity answers a narrower question
    than the one it is quoted for. PREREG.md names `headway` as the next candidate
    if the `progress` repair does not close the rate/mean divergence; this test
    pins the mechanism so the claim is not re-derived later from memory.
    """
    v0 = 12.0
    traj = straight(v0)
    smooth = lead_track(18.0, 6.0)
    wobbly = smooth.clone()
    wobbly[..., 1:-1, 0] += torch.tensor([-4.0, 5.0, -3.0])
    assert torch.equal(R._headway(traj, base_ctx(v0, lead=smooth)),
                       R._headway(traj, base_ctx(v0, lead=wobbly)))
    # ... while a QUANTILE over the same per-step gaps does move.
    def gap_q(lead, q=0.5):
        g = (lead[..., 1:, :] - traj[..., 1:, :]).norm(dim=-1) - 4.5
        return torch.quantile(g.clamp_min(0.0), q, dim=-1)
    assert not torch.equal(gap_q(smooth), gap_q(wobbly))


def test_cap_is_a_property_of_the_scene_not_of_the_candidate():
    """Same window, wildly different candidates -> the SAME achievable bound."""
    v0 = 12.0
    lead = lead_track(18.0, 6.0)
    ctx = base_ctx(v0, lead=lead)
    ref_free = (ctx["v0"] * H).clamp_min(5.0)
    a = R.achievable_along_ref(ctx, ref_free)
    b = R.achievable_along_ref(ctx, ref_free)
    assert torch.equal(a, b)
    # and it takes no trajectory argument at all -- a candidate cannot move it
    import inspect
    assert "traj" not in inspect.signature(R.achievable_along_ref).parameters


def test_no_ego_future_or_selector_output_enters_the_cap():
    """The reference must be computable with every forbidden key absent."""
    v0 = 12.0
    ctx = base_ctx(v0, lead=lead_track(18.0, 6.0))
    assert not (set(ctx) & R.FORBIDDEN_REWARD_INPUTS)
    assert "gt_traj" not in ctx
    ref = R.achievable_along_ref(ctx, (ctx["v0"] * H).clamp_min(5.0))
    assert torch.isfinite(ref).all()


# --------------------------------------------------------------------------- #
# 4. THE DEGENERACY GUARD -- THE TERM MUST STILL RANK                            #
# --------------------------------------------------------------------------- #
def test_the_term_still_ranks_candidates_after_the_repair():
    """A cap that made every candidate equal would move a rate for the wrong
    reason. On a fan spanning the achievable bound the term must keep a spread."""
    v0 = 12.0
    lead = lead_track(40.0, 10.0)          # ends at 60 m -> ref_lead = 60 - 28.5 = 31.5
    ctx = base_ctx(v0, lead=lead)
    fan = torch.cat([straight(s) for s in (2.0, 6.0, 10.0, 14.0)], dim=1)  # [1,4,1,5,2]
    p = R._progress(fan, ctx).reshape(-1)
    assert float(p.max() - p.min()) > 0.1
    assert len(set(round(float(x), 9) for x in p)) > 1


def test_a_fully_capped_fan_is_visibly_degenerate():
    """The failure mode the guard exists to see, exhibited on purpose."""
    v0 = 12.0
    lead = lead_track(6.0, 0.0)            # ref_lead floors at min_ref for everyone
    ctx = base_ctx(v0, lead=lead)
    fan = torch.cat([straight(s) for s in (8.0, 12.0, 16.0, 20.0)], dim=1)
    p = R._progress(fan, ctx).reshape(-1)
    assert float(p.max() - p.min()) == pytest.approx(0.0, abs=1e-7)


# --------------------------------------------------------------------------- #
# 5. THE COMPOSED REWARD STILL WORKS, AND THE ABSENCE RULE STILL HOLDS           #
# --------------------------------------------------------------------------- #
def test_composed_reward_runs_with_the_cap_on_and_shapes_are_unchanged():
    v0 = 12.0
    fan = torch.cat([straight(s) for s in (8.0, 12.0, 16.0)], dim=1)
    ctx = base_ctx(v0, lead=lead_track(30.0, 8.0))
    spec = R.RewardSpec(weights=dict(R.DEFAULT_WEIGHTS), dt=DT)
    r_on = spec(fan, ctx)
    r_off = spec(fan, {**ctx, "progress_lead_cap": False})
    assert r_on.shape == r_off.shape == fan.shape[:-2]
    assert torch.isfinite(r_on).all()


def test_absence_still_means_no_constraint():
    """No lead -> the cap adds nothing, in the composed reward too."""
    v0 = 12.0
    fan = torch.cat([straight(s) for s in (8.0, 12.0, 16.0)], dim=1)
    ctx = base_ctx(v0)
    spec = R.RewardSpec(weights=dict(R.DEFAULT_WEIGHTS), dt=DT)
    assert torch.equal(spec(fan, ctx), spec(fan, {**ctx, "progress_lead_cap": False}))
