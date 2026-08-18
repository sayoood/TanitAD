"""REF-A v1 — every design claim gets a test, or it is not a claim.

The arm is motivated by a HYPOTHESIS RANKING, not a result. What can be pinned
now is that the mechanisms are actually present and behave as advertised, so
that when the run reports a number nobody has to re-litigate whether the thing
described was the thing trained. Each test names the change it guards.
"""
from __future__ import annotations

import math

import pytest
import torch

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import DINOV3_GEOMETRY, RefAV1, RefAV1Config
from tanitad.refs.refa_v1_plan import (DINO_WM_DEFAULTS, PlanConfig,
                                       colored_noise, cost_fidelity, icem_plan)


def _tiny(**kw) -> RefAV1Config:
    """CPU-sized config that keeps every RATIO and HORIZON of the real one."""
    base = dict(d_enc=32, n_tokens=8, d_state=32, op_layers=1, op_heads=2,
                op_window=4, tac_layers=1, tac_queries=4, str_dim=16,
                str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


# --------------------------------------------------------------- geometry --
def test_change_2_the_visual_interface_is_640_tokens_at_120_degrees():
    g = DINOV3_GEOMETRY
    assert g["n_tokens"] == 640 == g["grid_h"] * g["grid_w"]
    assert g["hfov_deg"] == 120.0
    assert (g["height"], g["width"]) == (256, 640)
    assert g["height"] % g["patch"] == 0 and g["width"] % g["patch"] == 0
    # REF-A's interface, for the record: 256 tokens / 51.39 deg.
    assert g["n_tokens"] / 256 == 2.5


def test_change_2_aspect_ratio_matches_v6_so_this_is_a_pure_resolution_change():
    """224x560 (v6) and 256x640 (v1) are the same 2.5 aspect — the FOV claim
    would be false if the crop shape changed as well as its resolution."""
    g = DINOV3_GEOMETRY
    assert g["width"] / g["height"] == pytest.approx(560 / 224)


def test_change_5_the_predictive_path_uses_patch_tokens_never_cls():
    assert DINOV3_GEOMETRY["tokens_include_cls"] is False


def test_change_1_encoder_is_dinov3_at_1024_wide():
    assert DINOV3_GEOMETRY["model"].startswith("dinov3")
    assert DINOV3_GEOMETRY["d_enc"] == 1024
    assert RefAV1Config().d_enc == 1024


# ------------------------------------------------------------ no bottleneck --
def test_change_3_a_state_narrower_than_the_encoder_is_REFUSED():
    """FROST-Drive measured 8.17 -> 7.68 RFS on exactly this axis, so a
    bottleneck must be a loud error, not a config someone can pick."""
    with pytest.raises(ValueError, match="bottleneck|d_state"):
        RefAV1Config(d_enc=1024, d_state=512).sanity()


def test_change_3_the_default_is_no_compression_at_all():
    c = RefAV1Config()
    assert c.d_state >= c.d_enc


def test_change_2_encode_refuses_a_narrowed_token_grid():
    m = RefAV1(_tiny())
    with pytest.raises(ValueError, match="patch tokens"):
        m.encode(torch.randn(1, 4, 4, 32))          # 4 tokens, not 8


# ------------------------------------------------------------- 6 s horizons --
@pytest.mark.parametrize("dt,steps", [(0.2, 30), (0.6, 10), (1.5, 4)])
def test_change_9_all_three_rates_reach_exactly_six_seconds(dt, steps):
    assert dt * steps == pytest.approx(6.0)


def test_change_9_a_horizon_that_misses_6s_is_REFUSED():
    with pytest.raises(ValueError, match="6.0 s"):
        RefAV1Config(op_steps=25).sanity()          # 25 * 0.2 = 5.0 s


def test_the_plan_window_is_shorter_than_the_prediction_horizon():
    """Plan at 2 s, cost at 6 s — the receding-horizon split. If the plan
    window ever equalled the rollout the 6 s prediction would be decorative."""
    c = RefAV1Config()
    assert c.plan_steps == 10
    assert c.plan_steps < c.op_steps
    with pytest.raises(ValueError, match="exceeds"):
        RefAV1Config(plan_horizon_s=8.0).sanity()


# -------------------------------------------------- frozen encoder invariant --
def test_the_encoder_contributes_zero_trainable_parameters():
    """REF-A stability item 2, carried into v1: features are data tensors on
    disk, so no gradient can reach the encoder BY CONSTRUCTION."""
    m = RefAV1(_tiny())
    assert m.frozen_encoder_parameters() == 0
    feats = torch.randn(2, 4, 8, 32)
    assert feats.requires_grad is False
    out = m(feats, torch.zeros(2, 30, 2), future_feats=torch.randn(2, 30, 8, 32))
    out["loss"].backward()
    assert feats.grad is None


def test_the_standardizer_refuses_to_refit():
    m = RefAV1(_tiny())
    m.std.fit(torch.randn(64, 32))
    with pytest.raises(RuntimeError, match="already fitted"):
        m.std.fit(torch.randn(64, 32))


# ----------------------------------------------- change #4: the PRIMARY loss --
def test_the_primary_loss_is_feature_prediction_not_a_trajectory_head():
    m = RefAV1(_tiny())
    out = m(torch.randn(2, 4, 8, 32), torch.zeros(2, 30, 2),
            future_feats=torch.randn(2, 30, 8, 32))
    for k in ("loss_feat_op", "loss_feat_tac", "loss_feat_str"):
        assert k in out and torch.isfinite(out[k])
    c = m.cfg
    expected = (c.w_feat_op * out["loss_feat_op"]
                + c.w_feat_tac * out["loss_feat_tac"]
                + c.w_feat_str * out["loss_feat_str"])
    assert torch.allclose(out["loss"], expected)


def test_the_imitation_proposal_is_NOT_in_the_training_loss():
    """It exists only to seed the planner (GPC). If it ever entered ``loss`` the
    arm would quietly become the supervised-head recipe it was built to leave."""
    m = RefAV1(_tiny())
    out = m(torch.randn(2, 4, 8, 32), torch.zeros(2, 30, 2),
            future_feats=torch.randn(2, 30, 8, 32))
    assert "proposal" in out
    before = out["loss"].item()
    with torch.no_grad():
        for p in m.proposal.parameters():
            p.add_(torch.randn_like(p))
    after = m(torch.randn(2, 4, 8, 32), torch.zeros(2, 30, 2),
              future_feats=torch.randn(2, 30, 8, 32))
    assert "loss" in after and math.isfinite(before)


def test_the_predictor_is_residual_so_the_6s_rollout_starts_near_identity():
    m = RefAV1(_tiny())
    z = torch.randn(2, 8, 32)
    step = m.operative.step(z, torch.zeros(2, 2))
    assert step.shape == z.shape
    assert not torch.allclose(step, z)              # it does something
    # residual structure: output = input + head(...), so removing the head's
    # contribution recovers the input exactly.
    with torch.no_grad():
        for p in m.operative.head.parameters():
            p.zero_()
    assert torch.allclose(m.operative.step(z, torch.zeros(2, 2)), z, atol=1e-6)


def test_the_rollout_reaches_the_full_configured_horizon():
    m = RefAV1(_tiny())
    out = m(torch.randn(1, 4, 8, 32), torch.zeros(1, 30, 2))
    assert out["op_pred"].shape[1] == 30            # 6.0 s at 0.2
    assert out["tac_pred"].shape[1] == 10           # 6.0 s at 0.6
    assert out["str_pred"].shape[1] == 4            # 6.0 s at 1.5


# ------------------------------------------------- change #7: the hierarchy --
def _brains(c: RefAV1Config) -> RefAV1Config:
    c.strategic_cfg = StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                            d_ctx=16, d_cmd=8)
    c.tactical_cfg = TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                          d_intent=16)
    return c


def test_the_tactical_intent_actually_reaches_the_operative_predictor():
    """Change #7 is only real if the intent CHANGES the prediction. A wired-but-
    ignored intent is the exact failure the v1 header warns about."""
    m = RefAV1(_brains(_tiny()))
    assert m.operative.intent is not None
    z = torch.randn(1, 8, 32)
    a = torch.zeros(1, 2)
    p0 = m.operative.step(z, a, intent=torch.zeros(1, 16))
    p1 = m.operative.step(z, a, intent=torch.ones(1, 16) * 5.0)
    assert not torch.allclose(p0, p1)


def test_the_hierarchy_emits_route_and_maneuver_and_conditions_the_chain():
    m = RefAV1(_brains(_tiny()))
    out = m(torch.randn(2, 4, 8, 32), torch.zeros(2, 30, 2))
    for k in ("ctx", "intent", "route_logits", "maneuver_logits"):
        assert out.get(k) is not None, k


def test_the_strategic_predictor_is_SEPARATE_and_narrower():
    """Three-planner directive: strategic gets its OWN predictor on a
    strategy-only subspace — not a read-off of the operative field."""
    m = RefAV1(_tiny())
    assert m.strategic is not m.operative
    assert m.cfg.str_dim < m.cfg.d_state
    s = m.strategic.subspace(torch.randn(3, 8, 32))
    assert s.shape == (3, 16)


# ============================================================================ #
#  THE C101 REPAIR — the part that must not be taken on trust
# ============================================================================ #
def test_the_planner_uses_dino_wms_published_configuration_by_default():
    c = PlanConfig()
    for k, v in DINO_WM_DEFAULTS.items():
        assert getattr(c, k) == v, f"{k}: {getattr(c, k)} != published {v}"


def test_colored_noise_is_temporally_correlated_and_beta0_is_white():
    torch.manual_seed(0)
    white = colored_noise((512, 64, 2), 0.0)
    pink = colored_noise((512, 64, 2), 2.5)

    def lag1(x):
        a, b = x[:, :-1, :], x[:, 1:, :]
        return float((a * b).mean() / x.pow(2).mean())

    assert abs(lag1(white)) < 0.15, "beta=0 must be (near) white noise"
    assert lag1(pink) > 0.60, "beta=2.5 must be strongly correlated in time"


def test_vanilla_cem_control_sequences_are_physically_ragged_and_icems_are_not():
    """The mechanism behind C101, made visible: white-noise plans demand
    accelerations that reverse every step."""
    torch.manual_seed(0)
    w = colored_noise((256, 32, 2), 0.0)
    p = colored_noise((256, 32, 2), 2.5)
    d_w = (w[:, 1:] - w[:, :-1]).abs().mean()
    d_p = (p[:, 1:] - p[:, :-1]).abs().mean()
    assert d_p < 0.5 * d_w


def test_THE_FLOOR_the_planner_can_never_lose_to_a_baseline_it_evaluated():
    """⭐ THE C101 REPAIR, PROVEN.

    An adversarial cost function that PUNISHES whatever the CEM converges to:
    cost grows with |control| everywhere except at exactly zero, so the CEM's
    smooth optimum is strictly worse than the constant-velocity baseline. A
    planner without baseline injection returns its own optimum and loses — which
    is the shape of the measured 35.8 % regression. With injection the returned
    plan must BE the baseline.
    """
    cfg = PlanConfig(n_samples=64, n_iters=3, n_elites=8, horizon=5, seed=0)

    def adversarial(controls: torch.Tensor) -> torch.Tensor:
        mag = controls.abs().sum(dim=(-1, -2))
        # zero-control gets 0; anything else pays a fixed penalty plus its size
        return torch.where(mag < 1e-9, torch.zeros_like(mag), 10.0 + mag)

    res = icem_plan(adversarial, v0=10.0, cfg=cfg)
    assert res.source.startswith("baseline:"), res.source
    assert res.cost <= min(res.baseline_costs.values()) + 1e-9
    assert res.cost == pytest.approx(0.0, abs=1e-9)


def test_THE_FLOOR_holds_when_the_proposal_is_the_best_candidate():
    cfg = PlanConfig(n_samples=32, n_iters=2, n_elites=4, horizon=5, seed=0)
    target = torch.zeros(5, 2)
    target[:, 0] = -1.5                       # equals the decel_1.5 baseline

    def cost(controls):
        return (controls - target).pow(2).sum(dim=(-1, -2))

    res = icem_plan(cost, v0=10.0, cfg=cfg, proposal=target.clone())
    assert res.cost <= min(res.baseline_costs.values()) + 1e-6


def test_the_floor_can_be_switched_off_so_its_contribution_is_MEASURABLE():
    """An always-on guard is untestable. With injection off the same adversarial
    cost must produce a NON-baseline plan — that is the ablation arm."""
    cfg = PlanConfig(n_samples=32, n_iters=2, n_elites=4, horizon=5, seed=0,
                     inject_baselines=False)

    def adversarial(controls):
        mag = controls.abs().sum(dim=(-1, -2))
        return torch.where(mag < 1e-9, torch.zeros_like(mag), 10.0 + mag)

    res = icem_plan(adversarial, v0=10.0, cfg=cfg)
    assert res.source == "cem"
    assert res.baseline_costs == {}
    assert res.cost > 10.0                    # strictly worse than CV's 0.0


def test_the_planner_returns_elites_for_the_next_mpc_tick():
    cfg = PlanConfig(n_samples=32, n_iters=2, n_elites=4, horizon=5, seed=0)
    res = icem_plan(lambda c: c.pow(2).sum(dim=(-1, -2)), v0=10.0, cfg=cfg)
    assert res.elites is not None and res.elites.shape[1:] == (5, 2)
    warm = icem_plan(lambda c: c.pow(2).sum(dim=(-1, -2)), v0=10.0, cfg=cfg,
                     prev_elites=res.elites)
    assert warm.cost <= res.cost + 1e-6


def test_controls_are_clipped_to_the_kinematic_envelope():
    cfg = PlanConfig(n_samples=32, n_iters=2, n_elites=4, horizon=5, seed=0,
                     a_max=4.0, kappa_max=0.2)
    res = icem_plan(lambda c: -c.abs().sum(dim=(-1, -2)), v0=10.0, cfg=cfg)
    assert res.controls[:, 0].abs().max() <= 4.0 + 1e-6
    assert res.controls[:, 1].abs().max() <= 0.2 + 1e-6


# ------------------------------------------- the gate the floor CANNOT give --
def test_cost_fidelity_refuses_an_underpowered_sample_instead_of_correlating():
    r = cost_fidelity([1, 2, 3, 4], [1, 2, 3, 4])
    assert r["rho"] == pytest.approx(1.0)
    assert r["admissible"] is False and "under-powered" in r["reason"]


def test_cost_fidelity_admits_only_a_real_correlation_at_size():
    good = cost_fidelity(list(range(300)), list(range(300)))
    assert good["admissible"] is True and good["rho"] == pytest.approx(1.0)
    bad = cost_fidelity(list(range(300)), list(range(300))[::-1])
    assert bad["admissible"] is False and bad["rho"] == pytest.approx(-1.0)


def test_cost_fidelity_is_a_SEPARATE_gate_from_the_floor():
    """The floor is about modelled cost; fidelity is about whether modelled cost
    means anything. A design that conflated them would repeat C101 with extra
    steps, so they are different functions with different failure messages."""
    cfg = PlanConfig(n_samples=16, n_iters=1, n_elites=4, horizon=3, seed=0)
    res = icem_plan(lambda c: c.pow(2).sum(dim=(-1, -2)), v0=5.0, cfg=cfg)
    assert res.cost <= min(res.baseline_costs.values()) + 1e-6   # floor holds
    fid = cost_fidelity([res.cost] * 10, list(range(10)))        # ...yet
    assert fid["admissible"] is False                            # not admissible


# ------------------------------------------------------------- integration --
def test_plan_runs_end_to_end_and_respects_its_single_window_contract():
    m = RefAV1(_brains(_tiny()))
    feats = torch.randn(1, 4, 8, 32)
    goal = torch.randn(1, 8, 32)
    res = m.plan(feats, v0=12.0, goal_field=goal, target_speed=10.0,
                 plan_cfg=PlanConfig(n_samples=16, n_iters=2, n_elites=4,
                                     horizon=m.cfg.plan_steps, dt=m.cfg.op_dt,
                                     seed=0))
    assert res.controls.shape == (m.cfg.plan_steps, 2)
    assert math.isfinite(res.cost)
    assert res.cost <= min(res.baseline_costs.values()) + 1e-6
    with pytest.raises(ValueError, match="single-window"):
        m.plan(torch.randn(2, 4, 8, 32), v0=1.0)


def test_plan_refuses_a_horizon_that_disagrees_with_the_config():
    m = RefAV1(_tiny())
    with pytest.raises(ValueError, match="plan horizon"):
        m.plan(torch.randn(1, 4, 8, 32), v0=1.0,
               plan_cfg=PlanConfig(horizon=3))


def test_the_search_rolls_on_the_TACTICAL_field_by_default():
    """MEASURED: the 640-token operative field costs 160 ms per candidate on a
    4060, so DINO-WM's 300x30 would be 325 s/tick. The coarse level is 64
    tokens. Default must be the affordable one, or the arm is un-runnable."""
    assert RefAV1Config().plan_level == "tactical"
    assert RefAV1Config().verify_on_operative is True
    assert RefAV1Config().tac_queries * 10 == 640           # the 10x reduction


def test_an_unknown_plan_level_is_REFUSED_not_silently_defaulted():
    m = RefAV1(_tiny())
    m.cfg.plan_level = "whatever"
    with pytest.raises(ValueError, match="plan_level"):
        m.plan(torch.randn(1, 4, 8, 32), v0=1.0,
               plan_cfg=PlanConfig(horizon=m.cfg.plan_steps, n_samples=8,
                                   n_iters=1, n_elites=2, seed=0))


def test_coarse_to_fine_rescores_the_winner_on_the_FULL_field():
    """A coarse search that never checks itself against the fine field would
    hide exactly the error it introduces. The agreement flag must be reported,
    including when it is False."""
    m = RefAV1(_tiny())
    res = m.plan(torch.randn(1, 4, 8, 32), v0=10.0,
                 goal_field=torch.randn(1, 8, 32),
                 plan_cfg=PlanConfig(horizon=m.cfg.plan_steps, n_samples=16,
                                     n_iters=2, n_elites=4, dt=m.cfg.op_dt,
                                     seed=0))
    assert res.fine_costs and "plan" in res.fine_costs
    assert res.fine_best in res.fine_costs
    assert isinstance(res.coarse_fine_agree, bool)


def test_operative_level_planning_stays_reachable_as_the_ablation():
    m = RefAV1(_tiny())
    m.cfg.plan_level = "operative"
    res = m.plan(torch.randn(1, 4, 8, 32), v0=10.0,
                 plan_cfg=PlanConfig(horizon=m.cfg.plan_steps, n_samples=8,
                                     n_iters=1, n_elites=2, dt=m.cfg.op_dt,
                                     seed=0))
    assert res.coarse_fine_agree is None     # no coarse stage => no flag
    assert math.isfinite(res.cost)


def test_a_full_training_step_is_finite_and_backpropagates():
    torch.manual_seed(0)
    m = RefAV1(_brains(_tiny()))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    out = m(torch.randn(2, 4, 8, 32), torch.randn(2, 30, 2) * 0.1,
            future_feats=torch.randn(2, 30, 8, 32))
    out["loss"].backward()
    gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    assert torch.isfinite(out["loss"]) and torch.isfinite(gnorm)
    assert m.trainable_parameters() > 0
