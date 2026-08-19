"""REF-D — the SimWAM adaptation on our hierarchy.

The tests that matter pin the DESIGN DECISIONS, because each is a place where
the arm could silently become something else: a fan with a selector, a waypoint
head, or an inference path that reads the future.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig  # noqa: E402
from tanitad.refs.refa_v1p import ActionStreamPredictor  # noqa: E402
from tanitad.refs.refd import (PRIOR_GEOMETRY, FlowControlPolicy,  # noqa: E402
                               RefD, RefDConfig)


def _small(**kw):
    c = RefDConfig(d_enc=64, d_state=64, n_tokens=16, op_layers=1, tac_layers=1,
                   op_heads=4, tac_queries=4, plan_steps=8, policy_hidden=32,
                   flow_steps=3)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _full(**kw):
    c = _small(**kw)
    c.tactical_cfg = TacticalPolicyConfig()
    c.strategic_cfg = StrategicPolicyConfig()
    return c


# --------------------------------------------------------------------------- #
# horizons and guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dt,steps", [(0.2, 30), (0.6, 10), (1.5, 4)])
def test_every_rate_reaches_exactly_six_seconds(dt, steps):
    assert abs(dt * steps - 6.0) < 1e-9


def test_a_rate_that_misses_six_seconds_is_refused():
    with pytest.raises(ValueError, match="6.0 s"):
        _small(op_dt=0.25).sanity()


def test_a_state_narrower_than_the_encoder_is_refused():
    with pytest.raises(ValueError, match="change #3"):
        RefDConfig(d_enc=1024, d_state=512).sanity()


# --------------------------------------------------------------------------- #
# the load-bearing decisions
# --------------------------------------------------------------------------- #
def test_the_policy_emits_ONE_sequence_not_a_fan():
    """SEL-1 refuses every selector launch. REF-D does not fix the selector —
    it removes the need for one: the generator IS the policy."""
    c = _small()
    p = FlowControlPolicy(c, d_cond=c.d_state)
    out = p.sample(torch.zeros(3, c.d_state), torch.full((3,), 8.0))
    assert out.shape == (3, c.a_dim, c.plan_steps), out.shape
    assert out.dim() == 3, "a candidate axis would make this a fan again"


def test_the_policy_emits_CONTROLS_within_their_bounds():
    """Controls, never waypoints: a per-waypoint head amplified eps 25x in
    acceleration, and the v5f dense fan was 97.6 % infeasible steps."""
    c = _small()
    p = FlowControlPolicy(c, d_cond=c.d_state)
    out = p.sample(torch.randn(4, c.d_state), torch.full((4,), 12.0))
    assert out[:, 0].abs().max() <= c.a_max + 1e-5
    assert out[:, 1].abs().max() <= c.kappa_max + 1e-5


def test_the_noise_axis_is_DECLARED_and_both_arms_build():
    """OU is our measured preference for diversity; isotropic is what
    Flow-GRPO likelihoods require. The tension is real, so both must exist."""
    for n in ("ou", "iso"):
        c = _small(noise=n)
        c.sanity()
        p = FlowControlPolicy(c, d_cond=c.d_state)
        assert p.sample(torch.zeros(2, c.d_state), torch.zeros(2)).shape[1] == 2
    with pytest.raises(ValueError, match="noise"):
        _small(noise="pink").sanity()


def test_ou_noise_is_ACTUALLY_correlated_and_iso_is_not():
    """The property, MEASURED — not asserted from the formula."""
    torch.manual_seed(0)
    c = _small(plan_steps=256, ou_rho=0.9)
    ou = FlowControlPolicy(c, d_cond=c.d_state)._noise(64, "cpu")
    c2 = _small(plan_steps=256, noise="iso")
    iso = FlowControlPolicy(c2, d_cond=c2.d_state)._noise(64, "cpu")

    def lag1(x):
        a, b = x[:, :, :-1].reshape(-1), x[:, :, 1:].reshape(-1)
        return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))

    assert lag1(ou) > 0.7, lag1(ou)
    assert abs(lag1(iso)) < 0.1, lag1(iso)


def test_the_predictors_are_ACTION_TOKEN_predictors():
    """E-ACTSTREAM-1: token beats broadcast 5.9-9.9x at parameter parity."""
    m = RefD(_full())
    assert isinstance(m.operative, ActionStreamPredictor)
    assert isinstance(m.tactical, ActionStreamPredictor)


def test_the_tactical_heads_are_FACTORED_and_share_v6_tuples_BY_IDENTITY():
    """The retired 5-way mixed softmax must not return through a copy."""
    from tanitad.models import v6
    from tanitad.refs import refd
    assert refd.TACTICAL_LAT_ACTIONS is v6.TACTICAL_LAT_ACTIONS
    assert refd.TACTICAL_LON_ACTIONS is v6.TACTICAL_LON_ACTIONS
    m = RefD(_full())
    assert m.lat_head[-1].out_features == len(v6.TACTICAL_LAT_ACTIONS)
    assert m.lon_head[-1].out_features == len(v6.TACTICAL_LON_ACTIONS)


# --------------------------------------------------------------------------- #
# the isolated mask, as a CHECK not a convention
# --------------------------------------------------------------------------- #
def test_the_deployment_path_CANNOT_accept_a_future_field():
    RefD(_full()).assert_no_future_at_inference()


def test_the_isolation_check_actually_FIRES_on_a_mis_wired_path():
    """A guard that cannot fail is not a guard (the C13 class). Prove it
    rejects the shape it exists to reject."""
    m = RefD(_full())
    m.act = lambda field, v0, future_field=None, gen=None: {}
    with pytest.raises(RuntimeError, match="future"):
        m.assert_no_future_at_inference()


def test_act_runs_and_returns_no_future_keyed_output():
    c = _full()
    m = RefD(c).eval()
    out = m.act(torch.randn(2, c.op_window, c.n_tokens, c.d_enc),
                torch.full((2,), 9.0))
    assert set(out) == {"controls", "lat_logits", "lon_logits", "cond"}
    assert out["controls"].shape == (2, c.a_dim, c.plan_steps)


def test_act_REFUSES_a_3D_input_rather_than_silently_reshaping():
    """WideAdapter mixes temporally over the window; a 3-D input would drop
    that mixing and quietly train a different model."""
    c = _full()
    m = RefD(c).eval()
    with pytest.raises(ValueError, match="B, W, N"):
        m.act(torch.randn(2, c.n_tokens, c.d_enc), torch.full((2,), 9.0))


# --------------------------------------------------------------------------- #
# the prior, and the corrections that travel with it
# --------------------------------------------------------------------------- #
def test_the_prior_is_recorded_as_TRAINING_ONLY():
    assert "FROZEN" in PRIOR_GEOMETRY["role"]
    assert "never shipped" in PRIOR_GEOMETRY["role"]


def test_the_prior_size_is_the_CORRECTED_one():
    """Cosmos3 is Super 64 B / Nano 16 B / Edge 4 B. An earlier programme note
    said Nano was 2 B — that was Cosmos-Dreams, a distilled derivative."""
    assert PRIOR_GEOMETRY["params_b"] == 4.0
    assert "Thor" in PRIOR_GEOMETRY["runs_on"]


def test_the_record_does_not_claim_cosmos_beats_wan():
    """SimWAM Tab. 4 is 90.4 vs 90.3 with no CI — a tie. The choice is Thor,
    not a score."""
    assert "ties" in PRIOR_GEOMETRY["why_not_wan"]


def test_the_flow_loss_is_finite_and_backprops():
    c = _small()
    p = FlowControlPolicy(c, d_cond=c.d_state)
    loss = p.loss(torch.randn(4, c.a_dim, c.plan_steps),
                  torch.randn(4, c.d_state), torch.full((4,), 10.0))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(q.grad is not None and torch.isfinite(q.grad).all()
               for q in p.parameters())
