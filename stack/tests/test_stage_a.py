"""W3 ``stage_a_probes`` pure-part smoke — CPU-only, no checkpoint, no
corpus. The full path (frozen v5f trunk + v2 val corpus + GPU roll) is
pod-side and NOT exercised here.

What is pinned (the task's six groups):
  * the analytic unicycle reference on a hand case — straight-line constant
    accel integrated with the W4 discretisation (pre-update speed, translate
    then turn), plus the left-turn sign of a positive-kappa roll;
  * the steer<->kappa encoding round trip (steer = atan(WB*kappa),
    physicalai.py:621) and the counterfactual construction: kappa-space
    deltas through the encoding, only the LAST window action + futures
    perturbed, hold = zero-order-hold (pinned EQUAL to
    ``rollout_transitions(..., future_actions=None)`` on a mock predictor);
  * sign-metric logic on synthetic decodes: all-correct, all-wrong, mixed,
    and the admissibility exclusion (|analytic| < eps is not scored);
  * response-gain arithmetic on a hand case;
  * PCA subspace stat on synthetic deltas of KNOWN rank (dims for 80 % <=
    the planted rank; energy fraction ~1 inside the subspace) + the two
    not-computable branches;
  * gate-JSON branches: all-pass, sign-fail, empty channel -> pass None,
    P6-dim fail, and the hold row carrying no gate;
  * the 3-channel action-lift shape contract (``lift_actions3``, the canary
    speed-append) on mock tensors.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage_a_probes import (CF_CHANNELS, CHANNEL_SIGN_AXIS,  # noqa: E402
                            DT, EPS_RESP, GATE_P6_DIMS, GATED_CHANNELS,
                            WHEELBASE, analytic_endpoints,
                            apply_counterfactual, channel_stats,
                            kappa_of_steer, pca_subspace_stats,
                            rolled_action_sequence, steer_of_kappa,
                            w3_gate_dict)

B, W, H = 2, 8, 20


def _actions(steer=0.0, accel=0.0):
    """Constant recorded actions: aw2 [B, W, 2], fa2 [B, H, 2]."""
    aw2 = torch.zeros(B, W, 2)
    fa2 = torch.zeros(B, H, 2)
    aw2[..., 0] = steer
    aw2[..., 1] = accel
    fa2[..., 0] = steer
    fa2[..., 1] = accel
    return aw2, fa2


# ---------------------------------------------------------------------------
# analytic unicycle reference — hand case (the W4 discretisation)
# ---------------------------------------------------------------------------
def test_analytic_hand_case_straight_constant_accel():
    """v0=5, a=1, kappa=0, k=3: pre-update speeds are 5, 5.1, 5.2 so
    x = [0.5, 1.01, 1.53] (dx_k = v_k*dt BEFORE the speed update — the
    train_unicycle_readout convention the W4 head banked), y = 0."""
    aw2, fa2 = _actions(steer=0.0, accel=1.0)
    v0 = torch.full((B,), 5.0)
    wp = analytic_endpoints(aw2, fa2, v0, k=3)
    assert wp.shape == (B, 3, 2)
    assert torch.allclose(wp[:, :, 0],
                          torch.tensor([0.5, 1.01, 1.53]).expand(B, 3),
                          atol=1e-6)
    assert torch.allclose(wp[:, :, 1], torch.zeros(B, 3), atol=1e-9)


def test_analytic_positive_kappa_turns_left():
    """kappa > 0 (ego frame +y left) ends left of the centreline, and the
    lateral magnitude matches the small-angle expectation to first order."""
    kappa = 0.05
    aw2, fa2 = _actions(steer=float(steer_of_kappa(torch.tensor(kappa))))
    v0 = torch.full((B,), 10.0)
    wp = analytic_endpoints(aw2, fa2, v0, k=10)
    y_end = wp[0, -1, 1].item()
    assert y_end > 0.0
    # small-angle: y ~ 0.5 * kappa * (v*t)^2 = 0.5*0.05*100 = 2.5 m (loose)
    assert 1.5 < y_end < 3.5


def test_rolled_action_sequence_layout():
    """Step 0 = last WINDOW action, steps 1..k-1 = future actions — the
    exact order rollout_transitions consumes (metric_dynamics.py:258-261)."""
    aw2, fa2 = _actions()
    aw2[:, -1, 1] = 3.0                       # present action
    fa2[:, 0, 1] = 1.0
    fa2[:, 1, 1] = 2.0
    a_seq, kappa_seq = rolled_action_sequence(aw2, fa2, k=3)
    assert a_seq.shape == kappa_seq.shape == (B, 3)
    assert torch.allclose(a_seq[0], torch.tensor([3.0, 1.0, 2.0]))
    with pytest.raises(ValueError):
        rolled_action_sequence(aw2, fa2[:, :1], k=3)


# ---------------------------------------------------------------------------
# steer encoding + counterfactual construction
# ---------------------------------------------------------------------------
def test_steer_kappa_roundtrip():
    k = torch.linspace(-0.2, 0.2, 9)
    assert torch.allclose(kappa_of_steer(steer_of_kappa(k)), k, atol=1e-6)
    # the encoding really is atan(2.9 * kappa)
    assert torch.allclose(steer_of_kappa(torch.tensor(0.1)),
                          torch.atan(torch.tensor(0.29)))


def test_counterfactual_left_right_kappa_space():
    """Left adds +dkappa IN KAPPA SPACE (through the encoding), right the
    negative; only the last window action + futures move."""
    aw2, fa2 = _actions(steer=0.05, accel=0.5)
    for ch, s in (("left", +1.0), ("right", -1.0)):
        awc, fac = apply_counterfactual(aw2, fa2, ch, dkappa=0.02)
        assert awc.shape == aw2.shape and fac.shape == fa2.shape
        # kappa delta is exactly +-0.02 on the perturbed entries
        dk = kappa_of_steer(fac[..., 0]) - kappa_of_steer(fa2[..., 0])
        assert torch.allclose(dk, torch.full_like(dk, s * 0.02), atol=1e-6)
        dk_last = (kappa_of_steer(awc[:, -1, 0])
                   - kappa_of_steer(aw2[:, -1, 0]))
        assert torch.allclose(dk_last, torch.full_like(dk_last, s * 0.02),
                              atol=1e-6)
        # history + accel channel untouched
        assert torch.equal(awc[:, :-1], aw2[:, :-1])
        assert torch.equal(awc[..., 1], aw2[..., 1])
        assert torch.equal(fac[..., 1], fa2[..., 1])


def test_counterfactual_brake_throttle_and_bad_channel():
    aw2, fa2 = _actions(accel=0.5)
    awc, fac = apply_counterfactual(aw2, fa2, "brake", daccel=2.0)
    assert torch.allclose(awc[:, -1, 1], torch.full((B,), -1.5))
    assert torch.allclose(fac[..., 1], torch.full_like(fa2[..., 1], -1.5))
    assert torch.equal(awc[..., 0], aw2[..., 0])       # steer untouched
    awc, fac = apply_counterfactual(aw2, fa2, "throttle", daccel=2.0)
    assert torch.allclose(fac[..., 1], torch.full_like(fa2[..., 1], 2.5))
    with pytest.raises(ValueError):
        apply_counterfactual(aw2, fa2, "swerve")


class _MockPredictor(torch.nn.Module):
    """1-step head contract: predictor(win_s, win_a) -> (_, z_hat) where
    z_hat depends on the LAST state and LAST action."""

    def forward(self, win_s, win_a):
        z = 0.9 * win_s[:, -1] + win_a[:, -1].sum(-1, keepdim=True)
        return None, z


def test_hold_equals_zero_order_hold_roll():
    """The HOLD counterfactual is pinned byte-equal to rolling with
    future_actions=None (rollout_transitions' own zero-order-hold branch,
    metric_dynamics.py:259-260)."""
    from tanitad.models.metric_dynamics import rollout_transitions
    torch.manual_seed(0)
    pred = _MockPredictor()
    states = torch.randn(B, W, 4)
    aw2, fa2 = _actions(steer=0.03, accel=1.0)
    fa2[:, :, 1] = torch.randn(B, H)          # futures differ from last action
    _awh, fah = apply_counterfactual(aw2, fa2, "hold")
    aw3 = torch.cat([aw2, torch.ones(B, W, 1)], dim=-1)
    fah3 = torch.cat([fah, torch.ones(B, H, 1)], dim=-1)
    t_hold = rollout_transitions(pred, states, aw3, fah3, 5)
    t_zoh = rollout_transitions(pred, states, aw3, None, 5)
    for (za, zb), (zc, zd) in zip(t_hold, t_zoh):
        assert torch.equal(za, zc) and torch.equal(zb, zd)


# ---------------------------------------------------------------------------
# sign metric + gain arithmetic
# ---------------------------------------------------------------------------
def test_sign_metric_logic():
    d_an = np.full(10, 0.5)                    # all admissible, positive
    st = channel_stats(np.full(10, 0.2), d_an, expected_sign=+1.0)
    assert st["sign_rate"] == 1.0 and st["n_admissible"] == 10
    st = channel_stats(np.full(10, -0.2), d_an, expected_sign=+1.0)
    assert st["sign_rate"] == 0.0
    mixed = np.array([0.1, -0.1] * 5)
    st = channel_stats(mixed, d_an, expected_sign=+1.0)
    assert st["sign_rate"] == 0.5
    # negative expected sign (right/brake)
    st = channel_stats(np.full(4, -0.3), -d_an[:4], expected_sign=-1.0)
    assert st["sign_rate"] == 1.0


def test_sign_metric_admissibility_exclusion():
    """|analytic| < eps windows are excluded AND counted, never scored."""
    d_an = np.array([0.5, 0.5, EPS_RESP / 10, 0.0])
    d_wm = np.array([0.2, -0.2, -9.0, -9.0])   # the two inadmissible are wrong
    st = channel_stats(d_wm, d_an, expected_sign=+1.0)
    assert st["n_admissible"] == 2
    assert st["n_excluded_no_analytic_response"] == 2
    assert st["sign_rate"] == 0.5              # only the admissible pair
    st0 = channel_stats(d_wm[2:], d_an[2:], expected_sign=+1.0)
    assert st0["sign_rate"] is None and st0["gain_median"] is None
    with pytest.raises(ValueError):
        channel_stats(d_wm, d_an[:2], expected_sign=+1.0)


def test_gain_hand_case():
    """|d_wm|/|d_an|: doubled response -> median 2.0 (band edge, inclusive
    in the gate); halved -> 0.5."""
    d_an = np.array([0.5, -1.0, 2.0])
    st = channel_stats(2.0 * d_an, d_an, expected_sign=+1.0)
    assert st["gain_median"] == pytest.approx(2.0)
    st = channel_stats(0.5 * d_an, d_an, expected_sign=+1.0)
    assert st["gain_median"] == pytest.approx(0.5)
    assert st["gain_p25"] <= st["gain_median"] <= st["gain_p75"]


# ---------------------------------------------------------------------------
# P6 PCA subspace — synthetic known rank
# ---------------------------------------------------------------------------
def test_pca_known_rank():
    """Deltas planted in a 3-dim subspace of S=64: dims for 80 % <= 3 and
    every channel's energy fraction inside the subspace is ~1."""
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((64, 3)))[0].T   # [3, 64]
    deltas = {c: rng.standard_normal((50, 3)) @ basis
              for c in ("left", "right", "brake")}
    st = pca_subspace_stats(deltas)
    assert st["computable"] is True
    assert st["n_rows"] == 150 and st["latent_dim"] == 64
    assert st["dims_for_var_target"] <= 3
    for c in deltas:
        assert st["energy_fraction_in_subspace_per_channel"][c] == \
            pytest.approx(1.0, abs=1e-9)


def test_pca_full_rank_needs_many_dims():
    rng = np.random.default_rng(1)
    deltas = {"left": rng.standard_normal((200, 64))}
    st = pca_subspace_stats(deltas)
    assert st["dims_for_var_target"] > GATE_P6_DIMS   # isotropic noise fails


def test_pca_not_computable_branches():
    st = pca_subspace_stats({})
    assert st["computable"] is False
    st = pca_subspace_stats({"left": np.zeros((10, 8))})
    assert st["computable"] is False and "P3" in st["reason"]


# ---------------------------------------------------------------------------
# gate JSON — every branch
# ---------------------------------------------------------------------------
def _chan(sign_rate=0.99, gain=1.0, n=100):
    return {"n_grid": n, "n_admissible": n,
            "n_excluded_no_analytic_response": 0,
            "sign_rate": sign_rate, "gain_median": gain,
            "gain_p25": gain, "gain_p75": gain}


def _p6(dims=10):
    return {"computable": True, "n_rows": 400, "latent_dim": 64,
            "var_target": 0.8, "dims_for_var_target": dims,
            "energy_fraction_in_subspace_per_channel": {"left": 0.95}}


def test_gate_all_pass():
    g = w3_gate_dict({c: _chan() for c in GATED_CHANNELS}, _p6(10),
                     hold={"n_grid": 100, "endpoint_delta_median_m": 0.02,
                           "endpoint_delta_p90_m": 0.1})
    assert g["PASS"] is True
    for c in GATED_CHANNELS:
        assert g["channels"][c]["sign_gate"]["pass"] is True
        assert g["channels"][c]["gain_gate"]["pass"] is True
    assert g["p6"]["gate"]["pass"] is True
    assert "gate" not in g["channels"]["hold"]           # hold carries none
    assert "no sign/gain gate" in g["channels"]["hold"]["note"]


def test_gate_sign_fail_and_gain_fail():
    ch = {c: _chan() for c in GATED_CHANNELS}
    ch["left"] = _chan(sign_rate=0.90)                   # < 0.95
    g = w3_gate_dict(ch, _p6())
    assert g["channels"]["left"]["sign_gate"]["pass"] is False
    assert g["PASS"] is False
    ch["left"] = _chan(gain=2.5)                         # outside [0.5, 2.0]
    g = w3_gate_dict(ch, _p6())
    assert g["channels"]["left"]["gain_gate"]["pass"] is False
    assert g["PASS"] is False
    ch["left"] = _chan(gain=2.0)                         # band edge inclusive
    g = w3_gate_dict(ch, _p6())
    assert g["channels"]["left"]["gain_gate"]["pass"] is True


def test_gate_not_computable_channel_is_none_not_fake():
    ch = {c: _chan() for c in GATED_CHANNELS}
    ch["brake"] = {"n_grid": 50, "n_admissible": 0,
                   "n_excluded_no_analytic_response": 50,
                   "sign_rate": None, "gain_median": None}
    g = w3_gate_dict(ch, _p6())
    assert g["channels"]["brake"]["sign_gate"]["pass"] is None
    assert g["channels"]["brake"]["gain_gate"]["pass"] is None
    assert g["PASS"] is None                             # None, never a fake
    # missing channel entirely -> same
    g = w3_gate_dict({c: _chan() for c in ("left", "right", "throttle")},
                     _p6())
    assert g["PASS"] is None


def test_gate_p6_dim_fail_and_not_computable():
    ch = {c: _chan() for c in GATED_CHANNELS}
    g = w3_gate_dict(ch, _p6(dims=GATE_P6_DIMS + 1))
    assert g["p6"]["gate"]["pass"] is False and g["PASS"] is False
    g = w3_gate_dict(ch, {"computable": False, "reason": "no deltas"})
    assert g["p6"]["gate"]["pass"] is None and g["PASS"] is None
    # a False anywhere beats a None (fail dominates not-computable)
    ch["left"] = _chan(sign_rate=0.5)
    g = w3_gate_dict(ch, {"computable": False, "reason": "no deltas"})
    assert g["PASS"] is False


# ---------------------------------------------------------------------------
# action-lift shape contract (the canary 3-channel speed-append, mocked)
# ---------------------------------------------------------------------------
def test_action_lift_shape_contract():
    """``lift_actions3`` (train_p8_occupancy — the canary pattern,
    train_flagship_v4.py:578-580): [B,W,2]+[B,H,2]+v0 -> [B,W,3]+[B,H,3]
    with channel 2 == v0/SPEED_SCALE constant along the horizon."""
    from train_p8_occupancy import lift_actions3
    from tanitad.models.flagship_v15 import SPEED_SCALE
    aw2, fa2 = _actions(steer=0.1, accel=0.5)
    v0 = torch.tensor([4.0, 8.0])
    aw3, fa3 = lift_actions3(aw2, fa2, v0)
    assert aw3.shape == (B, W, 3) and fa3.shape == (B, H, 3)
    assert torch.equal(aw3[..., :2], aw2) and torch.equal(fa3[..., :2], fa2)
    for t, n in ((aw3, W), (fa3, H)):
        assert torch.allclose(t[..., 2],
                              (v0 / SPEED_SCALE)[:, None].expand(B, n))


def test_channel_tables_consistent():
    """The sign/axis table covers exactly the gated channels; hold is a
    counterfactual but not gated."""
    assert set(CHANNEL_SIGN_AXIS) == set(GATED_CHANNELS)
    assert set(GATED_CHANNELS) | {"hold"} == set(CF_CHANNELS)
    assert DT == 0.1 and WHEELBASE == 2.9
