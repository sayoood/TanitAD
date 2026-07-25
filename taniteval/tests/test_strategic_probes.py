"""HP-3 counterfactual route-swap probe — tests on SYNTHETIC models.

The probe's whole value is that it **discriminates**: a flat marginal model
must score ~0 divergence and a route-conditional one must score > 0. That is
not something to assert about the code — it is something to demonstrate, so the
fixtures below are two models that differ *only* in whether the strategic
command reaches the trajectory:

* :class:`FlatModel` — the strategic pass exists, the route logits follow the
  command perfectly (the **echo**), and the waypoints ignore it entirely. This
  is today's arms by construction (``route_target = _NAV_TO_ROUTE[nav_cmd]``).
  It must score **zero** divergence and **fail** HP-3.
* :class:`RouteConditionalModel` — the same, plus the command steers the
  waypoints laterally. It must score **> 0** divergence, **in the commanded
  direction**, and **pass**.

A probe that cannot tell these two apart is worthless, so both directions are
pinned. The echo control is pinned too: ``FlatModel`` scores ~1.0 on
``route_head_echo`` while diverging by 0 — the exact signature HPP-1 exists to
fix, and the reason a high route accuracy must never be read as route
understanding.

No GPU, no checkpoint, no pod. The real invocation is documented in
``strategic_probes.INVOCATION``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import refb_labels as rl

from taniteval import driving as drv
from taniteval import strategic_probes as SP

WP = SP.WP_STEPS
S_DIM, C_DIM, I_DIM = 16, 8, 8


# --------------------------------------------------------------------------- #
# synthetic arms                                                                #
# --------------------------------------------------------------------------- #
class _Strategic(torch.nn.Module):
    """ctx + route logits. The route logits ALWAYS echo the command — that is
    what every arm trained on the v1 labeler does, and the probe must not
    mistake it for route understanding."""

    def __init__(self, route_in_ctx=0.0):
        super().__init__()
        self.route_in_ctx = float(route_in_ctx)
        self.nav_emb = torch.nn.Embedding(4, C_DIM)

    def forward(self, states, nav):
        b = states.shape[0]
        base = states.mean(1)[:, :C_DIM]
        ctx = base + self.route_in_ctx * self.nav_emb(nav)
        # perfect echo: logits = one-hot of _NAV_TO_ROUTE[nav]
        route = torch.zeros(b, 3)
        for i in range(b):
            route[i, rl.route_target(int(nav[i]))] = 10.0
        return {"ctx": ctx, "route_logits": route}


class _Tactical(torch.nn.Module):
    """Waypoints, maneuver, intent. ``lat_gain`` decides whether the strategic
    ctx reaches the *trajectory* — the single knob that separates a flat
    marginal model from a route-conditional one."""

    def __init__(self, lat_gain=0.0):
        super().__init__()
        self.lat_gain = float(lat_gain)

    def forward(self, states, ctx):
        b = states.shape[0]
        v = states.mean(1)[:, 0].abs() + 5.0                 # a plausible speed
        # ctx[:, 0] carries the command's signature when route_in_ctx > 0
        lat = self.lat_gain * ctx[:, 0]
        wp = {}
        for j, k in enumerate(WP):
            t = (j + 1) / len(WP)
            wp[k] = torch.stack([v * t * 2.0, lat * t], dim=-1)
        man = torch.zeros(b, 5)
        man[:, 0] = 1.0
        if self.lat_gain:                    # a route-conditional arm also
            man[:, 1] = (lat > 0.05).float() * 5.0     # switches its maneuver
            man[:, 2] = (lat < -0.05).float() * 5.0
        return {"waypoints": wp, "intent": ctx[:, :I_DIM].contiguous(),
                "maneuver_logits": man,
                "target_latent": states[:, -1]}


class _Arm(torch.nn.Module):
    def __init__(self, lat_gain=0.0, route_in_ctx=0.0):
        super().__init__()
        self.strategic_policy = _Strategic(route_in_ctx)
        self.tactical_policy = _Tactical(lat_gain)

    def encode_window(self, fw):
        return fw.reshape(fw.shape[0], fw.shape[1], -1)[..., :S_DIM]

    def eval(self):
        return self


def FlatModel():
    """Route logits echo the command; the trajectory ignores it."""
    return _Arm(lat_gain=0.0, route_in_ctx=1.0)


def RouteConditionalModel(gain=1.5):
    """The command reaches the trajectory, laterally, in the right direction."""
    return _Arm(lat_gain=gain, route_in_ctx=1.0)


class _Ep:
    def __init__(self, eid, T=60, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.episode_id = eid
        self.feats = torch.rand(T, S_DIM, generator=g)
        self.actions = torch.rand(T, 2, generator=g) * 0.1
        self.poses = torch.zeros(T, 4)
        self.poses[:, 0] = torch.arange(T, dtype=torch.float32) * 1.0
        self.poses[:, 3] = 10.0


def _episodes(n=12, T=60):
    return [_Ep(i, T, seed=i) for i in range(n)]


def _sign_the_ctx(model):
    """Make the nav embedding carry +1 for LEFT and -1 for RIGHT on channel 0,
    so a nonzero ``lat_gain`` steers the commanded way. This is the *ground
    truth* the probe's direction check must recover."""
    w = model.strategic_policy.nav_emb.weight.data
    w.zero_()
    w[rl.NAV_LEFT, 0] = +1.0
    w[rl.NAV_RIGHT, 0] = -1.0
    w[rl.NAV_FOLLOW, 0] = 0.0
    return model


# ========================================================================== #
# 1. THE DISCRIMINATION — the probe's reason to exist                          #
# ========================================================================== #
@pytest.fixture(scope="module")
def flat():
    return SP.run(_sign_the_ctx(FlatModel()), _episodes(), "cpu", n_boot=300)


@pytest.fixture(scope="module")
def conditional():
    return SP.run(_sign_the_ctx(RouteConditionalModel()), _episodes(), "cpu",
                  n_boot=300)


def test_a_flat_model_scores_zero_divergence_and_FAILS(flat):
    """A model whose output ignores nav_cmd must score ~0 — by construction."""
    lr = flat["divergence"]["left_vs_right"]
    assert lr["cross_track_2s_m"]["mean"] == 0.0
    assert lr["wp_l2_mean_m"]["mean"] == 0.0
    assert lr["maneuver_changed_rate"]["mean"] == 0.0
    assert flat["HP3_route_conditional"] is False
    assert flat["HP3_divergence_separated"] is False
    assert "HP-3 FAILS" in flat["verdict"]
    assert "PC1 regression" in flat["verdict"]


def test_a_route_conditional_model_scores_above_zero_and_PASSES(conditional):
    lr = conditional["divergence"]["left_vs_right"]
    assert lr["cross_track_2s_m"]["mean"] > SP.MIN_DIVERGENCE_M
    assert lr["cross_track_2s_m"]["lo"] > 0
    assert lr["wp_l2_mean_m"]["mean"] > 0
    assert conditional["HP3_divergence_separated"] is True
    assert conditional["HP3_direction_correct"] is True
    assert conditional["HP3_route_conditional"] is True
    assert "HP-3 PASSES" in conditional["verdict"]


def test_the_two_arms_are_separated_by_the_paired_test(flat, conditional):
    """The arm-vs-arm form: paired on the same windows, MORE divergence wins."""
    n = flat["n_windows"]
    eid = [str(i // 6) for i in range(n)]
    zero = np.zeros(n)
    big = np.full(n, 1.0)
    d = SP.paired_hp3_delta(big, zero, eid, n_boot=300)
    assert d["estimator"] == "paired_episode_cluster_bootstrap"
    assert d["delta"] > 0 and d["separated"] is True
    assert "MORE divergence is BETTER" in d["_orientation"]


def test_divergence_scales_with_the_route_gain():
    a = SP.run(_sign_the_ctx(RouteConditionalModel(0.5)), _episodes(), "cpu",
               n_boot=100)
    b = SP.run(_sign_the_ctx(RouteConditionalModel(2.0)), _episodes(), "cpu",
               n_boot=100)
    assert (b["divergence"]["left_vs_right"]["cross_track_2s_m"]["mean"]
            > 3 * a["divergence"]["left_vs_right"]["cross_track_2s_m"]["mean"])


# ========================================================================== #
# 2. Divergence WITHOUT correctness must not pass                              #
# ========================================================================== #
def test_divergence_in_the_WRONG_direction_does_not_pass():
    """A model that moves under the command but the WRONG way is not
    route-following. HP-3 asks for a *different, CORRECT* trajectory."""
    m = _sign_the_ctx(RouteConditionalModel(-1.5))     # inverted response
    out = SP.run(m, _episodes(), "cpu", n_boot=300)
    assert out["HP3_divergence_separated"] is True
    assert out["HP3_direction_correct"] is False
    assert out["HP3_route_conditional"] is False
    assert "HP-3 PARTIAL" in out["verdict"]


def test_direction_score_is_measured_against_chance_not_zero(conditional, flat):
    assert conditional["direction"]["chance"] == 0.5
    assert conditional["direction"]["score"]["mean"] == 1.0
    assert conditional["direction"]["separated_above_chance"] is True
    # a flat model's lateral delta is exactly 0 -> sign 0 -> never "correct"
    assert flat["direction"]["separated_above_chance"] is False


def test_signed_lateral_deltas_point_the_commanded_way(conditional):
    d = conditional["direction"]
    assert d["signed_lateral_delta_left_m"]["mean"] > 0, "left => +y (ego)"
    assert d["signed_lateral_delta_right_m"]["mean"] < 0, "right => -y (ego)"


# ========================================================================== #
# 3. The ECHO control — a high route accuracy is NOT route understanding       #
# ========================================================================== #
def test_flat_model_echoes_the_command_perfectly_while_diverging_by_zero(flat):
    """The exact signature HPP-1 fixes, in one assertion."""
    echo = flat["route_head_echo"]
    assert echo["route_logit_follows_command_rate"] == 1.0
    assert all(v == 1.0 for v in echo["by_branch"].values())
    assert flat["divergence"]["left_vs_right"]["cross_track_2s_m"]["mean"] == 0.0
    assert "BY CONSTRUCTION" in echo["_read"]


def test_branch_route_targets_expose_the_circular_map():
    assert SP.BRANCH_ROUTE["left"] == rl.route_target(rl.NAV_LEFT)
    assert SP.BRANCH_ROUTE["right"] == rl.route_target(rl.NAV_RIGHT)
    assert SP.BRANCH_ROUTE["follow"] == rl.route_target(rl.NAV_FOLLOW)
    assert SP.BRANCH_ROUTE["left"] != SP.BRANCH_ROUTE["right"]


# ========================================================================== #
# 4. Structure, estimator and refusals                                         #
# ========================================================================== #
def test_one_encode_three_strategic_passes():
    """The probe must not re-encode per branch — that is the whole cost claim."""
    m = _sign_the_ctx(RouteConditionalModel())
    calls = {"n": 0}
    orig = m.encode_window

    def counted(fw):
        calls["n"] += 1
        return orig(fw)
    m.encode_window = counted
    strat_calls = {"n": 0}
    orig_s = m.strategic_policy.forward

    def counted_s(states, nav):
        strat_calls["n"] += 1
        return orig_s(states, nav)
    m.strategic_policy.forward = counted_s
    SP.run(m, _episodes(n=4), "cpu", n_boot=50)
    assert strat_calls["n"] == 3 * calls["n"], \
        "exactly three strategic passes per encode"


def test_all_intervals_are_the_decision_grade_estimator(conditional):
    drv.assert_no_deprecated_estimator(conditional, _path="hp3")
    lr = conditional["divergence"]["left_vs_right"]
    assert lr["cross_track_2s_m"]["estimator"] == "episode_cluster_bootstrap"
    assert lr["cross_track_2s_m"]["n_episodes"] == conditional["n_episodes"] == 12


def test_every_branch_pair_is_reported(conditional):
    for tag in ("left_vs_right", "left_vs_follow", "right_vs_follow"):
        assert tag in conditional["divergence"], tag
        row = conditional["divergence"][tag]
        for k in ("wp_l2_mean_m", "cross_track_2s_m", "cross_track_2s_p90_m",
                  "cross_track_tail", "ctx_cosine", "intent_cosine",
                  "maneuver_changed_rate"):
            assert k in row, f"{tag}.{k}"


def test_the_lateral_channel_is_the_headline(conditional):
    """M6: HP-3 is measured in the cross-track channel, not in ADE."""
    v = conditional["verdict"]
    assert "cross-track@2s" in v
    lr = conditional["divergence"]["left_vs_right"]
    # the cross-track divergence is a COMPONENT of the L2 at the same step, so
    # it can never exceed it. (It CAN exceed `wp_l2_mean_m`, which averages over
    # the horizon and is therefore a different quantity — comparing against that
    # is the mistake this assertion exists to prevent.)
    assert lr["cross_track_2s_m"]["mean"] <= lr["wp_l2_final_m"]["mean"] + 1e-6
    # ...and in a pure-lateral route response the two are EQUAL: the route
    # effect lives entirely in the channel M6 says to measure it in.
    assert lr["cross_track_2s_m"]["mean"] == pytest.approx(
        lr["wp_l2_final_m"]["mean"], rel=1e-4)


def test_an_arm_without_a_strategic_level_is_SKIPPED_not_passed():
    class NoStrategic:
        strategic_policy = None
        tactical_policy = None

        def eval(self):
            return self
    out = SP.run(NoStrategic(), _episodes(n=2), "cpu")
    assert "skipped" in out
    assert "A SKIP IS NOT A PASS" in out["skipped"]
    assert "HP3_route_conditional" not in out


def test_grounded_diagnostic_requires_a_step_readout():
    with pytest.raises(ValueError, match="step_readout"):
        SP.run(_sign_the_ctx(FlatModel()), _episodes(n=2), "cpu", grounded=True)


def test_pc2_note_is_emitted_and_the_scored_surface_is_declared(flat):
    """A zero here must never be read as a model verdict while the scored
    rollout is structurally intent-free."""
    assert flat["route_can_reach_scored_trajectory"] is False
    assert "structurally CANNOT" in flat["_pc2_note"]


def test_invocation_is_documented_for_when_a_pod_frees():
    inv = SP.INVOCATION
    assert "taniteval.strategic_probes" in inv
    assert "PYTHONPATH=/root/TanitAD/stack" in inv
    assert "_NAV_TO_ROUTE" in inv, "the label-circularity caveat must be in it"
    assert "EXPECTED, pre-registered outcome" in inv
