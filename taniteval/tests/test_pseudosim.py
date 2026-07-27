"""``taniteval.pseudosim`` — the protocol that makes closed-loop a MEASUREMENT.

WHAT THESE TESTS EXIST TO STOP
------------------------------
The whole claim of pseudo-simulation is *"0 % out of envelope BY
CONSTRUCTION"*. A claim like that is worth exactly as much as the assertion
behind it, and this program has shipped **three vacuous diagnostics** — guards
whose criterion could not fire. So the suite is deliberately two-directional:

* :func:`test_envelope_assertion_PASSES_on_the_shipped_grid` — the positive side;
* :func:`test_envelope_assertion_FAILS_just_outside_the_envelope` — ⭐ **the
  negative side. It feeds a grid 0.5 deg outside ``ENV_YAW_MAX`` and requires
  ``EnvelopeViolation``.** If this test ever passes trivially, the assertion has
  become decoration;
* :func:`test_the_falsifier_is_published_and_is_the_value_that_actually_fails` —
  the module *states* which value breaks it, and the test checks that the stated
  value really does;
* :func:`test_lateral_axis_is_refused_by_default` — the L-BAD verdict is
  enforced in code, not only in prose;
* :func:`test_composite_REFUSES_to_emit_when_every_component_saturates` — a
  metric that cannot fail is not emitted at all.

CPU only: a synthetic 3-episode corpus and a deterministic stub planner. No
checkpoint, no GPU, no pod, no corpus.
"""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from taniteval import ood as _ood
from taniteval import pseudosim as PS


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _episodes(n_ep=3, T=60, seed=0):
    """A synthetic corpus: straight-ish driving at ~10 m/s, 10 Hz."""
    g = torch.Generator().manual_seed(seed)
    eps = []
    for e in range(n_ep):
        yaw = torch.linspace(0.0, 0.15 * (e + 1), T)
        v = torch.full((T,), 10.0)
        x = torch.cumsum(v * torch.cos(yaw) * 0.1, 0)
        y = torch.cumsum(v * torch.sin(yaw) * 0.1, 0)
        poses = torch.stack([x, y, yaw, v], -1)
        frames = (torch.rand(T, 3, 16, 16, generator=g) * 255).to(torch.uint8)
        eps.append(SimpleNamespace(poses=poses, frames=frames))
    return eps


class _StubPlanner:
    """Deterministic: drive straight ahead at ``v0``, optionally steering back
    toward the reference by ``recover`` (0 = no recovery, 1 = full)."""

    def __init__(self, recover=0.0, horizon=20):
        self.recover, self.horizon = float(recover), int(horizon)
        self.calls = 0

    def traj(self, fw, v0, goal):
        b = fw.shape[0]
        self.calls += 1
        t = torch.arange(1, self.horizon + 1, dtype=torch.float32) * 0.1
        x = v0[:, None].cpu() * t[None]
        y = torch.zeros_like(x)
        return torch.stack([x, y], -1)


def _frames_of(ep, a, b):
    return ep.frames[a:b].float().div(255.0)


# --------------------------------------------------------------------------- #
# the envelope assertion — BOTH directions                                     #
# --------------------------------------------------------------------------- #
def test_envelope_assertion_PASSES_on_the_shipped_grid():
    proof = PS.assert_grid_in_envelope(PS.default_grid())
    assert proof["EXTRAPOLATION_frac_steps_any"] == 0.0
    assert proof["EXTRAPOLATION_frac_windows_any_step_out_of_envelope"] == 0.0
    assert _ood.verdict_class(proof["EXTRAPOLATION_VERDICT"]) == \
        _ood.CLASS_MEASUREMENT
    assert proof["ratio_is_lower_bound"] is False
    # the shipped grid uses the full measured heading envelope and no more
    assert proof["max_abs_dyaw_deg"] == pytest.approx(_ood.ENV_YAW_MAX)
    assert proof["max_abs_dlat_m"] == 0.0


def test_envelope_assertion_FAILS_just_outside_the_envelope():
    """⭐ THE TEST THAT PROVES THE ASSERTION IS NOT DECORATION."""
    bad = PS.GridSpec(dyaw_deg=(0.0, _ood.ENV_YAW_MAX + 0.5), dlon_steps=(0,))
    with pytest.raises(PS.EnvelopeViolation) as exc:
        PS.assert_grid_in_envelope(bad)
    assert "OUTSIDE the MEASURED envelope" in str(exc.value)


def test_the_falsifier_is_published_and_is_the_value_that_actually_fails():
    """The module publishes what would break it; the claim is then checked."""
    proof = PS.assert_grid_in_envelope(PS.default_grid())
    f = proof["falsifier"]
    assert f["smallest_failing_abs_dyaw_deg"] > _ood.ENV_YAW_MAX
    assert f["smallest_failing_abs_dlat_m"] > _ood.ENV_LAT_MAX
    # exactly at the edge is INSIDE (the envelope is inclusive) …
    PS.assert_grid_in_envelope(PS.GridSpec(dyaw_deg=(_ood.ENV_YAW_MAX,),
                                           dlon_steps=(0,)))
    # … and a hair beyond it is not.
    with pytest.raises(PS.EnvelopeViolation):
        PS.assert_grid_in_envelope(
            PS.GridSpec(dyaw_deg=(_ood.ENV_YAW_MAX * 1.0001,), dlon_steps=(0,)))


def test_envelope_assertion_runs_before_any_model_is_touched():
    """A bad grid must cost zero planner calls."""
    p = _StubPlanner()
    bad = PS.GridSpec(dyaw_deg=(_ood.ENV_YAW_MAX + 1.0,), dlon_steps=(0,))
    with pytest.raises(PS.EnvelopeViolation):
        PS.pseudo_evaluate(p, _episodes(), bad, frames_of=_frames_of)
    assert p.calls == 0


# --------------------------------------------------------------------------- #
# the lateral refusal — the L-BAD verdict, enforced in code                    #
# --------------------------------------------------------------------------- #
def test_lateral_axis_is_refused_by_default():
    with pytest.raises(PS.LateralAxisRefused) as exc:
        PS.GridSpec(dlat_m=(0.0, 1.0))
    assert "flat-road" in str(exc.value)
    assert "L-BAD" in str(exc.value)


def test_lateral_axis_needs_an_explicit_reason_not_just_a_flag():
    with pytest.raises(PS.LateralAxisRefused):
        PS.GridSpec(dlat_m=(0.0, 1.0), allow_lateral=True)      # no reason
    g = PS.GridSpec(dlat_m=(0.0, 1.0), allow_lateral=True,
                    lateral_override_reason="deliberate infidelity study")
    assert g.n_points == 2 * len(g.dyaw_deg) * len(g.dlon_steps)
    # an override does NOT bypass the envelope: 1.0 m is still inside 3.0 m
    assert PS.assert_grid_in_envelope(g)["EXTRAPOLATION_frac_steps_any"] == 0.0


# --------------------------------------------------------------------------- #
# no accumulation — the mechanism the protocol removes                         #
# --------------------------------------------------------------------------- #
def test_no_rollout_happens_and_it_is_recorded():
    grid = PS.GridSpec(dyaw_deg=(-12.0, 0.0, 12.0), dlon_steps=(0,))
    pw = PS.pseudo_evaluate(_StubPlanner(), _episodes(), grid,
                            frames_of=_frames_of, stride=8)
    assert pw["rollout_steps_executed"] == 0
    assert pw["traffic_mode"] == PS.TRAFFIC_MODE_LOG_REPLAY
    # every evaluated state's deviation IS its grid point — nothing accumulated
    assert set(np.unique(pw["pt_dyaw"].numpy())) == {-12.0, 0.0, 12.0}
    assert float(np.abs(pw["pt_dlat"].numpy()).max()) == 0.0
    assert pw["planner_calls"] == pw["traj"].shape[0]


def test_the_unperturbed_grid_point_applies_an_identity_warp():
    """dlat=0, dpsi=0 must leave the observation untouched, or the grid's
    origin is not the arm's real operating point."""
    Hm = PS.sampling_homography(0.0, 0.0)
    assert torch.allclose(Hm, torch.eye(3, dtype=Hm.dtype), atol=1e-9)


def test_traffic_mode_is_on_every_emitted_node():
    grid = PS.GridSpec(dyaw_deg=(-8.0, 0.0, 8.0), dlon_steps=(-5, 0, 5))
    pw = PS.pseudo_evaluate(_StubPlanner(recover=0.0), _episodes(), grid,
                            frames_of=_frames_of, stride=6)
    node = PS.emit(pw, arm="stub", n_boot=64)
    assert node["traffic_mode"] == PS.TRAFFIC_MODE_LOG_REPLAY
    assert "non-reactive" in node["traffic_mode_note"]
    assert node["envelope_proof"]["EXTRAPOLATION_frac_steps_any"] == 0.0
    assert node["rollout_steps_executed"] == 0


# --------------------------------------------------------------------------- #
# the composite and its discriminative-range gate                              #
# --------------------------------------------------------------------------- #
def test_discriminative_range_flags_a_saturated_component():
    dead = np.ones(500)
    live = np.linspace(0.0, 1.0, 500)
    r = PS.discriminative_range({"comfort": dead, "ego_progress": live})
    assert r["comfort"]["admissible"] is False
    assert r["comfort"]["ceiling_frac_ge_0p999"] == 1.0
    assert r["ego_progress"]["admissible"] is True


def test_discriminative_range_flags_zero_between_arm_spread():
    a = np.linspace(0.0, 1.0, 100)
    r = PS.discriminative_range(
        {"ego_progress": a},
        by_arm={"armA": {"ego_progress": a}, "armB": {"ego_progress": a}})
    assert r["ego_progress"]["between_arm_spread"] == 0.0
    assert r["ego_progress"]["admissible"] is False


def test_composite_REFUSES_to_emit_when_every_component_saturates():
    """⭐ The last agent fixed a vacuous diagnostic by refusing to emit one."""
    dead = {"ego_progress": np.ones(200), "recovery": np.ones(200),
            "comfort": np.ones(200), "no_collision": None, "ttc": None}
    r = PS.discriminative_range(dead)
    with pytest.raises(PS.VacuousMetric) as exc:
        PS.composite(dead, r)
    assert "REFUSING TO EMIT" in str(exc.value)


def test_collision_and_ttc_are_None_not_a_constant():
    grid = PS.GridSpec(dyaw_deg=(0.0, 12.0), dlon_steps=(0,))
    pw = PS.pseudo_evaluate(_StubPlanner(), _episodes(), grid,
                            frames_of=_frames_of, stride=8)
    sc = PS.score_windows(pw)
    assert sc["no_collision"] is None and sc["ttc"] is None
    assert "NOT COMPUTABLE" in sc["_unavailable"]["no_collision"]
    node = PS.emit(pw, arm="stub", n_boot=64)
    assert node["components"]["no_collision"]["ci"] is None
    assert node["components"]["no_collision"]["admissible"] is False
    # and the composite says out loud that it has no collision gate
    assert "no collision gate" in node["composite"]["_not_a_driving_score"].lower()


def test_composite_is_not_called_a_driving_score():
    grid = PS.GridSpec(dyaw_deg=(-12.0, 0.0, 12.0), dlon_steps=(0,))
    pw = PS.pseudo_evaluate(_StubPlanner(), _episodes(), grid,
                            frames_of=_frames_of, stride=8)
    node = PS.emit(pw, arm="stub", n_boot=64)
    comp = node["composite"]
    if "REFUSED_TO_EMIT" not in comp:
        # ⛔ The name CARRIES THE PROGRESS TERM (2026-07-28). A bare
        # "PSS_recovery_progress" is no longer emittable, because the published
        # `clamp_v1` term and the two-sided `twosided_v2` term are DIFFERENT
        # METRICS and a stable name over a changed definition is the exact
        # failure class this versioning exists to prevent.
        assert comp["name"] == "PSS_recovery_progress@twosided_v2"
        assert comp["progress_term"] == PS.PROGRESS_TERM_DEFAULT
        assert "Driving Score" in comp["_not_a_driving_score"]


# --------------------------------------------------------------------------- #
# the sub-scores behave as their docstrings claim                              #
# --------------------------------------------------------------------------- #
def test_recovery_separates_a_recovering_planner_from_a_drifting_one():
    """⭐ The error-recovery signal must actually ORDER two planners.

    Built directly on ``score_windows`` so the arithmetic is exercised without a
    model: same perturbed states, one plan that steers back onto the logged path
    and one that holds the perturbation. If the metric does not separate them it
    is decorative and must not ship."""
    n, Hh, dpsi_deg, v0 = 64, 20, 12.0, 10.0
    dpsi = np.deg2rad(dpsi_deg)
    t = torch.arange(1, Hh + 1, dtype=torch.float32) * PS.DT
    # the logged path: straight ahead at v0 in the REFERENCE frame
    ref = torch.zeros(n, Hh + 1, 2)
    ref[:, :, 0] = torch.cat([torch.zeros(1), v0 * t])[None].expand(n, -1)

    def _pw(traj):
        return {"traj": traj, "ref_path": ref, "ref_yaw": torch.zeros(n),
                "v0": torch.full((n,), v0),
                "pt_dyaw": torch.full((n,), dpsi_deg),
                "pt_dlat": torch.zeros(n), "pt_dlon": torch.zeros(n),
                "eid": [str(i % 8) for i in range(n)]}

    # (a) HOLD: drive straight in the planner's OWN perturbed frame
    hold = torch.stack([v0 * t, torch.zeros(Hh)], -1)[None].expand(n, -1, -1)
    # (b) RECOVER: land on the reference path — rotate the reference plan by
    #     -dpsi into the perturbed frame (the exact inverse of _cross_and_along)
    rx, ry = v0 * t, torch.zeros(Hh)
    rec = torch.stack([np.cos(dpsi) * rx + np.sin(dpsi) * ry,
                       -np.sin(dpsi) * rx + np.cos(dpsi) * ry],
                      -1)[None].expand(n, -1, -1)

    s_hold = PS.score_windows(_pw(hold.clone()))
    s_rec = PS.score_windows(_pw(rec.clone()))
    rc_hold = float(np.nanmean(s_hold["recovery"]))
    rc_rec = float(np.nanmean(s_rec["recovery"]))
    assert np.isfinite(rc_hold) and np.isfinite(rc_rec)
    assert rc_hold < 0.05, f"a non-recovering plan scored {rc_hold}"
    assert rc_rec > 0.95, f"a fully recovering plan scored {rc_rec}"
    assert rc_rec - rc_hold > 0.9        # the component ORDERS them


def test_recovery_is_not_gameable_by_standing_still():
    """⭐ THE DEFECT THE 2026-07-27 SMOKE FOUND, now a regression test.

    With a ``v0 * horizon`` denominator, a planner that barely moves had a tiny
    cross-track error and was PAID for it — the BLIND arm scored **+0.597 above
    the sighted one**. A metric that rewards stopping is worse than no metric.
    The denominator is now the plan's OWN along-track distance, so a stopped
    plan is UNDEFINED (excluded), never 1.0."""
    n, Hh, dpsi_deg, v0 = 32, 20, 12.0, 10.0
    t = torch.arange(1, Hh + 1, dtype=torch.float32) * PS.DT
    ref = torch.zeros(n, Hh + 1, 2)
    ref[:, :, 0] = torch.cat([torch.zeros(1), v0 * t])[None].expand(n, -1)
    pw = lambda tj: {"traj": tj, "ref_path": ref, "ref_yaw": torch.zeros(n),
                     "v0": torch.full((n,), v0),
                     "pt_dyaw": torch.full((n,), dpsi_deg),
                     "pt_dlat": torch.zeros(n), "pt_dlon": torch.zeros(n),
                     "eid": [str(i % 8) for i in range(n)]}
    stopped = torch.zeros(n, Hh, 2)
    rc = PS.score_windows(pw(stopped))["recovery"]
    assert np.isnan(rc).all(), (
        "a plan that does not move must not receive a recovery score; got "
        f"mean {np.nanmean(rc)}")
    # and it is not rescued by the composite either — its progress is zero
    sc = PS.score_windows(pw(stopped))
    assert float(np.nanmean(sc["ego_progress"])) < 0.05


def test_recovery_is_defined_and_low_for_a_drifting_model_planner():
    """The same finding through the full harness, on a stub model planner."""
    grid = PS.GridSpec(dyaw_deg=(-12.0, 12.0), dlon_steps=(0,))
    pw = PS.pseudo_evaluate(_StubPlanner(), _episodes(), grid,
                            frames_of=_frames_of, stride=8)
    rc = PS.score_windows(pw)["recovery"]
    fin = rc[np.isfinite(rc)]
    assert fin.size > 0, "recovery must be defined at perturbed grid points"
    assert float(np.nanmean(fin)) < 0.5


def test_recovery_is_undefined_at_the_unperturbed_point_by_construction():
    grid = PS.GridSpec(dyaw_deg=(0.0,), dlon_steps=(0,))
    pw = PS.pseudo_evaluate(_StubPlanner(), _episodes(), grid,
                            frames_of=_frames_of, stride=8)
    sc = PS.score_windows(pw)
    assert np.isnan(sc["recovery"]).all()


def test_ego_progress_is_1_for_a_planner_that_matches_the_human():
    grid = PS.GridSpec(dyaw_deg=(0.0,), dlon_steps=(0,))
    pw = PS.pseudo_evaluate(_StubPlanner(), _episodes(), grid,
                            frames_of=_frames_of, stride=8)
    sc = PS.score_windows(pw)
    ep = sc["ego_progress"]
    assert np.nanmean(ep) > 0.9


def test_proximity_weights_sum_to_one_and_peak_at_the_origin():
    pts = PS.default_grid().points()
    w = PS.proximity_weights(pts)
    assert w.sum() == pytest.approx(1.0)
    i0 = [i for i, p in enumerate(pts) if p[1] == 0.0 and p[2] == 0][0]
    assert w[i0] == pytest.approx(w.max())


def test_emit_names_its_estimator_and_refuses_the_bad_one():
    grid = PS.GridSpec(dyaw_deg=(-12.0, 0.0, 12.0), dlon_steps=(0,))
    pw = PS.pseudo_evaluate(_StubPlanner(), _episodes(), grid,
                            frames_of=_frames_of, stride=8)
    node = PS.emit(pw, arm="stub", n_boot=64)
    assert "episode_cluster_bootstrap" in node["_estimator"]
    assert "overlapping_holdout_se" in node["_refused_estimator"]
