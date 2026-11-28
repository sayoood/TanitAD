"""Pins for `tanitad.refs.feasible_decode` -- the feasibility-aware decode.

⛔ EVERY TEST HERE IS A CONTROL THAT MUST READ A KNOWN VALUE, and each one names WHICH
object it operated on. RETRACTION #30 (2026-09-05) was a lambda-sweep identity control
that PASSED while interpolating a tensor that exists at no point in the decode: it
checked the arithmetic, not the object. So `test_projection_is_exact_on_the_scorers_own
_flags` re-derives the flags through `fan_safety.score_paths` -- the CONSUMER -- rather
than through the module's own bookkeeping.
"""
from __future__ import annotations

import math

import pytest
import torch

from tanitad.refs import feasible_decode as FD
from tanitad.rl import rewards as RW


def _fs():
    """`taniteval/tools/fan_safety.py` is a SCRIPT dir, not a package -- importing it as
    `taniteval.tools.fan_safety` raises ModuleNotFound even with the repo on the path."""
    import importlib.util
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    path = os.path.join(repo, "taniteval", "tools", "fan_safety.py")
    if not os.path.isfile(path):
        pytest.skip("fan_safety.py not present in this tree")
    spec = importlib.util.spec_from_file_location("_fs_for_feasdec_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fan(seed: int = 0, n: int = 64, s: int = 5, scale: float = 3.0) -> torch.Tensor:
    """A deliberately WILD fan: origin-prepended, and violating by construction."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(n, s - 1, 2, generator=g) * scale
    d[..., 0] = d[..., 0].abs() + 4.0                 # keep it broadly forward
    p = torch.cat([torch.zeros(n, 1, 2), d.cumsum(dim=-2)], dim=-2)
    return p.double()


# --------------------------------------------------------------------------- #
# 1. the structural zero -- asserted through the SCORER, not through ourselves   #
# --------------------------------------------------------------------------- #
def test_projection_zeroes_the_scorers_own_flags():
    FS = _fs()
    p = _fan().float()
    v0 = torch.full((p.shape[0],), 12.0)
    before = FS.score_paths(p, v0, None)
    assert float(before["envelope"].float().mean()) > 0.5, \
        "the fixture is not violating; this test would pass vacuously"
    q = FD.project_feasible(p, mu=FD.MU_KAMM)
    after = FS.score_paths(q, v0, None)
    # ⛔ STRUCTURAL ZEROS, not small numbers.
    assert float(after["envelope"].float().mean()) == 0.0
    assert float(after["kamm_over"].float().mean()) == 0.0
    assert float(after["peak_g"].max()) <= FD.MU_KAMM + 1e-6


def test_box_only_zeroes_envelope_but_not_kamm():
    """`mu=None` is a DIFFERENT lever and must not be reported as the same one."""
    FS = _fs()
    p = _fan(seed=3).float()
    v0 = torch.full((p.shape[0],), 20.0)
    q = FD.project_feasible(p, mu=None)
    sc = FS.score_paths(q, v0, None)
    assert float(sc["envelope"].float().mean()) == 0.0
    # at 20 m/s the box permits |a_lat| up to v^2 * 0.2 = 80 m/s^2 >> 0.7 g, so the
    # friction circle is NOT implied by the box. If this ever reads 0 the fixture has
    # stopped exercising the distinction and the test is vacuous.
    assert float(sc["kamm_over"].float().mean()) > 0.0


# --------------------------------------------------------------------------- #
# 2. the DISABLED LEVER and the ROUND TRIP are two different controls            #
# --------------------------------------------------------------------------- #
def test_disabled_lever_returns_the_same_object():
    p = _fan(seed=1)
    assert FD.project_feasible(p, enabled=False) is p


def test_roundtrip_on_an_already_feasible_path_is_exact():
    """⛔ The control the DISABLED lever cannot give: the arithmetic actually runs."""
    v0 = 14.0
    t = torch.arange(5, dtype=torch.float64) * FD.DT_S
    straight = torch.stack([v0 * t, torch.zeros_like(t)], dim=-1)[None]
    out = FD.project_feasible(straight, mu=FD.MU_KAMM)
    assert float((out - straight).abs().max()) < 1e-9
    # A constant-curvature arc INSIDE the friction circle.
    # kappa = 0.02 at 14 m/s is a_lat = v^2 * kappa = 3.92 m/s^2 = 0.40 g.
    # kappa = 0.05 would be 9.80 m/s^2 = 1.00 g -- OUTSIDE 0.7 g, and it is what this
    # fixture originally used. Both satisfy |kappa| << 0.2, so the BOX passes it and only
    # the friction circle catches it: at speed, a curvature that looks gentle is not.
    kappa, s = 0.02, v0 * FD.DT_S
    h = torch.arange(4, dtype=torch.float64) * (v0 * kappa * FD.DT_S)
    arc = torch.cat([torch.zeros(1, 2),
                     torch.stack([s * torch.cos(h), s * torch.sin(h)], -1).cumsum(0)])[None]
    out = FD.project_feasible(arc, mu=FD.MU_KAMM)
    assert float((out - arc).abs().max()) < 1e-9


def test_ha0_floor_projects_to_itself():
    """The programme's shared trivial floor must be a fixed point."""
    FS = _fs()
    v0 = torch.tensor([0.0, 3.0, 12.0, 28.0], dtype=torch.float64)
    ha0 = FS.hold_v0_path(v0)
    out = FD.project_feasible(ha0, mu=FD.MU_KAMM)
    assert float((out - ha0).abs().max()) < 1e-9
    for mu in (None, 0.7, 0.3):
        st = FD.assert_feasible(FD.project_feasible(ha0, mu=mu), mu=mu)
        assert st["envelope_rate"] == 0.0


# --------------------------------------------------------------------------- #
# 3. the conventions this module is bound by                                    #
# --------------------------------------------------------------------------- #
def test_recover_controls_matches_rewards_kinematics_exactly():
    """⛔ ONE convention. A projection built on a different finite difference is
    approximately feasible, and reports a residual that looks like noise."""
    p = _fan(seed=7)
    speed, heading, accel, lat = FD.recover_controls(p)
    kin = RW.kinematics(p, FD.DT_S)
    assert torch.allclose(speed, kin.speed, atol=1e-12)
    assert torch.allclose(heading, kin.heading, atol=1e-12)
    assert torch.allclose(accel, kin.accel, atol=1e-12)
    assert torch.allclose(lat, kin.lat_acc, atol=1e-10)


def test_entry_clamp_binds_the_first_speed_and_is_off_by_default():
    p = _fan(seed=11)
    v0 = torch.full((p.shape[0],), 5.0, dtype=torch.float64)
    free = FD.project_feasible(p, v0, mu=FD.MU_KAMM, clamp_entry=False)
    bound = FD.project_feasible(p, v0, mu=FD.MU_KAMM, clamp_entry=True)
    s_free = FD.recover_controls(free)[0][..., 0]
    s_bound = FD.recover_controls(bound)[0][..., 0]
    assert float(s_free.max()) > 5.0 + FD.A_MAX_MPS2 * FD.DT_S      # unbound
    assert float(s_bound.max()) <= 5.0 + FD.A_MAX_MPS2 * FD.DT_S + 1e-9
    assert float(s_bound.min()) >= max(0.0, 5.0 - FD.A_MAX_MPS2 * FD.DT_S) - 1e-9


def test_projection_is_idempotent():
    """A projection onto a convex set is idempotent; if it is not, the second pass is
    changing the object and every downstream rate is a moving target."""
    p = _fan(seed=13)
    q1 = FD.project_feasible(p, mu=FD.MU_KAMM)
    q2 = FD.project_feasible(q1, mu=FD.MU_KAMM)
    assert float((q1 - q2).abs().max()) < 1e-9


def test_origin_is_required_and_preserved():
    p = _fan(seed=5)
    q = FD.project_feasible(p, mu=FD.MU_KAMM)
    assert float(q[..., 0, :].abs().max()) == 0.0
    with pytest.raises(ValueError):
        FD.project_feasible(p + 1.0, mu=FD.MU_KAMM)


def test_heading_aliasing_limit_is_stated_and_respected():
    """⚠️ The one edge: `recover_controls` wraps dtheta to +-pi, so a step turning by
    more than pi would read back wrapped. The bound is v * kappa_max * dt."""
    assert FD.ALIAS_SPEED_LIMIT_MPS == pytest.approx(math.pi / (0.2 * 0.5))
    p = _fan(seed=17, scale=8.0)
    q = FD.project_feasible(p, mu=FD.MU_KAMM)
    assert FD.max_heading_step(q) < math.pi


def test_a_stop_inside_the_window_does_not_manufacture_a_turn():
    """⛔ THE REGRESSION THIS MODULE ALMOST SHIPPED. A path that decelerates to a stop
    emits a zero-length step; `atan2(0, 0) == 0`, so the step before it reads as a turn
    through the whole previous heading. MEASURED: 2 of 51,200 fan candidates read
    |kappa| = 0.3396 against a 0.2 cap -- a stationary vehicle performing a hard turn.
    ⚠️ Both of the programme's recoveries do it, so a second opinion from
    `unicycle_controls_from_path` does NOT rescue it; only holding the heading does."""
    FS = _fs()
    dt = FD.DT_S
    # speeds [2, 4, 2, 0] on a curving path -- the offender's exact shape
    h = torch.tensor([0.0, 0.2, 0.34, 0.34], dtype=torch.float64)
    sp = torch.tensor([2.0, 4.0, 2.0, 0.0], dtype=torch.float64)
    d = torch.stack([sp * dt * torch.cos(h), sp * dt * torch.sin(h)], dim=-1)
    path = torch.cat([torch.zeros(1, 2, dtype=torch.float64), d.cumsum(0)])[None]
    v0 = torch.zeros(1, dtype=torch.float64)
    q = FD.project_feasible(path, v0, mu=FD.MU_KAMM, clamp_entry=True)
    sc = FS.score_paths(q.float(), v0.float(), None)
    assert float(sc["envelope"].float().mean()) == 0.0
    assert float(sc["kamm_over"].float().mean()) == 0.0
    # the stop is preserved to within a micron -- it is a stop, not a crawl
    assert float(FD.recover_controls(q)[0][..., -1].max()) < 1e-2   # a stop, not a crawl
    # ⛔ scored in float32 -- the dtype the CONSUMER actually uses. float64 exactness is
    # not the contract; a 1e-6 m held step is exact in float64 and noise in float32.
    assert float(FS.score_paths(q.float(), v0.float(), None)["envelope"].float().mean()) == 0.0


def test_a_path_that_never_moves_keeps_its_exact_zeros():
    """The epsilon must not fabricate motion where there was none: a stationary path has
    consistent headings and no spurious turn to remove."""
    zero = torch.zeros(1, 5, 2, dtype=torch.float64)
    q = FD.project_feasible(zero, torch.zeros(1, dtype=torch.float64), mu=FD.MU_KAMM)
    assert float(q.abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# 4. the DECODER HOOK -- the projection wired where the waypoints are moved      #
# --------------------------------------------------------------------------- #
def _decoder(**over):
    from tanitad.refs.refc import AnchoredDiffusionDecoder, DecoderConfig
    cfg = DecoderConfig(d=32, n_heads=2, layers=1, aux_hidden=16,
                        diffusion_steps=1, **over)
    anchors = torch.zeros(6, 8, 2)
    return AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=8, d_meas=4, d_ctx=4, tac_latent_dim=4,
        anchors=anchors, cfg=cfg, hierarchy=False, graft_maneuver=False,
        graft_target_latent=False, grounded_selector=False,
        horizons=(5, 10, 15, 20, 30, 40, 50, 60))


def test_decoder_hook_is_a_no_op_when_off_and_returns_the_same_object():
    """⛔ The flag defaults OFF and must then be byte-identical: a decode that
    silently rewrites a path when the lever is disabled makes every A/B in every
    downstream table uninterpretable."""
    from tanitad.refs.refc import DecoderConfig
    assert DecoderConfig().feasible_decode is False
    dec = _decoder()
    x = _fan(seed=21, n=6, s=8).float()[None]          # [1, 6, 8, 2]
    v = torch.full((1,), 12.0)
    assert dec._feasible(x, v) is x


def test_decoder_hook_zeroes_the_scorer_flags_on_the_2s_prefix():
    FS = _fs()
    dec = _decoder(feasible_decode=True, feasible_mu=0.7)
    x = _fan(seed=22, n=6, s=8).float()[None]
    v = torch.full((1,), 12.0)
    q = dec._feasible(x, v)
    assert q.shape == x.shape
    sc = FS.score_paths(FS.with_origin(q[..., :4, :]), v, None)
    assert float(sc["envelope"].float().mean()) == 0.0
    assert float(sc["kamm_over"].float().mean()) == 0.0
    # the 2-6 s tail is TRANSLATED, not reshaped: its own step vectors are unchanged
    d_in = x[..., 4:, :] - x[..., 3:-1, :]
    d_out = q[..., 4:, :] - q[..., 3:-1, :]
    assert float((d_in[..., 1:, :] - d_out[..., 1:, :]).abs().max()) < 1e-5


def test_decoder_hook_refuses_a_non_uniform_prefix_grid():
    """⛔ A projection run at a dt the scorer does not use is APPROXIMATELY
    feasible and reports a residual that looks like noise. Refuse, do not guess."""
    dec = _decoder(feasible_decode=True)
    dec.anchor_slots = torch.tensor([0, 4, 9, 19, 29, 39, 49, 59])
    with pytest.raises(ValueError, match="UNIFORM"):
        dec._feasible(_fan(seed=23, n=6, s=8).float()[None], torch.full((1,), 9.0))
