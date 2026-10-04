"""WP-RL (2026-10-04): the DDv2 RL binding for refcv7 and its PDMS-shaped reward.

Every control here reads a KNOWN value; every guard has a MUTATION that reintroduces a plausible
port error and must go RED (the expectation of each mutation test is a literal, never an
expression over the code under test). CPU only, toy decoder built with refcv7's own switches:
F1 random-t, F2 DD step, F3 per-layer cascade, F4 AdaLN, F5 emitting conf, agent cross-attention,
BEV coupling (1), NEW-1 residual prior ``ha0_ext_pose``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tanitad.models import refcv6_diffusion as rv6
from tanitad.models import refc_bev_coupling as bevc
from tanitad.refs import refc
from tanitad.rl import ddv2_refcv7 as R
from tanitad.rl import ddv2_rl as D
from tanitad.rl import pdm_proxy as P

H = (5, 10, 15, 20)


# --------------------------------------------------------------------------- #
# a toy refcv7-shaped decoder                                                   #
# --------------------------------------------------------------------------- #
def _dec(seed=0, n=5, layers=3, cascade=True):
    torch.manual_seed(seed)
    flags = rv6.DiffusionFlags(f1_random_t=True, f2_dd_step=True, f3_per_layer=cascade,
                               f4_adaln=True, f5_emitting_conf=True)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=layers, ff_mult=2, sampler="ddim",
                             refcv6=flags, cross_agent=True, residual_prior="ha0_ext_pose")
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=len(H), d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=torch.randn(n, len(H), 2), cfg=cfg, hierarchy=False, graft_maneuver=False,
        graft_target_latent=False, grounded_selector=False, horizons=H, v0_conditioned=True,
        control_units="alat")
    dec.attach_bev_coupling(bevc.BEVCouplingConfig(enable=True, d_model=32, d_bev=6,
                                                   n_points=len(H)), d_bev=6)
    ctrl = torch.stack([torch.linspace(-2.0, 2.0, n), torch.linspace(-1.5, 1.5, n)], dim=-1)
    ctrl[n // 2] = 0.0                                   # the prior itself (zero residual)
    dec.anchor_controls.copy_(ctrl)
    torch.manual_seed(seed + 7)
    with torch.no_grad():                                # a TRAINED-looking decoder: open every gate,
        for nm, p in dec.named_parameters():             # break every zero-init head
            if "gate" in nm:
                p.fill_(0.7)
            if "control_head" in nm or "control_heads" in nm:
                p.normal_(0, 0.3)
    return dec.eval()


def _fwd_inputs(b=2, seed=11):
    g = torch.Generator().manual_seed(seed)
    return dict(fmap=torch.randn(b, 16, 3, 5, generator=g), m=torch.randn(b, 8, generator=g),
                v_ms=torch.tensor([12.0, 3.5])[:b], agent_tokens=torch.randn(b, 6, 32, generator=g),
                agent_pad=torch.tensor([[False] * 4 + [True] * 2, [False] * 6])[:b],
                bev=torch.randn(b, 6, 120, 64, generator=g),
                residual_prior=(torch.tensor([0.8, -1.2])[:b], torch.tensor([0.02, -0.035])[:b]))


def _captured(dec, seed=5):
    kw = _fwd_inputs()
    with R.capture_sampler_inputs(dec) as rec:
        torch.manual_seed(seed)
        with torch.no_grad():
            out = dec(kw.pop("fmap"), kw.pop("m"), steps=2, **kw)
    assert len(rec) == 1
    return rec[0], out


# --------------------------------------------------------------------------- #
# binding                                                                       #
# --------------------------------------------------------------------------- #
def test_parity_native_sample_reproduces_the_deployed_sampler_bitwise():
    dec = _dec()
    inp, out = _captured(dec)
    assert inp.agents is not None and inp.bev is not None and inp.prior is not None
    eps = inp.replay_eps(dec)
    fan, u0 = R.native_sample(dec, inp, eps)
    assert torch.equal(fan, inp.out[0])
    assert torch.equal(u0, inp.out[1])
    assert torch.equal(out["anchor_traj"], inp.out[0])


@pytest.mark.parametrize("drop", ["agents", "bev"])
def test_MUTATION_a_binding_that_drops_agents_or_bev_breaks_parity(drop):
    dec = _dec()
    inp, _ = _captured(dec)
    eps = inp.replay_eps(dec)
    setattr(inp, drop, None)
    fan, _ = R.native_sample(dec, inp, eps)
    assert not torch.equal(fan, inp.out[0])               # RED: the mutant is caught


def test_MUTATION_a_binding_that_rolls_on_v0_instead_of_the_prior_speed_breaks_parity():
    dec = _dec()
    inp, _ = _captured(dec)
    eps = inp.replay_eps(dec)
    a0, k0, pv = inp.prior
    inp.prior = (a0, k0, pv + 1.0)
    inp.v = inp.prior[2]
    fan, _ = R.native_sample(dec, inp, eps)
    assert not torch.equal(fan, inp.out[0])


def test_groups_are_independent_queries():
    dec = _dec()
    inp, _ = _captured(dec)
    fn = R.make_x0_fn(dec, inp)
    x = R.anchor_state(dec, 2)
    with torch.no_grad():
        one = fn(x, 10)
        four = fn(D.tile_groups(x, 4), 10)
    n = x.shape[1]
    for g in range(4):
        assert torch.allclose(four[:, g * n:(g + 1) * n], one, atol=1e-6)


def test_MUTATION_a_layer_that_mixes_queries_is_caught(monkeypatch):
    dec = _dec()
    inp, _ = _captured(dec)
    orig = dec._decode_ctrl

    def mixing(kv, cond, x_path, t, *a, **k):
        conf, du = orig(kv, cond, x_path, t, *a, **k)
        return conf, du + du.mean(dim=1, keepdim=True)     # every query sees the others
    monkeypatch.setattr(dec, "_decode_ctrl", mixing)
    fn = R.make_x0_fn(dec, inp)
    x = R.anchor_state(dec, 2)
    with torch.no_grad():
        one = fn(x, 10)
        four = fn(torch.cat([x, x.flip(1), x * 0.5, x * 2.0], dim=1), 10)
    assert not torch.allclose(four[:, :x.shape[1]], one, atol=1e-6)


def test_tick_states_equal_the_decoders_own_roll_at_the_slot_ticks_bitwise():
    dec = _dec()
    inp, _ = _captured(dec)
    g = torch.Generator().manual_seed(3)
    x = R.anchor_state(dec, 2) + 0.2 * torch.randn(2, 5, len(H), 2, generator=g)
    st = R.tick_states(dec, x, inp, max(H))
    fan = R.roll_fan(dec, x, inp)
    assert torch.equal(st[..., list(H), :2], fan)
    assert torch.equal(st[..., 0, :], torch.stack([torch.zeros(2, 5), torch.zeros(2, 5),
                                                    torch.zeros(2, 5),
                                                    inp.prior[2][:, None].expand(2, 5)], -1))


def test_MUTATION_tick_states_rolled_from_v0_not_the_prior_is_caught():
    dec = _dec()
    inp, _ = _captured(dec)
    x = R.anchor_state(dec, 2)
    a0, k0, pv = inp.prior
    bad = R.tick_states(dec, x, (a0, k0, pv * 0 + 7.0), max(H))
    assert not torch.equal(bad[..., list(H), :2], R.roll_fan(dec, x, inp))


def test_cascade_stages_are_recorded_and_the_last_stage_is_the_policy_output():
    dec = _dec(layers=3)
    inp, _ = _captured(dec)
    stages: list = []
    fn = R.make_x0_fn(dec, inp, stages=stages)
    x = R.anchor_state(dec, 2)
    with torch.no_grad():
        y = fn(x, 6)
    st = R.stage_x0_list(stages)
    assert len(st) == 3 and not stages
    assert torch.equal(st[-1], y)
    assert not torch.equal(st[0], st[-1])


def test_capture_refuses_an_off_build():
    torch.manual_seed(0)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2, sampler="ddim")
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=len(H), d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=torch.randn(5, len(H), 2), cfg=cfg, hierarchy=False, graft_maneuver=False,
        graft_target_latent=False, grounded_selector=False, horizons=H, v0_conditioned=True,
        control_units="alat")
    with pytest.raises(D.Ddv2ConfigError, match="RESIDUAL"):
        with R.capture_sampler_inputs(dec):
            pass


def test_capture_restores_the_bound_method():
    dec = _dec()
    before = dec._sample.__func__
    _captured(dec)
    assert dec._sample.__func__ is before and "_sample" not in vars(dec)


# --------------------------------------------------------------------------- #
# the reward: analytic known values                                             #
# --------------------------------------------------------------------------- #
CFG = P.ProxyConfig(n_ticks=40)


def _straight(v=10.0, y0=0.0, dy_per_s=0.0, n=40, dt=0.1):
    t = torch.arange(n + 1, dtype=torch.float32) * dt
    yaw = math.atan2(dy_per_s, v) if v else 0.0
    return torch.stack([v * t, y0 + dy_per_s * t, torch.full_like(t, yaw), torch.full_like(t, v)], -1)


def _road_map(road_half=5.0, left_only=False, lane_line_y=None):
    """1000 x 600 fine codes: road (1) for |y| < road_half (or 0 < y < road_half), sidewalk (7)
    elsewhere, all seen; optional 0.1 m lane line (2)."""
    y = -30.0 + (torch.arange(600, dtype=torch.float32) + 0.5) * 0.1
    road = (y > 0) & (y < road_half) if left_only else (y.abs() < road_half)
    row = torch.where(road, torch.tensor(1), torch.tensor(7)).to(torch.uint8)
    codes = row[None].expand(1000, 600).clone()
    if lane_line_y is not None:
        j = int((lane_line_y + 30.0) / 0.1)
        codes[:, j] = 2
    return codes


def dac_control_ok(dac_fn=R.dac_fine, off_fn=R.offroad_cells) -> bool:
    """The DAC known values: on-road 1, leaving the road 0, a corner (not the centre) leaving 0,
    a lane line under a wheel 1, unseen area 1, a LEFT-only road read with the right sign."""
    off = off_fn(_road_map())
    ok = float(dac_fn(_straight()[None], off)[0]) == 1.0
    ok &= float(dac_fn(_straight(dy_per_s=2.5)[None], off)[0]) == 0.0           # drifts to y = 10
    ego_half_w = P.PROXY.ego_width / 2
    # the block rule resolves +-0.25 m, so the corners sit 0.4 m inside / outside the road edge;
    # in BOTH cases the footprint CENTRE is on the road (y0 3.45 / 4.25 < 5)
    centre_on_road = _straight(y0=5.0 - 0.4 - ego_half_w)        # left corners at y = 4.6
    corner_out = _straight(y0=5.0 + 0.4 - ego_half_w)            # left corners at y = 5.4 > 5
    ok &= float(dac_fn(centre_on_road[None], off)[0]) == 1.0
    ok &= float(dac_fn(corner_out[None], off)[0]) == 0.0
    off_ll = off_fn(_road_map(lane_line_y=1.0))
    ok &= float(dac_fn(_straight(y0=1.0)[None], off_ll)[0]) == 1.0
    unseen = torch.full((1000, 600), 255, dtype=torch.uint8)
    ok &= float(dac_fn(_straight(dy_per_s=2.5)[None], off_fn(unseen))[0]) == 1.0
    off_left = off_fn(_road_map(road_half=6.0, left_only=True))
    ok &= float(dac_fn(_straight(y0=3.0)[None], off_left)[0]) == 1.0           # left of centre: road
    ok &= float(dac_fn(_straight(y0=-3.0)[None], off_left)[0]) == 0.0          # right: sidewalk
    return bool(ok)


def test_DAC_known_values():
    assert dac_control_ok() is True


def test_DAC_known_values_hold_on_the_R2_rung_too():
    assert dac_control_ok(off_fn=R.offroad_cells_r2) is True


def test_R2_ignores_unclassified_ground_and_R1_does_not():
    """A-0's one difference, as literals: a road whose shoulder is code 0 ('seen, no map class')."""
    y = -30.0 + (torch.arange(600, dtype=torch.float32) + 0.5) * 0.1
    row = torch.where(y.abs() < 5.0, torch.tensor(1), torch.tensor(0)).to(torch.uint8)
    codes = row[None].expand(1000, 600).clone()
    onto_shoulder = _straight(dy_per_s=2.5)
    assert float(R.dac_fine(onto_shoulder[None], R.offroad_cells(codes))[0]) == 0.0
    assert float(R.dac_fine(onto_shoulder[None], R.offroad_cells_r2(codes))[0]) == 1.0


def test_MUTATION_R2_with_sidewalk_as_road_is_caught(monkeypatch):
    monkeypatch.setattr(R, "ROAD_CODES", R.ROAD_CODES + (7,))
    assert dac_control_ok(off_fn=R.offroad_cells_r2) is False


def test_MUTATION_DAC_sidewalk_counted_as_road_is_caught(monkeypatch):
    monkeypatch.setattr(R, "ROAD_CODES", R.ROAD_CODES + (7,))
    assert dac_control_ok() is False


def test_MUTATION_DAC_centre_point_instead_of_corners_is_caught():
    def centre_only(states, off, rule=R.DAC_RULE, cfg=P.PROXY):
        cx, cy = P._centers(states, cfg)
        ix = torch.floor(cx / rule.cell_m).long()
        iy = torch.floor((cy + rule.y_half_m) / rule.cell_m).long()
        h, w = off.shape
        inside = (ix >= 0) & (ix < h) & (iy >= 0) & (iy < w)
        o = off[ix.clamp(0, h - 1), iy.clamp(0, w - 1)] & inside
        return (~o.any(dim=1)).float()
    assert dac_control_ok(dac_fn=centre_only) is False


def test_MUTATION_DAC_lateral_axis_flipped_is_caught():
    def flipped(states, off, rule=R.DAC_RULE, cfg=P.PROXY):
        s = states.clone()
        s[..., 1] = -s[..., 1]
        s[..., 2] = -s[..., 2]
        return R.dac_fine(s, off, rule, cfg)
    assert dac_control_ok(dac_fn=flipped) is False


def _tracks(xy_list, T=50, static=(), lw=(4.5, 1.9), moving_v=0.0):
    """Agents at fixed (or constant-velocity along +x) positions over T ticks."""
    a = len(xy_list)
    xy = torch.zeros(T, a, 2)
    for j, (x, y) in enumerate(xy_list):
        xy[:, j, 0] = x + moving_v * torch.arange(T) * 0.1
        xy[:, j, 1] = y
    st = torch.zeros(a, dtype=torch.bool)
    for j in static:
        st[j] = True
    sp = torch.full((T, a), float(moving_v))
    return P.AgentTracks(xy=xy, yaw=torch.zeros(T, a), lw=torch.tensor([lw] * a),
                         valid=torch.ones(T, a, dtype=torch.bool), static=st, speed=sp)


def nc_ttc_control_ok(score=R.score_window) -> bool:
    hum = _straight(v=10.0)
    route = hum[:, :2] + torch.tensor([P.PROXY.rear_axle_to_center, 0.0])
    ok = True
    # a slower DYNAMIC agent 25 m ahead on the path: the ego's front edge reaches it -> NC 0
    s = score(hum[None], hum, _tracks([(25.0, 0.0)], moving_v=2.0), route, None, CFG)
    ok &= float(s["nc"][0]) == 0.0 and bool(s["constraint_fail"][0])
    # the same agent of a STATIC class -> NC 0.5 (NAVSIM's value), still a constraint fail
    s = score(hum[None], hum, _tracks([(25.0, 0.0)], static=(0,), moving_v=0.0), route, None, CFG)
    ok &= float(s["nc"][0]) == 0.5 and bool(s["constraint_fail"][0])
    # an agent 10 m to the side: no contact -> NC 1, TTC 1
    s = score(hum[None], hum, _tracks([(25.0, 10.0)], moving_v=10.0), route, None, CFG)
    ok &= float(s["nc"][0]) == 1.0 and float(s["ttc"][0]) == 1.0
    # TTC: a stopped agent just beyond where a stopping ego ends -- the ego decelerates to a halt
    # 0.3 m short of it (no contact, NC 1) but its constant-velocity projection meets it -> TTC 0
    v0, a = 10.0, -2.5
    t = torch.arange(41, dtype=torch.float32) * 0.1
    v = (v0 + a * t).clamp_min(0.0)
    x = torch.cumsum(torch.cat([torch.zeros(1), v[:-1] * 0.1]), 0)
    brake = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x), v], -1)
    front = float(x[-1]) + P.PROXY.rear_axle_to_center + P.PROXY.ego_length / 2
    s = score(brake[None], hum, _tracks([(front + 0.3 + 4.5 / 2, 0.0)], static=(), moving_v=0.0),
              route, None, CFG)
    ok &= float(s["nc"][0]) == 1.0 and float(s["ttc"][0]) == 0.0
    return bool(ok)


def test_NC_and_TTC_known_values():
    assert nc_ttc_control_ok() is True


def test_MUTATION_agents_dropped_is_caught():
    def no_agents(c, h, tracks, route, off, cfg, **k):
        t = P.AgentTracks(xy=tracks.xy, yaw=tracks.yaw, lw=tracks.lw,
                          valid=torch.zeros_like(tracks.valid), static=tracks.static, speed=tracks.speed)
        return R.score_window(c, h, t, route, off, cfg, **k)
    assert nc_ttc_control_ok(score=no_agents) is False


def test_MUTATION_static_and_dynamic_values_swapped_is_caught(monkeypatch):
    real = P.no_at_fault_collision

    def swapped(states, agents, cfg=P.PROXY):
        a = P.AgentTracks(xy=agents.xy, yaw=agents.yaw, lw=agents.lw, valid=agents.valid,
                          static=~agents.static, speed=agents.speed)
        return real(states, a, cfg)
    monkeypatch.setattr(P, "no_at_fault_collision", swapped)
    assert nc_ttc_control_ok() is False


def test_MUTATION_ttc_projection_removed_is_caught():
    cfg0 = P.ProxyConfig(n_ticks=40, ttc_offsets=(0,))

    def no_proj(c, h, t, r, o, cfg, **k):
        return R.score_window(c, h, t, r, o, cfg0, **k)
    assert nc_ttc_control_ok(score=no_proj) is False


def test_EP_saturates_above_the_human_and_discriminates_below():
    """STATED LIMITATION, measured: EP is 1.0 for any candidate at least as far along as the
    reference (NAVSIM's pairwise rule; the paper's reference is PDM-Closed, ours is the human),
    so over-speed is a FLAT direction of this reward; SPD (logged, weight 0) sees it."""
    hum = _straight(v=10.0, n=60)
    route = _straight(v=10.0, n=69)[:, :2] + torch.tensor([P.PROXY.rear_axle_to_center, 0.0])
    cfg = P.ProxyConfig(n_ticks=60)
    tr = _tracks([(400.0, 40.0)], T=70)
    c = torch.stack([_straight(v=15.0, n=60), _straight(v=5.0, n=60), hum])
    s = R.score_window(c, hum, tr, route, None, cfg)
    assert float(s["ep"][0]) == 1.0 and float(s["ep"][2]) == 1.0
    assert 0.45 < float(s["ep"][1]) < 0.55
    assert float(s["spd"][0]) == 0.0 and float(s["spd"][1]) == 1.0
    assert float(s["pdms"][0]) == float(s["pdms"][2])           # w_spd = 0: SPD never enters PDMS


def ep_comfort_control_ok(score=R.score_window) -> bool:
    """EP: a 0.5x-slow candidate reads EP in (0.45, 0.55); a 1.5x-fast one exactly 1.0.
    Comfort: a hard 6 m/s^2 launch (> NAVSIM's 2.40 max_lon_accel) reads C = 0; a steady
    cruise reads C = 1."""
    hum = _straight(v=10.0, n=60)
    route = _straight(v=10.0, n=69)[:, :2] + torch.tensor([P.PROXY.rear_axle_to_center, 0.0])
    cfg = P.ProxyConfig(n_ticks=60)
    tr = _tracks([(400.0, 40.0)], T=70)
    t = torch.arange(61, dtype=torch.float32) * 0.1
    v = 2.0 + 6.0 * t
    x = torch.cumsum(torch.cat([torch.zeros(1), v[:-1] * 0.1]), 0)
    launch = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x), v], -1)
    c = torch.stack([_straight(v=5.0, n=60), _straight(v=15.0, n=60), launch, hum])
    s = score(c, hum, tr, route, None, cfg)
    ok = 0.45 < float(s["ep"][0]) < 0.55 and float(s["ep"][1]) == 1.0
    ok &= float(s["comfort"][2]) == 0.0 and float(s["comfort"][3]) == 1.0
    return bool(ok)


def test_EP_and_comfort_known_values():
    assert ep_comfort_control_ok() is True


def test_MUTATION_EP_referenced_to_the_candidate_itself_is_caught(monkeypatch):
    real = P.ego_progress

    def self_ref(states, route, cfg=P.PROXY):
        r = real(states, route, cfg)
        return torch.full_like(r, float(r.max()))        # every proposal 'as far as the best'
    monkeypatch.setattr(P, "ego_progress", self_ref)
    assert ep_comfort_control_ok() is False


def test_MUTATION_comfort_thresholds_relaxed_tenfold_is_caught(monkeypatch):
    real = P.comfort

    def lax(states, cfg=P.PROXY):
        import dataclasses
        return real(states, dataclasses.replace(cfg, max_lon_accel=24.0, min_lon_accel=-40.5,
                                                max_abs_lon_jerk=41.3, max_abs_mag_jerk=83.7))
    monkeypatch.setattr(P, "comfort", lax)
    assert ep_comfort_control_ok() is False


def test_identity_human_as_candidate_scores_exactly_the_human():
    hum = _straight(v=8.0, dy_per_s=0.5)
    route = hum[:, :2]
    s = R.score_window(hum[None], hum, _tracks([(30.0, 6.0)], moving_v=8.0), route,
                       R.offroad_cells(_road_map(road_half=8.0)), CFG)
    assert float(s["pdms"][0]) == s["human"]["pdms"]


def test_chunking_never_changes_a_score():
    g = torch.Generator().manual_seed(5)
    hum = _straight(v=9.0)
    cands = torch.stack([_straight(v=float(v), dy_per_s=float(d)) for v, d in
                         zip(torch.rand(17, generator=g) * 15, torch.randn(17, generator=g))])
    tr = _tracks([(20.0, 0.5), (35.0, -3.0), (15.0, 4.0)], moving_v=3.0)
    off = R.offroad_cells(_road_map(road_half=4.0))
    a = R.score_window(cands, hum, tr, hum[:, :2], off, CFG, chunk=3)
    b = R.score_window(cands, hum, tr, hum[:, :2], off, CFG, chunk=1000)
    for k in ("nc", "dac", "ep", "ttc", "comfort", "pdms"):
        assert torch.equal(a[k], b[k])


def test_filter_tracks_is_exact_for_nc_and_ttc():
    g = torch.Generator().manual_seed(9)
    hum = _straight(v=12.0)
    cands = torch.stack([_straight(v=float(v), dy_per_s=float(d)) for v, d in
                         zip(torch.rand(30, generator=g) * 20, torch.randn(30, generator=g) * 2)])
    a = 40
    start = torch.rand(1, a, 2, generator=g) * torch.tensor([400.0, 300.0]) - torch.tensor([100.0, 150.0])
    vel = torch.randn(1, a, 2, generator=g) * 5.0
    xy = start + vel * (torch.arange(50, dtype=torch.float32) * 0.1)[:, None, None]
    tr = P.AgentTracks(xy=xy, yaw=torch.zeros(50, a), lw=torch.full((a, 2), 4.0),
                       valid=torch.ones(50, a, dtype=torch.bool), static=torch.zeros(a, dtype=torch.bool),
                       speed=torch.full((50, a), 1.0))
    allst = torch.cat([hum[None], cands])
    ft = R.filter_tracks(tr, allst)
    assert ft.xy.shape[1] < a                                  # the filter did something
    for fn in (P.no_at_fault_collision, P.ttc_within_bound):
        assert torch.equal(fn(allst, tr, CFG), fn(allst, ft, CFG))


class _FakeReader:
    def __init__(self, frames):
        self.frames = frames              # list of list of dicts

    def lookup(self, eid, f):
        fr = self.frames[f]
        import numpy as np
        return np.array([[a["cx"], a["cy"], a["yaw"], a["l"], a["w"], -1.0] for a in fr],
                        dtype=np.float32).reshape(-1, 6)

    def lookup_track_ids(self, eid, f):
        return [a["track_id"] for a in self.frames[f]]

    def lookup_classes(self, eid, f):
        return [a["cls"] for a in self.frames[f]]


def test_tracks_from_join_equals_the_reference_dict_builder():
    g = torch.Generator().manual_seed(2)
    T = 12
    poses = torch.cumsum(torch.randn(T, 4, generator=g) * torch.tensor([1.0, 0.3, 0.05, 0.0]), 0)
    frames = []
    for k in range(T):
        fr = []
        for j in range(4):
            if (k + j) % 5 == 0:
                continue
            fr.append({"track_id": f"t{j}", "cx": float(torch.randn(1, generator=g)) * 10,
                       "cy": float(torch.randn(1, generator=g)) * 5,
                       "yaw": float(torch.randn(1, generator=g)), "l": 4.0 + j, "w": 1.8,
                       "cls": "protruding_object" if j == 2 else "automobile"})
        frames.append(fr)
    ref = P.AgentTracks.from_frames(frames, poses)
    got = R.tracks_from_join(_FakeReader(frames), 0, 0, poses, T)
    for f in ("xy", "lw", "valid", "static", "speed"):
        assert torch.allclose(getattr(got, f).float(), getattr(ref, f).float(), atol=1e-4), f
    assert torch.allclose(torch.where(ref.valid, got.yaw, 0 * got.yaw),
                          torch.where(ref.valid, ref.yaw, 0 * ref.yaw), atol=1e-4)


def test_an_unlabelled_frame_gives_no_tracks_never_an_empty_road():
    class Gap(_FakeReader):
        def lookup(self, eid, f):
            return None if f == 3 else super().lookup(eid, f)
    frames = [[{"track_id": "a", "cx": 5.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0, "cls": "automobile"}]] * 6
    assert R.tracks_from_join(Gap(frames), 0, 0, torch.zeros(6, 4), 6) is None


# --------------------------------------------------------------------------- #
# the step arithmetic                                                           #
# --------------------------------------------------------------------------- #
def test_microbatches_sum_to_the_release_full_batch_loss_and_gradient():
    g = torch.Generator().manual_seed(4)
    B, M, T = 4, 6, 3
    adv = torch.relu(torch.randn(B, M, generator=g))
    w = D.step_loss_weights(adv, T)
    mu = torch.randn(B, M, T, requires_grad=True, generator=g)
    il = torch.rand(B, T, generator=g)

    def logp_of(mu_, i):
        return -(mu_[..., i] - 0.3) ** 2

    full = sum(D.per_step_loss(logp_of(mu, i), il[:, i].mean(), w, i) for i in range(T))
    (gf,) = torch.autograd.grad(full, mu)
    total, gm = 0.0, torch.zeros_like(mu)
    for rows in ([0, 1], [2, 3]):
        mu2 = mu.detach().clone().requires_grad_(True)
        part = 0.0
        for i in range(T):
            li, _ = R.microbatch_step_loss(logp_of(mu2[rows], i), w["coef_rl"][rows][..., i],
                                           w["il_coef"], il[rows, i].mean(), 0.5)
            part = part + li
        (gr,) = torch.autograd.grad(part, mu2)
        gm += gr
        total += float(part)
    assert abs(total - float(full)) < 1e-6
    assert torch.allclose(gm, gf, atol=1e-7)


def test_RLOFF_zero_weight_gives_exactly_the_IL_only_gradient():
    g = torch.Generator().manual_seed(6)
    B, M, T = 2, 4, 2
    adv = torch.relu(torch.randn(B, M, generator=g)) + 0.1
    w = D.step_loss_weights(adv, T)
    mu = torch.randn(B, M, T, generator=g, requires_grad=True)
    il_src = torch.randn(B, M, T, generator=g)

    def grad_with(rl_weight):
        mu_ = mu.detach().clone().requires_grad_(True)
        tot = 0.0
        for i in range(T):
            lp = -(mu_[..., i] ** 2)
            il_i = (mu_[..., i] - il_src[..., i]).abs().mean()
            li, _ = R.microbatch_step_loss(lp, w["coef_rl"][..., i] * rl_weight, w["il_coef"], il_i, 1.0)
            tot = tot + li
        return torch.autograd.grad(tot, mu_)[0]

    def il_only():
        mu_ = mu.detach().clone().requires_grad_(True)
        tot = sum(w["il_coef"] * (mu_[..., i] - il_src[..., i]).abs().mean() for i in range(T))
        return torch.autograd.grad(tot, mu_)[0]
    assert torch.equal(grad_with(0.0), il_only())
    assert not torch.equal(grad_with(1.0), il_only())          # the RL arm differs (control)


def test_shuffle_is_a_derangement_and_refuses_one_window():
    g = torch.Generator().manual_seed(0)
    for b in (2, 3, 8, 16):
        p = R.shuffle_permutation(b, g)
        assert sorted(p.tolist()) == list(range(b)) and not bool((p == torch.arange(b)).any())
    with pytest.raises(D.Ddv2ConfigError):
        R.shuffle_permutation(1, g)


def test_adamw_at_lr_zero_leaves_every_parameter_bitwise():
    dec = _dec()
    named = R.trainable_named(dec)
    before = {n: p.detach().clone() for n, p in named}
    opt = torch.optim.AdamW([p for _, p in named], lr=0.0, weight_decay=1e-4)
    for _, p in named:
        p.grad = torch.randn_like(p)
    opt.step()
    assert all(torch.equal(p, before[n]) for n, p in named)


def test_trainable_set_is_the_generator_only():
    dec = _dec()
    names = [n for n, _ in R.trainable_named(dec)]
    assert any(n.startswith("layers.") for n in names) and any(n.startswith("cascade.") for n in names)
    assert not any(n.startswith(("conf_head", "offset_head", "time_embed", "anchor")) for n in names)
