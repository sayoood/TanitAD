"""refcv5 WP-4 (`E-DDA-3`) — the control-space DDIM sampler.

Each test names its kind: *fails without the feature* / *control that must read
a known value* / *deliberate regression*.

⭐ The two load-bearing identities this file exists for:
  1. a CONSTANT control sequence rolled by ``roll_controls`` is **bit-identical**
     to ``AnchoredDiffusionDecoder.roll_bank`` — otherwise the sampler and the
     vocabulary integrate different physics;
  2. ``sampler="none"`` is **byte-identical** to refcv4b.
"""

from __future__ import annotations

import math

import pytest
import torch

from tanitad.refs import refc
from tanitad.refs import refc_sampler as rs


# ---------------------------------------------------------------------------
# 1. The schedule — pinned against the PUBLISHED constants and against diffusers
# ---------------------------------------------------------------------------
def test_scaled_linear_is_the_square_of_the_linspace():
    """CONTROL. `scaled_linear` squares the LINSPACE, not the betas. Getting it
    backwards yields a plausible-looking schedule that noises by the wrong
    amount at every t — which would silently change what `t = 8` means."""
    s = rs.DDIMSchedule()
    b = s.betas
    assert b.shape == (1000,)
    assert abs(float(b[0]) - rs.BETA_START) < 1e-12
    assert abs(float(b[-1]) - rs.BETA_END) < 1e-12
    # the midpoint of a SQUARED linspace is NOT the midpoint of a linear one
    lin_mid = 0.5 * (rs.BETA_START + rs.BETA_END)
    assert abs(float(b[499]) - lin_mid) > 1e-4


def test_sigma_at_t8_reproduces_the_PUBLISHED_0_0316():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. The whole control-space argument
    rests on this number: the design plan states `sqrt(1 - alpha_bar_8) =
    0.0316`, and derives from it that DD's metre-space noise is 0.90 m / 0.73 m
    per waypoint while ours is 0.126 / 0.095 m/s^2. If this drifts, the
    arithmetic in the plan is void."""
    s = rs.DDIMSchedule()
    sigma8 = float(s.sqrt_one_minus_abar(8))
    assert abs(sigma8 - 0.0316) < 5e-4, sigma8
    # and the control-space consequence, with the vocabulary's own normalisers
    assert abs(sigma8 * 4.0 - 0.126) < 5e-3      # a_lon, m/s^2
    assert abs(sigma8 * 3.0 - 0.095) < 5e-3      # a_lat, m/s^2


def test_sigma_at_t49_is_the_training_maximum():
    """CONTROL. The plan states sigma(t = 49) = 0.377 / 0.283 m/s^2."""
    s = rs.DDIMSchedule()
    sigma = float(s.sqrt_one_minus_abar(49))
    assert abs(sigma * 4.0 - 0.377) < 5e-3
    assert abs(sigma * 3.0 - 0.283) < 5e-3


def test_add_noise_at_sigma_zero_is_THE_IDENTITY():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE: at t = 0 the forward noising
    must return x0 scaled by sqrt(abar_0), and with ANY noise the deviation is
    exactly sqrt(1-abar_0) — the smallest the schedule offers."""
    s = rs.DDIMSchedule()
    x0 = torch.randn(3, 4, 8, 2)
    eps = torch.randn_like(x0)
    t = torch.zeros(3, dtype=torch.long)
    out = s.add_noise(x0, eps, t)
    a = float(s.sqrt_abar(0))
    sig = float(s.sqrt_one_minus_abar(0))
    assert torch.allclose(out, a * x0 + sig * eps, atol=1e-6)
    assert sig < 0.011          # ~sqrt(beta_0)


def test_add_noise_broadcasts_PER_SAMPLE_t():
    """FAILS WITHOUT THE FEATURE. A per-sample t must not silently broadcast
    along the wrong axis — the classic bug in this family."""
    s = rs.DDIMSchedule()
    x0 = torch.zeros(2, 3, 4, 2)
    eps = torch.ones_like(x0)
    out = s.add_noise(x0, eps, torch.tensor([0, 900]))
    assert float(out[0].max()) < float(out[1].min())


def test_infer_timesteps_is_DDs_ladder():
    """CONTROL. DD's inference is 'fresh eps at t = 8, two DDIM steps [10, 0]'.
    The 8 -> 10 shift is `steps_offset = 1`."""
    s = rs.DDIMSchedule()
    assert s.infer_timesteps(8, 2) == [10, 0]      # the PUBLISHED ladder
    assert s.infer_timesteps(8, 1) == [9]          # degenerate: no descent
    assert s.infer_timesteps(8, 0) == []


def test_ddim_step_returns_toward_x0():
    """FAILS WITHOUT THE FEATURE. A deterministic (eta = 0) DDIM step from a
    noisy state toward a predicted x0 must reduce the distance to x0."""
    s = rs.DDIMSchedule()
    x0 = torch.zeros(2, 4, 8, 2)
    eps = torch.randn_like(x0)
    t = torch.full((2,), 40, dtype=torch.long)
    x_t = s.add_noise(x0, eps, t)
    x_p = s.step(x0, x_t, t, torch.zeros_like(t))
    assert float((x_p - x0).abs().mean()) < float((x_t - x0).abs().mean())


def test_pinned_against_diffusers_where_available():
    """CONTROL against an INDEPENDENT implementation. The scheduler is
    re-implemented so pods need no new dependency (`uv pip install` has TWICE
    silently replaced torch with a wheel the driver cannot run); a
    re-implementation that is never checked is an unverified copy."""
    s = rs.DDIMSchedule()
    if not rs.assert_matches_diffusers(s):
        pytest.skip("diffusers not installed — cannot cross-check here")


# ---------------------------------------------------------------------------
# 2. roll_controls — the identity that ties the sampler to the vocabulary
# ---------------------------------------------------------------------------
def _v0_decoder(units="alat", n=5, horizons=(5, 10, 15, 20), seed=0,
                sampler="none"):
    torch.manual_seed(seed)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                             sampler=sampler)
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=len(horizons), d_meas=8, d_ctx=4,
        tac_latent_dim=4, anchors=torch.randn(n, len(horizons), 2), cfg=cfg,
        hierarchy=False, graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=horizons, v0_conditioned=True,
        control_units=units)
    torch.manual_seed(seed + 1)
    ctrl = torch.stack([torch.linspace(-2.0, 2.0, n),
                        torch.linspace(-1.5, 1.5, n)], dim=-1)
    dec.anchor_controls.copy_(ctrl)
    return dec


def test_controls_to_ticks_holds_each_slot_over_its_span():
    """FAILS WITHOUT THE FEATURE. horizons (5, 10, 15, 20) at tick 0.1 means
    slot spans of 5 ticks each, so an 8-slot control expands to 20 ticks."""
    u = torch.arange(4.0).reshape(1, 1, 4, 1).expand(1, 1, 4, 2).clone()
    t = rs.controls_to_ticks(u, (5, 10, 15, 20))
    assert t.shape == (1, 1, 20, 2)
    assert torch.equal(t[0, 0, :5, 0], torch.zeros(5))
    assert torch.equal(t[0, 0, 5:10, 0], torch.ones(5))
    assert torch.equal(t[0, 0, 15:, 0], torch.full((5,), 3.0))


def test_controls_to_ticks_handles_a_NONUNIFORM_grid():
    """CONTROL. The 6 s grid is non-uniform; a version that assumed a uniform
    span would put the control on the wrong ticks and be wrong only late."""
    u = torch.arange(3.0).reshape(1, 1, 3, 1).expand(1, 1, 3, 2).clone()
    t = rs.controls_to_ticks(u, (5, 10, 30))
    assert t.shape == (1, 1, 30, 2)
    assert torch.equal(t[0, 0, 10:, 0], torch.full((20,), 2.0))


@pytest.mark.parametrize("units", ["alat", "kappa"])
def test_constant_control_roll_is_BIT_IDENTICAL_to_roll_bank(units):
    """⛔⛔ THE LOAD-BEARING IDENTITY. The anchor vocabulary holds ONE
    (a_lon, a_lat) pair per anchor and `roll_bank` integrates it at the native
    tick. The sampler's state is a per-slot SEQUENCE; at sigma -> 0 that
    sequence IS the constant pair, so the two must produce the SAME geometry.

    If this ever fails, every anchored-Gaussian claim is measured against a fan
    the vocabulary never emitted — and nothing else in the suite would notice.
    """
    horizons = (5, 10, 15, 20)
    dec = _v0_decoder(units=units, horizons=horizons)
    v0 = torch.tensor([12.0, 3.0, 25.0])
    bank = dec.roll_bank(v0, None, 3, torch.float32)
    u = dec.anchor_control_seq(3, torch.float32)
    rolled = rs.roll_controls(
        u, v0, horizons, control_units=units, tick=dec.anchor_dt,
        alat_v_floor=dec.anchor_alat_v_floor, kappa_cap=dec.anchor_kappa_cap)
    assert rolled.shape == bank.shape
    assert torch.equal(rolled, bank), (
        f"max |diff| = {float((rolled - bank).abs().max()):.3e}")


def test_roll_controls_is_flyable_where_a_METRE_offset_is_not():
    """⭐ FAILS WITHOUT THE FEATURE, and it is the whole design argument in one
    assertion. A metre-space perturbation of the same magnitude DD uses implies
    a lateral acceleration no car can produce; the control-space one cannot,
    because the acceleration IS the state being perturbed."""
    horizons = tuple(range(5, 45, 5))            # 8 slots, 4 s
    n, b = 6, 1
    v0 = torch.tensor([15.0])
    u = torch.zeros(b, n, len(horizons), 2)
    u[..., 0] = 0.5                              # gentle accel
    base = rs.roll_controls(u, v0, horizons, control_units="alat")
    # control-space noise at the schedule's own t = 8 sigma
    s8 = float(rs.DDIMSchedule().sqrt_one_minus_abar(8))
    torch.manual_seed(0)
    u_n = u + torch.randn_like(u) * s8 * torch.tensor([4.0, 3.0])
    ctl = rs.roll_controls(u_n, v0, horizons, control_units="alat")
    # metre-space noise of DD's stated magnitude, applied per waypoint
    torch.manual_seed(0)
    met = base + torch.randn_like(base) * torch.tensor([0.90, 0.73])

    def implied_lat_accel(path):
        """|d^2 y / dt^2| between consecutive slots, dt = 0.5 s."""
        y = path[..., 1]
        d1 = (y[..., 1:] - y[..., :-1]) / 0.5
        return ((d1[..., 1:] - d1[..., :-1]) / 0.5).abs().max()

    a_ctl = float(implied_lat_accel(ctl))
    a_met = float(implied_lat_accel(met))
    assert a_met > 4.0, f"the metre-space arm should be wild, got {a_met:.2f}"
    assert a_ctl < a_met / 2.0, (a_ctl, a_met)


def test_roll_controls_rejects_a_slot_mismatch():
    """CONTROL. A silent shape mismatch here would roll the wrong physics."""
    with pytest.raises(ValueError):
        rs.roll_controls(torch.zeros(1, 2, 3, 2), torch.tensor([10.0]),
                         (5, 10, 15, 20))


def test_alat_to_curvature_has_ONE_spelling():
    """⛔ CONTROL. `roll_bank` must call the SAME function the sampler does —
    two copies is how the two would silently drift apart. This checks the
    arithmetic agrees with the formula the docstring states."""
    a_lat = torch.tensor([[3.0, -3.0]])
    v = torch.tensor([[10.0]])
    k = rs.alat_to_curvature(a_lat, v, 4.0, 0.12)
    assert torch.allclose(k, torch.tensor([[0.03, -0.03]]), atol=1e-6)
    # the floor binds at low speed, the cap binds at high a_lat
    k_slow = rs.alat_to_curvature(torch.tensor([[3.0]]), torch.tensor([[0.5]]),
                                  4.0, 0.12)
    assert float(k_slow) == pytest.approx(0.12)      # capped, not 12.0


# ---------------------------------------------------------------------------
# 3. The decoder integration
# ---------------------------------------------------------------------------
def test_sampler_none_constructs_no_sampler_modules():
    """⛔ CONTROL. `sampler='none'` must construct nothing, or RNG draw order
    changes and every pre-v5 checkpoint stops loading strictly."""
    dec = _v0_decoder(sampler="none")
    assert dec.control_head is None
    assert dec.time_mlp is None
    assert dec.sched is None


def test_sampler_none_is_BYTE_IDENTICAL_to_pre_v5():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE."""
    a = _v0_decoder(sampler="none", seed=11)
    b = _v0_decoder(sampler="none", seed=11)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_ddim_build_adds_exactly_the_expected_modules():
    """FAILS WITHOUT THE FEATURE."""
    dec = _v0_decoder(sampler="ddim")
    assert dec.control_head is not None
    assert dec.time_mlp is not None
    assert dec.sched is not None
    # zero-init: the first refinement pass is the identity
    assert float(dec.control_head.weight.detach().abs().max()) == 0.0
    assert float(dec.control_head.bias.detach().abs().max()) == 0.0


def test_ddim_forward_emits_a_ROLLED_fan_and_u0():
    """FAILS WITHOUT THE FEATURE. Under the sampler the emitted fan comes from
    the integrator, and the predicted clean CONTROL leaves the decoder — it is
    the only tensor the x0 loss can be computed on."""
    dec = _v0_decoder(sampler="ddim").eval()
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    v0 = torch.tensor([12.0, 20.0])
    out = dec(fmap, m, steps=2, v_ms=v0)
    assert out["anchor_traj"].shape == (2, 5, 4, 2)
    assert "u0_hat" in out
    assert out["u0_hat"].shape == (2, 5, 4, 2)
    assert torch.isfinite(out["anchor_traj"]).all()
    assert torch.isfinite(out["u0_hat"]).all()
    assert out["sel_tele"]["sampler"] == "ddim"


def test_zero_init_control_head_starts_the_fan_AT_THE_ANCHORED_GAUSSIAN():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. With `control_head` at its zero
    init, `u0_hat` must equal the NOISED anchor exactly — i.e. the sampler adds
    the anchored Gaussian and nothing else. That is what makes 'the vocabulary
    is the prior' true at step 0 rather than merely intended."""
    dec = _v0_decoder(sampler="ddim").eval()
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    v0 = torch.tensor([12.0, 20.0])
    norm = torch.tensor(dec.cfg.control_norm)
    torch.manual_seed(5)
    out = dec(fmap, m, steps=1, v_ms=v0)
    u_anchor = dec.anchor_control_seq(2, torch.float32)
    # u0_hat = un_t + 0, so in normalised space it is exactly the noised anchor
    dev = (out["u0_hat"] / norm) - (u_anchor / norm)
    s8 = float(dec.sched.sqrt_one_minus_abar(dec.cfg.sampler_infer_t))
    a8 = float(dec.sched.sqrt_abar(dec.cfg.sampler_infer_t))
    # the deviation is (a8 - 1) * un0 + s8 * eps: bounded by the schedule
    bound = abs(a8 - 1.0) * float((u_anchor / norm).abs().max()) + 6.0 * s8
    assert float(dev.abs().max()) <= bound + 1e-5


def test_ddim_refuses_a_FIXED_path_vocabulary():
    """⛔ CONTROL. The sampler's state IS the control sequence. A fixed-path
    bank carries `anchor_controls` of all zeros, so the anchored Gaussian would
    be centred on 'do nothing' — a silent, plausible-looking wrong experiment.
    It must REFUSE, loudly."""
    torch.manual_seed(0)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                             sampler="ddim")
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=4, d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=torch.randn(5, 4, 2), cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=(5, 10, 15, 20),
        v0_conditioned=False)
    with pytest.raises(ValueError, match="v0-CONDITIONED"):
        dec(torch.randn(1, 16, 3, 5), torch.randn(1, 8), steps=2)


def test_sampler_groups_gt_one_REFUSES_rather_than_mis_indexing():
    """⛔ CONTROL. G > 1 emits a [B, G*N, ...] fan while `loss_cls`'s a_star,
    the [B, N] priors and every `sel_idx` dump still assume N. Silently
    widening the fan would MIS-INDEX the anchor target. Refuse, and name the
    three call sites in the message."""
    torch.manual_seed(0)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                             sampler="ddim", sampler_groups=4)
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=4, d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=torch.randn(5, 4, 2), cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=(5, 10, 15, 20),
        v0_conditioned=True)
    with pytest.raises(NotImplementedError, match="loss_cls"):
        dec(torch.randn(1, 16, 3, 5), torch.randn(1, 8), steps=2,
            v_ms=torch.tensor([10.0]))


def test_ddim_backward_is_finite():
    """FAILS WITHOUT THE FEATURE: the sampler must be trainable. ~60 sequential
    integration steps in the graph is exactly where a NaN would appear."""
    dec = _v0_decoder(sampler="ddim").train()
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    out = dec(fmap, m, steps=2, v_ms=torch.tensor([12.0, 20.0]))
    (out["anchor_traj"].square().mean() + out["u0_hat"].square().mean()
     ).backward()
    grads = [p.grad for p in dec.parameters() if p.grad is not None]
    assert grads, "no gradient reached the decoder"
    for g in grads:
        assert torch.isfinite(g).all()
    assert dec.control_head.weight.grad is not None
    assert float(dec.control_head.weight.grad.abs().max()) > 0.0


def test_metre_space_arm_is_REACHABLE():
    """⛔ THE DELIBERATE REGRESSION, as a RUNNABLE code path and not a
    description. `sampler_space='metre'` is the DD-literal arm, pre-registered
    to FAIL the flyability gate. An arm that cannot be run cannot fail, and a
    regression that is only described has never been tested."""
    torch.manual_seed(0)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                             sampler="ddim", sampler_space="metre")
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=4, d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=torch.randn(5, 4, 2), cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=(5, 10, 15, 20),
        v0_conditioned=True).eval()
    out = dec(torch.randn(1, 16, 3, 5), torch.randn(1, 8), steps=2,
              v_ms=torch.tensor([10.0]))
    assert out["sel_tele"]["sampler_space"] == "metre"
    assert torch.isfinite(out["anchor_traj"]).all()


def test_time_embedding_is_CONTINUOUS_under_the_sampler():
    """FAILS WITHOUT THE FEATURE. The 3-row `nn.Embedding` cannot represent
    t ~ U[0, 50); DD injects a sinusoidal embedding of the CONTINUOUS t. Two
    different timesteps must give two different embeddings."""
    mlp = rs.build_time_mlp(32)
    e = mlp(torch.tensor([0.0, 8.0, 49.0]))
    assert e.shape == (3, 32)
    assert not torch.allclose(e[0], e[1])
    assert not torch.allclose(e[1], e[2])


def test_sinusoidal_emb_rejects_an_odd_width():
    """CONTROL: a silent off-by-one in the half-split would give a truncated
    embedding that still looks like a tensor."""
    with pytest.raises(ValueError):
        rs.SinusoidalPosEmb(31)


def test_hfov_style_sanity_the_sampler_does_not_touch_the_classifier_pass():
    """⛔ CONTROL. `anchor_logits` is the t = 0 classifier surface and is what
    `loss_cls` supervises. The sampler must not change it — otherwise the
    anchor target and the classifier drift apart."""
    dec_a = _v0_decoder(sampler="none", seed=21).eval()
    dec_b = _v0_decoder(sampler="ddim", seed=21).eval()
    # copy the shared weights so only the sampler differs
    sa = dec_a.state_dict()
    sb = dec_b.state_dict()
    for k in sa:
        if k in sb and sa[k].shape == sb[k].shape:
            sb[k] = sa[k].clone()
    dec_b.load_state_dict(sb)
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    v0 = torch.tensor([12.0, 20.0])
    with torch.no_grad():
        oa = dec_a(fmap, m, steps=2, v_ms=v0)
        ob = dec_b(fmap, m, steps=2, v_ms=v0)
    assert torch.equal(oa["anchor_logits"], ob["anchor_logits"])
    assert torch.equal(oa["anchor_bank"], ob["anchor_bank"])


def test_metre_space_arm_is_UNFLYABLE_and_the_control_arm_is_NOT():
    """⛔⛔ THE DELIBERATE REGRESSION MUST BE ABLE TO **FAIL**, not merely to
    RUN. `test_metre_space_arm_is_REACHABLE` above proves the code path exists;
    this proves the path produces the pathology it is pre-registered to produce.

    MEASURED 2026-09-06 — and the first implementation FAILED THIS. It used
    `metre_sigma_m` directly as a divisor, delivering `sigma(8) * 0.90 =
    0.028 m` of noise: **31.7x too gentle**, so the DD-literal arm would have
    sailed through the flyability gate and the whole comparison would have read
    'metre space is fine'. The normaliser must be DERIVED
    (`metre_sigma_m / sqrt(1 - abar(infer_t))` = 28.49 / 23.11) so the emitted
    noise IS DD's published 0.90 m / 0.73 m per waypoint.

    At the zero-init `control_head` this measures the SAMPLER's noise alone.
    """
    hz = tuple(range(5, 45, 5))          # 8 slots, dt = 0.5 s
    n = 32

    def mk(space):
        torch.manual_seed(0)
        cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                                 sampler="ddim", sampler_space=space)
        d = refc.AnchoredDiffusionDecoder(
            feat_dim=16, n_steps=len(hz), d_meas=8, d_ctx=4, tac_latent_dim=4,
            anchors=torch.randn(n, len(hz), 2), cfg=cfg, hierarchy=False,
            graft_maneuver=False, graft_target_latent=False,
            grounded_selector=False, horizons=hz, v0_conditioned=True,
            control_units="alat").eval()
        torch.manual_seed(1)
        d.anchor_controls.copy_(torch.stack(
            [torch.linspace(-1.0, 1.0, n), torch.linspace(-1.0, 1.0, n)], -1))
        return d

    def implied_alat(p):
        y = p[..., 1]
        d1 = (y[..., 1:] - y[..., :-1]) / 0.5
        return ((d1[..., 1:] - d1[..., :-1]) / 0.5).abs()

    fm, m, v = torch.randn(1, 16, 3, 5), torch.randn(1, 8), torch.tensor([15.0])
    got = {}
    for space in ("control", "metre"):
        with torch.no_grad():
            torch.manual_seed(7)
            got[space] = implied_alat(
                mk(space)(fm, m, steps=2, v_ms=v)["anchor_traj"])
    mu = 0.7 * 9.81
    ctl, met = got["control"], got["metre"]
    # the regression IS wild: DD's 0.73 m over a 0.5 s slot implies ~5.8 m/s^2
    assert float(met.mean()) > 3.0, float(met.mean())
    assert float((met > mu).float().mean()) > 0.05,         "the metre arm must break the friction circle, or it cannot fail the gate"
    # ...and the control-space arm is NOT, because the acceleration IS the state
    assert float(ctl.mean()) < float(met.mean()) / 4.0, (float(ctl.mean()),
                                                         float(met.mean()))
    assert float((ctl > mu).float().mean()) == 0.0,         "control-space samples are flyable BY CONSTRUCTION"
