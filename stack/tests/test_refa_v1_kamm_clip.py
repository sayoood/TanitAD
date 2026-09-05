"""D-REFAV1-COST-GEOMETRY / L4 - the FRICTION-CIRCLE (Kamm) CURVATURE CAP.

⛔ THE DEFECT, MEASURED. `PlanConfig.kappa_max` is a CONSTANT 0.2 1/m, so the
search may command the same curvature at 30 m/s as at 2 m/s. The tyre may not:
`a_lat = v^2 * kappa` has to stay inside `mu * g`.

Audited with the sibling stream's own scorer-derived instrument
(`tanitad.refs.feasible_decode.assert_feasible`, which re-derives the SCORER's
flags from a path rather than trusting any bookkeeping) over the banked p4 dumps
at `v0 >= 2 m/s`, n = 27 windows. ⭐ THE CONTROL PASSES AT THAT CUT: the
GROUND-TRUTH path reads `envelope_rate` **0.0000** and `kamm_over_rate`
**0.0000** (at `vmin = 0` it does NOT - `kappa = a_lat/v^2` explodes on
near-stationary windows and reads `max|kappa| 31.4` - so that block is
inadmissible and is reported as such).

  * refav1 `cl` is ENVELOPE-feasible BY CONSTRUCTION: `envelope_rate`
    **0.0000**, `max|kappa|` exactly **0.2000** = the clip.
  * and yet **29.6 %** of its plans leave the `mu = 0.7` friction circle,
    **42.1 %** at `v0 >= 5 m/s`, with `peak_g` up to **3.262** against a ground
    truth of **0.373**.
  * sustaining `GOAL_KAPPA_TURN = 0.08` at 20 m/s is `0.08 * 400 = 32 m/s^2
    = 3.26 g` - the observed maximum, recovered by an independent route.

⭐ THE CAP is `mu*g/v^2` on the CANDIDATE'S OWN speed profile (`v0` integrated
through its accel channel), so a plan that brakes into a curve is allowed the
curvature its braking earns. At 20 m/s, `mu = 0.7`: **0.0172 1/m**, 11.6x
tighter than the constant clip. Unlike a quadratic `W_KAPPA` penalty it is a
CONSTRAINT WITH UNITS, and it cannot be reproduced by any single weight.

TIER: n/a (arithmetic + a tiny random-init model). EVIDENCE CLASS: MEASURED.
"""
import math

import pytest
import torch

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config
from tanitad.refs.refa_v1_plan import G_MPS2, PlanConfig, _clip


def _cfg(**kw) -> RefAV1Config:
    base = dict(tac_vocab_version="v7.0", d_enc=16, d_state=16, n_tokens=8,
                op_layers=1, op_heads=2, op_window=2, tac_layers=1,
                tac_queries=4, str_dim=8, str_layers=1,
                strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1,
                                                    n_heads=2, d_ctx=16,
                                                    d_cmd=8),
                tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1,
                                                  n_heads=2, d_intent=16))
    base.update(kw)
    return RefAV1Config(**base)


def _model(seed: int = 0) -> RefAV1:
    torch.manual_seed(seed)
    m = RefAV1(_cfg()).eval()
    m.std.fit(torch.randn(256, m.cfg.d_enc))
    return m


def _pc(c: RefAV1Config, **kw) -> PlanConfig:
    base = dict(horizon=c.plan_steps, dt=c.op_dt, seed=0,
                n_samples=8, n_iters=2, n_elites=4)
    base.update(kw)
    return PlanConfig(**base)


def _window(c: RefAV1Config, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, c.op_window, c.n_tokens, c.d_enc, generator=g),
            10.0, torch.randint(0, 4, (1,), generator=g))


# --------------------------------------------------------------------------- #
#  A - OFF by default, bit-identical                                           #
# --------------------------------------------------------------------------- #
def test_a_default_is_off_and_clip_is_bit_identical():
    pc = PlanConfig(horizon=6, dt=0.2)
    assert pc.kamm_mu is None
    g_ = torch.Generator().manual_seed(4)
    c = torch.randn(32, 6, 2, generator=g_) * 3.0
    a = _clip(c, pc)
    b = _clip(c, pc, v0=25.0)          # v0 supplied but the cap is OFF
    ref = torch.stack([c[..., 0].clamp(-pc.a_max, pc.a_max),
                       c[..., 1].clamp(-pc.kappa_max, pc.kappa_max)], dim=-1)
    assert torch.equal(a, ref)
    assert torch.equal(b, ref)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_a2_plan_is_bit_identical_when_the_cap_is_off(seed):
    m = _model(seed=seed)
    feats, v0, nav = _window(m.cfg, seed=seed)
    with torch.no_grad():
        x = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg))
        y = m.plan(feats, v0=v0, nav_cmd=nav,
                   plan_cfg=_pc(m.cfg, kamm_mu=None))
    assert torch.equal(x.controls, y.controls)


# --------------------------------------------------------------------------- #
#  B - the cap reads its KNOWN VALUE                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("v,mu", [(20.0, 0.7), (30.0, 0.7), (10.0, 0.7),
                                  (20.0, 1.0)])
def test_b_cap_equals_mu_g_over_v_squared(v, mu):
    """The KNOWN VALUE: with zero accel the cap is exactly `mu*g/v0^2`, and it
    is applied to EVERY step. At v = 20, mu = 0.7 that is 0.017162 1/m."""
    pc = PlanConfig(horizon=5, dt=0.2, kamm_mu=mu)
    c = torch.zeros(1, 5, 2)
    c[..., 1] = pc.kappa_max              # ask for the constant clip's maximum
    out = _clip(c, pc, v0=v)
    want = min(pc.kappa_max, mu * G_MPS2 / (v ** 2))
    # float32 carries ~6e-8 relative resolution: a tighter tolerance would be a
    # statement about the dtype, not about the cap.
    assert torch.allclose(out[..., 1], torch.full((1, 5), want),
                          rtol=1e-6, atol=0.0)
    assert math.isclose(float(out[0, 0, 1]), want, rel_tol=1e-6)


def test_b2_the_headline_number():
    """20 m/s, mu = 0.7 -> 0.017162 1/m, i.e. 11.65x tighter than 0.2."""
    pc = PlanConfig(horizon=3, dt=0.2, kamm_mu=0.7)
    c = torch.zeros(1, 3, 2); c[..., 1] = 0.2
    out = float(_clip(c, pc, v0=20.0)[0, 0, 1])
    assert math.isclose(out, 0.7 * G_MPS2 / 400.0, rel_tol=1e-6)
    assert 0.0171 < out < 0.0172
    assert pc.kappa_max / out > 11.0


def test_c_below_the_floor_the_constant_clip_governs():
    """CONTROL that must read a KNOWN value: at a crawl the friction circle is
    NOT the binding constraint and the cap must not bite."""
    pc = PlanConfig(horizon=4, dt=0.2, kamm_mu=0.7, kamm_v_floor=2.0)
    c = torch.zeros(1, 4, 2); c[..., 1] = 0.2
    out = _clip(c, pc, v0=0.1)
    assert torch.allclose(out[..., 1], torch.full((1, 4), 0.2), atol=1e-9)


def test_d_a_braking_candidate_EARNS_curvature():
    """The cap is per-step on the candidate's OWN speed, so a plan that slows is
    allowed more curvature later than a plan that does not. A constant cap
    cannot express this and neither can any single `W_KAPPA`."""
    pc = PlanConfig(horizon=5, dt=0.2, kamm_mu=0.7)
    brake = torch.zeros(1, 5, 2)
    brake[..., 0] = -4.0                    # hard decel
    brake[..., 1] = 0.2
    hold = torch.zeros(1, 5, 2)
    hold[..., 1] = 0.2
    kb = _clip(brake, pc, v0=25.0)[0, :, 1]
    kh = _clip(hold, pc, v0=25.0)[0, :, 1]
    assert torch.all(kb[1:] > kh[1:] - 1e-12)
    # 25 m/s held vs braking at -4 m/s^2: the last step opens at 21.8 m/s, so
    # the KNOWN values are 0.7*g/625 = 0.010983 and 0.7*g/475.24 = 0.014445
    # -- a 1.315x gain, which is what "earning curvature" is worth here.
    assert math.isclose(float(kh[-1]), 0.7 * G_MPS2 / (25.0 ** 2), rel_tol=1e-6)
    assert math.isclose(float(kb[-1]), 0.7 * G_MPS2 / (21.8 ** 2), rel_tol=1e-5)
    assert float(kb[-1]) / float(kh[-1]) > 1.3
    assert math.isclose(float(kh[0]), float(kb[0]), rel_tol=1e-6)   # same v0


def test_e_sign_is_preserved_and_the_cap_is_symmetric():
    pc = PlanConfig(horizon=3, dt=0.2, kamm_mu=0.7)
    c = torch.zeros(2, 3, 2)
    c[0, :, 1] = 0.2
    c[1, :, 1] = -0.2
    out = _clip(c, pc, v0=20.0)[..., 1]
    assert float(out[0, 0]) > 0 and float(out[1, 0]) < 0
    assert math.isclose(float(out[0, 0]), -float(out[1, 0]), rel_tol=1e-12)


def test_f_a_curvature_already_inside_the_circle_is_untouched():
    """The cap must CLAMP, never rescale: a legal 0.01 stays 0.01 exactly."""
    pc = PlanConfig(horizon=3, dt=0.2, kamm_mu=0.7)
    c = torch.zeros(1, 3, 2); c[..., 1] = 0.01
    out = _clip(c, pc, v0=20.0)
    assert torch.equal(out[..., 1], c[..., 1])


def test_g_sanity_refuses_a_nonsense_mu():
    with pytest.raises(ValueError, match="kamm_mu"):
        PlanConfig(horizon=3, dt=0.2, kamm_mu=0.0).sanity()
    with pytest.raises(ValueError, match="kamm_mu"):
        PlanConfig(horizon=3, dt=0.2, kamm_mu=-0.7).sanity()
    with pytest.raises(ValueError, match="kamm_v_floor"):
        PlanConfig(horizon=3, dt=0.2, kamm_mu=0.7, kamm_v_floor=0.0).sanity()
    PlanConfig(horizon=3, dt=0.2, kamm_mu=0.7).sanity()          # the happy path


def test_h_end_to_end_a_capped_plan_never_leaves_the_circle():
    """THE ARM-LEVEL ASSERTION, and its same-breath CONTROL. Under the cap every
    planned step must satisfy `v^2*|kappa| <= mu*g` (+tol); the UNCAPPED control
    on the same window must be free to exceed it, or the window proves nothing."""
    m = _model(seed=21)
    feats, _v0, nav = _window(m.cfg, seed=21)
    v0 = 22.0
    with torch.no_grad():
        capped = m.plan(feats, v0=v0, nav_cmd=nav,
                        plan_cfg=_pc(m.cfg, kamm_mu=0.7))
        free = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg))

    def worst_g(ctrl):
        a = ctrl[:, 0]
        v = v0 + torch.cumsum(a, 0) * m.cfg.op_dt - a * m.cfg.op_dt
        v = v.clamp_min(2.0)
        return float((v.pow(2) * ctrl[:, 1].abs()).max() / G_MPS2)

    assert worst_g(capped.controls) <= 0.7 + 1e-6, (
        f"a capped plan left the friction circle: {worst_g(capped.controls)} g")
    assert float(free.controls[:, 1].abs().max()) <= _pc(m.cfg).kappa_max + 1e-9
