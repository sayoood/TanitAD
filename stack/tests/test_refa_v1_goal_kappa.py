"""`goal_kappa_turn` — the goal's SUSTAINED curvature, and its parity guarantee.

⭐ WHY THIS FILE EXISTS. MEASURED 2026-09-05
(`.../2026-09-05-refav1-goal-margin/ROOT_CAUSE.md`, 4786 windows / 141
episodes): `canonical_controls` can command exactly TWO sustained curvatures,
`0` and `±GOAL_KAPPA_TURN = 0.08` (R 12.5 m), because `NUDGE_*` and
`LANE_CHANGE_*` are S-curves (`+kap` then `−kap`) whose NET heading change is
zero. The corpus curves at R 100–1000 m. Consequence: on **90.6 %** of GT-turn
windows `LANE_KEEP` is the **vocabulary-optimal** token, and the head emits
`LANE_KEEP` on 85.2 % of ALL windows against a vocabulary-optimal 95.6 % — i.e.
**the head already turns MORE than its vocabulary justifies.** So the gate on
refav1's turning is the VOCABULARY, and it needed to become a knob.

⛔ THE PARITY CLAIM IS THE LOAD-BEARING ONE, exactly as for `lat_logit_bias`.
Every arm banked before this change ran `GOAL_KAPPA_TURN`. If the new argument
perturbed that path by a float, every cross-arm comparison in the programme
would silently stop being a comparison. `test_a_*` pins it.

⚠️ And parity tests rot into vacuity — a test that passes because nothing ran
proves nothing. Every parity assertion here is paired with a SAME-BREATH control
that MUST show a difference.
"""
from __future__ import annotations

import torch

from tanitad.refs.refa_v1 import (RefAV1, RefAV1Config, StrategicPolicyConfig,
                                  TacticalPolicyConfig, canonical_controls,
                                  GOAL_KAPPA_TURN, GOAL_TURN_S)
from tanitad.refs.refa_v1_plan import PlanConfig
from tanitad.models.v6 import tactical_lat_actions

LAT = list(tactical_lat_actions("v7.0"))
K, DT = 30, 0.2


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


def _window(c: RefAV1Config, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, c.op_window, c.n_tokens, c.d_enc, generator=g),
            10.0, torch.randint(0, 4, (1,), generator=g))


def _pc(c: RefAV1Config) -> PlanConfig:
    return PlanConfig(horizon=c.plan_steps, dt=c.op_dt, seed=0,
                      n_samples=32, n_iters=3, n_elites=8)


def _plan(m, feats, v0, nav, **kw):
    """⚠️ `cost_metric="ccos"` is not a stylistic choice — MEASURED
    2026-09-05, under `"cos"` at the shipped weights a CORRECTLY decoded TURN
    executes on 0 of 12 tiny-model windows at every search size and 0 of 38 real
    windows. A curvature test written against `"cos"` would assert that nothing
    happens and pass for the wrong reason (`D-REFAV1-DRIVE-GATE2`)."""
    kw.setdefault("cost_metric", "ccos")
    return m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg), **kw)


# ------------------------------------------------------------------ #
# A. PARITY — absent must be bit-identical to the shipped constant
# ------------------------------------------------------------------ #
def test_a_canonical_controls_absent_is_bit_identical():
    for lat in LAT:
        base = canonical_controls(lat, "CRUISE", 10.0, K, DT)
        none = canonical_controls(lat, "CRUISE", 10.0, K, DT, kappa_turn=None)
        explicit = canonical_controls(lat, "CRUISE", 10.0, K, DT,
                                      kappa_turn=GOAL_KAPPA_TURN)
        assert torch.equal(base, none), f"{lat}: None perturbed the legacy path"
        assert torch.equal(base, explicit), f"{lat}: explicit shipped != legacy"

    # ⛔ SAME-BREATH CONTROL: a DIFFERENT value must actually change a TURN,
    # else the assertions above could pass because the argument is ignored.
    changed = canonical_controls("TURN_L", "CRUISE", 10.0, K, DT,
                                 kappa_turn=0.02)
    assert not torch.equal(canonical_controls("TURN_L", "CRUISE", 10.0, K, DT),
                           changed), "kappa_turn is being ignored"


def test_a2_only_TURN_tokens_are_affected():
    """The knob must touch the SUSTAINED profile only. S-curve tokens size
    themselves from `v_ref` and must be untouched — otherwise this would
    silently re-scale lane changes too."""
    for lat in LAT:
        a = canonical_controls(lat, "CRUISE", 10.0, K, DT)
        b = canonical_controls(lat, "CRUISE", 10.0, K, DT, kappa_turn=0.02)
        if lat.startswith("TURN_"):
            assert not torch.equal(a, b), f"{lat} should have changed"
        else:
            assert torch.equal(a, b), f"{lat} must NOT be affected"
    # control: at least one token in each branch actually exists
    assert any(t.startswith("TURN_") for t in LAT)
    assert any(not t.startswith("TURN_") for t in LAT)


def test_b_emitted_curvature_is_exactly_what_was_asked():
    n_hold = int(round(GOAL_TURN_S / DT))
    for kt in (0.005, 0.02, 0.08, 0.15):
        for lat, sgn in (("TURN_L", 1.0), ("TURN_R", -1.0)):
            k = canonical_controls(lat, "CRUISE", 10.0, K, DT, kappa_turn=kt)[:, 1]
            assert torch.allclose(k[:n_hold],
                                  torch.full((n_hold,), sgn * kt)), (lat, kt)
            assert torch.equal(k[n_hold:], torch.zeros(K - n_hold)), (lat, kt)


def test_c_longitudinal_channel_is_untouched():
    """A lateral knob that moved the acceleration column would confound every
    longitudinal metric. Column 0 must be identical for every token."""
    for lat in LAT:
        for lon in ("CRUISE", "BRAKE_TO", "HOLD", "ACCELERATE"):
            a = canonical_controls(lat, lon, 10.0, K, DT)[:, 0]
            b = canonical_controls(lat, lon, 10.0, K, DT, kappa_turn=0.02)[:, 0]
            assert torch.equal(a, b), f"{lat}/{lon} accel changed"


# ------------------------------------------------------------------ #
# D. THROUGH plan() — the path an arm actually takes
# ------------------------------------------------------------------ #
def test_d_plan_absent_is_bit_identical():
    m = _model()
    feats, v0, nav = _window(m.cfg)
    a = _plan(m, feats, v0, nav)
    b = _plan(m, feats, v0, nav, goal_kappa_turn=None)
    c = _plan(m, feats, v0, nav, goal_kappa_turn=GOAL_KAPPA_TURN)
    assert torch.equal(a.controls, b.controls), "None perturbed plan()"
    assert torch.equal(a.controls, c.controls), "explicit shipped != legacy"
    assert a.goal_action["lat"] == b.goal_action["lat"]


def test_e_result_is_stamped():
    """A banked window must say which curvature its goal was built from, or two
    arms differing ONLY in this are indistinguishable in their dumps."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    assert _plan(m, feats, v0, nav).goal_kappa_turn is None
    assert _plan(m, feats, v0, nav, goal_kappa_turn=0.02).goal_kappa_turn == 0.02


def test_f_the_goal_field_actually_moves_on_a_turn_decode():
    """⛔ THE CONTROL THAT STOPS THIS WHOLE FILE BEING VACUOUS. If no seed ever
    decodes a TURN, every parity assertion above holds trivially. Force the
    decode with `lat_logit_bias` and require the GOAL CONTROLS to carry the
    requested curvature."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    force = torch.full((len(LAT),), -50.0)
    force[LAT.index("TURN_L")] = 50.0
    shipped = _plan(m, feats, v0, nav, lat_logit_bias=force)
    finer = _plan(m, feats, v0, nav, lat_logit_bias=force, goal_kappa_turn=0.02)
    assert shipped.goal_action["lat"] == "TURN_L"
    assert finer.goal_action["lat"] == "TURN_L"
    ks = shipped.goal_action["controls"][:, 1].abs().max().item()
    kf = finer.goal_action["controls"][:, 1].abs().max().item()
    assert abs(ks - GOAL_KAPPA_TURN) < 1e-6, ks
    assert abs(kf - 0.02) < 1e-6, kf
    # and the goal FIELD the search is scored against must differ too
    assert not torch.equal(shipped.goal_action["controls"],
                           finer.goal_action["controls"])
