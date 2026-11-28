"""E15 / S7 (GP-1 + GP-2) — the METRIC GOAL POINT on the RANKED SCORE.

⛔ WHAT THIS FILE IS FOR, AND WHY IT IS NOT `test_goal_point_wiring.py`.
That file proves the CONDITIONING edge (`GoalPointConditioning` -> z_tac / ctx)
is built, bit-inert at init and reachable once trained. This one proves the
SELECTION seam — the param-free geometric term that reaches `refc.py`'s ranked
score behind one zero-init gate — and it proves the three things a zero-init
check cannot give you:

1. the seam is REACHABLE (opening the gate moves the ranked score), so a
   `gate = 0.0000` reading at 40 k is a fact about training, not a dead wire;
2. the seam responds to the goal's VALUE, not merely its presence — the E13
   failure mode this whole edge exists to remove (content 2.3 %,
   presence 97.7 %);
3. the LEAK GUARD is a CONSTRUCTOR REFUSAL and is LOAD-BEARING — proved by
   MUTATION, i.e. by removing it and showing the leaking arm then becomes
   constructible. A guard nobody has watched fail is a comment.

⛔⛔ AND IT PINS THE UNITS. MEASURED 2026-09-06 while wiring GP-1: the head
emits a NORMALISED point and `anchor_goal_prior_at_time` consumes METRES, so
feeding the head's output straight in (which is what PREREG §8's draft snippet
did) put a 24 m goal at 2.5 m. Everything stayed finite and the seam trained;
a MIRRORED goal then moved the ranked score by 7.7e-4 and flipped 0 of 4 picks
on a rig whose anchors span ±66–82 m laterally. That is the pre-registered
value-sensitivity gate failing for a UNITS reason and reading as "the geometric
prior ignores the goal". `test_UNITS_*` below is the regression pin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.refs import goal_point as gp                      # noqa: E402
from tanitad.refs import refc                                  # noqa: E402
from tanitad.refs import refc_v3 as v3                         # noqa: E402


# ---------------------------------------------------------------------------
# rig
# ---------------------------------------------------------------------------

def _build(inject: bool = True, geo: bool = True, seed: int = 0):
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.goal_point_inject = inject
    cfg.goal_point_cfg = gp.GoalPointConfig(d_goal=8)
    cfg.core.graft_gp_point = geo
    torch.manual_seed(seed)
    return cfg, v3.RefCV3Model(cfg).eval()


def _frames(cfg, b: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.core.encoder.image_hw()
    return torch.rand(b, cfg.core.window, cfg.core.encoder.in_channels, h, w,
                      generator=g)


V0 = torch.tensor([3.0, 7.0, 5.0, 9.0])
#: 20 m ahead, 12 m LEFT, in the head's NORMALISED units (scale 40 m).
LEFT = torch.tensor([[0.5, 0.30]] * 4)
RIGHT = LEFT * torch.tensor([1.0, -1.0])          # the MIRROR


# ---------------------------------------------------------------------------
# 1. the seam is built only when asked, and costs ONE parameter
# ---------------------------------------------------------------------------

def test_the_gate_is_absent_by_default_so_the_live_build_did_not_move():
    assert refc.SelectionConfig().graft_gp_point is False
    assert refc.RefCConfig().graft_gp_point is False
    _, m = _build(inject=True, geo=False)
    assert m.core.decoder.gp_point_gate is None
    assert not [k for k in m.state_dict() if "gp_point_gate" in k]
    # same-breath POSITIVE control: the ON build DOES carry it, so a passing
    # OFF assertion is not just "the attribute name is misspelled".
    _, m2 = _build(inject=True, geo=True)
    assert m2.core.decoder.gp_point_gate is not None
    assert [k for k in m2.state_dict() if "gp_point_gate" in k]


def test_the_whole_selection_seam_costs_EXACTLY_ONE_parameter():
    """The score is param-free; only the gate is learned. A seam that quietly
    grew a head would make the E13-vs-E15 comparison a CAPACITY comparison."""
    _, off = _build(inject=True, geo=False)
    _, on = _build(inject=True, geo=True)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on - n_off == 1, (n_on, n_off)


def test_the_slot_and_scale_are_DERIVED_and_land_on_the_decoder():
    cfg, m = _build()
    # t_goal_s = 4.0 s at 10 Hz -> 40 ticks -> index 5 of V3_HORIZONS
    assert m.gp_slot == 5 == cfg.core.gp_slot == m.core.decoder.gp_slot
    assert m.core.decoder.gp_scale_m == cfg.goal_point_cfg.range_norm_m == 40.0


# ---------------------------------------------------------------------------
# 2. bit-inert at init, REACHABLE once the gate opens
# ---------------------------------------------------------------------------

def test_the_seam_is_BIT_INERT_at_init():
    cfg, m = _build()
    with torch.no_grad():
        out = m(_frames(cfg), v0=V0, steps=2)
    assert torch.equal(out["traj"], out["traj_base"])
    assert torch.equal(out["sel_idx"], out["sel_idx_base"])
    assert float(out["gp_point_gate_value"]) == 0.0


def test_MUTATION_opening_the_gate_moves_the_RANKED_SCORE():
    """⛔ THE HALF ZERO-INIT CANNOT GIVE YOU. An edge that is bit-inert at init
    AND unreachable after training is a dead wire that reads as a clean
    ablation — and it would read as `gate 0.0000` in the log either way."""
    cfg, m = _build()
    f = _frames(cfg)
    with torch.no_grad():
        before = m(f, v0=V0, steps=2)["sel_score"].clone()
        m.core.decoder.gp_point_gate.fill_(2.0)
        after = m(f, v0=V0, steps=2)["sel_score"]
    assert not torch.equal(before, after)
    assert float((before - after).abs().max()) > 1e-3


def test_the_gate_receives_GRADIENT_so_it_can_learn_to_open():
    """d(graft)/d(gate) = score != 0. `goal_gate` carries the same argument in
    refc_v3.py and it is the Caveat-B question in code: "not opened YET" vs
    "structurally cannot"."""
    cfg, m = _build()
    m.train()
    out = m(_frames(cfg), v0=V0, steps=2)
    out["sel_score"].sum().backward()
    g = m.core.decoder.gp_point_gate.grad
    assert g is not None and float(g.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 3. ⛔⛔ THE UNITS — the defect this wiring actually hit
# ---------------------------------------------------------------------------

def test_UNITS_the_prior_takes_METRES_and_the_head_emits_NORMALISED():
    """A KNOWN VALUE, not a shape check. `bank[:, :, slot]` is in metres; a
    goal at normalised (0.5, 0.3) with scale 40 is (20 m, 12 m). Feeding the
    NORMALISED pair straight in scores against a goal 2.5 m from the car."""
    cfg, m = _build()
    with torch.no_grad():
        bank = m(_frames(cfg), v0=V0, steps=2)["anchor_bank"]
    slot, scale = m.core.decoder.gp_slot, m.core.decoder.gp_scale_m
    ones = torch.ones(bank.shape[0])
    conv = gp.anchor_goal_prior_at_time_from_norm(LEFT, ones, bank, slot,
                                                  scale)
    metres = gp.anchor_goal_prior_at_time(LEFT * scale, ones, bank, slot,
                                          scale)
    raw = gp.anchor_goal_prior_at_time(LEFT, ones, bank, slot, scale)
    assert torch.allclose(conv, metres), "the helper must BE the metre form"
    assert not torch.allclose(conv, raw, atol=1e-3), (
        "the normalised and metre readings must differ — if they do not, this "
        "rig cannot detect the units defect and the pin is vacuous")
    # the known value: distance from the goal to a hand-computed anchor point
    p = bank[0, 3, slot]                                    # [2] metres
    d = float(torch.linalg.vector_norm(p - LEFT[0] * scale))
    assert float(conv[0, 3]) == pytest.approx(-d / scale, abs=1e-5)


def test_UNITS_the_decoder_uses_the_CONVERTING_form():
    """The seam inside `refc.py` must agree with the helper to the last bit —
    otherwise this file pins a function the model does not call."""
    cfg, m = _build()
    f = _frames(cfg)
    with torch.no_grad():
        m.core.decoder.gp_point_gate.fill_(0.0)
        base = m(f, v0=V0, steps=2, gp_point=LEFT)
        m.core.decoder.gp_point_gate.fill_(1.0)
        opened = m(f, v0=V0, steps=2, gp_point=LEFT)
        expect = gp.anchor_goal_prior_at_time_from_norm(
            LEFT, torch.ones(4), base["anchor_bank"],
            m.core.decoder.gp_slot, m.core.decoder.gp_scale_m)
    got = opened["sel_score"] - base["sel_score"]
    assert torch.allclose(got, expect.to(got.dtype), atol=1e-4), (
        float((got - expect).abs().max()))


# ---------------------------------------------------------------------------
# 4. VALUE sensitivity — the E13 failure mode, tested for directly
# ---------------------------------------------------------------------------

def test_a_MIRRORED_goal_moves_the_S7_RANKING_not_only_the_score():
    """⛔ E13's nav moved `g_str` 0.0103 on a VALUE change and 0.1902 on
    REMOVAL — 18.6x, a presence-gated bias. A param-free geometric score has no
    such degree of freedom: the term's OWN argmax must follow the goal."""
    cfg, m = _build()
    with torch.no_grad():
        bank = m(_frames(cfg), v0=V0, steps=2)["anchor_bank"]
    slot, scale = m.core.decoder.gp_slot, m.core.decoder.gp_scale_m
    ones = torch.ones(bank.shape[0])
    a = gp.anchor_goal_prior_at_time_from_norm(LEFT, ones, bank, slot, scale)
    b = gp.anchor_goal_prior_at_time_from_norm(RIGHT, ones, bank, slot, scale)
    assert not torch.equal(a.argmax(1), b.argmax(1)), (
        "a mirrored goal must not pick the same anchor")
    lat_a = bank[torch.arange(bank.shape[0]), a.argmax(1), slot, 1]
    lat_b = bank[torch.arange(bank.shape[0]), b.argmax(1), slot, 1]
    assert float((lat_b - lat_a).mean()) < 0.0, (
        "mirroring the goal to the RIGHT must move the picked anchor's "
        "lateral offset in the negative direction")


def test_a_MIRRORED_goal_moves_the_FINAL_PICK_once_the_seam_dominates():
    """⚠️ SCOPE, STATED. On a randomly-initialised 20-anchor smoke model the
    base confidence spread is ~47 while the S7 term's is ~2.5, so a mirrored
    goal at a small gate correctly does NOT flip the argmax. That is not
    evidence about the lever — it is evidence that this rig is not the surface
    the PREREG §4 gate is measured on (the live 117-anchor fan is). What this
    rig CAN prove is that the pathway is not severed: at a gate where the term
    dominates, the pick follows the goal."""
    cfg, m = _build()
    f = _frames(cfg)
    with torch.no_grad():
        m.core.decoder.gp_point_gate.fill_(1e3)
        a = m(f, v0=V0, steps=2, gp_point=LEFT)
        b = m(f, v0=V0, steps=2, gp_point=RIGHT)
    assert not torch.equal(a["sel_idx"], b["sel_idx"])
    assert not torch.equal(a["traj"], b["traj"])


def test_valid_ZERO_makes_the_term_EXACTLY_zero_the_PRESENCE_control():
    """X15: "withheld" and "genuinely zero" differ in the FLAG, and that only
    works if the values really are zero when the flag is. A withheld goal must
    leave the ranked score BIT-IDENTICAL to the goal-free one."""
    cfg, m = _build()
    f = _frames(cfg)
    with torch.no_grad():
        m.core.decoder.gp_point_gate.fill_(5.0)
        withheld = m(f, v0=V0, steps=2, gp_point=LEFT,
                     gp_valid=torch.zeros(4))
        none_at_all = m(f, v0=V0, steps=2, gp_point=torch.zeros(4, 2),
                        gp_valid=torch.zeros(4))
        present = m(f, v0=V0, steps=2, gp_point=LEFT)
    assert torch.equal(withheld["sel_score"], none_at_all["sel_score"])
    assert float(withheld["gp_point_score_absmean"]) == 0.0
    assert not torch.equal(withheld["sel_score"], present["sel_score"])


# ---------------------------------------------------------------------------
# 5. ⛔ THE LEAK GUARD — a CONSTRUCTOR REFUSAL, proved by MUTATION
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_goal", [2.0, 1.9, 1.0, 0.5, 0.0, -1.0])
def test_a_goal_INSIDE_the_scored_horizon_cannot_be_CONSTRUCTED(t_goal):
    """A goal point inside the 2 s scored horizon IS THE ANSWER. The refusal is
    structural — no arm that leaks can be built at all."""
    with pytest.raises(ValueError, match="strictly beyond the scored horizon"):
        gp.GoalPointConfig(t_goal_s=t_goal)


def test_MUTATION_the_leak_guard_is_LOAD_BEARING_not_decorative(monkeypatch):
    """⛔⛔ THE MUTATION. Asserting that a guard raises proves the guard raises;
    it does NOT prove the guard is what stops the leak. So: REMOVE it, and show
    the leaking arm becomes constructible AND trainable end-to-end — then put
    it back and show it does not. A guard whose removal changes nothing was
    never the thing protecting us."""
    real = gp.GoalPointConfig.__post_init__

    def neutered(self):
        if self.goal_mode not in ("time", "arc"):
            raise ValueError("goal_mode")

    # --- with the guard REMOVED, a t_goal INSIDE the horizon is buildable ---
    monkeypatch.setattr(gp.GoalPointConfig, "__post_init__", neutered)
    leaking = gp.GoalPointConfig(d_goal=8, t_goal_s=2.0)
    assert leaking.t_goal_s <= leaking.t_pred_s
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.goal_point_inject = True
    cfg.goal_point_cfg = leaking
    cfg.core.graft_gp_point = True
    torch.manual_seed(0)
    m = v3.RefCV3Model(cfg).eval()
    # and it is not merely constructible — it reads the SCORED slot
    assert m.gp_slot == 3, m.gp_slot      # 2.0 s == the scored endpoint
    with torch.no_grad():
        out = m(_frames(cfg), v0=V0, steps=2)
    assert out["goal_point"] is not None

    # --- guard RESTORED: the same arm is impossible -----------------------
    monkeypatch.setattr(gp.GoalPointConfig, "__post_init__", real)
    with pytest.raises(ValueError, match="strictly beyond the scored horizon"):
        gp.GoalPointConfig(d_goal=8, t_goal_s=2.0)


def test_the_guard_is_a_REFUSAL_and_not_a_WARNING(recwarn):
    """It must raise, not warn. A warning in a 40 k-step launch log is not a
    guard — it is a sentence nobody reads."""
    with pytest.raises(ValueError):
        gp.GoalPointConfig(t_goal_s=1.0)
    assert len(recwarn) == 0


def test_a_goal_time_that_is_not_a_bank_slot_is_REFUSED_at_build():
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.goal_point_inject = True
    cfg.goal_point_cfg = gp.GoalPointConfig(d_goal=8, t_goal_s=3.7)
    with pytest.raises(ValueError, match="not a model horizon"):
        v3.RefCV3Model(cfg)


# ---------------------------------------------------------------------------
# 6. the refusals that stop a manufactured negative
# ---------------------------------------------------------------------------

def test_the_geo_prior_WITHOUT_the_head_is_REFUSED():
    with pytest.raises(ValueError, match="needs goal_point_inject"):
        _build(inject=False, geo=True)


def test_a_supplied_gp_point_into_a_SEAMLESS_build_is_REFUSED():
    """A silently-dropped intervention would read as "corrupting the goal costs
    nothing" — the PREREG §4 gate failing for a wiring reason."""
    cfg, m = _build(inject=True, geo=False)
    with pytest.raises(ValueError, match="SILENTLY"):
        m(_frames(cfg), v0=V0, steps=2, gp_point=LEFT)


def test_a_slot_outside_the_bank_is_REFUSED_at_build_not_clamped():
    sel = refc.SelectionConfig(graft_gp_point=True, gp_slot=99,
                               gp_scale_m=40.0)
    cfg = refc.refc_smoke_config()
    cfg.trajectory = refc.TrajectoryConfig(horizons=v3.V3_HORIZONS)
    with pytest.raises(ValueError, match="not a slot of this decoder"):
        refc.AnchoredDiffusionDecoder(
            32, len(v3.V3_HORIZONS), cfg.measurement.d_out,
            cfg.strategic.d_ctx, cfg.tactical_latent_dim,
            torch.zeros(4, len(v3.V3_HORIZONS), 2), cfg.decoder,
            hierarchy=False, graft_maneuver=False,
            graft_target_latent=False, grounded_selector=False, sel=sel,
            horizons=v3.V3_HORIZONS)
