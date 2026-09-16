"""refcv6 §4/§5 — the tactical behaviour decoder, proven by MUTATION.

⛔⛔ THE RULE THIS FILE IS WRITTEN UNDER: *"Guards proven by MUTATION, not
inspection."* An assertion that cannot be made to FAIL proves nothing. Every
guard below is therefore tested twice — once that it passes on the correct
object, and once that it RAISES on a deliberately broken one — and where the
break is a behaviour rather than a value, the break is applied to a live object
(monkeypatched forward, maimed stamp, deleted mask) rather than asserted about.

The mutation table in the RESULT is generated from this file's
``MUTATIONS`` registry, so the table and the tests cannot drift.
"""
from __future__ import annotations

import math

import pytest
import torch

from tanitad.models.vocab_v7 import (TACTICAL_GOAL_TOKENS_V7,
                                     TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)
from tanitad.refs import refc
from tanitad.refs import refcv6_max_speed as v6ms
from tanitad.refs import refcv6_selection as v6sel
from tanitad.refs import refcv6_tactical as v6tac

#: ⭐ The mutation registry. ``(guard, what was broken, expected)``. The RESULT's
#: table is rendered from this, so a guard added without a mutation shows up as
#: a missing row rather than as a silently untested claim.
MUTATIONS: list[tuple[str, str, str]] = []


def _m(guard: str, broke: str, expect: str) -> None:
    MUTATIONS.append((guard, broke, expect))


# ===========================================================================
# 0. mirrored constants — pinned equal to their originals
# ===========================================================================

def test_mirrored_widths_equal_their_originals():
    """⛔ The nav width is MIRRORED in refcv6_tactical (importing refc.py into
    every consumer is the cost this avoids). A mirror that drifts is worse than
    an import, so it is pinned — the ``refc_tactical`` contract."""
    assert v6tac.N_NAV_COMMANDS == len(refc.NAV_COMMANDS)
    assert v6tac.N_GOAL_TOKENS == len(TACTICAL_GOAL_TOKENS_V7) == 22
    assert v6tac.N_LAT_ACTIONS == len(TACTICAL_LAT_ACTIONS_V7) == 8
    assert v6tac.N_LON_ACTIONS == len(TACTICAL_LON_ACTIONS_V7) == 8
    assert v6tac.N_QUERIES == 38
    assert v6tac.COND_DIMS == 10


def test_loss_budget_equals_the_trainers_maneuver_weight():
    """⛔ *"inside the existing MANEUVER_WEIGHT budget (do not inflate the
    total)"*. The budget is read from the trainer's own constant, not typed."""
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "refc_train.py")
    src = open(p, encoding="utf-8").read()
    assert "MANEUVER_WEIGHT = 0.1" in src, (
        "refc_train.MANEUVER_WEIGHT moved; the refcv6 split must move with it")
    w = v6tac.TacticalLossWeights()
    assert w.budget == pytest.approx(0.1)
    assert w.total() == pytest.approx(0.1)
    assert (w.goal_bce, w.lat_ce, w.lon_ce) == (0.05, 0.025, 0.025)


def test_budget_refusal_is_reachable():
    w = v6tac.TacticalLossWeights(goal_bce=0.05, lat_ce=0.05, lon_ce=0.05)
    with pytest.raises(ValueError, match="over the MANEUVER_WEIGHT budget"):
        w.assert_within_budget()
    _m("TacticalLossWeights.assert_within_budget",
       "weights raised to 0.15 (over the 0.10 budget)", "ValueError RAISED")


# ===========================================================================
# 1. the 4-way containing-window ladder
# ===========================================================================

@pytest.mark.parametrize("kmh,expect_bin,expect_over", [
    (0.0, 0, False), (29.99, 0, False), (30.0, 0, False),
    (30.01, 1, False), (50.0, 1, False),
    (50.01, 2, False), (96.0, 2, False), (100.0, 2, False),
    (100.01, 3, False), (120.0, 3, False),
    (120.01, 3, True), (136.08, 3, True),
])
def test_containing_window_boundaries(kmh, expect_bin, expect_over):
    """⛔ THE PI'S RULE, AT ITS BOUNDARIES. (0,30]->30, (30,50]->50,
    (50,100]->100, (100,120]->120, >120 clamps. 96 km/h -> 100, NOT the
    nearest value: a set speed is a LIMITER and the realised speed must sit
    INSIDE the window."""
    i, over = v6ms.speed_max_bin(kmh / 3.6)
    assert (i, over) == (expect_bin, expect_over)
    assert v6ms.SPEED_MAX_STEPS_KMH_V6[i] >= kmh or over


def test_tensor_and_scalar_binning_agree_elementwise():
    """A second implementation is a second rule. Fuzzed, not asserted."""
    g = torch.Generator().manual_seed(0)
    v = torch.rand(4096, generator=g) * 45.0            # 0..162 km/h
    idx, over = v6ms.speed_max_bin_tensor(v)
    for k in range(0, 4096, 97):
        i0, o0 = v6ms.speed_max_bin(float(v[k]))
        assert (int(idx[k]), bool(over[k])) == (i0, o0), float(v[k]) * 3.6


def test_invalid_row_is_all_zero_not_bin_zero():
    """⛔ "no set-speed known" and "the set speed is 30 km/h" are DIFFERENT
    inputs. Collapsing them is the X15 defect."""
    idx = torch.tensor([0, 2])
    valid = torch.tensor([0.0, 1.0])
    oh = v6ms.speed_max_onehot(idx, valid)
    assert oh[0].sum() == 0.0 and oh[1].sum() == 1.0 and oh[1, 2] == 1.0


def test_nonfinite_speed_refuses():
    with pytest.raises(ValueError, match="non-finite"):
        v6ms.speed_max_bin(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        v6ms.speed_max_bin_tensor(torch.tensor([1.0, float("inf")]))
    _m("speed_max_bin", "fed NaN (would silently bin to 0 = 30 km/h)",
       "ValueError RAISED")


# ===========================================================================
# 2. the stamp guard — BOTH directions, proven by mutation
# ===========================================================================

def test_stamp_guard_passes_on_the_real_stamp():
    v6ms.assert_speed_max_stamp_v6(
        {"speed_max_derivation_v6": v6ms.SPEED_MAX_DERIVATION_V6}, on=True)
    v6ms.assert_speed_max_stamp_v6({}, on=False)
    v6ms.assert_speed_max_stamp_v6({"speed_max_derivation_v6": None}, on=False)


def test_stamp_missing_while_channel_on_REFUSES():
    with pytest.raises(v6ms.SpeedMaxStampError, match="would carry no"):
        v6ms.assert_speed_max_stamp_v6({"arm": "hier"}, on=True)
    _m("assert_speed_max_stamp_v6", "channel ON, stamp absent from config",
       "SpeedMaxStampError RAISED")


def test_stamp_present_while_channel_off_REFUSES_the_mirror():
    """⛔ THE MIRROR CASE. A control stamped as conditioned manufactures a
    max-speed-conditioned arm out of an arm that fed nothing."""
    with pytest.raises(v6ms.SpeedMaxStampError, match="MIRROR"):
        v6ms.assert_speed_max_stamp_v6(
            {"speed_max_derivation_v6": v6ms.SPEED_MAX_DERIVATION_V6},
            on=False)
    _m("assert_speed_max_stamp_v6",
       "channel OFF but config carries the stamp (the MIRROR case)",
       "SpeedMaxStampError RAISED")


@pytest.mark.parametrize("token", v6ms.SPEED_MAX_STAMP_REQUIRED_V6)
def test_a_maimed_stamp_REFUSES_per_required_token(token):
    """⛔ THE GUARD IS ON CONTENT, NOT PRESENCE — and each required token is
    individually load-bearing. A stamp that said "max speed: on" would satisfy
    a presence check and tell a reader nothing."""
    maimed = v6ms.SPEED_MAX_DERIVATION_V6.replace(token, "")
    assert maimed != v6ms.SPEED_MAX_DERIVATION_V6
    with pytest.raises(v6ms.SpeedMaxStampError, match="does not declare"):
        v6ms.assert_speed_max_stamp_v6(
            {"speed_max_derivation_v6": maimed}, on=True)


def test_maimed_stamp_mutation_row():
    _m("assert_speed_max_stamp_v6",
       f"each of the {len(v6ms.SPEED_MAX_STAMP_REQUIRED_V6)} required tokens "
       f"deleted from the stamp, one at a time",
       "SpeedMaxStampError RAISED on every one")


def test_the_eight_step_stamp_does_NOT_satisfy_the_four_way_guard():
    """⛔ TWO LADDERS MAY NOT SHARE ONE KEY. E16's continuous 8-step stamp is a
    valid stamp for a DIFFERENT channel; accepting it here would let an arm
    describe a ceiling it never fed."""
    from tanitad.refs import max_speed_input as msi           # noqa: F401
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "refc_v3_train.py")
    src = open(p, encoding="utf-8").read()
    i = src.index("SPEED_MAX_DERIVATION = (")
    old_stamp = src[i:i + 600]
    assert "v_hi_ms" in old_stamp
    # The old stamp under the NEW key must still fail: it declares no ladder.
    with pytest.raises(v6ms.SpeedMaxStampError, match="does not declare"):
        v6ms.assert_speed_max_stamp_v6(
            {"speed_max_derivation_v6":
                "v7.2 g_tac.goals.SPEED_BAND.v_hi_ms (oracle, provenance "
                "ego-future: max of the ego's OWN REALISED speed over "
                "[t0+2 s, +6 s]; floor from .v_lo_ms; allow_oracle_nav=True)."},
            on=True)
    _m("assert_speed_max_stamp_v6",
       "E16's 8-step stamp supplied under the 4-way key",
       "SpeedMaxStampError RAISED (no ladder declared)")


# ===========================================================================
# 3. the behaviour decoder
# ===========================================================================

def _cond(b: int = 3, nav: int = 1, vmax: int = 2, v0: float = 8.0,
          a0: float = 0.4) -> torch.Tensor:
    nav_1h = torch.nn.functional.one_hot(
        torch.full((b,), nav, dtype=torch.long), v6tac.N_NAV_COMMANDS).float()
    vm_1h = v6ms.speed_max_onehot(torch.full((b,), vmax, dtype=torch.long))
    return v6tac.build_condition(nav_1h, vm_1h,
                                 torch.full((b, 1), v0),
                                 torch.full((b, 1), a0))


def _dec(d_agent: int = 32, d_bev: int = 16, d: int = 64, layers: int = 2):
    cfg = v6tac.TacticalDecoderConfig(d_model=d, n_layers=layers, n_heads=4,
                                      ff_mult=2, d_agent=d_agent, d_bev=d_bev)
    return v6tac.TacticalBehaviourDecoder(cfg)


def test_decoder_shapes_and_the_query_partition():
    dec = _dec()
    b, m, p = 3, 5, 7
    out = dec(_cond(b),
              agent_tokens=torch.randn(b, m, 32),
              agent_pad=torch.zeros(b, m, dtype=torch.bool),
              bev_tokens=torch.randn(b, p, 16))
    assert out["goal_logits"].shape == (b, 22)
    assert out["goal_conf"].shape == (b, 22)
    assert out["lat_logits"].shape == (b, 8)
    assert out["lon_logits"].shape == (b, 8)
    assert out["attn"].shape == (b, 38, m + p)
    assert torch.equal(out["n_scene"], torch.full((b,), m + p))


#: ⭐ PI 2026-09-16: the input becomes 256x1024 and the trunk becomes
#: resnet101, with resnet34 as the comparison run. At stride-16 that is a
#: **16 x 64** map (was 16 x 40) and **1024** channels (resnet34: **256**).
#: Both arms are parameterised here so neither surprises us at integration.
BACKBONES = [("resnet101", 1024), ("resnet34", 256)]
#: (grid_x, grid_y) of the lifted BEV. The OLD 16x40 geometry is kept beside the
#: new one deliberately: the decoder must be indifferent to the COUNT, and a
#: test that only ever saw one count could not show that.
BEV_GRIDS = [(16, 64), (16, 40), (30, 16)]


@pytest.mark.parametrize("name,d_bev", BACKBONES)
@pytest.mark.parametrize("gx,gy", BEV_GRIDS)
def test_bev_token_COUNT_is_runtime_and_WIDTH_is_declared(name, d_bev, gx, gy):
    """⛔ The token COUNT is read off the tensor — 16x40, 16x64 and 30x16 all
    work on one build. The WIDTH is a parameter shape and follows the trunk."""
    dec = _dec(d_agent=32, d_bev=d_bev, d=64, layers=1)
    b = 2
    feats = torch.randn(b, d_bev, gx, gy)
    tok = v6tac.bev_feats_to_tokens(feats)
    assert tok.shape == (b, gx * gy, d_bev)
    out = dec(_cond(b), bev_tokens=tok)
    assert out["goal_logits"].shape == (b, 22)
    assert out["attn"].shape == (b, 38, gx * gy)
    assert torch.equal(out["n_scene"], torch.full((b,), gx * gy))


@pytest.mark.parametrize("name,d_bev", BACKBONES)
def test_a_WRONG_bev_width_REFUSES_by_name(name, d_bev):
    """⛔ A width mismatch must name both backbones, not surface as a bare
    `mat1 and mat2 shapes cannot be multiplied` a hundred frames into a pod
    run."""
    other = dict(BACKBONES)["resnet34" if name == "resnet101" else "resnet101"]
    dec = _dec(d_agent=32, d_bev=d_bev, d=64, layers=1)
    with pytest.raises(v6tac.SceneInputRefused, match="resnet101"):
        dec(_cond(2), bev_tokens=torch.randn(2, 40, other))
    _m("TacticalBehaviourDecoder.forward",
       "BEV tokens of the OTHER backbone's width (resnet34 256 vs resnet101 "
       "1024) fed to a declared build",
       "SceneInputRefused RAISED naming both widths")


def test_bev_feats_to_tokens_refuses_a_non_grid():
    with pytest.raises(v6tac.SceneInputRefused, match=r"\[B, C, X, Y\]"):
        v6tac.bev_feats_to_tokens(torch.randn(2, 64, 40))
    _m("bev_feats_to_tokens", "a [B, P, C] token sequence passed as a grid",
       "SceneInputRefused RAISED")


def test_a_bev_GRID_passed_where_TOKENS_are_expected_REFUSES():
    dec = _dec(d_agent=32, d_bev=128, d=64, layers=1)
    with pytest.raises(v6tac.SceneInputRefused, match="FLAT token sequence"):
        dec(_cond(2), bev_tokens=torch.randn(2, 128, 16, 64))
    _m("TacticalBehaviourDecoder.forward",
       "a [B, C, X, Y] BEV feature map passed where flat tokens are expected",
       "SceneInputRefused RAISED (pointing at `bev_feats_to_tokens`)")


def test_decoder_has_NO_image_token_port():
    """⛔ STRUCTURAL, not declarative. The exclusion the PI named is enforced by
    the signature — there is nowhere to pass image tokens — exactly as
    ``AgentSlotDecoder.forward`` enforces the vision-only rule."""
    import inspect
    params = set(inspect.signature(
        v6tac.TacticalBehaviourDecoder.forward).parameters)
    assert params == {"self", "cond", "agent_tokens", "agent_pad",
                      "bev_tokens", "bev_pad"}
    assert not any("image" in p or "fmap" in p or "pooled" in p
                   for p in params)


def test_scene_only_declaration_refuses_image():
    with pytest.raises(v6tac.SceneInputRefused, match="not the scene"):
        v6tac.assert_scene_only(["agent", "image"])
    _m("assert_scene_only", "'image' added to the key/value source list",
       "SceneInputRefused RAISED")


def test_a_forward_with_NO_scene_REFUSES():
    dec = _dec()
    with pytest.raises(v6tac.SceneInputRefused, match="no scene reached"):
        dec(_cond(2))
    _m("TacticalBehaviourDecoder.forward",
       "called with agent_tokens=None AND bev_tokens=None",
       "SceneInputRefused RAISED (would emit the unconditional prior)")


def test_a_silently_dropped_scene_tensor_REFUSES_both_directions():
    """⛔ The refcv5 WP-6 rule: a privileged tensor that is silently dropped
    reads as "the scene does not help" in a result table."""
    dec = _dec(d_bev=0)
    with pytest.raises(v6tac.SceneInputRefused, match="d_bev = 0"):
        dec(_cond(2), agent_tokens=torch.randn(2, 3, 32),
            bev_tokens=torch.randn(2, 4, 16))
    dec2 = _dec(d_agent=0)
    with pytest.raises(v6tac.SceneInputRefused, match="d_agent = 0"):
        dec2(_cond(2), agent_tokens=torch.randn(2, 3, 32),
             bev_tokens=torch.randn(2, 4, 16))
    _m("TacticalBehaviourDecoder.forward",
       "a scene tensor passed to a build with that port switched off",
       "SceneInputRefused RAISED (both directions)")


def test_film_is_zero_init_so_the_condition_is_bit_inert_at_step_0():
    """⭐ The ``ctx_to_cond`` discipline: a fresh decoder is EXACTLY the
    unconditioned decoder, so every later change is attributable."""
    dec = _dec().eval()
    a = torch.randn(4, 6, 32)
    with torch.no_grad():
        o1 = dec(_cond(4, nav=1, vmax=0), agent_tokens=a)["goal_logits"]
        o2 = dec(_cond(4, nav=2, vmax=3, v0=30.0, a0=-2.0),
                 agent_tokens=a)["goal_logits"]
    assert torch.allclose(o1, o2, atol=0, rtol=0)
    # MUTATION: give the FiLM a non-zero projection and the two must differ.
    for lyr in dec.layers:
        torch.nn.init.normal_(lyr.film.proj.weight, std=0.5)
    with torch.no_grad():
        o3 = dec(_cond(4, nav=2, vmax=3, v0=30.0, a0=-2.0),
                 agent_tokens=a)["goal_logits"]
    assert not torch.allclose(o1, o3, atol=1e-6)
    _m("_QueryFiLM zero-init",
       "FiLM projections re-initialised to N(0, 0.5) after the equality check",
       "the two conditions now DIFFER -> the equality test can fail")


def test_a_fully_padded_row_is_zeroed_not_NaN():
    """⛔ torch's MHA returns NaN for a row with every key masked. A NaN in the
    loss reads as divergence, not as an empty scene."""
    dec = _dec(d_bev=0).eval()
    pad = torch.zeros(2, 4, dtype=torch.bool)
    pad[1] = True                                     # row 1: nothing visible
    with torch.no_grad():
        out = dec(_cond(2), agent_tokens=torch.randn(2, 4, 32), agent_pad=pad)
    assert torch.isfinite(out["goal_logits"]).all()
    assert int(out["n_scene"][1]) == 0


def test_build_condition_refuses_a_soft_block():
    b = 2
    soft = torch.full((b, v6tac.N_NAV_COMMANDS), 0.25)
    vm = v6ms.speed_max_onehot(torch.zeros(b, dtype=torch.long))
    with pytest.raises(ValueError, match="not a one-hot"):
        v6tac.build_condition(soft, vm, torch.zeros(b, 1), torch.zeros(b, 1))
    nav = torch.nn.functional.one_hot(torch.zeros(b, dtype=torch.long),
                                      v6tac.N_NAV_COMMANDS).float()
    soft_v = torch.full((b, v6ms.N_SPEED_MAX_BINS_V6), 0.25)
    with pytest.raises(ValueError, match="not a one-hot"):
        v6tac.build_condition(nav, soft_v, torch.zeros(b, 1),
                              torch.zeros(b, 1))
    _m("build_condition",
       "a SOFT (0.25 each) distribution smuggled into the nav / max-speed slots",
       "ValueError RAISED (a richer input than the PI authorised)")


def test_planner_feeds_are_detached():
    dec = _dec()
    out = dec(_cond(2), agent_tokens=torch.randn(2, 4, 32))
    f = v6tac.planner_feeds(out)
    for k in ("lat_logprob", "lon_logprob", "valid_behaviour", "valid_conf"):
        assert not f[k].requires_grad, k
    assert out["goal_logits"].requires_grad          # the LOSS path is attached


# ===========================================================================
# 4. the losses — never guarded, never pooled
# ===========================================================================

def _targets(b: int = 4, in_band: bool = True):
    y = (torch.rand(b, 22) > 0.7).float()
    w = torch.ones(b, 22)
    lat = torch.randint(0, 8, (b,)) if in_band else torch.full((b,), -100)
    lon = torch.randint(0, 8, (b,)) if in_band else torch.full((b,), -100)
    return y, w, lat, lon


def test_every_head_receives_a_gradient_even_at_weight_zero():
    """⛔⛔ MEASURED across the programme 2026-09-06: 42 of 138 optimizer
    tensors received NO gradient — 52.2 % of a declared budget — because terms
    were weighted 0.0 AND guarded behind ``if w > 0``. A guarded term makes
    ``p.grad`` None, which is indistinguishable from a head that was never
    wired. Here the weight multiplies, so ``p.grad is None`` stays a clean
    discriminator."""
    dec = _dec()
    # ⚠️ BOTH scene ports are fed. `p.grad is None` is only a discriminator for
    # parameters ON THE TAKEN PATH — an unfed `bev_in` is correctly None, and
    # the first version of this test asserted otherwise. That is the check
    # working: a port that receives no tensor receives no gradient, and that
    # is exactly what "never wired" is supposed to look like.
    out = dec(_cond(4), agent_tokens=torch.randn(4, 5, 32),
              bev_tokens=torch.randn(4, 3, 16))
    y, w, lat, lon = _targets(4)
    zero_w = v6tac.TacticalLossWeights(goal_bce=0.0, lat_ce=0.0, lon_ce=0.0)
    loss, tele = v6tac.tactical_behaviour_losses(
        out, goal_y=y, goal_w=w, lat_target=lat, lon_target=lon,
        weights=zero_w)
    loss.backward()
    for name, p in dec.named_parameters():
        assert p.grad is not None, f"{name} received NO gradient tensor"
        assert torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().max()) == 0.0, f"{name} moved at weight 0"
    assert float(loss) == 0.0
    _m("tactical_behaviour_losses",
       "all three weights set to 0.0 (the guarded-term failure mode)",
       "p.grad is a ZEROS tensor on every parameter, never None")


def test_an_all_out_of_band_batch_gives_a_finite_zero_not_NaN():
    """⛔ ``F.cross_entropy(..., reduction='mean')`` returns NaN when every
    element is ignored. A batch entirely outside the ±2 s band is the NORMAL
    path here (1,157 of 4,823 eval windows are in band)."""
    dec = _dec()
    out = dec(_cond(4), agent_tokens=torch.randn(4, 5, 32))
    y, w, lat, lon = _targets(4, in_band=False)
    loss, tele = v6tac.tactical_behaviour_losses(
        out, goal_y=y, goal_w=w, lat_target=lat, lon_target=lon)
    assert torch.isfinite(loss)
    assert tele["n_supervised_lat"] == 0 and tele["n_supervised_lon"] == 0
    assert float(tele["tac_lat_ce"]) == 0.0
    # MUTATION: torch's own mean-reduction on the same input is NaN.
    naive = torch.nn.functional.cross_entropy(
        out["lat_logits"], lat.clamp_min(0) * 0 - 100, ignore_index=-100)
    assert math.isnan(float(naive))
    _m("tactical_behaviour_losses (CE)",
       "a batch entirely OUTSIDE the ±2 s band; torch's mean-reduction on the "
       "same input is NaN",
       "finite 0.0 with n_supervised = 0 reported beside it")


def test_per_class_report_is_per_class_with_its_control():
    dec = _dec()
    out = dec(_cond(8), agent_tokens=torch.randn(8, 5, 32))
    y, w, lat, lon = _targets(8)
    rep = v6tac.per_class_report(out, goal_y=y, goal_w=w, lat_target=lat,
                                 lon_target=lon)
    assert set(rep["goal"]) == set(TACTICAL_GOAL_TOKENS_V7)
    # ⭐ THE CONTROL MUST READ ITS KNOWN VALUE: recall exactly 0.0 where the
    # majority is ABSENT and exactly 1.0 where it is PRESENT. A panel whose
    # control does not read those has a scoring bug, not a result.
    for tok, d in rep["goal_majority_control"].items():
        if d["recall"] is None:
            continue
        assert d["recall"] in (0.0, 1.0), (tok, d)
        assert d["recall"] == (1.0 if d["majority"] == "present" else 0.0)
    assert set(rep["lat"]) >= set(TACTICAL_LAT_ACTIONS_V7)
    assert "recall" not in rep            # ⛔ no pooled number anywhere
    assert "accuracy" not in rep


# ===========================================================================
# 5. admissibility — the PI's 2026-08-03 ruling, proven by mutation
# ===========================================================================

def _roles(goal_inputs=(), sel_inputs=(), sit=(), goal_nodes=("g_tac",)):
    return {"goal": list(goal_nodes),
            "situation_output": list(sit),
            "inference_inputs_of_goals": list(goal_inputs),
            "selection_inputs": list(sel_inputs)}


def test_situation_tokens_as_TARGETS_are_admissible():
    v6tac.assert_situation_tokens_are_targets_only(
        _roles(goal_inputs=["frames (via pooled_seq/ctx)"],
               sel_inputs=["anchor confidence"],
               sit=sorted(v6tac.SITUATION_OUTPUT_TOKENS)))


def test_a_situation_token_as_a_GOAL_INPUT_REFUSES():
    with pytest.raises(v6tac.SituationInputRefused, match="goal inputs"):
        v6tac.assert_situation_tokens_are_targets_only(
            _roles(goal_inputs=["frames", "tac_SIT one-hot"],
                   sel_inputs=["anchor confidence"]))
    _m("assert_situation_tokens_are_targets_only",
       "'tac_SIT one-hot' added to inference_inputs_of_goals",
       "SituationInputRefused RAISED")


def test_a_traffic_light_token_as_a_SELECTION_INPUT_REFUSES():
    with pytest.raises(v6tac.SituationInputRefused,
                       match="selection inputs"):
        v6tac.assert_situation_tokens_are_targets_only(
            _roles(goal_inputs=["frames"],
                   sel_inputs=["anchor confidence",
                               "TRAFFIC_LIGHT_REACT_RED probability"]))
    _m("assert_situation_tokens_are_targets_only",
       "'TRAFFIC_LIGHT_REACT_RED probability' added to selection_inputs",
       "SituationInputRefused RAISED")


def test_a_MISSING_selection_inputs_key_REFUSES_rather_than_passing():
    """⛔ A guard that silently accepts an absent key is a guard proven by
    INSPECTION: it reads green on a model that never declared its selection
    inputs at all."""
    roles = _roles(goal_inputs=["frames"], sel_inputs=["x"])
    roles.pop("selection_inputs")
    with pytest.raises(v6tac.SituationInputRefused, match="no `selection_inputs`"):
        v6tac.assert_situation_tokens_are_targets_only(roles)
    _m("assert_situation_tokens_are_targets_only",
       "the `selection_inputs` key deleted from the declaration",
       "SituationInputRefused RAISED (an absent key is not a pass)")


# ===========================================================================
# 6. the behaviour -> selection gate: the columns are DEAD, proven by mutation
# ===========================================================================

def test_the_admissibility_partition_is_17_plus_5():
    assert len(v6sel.SELECTION_REFUSED_TOKENS) == 5
    assert len(v6sel.SELECTION_ADMISSIBLE_TOKENS) == 17
    assert set(v6sel.SELECTION_REFUSED_TOKENS) == {
        "YIELD", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
        "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN"}


def test_gate_is_zero_init_so_the_ranked_score_is_unchanged_at_step_0():
    g = v6sel.BehaviourSelectionGate(37)
    out = g(torch.rand(4, 22))
    assert torch.equal(out, torch.zeros(4, 37))


def test_situation_columns_are_dead_and_the_proof_is_a_MUTATION():
    g = v6sel.BehaviourSelectionGate(37)
    rep = v6sel.assert_situation_columns_dead(g)
    assert rep["delta_refused"] == 0.0
    assert rep["delta_admissible"] > 0.0
    assert len(rep["refused_tokens"]) == 5


def test_deleting_the_mask_MAKES_THE_GUARD_FAIL():
    """⛔⛔ THE GUARD MUST BE ABLE TO FAIL. A build whose forward stops applying
    the admissibility mask must be REFUSED — this is the mutation that proves
    the check is a probe and not an inspection."""
    g = v6sel.BehaviourSelectionGate(37)

    def unmasked_forward(v):                    # the defect, reintroduced
        return g.proj(v.to(g.proj.weight.dtype))

    g.forward = unmasked_forward                # type: ignore[method-assign]
    with pytest.raises(v6sel.SituationInputRefused, match="VIOLATED"):
        v6sel.assert_situation_columns_dead(g)
    _m("assert_situation_columns_dead",
       "BehaviourSelectionGate.forward replaced by one that SKIPS the "
       "admissibility mask",
       "SituationInputRefused RAISED ('PI 2026-08-03 VIOLATED')")


def test_a_DEAD_gate_fails_the_POSITIVE_CONTROL():
    """⛔ "nothing moved" must not be allowed to pass because nothing is wired.
    A gate whose forward returns zeros regardless must be refused too."""
    g = v6sel.BehaviourSelectionGate(37)
    g.forward = lambda v: torch.zeros(v.shape[0], 37)   # type: ignore
    with pytest.raises(v6sel.SituationInputRefused,
                       match="positive control FAILED"):
        v6sel.assert_situation_columns_dead(g)
    _m("assert_situation_columns_dead",
       "the gate's forward replaced by a constant-zero function",
       "SituationInputRefused RAISED (the positive control catches a dead "
       "layer passing the refused-column check vacuously)")


# ===========================================================================
# 7. the speed ceiling as an ARGMAX FILTER, and the derived dt
# ===========================================================================

def test_step_durations_are_DERIVED_from_the_horizons():
    """⛔ horizons (5, 10, 15, 20) TICKS at 0.1 s are **0.5 s apart**. A typed
    0.1 would report every planned speed 5x too high and fail an obedient
    model. Same derivation ``refc.py`` makes for its prefix dt."""
    assert v6sel.step_durations_s((5, 10, 15, 20), 0.1) == pytest.approx(
        (0.5, 0.5, 0.5, 0.5))
    assert v6sel.step_durations_s((2, 5, 9), 0.1) == pytest.approx(
        (0.2, 0.3, 0.4))
    with pytest.raises(ValueError, match="strictly increasing"):
        v6sel.step_durations_s((10, 5))


def test_planned_max_speed_reads_the_true_speed_not_a_5x_one():
    """A straight plan at exactly 10 m/s over (5,10,15,20) ticks."""
    h = (5, 10, 15, 20)
    xs = torch.tensor([k * 0.1 * 10.0 for k in h])          # 5, 10, 15, 20 m
    cand = torch.stack([xs, torch.zeros_like(xs)], dim=-1).view(1, 1, 4, 2)
    v = v6sel.planned_max_speed(cand, horizons=h, tick_s=0.1)
    assert float(v) == pytest.approx(10.0, abs=1e-5)
    # MUTATION: the naive per-tick divisor is 5x wrong, and that is the bug
    # this derivation exists to prevent.
    naive = float(v6sel.planned_max_speed(
        cand, horizons=(1, 2, 3, 4), tick_s=0.1))
    assert naive == pytest.approx(50.0, abs=1e-4)
    _m("planned_max_speed",
       "the waypoint grid mis-declared as 1 tick apart instead of 5",
       "the reported speed is 5x (50.0 vs 10.0) — why the dt is DERIVED")


def test_speed_filter_keeps_the_compliant_and_counts_the_empty_rows():
    h = (5, 10, 15, 20)

    def straight(v_ms):
        xs = torch.tensor([k * 0.1 * v_ms for k in h])
        return torch.stack([xs, torch.zeros_like(xs)], dim=-1)

    # ⚠️ 30 km/h is 8.333 m/s, so 8.0 complies and 9.0 does NOT — the first
    # version of this test used 9.0 and expected it to pass, which is the
    # km/h-vs-m/s slip this module's unit discipline exists to catch.
    fan = torch.stack([straight(5.0), straight(8.0), straight(20.0)])  # [3,4,2]
    cand = fan.unsqueeze(0).repeat(2, 1, 1, 1)                  # [2, 3, 4, 2]
    f = v6sel.SpeedCeilingFilter(horizons=h)
    lim = torch.tensor([30.0 / 3.6, 1.0])       # row 1: nothing complies
    keep, tele = f(cand, lim)
    assert keep[0].tolist() == [True, True, False]
    # ⛔ a row with NO survivor keeps its WHOLE fan and is COUNTED
    assert keep[1].all()
    assert tele["speed_rows_empty"] == 1
    _m("SpeedCeilingFilter",
       "a row whose every candidate exceeds the ceiling",
       "the whole fan is kept and `speed_rows_empty` counts it — the only "
       "structural reason obedience can fall below 1.0")


def test_nav_compliance_is_identically_zero_where_the_command_has_no_side():
    """⛔ ``follow``/``straight`` carry no commanded side, so the predicate is
    UNDEFINED, not "failed" — nav is constant on ~75-79 % of windows."""
    left = torch.tensor([[[[1.0, 0.0], [2.0, 1.0]]]])          # turns left
    nav_follow = torch.tensor([refc.NAV_COMMANDS.index("follow")])
    nav_left = torch.tensor([refc.NAV_COMMANDS.index("left")])
    ok_f, m_f = v6sel.nav_compliance_prior(left, nav_follow, tau_rad=0.05)
    ok_l, m_l = v6sel.nav_compliance_prior(left, nav_left, tau_rad=0.05)
    assert float(ok_f.sum()) == 0.0 and not bool(m_f.any())
    assert float(ok_l.sum()) == 1.0 and bool(m_l.all())


# ===========================================================================
# 8. the model: parity when off, and the build-time refusals
# ===========================================================================

def _v3cfg(**kw):
    from tanitad.refs import refc_v3 as v3
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.tac_vocab_version = "v7.0"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_v3_parity_when_off():
    """⛔ With the refcv6 flags off the state_dict is UNCHANGED — a live run
    resumes through this file."""
    from tanitad.refs import refc_v3 as v3
    m = v3.RefCV3Model(_v3cfg())
    assert m.tac_decoder_v6 is None and m.max_speed_1h_v6 is None
    assert not any("tac_decoder_v6" in k or "tac_behaviour_gate_v6" in k
                   or "max_speed_1h_v6" in k for k in m.state_dict())
    br = v3.param_breakdown_v3(m)
    assert "tac_decoder_v6" not in br


def test_flat_build_REFUSES_the_refcv6_flags():
    from tanitad.refs import refc_v3 as v3
    cfg = v3.refc_v3_smoke_config(hier=False)
    cfg.tac_decoder_v6 = True
    with pytest.raises(ValueError, match="FLAT"):
        v3.RefCV3Model(cfg)
    _m("RefCV3Model.__init__",
       "tac_decoder_v6 on a FLAT (hier=False) build",
       "ValueError RAISED (it would build and never be called)")


def test_kin3_vocabulary_REFUSES_the_behaviour_decoder():
    from tanitad.refs import refc_v3 as v3
    cfg = _v3cfg(tac_decoder_v6=True)
    cfg.tac_vocab_version = "kin3"
    with pytest.raises(ValueError, match="v7 tactical vocabulary"):
        v3.RefCV3Model(cfg)
    _m("RefCV3Model.__init__", "tac_decoder_v6 under the kin3 vocabulary",
       "ValueError RAISED (22 logits that could never be supervised)")


def test_both_goal_heads_at_once_REFUSE():
    from tanitad.refs import refc_v3 as v3
    cfg = _v3cfg(tac_decoder_v6=True, tac_goal_tok_head=True)
    with pytest.raises(ValueError, match="Two heads emitting the same 22"):
        v3.RefCV3Model(cfg)
    _m("RefCV3Model.__init__",
       "tac_goal_tok_head AND tac_decoder_v6 both on",
       "ValueError RAISED (a confound, not an ablation)")


def test_both_max_speed_channels_at_once_REFUSE():
    from tanitad.refs import refc_v3 as v3
    cfg = _v3cfg(max_speed_input=True, max_speed_onehot_v6=True)
    with pytest.raises(ValueError, match="BOTH max-speed channels"):
        v3.RefCV3Model(cfg)
    _m("RefCV3Model.__init__",
       "E16's continuous 8-step channel AND refcv6's 4-way one-hot both on",
       "ValueError RAISED (two ceilings, two stamps, one condition)")


def test_no_scene_build_REFUSES_the_behaviour_decoder():
    from tanitad.refs import refc_v3 as v3
    cfg = _v3cfg(tac_decoder_v6=True)
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(d_model=64, n_layers=1,
                                                      n_heads=4, d_agent=64,
                                                      d_bev=0)
    with pytest.raises(ValueError, match="NEITHER agent slots"):
        v3.RefCV3Model(cfg)
    _m("RefCV3Model.__init__",
       "tac_decoder_v6 with agents OFF and d_bev = 0",
       "ValueError RAISED (it would attend to nothing)")


def test_provenance_roles_declares_the_refcv6_selection_inputs():
    from tanitad.refs import refc_v3 as v3
    from tanitad.refs.refc_agents import AgentSeamConfig
    cfg = _v3cfg(tac_decoder_v6=True)
    cfg.core.agents = AgentSeamConfig(enable=True)
    # ⚠️ AN INTEGRATION COUPLING, NAMED RATHER THAN WORKED AROUND. `refc.py`
    # REFUSES `agents.enable` without `decoder.cross_agent` (refcv5 WP-6: a
    # detector trained into a dead end). So giving the TACTICAL layer agent
    # slots forces agent cross-attention on the OPERATIVE decoder too — a
    # SECOND variable. A refcv6 tactical arm is therefore only one-variable
    # against a baseline that ALREADY has the agent seam on. Flagged in the
    # RESULT as an integration item; not silently relaxed, because the WP-6
    # guard belongs to another owner and is right.
    cfg.core.decoder.cross_agent = True
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(
        d_model=64, n_layers=1, n_heads=4, d_bev=16)
    m = v3.RefCV3Model(cfg)
    roles = m.provenance_roles()
    assert "selection_inputs" in roles
    assert sorted(roles["situation_output"]) == sorted(
        v6tac.SITUATION_OUTPUT_TOKENS)
    assert any("structurally dead" in s for s in roles["refused_edges"])
    # the build-time mutation proof travelled with the model
    assert m.tac_gate_mutation_proof_v6["delta_refused"] == 0.0
    assert m.tac_gate_mutation_proof_v6["delta_admissible"] > 0.0
    br = v3.param_breakdown_v3(m)
    assert br["tac_decoder_v6"] > 0 and br["tac_behaviour_gate_v6"] > 0


# ===========================================================================
# 8b. END TO END — only with the refc.py integration patch applied
# ===========================================================================

def _patch_applied() -> bool:
    """⛔ The refc.py wiring ships as a PATCH (another agent owns that file), so
    the end-to-end test SKIPS with a reason rather than failing on an unpatched
    tree. ⚠️ A skip is not a pass: the RESULT states which runs were made with
    the patch applied."""
    import inspect
    return "scene_hook" in inspect.signature(refc.RefCModel.forward).parameters


needs_patch = pytest.mark.skipif(
    not _patch_applied(),
    reason="refc.py integration patch not applied (see the RESULT's "
           "'apply here' note); the refcv6 seams are unreachable without it")


def _e2e_model():
    from tanitad.refs import refc_v3 as v3
    from tanitad.refs.refc_agents import AgentSeamConfig
    cfg = _v3cfg(tac_decoder_v6=True, max_speed_onehot_v6=True,
                 ego_state_inject=True)
    cfg.core.agents = AgentSeamConfig(enable=True)
    cfg.core.decoder.cross_agent = True
    cfg.core.ego_valid_channel = True     # a PRECONDITION of ego_state_inject
    cfg.core.graft_tac8_prior = True
    cfg.core.graft_behaviour_sel = True
    cfg.core.graft_nav_compliance = True
    cfg.core.nav_compliance_tau_rad = 0.08
    cfg.core.speed_ceiling_filter = True
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(
        d_model=64, n_layers=2, n_heads=4, ff_mult=2, d_bev=0)
    return v3.RefCV3Model(cfg), cfg


@needs_patch
def test_e2e_forward_reaches_every_refcv6_seam():
    import torch as t
    m, cfg = _e2e_model()
    m.eval()
    b, w = 2, cfg.core.window
    frames = t.randn(b, w, 1, 64, 64)   # the smoke encoder is 1-ch, 64 px
    ego = t.tensor([[8.0, 0.3, 0.0, 0.0, 1.0],
                    [22.0, -1.0, 0.0, 0.0, 1.0]])
    nav = t.tensor([refc.NAV_COMMANDS.index("left"),
                    refc.NAV_COMMANDS.index("follow")])
    with t.no_grad():
        out = m(frames, nav_cmd=nav, v0=ego[:, 0], ego_state=ego,
                v_max_ms=t.tensor([8.4, 33.0]),
                v_max_valid=t.tensor([1.0, 1.0]))
    # the tactical decoder ran, on the SCENE
    assert out["tacv6_injected"] is True
    assert out["tacv6_goal_logits"].shape == (b, 22)
    assert out["tacv6_lat_logits"].shape == (b, 8)
    assert int(out["tacv6_n_scene"].min()) >= 0
    # ⚠️ the decoder's telemetry lands in `out["sel_tele"]`, not at the top
    # level — the first version of this test looked at the top level and read a
    # KeyError as "the seam did not run". Same class of mistake as reading an
    # absence of data as a result.
    st = out["sel_tele"]
    # the nav-compliance term is on the forward path, with its gate at 0
    assert st["navc_gate"] == 0.0
    assert 0.0 <= st["navc_frac_complying"] <= 1.0
    # the speed filter ran and reported its rows
    assert "speed_frac_candidates_clipped" in st
    assert "speed_rows_empty" in st
    assert t.isfinite(out["traj"]).all()


@needs_patch
def test_e2e_the_image_only_lat3_prior_is_REPLACED_not_added():
    """⛔ 'REPLACES', per the spec. Feeding both puts two priors over the same
    axis on one surface and no ablation can separate them afterwards."""
    import torch as t
    m, _ = _e2e_model()
    dec = m.core.decoder
    assert dec.tac8_lat_to_anchor is not None
    assert float(dec.tac8_lat_to_anchor.weight.abs().max()) == 0.0   # zero-init
    assert float(dec.tac8_lon_to_anchor.weight.abs().max()) == 0.0
    n = dec.tac8_lat_to_anchor.weight.shape[0]
    feat = m.core.encoder.feat_dim
    fmap = t.randn(2, feat, 2, 2)
    meas = t.randn(2, m.cfg.core.measurement.d_out)
    with pytest.raises(ValueError, match="REPLACES"):
        dec(fmap, meas, lat_prior=t.log_softmax(t.randn(2, 3), -1),
            lon_prior=t.log_softmax(t.randn(2, 3), -1),
            tac_lat_prior=t.log_softmax(t.randn(2, 8), -1),
            tac_lon_prior=t.log_softmax(t.randn(2, 8), -1))
    _m("AnchoredDiffusionDecoder.forward",
       "BOTH the image-only lat3 prior and the tactical 8-wide posterior "
       "passed to one decode",
       "ValueError RAISED ('the 8-wide pair REPLACES the 3-wide one')")


@needs_patch
def test_e2e_a_scene_hook_with_no_scene_REFUSES():
    import torch as t
    from tanitad.refs import refc_v3 as v3
    cfg = _v3cfg(tac_decoder_v6=True)
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(
        d_model=64, n_layers=1, n_heads=4, d_agent=0, d_bev=16)
    m = v3.RefCV3Model(cfg)          # d_bev > 0, so the build is admissible
    b, w = 1, cfg.core.window
    with pytest.raises(ValueError, match="NEITHER agent tokens NOR BEV"):
        m(t.randn(b, w, 1, 64, 64), nav_cmd=t.zeros(b, dtype=t.long),
          v0=t.zeros(b))
    _m("RefCModel.forward (patched)",
       "a scene_hook supplied while the forward built neither agent nor BEV "
       "tokens",
       "ValueError RAISED (the decoder would attend to nothing)")


@needs_patch
def test_e2e_nav_compliance_tau_has_no_usable_default():
    from tanitad.refs import refc_v3 as v3
    from tanitad.refs.refc_agents import AgentSeamConfig
    cfg = _v3cfg(tac_decoder_v6=True)
    cfg.core.agents = AgentSeamConfig(enable=True)
    cfg.core.decoder.cross_agent = True
    cfg.core.graft_nav_compliance = True          # tau left at 0.0
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(
        d_model=64, n_layers=1, n_heads=4, d_bev=0)
    with pytest.raises(ValueError, match="nav_compliance_tau_rad"):
        v3.RefCV3Model(cfg)
    _m("AnchoredDiffusionDecoder.__init__ (patched)",
       "graft_nav_compliance on with nav_compliance_tau_rad left at 0.0",
       "ValueError RAISED (the tolerance is DERIVED per corpus and must be "
       "stated)")


# ===========================================================================
# 9. the acceptance instruments
# ===========================================================================

def _acc():
    import importlib
    return importlib.import_module("taniteval.refcv6_acceptance")


def test_tflip_absent_block_is_NOT_RUN_not_FAIL():
    acc = _acc()
    v = acc.tflip_verdict({"powered": True})
    assert v["verdict"] == "NOT_RUN"
    assert "never been run" in v["reason"].lower() or "NEVER" in v["reason"]


def test_tflip_uses_the_CI_LOWER_BOUND_not_the_mean():
    """⛔ n = 39 windows on refcv5-v2's panel. A PASS read off the point
    estimate at that n is the winner's-curse read."""
    acc = _acc()
    arm = {
        "powered": True,
        "flipped": {
            "follows_FED_command": {"mean": 0.62, "lo": 0.31, "hi": 0.85,
                                    "n_windows": 120, "n_episodes": 20},
            "follows_TRUE_command": {"mean": 0.10, "lo": 0.02, "hi": 0.22},
        },
        "paired_true_minus_shuffled": {"delta": 0.42, "lo": 0.40, "hi": 0.48,
                                       "separated": True, "n_windows": 120},
    }
    v = acc.tflip_verdict(arm)
    assert v["verdict"] == "FAIL"            # mean 0.62 clears, lo 0.31 does not
    arm["flipped"]["follows_FED_command"]["lo"] = 0.55
    assert acc.tflip_verdict(arm)["verdict"] == "PASS"
    _m("tflip_verdict",
       "a block whose MEAN (0.62) clears the 0.50 bar but whose CI lower "
       "bound (0.31) does not",
       "FAIL — the verdict reads the lower bound")


def test_tflip_unpowered_at_refcv5_v2s_own_n():
    acc = _acc()
    arm = {"powered": True,
           "flipped": {"follows_FED_command": {"mean": 0.9, "lo": 0.8,
                                               "n_windows": 39,
                                               "n_episodes": 4}},
           "paired_true_minus_shuffled": {"delta": 0.5, "lo": 0.45,
                                          "separated": True}}
    v = acc.tflip_verdict(arm)
    assert v["verdict"] == "UNPOWERED" and "39" in v["reason"]


def test_tzero_excludes_the_turn_classes_from_the_summary():
    """⛔ PI 2026-09-16: deriving the turn command from nav is BY DESIGN. The
    turn classes are reported but must not drive the diagnostic."""
    acc = _acc()
    true = {"TURN_L": {"n": 100, "recall": 0.9},
            "CRUISE": {"n": 100, "recall": 0.8},
            "BRAKE_TO": {"n": 100, "recall": 0.6},
            "MERGE": {"n": 5, "recall": 1.0}}
    zero = {"TURN_L": {"recall": 0.0},          # a permitted collapse
            "CRUISE": {"recall": 0.8},
            "BRAKE_TO": {"recall": 0.3},
            "MERGE": {"recall": 0.0}}
    r = acc.tzero_nonturn_report(true, zero)
    assert r["classes"]["TURN_L"]["is_turn_class"]
    assert "BY DESIGN" in r["classes"]["TURN_L"]["_note"]
    assert r["summary"]["n_classes_summarised"] == 2      # TURN_L and MERGE out
    assert r["summary"]["median_retained_fraction"] == pytest.approx(0.75)
    assert not r["classes"]["MERGE"]["powered"]


def test_obedience_reports_the_ADE_cost_or_refuses_to_stand_alone():
    acc = _acc()
    n = 200
    eid = [f"ep{i % 20}" for i in range(n)]
    gt = torch.full((n,), 60.0 / 3.6)                 # every window > 40 km/h
    planned = torch.full((n,), 29.0 / 3.6)
    planned[:1] = 45.0 / 3.6                          # one violator
    r = acc.speed_obedience_report(planned, gt, eid, rows_empty=1)
    assert r["n_windows"] == n and r["n_episodes"] == 20
    assert r["violation_ms"]["n_violating"] == 1
    assert r["ade_cost"]["status"] == "UNAVAILABLE"
    assert "report both or report neither" in r["ade_cost"]["reason"]
    assert r["rows_with_no_compliant_candidate"] == 1


def test_acceptance_panel_never_silently_drops_a_missing_instrument():
    acc = _acc()
    p = acc.acceptance_panel(tflip={"verdict": "PASS"},
                             obedience={"verdict": "PASS"})
    assert p["overall"] == "INCOMPLETE" and p["missing"] == ["T-ZERO"]
    p2 = acc.acceptance_panel(tflip={"verdict": "PASS"}, tzero={"x": 1},
                              obedience={"verdict": "PASS"})
    assert p2["overall"] == "PASS"
    _m("acceptance_panel", "one of the three instruments omitted",
       "overall = INCOMPLETE (a 2-of-3 panel must not read as a pass)")


# ===========================================================================
# 10. the mutation table the RESULT renders
# ===========================================================================

def test_zz_mutation_table(capsys):
    """Printed with ``-s``; the RESULT's table is this list."""
    rows = sorted(set(MUTATIONS))
    assert len(rows) >= 15, f"only {len(rows)} mutations registered"
    with capsys.disabled():
        print("\n\n| # | guard | mutation applied | result |")
        print("|---|---|---|---|")
        for i, (g, b, e) in enumerate(rows, 1):
            print(f"| {i} | `{g}` | {b} | {e} |")
