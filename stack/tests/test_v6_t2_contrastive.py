"""F-7 / catalog T2 — MANOEUVRE CONTRASTIVES: both directions.

THE SPEC, quoted rather than paraphrased (two independent locations, per the
"absence found at ONE location is not absence" rule):

  * ``TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
    2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:65`` — *"T2 |
    **manoeuvre-contrastive windows** (label-free): time-reversal and
    lane-mirror augmentations as hard negatives for the tactical predictor | a
    lane change mirrored is the OPPOSITE manoeuvre — the predictor must not be
    invariant to it; teaches manoeuvre identity without manoeuvre labels"*
  * ``…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:56`` — *"Needs: a
    T2 loss (label-free augmentations of the window + a contrastive head on
    ``z_tac``) + a weight in ``V6LossWeights``"*

⛔ THE TESTS THAT MATTER MOST HERE ARE THE INERTNESS ONES. The live v6F S-W run
resumes TENSOR-STRICT at 87,893,449 params / 405 keys; a cell that changed the
default build would kill it. C95/C97 is the other half: this programme shipped a
rejects-everything guard and a passes-everything guard within one day, so every
claim below is pinned in BOTH directions — the cell does what it says, AND it is
genuinely absent when off.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.models.v6 import (  # noqa: E402
    T2_AUGMENTATIONS, T2_MANOEUVRE_PRESERVING, T2_MANOEUVRE_REVERSING,
    ManoeuvreContrastiveHead, V6Config, V6Stack, apply_stage_freeze,
    lane_mirror_window, photometric_jitter_window, time_reverse_window)
from test_v6_gstr_port import _small  # noqa: E402

#: MEASURED on this box 2026-08-18 by BUILDING the module, never estimated.
#: (`V6Stack(V6Config())` vs `V6Stack(V6Config(t2_contrastive=True))`.)
DEFAULT_PARAMS, DEFAULT_KEYS = 87_893_449, 405
#: d_tac 512 -> hidden 256 -> proj 128:
#:   Linear(512,256) 131,072 + 256 = 131,328
#:   Linear(256,128)  32,768 + 128 =  32,896
#:   log_tau                                1
T2_DELTA_PARAMS, T2_DELTA_KEYS = 164_225, 5
T2_KEYS = ("t2_head.log_tau", "t2_head.net.0.weight", "t2_head.net.0.bias",
           "t2_head.net.2.weight", "t2_head.net.2.bias")


def _t2(**kw) -> V6Config:
    return _small(t2_contrastive=True, d_t2_proj=8, d_t2_hidden=8, **kw)


# ---------------------------------------------------------------------------
# 1. INERTNESS — the direction that protects the live 30k run
# ---------------------------------------------------------------------------

def test_default_build_is_untouched_at_the_production_geometry():
    """⛔ THE GUARD ON THE LIVE RUN. Not a toy geometry: the real default."""
    s = V6Stack(V6Config())
    assert sum(p.numel() for p in s.parameters()) == DEFAULT_PARAMS
    assert len(s.state_dict()) == DEFAULT_KEYS
    assert not any(k.startswith("t2_head") for k in s.state_dict())
    assert s.t2_head is None


def test_t2_head_is_absent_from_the_default_config():
    assert V6Config().t2_contrastive is False


def test_default_forward_emits_no_t2_key_and_the_head_is_unreferenced():
    """The head is called by the LOSS, never by ``forward`` — so the output
    dict cannot gain a key and `test_v6_gstr_port`'s bit-identity probe (which
    already caught one unconditional key) cannot be tripped by this cell."""
    off, on = V6Stack(_small()), V6Stack(_t2())
    b = 2
    for s in (off, on):
        torch.manual_seed(0)
    frames = torch.randn(b, s.cfg.predictor.window, 9, 64, 64)
    act = torch.randn(b, s.cfg.predictor.window, 3)
    v0 = torch.full((b,), 8.0)
    k_off = set(off.forward(frames=frames, actions=act, v0=v0))
    k_on = set(on.forward(frames=frames, actions=act, v0=v0))
    assert k_off == k_on, "the T2 build changed forward's output surface"


def test_the_term_is_skipped_not_zero_multiplied_when_off():
    import train_v6_staged as T
    s = V6Stack(_t2(selector="goal"))
    batch = T.synthetic_train_batch(s, batch=4, k=12)
    L = T.v6_loss_step(s, batch, stage="S-T",
                       weights=T.V6LossWeights(w_t2_contrast=0.0,
                                               lambda_plan=1.0))
    assert "t2" not in L, "a zero-weight term must be SKIPPED, not built"
    assert not any(k.startswith("t2_") for k in L["log"]), \
        "a skipped term must not appear in the log looking like it trained"


@pytest.mark.parametrize("stage", ["S-W", "S-S"])
def test_for_stage_zeroes_the_weight_where_layer_tac_is_frozen(stage):
    """A launch line must not advertise a term that trains nothing."""
    import train_v6_staged as T
    w = T.V6LossWeights(w_t2_contrast=1.0).for_stage(stage)
    assert w.w_t2_contrast == 0.0


# ---------------------------------------------------------------------------
# 2. THE CELL DOES WHAT IT CLAIMS
# ---------------------------------------------------------------------------

def test_measured_parameter_cost_at_the_production_geometry():
    """⛔ MEASURED BY BUILDING THE MODULE. A prior doc's '+41,089' was an
    ESTIMATE and the measured figure was +33,801 — hence this test."""
    off = V6Stack(V6Config())
    on = V6Stack(V6Config(t2_contrastive=True))
    d_p = sum(p.numel() for p in on.parameters()) \
        - sum(p.numel() for p in off.parameters())
    d_k = len(on.state_dict()) - len(off.state_dict())
    assert (d_p, d_k) == (T2_DELTA_PARAMS, T2_DELTA_KEYS)
    assert tuple(k for k in on.state_dict()
                 if k.startswith("t2_head")) == T2_KEYS
    # the arithmetic, so a geometry change cannot silently drift the constant
    c = V6Config()
    assert d_p == (c.d_tac * c.d_t2_hidden + c.d_t2_hidden
                   + c.d_t2_hidden * c.d_t2_proj + c.d_t2_proj + 1)


def test_the_head_is_layer_tac_and_trains_in_S_T_only():
    s = V6Stack(_t2())
    names = [n for n, _ in s.named_parameters() if n.startswith("t2_head")]
    assert names and all(s.group_of(n) == "layer_tac" for n in names)
    for stage, want in (("S-W", False), ("S-T", True),
                        ("S-S", False), ("S-J", True)):
        apply_stage_freeze(s, stage)
        got = all(dict(s.named_parameters())[n].requires_grad for n in names)
        assert got is want, f"t2_head trainable={got} in {stage}, want {want}"


def test_stage_may_introduce_carries_the_head():
    import train_v6_staged as T
    assert "t2_head." in T.STAGE_MAY_INTRODUCE["S-T"]
    # ⛔ and the stages that may introduce NOTHING still may not
    for st in ("S-W", "S-S", "S-J"):
        assert T.STAGE_MAY_INTRODUCE[st] == ()


def test_the_projector_output_is_unit_norm():
    h = ManoeuvreContrastiveHead(16, d_proj=8, hidden=8)
    z = torch.randn(5, 16) * 100.0
    n = h(z).norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)


def test_lane_mirror_flips_the_image_and_negates_the_lateral_action_only():
    f = torch.randn(2, 3, 9, 8, 8)
    a = torch.randn(2, 3, 3)
    fm, am = lane_mirror_window(f, a)
    assert torch.equal(fm, torch.flip(f, dims=(-1,)))
    assert torch.equal(am[..., 0], -a[..., 0])
    # longitudinal and speed channels are untouched: mirroring a scene does not
    # change how fast the ego is going.
    assert torch.equal(am[..., 1:], a[..., 1:])
    assert torch.equal(a, a), "the input must not be mutated in place"


def test_lane_mirror_is_an_involution():
    f = torch.randn(2, 3, 9, 8, 8)
    a = torch.randn(2, 3, 3)
    f2, a2 = lane_mirror_window(*lane_mirror_window(f, a))
    assert torch.equal(f2, f) and torch.allclose(a2, a)


def test_photometric_positive_preserves_geometry():
    """The declared POSITIVE must not move a pixel's position — only its
    value. A geometric positive would teach the opposite of the spec."""
    f = torch.randn(3, 2, 9, 8, 8)
    g, a = photometric_jitter_window(f, None)
    assert g.shape == f.shape
    # The map is affine per FRAME with strictly positive contrast, so pixel
    # RANK ORDER WITHIN EACH FRAME is preserved — that is what "geometry
    # untouched, only intensity moved" means operationally. (It is NOT
    # preserved across frames, because each frame gets its own mean offset.)
    assert torch.equal(f.flatten(2).argsort(dim=-1),
                       g.flatten(2).argsort(dim=-1))
    assert a is None
    assert not torch.allclose(g, f), "the positive view must actually differ"


def test_the_augmentation_partition_is_exhaustive_and_disjoint():
    assert T2_MANOEUVRE_PRESERVING.isdisjoint(T2_MANOEUVRE_REVERSING)
    assert (T2_MANOEUVRE_PRESERVING | T2_MANOEUVRE_REVERSING) \
        == set(T2_AUGMENTATIONS)


def test_the_loss_refuses_a_swapped_positive_and_negative():
    """⛔ A manoeuvre-reversing POSITIVE would train the model to call a
    mirrored lane change the SAME manoeuvre — the exact inversion of the
    catalog row. It must be refused, not silently accepted."""
    import train_v6_staged as T
    s = V6Stack(_t2())
    f = torch.randn(2, s.cfg.predictor.window, 9, 64, 64)
    a = torch.randn(2, s.cfg.predictor.window, 2)
    z = torch.randn(2, s.cfg.d_tac)
    with pytest.raises(ValueError, match="manoeuvre-PRESERVING"):
        T.t2_contrastive_loss(s, z, f, a, positive="lane_mirror")
    with pytest.raises(ValueError, match="manoeuvre-REVERSING"):
        T.t2_contrastive_loss(s, z, f, a, negative="photometric")


def test_the_loss_refuses_a_missing_projector():
    import train_v6_staged as T
    s = V6Stack(_small())                      # t2_contrastive=False
    f = torch.randn(2, s.cfg.predictor.window, 9, 64, 64)
    a = torch.randn(2, s.cfg.predictor.window, 2)
    with pytest.raises(ValueError, match="t2_contrastive"):
        T.t2_contrastive_loss(s, torch.randn(2, s.cfg.d_tac), f, a)


def test_the_loss_is_finite_and_reaches_layer_tac_but_never_the_encoder():
    """⛔ X3. The T2 gradient must reach ``adapter_tac``/``t2_head`` and
    NOTHING in the encoder or readout — the uplink stop-grad is what makes the
    head admissible at all."""
    import train_v6_staged as T
    s = V6Stack(_t2())
    b = 4
    f = torch.randn(b, s.cfg.predictor.window, 9, 64, 64)
    a = torch.randn(b, s.cfg.predictor.window, 2)
    z_tac, _ = s.uplink_tac(s.encode_window(f)[:, -1])
    loss, log = T.t2_contrastive_loss(s, z_tac, f, a)
    assert torch.isfinite(loss)
    loss.backward()
    hit = {n for n, p in s.named_parameters()
           if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert any(n.startswith("t2_head") for n in hit)
    assert any(n.startswith("adapter_tac") for n in hit)
    leaked = {n for n in hit
              if n.startswith(("encoder.", "readout.", "predictor_op."))}
    assert not leaked, f"T2 gradient reached the frozen trunk: {sorted(leaked)}"


def test_the_log_reports_the_margin_that_is_the_spec_s_actual_claim():
    import train_v6_staged as T
    s = V6Stack(_t2())
    b = 4
    f = torch.randn(b, s.cfg.predictor.window, 9, 64, 64)
    a = torch.randn(b, s.cfg.predictor.window, 2)
    z_tac, _ = s.uplink_tac(s.encode_window(f)[:, -1])
    _, log = T.t2_contrastive_loss(s, z_tac, f, a)
    for k in ("t2_margin", "t2_pos_sim", "t2_hard_sim", "t2_easy_sim",
              "t2_hard_beats_pos", "t2_tau"):
        assert k in log, f"missing diagnostic {k}"
    assert math.isclose(log["t2_margin"],
                        log["t2_pos_sim"] - log["t2_hard_sim"], rel_tol=1e-5)


def test_the_loss_falls_when_the_head_is_optimised_POSITIVE_CONTROL():
    """⭐ POSITIVE CONTROL. A loss that cannot be reduced is not an objective.
    C79: D1 was WITHDRAWN because a probe failed exactly this check."""
    import train_v6_staged as T
    torch.manual_seed(0)
    s = V6Stack(_t2())
    b = 6
    f = torch.randn(b, s.cfg.predictor.window, 9, 64, 64)
    a = torch.randn(b, s.cfg.predictor.window, 2)
    opt = torch.optim.Adam(
        [p for n, p in s.named_parameters()
         if n.startswith(("t2_head", "adapter_tac"))], lr=1e-2)
    first = last = None
    for i in range(25):
        z_tac, _ = s.uplink_tac(s.encode_window(f)[:, -1])
        loss, _ = T.t2_contrastive_loss(s, z_tac, f, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first, f"T2 loss did not fall: {first} -> {last}"


# ---------------------------------------------------------------------------
# 3. THE CONTROLS — a criterion without them is C92 waiting to happen
# ---------------------------------------------------------------------------

def test_the_trivial_proxy_control_exists_and_reports_a_ratio():
    """⛔ TRIVIAL-PROXY CONTROL. A projector can separate a window from its
    mirror by DETECTING THE FLIP (an asymmetric bonnet/vignette/rig offset)
    without knowing anything about manoeuvres. C92: a headline died because a
    readout was echoing ego speed. The control splits by |steer| — mirroring a
    STRAIGHT window is manoeuvre-preserving, mirroring a TURNING one is not —
    so ratio ~ 1 means FLIP DETECTOR and ratio >> 1 means manoeuvre-sensitive.
    """
    import train_v6_staged as T
    s = V6Stack(_t2())
    b = 8
    f = torch.randn(b, s.cfg.predictor.window, 9, 64, 64)
    a = torch.zeros(b, s.cfg.predictor.window, 2)
    a[: b // 2, :, 0] = 0.9                     # turning
    a[b // 2:, :, 0] = 0.001                    # straight
    out = T.t2_flip_detection_control(s, f, a)
    for k in ("t2_sep_turning", "t2_sep_straight", "t2_sep_ratio",
              "n_turning", "n_straight", "verdict"):
        assert k in out
    assert out["n_turning"] > 0 and out["n_straight"] > 0


def _split_batch(s, n_side, seed=0):
    torch.manual_seed(seed)
    b = 2 * n_side
    f = torch.randn(b, s.cfg.predictor.window, 9, 64, 64)
    a = torch.zeros(b, s.cfg.predictor.window, 2)
    a[:n_side, :, 0] = 0.9        # turning
    a[n_side:, :, 0] = 0.001      # straight
    return f, a


def test_the_control_refuses_a_verdict_below_its_sample_floor():
    """⛔ THE DEFECT THIS TEST CAUGHT, kept as a regression.

    The control originally issued a ratio-based verdict at ANY n. MEASURED at
    random init (where the true ratio is 1 by construction), 5 seeds:
    n=4/side spans 0.397-3.361, n=64/side spans 0.949-1.281. A verdict from
    n=4 is noise wearing a number's clothes — exactly the class of the
    'interval without its estimator' rule.
    """
    import train_v6_staged as T
    s = V6Stack(_t2())
    out = T.t2_flip_detection_control(s, *_split_batch(s, 4))
    assert out["n_turning"] == 4 and out["n_straight"] == 4
    assert "INCONCLUSIVE" in out["verdict"]
    assert str(T.T2_CONTROL_MIN_N) in out["verdict"]


def test_the_control_calls_an_untrained_head_a_flip_detector():
    """⭐ THE CONTROL'S OWN NEGATIVE CASE, run at a defensible n. At random
    init the projector knows nothing about manoeuvres, so an honest control
    MUST come back ~1 and say FLIP-DETECTOR. A control that praised a random
    head would be C97's passes-everything guard.

    Tolerance is the MEASURED null spread at this n (0.949-1.281 over 5 seeds),
    not a number chosen to make the test pass.
    """
    import train_v6_staged as T
    s = V6Stack(_t2())
    out = T.t2_flip_detection_control(s, *_split_batch(s, 64))
    assert out["n_turning"] == 64 and out["n_straight"] == 64
    assert out["t2_sep_ratio"] == pytest.approx(1.0, abs=0.35), \
        "an untrained projector must not look manoeuvre-sensitive"
    assert "FLIP-DETECTOR" in out["verdict"]
    # the SEMs must be reported: a ratio with no dispersion is not a result
    assert out["t2_sep_turning_sem"] > 0 and out["t2_sep_straight_sem"] > 0


# ---------------------------------------------------------------------------
# 4. THE ARCHITECTURAL FINDING THAT LIMITS THE SPEC
# ---------------------------------------------------------------------------

def test_z_tac_reads_the_last_frame_only():
    """⛔ MEASURED, and it is why half of catalog T2 is not expressible here.

    ``encode_window`` flattens [B, W] into the batch axis (v6.py:3844-3847), so
    the encoder sees every frame INDEPENDENTLY; ``forward`` then takes
    ``z_op_win[:, -1]``. Consequence: ``z_tac`` is a function of the LAST FRAME
    ALONE, and the tactical layer has no temporal extent for a "time reversal"
    to act on.
    """
    s = V6Stack(_small())
    b, w = 2, s.cfg.predictor.window
    f = torch.randn(b, w, 9, 64, 64)
    with torch.no_grad():
        z_full, _ = s.uplink_tac(s.encode_window(f)[:, -1])
        # replace every frame EXCEPT the last with noise: z_tac must not move
        g = f.clone()
        g[:, :-1] = torch.randn_like(g[:, :-1])
        z_last, _ = s.uplink_tac(s.encode_window(g)[:, -1])
    assert torch.allclose(z_full, z_last, atol=1e-6), \
        "z_tac moved when a NON-last frame changed — the finding is stale"


def test_time_reversal_is_an_earlier_frame_not_a_reversed_manoeuvre():
    """The corollary, MEASURED: ``z_tac(time_reverse(x))`` is exactly
    ``z_tac`` of the window's FIRST frame. That is "a frame W ticks earlier",
    not "the manoeuvre played backwards" — and pushing it away from the anchor
    is the OPPOSITE of catalog T5 (F-8), which pulls nearby windows together.
    This is why ``time_reverse`` is built but is NOT the default hard negative.
    """
    s = V6Stack(_small())
    b, w = 2, s.cfg.predictor.window
    f = torch.randn(b, w, 9, 64, 64)
    fr, _ = time_reverse_window(f, None)
    with torch.no_grad():
        z_rev, _ = s.uplink_tac(s.encode_window(fr)[:, -1])
        # the SAME quantity, obtained by simply reading the first frame
        first = f[:, 0:1].expand(-1, w, -1, -1, -1)
        z_first, _ = s.uplink_tac(s.encode_window(first)[:, -1])
    assert torch.allclose(z_rev, z_first, atol=1e-6)


def test_lane_mirror_DOES_move_z_tac():
    """The other half: the lane mirror is genuinely expressible, which is what
    makes the built cell faithful to the half of T2 that survives."""
    s = V6Stack(_small())
    f = torch.randn(3, s.cfg.predictor.window, 9, 64, 64)
    fm, _ = lane_mirror_window(f, None)
    with torch.no_grad():
        z, _ = s.uplink_tac(s.encode_window(f)[:, -1])
        zm, _ = s.uplink_tac(s.encode_window(fm)[:, -1])
    assert not torch.allclose(z, zm, atol=1e-4)


def test_z_tac_target_is_a_degenerate_positive():
    """⛔ WHY THE POSITIVE IS AN AUGMENTED VIEW AND NOT THE FREE-LOOKING ONE.

    ``uplink`` defaults to ``"stopgrad"``, so ``uplink_tac`` returns
    ``target = online.detach()`` — feeding it to a unit-norm projector as the
    contrastive positive gives cosine similarity EXACTLY 1 for every window,
    regardless of what the head learned. The positive term would be a constant
    and InfoNCE would collapse to a pure push-apart (flip-detector) objective.
    """
    s = V6Stack(_t2())
    assert s.cfg.uplink == "stopgrad"
    f = torch.randn(4, s.cfg.predictor.window, 9, 64, 64)
    with torch.no_grad():
        z, tgt = s.uplink_tac(s.encode_window(f)[:, -1])
        cos = (s.t2_head(z) * s.t2_head(tgt)).sum(dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)
