"""D-TAC1 — the FACTORISED tactical head (tanitad/refs/refc_tactical.py +
the ``factored_maneuver`` seam in tanitad/refs/refc.py + scripts/refc_train.py).

What the defect is, and therefore what these tests must pin (MEASURED source:
``TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
2026-08-03-lan-refc-e0/LAN_E0_RESULTS.md`` section 5, REF-C-base 30k, n = 859):
``accelerate`` 0/93 predicted, ``brake_stop`` 7/78, while the turns are emitted
at very nearly their true rate (106 vs 110, 71 vs 68). The longitudinal mass
lands entirely in ``lane_keep`` (675 predicted vs 510 true = +165; missing
longitudinal = 93 + 78 - 7 = 164).

Pinned here:
(a) every constant MIRRORED from scripts/refb_labels.py is equal to its original
    (refc_tactical must stay import-light, so equality is a TEST not an import);
(b) the COMPONENT-vs-FAMILY self-consistency control: collapsing the factorised
    labels reproduces ``classify_maneuver`` / ``classify_maneuver_v2``
    ELEMENTWISE over fuzzed kinematics — a divergence would silently train the
    arm on a different target and make the whole A/B non-attributable;
(c) derive <-> invert is an exact round-trip and derive emits a valid
    distribution, so ``maneuver_logits`` keeps its meaning for every downstream
    reader and the 0-GPU counterfactual decode (E-A1 in the pre-registration) is
    a real inverse and not an approximation;
(d) logit adjustment is the IDENTITY under a uniform prior at any tau (so an
    un-updated prior can never silently change a published decode) and DOES let
    a rare class win once the prior is skewed — the decision-rule claim, with
    its own negative control;
(e) the gated-flag discipline: flag OFF -> state_dict and forward outputs are
    byte-identical to today; flag ON -> the 5-way head is gone, the two heads
    and the two grafts are present, ``lon_to_anchor`` is EXACTLY zero at init
    (so its effect on selection is attributable) and still receives gradient;
(f) ``tactical_speed_input`` is LIVE: the tactical logits move when v0 moves and
    do NOT when the flag is off — the negative control for F1's wiring, run
    BEFORE any claim is made from that input;
(g) the trainer holds the total tactical aux weight at MANEUVER_WEIGHT, fails
    loud on labeler drift, and runs end-to-end with the seam on.
CPU-only, synthetic data.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels  # noqa: E402  (scripts/refb_labels.py)
import refc_train  # noqa: E402  (scripts/refc_train.py)
from tanitad.refs import refc_tactical as tac  # noqa: E402
from tanitad.refs.refc import (N_LAT_MAN, N_LON_MAN,  # noqa: E402
                               N_MANEUVERS, RefCModel, param_breakdown,
                               refc_config, refc_f1only_config,
                               refc_factored_config, refc_smoke_config)

from test_refc import _make_cached_root, _batch, _poses  # noqa: E402


def _factored_smoke(speed_input: bool = True, tau: float = 1.0):
    cfg = refc_smoke_config()
    cfg.factored_maneuver = True
    cfg.tactical_speed_input = speed_input
    cfg.man_prior_tau = tau
    return cfg


# ---------- (a) the mirrored constants are pinned to refb_labels --------------

def test_mirrored_constants_match_refb_labels():
    """refc_tactical mirrors refb_labels' thresholds because tanitad.refs.* must
    not import from scripts/ (the N_MANEUVERS convention, refc.py L97-99). The
    mirror is only safe if a test pins it — otherwise the label pipeline and the
    model's notion of the same label drift apart silently."""
    assert tac.YAW_TURN_RAD == refb_labels.YAW_TURN_RAD
    assert tac.DV_ACCEL_MS == refb_labels.DV_ACCEL_MS
    assert tac.DV_BRAKE_MS == refb_labels.DV_BRAKE_MS
    assert tac.STOP_V_MS == refb_labels.STOP_V_MS
    assert tac.MOVING_V_MS == refb_labels.MOVING_V_MS
    assert tac.MIN_ARC_M == refb_labels.MIN_ARC_M
    assert tac.CURV_TURN_MAN_PER_M == pytest.approx(
        refb_labels.CURV_TURN_MAN_PER_M)
    assert tac.LABEL_HORIZON == refb_labels.LABEL_HORIZON
    # 5-way index order (a permutation here would corrupt every collapse).
    assert (tac.LANE_KEEP, tac.TURN_LEFT, tac.TURN_RIGHT, tac.ACCELERATE,
            tac.BRAKE_STOP) == (refb_labels.LANE_KEEP, refb_labels.TURN_LEFT,
                                refb_labels.TURN_RIGHT, refb_labels.ACCELERATE,
                                refb_labels.BRAKE_STOP)
    assert tac.N_MAN5 == N_MANEUVERS == 5
    assert (N_LAT_MAN, N_LON_MAN) == (tac.N_LAT, tac.N_LON) == (3, 3)
    assert len(tac.LAT_CLASSES) == 3 and len(tac.LON_CLASSES) == 3
    # The table must be a total map over the product space.
    assert len(tac.COLLAPSE_TABLE) == 3
    assert all(len(row) == 3 for row in tac.COLLAPSE_TABLE)


# ---------- (b) THE self-consistency control ----------------------------------

def _fuzz_windows(n: int = 400, seed: int = 7):
    """Random unicycle windows spanning turns, braking, accelerating, stops."""
    g = torch.Generator().manual_seed(seed)
    pl, fut = [], []
    for _ in range(n):
        v0 = float(torch.rand(1, generator=g)) * 28.0
        yr = float(torch.rand(1, generator=g) - 0.5) * 1.4      # +-0.7 rad/s
        ac = float(torch.rand(1, generator=g) - 0.5) * 12.0     # +-6 m/s^2
        p = _poses(30, v0=v0, yaw_rate=yr, accel=ac)
        pl.append(p[0])
        fut.append(p[1:26])
    return torch.stack(pl), torch.stack(fut)


def test_collapse_reproduces_the_5way_labeler_elementwise():
    """COMPONENT-vs-FAMILY control (CLAUDE.md / brief): the factorised labels,
    pushed back through the priority collapse, must equal the shipped 5-way
    label for EVERY window under BOTH labelers. This is the check that makes the
    D-TAC1 arm attributable to structure rather than to a changed target."""
    pose_last, fut = _fuzz_windows()
    m5 = refb_labels.window_maneuver_labels(pose_last, fut, horizon=20)
    lat, lon = tac.window_factored_labels(pose_last, fut, horizon=20)
    assert torch.equal(tac.collapse(lat, lon), m5)

    m5v = refb_labels.window_maneuver_labels_v2(pose_last, fut, horizon=20)
    lat2, lon2 = tac.window_factored_labels_v2(pose_last, fut, horizon=20)
    assert torch.equal(tac.collapse(lat2, lon2), m5v)

    # The fuzz must actually EXERCISE the classes, else the control is vacuous
    # (a labeler that returned a constant would also "agree" on a constant fuzz).
    assert len(torch.unique(m5)) >= 4, "fuzz did not cover the 5-way vocabulary"
    assert len(torch.unique(lon)) == 3, "fuzz did not cover all 3 lon classes"
    assert len(torch.unique(lat)) >= 2, "fuzz did not cover the lateral axis"


def test_priority_collapse_destroys_longitudinal_information():
    """The mechanism, made explicit: on windows the 5-way calls a TURN, the
    factorised longitudinal class is still live and is thrown away. The count is
    the label-side half of the defect and must be > 0 on a corpus with turns."""
    pose_last, fut = _fuzz_windows(600, seed=11)
    m5 = refb_labels.window_maneuver_labels(pose_last, fut, horizon=20)
    lat, lon = tac.window_factored_labels(pose_last, fut, horizon=20)
    turning = (m5 == tac.TURN_LEFT) | (m5 == tac.TURN_RIGHT)
    destroyed = turning & (lon != tac.LON_STEADY)
    assert int(destroyed.sum()) > 0
    # ...and the 5-way class carries no trace of it: every such window reads as
    # a pure lateral class.
    assert not bool(((m5[destroyed] == tac.ACCELERATE)
                     | (m5[destroyed] == tac.BRAKE_STOP)).any())


def test_lon_classes_are_a_partition_not_another_priority_collapse():
    """brake and accelerate are DISJOINT at these thresholds, so `lon` is a real
    3-way partition. If someone widens DV_* until they overlap, the factorised
    head silently becomes another priority collapse — this catches that."""
    g = torch.Generator().manual_seed(5)
    dv = (torch.rand(4000, generator=g) - 0.5) * 40.0
    v0 = torch.rand(4000, generator=g) * 30.0
    v1 = (v0 + dv).clamp_min(0.0)
    dv = v1 - v0
    _, lon = tac.factor_from_kinematics(torch.zeros_like(dv), dv, v0, v1)
    brake_rule = (dv < tac.DV_BRAKE_MS) | ((v1 < tac.STOP_V_MS)
                                           & (v0 >= tac.MOVING_V_MS))
    accel_rule = dv > tac.DV_ACCEL_MS
    assert not bool((brake_rule & accel_rule).any())
    assert torch.equal(lon == tac.LON_BRAKE_STOP, brake_rule)


# ---------- (c) derive <-> invert ---------------------------------------------

def test_derive_is_a_distribution_and_invert_round_trips():
    g = torch.Generator().manual_seed(1)
    lat = torch.randn(64, 3, generator=g) * 3.0
    lon = torch.randn(64, 3, generator=g) * 3.0
    lp5 = tac.derive_man5_logprobs(lat, lon)
    assert lp5.shape == (64, 5)
    assert torch.allclose(lp5.exp().sum(-1), torch.ones(64), atol=1e-5)

    back_lat, back_lon = tac.invert_man5(lp5)
    assert torch.allclose(back_lat, torch.log_softmax(lat, -1), atol=1e-5)
    assert torch.allclose(back_lon, torch.log_softmax(lon, -1), atol=1e-5)


def test_invert_recovers_the_conditional_a_mixed_argmax_hides():
    """The instrument behind experiment E-A1. Build a 5-way posterior in which
    accelerate can NEVER be the argmax (a turn outranks it) yet the CONDITIONAL
    longitudinal posterior clearly prefers accelerate; the inversion must
    surface it. This is exactly the situation LAN_E0 measured at n = 859."""
    # P_lat = (lane_keep .45, turn_left .40, turn_right .15)
    # P_lon = (brake .10, steady .30, accel .60)  -> P5(accel) = .27 < .40
    lat = torch.tensor([[0.45, 0.40, 0.15]]).log()
    lon = torch.tensor([[0.10, 0.30, 0.60]]).log()
    lp5 = tac.derive_man5_logprobs(lat, lon)
    assert int(lp5.argmax(-1)) == tac.TURN_LEFT          # the mixed decision
    _, back_lon = tac.invert_man5(lp5)
    assert int(back_lon.argmax(-1)) == tac.LON_ACCELERATE  # the real one
    assert float(back_lon.exp()[0, tac.LON_ACCELERATE]) == pytest.approx(0.6,
                                                                        abs=1e-4)


# ---------- (d) the decision rule + its negative control ----------------------

def test_logit_adjust_is_identity_under_a_uniform_prior():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(128, 3, generator=g)
    uni = torch.full((3,), -math.log(3.0))
    for tau in (0.0, 0.5, 1.0, 4.0):
        adj = tac.logit_adjust(logits, uni, tau)
        assert torch.equal(adj.argmax(-1), logits.argmax(-1))


def test_logit_adjust_lets_a_rare_class_win_and_only_then():
    """A 12 %-prior class with a 0.30 posterior loses the raw argmax to a
    75 %-prior class at 0.55 and wins the balanced one. NEGATIVE CONTROL: with
    tau = 0 nothing moves, so any emission change is attributable to tau."""
    p = torch.tensor([[0.15, 0.55, 0.30]])            # brake / steady / accel
    prior = torch.tensor([0.115, 0.749, 0.137]).log()  # the MEASURED marginal
    raw = tac.logit_adjust(p.log(), prior, 0.0)
    adj = tac.logit_adjust(p.log(), prior, 1.0)
    assert int(raw.argmax(-1)) == tac.LON_STEADY
    assert int(adj.argmax(-1)) == tac.LON_ACCELERATE


def test_class_log_prior_and_ema_update():
    idx = torch.tensor([1, 1, 1, 0, 2])
    lp = tac.class_log_prior(idx, 3)
    assert torch.allclose(lp.exp(), torch.tensor([0.2, 0.6, 0.2]), atol=1e-5)

    model = RefCModel(_factored_smoke())
    assert torch.allclose(model.lon_log_prior.exp(),
                          torch.full((3,), 1 / 3), atol=1e-6)
    for _ in range(200):                       # EMA towards a skewed batch
        model.update_tactical_prior(torch.zeros(10, dtype=torch.long), idx,
                                    momentum=0.9)
    assert torch.allclose(model.lon_log_prior.exp().sum(), torch.tensor(1.0),
                          atol=1e-5)
    assert model.lon_log_prior.exp()[1] > 0.55
    assert torch.allclose(model.lat_log_prior.exp(),
                          torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)


# ---------- (e) gated-flag discipline -----------------------------------------

def test_flag_off_is_byte_identical(tmp_path):
    """The whole gated-flag contract: with factored_maneuver off the model is
    indistinguishable from one that never had the feature — same keys, same
    values under the same seed, same forward outputs."""
    torch.manual_seed(0)
    base = RefCModel(refc_smoke_config()).eval()
    torch.manual_seed(0)
    again = RefCModel(refc_smoke_config()).eval()
    sd_a, sd_b = base.state_dict(), again.state_dict()
    assert set(sd_a) == set(sd_b)
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k
    assert "maneuver_head.0.weight" in sd_a
    assert not any(k.startswith(("lat_head", "lon_head",
                                 "tactical_trunk")) for k in sd_a)
    assert "lat_log_prior" not in sd_a

    root = _make_cached_root(tmp_path)
    batch = _batch(root, base.cfg)
    out = base(batch["frames"], v0=batch["pose_last"][:, 3])
    assert out["maneuver_logits"].shape[-1] == N_MANEUVERS
    for k in ("lat_logits", "lon_logits", "lat_decision", "lon_decision",
              "maneuver_decision"):
        assert k not in out


def test_factored_keys_zero_init_and_gradient_reach(tmp_path):
    model = RefCModel(_factored_smoke())
    sd = model.state_dict()
    assert not any(k.startswith("maneuver_head") for k in sd)
    assert "decoder.maneuver_to_anchor.weight" not in sd
    for k in ("tactical_trunk.0.weight", "lat_head.weight",
              "lon_head.weight", "lat_log_prior",
              "lon_log_prior", "decoder.lat_to_anchor.weight",
              "decoder.lon_to_anchor.weight"):
        assert k in sd, k
    assert sd["lat_head.weight"].shape[0] == N_LAT_MAN
    assert sd["lon_head.weight"].shape[0] == N_LON_MAN
    # ZERO-INIT: the longitudinal selection surface starts EXACTLY off, so the
    # step-0 anchor ranking is today's lateral-only prior and every later change
    # is attributable to this seam (the ctx_to_cond discipline).
    assert float(model.decoder.lon_to_anchor.weight.abs().max()) == 0.0
    assert float(model.decoder.lat_to_anchor.weight.abs().max()) > 0.0

    # ...and gated is not DEAD: the zero-init graft still receives gradient.
    root = _make_cached_root(tmp_path)
    batch = _batch(root, model.cfg)
    model.train()
    losses = refc_train.compute_losses(model, batch, mode="diffusion")
    losses["loss"].backward()
    dead = [n for n, p in model.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad).all()]
    assert dead == []
    assert float(model.decoder.lon_to_anchor.weight.grad.abs().max()) > 0.0


def test_derived_maneuver_logits_stay_a_valid_5way_field(tmp_path):
    """Downstream readers (plan_fan HUD, eval harness, closed-loop logger) all
    take `maneuver_logits` [B, 5]. Factorising must not take that away."""
    model = RefCModel(_factored_smoke()).eval()
    root = _make_cached_root(tmp_path)
    batch = _batch(root, model.cfg)
    out = model(batch["frames"], v0=batch["pose_last"][:, 3])
    lp = out["maneuver_logits"]
    assert lp.shape[-1] == N_MANEUVERS
    assert torch.allclose(lp.exp().sum(-1), torch.ones(lp.shape[0]), atol=1e-5)
    # and it agrees with the factored decision at tau = 0
    model.cfg.man_prior_tau = 0.0
    out0 = model(batch["frames"], v0=batch["pose_last"][:, 3])
    assert torch.equal(out0["maneuver_decision"],
                       tac.collapse(out0["lat_logits"].argmax(-1),
                                    out0["lon_logits"].argmax(-1)))


def test_anchor_grafts_are_separately_ablatable(tmp_path):
    """Two summed rank-3 grafts, not one: `lon_to_anchor` can be zeroed and the
    anchor logits must return EXACTLY to the lateral-only ranking. That is the
    ablation the single rank-5 graft never allowed."""
    model = RefCModel(_factored_smoke()).eval()
    root = _make_cached_root(tmp_path)
    batch = _batch(root, model.cfg)
    frames, v0 = batch["frames"], batch["pose_last"][:, 3]

    with torch.no_grad():                    # make the lon graft live
        model.decoder.lon_to_anchor.weight.normal_(0.0, 0.5)
    live = model(frames, v0=v0)["anchor_logits"].clone()
    with torch.no_grad():
        model.decoder.lon_to_anchor.weight.zero_()
    off = model(frames, v0=v0)["anchor_logits"]
    assert not torch.allclose(live, off, atol=1e-6)

    with torch.no_grad():
        model.decoder.lat_to_anchor.weight.zero_()
    none = model(frames, v0=v0)["anchor_logits"]
    assert not torch.allclose(off, none, atol=1e-6)


def test_external_maneuver_logits_are_factorised_not_dropped(tmp_path):
    """An EXTERNAL tactical brain speaks the 5-way surface. With the factored
    seam on, `maneuver_to_anchor` no longer exists — the external prior must be
    inverted onto the two grafts rather than silently ignored."""
    model = RefCModel(_factored_smoke()).eval()
    with torch.no_grad():
        model.decoder.lon_to_anchor.weight.normal_(0.0, 0.5)
    root = _make_cached_root(tmp_path)
    batch = _batch(root, model.cfg)
    frames, v0 = batch["frames"], batch["pose_last"][:, 3]
    b = frames.shape[0]
    braking = torch.zeros(b, N_MANEUVERS)
    braking[:, tac.BRAKE_STOP] = 6.0
    accelerating = torch.zeros(b, N_MANEUVERS)
    accelerating[:, tac.ACCELERATE] = 6.0
    a = model(frames, v0=v0, maneuver_logits=braking)["anchor_logits"]
    c = model(frames, v0=v0, maneuver_logits=accelerating)["anchor_logits"]
    assert not torch.allclose(a, c, atol=1e-5)


# ---------- (f) F1: the speed input, with its negative control ----------------

def test_tactical_speed_input_is_live_and_gated(tmp_path):
    """NEGATIVE CONTROL FIRST (brief: prove the metric can discriminate). With
    the flag OFF the tactical logits must be BIT-IDENTICAL across a 0 -> 25 m/s
    change of v0 — that is the measured defect (`maneuver_head` reads `pooled`
    alone, refc.py forward). With it ON they must move."""
    root = _make_cached_root(tmp_path)
    slow = torch.zeros(4)
    fast = torch.full((4,), 25.0)

    off = RefCModel(_factored_smoke(speed_input=False)).eval()
    batch = _batch(root, off.cfg)
    a = off(batch["frames"], v0=slow)
    b = off(batch["frames"], v0=fast)
    assert torch.equal(a["lat_logits"], b["lat_logits"])
    assert torch.equal(a["lon_logits"], b["lon_logits"])   # THE defect

    on = RefCModel(_factored_smoke(speed_input=True)).eval()
    c = on(batch["frames"], v0=slow)
    d = on(batch["frames"], v0=fast)
    assert not torch.allclose(c["lon_logits"], d["lon_logits"], atol=1e-6)
    assert float((c["lon_logits"] - d["lon_logits"]).abs().max()) > 1e-4


def _f1only_smoke():
    """The shipped 5-way head + the ego speed, and NOTHING else changed."""
    cfg = refc_smoke_config()
    cfg.tactical_speed_input = True          # WITHOUT factored_maneuver
    return cfg


def test_speed_input_on_the_SHIPPED_5way_head_is_live_and_gated(tmp_path):
    """F1 in isolation — the defect and its fix on the head REF-C actually ships.

    NEGATIVE CONTROL FIRST: with the flag off, `maneuver_logits` must be
    BIT-IDENTICAL across a 0 -> 25 m/s change of v0. That is the measured defect
    (`man_logits = self.maneuver_head(pooled)`, the image embedding alone, while
    the label is dv = v(t+2s) - v(t)). An instrument that could not show the
    logits frozen there cannot certify that turning the flag on unfroze them.

    ⚠️ v0 DOES reach the decoder (via `measurement`), so `traj` moves in BOTH
    arms and is not a discriminator — the control has to be read on
    `maneuver_logits`, which is the tensor the tactical CE and the H19 anchor
    reweight consume.
    """
    root = _make_cached_root(tmp_path)
    slow = torch.zeros(4)
    fast = torch.full((4,), 25.0)

    off = RefCModel(refc_smoke_config()).eval()
    batch = _batch(root, off.cfg)
    a = off(batch["frames"], v0=slow)["maneuver_logits"]
    b = off(batch["frames"], v0=fast)["maneuver_logits"]
    assert torch.equal(a, b)                              # THE defect

    on = RefCModel(_f1only_smoke()).eval()
    c = on(batch["frames"], v0=slow)["maneuver_logits"]
    d = on(batch["frames"], v0=fast)["maneuver_logits"]
    assert not torch.allclose(c, d, atol=1e-6)
    assert float((c - d).abs().max()) > 1e-4
    # ...and the H19 anchor reweight therefore moves too: the speed reaches
    # SELECTION, not merely the reported class.
    assert not torch.allclose(on(batch["frames"], v0=slow)["anchor_logits"],
                              on(batch["frames"], v0=fast)["anchor_logits"],
                              atol=1e-6)


def test_f1only_is_the_shipped_head_widened_by_exactly_one_column(tmp_path):
    """Structure control: the F1 arm must differ from the base model in ONE
    weight matrix's input width and in nothing else — no new modules, no new
    buffers, no lost keys. Otherwise a delta is not attributable to the input."""
    base = RefCModel(refc_smoke_config())
    f1 = RefCModel(_f1only_smoke())
    sd_b, sd_f = base.state_dict(), f1.state_dict()
    assert set(sd_b) == set(sd_f)                 # same keys, both directions
    diff = [k for k in sd_b if sd_b[k].shape != sd_f[k].shape]
    assert diff == ["maneuver_head.0.weight"], diff
    assert sd_f["maneuver_head.0.weight"].shape[1] == \
        sd_b["maneuver_head.0.weight"].shape[1] + 1
    # the factored seam is absent — this arm is NOT the factored one
    assert not any(k.startswith(("lat_head", "lon_head", "tactical_trunk"))
                   for k in sd_f)
    assert "lat_log_prior" not in sd_f
    assert "decoder.maneuver_to_anchor.weight" in sd_f
    # ...and the new column is not dead: it receives gradient.
    root = _make_cached_root(tmp_path)
    batch = _batch(root, f1.cfg)
    f1.train()
    refc_train.compute_losses(f1, batch, mode="diffusion")["loss"].backward()
    g = f1.maneuver_head[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert float(g[:, -1].abs().max()) > 0.0, "the speed column got no gradient"


def test_f1only_is_not_a_capacity_change():
    """MEASURED at build, on the REAL base preset: exactly ``aux_hidden`` extra
    parameters — one input column into ``maneuver_head.0`` — and not one more.
    Pinned EXACTLY (not as a band) because the whole point of this arm is that a
    win cannot be attributed to capacity."""
    with torch.device("meta"):
        base = param_breakdown(RefCModel(refc_config()))
        f1 = param_breakdown(RefCModel(refc_f1only_config()))
        fact = param_breakdown(RefCModel(refc_factored_config()))
    aux_hidden = refc_config().decoder.aux_hidden
    assert f1["total"] - base["total"] == aux_hidden == 384
    assert f1["aux"] - base["aux"] == aux_hidden      # it lands in the aux heads
    assert f1["decoder"] == base["decoder"]           # graft untouched (rank-5)
    # strictly cheaper than the factored arm, which also splits the graft
    assert f1["total"] < fact["total"]
    cfg = refc_f1only_config()
    assert (cfg.tactical_speed_input, cfg.factored_maneuver) == (True, False)
    assert cfg.man_prior_tau == 0.0                   # decode rule UNCHANGED


def test_man_prior_tau_cannot_move_the_trajectory(tmp_path):
    """⭐ THE FOUR-FAMILY GUARANTEE, pinned as a test rather than argued.

    ``man_prior_tau`` is a REPORTING transform. If it could reach the trajectory,
    a "free read-out patch" would silently be a model change and the
    LONGITUDINAL / LATERAL / STRATEGIC / ADE families would all need re-scoring.
    It cannot: the graft consumes ``lat_prior``/``lon_prior``, which are never
    logit-adjusted. Same weights, same fixed input, two taus -> bit-identical
    everything except the two decision fields.

    (An earlier adversarial pass reported a 0.0039 leak here; it was confounded
    by drawing ``v0`` twice — ``v0`` reaches the decoder. One fixed input.)
    """
    torch.manual_seed(0)
    model = RefCModel(_factored_smoke(tau=0.0)).eval()
    root = _make_cached_root(tmp_path)
    batch = _batch(root, model.cfg)
    frames, v0 = batch["frames"], batch["pose_last"][:, 3]      # ONE draw
    # make the prior non-uniform, else tau is trivially inert
    with torch.no_grad():
        model.lon_log_prior.copy_(torch.tensor([0.1122, 0.7104,
                                                0.1774]).log())
    lo = model(frames, v0=v0)
    model.cfg.man_prior_tau = 2.0
    hi = model(frames, v0=v0)
    for k in ("traj", "anchor_logits", "anchor_traj", "offset", "sel_idx",
              "maneuver_logits", "lat_logits", "lon_logits", "route_logits",
              "pooled", "measurement", "ctx"):
        assert torch.equal(lo[k], hi[k]), f"man_prior_tau moved {k}"

    # VACUITY CONTROL — an invariance proved with an INERT knob proves nothing.
    # (i) tau = 2 under THIS prior is a live transform: it flips the argmax of a
    # posterior where the rare class is second. (ii) the model's decision field
    # really is that transform of its own logits, and at tau = 0 it is the plain
    # argmax — so the bit-identity above is the graft being untouched, not a
    # dead field. Deliberately NOT "the 4 smoke windows must flip": whether a
    # random-init head happens to sit near a boundary is luck, not evidence.
    probe = torch.tensor([[0.15, 0.55, 0.30]]).log()
    assert int(tac.logit_adjust(probe, model.lon_log_prior, 0.0).argmax(-1)) \
        != int(tac.logit_adjust(probe, model.lon_log_prior, 2.0).argmax(-1))
    assert torch.equal(hi["lon_decision"],
                       tac.logit_adjust(hi["lon_logits"], model.lon_log_prior,
                                        2.0).argmax(-1))
    assert torch.equal(lo["lon_decision"], lo["lon_logits"].argmax(-1))


# ---------- (g) trainer wiring ------------------------------------------------

def test_total_tactical_aux_weight_is_held_constant():
    """The first confound anyone raises about an added head is "the aux got
    louder". It is removed by construction: LAT + LON == the single 5-way
    weight, exactly."""
    assert refc_train.LAT_WEIGHT + refc_train.LON_WEIGHT == pytest.approx(
        refc_train.MANEUVER_WEIGHT)


def test_trainer_losses_and_reported_metrics(tmp_path):
    model = RefCModel(_factored_smoke())
    model.train()
    root = _make_cached_root(tmp_path)
    batch = _batch(root, model.cfg, n=6)
    d = refc_train.compute_losses(model, batch, mode="diffusion")
    for k in ("lat", "lon", "lat_acc", "lon_acc", "lon_active_pred",
              "lon_active_tgt", "graft_lat_norm", "graft_lon_norm",
              "conf_norm"):
        assert k in d, k
        assert torch.isfinite(d[k]).all(), k
    # `man` must equal the weighted mean of the two CEs, so MANEUVER_WEIGHT *
    # man == LAT_WEIGHT * lat + LON_WEIGHT * lon.
    expect = ((refc_train.LAT_WEIGHT * d["lat"]
               + refc_train.LON_WEIGHT * d["lon"])
              / refc_train.MANEUVER_WEIGHT)
    assert float(d["man"]) == pytest.approx(float(expect), rel=1e-5)
    # norm-parity monitor: the zero-init graft contributes exactly nothing yet.
    assert float(d["graft_lon_norm"]) == 0.0


def test_trainer_fails_loud_on_labeler_drift(tmp_path, monkeypatch):
    """If refc_tactical and refb_labels ever disagree the run must DIE, not
    train on an undocumented target. Simulated by poisoning the collapse."""
    model = RefCModel(_factored_smoke())
    model.train()
    root = _make_cached_root(tmp_path)
    batch = _batch(root, model.cfg)
    monkeypatch.setattr(tac, "collapse",
                        lambda lat, lon: torch.full_like(lat, 4))
    monkeypatch.setattr(refc_train.refc_tactical, "collapse",
                        lambda lat, lon: torch.full_like(lat, 4))
    with pytest.raises(ValueError, match="DRIFTED"):
        refc_train.compute_losses(model, batch, mode="diffusion")


def test_trainer_end_to_end_with_the_seam(tmp_path):
    root = _make_cached_root(tmp_path)
    out_dir = tmp_path / "run-factored"
    metrics = refc_train.main([
        "--data-root", str(root), "--out", str(out_dir), "--steps", "2",
        "--batch", "4", "--smoke", "--log-every", "1", "--mode", "diffusion",
        "--factored-maneuver", "--tactical-speed-input", "--man-prior-tau", "1.0",
    ])
    assert metrics["steps"] >= 2
    ck = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert "lat_head.weight" in ck["model"]
    assert "maneuver_head.2.weight" not in ck["model"]
    assert "lon_log_prior" in ck["model"]           # the prior travels
    import json
    cfgj = json.loads((out_dir / "config.json").read_text())
    assert cfgj["cfg"]["factored_maneuver"] is True
    assert cfgj["loss_weights"]["man_total_held_at"] == refc_train.MANEUVER_WEIGHT


def test_cli_rejects_the_orphan_DECODE_lever_only(tmp_path):
    """--man-prior-tau acts on the per-axis priors, which only the factored seam
    registers, so it stays coupled. --tactical-speed-input does NOT: it is the
    F1-only arm and coupling it left F1 estimable only as `full - f2only`."""
    root = _make_cached_root(tmp_path)
    with pytest.raises(SystemExit, match="requires --factored-maneuver"):
        refc_train.main(["--data-root", str(root), "--out", str(tmp_path / "x"),
                         "--steps", "1", "--batch", "4", "--smoke",
                         "--man-prior-tau", "1.0"])


def test_trainer_end_to_end_f1only(tmp_path):
    """The F1-only arm trains: 5-way head, 5-way CE, rank-5 graft, one extra
    input column. The checkpoint must prove it is NOT the factored arm."""
    import json
    root = _make_cached_root(tmp_path)
    out_dir = tmp_path / "run-f1only"
    metrics = refc_train.main([
        "--data-root", str(root), "--out", str(out_dir), "--steps", "2",
        "--batch", "4", "--smoke", "--log-every", "1", "--mode", "diffusion",
        "--tactical-speed-input",
    ])
    assert metrics["steps"] >= 2
    ck = torch.load(out_dir / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    assert "maneuver_head.0.weight" in ck["model"]
    assert "lat_head.weight" not in ck["model"]
    assert "lon_log_prior" not in ck["model"]
    cfgj = json.loads((out_dir / "config.json").read_text())
    assert cfgj["cfg"]["tactical_speed_input"] is True
    assert cfgj["cfg"]["factored_maneuver"] is False
    assert "man_total_held_at" not in cfgj["loss_weights"]


# ---------- capacity: this is a STRUCTURE change, not a capacity change -------

def test_factored_preset_is_not_a_capacity_change():
    """The A/B must not be confounded by parameter count. MEASURED at build, not
    estimated: the whole D-TAC1 delta is a few thousand parameters."""
    with torch.device("meta"):
        base = param_breakdown(RefCModel(refc_config()))
        fact = param_breakdown(RefCModel(refc_factored_config()))
    delta = abs(fact["total"] - base["total"])
    assert delta < 0.0005 * base["total"], (base["total"], fact["total"], delta)
    assert 90_000_000 < fact["total"] < 130_000_000
    cfg = refc_factored_config()
    assert (cfg.factored_maneuver, cfg.tactical_speed_input) == (True, True)
    assert cfg.man_prior_tau == 1.0


# ---------- the probe script (E-A1 / E-A2), end to end ------------------------

def test_tactical_probe_runs_end_to_end(tmp_path):
    """The pre-registered probe must be a ONE-COMMAND run before it is asked of
    a real checkpoint on a remote box — a script that only works on the eval pod
    is a script nobody can reproduce. Synthetic epcache, smoke config, CPU."""
    import json

    import refc_tactical_probe as probe

    # a val epcache in the on-disk contract taniteval/lan_probe read
    cfg = refc_smoke_config()
    val = tmp_path / "val"
    val.mkdir()
    g = torch.Generator().manual_seed(4)
    specs = [(0.0, 0.0), (0.5, 0.0), (0.0, -3.0), (0.0, 3.0), (-0.5, -2.0)]
    for i, (yr, ac) in enumerate(specs):
        T = 90
        poses = _poses(T, v0=6.0 + 2 * i, yaw_rate=yr, accel=ac)
        lat, lon = tac.window_factored_labels(
            poses[:1].expand(1, 4).clone(), poses[1:21][None], horizon=20)
        torch.save({"frames_u8": (torch.rand(T, cfg.encoder.in_channels, 64, 64,
                                             generator=g) * 255).to(torch.uint8),
                    "poses": poses,
                    "episode_id": f"ep{i:03d}",
                    "maneuvers": torch.zeros(T, dtype=torch.long)},
                   val / f"ep_{i:05d}.pt")

    torch.manual_seed(0)
    model = RefCModel(cfg)
    ck = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "step": 7}, ck)

    out = tmp_path / "probe.json"
    res = probe.main(["--ckpt", str(ck), "--val-dir", str(val), "--preset",
                      "smoke", "--out", str(out), "--device", "cpu",
                      "--stride", "4", "--batch", "8"])
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["E_A1_counterfactual_decode"]["n_windows"] > 20
    assert saved["E_A1_counterfactual_decode"]["n_episodes"] == len(specs)
    # every block the pre-registration reads must be present
    for k in ("control_label_source", "control_shuffled",
              "E_A1_counterfactual_decode", "E_A2_input_probe", "VERDICT"):
        assert k in saved, k
    a1 = saved["E_A1_counterfactual_decode"]
    assert set(a1["decode_factored_raw"]["per_class"]) == set(tac.LON_CLASSES)
    assert "estimator" in a1["ci_lon_active_correct"]         # named, always
    assert a1["ci_lon_active_correct"]["estimator"] == \
        "episode_cluster_bootstrap"
    assert res["VERDICT"]["reading"] in (
        "READOUT-limited (F2+F3 sufficient, part recoverable with no retrain)",
        "INPUT-limited (F1 required; F2 alone would be a null result)",
        "INDETERMINATE — see the pre-registration's tie-break")


def test_probe_refuses_a_post_dtac1_checkpoint(tmp_path):
    """The probe reads the 5-WAY head. Pointing it at a factored checkpoint must
    fail loud, not quietly measure a random-init head."""
    import refc_tactical_probe as probe
    from tanitad.refs import refc as _refc

    monkey = _refc.refc_smoke_config
    cfg = _factored_smoke()
    ck = tmp_path / "ck.pt"
    torch.save({"model": RefCModel(cfg).state_dict()}, ck)
    orig = _refc.refc_smoke_config
    try:
        _refc.refc_smoke_config = lambda: _factored_smoke()
        with pytest.raises(SystemExit, match="5-WAY"):
            probe.build_model(ck, "smoke", "cpu")
    finally:
        _refc.refc_smoke_config = orig
    assert _refc.refc_smoke_config is monkey
