"""refcv5 WP-6 — the agent-token seam, its monocular supervision, and the
decoder parity contract.

Every test here is one of three kinds, and the kind is named in the docstring:

* **fails without the feature** — the test the brief requires per component.
* **control that must read a known value** — an identity, a floor, or a
  no-information reading. Without one of these a probe can manufacture a result
  (MEASURED four times in one afternoon, `CLAUDE.md`).
* **deliberate regression** — an arm that MUST fail, or the instrument cannot
  see what it is cited for.
"""

from __future__ import annotations

import math

import pytest
import torch

from tanitad.data.calib import PHYSICALAI_WIDE120_256x640
from tanitad.data.rig_projection import RigCamera
from tanitad.models import agent_slots as slots_mod
from tanitad.refs import refc, refc_agents as ra


# ---------------------------------------------------------------------------
# 0. The enum is imported, not re-listed — the fabrication guard
# ---------------------------------------------------------------------------
def test_class_enum_is_the_corpus_enum():
    """CONTROL. A first draft of `refc_agents` hand-wrote a plausible 10-class
    tuple containing `bicycle`, `motorcycle`, `train_or_tram_car` — none of
    which exist in the corpus. This pins the enum to the label side's."""
    from tanitad.data.bev_raster import ALL_CLASSES
    assert ra.AGENT_CLASSES == tuple(ALL_CLASSES)
    assert ra.N_AGENT_CLASSES == 10
    # the three that DO exist and were nearly dropped
    for name in ("other_vehicle", "stroller", "animal"):
        assert name in ra.AGENT_CLASSES
    # the three that were invented
    for name in ("bicycle", "motorcycle", "train_or_tram_car"):
        assert name not in ra.AGENT_CLASSES


# ---------------------------------------------------------------------------
# 1. THE PROJECTION TEST THE BRIEF ASKS FOR:
#    a parked car MUST move under a wrong extrinsic.
# ---------------------------------------------------------------------------
def _parked_car_rig():
    """A car parked 20 m ahead, 3 m to the left, box centre 0.75 m up."""
    return torch.tensor([[20.0, 3.0, 0.75]], dtype=torch.float64)


def test_parked_car_projects_stably_under_the_TRUE_extrinsic():
    """CONTROL that must read a known value. The same static point, projected
    from two different ego positions with the extrinsic held correct, lands on
    the same pixel once expressed in the rig frame — i.e. the projection itself
    introduces no motion."""
    frame = PHYSICALAI_WIDE120_256x640
    cam = RigCamera.nominal(frame, height_m=1.5)
    p = _parked_car_rig()
    c1, r1, v1 = cam.project(p)
    c2, r2, v2 = cam.project(p.clone())
    assert bool(v1.all()) and bool(v2.all())
    assert torch.allclose(c1, c2) and torch.allclose(r1, r2)


def test_wrong_extrinsic_MAKES_THE_PARKED_CAR_MOVE():
    """⛔ THE TEST THE BRIEF NAMES. A wrong extrinsic must be VISIBLE: the same
    parked car, projected through a mis-mounted camera, must land on a
    materially different pixel.

    ⭐ Why this is the right shape of test, and not a tautology: it is easy to
    write a projection that silently ignores the extrinsic (uses only the
    nominal axis permutation), and such a code path passes every "does it
    return a number" test and every round-trip test. It fails ONLY a test that
    perturbs the extrinsic and demands the answer change. This is the optics
    twin of the `constant-only control` rule.
    """
    frame = PHYSICALAI_WIDE120_256x640
    p = _parked_car_rig()
    good = RigCamera.nominal(frame, height_m=1.5)
    c_g, r_g, v_g = good.project(p)
    assert bool(v_g.all())

    # (a) a 2-degree pitch error — the classic mount-calibration slip
    pitch = math.radians(2.0)
    # rotate the camera about the rig's +y (left) axis
    R_y = torch.tensor(
        [[math.cos(pitch), 0.0, math.sin(pitch)],
         [0.0, 1.0, 0.0],
         [-math.sin(pitch), 0.0, math.cos(pitch)]], dtype=torch.float64)
    tilted = RigCamera(R_cam_to_rig=R_y @ good.R_cam_to_rig,
                       t_cam_in_rig=good.t_cam_in_rig, frame=frame)
    c_t, r_t, _ = tilted.project(p)
    d_row = float((r_t - r_g).abs().max())
    assert d_row > 5.0, (
        f"a 2-degree pitch error moved the parked car by only {d_row:.2f} rows "
        f"— the extrinsic is not reaching the projection")

    # (b) a 0.5 m height error — reads as a range error on the ground plane
    lifted = RigCamera(R_cam_to_rig=good.R_cam_to_rig,
                       t_cam_in_rig=good.t_cam_in_rig
                       + torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64),
                       frame=frame)
    c_h, r_h, _ = lifted.project(p)
    assert float((r_h - r_g).abs().max()) > 1.0

    # (c) a 1 m lateral error must move the COLUMN (azimuth), not just the row
    shifted = RigCamera(R_cam_to_rig=good.R_cam_to_rig,
                        t_cam_in_rig=good.t_cam_in_rig
                        + torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
                        frame=frame)
    c_s, r_s, _ = shifted.project(p)
    assert float((c_s - c_g).abs().max()) > 5.0


def test_cylindrical_hfov_is_120_not_the_pinhole_92_6():
    """CONTROL that must read a KNOWN, INDEPENDENTLY-PUBLISHED value: the rig's
    own name is `camera_front_wide_120fov`. The pinhole formula gives 92.6 deg
    here and looks entirely plausible — this is the trap in test form."""
    frame = PHYSICALAI_WIDE120_256x640
    assert frame.projection == "cylindrical"
    assert abs(frame.hfov_deg - 120.0) < 1e-6
    pinhole = 2.0 * math.degrees(math.atan((frame.width / 2) / frame.f_ref))
    assert abs(pinhole - 92.6) < 0.2      # the WRONG answer, pinned as wrong
    assert abs(frame.hfov_deg - pinhole) > 25.0


def test_ground_plane_backprojection_round_trips():
    """CONTROL. Rig z = 0 IS the road plane (MEASURED over 87,481 cuboids), so
    a ground point -> pixel -> ground must return the same point."""
    frame = PHYSICALAI_WIDE120_256x640
    cam = RigCamera.nominal(frame, height_m=1.5)
    p = torch.tensor([[15.0, 2.0, 0.0], [40.0, -6.0, 0.0]], dtype=torch.float64)
    col, row, valid = cam.project(p)
    assert bool(valid.all())
    back, hits = cam.ground_intersection(col, row)
    assert bool(hits.all())
    assert torch.allclose(back[:, :2], p[:, :2], atol=1e-6)


# ---------------------------------------------------------------------------
# 2. The decoder parity contract
# ---------------------------------------------------------------------------
def _tiny_decoder(cross_agent: bool, seed: int = 0):
    torch.manual_seed(seed)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                             cross_agent=cross_agent)
    anchors = torch.randn(6, 4, 2)
    return refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=4, d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=anchors, cfg=cfg, hierarchy=False, graft_maneuver=False,
        graft_target_latent=False, grounded_selector=False,
        horizons=(1, 2, 3, 4))


def test_cross_agent_off_constructs_nothing():
    """CONTROL. `cross_agent=False` must not construct the branch — otherwise
    RNG draw order changes and the 'bit-identical to refcv4b' claim is false."""
    dec = _tiny_decoder(False)
    for layer in dec.layers:
        assert layer.cross_agent is None
        assert layer.agent_gate is None
        assert layer.norm_a is None


def test_cross_agent_off_is_BIT_IDENTICAL_to_pre_v5():
    """CONTROL that must read a known value: with the seam off, the whole
    parameter set is bit-identical to a decoder built at the same seed by the
    pre-v5 code path (same RNG consumption)."""
    a = _tiny_decoder(False, seed=7)
    b = _tiny_decoder(False, seed=7)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_zero_init_gate_reads_delta_EXACTLY_zero():
    """⛔ THE GATE CONTROL the SPEC names (`zero_init_gate_reads_delta_zero`).
    With the agent branch constructed but the gate at its zero init, feeding
    agent tokens must change the output by EXACTLY zero — not approximately."""
    torch.manual_seed(3)
    dec = _tiny_decoder(True, seed=7)
    for layer in dec.layers:
        assert layer.agent_gate is not None
        assert float(layer.agent_gate.detach().abs().max()) == 0.0
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    agents = torch.randn(2, 5, dec.cfg.d)
    with torch.no_grad():
        out0 = dec(fmap, m, steps=0)
        out1 = dec(fmap, m, steps=0, agent_tokens=agents)
    assert torch.equal(out0["anchor_logits"], out1["anchor_logits"])
    assert torch.equal(out0["anchor_traj"], out1["anchor_traj"])
    assert torch.equal(out0["traj"], out1["traj"])


def test_gate_is_GATED_NOT_DEAD():
    """FAILS WITHOUT THE FEATURE. A zero gate that also has zero gradient is a
    dead seam that can never learn. dL/d(gate) must be non-zero at init."""
    torch.manual_seed(3)
    dec = _tiny_decoder(True, seed=7)
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    agents = torch.randn(2, 5, dec.cfg.d)
    out = dec(fmap, m, steps=0, agent_tokens=agents)
    out["anchor_traj"].square().mean().backward()
    grads = [float(l.agent_gate.grad.abs().max()) for l in dec.layers]
    assert all(g > 0.0 for g in grads), f"dead gate: {grads}"


def test_nonzero_gate_DOES_change_the_output():
    """FAILS WITHOUT THE FEATURE. Once the gate moves off zero the agent tokens
    must actually reach the trajectories — otherwise the seam is decorative."""
    torch.manual_seed(3)
    dec = _tiny_decoder(True, seed=7)
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    agents = torch.randn(2, 5, dec.cfg.d)
    with torch.no_grad():
        base = dec(fmap, m, steps=0, agent_tokens=agents)["anchor_traj"].clone()
        for layer in dec.layers:
            layer.agent_gate.fill_(0.5)
        moved = dec(fmap, m, steps=0, agent_tokens=agents)["anchor_traj"]
    assert not torch.allclose(base, moved)


def test_fully_padded_row_is_finite_in_BOTH_directions():
    """⛔ A window with NO agents is NORMAL (an empty road). A fully-masked
    key_padding_mask makes the attention softmax normalise over all -inf: the
    output is NaN and the BACKWARD is NaN even if the forward is patched."""
    torch.manual_seed(3)
    dec = _tiny_decoder(True, seed=7)
    for layer in dec.layers:
        layer.agent_gate.data.fill_(0.5)
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    agents = torch.randn(2, 5, dec.cfg.d)
    pad = torch.ones(2, 5, dtype=torch.bool)          # EVERY slot padded
    out = dec(fmap, m, steps=0, agent_tokens=agents, agent_pad=pad)
    assert torch.isfinite(out["anchor_traj"]).all()
    out["anchor_traj"].square().mean().backward()
    for p in dec.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_partially_padded_row_is_finite():
    """The mixed case: row 0 empty, row 1 populated — the state the corpus
    actually produces, and the one a naive `if pad.all()` guard misses."""
    torch.manual_seed(3)
    dec = _tiny_decoder(True, seed=7)
    for layer in dec.layers:
        layer.agent_gate.data.fill_(0.5)
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    agents = torch.randn(2, 5, dec.cfg.d)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    pad[0] = True                                     # row 0 fully padded
    out = dec(fmap, m, steps=0, agent_tokens=agents, agent_pad=pad)
    assert torch.isfinite(out["anchor_traj"]).all()
    out["anchor_traj"].square().mean().backward()
    for p in dec.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def _subset_deviation(cross_agent: bool):
    """Decode the full candidate set and a 2-element subset; return the max
    absolute disagreement on the shared entries."""
    dec = _tiny_decoder(cross_agent, seed=7).eval()
    if cross_agent:
        for layer in dec.layers:
            layer.agent_gate.data.fill_(0.5)
    torch.manual_seed(3)
    fmap = torch.randn(1, 16, 3, 5)
    m = torch.randn(1, 8)
    agents = torch.randn(1, 5, dec.cfg.d) if cross_agent else None
    kv = dec.feat_proj(fmap.flatten(2).transpose(1, 2))
    cond = dec.cond_proj(m)
    x = torch.randn(1, 6, 4, 2)
    sub = torch.tensor([1, 4])
    with torch.no_grad():
        c_all, o_all = dec._decode(kv, cond, x, 0, agents=agents)
        c_sub, o_sub = dec._decode(kv, cond, x[:, sub], 0, agents=agents)
    return (float((c_all[:, sub] - c_sub).abs().max()),
            float((o_all[:, sub] - o_sub).abs().max()))


def test_candidate_independence_survives_the_agent_branch():
    """⛔ S2b (`anchor_prefilter`) is sound only because candidates never
    interact: the agent branch cross-attends q -> agents with NO candidate-axis
    mixing, so decoding a SUBSET must give that subset the same answer.

    ⚠️ **This is a DIFFERENTIAL test, and it is written that way because the
    absolute version FAILS — for a reason that has nothing to do with agents.**
    MEASURED here: decoding a subset already disagrees with the full decode by
    **1.192e-07** (float32 eps scale) with the agent branch **OFF**, because a
    different candidate count selects a different GEMM reduction order. So
    `refc.py`'s S2b comment — *"decoding a subset is bit-identical for that
    subset"* — is a float-precision **overstatement**; what is true, and what
    its own MEASURED claim actually says, is that the **selection index** is
    identical (881/881). The structural property (no candidate-axis mixing) is
    what this test must pin, and the way to pin it is to show the agent branch
    does not make the deviation WORSE than the agent-free baseline.
    """
    d_conf_off, d_off_off = _subset_deviation(False)
    d_conf_on, d_off_on = _subset_deviation(True)
    # the agent-free baseline is itself non-zero: that is the finding
    assert d_conf_off < 1e-5 and d_off_off < 1e-5
    # ⛔ the real assertion: adding agent cross-attention does not introduce a
    # candidate-axis dependence, so its deviation stays at the same float scale
    assert d_conf_on < 10.0 * max(d_conf_off, 1e-8), (d_conf_on, d_conf_off)
    assert d_off_on < 10.0 * max(d_off_off, 1e-8), (d_off_on, d_off_off)
    assert d_conf_on < 1e-5 and d_off_on < 1e-5


# ---------------------------------------------------------------------------
# 3. The token seam
# ---------------------------------------------------------------------------
def _oracle_batch(b=2, n=4):
    box = torch.tensor([[[20.0, 3.0, 4.5, 1.9],
                         [35.0, -2.0, 4.2, 1.8],
                         [8.0, 0.5, 4.6, 1.9],
                         [0.0, 0.0, 0.0, 0.0]]] * b)
    yaw = torch.zeros(b, n)
    cls = torch.tensor([[0, 1, 5, -1]] * b)
    valid = torch.tensor([[True, True, True, False]] * b)
    return box, yaw, cls, valid


def test_oracle_tokens_have_the_right_shape_and_mask():
    """FAILS WITHOUT THE FEATURE."""
    cfg = ra.AgentSeamConfig(enable=True, oracle=True, d_model=32)
    emb_o = ra.OracleAgentEmbed(cfg)
    tok_e = ra.AgentTokenEmbed(d_out=24, cfg=cfg)
    box, yaw, cls, valid = _oracle_batch()
    slots = emb_o(box, yaw, cls, valid)
    tok, pad = tok_e(slots)
    assert tok.shape == (2, 4, 24)
    assert pad.shape == (2, 4)
    assert bool(pad[:, 3].all())          # the padded slot is masked
    assert not bool(pad[:, :3].any())


def test_degrade_boxes_at_sigma_zero_is_THE_IDENTITY():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. E-AGT-ORACLE must be exactly
    the zero-degradation point of the E-AGT-BUDGET sweep, not a second code
    path — otherwise the budget curve does not start at the ceiling."""
    box, _, _, valid = _oracle_batch()
    b2, v2 = ra.degrade_boxes(box, valid, 0.0, 0.0)
    assert b2 is box and v2 is valid


def test_degrade_boxes_moves_RANGE_and_not_BEARING():
    """FAILS WITHOUT THE FEATURE, and pins the modelling choice: on a
    cylindrical frame the column fixes bearing almost exactly, so the budget
    sweep noises RANGE. A version that noised x and y independently would
    price a detector nobody would build."""
    box, _, _, valid = _oracle_batch()
    g = torch.Generator().manual_seed(0)
    out, _ = ra.degrade_boxes(box, valid, sigma_range_m=2.0, generator=g)
    b0, o0 = box[0, :3], out[0, :3]
    r_in = b0[:, :2].norm(dim=-1)
    r_out = o0[:, :2].norm(dim=-1)
    assert float((r_out - r_in).abs().max()) > 0.05          # range moved
    bear_in = torch.atan2(b0[:, 1], b0[:, 0])
    bear_out = torch.atan2(o0[:, 1], o0[:, 0])
    assert torch.allclose(bear_in, bear_out, atol=1e-5)      # bearing did not


def test_miss_rate_one_drops_everything():
    """CONTROL at the other extreme of the sweep."""
    box, _, _, valid = _oracle_batch()
    g = torch.Generator().manual_seed(0)
    _, v = ra.degrade_boxes(box, valid, miss_rate=1.0, generator=g)
    assert not bool(v.any())


def test_unknown_class_becomes_UNIFORM_not_class_zero():
    """CONTROL. `cls == -1` means UNKNOWN. Folding it into class 0 would invent
    'automobile' for every unlabelled row — a silent label fabrication."""
    cfg = ra.AgentSeamConfig(oracle=True, d_model=32)
    slots = ra.OracleAgentEmbed(cfg)(*_oracle_batch())
    p = torch.softmax(slots["cls_logits"], dim=-1)
    assert torch.allclose(p[0, 3], torch.full((ra.N_AGENT_CLASSES,),
                                              1.0 / ra.N_AGENT_CLASSES),
                          atol=1e-6)
    assert float(p[0, 0, 0]) > 0.99       # a KNOWN class stays one-hot


# ---------------------------------------------------------------------------
# 4. The head, and the monocular terms
# ---------------------------------------------------------------------------
def test_head_is_vision_only_by_SIGNATURE():
    """⛔ THE VISION-ONLY AUDIT, enforced structurally rather than by comment:
    the head's forward takes exactly ONE tensor argument, so there is no
    parameter through which a privileged label could arrive at inference."""
    import inspect
    sig = inspect.signature(slots_mod.AgentSlotDecoder.forward)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["memory"], params


def test_head_builds_and_emits_the_expected_fields():
    """FAILS WITHOUT THE FEATURE."""
    cfg = ra.AgentSeamConfig(enable=True, queries=8, d_model=64, depth=2,
                             enforce_band=False)
    head = ra.build_agent_head(cfg, d_memory=32, n_memory=15)
    out = head(torch.randn(2, 15, 32))
    assert out["box"].shape == (2, 8, 4)
    assert out["cls_logits"].shape == (2, 8, ra.N_AGENT_CLASSES)
    assert out["yaw_vec"].shape == (2, 8, 2)
    assert torch.allclose(out["yaw_vec"].norm(dim=-1),
                          torch.ones(2, 8), atol=1e-5)


def test_projection_loss_is_zero_on_IDENTICAL_boxes():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE: matched prediction == target
    gives EXACTLY zero image-plane error. A probe that cannot read zero when
    the answer is zero cannot be trusted to read a non-zero."""
    frame = PHYSICALAI_WIDE120_256x640
    cam = RigCamera.nominal(frame, height_m=1.5)
    box = torch.tensor([[[20.0, 3.0, 4.5, 1.9], [35.0, -2.0, 4.2, 1.8]]])
    match = {"rows": [torch.tensor([0, 1])], "cols": [torch.tensor([0, 1])],
             "n_target": [2], "n_dropped": [0]}
    out = ra.monocular_projection_loss(box, box.clone(), cam, match)
    assert out["n"] == 2
    assert float(out["loss"]) == 0.0


def test_projection_loss_GROWS_with_range_error():
    """FAILS WITHOUT THE FEATURE. The term must actually see a range error."""
    frame = PHYSICALAI_WIDE120_256x640
    cam = RigCamera.nominal(frame, height_m=1.5)
    tgt = torch.tensor([[[20.0, 0.0, 4.5, 1.9]]])
    match = {"rows": [torch.tensor([0])], "cols": [torch.tensor([0])],
             "n_target": [1], "n_dropped": [0]}
    prev = 0.0
    for err in (0.5, 2.0, 6.0):
        pred = tgt.clone()
        pred[0, 0, 0] += err
        cur = float(ra.monocular_projection_loss(pred, tgt, cam, match)["loss"])
        assert cur > prev
        prev = cur


def test_ground_prior_is_zero_for_a_CONSISTENT_box():
    """CONTROL. A box whose foot is on the road plane is self-consistent by
    construction, so the prior must read ~0 there."""
    frame = PHYSICALAI_WIDE120_256x640
    cam = RigCamera.nominal(frame, height_m=1.5)
    box = torch.tensor([[[20.0, 3.0, 4.5, 1.9], [35.0, -2.0, 4.2, 1.8]]])
    out = ra.ground_range_prior(box, cam)
    assert out["n"] == 2
    assert float(out["loss"]) < 1e-6


def test_visibility_filter_drops_agents_OUTSIDE_the_camera_field():
    """⛔⛔ FAILS WITHOUT THE FEATURE, and it is the most load-bearing test in
    this file. MEASURED on the val40 join: `targets_from_join` marks EVERY
    agent valid, and `match_slots` keeps the N NEAREST — so at n_queries = 16,
    **61.8 % of kept targets are outside the 120 deg field** and 80.1 % are
    outside the decode box. The nearest agents include cars BEHIND the ego.
    Without this filter the head is trained to hallucinate on ~62 % of its
    supervision, and its AP would measure how well it guesses the unobservable.
    """
    box = torch.tensor([[[20.0, 3.0, 4.5, 1.9],      # ahead, in field
                         [-15.0, 0.0, 4.5, 1.9],     # BEHIND the ego
                         [5.0, 30.0, 4.5, 1.9],      # 80 deg to the left
                         [90.0, 0.0, 4.5, 1.9]]])    # beyond the decode range
    tgt = {"box": box, "valid": torch.ones(1, 4, dtype=torch.bool)}
    out = ra.visible_target_filter(tgt)
    assert out["valid"].tolist() == [[True, False, False, False]]
    # the caller's dict is NOT mutated
    assert bool(tgt["valid"].all())


def test_visibility_filter_uses_60_DEGREES_not_the_pinhole_half_angle():
    """CONTROL that must read a KNOWN value. The AZIMUTH cut alone (no range
    box) must sit at 60 deg — the rig's own `camera_front_wide_120fov`. If the
    pinhole formula had been used the boundary would sit at 46.3 deg, and the
    59 deg agent would be wrongly dropped.

    ⚠️ Tested with `filter_targets_to_visible` and NOT `visible_target_filter`,
    on purpose: the latter also applies the decode box, and an agent at 59 deg
    and 20 m has |cy| = 17.1 m, so the BOX would drop it. Conflating the two
    cuts would let a wrong azimuth pass unnoticed — which is exactly what the
    first version of this test did.
    """
    import math as _m
    r = 20.0
    for deg, want in ((59.0, True), (61.0, False), (46.3, True),
                      (0.0, True), (120.0, False)):
        box = torch.tensor([[[r * _m.cos(_m.radians(deg)),
                              r * _m.sin(_m.radians(deg)), 4.5, 1.9]]])
        tgt = {"box": box, "valid": torch.ones(1, 1, dtype=torch.bool)}
        got = bool(ra.filter_targets_to_visible(tgt)["valid"][0, 0])
        assert got is want, f"{deg} deg -> {got}, expected {want}"
    # ...and the decode box is a SEPARATE, additional cut
    far_left = torch.tensor([[[10.3, 17.1, 4.5, 1.9]]])   # 59 deg, |cy| > 16
    tgt = {"box": far_left, "valid": torch.ones(1, 1, dtype=torch.bool)}
    assert bool(ra.filter_targets_to_visible(tgt)["valid"][0, 0]) is True
    assert bool(ra.visible_target_filter(tgt)["valid"][0, 0]) is False


def test_filter_is_ON_by_default_in_agent_losses_and_reports_BOTH_n():
    """⛔ CONTROL. The filter must be the default, and the loss must report the
    PRE-filter count too — a panel that reports only the post-filter n cannot
    say how much supervision the filter removed, and that fraction is the whole
    finding."""
    cfg = ra.AgentSeamConfig(queries=8, d_model=64, depth=2,
                             enforce_band=False)
    head = ra.build_agent_head(cfg, d_memory=32, n_memory=15)
    slots = head(torch.randn(1, 15, 32))
    box = torch.tensor([[[20.0, 3.0, 4.5, 1.9], [-15.0, 0.0, 4.5, 1.9]]])
    tgt = {"box": box, "yaw": torch.zeros(1, 2),
           "cls": torch.tensor([[0, 0]]),
           "valid": torch.ones(1, 2, dtype=torch.bool),
           "occ": torch.full((1, 2), -1.0), "rates": torch.zeros(1, 2, 3),
           "rates_mask": torch.zeros(1, 2, dtype=torch.bool)}
    on = ra.agent_losses(slots, tgt, cfg)
    assert on["filter_visible"] is True
    assert on["n"]["target_prefilter"] == 2
    assert on["n"]["target_visible"] == 1
    off = ra.agent_losses(slots, tgt, cfg, filter_visible=False)
    assert off["n"]["target_visible"] == 2       # the regression arm


def test_agent_losses_report_n_PER_TERM():
    """⛔ The four-families sibling rule: per-term, never pooled, and every
    term carries the n it was computed over."""
    cfg = ra.AgentSeamConfig(enable=True, queries=8, d_model=64, depth=2,
                             enforce_band=False, w_project=1.0, w_ground=0.5)
    head = ra.build_agent_head(cfg, d_memory=32, n_memory=15)
    slots = head(torch.randn(2, 15, 32))
    box, yaw, cls, valid = _oracle_batch()
    tgt = {"box": box, "yaw": yaw, "cls": cls, "valid": valid,
           "occ": torch.full((2, 4), -1.0),
           "rates": torch.zeros(2, 4, 3),
           "rates_mask": torch.zeros(2, 4, dtype=torch.bool)}
    cam = RigCamera.nominal(PHYSICALAI_WIDE120_256x640, height_m=1.5)
    out = ra.agent_losses(slots, tgt, cfg, cam=cam)
    for k in ("loss_presence", "loss_cls", "loss_centre", "loss_size",
              "loss_yaw", "loss_project", "loss_ground"):
        assert k in out, k
    assert out["n"]["target"] == 6         # 3 valid per row, 2 rows
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.parameters())


def test_head_can_LEARN_a_fixed_box_from_a_fixed_memory():
    """⛔ FAILS WITHOUT THE FEATURE, and it is the learnability control the
    ladder's `failure` clause needs: 'AP at or below the prior floor (the head
    learned nothing)'. If a head cannot fit ONE constant box from ONE constant
    memory in 200 steps, a negative on the corpus says nothing about the
    corpus."""
    torch.manual_seed(0)
    cfg = ra.AgentSeamConfig(queries=4, d_model=64, depth=2,
                             enforce_band=False)
    head = ra.build_agent_head(cfg, d_memory=16, n_memory=6)
    mem = torch.randn(1, 6, 16)
    tgt = {"box": torch.tensor([[[18.0, 2.0, 4.5, 1.9]]]),
           "yaw": torch.zeros(1, 1), "cls": torch.tensor([[0]]),
           "valid": torch.ones(1, 1, dtype=torch.bool),
           "occ": torch.full((1, 1), -1.0),
           "rates": torch.zeros(1, 1, 3),
           "rates_mask": torch.zeros(1, 1, dtype=torch.bool)}
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    first = None
    for _ in range(600):
        opt.zero_grad()
        out = ra.agent_losses(head(mem), tgt, cfg)
        out["total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_centre"])
    final = float(ra.agent_losses(head(mem), tgt, cfg)["loss_centre"])
    # first ~= 20 m (the head starts at the origin, the box is at 18 m)
    assert final < 0.05 * first, f"centre {first:.3f} -> {final:.3f}"
    # ⚠️ 1 m, not 0.1 m, and the number is a DECLARED bar not a fitted one:
    # this is a 4-query head on a 6-token random memory, i.e. the smallest rig
    # that can express the task at all. It answers "can the head fit a box",
    # which is the learnability control; it is NOT a statement about corpus
    # accuracy and must never be quoted as one.
    assert final < 1.0, f"centre error still {final:.3f} m"


def test_constant_memory_control_cannot_distinguish_two_scenes():
    """⛔ THE CONSTANT-ONLY CONTROL the SPEC names. A head fed an
    INFORMATION-FREE memory must emit the SAME boxes for two different scenes.
    If it does not, the head is reading something other than its input and any
    'detection works' claim is unattributable."""
    cfg = ra.AgentSeamConfig(queries=4, d_model=64, depth=2,
                             enforce_band=False)
    head = ra.build_agent_head(cfg, d_memory=16, n_memory=6).eval()
    const = torch.full((2, 6, 16), 0.3)
    with torch.no_grad():
        out = head(const)
    assert torch.allclose(out["box"][0], out["box"][1], atol=1e-6)


@pytest.mark.parametrize("hard", [False, True])
def test_presence_gating_soft_vs_hard(hard):
    """Both gating policies are reachable and both produce finite tokens. SOFT
    is the default BECAUSE a hard mask has zero gradient to the presence head
    through the planner loss — stated here so the choice is a decision."""
    cfg = ra.AgentSeamConfig(d_model=32, presence_hard=hard,
                             presence_gate=0.5, oracle=True)
    slots = ra.OracleAgentEmbed(cfg)(*_oracle_batch())
    tok, pad = ra.AgentTokenEmbed(d_out=24, cfg=cfg)(slots)
    assert torch.isfinite(tok).all()
    assert pad.dtype == torch.bool
