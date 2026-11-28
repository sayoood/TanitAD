"""refcv6 perception branch — the guard set.

⛔ **Every guard here is proven by MUTATION, not inspection.** Each
``test_MUT_*`` re-introduces the exact defect the guard exists to catch and
asserts the check goes RED; a guard that only ever runs against correct code has
never been shown to be able to fail.

The mutation table is reproduced in
``TanitAD Research Lab/Architecture & Inference/Research/
2026-09-16-refcv6-perception/RESULT.md``.

Real-data checks (the 135 SAM3 eval maps, the obstacle cuboids, the orientation
guard) live in ``test_refcv6_perception_realdata.py`` and skip honestly when the
artifacts are not on the box.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tanitad.data.bev_raster import GRID_DEFAULT, _cell_centers
from tanitad.data.lift_orientation import mirror_grid, rank_auc, sign_test_p
from tanitad.data.perception_targets import (
    MIN_MAP_COVERAGE, MapCoverageTooLow, collate_box_targets,
)
from tanitad.data.semantic_map_gt import CHANNELS, N_CHANNELS, raw_frame_index
from tanitad.models.agent_slots import (
    SLOT_SLICES, SLOT_WIDTH, AgentSlotDecoder, match_slots, slot_set_loss,
    targets_from_join,
)
from tanitad.models.bev_encoder import (
    BEVEncoder, BEVEncoderConfig, BEVMapBranch, FRAC_SUM_TOL, MAP_GRID_HW,
    map_metrics, map_soft_ce,
)
from tanitad.models.box3d_head import (
    SLOT3D_SLICES, SLOT3D_WIDTH, Box3DMemory, Box3DSlotDecoder, box3d_ap,
    box3d_set_loss, random_ap_base_rate, zh_targets,
)
from tanitad.models.refc_bev_coupling import (
    BEVCouplingConfig, BEVWaypointSampler, waypoints_to_bev_grid,
)

X_CELLS, Y_CELLS = MAP_GRID_HW


# =========================================================================== #
# 1. the BEV encoder / MAP head                                               #
# =========================================================================== #
def test_encoder_keeps_the_label_grid_exactly():
    enc = BEVEncoder(BEVEncoderConfig(d_in=8, d_model=16, d_out=16,
                                      dilations=(1, 2), norm_groups=4))
    out = enc(torch.randn(2, 8, X_CELLS, Y_CELLS))
    assert tuple(out.shape) == (2, 16, X_CELLS, Y_CELLS)
    assert MAP_GRID_HW == GRID_DEFAULT.shape == (120, 64)


def test_MUT_a_resampling_layer_is_refused():
    """MUTATION: the defect is any layer that changes the BEV grid — a pool, a
    transpose conv, or a stride-2 conv. Each one makes the prediction and the
    label live on different cells, which a mIoU cannot show."""
    enc = BEVEncoder(BEVEncoderConfig(d_in=4, d_model=8, d_out=8,
                                      dilations=(1,), norm_groups=4))
    for bad in (torch.nn.MaxPool2d(2), torch.nn.ConvTranspose2d(8, 8, 2, 2),
                torch.nn.Conv2d(8, 8, 3, stride=2, padding=1),
                torch.nn.Conv2d(8, 8, 3, padding=0)):          # wrong padding
        enc.blocks.append(bad)
        with pytest.raises(RuntimeError):
            enc._assert_stride_one()
        enc.blocks = enc.blocks[:-1]
    enc._assert_stride_one()                                   # green again


def _soft_target(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(b, N_CHANNELS, X_CELLS, Y_CELLS, generator=g)
    p = p / p.sum(1, keepdim=True)
    seen = torch.rand(b, X_CELLS, Y_CELLS, generator=g) > 0.3
    return p, seen


def test_map_soft_ce_is_the_soft_cross_entropy_and_counts_seen_cells():
    p, seen = _soft_target()
    logits = torch.zeros(2, N_CHANNELS, X_CELLS, Y_CELLS)
    out = map_soft_ce(logits, p, seen)
    # uniform logits -> every term is log(9), exactly
    assert out["n_cells"] == int(seen.sum())
    assert float(out["loss"]) == pytest.approx(math.log(N_CHANNELS), abs=1e-5)
    assert float(out["per_class"].sum()) == pytest.approx(float(out["loss"]), abs=1e-5)


def test_MUT_unseen_cells_leak_into_the_map_loss():
    """MUTATION: drop the ``seen`` mask and supervise every cell.

    ⛔ The defect stated EXACTLY: with the mask, an unseen cell receives
    **exactly zero gradient**; without it, it receives gradient like any other.
    A cell the camera never saw carries no map information — the SAM3 GT labels
    it ``not seen`` with fraction ~1 — so a head trained without the mask spends
    capacity learning the FOV, and the map term stops being about the map.

    ⚠️ Two weaker versions of this test were written first and both are recorded
    because each would have passed a build with no mask at all:

    * *"the two losses differ by more than 1e-3"* on RANDOM logits — they differ
      by **6.6e-4**, because random logits are ~log(9) everywhere;
    * *"the unmasked loss prefers a degenerate `not seen` predictor"* — it does
      not: MEASURED 4.9795 (degenerate) vs 4.4464 (good), so the ranking never
      inverts and the assertion was simply false.

    The gradient statement needs no threshold and no seed.
    """
    p, seen = _soft_target()
    p = p.clone()
    not_seen = ~seen
    p = torch.where(not_seen.unsqueeze(1), torch.zeros_like(p), p)
    p[:, N_CHANNELS - 1] = torch.where(not_seen, torch.ones_like(p[:, 0]),
                                       p[:, N_CHANNELS - 1])
    p = p / p.sum(1, keepdim=True)
    assert 0.2 < float(not_seen.float().mean()) < 0.5   # the defect needs a real
    #                                                     unseen share to exist

    def grad_of(mask):
        z = torch.zeros(p.shape, requires_grad=True)
        map_soft_ce(z, p, mask)["loss"].backward()
        return z.grad

    g_masked = grad_of(seen)
    g_mutated = grad_of(torch.ones_like(seen))
    per_cell_masked = g_masked.abs().sum(1)             # [B, X, Y]
    per_cell_mutated = g_mutated.abs().sum(1)

    # WITH the mask: every unseen cell gets EXACTLY zero, every seen cell does not
    assert float(per_cell_masked[not_seen].abs().max()) == 0.0
    assert float(per_cell_masked[seen].min()) > 0.0
    # WITHOUT it: the unseen cells are being trained on
    assert float(per_cell_mutated[not_seen].min()) > 0.0, (
        "the mutation changed nothing — the mask is not load-bearing")
    # and the counts say which cells were scored, unambiguously
    assert map_soft_ce(torch.zeros_like(p), p, torch.ones_like(seen))["n_cells"]         == seen.numel()
    assert map_soft_ce(torch.zeros_like(p), p, seen)["n_cells"] == int(seen.sum())


def test_MUT_a_target_that_is_not_a_distribution_is_refused():
    """MUTATION: hand the loss an un-normalised target (e.g. logits, or counts,
    or a channel subset). Without the sum check it trains happily on a target
    that is not a distribution."""
    p, seen = _soft_target()
    # the realistic defect: a channel is dropped (e.g. `not seen` "renormalised
    # away"), so the remaining mass no longer sums to 1 — and every value is
    # still a legal fraction, so a range check cannot see it
    bad = p.clone()
    bad[:, N_CHANNELS - 1] = 0.0
    assert float(bad.min()) >= 0.0 and float(bad.max()) <= 1.0
    with pytest.raises(ValueError, match="fractions summing to 1"):
        map_soft_ce(torch.zeros_like(p), bad, seen)
    # the mutation — turning the check off — lets it through, which is the point
    out = map_soft_ce(torch.zeros_like(p), bad, seen, check_sum=False)
    assert torch.isfinite(out["loss"])
    # a target outside [0, 1] altogether is refused by the range check
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        map_soft_ce(torch.zeros_like(p), p * 2.0, seen)
    # within the uint8 quantisation bound it is accepted
    ok = p * (1.0 + FRAC_SUM_TOL * 0.5)
    map_soft_ce(torch.zeros_like(p), ok, seen)


def test_map_soft_ce_refuses_a_float_mask_and_an_empty_batch_is_counted():
    p, seen = _soft_target()
    with pytest.raises(ValueError, match="seen must be bool"):
        map_soft_ce(torch.zeros_like(p), p, seen.float())
    none = torch.zeros_like(seen)
    out = map_soft_ce(torch.zeros_like(p), p, none)
    assert out["n_cells"] == 0 and float(out["loss"]) == 0.0


def test_map_metrics_are_per_class_with_frequencies():
    p, seen = _soft_target()
    logits = torch.zeros(2, N_CHANNELS, X_CELLS, Y_CELLS)
    logits.scatter_(1, p.argmax(1, keepdim=True), 10.0)        # perfect argmax
    m = map_metrics(logits, p, seen)
    assert m["acc"] == pytest.approx(1.0)
    assert len(m["iou"]) == N_CHANNELS == len(CHANNELS)
    assert float(m["freq_label"].sum()) == pytest.approx(1.0, abs=1e-9)


def test_map_branch_param_count_is_reported():
    br = BEVMapBranch()
    assert br.n_params == br.encoder.n_params + br.head.n_params
    assert br.encoder.cfg.receptive_field_cells == 1 + 2 * (1 + 2 + 4 + 8)


# =========================================================================== #
# 2. the 3-D box head                                                         #
# =========================================================================== #
def test_the_2d_decode_contract_is_intact():
    """Every 2-D field keeps its offset, and the 2-D decode is bit-identical."""
    for name, sl in SLOT_SLICES.items():
        assert SLOT3D_SLICES[name] == sl
    assert SLOT3D_WIDTH == SLOT_WIDTH + 2
    torch.manual_seed(0)
    base = AgentSlotDecoder(d_memory=32, n_memory=16, n_queries=7, d_model=32,
                            depth=1, n_heads=4, enforce_band=False)
    d3 = Box3DSlotDecoder(d_memory=32, n_memory=16, n_queries=7, d_model=32,
                          depth=1, n_heads=4, enforce_band=False)
    raw2 = torch.randn(3, 7, SLOT_WIDTH)
    raw3 = torch.cat([raw2, torch.randn(3, 7, 2)], dim=-1)
    a, b = base.decode(raw2), d3.decode(raw3)
    for k in ("presence_logit", "cls_logits", "box", "yaw_vec", "yaw", "rates",
              "occ_logit"):
        assert torch.equal(a[k], b[k]), f"3-D widening changed the 2-D key {k!r}"
    assert b["box3d"].shape == (3, 7, 7)
    assert torch.equal(b["box3d"][..., 0], b["box"][..., 0])
    assert torch.equal(b["box3d"][..., 5], b["h"])
    assert bool((b["h"] > 0).all()), "a height must decode positive"


def test_MUT_interleaving_the_new_fields_breaks_the_contract():
    """MUTATION: insert ``cz``/``h`` in the MIDDLE of the field list instead of
    appending. Every later slice shifts, so a 2-D checkpoint's head columns
    silently change meaning — and nothing about the shapes complains."""
    from tanitad.models.agent_slots import SLOT_FIELDS
    i = [n for n, _ in SLOT_FIELDS].index("yaw_sin")
    mutated = SLOT_FIELDS[:i] + (("cz", 1), ("h", 1)) + SLOT_FIELDS[i:]
    sl, off = {}, 0
    for n, w in mutated:
        sl[n] = slice(off, off + w)
        off += w
    moved = [k for k in SLOT_SLICES if sl[k] != SLOT_SLICES[k]]
    assert moved, "the mutation did not move anything — it is not the defect"
    # the import-time assertion in box3d_head is exactly this comparison
    with pytest.raises(RuntimeError, match="no longer mean what they meant"):
        for k, v in SLOT_SLICES.items():
            if sl[k] != v:
                raise RuntimeError(
                    f"3-D widening moved the 2-D field {k!r} from {v} to "
                    f"{sl[k]}: a 2-D checkpoint's head columns would no "
                    f"longer mean what they meant")


def test_memory_refuses_the_stride_32_map():
    """⛔ `image_hw` is REQUIRED (PI 2026-09-16) and the halved map is diagnosed
    BY NAME. Both geometries are covered in `test_refcv6_geometry_agnostic.py`;
    this pins the behaviour at the 640 cache's shape."""
    with pytest.raises(TypeError):
        Box3DMemory(d_image=12, d_bev=8, d_model=16)       # no image_hw
    hw = (16, 40)
    mem = Box3DMemory(d_image=12, d_bev=8, d_model=16, image_hw=hw)
    bev = torch.randn(2, 8, X_CELLS, Y_CELLS)
    assert mem(torch.randn(2, 12, *hw), bev).shape == (2, mem.n_tokens, 16)
    with pytest.raises(ValueError, match="STRIDE-32 map"):
        mem(torch.randn(2, 12, hw[0] // 2, hw[1] // 2), bev)


def _toy_batch(n_agents=6, n_q=12, seed=0, device=None):
    g = torch.Generator().manual_seed(seed)
    a = torch.zeros(n_agents, 6, dtype=torch.float64)
    a[:, 0] = torch.rand(n_agents, generator=g, dtype=torch.float64) * 50 + 5
    a[:, 1] = (torch.rand(n_agents, generator=g, dtype=torch.float64) - 0.5) * 24
    a[:, 2] = (torch.rand(n_agents, generator=g, dtype=torch.float64) - 0.5) * 2
    a[:, 3], a[:, 4] = 4.5, 1.9
    a[:, 5] = 0.0
    tgt = targets_from_join(a.numpy(), classes=["automobile"] * n_agents)
    cz = torch.rand(1, n_agents, generator=g) * 1.5 + 0.4
    h = torch.rand(1, n_agents, generator=g) * 1.0 + 1.2
    return tgt, cz, h


def test_no_zh_labels_is_the_2d_loss_exactly():
    """⛔ With ``zh_mask`` all False the total must be EXACTLY the 2-D total and
    the counts must say ``z = 0`` — never a zero-filled label."""
    tgt, _, _ = _toy_batch()
    t3 = zh_targets(tgt)                       # no cz/h given
    torch.manual_seed(1)
    d3 = Box3DSlotDecoder(d_memory=16, n_memory=8, n_queries=12, d_model=32,
                          depth=1, n_heads=4, enforce_band=False)
    pred = d3(torch.randn(1, 8, 16))
    m = match_slots(pred, t3)
    two = slot_set_loss(pred, t3, match=m)
    # ⛔ `visible_filter=False` HERE IS THE SCOPE, NOT A WORKAROUND. This test pins one
    # thing: the z/h terms add EXACTLY nothing when there is no 3-D label. Both sides
    # must therefore see the same target set, and `m` was built over the unfiltered one.
    # Since 2026-09-23 `box3d_set_loss` REFUSES a supplied match with the filter on,
    # precisely so this pairing has to be stated rather than assumed.
    three = box3d_set_loss(pred, t3, match=m, visible_filter=False)
    assert three["n"]["z"] == 0 and three["n"]["h"] == 0
    assert float(three["loss_z"]) == 0.0 and float(three["loss_h"]) == 0.0
    assert torch.equal(three["total"], two["total"])


def test_MUT_zero_filling_a_missing_height_biases_the_loss():
    """MUTATION: fill the missing z/h with 0 and mark them valid, instead of
    masking. The head is then taught that an unlabelled agent is 0 m tall and
    sits on the road plane — a real, learnable, wrong fact."""
    tgt, cz, h = _toy_batch()
    torch.manual_seed(1)
    d3 = Box3DSlotDecoder(d_memory=16, n_memory=8, n_queries=12, d_model=32,
                          depth=1, n_heads=4, enforce_band=False)
    pred = d3(torch.randn(1, 8, 16))
    masked = box3d_set_loss(pred, zh_targets(tgt))
    zero_filled = box3d_set_loss(pred, zh_targets(
        tgt, torch.zeros_like(cz), torch.zeros_like(h)))
    assert zero_filled["n"]["z"] > 0 and masked["n"]["z"] == 0
    assert float(zero_filled["total"].detach()) != pytest.approx(
        float(masked["total"].detach()), rel=1e-6)


def test_zh_terms_are_metres_and_reach_the_prediction():
    tgt, cz, h = _toy_batch()
    t3 = zh_targets(tgt, cz, h)
    torch.manual_seed(2)
    d3 = Box3DSlotDecoder(d_memory=16, n_memory=8, n_queries=12, d_model=32,
                          depth=1, n_heads=4, enforce_band=False)
    pred = d3(torch.randn(1, 8, 16))
    out = box3d_set_loss(pred, t3)
    assert out["n"]["z"] == out["n"]["matched"] > 0
    out["total"].backward()
    assert d3.head.weight.grad[SLOT3D_SLICES["cz"]].abs().sum() > 0
    assert d3.head.weight.grad[SLOT3D_SLICES["h"]].abs().sum() > 0


def test_matcher_is_unchanged_by_the_3d_widening():
    """The assignment must not move when 3-D is switched on — otherwise an
    ablation compares two different matchings, not two heads."""
    tgt, cz, h = _toy_batch()
    torch.manual_seed(3)
    d3 = Box3DSlotDecoder(d_memory=16, n_memory=8, n_queries=12, d_model=32,
                          depth=1, n_heads=4, enforce_band=False)
    pred = d3(torch.randn(1, 8, 16))
    m2 = match_slots(pred, tgt)
    m3 = match_slots(pred, zh_targets(tgt, cz, h))
    for a, b in zip(m2["rows"], m3["rows"]):
        assert torch.equal(a, b)
    for a, b in zip(m2["cols"], m3["cols"]):
        assert torch.equal(a, b)


# ---- the AP self-test that reads KNOWN values ----------------------------- #
def _perfect_pred(tgt, cz, h):
    n = int(tgt["box"].shape[1])
    return {
        "box": tgt["box"].clone(),
        "box3d": torch.stack([tgt["box"][..., 0], tgt["box"][..., 1], cz,
                              tgt["box"][..., 2], tgt["box"][..., 3], h,
                              tgt["yaw"]], dim=-1),
        "presence_logit": torch.full((1, n), 5.0),
        "cls_logits": F.one_hot(tgt["cls"].clamp_min(0), 10).float() * 10.0,
    }


def test_AP_of_perfect_predictions_is_exactly_one():
    tgt, cz, h = _toy_batch(n_agents=8)
    ap = box3d_ap(_perfect_pred(tgt, cz, h), zh_targets(tgt, cz, h),
                  dist_thresh_m=0.5)
    assert ap["n_gt"] == 8
    assert ap["ap"] == 1.0, f"perfect predictions scored AP {ap['ap']!r}, not 1.0"


def test_AP_of_random_predictions_is_near_the_closed_form_base_rate():
    """⛔ The comparison value is computed in CLOSED FORM
    (:func:`random_ap_base_rate`), not read off the same run."""
    tgt, cz, h = _toy_batch(n_agents=8, seed=5)
    t3 = zh_targets(tgt, cz, h)
    g = torch.Generator().manual_seed(11)
    n_q, trials = 40, 24
    extent = (55.0, 24.0, 0.0)              # the sampling box, BEV plane
    thresh = 2.0
    aps = []
    for t in range(trials):
        gg = torch.Generator().manual_seed(100 + t)
        box = torch.stack([
            torch.rand(1, n_q, generator=gg) * extent[0] + 5.0,
            (torch.rand(1, n_q, generator=gg) - 0.5) * extent[1],
            torch.full((1, n_q), 4.5), torch.full((1, n_q), 1.9)], dim=-1)
        pred = {"box": box,
                "presence_logit": torch.randn(1, n_q, generator=gg),
                "cls_logits": torch.randn(1, n_q, 10, generator=gg)}
        aps.append(box3d_ap(pred, t3, dist_thresh_m=thresh, use_z=False)["ap"])
    got = float(np.mean(aps))
    base = random_ap_base_rate(8, n_q, thresh, extent)
    assert 0.2 * base <= got <= 1.6 * base, (
        f"random AP {got:.4f} is not near the closed-form base rate {base:.4f}")
    assert got < 0.5, "random predictions must not look like a working detector"
    _ = g


def test_random_ap_base_rate_is_monotone_and_bounded():
    b = random_ap_base_rate(8, 40, 2.0, (55.0, 24.0, 0.0))
    assert 0.0 < b < 1.0
    assert random_ap_base_rate(16, 40, 2.0, (55.0, 24.0, 0.0)) > b   # more GT
    assert random_ap_base_rate(8, 40, 4.0, (55.0, 24.0, 0.0)) > b    # looser


# =========================================================================== #
# 3. the planner couplings                                                    #
# =========================================================================== #
def test_waypoints_to_bev_grid_lands_on_cell_centres():
    """A waypoint AT a cell centre must sample that cell exactly."""
    xc, yc = _cell_centers(GRID_DEFAULT)
    bev = torch.arange(X_CELLS * Y_CELLS, dtype=torch.float32).reshape(
        1, 1, X_CELLS, Y_CELLS)
    for i, j in ((0, 0), (5, 3), (60, 32), (119, 63), (17, 40)):
        wp = torch.tensor([[[[xc[i], yc[j]]]]], dtype=torch.float32)
        g = waypoints_to_bev_grid(wp, GRID_DEFAULT)
        got = F.grid_sample(bev, g.reshape(1, 1, 1, 2), mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        assert float(got) == pytest.approx(float(i * Y_CELLS + j), abs=1e-3), \
            f"cell ({i},{j}) sampled {float(got)}"


def test_MUT_transposing_the_grid_components_samples_the_wrong_cell():
    """MUTATION: emit ``(g_row, g_col)`` instead of ``(g_col, g_row)`` —
    DiffusionDrive's own ``[..., [1, 0]]`` swap, omitted. Shape-preserving,
    silent, and wrong everywhere off the diagonal."""
    xc, yc = _cell_centers(GRID_DEFAULT)
    bev = torch.arange(X_CELLS * Y_CELLS, dtype=torch.float32).reshape(
        1, 1, X_CELLS, Y_CELLS)
    i, j = 17, 40
    wp = torch.tensor([[[[xc[i], yc[j]]]]], dtype=torch.float32)
    good = waypoints_to_bev_grid(wp, GRID_DEFAULT)
    bad = good.flip(-1)
    v_good = float(F.grid_sample(bev, good.reshape(1, 1, 1, 2), mode="bilinear",
                                 padding_mode="zeros", align_corners=False))
    v_bad = float(F.grid_sample(bev, bad.reshape(1, 1, 1, 2), mode="bilinear",
                                padding_mode="zeros", align_corners=False))
    assert v_good == pytest.approx(i * Y_CELLS + j, abs=1e-3)
    assert abs(v_bad - v_good) > 1.0, "the transposition cost nothing"


def test_MUT_mirroring_the_bev_grid_swaps_left_and_right():
    """MUTATION (the ``R-2026-09-08-wpa-mirror`` class, in BEV): negate the
    lateral component. A waypoint 8 m LEFT then reads the cell 8 m RIGHT."""
    bev = torch.zeros(1, 1, X_CELLS, Y_CELLS)
    bev[0, 0, :, :Y_CELLS // 2] = 1.0            # mark everything on the RIGHT
    wp = torch.tensor([[[[20.0, 8.0]]]])         # 8 m LEFT -> value 0
    g = waypoints_to_bev_grid(wp, GRID_DEFAULT)
    v = float(F.grid_sample(bev, g.reshape(1, 1, 1, 2), mode="bilinear",
                            padding_mode="zeros", align_corners=False))
    vm = float(F.grid_sample(bev, mirror_grid(g).reshape(1, 1, 1, 2),
                             mode="bilinear", padding_mode="zeros",
                             align_corners=False))
    assert v == pytest.approx(0.0) and vm == pytest.approx(1.0)


def test_coupling_is_bit_identical_on_64_fixed_windows_when_gated_off():
    """⛔ The brief's requirement: *both behind zero-init gates that reproduce
    today's planner BIT-IDENTICALLY when off (prove it on 64 fixed windows)*."""
    torch.manual_seed(7)
    cfg = BEVCouplingConfig(d_model=32, d_bev=8, n_points=6)
    s = BEVWaypointSampler(cfg)
    assert float(s.gate.detach()) == 0.0, "the gate is not zero-init"
    for w in range(64):
        g = torch.Generator().manual_seed(1000 + w)
        q = torch.randn(2, 5, 32, generator=g)
        wp = torch.randn(2, 5, 6, 2, generator=g) * 10.0
        bev = torch.randn(2, 8, X_CELLS, Y_CELLS, generator=g)
        assert torch.equal(s(q, wp, bev), q), f"window {w} is not bit-identical"
        assert torch.equal(s(q, wp, None), q)         # disabled path
    s.gate.data.fill_(1e-6)                           # the MUTATION
    g = torch.Generator().manual_seed(1000)
    q = torch.randn(2, 5, 32, generator=g)
    wp = torch.randn(2, 5, 6, 2, generator=g) * 10.0
    bev = torch.randn(2, 8, X_CELLS, Y_CELLS, generator=g)
    assert not torch.equal(s(q, wp, bev), q), (
        "a non-zero gate changed nothing — the coupling is dead, not gated")


def test_the_gate_is_gated_not_dead():
    cfg = BEVCouplingConfig(d_model=16, d_bev=4, n_points=4)
    s = BEVWaypointSampler(cfg)
    q = torch.randn(1, 3, 16, requires_grad=True)
    wp = torch.randn(1, 3, 4, 2) * 5.0
    bev = torch.randn(1, 4, X_CELLS, Y_CELLS)
    s(q, wp, bev).sum().backward()
    assert s.gate.grad is not None and float(s.gate.grad.abs()) > 0.0


def test_learned_offsets_are_flagged_and_off_by_default():
    """⚠️ The released DiffusionDrive has NO learned offsets
    (``ddv2_src/blocks.py:80-108``); ours must say so."""
    d = BEVWaypointSampler().cfg.as_dict()
    assert d["learned_offsets"] is False
    assert d["provenance"] == "ddv2-faithful"
    assert "blocks.py:80-108" in d["released_source"]
    ext = BEVWaypointSampler(BEVCouplingConfig(learned_offsets=True))
    assert ext.is_extension and ext.provenance() == "tanitad-extension"
    assert ext.cfg.as_dict()["provenance"] == "tanitad-extension"
    # zero-init offsets: the extension starts AT the faithful sample positions
    q = torch.randn(1, 3, 256)
    wp = torch.randn(1, 3, 8, 2) * 5.0
    assert torch.equal(ext.sample_positions(q, wp), wp)


def test_coupling_refuses_a_waypoint_count_it_was_not_built_for():
    s = BEVWaypointSampler(BEVCouplingConfig(d_model=16, d_bev=4, n_points=4))
    with pytest.raises(ValueError, match="Not resized"):
        s(torch.randn(1, 3, 16), torch.randn(1, 3, 7, 2),
          torch.randn(1, 4, X_CELLS, Y_CELLS))


# =========================================================================== #
# 4. the loader hooks                                                         #
# =========================================================================== #
class _FakeGT:
    def __init__(self, n_frames=201):
        self.n_frames = n_frames


class _FakeStore:
    """A store whose clips exist or not, for the coverage mutation."""

    def __init__(self, present, n_frames=201):
        self.present = set(present)
        self.n_frames = n_frames
        self.layout_of = {}

    def open(self, clip_id):
        if str(clip_id) not in self.present:
            import errno
            raise FileNotFoundError(errno.ENOENT, "no GT", str(clip_id))
        return _FakeGT(self.n_frames)


def test_coverage_refuses_a_run_under_the_floor():
    from tanitad.data.perception_targets import require_map_coverage
    clips = [f"clip{i}" for i in range(10)]
    windows = [(c, w) for c in clips for w in range(10)]
    ok = require_map_coverage(windows, _FakeStore(clips))
    assert ok["verdict"] == "PASS" and ok["frac_ok"] == 1.0
    assert MIN_MAP_COVERAGE == 0.90
    # 8 of 10 clips -> 0.80 coverage: REFUSED
    with pytest.raises(MapCoverageTooLow, match="below the floor"):
        require_map_coverage(windows, _FakeStore(clips[:8]))
    rep = require_map_coverage(windows, _FakeStore(clips[:8]), raise_on_low=False)
    assert rep["verdict"] == "FAIL" and rep["frac_ok"] == pytest.approx(0.8)
    assert rep["n_no_file"] == 20
    # exactly at the floor passes; one window under does not
    assert require_map_coverage(windows, _FakeStore(clips[:9]))["frac_ok"] == 0.9


def test_MUT_out_of_range_frames_count_against_coverage():
    """MUTATION: treat a window whose LABEL FRAME does not exist as covered.
    The clip has a file, so a file-level coverage check passes — and the trainer
    then indexes past the end of the GT."""
    from tanitad.data.perception_targets import require_map_coverage
    clips = ["a", "b"]
    windows = [(c, w) for c in clips for w in (0, 50, 100, 199, 500)]
    rep = require_map_coverage(windows, _FakeStore(clips, n_frames=201),
                               raise_on_low=False)
    # raw = window + 2, so windows 199 and 500 -> frames 201 and 502, both past
    # the clip's 201 frames: 2 per clip, 2 clips
    assert rep["n_frame_out_of_range"] == 4
    assert rep["n_no_file"] == 0                 # the files are all there
    assert rep["frac_ok"] == pytest.approx(0.6)
    assert rep["verdict"] == "FAIL"


def test_raw_frame_index_is_the_window_plus_n_stack_minus_one():
    assert int(raw_frame_index(0, 3)) == 2
    assert list(raw_frame_index([0, 5, 10], 3)) == [2, 7, 12]
    assert list(raw_frame_index([0, 5], 1)) == [0, 5]
    with pytest.raises(ValueError):
        raw_frame_index(0, 0)


def test_collate_box_targets_pads_and_carries_the_masks():
    t1, cz1, h1 = _toy_batch(n_agents=3, seed=1)
    t2, cz2, h2 = _toy_batch(n_agents=6, seed=2)
    b = collate_box_targets([zh_targets(t1, cz1, h1), zh_targets(t2, cz2, h2)])
    assert b["box"].shape == (2, 6, 4) and b["valid"].shape == (2, 6)
    assert b["valid"][0].tolist() == [True] * 3 + [False] * 3
    assert b["zh_mask"][0].tolist() == [True] * 3 + [False] * 3
    assert float(b["occ"][0, 5]) == -1.0          # padding is NOT "not occluded"
    plain = collate_box_targets([t1, t2])         # no 3-D labels anywhere
    assert not bool(plain["zh_mask"].any())


# =========================================================================== #
# 5. the orientation guard's own instrument                                   #
# =========================================================================== #
def test_rank_auc_is_exact_on_known_cases():
    assert rank_auc(np.arange(60), np.arange(60) >= 30) == 1.0
    assert rank_auc(-np.arange(60.0), np.arange(60) >= 30) == 0.0
    assert rank_auc(np.zeros(60), np.arange(60) >= 30) == 0.5      # all ties
    assert math.isnan(rank_auc(np.arange(10), np.arange(10) >= 5))  # too few


def test_sign_test_p_is_exact():
    assert sign_test_p(10, 10) == pytest.approx(1 / 1024)
    assert sign_test_p(5, 10) == pytest.approx(0.623046875)
    assert sign_test_p(84, 115) < 1e-6
    assert sign_test_p(60, 115) > 0.05          # a 52 % win rate proves nothing


def test_mirror_grid_is_an_involution_and_a_copy():
    g = torch.randn(2, 3, 4, 2)
    assert torch.equal(mirror_grid(mirror_grid(g)), g)
    m = mirror_grid(g)
    m[..., 0] += 1.0
    assert not torch.equal(m, mirror_grid(g))   # it returned a copy


def test_orientation_probe_refuses_overlapping_fit_and_eval_clips():
    from tanitad.data.lift_orientation import orientation_probe
    X = np.random.RandomState(0).randn(200, 4)
    y = X[:, 0] > 0
    s = ("abc123def456", X, X[:, ::-1].copy(), y)
    with pytest.raises(ValueError, match="BOTH the fit and the eval set"):
        orientation_probe([s], [s])
