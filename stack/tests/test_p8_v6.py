"""P8-on-v6 CPU tests — the geometry seam, the refusals, and the port itself.

⛔ WHY THESE EXIST. P8 decodes a **Cartesian ego-frame** raster; v6's readout is
an **image-plane** grid over a 120° cylindrical field. Pointing a probe at the
wrong projection is the C9/C14 family — an instrument structurally unable to
report the answer it is cited for — and it is only catchable BEFORE the GPU run
if the geometry is arithmetic rather than folklore. So:

  * the field mask and the readout-column mapping are checked against
    INDEPENDENT analytic loops, not against themselves;
  * the mapping REFUSES a readout whose pool does not tile (overlapping
    adaptive bins ⇒ a cell belongs to two columns ⇒ an index would be fiction),
    and the refusal is exercised on a REAL geometry (the 176x624 sub-frame);
  * the metric functions REFUSE a raster of the wrong shape instead of letting
    torch broadcast a number nobody specified;
  * ``p8_latents_ex`` is run end to end against a real (tiny) ``V6Stack``
    through ``V6ProbeTrunk`` — the port's actual claim, not a mock of it.

Everything here is CPU and needs no corpus, no checkpoint and no join file.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig  # noqa: E402
from tanitad.data.bev_raster import (BEVGrid, GRID_DEFAULT,  # noqa: E402
                                     cell_azimuth_rad, cell_centers_xy,
                                     fov_census, fov_mask,
                                     readout_column_index,
                                     readout_row_ranges_m)
from tanitad.eval.v6_probe_trunk import V6ProbeTrunk  # noqa: E402
from tanitad.models.v6 import (ReadoutConfig, V6Config,  # noqa: E402
                               V6Stack, readout_grid_ranges)
from train_p8_occupancy import (BEVOccupancyHead, PARAM_BAND,  # noqa: E402
                                assert_raster_shape, build_fov_mask,
                                hold_future_actions, iou_at_05, iou_at_tau,
                                p8_gate_dict, p8_geometry_report,
                                p8_latents, p8_latents_ex, soft_dice_loss)

NX, NY = GRID_DEFAULT.shape                                   # (120, 64)
HALF_120 = math.radians(60.0)
HALF_117 = math.radians(58.5)


# ===========================================================================
# the field mask — against an independent loop, then against the census
# ===========================================================================
def test_fov_mask_matches_an_independent_analytic_loop():
    """Written as explicit loops over the same cell centers `rasterize` uses,
    NOT via the vectorised function under test."""
    exp = np.zeros((NX, NY), dtype=bool)
    for i in range(NX):
        xc = (i + 0.5) * 0.5
        for j in range(NY):
            yc = -16.0 + (j + 0.5) * 0.5
            exp[i, j] = abs(math.atan2(yc, xc)) <= HALF_120
    assert np.array_equal(fov_mask(GRID_DEFAULT, HALF_120), exp)


def test_cell_centers_and_azimuth_agree_with_the_raster_grid():
    x, y = cell_centers_xy(GRID_DEFAULT)
    assert x.shape == y.shape == (NX, NY)
    assert x[0, 0] == pytest.approx(0.25) and x[-1, 0] == pytest.approx(59.75)
    assert y[0, 0] == pytest.approx(-15.75) and y[0, -1] == pytest.approx(15.75)
    assert np.allclose(cell_azimuth_rad(GRID_DEFAULT), np.arctan2(y, x))


def test_v6_field_census_is_the_measured_590_of_7680():
    """⭐ THE FINDING, pinned. At v6's 120° field, 7.682 % of the pre-registered
    P8 target lies outside the camera entirely — ALL of it in the near band."""
    c = fov_census(half_angle_rad=HALF_120, n_cols=4,
                   projection="cylindrical", token_w=40, readout_rows=4)
    assert c["total_cells"] == 7680
    assert c["out_of_fov_cells"] == 590
    assert c["out_of_fov_frac"] == pytest.approx(0.076823, abs=1e-6)
    # every out-of-field cell is near: the far field is fully covered
    assert c["out_of_fov_max_x_m"] == pytest.approx(8.75)
    assert c["first_fully_visible_row"] == 18
    assert c["first_fully_visible_row_x_m"] == pytest.approx(9.25)


def test_out_of_field_cells_are_concentrated_in_the_near_band():
    """The mismatch is not spread thin — it eats half the band where
    distance-keeping lives, which is why it must be reported rather than
    averaged away."""
    m = fov_mask(GRID_DEFAULT, HALF_120)
    x, _y = cell_centers_xy(GRID_DEFAULT)
    near = x < 9.0
    assert int((~m).sum()) == 590
    assert int((~m & ~near).sum()) == 0, "no out-of-field cell beyond 9 m"
    frac_of_band = float((~m & near).sum()) / float(near.sum())
    assert frac_of_band == pytest.approx(0.512, abs=5e-3)


def test_legacy_square_pinhole_frame_is_far_worse_than_v6():
    """⭐ The mismatch PRE-DATES v6 and v6 REDUCES it. On CANONICAL_256
    (256x256 pinhole f_ref 266 — every pre-2026-07-27 number) 27.68 % of the
    target is unobservable and nothing below ~33 m is fully covered."""
    half = math.atan((256 / 2.0) / 266.0)
    c = fov_census(half_angle_rad=half, n_cols=4, projection="pinhole",
                   token_w=16, readout_rows=4)
    assert c["out_of_fov_cells"] == 2126
    assert c["out_of_fov_frac"] == pytest.approx(0.276823, abs=1e-6)
    assert c["first_fully_visible_row"] == 65
    v6 = fov_census(half_angle_rad=HALF_120, n_cols=4,
                    projection="cylindrical", token_w=40)
    assert c["out_of_fov_frac"] > v6["out_of_fov_frac"]


def test_fov_mask_refuses_a_degenerate_half_angle():
    for bad in (0.0, -0.1, math.pi, 4.0):
        with pytest.raises(ValueError):
            fov_mask(GRID_DEFAULT, bad)


# ===========================================================================
# the readout-column mapping — and its refusal
# ===========================================================================
def test_cylindrical_columns_are_equal_azimuth_wedges():
    """256x640 / patch 16 -> 40 token cols -> 4 readout cols = 160 px each; a
    cylindrical frame is LINEAR in azimuth, so each column is exactly 30°."""
    col = readout_column_index(GRID_DEFAULT, HALF_120, 4, "cylindrical",
                               token_w=40)
    az = np.degrees(cell_azimuth_rad(GRID_DEFAULT))
    for j, (lo, hi) in enumerate(((-60, -30), (-30, 0), (0, 30), (30, 60))):
        sel = col == j
        assert sel.any()
        assert az[sel].min() >= lo - 1e-9
        assert az[sel].max() < hi + 1e-9
    assert set(np.unique(col).tolist()) == {-1, 0, 1, 2, 3}


def test_outer_wedges_hold_little_of_the_target_and_none_of_its_far_field():
    """⭐ Half of v6's readout WIDTH covers 15.44 % of the target and reaches
    only 27.25 m; the target's far half lives entirely in the two inner
    columns. A cell-aware port that spent capacity uniformly would spend half
    of it on 15 % of the question."""
    col = readout_column_index(GRID_DEFAULT, HALF_120, 4, "cylindrical",
                               token_w=40)
    x, _y = cell_centers_xy(GRID_DEFAULT)
    outer = (col == 0) | (col == 3)
    assert int(outer.sum()) == 1186
    assert float(outer.sum()) / col.size == pytest.approx(0.154427, abs=1e-6)
    assert float(x[outer].max()) == pytest.approx(27.25)
    far = x >= 30.0
    assert sorted(set(col[far].ravel().tolist())) == [1, 2]


def test_pinhole_columns_are_not_the_cylindrical_ones():
    """The projection is READ, never assumed: tan-spaced edges put different
    cells in different columns than azimuth-spaced ones."""
    cyl = readout_column_index(GRID_DEFAULT, HALF_120, 4, "cylindrical",
                               token_w=40)
    pin_half = math.radians(50.0)          # a pinhole-expressible field
    pin = readout_column_index(GRID_DEFAULT, pin_half, 4, "pinhole",
                               token_w=40)
    assert not np.array_equal(cyl, pin)


def test_readout_column_index_refuses_a_non_tiling_pool():
    """⛔ THE REFUSAL, on a REAL geometry: the deployed 176x624 sub-frame gives
    39 token columns onto 4 readout columns. SpatialGridReadout then uses
    AdaptiveAvgPool2d, whose bins OVERLAP — an index would be fiction."""
    with pytest.raises(ValueError, match="NOT EXACT"):
        readout_column_index(GRID_DEFAULT, HALF_117, 4, "cylindrical",
                             token_w=39)
    # …and the exact case with the same field is fine
    ok = readout_column_index(GRID_DEFAULT, HALF_117, 4, "cylindrical",
                              token_w=40)
    assert ok.shape == (NX, NY)


def test_readout_column_index_refuses_bad_projection_and_impossible_pinhole():
    with pytest.raises(ValueError, match="projection"):
        readout_column_index(GRID_DEFAULT, HALF_120, 4, "fisheye")
    # 60° is pinhole-expressible (tan 60 = 1.73); >= 90° is not.
    readout_column_index(GRID_DEFAULT, HALF_120, 4, "pinhole")
    with pytest.raises(ValueError, match="tan diverges"):
        readout_column_index(GRID_DEFAULT, math.radians(95.0), 4, "pinhole")


def test_census_records_the_refusal_instead_of_dying():
    """A refused column mapping must not take the FOV mask down with it — the
    mask does not depend on tiling and stays valid."""
    c = fov_census(half_angle_rad=HALF_117, n_cols=4,
                   projection="cylindrical", token_w=39, readout_rows=4)
    assert c["readout_columns"]["exact"] is False
    assert "NOT EXACT" in c["readout_columns"]["reason"]
    assert c["in_fov_cells"] > 0 and c["out_of_fov_cells"] > 0
    json.dumps(c)                                    # stays serialisable


# ===========================================================================
# the row prior — pinned to v6's own, including its INVERTED order
# ===========================================================================
def test_row_ranges_pinned_to_the_v6_torch_original():
    for gh in (1, 2, 4, 7):
        np.testing.assert_allclose(
            readout_row_ranges_m(gh),
            readout_grid_ranges(gh, 3)[:, 0].numpy(), rtol=1e-6)


def test_row_zero_is_v6s_far_row_and_p8s_near_row():
    """⭐ The orders are OPPOSITE. Aligning the two grids row-for-row without
    flipping maps far onto near."""
    rows = readout_row_ranges_m(4)
    assert rows[0] > rows[-1]                        # v6 row 0 = FAR
    x, _y = cell_centers_xy(GRID_DEFAULT)
    assert x[0, 0] < x[-1, 0]                        # P8 row 0 = NEAR
    assert rows[0] == pytest.approx(80.0)
    assert rows[-1] == pytest.approx(3.0)


def test_row_prior_overhangs_the_target_grid_on_both_ends():
    """v6's declared prior is 3–80 m against a 0–60 m target: 5 % of the target
    is nearer than its near limit, and row 0's 80 m is off the grid entirely."""
    c = fov_census(half_angle_rad=HALF_120, n_cols=4,
                   projection="cylindrical", token_w=40, readout_rows=4)
    r = c["readout_rows"]
    assert r["frac_nearer_than_near_m"] == pytest.approx(0.05)
    assert r["cells_nearer_than_near_m"] == 384
    assert r["prior_far_m_beyond_grid"] is True
    assert "ESTIMATED" in r["_evidence_class"]


# ===========================================================================
# metric refusals — no silent broadcasting
# ===========================================================================
@pytest.mark.parametrize("bad", [(120, 1), (1, 64), (60, 64), (120, 32)])
def test_metrics_refuse_a_grid_they_were_not_built_for(bad):
    """torch would BROADCAST [B,120,1] and [B,1,64] against [B,120,64] and
    return a number — an IoU against a grid nobody specified."""
    logits = torch.zeros(2, NX, NY)
    tgt = torch.zeros(2, *bad)
    for fn in (lambda: iou_at_05(logits, tgt),
               lambda: iou_at_tau(logits, tgt, 0.5),
               lambda: soft_dice_loss(logits, tgt)):
        with pytest.raises(ValueError, match="grid mismatch"):
            fn()


def test_assert_raster_shape_accepts_the_right_grid_and_checks_the_mask():
    logits = torch.zeros(3, NX, NY)
    assert_raster_shape(logits, torch.zeros(3, NX, NY))
    with pytest.raises(ValueError, match="batch mismatch"):
        assert_raster_shape(logits, torch.zeros(2, NX, NY))
    with pytest.raises(ValueError, match="FOV mask"):
        assert_raster_shape(logits, torch.zeros(3, NX, NY),
                            torch.ones(60, 64, dtype=torch.bool))


def test_masked_iou_scores_only_the_kept_cells():
    """Occupancy placed ONLY outside the field: unmasked IoU is a perfect 1.0,
    masked IoU is NaN (empty union) — i.e. the mask changes the answer, which
    is the whole reason it is reported separately."""
    m = torch.from_numpy(fov_mask(GRID_DEFAULT, HALF_120))
    tgt = torch.zeros(1, NX, NY)
    tgt[0][~m] = 1.0
    logits = torch.where(tgt > 0.5, torch.full_like(tgt, 9.0),
                         torch.full_like(tgt, -9.0))
    assert float(iou_at_05(logits, tgt)[0]) == pytest.approx(1.0)
    assert math.isnan(float(iou_at_05(logits, tgt, mask=m)[0]))


def test_mask_does_not_change_a_fully_in_field_case():
    m = torch.from_numpy(fov_mask(GRID_DEFAULT, HALF_120))
    tgt = torch.zeros(1, NX, NY)
    tgt[0, 100, 32] = 1.0                             # x=50.25 m, dead ahead
    logits = torch.where(tgt > 0.5, torch.full_like(tgt, 9.0),
                         torch.full_like(tgt, -9.0))
    assert float(iou_at_tau(logits, tgt, 0.5)) == pytest.approx(1.0)
    assert float(iou_at_tau(logits, tgt, 0.5, mask=m)) == pytest.approx(1.0)


# ===========================================================================
# the gate's cell-set selector
# ===========================================================================
def _row(**kw):
    base = {"iou_enc": 0.4, "n_enc": 10, "iou_pred": 0.36, "n_pred": 10}
    base.update(kw)
    return {10: base}


def test_gate_defaults_to_the_incumbent_all_cells_set():
    g = p8_gate_dict(_row())
    assert g["PASS"] is True and g["gate_a"]["cell_set"] == "all"
    assert g["gate_a"]["ratio"] == pytest.approx(0.9)


def test_gate_can_read_the_in_fov_cell_set():
    per_k = _row(iou_enc_infov=0.5, n_enc_infov=10,
                 iou_pred_infov=0.2, n_pred_infov=10)
    g = p8_gate_dict(per_k, metric="in-fov")
    assert g["gate_a"]["cell_set"] == "in-fov"
    assert g["gate_a"]["ratio"] == pytest.approx(0.4)
    assert g["PASS"] is False                        # 0.4 < 0.8 retention


def test_gate_says_not_computable_when_the_cell_set_was_not_evaluated():
    g = p8_gate_dict(_row(), metric="in-fov")
    assert g["PASS"] is None
    assert "was not evaluated" in g["gate_a"]["reason"]


def test_gate_refuses_an_unknown_cell_set():
    with pytest.raises(ValueError, match="metric must be"):
        p8_gate_dict(_row(), metric="whatever")


# ===========================================================================
# the hold-action control
# ===========================================================================
def test_hold_future_actions_repeats_the_windows_last_action():
    aw = torch.randn(2, 6, 3)
    fa = torch.randn(2, 20, 3)
    held = hold_future_actions(aw, fa)
    assert held.shape == fa.shape
    assert torch.equal(held[:, 0], aw[:, -1])
    assert torch.equal(held[:, -1], aw[:, -1])


def test_hold_future_actions_handles_an_empty_horizon():
    aw = torch.randn(1, 6, 3)
    fa = torch.zeros(1, 0, 3)
    assert hold_future_actions(aw, fa).shape == (1, 0, 3)


# ===========================================================================
# the port itself — p8_latents_ex against a real V6Stack via V6ProbeTrunk
# ===========================================================================
@pytest.fixture(scope="module")
def tiny_v6():
    """A REAL V6Stack, sized so CPU tests stay cheap. 64x128 / patch 16 gives a
    4x8 token grid, which tiles exactly onto the 4x4 readout — so the column
    mapping is exact here too."""
    cfg = V6Config(
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=128,
                              patch_size=16, d_model=64, depth=1, n_heads=4),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=64, depth=1, n_heads=4, window=6,
                                  horizons=(1, 2, 4), action_dim=3,
                                  residual=True),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, f_blocks=1, aux_hidden=32,
        emission_hidden=32, d_plan_feat=32, sigreg_slices=8)
    stack = V6Stack(cfg)
    stack.eval()
    for p in stack.parameters():
        p.requires_grad_(False)
    return stack


def test_probe_trunk_exposes_the_v6_geometry_surface(tiny_v6):
    t = V6ProbeTrunk(tiny_v6)
    assert t.grid_shape == (4, 4)
    assert t.d_readout == 8 and t.n_cells == 16
    assert t.state_dim == 16 * 8 == int(tiny_v6.cfg.d_op)
    assert t.token_grid == (4, 8)
    assert t.in_channels == 9
    assert t.is_v6 is True
    assert t.frame is None, "no frame unless one was resolved — never a default"


def test_probe_trunk_cells_recovers_the_readout_layout(tiny_v6):
    t = V6ProbeTrunk(tiny_v6)
    z = torch.randn(3, t.state_dim)
    assert t.cells(z).shape == (3, t.n_cells, t.d_readout)


def _batch(stack, b=2, k=4):
    cfg = stack.cfg
    w = cfg.predictor.window
    c, h, wd = cfg.encoder.in_channels, cfg.encoder.image_size, \
        cfg.encoder.image_width
    torch.manual_seed(0)
    return {"frames": torch.randn(b, w, c, h, wd),
            "future_frames": torch.randn(b, k, c, h, wd),
            "actions": torch.randn(b, w, 2),
            "future_actions": torch.randn(b, k, 2),
            "pose_last": torch.randn(b, 4).abs()}


def test_p8_latents_runs_against_a_v6_trunk(tiny_v6):
    """⭐ THE PORT'S CLAIM, exercised: the P8 latent collector runs on a V6Stack
    through the adapter, with no v5 parameter names and no translation."""
    t = V6ProbeTrunk(tiny_v6)
    ks = (2, 4)
    b = _batch(tiny_v6, k=max(ks))
    z_t, z_enc, z_hat = p8_latents(t, b, ks, amp_on=False, want_pred=True,
                                   want_enc_k=True)
    assert z_t.shape == (2, t.state_dim)
    assert set(z_enc) == set(z_hat) == set(ks)
    for k in ks:
        assert z_enc[k].shape == (2, t.state_dim)
        assert z_hat[k].shape == (2, t.state_dim)
        assert torch.isfinite(z_hat[k]).all()


def test_p8_latents_ex_adds_the_hold_arm_with_matching_shapes(tiny_v6):
    t = V6ProbeTrunk(tiny_v6)
    ks = (2, 4)
    b = _batch(tiny_v6, k=max(ks))
    _zt, _ze, z_hat, z_hold = p8_latents_ex(t, b, ks, amp_on=False,
                                            want_pred=True, want_enc_k=True,
                                            want_hold=True)
    assert set(z_hold) == set(ks)
    for k in ks:
        assert z_hold[k].shape == z_hat[k].shape
        assert torch.isfinite(z_hold[k]).all()


class _ActionSensitiveTrunk:
    """A trunk whose predictor provably depends on the last action.

    Needed because an UNTRAINED predictor cannot exercise the control at all
    (see the FiLM test below) — so the "the two arms differ" property has to be
    checked against a predictor that is action-sensitive by construction."""

    state_dim = 8
    window = 3

    def __init__(self):
        self.predictor = self

    def encode_window(self, frames):
        b, t = frames.shape[:2]
        return frames.reshape(b, t, -1)[:, :, :self.state_dim].contiguous()

    def __call__(self, states, actions):
        return {1: states[:, -1] + actions[:, -1].sum(-1, keepdim=True)}


def test_hold_arm_diverges_when_the_predictor_reads_actions():
    t = _ActionSensitiveTrunk()
    torch.manual_seed(0)
    b = {"frames": torch.randn(2, t.window, 1, 1, 16),
         "future_frames": torch.randn(2, 4, 1, 1, 16),
         "actions": torch.randn(2, t.window, 2),
         "future_actions": torch.randn(2, 4, 2),
         "pose_last": torch.rand(2, 4)}
    _zt, _ze, z_hat, z_hold = p8_latents_ex(t, b, (4,), amp_on=False,
                                            want_pred=True, want_enc_k=True,
                                            want_hold=True)
    assert not torch.allclose(z_hold[4], z_hat[4], atol=1e-6)


def test_the_hold_control_is_vacuous_on_an_UNTRAINED_trunk(tiny_v6):
    """⚠️ A PROPERTY OF THE CONTROL, pinned so nobody reads a null result as
    evidence. The predictor's action conditioning is FiLM with ZERO-INIT
    (``predictor.py:39``, "same zero-init" for the modern block), so at random
    init the actions have EXACTLY no effect and the true-action and held-action
    rolls coincide. hold_over_pred == 1.0 on an untrained trunk therefore means
    "the conditioning has not trained", NOT "the scene is action-independent".
    The control only discriminates on a trained checkpoint."""
    t = V6ProbeTrunk(tiny_v6)
    b = _batch(tiny_v6, k=4)
    _zt, _ze, z_hat, z_hold = p8_latents_ex(t, b, (4,), amp_on=False,
                                            want_pred=True, want_enc_k=True,
                                            want_hold=True)
    assert torch.allclose(z_hold[4], z_hat[4], atol=1e-6)


def test_hold_arm_is_identical_when_the_future_actions_are_already_held(tiny_v6):
    """The control is a real control: with genuinely held future actions the
    two arms coincide, so a difference between them can only come from the
    action channel."""
    t = V6ProbeTrunk(tiny_v6)
    b = _batch(tiny_v6, k=4)
    b["future_actions"] = b["actions"][:, -1:].expand(-1, 4, -1).contiguous()
    _zt, _ze, z_hat, z_hold = p8_latents_ex(t, b, (4,), amp_on=False,
                                            want_pred=True, want_enc_k=True,
                                            want_hold=True)
    assert torch.allclose(z_hold[4], z_hat[4], atol=1e-6)


def test_p8_latents_ex_hold_arm_is_absent_unless_requested(tiny_v6):
    t = V6ProbeTrunk(tiny_v6)
    b = _batch(tiny_v6, k=4)
    out = p8_latents_ex(t, b, (4,), amp_on=False, want_pred=True,
                        want_enc_k=True)
    assert out[3] == {}


def test_occupancy_head_fits_the_v6_state_and_refuses_another(tiny_v6):
    """The head is built from the trunk's own width; a latent of any other
    width is refused rather than reshaped."""
    t = V6ProbeTrunk(tiny_v6)
    head = BEVOccupancyHead(t.state_dim, enforce_band=False)
    z = torch.randn(2, t.state_dim)
    assert head(z).shape == (2, NX, NY)
    with pytest.raises(ValueError, match="latent dim"):
        head(torch.randn(2, t.state_dim + 1))


def test_occupancy_head_is_in_band_at_the_v6_catalog_width():
    """v6's catalog readout (4x4 cells x d_readout 128) derives d_op = 2048 —
    the flagship width — so the pre-registered ~1M band holds with no
    re-tuning, and the band is ASSERTED rather than hoped."""
    head = BEVOccupancyHead(2048, enforce_band=True)
    assert PARAM_BAND[0] <= head.n_params <= PARAM_BAND[1]
    assert head.n_params == pytest.approx(985_000, rel=0.05)


# ===========================================================================
# the geometry report / mask builder
# ===========================================================================
class _Frame:
    """Minimal CanonicalFrame stand-in (the real one is exercised pod-side)."""

    def __init__(self, h, w, half, projection="cylindrical", f_ref=305.577):
        self.height, self.width, self._half = h, w, half
        self.projection, self.f_ref = projection, f_ref

    def half_angle_x_rad(self):
        return self._half

    @property
    def hfov_deg(self):
        return math.degrees(2 * self._half)

    @property
    def vfov_deg(self):
        return math.degrees(2 * math.atan((self.height / 2.0) / self.f_ref))


def test_geometry_report_without_a_frame_says_so_instead_of_defaulting():
    """An assumed field is a fabricated one — the report must decline."""
    rep = p8_geometry_report(None, generation="v6")
    assert rep["available"] is False and "no CanonicalFrame" in rep["reason"]
    mask, note = build_fov_mask(None)
    assert mask is None and "no frame resolved" in note


def test_geometry_report_carries_the_census_and_the_frame():
    rep = p8_geometry_report(_Frame(256, 640, HALF_120),
                             readout_grid=(4, 4), token_grid=(16, 40),
                             generation="v6")
    assert rep["available"] is True and rep["generation"] == "v6"
    assert rep["out_of_fov_cells"] == 590
    assert rep["readout_columns"]["exact"] is True
    assert rep["frame"]["hfov_deg"] == pytest.approx(120.0)
    assert rep["readout_grid"] == [4, 4] and rep["token_grid"] == [16, 40]
    json.dumps(rep)


def test_build_fov_mask_returns_a_bool_grid_and_a_note():
    mask, note = build_fov_mask(_Frame(256, 640, HALF_120), GRID_DEFAULT)
    assert mask.shape == (NX, NY) and mask.dtype == torch.bool
    assert int(mask.sum()) == 7090
    assert "7090/7680" in note


def test_mask_and_census_agree_on_a_non_default_grid():
    """The mask is derived from the grid it is handed, so a different BEVGrid
    gets a different mask — not the default one silently reused."""
    g = BEVGrid(x_fwd_m=30.0, y_half_m=8.0, cell_m=0.5)
    assert g.shape == (60, 32)
    m = fov_mask(g, HALF_120)
    c = fov_census(grid=g, half_angle_rad=HALF_120, n_cols=4,
                   projection="cylindrical", token_w=40)
    assert m.shape == (60, 32)
    assert c["in_fov_cells"] == int(m.sum())
    assert c["total_cells"] == 60 * 32
