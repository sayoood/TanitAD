"""Pins ``render_refcv6_map_video``'s drivable-IoU rule and its known-value control.

The rule is the trainer's (``refc_v3_train.py:4506-4521`` @ 82c2331): ``>= 0.5`` on BOTH the GT
fraction and the softmax probability, on ``map_seen & map_valid``. Every expected value below is a
LITERAL worked out by hand from the synthetic grid -- never an expression over the code under test --
and the control ships with mutation arms that must go RED.

Runs without the model, the data or the ``tanitad`` package: the tool is loaded by path and only its
pure functions are called.
"""
from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.join(os.path.dirname(_HERE), "tools", "render_refcv6_map_video.py")


def _load():
    spec = importlib.util.spec_from_file_location("render_refcv6_map_video_under_test", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()
C, DRV = 9, 1


def _frac_two_channel(drv):
    """4x4 drivable fractions; the rest of each cell's mass goes to channel 0."""
    drv = np.asarray(drv, np.float32)
    f = np.zeros((1, C, 4, 4), np.float32)
    f[0, DRV] = drv
    f[0, 0] = 1.0 - drv
    return torch.from_numpy(f)


def _logits(spec):
    """4x4 of codes -> [1, 9, 4, 4] logits with an EXACT p(drivable):
    'P' ~1.0 (positive), 'N' ~0.0 (negative), 'H' exactly 0.5, 'h' just below 0.5 (0.49998)."""
    z = np.zeros((1, C, 4, 4), np.float32)
    for i in range(4):
        for j in range(4):
            s = spec[i][j]
            if s == "P":
                z[0, DRV, i, j] = 20.0
            elif s == "N":
                z[0, 0, i, j] = 20.0
            elif s == "H":                       # two equal logits, the other seven -> exp underflow
                z[0, :, i, j] = -1.0e4
                z[0, 0, i, j] = 0.0
                z[0, DRV, i, j] = 0.0
            elif s == "h":
                z[0, :, i, j] = -1.0e4
                z[0, 0, i, j] = 0.0
                z[0, DRV, i, j] = 0.0
                z[0, 2, i, j] = -10.0            # a third class takes e^-10 of the mass
    return torch.from_numpy(z)


GT_DRV = [[0.8, 0.8, 0.2, 0.2],
          [0.8, 0.8, 0.2, 0.2],
          [0.2, 0.2, 0.5, 0.2],                  # (2,2) exactly 0.5 -> GT POSITIVE (>=)
          [0.2, 0.2, 0.2, 0.2]]
PRED = [["P", "h", "N", "N"],                    # (0,1) p = 0.49998 -> NEGATIVE (a FN)
        ["P", "P", "P", "N"],                    # (1,2) positive but lift-INVALID -> not scored
        ["N", "N", "H", "N"],                    # (2,2) p exactly 0.5 -> POSITIVE (>=)
        ["N", "N", "N", "P"]]                    # (3,3) positive but UNSEEN -> not scored


def _masks():
    seen = torch.ones(1, 4, 4, dtype=torch.bool)
    seen[0, 3, 3] = False
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[0, 1, 2] = False
    return seen, valid


def test_iou_rule_literal_4x4():
    seen, valid = _masks()
    r = M.drivable_iou_rule(_logits(PRED), _frac_two_channel(GT_DRV), seen, valid)
    # scored = 16 - (3,3) unseen - (1,2) invalid = 14; GT on them: (0,0) (0,1) (1,0) (1,1) (2,2) = 5;
    # prediction on them: (0,0) (1,0) (1,1) (2,2) = 4; intersection 4; union 5 -> 0.8
    assert r["n_scored"] == 14.0
    assert r["inter"] == 4.0
    assert r["union"] == 5.0
    assert r["iou"] == 0.8
    assert r["pred_drivable_frac"] == 4.0 / 14.0
    assert r["gt_drivable_frac"] == 5.0 / 14.0
    fn = r["_gt"] & ~r["_pr"]
    assert fn[0].nonzero().tolist() == [[0, 1]]            # the p = 0.49998 cell is the one FN


def test_iou_rule_without_valid_mask_scores_the_lift_invalid_cell():
    seen, _valid = _masks()
    r = M.drivable_iou_rule(_logits(PRED), _frac_two_channel(GT_DRV), seen, None)
    # (1,2) is now scored and is a FALSE POSITIVE: union 6, n_scored 15
    assert r["n_scored"] == 15.0
    assert r["union"] == 6.0
    assert r["iou"] == 4.0 / 6.0


def test_iou_rule_empty_union_reads_zero_like_the_trainer():
    seen, valid = _masks()
    all_n = [["N"] * 4 for _ in range(4)]
    r = M.drivable_iou_rule(_logits(all_n), _frac_two_channel(np.full((4, 4), 0.1)), seen, valid)
    assert r["union"] == 0.0 and r["iou"] == 0.0


def _mixed_frac():
    """A: argmax drivable AND >= 0.5; B: argmax drivable but 0.45 (MIXED); C: argmax edge (ch 5)."""
    f = np.zeros((1, C, 4, 4), np.float32)
    f[0, 0] = 1.0
    f[0, :, 0, 0] = 0.0
    f[0, DRV, 0, 0], f[0, 0, 0, 0] = 0.8, 0.2                              # A
    f[0, :, 0, 1] = 0.0
    f[0, DRV, 0, 1], f[0, 2, 0, 1], f[0, 0, 0, 1] = 0.45, 0.30, 0.25       # B (mixed)
    f[0, :, 0, 2] = 0.0
    f[0, DRV, 0, 2], f[0, 5, 0, 2] = 0.3, 0.7                              # C
    return torch.from_numpy(f)


def test_known_value_control_reads_exactly_one_and_the_literal_one_hot_does_not():
    frac = _mixed_frac()
    seen = torch.ones(1, 4, 4, dtype=torch.bool)
    z_c, info_c = M.gt_as_logits(frac, consistent=True)
    z_o, info_o = M.gt_as_logits(frac, consistent=False)
    assert info_c["n_mixed"] == 1 and info_c["n_impossible"] == 0
    r_c = M.drivable_iou_rule(z_c, frac, seen, None)
    assert r_c["iou"] == 1.0 and r_c["union"] == 1.0                     # the control
    r_o = M.drivable_iou_rule(z_o, frac, seen, None)
    assert r_o["inter"] == 1.0 and r_o["union"] == 2.0 and r_o["iou"] == 0.5   # B is argmax-drivable
    # the mixed cell keeps its argmax and sits under the threshold: e / (e + 8) = 0.253612
    p_b = float(z_c.softmax(dim=1)[0, DRV, 0, 1])
    assert abs(p_b - 0.253612) < 1e-5 and p_b < 0.5
    # both variants redraw the GT class map exactly (the panel half of control (a))
    gt_cls = frac.argmax(dim=1)
    assert torch.equal(M.argmax_classes(z_c), gt_cls)
    assert torch.equal(M.argmax_classes(z_o), gt_cls)


def test_control_mutations_go_red():
    frac = _mixed_frac()
    seen = torch.ones(1, 4, 4, dtype=torch.bool)
    z_c, _ = M.gt_as_logits(frac, consistent=True)
    # lateral mirror: A moves from (0,0) to (0,3) -> no overlap with the GT positive at (0,0)
    assert M.drivable_iou_rule(z_c.flip(-1), frac, seen, None)["iou"] == 0.0
    # the prediction's channels read one off: channel 1 now holds channel 0's logits
    r = M.drivable_iou_rule(z_c.roll(shifts=1, dims=1), frac, seen, None)
    assert r["inter"] == 0.0 and r["iou"] == 0.0


def test_control_on_a_full_size_random_map_panels_and_iou():
    g = np.random.default_rng(0)
    p = g.dirichlet(np.full(C, 0.6), size=(M.GRID_X, M.GRID_Y)).astype(np.float64)
    frac = torch.from_numpy(p.transpose(2, 0, 1)[None].astype(np.float32))
    seen = torch.from_numpy(g.random((1, M.GRID_X, M.GRID_Y)) > 0.2)
    valid = torch.from_numpy(g.random((1, M.GRID_X, M.GRID_Y)) > 0.1)
    z_c, info = M.gt_as_logits(frac, consistent=True)
    assert info["n_impossible"] == 0 and info["n_mixed"] > 0
    assert M.drivable_iou_rule(z_c, frac, seen, valid)["iou"] == 1.0
    z_o, _ = M.gt_as_logits(frac, consistent=False)
    assert M.drivable_iou_rule(z_o, frac, seen, valid)["iou"] < 1.0
    s_np, v_np = seen[0].numpy(), valid[0].numpy()
    gt_panel = M.class_map_array(frac[0].argmax(dim=0).numpy(), s_np, v_np)
    c_panel = M.class_map_array(M.argmax_classes(z_c)[0].numpy(), s_np, v_np)
    assert gt_panel.shape == (M.MAP_H, M.MAP_W, 3)
    assert np.array_equal(gt_panel, c_panel)
    # and a mirrored control does NOT redraw the GT panel
    m_panel = M.class_map_array(M.argmax_classes(z_c.flip(-1))[0].numpy(), s_np, v_np)
    assert not np.array_equal(gt_panel, m_panel)


def test_grid_orientation_ego_at_bottom_centre_right_is_right():
    a = np.zeros((M.GRID_X, M.GRID_Y), np.int64)
    a[0, 0] = 7                              # x in [0, 0.5) m, y in [-16, -15.5) m = nearest, RIGHT
    img = M.grid_to_image(a)
    assert img.shape == (600, 320)
    assert img[599, 319] == 7 and img[595, 315] == 7          # the bottom-right 5x5 block
    assert img[594, 319] == 0 and img[599, 314] == 0 and img[0, 0] == 0
    assert M.m2px(0.25, -15.75) == (317.5, 597.5)             # that cell's centre, in the block
    assert M.m2px(0.0, 0.0) == (160.0, 600.0)                 # the ego: bottom centre


def test_palette_never_reads_as_a_path():
    rep = M.assert_palette_disjoint()
    assert set(rep) == ({f"class{i}" for i in range(9)} | {"TP", "FP", "FN", "TN"}
                        | {f"agent_{n}" for n in M.AGENT_CLASSES})
    assert M.C_GT == (110, 231, 138) and M.C_SEL == (255, 158, 61)


# ---------------------------------------------------------------------------------------------- #
# --boxes: pure geometry and the known-value control's input                                      #
# ---------------------------------------------------------------------------------------------- #
def test_box_corners_literal():
    # a 4 x 2 m box at (10, 1) heading straight ahead: FL, FR, RR, RL
    c = M.box_corners_xy(10.0, 1.0, 4.0, 2.0, 0.0)
    assert np.allclose(c, [[12.0, 2.0], [12.0, 0.0], [8.0, 0.0], [8.0, 2.0]], atol=1e-12)
    # turned 90 deg LEFT: the length now runs along +y
    c90 = M.box_corners_xy(0.0, 0.0, 4.0, 2.0, math.pi / 2)
    assert np.allclose(c90, [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0], [-1.0, -2.0]], atol=1e-12)


def test_cuboid_corners_bottom_then_top():
    q = M.cuboid_corners(10.0, 1.0, 0.8, 4.0, 2.0, 1.6, 0.0)
    assert q.shape == (8, 3)
    assert np.allclose(q[:4, 2], 0.0) and np.allclose(q[4:, 2], 1.6)
    assert np.allclose(q[:4, :2], q[4:, :2])
    assert len(M.CUBOID_EDGES) == 12


def test_range_and_field_predicates():
    assert M.in_bev_range([0.0, 60.0, 60.1, 30.0, -0.1], [0.0, 16.0, 0.0, 16.1, 0.0]).tolist() == \
        [True, True, False, False, False]
    # 60 deg is IN (<=), 61 deg and anything behind is OUT
    x = [math.cos(math.radians(60)), math.cos(math.radians(61)), -1.0]
    y = [math.sin(math.radians(60)), math.sin(math.radians(61)), 0.0]
    assert M.in_camera_field(x, y).tolist() == [True, False, False]


def test_heading_change_literal():
    yaw = np.zeros(100)
    yaw[50:] = math.radians(35.0)            # a 35 deg turn 10 rows after row 40
    assert M.heading_change_deg(yaw, 40) == pytest.approx(35.0)
    assert M.heading_change_deg(yaw, 50) == 0.0          # already turned: no change AHEAD
    assert M.heading_change_deg(yaw, 99) == 0.0          # no future row at the clip end
    wrap = np.zeros(10)
    wrap[5:] = math.radians(179.0)
    wrap[:5] = math.radians(-179.0)                       # crossing the +-pi cut is 2 deg, not 358
    assert M.heading_change_deg(wrap, 0) == pytest.approx(2.0)


def test_gt_as_slots_copies_each_target_into_its_own_slot():
    box = torch.tensor([[10.0, 1.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0], [20.0, -3.0, 5.0, 2.0]])
    yaw = torch.tensor([0.1, 0.0, -0.2])
    cls = torch.tensor([0, -1, 5])
    valid = torch.tensor([True, False, True])
    s = M.gt_as_slots(box, yaw, cls, valid, n_slots=4)
    assert s["box"].shape == (1, 4, 4) and s["src"].tolist() == [0, 2, -1, -1]
    assert torch.equal(s["box"][0, :2], box[[0, 2]])
    assert s["presence_logit"][0].tolist() == [20.0, 20.0, -20.0, -20.0]
    assert int(s["cls_logits"][0, 0].argmax()) == 0 and int(s["cls_logits"][0, 1].argmax()) == 5
    assert torch.equal(s["box"][0, 2], torch.tensor([1000.0, 1000.0, 1.0, 1.0]))
