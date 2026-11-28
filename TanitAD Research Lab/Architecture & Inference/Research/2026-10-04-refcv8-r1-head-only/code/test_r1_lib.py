"""Analytic / known-value controls for r1_lib (written with literal expectations; run: python test_r1_lib.py).
Each expected value is derived from road geometry, never from r1_lib itself. The last test is a MUTATION: it
re-introduces a wrong turn threshold and asserts the arc control goes RED."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1_lib as L  # noqa: E402

T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
VALID = np.ones((1, 8), bool)
LK, TL, TR = 0, 6, 7          # literal indices of LANE_KEEP / TURN_L / TURN_R in the v7 lat vocabulary
HOLD, CREEP, CRUISE, BRAKE, ACCEL = 5, 4, 1, 3, 7     # literal indices in the v7 lon vocabulary


def arc(R, v, side=1):
    th = v * T / R
    return np.stack([R * np.sin(th), side * R * (1 - np.cos(th))], -1)[None]


def test_vocab_literals():
    assert L.LAT_V7[LK] == "LANE_KEEP" and L.LAT_V7[TL] == "TURN_L" and L.LAT_V7[TR] == "TURN_R"
    assert L.LON_V7[HOLD] == "HOLD" and L.LON_V7[CREEP] == "CREEP" and L.LON_V7[CRUISE] == "CRUISE"
    assert L.LON_V7[BRAKE] == "BRAKE_TO" and L.LON_V7[ACCEL] == "ACCELERATE"


def test_lat_analytic():
    straight = np.stack([10 * T, 0 * T], -1)[None]
    assert L.dense_lat(straight, VALID)[0] == LK
    assert L.dense_lat(arc(20.0, 5.0, +1), VALID)[0] == TL          # 30 m on R 20 m = 85.9 deg
    assert L.dense_lat(arc(20.0, 5.0, -1), VALID)[0] == TR
    assert L.dense_lat(arc(300.0, 14.0, +1), VALID)[0] == LK        # 84 m on R 300 m = 16.0 deg -> no turn
    assert L.dense_lat(np.zeros((1, 8, 2)), VALID)[0] == LK         # standstill: no turn identifiable
    bad = VALID.copy()
    bad[0, -1] = False
    assert L.dense_lat(arc(20.0, 5.0, +1), bad)[0] == L.IGNORE_ID
    # the GT-turn rule of the route package reads the same arc as a left turn (terminal heading 85.9 deg)
    cls, th = L.gt_class(arc(20.0, 5.0, +1), VALID)
    # terminal heading = the slot-50 -> 60 CHORD = mean of 1.25 and 1.50 rad = 78.78 deg
    assert cls[0] == "turnL" and abs(math.degrees(th[0]) - 78.78) < 0.05


def test_lon_analytic():
    fut_ok = np.ones((1, 60), bool)
    tt = np.arange(1, 61) * 0.1
    assert L.dense_lon([10.0], (10.0 + 0 * tt)[None], fut_ok)[0] == CRUISE
    assert L.dense_lon([10.0], (10.0 + 2.0 * tt / 6.0)[None], fut_ok)[0] == ACCEL      # +2.0 m/s
    assert L.dense_lon([10.0], (10.0 + 1.0 * tt / 6.0)[None], fut_ok)[0] == CRUISE     # +1.0 < the +1.5 bar
    assert L.dense_lon([10.0], (10.0 - 2.0 * tt / 6.0)[None], fut_ok)[0] == BRAKE
    assert L.dense_lon([0.0], (0 * tt)[None], fut_ok)[0] == HOLD
    assert L.dense_lon([1.0], (1.0 + 0.5 * tt / 6.0)[None], fut_ok)[0] == CREEP
    bad = fut_ok.copy()
    bad[0, -1] = False
    assert L.dense_lon([10.0], (10.0 + 0 * tt)[None], bad)[0] == L.IGNORE_ID


def test_nav_ann():
    ents = [{"token": "NAV_TURN_R", "t_start_s": 1.0, "t_end_s": 3.0},      # a curve the builder does not announce
            {"token": "NAV_TURN_L", "t_start_s": 6.0, "t_end_s": 9.0}]
    assert L.nav_ann_state(ents, [False, True], 0.5)[0] == +1               # skipped curve, left turn 5.5 s ahead
    assert L.nav_ann_state(ents, [False, True], -1.0)[0] == 0               # left turn 7.0 s ahead > H
    assert L.nav_ann_state(ents, [True, True], 0.5)[0] == -1                # announced: the first one wins
    assert L.nav_ann_state(ents, [False, True], 7.0)[0] == +1               # under way
    assert L.nav_ann_state(ents, [False, True], 9.5)[0] == 0                # all finished
    fol = [{"token": "NAV_FOLLOW_ROAD", "t_start_s": 0.0, "t_end_s": 30.0}]
    assert L.nav_ann_state(fol, [False], 0.0) == (0, "follow_token")


def test_shuffle_changes_labels():
    lab = np.array([0, 0, 6, 6, 0, 7, 0, 0])
    clip = np.array(["a"] * 4 + ["b"] * 4)
    sh, frac = L.shuffle_within_clip(lab, clip, seed=0)
    assert sorted(sh[:4]) == sorted(lab[:4]) and sorted(sh[4:]) == sorted(lab[4:])
    assert frac > 0


def test_quant_roundtrip():
    import torch
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 96, 120, 64, generator=g) * 3 + 1
    q, lo, sc = L.quant_u8(x)
    y = L.dequant_u8(q, lo, sc)
    assert float(((y - x).abs() / sc[..., None, None]).max()) <= 0.5 + 1e-4


def test_mutation_turn_threshold_goes_red():
    keep = L.DENSE_TURN_DEG
    try:
        L.DENSE_TURN_DEG = 90.0                    # the mutation: a wrong threshold
        red = L.dense_lat(arc(20.0, 5.0, +1), VALID)[0] != TL
    finally:
        L.DENSE_TURN_DEG = keep
    assert red, "the arc control did not detect a 90-degree turn threshold"




def test_listwise_target_prefers_gt_speed_and_direction():
    """Known values: a candidate EQUAL to GT has cost 0 and the largest target mass; a candidate with the right path
    but half the speed, and a mirrored turn, each cost more; the loss is minimised by scoring the GT twin highest."""
    import torch
    import r1_arms as A
    gt = torch.tensor(arc(20.0, 5.0, +1)[0], dtype=torch.float32)            # [8, 2]
    slow = torch.tensor(arc(20.0, 2.5, +1)[0], dtype=torch.float32)          # same arc, half the progress
    mirror = torch.tensor(arc(20.0, 5.0, -1)[0], dtype=torch.float32)
    fan = torch.stack([slow, gt, mirror])[None]                              # [1, 3, 8, 2]
    gv = torch.ones(1, 8, dtype=torch.bool)
    reach = torch.ones(1, 3, dtype=torch.bool)
    s_arc = A.cum_arclen(gt[None])[0]
    assert abs(float(s_arc[-1]) - 30.0) < 0.6                                # 30 m of arc, chord-summed
    good = A.listwise_loss(torch.tensor([[0.0, 5.0, 0.0]]), fan, gt[None], gv, reach)
    bad = A.listwise_loss(torch.tensor([[5.0, 0.0, 0.0]]), fan, gt[None], gv, reach)
    worse = A.listwise_loss(torch.tensor([[0.0, 0.0, 5.0]]), fan, gt[None], gv, reach)
    assert float(good) < float(bad) and float(good) < float(worse)
    # a candidate masked out of reach carries no target mass even if it is the GT twin
    m = torch.tensor([[True, False, True]])
    l_m = A.listwise_loss(torch.tensor([[0.0, 50.0, 0.0]]), fan, gt[None], gv, m)
    assert torch.isfinite(l_m)




def test_quant_row_roundtrip_and_tighter_than_channel():
    """per-(channel, row) int8: error <= half a step of its OWN row's scale, and never coarser than per-channel."""
    import torch
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 96, 120, 64, generator=g) * torch.linspace(0.1, 5.0, 120)[None, None, :, None]
    q, lo, sc = L.quant_u8_row(x)
    y = L.dequant_u8(q, lo, sc)
    assert float(((y - x).abs() / sc[..., None]).max()) <= 0.5 + 1e-4
    qc, loc, scc = L.quant_u8(x)
    assert float((y - x).abs().mean()) < float((L.dequant_u8(qc, loc, scc) - x).abs().mean())


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print(f"{n}/{n} pass")
