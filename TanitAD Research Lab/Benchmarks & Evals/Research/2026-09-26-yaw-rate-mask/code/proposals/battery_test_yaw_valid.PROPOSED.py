"""PROPOSED replacement for the refcv6 battery's `code/test_yaw_valid.py` -- NOT APPLIED.

Owner: the EvalFlyWheel battery package
(`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/`).
This file lives in the 2026-09-26-yaw-rate-mask package only as a proposal.

WHY IT IS NEEDED. The landed test's deliberate regression is the SHARED
`refav1_arm._components` cell itself: two of its three tests assert the shared cell
still reads the defect (pi rad/s on a stopped window; a pi/2 separated paired delta).
Once the shared cell is fixed (D-YAWMASK-1) those two go RED by design -- MEASURED
2026-09-26 in a clean tree: 3/3 green on the tip's refav1_arm.py, 1 green / 2 red on
the fixed one (`raw/battery_test_interaction.json`).

WHAT CHANGES. The regression arm no longer depends on the shared code being broken:
the historical unmasked cell is re-implemented HERE, verbatim, from four_families'
own geometry, and must still read pi on the jitter window; the shared cell must now
AGREE with the A3 `_valid` cell (NaN on the stopped window, 0.0 paired delta). Every
expected value is still a LITERAL.

run:  PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" python -m pytest -q test_yaw_valid.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refcv6_panel as P  # noqa: E402

DT = 0.5
STRAIGHT = np.array([[5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0]], dtype=np.float32)
STOPPED = np.zeros((4, 2), dtype=np.float32)
JITTER = np.array([[0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]], dtype=np.float32)


def _historical_unmasked_cell(Pp, G, dt):
    """The pre-2026-09-26 shared line, verbatim: every pair averaged, no mask."""
    import torch
    from taniteval import four_families as ff
    pt, gt = torch.as_tensor(Pp).float(), torch.as_tensor(G).float()
    Pg, Gg = ff._seq_geometry(pt, dt), ff._seq_geometry(gt, dt)
    return (Pg["yaw_rate"] - Gg["yaw_rate"]).abs().mean(1).numpy()


def test_valid_cell_reads_zero_on_an_exact_plan_and_drops_the_stopped_window():
    G = np.stack([STRAIGHT, STOPPED])
    Pp = np.stack([STRAIGHT, JITTER])
    v = P.yaw_rate_valid(Pp, G, DT)
    assert v[0] == 0.0
    assert math.isnan(v[1])


def test_the_historical_defect_is_still_detectable_on_the_same_data():
    # the regression arm, now independent of the shared code's state
    G = np.stack([STRAIGHT, STOPPED])
    Pp = np.stack([STRAIGHT, JITTER])
    old = _historical_unmasked_cell(Pp, G, DT)
    assert old[0] == 0.0
    assert abs(float(old[1]) - 3.141592653589793) < 1e-5


def test_the_shared_cell_now_agrees_with_the_a3_cell():
    ra = P.RR.ra3().ra
    G = np.stack([STRAIGHT, STOPPED])
    Pp = np.stack([STRAIGHT, JITTER])
    shared = ra._components(Pp, G, DT)[P.YAW_SHARED]
    assert shared[0] == 0.0
    assert math.isnan(float(shared[1])), "the shared cell still scores standstill jitter"


def test_paired_cells_agree_and_standstill_jitter_is_not_a_difference():
    ra = P.RR.ra3().ra
    G = np.stack([STRAIGHT, STOPPED] * 3)
    A_ = np.stack([STRAIGHT, STOPPED] * 3)
    B_ = np.stack([STRAIGHT, JITTER] * 3)
    eid = [0, 0, 1, 1, 2, 2]
    comps = {"a": ra._components(A_, G, DT), "b": ra._components(B_, G, DT)}
    for k, arr in (("a", A_), ("b", B_)):
        comps[k][P.YAW_VALID] = P.yaw_rate_valid(arr, G, DT)
    blk = P._attach_a3(ra._paired_families(comps, "a", "b", eid, {"a": "T1", "b": "T1"}, 200, 0),
                       comps, "a", "b", eid, 200, 0)
    lat = blk["families"]["lateral"]
    for key in (P.YAW_SHARED, P.YAW_VALID):
        c = lat[key]
        assert c["delta"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0, (key, c)
        assert not c["separated"]
        assert c["n_dropped_nonfinite"] == 3
    assert "pred.pair_valid AND gt.pair_valid" in blk["yaw_rate_cell"]
