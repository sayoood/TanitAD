"""SPEC AMENDMENT A3: the paired yaw-rate cell must be scored on steps that HAVE a path tangent.

Deliberate regression built in: the shared `refav1_arm._components` cell (unmasked) must SEE the
defect on the same data, so a green run proves the test can tell the two apart. Every expected
value is a LITERAL (an analytic target), never an expression over the code under test.

run:  PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" python -m pytest -q test_yaw_valid.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refcv6_panel as P  # noqa: E402

DT = 0.5
# a valid window: 10 m/s straight line on the 0.5 s grid (every step 5 m > min_ds 0.25 m)
STRAIGHT = np.array([[5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0]], dtype=np.float32)
# a stopped GT window: no displacement at all
STOPPED = np.zeros((4, 2), dtype=np.float32)
# a prediction that jitters 0.1 m per step around the origin: heading turns by pi/2 every step,
# so its yaw-rate is exactly (pi/2)/0.5 = pi rad/s on every step pair; every step is < min_ds
JITTER = np.array([[0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]], dtype=np.float32)


def test_valid_cell_reads_zero_on_an_exact_plan_and_drops_the_stopped_window():
    G = np.stack([STRAIGHT, STOPPED])
    Pp = np.stack([STRAIGHT, JITTER])
    v = P.yaw_rate_valid(Pp, G, DT)
    assert v[0] == 0.0                       # literal: an exact plan has zero yaw-rate error
    assert math.isnan(v[1])                  # literal: no valid step pair -> dropped, not scored


def test_shared_cell_sees_the_defect_on_the_same_data():
    ra = P.RR.ra3().ra
    G = np.stack([STRAIGHT, STOPPED])
    Pp = np.stack([STRAIGHT, JITTER])
    shared = ra._components(Pp, G, DT)[P.YAW_SHARED]
    assert shared[0] == 0.0
    assert abs(float(shared[1]) - 3.141592653589793) < 1e-5   # literal pi rad/s: the jitter is SCORED


def test_paired_cell_standstill_jitter_is_no_longer_a_separated_difference():
    ra = P.RR.ra3().ra
    # 3 episodes x (one valid window, one stopped-GT window). Arm a is exact everywhere; arm b is
    # exact on the valid windows and jitters on the stopped ones -- lateral skill is IDENTICAL.
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
    shared, valid = lat[P.YAW_SHARED], lat[P.YAW_VALID]
    # the defect: pi/2 rad/s "worse", separated, from standstill jitter alone
    assert abs(shared["delta"] - 1.5707963267948966) < 1e-5 and shared["separated"]
    # A3: identical lateral skill reads exactly zero, and the three stopped windows are counted out
    assert valid["delta"] == 0.0 and valid["lo"] == 0.0 and valid["hi"] == 0.0
    assert not valid["separated"]
    assert valid["n_dropped_nonfinite"] == 3
    assert valid["amendment"] == "A3" and blk["A3_note"].startswith("SPEC A3")
