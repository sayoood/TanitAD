"""refcv8 WP-C F4b -- FULL-SET ACCEPTANCE against the route package's banked packs (``raw/box_nms.json``).

The committed unit tests (``test_refcv8_det_nms.py``) run on small fixtures cut from this evidence. THIS file runs on all of it and
is SKIPPED where the evidence is not on the machine (``REFCV7_ROUTE_BIN`` or ``D:/refcv7_route_bin/2026-10-04``: ``eval_s0.packs.pkl``
= the 1,061 labelled EVAL-DIAG windows, ``train_s0.packs.pkl`` = the 1,074 labelled TRAIN-DIAG windows of refcv7-r101-s0 at step 50,400).

* EVAL (fast): the shipped radius + gate (``refcv7_det_nms_train.json``) reproduce, from the banked EVAL packs and WITHOUT the route
  package's code, every point estimate of ``raw/box_nms.json`` -- AP@2 m 0.2483 -> 0.3494 (box3d), 0.1314 -> 0.2996 (agent); the census at
  the re-fitted gate; the boxes per detected object 2.12 -> 1.07 / 2.60 -> 1.01; and the halving of conf_ratio at the OLD gate 0.2589.
  (The route package carried its re-fitted gates at 4 dp -- 0.2145 / 0.1809 -- so the census is asserted at THOSE gates, and the shipped
  full-precision gates are asserted to round to them.)
* TRAIN (slow, ``REFCV8_FULL_FIT=1``, ~5 min): ``fit_nms`` on all 1,074 TRAIN windows picks the same radii (2.5 m / 3.0 m) and reproduces
  the whole TRAIN grid (AP@2 m and the re-fitted gate at every radius) of ``raw/box_nms.json``.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest

from tanitad.eval import detection_metrics as det
from tanitad.eval import detection_nms as N

BIN = Path(os.environ.get("REFCV7_ROUTE_BIN", "D:/refcv7_route_bin/2026-10-04"))
SHIPPED = Path(N.__file__).resolve().parent.parent / "configs" / "refcv7_det_nms_train.json"
OLD_GATE = 0.2589

need_eval = pytest.mark.skipif(not (BIN / "eval_s0.packs.pkl").exists(), reason="the route package's banked EVAL packs are not on this machine")
need_train = pytest.mark.skipif(not (BIN / "train_s0.packs.pkl").exists(), reason="the route package's banked TRAIN packs are not on this machine")
slow = pytest.mark.skipif(os.environ.get("REFCV8_FULL_FIT") != "1", reason="set REFCV8_FULL_FIT=1 (about 5 minutes)")

# raw/box_nms.json <head>.eval.* (point estimates, 4 dp as banked)
BOX_NMS = {
    "box3d": dict(r=2.5, gate_4dp=0.2145, ap0=0.2483, ap1=0.3494, n_conf0=3301, tp0=1015, gate0_4dp=0.2567,
                  n_conf1=3372, tp1=1326, n_pos=3390, f1_0=0.3034, f1_1=0.3922, cr_old=0.4304, boxes0=2.1198, boxes1=1.0673),
    "agent": dict(r=3.0, gate_4dp=0.1809, ap0=0.1314, ap1=0.2996, n_conf0=3366, tp0=699, gate0_4dp=0.2307,
                  n_conf1=3621, tp1=1289, n_pos=3390, f1_0=0.2069, f1_1=0.3677, cr_old=0.0873, boxes0=2.596, boxes1=1.0123),
}
# raw/box_nms.json <head>.train_fit_grid (train_ap2m, train_pr_gate), 4 dp
TRAIN_GRID = {
    "box3d": {"none": (0.2562, 0.2567), "0.5": (0.259, 0.2561), "1": (0.2811, 0.2493), "1.5": (0.3205, 0.2364),
              "2": (0.3484, 0.2245), "2.5": (0.3519, 0.2145), "3": (0.3478, 0.2056), "4": (0.3068, 0.1917)},
    "agent": {"none": (0.1328, 0.2307), "0.5": (0.1419, 0.2282), "1": (0.195, 0.2158), "1.5": (0.2544, 0.204),
              "2": (0.2902, 0.1954), "2.5": (0.3067, 0.1873), "3": (0.3107, 0.1809), "4": (0.2716, 0.1714)},
}


@pytest.fixture(scope="module")
def eval_packs():
    with open(BIN / "eval_s0.packs.pkl", "rb") as fh:
        return pickle.load(fh)


@need_eval
@pytest.mark.parametrize("head", ["box3d", "agent"])
def test_FULL_eval_ap_census_and_duplicates_reproduce_box_nms_json(eval_packs, head):
    lit = BOX_NMS[head]
    cfg, _ = N.load_head_nms(SHIPPED)
    assert cfg[head]["radius_m"] == lit["r"] and round(cfg[head]["gate"], 4) == lit["gate_4dp"]
    P = eval_packs[head]
    assert len(P) == 1061
    Q = N.nms_packs(P, cfg[head]["radius_m"])
    assert round(N.ap2m(P), 4) == lit["ap0"] and round(N.ap2m(Q), 4) == lit["ap1"]
    c0 = N._census(P, lit["gate0_4dp"])
    assert (c0["n_conf"], c0["tp"], c0["n_pos"]) == (lit["n_conf0"], lit["tp0"], lit["n_pos"])
    assert round(c0["f1"], 4) == lit["f1_0"]
    c1 = N._census(Q, lit["gate_4dp"])                                  # the gate as the route package carried it (4 dp)
    assert (c1["n_conf"], c1["tp"], c1["n_pos"]) == (lit["n_conf1"], lit["tp1"], lit["n_pos"])
    assert round(c1["f1"], 4) == lit["f1_1"]
    assert round(N._census(Q, OLD_GATE)["conf_ratio"], 4) == lit["cr_old"] and N._census(Q, OLD_GATE)["conf_ratio_alarm"] == 1.0
    d0, d1 = N.duplicate_stats(P, OLD_GATE), N.duplicate_stats(Q, OLD_GATE)
    assert round(d0["boxes_per_object"], 4) == lit["boxes0"] and round(d1["boxes_per_object"], 4) == lit["boxes1"]
    # the shipped FULL-precision gate: at most one slot away from the 4 dp census, still in band, same TP count
    ck = N.nms_census_keys(P, head, cfg)
    assert abs(ck[f"eval_{head}_nms_n_conf"] - lit["n_conf1"]) <= 2 and ck[f"eval_{head}_nms_tp"] == lit["tp1"]
    assert ck[f"eval_{head}_nms_conf_ratio_alarm"] == 0.0
    assert round(ck[f"eval_{head}_nms_ap2m"], 4) == lit["ap1"]


@need_eval
def test_FULL_eval_gains_have_the_banked_sign_and_size(eval_packs):
    """dAP +0.1011 / +0.1682 and dF1 +0.0888 / +0.1608 (raw/box_nms.json), within the 4-dp rounding of the two ends."""
    for head, dap, df1 in (("box3d", 0.1011, 0.0888), ("agent", 0.1682, 0.1608)):
        lit = BOX_NMS[head]
        got_dap = round(N.ap2m(N.nms_packs(eval_packs[head], lit["r"])), 4) - round(N.ap2m(eval_packs[head]), 4)
        assert got_dap == pytest.approx(dap, abs=1.5e-4)
        assert (lit["f1_1"] - lit["f1_0"]) == pytest.approx(df1, abs=1.5e-4)


@need_eval
def test_FULL_the_planner_inputs_are_never_the_nmsd_packs(eval_packs):
    """Read-only on the real packs: running the NMS leaves the raw presence the planner reads bit-identical."""
    P = eval_packs["box3d"][:200]
    before = [p["logit"].tobytes() for p in P]
    N.nms_packs(P, 2.5)
    assert [p["logit"].tobytes() for p in P] == before


@need_train
@slow
@pytest.mark.parametrize("head", ["box3d", "agent"])
def test_FULL_train_fit_reproduces_the_whole_grid_and_the_chosen_radius(head):
    with open(BIN / "train_s0.packs.pkl", "rb") as fh:
        T = pickle.load(fh)[head]
    assert len(T) == 1074
    f = N.fit_nms(T)
    assert f["radius_m"] == BOX_NMS[head]["r"]
    for k, (ap, gate) in TRAIN_GRID[head].items():
        assert round(f["grid"][k]["train_ap2m"], 4) == ap, (head, k)
        assert round(f["grid"][k]["train_pr_gate"], 4) == gate, (head, k)
    cfg, _ = N.load_head_nms(SHIPPED)
    assert f["gate"] == pytest.approx(cfg[head]["gate"], rel=1e-12)       # the shipped file IS this fit
