"""Literal tests for the refcv7 battery's decision code (no GPU, no data).

    PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" python -m pytest -q test_r7_battery.py

Every expectation is a LITERAL, never an expression over the code under test. Each guard has a
deliberate-regression arm that must go RED (a check that shares the defect it checks for is green
forever -- CLAUDE.md).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ------------------------------------------------------------------ G0 term classes ---- #
def test_term_class_literals():
    import g0_refcv7 as G
    want = {
        "eval_goal_gate_grad": "EXCLUDED", "eval_box3d_calib_prec": "EXCLUDED",
        "eval_agent_calib_n_pos": "EXCLUDED", "eval_step": "EXCLUDED",
        "eval_box3d_conf_ratio_alarm": "DERIVED",
        "eval_box3d_n_conf": "DETECTION", "eval_agent_tp@gate": "DETECTION",
        "eval_box3d_n_ignore_masked_slots": "DETECTION",
        "eval_box3d_n_pos": "COUNT", "eval_map_hires_n_lane_0_20": "COUNT",
        "eval_n_map_hires_cells": "COUNT", "eval_box3d_det_npos_all_all": "COUNT",
        "eval_agent_npos_person": "COUNT", "eval_tacv6_n_supervised_lat": "COUNT",
        "eval_windows": "COUNT", "eval_agent_rows_with_cam": "COUNT",
        "eval_box3d_ap2m": "DETECTION", "eval_box3d_det_ap2_all_all": "DETECTION",
        "eval_agent_f1@gate": "DETECTION", "eval_box3d_rec@gate_person": "DETECTION",
        "eval_box3d_conf_ratio": "DETECTION", "eval_agent_centre_err_p50": "DETECTION",
        "eval_box3d": "MATCHED", "eval_box3d_presence_layer1": "MATCHED",
        "eval_agent_yaw": "MATCHED", "eval_box3d_vis1": "MATCHED",
        "eval_map_hires_inter_lane_0_20": "MAP10_COUNTS",
        "eval_map_hires_unionraw_edge_80_100": "MAP10_COUNTS",
        "eval_map_hires_iou_crosswalk_20_40": "MAP10_IOU",
        "eval_map_hires_lshare_lane_0_20": "SMOOTH_OR_STOCHASTIC",
        "eval_lon_tac": "SMOOTH_OR_STOCHASTIC",          # 'lo[n_]' must NOT read as a count
        "eval_tacv6_lon_ce": "SMOOTH_OR_STOCHASTIC",
        "eval_traj": "SMOOTH_OR_STOCHASTIC", "eval_cascade": "SMOOTH_OR_STOCHASTIC",
    }
    got = {k: G.term_class(k) for k in want}
    assert got == want


def test_term_class_regression_arm_substring_rule_goes_red():
    """The refcv6 G0's historical bug: a raw `'n_' in key` test calls `lon_tac` a COUNT."""
    def buggy(k):
        return "COUNT" if "n_" in k else "OTHER"
    assert buggy("eval_lon_tac") == "COUNT"          # the defect is real ...
    import g0_refcv7 as G
    assert G.term_class("eval_lon_tac") != "COUNT"   # ... and the shipped rule does not have it


# ------------------------------------------------------------------ G0 tolerance ---- #
def test_tol_ok_literals():
    import g0_refcv7 as G
    assert G.tol_ok("DETECTION", "eval_box3d_ap2m", 0.30, 0.315)[0] is True
    assert G.tol_ok("DETECTION", "eval_box3d_ap2m", 0.30, 0.325)[0] is False
    assert G.tol_ok("SMOOTH", "eval_traj", 1.0, 1.009)[0] is True
    assert G.tol_ok("SMOOTH", "eval_traj", 1.0, 1.011)[0] is False
    assert G.tol_ok("SMOOTH", "eval_law", 0.05, 0.0509)[0] is True     # abs 1e-3 below 0.1
    assert G.tol_ok("MAP10_COUNTS", "x", 100.0, 124.0)[0] is True      # 25-cell floor
    assert G.tol_ok("MAP10_COUNTS", "x", 100.0, 126.0)[0] is False
    assert G.tol_ok("COUNT", "eval_box3d_n_pos", 509.0, 509.0)[0] is True
    assert G.tol_ok("COUNT", "eval_box3d_n_pos", 509.0, 510.0)[0] is False


# ------------------------------------------------------------------ bars ---- #
def _cells(delta, lo, hi):
    return {"pairs": {"os_minus_ha0ext": {"families": {"ADE": {"ade_m": {
        "delta": delta, "lo": lo, "hi": hi, "separated": (lo > 0 or hi < 0),
        "n_windows": 4754, "n_episodes": 139}}}}}}


def test_bar_verdict_literals():
    import run_battery_r7 as RB
    sep = {0: _cells(-0.05, -0.07, -0.03), 1: _cells(-0.04, -0.06, -0.02)}
    v = {b["id"]: b["verdict"] for b in RB.bar_verdicts(sep, sep, True, 0.005)}
    assert v["BAR-R7-1"] == "PASS (single training seed; training-seed floor unmeasured)"
    assert v["BAR-R7-3"] == "PASS (single training seed; training-seed floor unmeasured)"
    assert v["BAR-R7-2"].startswith("NOT EVALUABLE")          # no refcv6@38k cell in this fixture
    one_fails = {0: _cells(-0.05, -0.07, -0.03), 1: _cells(-0.01, -0.03, +0.01)}
    v = {b["id"]: b["verdict"] for b in RB.bar_verdicts(one_fails, one_fails, True, 0.005)}
    assert v["BAR-R7-1"] == "FAILED"
    small = {0: _cells(-0.008, -0.010, -0.006), 1: _cells(-0.008, -0.010, -0.006)}
    v = {b["id"]: b["verdict"] for b in RB.bar_verdicts(small, small, True, 0.005)}
    assert v["BAR-R7-1"] == "NOT PROVEN (|delta| within 2x the inference-seed floor)"
    v = {b["id"]: b["verdict"] for b in RB.bar_verdicts(sep, sep, False, 0.005)}
    assert v["BAR-R7-1"] == "NOT EVALUATED (pipeline validation checkpoint; SPEC §6)"
    v = {b["id"]: b["verdict"] for b in RB.bar_verdicts(sep, sep, True, 0.005, pending=("BAR-R7-2",))}
    assert v["BAR-R7-2"] == "PENDING (the refcv6@38k S2 rolls run LAST in the milestone chain; SPEC A4)"
    assert v["BAR-R7-1"] == "PASS (single training seed; training-seed floor unmeasured)"
    only_one_seed = {0: _cells(-0.05, -0.07, -0.03)}
    v = {b["id"]: b["verdict"] for b in RB.bar_verdicts(only_one_seed, only_one_seed, True, 0.005)}
    assert v["BAR-R7-1"].startswith("NOT EVALUABLE")


# ------------------------------------------------------------------ GPU lock ---- #
def test_gpu_lock_exclusive_and_owner_only_release(monkeypatch):
    import gpu_lock
    d = tempfile.mkdtemp()
    monkeypatch.setattr(gpu_lock, "LOCK", os.path.join(d, "devbox_gpu.lock"))
    assert gpu_lock.try_acquire("jobA", 111) is True
    assert gpu_lock.try_acquire("jobB", 222) is False               # exclusive create
    assert gpu_lock.release("jobB")["released"] is False             # never breaks another's lock
    assert gpu_lock.read_lock()["job"] == "jobA"
    assert gpu_lock.release("jobA", 999)["released"] is False        # right job, wrong pid: kept
    assert gpu_lock.release("jobA", 111)["released"] is True
    assert gpu_lock.read_lock() is None
    # another session's lock with extra fields is read, and never removed by us
    with open(gpu_lock.LOCK, "w", encoding="utf-8") as f:
        json.dump({"job": "refe-snapshot-eval-019", "pid": 5, "token": "x", "utc": "t",
                   "win_pid_of_locker": 5, "host": "h", "acquired": "a"}, f)
    assert gpu_lock.read_lock()["job"] == "refe-snapshot-eval-019"
    assert gpu_lock.try_acquire("jobC", 7) is False
    assert gpu_lock.release("jobC", 7)["released"] is False
    assert gpu_lock.read_lock()["job"] == "refe-snapshot-eval-019"


# ------------------------------------------------------------------ micro-batching ---- #
def test_microbatch_pops_and_slices_ego_window_and_actions():
    """refcv7 NEW-1's one-shot `_ego_actions` rides with the ego window: both are POPPED and sliced
    per micro-part. Regression arm: the refcv6 wrapper (no `_ego_actions` handling) leaves it set."""
    import torch
    from microbatch import MicroBatchForward

    class Core:
        ego_hist = object()
        _ego_window = None
        _ego_actions = None

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Core()
            self.seen = []

        def forward(self, frames, ego_poses=None, ego_n_past=None, ego_actions=None, **kw):
            self.seen.append((int(frames.shape[0]), int(ego_poses.shape[0]),
                              None if ego_actions is None else ego_actions[:, 0, 0].tolist()))
            return {"traj": frames.sum(dim=1), "s": torch.tensor(float(frames.shape[0]))}

    m = M()
    m.core._ego_window = (torch.zeros(5, 4, 4), 4)
    m.core._ego_actions = torch.arange(5.).reshape(5, 1, 1).expand(5, 4, 2).contiguous()
    w = MicroBatchForward(m, [2, 3]).install()
    out = m(torch.ones(5, 3))
    w.remove()
    assert m.seen == [(2, 2, [0.0, 1.0]), (3, 3, [2.0, 3.0, 4.0])]
    assert m.core._ego_window is None and m.core._ego_actions is None
    assert tuple(out["traj"].shape) == (5,)
    assert abs(float(out["s"]) - 2.6) < 1e-6                      # (2*2 + 3*3) / 5, row-weighted

    # regression arm: a wrapper that pops only the window (the refcv6 form) leaves the actions set
    class OldWrapper(MicroBatchForward):
        def _forward(self, frames, *args, **kw):
            saved = self.model.core._ego_actions
            try:
                return super()._forward(frames, *args, **kw)
            finally:
                self.model.core._ego_actions = saved               # the historical omission
    m2 = M()
    m2.core._ego_window = (torch.zeros(5, 4, 4), 4)
    m2.core._ego_actions = torch.zeros(5, 4, 2)
    w2 = OldWrapper(m2, [2, 3]).install()
    m2(torch.ones(5, 3))
    w2.remove()
    assert m2.core._ego_actions is not None                         # the defect is visible


# ------------------------------------------------------------------ pull tool ---- #
def test_pull_metrics_state_literals():
    import pull_ckpt
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 4950, "loss": 1.0}) + "\n")
        f.write(json.dumps({"step": 5000, "loss": 1.0}) + "\n")
        f.write(json.dumps({"step": 5000, "eval_loss": 2.0}) + "\n")
        f.write('{"step": 5010, "loss"')                              # a line being appended
    s = pull_ckpt.metrics_state(p, 5000)
    assert s == {"n_rows": 3, "last_step": 5000, "eval_row_at_step": 1, "eval_error_at_step": 0}
    s = pull_ckpt.metrics_state(p, 5500)
    assert s["eval_row_at_step"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
