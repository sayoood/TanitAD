"""YAXD shared constants + the inference-time ax TRANSFORMS (PREREG_AXD.md section 2). numpy only; read-only imports of YAX (ax_common / ax_stats / ax_analyze).

Ego vector order (refe/planner.py:448-451): vx, vy, ax, ay, yaw_rate, steer, speed -> slot 2 = ax (rear-axle longitudinal acceleration, m/s^2).
History: the NAVSIM agent input in the W3 export carries 4 ego statuses at 0.5 s (t = -1.5, -1.0, -0.5, 0; MEASURED 2026-11-07 on all 923 A5 tokens: len 4 each) with
`ego_acceleration` (refe_navtest_seam.py:93 reads the same `ego_statuses`). The seam's own ego comes from the nuPlan DB at iteration 0 (refe_navtest_seam.py:243, history_rows=1).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
YAX = Path("E:/Projects/TanitAD-artifacts/yawrate_wp/YAX")
sys.path.insert(0, str(YAX))
import ax_common as C  # noqa: E402,F401  (paths: PKG, EVAL, D, U400, A5_TOKENS, BANKED_*, PY_D, PY_T, HARNESS)

RUN_ROOT = Path("C:/Users/Admin/yaxd_run")
YAX_MARKERS = YAX / "markers"
YAX_STATUS = YAX / "run_full" / "status.json"
PREREG = HERE / "PREREG_AXD.md"
AX = 2
DELTA = 0.5                                   # the ego probe's +-0.5 m/s^2 (refe_ego_probe.py:312)
TRANSFORMS = ("IDT", "AX0", "CLAMP0", "CLAMPm05", "SCALE05", "SMOOTH4")
CANDIDATES = ("AX0", "CLAMP0", "CLAMPm05", "SCALE05", "SMOOTH4")      # AX0 = the reference transform (the sim's inherited withhold), judged by the same rule
PDMS_ARMS = ("A0",) + CANDIDATES              # scored by the harness on every token
GAIN_BAR, GAIN_CI_BAR, PDMS_MARGIN, LON_BAR = 0.55, 0.92, -0.5, 0.3   # 0.92 = u400's CRUISE gain CI-lo, refe_ego_probe_20261106/RESULT.md (MEASURED)
POS_CTRL_MIN = 0.5                            # IDT down-gain CI-lo must exceed this or the gain instrument is VOID


def gain_arms():
    """(arm name, transform, d, mode) for the CRUISE perturbation arms. mode: 'inst' = only the t0 sample moves; 'sust' = every history sample moves too; 'abs' = the probe's literal set."""
    out = []
    for t in TRANSFORMS:
        for s, d in (("+", DELTA), ("-", -DELTA)):
            out.append((f"{t}@{s}", t, d, "inst"))
    out += [("SMOOTH4@+s", "SMOOTH4", DELTA, "sust"), ("SMOOTH4@-s", "SMOOTH4", -DELTA, "sust"), ("IDT@abs+", "IDT", DELTA, "abs"), ("IDT@abs-", "IDT", -DELTA, "abs")]
    return out


def transform(name: str, ax_raw: float, hist: np.ndarray | None, d: float = 0.0, mode: str = "inst") -> np.float32:
    """The ax value FED to the planner. ax_raw = the planner's own float32 slot-2 value; hist = the 3 export ax samples at t = -1.5, -1.0, -0.5 (SMOOTH4 only)."""
    r = np.float64(np.float32(ax_raw))
    r = np.float64(d) if mode == "abs" else r + np.float64(d)
    if name == "IDT":
        v = r
    elif name == "AX0":
        v = 0.0
    elif name == "CLAMP0":
        v = max(r, 0.0)
    elif name == "CLAMPm05":
        v = max(r, -0.5)
    elif name == "SCALE05":
        v = 0.5 * r
    elif name == "SMOOTH4":
        if hist is None or len(hist) != 3 or not np.all(np.isfinite(hist)):
            raise ValueError("SMOOTH4 needs the 3 history ax samples")
        h = np.asarray(hist, np.float64) + (np.float64(d) if mode == "sust" else 0.0)
        v = (h.sum() + r) / 4.0
    else:
        raise KeyError(name)
    return np.float32(v)


def history_ax(doc_tokens: dict, tok: str):
    """export ax at t = -1.5, -1.0, -0.5 and t0 (the last status)."""
    es = doc_tokens[tok]["ego_statuses"]
    a = np.asarray([float(s["ego_acceleration"][0]) for s in es], np.float64)
    return a[:-1], float(a[-1])


def v4(poses):
    """planned speed at 4 s on NAVSIM's 0.5 s grid: chord |p8 - p7| / 0.5 with the origin prepended (ax_analyze.plan_speed, step 8)."""
    p = np.concatenate([np.zeros_like(np.asarray(poses)[:, :1, :2], dtype=np.float64), np.asarray(poses)[:, :, :2].astype(np.float64)], 1)
    return np.linalg.norm(p[:, 8] - p[:, 7], axis=-1) / 0.5
