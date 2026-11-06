"""AGENT F audit pins (CPU). The T2 scorer on NuRec scenes: (1) the logged track fed as the 'plan' scores ~0 against itself; (2) the T2 `g` equals the T1 reader's own path
(`tool.exec_in_ego` on the logged native rows = `floors_kin.gt_waypoints`) bit for bit; (3) the stage-1 level (P10 heading ~61 deg, yaw rate ~283 deg/s on the launch start) is what the
T1 reader's own BANKED rows read on standstill-launch windows (the audit's verdict (a)).
Every `check_*` takes `mut=None`; `audit/run_mutations_audit.py` runs each with a deliberate regression that must raise AssertionError (RED).
Run from t2v8f/: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_audit.py -q   (the executor and the scorer run in two processes: the scorer binds the reader tree)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
AUD = HERE + "/../audit"
import numpy as np  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PL = HERE + "/../stage1/pull_box2/pull/plumbing"
CASES = [f"{PL}/00040136/objects/T2/P10/scene_0.npz:60", f"{PL}/7c72937c/objects/T2/P10/scene_1.npz:22"]
RIC0_ARR = "E:/Projects/TanitAD-artifacts/dagland/motion_c/ri_read/raw/arrays/RIC0_s0.npz"
_C = {}
PAIR = {None: "SELF  cl vs g          (T2 path)", "g_late": "MUT   cl vs g_shift1   (logged future 0.1 s late)", "g_late_knot": "MUT   cl vs g_shift2   (logged future 0.2 s late)",
        "skew1": "MUT   cl_skew1 vs g    (plan rolled one tick)", "yawflip": "MUT   cl vs g_yawflip  (yaw sign flipped)"}


def scored(cases=CASES, tag="stage1"):
    """Process A (executor, tip tree) then process B (scorer, reader tree); cached per case list."""
    key = tuple(cases)
    if key in _C:
        return _C[key]
    d = os.path.join(os.environ.get("T2AUDIT_TMP", "C:/Users/Admin/AppData/Local/Temp"), f"t2audit_{tag}").replace("\\", "/")
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    subprocess.run([PY, "-I", f"{AUD}/selfscore_run.py", f"{d}/in.npz"] + list(cases), check=True, capture_output=True, text=True, env=env)
    subprocess.run([PY, "-I", f"{AUD}/selfscore_score.py", f"{d}/in.npz", f"{d}/out.json"], check=True, capture_output=True, text=True, env=env)
    _C[key] = json.load(open(f"{d}/out.json", encoding="utf-8"))
    return _C[key]


def check_self_score(mut=None):
    """The logged plan, played back by the T2 executor, scores ~0 against the logged future on both NuRec starts (the standstill launch included)."""
    res = scored()
    for r in res:
        p = r["pairs"][PAIR[mut]]
        assert p["max_abs_knot_diff_m"] < 1e-9, f"{r['case']}: executed track differs from the logged future by {p['max_abs_knot_diff_m']:.3g} m"
        assert p["n_nan_rows"] == 0, f"{r['case']}: a row is NaN on the self-score"
        for k, v in p["rows"].items():
            assert abs(v) < 1e-6, f"{r['case']}: {k} = {v:.3g} on a logged-plan self-score (must be ~0)"
    return len(res)


def check_t1_path_equal(mut=None):
    """T2's logged future `g` equals the T1 reader's path on the same logged rows (`exec_in_ego`), bit for bit."""
    res = scored()
    for r in res:
        p = r["pairs"]["XCHK  g vs g_t1        (T2 g against T1 g)" if mut is None else "MUT   g_shift1 vs g_t1 (a g built 0.1 s late)"]
        assert p["max_abs_knot_diff_m"] <= 0.0, f"{r['case']}: g (T2) differs from the T1 reader's path by {p['max_abs_knot_diff_m']:.3g} m"
    return len(res)


def check_launch_class_t1(mut=None):
    """The reader's own BANKED P10 rows (RIC0_s0.npz `b_k`) on standstill-launch windows (v0 < 0.1 m/s, logged 6 s extent > 10 m: n 18) read heading ~61 deg and yaw rate ~283 deg/s
    (medians): the stage-1 launch start (61.4 deg, 283.0 deg/s) is that class, not a scorer defect. mut='moving' reads the v0 >= 2 class instead and must fail."""
    z = np.load(RIC0_ARR, allow_pickle=True)
    labels = [str(q) for q in z["labels"]]
    v0, g = z["v0"], z["g"]
    ext = np.linalg.norm(g[:, -1], axis=1)
    m = (v0 < 0.1) & (ext > 10.0) if mut is None else (v0 >= 2.0)
    n = int(m.sum())
    h = float(np.nanmedian(z[f"b_{labels.index('heading MAE')}"][m]))
    y = float(np.nanmedian(z[f"b_{labels.index('yaw-rate MAE')}"][m]))
    assert n == 18 and 50.0 <= h <= 70.0 and 250.0 <= y <= 300.0, f"banked P10 on the launch class: n {n}, heading median {h:.1f}, yaw-rate median {y:.1f}"
    return n, h, y


def test_self_score():
    assert check_self_score() == 2


def test_t1_path_equal():
    assert check_t1_path_equal() == 2


def test_launch_class_t1():
    check_launch_class_t1()
