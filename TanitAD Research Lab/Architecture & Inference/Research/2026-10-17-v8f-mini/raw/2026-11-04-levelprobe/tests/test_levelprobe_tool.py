"""Tests of the LEVELPROBE tool option (``--histswap cv_t0`` / ``cv_pred`` / ``--level-file``) on a REAL checkpoint and 2 real eval-141 windows (DAG-OACC s0; <= 4 threads; CPU).
  T1  the default path (no flag) of the new tool is BYTE-IDENTICAL to the shipped HISTSWAP-LF tool's (dump arrays, ticks arrays, result JSON minus wall clock);
  T2  ``--histswap cv_logged`` is byte-identical too (an existing mode is untouched);
  T3  cv_t0: v0_in equals the window's t0 speed at EVERY tick and differs from the executed speed on later ticks; its tick-0 plan equals cv_logged's (same speed at tick 0);
  T4  cv_pred with a constant level file (3.0): v0_in == 3.0 at every tick;
  T5  cv_pred whose level file is the logged t0 speed of each window reproduces cv_t0 BIT FOR BIT (the two modes differ only in where the constant comes from);
  T6  refusals: cv_pred without a level file; a level file with another mode; a window missing from the file; a negative level;
  T7  MUTATION: a copy of the new tool whose cv_t0 branch reads the speed at the tick (not at t0) must FAIL the T3 assertion (the check can go RED).
Run: python -I -m pytest -q -p no:cacheprovider tests/test_levelprobe_tool.py   (env LP_TOOL_NEW, LP_TOOL_OLD give the two tool paths; they must live in repo trees)."""
import json
import os
import shutil
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pytest  # noqa: E402

S = "F:/TanitAD-scratch/levelprobe_scratch"
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
NEW = os.environ.get("LP_TOOL_NEW", f"{S}/tree_dag/stack/scripts/v8f_t1_eval_levelprobe.py")
OLD = os.environ.get("LP_TOOL_OLD", f"{S}/tree_dag/stack/scripts/v8f_t1_eval_histswap_lf.py")
CK = f"{ART}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt"
T = f"{S}/test_runs"
ENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")


def run(tool, tag, extra, expect_rc=0):
    out = f"{T}/{tag}"
    if os.path.isdir(out):
        shutil.move(out, f"{T}/_old_{tag}_{len(os.listdir(T))}")
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "-I", tool, "run", "--ckpt", CK, "--tag", tag, "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache", "--eval-grid", f"{DATA}/eval_grid_windows.json",
           "--eval-join", f"{DATA}/join_eval141.npz", "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", f"{out}/ticks",
           "--dump-heads", "--device", "cpu", "--max-windows", "2", "--n-boot", "20", "--smooth", "sg72"] + list(extra)
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    assert r.returncode == expect_rc, (tag, r.returncode, (r.stdout + r.stderr)[-600:])
    return out, r


def load(out, tag):
    import glob
    dump = {}
    for f in sorted(glob.glob(f"{out}/dump_{tag}/ep*.npz")):
        z = np.load(f, allow_pickle=True)
        dump[os.path.basename(f)] = {k: z[k] for k in z.files}
    tk = {k: v for k, v in np.load(f"{out}/ticks/ticks_{tag}.npz", allow_pickle=True).items() if k not in ("tag", "ckpt")}
    js = json.load(open(f"{out}/t1v8f_{tag}.json", encoding="utf-8"))
    return dump, tk, js


def same_arrays(a, b):
    if set(a) != set(b):
        return False
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) if np.asarray(a[k]).dtype.kind in "fiub" else np.array_equal(a[k], b[k]) for k in a)


def strip_clock(js, tag="~"):
    s = json.dumps(js, sort_keys=True, default=str).replace(tag, "TAG").replace("\\\\", "/")
    d = json.loads(s)
    for k in list(d):
        if "clock" in k or "wall" in k or k.endswith("_s"):
            d.pop(k)
    return d


@pytest.fixture(scope="module")
def runs():
    os.makedirs(T, exist_ok=True)
    R = {}
    for name, tool, extra in (("OLD_default", OLD, []), ("NEW_default", NEW, []), ("OLD_logged", OLD, ["--histswap", "cv_logged"]), ("NEW_logged", NEW, ["--histswap", "cv_logged"]),
                              ("NEW_t0", NEW, ["--histswap", "cv_t0"])):
        out, _ = run(tool, name, extra)
        R[name] = load(out, name)
    return R


def test_t1_default_path_byte_identical(runs):
    (da, ta, ja), (db, tb, jb) = runs["OLD_default"], runs["NEW_default"]
    assert da.keys() == db.keys() and all(same_arrays(da[k], db[k]) for k in da)
    assert same_arrays(ta, tb)
    assert strip_clock(ja["t1v8f"], "OLD_default") == strip_clock(jb["t1v8f"], "NEW_default")


def test_t2_existing_cv_logged_untouched(runs):
    (da, ta, ja), (db, tb, jb) = runs["OLD_logged"], runs["NEW_logged"]
    assert da.keys() == db.keys() and all(same_arrays(da[k], db[k]) for k in da)
    assert same_arrays(ta, tb)


def check_t0(tk, dump):
    """v0_in == the window's t0 speed at every tick; and, on later ticks, != the executed speed on at least one window tick."""
    win, tick, v_in, v_ex = tk["win"], tk["tick"], tk["v0_in"].astype(np.float64), tk["v0"].astype(np.float64)
    ok_const, n_diff = True, 0
    for w in np.unique(win):
        m = win == w
        v_t0 = v_ex[m & (tick == 0)][0]
        ok_const &= bool(np.abs(v_in[m] - v_t0).max() <= 1e-4)
        n_diff += int((np.abs(v_in[m & (tick >= 1)] - v_ex[m & (tick >= 1)]) > 0.05).sum())
    return ok_const, n_diff


def test_t3_cv_t0_is_constant_at_t0_speed(runs):
    d, tk, js = runs["NEW_t0"]
    ok, n_diff = check_t0(tk, d)
    assert ok and n_diff > 0, (ok, n_diff)
    tl = runs["NEW_logged"][1]
    x0 = tk["tick"] == 0
    assert np.abs(tk["plan_raw"][x0] - tl["plan_raw"][tl["tick"] == 0]).max() <= 1e-4          # tick 0: logged speed at t0 == t0 speed
    assert str(js["t1v8f"]["histswap"]["mode"]) == "cv_t0"


def _levels_file(path, ci, ws, level):
    np.savez(path, clip_index=np.asarray(ci, np.int64), ws=np.asarray(ws, np.int64), level=np.asarray(level, np.float64))
    return path


def test_t4_t5_cv_pred(runs):
    d, tk, js = runs["NEW_t0"]
    ci_t = tk["ci"][tk["tick"] == 0]
    ws_t = tk["t0"][tk["tick"] == 0]
    lv_t0 = tk["v0"][tk["tick"] == 0].astype(np.float64)
    f3 = _levels_file(f"{T}/levels_const3.npz", ci_t, ws_t, np.full(len(ci_t), 3.0))
    out3, _ = run(NEW, "NEW_pred3", ["--histswap", "cv_pred", "--level-file", f3])
    _, t3, _ = load(out3, "NEW_pred3")
    assert np.abs(t3["v0_in"].astype(np.float64) - 3.0).max() <= 1e-4                           # T4
    fr = _levels_file(f"{T}/levels_t0speed.npz", ci_t, ws_t, lv_t0)
    outr, _ = run(NEW, "NEW_predT0", ["--histswap", "cv_pred", "--level-file", fr])
    dr, tr_, _ = load(outr, "NEW_predT0")
    assert dr.keys() == d.keys() and all(same_arrays(dr[k], d[k]) for k in dr), "cv_pred(level = t0 speed) must reproduce cv_t0's dump bit for bit"       # T5
    num = [k for k in tk if k != "histswap" and np.asarray(tk[k]).dtype.kind in "fiub"]
    assert all(np.array_equal(np.asarray(tr_[k]), np.asarray(tk[k]), equal_nan=True) for k in num), "cv_pred(level = t0 speed) must reproduce cv_t0's ticks bit for bit"


def test_t6_refusals(runs):
    tk = runs["NEW_t0"][1]
    ci_t, ws_t = tk["ci"][tk["tick"] == 0], tk["t0"][tk["tick"] == 0]
    for tag, extra, f in (("REF_nofile", ["--histswap", "cv_pred"], None),
                          ("REF_wrongmode", ["--histswap", "cv_t0", "--level-file", f"{T}/levels_const3.npz"], None),
                          ("REF_missing", ["--histswap", "cv_pred", "--level-file", _levels_file(f"{T}/levels_missing.npz", ci_t[:1], ws_t[:1], [3.0])], None),
                          ("REF_negative", ["--histswap", "cv_pred", "--level-file", _levels_file(f"{T}/levels_neg.npz", ci_t, ws_t, np.full(len(ci_t), -1.0))], None)):
        out = f"{T}/{tag}"
        cmd = [PY, "-I", NEW, "run", "--ckpt", CK, "--tag", tag, "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache", "--eval-grid", f"{DATA}/eval_grid_windows.json",
               "--eval-join", f"{DATA}/join_eval141.npz", "--out-dir", out, "--limit", "causal", "--device", "cpu", "--max-windows", "2", "--n-boot", "20", "--smooth", "sg72"] + extra
        r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
        assert r.returncode != 0, (tag, "must be refused", r.stdout[-300:])


def test_t7_mutation_goes_red(runs):
    src = open(NEW, encoding="utf-8").read()
    old = 'v = float(self.base_log["p10f"][self.t0][0])                  # the LOGGED measured speed at t0, held constant'
    assert src.count(old) == 1
    mut = src.replace(old, 'v = float(self.base_log["p10f"][j][0])')
    mp = os.path.join(os.path.dirname(NEW), "v8f_t1_eval_levelprobe_MUT.py")
    open(mp, "w", encoding="utf-8", newline="\n").write(mut)
    try:
        out, _ = run(mp, "MUT_t0", ["--histswap", "cv_t0"])
        _, tk, _ = load(out, "MUT_t0")
        ok, n_diff = check_t0(tk, None)
        assert not ok, "the mutated cv_t0 (speed read at the tick) must FAIL the constant-at-t0 check"
    finally:
        os.replace(mp, f"{T}/v8f_t1_eval_levelprobe_MUT.py.moved") if os.path.exists(mp) else None
