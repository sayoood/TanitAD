"""t1blind_dk: distance keeping BESIDE (PREREG_T1_BLIND.md section 6; decides nothing). dk141.py (the pinned instrument, unedited) reads an arm's one-shot plan `pred` [2399, 30, 2] in the window-origin ego frame against the logged lead future.
The T1-kin rollouts dump that plan: the TICK-0 rows of the tick file (`plan_raw` = the model's raw plan before the sg72 smoothing, window-origin frame because the executed pose at tick 0 is the logged pose at t0; row order = the eval grid = gt141's eid order,
checked below). So dk141 here reads the TICK-0 plan of RIC0_s and BLIND1_s from the same tool and flags (both with the causal speed bucket), NOT the executed rollouts, and NOT the eval tool's logged-bucket plans that wsp_edit_d's DK table used:
a BLIND1 - RIC0 delta is comparable only inside this table.
usage: t1blind_dk.py [--out-root F:/.../t1_blind/out] [--raw <dir>/raw] [--dk-dir F:/.../t1_blind/dk] [--arms RIC0,BLIND1]    (CUDA hidden)"""
import os
import sys
import argparse
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
DK = f"{ART}/dagland/motion_c/dk141/dk141.py"
GT = f"{ART}/dagland/motion_c/dk141/gt141.npz"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BANK_OUT = "F:/TanitAD-scratch/motion_c/ri/out"


def build_eval(ticks_path, dst, eid_ref):
    t = np.load(ticks_path, allow_pickle=True)
    m = t["tick"] == 0
    assert int(m.sum()) == len(eid_ref) == 2399 and np.array_equal(t["win"][m], np.arange(2399)), "tick-0 rows are not the 2,399 grid windows in grid order"
    assert np.array_equal(t["ci"][m], eid_ref), "tick-0 clip index differs from gt141's eid"
    pred = np.asarray(t["plan_raw"][m], np.float64)
    assert pred.shape == (2399, 30, 2) and np.isfinite(pred).all()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst, pred=pred, eid=np.asarray(t["ci"][m], np.int64))
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="F:/TanitAD-scratch/motion_c/t1_blind/out")
    ap.add_argument("--raw", default=f"{HERE}/raw")
    ap.add_argument("--dk-dir", default="F:/TanitAD-scratch/motion_c/t1_blind/dk")
    ap.add_argument("--arms", default="RIC0,BLIND1")
    ap.add_argument("--suffix", default="T1BLIND")
    a = ap.parse_args(argv)
    eid_ref = np.load(GT)["eid"]
    arms = a.arms.split(",")
    cmd = [PY, "-I", DK, "read"]
    for arm in arms:
        for s in (0, 1):
            tag = f"{arm}_s{s}"
            root = BANK_OUT if arm == "RIC0" else a.out_root
            dst = build_eval(f"{root}/{tag}/ticks/ticks_{tag}.npz", f"{a.dk_dir}/eval_{tag}.npz", eid_ref)
            cmd += ["--arm", f"{tag}={dst}"]
    for s in (0, 1):
        if "BLIND1" in arms and "RIC0" in arms:
            cmd += ["--pair", f"BLIND1_s{s}:RIC0_s{s}"]
        for arm in arms:
            cmd += ["--pair", f"{arm}_s{s}:P10"]
    cmd += ["--out", f"{a.raw}/DK141_READ_{a.suffix}.json", "--table-md", f"{a.raw}/DK141_TABLE_{a.suffix}.md", "--perwindow-out", f"{a.raw}/dk141_perwindow_{a.suffix}.npz"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    r = subprocess.run(cmd, env=env, cwd=HERE, capture_output=True, text=True)
    open(f"{HERE}/logs/dk141_read_{a.suffix}.out", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    print("dk141 rc", r.returncode, (r.stdout + r.stderr)[-300:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
