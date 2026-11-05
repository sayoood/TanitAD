#!/usr/bin/env python3
"""ka_new.py -- KA-NEW, the known answer of the NEW read tree (must pass before any CL read).

The C0 s0 checkpoint (M2b_s0_recft_C0, sha256 from SHA256SUMS.txt) is re-rolled with the RI tool of the NEW tree (F:/TanitAD-scratch/motion_c/cl_read/tree: level-head stack a3634276 / 9ffdf1b4)
on --scene logged --smooth sg72 over a 64-window evenly spaced subset (--max-windows 64, the tool's own subset rule, the smokes' pattern), with the reader's flags, and compared with the BANKED C0_s0 rollout
(C:/Users/Admin/rr_scratch/out/C0_s0, read only) on the SAME windows, matched by (clip index, t0, tick):
  ticks   plan, plan_raw, tau, P (every replan row)      dumps  cl (executed path), ol (tick-0 plan), fl (P10 floor), g, v0   (matched by (clip_index, ws))
PASS = every matched plan array within TOL = 1e-4 m (the readers' TOL) AND every window of the subset found in the banked rollout.  Writes raw/KA_NEW.json and logs/ka_new.out.
Attribution options (same script): --tree / --tag / --out-dir / --ref-ticks / --ref-dump / --result-name re-point the run and the reference (e.g. the OLD read tree, or another thread count) so a difference can be
attributed to the code, to the thread count, or to neither.
usage: python -I ka_new.py [--threads 12] [--max-windows 64] [--skip-run]"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
RESULT = "KA_NEW"
SCENE, FEAS = "logged", "off"
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TREE = "F:/TanitAD-scratch/motion_c/cl_read/tree"
CKD = f"{ART}/v8f-recover/M2b_s0_recft_C0"
BANK = "C:/Users/Admin/rr_scratch/out/C0_s0"
OUT = "F:/TanitAD-scratch/motion_c/cl_read/smoke/KA_NEW"
TAG = "KANEW_C0_s0"
TOOL = f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"
REF_TICKS, REF_DUMP = f"{BANK}/ticks/ticks_C0_s0.npz", f"{BANK}/dump_C0_s0"
TOL = 1e-4


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_tool(threads, n_win, first=False):
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), PYTHONDONTWRITEBYTECODE="1")
    cmd = ([PY, "-I", f"{HERE}/run_tool_subset.py", "--tool", TOOL, "--first", str(n_win), "--"] if first else [PY, "-I", TOOL]) + ["run", "--ckpt", f"{CKD}/ckpt_last.pt", "--tag", TAG,
           "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache", "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz",
           "--out-dir", OUT, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal", "--smooth", "sg72", "--scene", SCENE,
           "--dump-ticks", f"{OUT}/ticks", "--dump-heads", "--device", "cpu"] + ([] if first else ["--max-windows", str(n_win)]) + ["--n-boot", "20"] + (["--feasible", FEAS] if FEAS != "off" else [])
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    open(f"{PK}/logs/{RESULT}_run.out", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    if r.returncode:
        print("KANEW RUN FAILED rc", r.returncode, r.stderr[-1500:])
        sys.exit(1)
    print(f"{RESULT} run: {time.time() - t0:.0f} s, {threads} threads, {n_win} windows")


def dumps(d):
    o = {}
    for f in sorted(glob.glob(f"{d}/ep*.npz")):
        e = np.load(f)
        for k in e.files:
            o.setdefault(k, []).append(e[k])
    return {k: np.concatenate(v) for k, v in o.items()}


def main():
    global TOOL, TAG, OUT, RESULT, SCENE, FEAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--first", action="store_true", help="roll the FIRST N windows in-process (run_tool_subset.py: the banked batches) instead of the tool's evenly spaced --max-windows subset")
    ap.add_argument("--scene", default="logged")
    ap.add_argument("--feasible", default="off")
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--ref-ticks", default=REF_TICKS)
    ap.add_argument("--ref-dump", default=REF_DUMP)
    ap.add_argument("--result-name", default="KA_NEW")
    a = ap.parse_args()
    TOOL, TAG, OUT, RESULT, SCENE, FEAS = f"{a.tree}/stack/scripts/v8f_t1_eval_ri.py", a.tag, a.out_dir, a.result_name, a.scene, a.feasible
    want = [l.split()[0] for l in open(f"{CKD}/SHA256SUMS.txt") if l.strip().endswith("ckpt_last.pt")][0]
    got = sha(f"{CKD}/ckpt_last.pt")
    assert got == want, ("checkpoint sha256", got, want)
    print(f"{RESULT} checkpoint sha256 {got} == SHA256SUMS.txt line; tool sha256 {sha(TOOL)}")
    if not a.skip_run:
        run_tool(a.threads, a.max_windows, a.first)
    zn = {k: v for k, v in np.load(f"{OUT}/ticks/ticks_{TAG}.npz", allow_pickle=True).items()}
    zb = {k: v for k, v in np.load(a.ref_ticks, allow_pickle=True).items()}
    key = lambda z: {(int(c), int(t), int(k)): i for i, (c, t, k) in enumerate(zip(z["ci"], z["t0"], z["tick"]))}
    kn, kb = key(zn), key(zb)
    missing = [k for k in kn if k not in kb]
    idx = [(kn[k], kb[k]) for k in kn if k in kb]
    inew, ibank = np.array([i for i, _ in idx]), np.array([j for _, j in idx])
    res = {"card": "KA-NEW (CL read package): the new tree re-rolls C0 s0", "tol_m": TOL, "n_windows_requested": a.max_windows, "subset": "first N windows in the grid order (banked batches)" if a.first else "evenly spaced (tool --max-windows)", "n_tick_rows_new": len(kn), "n_tick_rows_matched": len(idx),
           "n_windows_new": len({(c, t) for c, t, _ in kn}), "rows_missing_in_banked": len(missing), "ckpt_sha256": got, "tool": TOOL, "tool_sha256": sha(TOOL), "threads": a.threads, "scene": SCENE, "feasible": FEAS, "reference_ticks": a.ref_ticks, "reference_dump": a.ref_dump, "ticks": {}, "dumps": {}}
    for nm in ("plan", "plan_raw", "tau", "P"):
        d = np.abs(zn[nm][inew].astype(np.float64) - zb[nm][ibank].astype(np.float64))
        res["ticks"][nm] = {"max_abs_m": float(d.max()), "median_abs_m": float(np.median(d)), "n_rows_over_tol": int((d.reshape(len(d), -1).max(1) > TOL).sum()), "bit_identical_rows": int((d.reshape(len(d), -1).max(1) == 0).sum())}
    for nm in ("gt", "v0", "age_t", "age_s", "pose", "s_ckpt", "t_goal_prob", "t_lat_logits", "t_lon_logits", "t_cons_pred"):
        if nm in zn and nm in zb:
            d = np.abs(zn[nm][inew].astype(np.float64) - zb[nm][ibank].astype(np.float64))
            res["ticks"][nm] = {"max_abs": float(d.max())}
    dn, db = dumps(f"{OUT}/dump_{TAG}"), dumps(a.ref_dump)
    kdn = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(dn["clip_index"], dn["ws"]))}
    kdb = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(db["clip_index"], db["ws"]))}
    dm = [(kdn[k], kdb[k]) for k in kdn if k in kdb]
    res["n_dump_windows_new"], res["n_dump_windows_matched"] = len(kdn), len(dm)
    jn, jb = np.array([i for i, _ in dm]), np.array([j for _, j in dm])
    for nm in ("cl", "ol", "fl", "g", "v0"):
        d = np.abs(dn[nm][jn].astype(np.float64) - db[nm][jb].astype(np.float64))
        res["dumps"][nm] = {"max_abs": float(d.max()), "bit_identical_windows": int((d.reshape(len(d), -1).max(1) == 0).sum()), "n_windows": int(len(d))}
    plan_max = max(res["ticks"][n]["max_abs_m"] for n in ("plan", "plan_raw", "tau", "P"))
    dump_max = max(res["dumps"][n]["max_abs"] for n in ("cl", "ol", "fl"))
    res["max_plan_diff_m"], res["max_dump_diff_m"] = plan_max, dump_max
    res["PASS"] = bool(not missing and len(dm) == len(kdn) and plan_max <= TOL and dump_max <= TOL)
    res["clock"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(res, open(f"{PK}/raw/{RESULT}.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"{RESULT} ticks:", json.dumps(res["ticks"]))
    print(f"{RESULT} dumps:", json.dumps(res["dumps"]))
    print(f"{RESULT} windows {res['n_windows_new']}, tick rows {len(kn)} matched {len(idx)}, dump windows {len(kdn)} matched {len(dm)}; max |plan diff| {plan_max:.3e} m (plan/plan_raw/tau/P), max |dump diff| {dump_max:.3e} m; TOL {TOL}")
    print(f"{RESULT} PASS" if res["PASS"] else f"{RESULT} FAIL")
    sys.exit(0 if res["PASS"] else 2)


if __name__ == "__main__":
    main()
