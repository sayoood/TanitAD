"""D-REFCV6-EQUALIZE-DROPPED / SPEC_REFCV7 FIX-3 integration check (CPU only, no GPU), SPLIT so that it fits
a shared dev box (Master Mind 2026-09-26 22:08): every arm is its OWN process that exits and frees its model;
the comparison is a fourth tiny process.

usage:
  python eq_as_trained_check.py roll --arm {R,A,B} <out_dir> [--pre-tree ..] [--post-tree ..] [--ckpt ..]
  python eq_as_trained_check.py compare <out_dir>

  R  the PRE-fix tree (the run's own code)                             -> the as-trained reference
  A  the POST-FIX tree, battery loader as shipped (override active)     -> must equal R BIT FOR BIT
  B  the POST-FIX tree, override replaced by a no-op (trunk as DECLARED) -> must DIFFER from R

RAM floors (both measured with GlobalMemoryStatusEx, box-wide FREE physical memory):
  * a roll STARTS only after >= 7.5 GB free on 3 consecutive samples (10 s apart);
  * while it runs, the wrapper polls every 5 s and KILLS its own child if free RAM falls below 4.0 GB,
    recording ABORTED_LOW_RAM (retried later by the runner), so it never pushes another stream under
    its guard (E1's scorer aborts after 120 s below 3 GB).
Each roll: DDIM draw = 0, CPU fp32 trunk, 4 eval windows over 2 clips of a PRE-fix checkpoint, outputs saved
under <out_dir>/<arm>/. `compare` writes <out_dir>/eq_as_trained_record.json (verdict PASS / FAIL).
"""
import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
#: the run's PRE-switch config (argv of refcv6-r101-s0 as launched); EQ_CONFIG overrides it (Thor: the kit copy, shipped)
KIT_CONFIG = os.environ.get("EQ_CONFIG", "D:/refcv6_eval_kit/ckpt/config.json")
#: the Master Mind's floors are the MINIMUM (start >= 7.5 GB x3, abort < 4.0 GB). MEASURED 22:11: one arm with
#: 4 windows and 6 threads took the box from 7.98 to 3.83 GB free (>= 4.15 GB) and aborted itself, so the arm now
#: rolls ONE window on 2 threads and starts only at >= 8.5 GB (stricter than 7.5, never looser).
START_FLOOR_GB, START_SAMPLES, ABORT_FLOOR_GB = 8.5, 3, 4.0
#: dev box: ONE window on 2 threads (RAM); Thor (Master Mind 23:14): EQ_N_CLIPS=2 EQ_N_WIN=2 EQ_THREADS=4
N_CLIPS, N_WIN_PER_CLIP = int(os.environ.get("EQ_N_CLIPS", "1")), int(os.environ.get("EQ_N_WIN", "1"))
THREADS = os.environ.get("EQ_THREADS", "2")
KEYS = ("os", "oracle_sel", "ha0_ext", "g", "extras.nav_true.traj", "extras.nav_true.tacv6_lat_logits",
        "extras.nav_true.tacv6_goal_logits", "extras.nav_true.sel_idx")
ARMS = {"R": ("pre", "as_trained"), "A": ("post", "as_trained"), "B": ("post", "skip")}


class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_gb() -> float:
    """Box-wide free physical memory: GlobalMemoryStatusEx on Windows, MemAvailable on Linux (Thor)."""
    if os.name != "nt":
        for line in open("/proc/meminfo", encoding="ascii"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
        raise SystemExit("[eq] /proc/meminfo has no MemAvailable")
    m = _MEMSTAT()
    m.dwLength = ctypes.sizeof(_MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def wait_start_floor(max_wait_s: int) -> dict:
    t0, ok = time.time(), 0
    while time.time() - t0 < max_wait_s:
        ok = ok + 1 if free_gb() >= START_FLOOR_GB else 0
        if ok >= START_SAMPLES:
            return {"status": "floor held", "free_gb": round(free_gb(), 2), "waited_s": round(time.time() - t0)}
        time.sleep(10)
    return {"status": "floor never held", "waited_s": round(time.time() - t0)}


def do_roll(a) -> int:
    out = Path(a.out_dir) / a.arm
    out.mkdir(parents=True, exist_ok=True)
    rec = {"arm": a.arm, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rec["start_floor"] = wait_start_floor(a.max_wait_s)
    if rec["start_floor"]["status"] != "floor held":
        rec["status"] = "SKIPPED_NO_RAM"
        json.dump(rec, open(out / "arm_record.json", "w", encoding="utf-8"), indent=1)
        return 3
    tree = a.pre_tree if ARMS[a.arm][0] == "pre" else a.post_tree
    mode = ARMS[a.arm][1]
    wj = Path(a.out_dir) / "windows.PRIVATE.json"
    if not wj.exists():
        w = json.load(open(HERE.parent / "raw" / "step5000" / "windows_s0.json", encoding="utf-8"))
        json.dump({c: sorted(w[c])[:N_WIN_PER_CLIP] for c in sorted(w)[:N_CLIPS]}, open(wj, "w", encoding="utf-8"))
    env = dict(os.environ, REFCV6_REPO=tree, PYTHONPATH=os.pathsep.join([f"{tree}/stack", f"{tree}/taniteval"]),
               OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="")
    # the A6 train override must never leak into this check; an EXPLICIT one for this check is kept
    #: (Thor: every remapped flag pointed back at Thor's own data paths -- EQ_REMAP_OVERRIDES)
    env.pop("REFCV6_REMAP_OVERRIDES", None)
    if os.environ.get("EQ_REMAP_OVERRIDES"):
        env["REFCV6_REMAP_OVERRIDES"] = os.environ["EQ_REMAP_OVERRIDES"]
    for f in (out / "roll.json",):
        if f.exists():
            f.unlink()
    cmd = [sys.executable, str(HERE / "eq_as_trained_roll.py"), "--mode", mode, "--ckpt", a.ckpt,
           "--config", KIT_CONFIG, "--seed", "0", "--dump-dir", str(out / "dump"), "--windows-json", str(wj),
           "--device", "cpu", "--out-json", str(out / "roll.json"), "--smoke-cpu-fp32-trunk"]
    rec.update({"tree": tree, "mode": mode, "ckpt": a.ckpt})
    min_free = 1e9
    with open(out / "roll.log", "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        while p.poll() is None:
            fg = free_gb()
            min_free = min(min_free, fg)
            if fg < ABORT_FLOOR_GB:
                p.kill()                                        # our OWN child, by handle
                p.wait()
                rec.update({"status": "ABORTED_LOW_RAM", "free_gb_at_abort": round(fg, 2)})
                break
            time.sleep(5)
    rec["min_free_gb_during_roll"] = round(min_free, 2)
    if "status" not in rec:
        rec["status"] = "DONE" if (out / "roll.json").exists() else "NO_ROLL_RECORD"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(rec, open(out / "arm_record.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rec))
    return 0 if rec["status"] == "DONE" else 2


def load_arm(d: Path):
    arrs = {}
    for f in sorted(glob.glob(str(d / "dump" / "ep*.npz"))):
        z = np.load(f)
        for k in z.files:
            arrs.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v) for k, v in arrs.items() if v[0].ndim}
    ex = np.load(d / "dump" / "refcv6_extras.npz")
    out.update({f"extras.{k}": ex[k] for k in ex.files})
    return out, json.load(open(d / "roll.json", encoding="utf-8"))


def do_compare(a) -> int:
    root = Path(a.out_dir)
    rec = {"tool": "eq_as_trained_check.py compare", "defect": "D-REFCV6-EQUALIZE-DROPPED / SPEC_REFCV7 FIX-3",
           "device": "cpu, fp32 trunk, DDIM draw = 0", "n_windows": N_CLIPS * N_WIN_PER_CLIP, "n_clips": N_CLIPS,
           "floors_gb": {"start": START_FLOOR_GB, "start_samples": START_SAMPLES, "abort": ABORT_FLOOR_GB},
           "arms": {x: json.load(open(root / x / "arm_record.json", encoding="utf-8")) for x in ARMS}}
    (R, rR), (A, rA), (B, rB) = (load_arm(root / x) for x in ("R", "A", "B"))
    rec["trunk_records"] = {x: r["model_record"].get("trunk_equalize_as_trained") for x, r in
                            (("R", rR), ("A", rA), ("B", rB))}
    rec["strict_load"] = {x: {"missing": r["model_record"]["state_dict"].get("missing"),
                              "unexpected": r["model_record"]["state_dict"].get("unexpected")}
                          for x, r in (("R", rR), ("A", rA), ("B", rB))}
    cmpA, cmpB = {}, {}
    for key in KEYS:
        if key in R and key in A:
            cmpA[key] = bool(np.array_equal(A[key], R[key], equal_nan=True))
        if key in R and key in B:
            same = bool(np.array_equal(B[key], R[key], equal_nan=True))
            cmpB[key] = {"bit_identical": same,
                         "max_abs_diff": (float(np.nanmax(np.abs(B[key].astype(np.float64) - R[key].astype(np.float64))))
                                          if not same and B[key].dtype.kind == "f" else None)}
    rec["A_equals_R"], rec["B_vs_R"] = cmpA, cmpB
    tA = rec["trunk_records"]["A"] or {}
    checks = {"A_bit_identical_to_R_on_every_key": bool(cmpA) and all(cmpA.values()),
              "B_differs_from_R_on_traj": not cmpB.get("extras.nav_true.traj", {}).get("bit_identical", True),
              "A_record_set_as_trained_43_to_0": (tA.get("status"), tA.get("declared_by_pin"), tA.get("as_trained"))
              == ("set as trained", 43, 0),
              "R_record_pre_fix": str((rec["trunk_records"]["R"] or {}).get("status", "")).startswith("pre-FIX-3"),
              "B_record_skipped": str((rec["trunk_records"]["B"] or {}).get("status", "")).startswith("SKIPPED"),
              "strict_load_clean": all(v["missing"] == [] and v["unexpected"] == [] for v in rec["strict_load"].values())}
    rec["checks"] = checks
    rec["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(rec, open(root / "eq_as_trained_record.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({"verdict": rec["verdict"], "checks": checks}, indent=1))
    return 0 if rec["verdict"] == "PASS" else 2


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roll")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("out_dir")
    r.add_argument("--pre-tree", default="C:/Users/Admin/ev6")
    r.add_argument("--post-tree", default="C:/Users/Admin/ev6_postfix")
    r.add_argument("--ckpt", default="D:/refcv6_eval_kit/ckpt/ckpt_30000.pt")
    r.add_argument("--max-wait-s", type=int, default=6 * 3600)
    c = sub.add_parser("compare")
    c.add_argument("out_dir")
    a = ap.parse_args()
    raise SystemExit(do_roll(a) if a.cmd == "roll" else do_compare(a))


if __name__ == "__main__":
    main()
