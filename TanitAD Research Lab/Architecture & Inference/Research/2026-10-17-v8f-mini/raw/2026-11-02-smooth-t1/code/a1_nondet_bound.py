#!/usr/bin/env python3
"""a1_nondet_bound.py -- AMENDMENT_1 section 3: the measured run-to-run nondeterminism, Y' vs Y'' (two identical --smooth off rollouts of DAG-OACC s0 at 16 threads).

    python a1_nondet_bound.py --out-dir F:/TanitAD-scratch/smooth_t1/out_m16 --raw <package>/raw_a1

Writes <raw>/NONDET_BOUND.json BEFORE the reader computes any row (smooth_t1_read_a1.py refuses without it).
tol_X = max(1e-4, N_X) for X in dump, floor, tick0, ticks, row (AMENDMENT_1.md section 3).  Also exports the helpers the reader shares (extract_run, per_key_dev, load_*).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import smooth_t1_read as R0  # noqa: E402   (the frozen-card reader: functions imported unchanged)

PK = os.path.dirname(HERE)
BASE_TOL = 1e-4
GROSS_FLOOR_GUARD_M = 1.0           # AMENDMENT_1 / READER_FIXES.md fix 7: the extractor refuses only on a gross floor mismatch
Y1, Y2 = "K1OFF_DAGOACC_s0", "OFFREP_DAGOACC_s0"          # Y' and Y'' of DAG-OACC s0


def extract_run(tag, arm, seed, out_dir, arr_dir):
    """The landed extractor (code/extract_smooth.py) on one run of out_dir, with the gross floor guard; skips an existing array file."""
    dst = f"{arr_dir}/{tag}.npz"
    if os.path.exists(dst):
        return dst
    os.makedirs(arr_dir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [R0.PY, "-I", f"{HERE}/extract_smooth.py", "--arm", arm, "--seed", str(seed), "--out", dst, "--dump-dir", f"{out_dir}/{tag}/dump_{tag}", "--ticks", f"{out_dir}/{tag}/ticks/ticks_{tag}.npz",
           "--floor-guard-m", str(GROSS_FLOOR_GUARD_M)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    open(f"{PK}/logs/a1_extract_{tag}.log", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"extract {tag} failed rc={r.returncode}: {r.stderr[-400:]}")
    return dst


def load_ticks(p):
    return {k: v for k, v in dict(np.load(p, allow_pickle=True)).items() if k not in ("tag", "ckpt")}


def per_key_dev(a, b):
    """-> {key: max abs deviation (0.0 when equal) or 'SHAPE' / 'DIFF'} over the keys of two dicts of arrays (float fields: abs deviation; others: equality)."""
    out = {}
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            out[k] = "SHAPE"
        elif x.dtype.kind in "fc":
            if np.array_equal(x, y, equal_nan=True):
                out[k] = 0.0
            else:
                out[k] = float(np.abs(np.where(np.isnan(x) & np.isnan(y), 0.0, x.astype(np.float64) - y.astype(np.float64))).max())
        else:
            out[k] = 0.0 if np.array_equal(x, y) else "DIFF"
    return out


def fmax(d, strict=True):
    """max of the numeric entries.  strict (default): raises on a SHAPE / DIFF entry (fail loud).  strict=False: a SHAPE / DIFF entry counts as +inf (it cannot be within any tolerance) and is listed by nonnumeric()."""
    bad = {k: v for k, v in d.items() if isinstance(v, str)}
    if bad and strict:
        raise SystemExit(f"UNVERIFIED: non-numeric deviation entries {bad}")
    return float("inf") if bad else (max(d.values()) if d else 0.0)


def nonnumeric(d):
    return sorted(k for k, v in d.items() if isinstance(v, str))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    t_start = time.time()
    arr_dir = f"{a.raw}/arrays"
    os.makedirs(a.raw, exist_ok=True)
    # gate: both hashes
    want = [ln.split()[2] for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if ln.startswith("PREREG_FROZEN.md sha256")][0]
    want2 = [ln.split()[2] for ln in open(f"{PK}/AMENDMENT_1_SHA.txt", encoding="utf-8") if ln.startswith("AMENDMENT_1.md sha256")][0]
    if R0.sha256_file(f"{PK}/PREREG_FROZEN.md") != want or R0.sha256_file(f"{PK}/AMENDMENT_1.md") != want2:
        print("REFUSED: card or amendment does not hash to its frozen sha256", file=sys.stderr)
        return 2
    GD = R0.load_gd()
    # dump and ticks, Y' vs Y''
    n1, d1 = R0.load_dump(f"{a.out_dir}/{Y1}/dump_{Y1}")
    n2, d2 = R0.load_dump(f"{a.out_dir}/{Y2}/dump_{Y2}")
    t1, t2 = load_ticks(f"{a.out_dir}/{Y1}/ticks/ticks_{Y1}.npz"), load_ticks(f"{a.out_dir}/{Y2}/ticks/ticks_{Y2}.npz")
    kd, kt = per_key_dev(d1, d2), per_key_dev(t1, t2)
    if set(d1) != set(d2) or set(t1) != set(t2) or n1 != n2:
        raise SystemExit("UNVERIFIED: Y' and Y'' dump / tick key sets or file counts differ")
    k0a, k0b = t1["tick"] == 0, t2["tick"] == 0
    if not (np.array_equal(t1["win"], t2["win"]) and np.array_equal(t1["tick"], t2["tick"])):
        raise SystemExit("UNVERIFIED: Y' and Y'' ticks are not tick-aligned")
    n_tick0 = float(np.abs(t1["plan"][k0a] - t2["plan"][k0b]).max())
    N = {"N_dump": fmax(kd), "N_floor": float(kd["fl"]), "N_tick0": n_tick0, "N_ticks": fmax(kt)}
    # rows: Y' - Y'' on the 19 rows (the same machinery as every row), after extracting both runs
    p1, p2 = extract_run(Y1, "CTRL", 0, a.out_dir, arr_dir), extract_run(Y2, "CTRL", 0, a.out_dir, arr_dir)
    z1, z2 = np.load(p1, allow_pickle=True), np.load(p2, allow_pickle=True)
    yb = np.load(f"{R0.GD_DIR}/raw/arrays/CTRL_s0.npz", allow_pickle=True)
    cls, _m = GD.classify(yb["v0"], yb["g"])
    labels = tuple(str(x) for x in yb["labels"])
    rows = {}
    for i, lab in enumerate(labels):
        r = GD.decompose(z1[f"a_{i}"], z2[f"a_{i}"], yb["eid"], cls, GD.POOL[lab], a.n_boot)
        m = np.isfinite(z1[f"a_{i}"]) & np.isfinite(z2[f"a_{i}"])
        rows[lab] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "same_nan_pattern": bool(np.array_equal(np.isfinite(z1[f"a_{i}"]), np.isfinite(z2[f"a_{i}"]))),
                     "per_window_max_abs_diff": float(np.abs(z1[f"a_{i}"][m] - z2[f"a_{i}"][m]).max()) if m.any() else 0.0}
    N["N_row"] = max(abs(v["delta"]) for v in rows.values())
    tol = {k.replace("N_", "tol_"): max(BASE_TOL, v) for k, v in N.items()}
    out = {"amendment": "AMENDMENT_1 section 3", "tier": "T1-kin", "evidence_class": "MEASURED (ours)", "threads": 16, "Y_prime": Y1, "Y_double_prime": Y2, "base_tol": BASE_TOL,
           "estimator_of_rows": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0 (gapdecomp.decompose)", "N": N, "tol": tol,
           "bitwise_identical_dump": bool(all(v == 0.0 for v in kd.values())), "bitwise_identical_ticks": bool(all(v == 0.0 for v in kt.values())),
           "dump_per_key_max_abs_dev": kd, "ticks_per_key_max_abs_dev": {k: v for k, v in kt.items() if v != 0.0}, "n_tick_float_fields_differing": sum(1 for v in kt.values() if v != 0.0),
           "n_ticks_with_plan_dev_gt_1e-4": int((np.abs(t1["plan"] - t2["plan"]).reshape(len(t1["plan"]), -1).max(1) > 1e-4).sum()), "n_ticks": int(len(t1["plan"])),
           "rows_Yp_minus_Ypp": rows, "same_nan_all_rows": bool(all(v["same_nan_pattern"] for v in rows.values())), "wall_s": round(time.time() - t_start, 1),
           "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    dst = f"{a.raw}/NONDET_BOUND.json"
    json.dump(out, open(dst, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("NONDET BOUND", json.dumps({"N": N, "tol": tol, "bitwise_identical_dump": out["bitwise_identical_dump"], "bitwise_identical_ticks": out["bitwise_identical_ticks"]}))
    print("WROTE", dst, "sha256", R0.sha256_file(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
