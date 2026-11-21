"""Control v3 of PREREG_V8F_PROOF_ADD37_N6 (identity): with --w-act-level at 0 the loss of the training steps of the NEW tree equals that of the OLD tree exactly.

Usage: v3_identity.py <old_tree_root> <new_tree_root> [out_json] [python_exe]
Each tree root is the folder that holds stack/ (and taniteval/). The trainer of each tree is run as its own subprocess (the two trees both carry a package named tanitad, so they cannot share
one interpreter) on the CPU, tiny synthetic world, same seed, same batch order, 4 steps, --w-act-cons 0.001, --log-every 1. The new tree runs twice: once without the level flag and once with an
explicit --w-act-level 0.0. Compared exactly: every numeric key of every train-log row except wall-clock keys. Prints one JSON line with pass and max_abs_diff; writes the same dict (with details) to out_json.
Also checks that the new tree with --w-act-level 0.01 does move the total (control: the check can read non-zero) and that the V = 0 rows hold no act_level key."""
from __future__ import annotations

import json
import os
import subprocess
import sys

OLD, NEW = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
PY = sys.argv[4] if len(sys.argv) > 4 else sys.executable
TIME_KEYS = ("time", "step_s", "elapsed", "wall", "sec", "ips", "per_s", "eta")
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--exclude", "1:3"]
BASE = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1", "--lead-sup", "on", "--w-lead-sup", "1.0",
        "--lr", "1e-4", "--warmup", "100", "--seed", "0", "--steps", "4", "--log-every", "1", "--ckpt-every", "100", "--w-act-cons", "0.001"]


def run(root, tag, extra):
    root = os.path.abspath(root)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "tmp", "v3", tag)
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(root, "stack"), os.path.join(root, "taniteval"), os.path.join(root, "stack", "scripts")])
    cmd = [PY, os.path.join(root, "stack", "scripts", "v8f_mini_train.py"), *G, *BASE, *extra, "--out", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        return None, (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:]
    rows = [json.loads(line) for line in open(os.path.join(out, "train_log.jsonl"), encoding="utf-8")]
    return rows, ""


def numeric(row):
    return {k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and not any(t in k.lower() for t in TIME_KEYS)}


def compare(a, b):
    worst, bad = 0.0, []
    if len(a) != len(b):
        return float("inf"), ["row count"]
    for i, (x, y) in enumerate(zip(a, b)):
        nx, ny = numeric(x), numeric(y)
        if set(nx) != set(ny):
            bad.append(f"row {i}: key sets differ {sorted(set(nx) ^ set(ny))}")
            continue
        for k in nx:
            dlt = abs(nx[k] - ny[k])
            worst = max(worst, dlt)
            if dlt != 0.0:
                bad.append(f"row {i} {k}: {nx[k]!r} vs {ny[k]!r}")
    return worst, bad


res = {"old_tree": os.path.abspath(OLD), "new_tree": os.path.abspath(NEW)}
old, e0 = run(OLD, "old", [])
new, e1 = run(NEW, "new_default", [])
new0, e2 = run(NEW, "new_explicit0", ["--w-act-level", "0.0"])
newv, e3 = run(NEW, "new_v001", ["--w-act-level", "0.01"])
if None in (old, new, new0, newv):
    res.update({"pass": False, "max_abs_diff": None, "error": {"old": e0, "new": e1, "new_explicit0": e2, "new_v001": e3}})
else:
    d1, b1 = compare(old, new)
    d2, b2 = compare(old, new0)
    moved = max(abs(x["total"] - y["total"]) for x, y in zip(old, newv))
    has_key = any("act_level" in k for r in new + new0 for k in r)
    lv_keys_on = all(("act_level" in r and "act_level_n" in r) for r in newv)
    res.update({"n_rows": len(old), "totals_old": [r["total"] for r in old], "totals_new": [r["total"] for r in new], "max_abs_diff_default": d1, "max_abs_diff_explicit0": d2, "mismatches": (b1 + b2)[:10],
                "control_level_on_moves_total_max_abs": moved, "act_level_key_present_with_V0": has_key, "act_level_keys_present_with_V_on": lv_keys_on,
                "max_abs_diff": max(d1, d2), "n_numeric_keys_compared": len(numeric(old[0]))})
    res["pass"] = bool(len(old) >= 3 and d1 == 0.0 and d2 == 0.0 and not b1 and not b2 and moved > 0.0 and not has_key and lv_keys_on)
if OUT:
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
print(json.dumps({"pass": res["pass"], "max_abs_diff": res["max_abs_diff"], "n_rows": res.get("n_rows")}))
