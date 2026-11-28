"""Prove `query_cache.py` is EXACT and measure what it buys -- three arms on the same frames.

  REF   REFE_QUERY_CACHE=0            nuplan's stateless queries (map cache ON in every arm)
  FAST  REFE_QUERY_CACHE=1            the production setting
  VER   REFE_QUERY_CACHE=1 + VERIFY   every hit AND every miss re-run through nuplan's original
                                      `execute_many`/`execute_one` and compared row by row
The three arms run CONCURRENTLY (same machine load for REF and FAST, so the wall-clock ratio is
fair). PASS = FAST rows == REF rows == VER rows byte for byte, VER verified > 0 with no mismatch.

  python diag_query_cache.py --log <log_name> --frames 8 --out <scratch dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logs_file = os.path.join(a.out, "logs.txt")
    open(logs_file, "w", encoding="utf-8").write(a.log + "\n")
    arms = {"REF": {"REFE_QUERY_CACHE": "0"},
            "FAST": {"REFE_QUERY_CACHE": "1"},
            "VER": {"REFE_QUERY_CACHE": "1", "REFE_QUERY_CACHE_VERIFY": "1"}}
    procs = {}
    for name, extra in arms.items():
        d = os.path.join(a.out, name)
        if os.path.exists(os.path.join(d, "targets_rank0.jsonl")):
            os.remove(os.path.join(d, "targets_rank0.jsonl"))
        env = dict(os.environ, REFE_MAP_CACHE="1", PYTHONIOENCODING="utf-8", **extra)
        log = open(os.path.join(a.out, f"{name}.log"), "w", encoding="utf-8")
        procs[name] = (subprocess.Popen(
            [sys.executable, "build_teacher_rollouts.py", "--source", "navtrain", "--logs-file",
             logs_file, "--limit-frames", str(a.frames), "--out", d, "--rank", "0"],
            cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time())
    wall, rc = {}, {}
    for name, (p, log, t0) in procs.items():
        rc[name] = p.wait()
        wall[name] = time.time() - t0
        log.close()
    rows, text = {}, {}
    for name in arms:
        text[name] = open(os.path.join(a.out, f"{name}.log"), encoding="utf-8").read()
        f = os.path.join(a.out, name, "targets_rank0.jsonl")
        rows[name] = open(f, encoding="utf-8").read().splitlines() if os.path.exists(f) else []
    res = {n: {"rc": rc[n], "rows": len(rows[n]), "wall_s": round(wall[n], 1)} for n in arms}
    for n in arms:
        m = re.search(r"query cache: .*", text[n])
        res[n]["cache"] = m.group(0) if m else None
        m = re.search(r"([0-9.]+) s/tuple", text[n])
        res[n]["s_per_tuple"] = float(m.group(1)) if m else None
    ver = re.search(r"verified ([0-9,]+)", text["VER"])
    n_ver = int(ver.group(1).replace(",", "")) if ver else 0
    checks = {
        "all_exit_0": all(v == 0 for v in rc.values()),
        "rows_nonempty": len(rows["REF"]) > 0,
        "FAST_rows_eq_REF": rows["FAST"] == rows["REF"],
        "VER_rows_eq_REF": rows["VER"] == rows["REF"],
        "VER_verified_gt_0": n_ver > 0,
        "no_mismatch": "MISMATCH" not in text["VER"] + text["FAST"],
    }
    speed = (res["REF"]["s_per_tuple"] / res["FAST"]["s_per_tuple"]
             if res["REF"]["s_per_tuple"] and res["FAST"]["s_per_tuple"] else None)
    out = {"arms": res, "checks": checks, "speedup_REF_over_FAST": speed, "verified": n_ver}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(a.out, "diag_query_cache.json"), "w"), indent=1)
    if all(checks.values()):
        print("ZZQCACHE_" + "EXACTZZ")
        return 0
    print("ZZQCACHE_" + "FAILZZ")
    return 1


if __name__ == "__main__":
    sys.exit(main())
