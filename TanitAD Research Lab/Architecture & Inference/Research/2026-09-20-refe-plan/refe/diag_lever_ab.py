"""A/B an EXACT data-prep speed lever: the rows must not change, only the clock.

Each arm is `NAME:KEY=VAL[,KEY=VAL...]` (environment for build_teacher_rollouts.py); the FIRST arm
is the reference. All arms run CONCURRENTLY on the same frames (same machine load, so the s/tuple
ratio is fair). PASS = every arm exits 0 and writes rows byte-identical to the reference.

  python diag_lever_ab.py --log <log> --frames 8 --out <dir> \
      --arm REF:REFE_SKIP_DEBUG=0 --arm LEAN:REFE_SKIP_DEBUG=1
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
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--device", default=None,
                    help="teacher device passed to every arm (production is `cpu` since "
                         "2026-09-24); omitted = the builder's own default")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logs_file = os.path.join(a.out, "logs.txt")
    open(logs_file, "w", encoding="utf-8").write(a.log + "\n")
    arms = []
    for spec in a.arm:
        name, _, kv = spec.partition(":")
        env = dict(x.split("=", 1) for x in kv.split(",") if x)
        arms.append((name, env))
    procs = {}
    for name, extra in arms:
        d = os.path.join(a.out, name)
        f = os.path.join(d, "targets_rank0.jsonl")
        if os.path.exists(f):
            os.remove(f)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", **extra)
        log = open(os.path.join(a.out, f"{name}.log"), "w", encoding="utf-8")
        procs[name] = (subprocess.Popen(
            [sys.executable, "build_teacher_rollouts.py", "--source", "navtrain", "--logs-file",
             logs_file, "--limit-frames", str(a.frames), "--out", d, "--rank", "0"]
            + (["--device", a.device] if a.device else []),
            cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time())
    res, rows = {}, {}
    for name, (p, log, t0) in procs.items():
        rc = p.wait(); log.close()
        txt = open(os.path.join(a.out, f"{name}.log"), encoding="utf-8").read()
        m = re.search(r"([0-9.]+) s/tuple", txt)
        f = os.path.join(a.out, name, "targets_rank0.jsonl")
        rows[name] = open(f, encoding="utf-8").read().splitlines() if os.path.exists(f) else []
        res[name] = {"rc": rc, "rows": len(rows[name]), "wall_s": round(time.time() - t0, 1),
                     "s_per_tuple": float(m.group(1)) if m else None, "env": dict(arms)[name]}
    ref = arms[0][0]
    checks = {"all_exit_0": all(r["rc"] == 0 for r in res.values()),
              "ref_rows_nonempty": len(rows[ref]) > 0}
    for name, _ in arms[1:]:
        checks[f"{name}_rows_eq_{ref}"] = rows[name] == rows[ref]
        if res[ref]["s_per_tuple"] and res[name]["s_per_tuple"]:
            res[name]["speedup_vs_ref"] = round(res[ref]["s_per_tuple"] / res[name]["s_per_tuple"], 3)
    out = {"arms": res, "checks": checks}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(a.out, "diag_lever_ab.json"), "w"), indent=1)
    ok = all(checks.values())
    print("ZZLEVER_" + ("EXACTZZ" if ok else "FAILZZ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
