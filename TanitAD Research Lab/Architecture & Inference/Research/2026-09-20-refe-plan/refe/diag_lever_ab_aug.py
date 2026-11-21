"""A/B an EXACT data-prep speed lever on the AUGMENTATION path: rows AND search stats must not change.

`diag_lever_ab.py` covers rank-0 rollouts. The augmentation search takes a different route through
the teacher -- per-candidate lane-rank / goal-horizon routes, `route_lane_rank_patch`, the route
sampler -- so a lever that is exact on rank 0 has not been shown exact here. Each arm is
`NAME:KEY=VAL[,KEY=VAL...]` (environment for augment_search.py); the FIRST is the reference. All arms
run CONCURRENTLY on the same frames of a small rank-0 bank (same load, so the time ratio is fair).
PASS = every arm exits 0, the reference accepts >= 1 row, and each arm's `targets_aug.jsonl` and
`aug_stats.jsonl` are byte-identical to the reference's. An arm named `MUT*` must DIFFER (it is the
mutation that proves the comparison can fail); it never counts toward PASS.

  python diag_lever_ab_aug.py --bank-glob "<bank>/r0_s*/targets_rank0.jsonl" --logs 2 \
      --frames-per-log 4 --out <dir> --device cpu \
      --arm REF:REFE_MAP_STEP_FAST=0 --arm FAST:REFE_MAP_STEP_FAST=1 \
      --arm MUT:REFE_MAP_STEP_FAST=1,REFE_MAP_STEP_MUTATE=route_1mm
Prints ZZLEVER_AUG_EXACTZZ or ZZLEVER_AUG_FAILZZ.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from diag_scorer_ego_view import pick_rows  # noqa: E402  (one log per shard file, complete rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-glob", required=True)
    ap.add_argument("--logs", type=int, default=2)
    ap.add_argument("--frames-per-log", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tau", default="0.3")
    ap.add_argument("--arm", action="append", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bank = os.path.join(a.out, "bank")
    os.makedirs(bank, exist_ok=True)
    rows = pick_rows(a.bank_glob, a.logs, a.frames_per_log)
    if not rows:
        print("ZZLEVER_AUG_FAILZZ no rows"); return 1
    open(os.path.join(bank, "targets_rank0.jsonl"), "w", encoding="utf-8").write("".join(rows))
    arms = []
    for spec in a.arm:
        name, _, kv = spec.partition(":")
        arms.append((name, dict(x.split("=", 1) for x in kv.split(",") if x)))
    procs = {}
    for name, extra in arms:
        d = os.path.join(a.out, name)
        shutil.rmtree(d, ignore_errors=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.pop("REFE_MAP_STEP_MUTATE", None)
        env.update(extra)
        log = open(os.path.join(a.out, f"{name}.log"), "w", encoding="utf-8")
        procs[name] = (subprocess.Popen(
            [sys.executable, "augment_search.py", "--bank", bank, "--out", d, "--tau", a.tau,
             "--device", a.device, "--limit-frames", str(a.frames_per_log)],
            cwd=HERE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT),
            log, time.time())
    res, data = {}, {}
    for name, (p, log, t0) in procs.items():
        rc = p.wait()
        log.close()
        d = os.path.join(a.out, name)
        got = {}
        for fn in ("targets_aug.jsonl", "aug_stats.jsonl"):
            f = os.path.join(d, fn)
            got[fn] = open(f, "rb").read() if os.path.exists(f) else b""
        data[name] = got
        res[name] = {"rc": rc, "rows": got["targets_aug.jsonl"].count(b"\n"),
                     "searched": got["aug_stats.jsonl"].count(b"\n"),
                     "wall_s": round(time.time() - t0, 1), "env": dict(arms)[name]}
    ref = arms[0][0]
    checks = {"all_exit_0": all(r["rc"] == 0 for r in res.values()),
              "ref_accepted_rows": res[ref]["rows"] > 0}
    for name, _ in arms[1:]:
        same = data[name] == data[ref]
        if name.startswith("MUT"):
            checks[f"{name}_differs_from_{ref}"] = not same
        else:
            checks[f"{name}_rows_and_stats_eq_{ref}"] = same
            if res[name]["wall_s"]:
                res[name]["speedup_vs_ref_wall"] = round(res[ref]["wall_s"] / res[name]["wall_s"], 3)
    out = {"frames": len(rows), "arms": res, "checks": checks}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(a.out, "diag_lever_ab_aug.json"), "w"), indent=1)
    ok = all(checks.values())
    print("ZZLEVER_AUG_" + ("EXACTZZ" if ok else "FAILZZ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
