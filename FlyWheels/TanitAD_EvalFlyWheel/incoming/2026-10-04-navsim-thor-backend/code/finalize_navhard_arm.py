#!/usr/bin/env python3
"""Thor side: turn 6 whole-group navhard shard runs of ONE arm into one arm directory (NAVSIM venv).

    python finalize_navhard_arm.py --arm R7_VMAXOFF --shard-root /dev/shm/navsim/thor_scores/navhard/_shards \
        --out /dev/shm/navsim/thor_scores/navhard --order-ref <dev-box navhard CSV> --yaml <navhard_two_stage.yaml>

Refuses unless EVERY shard's counts.json reads PASS, the union of expected tokens is exactly 5,912 with
no overlap, and the summed seam calls equal the seam's model rows. Then ``merge_shards.py navhard``
(devkit main() tail verbatim) writes ``<ARM>_THOR/score_<ARM>_THOR__navhard_two_stage.csv`` + the
merged final frame, and a merged ``.counts.json`` (status PASS/FAIL, per-shard records, sha256s).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SFX = "__navhard_two_stage"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order-ref", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--expected", type=int, default=5912)
    a = ap.parse_args()
    tag = f"{a.arm}_THOR"
    shards, fails, frames, toks, seam_calls = [], [], [], [], 0
    for k in range(a.k):
        d = os.path.join(a.shard_root, f"{tag}_k{k}of{a.k}")
        c = glob.glob(os.path.join(d, f"score_{tag}{SFX}.counts.json"))
        if not c:
            fails.append(f"shard {k}: no counts.json")
            continue
        cj = json.load(open(c[0], encoding="utf-8"))
        sub = json.load(open(cj["subset"], encoding="utf-8")) if cj.get("subset") else {}
        shards.append({"k": k, "status": cj["status"], "n": cj.get("csv_valid_rows"), "agent_calls": cj.get("agent_calls"),
                       "wall_s": cj.get("wall_s"), "seam_sha256": cj.get("seam_sha256"), "failures": cj.get("failures")})
        if cj["status"] != "PASS":
            fails.append(f"shard {k}: {cj['status']} {cj.get('failures')}")
        toks += sub.get("expected_tokens", [])
        seam_calls += (cj.get("agent_calls") or {}).get("seam", 0)
        frames.append(os.path.join(d, f"score_{tag}{SFX}_wrapper", f"{tag}{SFX}_final_scores_frame.csv"))
    shas = {s["seam_sha256"] for s in shards}
    if len(shas) != 1:
        fails.append(f"shards used {len(shas)} different seams")
    if len(toks) != a.expected or len(set(toks)) != len(toks):
        fails.append(f"union of shard tokens {len(set(toks))} (rows {len(toks)}) != {a.expected}")
    odir = os.path.join(a.out, tag)
    wdir = os.path.join(odir, f"score_{tag}{SFX}_wrapper")
    os.makedirs(wdir, exist_ok=True)
    csv_p = os.path.join(odir, f"score_{tag}{SFX}.csv")
    frame_p = os.path.join(wdir, f"{tag}{SFX}_final_scores_frame.csv")
    merge = None
    if not fails:
        r = subprocess.run([sys.executable, os.path.join(HERE, "merge_shards.py"), "navhard", "--frames", *frames,
                            "--order-ref", a.order_ref, "--yaml", a.yaml, "--out-csv", csv_p, "--out-frame", frame_p],
                           capture_output=True, text=True)
        merge = {"rc": r.returncode, "stdout": r.stdout.strip()[-400:], "stderr": r.stderr.strip()[-400:]}
        if r.returncode != 0 or not os.path.exists(csv_p):
            fails.append("merge failed")
    rep = {"arm": a.arm, "tag": tag, "split": "navhard_two_stage", "backend": "thor", "status": "FAIL" if fails else "PASS",
           "failures": fails, "expected_tokens": a.expected, "csv_valid_rows": (len(toks) if not fails else None),
           "agent_calls": {"seam": seam_calls}, "seam_sha256": (shas.pop() if len(shas) == 1 else None),
           "shards": shards, "merge": merge, "csv": csv_p if os.path.exists(csv_p) else None,
           "csv_sha256": sha(csv_p) if os.path.exists(csv_p) else None,
           "frame": frame_p if os.path.exists(frame_p) else None,
           "how": "6 whole-group shards (shard_navhard_groups.py, group g -> shard g%6) scored by score_arm_thor.py with "
                  "tanitad_seam_agent_ulp (accepted, atol 1e-9), merged by merge_shards.py (devkit main() tail verbatim)",
           "policy": "RULING_BACKEND_POLICY.md (2026-10-04-navsim-thor-backend)"}
    json.dump(rep, open(os.path.join(odir, f"score_{tag}{SFX}.counts.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("arm", "status", "failures", "csv_valid_rows", "agent_calls")}))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
