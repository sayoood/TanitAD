#!/usr/bin/env python3
"""REFe seam in K log-disjoint shards on ONE GPU, then a merge that restores the single-process row order.

    <seam venv python> seam_shards.py --k 3 --seam-script <pkg>/eval/refe_navtest_seam.py --ckpt snap.pt \
        --frames <dir> --db-dir <dir> --export <export.json.gz> --tokens <toks.json> --work <dir> \
        --out merged.npz --dump-out merged_props.npz [--arm REFe_sub200_ep016]

Why this is admissible: `refe_navtest_seam.py` runs one batch-1 forward per token and keeps no state across tokens
beyond per-log caches, so tokens are independent; it orders rows by SORTED log name, then by the scenario builder's
order inside a log. So each shard gets whole logs (round-robin over the sorted list), runs the UNCHANGED seam script
on its own token file, and the merge concatenates rows log by log in sorted order. G-REP then asserts the merged
seam and proposal dump are BITWISE equal to a single-process run on the same VM -- sharding is not trusted, it is
measured. Prints ZZSHARDS_OK <rows> <seconds> or ZZSHARDS_FAIL <why>.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seam-script", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-out", required=True)
    ap.add_argument("--arm", default="REFe")
    ap.add_argument("--seam-extra", action="append", default=[],
                    help="one extra seam flag per use, passed verbatim to every shard and recorded in the report, e.g. "
                         "--seam-extra=--sanitize-goal (the = form: argparse reads a leading dash as an option)")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.work, exist_ok=True)
    exp = json.load(gzip.open(a.export, "rt", encoding="utf-8"))["tokens"]
    doc = json.load(open(a.tokens, encoding="utf-8"))
    want = set(doc["tokens"] if isinstance(doc, dict) else doc)
    toks = [t for t in exp if t in want]
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    logs = sorted(by_log)
    shards = [logs[i::a.k] for i in range(a.k)]
    procs = []
    for i, sl in enumerate(shards):
        st = [t for lg in sl for t in by_log[lg]]
        tf = os.path.join(a.work, f"shard{i}_tokens.json")
        json.dump({"tokens": st, "token_log": {t: exp[t]["log_name"] for t in st}}, open(tf, "w"))
        cmd = [sys.executable, a.seam_script, "--ckpt", a.ckpt, "--frames", a.frames, "--db-dir", a.db_dir,
               "--export", a.export, "--tokens", tf, "--out", os.path.join(a.work, f"shard{i}.npz"),
               "--arm", a.arm, "--dump-proposals", os.path.join(a.work, f"shard{i}_props.npz")] + a.seam_extra
        log = open(os.path.join(a.work, f"shard{i}.log"), "w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(a.seam_script)),
                      log, i))
    rcs = {}
    for p, log, i in procs:
        rcs[i] = p.wait()
        log.close()
    txt = {i: open(os.path.join(a.work, f"shard{i}.log")).read() for i in range(a.k)}
    bad = [i for i in range(a.k) if rcs[i] != 0 or "ZZSEAM_OK" not in txt[i]]
    if bad:
        print(f"ZZSHARDS_FAIL shards {bad} rc {rcs}")
        return 1
    S = [np.load(os.path.join(a.work, f"shard{i}.npz")) for i in range(a.k)]
    Dm = [np.load(os.path.join(a.work, f"shard{i}_props.npz")) for i in range(a.k)]
    shard_of = {lg: i for i, sl in enumerate(shards) for lg in sl}
    rows_by_log: dict = {}
    for i in range(a.k):
        for j, t in enumerate(str(x) for x in S[i]["token"]):
            rows_by_log.setdefault(exp[t]["log_name"], []).append((i, j))
            assert str(Dm[i]["token"][j]) == t, "shard dump rows do not align with its seam rows"
    order = [ij for lg in logs for ij in rows_by_log.get(lg, [])]
    assert all(shard_of[exp[str(S[i]["token"][j])]["log_name"]] == i for i, j in order)
    np.savez(a.out, token=np.array([S[i]["token"][j] for i, j in order]),
             fingerprint=np.array([S[i]["fingerprint"][j] for i, j in order]),
             poses=np.stack([S[i]["poses"][j] for i, j in order]), sampling=S[0]["sampling"], arm=np.array(a.arm))
    np.savez(a.dump_out, token=np.array([Dm[i]["token"][j] for i, j in order]),
             fingerprint=np.array([Dm[i]["fingerprint"][j] for i, j in order]),
             proposals=np.stack([Dm[i]["proposals"][j] for i, j in order]),
             logits=np.stack([Dm[i]["logits"][j] for i, j in order]),
             pick=np.array([Dm[i]["pick"][j] for i, j in order], dtype=np.int64),
             select=Dm[0]["select"], rule=Dm[0]["rule"], repair_last_heading=Dm[0]["repair_last_heading"],
             ckpt=Dm[0]["ckpt"], sampling=Dm[0]["sampling"])
    reps = [json.load(open(os.path.join(a.work, f"shard{i}.report.json"))) for i in range(a.k)]
    json.dump({"k": a.k, "rows": len(order), "tokens_asked": len(toks), "shard_logs": [len(s) for s in shards],
               "shard_rows": [int(s["token"].shape[0]) for s in S],
               "shard_seconds": [r["seconds"] for r in reps],
               "frame_control_max_m": max(r["frame_control"]["max_m"] for r in reps),
               "seam_extra": a.seam_extra,
               "shard_sanitize_goal": [r.get("sanitize_goal") for r in reps],
               "wall_s": round(time.time() - t0, 1)},
              open(os.path.splitext(a.out)[0] + ".report.json", "w"), indent=1)
    # the goal switch must have REACHED every shard: a requested --sanitize-goal that a shard does not report (or one
    # reported without the request) fails the run instead of leaving the mismatch for a reader to spot
    want_san = "--sanitize-goal" in a.seam_extra
    got_san = [bool(r.get("sanitize_goal", False)) for r in reps]
    if any(g != want_san for g in got_san):
        print(f"ZZSHARDS_FAIL sanitize_goal requested {want_san} but shards report {got_san}")
        return 1
    ok = len(order) == len(toks)
    print(f"ZZSHARDS_{'OK' if ok else 'FAIL'} {len(order)} {time.time() - t0:.0f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
