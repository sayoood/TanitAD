#!/usr/bin/env python3
"""(c'2) of PREREG_3C.md -- the no-slowdown check on the KEPT twins.

    sample   --merged-dir <merge out dir> --bank-in <input bank dir> --out <dir> [--per-rank 300] [--seed 20261017]
             picks, per rank K, up to 300 kept twins (sorted by (log, token), then random.Random(seed).sample, ranks in order 1,2,3 from ONE generator) and writes
             <out>/bank_c2/targets_rank1.jsonl = the ORIGINAL input rows of those twins + <out>/sample.json.  It then PRINTS the command that re-rolls them
             "route-held, 12 s goal horizon, 4 s" (reroll_twins.py run --arm held, the probe's rr_held arm) -- run it, then:
    analyze  --merged-dir <merge out dir> --rr-held12 '<out>/rr_held12/s*/targets_reroll.jsonl' --out <cprime2.json>
             pairs each sampled twin's KEPT 4 s traj (from the merged bank) with its held-12 s 4 s traj; per rank: path-length ratio and end-speed change (math COPIED
             VERBATIM from probe/progress_check.py: plen, vend, `plen(base) > 1.0` for the ratio).  Judged per rank where n >= 5:
             PASS iff median path ratio >= 0.95 AND median end-speed change >= -0.3 m/s.  A failing judged rank => outcome 3C-SLOW.

DIFFERENCE FROM progress_check.py, deliberate: it keyed pairs by (log, token, r["rank"]) and r["rank"] is 1 for EVERY twin (the file is targets_rank1), so a token
carrying twins of two ranks collapses to one; this keys by aug.value (K).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

PER_RANK = 300
SEED = 20261017
MIN_N = 5
BAR_RATIO = 0.95
BAR_DV = -0.3


def plen(t):                                   # progress_check.py, verbatim
    pts = [(0.0, 0.0)] + [(p[0], p[1]) for p in t]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def vend(t, dt=0.2):                           # progress_check.py, verbatim
    return math.dist(t[-1][:2], t[-2][:2]) / dt


def key(r):
    return (r["log_name"], r["token"], int(r["aug"]["value"]))


def cmd_sample(a) -> int:
    audit = [json.loads(l) for l in Path(a.merged_dir, "audit_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [(r["log_name"], r["token"], int(r["k"])) for r in audit if r["commit"] and r["drop_reason"] is None]
    rng = random.Random(a.seed)
    chosen, per = [], {}
    for k in (1, 2, 3):
        ks = sorted(x for x in kept if x[2] == k)
        pick = rng.sample(ks, min(a.per_rank, len(ks)))
        per[str(k)] = {"kept": len(ks), "sampled": len(pick)}
        chosen += pick
    want = set(chosen)
    out = Path(a.out); (out / "bank_c2").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(Path(a.bank_in, "targets_rank1.jsonl"), encoding="utf-8") as fi, open(out / "bank_c2" / "targets_rank1.jsonl", "w", encoding="utf-8") as fo:
        for ln in fi:
            if '"lane_rank"' not in ln:
                continue
            r = json.loads(ln)
            if r.get("aug", {}).get("kind") == "lane_rank" and key(r) in want:
                fo.write(ln if ln.endswith("\n") else ln + "\n"); n += 1
    assert n == len(want), (n, len(want))
    (out / "sample.json").write_text(json.dumps({"seed": a.seed, "per_rank": per, "keys": [list(k) for k in chosen]}, indent=1), encoding="utf-8")
    print(f"sampled {n} twins {per} -> {out}/bank_c2")
    print("NEXT (pod; N shards, e.g. 90):  source /workspace/ops/bank3c/code/env_3c.sh; cd $PKG/refe; for ((i=0;i<N;i++)); do run3c $PY $CODE3C/reroll_twins.py run "
          f"--bank {out}/bank_c2 --out {out}/rr_held12/s$i --arm held --device cpu --log-shard $i/N --code-dir $CODE3C --refe-dir $PKG/refe --pkg-code-dir $PKG/code "
          "> logs_c2_$i.log 2>&1 & done; wait")
    return 0


def cmd_analyze(a) -> int:
    base = {}
    for fp in sorted(glob.glob(a.rr_held12)):
        for ln in open(fp, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln); base[key(r)] = r
    kept = {}
    for ln in open(Path(a.merged_dir, "targets_rank1.jsonl"), encoding="utf-8"):
        if '"lane_rank"' in ln:
            r = json.loads(ln)
            if r.get("aug", {}).get("kind") == "lane_rank":
                kept[key(r)] = r
    res = {"bar": {"path_ratio_median_ge": BAR_RATIO, "end_speed_change_median_ge": BAR_DV, "min_n": MIN_N}, "n_rr_held12": len(base), "ranks": {}}
    failed = []
    for k in (1, 2, 3):
        ks = sorted(x for x in base if x[2] == k and x in kept)
        rl = [plen(kept[x]["traj"]) / max(plen(base[x]["traj"]), 1e-6) for x in ks if plen(base[x]["traj"]) > 1.0]
        dv = [vend(kept[x]["traj"]) - vend(base[x]["traj"]) for x in ks]
        e = {"n_paired": len(ks), "n_moving": len(rl)}
        if rl:
            e.update(path_ratio_median=round(st.median(rl), 4), path_ratio_p10_p90=[round(sorted(rl)[int(0.1 * (len(rl) - 1))], 4), round(sorted(rl)[int(0.9 * (len(rl) - 1))], 4)],
                     share_path_ratio_lt_0_9=round(sum(x < 0.9 for x in rl) / len(rl), 3))
        if dv:
            e["end_speed_change_median_mps"] = round(st.median(dv), 3)
        judged = len(ks) >= MIN_N and len(rl) >= MIN_N
        e["judged"] = judged
        if judged:
            e["pass"] = e["path_ratio_median"] >= BAR_RATIO and e["end_speed_change_median_mps"] >= BAR_DV
            if not e["pass"]:
                failed.append(k)
        res["ranks"][str(k)] = e
    res["outcome"] = ("3C-SLOW (c'2 fails at rank(s) %s): next lever h8 + commit filter" % failed) if failed else ("c'2 PASS" if any(v["judged"] for v in res["ranks"].values()) else "c'2 NOT JUDGED (no rank has n >= 5)")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sample")
    s.add_argument("--merged-dir", required=True); s.add_argument("--bank-in", required=True); s.add_argument("--out", required=True)
    s.add_argument("--per-rank", type=int, default=PER_RANK); s.add_argument("--seed", type=int, default=SEED)
    n = sp.add_parser("analyze")
    n.add_argument("--merged-dir", required=True); n.add_argument("--rr-held12", required=True); n.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"sample": cmd_sample, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
