#!/usr/bin/env python3
"""P1 - bandwidth probe: per-stream AND aggregate MB/s at concurrency 1, 4 and 8.

The 2026-09-11 package measured FOUR single-stream samples (1.59 / 2.63 / 5.41 / 11.57
MB/s, a 7.3x spread) and correctly refused to quote a corpus wall-clock from them. This
probe measures what the corpus build actually needs: how aggregate throughput scales
with the number of concurrent clip streams.

Design (fixed before any transfer ran):
  * clips = the 139-clip B1 EVAL join in JOIN ORDER, skipping clips whose parquet is
    already cached (so no byte is fetched twice); the next 8 go to c=1, the next 8 to
    c=4, the next 8 to c=8. DIFFERENT clips per arm; chosen by position, never content.
  * each arm is a ProcessPool of `c` workers over its 8 clips (one HfFileSystem per
    worker process). Per-stream MB/s = member bytes / member stream time. Aggregate
    MB/s = arm bytes / arm wall time (first start -> last end).
  * a wall-time budget per arm (`--arm-budget-s`): once exceeded, no NEW clip starts;
    in-flight clips finish. `n` is reported per arm, so a truncated arm is visible.
  * every fetched parquet is CRC-checked (zipfile) and content-asserted (199 row groups,
    decodable non-zero spin 0) -- and KEPT, because P2 consumes it.

⚠️ Arms run sequentially in time, so a bandwidth change between arms is confounded with
concurrency. The arm order and absolute timestamps are recorded so that is auditable.

Usage:
  python p1_bandwidth.py --out ../raw/p1_bandwidth.json
"""
from __future__ import annotations

import argparse
import json
import lzma
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lidar_fetch import (  # noqa: E402
    LIDAR_CACHE, cached_path, chunk_map, fetch_clip, sha12,
)

JOIN = (Path(__file__).resolve().parents[4] / "Benchmarks & Evals" / "Research"
        / "2026-09-06-b1-agent-join" / "raw" / "b1eval_agents.jsonl.xz")


def join_clips() -> list[str]:
    seen: dict[str, None] = {}
    with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
        for line in fh:
            seen.setdefault(json.loads(line)["clip_id"], None)
    return list(seen)


def run_arm(conc: int, clips: list[str], chunks: dict, budget_s: float) -> dict:
    t_arm0 = time.time()
    recs: list[dict] = []
    pending = list(clips)
    with ProcessPoolExecutor(max_workers=conc) as ex:
        futs = {}
        while pending and len(futs) < conc:
            c = pending.pop(0)
            futs[ex.submit(fetch_clip, c, chunks[c], LIDAR_CACHE)] = c
        while futs:
            done = next(as_completed(list(futs)))
            futs.pop(done)
            r = done.result()
            recs.append(r)
            print(f"[p1] c={conc} {r['clip']} ok={r['ok']} "
                  f"{r.get('stream_MBps')} MB/s  cd={r.get('central_dir_s')} s  "
                  f"bytes={r.get('bytes_written')}  err={r.get('error', '')}", flush=True)
            if pending and (time.time() - t_arm0) < budget_s:
                c = pending.pop(0)
                futs[ex.submit(fetch_clip, c, chunks[c], LIDAR_CACHE)] = c
    ok = [r for r in recs if r.get("ok")]
    wall = (max(r["t_end_unix"] for r in recs) - min(r["t_start_unix"] for r in recs)) \
        if recs else float("nan")
    tot = sum(r["bytes_written"] for r in ok)
    per = [r["stream_MBps"] for r in ok]
    return {
        "concurrency": conc,
        "n_clips_requested": len(clips),
        "n_clips_attempted": len(recs),
        "n_ok": len(ok),
        "n_skipped_by_budget": len(pending),
        "arm_wall_s": round(wall, 2),
        "arm_bytes": tot,
        "aggregate_MBps": round(tot / 1e6 / wall, 3) if ok and wall > 0 else None,
        "per_stream_MBps": {
            "n": len(per),
            "values": per,
            "min": min(per) if per else None,
            "median": round(statistics.median(per), 3) if per else None,
            "max": max(per) if per else None,
        },
        "central_dir_s": sorted(r.get("central_dir_s") for r in ok),
        "clips": recs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "raw" / "p1_bandwidth.json"))
    ap.add_argument("--per-arm", type=int, default=8)
    ap.add_argument("--arms", default="1,4,8")
    ap.add_argument("--arm-budget-s", type=float, default=540.0)
    args = ap.parse_args()

    chunks = chunk_map()
    allc = join_clips()
    todo = [c for c in allc if not cached_path(c).exists()]
    arms = [int(a) for a in args.arms.split(",")]
    out = {
        "schema": "tanitad.lidar_bandwidth_probe/1",
        "evidence_class": "MEASURED (ours, dev box, this run)",
        "reader": "HfFileSystem.open(chunk zip) -> zipfile member stream -> disk "
                  "(lidar_fetch.fetch_clip; same transfer path as 09-11 p1_lidar_probe)",
        "join_clips": len(allc),
        "already_cached_in_join": len(allc) - len(todo),
        "design": f"join order, skip cached, next {args.per_arm} clips per arm, arms "
                  f"{arms} run sequentially; budget {args.arm_budget_s} s per arm "
                  "(no new clip starts after it)",
        "t_start_unix": time.time(),
        "arms": [],
    }
    pos = 0
    for conc in arms:
        clips = todo[pos: pos + args.per_arm]
        pos += args.per_arm
        print(f"[p1] arm c={conc}: {len(clips)} clips {[sha12(c) for c in clips]}", flush=True)
        a = run_arm(conc, clips, chunks, args.arm_budget_s)
        out["arms"].append(a)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"[p1] arm c={conc} aggregate {a['aggregate_MBps']} MB/s over "
              f"{a['arm_wall_s']} s, per-stream median {a['per_stream_MBps']['median']} "
              f"(n={a['per_stream_MBps']['n']})", flush=True)
    out["t_end_unix"] = time.time()
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[p1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
