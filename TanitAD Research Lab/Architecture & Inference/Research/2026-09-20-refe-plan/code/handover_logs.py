"""Hand the dev box the logs the pod would reach LAST -- a disjoint split, recomputed from live state.

The pod's rank-0 shards walk `sorted(pod_logs)[i::N]` in a seeded shuffle (build_teacher_rollouts,
seed 20260924). The logs at the TAIL of each shard's walk are the ones the pod would reach last, so
handing those to the dev box cannot collide with the pod's in-flight logs, and if the dev box runs
late the pod simply gets them back at the next rebalance. Only logs with a LOCAL DB on the dev box
are eligible (no fetch on the critical path) and only logs with NO rows anywhere yet.

  python code/handover_logs.py --pod-logs pod_logs_10hz.txt --touched touched.txt \
      --frames pod_log_frames.json --dev-db-root D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits \
      --pod-rate 186 --dev-rate 110 --shards 7 --out-dev dev_handover.txt --out-pod pod_logs_next.txt
`touched.txt`: every log with >= 1 row in any pod shard file OR in the dev box's files (both sides).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-logs", required=True)
    ap.add_argument("--touched", required=True)
    ap.add_argument("--frames", required=True, help="{'frames_per_log': {...}}")
    ap.add_argument("--dev-db-root", required=True)
    ap.add_argument("--pod-rate", type=float, required=True)
    ap.add_argument("--dev-rate", type=float, required=True)
    ap.add_argument("--dev-backlog", type=float, default=0.0,
                    help="frames the dev box still has queued: both machines must FINISH together, so "
                         "H = (dev_rate * pod_rem - pod_rate * backlog) / (dev_rate + pod_rate)")
    ap.add_argument("--shards", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--out-pod", required=True)
    a = ap.parse_args()
    pod = sorted({l.strip() for l in open(a.pod_logs, encoding="utf-8") if l.strip()})
    touched = {l.strip() for l in open(a.touched, encoding="utf-8") if l.strip()}
    fpl = json.load(open(a.frames, encoding="utf-8"))["frames_per_log"]
    local = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(a.dev_db_root, "**", "*.db"),
                                                          recursive=True)}
    remaining = [l for l in pod if l not in touched]
    rem_frames = sum(fpl.get(l, 0) for l in remaining)
    share = max(0.0, (a.dev_rate * rem_frames - a.pod_rate * a.dev_backlog) / (a.dev_rate + a.pod_rate))
    # the pod's own walk per shard, reversed: its LAST logs first
    tails = []
    for i in range(a.shards):
        names = pod[i::a.shards]
        random.Random(a.seed).shuffle(names)
        tails.append([l for l in reversed(names) if l not in touched])
    dev, got, k = [], 0, 0
    while got < share and any(tails):
        t = tails[k % a.shards]
        k += 1
        while t:
            l = t.pop(0)
            if l in local:
                dev.append(l)
                got += fpl.get(l, 0)
                break
    dev_set = set(dev)
    pod_next = [l for l in pod if l not in dev_set]
    open(a.out_dev, "w", encoding="utf-8").write("\n".join(sorted(dev)) + "\n")
    open(a.out_pod, "w", encoding="utf-8").write("\n".join(pod_next) + "\n")
    print(json.dumps({"pod_logs": len(pod), "touched": len(touched & set(pod)),
                      "remaining_logs": len(remaining), "remaining_frames": rem_frames,
                      "dev_share_target_frames": round(share), "dev_logs": len(dev),
                      "dev_frames": got, "pod_next_logs": len(pod_next),
                      "eligible_local": len(set(remaining) & local)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
