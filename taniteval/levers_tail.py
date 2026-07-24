#!/usr/bin/env python3
"""Tail-latency replicates for the flagship inference levers.

WHY THIS IS ITS OWN RUN. The mean is not the deployment number. On this arm the
eager tick's p50 is stable to ~1 % across identical back-to-back runs while its
**p99 swings ~26 %** (REF-C's swings 1 %) — and a control loop is specified on
its tail, not its median. A CUDA graph replays ONE fixed launch sequence, so the
prediction is that its run-to-run tail spread collapses; this measures it.

Run on an EXCLUSIVE GPU (take `gpu_lock.sh` first):
    PYTHONPATH=/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts \
      python3 taniteval/levers_tail.py [--model flagship-30k] [--precision fp32]
-> results/eff_levers_tail_<key>.json
"""
import argparse
import sys

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import efficiency as E  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser("levers_tail")
    ap.add_argument("--model", default="flagship-30k")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    a = ap.parse_args(argv)
    out = E.tail_replicates(a.model, precision=a.precision, reps=a.reps,
                            iters=a.iters, warmup=a.warmup)
    for name, blk in out["by_variant"].items():
        print(f"[tail] {name:18s} p50 spread {blk['p50_across_reps']} | "
              f"p99 spread {blk['p99_across_reps']} | "
              f"tail_ratio {blk['tail_ratio_mean']}", flush=True)
    print(f"[tail] clean={out['contamination_check']['valid']}", flush=True)


if __name__ == "__main__":
    main()
