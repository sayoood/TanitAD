#!/usr/bin/env python3
"""Measure the rerun .rrd artifact path of the replay viz backbone.

Backlog P0#1 asked for "episode -> .rrd replay/viz, measure .rrd size + write
time per episode (G-T1)". The logging schema (`tanitad/replay/rr_log.py`) has
existed since 2026-07-10; what has never been measured is what it *costs* and
whether its two sinks actually work together — mission P1 carries an open bug
"dual-sink (serve+rrd) empty file" with no measurement attached.

This drives RerunLogger with synthetic-but-contract-shaped TimestepRecords
(256x256x3 uint8 frames, 4 waypoints, 3 arms, monitor heads, an imagination
fan) so the measurement isolates the *logging* cost from model inference.

    python bench_rrd.py --windows 200 --out results.json

Reports, per arm-set and per jpeg setting: total .rrd bytes, bytes/window,
wall-clock, windows/s, and the dual-sink verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

for p in (Path(__file__).resolve().parents[4] / "stack",):
    sys.path.insert(0, str(p))

from tanitad.replay.engine import ArmOutput, TimestepRecord   # noqa: E402
from tanitad.replay.rr_log import RerunLogger                 # noqa: E402

ARMS = ("main", "refa", "refb")
H = W = 256


def make_records(n: int, arms=ARMS, seed: int = 0) -> list[TimestepRecord]:
    """Contract-shaped records: same dtypes/shapes the engine emits."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        gt = np.cumsum(rng.normal(0, 0.4, size=(4, 2)) +
                       np.array([[3.0, 0.0]]), axis=0)
        out = {}
        for a in arms:
            out[a] = ArmOutput(
                latency_ms=float(rng.uniform(4, 30)),
                waypoints=gt + rng.normal(0, 0.3, size=(4, 2)),
                action=rng.normal(0, 0.2, size=2),
                maneuver_probs=(lambda p: p / p.sum())(rng.random(5))
                if a == "refb" else None,
                maneuver_gt=int(rng.integers(0, 5)) if a == "refb" else None,
                conf=float(rng.random()), ood=float(rng.random()),
                sigma=float(rng.random()),
                imag_rel={k: float(rng.random()) for k in (1, 2, 4)},
                imag_traj={k: rng.normal(0, 3, size=2) for k in (1, 2, 4)})
        recs.append(TimestepRecord(
            step=i, corpus="physicalai-val-8c0d3047924e", episode_id=i // 50,
            ep_index=i // 50, t=i % 50,
            gt_waypoints=gt, gt_action=rng.normal(0, 0.2, size=2),
            speed=float(12 + rng.normal()), yaw_rate=float(rng.normal(0, 0.05)),
            arms=out,
            frame=rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)))
    return recs


def run(recs, rrd: Path, jpeg: int | None, serve: int | None = None) -> dict:
    t0 = time.perf_counter()
    lg = RerunLogger(rrd=str(rrd), serve=serve, jpeg_quality=jpeg,
                     app_id="rrd_bench")
    t_init = time.perf_counter() - t0
    t1 = time.perf_counter()
    for r in recs:
        lg.log_record(r)
    lg.close()
    wall = time.perf_counter() - t1
    time.sleep(0.5)                     # flush is async on the file sink
    size = rrd.stat().st_size if rrd.exists() else 0
    return {"windows": len(recs), "jpeg": jpeg, "serve": serve,
            "init_s": round(t_init, 3), "log_s": round(wall, 3),
            "windows_per_s": round(len(recs) / wall, 1),
            "rrd_bytes": size, "bytes_per_window": round(size / len(recs)),
            "rrd_empty": size < 1024}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).with_name("results.json"))
    ap.add_argument("--tmp", type=Path, default=None)
    a = ap.parse_args()

    tmp = a.tmp or Path(__file__).with_name("_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    recs = make_records(a.windows)

    import rerun
    res = {"rerun_version": rerun.__version__,
           "python": sys.version.split()[0],
           "frame_hw": [H, W], "arms": list(ARMS), "cases": []}

    # A: file sink only, jpeg-compressed frames (the shipped default)
    res["cases"].append({"case": "rrd_only_jpeg85",
                         **run(recs, tmp / "a.rrd", 85)})
    # B: file sink only, raw frames — the compression lever, measured
    res["cases"].append({"case": "rrd_only_raw",
                         **run(recs, tmp / "b.rrd", None)})
    # C: THE OPEN BUG — file + live web viewer at once. Does a.rrd stay empty?
    try:
        res["cases"].append({"case": "dual_sink_rrd_plus_serve",
                             **run(recs, tmp / "c.rrd", 85, serve=9399)})
    except Exception as exc:                                # noqa: BLE001
        res["cases"].append({"case": "dual_sink_rrd_plus_serve",
                             "error": f"{type(exc).__name__}: {exc}"})

    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
