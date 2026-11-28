"""Build the POSE-SYNC sidecar (X10): per cache row, how much LATER the camera image was captured than the
pose stored beside it.

    python stack/scripts/build_pose_sync_sidecar.py \
        --manifest <_v2manifest.pt> [--manifest ...] \
        --camera-dir <dir holding <clip_id>.timestamps.parquet> \
        --out <pose_sync.jsonl> [--clock-sidecar <refcv6_clip_clock_sidecar.jsonl>]

WHY IT EXISTS.  ``v2_compressed._resampled`` stores ``signals_at(ego, t_query)`` for ``t_query =
linspace(t_first, t_last, int(span * 10))`` but decodes the first camera frame AT OR AFTER each
``t_query`` (``searchsorted`` side='left').  The cache keeps the poses and drops ``t_frames``, so the
offset ``t_image - t_pose in [0, 33.4 ms)`` cannot be recovered from the cache.  See
``tanitad/data/pose_sync.py`` for the full statement and the correction.

WHAT IT READS.  Only the clip's camera ``*.timestamps.parquet`` and the cache manifest (clip ids,
``T_out``, ``n_stack``, ``episode_uid``).  It does NOT need the egomotion log or any frame.

CONTROLS (the build REFUSES when any fails -- nothing is guessed):
  K1  a SYNTHETIC 30 fps camera at a KNOWN geometry must give delta[0] == 0 and the closed form
      ``analytic_delta_s`` to < 5 us, before any real clip is read.
  K2  per clip: the grid it rebuilds must have EXACTLY the cache's row count
      (``n_target - (n_stack - 1) == T_out``); a clip whose grid does not match is SKIPPED AND COUNTED --
      its poses were not produced by this timeline.
  K3  per clip: ``stable_episode_id(clip_id) == manifest episode_uid`` (the join key the trainer uses).
  K4  optional ``--clock-sidecar``: the rebuilt ``dt_s`` must equal the clock sidecar's ``dt_s`` (which was
      recovered by an INDEPENDENT route -- inverting the poses on the 100 Hz egomotion log) to < 1e-6 s.

Rows are keyed on ``sid = stable_episode_id(clip_id)`` and carry NO clip id (the clip-id rule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanitad.data import pose_sync as PS  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def self_test() -> dict:
    """K1 -- a synthetic camera at a KNOWN geometry."""
    tf = np.arange(0, 20.0e6 + 1, 1e6 / 30.0).round()          # 30 fps, first frame at 0
    d, dt, n = PS.grid_delta(tf)
    da = PS.analytic_delta_s(tf[0], tf[-1], len(tf), n)
    ok = (abs(d[0]) < 1e-12 and float(np.abs(d - da).max()) * 1e6 < 5.0
          and 0.0 <= d.min() and d.max() * 1e3 < 33.4 and abs(dt - 20.0 / (n - 1)) < 1e-9)
    return {"ok": bool(ok), "delta0_us": float(d[0] * 1e6), "delta_max_ms": float(d.max() * 1e3),
            "analytic_maxabs_us": float(np.abs(d - da).max() * 1e6), "dt_s": dt, "n_target": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", action="append", default=[], help="a cache _v2manifest.pt (repeatable)")
    ap.add_argument("--camera-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clock-sidecar", default=None)
    ap.add_argument("--max-skipped-frac", type=float, default=0.01,
                    help="REFUSE to write when more than this fraction of clips is skipped (default 1 %%)")
    a = ap.parse_args(argv)
    k1 = self_test()
    if not k1["ok"]:
        raise SystemExit(f"[pose-sync] K1 self-test FAILED, nothing written: {k1}")
    if not a.manifest:
        raise SystemExit("[pose-sync] give at least one --manifest")
    import pandas as pd
    import torch

    clock = {}
    if a.clock_sidecar:
        for line in open(a.clock_sidecar, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                clock[int(r["sid"])] = float(r["dt_s"])
    rows, skipped, k4_checked, k4_worst = [], {"no_camera_ts": 0, "grid_rows_mismatch": 0, "sid_mismatch": 0,
                                               "k4_dt_mismatch": 0}, 0, 0.0
    n_total, seen = 0, set()
    for mp in a.manifest:
        man = torch.load(mp, map_location="cpu", weights_only=False)
        for i, cid in enumerate(man["clip_id"]):
            n_total += 1
            sid = stable_episode_id(cid)
            if sid != int(man["episode_uid"][i]):
                skipped["sid_mismatch"] += 1                         # K3
                continue
            if sid in seen:
                continue
            f = os.path.join(a.camera_dir, f"{cid}.timestamps.parquet")
            if not os.path.isfile(f):
                skipped["no_camera_ts"] += 1
                continue
            ts = pd.read_parquet(f)
            tf = ts[next(c for c in ts.columns if "time" in c.lower())].to_numpy(np.float64)
            d, dt, n = PS.grid_delta(tf)
            n_stack = int(man["n_stack"][i])
            if n - (n_stack - 1) != int(man["T_out"][i]):            # K2
                skipped["grid_rows_mismatch"] += 1
                continue
            if sid in clock:                                           # K4
                k4_checked += 1
                k4_worst = max(k4_worst, abs(dt - clock[sid]))
                if abs(dt - clock[sid]) > 1e-6:
                    skipped["k4_dt_mismatch"] += 1
                    continue
            rows.append({"sid": sid, "dt_s": dt, "delta_us": np.rint(d * 1e6).astype(np.int64).tolist()})
            seen.add(sid)
    n_skipped = sum(skipped.values())
    if not rows or n_skipped > a.max_skipped_frac * max(n_total, 1):
        raise SystemExit(f"[pose-sync] REFUSING to write: {n_skipped}/{n_total} clips skipped "
                         f"{skipped} (cap {a.max_skipped_frac:.1%}); nothing written")
    allv = np.concatenate([np.asarray(r["delta_us"], dtype=np.float64) for r in rows]) / 1e3
    meta = {"builder": "scripts/build_pose_sync_sidecar.py", "rule":
            "delta_us[k] = t_image[k] - t_pose[k]; t_pose = linspace(t_first, t_last, int(span*10))[k]; "
            "t_image = first camera frame >= t_pose (v2_compressed.py:119-121). Indexed by RAW grid row k "
            "(provider row r = k - (n_stack - 1)).",
            "n_total_clips": n_total, "n_rows_written": len(rows), "skipped": skipped,
            "k1_self_test": k1, "k4_checked": k4_checked, "k4_worst_abs_dt_s": k4_worst,
            "delta_ms": {"mean": float(allv.mean()), "min": float(allv.min()), "max": float(allv.max()),
                         "p95": float(np.quantile(allv, 0.95))},
            "inputs_md5": {os.path.basename(m): _md5(m) for m in a.manifest},
            "clock_sidecar": os.path.basename(a.clock_sidecar) if a.clock_sidecar else None}
    PS.write_pose_sync_sidecar(a.out, rows, meta)
    print(f"[pose-sync] wrote {len(rows)}/{n_total} clips -> {a.out}  delta mean {meta['delta_ms']['mean']:.3f} ms "
          f"max {meta['delta_ms']['max']:.3f} ms  skipped {skipped}  K4 checked {k4_checked} worst {k4_worst:.2e} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
