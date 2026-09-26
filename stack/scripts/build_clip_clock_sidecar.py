"""Build the CLIP CLOCK sidecar: each cache clip's true row clock on the v7/v8 label timeline.

    python stack/scripts/build_clip_clock_sidecar.py --cache <v2 cache dir> [--cache <dir> ...] \
        [--manifest <_v2manifest.pt> ...] --out <clip_clock.jsonl>

`--manifest` reads a copied `_v2manifest.pt` directly (poses + clip ids are all it needs), so a
sidecar can be built where the frames are not.

WHAT IT MEASURES (A16 2026-09-26; `tanitad/data/clip_clock.py` states why it matters). The cache
stores each row's (x, y) as `np.interp(t_query, t_log, x_log)` (`physicalai.signals_at`). This
inverts that map: every row's (x, y) is projected onto the clip's OWN 100 Hz egomotion polyline and
the log's clock is read at the projection -- no cadence, stride or n_stack is assumed. A straight
line through (raw row index, recovered time) then gives

    dt_s          the true row step                         (MEASURED median 0.1006666 s)
    grid_start_s  the first camera row after the log origin  (MEASURED median +0.113 s)

and the label timeline IS the log's clock from its first sample (`egomotion_source.load`,
`ts - ts[0]`), so a row's label time is `grid_start_s + (row + n_stack - 1) * dt_s`.

CONTROLS, and the row is REFUSED (counted, never guessed) when any fails:
  K1  the line fit's max residual must be < 5 ms (the grid IS a linspace: MEASURED 0.0008 ms median)
  K3  the recovered times must reproduce the row's logged speed to < 1e-3 m/s
  and before any clip, a SYNTHETIC linspace at a KNOWN dt 0.1 / start 0.037 s must be recovered
  to < 1e-6 relative / < 0.01 ms (the inversion itself works) -- or the build aborts.

Rows are keyed on `sid = stable_episode_id(clip_id)` and carry NO clip id (the clip-id rule).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanitad.data import clip_clock as cc  # noqa: E402
from tanitad.data import egomotion_source as ES  # noqa: E402
from tanitad.data.v2_dataset import load_or_build_manifest, stable_episode_id  # noqa: E402

V_MIN = 2.0
K1_MAX_MS = 5.0
K3_MAX_MS = 1e-3


def _log(clip_id: str):
    idx = ES._index()
    if clip_id not in idx:
        return None
    path, member = idx[clip_id]
    import pandas as pd
    if member:
        with zipfile.ZipFile(path) as z:
            df = pd.read_parquet(io.BytesIO(z.read(member)))
    else:
        df = pd.read_parquet(path)
    t = df["timestamp"].to_numpy(np.float64)
    o = np.argsort(t)
    return (t[o] / 1e6, df["x"].to_numpy(np.float64)[o], df["y"].to_numpy(np.float64)[o],
            np.hypot(df["vx"].to_numpy(np.float64)[o], df["vy"].to_numpy(np.float64)[o]))


def _invert(xy: np.ndarray, t, x, y):
    ax, ay, dx, dy = x[:-1], y[:-1], np.diff(x), np.diff(y)
    L2 = dx * dx + dy * dy
    out = np.full(len(xy), np.nan)
    k_prev = 0
    for i, (px, py) in enumerate(xy):
        lo, hi = max(k_prev - 50, 0), min(k_prev + 3000, len(ax))
        u = np.where(L2[lo:hi] > 0, ((px - ax[lo:hi]) * dx[lo:hi] + (py - ay[lo:hi]) * dy[lo:hi])
                     / np.where(L2[lo:hi] > 0, L2[lo:hi], 1), 0.0).clip(0.0, 1.0)
        d = np.hypot(ax[lo:hi] + u * dx[lo:hi] - px, ay[lo:hi] + u * dy[lo:hi] - py)
        k = int(np.argmin(d))
        kk = lo + k
        out[i] = t[kk] + u[k] * (t[kk + 1] - t[kk])
        k_prev = kk
    return out


def fit_clock(poses: np.ndarray, n_stack: int, log) -> dict:
    """provider poses [T, 4] + the 100 Hz log -> {grid_start_s, dt_s, ...} or {"refused": why}."""
    t, x, y, v = log
    mv = poses[:, 3] > V_MIN
    if int(mv.sum()) < 20:
        return {"refused": "fewer than 20 moving rows"}
    tr = _invert(poses[:, :2], t, x, y)
    j = np.arange(len(poses)) + (n_stack - 1)
    A = np.stack([np.ones(int(mv.sum())), j[mv]], 1)
    (a, b), *_ = np.linalg.lstsq(A, tr[mv], rcond=None)
    res = tr[mv] - (a + b * j[mv])
    k1 = float(np.max(np.abs(res))) * 1e3
    k3 = float(np.max(np.abs(np.interp(a + b * j, t, v)[mv] - poses[mv, 3])))
    out = {"grid_start_s": float(a - t[0]), "dt_s": float(b), "fit_resid_max_ms": round(k1, 6),
           "k3_speed_maxabs_ms": k3, "n_moving_rows": int(mv.sum())}
    if k1 >= K1_MAX_MS:
        out["refused"] = f"K1 fit residual {k1:.3f} ms >= {K1_MAX_MS}"
    elif k3 >= K3_MAX_MS:
        out["refused"] = f"K3 speed residual {k3:.2e} m/s >= {K3_MAX_MS}"
    else:
        lo, hi = cc.POSE_DT_BAND_S
        if not (lo <= b <= hi) or abs(a - t[0]) > 1.0:
            out["refused"] = f"clock outside the physical band (dt {b}, start {a - t[0]})"
    return out


def self_test() -> dict:
    """K2: a synthetic linspace at a KNOWN clock must come back exactly."""
    t = np.arange(0.0, 30.0, 0.01)
    x, y = 12.0 * t, 0.3 * np.sin(0.2 * t)
    v = np.hypot(np.full_like(t, 12.0), 0.06 * np.cos(0.2 * t))
    tq = 0.037 + 0.1 * np.arange(190)
    poses = np.stack([np.interp(tq, t, x), np.interp(tq, t, y), np.zeros_like(tq),
                      np.interp(tq, t, v)], 1).astype(np.float32).astype(np.float64)
    f = fit_clock(poses, 1, (t, x, y, v))
    ok = (abs(f["dt_s"] / 0.1 - 1) < 1e-6 and abs(f["grid_start_s"] - 0.037) * 1e3 < 0.01
          and "refused" not in f)
    return {"ok": bool(ok), **f}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", action="append", default=[])
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    k2 = self_test()
    if not k2["ok"]:
        raise SystemExit(f"[clip-clock] ⛔ K2 self-test FAILED, nothing written: {k2}")
    if not (a.cache or a.manifest):
        raise SystemExit("[clip-clock] ⛔ give --cache and/or --manifest")
    import torch
    mans = ([load_or_build_manifest(cd, rebuild=False, verbose=False) for cd in a.cache]
            + [torch.load(m, map_location="cpu", weights_only=False) for m in a.manifest])
    rows, refused, no_log = [], {}, 0
    for man in mans:
        for i, cid in enumerate(man["clip_id"]):
            log = _log(cid)
            sid = stable_episode_id(cid)
            if log is None:
                no_log += 1
                continue
            f = fit_clock(man["poses"][i].double().numpy(), int(man["n_stack"][i]), log)
            if "refused" in f:
                refused[str(sid)] = f["refused"]
                continue
            rows.append({"sid": sid, **f})
    if not rows:
        raise SystemExit("[clip-clock] ⛔ ZERO clips clocked -- nothing written")
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    dts = sorted(r["dt_s"] for r in rows)
    g0s = sorted(r["grid_start_s"] for r in rows)
    here = Path(__file__).resolve()
    meta = {"schema": cc.SIDECAR_SCHEMA, "built_by": "stack/scripts/build_clip_clock_sidecar.py",
            "builder_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
            "caches": [str(c) for c in a.cache],
            "manifests_md5": [hashlib.md5(Path(m).read_bytes()).hexdigest() for m in a.manifest],
            "n_rows": len(rows),
            "n_refused": len(refused), "refused_by_sid": refused, "n_without_log": no_log,
            "dt_s_median": dts[len(dts) // 2], "grid_start_s_median": g0s[len(g0s) // 2],
            "K2_self_test": k2, "controls": {"K1_max_ms": K1_MAX_MS, "K3_max_ms": K3_MAX_MS}}
    with open(a.out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[clip-clock] {len(rows)} clips clocked, {len(refused)} refused, {no_log} without a "
          f"log; dt median {meta['dt_s_median']:.7f} s, grid start median "
          f"{meta['grid_start_s_median']:+.4f} s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
