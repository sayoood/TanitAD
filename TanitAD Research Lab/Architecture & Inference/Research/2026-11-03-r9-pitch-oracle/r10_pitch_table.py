#!/usr/bin/env python3
"""r10_pitch_table.py -- per-frame egomotion pitch targets H and C for the eval clips, plus the SIGN / PHYSICALITY reads (R10-A, 2026-11-03).

Targets (REFCV9_DESIGN 2.3 R10; labels may use ego, PI 2026-08-03). ``pitch`` = ``euler_zyx_deg(q)[:, 1]`` of the egomotion quaternion
(VERBATIM from ``e0bc_census.py``; intrinsic ZYX = R = Rz(yaw) Ry(pitch) Rx(roll)), in degrees, 100 Hz:
  H = pitch - its 3 s centred moving average (``hp_dev``, 300 samples, edge-padded)   (the dynamic part)
  C = pitch - the clip mean                                                          (the design's definition)
SIGN: with the rig frame x forward, y LEFT, z up (``rig_projection.py``), ``Ry(+theta)`` takes the forward axis toward -z: pitch > 0 is NOSE
DOWN. That is a convention claim about the quaternion, so it is READ from the data as well (``sign_read``): braking (ax < 0) must dive the nose
(H > 0), i.e. the slope of H on the high-passed ax is NEGATIVE; and the grade (uphill, z rises) must read negative pitch.
CLOCK: cache frame i = raw grid row k = i + 2 (``physicalai.py`` ``poses_t = poses[n_stack-1:]``; the v9 label manifest), at
t = grid_start_s + k * dt_s from the clip-clock sidecar; egomotion timestamps are microseconds on the same clock. Pose-sync is OFF (the banked
rasters' clock). Output: counts and sha12 only -- no clip id is ever written.

    r10_pitch_table.py build --out table.npz   (defaults: the dev-box egomotion parquet dir, the refcv8 clock sidecar, the eval139 cache)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

K_OFF = 2                    # raw grid row = cache frame index + (n_stack - 1)
N_FRAMES = 210               # cache episodes carry 201 frames


def euler_zyx_deg(q):          # VERBATIM e0bc_census.py
    """q [N, 4] (qx, qy, qz, qw) -> [N, 3] (yaw, pitch, roll) degrees."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return np.degrees(np.stack([yaw, pitch, roll], 1))


def hp_dev(a, win=300):        # VERBATIM e0bc_census.py
    """deviation of a 100 Hz angle series from its centred moving average (win samples = 3 s)."""
    a = np.asarray(a, np.float64)
    pad = np.pad(a, (win // 2, win - win // 2), mode="edge")
    c = np.concatenate([[0.0], np.cumsum(pad)])
    return a - (c[win:win + len(a)] - c[:len(a)]) / win


def sha12_of(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:12]


def clip_series(df):
    """-> dict of 100 Hz arrays sorted by time: t_s, pitch_deg, H_deg, C_deg, ax, ax_hp, x, y, z."""
    d = df.sort_values("timestamp")
    t = d["timestamp"].to_numpy(np.float64) * 1e-6
    p = euler_zyx_deg(d[["qx", "qy", "qz", "qw"]].to_numpy(np.float64))[:, 1]
    ax = d["ax"].to_numpy(np.float64)
    return {"t_s": t, "pitch_deg": p, "H_deg": hp_dev(p), "C_deg": p - p.mean(), "ax": ax, "ax_hp": hp_dev(ax),
            "x": d["x"].to_numpy(np.float64), "y": d["y"].to_numpy(np.float64), "z": d["z"].to_numpy(np.float64)}


def at_frames(series, grid_start_s, dt_s, n=N_FRAMES, k_off=K_OFF):
    tf = grid_start_s + (np.arange(n) + k_off) * dt_s
    return {k: np.interp(tf, series["t_s"], series[k]) for k in ("pitch_deg", "H_deg", "C_deg")}


def _slope(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    return (float(((x - xm) * (y - ym)).sum() / sxx) if sxx > 0 else float("nan")), float(np.corrcoef(x, y)[0, 1])


def sign_read(all_series, n_boot=2000, seed=0):
    """Slope of H (deg) on ax_hp (m/s^2), pooled over clips with a clip-cluster bootstrap; and of the 3 s-low-passed pitch on the grade angle."""
    X = [s["ax_hp"][::10] for s in all_series]
    Y = [s["H_deg"][::10] for s in all_series]                    # 10 Hz thinning (autocorrelation)

    def pool(ix):
        return _slope(np.concatenate([X[i] for i in ix]), np.concatenate([Y[i] for i in ix]))

    sl, r = pool(range(len(X)))
    rng = np.random.default_rng(seed)
    bs = [pool(rng.integers(0, len(X), len(X)))[0] for _ in range(n_boot)]
    out = {"H_on_ax_hp": {"slope_deg_per_mps2": sl, "corr": r, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          "n_clips": len(X), "n_samples": int(sum(len(x) for x in X)),
                          "estimator": "clip-cluster bootstrap B %d seed %d, 10 Hz thinned" % (n_boot, seed),
                          "expected_if_nose_down_positive": "NEGATIVE (braking ax < 0 dives the nose => H > 0)"}}
    GX, GY = [], []
    for s in all_series:
        k = 150
        cs = np.cumsum(np.hypot(np.gradient(s["x"]), np.gradient(s["y"])))
        dz = np.zeros_like(s["z"])
        dd = np.zeros_like(s["z"])
        dz[k:-k] = s["z"][2 * k:] - s["z"][:-2 * k]
        dd[k:-k] = cs[2 * k:] - cs[:-2 * k]
        ok = dd > 5.0
        grade = np.degrees(np.arctan2(dz, np.where(ok, dd, 1.0)))
        lp = s["pitch_deg"] - s["H_deg"]
        GX.append(grade[ok][::10])
        GY.append(lp[ok][::10])
    gx, gy = np.concatenate(GX), np.concatenate(GY)
    if len(gx) > 100:
        sl2, r2 = _slope(gx, gy)
        out["lowpass_pitch_on_grade_deg"] = {"slope": sl2, "corr": r2, "n_samples": int(len(gx)),
                                             "note": "uphill (z rises) is nose-UP; slope ~ -1 would confirm positive = nose down with z up AND that the quaternion carries the road grade"}
    return out


def build(a):
    import pandas as pd
    sys.path.insert(0, str(Path(a.stack)))
    from tanitad.data.v2_dataset import stable_episode_id
    clips = sorted(Path(f).name.split(".")[0] for f in Path(a.clips).glob("*.v2ep.pt"))
    clock = {}
    for line in open(a.clock, encoding="utf-8"):
        r = json.loads(line)
        clock[int(r["sid"])] = r
    tab, series, miss = {}, [], []
    for c in clips:
        sid = int(stable_episode_id(c))
        if sid not in clock:
            miss.append(sha12_of(c))
            continue
        s = clip_series(pd.read_parquet(Path(a.ego_dir) / f"{c}.parquet"))
        assert np.all(np.diff(s["t_s"]) >= 0)
        f = at_frames(s, float(clock[sid]["grid_start_s"]), float(clock[sid]["dt_s"]))
        h = sha12_of(c)
        for k, v in f.items():
            tab[f"{h}__{k}"] = v.astype(np.float32)
        series.append(s)
    sr = sign_read(series)
    allH = np.concatenate([tab[k] for k in tab if k.endswith("__H_deg")])
    allC = np.concatenate([tab[k] for k in tab if k.endswith("__C_deg")])
    meta = {"tool": "r10_pitch_table.py", "n_clips": len(series), "clips_without_clock_sidecar_row_sha12": miss, "k_off": K_OFF, "n_frames": N_FRAMES,
            "H_abs_deg_p50_p95_p99": [float(np.percentile(np.abs(allH), q)) for q in (50, 95, 99)],
            "C_abs_deg_p50_p95_p99": [float(np.percentile(np.abs(allC), q)) for q in (50, 95, 99)], "sign_read": sr,
            "clock": "t = grid_start_s + (i + 2) * dt_s (clip-clock sidecar), pose-sync OFF", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    tab["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(a.out, **tab)
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n_clips", "H_abs_deg_p50_p95_p99", "C_abs_deg_p50_p95_p99")}), flush=True)
    print(json.dumps(sr, indent=1), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ego-dir", default="C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo")
    b.add_argument("--clock", default="E:/refcv8_eval_kit/data/refcv8_clip_clock_sidecar_ts33.jsonl")
    b.add_argument("--clips", default="E:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139")
    b.add_argument("--stack", default=str(Path(__file__).resolve().parents[3] / "stack"))
    b.add_argument("--out", required=True)
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
