"""Reconstruct Alpamayo's ground-truth future from egomotion — and PROVE it matches.

⛔ WHY THIS EXISTS. `sample_trajectories_from_data(data=model_inputs, ...)` gets
the model's INPUTS, which carry ego HISTORY but not the future. The GT future
lives in a different object inside `run_smoke`, so the trajectory capture came
back with `gt_xyz` empty. Re-running 40 samples to catch it would cost ~30 min of
GPU; the GT is just the ego's own future pose, which we already hold in the
staged `egomotion` parquet.

⭐ THE SELF-CHECK IS THE POINT. A reconstructed GT is worthless if the frame
convention is off by a sign or an axis — the error would be small, plausible, and
wrong. So this recomputes ADE between Alpamayo's OWN captured prediction and the
RECONSTRUCTED GT and compares it to the `min_ade_m` NVIDIA's code printed for the
same sample. Agreement to a few mm proves the convention; disagreement means the
reconstruction is rejected, not patched until it looks right.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lead_block_helpers import ego_series, index_zips, read_member  # noqa: E402


def future_in_t0_frame(ego, t0_s, ts_rel):
    """Ego positions at t0+ts_rel expressed in the ego frame at t0 -> [K,2].

    x forward, y left — the convention `ego_future_xyz` uses and the one our own
    `gt_ego_waypoints` uses, which is what makes the two comparable at all."""
    t = np.asarray(ego["t"])
    x0 = np.interp(t0_s, t, ego["x"])
    y0 = np.interp(t0_s, t, ego["y"])
    yaw0 = np.interp(t0_s, t, ego["yaw"])
    tq = t0_s + np.asarray(ts_rel)
    dx = np.interp(tq, t, ego["x"]) - x0
    dy = np.interp(tq, t, ego["y"]) - y0
    c, s = np.cos(yaw0), np.sin(yaw0)
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--ego-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max |ADE_reconstructed - ADE_reported| to accept, metres")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    rows = [r for r in rows if "min_ade_m" in r]
    idx = index_zips(a.ego_dir)
    ts_rel = np.arange(1, 65) * 0.1                     # 0.1 .. 6.4 s

    checked, out = [], {}
    for r in rows:
        i, clip = r["sample_index"], r["clip_id"]
        npz = os.path.join(a.traj_dir, f"traj_{i:04d}.npz")
        if not os.path.exists(npz) or clip not in idx:
            continue
        P = np.load(npz)["pred_xyz"]
        P = np.asarray(P).reshape(-1, P.shape[-2], P.shape[-1])[0][:, :2]
        df = read_member(idx[clip], clip)
        if df is None:
            continue
        G = future_in_t0_frame(ego_series(df), r["t0_us"] / 1e6, ts_rel)
        ade = float(np.linalg.norm(P - G, axis=-1).mean())
        checked.append({"i": i, "clip_id": clip, "ade_recon": ade,
                        "ade_reported": r["min_ade_m"],
                        "delta": ade - r["min_ade_m"]})
        out[i] = G.tolist()

    if not checked:
        raise SystemExit("nothing to check")
    d = np.array([c["delta"] for c in checked])
    print(f"convention check over {len(checked)} samples:")
    for c in checked[:8]:
        print(f"   i={c['i']:>3}  recon {c['ade_recon']:8.4f}  "
              f"reported {c['ade_reported']:8.4f}  delta {c['delta']:+.4f}")
    print(f"  |delta| mean {np.abs(d).mean():.4f}  max {np.abs(d).max():.4f} m")
    ok = np.abs(d).max() <= a.tol
    print(f"  VERDICT: {'MATCH — convention confirmed' if ok else 'MISMATCH — REJECTED'}")
    if not ok:
        # ⛔ Do not ship a GT that does not reproduce their own metric.
        raise SystemExit("reconstructed GT does not reproduce the reported ADE; "
                         "refusing to write it")
    json.dump({"gt_xy_by_index": out, "ts_rel_s": ts_rel.tolist(),
               "check": checked,
               "_validated": "reconstructed GT reproduces NVIDIA's own printed "
                             "min_ade_m to within %.4f m" % np.abs(d).max()},
              open(a.out, "w"))
    print(f"[out] {a.out}  {len(out)} clips")


if __name__ == "__main__":
    main()
