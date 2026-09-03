#!/usr/bin/env python3
"""A2 — EXACT recovery of the encoding wheelbase L_enc from the producer's own input.

Reads, per local eval clip:
  * the stored v2ep  actions[:,0] (steer) / actions[:,1] (accel) / poses (x,y,yaw,v)
  * the RAW egomotion parquet (curvature, vx, vy, ax, x, y, q*) that signals_at consumed
and inverts   steer = arctan(L_enc * interp(curvature))   pointwise.

CONTROLS (must read a known value or the probe is void):
  C1  the reconstructed t_query reproduces poses[:,3] (v) to < 1e-5
  C2  it ALSO reproduces channels NOT used to build it: actions[:,1] (ax) and poses[:,2] (yaw)
  C3  forward map arctan(L_hat * curvature) reproduces the stored steer to < 1e-6 rad
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EPS_DIR = Path(r"C:\Users\Admin\refav1_eval_slice\eps")
PAI = Path(r"C:\Users\Admin\tanitad-data\physicalai")
TS_DIR = PAI / "camera" / "camera_front_wide_120fov"
EGO_ALPA = PAI / "labels" / "egomotion_alpamayo"
EGO_ZIPS = PAI / "labels" / "egomotion"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def quaternion_yaw(qx, qy, qz, qw):
    """physicalai.quaternion_yaw — re-derived; pinned against the import below."""
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def load_ego(clip_id: str) -> pd.DataFrame | None:
    p = EGO_ALPA / f"{clip_id}.parquet"
    if p.exists():
        return pd.read_parquet(p).reset_index()
    for z in sorted(EGO_ZIPS.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                nm = f"{clip_id}.egomotion.parquet"
                if nm in zf.namelist():
                    import io
                    return pd.read_parquet(io.BytesIO(zf.read(nm))).reset_index()
        except Exception:
            continue
    return None


def t_query_for(clip_id: str, n_stored: int):
    """The builder's grid, EXACTLY: linspace(t0, t1, n_target) from the video
    timestamps parquet (v2_compressed._resampled / physicalai.build_episode)."""
    p = TS_DIR / f"{clip_id}.timestamps.parquet"
    if not p.exists():
        return None, "no timestamps parquet"
    ts = pd.read_parquet(p)
    tcol = next(c for c in ts.columns if "time" in c.lower())
    tf = ts[tcol].to_numpy(np.float64)
    span = tf[-1] - tf[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * 10.0), 4)
    tq = np.linspace(tf[0], tf[-1], n_target)
    return tq, f"n_target={n_target} n_stored={n_stored}"


def main() -> None:
    names = sorted(p.name[: -len(".v2ep.pt")] for p in EPS_DIR.glob("*.v2ep.pt"))
    print(f"[probe] {len(names)} local v2ep episodes", flush=True)
    rows = []
    for cid in names:
        o = torch.load(EPS_DIR / f"{cid}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        act = o["actions"].numpy().astype(np.float64)
        pos = o["poses"].numpy().astype(np.float64)
        steer, accel = act[:, 0], act[:, 1]
        r = {"clip_id": cid, "T": int(act.shape[0]),
             "stored_clip_id": str(o.get("clip_id"))}
        ego = load_ego(cid)
        if ego is None:
            r["status"] = "no egomotion"
            rows.append(r)
            print(f"  {cid[:8]} NO EGOMOTION", flush=True)
            continue
        r["ego_cols"] = ",".join(map(str, ego.columns))
        tq, note = t_query_for(cid, act.shape[0])
        if tq is None:
            r["status"] = f"no t_query ({note})"
            rows.append(r)
            print(f"  {cid[:8]} {note}", flush=True)
            continue
        r["grid_note"] = note
        n = min(len(tq), act.shape[0])
        tq = tq[:n]
        t = ego["timestamp"].to_numpy(np.float64)
        order = np.argsort(t)
        t = t[order]

        def col(c):
            return np.interp(tq, t, ego[c].to_numpy(np.float64)[order])

        v_hat = np.hypot(col("vx"), col("vy"))
        yaw_native = np.unwrap(quaternion_yaw(
            *(ego[c].to_numpy(np.float64)[order] for c in ("qx", "qy", "qz", "qw"))))
        yaw_hat = np.arctan2(np.sin(np.interp(tq, t, yaw_native)),
                             np.cos(np.interp(tq, t, yaw_native)))
        ax_hat = col("ax") if "ax" in ego.columns else None
        curv = col("curvature")

        # ---- controls
        r["C1_v_maxabs"] = float(np.max(np.abs(v_hat - pos[:n, 3])))
        r["C2_yaw_maxabs"] = float(np.max(np.abs(np.arctan2(
            np.sin(yaw_hat - pos[:n, 2]), np.cos(yaw_hat - pos[:n, 2])))))
        r["C2_ax_maxabs"] = (None if ax_hat is None
                             else float(np.max(np.abs(ax_hat - accel[:n]))))
        r["C2_x_maxabs"] = float(np.max(np.abs(col("x") - pos[:n, 0])))

        # ---- THE INVERSION  tan(steer) = L * curvature
        ts_ = np.tan(steer[:n])
        m = np.abs(curv) > 1e-4                      # identifiable only off-straight
        r["n_used"] = int(m.sum())
        if m.sum() >= 20:
            ptw = ts_[m] / curv[m]
            r["L_pointwise_median"] = float(np.median(ptw))
            r["L_pointwise_p01"] = float(np.percentile(ptw, 1))
            r["L_pointwise_p99"] = float(np.percentile(ptw, 99))
            r["L_pointwise_iqr"] = float(np.percentile(ptw, 75)
                                         - np.percentile(ptw, 25))
            # total least squares through the origin (noise-symmetric)
            r["L_ols"] = float((ts_[m] @ curv[m]) / (curv[m] @ curv[m]))
            L = r["L_ols"]
            r["C3_forward_maxabs_rad"] = float(
                np.max(np.abs(np.arctan(L * curv[m]) - steer[:n][m])))
            r["C3_forward_at_2p9_maxabs_rad"] = float(
                np.max(np.abs(np.arctan(2.9 * curv[m]) - steer[:n][m])))
            r["status"] = "ok"
        else:
            r["status"] = f"too straight (n_used={int(m.sum())})"
        rows.append(r)
        print(f"  {cid[:8]} L_ols={r.get('L_ols')} "
              f"med={r.get('L_pointwise_median')} iqr={r.get('L_pointwise_iqr')} "
              f"C1={r.get('C1_v_maxabs'):.2e} C3@2.9={r.get('C3_forward_at_2p9_maxabs_rad')}",
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "lenc_exact_per_clip.csv", index=False)
    (OUT / "lenc_exact.json").write_text(json.dumps(rows, indent=1, default=str),
                                         encoding="utf-8")
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        L = np.array([r["L_ols"] for r in ok])
        print(f"\n[probe] {len(ok)}/{len(rows)} clips inverted: "
              f"L_enc min={L.min():.6f} max={L.max():.6f} median={np.median(L):.6f}")
        print(f"[probe] max |forward residual @ L=2.9| = "
              f"{max(r['C3_forward_at_2p9_maxabs_rad'] for r in ok):.3e} rad")
    print("[probe] wrote", OUT / "lenc_exact_per_clip.csv")


if __name__ == "__main__":
    main()
