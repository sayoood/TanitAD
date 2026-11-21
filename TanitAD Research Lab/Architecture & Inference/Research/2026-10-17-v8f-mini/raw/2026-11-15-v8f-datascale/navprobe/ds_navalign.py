#!/usr/bin/env python3
"""DATASCALE nav alignment probe (pod-side, CPU, read-only).
Second, independent check on navigation/navigation.parquet, against the ego's own motion for the clips
whose egomotion parquet is kept in /root/datascale/keep10k/*/ego_members/:
  (1) countdown: in a run of rows '<manoeuvre> in <d>m' with one manoeuvre text, regress d on the ego's
      arc length s at the row's timestamp. A slope near -1 means the distance is measured along the path
      the ego drove.
  (2) direction: for 'Turn left' / 'Turn right' runs whose manoeuvre point lies inside the clip, compare
      the announced side with the sign of the ego's heading change around that point (x forward, y left,
      yaw counter-clockwise positive, so left = positive).
Writes one JSON. No address, port, pod id or token in this file."""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

META = "/root/datascale/meta"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/datascale/navalign.json"
MAXCLIPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
RX = re.compile(r"^(?P<man>.+?) in (?P<d>\d+)m$")
TURN_DEG = 30.0


def yaw(q):
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def main():
    files = {os.path.basename(f).split(".")[0]: f for f in glob.glob("/root/datascale/keep10k/*/ego_members/*.egomotion.parquet")}
    nav = pd.read_parquet(f"{META}/navigation/navigation.parquet", columns=["clip_id", "timestamp", "nav_text"])
    nav["clip_id"] = nav["clip_id"].astype(str)
    clips = sorted(set(files) & set(nav["clip_id"].unique()))[:MAXCLIPS]
    nav = nav[nav["clip_id"].isin(set(clips))]
    rep = {"ego_files": len(files), "clips_used": len(clips), "turn_deg": TURN_DEG}
    slopes, r2s, dmax, offs = [], [], [], []
    conf = {}
    ego_t_range = []
    for c, g in nav.groupby("clip_id"):
        e = pd.read_parquet(files[c]).sort_values("timestamp")
        et = e["timestamp"].to_numpy(np.float64)
        xy = e[["x", "y"]].to_numpy(np.float64)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        hd = yaw(e[["qx", "qy", "qz", "qw"]].to_numpy(np.float64))
        ego_t_range.append((et[0], et[-1]))
        g = g.sort_values("timestamp")
        m = g["nav_text"].astype(str).str.extract(RX)
        g = g.assign(man=m["man"], d=pd.to_numeric(m["d"], errors="coerce")).dropna(subset=["man"])
        if g.empty:
            continue
        run = (g["man"] != g["man"].shift()).cumsum()
        for _, r in g.groupby(run):
            t = r["timestamp"].to_numpy(np.float64)
            d = r["d"].to_numpy(np.float64)
            inside = (t >= et[0]) & (t <= et[-1])
            t, d = t[inside], d[inside]
            if len(t) < 5 or d.max() - d.min() < 10:
                continue
            st = np.interp(t, et, s)
            A = np.vstack([st, np.ones_like(st)]).T
            coef, *_ = np.linalg.lstsq(A, d, rcond=None)
            pred = A @ coef
            ss = ((d - d.mean()) ** 2).sum()
            slopes.append(float(coef[0]))
            r2s.append(float(1 - ((d - pred) ** 2).sum() / ss) if ss > 0 else float("nan"))
            dmax.append(float(d.max()))
            offs.append(float(np.median(d + st)))
            man = str(r["man"].iloc[0])
            side = "left" if man.lower().startswith("turn left") else "right" if man.lower().startswith("turn right") else None
            if side is None:
                continue
            sp = float(np.median(d + st))
            if sp - 15 < 0 or sp + 30 > s[-1]:
                continue
            h0 = np.interp(sp - 15, s, hd)
            h1 = np.interp(sp + 30, s, hd)
            dh = np.degrees(h1 - h0)
            obs = "left" if dh > TURN_DEG else "right" if dh < -TURN_DEG else "none"
            conf.setdefault(side, {}).setdefault(obs, 0)
            conf[side][obs] += 1
    sl = np.array(slopes)
    rr = np.array(r2s)
    rep["countdown_runs"] = int(len(sl))
    if len(sl):
        rep["slope_quantiles"] = {q: float(np.nanquantile(sl, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        rep["r2_quantiles"] = {q: float(np.nanquantile(rr, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        rep["frac_slope_in_-1.1_-0.9"] = float(np.mean((sl > -1.1) & (sl < -0.9)))
        rep["frac_slope_in_-1.1_-0.9_and_r2_ge_0.99"] = float(np.mean((sl > -1.1) & (sl < -0.9) & (rr >= 0.99)))
        rep["first_announce_distance_m_quantiles"] = {q: float(np.quantile(dmax, q)) for q in (0.05, 0.5, 0.95)}
    rep["turn_side_vs_observed_heading"] = conf
    tot = sum(v for o in conf.values() for v in o.values())
    agree = sum(conf.get(k, {}).get(k, 0) for k in ("left", "right"))
    rep["turn_n"] = int(tot)
    rep["turn_agree"] = int(agree)
    et0 = np.array(ego_t_range)
    if len(et0):
        rep["ego_t0_quantiles_us"] = {q: float(np.quantile(et0[:, 0], q)) for q in (0.05, 0.5, 0.95)}
        rep["ego_t1_quantiles_us"] = {q: float(np.quantile(et0[:, 1], q)) for q in (0.05, 0.5, 0.95)}
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print("NAVALIGN OK", OUT, rep["clips_used"], rep["countdown_runs"], rep["turn_n"], rep["turn_agree"])


if __name__ == "__main__":
    main()
