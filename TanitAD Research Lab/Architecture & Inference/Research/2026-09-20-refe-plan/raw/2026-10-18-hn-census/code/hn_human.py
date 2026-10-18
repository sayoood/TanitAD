#!/usr/bin/env python3
"""P6-HN step 0 -- SECONDARY expert E_H (PREREG s1): the logged human future from the nuPlan DB, for the held-out logs whose DB is present locally.

For a key (log, token): t0 = lidar_pc.timestamp; frame = the ego pose at lidar_pc.ego_pose_token (x forward, y left); the 20 poses at t0 + 0.2*k s (k=1..20), linear interpolation of
ego_pose x, y over the log's ego_pose series.  NaN where the DB is absent, the token is not found, or the future runs past the log.  Read-only sqlite.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I hn_human.py --joined ../raw/hn_joined.npz --out ../raw/human_future.npz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import sqlite3

SPLITS = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
DT_US = 200_000
T = 20


def find_db(log: str):
    for p in (f"{SPLITS}/{log}.db", f"{SPLITS}/trainval/{log}.db", f"{SPLITS}/test/{log}.db", f"{SPLITS}/mini/{log}.db"):
        if os.path.exists(p):
            return p
    return None


def yaw_from_quat(qw, qx, qy, qz):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def to_local(x, y, x0, y0, yaw0):
    dx, dy = x - x0, y - y0
    c, s = np.cos(yaw0), np.sin(yaw0)
    return c * dx + s * dy, -s * dx + c * dy


def future_for_token(ts, xs, ys, t0, x0, y0, yaw0):
    tq = t0 + DT_US * np.arange(1, T + 1)
    if tq[-1] > ts[-1] or t0 < ts[0]:
        return None
    fx, fy = np.interp(tq, ts, xs), np.interp(tq, ts, ys)
    lx, ly = to_local(fx, fy, x0, y0, yaw0)
    return np.stack([lx, ly], -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.joined)
    logs, toks = z["keys_log"], z["keys_token"]
    N = len(logs)
    out = np.full((N, T, 2), np.nan, np.float32)
    cov = {}
    for lg in np.unique(logs):
        p = find_db(str(lg))
        ix = np.where(logs == lg)[0]
        if p is None:
            cov[str(lg)] = {"db": None, "sets": int(len(ix)), "covered": 0}
            continue
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        ep = np.array(c.execute("select timestamp, x, y, qw, qx, qy, qz from ego_pose order by timestamp").fetchall(), dtype=np.float64)
        ts, xs, ys = ep[:, 0], ep[:, 1], ep[:, 2]
        n_ok = 0
        for i in ix:
            r = c.execute("select l.timestamp, e.x, e.y, e.qw, e.qx, e.qy, e.qz from lidar_pc l join ego_pose e on e.token = l.ego_pose_token where hex(l.token) = ?", (str(toks[i]).upper(),)).fetchone()
            if r is None:
                continue
            t0, x0, y0, qw, qx, qy, qz = r
            f = future_for_token(ts, xs, ys, float(t0), x0, y0, float(yaw_from_quat(qw, qx, qy, qz)))
            if f is not None:
                out[i] = f
                n_ok += 1
        c.close()
        cov[str(lg)] = {"db": os.path.relpath(p, SPLITS).replace("\\", "/"), "sets": int(len(ix)), "covered": int(n_ok)}
    np.savez_compressed(a.out, human=out)
    json.dump(cov, open(os.path.splitext(a.out)[0] + "_coverage.json", "w"), indent=1)
    print("covered sets", int((~np.isnan(out).any(axis=(1, 2))).sum()), "of", N, "logs with DB", sum(1 for v in cov.values() if v["db"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
