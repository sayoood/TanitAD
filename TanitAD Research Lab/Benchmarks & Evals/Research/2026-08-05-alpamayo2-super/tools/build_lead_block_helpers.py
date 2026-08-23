"""Egomotion readers shared with tools/build_lead_block.py — one implementation.

Kept as a tiny module rather than copy-pasted so the clip->time registration the
flagship alignment depends on cannot drift from the one the lead block uses.
"""
import io, os, zipfile
import numpy as np
import pandas as pd


def _quat_yaw(qx, qy, qz, qw):
    try:
        from lead_state_gate import quaternion_yaw
        return quaternion_yaw(qx, qy, qz, qw)
    except Exception:
        return np.arctan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))


def index_zips(d):
    idx = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".zip"):
            continue
        p = os.path.join(d, f)
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.endswith(".parquet"):
                    idx.setdefault(n.rsplit("/", 1)[-1].split(".")[0], p)
    return idx


def read_member(zpath, clip):
    if not zpath or not os.path.exists(zpath):
        return None
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.endswith(".parquet") and n.rsplit("/", 1)[-1].startswith(clip):
                return pd.read_parquet(io.BytesIO(z.read(n)))
    return None


def ego_series(df):
    t = df["timestamp"].to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    g = lambda c: df[c].to_numpy(np.float64)[o]                     # noqa: E731
    yaw = np.unwrap(_quat_yaw(g("qx"), g("qy"), g("qz"), g("qw")))
    return {"t": t[o], "x": g("x"), "y": g("y"), "yaw": yaw,
            "v": np.hypot(g("vx"), g("vy"))}
