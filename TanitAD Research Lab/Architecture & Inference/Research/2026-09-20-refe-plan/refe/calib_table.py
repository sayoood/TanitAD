"""Per-log camera calibration -- what PETR's 3D position embedding is built from.

⛔⛔ WHY THIS EXISTS (R22, 2026-09-23). REFe baked ONE vehicle's camera calibration into
`REFeConfig` and used it for every sample. The paper (p. 8) enriches the visual tokens with "3D
position embeddings [50]" -- [50] is PETR, which builds that embedding from EACH SAMPLE'S OWN
intrinsics and extrinsics. MEASURED over the 214 navtrain logs held locally: **12 vehicles, 12
distinct rigs per camera**; against the baked rig the camera ROTATION differs by a median
0.84-1.52 deg and up to **2.67 deg** (CAM_F0), translation by up to 9.3 cm, and one vehicle's
intrinsics by up to 38 px. The model's own comment called the gap "practically small -- the rigs
differ by centimetres": true of the translation, and silent about the rotation, which is what
displaces a lifted feature (~1.4 m at 30 m for 2.67 deg).

One vector per camera, native pixel units, in this order (16 floats):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, tx, ty, tz, qw, qx, qy, qz
-- exactly the fields `REFeConfig` bakes (`cam_fx..cam_cy`, `cam_distortion`, `cam_t`, `cam_q`),
from the same source (the nuPlan DB `camera` table), in the same conventions (Caltech distortion;
`R(q: w,x,y,z) @ p_cam + t -> p_ego`, metres).

Usage:  python calib_table.py --db-root <dir of .db> --out calib_table.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sqlite3
import sys

FIELDS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
          "tx", "ty", "tz", "qw", "qx", "qy", "qz")
CAMERAS_ALL = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0", "CAM_L1", "CAM_L2", "CAM_R1", "CAM_R2")
NATIVE_WH = (1920, 1080)


def _unpickle(b):
    # the DB pickles NESTED lists (a 3x3 intrinsic is [[..],[..],[..]]) or arrays -- flatten both
    import numpy as np
    return [float(x) for x in np.asarray(pickle.loads(b), dtype=float).ravel().tolist()]


def read_db(db_path: str, cameras=CAMERAS_ALL) -> dict:
    """{camera: [16 floats]} from one log DB's `camera` table. Raises on anything it cannot read."""
    c = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(camera)").fetchall()]
        out = {}
        for ch in cameras:
            row = c.execute("SELECT * FROM camera WHERE channel=?", (ch,)).fetchone()
            if row is None:
                continue
            r = dict(zip(cols, row))
            K = _unpickle(r["intrinsic"])            # 3x3 row-major
            dist = _unpickle(r["distortion"])[:5]
            t = _unpickle(r["translation"])[:3]
            q = _unpickle(r["rotation"])[:4]         # w, x, y, z (pyquaternion order)
            wh = (int(r.get("width", 0) or 0), int(r.get("height", 0) or 0))
            # ⛔ the model scales K by img/native with ONE native size; a rig at another native
            # resolution would be mis-scaled silently, so refuse it here instead
            if wh != NATIVE_WH:
                raise ValueError(f"{os.path.basename(db_path)} {ch}: native {wh} != {NATIVE_WH}")
            if len(K) != 9 or len(dist) != 5 or len(t) != 3 or len(q) != 4:
                raise ValueError(f"{os.path.basename(db_path)} {ch}: malformed calibration")
            out[ch] = [K[0], K[4], K[2], K[5], *dist, *t, *q]
        return out
    finally:
        c.close()


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("fields") != list(FIELDS):
        raise ValueError(f"{path}: field order {d.get('fields')} != {list(FIELDS)}")
    return d


def vectors(table: dict, log_name: str, cameras) -> list | None:
    """[n_cam][16] for one log in the rig's camera order, or None if any camera is missing."""
    e = table["logs"].get(log_name)
    if e is None or any(ch not in e for ch in cameras):
        return None
    return [e[ch] for ch in cameras]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-root", required=True, help="directory of nuPlan .db files (searched recursively)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dbs = sorted(glob.glob(os.path.join(a.db_root, "**", "*.db"), recursive=True))
    logs, bad = {}, []
    for p in dbs:
        try:
            logs[os.path.basename(p)[:-3]] = read_db(p)
        except Exception as e:  # noqa: BLE001
            bad.append((os.path.basename(p), f"{type(e).__name__}: {str(e)[:80]}"))
    rigs = {ch: len({tuple(v[ch]) for v in logs.values() if ch in v}) for ch in CAMERAS_ALL}
    out = {"fields": list(FIELDS), "native_wh": list(NATIVE_WH), "source": "nuPlan DB `camera` table",
           "n_logs": len(logs), "distinct_rigs_exact": rigs, "unreadable": bad, "logs": logs}
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(tmp, a.out)
    print(f"  calib table: {len(logs):,} logs, {len(bad)} unreadable -> {a.out}")
    print(f"  distinct EXACT rigs per camera: {rigs}")
    for b in bad[:5]:
        print(f"  unreadable: {b}")
    # ⛔ assert on the artifact: every DB found must be in the table, or say which are not
    print("ZZCALIB_TABLE_OKZZ" if logs and not bad else f"ZZCALIB_TABLE_PARTIAL_{len(bad)}ZZ")
    return 0 if logs and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
