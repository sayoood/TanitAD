#!/usr/bin/env python3
"""Synthetic fixture for verify_frames.py, built from FIRST-PARTY frames already local.

Gate log 2021.09.16.14.39.34_veh-42_01609_01687: 66 navtest tokens x 4 cameras; the frames come from the OpenScene test archives (refe_navtest/frames),
the timestamps from the local test nuPlan DB. A 10 Hz stream is faked by adding filler rows +-100..400 ms around each chosen frame.
Controls must read exactly what they should:
  control                              : 264/264 bytes equal, residual 0, exact timestamps equal
  zero control                         : reference vs itself 264/264
  mutation M1 (L0 <-> R0 swapped)      : byte gate must FAIL
  mutation M2 (timestamps +60 ms)      : byte gate AND exact-timestamp gate must FAIL. The residual gate does NOT (MEASURED 48.1 ms <= 50.5): a 10 Hz grid
                                         absorbs any shift < 50 ms by picking a neighbouring row, so G2 alone cannot see a clock offset; G3/G4 exist for that
  mutation M3 (one chosen row dropped) : byte gate must FAIL (1 missing or different)
  mutation M4 (one byte flipped)       : byte gate must FAIL (1 different)
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I code/test_verify_frames.py   (writes raw/test_verify_frames.out)
"""
import json
import os
import sqlite3
import sys
import tempfile
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_frames as V  # noqa: E402

LOG = "2021.09.16.14.39.34_veh-42_01609_01687"
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
rows = [r for r in json.load(open("D:/Projects/TanitAD/data/refe_navtest/frames/tokens_frames.json")) if r["log"] == LOG]
db = sqlite3.connect("file:D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test/%s.db?mode=ro" % LOG, uri=True)
ref, img_ts, lidar_ts = {}, {c: {} for c in CAMS}, {}
for ci, c in enumerate(CAMS):
    ctok = db.execute("SELECT token FROM camera WHERE channel=?", (c,)).fetchone()[0]
    nm2ts = {os.path.basename(fn): ts for ts, fn in db.execute("SELECT timestamp, filename_jpg FROM image WHERE camera_token=?", (ctok,))}
    z = zipfile.ZipFile("D:/Projects/TanitAD/data/refe_navtest/frames/%s_%s.zip" % (LOG, c))
    for r in rows:
        base = os.path.basename(r["cams"][ci])
        ref[(c, r["token"])] = z.read(base)
        img_ts[c][r["token"]] = nm2ts[base]
        lidar_ts[r["token"]] = r["timestamp_us"]


def build(cam, swap_to=None, shift_us=0, drop_token=None, flip_token=None):
    src = swap_to or cam
    tss, bs = [], []
    for r in rows:
        t = r["token"]
        if t == drop_token and cam == "CAM_F0":
            continue
        b = ref[(src, t)]
        if t == flip_token and cam == "CAM_F0":
            b = bytes([b[0] ^ 1]) + b[1:]
        base = img_ts[src][t] + shift_us
        tss.append(base)
        bs.append(b)
        for k in (-4, -3, -2, -1, 1, 2, 3, 4):
            tss.append(base + k * 100_000)
            bs.append(b"filler")
    o = np.argsort(tss, kind="stable")
    tss = np.asarray(tss)[o]
    bs = [bs[i] for i in o]
    keep, seen = [], set()
    for i, (t_, b_) in enumerate(zip(tss, bs)):
        if t_ in seen and b_ == b"filler":
            continue
        seen.add(t_)
        keep.append(i)
    return tss[keep], [bs[i] for i in keep]


def write_read(cam, **kw):
    ts, bs = build(cam, **kw)
    p = os.path.join(tempfile.mkdtemp(prefix="vf_"), cam + ".arrow")
    tbl = pa.table({"camera.timestamp": pa.array(ts, pa.int64()), "camera.jpeg": pa.array(bs, pa.binary())})
    with ipc.new_file(p, tbl.schema) as w:
        w.write_table(tbl)
    return V.read_arrow(p)


def run(**mut):
    sel, dts, picks = {}, [], {}
    for c in CAMS:
        kw = {}
        if mut.get("swap") and c in ("CAM_L0", "CAM_R0"):
            kw["swap_to"] = "CAM_R0" if c == "CAM_L0" else "CAM_L0"
        if mut.get("shift"):
            kw["shift_us"] = mut["shift"]
        if mut.get("drop") and c == "CAM_F0":
            kw["drop_token"] = rows[3]["token"]
        if mut.get("flip") and c == "CAM_F0":
            kw["flip_token"] = rows[5]["token"]
        ts, bs = write_read(c, **kw)
        q = [lidar_ts[r["token"]] for r in rows]
        pk, dt = V.nearest_rows(ts, q)
        dts.extend(dt.tolist())
        picks[c] = (ts, pk)
        for r, j in zip(rows, pk):
            sel[(c, r["token"])] = bs[j]
    return {"g3": V.g3_bytes_equal(sel, ref), "g2": V.g2_residual(dts),
            "g4": all(V.g4_exact_ts(picks[c][0], [img_ts[c][r["token"]] for r in rows], picks[c][1])["ok"] for c in CAMS)}


res = {"control": run(), "zero_control": V.g3_bytes_equal(ref, ref), "M1_swap_L0_R0": run(swap=True),
       "M2_shift_60ms": run(shift=60_000), "M3_drop_row": run(drop=True), "M4_flip_byte": run(flip=True)}
ok = (res["control"]["g3"]["ok"] and res["control"]["g2"]["ok"] and res["control"]["g4"] and res["zero_control"]["ok"]
      and res["control"]["g3"]["total"] == 4 * len(rows)
      and not res["M1_swap_L0_R0"]["g3"]["ok"] and not res["M2_shift_60ms"]["g3"]["ok"] and not res["M2_shift_60ms"]["g4"]
      and not res["M3_drop_row"]["g3"]["ok"] and not res["M4_flip_byte"]["g3"]["ok"])
res["ALL_AS_EXPECTED"] = bool(ok)
res["fixture"] = {"log": LOG, "tokens": len(rows), "frames_compared": 4 * len(rows)}


def _j(o):
    return bool(o) if isinstance(o, np.bool_) else str(o)


json.dump(res, open(os.path.join(HERE, "..", "raw", "test_verify_frames.out"), "w"), indent=1, default=_j)
print(json.dumps(res, indent=1, default=_j))
sys.exit(0 if ok else 1)
