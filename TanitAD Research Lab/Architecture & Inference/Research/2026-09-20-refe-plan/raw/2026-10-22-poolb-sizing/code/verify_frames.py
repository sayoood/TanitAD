#!/usr/bin/env python3
"""Post-download gates for pool-B frames (mirror Arrow route). Pure functions. Needs pyarrow (tanitad venv) for read_arrow only.

UNVERIFIED ASSUMPTION (settled by probe P0 before any bulk fetch): each camera.pcam_*.arrow has one timestamp column (microseconds) and one binary column holding
the JPEG bytes. The column discovery in read_arrow is a guess that P0 must confirm; every gate is a pure function of (timestamps, bytes), so only read_arrow changes
if the schema differs.

Gates (thresholds are MEASURED baselines, not tuned on pool B):
  G1 sha256 of the file == the HF LFS oid in the tree listing
  G2 nearest-row residual <= 50.5 ms for every (anchor, camera)  [baseline max 49.92 ms on 8,192 DB pairs, raw/residual_baseline.json]; images decode at 1920x1080
  G3 byte equality against a first-party reference (the gate log; the 3 first-party pool-B prefixes)
  G4 exact timestamp equality with the DB image row on DB-local logs
"""
import hashlib
import numpy as np


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def g1_sha(path, lfs_oid):
    got = sha256_file(path)
    return {"ok": got == lfs_oid, "sha256": got}


def read_arrow(path):
    """-> (ts_us int64 array sorted, [jpeg bytes]); IPC file or stream; column discovery by name/type (see the UNVERIFIED note)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    try:
        t = ipc.open_file(path).read_all()
    except pa.ArrowInvalid:
        t = ipc.open_stream(path).read_all()
    tcol = next(n for n in t.column_names if "timestamp" in n.lower())
    bcol = next(n for n, f in zip(t.column_names, t.schema) if pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type))
    ts = np.asarray(t[tcol].to_pylist(), dtype=np.int64)
    data = t[bcol].to_pylist()
    o = np.argsort(ts, kind="stable")
    return ts[o], [data[i] for i in o]


def nearest_rows(ts_rows, ts_query):
    """index of the nearest row and |dt| in ms for each query (the planner.FrameResolver rule, minus its 120 ms refusal)."""
    ts_rows = np.asarray(ts_rows)
    q = np.asarray(ts_query)
    j = np.clip(np.searchsorted(ts_rows, q), 1, len(ts_rows) - 1)
    left, right = ts_rows[j - 1], ts_rows[j]
    pick = np.where(np.abs(q - left) <= np.abs(right - q), j - 1, j)
    return pick, np.abs(ts_rows[pick] - q) / 1000.0


def g2_residual(dt_ms, limit_ms=50.5):
    dt_ms = np.asarray(dt_ms)
    return {"ok": bool(dt_ms.max() <= limit_ms), "n": int(dt_ms.size), "max_ms": float(dt_ms.max()),
            "p95_ms": float(np.percentile(dt_ms, 95)), "limit_ms": limit_ms}


def g2_decode(jpg_bytes, want_wh=(1920, 1080), n_max=None):
    import cv2
    bad = 0
    n = 0
    for b in jpg_bytes[:n_max]:
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        n += 1
        if im is None or (im.shape[1], im.shape[0]) != want_wh:
            bad += 1
    return {"ok": bad == 0, "n": n, "bad": bad}


def g3_bytes_equal(selected, reference):
    """selected / reference: {(camera, key): bytes}. Equal = identical sha256. Returns counts and the first mismatches."""
    keys = sorted(reference)
    eq = 0
    miss = []
    bad = []
    for k in keys:
        if k not in selected:
            miss.append(k)
            continue
        if hashlib.sha256(selected[k]).digest() == hashlib.sha256(reference[k]).digest():
            eq += 1
        else:
            bad.append(k)
    return {"ok": eq == len(keys), "equal": eq, "total": len(keys), "missing": miss[:5], "different": bad[:5]}


def g4_exact_ts(ts_rows, db_image_ts, picks):
    got = np.asarray(ts_rows)[np.asarray(picks)]
    return {"ok": bool(np.array_equal(got, np.asarray(db_image_ts))), "n": int(len(got))}
