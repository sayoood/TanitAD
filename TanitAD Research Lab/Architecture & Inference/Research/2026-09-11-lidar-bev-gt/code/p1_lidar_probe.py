#!/usr/bin/env python3
"""P1 - Does the corpus really ship LiDAR, and can we READ it without pulling 32 GB?

The LiDAR feature `lidar_top_360fov` ships as ONE ZIP PER CHUNK of ~100 clips, and
chunk_0000.zip is 32,340,273,976 B (MEASURED, `pai_sizes_and_revs.json`). Downloading a
chunk to read one clip is inadmissible.

A zip's central directory lives at the END of the file, so a ranged HTTP reader plus
`zipfile` streams ONLY the member we want. This script proves that end-to-end:

  chunk lookup -> ranged central-directory read -> single-member extract -> Draco decode

⛔ Every byte counted here is a byte the CONSUMER'S LOADER reads. The consumer is
`zipfile.ZipFile.open(<clip>.lidar_top_360fov.parquet)` over `HfFileSystem`, and the
per-clip cost reported is that member's COMPRESSED size in the central directory, not
the chunk's.

Usage:
  python p1_lidar_probe.py --n-clips 2 --out raw/lidar_probe.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
LIDAR_TMPL = "lidar/lidar_top_360fov/lidar_top_360fov.chunk_{chunk:04d}.zip"
LIDAR_MEMBER_TMPL = "{clip_id}.lidar_top_360fov.parquet"
CLIP_INDEX = r"C:\Users\Admin\tanitad-data\physicalai\clip_index.parquet"
KEYS = r"C:\Users\Admin\Desktop\Keys.txt"


def hf_token() -> str:
    """Read the token IN PLACE. Never printed, never written to an artifact."""
    txt = Path(KEYS).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"hf_[A-Za-z0-9]+", txt)
    if not m:
        raise SystemExit("no hf_ token found in Keys.txt")
    return m.group(0)


def redact(clip_id: str) -> str:
    """Clip ids are gated-confidential (parity_manifest.json). Only digests are quotable."""
    return "clip:" + hashlib.sha256(clip_id.encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=2)
    ap.add_argument("--out", default="raw/lidar_probe.json")
    ap.add_argument("--save-parquet-dir", default=None,
                    help="if set, write each extracted per-clip lidar parquet here")
    ap.add_argument("--clip-ids", default=None,
                    help="comma-separated clip ids; default = first N of the B1 eval join")
    ap.add_argument("--join", default=str(Path(__file__).resolve().parents[4]
                                          / "Benchmarks & Evals" / "Research"
                                          / "2026-09-06-b1-agent-join" / "raw"
                                          / "b1eval_agents.jsonl.xz"))
    args = ap.parse_args()

    import truststore
    truststore.inject_into_ssl()
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out: dict = {"schema": "tanitad.lidar_probe/1", "repo_id": REPO_ID, "clips": []}

    # ---- pick clips -------------------------------------------------------
    if args.clip_ids:
        clip_ids = [c.strip() for c in args.clip_ids.split(",") if c.strip()]
    else:
        import lzma
        clip_ids = []
        seen = set()
        with lzma.open(args.join, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                cid = rec.get("clip_id") or rec.get("clip")
                if cid and cid not in seen:
                    seen.add(cid)
                    clip_ids.append(cid)
                if len(clip_ids) >= args.n_clips:
                    break
        out["clip_source"] = "b1eval_agents.jsonl.xz (first N distinct clip_id)"
    clip_ids = clip_ids[: args.n_clips]

    # ---- chunk lookup -----------------------------------------------------
    idx = pq.read_table(CLIP_INDEX, columns=["clip_id", "chunk", "split"]).to_pydict()
    chunk_of = dict(zip(idx["clip_id"], idx["chunk"]))
    split_of = dict(zip(idx["clip_id"], idx["split"]))
    out["clip_index_rows"] = len(idx["clip_id"])

    fs = HfFileSystem(token=hf_token())

    for cid in clip_ids:
        rec: dict = {"clip": redact(cid)}
        chunk = chunk_of.get(cid)
        rec["chunk"] = chunk
        rec["split"] = split_of.get(cid)
        if chunk is None:
            rec["error"] = "clip not in clip_index.parquet"
            out["clips"].append(rec)
            continue

        path = f"datasets/{REPO_ID}/" + LIDAR_TMPL.format(chunk=chunk)
        rec["zip_path"] = LIDAR_TMPL.format(chunk=chunk)
        member = LIDAR_MEMBER_TMPL.format(clip_id=cid)

        t0 = time.time()
        try:
            info = fs.info(path)
            rec["zip_size_bytes"] = int(info["size"])
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"fs.info: {type(e).__name__}: {e}"
            out["clips"].append(rec)
            continue

        try:
            with fs.open(path, "rb") as fh:
                zf = zipfile.ZipFile(fh)
                names = zf.namelist()
                rec["members_in_zip"] = len(names)
                rec["member_present"] = member in names
                rec["central_dir_read_s"] = round(time.time() - t0, 2)
                if not rec["member_present"]:
                    # absence found at ONE spelling is not absence - probe the other
                    alt = [n for n in names if cid in n]
                    rec["alt_spellings"] = alt[:5]
                    rec["example_members"] = names[:3]
                    out["clips"].append(rec)
                    continue
                zi = zf.getinfo(member)
                rec["member_compress_size_bytes"] = int(zi.compress_size)
                rec["member_file_size_bytes"] = int(zi.file_size)
                rec["member_compress_type"] = int(zi.compress_type)
                t1 = time.time()
                blob = zf.read(member)
                rec["member_read_s"] = round(time.time() - t1, 2)
                rec["member_read_MBps"] = round(len(blob) / 1e6 / max(time.time() - t1, 1e-9), 2)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"zip: {type(e).__name__}: {e}"
            out["clips"].append(rec)
            continue

        rec["member_md5"] = hashlib.md5(blob).hexdigest()
        rec["member_bytes_pulled"] = len(blob)

        if args.save_parquet_dir:
            d = Path(args.save_parquet_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / member).write_bytes(blob)
            rec["saved_to"] = str(d / member)

        # ---- the parquet itself ------------------------------------------
        # ⛔ NEVER `ParquetFile.read()` THIS FILE. It is ~358 MB of Draco blobs in
        # 199 one-row row groups; a full read plus `to_pydict()` held >2 GB and made
        # NO progress for 8 minutes at 5.9 s of CPU - blocked, not computing, which
        # reads exactly like a hang. Row groups are one spin each, so the consumer
        # reads them one at a time and never materialises the clip.
        pf = pq.ParquetFile(io.BytesIO(blob))
        rec["n_rows"] = int(pf.metadata.num_rows)
        rec["n_row_groups"] = int(pf.metadata.num_row_groups)
        rec["columns"] = [f.name for f in pf.schema_arrow]
        rec["arrow_schema"] = str(pf.schema_arrow)
        small = [c for c in rec["columns"] if "draco" not in c and "pointcloud" not in c]
        cols = pq.read_table(io.BytesIO(blob), columns=small).to_pydict()
        cols["__blob0"] = [pf.read_row_group(0, columns=[c for c in rec["columns"]
                                                         if c not in small]).column(0)[0].as_py()]

        # timestamps: find the time-like column and report rate
        for tcol in ("spin_start_timestamp", "timestamp", "timestamp_ns", "ts",
                     "time", "capture_timestamp"):
            if tcol in cols:
                ts = [int(x) for x in cols[tcol]]
                rec["ts_col"] = tcol
                rec["ts_first"] = ts[0]
                rec["ts_last"] = ts[-1]
                d = [b - a for a, b in zip(ts, ts[1:])]
                if d:
                    rec["ts_delta_median"] = sorted(d)[len(d) // 2]
                    rec["ts_delta_min"] = min(d)
                    rec["ts_delta_max"] = max(d)
                    rec["span_raw"] = ts[-1] - ts[0]
                break

        # ---- decode ONE draco blob ---------------------------------------
        blob_col = "__blob0" if isinstance(cols["__blob0"][0], (bytes, bytearray)) else None
        rec["blob_col"] = [c for c in rec["columns"] if c not in small]
        if blob_col is not None:
            import DracoPy
            raw = bytes(cols[blob_col][0])
            rec["blob0_bytes"] = len(raw)
            rec["blob0_magic"] = raw[:8].hex()
            t2 = time.time()
            try:
                pc = DracoPy.decode(raw)
                rec["draco_decode_s"] = round(time.time() - t2, 3)
                import numpy as np
                pts = np.asarray(pc.points)
                rec["n_points"] = int(pts.shape[0])
                rec["point_dims"] = int(pts.shape[1]) if pts.ndim > 1 else 1
                rec["point_dtype"] = str(pts.dtype)
                rec["xyz_min"] = [float(x) for x in pts.min(axis=0)]
                rec["xyz_max"] = [float(x) for x in pts.max(axis=0)]
                rec["xyz_mean"] = [float(x) for x in pts.mean(axis=0)]
                # CONTENT assertion: a decode into a preallocated array can leave zeros
                rec["nonzero_frac"] = float((np.abs(pts).sum(axis=1) > 1e-9).mean())
                for attr in ("colors", "normals", "metadata"):
                    a = getattr(pc, attr, None)
                    if a is not None:
                        try:
                            import numpy as _np
                            arr = _np.asarray(a)
                            rec[f"draco_{attr}_shape"] = list(arr.shape)
                            rec[f"draco_{attr}_dtype"] = str(arr.dtype)
                        except Exception:
                            rec[f"draco_{attr}"] = str(a)[:200]
                rec["draco_attributes"] = [
                    {"unique_id": int(a["unique_id"]),
                     "num_components": int(a["num_components"]),
                     "numpy_dtype": str(np.asarray(a["data"]).dtype),
                     "min": [float(x) for x in np.asarray(a["data"]).min(axis=0)],
                     "max": [float(x) for x in np.asarray(a["data"]).max(axis=0)]}
                    for a in pc.attributes]
                # per-sweep decode timing over a few sweeps, ROW GROUP AT A TIME
                n_time = min(5, rec["n_row_groups"])
                bcol = [c for c in rec["columns"] if c not in small]
                t3 = time.time()
                tot = 0
                for i in range(n_time):
                    b = pf.read_row_group(i, columns=bcol).column(0)[0].as_py()
                    p = DracoPy.decode(bytes(b))
                    tot += int(np.asarray(p.points).shape[0])
                rec["decode_s_per_sweep"] = round((time.time() - t3) / n_time, 4)
                rec["mean_points_per_sweep"] = int(tot / n_time)
            except Exception as e:  # noqa: BLE001
                rec["draco_error"] = f"{type(e).__name__}: {e}"

        rec["total_s"] = round(time.time() - t0, 2)
        out["clips"].append(rec)
        print(json.dumps(rec, indent=1)[:2500], flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[p1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
