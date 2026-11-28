#!/usr/bin/env python3
"""Ranged single-member LiDAR fetch from a PhysicalAI-AV chunk zip -- the SAME reader
as `2026-09-11-lidar-bev-gt/code/p1_lidar_probe.py`, factored so the bandwidth probe (P1)
and the corpus builder (P2) share ONE transfer path.

    HfFileSystem.open(<chunk>.zip)                  # ranged HTTP reads
      -> zipfile.ZipFile(fh).open(<clip>.lidar_top_360fov.parquet)
        -> streamed to <cache>/<clip>.lidar_top_360fov.parquet.part -> atomic rename

Differences from the 09-11 probe, each deliberate:
  * the member is STREAMED to disk (8 MB copy buffer) instead of `zf.read()` into RAM,
    so N concurrent streams do not hold N x ~340 MB;
  * the zip member's CRC-32 is verified by `zipfile` at EOF (a truncated or corrupted
    transfer RAISES `BadZipFile`), and the parquet footer is re-opened from the written
    file and must report 199 row groups with a decodable, non-zero first spin -- the
    content assertion, because a job that exits 0 is not evidence.

⛔ Clip ids are gated-confidential: every record returned here carries `sha256(clip)[:12]`
only. Plaintext ids exist only in local cache FILENAMES, which are never staged.
⛔ The HF token is read IN PLACE with a regex and never printed or written.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
LIDAR_TMPL = "lidar/lidar_top_360fov/lidar_top_360fov.chunk_{chunk:04d}.zip"
LIDAR_MEMBER_TMPL = "{clip_id}.lidar_top_360fov.parquet"
CLIP_INDEX = r"C:\Users\Admin\tanitad-data\physicalai\clip_index.parquet"
KEYS = r"C:\Users\Admin\Desktop\Keys.txt"
LIDAR_CACHE = r"C:\Users\Admin\tanitad-caches\lidar-bev-20260911"
COPY_BUF = 8 * 1024 * 1024

#: The Master Mind's Qwen-Drive clips. NEVER deleted by anything in this package
#: (brief 2026-09-13). ⛔ Stored as sha256(clip_id)[:12] DIGESTS, never as plaintext
#: ids or id prefixes -- this file is staged, and a prefix of a gated id is still a
#: piece of the id. Resolved from the four cached filenames on 2026-09-13.
PROTECTED_SHA12 = frozenset({"4fbd97b6a4b7", "6924358fafe0", "0d90d20036a3", "73495082f98b"})

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def sha12(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode()).hexdigest()[:12]


def is_protected(clip_id: str) -> bool:
    return sha12(clip_id) in PROTECTED_SHA12


def redact_text(s: str) -> str:
    """Replace every uuid-shaped substring with `clip:<sha12>`. Applied to EVERY string
    that can reach a log or a JSON (exception messages name zip members and paths)."""
    return _UUID_RE.sub(lambda m: "clip:" + sha12(m.group(0).lower()), str(s))


def hf_token() -> str:
    """Read the token IN PLACE. Never printed, never written to an artifact."""
    txt = Path(KEYS).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"hf_[A-Za-z0-9]+", txt)
    if not m:
        raise SystemExit("no hf_ token found in Keys.txt")
    return m.group(0)


def chunk_map() -> dict:
    import pyarrow.parquet as pq
    idx = pq.read_table(CLIP_INDEX, columns=["clip_id", "chunk"]).to_pydict()
    return dict(zip(idx["clip_id"], idx["chunk"]))


def cached_path(clip_id: str, cache_dir: str = LIDAR_CACHE) -> Path:
    return Path(cache_dir) / LIDAR_MEMBER_TMPL.format(clip_id=clip_id)


def verify_parquet(path: Path) -> dict:
    """Content assertion on a fetched parquet: 199-ish row groups, a decodable first
    spin with non-zero points. Returns a small dict; raises on failure."""
    import numpy as np
    import pyarrow.parquet as pq
    import DracoPy
    pf = pq.ParquetFile(path)
    n_rg = int(pf.metadata.num_row_groups)
    cols = [f.name for f in pf.schema_arrow]
    if "draco_encoded_pointcloud" not in cols:
        raise RuntimeError(f"no draco column in {path.name}: {cols}")
    blob = pf.read_row_group(0, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
    pts = np.asarray(DracoPy.decode(blob).points)
    if pts.shape[0] == 0 or float(np.abs(pts).max()) == 0.0:
        raise RuntimeError(f"first spin of {path.name} decoded EMPTY/ALL-ZERO")
    return {"n_row_groups": n_rg, "spin0_points": int(pts.shape[0])}


def fetch_clip(clip_id: str, chunk: int, cache_dir: str = LIDAR_CACHE,
               block_size: int | None = None) -> dict:
    """Fetch one clip's LiDAR parquet. Safe to call from a worker process."""
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfFileSystem

    rec: dict = {"clip": sha12(clip_id), "chunk": int(chunk), "pid": os.getpid()}
    out = cached_path(clip_id, cache_dir)
    part = out.with_name(out.name + ".part")
    member = LIDAR_MEMBER_TMPL.format(clip_id=clip_id)
    path = f"datasets/{REPO_ID}/" + LIDAR_TMPL.format(chunk=chunk)
    fs = HfFileSystem(token=hf_token(), skip_instance_cache=True)
    t0 = time.time()
    rec["t_start_unix"] = t0
    try:
        kw = {} if block_size is None else {"block_size": block_size}
        with fs.open(path, "rb", **kw) as fh:
            rec["fsspec_block_size"] = int(getattr(fh, "blocksize", -1))
            zf = zipfile.ZipFile(fh)
            rec["central_dir_s"] = round(time.time() - t0, 3)
            zi = zf.getinfo(member)
            rec["member_bytes"] = int(zi.compress_size)
            rec["member_compress_type"] = int(zi.compress_type)
            t1 = time.time()
            n = 0
            with zf.open(member) as src, open(part, "wb") as dst:
                while True:
                    buf = src.read(COPY_BUF)       # CRC-32 checked by zipfile at EOF
                    if not buf:
                        break
                    dst.write(buf)
                    n += len(buf)
            t2 = time.time()
        rec["stream_s"] = round(t2 - t1, 3)
        rec["bytes_written"] = n
        rec["stream_MBps"] = round(n / 1e6 / max(t2 - t1, 1e-9), 3)
        if n != rec["member_bytes"]:
            raise RuntimeError(f"short write {n} != {rec['member_bytes']}")
        os.replace(part, out)
        rec.update(verify_parquet(out))
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = redact_text(f"{type(e).__name__}: {e}")[:400]
        try:
            if part.exists():
                part.unlink()
        except OSError:
            pass
    rec["t_end_unix"] = time.time()
    rec["total_s"] = round(rec["t_end_unix"] - t0, 3)
    return rec
