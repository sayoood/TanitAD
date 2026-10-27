"""fetch_fresh2_obstacle.py -- COPY of refcv9_lonhead/code/fetch_fresh_obstacle.py with ONLY IDS and OUT re-pointed at FRESH2 (PREREG_FRESH2 s1.2).
Original docstring:
fetch_fresh_obstacle.py -- the obstacle.offline parquet of each of the 300 FRESH clips -> F:/refcv9_lonhead/obstacle_fresh/<clip>.parquet
(the `--obstacle-dir` layout build_b1_agent_join.py reads, :742).  NOT RUN by the LONHEAD agent: it DOWNLOADS (~0.64 MB/clip MEASURED as the
mean STORED member size of a local chunk zip -> ~190 MB for 295 clips, by HTTP range reads of 281 chunk zips of
nvidia/PhysicalAI-Autonomous-Vehicles labels/obstacle.offline), which needs the PI's / user's explicit go.

  --local-only   extract ONLY the clips whose chunk zip is already on disk (C:/Users/Admin/tanitad-data/physicalai/labels/obstacle.offline,
                 5 of 300 MEASURED) -- no network.
  (default)      local first, then HF range reads for the rest (l1c_fetch.py's member_bytes pattern; STORED members, CRC-checked by zipfile).

No clip id is printed: progress and errors are counted; per-clip records go to the manifest as sha12 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
LOCAL = Path("C:/Users/Admin/tanitad-data/physicalai/labels/obstacle.offline")
IDS = Path("F:/l1c/fresh2/fresh2_ids.json")              # FRESH2 COPY: {clip_id, chunk} x 600 (PREREG_FRESH2 pool, ranks 421-1020; read-only)
OUT = Path("F:/l1c/fresh2/obstacle_fresh2")                # FRESH2 COPY: new dir under F:/l1c
SUFFIX = ".obstacle.offline.parquet"


def sha12(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def put(cid: str, b: bytes) -> None:
    tmp = OUT / (cid + ".parquet.tmp")
    tmp.write_bytes(b)
    os.replace(tmp, OUT / (cid + ".parquet"))


def from_local(cid: str, ch: int) -> bool:
    z = LOCAL / f"obstacle.offline.chunk_{ch:04d}.zip"
    if not z.is_file():
        return False
    with zipfile.ZipFile(z) as zf:
        nm = [i for i in zf.infolist() if i.filename.endswith(cid + SUFFIX)]
        if len(nm) != 1:
            return False
        put(cid, zf.read(nm[0]))
    return True


def from_hf(cid: str, ch: int, retries: int = 4) -> None:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    path = f"datasets/{REPO}/labels/obstacle.offline/obstacle.offline.chunk_{ch:04d}.zip"
    last = None
    for k in range(retries):
        try:
            with fs.open(path, "rb", block_size=8 * 1024 * 1024) as f, zipfile.ZipFile(f) as zf:
                nm = [i for i in zf.infolist() if i.filename.endswith(cid + SUFFIX)]
                if len(nm) != 1:
                    raise RuntimeError(f"{len(nm)} members match")
                if nm[0].compress_type != 0:
                    raise RuntimeError("member is not STORED")
                put(cid, zf.read(nm[0]))
                return
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:160]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = json.load(open(IDS, encoding="utf-8"))
    rec = {"n": len(ids), "local": 0, "hf": 0, "have": 0, "errors": 0, "per_clip": {}}
    todo = []
    for r in ids:
        cid, ch = r["clip_id"], int(r["chunk"])
        if (OUT / (cid + ".parquet")).is_file():
            rec["have"] += 1
            continue
        if from_local(cid, ch):
            rec["local"] += 1
            rec["per_clip"][sha12(cid)] = "local"
        else:
            todo.append((cid, ch))
    print(f"[fetch] have {rec['have']} local {rec['local']} remaining {len(todo)}", flush=True)
    if todo and not a.local_only:
        sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l1c/code")
        import hfenv                                             # token read IN PLACE (never printed); truststore TLS
        hfenv.init()
        with ThreadPoolExecutor(a.threads) as ex:
            fut = {ex.submit(from_hf, c, ch): c for c, ch in todo}
            for i, f in enumerate(as_completed(fut)):
                c = fut[f]
                try:
                    f.result()
                    rec["hf"] += 1
                    rec["per_clip"][sha12(c)] = "hf"
                except Exception as e:      # noqa: BLE001
                    rec["errors"] += 1
                    rec["per_clip"][sha12(c)] = f"error {type(e).__name__}"
                if (i + 1) % 25 == 0:
                    print(f"[fetch] {i + 1}/{len(todo)} hf {rec['hf']} errors {rec['errors']}", flush=True)
    rec["n_present"] = sum(1 for r in ids if (OUT / (r["clip_id"] + ".parquet")).is_file())
    (OUT / "fetch_manifest.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"ZZFETCHZZ present {rec['n_present']}/{rec['n']} local {rec['local']} hf {rec['hf']} errors {rec['errors']}", flush=True)
    return 0 if rec["n_present"] == rec["n"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
