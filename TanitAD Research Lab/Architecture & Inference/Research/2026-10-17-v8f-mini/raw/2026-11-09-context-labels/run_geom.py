"""Geometry labels (CPU) for the clips of splits/clips_table.parquet, resumable per 500-clip shard.

usage: run_geom.py [--splits train4512,eval] [--out F:/strat_out/geom] [--shard 500] [--procs 3] [--limit N]
A shard file `geom_<split>_<NNNN>.parquet` that already exists is skipped (idempotent); writes are atomic (tmp + replace).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strat_common as C

TABLE = os.environ.get("STRAT_OUT", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/strat_labels") + "/splits/clips_table.parquet"


def n_video_frames(clip_id):
    p = f"{C.CAM_DIR}/{clip_id}.timestamps.parquet"
    if not os.path.exists(p):
        return None
    return pq.ParquetFile(p).metadata.num_rows


def one_clip(args):
    r = args
    return C.clip_rows(r["clip_id"], {"country": r["country"], "month": r["month"], "hour_of_day": r["hour_of_day"]},
                       r["session_key"], r["chunk"], r["split"], n_video_frames(r["clip_id"]) if r["video_local"] else None)


def shard_path(out, split, i):
    return f"{out}/geom_{split}_{i:04d}.parquet"


def build_shard(out, split, i, recs, pool):
    path = shard_path(out, split, i)
    if os.path.exists(path):
        return path, 0.0, True
    t = time.time()
    rows = [x for part in pool.map(one_clip, recs) for x in part]
    tab = C.rows_to_table(rows)
    tab = tab.replace_schema_metadata({b"label_version": C.LABEL_VERSION.encode(), b"geom_code": C.sha256_file(C.__file__).encode()})
    tmp = path + ".tmp"
    pq.write_table(tab, tmp, compression="zstd")
    os.replace(tmp, path)
    return path, time.time() - t, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train4512,eval")
    ap.add_argument("--out", default="F:/strat_out/geom")
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    tab = pd.read_parquet(TABLE)
    os.makedirs(a.out, exist_ok=True)
    class Serial:
        def map(self, f, xs):
            return map(f, xs)
        def __enter__(self):
            return self
        def __exit__(self, *e):
            return False
    with (ProcessPoolExecutor(a.procs) if a.procs > 1 else Serial()) as pool:
        for split in a.splits.split(","):
            sub = tab[tab.split == split].sort_values("clip_id")
            if a.limit:
                sub = sub.head(a.limit)
            recs = sub.to_dict("records")
            for i in range(0, len(recs), a.shard):
                if os.path.exists("F:/strat_out/STOP"):
                    print("ZZ_GEOM_STOPPED", flush=True)
                    return
                p, dt, skipped = build_shard(a.out, split, i // a.shard, recs[i:i + a.shard], pool)
                print(json.dumps({"shard": os.path.basename(p), "skipped": skipped, "s": round(dt, 1)}), flush=True)
    print("ZZ_GEOM_DONE", flush=True)


if __name__ == "__main__":
    main()
