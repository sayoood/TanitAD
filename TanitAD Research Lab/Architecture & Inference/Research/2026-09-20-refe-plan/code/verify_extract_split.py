"""Verify a downloaded nuPlan split zip by BYTE COUNT against the S3 Content-Length, then extract
it on D:. CRC is verified by zipfile during extraction (ZipExtFile checks every member's CRC on
read and raises BadZipFile), so a separate testzip pass -- a second full read of ~96 GB from an
external drive -- is deliberately skipped.

Usage: python verify_extract_split.py <split> <zip_path> <expected_bytes>
Markers: SPLIT_DONE <split> / SPLIT_FAIL <split>
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import zipfile

split, zpath, expect = sys.argv[1], sys.argv[2], int(sys.argv[3])
DATA = "D:/Projects/TanitAD/data/nuplan"


def log(m):
    print(time.strftime("%H:%M:%S"), f"[{split}]", m, flush=True)


size = os.path.getsize(zpath) if os.path.exists(zpath) else -1
if size != expect:
    log(f"SIZE MISMATCH {size} != {expect} -> not extracting")
    print(f"SPLIT_FAIL {split}", flush=True)
    sys.exit(1)
log(f"size OK ({size/1e9:.2f} GB); extracting to {DATA} with per-member CRC ...")
t0 = time.time()
try:
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        log(f"{len(names)} members; first: {names[:2]}")
        z.extractall(DATA)
except zipfile.BadZipFile as e:
    log(f"CRC/zip FAILURE during extraction: {e}")
    print(f"SPLIT_FAIL {split}", flush=True)
    sys.exit(1)
db_dirs = sorted({os.path.dirname(os.path.join(DATA, n)) for n in names if n.endswith(".db")})
n_split = sum(1 for n in names if n.endswith(".db"))
n_all = len(glob.glob(DATA + "/**/*.db", recursive=True))
log(f"extracted in {(time.time()-t0)/60:.1f} min; {n_split} .db in this split; dirs {db_dirs}; total .db on D: {n_all}")
json.dump({"split": split, "zip": zpath, "bytes": size, "db_dirs": db_dirs, "n_db_split": n_split,
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(f"{DATA}/{split}_extract.json", "w"), indent=1)
print(f"SPLIT_DONE {split}", flush=True)
