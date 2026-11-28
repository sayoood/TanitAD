#!/usr/bin/env python3
"""Generic parallel, resumable, CONTENT-VERIFIED fetch of OpenScene/NavSim archives from the public
HF dataset `OpenDriveLab/OpenScene` — the navhard fetcher generalised to any file list.

First use: NAVSIM v1 `navtest` camera shards, PI-authorised in chat 2026-09-19:
*"download NavSIM v1: YES, but only on the D drive"*. 32 shards
`openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_{0..31}.tgz`,
127,882,665,618 B total (HEAD `X-Linked-Size`, MEASURED 2026-09-19). The LiDAR half is NOT fetched:
the stack is camera-only, and the comparable NAVSIM-v1 ladder is perception-free and front-camera.

Same contract as navhard_fetch.py: byte-range parts that resume from their own size; completion is
asserted on CONTENT (every part's exact size, joined size == X-Linked-Size, sha256 == X-Linked-ETag).
`--parts` is kept modest on purpose: this writes to D:, which training runs also read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import navhard_fetch as nf  # noqa: E402  (reuses head_meta / fetch_range / sha256_of)

HF = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/"
SETS = {
    "navtest_camera": [f"openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_{i}.tgz"
                       for i in range(32)],
}


def fetch_one(rel: str, dest: str, n_parts: int, log) -> dict:
    """One archive, reusing navhard_fetch's machinery; returns its receipt row."""
    nf.FILES = {"x": os.path.basename(rel)}          # fetch_file keys off FILES[key]
    nf.BASE = HF + os.path.dirname(rel) + "/"
    receipt: dict = {}
    ok = nf.fetch_file("x", dest, n_parts, receipt, log)
    row = receipt.get("x", {})
    row["ok"] = bool(ok)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--dest", required=True)
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--receipt", required=True)
    a = ap.parse_args()
    nf._ssl()
    os.makedirs(a.dest, exist_ok=True)

    def log(msg: str) -> None:
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    rec = {"authorised": "PI in chat 2026-09-19 ('download NavSIM v1: YES, but only on the D drive')",
           "dest": a.dest, "set": a.set, "files": {}}
    all_ok = True
    for rel in SETS[a.set]:
        row = fetch_one(rel, a.dest, a.parts, log)
        rec["files"][os.path.basename(rel)] = row
        all_ok &= row.get("ok", False)
        with open(a.receipt, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
    rec["all_verified"] = all_ok
    with open(a.receipt, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    log("ALL_VERIFIED" if all_ok else "⛔ NOT ALL VERIFIED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
