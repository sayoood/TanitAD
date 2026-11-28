#!/usr/bin/env python3
"""Final CONTENT check of the navtest frame bank (TANITAD venv): KB2/KB3 over the whole bank.

For EVERY token in the export: present in the index, its 4 rows re-read from the shard file,
``sha256(stack.tobytes())[:16]`` recomputed and compared to the index, ``mean_px ≥ 1.0``, shape
and dtype. Tokens never completed (a shard that never arrived, or a stitch failure) are listed
by name. Writes ``<bank>/BANK_DONE.json`` and ``raw/frame_bank_final.json`` (the repo copy).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--inputs", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    idx = json.load(open(os.path.join(a.bank, "index.json"), encoding="utf-8"))
    toks = json.load(gzip.open(a.inputs, "rt", encoding="utf-8"))["tokens"]
    mm_cache, bad, missing, n_ok = {}, [], [], 0
    for t in sorted(toks):
        e = idx["tokens"].get(t)
        if e is None:
            missing.append(t)
            continue
        f = os.path.join(a.bank, e["file"])
        if f not in mm_cache:
            mm_cache.clear()
            mm_cache[f] = np.load(f, mmap_mode="r")
        stack = np.ascontiguousarray(mm_cache[f][e["rows"]])
        sha = hashlib.sha256(stack.tobytes()).hexdigest()[:16]
        ok = (stack.shape == (4, 256, 640, 3) and stack.dtype == np.uint8 and sha == e["sha256"]
              and float(stack.mean()) >= 1.0)
        if ok:
            n_ok += 1
        else:
            bad.append({"token": t, "sha_index": e["sha256"], "sha_recomputed": sha,
                        "mean_px": float(stack.mean())})
    files = sorted(x for x in os.listdir(a.bank) if x.startswith("frames_s") and x.endswith(".npy"))
    rep = {"bank": a.bank, "n_export_tokens": len(toks), "n_indexed": len(idx["tokens"]),
           "n_verified": n_ok, "n_bad": len(bad), "bad": bad[:50], "n_missing": len(missing),
           "missing": missing[:200], "n_shard_files": len(files),
           "bank_bytes": sum(os.path.getsize(os.path.join(a.bank, x)) for x in files),
           "n_rigs": len(idx.get("rigs", {})), "frame": idx.get("frame"),
           "n_carry_jpgs_left": sum(1 for _ in __import__("pathlib").Path(a.bank, "_carry").rglob("*.jpg"))
           if os.path.exists(os.path.join(a.bank, "_carry")) else 0,
           "wall_s": round(time.time() - t0, 1),
           "pass": n_ok == len(toks) and not bad and not missing}
    json.dump(rep, open(os.path.join(a.bank, "BANK_DONE.json"), "w", encoding="utf-8"), indent=1)
    json.dump(rep, open(os.path.join(RAW, "frame_bank_final.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("bad", "missing")}, indent=1))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
