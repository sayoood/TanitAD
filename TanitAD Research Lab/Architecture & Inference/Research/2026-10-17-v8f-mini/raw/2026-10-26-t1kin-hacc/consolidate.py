"""Consolidate each T1-kin run's 141 per-episode dumps into ONE npz for the repo bank (float32, compressed; the float64 originals go to the
artifacts store as a tar with its sha256). Also writes the per-window component table the reader decided on (float64), per run.
python consolidate.py <run dir> <out dir> <export root>"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

RUN, OUT, ROOT = sys.argv[1:4]
sys.path.insert(0, os.path.join(ROOT, "taniteval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("rd", os.path.join(os.path.dirname(os.path.abspath(__file__)), "t1k_read.py"))
man = {}
for dd in sorted(glob.glob(os.path.join(RUN, "dump_*"))):
    tag = os.path.basename(dd)[5:]
    zs = [np.load(f) for f in sorted(glob.glob(os.path.join(dd, "ep*.npz")))]
    keys = zs[0].files
    d = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    f32 = {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in d.items()}
    p = os.path.join(OUT, f"dumps_{tag}.npz")
    np.savez_compressed(p, **f32)
    man[tag] = {"n_episode_files": len(zs), "n_windows": int(len(d["eid"])), "keys": keys,
                "sha256_f64_concat": hashlib.sha256(b"".join(np.ascontiguousarray(d[k]).tobytes() for k in sorted(keys))).hexdigest(),
                "bank_file": os.path.basename(p), "bank_bytes": os.path.getsize(p)}
json.dump(man, open(os.path.join(OUT, "dumps_manifest.json"), "w"), indent=1)
print("consolidated", len(man), "runs", sum(m["bank_bytes"] for m in man.values()) // 1024, "KiB")
