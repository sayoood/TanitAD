"""Consolidate the per-episode T1 causal dumps (141 files each) into one npz per run, so the T1-kin family rows can be re-read from banked small files.
Arrays are the exact concatenation orc_families.py uses (sorted ep*.npz): g, cl, v0, eid, clip_index, ws. No transformation."""
import glob, hashlib, os, numpy as np
ART = "E:/Projects/TanitAD-artifacts/v8f-orecover-t1"
OUT = "C:/Users/Admin/results_orc/bank"
for a in ("ORC", "ORC0"):
    for s in (0, 1):
        fs = sorted(glob.glob(f"{ART}/dump_{a}_s{s}_m10_causal_heads/ep*.npz"))
        parts = {k: [] for k in ("g", "cl", "v0", "eid", "clip_index", "ws")}
        for f in fs:
            with np.load(f) as z:
                for k in parts:
                    parts[k].append(z[k])
        d = {k: np.concatenate(v) for k, v in parts.items()}
        out = f"{OUT}/t1_causal_dump_{a}_s{s}.npz"
        np.savez_compressed(out, **d)
        print(out, len(fs), d["cl"].shape, os.path.getsize(out))
