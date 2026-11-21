"""Decode-free parity probe: does THIS host reproduce physicalai-train-e438721ae894?

Replicates build_pai_cache.py exactly up to (not including) the decode:
  discover_r0_clips(root) -> split_clips(val_frac=0.2, seed=0) -> cache_key(tr, params)
Standalone (no tanitad import) so it runs on any pod without shipping the stack.
Logic mirrored line-for-line from stack/tanitad/data/{physicalai,epcache}.py @ 4cb37f4.
"""
import hashlib, json, os, socket, sys
from pathlib import Path

PARITY_TRAIN_KEY = "e438721ae894"
PARITY_VAL_KEY   = "0c5f7dac3b11"

def source_id(s):            # epcache._source_id, dict branch
    return f"clip:{s['clip_id']}"

def cache_key(sources, params):   # epcache.cache_key
    ids = [source_id(s) for s in sources]
    return hashlib.sha1(json.dumps({"ids": ids, "params": params},
                                   sort_keys=True, default=str).encode()).hexdigest()[:12]

def discover_r0_clips(root):  # physicalai.discover_r0_clips
    import pandas as pd
    root = Path(root)
    sel = pd.read_parquet(root / "r0" / "r0_selection.parquet")
    chunk_of = dict(zip(sel["clip_id"].astype(str), sel["chunk"].astype(int)))
    out = []
    for mp4 in sorted((root / "r0" / "camera_front_wide").rglob("*.mp4")):
        clip_id = mp4.name.split(".")[0]
        if clip_id not in chunk_of:
            continue
        ts = mp4.with_name(mp4.name.replace(".mp4", ".timestamps.parquet"))
        ego_zip = root / "labels" / "egomotion" / f"egomotion.chunk_{chunk_of[clip_id]:04d}.zip"
        if ts.exists() and ego_zip.exists():
            out.append({"clip_id": clip_id, "mp4": mp4, "timestamps": ts, "ego_zip": ego_zip})
    return out, len(sel), len(list((root / "r0" / "camera_front_wide").rglob("*.mp4")))

def split_clips(clips, val_frac=0.2, seed=0):   # physicalai.split_clips
    import torch
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    val_i = set(perm[:n_val])
    return ([c for i, c in enumerate(clips) if i not in val_i],
            [c for i, c in enumerate(clips) if i in val_i])

def probe(root):
    r = {"root": str(root), "exists": Path(root).exists()}
    if not r["exists"]:
        return r
    try:
        clips, n_sel, n_mp4 = discover_r0_clips(root)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        return r
    r["mp4_on_disk"] = n_mp4
    r["rows_in_r0_selection_parquet"] = n_sel
    r["discovered_clips"] = len(clips)
    # canonical params dict — legacy wheelbase + canonical geometry both contribute {}
    params = {"size": 256, "n_stack": 3, "hz": 10, "calib": "ftheta_v2"}
    tr, va = split_clips(clips, val_frac=0.2, seed=0)
    r["split_train"], r["split_val"] = len(tr), len(va)
    r["train_key"] = cache_key(tr, params)
    r["val_key"] = cache_key(va, params)
    r["train_key_matches_parity"] = r["train_key"] == PARITY_TRAIN_KEY
    r["val_key_matches_parity"] = r["val_key"] == PARITY_VAL_KEY
    # what the 120deg cylindrical sibling key WOULD be
    wide = dict(params)
    wide["geom"] = "256x640f305.5775cyl"
    wide["projection_mode"] = "cylindrical"
    r["wide_train_key_if_geom_256x640f305.5775cyl"] = cache_key(tr, wide)
    return r

if __name__ == "__main__":
    roots = sys.argv[1:] or ["/workspace/data/physicalai_phase0"]
    out = {"host": socket.gethostname(), "probes": [probe(x) for x in roots]}
    print("PARITY_PROBE_JSON " + json.dumps(out))
