"""Download egomotion zips for EXACTLY the parity-selection chunks (phase0).

Parity-safe gap-fill: does NOT run physicalai_r0.py `select` (which resamples
and overwrites r0_selection.parquet). Reads the already-relayed parity
selection, takes its distinct chunk list, and downloads the immutable per-chunk
egomotion zips to <root>/labels/egomotion/ -- the exact path
discover_r0_clips() requires. Same HF files pod2 used -> build stays identical.
Auth via env HF_TOKEN (pod convention).
"""
import os
import sys
import time

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
TMPL = "labels/egomotion/egomotion.chunk_{:04d}.zip"
ROOT = "/workspace/data/physicalai_phase0"
TOK = os.environ.get("HF_TOKEN")


def path_for(c):
    return os.path.join(ROOT, TMPL.format(c))


def missing(chunks):
    out = []
    for c in chunks:
        p = path_for(c)
        if not (os.path.exists(p) and os.path.getsize(p) > 1_000_000):
            out.append(c)
    return out


def main():
    sel = pd.read_parquet(f"{ROOT}/r0/r0_selection.parquet")
    chunks = sorted(sel["chunk"].astype(int).unique().tolist())
    print(f"[ego] ensuring {len(chunks)} egomotion chunks present", flush=True)
    for attempt in range(1, 9):
        m = missing(chunks)
        if not m:
            break
        print(f"[ego] attempt {attempt}: {len(m)} missing", flush=True)
        for c in m:
            try:
                hf_hub_download(REPO, TMPL.format(c), repo_type="dataset",
                                local_dir=ROOT, token=TOK)
            except Exception as e:
                print(f"[ego] chunk {c} failed: {type(e).__name__}: {e}",
                      flush=True)
        time.sleep(3)
    m = missing(chunks)
    if m:
        print(f"[ego] STILL MISSING {len(m)}: {m[:25]}", flush=True)
        sys.exit(1)
    print(f"EGO_ALL_PRESENT n={len(chunks)}", flush=True)


if __name__ == "__main__":
    main()
