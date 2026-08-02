"""Fetch the egomotion zips needed by the r0 selection (pod3 gap-fill)."""
import os
import sys

import pandas as pd

sys.path.insert(0, "/workspace/TanitAD/stack")
try:
    from tanitad.keys import load_keys
    load_keys()
except Exception as e:
    print(f"keys helper: {e}")
from huggingface_hub import hf_hub_download

ROOT = "/workspace/data/physicalai"
sel = pd.read_parquet(f"{ROOT}/r0/r0_selection.parquet")
chunks = sorted(sel["chunk"].astype(int).unique())
print(f"{len(chunks)} egomotion chunks needed")
os.makedirs(f"{ROOT}/labels/egomotion", exist_ok=True)
for c in chunks:
    fn = f"labels/egomotion/egomotion.chunk_{c:04d}.zip"
    p = hf_hub_download("nvidia/PhysicalAI-Autonomous-Vehicles", fn,
                        repo_type="dataset", local_dir=ROOT)
    print(f"ok {fn}", flush=True)
print("EGO_FETCH_DONE")
