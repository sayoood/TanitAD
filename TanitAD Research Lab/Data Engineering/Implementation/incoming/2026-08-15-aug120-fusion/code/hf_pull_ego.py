"""Pull the 201 ego npz for the aug120 union (labels-scale payload, ~1 MB).

Verifies each npz loads with poses [T,4] before accepting it.
"""
import json
import os
import re
import shutil

import numpy as np
import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"
TOKEN = re.search(r"hf_[A-Za-z0-9]+",
                  open(KEYS, encoding="utf-8", errors="replace").read()).group(0)
from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
cov = json.load(open(os.path.join(ROOT, "coverage.json")))
todo = cov["todo"]
edir = os.path.join(ROOT, "ego")
os.makedirs(edir, exist_ok=True)

bad, total = [], 0
for cid in todo:
    dst = os.path.join(edir, f"{cid}.npz")
    if not os.path.exists(dst):
        p = hf_hub_download("Sayood/tanitad-ph0-aug120",
                            f"bridged_w120train_2400/ego/{cid}.npz",
                            repo_type="dataset", token=TOKEN)
        shutil.copyfile(p, dst)
    try:
        d = np.load(dst)
        po = d["poses"]
        assert po.ndim == 2 and po.shape[1] == 4 and po.shape[0] > 0
    except Exception as e:  # noqa: BLE001
        bad.append((cid, f"{type(e).__name__}: {e}"))
    total += os.path.getsize(dst)

print(f"ego npz pulled+verified: {len(todo) - len(bad)}/{len(todo)}  "
      f"bytes={total}")
if bad:
    print("BAD:", bad[:5])

# parquet schema check
import pandas as pd  # noqa: E402
df = pd.read_parquet(os.path.join(ROOT, "aux", "records.parquet"))
print("records.parquet columns:", list(df.columns))
print("tasks:", sorted(df["task"].unique()))
sub = df[df["clip_id"].isin(set(todo))]
print(f"rows for the 201 todo clips: {len(sub)}  distinct clips: "
      f"{sub['clip_id'].nunique()}")
print("EGO_DONE")
