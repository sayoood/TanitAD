"""S1 — pull the .v2ep.pt episodes the slot probe needs from HF.

Usage: s1_pull_episodes.py <outdir> <split:val|train> <lo> <hi> [workers]
Pulls episodes [lo, hi) in PROVIDER ORDER (sorted *.v2ep.pt names) — the same
order `build_obstacle_join.corpus_first_clips` and `v2_dataset._list_clips` use,
so the join's "first N clips" and these files are the same clips by construction.

Also pulls `_v2manifest.pt` for the split (the join reads poses from it).
Evidence class: MEASURED (ours) for the byte counts it prints.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import truststore
truststore.inject_into_ssl()
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

KEYS = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt")
TOK = re.search(r"hf_[A-Za-z0-9]+", KEYS.read_text(encoding="utf-8", errors="ignore")).group(0)
REPO = "Sayood/tanitad-physicalai-w120-256x640cyl"
SPLIT_DIR = {"val": "physicalai-val-0c5f7dac3b11-w120-256x640cyl",
             "train": "physicalai-train-e438721ae894-w120-256x640cyl"}

out = Path(sys.argv[1]); split = sys.argv[2]; lo = int(sys.argv[3]); hi = int(sys.argv[4])
NW = int(sys.argv[5]) if len(sys.argv) > 5 else 8
sd = SPLIT_DIR[split]
local = out / "cache"
local.mkdir(parents=True, exist_ok=True)

api = HfApi(token=TOK)
files = [s.rfilename for s in api.repo_info(REPO, repo_type="dataset", token=TOK).siblings]
eps = sorted(f for f in files if f.startswith(sd + "/") and f.endswith(".v2ep.pt"))
want = eps[lo:hi]
man = f"{sd}/_v2manifest.pt"
todo = ([man] if man in files else []) + want
print(f"[s1] {split}: {len(eps)} episodes in repo; pulling [{lo},{hi}) = {len(want)} "
      f"+ manifest; workers={NW}", flush=True)


def pull(rel: str):
    for attempt in range(4):
        try:
            p = hf_hub_download(REPO, rel, repo_type="dataset",
                                local_dir=str(local), token=TOK)
            return rel, os.path.getsize(p), None
        except Exception as ex:  # noqa: BLE001
            if attempt == 3:
                return rel, 0, repr(ex)[:200]
            time.sleep(3 * (attempt + 1))
    return rel, 0, "unreachable"


t0 = time.time(); done = 0; tot = 0; errs = []
with ThreadPoolExecutor(max_workers=NW) as ex:
    futs = [ex.submit(pull, r) for r in todo]
    for f in as_completed(futs):
        rel, n, err = f.result()
        done += 1; tot += n
        if err:
            errs.append({"file": rel, "err": err})
        if done % 5 == 0 or done == len(todo):
            dt = time.time() - t0
            print(f"[s1] {done}/{len(todo)}  {tot/1e9:.2f} GB  "
                  f"{tot/1e6/max(dt,1e-9):.2f} MB/s  {dt/60:.1f} min", flush=True)

rec = {"split": split, "lo": lo, "hi": hi, "n_pulled": done, "total_bytes": tot,
       "wall_s": round(time.time() - t0, 1),
       "MB_per_s": round(tot / 1e6 / max(time.time() - t0, 1e-9), 2),
       "errors": errs, "local_dir": str(local / sd),
       "n_in_repo": len(eps)}
print(json.dumps(rec, indent=1), flush=True)
(out / f"s1_pull_{split}_{lo}_{hi}.json").write_text(json.dumps(rec, indent=1),
                                                     encoding="utf-8")
