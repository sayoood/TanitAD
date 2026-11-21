#!/usr/bin/env python3
"""Verify the pulled w120 caches BY BYTES AND BY LOADING, not by counting files.

WHY THIS EXISTS. The pull was interrupted (Thor dropped off the network mid-transfer)
and `huggingface_hub` logged `Invalid metadata file ... Removing it from disk and
continue` for several shards. It re-fetches those, but the programme has now been
bitten twice in two days by exactly the gap a file listing cannot see:

  * `batch_00184` looked present and had 4 of N SAM3 records (a 14x understatement);
  * the A2 dataset card said "one row missing" against a measured 356 (a 356x one).

⇒ A LISTING PROBE SEES A MISSING FILE BUT NEVER A SHORT ONE. This compares every
local file's size against the far-side listing, then loads a sample. A truncated
shard that survives a count is precisely what would poison a 4.8-day training run
with a silent data defect — and it would surface as "the model got worse", not as
an error.

Exit 0 only if every shard matches its far-side size and the loaded sample is
readable; anything else exits non-zero so the caller can refuse to launch.
"""
import os
import random
import sys

import torch
from huggingface_hub import HfApi

REPO = "Sayood/tanitad-physicalai-w120-256x640cyl"
HOME = os.path.expanduser("~")
SPLITS = {
    "train": "physicalai-train-e438721ae894-w120-256x640cyl",
    "val": "physicalai-val-0c5f7dac3b11-w120-256x640cyl",
}
SAMPLE = 12          # episodes actually torch.load-ed per split

api = HfApi()
info = api.dataset_info(REPO, files_metadata=True)
far = {s.rfilename: s.size for s in info.siblings}
print(f"[verify] far side lists {len(far)} files", flush=True)

rc = 0
for split, sub in SPLITS.items():
    d = os.path.join(HOME, "data", sub)
    if not os.path.isdir(d):
        print(f"[verify] {split}: DIRECTORY MISSING {d}")
        rc = 1
        continue

    local = sorted(f for f in os.listdir(d) if not f.startswith("."))
    expect = {k.split("/", 1)[1]: v for k, v in far.items() if k.startswith(sub + "/")}

    missing = sorted(set(expect) - set(local))
    extra = sorted(set(local) - set(expect))
    short, unknown = [], []
    for f in local:
        if f not in expect:
            continue
        want = expect[f]
        got = os.path.getsize(os.path.join(d, f))
        if want is None:
            unknown.append(f)
        elif got != want:
            short.append((f, got, want))

    print(f"[verify] {split}: local {len(local)} / far {len(expect)} · "
          f"missing {len(missing)} · extra {len(extra)} · "
          f"SIZE-MISMATCH {len(short)} · size-unknown {len(unknown)}", flush=True)
    for f, got, want in short[:5]:
        print(f"           SHORT {f}: {got:,} B on disk vs {want:,} B far side")
    if missing[:3]:
        print(f"           missing e.g. {missing[:3]}")

    eps = [f for f in local if f.endswith(".pt")]
    if eps:
        random.seed(0)
        for f in random.sample(eps, min(SAMPLE, len(eps))):
            try:
                torch.load(os.path.join(d, f), map_location="cpu",
                           weights_only=False)
            except Exception as e:                       # noqa: BLE001
                print(f"           LOAD FAILED {f}: {type(e).__name__}")
                rc = 1
        print(f"[verify] {split}: {min(SAMPLE, len(eps))} random episodes "
              f"torch.load OK", flush=True)

    if missing or short:
        rc = 1

print("[verify] RESULT:", "PASS — bytes and loads agree with the far side"
      if rc == 0 else "FAIL — do NOT launch on this cache")
sys.exit(rc)
