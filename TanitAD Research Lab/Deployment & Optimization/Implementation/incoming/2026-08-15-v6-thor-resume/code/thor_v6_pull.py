#!/usr/bin/env python3
"""Pull everything the v6F S-W resume needs onto Thor, then verify by LOADING.

Layout mirrors the pod convention under $HOME (Thor has no /workspace):
  ~/data/physicalai-train-e438721ae894-w120-256x640cyl/   (85 GB, 2403 files)
  ~/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl/     (21.2 GB, 603 files)
  ~/experiments/v6F-SW-30k/{ckpt.pt, config.json}         (3.5 GB)

Verification is by size AND by torch.load of the checkpoint plus one episode —
size and exit codes are not evidence (a 21%-truncated episode survived every
prior size check on this very box).
"""
import json
import os
import time

from huggingface_hub import hf_hub_download, snapshot_download

HOME = os.path.expanduser("~")
DATA = os.path.join(HOME, "data")
OUT = os.path.join(HOME, "experiments", "v6F-SW-30k")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
print(f"[pull] start {time.strftime('%H:%M:%S')}", flush=True)

# 1. The small, blocking pieces first: checkpoint + config.
for f in ("config.json", "ckpt.pt", "metrics.json", "train_log.jsonl"):
    p = hf_hub_download("Sayood/tanitad-v6", f"v6F-SW-30k/{f}",
                        local_dir=os.path.join(HOME, "_v6pull"))
    dst = os.path.join(OUT, f)
    if os.path.abspath(p) != os.path.abspath(dst):
        os.replace(p, dst)
    print(f"[pull] {f}: {os.path.getsize(dst):,} B", flush=True)

# 2. The two caches (the long pole).
for pat, name in ((
    "physicalai-train-e438721ae894-w120-256x640cyl/*", "train"), (
    "physicalai-val-0c5f7dac3b11-w120-256x640cyl/*", "val")):
    print(f"[pull] cache {name} starting at +{time.time()-t0:.0f}s", flush=True)
    snapshot_download("Sayood/tanitad-physicalai-w120-256x640cyl",
                      repo_type="dataset", allow_patterns=[pat],
                      local_dir=DATA, max_workers=4)
    print(f"[pull] cache {name} done at +{time.time()-t0:.0f}s", flush=True)

# 3. Verify by loading, never by exit code.
import torch  # noqa: E402  (after downloads; import cost irrelevant here)
ck = torch.load(os.path.join(OUT, "ckpt.pt"), map_location="cpu",
                weights_only=False)
n_tensors = len(ck.get("stack", ck))
print(f"[verify] ckpt.pt loads: step={ck.get('step')} tensors={n_tensors}",
      flush=True)

for split, key in (("physicalai-train-e438721ae894-w120-256x640cyl", "train"),
                   ("physicalai-val-0c5f7dac3b11-w120-256x640cyl", "val")):
    d = os.path.join(DATA, split)
    files = sorted(os.listdir(d))
    eps = [f for f in files if f.endswith((".pt", ".v2ep.pt"))]
    probe = os.path.join(d, eps[len(eps) // 2])
    torch.load(probe, map_location="cpu", weights_only=False)
    print(f"[verify] {key}: {len(files)} files, mid-episode loads "
          f"({os.path.basename(probe)})", flush=True)

with open(os.path.join(OUT, "config.json")) as fh:
    cfg = json.load(fh)
print(f"[verify] banked config stage={cfg.get('stage')} run={cfg.get('run')}",
      flush=True)
print(f"[pull] ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
