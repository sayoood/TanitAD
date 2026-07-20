#!/usr/bin/env bash
set +e
cd /workspace/TanitAD/stack || exit 1
python3 - <<'PY'
import pandas as pd, torch, glob, os, hashlib
from pathlib import Path
root = "/workspace/data/physicalai_phase0"
# Canonical original order = discover_r0_clips order = sorted(rglob mp4 names) = sorted(clip_id)
# (flat dir, constant ".camera_front_wide_120fov.mp4" suffix). Rebuilt from the
# still-present r0_selection.parquet (reap deleted only *.mp4).
sel = pd.read_parquet(f"{root}/r0/r0_selection.parquet")
clip_ids = sorted(sel["clip_id"].astype(str).tolist())
assert len(clip_ids) == 3000, len(clip_ids)
# Reproduce split_clips EXACTLY: randperm(3000, seed 0), first 600 -> val.
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(clip_ids), generator=g).tolist()
n_val = max(1, int(len(clip_ids) * 0.2))
val_i = set(perm[:n_val])
train = [c for i, c in enumerate(clip_ids) if i not in val_i]
val   = [c for i, c in enumerate(clip_ids) if i in val_i]
assert len(train) == 2400 and len(val) == 600, (len(train), len(val))
# VALIDATION: reap protected val mp4s -> reconstructed val clip_ids should still have mp4s.
mp4dir = Path(root, "r0", "camera_front_wide")
val_present = sum((mp4dir/f"{c}.camera_front_wide_120fov.mp4").exists() for c in val)
print(f"[validate] reconstructed val clip_ids with mp4 still present: {val_present}/600 "
      f"(reap protected val -> should be ~600 => split reconstruction correct)")
# Skip positions -> canonical train clip_ids.
tdir = sorted(glob.glob(f"{root}/_epcache/physicalai-train-*"))[-1]
skips = sorted(int(os.path.basename(p)[5:10]) for p in glob.glob(f"{tdir}/skip_*"))
print(f"[skipset] cache={os.path.basename(tdir)} skips={len(skips)} usable_train={2400-len(skips)}")
skip_clipids = [train[i] for i in skips]
for i, c in zip(skips, skip_clipids):
    print(f"  train_pos {i:5d} -> clip_id {c}")
ids = ",".join(sorted(skip_clipids))
print("PARITY sorted skip clip_ids:", sorted(skip_clipids))
print("PARITY_HASH sha256(sorted skip clip_ids) =", hashlib.sha256(ids.encode()).hexdigest())
PY