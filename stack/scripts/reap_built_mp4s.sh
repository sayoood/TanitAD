#!/usr/bin/env bash
set +e
cd /workspace/TanitAD/stack || exit 1
python3 - <<'PY'
from tanitad.config import base250cam_config
from tanitad.data.physicalai import discover_r0_clips, split_clips
from pathlib import Path
import glob, os
root = "/workspace/data/physicalai_phase0"
cfg = base250cam_config()
clips = discover_r0_clips(root)
tr, va = split_clips(clips, val_frac=0.2, seed=cfg.train.seed)
# GATE 1: split must match the build exactly, else abort (no deletion).
assert len(tr) == 2400 and len(va) == 600, ("split mismatch", len(tr), len(va))
def mp4_of(c):
    m = c.get("mp4") if isinstance(c, dict) else None
    return Path(m) if m else None
# Recompute the DONE set FRESH (built episodes + skip markers) at delete time.
tdir = sorted(glob.glob(f"{root}/_epcache/physicalai-train-*"))[-1]
built = {int(os.path.basename(p)[3:8]) for p in glob.glob(f"{tdir}/ep_*.pt")}
skips = {int(os.path.basename(p)[5:10]) for p in glob.glob(f"{tdir}/skip_*")}
done = built | skips
# Protected sets we must NEVER delete: val mp4s + unbuilt train mp4s.
protect = set()
for c in va:
    m = mp4_of(c);  protect.add(str(m)) if m else None
for i, c in enumerate(tr):
    if i not in done:
        m = mp4_of(c);  protect.add(str(m)) if m else None
# Build the reap list: train mp4s at DONE positions, existing, not protected.
reap = []
for i in sorted(done):
    if i >= len(tr): continue
    m = mp4_of(tr[i])
    if not m or not m.exists(): continue
    if str(m) in protect:            # defensive: never reap a needed mp4
        print("CONFLICT-skip", m); continue
    reap.append(m)
tot = sum(m.stat().st_size for m in reap)
print(f"[reap] done_positions={len(done)} reaping={len(reap)} mp4s = {tot/1e9:.2f} GB "
      f"| protected(val+unbuilt)={len(protect)}")
# GATE 2: sanity bound.
assert len(reap) <= len(tr), "reap set too large"
freed = 0; errs = 0
for m in reap:
    try:
        sz = m.stat().st_size; m.unlink(); freed += sz
    except OSError as e:
        errs += 1; print("err", m, e)
print(f"[reap] DELETED {len(reap)-errs} files, freed {freed/1e9:.2f} GB, errors={errs}")
rem = list(Path(root, "r0", "camera_front_wide").glob("*.mp4"))
print(f"[reap] mp4s remaining: {len(rem)} (expect ~= 388 unbuilt-train + 600 val = ~988, grows-reapable as build advances)")
PY
echo "=== post-reap quota headroom (500MB dd) ==="
t=/workspace/_postreap_$$.bin
dd if=/dev/zero of="$t" bs=1M count=500 conv=fsync 2>/dev/null && echo "500MB write still OK" || echo "500MB write FAILED"
rm -f "$t"
echo "=== r0/camera_front_wide size now ==="; du -sh /workspace/data/physicalai_phase0/r0/camera_front_wide
echo "=== build still advancing + intact? ==="; ls /workspace/data/physicalai_phase0/_epcache/physicalai-train-*/ep_*.pt | wc -l; pgrep -c -f build_pai_cache.py