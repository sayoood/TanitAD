"""Price the per-image recursive glob that FrameStore.read fell back to on the pod layout.

A tree shaped like the extracted OpenScene pixels (shard/log/CAM/hash.jpg) at 1/10 of navtrain's
413 K frames, on LOCAL NTFS -- a LOWER bound for the pod's network MooseFS, where every directory
read is a network round-trip. Compares the old per-image glob with the new build-once index.
"""
import glob
import os
import shutil
import sys
import time

root = sys.argv[1]
S, L, C, F = 3, 40, 4, 86                     # 3 x 40 x 4 x 86 = 41,280 files (1/10 of 413 K)
if not os.path.isdir(root):
    t = time.time()
    for s in range(S):
        for l in range(L):
            for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")[:C]:
                d = os.path.join(root, f"navtrain_current_{s}", f"log_{s}_{l}", c)
                os.makedirs(d, exist_ok=True)
                for f in range(F):
                    open(os.path.join(d, f"{s:02d}{l:03d}{c[-2:]}{f:05d}.jpg"), "wb").close()
    print(f"built {S*L*C*F:,} files in {time.time()-t:.1f} s")
n = S * L * C * F
target = f"{S-1:02d}{L-1:03d}B0{F-1:05d}.jpg"
reps = 5
t = time.time()
for _ in range(reps):
    hits = glob.glob(os.path.join(root, "**", target), recursive=True)
g = (time.time() - t) / reps
assert hits, "glob found nothing"
t = time.time()
idx = {}
for dp, _dn, fns in os.walk(root):
    for fn in fns:
        idx.setdefault(fn, os.path.join(dp, fn))
w = time.time() - t
t = time.time()
for _ in range(100_000):
    idx.get(target)
lk = (time.time() - t) / 100_000
print(f"tree {n:,} files: ONE recursive glob {g*1000:.0f} ms; index build {w:.2f} s once; "
      f"lookup {lk*1e9:.0f} ns")
full = 413_152
per_sample_glob = 4 * g * full / n
print(f"at navtrain scale ({full:,} frames, linear in entries): {4*g*full/n:.2f} s of glob PER "
      f"SAMPLE (4 cameras) -- LOWER bound, local NTFS")
passes = 4_404_925
print(f"x {passes:,} sample-passes (25 epochs) = {per_sample_glob*passes/3600:,.0f} h of globbing "
      f"alone, vs ~1,540-1,670 A40-h of compute")
print(f"ZZGLOB_{g*1000:.0f}MS_PER_{n}ZZ")
if len(sys.argv) > 2 and sys.argv[2] == "--clean":
    shutil.rmtree(root)
