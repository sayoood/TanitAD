"""Why does the BEV paint not sit where the front image shows it? Measure the map-space disagreement of the SAME paint
observed from different frames, as a function of the range it was observed at.

Reference: frame j's own paint (classes 2-4) lifted within NEAR_M of the rig -- near lifting is accurate because a
ground-height error dz moves a lifted point by ~dz * range / camera_height. Test: paint of frames k (0 < |k-j| <= 10)
in world coordinates, binned by the range at which frame k observed it; for each test point the distance to the
nearest reference point, kept when < 3 m (same paint, not a different marking). A pose/timing error would grow with
|k - j|; a lifting error grows with observation range.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

NEAR_M = 8.0
BINS = [(0, 8), (8, 14), (14, 20), (20, 26), (26, 34)]


def main(npz_dir):
    files = sorted(Path(npz_dir).glob("[0-9][0-9][0-9].npz"))
    F = [dict(np.load(f, allow_pickle=True)) for f in files]
    paint = []
    for f in F:
        p = np.concatenate([f[f"pts_{k}"] for k in (2, 3, 4) if len(f[f"pts_{k}"])] or [np.zeros((0, 2), np.float32)])
        o = f["T_world_rig"][:2, 3]
        paint.append((p.astype(np.float64), np.hypot(p[:, 0] - o[0], p[:, 1] - o[1]) if len(p) else np.zeros(0)))
    by_bin = {b: [] for b in BINS}; by_dk = {}
    for j in range(len(F)):
        pj, rj = paint[j]
        ref = pj[rj <= NEAR_M]
        if len(ref) < 50:
            continue
        tree = cKDTree(ref)
        for k in range(max(0, j - 10), min(len(F), j + 11)):
            if k == j:
                continue
            pk, rk = paint[k]
            if not len(pk):
                continue
            d, _ = tree.query(pk, distance_upper_bound=3.0)
            ok = np.isfinite(d)
            for b in BINS:
                sel = ok & (rk >= b[0]) & (rk < b[1])
                if sel.any():
                    by_bin[b].append(d[sel])
            sel = ok & (rk < 14)
            if sel.any():
                by_dk.setdefault(abs(k - j), []).append(d[sel])
    print(f"{Path(npz_dir).name}: nearest-reference distance of the same paint seen from another frame (m)")
    print("  by observation range:")
    for b in BINS:
        if by_bin[b]:
            v = np.concatenate(by_bin[b]); print(f"    {b[0]:2d}-{b[1]:2d} m   n={len(v):7d}  median {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}")
    print("  by frame gap |k-j| (observations < 14 m only):")
    for dk in sorted(by_dk):
        v = np.concatenate(by_dk[dk]); print(f"    |dk|={dk:2d}  n={len(v):7d}  median {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
    print("ZZALIGN-DONEZZ")
