"""Pre-decode the LiDAR evidence of the map scorer on the dev box (DracoPy 2.0.0 lives here, not on Thor).

Calls mapq_ours.Clip.sweep_ego itself -- the exact function the calibrated map checks use -- one sweep at a time, and
saves its float64 (x, y, z, intensity) array keyed by SPIN INDEX, so Thor's scorer gets bit-identical evidence.
Output: D:/sam3map_lidar/<c8>/<spin:04d>.npy
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mapq"))
import mapq_ours as mo  # noqa: E402

c8 = sys.argv[1]
out = Path("D:/sam3map_lidar") / c8; out.mkdir(parents=True, exist_ok=True)
clip = mo.Clip(c8)
toks = sorted(p.name for p in (mo.V2 / f"seq_{c8}").iterdir() if p.is_dir())
done = {}
for t in toks:
    tr = json.loads((mo.V2 / f"seq_{c8}" / t / "meta.json").read_text(encoding="utf-8"))["t_ref_us"]
    si = int(np.abs(clip.mid - tr).argmin())
    if si in done:
        continue
    pts, _ = clip.sweep_ego(tr)
    np.save(out / f"{si:04d}.npy", pts)
    done[si] = int(len(pts))
print(f"{c8}: {len(toks)} frames -> {len(done)} sweeps, {sum(done.values())} points")
print("ZZDECODE-DONEZZ")
