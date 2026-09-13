"""Thor-side runner for the dev-box scripts, with Thor's data paths.

Why Thor: on 2026-09-13 the dev box sat at 97 % committed memory (0.6 GB commit headroom) under a training job, and
nothing may add load to a machine that is training.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V2 = Path("/home/nvidia/qwendrive/v2")

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "render":
        import sam3map_render as r
        r.V2 = V2
        sys.argv = ["sam3map_render.py"] + sys.argv[2:]
        r.main()
    elif what == "render3":
        import os
        import sam3map_render_v3 as r3
        r3.V2 = Path(os.environ.get("SAM3MAP_ROOT", str(V2)))
        sys.argv = ["sam3map_render_v3.py"] + sys.argv[2:]
        r3.main()
    elif what == "render4":
        import os
        import sam3map_render_v4 as r4
        r4.V2 = Path(os.environ.get("SAM3MAP_ROOT", str(V2)))
        sys.argv = ["sam3map_render_v4.py"] + sys.argv[2:]
        r4.main()
    elif what == "score":
        D = Path("/home/nvidia/sam3map/data")
        C8 = sys.argv[2]
        sys.path.insert(0, str(HERE / "stub"))      # DracoPy is not on Thor: sweeps were decoded on the dev box
        import numpy as np
        import mapq_ours as mo                       # imported first, so the scorer's own import reuses these paths
        mo.V2, mo.PRED, mo.LIDAR = V2, D / "preds", D
        mo.EGO_ZIP, mo.EXT = D / "egomotion.chunk_0768.zip", D / "sensor_extrinsics.chunk_0768.parquet"

        def sweep_cached(self, t):
            """mapq_ours.Clip.sweep_ego's own output, saved on the dev box by decode_sweeps.py, keyed by spin index."""
            si = int(np.abs(self.mid - t).argmin())
            return np.load(D / "sweeps" / C8 / f"{si:04d}.npy"), float((self.mid[si] - t) / 1e3)
        mo.Clip.sweep_ego = sweep_cached
        import sam3map_score as s
        sys.argv = ["sam3map_score.py"] + sys.argv[2:]
        s.main()
    else:
        raise SystemExit(f"unknown job {what}")
