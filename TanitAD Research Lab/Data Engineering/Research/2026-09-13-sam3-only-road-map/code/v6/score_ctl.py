"""CONTROL for the native-camera scorer patch: on the VIRTUAL views, the camera_model visibility must reproduce the old
pinhole visibility cell for cell (else banked arms would not be comparable); then the same mask on the 7 native cameras."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "stub"))
import mapq_ours as mo  # noqa: E402
V2 = Path("/home/nvidia/qwendrive/v2"); mo.V2 = V2
import sam3map_score_prenative as old  # noqa: E402
import sam3map_score as new  # noqa: E402

NAT = Path("/home/nvidia/sam3map/native7")
VIRT = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]
NATV = ["CAM_FW", "CAM_CL", "CAM_CR", "CAM_RL", "CAM_RR", "CAM_RT", "CAM_FT"]
bad = 0
for c8 in ("73495082f98b", "4fbd97b6a4b7"):
    toks = sorted(p.name for p in (V2 / f"seq_{c8}").iterdir() if p.is_dir())
    for j in (0, 48, len(toks) - 1):
        fd = V2 / f"seq_{c8}" / toks[j]
        grid, fb = old.ground_grid(np.load(fd / "lidar.npy").astype(np.float64))
        old.VIEWS = VIRT; new.VIEWS = VIRT; new.CALIB_ROOT = None
        a = old.camera_coverage(fd, grid, fb); b = new.camera_coverage(fd, grid, fb)
        new.VIEWS = NATV; new.CALIB_ROOT = NAT
        cn = new.camera_coverage(fd, grid, fb)
        bad += int((a != b).sum())
        print(f"{c8} j={j:2d} virtual: old {a.mean():.4f} new {b.mean():.4f} mismatched cells {int((a != b).sum())} | native7 {cn.mean():.4f}"
              f" | native7 but not virtual {float((cn & ~a).mean()):.4f}, virtual but not native7 {float((a & ~cn).mean()):.4f}", flush=True)
print(f"ZZSCORECTL-{bad}ZZ")
