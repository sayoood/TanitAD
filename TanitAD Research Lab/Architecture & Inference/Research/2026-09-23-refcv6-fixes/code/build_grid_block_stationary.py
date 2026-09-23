"""Train-split time base for the clips position registration cannot time.

Same construction the 2-D TRAIN build used for exactly these clips
(recon mode: read_reconstructed_poses -> grid_reference), so the 3-D
builder's C9 (round(t,4) == banked line t_s on EVERY line) can prove it.
"""
import sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "D:/Projects/TanitAD/stack/scripts"); sys.path.insert(0, "D:/Projects/TanitAD/stack")
import build_b1_agent_join as B
S = Path(sys.argv[1]); out = Path(sys.argv[2])
TS = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
EGO = Path("C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo")
ids = [l.strip() for l in (S / "stationary25.txt").read_text().splitlines() if l.strip()]
C, F, T, V = [], [], [], []
from taniteval import lead_source as ls  # noqa -- bootstrapped below if needed
for cid in ids:
    ego_df = B.load_parquet(EGO / (cid + ".parquet"))
    poses, t_grid_s, unit = B.read_reconstructed_poses(TS / (cid + ".timestamps.parquet"), ego_df, 3)
    ref = B.grid_reference(cid, ego_df, t_grid_s)[cid]
    # control: registration really is impossible on this clip (why it needs a block)
    B._bootstrap_taniteval()
    from build_obstacle_join import EgoTrack
    ego = EgoTrack(ego_df)
    try:
        ls.register_poses_to_time(poses[:, :2], ego.t, ego.x, ego.y); reg = "REGISTERS"
    except Exception as e:
        reg = "refused" if "egistration" in repr(e) else "OTHER:" + repr(e)[:60]
    disp = float(np.hypot(*(poses[-1, :2] - poses[0, :2])))
    print("%s n=%d t=[%.3f,%.3f] vmax=%.3f disp=%.2fm registration=%s" % (
        hashlib.sha256(cid.encode()).hexdigest()[:12], len(ref), ref[0][0],
        ref[len(ref) - 1][0], max(v for _, v in ref.values()), disp, reg))
    for i in sorted(ref):
        C.append(cid); F.append(i); T.append(ref[i][0]); V.append(ref[i][1])
np.savez(out, clip_id=np.array(C), frame=np.array(F, dtype=np.int64),
         t0_s=np.array(T, dtype=np.float64), speeds=np.array(V, dtype=np.float64))
print("WROTE", out, "rows", len(C), "clips", len(set(C)))
