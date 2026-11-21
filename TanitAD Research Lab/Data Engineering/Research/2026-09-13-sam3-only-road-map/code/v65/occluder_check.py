"""Why the reprojection leaves big unpainted areas near objects (PI question on the day video):
(1) box height convention: for tracked vehicle boxes (gt.npz, v2 ego frame) the native lidar.npy points (rig frame) inside each
    box footprint -- their lowest / 5th percentile / highest z against the box z and z + h, so the rig-frame bottom offset is measured,
    not assumed; plus the smooth LiDAR ground under the box;
(2) the persistent square on the road: the LiDAR 'obstacle' returns that fall into the CAM_FW pixel blocks around the square, in rig
    coordinates, over several frames.
Usage: occluder_check.py <c8> <npz dir> <frames comma> <u,v full-res pixel of the square>"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
ROOT = Path("/home/nvidia/sam3map/native7"); V2 = Path("/home/nvidia/qwendrive/v2")
c8, nd, frames = sys.argv[1], Path(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
su, sv = [float(x) for x in sys.argv[4].split(",")]
offs = []
for n in frames:
    z = np.load(nd / f"{n:03d}.npz", allow_pickle=True); tok = str(z["tok"]); fd = ROOT / f"seq_{c8}" / tok
    lid = np.load(fd / "lidar.npy").astype(np.float64)
    grid, fb = P.ground_grid(lid); sg = GS.smooth_grid(grid, fb)
    g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz"); boxes = g["boxes"][g["labels"] == 0]
    for b in boxes:
        x, y, bz, l, w, h, yaw = [float(v) for v in b[:7]]
        if np.hypot(x, y) > 20:
            continue
        c, s = np.cos(-yaw), np.sin(-yaw); dx, dy = lid[:, 0] - x, lid[:, 1] - y
        inside = (np.abs(dx * c - dy * s) <= l / 2 - 0.2) & (np.abs(dx * s + dy * c) <= w / 2 - 0.2)
        pz = lid[inside, 2]
        gz = float(GS.height(np.array([[x, y]]), sg)[0])
        if len(pz) < 30:
            continue
        above = pz[pz > gz + 0.15]
        if len(above) < 20:
            continue
        offs.append((float(np.percentile(above, 98)) - (bz + h), gz - bz))
        print(f"frame {n} box x {x:5.1f} y {y:5.1f} box z {bz:5.2f} h {h:4.2f} | LiDAR in footprint: n {len(pz)} ground(smooth) {gz:5.2f} "
              f"lowest-above-ground {above.min():5.2f} top p98 {np.percentile(above, 98):5.2f} -> top - (z+h) {np.percentile(above, 98) - (bz + h):+5.2f}, ground - z {gz - bz:+5.2f}")
    c_ = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    C = CM.Camera.from_calib(c_, fr["cam_order"].index("CAM_FW"), 1920, 1080)
    hz = GS.height(lid[:, :2], sg)
    u, v, ok = C.project_rig(lid[:, :3])
    near = ok & (np.abs(u - su) <= 40) & (np.abs(v - sv) <= 40) & (lid[:, 2] > hz + 0.3)
    if near.any():
        print(f"frame {n} obstacle returns within 40 px of the square: {int(near.sum())}, rig xyz:", np.round(lid[near][:6, :3], 2).tolist(), " height above ground:", np.round((lid[near, 2] - hz[near])[:6], 2).tolist())
    else:
        print(f"frame {n} no obstacle return within 40 px of the square")
if offs:
    a = np.array(offs)
    print("median over boxes: LiDAR top - (z + h) =", round(float(np.median(a[:, 0])), 3), " smooth ground - z =", round(float(np.median(a[:, 1])), 3), " n", len(a))
print("ZZOCC-DONEZZ")
