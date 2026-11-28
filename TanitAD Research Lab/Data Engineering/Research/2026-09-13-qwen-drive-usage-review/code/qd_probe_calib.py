"""Print the camera rig of every packed frame: the demo (their training rigs) and ours.

Answers, per camera: native image size, focal length AT MODEL RESOLUTION (896x512),
horizontal FOV, principal-point offset, and the camera's yaw / pitch / roll in the
EGO frame plus its mounting position. The demo frames are the reference the weights
were trained on; ours must be brought onto that reference, not the other way round.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

def ypr(R):
    # camera looks along its own +Z; image right is +X, image down is +Y
    fwd, right, down = R[:, 2], R[:, 0], R[:, 1]
    yaw = np.degrees(np.arctan2(fwd[1], fwd[0]))
    pitch = np.degrees(np.arcsin(np.clip(-fwd[2], -1, 1)))    # + = looking down
    # roll: angle of image-right against the horizontal plane
    roll = np.degrees(np.arcsin(np.clip(-right[2], -1, 1)))
    return yaw, pitch, roll

for root in sys.argv[1:]:
    for fd in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        fr = json.loads((fd / "frame.json").read_text())
        c = np.load(fd / "calib.npz")
        l2e = c["lidar2ego"]
        print(f"\n=== {fd.name}  type={fr['dataset_type']}  n_cam={len(fr['cam_order'])}  "
              f"lidar2ego_t={np.round(l2e[:3,3],3).tolist()}  "
              f"lidar2ego_yaw={np.degrees(np.arctan2(l2e[1,0], l2e[0,0])):.1f}")
        tags = [x.get("text") for x in fr["content"] if "text" in x]
        print("   tags:", tags)
        for i, cam in enumerate(fr["cam_order"]):
            K = c["cam_intrinsic"][i]
            R = l2e[:3, :3] @ c["sensor2lidar_rotation"][i]
            t = l2e[:3, :3] @ c["sensor2lidar_translation"][i] + l2e[:3, 3]
            w, h = Image.open(fd / "images" / f"{cam}.jpg").size
            sx, sy = 896 / w, 512 / h
            y, p, r = ypr(R)
            hfov = 2 * np.degrees(np.arctan((w / 2) / K[0, 0]))
            print(f"   {cam:7s} native {w}x{h}  fx@896={K[0,0]*sx:7.1f} fy@512={K[1,1]*sy:7.1f}  "
                  f"cx@896={K[0,2]*sx:6.1f} cy@512={K[1,2]*sy:6.1f}  hfov={hfov:5.1f}  "
                  f"yaw={y:7.1f} pitch={p:5.1f} roll={r:5.1f}  pos={np.round(t,2).tolist()}")
        gt = np.load(fd / "gt.npz")
        print("   gt:", {k: (gt[k].shape, str(gt[k].dtype)) for k in gt.files})
        if (fd / "lidar.npy").exists():
            L = np.load(fd / "lidar.npy"); print("   lidar:", L.shape, L.dtype)
