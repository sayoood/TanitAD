"""Which camera puts VEHICLES on the sidewalk / verge class? Same-frame evidence: each tracked vehicle box of the frame
(PhysicalAI obstacle.offline via gt.npz, labels == 0) is sampled at 9 footprint points; a point counts for a camera when
that camera's own lifted class-7 pixels cover its 0.15 m cell (and, as the reference, class-1 drivable pixels), split by
the camera's range to the point. Reported per camera and range band: share of vehicle footprint points on sidewalk and on
drivable, and the number of footprint points the camera sees at all (any class)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import sam3_paint as P
import camera_model as CM
import ground_surface as GS

SM = Path("/home/nvidia/sam3map"); V2 = Path("/home/nvidia/qwendrive/v2")
BANDS = ((0, 8), (8, 15), (15, 30))


def cells(xy):
    i = np.floor((xy[:, 0] + 30) / 0.15).astype(int); j = np.floor((xy[:, 1] + 15) / 0.15).astype(int)
    ok = (i >= 0) & (i < 400) & (j >= 0) & (j < 200)
    return i, j, ok


rep = {}
for c8 in sys.argv[1:]:
    acc = {}
    for f in sorted((SM / f"{c8}_v61").glob("[0-9][0-9][0-9].npz"))[::2]:
        z = np.load(f, allow_pickle=True); tok = str(z["tok"])
        g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz")
        vb = g["boxes"][g["labels"] == 0]
        if not len(vb):
            continue
        pts = []
        for b in vb:
            x, y, _, l, w, _, yaw = [float(v) for v in b[:7]]
            c_, s_ = np.cos(yaw), np.sin(yaw)
            for a in (-l / 3, 0, l / 3):
                for bb in (-w / 3, 0, w / 3):
                    pts.append((x + a * c_ - bb * s_, y + a * s_ + bb * c_))
        pts = np.array(pts)
        pi, pj, pok = cells(pts)
        fd = SM / "native7" / f"seq_{c8}" / tok
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
        for cam in [k[4:] for k in z.files if k.startswith("cls_")]:
            C = CM.Camera.from_calib(c, fr["cam_order"].index(cam))
            full = np.repeat(np.repeat(z[f"cls_{cam}"], 2, axis=0), 2, axis=1)
            dist_pts = np.hypot(pts[:, 0] - C.t[0], pts[:, 1] - C.t[1])
            maps = {}
            for name, mask in (("sidewalk", full == 7), ("drivable", np.isin(full, (1, 2, 3, 4, 6))), ("any", full > 0)):
                xy, _ = C.lift(mask, sg, stride=2)
                m = np.zeros((400, 200), bool)
                i, j, ok = cells(xy); m[i[ok], j[ok]] = True
                maps[name] = m
            for a, b in BANDS:
                sel = pok & (dist_pts >= a) & (dist_pts < b)
                k_ = acc.setdefault(f"{cam} {a}-{b}m", {"points": 0, "seen": 0, "sidewalk": 0, "drivable": 0})
                k_["points"] += int(sel.sum())
                k_["seen"] += int(maps["any"][pi[sel], pj[sel]].sum())
                k_["sidewalk"] += int(maps["sidewalk"][pi[sel], pj[sel]].sum())
                k_["drivable"] += int(maps["drivable"][pi[sel], pj[sel]].sum())
    rep[c8] = {k: {**v, "sidewalk_share_of_seen": round(v["sidewalk"] / max(v["seen"], 1), 3), "drivable_share_of_seen": round(v["drivable"] / max(v["seen"], 1), 3)}
               for k, v in sorted(acc.items())}
    print(c8)
    for k, v in rep[c8].items():
        if v["seen"] >= 50:
            print(f"  {k:18s} seen {v['seen']:6d}  on sidewalk {v['sidewalk_share_of_seen']:.3f}  on drivable {v['drivable_share_of_seen']:.3f}", flush=True)
(SM / "sidewalk_by_camera.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print("ZZSIDEWALK-DONEZZ")
