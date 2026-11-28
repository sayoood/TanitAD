"""Per class and observation range: does far-lifted paint land ON the paint seen up close? (margin-corrected)

The signed-offset estimator (align_bias2.py) reads ~0 bias, but it only scores MATCHED points and so cannot see a
mask that spills past the real paint. Here: reference = any paint (classes 2-4) lifted within 10 m of rig j by any
camera; test = class-c paint of frame k (0 < |k-j| <= 10) located within 7 m of rig j (so its true counterpart lies
inside the reference disc) and observed at range r from its camera. Reported per (class, range bin): the share of
test points within 0.5 m / 1.0 m of reference paint. A well-placed mask scores high at every range; a mask that
spills (coarse far segmentation) drops with range.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/nvidia/sam3paint")
import sam3_paint as P

V2 = Path("/home/nvidia/qwendrive/v2")
VIEWS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]
BINS = [(4, 8), (8, 12), (12, 16), (16, 22), (22, 30)]
NAMES = {2: "line", 3: "crosswalk", 4: "arrow/text"}


def lift_classes(c8, f):
    fd = V2 / f"seq_{c8}" / str(f["tok"])
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64))
    T = f["T_world_rig"]; out = []
    for cam in VIEWS:
        i = fr["cam_order"].index(cam)
        K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
        full = np.repeat(np.repeat(f[f"cls_{cam}"], 2, axis=0), 2, axis=1)
        cw = (T @ np.r_[t, 1.0])[:2]
        for k in (2, 3, 4):
            m = full == k
            if not m.any():
                continue
            xy = P.lift(m, K, R, t, grid, fb, stride=2)
            if len(xy):
                w = (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2]
                out.append((k, w, np.hypot(w[:, 0] - cw[0], w[:, 1] - cw[1])))
    return out


def main(npz_dir):
    npz_dir = Path(npz_dir); c8 = npz_dir.name.split("_")[0]
    F = [dict(np.load(p, allow_pickle=True)) for p in sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))]
    L = [lift_classes(c8, f) for f in F]
    acc = {(k, b): [0, 0, 0] for k in NAMES for b in BINS}
    for j in range(len(F)):
        oj = F[j]["T_world_rig"][:2, 3]
        ref = [w[np.hypot(w[:, 0] - oj[0], w[:, 1] - oj[1]) <= 10.0] for k, w, r in L[j]]
        ref = np.concatenate(ref) if ref else np.zeros((0, 2))
        if len(ref) < 50:
            continue
        tree = cKDTree(ref)
        for kk in range(max(0, j - 10), min(len(F), j + 11)):
            if kk == j:
                continue
            for k, w, r in L[kk]:
                sel = np.hypot(w[:, 0] - oj[0], w[:, 1] - oj[1]) <= 7.0
                if not sel.any():
                    continue
                d, _ = tree.query(w[sel], distance_upper_bound=1.0)
                rr = r[sel]
                for b in BINS:
                    bb = (rr >= b[0]) & (rr < b[1])
                    if bb.any():
                        a = acc[(k, b)]; a[0] += int(bb.sum()); a[1] += int((d[bb] <= 0.5).sum()); a[2] += int((d[bb] <= 1.0).sum())
    print(f"== {npz_dir.name}: share of far-lifted paint landing on near paint (<= 0.5 m / <= 1.0 m), margin-corrected")
    for k, name in NAMES.items():
        cells = []
        for b in BINS:
            n, a5, a10 = acc[(k, b)]
            cells.append(f"{b[0]:2d}-{b[1]:2d}m {a5 / n:5.1%}/{a10 / n:5.1%} (n {n})" if n >= 200 else f"{b[0]:2d}-{b[1]:2d}m    -    (n {n})")
        print(f"  {name:10s} " + "  ".join(cells))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
    print("ZZCLASS-DONEZZ")
