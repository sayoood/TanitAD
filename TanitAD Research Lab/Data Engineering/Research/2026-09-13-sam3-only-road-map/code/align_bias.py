"""Is the range-dependent BEV misalignment a SYSTEMATIC per-camera bias (calibration pitch / ground model) or noise?

Per camera, paint (classes 2-4) is re-lifted from the stored 960x540 rasters (exact, as in sam3map_refine.py) so each
point knows its camera and observation range. Reference = paint lifted within 8 m by ANY camera of frame j. For a
test point of camera c at frame k (0 < |k-j| <= 10) matched to its nearest reference within 1.5 m, the SIGNED offset
along the camera's horizontal viewing ray: s > 0 means the far observation lands BEYOND the near one (over-ranged).
A pitch error dtheta predicts s ~ r^2 * dtheta / h; a ground-height error dz predicts s ~ r * dz / h.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/nvidia/sam3paint")
import sam3_paint as P

V2 = Path("/home/nvidia/qwendrive/v2")
VIEWS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]
BINS = [(0, 8), (8, 12), (12, 16), (16, 22), (22, 30)]


GROUND = "G10"


def ground_variant(pts, kind):
    """2 m cells within 45 m. G10 = the extractor's 10th percentile; Gxx = that percentile; BAND = median of the returns
    within 0.15 m above the cell's 10th percentile (drops obstacles without biasing low); PLANE = one robust plane."""
    if kind == "G10":
        return P.ground_grid(pts)
    p = pts[np.hypot(pts[:, 0], pts[:, 1]) < 45]
    key = np.floor((p[:, 0] + 50) / 2).astype(int) * 50 + np.floor((p[:, 1] + 50) / 2).astype(int)
    order = np.argsort(key, kind="stable"); k, z = key[order], p[order, 2]
    grid = np.full(2500, np.nan)
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    for s_, e_ in zip(starts, np.r_[starts[1:], len(k)]):
        if e_ - s_ < 5:
            continue
        zz = z[s_:e_]
        if kind.startswith("G"):
            grid[k[s_]] = np.percentile(zz, int(kind[1:]))
        elif kind == "BAND":
            lo = np.percentile(zz, 10); grid[k[s_]] = np.median(zz[zz <= lo + 0.15])
    if kind == "PLANE":
        g10, fb10 = P.ground_grid(pts)
        cx = (np.arange(2500) // 50) * 2 - 50 + 1.0; cy = (np.arange(2500) % 50) * 2 - 50 + 1.0
        ok = np.isfinite(g10)
        A = np.c_[np.ones(ok.sum()), cx[ok], cy[ok]]; zc = g10[ok]; w = np.ones(ok.sum())
        near = p[np.hypot(p[:, 0], p[:, 1]) < 30]
        coef = np.linalg.lstsq(A * w[:, None], zc * w, rcond=None)[0]
        for _ in range(4):                                            # refit on returns within 0.15 m of the plane
            res = near[:, 2] - (coef[0] + coef[1] * near[:, 0] + coef[2] * near[:, 1])
            g = near[np.abs(res) < 0.15]
            coef = np.linalg.lstsq(np.c_[np.ones(len(g)), g[:, 0], g[:, 1]], g[:, 2], rcond=None)[0]
        grid = coef[0] + coef[1] * cx + coef[2] * cy
    fb = float(np.nanmedian(grid)) if np.isfinite(grid).any() else -0.3
    return grid, fb


def per_camera_paint(c8, f, tok):
    fd = V2 / f"seq_{c8}" / tok
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    grid, fb = ground_variant(np.load(fd / "lidar.npy").astype(np.float64), GROUND)
    T = f["T_world_rig"]; out = {}
    for cam in VIEWS:
        i = fr["cam_order"].index(cam)
        K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
        cls = f[f"cls_{cam}"]
        m = np.isin(np.repeat(np.repeat(cls, 2, axis=0), 2, axis=1), (2, 3, 4))
        if not m.any():
            continue
        xy = P.lift(m, K, R, t, grid, fb, stride=2)
        if not len(xy):
            continue
        w = (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2]
        cw = (T @ np.r_[t, 1.0])[:2]
        out[cam] = (w, cw, float(t[2]))
    return out


def main(c8):
    files = sorted((Path("/home/nvidia/sam3map") / c8).glob("[0-9][0-9][0-9].npz"))
    F = [dict(np.load(f, allow_pickle=True)) for f in files]
    cams = [per_camera_paint(c8, f, str(f["tok"])) for f in F]
    acc = {cam: {b: [] for b in BINS} for cam in VIEWS}
    fitr, fits = {cam: [] for cam in VIEWS}, {cam: [] for cam in VIEWS}
    for j in range(len(F)):
        ref = []
        for cam, (w, cw, h) in cams[j].items():
            r = np.hypot(w[:, 0] - cw[0], w[:, 1] - cw[1]); ref.append(w[r <= 8.0])
        ref = np.concatenate(ref) if ref else np.zeros((0, 2))
        if len(ref) < 50:
            continue
        tree = cKDTree(ref)
        for k in range(max(0, j - 10), min(len(F), j + 11)):
            if k == j:
                continue
            for cam, (w, cw, h) in cams[k].items():
                v = w - cw; r = np.hypot(v[:, 0], v[:, 1])
                sel = r > 8.0
                if not sel.any():
                    continue
                d, idx = tree.query(w[sel], distance_upper_bound=1.5)
                ok = np.isfinite(d)
                if not ok.any():
                    continue
                u = v[sel][ok] / r[sel][ok, None]
                s = np.sum((w[sel][ok] - ref[idx[ok]]) * u, axis=1)
                rr = r[sel][ok]
                for b in BINS:
                    bb = (rr >= b[0]) & (rr < b[1])
                    if bb.any():
                        acc[cam][b].append(s[bb])
                fitr[cam].append(rr); fits[cam].append(s)
    allb = [np.concatenate(acc[cam][b]) for cam in VIEWS for b in BINS[1:3] if acc[cam][b]]
    x = np.concatenate(allb) if allb else np.zeros(1)
    print(f"== {c8} ground={GROUND}: ALL CAMERAS 8-16 m  median signed {np.median(x):+.3f}  median |offset| {np.median(np.abs(x)):.3f}  n {len(x)}")
    for cam in VIEWS:
        line = []
        for b in BINS:
            if acc[cam][b]:
                x = np.concatenate(acc[cam][b]); line.append(f"{b[0]}-{b[1]}m med {np.median(x):+.2f} (n {len(x)})")
        if fitr[cam]:
            r = np.concatenate(fitr[cam]); s = np.concatenate(fits[cam])
            h = next(v[2] for fc in cams for c_, v in fc.items() if c_ == cam)      # camera height over the rig origin (on the road)
            dth = float(np.sum(s * r ** 2) / np.sum(r ** 4) * h)                    # least squares s = r^2 * dth / h
            line.append(f"| pitch-model fit dtheta {np.degrees(dth):+.2f} deg (h {h:.2f} m)")
        print(f"  {cam}: " + "  ".join(line))


if __name__ == "__main__":
    GROUND = sys.argv[1]
    for c in sys.argv[2:]:
        main(c)
    print("ZZBIAS-DONEZZ")
