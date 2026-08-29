"""Content validation of the NavSim eval corpus + two open questions settled.

Q1  WHERE is the unobserved 11 %? (row/col profile of src == -1)
Q2  Are the source jpgs ALREADY undistorted? -- decided by an EXPERIMENT, not
    by reading docs: build the seam both ways and measure discontinuity across
    the l0|f0 and f0|r0 boundaries. The correct model makes the seam smooth;
    the wrong one leaves a step. Control: the same metric measured at
    non-seam columns, which must read the image's own local roughness.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/_s2build/navsim")
import build_navsim_eval as B  # noqa: E402  (reuses the exact build geometry)

OUT = pathlib.Path("C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus")
srcs = sorted(OUT.glob("frames/*.src.npy"))
arrs = sorted(p for p in OUT.glob("frames/*.npy") if not p.name.endswith(".src.npy"))
print(f"scenes: {len(arrs)} | src maps: {len(srcs)}")

# --- Q1: geometry of the hole -------------------------------------------------
s = np.load(srcs[0])
un = s < 0
rows = un.mean(1)
cols = un.mean(0)
nz_rows = np.where(rows > 0)[0]
print(f"\nQ1 unobserved: {un.mean()*100:.2f}% of the frame")
print(f"   rows with ANY hole: {nz_rows.min()}..{nz_rows.max()} "
      f"(of 0..{s.shape[0]-1}); fully-black rows: {(rows == 1).sum()}")
print(f"   top band  rows 0..{np.where(rows < 1)[0].min()-1 if (rows==1).any() else -1}")
print(f"   bottom band rows {np.where(rows < 1)[0].max()+1 if (rows==1).any() else -1}..255")
print(f"   cols with ANY hole: {(cols > 0).sum()}/{s.shape[1]} "
      f"(max col hole frac {cols.max():.3f})")
mid = s[:, s.shape[1] // 2]
print(f"   centre column observed rows: {np.where(mid >= 0)[0].min()}.."
      f"{np.where(mid >= 0)[0].max()}")
print(f"   camera share: " + ", ".join(
    f"{c}={(s==i).mean()*100:.1f}%" for i, c in enumerate(B.CAMS)))
# the vertical arithmetic, stated so the number is checkable
import math  # noqa: E402
need = math.degrees(math.atan(127.5 / B.FRAME.f_ref)) * 2
have = math.degrees(math.atan(540.0 / 1545.0)) * 2
print(f"   frame needs VFOV {need:.1f} deg; NavSim cams have {have:.1f} deg "
      f"-> deficit {need-have:.1f} deg")

# --- Q2: is the distortion model correct? ------------------------------------
sc = B.load(sorted((B.ROOT / "synthetic_scene_pickles").glob("*.pkl"))[0])
fr = sc["frames"][0]
cd = fr["camera_dict"]
paths = [B.BLOBS / B.cam_get(cd, c)["data_path"] for c in B.CAMS]
from PIL import Image  # noqa: E402
ih, iw = np.asarray(Image.open(paths[1])).shape[:2]


def seam_metric(use_dist: bool):
    orig = B.distort
    if not use_dist:
        B.distort = lambda xn, yn, d: (xn, yn)
    try:
        grids, src, _ = B.build_map(cd, (ih, iw))
        img = B.sample(paths, grids, src).astype(np.float64)
    finally:
        B.distort = orig
    g = img.mean(-1)
    valid = src >= 0
    # column-to-column absolute difference, only where BOTH cols are observed
    d = np.abs(np.diff(g, axis=1))
    ok = valid[:, :-1] & valid[:, 1:]
    seam = np.zeros(d.shape[1], bool)
    for i in range(d.shape[1]):                 # a seam column: src changes
        if (src[:, i] != src[:, i + 1])[valid[:, i] & valid[:, i + 1]].any():
            seam[max(0, i - 1):i + 2] = True
    sd = d[:, seam][ok[:, seam]]
    nd = d[:, ~seam][ok[:, ~seam]]
    return float(sd.mean()), float(nd.mean()), int(seam.sum()), img


print("\nQ2 seam continuity (mean |dI/dx|, lower = smoother):")
res = {}
for flag, name in ((True, "WITH distortion (as built)"), (False, "WITHOUT (jpgs pre-rectified)")):
    sm, nm, ns, img = seam_metric(flag)
    res[name] = (sm, nm, img)
    print(f"   {name:34s} seam {sm:6.2f} | non-seam(control) {nm:6.2f} "
          f"| ratio {sm/max(nm,1e-6):5.2f} | seam cols {ns}")
best = min(res, key=lambda k: res[k][0] / max(res[k][1], 1e-6))
print(f"   -> {best} has the smoother seam")

# --- content sanity on the shipped bank --------------------------------------
means, zeros = [], []
for p in arrs[:40]:
    a = np.load(p)
    means.append(float(a.mean()))
    zeros.append(float((a == 0).all(-1).mean()))
print(f"\ncontent (40 scenes): pixel mean {np.mean(means):.1f} "
      f"[{min(means):.1f}..{max(means):.1f}] | all-zero px frac "
      f"{np.mean(zeros):.4f} | any scene < 2.0 mean: {sum(m < 2.0 for m in means)}")
