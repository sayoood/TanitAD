"""B2 — measure REF-C preprocessing damage: canonical f-theta vs naive resize.

Uses the scene's OWN 4K fisheye mp4 (camera_front_wide_120fov.mp4) + the USDZ f-theta
calib (backward poly inverted to TanitAD forward fw_poly, self-verified f_eff~=266).
Runs REF-C base on BOTH preprocessings and reports the plan divergence (GT-free), which
directly answers "how much does the naive closed-loop preprocessing corrupt REF-C's plan".
"""
import ast, glob, io, json, os, subprocess, sys, zipfile
for p in ("/root/taniteval", "/root/TanitAD/stack", "/root/TanitAD/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

from tanitad.data.calib import (F_REF, FThetaIntrinsics,  # noqa: E402
                                ftheta_crop_resize)
from tanitad.data.comma2k19 import stack_frames  # noqa: E402
from refc_v12_cache import load_frozen  # noqa: E402

BASE = ("/workspace/scene_dl/sample_set/26.04_release/"
        "01d503d4-449b-46fc-8d78-9085e70d3554/")
USDZ = BASE + "01d503d4-449b-46fc-8d78-9085e70d3554.usdz"
MP4 = BASE + "camera_front_wide_120fov.mp4"

# ---- 1. USDZ f-theta calib: backward (pixel->angle) ------------------------
import pandas as pd
z = zipfile.ZipFile(USDZ)
df = pd.read_parquet(io.BytesIO(z.read("clipgt/calibration_estimate.parquet")))
_cell = df.iloc[0]["calibration_estimate"]
_outer = ast.literal_eval(_cell) if isinstance(_cell, str) else _cell
_rj = _outer["rig_json"]
rig = json.loads(_rj) if isinstance(_rj, str) else _rj
s = next(x for x in rig["rig"]["sensors"] if "front:wide" in x.get("name", ""))
pr = s["properties"]
bw = [float(x) for x in pr["polynomial"].split()]      # theta = sum bw[i]*r^i
cx, cy, W, H = float(pr["cx"]), float(pr["cy"]), int(pr["width"]), int(pr["height"])
print(f"calib: cx={cx:.1f} cy={cy:.1f} {W}x{H} model={pr['Model']} "
      f"type={pr['polynomial-type']}", flush=True)

# ---- 2. invert backward -> TanitAD forward fw_poly (angle->pixel) ----------
r_max = ((max(cx, W - cx)) ** 2 + (max(cy, H - cy)) ** 2) ** 0.5
r_s = np.linspace(0.0, r_max, 5000)
theta = np.polyval(bw[::-1], r_s)                        # bw(r)
A = np.stack([theta ** k for k in (1, 2, 3, 4)], axis=1)
coef, *_ = np.linalg.lstsq(A, r_s, rcond=None)
fw = (0.0, *coef.tolist())                              # poly[0]=0
inv_err = float(np.abs(np.polyval(list(fw)[::-1], theta) - r_s).max())
intr = FThetaIntrinsics(poly=tuple(fw), cx=cx, cy=cy, width=W, height=H, per_clip=True)
print(f"fw_poly={tuple(round(c,4) for c in fw)} paraxial4K={intr.paraxial_focal:.1f} "
      f"invmax_err_px={inv_err:.3f}", flush=True)

# ---- 3. decode 32 frames @10Hz (native 4K) ---------------------------------
os.makedirs("/workspace/b2frames", exist_ok=True)
for f in glob.glob("/workspace/b2frames/*.jpg"):
    os.remove(f)
subprocess.run(["ffmpeg", "-y", "-i", MP4, "-vf", "fps=10", "-frames:v", "32",
                "-q:v", "2", "/workspace/b2frames/f%04d.jpg"],
               check=True, capture_output=True)
files = sorted(glob.glob("/workspace/b2frames/f*.jpg"))
frames = torch.stack([torch.from_numpy(np.array(Image.open(f).convert("RGB")))
                      .permute(2, 0, 1) for f in files])   # [N,3,H,W] uint8
print("decoded", tuple(frames.shape), flush=True)

# ---- 4. canonical vs naive preprocessing -----------------------------------
canon = ftheta_crop_resize(frames, intr, 256, center="principal")   # [N,3,256,256]
f_eff = float(ftheta_crop_resize.last_f_eff)
print(f"CANON SELF-CHECK: f_eff={f_eff:.1f} (F_REF={F_REF}) "
      f"{'OK' if abs(f_eff - F_REF) < 8 else 'FAIL'}", flush=True)
naive = Fn.interpolate(frames.float(), (256, 256), mode="bilinear",
                       align_corners=False).clamp(0, 255).to(torch.uint8)
canon9 = stack_frames(canon, 3)
naive9 = stack_frames(naive, 3)
print("stacked", tuple(canon9.shape), tuple(naive9.shape), flush=True)

# ---- 5. REF-C base on both -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg, step = load_frozen("/root/models/refc-base-30k/ckpt.pt", "base", None, device)
window = int(cfg.window)


@torch.no_grad()
def run(feats9, v0_mps=10.0):
    T = feats9.shape[0]
    starts = list(range(0, T - window + 1, window))
    if not starts:
        starts = [0]
    fw_ = torch.stack([feats9[t:t + window] for t in starts]).to(device).float().div_(255.0)
    v0 = torch.full((len(starts),), v0_mps, device=device)
    out = model(fw_, nav_cmd=None, v0=v0, steps=2)
    return out["traj"].cpu()          # [nw, 4, 2] ego-frame (x-fwd, y-left)


tc, tn = run(canon9), run(naive9)
d = (tc - tn).norm(dim=-1)            # [nw, 4] per-waypoint L2
print("\n=== RESULT ===", flush=True)
print("canon traj[0] (0.5/1/1.5/2s):", [[round(v, 2) for v in p] for p in tc[0].tolist()])
print("naive traj[0]:", [[round(v, 2) for v in p] for p in tn[0].tolist()])
print(f"n_windows={tc.shape[0]}")
print(f"PLAN DIVERGENCE canon-vs-naive: mean_per_wp={float(d.mean()):.3f} m  "
      f"@2s={float(d[:, -1].mean()):.3f} m  max={float(d.max()):.3f} m")
print(f"endpoint fwd(x) mean: canon={float(tc[:, -1, 0].mean()):.2f} m  "
      f"naive={float(tn[:, -1, 0].mean()):.2f} m")
print(f"(reference scale: REF-C base val ade@2s=0.4728, CV baseline 0.838)")
print("B2_DONE", flush=True)
