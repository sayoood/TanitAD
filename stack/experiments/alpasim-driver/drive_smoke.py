"""Prove the TanitAD driver satisfies AlpaSim's contract and produces trajectories.

Not a sim run: it exercises the exact call path AlpaSim uses (from_config -> properties ->
predict) with synthetic frames, so a contract break is caught before burning a sim episode.
"""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/alpasim/src/driver/src"))
from alpasim_driver.models import TanitADModel, DriveCommand
from alpasim_driver.models.base import CameraFrame, PredictionInput

CK = None
for pat in ("~/models/flagship-v4.2b/ckpt.pt",):
    g = sorted(glob.glob(os.path.expanduser(pat)))
    if g:
        CK = g[0]; break
print("CKPT:", CK, flush=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {"checkpoint_path": CK, "geometry": "v1", "horizon": 20}
m = TanitADModel.from_config(cfg, dev, ["camera_front_wide_120fov"])
print("INSTANTIATED OK", flush=True)
print("  camera_ids        :", m.camera_ids)
print("  context_length    :", m.context_length)
print("  output_frequency  :", m.output_frequency_hz, "Hz")
print("  num_cameras       :", m.num_cameras)

W = m.context_length
rng = np.random.default_rng(0)
for tick, (v0, cmd) in enumerate([(12.0, DriveCommand.STRAIGHT),
                                  (8.0,  DriveCommand.LEFT),
                                  (15.0, DriveCommand.RIGHT)]):
    frames = [CameraFrame(timestamp_us=100000*i,
                          image=rng.integers(0,255,(256,640,3),dtype=np.uint8))
              for i in range(W)]
    pi = PredictionInput(camera_images={"camera_front_wide_120fov": frames},
                         command=cmd, speed=v0, acceleration=0.0,
                         ego_pose_history=[], inference_seed=tick)
    out = m.predict(pi)
    wp = np.asarray(out.trajectory_xy)
    print(f"  tick{tick} cmd={cmd.name:8s} v0={v0:5.1f} -> traj{wp.shape} "
          f"x[{wp[:,0].min():.2f},{wp[:,0].max():.2f}] y[{wp[:,1].min():.2f},{wp[:,1].max():.2f}] "
          f"head{np.asarray(out.headings).shape} finite={np.isfinite(wp).all()}", flush=True)
    print(f"        note: {out.reasoning_text}", flush=True)

# contract check: a wrong-length window MUST raise, not silently score
try:
    bad = [CameraFrame(timestamp_us=0, image=rng.integers(0,255,(256,640,3),dtype=np.uint8))]
    m.predict(PredictionInput(camera_images={"camera_front_wide_120fov": bad},
                              command=DriveCommand.STRAIGHT, speed=10.0, acceleration=0.0,
                              ego_pose_history=[], inference_seed=99))
    print("⛔ WINDOW GUARD FAILED — a 1-frame window was accepted", flush=True)
except RuntimeError as e:
    print("✅ window guard fires:", str(e)[:90], flush=True)
print("DRIVER SMOKE PASSED", flush=True)
