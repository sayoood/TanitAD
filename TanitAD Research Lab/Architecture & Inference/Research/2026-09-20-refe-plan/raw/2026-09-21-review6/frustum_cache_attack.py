"""ATTACK: does the _frustum cache key cover the distortion switch?

Expectation is a LITERAL, not an expression over the code under test:
  * flipping cfg.undistort MUST change the frustum. If it does not, the declared ablation arm
    `undistort=False` is a no-op whenever a forward has already been taken on that instance.
  * an independently authored NumPy Caltech forward model is the reference for the inversion --
    NOT a re-run of the model's own iteration.
"""
import sys, numpy as np, torch
sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
from model import REFe, REFeConfig

cfg = REFeConfig.for_backbone("vits16")
m = REFe(cfg)
gh, gw = cfg.img_h // cfg.patch, cfg.img_w // cfg.patch
print(f"grid {gh}x{gw}, n_cam {cfg.n_cameras}, undistort={cfg.undistort}")

# --- A. fresh instances, so the cache cannot mask anything -------------------
a = REFe(REFeConfig.for_backbone("vits16"))._frustum(gh, gw, "cpu", torch.float32)
c2 = REFeConfig.for_backbone("vits16"); c2.undistort = False
b = REFe(c2)._frustum(gh, gw, "cpu", torch.float32)
print(f"A. fresh instances: max|undist - ideal| = {(a-b).abs().max().item():.6f}   (must be > 0)")

# --- B. THE ATTACK: flip the flag on a LIVE instance after one call ----------
m2 = REFe(REFeConfig.for_backbone("vits16"))
f1 = m2._frustum(gh, gw, "cpu", torch.float32).clone()
m2.cfg.undistort = False
f2 = m2._frustum(gh, gw, "cpu", torch.float32).clone()
d = (f1 - f2).abs().max().item()
print(f"B. SAME instance, cfg.undistort flipped True->False AFTER one call: max|delta| = {d:.6e}")
print(f"   -> the ablation is a {'NO-OP (cache key omits `undistort`)' if d == 0.0 else 'real change'}")
m3 = REFe(REFeConfig.for_backbone("vits16"))
g1 = m3._frustum(gh, gw, "cpu", torch.float32).clone()
m3.cfg.cam_distortion = (0.0, 0.0, 0.0, 0.0, 0.0)
g2 = m3._frustum(gh, gw, "cpu", torch.float32).clone()
print(f"B2. SAME instance, cam_distortion zeroed AFTER one call: max|delta| = {(g1-g2).abs().max().item():.6e}")
m4 = REFe(REFeConfig.for_backbone("vits16"))
h1 = m4._frustum(gh, gw, "cpu", torch.float32).clone()
m4.cfg.cam_t = tuple(tuple(x) for x in np.zeros((4,3)))
m4.cfg.cam_q = ((1.0,0,0,0),)*4
h2 = m4._frustum(gh, gw, "cpu", torch.float32).clone()
print(f"B3. SAME instance, EXTRINSICS collapsed AFTER one call: max|delta| = {(h1-h2).abs().max().item():.6e}")

# --- C. independently authored Caltech round trip ----------------------------
k1,k2,p1,p2,k3 = cfg.cam_distortion
def caltech_forward(xu, yu):
    """ideal -> distorted, written from the OpenCV/Caltech definition, not from model.py"""
    r2 = xu*xu + yu*yu
    rad = 1.0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xd = xu*rad + 2.0*p1*xu*yu + p2*(r2 + 2.0*xu*xu)
    yd = yu*rad + p1*(r2 + 2.0*yu*yu) + 2.0*p2*xu*yu
    return xd, yd
sx, sy = cfg.img_w/cfg.cam_native_w, cfg.img_h/cfg.cam_native_h
fx, fy, cx, cy = cfg.cam_fx*sx, cfg.cam_fy*sy, cfg.cam_cx*sx, cfg.cam_cy*sy
u = (np.arange(gw)+0.5)*(cfg.img_w/gw); v = (np.arange(gh)+0.5)*(cfg.img_h/gh)
vv, uu = np.meshgrid(v, u, indexing="ij")
xd_o = (uu-cx)/fx; yd_o = (vv-cy)/fy
# recover the model's ideal coords from its frustum: cam_pts x = xu*d, with d=near..far
fr = a.reshape(cfg.n_cameras, gh*gw, cfg.pos3d_depth_bins*3)[0].reshape(gh*gw, cfg.pos3d_depth_bins, 3).numpy()*cfg.pos3d_far_m
# undo the extrinsics of camera 0 to get camera-frame points
def q2m(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
R0 = q2m(cfg.cam_q[0]); t0 = np.array(cfg.cam_t[0])
cam0 = (fr - t0) @ R0                 # inverse of p@R.T + t
zz = cam0[:, -1, 2]                   # far bin depth
xu_m = cam0[:, -1, 0]/zz; yu_m = cam0[:, -1, 1]/zz
xd_r, yd_r = caltech_forward(xu_m, yu_m)
err = np.maximum(np.abs(xd_r - xd_o.ravel()), np.abs(yd_r - yd_o.ravel()))
print(f"C. ROUND TRIP through an INDEPENDENTLY AUTHORED forward model:")
print(f"   max |caltech_forward(model_ideal) - observed_distorted| = {err.max():.3e}  (normalised units)")
print(f"   median {np.median(err):.3e};  worst pixel index {int(err.argmax())}")
# D. does the ideal arm reproduce the pinhole EXACTLY?
frb = b.reshape(cfg.n_cameras, gh*gw, cfg.pos3d_depth_bins*3)[0].reshape(gh*gw, cfg.pos3d_depth_bins,3).numpy()*cfg.pos3d_far_m
cam0b = (frb - t0) @ R0
zb = cam0b[:, -1, 2]
xu_b = cam0b[:, -1, 0]/zb; yu_b = cam0b[:, -1, 1]/zb
e2 = np.maximum(np.abs(xu_b - xd_o.ravel()), np.abs(yu_b - yd_o.ravel()))
print(f"D. undistort=False vs the IDEAL pinhole (u-cx)/fx: max |delta| = {e2.max():.3e}  (must be ~0)")
# E. how far does the correction actually move a ray?
dd = np.hypot(xu_m - xd_o.ravel(), yu_m - yd_o.ravel())
print(f"E. correction magnitude in normalised coords: max {dd.max():.4f}, median {np.median(dd):.4f}")
print(f"   worst-case angular shift = {np.degrees(np.arctan(xd_o.ravel()[dd.argmax()]) - np.arctan(xu_m[dd.argmax()])):.3f} deg")
# F. convergence of the fixed-point at 20 iters vs 200
c200 = REFeConfig.for_backbone("vits16"); c200.undistort_iters = 200
f200 = REFe(c200)._frustum(gh, gw, "cpu", torch.float32)
print(f"F. 20 iters vs 200 iters: max|delta| (normalised by far_m) = {(a-f200).abs().max().item():.3e}")
