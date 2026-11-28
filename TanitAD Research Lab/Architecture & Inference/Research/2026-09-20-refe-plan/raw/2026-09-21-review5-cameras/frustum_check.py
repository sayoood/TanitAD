"""INDEPENDENT reconstruction of one patch ray, compared against REFe._frustum.
Nothing is imported from model.py except the module under test."""
import sys, math, json
import numpy as np, torch
sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
from model import REFe, REFeConfig

cfg = REFeConfig()
m = REFe.__new__(REFe)          # avoid building the 300M trunk; _frustum only needs cfg + cache
m.cfg = cfg; m._pos3d_cache = {}
gh, gw = cfg.img_h // cfg.patch, cfg.img_w // cfg.patch
P = gh * gw
print(f"grid {gh} x {gw} = {P} patches per camera; n_cameras={cfg.n_cameras}; tokens={P*cfg.n_cameras}")
fr = m._frustum(gh, gw, "cpu", torch.float32, cfg.n_cameras)[0].numpy()
print("frustum shape:", fr.shape, " expect", (cfg.n_cameras*P, 3*cfg.pos3d_depth_bins))

# ---- independent reconstruction ----
def q2m(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
sx, sy = cfg.img_w/cfg.cam_native_w, cfg.img_h/cfg.cam_native_h
fx, fy = cfg.cam_fx*sx, cfg.cam_fy*sy
cx, cy = cfg.cam_cx*sx, cfg.cam_cy*sy
print(f"rescaled intrinsics: sx={sx:.6f} sy={sy:.6f} -> fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
depths = np.linspace(cfg.pos3d_near_m, cfg.pos3d_far_m, cfg.pos3d_depth_bins)
def ray(cam_i, row, col):
    u = (col+0.5)*(cfg.img_w/gw); v = (row+0.5)*(cfg.img_h/gh)
    R = q2m(cfg.cam_q[cam_i]); t = np.array(cfg.cam_t[cam_i])
    out=[]
    for d in depths:
        p_cam = np.array([ (u-cx)/fx*d, (v-cy)/fy*d, d ])
        out.append(R@p_cam + t)
    return np.concatenate(out)/cfg.pos3d_far_m, u, v

print()
print("PATCH-BY-PATCH comparison (independent numpy vs model._frustum):")
maxerr=0.0
tests=[(0,gh//2,gw//2),(0,0,0),(1,gh//2,gw//2),(2,gh//2,gw//2),(3,gh//2,gw//2),(3,gh-1,gw-1),(1,5,7),(2,29,11)]
for ci,r,c in tests:
    mine,u,v = ray(ci,r,c)
    theirs = fr[ci*P + r*gw + c]
    e = np.abs(mine-theirs).max(); maxerr=max(maxerr,e)
    print(f"  cam{ci} {cfg.cameras[ci]:7s} patch(r={r:2d},c={c:2d}) px=({u:6.1f},{v:6.1f})  "
          f"model={np.round(theirs,4)}  mine={np.round(mine,4)}  maxerr={e:.2e}")
print(f"\n  MAX ABS ERROR over the sampled patches: {maxerr:.3e}")

print()
print("PHYSICAL SANITY -- the CENTRE patch's far point, in METRES in the ego frame:")
for ci in range(cfg.n_cameras):
    mine,_,_ = ray(ci, gh//2, gw//2)
    far = mine[3:]*cfg.pos3d_far_m
    near = mine[:3]*cfg.pos3d_far_m
    print(f"  {cfg.cameras[ci]:7s} near(d=1m) ego=({near[0]:7.2f},{near[1]:7.2f},{near[2]:6.2f})   "
          f"far(d=60m) ego=({far[0]:7.2f},{far[1]:7.2f},{far[2]:6.2f})")
print("  expectation: F0 far x~+60 ; B0 far x~-60 ; L0 far y>0 ; R0 far y<0")
print()
# do the four cameras actually get DIFFERENT codes?
blocks=[fr[i*P:(i+1)*P] for i in range(cfg.n_cameras)]
print("pairwise max|difference| between camera blocks (0.0 would mean the extrinsics are inert):")
for i in range(cfg.n_cameras):
    for j in range(i+1,cfg.n_cameras):
        print(f"   {cfg.cameras[i]} vs {cfg.cameras[j]}: {np.abs(blocks[i]-blocks[j]).max():.4f}")
