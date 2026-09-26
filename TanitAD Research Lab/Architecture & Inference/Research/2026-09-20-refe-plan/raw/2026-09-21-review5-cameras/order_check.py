"""Does token n*P+p in `visual` belong to camera n -- i.e. does the feature order match the
frustum's concatenation order?  Tested through the REAL REFe.forward, with only the heavy trunk
replaced by a tagging stub, and the answer read off reg_compress's own input via a pre-hook."""
import sys, torch, torch.nn as nn, numpy as np
sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
from model import REFe, REFeConfig

cfg = REFeConfig.for_backbone("vits16")     # width 384: order logic is width-independent
torch.manual_seed(0)
m = REFe(cfg)
gh, gw = cfg.img_h//cfg.patch, cfg.img_w//cfg.patch
P = gh*gw

class TagTrunk(nn.Module):
    """Returns, for each image in the B*N batch, a constant token equal to that image's own
    mean pixel value -- so a token's value IS the identity of the image it came from."""
    def forward(self, x):                       # [B*N,3,H,W] -> [B*N,P,width]
        tag = x.mean(dim=(1,2,3))               # [B*N]
        return tag[:,None,None].expand(x.shape[0], P, cfg.width).clone()
m.backbone = TagTrunk()

cap = {}
def pre(mod, args):
    cap["visual"] = args[1].detach().clone()
m.reg_compress.register_forward_pre_hook(pre)

B, N = 2, cfg.n_cameras
img = torch.zeros(B, N, 3, cfg.img_h, cfg.img_w)
for b in range(B):
    for n in range(N):
        img[b, n] = 100*(b+1) + (n+1)           # a unique constant per (batch, camera)
ego  = torch.zeros(B, cfg.ego_dim)
goal = torch.zeros(B, 2*cfg.n_goal_points)
m.eval()
with torch.no_grad():
    m(img, ego, goal)
vis = cap["visual"]                              # [B, N*P, width]  (AFTER pos3d was added)
print(f"visual shape {tuple(vis.shape)}  expect ({B}, {N*P}, {cfg.width})")

# strip the pos3d contribution so the surviving value is the trunk tag
with torch.no_grad():
    p3 = m._pos3d(img, N)[0]                     # [N*P, width]
print()
print("token block -> recovered trunk tag (expected 100*(b+1) + (cam+1)):")
ok = True
for b in range(B):
    for n in range(N):
        blk = vis[b, n*P:(n+1)*P] - p3[n*P:(n+1)*P]
        val = float(blk.mean()); spread = float(blk.std())
        exp = 100*(b+1) + (n+1)
        good = abs(val-exp) < 1e-2 and spread < 1e-3
        ok &= good
        print(f"  b={b} block {n} ({cfg.cameras[n]:7s}): recovered {val:8.3f}  expected {exp:8.3f}  "
              f"within-block std {spread:.2e}  {'OK' if good else 'MISMATCH'}")
print(f"\nFEATURE/POSITION ORDER AGREE: {ok}")

# and the complementary half: is the pos3d block for camera n really camera n's frustum?
fr = m._frustum(gh, gw, "cpu", torch.float32, N)[0]
print("\npos3d block identity cross-check (frustum far-point x, metres, centre patch):")
for n in range(N):
    far_x = float(fr[n*P + (gh//2)*gw + gw//2][3]) * cfg.pos3d_far_m
    print(f"  block {n} = {cfg.cameras[n]:7s}: far-point ego x = {far_x:+7.2f} m")
print("  (F0 must be ~+60, B0 ~-60; if these were swapped the order would be wrong)")

# NEGATIVE CONTROL: deliberately reverse the camera order in the INPUT and prove the test reddens
print("\n-- deliberate regression: feed the cameras in REVERSED order --")
img_r = img.flip(1)
with torch.no_grad():
    m(img_r, ego, goal)
vis_r = cap["visual"]
bad = 0
for n in range(N):
    blk = vis_r[0, n*P:(n+1)*P] - p3[n*P:(n+1)*P]
    val = float(blk.mean()); exp = 100*1 + (n+1)
    if abs(val-exp) >= 1e-2: bad += 1
print(f"  blocks that now MISMATCH: {bad} of {N}   "
      f"{'the order test is live' if bad>0 else 'INERT -- the test cannot fail'}")
