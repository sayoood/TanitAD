"""Definitive check: render GT fan under candidate projections on real frames.
A = current model      (f=266, H=1.22, horizon=128)   [red]
B = focal+height fix   (f=444, H=1.43, horizon=128)    [green]
C = VP-horizon shift    (f=266, H=1.22, horizon=150)   [magenta]
Pick straight-driving frames; save side comparisons. GT fan is real odometry.
"""
import glob, math, os
import numpy as np, torch
from PIL import Image, ImageDraw

WAYPOINT_STEPS = (5, 10, 15, 20); X_CLIP = 2.0
def proj(xy, h, w, f, H, horizon):
    x = np.clip(xy[:,0], X_CLIP, None); fe = f*(h/256.0)
    u = w/2 - fe*(xy[:,1]/x); v = horizon + fe*(H/x)
    return np.stack([u,v],1)
def ego_frame(dxy, yaw):
    c,s = math.cos(-yaw), math.sin(-yaw)
    return np.array([dxy[0]*c-dxy[1]*s, dxy[0]*s+dxy[1]*c])
def gt_fan(poses, last):
    yaw0=float(poses[last,2]); p0=poses[last,:2].numpy().astype(float)
    return np.array([ego_frame(poses[last+k,:2].numpy().astype(float)-p0, yaw0)
                     for k in WAYPOINT_STEPS])
def anchor_rgb(fr, last): return fr[last][-3:].permute(1,2,0).contiguous().numpy().astype(np.uint8)
SPD_MIN=float(os.environ.get("SPD_MIN","2.5"))
def straight(poses,last,need=20):
    if last+need>=poses.shape[0]: return False
    fan=gt_fan(poses,last); return float(poses[last,3])>SPD_MIN and fan[-1,0]>5 and abs(fan[-1,1])<1.2

CANDS = [("A_cur",266,1.22,128,(255,60,60)), ("B_fix",444,1.43,128,(60,230,60)),
         ("C_vp",266,1.22,150,(230,60,230))]

def render(rgb, fan, save):
    im=Image.fromarray(rgb).resize((512,512),Image.NEAREST); d=ImageDraw.Draw(im); sc=2.0
    d.line([(0,128*sc),(512,128*sc)],fill=(80,160,255),width=1)
    for _,f,H,hz,col in CANDS:
        px=proj(np.vstack([[0.,0.],fan]),256,256,f,H,hz)*sc
        d.line([tuple(p) for p in px],fill=col,width=3)
        for p in px[1:]: d.ellipse([p[0]-3,p[1]-3,p[0]+3,p[1]+3],fill=col)
    im.save(save)

def run(cache, tag, out, n=4):
    eps=sorted(glob.glob(os.path.join(cache,'ep_*.pt'))); saved=0
    for p in eps:
        d=torch.load(p,map_location='cpu',weights_only=False)
        fr,po=d['frames_u8'],d['poses']; T=fr.shape[0]
        for last in range(20,T-21,15):
            if straight(po,last):
                fan=gt_fan(po,last)
                render(anchor_rgb(fr,last),fan,os.path.join(out,f'{tag}_cmp_{saved}.png'))
                vs={c[0]:round(float(proj(fan,256,256,c[1],c[2],c[3])[-1,1]),1) for c in CANDS}
                print(f'{tag}_{saved} last={last} spd={float(po[last,3]):.1f} far_wp_x={fan[-1,0]:.1f} far_v={vs}')
                saved+=1; break
        if saved>=n: break

if __name__=='__main__':
    out='/workspace/gt_check2'; os.makedirs(out,exist_ok=True)
    run('/workspace/data/physicalai/_epcache/physicalai-val-8c0d3047924e','phys',out,4)
    run('/workspace/data/comma2k19/_epcache/comma2k19-val-61c46fca8f7f','comma',out,2)
