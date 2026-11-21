"""Long open-loop reel: flagship v1.6 (unicycle readout) vs v1arch, known overlays.

Panels (the established visual language of this programme's reels):
  LEFT   the model's ACTUAL input — the 256x256 front crop — with GT (pink),
         v1arch (orange) and v1.6 (green) projected in via FlatProjector.
  RIGHT  metric BEV, calibration-independent, same three paths + range rings.
  BANNER the caveats that must not be croppable: WM rollout under TRUE actions
         (not closed-loop), v1.6 = 2.11M readout on the FROZEN v1arch trunk.
  HUD    per-window ADE / net-yaw / jerk for both arms, and v0.
"""
import argparse, glob, json, math, os, shutil, subprocess
import numpy as np
import torch
from PIL import Image, ImageDraw

COL_GT = (238, 51, 119); COL_A = (255, 122, 61); COL_B = (118, 185, 0)
W_CAM, W_BEV, PAD, H_BAN = 512, 470, 10, 78

def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe: return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def geom(P):
    p = np.concatenate([np.zeros((1,2)), P], 0); d = p[1:]-p[:-1]
    ds = np.sqrt((d**2).sum(-1)+1e-12); sp = ds/0.1
    ac = (sp[1:]-sp[:-1])/0.1
    jerk = float(np.sqrt((((ac[1:]-ac[:-1])/0.1)**2).mean())) if len(ac)>2 else 0.0
    h = np.arctan2(d[:,1], d[:,0])
    dh = (h[1:]-h[:-1]+math.pi)%(2*math.pi)-math.pi
    ok = (ds[1:]>0.05)&(ds[:-1]>0.05)
    return float(np.where(ok, dh, 0).sum()), jerk

def bev(size, gt, a, b, fonts):
    w, h = size
    im = Image.new("RGB",(w,h),(8,11,15)); d = ImageDraw.Draw(im)
    allp=[p for p in (gt,a,b) if len(p)]
    xmax=max(12.0,max(float(np.max(p[:,0])) for p in allp)*1.1)
    ymax=max(4.0,max(float(np.max(np.abs(p[:,1]))) for p in allp)*1.3)
    pad=34; cx,by,top=w//2,h-pad,pad+14
    def m2px(X,Y): return (cx-(Y/ymax)*((w/2)-pad), by-(max(X,0.0)/xmax)*(by-top))
    step=10 if xmax>25 else 5; r=step
    while r<=xmax+0.1:
        _,py=m2px(r,0); d.line([(12,py),(w-12,py)],fill=(34,42,52))
        d.text((14,py-12),f"{r:g} m",fill=(96,106,120),font=fonts["tiny"]); r+=step
    d.line([(cx,top),(cx,by)],fill=(44,54,66))
    for path,col,wd in ((gt,COL_GT,6),(a,COL_A,3),(b,COL_B,3)):
        pts=[m2px(float(p[0]),float(p[1])) for p in path]
        if len(pts)>=2: d.line(pts,fill=col,width=wd)
    d.polygon([(cx-6,by),(cx+6,by),(cx,by-12)],fill=(232,236,242))
    for i,(lab,col) in enumerate((("ground truth",COL_GT),("v1arch (displacement)",COL_A),
                                  ("v1.6 (unicycle readout)",COL_B))):
        d.line([(14,h-50+i*15),(34,h-50+i*15)],fill=col,width=4)
        d.text((40,h-57+i*15),lab,fill=(200,208,218),font=fonts["tiny"])
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=40, help="window stride within an episode")
    ap.add_argument("--seconds", type=float, default=2.2); ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    from taniteval.corpus_overlay import FlatProjector
    from taniteval.data import load_frames
    from taniteval.flagship_overlay import _font
    fonts = {"big":_font(19),"hud":_font(14),"tiny":_font(11)}
    proj = FlatProjector(128.0)
    ffmpeg = _ffmpeg()
    W = PAD+W_CAM+PAD+W_BEV+PAD; H = H_BAN+W_CAM+PAD
    fr_dir = os.path.join(os.path.dirname(a.out) or ".", "_v16_frames")
    shutil.rmtree(fr_dir, ignore_errors=True); os.makedirs(fr_dir)
    files = sorted(glob.glob(os.path.join(a.corpus,"ep_*.pt")))
    dumps = sorted(glob.glob(os.path.join(a.dump,"ep*.npz")))
    n_out=0; per=max(1,int(a.seconds*a.fps))
    for fi,(epf,dpf) in enumerate(zip(files,dumps)):
        z=np.load(dpf); A,B,G,ws=z["a"],z["b"],z["g"],z["ws"]
        ep=load_frames([epf])[0]
        for wi in range(0,len(ws),a.every):
            t=int(ws[wi]); ga,aa,bb=G[wi],A[wi],B[wi]
            rgb=torch.as_tensor(ep.feats[t,-3:]).permute(1,2,0).numpy()
            cam=Image.fromarray(rgb).resize((W_CAM,W_CAM),Image.LANCZOS)
            cd=ImageDraw.Draw(cam)
            for path,col,wd in ((ga,COL_GT,6),(aa,COL_A,3),(bb,COL_B,3)):
                pts=proj(path)
                if len(pts)>=2: cd.line(pts,fill=col,width=wd)
            canvas=Image.new("RGB",(W,H),(6,9,12)); d=ImageDraw.Draw(canvas)
            canvas.paste(cam,(PAD,H_BAN))
            canvas.paste(bev((W_BEV,W_CAM),ga,aa,bb,fonts),(PAD+W_CAM+PAD,H_BAN))
            d.rectangle([0,0,W,H_BAN],fill=(10,14,19))
            d.text((PAD,5),f"flagship v1.6 (unicycle readout)  vs  v1arch — OOD-val q90 · "
                           f"ep {fi:02d} · window {t}",fill=(233,237,243),font=fonts["big"])
            d.text((PAD,30),"⛔ WM rollout under TRUE future actions — NOT closed-loop planning "
                            "(applies to BOTH arms). v1.6 = 2.11M readout on the FROZEN v1arch trunk.",
                   fill=(235,180,90),font=fonts["tiny"])
            d.text((PAD,45),"v1.6 registry key flagship-v16-unicycle — NOT the 2026-07 "
                            "flagship-v16-ab-ft ('old v1.6'). 2 s / 20 wp / dense 0.1 s grid.",
                   fill=(235,180,90),font=fonts["tiny"])
            nyA,jkA=geom(aa); nyB,jkB=geom(bb); nyG,_=geom(ga)
            adeA=float(np.linalg.norm(aa-ga,axis=-1).mean()); adeB=float(np.linalg.norm(bb-ga,axis=-1).mean())
            d.text((PAD,60),f"ADE  v1arch {adeA:.3f} m · v1.6 {adeB:.3f} m    "
                            f"net-yaw err  {abs(nyA-nyG):.3f} / {abs(nyB-nyG):.3f} rad    "
                            f"jerk {jkA:.1f} / {jkB:.1f} m/s³    v0 {float(ep.poses[t,3]):.1f} m/s",
                   fill=(150,160,175),font=fonts["hud"])
            for _ in range(per):
                canvas.save(os.path.join(fr_dir,f"f{n_out:05d}.png")); n_out+=1
        if (fi+1)%10==0: print(f"[{fi+1}] frames {n_out}",flush=True)
    subprocess.run([ffmpeg,"-y","-r",str(a.fps),"-i",os.path.join(fr_dir,"f%05d.png"),
                    "-c:v","libx264","-pix_fmt","yuv420p","-crf","23","-movflags","+faststart",
                    a.out],check=True,capture_output=True)
    shutil.rmtree(fr_dir,ignore_errors=True)
    print(f"[video] {a.out} {n_out} frames {n_out/a.fps:.0f}s",flush=True)

if __name__=="__main__":
    main()
