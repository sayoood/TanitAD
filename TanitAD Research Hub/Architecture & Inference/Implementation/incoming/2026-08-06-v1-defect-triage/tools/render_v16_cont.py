"""CONTINUOUS reel: GT + v1.6 only, real-time playback.

⛔ NOT piecewise: the stride-1 eval dump carries a prediction for EVERY frame, so each
video frame is the camera at time t with the 2 s v1.6 plan predicted FROM t, advancing at
10 fps = real time. 40 episodes back-to-back.
"""
import argparse, glob, math, os, shutil, subprocess
import numpy as np
import torch
from PIL import Image, ImageDraw

COL_GT=(238,51,119); COL_B=(118,185,0)
W_CAM,W_BEV,PAD,H_BAN=512,470,10,64

def _ffmpeg():
    exe=shutil.which("ffmpeg")
    if exe: return exe
    import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()

def bev(size,gt,b,fonts):
    w,h=size; im=Image.new("RGB",(w,h),(8,11,15)); d=ImageDraw.Draw(im)
    allp=[p for p in (gt,b) if len(p)]
    xmax=max(12.0,max(float(np.max(p[:,0])) for p in allp)*1.1)
    ymax=max(4.0,max(float(np.max(np.abs(p[:,1]))) for p in allp)*1.3)
    pad=34; cx,by,top=w//2,h-pad,pad+14
    def m2px(X,Y): return (cx-(Y/ymax)*((w/2)-pad), by-(max(X,0.0)/xmax)*(by-top))
    step=10 if xmax>25 else 5; r=step
    while r<=xmax+0.1:
        _,py=m2px(r,0); d.line([(12,py),(w-12,py)],fill=(34,42,52))
        d.text((14,py-12),f"{r:g} m",fill=(96,106,120),font=fonts["tiny"]); r+=step
    d.line([(cx,top),(cx,by)],fill=(44,54,66))
    for path,col,wd in ((gt,COL_GT,6),(b,COL_B,3)):
        pts=[m2px(float(p[0]),float(p[1])) for p in path]
        if len(pts)>=2: d.line(pts,fill=col,width=wd)
    d.polygon([(cx-6,by),(cx+6,by),(cx,by-12)],fill=(232,236,242))
    for i,(lab,col) in enumerate((("ground truth",COL_GT),("v1.6 (unicycle readout)",COL_B))):
        d.line([(14,h-36+i*15),(34,h-36+i*15)],fill=col,width=4)
        d.text((40,h-43+i*15),lab,fill=(200,208,218),font=fonts["tiny"])
    return im

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump",required=True); ap.add_argument("--corpus",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--fps",type=int,default=10)
    ap.add_argument("--frame-stride",type=int,default=1)
    a=ap.parse_args()
    from taniteval.corpus_overlay import FlatProjector
    from taniteval.data import load_frames
    from taniteval.flagship_overlay import _font
    fonts={"big":_font(18),"hud":_font(14),"tiny":_font(11)}
    proj=FlatProjector(128.0); ffmpeg=_ffmpeg()
    W=PAD+W_CAM+PAD+W_BEV+PAD; H=H_BAN+W_CAM+PAD
    fr=os.path.join(os.path.dirname(a.out) or ".","_v16c"); shutil.rmtree(fr,ignore_errors=True); os.makedirs(fr)
    files=sorted(glob.glob(os.path.join(a.corpus,"ep_*.pt")))
    dumps=sorted(glob.glob(os.path.join(a.dump,"ep*.npz")))
    n=0
    for fi,(epf,dpf) in enumerate(zip(files,dumps)):
        z=np.load(dpf); B,G,ws=z["b"],z["g"],z["ws"]
        ep=load_frames([epf])[0]
        for wi in range(0,len(ws),a.frame_stride):
            t=int(ws[wi]); gb,bb=G[wi],B[wi]
            rgb=torch.as_tensor(ep.feats[t,-3:]).permute(1,2,0).numpy()
            cam=Image.fromarray(rgb).resize((W_CAM,W_CAM),Image.LANCZOS)
            cd=ImageDraw.Draw(cam)
            for path,col,wd in ((gb,COL_GT,6),(bb,COL_B,3)):
                pts=proj(path)
                if len(pts)>=2: cd.line(pts,fill=col,width=wd)
            canvas=Image.new("RGB",(W,H),(6,9,12)); d=ImageDraw.Draw(canvas)
            canvas.paste(cam,(PAD,H_BAN))
            canvas.paste(bev((W_BEV,W_CAM),gb,bb,fonts),(PAD+W_CAM+PAD,H_BAN))
            d.rectangle([0,0,W,H_BAN],fill=(10,14,19))
            d.text((PAD,4),f"flagship v1.6 (flagship-v16-unicycle) — CONTINUOUS, real-time 10 fps · "
                           f"OOD-val q90 ep {fi:02d} · t={t*0.1:5.1f} s",
                   fill=(233,237,243),font=fonts["big"])
            d.text((PAD,28),"⛔ WM rollout under TRUE future actions — NOT closed-loop planning. "
                            "2 s / 20 wp plan re-predicted EVERY frame.",
                   fill=(235,180,90),font=fonts["tiny"])
            ade=float(np.linalg.norm(bb-gb,axis=-1).mean())
            d.text((PAD,44),f"ADE@2s {ade:.3f} m · v0 {float(ep.poses[t,3]):4.1f} m/s",
                   fill=(150,160,175),font=fonts["hud"])
            canvas.save(os.path.join(fr,f"f{n:05d}.png")); n+=1
        print(f"[{fi+1}/40] frames {n}",flush=True)
    subprocess.run([ffmpeg,"-y","-r",str(a.fps),"-i",os.path.join(fr,"f%05d.png"),
                    "-c:v","libx264","-pix_fmt","yuv420p","-crf","23","-movflags","+faststart",
                    a.out],check=True,capture_output=True)
    shutil.rmtree(fr,ignore_errors=True)
    print(f"[video] {a.out} {n} frames {n/a.fps:.0f}s",flush=True)

if __name__=="__main__":
    main()
