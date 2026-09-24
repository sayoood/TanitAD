#!/usr/bin/env python3
"""Per-frame image shift relative to the CAR, from car-fixed image content.

The windscreen sticker (translucent) and the bonnet edge (glossy) are rigid to the
car but their pixels are contaminated by the moving world behind/on them. At 20 m/s
the world changes completely within a second while the car-fixed layer does not,
so a ROLLING TEMPORAL MEDIAN over +-HALF frames recovers the car-fixed layer and
averages the world away. Each rolling-median image is then registered (phase
correlation on gradient magnitude, sub-pixel) against the median at a reference
time. Two regions at opposite corners give two INDEPENDENT estimates of the same
image shift: agreement is the validation.
"""
import sys, pathlib
import numpy as np, cv2
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(HERE))
import run_real as RR

REGIONS={"sticker":(40,420,0,600), "bonnet":(860,1080,160,1760)}   # y0,y1,x0,x1
HALF=15

def grad(g):
    g=cv2.GaussianBlur(g,(0,0),1.5)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return cv2.magnitude(gx,gy)

def run(frames, recs, stride=3):
    crops={k:[] for k in REGIONS}
    for f in frames:
        g=cv2.imread(str(RR.RUN/"frames"/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
        for k,(y0,y1,x0,x1) in REGIONS.items():
            crops[k].append(g[y0:y1,x0:x1])
    stacks={k:np.stack(v) for k,v in crops.items()}
    n=len(frames); ref_i=n//2
    win=np.hanning  # not used; kept simple
    res=[]
    ref={}
    for k in REGIONS:
        lo,hi=max(0,ref_i-HALF),min(n,ref_i+HALF+1)
        ref[k]=grad(np.median(stacks[k][lo:hi],axis=0).astype(np.float32))
    hw={k:cv2.createHanningWindow(ref[k].shape[::-1],cv2.CV_32F) for k in REGIONS}
    for i in range(0,n,stride):
        lo,hi=max(0,i-HALF),min(n,i+HALF+1)
        row=[frames[i], recs[frames[i]]["t_session_s"]]
        for k in REGIONS:
            m=grad(np.median(stacks[k][lo:hi],axis=0).astype(np.float32))
            (sx,sy),resp=cv2.phaseCorrelate(ref[k],m,hw[k])
            row+=[sx,sy,resp]
        res.append(row)
    return np.array(res)

if __name__=="__main__":
    recs={int(r["frame"]):r for r in RR.load_records(RR.RUN)}
    have={int(p.stem) for p in (RR.RUN/"frames").glob("*.jpg")}
    frames=[f for f in sorted(recs) if f in have and recs[f]["complete"]]
    A=run(frames,recs)
    np.save(sys.argv[1],A); print(f"{len(A)} samples -> {sys.argv[1]}")
