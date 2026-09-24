#!/usr/bin/env python3
"""Per-frame camera pointing from an object FIXED TO THE CAR: the windscreen sticker.

⛔ WHY. The phone ran electronic stabilisation, which moves the picture relative to
the car during the clip (lane vanishing point walks ~80 px, Part 30). A lane-based
estimate is noisy per frame and fails in curves and behind vehicles; smoothing it
lags a fast jump. The sticker strip glued inside the windscreen (top-left, with a
QR code) is rigid to the car, so ITS image position moves only when the image
moves relative to the car -- exactly the quantity the overlay needs, per frame,
with no lag and no dependence on the road.

Method: gradient-magnitude template (strip + QR + its right-hand corner) from one
reference frame, matched against EVERY frame with normalised cross-correlation
and a sub-pixel peak. Matching against a FIXED reference, not the previous frame,
means errors cannot accumulate.
"""
import sys, pathlib, json
import numpy as np, cv2
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(HERE))
import run_real as RR

REF=200
TPL=(250, 365, 25, 205)          # y0,y1,x0,x1 in the reference frame
SEARCH=70                        # +- px

def grad(g):
    g=cv2.GaussianBlur(g,(0,0),1.2).astype(np.float32)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return cv2.magnitude(gx,gy)

def subpix(r, y, x):
    def f(a,b,c):
        d=a-2*b+c
        return 0.0 if abs(d)<1e-9 else 0.5*(a-c)/d
    dy=f(r[y-1,x],r[y,x],r[y+1,x]) if 0<y<r.shape[0]-1 else 0.0
    dx=f(r[y,x-1],r[y,x],r[y,x+1]) if 0<x<r.shape[1]-1 else 0.0
    return y+dy, x+dx

def track(frames, recs):
    y0,y1,x0,x1=TPL
    ref=grad(cv2.imread(str(RR.RUN/"frames"/f"{REF:06d}.jpg"),cv2.IMREAD_GRAYSCALE))
    T=ref[y0:y1,x0:x1]
    out=[]
    for f in frames:
        g=cv2.imread(str(RR.RUN/"frames"/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        G=grad(g)
        sy0=max(0,y0-SEARCH); sx0=max(0,x0-SEARCH)
        S=G[sy0:y1+SEARCH, sx0:x1+SEARCH]
        r=cv2.matchTemplate(S,T,cv2.TM_CCOEFF_NORMED)
        _,mx,_,loc=cv2.minMaxLoc(r)
        py,px=subpix(r,loc[1],loc[0])
        dy=sy0+py-y0; dx=sx0+px-x0
        # second-best peak outside a 9 px exclusion: a unique match stands clear of it
        r2=r.copy(); yy,xx=loc[1],loc[0]
        r2[max(0,yy-9):yy+10, max(0,xx-9):xx+10]=-1
        out.append((f, recs[f]["t_session_s"], dx, dy, mx, float(r2.max())))
    return np.array(out)

if __name__=="__main__":
    recs={int(r["frame"]):r for r in RR.load_records(RR.RUN)}
    have={int(p.stem) for p in (RR.RUN/"frames").glob("*.jpg")}
    frames=[f for f in sorted(recs) if f in have and recs[f]["complete"]]
    A=track(frames,recs)
    out=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else HERE.parent/"attitude"/"fiducial_track.npy"
    np.save(out,A); print(f"{len(A)} frames -> {out}")
