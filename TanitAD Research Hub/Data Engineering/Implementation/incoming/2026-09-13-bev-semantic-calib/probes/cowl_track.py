#!/usr/bin/env python3
"""Per-frame image shift from the windscreen/bonnet COWL edge (PI's suggestion, 2026-09-24).

The dark cowl strip with the wiper arms at the bottom of the frame (rows ~1030-1080)
is opaque, non-reflective and rigid to the car, so -- unlike the glossy bonnet or the
translucent windscreen sticker -- it may register per SINGLE frame, with no temporal
median and hence full frame-rate bandwidth. Registered by phase correlation on
gradient magnitude against a fixed reference frame, per frame.
"""
import sys, pathlib, numpy as np, cv2
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(HERE))
import run_real as RR
Y0,Y1,X0,X1=1022,1080,120,1800

def grad(g):
    g=cv2.GaussianBlur(g,(0,0),1.0).astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3))

def run(frames, recs, half=0):
    crops=np.stack([cv2.imread(str(RR.RUN/"frames"/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)[Y0:Y1,X0:X1]
                    for f in frames])
    n=len(frames); ri=n//2
    def img(i):
        if half==0: return grad(crops[i])
        lo,hi=max(0,i-half),min(n,i+half+1)
        return grad(np.median(crops[lo:hi],axis=0).astype(np.uint8))
    ref=img(ri); hw=cv2.createHanningWindow(ref.shape[::-1],cv2.CV_32F)
    out=[]
    for i in range(n):
        (sx,sy),r=cv2.phaseCorrelate(ref,img(i),hw)
        out.append((frames[i],recs[frames[i]]["t_session_s"],sx,sy,r))
    return np.array(out)

if __name__=="__main__":
    recs={int(r["frame"]):r for r in RR.load_records(RR.RUN)}
    have={int(p.stem) for p in (RR.RUN/"frames").glob("*.jpg")}
    frames=[f for f in sorted(recs) if f in have and recs[f]["complete"]]
    for half in (0,4):
        A=run(frames,recs,half); np.save(sys.argv[1].replace(".npy",f"_h{half}.npy"),A)
        print(f"half={half}: {len(A)} frames, response median {np.median(A[:,4]):.3f}")
