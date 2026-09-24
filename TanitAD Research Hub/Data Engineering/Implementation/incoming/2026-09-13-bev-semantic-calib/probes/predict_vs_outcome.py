#!/usr/bin/env python3
"""Did the car end up where the ribbon said? -- measured against the LANE LINES.

The PI's original specification: a metric for "not leaving the road" with a clear,
verified goal, "since we know the future". This is that metric, per frame:

  PREDICTION  at frame F: the drawn ribbon's centre at range X, as a fraction of the
              lane width, relative to the lane lines fitted in frame F at that row.
  OUTCOME     at frame F', when the car has travelled X - X_NEAR metres: the ribbon's
              centre at X_NEAR (near field, where a yaw error moves it only
              X_NEAR*tan(err)), relative to the lane lines fitted in frame F'.

Both refer to the SAME patch of road. If the drawing is right, they are equal --
whatever the car was doing, including drifting across its lane. The difference, in
metres, is the drawing's error at range X for that frame. It assumes nothing about
the car being centred or parallel, only that the painted lines stay put.
"""
import sys, pathlib, numpy as np, cv2
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(HERE))
import lane_vp_wholeline as V, run_real as RR, bev_calib as BC

LANE_M=3.5
X_NEAR=8.0

def lanes(F, P, recs, cache):
    if F in cache: return cache[F]
    img=cv2.imread(str(RR.RUN/"frames"/f"{F:06d}.jpg"),cv2.IMREAD_COLOR)
    out=None
    if img is not None:
        rib=V.ribbon_uv(P,recs[F],np.arange(7.,60.,1.0))
        if len(rib)>=5:
            fl,fr,_=V.lane_lines(img,rib)
            if fl and fr:
                (al,bl,nl),(ar,br,nr)=fl,fr
                if nl>=6 and nr>=3 and -3.5<bl<-0.6 and 0.6<br<3.5:
                    vvp=(ar-al)/(bl-br)
                    if 440<vvp<520 and 380<(ar+br*720)-(al+bl*720)<750:
                        out=(al,bl,ar,br)
    cache[F]=out; return out

def ribbon_frac(F, X, P, recs, L):
    """Ribbon centre at range X (from the lens) as a lane fraction, + = RIGHT of centre."""
    r=recs[F]; px=np.asarray(r["x"],float); py=np.asarray(r["y"],float); m=px>0
    LON=float(P["longitudinal"]); xc=X+LON
    if xc>px[m].max(): return None
    uv=BC.project_ground(np.array([[xc,float(np.interp(xc,px[m],py[m]))]]),P)
    if not np.isfinite(uv).all(): return None
    u,v=float(uv[0,0]),float(uv[0,1])
    al,bl,ar,br=L; uL=al+bl*v; uR=ar+br*v
    if uR-uL<20: return None
    return (u-0.5*(uL+uR))/(uR-uL)

def run(frames, recs, yaw_of_t, hz=463.0, lat=-0.260, Xs=(20.,30.,40.)):
    t=np.array([recs[f]["t_session_s"] for f in frames])
    spd=np.array([recs[f]["speed_ms"] for f in frames])
    s=np.concatenate([[0.0],np.cumsum(0.5*(spd[1:]+spd[:-1])*np.diff(t))])
    idx={f:i for i,f in enumerate(frames)}
    cache={}; rows=[]
    for f in frames[::2]:
        i=idx[f]; P=V.params(yaw_of_t(t[i]),lat,hz)
        L=lanes(f,P,recs,cache)
        if L is None: continue
        for X in Xs:
            p=ribbon_frac(f,X,P,recs,L)
            if p is None: continue
            j=int(np.searchsorted(s, s[i]+X-X_NEAR))
            if j>=len(frames): continue
            f2=frames[j]; P2=V.params(yaw_of_t(t[j]),lat,hz)
            L2=lanes(f2,P2,recs,cache)
            if L2 is None: continue
            q=ribbon_frac(f2,X_NEAR,P2,recs,L2)
            if q is None: continue
            rows.append((f,t[i],X,p,q,(p-q)*LANE_M))
    return np.array(rows)
