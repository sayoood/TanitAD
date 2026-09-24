import sys, numpy as np, cv2
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import flow_scale as FS, run_real as RR
from trajlib import lane_calib as LC

recs = RR.load_records(RR.RUN); fdir = RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
step=1
V0,DV,ALL_V, D_list = [],[],[],[]
for pi in np.linspace(0,len(usable)-1,25).astype(int):
    a=usable[pi]; f0=int(a["frame"])
    i0=cv2.imread(str(fdir/f"{f0:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    p1=fdir/f"{f0+step:06d}.jpg"
    if i0 is None or not p1.exists(): continue
    i1=cv2.imread(str(p1),cv2.IMREAD_GRAYSCALE)
    m=FS.road_mask(i0)
    pts=cv2.goodFeaturesToTrack(i0,maxCorners=800,qualityLevel=0.005,minDistance=6,mask=m,blockSize=7)
    if pts is None: continue
    ALL_V.append(pts.reshape(-1,2)[:,1])
    q1,st1,_=cv2.calcOpticalFlowPyrLK(i0,i1,pts,None,**FS.LK)
    q0,st0,_=cv2.calcOpticalFlowPyrLK(i1,i0,q1,None,**FS.LK)
    ok=(st1.ravel()==1)&(st0.ravel()==1)&(np.linalg.norm((q0-pts).reshape(-1,2),axis=1)<0.6)
    uv0=pts.reshape(-1,2)[ok]; uv1=q1.reshape(-1,2)[ok]
    t=np.asarray(a["t"],float); dt=step/29.922
    D_list.append(float(np.interp(dt,t,np.asarray(a["x"],float))))
    V0.append(uv0[:,1]); DV.append(uv1[:,1]-uv0[:,1])
V0=np.concatenate(V0); DV=np.concatenate(DV); ALL_V=np.concatenate(ALL_V)
D=float(np.median(D_list))
print(f"step {step}: D = {D:.3f} m   seeds {len(ALL_V)}   tracked {len(V0)} "
      f"({100*len(V0)/len(ALL_V):.0f}%)\n")
print(" row band     seeds  tracked   rate    measured dv (median)   model dv @ f*h=1522, v_h=523")
edges=np.arange(580,1081,50)
for lo,hi in zip(edges[:-1],edges[1:]):
    ns=int(((ALL_V>=lo)&(ALL_V<hi)).sum())
    m=(V0>=lo)&(V0<hi); nt=int(m.sum())
    q=0.5*(lo+hi)-523.4
    pred=D*q*q/max(1522.6-D*q,1e-6)
    md=np.median(DV[m]) if nt>5 else float('nan')
    print(f" {lo:4d}-{hi:4d}  {ns:7d} {nt:8d}  {100*nt/max(ns,1):5.1f}%   {md:12.2f} px      {pred:10.2f} px")
