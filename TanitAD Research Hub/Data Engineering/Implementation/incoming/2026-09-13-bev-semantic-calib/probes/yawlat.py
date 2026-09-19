import sys, numpy as np, cv2
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import lane_width as LW, lag_scale as LS, run_real as RR
from trajlib import lane_calib as LC

# Whole-profile cross-correlation between two RANGE slabs. No peak is identified,
# so the selector instability that wrecked the slab comparison cannot occur: a
# residual yaw eps displaces the lane laterally by -x*eps, so the near->far shift
# is -(x_far - x_near)*eps regardless of which features are present.
recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
NEAR,FAR=(8.,12.),(16.,22.); dx=0.5*(FAR[0]+FAR[1])-0.5*(NEAR[0]+NEAR[1])
def shift(a,b,cell,maxlag_m=1.5):
    n=int(maxlag_m/cell); a=a-a.mean(); b=b-b.mean()
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    if na<=0 or nb<=0: return None
    best=(-9,0)
    for k in range(-n,n+1):
        aa=a[max(0,k):len(a)+min(0,k)]; bb=b[max(0,-k):len(b)+min(0,-k)]
        if len(aa)<20: continue
        r=float((aa*bb).sum()/(np.linalg.norm(aa)*np.linalg.norm(bb)+1e-9))
        if r>best[0]: best=(r,k)
    return best[1]*cell, best[0]
for yaw in (-7.01,-6.5,-7.5):
    P=dict(RR.NOMINAL); P.update(yaw=np.deg2rad(yaw), height=1.70, fx=1438.0, lateral=-0.12)
    P["pitch"]=LS.pitch_for_horizon(P,465.0)
    S=[]
    for pi in np.linspace(0,len(usable)-1,150).astype(int):
        f=int(usable[pi]["frame"]); im=cv2.imread(str(fdir/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        H=im.shape[0]; u,v=LC._ridge_points(im,int(0.55*H),int(0.92*H))
        if len(u)<150: continue
        uv=np.stack([u,v],1).astype(float)
        y1,p1=LW.lateral_profile(uv,P,x_win=NEAR,cell=0.02)
        y2,p2=LW.lateral_profile(uv,P,x_win=FAR,cell=0.02)
        if y1 is None or y2 is None: continue
        s=shift(p1,p2,0.02)
        if s and s[1]>0.45: S.append(s[0])
    if len(S)<20: print(f"yaw {yaw}: only {len(S)} frames locked"); continue
    S=np.array(S); med=float(np.median(S))
    eps=-med/dx
    print(f"yaw {yaw:+.2f} deg  n {len(S):3d}  near->far lateral shift {med:+.3f} m "
          f"(IQR {np.percentile(S,75)-np.percentile(S,25):.3f})  =>  residual yaw "
          f"{np.rad2deg(np.arctan(eps)):+.3f} deg  =>  true yaw {yaw+np.rad2deg(np.arctan(eps)):+.3f} deg")
