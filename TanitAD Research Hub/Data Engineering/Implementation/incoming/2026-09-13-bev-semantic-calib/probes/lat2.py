import sys, numpy as np, cv2
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import lane_width as LW, lag_scale as LS, run_real as RR
from trajlib import lane_calib as LC
recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
print("A pure LATERAL offset displaces the lane centre by the SAME amount at every")
print("range. A YAW error displaces it in proportion to range. Splitting by range")
print("separates them -- the two call for different repairs.\n")
def run(yaw, lat, tag):
    P=dict(RR.NOMINAL); P.update(yaw=np.deg2rad(yaw), height=1.70, fx=1438.0, lateral=lat)
    P["pitch"]=LS.pitch_for_horizon(P,465.0)
    slabs=[(8.,12.),(12.,16.),(16.,22.)]; out=[]
    for xw in slabs:
        M=[]
        for pi in np.linspace(0,len(usable)-1,140).astype(int):
            f=int(usable[pi]["frame"]); im=cv2.imread(str(fdir/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
            if im is None: continue
            H=im.shape[0]; u,v=LC._ridge_points(im,int(0.55*H),int(0.90*H))
            if len(u)<100: continue
            ys,pr=LW.lateral_profile(np.stack([u,v],1).astype(float),P,x_win=xw,cell=0.03)
            if ys is None: continue
            pk=LW.peaks_with_fwhm(ys,pr,min_rel=0.25)
            L=[d["y_m"] for d in pk if d["y_m"]<0]; R=[d["y_m"] for d in pk if d["y_m"]>0]
            if not L or not R: continue
            l,r=max(L),min(R)
            if 1.5<=r-l<=5.5: M.append(0.5*(l+r))
        out.append((np.median(M) if len(M)>10 else np.nan, len(M)))
    mids=[o[0] for o in out]
    xs=np.array([10.,14.,19.]); ok=np.isfinite(mids)
    sl=np.polyfit(xs[ok],np.array(mids)[ok],1)[0] if ok.sum()>=2 else np.nan
    print(f"{tag}")
    print("   "+"   ".join(f"x {a:.0f}-{b:.0f} m: {m:+.3f} m (n{n:3d})" for (a,b),(m,n) in zip(slabs,out)))
    print(f"   slope {sl:+.4f} m per m  =>  residual yaw {np.rad2deg(np.arctan(sl)):+.2f} deg,"
          f"  offset at x=0 {np.array(mids)[ok][0]-sl*xs[ok][0]:+.3f} m\n")
    return sl
run(-7.01,-0.12,"RENDERED (yaw -7.01, lateral -0.12)")
run(-7.01,-0.31,"lateral corrected to -0.31")
