import sys, numpy as np, cv2, pathlib
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib/probes")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import bev_calib as BC, lag_scale as LS, run_real as RR
import importlib.util
spec=importlib.util.spec_from_file_location("lr","probes/lane_residual.py"); lr=importlib.util.module_from_spec(spec); spec.loader.exec_module(lr)

recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
keep=[]
for r in usable:
    t=np.asarray(r["t"],float); yw=np.asarray(r["yaw"],float); m=np.abs(t)<0.35
    if m.sum()<3: continue
    if abs(np.rad2deg(np.polyfit(t[m],yw[m],1)[0]))<=0.6: keep.append(r)
imgs=[]
for pi in np.linspace(0,len(keep)-1,130).astype(int):
    f=int(keep[pi]["frame"]); im=cv2.imread(str(fdir/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    if im is not None: imgs.append(im)
ranges=[8.,10.,12.,15.,20.]
print("LEFT edge only (the SOLID line — the dashed right one mis-associates past 10 m)")
print("A yaw error makes the residual grow with range; the right yaw makes it FLAT.\n")
print(f"{'yaw':>7}  " + "  ".join(f"{x:>6.0f}m" for x in ranges) + "     slope    |resid| mean")
best=None
for yawd in (-8.4,-7.9,-7.3,-6.8,-6.3,-5.8,-5.3):
    P=dict(RR.NOMINAL); P.update(height=1.427, fx=1713.0, lateral=-0.126, yaw=np.deg2rad(yawd))
    P["pitch"]=LS.pitch_for_horizon(P,465.0)
    acc={}
    for im in imgs:
        for d in lr.residuals_one(im,P,ranges,1.75):
            if d["side"]=="L": acc.setdefault(d["x"],[]).append(d["resid_m"])
    med=[]
    for x in ranges:
        v=[q for q in acc.get(x,[]) if q is not None]
        med.append(np.median(v) if len(v)>=8 else np.nan)
    med=np.array(med); ok=np.isfinite(med)
    if ok.sum()<3: print(f"{yawd:7.2f}   too few"); continue
    sl=np.polyfit(np.array(ranges)[ok],med[ok],1)[0]
    score=float(np.mean(np.abs(med[ok])))
    if best is None or abs(sl)<abs(best[1]): best=(yawd,sl,score)
    print(f"{yawd:7.2f}  " + "  ".join(f"{m:+7.3f}" if np.isfinite(m) else "    nan" for m in med)
          + f"   {sl:+.4f}   {score:.3f}")
print(f"\n=> flattest at yaw {best[0]:+.2f} deg (slope {best[1]:+.4f} m/m)")
