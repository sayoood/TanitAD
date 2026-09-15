import sys, numpy as np, cv2
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import lane_width as LW, lag_scale as LS, run_real as RR
from trajlib import lane_calib as LC

recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
P=dict(RR.NOMINAL); P.update(yaw=np.deg2rad(-7.01), height=1.70, fx=1438.0, lateral=-0.12)
P["pitch"]=LS.pitch_for_horizon(P,465.0)
print("At the RENDERED calibration: f 1438, h 1.70, horizon 465, yaw -7.01, lateral -0.12\n")
print("If the ego were centred in its lane, the two bounding lines would sit")
print("symmetrically about y = 0. Their MIDPOINT is the lateral error.\n")
mids, lefts, rights, fr = [], [], [], []
for pi in np.linspace(0,len(usable)-1,140).astype(int):
    f=int(usable[pi]["frame"]); im=cv2.imread(str(fdir/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    H=im.shape[0]; u,v=LC._ridge_points(im,int(0.55*H),int(0.90*H))
    if len(u)<100: continue
    ys,pr=LW.lateral_profile(np.stack([u,v],1).astype(float),P,x_win=(8.,22.),cell=0.03)
    if ys is None: continue
    pk=LW.peaks_with_fwhm(ys,pr,min_rel=0.25)
    L=[d["y_m"] for d in pk if d["y_m"]<0]; R=[d["y_m"] for d in pk if d["y_m"]>0]
    if not L or not R: continue
    l,r=max(L),min(R)          # the two lines bounding the ego lane
    if not (1.5<=r-l<=5.5): continue
    lefts.append(l); rights.append(r); mids.append(0.5*(l+r)); fr.append(f)
mids=np.array(mids)
print(f"n = {len(mids)} frames with both bounding lines found")
print(f"   left line   median {np.median(lefts):+.3f} m")
print(f"   right line  median {np.median(rights):+.3f} m")
print(f"   lane width  median {np.median(np.array(rights)-np.array(lefts)):.3f} m")
ci=LW.cluster_bootstrap(mids.tolist(), fr)
print(f"   MIDPOINT    median {np.median(mids):+.3f} m" + (f"   95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""))
print(f"\n   => the corridor is offset by {np.median(mids):+.3f} m from the lane centre.")
print(f"      Correcting it means lateral_offset {-0.12:+.2f} -> {-0.12+np.median(mids):+.3f} m")
print(f"      (a lateral offset shifts the reconstruction one-for-one)")
