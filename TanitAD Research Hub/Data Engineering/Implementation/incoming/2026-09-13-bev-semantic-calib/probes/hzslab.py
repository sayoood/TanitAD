import sys, numpy as np, cv2
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import lane_width as LW, lag_scale as LS, run_real as RR
from trajlib import lane_calib as LC

recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
picks=np.linspace(0,len(usable)-1,120).astype(int)
imgs=[]
for pi in picks:
    f=int(usable[pi]["frame"]); im=cv2.imread(str(fdir/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    H=im.shape[0]; u,v=LC._ridge_points(im,int(0.55*H),int(0.95*H))
    if len(u)>=100: imgs.append((f,np.stack([u,v],1).astype(float)))
print(f"{len(imgs)} frames of ridge ink\n")
print("A correct horizon makes the reconstructed lane width the SAME at every range.")
print("A wrong one tilts the plane, so near and far disagree.\n")
print(f"{'horizon':>8}  {'8-13 m':>16}  {'13-18 m':>16}  {'18-25 m':>16}   far/near")
slabs=[(8.,13.),(13.,18.),(18.,25.)]
rows=[]
for hz in (430.,448.4,464.4,485.,500.,523.4,540.):
    P=dict(RR.NOMINAL); P.update(yaw=np.deg2rad(-7.01), height=1.17, fx=2444.4/1.17)
    P["pitch"]=LS.pitch_for_horizon(P,hz)
    res=[]
    for xw in slabs:
        W=[]
        for f,uv in imgs:
            ys,pr=LW.lateral_profile(uv,P,x_win=xw,cell=0.03)
            if ys is None: continue
            pk=LW.peaks_with_fwhm(ys,pr)
            for i in range(len(pk)-1):
                w=pk[i+1]["y_m"]-pk[i]["y_m"]
                if 1.2<=w<=6.5: W.append(w)
        res.append((np.median(W) if len(W)>15 else np.nan, len(W)))
    r=[x[0] for x in res]
    ratio=r[2]/r[0] if np.isfinite(r[0]) and np.isfinite(r[2]) and r[0]>0 else np.nan
    rows.append((hz,r,ratio))
    print(f"{hz:8.1f}  "+"  ".join(f"{v:6.3f} m (n{n:4d})" for v,n in res)+f"   {ratio:7.3f}")
good=[(abs(x[2]-1.0),x[0]) for x in rows if np.isfinite(x[2])]
good.sort()
print(f"\n=> most range-CONSISTENT horizon: {good[0][1]:.1f} px  (far/near = {1+good[0][0] if True else 0:.3f})")
for d,h in good: print(f"     horizon {h:6.1f}  |far/near - 1| = {d:.3f}")
