import sys, numpy as np, cv2, json, importlib.util
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import bev_calib as BC, lag_scale as LS, run_real as RR
spec=importlib.util.spec_from_file_location("lr","probes/lane_residual.py")
lr=importlib.util.module_from_spec(spec); spec.loader.exec_module(lr)

P=dict(RR.NOMINAL); P.update(fx=1713.0, height=1.427, lateral=-0.126, yaw=np.deg2rad(-6.80))
P["pitch"]=LS.pitch_for_horizon(P,465.0)
recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
keep=[]
for r in usable:
    t=np.asarray(r["t"],float); yw=np.asarray(r["yaw"],float); m=np.abs(t)<0.4
    if m.sum()<3: continue
    if abs(np.rad2deg(np.polyfit(t[m],yw[m],1)[0]))<=0.5: keep.append(r)
ranges=[8.,10.,12.,15.,20.]
slopes, offs, frames = [], [], []
for pi in np.linspace(0,len(keep)-1,300).astype(int):
    f=int(keep[pi]["frame"]); img=cv2.imread(str(fdir/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    d=[x for x in lr.residuals_one(img,P,ranges,1.75) if x["side"]=="L" and x["resid_m"] is not None]
    if len(d)<4: continue
    x=np.array([q["x"] for q in d]); y=np.array([q["resid_m"] for q in d])
    a,b=np.polyfit(x,y,1)
    if abs(a)>0.15: continue                       # a lane change or a bad association
    slopes.append(a); offs.append(b+a*12.0); frames.append(f)   # offset evaluated at 12 m
slopes=np.array(slopes); offs=np.array(offs)
print(f"{len(slopes)} straight frames with a usable LEFT-line residual profile\n")
print("If ONE calibration fitted every frame, the per-frame slope would be a tight")
print("cluster at 0. Its spread is the per-frame attitude error, in disguise.\n")
yawerr=np.rad2deg(np.arctan(slopes))
for name,v,u in (("residual slope",slopes,"m/m"),("=> implied yaw error",yawerr,"deg"),
                 ("lateral at 12 m",offs,"m")):
    print(f"  {name:22s} median {np.median(v):+7.3f} {u:5s}  p10 {np.percentile(v,10):+7.3f}"
          f"  p90 {np.percentile(v,90):+7.3f}   sd {np.std(v):6.3f}")
sp=np.percentile(yawerr,90)-np.percentile(yawerr,10)
print(f"\n  per-frame yaw error p10-p90 spread = {sp:.2f} deg")
print(f"  at 20 m that is {2*20*np.tan(np.deg2rad(sp/2)):.2f} m of lateral swing between frames.")
print("\n  => " + ("NO single yaw fits every frame; the camera-to-road attitude moves."
      if sp>0.8 else "a single yaw is adequate; the frame-to-frame variation is small."))
# how much of the lateral offset spread is the DRIVER wandering vs calibration?
print(f"\n  lateral-at-12 m p10-p90 spread = {np.percentile(offs,90)-np.percentile(offs,10):.2f} m")
print("     (this one is EXPECTED to vary — it is where the driver put the car in the lane)")
json.dump(dict(n=len(slopes), slope_sd=float(np.std(slopes)),
               yaw_p10=float(np.percentile(yawerr,10)), yaw_p90=float(np.percentile(yawerr,90)),
               yaw_sd=float(np.std(yawerr)), off_sd=float(np.std(offs)),
               frames=frames, slopes=slopes.tolist(), offs=offs.tolist()),
          open("raw/per_frame_attitude.json","w"), indent=2)
