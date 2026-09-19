import sys, numpy as np, cv2, json, importlib.util
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import bev_calib as BC, lag_scale as LS, run_real as RR
spec=importlib.util.spec_from_file_location("lr","probes/lane_residual.py")
lr=importlib.util.module_from_spec(spec); spec.loader.exec_module(lr)

FH=2444.6; H0=1.427; FX=1713.0
recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
keep=[]
for r in usable:
    t=np.asarray(r["t"],float); yw=np.asarray(r["yaw"],float); m=np.abs(t)<0.4
    if m.sum()<3: continue
    if abs(np.rad2deg(np.polyfit(t[m],yw[m],1)[0]))<=0.5: keep.append(r)
ranges=[8.,10.,12.,15.,20.]
HZ=np.arange(430.,505.,3.0)
def P_at(hz):
    P=dict(RR.NOMINAL); P.update(fx=FX, height=H0, lateral=-0.126, yaw=np.deg2rad(-6.80))
    P["pitch"]=LS.pitch_for_horizon(P,float(hz)); return P
PS={float(h):P_at(h) for h in HZ}

best_hz, flat_before, flat_after, frames = [], [], [], []
for pi in np.linspace(0,len(keep)-1,160).astype(int):
    f=int(keep[pi]["frame"]); img=cv2.imread(str(fdir/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    got=[]
    for hz in HZ:
        d=[q for q in lr.residuals_one(img,PS[float(hz)],ranges,1.75)
           if q["side"]=="L" and q["resid_m"] is not None]
        if len(d)<4: got.append(None); continue
        x=np.array([q["x"] for q in d]); y=np.array([q["resid_m"] for q in d])
        s,b=np.polyfit(x,y,1); got.append((abs(s), s, hz))
    ok=[g for g in got if g is not None]
    if len(ok)<8: continue
    i465=int(np.argmin(np.abs(HZ-465.0)))
    if got[i465] is None: continue
    flat_before.append(got[i465][1])
    m=min(ok, key=lambda g:g[0])
    best_hz.append(m[2]); flat_after.append(m[1]); frames.append(f)
best_hz=np.array(best_hz); fb=np.array(flat_before); fa=np.array(flat_after)
print(f"{len(best_hz)} straight frames, each given its OWN horizon row\n")
print(f"  residual slope with a FIXED horizon 465:  sd {np.std(fb):.4f} m/m  "
      f"(p10 {np.percentile(fb,10):+.4f}, p90 {np.percentile(fb,90):+.4f})")
print(f"  residual slope with a PER-FRAME horizon:  sd {np.std(fa):.4f} m/m  "
      f"(p10 {np.percentile(fa,10):+.4f}, p90 {np.percentile(fa,90):+.4f})")
print(f"  => per-frame horizon removes {100*(1-np.std(fa)/max(np.std(fb),1e-9)):.0f}% of the spread\n")
print(f"  best-fit horizon per frame: median {np.median(best_hz):.1f} px   "
      f"p10 {np.percentile(best_hz,10):.1f}   p90 {np.percentile(best_hz,90):.1f}   sd {np.std(best_hz):.1f}")
print(f"  p10-p90 spread = {np.percentile(best_hz,90)-np.percentile(best_hz,10):.1f} px "
      f"= {np.rad2deg(np.arctan((np.percentile(best_hz,90)-np.percentile(best_hz,10))/FX)):.2f} deg of pitch")
print(f"\n  a {np.percentile(best_hz,90)-np.percentile(best_hz,10):.0f} px horizon swing changes the range of a point at 20 m by "
      f"{FH/((465-np.percentile(best_hz,10))+FH/20.) - FH/((465-np.percentile(best_hz,90))+FH/20.):+.1f} m")
json.dump(dict(n=len(best_hz), hz_median=float(np.median(best_hz)),
               hz_p10=float(np.percentile(best_hz,10)), hz_p90=float(np.percentile(best_hz,90)),
               hz_sd=float(np.std(best_hz)), slope_sd_fixed=float(np.std(fb)),
               slope_sd_perframe=float(np.std(fa)), frames=frames,
               best_hz=best_hz.tolist()), open("raw/per_frame_horizon.json","w"), indent=2)
