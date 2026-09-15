import sys, numpy as np, cv2, json, importlib.util
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import bev_calib as BC, lag_scale as LS, run_real as RR
spec=importlib.util.spec_from_file_location("lr","probes/lane_residual.py")
lr=importlib.util.module_from_spec(spec); spec.loader.exec_module(lr)

# The deployment test: does a SMOOTHED flow-based per-frame horizon beat the fixed
# one, judged by an instrument it shares nothing with (the lane residual)?
d=json.load(open('raw/horizon_track.json'))
hf=np.array(d['frames'],float); hh=np.array(d['horizon'],float)
order=np.argsort(hf); hf,hh=hf[order],hh[order]
def smooth(k):
    if k<3: return hh.copy()
    if k%2==0: k+=1
    return np.convolve(hh, np.ones(k)/k, mode='same')
FX=1713.0; H0=1.427
recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
keep=[]
for r in usable:
    t=np.asarray(r["t"],float); yw=np.asarray(r["yaw"],float); m=np.abs(t)<0.4
    if m.sum()<3: continue
    if abs(np.rad2deg(np.polyfit(t[m],yw[m],1)[0]))<=0.5: keep.append(int(r["frame"]))
ranges=[8.,10.,12.,15.,20.]
cache={}
def P_at(hz):
    hz=round(float(hz),1)
    if hz not in cache:
        P=dict(RR.NOMINAL); P.update(fx=FX, height=H0, lateral=-0.126, yaw=np.deg2rad(-6.80))
        P["pitch"]=LS.pitch_for_horizon(P,hz); cache[hz]=P
    return cache[hz]
def slope_at(img,hz):
    d2=[q for q in lr.residuals_one(img,P_at(hz),ranges,1.75)
        if q["side"]=="L" and q["resid_m"] is not None]
    if len(d2)<4: return None
    x=np.array([q["x"] for q in d2]); y=np.array([q["resid_m"] for q in d2])
    s,_=np.polyfit(x,y,1)
    return None if abs(s)>0.15 else s
sel=[f for f in keep if hf.min()<=f<=hf.max()]
sel=[sel[i] for i in np.linspace(0,len(sel)-1,150).astype(int)]
SM={ "raw (no smoothing)":smooth(1), "0.9 s":smooth(3), "2.1 s":smooth(7), "3.9 s":smooth(13) }
res={k:[] for k in SM}; fixed=[]
imgs={}
for f in sel:
    img=cv2.imread(str(fdir/f"{f:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    s0=slope_at(img,465.0)
    if s0 is None: continue
    ok=True; tmp={}
    for k,series in SM.items():
        hz=float(np.interp(f,hf,series))
        s=slope_at(img,hz)
        if s is None: ok=False; break
        tmp[k]=s
    if not ok: continue
    fixed.append(s0)
    for k,v in tmp.items(): res[k].append(v)
fixed=np.array(fixed)
print(f"{len(fixed)} straight frames scored by the LANE residual (an instrument the")
print(f"horizon tracker shares no features or model with)\n")
print(f"  {'horizon source':>22}  {'|slope| median':>15}  {'slope sd':>9}")
print(f"  {'fixed 465 px':>22}  {np.median(np.abs(fixed)):14.4f}  {np.std(fixed):9.4f}")
best=None
for k in SM:
    v=np.array(res[k])
    print(f"  {'flow track, '+k:>22}  {np.median(np.abs(v)):14.4f}  {np.std(v):9.4f}")
    if best is None or np.median(np.abs(v))<best[1]: best=(k,np.median(np.abs(v)),np.std(v))
print(f"\n  best: flow track at {best[0]}  ->  |slope| median "
      f"{np.median(np.abs(fixed)):.4f} -> {best[1]:.4f} "
      f"({100*(1-best[1]/np.median(np.abs(fixed))):+.0f}%)")
if best[1] < 0.85*np.median(np.abs(fixed)):
    print("  => DEPLOY IT. A smoothed per-frame horizon measurably beats the constant.")
else:
    print("  => DO NOT DEPLOY YET. The tracker does not beat a constant on this evidence;")
    print("     its noise (sd ~15 px) is comparable to the signal it is chasing (~13 px).")
