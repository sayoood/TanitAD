"""v12 = cowl (fast, lag-free) + low-passed correction from predict-vs-outcome (absolute level).
Fit on HALF the prediction frames, evaluate on the OTHER half with a fresh predict-vs-outcome run."""
import sys, json, numpy as np
S="/tmp/claude-0/-home-user-TanitAD/d367c501-690c-51f0-8672-b9ccb83dd3ca/scratchpad"
P="/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib/probes"
sys.path.insert(0,P)
import predict_vs_outcome as PV, run_real as RR
recs={int(r["frame"]):r for r in RR.load_records(RR.RUN)}
have={int(p.stem) for p in (RR.RUN/"frames").glob("*.jpg")}
frames=[f for f in sorted(recs) if f in have and recs[f]["complete"] and recs[f]["speed_ms"]>8]
T=json.load(open(P+"/../attitude/attitude_track_v11_cowl.json"))
tt=np.array(T["t_session_s"]); yc=np.array(T["yaw_deg"]); hc=np.array(T["horizon_row"])
A=np.load(S+"/pvo_cowl.npy")                   # f, t, X, pred, actual, err_m (cowl yaw used)
pf=sorted(set(A[:,0].astype(int)))
fitF=set(pf[0::2]); testF=set(pf[1::2])
fit=A[np.isin(A[:,0].astype(int),list(fitF)) & (A[:,2]<=30)]
# required extra yaw at F so that the prediction lands on the outcome: +yaw moves the far end RIGHT
dyaw=np.degrees(-fit[:,5]/(fit[:,2]-PV.X_NEAR))
o=np.argsort(fit[:,1]); ft=fit[o,1]; fd=dyaw[o]
def smooth(tq,ts,vs,half=1.0,max_half=4.0,need=4):
    out=[]
    for x in tq:
        h=half
        while True:
            m=np.abs(ts-x)<=h
            if m.sum()>=need or h>=max_half: break
            h+=0.5
        out.append(np.median(vs[m]) if m.sum()>=need else np.nan)
    return np.array(out)
corr=smooth(tt,ft,fd)
# outside the validated span the correction fades to the global median, never extrapolates a local value
g=float(np.median(fd)); ok=np.isfinite(corr)
corr=np.where(ok,corr,g)
k=np.exp(-0.5*(np.arange(-15,16)/7.5)**2); k/=k.sum()           # ~0.5 s gaussian at 30 fps
corr=np.convolve(np.pad(corr,15,mode="edge"),k,"valid")
y12=yc+corr
print(f"fit pairs {len(fit)} from {len(fitF)} frames; correction median {g:+.2f} deg, "
      f"range {corr.min():+.2f}..{corr.max():+.2f}; yaw v12 {y12.min():+.2f}..{y12.max():+.2f}")
print(f"  at t=34.43: cowl {np.interp(34.43,tt,yc):+.2f} + correction {np.interp(34.43,tt,corr):+.2f} = {np.interp(34.43,tt,y12):+.2f}")
json.dump(dict(t_session_s=T["t_session_s"], yaw_deg=[round(float(x),4) for x in y12],
               horizon_row=T["horizon_row"],
               source=("v11 cowl track (fast) + low-passed predict-vs-outcome yaw correction "
                       "(running median +-1 s widened to >=4 pairs, max +-4 s, then 0.5 s gaussian; "
                       "fitted on half the prediction frames; outside the validated span the "
                       "correction is the global median). 2026-09-24")),
          open(S+"/attitude_track_v12.json","w"))
# ---- HELD-OUT: fresh predict-vs-outcome on the TEST frames only, v11 vs v12 ----
yaw11=lambda t: float(np.interp(t,tt,yc)); yaw12=lambda t: float(np.interp(t,tt,y12))
hz=lambda t: float(np.interp(t,tt,hc))
def run(yawf):
    out=[]; cache={}
    t_all=np.array([recs[f]["t_session_s"] for f in frames]); spd=np.array([recs[f]["speed_ms"] for f in frames])
    s=np.concatenate([[0.0],np.cumsum(0.5*(spd[1:]+spd[:-1])*np.diff(t_all))]); idx={f:i for i,f in enumerate(frames)}
    for f in sorted(testF):
        i=idx[f]; Pf=PV.V.params(yawf(t_all[i]),-0.260,hz(t_all[i])); L=PV.lanes(f,Pf,recs,cache)
        if L is None: continue
        for X in (20.,30.):
            p=PV.ribbon_frac(f,X,Pf,recs,L)
            if p is None: continue
            j=int(np.searchsorted(s,s[i]+X-PV.X_NEAR))
            if j>=len(frames): continue
            f2=frames[j]; P2=PV.V.params(yawf(t_all[j]),-0.260,hz(t_all[j])); L2=PV.lanes(f2,P2,recs,{})
            if L2 is None: continue
            q=PV.ribbon_frac(f2,PV.X_NEAR,P2,recs,L2)
            if q is None: continue
            out.append((t_all[i],X,(p-q)*PV.LANE_M))
    return np.array(out)
for nm,yf in (("v11 cowl only",yaw11),("v12 cowl + outcome-level",yaw12)):
    R=run(yf)
    print(f"\n HELD-OUT {nm}: {len(R)} pairs")
    for X in (20.,30.):
        a=R[R[:,1]==X]; b=[]
        for t0 in range(0,60,10):
            m=(a[:,0]>=t0)&(a[:,0]<t0+10); b.append(f"{np.median(a[m,2]):+.2f}" if m.sum()>=3 else "  n/a")
        w=(a[:,0]>=33.5)&(a[:,0]<=35.5)
        print(f"   X={X:.0f} m bins " + " ".join(b) + f"   |  33.5-35.5 s median {np.median(a[w,2]):+.2f} (n {w.sum()})"
              f"   all: median {np.median(a[:,2]):+.2f} sd {1.4826*np.median(np.abs(a[:,2]-np.median(a[:,2]))):.2f}")
