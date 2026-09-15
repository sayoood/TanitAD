import sys, numpy as np, cv2, os, json
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import flow_scale as FS, run_real as RR
cache="/tmp/claude-0/-home-user-TanitAD/d367c501-690c-51f0-8672-b9ccb83dd3ca/scratchpad/pairdata.npz"
if os.path.exists(cache):
    z=np.load(cache); V,DV,DD,PID,FR=z["V"],z["DV"],z["DD"],z["PID"],z["FR"]
else:
    recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
    usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
    V,DV,DD,PID,FR=[],[],[],[],[]; pid=0
    for pi in np.linspace(0,len(usable)-1,110).astype(int):
        a=usable[pi]; f0=int(a["frame"])
        i0=cv2.imread(str(fdir/f"{f0:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
        if i0 is None: continue
        m=FS.road_mask(i0,row_band=(0.55,0.77))
        for step in (1,2):
            p1=fdir/f"{f0+step:06d}.jpg"
            if not p1.exists(): continue
            i1=cv2.imread(str(p1),cv2.IMREAD_GRAYSCALE)
            uv0,uv1=FS.track_pair(i0,i1,n_pts=700,mask=m)
            if len(uv0)<60: continue
            t=np.asarray(a["t"],float); dt=step/29.922
            D=float(np.interp(dt,t,np.asarray(a["x"],float)))
            V.append(uv0[:,1]); DV.append(uv1[:,1]-uv0[:,1]); DD.append(np.full(len(uv0),D))
            PID.append(np.full(len(uv0),pid)); FR.append(np.full(len(uv0),f0)); pid+=1
    V=np.concatenate(V);DV=np.concatenate(DV);DD=np.concatenate(DD);PID=np.concatenate(PID);FR=np.concatenate(FR)
    np.savez(cache,V=V,DV=DV,DD=DD,PID=PID,FR=FR)
print(f"{len(V)} points over {len(set(PID.tolist()))} frame pairs, {len(set(FR.tolist()))} frames")

def model(q,D,A): 
    den=A-D*q
    return np.where(den>1e-6, D*q*q/np.maximum(den,1e-6), 1e6)

def cost(A,vh,demean,idx,V,DV,DD,PID,c=2.0):
    q=V-vh
    ok=q>25
    m=model(q,DD,A)
    r=DV-m
    if demean:
        # eliminate a per-pair constant row offset (pitch jitter / EIS)
        s=np.bincount(idx,weights=np.where(ok,r,0.0)); n=np.bincount(idx,weights=ok.astype(float))
        r=r-np.where(n[idx]>0, s[idx]/np.maximum(n[idx],1), 0.0)
    r=r[ok]
    return float(np.mean(np.log1p((r/c)**2))), int(ok.sum())

idx=np.searchsorted(np.array(sorted(set(PID.tolist()))), PID)
from scipy.optimize import minimize
for demean in (False, True):
    best=None
    for A0 in (1200.,1700.,2400.,3200.):
        for v0 in (420.,470.,520.):
            r=minimize(lambda x: cost(x[0]*1000, x[1]*100, demean, idx,V,DV,DD,PID)[0],
                       [A0/1000, v0/100], method="Nelder-Mead",
                       options=dict(maxfev=600,xatol=1e-5,fatol=1e-9))
            if best is None or r.fun<best.fun: best=r
    A,vh=best.x[0]*1000, best.x[1]*100
    c,n=cost(A,vh,demean,idx,V,DV,DD,PID)
    tag="WITH per-pair offset removed (pitch jitter / EIS eliminated)" if demean else "NO per-pair offset (assumes zero pitch rotation between frames)"
    print(f"\n{tag}")
    print(f"   horizon {vh:7.2f} px     f*h {A:8.1f} px*m     cost {c:.5f}   n {n}")
    for h in (1.03,1.17,1.30,1.45,1.60,1.75):
        f=A/h
        print(f"      h {h:4.2f} -> f {f:7.1f} px  HFOV {np.rad2deg(2*np.arctan(1920/(2*f))):5.1f} deg"
              f"{'  <== inside 1356-1628' if 1356<=f<=1628 else ''}")
    # how big is the per-pair offset it removed?
    if demean:
        q=V-vh; ok=q>25; r=DV-model(q,DD,A)
        s=np.bincount(idx,weights=np.where(ok,r,0.0)); nn=np.bincount(idx,weights=ok.astype(float))
        off=s/np.maximum(nn,1)
        off=off[nn>20]
        print(f"   per-pair row offset: median {np.median(off):+.2f} px, "
              f"p16-p84 [{np.percentile(off,16):+.2f}, {np.percentile(off,84):+.2f}], "
              f"|median| over pairs {np.median(np.abs(off)):.2f} px")
