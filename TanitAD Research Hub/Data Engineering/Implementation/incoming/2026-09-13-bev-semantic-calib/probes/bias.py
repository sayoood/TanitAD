import sys, numpy as np, cv2, os
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import flow_scale as FS, run_real as RR

recs = RR.load_records(RR.RUN); fdir = RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
rng=np.random.default_rng(0)
grid=np.arange(330,561,5.0)

def fit(V,DV,DD):
    def st(vh):
        q=V-vh; qp=q+DV; ok=(q>25)&(DV>0.5)&(qp>q)
        if ok.sum()<150: return None
        a=DD[ok]*q[ok]*qp[ok]/(qp[ok]-q[ok]); qq=q[ok]
        med=np.median(a); k=(a>0.3*med)&(a<3*med)
        if k.sum()<80: return None
        aa,qq=a[k],qq[k]
        i=rng.integers(0,len(aa),5000); j=rng.integers(0,len(aa),5000); s=np.abs(qq[i]-qq[j])>20
        return float(np.median((aa[i][s]-aa[j][s])/(qq[i][s]-qq[j][s]))), float(np.median(aa))
    sl,md=[],[]
    for vh in grid:
        r=st(float(vh)); sl.append(r[0] if r else np.nan); md.append(r[1] if r else np.nan)
    sl=np.array(sl); md=np.array(md); g=np.isfinite(sl)
    if not (g.sum()>3 and sl[g].min()<0<sl[g].max()): return None
    z=float(np.interp(0.0,sl[g],grid[g])); r=st(z)
    return z, r[1]

per={s:{"V":[],"DV":[],"DD":[],"seed":0,"trk":0} for s in (1,2,3,5)}
for pi in np.linspace(0,len(usable)-1,70).astype(int):
    a=usable[pi]; f0=int(a["frame"])
    i0=cv2.imread(str(fdir/f"{f0:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    if i0 is None: continue
    m=FS.road_mask(i0,row_band=(0.55,0.77))
    seeds=cv2.goodFeaturesToTrack(i0,maxCorners=700,qualityLevel=0.005,minDistance=6,mask=m,blockSize=7)
    if seeds is None: continue
    for step in (1,2,3,5):
        p1=fdir/f"{f0+step:06d}.jpg"
        if not p1.exists(): continue
        i1=cv2.imread(str(p1),cv2.IMREAD_GRAYSCALE)
        q1,s1,_=cv2.calcOpticalFlowPyrLK(i0,i1,seeds,None,**FS.LK)
        q0,s0,_=cv2.calcOpticalFlowPyrLK(i1,i0,q1,None,**FS.LK)
        ok=(s1.ravel()==1)&(s0.ravel()==1)&(np.linalg.norm((q0-seeds).reshape(-1,2),axis=1)<0.6)
        uv0=seeds.reshape(-1,2)[ok]; uv1=q1.reshape(-1,2)[ok]
        t=np.asarray(a["t"],float); dt=step/29.922
        D=float(np.interp(dt,t,np.asarray(a["x"],float)))
        per[step]["V"].append(uv0[:,1]); per[step]["DV"].append(uv1[:,1]-uv0[:,1])
        per[step]["DD"].append(np.full(len(uv0),D))
        per[step]["seed"]+=len(seeds); per[step]["trk"]+=int(ok.sum())

print(" step   travel   track rate    horizon    f*h     <- if selection bias drove the fit,")
print("                                                     these must MOVE with step")
for step in (1,2,3,5):
    d=per[step]
    if not d["V"]: continue
    V=np.concatenate(d["V"]); DV=np.concatenate(d["DV"]); DD=np.concatenate(d["DD"])
    r=fit(V,DV,DD)
    tr=100*d["trk"]/max(d["seed"],1)
    if r is None: print(f" {step:4d}  {np.median(DD):6.2f} m   {tr:5.1f}%      not bracketed"); continue
    print(f" {step:4d}  {np.median(DD):6.2f} m   {tr:5.1f}%     {r[0]:7.1f} px  {r[1]:8.1f} px*m   (n {len(V)})")
