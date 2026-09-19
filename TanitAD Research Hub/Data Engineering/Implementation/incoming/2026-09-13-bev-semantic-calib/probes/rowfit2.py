import sys, numpy as np, cv2, json
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import flow_scale as FS, run_real as RR
cache = "/tmp/claude-0/-home-user-TanitAD/d367c501-690c-51f0-8672-b9ccb83dd3ca/scratchpad/rowdata.npz"
import os
if os.path.exists(cache):
    z=np.load(cache); V,DV,DD,FR=z["V"],z["DV"],z["DD"],z["FR"]
else:
    recs = RR.load_records(RR.RUN); fdir = RR.RUN/"frames"
    usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
    V,DV,DD,FR=[],[],[],[]
    for pi in np.linspace(0,len(usable)-1,90).astype(int):
        a=usable[pi]; f0=int(a["frame"])
        i0=cv2.imread(str(fdir/f"{f0:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
        for step in (1,2,3):
            p1=fdir/f"{f0+step:06d}.jpg"
            if i0 is None or not p1.exists(): continue
            i1=cv2.imread(str(p1),cv2.IMREAD_GRAYSCALE)
            m=FS.road_mask(i0,row_band=(0.55,0.77))
            uv0,uv1=FS.track_pair(i0,i1,n_pts=700,mask=m)
            if len(uv0)<20: continue
            t=np.asarray(a["t"],float); dt=step/29.922
            D=float(np.interp(dt,t,np.asarray(a["x"],float)))
            V.append(uv0[:,1]); DV.append(uv1[:,1]-uv0[:,1]); DD.append(np.full(len(uv0),D)); FR.append(np.full(len(uv0),f0))
    V=np.concatenate(V);DV=np.concatenate(DV);DD=np.concatenate(DD);FR=np.concatenate(FR)
    np.savez(cache,V=V,DV=DV,DD=DD,FR=FR)
print(f"{len(V)} points, {len(set(FR.tolist()))} frames, rows {V.min():.0f}-{V.max():.0f}")

rng=np.random.default_rng(0)
def stats(vh, Vv,DVv,DDv):
    q=Vv-vh; qp=q+DVv
    ok=(q>25)&(DVv>0.5)&(qp>q)
    if ok.sum()<200: return None
    a=DDv[ok]*q[ok]*qp[ok]/(qp[ok]-q[ok]); qq=q[ok]
    med=np.median(a); keep=(a>0.3*med)&(a<3*med)
    if keep.sum()<100: return None
    aa,qq=a[keep],qq[keep]
    i=rng.integers(0,len(aa),6000); j=rng.integers(0,len(aa),6000); s=np.abs(qq[i]-qq[j])>20
    sl=float(np.median((aa[i][s]-aa[j][s])/(qq[i][s]-qq[j][s])))
    return dict(sl=sl, med=float(np.median(aa)), n=int(keep.sum()),
                riqr=float((np.percentile(aa,75)-np.percentile(aa,25))/np.median(aa)))
print("\n v_h    n     median f*h    trend (px*m/row)   relative IQR")
grid=np.arange(330,561,10.0); rows=[]
for vh in grid:
    r=stats(float(vh),V,DV,DD)
    if r: rows.append((vh,r)); print(f" {vh:5.0f} {r['n']:7d} {r['med']:11.1f} {r['sl']:+15.3f} {r['riqr']:14.3f}")
sl=np.array([r['sl'] for _,r in rows]); vv=np.array([v for v,_ in rows])
zc=np.interp(0.0, sl, vv) if sl.min()<0<sl.max() else None
iq=np.array([r['riqr'] for _,r in rows]); vmin=vv[iq.argmin()]
print(f"\n zero-trend horizon: {('%.1f px'%zc) if zc else 'NOT BRACKETED in 330-560'}")
print(f" min-spread horizon: {vmin:.0f} px  (relative IQR {iq.min():.3f})")
if zc:
    r=stats(float(zc),V,DV,DD); print(f" at the zero-trend horizon: f*h = {r['med']:.1f} px*m  (n {r['n']}, relative IQR {r['riqr']:.3f})")
    # frame-cluster bootstrap on BOTH
    frames=np.array(sorted(set(FR.tolist()))); boot=[]
    for _ in range(300):
        pick=rng.choice(len(frames),len(frames),replace=True)
        m=np.concatenate([np.flatnonzero(FR==frames[p]) for p in pick])
        s2=[]; 
        for vh in grid:
            rr=stats(float(vh),V[m],DV[m],DD[m]); s2.append(rr['sl'] if rr else np.nan)
        s2=np.array(s2); good=np.isfinite(s2)
        if good.sum()>3 and s2[good].min()<0<s2[good].max():
            z=np.interp(0.0,s2[good],grid[good]); rr=stats(float(z),V[m],DV[m],DD[m])
            if rr: boot.append((z,rr['med']))
    if len(boot)>50:
        b=np.array(boot)
        print(f" frame-cluster bootstrap (n={len(boot)}):  horizon 95% CI "
              f"[{np.percentile(b[:,0],2.5):.1f}, {np.percentile(b[:,0],97.5):.1f}] px   "
              f"f*h 95% CI [{np.percentile(b[:,1],2.5):.0f}, {np.percentile(b[:,1],97.5):.0f}] px*m")
        json.dump(dict(horizon=float(zc), fh=float(r['med']),
                       horizon_ci=[float(np.percentile(b[:,0],2.5)),float(np.percentile(b[:,0],97.5))],
                       fh_ci=[float(np.percentile(b[:,1],2.5)),float(np.percentile(b[:,1],97.5))],
                       n_points=int(len(V)), n_frames=int(len(frames))),
                  open("raw/rowflow_fit.json","w"), indent=2)
