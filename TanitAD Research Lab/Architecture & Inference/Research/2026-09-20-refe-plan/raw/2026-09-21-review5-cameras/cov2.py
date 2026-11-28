import json, math, itertools
import numpy as np
d=json.load(open('D:/Projects/TanitAD/data/nuplan_cam_calib.json'))
def q2m(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
yaw={ch: math.degrees(math.atan2(*( (q2m(d[ch]['q'])@np.array([0.,0.,1.]))[[1,0]] ))) for ch in d}
grid=np.arange(-180,180,0.05)
def report(hfov,label):
    def cov(chs):
        m=np.zeros_like(grid,bool)
        for ch in chs:
            dd=(grid-yaw[ch]+180)%360-180
            m|=np.abs(dd)<=hfov/2
        return m
    res=[]
    for c in itertools.combinations(sorted(d),4):
        m=cov(c); un=~m
        ext=np.concatenate([un,un]); mx=run=0
        for v in ext:
            run=run+1 if v else 0; mx=max(mx,run)
        res.append((m.mean(), min(mx,len(grid))*0.05, c))
    res.sort(key=lambda r:(-r[0],r[1]))
    ours=('CAM_B0','CAM_F0','CAM_L0','CAM_R0')
    rk=[i for i,(a,b,c) in enumerate(res,1) if set(c)==set(ours)][0]
    fo,go,_=[r for r in res if set(r[2])==set(ours)][0]
    print(f"\n### {label}  (HFOV {hfov:.2f} deg)")
    print(f"  {'rank':>4s} {'coverage':>9s} {'max gap':>8s}  cameras")
    for i,(f,g,c) in enumerate(res[:5],1):
        print(f"  {i:4d} {100*f:8.2f}% {g:7.1f}d  {', '.join(f'{ch[4:]}({yaw[ch]:+.0f})' for ch in c)}")
    print(f"  OURS F0/L0/R0/B0 : {100*fo:.2f}%  max gap {go:.1f} deg   RANK {rk} of {len(res)}")
    for alt in [('CAM_F0','CAM_L1','CAM_R1','CAM_B0'),('CAM_F0','CAM_L0','CAM_R0','CAM_L2')]:
        f2,g2,_=[r for r in res if set(r[2])==set(alt)][0]
        r2=[i for i,(a,b,c) in enumerate(res,1) if set(c)==set(alt)][0]
        print(f"  ALT  {'/'.join(x[4:] for x in alt):14s}: {100*f2:.2f}%  max gap {g2:.1f} deg  rank {r2}")
    m=cov(ours); un=~m; ivs=[];s=None
    for i,v in enumerate(un):
        if v and s is None: s=grid[i]
        if not v and s is not None: ivs.append((s,grid[i])); s=None
    if s is not None: ivs.append((s,grid[-1]))
    print("  OURS blind sectors:", ", ".join(f"[{a:+.0f},{b:+.0f}] ({b-a:.0f} deg)" for a,b in ivs))
    # all-8 coverage, for scale
    m8=cov(sorted(d)); print(f"  for scale, ALL EIGHT cameras: {100*m8.mean():.2f}%")
report(63.710,"A. ideal pinhole from the intrinsics (understates)")
report(72.249,"B. with the DB's own Caltech distortion -- the REAL sensor FOV")
