import sys, numpy as np
sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_mapsel/code')
import mapsel_data as D
z=np.load(D.V9_NPZ,allow_pickle=True); uu,_=D.clip_uuids(); pos={str(s):i for i,s in enumerate(z['clip__sha12'])}
idx=D.decisions_index(0); by={}
for k,(f,r) in idx.items(): by.setdefault(k[0],[]).append((k[1],f,r))
res={}
for c in sorted(by)[::7]:
    i=pos[D.sha12(uu[c])]; row0=int(z['clip__row0'][i]); k0=int(z['row__k'][row0]); n=int(z['clip__n_rows'][i])
    best=None
    for off in range(0,5):
        tot=ok=0
        for w,f,r in by[c]:
            d=np.load(f); 
            if not bool(d['r8_nav_known'][r]): continue
            m=w+off-k0
            if not 0<=m<n: continue
            dn=z['row__nav_d_next_m'][row0+m]
            if not np.isfinite(dn): continue
            tot+=1; ok+=int(abs(np.log1p(dn/50)-d['r8_nav_fed'][r][0])<1e-3)
        if tot and (best is None or ok/tot>best[1]): best=(off,ok/tot,tot)
    res[c]=best
print(res)
