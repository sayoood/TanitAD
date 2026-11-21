import sys, numpy as np
sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_mapsel/code')
import mapsel_data as D
z=np.load('E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz',allow_pickle=True)
uu,_=D.clip_uuids(); import hashlib; sha=[hashlib.sha256(u.encode()).hexdigest()[:12] for u in uu]
cs=list(z['clip__sha12']); print('sha12 order equal to sorted uuid order:', cs==sha, 'set equal', set(cs)==set(sha))
print('row0', z['clip__row0'][:3], 'k0', z['clip__k0'][:3], 'n_rows', z['clip__n_rows'][:3])
ci=0; r0=int(z['clip__row0'][cs.index(sha[ci])]); print('row__k first', z['row__k'][r0:r0+4])
idx=D.decisions_index(0)
import glob
# parity: decisions r8_nav_fed[:,0] vs log1p(nav_d_next_m/50)
best={}
rows=[(k,v) for k,v in idx.items() if k[0] in (0,5,50,100)]
for off in range(-2,6):
    tot=0; ok=0
    for (c,w),(f,r) in rows:
        d=np.load(f); fed=d['r8_nav_fed'][r]; kn=bool(d['r8_nav_known'][r])
        if not kn: continue
        i=cs.index(sha[c]); row0=int(z['clip__row0'][i]); k0=int(z['row__k'][row0])
        m=(w+off)-k0
        if not (0<=m<int(z['clip__n_rows'][i])): continue
        dn=z['row__nav_d_next_m'][row0+m]
        if not np.isfinite(dn): continue
        tot+=1; ok+= int(abs(np.log1p(dn/50.0)-fed[0])<1e-3)
    best[off]=(ok,tot)
print(best)
