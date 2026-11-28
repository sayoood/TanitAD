import sys
sys.path.insert(0,'F:/TanitAD-scratch/motion_c/lead_build/new/stack')
import numpy as np, time
from tanitad.data import v8f_mini_data as D
z=np.load('E:/Projects/TanitAD-artifacts/v9labels/inputs/lead_train.npz',allow_pickle=True)
sh=np.unique(z['sha12'])
cl=['clipA','clipB','clipC']
mp={D.sha12(c):s for c,s in zip(cl,sh[:3])}
orig=D.sha12
D.sha12=lambda c: mp.get(orig(c),orig(c))
J={'clip':np.array(cl),'off':np.array([0,110,220]),'n':np.array([110,110,50])}
t=time.time();r=D.lead_label_arrays(J,z);print('sec',time.time()-t)
print({k:(v if np.isscalar(v) else v.shape) for k,v in r.items()})
for ci,s in enumerate(sh[:3]):
    m=z['sha12']==s
    k=z['k'][m];ld=z['lead'][m];gp=z['gap'][m]
    d={int(a):(int(b),float(c)) for a,b,c in zip(k,ld,gp)}
    for j in range(J['n'][ci]):
        row=J['off'][ci]+j
        e=d.get(2*j)
        if e is None: assert not r['has'][row] and not r['near'][row]
        else:
            assert r['has'][row]
            nr = e[0]==1 and e[1]<=40
            assert r['near'][row]==nr,(ci,j,e)
            if nr: assert abs(r['y'][row]-min(max(min(e[1],40)/40,0),1))<1e-6
            else: assert r['y'][row]==0
print('cross-check ok', r['n_has'], r['n_near'])
