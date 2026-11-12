import json,lzma,glob,sys
import numpy as np
def load(paths,xz=False):
    o={}
    for p in paths:
        for l in (lzma.open(p,'rt') if xz else open(p)):
            r=json.loads(l); o[r['token']]=r
    return o
b=load(sorted(glob.glob('inputs/rollouts/*.xz')),True)
f=load(glob.glob('rolls/R0/rows_s*.jsonl'))
t=load(glob.glob(sys.argv[1]))
print('n',len(t),'ok',sum(1 for r in t.values() if 'traj20' in r))
dd=[];df=[]
for k,r in t.items():
    if 'traj20' not in r: continue
    dd.append(float(np.abs(np.array(r['traj20'])-np.array(b[k]['traj20'])).max()))
    df.append(float(np.abs(np.array(r['traj20'])-np.array(f[k]['traj20'])).max()))
dd=np.array(dd);df=np.array(df)
print('vs bank  >1e-6:',(dd>1e-6).sum(),'max',dd.max(),'zero',(dd==0).sum())
print('vs fresh(1thr) >1e-6:',(df>1e-6).sum(),'max',df.max())
