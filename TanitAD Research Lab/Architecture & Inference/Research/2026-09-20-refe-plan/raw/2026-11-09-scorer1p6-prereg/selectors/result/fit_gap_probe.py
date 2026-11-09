import json, sys, numpy as np
sys.path.insert(0,'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/selectors')
s=json.load(open('F:/tanitad_scratch/p6sel/out/split.json',encoding='utf-8'))
print({k:(len(v) if hasattr(v,'__len__') and not isinstance(v,str) else v) for k,v in s.items()})
Z=np.load('F:/tanitad_scratch/p6fit/fit_pack8.npz',allow_pickle=False)
names=[str(x) for x in Z['pdm_names']]; pd=names.index('pdms'); print(names)
L=Z['logits'].astype(np.float64); Y=Z['pdm'].astype(np.float64); log=Z['log'].astype(str)
b=set(s['logs_B'])
m=np.array([l in b for l in log]); print('FIT-B sets',m.sum(),'heldout share in B',Z['is_heldout'][m].mean())
sg=lambda x:1/(1+np.exp(-x))
p=sg(L)
rule=p[...,0]*p[...,1]*(5*p[...,2]+5*p[...,3]+2*p[...,4])/12
pick=rule.argmax(1)
for nm,mask in (('FIT-B',m),('FIT-A',~m),('all',np.ones_like(m)),('heldout',Z['is_heldout'])):
    I=np.where(mask)[0]; y=Y[I,:,pd]
    print(nm,'n',len(I),'pick',round(100*y[np.arange(len(I)),pick[I]].mean(),3),'best',round(100*y.max(1).mean(),3),'random',round(100*y.mean(1).mean(),3))
# head-by-head regret at the pick on FIT-B
I=np.where(m)[0]; idx=[names.index(n) for n in('nc','dac','ep','ttc','c')]
print('FIT-B sub at pick',{n:round(100*Y[I,pick[I],names.index(n)].mean(),2) for n in('nc','dac','ep','ttc','c')})
print('FIT-B sub at best-pdms',{n:round(100*Y[I,Y[I,:,pd].argmax(1),names.index(n)].mean(),2) for n in('nc','dac','ep','ttc','c')})
