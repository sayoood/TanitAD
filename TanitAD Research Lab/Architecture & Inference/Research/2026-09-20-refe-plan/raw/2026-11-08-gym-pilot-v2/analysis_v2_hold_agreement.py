import json, numpy as np
def rank(x):
    x=np.asarray(x,float);o=np.argsort(x,kind='stable');r=np.empty(len(x));r[o]=np.arange(len(x))
    for v in np.unique(x):
        i=x==v
        if i.sum()>1:r[i]=r[i].mean()
    return r
def sp(a,b):
    ra,rb=rank(a),rank(b)
    if ra.std()==0 or rb.std()==0:return float('nan')
    return float(np.corrcoef(ra,rb)[0,1])
m=json.load(open('run_main/results.json'))['v2']['returns_by_state_mode']
h=json.load(open('run_hold2/results.json'))['v2']['returns_by_state_mode']
k=list(m)[0]; print(k, len(m[k]), len(m), len(h))
for mode in ['log','idm']:
    ks=[k for k in m if k.endswith(mode)]
    R=[];T=[];T3=[];reg=[]
    for k in ks:
        a=np.array(m[k]);b=np.array(h[k]); R.append(sp(a,b)); T.append(a.argmax()==b.argmax()); T3.append(b.argmax() in np.argsort(-a)[:3]); reg.append(a.max()-a[b.argmax()])
    R=np.array(R)
    print(mode,len(ks),'spearman mean %.3f median %.3f min %.3f nan %d'%(np.nanmean(R),np.nanmedian(R),np.nanmin(R),np.isnan(R).sum()),'top1 %d/%d'%(sum(T),len(T)),'top3 %d'%sum(T3),'regret mean %.3f max %.3f'%(np.mean(reg),np.max(reg)),'mean best %.3f'%np.mean([max(m[k]) for k in ks]))
    print([round(x,2) for x in R])
a=np.concatenate([m[k] for k in m]);b=np.concatenate([h[k] for k in m]);print('pooled',sp(a,b))
for r in ['run_main','run_stall']:
    d=json.load(open(r+'/results.json'));print(r,{md:{t:(round(x['fired'],4),x['applicable']) for t,x in d['activation'][md].items() if t in('q_stall','q_free','q_gap','q_comf')} for md in('log','idm')})
