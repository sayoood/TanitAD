import lzma, json, numpy as np, hashlib
fn='E:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz'
L=[json.loads(l) for l in lzma.open(fn,'rt',encoding='utf-8')]
z=np.load('E:/Projects/TanitAD-artifacts/v9labels/inputs/lead_eval139.npz',allow_pickle=True)
sh=lambda c: hashlib.sha256(c.encode()).hexdigest()[:12]
# lead dict
lead={}
for s,k,ld,g in zip(z['sha12'],z['k'],z['lead'],z['gap']): lead[(str(s),int(k))]=(int(ld),float(g))
# nearest in-corridor agent ahead per (clip, frame)
cor={}
mind=[]
for l in L:
    best=np.nan
    for a in l['agents']:
        mind.append(np.hypot(a['cx'],a['cy']))
        if a['cx']>0 and abs(a['cy'])<1.5:
            d=a['cx']-a['l']/2
            if not(best<=d): best=d if np.isnan(best) else min(best,d)
    cor[(sh(l['clip_id']),l['frame'])]=best
mind=np.array(mind); print('agent dist quantiles',np.percentile(mind,[0,0.1,1,50]))
for off in range(-6,7):
    xs=[];ys=[]
    for (s,k),(ld,g) in lead.items():
        if ld==1 and g<40:
            c=cor.get((s,k+off),np.nan)
            if np.isfinite(c): xs.append(g); ys.append(c)
    xs=np.array(xs);ys=np.array(ys)
    print(off,len(xs),'corr',np.corrcoef(xs,ys)[0,1].round(4),'MAE',np.abs(xs-ys).mean().round(3),'median diff',np.median(ys-xs).round(3))
