import json, sys, statistics as st
def load(p):
    rows=[]
    for l in open(p, encoding='utf-8'):
        l=l.strip()
        if l.startswith('{'):
            try: rows.append(json.loads(l))
            except Exception: pass
    return rows
def summarize(tag, rows, bins, keys):
    print('==', tag, 'rows', len(rows), 'max step', max(r.get('step',0) for r in rows))
    for lo,hi in bins:
        sel=[r for r in rows if lo<=r.get('step',-1)<hi]
        out=[]
        for k in keys:
            v=[r[k] for r in sel if isinstance(r.get(k),(int,float))]
            out.append(f"{k}={st.mean(v):.4f}(n{len(v)})" if v else f"{k}=NA")
        print(f"  [{lo},{hi})", ' '.join(out))
r7=load('E:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl')
r8=load('E:/refcv8_eval_kit/thor_reads/metrics_20261017T173755.jsonl')
k=['traj','cascade','lr','eval_traj','ema_eval_traj']
summarize('refcv7', r7, [(0,1000),(4000,5000),(9500,10500),(14000,15500),(19000,20500),(29000,30500),(44000,46000),(48000,50401)], k)
summarize('refcv8', r8, [(0,250),(250,1000),(1000,2000),(4000,5000),(9500,10500),(14000,15500),(19000,20500),(20500,22000)], k+['r8_alloc_l1','r8_sat','r8_subscore','r8_cons','r8_v9_cons'])
for tag,rows in (('r7',r7),('r8',r8)):
    pts=[(r['step'],r['lr']) for r in rows if isinstance(r.get('lr'),(int,float))]
    pick=[p for p in pts if p[0] in (50,1000,1050,5000,10000,15000,20000,22000,30000,45000,50350,50400)]
    print(tag, [(s,f"{v:.3e}") for s,v in pick])
