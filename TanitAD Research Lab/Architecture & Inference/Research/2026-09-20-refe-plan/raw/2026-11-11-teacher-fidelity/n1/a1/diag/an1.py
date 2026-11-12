import json,csv,collections
import numpy as np
W={}
for ln in open("n1t_only.jsonl"): d=json.loads(ln); W[d["token"]]=d
H={}
for r in csv.DictReader(open("C:/Users/Admin/n1_run/inputs/refe_p4_teacher_navtest.csv",encoding="utf-8")):
    if r["token"]!="average": H[r["token"]]=r
print(len(W),len(H), list(H[next(iter(H))].keys()))
toks=sorted(set(W)&set(H))
ep_w=np.array([W[t]["tr"]["ep"] for t in toks]); ep_h=np.array([float(H[t]["ego_progress"]) for t in toks])
d=ep_w-ep_h; ad=np.abs(d)
print("n>1e-6",(ad>1e-6).sum(),"max",ad.max(),"pctiles",np.percentile(ad[ad>1e-6],[10,50,90,99]))
m=(ad>1e-6)&(ep_h>0)&(ep_h<1)
ratio=ep_w[m]/ep_h[m]
print("ratio pct",np.percentile(ratio,[0,1,10,50,90,99,100]))
print("sign +",(d[ad>1e-6]>0).sum(),"-",(d[ad>1e-6]<0).sum())
print("ep_h==1 among diffs", ((ep_h==1)&(ad>1e-6)).sum(), "ep_h==0", ((ep_h==0)&(ad>1e-6)).sum())
for lo,hi in [(0,0.01),(0.01,.2),(.2,.5),(.5,.9),(.9,.9999),(.9999,1.0001)]:
    s=(ep_h>=lo)&(ep_h<hi); print(lo,hi,s.sum(),"frac diff",(ad[s]>1e-6).mean() if s.sum() else None)
c=collections.defaultdict(lambda:[0,0])
for t,a in zip(toks,ad): c[W[t]["log"]][0]+=1; c[W[t]["log"]][1]+= a>1e-6
fr=np.array([v[1]/v[0] for v in c.values()]); print("logs",len(c),"frac-of-log-with-diff pct",np.percentile(fr,[0,10,25,50,75,90,100]))
# t_s dependence and ep magnitude
ts=np.array([W[t]["t_s"] or 0 for t in toks]); 
print("diff-token mean ep",ep_h[ad>1e-6].mean(),"nodiff mean ep",ep_h[ad<=1e-6].mean())
