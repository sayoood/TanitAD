import json,glob,os,lzma,collections,hashlib
import numpy as np
B="E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-lane-discipline/heldout_teacher_lane_labels.jsonl.xz"
def fk(r): k=r["key"]; return (k[0],k[1],int(k[2]),int(k[3]),int(r["ckpt_step"]))
bank_rows=[json.loads(l) for l in lzma.open(B,"rt",encoding="utf-8") if l.startswith("{")]
bk={}
for r in bank_rows: bk.setdefault(fk(r),[]).append(r)
print("bank rows",len(bank_rows),"unique full keys",len(bk),"dup",sum(len(v)>1 for v in bk.values()),"unique (log,token)",len({k[:2] for k in bk}))
tl=[]
for fp in sorted(glob.glob("s1aux/heldout_teacher_lane/heldout_teacher_lane/*.jsonl")):
    tl+= [json.loads(l) for l in open(fp,encoding="utf-8") if l.startswith("{")]
tk={}
for r in tl: tk.setdefault(fk(r),[]).append(r)
print("s1aux rows",len(tl),"unique full keys",len(tk),"dup",sum(len(v)>1 for v in tk.values()),"unique (log,token)",len({k[:2] for k in tk}))
print("key sets equal",set(bk)==set(tk),"common",len(set(bk)&set(tk)))
ex=0;teq=0;peq=0;maxd=0;n=0
for k in set(bk)&set(tk):
    a,b=bk[k][0],tk[k][0]; n+=1
    ex+= (a["teacher"]==b["teacher"] and a["props"]==b["props"])
    teq+= a["teacher"]==b["teacher"]; peq+= a["props"]==b["props"]
    for pa,pb in zip(a["props"],b["props"]):
        for c in pa: maxd=max(maxd,abs(pa[c]-pb[c]))
    for c in a["teacher"]: maxd=max(maxd,abs(a["teacher"][c]-b["teacher"][c]))
print("compared",n,"exact(teacher & props)",ex,"teacher-only",teq,"props-only",peq,"max abs diff",maxd)
# second check: PDM dac/ddc vs teacher violations
pd={}
for fp in glob.glob("s1aux/heldout_pdm/heldout_pdm/*.jsonl"):
    for l in open(fp,encoding="utf-8"):
        r=json.loads(l); pd[fk(r)]=r
rows=0;agree_dac=0;agree_ddc=0;tot=0
dacv=collections.Counter();ddcv=collections.Counter()
for k,v in tk.items():
    p=pd[k]
    for i,pr in enumerate(v[0]["props"]):
        tot+=1
        agree_dac+= (p["dac"][i]==1.0)==(pr["dac.violation"]==0.0)
        agree_ddc+= (p["ddc"][i]==1.0)==(pr["ddc.violation"]==0.0)
        dacv[(p["dac"][i],pr["dac.violation"])]+=1; ddcv[(p["ddc"][i],pr["ddc.violation"])]+=1
print("proposals",tot,"dac agree",agree_dac,"ddc agree",agree_ddc)
print("dac pairs",dacv.most_common(6)); print("ddc pairs",ddcv.most_common(6))
