import json,glob,collections,random
ho=set(open("ho/refe_heldout/heldout_logs.txt",encoding="utf-8").read().split())
def k(r): x=r["key"]; return (x[0],x[1],int(x[2]),int(x[3]))
pd={}; n=0; dup=0
for fp in sorted(glob.glob("p2pdm/x/train_pdm/*.jsonl")):
    for l in open(fp,encoding="utf-8"):
        if not l.startswith("{"): continue
        r=json.loads(l); n+=1
        kk=k(r)
        if kk in pd: dup+=1
        if all(len(r.get(c,()))==64 for c in("nc","dac","ep","ttc","c","ddc")):
            if kk not in pd or r["ckpt_step"]>=pd[kk]: pd[kk]=r["ckpt_step"]
print("train pdm rows",n,"keys",len(pd),"dup",dup,"in heldout logs",sum(x[0] in ho for x in pd))
tl={}
for fp in sorted(glob.glob("s1aux/train_teacher_lane/train_teacher_lane/*.jsonl")):
    for l in open(fp,encoding="utf-8"):
        if not l.startswith("{"): continue
        r=json.loads(l); kk=k(r)
        if kk not in tl or r["ckpt_step"]>=tl[kk]: tl[kk]=r["ckpt_step"]
print("train teacher lane keys",len(tl),"in heldout logs",sum(x[0] in ho for x in tl))
both=[x for x in pd if x in tl and x[0] not in ho]
print("both",len(both),"ckpt equal",sum(pd[x]==tl[x] for x in both))
cand=sorted(x for x in both if pd[x]==tl[x] and x[2]==0 and x[3]==0)
print("cand rank0",len(cand))
random.Random(20261108).shuffle(cand)
pick=cand[:64]
json.dump([list(x)+[pd[x]] for x in pick],open("../train_pick.json","w"))
print(pick[:3], collections.Counter(x[0] for x in pick).most_common(3))
