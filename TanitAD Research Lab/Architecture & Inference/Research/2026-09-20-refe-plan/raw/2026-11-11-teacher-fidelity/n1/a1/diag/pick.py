import json,csv,random
W={}
for ln in open("n1t_only.jsonl"): d=json.loads(ln); W[d["token"]]=d
H={r["token"]:r for r in csv.DictReader(open("C:/Users/Admin/n1_run/inputs/refe_p4_teacher_navtest.csv",encoding="utf-8")) if r["token"]!="average"}
flip=[];epd=[];same=[]
for t,w in W.items():
    h=H[t]; tr=w["tr"]
    if abs(tr["nc"]-float(h["no_at_fault_collisions"]))>1e-6 or abs(tr["ttc"]-float(h["time_to_collision_within_bound"]))>1e-6: flip.append(t)
    elif abs(tr["ep"]-float(h["ego_progress"]))>1e-6 and 0<float(h["ego_progress"])<1: epd.append(t)
    elif 0<float(h["ego_progress"])<1: same.append(t)
random.seed(1)
sel={"flip":flip,"epdiff":random.sample(epd,10),"same":random.sample(same,10)}
print({k:len(v) for k,v in sel.items()}, len(epd), len(same))
json.dump(sel,open("diag_tokens.json","w"))
# logs of flips and rows structure
print([ (t,W[t]["log"][:30]) for t in flip])
