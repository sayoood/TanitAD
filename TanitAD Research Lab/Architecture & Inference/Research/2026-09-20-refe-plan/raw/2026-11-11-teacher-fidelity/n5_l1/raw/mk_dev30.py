import json,glob
Q=json.load(open("dev/Q3_repairs.json",encoding="utf-8"))["instances"]
rolls={}
for fp in glob.glob("dev/rows/rolls/*_s*.jsonl"):
    import os; run=os.path.basename(fp).rsplit("_s",1)[0]
    for ln in open(fp,encoding="utf-8"):
        try:r=json.loads(ln)
        except Exception:continue
        if "traj20" in r: rolls[(run,r["token"])]=r["traj20"]
out=[]
for q in Q:
    out.append({"id":q["run"]+":"+q["token"],"token":q["token"],"log":q["log"],"tag":"dev30","t20":rolls[(q["run"],q["token"])]})
open("dev30.jsonl","w").write("\n".join(json.dumps(o) for o in out)+"\n")
print(len(out),len({o["token"] for o in out}))
