import json, glob, os
B="C:/Users/Admin/p5c_scratch/bank"
logs={}
for rk in (0,1):
    for line in open(f"{B}/targets_rank{rk}.jsonl",encoding="utf-8"):
        i=line.find('"log_name": "'); 
        r=json.loads(line); logs[r["log_name"]]=logs.get(r["log_name"],0)+1
have=set()
for s in ("mini","test","trainval"):
    for p in glob.glob(f"E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/{s}/*.db"): have.add(os.path.basename(p)[:-3])
for p in glob.glob("E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/*.db"): have.add(os.path.basename(p)[:-3])
miss=sorted(l for l in logs if l not in have)
open("C:/Users/Admin/p5c_scratch/missing_logs.txt","w").write("\n".join(miss)+"\n")
print(len(miss), sum(logs[l] for l in miss), sum(logs.values()))
