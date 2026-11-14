import numpy as np, os, sqlite3
d=np.load("F:/tanitad_scratch/p6fit/fit_pack.npz",allow_pickle=False)
found=open("C:/Users/Admin/agck/found_logs.txt").read().split()
R="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/"
lg=d["log"]; tok=d["token"]
idx=[i for i in range(len(lg)) if lg[i] in found and os.path.exists(R+lg[i]+".db")][:400:80]
hit=0
for i in idx:
    c=sqlite3.connect(f"file:{R}{lg[i]}.db?mode=ro&immutable=1",uri=True)
    hit+=c.execute("select count(*) from lidar_pc where lower(hex(token))=?",(str(tok[i]).lower(),)).fetchone()[0]; c.close()
print("fit-pack token found as nuPlan lidar_pc.token:",hit,"of",len(idx),"example",tok[idx[0]],lg[idx[0]])
