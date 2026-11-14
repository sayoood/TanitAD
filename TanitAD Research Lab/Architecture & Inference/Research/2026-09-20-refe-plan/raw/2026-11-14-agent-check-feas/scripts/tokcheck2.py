import numpy as np, os, glob
d=np.load("F:/tanitad_scratch/p6fit/fit_pack.npz",allow_pickle=False)
logs=sorted(set(d["log"].tolist()))
roots=["E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/","E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval/","E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/mini/","E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test/","C:/Users/Admin/rj5db/","F:/tanitad_scratch/rj5nt/dbB/","F:/tanitad_scratch/rj5nt/db/"]
found={}
for r in roots:
    n=0
    for l in logs:
        if os.path.exists(r+l+".db"): found.setdefault(l,[]).append(r); n+=1
    print(r,"has",n,"of",len(logs))
print("logs with a DB somewhere on this box",len(found),"of",len(logs))
tok=d["token"]; lg=d["log"]
cnt=np.array([ (lg==l).sum() for l in logs]); have=np.array([l in found for l in logs])
print("sets covered locally",int(cnt[have].sum()),"of",int(cnt.sum()))
ho=d["is_heldout"]; print("heldout sets covered", int(sum(((lg==l)&ho).sum() for l in logs if l in found)),"of",int(ho.sum()))
open("C:/Users/Admin/agck/found_logs.txt","w").write("\n".join(sorted(found)))
