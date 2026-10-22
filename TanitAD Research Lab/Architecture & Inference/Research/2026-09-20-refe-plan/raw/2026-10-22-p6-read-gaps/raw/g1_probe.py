import sys, time, json
sys.path.insert(0, "C:/Users/Admin/par18/p6gaps/sandbox/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval")
import stage3_full as S3
t=time.time()
p2=S3.pick_baseline("p2"); u4=S3.pick_baseline("u400")
print("p2 pin_ok",p2["pin_ok"],p2["csv_md5"],p2["seam_md5"],"| csv",p2["csv"],"| fam",p2["families"], "| report", p2["report"])
print("u400 pin_ok",u4["pin_ok"])
P=S3.load_table(p2["csv"]); U=S3.load_table(u4["csv"])
import gzip
exp=json.load(gzip.open(S3.EXPORT,"rt",encoding="utf-8"))["tokens"]
tl={t:v["log_name"] for t,v in exp.items()}
toks=sorted(P)
r=S3.read_one(P,U,toks,tl,"PDMS")
print({k:r[k] for k in("mean","ci95","base_mean_x100","arm_mean_x100","n","n_logs")}, round(time.time()-t,1),"s")
