import json,glob,os,shutil,tarfile,sys
R="F:/tanitad_scratch/stage0_20261108/data"
os.chdir(R)
# pixels
if not os.path.exists("train64_dl/.extracted"):
    with tarfile.open("train64_dl/pix.tar") as t: t.extractall("navtrain_pixels")
    open("train64_dl/.extracted","w").write("x")
imgs=[l.strip() for l in open("train64_dl/imgs.txt")]
def have(rel):
    return any(os.path.exists(f"navtrain_pixels/{d}/{rel}") for d in os.listdir("navtrain_pixels") if d.startswith("navtrain_current_"))
tg=[json.loads(l) for l in open("train64_dl/train_grow/targets_rank0.jsonl",encoding="utf-8")]
keep=[r for r in tg if all(have(i) for i in r["image"])]
print("train rows",len(tg),"with all 4 frames",len(keep))
K=lambda r:(r["log_name"],r.get("token",""),int(r["step"]),int(r.get("rank",0)))
keys={K(r) for r in keep}
os.makedirs("refe_navtrain/train_grow_s0",exist_ok=True)
with open("refe_navtrain/train_grow_s0/targets_rank0.jsonl","w",encoding="utf-8") as f:
    for l in open("refe_navtrain/train_grow/targets_rank0.jsonl",encoding="utf-8"): f.write(l if l.endswith("\n") else l+"\n")
    for r in keep: f.write(json.dumps(r)+"\n")
shutil.copy("refe_navtrain/train_grow/targets_rank1.jsonl","refe_navtrain/train_grow_s0/")
shutil.copy("refe_navtrain/train_grow/calib_table.json","refe_navtrain/train_grow_s0/")
os.makedirs("train_bank/sets",exist_ok=True)
with open("train_bank/sets/onpolicy_train.jsonl","w",encoding="utf-8") as f:
    for l in open("train64_dl/sets/onpolicy_train64.jsonl",encoding="utf-8"):
        r=json.loads(l)
        if K(r) in keys: f.write(l)
def sub(srcglob,dst,name):
    os.makedirs(dst,exist_ok=True); n=0
    with open(f"{dst}/{name}","w",encoding="utf-8") as f:
        for fp in sorted(glob.glob(srcglob)):
            for l in open(fp,encoding="utf-8"):
                if not l.startswith("{"): continue
                k=json.loads(l)["key"]
                if (k[0],k[1],int(k[2]),int(k[3])) in keys: f.write(l); n+=1
    return n
print("train_pdm rows",sub("p2pdm/x/train_pdm/*.jsonl","train_bank/train_pdm","pdm_train.jsonl"))
print("train_teacher_lane rows",sub("s1aux/train_teacher_lane/train_teacher_lane/*.jsonl","train_bank/train_teacher_lane","teacher_lane_train.jsonl"))
print("train sets",sum(1 for _ in open("train_bank/sets/onpolicy_train.jsonl")))
