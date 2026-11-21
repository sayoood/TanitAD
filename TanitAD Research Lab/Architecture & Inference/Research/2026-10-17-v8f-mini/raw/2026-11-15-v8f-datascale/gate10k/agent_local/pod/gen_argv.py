import json,sys,os
# maps the box-2 argv (D:\tanitad\...) to pod paths; builds the arm argvs. Run on the pod.
D="/root/dscale10k"; S="/root/strat1/inputs"
def mp(v):
    m={r"D:\tanitad\cache_full\native_tracks.pt":"/root/data/cache_pod/native_tracks.pt",
       r"D:\tanitad\cache_full":"/root/data/cache_pod",
       r"D:\tanitad\data\v9\v9_labels_train.npz":S+"/data/v9/v9_labels_train.npz",
       r"D:\tanitad\data\v9\lead_train4572.npz":S+"/data/v9/lead_train4572.npz",
       r"D:\tanitad\data\v9\refcv6_speed_max_v8_train.jsonl":S+"/data/v9/refcv6_speed_max_v8_train.jsonl",
       r"D:\tanitad\runs\v8f_mini_full\join_train.npz":S+"/join_train.npz",
       r"D:\tanitad\runs\v8f_mini_full\p10\p10_prior.pt":S+"/p10/p10_prior.pt"}
    return m.get(v,v)
A=json.load(open(D+"/argv/m2b_s0_box2.json"))["argv"]
Bs=json.load(open(D+"/argv/argvs_obd_box2.json"))["OBD-CTL_s0"]["argv"]
def setf(t,f,vals):
    t=list(t); i=t.index(f)
    j=i+1
    while j<len(t) and not (t[j].startswith("--") and len(t[j])>2): j+=1
    return t[:i+1]+list(vals)+t[j:]
def delf(t,f):
    t=list(t); i=t.index(f); j=i+1
    while j<len(t) and not (t[j].startswith("--") and len(t[j])>2): j+=1
    return t[:i]+t[j:]
def pod(t,out):
    t=[mp(x) for x in t]; t=setf(t,"--out",[out]); return t
mr=json.load(open(D+"/merged/merge_record.json"))
um=D+"/merged"
def union(t,out,stepsA=True):
    t=pod(t,out)
    t=setf(t,"--cache-dir",["/root/data/cache_union_dscale"])
    t=setf(t,"--v9-train",[um+"/v9_labels_union.npz"]); t=setf(t,"--v9-train-md5",[mr["v9"]["md5"]])
    t=setf(t,"--lead-train",[um+"/lead_union.npz"]); t=setf(t,"--speed-train",[um+"/speed_union.jsonl"])
    t=setf(t,"--tracks",[um+"/native_tracks_union.pt"]); t=setf(t,"--join-train",[um+"/join_union.npz"])
    t=delf(t,"--exclude"); t=t+(["--fit-join","--steps","19164"] if stepsA else ["--fit-join"])
    return t
os.makedirs(D+"/argv",exist_ok=True)
out={"B_s0.A":pod(A,D+"/runs/B_s0/A"),"U_s0.A":union(A,D+"/runs/U_s0/A"),
     "B_s0.B":pod(Bs,D+"/runs/B_s0/B"),"U_s0.B":union(Bs,D+"/runs/U_s0/B",False)}
for k,v in out.items(): json.dump({"argv":v},open(D+"/argv/%s.json"%k,"w"),indent=1)
print({k:len(v) for k,v in out.items()})
