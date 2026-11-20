"""numpy-only: builds the per-window label arrays (same rule as native40_dump.label_lookup) and a SYNTHETIC dump (logit/gap from labels + noise) to test the scorer. No model, no head number."""
import json, sys, hashlib, numpy as np
ART="E:/Projects/TanitAD-artifacts"
DATA=f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
J=np.load(f"{DATA}/join_eval141.npz", allow_pickle=False); clips=[str(c) for c in J["clip"]]
g=json.load(open(f"{DATA}/eval_grid_windows.json")); ci_of={c:i for i,c in enumerate(clips)}
wins=[(ci_of[c],int(t)) for c in g["order"] for t in g["windows"][c]]
z=np.load(f"{ART}/sfresh_inputs/inputs/v9/lead_eval139.npz", allow_pickle=True)
sh=np.asarray(z["sha12"]).astype(str); kk=np.asarray(z["k"],np.int64); ld=np.asarray(z["lead"],np.int64); gp=np.asarray(z["gap"],np.float64)
idx={(s,int(k)):i for i,(s,k) in enumerate(zip(sh,kk))}
s12=lambda c: hashlib.sha256(c.encode()).hexdigest()[:12]
N=len(wins); has=np.zeros(N,bool); lead=np.zeros(N,np.int64); gap=np.full(N,np.nan)
for w,(ci,t) in enumerate(wins):
    i=idx.get((s12(clips[ci]),2*t))
    if i is not None: has[w],lead[w],gap[w]=True,ld[i],gp[i]
near=has&(lead==1)&np.isfinite(gap)&(gap<=40)
print("windows",N,"episodes",len(set(w[0] for w in wins)),"has",has.sum(),"near",near.sum(),"neg",(has&~near).sum())
W=json.load(open("F:/TanitAD-scratch/v8proof/leversheet/LS_REF40_WINDOWS.json")); pos={(clips[c],t):i for i,(c,t) in enumerate(wins)}
m=0;f=0
for e,t,gt in zip(W["eid"],W["t"],W["gap_t"]):
    j=pos.get((e,int(t)))
    if j is None: continue
    f+=1; m+= bool(has[j]) and lead[j]==1 and abs(gap[j]-gt)<=1e-3
print("coord windows",len(W["eid"]),"found",f,"matched",m,"share",m/len(W["eid"]))
if len(sys.argv)>1:
    rng=np.random.default_rng(1)
    for s in (0,1):
        gh=np.where(near,np.clip(np.nan_to_num(gap)/40+rng.normal(0,.12,N),0,1),rng.random(N)*.5)
        lg=np.where(near,1.5,-1.5)+rng.normal(0,1.5,N)
        meta=json.dumps({"seed":s,"ckpt_sha256":"SYNTH"*12,"synthetic":True})
        np.savez(f"F:/TanitAD-scratch/native40/out_synth/dump_s{s}.npz",ci=np.array([w[0] for w in wins]),t0=np.array([w[1] for w in wins]),k=np.array([2*w[1] for w in wins]),
          eid=np.array([clips[w[0]] for w in wins]),frame_ok_now=np.ones(N,bool),lead_logit=lg,lead_gap_hat=gh,has=has,lead=lead,gap=gap,near=near,meta=meta)
