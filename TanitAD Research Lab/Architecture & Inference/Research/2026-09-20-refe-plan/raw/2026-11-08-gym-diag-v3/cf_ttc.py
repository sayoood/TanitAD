"""Counterfactual: same 24 arms, log mode, ttc predicate variants (diag copy gym_env_cf.py; pilot_v2 untouched).
usage: cf_ttc.py SAMPLER IDX MODE OUT.json   variants: base(0), off(-1), brake3(3), brake6(6)"""
import os,sys,json
os.environ["CUDA_VISIBLE_DEVICES"]="-1"; sys.dont_write_bytecode=True
sys.path.insert(0,"E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v3_diag")
import numpy as np, gym_env_cf as g
sampler,idx,mode,out=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4]
S=g.GymState((g.select_states if sampler=="pilot" else g.select_states_stall)(20,10)[idx])
res=dict(state=idx,sampler=sampler,mode=mode,variants={})
for name,b in (("off",-1.0),("brake3",3.0),("brake6",6.0)):
    g.TTC_BRAKE=b; rows=[]
    for k in range(24):
        logs=S.rollout(k,mode)
        h=[(c["cycle"],c["hard"],c["hard_step"]) for c in logs if c["hard"]]
        rows.append(dict(k=k,ret=g.episode_return(logs),hard=h[0] if h else None,n=len(logs),vend=logs[-1]["v"]))
    res["variants"][name]=rows
    best=max(rows,key=lambda r:r["ret"])
    print(name,"n_hard",sum(1 for r in rows if r["hard"]),"oracle",best["k"],round(best["ret"],3),best["hard"],"types",sorted(set(r["hard"][1] for r in rows if r["hard"])),flush=True)
    json.dump(res,open(out,"w"))
print("CF_DONE",flush=True)
