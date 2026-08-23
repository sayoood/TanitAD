import sys, json, numpy as np
sys.path.insert(0,"/workspace/TanitAD/taniteval")
from taniteval.ci import paired_episode_cluster_bootstrap as P
d=np.load("/workspace/ecr_v2_arrays.npz")
R,T,E=d["rollout"],d["tf"],d["episode"]
out={"n_windows":int(R.shape[0]),"n_episodes":int(np.unique(E).size),
     "metric":"1 - cos(z_hat_k, z_true_k)",
     "estimator":"PAIRED episode-cluster bootstrap (taniteval.ci), B=2000; interval is on the "
                 "DIFFERENCE e_rollout - e_teacher_forced. No CI on the RATIO is computed.","CR":{}}
for k in (4,8,16,20):
    a,b=R[:,k-1].astype(float),T[:,k-1].astype(float)
    r=P(a,b,E)
    out["CR"][f"k{k}"]={"e_rollout":round(float(a.mean()),6),"e_teacher_forced":round(float(b.mean()),6),
        "CR_point":round(float(a.mean()/max(b.mean(),1e-12)),4),
        "delta":r["delta"],"lo":r["lo"],"hi":r["hi"],"separated":bool(r["separated"]),
        "p_delta_gt0":r.get("p_delta_gt0"),"n_boot":r.get("n_boot")}
json.dump(out,open("/workspace/ecr_v2_ci.json","w"),indent=1)
print(json.dumps(out,indent=1))
