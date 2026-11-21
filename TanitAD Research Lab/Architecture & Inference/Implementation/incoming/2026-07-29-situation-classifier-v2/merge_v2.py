import json, numpy as np, os
G="/workspace/sitclf/bundle"; V="/workspace/sitclf/bundle_v2"; O="/workspace/sitclf/bundle_v2m"
meta=json.load(open(os.path.join(G,"sc_meta.json")))
g=np.load(os.path.join(G,"sc_labels.npz")); v=np.load(os.path.join(V,"sc_labels.npz"))
ep=v["episode"]; print("episode: dtype",ep.dtype,"min",int(ep.min()),"max",int(ep.max()),
                       "uniq",len(np.unique(ep)),"len",len(ep))
SITS=("lane_change","intersection")
order=np.argsort(ep,kind="stable")
bounds={}
u,starts=np.unique(ep[order],return_index=True)
ends=np.r_[starts[1:],len(ep)]
for k,s,e in zip(u,starts,ends): bounds[int(k)]=(int(s),int(e))
out={}; kept=[]; dT=0; dmiss=0
for m in meta:
    k=m["k"]; T=int(m["T"]); ge="c%d_ego"%k; gp="c%d_priv"%k
    if k not in bounds or ge not in g.files or gp not in g.files: dmiss+=1; continue
    s,e=bounds[k]
    if e-s!=T or len(g[ge])!=T: dT+=1; continue
    idx=order[s:e]
    out[ge]=g[ge]; out[gp]=g[gp]
    for sit in SITS:
        out["c%d_y_%s"%(k,sit)]=v["y_%s"%sit][idx]
        out["c%d_valid_%s"%(k,sit)]=v["valid_%s"%sit][idx]
    kept.append(m)
os.makedirs(O,exist_ok=True)
np.savez_compressed(os.path.join(O,"sc_labels.npz"), **out)
json.dump(kept,open(os.path.join(O,"sc_meta.json"),"w"))
print("kept",len(kept),"drop_T",dT,"drop_miss",dmiss)
for sit in SITS:
    y=np.concatenate([out["c%d_y_%s"%(m["k"],sit)] for m in kept])
    va=np.concatenate([out["c%d_valid_%s"%(m["k"],sit)] for m in kept]).astype(bool)
    print("  %-13s pos=%6d scorable=%7d base_rate=%.5f"%(sit,int(y.sum()),int(va.sum()),float(y[va].mean())))
