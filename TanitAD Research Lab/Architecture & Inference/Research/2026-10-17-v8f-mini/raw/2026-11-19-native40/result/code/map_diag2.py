import json, numpy as np, hashlib
ART="E:/Projects/TanitAD-artifacts"
z=np.load(f"{ART}/sfresh_inputs/inputs/v9/lead_eval139.npz", allow_pickle=True)
idx={(str(s),int(k)):i for i,(s,k) in enumerate(zip(z["sha12"].astype(str),z["k"]))}
s12=lambda c: hashlib.sha256(c.encode()).hexdigest()[:12]
W=json.load(open("F:/TanitAD-scratch/v8proof/leversheet/LS_REF40_WINDOWS.json"))
for name,f in (("k=2t",lambda g0,t:2*int(t)),("k=2*g0",lambda g0,t:2*int(g0)),("k=g0",lambda g0,t:int(g0)),("k=t",lambda g0,t:int(t))):
    m=0
    for e,g0,t,gt in zip(W["eid"],W["g0"],W["t"],W["gap_t"]):
        i=idx.get((s12(e),f(g0,t)))
        m+= i is not None and z["lead"][i]==1 and abs(float(z["gap"][i])-gt)<=1e-3
    print(name,m,"/534",m/534)
print("t-g0 values",set(np.array(W["t"])-np.array(W["g0"])))
