import json, numpy as np, hashlib, collections
ART="E:/Projects/TanitAD-artifacts"
z=np.load(f"{ART}/sfresh_inputs/inputs/v9/lead_eval139.npz", allow_pickle=True)
sh=np.asarray(z["sha12"]).astype(str); kk=np.asarray(z["k"],np.int64); ld=np.asarray(z["lead"],np.int64); gp=np.asarray(z["gap"],np.float64)
print({k:z[k].shape for k in z.files}, "k range", kk.min(), kk.max(), "k diffs", np.unique(np.diff(np.sort(kk[sh==sh[0]])))[:5])
s12=lambda c: hashlib.sha256(c.encode()).hexdigest()[:12]
W=json.load(open("F:/TanitAD-scratch/v8proof/leversheet/LS_REF40_WINDOWS.json"))
cnt=collections.Counter(); nf=0
for e,g0,t,gt in zip(W["eid"],W["g0"],W["t"],W["gap_t"]):
    m=(sh==s12(e))&(ld==1)&(np.abs(gp-gt)<=1e-3)
    ks=kk[m]
    if len(ks)==0: nf+=1
    for k in ks: cnt[int(k)-int(t)]+=1; cnt[('r',int(k)/ max(t,1))]+=0
print("windows with NO label row of that gap in the episode:",nf,"of",len(W["eid"]))
print("k - t offsets (top):",[(a,b) for a,b in cnt.most_common(8) if not isinstance(a,tuple)])
print("first windows: eid-short g0 t gap_t -> candidate k's")
for i in range(0,534,23):
    e,g0,t,gt=W["eid"][i],W["g0"][i],W["t"][i],W["gap_t"][i]
    m=(sh==s12(e))&(ld==1)&(np.abs(gp-gt)<=1e-3)
    print(e[:6],g0,t,gt,kk[m].tolist(), "2t=",2*int(t),"g0+2*11=",g0+22)
