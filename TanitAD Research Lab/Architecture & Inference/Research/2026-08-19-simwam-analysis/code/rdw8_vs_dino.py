"""Is rdw8 ACTUALLY better than frozen DINOv3? A PAIRED test, not two point estimates.

⛔ I claimed "rdw8 beats frozen DINOv3 on d_ego" from +0.3601 vs +0.3238 -- two
numbers that were each measured against a THIRD column (`lewm`), never against
each other. That is the same unpaired comparison that made me call E-DEC-1 a null.
Here the two columns are paired ON THE SAME HELD-OUT EPISODE.
"""
import sys, json, numpy as np, torch
from pathlib import Path
SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP)); sys.path.insert(0, str(SP/"sp2")); sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
import v7tiny_g2 as G, v7tiny_probe as P
dev = torch.device("cuda"); F = 100
clips = sorted((SP/"sp2/cache/physicalai-val-w120-256x640cyl").glob("*.v2ep.pt"))
COLS, PO = {}, []
for arm in ("lewm", "rdw8"):
    w, st = G.load_arm(arm, dev); col = []
    for c in clips:
        z,_,_ = G.encode_clip(w, c, dev, F); col.append(z.numpy().astype(np.float64))
        if arm == "lewm":
            d = torch.load(c, map_location="cpu", weights_only=False)
            PO.append(d["poses"].numpy().astype(np.float64)[:len(col[-1])])
    COLS[arm] = col; del w; torch.cuda.empty_cache()
COLS["dino"] = P.dinov3_encode(clips, F, dev)
COLS["constant"] = [np.ones((len(p),1)) for p in PO]
T = {"speed":[p[:,3:4] for p in PO],
     "d_ego":[np.concatenate([np.diff(p[:,:2],axis=0),np.zeros((1,2))]) for p in PO]}
n = len(clips)
def loeo(X, Y):
    out=[]
    for e in range(n):
        idx=[i for i in range(n) if i!=e]
        Xf=[np.asarray(X[i])[:len(Y[i])] for i in idx]; Yf=[Y[i][:len(np.asarray(X[i]))] for i in idx]
        Xs=[np.asarray(X[e])[:len(Y[e])]]; Ys=[Y[e][:len(np.asarray(X[e]))]]
        fn=getattr(P,"probe_fit_score",None)
        out.append(fn(Xf,Yf,Xs,Ys,128) if fn else P.probe(Xf+Xs,Yf+Ys,128)[0])
    return np.array(out,dtype=np.float64)
rep={"_evidence_class":"MEASURED (ours)","eval_tier":"T0-DIAGNOSTIC","n_episodes":n,
     "estimator":"LOEO PAIRED, rdw8 vs DINOv3 on the SAME held-out episode","targets":{}}
print(f"\n  PAIRED rdw8 vs frozen DINOv3 — {n} held-out val episodes\n")
print(f"  {'target':<9}{'pair':<20}{'mean d':>9}{'SE':>8}{'t':>7}{'favours 1st':>13}  reading")
print("  "+"-"*76)
for tn,Y in T.items():
    R={k:loeo(v,Y) for k,v in COLS.items()}
    rep["targets"][tn]={"r2":{k:round(float(R[k].mean()),4) for k in R}}
    for a,b in (("rdw8","dino"),("rdw8","lewm"),("dino","lewm"),("rdw8","rdw8")):
        d=R[a]-R[b]; m=float(d.mean()); se=float(d.std(ddof=1)/np.sqrt(len(d)))
        t=m/max(se,1e-12); k=int((d>0).sum())
        rd=("CONTROL — must be 0" if a==b else
            f"{a} BEATS {b}" if t>2.2 else f"{a} LOSES to {b}" if t<-2.2 else "NOT separable")
        rep["targets"][tn][f"{a}-{b}"]={"mean_delta":round(m,4),"se":round(se,4),"t":round(t,2),
                                        "n_favouring_first":k,"n":len(d)}
        print(f"  {tn:<9}{a+' - '+b:<20}{m:>+9.4f}{se:>8.4f}{t:>7.2f}{k:>9}/{len(d)}  {rd}")
    print()
c=[tn for tn in T if rep["targets"][tn]["rdw8-dino"]["t"]>2.2]
l=[tn for tn in T if rep["targets"][tn]["rdw8-dino"]["t"]<-2.2]
rep["verdict"]=(f"rdw8 beats DINOv3 on {c or 'NO target'}; loses on {l or 'none'}; "
                f"not separable elsewhere. ⇒ the earlier 'beats DINOv3' claim was an UNPAIRED "
                f"point-estimate comparison and is {'CONFIRMED' if c else 'NOT SUPPORTED'}.")
print(f"  VERDICT: {rep['verdict']}")
json.dump(rep, open(SP/"rdw8_vs_dino_paired.json","w"), indent=1)
