"""E-DEC-6b: is ego decodability coming from OPTICAL FLOW in the 3-frame stack?

E-DEC-7 argues the objective is satisfiable by "ego + noise" partly BECAUSE ego
motion is handed to the model free: the encoder input is [f_{t-2}, f_{t-1}, f_t]
stacked on channels, so inter-frame displacement is directly available.

`nstack1` on Thor cannot test this -- the cache has 9 channels baked in, so
n_stack=1 is a cache REBUILD (it crashed on exactly that). But the question is
answerable with ZERO training and ZERO rebuild: feed the SAME frozen encoder a
COLLAPSED stack [f_t, f_t, f_t]. Identical tensor shape, identical weights, but
every inter-frame difference is zero -- so any decodability that survives is NOT
coming from flow.

  ego R2 COLLAPSES when the stack is flattened -> flow IS the source; removing
      the shortcut is a real lever and E-DEC-6 is worth a cache rebuild.
  ego R2 SURVIVES -> the encoder reads ego from single-frame appearance, the
      shortcut story is wrong, and E-DEC-6 should be dropped.

Paired LOEO on the SAME held-out episodes. Constant control must read 0.0000.
"""
import sys, io, json, numpy as np, torch
from pathlib import Path
from PIL import Image
SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP)); sys.path.insert(0, str(SP/"sp2")); sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
import v7tiny_g2 as G, v7tiny_probe as P
dev = torch.device("cuda"); F = 100
clips = sorted((SP/"sp2/cache/physicalai-val-w120-256x640cyl").glob("*.v2ep.pt"))
arm = sys.argv[1] if len(sys.argv) > 1 else "rdw8"
world, st = G.load_arm(arm, dev)
NORM, FLAT, PO = [], [], []
for c in clips:
    d = torch.load(c, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
    m = min(F, len(off)-1)
    imgs=[torch.from_numpy(np.asarray(Image.open(io.BytesIO(raw[off[i]:off[i+1]])).convert("RGB")).copy()).permute(2,0,1).float()/255.0 for i in range(m)]
    for mode, dst in (("normal", NORM), ("flat", FLAT)):
        out=[]
        with torch.no_grad():
            for s in range(0, m, 8):
                ch=[]
                for i in range(s, min(s+8, m)):
                    idx = [max(i-j,0) for j in (2,1,0)] if mode=="normal" else [i,i,i]
                    ch.append(torch.cat([imgs[k] for k in idx],0))
                x = torch.stack(ch)[:,None].to(dev)
                z,_ = world.stack.encode_window(x, return_tokens=True)
                out.append((z[:,0] if z.dim()>2 else z).float().cpu().numpy())
        dst.append(np.concatenate(out).astype(np.float64))
    PO.append(d["poses"].numpy().astype(np.float64)[:m])
del world; torch.cuda.empty_cache()
COLS={"z_op NORMAL stack":NORM,"z_op FLAT stack (no flow)":FLAT,"constant":[np.ones((len(p),1)) for p in PO]}
T={"speed":[p[:,3:4] for p in PO],"d_ego":[np.concatenate([np.diff(p[:,:2],axis=0),np.zeros((1,2))]) for p in PO]}
n=len(clips)
def loeo(X,Y):
    o=[]
    for e in range(n):
        idx=[i for i in range(n) if i!=e]
        Xf=[np.asarray(X[i])[:len(Y[i])] for i in idx]; Yf=[Y[i][:len(np.asarray(X[i]))] for i in idx]
        Xs=[np.asarray(X[e])[:len(Y[e])]]; Ys=[Y[e][:len(np.asarray(X[e]))]]
        fn=getattr(P,"probe_fit_score",None)
        o.append(fn(Xf,Yf,Xs,Ys,128) if fn else P.probe(Xf+Xs,Yf+Ys,128)[0])
    return np.array(o,dtype=np.float64)
rep={"_evidence_class":"MEASURED (ours)","eval_tier":"T0-DIAGNOSTIC","arm":arm,"step":int(st),"n_episodes":n,"targets":{}}
print(f"\n  E-DEC-6b flow-shortcut test · {arm}@{st} · {n} held-out episodes\n")
print(f"  {'target':<9}{'column':<28}{'R2':>9}{'vs NORMAL':>11}{'t':>7}{'favours':>9}")
print("  "+"-"*70)
for tn,Y in T.items():
    R={k:loeo(v,Y) for k,v in COLS.items()}; base=R["z_op NORMAL stack"]; rep["targets"][tn]={}
    for k in COLS:
        dd=R[k]-base; m=float(dd.mean()); se=float(dd.std(ddof=1)/np.sqrt(len(dd))); t=m/max(se,1e-12)
        rep["targets"][tn][k]={"r2":round(float(R[k].mean()),4),"delta_vs_normal":round(m,4),"t":round(t,2),"n_favouring":int((dd>0).sum())}
        print(f"  {tn:<9}{k:<28}{float(R[k].mean()):>+9.4f}{m:>+11.4f}{t:>7.2f}{int((dd>0).sum()):>6}/{len(dd)}")
    print()
drops=[tn for tn in T if rep["targets"][tn]["z_op FLAT stack (no flow)"]["t"] < -2.2]
rep["verdict"]=(f"flattening the stack REMOVES decodability on {drops or 'NO target'} ⇒ "
  + ("optical flow IS the source of ego decodability; the shortcut is real and E-DEC-6 "
     "(n_stack=1) is worth a cache rebuild." if drops else
     "ego decodability SURVIVES without inter-frame motion ⇒ it comes from single-frame "
     "appearance, the flow-shortcut story is WRONG, and E-DEC-6 should be DROPPED."))
print(f"  VERDICT: {rep['verdict']}")
json.dump(rep, open(SP/"e_dec6b_flow.json","w"), indent=1)
