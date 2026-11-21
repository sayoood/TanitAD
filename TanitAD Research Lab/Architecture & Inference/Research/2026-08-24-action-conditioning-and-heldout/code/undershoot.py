"""E-DEC-47 — DOES THE PREDICTOR UNDER-SHOOT THE MAGNITUDE OF CHANGE?

Q2X came back BELOW CHANCE on every arm (0.068-0.175 vs 0.333): an EXTREME action
produces a prediction closer to the true future than the arm's OWN TRUE ACTION
does. The reading I offered is that the predictor UNDER-SHOOTS the magnitude of
change, so amplifying the action input accidentally compensates.

⭐ THAT READING HAS A DIRECT, FALSIFIABLE CONSEQUENCE. If it is right, rescaling
the predicted DELTA by some alpha should improve the fit, and the optimal alpha
should be > 1:
        zhat_alpha = z_last + alpha * (zhat - z_last)
    alpha* > 1  -> UNDER-SHOOT confirmed
    alpha* ~ 1  -> the reading is WRONG and Q2X needs another explanation
    alpha* < 1  -> the predictor OVER-shoots, and the Q2X story is backwards
⚠️ alpha is fit on FIT CLIPS and applied to a HELD-OUT clip, LOEO — fitting it on
the scored clip would guarantee an improvement and prove nothing.
"""
import json, os, pathlib, sys
import numpy as np, torch
SP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SP)); sys.path.insert(0, str(SP/"sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
LEAD = pathlib.Path(os.environ.get("SPD_CORPUS", str(SP/"sp2/cache/physicalai-val130-heldout")))
ARMS = os.environ.get("SPD_ARMS","postrain10k,splitp30k,rdw8p30k,champ30k").split(",")
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP/"undershoot.json")))
N_CLIPS, F, K = 20, 100, 4

def main():
    import v7tiny_g2 as G
    from tanitad.models.flagship_v15 import SPEED_SCALE
    dev = torch.device("cuda")
    clips = sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]
    present = [a for a in ARMS if (SP/f"v7tiny_{a}"/"ckpt.pt").is_file()]
    print("\n  E-DEC-47 - DOES THE PREDICTOR UNDER-SHOOT?  alpha* > 1 => yes\n", flush=True)
    print(f"  {'arm':<14}{'alpha*':>9}{'nrmse a=1':>12}{'nrmse a*':>11}{'gain':>9}{'clips':>7}")
    print("  " + "-"*64)
    rep = {"_evidence_class":"MEASURED (ours; dev-box RTX 4060)","eval_tier":"T0-DIAGNOSTIC",
           "method":"alpha fit on FIT clips, applied LOEO to the held-out clip","arms":{}}
    for arm in present:
        w, st = G.load_arm(arm, dev); W = int(w.window)
        per = []
        with torch.no_grad():
            for c in clips:
                z, act, spd = G.encode_clip(w, c, dev, F); zt = z.float()
                D_, T_ = [], []
                for i in range(0, len(zt)-W-K, 2):
                    win = zt[i:i+W][None].to(dev).clone()
                    base = zt[i+W-1].to(dev).reshape(-1)
                    aa = act[i:i+W][None].to(dev).float()
                    if aa.shape[1] != W: continue
                    vv = (spd[i]/SPEED_SCALE).view(1,1,1).expand(1,W,1).to(dev)
                    o = w.predictor(win, torch.cat([aa,vv],-1))[1].reshape(-1)[:zt.shape[1]]
                    D_.append((o-base).cpu().numpy()); T_.append((zt[i+W].to(dev)-base).cpu().numpy())
                if len(D_) > 10: per.append((np.stack(D_), np.stack(T_)))
        del w; torch.cuda.empty_cache()
        if len(per) < 6: continue
        a1, aS = [], []
        for i in range(len(per)):
            fit = [per[j] for j in range(len(per)) if j != i]
            d = np.concatenate([x[0] for x in fit]); t = np.concatenate([x[1] for x in fit])
            alpha = float((d*t).sum()/max((d*d).sum(),1e-30))     # LS optimum on FIT only
            dte, tte = per[i]
            tn = float(np.linalg.norm(tte))
            a1.append(float(np.linalg.norm(dte-tte))/max(tn,1e-30))
            aS.append(float(np.linalg.norm(alpha*dte-tte))/max(tn,1e-30))
        A = []
        for i in range(len(per)):
            fit = [per[j] for j in range(len(per)) if j != i]
            d = np.concatenate([x[0] for x in fit]); t = np.concatenate([x[1] for x in fit])
            A.append(float((d*t).sum()/max((d*d).sum(),1e-30)))
        am, n1, ns = float(np.mean(A)), float(np.mean(a1)), float(np.mean(aS))
        rep["arms"][arm] = {"step":int(st),"alpha_star":round(am,4),
                            "nrmse_alpha1":round(n1,4),"nrmse_alpha_star":round(ns,4),
                            "gain":round(n1-ns,4),"n_clips":len(per),
                            "reading":"UNDER-SHOOT" if am>1.1 else ("OVER-shoot" if am<0.9 else "well-scaled")}
        print(f"  {arm:<14}{am:>9.4f}{n1:>12.4f}{ns:>11.4f}{n1-ns:>9.4f}{len(per):>7}", flush=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
