"""select_clips.py -- the PRE-DECLARED clip-selection rule (written BEFORE any render; the rule is also in VIZ_REPORT.md).

RULE (seed 0, banked own-code ADE6 = eval_M2b_s0-kind-own.npz['ade6'], T0 open loop, 141 eps / 2,399 windows):
  * for each stratum in (JUNCTION, CURVE, NO-TURN) [window_strata_eval141.npz, PREREG_V8F_KIND Amendment 1]: sort that stratum's windows by
    (ade6, window index) ascending and take position (n-1)//2  (the lower median, deterministic);
  * plus the single window with the largest ade6 over all 2,399 (ties -> lowest index) as the FAILURE case;
  * render the WHOLE CLIP that contains each selected window.  If two selections fall in the same clip the next-ranked is NOT substituted:
    the clip is rendered once and both windows are marked.
  No window is excluded, no clip is swapped after looking at a render.
"""
import json, numpy as np
S="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz"
bk=np.load(S+"/work/banked/eval_M2b_s0-kind-own.npz"); st=np.load("C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-kind/window_strata_eval141.npz",allow_pickle=True)
grid=json.load(open(S+"/work/inputs/eval_grid_windows.json"))
order=grid["order"]; wins=[(c,t) for c in order for t in grid["windows"][c]]
ade=bk["ade6"]; stra=st["stratum"]; names=["JUNCTION","CURVE","NO-TURN"]
assert len(wins)==len(ade)==len(stra)==2399
sel={}
for k,n in enumerate(names):
    ix=np.nonzero(stra==k)[0]; o=sorted(ix, key=lambda i:(ade[i],i)); i=int(o[(len(o)-1)//2]); sel[n]=i
    print(n,"n=",len(ix),"median-pick window",i,"clip",wins[i][0],"t",wins[i][1],"ade6 %.3f"%ade[i], "stratum median(np) %.3f"%np.median(ade[ix]))
iw=int(np.argmax(ade)); sel["FAILURE(worst)"]=iw
print("WORST window",iw,wins[iw],"ade6 %.3f"%ade[iw],"stratum",names[stra[iw]])
out={n:{"window":i,"clip":wins[i][0],"t":int(wins[i][1]),"ade6_seed0_banked":float(ade[i]),"stratum":names[stra[i]],"ade6_seed1_banked":None} for n,i in sel.items()}
b1=np.load(S+"/work/banked/eval_M2b_s1-kind-own.npz")["ade6"]
for n,i in sel.items(): out[n]["ade6_seed1_banked"]=float(b1[i])
out["_meta"]={"n_windows":2399,"all_ade6_mean_seed0":float(ade.mean()),"all_ade6_median_seed0":float(np.median(ade))}
json.dump(out,open(S+"/out/selection.json","w"),indent=1)
print(json.dumps(out,indent=1))
