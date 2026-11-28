#!/usr/bin/env python3
"""seam_diag.py -- POST-HOC DIAGNOSTIC (not in the card; decides nothing).  Where along the 30-step window does the executed path's |kappa| sit, in the M-class 3-8 m/s band?  The T1-kin executor commits the first
replan_steps = 10 knots of each plan, so the executed path is three 10-step blocks joined at steps 10 and 20 (the seams).  S (SG 7, 2) smooths WITHIN a plan, never across the seams.
Per step index k: mean |kappa_k| over the band windows where the path (and the GT) are pair-valid at k, for Y (banked), X (smoothed), P10 and GT.  Output raw/SEAM_DIAG.json."""
import importlib.util
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
spec = importlib.util.spec_from_file_location("R", f"{HERE}/smooth_t1_read.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
GD = R.load_gd()
sys.path.insert(0, f"{R.TREE}/taniteval")
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", f"{R.TREE}/stack".replace("/", os.sep))
from taniteval import four_families as FF  # noqa: E402

OUT = sys.argv[1]
man = json.load(open(R.MANIFEST, encoding="utf-8"))
res = {"what": "POST-HOC diagnostic: mean |kappa| per step index of the executed path, M-class windows with v0 in [3, 8) m/s (413 windows); seams at steps 10 and 20; decides nothing", "tier": "T1-kin", "runs": {}}
geo = lambda X: FF._seq_geometry(torch.as_tensor(np.asarray(X, np.float64)), 0.2)  # noqa: E731
for k, (tag, arm, seed) in R.RUNS.items():
    if k == "K1OFF":
        continue
    eb = man["arms"][arm][str(seed)]["t1"]
    _, dy = R.load_dump(eb["dump_dir"])
    _, dx = R.load_dump(f"F:/TanitAD-scratch/smooth_t1/out/{tag}/dump_{tag}")
    cls, masks = GD.classify(dy["v0"], dy["g"])
    band = masks["M"] & (dy["v0"] >= R.BAND[0]) & (dy["v0"] < R.BAND[1])
    G = geo(dy["g"])
    per = {}
    for nm, d in (("Y", dy["cl"]), ("X", dx["cl"]), ("P10", dy["fl"]), ("GT", dy["g"])):
        P = geo(d)
        pv = (P["pair_valid"] & G["pair_valid"])[torch.as_tensor(band)]
        kap = P["curvature"].abs()[torch.as_tensor(band)]
        c = pv.sum(0).clamp_min(1)
        per[nm] = [float((torch.where(pv, kap, torch.zeros_like(kap)).sum(0) / c)[i]) for i in range(kap.shape[1])]
    n = len(per["Y"])
    seam = lambda v: float(np.mean([v[i] for i in range(n) if i % 10 in (9, 0, 1) and i >= 9]))        # steps 9,10,11 and 19,20,21 (the knots next to a seam)
    inner = lambda v: float(np.mean([v[i] for i in range(n) if not (i % 10 in (9, 0, 1) and i >= 9)]))  # every other step
    res["runs"][k] = {"n_steps": n, "per_step": per, **{f"{nm}_seam_mean": seam(per[nm]) for nm in per}, **{f"{nm}_inner_mean": inner(per[nm]) for nm in per}}
    print(k, " ".join(f"{nm}: seam {seam(per[nm]):.4f} inner {inner(per[nm]):.4f}" for nm in per))
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print("wrote", OUT)
