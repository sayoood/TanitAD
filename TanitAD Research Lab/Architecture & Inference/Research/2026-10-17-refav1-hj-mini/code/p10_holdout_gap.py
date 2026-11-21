"""Is P10-600 in-sample on the RA-1 holdout enough to matter? Refit P10 on the first 540 and the first 600 clips (sha12 order,
stride 8, seed 0, steer units -- v8f_mini_fit_p10's exact recipe and code), score both on the RA-1 holdout (clips 541-600),
paired episode-cluster bootstrap over the 60 holdout clips. Decision rule fixed BEFORE running: |delta ADE6| < 0.0213 m
(P10 training-seed floor, S1) => keep the shared P10-600; else switch RA-1 to P10-540 (pre-data amendment)."""
import json, os, sys, time
import numpy as np, torch
TREE = sys.argv[1]; OUT = sys.argv[2]
for p in (os.path.join(TREE, "stack"), os.path.join(TREE, "taniteval"), os.path.join(TREE, "stack", "scripts")):
    sys.path.insert(0, p)
import v8f_mini_fit_p10 as F
from tanitad.data import v8f_mini_data as D
from taniteval import floors_kin as fk
t0 = time.time()
m = torch.load("C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt", weights_only=False)
byc = {c: D.full_from_manifest(p.numpy(), ac.numpy()) for c, p, ac in zip(m["clip_id"], m["poses"], m["actions"])}
order = sorted(byc, key=D.sha12)
hold = order[540:600]
picker = [l.strip() for l in open(os.path.join(TREE, "TanitAD Research Lab", "Architecture & Inference", "Research",
          "2026-10-17-refav1-hj-mini", "raw", "s4b", "clips600.txt")) if l.strip()]
res = {"holdout_equals_picker_last60": picker[540:600] == hold, "picker_first600_equals_sha12_prefix": picker == order[:600],
       "n_picker": len(picker)}
kh, _yh, bh, gh, grph = F.build_rows(byc, hold, 8, "steer")
ade = {}
for n in (540, 600):
    kf, y, base, g, grp = F.build_rows(byc, order[:n], 8, "steer")
    fit = fk.fit_p10(kf, y, base, g, grp, seed=0)
    ade[n] = fk.ade(fk.predict_p10(fit, kh, bh), gh)
    res[f"p10_{n}"] = {"best": [int(fit[2][0]), float(fit[2][1]), int(fit[2][2])], "holdout_ade6_m": float(ade[n].mean())}
d = ade[600] - ade[540]
rng = np.random.default_rng(0); cl = np.unique(grph); idx = {c: np.where(grph == c)[0] for c in cl}
bs = [np.concatenate([d[idx[c]] for c in rng.choice(cl, len(cl))]).mean() for _ in range(2000)]
res.update(n_windows=int(len(d)), n_clusters=int(len(cl)), delta_600_minus_540_ade6_m=float(d.mean()),
           ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           kd_x_holdout_ade6_m=float(fk.ade(bh, gh).mean()), threshold_m=0.0213,
           decision=("KEEP P10-600" if abs(d.mean()) < 0.0213 else "SWITCH RA-1 TO P10-540"), seconds=time.time() - t0,
           estimator="paired episode-cluster bootstrap over the 60 holdout clips, B=2000 (episode variance only)")
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1))
