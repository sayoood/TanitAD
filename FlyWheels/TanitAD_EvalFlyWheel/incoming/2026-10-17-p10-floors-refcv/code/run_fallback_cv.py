"""S1 sensitivity: the PREREG FALLBACK P10 -- grouped 5-fold CV INSIDE eval139 (folds by episode), every hyper-parameter
fit on the fit folds only.  NOT the verdict (the primary is the TRAIN-fit P10); reported beside it.

usage: run_fallback_cv.py S2|S6      (needs arms_<H>_s{0,1}.npz from run_refcv7_floors.py)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_lib as L                                                                   # noqa: E402
from taniteval import floors_kin as fk                                               # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as pb                      # noqa: E402

import torch                                                                         # noqa: E402
torch.set_num_threads(int(os.environ.get("S1_THREADS", "3")))
H = sys.argv[1]
RAW = os.path.join(HERE, "..", "raw")
sel, n_fine = (L.H_S2, 20) if H == "S2" else (L.H_S6, 60)
tag = "dump_s0" if H == "S2" else "s6_s0"

eman = L.load_manifest(L.EVAL_MAN)
W = L.refcv_eval_windows(eman, L.BATT, tag, sel)
a0 = np.load(os.path.join(RAW, f"arms_{H}_s0.npz"))
a1 = np.load(os.path.join(RAW, f"arms_{H}_s1.npz"))
assert np.array_equal(a0["ws"], W["t0"]) and np.array_equal(a0["ep"], W["ep"])
ep, g = W["ep"], W["d_g"].astype(np.float64)
base = L.refcv_floors(W["kin"], n_fine, sel)["kd_x"].astype(np.float32)
kf = fk.kin_features(W["kin"])
y = (g - base).reshape(len(g), -1)
fold = fk.grouped_folds(ep, 5)
pred = np.empty_like(g)
sel_log = []
t = time.time()
for f in range(5):
    tr, te = fold != f, fold == f
    p, best = fk.fit_predict_p10(kf[tr], y[tr], base[tr], g[tr], ep[tr], kf[te], base[te])
    pred[te] = p
    sel_log.append({"fold": f, "n_fit": int(tr.sum()), "n_held": int(te.sum()), "wd": float(best[1]), "epochs": int(best[2])})
    print(f"[fallback {H}] fold {f} fit={tr.sum()} held={te.sum()} sel={best} {time.time()-t:.0f}s", flush=True)

ade_cv, fde_cv = fk.ade(pred, g), fk.fde(pred, g)
P10 = a0["P10"].astype(np.float64)
res = {"horizon": H, "LABEL": "FALLBACK sensitivity (grouped 5-fold CV inside eval139) -- NOT the verdict",
       "n_windows": int(len(g)), "n_episodes": int(len(np.unique(ep))), "selection_per_fold": sel_log,
       "ade_mean_P10cv": float(ade_cv.mean()), "fde_mean_P10cv": float(fde_cv.mean()),
       "ade_mean_P10_train_fit": float(fk.ade(P10, g).mean()),
       "P10cv_minus_P10trainfit": {"ade": pb(ade_cv, fk.ade(P10, g), ep, n_boot=2000, seed=0),
                                   "fde": pb(fde_cv, fk.fde(P10, g), ep, n_boot=2000, seed=0)}}
for s, a in ((0, a0), (1, a1)):
    o = a["os"].astype(np.float64)
    res[f"os_s{s}_minus_P10cv"] = {"ade": pb(fk.ade(o, g), ade_cv, ep, n_boot=2000, seed=0),
                                   "fde": pb(fk.fde(o, g), fde_cv, ep, n_boot=2000, seed=0)}
np.savez_compressed(os.path.join(RAW, f"fallback_P10cv_{H}.npz"), pred=pred, ep=ep, ws=W["t0"])
json.dump(res, open(os.path.join(RAW, f"fallback_cv_{H}.json"), "w"), indent=1)
print("ZZFALLBACK-%s-DONEZZ" % H, json.dumps({k: res[k] for k in ("ade_mean_P10cv", "ade_mean_P10_train_fit")}))
