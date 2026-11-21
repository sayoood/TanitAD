#!/usr/bin/env python3
"""Noise floor for the attribution refits: the SAME MLP (mlp_cfg) refit at seeds 1 and 2 for each design (seed 0 is refit_attribution.py's, the banked mlp_ego's seed).
Everything else identical (folds, holdout fold 0, early stopping on fold 0). Answers: would another MLP INIT/ORDER seed move the design ranking? CPU, 4 threads. No eval window."""
import json
import sys
import time

sys.path.insert(0, "F:/TanitAD-scratch/speedsub_t1_prep")
import refit_attribution as R  # noqa: E402  (sets the 4-thread env before numpy/torch)
import numpy as np  # noqa: E402

sr = R.load_instr()
z = np.load(R.TRAIN)
ego, Y, M, clips = z["ego"], z["y"], z["m"], z["clip"]
E = ego.astype(np.float32)
em, es = sr.standardiser(E)
Ez = (E - em) / es
folds = sr.fold_of(clips)
tr, ho = np.flatnonzero(folds != sr.HOLDOUT_FOLD), np.flatnonzero(folds == sr.HOLDOUT_FOLD)
sd = np.array([np.std(Y[M[:, c], c]) for c in range(5)])
out = {"schema": "speedsub-t1-prep-seeds-v1", "instrument_sha256": R.INSTR_SHA, "train_sha256": R.sha_file(R.TRAIN), "seeds": {}}
t0 = time.time()
for seed in (1, 2):
    cfg = dict(sr.MLP_CFG, seed=seed)
    for name, groups in R.DESIGNS.items():
        D_ = np.ascontiguousarray(Ez[:, R.cols_of(groups)])
        m = sr.mlp_fit(D_, Y, M, tr, ho, cfg)
        ho_sd = sr.masked_rmse(sr.mlp_predict(m, D_[ho]), Y[ho], M[ho]) / sd
        out["seeds"].setdefault(str(seed), {})[name] = {"best_epoch": int(m["best_epoch"]), "holdout_rmse_sd": [float(v) for v in ho_sd], "mean_sd": float(ho_sd.mean())}
        print(f"[seed {seed}] {name} ep {m['best_epoch']} mean {ho_sd.mean():.5f} {np.round(ho_sd, 4).tolist()} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(out, open(f"{R.OUT}/refit_seeds.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("[done]", flush=True)
