"""S1: fit P10 on refcv8 TRAIN windows (kinematics + futures only) for ONE horizon (S2 or S6). Run both in parallel.

usage: fit_p10_refcv.py S2|S6
Writes C:/Users/Admin/s1_floors_scratch/p10_<H>.pt (net state, stats, best) and raw/train_<H>_meta.json.
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

import torch                                                                         # noqa: E402
H = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SUF = "" if SEED == 0 else f"_seed{SEED}"
torch.set_num_threads(int(os.environ.get("S1_THREADS", "3")))
sel, n_fine = (L.H_S2, 20) if H == "S2" else (L.H_S6, 60)

t = time.time()
tman = L.load_manifest(L.TRAIN_MAN)
eman = L.load_manifest(L.EVAL_MAN)
overlap = set(tman["clip_id"]) & set(eman["clip_id"])
assert not overlap, f"train/eval clip overlap: {len(overlap)}"
W = L.refcv_train_windows(tman, sel, stride=5, first=7)
kin, g, ep = W["kin"], W["g"], W["ep"]
base = L.refcv_floors(kin, n_fine, sel)["kd_x"]
kf = fk.kin_features(kin)
y = (g - base).reshape(len(g), -1)
meta = {"horizon": H, "instants_frames": sel, "n_fine_ticks": n_fine, "train_manifest_md5": tman["md5"],
        "n_train_episodes": int(len(np.unique(ep))), "n_train_windows": int(len(g)),
        "window_rule": f"t0 = 7 + (e mod 5) + 5j for every t0 with t0 + {max(sel)} <= T-1 (stride 5 frames = 0.5 s)",
        "train_eval_clip_overlap": 0, "base": "kd_x on the native 0.1 s polyline, grid instants selected",
        "features": "kin_features((v0, a0, steer0)) -- 8 features, unchanged from fit_mlp_kin",
        "ade_base_train_m": float(fk.ade(base, g).mean())}
print(f"[fit_p10 {H}] windows={len(g)} episodes={meta['n_train_episodes']} base ADE (train)={meta['ade_base_train_m']:.4f} "
      f"prep {time.time()-t:.0f}s", flush=True)
t = time.time()
net, st, best = fk.fit_p10(kf, y, base, g, ep, seed=SEED)
meta["training_seed"] = SEED
meta["selection"] = {"wd": float(best[1]), "epochs": int(best[2])}
meta["fit_seconds"] = round(time.time() - t, 1)
pred_tr = fk.predict_p10((net, st, best), kf, base)
meta["ade_p10_train_m"] = float(fk.ade(pred_tr, g).mean())
print(f"[fit_p10 {H}] selected {meta['selection']} fit {meta['fit_seconds']}s  train ADE base->P10: "
      f"{meta['ade_base_train_m']:.4f} -> {meta['ade_p10_train_m']:.4f}", flush=True)
torch.save({"state": net.state_dict(), "xm": st[0], "xs": st[1], "ym": st[2], "best": best, "meta": meta},
           f"{L.SCRATCH}/p10_{H}{SUF}.pt")
json.dump(meta, open(os.path.join(HERE, "..", "raw", f"train_{H}{SUF}_meta.json"), "w"), indent=1)
print(f"ZZFIT-{H}{SUF}-DONEZZ")
