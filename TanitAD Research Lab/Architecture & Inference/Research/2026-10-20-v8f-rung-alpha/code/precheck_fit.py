"""PRE-CHECK step 1: fit P10-pose on the refcv8 TRAIN windows with the S1 recipe (kinematics + futures only), ONE horizon.

usage: precheck_fit.py S2|S6 [seed]    -> C:/Users/Admin/v8fa1/pre/p10pose_<H>[_seedN].pt + fit meta json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                   # noqa: E402
import p10pose_np as L                                                               # noqa: E402
from taniteval import floors_kin as fk                                               # noqa: E402
import torch                                                                         # noqa: E402

H = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SUF = "" if SEED == 0 else f"_seed{SEED}"
torch.set_num_threads(int(os.environ.get("S1_THREADS", "4")))
sel, n_fine = (L.H_S2, 20) if H == "S2" else (L.H_S6, 60)
OUT = "C:/Users/Admin/v8fa1/pre"
os.makedirs(OUT, exist_ok=True)
t = time.time()
tman, eman = L.load_manifest(L.TRAIN_MAN), L.load_manifest(L.EVAL_MAN)
overlap = set(tman["clip_id"]) & set(eman["clip_id"])
assert not overlap, f"train/eval clip overlap: {len(overlap)}"
W = L.train_windows(tman, sel)
kin, g, ep = W["kin"], W["g"], W["ep"]
base = L.pose_floors(kin, n_fine, sel)["kd_x_pose"]
kf = fk.kin_features(kin)
y = (g - base).reshape(len(g), -1)
meta = {"horizon": H, "instants_frames": sel, "train_manifest_md5": tman["md5"],
        "n_train_episodes": int(len(np.unique(ep))), "n_train_windows": int(len(g)), "train_eval_clip_overlap": 0,
        "window_rule": f"t0 = 7 + (e mod 5) + 5j for every t0 with t0 + {max(sel)} <= T-1",
        "features": "kin_features((v0, a0_1frame, kappa0_pose)) -- POSE ONLY",
        "base": "kd_x_pose = retime(damp50(ha0, ha0_ext_pose), ha0_ext_pose) on the native 0.1 s polyline",
        "ade_base_train_m": float(fk.ade(base, g).mean())}
print(f"[fit {H}] windows={len(g)} eps={meta['n_train_episodes']} base ADE(train)={meta['ade_base_train_m']:.4f} "
      f"prep {time.time()-t:.0f}s", flush=True)
t = time.time()
net, st, best = fk.fit_p10(kf, y, base, g, ep, seed=SEED)
meta["training_seed"] = SEED
meta["selection"] = {"wd": float(best[1]), "epochs": int(best[2])}
meta["fit_seconds"] = round(time.time() - t, 1)
pred = fk.predict_p10((net, st, best), kf, base)
meta["ade_p10pose_train_m"] = float(fk.ade(pred, g).mean())
print(f"[fit {H}] selected {meta['selection']} {meta['fit_seconds']}s train ADE {meta['ade_base_train_m']:.4f} -> "
      f"{meta['ade_p10pose_train_m']:.4f}", flush=True)
torch.save({"state": net.state_dict(), "xm": st[0], "xs": st[1], "ym": st[2], "best": best, "meta": meta},
           f"{OUT}/p10pose_{H}{SUF}.pt")
json.dump(meta, open(f"{OUT}/p10pose_{H}{SUF}_meta.json", "w"), indent=1)
print(f"ZZFIT-{H}{SUF}-DONEZZ")
