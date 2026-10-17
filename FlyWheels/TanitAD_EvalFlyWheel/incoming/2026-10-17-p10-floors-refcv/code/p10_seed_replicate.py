"""S1: the TRAINING-seed variance of the floor itself (H-ESTIM-SEED-1 discipline).

P10 is an MLP fit once; a separated `os - P10` interval answers "another draw of EPISODES?", never "another P10
TRAINING RUN?". Here P10 is refit with training seed 1 (same windows, same hyper-parameter search, only the init/batch order
differ) and the bar is re-read against it:  `P10_s1 - P10_s0` (the floor's own seed floor) and `os - P10_s1`.

usage: p10_seed_replicate.py   (needs p10_<H>_seed1.pt from `fit_p10_refcv.py <H> 1`)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_lib as L                                                                   # noqa: E402
from taniteval import floors_kin as fk                                               # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as pb                      # noqa: E402

import torch                                                                         # noqa: E402
RAW = os.path.join(HERE, "..", "raw")
eman = L.load_manifest(L.EVAL_MAN)
out = {"LABEL": "P10 TRAINING-seed replicate (seed 1 vs the primary seed 0); same TRAIN windows, same search"}
for H, sel, n_fine, tag in (("S6", L.H_S6, 60, "s6_s0"), ("S2", L.H_S2, 20, "dump_s0")):
    ck = torch.load(f"{L.SCRATCH}/p10_{H}_seed1.pt", weights_only=False)
    W = L.refcv_eval_windows(eman, L.BATT, tag, sel)
    fl = L.refcv_floors(W["kin"], n_fine, sel)
    net = fk._make_mlp(8, len(sel) * 2)
    net.load_state_dict(ck["state"])
    kf = fk.kin_features(W["kin"])
    p1 = fk.predict_p10((net, (ck["xm"], ck["xs"], ck["ym"]), ck["best"]), kf, fl["kd_x"].astype(np.float32))
    a0 = np.load(os.path.join(RAW, f"arms_{H}_s0.npz"))
    a1 = np.load(os.path.join(RAW, f"arms_{H}_s1.npz"))
    g, ep = a0["g"].astype(np.float64), a0["ep"]
    p0 = a0["P10"].astype(np.float64)
    ad = {"P10_s0": fk.ade(p0, g), "P10_s1": fk.ade(p1, g)}
    fd = {"P10_s0": fk.fde(p0, g), "P10_s1": fk.fde(p1, g)}
    r = {"selection_seed1": ck["meta"]["selection"], "mean_ade": {k: float(v.mean()) for k, v in ad.items()},
         "P10_s1_minus_P10_s0": {"ade": pb(ad["P10_s1"], ad["P10_s0"], ep, n_boot=2000, seed=0),
                                 "fde": pb(fd["P10_s1"], fd["P10_s0"], ep, n_boot=2000, seed=0)},
         "training_seed_floor_abs_mean_ade_diff_m": float(abs(ad["P10_s1"].mean() - ad["P10_s0"].mean()))}
    for s, a in ((0, a0), (1, a1)):
        o = a["os"].astype(np.float64)
        r[f"os_inf_s{s}_minus_P10_s1"] = {"ade": pb(fk.ade(o, g), ad["P10_s1"], ep, n_boot=2000, seed=0),
                                          "fde": pb(fk.fde(o, g), fd["P10_s1"], ep, n_boot=2000, seed=0)}
    if H == "S6":
        d0 = r["os_inf_s0_minus_P10_s1"]["ade"]["delta"]
        r["margin_over_training_seed_floor"] = (abs(d0) / r["training_seed_floor_abs_mean_ade_diff_m"]
                                                if r["training_seed_floor_abs_mean_ade_diff_m"] > 0 else None)
    out[H] = r
    print(H, json.dumps({k: (v if not isinstance(v, dict) or "delta" not in v else v["delta"]) for k, v in r.items() if k != "selection_seed1"})[:600])
json.dump(out, open(os.path.join(RAW, "p10_seed_replicate.json"), "w"), indent=1)
print("ZZSEEDREP-DONEZZ")
