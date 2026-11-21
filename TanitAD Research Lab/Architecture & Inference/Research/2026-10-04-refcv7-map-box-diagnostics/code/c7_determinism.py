"""C7 -- inference determinism of the map and box heads: the in-run 128 windows run twice (two processes) at
ckpt.pt. Compares every per-episode map count (both rules) and every slot's presence logit.
Usage: python c7_determinism.py <out dir> <result json>"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

o = Path(sys.argv[1])
a = torch.load(o / "inrun_final.acc.pt", map_location="cpu", weights_only=False)
b = torch.load(o / "inrun_final_rep.acc.pt", map_location="cpu", weights_only=False)
res = {"map": {}, "box": {}}
for d in ("pc", "raw"):
    for s in ("pred", "gt", "inter"):
        x, y = a["acc"][d][s].numpy(), b["acc"][d][s].numpy()
        res["map"][f"{d}.{s}"] = {"max_abs_diff": float(np.abs(x - y).max()), "total": float(x.sum())}
for k in ("phat", "q"):
    res["map"][f"hist.{k}"] = {"n_cells_moved_bins": int((a["hist"][k] - b["hist"][k]).abs().sum()) // 2,
                              "n_cells": int(a["hist"][k].sum())}
pa = pickle.load(open(o / "inrun_final.packs.pkl", "rb"))
pb = pickle.load(open(o / "inrun_final_rep.packs.pkl", "rb"))
for h in pa:
    la = np.concatenate([p["logit"] for p in pa[h]]).astype(np.float64)
    lb = np.concatenate([p["logit"] for p in pb[h]]).astype(np.float64)
    res["box"][h] = {"n_slots": int(la.size), "max_abs_logit_diff": float(np.abs(la - lb).max()),
                     "n_slots_differing": int((la != lb).sum()),
                     "n_gate_flips_0p5": int(((la >= 0) != (lb >= 0)).sum())}
mx_map = max(v["max_abs_diff"] for v in res["map"].values() if "max_abs_diff" in v)
mx_box = max(v["max_abs_logit_diff"] for v in res["box"].values())
moved = sum(v["n_cells_moved_bins"] for k, v in res["map"].items() if k.startswith("hist"))
res["summary"] = (f"map counts max |diff| {mx_map:g}; histogram cells changing bin {moved}; box presence logits "
                  f"max |diff| {mx_box:.2e} over {sum(v['n_slots'] for v in res['box'].values())} slots, gate flips "
                  f"{sum(v['n_gate_flips_0p5'] for v in res['box'].values())}")
res["verdict"] = ("PASS (bit-identical)" if (mx_map == 0 and mx_box == 0 and moved == 0) else
                  "NOT bit-identical -- magnitude in `raw/C7_determinism.json`")
Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
print(res["summary"], "|", res["verdict"])
