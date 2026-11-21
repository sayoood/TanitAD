"""CRUISE tokens of the full ego-probe run, by the analysis's own mask (analyze_ego_probe.py:96) -> tokens_cruise.json."""
import json, sys, numpy as np
sys.path.insert(0, "."); from analyze_ego_probe import gt_speed
Z = np.load("full/results.npz", allow_pickle=False)
ego = Z["ego_real"]; v0, ax0 = ego[:, 6], ego[:, 2]; gv = Z["gt_valid"].astype(bool)
vgt = np.where(gv[:, None], gt_speed(Z["gt_pose20"], Z["gt_t20"]), np.nan)
cruise = gv & (v0 > 5.0) & (np.abs(ax0) < 0.3) & (np.abs(vgt[:, 2] - v0) < 1.0)
toks = [str(t) for t in Z["token"][cruise]]
json.dump({"tokens": toks, "source": "full/results.npz CRUISE mask (analyze_ego_probe.py:96)"}, open("tokens_cruise.json", "w"), indent=0)
print("cruise tokens", len(toks), "logs", len(set(str(l) for l in Z["log"][cruise])))
