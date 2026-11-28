#!/usr/bin/env python3
"""Independent check of the saved pre-head features: logits == score_head(features) with P2's own score_head weights (fp16 input rounding admitted).
Also: the table / proposals.npz / features agree row for row.   C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe p2ho_featcheck.py --out featcheck.json"""
import argparse, json, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--final", default="C:/Users/Admin/p2heldout_work/final")
ap.add_argument("--ckpt", default="D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt")
a = ap.parse_args()
sd = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=False)
sd = sd.get("model", sd) if isinstance(sd, dict) else sd
keys = [k for k in sd if k.endswith("score_head.weight") or k.endswith("score_head.bias")]
W = sd[[k for k in keys if k.endswith("weight")][0]].float().numpy()
B = sd[[k for k in keys if k.endswith("bias")][0]].float().numpy()
T, F, Pz = (np.load(f"{a.final}/{n}", allow_pickle=False) for n in ("table_p2_heldout.npz", "prehead_features_p2_heldout.npz", "proposals.npz"))
feat = F["prehead_features"].astype(np.float32)
rec = feat @ W.T + B
d = np.abs(rec - T["logits"].astype(np.float32))
R = {"score_head_keys": keys, "W_shape": list(W.shape), "feat_shape": list(feat.shape), "max_abs_logit_diff_fp16_input": float(d.max()), "mean_abs_logit_diff": float(d.mean()),
     "logit_abs_max": float(np.abs(T["logits"]).max()), "tokens_equal_table_features": bool((T["token"] == F["token"]).all()), "tokens_equal_table_proposals": bool((T["token"] == Pz["token"]).all()),
     "proposals_equal_table": bool(np.array_equal(T["proposals"], Pz["proposals"])), "fingerprints_equal": bool((T["fingerprint"] == Pz["fingerprint"]).all()),
     "negative_control_shuffled_rows_max_abs": float(np.abs((np.roll(feat, 1, 0) @ W.T + B) - T["logits"]).max())}
R["PASS"] = bool(R["max_abs_logit_diff_fp16_input"] < 0.05 and R["tokens_equal_table_features"] and R["tokens_equal_table_proposals"] and R["proposals_equal_table"]
                 and R["negative_control_shuffled_rows_max_abs"] > 10 * R["max_abs_logit_diff_fp16_input"])
json.dump(R, open(a.out, "w"), indent=1)
print(json.dumps(R, indent=1)[:900])
