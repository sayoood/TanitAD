"""Per-token mean IoU over [0, 40) m with a log-cluster bootstrap (proxy_eval.boot, 10,000 resamples, seed 20260927 = the readout's estimator), from the head_pred npz files.
usage: token_iou.py <out.json> ARM=head_pred.npz [...]   (labels: the pinned navtest store on E:)"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

import p7_adapter  # noqa: E402,F401  (puts the instrument tree on sys.path and asserts its location)
import aux_s1_readout as R  # noqa: E402

assert os.path.abspath(R.__file__).replace("\\", "/").startswith(p7_adapter.EVAL), R.__file__
out = sys.argv[1]
src = R.open_labels("navtest", "E:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest")
res = {"estimator": "per-token IoU = inter/union over cells with 0<=x<40 m (cells 4..23 of 40) per frame; mean over tokens with union>0; log-cluster bootstrap proxy_eval.boot n=10000 seed 20260927 (another draw of LOGS only)", "arms": {}}
for spec in sys.argv[2:]:
    name, _, path = spec.partition("=")
    p = R.load_head_pred(path)
    st = R.frame_stats(p, src)
    b = 3                                                            # K2_BAND [0, 40)
    ok = st["valid"] & (st[f"union{b}"] > 0)
    iou = {str(i): float(st[f"inter{b}"][i] / st[f"union{b}"][i]) for i in np.flatnonzero(ok)}
    logs = {str(i): p["keys"][i][0] for i in np.flatnonzero(ok)}
    m, lo, hi = R.PE.boot(iou, logs, n=R.N_BOOT, seed=R.SEED)
    pooled = float(st[f"inter{b}"][st["valid"]].sum() / max(st[f"union{b}"][st["valid"]].sum(), 1e-12))
    res["arms"][name] = {"n_frames": len(p["keys"]), "n_labelled": int(st["valid"].sum()), "n_with_union": int(ok.sum()), "n_logs": len(set(logs.values())),
                         "per_token_mean_iou": m, "ci95": [lo, hi], "pooled_iou_0_40": pooled, "median_token_iou": float(np.median(list(iou.values()))) if iou else None,
                         "share_tokens_iou_ge_0.80": float(np.mean([v >= 0.80 for v in iou.values()])) if iou else None}
    print(name, json.dumps(res["arms"][name]), flush=True)
json.dump(res, open(out, "w", encoding="utf-8"), indent=1, sort_keys=True)
