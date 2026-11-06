"""Column-LR settlement probe (PREREG_YAWFT section 2). MECHANICS ONLY: u400's decoder reads the ep015 cache's scene_ctx (a DIFFERENT encoder: MEASURED in YF cmp_ckpt.out), so absolute losses are not
quotable and no accuracy is claimed. What it measures is the one thing the choice needs: how far a zero-initialised column 4 travels under the registered schedule (400 steps, cosine, AdamW wd 0) for each
lr, against the analytic Adam bound  norm <= sqrt(256) * lr * steps / 2  (cosine integrates to steps/2), and how much of the yaw-conditioned VAL loss gap each lr opens.
    python probe_col_lr.py   # CPU; writes probe_col_lr.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yaw_common as YC
import yaw_ft as FT

CACHE = YC.DATA / "refe_proxy" / "cache_train_ep015"
OUT = HERE / "lr_probe"
STEPS = 400
res = {"note": "ep015 scene_ctx (different encoder) with u400's decoder: mechanics only", "steps": STEPS, "accum": 1, "batch": 32, "grid": {}}
for lr in (1e-4, 3e-4, 1e-3, 3e-3):
    out = OUT / f"B_lr{lr:g}"
    t0 = time.time()
    rc = FT.main(["--cache", str(CACHE), "--manifest", str(HERE / "manifest_yaw_4k.json"), "--yaw-table", str(YC.TABLE_NAVTRAIN), "--arm", "B", "--rung", "R1", "--seed", "0",
                  "--steps", str(STEPS), "--accum", "1", "--batch", "32", "--lr-col", str(lr), "--val-every", "100", "--gate-frames", "256", "--out", str(out)])
    s = json.load(open(out / "summary.json")) if rc == 0 else {}
    mets = [json.loads(l) for l in open(out / "metrics.jsonl")] if rc == 0 else []
    res["grid"][str(lr)] = {"rc": rc, "seconds": round(time.time() - t0), "analytic_norm_bound": 16.0 * lr * STEPS / 2.0, "col4_norm_final": s.get("col4_norm"),
                            "val_final": s.get("val_final"), "val_final_yaw0": s.get("val_final_yaw0"), "val_start": s.get("val_start"),
                            "col4_norm_trace": [(m["step"], round(m["col4_norm"], 4)) for m in mets],
                            "val_trace": [(m["step"], round(m["val"]["loss"], 4), round(m["val_yaw0"]["loss"], 4)) for m in mets]}
    YC.jdump(res, HERE / "probe_col_lr.json")
print("ZZPROBE_OK")
