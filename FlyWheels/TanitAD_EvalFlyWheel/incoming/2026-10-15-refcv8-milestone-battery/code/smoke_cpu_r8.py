"""smoke_cpu_r8.py -- CPU-ONLY check of everything in the G0 path that does not need the GPU (so it can run while the
shared card is busy): re-pointed loader -> STRICT model load of the refcv8 checkpoint -> the eval dataset WITH the
perception joins and the pose-sync patch -> one item with and without pose-sync (the shift must be visible and tiny)
-> the in-run subset permutation -> the in-run eval row at the step.  No forward pass, no CUDA.

    python smoke_cpu_r8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_5000.pt --config E:/refcv8_eval_kit/ckpt/config.json \
        --metrics <metrics copy> --out E:/refcv8_eval_kit/smoke/smoke_cpu_r8.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_repoint  # noqa: E402

L = r8_repoint.apply()
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rec = {"tool": "smoke_cpu_r8.py", "started": time.strftime("%FT%T"), "cuda_used": False}
    tr = L.trainer()
    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, "cpu")
    sd = mrec["state_dict"]
    rec["model"] = {"step": sd.get("step"), "strict": sd.get("strict"), "missing": len(sd.get("missing") or []),
                    "unexpected": len(sd.get("unexpected") or []), "n_keys": sd.get("n_keys"),
                    "param_breakdown_equal": (mrec.get("param_breakdown") or {}).get("equal"),
                    "r8_emit": mrec.get("r8_emit"),
                    "anchor_file_vs_ckpt_max_abs_diff": {k: v["max_abs_diff"] for k, v in
                                                         (mrec.get("anchor_file_vs_ckpt_buffers") or {}).items()},
                    "argv_remap_sources": {k: v["source"] for k, v in mrec["argv_remap"]["remapped"].items()},
                    "build_s": mrec.get("build_s")}
    print(f"[smoke] model built+loaded strict ({time.time() - t0:.0f}s): {rec['model']}", flush=True)
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True)
    rec["dataset"] = {"n_episodes": drec["n_episodes"], "n_windows": drec["n_windows"],
                      "pose_sync": drec.get("pose_sync"), "keys": sorted(drec.keys()),
                      "label_clock_g3": (drec.get("label_clock") or {}).get("g3", {}).get("g3")
                      if isinstance(((drec.get("label_clock") or {}).get("g3")), dict) else None,
                      "r8_v9": {k: (drec.get("r8_v9") or {}).get(k) for k in ("md5", "n_rows", "lat_variant")}}
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), int(args.batch))
    rec["perm_sha256"] = hashlib.sha256(json.dumps(perm).encode()).hexdigest()
    rec["n_subset"] = len(perm)
    # one item with the correction ON (the dataset above) and the SAME window with it OFF (a plain twin)
    i = int(perm[0])
    a_on = e_ds[i]
    e_ds.pose_sync = None                                    # twin: the very same object with the correction off
    a_off = e_ds[i]
    ps = r8_repoint._STATE["pose_sync"]
    moved = {}
    for k in ("pose_last", "future_poses", "actions"):
        d = (a_on[k].float() - a_off[k].float()).abs()
        moved[k] = {"max_abs_diff": float(d.max()), "shape": list(a_on[k].shape)}
    rec["pose_sync_item_check"] = {"window_index": i, "moved": moved,
                                   "v0_on": float(a_on["pose_last"][3]), "v0_off": float(a_off["pose_last"][3]),
                                   "rule": "ON vs OFF on the same window; pose_last/future_poses must differ (the shift "
                                           "is <= 35 ms of motion) and be tiny"}
    rows = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in rows if r.get("step") == rec["model"]["step"] and "eval_loss" in r]
    rec["inrun_row"] = {"n_rows_at_step": len(ev), "eval_traj": ev[0].get("eval_traj") if ev else None,
                        "eval_loss": ev[0].get("eval_loss") if ev else None,
                        "n_eval_keys": sum(1 for k in (ev[0] if ev else {}) if k.startswith("eval_"))}
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["finished"] = time.strftime("%FT%T")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(rec, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
