"""e0f_probe.py -- E0f: anchor drift + TemporalFuse older-frame weight norms of the refcv8 EMA-48,852 checkpoint (REFCV9_DESIGN 4.1 row E0f).  CPU, reads one checkpoint.

    python e0f_probe.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --out raw/E0F_RESULT.json

* anchor drift: `_perception.box_refpts.xy` [300, 2] metres vs its INIT.  The init is reproduced from source (`LearnedRefPoints.__init__`, stack/tanitad/models/slot_query_select.py:
  torch.Generator().manual_seed(0); x = rand(k) * x_fwd_m; y = (rand(k) * 2 - 1) * y_half_m), grid (x_fwd_m, y_half_m) = config refcv6_perception.planner_crop_m = (60, 16).
  Control: the reproduced init is checked to lie inside the grid, and a freshly built module is not available here -> the init reproduction itself is UNVERIFIED against a
  built module (only against the source recipe); the drift is therefore reported with that caveat.
* TemporalFuse (`core.encoder.fuse16/32.proj`, concat1x1, K = 3): weight [c, K*c, 1, 1]; block K-1 (the newest frame) was the identity at init, every older block exactly zero
  (stack/tanitad/models/timm_trunk.py TemporalFuse).  Reported: Frobenius norm of the older blocks (frame 0 = oldest, frame 1) and of the newest block's deviation from I.
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    cfg = json.load(open(a.config, encoding="utf-8"))["refcv6_perception"]
    k = int(cfg["n_queries"])
    xf, yh = (float(v) for v in cfg["planner_crop_m"])
    g = torch.Generator().manual_seed(0)
    init = torch.stack([torch.rand(k, generator=g) * xf, (torch.rand(k, generator=g) * 2 - 1) * yh], dim=-1)
    xy = sd["_perception.box_refpts.xy"].float()
    d = (xy - init).norm(dim=-1)
    res = {"step": int(ck["step"]), "kind": ck["kind"], "n_queries": k, "grid_x_fwd_m": xf, "grid_y_half_m": yh,
           "init_recipe": "LearnedRefPoints seed 0 (source recipe, NOT verified against a built module)",
           "init_inside_grid": bool((init[:, 0] >= 0).all() and (init[:, 0] <= xf).all() and (init[:, 1].abs() <= yh).all()),
           "trained_inside_grid": bool((xy[:, 0] >= 0).all() and (xy[:, 0] <= xf).all() and (xy[:, 1].abs() <= yh).all()),
           "anchor_drift_m": {"mean": float(d.mean()), "p95": float(torch.quantile(d, 0.95)), "max": float(d.max()), "median": float(d.median()),
                              "share_moved_gt_1m": float((d > 1.0).float().mean()), "mean_abs_dx": float((xy[:, 0] - init[:, 0]).abs().mean()),
                              "mean_abs_dy": float((xy[:, 1] - init[:, 1]).abs().mean())},
           "temporal_fuse": {}}
    for nm in ("fuse16", "fuse32"):
        w = sd[f"core.encoder.{nm}.proj.weight"].float()[:, :, 0, 0]
        c = w.shape[0]
        kk = w.shape[1] // c
        blocks = [w[:, i * c:(i + 1) * c] for i in range(kk)]
        eye = torch.eye(c)
        res["temporal_fuse"][nm] = {"c": c, "K": kk, "frob_older_frame0_oldest": float(blocks[0].norm()), "frob_older_frame1": float(blocks[1].norm()),
                                    "frob_newest": float(blocks[-1].norm()), "frob_newest_minus_identity": float((blocks[-1] - eye).norm()),
                                    "frob_identity": float(eye.norm()), "older_over_newest": float(torch.cat(blocks[:-1], 1).norm() / blocks[-1].norm()),
                                    "bias_norm": float(sd[f"core.encoder.{nm}.proj.bias"].float().norm()),
                                    "older_blocks_exactly_zero": bool(all(float(b.abs().max()) == 0.0 for b in blocks[:-1]))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
