"""refcv7 NEW-2 -- what a PERFECT 0.5 m map scores at 10 cm (the ceiling of any 0.5 m head).

The 0.5 m GT itself (``cart_frac``, argmax over the 8 class channels -- the refcv6
baseline hook's exact rule) is nearest-upsampled onto the 10 cm grid and scored
against ``fine_codes`` with ``taniteval.map_hires_metrics``: per class x band pooled
IoU and the 0.2 m tolerance P / R / F1, clip-cluster bootstrap. Every eval-kit file,
every 10th frame (the census's sample). Two companion arms on the SAME windows:
``all_drivable`` (a constant) and, on the ODD half of the clips, the positional prior
FITTED on the EVEN half (a dry run of the control's mechanics -- the real control is
fitted on Thor's TRAIN GT). No model, no GPU; this is a property of the label grids.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from taniteval import map_hires_metrics as M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    files = sorted(a.gt_dir.glob("*.sam3mapgt.npz"))
    even = [p for i, p in enumerate(files) if i % 2 == 0]
    prior = M.PositionalPrior()
    for p in even:                                  # FIT on the even half
        with np.load(p, allow_pickle=False) as z:
            prior.add(p.name[:12], z["fine_codes"][::a.stride])
    pmap_fp = prior.fingerprint()
    t_all = M.WindowTable(arms=("oracle_05m", "all_drivable"))
    t_odd = M.WindowTable(arms=("oracle_05m", "prior_fit_even"))
    const = np.full((600, 320), 1, np.uint8)
    for i, p in enumerate(files):
        s12 = p.name[:12]
        with np.load(p, allow_pickle=False) as z:
            fine = z["fine_codes"]
            cart = z["cart_frac"]
        for t in range(0, fine.shape[0], a.stride):
            gt = fine[t]
            o = M.coarse_to_fine_codes(cart[t])
            t_all.add(s12, gt, {"oracle_05m": o, "all_drivable": const})
            if i % 2 == 1:
                t_odd.add(s12, gt, {"oracle_05m": o,
                                    "prior_fit_even": prior.predict_for(s12)})
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(files)} files, {time.time() - t0:.0f} s", flush=True)
    s_all = M.summarize(t_all, n_boot=a.n_boot)
    s_odd = M.summarize(t_odd, n_boot=a.n_boot)
    d_prior = {name: M.paired_delta(t_odd, "oracle_05m", "prior_fit_even", met, k, 0,
                                    n_boot=a.n_boot)
               for name, (met, k) in {"lane_iou_0_20": ("iou", 2),
                                      "crosswalk_iou_0_20": ("iou", 3),
                                      "edge_F1_0_20": ("F1", 5)}.items()}
    rec = {"what": "the 0.5 m SAM3 GT (cart_frac argmax over 8 classes, the refcv6 "
                   "baseline hook's rule), nearest-upsampled, scored at 10 cm",
           "n_files": len(files), "frame_stride": a.stride,
           "all_clips": s_all, "odd_clips_vs_prior_fit_even": s_odd,
           "oracle_minus_prior_paired": d_prior, "prior_fingerprint": pmap_fp,
           "elapsed_s": round(time.time() - t0, 1)}
    a.out.write_text(M.to_json(rec), encoding="utf-8")
    keep = {}
    for k, v in s_all["arms"]["oracle_05m"].items():
        if v.get("mean") is not None:
            keep[k] = (v["mean"], v.get("lo"), v.get("hi"))
    print(json.dumps({"n_windows": s_all["n_windows"], "n_clips": s_all["n_clips"],
                      "oracle_05m": keep}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
