"""WP-C: cut the two small committed fixtures the unit tests fit on (no clip / episode id of any kind is written).

  refcv8_box_packs_train_every9th.npz   every 9th TRAIN window (of the route package's 1,074 labelled train_s0 packs) x 2 heads,
                                        the SAME layout as refcv7_box_packs_eval_every9th.npz  (the NMS / gate FIT fixture)
  refcv8_map_hist_train_pooled.npz      the TRAIN-DIAG score histogram of ``logit(p_hat_c)`` pooled over the 5 range bands,
                                        [8, 2, 4000] int64 (GT-negative / GT-positive counts) + the 8 frozen class weights
                                        (the map-threshold FIT fixture)

Run:  python bank_fixtures.py --route-packs <train_s0.packs.pkl> --diag-acc <train_fitpass.acc.pt> --out-dir <fixtures dir>
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-packs", required=True)
    ap.add_argument("--diag-acc", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stride", type=int, default=9)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(a.route_packs, "rb") as fh:
        packs = pickle.load(fh)
    arrs = {}
    n_win = None
    for hd in ("box3d", "agent"):
        sub = packs[hd][:: a.stride]
        n_win = len(sub)
        gmax = max(len(p["gt_xy"]) for p in sub)
        pmax = max(len(p["pair_err"]) for p in sub)

        def pad(v, n, dtype):
            o = np.zeros((n,) + np.asarray(v).shape[1:], dtype=dtype)
            o[: len(v)] = v
            return o
        arrs[f"{hd}__logit"] = np.stack([p["logit"] for p in sub]).astype(np.float32)
        arrs[f"{hd}__xy"] = np.stack([p["xy"] for p in sub]).astype(np.float32)
        for k, dt in (("cls", np.int16), ("cls_corr", np.int16), ("matched", bool), ("exempt", bool)):
            arrs[f"{hd}__{k}"] = np.stack([p[k] for p in sub]).astype(dt)
        arrs[f"{hd}__gt_xy"] = np.stack([pad(p["gt_xy"], gmax, np.float32) for p in sub])
        arrs[f"{hd}__gt_cls"] = np.stack([pad(p["gt_cls"], gmax, np.int16) for p in sub])
        for k in ("pos", "ign", "hidden"):
            arrs[f"{hd}__{k}"] = np.stack([pad(p[k], gmax, bool) for p in sub])
        for k in ("pair_err", "pair_size_err", "pair_z_err"):
            arrs[f"{hd}__{k}"] = np.stack([pad(p[k], pmax, np.float64) for p in sub])
        arrs[f"{hd}__n_gt"] = np.asarray([len(p["gt_xy"]) for p in sub], np.int32)
        arrs[f"{hd}__n_pair"] = np.asarray([len(p["pair_err"]) for p in sub], np.int32)
    arrs["meta"] = np.asarray(
        f"refcv7-r101-s0 step 50400, route package train_s0 packs, every {a.stride}th of 1074 labelled TRAIN windows "
        f"({n_win} windows); ids removed; layout = refcv7_box_packs_eval_every9th.npz")
    np.savez_compressed(out / "refcv8_box_packs_train_every9th.npz", **arrs)
    import torch
    acc = torch.load(a.diag_acc, map_location="cpu", weights_only=False)
    h = acc["hist"]["phat"].sum(dim=1).numpy().astype(np.int64)
    np.savez_compressed(out / "refcv8_map_hist_train_pooled.npz", hist_phat=h, class_weight=acc["class_weight"].numpy(),
                        meta=np.asarray("refcv7-r101-s0 step 50400 train_fitpass.acc.pt hist.phat summed over the 5 bands; "
                                        "[8 classes, 2 (GT-neg, GT-pos), 4000 bins over logit [-20, 20]]"))
    print("wrote", [p.name for p in sorted(out.glob("refcv8_*.npz"))], [p.stat().st_size for p in sorted(out.glob("refcv8_*.npz"))])


if __name__ == "__main__":
    main()
