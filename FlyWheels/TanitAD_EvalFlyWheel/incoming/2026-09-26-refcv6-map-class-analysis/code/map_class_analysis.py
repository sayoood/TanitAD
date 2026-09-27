#!/usr/bin/env python3
"""map_class_analysis.py -- what does refcv6's SAM3 map head actually predict, per class?

PI (Sayed), 2026-09-26, after the map video, verbatim: *"the drivable area are little bit ok, but our
model is seldomly detecting other semantic classes, no roadmarkings, no roadmarking infrastructure in
intersections etc... We need to analyze it"*.

Input: the per-cell dump written by ``taniteval/tools/render_refcv6_map_video.py --boxes --dump-map``
(same process and model load as the boxes video, step 38,000): per window, softmax(map_logits) as
float16 ``[9, 120, 64]``, the GT fractions as the label file's uint8 (x255), ``map_seen``,
``map_valid``, plus a meta row (set membership, sha12, GT-path heading change over the valid 6 s).

Everything is computed on the SCORED cells, ``map_seen & map_valid`` -- the trainer's own cell set
(``refc_v3_train.py:4508-4510``). Every number carries its n.

Answers, in the brief's order:
  1. GT prevalence per class, three ways: argmax of the fractions; fraction >= 0.5; fraction > 0.
  2. The model per class: argmax prevalence; IoU / precision / recall under the trainer's rule
     (p >= 0.5 vs fraction >= 0.5); threshold-free ROC-AUC of p(class) against fraction >= 0.5 AND
     fraction > 0; mean p on GT-positive vs GT-negative cells; mean p vs mean GT fraction (the
     soft-CE calibration read).
  3. By range bin (0-10, 10-20, 20-40, 40-60 m) and intersection (heading change > 30 deg) vs straight
     (<= 5 deg) windows, for the thin classes.
  5. Resolution: which stride-16 feature pixel each BEV cell samples (the lift's own grid), per range
     bin; and the fraction of GT lane-line cells the lift reaches (map_valid) per range bin.

Controls (must read known values): a CONSTANT score reads AUC exactly 0.5; the GT fraction itself as
the score reads AUC exactly 1.0 against fraction >= 0.5; a SHUFFLED-cell score reads ~0.5.

    python map_class_analysis.py --dump C:/Users/Admin/qland/work/mapvid/mapdump_<tag> --out raw/
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

CHANNELS = ("seen, no map class", "drivable", "lane / road line", "crosswalk", "arrow / text",
            "non-drivable edge", "hatched area", "sidewalk / verge", "not seen")
THIN = (2, 3, 4, 5, 6)                      # the marking / infrastructure classes
RANGE_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0))
POS_THR_U8 = 128                            # frac >= 0.5  <=>  u8 >= 128 (127/255 = 0.498)
X_CELL = (np.arange(120) + 0.5) * 0.5       # row i -> x metres ahead
Y_CELL = -16.0 + (np.arange(64) + 0.5) * 0.5


def auc(scores: np.ndarray, labels: np.ndarray):
    """Exact ROC-AUC (Mann-Whitney U) with ties at average rank. None when a class is empty."""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    s = np.asarray(scores, dtype=np.float64)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    end = np.cumsum(cnt)
    avg_rank = end - (cnt - 1) / 2.0        # 1-based average rank of each tie group
    r = avg_rank[inv]
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ap_bin(scores: np.ndarray, labels: np.ndarray):
    """Average precision, sklearn's definition: sum_k (R_k - R_{k-1}) P_k over DISTINCT thresholds,
    so a constant score reads exactly the base rate. None when there is no positive."""
    labels = np.asarray(labels, bool).ravel()
    n_pos = int(labels.sum())
    if n_pos == 0 or labels.size == 0:
        return None
    s = np.asarray(scores, np.float64).ravel()
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    tp, fp = tp[last], fp[last]
    rec = tp / n_pos
    prec = tp / np.maximum(tp + fp, 1)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Chebyshev dilation by r cells of a bool [N, X, Y] mask (per window, no wrap)."""
    if r <= 0:
        return mask.copy()
    n, X, Y = mask.shape
    pad = np.pad(mask, ((0, 0), (r, r), (r, r)))
    out = np.zeros_like(mask)
    for dx in range(2 * r + 1):
        for dy in range(2 * r + 1):
            out |= pad[:, dx:dx + X, dy:dy + Y]
    return out


def load(dump: Path):
    meta = [json.loads(ln) for ln in open(dump / "meta.jsonl", encoding="utf-8") if ln.strip()]
    P, Fr, S, V = [], [], [], []
    for f in sorted(glob.glob(str(dump / "chunk_*.npz"))):
        z = np.load(f)
        P.append(z["probs"])
        Fr.append(z["frac"])
        S.append(z["seen"])
        V.append(z["valid"])
    return meta, np.concatenate(P), np.concatenate(Fr), np.concatenate(S), np.concatenate(V)


def class_block(p: np.ndarray, fu8: np.ndarray, c: int, rng: np.random.Generator,
                do_controls: bool = False) -> dict:
    """``p`` [n, 9] float, ``fu8`` [n, 9] uint8, over scored cells. One class's full readout."""
    pc = p[:, c].astype(np.float64)
    f = fu8[:, c]
    pos_thr, pos_any = f >= POS_THR_U8, f > 0
    pos_arg = fu8.argmax(1) == c
    pred_thr = pc >= 0.5
    inter = int((pred_thr & pos_thr).sum())
    union = int((pred_thr | pos_thr).sum())
    out = {
        "n_cells": int(pc.size),
        "gt_prevalence": {"argmax": float(pos_arg.mean()), "frac_ge_0.5": float(pos_thr.mean()),
                          "frac_gt_0": float(pos_any.mean()),
                          "n_argmax": int(pos_arg.sum()), "n_frac_ge_0.5": int(pos_thr.sum()),
                          "n_frac_gt_0": int(pos_any.sum())},
        "gt_label_mass_mean": float(f.astype(np.float64).mean() / 255.0),
        "model_mean_p": float(pc.mean()),
        "model_argmax_prevalence": float((p.argmax(1) == c).mean()),
        "model_p_ge_0.5_prevalence": float(pred_thr.mean()),
        "trainer_rule": {"iou": inter / union if union else None,
                         "precision": inter / int(pred_thr.sum()) if pred_thr.any() else None,
                         "recall": inter / int(pos_thr.sum()) if pos_thr.any() else None,
                         "tp": inter, "n_pred": int(pred_thr.sum()), "n_gt": int(pos_thr.sum())},
        "argmax_rule": {"precision": (float(((p.argmax(1) == c) & pos_arg).sum() /
                                            max((p.argmax(1) == c).sum(), 1))),
                        "recall": (float(((p.argmax(1) == c) & pos_arg).sum() / pos_arg.sum())
                                   if pos_arg.any() else None)},
        "auc": {"vs_frac_ge_0.5": auc(pc, pos_thr), "vs_frac_gt_0": auc(pc, pos_any),
                "vs_argmax": auc(pc, pos_arg)},
        "ap": {"vs_frac_ge_0.5": ap_bin(pc, pos_thr), "base_frac_ge_0.5": float(pos_thr.mean()),
               "vs_frac_gt_0": ap_bin(pc, pos_any), "base_frac_gt_0": float(pos_any.mean())},
        # calibration read: the soft CE's optimum is q_c == the cell's GT fraction; a head that SEES
        # the class tracks the fraction bin by bin, a blind one is flat
        "calibration_by_gt_frac": [
            {"gt_frac_bin": f"[{lo / 255:.2f},{hi / 255:.2f})" if hi <= 255 else f"[{lo / 255:.2f},1.00]",
             "n": int(m_.sum()),
             "mean_p": float(pc[m_].mean()) if m_.any() else None,
             "mean_gt_frac": float(f[m_].astype(np.float64).mean() / 255.0) if m_.any() else None,
             "frac_p_ge_0.5": float((pc[m_] >= 0.5).mean()) if m_.any() else None,
             "frac_argmax_is_c": float((p[m_].argmax(1) == c).mean()) if m_.any() else None}
            for lo, hi in ((0, 1), (1, 64), (64, 128), (128, 192), (192, 256))
            for m_ in [(f >= lo) & (f < hi)]],
        "mean_p_on_pos": {"frac_ge_0.5": float(pc[pos_thr].mean()) if pos_thr.any() else None,
                          "frac_gt_0": float(pc[pos_any].mean()) if pos_any.any() else None},
        "mean_p_on_neg": {"frac_lt_0.5": float(pc[~pos_thr].mean()) if (~pos_thr).any() else None,
                          "frac_eq_0": float(pc[~pos_any].mean()) if (~pos_any).any() else None},
        "p_quantiles_on_frac_gt_0": ([float(q) for q in np.quantile(pc[pos_any], [0.5, 0.9, 0.99])]
                                     if pos_any.any() else None),
        "max_p": float(pc.max()) if pc.size else None,
        "gt_frac_quantiles_on_frac_gt_0": ([float(q) / 255.0 for q in
                                            np.quantile(f[pos_any].astype(np.float64), [0.5, 0.9, 0.99])]
                                           if pos_any.any() else None),
    }
    if do_controls:
        out["controls"] = {
            "constant_score_auc": auc(np.full_like(pc, float(pos_thr.mean())), pos_thr)
            if pos_thr.any() else None,
            "gt_fraction_as_score_auc": auc(f.astype(np.float64), pos_thr) if pos_thr.any() else None,
            "shuffled_score_auc": auc(rng.permutation(pc), pos_thr) if pos_thr.any() else None,
            "shuffled_score_auc_frac_gt_0": auc(rng.permutation(pc), pos_any) if pos_any.any() else None,
        }
    return out


def resolution(dump: Path) -> dict:
    """From the lift's OWN grid: per range bin, how many distinct stride-16 feature pixels feed the
    ground-height BEV cells (heights_m[0]), and how many cells share one feature pixel."""
    out = {}
    for f in sorted(glob.glob(str(dump / "lift_geometry_*.npz"))):
        s12 = Path(f).stem.split("_")[-1]
        z = np.load(f)
        g, v = z["grid"], z["valid"].astype(bool)          # [Z, X, Y, 2], [Z, X, Y]
        fh, fw = [int(x) for x in z["feat_hw"]]
        kx = ((g[0, ..., 0] + 1.0) * fw - 1.0) / 2.0          # feature-pixel coords (align_corners=False)
        ky = ((g[0, ..., 1] + 1.0) * fh - 1.0) / 2.0
        vv = v[0]
        rows = {}
        for lo, hi in RANGE_BINS:
            sel = (X_CELL[:, None] >= lo) & (X_CELL[:, None] < hi) & np.ones((1, 64), bool)
            m = sel & vv
            if not m.any():
                rows[f"{lo:g}-{hi:g}m"] = {"n_cells": int(sel.sum()), "n_valid": 0}
                continue
            pix = set(zip(np.rint(kx[m]).astype(int).tolist(), np.rint(ky[m]).astype(int).tolist()))
            # ground footprint of ONE feature pixel along the forward axis: metres per feature row
            xs, ks = X_CELL[np.nonzero(m)[0]], ky[m]
            slope = np.polyfit(ks, xs, 1)[0] if np.ptp(ks) > 0 else float("nan")
            rows[f"{lo:g}-{hi:g}m"] = {"n_cells": int(sel.sum()), "n_valid": int(m.sum()),
                                       "n_distinct_feature_pixels": len(pix),
                                       "cells_per_feature_pixel": round(m.sum() / max(len(pix), 1), 2),
                                       "metres_per_feature_row_along_x": round(abs(float(slope)), 3)}
        out[s12] = {"feat_hw": [fh, fw], "heights_m": [float(h) for h in z["heights_m"]],
                    "by_range_ground_height": rows}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    dump, out_dir = Path(a.dump), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, P, Fu, S, V = load(dump)
    n_w = len(meta)
    if P.shape[0] != n_w:
        raise SystemExit(f"meta rows {n_w} != dumped windows {P.shape[0]}")
    scored = S & V                                           # [N, 120, 64]
    # flatten scored cells, keeping each cell's window index, range and lateral offset
    wi, xi, yi = np.nonzero(scored)
    p = P.transpose(0, 2, 3, 1)[wi, xi, yi].astype(np.float32)        # [n, 9]
    fu = Fu.transpose(0, 2, 3, 1)[wi, xi, yi]                         # [n, 9] uint8
    x_m, y_m = X_CELL[xi], Y_CELL[yi]
    turn = np.asarray([float(m.get("turn_deg", 0.0)) for m in meta])[wi]
    rng = np.random.default_rng(0)
    res = {
        "what": "per-class readout of refcv6's map head on the SCORED cells (map_seen & map_valid)",
        "n_windows": n_w, "n_scored_cells": int(p.shape[0]),
        "n_windows_by_set": {s: sum(1 for m in meta if s in m["set"])
                             for s in ("inrun_eval", "boxes_clip", "map_clip", "inrun_perm_ext")},
        "n_clips": len({m["clip_sha12"] for m in meta}),
        "prob_storage": "float16 softmax; trainer-rule thresholds at 0.5 therefore carry +-5e-4",
        "classes": {},
    }
    for c, name in enumerate(CHANNELS):
        res["classes"][name] = class_block(p, fu, c, rng, do_controls=True)
    # sum-to-one check on the stored GT (u8 rounding) -- the soft CE renormalises
    tot = fu.astype(np.int32).sum(1)
    res["gt_fraction_sum_u8"] = {"min": int(tot.min()), "max": int(tot.max()),
                                 "frac_exactly_255": float((tot == 255).mean())}
    # by range bin and by scene type, thin classes + drivable
    strat = {}
    for lo, hi in RANGE_BINS:
        m = (x_m >= lo) & (x_m < hi)
        strat[f"range_{lo:g}-{hi:g}m"] = {CHANNELS[c]: class_block(p[m], fu[m], c, rng)
                                          for c in (1,) + THIN}
        strat[f"range_{lo:g}-{hi:g}m"]["_n_cells"] = int(m.sum())
    for tag, m in (("intersection_turn_gt_30deg", turn > 30.0), ("straight_turn_le_5deg", turn <= 5.0)):
        strat[tag] = {CHANNELS[c]: class_block(p[m], fu[m], c, rng) for c in (1,) + THIN}
        strat[tag]["_n_cells"] = int(m.sum())
        strat[tag]["_n_windows"] = int(len({int(w) for w in np.unique(wi[m])}))
    res["stratified"] = strat
    # lane cells the lift reaches, per range (over SEEN cells, before the valid cut)
    seen_idx = np.nonzero(S)
    fu_seen_lane = Fu[seen_idx[0], 2, seen_idx[1], seen_idx[2]]
    v_seen = V[seen_idx]
    xs_seen = X_CELL[seen_idx[1]]
    reach = {}
    for lo, hi in RANGE_BINS:
        m = (xs_seen >= lo) & (xs_seen < hi) & (fu_seen_lane > 0)
        reach[f"{lo:g}-{hi:g}m"] = {"n_lane_cells_seen": int(m.sum()),
                                    "frac_lift_valid": float(v_seen[m].mean()) if m.any() else None}
    res["lane_cells_reached_by_lift"] = reach
    # ---- LOCATION PRIOR (leave-own-clip-out): where a class usually is, nothing about THIS scene -- #
    # The scene-blind floor. A head whose AUC/AP does not clear it has learned a map of WHERE classes
    # tend to be, not the classes. Estimated per cell over the scored cells of every OTHER clip.
    clip_of = np.asarray([m["clip_sha12"] for m in meta])
    uclips, cid = np.unique(clip_of, return_inverse=True)
    sc_i = scored.astype(np.int64)
    n_sc_clip = np.zeros((len(uclips), 120, 64), np.int64)
    np.add.at(n_sc_clip, cid, sc_i)
    n_sc_tot = n_sc_clip.sum(0)
    lp = {"what": "per-cell P(label | cell scored) from every OTHER clip (leave-own-clip-out)",
          "n_clips": int(len(uclips)), "classes": {}}
    for c in range(9):
        cls_rows = {}
        for tag, lab in (("frac_ge_0.5", Fu[:, c] >= POS_THR_U8), ("frac_gt_0", Fu[:, c] > 0)):
            pos_clip = np.zeros((len(uclips), 120, 64), np.int64)
            np.add.at(pos_clip, cid, (lab & scored).astype(np.int64))
            prior_grid = (pos_clip.sum(0)[None] - pos_clip) / np.maximum(n_sc_tot[None] - n_sc_clip, 1)
            prior_flat = prior_grid[cid[wi], xi, yi]
            y_ = lab[wi, xi, yi]
            cls_rows[tag] = {"auc_location_prior": auc(prior_flat, y_),
                             "ap_location_prior": ap_bin(prior_flat, y_),
                             "auc_model": res["classes"][CHANNELS[c]]["auc"][f"vs_{tag}"],
                             "ap_model": res["classes"][CHANNELS[c]]["ap"][f"vs_{tag}"],
                             "base_rate": float(y_.mean()), "n_pos": int(y_.sum())}
        lp["classes"][CHANNELS[c]] = cls_rows
    res["location_prior_control"] = lp
    # ---- TOLERANCE: are the head's most-confident cells ON / NEAR a real marking? ----------------- #
    # R-precision at Chebyshev tolerance r cells: take the model's top-k scored cells, k = the number
    # of GT-presence (frac > 0) cells, and count how many lie within r cells of a GT-presence cell.
    # The location prior's top-k is the scene-blind comparator. 1 cell = 0.5 m.
    tol = {}
    for c in THIN:
        gt_any = (Fu[:, c] > 0) & S                                   # presence, before the valid cut
        k = int(gt_any[wi, xi, yi].sum())
        if k == 0:
            tol[CHANNELS[c]] = {"k": 0}
            continue
        pos_clip = np.zeros((len(uclips), 120, 64), np.int64)
        np.add.at(pos_clip, cid, ((Fu[:, c] > 0) & scored).astype(np.int64))
        prior_flat = ((pos_clip.sum(0)[None] - pos_clip) /
                      np.maximum(n_sc_tot[None] - n_sc_clip, 1))[cid[wi], xi, yi]
        row = {"k": k, "n_scored_cells": int(wi.size)}
        dil = {r: dilate(gt_any, r) for r in (0, 1, 2)}
        rng_t = np.random.default_rng(1)
        for name_s, sc in (("model", p[:, c].astype(np.float64)), ("location_prior", prior_flat),
                           ("random", rng_t.random(wi.size))):
            top = np.argsort(-sc, kind="mergesort")[:k]
            row[name_s] = {}
            for r in (0, 1, 2):
                row[name_s][f"within_{r}_cells"] = float(dil[r][wi[top], xi[top], yi[top]].mean())
            row[name_s]["score_at_k"] = float(sc[top[-1]])
        tol[CHANNELS[c]] = row
    res["tolerance_r_precision"] = tol
    res["resolution"] = resolution(dump)
    # the soft-CE literal example (bev_encoder.map_soft_ce: -sum_c p_c log q_c, renormalised p)
    ptarget = np.zeros(9)
    ptarget[2], ptarget[1] = 0.25, 0.75
    ent = float(-(ptarget[ptarget > 0] * np.log(ptarget[ptarget > 0])).sum())

    def ce(q):
        q = np.asarray(q, np.float64)
        return float(-(ptarget * np.log(np.clip(q, 1e-12, 1))).sum())
    q_lane = np.full(9, 0.1 / 7)
    q_lane[2], q_lane[1] = 0.6, 0.3
    res["soft_ce_literal_example"] = {
        "cell": "25 % lane / road line + 75 % drivable",
        "optimum_q": "q = p exactly (Gibbs' inequality): q_lane 0.25, q_drivable 0.75",
        "ce_at_optimum_nats": round(ent, 4),
        "argmax_at_optimum": "drivable", "p_lane_ge_0.5_at_optimum": False,
        "ce_if_head_says_lane_0.6_drivable_0.3": round(ce(q_lane), 4),
        "penalty_for_calling_it_lane_nats": round(ce(q_lane) - ent, 4)}
    res["wall_s"] = round(time.time() - t0, 1)
    p_out = out_dir / "map_class_analysis.json"
    p_out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    # console summary
    print(f"windows {n_w}, scored cells {p.shape[0]:,}")
    for name, b in res["classes"].items():
        tr, au = b["trainer_rule"], b["auc"]
        print(f"{name:20s} GT argmax {b['gt_prevalence']['argmax']:.4f} >=.5 "
              f"{b['gt_prevalence']['frac_ge_0.5']:.4f} >0 {b['gt_prevalence']['frac_gt_0']:.4f} | "
              f"model argmax {b['model_argmax_prevalence']:.4f} meanp {b['model_mean_p']:.4f} "
              f"mass {b['gt_label_mass_mean']:.4f} | IoU {tr['iou']} R {tr['recall']} | AUC>=.5 "
              f"{au['vs_frac_ge_0.5']} AUC>0 {au['vs_frac_gt_0']} | ctl {b['controls']}")
    print(f"wrote {p_out} ({res['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
