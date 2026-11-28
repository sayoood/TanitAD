"""Map-signal audit, task 2: the ANALYTIC per-class signal of the map loss.

Function class and assumptions (stated, not hidden):
  * The quantity analysed is the gradient of the loss w.r.t. the LOGITS z (the map head's output),
    per cell: soft CE  dL/dz = q - p ; hard CE  dL/dz = w_y (q - e_y).  Everything upstream (head,
    BEV encoder, lift, trunk) receives this vector through the SAME Jacobian for every class, so the
    per-class split AT THE LOGITS is the per-class split of the signal the whole branch receives,
    up to that Jacobian (task 3 measures the split at the lift output on the real model).
  * Class attribution: p sums to 1, so  q - p = sum_c p_c (q - e_c).  Class c's part of a cell's
    gradient is p_c (q - e_c); its size is p_c * ||q - e_c||_2.  Summed over supervised cells this
    is A_c; the share is A_c / sum A.  Two companions: the OWN-LOGIT PULL P_c = sum p_c (1 - q_c)
    (the only term that raises class c), and the LOSS share L_c = sum p_c (-log q_c).
  * States of the model (q), each an explicit function of the label, no training involved:
      S0 uniform init      q_c = 1/C everywhere
      S1 constant prior    q = the global label mass m (the best spatially-blind predictor)
      S2 drivable-majority q = (1-eps) p~ + eps/C,  p~ = p with ALL thin-class mass (lane, crosswalk,
                           arrow/text, edge, hatched) moved to DRIVABLE; eps = 0.05.  A model that is
                           right about the big classes (and not-seen) and never predicts a thin one:
                           the refcv6@35k picture (drivable IoU 0.67, lane 0.009).
      S3 blurred oracle    q = the label itself, Gaussian-blurred by sigma (a model with the right
                           information and a localisation error sigma): the DECISION-RULE ceiling.
  * Data: the eval kit's 137 SAM3 GT files (the only GT on this machine) -- a PROXY for the TRAIN
    distribution the weights must come from.  Every 10th frame (the census's sampling).
  * 0.5 m / soft / 9 classes = refcv6 as trained: supervised cells = seen (not-seen <= 0.5, the
    reader's integer rule) AND the lift's map_valid (stride 16, 416x1024, the per-clip extrinsics,
    the bottom-43-row observed mask the LiftGeometryBank carried).  The not-seen channel IS a class.
  * 10 cm / hard / 8 classes = NEW-2 as designed: supervised cells = fine_codes != 255.  Median-
    frequency weights w_c = median(f)/f_c clipped at 25, under the TWO definitions in use:
      MF-global  f_c = n_c / N_seen
      MF-present f_c = n_c / (seen cells of the frames where c is present)   (Eigen & Fergus 2015)
Output: raw/analytic_class_signal.json.  CPU; reads GT only; no model.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "raw" / "analytic_class_signal.json"
KIT = Path("D:/refcv6_eval_kit/data")
EVERY = 10
KEEP_EVERY = 50
EPS = 0.05
C9 = ["seen-no-class", "drivable", "lane/road line", "crosswalk", "arrow/text", "non-drivable edge",
      "hatched", "sidewalk/verge", "not-seen"]
C8 = C9[:8]
THIN = [2, 3, 4, 5, 6]
DRV = 1
STATES = ("S0_uniform", "S1_constant_prior", "S2_drivable_majority")
MASKS = ("seen", "seen_and_lift_valid")


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def lift_tools():
    import tanitad
    assert "msa_tree_" in tanitad.__file__, tanitad.__file__
    import torch
    from tanitad.models.bev_lift import build_lift_geometry
    from tanitad.models.trunk_shapes import frame_for_width
    frame = frame_for_width(1024, 416)
    obs = torch.ones(frame.height, frame.width, dtype=torch.bool)
    obs[-43:, :] = False
    extr = json.load(open(KIT / "refcv6_train_eval139_extrinsics.json", encoding="utf-8"))
    return {sha12(k): e for k, e in extr.items()}, frame, obs, build_lift_geometry


def soft_q(p: np.ndarray, state: str, m_global: np.ndarray) -> np.ndarray:
    C, n = p.shape
    if state == "S0_uniform":
        return np.full_like(p, 1.0 / C)
    if state == "S1_constant_prior":
        return np.repeat(m_global[:, None], n, axis=1)
    pt = p.copy()
    pt[DRV] += pt[THIN].sum(0)
    pt[THIN] = 0.0
    return (1 - EPS) * pt + EPS / C


def attrib(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """-> [3, C]: rows A (attributed gradient size), P (own-logit pull), L (loss), summed over cells."""
    C = p.shape[0]
    q = np.clip(q, 1e-12, 1.0)
    qq = (q * q).sum(0)
    out = np.zeros((3, C))
    for c in range(C):
        nrm = np.sqrt(np.maximum(qq - 2 * q[c] + 1.0, 0.0))        # ||q - e_c||
        out[0, c] = float((p[c] * nrm).sum())
        out[1, c] = float((p[c] * (1 - q[c])).sum())
        out[2, c] = float((p[c] * -np.log(q[c])).sum())
    return out


def iou_rules(q: np.ndarray, lab: np.ndarray, mask: np.ndarray, rules: dict, C: int) -> dict:
    """q [C,H,W] posterior; each rule = weights w (None = plain argmax); -> {rule: (I[C], U[C])}."""
    out = {}
    for name, w in rules.items():
        sc = q if w is None else q * np.asarray(w, dtype=q.dtype)[:, None, None]
        pred = sc.argmax(0)
        I, U = np.zeros(C), np.zeros(C)
        for c in range(C):
            pc, lc = (pred == c) & mask, (lab == c) & mask
            I[c] = (pc & lc).sum()
            U[c] = (pc | lc).sum()
        out[name] = (I, U)
    return out


def main() -> None:
    t0 = time.time()
    files = sorted(glob.glob(str(KIT / "sam3_gt_eval_thor137" / "*.sam3mapgt.npz")))
    assert len(files) == 137, len(files)
    extr12, frame, obs, blg = lift_tools()
    lv_by = {}
    for f in files:
        s12 = Path(f).name.split(".")[0]
        lv_by[s12] = blg(extr12[s12], frame=frame, stride=16, observed=obs).valid.any(0).numpy()
    # ------------------------------------------------------------ pass 1: masses + 10 cm counts
    mass = {k: np.zeros(9) for k in MASKS}
    ncell = {k: 0 for k in MASKS}
    n_fine = np.zeros(8, np.int64)
    n_pres_den = np.zeros(8, np.int64)
    n_band = np.zeros((3, 8), np.int64)
    n_frames = 0
    fine_keep, coarse_keep = [], []
    for fi, f in enumerate(files):
        s12 = Path(f).name.split(".")[0]
        lv = lv_by[s12]
        z = np.load(f)
        cf = z["cart_frac"]
        fc = z["fine_codes"]
        for t in range(0, cf.shape[0], EVERY):
            u8 = cf[t].astype(np.int32)
            seen = 2 * (255 - u8[8]) >= 255                  # semantic_map_gt.py:330-331
            p = u8.astype(np.float64) / 255.0
            p = p / np.maximum(p.sum(0, keepdims=True), 1e-6)  # map_soft_ce renormalisation
            for key, m in (("seen", seen), ("seen_and_lift_valid", seen & lv)):
                mass[key] += p[:, m].sum(1)
                ncell[key] += int(m.sum())
            c = fc[t]
            sn = c != 255
            bc = np.bincount(c[sn].ravel(), minlength=8)[:8]
            n_fine += bc
            n_pres_den += np.where(bc > 0, int(sn.sum()), 0)
            for b, (lo, hi) in enumerate(((0, 200), (200, 400), (400, 600))):
                cb = c[lo:hi]
                n_band[b] += np.bincount(cb[cb != 255].ravel(), minlength=8)[:8]
            if t % KEEP_EVERY == 0:
                fine_keep.append(c[:200].copy())
                coarse_keep.append((p[:, :40].astype(np.float32), (seen & lv)[:40].copy()))
            n_frames += 1
        del z, cf, fc
        if fi % 25 == 0:
            print(f"[analytic] pass1 {fi + 1}/137 ({time.time() - t0:.0f} s)", flush=True)
    m_glob = {k: mass[k] / max(ncell[k], 1) for k in MASKS}
    # ------------------------------------------------------------ pass 2: 0.5 m soft attribution
    acc = {k: {s: np.zeros((3, 9)) for s in STATES} for k in MASKS}
    argm = {k: {s: np.zeros(9) for s in STATES} for k in MASKS}
    ent = {k: np.zeros(9) for k in MASKS}
    for fi, f in enumerate(files):
        s12 = Path(f).name.split(".")[0]
        lv = lv_by[s12]
        cf = np.load(f)["cart_frac"]
        for t in range(0, cf.shape[0], EVERY):
            u8 = cf[t].astype(np.int32)
            seen = 2 * (255 - u8[8]) >= 255
            p = u8.astype(np.float64) / 255.0
            p = p / np.maximum(p.sum(0, keepdims=True), 1e-6)
            for key, m in (("seen", seen), ("seen_and_lift_valid", seen & lv)):
                pm = p[:, m]
                ent[key] += (pm * -np.log(np.clip(pm, 1e-12, 1))).sum(1)
                for s in STATES:
                    q = soft_q(pm, s, m_glob[key])
                    acc[key][s] += attrib(pm, q)
                    argm[key][s] += np.bincount(q.argmax(0), minlength=9)
        del cf
        if fi % 25 == 0:
            print(f"[analytic] pass2 {fi + 1}/137 ({time.time() - t0:.0f} s)", flush=True)
    res: dict = {"n_files": len(files), "n_frames": n_frames, "every_nth_frame": EVERY,
                 "eps_S2": EPS, "classes_9": C9, "thin_classes": [C9[i] for i in THIN]}
    soft = {}
    for key in MASKS:
        o = {"n_cells": ncell[key], "label_mass_share": m_glob[key].tolist(),
             "soft_target_entropy_floor_mean": float(ent[key].sum() / ncell[key]),
             "soft_target_entropy_floor_by_class": (ent[key] / ncell[key]).tolist()}
        for s in STATES:
            a = acc[key][s]
            o[s] = {"share_A_attributed_grad": (a[0] / a[0].sum()).tolist(),
                    "share_P_own_logit_pull": (a[1] / a[1].sum()).tolist(),
                    "share_L_loss": (a[2] / a[2].sum()).tolist(),
                    "mean_loss_per_cell": float(a[2].sum() / ncell[key]),
                    "argmax_pred_share": (argm[key][s] / ncell[key]).tolist()}
        soft[key] = o
    res["soft_0p5m_refcv6"] = soft
    ms = np.array(soft["seen_and_lift_valid"]["label_mass_share"])
    res["controls"] = {
        "S0_shares_equal_mass_share_max_abs_diff": float(max(
            np.abs(np.array(soft["seen_and_lift_valid"]["S0_uniform"][k]) - ms).max()
            for k in ("share_A_attributed_grad", "share_P_own_logit_pull", "share_L_loss"))),
        "S0_loss_per_cell": soft["seen_and_lift_valid"]["S0_uniform"]["mean_loss_per_cell"],
        "S0_loss_per_cell_want_log9": math.log(9),
        "S1_loss_per_cell": soft["seen_and_lift_valid"]["S1_constant_prior"]["mean_loss_per_cell"],
        "S1_loss_per_cell_want_H(mass)": float(-(ms * np.log(ms)).sum()),
    }
    # ------------------------------------------------------------ 10 cm hard, NEW-2
    N = int(n_fine.sum())
    f_glob = n_fine / N
    f_pres = n_fine / np.maximum(n_pres_den, 1)
    wg_raw = np.median(f_glob) / f_glob
    wp_raw = np.median(f_pres) / f_pres
    W = {"none": np.ones(8), "MF_global_clip25": np.minimum(wg_raw, 25.0),
         "MF_present_clip25": np.minimum(wp_raw, 25.0)}
    hard = {"n_seen_cells": N, "class_share_of_seen": f_glob.tolist(),
            "class_share_by_band_0_20_40_60": (n_band / n_band.sum(1, keepdims=True)).tolist(),
            "f_present": f_pres.tolist(),
            "weights_raw_unclipped": {"MF_global": wg_raw.tolist(), "MF_present": wp_raw.tolist()},
            "weights": {k: v.tolist() for k, v in W.items()}}
    C = 8
    tq = 0.1 / (C - 1)
    big = math.sqrt(0.1 ** 2 + (C - 1) * tq ** 2)                     # ||q - e_c||, q_c = 0.9
    thin = math.sqrt(0.9 ** 2 + (1 - tq) ** 2 + (C - 2) * tq ** 2)    # q on drivable = 0.9
    n0 = math.sqrt((1 - 1 / C) ** 2 + (C - 1) / C ** 2)
    for wn, w in W.items():
        A0 = w * n_fine * n0
        nr = np.array([thin if c in THIN else big for c in range(C)])
        A2 = w * n_fine * nr
        L2 = w * n_fine * np.array([-math.log(tq) if c in THIN else -math.log(0.9) for c in range(C)])
        hard[wn] = {"S0_uniform_share": (A0 / A0.sum()).tolist(),
                    "S2_drivable_majority_share": (A2 / A2.sum()).tolist(),
                    "S2_loss_share": (L2 / L2.sum()).tolist(),
                    "bayes_rule_beats_drivable_iff_Pc_over_Pdrv_gt": (w[DRV] / w).tolist(),
                    "two_class_posterior_threshold_vs_drivable": (w[DRV] / (w + w[DRV])).tolist()}
    res["hard_10cm_new2"] = hard
    # ------------------------------------------------------------ S3: decision-rule ceilings 0-20 m
    ceil = {"what": "q = Gaussian-blurred label (renormalised) = a model with the right information "
                    "and localisation error sigma; prediction = argmax_c w_c q_c (w = the weights a "
                    "weighted-CE-trained model tilts its softmax by); IoU vs the label argmax; "
                    f"rig x 0-20 m only; every {KEEP_EVERY}th frame",
            "n_frames": len(fine_keep), "10cm": {}, "0p5m": {}}
    rules10 = {"argmax_unweighted": None, "argmax_MF_global": W["MF_global_clip25"],
               "argmax_MF_present": W["MF_present_clip25"]}
    for sig in (0.0, 0.05, 0.10, 0.20, 0.30):
        tot = {r: [np.zeros(8), np.zeros(8)] for r in rules10}
        for c in fine_keep:
            sn = c != 255
            oh = np.stack([((c == j) & sn).astype(np.float32) for j in range(8)])
            q = oh if sig == 0 else np.stack([gaussian_filter(oh[j], sigma=sig / 0.1, mode="nearest")
                                              for j in range(8)])
            q = q / np.maximum(q.sum(0, keepdims=True), 1e-12)
            for r, (I, U) in iou_rules(q, np.where(sn, c, 255), sn, rules10, 8).items():
                tot[r][0] += I
                tot[r][1] += U
        for r in rules10:
            I, U = tot[r]
            ceil["10cm"][f"sigma_{sig:.2f}m|{r}"] = {C8[j]: (round(float(I[j] / U[j]), 4) if U[j] else None)
                                                     for j in range(8)}
        print(f"[analytic] 10 cm ceiling sigma {sig} ({time.time() - t0:.0f} s)", flush=True)
    for sig in (0.0, 0.25, 0.5, 1.0):
        I, U = np.zeros(9), np.zeros(9)
        for p20, m20 in coarse_keep:
            lab = p20.argmax(0)
            q = p20 if sig == 0 else np.stack([gaussian_filter(p20[j], sigma=sig / 0.5, mode="nearest")
                                               for j in range(9)])
            q = q / np.maximum(q.sum(0, keepdims=True), 1e-12)
            (i_, u_), = iou_rules(q, lab, m20, {"a": None}, 9).values()
            I += i_
            U += u_
        ceil["0p5m"][f"sigma_{sig:.2f}m|soft_argmax"] = {C9[j]: (round(float(I[j] / U[j]), 4) if U[j] else None)
                                                         for j in range(9)}
    res["decision_rule_ceilings_0_20m"] = ceil
    res["elapsed_s"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["controls"], indent=1))
    print("done", res["elapsed_s"], "s")


if __name__ == "__main__":
    main()
