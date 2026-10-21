"""perc8_score.py -- the eval-grade BOX read of a perception pass: pooled AP / class-mean AP (4 thresholds), AUROC, the census at
the declared and refitted gates, NMS, boxes per object -- with the EPISODE-CLUSTER BOOTSTRAP, marginal and PAIRED, from the
per-window rows `perc8_rows.py` built with the programme's own matcher.  CPU only.

THE ONE THING THAT IS NOT THE PROGRAMME'S CODE: the bootstrap's inner reducer.  `detection_metrics._ap` pools python tuples and
re-sorts them per draw (3 924 s for refcv7's own perc_score.json).  Here the rows are sorted ONCE (stable, by -conf, window order
= the reference's pooled order) and a draw's clip multiplicities enter as WEIGHTS: a row repeated w times (same conf, same hit)
contributes  w - FPb * (H(TPb+FPb+w) - H(TPb+FPb))  to the all-point AP numerator (H = harmonic numbers; w = 1 gives
(TPb+1)/(TPb+FPb+1) -- the reference's precision at that row).  EXACT up to the order of EQUAL-confidence rows from DIFFERENT
windows inside a resampled draw.  ⛔ Controls (run on every invocation, recorded in the output):
  C1  weights = 1  -> AP equals `detection_metrics._ap` on the pooled list, for the 4 thresholds x all-class (to 1e-9);
  C2  the draw machinery (`np.random.default_rng(seed).choice(uniq, ...)`) is `ci._draws`'s: a recall@gate paired CI computed
      here equals `ci.paired_episode_cluster_bootstrap` with a callable reducer, bit for bit;
  C3  weighted AUROC at weights = 1 equals `detection_metrics.auroc` on the pooled slots.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time

import numpy as np

THR = (0.5, 1.0, 2.0, 4.0)
THR_KEY = ("0p5", "1", "2", "4")
BAND_NAMES = ("all", "0_20", "20_40", "40_60")


def harmonic_table(m):
    h = np.zeros(m + 2, np.float64)
    h[1:] = np.cumsum(1.0 / np.arange(1, m + 2, dtype=np.float64))
    return h


class Rows:
    """Sorted rows of ONE arm on the common windows, ready for weighted AP."""

    def __init__(self, npz, win_map, clip_of_common):
        z = np.load(npz, allow_pickle=False)
        self.npos_win = z["npos"][win_map >= 0]                      # [Wc, 11, 4] (window order = pickle order)
        # common windows in the order of win_map's image (the pickle order of THIS arm filtered)
        self.common_of_win = win_map                                 # arm window idx -> common idx (-1 = absent)
        self.sets = {}
        for ti in range(4):
            for name in ("all", "cls"):
                conf = z[f"t{ti}_{name}_conf"]
                win = z[f"t{ti}_{name}_win"]
                cw = win_map[win]
                keep = cw >= 0
                conf, hit, band, cw = conf[keep], z[f"t{ti}_{name}_hit"][keep], z[f"t{ti}_{name}_band"][keep], cw[keep]
                cc = z[f"t{ti}_{name}_c"][keep] if name == "cls" else None
                order = np.argsort(-conf, kind="stable")
                rec = {"hit": hit[order].astype(np.int8), "clip": clip_of_common[cw[order]].astype(np.int32),
                       "band": band[order]}
                if cc is not None:
                    rec["c"] = cc[order]
                self.sets[(ti, name)] = rec
        # per-(t, class) and per-(t, band) pre-extracted subsets (sorted order preserved)
        self.sub = {}
        for ti in range(4):
            s = self.sets[(ti, "cls")]
            for c in range(10):
                m = s["c"] == c
                self.sub[(ti, "c", c)] = {"hit": s["hit"][m], "clip": s["clip"][m], "band": s["band"][m]}
            a = self.sets[(ti, "all")]
            for b in range(3):
                m = a["band"] == b
                self.sub[(ti, "b", b)] = {"hit": a["hit"][m], "clip": a["clip"][m]}
        self.n_rows_max = max(len(v["hit"]) for v in self.sets.values())


def ap_w(hit, clip, mult, npos, H):
    """Weighted all-point AP (see module docstring)."""
    if npos <= 0:
        return float("nan")
    w = mult[clip]
    tpw = w * hit
    fpw = w - tpw
    ctp = np.cumsum(tpw)
    cfp = np.cumsum(fpw)
    m = tpw > 0
    tpb = (ctp - tpw)[m]
    fpb = (cfp - fpw)[m]
    wm = w[m]
    tf = tpb + fpb
    s = wm - fpb * (H[tf + wm] - H[tf])
    return float(s.sum() / npos)


class ClipSums:
    def __init__(self, per_window, clip_of_common, n_clips):
        pw = np.asarray(per_window, np.float64)
        self.sum = np.zeros((n_clips,) + pw.shape[1:], np.float64)
        np.add.at(self.sum, clip_of_common, pw)

    def tot(self, mult):
        return np.tensordot(mult.astype(np.float64), self.sum, axes=(0, 0))


def pooled_ap_all(rows, ti, mult, npos_tot, H, band=None):
    if band is None:
        s = rows.sets[(ti, "all")]
        return ap_w(s["hit"], s["clip"], mult, npos_tot, H)
    s = rows.sub[(ti, "b", band)]
    return ap_w(s["hit"], s["clip"], mult, npos_tot, H)


def class_aps(rows, ti, mult, npos_cls, H):
    out = []
    for c in range(10):
        s = rows.sub[(ti, "c", c)]
        out.append(ap_w(s["hit"], s["clip"], mult, npos_cls[c], H))
    return out


def map_mean(aps, npos_cls):
    v = [a for a, n in zip(aps, npos_cls) if n > 0 and a == a]
    return float(np.mean(v)) if v else float("nan")


def weighted_auc(inv, ng, w, y):
    """Weighted Mann-Whitney AUROC, ties = 0.5 (average ranks), groups = unique scores."""
    wp = np.bincount(inv, weights=w * y, minlength=ng)
    wn = np.bincount(inv, weights=w * (1.0 - y), minlength=ng)
    cn = np.cumsum(wn) - wn
    n1, n0 = wp.sum(), wn.sum()
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    return float((wp * (cn + 0.5 * wn)).sum() / (n1 * n0))


def pct_ci(v, alpha=0.05):
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cell_marginal(point, boots):
    b = np.asarray(boots, np.float64)
    ok = b[~np.isnan(b)]
    lo, hi = pct_ci(ok) if ok.size else (float("nan"), float("nan"))
    return {"mean": point, "lo": lo, "hi": hi, "se": float(ok.std(ddof=1)) if ok.size > 1 else float("nan"),
            "n_boot_defined": int(ok.size)}


def cell_paired(point_a, point_b, ba, bb):
    d = np.asarray(ba, np.float64) - np.asarray(bb, np.float64)
    ok = d[~np.isnan(d)]
    lo, hi = pct_ci(ok) if ok.size else (float("nan"), float("nan"))
    return {"delta": point_a - point_b, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0),
            "p_delta_gt0": float((ok > 0).mean()) if ok.size else float("nan"), "n_boot_defined": int(ok.size)}


# ---------------------------------------------------------------------------------------------------------------- #
# per-window census (gate, NMS) from the packs                                                                      #
# ---------------------------------------------------------------------------------------------------------------- #
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


CENSUS_COLS = ("tp", "n_conf", "n_pos", "sum_cnt", "n_det", "n_ge2", "acc_raw", "n_acc")


def census_one(pk, gate, nms_cfg=None):
    """[tp, n_conf, n_pos, sum_cnt, n_det, n_ge2, acc_raw, n_acc] + tp_by_class[10] for one window at `gate`."""
    from tanitad.eval import detection_metrics as dm
    from tanitad.eval import detection_nms as nms
    q = pk if nms_cfg is None else nms.nms_pack(pk, nms_cfg["radius_m"], nms_cfg.get("p_floor", nms.P_FLOOR))
    rows = dm.greedy_rows(q, dm.PR_DIST_M, None, min_conf=float(gate))
    tp = sum(1 for r in rows if r[1] == 1)
    n_conf = sum(1 for r in rows if r[1] >= 0)
    n_pos = int(q["pos"].sum())
    p = _sig(q["logit"])
    gx = np.asarray(q["gt_xy"])[np.asarray(q["pos"], bool)].astype(np.float64)
    sum_cnt = n_det = n_ge2 = 0
    if len(gx):
        d = np.sqrt(((np.asarray(q["xy"])[None, :, :].astype(np.float64) - gx[:, None, :]) ** 2).sum(-1))
        cnt = ((d <= dm.PR_DIST_M) & (p >= float(gate))[None]).sum(1)
        det1 = cnt[cnt >= 1]
        sum_cnt, n_det, n_ge2 = int(det1.sum()), int(len(det1)), int((det1 >= 2).sum())
    tpc = np.zeros(10, np.int64)
    acc_raw = n_acc = 0
    for r in rows:
        if r[1] != 1:
            continue
        gc = int(q["gt_cls"][r[4]])
        if 0 <= gc < 10:
            tpc[gc] += 1
            n_acc += 1
            acc_raw += int(int(q["cls"][r[3]]) == gc)
    return np.array([tp, n_conf, n_pos, sum_cnt, n_det, n_ge2, acc_raw, n_acc], np.float64), tpc


def census_table(recs, gate, nms_cfg=None):
    per, tpc = [], []
    for (_, _, pk) in recs:
        a, b = census_one(pk, gate, nms_cfg)
        per.append(a)
        tpc.append(b)
    return np.stack(per), np.stack(tpc)


def ratios(tot):
    tp, n_conf, n_pos, sum_cnt, n_det, n_ge2, acc_raw, n_acc = [float(x) for x in tot]
    prec = tp / n_conf if n_conf else float("nan")
    rec = tp / n_pos if n_pos else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else float("nan")
    return {"prec": prec, "rec": rec, "f1": f1, "conf_ratio": n_conf / n_pos if n_pos else float("nan"),
            "boxes_per_object": sum_cnt / n_det if n_det else float("nan"),
            "frac_objects_ge2": n_ge2 / n_det if n_det else float("nan"),
            "cls_acc_tp": acc_raw / n_acc if n_acc else float("nan"), "tp": tp, "n_conf": n_conf, "n_pos": n_pos,
            "n_detected_objects": n_det}
