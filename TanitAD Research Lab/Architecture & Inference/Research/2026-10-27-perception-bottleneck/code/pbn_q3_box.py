"""pbn_q3_box.py -- PERC-BOTTLENECK Q3 (PREREG_PERC_BOTTLENECK.md): the refcv8 EMA-48,852 box head on eval139, CPU only,
from PERC8's banked eval-grade dump (`perc8/raw/A_nosync/perc_det.pkl`, the box3d packs of `detection_metrics.window_packs`).

Reads: pooled / per-class / per-range-band / per-size-group AP@2 m (programme `greedy_rows` + `_ap`, re-derived and checked
against PERC8's banked 0.328), the class-confusion matrix of the 2 m greedy TPs (GT class x predicted argmax class),
per-predicted-class size error |dl|+|dw| on the Hungarian pairs, and controls C-MAJ (majority class) and the
re-derivation control. Episode-cluster bootstrap (1,000 draws, seed 0) on the headline cells.

    PYTHONPATH=<tree>/stack python -X utf8 pbn_q3_box.py --pkl <perc_det.pkl> --out <RESULT_Q3.json>
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np

SIZE_GROUP = {  # PROXY for the prereg's GT-footprint size class: the banked pack carries no GT l/w (deviation, disclosed)
    "small": ("person", "rider", "stroller", "animal", "protruding_object"),
    "medium": ("automobile", "other_vehicle"),
    "large": ("heavy_truck", "bus", "trailer"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    from tanitad.eval import detection_metrics as dm
    classes = list(dm._classes())
    C = len(classes)
    det = pickle.load(open(a.pkl, "rb"))
    packs = []
    for r in det:
        for pk in (r.get("box3d") or []):
            packs.append((r["sha12"], pk))
    eps = sorted({s for s, _ in packs})
    e_of = np.array([eps.index(s) for s, _ in packs])
    # ---- per-window rows (class-agnostic 2 m greedy), kept per window for the bootstrap ------------------------ #
    rows_w, ngt_w, conf_w = [], [], []
    band_names = [b[0] for b in dm.BANDS_X_M]
    for _, pk in packs:
        rr = dm.greedy_rows(pk, 2.0)
        rows_w.append(rr)
        pos = pk["pos"]
        ngt_w.append({"all": int(pos.sum()),
                      **{f"band:{b}": int(sum(1 for g in np.nonzero(pos)[0] if dm._band_of(float(pk["gt_xy"][g, 0])) == b))
                         for b in band_names}})
        cm = np.zeros((C, C), np.int64)
        for (_c, kind, _b, s, g) in rr:
            if kind == 1:
                cm[int(pk["gt_cls"][g]), int(pk["cls"][s])] += 1
        conf_w.append(cm)

    def ap_of(idx, key="all"):
        rows, n = [], 0
        for i in idx:
            n += ngt_w[i][key]
            if key == "all":
                rows += rows_w[i]
            else:
                b = key.split(":", 1)[1]
                rows += [x for x in rows_w[i] if x[2] == b]
        return dm._ap(rows, n)

    allidx = np.arange(len(packs))
    out = {"tool": "pbn_q3_box.py", "pkl": a.pkl, "n_windows": len(packs), "n_episodes": len(eps), "classes": classes,
           "n_pos": int(sum(w["all"] for w in ngt_w))}
    out["ap2_all"] = ap_of(allidx)
    out["control_rederive_vs_perc8_0.328"] = {"banked": 0.328, "read": out["ap2_all"],
                                              "pass": abs(out["ap2_all"] - 0.328) < 0.0015}
    out["ap2_band"] = {b: ap_of(allidx, f"band:{b}") for b in band_names}
    out["n_pos_band"] = {b: int(sum(w[f"band:{b}"] for w in ngt_w)) for b in band_names}
    # per class (class-filtered greedy, the programme's per-class AP)
    pc_rows, pc_n = defaultdict(list), defaultdict(int)
    for _, pk in packs:
        for c in range(C):
            pc_rows[c] += dm.greedy_rows(pk, 2.0, cls=c)
            pc_n[c] += int((pk["pos"] & (pk["gt_cls"] == c)).sum())
    out["ap2_class"] = {classes[c]: dm._ap(pc_rows[c], pc_n[c]) for c in range(C)}
    out["n_pos_class"] = {classes[c]: pc_n[c] for c in range(C)}
    # size groups (proxy): class-filtered rows pooled over the group's classes
    out["ap2_size_group_proxy"] = {}
    for gname, members in SIZE_GROUP.items():
        cs = [classes.index(m) for m in members if m in classes]
        out["ap2_size_group_proxy"][gname] = {"classes": [classes[c] for c in cs],
                                              "ap2": dm._ap(sum((pc_rows[c] for c in cs), []), sum(pc_n[c] for c in cs)),
                                              "n_pos": int(sum(pc_n[c] for c in cs))}
    # confusion of the class-agnostic 2 m TPs
    cm = sum(conf_w)
    out["confusion_tp2m"] = {"rows_gt_cols_pred": cm.tolist()}
    tot = int(cm.sum())
    acc = float(np.trace(cm) / tot) if tot else float("nan")
    maj = int(np.argmax(cm.sum(1)))
    c_maj = float(cm.sum(1)[maj] / tot) if tot else float("nan")
    non_auto = [c for c in range(C) if c != classes.index("automobile")]
    na_tp = int(cm[non_auto].sum())
    na_as_auto = int(cm[non_auto, classes.index("automobile")].sum())
    out["class_read"] = {"n_tp": tot, "tp_class_accuracy": acc, "C_MAJ_majority_class": classes[maj], "C_MAJ_accuracy": c_maj,
                         "non_automobile_tp": na_tp, "non_automobile_predicted_automobile": na_as_auto,
                         "non_automobile_share_as_automobile": (na_as_auto / na_tp) if na_tp else float("nan"),
                         "per_class_recall_of_class_given_tp": {classes[c]: (float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None)
                                                                for c in range(C)},
                         "COLLAPSED": bool(acc <= c_maj + 0.02 or (na_tp and na_as_auto / na_tp >= 0.9))}
    # size error on the Hungarian pairs: the pack stores pair_err / pair_size_err but not the pair's (slot, GT) indices, so each
    # pair is recovered as the UNIQUE (matched slot, GT) whose BEV distance equals its pair_err (|diff| < 1e-4 m); ambiguous -> dropped
    se = defaultdict(list)
    n_pairs = n_unres = 0
    for _, pk in packs:
        sl = np.nonzero(pk["matched"])[0]
        if len(sl) == 0 or len(pk["pair_err"]) == 0:
            continue
        D = np.linalg.norm(pk["xy"][sl][:, None, :].astype(np.float64) - pk["gt_xy"][None].astype(np.float64), axis=-1)
        for i in range(len(pk["pair_err"])):
            n_pairs += 1
            hit = np.argwhere(np.abs(D - float(pk["pair_err"][i])) < 1e-4)
            if len(hit) != 1:
                n_unres += 1
                continue
            se[classes[int(pk["gt_cls"][hit[0][1]])]].append(float(pk["pair_size_err"][i]))
    out["size_err_l1_lw_m"] = {k: {"n": len(v), "median": float(np.median(v)), "mean": float(np.mean(v))} for k, v in se.items()}
    out["size_err_pairs"] = {"n_pairs": n_pairs, "gt_class_unresolved": n_unres,
                             "C_SIZE": "NOT COMPUTABLE from the banked pack (no GT l/w stored) -- disclosed deviation"}
    # bootstrap (episode clusters) on pooled AP, band APs and TP class accuracy -- vectorised AP, the SAME
    # all-point arithmetic as box3d_head.ap_from_rows (stable sort on -conf; DontCare rows dropped), checked on the full set
    W_conf, W_kind, W_band = [], [], []
    for rr in rows_w:
        q = [x for x in rr if x[1] >= 0]
        W_conf.append(np.array([x[0] for x in q], np.float64))
        W_kind.append(np.array([x[1] for x in q], np.int8))
        W_band.append(np.array([band_names.index(x[2]) if x[2] in band_names else -1 for x in q], np.int8))

    def fast_ap(idx, key):
        cf = np.concatenate([W_conf[i] for i in idx]); kd = np.concatenate([W_kind[i] for i in idx])
        if key == "all":
            n = sum(ngt_w[i]["all"] for i in idx)
        else:
            b = band_names.index(key.split(":", 1)[1]); m = np.concatenate([W_band[i] for i in idx]) == b
            cf, kd = cf[m], kd[m]; n = sum(ngt_w[i][key] for i in idx)
        if n == 0:
            return float("nan")
        o = np.argsort(-cf, kind="stable"); k = kd[o].astype(np.float64)
        tp = np.cumsum(k); fp = np.cumsum(1 - k); rec = tp / n; prec = tp / np.maximum(tp + fp, 1e-12)
        return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    out["control_fast_ap_equals_programme"] = {"programme": out["ap2_all"], "fast": fast_ap(allidx, "all"),
                                               "pass": abs(out["ap2_all"] - fast_ap(allidx, "all")) < 1e-9}
    rng = np.random.default_rng(0)
    by_e = [np.nonzero(e_of == k)[0] for k in range(len(eps))]
    keys = ["all"] + [f"band:{b}" for b in band_names]
    bs = {k: [] for k in keys}
    bs_acc = []
    for _ in range(a.draws):
        pick = rng.integers(0, len(eps), len(eps))
        idx = np.concatenate([by_e[k] for k in pick])
        for k in keys:
            bs[k].append(fast_ap(idx, k))
        cmb = sum((conf_w[i] for i in idx), np.zeros((C, C), np.int64))
        bs_acc.append(float(np.trace(cmb) / max(1, cmb.sum())))
    out["ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items()}
    out["ci95"]["tp_class_accuracy"] = [float(np.percentile(bs_acc, 2.5)), float(np.percentile(bs_acc, 97.5))]
    out["estimator"] = f"episode-cluster bootstrap, {a.draws} draws, seed 0, percentile; answers another draw of EPISODES; one training seed"
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: out[k] for k in ("ap2_all", "control_rederive_vs_perc8_0.328", "control_fast_ap_equals_programme", "ap2_band", "class_read", "ci95")}, indent=1))


if __name__ == "__main__":
    main()
