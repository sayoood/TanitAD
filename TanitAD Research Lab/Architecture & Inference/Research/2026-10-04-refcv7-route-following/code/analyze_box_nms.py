"""SPEC.md sec. 7: duplicate boxes per object and an inference-time NMS (centre distance / BEV IoU).

Packs: the trainer's own `_det_pack_box3d` / `_det_pack_agent` from run_route.py (EVAL = eval_s0, TRAIN =
train_s0 -- the fit set). Box geometry for the IoU NMS: the captured box3d slots (x, y, l, w, yaw). TP / FP labels
come from the launch tree's own `detection_metrics.greedy_rows` (score-ordered greedy 2 m, DontCare excluded).
NMS removes slots from the pack (it never re-scores), so every downstream number is the module's arithmetic on
the surviving slots.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_metrics as rm  # noqa: E402
from tanitad.eval import detection_metrics as det  # noqa: E402

GATE_BRIEF = 0.2589          # the run's in-run eval_box3d_calib_pr_gate at step 50,400 (as briefed)
P_FLOOR = 0.05               # NMS acts on slots with p >= 0.05; the flat tail below is left untouched in EVERY arm
R_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
IOU_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
B = 1000


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def subset_pack(pk, keep):
    q = dict(pk)
    for k in ("logit", "xy", "cls", "cls_corr", "matched", "exempt"):
        if k in q:
            q[k] = q[k][keep]
    return q


_IOU_CACHE: dict = {}


def iou_matrix(key, xy, g, idx):
    """pairwise BEV IoU among the slots `idx` of one window (cached: every IoU threshold reuses it)."""
    if key in _IOU_CACHE:
        return _IOU_CACHE[key]
    n = len(idx)
    M = np.zeros((n, n))
    rad = 0.5 * np.hypot(g[0][idx, 2], g[0][idx, 3])
    for a in range(n):
        for b in range(a + 1, n):
            i, j = idx[a], idx[b]
            if math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]) > rad[a] + rad[b]:
                continue
            v = rm.bev_iou((g[0][i, 0], g[0][i, 1], g[0][i, 2], g[0][i, 3], g[1][i]),
                           (g[0][j, 0], g[0][j, 1], g[0][j, 2], g[0][j, 3], g[1][j]))
            M[a, b] = M[b, a] = v
    _IOU_CACHE[key] = M
    return M


def apply_nms(packs, mode, thr, geo=None, tag=""):
    out = []
    for pk in packs:
        p = sig(pk["logit"])
        hi = p >= P_FLOOR
        if mode == "centre":
            keep_hi = rm.nms_keep(np.where(hi, p, -1.0), pk["xy"].astype(np.float64), mode, thr, min_p=P_FLOOR)
        else:
            idx = np.nonzero(hi)[0]
            order = np.argsort(-p[idx], kind="mergesort")
            idx = idx[order]
            M = iou_matrix((tag, pk["win"]), pk["xy"].astype(np.float64), geo[pk["win"]], idx)
            kept = []
            keep_hi = np.zeros(len(p), bool)
            for a in range(len(idx)):
                if all(M[a, b] <= thr for b in kept):
                    kept.append(a)
            keep_hi[idx[kept]] = True
        out.append(subset_pack(pk, keep_hi | ~hi))
    return out


def rows_all(packs):
    """(conf, kind, ep_index_of_pack) for every greedy-2 m row; DontCare dropped."""
    C, K, E = [], [], []
    for i, pk in enumerate(packs):
        for (c, k, *_r) in det.greedy_rows(pk, det.PR_DIST_M, None):
            if k < 0:
                continue
            C.append(c)
            K.append(k)
            E.append(i)
    return np.asarray(C), np.asarray(K, float), np.asarray(E)


def ap_weighted(C, K, Ew, npos_w):
    o = np.argsort(-C, kind="mergesort")
    k, w = K[o], Ew[o]
    tp = np.cumsum(w * k)
    fp = np.cumsum(w * (1 - k))
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(w * k / npos_w * prec)) if npos_w > 0 else float("nan")


def census(packs, gate):
    rows = [r for pk in packs for r in det.greedy_rows(pk, det.PR_DIST_M, None, min_conf=gate)]
    tp = sum(1 for r in rows if r[1] == 1)
    nc = sum(1 for r in rows if r[1] >= 0)
    npos = int(sum(int(pk["pos"].sum()) for pk in packs))
    prec = tp / nc if nc else float("nan")
    rec = tp / npos if npos else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (nc and npos and prec + rec > 0) else float("nan")
    return {"n_conf": nc, "tp": tp, "n_pos": npos, "prec": prec, "rec": rec, "f1": f1,
            "conf_ratio": nc / npos if npos else float("nan")}


def dup_stats(packs, gate):
    """per VIS-1 positive GT: # slots with p >= gate whose BEV centre is within 2 m."""
    cnt = []
    spread = []
    dupfp = 0
    nfp = 0
    for pk in packs:
        p = sig(pk["logit"])
        gx = pk["gt_xy"][pk["pos"]].astype(np.float64)
        sel = p >= gate
        if len(gx):
            d = np.sqrt(((pk["xy"][None, :, :].astype(np.float64) - gx[:, None, :]) ** 2).sum(-1))   # [G, N]
            near = (d <= det.PR_DIST_M) & sel[None]
            n = near.sum(1)
            cnt += n.tolist()
            for gi in range(len(gx)):
                if n[gi] >= 2:
                    pp = p[near[gi]]
                    spread.append(float(pp.max() - pp.min()))
        allg = pk["gt_xy"][pk["pos"] | pk["ign"]].astype(np.float64)
        for (c, k, band, s, g) in det.greedy_rows(pk, det.PR_DIST_M, None, min_conf=gate):
            if k == 0:
                nfp += 1
                if len(allg) and np.sqrt(((allg - pk["xy"][s].astype(np.float64)) ** 2).sum(-1)).min() <= det.PR_DIST_M:
                    dupfp += 1
    cnt = np.asarray(cnt)
    det1 = cnt[cnt >= 1]
    return {"n_pos": int(len(cnt)), "n_pos_with_ge1": int(len(det1)),
            "mean_slots_per_detected_object": round(float(det1.mean()), 4) if len(det1) else None,
            "frac_detected_objects_with_ge2": round(float((det1 >= 2).mean()), 4) if len(det1) else None,
            "frac_detected_objects_with_ge3": round(float((det1 >= 3).mean()), 4) if len(det1) else None,
            "hist_slots_per_object_0_to_8plus": [int((cnt == i).sum()) for i in range(8)] + [int((cnt >= 8).sum())],
            "dup_score_spread_median": round(float(np.median(spread)), 4) if spread else None,
            "n_fp": nfp, "frac_fp_within_2m_of_a_gt": round(dupfp / nfp, 4) if nfp else None}


def score_arm(packs, gate_fit, eps_of_pack, draws, ref=None):
    """EVAL row for one arm: census at the TRAIN-refitted gate and at 0.2589, AP@2 m, dups; bootstrap vs ref."""
    C, K, E = rows_all(packs)
    npos_pack = np.array([int(pk["pos"].sum()) for pk in packs], float)
    eps, inv = rm.cluster_index(eps_of_pack)
    ne = len(eps)
    ap = ap_weighted(C, K, np.ones_like(C), npos_pack.sum())
    out = {"AP2m": round(ap, 4), "gate_fit": gate_fit,
           "at_fit_gate": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in census(packs, gate_fit).items()},
           "at_0p2589": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in census(packs, GATE_BRIEF).items()},
           "at_0p5": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in census(packs, 0.5).items()},
           "dups_at_fit_gate": dup_stats(packs, gate_fit), "dups_at_0p2589": dup_stats(packs, GATE_BRIEF),
           "dups_at_0p5": dup_stats(packs, 0.5)}
    # bootstrap: per-draw AP (weighted rows) and F1 at the fit gate
    ep_row = inv[E]
    ep_npos = np.bincount(inv, weights=npos_pack, minlength=ne)
    conf_g = C >= gate_fit
    tp_ep = np.bincount(ep_row, weights=(K * conf_g), minlength=ne)
    nc_ep = np.bincount(ep_row, weights=conf_g.astype(float), minlength=ne)
    aps, f1s = [], []
    for dr in draws:
        w_ep = np.bincount(dr, minlength=ne).astype(float)
        aps.append(ap_weighted(C, K, w_ep[ep_row], float(w_ep @ ep_npos)))
        tp, nc, npv = w_ep @ tp_ep, w_ep @ nc_ep, w_ep @ ep_npos
        pr = tp / nc if nc else np.nan
        rc = tp / npv if npv else np.nan
        f1s.append(2 * pr * rc / (pr + rc) if (nc and npv and pr + rc > 0) else np.nan)
    aps, f1s = np.asarray(aps), np.asarray(f1s)
    out["AP2m_ci95"] = rm.ci95(aps)
    out["F1_fit_gate_ci95"] = rm.ci95(f1s)
    out["_boot"] = (aps, f1s)
    if ref is not None:
        out["dAP_vs_noNMS_ci95"] = rm.ci95(aps - ref["_boot"][0])
        out["dF1_vs_noNMS_ci95"] = rm.ci95(f1s - ref["_boot"][1])
        out["dAP_vs_noNMS"] = round(out["AP2m"] - ref["AP2m"], 4)
        out["dF1_vs_noNMS"] = (round(out["at_fit_gate"]["f1"] - ref["at_fit_gate"]["f1"], 4)
                               if isinstance(out["at_fit_gate"]["f1"], float) else None)
    return out


def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--diag-eval-packs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.data)
    ev = pickle.load(open(D / "eval_s0.packs.pkl", "rb"))
    trp = pickle.load(open(D / "train_s0.packs.pkl", "rb"))
    ze = np.load(D / "eval_s0.npz", allow_pickle=True)
    zt = np.load(D / "train_s0.npz", allow_pickle=True)
    geo_e = list(zip(ze["box_xylw"].astype(np.float64), ze["box_yaw"].astype(np.float64)))
    geo_t = list(zip(zt["box_xylw"].astype(np.float64), zt["box_yaw"].astype(np.float64)))
    res = {"definition": {
        "dup": "per VIS-1 positive GT, # slots with p >= gate whose BEV centre is within 2 m (det.PR_DIST_M)",
        "nms": f"score-ordered greedy suppression per window, all classes together, on slots with p >= {P_FLOOR} "
               "(the tail below is untouched in every arm, so AP is compared on the same tail)",
        "gate_brief": GATE_BRIEF, "fit": "r* / IoU* = argmax TRAIN AP@2m (tie -> milder); gate = TRAIN P = R "
                                        "re-fitted after NMS (det.pr_equal_gate)"}}
    # C8 control: the analytic IoU cases are pinned by test_route_metrics.py; C9: my packs vs the banked diag packs
    for hd in ("box3d", "agent"):
        bad = 0
        for pk in ev[hd]:
            if hd == "box3d":
                if np.abs(pk["logit"] - ze["box_logit"][pk["win"]]).max() > 0:
                    bad += 1
        res.setdefault("C_pack_vs_capture", {})[hd] = {"n_packs": len(ev[hd]), "n_logit_mismatch_vs_capture": bad}
    if a.diag_eval_packs:
        dg = pickle.load(open(a.diag_eval_packs, "rb"))
        for hd in ("box3d", "agent"):
            A, Bp = ev[hd], dg[hd]
            same_order = [p.get("sha12") for p in A] == [p.get("sha12") for p in Bp]
            mx = max(float(np.abs(x["logit"] - y["logit"]).max()) for x, y in zip(A, Bp)) if same_order else None
            flips = (sum(int(((sig(x["logit"]) >= GATE_BRIEF) != (sig(y["logit"]) >= GATE_BRIEF)).sum())
                         for x, y in zip(A, Bp)) if same_order else None)
            res.setdefault("C9_vs_diag_packs", {})[hd] = {"n_mine": len(A), "n_diag": len(Bp),
                                                         "same_window_order": same_order,
                                                         "logit_max_abs_diff": mx, "gate_0p2589_flips": flips}
    # control: the cached-matrix IoU NMS == the reference route_metrics.nms_keep (first 25 TRAIN box3d packs)
    nbad = 0
    for pk in trp["box3d"][:25]:
        p = sig(pk["logit"])
        g = geo_t[pk["win"]]
        boxes = [(float(g[0][i, 0]), float(g[0][i, 1]), float(g[0][i, 2]), float(g[0][i, 3]), float(g[1][i]))
                 for i in range(len(p))]
        ref = rm.nms_keep(np.where(p >= P_FLOOR, p, -1.0), pk["xy"].astype(np.float64), "iou", 0.1, boxes=boxes,
                          min_p=P_FLOOR)
        mine = apply_nms([pk], "iou", 0.1, geo_t, tag="ctrl")[0]
        nbad += int(len(mine["logit"]) != int((ref | (p < P_FLOOR)).sum()))
    res["C_iou_nms_cached_vs_reference"] = {"n_packs": 25, "n_mismatch": nbad}
    for hd in ("box3d", "agent"):
        print(f"[box] {hd}", flush=True)
        E, T = ev[hd], trp[hd]
        eps_e = [pk.get("sha12") for pk in E]
        draws = rm.make_draws(eps_e, B=B, seed=0)
        modes = [("none", None)] + [("centre", r) for r in R_GRID] + ([("iou", t) for t in IOU_GRID]
                                                                       if hd == "box3d" else [])
        fit = {}
        for mode, thr in modes:
            Tn = T if mode == "none" else apply_nms(T, mode, thr, geo_t, tag="train_" + hd)
            C, K, _E = rows_all(Tn)
            npos = sum(int(pk["pos"].sum()) for pk in Tn)
            g = det.pr_equal_gate(Tn)
            fit[f"{mode}|{thr}"] = {"train_AP2m": round(ap_weighted(C, K, np.ones_like(C), npos), 4),
                                    "train_pr_gate": round(float(g["gate"]), 4),
                                    "train_prec_at_gate": round(float(g["precision"]), 4)}
            print(f"  fit {mode} {thr}: {fit[f'{mode}|{thr}']}", flush=True)
        best = {}
        for fam, grid, milder in (("centre", R_GRID, min), ("iou", IOU_GRID, max)):
            ks = [f"{fam}|{t}" for t in grid if f"{fam}|{t}" in fit]
            if not ks:
                continue
            top = max(fit[k]["train_AP2m"] for k in ks)
            cands = [float(k.split("|")[1]) for k in ks if fit[k]["train_AP2m"] == top]
            best[fam] = milder(cands)
        r = {"train_fit_grid": fit, "chosen": best}
        ref = score_arm(E, fit["none|None"]["train_pr_gate"], eps_e, draws)
        rows = {"no_NMS": ref}
        for fam, thr in best.items():
            En = apply_nms(E, fam, thr, geo_e, tag="eval_" + hd)
            rows[f"{fam}_NMS({thr})"] = score_arm(En, fit[f"{fam}|{thr}"]["train_pr_gate"], eps_e, draws, ref=ref)
        # EVAL grid for transparency (not used for selection)
        grid_eval = {}
        for mode, thr in modes[1:]:
            En = apply_nms(E, mode, thr, geo_e, tag="eval_" + hd)
            C, K, _E = rows_all(En)
            npos = sum(int(pk["pos"].sum()) for pk in En)
            grid_eval[f"{mode}|{thr}"] = {"AP2m": round(ap_weighted(C, K, np.ones_like(C), npos), 4),
                                          **{f"{k}@0.2589": (round(v, 4) if isinstance(v, float) else v)
                                             for k, v in census(En, GATE_BRIEF).items()
                                             if k in ("prec", "rec", "f1", "conf_ratio")},
                                          "dup_mean@0.2589": dup_stats(En, GATE_BRIEF)["mean_slots_per_detected_object"]}
        for v in rows.values():
            v.pop("_boot", None)
        r["eval"] = rows
        r["eval_grid_not_for_selection"] = grid_eval
        r["train_dups_noNMS_at_0p2589"] = dup_stats(T, GATE_BRIEF)
        res[hd] = r
    Path(a.out).write_text(json.dumps(sanitize(res), indent=1, allow_nan=False), encoding="utf-8")
    print("[box] wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
