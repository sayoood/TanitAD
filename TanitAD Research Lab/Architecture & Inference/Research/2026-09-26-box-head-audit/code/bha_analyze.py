#!/usr/bin/env python3
"""bha_analyze.py -- box-head audit (2026-09-26): score the per-slot dumps written by ``bha_dump.py``.

Every match, loss and filter is the TIP TREE's OWN function, imported, never re-implemented:
``refc_agents.visible_target_filter`` (the trainer's target set), ``agent_slots.match_slots`` (the training
assignment), ``box3d_head.box3d_match_rows`` (the programme's greedy AP matcher, 2 m BEV), ``box3d_head.
box3d_set_loss`` / ``agent_slots.slot_set_loss`` (the losses), and the decoders' own ``decode``.

Outputs one JSON per call with, per set x head:
  * the in-run control (inrun set only): the run's own ``eval_box3d_*`` / ``eval_agent_*`` row at step 38,000
    recomputed batch by batch (8 x 16, labelled rows, the trainer's aggregation = mean over batches);
  * presence distributions for Hungarian-matched vs unmatched slots, and for greedy TP vs FP;
  * AUROC of presence three ways (matched / objectness / greedy-TP), pooled and within-window, with
    clip-cluster bootstrap CIs, and the controls that must read known values (constant presence -> 0.5
    exactly; GT written into the slot format -> 1.0; presence permuted within window -> ~0.5);
  * the precision = recall gate, best-F1 gate, P/R at 0.5 and at the tilt-corrected 10/11, AP@2 m BEV;
  * confident-slot count vs target count, by crowdedness;
  * the FALSE-POSITIVE decomposition at the gate (duplicate / mislocalised 2-5 m / filtered-GT / out-of-field /
    hallucination) and the FALSE-NEGATIVE decomposition;
  * per-class confusion; box error split into range / lateral / size / yaw / z / h by range band;
  * calibration (raw p and the tilt-corrected p' = sigmoid(logit - ln 10)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

LN10 = math.log(10.0)
BANDS = ((0, 10), (10, 20), (20, 30), (30, 45), (45, 60.0001))
K_BINS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 1000))


# --------------------------------------------------------------------------------------------- #
# small statistics                                                                               #
# --------------------------------------------------------------------------------------------- #
def auroc(scores, labels) -> float:
    """Mann-Whitney AUROC with AVERAGE ranks for ties (so a constant score reads exactly 0.5)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    ranks = (start + (cnt + 1) / 2.0)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def qs(x, q=(0.05, 0.25, 0.5, 0.75, 0.95)):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()),
            **{f"p{int(round(100 * qq))}": float(np.quantile(x, qq)) for qq in q}}


def pr_curve(rows, n_gt):
    """rows: (conf, hit) pooled; returns sorted conf, precision, recall arrays."""
    if not rows:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    a = np.asarray(rows, dtype=np.float64)
    o = np.argsort(-a[:, 0], kind="stable")
    c, h = a[o, 0], a[o, 1]
    tp = np.cumsum(h)
    prec = tp / np.arange(1, len(h) + 1)
    rec = tp / max(n_gt, 1)
    return c, prec, rec


def at_gate(rows, n_gt, g):
    a = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 2))
    m = a[:, 0] >= g
    tp = float(a[m, 1].sum())
    n = int(m.sum())
    return {"gate": float(g), "n_conf": n, "tp": int(tp), "precision": (tp / n) if n else float("nan"),
            "recall": tp / max(n_gt, 1), "n_gt": int(n_gt)}


def pr_equal_gate(rows, n_gt):
    c, prec, rec = pr_curve(rows, n_gt)
    if c.size == 0:
        return None
    d = prec - rec
    k = int(np.argmax(d <= 0)) if (d <= 0).any() else len(d) - 1
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    kb = int(np.argmax(f1))
    return {"pr_equal": {"gate": float(c[k]), "precision": float(prec[k]), "recall": float(rec[k]),
                         "rank": k + 1},
            "best_f1": {"gate": float(c[kb]), "f1": float(f1[kb]), "precision": float(prec[kb]),
                        "recall": float(rec[kb])},
            "ap_all_point": float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------------------------- #
# per-window scoring                                                                             #
# --------------------------------------------------------------------------------------------- #
class Scorer:
    def __init__(self, tree: str, cls_weight):
        for p in (os.path.join(tree, "stack"), os.path.join(tree, "taniteval")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import torch
        import tanitad
        want = os.path.normcase(os.path.abspath(os.path.join(tree, "stack")))
        if not os.path.normcase(os.path.abspath(tanitad.__file__)).startswith(want):
            raise SystemExit(f"[bha] tanitad from {tanitad.__file__}, not {want}")
        self.tanitad_file = tanitad.__file__
        from tanitad.models import agent_slots as AS
        from tanitad.models import box3d_head as B3
        from tanitad.refs import refc_agents as RA
        self.torch, self.AS, self.B3, self.RA = torch, AS, B3, RA
        self.dec = {"agent": AS.AgentSlotDecoder(8, 4, n_queries=100, d_model=16, depth=1, n_heads=2,
                                                 enforce_band=False),
                    "box3d": B3.Box3DSlotDecoder(8, 4, n_queries=100, d_model=16, depth=1, n_heads=2,
                                                 enforce_band=False)}
        self.cw = None if cls_weight is None else torch.tensor(cls_weight, dtype=torch.float32)
        r = AS.SlotDecodeRanges()
        self.x_max, self.y_half, self.half = float(r.x_fwd_m), float(r.y_half_m), float(RA.FOV_HALF_ANGLE_RAD)

    def in_field(self, x, y):
        return (x >= 0) & (x <= self.x_max) & (np.abs(y) <= self.y_half) & (np.arctan2(np.abs(y), x) <= self.half)

    def pred(self, r, head):
        torch = self.torch
        raw = (r["b3_raw"] if head == "box3d" else r["ag_raw"]).float()[None]
        with torch.no_grad():
            return self.dec[head].decode(raw)

    def tgt(self, r):
        torch = self.torch
        A = int(r["agent_box"].shape[0])
        t = {"box": r["agent_box"].float()[None], "yaw": r["agent_yaw"].float()[None],
             "cls": r["agent_cls"].long()[None], "valid": torch.ones(1, A, dtype=torch.bool),
             "occ": r["agent_occ"].float()[None], "rates": r["agent_rates"].float()[None],
             "rates_mask": r["agent_rates_mask"].bool()[None]}
        if "agent_cz" in r:
            t.update(cz=r["agent_cz"].float()[None], h=r["agent_h"].float()[None],
                     zh_mask=r["agent_zh_mask"].bool()[None])
        return t

    def window(self, r, head, pred=None):
        """Everything per slot and per matched pair for ONE window and ONE head."""
        torch = self.torch
        pred = pred if pred is not None else self.pred(r, head)
        tgt = self.tgt(r)
        tv = self.RA.visible_target_filter(tgt)
        m = self.AS.match_slots(pred, tv)
        per_elem, n_gt = self.B3.box3d_match_rows(pred, tv, dist_thresh_m=2.0, use_z=False, score="presence")
        rows = per_elem[0]
        N = int(pred["presence_logit"].shape[1])
        logit = pred["presence_logit"][0].double().numpy()
        p = 1.0 / (1.0 + np.exp(-logit))
        box = pred["box"][0].double().numpy()
        yaw = pred["yaw"][0].double().numpy()
        cls = pred["cls_logits"][0].argmax(-1).numpy()
        gbox = tgt["box"][0].double().numpy()
        gyaw = tgt["yaw"][0].double().numpy()
        gcls = tgt["cls"][0].numpy()
        vis = tv["valid"][0].numpy().astype(bool)
        A = gbox.shape[0]
        matched = np.zeros(N, bool)
        mcol = np.full(N, -1)
        if m["rows"][0].numel():
            matched[m["rows"][0].numpy()] = True
            mcol[m["rows"][0].numpy()] = m["cols"][0].numpy()
        hit = np.zeros(N, bool)
        hcol = np.full(N, -1)
        order = []
        for (conf, h, i, j) in rows:
            order.append(i)
            if h:
                hit[i] = True
                hcol[i] = j
        # distances slot centre -> targets (BEV)
        if A:
            d = np.sqrt(((box[:, None, :2] - gbox[None, :, :2]) ** 2).sum(-1))       # [N, A]
            dvis = np.where(vis[None, :], d, np.inf).min(1) if vis.any() else np.full(N, np.inf)
            dall = d.min(1)
            dfil = np.where(~vis[None, :], d, np.inf).min(1) if (~vis).any() else np.full(N, np.inf)
        else:
            d = np.zeros((N, 0))
            dvis = dall = dfil = np.full(N, np.inf)
        out = {"N": N, "p": p, "logit": logit, "matched": matched, "mcol": mcol, "hit": hit, "hcol": hcol,
               "order": np.asarray(order), "conf_rows": [(float(c), int(h)) for (c, h, _, _) in rows],
               "n_vis": int(vis.sum()), "n_valid": int(A), "dvis": dvis, "dall": dall, "dfil": dfil,
               "in_field": self.in_field(box[:, 0], box[:, 1]), "box": box, "yaw": yaw, "cls": cls,
               "gbox": gbox, "gyaw": gyaw, "gcls": gcls, "vis": vis, "d": d,
               "n_dropped": int(sum(m["n_dropped"]))}
        if head == "box3d":
            out["cz"] = pred["cz"][0].double().numpy()
            out["h"] = pred["h"][0].double().numpy()
            if "cz" in tgt:
                out["gcz"] = tgt["cz"][0].double().numpy()
                out["gh"] = tgt["h"][0].double().numpy()
                out["gzh"] = tgt["zh_mask"][0].numpy().astype(bool)
        return out


# --------------------------------------------------------------------------------------------- #
# FP / FN decomposition at a gate                                                                #
# --------------------------------------------------------------------------------------------- #
FP_CATS = ("duplicate_le2m", "mislocalised_2to5m", "filtered_gt_le2m", "filtered_gt_2to5m",
           "out_of_field_pred", "hallucination_gt5m")


def fp_fn_at(w, g):
    """Mutually exclusive FP categories (first match wins, in FP_CATS order) + FN categories."""
    conf = w["p"] >= g
    fp = conf & ~w["hit"]
    cats = {c: 0 for c in FP_CATS}
    for i in np.nonzero(fp)[0]:
        if w["dvis"][i] <= 2.0:
            cats["duplicate_le2m"] += 1
        elif w["dvis"][i] <= 5.0:
            cats["mislocalised_2to5m"] += 1
        elif w["dfil"][i] <= 2.0:
            cats["filtered_gt_le2m"] += 1
        elif w["dfil"][i] <= 5.0:
            cats["filtered_gt_2to5m"] += 1
        elif not w["in_field"][i]:
            cats["out_of_field_pred"] += 1
        else:
            cats["hallucination_gt5m"] += 1
    # FN: visible targets not hit by a confident slot
    fn = {"below_gate_slot_le2m": 0, "confident_slot_le2m_taken_elsewhere": 0, "no_slot_le2m": 0}
    hit_conf_cols = set(int(j) for i, j in enumerate(w["hcol"]) if j >= 0 and conf[i])
    if w["n_valid"]:
        for j in np.nonzero(w["vis"])[0]:
            if int(j) in hit_conf_cols:
                continue
            dj = w["d"][:, j]
            if (dj[~conf] <= 2.0).any():
                fn["below_gate_slot_le2m"] += 1
            elif (dj[conf] <= 2.0).any():
                fn["confident_slot_le2m_taken_elsewhere"] += 1
            else:
                fn["no_slot_le2m"] += 1
    return cats, fn, int(fp.sum()), int(conf.sum())


# --------------------------------------------------------------------------------------------- #
# controls                                                                                      #
# --------------------------------------------------------------------------------------------- #
def gt_as_slot_pred(sc, r, head):
    """GT visible targets written INTO the slot format (presence +20, class one-hot), all other slots
    empty (-20) and parked at (1000, 1000) m. Must read AUROC 1.0 and P = R = 1."""
    torch = sc.torch
    tgt = sc.tgt(r)
    tv = sc.RA.visible_target_filter(tgt)
    idx = tv["valid"][0].nonzero(as_tuple=False).flatten()
    N = 100
    box = torch.tensor([[1000.0, 1000.0, 1.0, 1.0]]).repeat(N, 1)
    yaw = torch.zeros(N)
    pres = torch.full((N,), -20.0)
    cl = torch.zeros(N, len(sc.AS.AGENT_CLASSES))
    for s, j in enumerate(idx.tolist()[:N]):
        box[s] = tgt["box"][0][j]
        yaw[s] = tgt["yaw"][0][j]
        pres[s] = 20.0
        if int(tgt["cls"][0][j]) >= 0:
            cl[s, int(tgt["cls"][0][j])] = 20.0
    yv = torch.stack([torch.sin(yaw), torch.cos(yaw)], -1)
    return {"box": box[None], "yaw": yaw[None], "yaw_vec": yv[None], "presence_logit": pres[None],
            "cls_logits": cl[None], "rates": torch.zeros(1, N, 3), "occ_logit": torch.zeros(1, N)}


# --------------------------------------------------------------------------------------------- #
# the set-level summary                                                                          #
# --------------------------------------------------------------------------------------------- #
def summarise(ws, clips, head, rng, n_boot=1000):
    """ws: list of per-window dicts (labelled windows only) for one head; clips: sha12 per window."""
    P = np.concatenate([w["p"] for w in ws])
    L = np.concatenate([w["logit"] for w in ws])
    M = np.concatenate([w["matched"] for w in ws])
    H = np.concatenate([w["hit"] for w in ws])
    OBJ = np.concatenate([w["dvis"] <= 2.0 for w in ws])
    INF = np.concatenate([w["in_field"] for w in ws])
    wid = np.concatenate([np.full(w["N"], k) for k, w in enumerate(ws)])
    n_gt = int(sum(w["n_vis"] for w in ws))
    rows_all = [cr for w in ws for cr in w["conf_rows"]]
    out = {"n_windows": len(ws), "n_clips": len(set(clips)), "n_slots": int(P.size), "n_visible_targets": n_gt,
           "n_matched_slots": int(M.sum()), "targets_per_window": qs([w["n_vis"] for w in ws]),
           "n_dropped_by_budget": int(sum(w["n_dropped"] for w in ws))}
    # presence distributions
    out["presence_matched"] = qs(P[M])
    out["presence_unmatched"] = qs(P[~M])
    out["presence_greedy_tp"] = qs(P[H])
    out["presence_greedy_nontp"] = qs(P[~H])
    out["presence_unmatched_in_field"] = qs(P[~M & INF])
    out["presence_unmatched_out_of_field"] = qs(P[~M & ~INF])
    out["frac_unmatched_ge_0.5"] = float((P[~M] >= 0.5).mean()) if (~M).any() else None
    out["frac_matched_ge_0.5"] = float((P[M] >= 0.5).mean()) if M.any() else None
    # AUROC three ways, pooled and within-window
    out["auroc_matched_pooled"] = auroc(P, M)
    out["auroc_objectness_pooled"] = auroc(P, OBJ)
    out["auroc_greedy_tp_pooled"] = auroc(P, H)
    ww = [auroc(w["p"], w["matched"]) for w in ws if 0 < w["matched"].sum() < w["N"]]
    out["auroc_matched_within_window"] = qs([x for x in ww if x == x])
    wo = [auroc(w["p"], w["dvis"] <= 2.0) for w in ws if 0 < (w["dvis"] <= 2.0).sum() < w["N"]]
    out["auroc_objectness_within_window"] = qs([x for x in wo if x == x])
    # controls on the SAME slots
    out["control_constant_presence_auroc_matched"] = auroc(np.zeros_like(P), M)
    out["control_constant_presence_auroc_objectness"] = auroc(np.zeros_like(P), OBJ)
    Pp = P.copy()
    for k in range(len(ws)):
        sel = wid == k
        Pp[sel] = rng.permutation(Pp[sel])
    out["control_permuted_within_window_auroc_matched"] = auroc(Pp, M)
    # gates
    out["gates"] = {"0.5": at_gate(rows_all, n_gt, 0.5), "10/11": at_gate(rows_all, n_gt, 10.0 / 11.0)}
    pe = pr_equal_gate(rows_all, n_gt)
    out["pr"] = pe
    if pe:
        out["gates"]["pr_equal"] = at_gate(rows_all, n_gt, pe["pr_equal"]["gate"])
        out["gates"]["best_f1"] = at_gate(rows_all, n_gt, pe["best_f1"]["gate"])
    # confident count vs targets, by crowdedness
    crowd = {}
    for lo, hi in K_BINS:
        sel = [w for w in ws if lo <= w["n_vis"] <= hi]
        if not sel:
            continue
        row = {"n_windows": len(sel), "targets_mean": float(np.mean([w["n_vis"] for w in sel]))}
        for gname, g in (("0.5", 0.5), ("10/11", 10 / 11), ("pr_equal", (pe or {}).get("pr_equal", {}).get("gate"))):
            if g is None:
                continue
            nc = [int((w["p"] >= g).sum()) for w in sel]
            row[f"conf_mean@{gname}"] = float(np.mean(nc))
            row[f"all100@{gname}"] = int(sum(1 for x in nc if x == 100))
            rr = [cr for w in sel for cr in w["conf_rows"]]
            ag = at_gate(rr, sum(w["n_vis"] for w in sel), g)
            row[f"precision@{gname}"] = ag["precision"]
            row[f"recall@{gname}"] = ag["recall"]
        row["sum_p_mean"] = float(np.mean([w["p"].sum() for w in sel]))
        row["sum_p_tiltcorr_mean"] = float(np.mean([(1 / (1 + np.exp(-(w["logit"] - LN10)))).sum() for w in sel]))
        crowd[f"K{lo}-{hi}"] = row
    out["by_crowdedness"] = crowd
    # FP / FN decomposition at 0.5 and at the P=R gate
    dec = {}
    for gname, g in (("0.5", 0.5), ("10/11", 10 / 11), ("pr_equal", (pe or {}).get("pr_equal", {}).get("gate"))):
        if g is None:
            continue
        C = {c: 0 for c in FP_CATS}
        FN = {}
        nfp = nconf = 0
        for w in ws:
            c, fn, a_, b_ = fp_fn_at(w, g)
            for k, v in c.items():
                C[k] += v
            for k, v in fn.items():
                FN[k] = FN.get(k, 0) + v
            nfp += a_
            nconf += b_
        dec[gname] = {"gate": g, "n_confident": nconf, "n_fp": nfp,
                      "fp_categories": C, "fp_fractions": {k: (v / nfp if nfp else None) for k, v in C.items()},
                      "fn_categories": FN, "n_fn": int(sum(FN.values()))}
    out["fp_fn_decomposition"] = dec
    # calibration
    cal = []
    edges = np.linspace(0, 1, 11)
    Pt = 1 / (1 + np.exp(-(L - LN10)))
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (P >= lo) & ((P < hi) if hi < 1 else (P <= hi))
        s2 = (Pt >= lo) & ((Pt < hi) if hi < 1 else (Pt <= hi))
        cal.append({"bin": [float(lo), float(hi)], "n": int(s.sum()),
                    "p_mean": float(P[s].mean()) if s.any() else None,
                    "frac_matched": float(M[s].mean()) if s.any() else None,
                    "frac_objectness": float(OBJ[s].mean()) if s.any() else None,
                    "n_tiltcorr": int(s2.sum()),
                    "ptilt_mean": float(Pt[s2].mean()) if s2.any() else None,
                    "frac_matched_tiltcorr": float(M[s2].mean()) if s2.any() else None})
    out["calibration"] = cal

    def ece(pp, yy):
        e = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (pp >= lo) & ((pp < hi) if hi < 1 else (pp <= hi))
            if s.any():
                e += s.mean() * abs(pp[s].mean() - yy[s].mean())
        return float(e)
    out["ece_raw_vs_matched"] = ece(P, M.astype(float))
    out["ece_tiltcorr_vs_matched"] = ece(Pt, M.astype(float))
    out["ece_raw_vs_objectness"] = ece(P, OBJ.astype(float))
    # matched pairs: localisation vs presence
    pe_rows = []
    for w in ws:
        for i in np.nonzero(w["matched"])[0]:
            j = int(w["mcol"][i])
            dx = w["box"][i, 0] - w["gbox"][j, 0]
            dy = w["box"][i, 1] - w["gbox"][j, 1]
            pe_rows.append((w["p"][i], math.hypot(dx, dy)))
    if pe_rows:
        a = np.asarray(pe_rows)
        far = a[:, 1] > 2.0
        out["hungarian_pairs"] = {"n": int(a.shape[0]), "frac_err_gt2m": float(far.mean()),
                                  "frac_err_gt2m_and_conf": float((far & (a[:, 0] >= 0.5)).mean()),
                                  "presence_err_le2m": qs(a[~far, 0]), "presence_err_gt2m": qs(a[far, 0]),
                                  "centre_err": qs(a[:, 1]),
                                  "spearman_presence_vs_err": float(_spearman(a[:, 0], a[:, 1]))}
    # per-class confusion on Hungarian pairs within 2 m, and per-class recall at 0.5
    C = len(CLASSES)
    conf_all = np.zeros((C, C), int)
    conf_le2 = np.zeros((C, C), int)
    rec_cls = {c: [0, 0] for c in CLASSES}
    for w in ws:
        for i in np.nonzero(w["matched"])[0]:
            j = int(w["mcol"][i])
            gc = int(w["gcls"][j])
            if gc < 0:
                continue
            conf_all[gc, int(w["cls"][i])] += 1
            if math.hypot(*(w["box"][i, :2] - w["gbox"][j, :2])) <= 2.0:
                conf_le2[gc, int(w["cls"][i])] += 1
        hitc = set(int(j) for i, j in enumerate(w["hcol"]) if j >= 0 and w["p"][i] >= 0.5)
        for j in np.nonzero(w["vis"])[0]:
            gc = int(w["gcls"][j])
            if gc < 0:
                continue
            rec_cls[CLASSES[gc]][1] += 1
            rec_cls[CLASSES[gc]][0] += int(int(j) in hitc)
    out["class_confusion_hungarian_all"] = conf_all.tolist()
    out["class_confusion_hungarian_le2m"] = conf_le2.tolist()
    out["class_accuracy_hungarian_all"] = {CLASSES[k]: (float(conf_all[k, k] / conf_all[k].sum())
                                                        if conf_all[k].sum() else None, int(conf_all[k].sum()))
                                           for k in range(C)}
    out["class_accuracy_hungarian_le2m"] = {CLASSES[k]: (float(conf_le2[k, k] / conf_le2[k].sum())
                                                         if conf_le2[k].sum() else None, int(conf_le2[k].sum()))
                                            for k in range(C)}
    out["pred_class_hist_matched"] = np.bincount(np.concatenate(
        [w["cls"][w["matched"]] for w in ws]).astype(int), minlength=C).tolist()
    out["recall_at_0.5_by_class"] = {k: {"recall": (v[0] / v[1]) if v[1] else None, "n_gt": v[1]}
                                     for k, v in rec_cls.items()}
    # box error by range band (Hungarian pairs AND greedy TP pairs at 0.5)
    out["box_error_by_band"] = {"hungarian": _band_errors(ws, head, "matched"),
                                "greedy_tp_at_0.5": _band_errors(ws, head, "hit")}
    # clip-cluster bootstrap
    out["bootstrap"] = _bootstrap(ws, clips, rng, n_boot)
    return out


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1] if len(a) > 2 else float("nan")


def _band_errors(ws, head, which):
    res = {}
    for lo, hi in BANDS:
        E = {"centre_l2": [], "range_err_signed": [], "lateral_err": [], "dl": [], "dw": [], "yaw_abs_deg": [],
             "yaw_abs_mod180_deg": [], "dz": [], "dh": []}
        for w in ws:
            sel = np.nonzero(w[which] & ((w["p"] >= 0.5) if which == "hit" else True))[0]
            for i in sel:
                j = int(w["mcol"][i] if which == "matched" else w["hcol"][i])
                if j < 0:
                    continue
                gx, gy = w["gbox"][j, 0], w["gbox"][j, 1]
                rg = math.hypot(gx, gy)
                if not (lo <= rg < hi):
                    continue
                px, py = w["box"][i, 0], w["box"][i, 1]
                rp = math.hypot(px, py)
                E["centre_l2"].append(math.hypot(px - gx, py - gy))
                E["range_err_signed"].append(rp - rg)
                E["lateral_err"].append(rg * abs(wrap(math.atan2(py, px) - math.atan2(gy, gx))))
                E["dl"].append(w["box"][i, 2] - w["gbox"][j, 2])
                E["dw"].append(w["box"][i, 3] - w["gbox"][j, 3])
                dyaw = abs(wrap(w["yaw"][i] - w["gyaw"][j]))
                E["yaw_abs_deg"].append(math.degrees(dyaw))
                E["yaw_abs_mod180_deg"].append(math.degrees(min(dyaw, math.pi - dyaw)))
                if head == "box3d" and "gzh" in w and w["gzh"][j]:
                    E["dz"].append(w["cz"][i] - w["gcz"][j])
                    E["dh"].append(w["h"][i] - w["gh"][j])
        row = {"n": len(E["centre_l2"])}
        for k, v in E.items():
            if v:
                row[k] = qs(v)
                if k in ("range_err_signed", "dl", "dw", "dz", "dh"):
                    row[k + "_abs"] = qs(np.abs(v))
        res[f"{lo}-{int(round(hi))}m"] = row
    return res


def _bootstrap(ws, clips, rng, n_boot):
    by = {}
    for k, c in enumerate(clips):
        by.setdefault(c, []).append(k)
    keys = sorted(by)
    stats = {"auroc_matched": [], "auroc_objectness": [], "precision@0.5": [], "recall@0.5": [],
             "pr_equal_gate": [], "pr_equal_value": [], "precision@10/11": [], "recall@10/11": []}
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(keys), len(keys))
        idx = [k for c in pick for k in by[keys[c]]]
        P = np.concatenate([ws[k]["p"] for k in idx])
        M = np.concatenate([ws[k]["matched"] for k in idx])
        O = np.concatenate([ws[k]["dvis"] <= 2.0 for k in idx])
        stats["auroc_matched"].append(auroc(P, M))
        stats["auroc_objectness"].append(auroc(P, O))
        rows = [cr for k in idx for cr in ws[k]["conf_rows"]]
        ng = sum(ws[k]["n_vis"] for k in idx)
        a = at_gate(rows, ng, 0.5)
        b = at_gate(rows, ng, 10 / 11)
        stats["precision@0.5"].append(a["precision"])
        stats["recall@0.5"].append(a["recall"])
        stats["precision@10/11"].append(b["precision"])
        stats["recall@10/11"].append(b["recall"])
        pe = pr_equal_gate(rows, ng)
        stats["pr_equal_gate"].append(pe["pr_equal"]["gate"] if pe else float("nan"))
        stats["pr_equal_value"].append(pe["pr_equal"]["precision"] if pe else float("nan"))
    return {k: {"lo95": float(np.nanquantile(v, 0.025)), "hi95": float(np.nanquantile(v, 0.975)),
                "n_boot": int(n_boot), "unit": "clip (sha12) cluster"} for k, v in stats.items()}


# --------------------------------------------------------------------------------------------- #
# the in-run control: the run's own eval row, recomputed                                          #
# --------------------------------------------------------------------------------------------- #
def inrun_control(sc, rows, eval_row):
    torch = sc.torch
    if len(rows) != 128:
        return {"status": f"SKIPPED: {len(rows)} windows, need 128"}
    acc_b3, acc_ag, nb = {}, {}, 0
    for b in range(8):
        grp = [r for r in rows[16 * b:16 * b + 16] if r["agent_label"]]
        if not grp:
            continue
        nb += 1
        Amax = max(1, max(int(r["agent_box"].shape[0]) for r in grp))

        def padt(key, shape_tail, dtype, fill=0):
            out = torch.full((len(grp), Amax, *shape_tail), fill, dtype=dtype)
            for k, r in enumerate(grp):
                a = int(r["agent_box"].shape[0])
                if a:
                    out[k, :a] = r[key].to(dtype).reshape(a, *shape_tail)
            return out
        valid = torch.zeros(len(grp), Amax, dtype=torch.bool)
        for k, r in enumerate(grp):
            valid[k, :int(r["agent_box"].shape[0])] = True
        tgt = {"box": padt("agent_box", (4,), torch.float32), "yaw": padt("agent_yaw", (), torch.float32),
               "cls": padt("agent_cls", (), torch.long, -1), "valid": valid,
               "occ": padt("agent_occ", (), torch.float32, -1.0), "rates": padt("agent_rates", (3,), torch.float32),
               "rates_mask": padt("agent_rates_mask", (), torch.bool, False),
               "cz": padt("agent_cz", (), torch.float32), "h": padt("agent_h", (), torch.float32),
               "zh_mask": padt("agent_zh_mask", (), torch.bool, False)}
        with torch.no_grad():
            p3 = sc.dec["box3d"].decode(torch.stack([r["b3_raw"].float() for r in grp]))
            pa = sc.dec["agent"].decode(torch.stack([r["ag_raw"].float() for r in grp]))
            b3 = sc.B3.box3d_set_loss(p3, tgt, cls_class_weight=sc.cw, visible_filter=True)
            tva = sc.RA.visible_target_filter(tgt)
            ag = sc.AS.slot_set_loss(pa, tva, match=sc.AS.match_slots(pa, tva), cls_class_weight=sc.cw)
        for k in ("presence", "cls", "centre", "size", "yaw", "z", "h", "occ", "rates"):
            if f"loss_{k}" in b3:
                acc_b3[k] = acc_b3.get(k, 0.0) + float(b3[f"loss_{k}"])
        acc_b3["n_target"] = acc_b3.get("n_target", 0.0) + float(b3["n"]["target"])
        for k in ("presence", "cls", "centre", "size", "yaw"):
            acc_ag[k] = acc_ag.get(k, 0.0) + float(ag[f"loss_{k}"])
        acc_ag["n_target"] = acc_ag.get("n_target", 0.0) + float(ag["n"]["target"])
    res = {"n_batches": nb, "terms": {}}
    for head, acc, pre in (("box3d", acc_b3, "eval_box3d_"), ("agent", acc_ag, "eval_agent_")):
        for k, v in acc.items():
            key = pre + ("n_target" if k == "n_target" else k)
            mine = v / max(nb, 1)
            ref = eval_row.get(key)
            res["terms"][key] = {"recomputed_cpu": round(mine, 5), "inrun_row": ref,
                                 "abs_diff": (None if ref is None else round(abs(mine - ref), 5)),
                                 "rel_diff": (None if not ref else round(abs(mine - ref) / abs(ref), 4))}
    return res


CLASSES = None


def main(argv=None):
    global CLASSES
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--sets", default="inrun,clipgrid,train")
    ap.add_argument("--metrics", default=None, help="metrics.jsonl for the in-run control")
    ap.add_argument("--eval-step", type=int, default=38000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    rec_any = None
    res = {"tool": "bha_analyze.py", "argv": sys.argv[1:], "sets": {}}
    for s in [x for x in a.sets.split(",") if x]:
        files = sorted(Path(a.dump_dir).glob(f"{s}_chunk*.pt"))
        if not files:
            res["sets"][s] = {"status": "NO DUMP"}
            continue
        recj = Path(a.dump_dir) / f"{s}_record.json"
        rec = json.loads(recj.read_text(encoding="utf-8")) if recj.exists() else {}
        rec_any = rec_any or rec
        rows = []
        for f in files:
            rows += torch.load(f, weights_only=False)
        cw = (rec.get("model") or {}).get("cls_class_weight")
        sc = Scorer(a.tree, cw)
        from tanitad.models import agent_slots as AS
        CLASSES = tuple(AS.AGENT_CLASSES)
        rng = np.random.default_rng(a.seed)
        sres = {"dump_record": {k: rec.get(k) for k in ("ckpt", "config", "selection", "n_windows_selected",
                                                        "s_per_window", "train_selection", "model", "departures")},
                "n_rows": len(rows), "n_labelled": sum(1 for r in rows if r["agent_label"]),
                "tanitad_file": sc.tanitad_file}
        # control 0: re-decode == the model's own decode
        mx = {"box3d_presence": 0.0, "agent_presence": 0.0, "box3d_box": 0.0, "agent_box": 0.0}
        for r in rows:
            p3, pa = sc.pred(r, "box3d"), sc.pred(r, "agent")
            mx["box3d_presence"] = max(mx["box3d_presence"], float((p3["presence_logit"][0] - r["b3_pres"]).abs().max()))
            mx["agent_presence"] = max(mx["agent_presence"], float((pa["presence_logit"][0] - r["ag_pres"]).abs().max()))
            mx["box3d_box"] = max(mx["box3d_box"], float((p3["box"][0] - r["b3_box"]).abs().max()))
            mx["agent_box"] = max(mx["agent_box"], float((pa["box"][0] - r["ag_box"]).abs().max()))
        sres["control_redecode_max_abs_diff"] = mx
        # the in-run control
        if s == "inrun" and a.metrics:
            ev = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
            ev = [e for e in ev if e.get("step") == a.eval_step and "eval_loss" in e]
            if len(ev) == 1:
                sres["inrun_control"] = inrun_control(sc, rows, ev[0])
            else:
                sres["inrun_control"] = {"status": f"{len(ev)} eval rows at step {a.eval_step}"}
        lab = [r for r in rows if r["agent_label"]]
        clips = [r["sha12"] for r in lab]
        for head in ("box3d", "agent"):
            ws = [sc.window(r, head) for r in lab]
            sres[head] = summarise(ws, clips, head, rng, a.n_boot)
        # the GT-as-slots control through the SAME summary (box head's code path, BEV)
        wsc = [sc.window(r, "agent", pred=gt_as_slot_pred(sc, r, "agent")) for r in lab]
        g = summarise(wsc, clips, "agent", rng, 50)
        sres["control_gt_as_slots"] = {k: g.get(k) for k in ("auroc_matched_pooled", "auroc_objectness_pooled",
                                                            "auroc_greedy_tp_pooled", "gates", "pr",
                                                            "fp_fn_decomposition")}
        res["sets"][s] = sres
        print(f"[bha-analyze] {s}: {len(rows)} windows ({sres['n_labelled']} labelled) done "
              f"({time.time() - t0:.0f} s)", flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"[bha-analyze] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
