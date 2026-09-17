#!/usr/bin/env python3
"""RUNG 2 -- the BOX panel: AP, recall at precision, lead range MAE, lead AUC, and the
analytic controls, with a CLIP-cluster bootstrap.

    python p_box_eval.py --runs C:\\Users\\Admin\\d3_out\\box --out raw/box_panel.json

AP@tau: centre-distance matching, one-to-one, greedy by score, over ALL test rows;
all-point-interpolated average precision. Clusters = CLIPS, never frames.

⭐ THE CONTROLS AND THEIR KNOWN VALUES
  `prior`  the TRAIN-split empirical centre-rate map, emitted identically in every test
           row -- the zero-image predictor. Computed here, never trained.
  `const`  a CONSTANT lead predictor. Its accuracy is EXACTLY the test prevalence and its
           AUC is EXACTLY 0.5. ⚠️ This is the honest mapping of the pre-registration's
           occupancy-shaped `const` onto a BOX panel, and it is named as such.
  within-clip shuffle: the lead labels PERMUTED INSIDE EACH CLIP. Clip identity survives
           it; frame-level evidence does not. Mandatory -- clip identity explained most of
           refav1's `n_agents` decode.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

TAUS = (1.0, 2.0, 4.0)


def average_precision(score, is_tp, n_gt):
    """All-point-interpolated AP from a scored TP/FP list."""
    if n_gt == 0 or len(score) == 0:
        return 0.0
    o = np.argsort(-score, kind="stable")
    tp = np.cumsum(is_tp[o])
    fp = np.cumsum(1 - is_tp[o])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def match(pred_xy, pred_s, row_id, gt_xy, gt_row, tau):
    """Greedy one-to-one centre matching within `tau` metres, per row."""
    o = np.argsort(-pred_s, kind="stable")
    gt_by_row = {}
    for i, r in enumerate(gt_row):
        gt_by_row.setdefault(int(r), []).append(i)
    used = np.zeros(len(gt_xy), dtype=bool)
    is_tp = np.zeros(len(pred_s), dtype=np.float64)
    for k in o:
        cand = gt_by_row.get(int(row_id[k]))
        if not cand:
            continue
        best, bd = -1, tau
        for gi in cand:
            if used[gi]:
                continue
            d = float(np.hypot(*(pred_xy[k] - gt_xy[gi])))
            if d <= bd:
                best, bd = gi, d
        if best >= 0:
            used[best] = True
            is_tp[k] = 1.0
    return is_tp


def recall_at_precision(score, is_tp, n_gt, p_min=0.5):
    if n_gt == 0 or len(score) == 0:
        return 0.0
    o = np.argsort(-score, kind="stable")
    tp = np.cumsum(is_tp[o])
    fp = np.cumsum(1 - is_tp[o])
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / n_gt
    ok = prec >= p_min
    return float(rec[ok].max()) if ok.any() else 0.0


def auc(scores, labels):
    """Rank AUC; exactly 0.5 for any constant score."""
    labels = np.asarray(labels, dtype=np.float64)
    n1, n0 = labels.sum(), (1 - labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.asarray(scores, dtype=np.float64))) + 1.0
    # average ranks for ties, so a constant score reads EXACTLY 0.5
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    rr = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        rr[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((rr[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def heatmap_peaks(d, cfg, topk=20):
    """CenterNet decode: 3x3 max-pool NMS on the sigmoid heatmap + sub-cell offsets.

    ⭐ WHY A SECOND DETECTOR. MEASURED here: at this budget the DETR slots are still near
    their init (all 20 queries within a metre of the prior mean), so slot AP@2m 0.0148 sits
    BELOW the zero-image `prior` 0.0221 -- a well-known DETR convergence property, not a
    statement about the tokens. The dense heatmap converges in the same steps. Both
    detectors are reported for every arm and the bars are applied to both.
    """
    H = d["heatmap"].astype(np.float32)                      # [B, GX, GY]
    OFF = d["offsets"].astype(np.float32)                    # [B, GX, GY, 2]
    GXc, GYc = cfg["grid_cells"]
    CELL, Y_ABS = cfg["cell_m"], cfg["y_abs_m"]
    B = H.shape[0]
    pad = np.pad(H, ((0, 0), (1, 1), (1, 1)), constant_values=-1.0)
    mx = np.max(np.stack([pad[:, i:i + GXc, j:j + GYc]
                          for i in range(3) for j in range(3)], 0), 0)
    keep = (H >= mx)
    xs, ys, ss = [], [], []
    for b in range(B):
        ii, jj = np.nonzero(keep[b])
        sc = H[b, ii, jj]
        o = np.argsort(-sc)[:topk]
        ii, jj, sc = ii[o], jj[o], sc[o]
        xs.append((ii + 0.5) * CELL + OFF[b, ii, jj, 0])
        ys.append((jj + 0.5) * CELL - Y_ABS + OFF[b, ii, jj, 1])
        ss.append(sc)
    n = np.array([len(v) for v in ss])
    prow = np.repeat(np.arange(B), n)
    return (np.stack([np.concatenate(xs), np.concatenate(ys)], 1),
            np.concatenate(ss), prow)


def flatten(d, det="slots", cfg=None):
    """A run's test_pred.npz -> flat prediction and GT tables."""
    rows = d["rows"]
    B, S = d["boxes"].shape[0], d["boxes"].shape[1]
    if det == "heatmap":
        pred_xy, pred_s, prow = heatmap_peaks(d, cfg)
    else:
        pred_xy = d["boxes"][:, :, :2].reshape(-1, 2)
        pred_s = d["score"].reshape(-1)
        prow = np.repeat(np.arange(B), S)
    gt_xy, gt_row = [], []
    for i in range(B):
        k = int(d["gt_n"][i])
        if k:
            gt_xy.append(d["gt_boxes"][i, :k, :2])
            gt_row.append(np.full(k, i))
    gt_xy = np.concatenate(gt_xy) if gt_xy else np.zeros((0, 2))
    gt_row = np.concatenate(gt_row).astype(int) if len(gt_row) else np.zeros(0, int)
    return rows, pred_xy, pred_s, prow, gt_xy, gt_row


def score_arm(d, clip_of_row, prior_xy=None, prior_s=None, det="slots", cfg=None):
    rows, pxy, ps, prow, gxy, grow = flatten(d, det, cfg)
    if prior_xy is not None:
        B = d["boxes"].shape[0]
        S = len(prior_xy)
        pxy = np.tile(prior_xy, (B, 1))
        ps = np.tile(prior_s, B)
        prow = np.repeat(np.arange(B), S)
    out = {}
    for tau in TAUS:
        tp = match(pxy, ps, prow, gxy, grow, tau)
        out[f"AP@{tau:g}m"] = average_precision(ps, tp, len(gxy))
        if tau == 2.0:
            out["recall@prec0.50_2m"] = recall_at_precision(ps, tp, len(gxy), 0.5)
            out["_tp2"] = tp
    out["_pred"] = (pxy, ps, prow)
    out["_gt"] = (gxy, grow)
    out["n_gt"] = int(len(gxy))
    out["n_pred"] = int(len(ps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=r"C:\Users\Admin\d3_out\box")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    runs = sorted(p for p in Path(a.runs).iterdir()
                  if (p / "test_pred.npz").exists())
    print(f"[panel] {len(runs)} arms: {[p.name for p in runs]}")

    panel, keep, cfgs = {}, {}, {}
    rng = np.random.default_rng(a.seed)
    ref = None
    for p in runs:
        d = dict(np.load(p / "test_pred.npz"))
        cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        cfgs[p.name] = cfg
        r = score_arm(d, d["clip_of_row"], det="heatmap", cfg=cfg)
        rs = score_arm(d, d["clip_of_row"], det="slots", cfg=cfg)
        lead_auc = auc(d["lead"], d["gt_lead"])
        # within-clip shuffle of the LABELS
        lab = d["gt_lead"].copy()
        for c in np.unique(d["clip_of_row"]):
            m = np.nonzero(d["clip_of_row"] == c)[0]
            lab[m] = rng.permutation(lab[m])
        row = {"tag": p.name, "arm": cfg["arm"], "tokens": cfg["tokens"],
               "seed": cfg["seed"], "n_test_rows": int(len(d["rows"])),
               "n_test_clips": int(len(np.unique(d["clip_of_row"]))),
               "n_gt_boxes": r["n_gt"], "wall_s": cfg.get("wall_s"),
               "lead_prevalence_test": cfg.get("lead_prevalence_test"),
               **{k: round(v, 4) for k, v in r.items() if not k.startswith("_")},
               "detector": "heatmap peaks (3x3 NMS + sub-cell offsets)",
               "slots": {k: round(v, 4) for k, v in rs.items()
                         if not k.startswith("_")},
               "lead_auc": round(lead_auc, 4),
               "lead_auc_within_clip_shuffle": round(auc(d["lead"], lab), 4),
               "lead_auc_minus_shuffle": round(lead_auc - auc(d["lead"], lab), 4)}
        # closest-in-path range MAE: the highest-scoring prediction with |y| <= 2 m
        m = np.isfinite(d["gt_lead_range"])
        if m.sum() > 5:
            err = []
            for i in np.nonzero(m)[0]:
                b, s = d["boxes"][i], d["score"][i]
                inpath = np.abs(b[:, 1]) <= 2.0
                if not inpath.any():
                    continue
                j = np.nonzero(inpath)[0][np.argmax(s[inpath])]
                err.append(abs(float(b[j, 0]) - float(d["gt_lead_range"][i])))
            row["lead_range_mae_m"] = round(float(np.mean(err)), 4) if err else None
            row["lead_range_n"] = int(len(err))
        panel[p.name] = row
        keep[p.name] = (d, r)
        if cfg["arm"] == "main" and cfg["seed"] == 0:
            ref = ref or p.name
        print(f"  {p.name:16s} AP@2m {row['AP@2m']:.4f}  AP@1m {row['AP@1m']:.4f}  "
              f"AP@4m {row['AP@4m']:.4f}  leadAUC {row['lead_auc']:.4f} "
              f"(-shuf {row['lead_auc_minus_shuffle']:+.4f})", flush=True)

    # -- analytic controls --------------------------------------------------- #
    d0, _ = keep[runs[0].name]
    cfg0 = json.loads((runs[0] / "config.json").read_text(encoding="utf-8"))
    GX, GY = cfg0["grid_cells"]
    CELL, Y_ABS = cfg0["cell_m"], cfg0["y_abs_m"]
    # the TRAIN-split centre rate, rebuilt here from the join's own targets
    import p_box_head as H
    idxf = np.load(Path(cfg0["tok_dir"]) / "index.npz")
    T = H.build_targets(idxf)
    tr = np.nonzero(T["has_row"] & (T["split"] == 0))[0]
    rate = np.zeros((GX, GY), dtype=np.float64)
    for r_ in tr:
        for j in range(int(T["nbox"][r_])):
            ix = int(T["boxes"][r_, j, 0] / CELL)
            iy = int((T["boxes"][r_, j, 1] + Y_ABS) / CELL)
            if 0 <= ix < GX and 0 <= iy < GY:
                rate[ix, iy] += 1
    rate /= max(len(tr), 1)
    flat = rate.reshape(-1)
    top = np.argsort(-flat)[:cfg0["n_slots"]]
    prior_xy = np.stack([(top // GY + 0.5) * CELL,
                         (top % GY + 0.5) * CELL - Y_ABS], axis=1)
    prior_s = flat[top]
    pr = score_arm(d0, d0["clip_of_row"], prior_xy=prior_xy, prior_s=prior_s,
                   det="slots", cfg=cfg0)
    panel["prior"] = {"tag": "prior", "arm": "prior (analytic, zero-image)",
                      "n_gt_boxes": pr["n_gt"],
                      **{k: round(v, 4) for k, v in pr.items()
                         if not k.startswith("_")}}
    prev = float(d0["gt_lead"].mean())
    panel["const"] = {
        "tag": "const", "arm": "const (analytic)",
        "lead_accuracy_is_prevalence": round(prev, 6),
        "lead_auc": round(auc(np.zeros(len(d0["gt_lead"])), d0["gt_lead"]), 6),
        "_known_values": "accuracy == test prevalence EXACTLY; AUC == 0.5000 EXACTLY"}
    print(f"  {'prior':16s} AP@2m {panel['prior']['AP@2m']:.4f}")
    print(f"  {'const':16s} lead AUC {panel['const']['lead_auc']:.6f} "
          f"(prevalence {prev:.6f})")

    # -- clip-cluster bootstrap on the MARGINS -------------------------------- #
    # ⭐ THE TP FLAGS ARE EXACTLY INVARIANT UNDER CLIP RESAMPLING. Matching is
    # per-ROW (a prediction can only match GT in its own row) and rows nest inside
    # clips, so resampling clips never changes which predictions are TP -- only
    # which rows enter the pooled precision/recall curve. The match is therefore
    # computed ONCE per arm per tau and the bootstrap re-pools it. (Re-running the
    # greedy matcher inside 1,000 draws would be the same numbers, 3 orders of
    # magnitude slower.)
    def arm_tables(tag, prior=False, det="heatmap"):
        d, c = keep[tag][0], cfgs[tag]
        rows, pxy, ps, prow, gxy, grow = flatten(d, det, c)
        if prior:
            B, S = d["boxes"].shape[0], len(prior_xy)
            pxy = np.tile(prior_xy, (B, 1))
            ps = np.tile(prior_s, B)
            prow = np.repeat(np.arange(B), S)
        tp = match(pxy, ps, prow, gxy, grow, 2.0)
        clip_of = d["clip_of_row"]
        pclip, gclip = clip_of[prow], clip_of[grow]
        # precompute per-clip index blocks ONCE: rescanning 134k rows per clip per
        # draw is 4.7 billion comparisons for a 1,000-draw bootstrap.
        cl = np.unique(clip_of)
        pidx = {int(c): np.nonzero(pclip == c)[0] for c in cl}
        gcnt = {int(c): int((gclip == c).sum()) for c in cl}
        return ps, tp, pidx, gcnt

    clips = np.unique(d0["clip_of_row"])
    # ⛔ BOTH DETECTORS GET THE SAME BOOTSTRAP. Picking the detector after seeing which
    # one wins would be a selection the pre-registration does not license.
    TABS = {dt: {t: arm_tables(t, det=dt) for t in keep} for dt in ("slots", "heatmap")}
    for dt in TABS:
        TABS[dt]["prior"] = arm_tables(runs[0].name, prior=True, det=dt)

    def ap_draw(tab, cs):
        ps, tp, pidx, gcnt = tab
        pi = np.concatenate([pidx[int(c)] for c in cs])
        n_gt = int(sum(gcnt[int(c)] for c in cs))
        return average_precision(ps[pi], tp[pi], n_gt)

    margins = {"slots": {}, "heatmap": {}}
    names = list(panel)
    main_tags = [t for t in names if panel[t].get("arm") == "main" and t in keep]
    for dt in ("slots", "heatmap"):
      tabs = TABS[dt]
      for tag in main_tags:
        others = [t for t in names
                  if t in keep and panel[t].get("arm") in ("shuffled", "pixel", "mirror")
                  and panel[t].get("tokens") in (panel[tag].get("tokens"), "pix64")]
        peers = [t for t in main_tags if t != tag]
        for o in others + peers + ["prior"]:
            rs = np.random.default_rng(a.seed)
            draws = np.empty(a.n_boot)
            base = ap_draw(tabs[tag], clips) - ap_draw(tabs[o], clips)
            for b in range(a.n_boot):
                cs = rs.choice(clips, len(clips), replace=True)
                draws[b] = ap_draw(tabs[tag], cs) - ap_draw(tabs[o], cs)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            margins[dt][f"{tag}_minus_{o}"] = {
                "delta_AP@2m": round(float(base), 4),
                "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                "separated": bool(lo > 0 or hi < 0), "n_boot": int(a.n_boot),
                "n_clips": int(len(clips))}
            print(f"  [{dt:7s}] {tag} - {o}: {base:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] sep="
                  f"{margins[dt][f'{tag}_minus_{o}']['separated']}", flush=True)

    out = {"_what": "Rung 2 box panel on frozen refcv5-v2 tokens vs the B1 EVAL agent join",
           "evidence_class": "MEASURED (ours)",
           "matching": "centre distance, one-to-one, greedy by score; "
                       "all-point-interpolated AP",
           "clusters": "CLIPS (test split), never frames",
           "n_boot": a.n_boot, "seed": a.seed,
           "arms": panel, "margins": margins}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("[out]", a.out)


if __name__ == "__main__":
    main()
