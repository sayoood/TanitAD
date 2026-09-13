#!/usr/bin/env python3
"""P4 eval - the `E-BEVHEAD-FROZEN-1` panel: gates G1/G2, bars B1-B4, per-band metrics,
paired clip-cluster bootstrap. Reads the trained arms' `runs/<arm>/` outputs.

Estimators (PREREG §4, fixed before any number):
  * AP  -- pooled over all scored test cells, `sklearn.metrics.average_precision_score`
           (the POINT estimate). Bootstrap resamples recompute AP from per-clip score
           histograms (2,001 probability bins); the binning error on the full test set is
           measured and reported next to every CI.
  * IoU -- occupied class at tau chosen on VAL per arm (grid 0.05..0.95), from per-clip
           TP/FP/FN counts.
  * CI  -- PAIRED clip-cluster bootstrap over TEST clips, 2,000 resamples, seed 0: every arm
           is resampled with the SAME clip multiset, so a difference's CI is paired.
  ⛔ `overlapping_holdout_se` is never used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p4_bev_head as H  # noqa: E402

NB = 2001
BANDS = {"0-15m": (0, 12), "15-30m": (12, 24), "30-45m": (24, 36), "45-60m": (36, 48)}
TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def ap_exact(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if y.any() and (~y).any() else float("nan")


def ap_from_hist(hp: np.ndarray, hn: np.ndarray) -> np.ndarray:
    """AP from [R, NB] positive/negative histograms (bin NB-1 = highest score)."""
    tp = np.cumsum(hp[:, ::-1], axis=1)
    fp = np.cumsum(hn[:, ::-1], axis=1)
    P = tp[:, -1:]
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)
    rec = tp / np.maximum(P, 1e-12)
    drec = np.diff(np.concatenate([np.zeros((rec.shape[0], 1)), rec], axis=1), axis=1)
    return (drec * prec).sum(axis=1)


def iou_counts(y, s, tau):
    pred = s >= tau
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="main,main_s1,shuffled,pixel")
    ap.add_argument("--runs", default=str(H.WORK / "runs"))
    ap.add_argument("--out", default=str(HERE.parent / "raw" / "p4_panel.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alias", default="", help="comma list name=run_dir")
    ap.add_argument("--label", default="E-BEVHEAD-FROZEN-1 frozen rung")
    ap.add_argument("--extra-train", action="store_true",
                    help="L4: the FIRST arm trained with the extra clips; its train rows define prior/const")
    args = ap.parse_args()

    D = H.load_panel_data()
    if args.extra_train:
        Xr = H.load_extra_rows()
        for k in ("occ", "mB", "mC"):
            D[k] = np.concatenate([D[k], Xr[k]])
        D["clip_of_row"] = np.concatenate([D["clip_of_row"], np.full(len(Xr["occ"]), -1, np.int32)])
    arms_trained = [a for a in args.arms.split(",") if a]
    alias = dict(kv.split("=", 1) for kv in args.alias.split(",") if kv)
    run_dir = {a: Path(args.runs) / alias.get(a, a) for a in arms_trained}
    rows0 = np.load(run_dir[arms_trained[0]] / "rows.npz")
    tr, va, te = rows0["train"], rows0["val"], rows0["test"]
    for a in arms_trained[1:]:
        r = np.load(run_dir[a] / "rows.npz")
        if not (np.array_equal(r["test"], te) and np.array_equal(r["val"], va)):
            raise SystemExit(f"arm {a} was scored on DIFFERENT rows -- the panel is not paired")
    occ, mB, mC = D["occ"], D["mB"], D["mC"]
    clip_te = D["clip_of_row"][te]
    clips = np.unique(clip_te)

    # ---------------- scores per arm on test (and val for tau) ----------------
    train_marg = float(occ[tr][mB[tr]].mean())
    cell_cnt = mB[tr].sum(0)
    cell_pos = (occ[tr] & mB[tr]).sum(0)
    prior = np.where(cell_cnt > 0, cell_pos / np.maximum(cell_cnt, 1), train_marg).astype(np.float32)
    scores_te, scores_va = {}, {}
    scores_te["const"] = np.full(occ[te].shape, train_marg, np.float32)
    scores_va["const"] = np.full(occ[va].shape, train_marg, np.float32)
    scores_te["prior"] = np.broadcast_to(prior, occ[te].shape)
    scores_va["prior"] = np.broadcast_to(prior, occ[va].shape)
    cfgs = {}
    for a in arms_trained:
        scores_te[a] = np.load(run_dir[a] / "test_probs.npy").astype(np.float32)
        scores_va[a] = np.load(run_dir[a] / "val_probs.npy").astype(np.float32)
        cfgs[a] = json.loads((run_dir[a] / "config.json").read_text(encoding="utf-8"))
    arms = ["const", "prior"] + arms_trained

    yT, mT, mTC = occ[te], mB[te], mC[te]
    yV, mV = occ[va], mB[va]
    res: dict = {"schema": "tanitad.bevhead_panel/1", "label": args.label,
                 "evidence_class": "MEASURED (ours, dev-box RTX 4060; test split scored once)",
                 "tier": "N/A -- representation probe, not a driving evaluation",
                 "estimator": "paired clip-cluster bootstrap over TEST clips, n_boot=%d, seed 0; "
                              "AP per resample from per-clip 2001-bin score histograms" % args.n_boot,
                 "n": {"train_clips": int(len(np.unique(D["clip_of_row"][tr][D["clip_of_row"][tr] >= 0]))),
                       "extra_train_rows": int((D["clip_of_row"][tr] < 0).sum()),
                       "val_clips": int(len(np.unique(D["clip_of_row"][va]))),
                       "test_clips": int(len(clips)), "train_rows": int(len(tr)),
                       "val_rows": int(len(va)), "test_rows": int(len(te)),
                       "test_scored_cells_ruleB": int(mT.sum()),
                       "test_scored_cells_ruleC": int(mTC.sum()),
                       "excluded_clips": len(D["excluded_clips"])},
                 "prevalence_test_ruleB": float(yT[mT].mean()),
                 "prevalence_test_ruleC": float(yT[mTC].mean()),
                 "train_marginal_ruleB": train_marg,
                 "d": {a: {"d_in": cfgs[a]["d_in"], "grid": cfgs[a]["grid"],
                           "head_params": cfgs[a]["head_params"], "tokens": cfgs[a]["tokens"],
                           "best_step": cfgs[a].get("best_step"),
                           "best_val_ap_B": cfgs[a].get("best_val_ap_B")} for a in arms_trained},
                 "arms": {}, "alias": alias}

    # ---------------- point estimates ----------------
    tau_of = {}
    for a in arms:
        sv, st = scores_va[a], scores_te[a]
        ious = []
        for t in TAUS:
            tp, fp, fn = iou_counts(yV[mV], sv[mV], t)
            ious.append(tp / max(tp + fp + fn, 1))
        tau = float(TAUS[int(np.argmax(ious))])
        tau_of[a] = tau
        rec = {"tau_from_val": tau,
               "ap_all_B": ap_exact(yT[mT], st[mT]),
               "ap_all_C": ap_exact(yT[mTC], st[mTC])}
        tp, fp, fn = iou_counts(yT[mT], st[mT], tau)
        rec["iou_all_B"] = tp / max(tp + fp + fn, 1)
        tp, fp, fn = iou_counts(yT[mTC], st[mTC], tau)
        rec["iou_all_C"] = tp / max(tp + fp + fn, 1)
        bands = {}
        for bn, (r0, r1) in BANDS.items():
            mb = np.zeros_like(mT)
            mb[:, r0:r1, :] = mT[:, r0:r1, :]
            tp, fp, fn = iou_counts(yT[mb], st[mb], tau)
            bands[bn] = {"ap": ap_exact(yT[mb], st[mb]), "iou": tp / max(tp + fp + fn, 1),
                         "prevalence": float(yT[mb].mean()) if mb.any() else float("nan"),
                         "n_cells": int(mb.sum())}
        m030 = np.zeros_like(mT)
        m030[:, 0:24, :] = mT[:, 0:24, :]
        tp, fp, fn = iou_counts(yT[m030], st[m030], tau)
        rec["iou_0_30m_B"] = tp / max(tp + fp + fn, 1)
        rec["ap_0_30m_B"] = ap_exact(yT[m030], st[m030])
        rec["bands_B"] = bands
        res["arms"][a] = rec

    # ---------------- per-clip histograms for the bootstrap ----------------
    def hists(a, mask):
        hp = np.zeros((len(clips), NB))
        hn = np.zeros((len(clips), NB))
        for k, c in enumerate(clips):
            rr = clip_te == c
            y = yT[rr][mask[rr]]
            s = scores_te[a][rr][mask[rr]]
            b = np.clip(np.round(s * (NB - 1)).astype(int), 0, NB - 1)
            hp[k] = np.bincount(b[y], minlength=NB)
            hn[k] = np.bincount(b[~y], minlength=NB)
        return hp, hn

    def counts(a, mask, tau, rows_mask=None):
        c3 = np.zeros((len(clips), 3))
        for k, c in enumerate(clips):
            rr = clip_te == c
            mk = mask[rr]
            if rows_mask is not None:
                mk = mk & rows_mask[rr]
            c3[k] = iou_counts(yT[rr][mk], scores_te[a][rr][mk], tau)
        return c3

    band030 = np.zeros_like(mT)
    band030[:, 0:24, :] = True
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(clips), len(clips)), minlength=len(clips))
                  for _ in range(args.n_boot)]).astype(np.float64)
    H_ = {a: hists(a, mT) for a in arms}
    C_ = {a: counts(a, mT, tau_of[a]) for a in arms}
    C030 = {a: counts(a, mT, tau_of[a], band030) for a in arms}
    boot_ap = {a: ap_from_hist(W @ H_[a][0], W @ H_[a][1]) for a in arms}
    binerr = {a: abs(float(ap_from_hist(H_[a][0].sum(0)[None], H_[a][1].sum(0)[None])[0])
                     - res["arms"][a]["ap_all_B"]) for a in arms}

    def iou_boot(C):
        S = W @ C
        return S[:, 0] / np.maximum(S.sum(1), 1)

    for a in arms:
        lo, hi = np.percentile(boot_ap[a], [2.5, 97.5])
        res["arms"][a]["ap_all_B_ci95"] = [float(lo), float(hi)]
        res["arms"][a]["ap_hist_binning_error"] = binerr[a]
        b030 = iou_boot(C030[a])
        res["arms"][a]["iou_0_30m_B_ci95"] = [float(np.percentile(b030, 2.5)),
                                              float(np.percentile(b030, 97.5))]

    def diff(a, b):
        d = boot_ap[a] - boot_ap[b]
        pt = res["arms"][a]["ap_all_B"] - res["arms"][b]["ap_all_B"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta_ap_B": pt, "ci95": [float(lo), float(hi)], "separated": bool(lo > 0 or hi < 0)}

    pairs = {}
    for a, b in (("main", "shuffled"), ("main", "prior"), ("main", "pixel"), ("main", "main_s1"),
                 ("shuffled", "prior"), ("pixel", "prior"), ("main_s1", "shuffled")):
        if a in arms and b in arms:
            pairs[f"{a}-{b}"] = diff(a, b)
    extra = [a for a in arms_trained if a not in ("main", "main_s1", "shuffled", "pixel")]
    for a in extra:
        for b in ("main", "shuffled", "prior", "pixel"):
            if b in arms:
                pairs[f"{a}-{b}"] = diff(a, b)
        if f"{a}_s1" in arms:
            pairs[f"{a}-{a}_s1"] = diff(a, f"{a}_s1")
    res["pairs"] = pairs

    # ---------------- gates and bars (PREREG §5, literals) ----------------
    A = res["arms"]
    F = abs(A["main"]["ap_all_B"] - A["main_s1"]["ap_all_B"]) if "main_s1" in A else float("nan")
    res["replicate_floor_F"] = F
    G1 = abs(A["const"]["ap_all_B"] - res["prevalence_test_ruleB"]) <= 1e-6
    G2 = A["shuffled"]["ap_all_B"] <= A["prior"]["ap_all_B"] + 0.02 if "shuffled" in A else None

    def bar(pair, min_delta, floor):
        p = pairs[pair]
        return bool(p["delta_ap_B"] >= min_delta and p["ci95"][0] > 0 and p["delta_ap_B"] >= 3 * floor)

    def verdict_for(lever):
        # a lever with its OWN replicate uses max(F_main, |lever - lever_s1|); without one it
        # BORROWS the main rig's floor, and the verdict says so
        floor, floor_src = F, "borrowed from main/main_s1"
        if f"{lever}_s1" in A:
            own = abs(A[lever]["ap_all_B"] - A[f"{lever}_s1"]["ap_all_B"])
            floor, floor_src = max(F, own), f"max(F_main, |{lever} - {lever}_s1| = {own:.5f})"
        out = {"replicate_floor_used": floor, "floor_source": floor_src if lever != "main" else "main/main_s1"}
        out["B1_information"] = bar(f"{lever}-shuffled", 0.05, floor)
        out["B2_beyond_position"] = bar(f"{lever}-prior", 0.05, floor)
        out["B3_beyond_pixels"] = (bar(f"{lever}-pixel", 1e-12, floor) if f"{lever}-pixel" in pairs
                                   else "NOT RE-TESTED (no pixel arm on this training set)")
        out["B4_usefulness"] = bool(A[lever]["ap_all_B"] >= 0.60 and A[lever]["iou_0_30m_B"] >= 0.45)
        out["PASS"] = all(v is True for k, v in out.items() if k.startswith("B") and not isinstance(v, str))
        if any(isinstance(v, str) for k, v in out.items() if k.startswith("B")):
            out["PASS_scope"] = "B3 not re-tested"
        return out

    res["gates"] = {"G1_const_reads_prevalence": bool(G1),
                    "G1_detail": [A["const"]["ap_all_B"], res["prevalence_test_ruleB"]],
                    "G2_shuffled_le_prior_plus_0.02": G2,
                    "G2_detail": [A.get("shuffled", {}).get("ap_all_B"), A["prior"]["ap_all_B"]]}
    res["verdict_main"] = verdict_for("main") if G1 and G2 else "INSTRUMENT GATE FAILED (F3)"
    for a in extra:
        if a.endswith("_s1"):
            continue                      # a replicate is a floor, not a lever
        res[f"verdict_{a}_POST_HOC_LEVER"] = verdict_for(a) if G1 and G2 else "INSTRUMENT GATE FAILED"

    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---------------- printed table ----------------
    print(f"n: {res['n']}")
    print(f"prevalence test B {res['prevalence_test_ruleB']:.4f} C {res['prevalence_test_ruleC']:.4f}"
          f"  train marginal {train_marg:.4f}  replicate floor F {F:.4f}")
    print(f"{'arm':10s} {'AP_B':>7s} {'CI95':>17s} {'AP_C':>7s} {'IoU_B':>6s} {'IoU0-30':>7s} {'tau':>5s}  bands AP (0-15/15-30/30-45/45-60)")
    for a in arms:
        r = A[a]
        bb = "/".join(f"{r['bands_B'][b]['ap']:.3f}" for b in BANDS)
        print(f"{a:10s} {r['ap_all_B']:7.4f} [{r['ap_all_B_ci95'][0]:.4f},{r['ap_all_B_ci95'][1]:.4f}] "
              f"{r['ap_all_C']:7.4f} {r['iou_all_B']:6.4f} {r['iou_0_30m_B']:7.4f} {r['tau_from_val']:5.2f}  {bb}")
    for k, p in pairs.items():
        print(f"{k:20s} dAP {p['delta_ap_B']:+.4f} CI [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] sep={p['separated']}")
    print("gates", res["gates"])
    print("verdict main", res["verdict_main"])
    for a in extra:
        if not a.endswith("_s1"):
            print(f"verdict {a} (POST-HOC lever)", res[f"verdict_{a}_POST_HOC_LEVER"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
