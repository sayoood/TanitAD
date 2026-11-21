"""B4 - MATCHED-CAPACITY vision heads for the situation classifier.

THE QUESTION
------------
P8 measured that on `roundabout` a small ridge beat the deployed transformer on the SAME vision
features and concluded *"the head is the bottleneck, not the features"*. Two arms are not a curve.
This sweeps the deployed architecture across its whole capacity range on ONE substrate and asks
where the curve actually peaks - and whether the linear floor really beats the deployed rung.

PI RULING 2026-08-03, binding: labels may use ego; INFERENCE IS VISION-ONLY. Every arm here reads
the frozen v1 camera latents and nothing else. No ego-input arm is built.

THE LADDER (rungs are PARAMETER BUDGETS, widths solved by
`tanitad.eval.sitclf.width_for_param_budget`, so a change of PCA rank cannot silently move a rung)

  LIN   closed-form ridge, `sitclf.ridge_scores` - the same estimator that produced the banked
        `ridge_img` arm. Four representations at the linear floor:
          ridge_pca16_w8   129 params/head  <- the banked `ridge_img` recipe EXACTLY
          ridge_pca64_w8   513
          ridge_pca256_w8  2049             <- param-matched to the next row
          ridge_raw2048_w1 2049             <- same budget, NO pca and NO temporal window
  TF    `sitclf.CausalSitHead` (== the deployed `sc_train.SitHead`) at
          d = 8 / 16 / 32 / 64 / 128 / 196 / 296
        = 2,068 / 7,332 / 27,460 / 106,116 / 417,028 / 971,772 / 2,207,572 params.
        d=128 IS the deployed arm's exact configuration and parameter count.
  REP   the deployed budget on PCA-64 instead of PCA-16, to separate "more head" from "more input".
  OPT   the deployed and the largest rung at 3x the epoch budget - B4 exists to discriminate an
        "optimisation shortcut" from a "representation limit", and without this control a losing
        big head is indistinguishable from an undertrained one.

CONTROLS - the permuted-feature null is fitted and scored BEFORE any real arm is quoted
  NEG_FEATURE  every rung refitted with the image features permuted ACROSS clips, labels untouched.
               A rung that scores above its base rate on destroyed features would mean the protocol
               manufactures signal and the whole ladder is void.
  NEG_LABEL    the floor and the deployed rung scored against labels permuted across whole CLUSTERS.
  SELF-CONSISTENCY  the TACTICAL family's AP-lift is recomputed independently on the family's own
               row set and asserted equal to the arm table's - a component-vs-family control, so a
               family number cannot drift from the components it summarises.

PROTOCOL
  * 2 outer folds over whole clip CLUSTERS; a rung's score for a row comes from a fit that never saw
    that row's clip.
  * inside each outer TRAIN half, 20 % of clusters are held out to SELECT THE EPOCH (transformer) or
    LAMBDA (ridge). This matters: the banked `head_img` was selected at epoch 1 of 8, so a
    fixed-epoch ladder would not be comparing like with like. Every rung gets the same protocol.
  * PCA is fitted on the FIT rows of that fold only.
  * every rung is scored on IDENTICAL rows, so the paired estimator is valid.
  * scores are banked to .scores.npz after EVERY rung, so a killed run still leaves a re-analysable
    artifact and any re-scoring needs no GPU.

usage:
  python run_matched_capacity.py --substrate <npz> --out results.json [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.ap_ci import (ap_episode_cluster_bootstrap,        # noqa: E402
                                ap_lift, average_precision,
                                paired_ap_episode_cluster_bootstrap)
from tanitad.eval.sitclf import (CausalSitHead, causal_window,       # noqa: E402
                                 clip_runs, cluster_folds,
                                 head_param_count, predict_sit_head,
                                 ridge_param_count, ridge_scores,
                                 width_for_param_budget)
from tanitad.eval.sitclf_deploy import (ScoreBundle,                 # noqa: E402
                                        four_family_report,
                                        permute_labels_by_cluster,
                                        precision_recall_at_budget)

WIN = 8                       # sc_train.py:37 - held FIXED so capacity is the only axis
EPOCHS = 8                    # sc_train.py CONFIGS
POS_WEIGHT = 20.0
SEL_FRAC = 0.20               # clusters held out of each fold's fit to select epoch / lambda
TF_BUDGETS = (2_000, 7_000, 25_000, 100_000, 417_028, 950_000, 2_170_000)
DEPLOYED_RUNG = "tf_pca16_d128"
FLOOR_RUNG = "ridge_pca16_w8"
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)   # sc_train.py:46


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# feature construction                                                        #
# --------------------------------------------------------------------------- #
def fit_pca(F, rows, r, seed=0, n_sample=150_000, device="cpu"):
    """PCA mean + top-r basis on the given rows ONLY (leak-free per fold)."""
    g = np.random.default_rng(seed)
    idx = rows if rows.size <= n_sample else np.sort(g.choice(rows, n_sample, replace=False))
    A = torch.from_numpy(np.asarray(F[idx], dtype=np.float32)).to(device)
    mu = A.mean(0, keepdim=True)
    A = A - mu
    _u, _s, v = torch.svd_lowrank(A, q=min(r + 16, A.shape[1] - 1), niter=4)
    del A
    if device != "cpu":
        torch.cuda.empty_cache()
    return mu.cpu().numpy(), v[:, :r].cpu().numpy().astype(np.float32)


def project(F, mu, W):
    """`arm_features`'s image block: PCA then divide by the global mean-abs."""
    img = (np.asarray(F, dtype=np.float32) - mu) @ W
    img /= max(float(np.abs(img).mean()), 1e-6)
    return img


def standardise_raw(F, rows):
    """Raw 2048-d readout, standardised on `rows` only (the no-PCA representation)."""
    X = np.asarray(F, dtype=np.float32)
    mu = X[rows].mean(0, keepdims=True)
    sd = np.maximum(X[rows].std(0, keepdims=True), 1e-3)
    return (X - mu) / sd


# --------------------------------------------------------------------------- #
# the fold machinery                                                          #
# --------------------------------------------------------------------------- #
def fold_splits(cc, folds, f, seed=0):
    """-> (fit_rows, sel_rows, test_rows) with the selection split on CLUSTERS."""
    te = folds == f
    tr = ~te
    tr_cl = np.unique(cc[tr])
    rng = np.random.default_rng(seed + f)
    perm = rng.permutation(tr_cl)
    n_sel = max(1, int(round(SEL_FRAC * len(tr_cl))))
    sel_cl = set(int(x) for x in perm[:n_sel])
    is_sel = np.array([int(x) in sel_cl for x in cc])
    return (tr & ~is_sel), (tr & is_sel), te


def mean_ap(y, s, v):
    """Mean AP over the situations that have a positive - the selection score."""
    vals = []
    for i in range(y.shape[1]):
        m = v[:, i]
        if m.sum() < 50 or y[m, i].sum() < 5:
            continue
        vals.append(average_precision(y[m, i], s[m, i]))
    return float(np.mean(vals)) if vals else float("nan")


def train_tf_fold(X, Y, V, fit, sel, te, in_dim, d, device, seed=0, tag="",
                  epochs=EPOCHS):
    """One outer fold of one transformer rung; epoch selected on `sel`, scores on `te`."""
    torch.manual_seed(seed)
    m = CausalSitHead(in_dim, WIN, d=d, n_out=Y.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    pw = torch.full((Y.shape[1],), POS_WEIGHT, device=device)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    Xf = torch.from_numpy(np.ascontiguousarray(X[fit], dtype=np.float32))
    Yf = torch.from_numpy(np.ascontiguousarray(Y[fit], dtype=np.float32))
    Vf = torch.from_numpy(np.ascontiguousarray(V[fit], dtype=np.float32))
    n = Xf.shape[0]
    g = torch.Generator().manual_seed(seed)
    Ysel, Vsel = Y[sel].astype(np.int64), V[sel].astype(bool)
    best = (-np.inf, None, 0)
    curve = []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, 1024):
            j = perm[b:b + 1024]
            vb = Vf[j].to(device)
            if float(vb.sum()) == 0:
                continue
            xb = Xf[j].to(device).view(len(j), WIN, in_dim)
            loss = (lossf(m(xb), Yf[j].to(device)) * vb).sum() / vb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        a = mean_ap(Ysel, predict_sit_head(m, X[sel], in_dim=in_dim, device=device), Vsel)
        curve.append(round(float(a), 5))
        if np.isfinite(a) and a > best[0]:
            best = (a, predict_sit_head(m, X[te], in_dim=in_dim, device=device), ep + 1)
    log(f"    {tag}d={d} fit {int(fit.sum()):,} sel {int(sel.sum()):,} te {int(te.sum()):,} "
        f"epoch*={best[2]}/{epochs} sel_mAP={best[0]:.5f} curve={curve}")
    return best[1], best[2], curve


def train_ridge_fold(X, Y, V, fit, sel, te, tag=""):
    """One outer fold of the ridge floor; lambda selected on `sel`, scores on `te`."""
    best = (-np.inf, None)
    for lam in LAMBDAS:
        a = mean_ap(Y[sel].astype(np.int64),
                    ridge_scores(X[fit], Y[fit], V[fit], X[sel], lam=lam),
                    V[sel].astype(bool))
        if np.isfinite(a) and a > best[0]:
            best = (a, lam)
    lam = best[1] if best[1] is not None else 1.0
    log(f"    {tag}ridge fit {int(fit.sum()):,} lam*={lam:g} sel_mAP={best[0]:.5f}")
    return ridge_scores(X[fit], Y[fit], V[fit], X[te], lam=lam), lam


# --------------------------------------------------------------------------- #
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--substrate", required=True)
    ap_.add_argument("--out", default="results_matched_capacity.json")
    ap_.add_argument("--n-boot", type=int, default=2000)
    ap_.add_argument("--strata-n-boot", type=int, default=500)
    ap_.add_argument("--device", default=None)
    a = ap_.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_scores = Path(a.out).with_suffix(".scores.npz")

    z = np.load(a.substrate)
    F = z["F"]
    Yi = z["Y"].astype(np.int64)
    Y = z["Y"].astype(np.float32)
    E = z["E"]
    cc = z["clip_cluster"]
    sits = [str(s) for s in z["situations"]]
    st, en = clip_runs(cc)
    folds = cluster_folds(cc, 2, seed=0)
    log(f"substrate {F.shape[0]:,} rows x {F.shape[1]}  {len(st)} clips  device={device}")

    # rows with a full in-clip causal history - identical for every arm
    _, hist_ok = causal_window(np.zeros((len(cc), 1), np.float32), st, en, WIN)
    V = z["V"].astype(bool) & hist_ok[:, None]
    Vf = V.astype(np.float32)
    log(f"common scorable rows (full {WIN}-frame history): {int(hist_ok.sum()):,}")
    for i, s in enumerate(sits):
        log(f"  {s}: scorable {int(V[:, i].sum()):,} pos {int(Yi[V[:, i], i].sum()):,} "
            f"base {float(Yi[V[:, i], i].mean()):.5f}")

    # --- the clip-level permutation that defines NEG_FEATURE ------------------
    rng = np.random.default_rng(0)
    perm_clip = rng.permutation(len(st))
    shuf_rows = np.arange(len(cc))
    for i, (s0, e0) in enumerate(zip(st, en)):
        j = perm_clip[i]
        n = min(e0 - s0, en[j] - st[j])
        shuf_rows[s0:s0 + n] = st[j] + np.arange(n)
        if e0 - s0 > n:
            shuf_rows[s0 + n:e0] = st[j] + n - 1

    splits = [fold_splits(cc, folds, f, seed=0) for f in (0, 1)]
    scores: dict[str, np.ndarray] = {}
    spec: dict[str, dict] = {}

    def rung_features(r, fold, shuf):
        """Windowed features for one fold, fitted on that fold's FIT rows only."""
        fit = splits[fold][0]
        src = F[shuf_rows] if shuf else F
        if r == 0:                             # raw readout, NO temporal window
            return standardise_raw(src, np.flatnonzero(fit))
        mu, W = fit_pca(src, np.flatnonzero(fit), r, seed=0, device=device)
        X, _ = causal_window(project(src, mu, W), st, en, WIN)
        return X

    def run_rung(name, kind, r, budget=None, shuf=False, epochs=EPOCHS):
        t0 = time.time()
        out = np.zeros_like(Y, dtype=np.float32)
        chosen, curves = [], []
        d = width_for_param_budget(budget, r, WIN, Y.shape[1]) if kind == "tf" else None
        for f in (0, 1):
            X = rung_features(r, f, shuf)
            fit, sel, te = splits[f]
            if kind == "tf":
                s_te, ep, curve = train_tf_fold(X, Y, Vf, fit, sel, te, r, d, device,
                                                seed=0, tag=f"{name} ", epochs=epochs)
                chosen.append(ep)
                curves.append(curve)
            else:
                s_te, lam = train_ridge_fold(X, Y, Vf, fit, sel, te, tag=f"{name} ")
                chosen.append(lam)
            out[te] = s_te
            del X
        if kind == "tf":
            npar = head_param_count(r, WIN, d, Y.shape[1])
            spec[name] = {"kind": "CausalSitHead", "pca_r": r, "win": WIN, "d": d,
                          "params_per_head": npar, "budget": budget,
                          "epochs_trained": epochs, "epochs_selected": chosen,
                          "sel_curve": curves, "shuffled_features": shuf}
        else:
            flat = r * WIN if r > 0 else int(F.shape[1])
            spec[name] = {"kind": "closed-form ridge",
                          "pca_r": (r if r > 0 else "RAW-2048"),
                          "win": (WIN if r > 0 else 1), "flat_dim": flat,
                          "params_per_head": ridge_param_count(flat, per_situation=False),
                          "lambda_selected": chosen, "shuffled_features": shuf}
        scores[name] = out
        log(f"  {name}: {spec[name]['params_per_head']:,} params/head ({time.time()-t0:.0f}s)")
        np.savez_compressed(out_scores, clip_cluster=cc, y=z["Y"], valid=V.astype(np.uint8),
                            ego=E, cache_tag=z["cache_tag"], folds=folds,
                            situations=np.array(sits), **scores)

    # (name, kind, pca_r, param budget, epochs)
    ladder = [("ridge_pca16_w8", "ridge", 16, None, 0),
              ("ridge_pca64_w8", "ridge", 64, None, 0),
              ("ridge_pca256_w8", "ridge", 256, None, 0),
              ("ridge_raw2048_w1", "ridge", 0, None, 0)]
    for b in TF_BUDGETS:
        ladder.append((f"tf_pca16_d{width_for_param_budget(b, 16, WIN, 3)}", "tf", 16, b, EPOCHS))
    ladder.append(("tf_pca64_d128", "tf", 64, 417_028, EPOCHS))     # representation axis
    # OPTIMISATION-BUDGET CONTROL. B4 exists to separate an "optimisation shortcut"
    # from a "representation limit". Without this, "the big head loses" is
    # indistinguishable from "the big head is undertrained at a fixed 8 epochs";
    # the same capacity at 3x the budget answers that directly.
    ladder.append(("tf_pca16_d128_ep24", "tf", 16, 417_028, 24))
    ladder.append(("tf_pca16_d296_ep24", "tf", 16, 2_170_000, 24))
    order = [n for n, _k, _r, _b, _e in ladder]

    # ---- STAGE 1: THE PERMUTED-FEATURE NULL, FITTED AND SCORED FIRST --------
    log("STAGE 1 - NEG_FEATURE (image features permuted ACROSS clips), fitted FIRST")
    for name, kind, r, b, e in ladder:
        run_rung("NEG_FEAT__" + name, kind, r, budget=b, shuf=True, epochs=(e or EPOCHS))
    null_check = {}
    for i, s in enumerate(sits):
        m = V[:, i]
        yv = Yi[m, i]
        null_check[s] = {n[len("NEG_FEAT__"):]: round(ap_lift(yv, scores[n][m, i]), 4)
                         for n in scores if n.startswith("NEG_FEAT__")}
        log(f"  NULL {s} ap-lift: " + " ".join(f"{k}={v:.3f}" for k, v in null_check[s].items()))

    # ---- STAGE 2: the real ladder ------------------------------------------
    log("STAGE 2 - the real ladder")
    for name, kind, r, b, e in ladder:
        run_rung(name, kind, r, budget=b, shuf=False, epochs=(e or EPOCHS))

    # ---- STAGE 3: NEG_LABEL -------------------------------------------------
    log("STAGE 3 - NEG_LABEL (labels permuted across whole CLUSTERS)")
    neg_label = {}
    for i, s in enumerate(sits):
        m = V[:, i]
        y_perm = permute_labels_by_cluster(Yi[:, i], cc, seed=1)[m]
        neg_label[s] = {n: round(ap_lift(y_perm, scores[n][m, i]), 5)
                        for n in (FLOOR_RUNG, DEPLOYED_RUNG)}
        log(f"  NEG_LABEL {s}: {neg_label[s]}")

    # ---- STAGE 4: the capacity curve, with intervals ------------------------
    log("STAGE 4 - AP / AP-lift with the episode-cluster bootstrap")
    res = {"_what": "BACKLOG B4 - matched-capacity VISION-ONLY situation heads",
           "_inference_inputs": "CAMERA ONLY (frozen v1 encoder latents). No ego channel.",
           "substrate": str(a.substrate),
           "substrate_meta": json.loads(Path(a.substrate).with_suffix(".meta.json").read_text()),
           "protocol": {"win": WIN, "epochs": EPOCHS, "sel_frac": SEL_FRAC,
                        "outer_folds": 2, "fold_unit": "clip cluster",
                        "selection": "held-out 20% of TRAIN clusters, mean AP over situations",
                        "pca": "fitted on the FIT rows of each fold only",
                        "estimator": "paired_ap_episode_cluster_bootstrap (taniteval draws)",
                        "n_boot": a.n_boot, "strata_n_boot": a.strata_n_boot,
                        "deployed_rung": DEPLOYED_RUNG, "floor_rung": FLOOR_RUNG},
           "rungs": spec,
           "controls": {"NEG_FEATURE_ap_lift": null_check, "NEG_LABEL_ap_lift": neg_label},
           "per_situation": {}}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    for i, s in enumerate(sits):
        m = V[:, i]
        yv = Yi[m, i]
        eid = cc[m]
        ones = np.ones(yv.size, bool)
        row = {"n_scorable": int(m.sum()), "n_pos": int(yv.sum()),
               "base_rate": round(float(yv.mean()), 6),
               "n_clusters": int(len(np.unique(eid))),
               "n_clusters_with_a_positive": int(len(np.unique(eid[yv > 0]))),
               "arms": {}, "paired": {}}
        for n in order:
            sc = scores[n][m, i].astype(np.float64)
            r_ = ap_episode_cluster_bootstrap(yv, sc, eid, n_boot=a.n_boot, lift=True)
            r_["ap"] = round(average_precision(yv, sc), 5)
            r_["params_per_head"] = spec[n]["params_per_head"]
            r_["n_nonfinite"] = int((~np.isfinite(sc)).sum())
            r_["op_5pct"] = precision_recall_at_budget(yv, sc, ones)
            r_["paired_vs_own_null"] = paired_ap_episode_cluster_bootstrap(
                yv, sc, scores["NEG_FEAT__" + n][m, i].astype(np.float64), eid,
                n_boot=a.n_boot, lift=True)
            row["arms"][n] = r_
            pn = r_["paired_vs_own_null"]
            log(f"  {s} {n:>19}: AP={r_['ap']:.5f} lift={r_['point']:.3f} "
                f"P@5%={r_['op_5pct']['precision']:.4f} R@5%={r_['op_5pct']['recall']:.4f} "
                f"vs-null {pn['delta']:+.3f} [{pn['lo']:+.3f},{pn['hi']:+.3f}]"
                f"{' SEP' if pn['separated'] else ''}")

        base = scores[DEPLOYED_RUNG][m, i].astype(np.float64)
        for n in order:
            if n == DEPLOYED_RUNG:
                continue
            row["paired"][f"{n} - {DEPLOYED_RUNG}"] = paired_ap_episode_cluster_bootstrap(
                yv, scores[n][m, i].astype(np.float64), base, eid, n_boot=a.n_boot, lift=True)
        peak = max(order, key=lambda n: row["arms"][n]["point"])
        row["peak_rung"] = peak
        if peak not in (DEPLOYED_RUNG, FLOOR_RUNG):
            row["paired"][f"PEAK {peak} - {FLOOR_RUNG}"] = paired_ap_episode_cluster_bootstrap(
                yv, scores[peak][m, i].astype(np.float64),
                scores[FLOOR_RUNG][m, i].astype(np.float64), eid, n_boot=a.n_boot, lift=True)
        res["per_situation"][s] = row
        log(f"  {s}: PEAK = {peak}")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---- STAGE 5: the FOUR FAMILIES, peak rung vs the deployed rung ---------
    log("STAGE 5 - the four families (peak rung vs deployed rung)")
    bundle = ScoreBundle(situations=np.array(sits), arms=order, y=z["Y"],
                         valid=V.astype(np.uint8), clip_cluster=cc,
                         scores={k: v for k, v in scores.items() if not k.startswith("NEG_")},
                         ego=E, source=str(a.substrate))
    res["four_families"] = {}
    for j, s in enumerate(sits):
        peak = res["per_situation"][s]["peak_rung"]
        res["four_families"][s] = four_family_report(
            bundle, s, fused=scores[peak][:, j].astype(np.float64),
            baseline=scores[DEPLOYED_RUNG][:, j].astype(np.float64),
            baseline_name=DEPLOYED_RUNG, fused_name=f"PEAK::{peak}",
            n_boot=a.n_boot, strata_n_boot=a.strata_n_boot)
        log(f"  families {s}: peak={peak} done")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---- STAGE 6: SELF-CONSISTENCY control ---------------------------------
    log("STAGE 6 - component-vs-family self-consistency")
    sc_ctl = {}
    for i, s in enumerate(sits):
        peak = res["per_situation"][s]["peak_rung"]
        fam = res["four_families"][s]["families"]["TACTICAL"]
        # the family drops rows where EITHER arm is non-finite; recompute on the
        # family's OWN row set, or the control tests two different denominators
        # against each other and can only ever fail.
        a_ = scores[peak][:, i].astype(np.float64)
        b_ = scores[DEPLOYED_RUNG][:, i].astype(np.float64)
        m = V[:, i] & np.isfinite(a_) & np.isfinite(b_)
        yv = Yi[m, i]
        indep = {DEPLOYED_RUNG: round(ap_lift(yv, b_[m]), 5),
                 f"PEAK::{peak}": round(ap_lift(yv, a_[m]), 5)}
        # nan == nan must count as AGREEMENT: a situation with no positive in the
        # family's row set gives nan on both sides, and reporting that as a
        # control FAILURE would be a false alarm about the instrument.
        def _same(u, v):
            return (np.isnan(u) and np.isnan(v)) or abs(u - v) < 1e-9
        agree = all(_same(float(indep[k]), float(fam["ap_lift"][k])) for k in indep)
        sc_ctl[s] = {"n_rows": int(m.sum()), "family_ap_lift": fam["ap_lift"],
                     "recomputed_independently": indep, "identical": bool(agree)}
        log(f"  {s}: family == components -> {agree}")
    res["controls"]["SELF_CONSISTENCY_component_vs_family"] = sc_ctl

    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {a.out} and {out_scores}")


if __name__ == "__main__":
    main()
