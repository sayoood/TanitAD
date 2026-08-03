"""P8 — scenario-classifier optimisation: measure the two defects, fix them, and
report before/after with the binding estimator and two negative controls.

SUBSTRATE (0 GPU-h of pod time; dev-box RTX 4060 or CPU)
    …/2026-07-26-situation-classifier/artifacts/heldout_frames.npz
    308,973 held-out frames, 1,610 clip clusters, disjoint from the clips the
    banked heads were trained on (`train_summary.json`: 147,022 train windows).

PROTOCOL
  * Every arm is scored on the SAME rows — those with a full 50-frame causal
    history inside their clip — so the paired estimator is valid.
  * Everything I fit is fitted OUT-OF-FIT on a 2-fold split over whole CLUSTERS.
  * The architecture is held FIXED (`tanitad.eval.sitclf.CausalSitHead`, the
    deployed `sc_train.SitHead` with the window exposed) so the WIN sweep is
    attributable to the window and nothing else.

Usage:  python run_sitclf_opt.py --out results_sitclf_opt.json [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.ap_ci import (                                  # noqa: E402
    ap_episode_cluster_bootstrap, average_precision,
    paired_ap_episode_cluster_bootstrap)
from tanitad.eval.sitclf import (                                 # noqa: E402
    causal_window, clip_runs, cluster_folds, late_fuse_scores,
    predict_sit_head, train_sit_head)

NPZ = (REPO / "TanitAD Research Hub/Architecture & Inference/Implementation"
       / "incoming/2026-07-26-situation-classifier/artifacts/heldout_frames.npz")
WINS = (8, 25, 50)          # 8 == sc_train.py:37's pre-registered constant
EPOCHS, POS_WEIGHT, DMODEL = 8, 20.0, 128     # sc_train.py CONFIGS[0] + run_fold
N_BOOT = 2000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit_2fold(X, Y, V, folds, win, in_dim, device, seed=0, tag=""):
    """Out-of-fit sigmoid scores [N, 3] from the fixed architecture at `win`."""
    out = np.zeros_like(Y, dtype=np.float32)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        log(f"    {tag}win={win} fold{f}: train {tr.sum():,} -> score {te.sum():,}")
        m = train_sit_head(X[tr], Y[tr], V[tr], win=win, in_dim=in_dim,
                           epochs=EPOCHS, pos_weight=POS_WEIGHT, d=DMODEL,
                           seed=seed, device=device, log=None)
        out[te] = predict_sit_head(m, X[te], in_dim=in_dim, device=device)
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default="results_sitclf_opt.json")
    ap_.add_argument("--device", default=None)
    ap_.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap_.parse_args()

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}  npz={NPZ.name}")

    z = np.load(NPZ, allow_pickle=True)
    sits = [str(s) for s in z["situations"]]
    y, valid, cc = z["y"], z["valid"], z["clip_cluster"]
    E = z["ego"].astype(np.float32)                 # [v, alon_pre, omega_pre]
    st, en = clip_runs(cc)
    folds = cluster_folds(cc, 2, seed=0)

    # ---- common row set: full history at the LONGEST window, for all arms ----
    feats, oks = {}, {}
    for w in WINS:
        feats[w], oks[w] = causal_window(E, st, en, w)
    common = np.logical_and.reduce([oks[w] for w in WINS])
    log(f"common scorable rows {common.sum():,} / {len(cc):,} "
        f"({len(np.unique(cc[common]))} clusters)")

    Y = y.astype(np.float32)
    # validity mask fed to the loss = per-situation validity AND full history
    V = (valid > 0).astype(np.float32) * common[:, None].astype(np.float32)

    # ---- fit the sweep ----
    sweep = {}
    for w in WINS:
        t0 = time.time()
        sweep[w] = fit_2fold(feats[w], Y, V, folds, w, 3, device)
        log(f"  win={w} done in {time.time()-t0:.0f}s")

    # ---- NEGATIVE CONTROL 2: same fit, ego features permuted ACROSS clips ----
    # (labels untouched). If this scores above chance the protocol manufactures
    # signal and the whole sweep is void.
    rng = np.random.default_rng(0)
    perm_clip = rng.permutation(len(st))
    shuf_rows = np.arange(len(cc))
    for i, (s0, e0) in enumerate(zip(st, en)):
        j = perm_clip[i]
        n = min(e0 - s0, en[j] - st[j])
        shuf_rows[s0:s0 + n] = st[j] + np.arange(n)
        if e0 - s0 > n:
            shuf_rows[s0 + n:e0] = st[j] + n - 1
    Wbest = WINS[-1]
    Xshuf, _ = causal_window(E[shuf_rows], st, en, Wbest)
    log("  NEGATIVE CONTROL: ego permuted across clips")
    neg_feat = fit_2fold(Xshuf, Y, V, folds, Wbest, 3, device, tag="NEG-")

    # ---- BANK THE RAW SCORES FIRST -------------------------------------- #
    # The bootstrap below is the long pole. Writing the out-of-fit scores now
    # means a killed run still leaves a re-analysable artifact instead of
    # nothing — and any later re-scoring needs no GPU at all.
    np.savez_compressed(
        Path(a.out).with_suffix(".scores.npz"),
        clip_cluster=cc, y=y, valid=valid, common=common, folds=folds,
        situations=np.array(sits), neg_ego_permuted=neg_feat,
        **{f"ours_ego_win{w}": sweep[w] for w in WINS})
    log(f"banked out-of-fit scores -> {Path(a.out).with_suffix('.scores.npz')}")

    # ---- score ----
    res = {"meta": {
        "substrate": str(NPZ), "n_rows_total": int(len(cc)),
        "n_rows_common": int(common.sum()),
        "n_clusters": int(len(np.unique(cc[common]))),
        "wins": list(WINS), "epochs": EPOCHS, "pos_weight": POS_WEIGHT,
        "d_model": DMODEL, "n_boot": a.n_boot, "device": device,
        "architecture": "tanitad.eval.sitclf.CausalSitHead (== sc_train.SitHead)",
        "estimator": "paired_ap_episode_cluster_bootstrap over clip clusters",
        "note": ("banked arms were trained on a DISJOINT clip set (147,022 "
                 "windows); arms fitted here use ~1/2 of the 1,610 held-out "
                 "clusters per fold. Rows are identical across all arms.")}}

    for i, sit in enumerate(sits):
        m = common & (valid[:, i] > 0)
        yv = y[:, i].astype(np.int64)[m]
        eid = cc[m]
        arms = {
            "banked_head_ego": z["head_ego"][m, i].astype(np.float64),
            "banked_head_img": z["head_img"][m, i].astype(np.float64),
            "banked_head_img_ego_DEPLOYED": z["head_img_ego"][m, i].astype(np.float64),
            "banked_head_img_shuf_NULL": z["head_img_shuf"][m, i].astype(np.float64),
            "banked_heur_kin": z["heur_kin"][m, i].astype(np.float64),
        }
        for w in WINS:
            arms[f"ours_ego_win{w}"] = sweep[w][m, i].astype(np.float64)
        arms["NEG_ego_permuted_across_clips"] = neg_feat[m, i].astype(np.float64)

        # late fusion: our best ego head + the banked image head, out-of-fit
        best_w = WINS[-1]
        fuse_cols = np.stack([z["head_img"][:, i].astype(np.float64),
                              sweep[best_w][:, i].astype(np.float64)], 1)
        fused = late_fuse_scores(fuse_cols, y[:, i], m, folds)
        arms[f"ours_LATEFUSE(img, ego_win{best_w})"] = fused[m]
        # negative control 3: fuse the SHUFFLED image arm instead
        fuse_neg = np.stack([z["head_img_shuf"][:, i].astype(np.float64),
                             sweep[best_w][:, i].astype(np.float64)], 1)
        arms["NEG_LATEFUSE(img_SHUF, ego)"] = late_fuse_scores(
            fuse_neg, y[:, i], m, folds)[m]

        # NEGATIVE CONTROL 1: permute the LABEL across clusters
        cl = np.unique(eid)
        rmap = {c: p for c, p in zip(cl, np.random.default_rng(1).permutation(cl))}
        ord_by = np.argsort(np.array([rmap[c] for c in eid], dtype=eid.dtype),
                            kind="mergesort")
        y_perm = yv[ord_by]

        row = {"n_scorable": int(m.sum()), "n_pos": int(yv.sum()),
               "base_rate": float(yv.mean()), "n_clusters": int(len(cl)),
               "arms": {}, "paired": {}}
        for name, s in arms.items():
            r = ap_episode_cluster_bootstrap(yv, s, eid, n_boot=a.n_boot, lift=True)
            r["ap"] = round(average_precision(yv, s), 5)
            row["arms"][name] = r
        row["arms"]["NEG_label_permuted_across_clusters"] = {
            "ap": round(average_precision(y_perm, arms["banked_head_ego"]), 5),
            "note": "labels shuffled between clusters; must land at the base rate"}

        base = "banked_head_img_ego_DEPLOYED"
        for name in (f"ours_ego_win{WINS[-1]}", f"ours_LATEFUSE(img, ego_win{WINS[-1]})",
                     "banked_head_ego"):
            row["paired"][f"{name} - {base}"] = paired_ap_episode_cluster_bootstrap(
                yv, arms[name], arms[base], eid, n_boot=a.n_boot, lift=True)
        for name in (f"ours_ego_win{WINS[-1]}", f"ours_LATEFUSE(img, ego_win{WINS[-1]})"):
            row["paired"][f"{name} - banked_head_ego"] = (
                paired_ap_episode_cluster_bootstrap(
                    yv, arms[name], arms["banked_head_ego"], eid,
                    n_boot=a.n_boot, lift=True))
        row["paired"][f"ours_ego_win{WINS[-1]} - ours_ego_win8"] = (
            paired_ap_episode_cluster_bootstrap(
                yv, arms[f"ours_ego_win{WINS[-1]}"], arms["ours_ego_win8"], eid,
                n_boot=a.n_boot, lift=True))
        row["paired"][f"ours_ego_win{WINS[-1]} - NEG_ego_permuted"] = (
            paired_ap_episode_cluster_bootstrap(
                yv, arms[f"ours_ego_win{WINS[-1]}"],
                arms["NEG_ego_permuted_across_clips"], eid, n_boot=a.n_boot,
                lift=True))
        res[sit] = row
        log(f"  {sit}: " + "  ".join(
            f"{k.replace('banked_','b.').replace('ours_','o.')}={v['ap']:.4f}"
            for k, v in row["arms"].items() if "ap" in v))

    Path(a.out).write_text(json.dumps(res, indent=1))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
