"""⭐ THE DISCRIMINATOR BETWEEN C106's TWO POSSIBLE MECHANISMS —
"SUBTRACTED" vs "SWAMPED".

C106 concludes the S-W objective is **subtracting** linearly readable geometry.
MEASURED here first (`raw/rank.json`, `raw/layerscale.json`): the trained
encoder's token field is **rank-collapsed** — 97.6 % of all token-channel
variance sits in ONE direction, effective rank 1.22 of 768, against ~68 for the
random init. That admits a SECOND reading C106 never tested:

  (a) SUBTRACTED — the geometry is gone from the token span. Nothing recovers it.
  (b) SWAMPED    — the geometry is still in the span but pushed into
                   LOW-VARIANCE directions, so a ridge on a random projection,
                   which is dominated by the top direction, cannot reach it.

⛔ THE REMEDIES ARE DIFFERENT, so the distinction is not academic. (a) indicts
the objective and argues for distillation from step 0. (b) indicts the
CONDITIONING of the representation and is fixable at the readout — far cheaper.

THE TEST: re-fit the identical readout on a PCA-WHITENED design. Whitening
rescales every retained principal direction to unit variance, so a signal living
at 0.1 % of the variance gets the same weight as the dominant one. If ours rises
toward randenc under whitening, the geometry was SWAMPED. If it does not move,
it was SUBTRACTED and C106's phrasing stands.

⚠️ WHITENING IS NOT FREE AND THAT IS WHY BOTH ARMS GET IT. It amplifies noise
directions and can only be read as a CONTRAST between arms given the same
treatment, never as an absolute. The matched-random null gets it too.

⛔ NOTHING IS RE-IMPLEMENTED: `build_features` and `fit_one` are er10's.
PARITY: SELECTS NOTHING. TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[6]
_INC = _REPO / "TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
for _p in (_REPO / "taniteval", _REPO / "stack",
           _INC / "2026-08-17-probe-positive-control/code",
           _INC / "2026-08-17-slot-probe-parity/code",
           _INC / "2026-08-17-latent-linear-ladder/code",
           _INC / "2026-08-18-pooling-ladder-ER10/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import er10_pool_ladder as ER                                    # noqa: E402
import ll1_ladder as LL                                          # noqa: E402

ALPHAS = [1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--join-file", required=True)
    ap.add_argument("--targets", nargs="+", default=["ego_v0", "lead_gap"])
    ap.add_argument("--whiten-k", type=int, nargs="+", default=[0, 16, 64, 256])
    ap.add_argument("--proj-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ridge-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available())
                       else "cpu")
    t0 = time.time()
    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    th, tw = int(meta["token_grid"][0]), int(meta["token_grid"][1])
    d_model = int(rows[0]["tokens"].shape[-1])
    decl = json.loads(Path(a.split_json).read_text("utf-8"))
    ev_c, tr_c = set(decl["eval_clips"]), set(decl["train_clips"])
    idx_tr = [i for i, r in enumerate(rows) if r["clip_id"] in tr_c]
    idx_ev = [i for i, r in enumerate(rows) if r["clip_id"] in ev_c]
    sub = [rows[i] for i in idx_tr + idx_ev]
    pos_tr = np.arange(len(idx_tr))
    pos_ev = np.arange(len(idx_tr), len(sub))
    ctr_all = np.array([sub[i]["clip_id"] for i in pos_tr])
    cev_all = np.array([sub[i]["clip_id"] for i in pos_ev])
    eev_all = np.array([sub[i]["episode_uid"] for i in pos_ev])
    v0_all = np.array([float(r["v0"]) for r in sub])
    tvals = {t: tuple(np.array(x) for x in
                      zip(*[LL.target_of(r, {}, t, 0) for r in sub]))
             for t in a.targets}
    feats, n_units = ER.build_features(sub, "p40", a.proj_seeds, th, tw,
                                       d_model, dev, None, 0.0, None, None)
    del blob, rows
    print(f"[wh] {a.tag}: {len(sub)} rows, {n_units} units ({time.time()-t0:.0f} s)",
          flush=True)

    out = {"_evidence_class": "MEASURED (ours; PCA-WHITENED re-fit of the SAME "
                              "readout on the SAME banked windows)",
           "eval_tier": "T0-DIAGNOSTIC",
           "question": "SUBTRACTED (whitening does not help) vs SWAMPED "
                       "(whitening recovers the signal)",
           "tag": a.tag, "cache": str(a.cache),
           "arm": "p40 (the DEPLOYED AvgPool2d((4,10)))",
           "whiten_k": a.whiten_k, "alphas": ALPHAS,
           "proj_seeds": a.proj_seeds, "ridge_seeds": a.ridge_seeds,
           "ridge_intercept_col": -1,
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
           "forbidden": "overlapping_holdout_se",
           "solve_source": "er10_pool_ladder.fit_one / build_features "
                           "(IMPORTED from the producer)",
           "parity": "SELECTS NOTHING — banked window set read verbatim",
           "note": "whitening amplifies noise directions; it is readable ONLY "
                   "as a CONTRAST between arms given the identical treatment",
           "targets": {}}

    for tname in a.targets:
        y, ok = tvals[tname]
        mtr, mev = ok[pos_tr], ok[pos_ev]
        ytr, yev = y[pos_tr][mtr].astype(np.float64), y[pos_ev][mev].astype(np.float64)
        ctr, cev = ctr_all[mtr], cev_all[mev]
        eev = eev_all[mev]
        v0ev = None if tname == "ego_v0" else v0_all[pos_ev][mev]
        byk = {}
        for K in a.whiten_k:
            cells = {}
            for s in a.proj_seeds:
                X = feats[s]
                Xtr, Xev = X[pos_tr][mtr], X[pos_ev][mev]
                mu, sd = Xtr.mean(0), Xtr.std(0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                Ztr0, Zev0 = (Xtr - mu) / sd, (Xev - mu) / sd
                if K and K > 0:
                    # PCA on the PROBE-TRAIN rows only — never on eval.
                    U, S, Vt = np.linalg.svd(Ztr0, full_matrices=False)
                    k = min(int(K), (S > 1e-9).sum())
                    W = Vt[:k].T / np.maximum(S[:k], 1e-9)      # unit-variance
                    Ztr0, Zev0 = Ztr0 @ W, Zev0 @ W
                Ztr = np.concatenate([Ztr0, np.ones((Ztr0.shape[0], 1))], 1)
                Zev = np.concatenate([Zev0, np.ones((Zev0.shape[0], 1))], 1)
                for rs in a.ridge_seeds:
                    rec, _ = ER.fit_one(Ztr, ytr, ctr, Zev, yev, eev, cev,
                                        ALPHAS, 0.25, rs, a.n_boot, v0ev,
                                        0.10, -1)
                    cells[f"p{s}|r{rs}"] = rec
            r2s = [c["r2_ceiling"] for c in cells.values()]
            byk[str(K)] = {
                "n_components": (int(K) if K else int(next(iter(feats.values())).shape[1])),
                "r2_ceiling_mean": round(float(np.mean(r2s)), 5),
                "r2_ceiling_sd": round(float(np.std(r2s)), 5),
                "r2_ceiling_min": round(float(np.min(r2s)), 5),
                "r2_ceiling_max": round(float(np.max(r2s)), 5),
                "K1_PASS_count": int(sum(bool(c["K1_PASSES"]) for c in cells.values())),
                "n_cells": len(cells),
                "pred_sd_over_gt_sd_mean": round(float(np.mean(
                    [c["pred_sd_over_gt_sd"] for c in cells.values()])), 4),
                "per_cell": cells}
            print("  %-12s K=%-5s r2c=%.5f±%.5f  K1pass=%d/%d  psd/gsd=%.3f"
                  % (tname, K or "none", byk[str(K)]["r2_ceiling_mean"],
                     byk[str(K)]["r2_ceiling_sd"], byk[str(K)]["K1_PASS_count"],
                     byk[str(K)]["n_cells"],
                     byk[str(K)]["pred_sd_over_gt_sd_mean"]), flush=True)
        out["targets"][tname] = {
            "unit": LL.UNITS[tname], "n_eval": int(mev.sum()),
            "n_eval_clusters": int(len(np.unique(eev))),
            "by_whiten_k": byk}
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), "utf-8")
    print(f"[wh] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
