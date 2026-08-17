"""PC6 — ⭐ THE DISAMBIGUATION: a RIDGE REGRESSION straight from the memory
tensor to the GT lead gap, on the identical windows, split and estimator.

⛔ WHY THIS IS NECESSARY AND WHAT IT IS NOT.
PC1 showed the GT-oracle memory FAILS the slot probe (10.18 m vs a 5.13 m
constant). Two readings remain:
  (a) the ORACLE ENCODING does not actually contain the answer -> my control is
      the broken thing and nothing follows about F-18;
  (b) the encoding contains it and the SLOT-PROBE APPARATUS cannot extract it.
A linear readout settles (a) vs (b) in seconds: if a ridge on the same tensor
recovers the lead gap, the information is demonstrably there and linearly
available, and (b) is the answer.

⛔ THIS IS A DIFFERENT INSTRUMENT AND MAY NEVER BE COMPARED TO THE F-18 SLOT
NUMBERS AS THOUGH IT WERE THE SAME PROBE. It is reported for exactly one job:
to say whether the information is present. It is run on the ORACLE, on the REAL
v6 arms and on the RANDOM-LATENT NULL, so it carries its own floor.

METHOD — deliberately the dullest possible.
  * features: the memory tensor flattened ([16 x 128] = 2048), z-scored with
    the PROBE-TRAIN mean/sd, plus a bias;
  * target: the GT lead gap in metres, on GT-lead windows only;
  * ridge alpha chosen on an EPISODE-DISJOINT inner split of the PROBE-TRAIN
    clips (never the eval clips), by mean abs error;
  * scored on the SAME eval windows the slot probe's paired set uses, with the
    SAME ``paired_episode_cluster_bootstrap`` against C-CONST and C-EPMEAN.

⚠️ A ridge cannot ABSTAIN, so its window set is every GT-lead eval window. That
is a superset of the slot probe's paired set (which drops windows where some
arm emitted no in-corridor slot); both n are printed and the slot probe's own
subset is scored as well, so the two are never silently compared on different
windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp2_probe as SP                                          # noqa: E402
from taniteval.ci import (episode_cluster_bootstrap,             # noqa: E402
                          paired_episode_cluster_bootstrap)


def ridge_fit(X, y, alpha):
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    ap.add_argument("--inner-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    decl = json.loads(Path(a.split_json).read_text("utf-8"))
    ev_c, tr_c = set(decl["eval_clips"]), set(decl["train_clips"])

    def pack(sel_clips):
        idx = [i for i, r in enumerate(rows) if r["clip_id"] in sel_clips]
        g = np.array([SP.gt_lead_gap(rows[i]["agents"])
                      if SP.gt_lead_gap(rows[i]["agents"]) is not None
                      else np.nan for i in idx])
        keep = ~np.isnan(g)
        idx = [i for i, k in zip(idx, keep) if k]
        X = np.stack([rows[i]["cells"].numpy().reshape(-1).astype(np.float64)
                      for i in idx])
        y = g[keep]
        eid = np.array([rows[i]["episode_uid"] for i in idx])
        cid = np.array([rows[i]["clip_id"] for i in idx])
        return X, y, eid, cid

    Xtr, ytr, _etr, ctr = pack(tr_c)
    Xev, yev, eev, _cev = pack(ev_c)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0

    def prep(X):
        return np.concatenate([(X - mu) / sd, np.ones((X.shape[0], 1))], 1)

    Ztr, Zev = prep(Xtr), prep(Xev)

    # --- alpha on an EPISODE-DISJOINT inner split of the PROBE-TRAIN clips ---
    rng = np.random.default_rng(a.seed)
    clips = np.array(sorted(set(ctr.tolist())))
    rng.shuffle(clips)
    n_in = max(1, int(round(len(clips) * a.inner_frac)))
    inner = set(clips[:n_in].tolist())
    m_in = np.array([c in inner for c in ctr])
    best, best_mae = None, np.inf
    tried = {}
    for al in a.alphas:
        w = ridge_fit(Ztr[~m_in], ytr[~m_in], al)
        mae = float(np.abs(Ztr[m_in] @ w - ytr[m_in]).mean())
        tried[f"{al:g}"] = round(mae, 4)
        if mae < best_mae:
            best, best_mae = al, mae
    w = ridge_fit(Ztr, ytr, best)
    pred = Zev @ w

    const_m = float(np.median(ytr))
    ep_of = np.array([rows[i]["clip_id"] for i, r in enumerate(rows)
                      if r["clip_id"] in ev_c
                      and SP.gt_lead_gap(r["agents"]) is not None])
    epmean = np.full(len(yev), const_m, dtype=np.float64)
    for c in np.unique(ep_of):
        pos = np.nonzero(ep_of == c)[0]
        tot = float(np.sum(yev[pos]))
        for k in pos:
            epmean[k] = ((tot - yev[k]) / (pos.size - 1)) if pos.size > 1 \
                else const_m

    e_arm = np.abs(pred - yev)
    e_con = np.abs(const_m - yev)
    e_ep = np.abs(epmean - yev)
    arm = episode_cluster_bootstrap(e_arm, eev, n_boot=a.n_boot)
    con = episode_cluster_bootstrap(e_con, eev, n_boot=a.n_boot)
    k1 = paired_episode_cluster_bootstrap(e_arm, e_con, eev, n_boot=a.n_boot)
    k5 = paired_episode_cluster_bootstrap(e_arm, e_ep, eev, n_boot=a.n_boot)
    out = {"_evidence_class": "MEASURED (ours; ridge readout — a DIFFERENT "
                              "instrument from the F-18 slot probe and never "
                              "to be quoted as one)",
           "eval_tier": "T0-DIAGNOSTIC",
           "arm": a.label, "run_stamp": meta.get("run_stamp"),
           "memory_shape": list(rows[0]["cells"].shape),
           "n_features": int(Ztr.shape[1]),
           "n_train_windows": int(Ztr.shape[0]),
           "n_eval_windows": int(Zev.shape[0]),
           "n_eval_clusters": int(len(np.unique(eev))),
           "alpha_chosen": best, "alpha_inner_mae": tried,
           "inner_split_clips": n_in, "estimator":
               "taniteval.ci.paired_episode_cluster_bootstrap",
           "forbidden": "overlapping_holdout_se",
           "ridge_err_m": round(float(e_arm.mean()), 4),
           "ridge_ci": [arm["lo"], arm["hi"]],
           "ridge_median_m": round(float(np.median(e_arm)), 4),
           "c_const_m_value": round(const_m, 4),
           "c_const_err_m": round(float(e_con.mean()), 4),
           "c_const_ci": [con["lo"], con["hi"]],
           "c_epmean_err_m": round(float(e_ep.mean()), 4),
           "K1_delta": k1["delta"], "K1_lo": k1["lo"], "K1_hi": k1["hi"],
           "K1_separated": k1["separated"],
           "K1_PASSES": bool(k1["separated"] and k1["delta"] < 0),
           "K5_delta": k5["delta"], "K5_lo": k5["lo"], "K5_hi": k5["hi"],
           "K5_separated": k5["separated"],
           "K5_PASSES": bool(k5["separated"] and k5["delta"] < 0),
           "corr_pred_gt": round(float(np.corrcoef(pred, yev)[0, 1]), 4),
           "pred_sd_m": round(float(pred.std()), 3),
           "gt_sd_m": round(float(yev.std()), 3)}
    Path(a.out).write_text(json.dumps(out, indent=1), "utf-8")
    print("  %-34s ridge=%7.3f [%.3f,%.3f]  const=%7.3f  K1=%+7.3f "
          "[%+.3f,%+.3f] %-8s K5=%+7.3f %-8s r=%+.3f  (alpha %g, n_ev %d/%d cl)"
          % (a.label, out["ridge_err_m"], arm["lo"], arm["hi"],
             out["c_const_err_m"], out["K1_delta"], out["K1_lo"], out["K1_hi"],
             "K1 PASS" if out["K1_PASSES"] else "K1 fail", out["K5_delta"],
             "K5 PASS" if out["K5_PASSES"] else "K5 fail", out["corr_pred_gt"],
             best, out["n_eval_windows"], out["n_eval_clusters"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
