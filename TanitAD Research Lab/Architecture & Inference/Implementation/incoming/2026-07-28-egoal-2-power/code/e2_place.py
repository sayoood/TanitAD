#!/usr/bin/env python3
"""E-GOAL-2 -- the E-GOAL-1 placement, re-run at n = 600 episodes.

⛔ ONLY `n` CHANGES. The residual pools, the injection, both resamplers, the
seeds, the number of draws and the estimator are all taken from E-GOAL-1 and
are NOT re-derived here:

  * `realise` / `goal_reference` / `pick_nearest_to` / `ade` are IMPORTED from
    `…/2026-07-27-egoal-1-lead-vehicle/code/eg_place.py`, not re-implemented.
    If a single line of the pick rule differed, the n = 40 fidelity gate (F-B)
    below could not reproduce EGOAL_1.md §0.3 to four decimals.
  * the along-track residual pools come verbatim from
    `…/2026-07-27-egoal-1-lead-vehicle/raw/eg_oof_pred_gbm.npz`. Nothing is
    re-fit to chase a bar (`GATE_PROTOCOL §0.3`, forking paths).
  * estimator: paired episode-cluster bootstrap, `taniteval/ci.py`, B = 2000,
    unit = the val episode. `overlapping_holdout_se` is NEVER called.

⚠️ THE ONE THING THAT CANNOT BE HELD IDENTICAL, declared in PRE_REGISTRATION §4:
the cross-track background. E-GOAL-1 holds it at the parent's `H_ridge_all_raw`,
whose features include v4-latent blocks from `eh2_cache.pt` that exist ONLY on
the canonical 881 windows. At n = 600 the cross-track is a ridge on the
FAN-ONLY blocks (`F_ego` ⧺ `F_ans`) with the identical ridge machinery
(`gi_tautology.fit_ridge`, imported). `--cross reduced` runs that head at ANY n,
which is what makes the n = 40 bridge control (F-B2) possible: the same reduced
cross-track is pushed through the n = 40 placement and the published recoveries
must survive.

Run:
    python e2_place.py --fan <fan.pt> --cross parent   --tag n40_parent
    python e2_place.py --fan <fan.pt> --cross reduced  --tag n40_reduced
    python e2_place.py --fan <fan600.pt> --cross reduced --tag n600
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STREAM = HERE.parent
ARCH = STREAM.parent
REPO = STREAM.parents[4]
EG1 = ARCH / "2026-07-27-egoal-1-lead-vehicle"
GI = ARCH / "2026-07-27-goal-input"

sys.path.insert(0, str(EG1 / "code"))
sys.path.insert(0, str(GI / "code"))
sys.path.insert(0, str(REPO / "taniteval"))

# --- E-GOAL-1's construction, IMPORTED VERBATIM ---------------------------- #
from eg_place import (A0_COMMITTED, FRAC, GOAL_COMMITTED,  # noqa: E402
                      HEAD_COMMITTED, HEADROOM, N_SEEDS, ade, goal_reference,
                      pick_nearest_to, realise)
from eg_common import ci_paired, ci_single, r4, sep  # noqa: E402

#: E-GOAL-1 §0.3, the numbers the n = 40 fidelity gate must reproduce
EG1_RECOVERY = {"E0_v0|iid": -0.661, "CV|iid": -0.577, "E1_ego|iid": 0.236,
                "E1_L|iid": 0.259, "E1_L_X|iid": 0.254, "E1_L_X_D|iid": 0.257,
                "P_ORACLE|iid": 0.598, "E1_ego|by_speed": 0.203,
                "E1_L|by_speed": 0.229}
EG1_PARENT_RESAMP = {"iid": -0.331, "by_speed": -0.349}


# =========================================================================== #
# the cross-track background                                                  #
# =========================================================================== #
def fan_blocks(d) -> dict:
    """`F_ego` and `F_ans` exactly as `gi_tautology.build_features` builds them.

    These are the two blocks that depend ONLY on the fan dump, so they exist at
    any n. `F_lat` / `F_conf_raw` come from `eh2_cache.pt` (v4 latents on the
    canonical 881 windows only) and are the reason a reduced head is needed.
    """
    fan = d["fan"].numpy().astype(np.float64)
    cv = d["cv"].numpy().astype(np.float64)
    v0 = d["v0"].numpy().astype(np.float64)
    logits = d["logits"].numpy().astype(np.float64)
    sel = d["sel"].numpy()
    W = len(fan)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    fan_end = fan[:, :, -1, :]
    F_ego = np.concatenate([v0[:, None], cv.reshape(W, 8)], 1)
    F_ans = np.concatenate([fan[np.arange(W), sel][:, -1, :],
                            (p[:, :, None] * fan_end).sum(1)], 1)
    return {"F_ego": F_ego, "F_ans": F_ans,
            "sel_end": fan[np.arange(W), sel][:, -1, :],
            "cv_end": cv[:, -1, :]}


def reduced_head(d, g_end, eid):
    """Ridge on `F_ego` ⧺ `F_ans`, OUT-OF-FOLD, episode-disjoint.

    The ridge machinery (`ALPHAS`, inner `GroupKFold` over TRAIN-fold episodes,
    `StandardScaler`) is `gi_tautology`'s, IMPORTED -- so the only difference
    from the parent's `H_ridge_all_raw` is the two dropped eh2 blocks.
    """
    from sklearn.preprocessing import StandardScaler

    from gi_common import episode_folds
    from gi_tautology import fit_ridge

    B = fan_blocks(d)
    X = np.concatenate([B["F_ego"], B["F_ans"]], 1)
    pred = np.zeros((len(g_end), 2))
    alphas = []
    for tr, te in episode_folds(eid):
        sc = StandardScaler().fit(X[tr])
        m, a = fit_ridge(sc.transform(X[tr]), g_end[tr], eid[tr])
        pred[te] = m.predict(sc.transform(X[te]))
        alphas.append(float(a))
    return pred, alphas, B


# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan", required=True)
    ap.add_argument("--cross", choices=["parent", "reduced", "sel",
                                        "parent_resampled"],
                    default="parent_resampled")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pools", default=str(EG1 / "raw" / "eg_oof_pred_gbm.npz"))
    ap.add_argument("--extra-pools", default=str(STREAM / "raw"
                                                 / "e2_extra_pools.npz"))
    ap.add_argument("--windows", default=str(EG1 / "raw" / "eg_windows.parquet"))
    ap.add_argument("--drop-eid", default="", help="comma-separated eids to DROP "
                                                   "(leak remediation)")
    ap.add_argument("--curve", action="store_true",
                    help="also run the family-matched requirement-curve sweep")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    t_start = time.time()

    d = torch.load(a.fan, map_location="cpu", weights_only=False)
    fan = d["fan"].numpy().astype(np.float64)
    gt = d["gt"].numpy().astype(np.float64)
    v0 = d["v0"].numpy().astype(np.float64)
    eid = np.asarray([str(x) for x in d["eid"]])
    sel = d["sel"].numpy()

    cvw = d["cv"].numpy().astype(np.float64)
    logits = d["logits"].numpy().astype(np.float64)
    if a.drop_eid:
        drop = set(a.drop_eid.split(","))
        keep = np.array([e not in drop for e in eid])
        print(f"[place] LEAK REMEDIATION: dropping {len(drop)} episodes "
              f"({(~keep).sum()} windows)", flush=True)
        fan, gt, v0, eid, sel, cvw, logits = (
            fan[keep], gt[keep], v0[keep], eid[keep], sel[keep], cvw[keep],
            logits[keep])
    #: the ONLY fields the cross-track head may see, re-packed after any drop
    dcut = {"fan": torch.as_tensor(fan), "cv": torch.as_tensor(cvw),
            "v0": torch.as_tensor(v0), "logits": torch.as_tensor(logits),
            "sel": torch.as_tensor(sel)}
    g_true = gt[:, -1, :]
    W, NEP = len(gt), len(np.unique(eid))
    print(f"[place] {a.tag}: {W} windows / {NEP} episode clusters", flush=True)

    a0 = ade(fan[np.arange(W), sel], gt)
    r_goal = realise(g_true, fan, gt)
    oracle = ade(fan[np.arange(W)[:, None],
                     np.argmin(np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1),
                               axis=1)[:, None]].squeeze(1), gt)
    headroom = float(a0.mean() - r_goal.mean())

    res = {"_stream": "2026-07-28-egoal-2-power", "_tag": a.tag,
           "_fan": str(a.fan), "_cross_mode": a.cross,
           "_estimator": ("paired_episode_cluster_bootstrap, taniteval/ci.py, "
                          "B=2000, unit=val episode. overlapping_holdout_se "
                          "NEVER called."),
           "_n_seeds": N_SEEDS,
           "deployment": {"n_windows": int(W), "n_episode_clusters": int(NEP),
                          "a0_as_trained": r4(a0.mean()),
                          "r_goal2s_true_goal": r4(r_goal.mean()),
                          "oracle_in_fan": r4(oracle.mean()),
                          "headroom": r4(headroom)},
           "arms": {}}

    # ---------------- F-B: the n = 40 fidelity gate -------------------------
    if W == 881:
        fid = {"a0": (r4(a0.mean()), A0_COMMITTED),
               "r_goal2s": (r4(r_goal.mean()), GOAL_COMMITTED),
               "oracle_in_fan": (r4(oracle.mean()), 0.1640),
               "headroom": (r4(headroom), HEADROOM)}
        ok = all(abs(m - c) < 5e-4 for m, c in fid.values())
        res["F_B_pipeline_fidelity"] = {"checks": {k: {"mine": m,
                                                       "committed": c}
                                                   for k, (m, c) in fid.items()},
                                        "passes": bool(ok)}
        print("[F-B]", json.dumps(res["F_B_pipeline_fidelity"]["checks"]),
              "->", ok, flush=True)
        if not ok:
            raise RuntimeError("F-B FAILED: the parent's pipeline does not "
                               "reproduce; no number may be quoted")

    # ---------------- the cross-track background ----------------------------
    if a.cross == "parent":
        P = np.load(GI / "raw" / "gi_head_preds.npz", allow_pickle=True)
        assert np.abs(P["gt_end"] - g_true).max() == 0.0, "gt_end mismatch"
        head = P["best"]
        res["cross_track"] = {"mode": "parent H_ridge_all_raw (eh2 features)",
                              "mae_cross_m": r4(np.abs(head[:, 1]
                                                       - g_true[:, 1]).mean()),
                              "mae_along_m": r4(np.abs(head[:, 0]
                                                       - g_true[:, 0]).mean())}
    elif a.cross == "parent_resampled":
        # ⭐⭐ THE CARRIER. E-GOAL-1's background is the PARENT head's learned
        # cross-track, which cannot be recomputed at n = 600 (its features are
        # v4 latents that exist only on the canonical 881 windows). What CAN be
        # carried across is its ERROR STRUCTURE: draw the cross-track residual
        # from the parent head's own 881 empirical cross residuals, with the
        # SAME resampling machinery the along axis already uses. The background
        # then has the parent's exact cross-track error distribution at any n.
        #
        # ⚠️ It is drawn ONCE PER SEED from an INDEPENDENT rng stream, so the
        # background is IDENTICAL ACROSS ARMS within a seed -- a background that
        # differed per arm would not be a background, it would be a treatment.
        P = np.load(GI / "raw" / "gi_head_preds.npz", allow_pickle=True)
        pc_resid = P["best"][:, 1] - P["gt_end"][:, 1]
        head = np.stack([np.zeros(W), np.zeros(W)], 1)     # unused; see below
        cross_pool = pc_resid
        res["cross_track"] = {
            "mode": ("PARENT-RESAMPLED: the parent head's own 881 cross-track "
                     "residuals, resampled i.i.d. per seed onto the true "
                     "cross-track. Carries E-GOAL-1's cross error STRUCTURE to "
                     "any n; identical across arms within a seed."),
            "parent_cross_pool_n": int(len(pc_resid)),
            "parent_cross_mae_m": r4(float(np.mean(np.abs(pc_resid)))),
            "parent_cross_rms_m": r4(float(np.sqrt(np.mean(pc_resid ** 2))))}
    elif a.cross == "sel":
        # ⭐ THE SELECTOR'S OWN 2 s ENDPOINT. Zero fitting, zero folds, zero
        # hyper-parameters -- so it is defined IDENTICALLY at n = 40 and
        # n = 600 and cannot drift between them. It is also the most literal
        # reading of "hold the other axis at its LEARNED value": this IS the
        # cross-track the deployed system already produces.
        B = fan_blocks(dcut)
        head = B["sel_end"]
        res["cross_track"] = {
            "mode": ("SELECTOR'S OWN endpoint (fan[sel, -1]). No fit, no fold, "
                     "no hyper-parameter -- identical construction at any n."),
            "mae_cross_m": r4(np.abs(head[:, 1] - g_true[:, 1]).mean()),
            "mae_along_m": r4(np.abs(head[:, 0] - g_true[:, 0]).mean()),
            "rms_along_m": r4(float(np.sqrt(np.mean(
                (head[:, 0] - g_true[:, 0]) ** 2)))),
            "rms_cross_m": r4(float(np.sqrt(np.mean(
                (head[:, 1] - g_true[:, 1]) ** 2))))}
    else:
        head, alphas, _B = reduced_head(dcut, g_true, eid)
        res["cross_track"] = {
            "mode": ("REDUCED ridge on F_ego + F_ans (fan-only blocks); the "
                     "parent's F_lat / F_conf_pca come from eh2_cache.pt, "
                     "which exists only on the canonical 881 windows"),
            "ridge_alphas": alphas,
            "mae_cross_m": r4(np.abs(head[:, 1] - g_true[:, 1]).mean()),
            "mae_along_m": r4(np.abs(head[:, 0] - g_true[:, 0]).mean()),
            "rms_along_m": r4(float(np.sqrt(np.mean(
                (head[:, 0] - g_true[:, 0]) ** 2)))),
            "rms_cross_m": r4(float(np.sqrt(np.mean(
                (head[:, 1] - g_true[:, 1]) ** 2))))}
    if a.cross == "parent_resampled":
        #: [N_SEEDS, W] -- one shared background per seed, identical for every arm
        cross_by_seed = np.stack(
            [g_true[:, 1] + cross_pool[np.random.default_rng(5000 + s)
                                       .integers(0, len(cross_pool), W)]
             for s in range(N_SEEDS)])
        head_cross = None
        res["cross_track"]["realised_cross_mae_m"] = r4(
            float(np.mean(np.abs(cross_by_seed - g_true[:, 1][None]))))
    else:
        cross_by_seed = None
        head_cross = head[:, 1]
    print("[place] cross-track:", json.dumps(
        {k: v for k, v in res["cross_track"].items()
         if k != "ridge_alphas"})[:300], flush=True)

    # ---------------- the frozen residual pools -----------------------------
    z = np.load(a.pools, allow_pickle=True)
    y = z["y"]
    pools = {}
    for arm in ("CV", "E0_v0", "E1_ego", "E1_L", "E1_L_X", "E1_L_X_D",
                "P_ORACLE"):
        k = f"pred_{arm}"
        if k in z.files:
            pools[arm] = z[k] - y
    extra = Path(a.extra_pools)
    if extra.exists():
        ez = np.load(extra, allow_pickle=True)
        ey = ez["y"]
        for arm in [f[5:] for f in ez.files if f.startswith("pred_")]:
            pools[arm] = ez[f"pred_{arm}"] - ey
        print(f"[place] + extra pools {sorted(f[5:] for f in ez.files if f.startswith('pred_'))}",
              flush=True)

    import pandas as pd
    wdf = pd.read_parquet(a.windows)
    keep = (np.isfinite(wdf["y_long"]) & np.isfinite(wdf["v"])
            & (wdf["obst_cov"] > 0))
    pool_v = wdf.loc[keep, "v"].to_numpy(np.float64)
    assert len(pool_v) == len(y), "pool/window misalignment"

    edges = np.quantile(pool_v, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    pool_bin = np.clip(np.digitize(pool_v, edges[1:-1]), 0, 9)
    can_bin = np.clip(np.digitize(v0, edges[1:-1]), 0, 9)
    by_bin = {b: np.flatnonzero(pool_bin == b) for b in range(10)}

    realised_by: dict = {}

    def place(pool, tag, pool_bins=None):
        """E-GOAL-1's injection, verbatim: replace the ALONG coordinate with
        `true + resampled residual`, hold the cross-track at its learned value,
        average over N_SEEDS draws, then bootstrap over EPISODES.

        `pool_bins` is the speed decile of every POOL row. For the frozen
        dev-corpus pools that is `pool_bin` (99,935 rows); for the head's own
        residual -- the decorrelation control -- the pool IS the canonical
        window set, so its bins are `can_bin`. E-GOAL-1's `eg_place.py` makes
        exactly this distinction (its control block re-derives `pb` from `v0`
        rather than reusing `by_bin`), and getting it wrong silently indexes
        one array with another array's indices.
        """
        nonlocal realised_by
        bins = pool_bin if pool_bins is None else pool_bins
        idx_by = ({b: np.flatnonzero(bins == b) for b in range(10)}
                  if pool_bins is not None else by_bin)
        out = {}
        for mode in ("iid", "by_speed"):
            acc = np.zeros(W)
            for s in range(N_SEEDS):
                rng = np.random.default_rng(1000 + s)
                if mode == "iid":
                    draw = pool[rng.integers(0, len(pool), W)]
                else:
                    draw = np.empty(W)
                    for b in range(10):
                        m = can_bin == b
                        src = (idx_by[b] if len(idx_by[b])
                               else np.arange(len(pool)))
                        draw[m] = pool[rng.choice(src, m.sum())]
                xc = head_cross if cross_by_seed is None else cross_by_seed[s]
                acc += realise(np.stack([g_true[:, 0] + draw, xc], 1), fan, gt)
            rr = acc / N_SEEDS
            pair = ci_paired(rr, a0, eid)
            rec = float((a0.mean() - rr.mean()) / headroom)
            out[mode] = {
                "along_rms_m_pool": r4(float(np.sqrt(np.mean(pool ** 2)))),
                "along_mae_m_pool": r4(float(np.mean(np.abs(pool)))),
                "realised_ade_0_2s": ci_single(rr, eid),
                "realised_rms_family": ci_single(rr ** 2, eid, reduce="rms"),
                "recovery_of_headroom": r4(rec),
                "paired_vs_as_trained": pair,
                "separated": sep(pair),
                "verdict": ("BETTER" if (sep(pair) and pair["delta"] < 0)
                            else "WORSE" if (sep(pair) and pair["delta"] > 0)
                            else "not separated")}
            realised_by[f"{tag}|{mode}"] = rr
            print(f"  {tag:16s} {mode:9s} rms={out[mode]['along_rms_m_pool']:.4f} "
                  f"realised={rr.mean():.4f} rec={100*rec:+.1f}% "
                  f"d={pair['delta']:+.4f} [{pair['lo']:+.4f},{pair['hi']:+.4f}] "
                  f"{out[mode]['verdict']}", flush=True)
        return out

    # ---- the decorrelation control: THE head's own along residual ----------
    # For `parent_resampled` the along control pool is the PARENT head's own
    # 881 along residuals -- i.e. E-GOAL-1's control verbatim, carried to any n
    # by the same resampler as its cross partner. It MUST stay separated-WORSE.
    if a.cross == "parent_resampled":
        _P = np.load(GI / "raw" / "gi_head_preds.npz", allow_pickle=True)
        ctrl_pool = _P["best"][:, 0] - _P["gt_end"][:, 0]
        # The control pool's rows ARE the canonical 881 windows, so its speed
        # bins must come from the canonical v0 -- NOT from `pool_bin` (the
        # 99,935-row dev corpus) and NOT from `can_bin` when W = 13,198.
        # ⭐ Admissible only because F-A part A PROVED the 600-episode dump's
        # first 881 rows are BIT-IDENTICAL to the 40-episode dump
        # (`raw/e2_fanmatch_pod2_40_vs_600.json`, max|diff| = 0.0 on every
        # field): v0[:881] therefore IS the canonical v0.
        assert len(ctrl_pool) == 881, "parent control pool is not the 881"
        ctrl_bins = np.clip(np.digitize(v0[:881], edges[1:-1]), 0, 9)
    else:
        ctrl_pool = head[:, 0] - g_true[:, 0]
        ctrl_bins = can_bin
    res["decorrelation_control"] = {
        "what": ("the learned head's OWN along-track residual pushed through "
                 "the IDENTICAL resampler. At n = 40 with --cross parent this "
                 "is E-GOAL-1's control verbatim (published -33.1 % iid / "
                 "-34.9 % by_speed against its actual correlated -10.4 %)."),
        "along_rms_m": r4(float(np.sqrt(np.mean(ctrl_pool ** 2)))),
        "resampled": place(ctrl_pool, "HEAD_RESAMP", pool_bins=ctrl_bins)}
    if a.cross == "parent" and W == 881:
        res["decorrelation_control"]["actual_correlated_realised"] = \
            r4(realise(head, fan, gt).mean())
        res["decorrelation_control"]["actual_correlated_recovery"] = \
            r4(float((a0.mean() - realise(head, fan, gt).mean()) / headroom))

    for arm, pool in pools.items():
        res["arms"][arm] = place(pool, arm)

    # ---- ⭐ THE MECHANISM TEST: arm-vs-arm, paired on the SAME windows ------
    # `E1_ego` vs as-trained says the head helps. It does NOT say WHAT in the
    # head helps. These three contrasts do, and the third one is the control
    # that must come back NULL:
    #   E1_ego vs E1_nohist      -- what one second of speed/accel history buys
    #   E1_ego vs E1_noise_hist  -- the same comparison against NOISE in the
    #                               same four columns (capacity held fixed)
    #   E1_nohist vs E1_noise_hist -- ⛔ MUST BE NULL. Real absence of history
    #                               and fake presence of history must be
    #                               indistinguishable, or the noise arm is
    #                               leaking something.
    CONTRASTS = [("E1_ego", "E1_nohist", "what 1 s of speed/accel history buys"),
                 ("E1_ego", "E1_noise_hist",
                  "history vs NOISE in the same columns (capacity fixed)"),
                 ("E1_nohist", "E1_noise_hist",
                  "⛔ MUST BE NULL: dropped history vs fake history"),
                 ("E1_ego", "E1_L", "the whole lead-vehicle block"),
                 ("E1_ego", "E0_v0", "history+kinematics vs v0 alone")]
    res["mechanism_contrasts"] = {}
    for x, y, why in CONTRASTS:
        for m in ("iid", "by_speed"):
            kx, ky = f"{x}|{m}", f"{y}|{m}"
            if kx in realised_by and ky in realised_by:
                pc = ci_paired(realised_by[kx], realised_by[ky], eid)
                res["mechanism_contrasts"][f"{x}__vs__{y}|{m}"] = {
                    "_what": why, "paired_delta_ade": pc, "separated": sep(pc),
                    "sign": ("x_better" if pc["delta"] < 0 else "y_better")}
                # ASCII only: the dev box console is cp1252 and a Greek delta
                # here raised UnicodeEncodeError mid-run (the JSON was never
                # written and a grep-filtered log hid the traceback entirely).
                print(f"  d[{x} vs {y:14s}] {m:9s} {pc['delta']:+.4f} "
                      f"[{pc['lo']:+.4f},{pc['hi']:+.4f}] "
                      f"{'SEP' if sep(pc) else 'null'}", flush=True)

    # ---- the FAMILY-MATCHED requirement curve ------------------------------
    # ⚠️ class BAR-INHERITED-FROM-THE-WRONG-FAMILY. The inherited 0.813 m is an
    # `ISO`-family number; the measured heads are near-unbiased (alpha ~0.996,
    # so not `SHRINK`) and heavy-tailed (RMS/MAE ~1.87, so not `ISO`). The bar
    # must be RUN, not read -- scale this stream's own residual pool by k and
    # sweep k through the real REF-C rule. E-GOAL-1 measured sigma_0 = 1.1434 m
    # on the 40-episode deployment; whether that survives the (easier)
    # 600-episode deployment is a question only this sweep answers.
    if a.curve and "E1_L" in pools:
        base = pools["E1_L"]
        curve = []
        for k in (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
            acc = np.zeros(W)
            for s in range(N_SEEDS):
                rng = np.random.default_rng(2000 + s)
                draw = k * base[rng.integers(0, len(base), W)]
                xc = head_cross if cross_by_seed is None else cross_by_seed[s]
                acc += realise(np.stack([g_true[:, 0] + draw, xc], 1), fan, gt)
            rr = acc / N_SEEDS
            pair = ci_paired(rr, a0, eid)
            curve.append({"scale_k": k,
                          "along_rms_m": r4(k * np.sqrt(np.mean(base ** 2))),
                          "realised": r4(rr.mean()),
                          "recovery": r4(float((a0.mean() - rr.mean())
                                               / headroom)),
                          "separated": sep(pair),
                          "paired_delta": pair["delta"]})
            print(f"  curve k={k:<6} rms={curve[-1]['along_rms_m']:.4f} "
                  f"realised={curve[-1]['realised']:.4f} "
                  f"recovery={100*curve[-1]['recovery']:+.1f}%", flush=True)
        xs = np.array([c["along_rms_m"] for c in curve])
        ys = np.array([c["recovery"] for c in curve])

        def x_at(target):
            for i in range(len(ys) - 1):
                if ((ys[i] - target) * (ys[i + 1] - target) <= 0
                        and ys[i] != ys[i + 1]):
                    f = (ys[i] - target) / (ys[i] - ys[i + 1])
                    return r4(xs[i] + f * (xs[i + 1] - xs[i]))
            return None

        res["family_matched_curve"] = {
            "family": ("EMPIRICAL -- this stream's own frozen OOF along-track "
                       "residuals, scaled. Near-unbiased and heavy-tailed: "
                       "matches NEITHER `ISO` NOR `SHRINK`, which is why the "
                       "rule is RUN and not read off a curve."),
            "monotone": bool(np.all(np.diff(ys) <= 1e-9)),
            "sigma_0_breakeven_along_rms_m": x_at(0.0),
            "sigma_50_halfprize_along_rms_m": x_at(0.5),
            "EGOAL_1_n40_sigma_0_m": 1.1434,
            "EGOAL_1_n40_sigma_50_m": 0.5907,
            "inherited_ISO_sigma_0_m": 0.813,
            "grid": curve}
        print("[curve] sigma_0 = "
              f"{res['family_matched_curve']['sigma_0_breakeven_along_rms_m']} m"
              f"  sigma_50 = "
              f"{res['family_matched_curve']['sigma_50_halfprize_along_rms_m']} m"
              f"  (E-GOAL-1 n=40: 1.1434 / 0.5907; inherited ISO: 0.813)",
              flush=True)

    # ---- F-B gate on the RECOVERIES themselves (n = 40 only) ---------------
    # ⚠️ compared against the RAW JSON `eg_place.json`, never against the
    # doc-rounded table in EGOAL_1.md (primary sources only).
    if W == 881:
        ref = json.loads((EG1 / "raw" / "eg_place.json").read_text())
        got, want, dev = {}, {}, {}
        for k in res["arms"]:
            for m in ("iid", "by_speed"):
                rk = f"{k}|{m}"
                if rk in ref["arms"]:
                    got[rk] = res["arms"][k][m]["recovery_of_headroom"]
                    want[rk] = ref["arms"][rk]["recovery_of_headroom"]
                    dev[rk] = round(100 * (got[rk] - want[rk]), 3)
        for m in ("iid", "by_speed"):
            rk = f"HEAD_RESAMP|{m}"
            got[rk] = res["decorrelation_control"]["resampled"][m][
                "recovery_of_headroom"]
            want[rk] = ref["decorrelation_control"]["resampled"][m][
                "recovery_of_headroom"]
            dev[rk] = round(100 * (got[rk] - want[rk]), 3)
        res["F_B_recovery_fidelity"] = {
            "_source": str(EG1 / "raw" / "eg_place.json"),
            "eg_place_json": want, "mine": got,
            "deviation_recovery_points": dev,
            "max_abs_deviation_points": max(abs(v) for v in dev.values()),
            "doc_table_EGOAL_1_for_reference": EG1_RECOVERY,
            "parent_resamp_doc": EG1_PARENT_RESAMP}
        print("[F-B recovery] max |dev| vs eg_place.json = "
              f"{res['F_B_recovery_fidelity']['max_abs_deviation_points']:.3f} "
              "recovery points", flush=True)

    res["_wall_s"] = round(time.time() - t_start, 1)
    out = Path(a.out) if a.out else STREAM / "raw" / f"e2_place_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[place] -> {out}  ({res['_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
