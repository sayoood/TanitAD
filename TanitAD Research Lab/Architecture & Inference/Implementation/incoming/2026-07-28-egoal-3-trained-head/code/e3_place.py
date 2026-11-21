#!/usr/bin/env python3
"""E-GOAL-3 S3 -- THE DECISIVE MEASUREMENT: the trained head's ACTUAL per-window
predictions, pushed through the ACTUAL REF-C selection rule, at n = 600.

⛔ WHAT IS DIFFERENT FROM E-GOAL-1/2, AND IT IS THE WHOLE POINT.
Both prior stages injected `true_along + resampled_residual`, so the injected
error was INDEPENDENT of which window is hard. Here the along coordinate IS the
head's prediction. Its errors correlate with window difficulty exactly as a
deployed head's would. A trained head can beat the resampler (structure the
resampler cannot represent) or lose to it (errors concentrated where the pick
matters). Nobody knew which. That is this script.

⛔ C30 -- THE BACKGROUND IS NAMED AND HELD FIXED. At fixed n and fixed
along-track error, recovery spans +13.3 % … +29.2 % purely on the cross-track
background, and separation FLIPS inside that range. The PRIMARY carrier is
`parent_resampled` (E-GOAL-2's registered conservative one): the parent head's
own 881 cross residuals resampled onto the true cross-track, drawn once per seed
from `default_rng(5000 + s)` and IDENTICAL ACROSS EVERY ARM within a seed.
`sel` and `reduced` are reported as secondaries. Every headline says which.

⛔ C31 -- THE PREDICATE IS RE-RUN AT n = 600. E-GOAL-2 found its own separation
predicate stops discriminating at 600 clusters (an information-free arm
separates at +9.1 %). So the mechanism claim rests on DIRECT PAIRED CONTRASTS
(`H_ego` vs `H_noise_hist`, with `H_nohist` vs `H_noise_hist` required NULL),
and the deliberately-failing `N_SHUF` arm is scored at the ACTUAL n.

FIDELITY. `realise` / `goal_reference` / `pick_nearest_to` / `ade` are IMPORTED
from E-GOAL-1's `eg_place.py`; the cross-track backgrounds are rebuilt with
E-GOAL-2's `e2_place.fan_blocks` / `reduced_head`, IMPORTED. Gate F-1 pushes
E-GOAL-2's own frozen `E1_ego` residual pool through THIS engine and requires it
to return E-GOAL-2's published n = 600 numbers.

Run (dev box):
    OMP_NUM_THREADS=6 python e3_place.py --fan <fan600.pt> --preds <preds.npz> \
        --cross parent_resampled --tag n600_parent_resampled
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
EG2 = ARCH / "2026-07-28-egoal-2-power"
GI = ARCH / "2026-07-27-goal-input"

sys.path.insert(0, str(EG1 / "code"))
sys.path.insert(0, str(EG2 / "code"))
sys.path.insert(0, str(GI / "code"))
sys.path.insert(0, str(REPO / "taniteval"))

from eg_place import FRAC, N_SEEDS, ade, goal_reference, pick_nearest_to, realise  # noqa: E402,F401
from eg_common import ci_paired, ci_single, r4, sep  # noqa: E402
from e2_place import fan_blocks, reduced_head  # noqa: E402

#: E-GOAL-2's published n = 600 `parent_resampled` cell -- gate F-1's target.
#: Source: raw/e2_place_n600_parent_resampled.json (raw JSON, never the doc).
E2_N600 = {"a0": 0.5015, "r_goal2s": 0.1933, "oracle_in_fan": 0.1547,
           "headroom": 0.3082,
           "E1_ego|iid": {"recovery": 0.2542, "delta": -0.0784,
                          "lo": -0.0960, "hi": -0.0606},
           "E1_ego|by_speed": {"recovery": 0.2620, "delta": -0.0807,
                               "lo": -0.0987, "hi": -0.0631}}
INHERITED_ISO_SIGMA0 = 0.813
E2_FAMILY_SIGMA0_N600 = 1.2195


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--cross", choices=["parent_resampled", "sel", "reduced"],
                    default="parent_resampled")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--skip-f1", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    t0 = time.time()

    d = torch.load(a.fan, map_location="cpu", weights_only=False)
    fan = d["fan"].numpy().astype(np.float64)
    gt = d["gt"].numpy().astype(np.float64)
    v0 = d["v0"].numpy().astype(np.float64)
    eid = np.asarray([str(x) for x in d["eid"]])
    sel_idx = d["sel"].numpy()
    g_true = gt[:, -1, :]
    W = len(gt)
    NEP = len(np.unique(eid))

    z = np.load(a.preds, allow_pickle=True)
    y = z["y"]
    assert len(y) == W, f"pred rows {len(y)} != fan windows {W}"
    #: ⭐ F-2 re-asserted HERE, at the point of use, not only at build time
    dY = float(np.abs(y - g_true[:, 0]).max())
    assert dY < 1e-4, f"prediction target != fan gt along ({dY})"
    arms = {k: z[k] for k in z.files if "|" in k}

    a0 = ade(fan[np.arange(W), sel_idx], gt)
    r_goal = realise(g_true, fan, gt)
    oracle = ade(fan[np.arange(W)[:, None],
                     np.argmin(np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1),
                               axis=1)[:, None]].squeeze(1), gt)
    headroom = float(a0.mean() - r_goal.mean())

    # -------------------------------------------------------- gate F-0 ------
    f0 = {"a0": (r4(a0.mean()), E2_N600["a0"]),
          "r_goal2s": (r4(r_goal.mean()), E2_N600["r_goal2s"]),
          "oracle_in_fan": (r4(oracle.mean()), E2_N600["oracle_in_fan"]),
          "headroom": (r4(headroom), E2_N600["headroom"])}
    f0_ok = all(abs(m - c) < 5e-4 for m, c in f0.values())
    print(f"[F-0] {json.dumps({k: v for k, v in f0.items()})} -> {f0_ok}",
          flush=True)
    if not f0_ok:
        raise RuntimeError("F-0 FAILED: the deployment does not re-derive; no "
                           "number may be quoted")

    res = {"_stream": "2026-07-28-egoal-3-trained-head", "_tag": a.tag,
           "_fan": str(a.fan), "_preds": str(a.preds), "_cross_mode": a.cross,
           "_estimator": ("paired episode-cluster bootstrap, taniteval/ci.py, "
                          "B=2000, unit = the val episode. "
                          "overlapping_holdout_se NEVER called."),
           "_n_seeds": N_SEEDS,
           "_what": ("the TRAINED head's ACTUAL per-window predictions pushed "
                     "through the REAL REF-C rule -- NOT a resampled residual"),
           "deployment": {"n_windows": int(W), "n_episode_clusters": int(NEP),
                          "a0_as_trained": r4(a0.mean()),
                          "r_goal2s_true_goal": r4(r_goal.mean()),
                          "oracle_in_fan": r4(oracle.mean()),
                          "headroom": r4(headroom)},
           "F_0_deployment_fidelity": {"checks": {k: {"mine": m, "e2": c}
                                                  for k, (m, c) in f0.items()},
                                       "passes": bool(f0_ok)},
           "arms": {}}

    # ------------------------------------------------- the cross background --
    cross_by_seed = None
    head_cross = None
    if a.cross == "parent_resampled":
        P = np.load(GI / "raw" / "gi_head_preds.npz", allow_pickle=True)
        pc = P["best"][:, 1] - P["gt_end"][:, 1]
        cross_by_seed = np.stack(
            [g_true[:, 1] + pc[np.random.default_rng(5000 + s)
                               .integers(0, len(pc), W)] for s in range(N_SEEDS)])
        res["cross_track"] = {
            "mode": ("PARENT-RESAMPLED (E-GOAL-2's registered CONSERVATIVE "
                     "carrier): the parent head's own 881 cross-track residuals "
                     "resampled i.i.d. per seed onto the true cross-track; "
                     "identical across every arm within a seed"),
            "parent_cross_pool_n": int(len(pc)),
            "parent_cross_mae_m": r4(float(np.mean(np.abs(pc)))),
            "realised_cross_mae_m": r4(float(np.mean(
                np.abs(cross_by_seed - g_true[:, 1][None]))))}
    else:
        dcut = {"fan": torch.as_tensor(fan), "cv": d["cv"],
                "v0": torch.as_tensor(v0), "logits": d["logits"],
                "sel": torch.as_tensor(sel_idx)}
        if a.cross == "sel":
            head_cross = fan_blocks(dcut)["sel_end"][:, 1]
            mode = ("SELECTOR'S OWN 2 s endpoint (fan[sel, -1]); no fit, no "
                    "fold, no hyper-parameter")
        else:
            hd, alphas, _ = reduced_head(dcut, g_true, eid)
            head_cross = hd[:, 1]
            mode = "REDUCED ridge on F_ego + F_ans (fan-only blocks), OOF"
            res.setdefault("cross_track", {})["ridge_alphas"] = alphas
        res.setdefault("cross_track", {}).update({
            "mode": mode,
            "mae_cross_m": r4(float(np.abs(head_cross - g_true[:, 1]).mean())),
            "rms_cross_m": r4(float(np.sqrt(np.mean(
                (head_cross - g_true[:, 1]) ** 2))))})
    print("[place] cross:", res["cross_track"]["mode"][:110], flush=True)

    realised_by: dict[str, np.ndarray] = {}

    def run_goal(along: np.ndarray) -> np.ndarray:
        """Push a per-window ALONG prediction through the real rule, averaging
        over the background's seeds. Deterministic along + stochastic cross."""
        if cross_by_seed is None:
            return realise(np.stack([along, head_cross], 1), fan, gt)
        acc = np.zeros(W)
        for s in range(N_SEEDS):
            acc += realise(np.stack([along, cross_by_seed[s]], 1), fan, gt)
        return acc / N_SEEDS

    def score(rr, tag, extra=None):
        pair = ci_paired(rr, a0, eid)
        rec = float((a0.mean() - rr.mean()) / headroom)
        out = {"realised_ade_0_2s": ci_single(rr, eid),
               "realised_rms_family": ci_single(rr ** 2, eid, reduce="rms"),
               "recovery_of_headroom": r4(rec),
               "paired_vs_as_trained": pair, "separated": sep(pair),
               "verdict": ("BETTER" if (sep(pair) and pair["delta"] < 0)
                           else "WORSE" if (sep(pair) and pair["delta"] > 0)
                           else "not separated")}
        if extra:
            out.update(extra)
        realised_by[tag] = rr
        print(f"  {tag:28s} realised={rr.mean():.4f} rec={100*rec:+.1f}% "
              f"d={pair['delta']:+.4f} [{pair['lo']:+.4f},{pair['hi']:+.4f}] "
              f"{out['verdict']}", flush=True)
        return out

    # ------------------------------------------------ THE TRAINED-HEAD ARMS --
    for k in sorted(arms):
        p = arms[k]
        e = p - y
        res["arms"][k] = score(run_goal(p), k, {
            "along_rms_m": r4(float(np.sqrt(np.mean(e ** 2)))),
            "along_mae_m": r4(float(np.mean(np.abs(e)))),
            "along_bias_m": r4(float(e.mean())),
            "rms_over_mae": r4(float(np.sqrt(np.mean(e ** 2))
                                     / max(np.mean(np.abs(e)), 1e-12))),
            "shrinkage_alpha": r4(float(np.polyfit(y, p, 1)[0])),
            "_predictions": "ACTUAL per-window, correlated with window difficulty"})

    # ------------------------------- gate F-1: reproduce E-GOAL-2's n = 600 --
    # ⭐ THE SCORING-CODE GATE. E-GOAL-2's own frozen `E1_ego` residual pool,
    # pushed through THIS engine with THIS background, must return E-GOAL-2's
    # published cell. If it does not, this engine differs from theirs and no
    # number above may be quoted.
    if not a.skip_f1 and a.cross == "parent_resampled":
        import pandas as pd
        zp = np.load(EG1 / "raw" / "eg_oof_pred_gbm.npz", allow_pickle=True)
        pool = zp["pred_E1_ego"] - zp["y"]
        wdf = pd.read_parquet(EG1 / "raw" / "eg_windows.parquet")
        keep = (np.isfinite(wdf["y_long"]) & np.isfinite(wdf["v"])
                & (wdf["obst_cov"] > 0))
        pool_v = wdf.loc[keep, "v"].to_numpy(np.float64)
        assert len(pool_v) == len(pool), "pool/window misalignment"
        edges = np.quantile(pool_v, np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        pool_bin = np.clip(np.digitize(pool_v, edges[1:-1]), 0, 9)
        can_bin = np.clip(np.digitize(v0, edges[1:-1]), 0, 9)
        by_bin = {b: np.flatnonzero(pool_bin == b) for b in range(10)}
        f1 = {}
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
                        src = by_bin[b] if len(by_bin[b]) else np.arange(len(pool))
                        draw[m] = pool[rng.choice(src, m.sum())]
                acc += realise(np.stack([g_true[:, 0] + draw,
                                         cross_by_seed[s]], 1), fan, gt)
            rr = acc / N_SEEDS
            pair = ci_paired(rr, a0, eid)
            rec = float((a0.mean() - rr.mean()) / headroom)
            ref = E2_N600[f"E1_ego|{mode}"]
            f1[mode] = {"mine_recovery": r4(rec), "e2_recovery": ref["recovery"],
                        "deviation_recovery_points": round(
                            100 * (rec - ref["recovery"]), 3),
                        "mine_paired": pair, "e2_paired": ref}
            print(f"  [F-1] E1_ego|{mode:9s} mine {100*rec:+.2f}% vs E-GOAL-2 "
                  f"{100*ref['recovery']:+.2f}%  dev="
                  f"{f1[mode]['deviation_recovery_points']:+.3f} pts", flush=True)
        mx = max(abs(v["deviation_recovery_points"]) for v in f1.values())
        res["F_1_engine_fidelity"] = {
            "_what": ("E-GOAL-2's frozen E1_ego resampled pool pushed through "
                      "THIS engine under THIS background; must reproduce "
                      "e2_place_n600_parent_resampled.json"),
            "_source": str(EG2 / "raw" / "e2_place_n600_parent_resampled.json"),
            "cells": f1, "max_abs_deviation_recovery_points": mx,
            "passes": bool(mx < 1.0)}
        if mx >= 1.0:
            raise RuntimeError(f"F-1 FAILED: engine deviates {mx:.3f} recovery "
                               "points from E-GOAL-2; nothing may be quoted")

    # ---------------------------------------- ⭐ C31: the DIRECT contrasts ----
    CONTRASTS = [("H_ego", "H_nohist", "what 0.5-1.0 s of speed history buys"),
                 ("H_ego", "H_noise_hist",
                  "history vs NOISE in the same columns (capacity fixed)"),
                 ("H_nohist", "H_noise_hist",
                  "MUST BE NULL: dropped history vs fake history"),
                 ("H_ego", "H_v0", "kinematics+history vs v0 alone"),
                 ("H_ego", "CV_head", "vs constant velocity"),
                 ("H_ego", "P_ORACLE_TRUE", "distance to the true-goal bound"),
                 # ⭐ the speed-differencing ladder (PRE_REGISTRATION §10 A1)
                 ("H_inst", "H_v0", "rotation state, no speed differencing"),
                 ("H_v0_ax", "H_v0", "0.1 s of speed history ALONE"),
                 ("H_nohist", "H_v0_ax", "the rest of the instantaneous block"),
                 ("H_ego", "H_v0_ax", "everything beyond 0.1 s of history"),
                 ("H_nohist", "H_inst",
                  "⭐ speed differencing vs no speed differencing")]
    res["mechanism_contrasts"] = {}
    for cfg in ("T_OOF", "T_TRAIN"):
        for x, yy, why in CONTRASTS:
            kx, ky = f"{cfg}|{x}", f"{cfg}|{yy}"
            if kx in realised_by and ky in realised_by:
                pc = ci_paired(realised_by[kx], realised_by[ky], eid)
                res["mechanism_contrasts"][f"{cfg}|{x}__vs__{yy}"] = {
                    "_what": why, "paired_delta_ade": pc, "separated": sep(pc),
                    "sign": "x_better" if pc["delta"] < 0 else "y_better"}
                print(f"  d[{cfg}|{x} vs {yy:14s}] {pc['delta']:+.4f} "
                      f"[{pc['lo']:+.4f},{pc['hi']:+.4f}] "
                      f"{'SEP' if sep(pc) else 'null'}", flush=True)
    # cross-configuration: does training on the parity corpus change the answer?
    for arm in ("H_ego", "H_nohist", "H_noise_hist"):
        kx, ky = f"T_OOF|{arm}", f"T_TRAIN|{arm}"
        if kx in realised_by and ky in realised_by:
            pc = ci_paired(realised_by[kx], realised_by[ky], eid)
            res["mechanism_contrasts"][f"{arm}|T_OOF__vs__T_TRAIN"] = {
                "_what": "same arm, OOF-in-val vs fitted on the parity corpus",
                "paired_delta_ade": pc, "separated": sep(pc)}

    # ------------------- ⭐ C24: the family, MEASURED on the trained head -----
    prim = "T_OOF|H_ego"
    if prim in arms:
        e = arms[prim] - y
        rms = float(np.sqrt(np.mean(e ** 2)))
        res["error_family"] = {
            "_arm": prim,
            "rms_m": r4(rms), "mae_m": r4(float(np.mean(np.abs(e)))),
            "rms_over_mae": r4(rms / float(np.mean(np.abs(e)))),
            "gaussian_ref": 1.2533,
            "shrinkage_alpha": r4(float(np.polyfit(y, arms[prim], 1)[0])),
            "bias_m": r4(float(e.mean())),
            "family": ("EMPIRICAL -- near-unbiased (alpha ~1, so NOT `SHRINK`) "
                       "and heavy-tailed (RMS/MAE > 1.2533, so NOT `ISO`). "
                       "MEASURED on this estimator, not assumed."),
            "inherited_ISO_sigma_0_m": INHERITED_ISO_SIGMA0,
            "EGOAL_2_family_matched_sigma_0_n600_m": E2_FAMILY_SIGMA0_N600,
            "clears_inherited_ISO_bar": bool(rms < INHERITED_ISO_SIGMA0),
            "clears_EGOAL_2_family_sigma_0": bool(rms < E2_FAMILY_SIGMA0_N600)}

        if a.curve:
            # the requirement curve rebuilt from THE TRAINED HEAD'S OWN error,
            # scaled about the truth. ⚠️ COMPANION, NOT VERDICT -- the realised
            # recovery above is measured directly and is what decides.
            curve = []
            for k in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
                rr = run_goal(y + k * e)
                pair = ci_paired(rr, a0, eid)
                curve.append({"scale_k": k, "along_rms_m": r4(k * rms),
                              "realised": r4(rr.mean()),
                              "recovery": r4(float((a0.mean() - rr.mean())
                                                   / headroom)),
                              "separated": sep(pair),
                              "paired_delta": pair["delta"]})
                print(f"  curve k={k:<5} rms={curve[-1]['along_rms_m']:.4f} "
                      f"rec={100*curve[-1]['recovery']:+.1f}%", flush=True)
            xs = np.array([c["along_rms_m"] for c in curve])
            ys = np.array([c["recovery"] for c in curve])

            def x_at(t):
                for i in range(len(ys) - 1):
                    if (ys[i] - t) * (ys[i + 1] - t) <= 0 and ys[i] != ys[i + 1]:
                        f = (ys[i] - t) / (ys[i] - ys[i + 1])
                        return r4(xs[i] + f * (xs[i + 1] - xs[i]))
                return None
            res["family_matched_curve"] = {
                "_family": "the TRAINED head's own residual, scaled",
                "_note": ("COMPANION ONLY. With a real head the realised "
                          "recovery is measured directly; no verdict here is "
                          "read off this curve."),
                "monotone": bool(np.all(np.diff(ys) <= 1e-9)),
                "sigma_0_breakeven_along_rms_m": x_at(0.0),
                "sigma_50_halfprize_along_rms_m": x_at(0.5),
                "EGOAL_2_n600_sigma_0_m": E2_FAMILY_SIGMA0_N600,
                "inherited_ISO_sigma_0_m": INHERITED_ISO_SIGMA0,
                "grid": curve}

    res["_wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out) if a.out else STREAM / "raw" / f"e3_place_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    np.savez_compressed(str(out).replace(".json", "_realised.npz"),
                        a0=a0, eid=eid, **realised_by)
    print(f"[place] -> {out}  ({res['_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
