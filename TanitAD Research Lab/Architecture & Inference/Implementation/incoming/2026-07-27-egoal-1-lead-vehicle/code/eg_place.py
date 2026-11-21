#!/usr/bin/env python3
"""E-GOAL-1 S4 -- placement on the requirement curve, by RUNNING THE RULE.

⚠️ RETRACTION CLASS `RMS-PLACED-ON-A-NOISE-CURVE` (parent stream, measured
2026-07-27): grading a real estimator by reading its RMS off an isotropic-noise
curve OVER-PREDICTED the damage by 5.7x, because real error is heavy-tailed.
The parent's own instruction is *"grade it by running it through the rule."*
That is what this script does, and the naive curve read is reported BESIDE it as
the counter-example, never instead of it.

METHOD
------
1. FIDELITY -- reproduce the parent's committed pipeline numbers from raw before
   quoting a new one: as-trained `ade_0_2s` 0.4714, `R_goal2s` (true 2 s goal)
   0.2009, `R_head_oof` 0.4996, and the `pick_nearest_to(GT) == oracle_in_fan`
   per-window identity. Any mismatch aborts.
2. INJECT -- take the parent's LEARNED cross-track (its realistic value, exactly
   as its S5 decomposition does) and replace the ALONG-track coordinate with
   `true_along + residual`, where `residual` is RESAMPLED FROM THIS STREAM'S
   MEASURED OUT-OF-FOLD ALONG-TRACK RESIDUALS. The family is therefore the
   estimator's OWN empirical error distribution -- not `ISO`, not `SHRINK`, not
   a Gaussian.
   Two resamplers, both reported:
     `iid`     -- draw residuals uniformly from the pool
     `by_speed`-- draw from the pool restricted to a matched ego-speed decile,
                  preserving the heteroscedasticity a single RMS hides
3. PLACE -- realised `ade_0_2s`, recovery of the -0.2705 headroom, paired
   episode-cluster bootstrap vs as-trained, and the naive curve read beside it.

Run:  OMP_NUM_THREADS=6 python eg_place.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eg_common import (BAR_BREAKEVEN_M, BAR_HALF_M, N_BOOT, REPO,  # noqa: E402
                       ci_paired, ci_single, r4, sep)

FRAC = np.array([0.25, 0.5, 0.75, 1.0])[None, :, None]
A0_COMMITTED = 0.4714
GOAL_COMMITTED = 0.2009
HEAD_COMMITTED = 0.4996
HEADROOM = 0.2705            # 0.4714 - 0.2009
N_SEEDS = 16


def ade(traj, gt):
    return np.linalg.norm(traj - gt, axis=-1).mean(-1)


def goal_reference(goal):
    return goal[:, None, :] * FRAC


def pick_nearest_to(ref, fan):
    """(!) the proximity metric MUST be the same mean-over-waypoint L2 that
    scores the pick; a flat-8 L2 has a different argmin and produced a stable,
    plausible, wrong number upstream."""
    d = np.linalg.norm(fan - ref[:, None], axis=-1).mean(-1)
    return d.argmin(1)


def realise(goal, fan, gt):
    idx = pick_nearest_to(goal_reference(goal), fan)
    return ade(fan[np.arange(len(fan)), idx], gt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(Path(__file__).resolve().parent.parent
                                         / "raw"))
    ap.add_argument("--model", default="gbm")
    a = ap.parse_args(argv)
    raw = Path(a.raw)

    d = torch.load(REPO / "taniteval" / "results" / "fan_refc-xl-30k.pt",
                   map_location="cpu", weights_only=False)
    fan = d["fan"].numpy().astype(np.float64)
    gt = d["gt"].numpy().astype(np.float64)
    v0 = d["v0"].numpy().astype(np.float64)
    eid = np.asarray([str(x) for x in d["eid"]])
    sel = d["sel"].numpy()
    g_true = gt[:, -1, :]

    P = np.load(REPO / "TanitAD Research Hub" / "Architecture & Inference"
                / "Implementation" / "incoming" / "2026-07-27-goal-input"
                / "raw" / "gi_head_preds.npz", allow_pickle=True)
    head = P["best"]                       # parent's best OOF goal head, [881,2]
    assert np.abs(P["gt_end"] - g_true).max() == 0.0, "parent gt_end mismatch"

    a0 = ade(fan[np.arange(len(fan)), sel], gt)
    r_goal = realise(g_true, fan, gt)
    r_head = realise(head, fan, gt)
    oracle = ade(fan[np.arange(len(fan))[:, None],
                     np.argmin(np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1),
                               axis=1)[:, None]].squeeze(1), gt)

    fid = {
        "a0_as_trained": {"mine": r4(a0.mean()), "committed": A0_COMMITTED},
        "r_goal2s_true_goal": {"mine": r4(r_goal.mean()),
                               "committed": GOAL_COMMITTED},
        "r_head_oof": {"mine": r4(r_head.mean()), "committed": HEAD_COMMITTED},
        "oracle_in_fan": {"mine": r4(oracle.mean()), "committed": 0.1640},
        "delta_a0_minus_goal": {"mine": r4(a0.mean() - r_goal.mean()),
                                "committed": HEADROOM},
    }
    fid["passes"] = all(abs(v["mine"] - v["committed"]) < 5e-4
                        for k, v in fid.items() if isinstance(v, dict))
    print("[place] fidelity:", json.dumps(fid, indent=1), flush=True)
    if not fid["passes"]:
        raise RuntimeError("S4 FIDELITY FAILED -- the parent's pipeline does not "
                           "reproduce; no new number may be quoted")

    # ------------------------------------------------------------- residuals #
    z = np.load(raw / f"eg_oof_pred_{a.model}.npz", allow_pickle=True)
    y = z["y"]
    pools = {}
    for arm in ("CV", "E0_v0", "E1_ego", "E1_L", "E1_L_X", "E1_L_X_D",
                "P_ORACLE"):
        k = f"pred_{arm}"
        if k in z.files:
            pools[arm] = z[k] - y
    # ego speed of every pool row, for the speed-matched resampler
    pool_v = None
    import pandas as pd
    wdf = pd.read_parquet(raw / "eg_windows.parquet")
    keep = (np.isfinite(wdf["y_long"]) & np.isfinite(wdf["v"])
            & (wdf["obst_cov"] > 0))
    pool_v = wdf.loc[keep, "v"].to_numpy(np.float64)
    assert len(pool_v) == len(y), "pool/window misalignment"

    # speed deciles shared by pool and canonical windows
    edges = np.quantile(pool_v, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    pool_bin = np.clip(np.digitize(pool_v, edges[1:-1]), 0, 9)
    can_bin = np.clip(np.digitize(v0, edges[1:-1]), 0, 9)
    by_bin = {b: np.flatnonzero(pool_bin == b) for b in range(10)}

    res = {"what": "E-GOAL-1 S4 -- measured error structure run THROUGH the rule",
           "fidelity": fid,
           "bars": {"breakeven_along_rms_m": BAR_BREAKEVEN_M,
                    "halfprize_along_rms_m": BAR_HALF_M,
                    "source": "GOAL_INPUT.md S5 conditional along-track spec"},
           "canonical": {"n_windows": int(len(gt)),
                         "n_episodes": int(len(np.unique(eid))),
                         "a0": r4(a0.mean()), "r_goal2s": r4(r_goal.mean()),
                         "headroom": HEADROOM},
           "arms": {}}

    head_cross = head[:, 1]

    # ⛔⛔ THE CONTROL THAT DECIDES WHETHER ANY OF THIS IS REAL.
    # The injection replaces the along coordinate with `true + resampled
    # residual`, so the injected error is INDEPENDENT of which window is hard.
    # A real head's error CORRELATES with window difficulty (a hard window gets
    # both a bad goal and a bad pick), and decorrelating it is OPTIMISTIC. The
    # parent's own S5 decomposition has exactly this property.
    # So: take the PARENT'S OWN along-track residual and push it through the
    # IDENTICAL resampler. Its actual, correlated value is already known
    # (`R_head_oof` = 0.4996, recovery -10.4 %). If the resampled version lands
    # near -10.4 %, decorrelation costs nothing and this stream's arms stand. If
    # it lands far above, every arm below is inflated by that much and must be
    # discounted -- which is stated either way.
    parent_resid = head[:, 0] - g_true[:, 0]
    ctrl = {}
    for mode in ("iid", "by_speed"):
        acc = np.zeros(len(gt))
        for s in range(N_SEEDS):
            rng = np.random.default_rng(1000 + s)
            if mode == "iid":
                draw = parent_resid[rng.integers(0, len(parent_resid), len(gt))]
            else:
                draw = np.empty(len(gt))
                for b in range(10):
                    m = can_bin == b
                    pb = np.flatnonzero(
                        np.clip(np.digitize(v0, edges[1:-1]), 0, 9) == b)
                    src = pb if len(pb) else np.arange(len(parent_resid))
                    draw[m] = parent_resid[rng.choice(src, m.sum())]
            acc += realise(np.stack([g_true[:, 0] + draw, head_cross], 1), fan, gt)
        rr = acc / N_SEEDS
        pair = ci_paired(rr, a0, eid)
        ctrl[mode] = {
            "along_rms_m": r4(np.sqrt(np.mean(parent_resid ** 2))),
            "realised_ade_0_2s": ci_single(rr, eid),
            "recovery_of_headroom": r4((a0.mean() - rr.mean()) / HEADROOM),
            "paired_vs_as_trained": pair, "separated": sep(pair)}
        print(f"  PARENT_RESAMP {mode:9s} rms="
              f"{ctrl[mode]['along_rms_m']:.4f} m  realised={rr.mean():.4f}  "
              f"recovery={100*ctrl[mode]['recovery_of_headroom']:+.1f}%  "
              f"(actual correlated head: -10.4 %)", flush=True)
    res["decorrelation_control"] = {
        "what": ("the parent's OWN along-track residual pushed through the "
                 "IDENTICAL resampler, against its known correlated value"),
        "parent_actual_correlated_recovery": -0.104,
        "parent_actual_realised": HEAD_COMMITTED,
        "resampled": ctrl,
        "reading": ("if resampled ~= -10.4 %, decorrelation is free and this "
                    "stream's arms stand as measured; if far above, every arm "
                    "is inflated by the same amount")}

    for arm, pool in pools.items():
        rms = float(np.sqrt(np.mean(pool ** 2)))
        for mode in ("iid", "by_speed"):
            acc = np.zeros(len(gt))
            for s in range(N_SEEDS):
                rng = np.random.default_rng(1000 + s)
                if mode == "iid":
                    draw = pool[rng.integers(0, len(pool), len(gt))]
                else:
                    draw = np.empty(len(gt))
                    for b in range(10):
                        m = can_bin == b
                        src = by_bin[b] if len(by_bin[b]) else np.arange(len(pool))
                        draw[m] = pool[rng.choice(src, m.sum())]
                goal = np.stack([g_true[:, 0] + draw, head_cross], 1)
                acc += realise(goal, fan, gt)
            realised = acc / N_SEEDS
            ci = ci_single(realised, eid)
            pair = ci_paired(realised, a0, eid)
            rec = (a0.mean() - realised.mean()) / HEADROOM
            res["arms"][f"{arm}|{mode}"] = {
                "along_rms_m_measured_on_dev_corpus": r4(rms),
                "speed_err_ms": r4(rms / 2.0),
                "realised_ade_0_2s": ci,
                "recovery_of_headroom": r4(rec),
                "paired_vs_as_trained": pair,
                "separated": sep(pair),
                "verdict": ("BETTER" if (sep(pair) and pair["delta"] < 0)
                            else "WORSE" if (sep(pair) and pair["delta"] > 0)
                            else "not separated"),
                "clears_breakeven_bar": bool(rms < BAR_BREAKEVEN_M),
                "clears_halfprize_bar": bool(rms < BAR_HALF_M)}
            print(f"  {arm:10s} {mode:9s} rms={rms:.4f} m  realised="
                  f"{realised.mean():.4f}  recovery={100*rec:+.1f}%  "
                  f"{res['arms'][f'{arm}|{mode}']['verdict']}", flush=True)

    # ---- ⭐ THE FAMILY-MATCHED REQUIREMENT CURVE --------------------------- #
    # The brief asks for a placement "using a family that matches its ERROR
    # STRUCTURE". Neither pre-registered family does: the measured heads are
    # near-UNBIASED (shrinkage alpha ~0.996, so not `SHRINK`) and strongly
    # HEAVY-TAILED (RMS/MAE ~1.87 against a Gaussian's 1.2533, so not `ISO`).
    # So the family is built from the estimator itself: scale THIS stream's own
    # OOF residual pool by k and sweep k. The resulting sigma_0 / sigma_50 are
    # the break-even and half-prize requirements FOR A SUPPLIER WITH THIS ERROR
    # SHAPE, and they are what should be quoted at a supplier instead of 0.813 m.
    base = pools["E1_L"]
    curve = []
    for k in (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
        acc = np.zeros(len(gt))
        for s in range(N_SEEDS):
            rng = np.random.default_rng(2000 + s)
            draw = k * base[rng.integers(0, len(base), len(gt))]
            acc += realise(np.stack([g_true[:, 0] + draw, head_cross], 1), fan, gt)
        rr = acc / N_SEEDS
        pair = ci_paired(rr, a0, eid)
        curve.append({"scale_k": k,
                      "along_rms_m": r4(k * np.sqrt(np.mean(base ** 2))),
                      "realised": r4(rr.mean()),
                      "recovery": r4((a0.mean() - rr.mean()) / HEADROOM),
                      "separated": sep(pair),
                      "paired_delta": pair["delta"]})
        print(f"  curve k={k:<6} rms={curve[-1]['along_rms_m']:.4f} m  "
              f"realised={curve[-1]['realised']:.4f}  "
              f"recovery={100*curve[-1]['recovery']:+.1f}%", flush=True)

    xs = np.array([c["along_rms_m"] for c in curve])
    ys = np.array([c["recovery"] for c in curve])

    def x_at(target):
        for i in range(len(ys) - 1):
            if (ys[i] - target) * (ys[i + 1] - target) <= 0 and ys[i] != ys[i + 1]:
                f = (ys[i] - target) / (ys[i] - ys[i + 1])
                return r4(xs[i] + f * (xs[i + 1] - xs[i]))
        return None

    res["family_matched_curve"] = {
        "family": ("EMPIRICAL -- this stream's own OOF along-track residuals, "
                   "scaled. Near-unbiased (alpha~0.996) and heavy-tailed "
                   "(RMS/MAE ~1.87 vs Gaussian 1.2533): matches NEITHER `ISO` "
                   "NOR `SHRINK`, which is precisely why the rule was run "
                   "instead of a curve being read."),
        "monotone": bool(np.all(np.diff(ys) <= 1e-9)),
        "sigma_0_breakeven_along_rms_m": x_at(0.0),
        "sigma_50_halfprize_along_rms_m": x_at(0.5),
        "parent_ISO_conditional_sigma_0_m": BAR_BREAKEVEN_M,
        "parent_ISO_conditional_sigma_50_m": BAR_HALF_M,
        "grid": curve}
    print("[place] family-matched sigma_0 = "
          f"{res['family_matched_curve']['sigma_0_breakeven_along_rms_m']} m "
          f"(parent ISO-conditional: {BAR_BREAKEVEN_M} m)", flush=True)

    # ---- the naive curve read, reported as the COUNTER-EXAMPLE it is ------- #
    gi = json.loads((REPO / "TanitAD Research Hub" / "Architecture & Inference"
                     / "Implementation" / "incoming" / "2026-07-27-goal-input"
                     / "raw" / "gi_place.json").read_text())
    res["naive_curve_read"] = {
        "what": ("what you would conclude by reading the measured RMS off the "
                 "parent's conditional along-track curve INSTEAD of running it"),
        "parent_conditional_spec": {"sigma_50_m": BAR_HALF_M,
                                    "sigma_0_m": BAR_BREAKEVEN_M},
        "warning": ("retraction class RMS-PLACED-ON-A-NOISE-CURVE: this read "
                    "over-predicted damage 5.7x upstream. The measured rows "
                    "above are the operative numbers."),
    }
    res["error_structure"] = {
        arm: {"rms_m": r4(np.sqrt(np.mean(p ** 2))),
              "mean_abs_m": r4(np.mean(np.abs(p))),
              "rms_over_mae": r4(np.sqrt(np.mean(p ** 2)) / np.mean(np.abs(p))),
              "gaussian_ref": 1.2533,
              "bias_m": r4(p.mean()),
              "shrinkage_alpha": r4(np.polyfit(y, y + p, 1)[0]),
              "family": ("SHRINK-like (biased regressor)"
                         if np.polyfit(y, y + p, 1)[0] < 0.97 else "near-unbiased"),
              "p95_abs_m": r4(np.percentile(np.abs(p), 95))}
        for arm, p in pools.items()}

    (raw / "eg_place.json").write_text(json.dumps(res, indent=1))
    print(f"[place] -> {raw/'eg_place.json'}", flush=True)


if __name__ == "__main__":
    main()
