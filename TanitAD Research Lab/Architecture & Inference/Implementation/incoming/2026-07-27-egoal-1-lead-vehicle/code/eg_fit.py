#!/usr/bin/env python3
"""E-GOAL-1 S2/S3 -- the leak audit and the out-of-fold re-fit.

Axis: **ALONG-TRACK RMS in metres** of a 2 s displacement prediction, against the
parent stream's pre-registered conditional bars 0.813 m (break-even) / 0.439 m
(half prize), and the same quantity converted to 2 s-mean SPEED error in m/s
(0.406 / 0.219).

Every fitted number is OUT-OF-FOLD and CLIP-DISJOINT (5 folds, seed 0).
Estimator: episode-cluster bootstrap, B = 2000, unit = the clip. Paired form for
every contrast. `overlapping_holdout_se` is never called.

BOTH-DIRECTIONS VALIDATION, mandatory and reported whatever the verdict:
  * `N_SHUF`   lead block permuted ACROSS clips -- must not help.
  * `P_ORACLE` v(t + 2 s) added -- must help enormously, or the instrument
               cannot return CONFIRM and the run is VOID.

Run:  OMP_NUM_THREADS=6 python eg_fit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eg_common import (BAR_BREAKEVEN_M, BAR_BREAKEVEN_MS, BAR_HALF_M,  # noqa: E402
                       BAR_HALF_MS, DENS_COLS, EGO_COLS, HORIZON_S, LEAD_COLS,
                       N_BOOT, REF_HEAD_ALONG_M, REF_HEAD_MS, SEED, XTRA_COLS,
                       assert_no_oracle, ci_paired, ci_single, clip_folds,
                       n_to_separate, r4, sep)

V0_COLS = ["v"]
LEAD_ALL = LEAD_COLS + XTRA_COLS


def fit_predict(Xtr, ytr, Xte, kind: str, seed: int = SEED) -> np.ndarray:
    """Identical hyper-parameters for every arm, so extra columns can only help
    an arm through held-out generalisation."""
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    if kind == "ridge":
        mu = np.nanmean(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), mu, Xtr)
        Xte = np.where(np.isnan(Xte), mu, Xte)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Ztr = np.column_stack([(Xtr - mu) / sd, np.ones(len(Xtr))])
        Zte = np.column_stack([(Xte - mu) / sd, np.ones(len(Xte))])
        lam = np.eye(Ztr.shape[1])
        lam[-1, -1] = 0.0
        w = np.linalg.solve(Ztr.T @ Ztr + lam, Ztr.T @ ytr)
        return Zte @ w
    raise KeyError(kind)


def build_arms(df: pd.DataFrame, rng) -> dict:
    """Feature matrices. EVERY name passes `assert_no_oracle` except the
    explicitly-labelled positive control, which is quoted only as a bound."""
    clips = df["clip"].to_numpy(str)
    uniq = np.unique(clips)
    idx_by = {u: np.flatnonzero(clips == u) for u in uniq}
    order = rng.permutation(len(uniq))
    remap = {u: uniq[order[i]] for i, u in enumerate(uniq)}

    def take(cols):
        assert_no_oracle(cols)
        return df[list(cols)].to_numpy(np.float64)

    ego = take(EGO_COLS)
    v0 = take(V0_COLS)
    lead = take(LEAD_COLS)
    dens = take(DENS_COLS)
    xtra = take(XTRA_COLS)
    leadall = take(LEAD_ALL)

    shuf = np.empty_like(lead)          # permute the lead block ACROSS clips
    for u in uniq:
        src, dst = idx_by[remap[u]], idx_by[u]
        shuf[dst] = lead[np.resize(src, len(dst))]

    # ⛔ ORACLES -- future speed. Present ONLY to prove the instrument can return
    # CONFIRM; never quoted as a capability (retraction class BOUND-QUOTED-AS-
    # CAPABILITY). They deliberately bypass assert_no_oracle.
    oracle = np.column_stack([ego, df["v_fut_2s"].to_numpy(np.float64)])
    # the STRONG positive control. ⚠️ Its FIRST form -- the raw future speed
    # profile -- turned out to be a weak control, and the reason is a MODEL-CLASS
    # artifact, not an information statement: y_long is the INTEGRAL of speed, an
    # additive function of 5 continuous inputs, which a 31-leaf tree ensemble
    # approximates badly. Handing the same information to the model in the form
    # it can actually use (the trapezoidal integral as ONE monotone column) is
    # what makes this a control rather than another measurement. Both forms are
    # reported, because the gap between them is itself the evidence that the
    # weak form's number is about the regressor and not about the data.
    oracle_strong = np.column_stack(
        [ego, df["v_fut_int"].to_numpy(np.float64)]
        + [df[f"v_fut_{h}"].to_numpy(np.float64)
           for h in ("0p5", "1p0", "1p5", "2s")])
    oracle_profile = np.column_stack(
        [ego] + [df[f"v_fut_{h}"].to_numpy(np.float64)
                 for h in ("0p5", "1p0", "1p5", "2s")])

    return {
        "E0_v0": v0,
        "E1_ego": ego,
        "E1_L": np.column_stack([ego, lead]),
        "E1_L_D": np.column_stack([ego, lead, dens]),
        "E1_L_X": np.column_stack([ego, leadall]),
        "E1_L_X_D": np.column_stack([ego, leadall, dens]),
        "L_only": lead,
        "N_SHUF": np.column_stack([ego, shuf]),
        "P_ORACLE": oracle,
        "P_ORACLE_STRONG": oracle_strong,
        "P_ORACLE_PROFILE": oracle_profile,
    }


def oof(df, y, arms, kind, k=5, seed=SEED) -> dict:
    pred = {a: np.full(len(y), np.nan) for a in arms}
    clips = df["clip"].to_numpy(str)
    for f, (tr, te) in enumerate(clip_folds(clips, k=k, seed=seed)):
        for a, X in arms.items():
            pred[a][te] = fit_predict(X[tr], y[tr], X[te], kind)
        print(f"    fold {f+1}/{k} done", flush=True)
    return pred


def arm_block(err: np.ndarray, eid, label: str, y=None) -> dict:
    """Along-track RMS in metres + the same error as a 2 s-mean SPEED error."""
    se = err ** 2
    rms = ci_single(se, eid, reduce="rms")
    mae = ci_single(np.abs(err), eid, reduce="mean")
    p = float(np.sqrt(np.mean(se)))
    return {
        "arm": label,
        "along_rms_m": rms,
        "along_mae_m": mae,
        "speed_err_ms": {"mean": r4(p / HORIZON_S),
                         "lo": r4(rms["lo"] / HORIZON_S),
                         "hi": r4(rms["hi"] / HORIZON_S),
                         "definition": "along_rms_m / 2 s (parent's conversion)"},
        "vs_bars": {
            "breakeven_0p813_m": ("CLEARS" if p < BAR_BREAKEVEN_M else "MISSES"),
            "halfprize_0p439_m": ("CLEARS" if p < BAR_HALF_M else "MISSES"),
            "ratio_to_breakeven": r4(p / BAR_BREAKEVEN_M),
            "ratio_to_halfprize": r4(p / BAR_HALF_M),
            "ci_upper_below_breakeven": bool(rms["hi"] < BAR_BREAKEVEN_M)},
        "error_structure": {
            "rms_over_mean_abs": r4(p / float(np.mean(np.abs(err)))),
            "gaussian_reference": 1.2533,
            "note": ("RMS/MAE above the Gaussian 1.2533 means heavy-tailed error; "
                     "the parent measured that placing such an RMS on an "
                     "isotropic-noise curve OVER-PREDICTS damage 5.7x "
                     "(class RMS-PLACED-ON-A-NOISE-CURVE)"),
            "bias_m": r4(float(np.mean(err))),
            # a > 1 slope means the prediction over-disperses; a < 1 slope is the
            # SHRINK family the parent measured to be the STRICTEST (sigma0 0.721
            # vs 0.955 m) -- an L2-trained regressor lands here by construction
            "shrinkage_alpha_vs_target": (
                None if y is None
                else r4(float(np.polyfit(y, y + err, 1)[0]))),
            "p95_abs_err_m": r4(float(np.percentile(np.abs(err), 95)))},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(Path(__file__).resolve().parent.parent
                                         / "raw"))
    ap.add_argument("--model", default="gbm", choices=["gbm", "ridge"])
    a = ap.parse_args(argv)
    raw = Path(a.raw)

    df = pd.read_parquet(raw / "eg_windows.parquet")
    n_all = len(df)

    # ⛔ the positive control's ORACLE column: v at t + HORIZON_S. Built on the
    # FULL clip grid by interpolation on `t_s` (not a row shift), so a clip with
    # gaps cannot silently produce a mis-shifted "future" that is really a
    # present value -- that would make the control look weak and void the run.
    horizons = {"0p5": 0.5, "1p0": 1.0, "1p5": 1.5, "2s": HORIZON_S}
    futs = {k: np.full(len(df), np.nan) for k in horizons}
    for _, idx in df.groupby("clip").indices.items():
        idx = np.sort(idx)
        t = df["t_s"].to_numpy()[idx]
        v = df["v"].to_numpy()[idx]
        for k, h in horizons.items():
            futs[k][idx] = np.interp(t + h, t, v)
    for k in horizons:
        df[f"v_fut_{k}"] = futs[k]
    # trapezoidal integral of speed over [t, t+2] at 0.5 s spacing -- i.e. the
    # distance the car is ABOUT to travel. One monotone column, so the tree
    # ensemble can actually use it.
    df["v_fut_int"] = 0.5 * (0.5 * df["v"].to_numpy(np.float64)
                             + futs["0p5"] + futs["1p0"] + futs["1p5"]
                             + 0.5 * futs["2s"])

    keep = (np.isfinite(df["y_long"]) & np.isfinite(df["v"])
            & (df["obst_cov"] > 0))
    df = df[keep].reset_index(drop=True)

    y = df["y_long"].to_numpy(np.float64)
    eid = df["clip"].to_numpy(str)
    rng = np.random.default_rng(SEED)
    arms = build_arms(df, rng)

    n_clips = int(df["clip"].nunique())
    print(f"[fit] {len(df)} windows / {n_clips} clips "
          f"({n_all - len(df)} rows dropped: no target, no ego, or no obstacle "
          f"coverage)", flush=True)
    print(f"[fit] model={a.model}, 5 clip-disjoint folds", flush=True)
    pred = oof(df, y, arms, a.model)

    # CV reference: constant velocity, NO FIT -- the parent's H1_cv analogue
    pred["CV"] = HORIZON_S * df["v"].to_numpy(np.float64)

    err = {k: pred[k] - y for k in pred}
    out = {
        "what": "E-GOAL-1 S3 -- OOF along-track RMS with and without lead state",
        "axis": "2 s ALONG-TRACK displacement error, metres",
        "model": a.model,
        "folds": {"k": 5, "unit": "clip", "seed": SEED,
                  "disjoint": "no clip is in both the fit and the score"},
        "estimator": {"name": "episode_cluster_bootstrap / paired form",
                      "n_boot": N_BOOT, "unit": "clip",
                      "overlapping_holdout_se": "NEVER CALLED"},
        "bars": {"breakeven_along_rms_m": BAR_BREAKEVEN_M,
                 "halfprize_along_rms_m": BAR_HALF_M,
                 "breakeven_speed_ms": BAR_BREAKEVEN_MS,
                 "halfprize_speed_ms": BAR_HALF_MS,
                 "best_head_to_date_along_rms_m": REF_HEAD_ALONG_M,
                 "best_head_to_date_speed_ms": REF_HEAD_MS,
                 "source": "GOAL_INPUT.md S5 conditional spec (parent stream)"},
        "n_windows": int(len(df)), "n_clips": n_clips,
        "target": {"mean_m": r4(y.mean()), "sd_m": r4(y.std())},
        "arms": {}, "paired": {},
    }

    order = ["CV", "E0_v0", "E1_ego", "E1_L", "E1_L_D", "E1_L_X", "E1_L_X_D",
             "L_only", "N_SHUF", "P_ORACLE", "P_ORACLE_PROFILE",
             "P_ORACLE_STRONG"]
    for k in order:
        out["arms"][k] = arm_block(err[k], eid, k, y=y)
        b = out["arms"][k]
        print(f"  {k:10s} along_rms={b['along_rms_m']['mean']:.4f} m "
              f"[{b['along_rms_m']['lo']:.4f},{b['along_rms_m']['hi']:.4f}] "
              f"speed={b['speed_err_ms']['mean']:.4f} m/s "
              f"{b['vs_bars']['breakeven_0p813_m']}", flush=True)

    # ---- paired contrasts, the decision statistics ------------------------ #
    # ⚠️ TWO AXES, and the reason is measured, not stylistic. The BAR's axis is
    # the RMS, so the RMS is primary. But the RMS reducer is tail-dominated: a
    # handful of clips carry most of the squared error, so a clip bootstrap on
    # it is WIDE. The MAE on the same per-window errors is reported beside every
    # contrast as the higher-power secondary read. A claim is only made when the
    # two agree; where they disagree, that is stated.
    def pair(a_, b_, why):
        se_a, se_b = err[a_] ** 2, err[b_] ** 2
        ci = ci_paired(se_a, se_b, eid, reduce="rms")
        ci["what"] = f"along RMS({a_}) - along RMS({b_}), metres"
        ci["why"] = why
        ci["rel_reduction"] = r4(-ci["delta"] / float(np.sqrt(np.mean(se_a))))
        ci["n_clips_to_separate"] = (
            None if ci["separated"]
            else round(n_to_separate(ci["delta"], ci["ci95"],
                                     ci["n_episodes"]), 1))
        mae = ci_paired(np.abs(err[a_]), np.abs(err[b_]), eid, reduce="mean")
        mae["rel_reduction"] = r4(-mae["delta"] / float(np.mean(np.abs(err[a_]))))
        mae["n_clips_to_separate"] = (
            None if mae["separated"]
            else round(n_to_separate(mae["delta"], mae["ci95"],
                                     mae["n_episodes"]), 1))
        ci["secondary_MAE"] = mae
        return ci

    out["paired"] = {
        "E1_ego__vs__E1_L": pair(
            "E1_ego", "E1_L",
            "⭐ THE PRE-REGISTERED TEST: does lead-vehicle state move along-track RMS?"),
        "E1_ego__vs__E1_L_X": pair(
            "E1_ego", "E1_L_X", "best-effort lead feature set"),
        "E1_ego__vs__E1_L_X_D": pair(
            "E1_ego", "E1_L_X_D", "lead + derived + traffic density"),
        "E1_ego__vs__N_SHUF": pair(
            "E1_ego", "N_SHUF",
            "⛔ NEGATIVE CONTROL: a shuffled lead block must NOT help"),
        "E1_ego__vs__P_ORACLE": pair(
            "E1_ego", "P_ORACLE",
            "⛔ ORACLE (not a capability): future speed at t+2 s only"),
        "E1_ego__vs__P_ORACLE_PROFILE": pair(
            "E1_ego", "P_ORACLE_PROFILE",
            "⛔ ORACLE: the raw future speed profile. Weak in a tree ensemble "
            "because y_long is an ADDITIVE function of it -- a model-class "
            "artifact, reported to show why the next row is the real control."),
        "E1_ego__vs__P_ORACLE_STRONG": pair(
            "E1_ego", "P_ORACLE_STRONG",
            "⛔ POSITIVE CONTROL (ORACLE, not a capability): the whole future "
            "speed profile. MUST be separated-better, or the RMS axis has no "
            "power at this n and no non-separation on it means anything."),
        "E0_v0__vs__E1_ego": pair(
            "E0_v0", "E1_ego",
            "⭐ what EGO HISTORY buys over the parent head's v0-only ego content"),
        "CV__vs__E1_ego": pair(
            "CV", "E1_ego", "learned ego vs constant velocity"),
    }
    for k, v in out["paired"].items():
        s = v["secondary_MAE"]
        print(f"  {k:30s} RMS d={v['delta']:+.4f} "
              f"[{v['lo']:+.4f},{v['hi']:+.4f}] rel={100*v['rel_reduction']:+.2f}% "
              f"sep={v['separated']} | MAE d={s['delta']:+.4f} "
              f"[{s['lo']:+.4f},{s['hi']:+.4f}] rel={100*s['rel_reduction']:+.2f}% "
              f"sep={s['separated']}", flush=True)

    # ---- C19: the conditional result and its firing rate ------------------ #
    m = df["lead_present"].to_numpy() > 0
    rate = float(m.mean())
    sub = ci_paired(err["E1_ego"][m] ** 2, err["E1_L"][m] ** 2, eid[m],
                    reduce="rms")
    sub_mae = ci_paired(np.abs(err["E1_ego"][m]), np.abs(err["E1_L"][m]), eid[m],
                        reduce="mean")
    whole = out["paired"]["E1_ego__vs__E1_L"]
    out["conditional_lead_present"] = {
        "firing_rate": r4(rate),
        "n_windows": int(m.sum()),
        "n_clips": int(len(np.unique(eid[m]))),
        "paired_along_rms_delta_m": sub,
        "paired_along_mae_delta_m": sub_mae,
        "whole_set_policy_value_m": whole["delta"],
        "C19": ("a conditional result is reported with its firing rate AND its "
                "whole-set policy value; a stratum win is not a deployable win"),
    }
    print(f"  lead_present subgroup: rate={rate:.4f} "
          f"d={sub['delta']:+.4f} [{sub['lo']:+.4f},{sub['hi']:+.4f}] "
          f"sep={sub['separated']} | whole-set {whole['delta']:+.4f}", flush=True)

    # ---- tail diagnostics: WHY the RMS axis is wide ----------------------- #
    se = err["E1_ego"] ** 2
    by_clip = pd.Series(se).groupby(eid).sum().sort_values(ascending=False)
    share = by_clip.cumsum() / by_clip.sum()
    out["tail_diagnostics"] = {
        "what": ("the along-track RMS is dominated by a few clips, which is why "
                 "a clip-cluster bootstrap on it is wide; the MAE read is "
                 "reported beside every contrast for power"),
        "arm": "E1_ego",
        "frac_of_squared_error_in_top_1pct_clips": r4(
            float(share.iloc[max(0, int(0.01 * len(share)) - 1)])),
        "frac_of_squared_error_in_top_5pct_clips": r4(
            float(share.iloc[max(0, int(0.05 * len(share)) - 1)])),
        "frac_of_squared_error_in_top_10pct_clips": r4(
            float(share.iloc[max(0, int(0.10 * len(share)) - 1)])),
        "per_window_rms_over_mae": r4(
            float(np.sqrt(se.mean()) / np.mean(np.abs(err["E1_ego"])))),
    }

    # ---- both-directions validation verdict ------------------------------- #
    pc = out["paired"]["E1_ego__vs__P_ORACLE_STRONG"]
    pw = out["paired"]["E1_ego__vs__P_ORACLE"]
    nc = out["paired"]["E1_ego__vs__N_SHUF"]
    out["controls"] = {
        "positive_P_ORACLE_STRONG": {
            "required": "must be separated-better on the RMS axis",
            "delta_m": pc["delta"], "separated": pc["separated"],
            "mae_delta_m": pc["secondary_MAE"]["delta"],
            "mae_separated": pc["secondary_MAE"]["separated"],
            "passes": bool(pc["separated"] and pc["delta"] > 0.05)},
        "positive_P_ORACLE_weak": {
            "note": ("future speed at t+2 s ALONE; informative about how much of "
                     "the residual is future ego intent, not a control"),
            "delta_m": pw["delta"], "separated": pw["separated"],
            "mae_delta_m": pw["secondary_MAE"]["delta"],
            "mae_separated": pw["secondary_MAE"]["separated"]},
        "negative_N_SHUF": {
            "required": "must NOT be separated-better",
            "delta_m": nc["delta"], "separated": nc["separated"],
            "direction": ("shuffle HURT (expected)" if nc["delta"] < 0
                          else "shuffle HELPED (pipeline leaks)"),
            "passes": bool(not (nc["separated"] and nc["delta"] > 0))},
    }
    out["controls"]["run_valid"] = bool(
        out["controls"]["positive_P_ORACLE_STRONG"]["passes"]
        and out["controls"]["negative_N_SHUF"]["passes"])

    (raw / f"eg_fit_{a.model}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(raw / f"eg_oof_pred_{a.model}.npz",
                        y=y, clip=eid, lead_present=m,
                        **{f"pred_{k}": pred[k] for k in order})
    print(f"[fit] controls valid = {out['controls']['run_valid']}", flush=True)
    print(f"[fit] -> {raw / f'eg_fit_{a.model}.json'}", flush=True)


if __name__ == "__main__":
    main()
