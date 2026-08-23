#!/usr/bin/env python3
"""E-GOAL-4 S1 -- ⭐⭐ THE DECISIVE MEASUREMENT: does the +46.3 % survive JOINT
TRAINING?

E-GOAL-3 measured a goal injected into a FROZEN fan through a FIXED rule
(`argmin_c |candidate - goal*FRAC|`). This script replaces that rule with a
TRAINED SELECTOR that consumes the goal as an INPUT FEATURE, and measures the
pick it actually makes.

  rows      one per (window, candidate) -- 13,198 x 256 = 3,378,688
  label     each candidate's own realised `ade_0_2s` (`eg_place.ade`, IMPORTED)
  pick      argmin_c score(w, c)
  scored    ade(fan[w, argmin], gt[w]) -- the IDENTICAL function every prior
            stage used, IMPORTED, not re-implemented
  folds     5 EPISODE-DISJOINT, `eg_common.clip_folds(epi.astype(str), 5, 0)` --
            gate G-2 proved these are BIT-IDENTICAL to the goal head's own

⭐ `d_rule` -- the fixed rule's own statistic -- IS ONE OF THE FED COLUMNS, so
the learner CAN express `argmin d_rule` exactly. A shortfall therefore cannot be
blamed on model class (`CONTROL-WEAK-BY-MODEL-CLASS` is closed by construction).

⛔ C30 the background is NAMED, drawn ONCE from default_rng(5000+0), and used
   BIT-IDENTICALLY by every arm -- trained and fixed alike.
⛔ C31 the negative control (`S_goal_shuf`, a REAL goal from the WRONG episode)
   is re-run at the ACTUAL n = 600, and the verdict rests on the DIRECT paired
   contrast `S_goal` vs `S_nogoal`, not on "does it separate".
⛔ C33 nothing here is a proxy: every arm's realised ADE is measured directly.

Run:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 python e4_select.py \
          --fan <fan600.pt> --feat <val.npz> --preds <preds.npz> \
          --cross parent_resampled --full
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_common import (ALL_COLS, DEPLOY_REF, E3_FIXED, EG1, F_ANS, F_CTX,  # noqa: E402
                       F_GOAL, HGB_KW, SEED, STREAM, ade, build_goal,
                       build_static, ci_paired, ci_single, clip_folds,
                       cross_background, goal_reference, labels, load_all,
                       pick_nearest_to, r4, realise, sep, shuffle_across_episodes)

N_ANS, N_CTX, N_GOAL = len(F_ANS), len(F_CTX), len(F_GOAL)
IX_STATIC = list(range(N_ANS + N_CTX))
IX_GOAL = list(range(N_ANS + N_CTX, N_ANS + N_CTX + N_GOAL))


def fit_score(Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(**HGB_KW)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--cross", choices=["parent_resampled", "sel"],
                    default="parent_resampled")
    ap.add_argument("--full", action="store_true",
                    help="the whole arm set (primary background)")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    tag = a.cross

    D = load_all(a.fan, a.feat, a.preds)
    fan, gt, eid = D["fan"], D["gt"], D["eid"]
    W, C = D["logits"].shape
    g_true = gt[:, -1, :]
    sel = D["sel"]
    epi = np.asarray(D["preds"]["epi"])
    epis = epi.astype(str)

    a0 = ade(fan[np.arange(W), sel], gt)
    r_goal = realise(g_true, fan, gt)
    lab = labels(D)                                       # [W, C]
    headroom = float(a0.mean() - r_goal.mean())
    assert abs(a0.mean() - DEPLOY_REF["a0"]) < 5e-4, "G-0"
    assert abs(headroom - DEPLOY_REF["headroom"]) < 5e-4, "G-0"

    R = {"_stream": "2026-07-28-egoal-4-joint", "_stage": "S1 joint selector",
         "_tag": tag, "_cross_mode": a.cross,
         "_estimator": ("paired episode-cluster bootstrap, taniteval/ci.py, "
                        "B=2000, unit = the val episode. "
                        "overlapping_holdout_se NEVER called."),
         "_coupling": ("a per-candidate LEARNED SELECTOR over the frozen "
                       "256-anchor REF-C-XL fan, trained on each candidate's "
                       "realised ade_0_2s, with the goal head's prediction as "
                       "an INPUT FEATURE, evaluated by the pick it makes"),
         "_regressor": {k: (v if not isinstance(v, np.generic) else float(v))
                        for k, v in HGB_KW.items()},
         "_cols": {"F_ans": F_ANS, "F_ctx": F_CTX, "F_goal": F_GOAL},
         "deployment": {"n_windows": int(W), "n_candidates": int(C),
                        "n_rows": int(W * C),
                        "n_episode_clusters": int(len(np.unique(eid))),
                        "a0_as_trained": r4(a0.mean()),
                        "r_goal2s_true_goal": r4(r_goal.mean()),
                        "oracle_in_fan": r4(lab.min(1).mean()),
                        "headroom": r4(headroom)},
         "arms": {}, "contrasts": {}}

    # ------------------------------------------------------- the background --
    cross1, pool_n = cross_background(a.cross, g_true, fan, sel, 0, W)
    R["cross_track"] = {
        "mode": a.cross,
        "seeds_used_by_TRAINED_arms": 1,
        "seed": 5000,
        "cross_mae_m": r4(float(np.abs(cross1 - g_true[:, 1]).mean())),
        "pool_n": int(pool_n),
        "future_blind": bool(a.cross == "sel"),
        "_note": ("drawn ONCE and used BIT-IDENTICALLY by every arm, trained "
                  "and fixed alike (C30). A trained arm cannot average over 16 "
                  "draws without 16 refits per fold; gate G-4 shows the single "
                  "draw is representative.")}

    # ------------------------------------------------------------- the goal --
    g_head = D["preds"]["T_OOF|H_v0_ax"].astype(np.float64)   # v + ax_fd, OOF
    g_ego = D["preds"]["T_OOF|H_ego"].astype(np.float64)
    y_true = D["preds"]["y"].astype(np.float64)
    assert np.abs(y_true - g_true[:, 0]).max() < 1e-4, "target != fan gt along"
    rng = np.random.default_rng(20260728)
    g_shuf = shuffle_across_episodes(g_head, epi, np.random.default_rng(4242))
    cr_shuf = shuffle_across_episodes(cross1, epi, np.random.default_rng(4243))
    g_cv = D["preds"]["T_OOF|CV_head"].astype(np.float64)

    GOALS = {
        "head":   np.stack([g_head, cross1], 1),
        #: TRUE along + the SAME background cross -- matches `FIXED_oracle`
        #: exactly, so the two are the same goal through two decision rules.
        "oracle": np.stack([y_true, cross1], 1),
        #: the full true 2-D endpoint -- the strongest bound, never a capability.
        "oracle2d": np.stack([y_true, g_true[:, 1]], 1),
        "cv":     np.stack([g_cv, cross1], 1),
        "shuf":   np.stack([g_shuf, cr_shuf], 1),
        "ego":    np.stack([g_ego, cross1], 1),
        #: ⭐ AMENDMENT A3 -- the 2x2 that decomposes the goal into its two axes.
        #: Registered because the S0 audit showed `parent_resampled`'s CROSS is
        #: future-derived BY CONSTRUCTION: without this split, an oracle-cross
        #: contribution would be indistinguishable from the along-track lever
        #: E-GOAL-2/3 identified.
        "alongonly": np.stack([g_head, cr_shuf], 1),   # real along, fake cross
        "crossonly": np.stack([g_shuf, cross1], 1),    # fake along, real cross
    }

    # ===================================================== the FIXED arms ====
    # G-1 / G-4: the fixed rule, 1 seed and 16, against E-GOAL-3's raw JSON.
    def fixed(along, mode, nseed):
        acc = np.zeros(W)
        for s in range(nseed):
            cr, _ = cross_background(mode, g_true, fan, sel, s, W)
            acc += realise(np.stack([along, cr], 1), fan, gt)
        return acc / nseed

    realised = {}
    picks = {}
    ties = {}

    def score_arm(rr, name, extra=None):
        pair = ci_paired(rr, a0, eid)
        rec = float((a0.mean() - rr.mean()) / headroom)
        out = {"realised_ade_0_2s": ci_single(rr, eid),
               "recovery_of_headroom": r4(rec),
               "paired_vs_as_trained": pair, "separated": sep(pair),
               "verdict": ("BETTER" if (sep(pair) and pair["delta"] < 0)
                           else "WORSE" if (sep(pair) and pair["delta"] > 0)
                           else "not separated")}
        if extra:
            out.update(extra)
        realised[name] = rr
        R["arms"][name] = out
        print(f"  {name:22s} realised={rr.mean():.4f} rec={100*rec:+7.2f}% "
              f"d={pair['delta']:+.4f} [{pair['lo']:+.4f},{pair['hi']:+.4f}] "
              f"{out['verdict']}", flush=True)
        return out

    print(f"[e4] background={a.cross}  a0={a0.mean():.4f} "
          f"headroom={headroom:.4f}", flush=True)
    print("[e4] --- FIXED RULE (E-GOAL-3's rule, same goal, same background) ---",
          flush=True)
    for nm, along in (("FIXED_goal", g_head), ("FIXED_goal_ego", g_ego),
                      ("FIXED_oracle", y_true), ("FIXED_cv", g_cv),
                      ("FIXED_shuf", g_shuf)):
        if nm == "FIXED_shuf":
            rr = realise(GOALS["shuf"], fan, gt)
        elif nm == "FIXED_oracle" and a.cross == "parent_resampled":
            rr = fixed(along, a.cross, 1)
        else:
            rr = fixed(along, a.cross, 1)
        score_arm(rr, nm, {"_seeds": 1, "_rule": "argmin_c |cand - goal*FRAC|"})

    n16 = 16 if a.cross == "parent_resampled" else 1
    g1 = {}
    for nm, along, ref in (("FIXED_goal", g_head, "H_v0_ax"),
                           ("FIXED_goal_ego", g_ego, "H_ego"),
                           ("FIXED_cv", g_cv, "CV_head"),
                           ("FIXED_oracle", y_true, "P_ORACLE_TRUE")):
        rr16 = fixed(along, a.cross, n16)
        rec16 = float((a0.mean() - rr16.mean()) / headroom)
        e3 = E3_FIXED[a.cross][ref]
        g1[nm] = {"rec_at_{}_seeds".format(n16): r4(rec16),
                  "rec_at_1_seed": R["arms"][nm]["recovery_of_headroom"],
                  "EGOAL_3_raw_json": e3,
                  "dev_vs_EGOAL_3_recovery_points": round(100 * (rec16 - e3), 3),
                  "dev_1seed_vs_16seed_points": round(
                      100 * (R["arms"][nm]["recovery_of_headroom"] - rec16), 3)}
        print(f"  [G-1/G-4] {nm:15s} {n16}-seed {100*rec16:+.2f}% vs E-GOAL-3 "
              f"{100*e3:+.2f}% (dev {g1[nm]['dev_vs_EGOAL_3_recovery_points']:+.3f} "
              f"pts) ; 1-seed dev {g1[nm]['dev_1seed_vs_16seed_points']:+.3f} pts",
              flush=True)
    mx1 = max(abs(v["dev_vs_EGOAL_3_recovery_points"]) for v in g1.values())
    mx4 = max(abs(v["dev_1seed_vs_16seed_points"]) for v in g1.values())
    R["G_1_fixed_rule_reproduces_EGOAL_3"] = {
        "_source": "e3_place_n600_%s.json -> arms[T_OOF|*].recovery_of_headroom"
                   % a.cross,
        "cells": g1, "max_abs_dev_recovery_points": mx1,
        "passes": bool(mx1 < 1.0)}
    R["G_4_one_seed_is_representative"] = {
        "max_abs_dev_recovery_points": mx4, "tolerance": 1.5,
        "passes": bool(mx4 < 1.5)}
    if mx1 >= 1.0:
        raise RuntimeError(f"G-1 FAILED: {mx1:.3f} recovery points from E-GOAL-3")

    # ================================================ the TRAINED selector ===
    print("[e4] building the candidate matrix ...", flush=True)
    S = build_static(D).reshape(W * C, -1)
    G = {k: build_goal(D, v).reshape(W * C, -1) for k, v in GOALS.items()}
    Y = lab.reshape(-1).astype(np.float32)
    Yz = (lab - lab.mean(1, keepdims=True)).reshape(-1).astype(np.float32)
    gc.collect()
    print(f"[e4] static {S.shape} {S.nbytes/1e6:.0f} MB  "
          f"goal-variants {len(G)}  ({time.time()-t0:.0f}s)", flush=True)

    folds = list(clip_folds(epis, k=5, seed=SEED))
    win_te = [np.flatnonzero(te) for _, te in folds]

    def rows_of(win_idx):
        return (win_idx[:, None] * C + np.arange(C)).reshape(-1)

    def realise_from_score(sc_2d):
        """argmin over candidates; ties -> first index (arbitrary w.r.t.
        quality, which is the honest reading of `the model cannot tell these
        apart`). The tie count is reported."""
        idx = sc_2d.argmin(1)
        mn = sc_2d.min(1, keepdims=True)
        ntie = (sc_2d <= mn + 1e-12).sum(1)
        return ade(fan[np.arange(W), idx], gt), idx, ntie

    def run_trained(name, goal_key, use_static=True, target="raw",
                    extra_col=None, insample=False, coadapt_goal=None):
        sc = np.full((W, C), np.nan, np.float64)
        tgt = Y if target == "raw" else Yz
        if coadapt_goal is None:
            blocks = []
            if use_static:
                blocks.append(S)
            if goal_key is not None:
                blocks.append(G[goal_key])
            if extra_col is not None:
                blocks.append(extra_col)
            X = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
        else:
            X = None
        if insample:
            p = fit_score(X, tgt, X)
            sc = p.reshape(W, C)
        elif coadapt_goal is not None:
            for f, (_, te) in enumerate(folds):
                tr_w = np.flatnonzero(~te)
                te_w = np.flatnonzero(te)
                Xc = np.hstack([S, coadapt_goal[f]]) if use_static \
                    else coadapt_goal[f]
                rtr, rte = rows_of(tr_w), rows_of(te_w)
                sc[te_w] = fit_score(Xc[rtr], tgt[rtr],
                                     Xc[rte]).reshape(len(te_w), C)
                del Xc
                gc.collect()
        else:
            for f, (_, te) in enumerate(folds):
                tr_w, te_w = np.flatnonzero(~te), np.flatnonzero(te)
                rtr, rte = rows_of(tr_w), rows_of(te_w)
                sc[te_w] = fit_score(X[rtr], tgt[rtr],
                                     X[rte]).reshape(len(te_w), C)
        assert np.isfinite(sc).all(), f"{name}: missing score"
        rr, idx, ntie = realise_from_score(sc)
        picks[name] = idx
        ties[name] = ntie
        ex = {"_features": (("F_ans+F_ctx" if use_static else "")
                            + ("+F_goal(%s)" % goal_key if goal_key else "")
                            + ("+head_deg" if extra_col is not None else "")),
              "_target": ("per-candidate ade_0_2s" if target == "raw"
                          else "per-candidate ade_0_2s CENTRED WITHIN WINDOW"),
              "_config": ("IN-SAMPLE (fit and scored on the same rows)"
                          if insample else "5-fold episode-disjoint OUT-OF-FOLD"),
              "ties_at_min_mean": r4(float(ntie.mean())),
              "ties_at_min_p95": int(np.percentile(ntie, 95)),
              "frac_windows_with_ties": r4(float((ntie > 1).mean())),
              "pick_equals_as_trained_frac": r4(float((idx == sel).mean()))}
        score_arm(rr, name, ex)
        del X
        gc.collect()

    print("[e4] --- TRAINED SELECTOR ---", flush=True)
    run_trained("S_nogoal", None)
    run_trained("S_goal", "head")
    run_trained("S_goal_shuf", "shuf")
    run_trained("S_goal_oracle", "oracle")
    run_trained("S_goal_oracle2d", "oracle2d")
    run_trained("S_goal_alongonly", "alongonly")     # ⭐ A3
    run_trained("S_goal_crossonly", "crossonly")     # ⭐ A3
    if a.full:
        run_trained("S_goalonly", "head", use_static=False)
        run_trained("S_goal_cv", "cv")
        run_trained("S_goal_z", "head", target="z")
        run_trained("S_nogoal_z", None, target="z")
        run_trained("S_goal_ego", "ego")
        # ⛔ C23 POWER CONTROL: a real FUTURE field, fed deliberately.
        hd = np.repeat(D["head_deg"][:, None], C, 1).reshape(-1, 1).astype(np.float32)
        run_trained("S_LEAK", "head", extra_col=hd)
        run_trained("S_goal_INSAMPLE", "head", insample=True)
        run_trained("S_nogoal_INSAMPLE", None, insample=True)

        # ---------- G-6 + the CO-ADAPTATION arm: refit the goal head per fold --
        from eg_fit import fit_predict
        Xh = D["X"][:, [0, 1]]                       # v, ax_fd -- the brief's list
        oof_chk = np.full(W, np.nan)
        coad = []
        for f, (_, te) in enumerate(folds):
            tr_w, te_w = np.flatnonzero(~te), np.flatnonzero(te)
            oof_chk[te_w] = fit_predict(Xh[tr_w], y_true[tr_w], Xh[te_w], "gbm")
            gin = fit_predict(Xh[tr_w], y_true[tr_w], Xh[tr_w], "gbm")
            gg = g_head.copy()
            gg[tr_w] = gin                       # IN-SAMPLE on the training rows
            coad.append(build_goal(D, np.stack([gg, cross1], 1)).reshape(W * C, -1))
        dmax = float(np.abs(oof_chk - g_head).max())
        R["G_6_head_refit_reproduces_EGOAL_3"] = {
            "_what": "my per-fold refit of `v + ax_fd` vs e3_head_preds T_OOF|H_v0_ax",
            "max_abs_delta_m": dmax, "passes": bool(dmax < 1e-9)}
        print(f"  [G-6] head refit max|Δ| = {dmax:.3e}", flush=True)
        run_trained("S_goal_coadapt", "head", coadapt_goal=coad)
        del coad
        gc.collect()

    # ------------------------------------------- the goal-degradation ladder --
    if a.ladder:
        e = g_head - y_true
        R["ladder"] = []
        for k in (0.0, 0.5, 1.5, 2.0, 3.0):
            gk = np.stack([y_true + k * e, cross1], 1)
            G["_k"] = build_goal(D, gk).reshape(W * C, -1)
            run_trained(f"S_goal_k{k}", "_k")
            rrf = realise(gk, fan, gt)
            score_arm(rrf, f"FIXED_goal_k{k}", {"_seeds": 1})
            R["ladder"].append({
                "k": k,
                "along_rms_m": r4(float(np.sqrt(np.mean((k * e) ** 2)))),
                "trained": R["arms"][f"S_goal_k{k}"]["recovery_of_headroom"],
                "trained_sep": R["arms"][f"S_goal_k{k}"]["separated"],
                "fixed": R["arms"][f"FIXED_goal_k{k}"]["recovery_of_headroom"]})
            del G["_k"]
            gc.collect()

    # ================================ ⭐ C31: THE DIRECT PAIRED CONTRASTS ====
    CON = [("S_goal", "S_nogoal", "⭐⭐ THE VERDICT: what the goal buys a TRAINED selector"),
           ("S_goal", "FIXED_goal", "⭐ trained selector vs the FIXED rule, SAME goal"),
           ("S_goal", "S_goal_shuf", "goal vs a REAL goal from the WRONG episode"),
           ("S_nogoal", "S_goal_shuf", "⛔ MUST BE NULL: no goal vs fake goal"),
           ("S_goal_alongonly", "S_nogoal", "⭐ A3: the ALONG axis alone"),
           ("S_goal_crossonly", "S_nogoal", "⭐ A3: the CROSS axis alone"),
           ("S_goal", "S_goal_alongonly", "⭐ A3: what the cross adds on top of along"),
           ("S_goal", "S_goal_crossonly", "⭐ A3: what the along adds on top of cross"),
           ("S_nogoal", "FIXED_cv", "the trained selector vs a CV goal through the rule"),
           ("S_nogoal", "FIXED_goal", "the trained re-scorer vs the goal rule"),
           ("S_goal", "S_goal_oracle", "distance to the trained-with-true-goal bound"),
           ("FIXED_goal", "FIXED_oracle", "the same distance, fixed rule"),
           ("S_goal_INSAMPLE", "S_goal", "⭐ in-sample vs out-of-fold (co-adaptation)"),
           ("S_goal_coadapt", "S_goal", "⭐ head/selector co-adaptation"),
           ("S_goal_z", "S_goal", "the loss (within-window centred target)"),
           ("S_goal_z", "S_nogoal_z", "the verdict contrast under the z loss"),
           ("S_goalonly", "FIXED_goal", "can the learner express the rule at all?"),
           ("S_goal_cv", "S_nogoal", "an information-poor goal, trained"),
           ("S_LEAK", "S_goal", "⛔ C23 POWER: a real FUTURE field, fed deliberately"),
           ("S_goal_ego", "S_goal", "the 10-column head vs `v + ax_fd`")]
    for x, yy, why in CON:
        if x in realised and yy in realised:
            pc = ci_paired(realised[x], realised[yy], eid)
            R["contrasts"][f"{x}__vs__{yy}"] = {
                "_what": why, "paired_delta_ade": pc, "separated": sep(pc),
                "sign": "x_better" if pc["delta"] < 0 else "y_better",
                "recovery_points": round(
                    100 * (R["arms"][yy]["recovery_of_headroom"]
                           - R["arms"][x]["recovery_of_headroom"]), 2)}
            print(f"  d[{x:17s} vs {yy:17s}] {pc['delta']:+.4f} "
                  f"[{pc['lo']:+.4f},{pc['hi']:+.4f}] "
                  f"{'SEP' if sep(pc) else 'null'}   {why[:44]}", flush=True)

    # ------------------------------------------------------- the verdict -----
    if "S_goal" in R["arms"]:
        sg = R["arms"]["S_goal"]
        c_a0 = sg["separated"] and sg["paired_vs_as_trained"]["delta"] < 0
        cvs = R["contrasts"].get("S_goal__vs__S_nogoal", {})
        c_ng = bool(cvs.get("separated") and
                    cvs["paired_delta_ade"]["delta"] < 0)
        rec = sg["recovery_of_headroom"]
        ng = R["arms"]["S_nogoal"]["recovery_of_headroom"]
        pc = cvs.get("paired_delta_ade", {})
        R["VERDICT"] = {
            "_rule": "PRE_REGISTRATION §7",
            "separated_better_than_A0": bool(c_a0),
            "separated_better_than_S_nogoal": c_ng,
            "recovery": rec, "CONFIRM_threshold": 0.37,
            "verdict": ("CONFIRM" if (c_a0 and c_ng and rec >= 0.37)
                        else "PARTIAL" if (c_a0 and c_ng)
                        else "REFUTE"),
            "EGOAL_3_fixed_rule_recovery": E3_FIXED[a.cross]["H_v0_ax"],
            "share_of_the_fixed_rule": r4(rec / E3_FIXED[a.cross]["H_v0_ax"]),
            # ⭐⭐ THE DECOMPOSITION THE HEADLINE MUST CARRY. The fixed rule's
            # +46.3 % is measured against the AS-TRAINED selector. For "does v5
            # need a goal INPUT" the counterfactual is a TRAINED selector
            # WITHOUT one, and that is a different, larger baseline.
            "S_nogoal_recovery": ng,
            "goal_marginal_recovery_points": round(100 * (rec - ng), 2),
            "goal_marginal_paired": pc,
            "goal_marginal_recovery_points_ci": [
                round(-100 * pc.get("hi", np.nan) / headroom, 2),
                round(-100 * pc.get("lo", np.nan) / headroom, 2)] if pc else None,
            "_reading": (
                "the fixed rule's +46.3 % vs A0 CONFLATES two effects: replacing "
                "the as-trained selector with a better decision rule, and adding "
                "goal INFORMATION. `S_nogoal` measures the first; the marginal "
                "above measures the second.")}
        print(f"\n[VERDICT/{a.cross}] {R['VERDICT']['verdict']}  "
              f"rec={100*rec:+.2f}% vs fixed "
              f"{100*E3_FIXED[a.cross]['H_v0_ax']:+.2f}% "
              f"({100*R['VERDICT']['share_of_the_fixed_rule']:.1f} % of it)",
              flush=True)

    R["_wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out) if a.out else STREAM / "raw" / f"e4_select_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    np.savez_compressed(str(out).replace(".json", "_realised.npz"),
                        a0=a0, eid=eid,
                        **{f"r|{k}": v for k, v in realised.items()},
                        **{f"tie|{k}": v for k, v in ties.items()},
                        **{f"pick|{k}": v for k, v in picks.items()})
    print(f"[e4] -> {out}  ({R['_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
