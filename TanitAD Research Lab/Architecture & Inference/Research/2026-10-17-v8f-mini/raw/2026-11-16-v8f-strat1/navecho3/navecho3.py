"""navecho3.py -- D-STRAT1-TACLAT-F1: on the windows labelled TURN_L / TURN_R, is the text nav's cue MISSING at NOW, or PRESENT and unread by the head?

CPU only, zero training, nothing re-dumped.  Imports (never edits) navecho.py, navecho2.py and fam/fam_extra.py.
Tier of every number: T1-kin tick 0 (lateral head argmax at the window's first tick under the T1 feed; nav_in = the vector fed at NOW) vs the v9 label lat_cls.
nav_in layout (v8f_mini_data.py:219-222 v9, :390-394 text; the dump holds batch["nav"][b, -1], v8f_t1_eval.py:370):
  channels 0-2 one-hot FOLLOW / TURN_L / TURN_R (all-zero = no nav row), channel 3 = distance to the next announced turn / 100 m clipped to [0, 3], channel 4 = distance valid.
Ego speed at NOW = dump `v0` (m/s, fam_extra.load_dump).  Label window: lat_cls looks 2 s to 8 s ahead (build_v9_labels.py:802-891).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAVECHO2 = os.environ.get("NAVECHO2_DIR", "F:/TanitAD-scratch/strat1/navecho2")
if NAVECHO2 not in sys.path:
    sys.path.insert(0, NAVECHO2)
import navecho2 as N2  # noqa: E402  (also imports navecho -> puts fam / read / taniteval on sys.path)

N = N2.N
F, L = N.F, N.L
IGNORE, LAT9, NB, SEED = N.IGNORE, N.LAT9, N.NB, N.SEED
TURN_L, TURN_R = 1, 2                    # lat_cls ids; the nav token keys use the same ids for TURN_L / TURN_R (navecho.nav_key)
CORRECT, WRONG, NONE = 0, 1, 2           # state of a feed's cue at NOW for a window labelled TURN_L / TURN_R
STATE_NAMES = ("names the correct turn", "names the other side", "names no turn (FOLLOW or no row)")
SHARE_BAR = 0.5
RULE_ARMS = (("S1", "s0"), ("S1", "s1"), ("N1", "s0"), ("N1", "s1"))      # the committed text arms and seeds
BESIDE_ARMS = (("S1d", "s0"), ("S1d", "s1"), ("S1", "s10"), ("S1", "s11"), ("N1", "s10"), ("N1", "s11"), ("S1d", "s10"), ("S1d", "s11"))
N_PERM_REPEATS = 20
N_SHUFFLES = 10
WIN_LO_S, WIN_HI_S = 2.0, 8.0
TIER = N2.TIER
EST = (f"episode-cluster bootstrap (taniteval.ci stat / paired stat), B={NB}, seed {SEED}, over the episodes holding the labelled turn windows; "
       "'another draw of EPISODES' only (training-run and inference variance are not in it)")


# ------------------------------------------------------------------------------------------------------------------ cue states
def cue_state(key, label):
    """state of a feed's token at NOW for windows labelled TURN_L / TURN_R.  key: navecho.nav_key (0 FOLLOW, 1 TURN_L, 2 TURN_R, 3 no row); label: lat_cls (1 or 2)."""
    key, label = np.asarray(key), np.asarray(label)
    correct = key == label
    wrong = np.isin(key, (TURN_L, TURN_R)) & ~correct
    return np.where(correct, CORRECT, np.where(wrong, WRONG, NONE)).astype(np.int64)


def miss_split(missed, text_state, v9_state):
    """partition of the head's missed turns by the cue at NOW.
    cue_missing: text names no correct turn AND v9 names it.  cue_present: text names the correct turn.  neither: text names no correct turn and v9 does not either."""
    missed, ts, vs = np.asarray(missed, bool), np.asarray(text_state), np.asarray(v9_state)
    return {"cue_missing": missed & (ts != CORRECT) & (vs == CORRECT), "cue_present": missed & (ts == CORRECT),
            "neither": missed & (ts != CORRECT) & (vs != CORRECT)}


def reading(cells, bar=SHARE_BAR):
    """committed rules.  cells: {(family, seed): {"share_cue_missing": x, "share_cue_present": y}} over the RULE_ARMS.
    CUE-MISSING: share_cue_missing >= bar in every cell; CUE-PRESENT-MODEL: share_cue_present >= bar in every cell; else MIXED."""
    if all(c["share_cue_missing"] >= bar for c in cells.values()):
        return "CUE-MISSING"
    if all(c["share_cue_present"] >= bar for c in cells.values()):
        return "CUE-PRESENT-MODEL"
    return "MIXED"


def distance_m(nav):
    """distance to the next announced turn in metres (channel 3 x 100, clipped at 300 by the feed); NaN where channel 4 says invalid"""
    nav = np.asarray(nav, np.float64)
    return np.where(nav[:, 4] == 1, nav[:, 3] * 100.0, np.nan)


def time_to_turn_s(d_m, v0):
    """distance / ego speed at NOW (s); NaN where the speed is below 0.5 m/s or the distance is NaN"""
    d_m, v0 = np.asarray(d_m, np.float64), np.asarray(v0, np.float64)
    return np.where(v0 >= 0.5, d_m / np.where(v0 >= 0.5, v0, 1.0), np.nan)


# ------------------------------------------------------------------------------------------------------------------ bootstrap helpers
def share_ci(CI, flags, den, eid, nb=NB):
    """sum(flags) / sum(den) with the episode-cluster CI.  items = (flag, den) per window."""
    items = [(int(f), int(d)) for f, d in zip(flags, den)]

    def stat(it):
        s = sum(d for _, d in it)
        return sum(f for f, _ in it) / s if s else float("nan")
    r = CI.stat_cluster_bootstrap(items, np.asarray(eid), stat, n_boot=nb, seed=SEED)
    return {"n_num": int(np.sum(flags)), "n_den": int(np.sum(den)), **{k: v for k, v in r.items() if k in ("point", "lo", "hi", "undefined")}}


def mean_pair(CI, a, b, eid, nb=NB):
    """paired mean(a) - mean(b) over the same windows (NaN windows dropped by the caller), episode-cluster CI"""
    r = CI.paired_stat_cluster_bootstrap([float(x) for x in a], [float(x) for x in b], np.asarray(eid), lambda it: float(np.mean(it)) if len(it) else float("nan"), n_boot=nb, seed=SEED)
    return {k: v for k, v in r.items() if k in ("a", "b", "delta", "lo", "hi", "separated", "undefined")}


def q(x, p):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else None


# ------------------------------------------------------------------------------------------------------------------ classifier + permutation importance
def make_clf(params):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**params)


def fit_folds(X, y, eid, params=N2.HGB_PARAMS, k=N2.N_FOLDS, seed=0, K=5):
    """one model per episode-grouped fold, fit on the labelled windows of the other folds (the navecho2 settings)"""
    X, y = np.asarray(X, np.float64), np.asarray(y)
    fold = N2.episode_folds(eid, k, seed)
    lab = (y != IGNORE) & (y >= 0) & (y < K)
    return {f: make_clf(params).fit(X[(fold != f) & lab], y[(fold != f) & lab]) for f in range(k)}, fold


def predict_oof(models, fold, X):
    pred = np.zeros(len(X), np.int64)
    for f, m in models.items():
        te = fold == f
        if te.any():
            pred[te] = m.classes_[m.predict_proba(np.asarray(X, np.float64)[te]).argmax(1)]
    return pred


GROUPS = {"ch0_FOLLOW": [0], "ch1_TURN_L": [1], "ch2_TURN_R": [2], "ch3_distance": [3], "ch4_distance_valid": [4], "token_group_ch0-2": [0, 1, 2], "distance_group_ch3-4": [3, 4]}


def permutation_importance(X, y, eid, mask=None, n=N_PERM_REPEATS, seed=SEED, K=5):
    """out-of-fold, episode-grouped (navecho2 classifier): for each channel group, the channel(s) of the TEST fold are permuted across its windows (n repeats) and the pooled
    out-of-fold macro-F1 is compared with the unpermuted one.  importance = base - permuted (mean over repeats, with min / max)."""
    X = np.asarray(X, np.float64)
    models, fold = fit_folds(X, y, eid)
    ys = y if mask is None else N.masked(y, mask)
    base_pred = predict_oof(models, fold, X)
    base = N.macro_f1(ys, base_pred, K)
    rng = np.random.default_rng(seed)
    out, preds0 = {}, {}
    for g, cols in GROUPS.items():
        vals = []
        for r in range(n):
            Xp = X.copy()
            for f in range(N2.N_FOLDS):
                idx = np.where(fold == f)[0]
                p = rng.permutation(len(idx))
                Xp[np.ix_(idx, cols)] = X[idx[p]][:, cols]
            pr = predict_oof(models, fold, Xp)
            if r == 0:
                preds0[g] = pr
            vals.append(N.macro_f1(ys, pr, K))
        out[g] = {"importance_mean": float(base - np.mean(vals)), "importance_min": float(base - np.max(vals)), "importance_max": float(base - np.min(vals)), "permuted_macro_f1_mean": float(np.mean(vals))}
    return {"base_macro_f1": base, "n_repeats": n, "groups": out}, base_pred, preds0


def shuffled_unweighted(X, y, eid, n=N_SHUFFLES, K=5):
    """nav permuted across windows, out-of-fold, classifier WITHOUT class weights (navecho2's other settings); reported beside, navecho2's weighted K-SHUFFLE band is not re-read"""
    params = {k: v for k, v in N2.HGB_PARAMS.items() if k != "class_weight"}
    rng = np.random.default_rng(SEED)
    X, v = np.asarray(X), []
    for _ in range(n):
        Xp = X[rng.permutation(len(y))]
        models, fold = fit_folds(Xp, y, eid, params)
        v.append(N.macro_f1(y, predict_oof(models, fold, Xp), K))
    return {"n_permutations": n, "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}


# ------------------------------------------------------------------------------------------------------------------ the main analysis
def turn_tables(CI, side, ts, vs, d_t, d_v, v0, eid, nb):
    """Q1 and Q2 for the windows labelled one side"""
    n = len(side)
    t2 = {"n": int(n), "n_episodes": int(len(np.unique(eid)))}
    t2["table_text_x_v9"] = {f"text {STATE_NAMES[a]} | v9 {STATE_NAMES[b]}": int(((ts == a) & (vs == b)).sum()) for a in (CORRECT, WRONG, NONE) for b in (CORRECT, WRONG, NONE)}
    t2["text_correct"] = share_ci(CI, ts == CORRECT, np.ones(n), eid, nb)
    t2["v9_correct"] = share_ci(CI, vs == CORRECT, np.ones(n), eid, nb)
    t2["v9_minus_text_correct_paired"] = mean_pair(CI, (vs == CORRECT).astype(float), (ts == CORRECT).astype(float), eid, nb)
    both = (ts == CORRECT) & (vs == CORRECT)
    tt_t, tt_v = time_to_turn_s(d_t, v0), time_to_turn_s(d_v, v0)
    b = {"n_windows": int(both.sum()), "n_episodes": int(len(np.unique(eid[both]))) if both.any() else 0}
    if both.any():
        dd = d_t[both] - d_v[both]
        okd = np.isfinite(dd)
        b["distance_m"] = {"text_median": q(d_t[both], 50), "text_iqr": [q(d_t[both], 25), q(d_t[both], 75)], "v9_median": q(d_v[both], 50), "v9_iqr": [q(d_v[both], 25), q(d_v[both], 75)],
                           "diff_text_minus_v9_median": q(dd, 50), "diff_iqr": [q(dd, 25), q(dd, 75)], "n_distance_pairs": int(okd.sum()),
                           "n_text_longer": int((dd[okd] > 0).sum()), "n_text_shorter": int((dd[okd] < 0).sum()), "n_equal": int((dd[okd] == 0).sum())}
        if okd.sum() > 1:
            b["distance_m"]["mean_diff_text_minus_v9"] = mean_pair(CI, d_t[both][okd], d_v[both][okd], eid[both][okd], nb)
        b["ego_speed_mps_median"] = q(v0[both], 50)
        for nm, tt in (("text", tt_t), ("v9", tt_v)):
            x = tt[both]
            ok = np.isfinite(x)
            b[f"time_to_turn_s_{nm}"] = {"median": q(x, 50), "iqr": [q(x, 25), q(x, 75)], "n": int(ok.sum()),
                                         "share_below_2s": float((x[ok] < WIN_LO_S).mean()) if ok.any() else None,
                                         "share_in_2_to_8s": float(((x[ok] >= WIN_LO_S) & (x[ok] <= WIN_HI_S)).mean()) if ok.any() else None,
                                         "share_above_8s": float((x[ok] > WIN_HI_S).mean()) if ok.any() else None}
        dtt = tt_t[both] - tt_v[both]
        t2["_dtt"] = None
        b["time_to_turn_diff_text_minus_v9_median_s"] = q(dtt, 50)
    t2["both_name_correct_turn"] = b
    # beside: each feed alone, over ALL its correct windows
    for nm, st, tt, dd_ in (("text", ts, tt_t, d_t), ("v9", vs, tt_v, d_v)):
        m = st == CORRECT
        x = tt[m]
        ok = np.isfinite(x)
        t2[f"{nm}_correct_windows_time_to_turn"] = {"n": int(m.sum()), "n_with_time": int(ok.sum()), "median_s": q(x, 50), "iqr_s": [q(x, 25), q(x, 75)],
                                                   "share_in_2_to_8s": float(((x[ok] >= WIN_LO_S) & (x[ok] <= WIN_HI_S)).mean()) if ok.any() else None,
                                                   "distance_median_m": q(dd_[m], 50), "distance_at_300_clip": int(np.sum(dd_[m] >= 299.99))}
    t2.pop("_dtt", None)
    return t2


def run(dumps, out_dir, nb=NB, log=print):
    from taniteval import ci as CI
    t0 = time.time()
    join = dict(np.load(L.pinned("join_eval141.npz"), allow_pickle=False))
    ref = F.load_dump(F.dump_dir(dumps, "C0_s0"))
    eid, ci_, ws, v0 = ref["eid"], ref["clip_index"], ref["ws"], ref["v0"]
    y = F.v9_rows(join, ci_, ws)["lat"]
    arms = [f"{f}_{s}" for f in N.FAMILIES for s in N.SEEDS] + ["C0_rep"]
    A, prob = {}, []
    for a in arms:
        lg, nav = N.tick0(F.ticks_path(dumps, a), ci_, ws)
        A[a] = {"pred": lg.argmax(1), "nav": nav}
    for a in arms:
        ref_a = "C0_s0" if a.startswith("C0") else "N1_s0"
        if not np.array_equal(A[a]["nav"], A[ref_a]["nav"]):
            prob.append(f"K-FEED: {a} nav_in differs from {ref_a}")
    nav_v9, nav_txt = A["C0_s0"]["nav"], A["N1_s0"]["nav"]
    key_v9, key_txt = N.nav_key(nav_v9), N.nav_key(nav_txt)
    landed = json.load(open(os.path.join(N.FAM, "out", "RESULT_FAM_EXTRA.json"), encoding="utf-8"))
    for a in arms:
        lf = landed["families"]["TACTICAL"][a]["TAC-SEL-LAT"]["macro_f1"]["point"]
        hm = N.macro_f1(y, A[a]["pred"])
        if abs(hm - lf) > 1e-12:
            prob.append(f"K-LANDED: {a} {hm!r} != {lf!r}")
    turn = np.isin(y, (TURN_L, TURN_R))
    res = {"tool": "navecho3.py", "question": "D-STRAT1-TACLAT-F1: cue missing or cue present on labelled turns", "evidence_class": "MEASURED (ours; banked T1-kin dumps, zero training, CPU)",
           "tier": TIER, "estimator": EST, "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))), "n_labelled": int((y != IGNORE).sum()),
           "n_turn_windows": int(turn.sum()), "n_turn_episodes": int(len(np.unique(eid[turn]))),
           "turn_support": {"TURN_L": int((y == TURN_L).sum()), "TURN_R": int((y == TURN_R).sum())}}
    ets = eid[turn]
    ts, vs = cue_state(key_txt[turn], y[turn]), cue_state(key_v9[turn], y[turn])
    d_t, d_v, v0t = distance_m(nav_txt)[turn], distance_m(nav_v9)[turn], v0[turn]
    yt = y[turn]
    # ---- Q1 / Q2
    res["Q1_Q2_by_side"] = {}
    for nm, lab in (("TURN_L", TURN_L), ("TURN_R", TURN_R), ("BOTH_SIDES", None)):
        m = np.ones(len(yt), bool) if lab is None else yt == lab
        res["Q1_Q2_by_side"][nm] = turn_tables(CI, yt[m], ts[m], vs[m], d_t[m], d_v[m], v0t[m], ets[m], nb)
        log(f"Q1/Q2 {nm} {time.time() - t0:.0f}s", flush=True)
    # ---- Q3
    q3 = {}
    for fam, s in RULE_ARMS + BESIDE_ARMS + (("C0", "s0"), ("C0", "s1"), ("C0", "s10"), ("C0", "s11")):
        a = f"{fam}_{s}"
        pred = A[a]["pred"][turn]
        missed = pred != yt
        own = vs if fam == "C0" else ts
        sp = miss_split(missed, ts, vs)
        row = {"feed": "v9" if fam == "C0" else "text", "n_turn_windows": int(len(yt)), "n_missed": int(missed.sum()), "head_recall_on_turns": float(1 - missed.mean()),
               "missed_by_text_state": {STATE_NAMES[k]: int((missed & (ts == k)).sum()) for k in (CORRECT, WRONG, NONE)},
               "missed_by_v9_state": {STATE_NAMES[k]: int((missed & (vs == k)).sum()) for k in (CORRECT, WRONG, NONE)}}
        for nm, fl in sp.items():
            row[f"share_{nm}"] = share_ci(CI, fl, missed, ets, nb)
        row["own_feed_cue_present_share_of_missed"] = share_ci(CI, missed & (own == CORRECT), missed, ets, nb)
        row["recall_when_own_cue_correct"] = share_ci(CI, (~missed) & (own == CORRECT), own == CORRECT, ets, nb)
        row["recall_when_own_cue_not_correct"] = share_ci(CI, (~missed) & (own != CORRECT), own != CORRECT, ets, nb)
        row["head_pred_counts_on_missed"] = {LAT9[k]: int((missed & (pred == k)).sum()) for k in range(5)}
        q3[a] = row
    res["Q3_missed_turns_by_arm"] = q3
    cells = {(f, s): {"share_cue_missing": q3[f"{f}_{s}"]["share_cue_missing"]["point"], "share_cue_present": q3[f"{f}_{s}"]["share_cue_present"]["point"]} for f, s in RULE_ARMS}
    verdict = reading(cells)
    res["reading"] = {"rule": f"CUE-MISSING: share of missed turns with no correct text turn at NOW while v9 names it >= {SHARE_BAR} in every cell of S1 / N1 x s0 / s1; "
                              f"CUE-PRESENT-MODEL: share with the correct text turn at NOW >= {SHARE_BAR} in every cell; else MIXED",
                      "cells": {f"{f}_{s}": c for (f, s), c in cells.items()}, "verdict": verdict}
    # ---- permutation importance and the unweighted shuffle (beside)
    imp = {}
    masks = {"ALL": np.ones(len(y), bool), "INF_ANY": N.split_masks(nav_v9, nav_txt)["INF_ANY"]}
    for feed, X in (("v9", nav_v9), ("text", nav_txt)):
        imp[feed] = {}
        for sp, m in masks.items():
            r, base_pred, preds0 = permutation_importance(X, y, eid, m)
            if sp == "ALL":
                ys, names = y, list(LAT9)
                for g, pr in preds0.items():
                    r["groups"][g]["paired_drop_repeat0"] = F.conf_pair(CI, ys, base_pred, pr, 5, names, eid, None, nb)["macro_f1"]
                ref_oof = N2.crossfit_hgb(X, y, eid)[0]
                if not np.array_equal(ref_oof, base_pred):
                    prob.append(f"K-BASE: {feed} out-of-fold predictions differ from navecho2.crossfit_hgb")
            imp[feed][sp] = r
        log(f"importance {feed} {time.time() - t0:.0f}s", flush=True)
    res["permutation_importance"] = imp
    maj = N.majority_f1(y)
    res["shuffle_unweighted_beside"] = {"majority_baseline_macro_f1": maj, "by_feed": {f: shuffled_unweighted(X, y, eid) for f, X in (("v9", nav_v9), ("text", nav_txt))},
                                        "navecho2_weighted_K_SHUFFLE_for_reference": {"v9": 0.230, "text": 0.145}}
    # ---- controls
    sc = cue_state(np.where(yt == TURN_L, TURN_L, TURN_R), yt)
    res["controls"] = {"K-SELF": {"self_pair_delta": mean_pair(CI, (ts == CORRECT).astype(float), (ts == CORRECT).astype(float), ets, nb)["delta"], "pass": None},
                       "K-REACH": {"label_as_token_all_correct": bool((sc == CORRECT).all()), "reading": "a token equal to the label reads CORRECT on every turn window"},
                       "K-FEED/K-LANDED/K-BASE problems": prob}
    res["controls"]["K-SELF"]["pass"] = res["controls"]["K-SELF"]["self_pair_delta"] == 0.0
    if not res["controls"]["K-SELF"]["pass"]:
        prob.append("K-SELF: a feed paired with itself does not read exactly 0")
    res["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "RESULT_NAVECHO3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=N.READ + "/t1")
    ap.add_argument("--out", default=HERE + "/out")
    ap.add_argument("--n-boot", type=int, default=NB)
    a = ap.parse_args()
    r = run(a.dumps, a.out, a.n_boot)
    print(json.dumps({"verdict": r["reading"]["verdict"], "problems": r["controls"]["K-FEED/K-LANDED/K-BASE problems"], "elapsed_s": r["elapsed_s"]}))
