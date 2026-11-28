"""navecho2.py -- D-STRAT1-TACLAT-F1, flexible nav-only classifier: is C0 lateral head F1 the label construction read back through the nav channels?

CPU only, zero training of the driving model, nothing re-dumped.  Imports (never edits) navecho.py and fam/fam_extra.py.
Tier of every number: T1-kin tick 0 (lateral head argmax at the first tick of each window under the T1 feed) vs the v9 label lat_cls (5-way).
Input of the classifier = nav_in at that tick: 5 channels (0-2 one-hot FOLLOW/TURN_L/TURN_R, 3 = distance/100 clipped [0,3], 4 = distance valid).
The dump carries only the vector fed at NOW (v8f_t1_eval.py:370 batch["nav"][b, -1]); no earlier-frame nav history is in the dump, so none is used.
Classifier: sklearn HistGradientBoostingClassifier, hyper-parameters fixed before scoring.  Episode-grouped 5-fold cross-fitting, out-of-fold only.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAVECHO = os.environ.get("NAVECHO_DIR", "F:/TanitAD-scratch/strat1/navecho")
if NAVECHO not in sys.path:
    sys.path.insert(0, NAVECHO)
import navecho as N  # noqa: E402  (also puts fam / read / taniteval on sys.path)

F, L = N.F, N.L
IGNORE, LAT9, NB, SEED = N.IGNORE, N.LAT9, N.NB, N.SEED
HGB_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, class_weight="balanced", random_state=0, early_stopping=False)
N_FOLDS = 5
ECHO_RATIO = 0.90
FEED_FRAC = 2.0 / 3.0
FAMILIES, SEEDS = N.FAMILIES, N.SEEDS
N_SHUFFLES = 10
TIER = N.TIER.replace("the nav-only predictor sees only nav_in at that tick", "the flexible nav-only classifier sees only the 5 nav_in channels at that tick")
EST = (f"episode-cluster bootstrap (taniteval.ci, paired stat for head - nav-only), B={NB}, seed {SEED}, 141 episodes; 'another draw of EPISODES' only; "
       "classifier: out-of-fold from episode-grouped 5-fold cross-fitting")


def episode_folds(eid, k=N_FOLDS, seed=0):
    """fold id per window: unique episodes permuted (seed) and dealt round-robin into k folds; an episode lives in exactly one fold"""
    eid = np.asarray(eid)
    u = np.unique(eid)
    order = np.random.default_rng(seed).permutation(u)
    fold_of = {int(e): i % k for i, e in enumerate(order)}
    return np.asarray([fold_of[int(e)] for e in eid], np.int64)


def make_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**HGB_PARAMS)


def crossfit_hgb(X, y, eid, k=N_FOLDS, seed=0, K=5, ignore=IGNORE):
    """out-of-fold class prediction for every window: the model that scores fold f is fit on the labelled windows of the other folds only"""
    X, y = np.asarray(X, np.float64), np.asarray(y)
    fold = episode_folds(eid, k, seed)
    pred = np.zeros(len(y), np.int64)
    lab = (y != ignore) & (y >= 0) & (y < K)
    for f in range(k):
        te, tr = fold == f, (fold != f) & lab
        if not te.any():
            continue
        ytr = y[tr]
        if len(np.unique(ytr)) < 2:
            pred[te] = int(np.bincount(ytr, minlength=K).argmax()) if len(ytr) else 0
            continue
        clf = make_clf().fit(X[tr], ytr)
        pred[te] = clf.classes_[clf.predict_proba(X[te]).argmax(1)]
    return pred, fold


def macro_f1(y, p, K=5):
    return N.macro_f1(y, p, K)


def label_onehot(y, K=5):
    y = np.asarray(y)
    o = np.zeros((len(y), K))
    m = (y >= 0) & (y < K)
    o[np.where(m)[0], y[m]] = 1.0
    return o


def shuffled_control(X, y, eid, n=N_SHUFFLES, K=5):
    rng = np.random.default_rng(SEED)
    v = [macro_f1(y, crossfit_hgb(np.asarray(X)[rng.permutation(len(y))], y, eid, K=K)[0], K) for _ in range(n)]
    return {"n_permutations": n, "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}


def reading(ratio, diff, gap):
    """committed rules.  ratio: {s0,s1} flexible-v9 F1 / C0 head F1.  diff: {s0,s1} flex_v9 - flex_text F1 (paired, same windows).  gap: {s0,s1} C0 - S1 head F1.
    ECHO-CONT: both ratios >= 0.90.  FEED-INFO (and not ECHO-CONT): at both seeds diff >= 2/3 of a positive gap.  else MODEL."""
    if all(ratio[s] >= ECHO_RATIO for s in ("s0", "s1")):
        return "ECHO-CONT"
    if all(gap[s] > 0 and diff[s] >= FEED_FRAC * gap[s] for s in ("s0", "s1")):
        return "FEED-INFO"
    return "MODEL"


def masked(y, m):
    return N.masked(y, m)


def run(dumps, out_dir, nb=NB, log=print):
    from taniteval import ci as CI
    t_start = time.time()
    join = dict(np.load(L.pinned("join_eval141.npz"), allow_pickle=False))
    ref = F.load_dump(F.dump_dir(dumps, "C0_s0"))
    eid, ci_, ws = ref["eid"], ref["clip_index"], ref["ws"]
    y = F.v9_rows(join, ci_, ws)["lat"]
    arms = [f"{f}_{s}" for f in FAMILIES for s in SEEDS] + ["C0_rep"]
    A = {}
    for a in arms:
        lg, nav = N.tick0(F.ticks_path(dumps, a), ci_, ws)
        A[a] = {"pred": lg.argmax(1), "nav": nav}
    prob = []
    for a in arms:
        ref_a = "C0_s0" if a.startswith("C0") else "N1_s0"
        if not np.array_equal(A[a]["nav"], A[ref_a]["nav"]):
            prob.append(f"K-FEED: {a} nav_in differs from {ref_a}")
    nav_v9, nav_txt = A["C0_s0"]["nav"], A["N1_s0"]["nav"]
    res = {"tool": "navecho2.py", "question": "D-STRAT1-TACLAT-F1 flexible nav-only classifier", "evidence_class": "MEASURED (ours; banked T1-kin dumps, zero training, CPU)",
           "tier": TIER, "estimator": EST, "classifier": {"name": "sklearn HistGradientBoostingClassifier", "params": HGB_PARAMS, "folds": N_FOLDS,
           "grouping": "episode (each episode in exactly one fold)", "input": "nav_in, 5 channels at the first tick of the window (no frame history in the dump)"},
           "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))), "n_labelled": int((y != IGNORE).sum())}
    masks = N.split_masks(nav_v9, nav_txt)
    masks = {"ALL": np.ones(len(y), bool), "INF_ANY": masks["INF_ANY"], "UNINF_BOTH": masks["UNINF_BOTH"]}
    res["splits"] = N.split_counts(y, eid, masks)
    # ---- the classifiers
    oof, X = {}, {"v9": nav_v9, "text": nav_txt}
    for feed in X:
        oof[feed], fold = crossfit_hgb(X[feed], y, eid)
    res["fold_episodes"] = {str(f): int(len(np.unique(np.asarray(eid)[fold == f]))) for f in range(N_FOLDS)}
    landed = json.load(open(os.path.join(N.FAM, "out", "RESULT_FAM_EXTRA.json"), encoding="utf-8"))
    head = {a: macro_f1(y, A[a]["pred"]) for a in arms}
    for a in arms:
        lf = landed["families"]["TACTICAL"][a]["TAC-SEL-LAT"]["macro_f1"]["point"]
        if abs(head[a] - lf) > 1e-12:
            prob.append(f"K-LANDED: {a} {head[a]!r} != {lf!r}")
    # ---- controls
    shuf = {f: shuffled_control(X[f], y, eid) for f in X}
    lab_oof, _ = crossfit_hgb(label_onehot(y), y, eid)
    maj = N.majority_f1(y)
    res["controls"] = {"majority_baseline_macro_f1": maj,
                       "K-SHUFFLE": {"by_feed": shuf, "pass": bool(all(abs(shuf[f]["mean"] - maj) <= 0.08 for f in shuf)), "reading": "nav permuted across windows (so episodes), out-of-fold, within 0.08 of the majority score (band fixed before the run)"},
                       "K-REACH": {"label_as_input_macro_f1": macro_f1(y, lab_oof), "pass": bool(macro_f1(y, lab_oof) == 1.0), "reading": "the label itself as the input reads 1.0 out-of-fold"}}
    if not res["controls"]["K-SHUFFLE"]["pass"]:
        prob.append("K-SHUFFLE not near majority")
    if not res["controls"]["K-REACH"]["pass"]:
        prob.append("K-REACH: label-as-input is not 1.0")
    # ---- flexible classifier vs heads, per split
    names = list(LAT9)
    ydict, tab = {sp: masked(y, m) for sp, m in masks.items()}, {}
    FL = {}
    for sp, ys in ydict.items():
        pr = F.conf_pair(CI, ys, A["C0_rep"]["pred"], A["C0_s0"]["pred"], 5, names, eid, None, nb)
        FL[sp] = {k: abs(pr[k]["delta"]) for k in ("macro_f1",) if k in pr}
        tab[sp] = {}
        for feed in X:
            b = F.conf_block(CI, ys, oof[feed], 5, names, eid, nb)
            tab[sp][f"flex_{feed}"] = {k: b[k] for k in ("n", "n_episodes", "macro_f1", "accuracy", "support")}
        for a in arms:
            if a == "C0_rep":
                continue
            feed = "v9" if a.startswith("C0") else "text"
            b = F.conf_block(CI, ys, A[a]["pred"], 5, names, eid, nb)
            d = F.conf_pair(CI, ys, A[a]["pred"], oof[feed], 5, names, eid, FL[sp], nb)["macro_f1"]
            fx = tab[sp][f"flex_{feed}"]["macro_f1"]["point"]
            tab[sp][a] = {"feed": feed, "head_macro_f1": b["macro_f1"], "n": b["n"], "n_episodes": b["n_episodes"],
                          "flex_macro_f1": fx, "head_minus_flex_paired": d,
                          "ratio_flex_to_head": fx / b["macro_f1"]["point"] if b["macro_f1"]["point"] else None}
        # feed information: flex_v9 - flex_text (paired, same windows) and head gaps
        tab[sp]["flex_v9_minus_flex_text_paired"] = F.conf_pair(CI, ys, oof["v9"], oof["text"], 5, names, eid, FL[sp], nb)["macro_f1"]
        for s in SEEDS:
            for fam in ("S1", "N1", "S1d"):
                tab[sp][f"head_gap_C0_{s}_minus_{fam}_{s}"] = F.conf_pair(CI, ys, A[f"C0_{s}"]["pred"], A[f"{fam}_{s}"]["pred"], 5, names, eid, FL[sp], nb)["macro_f1"]
        log(f"split {sp} {time.time() - t_start:.0f}s", flush=True)
    res["by_split"] = tab
    res["floor_F_abs_C0rep_minus_C0s0"] = FL
    # ---- committed reading (ALL windows, S1 as the text arm; N1 / S1d / s10 / s11 reported beside)
    ratio = {s: tab["ALL"][f"C0_{s}"]["ratio_flex_to_head"] for s in ("s0", "s1")}
    diff = {s: tab["ALL"]["flex_v9_minus_flex_text_paired"]["delta"] for s in ("s0", "s1")}
    gap = {s: tab["ALL"][f"head_gap_C0_{s}_minus_S1_{s}"]["delta"] for s in ("s0", "s1")}
    verdict = reading(ratio, diff, gap)
    extra = {s: {"ratio": tab["ALL"][f"C0_{s}"]["ratio_flex_to_head"], "gap_C0_minus_S1": tab["ALL"][f"head_gap_C0_{s}_minus_S1_{s}"]["delta"],
                 "gap_C0_minus_N1": tab["ALL"][f"head_gap_C0_{s}_minus_N1_{s}"]["delta"], "gap_C0_minus_S1d": tab["ALL"][f"head_gap_C0_{s}_minus_S1d_{s}"]["delta"]} for s in SEEDS}
    res["reading"] = {"rule": "ECHO-CONT: flexible v9 classifier >= 90 % of C0 head macro-F1 at s0 AND s1; FEED-INFO (not ECHO-CONT): flex_v9 - flex_text >= 2/3 of C0 - S1 head gap (gap > 0) at s0 AND s1; else MODEL",
                      "ratio_flex_v9_to_C0_head": ratio, "flex_v9_minus_flex_text": diff, "head_gap_C0_minus_S1": gap, "two_thirds_of_gap": {s: FEED_FRAC * gap[s] for s in gap},
                      "per_seed": extra, "verdict": verdict}
    res["controls"]["problems"] = prob
    res["elapsed_s"] = round(time.time() - t_start, 1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "RESULT_NAVECHO2.json"), "w", encoding="utf-8") as f:
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
    print(json.dumps({"verdict": r["reading"]["verdict"], "problems": r["controls"]["problems"], "elapsed_s": r["elapsed_s"]}))
