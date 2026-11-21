"""POST-HOC diagnostic of the S0-P episode-shuffle control (MM, 2026-11-12).  Decides nothing in the card's table.

RESULT_S0P.json read the E arm's episode-shuffled control at AUROC 0.3833 [0.2891, 0.4755]: BELOW chance, CI excluding 0.5.  A shuffled-label
control should sit near 0.5.  HYPOTHESIS: `episode_permuted_labels` takes each clip's windows in ARRAY order, and the training sample is stored
positives-first (s0p_lib.draw_sample: idx_pos, then idx_neg), so each clip's label sequence is label-SORTED ([1..1, 0..0]).  np.resize then truncates a
borrowed sequence to the target clip's length: a short target clip keeps only the head of the borrowed sequence, i.e. its positives.  Clips with few
sampled windows (non-stopping clips) then receive borrowed positives preferentially, and the shuffled labels anti-correlate with the stop cue.

This script (1) reproduces the as-run control bit for bit, (2) measures P(shuffled = 1 | real label) and the shuffled prevalence by clip sample size,
(3) re-runs the control with each clip's windows taken in TIME order (window start), the same clip permutation (seed 2), the same hyperparameters,
and reads both with the card's estimator (stat_cluster_bootstrap over eval episodes, B 2,000, seed 0).
Writes C:/Users/Admin/s0p_state/POSTHOC_EPSHUFFLE.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import s0p_fit as F  # noqa: E402
import s0p_lib as L  # noqa: E402

TREE = os.environ.get("S0P_TREE", "F:/TanitAD-scratch/stopbal_read/tree")
PULL = os.environ.get("S0P_PULL", "C:/Users/Admin/s0p_pull")
WORK = os.environ.get("S0P_WORK", "C:/Users/Admin/s0p_work")
EVAL = os.environ.get("S0P_EVAL", "F:/TanitAD-scratch/stopperc_s0p/s0p_eval.npz")
OUT = "C:/Users/Admin/s0p_state/POSTHOC_EPSHUFFLE.json"


def ep_perm_temporal(y, clip, ws, seed):
    """The same clip permutation as F.episode_permuted_labels (same sorted clip set, same seed), with each clip's windows in window-start order."""
    order = np.lexsort((ws, clip))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return F.episode_permuted_labels(y[order], clip[order], seed)[inv]


def main():
    sys.path.insert(0, TREE)
    L.setup_tree(TREE)
    from taniteval import ci as CI
    tr, ev = F.load_any(PULL), F.load_any(EVAL)
    y, clip, ws = np.asarray(tr["y"], np.int64), np.asarray(tr["clip"]), np.asarray(tr["ws"], np.int64)
    yev, eid = np.asarray(ev["y"], bool), np.asarray(ev["eid"], np.int64)
    res = {"note": "POST-HOC, decides nothing; card PREREG_V8F_STOPPERC.md section 4 lists the control without a bar",
           "n_train": int(len(y)), "n_train_pos": int(y.sum())}
    # is the within-clip array order label-sorted?
    sorted_clips = 0
    multi = 0
    for c in np.unique(clip):
        s = y[clip == c]
        if len(s) > 1 and s.min() != s.max():
            multi += 1
            sorted_clips += int((np.diff(s) <= 0).all())
    res["clips_with_both_labels"] = multi
    res["of_which_array_order_is_positives_first"] = sorted_clips
    y_run = F.episode_permuted_labels(y, clip, 2)
    y_tmp = ep_perm_temporal(y, clip, ws, 2)
    sizes = {c: int((clip == c).sum()) for c in np.unique(clip)}
    n_of = np.array([sizes[c] for c in clip])
    q = np.quantile(n_of, [1 / 3, 2 / 3])
    for name, ys in (("as_run", y_run), ("temporal", y_tmp)):
        d = {"prevalence": float(ys.mean()),
             "p_shuf1_given_real1": float(ys[y == 1].mean()), "p_shuf1_given_real0": float(ys[y == 0].mean()),
             "corr_with_real": float(np.corrcoef(ys, y)[0, 1])}
        for lab, m in (("clip_size_low_third", n_of <= q[0]), ("clip_size_mid_third", (n_of > q[0]) & (n_of <= q[1])), ("clip_size_high_third", n_of > q[1])):
            d[lab] = {"n": int(m.sum()), "real_prev": float(y[m].mean()), "shuf_prev": float(ys[m].mean())}
        res[name] = d
    scores = {}
    for arm in ("E", "V40+E"):
        cfg = json.load(open(os.path.join(WORK, f"fit_{arm}_linear.json"), encoding="utf-8"))["best"]
        g = F.GRID_OF.get(arm)
        Ptr = np.load(os.path.join(WORK, f"proj_train_{g}.npy"), mmap_mode="r") if g else None
        Pev = np.load(os.path.join(WORK, f"proj_eval_{g}.npy"), mmap_mode="r") if g else None
        Xtr, Xev = F.build_features(arm, Ptr, np.asarray(tr["E"]), cfg["dim"]), F.build_features(arm, Pev, np.asarray(ev["E"]), cfg["dim"])
        scores[f"{arm}|as_run"] = F._score(F._make_model("linear", cfg).fit(Xtr, y_run), Xev)
        scores[f"{arm}|temporal"] = F._score(F._make_model("linear", cfg).fit(Xtr, y_tmp), Xev)
        print(arm, "fitted", flush=True)

    def stat(sel):
        sel = np.asarray(sel)
        a = {k: L.auroc(v[sel], yev[sel]) for k, v in scores.items()}
        for arm in ("E", "V40+E"):
            a[f"D|{arm}|temporal - as_run"] = a[f"{arm}|temporal"] - a[f"{arm}|as_run"]
        return a

    B = CI.stat_cluster_bootstrap(list(range(len(yev))), eid, stat, n_boot=F.NB, seed=F.BSEED)
    res["auroc"] = {k: {kk: v[kk] for kk in ("point", "lo", "hi")} for k, v in B.items()}
    for arm in ("E", "V40+E"):
        r = B[f"{arm}|temporal"]
        res["auroc"][f"{arm}|temporal"]["ci_covers_0.5"] = bool(r["lo"] <= 0.5 <= r["hi"])
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
