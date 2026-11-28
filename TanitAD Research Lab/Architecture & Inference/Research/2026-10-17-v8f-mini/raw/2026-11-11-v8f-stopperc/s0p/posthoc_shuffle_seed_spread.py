"""POST-HOC (MM, 2026-11-12), decides nothing: the spread of the S0-P shuffle controls over the PERMUTATION draw.

posthoc_episode_shuffle.py refuted the label-sorted-order explanation of the E arm's episode-shuffled AUROC 0.3833 [0.2891, 0.4755]: the train arrays
are already in window-start order within a clip (as-run == time-order labels, AUROC difference exactly 0.0), and the shuffled labels are uncorrelated
with the real ones (r 0.0011).  Remaining HYPOTHESIS: 0.3833 is ONE draw of the permutation.  The episode-cluster bootstrap resamples EVAL episodes with
the fitted probe held fixed, so its CI cannot see the variance of the training-side permutation (the H-ESTIM-SEED-1 class: the interval answers
"another draw of episodes", not "another draw of the shuffle").  An episode shuffle moves labels in clip-sized blocks, so its effective n is the
number of clips, and its spread should be far wider than the window-level label shuffle's.

Reads: eval AUROC (point) of the linear probe, the real fit's hyperparameters, over permutation seeds 100..139 (E) and 100..119 (V40+E), for both
the episode shuffle (F.episode_permuted_labels) and the window-level label shuffle.  The as-run seeds (episode 2, label rng(1)) are reported beside.
Writes C:/Users/Admin/s0p_state/POSTHOC_SHUFFLE_SPREAD.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import s0p_fit as F  # noqa: E402
import s0p_lib as L  # noqa: E402

PULL = os.environ.get("S0P_PULL", "C:/Users/Admin/s0p_pull")
WORK = os.environ.get("S0P_WORK", "C:/Users/Admin/s0p_work")
EVAL = os.environ.get("S0P_EVAL", "F:/TanitAD-scratch/stopperc_s0p/s0p_eval.npz")
OUT = "C:/Users/Admin/s0p_state/POSTHOC_SHUFFLE_SPREAD.json"
SEEDS = {"E": list(range(100, 140)), "V40+E": list(range(100, 120))}


def summ(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
            "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95)), "values": [float(x) for x in v]}


def main():
    tr, ev = F.load_any(PULL), F.load_any(EVAL)
    y, clip = np.asarray(tr["y"], np.int64), np.asarray(tr["clip"])
    yev = np.asarray(ev["y"], bool)
    res = {"note": "POST-HOC, decides nothing; point AUROC on eval-141 (2,365 windows, 131 positives) per permutation seed",
           "n_train_clips": int(len(np.unique(clip)))}
    for arm, seeds in SEEDS.items():
        cfg = json.load(open(os.path.join(WORK, f"fit_{arm}_linear.json"), encoding="utf-8"))["best"]
        g = F.GRID_OF.get(arm)
        Ptr = np.load(os.path.join(WORK, f"proj_train_{g}.npy"), mmap_mode="r") if g else None
        Pev = np.load(os.path.join(WORK, f"proj_eval_{g}.npy"), mmap_mode="r") if g else None
        Xtr, Xev = F.build_features(arm, Ptr, np.asarray(tr["E"]), cfg["dim"]), F.build_features(arm, Pev, np.asarray(ev["E"]), cfg["dim"])
        ep, lab = [], []
        for s in seeds:
            ep.append(L.auroc(F._score(F._make_model("linear", cfg).fit(Xtr, F.episode_permuted_labels(y, clip, s)), Xev), yev))
            lab.append(L.auroc(F._score(F._make_model("linear", cfg).fit(Xtr, np.random.default_rng(s).permutation(y)), Xev), yev))
            print(arm, s, round(ep[-1], 4), round(lab[-1], 4), flush=True)
        e = summ(ep)
        e["frac_below_as_run"] = float(np.mean(np.asarray(ep) <= {"E": 0.3833263854244261, "V40+E": 0.42839325620015445}[arm]))
        res[arm] = {"cfg": cfg, "episode_shuffle": e, "label_shuffle": summ(lab),
                    "as_run": {"episode_seed2": {"E": 0.3833263854244261, "V40+E": 0.42839325620015445}[arm],
                               "label_rng1": {"E": 0.5143292078700444, "V40+E": 0.5366439549775502}[arm]}}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({a: {k: {kk: vv for kk, vv in v.items() if kk != "values"} if isinstance(v, dict) else v
                          for k, v in r.items()} if isinstance(r, dict) else r for a, r in res.items()}, indent=1))


if __name__ == "__main__":
    main()
