"""tac_controls_r8.py -- the TACTICAL family's head accuracy, read against the controls that say whether it is VISION SKILL.

Why (MEASURED 2026-10-20, GAC R8-NAV-FEED-40K, TAC_ECHO_CHECK.json): on eval139 the lateral label is class 0 on 89.7 % of windows,
so a CONSTANT predictor scores 0.897; refcv8's head scored 0.934 on the per-frame feed and 0.587 on the clip feed, and it largely
follows the fed nav token.  A head accuracy quoted bare is therefore not admissible.  This tool reads a battery tree's decision dumps
(``dump_s<seed>/decisions/ep###.npz``) and reports, per head (lat, lon) and per inference seed:

  * head accuracy and macro-F1 on the windows with a label (label >= 0);
  * head - CONSTANT (the majority class of the SAME windows: the no-information value, optimistic for the control);
  * head(nav_true) - head(nav_zero)      -- what the nav input adds (the intervention read);
  * head(nav_true) - head(nav_shuffled)  -- whether the head reads the token it is given;
  * the echo index: share of windows where the prediction equals the modal prediction of its fed token.

Every delta is a PAIRED episode-cluster bootstrap (taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0): it answers
'another draw of EPISODES' at a fixed checkpoint, never 'another training run'.  A pass that was not run (all -1) reads NOT_RUN,
never a number.  CPU, numpy.

    python tac_controls_r8.py <battery tree> --out <json> [--seeds 0 1]
"""
import argparse, collections, glob, json, os, sys

import numpy as np

HEADS = ("lat", "lon")
PASSES = ("nav_true", "nav_zero", "nav_shuffled")
MUTATION = os.environ.get("TAC_CONTROLS_MUTATION", "")      # "constant_from_pred": the control's class read from the PREDICTIONS


def load_seed(tree: str, seed: int) -> dict | None:
    fs = sorted(glob.glob(os.path.join(tree, f"dump_s{seed}", "decisions", "ep*.npz")))
    if not fs:
        return None
    R = collections.defaultdict(list)
    for f in fs:
        z = np.load(f)
        n = len(z["ws"])
        ep = int(os.path.basename(f)[2:5])
        R["eid"].append(np.full(n, ep, np.int64))
        R["nav_cmd"].append(z["nav_cmd"] if "nav_cmd" in z.files else np.full(n, -1, np.int64))
        for h in HEADS:
            R[f"{h}_label"].append(z[f"{h}_label"])
            for p in PASSES:
                k = f"{h}_pred_{p}"
                R[k].append(z[k] if k in z.files else np.full(n, -1, np.int64))
    out = {k: np.concatenate(v) for k, v in R.items()}
    out["n_files"] = len(fs)
    return out


def macro_f1(pred, lab) -> float:
    cls = sorted(set(lab.tolist()))
    f = []
    for c in cls:
        tp = float(np.sum((pred == c) & (lab == c))); fp = float(np.sum((pred == c) & (lab != c))); fn = float(np.sum((pred != c) & (lab == c)))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return round(float(np.mean(f)), 4) if f else float("nan")


def head_report(R: dict, h: str, ci, n_boot: int) -> dict:
    lab = R[f"{h}_label"]
    ok = lab >= 0
    if not ok.any():
        return {"status": "NO_LABEL"}
    lab, eid, fed = lab[ok], R["eid"][ok], R["nav_cmd"][ok]
    pt = R[f"{h}_pred_nav_true"][ok]
    src = pt if MUTATION == "constant_from_pred" else lab
    major = collections.Counter(src.tolist()).most_common(1)[0][0]
    c_true = (pt == lab).astype(float)
    c_const = (lab == major).astype(float)
    rec = {"n_windows": int(ok.sum()), "n_episodes": int(len(set(eid.tolist()))),
           "acc_nav_true": round(float(c_true.mean()), 4), "macro_f1_nav_true": macro_f1(pt, lab),
           "constant_class": int(major), "acc_constant": round(float(c_const.mean()), 4),
           "label_counts": {int(k): int(v) for k, v in sorted(collections.Counter(lab.tolist()).items())}}
    rec["head_minus_constant"] = ci.paired_episode_cluster_bootstrap(c_true, c_const, eid, n_boot=n_boot, seed=0)
    for p in ("nav_zero", "nav_shuffled"):
        pp = R[f"{h}_pred_{p}"][ok]
        if (pp < 0).all():
            rec[f"head_minus_{p}"] = {"status": "NOT_RUN", "reason": f"{h}_pred_{p} is the -1 sentinel on every window"}
            rec[f"acc_{p}"] = None
            continue
        rec[f"acc_{p}"] = round(float((pp == lab).mean()), 4)
        rec[f"head_minus_{p}"] = ci.paired_episode_cluster_bootstrap(c_true, (pp == lab).astype(float), eid, n_boot=n_boot, seed=0)
    if (fed >= 0).any():
        mode = {int(v): collections.Counter(pt[fed == v].tolist()).most_common(1)[0][0] for v in set(fed.tolist())}
        rec["echo_index_pred_eq_token_mode"] = round(float(np.mean([mode[int(t)] == int(p) for t, p in zip(fed, pt)])), 4)
    sep = rec["head_minus_constant"]
    rec["reading"] = ("ABOVE the constant control, separated" if sep["separated"] and sep["delta"] > 0 else
                      "BELOW the constant control, separated" if sep["separated"] else "NOT separated from the constant control")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree"); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    from taniteval import ci
    res = {"tree": a.tree, "estimator": "paired_episode_cluster_bootstrap, B %d, seed 0: another draw of EPISODES at a fixed "
           "checkpoint and inference seed; never another training run (H-ESTIM-SEED-1)" % a.n_boot,
           "tier": "T1 open loop on logged frames (the battery's own tier); NOT driving performance", "seeds": {}}
    for s in a.seeds:
        R = load_seed(a.tree, s)
        if R is None:
            res["seeds"][str(s)] = {"status": "ABSENT", "reason": f"no dump_s{s}/decisions under {a.tree}"}
            continue
        res["seeds"][str(s)] = {"n_episode_files": R["n_files"], **{h: head_report(R, h, ci, a.n_boot) for h in HEADS}}
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    for s, v in res["seeds"].items():
        for h in HEADS:
            r = v.get(h) or {}
            if "acc_nav_true" in r:
                d = r["head_minus_constant"]
                print(f"seed {s} {h}: acc {r['acc_nav_true']} vs constant {r['acc_constant']} -> {d['delta']:+} [{d['lo']}, {d['hi']}] "
                      f"({r['reading']}); zero {r.get('acc_nav_zero')} shuf {r.get('acc_nav_shuffled')}; n {r['n_windows']}/{r['n_episodes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
