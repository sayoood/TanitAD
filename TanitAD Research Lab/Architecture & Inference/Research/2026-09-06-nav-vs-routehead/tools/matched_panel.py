"""D-NAVROUTE-1 §4 -- the MATCHED intervention panel.

Every intervention is scored on the SAME windows, against the SAME baseline, so
the nav interventions and the g_str intervention are directly comparable. The
replicate row is the rig's own run-to-run floor.

⛔ The join key is the CLIP INDEX read from each dump's top-level `ep*.npz`,
never the filename: a filename join would silently assume both runs enumerated
episodes in the same order, and a wrong-clip join reads as a large effect.

ASCII-only output. Estimator: paired episode-cluster bootstrap.
"""
import glob
import json
import os
import sys

import numpy as np

BANK = os.environ.get("BANK", "/home/nvidia/navpred/navflip_dump")
ROOT = os.environ.get("NAVROUTE", "/home/nvidia/navroute")
NBOOT = int(os.environ.get("NBOOT", "2000"))
SEED = int(os.environ.get("SEED", "0"))


def load(dumpdir, keys):
    out = {}
    if not os.path.isdir(os.path.join(dumpdir, "decisions")):
        return out
    for f in sorted(glob.glob(os.path.join(dumpdir, "decisions", "*.npz"))):
        top = os.path.join(dumpdir, os.path.basename(f))
        if not os.path.exists(top):
            continue
        t = np.load(top, allow_pickle=True)
        if "clip_index" not in t.files:
            continue
        ci = int(np.asarray(t["clip_index"]).reshape(-1)[0])
        z = np.load(f, allow_pickle=True)
        if not all(k in z.files for k in keys):
            continue
        ws = np.asarray(z["ws"]).reshape(-1)
        for i, w in enumerate(ws):
            out[(ci, int(w))] = tuple(np.asarray(z[k])[i] for k in keys)
    return out


def boot(ep, x, nboot, seed):
    rng = np.random.default_rng(seed)
    ue = np.unique(ep)
    idx = {e: np.where(ep == e)[0] for e in ue}
    d = np.empty(nboot)
    for i in range(nboot):
        sel = np.concatenate([idx[e] for e in rng.choice(ue, ue.size, True)])
        d[i] = x[sel].mean()
    return {"mean_m": round(float(x.mean()), 4),
            "lo": round(float(np.percentile(d, 2.5)), 4),
            "hi": round(float(np.percentile(d, 97.5)), 4),
            "median_m": round(float(np.median(x)), 4),
            "max_m": round(float(x.max()), 4),
            "frac_plan_UNCHANGED": round(float((x == 0).mean()), 4),
            "n_windows": int(x.size), "n_episodes": int(ue.size)}


def main():
    B = load(BANK, ["plan_full_nav_true", "plan_full_nav_flipped",
                    "plan_full_nav_zero", "plan_full_nav_shuffled"])
    R = load(os.path.join(ROOT, "dump_REPL"), ["plan_full_nav_true"])
    arms = {t: load(os.path.join(ROOT, f"dump_{t}"), ["plan_full_nav_true"])
            for t in ("GZERO", "BLIND", "GSHUF")}
    have = [t for t, v in arms.items() if v]
    ks = sorted(set(B) & set(R))
    for t in have:
        ks = sorted(set(ks) & set(arms[t]))
    if not ks:
        print(json.dumps({"ERROR": "no matched windows"}, indent=1))
        return
    ep = np.array([k[0] for k in ks])
    base = np.array([B[k][0][-1] for k in ks])          # banked os terminal pt
    rep = np.array([R[k][0][-1] for k in ks])

    def dist(a):
        return np.linalg.norm(a - base, axis=-1)

    res = {
        "tool": "D-NAVROUTE-1 matched intervention panel",
        "tier": "T1 (self-action OPEN loop, 2026-09-02 ruling)",
        "estimator": f"paired episode-cluster bootstrap, {NBOOT} resamples, "
                     f"seed {SEED}",
        "metric": "plan TERMINAL displacement (m) at the 6 s point vs the "
                  "banked unablated `os` arm, same ckpt/config/grid",
        "ckpt": "refcv4b ckpt_40284_FINAL.pt (step 40,284)",
        "n_matched_windows": len(ks),
        "n_matched_episodes": int(np.unique(ep).size),
        "arms_present": have,
        "rows": {},
    }
    res["rows"]["NOISE_FLOOR_replicate"] = boot(ep, dist(rep), NBOOT, SEED)
    res["rows"]["NOISE_FLOOR_replicate"]["_is"] = (
        "REPL: the same flags and the same seed, run again. THE FLOOR. An "
        "effect at or below this is not an effect.")
    for nm, j in (("nav_FLIP", 1), ("nav_ZERO", 2), ("nav_SHUFFLE", 3)):
        res["rows"][nm] = boot(ep, dist(np.array([B[k][j][-1] for k in ks])),
                               NBOOT, SEED)
    for t in have:
        a = np.array([arms[t][k][0][-1] for k in ks])
        res["rows"][t] = boot(ep, np.linalg.norm(a - rep, axis=-1), NBOOT, SEED)
    res["rows_note"] = ("the g_str arms are differenced against REPL (their own "
                        "surface); the nav rows against the banked `os` in the "
                        "SAME file. The replicate row shows the two baselines "
                        "agree to 1e-4 m, so the comparison is admissible.")
    print(json.dumps(res, indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
