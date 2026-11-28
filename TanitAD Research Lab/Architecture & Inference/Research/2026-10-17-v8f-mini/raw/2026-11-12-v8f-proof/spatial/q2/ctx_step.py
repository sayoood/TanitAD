"""ctx_step.py -- episode-cluster bootstrap of the one-step and the +2.0 s latent errors of the level-1 predictor against the copy (files DIAG_CTX_<ARM>.npz
of ctx_roll.py).  Tier T0, CPU.  Statistic: ratio of summed errors, predictor over copy (below 1 = better than the copy); B 2,000, seed 0; the interval
is the 2.5 and 97.5 percentile.  Control s1: the copy against itself reads exactly 1; s2: errors shuffled across windows keep the point ratio (it is a ratio of sums).
  python ctx_step.py --dir C:/Users/Admin/v8proof_c/ctxroll --arms G_s0,G_s1,W_s0 --out spatial/q2/ctx/STEP_CTX.md"""
import argparse
import json
import os
import sys

import numpy as np


def boot_ratio(num, den, epi, n_ep, B=2000, seed=0):
    a = np.bincount(epi, weights=num, minlength=n_ep)
    b = np.bincount(epi, weights=den, minlength=n_ep)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_ep, size=(B, n_ep))
    bs = a[idx].sum(1) / b[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(a.sum() / b.sum()), float(lo), float(hi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res, ok = {}, True
    L = ["# One-step and +2.0 s latent error of the level-1 predictor over the error of the copy", "",
         "MEASURED, tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).",
         "Ratio of summed squared errors over the held-out windows, predictor over copy; below 1 means better than the copy. Episode-cluster bootstrap, B 2,000.", ""]
    rows = []
    for arm in a.arms.split(","):
        z = np.load(os.path.join(a.dir, "DIAG_CTX_%s.npz" % arm), allow_pickle=True)
        eid = z["eid"].astype(str)
        uep = sorted(set(eid))
        ix = {e: i for i, e in enumerate(uep)}
        epi = np.array([ix[e] for e in eid])
        n_ep = len(uep)
        r = {}
        for key, num, den in (("step_context_1", "step_c1", "step_copy"), ("step_context_3", "step_c3", "step_copy"), ("lat2s_R1", "lat_R1", "lat_copy"),
                              ("lat2s_R3h", "lat_R3h", "lat_copy"), ("lat2s_R3g", "lat_R3g", "lat_copy"), ("lat2s_frame0_R1", "f0_R1", "f0_copy")):
            r[key] = boot_ratio(z[num], z[den], epi, n_ep)
        s1 = boot_ratio(z["step_copy"], z["step_copy"], epi, n_ep)
        perm = np.random.default_rng(3).permutation(len(eid))
        s2 = float(z["step_c1"][perm].sum() / z["step_copy"][perm].sum())
        c = {"s1_copy_over_copy_is_1": bool(s1 == (1.0, 1.0, 1.0)), "s2_shuffle_keeps_point": bool(abs(s2 - r["step_context_1"][0]) < 1e-12)}
        ok &= all(c.values())
        res[arm] = {"windows": int(len(eid)), "episodes": n_ep, "ratios": r, "controls": c}
        rows.append("| %s | %d | %s | %s | %s | %s | %s |" % ((arm, len(eid)) + tuple("%.3f [%.3f, %.3f]" % r[k] for k in ("step_context_1", "step_context_3", "lat2s_frame0_R1", "lat2s_R1", "lat2s_R3h"))))
    L += ["## Controls (first)", ""] + ["* %s: %s" % (arm, json.dumps(res[arm]["controls"])) for arm in res] + ["* ALL_PASS: %s" % ok, "",
          "| arm | windows | one step, context 1 | one step, context 3 | +2.0 s from frame 0, context 1 | +2.0 s from frame 4, context 1 | +2.0 s from frame 4, true history |", "|---|---|---|---|---|---|---|"] + rows
    L += ["", "Limits: held-out windows only; a ratio below 1 at +2.0 s is also what a shrinkage toward the mean state gives; one checkpoint per arm and seed.", ""]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    json.dump({"arms": res, "controls_pass": bool(ok)}, open(a.out[:-3] + ".json", "w", encoding="utf-8"), indent=1)
    print("\n".join(L))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
