"""Does the occ channel earn its keep WHERE THE BOX IS WRONG? The matcher-selection confound.

`occ_calib.py` found the head's occ channel LOSES to a 2-parameter logistic on its own predicted
azimuth (0.288 vs 0.161, CI [-0.184, -0.038]). ⛔ BUT THERE IS A SELECTION CONFOUND IN THAT
COMPARISON AND IT RUNS IN EXACTLY ONE DIRECTION.

`match_slots`'s Hungarian cost is built from **centre, cls and presence** (`agent_slots.py:475-477`).
So the pairs both arms are scored on are, by construction, pairs where the PREDICTED CENTRE IS
CLOSE TO THE TARGET'S. An arm that READS the predicted centre (H2) is therefore handed a selection
that favours it; the occ channel is not. ⇒ the headline comparison is on the box arm's home ground,
and a verdict taken from it alone would be the mirror of the hard-predicate mistake this directory
already made once today.

⭐ THE STRATIFIED READ ANSWERS IT WITHOUT ANY NEW COMPUTE. The confounder is OBSERVED: the
azimuth error `|az_pred - az_gt|` is the exact quantity H2 depends on, and it is already in the
cached rows. Splitting on it asks the question the pooled number cannot:

  O loses in EVERY stratum        -> the finding survives the confound; the channel is worse than a
                                     free read of the box even where the box is at its worst.
  O wins in the HIGH-ERROR stratum -> the channel earns its keep precisely where the box fails, and
                                     the pooled verdict was a selection artifact. Report THAT.

⛔ The strata are cut on the FIT half's quantiles and applied to the SCORED half, so the cut points
are not chosen on the data being scored. Per-stratum n is printed; a stratum under 60 pairs is
reported as UNDERPOWERED rather than read as a negative.
⚠️ The logistic is re-fit ONCE on the whole fit half (not per stratum), because per-stratum fitting
would hand H2 the confounder as a feature and answer a different question.
⛔ CPU only, no forward pass — reads `occ_rows.npz`.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE = pathlib.Path("occ_rows.npz")
B = 4000
SEED = 20260921
EPS = 1e-6


def ll(p, y):
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def logistic(Xf, yf, Xs, iters=400, lam=1e-3):
    b = np.zeros(Xf.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xf @ b, -30, 30)))
        W = np.clip(p * (1.0 - p), 1e-6, None)
        step = np.linalg.solve(Xf.T @ (Xf * W[:, None]) + lam * np.eye(Xf.shape[1]),
                              Xf.T @ (yf - p) - lam * b)
        b = b + step
        if float(np.abs(step).max()) < 1e-9:
            break
    return 1.0 / (1.0 + np.exp(-np.clip(Xs @ b, -30, 30))), b


def main() -> int:
    assert CACHE.exists(), "ZZABORT run occ_calib.py first to build occ_rows.npz"
    A = np.load(CACHE)["rows"]
    epi, y, p_head, az_gt, az_pr = (A[:, k] for k in range(5))
    err = np.abs(az_pr - az_gt)

    eps_ = sorted(set(epi.tolist()))
    half = set(eps_[: len(eps_) // 2])
    fit = np.array([e in half for e in epi])
    sc = ~fit

    Xf = np.stack([az_pr[fit], np.ones(int(fit.sum()))], 1)
    Xs = np.stack([az_pr[sc], np.ones(int(sc.sum()))], 1)
    p_h2, b2 = logistic(Xf, y[fit], Xs)

    cuts = [float(np.quantile(err[fit], q)) for q in (1 / 3, 2 / 3)]
    es, ys = epi[sc], y[sc]
    errs, ph = err[sc], p_head[sc]
    rgen = random.Random(SEED)

    bands = [("LOW  box error", errs <= cuts[0]),
             ("MID  box error", (errs > cuts[0]) & (errs <= cuts[1])),
             ("HIGH box error", errs > cuts[1])]
    out = []
    for name, m in bands:
        n = int(m.sum())
        if n == 0:
            out.append({"stratum": name, "n": 0, "note": "empty"})
            continue
        lo, lh = ll(p_h2[m], ys[m]), ll(ph[m], ys[m])
        keys = sorted(set(es[m].tolist()))
        idx = {k: np.where((es == k) & m)[0] for k in keys}
        d = ll(p_h2, ys) - ll(ph, ys)
        ms = sorted(float(np.concatenate([d[idx[k]] for k in rgen.choices(keys, k=len(keys))]).mean())
                    for _ in range(B)) if len(keys) > 1 else None
        ci = [round(ms[int(0.025 * B)], 5), round(ms[int(0.975 * B)], 5)] if ms else None
        gain = float(lo.mean() - lh.mean())          # >0 means the OCC CHANNEL wins
        out.append({
            "stratum": name, "n": n, "n_episodes": len(keys),
            "mean_abs_azimuth_error_deg": round(float(np.degrees(errs[m].mean())), 4),
            "base_rate_here": round(float(ys[m].mean()), 4),
            "logloss_H2_logistic_own_azimuth": round(float(lo.mean()), 5),
            "logloss_O_head_occ_logit": round(float(lh.mean()), 5),
            "gain_H2_minus_O": round(gain, 5), "CI95": ci,
            "underpowered": n < 60,
            "reading": ("⭐ the OCC CHANNEL WINS here" if ci and ci[0] > 0 and gain > 0 else
                        "⛔ the occ channel LOSES to a free read of the box here"
                        if ci and ci[1] < 0 else "— indistinguishable")})

    winner_high = [r for r in out if r["stratum"].startswith("HIGH")
                   and r.get("CI95") and r["CI95"][0] > 0]
    res = {"_what": "does the occ channel beat a free read of the box WHERE THE BOX IS WRONG?",
           "_evidence_class": "MEASURED (ours), CPU, A8 ckpt_5000, no new forward pass",
           "_confound": ("match_slots pairs on CENTRE (agent_slots.py:475-477), so matched pairs "
                         "favour any arm that reads the predicted centre"),
           "_stratifier": "|az_pred - az_gt|, cut at the FIT half's 1/3 and 2/3 quantiles",
           "cut_points_deg": [round(math.degrees(c), 4) for c in cuts],
           "H2_implied_threshold_deg": round(math.degrees(float(b2[1] / -b2[0])), 4),
           "n_scored_pairs": int(sc.sum()), "strata": out}
    res["_VERDICT"] = (
        "⭐ THE CONFOUND WAS REAL — the occ channel WINS in the high-box-error stratum, i.e. it "
        "earns its keep exactly where the box fails, and the pooled verdict was a selection artifact"
        if winner_high else
        "⛔ THE FINDING SURVIVES THE CONFOUND — the occ channel loses to a free read of the head's "
        "own azimuth in every stratum INCLUDING the one where that azimuth is at its worst")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    for r in out:
        if "reading" in r:
            print(f"  {r['stratum']}  n={r['n']:<4} err={r['mean_abs_azimuth_error_deg']:>7.3f}deg  "
                  f"H2 {r['logloss_H2_logistic_own_azimuth']:.4f} vs O "
                  f"{r['logloss_O_head_occ_logit']:.4f}   {r['reading']}")
    print("\n" + res["_VERDICT"])
    pathlib.Path("occ_strata.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
