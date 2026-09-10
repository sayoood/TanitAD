#!/usr/bin/env python3
"""B1's NEXT LEVER: is the signal population's 6 EPISODES the CORPUS's limit, or the
PROBE's definition? -- a POWER CURVE, not a gate proposal.

⛔ WHY THIS EXISTS. `D-RL-GREWARD-UNREACHABLE-1` measured that `G-REWARD`'s <= 30 %
ceiling is unreachable by every admissible reward (min 0.3840 over 2,400 weightings)
because its population is ~99 % windows carrying no safety information. The corrected,
signal-bearing form reads **0.0635 [0.0000, 0.3158]** -- but on **n = 63 windows /
6 EPISODES**, post-hoc, with an upper bound that STRADDLES the ceiling. The package's own
stated limit is that it *"may not become the gate until it is pre-registered on its own,
on a corpus with enough lead-conflict episodes to carry a bootstrap."*

⇒ The question that makes B1 DECIDABLE rather than a judgement call is not *"which
definition passes?"* -- that is outcome selection -- but **"how many episodes does each
principled population actually contain, and how wide is the interval there?"**

⛔⛔ PRE-REGISTERED BEFORE ANY RATE WAS COMPUTED, and binding on this file:

 1. The population is defined by a **PROPERTY OF THE SCENE**, never of the outcome. The
    scene property is `gt_time_gap_min_s` -- the HUMAN's own minimum time gap to the lead
    over the window, which is recorded before any candidate exists and does not depend on
    what any policy does. ⛔ No population here is defined by whether a term "fired", by
    a rate, or by anything a reward computed.
 2. The threshold LADDER is fixed in advance and is exhaustive:
    **(inf, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0) s**. Every rung is reported -- ⛔ the
    ladder is NOT truncated at the rung that passes.
 3. **Every rung reports n_windows, n_episodes, the rate, and its episode-cluster CI**,
    plus the two continuity columns (the sibling's "a safety term differs" population and
    the all-window rate) so this panel can be read against the banked one.
 4. ⛔ **THIS SCRIPT SELECTS NOTHING.** It emits a curve. Choosing the gate's population
    is a PI / Master-Mind decision, and a gate chosen after seeing which rung passes is a
    moved goalpost -- which is the whole reason B1 was escalated instead of patched.
 5. ⭐ CONTROLS THAT MUST READ KNOWN VALUES: the `inf` rung must reproduce the BANKED
    all-window rate exactly (channel control); `frozen` (the do-nothing path) must stay
    far below the human at every rung; and the rate at a rung with n = 0 is reported as
    `null`, never as 0.0.

0 GPU. Reads only banked per-window rows. Tier: T0 instrument probe, NON-PARITY RL-fit
windows. Evidence class: MEASURED (ours).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

#: ⛔ FIXED IN ADVANCE (pre-registration item 2). Descending = tighter conflict.
LADDER = (float("inf"), 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0)

#: The banked `veto_feas_comfort + robust_contact` weighting -- the repaired reward whose
#: signal-window rate is the quantity under study. Fixed here, not swept.
WEIGHTS = {"progress": 0.3, "collision": 1.0, "headway": 0.3}

SAFETY_TERMS = ("collision",)


def composed(row_side, weights):
    return sum(w * float(row_side.get(k, 0.0)) for k, w in weights.items())


def cluster_ci(flags, eids, n_boot=4000, seed=0, alpha=0.05):
    """Episode-cluster bootstrap on a per-window 0/1 rate.

    ⚠️ ANSWERS ONE QUESTION ONLY: 'would another draw of EPISODES say this?' -- not
    another training run and not another inference run.
    """
    flags = np.asarray(flags, dtype=np.float64)
    eids = np.asarray(eids)
    uniq = np.unique(eids)
    if len(uniq) == 0 or flags.size == 0:
        return None, None
    idx = {e: np.flatnonzero(eids == e) for e in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        draws[b] = flags[sel].mean()
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="humanflag per-window JSON")
    ap.add_argument("--lead-mode", default="track", choices=("track", "static"))
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    with open(a.rows, encoding="utf-8") as fh:
        d = json.load(fh)
    pw = d["per_window"]
    hk, vk, fk = ("human_" + a.lead_mode, "hold_v0_" + a.lead_mode,
                  "frozen_" + a.lead_mode)

    eid = np.array([r["eid"] for r in pw])
    gap = np.array([float(r.get("gt_time_gap_min_s", np.nan)) for r in pw])
    h = np.array([composed(r[hk], WEIGHTS) for r in pw])
    v = np.array([composed(r[vk], WEIGHTS) for r in pw])
    f = np.array([composed(r[fk], WEIGHTS) for r in pw])
    ge = (v >= h - 1e-12).astype(np.float64)
    fge = (f >= h - 1e-12).astype(np.float64)
    # continuity column: the sibling's population -- a SAFETY TERM differs at all
    differs = np.zeros(len(pw), dtype=bool)
    for t in SAFETY_TERMS:
        differs |= np.array([abs(float(r[hk].get(t, 0.0))
                                 - float(r[vk].get(t, 0.0))) > 1e-12 for r in pw])

    print("== G-REWARD POWER CURVE (pre-registered ladder; SELECTS NOTHING)")
    print("rows        : %s" % a.rows)
    print("lead model  : %s   weights: %s" % (a.lead_mode, WEIGHTS))
    print("corpus      : %d windows / %d episodes"
          % (len(pw), len(np.unique(eid))))
    print("n finite gt_time_gap_min_s: %d" % int(np.isfinite(gap).sum()))
    print("")
    print("%-12s %9s %9s %9s   %-20s %9s" % (
        "population", "n_win", "n_eps", "rate", "episode-cluster CI", "frozen"))

    out = {"_what": "G-REWARD power curve over a pre-registered scene-property ladder",
           "_evidence_class": "MEASURED (ours)",
           "_tier": "T0 instrument probe, NON-PARITY RL-fit windows",
           "_prereg": "ladder fixed before any rate was computed; every rung reported; "
                      "this script SELECTS NOTHING -- the gate's population is a PI / "
                      "Master-Mind decision",
           "_estimator": "episode-cluster bootstrap; answers 'would another draw of "
                         "EPISODES say this?' only (H-ESTIM-SEED-1)",
           "lead_mode": a.lead_mode, "weights": WEIGHTS,
           "n_windows_total": len(pw),
           "n_episodes_total": int(len(np.unique(eid))),
           "ladder": [], "continuity": {}}

    for thr in LADDER:
        m = np.isfinite(gap) & (gap <= thr) if np.isfinite(thr) else np.ones(len(pw), bool)
        n, ne = int(m.sum()), int(len(np.unique(eid[m]))) if m.any() else 0
        if n == 0:
            print("%-12s %9d %9d %9s   %-20s %9s"
                  % ("<= %.1f s" % thr, 0, 0, "null", "null", "null"))
            out["ladder"].append({"threshold_s": thr, "n_windows": 0,
                                  "n_episodes": 0, "rate": None, "ci": None})
            continue
        rate = float(ge[m].mean())
        lo, hi = cluster_ci(ge[m], eid[m], a.n_boot, a.seed)
        label = "all" if not np.isfinite(thr) else "<= %.1f s" % thr
        print("%-12s %9d %9d %9.4f   [%7.4f, %7.4f] %9.4f"
              % (label, n, ne, rate, lo, hi, float(fge[m].mean())))
        out["ladder"].append({
            "threshold_s": (None if not np.isfinite(thr) else thr),
            "label": label, "n_windows": n, "n_episodes": ne,
            "hold_v0_ge_human": rate, "ci": [lo, hi],
            "frozen_ge_human": float(fge[m].mean()),
            "ci_upper_below_ceiling": bool(hi < 0.30)})

    n2, ne2 = int(differs.sum()), int(len(np.unique(eid[differs])))
    if n2:
        r2 = float(ge[differs].mean())
        lo2, hi2 = cluster_ci(ge[differs], eid[differs], a.n_boot, a.seed)
        print("%-12s %9d %9d %9.4f   [%7.4f, %7.4f] %9.4f"
              % ("term-differs", n2, ne2, r2, lo2, hi2,
                 float(fge[differs].mean())))
        out["continuity"] = {"population": "a safety term differs (the banked "
                                           "signal-window definition)",
                             "n_windows": n2, "n_episodes": ne2,
                             "hold_v0_ge_human": r2, "ci": [lo2, hi2]}

    print("")
    print("-- CONTROLS")
    all_rate = float(ge.mean())
    print("   all-window rate      %.6f   (channel control vs the banked 0.441780)"
          % all_rate)
    print("   frozen, all windows  %.6f   (must stay far below the human)"
          % float(fge.mean()))
    out["controls"] = {"all_window_rate": all_rate,
                       "banked_all_window_rate": 0.441780259484316,
                       "abs_err": abs(all_rate - 0.441780259484316),
                       "frozen_all_windows": float(fge.mean())}
    print("   abs err vs banked    %.6f" % out["controls"]["abs_err"])
    print("")
    print("⛔ THIS PANEL SELECTS NOTHING. Choosing the gate's population is a PI / "
          "Master-Mind decision.".encode("ascii", "replace").decode())

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
