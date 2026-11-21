"""PROBE 1 (DESCRIPTIVE, eval dumps, read-only): an EXACT additive decomposition of the L2 excess (curvature MAE at GT pair speed >= 5.2 m/s, arm - P10)
over the 29 step-pairs of the executed track, grouped by the pair's position inside its 2 s plan segment.
  pair p = 10 n + q compares the chords into plan-n knots q+1 and q+2 (q = 0..8: inside plan n); q = 9 is the SEAM (plan n's last chord vs plan n+1's first).
  w_cross of HACC is [1, 0, 1, 0, .37, .55, ...] on knots 1..6, so q = 0..3 are the pairs whose three knots straddle the alternating weights.
Per window the L2 value = sum_p share_p (control: equals lat_rows to 1e-12), so arm - P10 = sum over groups of paired deltas, each with the episode-cluster bootstrap.
python probe1_pair_profile.py > probe1.log"""
import json
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402

GROUPS = {"plan0_early_q0-3": [0, 1, 2, 3], "plan0_rest_q4-8": [4, 5, 6, 7, 8],
          "seams_q9": [9, 19],
          "own_early_q0-3": [10, 11, 12, 13, 20, 21, 22, 23], "own_rest_q4-8": [14, 15, 16, 17, 18, 24, 25, 26, 27, 28]}
out = {"what": "L2 (curvature MAE, GT pair speed >= 5.2 m/s) arm - P10, exact per-pair decomposition; T1-kin m=10, eval-141", "cfg": {}}
for cfg in ("causal", "logged", "plainB_logged", "strict", "hlatB"):
    out["cfg"][cfg] = {}
    for s in (0, 1):
        d = C.load(cfg, s)
        G, X, F = d["g"], d["cl"], d["fl"]
        pa, pf = C.pair_err(X, G), C.pair_err(F, G)
        ca, cf = C.check_against_rows(X, G, pa), C.check_against_rows(F, G, pf)
        ok = (pa["c"] > 0) & (pf["c"] > 0)
        tot = C.boot(pa["sh"].sum(1)[ok], pf["sh"].sum(1)[ok], d["eid"][ok])
        per_pair = (pa["sh"][ok] - pf["sh"][ok]).mean(0)
        grp = {}
        for g, ps in GROUPS.items():
            grp[g] = C.boot(pa["sh"][ok][:, ps].sum(1), pf["sh"][ok][:, ps].sum(1), d["eid"][ok])
        # where the arm's excess lives, per q (pooled over plans), as a share of the total
        byq = [float(per_pair[[p for p in range(29) if p % 10 == q]].sum()) for q in range(10)]
        out["cfg"][cfg][f"s{s}"] = {"control_vs_lat_rows": {"arm": ca, "p10": cf}, "n_windows": int(ok.sum()), "L2_delta": tot,
                                    "per_pair_contrib": [round(float(x), 6) for x in per_pair], "by_q": [round(x, 6) for x in byq], "groups": grp,
                                    "fast_pairs_per_window_mean": float(pa["fast"][ok].sum(1).mean())}
        print(f"== {cfg} s{s}: controls arm {ca['PASS']} ({ca['max_abs']:.1e}) p10 {cf['PASS']} | n {ok.sum()} | L2 delta {tot['delta']:+.4f} [{tot['lo']:+.4f}, {tot['hi']:+.4f}]")
        print("   by q (0..9; 9 = seam):", " ".join(f"{x:+.4f}" for x in byq))
        for g, r in grp.items():
            print(f"   {g:18s} {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] sep {r['separated']}")
json.dump(out, open("C:/Users/Admin/lcdiag/probe1.json", "w"), indent=1)
