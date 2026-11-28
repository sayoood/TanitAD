#!/usr/bin/env python3
"""Post-fit DIAGNOSTIC (changes nothing, recomputes no rule): how far can a p_DAC threshold reach on the RL-val B picks?  Explains why V(tau) is flat."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import selrule as SR
RV, RR = SR.RV, SR.RR
keys, S, tg, st = SR.load_rlval("C:/Users/Admin/rls1b_dl/c0/final/scores_step0_base.npz", sys.argv[1])
P = RV.sigmoid(S); B = SR.picks(("B",), P)
ar = np.arange(len(B))
pd = P[ar, B, SR.COL["DAC"]]; true_dac = tg[ar, B, SR.COL["DAC"]]
bad = true_dac == 0.0
out = {"n_sets": int(len(B)), "B_picks_with_true_DAC0": int(bad.sum()),
       "p_DAC_at_B_pick_quantiles_all": {q: float(np.quantile(pd, q)) for q in (0.001, 0.01, 0.05, 0.5)},
       "p_DAC_at_B_pick_quantiles_among_DAC0_picks": {q: float(np.quantile(pd[bad], q)) for q in (0.05, 0.25, 0.5, 0.75)},
       "sets_where_B_pick_has_p_DAC_below_tau": {str(t): int((pd < t).sum()) for t in SR.GRID_TAU},
       "DAC0_picks_where_B_pick_has_p_DAC_below_tau": {str(t): int((bad & (pd < t)).sum()) for t in SR.GRID_TAU},
       "sets_where_some_proposal_clears_tau": {str(t): int(((P[..., SR.COL["DAC"]] >= t).any(1)).sum()) for t in SR.GRID_TAU},
       "reading": "a veto can only move a pick whose own p_DAC is below tau; the DAC=0 picks are mostly CONFIDENT (high p_DAC), which no threshold reaches"}
json.dump(out, open(os.path.join(HERE, "FIT_DIAGNOSTIC_VETO_REACH.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
print(json.dumps(out, indent=1, sort_keys=True))
