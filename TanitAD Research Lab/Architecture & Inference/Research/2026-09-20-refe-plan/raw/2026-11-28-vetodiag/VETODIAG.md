# VETODIAG (navtest DIAGNOSTIC on the banked C1 T1 harness-exact table; NOT a result, NOT an adoption)
Evidence: MEASURED, E:/Projects/TanitAD-artifacts/vetodiag/VETODIAG.json (script vetodiag.py). Boot = p5c_read.Boot (136 logs, 10,000 draws, seed 20260927). Checks passed: argmax agg32(s_plan)==k_ship 12,146/12,146; mean ship PDMS 92.5548.
Column order of s_plan: nc, dac, ep, ttc, c, ddc (safehead_data.py ROW_NAMES).

1. Failure set: zeroed (ship nc==0 or dac==0, a passing alternative exists) n=336 (NC 138, DAC 198, both 0). Regret to best passing proposal = 2.723 PDMS pts. 21 more tokens have a zero ship pick and no passing alternative (336+21=357, the expected figure).
2. Scorer knew? Weakly. AUC of p_fail=1-s(nc)s(dac) = 0.763. Median p_fail on zeroed 0.024 vs 0.0043 non-zeroed; zeroed 95th pct 0.40, non-zeroed 95th pct 0.077. Median s(dac) 0.989 on zeroed (non-zeroed 0.999): the scorer is confident on most failures.
3. Veto-only, 5-fold log-CV OOF (fold taus 1.0,1.0,1.0,0.1,1.0; i.e. almost no switching): 16 switched, 0 zeroed hit (precision 0, recall 0). d PDMS -0.0255 [-0.0567, -0.0013]; d EP -0.0331 [-0.0784, -0.0014]; NC 0 [0,0]; DAC -0.0082 [-0.0250, 0]; TTC -0.0165 [-0.0420, 0]; C 0 [0,0].
   In-sample curve (NOT a result): d PDMS <= 0 at every tau (tau 0.001: -0.28 with 2,159 switches; tau 0.1: -0.022, 52 switches, precision 0.15; tau>=0.35: 0 switches). No tau improves PDMS or EP.
4. Oracle ceilings (descriptive): (a) perfect detector + current agg32 ranker over passing proposals: d PDMS +2.340 [1.927, 2.791], d EP +2.548 [2.115, 3.026], NC +1.132, DAC +1.630, TTC +0.420, C 0; 0 of 336 switches worse. (b) true-best passing: d PDMS +2.723 [2.265, 3.229].

Implication: the ceiling is large (+2.3 to +2.7) but the scorer's own NC/DAC logits cannot find the failures (AUC 0.76, no tau beats 0), so a veto-only lever on C1's own heads is not worth a pre-registration; only a detector with new information (e.g. the SAFE-HEAD / independent safety head) could be, and its bar is AUC far above 0.76.
