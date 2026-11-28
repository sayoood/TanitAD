# HANDOFF DOMPREM (PREREG_V8F_PROOF_ADD14 sha 70b48463...; MEASURED, T1-kin, zero training; STRATEGIC: PAUSED, PI item 25)

Controls d1..d5: ALL PASS (d1 both pairs and seeds exact: 1,305 / 1,094, joint counts, RIGHT counts). Run exit 0. Tests: 9 OK.

**Card outcomes (pair 1 decides)**
* Q1: DOM-PREMISE-PARTIAL. PREMISE-HOLDS at both seeds only in v0 [8,15) (m_HIER +0.153 [+0.084, +0.229] s0; +0.118 [+0.022, +0.215] s1; 923 windows, 85 episodes). POOLED is UNDECIDED at both seeds (m +0.018 [-0.045, +0.079]; -0.000 [-0.061, +0.059]). [0,2) is PREMISE-FAILS at both seeds, [15,inf) FAILS at s0 and UNDECIDED at s1. Pair 2: DOM-PREMISE-NOT-MET.
* Q2 (printed): S < 1 in every context (pooled S 0.149 [0.047, 0.232]; v0 [8,15) 0.263), so NOT-SEPARATED everywhere; r90_HIER 8.7 to 15.4 m against tol 2.0, so EXEC-WIDE everywhere. The geometry premise is not met in any context.
* Q3: HIT-UNDECIDED. SELECTION-CRITICAL pooled, tol 2.0: HIER - FLAT +0.032 [+0.011, +0.053] at s0, +0.012 [-0.007, +0.032] at s1. HIER minus best single plan on those windows: +0.030 [-0.006, +0.063] / +0.028 [-0.007, +0.061] (plan share 0.047, interval holds the disc fixed). Hit rates are low: HIER 0.077 / 0.075 on critical windows.

**Not clean**: v0 [15,inf) D point (0.870 m) sits outside its bootstrap interval (top-2 cells flip between draws); reading choice 2 (fixed disc for the best-plan interval) is conservative for the hierarchy.

Deviations / reading choices: see DOMPREM_DEFS.md section 3 (13 items, none moves a threshold, bar or outcome rule).

## Manifest (all in F:/TanitAD-scratch/v8proof/domprem/)
domprem.py, test_domprem.py, RESULT_DOMPREM.json, VERDICT_DOMPREM.md, DOMPREM_DEFS.md, run.log, tests.log, HANDOFF.md, dbg.py (scratch, not part of the read).
