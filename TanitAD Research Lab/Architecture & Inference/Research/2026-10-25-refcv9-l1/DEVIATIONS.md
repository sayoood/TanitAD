# DEVIATIONS from PREREG_L1.md (sealed 2026-10-25 10:10, sha256 41ba4b2f...)

1. 2026-10-25 ~11:30 -- K4 at inference seed 0 mis-specified (wrong reference artifact: the banked dump_s0 is the FULL arm-set draw).
   Fixed by PREREG_L1_AMENDMENT1.md (sealed before any fit/score; sha256 in PREREG_L1_AMENDMENT1.sha256).
   Code: l1_fit_score.py bank-control block -- K4 reference at s0 = r8oracle/run_48852ema_pf/dump_s0 (os-only child), K4b binding only at
   s1, K6 gets a PASS flag (<= 1e-3 relative); l1_verdict.py adds K6 to the control set. No arm, bar, split or grid changed.
2. 2026-10-25 ~11:45 -- AMENDMENT 2 (Master Mind request): adds a secondary braking-AUC read on S6 (L2 definition reused byte-identical).
   Code: l1_fit_score.py now banks every arm's plan per surface/seed (result/arms_{S2,S6}_s{seed}.npz; scoring unchanged); new l1_roc.py.
   Sealed before any fit/score (PREREG_L1_AMENDMENT2.sha256). No arm, bar, split or grid changed.
