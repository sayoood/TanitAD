# EVIDENCE: deliberate regression on REAL data (COMBO_TEST_MUTATION=skip_retime, EMA-30k, eval gains) -- OUTCOME VOID (controls failed: C-K1R)
Run with script sha256 35e0c3a25240... (the first run's version; the later edits only add the additivity block and the K=2 label wording). Written to a scratch dir, never to the production directory (the script refuses).

# COMBO_ENS_RETIME -- step30000ema_unc / gains: eval

Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  Estimator: paired episode-cluster bootstrap, models FIXED, B 2000, seed 0: answers 'another draw of EPISODES' only; the disjoint draw sets A/B are the inference-run replicate; NOT another training run (single training seed).
Status: EXPLORATORY (eval-fitted gains, post-hoc arm, milestone checkpoint); bar of record = single-draw `os`.
Bar of record: single-draw `os` (BAR-R8-E1, vs `ha0_ext_pose`, D-R8-E1-REF).  A combo is a NEW arm (inference cost up to xK).

prereg sha256 `a888e7cc99c3346dcbe71244f11c1d6f6ce654bccec9aeb7ec764b7050701287`; script sha256 `35e0c3a25240`; gains `C:\Users\Admin\qland\hold_r8batt\stage\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-10-15-refcv8-milestone-battery\raw\a0k0_retime\FROZEN_GAINS_RETIME_ema30k.json` sha256 `4ecde275554b`.

## OUTCOME (K=4, sets A and B): **VOID (controls failed: C-K1R)**

* qualifier: EXPLORATORY (eval-fitted gains, post-hoc arm, milestone checkpoint); bar of record = single-draw `os`
* qualifier: EVAL-FITTED GAINS (in-sample on the eval episodes; EXPLORATORY ONLY)
* qualifier: tree step30000ema_unc
* qualifier: single training seed
* qualifier: S6 K=4 UNTESTED (S6 draws exist for inference seeds 0,1 only)
* qualifier: transfer unmodelled: gains fitted on single-draw plans
* **VOID: controls failed -- ['C-K1R']** (would have been {'label': 'NO-STACK', 'reasons': ['not separated better (beyond 2x floor, both sets) than: ENS-4 alone', 'ACC-LAT S2 K=4 not a non-vacuous HOLD', 'ACC-LAT S6 K=2 not a non-vacuous HOLD (or UNAVAILABLE)']})

## Controls

| control | PASS |
|---|---|
| C-MASK | True |
| C-ALIGN | True |
| C-ID | True |
| C-DIST | True |
| C-R0926 | True |
| C-G1 | True |
| C-G0 | True |
| C-K1 | True |
