# SPEEDAVG_READ: PREREG_V8F_SPEEDAVG executed as written

**Outcome (the PREREG table applied verbatim): MIXED.**

Tier: T1-class open loop (own held outputs; no recorded future action). Estimator: paired episode-cluster bootstrap (taniteval/ci.py), B=2000, 141 episodes / 2399 windows; answers 'another draw of EPISODES' only; BLIND to training variance (H-ESTIM-SEED-1): the second seed is the replicate; a separated CI at two seeds is necessary, not sufficient, for a lever claim.

Committed next step, quoted from the PREREG: *"reported per seed; seeds before windows"*

| read | seed 0 | seed 1 | bar |
|---|---|---|---|
| PRIMARY ADE6 SPEEDAVG - X_s | -0.0414 [-0.0730, -0.0110] * | +0.0189 [-0.0116, +0.0483] | <= -0.1207 and separated at BOTH |
| CAPTURE | 0.058 | -0.028 | beside |
| SECONDARY not separated-worse | True | False | both |

## The outcome table (verbatim from the PREREG)

| outcome | condition | next (committed) | this read |
|---|---|---|---|
| **VARIANCE PAYS** | PRIMARY and SECONDARY at both seeds | the two-seed speed average joins `forward_infer`; next, distil it into one T (a train-time arm under its own PREREG) so the deployed model needs one T |  |
| **PARTIAL** | PRIMARY < 0 and separated at both seeds but above −F | report CAPTURE; next: a temporal average of T's own speed cells over its last ticks (single model, zero training), under its own PREREG |  |
| **BIAS, NOT VARIANCE** | PRIMARY not separated at either seed | T's speed error is systematic; next: a T-side speed-head arm that targets bias (a re-weighted or class-balanced speed loss), under its own PREREG |  |
| **MIXED** | anything else | reported per seed; seeds before windows | **YES** |

## Controls

PASS: True
* seed 0: control 1 True (max dev 0.0), control 2 True (max dev 4.31e-06 m), control 3 True ({'kindx_own': 7.287375651809924e-06, 'banked_standard_eval': 2.3953600714499146e-05}), guards True
* seed 1: control 1 True (max dev 0.0), control 2 True (max dev 1.46e-05 m), control 3 True ({'kindx_own': 1.2244592419108358e-05, 'banked_standard_eval': 1.2244592419108358e-05}), guards True
* cross-seed (both avg arms averaged the same pair): True

Reading: PREREG_V8F_SPEEDAVG: MIXED; PRIMARY (avg - own, ADE6 m): s0 -0.0414 [-0.0730, -0.0110], s1 +0.0189 [-0.0116, +0.0483] ns vs -F = -0.1207; CAPTURE s0 0.06, s1 -0.03; SECONDARY not separated-worse: s0 True, s1 False.
