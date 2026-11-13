# PREREG_V8F_PROOF addendum 5: CUEPROBE, is the stop cue readable from the frozen tokens, and what does the verified shield gain with it (frozen 2026-11-13 01:03:37 dev-box clock, before its data)

RULESHIELD (addendum 3, read) left one lever: with the logged rest distance the verified shield gains 3.51 / 3.37 m on stop windows (2-6 s ADE, T1-kin) and removes every overrun; with the stop position the model emits it gains nothing (median error 11.8 / 9.6 m, flag on 10 / 18 of 131 stop windows). This card measures the cue itself, with zero training: closed-form probes on the cached frozen tokens, cross-fitted by episode, then fed to the same verified shield. Tier T0 for the probe reads, T1-kin on logged frames for the shield reads. A diagnostic of what the input interface carries; not a deployable arm and never driving performance. STRATEGIC: PAUSED (PI item 25).

## Data (all banked, eval-141; nothing is trained)
* Windows: the 2,399 first-tick windows of RULESHIELD (2,365 with label, 131 stop windows), same plans (banked dumps of stopbal_BAL-NULL_s0 / _s1), same definitions (rule/RULE_DEFS.md). Rest distance d_true = the RULESHIELD along-path arc to the logged rest point.
* Inputs at the now-frame of each window only (no later frame): E = ego f8 (measured ego motion, a model input by PI ruling); V40 = pool4x10 tokens; V160 = pool8x20 tokens; V40+E; V160+E. Token-to-frame mapping as in res/RES_DEFS.md item a.
* Probe: the RESPROBE kernel ridge (same code path, same alpha grid, alpha chosen inside the training folds), 5 folds by episode. Every prediction for a window comes from a model that never saw its episode.

## Reads
1. FLAG: stop-window probability from each input set; AUROC over the 2,365 labelled windows, episode-cluster bootstrap B 2,000, paired differences against E.
2. DIST: rest distance on the 131 stop windows (probe fitted on stop windows of the training folds); median absolute error, R2, both against two floors: the training-fold median (constant) and the E set.
3. SHIELD with the probe cue, the verified shield of RULESHIELD unchanged (a = 3.0 m/s2, dt 0.2 s). Four arms, each against NULL, paired episode-cluster bootstrap:
   * PF-PD: active where the flag probability >= tau; d_0 = d_hat - q.
   * OF-PD: active on the true stop windows; d_0 = d_hat - q (isolates the distance error).
   * PF-OD: active where the flag probability >= tau; d_0 = d_true on stop windows, d_hat - q elsewhere (isolates the flag error).
   * OF-OD = the banked ORACLE (control k2 below).
   tau: per training fold, the smallest threshold with false-positive rate <= 0.02 on the training non-stop windows. q: split-conformal margin, the 0.9 quantile of (d_hat - d_true) on the training folds (inner cross-fitted residuals), so that d_0 <= d_true with probability >= 0.9 under exchangeability of episodes. The input set used in the shield arms is the one with the lowest inner-fold DIST error (chosen inside training folds, never on the held-out fold).
4. Coverage: share of held-out stop windows with d_0 <= d_true, against the nominal 0.9.
Four metric families for every shield arm, as in RULESHIELD: LONGITUDINAL (speed error, overrun share and mean), LATERAL (cross-track, heading, curvature, yaw rate), TACTICAL (flag hit rate and false-positive rate at tau, confusion counts), STRATEGIC PAUSED. Stop windows and non-stop windows are reported apart, plus all windows.

## Outcomes (fixed now)
* FLAG: VISION-ADDS if AUROC(V160+E) - AUROC(E) >= 0.02 with the paired interval excluding 0; VISION-FLAT otherwise. Also report V40+E against E the same way.
* DIST: CUE-USABLE if the best set has median absolute error <= 5.0 m and is better than both floors with the paired interval excluding 0; CUE-COARSE if 5.0 to 8.0 m; CUE-ABSENT if above 8.0 m or not better than the floors.
* SHIELD (arm PF-PD): CUE-PROBE-GAINS if the stop-window 2-6 s ADE change is negative with the interval excluding 0, its size is at least 0.30 of the ORACLE gain at that seed, and the non-stop 0-6 s ADE change has an upper interval end <= +0.05 m; CUE-PROBE-COSTS if the non-stop cost bar fails; CUE-PROBE-NO-GAIN otherwise. Both seeds are reported; the outcome needs both.
* Coverage: COVERAGE-HOLDS if the held-out share is inside the binomial 95 % band around 0.9 for n = 131 (0.84 to 0.95), else COVERAGE-FAILS (named, with the direction).

## Controls (each must read its known value, or the read is not valid)
* k1 shuffled labels: FLAG and DIST labels permuted across episodes; AUROC within 0.5 +/- 0.05 and DIST not better than the constant floor.
* k2 OF-OD through this pipeline equals the banked RULESHIELD ORACLE numbers (stop-window 2-6 s ADE change to 1e-9 m, overrun 0).
* k3 with no active window the plan is bit-identical to NULL.
* k4 leak guard: an assertion that every token row used is at or before the now-frame of its window; a deliberate regression (tokens taken 2.0 s later) must raise FLAG AUROC, otherwise the guard is not proven to see a leak.

## What each outcome commits
* CUE-USABLE or CUE-PROBE-GAINS: the next arm is a trained stop-distance output with the conformal margin, behind the launch gate (pod).
* CUE-ABSENT at V40 and V160: theorem T5 stands for the stop cue; the next arm is the un-pooled 16x40 grid and a native-resolution centre crop on the pod, same probes.
* VISION-FLAT with CUE-ABSENT: the cue is not in the pooled input at all; no head on these tokens can supply it.

## Limits stated in advance
131 stop windows from 141 episodes: intervals will be wide. Closed-form probes have no training seed; the fold assignment is the only free choice, so the read is repeated for 5 fold seeds and the spread is reported beside the interval. The interval answers another draw of episodes. Cross-fitting on the evaluation clips is a decodability diagnostic, not a model. Deviations are recorded in the verdict, never by editing this card.
