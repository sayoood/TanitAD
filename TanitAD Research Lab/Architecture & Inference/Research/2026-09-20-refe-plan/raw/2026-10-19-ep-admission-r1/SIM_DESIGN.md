# SIM_DESIGN: false-refusal rate and power of the EP-ADMISSION Revision-1 gate (EP-ADMISSION-R1, Research Lab agent, 2026-10-19)

**Written and hashed BEFORE any simulation was run.** The simulator (`code/ep_admission_sim.py`) prints this file's sha256 into its log and refuses to start if the file is missing. Nothing below was tuned on a simulation output. Evidence class of every rate this design produces: **SIMULATED on the real estimator** (not MEASURED on a candidate).

## 1. Question

`DRAFT_PREREG_EP_ADMISSION.md` REVISION 1 states two numbers as ESTIMATED and asks the sealed version to SIMULATE them on the real estimator: A1-r1 refuses a calibrated candidate about 5 % of the time across 2 seeds, and refuses an EP-R1-sized over-prediction (+0.135) with power about 1. We measure, for the rule exactly as coded in `ep_admission.py` (`a1_verdict_r1`, `a2_verdict_r1`, `a1_verdict_r1_seeds`, `a2_verdict_r1_seeds`):

1. the **false-refusal rate** of A1-r1 and of A2-r1 (and of the two together) on CALIBRATED replicates;
2. the **power** of A1-r1 at a planted over-prediction of **+0.135** (EP-R1's size) and **+0.05**, plus a short curve (+0.01 / +0.02 / +0.10) for the shape;
3. (extra, cheap) the power of A2-r1 at a planted under-prediction of risk (observed / predicted = 1.25 and 1.5).

The rule under test (R1): FAIL-A1 iff on ANY seed D_M has lower CI > 0 OR point > +0.05. FAIL-A2 iff for any of NC / DAC / TTC on ANY seed the observed / predicted unsafe mass at the candidate's picks has lower CI > 1.0. Seeds: 2.

## 2. Data (all real, nothing re-selected)

* S0 bank: `pickrule/run/prepare/arrays.npz` (15,808 sets, 117 logs; sha256 must equal `epreliab_lib.ARRAYS_SHA256` = `29130bcd...`; `ep_admission.py selftest-real` proved `load_bank_units` equals it bit for bit).
* The two real candidate seeds: EP-R1's EP-led composite rows `ep_OOF_s0`, `ep_OOF_s1` of the same file (the candidate whose over-prediction motivated the gate). A candidate's rows are P2's `z6` with the EP column replaced; NC / DAC / TTC rows are P2's.
* **Geometries** (how many picks differ from P2's): `G78` = EP-R1 as is (seed 0: 12,314 of 15,808 changed, 77.9 %; this is the PRIMARY geometry); `G25` and `G05` = the same two EP rows used on a deterministic random subset of sets (per-set Bernoulli q = 0.32 and 0.065, rng seed 20261029, the same subset for both seeds), giving about 25 % and 5 % changed picks. The realised changed counts are recorded.

## 3. The estimator is the real one, applied to per-log sufficient statistics

The decision statistics are ratios of per-log sums (`pickrule_lib.Stats` bincounts set-level terms per log, then `fastneg_lib.diff_ci` / `ratio_ci` bootstrap the 117 logs). The simulator computes the per-log sums once and calls **the landed `fastneg_lib.diff_ci` and `fastneg_lib.ratio_ci` themselves** (sha-pinned through `epreliab_lib`) on each replicate, with the **registered draws `boot_draws(117, 10000, 20260927)`** (paired log-cluster bootstrap, B = **10,000**, seed 20260927, percentile 2.5 / 97.5). Because the per-log shortcut makes B = 10,000 cheap, **no B = 2,000 shortcut is used** and none needs to be shown equal.

Equivalence is CHECKED, not assumed (section 7, PC3): the identity draw reproduces `epreliab_lib.core_stats` / `reliability_stats` (the functions `ep_admission.analyze_candidate` calls) bit for bit on the real data, and 3 materialised set-level log-resamples reproduce the shortcut to 1e-12.

## 4. How a CALIBRATED replicate is generated

**Per-log components** (117-vectors) for each seed and geometry, from the real data: split the sets into UNCHANGED (candidate pick = P2's pick) and CHANGED. EP terms `y` (label) and `s` (the candidate's sigma) at: the shared pick of unchanged sets (`Y_u`, `S_u`), P2's pick of changed sets (`Y_pc`, `S_pc`), the candidate's pick of changed sets (`Y_ec`, `S_ec`). So `M_p = (Y_u + Y_pc) / (S_u + S_pc)`, `M_e = (Y_u + Y_ec) / (S_u + S_ec)`, `D_M = M_p - M_e`. A2: for h in {NC, DAC, TTC}, per-log observed unsafe mass `OBS_h = sum(1 - y_h)` and predicted `PRED_h = sum(1 - sigma_h)` at the candidate's picks.

**Calibration by construction.** Rescale the PREDICTED side only, by three EP scalars and one scalar per head, so that every population ratio is exactly 1:

* `a_u = sum Y_u / sum S_u`, `a_p = sum Y_pc / sum S_pc`, `a_e = sum Y_ec / sum S_ec`; `S'_u = a_u S_u`, `S'_pc = a_p S_pc`, `S'_ec = a_e S_ec`. Then `M_p = M_e = 1`, `D_M = 0` exactly (observed / predicted EP mass = 1 within each pick group).
* `k_h = sum OBS_h / sum PRED_h`; `PRED'_h = k_h PRED_h`. Then observed / predicted unsafe mass = 1 exactly for each head.

What is kept REAL: which sets and slots are picked, the labels (so the per-log label noise and its clustering, the thing an i.i.d. Monte Carlo got 3.4x too small), the heteroscedastic spread of sigma, the sparse unsafe events. What is removed: only the average bias of the predictions. This is the estimator's own definition of "calibrated at the candidate's picks".

**A replicate = a draw of logs** (generator A, PRIMARY): `idx = rng.integers(0, 117, 117)` with `rng = default_rng([20261027, rep])`; every per-log vector of both seeds and all heads is indexed by the SAME `idx` (the two seeds see the same logs, so their decisions are correlated as they would be in practice). The inner estimator then bootstraps the 117 drawn logs as its clusters. This is a nested (double) bootstrap: its false-refusal rate is the frequency with which the CI rule fires when another draw of logs is taken from a population whose true D_M is 0 and true ratios are 1.

**Cross-check generator B** (wild cluster sign-flip, A1 only, `G78`): `Y*_l = S'_l + w_l (Y_l - S'_l)` per component with one Rademacher `w_l` per log (`default_rng([20261028, rep])`), predictions `S'` unchanged. No duplicated logs, mean exactly calibrated by symmetry, per-log residual structure and its seed-to-seed coupling kept. If A and B disagree materially the disagreement is reported and the larger false-refusal rate is the one quoted for sealing.

## 5. PLANTED over-prediction (power)

Population over-prediction `delta` in D_M units, on top of the calibrated replicate, by inflating the candidate's predicted EP mass at its CHANGED picks only (the lib's C-POWER convention): `S''_ec = f S'_ec` with `f = ((sum Y_u + sum Y_ec) / (1 - delta) - sum Y_u) / sum Y_ec` so that `M_e = 1 - delta` and the population `D_M = +delta` exactly; `M_p` stays 1. The same `idx` / `w` draws and the same labels as the calibrated replicate (common random numbers: condition differences are not Monte Carlo noise). `delta` in {0.01, 0.02, 0.05, 0.10, 0.135} on `G78`; {0.05, 0.135} on `G25`, `G05`. The implied `f` is recorded; on `G25` and `G05` the same `delta` needs a larger per-pick inflation (a 5 %-changed candidate over-predicting each changed pick by far more); that is reported, not hidden.

A2 plant: `PRED''_h = k_h PRED_h / rho` so observed / predicted = `rho` in {1.25, 1.5}, one head at a time (NC, DAC, TTC), `G78`, both seeds.

## 6. Conditions, replicate counts, outputs

* **N = 1,000 replicates per condition** (>= 200 required; the Wilson 95 % half-width at p = 0.05 is 1.3 pp). Fixed by this design; lowered only if the measured cost forces it, and then never below 200 and said so in the result.
* Conditions: for each of `G78`, `G25`, `G05`: calibrated (A1 and A2 decided on every replicate) and the A1 plants; `G78`: A2 plants; generator B on `G78`: calibrated and `delta` in {0.05, 0.135}.
* Cost is MEASURED on 3 replicates first; cap about 3 CPU-h; a run over about 4 minutes is launched detached and the session ends with `HANDOFF.md`.
* Reported per condition: the rule's refusal rate with a **Wilson 95 % interval**; per seed; seed-to-seed co-failure (P(both), P(either), and the rate expected if the seeds were independent); for A1 which conjunct fired (lower CI > 0 only / point > 0.05 only / both); for A2 per head; and, on the SAME replicates, the DRAFT rule's refusal rate (A1: upper CI <= +0.02; A2: CI overlaps [0.8, 1.25]) so the revision's effect is read on identical data. Also the replicate mean / SD of D_M point and of the four reliability ratios.

## 7. The PLANTED check that the generator is calibrated, and the other controls (all gating; any failure VOIDS the simulation)

* **PC1, calibrated by construction.** Over the 1,000 calibrated replicates (generator A and B, each geometry): the mean of `M_p` and of `M_e` is within **0.01** of 1, the mean `D_M` point within **0.005** of 0, and the mean observed / predicted unsafe mass of NC, DAC, TTC within **0.05** of 1 (a looser bar for the sparse unsafe events: NC has about 195 predicted unsafe pairs over 117 logs, so a ratio of sums has a finite-sample bias). Tolerances are fixed here, before any output.
* **PC2, the check has teeth (RED arms).** The same check run on (a) the UNCALIBRATED replicates (the real labels against EP-R1's own sigma) must FAIL (mean `D_M` near +0.135), and (b) on the +0.135 plant must FAIL the "calibrated" form and PASS the "planted delta" form (|mean D_M - delta| <= 0.005). A check that cannot tell 0 from 0.135 would void the study.
* **PC3, the shortcut IS the estimator.** (i) identity draw: per-log pipeline `D_M` and the three A2 ratios equal `epreliab_lib.core_stats` / `reliability_stats` on the full real arrays to 0.0 for both seeds of `G78` (and equal the landed EP-RELIAB-PICKS numbers: D_M +0.1349 [+0.1114, +0.1640] seed 0, +0.1331 [+0.1101, +0.1618] seed 1); (ii) three materialised set-level log-resamples equal the shortcut to 1e-12; (iii) one full `ep_admission.analyze_candidate` call on `G78` seed 0 equals the shortcut's D_M and A2 CIs to 0.0.
* **PC4, the decision code is the module's.** Verdicts come from `ep_admission.a1_verdict_r1` etc. (its sha256 logged); on every replicate an inline restatement of the prose rule (`lo > 0 or point > 0.05`; `lo > 1.0`) must agree with the module's verdict (0 disagreements).
* **PC5, determinism.** The first 20 replicates of one condition are recomputed in a second, separate process and are bit-identical.
* **PC6, the identity candidate.** P2 as the candidate (all per-log differences zero) gives `D_M` exactly [0, 0, 0] and ADMIT-A1 on every replicate.

## 8. Flags committed in advance (what would "change the sealing")

These are reporting thresholds only; I do not change the rule, the Master Mind seals it.

* **F1**: the two-seed A1-r1 false-refusal rate on `G78` has a Wilson **upper** bound above **10 %** (the draft says about 5 %).
* **F2**: the two-seed A2-r1 false-refusal rate (three heads x two seeds = six one-sided tests) has a Wilson **lower** bound above **10 %**.
* **F3**: A1-r1 power at +0.135 on `G78` has a Wilson lower bound below **95 %**, or at +0.05 below **80 %**.
* **F4**: generator A and B false-refusal rates differ by more than 3 pp.
* **F5**: any control PC1-PC6 fails.
Whatever the numbers, the rates are reported with their intervals.

## 9. What these rates are NOT

* The interval answers only "another draw of LOGS" (`H-ESTIM-SEED-1`): the two seeds here are two EP-R1 trainings on the same logs, but a real candidate's run-to-run training variance beyond the 2 seeds, and its inference variance, are NOT in the false-refusal rate.
* Calibrated here means D_M = 0 and observed / predicted = 1 at the picks on average. A candidate that is calibrated on average but wrong in a log-concentrated way (a few logs carry all the over-prediction) is not simulated; the planted over-prediction is spread over all changed picks, which is the EASY case for detection.
* P2 is in-sample on S0, which biases the realised side toward OVER (conservative for A1); the planted and calibrated replicates inherit the real bank's P2 labels, so this is in the data, not modelled.
* Outer resampling draws 117 logs with replacement from the 117 real ones; replicates therefore contain duplicate logs (about 74 distinct). Generator B has no duplicates and is the cross-check for that.
* Tier: SELECTION, SEEN scenes, label-lookup PDMS; T0-class. Nothing here is a navtest or driving claim.
