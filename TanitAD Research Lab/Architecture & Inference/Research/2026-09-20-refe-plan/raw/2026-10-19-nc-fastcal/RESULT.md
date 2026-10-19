# RESULT: NC-FASTCAL (Research Lab agent, 2026-10-19). 0 GPU, CPU, selection tier, SEEN scenes, P2 IN-SAMPLE

**Registration:** `PREREG_NC_FASTCAL.md` sha256 `4db61ce1…cc4f`, sealed 2026-10-19T10:24:44+02:00 before any array of this arm was opened; **no amendment**. Every stage refuses on a hash mismatch. MEASURED, SELECTION tier (T0-class), SEEN scenes, P2 in-sample; paired log-cluster bootstrap 10,000 / seed 20260927 (answers "another draw of LOGS" only, `H-ESTIM-SEED-1`). Label-lookup PDMS, not a navtest number.

## 0. Headline

**NO-RECOVER on both seeds; C-NULL silent (0 of 5), so the read is not VOID.** A per-stratum Platt calibration of P2's NC logit, fitted on training folds only, changes nothing that matters: the calibrated EP-led composite still loses NC-unsafe picks and PDMS against P2's rule, and by the same amount as the uncalibrated one.

| seed | dNC vs P2 (NC-unsafe pick rate) | (i) upper < +0.001 | dPDMS x100 vs P2 | (ii) lower >= 0 |
|---|---|---|---|---|
| 0 | **+0.0029 [+0.0009, +0.0054]** | False | **-1.205 [-1.542, -0.882]** | False |
| 1 | **+0.0030 [+0.0009, +0.0055]** | False | **-1.212 [-1.549, -0.890]** | False |

Beside it, the uncalibrated composite: dNC +0.0029 [+0.0009, +0.0054] / +0.0030 [+0.0010, +0.0054]; dPDMS -1.145 [-1.473, -0.840] / -1.174 [-1.503, -0.858]. Both conditions fail by wide margins and on both seeds: (i) the interval EXCLUDES zero from above (lower +0.0009, upper 5x the bar), (ii) even the UPPER bound of dPDMS is negative (-0.88 / -0.89). P2: NC-unsafe 0.0075, PDMS x100 90.365. **Committed next arm (PREREG s6, NO-RECOVER): NC calibration closes; the line goes to PRIVGEOM-SAFE (already queued) and the trunk.** The forecast (NO-RECOVER about 0.97; (i) alone 0.15; (ii) alone 0.01) held.

**Three things the Master Mind must know:**
1. **The lever has no purchase because the row is already calibrated.** Fitted (a, b) per stratum and fold sit at a = 0.99 to 1.16 and |b| <= 0.21 except one S fold (-0.385) (`RESULT_tables.md` s3). Held-out BCE improves by at most 0.0004 nats (Q3, Q4), gets WORSE in Q1 (-0.0006), Q2 (-0.0004) and S (-0.019: 195 sets, per-stratum Platt over-fits the small stratum); global Platt gains 0.0001. The calibration moves 993 / 981 of 15,808 composite picks, and the result is slightly WORSE in PDMS (-1.205 vs -1.145). Calibrating P2's own picks (descriptive arm PC) gives dNC +0.0006 [+0.0002, +0.0011], dPDMS +0.070 [-0.016, +0.149].
2. ⚠️ **A POST-HOC, DESCRIPTIVE reading cuts against the "winner's curse" explanation in `MM_RULING_PICKRULE.md` s3 and needs your ruling.** At the SELECTED picks, observed unsafe mass over predicted unsafe mass (sum of 1 - NC label over sum of 1 - NC sigmoid) is **0.895 [0.664, 1.168]** (seed 0; 0.910 [0.683, 1.177] seed 1) for the uncalibrated EP-led composite: the NC row is NOT overconfident at its own picks on average. The composite's 46 extra label-0 picks over P2 (seed 0: 164 vs 118) sit in the raw-sigmoid bins below 0.97 (+24 in [0.5, 0.9), +24 in [0.9, 0.97), +8 in [0.97, 0.99), -10 in [0.99, 1]); it picks 1,643 / 1,627 proposals (seed 0 / 1) below 0.97 where P2 picks 1,101. So most of the NC excess reads as risk the row STATES and the product rule ACCEPTS, not as an error calibration could remove. The one bin where observed exceeds predicted is [0.5, 0.9) (37.0 vs 26.4; n 180; P2 8.0 vs 8.6, n 62), a small-n signal with no interval. Written after `result.json` existed; bins fixed before the run; no authority; needs its own pre-registration before it can move any claim (`posthoc_diag.json`).
3. **Condition (ii) is the binding one.** P2 is in-sample and the composite out-of-fold, and every NC-only change tried so far on this rig (the audit's veto and NC power gate k = 1 to 16, and this calibration) leaves dPDMS between -1.1 and -1.6 (INFERENCE from those rows, not a proof of unreachability). That is a limit of the rig, fixed by the brief and not softened; it does not say the composite is worse than P2 out of sample.

## 1. What was run

* **Data, rows, folds, seeds, composite:** identical to PICKRULE-AUDIT-NC (its landed `arrays.npz`, sha256 `29130bcd…`, verified against its prepare report; EP-R1 c\* regenerated OOF, seeds 0 / 1; the audit's reimplemented `planner.py:558-561` rule; landed code pinned by sha256: `pickrule_lib` `555c0a34…`, `epr1` `a751c63e…`, `fastneg_lib` `0c0b399b…`). Strata: the census's label-free `strata.json` (`67f43e55…`) and `speeds.npz` (`433fd274…`).
* **Calibration (PRIMARY):** per stratum (S, Q1..Q4) and per fold, `z' = a z + b`, soft-BCE on the NC label (0 / 0.5 / 1), deterministic Newton from (1, 0), fitted on the four training folds and applied to the held-out fold (grouped by log). 25 fits, **0 failures**, at most 4 Newton iterations, max |gradient| 8.8e-11. A unit test checks the solver against L-BFGS-B to 1e-6.
* **Decision (PREREG s6):** RECOVERS iff on BOTH seeds (i) the upper CI of dNC < +0.001 AND (ii) the lower CI of dPDMS x100 >= 0. Not met on either seed.

## 2. All arms against P2's rule

| arm | NC-unsafe rate | dNC | DAC-unsafe | dDAC (no authority) | PDMS x100 | dPDMS x100 | picks changed vs P2 |
|---|---|---|---|---|---|---|---|
| P2 | 0.0075 | | 0.0228 | | 90.365 | | |
| U seed 0 / 1 (uncalibrated, beside) | 0.0104 / 0.0105 | +0.0029 / +0.0030 | 0.0300 / 0.0297 | +0.0071 / +0.0069 | 89.219 / 89.191 | -1.145 / -1.174 | 12,314 / 12,292 |
| **C seed 0 / 1 (PRIMARY)** | 0.0104 / 0.0104 | +0.0029 / +0.0030 | 0.0305 / 0.0300 | +0.0077 / +0.0072 | 89.160 / 89.152 | -1.205 / -1.212 | 12,430 / 12,439 |
| G seed 0 / 1 (global Platt, descriptive) | 0.0103 / 0.0103 | +0.0028 / +0.0028 | 0.0300 / 0.0297 | +0.0072 / +0.0069 | 89.213 / 89.207 | -1.152 / -1.158 | 12,275 / 12,277 |
| PC (P2's EP, calibrated NC; descriptive) | 0.0081 | +0.0006 | 0.0226 | -0.0003 | 90.434 | +0.070 | 2,446 |

(Intervals for every cell: `raw/results/RESULT_tables.md`.) The calibrated composite's newly NC-unsafe picks (84 / 85 sets) have a median calibrated NC sigmoid of 0.965 and none below 0.5, as for the uncalibrated one (82 / 83, 0.960).

## 3. Controls

| control | result |
|---|---|
| C-JOIN-STRATA (label-free, gating) | PASS: `key_log` equals the arrays' log names set by set; `pick` equals `top8[:,0]`; `code` recomputed from `r` and the cut points equals the stored code; 249,808 rows per Q stratum and 12,480 in S (195 sets) |
| C-ID (a) identity calibration | **PASS: NC column bit-equal to P2's; picks equal the uncalibrated composite's on 15,808 / 15,808 sets, both seeds** |
| C-ID (b) rule reproduction | PASS: the rule equals the banked pick and `epr1.picks_np` on 15,808 / 15,808 |
| C-ID (c) landed reproduction | **PASS, max abs difference exactly 0.0** against the audit's `result.json` (every registered field point / lo / hi), 12,314 / 12,292 picks changed as banked |
| C-REACH (planted) | PASS: on a synthetic planted bank the real `apply_calibration` equals `a_{fold,stratum} z + b_{fold,stratum}` on every proposal, a planted `b = -3` in one stratum shifts exactly that stratum and moves the pick on some sets and not all and no set without it; on the real arrays the same planted table shifts exactly the 249,808 Q4 proposals (INFORMATIONAL: it moves 845 / 833 real picks). C-FOLD: flipping fold 2's labels leaves its parameters bit-identical |
| C-REACH-DECISION | PASS: branch table exact; **27 RED mutants** (9 calibration code path, 18 decision functions) are each caught, plus 2 permutation mutants and a stratum-blind fit mutant |
| **C-NULL** (codes permuted within token, 5 replicates, refit each time) | **RECOVERS fired on 0 of 5**; condition (i) alone on both seeds also 0 of 5; VOID = False. Per replicate dNC upper +0.0050 to +0.0056, dPDMS lower -1.467 to -1.531 |
| tests | `test_ncfastcal.py`: 20 OK (`raw/logs/test_ncfastcal.log`) |

⚠️ **Honesty on the null:** the real-data null is silent for a weak reason (the bar of RECOVERS is out of reach, so nothing can fire). Its POWER is shown on planted banks (`test_ncfastcal.py`; numbers in `raw/logs/planted_power_report.log`): a speed-conditional miscalibration makes the true strata RECOVER while the five within-token permutations do not, a stratum-blind fit does not recover it, and a uniformly miscalibrated bank makes the null fire 5 of 5 (mapped to VOID).

## 4. Amendments and disclosures

* **No amendment.** All gate checks passed on the first real-data run.
* **Disclosures (PREREG s0):** I had read the audit's `RESULT.md`, hence the uncalibrated composite's numbers and the audit's post-hoc D4 (monotone NC power gate: NC level restored, PDMS worse). No array of this arm was opened before sealing.
* **My own test, not a registered item:** the first test run failed once (`raw/logs/test_ncfastcal_attempt1.log`): I had expected an all-safe-labels stratum to be rejected, but the REGISTERED fit-failure rule accepts the point where the gradient reaches 1e-8 (b about 23, within the 50 bound). I corrected the test to document that edge; the rule is unchanged and it did not occur (all 25 fits converged with finite, moderate parameters).
* ⚠️ **Environment (the Master Mind's C: notice):** C: was full (0 bytes free) shortly after the run (about 10:35); the only `Errno 28` I saw was one stdout flush of a read-only inspection command (no output file). Integrity of everything I rely on was then checked: `result.json`, `gate_report.json`, `posthoc_diag.json` parse; every array of `nc_cal_oof.npz` loads and is finite; the code hashes recorded in `result.json` equal the current files; the prereg hashes to its seal; every log ends on its terminal line. And the analysis was RE-RUN from scratch into a separate directory (`raw/logs/analyze_repro.log`): `result.json` is identical (all fields except `created` / `elapsed_s`) and `nc_cal_oof.npz` is bit-identical, so the run is deterministic and the outputs are not truncated. No full-repo export was made; only the landed census, EP-R1 and audit files named in the prereg were exported byte-exact from the tip.

## 5. Next arm and what remains (Rule zero)

* **Committed:** NC calibration closes; PRIVGEOM-SAFE (already queued) and the trunk (P6, about 25 Oct). No GPU, spend or PI decision blocks anything here.
* **The cheapest remaining NC-only lever I did NOT run** is a risk-budgeted (pessimistic) selection, because the post-hoc reading says the excess is risk the row states: bins below 0.97 carry it. The audit's veto stopped at tau = 0.9 and its replacements were near-twins the same row misreads (audit D3), and its power gate restored the NC level at a PDMS cost (D4); I would not spend a run on it before PRIVGEOM-SAFE, but the bin table is the pre-registration input if you want one.
* **Rulings requested:** (a) item 2 above versus the winner's-curse reading of the audit; (b) whether the next NC-side registration should carry the reliability-at-picks table as a pre-registered diagnostic.

## 6. Proposed GAC / BACKLOG text (for the Master Mind to land; I did not edit them)

**GAC, new row:** `| **D-REFE-NCFASTCAL-1** (2026-10-19) | **NC-FASTCAL reads NO-RECOVER on both seeds: speed-conditional (per-stratum) Platt calibration of P2's NC logit does not recover the EP-led composite's NC-unsafe pick rate or PDMS against P2.** PREREG sha256 4db61ce1…cc4f, no amendment. Same S0 f10 rig as PICKRULE-AUDIT-NC (EP-R1 c* OOF, 15,808 sets, 117 logs, P2 in-sample), strata = the census's (67f43e55…), 25 OOF fits, 0 failures. Calibrated composite vs P2: dNC +0.0029 [+0.0009, +0.0054] / +0.0030 [+0.0009, +0.0055] (bar: upper < +0.001), dPDMS x100 -1.205 [-1.542, -0.882] / -1.212 [-1.549, -0.890] (bar: lower >= 0); uncalibrated beside it +0.0029 / -1.145 and +0.0030 / -1.174. The row is already calibrated per stratum (a 0.99-1.16, |b| mostly <= 0.2; held-out BCE gain <= 0.0004 nats, negative in S, Q1, Q2). Controls: identity bit-exact 15,808 / 15,808, landed audit reproduced to 0.0, planted reach, 27 RED mutants caught, C-NULL 0 of 5 (VOID False), 20 tests. POST-HOC DESCRIPTIVE (no authority): observed / predicted unsafe mass at the EP-led picks 0.895 [0.664, 1.168], i.e. NC is not overconfident at its picks on average; the excess sits in raw-sigmoid bins below 0.97. MEASURED, SELECTION tier, SEEN scenes, P2 in-sample; paired log-cluster bootstrap 10,000 / seed 20260927. NC calibration closes; PRIVGEOM-SAFE and the trunk remain. `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-nc-fastcal/RESULT.md` | MEASURED (selection tier) |`

**Also append to `D-REFE-PICKRULE-NC-1` / `H-REFE-RL1-NCRES`:** UPDATE 2026-10-19 (NC-FASTCAL): per-stratum calibration of the NC row does not move the composite; post-hoc, the NC row is not overconfident at the EP-led picks on average (observed / predicted unsafe mass 0.895 [0.664, 1.168]), which qualifies the "confidently wrong" reading. Unpromoted until pre-registered.

**BACKLOG:** mark `NC-FASTCAL` / `NC-CALIB-FASTBAND` DONE (NO-RECOVER); keep `PRIVGEOM-SAFE` next; optionally add `NC-RISKBUDGET` (0 GPU; registers a risk-budgeted selection with the reliability-at-picks bins as the pre-registered diagnostic, both outcomes committed, replicate arm) at lower priority than PRIVGEOM-SAFE.

## 7. Manifest

See `MANIFEST.md` (every file, size, sha256, where it lives). Staged under `C:/Users/Admin/par18/ncfastcal/out/`; `paths.json` is the landing list. Nothing is committed or pushed. Held only on the dev box: `C:/Users/Admin/par18/ncfastcal/run/` (the same files as staged) and the audit's `arrays.npz` it reads (READ ONLY, regenerable by `pickrule_run.py prepare`).
