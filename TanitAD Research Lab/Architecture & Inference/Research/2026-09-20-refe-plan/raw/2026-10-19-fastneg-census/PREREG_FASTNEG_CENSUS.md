# PREREG FASTNEG-CENSUS: is P-B (the fast band an EP-led pick selects) a COVERAGE gap, a FEATURE gap, or no row-resolution gap at all? (Master Mind, 2026-10-19)

**Written BEFORE any HEADROW-PROBE outcome was read and BEFORE any label was stratified by speed.**
- Design source: D1 of `TanitAD Research Lab/Architecture & Inference/Research/2026-10-19-safety-head-lit/REPORT.md` §4.5. Its weaknesses are fixed here: the reference speed, standstill, the deficit condition and a powered null.
- Tier: SELECTION, on SEEN scenes (`D-L1-G0-1`). ⛔ No dev table and no navtest are read. 0 GPU.

## 1. Question
`H-REFE-RL1-NCRES` (three arms): an EP-led pick lands on fast proposals that the NC / DAC rows call safe and are not.
- HEADROW-PROBE separates CAPACITY from FEATURE on the pooled rows.
- A **COVERAGE** cause looks FEATURE-limited to a probe fitted on the same sparse labels: the rows saw too few fast, off-expert, unsafe examples to use information they have.
- This census reads the fast stratum ON ITS OWN.

## 2. Data (identical to HEADROW-PROBE)
- S0's bank: `feat [S,64,256]` fp16 and `labels6`, sha256-verified by the `srttc_r1.py` / `epr1.py` loaders.
- f = 10 %, the IDENTICAL fit logs, the same 5-fold grouped-by-log CV, seeds {0, 1}.
- Rows: NC (col 0) and DAC (col 1), each in its registered label convention.

## 3. Speed ratio and strata, fixed from SPEEDS ONLY
- For each (token, proposal): `v_p` = the proposal's mean speed over its horizon, from its trajectory. `v_b` = the mean speed of **P2's frozen base pick** in the same token. The base pick is used, not the teacher, so the reference exists in every token and is the pick an EP-led change departs from.
- `r = v_p / v_b`.
- **Standstill tokens** (`v_b < 0.5 m/s`) are stratum **S** and are reported separately, never in the quartiles.
- **Strata Q1..Q4** = quartiles of `r` pooled over every non-S (token, proposal) row of the f10 fit set.
  - Cut points are computed and written to `strata.json` (with its sha256) by a script that does NOT open `labels6`.
  - That file is committed to the run dir BEFORE the label pass runs; the label pass refuses to start without it.
- **Fast stratum F = Q4.**
- ⛔ If the trajectories cannot be found for the bank's proposals: STOP and report the files probed. No other definition of speed is substituted.

## 4. Measurements (paired log-cluster bootstrap, 10,000 resamples, seed 20260927; CIs over logs)
- **(a) Label support per stratum:** row counts, the unsafe share (NC = 0 or DAC = 0), and the share of all unsafe rows. Descriptive; no rule.
- **(b) Resolution per stratum:** within-token AUC (unsafe vs safe pairs inside a token, the measure that read 0.46 on the near-twin grazes) for each of:
  - (i) P2's frozen NC / DAC rows (in-sample for P2; flagged as such);
  - (ii) HEADROW-PROBE's L-arm and M-arm OOF predictions, when its addendum files exist.
  - (ii) is read only from the probe's own saved files. If they are absent, (ii) is reported MISSING and the rule below runs on (c)'s L refits alone.
- **(c) Fast-stratum learning curve:** refit the L arm (HEADROW-PROBE's linear settings: Adam lr 1e-3, wd 1e-4, 20 epochs, BCE) with F's TRAINING rows subsampled to 25 / 50 / 100 %. Other strata stay at 100 %; the subsample is nested and seeded. Read F's OOF within-token AUC on the full held-out folds.
- **(c0) Powered null (`RETRACTION_LOG` R51):** 5 replicate pairs of two independent 100 % L refits that differ only in the subsample / shuffle seed. Apply the COVERAGE rule's slope test to each pair. The false-fire rate is reported; if the null fires on ≥ 2 of 5 replicates, the census is VOID.

## 5. Decision rule (fixed now)
Per row (NC, DAC), per seed, on the L arm at 100 %:
- **DEFICIT** iff F's within-token AUC minus pooled Q1–Q3's has an upper CI < −0.02.
- **NO DEFICIT** iff its lower CI > −0.02. Otherwise INCONCLUSIVE.

Given DEFICIT on both seeds:
- **COVERAGE-LIMITED** iff AUC_F(100 %) − AUC_F(25 %) has a lower CI > +0.01 on both seeds.
- **FEATURE-LIMITED** iff its upper CI < +0.01 on both seeds.
- Otherwise INCONCLUSIVE.

A row is classified only if both seeds agree; the census outcome is the NC and DAC pair.

## 6. Committed next arms
| outcome | next arm |
|---|---|
| COVERAGE on NC or DAC | **FASTNEG-BANK** (SH1): a fast / lead-ward `fast` family in `build_scorer_targets.py`, EST 26.4 CPU-h, then HN-screen arm C under an amendment of `PREREG_P6HN_SCREEN` (3 seeds + an A replicate; endpoints E1–E3 of the Lab report §4.5) |
| FEATURE on both | **PRIVGEOM-PROBE** (D2, CPU EST 13.5 h): privileged PDM geometry scalars added to the token as a diagnostic only. Which scalars carry the jump names the scorer-branch lever (SH3 / SH5); no jump → the trunk (P6 read; SH7 / SH8 are A40 and PI-gated) |
| NO DEFICIT on both | P-B is not a fast-band resolution deficit. The leak is in how the pick rule COMPOSES the rows (`planner.py:558-561`). Next: a 0-GPU pick-rule audit on S0: the unsafe-pick rate of an EP-led composite as a function of the NC / DAC veto threshold, with the rows unchanged |
| INCONCLUSIVE | re-run ONCE at f = 25 % (the next nested size), cost named before the run |

## 7. Limits stated in advance
- SEEN scenes; selection tier; in-sample for P2's rows in (b)(i).
- Within-token AUC ignores calibration, but the pick rule thresholds the rows. That is why the NO-DEFICIT outcome's next arm is the threshold audit.
- One rig, two seeds of the refit; the training-variance question (`H-ESTIM-SEED-1`) is answered for the refit, not for P2.
