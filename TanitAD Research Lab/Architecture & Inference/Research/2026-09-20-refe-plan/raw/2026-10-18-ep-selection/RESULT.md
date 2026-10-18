# RESULT: EP-1. P2's pick loses ego progress because it is safe-but-slow, not because the fan lacks fast plans; four zero-training selection rules were pre-registered and none clears its fit gate, so the single navtest confirm was NOT spent

Agent for the Master Mind, 2026-10-18. CPU only, nothing trained. Package target `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-ep-selection/`. **Integration needed: the Master Mind lands this folder and the two draft register rows below. Every file exists in ONE place (this dev box, `C:/Users/Admin/par18/ep/out/`) until landed.**

**Headline.**
1. **Diagnosis (MEASURED, fit table = P2's own navtrain held-out table, 1,705 tokens / 24 logs).** Three tokens in four hold a safe plan with EP = 1 (best-safe EP 0.978 vs the pick's 0.851, gap 0.127 [0.102, 0.150]); picking it would close 99.6 % of the oracle gap (+8.07 [6.23, 10.12] of +8.10 PDMS x100). The pick is 2.9 m shorter at 4 s and sits at length percentile 0.41. Cause: the EP head is an **unconditional** expectation (violators scored 0; `E[NC*DAC*EP]` 0.7216 vs mean pEP 0.7186) and the rule multiplies by `pNC*pDAC` again, a length-increasing double charge (effective EP term 0.70 vs true 0.96 on the longest plans). The remaining margin is decided by pDAC, TTC and pNC, with a median top-two gap of 0.0015.
2. **Candidates (pre-registered with sha256 before any outcome).** C1 XPDMS (no double count), C2 BAND (progress tie-break in the rule's near-ties), C3 FLOOR (progress above a predicted-safety floor), then one round-2 guarded variant C4 BANDGUARD, which `PREREG_EP1.md` O-B had committed as the next arm for a progress-for-safety trade. **All four are ELIMINATED AT FIT (O-D); navtest was never opened.**
3. **Blocker, MEASURED.** Moving to longer plans buys EP and pays TTC and violations. Best fit candidate C4: PDMS **+0.61 [+0.19, +1.20]** in-sample, nested out-of-fold **+0.51 [+0.13, +1.01]**, 5 of 6 log-folds positive, but TTC **-0.70 [-1.34, -0.08]** separates adversely (gate F2), as in the pre-registered bar. The TTC head ranks contenders at 0.72 concordance (D-REFE-SELRANK-1), so no rule over the six existing outputs can keep the progress without the TTC cost.
4. **Next lever (RULE ZERO).** Training-side: EP conditional on geometry-safe plus a pairwise EP ranking term, bundled with SR-TTC's TTC ranking term, head-only on a P2 forward over navtrain. Cost ESTIMATED: ~0 marginal GPU-h on top of SR-TTC's forward (~7 GPU-h in total, ~1.0-1.3 per 3,137 sets), plus CPU head training. Blocked by dev-box GPU contention and the PI/MM GPU decision. Not run: the brief forbids training.

## 0. Tier, estimator, class
- **Fit set** (the only place anything was fitted): P2's own navtrain held-out table `D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz`, sha256 `8eacc19b35b12aee9b0ab047716c901399a711d22ddd3c25cd890ecf2f02004b` (verified), package `raw/2026-10-18-p2-heldout-table/`. **Correction to `D-REFE-SELRANK-1` section 7**, which wrote that no P2 held-out dump exists: it exists (landed after that read; absence at one location). Disjointness from navtest (logs and tokens) and from P2's training bank is that package's `disjointness_proof.json` (INHERITED, PASS; not re-run).
- **Tier**: SELECTION tier (NAVSIM v1.1 PDMS of the selected proposal looked up in the table), never an endpoint. **T1 confirm on navtest: not performed.**
- **Estimator**: paired log-cluster bootstrap `zero_tail.Boot` (sha256 `149c3e97...`), 10,000 resamples, seed 20260927, same draws for every term; answers "another draw of LOGS" only (24 logs here); P2 is one training seed, deterministic inference (H-ESTIM-SEED-1).
- **Class**: MEASURED (ours) for every number below unless marked ESTIMATED; read off `raw/*.json`.
- Gates: PDMS formula reproduces the table (max abs diff 0.0; DDC-in-product control 1.0), planner rule reproduces `pick` 1,705 / 1,705, identity controls 1,705 / 1,705 for C1-C4, shuffled-proxy controls <= +0.05 (C1 -3.31, C2 -0.06, C3 -1.80, C4 -0.08), baseline PDMS x100 **90.64**.

## 1. Diagnosis (full text and tables: `DIAGNOSIS.md`)
(a) fan: EP best safe 0.978 vs pick 0.851; best-safe-EP proposal gains +8.07 PDMS x100. (b) EP head: calibrated on all proposals by length quintile (0.678/0.676 ... 0.671/0.675) so it is unconditional; on geometry-safe plans bias -0.043 (shortest quintile) to **-0.168 [-0.195, -0.140]** (longest); `pEP/(pNC*pDAC)` repairs it to within 0.025; within-token rank power on safe plans: plan length 0.904, `pEP/g` 0.791, raw `pEP` 0.363. (c) 1,039 of 1,705 tokens (61 %) have a truly better, safe, longer plan (+7.12 [5.17, 9.24] if swapped); margin terms: pNC +0.137, pDAC +0.120, TTC +0.117, **EP +0.049** (favours the pick in 68 %), C 0.000. (d) normaliser: reference INFERRED (ESTIMATED) at median 18 m; one metre costs 0.134 EP below 10 m reference, 0.024 above 35 m; the average shortfall is 3.4 m.

## 2. Candidates and fit read (pre-registered: `PREREG_EP1.md` sha256 `1fcf61f6...`, `PREREG_EP1_R2.md` sha256 `8443f8b5...`; code hashes `FIT_CODE.sha256`, `FIT_R2_CODE.sha256`, written before the runs)
Paired difference vs P2's pick on the fit table, x100 unless stated, [95 % log-cluster interval]:

| | C1 XPDMS | C2 BAND (delta 0.02) | C3 FLOOR | C4 BANDGUARD (0.02, 0.02) |
|---|---|---|---|---|
| picks changed of 1,705 | 366 | 1,437 | 0 (identity chosen) | 1,340 |
| **PDMS** in-sample | +0.10 [-0.17, +0.35] | +0.42 [-0.22, +1.10] | 0.00 (all tau < 1 lose, -0.35 to -5.47) | **+0.61 [+0.19, +1.20]** |
| PDMS nested out-of-fold | n/a (no parameter) | +0.23 [-0.42, +0.84] | 0.00 | **+0.51 [+0.13, +1.01]** |
| log-folds positive | 3 of 6 | 3 of 6 | 0 of 6 | **5 of 6** |
| **LONGITUDINAL**: EP | +0.57 [0.28, 0.80] | +2.98 [2.15, 3.72] | 0 | **+2.03 [1.50, 2.77]** |
| **LONGITUDINAL**: TTC | 0.00 [-0.27, +0.31] | **-1.58 [-2.45, -0.77]** ADVERSE | 0 | **-0.70 [-1.34, -0.08]** ADVERSE |
| **LATERAL**: DAC | -0.24 [-0.55, +0.06] | -0.18 [-0.72, +0.40] | 0 | +0.12 [-0.24, +0.62] |
| collision NC (gate component) | 0.00 [0, 0] | -0.18 [-0.51, +0.06] | 0 | -0.06 [-0.31, +0.13] |
| comfort C / DDC | 0 / -0.29 [-0.64, 0] | 0 / -0.50 [-1.24, +0.07] | 0 / 0 | 0 / -0.26 [-0.83, +0.18] |
| zero-PDMS picks (baseline 56) | 60 | 62 | 56 | 55 |
| mean plan length at pick (baseline 18.00 m) | 18.26 | 19.39 | 18.00 | 18.98 |
| gate F1 / F2 / F3 | fails (3 of 6 folds) / passes / passes | fails / **fails (TTC)** / passes | fails / passes / passes | passes / **fails (TTC)** / passes |
| outcome | ELIMINATED AT FIT | ELIMINATED AT FIT | ELIMINATED AT FIT | **ELIMINATED AT FIT (O-D)** |

**TACTICAL and STRATEGIC families** were not produced: tactical needs `families6.py` on the candidate's executed plan, which the pre-registration scheduled for the navtest confirm only (UNVERIFIED whether it runs on navtrain tokens); STRATEGIC is **PAUSED** (PI ruling 2026-09-19, ITEM 25), never a pass; distance-keeping is UNAVAILABLE (no lead-agent track). The families are therefore not reported for the candidates: a work item, not a pass.

**Why they fail, exactly (`raw/mech.json`, sequential decomposition, gate first, max error 2.4e-16).** The payoff is asymmetric: a longer pick gains about +3 on most moved tokens and a new violation costs about -50. C1: EP channel +0.33 [0.21, 0.45], gate channel -0.21 [-0.47, 0.02], TTC -0.02; 226 moved picks gain (+2.9 each), 9 lose (-54.5 each, 5 newly zeroed). C2: EP channel +1.48 [1.26, 1.73], TTC channel **-0.71 [-1.06, -0.38]**, gate -0.35; 872 gain (+3.5 each), 46 lose (-49.8 each, 12 newly zeroed). Informative ceiling (not a gate): the band rule with the TRUE EP as its progress proxy reaches only +2.75, so even a perfect progress signal inside the rule's near-ties recovers a third of the +8.10 oracle gap; the rest needs safety re-ranking.

## 3. Outcome against the pre-registered table
O-D (ELIMINATED AT FIT) for every candidate. The single navtest confirm (T1, 12,146 tokens / 136 logs, versus P2's pick) was **not run and remains available**. I did not waive F2 for C4: its TTC interval upper bound is -0.08, a thin miss, and moving the goalpost after the data is excluded by the rule. **Decision for the Master Mind/PI (a call, not a result): spend the navtest look on C4 as a declared near-miss (B1 at m = 1 has a fit-set lower bound of +0.19; B2 on TTC is expected to fail again), or keep it for the SR-TTC-retrained scorer. My recommendation: keep it.** A fifth fit-set variant would be a fifth fork on the same 24 logs.

## 4. Proposed register rows (not edited; the Master Mind lands them)

| **D-REFE-EP-1** (2026-10-18) | **P2's EP loss at the pick is a safe-but-slow pick, not a missing fan, and the cause is a double charge: the EP head is an unconditional expectation (violators scored 0; `E[NC*DAC*EP]` 0.7216 vs mean pEP 0.7186, calibrated by length quintile) and `planner.py:558-561` multiplies `pNC*pDAC` on top, so the effective EP term is 0.70 vs a true 0.96 on the longest plans.** MEASURED on P2's own navtrain held-out table (1,705 tokens / 24 logs, SELECTION tier, descriptive, nothing fitted): the best safe proposal has EP 0.978 [0.966, 0.988] vs the pick's 0.851 [0.823, 0.881] (gap 0.127 [0.102, 0.150]); picking it would give +8.07 [6.23, 10.12] PDMS x100 = 99.6 % of the oracle gain; the pick is 2.9 m shorter at 4 s; 1,039 of 1,705 tokens have a truly better safe longer plan; margin terms pNC +0.137, pDAC +0.120, TTC +0.117, EP +0.049. Paired log-cluster bootstrap, 10,000, seed 20260927 (another draw of LOGS only; 24 logs; P2 one training seed). `raw/2026-10-18-ep-selection/` (DIAGNOSIS.md, raw/diag.json, raw/diag2.json) | **MEASURED, fit-set descriptive.** Corrects `D-REFE-SELRANK-1` section 7: P2's held-out table exists. |

| **D-REFE-EP-2** (2026-10-18) | **No zero-training selection rule over P2's six outputs and its own plans clears its fit gate: four pre-registered candidates ELIMINATED AT FIT, navtest not opened.** Best, C4 BANDGUARD (delta 0.02, eps 0.02): PDMS +0.61 [+0.19, +1.20] in-sample, +0.51 [+0.13, +1.01] nested out-of-fold, 5 of 6 folds, EP +2.03 [1.50, 2.77], but TTC -0.70 [-1.34, -0.08] separates adversely (F2). C1 +0.10 [-0.17, +0.35]; C2 +0.42 [-0.22, +1.10] with TTC -1.58 [-2.45, -0.77]; C3 loses at every tau. Mechanism: moves to longer plans buy EP and pay TTC/violations at +3 vs -50 per moved token; the heads' ranking cannot localise the losers (TTC contender concordance 0.72). Controls held (identity 1,705/1,705; shuffled proxies <= +0.05). Fit table sha256 `8eacc19b...`; prereg sha256 `1fcf61f6...` and `8443f8b5...` written before any outcome. Next lever: training-side EP conditional + pairwise ranking, bundled with SR-TTC (cost ESTIMATED ~0 marginal GPU-h on SR-TTC's ~7 GPU-h forward). `raw/2026-10-18-ep-selection/` | **MEASURED (fit-set), SELECTION tier.** No claim about navtest. Supports `D-REFE-SELRANK-1`'s refutation of CPU recalibration, with a structurally different rule family (W5/CAL14 could not express C1-C4). |

Draft BACKLOG line: `EP-1 | DONE 2026-10-18, 0 GPU (raw/2026-10-18-ep-selection/RESULT.md, D-REFE-EP-1/-2): double-charge diagnosed; 4 zero-training rules eliminated at fit (best C4 +0.61 fit, TTC -0.70 adverse); navtest look UNSPENT | decision: spend it on C4 as a near-miss, or hold for the SR-TTC-retrained scorer; training-side EP term rides on SR-TTC's P2 forward`.

## 5. Limits
- 24 fit logs: every fit interval is wide, and four fits on one set is a forking path; the guard against it is that nothing was promoted and navtest is untouched.
- The fit set (pick 90.64, oracle 98.74) differs in level from navtest (88.82, 98.23); transfer of any fit result is an ASSUMPTION.
- The EP reference in DIAGNOSIS (d) is inferred (ESTIMATED); the disjointness proof is INHERITED.
- C4 had two parameters and was designed after seeing C2's fit read; it is labelled round 2 for that reason, and its prereg was hashed before its first number.
- One training seed, deterministic inference: no interval speaks to another training run.

## Deliverable manifest
All under `C:/Users/Admin/par18/ep/out/` (dev box only; repo-relative target above; `paths.json` lists `[repo path, source, "raw"]`). **Everything is in ONE place until the Master Mind lands it.** Not staged to git by instruction.

| artifact | where |
|---|---|
| `RESULT.md`, `DIAGNOSIS.md`, `PREREG_EP1.md` (+ `.sha256`), `PREREG_EP1_R2.md` (+ `.sha256`), `FROZEN.json`, `FROZEN_R2.json`, `FIT_CODE.sha256`, `FIT_R2_CODE.sha256`, `DIAGNOSIS.sha256`, `MANIFEST.sha256` | dev box only |
| `raw/diag.json`, `raw/diag2.json`, `raw/fit_read.json`, `raw/mech.json`, `raw/fit_read_r2.json`, `raw/diag.log` | dev box only |
| `code/ep_common.py`, `ep_diag.py`, `ep_diag2.py`, `ep_rules.py`, `ep_fit.py`, `ep_mech.py`, `ep_fit_r2.py`, `zero_tail.py` (vendored, sha256 `149c3e97...`) | dev box only |
Reproduce: `CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe code/ep_diag.py --out raw/diag.json` then `ep_diag2.py`, `ep_fit.py --out raw/fit_read.json --frozen FROZEN.json`, `ep_mech.py`, `ep_fit_r2.py --out raw/fit_read_r2.json --frozen FROZEN_R2.json` (each run under 10 s).
