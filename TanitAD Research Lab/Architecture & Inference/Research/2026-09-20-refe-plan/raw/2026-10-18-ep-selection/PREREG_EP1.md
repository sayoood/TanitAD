# PREREG_EP1: three zero-training selection candidates for P2's ego-progress loss

Written 2026-10-18 by the EP-1 agent, **before any candidate rule's PDMS, pick or component was computed on any table**. Step 1 (`DIAGNOSIS.md`, `raw/diag.json`, `raw/diag2.json`) is descriptive and ran on the fit table only; it evaluated no candidate. The navtest table has not been opened.

## 0. Frame

- **Question.** Can a rule that changes ONLY how P2's six existing head outputs and its own 8-pose plans are turned into a pick (no training, no new head) raise the paired PDMS on navtest, without paying in NC, DAC or TTC?
- **Baseline `B`** = P2's deployed rule: `p = sigmoid(logits)`, `g = pNC*pDAC`, `W = (5 pEP + 5 pTTC + 2 pC)/12`, `S0 = g*W`, pick = `argmax S0` (`refe/planner.py:520, 558-561`). It reproduces `pick` on 1,705 / 1,705 fit tokens (`raw/diag.json` gates).
- **Fit set (the only place anything is fitted or chosen).** P2's own navtrain held-out table, 1,705 tokens / 24 logs, sha256 `8eacc19b35b12aee9b0ab047716c901399a711d22ddd3c25cd890ecf2f02004b`. **Confirm set (opened ONCE).** P2's full-navtest table `table_p2_full.npz`, 12,146 tokens / 136 logs, sha256 prefix `33441082b7044c95`. T1, versus P2's own pick, paired over the 136 logs.
- **Estimator.** Paired log-cluster bootstrap, `zero_tail.Boot` (sha256 `149c3e97...`), 10,000 resamples, seed 20260927, the same draws for every term. It answers "another draw of LOGS" only; P2 is one training seed and its inference is deterministic (H-ESTIM-SEED-1). Metric: PDMS of the selected proposal, looked up in the table, x100. A pick is looked up, never re-scored.
- **Progress proxy `L_j`** (used by C2 and C3) = Euclidean distance of the 4 s pose (pose index 7, x/y) from the origin, in metres, of proposal j's OWN plan. It is a property of the plan, available at inference, and uses no label and no ego state beyond the plan. Step 1: within-token Spearman with true EP on gs proposals **0.904** (raw `pEP` 0.363, `pEP/g` 0.791).
- **Ties.** `numpy.argmax`, lowest index.

## 1. The three candidates (rules fixed here; parameters fitted on the fit set only)

Each is derived from step 1 (`DIAGNOSIS.md`): the EP head is an unconditional expectation that the rule charges for violation twice (b); the rule's margins are tiny, median top-two 0.0015 (c); true EP among safe plans is almost a function of plan length (b).

**C1, XPDMS (risk-calibrated expected PDMS; 0 parameters).** `S1 = (5 pEP + g (5 pTTC + 2 pC)) / 12`; pick `argmax S1`. Rationale: `E[NC*DAC*EP]` is measured equal to `E[pEP]` (0.7216 vs 0.7186), so the EP term must NOT be multiplied by `g` again; TTC and C terms keep the product, since the TTC head is not an expectation with violators set to 0.
**C2, BAND (progress tie-break inside the rule's own near-ties; 1 parameter `delta`).** `A = {j : ln S0_j >= max_i ln S0_i - delta}`; pick `argmax_{j in A} L_j`. Grid `delta in {0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}`; `delta = 0` is the identity (baseline pick).
**C3, FLOOR (progress among plans whose predicted safety is at least `tau`; 1 parameter).** `Q_j = pNC_j pDAC_j pTTC_j`; `A = {j : Q_j >= tau}`; if `A` is non-empty pick `argmax_{j in A} L_j`, else the baseline pick. Grid `tau in {1.0, 0.99, 0.98, 0.97, 0.95, 0.93, 0.90, 0.85, 0.80, 0.70}`; `tau = 1.0` is the identity (no probability reaches 1).

## 2. Fit-set procedure (step 3)

1. **Identity controls** (must reproduce the baseline pick on 1,705 / 1,705): C2 at `delta = 0`; C3 at `tau = 1.0`; C1 written as `(5 pEP g + g(5 pTTC + 2 pC))/12 = S0`.
2. **Deliberate-regression controls** (must read ~0, to prove the gain is information and not form): C2 and C3 with `L` replaced by a within-token random permutation of `L` (seed 20260927); C1 with `pEP` replaced by a within-token random permutation of `pEP`. Instrument valid only if the best-over-grid paired dPDMS of each control is **<= +0.05** (x100). If a control exceeds it, the candidate is held and the instrument is reported broken.
3. **Parameter choice.** For each grid value compute the paired dPDMS (x100) of the candidate pick against the baseline pick on the 24 fit logs. `theta*` = the grid value with the largest dPDMS; among values within 0.05 of the maximum, the one closest to the identity (smallest `delta`, largest `tau`).
4. **Nested grouped cross-validation** (C2, C3): 6 folds of 4 logs, assignment `numpy.random.default_rng(20260927).permutation(24).reshape(6, 4)` over the sorted log names; within each outer fold `theta` is chosen on the other 20 logs by rule 3 and scored on the held-out 4; report the pooled out-of-fold dPDMS and the per-fold chosen `theta`.
5. **Fit gate. A candidate goes to the navtest confirm iff ALL hold:**
   - **F1**: dPDMS > 0 (point, x100) AND positive in >= 4 of the 6 log-folds. For C2/C3 the dPDMS is the pooled **out-of-fold** value of step 4; for C1 it is the in-sample value (nothing is fitted).
   - **F2**: at `theta*`, in-sample, no NC, DAC or TTC component of the pick separates adversely (paired 95 % interval of candidate - baseline entirely below 0).
   - **F3**: its control (step 2) is <= +0.05.
6. The fit read is reported per candidate with the four-family table as far as the table allows (longitudinal: EP, TTC, speed/progress from the plan; lateral: DAC, plan lateral extent; tactical and STRATEGIC: PAUSED / not estimable in a lookup table). Tactical and strategic families are produced at the confirm only (needs `families6.py`).
7. If NO candidate passes the fit gate, **navtest is not opened**, and the turn ends with the named mechanism and the training-side lever (section 4, O-D).
8. Before navtest is read, `FROZEN.json` (candidates, `theta*`, m) is written and its sha256 recorded beside it.

## 3. The single navtest confirm (step 4)

Opened once, for all candidates that passed section 2 (m <= 3). Gates first (any failure stops the confirm): table sha256 prefix; `pick` reproduced on 12,146 / 12,146 from the logits; PDMS formula reproduces `pdms` (max abs diff 0.0) and the DDC-in-product control reads 1.0; identity controls reproduce the baseline pick. Candidate rule uses the frozen `theta*` only.

**Bar for a candidate (both):**
- **B1**: paired dPDMS (candidate - P2's pick) lower bound **> 0** at the Bonferroni-adjusted level: percentile `2.5 / m` of the same draws (m = number of confirmed candidates; m = 1 gives the ordinary 95 % lower bound).
- **B2**: **no NC, DAC or TTC family component separates adversely**: for each of NC, DAC, TTC the paired interval (candidate - baseline, NOT adjusted, 95 %) must not lie entirely below 0.

Reported with every candidate, never pooled: NC, DAC, TTC, EP, C, DDC components (x100) and zero-PDMS count; the four families by `families6.py` on the candidate's executed plan (longitudinal incl. EP and TTC, speed MAE, along-track MAE, progress ratio; lateral incl. DAC, heading, yaw-rate, curvature, cross-track; tactical decision accuracies and goal error; STRATEGIC **PAUSED** per the PI's 2026-09-19 ruling, never a pass); distance-keeping UNAVAILABLE (no lead-agent track passed to `families6.py`), a work item.

## 4. Committed outcomes and the next arm for each (before any number)

| outcome | rule | what it means | committed next arm |
|---|---|---|---|
| **O-A PASS** | B1 and B2 both hold | the rule is a deployable zero-training selection rule for P2; deploying it is the PI's call | (1) transfer read on P6's own held-out/navtest table when it exists (the transfer is an ASSUMPTION until then); (2) if two candidates pass, their combination (C2's band on C1's score) is the next candidate, to be confirmed on P6's navtest, since navtest is consumed here |
| **O-B PDMS-positive but ADVERSE** | B1 holds, B2 fails on NC, DAC or TTC | the rule buys progress with safety, the FP-NONE pattern | not deployable; the safety-constrained variant (same rule, restricted to proposals whose predicted `pNC pDAC pTTC` is not below the baseline pick's by more than an `eps` fitted on the fit set); its confirm rides with P6's navtest because no log-disjoint set remains here |
| **O-C NULL** | B1 fails, B2 holds | head ranking noise or a mean-zero trade; selection rules cannot reach the pool | training-side EP: a **progress-aware scorer loss** (EP head trained conditional on gs plus a pairwise EP ranking term among gs plans). Cost ESTIMATED: ~0 extra GPU-h if bundled into the SR-TTC P2 forward over navtrain (about 1.0-1.3 GPU-h per 3,137 sets, ~7 GPU-h for the SR-TTC arm in total per `D-REFE-SELRANK-1`), a head-only refit |
| **O-D ELIMINATED AT FIT** | no candidate passes section 2, navtest unopened | the zero-training reach is nil on the fit set | the same training-side arm; navtest stays available for it |
| **O-E INSTRUMENT BROKEN** | a control exceeds +0.05 | the gain is form, not information | fix the rule's form, re-register |

Per candidate, a navtest result below the bar eliminates that candidate only; the other candidates' rows stand. No parameter, grid or bar is changed after seeing a number: a change is a new pre-registration.

## 5. Not allowed

Training anything; reading navtest before section 3; changing a grid, a gate or the bar after a number; fitting on navtest; writing into `raw/2026-10-18-sel-rank/` (read its table only); committing, pushing or editing a register.
