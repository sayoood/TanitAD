# PREREG EP-ADMISSION: the EP-reliability admission gate, RULES FROZEN (Master Mind, 2026-10-19)

**Status: the rules are FROZEN before any candidate bank exists.** This file consolidates `DRAFT_PREREG_EP_ADMISSION.md` + its Revision 1 + `MM_RULING_EPADM_R1.md` + the EP-ADMISSION-R2-CONFIRM result. Its sha256 is in `PREREG_EP_ADMISSION.sha256`.
- Per candidate, the read appends ONLY the candidate block (§6): checkpoint path and sha256, the bank's sha256, the seeds.
- Rules, thresholds and estimator are never edited after a candidate's bank exists. A change needs a new pre-registration.

## 1. Question
For candidate C (any EP-led change: a P6 checkpoint, R0e-pick, an RL or head change, L2), against P2:
- Is C's EP row RELIABLE at C's own picks (A1)?
- Are its safety rows not OVER-CONFIDENT there (A2)?
These are asked BEFORE any navtest look is spent.

## 2. Data and estimator
- A selection-tier bank built from C's own forward on S0's tokens (`raw/2026-10-19-p6-read-prep/code/build_candidate_bank.py`): C's rows and C's picks, with P2's picks as the pair.
- Estimator: paired LOG-cluster bootstrap, B = 10,000, seed 20260927, over the bank's logs. It answers "another draw of LOGS" only, not another training run or inference run (`H-ESTIM-SEED-1`).
- Tool: `raw/2026-10-19-p6-read-prep/code/ep_admission.py` **`--rules r1`**. The run JSON records `rules`, and a read without `r1` is void.
- Seeds: the candidate's 2 training seeds. "Any seed" below quantifies over them.
- ⚠️ S0's tokens are navtrain FIT tokens: a candidate trained on navtrain is IN-SAMPLE there. The bias is toward OVER, which is conservative for A1. Every report states it.

## 3. Rules (verbatim from the code, `--rules r1`)
- **A1 (EP reliability at the picks):** D_M = predicted − realised EP mass at C's picks against P2's (rank-free mass form when C's EP is off the label scale).
  - **FAIL-A1 iff, on ANY seed, the D_M lower CI > 0 OR the D_M point > +0.05.** Otherwise ADMIT-A1.
- **A2 (safety reliability at the picks):** observed / predicted unsafe mass at C's picks for NC, DAC and TTC.
  - **FAIL-A2 iff, for ANY of NC / DAC / TTC, on ANY seed, the lower CI > 1.05.** Ratios below 1 are conservative and never fail.
- **A3 (realised gain):** realised dPDMS ×100 vs P2 on the bank, with its CI. REPORTED, never gated; the navtest look owns that decision.
- **Admission:** ADMIT iff ADMIT-A1 AND ADMIT-A2. A non-admitted candidate gets NO navtest look; its failure, with the attribution of EP-RELIAB (D), is the result.
- **B (pessimistic-EP pick rule, secondary):** `EP_used = EP_slot + lam × (EP_row − EP_slot)`.
  - EP_slot is the EP ROW's own per-slot mean on training folds (slot source `row`).
  - lam is the out-of-fold within-set calibration slope.
  - RECOVERS iff the dPDMS lower CI ≥ 0 AND the NC / DAC unsafe-rate differences have upper CI ≤ +0.001, on both seeds.

## 4. The gate's own error rates (MEASURED by simulation on the real estimator and the real S0 bank; Wilson 95 %)
| quantity | banked replicates (R1) | FRESH replicates (R2-CONFIRM, offset 1,000,000) |
|---|---|---|
| A1 false refusal, calibrated, any seed (G78) | 2.7 % [1.9, 3.9] | 3.3 % [2.4, 4.6] |
| A2 (1.05) false refusal, any head, any seed (G78) | 1.9 % (post-hoc) | 1.8 % [1.1, 2.8] |
| A1 power at +0.135 / +0.05 | 100 % / 95-99.8 % | (not re-run) |
| A2 (1.05) DAC power at ×1.25 / ×1.5 (G78) | 78.4 % / 99.9 % | 77.4 % [74.7, 79.9] / 99.9 % |
| A2 (1.05) NC power at ×1.25 | 26.4 % | 25.4 % |

Limits, stated in every read:
- **(i) NC miscalibration at ×1.25 is NOT tested with useful power** (about 195 predicted unsafe pairs).
- **(ii) A2's DAC power falls with fewer changed picks:** 58 % (G25) and 54 % (G05) at ×1.25, FRESH replicates.
- **(iii) A1 is a per-changed-pick test.** A candidate that changes few picks is held to a proportionally smaller absolute D_M.
- **(iv) EP-R1's two seeds were nearly one candidate** (D_M correlation 0.9996). Two independent retrains compound the any-seed rates: A1 up to about 6.5 %, A2 up to about 3.6 % (ESTIMATED as 1 − (1 − p)²).

## 5. Controls
The same C-ID / C-REACH / C-NULL set as EP-RELIAB-PICKS. Under r1 a C-NULL replicate fires about 2 % (4 / 200 on EP-R1's geometry), so P(2 of 5 fire and VOID a good read) is about 0.4 % (ESTIMATED).

## 6. Candidate block (appended per read; nothing above changes)
```
candidate:          <name>
checkpoint(s):      <path>  sha256 <...>   (seed 0) / <path> sha256 <...> (seed 1)
bank:               <path>  sha256 <...>
appended:           <date, time>  by <who>  BEFORE the bank was read
```
