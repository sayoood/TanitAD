# LAT-TOK result: FAIL, admissible (2026-11-16)

**Pre-registration:** `AMENDMENT_1_BCAL_DRAFT.md` section 2.2 item 1, frozen at sha256 `4e122aa5…` (commit 76a8012f) before any data. The MM rulings on admissibility and on the readout wording were written to LOOP_STATE before the result was read.

**Tier and evidence class:** a zero-training probe on TRAIN latents. It is a world-model diagnostic, not a driving metric, and carries no T0/T1 tier. Every number below is MEASURED.

**Estimator:** clip-cluster percentile bootstrap, B 1000, seed 0, over the 141 TRAIN clips, with all sets on the same resamples. It answers "another draw of clips". The adapter and the fitted readouts are held fixed. n_pairs = 8,306 per chassis, equal to kstep.

## The question
The present level-1 state reads the 2 s yaw rate only through the state. Its transition-only slope s_swap has a CI that includes 0 (amendment section 1.4). Can the frozen R35 adapter's *unpooled* tokens do better? The feature sets are:
* (a) the 40-token mean;
* (b) the 10 column means;
* (c) all 40 tokens.

## Verdict: FAIL on all four cells, both chassis
PASS needed, for (b) or (c), on BOTH chassis: the paired yaw dR2 against (a) with its CI wholly above 0, AND the yaw s_swap CI wholly above 0. Four cells were tried, uncorrected.

| cell | R35_s0 dR2 yaw vs (a) | s0 s_swap yaw | R35_s1 dR2 yaw vs (a) | s1 s_swap yaw |
|---|---|---|---|---|
| linear D, (b) | −0.0143 [−0.0268, −0.0035] | −0.0072 [−0.0165, +0.0010] | −0.0178 [−0.0293, −0.0076] | −0.0101 [−0.0184, −0.0027] |
| linear D, (c) | −0.0281 [−0.0520, −0.0067] | −0.0069 [−0.0231, +0.0083] | −0.0326 [−0.0545, −0.0130] | −0.0120 [−0.0264, +0.0022] |
| MLP pair, (b) | +0.1080 [−0.0286, +0.2345] | +0.0440 [−0.0117, +0.0943] | +0.1313 [−0.0055, +0.2607] | +0.0338 [−0.0125, +0.0764] |
| MLP pair, (c) | −0.7874 [−1.2249, −0.4779] | +0.0131 [−0.0440, +0.0658] | −0.9101 [−1.3797, −0.6068] | +0.0110 [−0.0331, +0.0518] |

Reference R² on yaw for set (a): linear D −0.0076 (s0) and −0.0084 (s1); MLP pair +0.1449 and +0.1142.

* The closest cell is the MLP on the column means. Its dR2 point is +0.11 and +0.13, but both CIs reach below 0, and its s_swap CIs include 0.
  * HYPOTHESIS only: the column layout may carry some state-level yaw. Nothing here shows it carries the yaw *transition*.
* The MLP on all 40 tokens (10,240-d input) overfits: yaw R² is −0.64 and −0.80.

## Admissibility: ADMISSIBLE (control `out/lattok_ctrl.json`)
The pod cache is a bf16 re-encode, not box 2's cache_full: 0 of 141 clip md5s match, and pod z1 differs from the banked z1 by rel L2 0.0152 (s0) and 0.0148 (s1). The control runs kstep's own readouts on the banked z1 and on z1 projected from the pod tokens (`wm.z1_from_mean(set a)`). The tolerance is ±0.02 R², fixed before the run.

| check | R35_s0 | R35_s1 |
|---|---|---|
| banked z1, ridge on D (acc), minus kstep r2_readout_acc | 0.0000 | 0.0000 |
| banked z1, MLP pair (yaw), minus kstep r2_readout_yaw (0.2003 / 0.2406) | 0.0000 | 0.0000 |
| pod minus banked z1, ridge acc R² | −0.0004 | −0.0004 |
| pod minus banked z1, MLP yaw R² | −0.0034 | −0.0035 |

The bf16 re-encode moves the readouts by at most 0.0035 R², so the verdict does not depend on the cache.

⚠️ For the record: the first admissibility ruling (LOOP_STATE ~14:15Z) quoted a linear-D yaw reference of −0.048 / −0.050. That number came from the worker's hand-back and appears in no kstep file. Root cause: an INHERITED number placed in a pre-registered control. The control above replaced it before its result was read.

## Next levers (Rule Zero)
1. **Adapter training** for lateral. This is the pre-registered FAIL branch: 2 s heading change is not readable from the frozen adapter's tokens as a transition. It needs an architecture or training card and the PI's go. Named, not run.
2. **LON-PLAN** (longitudinal, zero training): the prior-regularised planner on R35_s{0,1}. Blocked on the HWM-PLAN λ-dose amendment, which is the MM's to draft.
3. The ACTRSP arms stay held until the PI picks Option A or B. This result is evidence for A on the lateral channel: no level-1 or token-level readout calibrates yaw.

## Files
* Scripts: `lattok_feats.py`, `lattok_analysis.py`, `run_lattok.sh`, `summarize_lattok.py`, `make_ctrl.py`, `lattok_ctrl.py`.
* Logs and outputs: `smoke_log.txt`, `HANDOFF.md`, `deletions_lattok.log`, `out/lattok_result.json`, `out/lattok.log`, `out/lattok_ctrl.json`, `out/lattok_ctrl.log`.
