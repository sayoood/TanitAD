# AMENDMENT 1 to PREREG_SEL_CONFIRM.md (sha256 ad12c2d9...): the frame control's reference

**Date:** 2026-11-08 ~08:15Z, Master Mind.
**Written before any fresh-token score was read.** Only the 6-token smoke identity has been seen. The rule, the tokens, the strata, the bars and the power are unchanged.

## Finding (MEASURED, `fc_diag/FC_FINDING.md`, `fc_census.json`; 5,877 tokens, 75 logs)
The dump stopped on the frame control (card section 6: <= 1 mm on every token; `selc_dump.py:277`).
- 17 tokens fail (0.29 %), in 2 logs: `2021.06.03.12.02.06_veh-35_00233_00609` (9 tokens, up to 5.1237 m) and `2021.06.03.17.06.58_veh-35_02943_03220` (8 tokens, up to 2.4174 m).
- Cause: the W3 export takes the next 8 frames of the OpenScene ~2 Hz list, and these logs lack one frame there (a ~1.0 s gap). From the gap on, export step k is the pose at (k+1)*0.5 s, while the DB future is sampled on the nominal 0.5 s grid.
- Pairing each export step with the DB pose at that step's own meta timestamp gives <= 4e-15 m on all 17. Anchor timestamps match on all 5,877.
- Model inputs, proposals, picks and PDMS do not read the DB future or `human_future_poses`. The only consumer is the secondary human-ADE (`selc_analyze.py:123-137`).

## Ruling: option A
1. **The frame control compares the export against the DB future sampled at the export's own meta timestamps.** The bar stays <= 1 mm on every token. The control's purpose is to prove that the seam's ego frame is right; a missing frame in the export's time grid is not a frame error.
2. **The secondary human-ADE masks the 17 named tokens** (listed in `fc_census.json`) and reports the n masked. It decides nothing.
3. **The check must stay able to fail:** a RED arm with a one-step (0.5 s) time shift, and a RED arm with a rotated frame, must both fail it. The old reference must reproduce exactly the 17 failures.
4. The 14 finished chunks stay valid, since their control passed under the stricter reference. `CODE_PIN.json` is re-pinned, and the chain resumes idempotently.

Option B (exclude the 17 tokens) was rejected: it would change the token list, the strata and the power to protect a check whose purpose is not at stake.
