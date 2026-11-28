# H-P6-SELECT-ARM-1 result at nomination: outcome A2, NO GAIN (2026-11-10)

Card: `PREREG_P6_SELECTORS_ARM.md` (sha256 ffc3483e44695ebd..., checked on disk 2026-11-10), which runs the nine variants of `PREREG_P6_SELECTORS.md` (sha256 d4778ca40be22cbe..., checked by the tool) with Amendment 1's speed proxy. All numbers MEASURED on navtrain FIT logs: `nominees.json`, `split.json`, `variants/*/meta.json`, `fit_selectors_C0.log` in this folder. Tier: FIT-B is a selection set, not a driving tier. Estimator: paired log-cluster bootstrap on FIT-B, 10,000 draws, seed 20260927, which answers "another draw of logs" only.

**Arm.** C0. `PREREG_SCORER1P6.md` adopted no arm (outcome O5), so card section 1 names C0.

**Inputs.**
- C0's logits on the 8,489 FIT sets: `fit_arm_logits.py` on the L40 pod, 2026-11-10, rc 0, `ZZFITARM_OK`. The file records pack sha256 `e58653c5...`, equal to the pod's `fit_pack.npz`.
- The four arm files (C0, C0r, S1E, P6_control) were copied to the dev box and checked by sha256 against the pod.
- `make_arm_pack.py` replaced only `logits`. Every other array was checked equal by content on the written file. C0 pack sha256 `c023e31b0b9706a4...`.
- Control: the same join on `P6_control` reproduces the original pack's logits within 5.1e-7 (rounding of the 6-decimal pack).
- C0's logits differ from P6's by a mean absolute 0.36 to 0.90 per head.
- Proposal identity (card section 1): the pod tool refused any proposal off by more than 2e-5 m or rad, and passed.

**Split.** The same FIT-A/FIT-B split as the first card. The file's sha256 is `ee53393f...` and not `8a22f2e0...` only because `split.json` embeds the pack sha256. `logs_A`, `logs_B`, `n_sets_A` (5,975 sets, 74 logs) and `n_sets_B` (2,514 sets, 27 logs) are identical, field by field.

**Verdict as written.** No family has a variant with CI-lo > 0 against C0's own rule on FIT-B, so none nominates. Nothing is read on E2 or E3, and K stays 0. Card section 4 commits this to **A2 NO GAIN**: selection on outputs is closed as a lever on this model, in sample and partly out of sample. The remaining levers are Batch 2's.

| variant | FIT-B PDMS x100 | delta against C0's rule [95 % CI] |
|---|---|---|
| R1 rule shape | 93.486 | +0.036 [-0.211, +0.224] |
| R2 rule shape | 93.573 | +0.123 [-0.200, +0.418] |
| C_platt | 93.439 | -0.011 [-0.161, +0.104] |
| C_isotonic | 93.573 | +0.123 [-0.071, +0.277] |
| C_platt_binned | 93.496 | +0.046 [-0.086, +0.191] |
| S_gbdt_pdms | 93.398 | -0.052 [-0.339, +0.241] |
| S_gbdt_heads | 93.442 | -0.007 [-0.360, +0.254] |
| S_mlp_heads | 92.416 | -1.034 [-1.670, -0.443] |
| P_pg | 93.778 | +0.328 [-0.044, +0.657] |

C0's own rule on FIT-B: 93.450. P6's rule on the same sets in the first card: 93.707. That comparison is unpaired here, and P6's heads trained on these logs while C0's scorer did not, so it is descriptive only.

## What this does and does not say

- "Partly out of sample" (card): C0's scorer weights never trained on the FIT logs, but the trunk and the proposal generator are P6's and did see them. This test cannot reach a scorer that reads the scene better. It shows that re-weighting or re-ranking the six heads' outputs does not raise PDMS by a measurable amount, with P6's logits or C0's.
- The policy-gradient selector moved from -0.013 on P6's logits to +0.328 on C0's, and its interval still holds zero. That is not a nomination, and it is not carried as a finding.
- No navtest number was produced, so no driving claim is made and the four metric families have nothing to report. STRATEGIC is PAUSED (PI 2026-09-19), not a pass.

## Next lever (RULE ZERO)

1. **S1Z** (`PREREG_S1Z_P6.md`): a pair loss on the near-twin failures, changing what the scorer reads, not how its outputs are combined. Its first pod run was refused at the preflight on 2026-11-10 because G18 probed a micro-batch without a pair. The gate fix (`s1z/fix2/`) is landed with this result, and the re-run is the next pod step.
2. O5's committed replicates: S1E seed 1 and a second C0 replicate.
3. Batch 2 in Shapley order: dose (1,600 updates), privileged-teacher distillation, the constrained-RL arm. Pod time for these is checked with the PI first.
