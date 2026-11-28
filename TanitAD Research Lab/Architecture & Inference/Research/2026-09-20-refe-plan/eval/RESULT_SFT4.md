# SFT-4 result: arm B at update 400 is ADOPT on the 923 confirmation tokens and CONFIRMED on the full 12,146-token navtest (88.25 vs 84.46 PDMS; a registered deviation; the recipe is unread)

> Pre-registration: `eval/PREREG_SFT4.md`, Amendment 5 (landed in `1730bc29` at 07:46:50 UTC on 2026-10-11, before
> the read started at 07:47:31 UTC).
> Raw: `raw/2026-10-05-sft-stage2/sft4_B_u400/` (`result.json`, `score_status.json`, `seam.log`, the families of both
> arms). The two seam `.npz` files stay on the dev box (`E:/Projects/TanitAD/data/refe_navtest/seams/`).
> Why only this checkpoint: `STOP_CONSEQUENCES_2026-10-11.md`. The pod stopped, and nothing of SFT-4 beyond update 400
> left it.
>
> **Tier:** NAVSIM v1.1 PDMS, ego pseudo-simulation of an open-loop plan against logged agents. It is not closed loop.
>
> **Interval:** paired log-cluster bootstrap over the 43 confirmation logs, 10,000 resamples, seed 20260927. It answers
> *another draw of logs* only. There is one fine-tune seed and inference is deterministic, so training variance is
> unmeasured (H-ESTIM-SEED-1).

## Verdict (the registered rule): ADOPT

ADOPT iff the lower bound > 0 AND no longitudinal or lateral family component separates adversely; REFUTED iff the
upper bound < 0; otherwise NOT PROVEN.

- The checkpoint is `model_sft4_B_u400.pt`, md5 `8e382eeb…`, 1,288,489,838 B. It was built by
  `eval/build_sft_ckpt.py` from `scorer_states_u400.pt` (md5 `4dded9b1…`) on `model_final.pt` (md5 `b54773f8…`).
- The selection rule is `navsim_v1_lane`, in the unchanged seam (route fix ON, A7 ON).

| | deployed | SFT-4 arm B, update 400 |
|---|---|---|
| PDMS on the 923 tokens | 85.61 | **88.65** |
| paired Δ ×100 [95 % CI] | – | **+3.04 [+1.21, +4.87]** |
| better / worse / tied | – | 339 / 271 / 313 |
| picks scoring 0 | 56 | 36 |
| logs | 43 | 43 |

MEASURED, `raw/2026-10-05-sft-stage2/sft4_B_u400/result.json`.

**The arm's PDMS sub-scores** (×100, 923 tokens):

| NC | DAC | EP | TTC | C | DDC | PDMS |
|---|---|---|---|---|---|---|
| 98.59 | 97.51 | 81.72 | 95.23 | 99.89 | 97.35 | 88.65 |

**Gates: all pass.**
- Harness: status PASS, 923 / 923 tokens, 0 failed, 923 valid CSV rows, C1 max |Δ| 0.0.
- Seam: rc 0 in 1,236 s, 923 rows, 0 misses. Frame control: max 7.3e-15 m, median 9.0e-16 m, against a 0.001 m bar.
- Consistency: the 86 tokens whose executed poses equal the deployed poses all score identically (0 differ).

## Four families (families6, on the same 923 tokens)

| family | component | deployed | arm B u400 |
|---|---|---|---|
| longitudinal | speed MAE (m/s) | 0.857 | 0.749 |
| longitudinal | along-track MAE (m) | 1.226 | 1.066 |
| longitudinal | speed bias (m/s) | −0.301 | −0.330 |
| longitudinal | along-track bias (m) | −0.440 | −0.490 |
| lateral | heading MAE (°) | 2.084 | 1.924 |
| lateral | cross-track MAE (m) | 0.325 | 0.276 |
| lateral | curvature MAE (1/m) | 0.0207 | 0.0188 |
| lateral | yaw-rate MAE (°/s) | 2.774 | 2.640 |
| tactical | lateral decision: accuracy / κ | 0.945 / 0.848 | 0.951 / 0.866 |
| tactical | longitudinal decision: accuracy / κ | 0.698 / 0.513 | 0.729 / 0.563 |
| tactical | 5-way collapsed: accuracy / κ | 0.775 / 0.688 | 0.787 / 0.703 |
| strategic | – | UNAVAILABLE in NAVSIM (no strategic decision is scored; the arm has no strategic layer) | same |

MEASURED, `sft4_B_u400/deployed_families.json` and `sft4_B_u400/sft4_B_u400_families.json`.

- **None of the 21 longitudinal / lateral components separates adversely** (`families_adverse_arm_vs_deployed`: `[]`).
- Every MAE point estimate improves.
- The two biases grow slightly: the arm is a little more conservative along-track (−0.49 m vs −0.44 m) and in speed
  (−0.33 vs −0.30 m/s).
- The tactical rows carry no interval, so they are reported, not tested.

## What this licenses, and what it does not

**It licenses one sentence:** *this checkpoint (arm B at update 400) selects better than the deployed scorer on SPEC_NAVTEST
Amendment 5's 923 confirmation tokens.*

**It does NOT license "the SFT-4 recipe works".** The five deviations, stated before the read (Amendment 5), all hold:
1. **A non-final checkpoint.** It is update 400 of 2,335. The registered read was at the final update.
2. **Stage 1 never selected it.** B is read because it is the only survivor. At update 800 the two arms tied on
   held-out (+2.27 each; INHERITED from the handoff, since no SFT-4 log was banked).
3. **Incomplete labels.** It trained on 146,185 of 174,912 PDM labels (Amendment 4).
4. **Its exploratory read had been seen.** W3's 200 tokens (+4.24 [+1.36, +7.39]) share no token with the 923
   (MEASURED, intersection 0), but they informed the decision to run this read.
5. **One fine-tune seed and deterministic inference.**

**The 923 tokens are being reused.** This is their second scorer read after SFT-3 arm A (−0.20 [−1.30, +0.92]), and
SFT-4R will be the third (`eval/PREREG_SFT4R.md`). Each read is a fixed rule that tunes nothing on them, but repeated
reads on one confirmation set accumulate selection.

⇒ By `PREREG_SFT4.md` Amendment 5 and `PREREG_SFT4R.md`, an ADOPT here **requires the full 12,146-token navtest before
any claim**. It is a confirmation, not a courtesy.

**For comparison** (EXPLORATORY, decides nothing): the W3 sub200 peek of the same checkpoint read +4.24 [+1.36, +7.39]
(84.35 → 88.59; better / worse / tied 84 / 46 / 70; zero picks 14 → 9). On disjoint tokens, the registered read is
smaller (+3.04) and its interval is narrower.

**Mechanism (ESTIMATED reading, not a claim):** SFT-3 arm A had the same base, the same trainable set and the same lr,
but the TEACHER'S labels, and it read −0.20 on these tokens. This arm read +3.04 after only 400 updates on the paper's
PDM targets plus the lane output. The difference is the label source, which REVIEW_7 diagnosed as defect 1. Whether
it is the labels or the lane output is NOT separable here: arm A of SFT-4 (no lane rule) did not survive the stop.

## Timing

| event | UTC |
|---|---|
| Amendment 5 landed (`1730bc29`) | 2026-10-11 07:46:50 |
| read started | 07:47:31 |
| seam done (1,236 s), then harness (109.4 s) and families | – |
| dev-box GPU lock released; exit 0 | 08:08:07 (10:08:07 Berlin) |

## Full navtest (12,146 tokens, 136 logs): CONFIRMED

> This is the registered confirmation read from "Next arm" §1 below. The rule, the token file and both deviations
> were committed before any score of this run was read: Deviation 1 in `7592771` at 13:34:21 UTC, Deviation 2 in
> `8638d4da` at 13:51:36 UTC, and the scoring rerun started after that.
> Raw: `raw/2026-10-05-sft-stage2/sft4_B_u400_navtest_full/` (`result.json`, `score_status.json`, `seam_report.json`,
> `seam.log`, the families of both arms, `unseen_subset.json`, `sensitivity_minus_c1.json`, and both failed attempts'
> results and logs). The two seam `.npz` files stay on the dev box.
>
> **Tier:** NAVSIM v1.1 PDMS, an ego pseudo-simulation of an open-loop plan against logged agents. It is not closed
> loop.
>
> **Interval:** paired log-cluster bootstrap over the 136 navtest logs, 10,000 resamples, seed 20260927. It answers
> *another draw of logs* only. There is one fine-tune seed and inference is deterministic, so training variance is
> unmeasured (H-ESTIM-SEED-1).

**Verdict: CONFIRMED.** The script prints ADOPT, which maps one to one onto CONFIRMED.
- The lower bound is above 0: +3.05.
- None of the 21 longitudinal and lateral components separates adversely.

`scored_as`: "the whole navtest filter (no --tokens)" (Deviation 2).

| | deployed (route fix, `R6_A1`) | SFT-4 arm B, update 400 |
|---|---|---|
| PDMS on all 12,146 navtest tokens | 84.46 | **88.25** |
| paired Δ ×100 [95 % CI] | – | **+3.79 [+3.05, +4.54]** |
| better / worse / tied | – | 5,167 / 3,188 / 3,791 |
| picks scoring 0 | 837 | 529 |
| logs | 136 | 136 |

MEASURED, `raw/2026-10-05-sft-stage2/sft4_B_u400_navtest_full/result.json`. The deployed side is the route-fix run's
own CSV on the same tokens. Its sub-scores come from `raw/2026-10-04-navtest-final/routefix_full/3_parse.json`.

**PDMS sub-scores** (×100, all 12,146 tokens):

| | NC | DAC | EP | TTC | C | DDC | PDMS |
|---|---|---|---|---|---|---|---|
| deployed (`R6_A1`) | 98.13 | 94.71 | 76.70 | 93.69 | 99.93 | 97.61 | 84.46 |
| **SFT-4 arm B u400** | **98.94** | **96.58** | **81.20** | **95.63** | **99.97** | **97.66** | **88.25** |
| HUMAN (`eval/RESULT_NAVTEST_FINAL.md`) | 100.00 | 100.00 | 86.96 | 100.00 | 99.90 | – | 94.55 |

- Every sub-score rises.
- EP moves the most (+4.49), then TTC (+1.94) and DAC (+1.87).

**Gates: all pass.**
- Harness: status PASS, 12,146 / 12,146 tokens, 0 failed, 12,146 valid CSV rows, C1 max |Δ| 0.0, 1,335.7 s.
- Seam: 12,146 rows, 0 misses, 17,496.8 s. It was written once, by attempt 1, and reused unchanged by both reruns.
- Consistency: the 1,226 tokens whose executed poses equal the deployed poses all score identically (0 differ).
- Frame control (reported, not gated): max 6.18063877792441 m, median 9.2e-16 m.
  - The max is the declared C1 set (36 tokens, 4 logs; Deviation 1).
  - It is identical to the deployed route-fix seam's max.

### Four families (families6, on the same 12,146 tokens, 136 logs)

| family | component | deployed | arm B u400 |
|---|---|---|---|
| longitudinal | speed MAE (m/s) | 0.889 | 0.740 |
| longitudinal | along-track MAE (m) | 1.258 | 1.035 |
| longitudinal | acceleration MAE (m/s²) | 0.580 | 0.519 |
| longitudinal | speed bias (m/s) | −0.328 | −0.230 |
| longitudinal | along-track bias (m) | −0.466 | −0.311 |
| longitudinal | ego-progress ratio, mean | 0.980 | 1.003 |
| longitudinal | distance-keeping (headway / TTC to a lead) | UNAVAILABLE: no lead-agent track in the seam | same |
| lateral | heading MAE (°) | 2.716 | 2.532 |
| lateral | cross-track MAE (m) | 0.426 | 0.378 |
| lateral | final cross-track MAE (m) | 1.017 | 0.865 |
| lateral | curvature MAE (1/m) | 0.0237 | 0.0199 |
| lateral | yaw-rate MAE (°/s) | 3.232 | 2.983 |
| tactical | lateral decision: accuracy / κ | 0.917 / 0.848 | 0.925 / 0.863 |
| tactical | longitudinal decision: accuracy / κ | 0.642 / 0.459 | 0.679 / 0.516 |
| tactical | 5-way collapsed: accuracy / κ | 0.770 / 0.701 | 0.783 / 0.718 |
| strategic | – | UNAVAILABLE in NAVSIM: no strategic decision is scored, and the arm has no strategic layer. The family is also PAUSED by the PI's 2026-09-19 ruling. | same |

MEASURED, `sft4_B_u400_navtest_full/deployed_families.json` and `sft4_B_u400_navtest_full_families.json`
(`families.refcv6`, cluster unit `log_name`, 136 logs, `_ci_coverage` complete).

- **None of the 21 longitudinal and lateral components separates adversely** (`families_adverse_arm_vs_deployed`:
  `[]`).
- Every MAE point estimate improves.
- **Unlike the 923-token read, both biases SHRINK here:**
  - along-track bias −0.47 → −0.31 m;
  - speed bias −0.33 → −0.23 m/s;
  - the mean ego-progress ratio crosses from 0.980 to 1.003.
  - On the full set, the arm is less conservative than the deployed scorer, not more.
- The tactical rows carry no interval, so they are reported, not tested.
- **Anti-echo:**
  - Both arms beat the hold-v0 control (`BEATS_HOLDV0`).
  - The copy detector is CLEAN: echo index 0.0019 for the deployed scorer and 0.0013 for the arm, against 0.0338 for
    the logged human.

**Proposals (open-loop ADE over NAVSIM's 8 poses vs the human future, `seam_report.json`; reported, decides
nothing):**
- 64 proposals per token; the arm's pick 1.186 m.
- Random pick (a random selector's expectation) 1.471 m; oracle best-of-64 0.334 m.
- The pick closes 25 % of the random-to-oracle gap.
- ⚠️ ADE against the human is not PDMS, so this ranks no lever on its own. It does show that the proposal set holds
  far better plans than the selector picks.

### Reported, not gating (committed in §1 before the run)

| read | tokens / logs | deployed → arm | paired Δ ×100 [95 % CI] | better / worse / tied | zero picks |
|---|---|---|---|---|---|
| **primary (gating)** | 12,146 / 136 | 84.46 → 88.25 | +3.79 [+3.05, +4.54] | 5,167 / 3,188 / 3,791 | 837 → 529 |
| unseen by any read of this checkpoint | 11,023 / 122 | 84.36 → 88.21 | +3.85 [+3.06, +4.66] | 4,744 / 2,871 / 3,408 | 767 → 484 |
| minus the 36 C1 tokens (Deviation 1) | 12,110 / 136 | 84.47 → 88.22 | +3.75 [+3.02, +4.51] | 5,141 / 3,186 / 3,783 | 832 → 529 |

MEASURED, `unseen_subset.json` (token-file md5 `ab59e479…`) and `sensitivity_minus_c1.json` (`bc813be8…`). Every
input gate passes: md5, n, logs, every token logged, 0 missing on either side.

- **The three reads agree.**
- The 11,023 tokens that no read of this checkpoint has touched give the same effect as the full set (+3.85 vs +3.79).
  ⇒ The 923-token confirmation set's reuse did not manufacture the result.
- Removing the C1 tokens moves the effect by 0.04.

### What this licenses, and what it does not

**It licenses:** *this checkpoint (arm B at update 400) selects better than the deployed scorer on the full NAVSIM v1.1
navtest: 88.25 vs 84.46 PDMS, +3.79 [+3.05, +4.54] over 136 logs, with no adverse longitudinal or lateral component.*
It is a deployment **CANDIDATE**. Deploying it is the PI's call.

**It does NOT license:**
- "the SFT-4 recipe works". Four of the 923-token read's five deviations still hold: a non-final checkpoint (1),
  never selected by stage 1 (2), trained on incomplete labels (3), and one fine-tune seed (5). Only deviation 4, the
  seen exploratory read, is answered here, by the unseen read above.
- a lever claim. That needs the training-seed replicate (§2 below; H-ESTIM-SEED-1).
- any statement about the labels versus the lane output. SFT-4 arm A did not survive the stop.

**Where it leaves the gap to the reference:**
- HUMAN is 94.55, so about 6.3 PDMS remain.
- Measured against HUMAN, the largest single sub-score gap is still **EP**: 81.20 against 86.96 for HUMAN
  (`eval/RESULT_NAVTEST_FINAL.md`).

### Timing (UTC, 2026-10-11)

| event | UTC |
|---|---|
| seam started | 08:37:22 |
| attempt 1 ends: 12,146 rows, 0 misses, then exit 1 at the frame-control gate (17,539 s) | 13:29:42 |
| Deviation 1 landed (`7592771`) | 13:34:21 |
| attempt 2: six harness attempts die on `WinError 206` | 13:34:40 → 13:46:44 |
| Deviation 2 landed (`8638d4da`) | 13:51:36 |
| scoring rerun: harness PASS (1,335.7 s), then families | → 14:15:46 |
| unseen and minus-C1 reads | 14:16:18 |

## Next arm (Rule Zero), in order

### 1. The full 12,146-token navtest of this checkpoint, on the dev box (0 pods). Rule committed here, before it starts. DONE 2026-10-11: CONFIRMED, +3.79 [+3.05, +4.54]; see "Full navtest" above.

- **Command:** `eval/stage2_sft.py --ckpt …/sft4_peek/model_sft4_B_u400.pt --name sft4_B_u400_navtest_full
  --rule navsim_v1_lane --tokens E:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json`. It runs in the
  unchanged seam against the deployed route-fix run's own scores on the same tokens (84.46 PDMS, `RESULT_NAVTEST_FINAL.md`).
- **Token file:** md5 `3a8b3653…`. It holds all 12,146 tokens, 136 logs: the deployed CSV's tokens, MEASURED equal to
  the export and the deployed seam.
- **Rule (unchanged from stage 2):** paired log-cluster bootstrap over the 136 logs, 10,000 resamples, seed 20260927.
  - **CONFIRMED** iff the lower bound > 0 AND no longitudinal or lateral component separates adversely.
  - **REFUTED** iff the upper bound < 0.
  - Otherwise **NOT CONFIRMED**.
  - The script prints its own verdict words (ADOPT / REFUTED / NOT PROVEN); they map one to one onto these.
- **Reported, not gating (committed now):** the same paired difference on the **11,023 tokens over 122 logs that no
  read of this checkpoint has touched**. These are the full navtest minus Amendment 5's 923 and W3's 200
  (`navtest_unseen_by_sft4u400_tokens.json`, md5 `ab59e479…`). It is computed from the same two CSVs, with no new
  inference.
- **What CONFIRMED licenses:** this checkpoint is a deployment CANDIDATE. Deploying it is the PI's call. The lever is
  still not claimed until the seed replicate below.
- **Cost:** ESTIMATED from this read's rate (1.34 s per token in the seam): ~4.5 h of seam, ~0.5 h to score, on the
  dev-box GPU, with no pod.
- **Deviation, registered 2026-10-11 BEFORE any score of this run was read.**
  - **What happened:** the run stopped as `INCOMPLETE (seam failed)` after 17,539 s. The seam had written all
    12,146 / 12,146 rows with 0 misses (`seams/stage2_sft4_B_u400_navtest_full.report.json`). It then exited 1 at its
    frame-control gate, which reported a max of 6.18063877792441 m against the 0.001 m bar.
  - **Why it is not this checkpoint:** the frame control compares the log's own future
    (`get_ego_future_trajectory`) against the export's `human_future_poses`. No planner is involved
    (`refe_navtest_seam.py`). The failing tokens are the declared 36 tokens / 4 logs nuPlan-DB vs OpenScene pose
    disagreement (`raw/2026-10-04-navtest-final/frame_control_navtest_full.json`, max 6.1806 m). The deployed
    route-fix seam that produced the 84.46 fails the same tokens with the identical max (`frame_control_parts_max_m`
    6.18063877792441). The 923-token read never hit it because none of the 36 were in it (max 7.3e-15, rc 0).
  - **Recovery:** rerun the identical command. `stage2_sft.py` reuses the written seam and goes straight to scoring.
  - **Primary read:** unchanged — all 12,146 tokens and the rule above. The deployed 84.46 also includes the 36.
  - **Sensitivity read, REPORTED NOT GATING:** the same paired difference without the 36 C1 tokens: 12,110 tokens
    over 136 logs, `E:/…/refe_navtest/navtest_full_minus_c1_tokens.json`, md5 `bc813be8…`, by
    `unseen_subset.py --tokens … --expect-md5 bc813be8 --expect-n 12110 --expect-logs 136`. If the two reads
    disagree on the verdict, the primary decides, and the disagreement is reported.
- **Deviation 2, registered 2026-10-11 BEFORE any score of this run was read.**
  - **What happened:** the rerun reused the seam and went to scoring. All six harness attempts died in
    `CreateProcess` with `WinError 206` (`raw/…/sft4_B_u400_navtest_full/score_attempt2_winerr206_{1..6}.log`, six
    identical tracebacks). The run ended `INCOMPLETE (the harness did not PASS)` with an empty harness status
    (`result_attempt2_winerr206.json`). Nothing was scored, so there is no number to have seen.
  - **Cause (MEASURED):** `--tokens` becomes one Hydra override, `scene_filter.tokens=[…]`. For 12,146 tokens it is
    230,812 characters, against Windows' 32,767-character command-line limit. The 923-token read stayed under it.
  - **Fix (`eval/stage2_sft.py`):** when the token set equals the whole navtest export set, `--tokens` is omitted and
    the navtest scene filter itself is the restriction. The set equality is MEASURED: the token file and the export
    hold the same 12,146 tokens. This is the path the deployed 84.46 was scored on. It is fail-closed: without
    `--tokens`, the harness's C2 guard demands that every token of the filter scores, with 0 failed.
    `result.json` records which path ran (`scored_as`). Any proper subset still passes `--tokens`.
  - **Unchanged:** the checkpoint, the seam (reused, 12,146 rows, 0 misses), the primary read, the rule, the
    estimator, the unseen read and the sensitivity read.

### 2. A training-seed replicate (H-ESTIM-SEED-1). Needs a pod.

### 3. SFT-4R and then P2, as in `PREREG_SFT4R.md`. Blocked on the PI's pod check and a write-scoped HF token.
- These run whatever step 1 returns.
- `refe/hf_egress.py` is SFT-4R's launch blocker, and it is 0-pod work.
