# SFT-4 result: arm B at update 400 is ADOPT on the 923 confirmation tokens (a registered deviation; the recipe is unread)

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

## Next arm (Rule Zero), in order

### 1. The full 12,146-token navtest of this checkpoint, on the dev box (0 pods). Rule committed here, before it starts.

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

### 2. A training-seed replicate (H-ESTIM-SEED-1). Needs a pod.

### 3. SFT-4R and then P2, as in `PREREG_SFT4R.md`. Blocked on the PI's pod check and a write-scoped HF token.
- These run whatever step 1 returns.
- `refe/hf_egress.py` is SFT-4R's launch blocker, and it is 0-pod work.
