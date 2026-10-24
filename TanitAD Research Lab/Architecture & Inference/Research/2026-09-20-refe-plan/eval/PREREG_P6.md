# P6: the paper-faithful full REFe run on one A40 (pre-registration, IN FORCE 2026-10-15)

> ⭐ **IN FORCE. Finalised 2026-10-15 ~17:15Z, BEFORE P6's first training step.**
> - P6's first GPU stage waits for P2-s1, which was training at 16:30Z (GPU 100 %).
> - Both preconditions hold: (1) P2 R2 is GO (`eval/RESULT_P2.md`, `78e9bc9`); (2) P6-T, P6-N, P6-Y and P6-D were decided on 2026-10-11,
>   and P6-C on 2026-10-14.
> - **From here on, nothing below may change without a dated amendment written before the read.** The DRAFT notes further down are kept as history.
>
> **The 11-vCPU configuration, MEASURED on the pod** (`/workspace/data/refe_p6/chain.log`, `lab_early.json`; ⛔ pod paths, mirrored into this
> amendment because the pod is not durable):
> - **Code.** Ship commit `963638a`, ship-tree sha256 `3a226316…`, `p6_pod.sh` md5 `cb627bfe075e` (= the chain's own start line).
> - **Launch.** 16:13:29Z with `P6_FREE_NAVLOGS=1 P6_YAW_WRAP=off P6_DD_GB=64`, in **overlap**: CPU-only stages run beside P2-s1 at nice 19,
>   and the first GPU stage waits for it.
> - **Cores 11** (cgroup v1 cfs): **label workers 7, cache workers 11**. Overlap cap 8 (`ZZOVERLAP_CAP`, census burn 1.80–1.81 cores).
> - **Early G-LIVE-LAB**, on the first 3 logs' D9 parts, beside P2-s1's trainer:
>   - **identity:** 64 of 64 compared sets are bit-identical to the offline labeller (256 labelled, 86 distinct tokens);
>   - **stale proposal:** a one-step-stale proposal was refused;
>   - **throughput:** **10.887 sets/s at 7 workers**, against the gate's **6.735** (1.5 × the needed 4.49). It was measured beside a training
>     run, so it is a LOWER bound on the pool-alone rate;
>   - **verdict:** FIT on both.
>   - The launch gate re-measures G-LIVE-LAB at launch, and **that** reading binds. This one only de-risks it.
> - ⚠️ **An instrument defect, with no effect on the run.** The chain logged the early reading as `ZZLAB_BAR bad … identity=False`.
>   - The cause: `p6_overlap.lab_bar` demanded `identity_equal_sets == n_sets` (256). The labeller compares `min(n_sets, 64)` sets, and the
>     gate's `judge_lab` needs ≥ 64.
>   - Nothing refuses on that line: the chain's refusal reads `ZZLAB_IDENTITY_PASS`, and it passed.
>   - Fixed in `refe/p6_overlap.py` in the same commit as this amendment. `lab_bar` now equals `judge_lab`'s D1(a)/D1(c) on every fixture
>     (`test_p6_overlap.py` 54, with the old rule RED).
> - **Seed floor F.** P2-s1 started training at 16:13:26Z, and F is read from it under the rule below. If it yields no valid read, the S1
>   no-head arms are the fallback, as written.
> - **Related since the draft, not a change to this arm.** The S1 aux drivable-area head was NOT promoted (`raw/2026-10-15-s1-read/`): frozen
>   visual features reach navtest P1 IoU ~0.50 against K2's 0.80. This bears on P7.1's DESIGN (below), not on P6: a P7.1 head on a frozen trunk
>   inherits that ceiling unless the scene-tap probe (`raw/2026-10-15-scene-tap-probe/`) or a trunk-trained variant says otherwise.

> ⛔ **DRAFT. It is not in force until both of these hold:**
> 1. P2 has a verdict under `eval/PREREG_P2.md`.
> 2. The PI has decided P6-T, P6-C, P6-N and P6-Y (`SPEC_P6_PAPER_FAITHFUL.md` §7).
>
> It is then finalised and **landed in git before the first P6 training step**. After landing, nothing below may
> change without a dated amendment written before the read.
>
> Design: `SPEC_P6_PAPER_FAITHFUL.md`. Confidence: `P6_CONFIDENCE.md`.
> Written 2026-10-11.
>
> **State on 2026-10-14 (still a DRAFT, so edited in place):**
> - Precondition 1 **holds**. P2 R2 is **GO**: +0.57 PDMS [+0.02, +1.18] over u400 on the full navtest (`eval/RESULT_P2.md`, landed `78e9bc9`).
> - Precondition 2: the PI decided P6-T (25 epochs), P6-N (command), P6-Y (unwrapped) and P6-D on 2026-10-11.
>   **P6-C was decided 2026-10-14:** the PI said, verbatim, *"Find a solution with our 11 cpu pod"*. P6 runs on the existing A40 pod, which has an 11-vCPU cgroup.
>   - The feasibility plan and a CPU-only pod probe are in build (`eval/P6_ON_11VCPU.md`, `refe/p6_cpu_probe.sh`).
>   - The probe runs in a non-training window before P6. The chain's own G-LIVE-LAB gate (≥ 6.735 sets/s at the launch worker count) is unchanged.
> - **The P2 seed replicate is ordered** (PI 2026-10-14, *"Replicate p2"*; `eval/PREREG_P2_REPLICATE.md`). It supplies P6's training-seed floor F.
> - Pod order (PI): P2 ✅ → RL-S1 (training since 2026-10-14 04:33 Berlin) → S1 aux fine-tunes → **P2-s1** → P6.
> - Still to do before P6's first step: finalise this file (the 11-vCPU configuration, once the probe has measured it) and land it.

## The arm

**Data and schedule.**
- REFe ViT-L/16 trained from DINOv3 with fresh rank-32 Q/V LoRA and fresh heads.
- The final run's bank: 189,888 tuples over 103,037 scenes, full from step 0.
- 25 scene-epochs = 10,075 optimiser steps at 4 × 64 (25 × ⌈103,037 / 256⌉; corrected pre-data on 2026-10-11 from 10,063).

**Optimiser.** AdamW, lr 2e-4 cosine to 0, no warm-up, wd 0.01, clip 1.0, bf16 + TF32 (declared departure), seed 0.

**Scorer.**
- Online NAVSIM PDM targets on each sample's own current proposals, at lag 0 (labeller L1).
- Context un-detached; candidates detached.
- Six components, selected by `navsim_v1`.

**Inputs and losses.**
- Command nav input: P6-N, decided. It is `--nav-mode command`, hard-wired in the argv of `refe/p6_pod.sh`.
- Unwrapped yaw L1: P6-Y, decided. It is `--yaw-wrap off`, from `P6_YAW_WRAP`, whose default is `off`.
- The launch-gate token binds both, because it is minted over the exact argv.

**Launch.** Only on a launch-gate PASS token (SPEC §5).

## Primary endpoint

**Full navtest PDMS:** 12,146 tokens over 136 logs, NAVSIM v1.1.

**Eval settings.**
- The harness is the one that produced `eval/RESULT_SFT4.md`, on the dev box and never on the pod.
- The checkpoint is `model_final.pt`.
- The plan transform is the same one used for the labels (A7 repair ON).
- Route fix: N/A in command mode, ON in goal-point mode.

**Tier.** Pseudo-simulation of an open-loop plan against logged agents. **Not closed loop.**

**Comparators**, both paired on the identical tokens:

| comparator | PDMS on the full navtest | source |
|---|---|---|
| **C1: SFT-4 arm B u400** | **88.25** | `eval/RESULT_SFT4.md`, MEASURED, with its own seam: route fix ON, A7 ON, `navsim_v1_lane` |
| **C2: P2** | **88.82** | `eval/RESULT_P2.md`, MEASURED 2026-10-14, with its own seam: goal fix `pdm_route`, A7 ON, `navsim_v1` |

The comparison is **system against system**: each arm runs with its own registered selection rule.

**Estimator.**
- Paired log-cluster bootstrap over the 136 logs.
- **10,000 resamples, seed 20260927.**
- Reported for the difference and for P6's level.

## Rule (committed now)

**Versus C1 (and, separately, C2):**
- **ADOPT** iff the lower bound of Δ > 0 AND no longitudinal or lateral family component separates adversely.
- **REFUTED** iff the upper bound of Δ < 0.
- Otherwise **NOT PROVEN**.

**Against the goal of 94.55** (PUBLISHED, Table A13):
- **GOAL MET** iff P6's point estimate ≥ 94.55 AND ADOPT vs C1.
- **GOAL NOT EXCLUDED** iff the point estimate < 94.55 ≤ the upper bound of P6's level CI.
- **BELOW GOAL** iff the upper bound < 94.55.

⚠️ REVIEW_7 §1.2 bounds the harness difference to the paper at ≤ ~0.25 PDMS (ESTIMATED, our harness stricter). It is
**reported and never subtracted**.

## Four families (binding; per family, never pooled)

The instrument is the one that produced `sft4_B_u400_families.json`. It runs on the identical tokens, and each tested
component carries a paired CI.

| family | components |
|---|---|
| **LONGITUDINAL** | speed MAE and bias; along-track MAE and bias; EP and TTC sub-scores; headway/time-gap to the lead agent where a lead exists. Each comes with its n and, when there is no lead, the reason. |
| **LATERAL** | heading MAE; cross-track MAE; curvature MAE; yaw-rate MAE; DAC sub-score. **Last-pose heading error** is reported separately, as the D5 check. |
| **TACTICAL** | lateral, longitudinal and 5-way decision accuracy/κ; **command adherence** of the selected proposal; **selection quality** as random / actual / best-of-64 on the same tokens (a random control is mandatory). |
| **STRATEGIC** | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25). This is not a pass, and it is not omitted. |

## What the interval answers, and the replicates

**The bootstrap answers one question only:** *would another draw of logs say this?*

**Inference variance.**
- REFe inference is deterministic. This is **asserted**: a second full eval of the same checkpoint must be
  bit-identical in per-token PDMS.
- A mismatch is a defect, investigated before any read.

**Training variance (H-ESTIM-SEED-1).**
- A second 7-day P6 seed is not in budget.
- The seed floor is therefore taken from the **cheapest available replicate**, in this order:
  1. P2's training-seed replicate (`eval/PREREG_P2.md`, on ADOPT).
     - ⚠️ 2026-10-14: P2 was ADOPT and GO. `PREREG_P2.md`'s ADOPT clause ("the seed replicate, then P6") still stands, because Amendment 1 replaced only the NO-GO clause.
     - **Ordered by the PI on 2026-10-14** as P2-s1 (`eval/PREREG_P2_REPLICATE.md`). It runs after S1 and before P6. Its floor is **F = |mean Δ_seed|**, a one-draw estimate.
     - Cost: about **21 A40-h** (MEASURED for P2: 42 half-hourly egress ticks), plus a 5.3 h dev-box navtest read.
     - ✅ **READ 2026-10-16 (a VALID read, all 25 gates PASS): F = 0.33 PDMS×100**. Δ_seed = +0.33 [+0.04, +0.64]; P2-s0 88.82, P2-s1 89.15 (`eval/RESULT_P2REP.md`). P2-s1's own outcome is NOT REPLICATED (adverse TTC), which does not void the floor. Item 2 below is therefore not needed.
  2. Only if P2-s1 yields no valid read: the three S1 no-head arms (`N-s0/N-s1/N-s2`: the SFT-4 recipe on u400, 400 updates, seeds 0–2; `PREREG_AUX_DRIVABLE_S1.md`).
     - They are read on the full navtest on the dev box, at about 5.3 h each (MEASURED for u400).
     - This read belongs to P6, not to S1: S1 computes no navtest PDMS (its §4.2), and N carries no head, so the read cannot touch S1's H2 question.
     - ⚠️ It is a **scorer-only** floor (frozen trunk). It may understate the seed variance of a joint fine-tune, and is reported as such.
- A lever claim needs **Δ's lower bound > that floor**.
- If the floor is unmeasured at read time, P6 is reported as **"separated, single seed, not yet a lever claim"**.
- A P6 seed-1 replicate is requested **only if P6 is ADOPT and the delta is within 2× the floor**.

**Reported, not gating:**
- the per-epoch snapshots on the 923 confirmation tokens;
- the in-run coverage, lag, stall and `op_drift` series;
- all six sub-scores.

## Next arm, by outcome (Rule Zero; committed now)

**GOAL MET:**
1. the seed replicate;
2. then the first beyond-paper arm, P7.1: a training-only map-supervised drivable-area/occupancy aux head (admissible,
   PI 2026-10-11);
3. then the scorer ensemble (P7.2).

**ADOPT, below or not excluding goal:**
- Deploy P6.
- The next lever is set by the sub-score gap to Table A13:
  - **EP gap largest** ⇒ the selection rule and progress calibration on P6's scorer (0 GPU, dev box);
  - **DAC/NC/TTC gap largest** ⇒ P7.1, then P7.2.

**NOT PROVEN:**
- Read the in-run series against the SPEC's expectations: coverage ≈ 100 %, lag 0, the scorer's held-out ranking.
- The cheapest next experiment is a **scorer-only PDM fine-tune on P6** (the SFT-4 recipe), which took SFT-4 +3.79.

**REFUTED:**
- Bisect P6 against SFT-4 by the deltas that differ: nav mode, yaw form, from-scratch vs warm start.
- Use 0-GPU dev-box reads first: the selection rule on P6's proposals, and goal vs command on the same tokens.

## Amendment 2026-10-22 (pod move), written BEFORE P6 resumes and before any read

**The endpoint, comparators, estimator, rule, goal bar, four families and next-arm table above are UNCHANGED.** This amendment only discloses how the run continues.

1. **Interruption.** The A40 pod stopped on 2026-10-20 at about 13:00Z when the RunPod credit ran out. Its volume could not be recovered.
   - The run resumes from HF `Sayood/tanitad-refe-p6-paper-faithful` `run/ckpt_last.pt` (pushed 12:52:56Z), at epoch 6 of 25.
   - The steps after that push are lost and are re-trained.
   - Whether the resumed data order equals an uninterrupted run's is **UNVERIFIED**.
2. **The bank is rebuilt, not copied** (`raw/2026-10-22-p6-newpod-rebuild/RESULT.md`).
   - The 74-twin cross-pod control was bit-equal.
   - Resume is allowed only if `train_grow_cmd_3c` reproduces rank1 `2a63acc2…` and rank0 `daa23b96…`.
   - The trainer's own preflight then requires `targets_sha256 8775e73e…`, and the ARGS sha `570619873b1ebaa0` is pinned.
   - A mismatch means **no resume**. A restart from step 0 would be a new arm with its own amendment.
3. **Disclosed deviations of the rebuild** (that package, §5 and Addendum 1):
   - a different CPU (Xeon Gold 6342);
   - the 3-C preflight reads the banked I1–I4 `controls.json`, backed by the bit-equal cross-pod control;
   - the prebuild chain ran at `bcc04b5`, identical to the tip up to `cache` except the resume-bank guard;
   - the DBs sit on local container disk;
   - `build_twin_parts_3c_pp.sh` adds `$PKG/refe` to PYTHONPATH, one line, for the acceptance check.
4. **Context for the read, registered now:** `D-REFE-SESSION-EXPOSURE-1` (navtest shares 0 of its 44 sessions with navtrain). P6's navtest read is therefore an unseen-session read, like every navtest read. It changes nothing in the rule.

## Amendment 2026-10-22 (b): read instrumentation, written BEFORE any P6 checkpoint touched navtest

**The endpoint, comparators, estimator, rule, goal bar and next-arm table are UNCHANGED.** This amendment fixes how the registered read is instrumented. Full text and evidence: `raw/2026-10-22-p6-read-gaps/MM_RULING_P6_READ_GAPS.md`.

1. **Scoring caller.** The tip's post-NT-FIX1 caller runs with the banked harness under `eval/harness/`. It is hash-pinned, and `stage2_sft.py` refuses a mismatch before taking the GPU lock.
   - Its declared one-line drift is `RUN_V1_FIX1_ALLOW_DRIFT=1`.
   - **Read-day preflight:** the 24-token u400 control must reproduce csv sha256 `a71dd17c…`, or the read stops before any P6 output exists.
2. **C2 comparator.** It is in `stage3_full.py`. The control reproduces the published C2 − C1 = +0.571 [+0.018, +1.176] bit-exactly from banked per-token tables.
3. **TACTICAL rows.**
   - Selection quality (random / actual / best-of-64) comes from the `--dump-proposals` dump. Its instrument reproduces P2's 73.0031 / 88.8184 / 98.2311 to 1e-9.
   - **Command adherence is DEFINED here, because line 125 named it without a definition:**
     - **Classing:** the canonical labeller on the executed (A7) plan.
     - **Statistics:** accuracy and Cohen's κ, on all tokens and on the turn-active subset (LEFT ∪ RIGHT commands, 4,076 tokens). Turn-active accuracy is the headline cell.
     - **Always reported:** the 3×3 confusion matrix, and a random-proposal floor from the dump.
     - **Interval:** a paired log-cluster bootstrap against C1 and against C2 separately.
     - **Authority:** none over the rule.

## Amendment 2026-10-25 (c): rollback to step 2,603 after the twin-coverage defect, written BEFORE the re-run trains a step

**The endpoint, comparators, estimator, rule, goal bar, four families and next-arm table above are UNCHANGED.** This amendment discloses a defect in the run after the 2026-10-22 resume and the PI's decision about it.

1. **The defect (MEASURED).**
   - Source: the banked `run/metrics.jsonl` on HF `Sayood/tanitad-refe-p6-paper-faithful`, main head `79b7ca3624ff`.
   - Scorer-target coverage per window:

     | steps | coverage |
     |---|---|
     | 0–2,602 | **1.0000** |
     | 2,603–3,220 | **0.8946** |
     | 3,220–4,025 | **0.8941** |
     | 4,025–4,332 | **0.8936** |

   - The labeller logged `no_parts` 29 of 256 sets at step 4,332, when the pod stopped (credit ran out, 2026-10-24 ~03:45Z).
   - In those 1,730 steps the 3-C lane-rank twins trained with the trajectory loss only.
2. **Cause.**
   - The new-pod rebuild driver built and D7-verified the twin parts into `metric_cache_twins_3c`.
   - It omitted POST_MERGE step (d) (`raw/2026-10-17-twin-bank-3c/post_merge/POST_MERGE.md`): link the KEPT twins' parts into `metric_cache`.
   - The chain's `lanes` stage then rebuilt the twins with the DriveRL route patch, which PDM-Closed never consults.
   - D7 read `DISTINCT_BAD`, differ 0/40 at K = 1, 2, 3.
   - `P6_D7_FALLBACK=1` quarantined 18,489 / 2,781 / 551 parts without a refusal.
   - Retraction `R67`.
3. **PI decision 2026-10-25 (option B): the 1,730 steps are DISCARDED, not continued.**
   - The run resumes from HF commit `fc2adb00b59e` `run/ckpt_last.pt`: step 2,603, sha256 `eda651447f31846a…`, `config.json` byte-equal to main's. This is the checkpoint the 2026-10-22 resume used.
   - The discarded state stays on HF (main head `79b7ca3624ff`, tagged before the re-run's first push). It is never read as P6.
4. **Before the chain starts:**
   - the kept twins' parts are linked: `pod_helpers.py parts` reads plan_ok, then apply ok with `dst_missing_after=0`;
   - the chain's own `p6_prep.py lane-distinct` reads OK at K = 1, 2, 3;
   - `P6_D7_FALLBACK=0`, so a D7 failure REFUSES;
   - a guard on the first labeller events requires `no_parts == 0`.
5. **Pod.**
   - An A40, by RunPod migration of the 2026-10-21 volume. The cgroup CPU quota is 7.65.
   - The 1,192 nuPlan DBs are re-fetched from the public nuPlan bucket to local disk, with the same fetcher and the same 1,192/1,192 plan.
   - As in Amendment 2026-10-22 §1, whether the resumed data order equals an uninterrupted run's is **UNVERIFIED**.
6. **Speed work, committed now.**
   - The PI ordered a 50 % cut in step time. Compute-only changes may be applied, at a checkpoint boundary only, if both of these hold:
     - they are shown math-equivalent: fixed-batch loss, gradient and one-step parameter delta within the baseline's own run-to-run floor, and online labels bit-identical;
     - they are disclosed in a further amendment before they run.
   - A lever that changes the recipe (resolution, proposals, frozen layers, precision below bf16, schedule, sampling) is not P6. It is a new arm and a PI decision.
