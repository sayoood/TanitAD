<!-- Final report of a read-only survey sub-agent ("Extract REFe review lessons for transfer"),
transcribed verbatim from the message it delivered to the coordinator on 2026-09-27. Its transcript
file was empty at extraction time, so programmatic extraction was impossible. Snapshot b3f7ea6f.
Numbers are as the survey reported them; the coordinator's own spot-checks are listed in
../STATE_AND_GAPS.md. -->

# REFe → v7F / refav1: transferable lessons (read-only survey of `b3f7ea6f`)

## Executive summary
1. All six reviews covered. PAPER_CONFORMANCE_REVIEW is Review 1 (its "weight decay: not stated" row, `PAPER_CONFORMANCE_REVIEW.md:102`, is the one REVIEW_3 refutes at `REVIEW_3_FULL.md:127`); FIX_VERIFICATION_REVIEW is Review 2 (`REVIEW_3_FULL.md:43`, `FIX_VERIFICATION_REVIEW.md:59`). No standalone REVIEW_1/2 files exist.
2. Every round found blocking defects that the previous round's own tests passed; the later rounds won by primary sources, independent instruments, mutation arms, and RUNNING everything (incl. an end-to-end rehearsal of the pod scripts), not by reading harder.
3. REFe's trajectory is not yet a positive result: NAVSIM v1 PDMS (200-token subset) pick 31.33 → 49.40 over epochs 1–13, below the do-nothing STOP floor 62.58 at every point (`MODEL_REGISTRY.md:5246`).
4. The one lever that clearly worked is DIVERSITY: proposal spread 0.122 → 4.34 m as goal-augmented twins entered the bank; best-of-64 rose to 91.2 PDMS (`GOALS_AND_CLAIMS.md:15831-15832`) — coincident with training progress, not isolated.
5. SELECTION failed: skill 0.460 → 0.162 → 0.013 under fixed-candidate supervision; the on-policy switch then FAILED Amendment 4 (0.073 [−0.047, 0.199] vs 0.25), under-dosed (1,618 on-policy sets at the start of epoch 12); the epoch-5 "peak" was the old selection rule favouring short, cautious plans.
6. v7F (never trained) carries three of REFe's trunk-input defects today: the DINOv3 seed has no ImageNet normalisation (REFe's R3-1 class, found in this survey), a learned `pos` table left at random init, and CLS/registers dropped (both declared); its anchor is a frozen copy of that seed, not published DINOv3.
7. refav1's trunk path is clean at source (HF model, processor mean/std, RGB). Its failure is REFe's selection class: the planner returns an injected baseline on 282/282 windows and the shipped cost's optimum is the do-nothing plan; its fidelity gate `cost_fidelity` is a pooled cross-window estimator with no caller outside tests.
8. New beyond A16's seven questions: refav1 fits its frozen-feature standardiser on one batch although its docstring says "over the train corpus"; `eval/write_result_e6.py` hardcodes verdict prose that contradicts the epoch-5 table; REVIEW_4's "teacher not reproducible" used a false replicate (the rank override was an environment variable).
9. Integration escalations: REFe's live on-policy trainer, the NAVSIM labellers and three registry-cited eval results are absent from the repo at this ref; `stack/scripts/launch_gate.py`, which CLAUDE.md makes binding for any launch, does not exist at the ref (v7F and refav1 cannot launch under that rule); BACKLOG A16 is open and R24 is stale.
10. Cheapest positive-direction moves, all CPU, 0 training: v7F — fold mean/std into the seed's patch conv and measure the seeded trunk against published DINOv3 with REFe's `diag_rope.py` probe pattern; refav1 — an E-6-style oracle/random/pick table over the iCEM population on the banked checkpoint.

Evidence legend. INHERITED: every REFe, refav1 and v7F number is quoted from the cited `path:line` (the sources mark them MEASURED). MEASURED (source): facts established by reading code or grepping at the ref, each with a same-breath control. Tiers: REFe navtest = NAVSIM v1 PDMS (open-loop plan pseudo-simulated against logged agents); teacher = nuPlan closed-loop CLS; refav1 = open loop, self-action (registry re-label). Nothing was run except greps and `git cat-file`/`git show` at the ref.

## 1. Findings catalogue (condensed, by class)

| Class | Defect | How found | Severity | Fixed? / proof |
|---|---|---|---|---|
| Trunk input | Frozen DINOv3 fed un-normalised [0,1] images; rel L2 0.562 (worse than no position encoding, 0.529) | Displacement probe vs the correctly normalised trunk, with SELF/JITTER/BGR controls (`REVIEW_3_FULL.md:164-206`) | Blocking | Fixed (buffers in trunk, `REVIEW_4_FINAL.md:101`); guard arm catches it (`REVIEW_5_CAMERAS.md:459`) |
| Trunk position | Random frozen `pos` table where DINOv3 uses RoPE; exposed by 305,045,504 − 303,079,424 = 1920×1024 (`PAPER_CONFORMANCE_REVIEW.md:349-372`) | Parameter arithmetic + checkpoint key census | Blocking | Fix #1 WRONG: 0.760 displacement, worse than none 0.585 (`FIX_VERIFICATION_REVIEW.md:144-163`); fix #2 bit-identical, `ROPE_MATCHES_REFERENCE` (`REVIEW_4_FINAL.md:58-62`) |
| Trunk tokens | DINOv3's CLS + 4 registers dropped | Strict loader (`REFE_MODEL.md:87-92`) | High | Fixed; magnitude never measured |
| Trunk load | Trainer never loaded the trunk; ~26 A40-days on random weights (`RETRACTION_LOG.md:17297-17348`) | Building the model as `train.py` does, then mapping: 186/186 tensors moved | Blocking | Fixed; returns 3 on an incomplete map |
| 3D position encoding | Native 1920-px intrinsics on a 960-px input (31.34° vs 62.85° FOV); geometry-free; distorted images; one baked rig | Calling `_frustum` on a stub; OpenCV reference (`REVIEW_4_FINAL.md:228-269`; `REVIEW_5_CAMERAS.md:182-212`; `READINESS.md:45`) | High | Fixed; `diag_calib` 8/8 at 1.3–1.7e-5 m |
| Wiring | Registers concatenated, not compressing (1,936 tokens, identical storage pointer) | Forward hooks + `data_ptr` (`PAPER_CONFORMANCE_REVIEW.md:209-239`) | Blocking | Fixed; hook arm can go red |
| Wiring | Extra `visual_ctx.detach()` promoted to an "invariant" | Undeclared-departure audit (`REVIEW_5_CAMERAS.md:230-247`); grad 401.6 vs 0 on mutation (`REVIEW_6_FINAL.md:600-626`) | Medium | Declared as a flag; no mutation arm |
| Label / teacher | Rank-1 trajectories joined to rank-0 scores on 540/1,080 tuples | Joining the banks on a key census (`PAPER_CONFORMANCE_REVIEW.md:159-207`) | Blocking | Fixed; then rank 1 was a relabelled copy (0/1,091 differed, `GOALS_AND_CLAIMS.md:15815`) |
| Label / teacher | Camera frame paired with an ego pose from a closed-loop rollout; 83 % > 0.5 m away | External pose comparison (`GOALS_AND_CLAIMS.md:15816`) | Blocking | Per-frame re-seeded rollouts, `SIGNALS_CONSISTENT` |
| Label / teacher | Horizon halved (10 Hz log, stride); navtrain 2 s @ 10 Hz | Identity: displacement/horizon vs speed (`REFE_MODEL.md:94-101`; `RETRACTION_LOG.md:17350-17395`) | Blocking | Derived from `database_interval`; the fix was off by one (R17) |
| Label / teacher | Frozen NPCs manufacture collisions; the splice fix added zero-area phantoms and tail-indexing (+3.40 s) | Discriminating splice control; three-arm A/B/C (`REVIEW_3_FULL.md:210-258`; `REVIEW_4_FINAL.md:126-193`) | Blocking | Fix #1 introduced a new instance |
| Label / teacher | Comfort constant within frame; estimator switches at k ≥ 10; latches carry the artefact past the filter | Per-prefix tables (`REVIEW_3_FULL.md:262-320`; `REVIEW_4_FINAL.md:195-226`) | High | Partially; the teacher's comfort label later proved uninformative on the student's own proposals (`GOALS_AND_CLAIMS.md:15834`) |
| Label / teacher | Teacher's DAC flags 0/1,024 proposals where NAVSIM fails 27–33 % | Vendored NAVSIM, 25,600/25,600 agreement (`GOALS_AND_CLAIMS.md:15833`) | High | Replaced in the labellers (not in repo, see §6) |
| Label / teacher | Devkit filter dropped 25.4 % of frames (biased to short logs); a scorer "abort" dropped 44 % | Split counters; correlation test (`ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md:200-228`) | High | Fixed |
| Units / rates | Radians averaged with metres in WTA; derived speed 2× (0.1 vs 0.2 s); two time bases; yaw seed forced to 0 | Constructed counterexample; identity vs the log's recorded speed (`PAPER_CONFORMANCE_REVIEW.md:325-347`; `FIX_VERIFICATION_REVIEW.md:85`; `REVIEW_3_FULL.md:324-361`) | High | Fixed; the dt guard was first algebraically blind (F3b) |
| Estimator / selection | Selection = sum of raw logits; then a v2-shaped rule scored against v1 | Primary `pdm_scorer.py` (`FIX_VERIFICATION_REVIEW.md:95`; `RETRACTION_LOG.md:17813-17833`) | High | v1 rule adopted; harm guard +0.78 [−0.30, +1.82] |
| Estimator / selection | Pooled variance guard called comfort "discriminating" | Within-frame guard (`REFE_MODEL.md:254-280`) | High | Fixed |
| Estimator / selection | SELECTION-BOUND: best-of-64 91.2 vs pick 43.8 vs random 43.2 | E-6: every proposal scored by the unchanged harness (`eval/RESULT_E6_sub200_ep011.md:14-24`) | Blocking for any result | Open (Amendment 4 FAILED) |
| Guards that could not go red | 9 inert arms (R3) and 13 (R4): hardcoded `diag_rope`, stubbed `_hold`, prefix-covered loader, LoRA scale 0 undetected, config-derived inputs | "Construct the regression; does it go red?" (`REVIEW_3_FULL.md:517-547`; `REVIEW_4_FINAL.md:415-449`; `REVIEW_5_CAMERAS.md:482-501`) | Class | Several fixed; `diag_consumer_conformance.py` written |
| Pipeline / resource | No checkpoint/resume; per-image recursive glob; trainer blind to the augmentation file; `--epochs` × accum 64× overrun; score-loss normaliser off (0.75) | Writing and rehearsing the pod runbook (`READINESS.md:41-47,105-106`); running `--accum` (`REVIEW_6_FINAL.md:143-169`) | Blocking | Fixed; B2/B3 fixed in code (`refe/train.py:908-923,1136-1146`), but the post-fix equivalence run is not banked (only the pre-fix `raw/2026-09-21-review6/accum_equivalence.txt`) |
| Pipeline / resource | 1-camera cost carried to a 4-camera model; paging read as compute; sparse TF32 column; unmeasured A40 multiplier | Residency guard; recomputed arithmetic (`REVIEW_5_CAMERAS.md:44-62,315-355`; `REVIEW_6_FINAL.md:316-370`) | High (provisioning) | Retracted and re-derived |
| Pipeline / resource | Uploaders and assembler died silently; eval venv under %TEMP% deleted | Counting rows on both sides (`READINESS.md:112`) | High | Re-uploaded / rebuilt |

Conflict (not resolved in the package): `REVIEW_4_FINAL.md:479-487` calls the teacher rig "not reproducible across launches" from `m-nr-n` vs `m-nr-r1`; but `m-nr-r1` is the rank-1 route run (`PAPER_CONFORMANCE_REVIEW.md:169`), the rank is an environment variable invisible to `overrides.yaml` (`code/run_stage1_rollouts.sh:12`), and the true replicate reads exactly 0 (`raw/stage1_rank0_replicate.txt`; `FINDING_TEACHER_STALL.md:725-742`).

## 2. The review method, and a reusable pre-launch checklist

What each round added: R1 a row-by-row spec table (MATCHES / DIFFERS-DELIBERATELY / DIFFERS-UNINTENDED / NOT-IMPLEMENTED), each row MEASURED by running code, with md5 read stamps. R2 "a fix passing its own test is not evidence": independent probes plus mutation arms. R3 the paper read primary (PDF text, Table A12) on a frozen package, and an audit of which controls can go red. R4 a constructed regression per instrument, a fix-introduces-new-instance hunt, an index/commit-hazard audit. R5 scope shifted to consumers of the config. R6 ran every instrument, recorded exit codes, recomputed cost arithmetic from raw inputs. Readiness loop: rehearsed both pod scripts end to end on a fake `/workspace` (R19–R23). Eval: an early checkpoint on day 1 (`eval/RESULT_E0_E2.md:13-30`); every candidate scored (E-6).

Checklist for a model line about to train:
1. Read the reference's hyper-parameter and architecture tables primary, row by row.
2. Freeze the code under review and stamp md5s.
3. For each pretrained component: which line loads it, and what fails if that line is deleted? Measure feature displacement against the published model, with SELF and JITTER controls.
4. Hook every cross-attention; print context shapes and storage pointers.
5. Check one identity per derived quantity against an independent reference (the log's own speed, an analytic arc).
6. For every metric or label term: what does its producer write? Is it ordinal? Does it carry state? Does it vary within the candidate set?
7. For every guard: construct the historical regression and require RED; make expectations literals or read them from a different consumer.
8. For every filter you did not write: count what it removes and test whether the removal correlates with anything.
9. Rehearse the real launch scripts end to end; count rows at every handoff.
10. Evaluate the first checkpoint through the real eval path, with the do-nothing floor and an oracle/random/pick triple.

## 3. REFe's trajectory as a lever study

| Lever | Measured effect | Tier | Verdict |
|---|---|---|---|
| Goal-augmented twins (TAU 0.3, 71 % yield) | Spread 0.122 → 4.34 m (epochs 3→5); oracle ADE 0.359 vs random 2.244 m at epoch 8; best-of-64 81.6 → 91.2 (`GOALS_AND_CLAIMS.md:15831-15832`) | NAVSIM v1 | Positive on proposals, confounded with training progress |
| Fixed-candidate scorer supervision | Skill 0.460 / 0.162 / 0.013 at epochs 5 / 8 / 11 (`MODEL_REGISTRY.md:5247`) | NAVSIM v1 | Negative (the scorer got worse as it trained) |
| On-policy scorer + NAVSIM DAC label | Epoch 13: skill 0.073 [−0.047, 0.199] FAILURE; DAC within-scene AUC 0.53 → 0.65; comfort 0.39; pick vs STOP −13.18 [−18.87, −7.14] (`GOALS_AND_CLAIMS.md:15833`) | NAVSIM v1 | Component learned; selection null; under-dosed |
| NAVSIM comfort label (v3) | Teacher's label 0.375 on 99.8 %; 41.7 % agreement (`:15834`) | — | Deployed; effect not measured at the ref |
| Selection rule v2 → v1 | +0.78 [−0.30, +1.82] on 923 disjoint tokens; epoch 5 re-selected 53.5 → 35.8 (`MODEL_REGISTRY.md:5250`) | NAVSIM v1 | Null; exposed the false peak |
| Sim rate 20 → 10 Hz | 1.64× cheaper; targets moved 0.23 m ADE; teacher PDM-like 0.9727 → 0.9752 (`READINESS.md:49`) | Our calculators | Cost lever |
| Exact data-prep levers, bf16 + compile | 35 → ~130–232 rows/min; 0.2236 s/sample (`MODEL_REGISTRY.md:5257`) | — | Throughput |
| Teacher test-time search (N 1 → 64) | 97.19 → 97.03 non-reactive; −0.227 reactive; the selected plan brakes more than the candidate pool (−2.37 vs −0.48) (`FINDING_TEACHER_STALL.md:683-718`) | nuPlan CL, 8 scenarios | Negative: an argmax over a larger pool harvests the critic's bias |

How to get a positive result fast: decompose before building (one E-6 table, 0 extra training, located ~47 PDMS of headroom in selection, not in the proposals); supervise the selector on the distribution it faces, with labels computed by the benchmark's own code (verified 100 %); always print the do-nothing floor and the pick's behaviour (on PDMS-like scores, bias toward stopping reads as a gain); diversity came from data (twins), not from heads.

## 4. Transfer map

| # | REFe lesson | v7F | refav1 | Cheapest check |
|---|---|---|---|---|
| 1 | Trunk normalisation (R3-1) | YES, FAILS at source. `v6.py:5533` feeds `_contract.py:51-56`'s [0,1] frames; the converter copies the patch conv as identity (`dinov3_seed_checkpoint.py:344-349`); normalisation absent from `DECLARED_LOSSES` (`:568-578`) and the seed stamp; ImageNet stats only in O7's teacher (`train_v6_staged.py:1007-1009`), which v7F sets to 0 | NO. HF mean/std (`dinov3_fp8_encode_ship.py:98-104,123-124`); RGB (`v2_compressed.py:90,143`) | CPU probe: seeded ViTEncoder vs `DINOv3ViTModel` on one real frame, rel L2/cos with SELF/JITTER. Fix: fold mean/std into `patch.weight/bias` (exact affine fold); input stays identical across arms (`PREREG_V7F.md:138-150`) |
| 2 | Positional scheme with no checkpoint counterpart | YES, declared. `pos` trunc_normal(0.02) at init (`encoder.py:125-126`; stamp `left_at_init_keys ['pos']`) | NO (HF RoPE) | Same probe with a RoPE-off arm; lever: port `refe/model.py:214,262` (bit-identical RoPE) |
| 3 | Pretrained special tokens dropped | YES, declared (`dinov3_seed_checkpoint.py:41-44`) | NO (sliced after the forward, `:125`) | Mask the registers in the HF model and measure rel L2 |
| 4 | Anchor / "frozen copy" semantics | Anchor = deepcopy of the seed (`train_v6_staged.py:8599-8612`), so it anchors to the non-DINOv3 function (HYPOTHESIS: this resists the fixes) | n/a | Measure anchor vs published DINOv3 at step 0 |
| 5 | Declared-but-not-built levers / disabling defaults | YES: LDAD absent (0 files in `stack/`, control 3); `--horizons` default `[1,2,4]` (`:9266`) still missing from the §9 launch line | Levers default 0.0 (`refa_v1_train.py:361-397,463`) | G-DVB/G-LIVE; `launch_gate.py` absent at the ref |
| 6 | Within-candidate inertness | n/a at S-W | YES: `cos` goal term range 1.79e-07 vs 2e-03 penalty; HOLD stratum ptp 0.0 on 133/282 (`MODEL_REGISTRY.md:2095-2096`) | Per-term within-window ptp census |
| 7 | Pooled estimator for a within-scene choice | G-ACT style gates (UNVERIFIED) | YES: `cost_fidelity` pooled (`refa_v1_plan.py:376-411`), no caller outside tests (two probes) | Within-window Spearman plus oracle/random/pick (`eval/selection_readout.py` shape) |
| 8 | Selection-bound decomposition | Later (G-DRIVE) | YES: baseline on 282/282 (`MODEL_REGISTRY.md:2129-2137`); better plans exist (interior W_KAPPA optimum 0.8934 vs 0.9251; turns −0.2326 m, `:2096`) | Score the iCEM population with the unchanged taniteval harness |
| 9 | Label must equal the benchmark metric | UNVERIFIED | Cost terms are proxies for the four families | Agreement % between cost-term ranking and family metrics |
| 10 | Statistic read from a label, not content | `step_readout_op` random-init hazard (`PREREG_V7F.md:391-400`) | YES: `baseline_won_frac` from the label (`taniteval/tools/refav1_arm.py:2521`), 0.6631 vs true 1.0 | Compute from bit-identity |
| 11 | Do-nothing floor | YES: no arm beats hold (`V7_LAUNCH_GATE.md:19-35`) | YES: `cl` worse than `ha0` +0.0158 (`MODEL_REGISTRY.md:2094`) | Report the fraction bit-identical to the floor |
| 12 | Argmax harvests bias | Later | YES in shape (selections collapse onto baselines) | `code/switch_direction.py` pool-vs-selection |
| 13 | Replicate identity includes env vars | Training variance (CLAUDE.md `H-ESTIM-SEED-1`) | iCEM inference floor ≈0.30 m | argv and env audit |
| 14 | Units / rate identity | `--o5-k 60` = 6 s assumes 0.1 s (UNVERIFIED) | Steer ≈ 2.9κ (`MODEL_REGISTRY.md:2168-2172`); grid refusal (`refav1_loader.py:190-192`) | Derived-vs-logged identity |
| 15 | Operating-point statistics from a small sample | n/a | NEW: standardiser fit on `max(bs, 8)` windows (`refa_v1_train.py:581-582`) vs "over the train corpus" (`refa_v1.py:1112-1114`) | Checkpoint buffers vs stats over ≥200 episodes |
| 16 | Primary recipe table | Trunk LR ×0.1 / warmup 2000 are prereg choices | Constant LR, no scheduler (`refa_v1_train.py:347-351,549-551`) | One primary-table read (UNVERIFIED for DINO-WM) |
| 17 | Handoff integrity | Prereg line once froze the trunk (`S-S`, `PREREG_V7F.md:308`) | Done-marker trap: 80 relaunches (`MODEL_REGISTRY.md:2114-2123`) | Rows on both sides; consumer run on the producer's output |

## 5. Directly reusable REFe assets
All under `…/2026-09-20-refe-plan/`, verified present in the snapshot: `refe/diag_rope.py` (displacement probe with verdict token) and the banked `Library/refs/dinov3_rope_position_encoding.py`; `refe/model.py:214,262` (RoPE) and `refe/load_dinov3.py` (strict loader); `refe/diag_consumer_conformance.py`, `refe/diag_train_resume.py`, `refe/diag_amp.py`; `eval/proposal_table.py`, `eval/selection_readout.py`, `eval/selftest_selection_readout.py` (the decision rule's mutation self-test); `code/switch_direction.py`, `code/stall_census.py`, `raw/2026-09-21-camera-scaling/cam_scaling.py` (residency guard). Do not reuse `eval/write_result_e6.py:101-105` as is: it hardcodes "indistinguishable from random" and "rank at chance", which the ep005 file states against skill 0.460 [0.376, 0.537] (`RESULT_E6_sub200_ep005.md:24,66`). `refe/diag_guard_audit.py` and `refe/navsim_dac.py` are absent at the ref.

## 6. Escalations and conflicts
- Stranded at the ref (checked with `git cat-file` on `b3f7ea6f`): `eval/RESULT_E6_sub200_ep012.md`, `RESULT_A4_sub200_ep013.md`, `RESULT_A5_a5confirm_ep013.md`, `raw/a5_confirm`; `refe/navsim_dac.py`, `refe/navsim_pdm/`, `validate_navsim_*.py`, `rule_*.py`, `code/switch_onpolicy.sh`, `code/pod_train_v2.sh`, and three `raw/2026-09-26-*` dirs. `refe/train.py` at the ref has 0 `--scorer-mode` hits (control `ScorerBank` 3), and `refe/onpolicy_label.py` has 0 NAVSIM hits (control 2). The registry (`MODEL_REGISTRY.md:5247-5250`) cites all of these.
- Launch gate missing: CLAUDE.md `:1286-1310` requires `stack/scripts/launch_gate.py`; absent (two probes). Only refcv6's `prelaunch_gate.py` exists.
- Stale steering entries: `BACKLOG.md:166` (R24) is resolved per `PI_DECISION_QUEUE.md:1273-1317`; `BACKLOG.md:35` (A16) is still open (this report is a read-only first pass, not a closure); the registry's `refa_v1_plan.py:277-281` citation (`MODEL_REGISTRY.md:2156`) now points at the iCEM loop, the line has drifted.
- PI decisions needed: defaulting any refav1 cost lever (`ccosh`, `kamm_mu`, W_KAPPA) — registry: "a PI / Master Mind decision"; v7F's slot.

## Deliverable manifest (as delivered)
| Artifact | Where | Only copy? |
|---|---|---|
| This report | Returned to the caller (message only) | Yes, by design: the caller banks it (now banked here) |

## Could not read or verify (as delivered)
The snapshot is a partial extraction (40 of 251 A&I research dirs at the ref; `2026-09-18-dinov3-b16-seed/raw/dinov3_vitb16_seed.json` read via `git show b3f7ea6f:`); Amendment 4/5 and epoch-12 raw results quoted from the registry only (absent at the ref); every magnitude for v7F's three trunk-input defects on ViT-B UNMEASURED (REFe's figures are ViT-S); the DINO-WM training schedule; v7F's frame interval behind `--o5-k 60`; B3's post-fix proof; the effect of the comfort v3 label.
