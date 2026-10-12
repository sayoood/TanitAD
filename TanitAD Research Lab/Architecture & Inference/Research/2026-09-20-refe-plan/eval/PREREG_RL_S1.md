# PREREG RL-S1 (DRAFT, NOT BINDING): scorer-only exhaustive policy gradient on a frozen REFe fan

> Status: **DRAFT.** It becomes binding only when it is landed in git before the first training step, with (a) the base-checkpoint question below
> answered, (b) the code of "What has to exist" built, (c) a launch-gate PASS token for the exact argv (`refe_rls1` profile, CLAUDE.md 2026-09-26).
> Source: `Research/2026-10-11-refe-data-and-rl/RL_POSTTRAIN_REVIEW.md` sections 3, 5.2, 6 (transcribed, not redesigned). Every [P]/[M] number below is the
> review's; this file re-verifies none of them. Evidence classes: [M] ours + artifact, [P] published, [I] inherited, [E] estimated, [H] hypothesis, [U] unverified.
> Order on the A40 (PI 2026-10-12): **P2 -> RL-S1 -> P6.**

## OPEN items (for the Master Mind / PI; this draft chooses neither)

| id | question | state |
|---|---|---|
| ~~OPEN-BASE~~ **DECIDED: A** (Master Mind 2026-10-12, section below) | **Which checkpoint is the base.** Variant **A**: SFT-4 arm B u400 (`model_sft4_B_u400.pt`, md5 `8e382eeb...`, full-navtest 88.25 [M, `eval/RESULT_SFT4.md`]) on the frozen `model_final.pt` fan, as the review wrote it. Variant **B**: P2's `model_final.pt`, *if* P2 clears its GO rule (`eval/PREREG_P2_AMENDMENT_1.md` section 4: lo > 0 AND no adverse sub-score vs u400). The two are written out in full below; do not read either as chosen. | **OPEN** |
| ~~D3~~ | Lane-change arm R2. | **DECIDED, PI 2026-10-12 ("yes include the lane change arm"; Master Mind, landed 5e4a998): R2 is IN.** One seed. Read on the TACTICAL family (lane-change precision / recall, lateral-decision kappa). Kill if the lane-change rate exceeds 2x the label base rate (K5). **No PDMS credit is expected** (review 4.2: label active on ~2.7 % of frames). |

### The two base variants, side by side (the differences are mechanical and each changes a number in this document)

| | **A: SFT-4 u400 on the frozen fan** | **B: P2 model_final** |
|---|---|---|
| scorer outputs | **7** (6 PDM components + teacher-lane) [M, `ckpt_io.config_for_checkpoint`, `planner.py` rule `navsim_v1_lane`] | **6** (PREREG_P2: "No lane head in P2") |
| selection rule that the policy `pi` MUST mirror | `navsim_v1_lane` = NC*DAC*(5EP+5TTC+2C)/12 * p_lane [M, `planner.py:552-557`; RESULT_SFT4 line 23] | `navsim_v1` = NC*DAC*(5EP+5TTC+2C)/12 [M, `planner.py:558-561`] |
| the fan | `model_final.pt`'s 64 proposals; generator, trunk, LoRA frozen, so the fan is bit-identical to the one SFT-4 trained on | **P2 changed the proposals** (joint fine-tune, scorer sees the trunk): the 64 proposals must be **re-dumped and re-labelled** (dump 1.0-3.1 h on the A40 + 25-85 CPU-h of labels [E]) |
| "scenes the base scorer did not train on" | the SFT-4 `data` event list (exact list [U], to be frozen) | P2 trained on **174,912 served sets**; the not-trained-on partition must be recomputed from P2's own bank. **If P2 saw essentially all navtrain sets, variant B has no clean RL-train pool** [**MEASURED 2026-10-12: it saw 99,495 of 101,583 eligible tokens; 2,088 tokens (4.1 % of 51,000) are unseen**; see "OPEN-BASE pool measurement" below], and that alone may decide OPEN-BASE |
| A0 drift check | expected 88.25 | P2's own full-navtest read (the R2 read of Amendment 1) |
| extra work before launch | `scorer_finetune.py` must LOAD a 7-output checkpoint (see "What has to exist", item 3) | none for loading; everything else as A |

### OPEN-BASE pool measurement (2026-10-12; `raw/rl_pool/rl_pool_by_base.json`, instrument `eval/rl_pool_measure.py`)

Eligible population [M]: `splits/navtrain.yaml` 103,288 tokens / 1,192 logs, **0** tokens and 0 logs in navtest, minus the 24 SFT held-out logs (1,705 tokens whose log is known) = **101,583 tokens**. The yaml pairs no token with a log; the token-to-log table (103,037 tokens) comes from the main run's target bank, and 251 yaml tokens have no recoverable log.

* **B (P2): no pool [M].** P2 trains 3 epochs on all 174,912 served sets = 99,495 tokens / 1,167 logs (the banked bank reproduces the carve exactly: 177,836 served = 174,912 + 2,924). Unseen: **2,088 tokens** = the tokens with no on-policy set at all, over >= 632 logs, each of which also holds trained tokens. That is 4.1 % of the planned 51,000, and below the 3,000 scenes RL-val alone needs.
* **A (u400): a pool in count, not in membership [M bounds, E expectation].** u400 is update 400 x 64 = 25,600 sets, so it touched at most 25,600 of the 99,495 trained tokens: **>= 73,895 are certainly unseen (75,983 with the 2,088 set-less tokens), about 77.5k expected**. The draw itself is not recoverable (seed 0 over the launch-time ordered index; the SFT-4 `data` event and `sft.log` were never banked and `train_pdm` died with the pod). The review's "28,727 labels not written at arming" are SETS, not scenes: in relabel order they are mostly second (rank-1 augmented) views, so only about **3,027 tokens** [E] have no labelled set at all. Any RL-train / RL-val partition of A is therefore a random partition, with about 24 % of its tokens expected to overlap u400's draw.
* **Neither base is clean at the scorer-lineage level [M].** Both start from `model_final`'s scorer, trained on navtrain in the main run: fixed bank in epochs 1-11 (targets for 189,858 of 189,888 keys) and on-policy sets in epochs 12-27 (146,788 sets at the last reload). My per-epoch reconstruction from each set's own label timestamp matches the trainer's own counts within 0.1 % (`rl_pool_by_base.json`, `main_run_scorer_exposure_L0`). At most **8,531 tokens** escaped the on-policy phase, before the fixed-bank phase is subtracted.
* **Almost every RL scene needs a fresh frozen-fan dump [M].** Only 22,742 served training sets carry proposals from the step-10,075 checkpoint, which IS `model_final.pt` (all 276 trainable tensors bit-identical, equal frozen-trunk fingerprints: `raw/rl_pool/ckpt_last_vs_model_final.json`); 13,713 of them are plain rank-0 views (the others are route/goal-augmented rank-1 views). The other 152,170 come from older generator checkpoints, and no PDM label survives off the pods, so even the 13,713 need labels. The 2,126 tokens with no set at all (2,088 outside the held-out logs) need a dump in either variant. Fitness of the dumper: `HANDOFF_RL_S1.md`, 2026-10-12 RL-S1-S1 section.
* **Reading, for the Master Mind:** if pool size decides OPEN-BASE it decides against B. A's pool is large in count but cannot be named; the cost of that is the ~24 % overlap above, which both arms of a C-vs-R pair share equally (ESTIMATED argument, not measured).

### OPEN-BASE DECIDED: base = variant A (Master Mind, 2026-10-12, before any RL data exists)

**RL-S1 starts from SFT-4 arm B u400 (`model_sft4_B_u400.pt`) on the frozen `model_final.pt` fan.** Three reasons, all measured above:
1. **B fails this pre-registration's own pool requirement:** 2,088 tokens unseen by P2's scorer (1,837 placeable) against 51,000 planned and
   3,000 needed for RL-val alone; `refe/rl_partition.py` REFUSES variant B (`raw/rl_pool/partition_runs/B_p2_refused.txt`).
2. **A's fan already exists in part:** `ckpt_last@10075` is bit-identical to `model_final.pt` (276/276 tensors), so 22,742 served sets ARE
   the frozen fan, and the PDM relabel P2 is running now labels them.
3. **A does not wait on P2's read.** P2's R2 read cannot reopen this: a P2 that beats u400 still has no pool. Order on the A40 is unchanged
   (P2 -> RL-S1 -> P6). The PI may overrule; the choice is reversible until the launch token is issued.

**Partition frozen (pre-data):** tool `refe/rl_partition.py` (`rl_partition/1`), seed **20261012**, variant-A run
`raw/rl_pool/partition_runs/A_no_trained_list_seed20261012.manifest.json` (partition_id `272cfd35...`): 48,000 RL-train / 3,000 RL-val
scenes over 576 / 75 logs, log-disjoint, navtest and the 24 SFT held-out logs excluded. The launch quotes this manifest's sha256. One
change remains admissible before any RL number exists, and only one: re-running the SAME tool and seed with an availability filter (tokens
whose images are on the pod), if the dumper cannot reach some keys. It must be logged here with the old and new partition_id.
**Stated cost:** about 24 % of these scenes were in u400's SFT-4 draw (ESTIMATED; the draw cannot be named). Every C and R arm shares it
equally, and the decisive read is full navtest, which no arm trained on.

## 1. Question and lever

Does a scorer-only policy-gradient step on the frozen 64-hypothesis fan, with an **exact** expected reward over all 64 proposals (no sampling), beat the
SFT-continued control on full-navtest PDMS? Prior [E, review 0.1]: **+0.2 to +1.2 PDMS**, i.e. 3-19 % of the 6.30-point gap. P2 / P6 remain the main path.

**Lever (one thing changes):** arm B's loss gains `lam_pg(t) * [ -J + beta * KL(pi_SFT || pi) ]`, with `pi_i = softmax_i(log(A_i + eps) / T)`, `J = sum_i pi_i R_i`,
`T = 0.1`, `beta = 0.1`, `lam_pg` ramped 0 -> 1 over 200 updates, grad-clip 1.0 [H]. BCE on the six sub-scores is unchanged.

> **CORRECTION to review 5.2, found while reading source (not a redesign).** The review writes the selection score as `A_i = NC*DAC*DDC*(5EP+5TTC+4C)/14`
> citing `train.py:611-614`. That is the **`v2_shape` rule**. The rule the 88.25 checkpoint actually deploys is **`navsim_v1_lane`**
> (`planner.py:552-557`; `RESULT_SFT4.md` line 23), and P2 will deploy `navsim_v1`. A policy gradient on the wrong `A_i` optimises a ranking that is never used at
> inference. This draft therefore requires `A_i` to equal **the planner's own rule for the base** (`A_i` = `scorer_finetune.sel_score` for the 7-output base).
> The registry-vs-source conflict is reported, not resolved here.

**Trainable set:** `score_q_mlp`, `score_dec`, `score_head` only; context detached; the 64 proposals are bit-identical (SFT-4 recipe, `PREREG_SFT4.md` l.32).
**Recipe:** SFT-4's optimiser (AdamW, lr 1e-4, 50 warm-up, cosine, wd 0.01, batch 8 x accum 8, bf16), 1,000 updates, final update is the arm's checkpoint.

## 2. Reward (review 3; implemented in `refe/rl_reward.py`, 28 checks + 26 RED mutations in `refe/test_rl_reward.py`)

`R_i = PDMS_i - 0.15 D_i + 0.5 A_prog_i  [- 0.05 K_i + 0.05 L_i  only in R2]`.

| term | in arms | note |
|---|---|---|
| `PDMS_i` | all | recomputed from the labelled sub-scores, NC*DAC*(5EP+5TTC+2C)/12 |
| `D_i` drivable-area excess | R0 R1 R2 | mean over the 41 executed states of min(gap/0.5 m, 1); needs the part-2 labeller |
| `A_prog_i` (g2) | R0 R1 R2 | clip(ep_i - best-safe ep, -1, 0), safe = NC = DAC = TTC = 1 |
| `K_i` lane consistency | **R2 only** | outside intersections; deadband 0.25 m; suppressed where o >= 1.5 m for the heading part |
| `L_i` lane change | **R2 only** | +progress_gain only for a SAFE executed change under a firing human label; -1 for an unlabelled change; minus a dither flag |

Two interpretations the review left open are flagged in `rl_reward.py` ("safe" for g2; K's straddle term read literally). **The guard finding is RULED (Master Mind,
2026-10-12):** uncapped, `w_prog = 0.5` put a TTC-violating full-speed plan (PDMS 0.583) above a safe half-speed plan (0.792 - 0.25 = 0.542), i.e. it rewarded
crossing the safety boundary, which review principle P4 forbids. No fixed `w_prog` can guarantee P4, so the fix is STRUCTURAL, not a smaller weight:
`rl_reward.guard_penalty` caps each safe proposal's penalty at half its pre-guard margin over the best UNSAFE proposal below it. The guard still reorders
proposals inside the safe set (progress vs comfort, the slow-pick hack it targets); it can never push a safe plan to or below an unsafe one it beat.
`w_prog` stays 0.5 [H]. Pinned: `guard_never_crosses_safety_boundary` (16.5/24 = 0.6875 > 7/12), `guard_reorders_inside_safe`, the P4 property over 300 random
fans (> 1,000 safe-over-unsafe pairs, 0 violations), and mutations `P4_cap_removed` / `P4_cap_keep_1_ties` / `P4_cap_counts_safe_rivals`, all RED
(`test_rl_reward.py` -> `ZZTEST_PASS 31`).

## 3. Arms

Base per OPEN-BASE. "A/B" below are the two arms of one `scorer_finetune.py` pass (one trunk forward, **paired by construction**): arm A = BCE only, arm B = BCE + EPO.
Whether each run yields `(C_s, R_s)` from one pass (3 runs: seeds 0, 1, and R2) or five separate runs as the review's 6.2 budgets is **OPEN for the implementer**; the
paired pass is cheaper (about 12 h vs 20 h of A40 [E, from 14.4 s/update [I]]) and gives the within-seed pairing for free.

| arm | what changes | seeds |
|---|---|---|
| **A0** anchor | base, not trained further; re-read on the same harness (drift check) | - |
| **C0 / C1** | SFT-continued control: BCE only, identical sets / order / steps | 0, 1 |
| **R0 / R1** | RL-core: BCE + EPO, reward `PDMS - 0.15 D + g2` | 0, 1 |
| **R2** | R0 + `K` + `L` (**decided IN**, D3) | 0 (single seed) |
| **I-rep** | inference replicate of the C0 and R0 checkpoints (eval batch 8 vs 32, shuffled order, non-deterministic kernels) | - |

Data: RL-train 48,000 scenes the base scorer did not train on, log-disjoint from RL-val 3,000 scenes (>= 60 logs); labels computed once and shared by all arms; the
RL-val read is a table lookup (the true PDMS of a scorer's pick). Navtrain partition, not a re-selection of any programme corpus (parity rule not involved).

## 4. Endpoints and the bar (as the review 6.3; nothing here is new)

* **Primary:** full-navtest PDMS, 12,146 tokens, 136 logs, NAVSIM v1.1, paired log-cluster bootstrap (10,000, seed 20260927). Answers *another draw of logs*, **not**
  another training run or inference run. Tier: pseudo-simulation of a 4 s open-loop plan; not closed loop.
* **ADOPT-FOR-NEXT-STAGE** iff all of: (1) `R0 - C0` and `R1 - C1` each >= +0.25 and their mean >= +0.5; (2) pooled paired CI lower bound > 0; (3) mean contrast > 2 x
  max(|R0 - R1|, |C0 - C1|, inference-replicate |delta|) (crude with two pairs: necessary, not sufficient); (4) no longitudinal / lateral families6 component separates
  adversely and the g3-g5 telemetry bounds hold; (5) every sub-score paired delta reported, EP included.
* **REFUTED** iff the pooled upper bound < +0.25. **Otherwise NOT PROVEN** (a likely outcome at +0.2 to +1.2 [P]; a registered outcome).
* **R2 is read on the TACTICAL family, not PDMS:** lane-change precision / recall against the labels, lateral-decision kappa; **no PDMS credit expected** (review 4.2).
  R2 - R0 is n = 1 pair, read against the R0/R1 floor: necessary, not sufficient.
* **Four-family guards:** LONGITUDINAL and LATERAL by the adverse-separation rule; TACTICAL decision kappa may not fall by > 0.03 vs C0; **STRATEGIC: PAUSED** (PI ruling
  2026-09-19, item 25), reported as paused, not as a pass, not omitted.
* **Checkpoint:** the final update of each arm; no navtest early stopping; hyper-parameter and early-stop decisions use RL-val lookups only.

## 5. Kill conditions (review 6.4; K5 amended for D3)

| code | condition | action |
|---|---|---|
| K0 | no launch-gate PASS token; a new flag without a G-DVB entry | do not start |
| K1 | `lam_pg = 0` run does not reproduce C0 at 50 updates (bf16 tolerance); the EPO gradient on the scorer reads 0 (control must read exactly 0.0 with the term off); the mutation arm stays green | gate fails, no GPU-day spent |
| K2 | grad norm > 20 x the control's median, or NaN | stop the arm, hand off |
| K3 | any g3 / g4 / g5 telemetry bound breached at two consecutive 250-update checks on RL-val | stop the arm |
| K4 | RL-val true PDMS of R below C by > 0.5 at two consecutive checks | stop the arm |
| K5 | **R2's lane-change rate above 2 x the label base rate** (decided D3: R2 in; this is its kill) | stop R2 |
| K6 | any arm exceeds 6 h on the A40 | stop, HANDOFF, fresh agent |

## 6. Outcomes committed in advance, and the next lever (RULE ZERO)

* **ADOPT:** add the EPO term to P6's scorer loss (labels are computed every step anyway); then the CLOVER-style alternating generator stage at P6 scale.
* **NOT PROVEN with EP flat:** the decision-focused hypothesis is unsupported; next lever = calibration of the TTC / NC / DAC heads (REVIEW_7 l.209-218), which the BCE-only
  control already tests. Re-read C0 against A0 before concluding anything about RL.
* **REFUTED:** remaining levers with blockers named: (a) P6 (blocked on P2's verdict, a pod with >= 24 vCPU, the PI's P6-T/C/N/Y decisions); (b) generator-side alternation
  (blocked on a stochastic-policy design and A100-class compute); (c) tracker-aware targets (P5), a zero-training EP lever.

## 7. What has to exist before launch (state updated 2026-10-12 by the Master Mind)

| item | state |
|---|---|
| `refe/rl_reward.py` + `test_rl_reward.py` | **BUILT** (9c75945; P4 guard cap ruled the same day): `ZZTEST_PASS 31`, 29 RED mutations |
| EPO term in `scorer_finetune.py`, default OFF, bit-identical identity test, non-zero-gradient + exact-0.0 control, mutations | **BUILT** (ac4d3b8): `test_rl_epo.py` `ZZTEST_PASS 42`, 41 RED; preflight G11; OFF path bit-identical to blob 92b52dac |
| `scorer_finetune.py` can load a **7-output** checkpoint (it calls `REFeConfig.for_backbone` without `config_for_checkpoint`, and `--lane-head` assumes a 6-row head) | **BUILT** (ac4d3b8): cfg sized from the checkpoint, a 7-row head left untouched; stubbed in tests, so the first A40 preflight is the final check of the u400 strict load |
| `refe_rls1` launch-gate profile (G-DVB entry per new flag, R0/R1 and R2 argvs accepted) | **BUILT** (bd45e96): `test_launch_gate.py --static` 316 / `--no-banked` 302; the FULL suite re-runs once the dev box has memory (its e2e G-CKPT arm OOM'd on 2026-10-12, environmental) and G-SUITE re-checks it on the pod |
| on-policy proposal dumper (~51,000 scenes), extended labeller (`D`, `K`, `L`), RL-train / RL-val log-disjoint partition, pod chain script | labeller **BUILT** (eceb88f; G-REWARD-ID 0 disagreements on 91,520 proposals); partition **BUILT + FROZEN** (b0ea104, seed 20261012); dumper + `--dump-scores` **BUILT** (bd45e96; same-GPU identity with the old dumper EXACT, 500/500; batch invariance FAILED its bar and is inherited from the old dumper bit for bit; cross-GPU max abs delta 0.25: `raw/rl_fan/RESULT_IDENTITY.md`, MEASURED 2026-10-12); pod chain **IN BUILD** (2026-10-12) |


## 8. Partition availability check -- 2026-10-12 (RL-S1 priority 2; appended, nothing above is edited)

**NO availability filter was applied. The partition is unchanged: `partition_id` stays `272cfd35...`** (48,000 RL-train / 3,000 RL-val, seed 20261012, manifest
`raw/rl_pool/partition_runs/A_no_trained_list_seed20261012.manifest.json`; the two scene TSVs are now banked beside it, sha256 equal to the manifest's).

* MEASURED (banked artifacts, `raw/rl_fan/probe_image_coverage.py` -> `raw/rl_fan/image_coverage.json`, `ZZCOVERAGE_COMPLETE`): on the ORIGINAL A40 pod every one of the 51,000 scenes
  resolves all four camera frames. Two routes that share no assumption: the main run's `train.log` (29 grow-epoch load lines, every one "N/N tuples resolve ALL cameras",
  last 189,888/189,888, 413,152 frames indexed) and arithmetic (103,288 yaml tokens x 4 = 413,152; rank-0 bank names 412,148 distinct frames; the 1,004 = 251 x 4 difference
  is exactly the 251 tokens with no bank row; all 51,000 partition scenes ARE bank rows with agreeing logs).
* UNVERIFIED -- INHERITED: the P2 pod (where RL-S1 runs) builds `navtrain_pixels` from the same 32 public shards (`code/pod_dataprep.sh`), but nobody has measured it. The measurement
  is `refe/rl_fan_dump.py --plan-only` on that pod BEFORE the first dump (exit 0 = complete; exit 3 + `unavailable_scenes_*.tsv` = gap).
* If a gap is found, the ONLY admissible remedy is `refe/rl_pool_available.py` (filters the POOL token table; `rl_partition.py` is not edited, same tool, same seed, same
  parameters), then this section gets the OLD and NEW partition_id BEFORE any RL number exists. Property-tested on the real `partition()`: removing a scene the partition did
  not use leaves train and val identical; removing a train scene replaces exactly it, by the next scene in the frozen order.
* Not a pre-registration change: no lever, bar, kill rule or outcome in sections 1-7 is touched.

## 9. TL-1 re-cut -- 2026-10-12 (PI: "Paper-pure, smaller pool"; appended, nothing above is edited; written BEFORE any RL number exists)

**Reason (PI, 2026-10-12):** u400 is a 7-output scorer, so every arm needs teacher-lane labels for the RL scenes. They are the DriveRL teacher simulator's own signals (paper-pure, the SFT-4 recipe), which cost
about 10.2 CPU-s per set; a lean re-implementation would have been a different label. The PI chose to keep them paper-pure and shrink the pool. Section 8 allowed ONE re-cut before any RL number exists (an availability
filter); this is a second, PI-ordered one, made in the same window (no RL number, no training step, no label of any arm exists), by the SAME tool and seed, and it is logged here as the prereg requires.

| | OLD (sections 0-8) | NEW (this section) |
|---|---|---|
| tool, seed, other parameters | `refe/rl_partition.py` (`rl_partition/1`), seed **20261012**, `--n-val 3000 --min-val-logs 60`, no trained-keys list | identical |
| `--n-train` | 48,000 (576 logs) | **16,000 (166 logs)** |
| `partition_id` | `272cfd357b81db535a1214fc344f71469219c17ac3c8203698d5c2b0f1befa49` | **`9995a39aeeac3833dd3f679b75e883b056e3d4d235bb9dc986578a81a7a256fa`** |
| manifest sha256 | `6a2b5aec3ae2c2f3ea00b826f7414aad4a4617f9e9a793b16332e9ccb2b142f0` | **`f8a1640cb61df48ea86e6afdb9d7c357bd6443073b367779585f5238bb723edb`** |
| RL-train TSV sha256 | `aaa2bcd530c2339241b5b5caec08b2bbf71698606ff924301e9b750ffce426bc` | `136923539e3537aaebce9cc20a8d98ddd4df6217dd2102de60a0206d612bf0a5` |
| RL-val TSV sha256 | `39f425dc3f0866ec6e6bb4e553e85f19287e65273e261d6743dd9c2ad5b8b24c` (3,000 scenes, 75 logs) | **the same bytes** |
| files | `raw/rl_pool/partition_runs/A_no_trained_list_seed20261012{.manifest.json,/}` (kept as the record) | `raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000{.manifest.json,/}` |

MEASURED (2026-10-12, `refe/rl_partition.py` re-run on the shipped inputs): (1) the tool reproduces the frozen 48,000 / 3,000 partition **byte for byte** (both TSV sha256 equal to the old manifest's), so the inputs and the
code are the ones that made it; (2) the new RL-train list is an **exact prefix** of the old RL-train order (the first 16,000 of 48,000 rows; it ends inside its 166th log, 64 of that log's 82 scenes, as the tool's `take` rule does), pinned in
`refe/test_rls1_chain.py`; (3) **RL-val is byte-identical**, so every RL-val quantity (scenes, labels, the A0 base read, the K3/K4/K5 baselines) is unchanged. The re-cut drops 32,000 RL-train scenes and nothing else.

**Bias, stated:** the 16,000 are the scenes of the first 166 logs in the frozen `sha256(seed | log)` order, whole logs, hash order inside a log: a draw of logs that no scene property (route, speed, map, manoeuvre) selected. The
stated cost of section 0 stands and now applies to the smaller pool: about 24 % of the scenes were in u400's SFT-4 draw (ESTIMATED; the draw cannot be named), shared equally by every C and R arm. Fewer logs (166 vs 576) means a
narrower coverage of road layouts per arm; RL-val still spans 75 logs and the decisive read is full navtest, which no arm trained on.

**Considered and NOT adopted: choosing RL scenes whose queue rows or frozen-fan sets already exist.** The old queue / set files hold the 64 proposals for at most 22,742 served sets (13,713 plain rank-0 views); the simulator
cost that dominates (about 10.2 CPU-s per set) is paid per set whatever produced the proposals, no PDM label survives off the dead pod, and those files are on the dev box, not on the pod. The only saving is the dump of at most
13,713 sets: 13,713 x 0.2313 s = 0.9 h of A40 (E) out of about 8 h. A rule restricted to those keys would add a selection (rank-0 views of the first logs the old dump reached) for under an hour. No such rule is registered.

**The recipe consequence (declared, one number):** `--max-updates 1000` is the pre-registered budget and `--epochs 2` was only ever headroom (48,000 / 64 x 2 = 1,500 > 1,000). On 16,000 scenes `--epochs 2` gives
ceil(16,000 / 8 x 2 / 8) = **500 updates** (the trainer's `total = min(...)` ends silently), which would halve the budget, break the 250-update checks' schedule and the `update1000` reads. The chain therefore runs
**`--epochs 4`** (exactly 1,000 updates), and `rls1_checks.py updates-check` REFUSES any arm whose planned updates differ from `--max-updates`. Every scene is seen 4 times instead of 1.33 (BCE and EPO alike, all arms
identically). `--epochs` is not pinned by the gate. The alternative, 500 updates at `--epochs 2`, changes the ramp (200 of 500 updates), K3/K4 timing and the cost model, so it was not taken; it is the Master Mind's / PI's call.

**Unchanged:** every lever, bar, kill rule and outcome of sections 1-7. Section 3's "48,000 scenes" and the 51,000 of the OPEN-BASE tables read 16,000 / 19,000 from here on.
