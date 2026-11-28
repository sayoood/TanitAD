# PRE-REGISTRATION -- measure 5 ("teach the scorer about slow plans", label_version 4): EFFECTIVENESS on the dev box

Written 2026-09-27 (Berlin), BEFORE any fine-tune in this design has run. Both outcomes are committed below.
PI directive (2026-09-27): implement every measure, use one in the live run ONLY after its effectiveness AND validity are
proven. Validity is measured (section 1). This document is the effectiveness proof's design (sections 2-7). Nothing here
changes the live run; the live-deployment procedure (section 8) runs only after ADOPT + the PI's go.

## 1. What is already proven -- VALIDITY (eval/raw/e6_sub200_ep015/v4_label_validity*.json)

Instrument: `eval/validate_slow_labels_v4.py`, the REAL labeller CLI (`refe/onpolicy_label_v4.py`, production queue
protocol) on W3's 200 navtest tokens, snapshot 015's own proposals (+ the table's logits -> scorer_top sources),
compared slot by slot with NAVSIM's harness on the SAME trajectories (E-6 table for the 64 originals;
`eval/slow_copies.py`'s harness runs for the copies; STOP re-scored through the REFe path). EP reference on navtest = the
HUMAN logged future (declared: navtest has no DriveRL teacher plan).

| check | result |
|---|---|
| copies built by the labeller == the harness-scored copies (native 20-pose and NAVSIM 8-pose grids) | max abs diff **0.0** (200 x 64 x 2) |
| NAVSIM drivable area, production worker, 81 slots x 200 tokens | **16,200 / 16,200 agree (100.00 %)** |
| NAVSIM comfort, same slots | **16,200 / 16,200 (100.00 %)** |
| wide arm: every harness csv present (all 64 slots at 0.75, 30 slots + top-16 at 0.5, STOP) | 35,563 unique trajectories, **0 disagreements** on either; the harness itself identical on all 5,389 trajectories it scored twice |
| MUTATION: a copy slot labelled with its SOURCE's labels | **DETECTED**: 1,565 / 3,200 copy slots disagree with the harness (685 DAC, 1,354 comfort); the labeller's own self-check REFUSES 4/4 such sets; the real mutated worker's copy labels equal the emulation 64/64 |
| flag OFF == the v3 labeller | 4/4 set lines identical to the REAL `onpolicy_label.py` output (timing fields excluded) |
| originals' labels with copies in the same call == v3's | 4/4 sets: every kept slot, every dropped slot (side file), ndiff, teacher row identical |
| read back through `train.OnPolicyBank` (the live trainer's loader) | 200/200 sets, 0 incomplete, NAVSIM DAC on all, loader components == line components on every slot |
| trainer reachability (`refe/diag_slow_labels_v4_train.py`, real train.py) | v4 supersedes v3; APPEND (64+17) refused 6/6 as incomplete; an extra set REPLACES the own set; copy without NAVSIM DAC detected; learning arm: P(no collision) on NEW slow plans 0.998 (copies' own labels) vs 0.732 when copies carry their source's labels (gap +0.266, bar >= +0.20) |

Teacher components vs NAVSIM (they are a different simulator and were expected to differ), from the same run:

| slots (n) | collision agree | teacher-says-collision / NAVSIM-no | TTC agree | EP r (harness NC.DAC.DDC = 1) | DDC agree |
|---|---|---|---|---|---|
| top-8 originals (1,600) | 92.3 % | 58 vs 66 | 74.7 % | 0.43 | 87.3 % |
| other originals (11,200) | 90.3-90.9 % | 459 vs 608 | 75.9 % | 0.56-0.57 | 84.8-85.2 % |
| **0.75x copies (1,600)** | **97.5 %** | **19 vs 21** | 90.5 % | 0.54 | 92.4 % |
| **0.5x copies (1,600)** | **85.6 %** | **222 vs 9 -- ONE-DIRECTIONAL** | 94.6 % | 0.75 | 96.9 % |
| STOP (200) | 97.0 % | 0 vs 6 | 96.5 % | -- | 100 % |

⛔ **The teacher's collision label is INVALID on 0.5x copies**: it calls 222 of 1,600 half-speed copies a collision that
NAVSIM does not count (9 the other way). `NuPlanCollision` fires on any overlap while the ego moves > 0.05 m/s and has no
at-fault logic (`BEHIND_COS_THRESHOLD` defined, never used; driverl `.../collision/nuplan_collision.py:19,42`), and a
half-speed plan also asks for an abrupt slow-down that NAVSIM's LQR tracker cannot follow. Taught as is, it would say
"half speed = collision" -- the opposite of measure 5's lesson.
⭐ Next lever, run: `--nc-at-fault` (the teacher's OWN at-fault rule from `NuPlanTTC`, exported; section 1b).

**Label-ranking ceiling** (a scorer that learned the labels perfectly, ranking each token's candidates by the labels' v1
aggregate; per-token concordance with harness PDMS over pairs the harness does not tie, token mean): originals only
**0.819** (182 tokens); pairs involving 0.75x / 0.5x / STOP **0.869**; involving 0.75x / STOP only **0.906**; involving
0.5x only **0.830**. The v4 labels carry MORE ranking information about slow plans than v3's labels carry about the 64.

### 1b. The at-fault lever (`--nc-at-fault`, own flag, default OFF) -- run on all 200 tokens
`v4_label_validity_sets_ncaf.json` (same tokens, same copies; DAC/comfort again 16,200/16,200). The NC key the trainer
reads becomes the teacher's OWN at-fault event (NuPlanTTC's rule, exported); the raw value is kept beside it.
Self-test `nc_at_fault_selftest.json`: without the export it RAISES (no silent fallback); with it only the NC key
changes, the raw value is preserved, ndiff is unchanged, and the v3 path is untouched by the wrapper.

| slots | raw teacher: agree, collision/NAVSIM-no vs the reverse | at-fault: agree, same split |
|---|---|---|
| originals (12,800) | 90.3-92.3 %, 517 vs 674 | 90.0-92.1 %, 441 vs 791 |
| 0.75x copies (1,600) | 97.5 %, 19 vs 21 | 97.9 %, **2 vs 31** |
| 0.5x copies (1,600) | 85.6 %, **222 vs 9** | **93.5 %, 62 vs 42** |
| STOP (200) | 97.0 %, 0 vs 6 | 97.0 %, 0 vs 6 |

It repairs the 0.5x bias but tips the 0.75x copies one-directionally the other way (lenient), and moves the originals'
errors toward misses at unchanged agreement. Label-ranking ceiling with it: C1 0.885 (from 0.869), 0.5x-only 0.861
(from 0.830), C2 0.907 (from 0.906). ⇒ NEITHER collision variant makes BOTH 0.75x and 0.5x valid by the section-2 rule.
Diagnosis (`eval/diag_v4_nc_residual.py`, `v4_nc_residual.json`): my stated hypothesis -- the residual comes from
NAVSIM tracking the plan (LQR) while the teacher injects it -- HOLDS for the originals (max |plan - NAVSIM-simulated
ego| median 4.02 m where the labels disagree vs 1.65 m where they agree) and is REFUTED for the copies (0.5x: 2.43 m
disagree vs 2.88 m agree; 0.75x: 0.88 vs 1.33 m) -- every half-speed copy is tracked ~2-3 m off, agree or not. The
residual on copies is the rules: NAVSIM counts lateral hits in multi-lane / non-drivable states and static objects
(0.5), samples 41 states at 0.1 s over ALL objects, while the teacher samples prefixes at 0.4 s (stride 2) over the 9
nearest agents. ⇒ **The lever that would admit 0.5x copies is a NAVSIM-faithful collision label ON THE SIMULATED
STATES** (vendoring `PDMScorer._calculate_no_at_fault_collision` + a PDM observation built from the scenario's logged
objects), the same pattern that made drivable area and comfort 100 %. Not built here: BLOCKED on scope (multi-hour
vendoring + its own 100 % validation against the harness NC column, which already exists for every slot above).

## 2. The configuration under test -- fixed by a rule stated now

RULE: test the configuration whose every trainer-read component passed validity ON THE COPIES: collision agreement on
copies >= the originals' (>= 90 %) and NOT one-directional (neither disagreement direction > 3x the other).
**C2 = `--slow-copies --slow-factors 0.75 --slow-n-src 8` (STOP on, sources scorer_top, `--slow-frac 0.5`), raw teacher
collision.** 0.75x: 97.5 %, 19 vs 21; STOP: 97.0 %, 0 vs 6. 0.5x fails the rule (85.6 %, 222 vs 9) and is NOT in C2.
C2 replaces 9 of the 64 slots in half of the sets: labelling +9 rollouts on 65 (+14 %) per carrying set, ~+7 % overall.
C2 is the CODE DEFAULT of `refe/onpolicy_label_v4.py --slow-copies` (factors 0.75); 0.5 is opt-in only. The validity
runs above named `--slow-factors 0.75,0.5` explicitly (eval/validate_slow_labels_v4.py).

## 3. Design -- a DEV-BOX scorer-only fine-tune (no pod is available; PI 2026-09-27)

WHY SCORER-ONLY IS EXACT FOR THIS QUESTION: the live scorer is detached from BOTH the trajectories and the visual context
(`model.py:746`, `:751`, `detach_scorer_context=True`), so the score loss reaches only `score_q_mlp`, `score_dec`,
`score_head`. A fine-tune of those modules on a frozen, cached context (`visual_ctx = scene_proj(visual)`, the scoring
decoder's K/V) is the live score-loss path, with ONE declared simplification: the live trunk keeps moving under the WTA
loss while the cached one is snapshot 015's.

Starting point: `snap_epoch015.pt` (md5 `d7c59f4f2fbcbde3e2dec8f67d63a7e7`, the snapshot validity used).

### 3a. Data -- OPTION (a), RECOMMENDED: a navtrain subset, frames copied read-only from the pod
* **Already on the dev box** (MEASURED 2026-09-27): navtrain DBs for **214 of 1,192** navtrain logs (16.1 GB,
  `D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval`); bank rows `D:/Projects/TanitAD/data/refe_navtrain10/r0/
  targets_rank0.jsonl` (33,704 rank-0 rows over 408 logs, 214 of them DB-local). ⚠️ NOT
  `D:/Projects/TanitAD/data/refe_targets_4cam/`: that is the early 7-log bank (2,182 rows; only 2 logs have a local DB).
  Calibration: built locally from those DBs (`refe/calib_table.py --db-root`).
* **Sample**: rank-0 rows of the 214 DB-local logs; **1,600 train samples from 170 logs + 300 val samples from the other
  44 logs** (log-disjoint), drawn with seed 20260927. Gate: each chosen row's key and teacher `traj` equal the POD bank's
  row (`/workspace/data/refe_navtrain/train_grow/targets_rank0.jsonl`) -- the coordinator compares hashes of the 1,900 rows.
* **Copy from the pod (the only transfer)**: the 4 camera frames of each chosen sample = the row's `image` paths, from
  `/workspace/data/navtrain_pixels` (FrameStore: loose files or `<log>_<CAM>.zip`), delivered as LOOSE files
  `<root>/<log>/<CAM>/<id>.jpg` (FrameStore's loose-file index reads them). ⚠️ Extract single frames; do not ship whole
  per-log zips (a navtest per-log-camera zip is 16-36 MB for 79-159 frames). **Bytes: 1,900 x 4 x ~215 KB (MEASURED
  median frame, local navtest zips) = ~1.6 GB.** Nothing else is needed from the pod: no DBs (local), no on-policy sets
  (regenerated below), no calib (built locally). Optional cross-check: the pod's v3 sets for the same keys (~65 KB each).
* **Generated locally**: ONE snapshot-015 forward per sample (bf16 autocast + TF32, the trainer's numerics, as
  `onpolicy_dump.py`) writes the shared trunk cache (section 6) AND the sample's 64 proposals + logits -> props chunks in
  `onpolicy_dump.py --emit-logits`'s line format -> `onpolicy_label_v4.py --slow-copies --slow-factors 0.75
  --slow-frac 1.0` (every sample labelled with its copies; the arms below choose per sample) on the dev box's CPUs.

### 3b. Data -- OPTION (b): a navtest MECHANISM test (no transfer) -- NOT recommended as the adoption test
Fine-tune on Amendment 5's 923 tokens (43 logs), test on W3's 200 (93 disjoint logs); frames and DBs are local; the LABEL
SOURCE is the NAVSIM harness (81 harness runs over the 923 tokens: the 64 originals + the copies + STOP).
⛔ **CONTAMINATION, stated plainly: the scorer would be trained on the benchmark's own verdicts on navtest scenes.** The
test logs are disjoint, but the four cities, the maps, the sensor rig and the LABEL FUNCTION are shared with every
reported navtest number. **Such a scorer must never produce a reported number**: no PDMS, no E-6 row, no registry or
claims entry, no comparison with the programme's navtest points; its weights are quarantined and deleted after the
readout. It answers "can CORRECT slow-plan labels teach this scorer?" (a ceiling), not "do the v4 labels?" -- and the v4
labels' content is exactly what validity found wanting at 0.5x. Use it only as a follow-up if (a) fails, to separate
"the labels are wrong" from "the mechanism cannot work".

### 3c. Arms (identical samples, order, steps and seeds; only the label content differs)
* **A0** -- snapshot 015 as is (no fine-tune): the reference point.
* **V3 (CONTROL)** -- every set = the sample's 64 originals with their labels (label_version 3 content: the v4 line's
  kept slots + the side file's dropped slots; proven identical to the real v3 labeller in section 1).
* **V4** -- C2 at `--slow-frac 0.5`: carrying samples (the labeller's own hash, frac 0.5) use the served v4 set (55
  originals + 8 x 0.75 copies + STOP), the others the V3 set.
* Reported, not gating: **V4-all** (frac 1.0: the cost of mixing every set).
* 3 seeds per fine-tuned arm (batch order); A0 is deterministic.

### 3d. Training (the live score-loss term, scorer modules only)
AdamW, **lr 6.9e-5 constant** (the live cosine at the epoch-15 boundary: 2e-4 x 0.5 x (1 + cos(pi x 15/25)); the exact
value is read from the pod's `metrics.jsonl` before the run and recorded), weight decay 0.01, score weight 0.1, batch 16
sets, **4 epochs** over the 1,600 train samples; loss = `train.py:1339-1342` (BCE with logits on the six components, mean
over components, summed over the set's slots, / (64 x batch)) with the NAVSIM drivable-area override of `train.py:457-460`;
the served set scored as its own set through `REFe.score_trajectories` on the cached context. navtrain-val BCE per
component is logged every epoch (reported).

## 4. Evaluation -- W3's 200 navtest tokens (93 logs); EVERY trajectory below is already harness-scored
Candidates per token: the 64 originals (E-6 table) + the C2 extras (the 0.75x copies of snapshot 015's own top-8 =
harness `f075_r00..r07`, and STOP = `stopzeros`) -- the SAME trajectories for every arm. Contexts: the snapshot-015 cache.
* **E1 PRIMARY -- slow-plan ranking.** Each arm scores the extras with the MASKED route (the 64 keep their pure-set
  scores; each extra attends to the 64 and itself -- `stop_candidate_probe.score_masked`, the arrangement Amendment 6's
  probe validated). Per token: the fraction of pairs with >= 1 extra, harness PDMS not tied, that the arm's predicted v1
  aggregate orders like the harness (ties 0.5); token mean. The plain-set route (64 + extras attending to each other,
  the arrangement V4 trains in) is REPORTED beside it, not gating.
* **E3 NON-INFERIORITY on the 64** (the shipped route, the 64 alone): (E3a) the same pair concordance over pairs of
  originals; (E3b) PDMS of the argmax pick among the 64.
* **E2** (reported): PDMS of the pick among 64 + extras (masked), with Amendment 6's family guard (`families6.py`).
* **E4** (reported): predicted NC / TTC / EP of STOP and of the copies vs the harness means (the STOP probe measured the
  untrained scorer at NC 0.76 vs 0.97).
* **Estimator**: per-token V4 - V3 (each arm's seed mean), **paired log-cluster bootstrap over the 93 logs, 10,000
  resamples, 95 % percentile, seed 20260927**. It answers "another draw of EPISODES". **Seed floor**: the largest
  |difference| between two seeds of the same arm in the token-mean metric; an effect below it is not an effect
  (CLAUDE.md, H-ESTIM-SEED-1). Inference is deterministic (the STOP probe: second forward identical).

## 5. Decision -- both outcomes committed now
* **ADOPT** (effectiveness proven) iff ALL: E1 V4 - V3 >= **+0.03** with CI lower bound **> 0** and the effect larger than
  the seed floor; E3a CI lower bound **> -0.01**; E3b CI lower bound **> -2.0** PDMS points. -> section 8, with the PI's go.
* **REFUTED** iff E1's CI upper bound < 0 (the labels make slow-plan ranking WORSE) -> not deployed; next lever: a
  NAVSIM-faithful collision label on the simulated states (the DAC/comfort pattern) and a re-test.
* Otherwise **NOT PROVEN** -> not deployed.
* Reported, never gating: A0 vs V3 (what the fine-tune alone does), V4-all vs V4, E2 with its family guard, E4, the
  navtrain-val BCE per component.

## 6. The SHARED trunk cache (one format for measure 5 AND the training-side decoder proxy)
One directory per (checkpoint, split): `D:/Projects/TanitAD/data/refe_trunk_cache/<ckpt md5[:8]>/<split>/`.
* `manifest.json`: checkpoint path + md5 + sha256; `REFeConfig` (backbone, n_cameras, img_h, img_w, patch, dec_width,
  n_registers, n_proposals, horizon_steps); numerics (bf16 autocast + TF32, eval, no grad); per-sample calib ON; the
  sha256 of model.py, ckpt_io.py, load_dinov3.py and the builder; the ordered sample keys `log|token|step|rank`; shard
  layout; GPU; timestamps.
* shards `shard_NNNNN.npz` (64 samples each), arrays aligned to the manifest order:
  `key` [n] str · **`visual_ctx`** [n, 7680, 256] (= `scene_proj(visual)`, the SCORER's K/V, model.py:732) ·
  **`scene_ctx`** [n, 64, 256] (= `scene_proj(reg_compress(registers, visual))`, the TRAJECTORY decoder's K/V,
  model.py:730-731) · **`q0`** [n, 64, 256] (= `queries + ego_tok`, the trajectory decoder's input, model.py:739) ·
  `ego` [n, 7] · `goal` [n, 4] · `calib` [n, 4, 16] float64 · `teacher` [n, 20, 3] (the bank row's WTA target; zeros on
  navtest) · **`props`** [n, 64, 20, 3] · **`logits`** [n, 64, 6].
  Store `visual_ctx` as float16 only if the bf16 -> fp16 round trip is exact on every shard (asserted), else float32.
* Tier 2, only if a proxy trains `reg_compress` / `scene_proj`: `visual` [n, 7680, 1024] (backbone tokens + pos3d).
* **Build gate** (per shard, before it is used): the decoders re-run from the cached tensors reproduce the forward's
  `props` and `logits` (same autocast) to max |d| <= 1e-3, and a deliberately shuffled `visual_ctx` must FAIL it.
* **Bytes**: Tier 1 ~4.0 MB/sample (visual_ctx 3.93 MB) -> 2,100 samples ~8.4 GB; Tier 2 +15.7 MB/sample (+33 GB).
  D: has ~158 GB free.

## 7. Compute and bytes (the 4060 is busy 16:25-17:45, ~17:45-20:00, then ~80 min every ~6.9 h)
| item | cost | class |
|---|---|---|
| pod -> dev box | ~1.6 GB of frames (1,900 samples) | ESTIMATED from the MEASURED median frame |
| forward + cache, 1,900 navtrain + 200 navtest | ~0.7-0.9 GPU-h (1.1-1.5 s/sample); upper bound 4.2 h at the STOP probe's 7.2 s/token full-planner path | ESTIMATED; a 20-sample timing gate runs first |
| v4 labels, 1,900 samples, CPU only | ~4 h wall at 5-6 processes (MEASURED today: 200 full v4 sets in ~25 min at 5 processes) | MEASURED rate |
| fine-tune: 2 gating arms + V4-all, x 3 seeds x 4 epochs | ~30-45 min (I/O-bound: 6.2 GB of context per epoch from D:) | ESTIMATED |
| evaluation, 200 tokens x 73 candidates x 10 models | minutes | ESTIMATED |
| disk | cache ~8.4 GB (Tier 1), labels ~150 MB | ESTIMATED |
The forward and the fine-tune fit the free windows (e.g. from 20:00); the labelling is CPU-only and can run meanwhile.

## 8. Live deployment -- only after ADOPT and the PI's go
Pod files change ONLY under `/workspace/refe-op` (the on-policy pipeline's code; the live trainer's `/workspace/refe-plan`
is untouched and is NOT restarted): NEW `refe/onpolicy_label_v4.py`, `refe/slow_copies.py`; `code/measure5_v4_hooks.patch`
applied (`refe/onpolicy_dump.py --emit-logits`, `code/op_pipeline.sh OP_SLOW`, `code/op_status.py` v4 counters); and
`refe/planner.py` must be the Amendment-5 version (the preflight checks the aggregate). Every file md5-verified after the
ship (never git on the pod).
1. Preflight, writes nothing: `$OP_PY refe/onpolicy_label_v4.py --queue x --out x --rank 0 --preflight --slow-copies
   --slow-factors 0.75` (and `--rank 1`) -> `ZZOPLABEL4_PREFLIGHT_OK`.
2. `touch $ROOT/STOP` -> the supervisor stops the dump + labellers and exits; wait until `ps` shows none (flock race).
3. `rm $ROOT/STOP; OP_SLOW=1 OP_SLOW_ARGS="--slow-factors 0.75 --slow-frac 0.5" setsid nohup bash op_pipeline.sh >
   $ROOT/logs/sup.out 2>&1 &` (same worker names, so each labeller resumes its own in-flight chunk).
4. Verify from processes and artifacts, not the launch: `ps -eo args | grep -c "[o]npolicy_label_v4.py --slow-copies"`
   == W0+W1; the dump's cmdline carries `--emit-logits`; new lines have `label_version` 4 and `slow.sources` ==
   `scorer_top`; every `status_r*_w*.json` has `selfcheck_failed` 0; `op_status.py` shows `v4` > 0.
5. The trainer reads the v4 sets at its next epoch boundary: its `ON-POLICY scorer bank` line must keep `0 incomplete`.
6. Declared the same turn (Master Mind): MODEL_REGISTRY §14.1 "Label change (label_version 4, mid-run)" row and declared
   departure (8) "a label-free half of the on-policy sets carries 8 slowed copies (0.75x, the dump-time scorer's own
   top-8) + STOP in place of 9 randomly chosen non-source slots, labelled by the same labeller functions";
   GOALS_AND_CLAIMS row with this document's hash. The trainer's argv does not change (no restart, so no
   `--declare-change`); the declaration lives in the registry, the labeller's status json and op_status.
7. Rollback: restart the supervisor with `OP_SLOW` unset (v3 labellers). ⚠️ Already-written v4 lines keep superseding v3
   for their samples ((ckpt_step, 4) > (ckpt_step, 3)); never delete or truncate a set file inside an epoch (the
   trainer's byte snapshot refuses a file SHORTER than recorded, `train.py:225-230`) -- move v4 files aside only right
   after an epoch boundary.

## 9. What is NOT in the tested configuration, and what would bring it in
* **0.5x copies**: excluded -- the raw teacher collision label is one-directionally wrong on them (222 vs 9) and the
  at-fault variant, which balances them (62 vs 42), makes the 0.75x copies one-directionally lenient (2 vs 31)
  (section 1b). Admission needs the NAVSIM-faithful collision label on simulated states; then re-run
  `eval/validate_slow_labels_v4.py` (`label --arm <name>`, `analyze --arm <name>`) and apply the section-2 rule.
* **`--nc-at-fault`**: implemented, validated as a mechanism (self-test + the table in 1b), NOT recommended: it
  changes the originals' collision labels too (a second lever inside measure 5) without improving their agreement.
* **Ranks 8-15 as sources (17 copies at 0.75x)**: the harness has them (`f075_r08..r15`), the teacher labels were
  not computed for them here -- unvalidated, so not in C2.
* **Rank 1 (augmented routes)**: validity measured on rank 0 only; the v4 code path is the same with the route patch.
