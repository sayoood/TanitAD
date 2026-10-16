# PREREG: scene-tap vs visual-tap probe for the REFe aux drivable-area head (zero A40)

Written 2026-10-15 by the probe-build agent, **before any held-out IoU of this probe exists** (no unit has been trained when this file is written; `folds.json` and the instrument facts below
are the only measurements made). The design in sections 2-5 is COMMITTED by the Master Mind; the agent added only the instrument facts, each with a `file:line` or an artifact path.

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan` at commit `8e106f9805984ff120631640bba586231de1833e` (fresh archive of the remote tip; the E:/D: trees are stale and were not used).
Evidence classes: MEASURED (ours, artifact named) | INHERITED (another file, not re-derived here) | HYPOTHESIS.

## 1. Question, and why it is the next lever

S1's aux drivable-area head, reading the frozen trunk's `visual_ctx` [B, 7680, 256], scores navtest **P1 IoU [0, 40) = 0.496 / 0.521 / 0.486** over three seeds against K2's bar of **0.80**
(`raw/2026-10-15-s1-read/S1_READ.json`, MEASURED; `REPORT.md`). The pre-registered next-arm row "P1/P3 fail (K2)" names three levers: the **scene tap** (`scene_ctx` [B, 64, 256]), an **unfrozen-trunk graft**
(a PI decision), and a **1 m grid** (ranked last by S1 prereg R3).

This probe answers, on the dev box and without the A40, **whether frozen features can carry the road at all, and through which tap**: a head-only fit on the CACHED frozen features, visual vs scene,
log-disjoint folds, both outcomes committed below.

## 2. Instrument facts (all MEASURED 2026-10-15)

| fact | value | source |
|---|---|---|
| probe tokens | **1,058** = every navtest token that has BOTH a cached `visual_ctx` AND a P1 label | `raw/2026-10-11-c3-selection-headroom/tokens.json` + `table.npz` (same set, 1,058); `ls c3_full_ctx`: 0 of 1,058 ctx files missing; `folds` command: `n_labelled` 1,058 |
| probe logs | **123** (= `perception_navtest.json` `n_logs` for H1-s0) | `raw/2026-10-15-s1-read/instrument/perception_navtest.json` |
| P1 labels | the navtest label store, `store_sha256` `985a3bc77f3ab7a96b0dceace48e73b80e5b69b70710958635060939bce0d7c1`, 12,146 keys, 136 logs; pinned by `aux_s1_readout.LABEL_PINS["navtest"]` and checked by `AuxLabelSource` on open | `D:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest/manifest.json` |
| visual ctx | `D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full_ctx/ctx_<token>.npy`, float32 [7680, 256], 2,116 files (1,058 ctx + 1,058 logits), the planner's fp32 path; identity vs `table.npz` logits **bit-identical** (`c3_full/identity.json`: 1,058/1,058, max abs delta 0.0) | `aux_navtest_forward.py` module docstring; `c3_full/identity.json` |
| log id | the token's log exactly as `aux_s1_readout` keys it: `tokens.json["token_log"][token]`, which is the label-store shard name and the `log` column of every `head_pred_*.npz` | `aux_s1_readout.py:frame_stats` (key = `(log, token)`) |
| folds | `fold = int(sha256(log_id.encode("utf-8")).hexdigest(), 16) % 5`; **tokens per fold 197 / 242 / 196 / 218 / 205; logs per fold 27 / 27 / 24 / 27 / 18**; so a fold model trains on 861 / 816 / 862 / 840 / 853 tokens from 96 / 96 / 99 / 96 / 105 logs | `folds.json` (this folder) |
| `folds.json` sha256 | **`20dec3431663de2201190020f97c74f7975468a8e9e178f3db4d52fbacd5885f`** (87,225 bytes; LF; `json.dumps(sort_keys=True, indent=1)`; rows `[token, log, fold]` in `table.npz` token order). The run refuses to start if the folds it rebuilds hash differently (`--folds-sha256`). | `scene_tap_probe.py folds` |

### 2.1 The S1 head's hyper-parameters (read from source and from the S1 H1 arms' argv; the probe copies them)

| item | value | source |
|---|---|---|
| head | `AuxDrivable`, depth 2, `feed="off"` (H1), 2,440,450 parameters, 40 x 32 cells of 2 m | `refe/aux_drivable_head.py:76-90`; H1-s0 `sft.jsonl` `data` event `aux_drv.n_params` 2440450 |
| loss | `aux_loss` = BCE(logit, block mask fraction) + `w_sdf` x Huber(sdf/6, target/6, beta 0.1); `w_sdf` = **1.0** (`AuxDrivableConfig` default; the trainer builds the config without overriding it) | `aux_drivable_head.py:87, 218-232`; `scorer_finetune.py:610` |
| weight in the total | `--w-aux-drv 0.1`, via `combine` (always multiplied in) | `s1aux_pod.sh:123`; `scorer_finetune.py:776` |
| head optimiser | its OWN `AdamW(lr = --aux-drv-lr 2e-4, weight_decay = 0.01)`, grad-clip norm **1.0** | `scorer_finetune.py:613, 1362`; `s1aux_pod.sh:123` |
| schedule | linear warm-up over `--warmup` (parser default **50**) then cosine to 0 at the last update, at the head's peak 2e-4 | `scorer_finetune.py:1331, 1338`; `s1aux_pod.sh:115` (defaults not named in the argv) |
| S1 budget | `--max-updates 400`, `--batch 8 --accum 8` = **64 frames per update**, i.e. 25,600 draws | `s1aux_pod.sh:120`; H1-s0 `sft.jsonl` `start` event: `updates 400, micro_per_update 8, batch 8` |
| S1 training precision | the trunk and head run under `torch.autocast("cuda", bfloat16)`; the readout evaluates in fp32 on the planner's fp32 ctx (A2.3) | `scorer_finetune.py:692`; `aux_navtest_forward.py` PRECISION paragraph |

### 2.2 What the S1 head actually SAW (the number that sizes the data lever) - corrected by the Master Mind 2026-10-15

* **`AuxLabelSource(..., max_logs=64)` (`scorer_finetune.py:611`) is NOT a training-data limit and "S1 trained on 64 logs" is FALSE.** `max_logs` bounds an LRU cache of decoded shards
  (`aux_drivable_head.py:410-478`: one shard per log is kept, the oldest evicted past 64, `popitem(last=False)` at :477-478). Every one of the store's logs is readable.
* The label store holds **103,139 keys over 1,192 logs** (`label_store/manifest.json`, MEASURED); the S1 bank is 103,037 frames over **1,191 logs**, 24 held out (`raw/2026-10-13-aux-s1-data-audit/S1_DATA_AUDIT.md:68`).
  The training draw is `tr_idx`, **174,912 rows** (`S1_DATA_AUDIT.md:33`; H1-s0 `sft.jsonl` `data` event `train_sets 174912`).
* **Frames seen = updates x effective batch, from the arms' own `start` event** (H1-s0 `sft.jsonl`: `updates 400, micro_per_update 8, batch 8`): **400 x 64 = 25,600 frame draws per head**, about **0.146 of one epoch** of `tr_idx`.
  (MEASURED from the arms' log; `--max-updates 400` in `s1aux_pod.sh:120`.)
* **Distinct frames and logs of that draw: 23,936 / 23,954 / 23,972 distinct frames over 1,133 / 1,143 / 1,145 distinct logs** for seeds 0 / 1 / 2, MEASURED by emulating the trainer's own draw
  (`raw/2026-10-13-aux-s1-data-audit/evidence/sim_draws.json`, torch 2.11). The arms' logs do not record distinct frames or logs, so these two are **UNVERIFIED against the pod's actual draw**
  (the audit's own caveat: "the pod's torch may draw a different, equal-sized subset").
* ⇒ **The data lever is "more head updates / frames seen", not "more logs".** S1's head saw ~25.6 k frame draws in ONE pass; the probe's fold model sees 16,000 draws (2,000 x 8) over ~850 tokens, i.e. ~19 passes
  over ~100 logs. The probe therefore has FEWER distinct frames and logs than S1 but about as many gradient draws. Section 6 states what that does to the reading.

## 3. Design (COMMITTED)

**Data and folds.** The 1,058 tokens of section 2; K = 5 log-disjoint folds by the rule above.

**Arms.** All use the S1 head's own config and hyper-parameters (section 2.1).

| arm | what it is |
|---|---|
| **V** | tap "visual": the head reads `visual_ctx` [1058, 7680, 256] |
| **S** | tap "scene": identical to V except the tap; reads `scene_ctx` [1058, 64, 256] |
| **C** | the shuffle control: V, but the train-fold labels are **deranged across the train tokens** (`AUX.derangement(train tokens, 20260927 + fold)`: one fixed derangement per fold, no fixed point). It learns the label MARGINAL only. Evaluation always uses the TRUE labels. |

**Training protocol.**
* V and S: seeds {0, 1, 2}. C: seed 0 only. Every (arm, seed, fold) trains on the 4 other folds and predicts its held-out fold once: **35 trainings** (15 + 10 + 10).
* **Budget FIXED: 2,000 updates at batch 8**, no accumulation, no early stopping, no selection on held-out data. (S1 ran 400 updates at an effective batch of 64; the probe's 16,000 draws over ~850 tokens is ~19 epochs.)
* The head's init seed is the unit's seed (`AuxDrivableConfig.seed`, the trainer's `--seed`), so V, S and C at seed 0 start from the SAME head; the batch order is a seeded `randperm` epoch sampler (`seed * 100 + fold`).
* Training precision as S1: bf16 autocast on the GPU, fp32 master weights; **prediction in fp32** (the readout's convention). `visual_ctx` is held in RAM as **float32 (no fp16)**: 8.3 GB against 17.9 GB free when measured.
* **Lowering the update count** is allowed ONLY before any held-out IoU is computed, and ONLY if a measured compute estimate of the full grid exceeds **8 h**; the estimate and the new value go into Amendment A1 below.
  The 3 seeds, the 5 folds and the identical protocol across arms may NOT be reduced.

**Metrics.** Every IoU below is read by `aux_s1_readout.frame_stats` + `aux_s1_readout.perception_reads` (reused, never reimplemented; `guard_p1_function` fails the run if either object has been replaced).
* **Held-out:** pooled held-out **P1 IoU [0, 40)**, all labelled frames, each token predicted exactly once (a fold model never predicts a token it trained on).
* **In-sample:** the train-fold IoU [0, 40) of each fold model on its own training tokens (true labels, also for C), averaged over the 5 folds, then over seeds.
* **Secondary:** SDF MAE (P2: overall and near-boundary), and the other P1 bands, per arm per seed.
* **Init floor:** the untrained head (same seeds 0 / 1 / 2, both taps) on all 1,058 tokens, same readout.

**Estimator.** Paired log-cluster bootstrap over the 123 navtest logs, **10,000 resamples, seed 20260927** (`aux_s1_readout.boot_weights`: the same draw as `proxy_eval.boot`).
Per resample: the pooled IoU [0, 40) per arm per seed, the mean over seeds, then **S-V, V-C, S-C**; the CI is the 2.5-97.5 percentile of the resample differences, the point is the difference of the full-data means.
* **Seed floor** of an arm = max pairwise |IoU_i - IoU_j| over its 3 seeds on the full set. C has one seed, so it has **no** floor; V-C and S-C then apply V's / S's floor only, and the result lists the missing one.
* **"Separated beyond the floor"** = the paired 95 % CI excludes 0 **AND** |point| exceeds BOTH arms' seed floors (those that exist).
* **What the interval answers:** "another draw of LOGS" (the bootstrap) plus "another TRAINING SEED" (the floor). **It does NOT answer "another training corpus"**, and the five fold models are held fixed inside the
  resamples (the folds share 3/4 of their training data, so the pooled predictions are not independent draws). Name the question in any quote of it (`H-ESTIM-SEED-1`).

## 4. Committed outcomes (verbatim table + the mechanical rule)

The K2 rule is the S1 one: **the point estimate against 0.80**. "V held-out" / "S held-out" below is the **mean over the three seeds of the pooled held-out IoU [0, 40)**; every per-seed value is reported beside it.

| V held-out | S held-out / S-V | reading | next lever |
|---|---|---|---|
| >= 0.80 | any | frozen VISUAL features carry the road; S1's 0.50 is a head-TRAINING limit (data / domain), not a feature limit | S1c: the visual head with more label logs (cost it); scene tap not needed |
| < 0.80 | S >= 0.80 and S-V separated > 0 beyond floor | the SCENE tap carries the road, the visual tap does not | S1b: a scene-tap A40 arm (cost it) |
| < 0.80 | S < 0.80, S-V separated > 0 beyond floor | scene is the better frozen tap but not enough | unfrozen-trunk graft = PI decision (scene tap preferred), plus the in-sample read below |
| < 0.80 | S-V not separated, or adverse | scene tap ELIMINATED as the K2 lever | unfrozen-trunk graft = PI decision, plus the in-sample read below |

Mechanical form (`scene_tap_probe.decide`): `ROW1_V_CARRIES` if V >= 0.80; else `ROW2_S_CARRIES` if S >= 0.80 and S-V is separated beyond the floor with a positive point; else `ROW3_S_BETTER_NOT_ENOUGH` if S < 0.80 and
S-V is separated beyond the floor with a positive point; else `ROW4_SCENE_ELIMINATED` (S-V not separated, or separated and negative). The four rows are exhaustive. Thresholds are compared with `>=` / `<` on the unrounded value.

**In-sample read (applies in every row; arm V).**
* **D1:** train-fold IoU >= 0.80 while held-out < 0.80 => generalisation / data-limited, so **more label data is a lever** (committed wording: "more label logs"; sized as frames seen / head updates, section 2.2).
* **D2:** train-fold IoU < 0.80 => the head cannot fit these frozen features even in-sample, so **the tap or the trunk is the lever**.
* (If V's held-out clears 0.80 the in-sample read is reported as not applicable.)

**Prior control.** V-C is "the features' contribution over the label marginal". If V-C is NOT separated beyond the floor, **every V reading is PRIOR-DOMINATED** and is reported that way.

## 5. Guards, each with a deliberate-regression arm (`test_scene_tap_probe.py`)

| guard | what it asserts | RED mutation |
|---|---|---|
| (a) `guard_split` | no token AND no log on both sides of a fold; the two sides cover every token once | one held-out token injected into the train side |
| (b) `guard_tap` | the head's input IS the declared tap tensor (`AUX.tap_is`: same storage and shape) and not the other tap | `select_ctx` returns the visual tensor for arm S |
| (c) `guard_p1_function` | `P1_READ is ASR.perception_reads`, `STATS_FN is ASR.frame_stats`, the K2 band / bar; plus a behavioural IoU oracle | a reimplemented copy swapped in |
| (d) `guard_label_perm` | C's labels are deranged within the train fold only; held-out and non-train labels bit-identical | a permutation over ALL tokens |
| (e) `guard_batch` | no held-out token is ever in a training batch (checked on every one of the 2,000 batches) | a batch drawer that ignores the train indices |

`PROBE_RESULT.json` records the outcome row computed by `decide` from the numbers above and nothing else; **no interpretation beyond the mechanical row is written by the job.**

## 6. What this probe does and does NOT answer (stated before the data)

* **It is a different regime from S1.** The probe's head sees ~850 distinct labelled frames from ~100 logs, ~19 passes (16,000 draws); S1's saw ~24,000 distinct frames from ~1,140 logs in ~1 pass (25,600 draws). A low V here is therefore **confounded with
  the number of DISTINCT frames and logs** (not with the number of gradient draws, which are comparable), which is exactly what D1 / D2 and the V-C control are for, and is why S1c (the committed "more label logs" row) is a row of its own. A high V here is the strong result: frozen visual features carry the road even from
  ~850 frames.
* **Domain.** The probe trains and tests on navtest frames (log-disjoint folds); S1 trained on navtrain and tested on navtest. Within-navtest folds remove that shift, so a V above S1's 0.50 mixes "features can carry it" with
  "no train-to-test domain shift". The Master Mind reads this against S1's P1-HO (held-out navtrain, 0.631 / 0.643 / 0.621).
* **Tier:** T0-like. A head's perception on teacher-forced frames; never driving performance.
* **Estimator question:** logs + training seed, per section 3. Not another corpus, not another inference run (the heads are deterministic functions of the cached ctx).

## 7. Admissibility of `scene_ctx` (reported, not decided; the finding is in `P2_ADMISSIBILITY.md`)

SPEC 6 says the head's input carries no ego, goal, navigation command or map. See `P2_ADMISSIBILITY.md` (hop-by-hop `file:line`).

## 8. Amendments

### A1 (2026-10-15, written before any held-out IoU of this probe exists): the measured compute estimate and the final update count

**The update count stays at 2,000.** The full-grid estimate is far under the 8 h bar that alone licenses lowering it.

MEASURED by `scene_tap_probe.py smoke` (`smoke/smoke_timing.json`; 20 timed updates after a 5-update warm-up, batch 8, RTX 4060 shared with the PI's desktop; **timing only, no metric computed**):

| quantity | value |
|---|---|
| arm V/C step (real `visual_ctx`, [8, 7680, 256]) | **0.1135 s / update** (peak 0.58 GB allocated) |
| arm S step (a SYNTHETIC [8, 64, 256] ctx of the real shape; only its timing is used) | **0.0245 s / update** (peak 0.36 GB) |
| fp32 prediction | 0.0130 s / token (V), 0.0020 s / token (S) |
| `visual_ctx` read from D: (exFAT, cold) | 0.114 s / file, so ~2 min for 1,058 files; free RAM 17.9 GB against 8.3 GB needed: **float32 in RAM, no fp16** |

ESTIMATED grid (from the rows above): 20 V-type units (15 V + 5 C) x (227 s train + ~14 s predict + in-sample read) ~ 83 min; 15 S units x ~60 s ~ 15 min; 6 init floors ~ 1 min; ctx load ~ 2 min;
shared-GPU lock re-acquisitions (>= 47 s each, one per 20-min lease) ~ 6 min. **Total ~ 1.8-2 h**, against the 8 h bar.
ESTIMATED scene extraction (stage 1 of the chain): the 8-token smoke ran 1.18 s per token steady-state (first token 3.1 s); the earlier all-in `c3_full` pass cost 5.5 s / token, so **1.0-1.6 h** for 1,058 tokens. Chain total ~ 3-3.6 h.

Same-forward smoke (8 tokens, one log, `aux_navtest_forward/c3_full_scene_ctx/extract_log.jsonl`): `visual_ctx` recomputed in the same forward vs the `c3_full_ctx` cache: **max |d| = 0.0 (bit-identical, 8 of 8)**, scorer logits
bit-identical to `table.npz` (8 of 8), tap identities True (8 of 8). The full stage re-checks every token and STOPS on the first |d| > 1e-4.

Execution (committed): ONE detached chain (`chain.py`): stage 1 extraction (resumable, shared GPU lock in <= 20-min stretches) -> stage 2 the grid in the order V-s0 (5 folds), S-s0, C-s0, V-s1, S-s1, V-s2, S-s2, so a killed run
still yields complete (arm, seed) groups; the init floors first. Units are resumable (`D:/Projects/TanitAD-artifacts/scene_tap_probe/units/*.npz`); a unit whose config sha256 differs is refused, never mixed.
Tests: `test_scene_tap_probe.py` 19 passed (`logs/test_scene_tap_probe.log`). A plumbing dry run of `run_unit` on the real universe with random ctx and 3 updates (nothing scored) passed before launch.

### A2 (2026-10-16 ~00:45 Berlin, Master Mind; written AFTER the 2,000-update result, BEFORE any 10,000-update unit exists): does the head under-fit, or do the frozen features fall short?

**Why.**
- The 2,000-update grid read `ROW4_SCENE_ELIMINATED`, and its in-sample read was **D2**: V in-sample 0.704, held-out 0.637.
- D2 at a single update budget cannot tell two causes apart:
  - *(a)* the frozen visual features cannot express the road at 0.80;
  - *(b)* the head was under-trained (2,000 updates × batch 8 is about 19 passes over ~850 train tokens, cosine to 0).
- The registered next lever, an unfrozen-trunk graft, is a PI decision and costs A40 time. ⇒ Rule Zero: run the cheapest discriminator first.
- **Nothing in sections 2–5 changes.** ROW4 stands as read. A2 adds one arm and its own committed outcomes.

**Arm U10k.**
- The same as arm V, seed 0, with the same folds (`folds.json` sha256 `20dec343…`), tokens, labels, head and optimizer.
- The ONE change is **`n_updates` = 10,000**. The learning-rate schedule is the code's own function of `n_updates`, unchanged.
- Units go to a SEPARATE work dir. The config sha256 differs, so the code refuses to mix them with the 2,000-update units.
- Only arm V, seed 0, is run (5 units). No S, no C, no other seeds.

**Reads (same estimator: paired log-cluster bootstrap over the 123 logs, 10,000 resamples, seed 20260927).**
- **H:** U10k's pooled held-out IoU [0, 40).
- **I:** U10k's in-sample IoU [0, 40) (train folds).
- **Δ = H(U10k) − H(V, seed 0, 2,000 updates).** Paired on the same tokens; "separated" means beyond V's seed floor, 0.0036.

**Committed outcomes (exhaustive, applied in this order, thresholds compared on unrounded values).**

| row | condition | reading | next lever |
|---|---|---|---|
| **U1** | H ≥ 0.80 | the 2,000-update budget was the limit; frozen visual features carry the road | S1c: S1's visual head trained ≥ 5× longer (cost it on the dev box first, then the A40) |
| **U2** | H < 0.80, I ≥ 0.80, Δ separated > 0 | longer training helps and the head can fit the features; generalisation is the limit | S1c with more updates AND more label logs, each as its own registered pair |
| **U3** | H < 0.80, I < 0.80 | the head cannot fit frozen visual features even at 5× updates ⇒ **the frozen trunk is the limit** | the trunk: the graft (PI decision) and P6's trained trunk |
| **U4** | H < 0.80, I ≥ 0.80, Δ not separated or ≤ 0 | it memorises without generalising | more label logs is the only frozen-trunk lever; the trunk otherwise |

**Reported, never gating:**
- the near-boundary and overall SDF MAE;
- the per-fold IoU;
- the train-loss curve's last 10 % (did it plateau?);
- the wall time.

### A2.1 (2026-10-16 ~01:15 Berlin, Master Mind; BEFORE any 10,000-update unit exists — the grid is still waiting for the GPU lock): what "separated" means in A2
A2's Δ clause says "separated = beyond V's seed floor". It is read with the probe's OWN rule (section 4 mechanical form, `scene_tap_probe.decide`): **the paired interval excludes 0 AND |Δ point| exceeds V's seed floor**. The floor is the value in `PROBE_RESULT.json` (`arms.V.seed_floor` = 0.003552…; A2's "0.0036" is that number rounded). `A2_RESULT.json` reports `row_depends_on_the_ci_clause`; a U2 / U4 that depends on it is quoted with that flag.
Implementation note (no effect on the contract): the A2 copy of the GPU lease never deletes another job's lock (the banked copy removed a > 10-min lock whose pid Windows could not see, which would steal an MSYS-pid lock of another session).

### A2 RESULT (2026-10-16 04:26 Berlin, mechanical; `raw/2026-10-16-scene-tap-a2/A2_RESULT.json`): **U2**
- **H = 0.675** held-out (95 % logs-only CI 0.643–0.708).
- **I = 0.894** in-sample.
- **Δ vs 2,000 updates = +0.040 [+0.032, +0.049]**: separated, and NOT dependent on the CI clause.
- ⇒ the head CAN fit the frozen visual features. **D2 at 2,000 updates was under-training, not a feature limit.** Generalisation is the limit, from ~100 training logs.
- Committed next lever: S1c with more updates AND more label logs, each as its own registered pair. S1c needs the A40 ⇒ the PI's decision.

### A3: CONSIDERED AND NOT RUN (2026-10-16 ~04:50 Berlin, Master Mind): the question is already answered by banked data
**The question.** A3 was drafted to ask whether S1's navtest 0.50 is update starvation: the same head, in-domain, at S1's 400 updates.

**The answer, already banked.** S1's own read banks that head on **navtrain held-out logs** at S1's exact budget. MEASURED, `raw/2026-10-15-s1-read/S1_READ.json` `outcome.per_arm.k2.*.p1_ho`:

| arm | navtest P1 IoU [0, 40) | navtrain held-out (24 logs) |
|---|---|---|
| H1-s0 | 0.496 | 0.631 |
| H1-s1 | 0.521 | 0.643 |
| H1-s2 | 0.486 | 0.621 |

The H2 arms read the same to 4 decimals.

**What that means.**
- In its own domain, at 400 updates, S1's head reaches **~0.63**, the same as the probe's navtest-trained head on navtest (0.635 at 2,000 updates).
- **The deficit is a navtrain → navtest gap of ≈ 0.13**, consistent across 3 seeds. It is not update starvation.
- The comparison is cross-instrument: different held-out sets, unpaired. It is descriptive, not a separated effect.

**Lever ranking (Rule Zero item 5).**
- The gap (≈ 0.13) is about **3×** the +0.040 that 5× more updates bought in-domain (A2).
- ⇒ the first lever is to **LOCATE the navtrain → navtest shift**. Candidates: label construction, city or scene composition, region statistics. That is a 0-GPU diagnostic.
- S1c comes after it; its expected value depends on what the shift is.

### A3R (2026-10-16 ~05:30 Berlin, Master Mind; supersedes the "A3 NOT RUN" note above; written BEFORE any 400-update unit exists): budget or city prior?
**Why "not run" was wrong.** The banked S1 held-out vs navtest comparison it relied on scores different POPULATIONS (R38; `raw/2026-10-16-aux-shift-locate/`). The question stays open: on the SAME 1,058 tokens, S1's heads read **0.501** (mean of 0.496 / 0.521 / 0.486; trained on navtrain, 65.5 % LV, 400 updates × 64 frames). The probe's navtest-cross-fold head reads 0.635 at 2,000 updates.

**Arm U400.**
- Arm V, seed 0, the same 5 folds (`folds.json` sha256 `20dec343…`), and the same head, optimizer, warm-up 50 and cosine.
- **`n_updates` = 400 at batch 8**, a strictly smaller frame budget than S1 (3,200 vs 25,600 draws).
- It is read by `a2_read.py --updates-a2 400` (H = pooled held-out IoU [0, 40)). A2's U1-U4 rows are NOT used. The units go to a separate work dir.

**Committed outcomes (on H400, unrounded; the thresholds are the shift-locate study's, adopted before any 400 unit exists):**

| row | condition | reading | consequence |
|---|---|---|---|
| **R-BUDGET** | H400 ≤ 0.53 | at S1's update count, even the test-like training mix reads near S1 ⇒ S1 is **update-budget-limited** | S1c = more updates, natural sampling |
| **R-PRIOR** | H400 ≥ 0.605 | with 8× FEWER frames, but a navtest-like city / failure mix, it reads near the 2,000-update probe ⇒ S1's **training city prior** (65 % LV) limits it on this set | S1c with city-balanced (and hard-example) sampling, then more updates |
| **R-BOTH** | 0.53 < H400 < 0.605 | both contribute | S1c with both, each as its own registered pair |

- **Reported, never gating:** the per-city H400 (BOS / LV / PIT / SG), I400, and the curve 400 / 2,000 / 10,000.
- **The caveat, stated now:** U400 trains on navtest folds, which share the test set's composition. That is the point of the contrast, but it means R-PRIOR is a statement about training-mix sensitivity, NOT a deployable recipe (navtest labels are never training data for a deployed head).
