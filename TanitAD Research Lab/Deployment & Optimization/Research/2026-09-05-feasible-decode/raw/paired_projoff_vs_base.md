# PAIRED OPEN-LOOP COMPARISON — base vs projoff

`paired_openloop.py` · generated `20260905T173247Z` · **0 GPU** (it reads banked dumps only)

## LINE ONE: beats the trivial floor `ha0` on ADE (paired, separated): **base, projoff**

## What this compares, and the vocabulary it uses

⛔ **Every arm here is OPEN LOOP** (PI ruling 2026-09-02). The model is not controlling the vehicle; its trajectory does not affect the ego data it is next fed. That is true of refav1's `cl` — a predictor consuming its own planner's actions — exactly as it is of refcv3's `os`. **The words "closed loop" do not apply to any number in this document**; a closed-loop read needs AlpaSim or a real vehicle and does not exist in this programme.

| | A | B |
|---|---|---|
| model | `base` | `projoff` |
| arm | `os` | `os` |
| tier | `T1` | `T1` |
| checkpoint | `/workspace/experiments/refcv3-b1-v72-30k/ckpt_40284_FINAL.pt` | `/workspace/experiments/refcv3-b1-v72-30k/ckpt_40284_FINAL.pt` |
| step | `40284` | `40284` |
| dump grid | dt 0.5 s, K 4 | dt 0.5 s, K 4 |

**The asymmetry, stated plainly.** `refcv3` is a **supervised ONE-SHOT anchor model** — 128 anchors x 8 slots, no action input, no rollout, no per-step decode — consuming vision + a v7.2 nav token + ONE ego scalar `v0`. `refav1` is a **latent world model + iCEM planner** that rolls out and consumes a 3-channel action including speed. They are not the same kind of system, which is why the admissible statistic is each arm's **margin over the same trivial floor**, paired on the same windows — never the two arms as levels (`D-HF-COMPARABILITY`).

## The intersection, reported honestly

* **n = 4823 shared windows over 141 episodes.** Key: `(clip_id, RAW 10 Hz frame index of the window origin t0)`.
* A carried 4823 windows over 141 clips (0 dropped); B carried 4823 over 141 (0 dropped).
* **Common instants: [0.5, 1.0, 1.5, 2.0] s** (raw-frame offsets [5, 10, 15, 20]). A dropped slots [], B dropped []. The grids are matched in **integer raw frames**; there is no resampling and no interpolation.
* The bootstrap resamples **episodes**, not windows — 141 is the sample size that matters.

## The controls that had to read a known value

| control | expected | measured | pass |
|---|---|---|---|
| window key: GT identity across two independent pipelines | `0.0` exactly | `0.000e+00` m | ✅ |
| window key: `v0` identity | `0.0` exactly | `0.000e+00` m/s | ✅ |
| C1 (A) floor reads the **no-information value** `x=v0·t, y=0` | `0.0` exactly | `dx 2.289e-05` m, `|y| 0.000e+00` m | ✅ |
| C1 (B) floor reads the **no-information value** `x=v0·t, y=0` | `0.0` exactly | `dx 2.289e-05` m, `|y| 0.000e+00` m | ✅ |
| C2 the floor is bit-comparable across the two architectures | `0.0` exactly | `0.000e+00` m | ✅ |
| C3 the DERIVED floor against the DUMPED one | `0.0` | `2.289e-05` m | ✅ |
| declared labels `lat_label` agree where BOTH sides label | identical on all 1157 | identical (A labels 1157, B labels 1157 of 4823) | ✅ |
| declared labels `lon_label` agree where BOTH sides label | identical on all 1157 | identical (A labels 1157, B labels 1157 of 4823) | ✅ |
| declared labels `route_label` agree where BOTH sides label | identical on all 3622 | identical (A labels 3622, B labels 3622 of 4823) | ✅ |

**Floor provenance.** A: DUMPED — the `ha0` arm written by taniteval/tools/refcv3_arm.py  
B: DUMPED — the `ha0` arm written by taniteval/tools/refcv3_arm.py

⚠️ ONE array, side A's, used for BOTH margins — licensed by control C2 (max|A-B| = 0.000e+00 m). ⚠️ THE ALGEBRAIC CONSEQUENCE, STATED RATHER THAN HIDDEN: with an identical floor, (B_arm - floor) - (A_arm - floor) equals (B_arm - A_arm) exactly. The margin framing is still the one reported, because it is what stays interpretable when the floor is NOT identical and it is what D-HF-COMPARABILITY binds — but a reader is owed the identity.

## Shape before metrics — the profiles that decide whether the read is VOID

| arm | straight | const speed | **CONSTANT-VELOCITY** | identical to |
|---|---|---|---|---|
| `base:os` | 0.0000 | 0.0000 | **0.0000** | `projoff:os` 4823/4823 |
| `projoff:os` | 0.0000 | 0.0000 | **0.0000** | `base:os` 4823/4823 |
| `shared:ha0` | 1.0000 | 1.0000 | **1.0000** | — |

* selection profile `base`: **50 distinct anchors**, modal #57 at 0.1482, entropy 2.8425, agrees with the GT-nearest oracle on 0.5652
* selection profile `projoff`: **50 distinct anchors**, modal #57 at 0.1482, entropy 2.8425, agrees with the GT-nearest oracle on 0.5652

## The four families, separately — never pooled

`A-floor` and `B-floor` are each arm's **margin over `ha0`** (paired, episode-cluster bootstrap). `(B-f)-(A-f)` is the **difference of margins** — the only admissible cross-model statistic. For error metrics NEGATIVE is better; for the accuracy rows (`*_correct`) POSITIVE is better.

### ADE — family verdict: **no separation**

| metric | `base:os` | `projoff:os` | `shared:ha0` | A−floor | B−floor | (B−f)−(A−f) | 95% CI | separated |
|---|---|---|---|---|---|---|---|---|
| `ade_m` | 0.4419 | 0.4419 | 0.6723 | -0.2304* | -0.2304* | **0.0000** | [0.0000, 0.0000] | no |
| `fde_m` | 0.9288 | 0.9288 | 1.4029 | -0.4741* | -0.4741* | **0.0000** | [0.0000, 0.0000] | no |

### LONGITUDINAL — family verdict: **no separation**

| metric | `base:os` | `projoff:os` | `shared:ha0` | A−floor | B−floor | (B−f)−(A−f) | 95% CI | separated |
|---|---|---|---|---|---|---|---|---|
| `LON_speed_mae_mps` | 0.4516 | 0.4516 | 0.4880 | -0.0364* | -0.0364* | **0.0000** | [0.0000, 0.0000] | no |
| `LON_along_mae_m` | 0.4030 | 0.4030 | 0.4705 | -0.0674* | -0.0674* | **0.0000** | [0.0000, 0.0000] | no |
| `LON_accel_mae_mps2` | 0.6806 | 0.6806 | 0.4786 | 0.2020* | 0.2020* | **0.0000** | [0.0000, 0.0000] | no |

### LATERAL — family verdict: **no separation**

| metric | `base:os` | `projoff:os` | `shared:ha0` | A−floor | B−floor | (B−f)−(A−f) | 95% CI | separated |
|---|---|---|---|---|---|---|---|---|
| `LAT_cross_mae_m` | 0.1084 | 0.1084 | 0.3132 | -0.2048* | -0.2048* | **0.0000** | [0.0000, 0.0000] | no |
| `LAT_heading_mae_deg` | 1.3591 | 1.3591 | 2.8715 | -1.3741* | -1.3741* | **0.0000** | [0.0000, 0.0000] | no |
| `LAT_yaw_rate_mae_radps` | 0.2176 | 0.2176 | 0.0476 | 0.1700* | 0.1700* | **0.0000** | [0.0000, 0.0000] | no |

### TACTICAL — family verdict: **no separation**

| metric | `base:os` | `projoff:os` | `shared:ha0` | A−floor | B−floor | (B−f)−(A−f) | 95% CI | separated |
|---|---|---|---|---|---|---|---|---|
| `TAC_traj_lat_correct` | 0.9540 | 0.9540 | 0.8659 | 0.0881* | 0.0881* | **0.0000** | [0.0000, 0.0000] | no |
| `TAC_traj_lon_correct` | 0.7477 | 0.7477 | 0.7576 | -0.0100 | -0.0100 | **0.0000** | [0.0000, 0.0000] | no |
| `TAC_declared_lat_correct` | 0.7105 | 0.7105 | 0.6759 | 0.0346 | 0.0346 | **0.0000** | [0.0000, 0.0000] | no |
| `TAC_declared_lat_correct_navshuf` | 0.6863 | 0.6863 | 0.6759 | 0.0104 | 0.0104 | **0.0000** | [0.0000, 0.0000] | no |
| `TAC_declared_lat_correct_navzero` | 0.6811 | 0.6811 | 0.6759 | 0.0052 | 0.0052 | **0.0000** | [0.0000, 0.0000] | no |
| `TAC_declared_lon_correct` | 0.5134 | 0.5134 | 0.2982 | 0.2152* | 0.2152* | **0.0000** | [0.0000, 0.0000] | no |
| `TAC_declared_lon_correct_navshuf` | 0.4987 | 0.4987 | 0.2982 | 0.2005* | 0.2005* | **0.0000** | [0.0000, 0.0000] | no |
| `TAC_declared_lon_correct_navzero` | 0.4952 | 0.4952 | 0.2982 | 0.1971* | 0.1971* | **0.0000** | [0.0000, 0.0000] | no |

### STRATEGIC — family verdict: **no separation**

| metric | `base:os` | `projoff:os` | `shared:ha0` | A−floor | B−floor | (B−f)−(A−f) | 95% CI | separated |
|---|---|---|---|---|---|---|---|---|
| `STR_route_correct` | 0.7667 | 0.7667 | 0.6742 | 0.0925* | 0.0925* | **0.0000** | [0.0000, 0.0000] | no |
| `STR_route_correct_navshuf` | 0.7667 | 0.7667 | 0.6742 | 0.0925* | 0.0925* | **0.0000** | [0.0000, 0.0000] | no |
| `STR_route_correct_navzero` | 0.7667 | 0.7667 | 0.6742 | 0.0925* | 0.0925* | **0.0000** | [0.0000, 0.0000] | no |

`*` on a margin means that arm's own paired interval against the floor excludes zero.

**The floor for the two decision families is not `ha0`.** `ha0` is a trajectory and carries no decision head, so a margin against it is undefined for a categorical row. The majority rate is the no-information value for a classifier and — like `ha0` — it is shared across the two models by construction, because it is a property of the LABELS alone. Per key: `lat_label` majority class 0 at 0.6759 over n=1157; `lon_label` majority class 1 at 0.2982 over n=1157; `route_label` majority class 1 at 0.6742 over n=3622.

### The nav-echo controls (STRATEGIC and declared TACTICAL are inadmissible without them)

⛔ a route/tactical accuracy under TRUE nav is an ECHO INDEX, not skill — nav is an INPUT. Quote the shuffled and zero conditionings beside it, or the claim is inadmissible (BACKLOG R39). `nav_shuffled` withholds the PAIRING (the marginal is preserved exactly, so the model still sees a plausible token everywhere); `nav_zero` withholds the SIGNAL and is the DEPLOYMENT condition. A shuffle cannot stand in for a zero.

| head / control | side | delta | 95% CI | separated |
|---|---|---|---|---|
| `TAC_declared_lat:true_minus_navshuffled` | `base:os` | 0.0242 | [-0.0017, 0.0544] | no |
| `TAC_declared_lat:true_minus_navshuffled` | `projoff:os` | 0.0242 | [-0.0017, 0.0544] | no |
| `TAC_declared_lat:true_minus_navshuffled` | `cross_B_minus_A` | 0.0000 | [0.0000, 0.0000] | no |
| `TAC_declared_lat:true_minus_navzero` | `base:os` | 0.0294 | [-0.0095, 0.0700] | no |
| `TAC_declared_lat:true_minus_navzero` | `projoff:os` | 0.0294 | [-0.0095, 0.0700] | no |
| `TAC_declared_lat:true_minus_navzero` | `cross_B_minus_A` | 0.0000 | [0.0000, 0.0000] | no |
| `TAC_declared_lon:true_minus_navshuffled` | `base:os` | 0.0147 | [-0.0052, 0.0354] | no |
| `TAC_declared_lon:true_minus_navshuffled` | `projoff:os` | 0.0147 | [-0.0052, 0.0354] | no |
| `TAC_declared_lon:true_minus_navshuffled` | `cross_B_minus_A` | 0.0000 | [0.0000, 0.0000] | no |
| `TAC_declared_lon:true_minus_navzero` | `base:os` | 0.0182 | [-0.0112, 0.0464] | no |
| `TAC_declared_lon:true_minus_navzero` | `projoff:os` | 0.0182 | [-0.0112, 0.0464] | no |
| `TAC_declared_lon:true_minus_navzero` | `cross_B_minus_A` | 0.0000 | [0.0000, 0.0000] | no |
| `STR_route:true_minus_navshuffled` | `base:os` | 0.0000 | [0.0000, 0.0000] | no |
| `STR_route:true_minus_navshuffled` | `projoff:os` | 0.0000 | [0.0000, 0.0000] | no |
| `STR_route:true_minus_navshuffled` | `cross_B_minus_A` | 0.0000 | [0.0000, 0.0000] | no |
| `STR_route:true_minus_navzero` | `base:os` | 0.0000 | [0.0000, 0.0000] | no |
| `STR_route:true_minus_navzero` | `projoff:os` | 0.0000 | [0.0000, 0.0000] | no |
| `STR_route:true_minus_navzero` | `cross_B_minus_A` | 0.0000 | [0.0000, 0.0000] | no |

## ⛔ CLOSED LOOP — the other half of the question, and it is NOT in the table above

The PI asked for **open AND closed** loop. Everything above is the **open-loop half**. ⛔ do not present the open-loop table below as the answer to 'open AND closed loop'. It is the open-loop half.

* **Not measured here, and this tool cannot measure it.** this tool reads OPEN-LOOP dumps: banked `[N, K, 2]` trajectories scored against recorded GT. Nothing in a dump can become a closed-loop number — the world never responded.
* **But the harness EXISTS** — `stack/experiments/alpasim-gsplat/closedloop_drive.py` — so this is a NOT-YET-RUN, not a NOT-POSSIBLE. It steps a kinematic bicycle from the model's own (steer, accel) and re-renders the next observation from the resulting ego pose (closedloop_drive.py:518-533) — so the next observation IS a consequence of the model's output, which is the PI's definition of closed loop.
* It has already produced a published panel: flagship v1 vs REF-C base — 9 starts x 50 ticks on Thor, 437 paired windows, paired episode-cluster bootstrap.
* **refcv3:** NOT YET RUN — ESTIMATED ~1.5 engineer-days + ~1 GPU-hour, gated on Thor freeing (INHERITED estimate, not measured here)
* **refav1:** NOT YET RUN — ESTIMATED 1-2 weeks, blocked on the action-unit contract decision (INHERITED estimate, not measured here)

## ⚠️ Are the two runs even on the same corpus?

**UNVERIFIED** — neither run publishes a CHECKED corpus key on both sides, so the TRAIN corpora cannot be shown identical from the configs. A: no parity field / checked=None / key=None. B: no parity field / checked=None / key=None.

* ⭐ **What IS established:** the EVAL side, by the window-key proof: the two dumps' GT and v0 are bit-identical on every shared window, so both arms are scored on the SAME moments of the SAME clips.
* ⛔ **What is NOT:** ⛔ that the two runs TRAINED on the same episodes, and ⛔ that either run's train split is disjoint from these 20 eval clips. Both are properties of the RUNS, not of this adapter, and neither can be read from a dump. The probe that settles them is an episode-id set intersection between each run's train cache and this eval split (REFCV3_ARM.md GATE 3).

`base` publishes: `train_labels_md5` = `0ff902130ce76886b8a925eceed9e3a5`, `train_labels_n_records` = `4572`, `train_labels_path` = `/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz`. Missing: the run's own config.json was not supplied (--a-run-config / --b-run-config).
`projoff` publishes: `train_labels_md5` = `0ff902130ce76886b8a925eceed9e3a5`, `train_labels_n_records` = `4572`, `train_labels_path` = `/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz`. Missing: the run's own config.json was not supplied (--a-run-config / --b-run-config).

## ⛔ What this comparison does NOT establish

* ⛔ NOTHING ABOUT CLOSED-LOOP DRIVING. Every arm here is OPEN LOOP: neither model controls the vehicle, and the recorded eval ego data arrives regardless of what either predicts. A closed-loop number needs AlpaSim or a real test vehicle and does not exist in this programme.
* ⛔ NOT that one architecture is better than the other. The two systems are different KINDS: refav1 is a latent world model + iCEM planner that rolls out under a 3-channel action including speed; refcv3 is a supervised one-shot 128-anchor x 8-slot trajectory model with NO action input, NO rollout and NO per-step decode, consuming vision + a v7.2 nav token + one ego scalar v0. What is measured is each system's MARGIN OVER THE SAME TRIVIAL FLOOR on the same windows — not a like-for-like of the mechanisms.
* ⛔ NOT a statement about either model's own grid. The table is computed on the COMMON instants only; each dump's finer slots are dropped, so these levels must not be quoted against a single-arm read on a finer grid.
* ⛔ NOT a deployment number for refcv3 while its nav token is fed: the v7.2 nav_cmd is an ORACLE (provenance ego-future) that will not exist at deployment. The deployment-relevant margin is the nav-withheld arm's (`os_navzero - ha0`), reported beside it when that arm is in the dump.
* ⛔ NOT a route-head capability claim from the nav_true row alone. Nav is an INPUT; flagship v1's route head scored 1.0000 by echoing it. Only the nav-shuffled and nav-zero conditionings carry information about skill.
* ⛔ NOT distance-keeping, headway, time-gap or TTC. Those need the lead-block join, which is per-dump and is NOT re-derived here — read them from each side's own single-arm record. Their absence here is a WORK ITEM, not a pass.
* ⛔ NOT a claim that either model would drive: an open-loop trajectory error on recorded data does not bound compounding error under its own control.
* ⛔ NOT evidence of NO effect where a row reads 'not separated'. The episode-cluster bootstrap resamples the 141 episodes in the intersection and nothing else; below 10 episodes such a row is UNDERPOWERED — a statement about the sample, not about the models.

## Estimator

* **point**: FULL-SET pooled mean over the shared windows
* **interval**: taniteval.ci.episode_cluster_bootstrap (single arm) / paired_episode_cluster_bootstrap (every difference)
* **cluster**: the CLIP — windows inside one clip are strongly dependent
* **n_boot**: 2000
* **seed**: 0
* **forbidden**: overlapping_holdout_se appears nowhere; it biases the POINT ESTIMATE bidirectionally (-6.67% to +11.69%), up to a sign flip
* **reading_rule**: ⛔ two overlapping single-arm CIs are NOT a null result and two disjoint ones are NOT the paired test. Only the paired difference interval decides.

## Re-fire against the FINAL checkpoints

```
python taniteval/tools/paired_openloop.py --a-dump <refav1 FINAL dump> --a-name base --a-arm os --b-dump <refcv3 FINAL dump> --b-name projoff --b-arm os  --a-run-config <refav1 run config.json> --b-run-config <refcv3 run config.json> --floor ha0 --n-boot 2000 --seed 0 --out taniteval/results/paired-openloop-base-vs-projoff-<UTC>.json --md <RESULT.md>
```
