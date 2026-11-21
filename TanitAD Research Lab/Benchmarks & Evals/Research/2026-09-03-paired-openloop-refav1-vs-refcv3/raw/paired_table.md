# PAIRED OPEN-LOOP COMPARISON — refav1 vs refcv3

`paired_openloop.py` · generated `20260903T204638Z` · **0 GPU** (it reads banked dumps only)

## ⛔⛔ LINE ONE: **THE READ IS VOID, NOT NEGATIVE.**

* refav1:cl is a CONSTANT-VELOCITY plan on 100.0% of the shared windows — the instrument saw the trivial baseline, not the model. The read is VOID, not negative (D-REFAV1-PAIRED-READ-VOID).

A VOID read means the instrument saw a **degenerate arm** — it measured the trivial baseline, not the model. Every number below is banked for audit and ⛔ **must not be read as "model X beats model Y"**.

⛔ **`refav1:cl` IS the floor.** It is bit-identical to `shared:ha0` on 120/120 shared windows, so its margin is exactly `0.0000` on every row below. This is not "loses to the floor" — the model's emitted trajectory and the no-information baseline are the same array.

## What this compares, and the vocabulary it uses

⛔ **Every arm here is OPEN LOOP** (PI ruling 2026-09-02). The model is not controlling the vehicle; its trajectory does not affect the ego data it is next fed. That is true of refav1's `cl` — a predictor consuming its own planner's actions — exactly as it is of refcv3's `os`. **The words "closed loop" do not apply to any number in this document**; a closed-loop read needs AlpaSim or a real vehicle and does not exist in this programme.

| | A | B |
|---|---|---|
| model | `refav1` | `refcv3` |
| arm | `cl` | `os` |
| tier | `T1` | `T1` |
| checkpoint | `C:\Users\Admin\refav1_eval_slice\ckpt_ep2\ckpt.pt` | `C:/Users/Admin/run_refcv3_ol/ckpt/ckpt_30000.pt` |
| step | `1000` | `30000` |
| dump grid | dt 0.2 s, K 10 | dt 0.5 s, K 4 |

**The asymmetry, stated plainly.** `refcv3` is a **supervised ONE-SHOT anchor model** — 128 anchors x 8 slots, no action input, no rollout, no per-step decode — consuming vision + a v7.2 nav token + ONE ego scalar `v0`. `refav1` is a **latent world model + iCEM planner** that rolls out and consumes a 3-channel action including speed. They are not the same kind of system, which is why the admissible statistic is each arm's **margin over the same trivial floor**, paired on the same windows — never the two arms as levels (`D-HF-COMPARABILITY`).

## The intersection, reported honestly

* **n = 120 shared windows over 20 episodes.** Key: `(clip_id, RAW 10 Hz frame index of the window origin t0)`.
* A carried 140 windows over 20 clips (20 dropped); B carried 3419 over 20 (3299 dropped).
* **Common instants: [1.0, 2.0] s** (raw-frame offsets [10, 20]). A dropped slots [2, 4, 6, 8, 12, 14, 16, 18], B dropped [5, 15]. The grids are matched in **integer raw frames**; there is no resampling and no interpolation.
* The bootstrap resamples **episodes**, not windows — 20 is the sample size that matters.

## The controls that had to read a known value

| control | expected | measured | pass |
|---|---|---|---|
| window key: GT identity across two independent pipelines | `0.0` exactly | `0.000e+00` m | ✅ |
| window key: `v0` identity | `0.0` exactly | `0.000e+00` m/s | ✅ |
| C1 (A) floor reads the **no-information value** `x=v0·t, y=0` | `0.0` exactly | `dx 7.629e-06` m, `|y| 0.000e+00` m | ✅ |
| C1 (B) floor reads the **no-information value** `x=v0·t, y=0` | `0.0` exactly | `dx 1.526e-05` m, `|y| 0.000e+00` m | ✅ |
| C2 the floor is bit-comparable across the two architectures | `0.0` exactly | `1.526e-05` m | ✅ |
| C3 the DERIVED floor against the DUMPED one | `0.0` | `1.526e-05` m | ✅ |
| declared labels `lat_label` agree where BOTH sides label | identical on all 40 | identical (A labels 40, B labels 40 of 120) | ✅ |
| declared labels `lon_label` agree where BOTH sides label | identical on all 40 | identical (A labels 40, B labels 40 of 120) | ✅ |
| declared labels `route_label` agree where BOTH sides label | identical on all 35 | DISAGREE (A labels 40, B labels 105 of 120) | ⛔ |

**Floor provenance.** A: DERIVED from the dumped `v0` through the programme's own integrator (refav1_arm.hold_v0_controls -> paths_from_controls -> refa_v1_plan.unicycle_paths) at this dump's own dt=0.2s, K=10, then index-selected. This dump predates the `ha0` arm; `ha0` is the ONE arm recoverable without the model, because it consumes nothing but v0.  
B: DUMPED — the `ha0` arm written by taniteval/tools/refcv3_arm.py

⚠️ ONE array, side A's, used for BOTH margins — licensed by control C2 (max|A-B| = 1.526e-05 m). ⚠️ THE ALGEBRAIC CONSEQUENCE, STATED RATHER THAN HIDDEN: with an identical floor, (B_arm - floor) - (A_arm - floor) equals (B_arm - A_arm) exactly. The margin framing is still the one reported, because it is what stays interpretable when the floor is NOT identical and it is what D-HF-COMPARABILITY binds — but a reader is owed the identity.

## Shape before metrics — the profiles that decide whether the read is VOID

| arm | straight | const speed | **CONSTANT-VELOCITY** | identical to |
|---|---|---|---|---|
| `refav1:cl` | 1.0000 | 1.0000 | **1.0000** | `refav1:cl_navshuf` 105/120, `refav1:ha` 1/120, `refcv3:ha` 1/120, `shared:ha0` 120/120 |
| `refav1:cl_navshuf` | 1.0000 | 0.8750 | **0.8750** | `refav1:cl` 105/120, `refav1:ha` 1/120, `refcv3:ha` 1/120, `shared:ha0` 105/120 |
| `refav1:ha` | 0.0250 | 0.0083 | **0.0083** | `refav1:cl` 1/120, `refav1:cl_navshuf` 1/120, `refcv3:ha` 1/120, `shared:ha0` 1/120 |
| `refcv3:os` | 0.0000 | 0.0000 | **0.0000** | `refcv3:os_navshuf` 69/120 |
| `refcv3:os_navzero` | 0.0000 | 0.0000 | **0.0000** | — |
| `refcv3:os_navshuf` | 0.0000 | 0.0000 | **0.0000** | `refcv3:os` 69/120 |
| `refcv3:ha` | 0.0250 | 0.0083 | **0.0083** | `refav1:cl` 1/120, `refav1:cl_navshuf` 1/120, `refav1:ha` 1/120, `shared:ha0` 1/120 |
| `shared:ha0` | 1.0000 | 1.0000 | **1.0000** | `refav1:cl` 120/120, `refav1:cl_navshuf` 105/120, `refav1:ha` 1/120, `refcv3:ha` 1/120 |

* selection profile `refav1`: ABSENT — refav1's sidecar carries no `sel_idx` (it is not an anchor model)
* selection profile `refcv3`: **24 distinct anchors**, modal #44 at 0.2083, entropy 2.6666, agrees with the GT-nearest oracle on 0.6167

## The four families, separately — never pooled

`A-floor` and `B-floor` are each arm's **margin over `ha0`** (paired, episode-cluster bootstrap). `(B-f)-(A-f)` is the **difference of margins** — the only admissible cross-model statistic. For error metrics NEGATIVE is better; for the accuracy rows (`*_correct`) POSITIVE is better.

### ADE — family verdict: **VOID — a degenerate arm entered it**

| metric | `refav1:cl` | `refcv3:os` | `shared:ha0` | A−floor | B−floor | (B−f)−(A−f) | 95% CI | separated |
|---|---|---|---|---|---|---|---|---|
| `ade_m` | 0.8635 | 0.6590 | 0.8635 | 0.0000 | -0.2045* | **-0.2045** | [-0.3997, -0.0178] | **YES** |
| `fde_m` | 1.3633 | 1.0163 | 1.3633 | 0.0000 | -0.3470* | **-0.3470** | [-0.6606, -0.0447] | **YES** |

### LONGITUDINAL — family verdict: **VOID — a degenerate arm entered it**

| metric | `refav1:cl` | `refcv3:os` | `shared:ha0` | A−floor | B−floor | (B−f)−(A−f) | 95% CI | separated |
|---|---|---|---|---|---|---|---|---|
| `LON_speed_mae_mps` | 0.4688 | 0.4733 | 0.4688 | 0.0000 | 0.0044 | **0.0044** | [-0.1281, 0.1265] | no |
| `LON_along_mae_m` | 0.5741 | 0.5903 | 0.5741 | 0.0000 | 0.0162 | **0.0162** | [-0.1342, 0.1651] | no |
| `LON_accel_mae_mps2` | 0.4350 | 0.4842 | 0.4350 | 0.0000 | 0.0492 | **0.0492** | [-0.0928, 0.1718] | no |

### LATERAL — family verdict: **VOID — a degenerate arm entered it**

| metric | `refav1:cl` | `refcv3:os` | `shared:ha0` | A−floor | B−floor | (B−f)−(A−f) | 95% CI | separated |
|---|---|---|---|---|---|---|---|---|
| `LAT_cross_mae_m` | 0.4459 | 0.1650 | 0.4459 | 0.0000 | -0.2808* | **-0.2808** | [-0.4276, -0.1450] | **YES** |
| `LAT_heading_mae_deg` | 2.1379 | 0.7442 | 2.1379 | 0.0000 | -1.3937* | **-1.3937** | [-2.2889, -0.6004] | **YES** |
| `LAT_yaw_rate_mae_radps` | 0.0381 | 0.0727 | 0.0381 | 0.0000 | 0.0346 | **0.0346** | [-0.0243, 0.1195] | no |

### TACTICAL — family verdict: **VOID — a degenerate arm entered it**

| metric | `refav1:cl` | `refcv3:os` | `shared:ha0` | A−floor | B−floor | (B−f)−(A−f) | 95% CI | separated |
|---|---|---|---|---|---|---|---|---|
| `TAC_traj_lat_correct` | 0.9417 | 0.9917 | 0.9417 | 0.0000 | 0.0500 | **0.0500** | [0.0000, 0.1083] | no |
| `TAC_traj_lon_correct` | 0.8417 | 0.7500 | 0.8417 | 0.0000 | -0.0917 | **-0.0917** | [-0.2000, 0.0083] | no |
| `TAC_declared_lat_correct` | 0.6750 | 0.7750 | 0.6500 | 0.0250 | 0.1250 | **0.1000** | [-0.1500, 0.3500] | no |
| `TAC_declared_lat_correct_navshuf` | 0.5750 | 0.7250 | 0.6500 | -0.0750 | 0.0750 | **0.1500** | [0.0000, 0.3250] | no |
| `TAC_declared_lat_correct_navzero` | 0.6500 | 0.7000 | 0.6500 | 0.0000 | 0.0500 | **0.0500** | [-0.1000, 0.2250] | no |
| `TAC_declared_lon_correct` | 0.5500 | 0.5250 | 0.4500 | 0.1000 | 0.0750 | **-0.0250** | [-0.3000, 0.2500] | no |
| `TAC_declared_lon_correct_navshuf` | 0.4250 | 0.5000 | 0.4500 | -0.0250 | 0.0500 | **0.0750** | [-0.2000, 0.3500] | no |
| `TAC_declared_lon_correct_navzero` | 0.4500 | 0.4750 | 0.4500 | 0.0000 | 0.0250 | **0.0250** | [-0.2500, 0.2750] | no |

### STRATEGIC — family verdict: **REFUSED**

⛔ **REFUSED** (n = 120 shared windows over 20 episodes). no metric in this family is computable on the shared windows. `route_label`: ⛔ this is NOT a coverage difference — on these windows the two tools assign DIFFERENT labels to the SAME (clip, RAW frame). Scoring both heads against 'the label' would score two different questions, so the row is REFUSED. ESCALATION, not a caveat: one of the two derivations is wrong, or they are two different quantities sharing a name.

*n/a — inputs missing (WORK ITEM, not a pass)*

`*` on a margin means that arm's own paired interval against the floor excludes zero.

**The floor for the two decision families is not `ha0`.** `ha0` is a trajectory and carries no decision head, so a margin against it is undefined for a categorical row. The majority rate is the no-information value for a classifier and — like `ha0` — it is shared across the two models by construction, because it is a property of the LABELS alone. Per key: `lat_label` majority class 0 at 0.6500 over n=40; `lon_label` majority class 1 at 0.4500 over n=40.

### The nav-echo controls (STRATEGIC and declared TACTICAL are inadmissible without them)

⛔ a route/tactical accuracy under TRUE nav is an ECHO INDEX, not skill — nav is an INPUT. Quote the shuffled and zero conditionings beside it, or the claim is inadmissible (BACKLOG R39). `nav_shuffled` withholds the PAIRING (the marginal is preserved exactly, so the model still sees a plausible token everywhere); `nav_zero` withholds the SIGNAL and is the DEPLOYMENT condition. A shuffle cannot stand in for a zero.

| head / control | side | delta | 95% CI | separated |
|---|---|---|---|---|
| `TAC_declared_lat:true_minus_navshuffled` | `refav1:cl` | 0.1000 | [-0.0500, 0.2750] | no |
| `TAC_declared_lat:true_minus_navshuffled` | `refcv3:os` | 0.0500 | [0.0000, 0.1500] | no |
| `TAC_declared_lat:true_minus_navshuffled` | `cross_B_minus_A` | -0.0500 | [-0.2500, 0.1500] | no |
| `TAC_declared_lat:true_minus_navzero` | `refav1:cl` | 0.0250 | [-0.1500, 0.2000] | no |
| `TAC_declared_lat:true_minus_navzero` | `refcv3:os` | 0.0750 | [0.0000, 0.1500] | no |
| `TAC_declared_lat:true_minus_navzero` | `cross_B_minus_A` | 0.0500 | [-0.1500, 0.2500] | no |
| `TAC_declared_lon:true_minus_navshuffled` | `refav1:cl` | 0.1250 | [-0.0500, 0.3000] | no |
| `TAC_declared_lon:true_minus_navshuffled` | `refcv3:os` | 0.0250 | [-0.0500, 0.1000] | no |
| `TAC_declared_lon:true_minus_navshuffled` | `cross_B_minus_A` | -0.1000 | [-0.2750, 0.0750] | no |
| `TAC_declared_lon:true_minus_navzero` | `refav1:cl` | 0.1000 | [-0.1000, 0.3000] | no |
| `TAC_declared_lon:true_minus_navzero` | `refcv3:os` | 0.0500 | [-0.0500, 0.1500] | no |
| `TAC_declared_lon:true_minus_navzero` | `cross_B_minus_A` | -0.0500 | [-0.2500, 0.1500] | no |

### ⭐ Why `ha` is not the shared floor — measured, not argued

`ha` — 'hold the last observed action' — carries the SAME NAME on both sides and is NOT the same arm. It differs in (i) the ACTION-UNIT convention (`action_units` above: channel 0 of a recorded v2ep action is a road-wheel STEER angle, and a dump that integrates it as a curvature over-rotates by ~L = 2.9x) and (ii) the HOLD RULE and the tick it is differenced on. ⛔ The difference below is NOT attributable to either cause alone — it is the combined size of both, and that is exactly the point: a same-named control is not a shared floor. `ha0` is, because it is EXACTLY ZERO in either unit and depends on nothing but v0.

| metric | `(refcv3:ha - shared:ha0) - (refav1:ha - shared:ha0)` | 95% CI | separated |
|---|---|---|---|
| `ade_m` | **-0.6404** | [-0.9856, -0.3355] | **YES** |
| `fde_m` | **-1.0630** | [-1.6341, -0.5557] | **YES** |
| `LON_speed_mae_mps` | **-0.0216** | [-0.0441, -0.0027] | **YES** |
| `LON_along_mae_m` | **-0.2326** | [-0.4173, -0.0795] | **YES** |
| `LON_accel_mae_mps2` | **-0.0023** | [-0.0165, 0.0118] | no |
| `LAT_cross_mae_m` | **-0.5898** | [-0.8954, -0.3305] | **YES** |
| `LAT_heading_mae_deg` | **-3.3219** | [-5.0718, -1.8999] | **YES** |
| `LAT_yaw_rate_mae_radps` | **-0.0640** | [-0.0973, -0.0363] | **YES** |
| `TAC_traj_lat_correct` | **0.1417** | [0.0500, 0.2500] | **YES** |
| `TAC_traj_lon_correct` | **0.0083** | [-0.0167, 0.0333] | no |

**Label-coverage asymmetry between the two tools** (a finding about the label derivations, not about either model):

* the two tools do not label the same windows: A labels 40, B labels 105 of 120. The comparison is therefore SCOPED to the 35 windows BOTH label — an honest restriction, not a silent drop. The asymmetry itself is a finding about the two label derivations, not about either model.

## ⛔ CLOSED LOOP — the other half of the question, and it is NOT in the table above

The PI asked for **open AND closed** loop. Everything above is the **open-loop half**. ⛔ do not present the open-loop table below as the answer to 'open AND closed loop'. It is the open-loop half.

* **Not measured here, and this tool cannot measure it.** this tool reads OPEN-LOOP dumps: banked `[N, K, 2]` trajectories scored against recorded GT. Nothing in a dump can become a closed-loop number — the world never responded.
* **But the harness EXISTS** — `stack/experiments/alpasim-gsplat/closedloop_drive.py` — so this is a NOT-YET-RUN, not a NOT-POSSIBLE. It steps a kinematic bicycle from the model's own (steer, accel) and re-renders the next observation from the resulting ego pose (closedloop_drive.py:518-533) — so the next observation IS a consequence of the model's output, which is the PI's definition of closed loop.
* It has already produced a published panel: flagship v1 vs REF-C base — 9 starts x 50 ticks on Thor, 437 paired windows, paired episode-cluster bootstrap.
* **refcv3:** NOT YET RUN — ESTIMATED ~1.5 engineer-days + ~1 GPU-hour, gated on Thor freeing (INHERITED estimate, not measured here)
* **refav1:** NOT YET RUN — ESTIMATED 1-2 weeks, blocked on the action-unit contract decision (INHERITED estimate, not measured here)

## ⚠️ Are the two runs even on the same corpus?

**UNVERIFIED** — neither run publishes a CHECKED corpus key on both sides, so the TRAIN corpora cannot be shown identical from the configs. A: no parity field / checked=None / key=None. B: False / checked=False / key=None.

* ⭐ **What IS established:** the EVAL side, by the window-key proof: the two dumps' GT and v0 are bit-identical on every shared window, so both arms are scored on the SAME moments of the SAME clips.
* ⛔ **What is NOT:** ⛔ that the two runs TRAINED on the same episodes, and ⛔ that either run's train split is disjoint from these 20 eval clips. Both are properties of the RUNS, not of this adapter, and neither can be read from a dump. The probe that settles them is an episode-id set intersection between each run's train cache and this eval split (REFCV3_ARM.md GATE 3).

`refav1` publishes: `train_cache` = `/home/nvidia/data/refav1-fp8-train`, `train_episodes` = `/home/nvidia/data/physicalai-b1-w120-256x640cyl`, `train_labels_path` = `/home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz`, `train_nav_path` = `/home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz`. Missing: train_labels_manifest (not in the dump manifest); v2_parity block (this trainer publishes none).
`refcv3` publishes: `train_labels_md5` = `0ff902130ce76886b8a925eceed9e3a5`, `train_labels_n_records` = `4572`, `train_labels_path` = `/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz`, `parity` = `False`, `parity_checked` = `False`, `corpus_key` = `None`, `clips_present` = `4572`, `train_cache_dirs` = `['/root/data/train']`, `train_cache` = `['/root/data/train']`.

## ⛔ What this comparison does NOT establish

* ⛔⛔ THIS READ IS VOID. refav1:cl is a CONSTANT-VELOCITY plan on 100.0% of the shared windows — the instrument saw the trivial baseline, not the model. The read is VOID, not negative (D-REFAV1-PAIRED-READ-VOID). A VOID read is not a negative result: the instrument saw a degenerate arm, so it measured the baseline, not the model.
* ⛔ NOTHING ABOUT CLOSED-LOOP DRIVING. Every arm here is OPEN LOOP: neither model controls the vehicle, and the recorded eval ego data arrives regardless of what either predicts. A closed-loop number needs AlpaSim or a real test vehicle and does not exist in this programme.
* ⛔ NOT that one architecture is better than the other. The two systems are different KINDS: refav1 is a latent world model + iCEM planner that rolls out under a 3-channel action including speed; refcv3 is a supervised one-shot 128-anchor x 8-slot trajectory model with NO action input, NO rollout and NO per-step decode, consuming vision + a v7.2 nav token + one ego scalar v0. What is measured is each system's MARGIN OVER THE SAME TRIVIAL FLOOR on the same windows — not a like-for-like of the mechanisms.
* ⛔ NOT a statement about either model's own grid. The table is computed on the COMMON instants only; each dump's finer slots are dropped, so these levels must not be quoted against a single-arm read on a finer grid.
* ⛔ NOT a deployment number for refcv3 while its nav token is fed: the v7.2 nav_cmd is an ORACLE (provenance ego-future) that will not exist at deployment. The deployment-relevant margin is the nav-withheld arm's (`os_navzero - ha0`), reported beside it when that arm is in the dump.
* ⛔ NOT a route-head capability claim from the nav_true row alone. Nav is an INPUT; flagship v1's route head scored 1.0000 by echoing it. Only the nav-shuffled and nav-zero conditionings carry information about skill.
* ⛔ NOT distance-keeping, headway, time-gap or TTC. Those need the lead-block join, which is per-dump and is NOT re-derived here — read them from each side's own single-arm record. Their absence here is a WORK ITEM, not a pass.
* ⛔ NOT a claim that either model would drive: an open-loop trajectory error on recorded data does not bound compounding error under its own control.
* ⛔ NOT evidence of NO effect where a row reads 'not separated'. The episode-cluster bootstrap resamples the 20 episodes in the intersection and nothing else; below 10 episodes such a row is UNDERPOWERED — a statement about the sample, not about the models.

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
python taniteval/tools/paired_openloop.py --a-dump <refav1 FINAL dump> --a-name refav1 --a-arm cl --b-dump <refcv3 FINAL dump> --b-name refcv3 --b-arm os --a-extra cl_navshuf --a-extra ha --b-extra os_navzero --b-extra os_navshuf --b-extra ha --a-run-config <refav1 run config.json> --b-run-config <refcv3 run config.json> --floor ha0 --n-boot 2000 --seed 0 --out taniteval/results/paired-openloop-refav1-vs-refcv3-<UTC>.json --md <RESULT.md>
```
