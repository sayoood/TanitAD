<title>What DriveZero actually released — the physics and the scoring, not the traffic and not the optimizer; and the 5.7 M teacher verified to the parameter</title>

# DriveRL released-artifact teardown

`Research Lab · 2026-09-20 (LAB-RUN-017) · companion to RESULT.md · ⭐ PI authorised use of the repo, inference code and checkpoints on 2026-09-20`
⚠️ **Authorisation noted, and one fact recorded alongside it:** no LICENSE file exists anywhere in the repository outside the vendored `nuplan-devkit/LICENSE.txt` (that one is nuPlan's, not Xiaomi's). The PI's authorisation governs **our** use; it does not change the upstream licence status, which still matters if anything is ever redistributed or shipped. Recorded once, then proceeded.

**Method:** shallow clone (`--depth 1 --no-recurse-submodules`), 110 MB, **63 Python files** under `DriveRL/src` + `scripts`. All numbers below are **MEASURED from the artifact**, not read from the paper.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **R1** | ⭐⭐⭐ **The teacher's 5.7 M is confirmed to the parameter.** `checkpoint_2400.pt` holds **63 tensors, 5,702,413 parameters** (fp32 ⇒ 22,809,652 B of weights inside a 22,832,846 B file). The paper's "5.7 M" is **exact**. | MEASURED |
| **R2** | ⭐⭐⭐ **The RL environment is published — but it is the TTS rollout engine, not the training simulator.** `env/engine/engine.py`'s own docstring: *"Dynamics and reward evaluation used by DriveRL **test-time scaling**"* / *"Evaluate short model-based rollouts for **TTS candidate selection**"*. ⇒ **both** the README ("inference code and checkpoints") **and** the natural reading ("they published the RL environment") are wrong in opposite directions. | MEASURED |
| **R3** | ⭐⭐⭐ **What ships is nonetheless the hardest half to guess: the exact reward.** Six reward calculators (`NuPlanCollision · GoalReaching · OffRoad · CenterLine · Comfort · NuPlanTTC`), two dynamics models, and **every weight and threshold as a literal** in `release/configs/driverl_teacher.yaml`. The paper says *"six soft driving-quality scores"* and **never names them**. The code does. | MEASURED |
| **R4** | ⭐⭐ **The decomposed critic has exactly 9 channels, and they are named.** Measured head: `value_head.2 → (9, 1024)`. Named in `reward_decomposition.py`: `hard · goal · soft_cross_lane · soft_centerline · soft_curb_clearance · soft_comfort · soft_ttc · soft_overspeed · soft_other`. | MEASURED |
| **R5** | ⛔ **No trainer.** `grep` over all 63 files for `torch.optim`, `.backward()`, `surrogate`, `entropy_coef`, `gae`, `Muon` returns **nothing**. **No IDM, no self-play, no world/scene manager either** (`IDM|self_play|selfplay` ⇒ 0 hits). ⇒ the background-actor behaviour providers and the scene lifecycle — the parts that make 196,608 worlds run — are **not** released. | MEASURED |
| **R6** | ⭐⭐ **But the released core IS batched, which is the reusable part.** `(num_envs, num_agents, …)` tensors run through the engine and every reward calculator. It is a **GPU-vectorised reward + dynamics evaluator**, i.e. the computational substrate of the training environment with the traffic and the optimiser removed. | MEASURED |
| **R7** | ⭐ **`domain_randomization/` is a TRAINING artifact that survived the export.** Its config randomises **reward weights and thresholds** (`collision_reward_weight`, `ttc_weight`, `off_road_weight`, `goal_reaching_threshold`, `comfort_weight`, `curb_clearance_*`, `over_speed_threshold`…), which nobody does at inference. The released YAML pins `enabled: false` and comments *"Deterministic reward values used by the inference/TTS runtime."* ⇒ this is a **trimmed export of the training env**, not a purpose-built inference wrapper. | MEASURED |
| **R8** | ⛔⭐ **The two released checkpoints were trained under DIFFERENT rewards, which the paper never says.** `driverl_teacher.yaml` vs `driverl_selfplay.yaml` differ in reward literals — e.g. **`ttc_weight` 0.5 → 0.0** and **`ttc_time` 2.5 → 4.0**, plus off-road / cross-lane / wrong-way / goal-reaching terms. ⇒ **the self-play result is not a one-variable change over the teacher**, and any comparison between them is confounded. | MEASURED |
| **R9** | ⚠️ **The released checkpoints are self-declared un-evaluated.** `driverl_checkpoints.yaml`: `evaluation_status: artifact_bundled_pending_public_eval` and **`evaluation_tts_enabled: false`**. ⇒ the artifact is **not** certified to reproduce the paper's table, and TTS is **off** by default in the release. | MEASURED |

---

## 1 · The teacher network, measured from the weights

63 tensors, **5,702,413 params**. Parameter budget by submodule:

| submodule | params | share | what it is |
|---|---|---|---|
| `value_head` | 1,058,825 | **18.6 %** | 1024→1024→**9** — the decomposed critic (R4) |
| `final_mlp` | 1,053,700 | **18.5 %** | 1024→1024→**4** — two Beta distributions × (α, β) |
| `combiner_mlp` | 1,050,624 | **18.4 %** | **1025**→1024 — 4 × 256 fused features **+ 1 scalar** |
| `extra_agent_layers` | 789,760 | 13.8 % | the 2nd agent attention block |
| `agent_ffn` / `lane_attn_ffn` | 525,568 each | 9.2 % each | 256→1024→256 FFNs |
| `agent_self_attention` / `lane_cross_attn` | 263,168 each | 4.6 % each | 4-head, 256-d |
| `kinematics_mlp` | 68,352 | 1.2 % | **9**→256→256 — 9 ego kinematic features |
| `goal_position_mlp` | 65,792 | 1.2 % | 256→256 |
| `agent_mlp` | 17,728 | 0.3 % | **16**→64→256 — 16 features per agent |
| `lane_bound_mlp` | 17,344 | 0.3 % | **10**→64→256 — 10-D map tokens (matches appendix A.2) |
| `default_lane_embedding` | 256 | 0.0 % | learned fallback when no lane is present |

⭐⭐ **The shape of the model is the finding: 55.5 % of the teacher is three 1024-wide MLP heads, and only ~32 % is attention.** The perception-style machinery is nearly free; the policy/value heads dominate. A 5.7 M net is trivially trainable on one GPU — **which confirms that the 96 GPUs buy SIMULATION THROUGHPUT, not model capacity.** That is the single most important structural fact for anyone trying to reproduce this.

**Input widths recovered:** 16 per agent · 10 per map element · 9 ego kinematics · 2 goal points (sinusoidal → shared MLP → mean-pooled).
**Output:** 4 logits = α, β for **longitudinal jerk** and **steering-angle rate**.

---

## 2 · The reward, in full — the part the paper withholds

The paper says only *"six soft driving-quality scores q_{t,k} ∈ [0,1] covering safety, compliance, and comfort"*, multiplied together. The artifact names them and prices them.

**The six soft criteria** (from `reward_decomposition.py`): `cross_lane · centerline · curb_clearance · comfort · ttc · overspeed` — plus `hard` and `goal` and an `other` catch-all ⇒ the **9** critic channels.

**Weights and thresholds** (`release/configs/driverl_teacher.yaml`, deterministic defaults):

| term | value |
|---|---|
| `collision_reward_weight` / `collision_segment_time` | **−1.0** / 0.2 s |
| `off_road_weight` | **−1.0** |
| `ttc_weight` / `ttc_time` | **0.5** / **2.5 s** |
| `cross_lane_weight` / `wrong_way_weight` | 0.4 / 0.2 |
| `goal_reaching_threshold` / `goal_reaching_weight` | **3.0 m** / 0.3 |

**Vehicle and comfort constants** (all literals, all released):

| | |
|---|---|
| wheelbase / rear-axle-to-center | **3.089 m** / 1.461 m |
| max steering angle / rate | 1.0472 rad (= π/3) / **0.8 rad/s** |
| max acceleration | 4.0 m/s² |
| long. jerk range | **[−8.0, +5.0] m/s³** |
| accel range (long / lat) | [−6.5, +1.5] / [−3.0, +3.0] m/s² |
| accel & steering time constants | 0.2 s / 0.05 s |
| nuPlan comfort bounds | lon accel [−4.05, 2.40] · \|lat accel\| 4.89 · \|lon jerk\| 4.13 · \|jerk mag\| 8.37 · \|yaw rate\| 0.95 · \|yaw accel\| 1.93 |
| TTS action grid | `num_jerk_lat_actions` **64** · `num_jerk_long_actions` **24** |

⭐ **The attribution rule for the decomposed critic is also released and is non-obvious.** `split_soft_product_reward` takes the *already-computed multiplicative* reward and splits it across channels **in proportion to each component's "badness" `(1 − score)`**, falling back to an even split when every active score is perfect. That is how a product reward is made additive enough for per-channel GAE — a design detail no paper text would have given us.

⚠️ **A cross-check for our own regime boundary:** they use a **single wheelbase of 3.089 m** for all of nuPlan. We measured that a constant wheelbase carries a > 5 % error on 98.2 % of our clips (`MODEL_REGISTRY §0.1.1`). Their choice is *not* evidence that a constant is fine — nuPlan is a narrower vehicle population than PhysicalAI — but it is worth recording that the field's reference implementation does it.

---

## 3 · The released-vs-withheld boundary

| component | released? | note |
|---|---|---|
| Teacher weights (2 checkpoints) | ✅ | `checkpoint_2400.pt` (paper's 2,400 updates) and `checkpoint_3800.pt` (self-play, **different reward** — R8) |
| Policy + value network code | ✅ | `vanilla_net.py`, `vanilla_net_agent.py` (33.8 KB) |
| **Reward calculators (×6)** | ✅ | collision, goal, off-road, centerline, comfort, TTC |
| **Reward decomposition** | ✅ | the 9 named channels + the badness-weighted split |
| **Dynamics models (×2)** | ✅ | `nuplan_bicycle_model.py`, **`jerk_bicycle_model.py`** |
| Batched engine `(num_envs, num_agents)` | ✅ | but scoped to **TTS rollouts** |
| **Value-guided test-time search** | ✅ | `nuplan/tts.py`, 37.5 KB — ⚠️ `evaluation_tts_enabled: false` in the release |
| Domain randomisation (reward params) | ✅ | a training artifact left in (R7) |
| nuPlan feature builder / planner / controller | ✅ | `feature_builder.py` 44.6 KB, `planner.py`, `controller.py` |
| **PPO trainer** | ⛔ | no optimiser, no backward, no GAE, no Muon |
| **Background-actor providers** (IDM, front-braking, self-play) | ⛔ | 0 hits |
| **World/scene lifecycle** (log seeding, scene-consistent insertion, resets, 2,048 worlds/rank) | ⛔ | not present |
| **DriveZero student** | ⛔ | TODO |
| **DriveVFM backbone + weights** | ⛔ | TODO |

**One sentence:** *they shipped the physics and the scoring, not the traffic and not the optimiser.*

---

## 4 · What this changes for our plan

| plan item | change |
|---|---|
| **Stage 4 (RL teacher)** — was "gated, possibly never" | ⭐⭐ **Materially cheaper.** The reward (the hardest thing to guess, and the thing most RL papers omit) is **free and exact**. What we would have to write is a **batched world manager + IDM** — which is engineering we can scope, not research we have to invent |
| **DZ-5 (measure sim throughput)** — was "after `alpasim_runtime`" | ⭐⭐⭐ **Unblocked NOW and much sharper.** Their engine is already batched over `num_envs`. We can benchmark **their** reward+dynamics core on our RTX 4060 and get a real *scoring-half* sim-steps/GPU-hour number **without writing anything and without waiting for AlpaSim**. That number is the decision input Stage 4 needs |
| **DZ-4 (rate-level action space)** | ⭐⭐ **`jerk_bicycle_model.py` is directly transplantable**, with every constant (jerk range, time constants, caps) already tuned. This drops from "design an action space" to "port a file" |
| **DZ-3 (per-term critic)** | ⭐⭐ upgraded from a sketch to a **specification**: 9 named channels + the badness-weighted product split, verbatim |
| **I-1 / LR15-2** | ⭐ `tts.py` (37.5 KB) is a **reference implementation** of value-guided search with a conservative margin — directly readable against our fan-scoring |
| **DZ-7 (licence)** | ⭐ **PI authorised our use 2026-09-20.** ⚠️ The finding stands on the record: no Xiaomi LICENSE file exists in the repo (the only one is the vendored nuPlan devkit's). Re-check before anything is redistributed or ships in a product |

### New proposed row

**`DZ-9` (Deploy, 0 GPU → hours) — benchmark the released DriveRL engine on our 4060.** Run their batched reward + `nuplan_bicycle_model` over a sweep of `num_envs` and measure **sim-steps per GPU-second**, converted to sim-hours per GPU-hour, against the **DERIVED ~1,430 sim-h/GPU-h** their full stack achieved. **Committed:** this measures the **scoring half only** and must be reported as such — the traffic providers and scene management they did *not* release are pure additional cost, so our number is an **upper bound** on what we could achieve, never an estimate of the whole. ⇒ feeds the Stage-4 go/no-go directly.

`Deliverables: RELEASED_ARTIFACT_TEARDOWN.md · (clone kept in the session scratchpad, NOT added to the repo)`

---

## 5 · ⭐ Can we measure the CHECKPOINT'S QUALITY? (PI question, 2026-09-20)

⛔ **Not with DZ-9. DZ-9 measures THROUGHPUT, not quality** — it times the batched reward + dynamics core and returns sim-steps per GPU-second. A fast engine says nothing about whether the policy drives well. The two are separate measurements with separate requirements, and conflating them is exactly the *true-but-wrong-for-the-reader* class.

### 5.1 What each measurement actually needs

| measurement | needs | have it? | what it tells us |
|---|---|---|---|
| **DZ-9 — engine throughput** | the repo only; synthetic tensors | ✅ **now** | sim-steps/GPU-s ⇒ the Stage-4 go/no-go. **NOT quality** |
| **Preflight — artifact integrity** | repo (+ maps); `run_driverl_nuplan_eval.sh preflight` ⇒ `PREFLIGHT_OK` | ✅ **now** | the checkpoint loads and the config is coherent. **NOT quality** |
| **Headline reproduction** — the six-task CLS suite | nuPlan **v1.1 log DBs** for the 1,651 benchmark tokens + maps | ⛔ **DBs absent** | the only thing that reproduces 95.16 / 89.97 / 94.50 |
| ⭐ **Relative quality — the TTS N-sweep** | **any** nuPlan DB subset, e.g. the **13 GB mini split** | ⛔ PI decision | ⭐⭐ **a real, paired quality measurement — and it is I-1's own question** |

### 5.2 The data requirement is far smaller than assumed — sensors are NOT needed

⭐ `docs/driverl_nuplan_data.md`: *"The release configurations use **box tracks and map features**."* DriveRL is privileged/structured — it consumes agent boxes, vector map and traffic lights. **No camera or LiDAR blobs are required.** That removes the multi-TB sensor pull from the problem entirely, and it is a different (much smaller) requirement from the ~450 GB NAVSIM sensor set that gates backlog row 3.

| component | size | status |
|---|---|---|
| `nuplan-maps-v1.0` (`NUPLAN_MAPS_ROOT`) | **1.4 GB** — all four cities (`sg-one-north`, `us-ma-boston`, `us-nv-las-vegas-strip`, `us-pa-pittsburgh-hazelwood`) | ✅ **ALREADY LOCAL** at `data/nuplan-maps/nuplan-maps-v1.0` |
| nuPlan **mini** split, DB only | **13 GB** (7 h, 64 logs) | ⛔ not held — **PI decision** |
| nuPlan **full** trainval + test, DB only | **~1.8 TB** (1,282 h, 15,910 logs ⇒ ≈ 113 MB/log) | ⛔ not held; **too large for this box** |

`RELAYED` (secondary sources, not read at a primary): the 13 GB / 1.8 TB figures and the log counts. ⚠️ **The per-split size of `splits/test` alone was NOT recovered at this probe** — and it matters, because **4 of the 6 tasks (`test14hard_nr/r`, `test14random_nr/r`) read only `splits/test`**, while `val14_nr/r` read `splits/trainval`. Named as a gap rather than estimated.

⭐ The release anticipates a partial holding: `DRIVERL_EVAL_DB_LINK_ROOT` points at directories of **symlinks to selected DBs**. ⛔ **But D: is exFAT — no symlinks (`WinError 1`)**, so on this box that path needs **copies**, or a different filesystem. A concrete, known trap.

### 5.3 The run is single-GPU-capable

`docs/driverl_repro_eval.md` documents `DRIVERL_EVAL_LOCAL_GPUS=1` with `DRIVERL_EVAL_WORKER_MODE=ray_local` and `DRIVERL_EVAL_PARALLEL_WORKERS=8`. ⇒ **the 4060 is a supported configuration**; this is not a 96-GPU job. Wall-clock for 1,651 scenarios × 15 s at 1 GPU is **unknown and should be measured on a handful of scenarios first**, not assumed.

The **deterministic TTS suite** is documented too: `DRIVERL_EVAL_TTS_ENABLED=1`, `TTS_NUM_CANDIDATES=8`, `TTS_SEED=42`, horizon `5`, `total_return_switch_margin=0.03`, candidate 0 = deterministic argmax. ⇒ **the N-sweep is a supported, documented knob.**

### 5.4 ⛔ Two integrity caveats that make an external check WORTH something

1. `driverl_checkpoints.yaml` self-declares **`evaluation_status: artifact_bundled_pending_public_eval`** ⇒ **nobody has publicly verified that these weights reproduce the paper's table.** If we run it, we are the first external check.
2. **`evaluation_tts_enabled: false`** in the release ⇒ TTS is **off** by default, so the bundled artifact's default behaviour is the *non*-TTS policy.
3. ⚠️ `docs/driverl_repro_eval.md` prints the **self-play** TTS reward defaults — off-road **−2.0**, cross-lane **1.0**, TTC **0.0**, goal threshold **1.5 m**, goal reward **0.5**, comfort **0.0** — against the teacher YAML's off-road −1.0, cross-lane 0.4, TTC 0.5, goal 3.0 m / 0.3. **Independent confirmation of R8**: the two releases are not a one-variable pair.

### 5.5 Proposed rows

**`DZ-10` (B&E, 13 GB download ⛔ PI) — run the TTS N-sweep ourselves on the released teacher.** With the nuPlan **mini** split + our existing maps, run the documented closed-loop eval on a fixed scenario set at **N ∈ {1 (no TTS), 8, 16, 32, 64}**, seed 42, horizon 5, margin 0.03, on the 4060. **Committed:** this **cannot** reproduce their headline numbers (mini tokens ≠ val14/test14 tokens) and must never be reported as such — it is a **paired, within-scenario** ablation. ⭐ **It answers I-1's quality half on a real published policy with a real learned critic**: if the N-sweep on mini shows the same shape as their Table 3 (noise below N=32, ≈ +0.5 at N=64), I-1's expected payoff is confirmed small on independent evidence; if it shows more, our fan-scoring comparison has been mis-scoped. **Controls:** N=1 must reproduce the non-TTS run exactly; candidate 0 is the argmax, so N=8 must never score below N=1 by more than the margin permits.

**`DZ-11` (B&E ⛔ PI, large) — headline reproduction.** The full six-task suite needs the benchmark tokens' DBs. ⭐ **Cheapest useful subset: `splits/test` only, which serves 4 of the 6 tasks** (`test14hard_nr/r`, `test14random_nr/r`) — and test14-hard is where the spread lives (DriveRL 89.97/89.18 vs Log-Replay 85.96/68.80). **Blocked on:** the `splits/test` size, which is an open gap (§5.2), and a PI download decision. ⚠️ nuPlan requires registration and carries its own non-commercial terms — **a separate authorisation from the repo permission already granted.**
