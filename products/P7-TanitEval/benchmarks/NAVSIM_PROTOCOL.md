# NAVSIM — the exact published protocol

**Owner:** TanitAD_EvalFlyWheel · **Product:** P7 TanitEval
**Purpose:** the citable specification our NavSim implementation is built against.
**Consumed by:** `products/P7-TanitEval/CRITERIA_REGISTRY.json` → `benchmarks.navsim`
(currently `pending: true`; see §8).
**Written:** 2026-08-23. **Devkit pinned at** `autonomousvision/navsim@0a380a9063d7162ec93d0f51e9990ebac585f720`
(`main`, committed 2025-10-27T21:19:43Z).

> ⛔ **This document exists because a wrong formula costs weeks.** Every number and
> every formula below carries its source. Where two primary sources disagree, both are
> named. Anything not established from a primary source is marked **UNVERIFIED** rather
> than reconstructed.

---

## 0. Evidence classes used here

| class | meaning |
|---|---|
| **PUB-PAPER** | stated in a NavSim paper — arXiv id + section/table given |
| **PUB-DEVKIT** | read from the official devkit source at the pinned SHA — file path given. A primary artifact, not a summary. |
| **PUB-LB** | read from the official HuggingFace leaderboard endpoint, with the retrieval date. A **moving** server: a snapshot, not a constant. |
| **INHERITED** | obtained by another agent in this programme and **not** re-verified by the author of this document |
| **UNVERIFIED** | could not be established from a primary source |

Three primary source families are used throughout:

| ref | what |
|---|---|
| **[N1]** | Dauner et al., *NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking*, NeurIPS 2024 Datasets & Benchmarks — **arXiv 2406.15349** (v1 2024-06-21, v2 2024-10-31; **v2 used here**) |
| **[N2]** | Cao et al., *Pseudo-Simulation for Autonomous Driving*, CoRL 2025 — **arXiv 2506.04218** (**v3 used here**; ⚠️ v1 differs numerically, see §6.2) |
| **[DK]** | `github.com/autonomousvision/navsim` at the pinned SHA — `docs/*.md` and `navsim/**` source |

⚠️ **The name "NavSim v2" is used for two different things.** [N2] is the *paper*; NavSim
v2.x is the *devkit release line*. They are not versioned together — the devkit's EPDMS
implementation changed after [N2]v1 was posted (§6.2).

---

## 1. Version matrix

**All rows PUB-DEVKIT** (`README.md` changelog at the pinned SHA), except where noted.

| release | date | what it introduced | metric in force | eval split | leaderboard |
|---|---|---|---|---|---|
| v0.1 | 2024-02-20 | initial demo; OpenScene-mini blobs; naive `ConstantVelocity` agent | — | — | — |
| v0.2 | 2024-03-11 | mini + test split integration; privileged `Human` agent | — | — | — |
| v0.3 | 2024-03-25 | leaderboard submission code | — | — | — |
| v0.4 | 2024-04-03 | trainval download; **Ego-status MLP agent** + training pipeline | — | — | — |
| **v1.0** | **2024-04-21** | **official devkit for AGC 2024 (CVPR)**; parallelised metric caching; TransFuser baseline; standardised filtered splits | **PDMS** | `navtest` | AGC2024 |
| **v1.1** | **2024-09-03** | HF `navtest` leaderboard; baseline checkpoints; updated paper | **PDMS** | `navtest` | `AGC2024-P/e2e-driving-navtest` |
| **v2.0** | **2025-02-28** | **EPDMS** (extends PDMS with more metrics + penalties); **two-stage pseudo-closed-loop simulation**; reactive traffic-agent policies | **EPDMS** | — | — |
| v2.1 | 2025-04-08 | dataset for the HF warmup leaderboard; two-stage reactive traffic agents | EPDMS | `warmup_two_stage` | warmup |
| v2.1.1 | 2025-04-13 | warmup dataset minor fixes | EPDMS | | |
| **v2.1.2** | **2025-04-24** | `navhard_two_stage` released; **EPDMS updated** | **EPDMS** | `navhard_two_stage` | warmup |
| **v2.2** | **2025-04-28** | **official devkit for AGC 2025**; `private_test_hard` released; `openscene_meta_datas` bugfix for `navhard`/`warmup` | **EPDMS** | `private_test_hard_two_stage` | `AGC2025/e2e-driving-2025` |
| — | 2025-07-16 | ICCV warmup leaderboard + registration system | EPDMS | `warmup_two_stage` | `AGC2025/e2e-driving-warmup-iccv` |
| — | **2025-09-29** | **bugfix**: `"multiplicative_metrics_prod"` and `"weighted_metrics"` were not correctly excluded by the human filter (issue #151) | EPDMS | | |

**Branch discipline (PUB-DEVKIT, `README.md`):** `main` is NavSim **v2**. NavSim **v1** and
its `navtest` leaderboard live on the **`v1.1` branch**. ⛔ **Do not compute v1 PDMS from
`main`** — the scorer there is the EPDMS scorer.

**Challenge variants**

| challenge | year | devkit | split | metric |
|---|---|---|---|---|
| AGC 2024 (CVPR) — *End-to-End Driving at Scale* | 2024 | v1.0 | private test | PDMS |
| AGC 2025 (CVPR) — *NAVSIM v2 End-to-End Driving* | 2025 | v2.2 | `private_test_hard_two_stage` | EPDMS (two-stage) |
| ICCV 2025 warmup | 2025 | v2.2 | `warmup_two_stage` | EPDMS (two-stage) |

AGC 2025 submission deadline 2025-05-11 00:00:00 UTC; one upload per day; ≈2 h evaluation
turnaround (PUB-DEVKIT, `README.md` + `docs/submission.md`).

---

## 2. Metric definitions

### 2.1 PDMS — NavSim **v1** only

**PUB-PAPER** [N1] §3, Eq. (1), reproduced verbatim from `docs/metrics.md` (`main`):

```
PDMS = ( ∏            score_m ) · ( Σ_{m ∈ {TTC, EP, C}}  w_m · score_m
          m ∈ {NC, DAC}              ─────────────────────────────────── )
                                       Σ_{m ∈ {TTC, EP, C}}  w_m
```

| term | role | weight | range |
|---|---|---|---|
| **NC** — no at-fault collisions | multiplier | — | {0, ½, 1} |
| **DAC** — drivable area compliance | multiplier | — | {0, 1} |
| **EP** — ego progress | weighted | **5** | [0, 1] |
| **TTC** — time to collision within bound | weighted | **5** | {0, 1} |
| **C** — comfort | weighted | **2** | {0, 1} |

**Weighted denominator = 5 + 5 + 2 = 12.** PDMS ∈ [0, 1]; reported ×100.
Weights are *"the cost weights used by the PDM-Closed planner and the 2023 nuPlan
Challenge"* [N1] §3.

⚠️ **DDC exists in the v1 code with weight 0.** PUB-DEVKIT, `v1.1` branch,
`.../scoring/pdm_scorer.py:39-48`: `progress_weight=5.0, ttc_weight=5.0,
comfortable_weight=2.0, driving_direction_weight=0.0`. So `navtest` leaderboard rows
carry a DDC column that **contributes nothing** to PDMS. Do not read it as a scored term.

**Computed per frame and averaged across frames** [N1] §3.

### 2.2 EPDMS — NavSim **v2**

**PUB-DEVKIT**, `docs/metrics.md`, verbatim (identical in form to [N2] Eq. (1)):

```
EPDMS = ( ∏                        filter_m(agent, human) )
          m ∈ {NC, DAC, DDC, TLC}

        · ( Σ_{m ∈ {TTC, EP, HC, LK, EC}}  w_m · filter_m(agent, human)
            ──────────────────────────────────────────────────────────── )
              Σ_{m ∈ {TTC, EP, HC, LK, EC}}  w_m

with   filter_m(agent, human) = 1.0        if m(human) = 0
                              = m(agent)   otherwise
```

| term | role | weight | range | new in v2? |
|---|---|---|---|---|
| **NC** — no at-fault collisions | multiplier | — | {0, ½, 1} | no |
| **DAC** — drivable area compliance | multiplier | — | {0, 1} | no |
| **DDC** — driving direction compliance | multiplier | — | {0, ½, 1} | **yes** (was weight-0 in v1) |
| **TLC** — traffic light compliance | multiplier | — | {0, 1} | **yes** |
| **EP** — ego progress | weighted | **5** | [0, 1] | no |
| **TTC** — time to collision within bound | weighted | **5** | {0, 1} | no |
| **LK** — lane keeping | weighted | **2** | {0, 1} | **yes** |
| **HC** — history comfort | weighted | **2** | {0, 1} | **yes** (revision of v1 C) |
| **EC** — extended comfort | weighted | **2** | {0, 1} | **yes** |

**Weighted denominator = 5 + 5 + 2 + 2 + 2 = 16.**

Sub-scores confirmed identically in [N2] Table 1 and `docs/metrics.md`; weights confirmed a
third time in **PUB-DEVKIT** `navsim/planning/script/config/pdm_scoring/scorer/pdm_scorer.yaml`.

**Origin of the new sub-metrics.** LK and TLC come from **Hydra-MDP++** (arXiv 2503.12820),
cited as [N2] ref [44]; the devkit source says so in comments —
*"Re-implementation of hydraMDP++'s traffic light compliance metric"*,
*"Revised implementation of hydraMDP++'s lane keeping metric"* (PUB-DEVKIT,
`pdm_scorer.py:584, 617`).

#### The human penalty filter — read this before implementing

Documented intent [N2] §3.1: *"If a rule violation is also committed by the human expert
driver in the same scene, the penalty is ignored."* Four facts about the **implementation**
that the prose does not say (all **PUB-DEVKIT**, `navsim/evaluate/pdm_score.py:171-219`):

1. ⚠️ **It applies to every sub-metric, not only DDC.** The code loops over all columns of
   the human's result and sets the agent's value to `1` wherever the human's is `0`, with
   `skip_columns = {"multiplicative_metrics_prod", "weighted_metrics",
   "weighted_metrics_array", "pdm_score"}`. The config comment
   (`pdm_scorer.yaml`) still says *"now only for driving_direction_compliance"* —
   **the comment is stale relative to the code.** Source conflict; the code is what runs.
2. **It fires only on real frames.** Guarded by `metric_cache.scene_type ==
   SceneFrameType.ORIGINAL` ⇒ **Stage-2 synthetic frames are never human-filtered.**
3. **EC is never human-filtered** — `two_frame_extended_comfort` is injected downstream
   (§2.5) and is not a column of `pdm_result` when the filter runs.
4. ⚠️ **The filter is OFF by default for the traffic scorer.** `pdm_scorer.yaml` sets
   `human_penalty_filter: True`; `pdm_and_traffic_scorer.yaml` **omits the key**, and
   `PDMScorerConfig.human_penalty_filter` defaults to `None` (falsy). Choosing a scorer
   silently changes the metric. The 2025-09-29 changelog entry is the bugfix that produced
   `skip_columns`; pre-2025-09-29 EPDMS numbers used a different filter.

### 2.3 Sub-metric implementations and thresholds

All **PUB-DEVKIT**, `navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py`
and `.../pdm_comfort_metrics.py`, with defaults from
`config/pdm_scoring/scorer/pdm_scorer.yaml`. Time index runs over
`num_poses + 1 = 41` states (§4).

**NC — no at-fault collisions** (`_calculate_no_at_fault_collision`, l.363-420)
At each of the 41 steps, query which tracked objects the ego polygon intersects. Red-light
tokens and already-collided tokens are skipped. A collision is **at-fault** iff
`collision_type ∈ {ACTIVE_FRONT_COLLISION, STOPPED_TRACK_COLLISION}`, **or** it is
`ACTIVE_LATERAL_COLLISION` **and** the ego is in multiple lanes or in non-drivable area.
Score = **0.0** if the object type is in nuPlan's `AGENT_TYPES` (vehicles, pedestrians,
bicycles), **0.5** otherwise (static objects). Score is the running minimum. Non-at-fault
contacts are recorded and thereafter ignored.

**DAC — drivable area compliance** (l.422-429)
`0.0` if **at any step at least one of the four ego-box corners** lies outside every
drivable polygon (`ROADBLOCK`, `INTERSECTION`, `DRIVABLE_AREA`, `CARPARK_AREA`); else `1.0`.

**DDC — driving direction compliance** (l.431-474)
Per-step centre-point displacement is accumulated only while the ego centre is **not** in
any on-route drivable polygon (`ONCOMING_TRAFFIC`) **and not inside an intersection**. Take
the maximum over sliding windows of `driving_direction_horizon = 1.0 s` (10 steps):

| max oncoming progress in any 1 s window | score |
|---|---|
| `< 2.0 m` (`driving_direction_compliance_threshold`) | **1.0** |
| `< 6.0 m` (`driving_direction_violation_threshold`) | **0.5** |
| otherwise | **0.0** |

**TLC — traffic light compliance** (l.582-613)
`0.0` if the ego polygon intersects a `red_light_token` geometry at **any** step; else `1.0`.

**EP — ego progress** (l.476-490 + `_aggregate_pdm_scores` l.223-251)
Raw progress = (centreline projection of the ego centre at the last step) − (at the first
step), clipped at ≥ 0. Normalisation is **relative to the PDM-Closed proposal evaluated in
the same batch** — `pdm_score.py:144` stacks `[PDM-Closed trajectory, agent trajectory]`:

```
masked          = raw_progress * multiplicative_score          # per proposal
norm_constant   = max(masked)                                   # over the 2 proposals
if norm_constant > progress_distance_threshold (5.0 m):
    EP = clip(raw_progress / norm_constant, 0, 1)
else:
    EP = 1.0                                                    # all proposals
```

⚠️ **Consequence:** an agent that legitimately progresses *further* than PDM-Closed becomes
its own normaliser and is clipped to 1.0. And when the safe upper bound is ≤ 5 m,
**every** proposal receives EP = 1.0.

⚠️ **EP normalisation differs between v1 and v2 — this is a real, citable divergence.**
PUB-DEVKIT `v1.1` branch `pdm_scorer.py:156-181` normalises the **masked** numerator
(`raw_progress = self._progress_raw * multiplicate_metric_scores`; then
`raw_progress / max_raw_progress`) and, in the ≤ 5 m branch, explicitly sets
`normalized_progress[multiplicate_metric_scores == 0.0] = 0.0`. `main` normalises the
**unmasked** numerator and has no such zero-out. A violating proposal therefore scores
EP = 0 under v1 and EP > 0 under v2.

**TTC — time to collision within bound** (l.492-580)
Defaults to `1.0`. `future_time_idcs = arange(0, int(1.0 * 10), 3) = [0, 3, 6, 9]` ⇒ the ego
box is projected forward at **0.0, 0.3, 0.6, 0.9 s** at constant velocity and heading (the
box's `CENTER` slot is overwritten with `FRONT_LEFT` so the projected shape is the
exterior). Evaluated for `time_idx ∈ [0, num_poses − max(future_time_idcs)] = [0, 31]`,
i.e. **t = 0.0 … 3.1 s**. A hit sets TTC = 0 unless the ego speed is below
`stopped_speed_threshold = 5e-3 m/s`, the token is a red light, or the token was already
counted. Fault logic: `is_agent_ahead(ego, track)`, or (ego in multiple lanes / non-drivable
area / inside an intersection) and `not is_agent_behind(ego, track)`.

**LK — lane keeping** (l.615-654, v2 only)
Lateral deviation = distance from the ego centre to the centreline linestring. Score `0.0`
iff the deviation exceeds `lane_keeping_deviation_limit = 0.5 m` for
`ceil(lane_keeping_horizon_window / interval) = ceil(2.0 / 0.1) = 20` **consecutive** steps.
⚠️ Steps inside an `INTERSECTION` are `continue`d — **they neither count nor reset the
consecutive counter**, so a run of exceedances is bridged across an intersection. This is
implementation behaviour, not documented prose; implement it exactly.

**C / HC — comfort** (`pdm_comfort_metrics.py`)
All six criteria must hold simultaneously (`ego_is_comfortable`, l.350-379):

| criterion | bound |
|---|---|
| longitudinal acceleration | `−4.05 … +2.40 m/s²` |
| absolute lateral acceleration | `≤ 4.89 m/s²` |
| absolute jerk magnitude | `≤ 8.37 m/s³` |
| absolute longitudinal jerk | `≤ 4.13 m/s³` |
| absolute yaw acceleration | `≤ 1.93 rad/s²` |
| absolute yaw rate | `≤ 0.95 rad/s` |

**HC (v2)** = the same six criteria evaluated on the trajectory **padded with the human's
past states** (`_calculate_history_comfort`, l.656-700; past interpolated at 0.1 s). If
`human_past_trajectory is None`, HC defaults to `1.0`. **C (v1)** = the same six criteria on
the trajectory alone, no history padding.

**EC — extended comfort / two-frame** (`ego_is_two_frame_extended_comfort`, l.429-467, v2 only)
Compares the current frame's trajectory with the **previous frame's** (Δt ≈ 0.5 s) over their
overlapping portion. For each of four features — acceleration magnitude, jerk magnitude, yaw
rate, yaw acceleration — take the elementwise difference between the two trajectories, then
its RMS over time. All four must be within bound:

| feature | threshold |
|---|---|
| RMS Δacceleration | `≤ 0.7 m/s²` |
| RMS Δjerk | `≤ 0.5 m/s³` |
| RMS Δyaw rate | `≤ 0.1 rad/s` |
| RMS Δyaw acceleration | `≤ 0.1 rad/s²` |

⚠️ **EC needs a second, temporally adjacent frame.** `scene_aggregator.py:62` asserts
`0 < observation_interval < 0.55 s`. A single isolated frame cannot be scored on EC.

### 2.4 ⚠️ The per-frame `pdm_score` column is NOT the EPDMS

**PUB-DEVKIT.** `PDMScorer._aggregate_pdm_scores` (l.240-246) explicitly **masks EC out** and
divides by `5+5+2+2 = 14`. The EPDMS with EC (denominator 16) is assembled downstream in
`run_pdm_score.py::compute_final_scores` (l.187-218), which injects
`two_frame_extended_comfort` into the weighted vector and recomputes
`score = multiplicative_metrics_prod × (Σ w·m / Σ w)`.

⇒ **The column to read is `score`, never `pdm_score`.** Reading `pdm_score` yields a
14-denominator, EC-free number that looks like a valid EPDMS.

### 2.5 Two-stage pseudo-closed-loop aggregation (v2)

**PUB-DEVKIT** `docs/metrics.md` + `scene_aggregator.py` + `run_pdm_score.py`; **PUB-PAPER**
[N2] §3.2, Eq. (2).

1. **Stage 1** — score the real initial observation over a 4 s horizon with EPDMS.
2. **Stage 2** — score the agent on **pre-generated synthetic observations** placed near the
   *expert's* 4 s endpoint (not the agent's). Each is scored with the same 4 s EPDMS.
3. **Aggregate** — Gaussian-weight the Stage-2 scores by how close each one's start point is
   to where the agent actually ended in Stage 1, then **multiply** Stage 1 by aggregated
   Stage 2:

```
s_combined = s1 · s2 ,   s2 = Σ_i ŵ_i · s2_i ,   ŵ_i = w_i / Σ_j w_j ,
w_i = exp( − ‖x_i − x̂‖² / (2σ²) ) ,   σ² = 0.1        # SceneAggregator.sigma_squared
```

`x_i` = start point of the *i*-th Stage-2 scenario; `x̂` = the agent's Stage-1 endpoint
(`endpoint_x/endpoint_y` vs `start_point_x/start_point_y`, `scene_aggregator.py:28-47`).
Degenerate guard: if the weight sum is 0 or NaN, weights become uniform `1/n`.

**The full aggregation actually implemented** (`run_pdm_score.py::calculate_individual_mapping_scores`,
l.242-290) works on **frame pairs**, because EC needs two adjacent frames. For each mapping
key `(orig_token, prev_token)` with its list of Stage-2 pairs:

```
group1 = score(orig_token)  ×  gaussian_weighted_mean(second-stage "now"  tokens)
group2 = score(prev_token)  ×  gaussian_weighted_mean(second-stage "prev" tokens)
scene  = (group1 + group2) / 2                    # elementwise, per sub-metric column
final  = mean over all mapping keys
```

**Design choices, with their justification** (PUB-PAPER [N2] §4.1): σ² = 0.05 and the default
σ² = 0.1 give the highest correlation to closed-loop; multiplicative Stage-1×Stage-2
aggregation beats arithmetic-mean and hybrid; two-stage reaches Pearson **r = 0.89**
(R² = 0.80) vs **r = 0.83** (R² = 0.70) single-stage, over 83 planners.

**Stage-2 generation** (PUB-PAPER [N2] §3.2) — needed only if we ever build our own split:
sample start points around the **expert's** 4 s endpoint — laterally every 0.5 m up to 2.0 m
each side, longitudinally every 5.0 m over the range reachable at ±4.0 m/s² for 4 s (up to 20
candidates at high speed); match each to the nearest human trajectory for heading + history,
discarding candidates differing by > 1.0 m/s velocity, > 1.0 m/s² acceleration or > 20° heading;
reject any start point violating NC/DAC/DDC/TLC; drop scenes left with < 5 valid observations.
Rendering is a **single-traversal modified MTGS** (arXiv 2503.12552) with LiDAR-registration +
bundle-adjustment pose init and pose optimisation; LPIPS 0.253 vs Street Gaussians 0.354
([N2] Table 3b). At 100 % density there are **≈12 Stage-2 observations per Stage-1
observation ⇒ 13 planner inferences per scenario**, against 80 for an 8 s nuPlan closed-loop
rollout at 10 Hz — the **6×** compute claim ([N2] §4.1).

---

## 3. Data requirements

### 3.1 Dependency chain

**nuPlan** (arXiv 2106.11810) → **OpenScene** (a redistribution of nuPlan at 2 Hz) → NavSim
scene filters. [N1] §3.1: OpenScene is *"120 hours of driving at a reduced frequency of 2 Hz
… a 90 % reduction of data storage requirements compared to nuPlan from over 20 TB to 2 TB."*
NavSim itself is dataset-agnostic in principle; it needs annotated HD maps, object boxes and
sensor data.

### 3.2 Splits and sizes

**PUB-DEVKIT** `docs/splits.md`. Splits are *scene filters* over downloadable *dataset splits*.
`navtrain` derives from `trainval`; `navtest` and `navhard_two_stage` derive from `test`.
**NavSim splits contain overlapping scenes; the OpenScene standard splits do not.**

| group | split | logs | sensors | config |
|---|---|---|---|---|
| OpenScene | `trainval` | 14 GB | **> 2000 GB** | `train_test_split=trainval` |
| OpenScene | `test` | 1 GB | 217 GB | `train_test_split=test` |
| OpenScene | `mini` | 1 GB | 151 GB | `train_test_split=mini` |
| **NavSim** | **`navtrain`** | 14 GB | **445 GB** (*300 GB without history*) | `train_test_split=navtrain` |
| **NavSim** | **`navtest`** | 983 MB | **223 GB** | `train_test_split=navtest` |
| **NavSim** | **`navhard_two_stage`** | 892 MB | **31 GB** | `train_test_split=navhard_two_stage` |
| Competition | `warmup_two_stage` | 27 MB | 1.2 GB | `train_test_split=warmup_two_stage` |
| Competition | `private_test_hard_two_stage` | 14 MB | 11 GB | `train_test_split=private_test_hard_two_stage` |

`navtrain` sensors are downloadable separately (`download_navtrain_aws.sh` /
`download_navtrain_hf.sh`) but still require the `trainval` **logs**. MD5s for the eight
`navtrain_{current,history}_{1..4}.tgz` archives are published in `docs/splits.md` —
**use them; missing files on `navtrain` are a reported recurring failure.**

### 3.3 Counts

| split | count | source |
|---|---|---|
| `navtrain` | **103k samples** | PUB-PAPER [N1] §3.1 |
| `navtest` | **12k samples** | PUB-PAPER [N1] §3.1 |
| `navhard` | **450 Stage-1 + 5462 Stage-2 observations** | PUB-PAPER [N2] §4.2 |
| combined navtrain+navtest standalone download | **450 GB** | PUB-PAPER [N1] §3.1 (`docs/splits.md` gives 445 + 223 GB separately — see §9) |
| `navmini` | **UNVERIFIED** — a "396 scenarios" figure surfaced in a secondary summary and is **not** in `docs/splits.md` or either paper. Do not quote. |
| `private_test_hard_two_stage` | **UNVERIFIED** — not stated anywhere primary; only 14 MB logs / 11 GB sensors are published |

### 3.4 Scenario filtering — how navtrain/navtest were built

**PUB-PAPER** [N1] §3.1, verbatim: *"We remove highly simplistic scenes by detecting if the
previously mentioned constant velocity agent exceeds a PDMS of 0.8. Similarly, we remove
scenes in which the human trajectory results in a PDMS of less than 0.8."* The second rule
also removes noisy annotations (bad boxes).

Effect, PUB-PAPER [N1] §3.1: on unfiltered OpenScene the constant-velocity agent scores
**79 %** PDMS and the human **91 %**; after filtering CV drops to **22 %** and the human
rises to **95 %**.

### 3.5 Sensor configuration

**PUB-PAPER** [N1] §3.1, verbatim: *"Our agent input, based on OpenScene, comprises eight
cameras, each with a resolution of 1920 × 1080 pixels, and a merged LiDAR point cloud from
five sensors. The input includes the current time-step and optionally 3 past frames,
totaling 1.5 s at 2 Hz."*

Devkit camera names (**INHERITED**, `navsim/common/dataclasses.py`): `cam_f0, cam_l0, cam_l1,
cam_l2, cam_r0, cam_r1, cam_r2, cam_b0`. LiDAR is a single `(6, n)` float32 array
`(x, y, z, intensity, ring, lidar_id)`.

⚠️ **Source conflict on history length.** [N1] §3.1 says **1.5 s** (current + 3 past at 2 Hz);
`docs/agents.md` says *"2 seconds into the past … (i.e., 4 frames)"*. `SceneFilter.num_history_frames = 4`.
4 frames at 0.5 s spacing span 1.5 s of elapsed time. **Use 4 frames; state the span you mean.**

### 3.6 Horizon and rate

Planning horizon **h = 4 s** [N1] §3. Simulation at **10 Hz** over that horizon [N1] §3.
Devkit: `proposal_sampling: num_poses: 40, interval_length: 0.1`
(PUB-DEVKIT `config/common/default_common.yaml`) ⇒ **41 states including t = 0**.
The scorer asserts `states.shape[1] == num_poses + 1`.
The **data** is 2 Hz; the **simulation** is 10 Hz — the agent's 4 s output is interpolated.

---

## 4. Simulation semantics — what is and is not simulated

**The agent is queried exactly once per scene.** [N1] §3, verbatim: *"driving agents are only
queried in the initial frame of each scene. Afterwards, the planned trajectory is kept fixed
for the entire trajectory duration. Over this short horizon, no environmental feedback is
provided to the driving agent, and the NAVSIM evaluation is purely based on the initial
real-world sensor sample."*

**Ego propagation.** [N1] §3, verbatim: *"An LQR controller is applied at each simulation
iteration to calculate steering and acceleration values, and a kinematic bicycle model
propagates the ego vehicle. We execute this pipeline at 10 Hz over the 4 s trajectory
horizon."* PUB-DEVKIT `pdm_simulator.py`: `BatchLQRTracker` + `BatchKinematicBicycleModel`,
`_tracker._discretization_time = 0.1`, stepping `time_idx = 1 … 40` from the real initial ego
state. Vehicle parameters are nuPlan's `get_pacifica_parameters()`.

⚠️ **Velocity and acceleration in the submitted trajectory are DISCARDED.** PUB-DEVKIT
`navsim/evaluate/pdm_score.py:41`, verbatim comment: `# NOTE: velocity and acceleration
ignored by LQR + bicycle model`. The agent's output is **poses only**; dynamics are whatever
the LQR + bicycle model produce while tracking them.

| aspect | v1 | v2 |
|---|---|---|
| Ego feedback into the planner | ❌ none | ❌ none — *"the ego-vehicle itself remains nonreactive"* (`docs/traffic_agents.md`) |
| Background **vehicles** | log-replay, non-reactive | **IDM-reactive by default** in two-stage runs |
| Pedestrians / static / non-vehicle actors | log-replay | log-replay — *"follow pre-recorded log data"* |
| Sensor simulation | ❌ none (real sensors only) | Stage 2 uses **pre-rendered** MTGS views; still not simulated at query time |
| Error recovery / distribution shift | ❌ not probed | ✅ probed via Stage-2 synthetic start points |

**Traffic-agent policies** (PUB-DEVKIT `docs/traffic_agents.md`): `log_replay` (non-reactive,
= v1), `constant_velocity` (**debugging only**), `navsim_IDM` (reactive). Selected via
`traffic_agents=non_reactive|reactive` for single-stage; **two-stage runs use reactive by
default**. IDM parameters (PUB-DEVKIT `navsim_IDM_traffic_agents.yaml`): `target_velocity 10 m/s`,
`min_gap_to_lead_agent 1.0 m`, `headway_time 1.5 s`, `accel_max 1.0 m/s²`, `decel_max 2.0 m/s²`,
`radius 100 m`, `minimum_path_length 20 m`, `idm_snap_threshold 3.0 m`,
`add_open_loop_parked_vehicles true`, `open_loop_detections_types []`.

**What NavSim does NOT do — state this plainly in any comparison:** it does not simulate
sensors at query time; it does not let the ego react; it does not run beyond 4 s (v1) or
2 × 4 s (v2); it has no perception noise model; and [N2] "Limitations" states the authors
*"do not yet demonstrate or claim direct correlation with performance metrics from real-world
vehicle deployment."*

---

## 5. Agent inputs, and the submission format

### 5.1 What an agent may consume at inference

**INHERITED** (established by a parallel research stream from `navsim/common/dataclasses.py`
and `docs/agents.md`; the paper-level facts are PUB-PAPER [N1] §3/§3.1). `AgentInput` is the
**only** object `compute_trajectory()` receives; the `Scene` object (maps, boxes, tracks,
`roadblock_ids`, traffic lights, future poses) is **training-only** and
`run_create_submission_pickle.py` refuses an agent declaring `requires_scene=True`.

| input | form | at inference? | privileged under our **vision-only** rule? |
|---|---|---|---|
| 8 cameras, 1920×1080 | `Cameras` | ✅ | **No** — this is the vision channel |
| camera intrinsics / extrinsics / distortion | `Camera` | ✅ | **No** — calibration |
| merged LiDAR from 5 sensors, `(6, n)` | `Lidar` | ✅ | **No** (a sensor), but **not vision** — excluded under a strict camera-only reading |
| `ego_velocity` (2-vector, local) | `EgoStatus` | ✅ | ⛔ **YES** |
| `ego_acceleration` (2-vector, local) | `EgoStatus` | ✅ | ⛔ **YES** |
| `ego_pose` per history frame (past ego trajectory) | `EgoStatus` | ✅ | ⛔ **YES** |
| `driving_command` one-hot | `EgoStatus` | ✅ | ⚠️ **Route/goal signal, not ego kinematics** — see below |
| history: 4 frames (`num_history_frames = 4`) | | ✅ | inherits whatever it carries |
| maps / lane graph / boxes / tracks / traffic-light state / future human trajectory | `Scene` | ❌ | train-only |

**Selective sensor use.** `get_sensor_config()` returns a `SensorConfig` with a flag (or a
per-frame index list) per sensor stream; `SensorConfig.build_no_sensors()` is what the
`EgoStatusMLPAgent` returns. ⛔ **There is no switch that removes ego status** — `ego_statuses`
is always populated and nothing in the framework verifies that an agent declined to read it.
**Our vision-only claim will have to be enforced and evidenced on our side.**

**The driving command.** `docs/agents.md`, verbatim: *"to disambiguate driver intention, we
provide a discrete driving command, indicating whether the intended route is towards the
left, straight or right direction. There is also a fourth command, representing 'unknown' …
**Importantly, the driving command in NAVSIM is based solely on the desired route, and does
not entangle information regarding obstacles and traffic signs** (as was prevalent on prior
benchmarks such as nuScenes)."* [N1] §2 states the design intent: *"We derive a navigation
goal from the lane graph instead of the human trajectory to prevent label leakage."*

⚠️ **Source conflict on its cardinality:** [N1] §3 says *"a one-hot vector with three
categories: left, straight, or right"*; the devkit `EgoStatus.driving_command` and
`docs/agents.md` are **4-dim** `(left, forward, right, unknown)`. **Implement 4.**

Derivation (INHERITED, OpenScene `helpers/driving_command.py`): project the ego onto the
route lane-graph centreline (Dijkstra over `route_roadblock_ids`), interpolate **20 m**
ahead, convert to ego frame; `y ≥ +2 m → left`, `y ≤ −2 m → right`, else `forward`;
`unknown` when the route is unrecoverable.

⚠️ **Open leak question, UNVERIFIED and load-bearing for us:** whether nuPlan's
`route_roadblock_ids` is itself derived from the expert's driven path. The devkit reads it
straight from the nuPlan DB with no derivation exposed. This is the same family as our own
*"a supplied route is optimistic by construction on PhysicalAI"* rule. **Run this check before
adopting a NavSim-style route command as evidence of route-following skill.**

### 5.2 Submission format

**PUB-DEVKIT** `docs/submission.md` + `scripts/submission/`.

- Produce **`submission.pkl`** — a pickle containing **one trajectory per test token** —
  via `run_create_submission_pickle.py`
  (`scripts/submission/run_cv_create_submission_pickle{,_warmup,_navhard}.sh`).
- Set `TRAIN_TEST_SPLIT` to the target split (`private_test_hard_two_stage`,
  `warmup_two_stage`, `navhard_two_stage`).
- **Required metadata, or the submission is invalid:** `TEAM_NAME`, `AUTHORS`, `EMAIL`,
  `INSTITUTION`, `COUNTRY`.
- Upload `submission.pkl` as a **HuggingFace *model* repo** (private is accepted), then paste
  the `Username/model` link into the competition Space's *New Submission* form.
- **One submission per day.** ≈2 h evaluation. The filename must be exactly `submission.pkl`.
- Leaderboard 2024 additionally requires open-source training + inference code and
  checkpoints, encoded as `TEAM_NAME = "<a href=Link/to/repository>Method name</a>"`;
  entries without it are periodically removed (~every 6 months).
- The trajectory itself: poses in the **ego frame**, `(x, y, heading)`, with a
  `trajectory_sampling`; velocity/acceleration are ignored (§4).

⛔ **Training on `test` / `navtest` / `navhard_two_stage` / `warmup_two_stage` /
`private_test_two_stage` is forbidden** for challenge submissions. Other public datasets and
pretrained weights are allowed but must be disclosed in the technical report to be
award-eligible (PUB-DEVKIT `docs/splits.md`, `README.md`).

⚠️ **The leaderboard does not record modality.** Only the five metadata fields above are
carried. Every "camera-only" claim on a leaderboard row is the authors', not the server's.

---

## 6. Published baselines

⛔ **Four mutually incomparable protocols exist. Never merge two of them.**

| | metric | split | typical range |
|---|---|---|---|
| **P1** NavSim v1 | PDMS | `navtest` (12k) | 20 – 95 |
| **P2** NavSim v2 pseudo-sim | EPDMS combined (S1 × S2) | `navhard_two_stage` (450 + 5462) | 11 – 55 |
| **P3** AGC2025 challenge | EPDMS combined (S1 × S2) | `private_test_hard_two_stage` | 17 – 53 |
| **P4** ⚠️ method papers' "NavSim v2" | EPDMS, **single-stage** | **`navtest`** | 64 – 87 |

**P4 is not navhard**, and the NavSim authors object to it in print — [N2] §4.2, verbatim:
*"we discourage the use of self-reported and unofficial "NAVSIM v2" benchmark splits, such as
reporting the EPDMS on the NAVSIM v1 navtest dataset without conducting two-stage
pseudo-simulation."* A P4 number sits ~35 points above anything on the navhard leaderboard.

### 6.1 P1 — NavSim v1, `navtest`, PDMS

**PUB-PAPER** [N1] Table 1 (all rows produced by the NavSim authors — UniAD, PARA-Drive and
TransFuser did **not** self-report these):

| Method | Ego status | Image | LiDAR | Video | NC | DAC | TTC | Comf. | EP | **PDMS** |
|---|:-:|:-:|:-:|:-:|---|---|---|---|---|---|
| Constant Velocity | ✓ | | | | 68.0 | 57.8 | 50.0 | 100 | 19.4 | **20.6** |
| Ego Status MLP | ✓ | | | | 93.0 | 77.3 | 83.6 | 100 | 62.8 | **65.6** |
| LTF | ✓ | ✓ | | | 97.4 | 92.8 | 92.4 | 100 | 79.0 | **83.8** |
| TransFuser | ✓ | ✓ | ✓ | | 97.7 | 92.8 | 92.8 | 100 | 79.2 | **84.0** |
| UniAD | ✓ | ✓ | | ✓ | 97.8 | 91.9 | 92.9 | 100 | 78.8 | **83.4** |
| PARA-Drive | ✓ | ✓ | | ✓ | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | **84.0** |
| **Human** | | | | | 100 | 100 | 100 | 99.9 | 87.5 | **94.8** |

**INHERITED** — self-reported `navtest` PDMS from method papers (each is that paper's own
run, not the NavSim authors'):

| Method | Backbone / modality | **PDMS** | source |
|---|---|---|---|
| VADv2-V8192 | camera + LiDAR (**Hydra-MDP's** reimplementation, not VADv2's) | 80.9 | 2406.06978v4 Tab. 1 |
| Hydra-MDP-V8192 | camera + LiDAR, R34 | 83.0 | 2406.06978v4 Tab. 1 |
| Hydra-MDP-V8192-W-EP | camera + LiDAR, R34 | 86.5 | 2406.06978v4 Tab. 1 |
| Hydra-MDP++ | **camera only**, R34 | 86.6 | 2503.12820v1 Tab. 1 |
| DiffusionDrive | camera + LiDAR, R34 | **88.1** *(confirmed by the live LB at 88.0157)* | 2411.15139 Tab. 1 |
| WoTE | camera + LiDAR | 88.3 | 2504.01941v2 Tab. 1 |
| VADv2 (updated) | camera | 89.3 | 2402.13243v2 Tab. 3 ⚠️ mis-captioned |
| Hydra-MDP-A / -B / -C | ViT-L / V2-99 / V2-99 | 89.9 / 90.3 / 91.0 | 2406.06978v4 Tab. 2 |
| DriveSuprim | R34 / V2-99 / ViT-L | 89.9 / 92.1 / 93.5 | 2506.06659v3 Tab. 2 |
| GoalFlow | camera + LiDAR + ego | 90.3 | 2503.05689v6 Tab. 1 |
| **GoalFlow†** | **+ ground-truth-endpoint goal — PRIVILEGED, not deployable** | 92.1 | 2503.05689v6 Tab. 1 |

⚠️ **GoalFlow†'s dagger means the goal point is the ground-truth trajectory endpoint.** It is
the exact construction our 2026-08-03 goal-admissibility rule warns about. Never quote it as
a deployable result.

**INHERITED — live `navtest` leaderboard** (`AGC2024-P/e2e-driving-navtest`, retrieved
2026-08-23; ⚠️ moving target). Top rows: iDriveVLA **94.8517**, ChainFlow-VLA 94.8468,
TOAD 94.7169, DrivoR 94.5945, CLOVER 94.4867, RAP 93.798, CLEAR 93.7435. Reference rows:
DiffusionDrive 88.0157, LEAD-LTFv6 86.4326, Baseline TransFuser **83.8822 ± 0.4477** (3 seeds),
Baseline LTF 83.5239 ± 0.552 (3 seeds), Baseline Ego Status MLP **66.3989 ± 0.9406** (3 seeds),
CV Baseline 20.6517. These reconcile with [N1] Table 3 (TransFuser 83.9±0.4, LTF 83.5±0.6,
MLP 66.4±0.9, CV 20.6). ⚠️ The leaderboard's DDC column has **weight 0** in PDMS (§2.1).

### 6.2 P2 — NavSim v2, `navhard_two_stage`, EPDMS combined

**PUB-PAPER** [N2]**v3** Table 2 (*"navhard leaderboard. Snapshot from 03/2026"*), S1 = real
observations, S2 = synthetic:

| | | CV | Ego MLP | LTF | NavFormer | LTFv6 | RAP | ZTRS | GuideFlow | SimScale | DrivoR | PDM-C* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NC | S1 | 88.8 | 93.2 | 96.2 | 96.2 | 96.5 | 97.1 | 98.8 | 99.5 | 99.5 | 99.1 | 94.4 |
| | S2 | 83.2 | 77.2 | 77.7 | 85.7 | 79.8 | 83.2 | 91.1 | 91.4 | 94.5 | 92.3 | 90.5 |
| DAC | S1 | 42.8 | 55.7 | 79.5 | 92.4 | 86.6 | 94.4 | 97.5 | 98.0 | 99.1 | 98.2 | 98.8 |
| | S2 | 59.1 | 51.9 | 70.2 | 81.0 | 75.5 | 83.8 | 90.4 | 89.5 | 94.2 | 91.6 | 90.6 |
| DDC | S1 | 70.6 | 86.6 | 99.1 | 95.7 | 99.2 | 98.7 | 100.0 | 99.4 | 99.8 | 99.3 | 100 |
| | S2 | 76.5 | 74.4 | 84.2 | 83.5 | 86.2 | 87.3 | 95.7 | 95.1 | 95.7 | 97.3 | 95.4 |
| TLC | S1 | 99.3 | 99.3 | 99.5 | 99.6 | 99.5 | 99.7 | 100.0 | 99.3 | 100.0 | 99.7 | 99.5 |
| | S2 | 98.0 | 98.2 | 98.0 | 97.6 | 97.8 | 98.0 | 99.0 | 98.8 | 99.2 | 99.0 | 98.4 |
| EP | S1 | 77.5 | 81.2 | 84.1 | 83.8 | 84.4 | 83.8 | 66.6 | 79.6 | 69.6 | 75.4 | 100 |
| | S2 | 71.3 | 77.1 | 85.1 | 90.1 | 89.5 | 86.8 | 63.5 | 77.5 | 75.8 | 75.7 | 100 |
| TTC | S1 | 87.3 | 92.2 | 95.1 | 96.0 | 95.1 | 96.8 | 98.8 | 99.3 | 99.5 | 98.6 | 93.5 |
| | S2 | 81.1 | 75.0 | 75.6 | 82.4 | 76.0 | 80.3 | 89.7 | 89.5 | 92.8 | 90.5 | 86.6 |
| LK | S1 | 78.6 | 83.5 | 94.2 | 94.7 | 94.4 | 94.6 | 96.2 | 94.8 | 95.7 | 94.8 | 99.3 |
| | S2 | 47.9 | 40.8 | 45.4 | 48.2 | 50.0 | 52.2 | 60.4 | 52.5 | 60.0 | 56.0 | 74.2 |
| HC | S1 | 97.1 | 97.5 | 97.5 | 96.4 | 97.7 | 96.4 | 96.6 | 97.1 | 95.5 | 97.5 | 87.7 |
| | S2 | 97.1 | 97.8 | 95.7 | 94.9 | 95.2 | 95.1 | 97.6 | 93.5 | 96.0 | 98.4 | 91.9 |
| EC | S1 | 60.4 | 77.7 | 79.1 | 60.9 | 76.4 | 66.2 | 44.0 | 58.2 | 28.4 | 70.2 | 36.0 |
| | S2 | 61.9 | 79.8 | 75.9 | 48.4 | 66.7 | 52.4 | 66.0 | 51.0 | 43.2 | 44.7 | 29.7 |
| **EPDMS** | | **11.4** | **14.1** | **25.1** | **34.1** | **31.9** | **39.6** | **48.1** | **51.5** | **53.2** | **54.5** | **56.6** |

\* **PDM-Closed is a privileged rule-based planner** with ground-truth perception and HD maps.
It is not a submission and is **not on the leaderboard**. ⛔ **`navhard` has no Human row** —
do not quote a "human ceiling" there.

**PUB-LB, verified independently by the author of this document** (`POST
https://agc2025-e2e-driving-navhard.hf.space/leaderboard`, `{"lb":"public"}`, retrieved
2026-08-23; the score column is `extended_pdm_score_combined`):

| # | entry | EPDMS | submitted |
|---|---|---|---|
| 1 | DrivoR | **54.5742** | 2026-01-30 |
| 2 | SimScale | 53.2483 | 2025-11-27 |
| 3 | GuideFlow | 51.5034 | 2025-12-22 |
| 4 | ZTRS | 48.1162 | 2025-11-04 |
| 5 | RAP | 39.6124 | 2025-10-25 |
| 6 | Baseline: NavFormer | 34.1075 | 2026-03-19 |
| 7 | LEAD-LTFv6 | 31.9101 | 2025-11-21 |
| 8 | Baseline: LTF | 25.1218 | 2025-10-21 |
| 9 | Baseline: MLP | 14.1725 | 2025-10-21 |
| 10 | Baseline: CV | 11.4816 | 2025-10-21 |

**All ten rows agree with [N2]v3 Table 2 to truncation.** ✅

#### ⚠️ Source conflict, and its resolution

A parallel stream reported a disagreement between "the CoRL paper" and the leaderboard —
CV 10.9 vs 11.4816, MLP 12.7 vs 14.1725, LTF 23.1 vs 25.1218 — **with identical sub-scores**.
That disagreement is real but it is **between arXiv versions of [N2]**, not between the paper
and the server. **Both columns below were read directly from the arXiv HTML of the two
versions by the author of this document (PUB-PAPER, not inherited):**

| arm | **[N2]v1** Tab. 2 | **[N2]v3** Tab. 2 | live LB (2026-08-23) |
|---|---|---|---|
| CV | 10.9 | **11.4** | 11.4816 |
| Ego MLP | 12.7 | **14.1** | 14.1725 |
| LTF | 23.1 | **25.1** | 25.1218 |
| PDM-Closed | 51.3 | **56.6** | *(not submitted)* |

⇒ **[N2]v3 is aligned with the deployed leaderboard; [N2]v1 is not.** The CV/MLP/LTF
**sub-scores are identical** across the two versions while the combined EPDMS moved, so the
**aggregation** changed, not the per-scene scoring (consistent with the v2.1.2 → v2.2 EPDMS
update and the 2025-09-29 human-filter bugfix). PDM-Closed's sub-scores *did* move
(e.g. TTC S2 83.1 → 86.6, EC S2 25.4 → 29.7, LK S2 73.7 → 74.2). [N2]v1 also carries only
**four** planners (CV, MLP, LTF, PDM-C); v3 carries **eleven**, adding NavFormer and six
community submissions. ⛔ **Cite [N2]v3 or the leaderboard. Never quote [N2]v1's navhard
numbers as current**, and never mix the two.

### 6.3 P3 — AGC2025, `private_test_hard_two_stage`, EPDMS combined

**INHERITED** (`AGC2025/e2e-driving-2025`, retrieved 2026-08-23). Entries are **team names**;
no method identification and no modality. **A different split from navhard — do not merge.**

Top 10: `Simple` **53.0582** · `bjtu_jia_team&qcraft` 51.3117 · `DRL_CASIA&XIAOMI` 51.081 ·
`ONE` 50.9529 · `aduiduidui` 50.511 · `Westwell` 50.017 · `GG Bone` 50.0081 · `DiffVLA++`
49.1238 · `Genshin, launch!` 48.8439 · `ACM Lab` 48.4414.
Reference: `baseline_constant_velocity` **17.5071**.
Warmup split (`AGC2025/e2e-driving-warmup-iccv`): Hawaiidriver 43.5985,
`baseline_constant_velocity` 18.5356, TJU-smartiot 8.5018 — only 3 entries.

⛔ **Do not name an "AGC2025 winner".** The rank-1 team name `Simple` has **no primary
technical report** tying it to a named method. **UNVERIFIED.**

### 6.4 P4 — ⚠️ "EPDMS on navtest", single-stage (method papers)

Listed **only so these are not mistaken for navhard**. **INHERITED**, 2506.06659v3 Table 3:
Human Agent 90.3 · Ego Status MLP 64.0 · Transfuser (R34) 76.7 · HydraMDP++ (R34) 81.4 ·
DriveSuprim (R34) 83.1 · HydraMDP++ (V2-99) 85.1 · DriveSuprim (V2-99) 86.0 ·
HydraMDP++ (ViT-L) 85.6 · **DriveSuprim (ViT-L) 87.1**.

🚩 **These conflict with Hydra-MDP++'s own P4 table** (2503.12820v1 Tab. 2, navsim v2.0-era):
Transfuser **77.8** vs 76.7, and Transfuser **LK 67.6 vs 92.7** (×1.37). The LK gap is not
noise — the EPDMS implementation changed between v2.0 and v2.2. **The two P4 tables are not
comparable to each other**, let alone to P2.

---

## 7. ⭐ The ego-status shortcut — what NavSim claims, and whether it holds

This section is **load-bearing for TanitAD**, because our inference is **vision-only** (PI,
2026-08-03, binding).

### 7.1 The problem NavSim names

**PUB-PAPER** [N1] §2, verbatim:
> *"Next, most planning models in nuScenes receive the human trajectory endpoint as a
> discrete direction command […], thereby leaking ground-truth information into inputs.
> Moreover, about 75 % of the scenarios in nuScenes involve trivial straight driving […],
> leading to simple solutions when extrapolating the ego-motion. For instance, AD-MLP
> demonstrates that an MLP on the kinematic ego status (ignoring perception completely) can
> achieve state-of-the-art displacement errors. Such blind agents are undeniably dangerous,
> which highlights a broader concern: displacement metrics are not correlated to closed-loop
> driving. […] **We derive a navigation goal from the lane graph instead of the human
> trajectory to prevent label leakage**, and propose principled simulation-based metrics as
> an alternative to displacement errors."*

The underlying findings (**INHERITED**): AD-MLP (arXiv 2305.10430) reaches nuScenes L2
avg **0.29 m** / collision **0.19 %** from a 21-dim input of past poses + velocity +
acceleration + a 3-way command, **beating** VAD-Base (0.37) and UniAD (1.03).
BEV-Planner (arXiv 2312.03031) removes AD-MLP's past-trajectory leak and *still* gets
L2 avg **0.35** from ego velocity/acceleration/yaw + command; it measures **73.9 %** of
nuScenes as straight driving, and shows that blanking all of VAD's camera inputs moves L2
only 0.37 → 0.46 while scaling ego velocity ×0.5 sends it to **3.19**.

### 7.2 NavSim's three design answers

1. **Scenario filtering** — drop scenes the constant-velocity agent already solves
   (PDMS > 0.8) and scenes the human fails (PDMS < 0.8) (§3.4).
2. **Non-reactive simulation + PDMS** instead of displacement error (§2, §4).
3. **A route-derived navigation goal**, from the lane graph rather than the human endpoint
   (§5.1).

### 7.3 Is the claim supported? — Yes, partially on v1; decisively on v2

| benchmark | metric | CV floor | **ego-status-only** | best sensor agent | ceiling | blind agent's share of the sensor margin over CV |
|---|---|---|---|---|---|---|
| nuScenes | L2 avg ↓ | GoStraight 0.83 | **Ego-MLP 0.35** | VAD-Base 0.37 · UniAD 0.46 | — | **ego-only WINS** |
| **navtest** (P1) | PDMS ↑ | 20.6 | **65.6** | TransFuser/PARA-Drive **84.0** | Human **94.8** | **71 %** |
| navtest live LB | PDMS ↑ | 20.65 | 66.40 ± 0.94 | TransFuser 83.88 | — | 65 % |
| **navhard** (P2) | EPDMS ↑ | **11.4** | **14.1** | LTF **25.1** · DrivoR **54.5** | PDM-C (privileged) **56.6** | **≈ 6 %** vs DrivoR; **21 %** vs LTF |

**PUB-PAPER** [N1] §4.2, verbatim: *"The Ego Status MLP achieves a PDMS of 65.6, showing the
value of the acceleration and navigation goal for avoiding collisions and driving off-road.
**However, we observe a clear gap between agents relying solely on the ego status and those
considering sensor data, in contrast to results on nuScenes.**"*

**Verdict.** On nuScenes the shortcut *wins*. On **navtest** it is demoted but still retains
**~65–71 %** of the CV→sensor margin — real, and the authors do not hide it. On **navhard**
it collapses: the blind MLP is 14.1 against DrivoR's 54.5, and on Stage-2 (perturbed)
observations it is **worse than constant velocity** on three safety sub-scores
(NC 77.2 vs 83.2, TTC 75.0 vs 81.1, LK 40.8 vs 47.9), with DAC 51.9 vs LTF's 70.2 — it
leaves the drivable area about half the time. **NavSim v2/navhard is the version where the
claim is decisively supported.**

Three caveats, stated so we do not over-claim:
1. ⚠️ **navhard EPDMS is not on navtest PDMS's scale.** Part of the widening is the
   multiplicative S1×S2 aggregation. The **margin-share** reading (71 % → ~6–21 %) is the
   defensible one; the raw point drop is not.
2. **navhard has no human ceiling** — the top reference is a privileged planner.
3. ⛔ **NavSim publishes no fully ego-free number.** Even "camera-only" LTF (83.8 PDMS)
   consumes `driving_command + ego_velocity + ego_acceleration`.

### 7.4 The number to design against

**PUB-PAPER** [N1] Table 2 (TransFuser ablations on `navtest`), **INHERITED** extraction:

| arm | inputs removed | PDMS |
|---|---|---|
| A1 / A2 / A3 | seeds, full ego status | 83.3 / 84.0 / 84.4 |
| **B2** "Goal and velocity only" | acceleration removed | **82.3** |
| **B1** "Goal only" | **velocity and acceleration removed** | **81.8** |

[N1]: *"Discarding velocity and acceleration (B1) lowers PDMS by 1.5−2.6 […] **We conclude
that while TransFuser benefits from the ego status, it is not purely relying on the kinematic
state for planning.**"*

⇒ **On navtest, stripping all ego kinematics while keeping the route goal costs ~1.5–2.6 PDMS
and still beats the ego-status MLP by ~16 points.** That is the ceiling a vision-only-plus-goal
arm should be measured against — **not** the 84.0 full-ego row. ⚠️ B1 is not fully ego-free:
it keeps the goal. There is **no published NavSim number for an arm with neither ego state nor
goal** — **UNVERIFIED**, and a candidate experiment for us.

---

## 8. Mapping onto TanitAD's binding criteria

Our four metric families (PI, 2026-08-02, binding) and leak guards, as ids in
`products/P7-TanitEval/CRITERIA_REGISTRY.json`. **NavSim does not report ADE at all** — it is
a wholly different instrument, which is exactly why it is worth adopting.

| our family / criterion | NavSim term | covered? |
|---|---|---|
| `long.progress` — ego progress | **EP** | ✅ direct (note the PDM-Closed-relative normalisation, §2.3) |
| `long.distance_keeping` — headway / time-gap / TTC to lead | **TTC** (binary, 0.9 s projection) | ⚠️ **PARTIAL** — binary within-bound, not a headway or time-gap in metres/seconds. Our criterion needs the continuous quantity. |
| `long.target_speed` — target-speed accuracy | — | ❌ **ABSENT in NavSim.** Must be computed by us; refuse explicitly if not. |
| `long.along_track_error` | — | ❌ absent (NavSim has no displacement metric) |
| `lat.cross_track` | **LK** is a thresholded centreline deviation (0.5 m for 2 s) | ⚠️ **PARTIAL** — binary, not an error in metres |
| `lat.heading` / `lat.curvature` / `lat.yaw_rate` | comfort bounds on yaw rate (0.95 rad/s) and yaw accel (1.93 rad/s²) are **feasibility gates**, not errors | ❌ **ABSENT as errors** |
| `tac.manoeuvre_decision` / `tac.confusion` / `tac.goal_selection` | — | ❌ **ABSENT.** NavSim scores the executed trajectory, never a manoeuvre decision. |
| `strat.decision` / `strat.route_goal` | `driving_command` is an **input**, never scored | ❌ **ABSENT** |
| safety (no family of ours) | **NC, DAC, DDC, TLC** | ➕ NavSim adds four rule-compliance terms we do not currently instrument |
| comfort (no family of ours) | **C / HC, EC** | ➕ NavSim adds a *cross-frame consistency* term (EC) we do not have |

**Consequences for adoption:**
1. ⛔ **An EPDMS number alone does not satisfy our four-families rule.** Three of four families
   are ABSENT from NavSim by construction. A NavSim artifact must either add them or
   **refuse them explicitly with a reason and an `n`** (the skill's REFUSED state), never
   omit them silently.
2. ✅ **Leak guard `vision_only_inference`:** NavSim *permits* ego state at inference and does
   not verify abstention. Our artifact must state `protocol.inference_inputs` itself; the
   framework will not do it for us (§5.1).
3. ⚠️ **Leak guard `goal_situation_disjoint`:** `driving_command` is route-derived and
   explicitly disentangled from obstacles and signs — admissible in principle. But its
   upstream `route_roadblock_ids` provenance is **UNVERIFIED** (§5.1). Settle that before
   quoting a route-conditioned NavSim result.
4. ⚠️ **Leak guard `route_head_echo`:** NavSim never scores the route, so it cannot manufacture
   the echo defect we found in flagship v1 — but it also cannot detect it.
5. **Estimator:** NavSim publishes **point estimates only**; the leaderboard's only dispersion
   is a multi-seed ± on three baseline rows. Our `taniteval/ci.py` episode-cluster bootstrap
   does not transfer directly, because NavSim's unit is a **scene token**, not an episode.
   ⇒ **An open design question**, not a solved one. Do not report a NavSim CI until the
   clustering unit is settled.

**Registry hand-off.** `benchmarks.navsim` is `pending: true` with
`protocol_doc` already pointing at this file. Filling `criteria` must ship **with its
deliberate-regression test** in the same commit (programme rule §6.5). That is the next work
item and is **not** done by this document.

---

## 9. What is UNVERIFIED

| # | item | status |
|---|---|---|
| 1 | `navmini` scenario count ("396") | **UNVERIFIED** — appears only in a secondary summary; absent from `docs/splits.md` and both papers. Do not quote. |
| 2 | `private_test_hard_two_stage` split size | **UNVERIFIED** — only 14 MB logs / 11 GB sensors published. A "~280 Stage-1 frames" figure inferred from sub-score granularity is an **inference**, not a source. |
| 3 | AGC2025 challenge winner identity | **UNVERIFIED** — rank 1 is a team name (`Simple`) with no primary technical report. |
| 4 | Provenance of nuPlan `route_roadblock_ids` — is the route derived from the expert path? | **UNVERIFIED** and load-bearing for our goal-admissibility rule (§5.1). |
| 5 | Storage total: [N1] says the curated benchmark is **450 GB**; `docs/splits.md` lists navtrain 445 GB + navtest 223 GB separately | **Sources disagree**; both named. Likely 450 GB ≈ navtrain-only. Budget from `docs/splits.md`. |
| 6 | History span: **1.5 s** ([N1] §3.1) vs *"2 seconds into the past"* (`docs/agents.md`), both = 4 frames | **Sources disagree**; both named. |
| 7 | Driving command cardinality: **3** ([N1] §3) vs **4** (devkit + `docs/agents.md`) | **Sources disagree**; implement 4. |
| 8 | Human-penalty-filter scope: *"only for driving_direction_compliance"* (`pdm_scorer.yaml` comment) vs all sub-metrics (code) | **Doc/code conflict**; the code runs. |
| 9 | TransFuser on navtest: **84.0** (NavSim authors, [N1] Tab. 1) vs **78.0** (Hydra-MDP's reproduction, 2406.06978v4 Tab. 1) | Both named. Hydra-MDP's Eq. 12 treats DDC as a live multiplier where official v1 PDMS gives it weight 0 (navsim issue #14). **Use 84.0.** |
| 10 | WoTE's baseline column (2504.01941v2 Tab. 1) lists a "VADv2" row byte-identical to Hydra-MDP-V8192's, and CV as 21.6 where every other source says 20.6 | **INHERITED**, flagged unreliable. WoTE's own row (88.3) stands. |
| 11 | VADv2 (2402.13243v2) table captions name benchmarks their columns do not match; per its own Appendix its "3DGS" table is a **private 2000 h dataset, not NavSim** | **INHERITED.** Its EPDMS rows are **inadmissible as NavSim results**; its navtest PDMS 89.3 is usable with the caveat. |
| 12 | Modality of every live-leaderboard row | **UNVERIFIED** — neither leaderboard API exposes a modality field (§5.2). |
| 13 | Hydra-MDP's 91.3 on Leaderboard 1.1 ([N1] Tab. 3) | **PUB-PAPER** but no longer re-checkable — Hydra-MDP is absent from the live navtest leaderboard. |
| 14 | Per-team numeric scores for the CVPR 2024 challenge (463 submissions, 143 teams) | In [N1] supplementary, **not retrieved**. |
| 15 | Whether our episode-cluster bootstrap transfers to NavSim's scene-token unit | **Open design question** (§8.5) |

---

## 10. Banked sources

All banked with `tools/kb_add.py --tag benchmarks --cited-by <this file>` into
`TanitAD Research Lab/Library/`; verify with `python tools/kb_add.py --verify`.

| arXiv | title | what it supplies here |
|---|---|---|
| **2406.15349** | NAVSIM (NeurIPS 2024 D&B) | PDMS formula; task; splits; filtering; sensors; Tab. 1 navtest; Tab. 2 ego ablations |
| **2506.04218** | Pseudo-Simulation for Autonomous Driving (CoRL 2025) | EPDMS formula + weights; two-stage aggregation; navhard counts; Tab. 2 navhard |
| **2503.12820** | Hydra-MDP++ | origin of the EPDMS TLC and LK sub-metrics; a P4 table |
| **2306.07962** | Parting with Misconceptions (PDM, CoRL 2023) | the planner and scoring function PDMS is named after |
| **2106.11810** | nuPlan | the closed-loop score PDMS reimplements; the source dataset |
| **2305.10430** | AD-MLP | the ego-status shortcut on nuScenes |
| **2312.03031** | BEV-Planner — *Is Ego Status All You Need?* | quantifies the shortcut; proposes CCR |
| **2205.15997** | TransFuser (PAMI) | the NavSim baseline and its LTF camera-only variant |
| **2212.10156** | UniAD | navtest baseline |
| **2406.06978** | Hydra-MDP | 2024 challenge winner; NavFormer head; the DDC/issue-#14 caveat |
| **2411.15139** | DiffusionDrive | navtest PDMS 88.1 |
| **2503.12552** | MTGS | the renderer generating Stage-2 synthetic views |
| **2506.06659** | DriveSuprim | P4 tables |
| **2503.05689** | GoalFlow | navtest PDMS; the privileged GT-endpoint-goal row |
| **2504.01941** | WoTE | navtest PDMS 88.3 |
| **2402.13243** | VADv2 | navtest PDMS 89.3 (captions unreliable) |

**Non-arXiv primaries:** `github.com/autonomousvision/navsim` at
`0a380a9063d7162ec93d0f51e9990ebac585f720` — `README.md`, `docs/{metrics,splits,submission,traffic_agents,agents,cache}.md`,
`navsim/evaluate/pdm_score.py`, `navsim/planning/script/run_pdm_score.py`,
`navsim/planning/simulation/planner/pdm_planner/scoring/{pdm_scorer,pdm_comfort_metrics,scene_aggregator}.py`,
`navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py`,
`navsim/planning/simulation/planner/pdm_planner/utils/pdm_enums.py`,
`navsim/planning/script/config/{common/default_common.yaml,pdm_scoring/scorer/*.yaml,common/traffic_agents_policy/*.yaml}`,
and the **`v1.1` branch** copies of `pdm_scorer.py` / `pdm_enums.py` for the v1 definitions.
**Leaderboards:** `AGC2024-P/e2e-driving-navtest`, `AGC2025/e2e-driving-navhard`,
`AGC2025/e2e-driving-2025`, `AGC2025/e2e-driving-warmup-iccv` (all HuggingFace Spaces;
snapshots dated 2026-08-23).
