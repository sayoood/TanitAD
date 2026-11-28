# nuScenes — the exact published protocols

**Owner:** TanitAD_EvalFlyWheel · **Product:** P7 TanitEval
**Registry hook:** `products/P7-TanitEval/CRITERIA_REGISTRY.json` → `benchmarks.nuscenes` (currently `pending: true`)
**Written:** 2026-08-23 · **Evidence class on every claim below.**

> ⛔ **The single most important sentence in this document.**
> nuScenes has **four** distinct protocols. Three of them (detection, tracking, prediction) are
> official, devkit-enforced, and admissible. The fourth — **open-loop planning** — **has no official
> protocol at all**: there is no nuScenes planning challenge, no devkit evaluation code, and no
> leaderboard. Every published L2/collision number comes from one of at least four **mutually
> incompatible research codebases**, and the choice of codebase moves the number by up to **70 %**
> and can **flip the ranking between two methods**. Numbers from different implementations are not
> comparable and must never be placed in the same column.

---

## 1. Protocol matrix

| Task | Official protocol? | Metric set | Eval code owner | Split |
|---|---|---|---|---|
| **Detection** | ✅ YES — nuScenes Detection Challenge | **NDS**, mAP, ATE, ASE, AOE, AVE, AAE | `nuscenes-devkit` `eval/detection/` | train / val / test |
| **Tracking** | ✅ YES — nuScenes Tracking Challenge | **AMOTA**, AMOTP, MOTAR, + CLEAR-MOT suite | `nuscenes-devkit` `eval/tracking/` | train / val / test |
| **Prediction** | ✅ YES — nuScenes Prediction Challenge | minADE_k, minFDE_k, MissRate_2_k, **OffRoadRate** | `nuscenes-devkit` `eval/prediction/` | train / train_val / val (instance-sample pairs) |
| **Open-loop planning** | ⛔ **NO. Community convention only.** | L2 @1/2/3 s, collision rate @1/2/3 s | ST-P3 / UniAD / VAD / AD-MLP / BEV-Planner / PARA-Drive — **each different** | nuScenes val (6 019 samples) |

*Evidence: [MEASURED-from-source] the three official protocols each ship a README plus executable
config in `nutonomy/nuscenes-devkit`; no `eval/planning/` directory exists in the devkit.*

⚠️ **Consequence for us:** rows 1–3 can be adopted verbatim. Row 4 cannot — see §6 and §8.

---

## 2. Detection — exact definitions

### 2.1 NDS (nuScenes Detection Score)

**[PUBLISHED — arXiv 1903.11027, banked]** the paper states the formula as

```
NDS = (1/10) · [ 5 · mAP + Σ_{mTP ∈ TP} (1 − min(1, mTP)) ]
```

with `TP = {mATE, mASE, mAOE, mAVE, mAAE}`.

**[MEASURED-from-source — `nuscenes/eval/detection/data_classes.py`, `DetectionMetrics.nd_score`]**
the devkit implements it as:

```python
@property
def nd_score(self) -> float:
    # Summarize.
    total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))
    # Normalize.
    total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))
    return total
```

with each TP score clipped as

```python
score = 1.0 - tp_errors[metric_name]
score = max(0.0, score)   # Bound scores to minimum 0
```

and `mean_ap_weight = 5` from `detection_cvpr_2019.json`. **The two agree**: `max(0, 1−e) ≡ 1 − min(1, e)`,
and the denominator is `5 + 5 = 10`. ⇒ **mAP carries 50 % of NDS; the five TP metrics carry 10 % each.**

### 2.2 mAP — center distance, NOT IoU

**[MEASURED-from-source — `eval/detection/README.md` + `configs/detection_cvpr_2019.json`]**

- `dist_fcn` = `"center_distance"` — matching is by **2D center distance on the ground plane**.
  ⛔ **There is no IoU anywhere in nuScenes detection matching.** This is the single most common
  misimplementation of this benchmark.
- `dist_ths` = **`[0.5, 1.0, 2.0, 4.0]` metres**. AP is computed at each threshold; mAP is the mean
  over the four thresholds **and** over the 10 classes.
- `min_recall` = `0.1`, `min_precision` = `0.1`, `max_boxes_per_sample` = `500`.
- The precision–recall curve is sampled at **101 recall points** (`DetectionMetricData.nelem = 101`,
  `np.linspace(0, 1, 101)`).

**AP integration [MEASURED-from-source — `eval/detection/algo.py::calc_ap`]:**

```python
def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1
    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)
```

⚠️ Note the two non-obvious details a reimplementation gets wrong: the **`+1` offset** that excludes
the min-recall bin, and the **renormalisation by `1/(1 − min_precision)`** after subtracting the
precision floor.

### 2.3 The five TP metrics

All five are computed **at a single matching threshold `dist_th_tp = 2.0 m` center distance**
(not averaged over the four mAP thresholds).

| metric | definition **[PUBLISHED — arXiv 1903.11027 + devkit README]** | unit |
|---|---|---|
| **ATE** — Average Translation Error | Euclidean center distance in 2D | m |
| **ASE** — Average Scale Error | `1 − IoU` in 3D **after aligning orientation and translation** | dimensionless |
| **AOE** — Average Orientation Error | smallest yaw-angle difference between prediction and GT | rad |
| **AVE** — Average Velocity Error | absolute velocity error, L2 norm of the 2D velocity difference | m/s |
| **AAE** — Average Attribute Error | `1 − acc`, `acc` = attribute classification accuracy | dimensionless |

**Aggregation [MEASURED-from-source — `algo.py::calc_tp`]:**

```python
def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """
    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))
```

⇒ a TP metric is the **mean of the per-recall-bin error over all operating points with recall > 10 %,
up to the maximum recall the detector achieved**. If a class never reaches 10 % recall its TP error
is set to **1.0** (worst). The devkit README states the same rule in prose, verbatim:
*"Matching and scoring happen independently per class and each metric is the average of the
cumulative mean at each achieved recall level above 10%."*

**Normalisation into NDS:** `TP_score = max(1 − TP_error, 0.0)`. ⚠️ ATE/AVE are in **metres / m·s⁻¹**,
so `1 − error` is a unit-mixing convention, not a normalised quantity. It saturates at 0 for
ATE > 1 m or AVE > 1 m/s.

**Per-class exemptions [MEASURED-from-source — `eval/detection/README.md`, all four verbatim]:**

- *"Velocity error for barriers and cones are ignored."*
- *"Attribute error for barriers and cones are ignored."*
- *"Orientation errors for cones are ignored."*
- ⚠️ *"Orientation error is evaluated at 360 degree for all classes except barriers where it is only
  evaluated at 180 degrees."* — barriers are symmetric front-to-back; an implementation that scores
  them at 360° will systematically over-report AOE.

### 2.4 The 10 detection classes

`barrier, bicycle, bus, car, construction_vehicle, motorcycle, pedestrian, traffic_cone, trailer, truck`

**[PUBLISHED — arXiv 1903.11027]** these are the subset of the dataset's **23 annotation classes**
retained after merging similar classes and dropping classes with **< 10 000 annotations**.

### 2.5 Official filtering rules

**[MEASURED-from-source — `detection_cvpr_2019.json` + `eval/detection/README.md`]**

- **Per-class evaluation range (`class_range`, metres):**
  `car 50 · truck 50 · bus 50 · trailer 50 · construction_vehicle 50 · pedestrian 40 · motorcycle 40 · bicycle 40 · traffic_cone 30 · barrier 30`
  Boxes beyond a class's range are removed (GT **and** predictions).
- **Min points:** *"All boxes (GT) without lidar or radar points in them are removed."*
  ⚠️ **The threshold is `num_lidar_pts + num_radar_pts > 0`, i.e. ≥ 1 point — not a larger count.**
  Predicted boxes are **not** filtered by point count.
- **Bike racks:** all bicycle and motorcycle boxes (GT **and** prediction) falling inside a
  `bicycle_rack` annotation are removed.
- **`max_boxes_per_sample = 500`** — a submission exceeding this is rejected.

---

## 3. Prediction — exact definitions

**[MEASURED-from-source — `eval/prediction/metrics.py` + `configs/predict_2020_icra.json`]**

The official challenge config `predict_2020_icra.json` specifies:

| field | value |
|---|---|
| horizon (`seconds`) | **6 s** |
| sampling | **2 Hz** ⇒ **12 waypoints** per prediction |
| state | x–y only (2D) |
| max modes submitted | **25** |
| metrics | `MinADEK`, `MinFDEK`, `MissRateTopK`, `OffRoadRate` |
| `k_to_report` | **`[1, 5, 10]`** for MinADEK, MinFDEK and MissRateTopK |
| `MissRateTopK.tolerance` | **2.0 m** |
| aggregator | `RowMean` (mean over all instance-sample pairs) for all four |

**Definitions:**

- **minADE_k** — *"The average of pointwise L2 distances between the predicted trajectory and ground
  truth over the `k` most likely predictions."* (mean over the 12 waypoints, min over the top-k modes,
  then RowMean over agents).
- **minFDE_k** — L2 distance between the **final** points; min over top-k; mean over agents.
- **MissRate_2_k** — *"If the maximum pointwise L2 distance between the prediction and ground truth
  is greater than 2 meters, we define the prediction as a miss."* ⚠️ **max over time, not final-point** —
  this is stricter than the Argoverse/Waymo miss-rate convention.
- **OffRoadRate** — *"Fraction of trajectories not entirely contained in drivable area."* Computed
  against the **map expansion's drivable-area mask**; a trajectory counts as off-road if
  `out_of_bounds or not np.all(drivable_area[index_row, index_col])`.
  ⚠️ **Requires the map expansion.** Not computable from the base dataset.

**Splits [MEASURED-from-source — `eval/prediction/splits.py`]:**
named splits are `'mini_train'`, `'mini_val'`, `'train'`, `'train_val'`, `'val'`.
`NUM_IN_TRAIN_VAL = 200` — the **first 200 scenes of the base train split become `train_val`**;
the remainder is `train`. A prediction instance is addressed as the string
`{instance_token}_{sample_token}`.

⚠️ **UNVERIFIED:** the exact instance-sample-pair counts per split. Neither the devkit README, the
splits module, nor the tutorial states them, and `nuscenes.org` is a JS SPA that returns no fetchable
text. **Do not quote a count until it is produced by running `get_prediction_challenge_split()`.**

⚠️ **`pip install nuscenes-devkit` is not sufficient for this task.** **[MEASURED-from-source —
`docs/installation.md`]**: *"The pip package does not support the prediction or tracking code."*
A source checkout is required.

---

## 4. Tracking — exact definitions

**[MEASURED-from-source — `eval/tracking/README.md`; metric origin PUBLISHED — AB3DMOT, arXiv 1907.03961, banked]**

- **Matching:** 2D center distance on the ground plane, **threshold 2 m** (same as detection TP).
- **Recall sampling:** **n = 40** recall thresholds, from `1/(n−1)` to `1`; points with **recall < 0.1
  are excluded**.
- **MOTAR** (recall-normalised MOTA), the quantity AMOTA integrates:

  ```
  MOTAR = max( 0 , 1 − (IDS_r + FP_r + FN_r − (1 − r)·P) / (r·P) )
  ```

  where `r` is the recall operating point and `P` the number of GT positives. The `(1 − r)·P` term
  removes the FNs that are unavoidable at recall `r`, so the metric spans [0, 1] at every recall.
- **AMOTA** = `(1/(n−1)) · Σ_r MOTAR` — the average of MOTAR over the recall thresholds.
- **AMOTP** = `(1/(n−1)) · Σ_r ( Σ_{i,t} d_{i,t} / Σ_t TP_t )` — recall-averaged mean position error
  of matched pairs. **Lower is better** (it is an error, unlike AMOTA).
- **7 tracking classes:** `bicycle, bus, car, motorcycle, pedestrian, trailer, truck`
  (the 10 detection classes minus the three static ones: barrier, traffic_cone, construction_vehicle).

  ⚠️ **UNVERIFIED / flagged inconsistency:** the devkit README describes the tracking classes as the
  detection classes "removing static objects", which yields 7, but `construction_vehicle` is not
  static. Treat the **enumerated list of 7 above** as authoritative and the rationale sentence as
  loose prose.
- **Ranges:** 40 m for bicycle/motorcycle, 50 m for the rest.
- **Filtering:** bike-rack removal and the "GT boxes with zero lidar+radar points are removed" rule
  carry over from detection. GT and predicted tracks are **linearly interpolated** to avoid
  fragmentation artifacts.

---

## 5. ⭐ Open-loop planning — and the protocol inconsistency

### 5.1 The nominal protocol

Every implementation shares this outer shape:

- Predict the ego trajectory over a **3 s horizon at 2 Hz** ⇒ 6 waypoints at
  `{0.5, 1.0, 1.5, 2.0, 2.5, 3.0} s`, in the ego frame.
- Report **L2 (m)** against the recorded human trajectory at **1 s / 2 s / 3 s** plus an average.
- Report **collision rate (%)** — rasterise other agents into a BEV occupancy grid, place the ego
  footprint (1.85 m × 4.084 m) at each predicted waypoint, count overlaps.
- Evaluate on **nuScenes val**, nominally 6 019 samples.

**Everything inside that shape differs between implementations.** There are four documented axes.

### 5.2 ⭐ Axis 1 — L2 averaging: **at-timestep (NoAvg) vs averaged-up-to-timestep (TemAvg)**

**This is settled from a primary source: UniAD's own configuration file.**

**[MEASURED-from-source — `OpenDriveLab/UniAD`, `projects/configs/stage2_e2e/base_e2e.py`, verbatim]:**

```python
# there exists multiple interpretations of the planning metric, where it differs between uniad and stp3/vad
# uniad: computed at a particular time (e.g., L2 distance between the predicted and ground truth future trajectory at time 3.0s)
# stp3: computed as the average up to a particular time (e.g., average L2 distance between the predicted and ground truth future trajectory up to 3.0s)
planning_evaluation_strategy = "uniad"  # uniad or stp3
```

and the switch is executed in `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py`:

```python
planning_tab.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
...
    if planning_evaluation_strategy == "stp3":
        row_value.append("%.4f" % float(value[: i + 1].mean()))
    elif planning_evaluation_strategy == "uniad":
        row_value.append("%.4f" % float(value[i]))
```

**PARA-Drive states the same thing as an equation [PUBLISHED — CVPR 2024, banked as `paradrive-cvpr2024`, verbatim]:**

> *"Averaging over the time dimension: Taking the L2 error at the 3-second horizon for example, UniAD
> computes the L2^{3s} by averaging over N samples in the val set, whereas VAD and AD-MLP compute this
> error by averaging over both samples and time intervals"*

```
L2^{3s} = (1/N) · Σ_N (1/6) · Σ_{t ∈ {0.5s, 1s, 1.5s, 2s, 2.5s, 3s}} L2^t          (PARA-Drive Eq. 1)
```

> *"resulting in significantly smaller L2 errors and collision rates for VAD and AD-MLP compared to UniAD"*

**Corroborated independently in the source of each implementation:**

| implementation | `compute_L2` behaviour | convention |
|---|---|---|
| **UniAD** `planning_head_plugin/planning_metrics.py` | returns a **per-timestep vector**; reporting indexes `value[i]` | **NoAvg (at-timestep)** — the default |
| **ST-P3** `stp3/metrics.py` | accumulates per-timestep; reporting takes the running mean | **TemAvg** |
| **VAD** `VAD/planner/metric_stp3.py` | `ade = sum(...)/pred_len` over the slice `[:cur_time]` — a **scalar** | **TemAvg** |
| **AD-MLP** | README: *"follow instruction in ST-P3 for running its evaluation process"*, runs `deps/stp3/evaluate_for_mlp.py` | **TemAvg** |
| **BEV-Planner** `fbbev/planner_head/metric_stp3.py` | header: *"calculate planner metric same as stp3"* | **TemAvg** (+ their own corrections) |

VAD's call site, verbatim:

```python
for i in range(future_second):
    if fut_valid_flag:
        cur_time = (i+1)*2
        traj_L2 = self.planning_metric.compute_L2(
            pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
            gt_ego_fut_trajs[0, :cur_time]
        )
```

⇒ `plan_L2_3s` is the mean over **all six** waypoints, not the error at 3 s.

**⭐ How large is the effect? [PUBLISHED — PARA-Drive Table 1, MEASURED by its authors]**
They transform VAD's protocol into UniAD's one change at a time, on the **same checkpoint**:

| step | L2 Ave₁,₂,₃ₛ | Col. Ave₁,₂,₃ₛ |
|---|---:|---:|
| VAD, VAD protocol (reproduction) | **0.7192** | 0.21 |
| **+ Remove averaging over time** | **1.2238** | 0.51 |
| + Frame masking strategy | 1.0665 | 0.44 |
| + Excluding pedestrians | 1.0665 | 0.38 |
| = UniAD protocol | | |

**The averaging convention alone moves the same model from 0.72 m to 1.22 m — a +70 % change.**

### 5.3 Axis 2 — agent filtering in the collision map

**[PUBLISHED — PARA-Drive §3.2, verbatim]** *"UniAD excludes pedestrians from the GT occupancy map,
leading to lower collision rates compared to VAD and AD-MLP. On the other hand, UniAD includes
invisible objects in every frame, making the evaluation more challenging"*.

### 5.4 Axis 3 — frame masking / valid-sample definition

**[PUBLISHED — PARA-Drive §3.2, verbatim]** *"VAD and AD-MLP exclude data clips if any one of the
frames is invalid in the data sequence, while UniAD includes such clips but assigns a zero error to
those invalid frames. The inclusion of these clips with frames assigned with zero error can reduce
the overall error rate"*.

**[PUBLISHED — BEV-Planner]** *"For 6019 samples of nuScenes val split, the number of final valid
samples is 5119 (85% of all samples)."* — i.e. **15 % of val is silently dropped**, and *which* 15 %
differs by implementation.

**[PUBLISHED — BEV-Planner]** ST-P3 additionally has a **data defect**: *"ST-P3 mistakenly used
samples from other scenes while generating GT of these tail samples, so errors occurred during
training and testing."* ⇒ **ST-P3's own published planning numbers are not trustworthy at all**, and
BEV-Planner marks its ST-P3 row with a dagger for exactly this reason.

**[PUBLISHED — PARA-Drive §3.2]** *"the nuScenes dataset ... consists of relatively short data
clips — typically only 40 frames each. Excluding even a few frames in each data clip can lead to a
performance change ranging from 5% to 10%."*

### 5.5 Axis 4 — first-frame handling

**[PUBLISHED — PARA-Drive §3.2, verbatim]** *"AD-MLP addresses this by excluding the first two frames
in their evaluation protocol whereas UniAD and VAD do not, leading to artificially higher errors in
the evaluation of UniAD and VAD."*

Removing that first-frame noise is worth another **0.16 m** on VAD (1.0665 → 0.9086) and **0.13 m** on
UniAD (1.0788 → 0.9474) — PARA-Drive Table 1.

### 5.6 ⭐⭐ Axis 5 — the collision computation itself is broken in the legacy protocols

Three independent groups document the same defects.

**Ego-vehicle box.** All legacy implementations use `W = 1.85 m`, `H = 4.084 m`
**[MEASURED-from-source — identical constants in UniAD, VAD and BEV-Planner `metric_stp3.py`]**, with
BEV bounds `X_BOUND = Y_BOUND = [-50.0, 50.0, 0.5]` ⇒ a **200 × 200 grid at 0.5 m/cell**.

- **[PUBLISHED — PARA-Drive]** *"Axis-aligned ego vehicle representation: Existing evaluation of
  calculating collision rates can generate artificial false positives and negatives due to the neglect
  of the ego vehicle's orientation"*.
- **[PUBLISHED — SparseDrive, arXiv 2405.19620, verbatim]** *"previous benchmark convert obstacle
  bounding boxes into occupancy map with a grid size of 0.5m, resulting in false collisions in certain
  cases"* and *"The heading of ego vehicle is not considered and assumed to remain unchanged"*.
- **[PUBLISHED — BEV-Planner]** collisions were treated as independent per timestep, `CR(t) = Σ𝕀ᵢ / N`;
  they change it to `(Σ𝕀ᵢ) > 0` — one collision anywhere on the trajectory is one collision — and
  rasterise at **0.1 m** instead of 0.5 m.

**⭐ THE MEASUREMENT THAT SETTLES IT [PUBLISHED — PARA-Drive Tables 2 and 8, MEASURED by its authors].**
They run the metric on the **ground-truth human trajectory**:

| protocol | collision rate of the **GT human trajectory** | best published method under the same protocol |
|---|---:|---:|
| UniAD evaluation methodology | **0.36 %** (Table 8) / 0.384 % (Table 2) | UniAD 0.31 % |
| VAD evaluation methodology | **0.96 %** | VAD 0.22 % |
| PARA-Drive standardized (oriented box + 0.1 m grid) | **0.00 %** | — |

⛔ **Under both legacy protocols the recorded human driver "collides" more often than the models do.**
The metric's false-positive floor **exceeds the entire published spread between methods**. Any
collision number produced with an axis-aligned ego box on a 0.5 m grid is measuring rasterisation
artifacts, not safety, and is **not decision-grade**.

*(This is the same failure class as CLAUDE.md's `overlapping_holdout_se`: an instrument whose noise is
larger than the effect it is quoted for, used for years because it produced plausible-looking numbers.)*

### 5.7 ⭐ The ranking flips

**[PUBLISHED — PARA-Drive Table 8, MEASURED]** the same two checkpoints, scored under both legacy
protocols:

| eval methodology | method | L2 1 s | 2 s | 3 s | **Ave₁,₂,₃ₛ** | Col. Ave |
|---|---|---:|---:|---:|---:|---:|
| **UniAD** | GT trajectory | — | — | — | — | **0.36** |
| **UniAD** | VAD | 0.50 | 1.02 | 1.68 | **1.07** | 0.38 |
| **UniAD** | UniAD | 0.48 | 0.96 | 1.65 | **1.03** | 0.31 |
| **UniAD** | PARA-Drive | 0.40 | 0.77 | 1.31 | **0.83** | 0.30 |
| **VAD** | GT trajectory | — | — | — | — | **0.96** |
| **VAD** | VAD | 0.41 | 0.70 | 1.05 | **0.72** | 0.22 |
| **VAD** | UniAD | 0.48 | 0.74 | 1.07 | **0.76** | 0.17 |
| **VAD** | PARA-Drive | 0.25 | 0.46 | 0.74 | **0.48** | 0.25 |

⛔ **Under UniAD's protocol, UniAD beats VAD (1.03 < 1.07). Under VAD's protocol, VAD beats UniAD
(0.72 < 0.76). Same checkpoints, opposite conclusion, from the evaluation convention alone.**

And PARA-Drive's headline: *"under a unified evaluation protocol, the gap in L2 errors between VAD and
UniAD narrows significantly (previous δ = 1.03 − 0.72, current δ = 0.9474 − 0.9086)"* — **the apparent
0.31 m gap between the two most-cited planners is really 0.039 m; ~87 % of it was protocol artifact.**

*(Direct resonance with CLAUDE.md: the `heldout` vs `full_set` correction moved paired deltas up to
×−4.15 **including a sign flip**. This is that defect, in the published literature, at field scale.)*

---

## 6. ⭐ The ego-status shortcut critique

### 6.1 AD-MLP — perception is not needed to win

**[PUBLISHED — "Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes",
arXiv 2305.10430, Zhai et al., banked]**

Architecture, verbatim: `Linear₅₁₂←₂₁ – ReLU – Linear₅₁₂←₅₁₂ – ReLU – Linear₁₈←₅₁₂`.
Inputs: *"ego vehicle's motion trajectories of the past Tp=4 frames, instantaneous velocity and
acceleration"* plus a 3-way one-hot high-level command. **No camera. No LiDAR. No perception at all.**

Their ablation, in their own evaluation (**ST-P3 / TemAvg protocol**):

| inputs | L2 1 s | 2 s | 3 s | **Avg** | Col. 1 s | 2 s | 3 s | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trajectory only | 0.53 | 0.91 | 1.48 | **0.97** | 0.17 | 0.46 | 0.83 | **0.49** |
| + velocity | 0.33 | 0.48 | 0.66 | **0.49** | 0.21 | 0.29 | 0.40 | **0.30** |
| + acceleration | 0.24 | 0.32 | 0.49 | **0.35** | 0.18 | 0.22 | 0.28 | **0.23** |
| + high-level command | 0.20 | 0.26 | 0.41 | **0.29** | 0.17 | 0.18 | 0.24 | **0.19** |

Abstract, verbatim: *"Our simple method achieves similar end-to-end planning performance on the
nuScenes dataset with other perception-based methods, reducing the average L2 error by about 20%."*

### 6.2 BEV-Planner — isolating the ego-status contribution

**[PUBLISHED — "Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?",
arXiv 2312.03031, Li et al., CVPR 2024, banked]**

Their Table 1 turns ego status on and off inside the *same* architectures. `Ego in BEV` = ego state
fused into the BEV features; `Ego in Planner` = ego state fed directly to the planning head.
CCR = **Curb Collision Rate**, their new metric: *"the collision rate between the predicted
trajectories and curbs (road boundaries)"*, rasterised at 0.1 m.

| # | method | Ego in BEV | Ego in Planner | L2 1/2/3 s | **L2 Avg** | Col. 1/2/3 s | **Col. Avg** | **CCR Avg** |
|---|---|---|---|---|---:|---|---:|---:|
| 0 | ST-P3 † | ✗ | ✗ | 1.59/2.64/3.73 | **2.65** | 0.69/3.62/8.39 | **4.23** | 8.37 |
| 1 | UniAD | ✗ | ✗ | 0.59/1.01/1.48 | **1.03** | 0.16/0.51/1.64 | **0.77** | 1.93 |
| 2 | UniAD | ✓ | ✗ | 0.35/0.63/0.99 | **0.66** | 0.16/0.43/1.27 | **0.62** | 1.72 |
| 3 | UniAD | ✓ | ✓ | 0.20/0.42/0.75 | **0.46** | 0.02/0.25/0.84 | **0.37** | 1.59 |
| 4 | VAD-Base | ✗ | ✗ | 0.69/1.22/1.83 | **1.25** | 0.06/0.68/2.52 | **1.09** | 3.82 |
| 5 | VAD-Base | ✓ | ✗ | 0.41/0.70/1.06 | **0.72** | 0.04/0.43/1.15 | **0.54** | 2.72 |
| 6 | VAD-Base | ✓ | ✓ | 0.17/0.34/0.60 | **0.37** | 0.04/0.27/0.67 | **0.33** | 2.47 |
| 7 | **GoStright** (constant straight) | — | ✓ | 0.38/0.79/1.33 | **0.83** | 0.15/0.60/2.50 | **1.08** | 8.62 |
| 8 | **Ego-MLP** (ego status only) | — | ✓ | 0.15/0.32/0.59 | **0.35** | 0.00/0.27/0.85 | **0.37** | 2.93 |
| 9 | BEV-Planner* | ✗ | ✗ | 0.27/0.54/0.90 | **0.57** | 0.04/0.35/1.80 | **0.73** | 3.98 |
| 10 | BEV-Planner | ✗ | ✗ | 0.30/0.52/0.83 | **0.55** | 0.10/0.37/1.30 | **0.59** | 4.26 |
| 11 | BEV-Planner+ | ✓ | ✗ | 0.28/0.42/0.68 | **0.46** | 0.04/0.37/1.07 | **0.49** | 4.21 |
| 12 | BEV-Planner++ | ✓ | ✓ | 0.16/0.32/0.57 | **0.35** | 0.00/0.29/0.73 | **0.34** | 3.16 |

† ST-P3's row is flagged by the authors for the tail-sample GT defect (§5.4).
*Spelling `GoStright` is as printed in the paper.*

**What this table measures:**
- **Ego status is worth 55–70 % of L2 with zero change to perception.** UniAD 1.03 → 0.46; VAD 1.25 → 0.37.
- **Row 8 vs rows 3 and 6: an MLP with no perception at all (0.35) beats UniAD-with-ego (0.46) and
  matches VAD-with-ego (0.37) and BEV-Planner++ (0.35).**
- **Row 7: a hard-coded straight line (0.83) beats UniAD without ego status (1.03).**

Abstract, verbatim: *"These models tend to rely predominantly on the ego vehicle's status for future
path planning"* and *"we suggest the community reassess relevant prevailing research and be cautious
whether the continued pursuit of state-of-the-art would yield convincing and universal conclusions."*

### 6.3 ⭐ The second leak: the "high-level command" is derived from the ground-truth future

This is less discussed than ego status and is **worse**.

**[MEASURED-from-source — `hustvl/VAD`, `tools/data_converter/vad_nuscenes_converter.py`, verbatim]:**

```python
if ego_fut_trajs[-1][0] >= 2:
    command = np.array([1, 0, 0])  # Turn Right
elif ego_fut_trajs[-1][0] <= -2:
    command = np.array([0, 1, 0])  # Turn Left
else:
    command = np.array([0, 0, 1])  # Go Straight
```

`ego_fut_trajs[-1][0]` is the **final lateral offset of the ground-truth future trajectory**,
thresholded at ±2 m. **The "input" is the label, quantised into 3 bins, and handed back to the model.**

**[PUBLISHED — NAVSIM, arXiv 2406.15349, verbatim]** *"Most planning models in nuScenes receive the
human trajectory endpoint as a discrete direction command, thereby leaking ground-truth information
into inputs."*

For completeness, the ego-status vector VAD feeds (`gt_ego_lcf_feat`, 9-D)
**[MEASURED-from-source, same file]**: `[vx, vy, ax, ay, yaw-rate w, ego_length, ego_width, v0, Kappa]`.

⭐ **This is the TanitAD route-head echo defect, published as state of the art.** Our own measurement
(CLAUDE.md, 2026-08-03): *flagship v1's route head is an exact bijection of the nav we feed it
(369/369 and 81/81) and scored 1.0000 — an echo of its own input read as skill.* The nuScenes planning
literature has the same construction, at field scale, in the leaderboard.

### 6.4 The distribution problem — why the shortcut is sufficient

- **[PUBLISHED — BEV-Planner, verbatim]** *"73.9% of the nuScenes data involve scenarios of driving
  straightforwardly, as reflected by the distribution of the trajectory in Fig. 2."*
- **[PUBLISHED — NAVSIM, verbatim]** *"About 75% of the scenarios in nuScenes involve trivial straight
  driving, leading to simple solutions when extrapolating the ego-motion. For instance, AD-MLP
  demonstrates that an MLP on the kinematic ego status (ignoring perception completely) can achieve
  state-of-the-art displacement errors."*
- **[PUBLISHED — PARA-Drive, MEASURED]** excluding frames whose command is *"keep forward"* leaves
  **686 challenging key-frames** on the nuScenes val set — **≈ 11 % of the 6 019 val samples.**

⇒ **A metric averaged over full val is ~89 % dominated by frames where constant-velocity
extrapolation is the correct answer.**

### 6.5 ⭐ Where the shortcut actually breaks — the discriminating measurement

**[PUBLISHED — PARA-Drive Table 6, MEASURED, all rows under PARA-Drive's *standardized* protocol]**

| scenario set | method | ego states? | Col. Ave_all | L2 Ave_all | Offroad % | Offlane % |
|---|---|---|---:|---:|---:|---:|
| **val** (full) | UniAD | No | 0.40 | 0.8317 | 0.91 | 1.74 |
| **val** | VAD | No | 0.30 | 0.7830 | 1.03 | 1.93 |
| **val** | PARA-Drive | No | 0.17 | **0.5574** | 0.12 | 0.83 |
| **val** | **AD-MLP** | **Yes** | 0.20 | **0.5568** | **1.21** | **2.45** |
| **val** | PARA-Drive+ | Yes | 0.13 | 0.4939 | 0.11 | 0.78 |
| **targeted** (686) | UniAD | No | 0.15 | 0.9935 | — | — |
| **targeted** | VAD | No | 0.34 | 1.0840 | — | — |
| **targeted** | PARA-Drive | No | **0.14** | 0.9082 | — | — |
| **targeted** | **AD-MLP** | **Yes** | **0.94** | 0.9360 | — | — |
| **targeted** | PARA-Drive+ | Yes | 0.05 | 0.7018 | — | — |

⭐ **The two numbers that matter:**
1. **On full val, the perception-free AD-MLP (L2 0.5568) is a statistical tie with the full perception
   stack PARA-Drive (0.5574).** The benchmark cannot tell them apart.
2. **On the 686 non-straight frames, AD-MLP's collision rate is 0.94 % vs PARA-Drive's 0.14 % — 6.7×
   worse.** And its map compliance is 10× worse off-road (1.21 vs 0.12) and 3× worse off-lane
   (2.45 vs 0.83) **on full val**.

**[PUBLISHED — PARA-Drive §4.1, verbatim]** *"We find that, in the targeted scenarios as well as the
map compliance error rates, AD-MLP has significantly worse performance than PARA-Drive. This suggests
that the open-loop evaluation scheme is still very informative and emphasizes again the importance of
our standardized and enhanced evaluation methodology."*

⇒ **The shortcut is invisible to the standard protocol and clearly visible to a corrected one.**
This is the hinge of the recommendation in §8.

---

## 7. Published baselines — every row tagged with its protocol

⛔ **A row may only be compared with rows carrying the same protocol tag.** Cross-tag comparison is the
error this document exists to prevent.

### 7.1 Protocol tags used below

| tag | meaning |
|---|---|
| `uniad-noavg` | L2 at the timestep; UniAD's frame masking (invalid frames scored 0); pedestrians **excluded** from the collision map; axis-aligned ego box; 0.5 m grid |
| `stp3-temavg` | L2 averaged over all waypoints up to T; clips with any invalid frame dropped; pedestrians included; axis-aligned ego box; 0.5 m grid. Used by ST-P3, VAD, AD-MLP |
| `bevplanner` | ST-P3 lineage + 0.1 m rasterisation, trajectory-level collision (`Σ𝕀 > 0`), estimated yaw, 5 119 valid val samples, adds CCR |
| `paradrive-std` | oriented ego box, 0.1 m grid (1000×1000 BEV), pedestrians included, first frame removed, + map-compliance and the 686-frame targeted subset |
| `sparsedrive` | L2 per VAD (`stp3-temavg`); collision re-implemented with estimated yaw and box-vs-box overlap |

### 7.2 `uniad-noavg`

| method | ego status | L2 1 s | 2 s | 3 s | Avg | Col 1 s | 2 s | 3 s | Avg | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NMP | — | — | — | 2.31 | — | — | — | 1.92 | — | UniAD Tab. 7 |
| SA-NMP | — | — | — | 2.05 | — | — | — | 1.59 | — | UniAD Tab. 7 |
| FF (LiDAR) | — | 0.55 | 1.20 | 2.54 | 1.43 | 0.06 | 0.17 | 1.07 | 0.43 | UniAD Tab. 7 |
| EO (LiDAR) | — | 0.67 | 1.36 | 2.78 | 1.60 | 0.04 | 0.09 | 0.88 | 0.33 | UniAD Tab. 7 |
| ST-P3 | No | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 | UniAD Tab. 7 |
| **UniAD** | No | 0.48 | 0.96 | 1.65 | **1.03** | 0.05 | 0.17 | 0.71 | **0.31** | UniAD Tab. 7 |
| VAD (re-scored) | No | 0.50 | 1.02 | 1.68 | 1.07 | 0.02 | 0.28 | 0.85 | 0.38 | PARA-Drive Tab. 8 |
| PARA-Drive (re-scored) | No | 0.40 | 0.77 | 1.31 | **0.83** | 0.07 | 0.25 | 0.60 | 0.30 | PARA-Drive Tab. 8 |
| ⚠️ **GT human trajectory** | — | — | — | — | — | 0.35 | 0.38 | 0.35 | **0.36** | PARA-Drive Tab. 8 |

### 7.3 `stp3-temavg`

| method | ego status | L2 1 s | 2 s | 3 s | Avg | Col 1 s | 2 s | 3 s | Avg | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ST-P3 (own paper) ⚠️ GT defect | No | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 | ST-P3 Tab. 3 |
| VAD-Tiny | Yes | 0.46 | 0.76 | 1.12 | 0.78 | 0.21 | 0.35 | 0.58 | 0.38 | VAD paper |
| **VAD-Base** | Yes | 0.41 | 0.70 | 1.05 | **0.72** | 0.07 | 0.17 | 0.41 | **0.22** | VAD paper |
| UniAD (re-scored) | No | 0.48 | 0.74 | 1.07 | 0.76 | 0.12 | 0.13 | 0.28 | 0.17 | PARA-Drive Tab. 8 |
| PARA-Drive (re-scored) | No | 0.25 | 0.46 | 0.74 | **0.48** | 0.14 | 0.23 | 0.39 | 0.25 | PARA-Drive Tab. 8 |
| **AD-MLP** (no perception) | **Yes** | 0.20 | 0.26 | 0.41 | **0.29** | 0.17 | 0.18 | 0.24 | **0.19** | AD-MLP paper |
| ⚠️ **GT human trajectory** | — | — | — | — | — | 1.02 | 0.96 | 0.91 | **0.96** | PARA-Drive Tab. 8 |

⚠️ AD-MLP's released checkpoint is separately reported as leaky: **[PUBLISHED — PARA-Drive fn. 5]**
*"the released model checkpoint along with the data file is trained with GT data leakage"*
(ref. `E2E-AD/AD-MLP` issue #4). PARA-Drive re-implemented it rather than using the release.

### 7.4 `sparsedrive`

| method | ego status | L2 1 s | 2 s | 3 s | Avg | Col 1 s | 2 s | 3 s | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FF (LiDAR) | — | 0.55 | 1.20 | 2.54 | 1.43 | 0.06 | 0.17 | 1.07 | 0.43 |
| EO (LiDAR) | — | 0.67 | 1.36 | 2.78 | 1.60 | 0.04 | 0.09 | 0.88 | 0.33 |
| ST-P3 | No | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| UniAD † | No | 0.45 | 0.70 | 1.04 | 0.73 | 0.62 | 0.58 | 0.63 | 0.61 |
| VAD † | Yes | 0.41 | 0.70 | 1.05 | 0.72 | 0.03 | 0.19 | 0.43 | 0.21 |
| SparseDrive-S | Yes | 0.29 | 0.58 | 0.96 | **0.61** | 0.01 | 0.05 | 0.18 | **0.08** |
| SparseDrive-B | Yes | 0.29 | 0.55 | 0.91 | **0.58** | 0.01 | 0.02 | 0.13 | **0.06** |

† reproduced by the SparseDrive authors with official checkpoints.
⚠️ Note UniAD's collision jumps to 0.61 here — SparseDrive re-adds the pedestrians UniAD's own
protocol excludes (§5.3).

### 7.5 `bevplanner` — see the full table in §6.2

Headline rows: UniAD (no ego) **1.03** / 0.77 · VAD-Base (no ego) **1.25** / 1.09 ·
BEV-Planner (no ego) **0.55** / 0.59 · Ego-MLP (ego only) **0.35** / 0.37 ·
GoStright **0.83** / 1.08.

### 7.6 `paradrive-std` — see the full table in §6.5

Headline rows on full val: PARA-Drive (no ego) L2 Ave_all **0.5574** ·
AD-MLP (ego only) **0.5568** · VAD **0.7830** · UniAD **0.8317**.

---

## 8. Data requirements

| item | fact | evidence |
|---|---|---|
| Scenes | **1 000**, each **20 s** | PUBLISHED — arXiv 1903.11027 |
| Keyframes (samples) | **~40 000**, annotated at **2 Hz** | PUBLISHED — 1903.11027 |
| 3D box annotations | **1.4 M** | PUBLISHED — 1903.11027 |
| Sensors | **6 cameras, 5 radars, 1 lidar** (cameras 12 Hz, radar 13 Hz, lidar 20 Hz raw) | PUBLISHED — 1903.11027 |
| Annotation classes | **23** → **10** detection classes → **7** tracking classes | PUBLISHED — 1903.11027 |
| Planning val samples | **6 019** (implementations keep ~5 119 valid) | PUBLISHED — BEV-Planner |
| Devkit | `pip install nuscenes-devkit`; **prediction + tracking need a source checkout** | MEASURED-from-source — devkit README + `docs/installation.md` |
| Python | *"tested for Python 3.9 and Python 3.12, but we recommend to use Python 3.12"* | MEASURED-from-source — `docs/installation.md` |
| Expansions | **CAN bus expansion**, **map expansion**, nuImages, nuScenes-lidarseg, Panoptic nuScenes | MEASURED-from-source — devkit README |
| **Licence** | *"The nuScenes data is published under CC BY-NC-SA 4.0 license, which means that anyone can use this dataset for non-commercial research purposes."* | PUBLISHED — arXiv 1903.11027, verbatim |

### 8.1 CAN bus expansion — what it actually contains

**[MEASURED-from-source — `python-sdk/nuscenes/can_bus/README.md`]**

| message | rate | key fields |
|---|---|---|
| `pose` | 50 Hz | position, **velocity (ego frame)**, **acceleration**, orientation, **rotation rate** |
| `imu` | 100 Hz | linear acceleration, quaternion, rotation rate |
| `steeranglefeedback` | 100 Hz | steering angle (rad, range [−7.7, 6.3]) |
| `vehicle_monitor` | 2 Hz | vehicle speed, **yaw rate (deg/s)**, steering angle/speed, brake, throttle, gear, RPM |
| `zoesensors` | 794–973 Hz | brake / steering / throttle sensor values |
| `zoe_veh_info` | 100 Hz | wheel speeds, longitudinal + transversal acceleration, torque, odometry |
| `route` | derived | 2D coordinates of the vehicle path, ~50 m before/after the scene |

⛔ **This is the exact channel that produces the ego-status shortcut in §6.** Under TanitAD's binding
vision-only rule, **the CAN bus expansion is a LABEL source, not an inference input.**

### 8.2 Map expansion

Required for `OffRoadRate` (prediction) and for any drivable-area / curb / lane metric
(BEV-Planner's CCR, PARA-Drive's off-road / off-lane). ⚠️ Not needed for detection or tracking.

### 8.3 UNVERIFIED data facts

- ⚠️ **Download sizes in GB** for full trainval, mini, CAN bus expansion, map expansion.
  `nuscenes.org` is a client-rendered SPA and returns no fetchable text; the devkit docs, the paper
  and the AWS Open Data registry page all omit sizes. **Marked UNVERIFIED — do not quote a size.**
- ⚠️ **Commercial licence status.** The paper says CC BY-NC-SA 4.0 / non-commercial research
  (quoted above, primary). The **AWS Open Data registry page for `motional-nuscenes` lists the
  licence field as "Commercial"** and points at `nuscenes.org/terms-of-use`, which could not be
  fetched. **These two are in conflict.** For TanitAD purposes treat nuScenes as
  **RESEARCH-ONLY (non-commercial)** until Sayed obtains a written commercial licence. Do not build
  a product deliverable on it.
- ⚠️ Train/val/test **scene counts** (commonly cited as 700/150/150) — not verified from a primary
  source in this pass.
- ⚠️ Prediction split **instance-sample-pair counts** (§3).

---

## 9. ⭐ Is nuScenes open-loop planning admissible as a TanitAD criterion?

### 9.1 Verdict

> **NO — not as a criterion, not as a gate, and not as evidence of driving skill.**
> **YES — as a tagged external-comparability row, under six hard conditions (§9.3).**
>
> The nuScenes open-loop planning protocol is **the field's canonical example of the exact defect
> TanitAD's binding rules exist to prevent**. Adopting it as a criterion would import, in one step,
> both the ego-status leak the PI forbade on 2026-08-03 and the route-head echo we measured and
> retracted internally.

**This is not a soft finding and it should not be softened to make adoption easier.**

### 9.2 The six findings that force the verdict

Each is MEASURED or PUBLISHED from a banked primary source. None is inherited.

1. **There is no protocol to adopt.** No nuScenes planning challenge, no devkit eval code, no
   leaderboard. Adopting "the nuScenes planning benchmark" means adopting *one research group's
   codebase* and calling it a standard.
2. **The convention outweighs the model.** The averaging switch alone moves the same VAD checkpoint
   from **0.72 → 1.22 m (+70 %)**, and the UniAD-vs-VAD ranking **flips sign** between the two legacy
   protocols. *(CLAUDE.md class: the `overlapping_holdout_se` sign-flip.)*
3. **The collision metric's noise floor exceeds the signal.** The **ground-truth human trajectory**
   scores 0.36 % (UniAD protocol) / **0.96 % (VAD protocol)** collision — worse than the models it is
   the target for. Every published collision number under the legacy protocols is unquotable.
4. **The task is solved without perception.** AD-MLP: no camera, no LiDAR, **L2 avg 0.29** vs UniAD's
   1.03. Under a corrected protocol AD-MLP (0.5568) **ties** a full perception stack (0.5574).
   A hard-coded straight line beats UniAD-without-ego.
5. **Two label leaks are standard practice in the input.** Ego status is worth 55–70 % of L2; and the
   "high-level command" is literally `ego_fut_trajs[-1][0]` — **the ground-truth future — thresholded
   at ±2 m**. *(This is our route-head echo, published as SOTA.)*
6. **The distribution hides all of it.** 73.9–75 % straight driving; only **686 of 6 019** val
   frames are non-"keep forward". A full-val average is ~89 % constant-velocity extrapolation, and
   the shortcut's failure (**6.7× worse collision on the targeted subset**) is invisible there.

**And it does not predict what we care about.** **[PUBLISHED — NAVSIM, verbatim]** *"Displacement
metrics are not correlated to closed-loop driving."* TanitAD has measured the same thing
independently: open-loop ADE **0.45 m** → closed-loop **1.69 m**.

### 9.3 If we report it anyway — the six binding conditions

Reporting it *for external comparability* is legitimate: reviewers will ask, and "we never ran a
community benchmark" is a real weakness. But it ships with a harness, not bare.

1. ⛔ **The protocol tag is part of the number.** Every row is stamped
   `nuscenes-planning/{uniad-noavg | stp3-temavg | bevplanner | paradrive-std | sparsedrive}`.
   **An untagged L2 or collision number is INADMISSIBLE** — mechanically the same rule as
   CLAUDE.md's `heldout` vs `full_set` stamp.
2. ⛔ **Publish the GT-trajectory control on the same protocol, in the same table.** If the human
   trajectory scores a nonzero collision rate, that is the instrument's false-positive floor, and no
   collision number within ~2× of it may be quoted. **This is the echo test, applied to a borrowed
   instrument.**
3. ⛔ **TanitAD runs the vision-only arm only.** Our binding rule forbids ego state at inference, so
   we **structurally cannot produce the configuration that generates the headline leaderboard
   numbers**. ⇒ Our number will look worse than UniAD's 0.46 and VAD's 0.37. **That is correct and
   must be stated, not apologised for.** Compare only against the `Ego in BEV ✗ / Ego in Planner ✗`
   rows: **UniAD 1.03 · VAD-Base 1.25 · BEV-Planner 0.55 · ST-P3 2.65.**
4. ⛔ **The high-level command must be refused or replaced.** Feeding `gt_ego_fut_cmd` imports a
   GT-derived input and violates the 2026-08-03 goal-input rule. Either run command-free and say so,
   or supply a **predicted** goal — which is what the PI's rule already prefers, and which sidesteps
   this by construction.
5. ⛔ **Report the targeted non-straight subset (686 frames) alongside full val, or not at all.**
   Full val alone cannot distinguish a driving policy from an extrapolator — measured, §6.5.
   Report the `n` for both.
6. ⛔ **The four families still bind.** L2 + collision is at best a partial **LATERAL** + weak
   **TACTICAL** proxy. **LONGITUDINAL** (target speed, headway / time-gap / TTC) and **STRATEGIC** are
   not covered by this protocol at all, and must be reported from our own instruments or **explicitly
   REFUSED with a reason and its `n`** — the registry's admissible-refusal path, never silent absence.
   Add **map compliance** (off-road / off-lane, or BEV-Planner's CCR), since that is the one place
   the ego-only shortcut is visibly punished on full val (AD-MLP off-road **1.21** vs PARA-Drive **0.12**).

### 9.4 What to adopt instead / in addition

- ✅ **Adopt nuScenes DETECTION, TRACKING and PREDICTION without reservation.** They have official
  devkit-enforced protocols, a real leaderboard, no ego-status pathway into the metric, and — the
  decisive point — **their labels are not functions of the model's inference inputs.** They pass the
  leak test in `leak_guards.vision_only_inference` cleanly. These are the rows that make "we ran a
  community benchmark" true.
- ⭐ **Prefer NAVSIM / PDMS as the external *planning* benchmark.** It was built specifically to fix
  what is broken here: non-reactive simulation, a score composed of No-Collision, Drivable-Area
  Compliance, Ego Progress, TTC and Comfort rather than imitation distance. Sibling stream:
  `products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md`.
- ⭐ **The strongest use of nuScenes planning for us is adversarial, not competitive.** Run
  **our own** GoStright / Ego-MLP controls against **our own** vision-only arm on **our own** corpus.
  If a constant-velocity extrapolator matches a TanitAD arm on any metric, that metric is not
  measuring driving. That instrument is cheap, needs no GPU, and is worth more to the programme than
  a leaderboard row.

### 9.5 Registry action (ESCALATION — needs wiring in)

`CRITERIA_REGISTRY.json` → `benchmarks.nuscenes` is `pending: true` with `criteria: []` and already
points at this file. **It should stay `pending` for planning**, and the split is deliberate:

- **`benchmarks.nuscenes.detection` / `.tracking` / `.prediction`** — ready to be filled and enforced
  from §2–§4 of this document.
- **`benchmarks.nuscenes.planning`** — should be added as an **external-comparability-only** entry
  that is **never `required: true`**, carrying a mandatory `protocol_tag` key and a mandatory
  `gt_control` key (condition 2 above).

⚠️ Per the skill's §6.5 rule, a registry change ships **with its regression arm** in
`tools/tests/test_criteria_check.py` in the same commit. **I have not edited the registry** — that is
outside this brief's deliverable and would land untested. **Escalating to the EvalFlyWheel owner.**

---

## 10. What is UNVERIFIED in this document

| # | claim | status |
|---|---|---|
| 1 | nuScenes download sizes (GB) for trainval / mini / CAN bus / map expansion | **UNVERIFIED** — no fetchable primary source; `nuscenes.org` is a JS SPA |
| 2 | Prediction split instance-sample-pair counts (train / train_val / val) | **UNVERIFIED** — absent from devkit README, `splits.py` and the tutorial |
| 3 | Train/val/test scene counts (700/150/150) | **UNVERIFIED** — not confirmed from a primary source in this pass |
| 4 | Commercial licence status | **CONFLICT** — paper says CC BY-NC-SA 4.0 non-commercial; AWS Open Data registry lists "Commercial". Treat as research-only |
| 5 | ST-P3's *paper* does not itself state the L2 averaging convention | The TemAvg attribution rests on (a) UniAD's config comment naming `stp3` as average-up-to-T, (b) VAD's copy of `metric_stp3.py`, (c) PARA-Drive Eq. 1. Three independent primaries agree; ST-P3's own prose is silent |
| 6 | Whether UniAD's `planning_evaluation_strategy` switch existed at the time of the CVPR 2023 publication | **UNVERIFIED** — current `main` is quoted; git history not checked. UniAD's published Table 7 numbers are consistent with `"uniad"` (at-timestep) |
| 7 | Tracking class rationale ("removing static objects" → 7) is internally inconsistent w.r.t. `construction_vehicle` | Flagged; the enumerated list is authoritative |
| 8 | BEV-Planner's per-timestep UniAD row (0.59/1.01/1.48) does not match UniAD's own (0.48/0.96/1.65) despite an identical average of 1.03 | **UNEXPLAINED.** Their valid-sample set (5 119) differs. Do not treat BEV-Planner's per-timestep values as interchangeable with UniAD's |
| 9 | AMOTP's exact per-frame distance normalisation | Quoted from the devkit README summary; the full `eval/tracking/algo.py` was not read line-by-line. Read it before implementing |
| 10 | PARA-Drive's standardized evaluation code release | The paper states it "will be released"; not verified as available |

---

## 11. Banked primary sources

All banked via `tools/kb_add.py`; verify with `python tools/kb_add.py --verify`.
Index: `TanitAD Research Lab/Library/LIBRARY.md`.

| key | paper |
|---|---|
| `1903.11027` | nuScenes: A multimodal dataset for autonomous driving |
| `1907.03961` | 3D Multi-Object Tracking: A Baseline and New Evaluation Metrics (AB3DMOT — origin of AMOTA/AMOTP) |
| `2207.07601` | ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning |
| `2212.10156` | Planning-oriented Autonomous Driving (UniAD) |
| `2303.12077` | VAD: Vectorized Scene Representation for Efficient Autonomous Driving |
| `2305.10430` | Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes (AD-MLP) |
| `2312.03031` | Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving? (BEV-Planner) |
| `2405.19620` | SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation |
| `2406.15349` | NAVSIM |
| `paradrive-cvpr2024` | PARA-Drive: Parallelized Architecture for Real-time Autonomous Driving (CVPR 2024) — local bank, not on arXiv |

**Source code read directly (primary, not banked as files):**

- `nutonomy/nuscenes-devkit` — `eval/detection/{README.md, algo.py, data_classes.py, configs/detection_cvpr_2019.json}`,
  `eval/prediction/{README.md, metrics.py, splits.py, configs/predict_2020_icra.json}`,
  `eval/tracking/README.md`, `can_bus/README.md`, `docs/installation.md`
- `OpenDriveLab/UniAD` — `projects/configs/stage2_e2e/base_e2e.py`,
  `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py`,
  `projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py`
- `OpenDriveLab/ST-P3` — `stp3/metrics.py`
- `hustvl/VAD` — `projects/mmdet3d_plugin/VAD/planner/metric_stp3.py`,
  `tools/data_converter/vad_nuscenes_converter.py`
- `NVlabs/BEV-Planner` — `mmdet3d/models/fbbev/planner_head/metric_stp3.py`
- `E2E-AD/AD-MLP` — `README.md`
