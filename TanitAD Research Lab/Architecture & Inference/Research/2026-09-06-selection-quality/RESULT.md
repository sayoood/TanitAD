# SELECTION QUALITY — the three refcv4b video observations, answered

**Arm:** `refcv4b-b1-v72-40k`, `ckpt_40284_FINAL.pt`, md5 **`99b573e8277d94a5e3bfbf630cb4d751`**
— verified against the pod's own md5 file **and** against `LANDING_RESULT.md`'s published md5.
**Tier:** T1 (self-action **OPEN** loop; the model consumes no actions — PI ruling 2026-09-02).
**Evidence class:** MEASURED (ours) unless a line says otherwise.
**Estimator:** episode-cluster bootstrap (`taniteval/ci.py`), paired where two arms share windows.
⛔ `overlapping_holdout_se` is not used anywhere in this package.

**Corpus / grid.** 141 episodes, local v2 cache `C:/Users/Admin/refav1_eval_full/eps`; labels
`s2_labels_v7.2_eval.jsonl.gz` md5 **`aa12c948f062181c3297265b51526ec5`** — the exact blob the
published refcv4b T1 record used. Rolled **at stride 1** (24,114 windows, 0.1 s apart) because the
PI's evidence for observation (6) is *frames 032 and 034 of clip `73e750eb`*, **0.2 s apart**, and
the published grid is stride 5 (0.5 s) and structurally cannot see it. The published
4,823-window grid is the `ws % 5 == 2` subset of this one, and is used as a cross-check.

**Compute.** Dev-box RTX 4060 only. ⛔ The A40 (refcv5 to 2026-09-08) was not touched.

---

## 0. Provenance correction — the "banked refcv4b T1 dump" is not refcv4b

The dump reachable on local disk (`.../8e7cfa33-.../dump40284/refcv3_40284_dump`, referenced by
`taniteval/results/refcv3-40284-stratified.json`) is **`refcv3-b1-v72-30k` @ 40284**, not refcv4b.
Proven positively rather than by filename: recomputing the kin3 tactical confusion from that
dump's own `g`/`os` with the labeller the published block names gives

| | lane_keep | turn_left | turn_right |
|---|---|---|---|
| **GT row sums, recomputed** | 4176 | 251 | 396 |
| **GT row sums, published** | 4176 | 251 | 396 |
| **prediction, recomputed** | 4057 / 37 / 82 … | | |
| **prediction, published** | 4067 / 36 / 73 … | | |

— the **ground truth matches exactly** (same windows, same corpus) while the **predictions do
not** (accuracy 0.9540 vs 0.9583, longitudinal 0.7477 vs 0.8258). Same grid, different
checkpoint. refcv4b's own dump lives only at `pod:/workspace/eval/refcv4b_t1_dump`.

⇒ **The refcv4b arm was re-rolled locally** from the md5-verified checkpoint. Every refcv4b number
below comes from that roll.

---

## 1. Observation (12) — LANE CHANGES: it is a **LABEL** defect first, a **VOCABULARY** defect
second, and neither a loss nor a learning failure

### 1.1 The label prevalence, printed beside every recall it will accompany

`a_tac.lat` is the target of the v7.2 8-way tactical **lateral action** head.

| `a_tac.lat` | v7.2 **TRAIN** blob `0ff902130ce76886b8a925eceed9e3a5` | v7.2 **EVAL** blob `aa12c948f062181c3297265b51526ec5` |
|---|---|---|
| LANE_KEEP | 2958 / 4572 | 99 / 147 |
| **LANE_CHANGE_L** | **0 / 4572** | **0 / 147** |
| **LANE_CHANGE_R** | **0 / 4572** | **0 / 147** |
| **ABORT_LC** | **0 / 4572** | **0 / 147** |
| NUDGE_L | 488 / 4572 | 14 / 147 |
| NUDGE_R | 592 / 4572 | 21 / 147 |
| TURN_L | 275 / 4572 | 5 / 147 |
| TURN_R | 259 / 4572 | 8 / 147 |

**Read-control:** 5 of the 8 rows are non-zero, so the field was really read; a 0 here is an
absence, not a failed read. (The first version of this probe counted `g_tac.tokens`, a key that
does not exist, and produced an all-zero goal table that looked exactly like a finding. Every
vocabulary in this package is therefore printed **in full**, never only the rows of interest.)

The same shape appears on the longitudinal axis: `YIELD_MERGE` is **0 / 4572** while the other
seven classes run 131–1243. That class is used below as an **independent control**.

### 1.2 The mechanism, from source: the emitter cannot express a lane change

`stack/scripts/s2_geom_emit_v7.py :: tactical_actions()` — the function that mints `a_tac.lat` —
has exactly four assignment sites, and its reachable set is
`{LANE_KEEP, NUDGE_L, NUDGE_R, TURN_L, TURN_R}`: **five of the eight frozen tokens**. There is no
branch, under any input, that can emit `LANE_CHANGE_L`, `LANE_CHANGE_R` or `ABORT_LC`. The tokens
exist in the frozen vocabulary and size the head's 8-logit output, but the label generator can
never produce a positive example of them.

The underlying kinematic classifier says so itself. `stack/tanitad/data/ego_manoeuvre.py` calls
`NUDGE` on `abs(lat_peak_m) >= NUDGE_LAT_M` (1.0 m) and its own docstring records that **NUDGE has
NO UPPER BOUND** and that `lat_peak_m` is *"the only field that can distinguish a 1 m wobble from an
absorbed lane change."* A lane change is, by construction, filed as a NUDGE.

### 1.3 The corpus does contain lane changes — three layers disagree with each other

| layer | LANE-CHANGE evidence, v7.2 TRAIN (4,572 clips) |
|---|---|
| VLM chain-of-thought (`cot_tokens.lane_change` non-null) | **167 / 4572** |
| tactical **GOAL** set (`g_tac.goals` ∋ LANE_CHANGE_L/R) | **38 / 4572** (L 23, R 15) |
| tactical **ACTION** (`a_tac.lat`) | **0 / 4572** |

**Read-control:** all 22 goal tokens are non-zero (FOLLOW_LANE 3629, SPEED_BAND 4572,
GAP_TARGET 368, MERGE 79, TAKE_EXIT_R 128, OVERTAKE_VEHICLE 20 …), and 10 of the 11 `cot_tokens`
fields are non-null somewhere, so these are genuine counts.

On the **38** clips whose goal layer literally says LANE_CHANGE, the emitted action is
`NUDGE_L` **15** / `NUDGE_R` **23** — **38 of 38 NUDGE, 0 LANE_CHANGE, 0 anything else.**

⚠️ **What is NOT usable, and why.** The persisted `a_tac.lat_args.lat_peak_m` looked like the way
to size "how much NUDGE mass sits at lane width". It is **INCONCLUSIVE and must not be quoted**:
its values run to ±305 m, and it carries the WRONG SIGN on NUDGE clips (`NUDGE_L` with
`lat_peak_m = −24.517`), which contradicts the class rule `NUDGE_{L if lat>0 else R}` that
`ego_manoeuvre.py` decides on. The blob's `horizon.available_s` is 35.0 s, so the persisted value is
a **whole-horizon lateral excursion**, not the per-decision band value the NUDGE gate uses. Two
probes for the writer (`grep lat_peak_m` across `stack/scripts`, `stack/tanitad/data`,
`stack/tanitad/lake`; and the current emitter's own `lat_args` construction, which is
`{"within_m": …}` only) found no site in the present tree that writes it. **A work item, not a
finding.**

### 1.4 The head: three logits that never received a gradient

Predicted from the label census **before** the weights were opened: a class with zero positives is
never a target, so if the trainer masks the loss those rows are untouched.

`lat_head_tac.weight` (8 × 512) row norms:

| class | ‖w_row‖ | class | ‖w_row‖ |
|---|---|---|---|
| LANE_KEEP | 0.6784 | NUDGE_L | 0.7875 |
| **LANE_CHANGE_L** | **1.5562** | NUDGE_R | 0.7230 |
| **LANE_CHANGE_R** | **1.5595** | TURN_L | 0.8454 |
| **ABORT_LC** | **1.5461** | TURN_R | 0.8394 |

`lon_head_tac.weight` — the **independent control**, a different head and a different class named
in advance: `YIELD_MERGE` **1.5695** against 0.6281–0.7665 for the seven occurring classes.

**4 of 4 zero-positive classes sit in 1.546–1.570; 12 of 12 occurring classes sit in
0.628–0.845. No overlap.**

The direct discriminator is in the checkpoint's own optimizer state (AdamW, step 40,127).
Per-row `sum(exp_avg_sq)`:

| head | rows | value |
|---|---|---|
| `lat_head_tac.weight` rows 1,2,3 = **LANE_CHANGE_L / LANE_CHANGE_R / ABORT_LC** | dead | **4.3e-15 … 1.1e-14** |
| `lat_head_tac.weight` rows 0,4,5,6,7 | live | 4.5e-04 … 8.5e-03 |
| `lon_head_tac.weight` row 2 = **YIELD_MERGE** | dead | **1.2e-15** |
| `lon_head_tac.weight` other rows | live | 3.9e-05 … 1.2e-02 |

A ratio of **~10¹¹–10¹²**, with the same pattern in both bias tensors (1e-17/1e-18 vs 1e-7).
⇒ **`p.grad` was effectively `None` for exactly the four classes the labels never contain, and for
no others.** The loss did not "under-weight" lane changes; there was nothing to weight.

### 1.5 The anchor-expressibility test — can any of the 117 candidates express a lane change?

⛔ Model-free. The fan is a deterministic function of `(anchor_controls, v0)`:
`refc.py::RefCDecoder._anchor_bank` builds `ctrl[b,n,k,:] = (a_lon[n], kappa[b,n])` with **the same
control at every step k**, `kappa[b,n] = clamp(a_lat[n] / max(v0_b, 4.0)², ±0.12)`, integrated by
`kinematic.rollout_unicycle`. `anchor_controls` was read **out of the checkpoint**
(`model:core.decoder.anchor_controls`) and is **byte-identical** to the sibling turn-coverage
package's banked `anchors_live_refcv4b.pt['controls']` (whose own record says
`sha256_matches_live_pod_anchors: true`). Grid: **13 `a_lon` × 9 `a_lat` = 117**, `a_lat` ∈
{−3.0 … +3.0} m/s² (units from `refc.py`'s `anchor_controls` doc; the run's argv carries
`--anchor-control-units alat`).

**Controls that read known values.** Straight candidate #4 (`a_lat = 0`): `max|y(6 s)| = 0.000e+00`
and `max|dyaw(6 s)| = 0.000e+00`, exactly. Closed form for `a_lat = −3.0` at `v0 = 10 m/s`:
`|dyaw(6 s)| = 103.132°` predicted, **103.132°** integrated.

**The structural answer.** Over 4,823 windows × 117 candidates = 564,291 pairs, the fraction whose
yaw is **monotone** over 0–6 s is **1.000000**. Every candidate departs its heading and never
returns. A completed lane change is by definition non-monotone in heading — out, then back — so
**the selection vocabulary cannot express one.**

**The approximate answer, which is the honest refinement.** A *shallow arc* can still place the
vehicle a lane width across:

| horizon | LANE-CHANGE BOX \|lat\| ∈ [2.5, 5.0) m ∧ \|dyaw\| ≤ 10° | windows with ≥ 1 such candidate | TURN box \|dyaw\| ≥ 30° (control) |
|---|---|---|---|
| 6.0 s | 5,648 / 564,291 | 1,748 / 4,823 (36.2 %) | 265,786 / 564,291 |
| 4.0 s | 13,226 / 564,291 | 1,756 / 4,823 (36.4 %) | 213,012 / 564,291 |

and among **every** pair that does reach ≥ 2.5 m of lateral offset at 6 s, the **minimum** residual
heading is **4.80°** (p05 11.00°, median 44.36°). ⇒ the vocabulary can approximate the
displacement, always as an arc that is **still turning at the end**, and 63.8 % of windows have no
candidate that can do even that.

### 1.6 One line

**The lane-change absence is a LABEL defect (the emitter has no branch for it, so 0 / 4,572
positives and 3 dead logits), compounded by a VOCABULARY defect (117 constant-curvature arcs,
yaw monotone on 100.0000 % of pairs, so no candidate is an S-shape). It is NOT a loss-weighting
failure and NOT a learning failure — the head was never given a single example to learn from.**

---

## 2. The environmental half of observation (2) — there is no lane geometry to score against

Three independent probes, each printing its container in full so the non-lane entries are the
control that the container was really read:

1. **The episode artifact the planner consumes** (`*.v2ep.pt`): 14 keys —
   `jpeg_buf, jpeg_len, actions, poses, n_stack, image_size, episode_id, clip_id, quality,
   image_h, image_w, frame, projection_mode, codec`. Keys matching
   `lane|map|road_edge|boundary|marking|polyline|centerline|drivable|curb|xodr|graph`: **none**.
2. **The v7.2 label record** (22 top-level keys). The only three matches anywhere in the record are
   `cot_source.meta_action.lane = "Lane Keep"` (VLM text), `cot_tokens.lane_change` (a nullable
   text flag) and the goal token `FOLLOW_LANE`. All three are **language**, none is geometry.
3. **The pinned feature read-set** (`stack/tests/test_physicalai_feature_readset.py`, asserted
   against source): 2 / 5 / 6 features by layer — `egomotion`, `camera_front_wide_120fov`,
   `camera_intrinsics`, `sensor_extrinsics`, `vehicle_dimensions`, `obstacle.offline`. **None is a
   lane, map, road-edge or drivable-area feature**, and `obstacle.offline`'s enum is 10 dynamic
   agent classes.

⇒ **"the selected path cuts road marks" cannot be measured on this corpus.** The only environmental
signal that exists is `obstacle.offline` (dynamic agents), which supports headway/collision terms
and cannot see a painted line. **Reported as a gap; no proxy invented.**

### 1.7 What refcv4b actually emits — the recall, beside the prevalence

Local stride-1 roll, **24,114 windows / 141 episodes**, tier **T1**.

**Emission over all 24,114 windows** (v7.2 tactical LATERAL head):

| class | emitted | | class | emitted |
|---|---|---|---|---|
| LANE_KEEP | 20,276 (84.0839 %) | | NUDGE_L | 819 (3.3964 %) |
| **LANE_CHANGE_L** | **0 (0.0000 %)** | | NUDGE_R | 1,622 (6.7264 %) |
| **LANE_CHANGE_R** | **0 (0.0000 %)** | | TURN_L | 351 (1.4556 %) |
| **ABORT_LC** | **0 (0.0000 %)** | | TURN_R | 1,046 (4.3377 %) |

**Agreement on the 5,781 windows that carry a label** (23.97 % of the grid; elsewhere the label is
`IGNORE_ID = −100`):

| class | n_true | n_pred | recall | precision |
|---|---|---|---|---|
| LANE_KEEP | 3,895 | 4,918 | 0.9340 | 0.7397 |
| **LANE_CHANGE_L** | **0** | **0** | **undefined** | **undefined** |
| **LANE_CHANGE_R** | **0** | **0** | **undefined** | **undefined** |
| **ABORT_LC** | **0** | **0** | **undefined** | **undefined** |
| NUDGE_L | 533 | 166 | 0.1839 | 0.5904 |
| NUDGE_R | 820 | 324 | 0.2585 | 0.6543 |
| TURN_L | 205 | 139 | 0.5463 | 0.8058 |
| TURN_R | 328 | 234 | 0.3598 | 0.5043 |

⛔ **The lane-change recall is UNDEFINED, not zero.** `n_true = 0`. A "0.0000 recall" line for these
classes would be a category error — there is no denominator. Read-control: 5 of 8 classes have
`n_true > 0` and 5 of 8 have `n_pred > 0`, and **they are the same five**.

On the **855 windows whose clip the CoT / goal layer flags as a lane change** (5 of the 141 eval
clips), the model emits `LANE_KEEP` 810, `NUDGE_L` 42, `NUDGE_R` 3 — **zero lane changes** — and the
v7.2 label there is `IGNORE` 650, `LANE_KEEP` 164, `NUDGE_L` 41. Model and label agree, and both
are wrong about the same thing.

### 1.8 The runner-up margin — the lane-change logits are 33 apart, never in the top five

The brief's second candidate ("present but never argmax"). Raw `lat_head_tac` / `lon_head_tac`
logits captured with a forward hook on the **unmodified** pipeline, n = 855 rows / 5 episodes:

| LATERAL class | mean logit | n argmax | mean rank | **best rank ever** | mean margin to argmax |
|---|---|---|---|---|---|
| LANE_KEEP | +1.6489 | 805 | 1.06 | 1 | 0.0416 |
| **LANE_CHANGE_L** | **−31.5622** | **0** | 7.59 | **6** | **33.2526** |
| **LANE_CHANGE_R** | **−31.3805** | **0** | 6.81 | **6** | **33.0710** |
| **ABORT_LC** | **−31.3355** | **0** | 6.60 | **6** | **33.0260** |
| NUDGE_L | −1.0014 | 1 | 3.09 | 1 | 2.6919 |
| NUDGE_R | +0.0390 | 13 | 2.32 | 1 | 1.6515 |
| TURN_L | −5.1831 | 0 | 4.57 | 3 | 6.8736 |
| TURN_R | −4.6781 | 36 | 3.96 | 1 | 6.3685 |

Independent control, the other head: `YIELD_MERGE` (its only zero-positive class) mean logit
**−30.8959**, **best rank 8 — always last**, margin **33.5902**; every occurring longitudinal class
sits between −8.90 and +1.71.

⇒ **The four zero-positive classes share a −31 logit dead-zone that no occurring class comes near**
(the next worst is `HOLD` at −8.90). They are not narrowly losing an argmax; they are 33 logits away
and never enter the top five. **"Present but never argmax" is REFUTED for the lane-change classes.**

⚠️ **But it is TRUE for a different class, and that is a separate defect worth its own work item.**
`FOLLOW` has **410 true labels** on the eval grid, is emitted **0 / 24,114 times**, recall
**0.0000** — and its mean logit is **−0.0072** with a **best rank of 3** and a mean margin of only
**2.70**. That is the classic minority-class argmax failure (`refc_tactical`'s own F3), and its fix
— prior-corrected decoding, `refc_tactical.logit_adjust` — already exists in the programme.
It is NOT the lane-change mechanism and must not be conflated with it.

### 1.9 RULE ZERO — the diagnosis is not the product: how big is the fix?

The cheapest experiment that decides whether repairing the emitter earns a retrain is to measure
how much lane-change behaviour the **ground truth actually contains**. Model-free, read straight off
the recorded ego poses over 0–6 s, no label pipeline and no checkpoint in the path.

Definitions declared first. `lat_T = y(6 s)`, `dyaw_T = yaw(6 s) − yaw(0)`, `dyaw_peak = max|dyaw(t)|`;
**RETURNS** ⇔ `|dyaw_T| ≤ 10°` **and** `dyaw_peak ≥ 3°` (the heading leaves and comes back — the
non-monotone shape §1.5 proved no candidate has).

| | n / 18,615 scoreable windows | |
|---|---|---|
| heading LEAVES AND RETURNS | **4,615** | 24.79 % |
| **LANE CHANGE strict** (`\|lat\| ∈ [2.5, 5.0) m` ∧ RETURNS) | **1,297** | **6.97 %** |
| LANE CHANGE relaxed (`\|lat\| ≥ 2.5 m` ∧ RETURNS) | **2,363** | 12.69 % |
| TURN box `\|dyaw\| ≥ 30°` — the **CONTROL**, must be non-zero | 2,602 | 13.98 % |
| **clips containing ≥ 1 relaxed lane-change window** | **79 / 141** | |

Controls: `max|dump v0 − poses[ws+2, 3]| = 0.000e+00`; 5,499 of 24,114 windows excluded for not
having a full 6 s of recorded future, **counted, never truncated**.

⇒ **The corpus demands a lane-change-shaped manoeuvre on 6.97 % of scoreable windows and in 79 of
141 clips. The label pipeline records ZERO of them and the vocabulary can express none of them.**
That is the size of the fix, and it is not marginal.

---

## 3. Observation (6) — TEMPORAL SELECTION STABILITY: the new instrument, floored and ceilinged

⛔ **Determinism established first, two ways**, because "jitter" attributed to a sampler is not a
model defect. (a) SOURCE: `refc.py`'s refinement loop is
`noise = randn_like(x) * noise_std if self.training else zeros_like(x)` — the noise is **zeroed
outside training**, and the stochastic WP-4 sampler that CANNOT be zeroed is refcv5's path
(`_loop_steps = 0 if self.control_head is not None`), not refcv4b's. (b) MEASURED: two separate
processes, same flags, same seed → `sel_idx`, `lat_pred_nav_true` and the **full `os` waypoint
array bit-identical, max|diff| = 0.0**. ⇒ **inference-run variance on this arm is exactly zero**, so
every switch below is the model reacting to its input.

**Definition.** Over consecutive windows of the same clip on the stride-1 grid (**0.1 s apart**), the
fraction of adjacent pairs at which the quantity CHANGES. Lower = more stable.
n = 23,973 adjacent pairs / 141 episodes; episode-cluster bootstrap.

| quantity | value | 95 % CI |
|---|---|---|
| **SELECTION switch rate — refcv4b** | **0.1552** | [0.1416, 0.1686] |
| **REFERENCE (the "ceiling"): the GT's own best-in-fan choice** | **0.0806** | [0.0732, 0.0885] |
| FLOOR-A: the arm's own picks SHUFFLED in time | 0.5736 | [0.5319, 0.6154] |
| FLOOR-B: a uniform random pick from the 117-fan | 0.9907 | [0.9894, 0.9919] |
| TACTICAL LATERAL token switch rate | 0.0282 | [0.0227, 0.0343] |
| TACTICAL LONGITUDINAL token switch rate | 0.0779 | [0.0682, 0.0881] |
| **WAYPOINT disagreement (m)** | **0.1302 m** | [0.1219, 0.1384] |
| **CEILING / control C3: the GT path against itself** | **0.0145 m** | [0.0129, 0.0161] |
| FLOOR: an independent random candidate each frame | 2.7142 m | [2.6350, 2.7866] |

⚠️ **What the "ceiling" is.** It is not an upper bound on achievable stability — a frozen selector
would score 0. It is **how much the correct choice would itself have to move**, computed by binding
the GT path against the SAME per-window v0-conditioned fan the model chooses from. Read that way:

* **refcv4b switches its selection 1.93× as often as the right answer changes.**
* Its plan moves **9.0×** as much frame-to-frame as the GT path does under the identical transform,
  which is **4.29 %** of the way from that ceiling to the random-candidate floor.

**Dwell times — this is the PI's observation, quantified.** 3,861 runs; mean 6.25 frames (0.62 s),
**median 2 frames = 0.2 s**, p90 14, max 172.

| run length | count | share |
|---|---|---|
| ≤ 1 frame (0.1 s) | 1,498 / 3,861 | **38.80 %** |
| ≤ 2 frames (0.2 s) | 2,110 / 3,861 | **54.65 %** |
| ≤ 3 frames (0.3 s) | 2,454 / 3,861 | 63.56 % |
| ≤ 5 frames (0.5 s) | 2,897 / 3,861 | 75.03 % |

⇒ *"the selection jumps between consecutive frames"* is **the median behaviour**: more than half of
all selection runs are gone within 0.2 s, and 38.8 % last a single frame. The PI's clip `73e750eb`
frames 032/034 are not an outlier.

**Does the jitter move the DECISION or only the waypoints?** Of the **3,720** selection switches:

| | n | share of switches |
|---|---|---|
| also moves a tactical token | **692** | **18.60 %** |
| waypoints only | 3,028 | 81.40 % |
| a token moves with **no** selection switch | 1,785 | — |

⇒ four in five switches are a waypoint wobble, but one in five is a genuine change of manoeuvre —
and 1,785 manoeuvre changes happen without the selection moving at all, i.e. the tactical head and
the selector are **not synchronised**.

**Association with error** (not causal — same-window, so the direction is not identified): `os` ADE
at a switching window **0.3903 m** (n = 3,720) vs **0.2800 m** at a holding window (n = 20,253);
selection regret 0.3534 vs 0.2029 m.

---

## 4. Observation (2) — a VALID selection regret, and the proof that its ceiling is not `a_star`

### 4.1 The ceiling construction, and why it is not the invalid one

`taniteval/tools/refcv3_arm.py` binds `a_star` against `anchors_bank =
model.core.decoder.anchors` — the **fixed bank rolled once at `ref_speed_ms`**, which is exactly the
binding the trainer forbids for a v0-conditioned vocabulary. This package binds instead against the
**per-window v0-conditioned emitted fan**, rebuilt model-free from `anchor_controls` + `v0` through
the programme's own integrator. **Demonstrated, not asserted:**

| | ADE, 24,114 windows, T1 |
|---|---|
| the dumped `a_star` candidate (the INVALID "ceiling") | **1.4491 m** |
| the SELECTED candidate `F[sel_idx]` | 0.4252 m |
| **our best-in-fan (the valid ceiling)** | **0.1993 m** |
| a uniform random candidate | 2.3399 m |
| the model's refined output `os` | 0.2970 m |

* The two argmins are **identical on only 9.52 %** of windows — they are different objects.
* `a_star` scores **worse than the arm it is supposed to bound** (1.4491 > 0.4252) and 7.3× worse
  than our ceiling. **Ours is ≤ every arm it bounds.** That is the definitional test, and it passes.
* **Control C5 (index mapping):** `ADE(F[sel_idx]) = 0.4252 m` vs `ADE(F[random]) = 2.3399 m` — a
  wrong `sel_idx → anchor_controls` mapping would read as the random draw. It does not.

### 4.2 The regret, per family, never pooled

`regret = cost(selected candidate) − cost(best candidate in the SAME emitted fan)`. This is the
**selection** regret: the decoder's free-waypoint refinement sits on top of it and is not
re-attributed here.

| family | n | selected | best-in-fan | **regret** | p50 | p90 | p99 | frac optimal | random-pick floor |
|---|---|---|---|---|---|---|---|---|---|
| LONG. `along_mae_m` | 24,114 | 0.3368 | 0.0713 | **0.2655** | 0.1589 | 0.6288 | 1.5911 | 0.1272 | 1.5814 |
| LONG. `speed_mae_mps` | 24,114 | 0.3462 | 0.1573 | **0.1890** | 0.0044 | 0.5542 | 1.5644 | 0.3043 | 1.6479 |
| LAT. `cross_mae_m` | 24,114 | 0.1725 | 0.0544 | **0.1181** | 0.0215 | 0.3795 | 1.1457 | 0.0443 | 1.2170 |
| LAT. `curv_mae_1pm` | 22,944 | 0.0042 | 0.0022 | **0.0020** | 0.0000 | 0.0029 | 0.0443 | 0.6450 | 0.0269 |
| LAT. `heading_mae_deg` | 24,114 | 1.9737 | 1.2238 | **0.7499** | 0.0000 | 2.2746 | 10.7753 | 0.6659 | 8.3204 |
| (ADE, beside them) | 24,114 | 0.4252 | 0.1993 | 0.2259 | 0.0273 | 0.6398 | 1.6227 | 0.4713 | 2.1406 |

**The distribution, not the mean, is the finding.** On ADE the selector is **exactly optimal on
47.13 %** of windows and its median regret is 0.0273 m — but p90 is **0.6398 m** and p99 is
**1.6227 m**. The PI's complaint is about the tail, and the tail is real: on roughly one window in
ten the fan contained a candidate more than 0.64 m better than the one taken.

**TACTICAL family** — ⛔ read on the **v2 CURVATURE gate** (`|kappa| ≥ 1/60`), never the
`|dyaw| > 0.15` v1 gate (the v1 read is given beside it only because the published block uses it).
GT lateral class counts: v2 [22,208 / 686 / 1,220] vs v1 [20,880 / 1,252 / 1,982].

| axis (v2 gate) | selected is WRONG | fan contained a CORRECT candidate | **wrong ALTHOUGH one was available** | median candidates carrying the right class |
|---|---|---|---|---|
| LATERAL | 3.82 % | 99.96 % | **910 / 24,114 (3.77 %)** | 77 of 117 |
| LONGITUDINAL | 18.05 % | 99.99 % | **4,350 / 24,114 (18.04 %)** | 27 of 117 |

(v1 gate, for comparability only: lateral 6.27 %, longitudinal 18.05 %.)

⇒ **This is observation (2) in its exact form.** On 4,350 windows the fan contained a candidate
carrying the correct longitudinal manoeuvre — a median of 27 of them — and the selector took one
that did not.

**STRATEGIC family: UNAVAILABLE** — no per-window strategic label exists on this grid. The v7.2
record carries ONE `g_str` token per CLIP, and the nav command is a MODEL INPUT (PI: it simulates
the vehicle's nav system), never a target. n = 0. A work item, not a pass.
**DISTANCE-KEEPING: UNAVAILABLE** — the banked b1 eval lead block is per-frame on a 0.2 s 10-step
grid while the scored fan is on the 0.5 s 4-step grid, and `taniteval.lead_metrics.distance_keeping`
loops per window in Python, so 117 candidates × 24 k windows is not runnable as written. Lead
coverage is **8,341 / 29,556 frames (28.22 %)**. n = 0. A work item, not a pass.

### 4.3 WHICH axis carries the defect — the lever ranking

The 117 candidates are a 13 × 9 grid of `(a_lon, a_lat)`. Comparing the selected cell to the
best-in-fan cell:

| | value |
|---|---|
| exactly right on **a_lon** | 13,181 / 24,114 (**54.66 %**) |
| exactly right on **a_lat** | 19,596 / 24,114 (**81.26 %**) |
| exactly right on both | 11,366 / 24,114 (47.13 %) |
| mean grid steps off on a_lon (max 12) | 0.675 |
| mean grid steps off on a_lat (max 8) | 0.264 |
| ADE with **a_lon oracle**, a_lat kept as chosen | **0.2577 m** |
| ADE with **a_lat oracle**, a_lon kept as chosen | 0.3780 m |
| ADE selected → best-in-fan | 0.4252 → 0.1993 m |

⇒ **The selection defect is LONGITUDINAL.** Fixing the `a_lon` index alone recovers
(0.4252 − 0.2577) / (0.4252 − 0.1993) = **74.2 %** of the total regret; fixing `a_lat` alone
recovers **20.9 %**. That ranks WP-7's target: **train the ranker on the longitudinal axis first.**

---

## 5. RULE ZERO — two levers run in the same turn, both pre-registered

### 5.1 A selection DEADBAND — REFUTED, and that is informative

Deployable by construction (reads only the arm's own selection history and the grid coordinates,
never the ground truth): hold the previous pick while the new one is within K grid steps on both
axes. Pre-registered: *if the small moves are noise the deadband LOWERS every family; if they are
the model tracking the scene it RAISES them.*

| K | switch rate (baseline 0.1552, GT ref 0.0806) | ADE Δ | along Δ | cross Δ | speed Δ | heading Δ |
|---|---|---|---|---|---|---|
| 1 | **0.0242** | +0.0486 [+0.0324, +0.0655] **sep** | +0.0307 **sep** | +0.0255 **sep** | +0.0366 **sep** | +0.1224 **sep** |
| 2 | 0.0106 | +0.1350 **sep** | +0.0818 **sep** | +0.0797 **sep** | +0.0939 **sep** | +0.5440 **sep** |

**FAILED, on every family, at both K, all separated.** ⇒ **the frame-to-frame selection movement is
NOT free noise.** Freezing it is strictly worse. The 0.1552 switch rate buys something real, even
though it is 1.93× the reference — so the fix for observation (6) is **not** hysteresis on the
selection.

### 5.2 An OUTPUT smoothing filter — the lever that works, on the LATERAL family only

`os_α(t) = (1−α)·os(t) + α·T[os(t−1)]`, where T is the exact rigid transform into frame t.
Zero GPU; paired episode-cluster bootstrap, n = 24,114.

| α | ADE | along | cross | speed | **curvature** | **heading** |
|---|---|---|---|---|---|---|
| 0.25 | +0.0008 (not sep) | −0.0001 (not sep) | +0.0009 **sep WORSE** | **−0.0028 sep BETTER** | **−0.0011 sep BETTER** | **−0.1166 sep BETTER** |
| 0.50 | +0.0080 **sep WORSE** | +0.0057 **sep WORSE** | +0.0034 **sep WORSE** | +0.0003 (not sep) | **−0.0020 sep BETTER** | **−0.1492 sep BETTER** |
| 0.75 | +0.0215 **sep WORSE** | +0.0177 **sep WORSE** | +0.0074 **sep WORSE** | +0.0097 **sep WORSE** | **−0.0026 sep BETTER** | **−0.1197 sep BETTER** |

⛔ **The LATERAL row read on the programme's own estimator, with its straight-line floor beside it**
(`ha0` = `a = 0, κ = 0` at the measured `v0`):

| arm | `curvature_mae_1pm` | `heading_mae_deg` | `cross_mae_m` |
|---|---|---|---|
| **`ha0` — the straight-line FLOOR** | **0.006841** | 2.8066 | 0.3133 |
| `os` — the arm | **0.008024** | 1.3198 | 0.0979 |
| `os` smoothed α = 0.25 | **0.007391** | 1.3081 | 0.0988 |
| `os` smoothed α = 0.50 | **0.006882** | 1.3088 | 0.1014 |

⇒ **α = 0.25 is a separated, essentially free lateral gain**: curvature MAE −0.0011 (−7.9 %, and
53 % of the way from the arm to the straight-line floor) and heading −0.1166° for +0.0009 m of
cross-track and no separated ADE change. α = 0.50 puts curvature **at** the straight-line floor
(0.006882 vs 0.006841) but costs a separated +0.0080 m ADE. **A knob for the PI, not a free win —
and it is an OUTPUT filter, so it does not fix the selection.**

⚠️ **Attribution.** A sibling traced refcv4b's curvature over-shoot to the **decoder's free-waypoint
refinement** (it halves ADE and doubles curvature error). This filter partially undoes that
refinement's lateral cost; the credit belongs there, **not** to selection. Stated so the two are not
conflated.

---

## 6. Cross-validation: the local roll IS the published arm

The published refcv4b T1 record scores the same checkpoint on the stride-5 grid. Restricting this
roll to `ws % 5 == 2` recovers those windows (4,927 here vs 4,823 published — this roll skips fewer
edge windows):

| arm | local `curvature_mae_1pm` | published | delta |
|---|---|---|---|
| `os` | **0.007975** | 0.008097 | **−0.000122** (−1.5 %) |
| `ha0` | **0.006846** | 0.006802 | **+0.000044** (+0.6 %) |

⇒ the locally re-rolled arm reproduces the published lateral headline to ~1.5 %. Every number above
is about the arm in the PI's video.

---

## 7. Verdicts, and what each one licenses

| PI observation | verdict | mechanism | next lever |
|---|---|---|---|
| **(12) lane changes never activate** | **CONFIRMED — a LABEL defect first, a VOCABULARY defect second** | `tactical_actions()` has no branch that can emit LANE_CHANGE ⇒ 0 / 4,572 positives ⇒ 3 logits with a dead Adam second moment and a −31 dead-zone; and 117 constant-curvature arcs, yaw monotone on 100.0000 % of pairs, cannot draw an S-shape | fix the emitter (`ego_manoeuvre` already persists the evidence), rebuild labels, and add a two-segment / sign-changing candidate family. **Demand: 6.97 % of windows, 79 / 141 clips.** ⛔ Needs a retrain — a PI/compute decision. |
| **(6) selection jumps between frames** | **CONFIRMED and quantified** | switch rate 0.1552 vs a 0.0806 reference; **median dwell 0.2 s, 38.8 % single-frame**; 18.6 % of switches move a manoeuvre token | ⛔ **NOT hysteresis** — the deadband is separated WORSE on every family. **Output smoothing α = 0.25** is a separated lateral gain and is available today. The structural fix is a ranker with a temporal term (WP-7). |
| **(2) the selector picks worse than the fan contains** | **CONFIRMED, with a valid ceiling** | regret p90 **0.6398 m** ADE, and **4,350 / 24,114** windows where the fan held a correct longitudinal manoeuvre and the selector missed it | **the defect is LONGITUDINAL**: `a_lon` oracle recovers **74.2 %** of the regret, `a_lat` only 20.9 %. WP-7 should train the ranker on `a_lon` first. |
| (2), environmental — *"cuts road marks"* | **NOT MEASURABLE on this corpus** | no lane, map, road-edge or drivable-area feature exists (three probes); `obstacle.offline` is 10 dynamic agent classes | needs AlpaSim's `map.xodr` or an external corpus. **A named blocker.** No proxy invented. |

**Blocked, and named:** the lane-change repair needs a label rebuild plus a retrain (A40 busy to
2026-09-08; PI decision). The road-mark criterion needs a lane-geometry corpus we do not have. The
strategic and distance-keeping regret families need, respectively, a per-window route label and a
vectorised `distance_keeping` — both work items with their reasons and their n recorded above.

**Estimator note.** Every interval here is an episode-cluster bootstrap over the 141 eval episodes,
paired where two arms share windows. `H-ESTIM-SEED-1` (a separated one-seed CI is necessary, not
sufficient) does **not** bite on §5's levers: both are deterministic post-hoc transforms of ONE
checkpoint's banked output, verified bit-reproducible, so there is no training and no inference
variance between the compared arms. What it does bite on is **generalisation to another training
run** — none of these effects has been replicated on a second refcv4b seed, and that is stated
rather than assumed.
