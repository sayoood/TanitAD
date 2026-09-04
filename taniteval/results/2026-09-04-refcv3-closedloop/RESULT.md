# refcv3 @ step 40,284 — CLOSED LOOP, and the matched open-loop pair beside it

**Arm:** `refcv3-b1-v72-30k` (`--arm hier --size base`), **step 40,284** (the run's final step) ·
**host:** Jetson Thor (`tanitad-thor-wifi`) · **instrument:**
`stack/experiments/alpasim-gsplat/closedloop_drive.py` + `openloop_drive.py` → `cl_metrics.py` ·
**written:** 2026-09-04 · **owner:** Architecture & Inference FlyWheel
**Answers:** the PI's standing request — *"compare the open and closed loop performance of both
refav1 and refcv3"* — for **refcv3**. (refav1 remains blocked on the A1 action-unit decision;
`CLOSED_LOOP_FEASIBILITY_2026-09-03.md` §3.)

---

## ⭐ VERDICT IN FOUR LINES

1. **refcv3 drives closed-loop.** `ade_0_2s` = **2.8755 m** on 435 paired windows / 9 clusters,
   scene `00040136-…`, `--condition empty`.
2. **refcv3 and refc-base are STATISTICALLY INDISTINGUISHABLE in closed loop on ADE** —
   **+0.2185 m [−0.6198, +1.1586]**, interval straddles zero. That is the pre-registered
   falsifier's own wording: *"the arms are indistinguishable in closed loop"*, **not** *"refcv3 is
   better but underpowered"*. **Two** metrics do separate and **both favour refc-base**
   (yaw-rate error, manoeuvre agreement).
3. ⭐ **The open→closed penalty, MEASURED on IDENTICAL PIXELS:** refcv3 **0.8431 → 2.8755 m
   (3.41×)**; refc-base **0.6444 → 2.6554 m (4.12×)**; flagship-v1 **7.1479 → 9.6955 m (1.36×)**.
   Closing the loop costs refcv3 **less** than it costs refc-base, and flagship-v1's closed-loop
   disaster is revealed as mostly an **open-loop prediction failure**, not control drift.
4. ⛔ **THE LOOP IS CLOSED ON PERCEPTION, NOT ON INTERACTION.** See §6 — it is the most important
   section here and it travels with every number above.

**Both blocking preconditions are satisfied:** the **leak check is CLOSED with a positive control**
(§1) and the **patch-neutrality control is bit-exact** (§2). Nothing here carries a
`LEAK-UNVERIFIED` stamp.

---

## 1 · ⛔ PRECONDITION 1 — THE LEAK CHECK, CLOSED, WITH A POSITIVE CONTROL

**Verdict: NO LEAK.** The rendered scene `00040136-e651-4abd-991d-0655ccda9430` is in neither
refcv3's training corpus nor its eval set. Artifact: `raw/LEAK_CHECK.json`.

**Why the previous check proved nothing, and what fixed it.** `CLOSED_LOOP_FEASIBILITY_2026-09-03.md`
§9.2 tested the NuRec uuids against two alpamayo lists and found them in *neither*, which is
consistent with the id namespaces simply being different — an uninformative absence. This check
carries **its own positive control**: two of the 79 NuRec scene directories **are** present in
refcv3's caches, which *proves* the namespaces coincide and therefore that an absence is real.

| probe | path-binding | n train | n eval | render scene in train | in eval | positive control |
|---|---|---|---|---|---|---|
| **1** | pod directory listing of the **v2 episode cache the trainer read** (`/root/data/{train,eval}`) | 4,573 | 142 | **NO** | **NO** | `470c4ecc-…` **in train**, `d85682b8-…` **in eval** ✅ |
| **2** | the gzipped **v7.2 label manifest** named in the run's own `config.json` argv — different file, format and producer | 4,572 | 147 | **NO** | **NO** | same two ids, **cross-pattern exact** (train-only / eval-only) ✅ |

⚠️ **`470c4ecc-8d6b-46dc-ba60-fd16888c5797` IS a NuRec scene directory AND IS in refcv3's training
set.** It ships no renderable `volume.nurec`, so it could not have been used — but a future panel
must not use it. The second renderable scene `7c72937c-…` is clean.

---

## 2 · ⛔ PRECONDITION 2 — THE POSITIVE CONTROL, AND IT IS BIT-EXACT

The brief required re-running **refc-base unchanged** through the new geometry path and the old one:
*"if its number moves, you are measuring the geometry swap, not refcv3."*

⚠️ **The literal cross-geometry form of that control is IMPOSSIBLE BY CONSTRUCTION and I say so
rather than faking it.** refc-base's encoder is fixed at **256×256 in_channels=9**; feeding it a
256×640 cylindrical raster is not "refc-base through the new path", it is a shape error. Geometry is
an **attribute of the arm**, not a nuisance parameter that can be swapped between arms
(see §7, limitation L1).

**What IS executable — and it is the control that actually matters — is patch neutrality:** does the
geometry patch plus the whole current-`stack/` code overlay move the historical arms at all? Answer,
at the **raw trajectory level**, before any metric:

| arm | steps compared | **bit-identical** | max abs Δplan | max abs Δego | max abs Δv | `f_eff` new vs banked |
|---|---|---|---|---|---|---|
| **refc-base** | 450 | **450 / 450** | **0.0** | **0.0** | **0.0** | 266.0139895990537 = 266.0139895990537 |
| **flagship-v1** | 450 | **450 / 450** | **0.0** | **0.0** | **0.0** | 266.0139895990537 = 266.0139895990537 |

`MEASURED` — `raw/CONTROL_patch_neutrality.json`, against the 2026-08-03 banked rollouts on Thor
(`~/cl_out_hq/panel_*_empty/`).

And at the **metric** level, two independent confirmations:

* `raw/CTRL_refcbase_new_vs_banked.json` — refc-base (new tree, patched harness) vs refc-base
  (banked): **15 / 15 paired metrics read `delta = 0.0` with a ZERO-WIDTH CI `[0.0, 0.0]`** over
  437 windows. Zero non-zero entries.
* `raw/CTRL_repro_flagship_vs_refc.json` — the published headline contrast **reproduces to four
  decimals**: `ade_0_2s` **+7.1642 [+5.2654, +8.9661]**, identical to the banked
  `HQ_flagship_vs_refc_empty.json`.

⇒ **The contrast is admissible.** Any refcv3-vs-refc-base difference is attributable to the arm.

---

## 3 · WHAT WAS BUILT (the five gaps, and how each was closed)

| gap | closed by | verification |
|---|---|---|
| **G1 geometry** | `_BasePolicy.canon_mode`; new `_canon_cyl` using `calib.cylindrical_rectify` + `PHYSICALAI_WIDE120_256x640`, fed the renderer's **per-clip** principal point over the wire (`require_per_clip=True` left ON) | `f_eff = 305.5774907364391` **exactly** = `frame.f_ref`; `observed_frac = 0.909998`; rig **B** (cy 751.179) |
| **G2 policy** | `RefCV3Policy` **imports** `refcv3_arm.load_model` — not forked | strict load: **missing = [] , unexpected = []**; `param_breakdown.total` 107,032,901 cross-checked against `config.json` |
| **G3 controller/grid** | `LOOKAHEAD_IDX = 0` kept; the grid is **asserted, not assumed** | checkpoint horizons `[5, 10, 15, 20, 30, 40, 50, 60]`; first four **== `WP_STEPS (5,10,15,20)`** ✅. `window = 8 == WINDOW` ✅, `in_channels = 9 == 3*STACK` ✅, `image_hw = [256, 640]` ✅ |
| **G4 checkpoint** | `ckpt_step40284_frozen.pt` pod→Thor over **direct SSH** (428 MB, < 60 s) | md5 `b1ed7075ff730d0993d2eaa3c86f6b56` **both ends**; checkpoint's own **`step` field = 40284** |
| **G5 wheelbase** | **declared, not changed.** refcv3 emits waypoints, not actions, so the harness's `L = 2.7` appears in `wp_to_control`'s `atan(L·κ)` **and** in the bicycle's `v/L·tan(steer)` and **cancels exactly**. No `L` reaches the model. | stated here; nothing edited |

⚠️ **Corrections to the brief, both MEASURED today:**
1. **`ckpt_40284_FINAL.pt` HAS been written** (2026-09-04 03:47, 1,284,991,701 B) — the brief's
   "never written" is stale. A third file `ckpt_step40284_frozen.pt` (428,518,255 B) also exists.
   **All three carry `step = 40284` and an identical 544-key state_dict** (same key-md5, same
   weight sum over the first 20 tensors), so the small frozen file was shipped instead of 1.28 GB.
2. **`observed_frac` is 0.910, not ≈1.0.** The feasibility doc predicted ≈1.0; `calib.py`'s own
   MEASURED note (8.897 % masked on rig B over 3,000 clips) predicts ≈0.911. **The measurement
   lands on the second**, which is an independent cross-check that the crop box is right: this rig
   is rig B and this is exactly what refcv3's own rig-B training frames look like. The refusal
   threshold was therefore set at 0.85, not at 1.0.

⭐ **The FOV trap, confirmed numerically.** On this **cylindrical** frame the column is linear in
azimuth: `2·(W/2)/f_ref` = **120.0000°**, matching the rig's own name `camera_front_wide_120fov`.
The pinhole formula `2·atan((W/2)/f)` returns a plausible-looking **92.6414°** and is wrong here.
Both are banked in the rollout's `canon` block.

---

## 4 · ⭐ OPEN VS CLOSED LOOP — THE PI'S QUESTION, ON IDENTICAL PIXELS

**Why this is the right comparison and the real-corpus open-loop suite is not.** The banked refcv3
open-loop reads are on **141 real held-out episodes** — a *different corpus* from the NuRec
reconstruction, on which REF-C is MEASURED **3.21× OOD**. Comparing them directly to a closed-loop
number here would confound corpus and loop. So the open loop was **re-measured on this scene**, at
the same step, with `openloop_drive.py`.

⭐ **This also yields refcv3's OWN, STEP-MATCHED sim-vs-real OOD factor** — previously only REF-C
v1.2's 3.21× was available. The sibling epoch-end read landed today at the **same step 40,284**
(commit `cb8ca05`, `RESULT-refcv3-40284-openloop.md`, n = 4,823 windows / 141 episodes):
**`os` = 0.4419 m [0.4098, 0.4743] on real footage.** Against this panel's **0.8431 m** on the
reconstruction, refcv3's open-loop OOD factor is **1.91×** — `MEASURED`, step-matched, and
**substantially milder than REF-C v1.2's 3.21×**. ⚠️ One scene vs 141 episodes; treat it as a
first read, not a corpus-wide constant.

⛔ **And the sibling read carries the fact that most sharpens §8.1:** in open loop on the real
corpus, the **hold-action control `ha` (0.2996) BEATS the deployed arm `os` (0.4419)**. A
closed-loop panel with **no trivial floor at all** therefore cannot be assumed to be measuring
skill.

⭐ **The pairing is an IDENTITY, not a determinism claim.** One render pass drove all three arms in
one process. **MEASURED:** the 199-frame `frame_md5` list is byte-identical across all three arms
(digest `9a058d5d8cdf892d61cfde616f1ecf85`). Each arm then canonicalises those same native frames
into its own geometry.

| arm | **OPEN** `ade_0_2s` (m) | **CLOSED** `ade_0_2s` (m) | closed / open |
|---|---|---|---|
| **refcv3** (step 40,284) | **0.8431** | **2.8755** | **3.41×** |
| refc-base (step 29,999) | 0.6444 | 2.6554 | 4.12× |
| flagship-v1 (step 29,999) | 7.1479 | 9.6955 | 1.36× |

*Open loop n = 190 windows / 9 clusters (170 paired); closed loop n = 435–450 windows / 9 clusters
(435 paired). Same scene, same render config, same scorer, same estimator.*

**Three readings, in order of confidence:**

1. **Closing the loop roughly triples refcv3's ADE (3.41×).** That is the honest size of the
   perception-drift penalty on this scene.
2. **refcv3 pays LESS to close the loop than refc-base does (3.41× vs 4.12×)** — refcv3 starts
   worse open-loop and ends up statistically level closed-loop. ⚠️ Neither ratio carries a CI (a
   ratio of two paired means across two different window sets); treat the **ordering** as the
   finding and the ratio as `MEASURED but interval-free`.
3. ⭐ **flagship-v1's closed-loop catastrophe is mostly NOT control drift.** It is already at
   **7.1479 m open-loop** on this reconstruction. The published +7.16 m closed-loop gap has always
   been read as imagination-in-the-loop divergence; on this scene **the open loop already carries
   6.30 m of it** (`OL` refcv3 − flagship = **−6.3048 [−7.7699, −5.0017]**, separated).

⭐ **A control that must read a known value, and does.** In open loop the ego **is** the logged
path, so `cross_track_abs_m`, `dist_to_gt_traj_m`, `executed_speed_err_ms` and
`route_corridor_departure_rate` **must** be exactly 0. **All four read exactly `0.0000` with
zero-width CIs on all three arms.** That is the open-loop harness reading its own no-information
value, which is what makes the closed-loop contrast interpretable.

---

## 5 · THE FOUR FAMILIES — CLOSED LOOP, refcv3 vs refc-base

**Tier:** closed loop on perception (§6). **Estimator:** `paired_episode_cluster_bootstrap`,
2,000 resamples. ⛔ **Never `overlapping_holdout_se`.**
**Clusters = 9 disjoint rollout starts of ONE clip** — *not* independent episodes, and never to be
confused with the n = 40 val bootstrap. **n = 435 paired windows** (refcv3 435, refc-base 437,
flagship-v1 450 of 450 raw ticks; the shortfall is per-arm window validity along each arm's own
driven path).

Per family, never pooled. **Δ = refcv3 − refc-base; for every error metric, negative = refcv3 better.**

### ADE
| metric | refcv3 | refc-base | Δ paired [95 %] | separated |
|---|---|---|---|---|
| `ade_0_2s` | 2.8755 | 2.6554 | **+0.2185 [−0.6198, +1.1586]** | **no** |
| `de_0_5s` | 0.9963 | 0.9142 | — | — |
| `de_1s` | 2.1713 | 1.9796 | — | — |
| `de_1_5s` | 3.3884 | 3.1967 | — | — |
| `de_2s` | 4.9459 | 4.5312 | — | — |

### LONGITUDINAL
| metric | refcv3 | refc-base | Δ paired [95 %] | separated |
|---|---|---|---|---|
| `abs_target_speed_err_ms` | 1.8994 | 1.7048 | +0.1943 [−0.3956, +0.8721] | no |
| `abs_executed_speed_err_ms` | — | — | +0.1349 [−0.3867, +0.7250] | no |
| `along_track_ade_m` | 2.8266 | 2.6104 | +0.2149 [−0.6368, +1.1493] | no |
| `along_track_2s_m` | 4.8300 | 4.4252 | — | — |
| **distance-keeping** (`real_lead_*`, `synth_lead_*`, headway / time-gap / TTC) | **UNAVAILABLE** | **UNAVAILABLE** | **UNAVAILABLE** | — |

⛔ **The distance-keeping half of LONGITUDINAL is UNAVAILABLE, n = 0, reason `"no jointly finite
windows"`** — the `empty` condition has no lead vehicle in frame. Reported with its reason and its
n per the four-families rule, **never silently dropped**. **This is a work item, not a pass**: a
`--condition lead25/lead15/lead8/cutin` panel would produce it, and is the single cheapest
extension of this experiment (~2 min of GPU per arm per condition).

### LATERAL
| metric | refcv3 | refc-base | Δ paired [95 %] | separated |
|---|---|---|---|---|
| `heading_err_rad` | 0.0269 | 0.0357 | −0.0089 [−0.0236, +0.0038] | no |
| `curvature_err_1pm` | 0.00110 | 0.00160 | −0.0004 [−0.0011, +0.0001] | no |
| **`yawrate_err_rads`** | **0.0206** | **0.0132** | **+0.0074 [+0.0027, +0.0119]** | ⛔ **YES — refc-base better** |
| `cross_track_abs_m` | 0.9872 | 0.6304 | +0.3554 [−0.1907, +0.7695] | no |
| `lateral_ade_m` | 0.3749 | 0.3096 | +0.0648 [−0.0403, +0.1802] | no |

⚠️ **The one clean lateral separation goes AGAINST refcv3**: its yaw-rate error is **56 % higher**.
Heading and curvature both point the other way and neither separates, so the honest statement is
*"refcv3's commanded yaw rate is noisier while its heading and curvature are not distinguishable"* —
a jerk/smoothness signature, not a pointing error.

### TACTICAL
| metric | refcv3 | refc-base | Δ paired [95 %] | separated |
|---|---|---|---|---|
| **`manoeuvre_plan_eq_logged`** | **0.2276** | **0.2998** | **−0.0736 [−0.1309, −0.0244]** | ⛔ **YES — refc-base better** |
| `manoeuvre_head_eq_logged` | 0.1701 | 0.1556 | +0.0138 [−0.0267, +0.0506] | no |
| `manoeuvre_exec_eq_plan` | 0.6444 | 0.8778 | — | ⛔ **NOT COMPARABLE — see below** |

⛔ **`manoeuvre_exec_eq_plan` MUST NOT be read as "refc-base executes its plan better."** refc-base's
plan share is `lane_keep 0.8284 / brake_stop 0.1716` and **`accelerate` 0.0000**; the metric
collapses towards *"how often was the plan the majority class"* — the constant-predictor tie the
2026-08-03 panel already flagged for this metric. refcv3's plan share is
`lane_keep 0.5310 / accelerate 0.4552 / brake_stop 0.0138`.

⭐ **And that difference is itself the most interesting qualitative finding in this panel.** The
logged class share on this clip is `lane_keep 0.3655 / accelerate 0.2138 / brake_stop 0.4207`.
**refcv3 emits `accelerate` in closed loop (45.5 % of plans); refc-base emits it exactly zero
times.** The programme's long-standing longitudinal-blindness defect (*"0/881 accelerate"*, the
5-way softmax mixing LAT and LON) is **visibly absent from refcv3's closed-loop behaviour**.
⚠️ This is a **class-emission** observation, not an accuracy claim — refcv3's agreement with the
logged manoeuvre is *lower*, and this clip is decelerating (`brake_stop` is the plurality class),
so emitting `accelerate` 45 % of the time is not obviously right either. **Stated as a signature to
follow up, not as a win.**

### STRATEGIC
| metric | refcv3 | refc-base | Δ paired [95 %] | separated |
|---|---|---|---|---|
| `route_corridor_departure_rate` | 0.1931 | 0.0938 | +0.0989 [−0.0506, +0.2200] | no |
| `route_label_valid_rate` | 0.3931 | 0.3822 | — | — |
| `route_head_eq_logged` | 0.8012 | 0.1796 | +0.6485 [+0.5453, +0.7654] | ⛔ **DEGENERATE — see below** |

⛔⛔ **`route_head_eq_logged` SEPARATES AND MUST NOT BE QUOTED AS A STRATEGIC-ACCURACY WIN.** Three
independent degeneracies, all read from the primary artifact:

1. **The logged route has ONE class.** `route_logged_share` = `straight 0.393 / unknown (gray zone)
   0.607`, **`left = 0.000`, `right = 0.000`**. A **constant `straight` predictor scores 1.0000.**
   refcv3 reads 0.8012 and refc-base 0.1796 — this measures *how often each head emits the only
   class present*, not route skill.
2. **The nav-echo guard COULD NOT RUN.** The artifact's own
   `route_head_nav_echo_check.verdict` = **`UNIDENTIFIABLE`** — `n_distinct_nav = 1` (`follow`
   throughout), so an echo cannot be distinguished from a constant. Both arms carry
   `NAV_ECHO_UNIDENTIFIABLE: true`.
3. **Macro-F1 is near the floor for both.** refcv3 **0.2224**, refc-base **0.0761** — because the
   `left`/`right` columns have `support_n_true = 0` while the heads still fire them (refcv3 7+27
   times, refc-base 74+63).

⇒ **STRATEGIC accuracy is UNAVAILABLE on this scene.** The feasibility doc predicted exactly this
(*"degenerate on a junction-free 20 s clip"*) and it is confirmed. **A junction scene is required
before any strategic-accuracy claim** — and we hold only **2 renderable NuRec volumes**, one of
which is this one. `route_corridor_departure_rate` remains reportable but is a corridor around the
**logged path**, not a road edge (§6).

### The three-way reference row (for context, same windows)
`refcv3 − flagship-v1`, closed loop, n = 435: `ade_0_2s` **−6.9554 [−9.3506, −4.5209]**,
`cross_track_abs_m` **−4.1473 [−6.9461, −1.7369]**, `abs_target_speed_err_ms`
**−6.2128 [−8.1403, −4.4064]**, `route_corridor_departure_rate` **−0.4046 [−0.5425, −0.2621]** —
all separated, all favouring refcv3. flagship-v1's absolute `ade_0_2s` is **9.6955 m**.

---

## 6 · ⭐⭐ THE SCOPE LIMIT — the sentence that travels with every number above

> **The loop is closed on PERCEPTION, not on INTERACTION. The ego's own trajectory determines every
> future observation — that is real, and it is what the PI asked for — but *nothing in the scene
> responds to the ego*. Other agents are replayed or scripted, there is no map and no drivable-area
> polygon, and the vehicle model has no collision response. ⇒ This harness measures HOW WELL THE
> MODEL DRIVES ITSELF. It cannot measure WHAT HAPPENS WHEN IT DRIVES BADLY.**

Concretely, **this panel CANNOT produce** and nothing above should be read as implying:

| not available | why, from source |
|---|---|
| **Off-road / drivable-area departure rate** | there is **no map in the harness at all** (`xodr\|off_road\|drivable` → **zero hits** across `closedloop_drive.py`, `cl_metrics.py`, `actor_map.py`). `route_corridor_departure_rate` is literally `int(abs(ct) > CORRIDOR_M)` on cross-track **to the logged trajectory** (`cl_metrics.py:386`). A model that departs the corridor may be perfectly on-road; one inside it may be off-road. **refcv3's 0.1931 is not an off-road rate.** |
| **A real collision rate** | the only collision metric is a lane-gated geometric headway test against a **scripted** lead; the bicycle has **no collision response** and drives through everything. On `empty` it is `n = 0` anyway. |
| **Anything REACTIVE** | actors are replayed or on a fixed script. Nothing brakes, yields or swerves because of us ⇒ no right-of-way, no negotiation, no merge, no induced collision. |
| **Absolute rates of any kind** | ⛔ **WITHIN-SIM RELATIVE.** The artifact's own `within_sim_note`: REF-C open-loop ADE **1.5157** on these reconstructions vs **0.4728** on real footage — **3.21× OOD**. **Orderings survive; absolute rates do not.** refcv3's 2.8755 m is not a road number. |
| **Strategic accuracy** | degenerate — §5 STRATEGIC, three independent reasons. |
| **A 40-episode-style interval** | the bootstrap clusters are **disjoint segments of ONE clip**, stated in the JSON's own `estimator_note`. |

⚠️ And from the harness's own docstring: the bicycle is **planar**, and the z/roll/pitch
ground-following is a **harness choice, not a physics engine**.

---

## 7 · LIMITATIONS THAT ARE SPECIFIC TO *THIS* COMPARISON

* **L1 — the arm/geometry confound is STRUCTURAL and cannot be removed here.** refcv3 is a 256×640
  cylindrical model and refc-base a 256×256 f-theta model; each must be fed the raster it was
  trained at. **The contrast is therefore arm-as-deployed vs arm-as-deployed, with geometry as an
  attribute of the arm.** What §2's control establishes is the narrower, and the only achievable,
  claim: **the patch did not move refc-base by a single bit**, so the difference is not an artifact
  of my edits. It does *not* establish that a 256×640 refc-base would score the same as a 256×256
  one — that model does not exist.
* **L2 — n = 1 scene, 9 clusters, one 20 s clip.** The ceiling is **not compute**: only **2 of 79**
  NuRec scene directories hold a real `volume.nurec` (78 hold an mp4; one is a 110-byte stub).
  A second scene would double n; more would need provisioning, which is the PI's.
* **L3 — the arms sit at different steps.** refcv3 **40,284**; refc-base and flagship-v1 **29,999**.
  Stated, not corrected.
* **L4 — refcv3 is not parity-comparable at the LEVEL** with refc-base/refc-xl
  (`config.json`: `v2_parity.parity = false`), as its own open-loop RESULT records. What travels
  across a parity break is each arm's margin over its **own** trivial control — and this panel does
  **not** carry a closed-loop `ha0`/hold-action floor. **That is the highest-value next
  measurement** (§8).
* **L5 — nav is a single value** (`follow`) for the whole clip, so nothing here tests route
  conditioning, and the closed-loop `nav_cmd` is derived from the **logged** future route
  (`nav_from_route`) — optimistic by construction, exactly as the open-loop suite's oracle-nav
  caveat says.
* **L6 — the closed/open ratios in §4 carry no interval.**

---

## 8 · WHAT SHOULD HAPPEN NEXT (ranked, each cheap)

1. ⭐ **A closed-loop trivial floor.** Add a `hold-action` / `constant-velocity` policy to
   `closedloop_drive.py` (~20 lines, no GPU model). Without it, "refcv3 = 2.8755 m" has no bar, and
   L4 means the cross-arm level is not admissible anyway. **This is the single highest-value item.**
2. **A `--condition lead25 / lead15 / lead8 / cutin` panel** — makes the **distance-keeping half of
   LONGITUDINAL** available instead of `UNAVAILABLE, n = 0`. ~2 min GPU per arm per condition.
3. **The second renderable scene** (`7c72937c-…`, leak-clean) — doubles n for ~5 min of GPU.
4. **A junction scene** — the only way to make STRATEGIC non-degenerate. **Provisioning ⇒ PI.**
5. **refav1 closed loop** stays blocked on the **A1** action-unit decision.

---

## 9 · ESCALATIONS

1. ⛔⛔ **A closed-loop trivial floor does not exist in this harness.** Every closed-loop number the
   programme has published — including these — is a bare level or a cross-arm delta with **no
   `ha0`-equivalent bar**. Item 8.1. **This needs a decision to schedule, not a discussion.**
   ⭐ **It is now urgent, not tidy:** today's step-matched open-loop read (`cb8ca05`) shows the
   **hold-action control beating the deployed arm** on the real corpus (`ha` 0.2996 vs `os` 0.4419).
   A closed-loop panel that cannot see its own trivial floor cannot tell a driving model from a
   coasting one.
2. ⛔ **`route_head_eq_logged` separates at +0.6485 and is DEGENERATE.** It is exactly the shape of
   number that gets quoted out of a JSON as a hierarchy win. It is flagged in the artifact and here;
   it must not reach the registry or the paper.
3. ⚠️ **`CLOSED_LOOP_FEASIBILITY_2026-09-03.md` needs two corrections** (both MEASURED here):
   `ckpt_40284_FINAL.pt` **does** exist, and the predicted `observed_frac ≈ 1.0` is **0.910**
   (`calib.py`'s own rig-B figure was right).
4. ⚠️ **`PROGRAM_OVERVIEW.md` §5.0.1 still prints the superseded "ADE saw nothing" table** while
   line 45 carries retraction R-2026-08-03-C. Unfixed, and it reached me through a brief again.
5. ⚠️ **The Thor gsplat JIT needs `PATH=$HOME/venvs/tanitad-edge/bin:…` AND `CPATH=…`**, or it dies
   with a misleading *"Ninja is required"*. Cost 1 round-trip today; already in auto-memory
   (`nurec-scenes-are-open-msgpack-gsplat-works-on-thor`), and `run_panel_v3.sh` now carries both.
6. ⭐ **Pod→Thor DIRECT SSH WORKS** — 428 MB in < 60 s, `tanitad-refcv3` → Thor, after appending
   Thor's **public** key to the pod's `authorized_keys`. The HF relay was not needed and HF quota
   was not touched.

---

## 10 · DELIVERABLE MANIFEST

| artifact | where it lives | only one place? |
|---|---|---|
| This document | `repo:taniteval/results/2026-09-04-refcv3-closedloop/RESULT.md` | no |
| Leak check (both probes + positive control) | `repo:…/raw/LEAK_CHECK.json` | no |
| Patch-neutrality control (bit-exact) | `repo:…/raw/CONTROL_patch_neutrality.json` | no |
| Metric-level controls | `repo:…/raw/CTRL_refcbase_new_vs_banked.json`, `…/raw/CTRL_repro_flagship_vs_refc.json` | no |
| **CLOSED-loop** four families | `repo:…/raw/V3_refcv3_vs_refc-base_empty.json`, `…/raw/V3_refcv3_vs_flagship_empty.json` | no |
| **OPEN-loop** four families (matched) | `repo:…/raw/OL_refcv3_vs_refc-base.json`, `…/raw/OL_refcv3_vs_flagship.json` | no |
| refcv3 closed-loop raw rollouts | `repo:…/raw/rollouts_refcv3_closedloop_empty.json` | no |
| Open-loop raw rollouts, all 3 arms + summary | `repo:…/raw/rollouts_*_openloop.json`, `…/raw/openloop_summary.json` | no |
| Driver log | `repo:…/raw/run_panel_v3.log` | no |
| **Harness patch (G1/G2/G3)** | `repo:stack/experiments/alpasim-gsplat/closedloop_drive.py` | no |
| **Open-loop arm registration + window guard** | `repo:stack/experiments/alpasim-gsplat/openloop_drive.py` | no |
| refcv3 checkpoint on Thor | `tanitad-thor-wifi:/home/nvidia/models/refcv3-b1-v72-40284/{ckpt.pt,config.json}` md5 `b1ed7075ff730d0993d2eaa3c86f6b56` | ⚠️ **no — the source `tanitad-refcv3:/workspace/experiments/refcv3-b1-v72-30k/` still holds it** |
| Thor run tree `tanitad_cl_v3` + `run_panel_v3.sh` | `tanitad-thor-wifi:/home/nvidia/` | ⚠️ **YES, Thor only** — but it is a *derived* overlay of repo code + `run_panel_v3.sh`, whose exact protocol is transcribed in §3/§4 and which is `run_panel_hq.sh` with the arm list changed |
| refc-base / flagship-v1 closed-loop rollouts | not re-banked — **bit-identical** to the 2026-08-03 artifacts already in `repo:stack/experiments/alpasim-gsplat/results/closedloop-hq-render/rollouts/` | no |

**Nothing that took real effort lives only on a host.** No training was disturbed (Thor was idle at
0 % before the first launch; the refcv3 pod's run is `done: true` at 40,284 and its A40 shows
0 MiB used). Peak Thor device memory `torch.cuda.max_memory_allocated()` = **1.53 GB**.
Total GPU: **≈9 minutes**.
