# REVIEW 4 — REFe, the fourth independent review

**Date** 2026-09-20 · **Scope** the whole package: correctness, internal consistency, consistency
with the specification, with the paper, and with the released DriveRL implementation.
**Mandate** the PI asked me to confirm the fixes and the entire consistency of the package.
**Standing assumption, and it held:** three prior rounds each found defects the previous round's own
tests had passed. This round has its own.

Evidence classes used throughout: **MEASURED** (ours + artifact path) · **PUBLISHED** (cited) ·
**INHERITED** · **ESTIMATED** · **HYPOTHESIS**. Every MEASURED number below was produced by me today
by calling the shipped code, not by re-reading a previous report.

---

## 1 · Headline

⛔ **The package is NOT ready to produce a published REFe number, and the blocker is not the model —
it is the scorer target bank. Two of the six PDM components are, on today's default bank, close to a
stationary-vs-moving detector, and the mechanism was introduced by THIS ROUND'S HEADLINE FIX.**

The three things that block a published number, in order:

1. ⛔⛔ **The NPC splice manufactures collisions on the teacher's own realised path.** It writes the
   log's positions and validity masks for the background agents but **never their sizes**, so an
   agent present in the log's future and absent from the 5-row history becomes `valid=True` at a
   real position with a **zero-area polygon**, and DriveRL's polygon-intersection test fires against
   it. MEASURED with a three-arm control on 2/2 frames: no-splice → `collision.info` **0.0000**;
   shipped splice → **1.0000**, the single hit being a polygon of extent **0.00 × 0.00 m** at a
   centre distance of **8.83 m**; splice + sizes → **0.0000** again. This is the third review's
   defect — *"a frozen background MANUFACTURES collisions"* — **not removed but replaced by a new
   mechanism that manufactures collisions**, and the state it replaced was cleaner on the expert's
   own path.
2. ⛔⛔ **The splice is also mis-indexed in time.** It slices the log by TAIL (`lp[..., -n:]`) while
   the candidate prefix it must match is the HEAD of the horizon. MEASURED by calling the real
   function with row-index-valued tensors: at the first prefix (k=3) the NPCs are placed **+3.40 s
   ahead of the ego**; only the final full-horizon prefix (k=20) is aligned. At the shipped
   `--stride 2`, **9 of 10 prefix evaluations are misaligned**. Aligning it changes **15 of 216**
   banked component cells over 6 frames, **every one of them in the direction "manufactured
   collision / TTC 0"**, twice on the teacher's own path.
3. ⛔⛔ **The live default scorer bank was built by code that no longer exists.** `train.py`'s default
   `--scorer-targets` is `D:/Projects/TanitAD/data/refe_scorer_targets_full`, written
   **21:57:54**; `score_proposals.py` was last modified **22:33:10**. MEASURED: **0 of 2,165** rows
   carry `dac.violation`, **0 of 2,165** carry any `raw.*` key, and a rebuild of one banked frame
   with today's code differs in **6 of 40** numeric cells. Because `dac.violation` is absent,
   `ScorerBank.components()` silently falls back to the `off_road.OffRoad.info == 1` test that this
   round's fix was written to eliminate — and **173 of 2,165 rows (8.0 %)** sit at category 3 or 4
   and are therefore read as **CLEAN drivable-area**.

**ESCALATION — PI decision required.** The scorer bank must be rebuilt after (1) and (2) are fixed,
and the rebuild is not free: `build_scorer_targets.py`'s own MEASURED cost is 444 ms per candidate at
stride 2, ~11 candidates per frame, so the 197-frame default bank is ≈16 min of CPU and the
useful-size bank is hours. **No REFe scorer number — and therefore no closed-loop REFe number, since
`_pick` consumes the scorer — is admissible until that rebuild lands.** The trajectory (WTA) half of
the model is unaffected and can train now.

### What is genuinely fixed, and is now quotable

* **The rotary encoding is correct.** `diag_rope.py` RUN today on the real ViT-S checkpoint:
  `rel L2 0.00000 / cos 1.00000` against the reference on BOTH patch features and block-0 attention
  logits, with the SELF control at exactly 0 and the no-rope control at 0.58114. Verdict token
  `ROPE_MATCHES_REFERENCE`, exit 0. Review 2's finding is closed. The newly banked primaries confirm
  every element line by line.
* **The register compression, the decoder asymmetry, the WTA metric, the cosine schedule, weight
  decay 0.01, the PDM aggregation rule, `requires_scenario`, the goal frame and the ego vector** all
  check out — see §2. `diag_architecture.py`, `diag_planner_holds.py`, `diag_schedule.py`,
  `diag_scorer_components.py` and `validate_model.py` all exit 0 today.
* **The ImageNet normalisation is in place and reaches every consumer** (a buffer inside the trunk).

### The other four things that must be said

* ⛔ **The PETR-style `pos3d` does not do what its own justification claims, and its intrinsics are in
  the wrong units.** It is a compile-time constant — identical for every sample, no per-sample
  intrinsics, no extrinsics at all — so it is a per-token table by another parameterisation, which is
  exactly the property the change was made to escape. And it applies **1920×1080 intrinsics to a
  960×512 input**: MEASURED horizontal ray span **31.34°** instead of **62.85°**, with x and y
  **never crossing zero**. (§3, defect N4.)
* ⛔ **The seven carried latches propagate an artefact past the filter built to exclude it.** MEASURED
  counterfactual: **4 of 11 candidates** — the entire lateral family — have their banked comfort
  dragged from **1.0000 to 0.7500** by a latch armed at prefix k=3, a prefix whose own comfort value
  `KEY_MIN_PREFIX` declares inadmissible. Ratio exactly 0.7500 = `comfort.py`'s penalty multiplier.
  (§3, defect N3.)
* ⛔ **`REFePlanner._hold()` is broken and no instrument executes it.** MEASURED: 21 states, **1
  distinct timestamp**; `get_state_at_time(t0)` raises `AssertionError: angle is not finite` and
  `get_state_at_time(t0+0.5 s)` raises "not in trajectory time window". `diag_planner_holds.py`
  passes only because it **stubs `_hold`**. (§3, defect N5.)
* ⛔ **Three constructed regressions walk straight past the validator** (§4): removing only
  `visual_ctx.detach()`, disabling LoRA permanently, and adding an un-checkpointed parameter inside a
  Block all read GREEN.

**Teacher result:** the 89.755151 is CONFIRMED as computed, tier **closed-loop nuPlan CLS,
non-reactive**, class **MEASURED (ours)** — but it is **0.214849 BELOW** the published 89.97 and the
rig is **not reproducible across launches**. See §5; it is admissible as our own anchor, not as a
reproduction of the paper's number.

---

## 2 · Confirmation table — one row per change since Review 3

| # | claim (from the change list) | what I MEASURED | file:line | verdict |
|---|---|---|---|---|
| 1 | ImageNet normalisation applied inside the trunk as buffers | `img_mean`/`img_std` registered non-persistent and applied on the first line of `forward`, so trainer, planner and validator all get it; `train.py:298` and `planner.py:260` both feed BGR→RGB `/255.0`, i.e. exactly the operating point the buffers expect | `refe/model.py:290-293`, `:316` | **CONFIRMED** |
| 2 | engine `frame_time_interval` tied to `TRAJ_DT` | `EngineMergedConfig(..., frame_time_interval=_SP.TRAJ_DT)`; `TRAJ_DT = 0.2` and `build_log_scenario_data` uses `history_sample_interval = 0.2` for its futures, so the two clocks agree | `refe/scorer_gate.py:112-115`; `refe/score_proposals.py:57`; DriveRL `src/driverl/nuplan/feature_builder.py:293` | **CONFIRMED** |
| 3 | yaw rate seeded from the anchor | `dyaw[:, 0] = yaw[:,0] - ori[:,0,-(T+1)]` before wrapping and dividing | `refe/score_proposals.py:330-336` | **CONFIRMED** — but the seed is **conditional on `ori.shape[2] >= T+1`**, which only holds because `inject_proposal` extends orientations **only when `yaw is not None`** (`:171`). The production caller always passes yaw (`build_scorer_targets.py:234`); a `yaw=None` call silently restores the zero seam with no warning. **PARTIAL on robustness** |
| 4 | weight decay 1e-4 → 0.01 | `--weight-decay` default `0.01`, passed to `AdamW` | `refe/train.py:524`, `:383` | **CONFIRMED** |
| 5a | NPC futures spliced from `log_sd` | the splice runs and does write log positions | `refe/score_proposals.py:220-244` | **WRONG** — see N1/N2. It writes positions, orientations, velocities and masks but **never `agent_size_all`**, and it slices by tail rather than by prefix index |
| 5b | agent 0 protected | analytic control: after the real `_splice_npc_futures`, agent 0's row still carries only the candidate markers (`all(v >= 1000) == True`) at every prefix tested | `refe/score_proposals.py:234` | **CONFIRMED** |
| 5c | ALL SEVEN stateful latches carried, not three | all seven present; located in DriveRL source: `_direction_progress_last/_baseline_last/_progress_buffer` (`center_line.py:724-756`), `_cross_lane_deadband_counter` (`off_road.py:183-206`), `_comfort_triggered` (`comfort.py:177-185`), `_nuplan_ttc_triggered` + `_nuplan_ttc_collided_matrix` (`nuplan_ttc.py:147-173`). `carry` is created **per candidate** inside `score_proposal_rollout`, so there is **no cross-candidate contamination** — I checked the two other routes too: the calculators hold no per-frame state on `self` (only config), and none of them writes to `log_scenario_data` | `refe/score_proposals.py:490-492`, `:612` | **PARTIAL** — carrying is correct *per se* and cross-candidate contamination is absent, but two of the seven are **monotone OR-latches armed at prefixes the aggregation declares inadmissible**. See N3 |
| 6 | `KEY_MIN_PREFIX = {"comfort": 10}` | present and applied through `_min_prefix_for` in the aggregation filter | `refe/score_proposals.py:69-74`, `:619-621` | **CONFIRMED present, DEFEATED in effect** — it removes the artefact's VALUE and the latch carries the artefact's EFFECT past it (N3) |
| 7 | nominal off-road split into `dac.violation` / `ddc.violation` booleans derived per prefix | both emitted per prefix from the integer category; `ddc.violation` aggregates under `max` as an event | `refe/score_proposals.py:569-573`, `:632-634` | **CONFIRMED in code, ABSENT from the live bank** — 0 of 2,165 rows carry `dac.violation`, and the consumer's fallback re-reads the defective path for 173 rows (8.0 %). See N6 |
| 8 | `pos3d`: learned table → PETR-style MLP over analytic camera-frustum coords | the MLP exists and reads the config intrinsics; but the frustum is a **function of `(gh, gw, device, dtype)` only** — two independent builds are bit-identical, 1920 distinct vectors for 1920 tokens — and the intrinsics are the **native 1920×1080** values applied to a **960×512** input | `refe/model.py:450-477`, `:98-106`; resize at `refe/train.py:298` | **WRONG** — see N4 |
| 9 | `reg_compress`: MLP ratio 4→1, heads 8→16 | `reg_mlp_ratio=1`, `reg_heads=16`; measured `reg_compress = 6,303,744` params (was 12,598,272) | `refe/model.py:94-95`, `:413-414` | **CONFIRMED** — but the PI is recorded as accepting **12,598,272** (`REVIEW_3_FULL.md:103`); nothing in the package records a sign-off on the halving. **ESCALATE** |
| 10 | `score_q_mlp` added — the scoring decoder encodes the candidate TRAJECTORY | `s = score_q_mlp((traj.detach() if detach_scorer else traj).flatten(2))`, then `score_dec` cross-attends `visual_ctx.detach()`. MEASURED: a score-only backward gives gradient to **78 tensors, all in `score_q_mlp`/`score_dec`/`score_head`** and nothing else | `refe/model.py:445-447`, `:515-520` | **CONFIRMED** — the scorer is genuinely detached on both paths today |
| 11 | sinusoidal (Fourier) goal encoding in `ego_enc` | `goal_freqs = 2**arange(4)` frozen; `goal_dim = 2*2*(1+2*4) = 36`; measured `ego_enc = 77,312` params | `refe/model.py:421-426` | **CONFIRMED** — but `MODULE_SIZING_STUDY.md:478` recommends **8** bands / 64 channels and the code ships **4**; the study's own total assumes 8. **PARTIAL vs the study** |
| 12 | `REFe.forward(..., detach_scorer=True)`; the regression arm calls the REAL forward with it flipped | `validate_model.py:129` calls `m(img, ego, goal, detach_scorer=False)`; MEASURED it goes RED (queries grad non-zero, 12 modules touched) | `refe/validate_model.py:129-135`; `refe/model.py:479` | **CONFIRMED** — and it is a real improvement over the hand-built arm. But it guards **one of the two** detach points; see N7 |
| 13 | `raw.` prefix on `collision.NuPlanCollision.reward` and `ttc.NuPlanTTC.info` | both in `NON_AUTHORITATIVE`, renamed at harvest; MEASURED present in a fresh rollout (`raw.ttc.NuPlanTTC.info`, `raw.collision.NuPlanCollision.reward`) | `refe/score_proposals.py:81`, `:550` | **CONFIRMED in code, ABSENT from the live bank** (0 of 2,165 rows) |
| 14 | `diag_rope.py` given assertions, a verdict token, an exit code, and a live `build_axial_rope` call | RUN today: 5/5 PASS, `ROPE_MATCHES_REFERENCE`, exit 0; the span row is derived from a live `build_axial_rope` call at `:130` | `refe/diag_rope.py:130`, `:182-204` | **CONFIRMED** — with one display caveat (N9) and one residual scope gap (N10) |
| 15 | `diag_planner_holds.py` arms 1 and 3 now call `compute_planner_trajectory` | both call the real method (`:97`, `:113`); RUN today, 4/4 PASS | `refe/diag_planner_holds.py:92-116` | **PARTIAL** — they call the real method but **stub `_hold` and `_image_for`**, and the real `_hold` is broken (N5) |
| 16 | `diag_scorer_components.py` EP arms no longer assert the diagnostic's own clamp | the two arms now assert the RAW advance exceeds 1.05× the teacher's AND the clamped value stops at exactly 1.0 — two independent facts | `refe/diag_scorer_components.py:164-169` | **CONFIRMED** |
| 17 | `validate_model.py` asserts peak memory | `fails += [] if _pk < 44.0 else [...]` | `refe/validate_model.py:144-149` | **PARTIAL** — only the A40 bound is asserted, and it cannot fail on any rig this will run on; the 8 GB bound is printed, not asserted. Two adjacent lines print **GiB** (`:138`, `/2**30`) and compute **decimal GB** (`:146`, `/1e9`) under the same word "GB" |
| 18 | `POD_HANDOFF.md` corrected: arm A 6.3 / arm B 25.3 A40-h, peak 9.58 GB, fidelity 5.88 % | fidelity 5.88 % reproduces (5.8824 %). Arms A/B reproduce arithmetically but arm A was computed on **1,746** tuples while the same file states the bank as **2,182** (→ 7.88 h). **Peak 9.58 GB has no artifact**: `raw/refe_vitl_devbox_memory.txt` is **0 bytes**, and I measured **9.50 GiB** today. Arm C (~55 h) was not recomputed and contradicts `MODULE_SIZING_STUDY.md:737` (222.08 h) by 4.0×. The sentence the file's own §CORRECTIONS says "is withdrawn" is **still present at line 42** | `POD_HANDOFF.md:36-42`, `:60`, `:126-141` | **PARTIAL / WRONG** — see §6 |

---

## 3 · New defects, ranked

### N1 — ⛔⛔ BLOCKING · The NPC splice manufactures collisions with zero-area phantom agents

`_splice_npc_futures` writes `agent_positions_all`, `agent_orientation_all`, `agent_velocity_all` and
**`npc_mask_all`** from the log, and never `agent_size_all`. `inject_proposal` extends sizes by
repeating the *candidate object's own* last history value, which is **zero** for an agent that was not
tracked in the 5-row history. The splice then marks that agent **valid** at a **real position** with a
**zero-area polygon**, and `NuPlanCollision._polygon_overlap_pairs` — a segment/polygon intersection
over four zero-length edges — returns a hit.

**MEASURED (three-arm control, `scratchpad/m5_phantom.py`), candidate = the teacher's OWN realised path:**

| frame | agents valid in log future but NOT in history | arm A: no splice | arm B: **SHIPPED** | arm C: splice + sizes |
|---|---|---|---|---|
| `99ca544752f255ad` @60 | 11 | hits `[]`, collision **0.0000**, ttc **1.0000** | hits `[122]` — **zero-area** — collision **1.0000**, ttc **0.0000** | hits `[]`, collision **0.0000**, ttc **1.0000** |
| `b2a5c363d1dd5abe` @60 | 14 | hits `[]`, collision **0.0000**, ttc **1.0000** | hits `[40]` — **zero-area** — collision **1.0000**, ttc **0.0000** | hits `[]` at full prefix |

The offending polygon is printed verbatim: `[[39.99, -9.28]] × 4`, extent `0.00 × 0.00 m`, centre
distance **8.832 m** from a 5.23 × 2.43 m ego box. Control: the ego's own polygon stays real in every
arm (`5.23 × 2.43`), so the splice is not corrupting the candidate.

**Corpus-level consequence, MEASURED over the live default bank (2,165 rows / 197 frames):**

| candidate | collision (1.0 = clean) | ttc | comfort |
|---|---|---|---|
| `teacher` | **0.6345** | **0.6318** | 0.8325 |
| `stopped` | **1.0000** | **1.0000** | 0.6015 |
| `jerky` | 0.3756 | 0.1858 | 0.5000 |

⇒ the expert's own 93.61-PDMS path records an at-fault collision in **36.6 %** of banked frames, and
**the stationary candidate is the only one that is perfectly clean on both components.**
`NuPlanCollision` gates on `ego_moving` (`collision/nuplan_collision.py:42-43`), so a stopped ego can
*never* register a collision — the two components together therefore approximate a **stop detector**,
and a scorer trained on them learns that standing still is the safest thing to do. That is the
NavSim-warmup STOP pathology in a new costume.

⚠️ **The `variance_report` guard cannot see this**, because the component *does* vary within the frame
(stopped vs moving). The guard asks *"does it vary?"*; the failure is *"it varies for the wrong
reason."* Same family as the guard the third review already replaced once.

**Minimal fix (describe only, not applied):** splice `agent_size_all` alongside the other per-step
tensors in `_splice_npc_futures`, or — safer — restrict the splice to agents that are valid in BOTH
`sd`'s history and the log, and leave the rest frozen. Add a control that must read a known value:
the teacher's own path must score `collision.info == 0` on a frame where the teacher did not crash.

### N2 — ⛔⛔ BLOCKING · The splice is indexed by tail, not by prefix

`_splice_npc_futures(out, log_sd, T)` receives `T` = the **prefix** length k, and writes
`op[:, 1:a, -n:, :] = lp[:, 1:a, -n:, :]`. `op`'s last k rows are candidate steps **1…k**;
`lp`'s last k rows are log future steps **(21−k)…20**.

**MEASURED by calling the real function with row-index-valued tensors (`scratchpad/m1_splice_and_frustum.py`):**

| prefix k | ego steps present | NPC log rows written | NPC time lead |
|---|---|---|---|
| 3 | 1,2,3 | 18,19,20 | **+3.40 s** |
| 5 | 1…5 | 16…20 | +3.00 s |
| 9 | 1…9 | 12…20 | +2.20 s |
| 15 | 1…15 | 6…20 | +1.00 s |
| 19 | 1…19 | 2…20 | +0.20 s |
| 20 | 1…20 | 1…20 | **0.00 s — the only aligned prefix** |

At the shipped `--stride 2`, prefixes are {3,5,…,19,20}: **9 of 10 are misaligned.** Effect on the
bank, MEASURED over 3 logs × 2 steps × 6 candidates × 6 components = 216 cells: **15 cells change**
when the splice is aligned, **all 15 in the direction "manufactured collision / TTC 0"**, twice on the
teacher's own path (`485e78d3d403@30`, `@60`).

**Minimal fix:** `T0 = op.shape[2] - T` is the history length; slice `lp[:, 1:a, T0:T0+n]`. I verified
this produces the identity mapping at every prefix.

### N3 — ⛔ HIGH · The monotone latches carry an artefact past the filter built to exclude it

`_comfort_triggered` (`comfort.py:177-187`) and `_nuplan_ttc_triggered` (`nuplan_ttc.py:167-173`) are
`prev | triggered` latches that permanently multiply their score by **0.75** and **0.5**. The rollout
arms them from prefix **k = MIN_PREFIX = 3**, while `KEY_MIN_PREFIX = {"comfort": 10}` — added this
round, precisely because *"below 15 history samples it is reading a boundary, not the candidate"* —
filters only the *values* from k < 10.

**MEASURED counterfactual (`scratchpad`, arm B starts the carry dict at k = 10):**

| candidate | comfort @k=3 | A: banked (SHIPPED) | B: banked, no early latch | ratio |
|---|---|---|---|---|
| teacher | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **lat−4 / lat−2 / lat+2 / lat+4** | **0.5000** | **0.7500** | **1.0000** | **0.7500** |
| lon x0.5 / stopped | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| lon x1.5 / jerky / reverse / over-curb | — | 0.3750 | 0.3750 | 1.0000 |

**4 of 11 candidates are affected, and only the lateral family.** 0.7500 is exactly
`_apply_triggered_penalty`'s multiplier. Un-latched, the smooth lateral offsets read **1.0000 —
identical to the teacher**, which is the physically correct answer and is what the third review
itself predicted. The "comfort now discriminates" signal for the lateral family is therefore an
artefact of a prefix the code declares inadmissible.

**Minimal fix:** start the rollout at `max(MIN_PREFIX, max(KEY_MIN_PREFIX.values()))` for the latching
components, or reset the latched attributes when crossing into the admitted range.
**And add the discriminating arm:** the deliberate violator (`jerky`) must separate from the smooth
lateral family — on the live bank both read **exactly 0.5000**, i.e. today they do not.

⚠️ A second, smaller instance in the same family: with `--stride > 1` each call advances the
candidate by `stride × TRAJ_DT` while `_direction_progress_buffer`'s 1.0 s window counts **calls**, so
the direction detector's window is `stride`× wider in real time. `--stride` is documented as a cost
knob; it changes a metric's definition.

### N4 — ⛔ HIGH · `pos3d` is a table by another route, and its intrinsics are in the wrong units

Two independent problems, both MEASURED by calling `REFe._frustum` on a stub self
(`scratchpad/m1_splice_and_frustum.py`):

**(a) Wrong units.** The corpus is **1920×1080** with `fx 1545.000, cx 960.000, cy 560.000`
(`raw/refe_cam_calibration_variance.txt:14`, 1,349/1,349 DBs read). The network input is
**960×512** (`train.py:298`, `planner.py:260` — a non-aspect-preserving `cv2.resize`). The config
carries the **native** intrinsics unrescaled.

| quantity | as coded | correct for a 960×512 input |
|---|---|---|
| fx / cx | 1545.0 / 960.0 | **772.50 / 480.00** |
| fy / cy | 1545.0 / 560.0 | **732.44 / 265.48** (the two scales differ: 0.5000 vs 0.4741) |
| ray tan(x) | **[−0.6162, −0.0052]** | [−0.6110, +0.6110] |
| horizontal span | **31.34°**, boresight offset −0.30° | **62.85°**, symmetric |
| realised x / y over all 1920 tokens | x ∈ [−0.6162, −0.0001], y ∈ [−0.3573, −0.0006] — **neither crosses zero** | symmetric about 0 |

The principal point lies **outside the token grid**, so every patch is encoded as lying left of and
above the optical axis. The docstring's *"normalise to ~[−1, 1]"* (`model.py:470`) is also not what
the code produces: the realised span is **[−0.6162, +1.0000]** and only `z` reaches 1.

⚠️ Related, and worth one line: `pos3d_depth_bins = 2` with `linspace(1, 60)`. Dividing by
`pos3d_far_m` makes the **near** bin's coordinates 1/60 of the far bin's — x ∈ [−0.0103, −0.0000]. The
near bin is close to information-free, so the 6-dim input is effectively 3 useful numbers.

**(b) It does not condition on geometry.** `_frustum(gh, gw, device, dtype)` takes **no image and no
per-sample calibration**; two independent builds are bit-identical; it yields **1920 distinct vectors
for 1920 tokens**. The justification in `model.py:388-396` and `code/probe_cam_calib.py:107-109` is
that a table *"CANNOT condition on the camera geometry that varies across our own corpus"* (2 distinct
intrinsics, 22–24 distinct extrinsics) — **that argument applies verbatim to the shipped MLP**, which
reads one hardcoded intrinsic and **no extrinsics at all**. PETR's PE conditions per-sample precisely
because it maps the frustum into the ego frame with the per-sample `img2lidar`; that step is absent.

⇒ The change is defensible as a **smoother, 372×-cheaper reparameterisation of a per-token bias**
(528,896 params vs 1,966,080), and that is how it should be described. It is **not** what the
docstring and `PAPER_CONFORMANCE_REVIEW.md` D7 claim it is.

**Minimal fix:** rescale the intrinsics to the network input (`fx*W/1920`, `fy*H/1080`, likewise the
principal point) — or, better, store the native intrinsics and the input geometry separately and
derive. Then either (i) accept and *document* that with one camera and a fixed resize the encoding is
a constant, or (ii) pass per-sample `K` and `RT` through and make the claim true.

### N5 — ⛔ HIGH · `REFePlanner._hold()` returns a zero-duration trajectory, and no instrument runs it

`_hold` builds `InterpolatedTrajectory([ego] + [ego] * horizon_steps)` — 21 states that all carry
`ego.time_point`. MEASURED on a real `EgoState`:

```
_hold returned InterpolatedTrajectory with 21 states
  distinct timestamps: 1 of 21
  get_state_at_time(t0)      -> AssertionError: angle is not finite
  get_state_at_time(t0+0.5s) -> AssertionError: Interpolation time ... not in trajectory time window!
```

scipy divides by a zero x-spacing and returns NaN. In the real harness the **first** unresolvable
frame would therefore crash the simulation, not hold. `planner.py:159-160` and
`diag_planner_holds.py` arm 3 both assert the opposite — and the diagnostic passes **only because it
replaces `_hold` with `lambda ego: "HELD"` at line 111**. The instrument written to prove the hold
path is safe never executes the hold path.

**Minimal fix:** build each held state with `ego.time_point + TimePoint(dt_us*i)` via
`ES.build_from_rear_axle`, exactly as `_to_trajectory` already does; then add an arm that calls the
real `_hold` and requires `get_state_at_time(t0 + horizon)` to return.

### N6 — ⛔ HIGH · The live default scorer bank predates its own builder, and the consumer falls back silently

`train.py:508` defaults `--scorer-targets` to `D:/Projects/TanitAD/data/refe_scorer_targets_full`.

| fact | MEASURED |
|---|---|
| bank written | **2026-09-20 21:57:54** |
| `score_proposals.py` last modified | **2026-09-20 22:33:10** (35 min later) |
| rows carrying `dac.violation` | **0 / 2,165** |
| rows carrying any `raw.*` key | **0 / 2,165** |
| rows with `off_road.OffRoad.info` ∈ {3,4} | **173 / 2,165 (8.0 %)** — the `train.py` fallback reads every one as **CLEAN** drivable-area |
| control: rows carrying `off_road.OffRoad.info` at all | **2,165 / 2,165** — so "0 dac rows" is a claim about CONTENT, not a failed read |
| rebuild of one banked frame with today's code | **6 of 40** numeric cells differ; `progress.advance_m` reproduces to 4 dp for every candidate, so the rig is otherwise identical |

The six differing cells: `teacher`/`lat-2` `ttc_reward` **1.0000 → 0.0000** (the N1/N2 phantom),
`lat-2`/`lat+2` comfort **0.5000 → 0.7500** (N3), `jerky` collision **1.0000 → 0.0000**, `jerky`
comfort **0.5000 → 0.3750**.

⚠️ `ScorerBank` has a staleness guard for **`rank`** (`train.py:348-352`, loud and correct) and **none
for the two fields this round added**. The consumer's `dac.violation` fallback (`train.py:201`,
`:205-206`) is the exact `cat == 1` test `score_proposals.py:561-568` documents as defective.
**Minimal fix:** count rows missing `dac.violation` at load and print the same loud warning the rank
guard prints; refuse if the fraction is 100 %.

### N7 — ⚠️ MEDIUM · The detach guard watches one of the two detach points

`model.py` detaches twice: `traj.detach()` (`:517`) and `visual_ctx.detach()` (`:519`).
`validate_model.py:110` inspects **only `m.queries.grad`**.

**Constructed regression (`scratchpad/m6_detach_instrument.py`): remove only `visual_ctx.detach()`.**

| arm | validator verdict | modules receiving gradient from a score-only loss |
|---|---|---|
| shipped | GREEN (correct) | `score_q_mlp`, `score_dec`, `score_head` only (78 tensors) |
| R1 `detach_scorer=False` (the documented arm) | **RED** ✓ | 12 modules incl. `queries`, `dec`, `traj_head` |
| **R2 remove `visual_ctx.detach()` only** | **GREEN — the arm cannot see it** | `scene_proj` ✓, `pos3d_mlp` ✓, **`backbone` LoRA** ✓, `score_*` |

The score loss would then train the shared representation and the frozen trunk's adapters — the
paper's requirement, violated, invisible. **Minimal fix:** assert `scene_proj.weight.grad` and
`backbone.blocks[0].attn.q.A.grad` are also None/zero in the same arm.

### N8 — ⚠️ MEDIUM · `_derive_ego_kinematics` seeds yaw only when orientations were extended

`inject_proposal:171` extends `agent_orientation_all` **only if `yaw is not None`**. When a caller
passes `yaw=None` (a supported signature), orientations stay 5 rows while positions grow to 25, so
`_derive_ego_kinematics` falls through to `atan2(vel)` **and** skips the anchor seed at `:331` — the
yaw-rate seam the third review's fix removed comes straight back, silently. Production always passes
yaw, so this is latent, not live. **Minimal fix:** extend orientations unconditionally (holding the
last value when yaw is None), or refuse `yaw=None`.

### N9 — ℹ️ LOW · `diag_rope.py`'s "ANGLE SPAN" row invites the conclusion its own check was fixed to avoid

The row prints `reference y ∈ [−5.498, +5.498]` against `REFe y ∈ [−2.356, +2.356]` — because the
reference side is the **unwrapped** analytic ladder and the REFe side is recovered via `atan2` and
therefore **wrapped**. The comment at `:190-196` records exactly this trap for the *check* and leaves
it in the *display*. A reader comparing those two rows would conclude a divergence that the cos/sin
tables (`mean |ref − ours| = 0.0000`) and the feature comparison (`rel L2 0.00000`) refute.

### N10 — ℹ️ LOW / UNVERIFIED · The rope's instantiation arguments are INHERITED, not PUBLISHED

The banked primary gives the **class**; it does not give the arguments the DINOv3 ViT-L is built with.
`normalize_coords="separate"` and `base=100.0` are **class defaults**
(`Library/refs/dinov3_rope_position_encoding.py:22,25`), and `"separate"` vs `"max"` differ materially
on a **32 × 60** grid. `diag_rope.py` transcribes the `"separate"` branch by hand rather than
importing the banked file. The transcription is faithful (I checked it element by element), but the
*choice of branch* is unverified for this checkpoint. Related, and in the package's favour: the banked
reference registers `periods` as a **persistent** buffer, so Meta's own checkpoint would carry a rope
tensor — MEASURED, **all three timm checkpoints on D: carry zero rope/position tensors**, so
`model.py:147-148`'s *"DINOv3 stores no rope tensor"* is **true for the mirror we use** and should say
so. **The cheapest closing move:** import the banked file directly in `diag_rope.py` instead of
re-deriving it, and cross-check the trunk against `timm`'s own DINOv3 — `timm` is **not installed**
and I did **not** install it (documented torch-breaking hazard).

### N11 — ℹ️ LOW · Small correctness / usability items

* `validate_model.py:31-32` — **`--device cpu` is ignored**: the ternary falls through to
  `"cuda" if torch.cuda.is_available()`. MEASURED: the run reports `forward/backward on cuda` when
  asked for cpu.
* `validate_model.py:5,12` — the docstring promises *"5. winner-takes-all really routes gradient to
  exactly one proposal"*. **No such assertion exists**; lines 99-103 print `idx` and whether a grad is
  present, and add nothing to `fails`.
* `validate_model.py:6` — *"REFe is ViT-S with one camera"*. The default config is **ViT-L**.
* `planner.py:192-201` states the aggregation as `EP(5)·TTC(5)·comfort(2)` over a **denominator 12**;
  `PDM_W` at `:205` is `(5,5,4)` = **14**, and `:227-230` says 14. The code is 14; lines 192-201 are
  stale and three lines above the constant that refutes them.
* `planner.py:296-301` — every future state of an emitted trajectory carries the ego's **current**
  velocity, acceleration and steering angle, which is inconsistent with the emitted positions. Benign
  under a pure position tracker; not benign for any metric that reads velocity off the plan.
* `model.py:365` — `super().__init__()` is called **twice** in `RegisterCompress.__init__`. Harmless
  as written (nothing is registered between the two calls) and worth removing.
* `model.py:260` — the class is still named `VitS16` and is used for ViT-L.
* `score_proposals.py:212-216` — `update_nearest_neighbors()` is wrapped in a bare
  `except Exception: pass`, so a failure leaves stale KNN indices feeding the collision and TTC
  detectors with no signal. Given N1, this path deserves a counter rather than a silent pass.
* `score_proposals.py:190` — when `build_log_scenario_data` raises, `log_sd = sd` and the splice is
  **skipped entirely** with no message and no counter. A bank could be half-spliced and look uniform.

### N12 — ⛔ HIGH (operational) · Three staged blobs are STALE, and a commit now would land the older code

Not a code defect — a **commit hazard**, and it sits on the two most load-bearing files in the
package. MEASURED at 23:08 today, with all three blobs read and length-asserted:

| path | git status | index blob | worktree blob | index lines | worktree lines | lines present ONLY in the staged blob |
|---|---|---|---|---|---|---|
| `refe/model.py` | `AM` | `7d7fbc5f…` | `3737d03d…` | 560 | 569 | **2** |
| `refe/validate_model.py` | `AM` | `44d499ab…` | `730064be…` | 166 | **156** | **17** |
| `raw/refe_validate.txt` | `AM` | `85c102ad…` | `c864375e…` | 35 | 35 | **35 — a different run entirely** |

None of the three is in `HEAD` (they are new files), so `git commit` on the current index would
commit the **pre-22:35 revisions**. `validate_model.py`'s staged blob is not a subset of the
worktree's — 17 lines exist only in the staged version — so this is a genuinely older revision, not
a truncation. This is the `CLAUDE.md` *"stale content hides behind a plain `M`"* class: the listing
shows `M`, which is indistinguishable from a sibling's legitimate staged edit.

⚠️ For the record and to close the obvious question: **I did not cause this.** `model.py` and
`validate_model.py` were last written at **22:35:12** and `raw/refe_validate.txt` at **17:32:18**;
my first measurement ran after 22:35 and I wrote no file in the repo except this review (23:08:04).
**Minimal fix:** `git add` the three paths before any commit, and verify index == worktree by blob
comparison with both sides length-asserted.

---

## 4 · Instruments audit — with a constructed regression for each

Every instrument was RUN today. All pass. The column that matters is the last one.

| instrument | run today | constructed regression | does it go RED? |
|---|---|---|---|
| `load_dinov3.py` CONTROL 3 (no unconsumed checkpoint tensors) | `0 unconsumed` on ViT-S | swap a block's `mlp.fc1` for a differently-shaped one | **YES** — `cp()` raises on a shape mismatch |
| `load_dinov3.py` "target tensors NOT loaded and NOT expected" | `missing = []` | **add `blocks.0.gamma_3`, an un-checkpointed frozen Parameter** | ⛔ **NO.** MEASURED `missing = []`. The `covered` test at `:113` reduces the loaded name `"blocks.0.q"` to the prefix `"blocks.0"`, so **every** parameter named `blocks.N.*` is trivially covered — i.e. 99.4 % of the trunk. The docstring's *"95 % pretrained and 5 % random"* scenario is exactly what this cannot detect |
| `load_dinov3.py` CONTROL 2 (15 tensors per block) | PASS | drop up to **6** tensors anywhere | ⛔ **NO** — the threshold is `len(loaded) >= 15*depth` and `loaded` carries 6 non-block entries, so it absorbs 6 losses |
| `validate_model.py` check 2 (LoRA identity at init) | `0.000e+00` PASS | **set `LoRALinear.scale = 0.0` permanently** | ⛔ **NO.** MEASURED PASS, and total \|grad\| on every backbone LoRA tensor is **0.000e+00** — the only trainable part of the frozen trunk can never learn, and no arm anywhere reports it. Check 3 still passes because `requires_grad` is `True`. **Minimal fix:** after one optimiser step, assert at least one LoRA `B` has moved |
| `validate_model.py` check 4 (scorer detached) | PASS | `detach_scorer=False` | **YES** (R1) |
| same | PASS | **remove only `visual_ctx.detach()`** | ⛔ **NO** — N7 |
| `validate_model.py` check 5 (WTA routes to one proposal) | — | any change to WTA | ⛔ **NOT IMPLEMENTED** — promised in the docstring, absent from `fails` |
| `validate_model.py` peak-memory assert | `9.50 GiB`, `< 44 GB` PASS | double the activation cost | ⛔ **effectively NO** — the only asserted bound is the A40's 48 GB, ~4.6× today's peak |
| `diag_rope.py` | 5/5 PASS, exit 0 | revert to `repeat_interleave(2)` / raw indices / drop the 2π | **YES** — this is the one instrument that provably separates a correct rotary encoding from a plausible wrong one |
| `diag_architecture.py` D2 | PASS | re-concatenate registers instead of compressing | **YES** — context shape and storage-pointer checks both fire |
| `diag_architecture.py` D7 | PASS | replace the rope with a *different* rotation | ⛔ **NO** — the three arms (table gone / vectors moved / norm preserved) pass for any rotation. Acknowledged in `diag_rope.py`'s own header; `diag_rope.py` now covers it |
| `diag_architecture.py` D6 | PASS | restore the mean-over-channels WTA | **YES** — the counterexample flips |
| `diag_planner_holds.py` arms 1/2/4 | PASS | remove the refusal from `compute_planner_trajectory` | **YES** — the arms call the real method |
| `diag_planner_holds.py` arm 3 (the hold is safe) | PASS | **the real `_hold` is already broken** | ⛔ **NO** — the arm stubs `_hold` (`:111`). MEASURED the real one raises. See N5. **Also:** `make()` bypasses `__init__` and sets `max_consec_holds` itself, so deleting that assignment from `REFePlanner.__init__` leaves the diagnostic green |
| `diag_scorer_components.py` D3 ("comfort now VARIES") | PASS | make comfort constant except for the k<10 boundary artefact | ⛔ **NO** — MEASURED, that is *already the situation for 4 of 11 candidates*, and the arm passes on 0.75× values. It also cannot see that `jerky` (the deliberate violator) and the smooth lateral family read **exactly 0.5000** on the live bank. **Minimal fix:** require `jerky < lat±2` |
| `diag_scorer_components.py` STRUCT "no dead code after a return" | PASS | put `if True: return out` mid-function | ⛔ **NO** — the AST scan inspects only the **top-level** statements of `inject_proposal` |
| `diag_scorer_components.py` dt guard | PASS (2.884 vs 2.816 m/s) | halve `TRAJ_DT` | **YES** — the reference is the log's own speed at t0, independent of the quantity under test. This is the strongest arm in the package |
| `diag_scorer_components.py` D5 aggregation | 5/5 PASS | restore the logit sum | **YES** — a named control asserts the old rule picks the colliding candidate |
| `diag_schedule.py` | 5/5 PASS | remove the scheduler | **YES** — `--no-cosine` control stays flat |
| `scorer_gate.py` | not re-run (needs a scene with a curb in range) | the splice phantom of N1 | ⛔ **NO** — its three checks are about the teacher being clean, the over-curb path being caught, and rollout ≥ endpoint. MEASURED: the teacher reads `collision.info` 1.0 from a zero-area phantom on 2/2 frames and **the gate does not look at collision at all** |
| `ScorerBank.variance_report` | runs at every training start | make a component vary only because the stationary candidate differs | ⛔ **NO** — N1's collision/ttc pattern passes at 73.6 % / 95.9 % "varies in" |
| `ScorerBank` rank-staleness guard | 0 legacy rows on the live bank | build a bank without `dac.violation` | ⛔ **NO** — there is no equivalent guard; MEASURED 0/2,165 rows carry it and nothing says so |
| `train.py --break-optimizer` | not re-run | hand the optimiser an empty parameter set | **YES** — the model cannot change, so the overfit threshold cannot be met |

**Summary: 13 instrument arms cannot go red on a regression I was able to construct** — a different
set from the nine Review 3 reported, and counted by tallying the ⛔ rows above rather than by estimate.
Four of them (`load_dinov3`'s missing-target control, the LoRA-identity check, `diag_planner_holds`
arm 3, and `diag_scorer_components` D3) guard something that is **already broken or already an
artefact today**, which is the property that makes an inert guard worse than no guard.

---

## 5 · The teacher result — admissibility

**Claim:** the released DriveRL teacher scored **89.755151** over **272/272** Test14-hard non-reactive
scenarios, 0 failures.

**Verdict: CONFIRMED as computed. Tier: closed-loop nuPlan CLS, non-reactive, 15 s @ 10 Hz.
Evidence class: MEASURED (ours). NOT admissible as a reproduction of the paper's published figure.**

| sub-claim | measurement | source |
|---|---|---|
| headline | `final_score` 0.8975515117148831 → **89.7551511715**; recomputed from the 272 per-scenario rows with the official CLS formula, max residual **1.11e-16** | `C:/dzo/h0920/closed_loop_nonreactive_agents_driverl_test14hard_nr/aggregator_metric/…2026.09.20.21.30.22.parquet` |
| 272 scenarios | 272 per-scenario rows, 272 `runner_report` rows, 272 token dirs, 272 `.msgpack.xz` — **all four token sets exactly equal, symmetric difference 0** | same run dir |
| 272 is the expected count | filter declares 272 unique tokens, `num_scenarios_per_type: null`, `limit_total_scenarios: null`; the paper states "Test14-hard (272)" | `DriveRL/nuplan-devkit/nuplan/planning/script/config/common/scenario_filter/driverl_test14_hard.yaml:18-294` · `drivezero_report.pdf` |
| 0 failures | `succeeded=True` 272/272, `error_message=None` 272/272; `exit_on_failure: true` and no try/except in `simulations_runner.py`, so a failure could not have been swallowed | `runner_report.parquet`; `…/simulation/runner/executor.py:28,43-49` |
| no silent degradation | 149–150 steps per scenario spanning 14.794–14.900 s (no short run); ego progress **72.72 m** mean vs expert **72.88 m**; the 14 zero-score scenarios have identifiable causes | `…/metrics/ego_jerk.parquet`, `…/metrics/ego_progress_along_expert_route.parquet` |
| missing-metric hazard | `_compute_scenario_score` silently `continue`s past a `None` column, which for a multiplier can only **raise** the score. Audited: **0 of 272** rows null in any of the 8 scoring columns | `…/metrics/aggregator/weighted_average_metric_aggregator.py:112` |
| aggregation basis | per-scenario micro-average. The macro mean over the 14 type rows would be **89.949956** — a +0.19 inflation that was **not** taken | same parquet |
| weighting is the published one | multipliers NC · DAC · progress · DDC; weighted EP 5.0, TTC 5.0, speed-limit 4.0, comfort 2.0 | `…/metric_aggregator/closed_loop_nonreactive_agents_weighted_average.yaml:7-19` |
| scoring path uncontaminated | `grep -rn "driverl"` → **0 hits** under `metrics/` and `simulation/` (control: 3 hits under `script/`, 47 `.py` files under `metrics/`) | DriveRL tree |
| oracle-future leak | `compute_planner_trajectory` builds `log_scenario_data` only when `future_steps != 0`; the teacher config declares **0** | `src/driverl/nuplan/planner.py:228-238`; `release/configs/driverl_teacher.yaml:106` |
| checkpoint | `checkpoint_2400.pt`, `update=2400`, **5,702,413** params = the paper's "5.7M"; `strict_checkpoint: true` | `release/checkpoints/checkpoint_2400.pt` |

**Why it is not a reproduction of 89.97.**

1. **It is 0.214849 BELOW the published figure.** PUBLISHED: DriveRL Test14-hard NR = **89.97**
   (`drivezero_report.pdf`, main table and Table 3). Ours: **89.755151**.
2. ⛔ **The rig is not reproducible across launches, and this is MEASURED, not assumed.** An
   identical-configuration replicate pair already on this machine — `C:/dzo/m-nr-n` vs
   `C:/dzo/m-nr-r1`, whose `overrides.yaml` differ only in output path and experiment name — scored
   **96.849837** vs **97.784383**, a spread of **0.934547 points with 8 of 10 scenarios scoring
   differently**. There is **no replicate of the Test14-hard run**, so 89.755151 is **one draw and
   carries no interval**. Scaling the replicate's per-scenario delta SD to n = 272 gives an
   **ESTIMATED** run-to-run SE of ≈**0.20 points** — the gap to 89.97 is ≈1.1 SE, i.e.
   indistinguishable in either direction. One scenario flipping 1.0 → 0.0 moves the headline by
   **0.367647**.
3. **Protocol caveat for cross-method comparison only.** The controller is
   `DriveRLOneStageController`, not nuPlan's stock LQR + kinematic-bicycle two-stage controller. This
   is **declared in the paper** (Table A4's constants byte-match
   `driverl_one_stage_controller.yaml:8-33`), so DriveRL-vs-DriveRL is fine — but the baseline rows in
   the same published table were produced under the stock controller, which carries tracking error
   this one does not.

⇒ **Quote it as: "MEASURED (ours), closed-loop nuPlan CLS non-reactive, Test14-hard 272/272, 89.755151
— one draw, no interval; the rig's MEASURED run-to-run spread on a 10-scenario replicate pair is 0.93
points."** The cheapest thing that would close the gap is **one unchanged re-run of `h0920`**, reported
as a pair.

⚠️ **Not verifiable offline, and stated rather than hidden:** the devkit fork was not byte-diffed
against upstream (single squashed commit, no `.git` under `nuplan-devkit/`); the per-step planner
invocation count is inferred from `policy_interval_s` defaulting to 0.0 and is not recorded in any
artifact; and `worker.log_to_driver=false` means the 272 worker logs **do not exist**, so the
"no degradation" finding rests on the runner report, `exit_on_failure`, and metric content — not on
logs.

---

## 6 · Stale documentation

MEASURED ground truth, `REFe(REFeConfig())` instantiated today:
**322,021,958 total / 18,942,530 trainable / 5.8824 %**, `reg_compress` 6,303,744,
`pos3d_mlp` 528,896, `ego_enc` 77,312, `dec` = `score_dec` = 4,213,760 (4 layers each),
`score_q_mlp` 81,408, `scene_proj` 262,400, `backbone.frozen` 303,079,424, `backbone.lora` 3,145,728.
4-camera fidelity check: **322,071,110 / 18,991,682 / 5.8967 %**.

### Blocking

| document:line | says | is | note |
|---|---|---|---|
| `refe/model.py:18, 48, 64` | "316.9 M / 11.88 M trainable" | 322.02 M / 18.94 M | **the source lies about itself** |
| `POD_HANDOFF.md:49` | 316.9 M / 11.88 M | 322.02 M / 18.94 M | contradicts line 137 of the same file |
| `POD_HANDOFF.md:137-138` | 18,934,338 / 322,013,762 | 18,942,530 / 322,021,958 | Δ = exactly the Fourier goal encoding |
| `REFE_MODEL.md:28` · `REFE_PLAN.md:128` | 329.7 M / 26.58 M / 8.06 % | 322.02 / 18.94 / 5.88 | |
| `REFE_MODEL.md:53, 57` · `REFE_PLAN.md:137` | 4-cam fidelity 5.52 %, "within **0.03** pp" | **5.8967 %**, **0.41 pp** | the independent fidelity check |
| `REFE_PLAN.md:128` | "trainable is now **43 % ABOVE** 18.58 M" | **+1.95 %** | |
| `MODULE_SIZING_STUDY.md:61, 579` | `score_head` → 396,294 (six per-component MLPs) | still `nn.Linear(256, 6)` = **1,542** | recommendation **not implemented**; the study's totals assume it is |
| `MODULE_SIZING_STUDY.md:57, 478-479, 573` | goal encoding at **8** frequencies / 64 channels | `n_goal_freq = 4` → 36 channels | implemented at half strength, undocumented |
| `POD_HANDOFF.md:60, 141` | peak **9.58 GB** | MEASURED **9.50 GiB** today; `raw/refe_vitl_devbox_memory.txt` is **0 bytes** | the cited artifact is empty; the only raw peak is `raw/refe_validate.txt:33` = 9.42 GB **from the pre-fix architecture** |
| `POD_HANDOFF.md:15`, `code/refe_expt_cost.py:4-5` | the 0.780 s/sample basis cited to `raw/refe_wta_batch_starvation.txt` | that file contains **no timing at all** | **every** A40-hour figure descends from it (6.3 / 25.3 / 55 / 608 / 222) |
| `POD_HANDOFF.md:36` vs `:54` | arm A 6.3 h computed on 1,746 tuples; the same file states the bank as 2,182 | → **7.88 h** | |
| `POD_HANDOFF.md:38, 42` vs `MODULE_SIZING_STUDY.md:737` | arm C ~55 h vs **222.08** h | 4.0× contradiction | arm C was never recomputed |
| `POD_HANDOFF.md:42` | "it costs under three GPU-hours" | the same file's `:128-130` says this sentence **"is withdrawn"** | the correction did not edit the sentence it withdrew |
| `POD_HANDOFF.md:97` · `REFE_MODEL.md:15, 145, 155` | the scorer targets are "not yet wired into `train.py`" | wired since `train.py:101-175, 344, 424` | flagged at `PAPER_CONFORMANCE_REVIEW.md:448` and **still uncorrected** |

### Stale — the three prior reviews' own tables

* `PAPER_CONFORMANCE_REVIEW.md:96, 439, 566` — "319,031,874 / 13,986,370 / 4.384 %" → now
  322,021,958 / 18,942,530 / 5.8824 %. `:396-397` "30.1 % of trainable" → **22.25 %**.
  `:111` "no camera intrinsics read anywhere" → SUPERSEDED (but see N4).
* `FIX_VERIFICATION_REVIEW.md:66-68, 84, 203` — "+12,598,272, 43 % above 18.58 M" → `reg_compress`
  is **6,303,744**, +1.95 %.
* `REVIEW_3_FULL.md:52, 103, 113, 556-558, 564-572` — the whole parameter block is superseded; the
  "fidelity inversion" (9.693 % at 4 cameras) is itself undone (**5.8967 %**); the overshoot above
  sub-300M is **22.02 M**, not 29.66 M; and there are now **four** live headline counts, not three.
* `REVIEW_3_FULL.md:103` records the PI accepting `reg_compress` = **12,598,272**. The shipped module
  is **6,303,744**. **ESCALATE — no sign-off on the halving is recorded anywhere in the package.**
* `MODULE_SIZING_STUDY.md:626-641` — the GMAC figures were computed with the ratio-4 `reg_compress`
  and the 1,966,080 table. **UNVERIFIED** at the shipped config.
* `refe/README_DIAGNOSTICS.md:44` quotes `diag_scorer_components`'s key number as *"derived speed
  6.197 against displacement/time 6.196"* — that is the **algebraically blind** check the file itself
  says it removed (`:125-133`). The README advertises a refuted arm. It also lists 12 `diag_*` files;
  there are **13** — `diag_rope.py`, the strongest one, is undocumented.
* `REFE_MODEL.md:260, 264` still reports `comfort` varying in **0.0 %** of frames; the live bank
  reads **80.2 %** (and, per N3, for the wrong reason for 4 of 11 candidates).
* Cross-doc contradictions still open: target-bank size **1,746 / 1,964 / 2,182**; scenario count
  **8 / 10**; scorer frame coverage **14.6 % / 22.7 % / 25 % / 49.5 %**; augmentation diversity
  71.6 %/96.9 % in `REFE_MODEL.md:142-143` vs 73.3 %/97.5 % in its own artifact
  `raw/refe_target_bank.txt:32-33`.
* `RESULT.md:18` cites `raw/dz_budget.txt` — **the file does not exist** in this package.
* `refe/train.py:102` — `ScorerBank` docstring says the key is `(log_name, step)`; it is
  `(log_name, token, step, rank)` (`:173`).
* `POD_HANDOFF.md:78` — "refe/ (7 files)"; there are **26 `.py`** plus a README and 2 `.sh`.

---

## 7 · What I could not do, plainly

* **`timm` is not installed**, so I could not cross-check the trunk against an independently authored
  DINOv3 implementation. I did **not** install it (documented torch-breaking hazard). That check, plus
  importing the banked reference directly into `diag_rope.py`, is the cheapest remaining hardening of
  N10.
* **I did not re-run `h0920`**, so the teacher headline still has no interval. One unchanged re-run
  would give it one.
* **I did not rebuild the scorer bank** — it is a multi-hour CPU job and the fixes for N1/N2 must land
  first.
* `validate_model.py` insisted on running on CUDA (N11), so the CPU path is untested by me.
* The `MODULE_SIZING_STUDY.md` GMAC figures and the `0.780 s/sample` cost basis remain **UNVERIFIED**;
  I could not find an artifact behind either.

---

## 8 · Manifest

| artifact | where it lives | state |
|---|---|---|
| **this review** | `D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/REVIEW_4_FINAL.md` | **staged** (`git add`), not committed, not pushed |
| splice-alignment + frustum measurement | `…/scratchpad/m1_splice_and_frustum.py`, `m1_out.txt` | scratchpad only |
| splice effect on banked targets (1 frame, 2 arms) | `…/scratchpad/m2_effect_on_bank.py` | scratchpad only |
| splice generality + latch timeline (6 frames) | `…/scratchpad/m3_multiframe.py`, `m3_out.txt` | scratchpad only |
| collision root-cause probe | `…/scratchpad/m4_collision_root.py` | scratchpad only |
| **phantom three-arm control (A/B/C)** | `…/scratchpad/m5_phantom.py` | scratchpad only |
| **constructed regressions R1–R4 against the validator** | `…/scratchpad/m6_detach_instrument.py` | scratchpad only |
| validator run output (CUDA, batch 2) | `…/scratchpad/validate_out.txt` | scratchpad only |
| `diag_scorer_components` run output | `…/scratchpad/dsc.txt` | scratchpad only |
| teacher-audit scratch scripts | `…/scratchpad/recompute.py`, `tokens.py`, `replicate2.py`, `pdf3.py` | scratchpad only |
| measured `param_report` | `…/scratchpad/param_report.json` | scratchpad only |

Scratchpad root:
`C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/bd7d00af-b98e-42f1-a53c-cb4113059b0f/scratchpad/`

**No file in `D:/Projects/TanitAD` was modified except the addition of this review. No code was
changed — every fix above is described, not applied. Nothing under `C:/dzo/` or the DriveZero repo was
modified. Nothing was committed or pushed, and no branch was switched.**

⚠️ **Before anyone commits this package:** `refe/model.py`, `refe/validate_model.py` and
`raw/refe_validate.txt` carry staged blobs that are OLDER than the worktree (N12). `git add` those
three paths first, or the commit lands the pre-22:35 code.

### The three things the PI must decide

1. **Rebuild the scorer bank** after N1 and N2 are fixed — no REFe scorer or closed-loop number is
   admissible before that. Cost is CPU-hours, not GPU.
2. **`reg_compress` 12,598,272 → 6,303,744** was shipped without a recorded sign-off, against a PI
   decision the package records at the larger value.
3. **The `pos3d` claim** (N4): rescale the intrinsics, then choose — accept a constant encoding and
   say so, or plumb per-sample `K`/`RT` and make the PETR claim true.
