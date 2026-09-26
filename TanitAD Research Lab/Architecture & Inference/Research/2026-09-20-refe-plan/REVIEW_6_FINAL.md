# REVIEW 6 — did tonight's fixes land, and what did they break?

*Independent adversarial review, 2026-09-21 ~01:00–03:30 Europe/Berlin. Reviewer: Architecture &
Inference. No `refe/*.py` was modified by me. Every number carries its evidence class and artifact.*

Evidence classes: **MEASURED** (ours, with the path) · **PUBLISHED** (cited) · **ANALYTIC**
(arithmetic from source, no run) · **DERIVED** (arithmetic over our own measurements) ·
**INHERITED** (another agent/doc, stated as such) · **UNVERIFIED**.

⚠️ **The package moved twice under this review.** `diag_rank_distinctness.py` appeared at 01:03 and
the rank-1 target bank was rebuilt mid-pass. Every bank number below is **RE-MEASURED after** the
rebuild (`raw/2026-09-21-review6/bank_audit2_REMEASURED.txt`); the pre-rebuild measurement is kept
beside it because it is the independent confirmation of the defect the author self-reported.

---

## 1 · Headline verdict

# ⛔ NO — not ready to train on an A40 tomorrow. Three blockers, and the first is not an implementation detail.

The **model** is in good shape: every instrument I ran is green, the distortion inversion is
correct against an independently authored reference, and the `ego_dim` retraction is right for a
reason I verified from the paper rather than from the retraction. **What is not ready is the
corpus and the command line you would type.**

| # | blocker | class | why it stops an A40 run |
|---|---|---|---|
| **B1** | **The 4-camera bank is UNTRAINABLE. 3 of every 4 image paths point at files that do not exist.** MEASURED over the first 200 rows: `CAM_F0` **200/200** basenames resolve on disk; `CAM_L0` / `CAM_R0` / `CAM_B0` **0/200, 0/200, 0/200**. The same-breath control (CAM_F0 at 200/200, 31,500 JPEGs indexed) proves this is a claim about the **data**, not about my search. | MEASURED | D2 from Review 5 was never fixed and is not claimed as fixed — but "bank rebuilt, 4 distinct channel paths per row" reads as a usable bank, and it is not one. `build_targets.py` was run **without `--images`**, so it wrote DB-relative paths for all four channels without ever checking they exist. The builder's own `missing_image_file` counter reads **0** because that counter only runs when `--images` is given. |
| **B2** | **`--epochs` mis-derives the step count by exactly the accumulation factor**, so `TRAINING_TIME.md`'s own recommended command `--batch 4 --accum 64` would run **64× too many optimiser steps** and the cosine `T_max` would span the wrong horizon. | MEASURED + ANALYTIC | `train.py:436` computes `per_epoch = ceil(len(ds)/a.batch)` — the **micro**-batch — while `step` counts **optimiser** steps. This is Review 5 §4.4 item 4, verbatim, unfixed. |
| **B3** | **The scorer loss's per-batch normaliser was not fixed for accumulation** (Review 5 §4.4 item 1). `train.py:493` divides by the **micro**-batch's covered-candidate count, then `:525` divides again by `accum`. The result is an average of per-micro-batch means, which re-weights samples by how many covered neighbours they happened to share — and at the measured ~10 % coverage most micro-batches contribute **zero**, so the effective score-loss weight is scaled by (covered micros / accum). | ANALYTIC (source) | A true batch of 256 normalises once over all 256. This is not equivalence, and the flag's help text claims it "reproduces the GRADIENT". |

⚠️ **Two of these three were named in Review 5 with the fix written out.** B2 and B3 are not new
discoveries; they are Review 5 §4.4 items 4 and 1, carried forward while `--accum` shipped around
them. The commit added the accumulation *mechanism* and left the two things §4.4 said the
mechanism would not restore.

⭐ **What IS ready, and should be said plainly:** the architecture, the geometry, the schedule, the
guards, and the retraction. If the corpus existed, B2 and B3 are a few lines each.

### The one number that moved most under attack

**`TRAINING_TIME.md`'s optimistic end — "99 days" — rests on an A40/4060 multiplier of 3.0 that
none of the three sources the document itself cites supports.** All three land at **2.475 / 2.559 /
2.477**. Recomputed on the document's own evidence the answer is **116–128 days**, not 99–128
(MEASURED arithmetic, `raw/2026-09-21-review6/training_time_recompute.txt`). The pessimistic end is
sound; the headline range is 17 days too generous at the bottom.

---

## 2 · One row per claimed fix

| # | claim | verdict | evidence |
|---|---|---|---|
| **1** | **D1** `build_targets.py` crashed on its success path; fixed; bank rebuilt at `data/refe_targets_4cam` (2,182 tuples, 73.2 % coverage, 4 distinct channel paths per row) | ⚠️ **PARTLY VERIFIED — the crash is fixed, two of the three numbers are wrong** | The fix is real and correct: `:166` now `cam_ts[ch].size`, `:243` `sum(v.size for v in cam_ts.values())`, and the builder produced a bank. ⛔ **"2,182 tuples" DOUBLE-COUNTS.** MEASURED before the rebuild: rank0 and rank1 were **1,091/1,091 rows identical apart from the `rank` field**, union of `(token, step)` = **1,091**. *(The author self-reported this defect mid-review and rebuilt rank 1; my measurement is the independent confirmation, taken before that message arrived.)* After the rebuild, `0/1091` rows are identical — the fix is real — but the bank is still **1,091 distinct frames from 7 logs**, banked twice under two route variants. ⛔ **"4 distinct channel paths per row" is TRUE and MISLEADING**: 4 distinct paths per row, 4/4 channels named, **1 of 4 resolvable on disk** (B1). ✅ "73.2 % coverage" reproduces — but it is `kept/steps = 109/149`, i.e. **horizon truncation**, and has nothing to do with cameras: `dropped_short_horizon` is 40 on every log, `dropped_no_camera` 0. |
| **2** | **D3/D6** `validate_model.py` and `diag_architecture.py` fixed; `--backbone` added; `--device` made real | ⚠️ **VERIFIED for D6, INCOMPLETE for D3** | MEASURED, my runs: `diag_architecture.py` → **exit 0, `ARCHITECTURE_CONFORMS`**, 9/9 arms PASS. `validate_model.py --backbone vits16 --device cpu --batch 1` → **exit 0, `VALIDATE_OK`**, all six runtime arms reached including both constructed regressions. ⛔ **But section 6 — the peak-memory assertion — is guarded by `if dev == "cuda"` (`:205`, and the assertion itself at `:212-217`) and therefore still never runs on the path that now works.** D3's stated grievance was that "sections 4–6 **including the peak-memory assertion** have never executed"; 4 and 5 now execute, 6 does not, and under the shipped ViT-L×4 config it cannot (9.49 GiB against 8.00 GiB). The assertion has still never fired under any config. |
| **3** | **D4** `ego_dim`: Table A2 is the **Teacher's** schema, so citing it for the student was a scope error; now 7 kinematic scalars | ✅ **VERIFIED — and the reading of the paper is correct, checked independently** | PUBLISHED, verified by a second reader against the PDF, not against the retraction: Table A2's caption is **"Structured Teacher input schema"** (p. 25); its lead-in on p. 24 reads *"…the resulting structured Teacher inputs"*; it sits in **Appendix A, "DriveRL Supplementary Material" §A.2**, while the student's tables are A12–A14 in Appendix B. Its neighbours are actor history 15-D and vector-map 10-D — inputs a camera-only student does not have. **The paper dimensions the student's ego NOWHERE**: an exhaustive sweep of all 32 pages finds `9-D` exactly once (the Table A2 teacher row), and `9-dim`/`9D`/`7-dim`/`8-dim`/`ego status`/`proprio` **zero times**, against a same-breath control of `DriveZero` 69 / `ego` 55 / `Table` 45 in the same read. **Table A12, the only student configuration table, has no ego row at all.** All the paper says is (p. 7) the student takes the images, *the ego kinematics* and a navigation command, and (p. 8) that these are embedded into one ego token. ⇒ **The retraction is right and the scope error was real.** See §3 D-B for two caveats that do not overturn it. |
| **4** | **D5/D8** `planner.py` made 4-camera; `FrameStore` fall-through fixed | ✅ **VERIFIED in code, UNVERIFIED end to end** | `planner.py:222-226` now builds the 7-element vector and `:227-230` **raises** if it disagrees with `cfg.ego_dim` — a real consumer guard, not a comment. `diag_planner_holds.py` → **exit 0, `PLANNER_HOLDS_GUARDED`**, 4/4 arms PASS including two controls. ⚠️ I could not exercise the planner against real four-camera frames, because they do not exist (B1); the camera path is verified by reading, not by running. |
| **5** | **D7** Caltech inversion inside the cached frustum; round trip 1.6e-13; `undistort=False` ablation control | ⚠️ **THE MATHS IS VERIFIED. THE ABLATION IS NOT REACHABLE THE WAY THE PACKAGE'S OWN INSTRUMENTS MUTATE.** | ✅ MEASURED against an **independently authored** NumPy Caltech forward model (written from the OpenCV definition, not from `model.py`): `max |caltech_forward(model_ideal) − observed_distorted| = **1.510e-07**` in normalised coordinates over all 1,920 patch centres, median 5.93e-08. That is float32 round-trip noise. ✅ `undistort=False` reproduces the ideal pinhole `(u−cx)/fx` to **1.536e-07** — the control is honest. ✅ 20 iterations vs 200: **max |delta| = 0.000e+00**, the fixed point is fully converged. ⛔⛔ **But the `_frustum` cache key `(gh, gw, device, dtype, n_cam)` omits `undistort`, `cam_distortion` AND the extrinsics.** MEASURED on a live instance after one call: flipping `cfg.undistort` → **0.000000e+00**; zeroing `cam_distortion` → **0.000000e+00**; collapsing `cam_t`/`cam_q` → **0.000000e+00**. See §3 D-A. |
| **6** | **D11** the extra `visual_ctx.detach()` is declared as REFe's departure and is a config flag | ✅ **VERIFIED** | `model.py:145-153` declares it as a departure and names p. 8's actual wording; `REFeConfig.detach_scorer_context` exists; `train.py:615` exposes `--scorer-sees-trunk`; `validate_model.py:186-195` now says *"REFe's EXTRA detach of visual_ctx, not the paper's"* in the live output, which I read in my own run. The arm still asserts, which is right — an undeclared removal should still fail. |
| **7** | **D12–D17** comment falsehood, stale 12,598,272, 0-byte artifact, `--n-cameras`, dead line, rig-locking scoped | ✅ **VERIFIED, 6 of 6** | D12: `model.py:78-86` now states the paper is the whole justification and records the 70.76 %/22-of-70 refutation. D13: MEASURED `reg_compress = **6,303,744**` at ViT-L (`param_probe.txt`), matching the corrected comment. D14: `raw/refe_vitl_devbox_memory.txt` is **1,715 B**, no longer 0. D15: `train.py:612` `--n-cameras` exists and `model.py`'s guard message points at it. D16: `rel = rels[0]` is gone. D17: `model.py:146-156` scopes the claim to "resolution-free and per-camera", never "a function of the rig". |
| **8** | **Gradient accumulation**: `step` counts optimiser steps, loss divided by accum, clip+step+sched at the window boundary | ⚠️ **THE MECHANISM IS CORRECT. THE TWO THINGS REVIEW 5 SAID IT WOULD NOT RESTORE WERE NOT FIXED.** | ✅ `train.py:525-534` is right: `/a.accum`, `micro % a.accum: continue`, then clip → step → `zero_grad` → `sched.step()` exactly once per window, and `step` is incremented only there. ⛔ B2 (`--epochs` over-counts by `accum`) and ⛔ B3 (the score-loss normaliser) — see §1 and §3 D-C. |
| **9** | **NEW INSTRUMENT** `diag_consumer_conformance.py`, 8 arms, `--self-test` 8/8 RED alone | ✅ **THE SELF-TEST IS GENUINE. THE ARMS ARE NOT EXPRESSIONS OVER THE CODE THEY TEST.** ⚠️ Two gaps, one of which fires today | MEASURED: `--self-test` → **`SELF_TEST_OK`**, 8 arms RED under their own historical divergence with a live green control. The construction is right: code consumers are read from **source by AST** (`planner.py`'s literal 7, `train.py`/`build_targets.py`'s `CAMERAS` tuples), and arm 7 is a read-control. ⛔ **Its `--targets` DEFAULT points at the STALE bank** (`data/refe_targets`), where it reports **`CONSUMERS_DIVERGED`, exit 1**, failing arms 1–3. Run against `refe_targets_4cam` it is `CONSUMERS_CONFORM`, exit 0. A default that fails is a default nobody will trust. See §3 D-E for the divergence it does not catch. |
| **10** | **TRAINING_TIME**: camera scaling measured linear; ViT-L 4-cam 3.05–3.26 s/sample; full scale 2,380–3,080 A40-h = 99–128 days | ⚠️ **THE MEASUREMENT IS SOUND. THREE ARITHMETIC/ATTRIBUTION DEFECTS IN THE DERIVATION.** | Recomputed independently from `raw/2026-09-21-camera-scaling/cam_scaling.txt` — see §4. The linearity result reproduces exactly (3.8740 / 4.0824). The **3.26** upper bound, the **3.0** multiplier and the **"2.4–2.6× slower"** gap do not. |
| **11** | *(landed mid-review)* `--rank` is a label not a selector; rank 1 rebuilt; new `diag_rank_distinctness.py` | ✅ **THE DEFECT WAS REAL — I MEASURED IT INDEPENDENTLY BEFORE THE MESSAGE ARRIVED — AND THE FIX IS REAL** | MEASURED before: `1091/1091` rows identical apart from `rank`. MEASURED after: `0/1091`; `traj` differs on **1,091/1,091**, `goal` on **1,064**, `ego` on **1,054**; 4 s endpoint separation median **1.415 m**. ⚠️ But **25.7 % of the "new" rank-1 rows are within 0.50 m at 4 s and 9.0 % within 0.10 m** (min 0.0029 m) — the instrument's `!=` test cannot see that, and "1,091/1,091 trajectories differ" is a weaker claim than it sounds. See §3 D-D and D-F. |

---

## 3 · New defects, ranked

### ⛔⛔ D-A · BLOCKER-CLASS · the `_frustum` cache key omits the three things an ablation would change

`model.py:554` — `key = (gh, gw, str(device), str(dtype), n_cam)`.

It covers the **geometry**. It does **not** cover `cfg.undistort`, `cfg.cam_distortion`,
`cfg.undistort_iters`, or `cfg.cam_t` / `cfg.cam_q`.

MEASURED (`raw/2026-09-21-review6/frustum_cache_attack.txt`), on a live instance, after one call:

| mutation applied after one `_frustum` call | max abs delta | expected |
|---|---|---|
| `cfg.undistort = True → False` | **0.000000e+00** | > 0 |
| `cfg.cam_distortion = (0,0,0,0,0)` | **0.000000e+00** | > 0 |
| `cam_t` zeroed **and** `cam_q` collapsed to identity | **0.000000e+00** | > 0 |
| *(control)* two **fresh instances**, undistort True vs False | **0.148615** | > 0 ✅ |

⭐ **Why this is worse than a cache bug.** `model.py:573` states that `undistort=False`
*"is the ablation arm and reproduces the old ideal-pinhole behaviour EXACTLY — that is what makes
this change measurable rather than asserted."* It is measurable **only if the arm is built as a
fresh model**. The package's own guard instrument mutates config on a **live** model — and
`diag_guard_audit.py:118,121` has to call `net._pos3d_cache.clear()` by hand to make its `pos3d`
arm fire. So the hazard is already known, and it is patched **in exactly one arm**, by hand,
instead of in the key. Review 5's own `frustum_check.py:10` had to do `m._pos3d_cache = {}` too.

⛔ **And there is no `undistort` arm anywhere.** Not in `diag_guard_audit.py` (6 arms, none), not
in `validate_model.py`, not in `diag_consumer_conformance.py`. The single largest geometric change
of the night ships with a declared ablation switch and **zero instruments that exercise it**.
`train.py:387-388` does set the flag **before** `REFe(cfg)` is built, so the *training* ablation
`--no-undistort` is safe — it is the instrument-style mutation that is silently inert.

**Minimal fix:** put `(c.undistort, c.cam_distortion, c.undistort_iters, c.cam_t, c.cam_q)` in the
key, and add a seventh arm to `diag_guard_audit.py` whose mutation is `cfg.undistort = False` —
which, today, would report `CATCHES` while catching nothing.

### ⛔ D-B · HIGH · `ego_dim = 7` is right in principle, but the vector is 6 scalars and a copy — and it is NOT the teacher's 7

The retraction (§2 row 3) is correct. Two things about the *replacement* are not.

1. **Slot 7 is `math.hypot(vx, vy)`** — `build_targets.py:228`, `planner.py:225`. That is an exact
   deterministic function of slots 1 and 2. The comment at `build_targets.py:218` says
   **"SEVEN REAL KINEMATIC SCALARS — no pad, no constants"**; it is **six independent scalars plus
   one derived**. A redundant feature is harmless to a learner — but the sentence is doing
   rhetorical work against the 8-D version's "hardcoded trailing 0.0", and it overstates by one.
2. **The comment implies the 7 are the teacher's 9 minus the two geometry slots. They are not.**
   MEASURED from DriveRL's own released artifact (`release/checkpoints/checkpoint_2400.pt`,
   `model.kinematics_mlp.0.weight` has `in=9`), cross-checked against
   `src/driverl/agents/learning/vanilla_net_agent.py:122` and the concatenation at `:752-777`:
   the teacher's nine are **`[vx, vy, heading, length, width, steering_angle, steering_rate, accel,
   yaw_rate]`**. Drop `length`/`width` and the remaining seven are **heading, steering_rate and a
   single scalar `accel`** — REFe instead carries **`ax`, `ay` and `speed`** and carries **neither
   `heading` nor `steering_rate`**. ⇒ REFe's 7 is a **different seven**, which is *allowed* (the
   paper constrains nothing here — §2 row 3), but the code should not present it as a subtraction.

✅ **The "constant on nuPlan" justification is VERIFIED, and on stronger evidence than the comment
claims** (`raw/2026-09-21-review6/analysis_120.txt`, `probe_c_simlog.txt`). MEASURED over **120 log
DBs spanning 23 distinct vehicle ids and 4 cities** (Las Vegas, Boston, Pittsburgh, Singapore), by
**two independent routes**:

* **Route A — the devkit** (`car_footprint.vehicle_parameters`): every field reads `distinct = 1` —
  `length {5.176: 120}`, `width {2.297: 120}`, `wheel_base {3.089: 120}`, `vehicle_name
  {'pacifica': 120}`, `vehicle_type {'gen1': 120}`. **Control**: `rear_axle_x`, `rear_axle_y` and
  `time_us` each read **distinct = 119/120**, so 120 *different* files really were opened — the
  constancy is a claim about the data, not about the probe.
* **Route B — raw `sqlite3`, devkit not involved**: the only dimension-named columns anywhere in
  the schema are `camera.height/width`, `lidar_box.{h,l,w}` and `track.{h,l,w}`. **Zero** belong to
  an ego table (`ego_pose`/`log`/`scene`/`lidar_pc`). ⇒ nuPlan's DB **cannot** carry a per-log ego
  dimension, so the constancy is structural rather than a sampling accident.
* A third route on the rollout logs themselves: 597 ego states across 4 simulation logs →
  **1 distinct** `(length, width, wheel_base, rear_axle_to_center, half_width, name, type)`.

⇒ **Dropping `length`/`width` is right, and for a better reason than the comment gives.** Only the
"seven real kinematic scalars" wording (point 1) overstates.

### ⛔ D-C · HIGH · `--accum` breaks `--epochs`, and the one diagnostic that justifies `--accum` ignores it

**MEASURED**, `raw/2026-09-21-review6/accum_b.txt` and `accum_a.txt` — an 8-tuple bank,
`--batch 2 --epochs 2`:

| run | printed schedule | optimiser steps | micro-batches | samples consumed | epochs of data |
|---|---|---|---|---|---|
| `--accum 1` | `2 x 4 steps/epoch = 8 steps` | 8 | 8 | 16 | **2** ✅ |
| `--accum 4` | `2 x 4 steps/epoch = 8 steps` | 8 | 32 | 64 | **8** ⛔ |

The printed line is **identical**. `train.py:436` computes `per_epoch = ceil(len(ds)/a.batch)` at
the **micro**-batch size while `step` counts **optimiser** steps.

⇒ **`--epochs E --accum A` trains for `E × A` epochs of data.** `TRAINING_TIME.md` §6 recommends
**`--batch 4 --accum 64`** to reproduce the paper's effective 256. Typed together with the paper's
`--epochs 25`, that is **1,600 epochs**, and every wall-clock figure in §4 — which is built on
8,425,000 sample-passes — is understated by **64×**. Review 5 §4.4 item 4 wrote this out verbatim.

**Minimal fix:** `per_epoch = max(ceil(len(ds) / max(a.batch * a.accum, 1)), 1)`.

⚠️ **And `train.py:401`'s WTA-reach print uses `a.batch`, not `a.batch * a.accum`.** MEASURED in
the same run: at `--batch 2 --accum 4` it printed *"at most **2** of 64 proposals updated per step
(3.1 %)"* when the true per-optimiser-step reach is **8 of 64 (12.5 %)**. The flag's own help text
says accumulation *"decides whether the 64 proposals exist"*; the diagnostic that would show it
working is computed over the wrong variable. Same for the per-step `winners N/B` log line, which
reports the **last micro-batch**, not the window.

### ⛔ D-D · HIGH · the two rank files pair IDENTICAL camera frames with DIVERGED ego states

This is not an instrument defect; it is a property of the corpus, and it is the one thing in this
package I would want settled before spending A40-days.

`build_targets.py:151` reads `smps = log.simulation_history.data` — a **closed-loop teacher
rollout**. `:157` takes the camera frames from the **log DB**, paired by **timestamp**. So the
image is the real recorded frame at that simulated time, and the ego is wherever the rollout put it.

MEASURED (`raw/2026-09-21-review6/rank_ego_divergence.txt`), rank 0 vs rank 1 on 1,091 shared keys:

* **image list IDENTICAL on 1,091/1,091** — as `diag_rank_distinctness.py` arm 3 *requires*;
* ego speed differs by **> 0.5 m/s on 23.6 %** of keys and **> 2.0 m/s on 4.5 %** (max 4.42 m/s);
* and the gap **grows with step in 6 of 7 logs** — the closed-loop divergence signature:

| log | n | first 25 % | last 25 % |
|---|---|---|---|
| 2021.05.12.23.36.44_veh-35_01133_01535 | 327 | 0.005 | **0.774** m/s |
| 2021.06.07.12.54.00_veh-35_01843_02314 | 109 | 0.000 | **0.761** m/s |
| 2021.06.09.14.58.55_veh-35_01894_02311 | 218 | 0.019 | **1.514** m/s |

⇒ **At least one of the two rows at every late key pairs a real image with an ego that was never at
that pose.** The larger the drift, the less the perception input describes the scene the label was
produced in.

⚠️ **Scope this honestly.** PUBLISHED Table A12: *"Trajectory supervision: Teacher rollouts; no
human trajectories"*, so the paper does this too — **but the paper has SimScale**, 237 K
**re-rendered** out-of-distribution scenes, which is precisely the machinery that closes a
rollout-vs-sensor gap, and REFe has none. REFe's arm A (navtrain-only) is the one the paper runs
*without* SimScale — and NAVSIM navtrain is a one-shot setting, not a ~15 s closed-loop rollout, so
the gap is not obviously inherited. **UNVERIFIED** how large the cost is.
**The cheap instrument is a per-row drift column** — the rollout's displacement from the log ego at
the same timestamp — banked in the tuple, so a drift threshold becomes a filter rather than a
hypothesis. Nothing in the bank records it today.

### ⚠️ D-E · MEDIUM · three divergences `diag_rank_distinctness.py` does not catch, with a live control

MEASURED (`raw/2026-09-21-review6/rank_instrument_attack.txt`). Each arm's expectation is a
**literal** — "exit 1, `RANKS_ARE_A_RELABELLED_COPY`" — never an expression over the instrument.

| attack | instrument says | should say | |
|---|---|---|---|
| **A** rank**2** is a relabelled copy of rank 0 (3 rank files present) | exit 0 `RANKS_ARE_DISTINCT` | exit 1 | ⛔ **MISSED** |
| **B** rank 1 = rank 0 + **1e-9 m** on every coordinate | exit 0 `RANKS_ARE_DISTINCT` | exit 1 | ⛔ **MISSED** |
| **C** rank 1 keeps its own traj/goal, `ego` **copied** from rank 0 | exit 0 `RANKS_ARE_DISTINCT` | exit 1 | ⛔ **MISSED** |
| *(control)* rank 1 is a relabelled copy of rank 0 | **exit 1 `RANKS_ARE_A_RELABELLED_COPY`** | exit 1 | ✅ the probe is sensitive |

* **A** is the instrument only ever comparing `names[0]` vs `names[1]` (`:129-130`). With the route
  augmentation the package already has (`code/augment_routes.py`), >2 ranks is the expected state.
* **B** is `a[k]["traj"] != b[k]["traj"]` — a **bitwise** novelty test with no magnitude. It is not
  hypothetical: on the **real** fixed data, **25.7 % of rank-1 rows are within 0.50 m of rank 0 at
  4 s, 9.0 % within 0.10 m, minimum 0.0029 m**. "1,091/1,091 trajectories differ" is therefore a
  much weaker statement than it reads, and a quarter of the new supervision is near-duplicate.
* **C** is that `compare()` never looks at `ego`, which is a **model input**. On the real data it
  differs on 1,054/1,091, so a copy is a real corruption.
* ⚠️ `_median_sep` uses `goal[:2]` only — the **first** of the two goal points. Reporting-only.

### ⚠️ D-F · MEDIUM · `diag_consumer_conformance.py`'s default target path is the stale bank

MEASURED: with the shipped default (`--targets D:/Projects/TanitAD/data/refe_targets`) it reports
**`CONSUMERS_DIVERGED`, exit 1** — arms 1, 2 and 3 FAIL, because that bank is 8-D and 1-camera.
With `--targets …/refe_targets_4cam` it is **`CONSUMERS_CONFORM`, exit 0**. The instrument is
correct; its default makes it look broken, and a default that fails is a default nobody will trust.

⚠️ **And its `--self-test` exit code is confounded:** `--self-test` alone returns **1**, because the
final `return` is governed by the **live** comparison, not by the self-test. A green self-test and a
red conformance check share one exit code, so a CI wrapper cannot tell "the instrument is inert"
from "the consumers diverged".

⚠️ **The divergence it cannot catch:** every arm is a **shape** assertion. A bank whose `ego`
vector is 7 numbers in a **different order** (or in different units — m/s vs km/h), or whose
`cameras` list is right while the `image` paths are in a different order, passes all eight arms.
Arm 2 checks the `cameras` **field**; nothing checks that `image[i]` is actually camera
`cameras[i]` — and on this bank that is verifiable for free, because the channel name is in the
path. **MEASURED as the free extra check I ran** (`raw/2026-09-21-review6/bank_audit.txt`): all
2,182 rows do carry `CAM_F0/L0/R0/B0` in positions 0–3, so the property holds today; it is simply
unguarded. ⛔ **The same instrument also cannot see B1** — it asserts the *count* of image paths,
never that any of them resolves to a file.

### ⚠️ D-G · MEDIUM · `validate_model.py`'s §1 compares a fraction its own docstring says is comparable, and is not

`validate_model.py:5-6`: *"The fraction is comparable across backbone sizes even though the
absolute counts are not: REFe is ViT-S with one camera, theirs is ViT-L with four."*

MEASURED (`raw/2026-09-21-review6/param_probe.txt`):

| backbone | 4 cameras: total | trainable | **trainable %** |
|---|---|---|---|
| vits16 | 31,949,830 | 10,362,882 | **32.43 %** |
| vitb16 | 99,598,534 | 13,957,314 | **14.01 %** |
| vitl16 | **322,070,854** | **18,991,426** | **5.90 %** |

A **5.5× spread**. The fraction is *not* comparable across backbone sizes — the trainable head is
nearly size-invariant while the frozen trunk is not. `--backbone vits16` (the only way to run the
validator on this box) therefore prints **32.43 %** next to the paper's **5.49 %** with no note.
⚠️ The docstring is also stale twice: REFe is now ViT-L with **four** cameras.

✅ **The counts themselves check out.** ViT-L × 4 = **322,070,854 / 18,991,426 / 5.90 %**; the
1→4-camera delta is **exactly +49,152 = 48 × 1024** (registers), and `reg_compress` is **6,303,744**
— D13's corrected literal, confirmed. *(Review 5's 322,071,366 was the ego_dim-9 build; 9→7 removes
2 × 256 = 512, which is exactly the difference.)*

### ⚠️ D-H · LOW · `validate_model.py`'s peak-memory assertion has still never run

Sections 4 and 5 now reach their arms (D3 fixed). Section 6 does not: `:205` opens the block and the assertion sits at `:212-217`, all behind
`if dev == "cuda"`, and the only device that makes the *shipping* config runnable is an A40. So the
assertion `_pk < 44.0` — added because "three documents quoted a stale 5.36 GB" — remains
**unexercised under every configuration anyone has run**. It is a correct assertion with no
measurement behind it, and the first pod hour should be where it fires, not where it is discovered.

### ⚠️ D-I · LOW · `--device` parsing will `IndexError` on a trailing flag

`validate_model.py:36-37` and `:112`: `argv[argv.index("--device") + 1]` with no bounds check and no
membership check. `--device` as the last argument raises `IndexError`; `--device gpu` is accepted
and fails later inside torch. The *reported* fix (the flag is no longer silently ignored) is real
and verified — this is the remaining rough edge. Same shape at `--backbone` and `--batch`.

⚠️ **Two further HIGH defects — D-J and D-K — are in §4b.** They were found after this section
was drafted, by running every instrument rather than by reading.

---

## 4 · The training-time numbers, independently recomputed

Recomputed from `raw/2026-09-21-camera-scaling/cam_scaling.txt` **transcribed as input**, with no
figure taken from `TRAINING_TIME.md`. Script + output:
`raw/2026-09-21-review6/training_time_recompute.py` / `.txt`.

### 4.1 What reproduces

| claim | recomputed | verdict |
|---|---|---|
| ViT-S t(4)/t(1) | **3.8740** | ✅ (doc: 3.874) |
| ViT-B t(4)/t(1) | **4.0824** | ✅ (doc: 4.087 — rounds from 4.0824, fine) |
| ViT-L t(2)/t(1) | **2.0685** | ✅ (doc: 2.067) |
| ViT-L 4-cam from the measured band | **3.0527 … 3.2170 s/sample** | ✅ (doc: 3.05–3.22) |
| 337,000 × 25 | **8,425,000** sample-passes | ✅ |
| arm B (ViT-B, 337K×25 at 1.139) | **2,665** 4060-h | ✅ |
| arm A (ViT-B, 100K×25 at 1.139) | **791** 4060-h | ✅ |
| arm A− (ViT-S, 100K×25 at 0.492) | **342** 4060-h | ✅ |

**The linearity finding is sound and is the document's best contribution.** Measuring the camera
scaling at a backbone that *fits*, instead of quoting a paging number, is exactly the right cheap
discriminating experiment, and the probe's residency guard (`headroom > 1.0` GiB, else
`TOO CLOSE TO THE CARD — not quotable`) is a real guard: it refused the ViT-L 4-cam row at 27.074 s.

### 4.2 Three defects in the derivation

**(a) The 3.26 upper bound is attributed to a ratio that does not produce it.**
The document says *"Taking the ViT-L per-doubling ratio (2.067) instead gives 3.26."*

| route | value |
|---|---|
| `t(1) × 2.0685²` | **3.3717** |
| `t(2) × 2.0685` | **3.3717** |
| `t(2) × 2.000` ← the **analytic** doubling | **3.2600** |

⇒ **3.26 is the analytic ×2 wearing the measured ratio's name.** The honest upper bound on the
stated method is **3.372 s/sample**, which moves arm D from 7,629 to **7,891 4060-hours**.

**(b) The A40 multiplier band's optimistic end is unsupported by its own three sources.**
The document writes *"2.48–3.0×, ESTIMATED"* and then justifies it with *"Three independent ratios
landing within 3 % of each other."* Recomputed from the same three PUBLISHED figures:

| metric | A40 | RTX 4060 | ratio |
|---|---|---|---|
| FP32 TFLOPS | 37.4 | 15.11 | **2.475×** |
| memory bandwidth GB/s | 696 | 272 | **2.559×** |
| TF32 Tensor TFLOPS | 74.8 | 30.2 | **2.477×** |

**Max = 2.559. Nothing in the cited evidence reaches 3.0.** The 3.0 is INHERITED from
`MODULE_SIZING_STUDY.md`'s unmeasured "~3×" and smuggled in beside three sources that all say ~2.5.
It is also the end that produces the headline's optimistic figure:

| band | arm D A40-hours | **days on one A40** |
|---|---|---|
| document's 2.48–3.0 | 2,379 – 3,076 | **99.1 – 128.2** |
| the band its own evidence supports (2.48–2.559) | 2,789 – 3,076 | **116.2 – 128.2** |

⇒ **"99–128 days" should read "116–128 days"** until the multiplier is measured. The document is
right that measuring it is a three-minute pod job; until then the low end is 17 days too generous.

**(c) "REFe is 2.4–2.6× slower than theirs" depends on reading a SPARSE spec figure as dense.**
§5 converts the paper's 608 GPU-hours through *"H20 TF32 tensor ~148 TFLOPS vs the 4060's 30.2,
≈4.9×"*. NVIDIA's H20 datasheet lists **148 TFLOPS as the TF32 Tensor figure *with sparsity*;
the dense TF32 figure is ~74**, and its non-tensor FP32 is ~44 — and **the paper trains FP32**
(Table A12).

| H20 basis | ratio vs 4060 | 608 GPU-h predicts | our 7,140–7,630 is |
|---|---|---|---|
| 148 (sparse TF32) — the document's | 4.90× | 2,980 4060-h | **2.4–2.6× slower** |
| 74 (dense TF32) | 2.45× | 1,490 4060-h | **4.8–5.1× slower** |
| 44 (FP32, the paper's precision) | 1.46× | 886 4060-h | **8.1–8.6× slower** |

⇒ **The gap the document states plainly and refuses to explain away is, on a dense basis, roughly
twice what it says — and on the paper's own precision, more than three times.** This does not make
the model wrong; it makes the "REFe could plausibly get ~2× cheaper with engineering" sentence an
underestimate of the work. ⚠️ **I could not independently re-verify the H20 datasheet offline**;
the dense/sparse distinction is the standard NVIDIA footnote convention and the document should
state which column it used either way. Treat this row as **ANALYTIC on a PUBLISHED figure whose
column is unconfirmed**, and settle it with one datasheet lookup before quoting either number.

### 4.3 Two things the arithmetic is silent about, and both favour the pod

* **The basis is batch 1.** Every s/sample above is `batch=1` on a 4060. An A40 at batch 4–8 has a
  materially better per-sample cost, and the document's own §6 says so — but the headline range
  does not reflect it. ⇒ the A40 figures are an **upper bound**, and should be labelled one.
* **The 337 K × 25 arithmetic assumes `--epochs 25` means 25 epochs.** Under D-C it does not, if
  `--accum` is used — and §6 recommends `--accum 64`. **These two documents contradict each other**
  and the contradiction is 64×.

### 4.4 The recommendation survives

⭐ **Arm A (ViT-B, navtrain only) is the right first arm and the reasoning is sound**, including the
strongest argument in the document: Table A13 trains ViT-S/B/L on navtrain **only, without SimScale,
all other settings fixed**, so arm A reproduces a published row rather than improvising a smaller
one. That matches the programme's "reproduce on THEIR data first" rule. Its step time is the one
that is MEASURED at four cameras. ✅ Verified against the paper text (p. 31, §B.2).
Under the corrected multiplier, arm A is **309–319 A40-hours = 12.9–13.3 days**, not 11–13.

---

## 4b · Two more defects, found only by RUNNING every instrument (D-J, D-K)

*These sit after §4 because I found them after drafting §3, and I would rather the reader see the
order the evidence arrived in than a tidied history. They rank with D-C and D-D, not below them.*

### ⛔ D-J · HIGH · `diag_cameras.py` still publishes paging as compute, still banks nothing — and exits 0 with a success marker

This is Review 5's D10, unfixed, and it is now **actively dangerous** because a second instrument
in the same directory disagrees with it.

MEASURED, my run, exit **0**, final line `CAMERA_COST_MEASURED`:

```
  cams batch  tokens  regs    trainable        total  peak GB     sec
     1     1    1920    16   18,942,274  322,021,702     3.55    1.19
     4     1    7680    64   18,991,426  322,070,854    10.09   23.56
     4     2    7680    64   18,991,426  322,070,854    18.89   90.43
  4 cameras vs 1, at batch 1:
    tokens     x4.00
    peak mem   x2.85  (3.55 -> 10.09 GB)
    time       x19.78
```

* **10.09 GB and 18.89 GB on an 8.00 GiB card.** `cam_scaling.py` — the *new* probe, in the same
  package — explicitly **refuses** exactly this row (`TOO CLOSE TO THE CARD -- not quotable`,
  headroom guard in `main()`). `diag_cameras.py` prints it and returns success.
* **`time x19.78`** is the bus, not the model. `TRAINING_TIME.md` §2 exists to retract this number
  and does so persuasively — while the instrument that produced its ancestor still ships and still
  exits 0.
* ⚠️ **`peak mem x2.85` against `tokens x4.00` is the tell, and the script reports it as a
  finding.** Memory scaling *below* token scaling on a card that cannot hold the allocation is the
  signature of the measurement being capped, not of an efficiency.
* **It banks nothing.** MEASURED: `grep -cE "open\(|\.write\(|json\.dump|raw/" diag_cameras.py` → **0**,
  against a same-breath control `grep -c print` → **13**. So the absence is about the file, not my
  search. Review 5 required it to write to `raw/` before its output is cited again.

⇒ **Two instruments in one directory now give 4-camera timings that differ by 24×, and the wrong
one is the one whose name says "cameras" and whose exit code says OK.** Minimal fix: give
`diag_cameras.py` `cam_scaling.py`'s residency guard, or delete it and point the README at
`cam_scaling.py`.

### ⛔ D-K · HIGH · scorer coverage does NOT double — the rank-1 scorer bank is in a directory the trainer never reads

MEASURED (`raw/2026-09-21-review6/scorer_cov.txt`), joining each scorer bank onto the 2,182 target
tuples on `(log_name, token, step, rank)`:

| bank | rows | frames | rank histogram | join onto the target bank |
|---|---|---|---|---|
| `data/refe_scorer_targets_full/scorer_targets.jsonl` | 2,332 | 212 | `{0: 2332}` | **212 = 9.7 %** |
| `data/refe_scorer_targets_r1/scorer_targets_rank1.jsonl` *(rebuilt 01:31)* | 2,332 | 212 | `{1: 2332}` | **212 = 9.7 %** |
| both together (rank-blind check) | — | 424 | — | **424 = 19.4 %** |

`ScorerBank.__init__` (`train.py:168`) globs `scorer_targets*.jsonl` inside **one** directory, and
`--scorer-targets` defaults to `refe_scorer_targets_full` (`:588`). The rank-1 file is in a
**different** directory.

⇒ **A run started right now sees 9.7 %, not 19.4 %.** ⚠️ And note what the rank fix actually did to
the number the operator sees: my first CPU run, before the rebuild, read *"scorer bank: 4,664 rows
over 424 frames"* — the duplicate sitting in the same directory, inflating coverage to a fake
19.4 %. After the rebuild the same command reads **2,332 rows over 212 frames**. The honest
coverage **halved**, which is the correct direction and should be stated that way rather than as
"coverage will double". To actually reach 19.4 %, move `scorer_targets_rank1.jsonl` into
`refe_scorer_targets_full/` (the glob will pick it up) or point `--scorer-targets` at a merged
directory — and re-measure.

✅ **The 9.7 % join prediction itself is CONFIRMED**, independently, from the artifacts: 212 of
2,182. The reported live 10.4 % (10 of 96) is consistent with it.

---

## 5 · What I could not verify, and why

| # | claim | why not closed |
|---|---|---|
| 1 | **The H20 TF32 figure's dense/sparse column** (§4.2c) | No offline datasheet on this box and I would not quote a spec from memory. The arithmetic is exact for any basis; the *choice* of basis is the open question, and it swings the headline gap from 2.4–2.6× to 4.8–5.1×. **One datasheet lookup settles it** and it must be done before either number is quoted again. |
| 2 | ~~That `length`/`width` are empirically constant across nuPlan's logs~~ | ✅ **CLOSED during this review** — 120 log DBs, 23 vehicle ids, 4 cities, two independent routes, with a pose/time control at 119/120. See D-B. Listed here because it was open when §5 was drafted, and because the correction runs *towards* the claimant, not away from them. |
| 3 | **The PDMS cost of the distortion correction** | Unchanged from Review 5: needs a trained `undistort=False` ablation arm. ⛔ And per D-A that arm is currently inert under instrument-style mutation — it is only valid if built as a fresh model, which `train.py --no-undistort` does correctly. |
| 4 | **Anything at the shipped config (ViT-L × 4 cameras) on a GPU** | Capacity, not failure: 9.49 GiB against 8.00 GiB. Every runtime number above is ViT-S or ViT-B, or CPU. The ViT-L parameter accounting needs no device and **is** measured. |
| 5 | **The planner and the trainer against real 4-camera frames** | Blocked by B1 — three of four channels have no images on this box. `planner.py`'s camera path is verified by reading and by its own shape guard, not by running. |
| 6 | **The A40/4060 multiplier** | Still ESTIMATED, as the document says. My correction is to the *band's* upper end, not a measurement of the multiplier. The document's own advice — re-run `cam_scaling.py` on the pod in the first three minutes — is right and should be the first pod job. |
| 7 | **Whether the rollout/sensor mismatch (D-D) costs anything** | Needs the drift column that does not exist yet. I quantified the *divergence*, not its *effect*. |

⚠️ **And one thing I deliberately did not do:** I did not re-derive the geometry, token order,
quaternion convention or intrinsic rescaling — Review 5 verified those at 8.8e-08 with a live
negative control, nothing tonight touched them, and re-running someone else's correct probe
measures determinism, not correctness.

---

## 6 · Manifest

**Deliverable (staged by explicit path, NOT committed, NOT pushed):**

| artifact | location |
|---|---|
| this review | `D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/REVIEW_6_FINAL.md` |

**Raw evidence, banked in the repo** at
`D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-21-review6/`:

| file | what it establishes |
|---|---|
| `frustum_cache_attack.py` / `.txt` | the Caltech inversion verified at **1.510e-07** against an independently authored forward model; `undistort=False` == ideal pinhole at 1.536e-07; 20 vs 200 iterations identical; **and the three cache-key blind spots at exactly 0.000000e+00** (D-A) |
| `bank_audit.py` / `.txt` | the 4-camera bank's schema, per-row channel paths, per-log stats, and the `kept/steps = 109/149` origin of "73.2 %" |
| `bank_audit2.py` / `.txt` / `_REMEASURED.txt` | **before**: 1,091/1,091 rows identical apart from `rank`; **after** the rebuild: 0/1,091. Both runs also show `CAM_F0` 200/200 resolvable and `CAM_L0/R0/B0` **0/200** (B1) |
| `rank_ego_divergence.py` / `.txt` | identical frames on 1,091/1,091 keys against a growing ego-speed gap; the per-log first-quarter→last-quarter drift table (D-D) |
| `rank_instrument_attack.py` / `.txt` | three constructed rank divergences the new instrument misses, with a live control that it catches (D-E) |
| `accum_equivalence.py` / `.txt` | trajectory term **exactly** equivalent (0.000e+00); score term ratio **0.7500** against a literal expectation of 1.0000 (B3) |
| `accum_a.txt` / `accum_b.txt` | `--epochs 2` prints the **same** step count at `--accum 1` and `--accum 4` while consuming 2 vs 8 epochs of data (B2 / D-C) |
| `training_time_recompute.py` / `.txt` | §4 in full: the ratios that reproduce, the 3.26 mis-attribution, the 2.475/2.559/2.477 multiplier ceiling, the three H20 bases |
| `param_probe.py` / `.txt` | parameter accounting at ViT-S/B/L × 1 and 4 cameras; the 5.5× trainable-fraction spread (D-G); `reg_compress = 6,303,744` |
| `scorer_cov.py` / `.txt` | the 9.7 % join, per bank, and that the two rank banks are in different directories (D-K) |
| `validate_vits_cpu.txt` | `VALIDATE_OK`, exit 0 |
| `diag_architecture.txt`, `diag_rope.txt`, `diag_schedule.txt`, `diag_guard_audit.txt`, `diag_planner_holds.txt`, `diag_cameras.txt` | the instrument runs, with exit codes |
| `ccf_selftest.txt` / `ccf_4cam.txt` | `SELF_TEST_OK` + `CONSUMERS_DIVERGED` on the default path; `CONSUMERS_CONFORM` on `refe_targets_4cam` (D-F) |
| `rank_distinct.txt` | `RANKS_ARE_DISTINCT` + `SELF_TEST_OK`, exit 0 |
| `train_overfit_gpu.txt` | `TRAIN_CONTROL_OK`, L1 1.7351 → 0.4044 (76.7 %) |
| `diag_*_witharg.txt` | the positional-argument diagnostics run against a real rollout log |
| `diag_aimed/batch/cost/curbdist/determinism/enrich_cost/goal/keys.txt` | the bare-invocation exits (see the table below) |

### Every `refe/diag_*.py` and `validate_model.py`, exit codes I actually observed

| script | invocation | exit | final line |
|---|---|---|---|
| `validate_model.py` | `--backbone vits16 --device cpu --batch 1` | **0** | `VALIDATE_OK` |
| `diag_architecture.py` | bare | **0** | `ARCHITECTURE_CONFORMS` |
| `diag_rope.py` | bare | **0** | `ROPE_MATCHES_REFERENCE` |
| `diag_schedule.py` | bare | **0** | `SCHEDULE_OK` |
| `diag_guard_audit.py` | bare | **0** | `GUARDS_CAN_FAIL` (6/6) |
| `diag_planner_holds.py` | bare | **0** | `PLANNER_HOLDS_GUARDED` (4/4) |
| `diag_rank_distinctness.py` | `--self-test` | **0** | `SELF_TEST_OK` + `RANKS_ARE_DISTINCT` |
| `diag_consumer_conformance.py` | `--self-test` (default targets) | **1** | `SELF_TEST_OK` + `CONSUMERS_DIVERGED` ⚠️ D-F |
| `diag_consumer_conformance.py` | `--targets …/refe_targets_4cam` | **0** | `CONSUMERS_CONFORM` |
| `diag_cameras.py` | bare | **0** | `CAMERA_COST_MEASURED` ⛔ D-J |
| `diag_scorer_components.py` | `<rollout log>` | **0** | `SCORER_COMPONENTS_CONFORM` |
| `diag_keys.py` | `<rollout log>` | **0** | (report, no marker) |
| `diag_goal.py` | `<rollout log>` | **0** | (report, no marker) |
| `diag_aimed / batch / cost / curbdist / determinism / enrich_cost / goal / keys` | **bare** | **1** | `IndexError: list index out of range` |

⚠️ The eight `IndexError`s are **invocation, not regression** — `README_DIAGNOSTICS.md:7` documents
`python diag_<name>.py <path-to-a-simulation_log.msgpack.xz>`, and the three I re-ran with a real
log all exit 0. They are listed so the count is honest, and because a raw traceback where a usage
message belongs is the reason someone will read eight failures as eight broken tools.

**Files read but NOT modified:** all of `refe/*.py`, `TRAINING_TIME.md`, `REVIEW_5_CAMERAS.md`,
`README_DIAGNOSTICS.md`, `data/nuplan_cam_calib.json`, the target and scorer banks under
`D:/Projects/TanitAD/data/`, `C:/dzo/m-nr-n` rollout logs (read-only), `C:/Users/Admin/dz/DriveZero`
(`drivezero_report.pdf`, `DriveRL/`, `release/checkpoints/checkpoint_2400.pt`).

**Primary sources quoted:** `drivezero_report.pdf` pp. 7, 8, 24, 25, 30, 31 (re-extracted this
session and cross-checked against the Library's banked copy — identical on pages 2–32);
DriveRL's released `vanilla_net_agent.py` and `checkpoint_2400.pt`.

---

## 7 · What I would do first, in order

1. ⛔ **Fix D-C's one line** (`per_epoch` over `batch * accum`) before anyone types the recommended
   pod command. It is the cheapest blocker and the most expensive to discover late.
2. ⛔ **Put `undistort`, `cam_distortion` and the extrinsics in the `_frustum` cache key**, and add
   the seventh guard-audit arm. Eight characters and a test; today the package's headline geometric
   change has no arm that can go red.
3. ⛔ **Decide B1 at the PI level, not in code.** Four channels of navtrain is the ~4× pull already
   escalated in Review 5 §5; nothing downstream is real until it lands, and the bank currently
   *looks* four-camera while being one.
4. **Fix B3** by accumulating the score loss's numerator and denominator separately.
5. **Merge the two scorer bank directories** (D-K) and re-print the coverage.
6. **Correct `TRAINING_TIME.md`'s three derivation defects** (§4.2) and relabel the A40 columns as
   an upper bound at batch 1.
7. **Bank a drift column** in `build_targets.py` (D-D) — it is a few lines at build time and it is
   the only thing that turns a suspicion about the corpus into a filter.

---

## 8 · Addendum — findings that landed after §7 was written

### ⚠️ D-L · LOW · `data/nuplan_cam_calib.json` names its units for **4 of its 8** channels

⭐ **Credit first, because this is the programme's own anchors.pt lesson applied correctly.** The
four shipped cameras each carry `distortion_model: "caltech:k1,k2,p1,p2,k3"`, `K_units:
"pixels at 1920x1080"`, `image_size_wh: [1920, 1080]`, `extrinsics_convention:
"R(q:w,x,y,z) @ p_cam + t -> p_ego (metres)"`, and a `_provenance` block naming the source and the
date. That is exactly what `CLAUDE.md`'s units rule asks for, and it is the thing that would have
prevented the 396-g anchor misreading.

⛔ **`CAM_L1`, `CAM_L2`, `CAM_R1` and `CAM_R2` carry only `K`, `t`, `q`** — no distortion, no
`distortion_model`, no `K_units`, no `extrinsics_convention`. So the file that fixes
"an artifact must name its units" fixes it for half its entries, and the unannotated half is
precisely the alternative rig the package's own coverage sweep found better
(`F0/L1/R1/B0`, 78.10 % vs 70.76 %). Whoever runs that ablation will open four entries whose units
and convention are not stated.

⚠️ Related, and stated because it is a real limit rather than a defect: `REFeConfig.cam_distortion`
is **one tuple applied to all cameras**, not a per-channel read of this JSON. Today that is correct
(all four share the coefficients) and it is also why the missing per-channel entries do not bite
yet.

### ✅ D17's scoping is confirmed, and its "practically small" claim is now measured

MEASURED (`raw/2026-09-21-review6/rigvar_camera_extrinsics.txt`), 64 logs, 0 errors, 13 vehicle ids:
**13 distinct `CAM_F0` translation vectors, exactly one per vehicle id.** Their spread is
**x 1.624–1.686 m (6.2 cm), y −0.026 to −0.005 m (2.1 cm), z 1.491–1.535 m (4.4 cm)**.

⇒ Both halves of `model.py:146-156` hold: the encoding **is** rig-locked (13 rigs here, ~22 over
300 logs in Review 5), and the practical magnitude **is** centimetres. The comment's scoping to
"resolution-free and per-camera, never a function of the rig" is the honest wording and it is
supported.

### ⚠️ D-M · LOW · the one arm defending REFe's declared departure has no deliberate-regression control

`model.py` carries **two independent detaches**, and only one of them has a mutation arm:

| line | switch | what it detaches | whose it is | does the validator FLIP it? |
|---|---|---|---|---|
| `:671` | `forward(detach_scorer=...)` | `traj` | **the paper's** (p. 8) | ✅ yes — `detach_scorer=False`, and it goes RED |
| `:676` | `cfg.detach_scorer_context` | `visual_ctx` | **REFe's extra** (D11) | ⛔ **no** — asserted, never mutated |

MEASURED (`raw/2026-09-21-review6/detach_ctx_arm.txt`) — the flip the file does not make:

```
  detach_scorer_context=True  (shipped): backbone LoRA grad   0.000000   scene_proj grad      0.000000
  detach_scorer_context=False (mutated): backbone LoRA grad 401.570572   scene_proj grad 172142.449707
```

✅ **The arm is LIVE** — `backbone_clean` reads `True` clean and `False` mutated, so it would catch
an undeclared removal, which is what D11 asked for. ⛔ **But nothing in the package demonstrates
that**, and by this programme's own rule ("guards need mutation, not inspection") an assertion
without a mutation is an assertion whose sensitivity is assumed. It is **four lines** to add, it
needs no cache-clearing (unlike D-A — `:676` is read at forward time), and it is the arm guarding
the single declared departure REFe makes from the paper.

⚠️ **The magnitudes are also the argument for running the ablation rather than assuming it is
cosmetic:** with the extra detach removed, `scene_proj` receives a gradient of **172,142** and the
backbone LoRA **401.6** from the score loss alone. That is not a rounding difference in how much
the encoder is adapted — and how much the encoder is adapted is the variable REFe exists to study.

### ⏳ Still open at the close of this review

The H20 dense-vs-sparse TF32 column (§4.2c) was delegated to a second reader and had not returned
when I closed. **I am not going to predict its answer.** The arithmetic in §4.2c is exact for each
of the three bases; what is unresolved is which column NVIDIA's 148 TFLOPS figure belongs to, and
that single lookup moves the headline "REFe is 2.4–2.6× slower" to as much as 4.8–5.1×. Until it is
checked, `TRAINING_TIME.md` §5 should name the column it used.
