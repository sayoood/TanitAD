# SPEC — refcv6 on the NavSim suite (warmup_two_stage · navhard_two_stage · navtest v1) — PRE-REGISTERED

**Written 2026-09-24 ~00:35 Berlin, BEFORE any refcv6 NavSim score exists.** What exists at this
moment: the harness-reproduction scores of two MODEL-FREE controls (§0, deliverable 1), a 3-scene
bridge smoke and 4-scene control tests that produced **trajectories, not scores** (no scorer was
run on any refcv6 plan). The git blob of this file is recorded in `raw/SPEC_PREREG_HASH.txt` at
the moment it is staged. Brief: `../BRIEF.md` (PI 2026-09-23, *"inclduing the navsimn suite"*) and
the Master Mind's stream brief (this package). Predecessors whose findings bind this SPEC: E2
(`../../2026-09-19-navsim-refcv4b-bridge/`), W3 (`../../2026-09-19-navsim-v1-navtest/`), W7
(`../../2026-09-20-navhard-refcv4b/`), the navhard stage-1 extraction
(`../../2026-09-20-navhard-stage1-extract/`).

## 0. Facts this SPEC is built on (MEASURED 2026-09-23/24 unless marked)

| fact | evidence |
|---|---|
| ⭐ **The harness still reads its known values.** E2's CV_official and STOP_zero re-scored through this package's driver (`code/score_arm6.py`: E1's current wrapper sha `97cf493a…`, E2's seam agent imported, `PYTHONHASHSEED=1`) reproduce E2's banked CSVs **cell for cell, max \|Δ\| = 0.0 over 220 tokens × every column**, and S2-EPDMS-u / the 3 official rows **exactly** (CV 0.39713167136274696 / 0.4602892129691729 / 0.3341286472565139 / 0.1853562745165113; STOP 0.5212469877807624 / 0.5777242796695368 / 0.5212469877807625 / 0.3009023137456225) | `raw/HARNESS_REPRO.json`, `raw/harness_repro/` |
| The C: devkit carries E1's IDM degenerate-path patch (`navsim_idm_agent_manager.py:84-89`) on top of `0a380a9` + the loader patch; inert on warmup (the reproduction above is bit-exact against E2's pre-patch run) | source read; `raw/HARNESS_REPRO.json` |
| **refcv6's frame** = `trunk_shapes.FRAME_416x1024` = `frame_for_width(1024, 416)`: 416 × 1024, `f_ref` 488.92398517830253, **cylindrical** (column linear in azimuth), hfov **120.0000°**, vfov **46.0921°** (the builder manifest's own 46.09213171161337) | `code/frames416.py`; `tests/test_frames_lift6.py` (KG-416, literal) |
| NavSim → that frame: E2's verbatim 3-camera stitch (CAM_L0/F0/R0 → one virtual cylindrical camera at F0's boresight, rolled level), re-pointed at the 416 frame; the re-pointing reproduces the DataFlyWheel 256×640 bank **bit-exactly** (KB-256: 6/6 and 3/3) | `code/frames416.py`, `tests/test_frames_lift6.py` |
| Banks built at 416 × 1024: warmup stage 2 **204/204** (3 history frames), navhard stage 2 **5,462/5,462** and stage 1 **450/450** (t0 frame; stage-1 jpgs = the verified extraction root, `VERIFY.txt` COMPLETE 3,375/3,375), **0 failures** | `C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/*/BUILD_REPORT.json` |
| navtest v1 data IS on this box: 136 logs, the 32 camera archives (127,882,665,618 B, sha256 == HF ETag), W3's metric cache (12,146) and W3's 256×640 bank — which refcv6 cannot use; the 416×1024 navtest bank is being built from the archives (W3's builder, imported) | `code/build_navtest416.py`; W3 `RESULT.md` §1.4 |
| Posted limit of the ego lane at t0 (nuPlan map, `speed_limit_mps`): warmup **136/220** tokens (PIT 136/136 all 25 mph; BOS 0/64 and SG 0/20 = lane found, no limit) | `raw/inputs/speed_limits_warmup_two_stage.json` |
| NavSim's ego origin is **~0.36 m above the road** (median vehicle-cuboid bottom −0.3557 m over 74/76 logs, range −0.50…−0.15); `lidar2ego` is the identity on every frame read | `raw/inputs/road_plane_navhard_warmup_logs.json` |
| refcv6's sampler is **STOCHASTIC at eval** (one fresh eps at `sampler_infer_t = 8`, `refc.py` "stochastic at eval, by design"); on the step-1,000 kit checkpoint two seeds moved one scene's plan by **24.2 m** | 1-scene probe (scratch); `tests/test_model_seam6.py` K0 (same seed ⇒ bit-identical) |
| ⛔ **refcv6's trunk does NOT equalize the bottom rows, although argv says `--equalize-bottom-rows 43`.** `_pin_trainer_cfg` sets `cfg.core.encoder.trunk_equalize_bottom_rows` as an AD-HOC attribute (`refc_v3_train.py:381`) and 78 lines later rebuilds the encoder config with `dataclasses.replace` for `--image-hw` (`:459`), which carries only declared fields — `CNNEncoderConfig` declares none by that name. The built `TimmResNetTrunk` reads `equalize_bottom_rows = 0`; `normalise()` zeroes nothing. The LIFT bank does get 43 (the trainer passes it from `args`). Files byte-identical at the run commit `287d72e` and the tip `fe5872f` (blobs `7cbe426a…`, `01c45b2d…`); the live run's `config.json` carries `"equalize_bottom_rows": 43` (the perception stamp, from args) | loaded model (scratch probe); `git show 287d72e:…` |
| GPU gate at SPEC time: **closed** — 7 other python processes in `--query-compute-apps` (another session's `augment_search.py` shards) and free host RAM 6.9 GB < 8 GB; CPU fp32 forward **8.1 s/scene** (native dedup, 10 backbone passes), ~3 s with the exact dedup | `run_bridge6.gpu_gate`; probes |

## 1. Question

For a refcv6-r101-s0 checkpoint, run zero-shot on NavSim with every input it was trained on built
from NavSim's own data, **does it drive better than the floors — constant velocity (CV), the
all-zero STOP plan, and the kinematic echo of its own ego inputs (ECHO) — on the official scorer**,
and what do its inputs (frames, nav, max speed) contribute, read against its own inference-seed
floor?

## 2. The inputs (every one an explicit, tested function in `code/refcv6_bridge.py`)

| input | construction | test |
|---|---|---|
| **frames** | the 416 × 1024 stitch (§0); 10 raw 10 Hz slots ← NavSim's 2 Hz history: **ST (primary)** every slot ← the t0 frame; **NT (warmup sensitivity)** nearest in time (E2's `slot_sources`, imported); 8 rows × 3-frame D-015 stacks via the trainer's own `stack_frames` | KG-416, KB-256, KT (tamper) |
| **nav** | NavSim `driving_command` one-hot (0 left, 1 straight, 2 right, 3 unknown — order verified by E2 K9) → v7 token row: left→`left`(1), straight→`follow`(0), right→`right`(2), unknown→`follow`(0, the trainer's unlabeled default); E2's `nav_index_from_command` imported (refuses a non-one-hot). ⚠️ semantics differ: v7's `NAV_TURN_L/R` is a junction turn from the ego future; NavSim's is route-geometry 20 m ahead | literal map test |
| **v0** | `hypot(vx, vy)` at t0 (the core's `v0`; PI ruling 2026-09-02) | KI (moves the plan) |
| **ego history** | 8 × (x, y, yaw, v) at t0−0.7 … t0 (10 Hz), LINEAR in time between the declared 2 Hz states at t0−1.0 / −0.5 / 0 (yaw unwrapped); the encoder's own `ego_channels_from_poses` (backward differences, dt 0.1) then reads each 0.5 s segment's mean accel / yaw rate | constant-accel literal 2.0 m/s², constant yaw rate, ±π unwrap, wrong-grid arm RED |
| **max speed** | the posted limit (`speed_limit_mps`) of the lane the ego occupies at t0 (containing lane with heading within 60°, else nearest ≤ 2 m; `code/export_speed_limits.py`), fed RAW with `v_max_valid = 1`; the model applies its containing-window ladder {30, 50, 100, 120} km/h ONCE. **Fallback where no limit exists: `v_max_valid = 0` = the model's all-zero one-hot ("no ceiling known").** ⚠️ The run's train sidecar is 4,572/4,572 valid, so the all-zero row is OUT OF DISTRIBUTION for this model — stated, not hidden; `R6_VMAXOFF` measures the channel. ⚠️ Admissibility: the brief records the PI ruling 2026-09-19 as making nuPlan map speed limits admissible for this input; the ruling's recorded text concerns LABELS — the reading is the brief's and is surfaced in RESULT.md | literal bins (25 mph → 50, 35 → 100, 15 → 30), all-zero unknown, withheld arm |
| **BEV lift geometry** | the stitch's virtual camera in refcv6's rig frame (rotation = the stitch's `M`; centre = CAM_F0's mount; height = F0.z − road plane); `build_lift_geometry` with the trainer's parameters (stride 16, heights 0/0.5/1.5/2.5 m, `GRID_DEFAULT`, bottom-43-rows unobserved) | analytic road-point row, road-offset arm RED |
| **output** | `out["traj"]` 8 knots (0.5 … 6 s) → E2's `knots_to_navsim` (imported): C² not-a-knot spline in time through the origin, evaluated at 0.5 … 4 s; heading = the tangent, held below 1 m/s | CV line exact, circle-arc heading, wrong-times arm RED |

**Declared-input seam** (`navsim.ego_enforcement`): `declare6` copies only the arm's fields
(`ego_pose[1..3]`, `ego_velocity[1..3]`, and `driving_command[3]` for nav arms); undeclared fields
randomised ⇒ byte-identical inputs, declared ones mutated ⇒ changed (`tests/test_inputs6.py`).
**Stage-1 tokens** get a real refcv6 row where the scene's ORIGINAL frames exist (navhard: all 450)
or the arm reads no pixels (`R6_BLIND`); otherwise the devkit CV agent is the DECLARED stand-in and
the arm's official two-stage row is HYBRID (warmup camera arms: 0/192 stage-1 jpgs, E2).

**Model**: rebuilt by `taniteval/tools/refcv3_arm.load_model` (trainer parser + `_pin_trainer_cfg`,
config cross-check incl. `param_breakdown`, perception branch rebuilt, **strict** load), with two
recorded argv edits (`--trunk-compile` dropped — no Triton on Windows; two rebuild-time Thor paths
→ md5-verified local copies). **Evaluated AS BUILT**: trunk equalization 0 (what the run computes,
§0), lift mask 43. **Precision**: CUDA → as trained (bf16 + NHWC trunk); CPU → fp32 (bf16 is
emulated here, 98 s vs 9 s/scene) — one device and one precision per checkpoint, a mixture
refused. **Exact-duplicate dedup** on (MEASURED bit-identical to the trunk's native path on CPU:
KD, max |Δ| 0.0 over 4 scenes); on CUDA the runner re-measures KD before use. **Inference seed**:
per scene `sha256("<base>:<token>")` → common random numbers across arms.

## 3. Arms

| arm | frames | nav | max speed | seed | role |
|---|---|---|---|---|---|
| **R6_A1** | ST | cmd | map | 0 | **primary (bar arm)** |
| R6_A1_s1 | ST | cmd | map | 1 | inference-seed replicate — the floor every lever is read against |
| R6_A1NT | NT | cmd | map | 0 | time-construction sensitivity (warmup only; 3-frame bank) |
| R6_BLIND | constant grey (bank mean) | cmd | map | 0 | frames-blind — the registered deliberate regression; also runs stage 1 |
| R6_NAVOFF | ST | withheld (`nav_cmd=None`) | map | 0 | nav contribution |
| R6_VMAXOFF | ST | cmd | withheld (`valid=0` everywhere) | 0 | max-speed contribution |
| CV_official | — | — | — | — | devkit `ConstantVelocityAgent` |
| STOP_zero | — | — | — | — | all-zero plan (E2's seam) |
| ECHO_ha0_ext | — | — | — | — | constant measured a0 and κ0 (E2's seam; model-free, identical tokens) |

## 4. Statistics, estimators, stamps

* **warmup_two_stage** — S2-EPDMS-u (E2 SPEC §2: devkit stage-2 aggregation of the official `score`,
  uniform within-group weights), 204 stage-2 tokens; paired per-scene W/T/L; sub-metric means.
  ⛔ **No interval**: 7 logs < the 8-cluster floor (`navsim_ci.MIN_CLUSTERS`). Official two-stage
  EPDMS only for arms that run stage 1 (R6_BLIND, CV, STOP, ECHO).
* **navhard_two_stage** — **the official two-stage EPDMS (`extended_pdm_score_combined`)**, DEFINED
  for camera arms here because all 450 stage-1 scenes have original frames; plus S2-EPDMS-u.
  Interval: `navsim_log_cluster_bootstrap` / paired variant (`taniteval/adapters/navsim_ci.py`,
  76 log clusters, B 2000, seed 0, aggregation `two_stage_mapping_key_mean`). The floors are W7's
  banked model-free runs (`…/20260921T122714Z-…-859e25`: STOP 0.2985, CV 0.1148, ECHO 0.1429),
  and CV is RE-SCORED by this package as the navhard harness control (must reproduce
  0.11481608441648).
* **navtest (v1.1)** — PDMS = mean of `score` over 12,146 tokens ×100 (W3's harness, `PDMS_v1_navtest`);
  interval: W2's registered log-cluster bootstrap (136 logs). Floors: W3's banked STOP 61.8202,
  CV 20.6517, HUMAN 94.5514.
* **Every interval names its question**: it answers *"another draw of LOGS?"* only; training
  variance is untested (one checkpoint, one run); inference variance is read from R6_A1_s1.
* Stamps: tier **T1-family**; stage-1 loop **OPEN**; stage 2 **UNRULED**; v2 background
  **IDM-reactive**, v1 **non-reactive (logged)**; ⛔ never closed loop; **zero-shot** (PhysicalAI-AV
  B1 → nuPlan cameras), 3-camera stitch, non-parity.

## 5. The bars (committed, both outcomes) — for MILESTONE checkpoints (step ≥ 5,000)

* **BAR-R6-W1 (warmup):** S2-EPDMS-u(R6_A1) > max(CV, STOP, ECHO) on the identical 204 tokens —
  point estimates, no interval. **AND** the margin over that max must exceed the seed floor
  |S2(R6_A1) − S2(R6_A1_s1)|; a margin inside it reads **"not separated from inference noise"**.
* **BAR-R6-H1 (navhard):** official two-stage EPDMS(R6_A1) > max(STOP, CV, ECHO), with the paired
  log-cluster interval vs STOP excluding 0 — and the same seed-floor condition.
* **BAR-R6-T1 (navtest):** PDMS(R6_A1) > max(STOP, CV) paired on 12,146 tokens, interval vs STOP
  excluding 0. Stretch (reported, not a bar): the published Ego-Status-MLP (65.6 / 66.4, arXiv
  2406.15349) and DiffusionDrive **88.1** (the community headline, as quoted by the brief; to be
  cited from its primary before it is written as PUBLISHED).
* **PASS** → "refcv6 @ step N beats doing nothing on <split> (zero-shot, stated stamps)". **FAIL** →
  reported as FAILED, decomposed (multipliers, per command, per speed band — W3/W7's ladder), and
  the next lever named (Rule Zero).
* ⚠️ W3 showed **BAR-W3-M1 is gameable by a command-gated stop** (62.48 > STOP). Every bar row
  therefore carries the **stop fraction** (plans moving < 1 m in 4 s) beside it; an admissibility
  clause against stop-gaming is a PI decision (named blocker), not assumed here.
* **Lever reads (secondary, not bars):** A1 − BLIND (vision), A1 − NAVOFF (nav), A1 − VMAXOFF (max
  speed), A1 − A1NT (time construction) — each a paired per-scene difference, **read against the
  seed floor**; an effect inside it is not an effect.

## 6. The step-1,000 kit checkpoint — PIPELINE VALIDATION ONLY, NEVER A RESULT

`D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt` (md5 `7c3ad3c1fbf3d30be5c7733e7b436c65`, verified; 1,000
of 50,400 steps). It runs every arm on **warmup**, **R6_A1 on navhard** (all 5,912 tokens: the first
defined camera-arm two-stage EPDMS path on this box) and **R6_A1 on W3's 200-token navtest subset**
(`A1_sub200_tokens.json`, floors restricted to the same tokens), to validate: the count guards, the
seam, the stage-1 real-frame path, the input reach, the seed floor's size. ⛔ Its scores are
labelled **PIPELINE-VALIDATION** in every artifact and are never quoted as refcv6's NavSim
performance; no bar is evaluated on it.

## 7. Controls (each must read a known value, or the affected numbers are void)

| id | control | pass |
|---|---|---|
| KH | harness reproduction (CV, STOP on warmup vs E2) | max \|Δ\| ≤ 1e-9 per cell — **PASSED, 0.0** |
| KH-nav | CV re-scored on navhard | official two-stage EPDMS == 0.11481608441648 (W7) to 1e-12 |
| K1 | frame integrity | every bank array's sha256[:16] == provenance before use (refused otherwise; KT proves the refusal) |
| KG/KB | frame geometry | KG-416 azimuth-linear < 1e-6 rad; KB-256 bit-exact |
| K5/K6 | declared-input seam | undeclared randomised ⇒ identical; declared mutated ⇒ changed |
| K0 | determinism | same scene + seed ⇒ bit-identical plan |
| KD | exact dedup | == native path (CPU: max \|Δ\| 0.0 measured) |
| KI | input reach | frames, nav, v0, history each move ≥ 1 of 4 plans; max speed moves the tactical logits |
| K8 | count guards | log successful == expected, failed == 0, CSV rows valid, agent calls == seam declaration |
| KX | seam transparency | E2's K4 (seam-CV ≡ official CV, 220 × 20, max \|Δ\| 0.0) holds for the unchanged seam agent; STOP_zero through the seam reproduces E2 exactly (KH) |

## 8. Four metric families

Warmup stage 2 and navhard stage 2 carry no GT future (synthetic starts) ⇒ LONGITUDINAL / LATERAL /
TACTICAL **UNAVAILABLE, reason + n = 0** there. **navhard STAGE 1 (n = 450, real frames, logged
human futures in the export)** makes them COMPUTABLE for R6_A1 through
`taniteval/adapters/navsim.py` (`scenes_to_win` → `four_families_block`, log-cluster CI); navtest
through the same adapter. STRATEGIC: not applicable to this arm (strategic layer off,
`--no-strategic`) and NavSim scores no strategic decision — per family, with the reason.

## 9. Amendment A1 — made 2026-09-24 ~01:10 Berlin, BEFORE any navhard or navtest refcv6 score

**What happened.** The step-1,000 navhard bridge REFUSED stage-1 token #243 (and stopped, as
designed): its history frames sit at (−2.0, −1.5, −0.5, 0) s — the log DROPPED a 2 Hz frame, so the
§2 construction's (−1.0, −0.5, 0) states do not exist. Census from the exports: warmup **0/220**,
navhard **2/5,912** (both stage 1), navtest **15/12,146** carry such a history (patterns (−2.0, −1.5,
−0.5, 0), (−2.0, −1.5, −1.0, 0), (−2.0, −1.0, −0.5, 0)).

**The amendment.** The ego-history rule is unchanged — linear interpolation in time between the
declared states (the last three history frames) — but is applied at their ACTUAL times, under guards
that keep it an interpolation: t0 = 0; strictly increasing; the oldest used state at or before the
window's first slot (−0.7 s: never an extrapolation); no gap over 1.0 s. Anything else is refused.
Frames are unaffected (ST reads only t0; NT already uses the actual times).

**Why it cannot have been tuned on a score.** It touches NO warmup scene (all 220 are on the grid),
and no navhard or navtest refcv6 score existed when this was written. ⚠️ Stated exactly: ONE warmup
refcv6 score did exist — R6_A1's warmup CSV landed at 2026-09-23T23:03:25Z, minutes before this
amendment — and it is unaffected by construction (0/220 scenes carry a dropped frame). Tests: `tests/test_inputs6.py::test_hist_dropped_frame_is_interpolated_at_actual_times`
(the literal 2.0 m/s² survives every dropped-frame pattern) and
`::test_hist_refuses_gaps_extrapolation_and_bad_t0` (a 1.2 s gap, an extrapolation and t0 ≠ 0 are
refused).

## 10. Amendment A2 — made 2026-09-24 ~01:25 Berlin: a LOADER defect found by control KL; no protocol change

**What happened.** Control KL (`tests/test_loader_crosscheck.py`: this package's loader vs the battery
stream's trainer-exact loader, same NavSim scene, inputs, lift, precision and seed) FAILED: the
selections differed on 3/3 scenes and plans by up to 66 m. Traced to ONE attribute:
`core.decoder.anchor_control_units` = **'kappa'** in this package's build vs **'alat'** in the
trainer's. `refc_v3_train.train()` reads the anchor artifact (`_read_anchor_artifact`, `:6726`)
BEFORE `_pin_trainer_cfg`, so the file's DECLARED units (`alat`) and derivation constants reach the
decoder; `taniteval/tools/refcv3_arm.rebuild_config` pins WITHOUT that read, and this run's argv
carries no `--anchor-control-units` (the file declares it) — so the rebuilt decoder rolled a
lateral-acceleration vocabulary as curvature (anchor bank off by up to 108 m).

**The fix** (`code/refcv6_bridge.load_refcv6`): replay train()'s order — the artifact read runs before
the pin — and REFUSE a build whose decoder units are not the artifact's `alat`. After the fix KL
passes **bit-exactly** (max |Δ| 0.0, identical selections, 3/3), and K0 / KD / KI / KT pass again.

**Consequences, stated.** Every refcv6 trajectory or score produced before the fix is VOID and is
quarantined in `raw/VOID_wrong_anchor_units/` (the first warmup step-1,000 run incl. R6_A1's CSV, the
runner check, 292 navhard rows); §0's "two seeds moved one plan by 24.2 m" was measured on the
mis-built model and is WITHDRAWN (re-measured by R6_A1_s1). No bar, arm, statistic or input changed;
the step-1,000 runs are re-done from scratch with the correct build.

## 11. Amendment A3 — made 2026-09-24 ~01:50 Berlin (23:50Z), BEFORE any corrected refcv6 score: the device rule is per SPLIT, and cross-checkpoint reads carry a precision floor

**What was inconsistent.** §2 says "one device and one precision per checkpoint"; the runner
(`code/run_navsim_refcv6.py`) decides the device per SPLIT (all arms of a split share it; the bridge
refuses a mixture within an arm). On this shared box the GPU gate opens and closes with other
sessions' jobs, so a per-checkpoint rule would either idle a whole milestone on a busy card or force
every split onto CPU because one split started while the card was taken.

**The amendment.** The unit is the **split**: every arm of a split — including the seed replicate
`R6_A1_s1` and every lever arm — runs on ONE device and ONE precision (CUDA → bf16 + NHWC as
trained; CPU → fp32), recorded in every seam manifest and in `MILESTONE_SUMMARY.json`. No bar, arm,
statistic or input changes: every bar (§5) and every lever read compares arms WITHIN one split.

**What it costs, and the control that prices it.** Across checkpoints on the same split the devices
may differ (e.g. warmup@5000 on CPU fp32, warmup@10000 on CUDA bf16), so a step-to-step difference
could contain a precision effect. New control **KP (precision floor)**: the same checkpoint, the same
per-scene seeds, `R6_A1` on warmup, CPU fp32 vs CUDA bf16 — per-scene 4 s endpoint distance and the
S2-EPDMS-u difference — measured the first time the GPU gate opens. Any cross-checkpoint comparison
whose devices differ is read against KP (and against the seed floor); a difference inside either is
not a training effect.

**Why it cannot have been tuned on a score.** At this moment NO corrected refcv6 score exists: the
corrected step-1,000 warmup bridge has written trajectories only (its scoring queue starts after the
bridge); the navhard step-1,000 bridge has 138 corrected rows and no score; no milestone exists (Thor
at step 2,621).

## 12. Amendment A4 — made 2026-09-24 ~02:06 Berlin (00:06Z), BEFORE any corrected refcv6 score: the max-speed channel's TRAINED semantics are an ego-future ORACLE — measured against the map input, never substituted for it

**What was found (from the run's own record, not inferred).** refcv6 was trained with
`v_max` = `g_tac.goals.SPEED_BAND.v_hi_ms` = **the max of the ego's OWN REALISED speed over
[t0+2 s, t0+6 s]**, containing-window one-hot over {30, 50, 100, 120} km/h: run `config.json`
(md5 `0c9665f593fa321fd36e8e9d9fef41ac`) `max_speed_onehot_v6.provenance = "ego-future (oracle
INPUT, ~1.7555 bits)"` (line 670) and `refcv6_max_speed.train._derivation` (line 1894);
`tanitad/data/v7_labels.py::oracle_max_speed`: *"deployment supplies a LIMIT the driver may not
reach. That is a TRAIN/DEPLOY MISMATCH, not a leak"*; `tanitad/refs/refcv6_max_speed.py`: *"The PI
authorised it as a declared oracle INPUT standing in for a map/nav set-speed service."* Because
the window CONTAINS the realised max, the training one-hot told the model a LOWER bound as well as
an upper one (bin 50 = "your max over the next 2–6 s will be above 30 km/h"). The map posted limit
this SPEC feeds (§2, R6_A1 — the brief's input and the PI's own stand-in) is an upper bound only.

**Nothing in §1–§11 changes.** R6_A1 keeps the map input and remains the only bar arm; every bar,
statistic and floor is as written. Added, as MEASUREMENTS of the mismatch:

1. **Input census (label-free, no model), navtest, all 12,146 tokens** — `code/vmax_oracle.py`:
   the training definition evaluated on NavSim's 2 Hz log (max |v| over the frames at t0+2.0 … 6.0
   s, 9 samples, `ego_dynamic_state`; it can only UNDER-read the 10 Hz max), its bin distribution vs
   the training shares (38.08 / 34.84 / 22.18 / 4.90 %), and the map-bin → oracle-bin confusion with
   the share of tokens where the map tells the model a HIGHER band than the human drove.
2. **Two navtest DIAGNOSTIC arms on W3's 200-token subset** (`A1_sub200_tokens.json`), at every
   checkpoint the navtest split runs: **R6_VMAXOFF** (§3's withheld arm, extended to navtest) and
   ⛔ **R6_VMAXORACLE — PRIVILEGED**: the one-hot of the human's own realised max over
   [t0+2, t0+6] s (the training definition, item 1); a token without that log window gets
   `valid = 0`, counted. It is **never a result, never in a bar, never set beside a published
   number**; it exists to price the train/deploy mismatch. Reads, paired on the 200 tokens with
   W2's paired log-cluster interval: **ORACLE − A1** (what the deployable stand-in costs relative
   to the trained semantics) and **A1 − VMAXOFF** (what the map input is worth over none).
   ⚠️ navtest carries no `_s1` arm, so the inference-seed floor is cited from warmup/navhard.

**Why it cannot have been tuned on a score.** At this moment no navtest refcv6 score exists (the
416 navtest bank has 6 of 32 shards), and no corrected warmup or navhard score exists (the warmup
bridge is on its last arm; nothing is scored).

**A4 addendum (00:12Z, still before any navtest refcv6 score; the census below is label-free).**
The census ran (`raw/inputs/vmax_oracle_navtest.json`, twice, identical): oracle available on
**12,084 / 12,146** tokens (62 lack the [t0+2, t0+6] s log window); oracle bins **30: 75.74 %,
50: 23.50 %, 100: 0.75 %** (training: 38.08 / 34.84 / 22.18 / 4.90 %); on the **6,623** tokens with
both a map limit and an oracle, the map bin is **ABOVE** the oracle's on **3,474 (52.45 %)**, equal
on 3,034 (45.81 %), below on 115 (1.74 %); **5,461** tokens have an oracle but NO map limit (their
R6_A1 input is the all-zero "unknown" row). ⇒ The two diagnostic reads are therefore also reported
**per census group** — no map / map = oracle / map > oracle / map < oracle — because ORACLE − A1
mixes two different mechanisms (an OOD "unknown" row vs an over-high band) and a pooled number
would hide which one carries it.
