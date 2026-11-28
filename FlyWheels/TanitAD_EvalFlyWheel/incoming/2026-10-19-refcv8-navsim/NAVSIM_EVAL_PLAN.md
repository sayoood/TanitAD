# NAVSIM_EVAL_PLAN — scoring refcv8 (final step 48,852) on NAVSIM, zero-shot

EvalFlyWheel · 2026-10-19 (Europe/Berlin) · package `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/`
(written into the qland hold `C:/Users/Admin/qland/hold_r8navsim/stage/…`; nothing committed, nothing staged).

**Verdict: BUILDABLE, about 1 agent-day of porting plus about 7–10 GPU-hours on the 4060 or Thor.**
Almost everything already exists:
* the NAVSIM v1.1 and v2 harnesses, both reproduced against the leaderboard;
* the navtest metric cache;
* a 416×1024 cylindrical navtest frame bank, 12,146 tokens;
* the navhard and warmup banks;
* the refcv7 bridge, which scored refcv7-50,400 at navtest PDMS 69.99 (map ceiling) and 70.20 (LEGAL inputs).

refcv8's NAVSIM bars are **already pre-registered**, in `Project Steering/SPEC_REFCV8_DRAFT.md` §6 (BAR-R8-N1…N4, LEGAL and
PRIVILEGED rows, controls, Rule-Zero table). This plan does not re-register them. It names what must be built, what each
step costs, and which decisions belong to the PI.

Tier on every number: **NavSim open-loop benchmark (T1-family), zero-shot PhysicalAI-AV → nuPlan cameras, non-parity; v1
background non-reactive, v2 IDM-reactive; never closed loop.**

---

## 0. Corrections to the brief (read first)

| brief said | source says | evidence |
|---|---|---|
| camera input "CYLINDRICAL 256×640, f_ref 305.577" | refcv8 runs at **416×1024 cylindrical, f_ref 488.924 px** (= 305.577 × 1024/640; hfov 120.000°, vfov 46.09°). 256×640 was refcv4b's frame. | argv `--image-hw 416 1024` (`stack/ops/runs.d/refcv8-r101-s0.argv.json`); `tanitad.models.trunk_shapes.FRAME_416x1024`; MEASURED in `raw/camera_smoke/camera_smoke.json` S1 |
| "build the remap/stitch" | **already built and banked** for all three splits: `frames416.py`, a pure-rotation virtual cylinder whose boresight is CAM_F0's, rolled level; per pixel, the camera with the largest cos among L0/F0/R0 is taken. | `…/2026-09-23-refcv6-standard-tests/navsim/code/frames416.py:2-50`; banks listed in §3 |
| "the vendored PDM simulator path" as the scorer | The **official devkit scorer** (v1.1 `run_pdm_score`, through the seam agent) is the scorer of record. The vendored `refe/navsim_pdm` + `navsim_dac.py` reproduces DAC and comfort only. | §2 |
| "a speed ceiling from nuPlan map lane speed limits" | That is the **PRIVILEGED** row. The pre-registered bar row (LEGAL) feeds the **trained unknown row**: NavSim's `AgentInput` carries no map, so leaderboard legality excludes the map limit. Our own admissibility rules would allow it. | SPEC_REFCV8 §6.1; `refcv8_train.speed_input_treatment` (`legal` → unknown) |

---

## 1. Input mapping — what refcv8 consumes, and where each input comes from on NAVSIM

refcv8 is built by the refcv7 loader (`stack/tanitad/eval/refcv7_loader.build_model`) as re-pointed by
`…/2026-10-15-refcv8-milestone-battery/code/r8_repoint.py`. G0 has passed through it on refcv8 milestones.

The forward signature is `refc_v3.py:2281-2300`:
* inputs: `frames, nav_cmd, v0, steps, …, nav_args, v_max_ms, v_max_valid, ego_poses, ego_n_past, …`;
* new in refcv8: `r8_nav`, `r8_rc`.

The NAVSIM-side functions are the refcv6 bridge's (`refcv6_bridge.py`), which refcv7 imported unchanged
(`refcv7_bridge.py:96-133`).

| input | refcv8 (source) | NAVSIM / OpenScene source (built?) | admissible? | gap and likely effect |
|---|---|---|---|---|
| **camera** | 416×1024 cylindrical, f 488.924. 10 raw frames at 10 Hz (t0−0.9…t0) = 8 window rows × 3-frame D-015 stack (`refcv6_bridge.py:47-52`). `--trunk-dedup-frames`. | CAM_L0/F0/R0 (1920×1080 pinhole + Brown distortion, fx 1545, hfov 63.7°, vfov 38.5° — MEASURED, smoke S3) are re-projected into the virtual cylinder. **BUILT**: navtest bank `D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank` (12,146 tokens, 4 frames each, sha-checked per load). NavSim gives 4 frames at 2 Hz, so the 10 slots are filled by `slot_sources`: **ST** = every slot ← t0 (the primary), **NT** = nearest in time (the sensitivity arm). | vision ✓ | (a) **No motion in the stack under ST.** The model trained on 10 Hz motion stacks. (b) **About 12 % unobserved**: nuPlan vfov 38.5° < 46.1°; observed 0.8817 (smoke). (c) **3-camera seams and parallax**: pure rotation, camera centres about 0.3 m apart. (d) **Pitch and height**: in the smoke pair the horizon sits **23.3 px (2.7°) higher** on NAVSIM (CAM_F0 pitched 2.3° down) than on PhysicalAI (+0.43° up). CAM_F0 is 1.52 m above the lidar origin; the PhysicalAI camera is 1.25 m above the vehicle origin. The lift uses the NavSim rig (`refcv7_bridge.hires_lift_for_rig`), so BEV geometry is corrected, but the trunk sees a shifted road plane. (e) Native f-theta 120° vs a pinhole stitch. |
| **past ego poses** | `--ego-history`: window 8 × (x, y, yaw, v) at 10 Hz over t0−0.7…t0, `set_ego_window(poses, 8, actions=None)` | `ego_history_poses` (`refcv6_bridge.py:124-160`): LINEAR interpolation in time between the 2 Hz `ego_pose` / `ego_velocity` states at t0−1.0, −0.5 and 0. Yaw is unwrapped first. It refuses extrapolation and gaps over 1 s; 15 navtest tokens with a dropped frame go through amendment A1. **BUILT.** | measured odometry up to t0 ✓ | Inside each 0.5 s segment the encoder sees that segment's **mean** acceleration and yaw rate. That is a smoothed, 0.25 s-lagged view against PhysicalAI's 0.1 s backward differences. |
| **v0** | `v0` = \|velocity\| at t0 | `v0_of` = `hypot(vx, vy)` of `ego_velocity[t0]` (`refcv6_bridge.py:115-121`). **BUILT.** | ✓ (PI 2026-09-02) | none known |
| **a0, yaw rate** | not separate inputs. `--residual-prior ha0_ext_pose`: the model's prior derives a0 (backward accel) and κ0 from yaw rate / v **from the ego window** inside its forward (`refcv7_bridge.prior_model_free`; `kinematic_prior.prior_controls`) | the same window. NavSim's `ego_status.acceleration` exists but is NOT read (not declared; `refcv6_bridge.FIELDS`). | ✓ (PI 2026-10-18: IMU accel and yaw rate are admissible; curvature is not, and none is fed) | 2 Hz smoothing as above. Control **KPR** (the emitted prior equals the bridge's model-free prior, ≤ 1e-4 m) is inherited from refcv7. |
| **nav command** | `nav_cmd` token (v9, `--r8-nav-from-v9`) + **`r8_nav` [B,6]** = (log1p d_next/50, log1p d_end/50, dyaw/90, side, log1p lookahead/50, **known**) (`refcv8_train.py:66-72, 367-380`) | `driving_command` one-hot → left → TURN_L, forward → FOLLOW, right → TURN_R, unknown → FOLLOW (`refcv8_train.NAVSIM_LEGAL_NAV_CMD`, `:71-72`; E2 `NAVSIM_CMD_TO_NAV_NAME`). **`r8_nav` = None**, which `refc_v3.py:1968-1973` turns into exactly zeros [B,6]. That is the trained known=0 row (`--r8-nav-args-dropout 0.5`, `refcv8_train.py:563-567`). `r8_rc` = None (`--r8-no-rc`). | ✓ the nav command (leaderboard-legal) | NavSim's command fires on curves (\|y\| ≥ 2 m at 20 m ahead) and carries **no distance**. The v9 token marks announced junction turns, with distance. D6 measured premature turns on about 59 % of 724 command-but-straight navhard scenes (refcv7 30k; SPEC_REFCV8 §6.1). ⚠️ The NavSim command is itself derived from the logged future route; this is standard leaderboard practice, and disclosed. |
| **speed ceiling / road type** | `--max-speed-input-v6 --r8-speed-input label`: a 4-way one-hot {30, 50, 100, 120} km/h, containing-window. The unknown row (all-zero, valid 0) is **trained at p 0.45** (`--r8-speed-unknown-p 0.45`; `refcv8_train.py:76, 313-342`). | **LEGAL (bar row): the unknown row**, `v_max_ms 0, v_max_valid 0`, identical to refcv7's `R7_VMAXOFF` feed. **PRIVILEGED (diagnostic)**: the nuPlan-map lane limit (`export_speed_limits.py`; navtest export **MEASURED: 6,658 / 12,146 tokens carry a limit, 5,488 do not (45.2 %)**). Snapped to the road-type ladder (ESTIMATED, my reading): **50 km/h on 5,141, 30 on 1,348, 100 on 169** (`raw/navtest_speed_limit_census.json`). | LEGAL ✓. PRIVILEGED: admissible under the programme's rules (a map attribute, past-only) but **not leaderboard-legal** | refcv8's label is the Alpamayo speed-band upper edge, i.e. a **desired** speed, not a posted limit. US 25 and 30 mph (40.2 and 48.3 km/h) both snap to the 50 bin. Under LEGAL the ceiling filter and the plan cap are **inert** (ceiling +inf). |
| **route checkpoint** | none (`--r8-no-rc`) | — | — | — |
| **tactical / situation** | internal (the tactical head conditions fan generation and selection). No goal input carries a classifier output. | — | ✓ (goal/situation disjointness holds: there is no goal input) | — |
| **history length / rate** | 0.9 s of frames at 10 Hz; 0.7 s ego window at 10 Hz | NavSim: 4 frames and 4 ego states at 2 Hz over 1.5 s | — | see the camera and ego rows. The PhysicalAI-only clock corrections (`--pose-sync-sidecar`, `--clip-clock-sidecar`) do not apply on NAVSIM. |

Other inputs in the argv that are **not** inference inputs: `--vis1-sidecar`, `--agent-join`, `--join3d`, `--map-gt-root`
and `--v7/v9 labels` are targets or labels. The refcv7 bridge asserts `agent_gt = None` at inference ("no oracle agents",
`refcv7_bridge.py` docstring §2). The refcv8 port must re-assert this. *(UNVERIFIED per flag for refcv8 until the port's
premise list runs.)*

**Domain gaps not visible in the inputs, MEASURED where stated:**
* **Left-hand traffic**: `sg-one-north` holds **1,914 / 12,146 navtest tokens (15.8 %)** across 13 of 136 logs
  (`raw/navtest_city_census.json`). PhysicalAI-AV's handedness mix is **UNVERIFIED**; no census was found in the repo.
* **Rig** (above).
* **US driving style and nuPlan agents.**
* **Exposure**: the smoke's PhysicalAI frame is a dusk clip, mean px 58.9 against 88–91. That is one sample, not a
  statistic.

## 2. Output mapping

refcv8 emits 8 knots at horizons (5, 10, 15, 20, 30, 40, 50, 60) × 0.1 s (`refcv7_bridge.HORIZONS`; refcv8 is
warm-started from refcv7 and its argv changes no horizon).

NAVSIM wants 8 poses (x, y, heading) at 0.5…4.0 s. The conversion is **built and validated**: E2's `knots_to_navsim`
(`…/2026-09-19-navsim-refcv4b-bridge/code/tanitad_navsim_bridge.py:292-330`), imported unchanged by refcv6 and refcv7.
* **Interpolation**: a C2 not-a-knot cubic spline in time, per axis, through the origin (t = 0) and the 8 knots,
  evaluated at 0.5…4.0 s. Knots are reproduced at 0.5/1/1.5/2/3/4 s; 2.5 and 3.5 s are interpolated; the 5 s and 6 s
  knots shape the spline but are not emitted.
* **Heading**: atan2 of the spline tangent (the rear-axle path tangent), held at the previous value where \|p′\| <
  1.0 m/s, and taken on the branch nearest the previous value.
* **Axes**: x-forward / y-left / CCW in both systems, with no permutation.
* **Validation**: GT round-trip < 1e-3 m (E2 K2/K3). The HUMANseam arm reproduced the human through the seam exactly
  to float32 (W3 `RESULT.md` §4 row 3).

**refcv8-specific post-processing, applied to the 8 knots BEFORE the spline:**
1. **The plan cap** (`plan_vmax_clamp` / `cap_plan_to_limit`, A-R8-CLAMP-1): inert on LEGAL (no ceiling), active on
   PRIVILEGED.
2. **The re-timing stage** (`stack/tanitad/retime_stage.py`, D-R8-SHIP-RETIME, PI decided 2026-10-19 (a)): **shipped
   iff SHIP-R8 passes at 48,852**. Its prior `ha0_ext_pose` is computable from the NAVSIM window (`prior_model_free`), so
   the stage can run on the NAVSIM seam. It enters as its own arm (`R8_RT`) on the **same forward**, so it costs no
   extra GPU.

**The vendored PDM path** (`TanitAD Research Lab/…/2026-09-20-refe-plan/refe/navsim_dac.py` + `refe/navsim_pdm/`, LQR +
kinematic bicycle verbatim from navsim @ 3e8291b) can take refcv8's output **as-is**. Its `navsim_dac_and_comfort(props,
ego_state, map_api, grid=…)` accepts already-NAVSIM 8-pose plans when `grid != "refe20"` (`navsim_dac.py:180, 193, 207`),
and it reproduces NAVSIM's DAC at 100.00 % (memory note `navsim-dac-needs-its-lqr-simulator`). It gives **DAC and
comfort only**: no NC, TTC or EP, so no PDMS. Use it for fan diagnostics such as D6 P4's changed-pick DAC. **The PDMS of
record is the devkit scorer through the seam agent.** The v1.1 harness reproduces CV at **20.6517**, the leaderboard
value to 4 dp (MEASURED, W3 `RESULT.md` §3.1).

## 3. Data and compute

**Already on the dev box** (MEASURED by listing; "absent" means probed at two paths and under name variants). D: is currently `subst D: E:\` (INHERITED, refcv7 navsim `RESULT.md` §6.1), so D: and E: are one exFAT disk:

| item | where | state |
|---|---|---|
| navtest (v1.1 PDMS, 12,146 tokens / 136 logs) | metric cache `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest` (137 entries); inputs `…/inputs/navtest_inputs.json.gz`; **416 bank** `D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank` (22 GB, 32 shards, index.json); raw camera shards `D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera` (32 tgz); logs `…/data/openscene/navsim_logs/test` (147 pkl) | ready |
| navtest v2 metric cache (EPDMS on navtest, refcv7 addendum) | `C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2` | present |
| navhard two-stage (v2 EPDMS, 5,462 stage-2 / 450 clusters) | data `D:/…/data/openscene/navhard_two_stage` (EXTRACT_DONE) + `data/navsim-v2/`; metric cache `C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage`; banks `C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/navhard_s1`, `navhard_s2` (10,924 files); exports `C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage` | ready |
| warmup two-stage (220 scenes, 16 clusters) | data `D:/…/data/openscene/warmup_two_stage` (sensor blobs extracted); bank `C:/…/refcv6-navsim-20260923/warmup_two_stage` (204 keys) | ready |
| nuPlan maps | `D:/Archive/devbox-C/navsim/data/maps` (nuplan-maps-v1.0: Boston, Pittsburgh, Las Vegas, Singapore) + `nuplan-maps-v1.1.zip` | ready |
| devkits / venvs | v1.1 `D:/Archive/devbox-C/navsim/navsim-3e8291b…` + nuplan-devkit v1.2; v2 `C:/Users/Admin/navsim-crun/venv`; TanitAD venv `C:/Users/Admin/venvs/tanitad` | ready |
| refcv8 checkpoints | `D:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt`, `ckpt_ema_30000.pt`, `ckpt_40000.pt`. **Final `model_ema.pt` / `ckpt.pt` at 48,852 arrive about 2026-10-20 evening.** | partial |
| **navmini** | absent (`D:/Archive/devbox-C/navsim/data`, `C:/Users/Admin/navsim-crun`, name `*navmini*`). Not needed: it is not a benchmark split. | absent |
| navtrain | the A40 pod per PI 2026-09-20 (449 GB). **Not needed** for zero-shot scoring. | n/a |

**First score:** **v1.1 PDMS on full navtest, LEGAL row.** This is BAR-R8-N1 and the comparable public number.

**Beside it:**
* navhard two-stage EPDMS, the LEGAL row and BAR-R8-N2, which is the decisive NavSim bar per SPEC §6.2;
* warmup S2-EPDMS-u (BAR-R8-N3);
* navtest PDMS vs refcv7-50,400 LEGAL (BAR-R8-N4).

**Cost.** Forward rates are refcv7's (MEASURED); refcv8's own forward cost is **UNVERIFIED**. ESTIMATE ×1.0–1.3 for the r8
heads, because nothing on refcv8 has been timed at NAVSIM shape.

| step | unit cost (evidence) | arms | wall |
|---|---|---|---|
| bridge forward, navtest | refcv7 CUDA bf16 on the 4060: **0.35–0.42 s/scene** (MEASURED, refcv7 navsim `RESULT.md` §4). CPU fp32: **2.58 s/scene** cumulative, rising to about 11.6 under contention (MEASURED, §6.1). | LEGAL EMA s0, LEGAL EMA s1, LEGAL raw s0, PRIVILEGED EMA s0 (the retime and cap arms reuse the s0 forward) | **≈ 5–7 GPU-h** on the 4060. ⛔ The CPU fallback is about 35–45 h: not acceptable. |
| bridge forward, navhard (5,912 tokens) | same rate | LEGAL EMA s0, s1, raw s0 | ≈ 2–3 GPU-h |
| bridge forward, warmup (220) | same rate | LEGAL EMA s0, s1, raw, **R8_BLIND** | minutes |
| scoring, navtest v1.1 | **15–25 min per arm**, one CPU worker, about 0.45 GB RSS (MEASURED, W3 `RESULT.md` §2) | 4 model + R8_RT + floors (STOP / CV / HUMAN / PRIOR_ha0p already banked) | ≈ 2 h CPU |
| scoring, navhard two-stage | **≈ 3.2 h CPU per arm**, ≤ 4 in parallel, RAM guard (MEASURED, refcv7) | 3–4 | ≈ 3.5–7 h wall |
| four families + bars | seconds (MEASURED, W3 §2) | all | — |

**Total: about 7–10 GPU-hours plus about 6–9 CPU-hours wall. No spend.**

**Where it should run (PI decision D-NS8-1):**
1. **Dev box 4060**, once the refcv8 final chain releases the GPU lock. Zero cost, and every floor arm (STOP / CV /
   HUMAN / PRIOR) is already banked on this backend, so every comparison is backend-consistent.
2. **Thor**, once refcv8 training ends (about 2026-10-20 15:00–18:00). `RULING_BACKEND_POLICY.md` §3 says *"every refcv8
   NavSim row goes to Thor first"*. Mixed-backend comparisons are admissible at the stated ulp tolerances, but the
   floors must then be re-scored on Thor or the comparison stated as mixed. ⛔ This is outside this brief: no Thor
   action was taken.
3. The **A40 pod** is not recommended. It would need the 22 GB bank plus caches shipped (about 5 h at the 1.2 MB/s
   uplink) or rebuilt from HF (128 GB of camera shards plus about 6 h of stitching), plus spend.

## 4. Feasibility smoke — DONE, PASS (CPU, about 1 s of compute, no model)

`code/camera_smoke_r8.py` → `raw/camera_smoke/camera_smoke.json`, `raw/camera_smoke/camera_smoke_side_by_side.png`
(`raw/camera_smoke.stdout.txt`). MEASURED:

| check | result |
|---|---|
| S1: the frame contract | argv 416×1024 == `FRAME_416x1024`: cylindrical, f_ref 488.924 = 1.6 × 305.577, hfov 120.000°. Column linear in azimuth to 4.1e-5 px (float32 rays). |
| S2: re-stitch of one NAVSIM warmup scene **from its raw CAM_L0/F0/R0 jpgs** through the launch tree's `frames416.build_one` | **bit-identical** to the bank the refcv7 suite fed its model (sha16 `d44e7bc225ec67a7` ×3, max \|Δ\| 0 px; rig key equal). Observed 0.8817. **0.73 s CPU per scene** (3 frames). |
| S3 / S3b: an **independent** remap written in the smoke (plain pinhole, no distortion, nearest) of the raw CAM_F0 jpg over \|az\| < 15°, \|el\| < 8° | mean abs diff **6.27** grey levels on 35,328 px. **Control**: the same remap shifted +2° reads **24.25** (3.9× worse). The stitch geometry is right. ⚠️ S3's boresight-column test is a structural identity, reported but not relied on. |
| S4: a navtest token from the 416 bank | sha matches index.json. |
| S5: a PhysicalAI eval139 frame (`refcv6-b1-416x1024-eval139`) | stored `frame` == `FRAME_416x1024`, so it is the **same canonical frame on both sides**. |
| S6: side-by-side PNG | PhysicalAI / NAVSIM warmup / NAVSIM navtest, with analytic azimuth ticks ±30°/±60° and analytic horizon rows: **PhysicalAI 211.2 vs NAVSIM 187.9** (pitch +0.43° up vs 2.3° down). Visible seams at about ±32° and unobserved corner wedges on NAVSIM. |

This smoke does **not** test refcv8's forward on a NAVSIM token. That is step B1, deferred tonight because the dev-box GPU
is busy and a CPU forward of the 100 M-param model at 416×1024 would eat into the shared RAM. Free commit was 10.0 GB at
22:2x, and the brief's gate is 5 GB.

## 5. The plan

### 5.1 Steps

| # | step | built? | compute | when |
|---|---|---|---|---|
| **B0** | **Addendum A-R8-NS1 to SPEC_REFCV8 §6**, registered **before any refcv8 NavSim number**. It adds four things. (i) The bars read the **EMA** weights (`model_ema.pt`) with raw beside (PI 2026-10-16). (ii) The `R8_RT` arm, and whether it may carry a bar depends on the SHIP-R8 verdict (D-R8-SHIP-RETIME). (iii) The PRIVILEGED speed mapping: the map limit snapped to the containing road-type bin, with "no limit" → unknown row. (iv) Pre-registered **strata**: city (Singapore vs US), command (straight vs turn), t0-speed band, ≤ 5 m EP-clause fired. | new (doc) | 0 | now |
| **B1** | **`refcv8_bridge.py` + `run_navsim_refcv8.py`**, a port of `refcv7_bridge.py` / `run_navsim_refcv7.py`. (a) Load through `r8_repoint.apply()` then `refcv7_loader.build_model`, EMA or raw by path. (b) The premise list changes "no nav-args" to the refcv8 set: `r8_enabled`, `--r8-no-rc`, nav-args-dropout 0.5, `speed_input == "label"`, unknown p 0.45, horizons 5…60, 416×1024, 117 anchors, `ha0_ext_pose`. (c) `forward_kwargs8` = the trainer's `compute_losses_v3` keyword set **plus** `r8_nav` / `r8_rc` (None → the trained zero rows) and the LEGAL `v_max` (0, 0). The model's `_r8_legal_row = True` is stamped and recorded. (d) The cap and the retime stage run after the forward and before `knots_to_navsim`. (e) Arms are listed in §5.2. **Tests**: KT8 (on a real PhysicalAI eval window the bridge feed equals the trainer's captured feed, bit-identical plan), K0, KD, KPR, KL-lift, a LEGAL-row literal test (`r8_nav` all zeros, `v_max_valid` 0), and mutation arms (`kw_drop`, `wrong_grid`, `legal_off`: feeding a nav arg must go RED). | port (refcv7 took about 1 day from SPEC to validated path, per its `RESULT.md` timestamps) | CPU tests ≤ 5 GB, gated | 10-20 daytime |
| **B2** | **Validation read on `ckpt_ema_40000.pt`**: warmup (220) + navtest sub-200. ⛔ VALIDATION ONLY: it proves the path and is never refcv8's NavSim performance. | — | CUDA minutes, or CPU about 20 min | when a gap opens before the final checkpoint |
| B3 | **Final read, 48,852 EMA (bars) + raw (beside)**: navtest → navhard → warmup, per §3, then BARS.json, four families and strata. | runner from B1 | 7–10 GPU-h + 6–9 CPU-h | after the checkpoint and the PI's D-NS8-1 |
| B4 | **Ego-Status-MLP control**: the devkit's own agent with the published `navsim_baselines/ego_status_mlp` seeds 0–2 (6,518,594 / 6,518,530 / 6,518,530 B, HF listing in W3 `RESULT.md` §4), scored in the same harness. | needs a **download approval** | CPU, about 20 min per seed | any time after approval |
| B5 | Report + register rows (§5.4) + leaderboard examination pack (§5.5) | — | 0 | after B3 |

### 5.2 Arms (navtest; navhard and warmup take the starred ones)

| arm | inputs | role |
|---|---|---|
| **R8_LEGAL** ★ (EMA, inference seed 0) | bare command, `r8_nav` zeros, unknown speed, ST frames | **bar arm** (N1 / N2 / N3 / N4) |
| **R8_LEGAL_s1** ★ | same, seed 1 | inference-variance floor (refcv7: 69.99 vs 69.85) |
| R8_LEGAL_raw ★ | raw weights | reported beside (PI 2026-10-16) |
| R8_RT | R8_LEGAL forward + retime stage | the deployed planner if SHIP-R8 passes |
| R8_PRIV | map limit snapped to road type; command as LEGAL | diagnostic only, never a bar; PRIV − LEGAL is reported with a paired CI |
| R8_NT (optional) | nearest-time frames | the static-stack sensitivity |
| **R8_BLIND** ★ (warmup) | constant grey | deliberate regression: must FAIL N3, else the panel is VOID (SPEC §6.2) |
| floors: STOP, CV, HUMAN, **PRIOR_ha0p** | banked, same harness | STOP 61.8202, CV 20.6517, HUMAN 94.5514, PRIOR_ha0p 58.6613 (navtest, refcv7 step-50400 `summary_navtest.json`, MEASURED) |
| refs: refcv7-50,400 LEGAL / A1, refcv4b | banked | 70.2048 [68.55, 71.82] (Thor; MODEL_REGISTRY §15.4.1) / 69.9904 / 58.9738 |

**PRIOR_ha0p** is the in-house **ego-status-only planner**: v0, a0 and yaw rate from the window, extrapolated through
refcv8's own residual prior (Δ = 0). It is the cleanest "does vision add anything" control. **The published
Ego-Status-MLP** (65.6 in Tab. 1 / 66.4 in Tab. 3, arXiv 2406.15349, banked) is B4.

### 5.3 Expected failure modes of zero-shot transfer, and the reading each produces

Priors from refcv7-50,400 on the same harness (MEASURED, step50400 `summary_navtest.json`): NC 89.59, DAC 87.21, TTC
78.73, EP 68.35, C 99.93. The navhard anatomy at 30k is D6: DAC-zero 26.0 %, NC-zero 15.6 %, 94.8 % of those front
collisions, plan +3.87 m/s over the reference.

1. **Camera geometry and rig.** Static ST stack, 12 % unobserved, seams, a 2.7° pitch offset, camera height. → Expected
   as DAC loss at speed, and as the PRIV/LEGAL-independent gap to refcv7.
2. **Nav semantics.** Curve-firing commands with no distance → premature or late turns. → LEFT/RIGHT command stratum
   DAC; D6 classes ROUTE-FOLLOWING and OVER-STEER.
3. **Speed input.** LEGAL is the unknown row. refcv8's longitudinal behaviour was shaped mostly under label ceilings,
   and E1 found its progress mis-timed (eval gain 0.46). → EP and TTC; the `R8_RT` arm and the PRIV − LEGAL gap isolate
   it.
4. **Driving style and country.** Singapore left-hand is 15.8 % of tokens. → The city stratum. A Singapore-only collapse
   with an intact US stratum is a handedness finding, not a model-wide one.
5. **History at 2 Hz.** Smoothed a0 and ψ̇0. → KPR confirms the prior is computed as specified; the effect shows as
   EP/TTC at speed changes.
6. **The protocol itself.** STOP scores 61.82: braking progress plus the ≤ 5 m EP ≡ 1 clause on 8.28 % of tokens. →
   Every row is read against STOP, and BAR-W3-M1's gaming loophole stands ("STOP at turns" = 62.48), so **a stop-rate
   column is mandatory**.

Rule-Zero next levers are already tabulated in SPEC_REFCV8 §6.2 "Outcome". Additional, cheap, no training:
* if Singapore alone fails, a **mirror arm** (flip the image and lateral, swap L/R commands) on the sg tokens; this
  needs pre-registration;
* if DAC dominates, D6 P4's map-head drivable gate on the refcv8 fan, scored through `navsim_dac.py`.

### 5.4 How it is reported

* **Per-metric rows, never a lone PDMS.** NC, DAC, TTC, C, EP (+ DDC at weight 0) and PDMS for every arm, **beside STOP,
  CV, HUMAN, PRIOR_ha0p and refcv7 LEGAL**. Paired log-cluster bootstrap (136 logs; nuplan_drive 44 beside; B 2,000) vs
  STOP and vs refcv7. W/T/L counts, stop-rate, fraction where the EP clause fires.
* navhard: official two-stage EPDMS, per stage, with clustering on `corresponding_original_scene`, never per scene.
  warmup **never carries an interval** (7 logs < 8).
* **Four families**, via `families6/7.py`:
  * LONGITUDINAL: speed agreement, a compliance row (PRIV only), distance-keeping **UNAVAILABLE**: no lead tracks in
    this harness, n stated; a work item;
  * LATERAL: heading, curvature, yaw-rate, cross-track;
  * TACTICAL: lat/lon κ against the command;
  * STRATEGIC: **UNAVAILABLE** (no strategic label on NAVSIM, and the layer is OFF in refcv8; `--no-strategic`).
* Stamps on every number: tier T1-family open loop, zero-shot, non-parity, backend, device and precision, EMA or raw,
  inference seed, LEGAL or PRIVILEGED.
* Registry and claims: the **Master Mind** lands MODEL_REGISTRY §15 rows and GOALS_AND_CLAIMS rows (proposed:
  `R-REFCV8-NAVSIM`, `F-NAVTEST-SG-SHARE` = 15.8 %, `F-NAVTEST-NOLIMIT-SHARE` = 45.2 %, MEASURED).

### 5.5 Before vs after the final checkpoint

| before 48,852 (now → 10-20 evening) | after |
|---|---|
| B0 addendum; B1 port + CPU tests; B2 validation on EMA-40000 (when a GPU gap or RAM headroom allows); B4 download + score (if approved); floors are already banked | B3 bars; R8_RT (needs the SHIP-R8 verdict); report; leaderboard pack |

### 5.6 Leaderboard examination outline (for the PI's go/no-go — ⛔ NO submission)

1. **Which board.**
   * NAVSIM v1: navtest, PDMS. The split is public and our harness reproduces it (CV 20.6517 = LB to 4 dp), so **our
     local navtest score is already the number that board would show**, and no submission is needed to know the rank.
   * NAVSIM v2: navhard / private-test two-stage EPDMS. Its submission format, team registration and whether it is
     still open are **to retrieve, UNVERIFIED here**.
2. **Entries table** (to retrieve from the live boards and the papers, primary sources banked through
   `tools/kb_add.py`). For each: score per sub-metric, training data (navtrain or extra), sensors (cameras or LiDAR),
   model size, ensembles or test-time tricks, date. In-repo anchors, banked: TransFuser 83.9 (LB 83.8822), LTF 83.5,
   Ego-MLP 66.4, CV 20.6, Human 94.8 (arXiv 2406.15349); DriveZero matched row 94.55 (Table A13, D-REFE-GAP-1); our REFe
   83.24 [81.77, 84.63] (D-REFE-NAVTEST-1, trained on navtrain).
3. **Comparability caveats** printed on any comparison:
   * refcv8 is **camera-only, a single front stitch (3 of 8 cameras), zero-shot (no navtrain), non-parity**;
   * HUMAN is CLOSE, not reproduced (94.55 vs 94.8);
   * the STOP floor (61.82) is not shown on the board.
4. **Where refcv8 would rank**: the LEGAL row only, matched protocol, with the seed floor.
5. **Options**: (a) report internally only; (b) submit the v1 navtest LEGAL row; (c) wait for a navtrain-adapted
   variant. For each, what it publishes under the programme's name.
6. **The ask**: an explicit PI go/no-go. Nothing is submitted without it.

## 6. Decisions and blockers

| id | decision / blocker | owner | default if silent |
|---|---|---|---|
| **D-NS8-1** | where B3 runs: dev-box 4060 (backend-consistent with the banked floors) or Thor (backend policy "Thor first"; needs Thor-side floors or a stated mixed backend) | PI | dev box, after the refcv8 final chain frees the GPU |
| D-NS8-2 | download the Ego-Status-MLP checkpoints (3 × about 6.5 MB, HF `autonomousvision/navsim_baselines`) for B4 | PI / user | not downloaded; PRIOR_ha0p stands in as the ego-status control |
| D-NS8-3 | does `R8_RT` carry the bar? Tied to SHIP-R8 at 48,852 (D-R8-SHIP-RETIME (a)). | follows SHIP-R8 | the bar stays on `R8_LEGAL`; `R8_RT` beside |
| D-NS8-4 | leaderboard go/no-go after §5.6 | PI | no submission |
| blocker | refcv8's forward cost and RAM at NAVSIM shape are **UNVERIFIED** (no GPU tonight; CPU forward deferred for RAM) | B1/B2 | — |
