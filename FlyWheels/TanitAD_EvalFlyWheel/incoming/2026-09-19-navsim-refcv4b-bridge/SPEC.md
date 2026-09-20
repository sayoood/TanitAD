# SPEC — E2: refcv4b on the OFFICIAL NavSim v2 scorer, `warmup_two_stage` (PRE-REGISTERED)

**Written 2026-09-19, BEFORE any scoring run.** Its git blob hash is recorded in
`raw/SPEC_PREREG_HASH.txt` at the moment it was staged, and every score log in `raw/` is
timestamped after it. Brief: `BRIEF.md` (this folder). Owner: EvalFlyWheel stream E2.

## 0. ⛔ The scope correction this SPEC is built around (MEASURED before scoring)

The brief assumed a camera agent can be scored on the full two-stage protocol here. **It cannot.**

| fact | evidence (MEASURED 2026-09-19) |
|---|---|
| The scorer evaluates **16 stage-1** (original-log) + **204 stage-2** (synthetic) tokens | devkit `SceneLoader` over `warmup_two_stage`: `tokens_stage_one` 16, `reactive_tokens_stage_two` 204 (`raw/export_agent_inputs.log`) |
| The stage-1 scenes' camera frames are **not on this box**: **0 / 192** `cam_l0/f0/r0` history jpgs exist | `raw/navsim_agent_inputs.json` → `stage1_cam_files_present` 0 / `stage1_cam_files_expected` 192 |
| Every jpg under `warmup_two_stage/sensor_blobs` is a **17-hex synthetic render** (16-hex originals: 0 in all 7 logs); `openscene_meta_datas/` is the synthetic renders' metadata (510/510 frames 17-hex) | `ls` census + unpickle probe, this session |
| The stage-1 originals live in the OpenScene **test** camera archives (`openscene_sensor_test_camera_{0..31}.tgz`, `download_test.sh`) — not downloaded, and downloading is out of scope for this stream | devkit `download/download_test.sh`; brief "download NOTHING" |
| The DataFlyWheel bank covers exactly the **204 stage-2** scenes (it globs `synthetic_scene_pickles/*.pkl`) | `build_navsim_eval.py:155`; bank 204 × `.npy` |

⇒ **No refcv4b camera arm can produce the 16 stage-1 trajectories.** The official two-stage
EPDMS needs them twice over: as the stage-1 factor of every group score, and as the endpoints
that set the stage-2 pseudo-closed-loop weights (`scene_aggregator.py:20-47`, Gaussian kernel
σ² = 0.1 m² around the agent's OWN stage-1 endpoint). ⛔ **Therefore no number in this package
is, or may be labelled, a refcv4b two-stage EPDMS.** What IS computable for a camera arm is the
**official per-scene EPDMS row of every stage-2 scene** (all multipliers + weighted terms + the
two-frame EC, which for stage 2 is computed between the two synthetic scenes of a pair and needs
no stage-1 input, `scene_aggregator.py:111-122`).

## 1. Question

Does refcv4b (registry §4.6 `refcv4b-b1-v72-40k`, `ckpt_40284_FINAL.pt`, md5
`99b573e8277d94a5e3bfbf630cb4d751`), run zero-shot on NavSim's stage-2 synthetic scenes with its
measured t0 ego state and the NavSim driving command, drive better than the devkit's
constant-velocity agent **on the same 204 scenes**, as judged by the official scorer's per-scene
EPDMS? And what do ego status and the command contribute (paired arms)?

## 2. Primary statistic — S2-EPDMS-u

For one arm: take the official `score` column (EPDMS per scene — multiplicative product × weighted
mean with the two-frame EC injected, `run_pdm_score.py::compute_final_scores`, **never**
`pdm_score`) for every stage-2 token, and aggregate EXACTLY as the devkit's stage-2 aggregate
(`run_pdm_score.py::calculate_individual_mapping_scores`) with ONE change: **uniform weights inside
each group** instead of the agent's stage-1-endpoint kernel weights. Uniform weighting is the
devkit's OWN fallback when the kernel mass vanishes (`scene_aggregator.py:42-43`).

    S2-EPDMS-u = mean over the 16 groups g of ( mean over the stage-2 tokens of g of score )
    groups     = for each of the 8 reactive_all_mapping entries (orig, prev, pairs):
                 {pair[0] of every pair} (owned by orig) and {pair[1] of every pair} (owned by prev)

The same statistic is computed for every arm INCLUDING the CV reference, on the identical 204
tokens. Secondary aggregates (reported, not decisive): the plain mean over 204 scenes; each
sub-metric's mean over the 204 scenes (NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC); per-log means.

## 3. Arms (same checkpoint, same 204 stage-2 scenes, CPU inference)

| arm | frames | ego block fed | core `v0` | nav | declared EgoStatus fields |
|---|---|---|---|---|---|
| **A1** `A1_ego_cmd` (bar arm) | ST | (v0, a_long, yaw_rate, κ, keep=1) | v0 | NavSim command → nav | `ego_velocity[t0]`, `ego_acceleration[t0]`, `driving_command[t0]` |
| **A2** `A2_vision_pure` | ST | (0,0,0,0, keep=0) | None | None | none |
| **A3** `A3_ego_nocmd` | ST | as A1 | v0 | None | `ego_velocity[t0]`, `ego_acceleration[t0]` |
| A1NT (declared sensitivity) | NT | as A1 | v0 | as A1 | as A1 |
| A2NT (declared sensitivity) | NT | as A2 | None | None | none |
| A4 `A4_blind_ego_cmd` (diagnostic) | BLIND | as A1 | v0 | as A1 | as A1 |
| CV (reference) | — | — | — | — | devkit `ConstantVelocityAgent` (reads `ego_velocity[t0]`) |

**Formulas (t0 = the last NavSim history frame; nothing later is read):**
`v0 = hypot(vx, vy)` (as `physicalai.signals_at`: `v = hypot`), `a_long = ax` (the dataset's own
longitudinal acceleration, as PhysicalAI's `ax`), `κ = clip(ay / max(v0, 4.0)², ±0.12)` (the
decoder's own `alat_v_floor_ms` / `kappa_cap_inv_m`, `anchors.units.json`), `yaw_rate = v0·κ`
(`refc_v3.ego_state_at_t0`: `r0 = v0·k0`), `keep = 1`. A2 feeds zeros + `keep = 0` (the model
multiplies the values by `keep`, `refc_v3.py:1353-1361`) and `v0 = None` at the core — exactly the
registered EGO-ZERO ablation — plus `nav_cmd = None` (the registered nav-ZERO arm).
**Nav map** (NavSim one-hot index → `refb.NAV_COMMANDS`): 0 left → `left`, 1 straight →
`follow`, 2 right → `right`, 3 unknown → `follow` (the trainer's `unlabeled_default`). The index
order is verified empirically on the logs before use (control K9). ⚠️ Semantics differ and are
stated, not hidden: refcv4b's token is a per-clip junction TURN (`NAV_TURN_L/R`) vs NavSim's
route-point-20-m-ahead (`y ≥ ±2 m`, which also fires on lane changes and curves).

**Time construction** (refcv4b reads W = 8 rows × a 3-frame channel stack at 10 Hz = raw frames
t0−0.9 … t0; NavSim gives 4 frames at 2 Hz):
* **ST (primary)** — every raw slot ← the t0 frame. Each row is an in-distribution stack (a
  stationary camera); no motion is fabricated; the current view is exact.
* **NT (declared sensitivity)** — every raw slot ← the NavSim frame nearest in time (slots
  −0.9,−0.8 ← −1.0 s; −0.7…−0.3 ← −0.5 s; −0.2…0 ← t0). Uses real history, at 5× the
  training frame spacing at each jump.
* **BLIND** — every pixel = one constant grey (the bank's mean pixel value, rounded): the
  registered deliberate regression; reads no pixels, so it is the only arm that also runs stage 1.

**Frames:** the DataFlyWheel `wide` variant (256×640 cylindrical, `PHYSICALAI_WIDE120_256x640`,
f_ref 305.5775 — the frame refcv4b trained on, `config.json` `image_hw [256, 640]`), sha256[:16]
of every scene verified against `frames_provenance.parquet` before use, unobserved pixels black
(89.29 % observed mean), **never resized**. `rig_clean` (176×624) is NOT the model's frame.

**Output conversion:** a C² cubic spline in time ('not-a-knot') through the origin and the 8
knots (0.5, 1, 1.5, 2, 3, 4, 5, 6 s), evaluated at NavSim's 0.5…4.0 s (8 poses): knots
reproduced exactly; **2.5 s and 3.5 s interpolated**; heading = the spline tangent (the rear-axle
path tangent = kinematic-bicycle yaw at the rear axle), held where |p′| < 1.0 m/s (see §9). Axes need no
permutation (both x-fwd / y-left / CCW).

**Stage-1 stand-in** (camera arms only): the 16 stage-1 tokens are answered by the devkit's own
`ConstantVelocityAgent` so the official script runs end to end. Those rows, the stage-2 kernel
weights they induce, and the script's `extended_pdm_score_*` summary rows are **HYBRID** and are
never reported as any refcv4b arm's number.

## 4. The bar (committed, both outcomes)

**BAR-E2-1:** `S2-EPDMS-u(A1) > S2-EPDMS-u(CV)` on the identical 204 stage-2 tokens.
Reported with the paired per-scene deltas (win / tie / loss counts, per-log means). **No CI:**
7 logs < the RG-14 floor of 8 (`estimator.interval = UNAVAILABLE`).
* **PASS** → "refcv4b zero-shot beats constant velocity on NavSim warmup STAGE-2 scenes (uniform
  weighting, 204 scenes / 7 logs, no interval)". NOT a two-stage EPDMS claim, NOT closed loop.
* **FAIL** → reported as FAILED, then (Rule Zero) decomposed per sub-metric and stage, and the
  pre-declared lever ladder below is walked in the same run while levers are cheap.

⚠️ **Honest prior.** refcv4b only TIES hold-action at T1 on its own corpus (registry §4.6) and
this is zero-shot (PhysicalAI → nuPlan synthetic renders, different cameras, 2 Hz history). CV is
not a weak floor on NavSim's short horizon. I expect FAIL more likely than PASS.

**Pre-declared lever ladder if A1 fails** (cheapest first; every result reported whichever way it
falls, none may replace A1 as the bar arm without its own pre-registration):
1. **A4 vs A1** — is vision helping or hurting zero-shot? (frames-blind is already an arm)
2. **A1NT vs A1** — is the static-history construction the problem?
3. **A3 vs A1** — does the command mapping hurt (semantics differ)?
4. The dominant failing sub-metric names the next input-construction fix, if one exists.

## 5. Secondary reads (reported, not decisive)

* **A1 − A2** = the paired ego-status + command contribution (the Lab's recommendation 3;
  `…/Opponent Analysis/Research/2026-08-28-navsim-v2-camera-lane/RESULT.md`).
* A1 − A3 (command), A1 − A4 (vision), A1 − A1NT and A2 − A2NT (time construction).
* A4's stage-1 rows are its own (no pixels needed): its official two-stage `score` rows are
  therefore a real two-stage EPDMS **of the frames-blind diagnostic arm only**, reported as such.
* CV's own official two-stage EPDMS (stage-1 endpoints its own) for context.

## 6. Controls — each must read a known value, or the arm's numbers are void

| id | control | pass condition |
|---|---|---|
| K1 | frame integrity | 204/204 bank sha256[:16] == provenance; shape u8 (4,256,640,3); mean px ≥ 1 |
| K2 | GT round trip | logged human futures (devkit `Scene.get_future_trajectory`) at the knot times → the SAME `knots_to_navsim` → knot poses reproduce the human poses **< 1e-3 m**; 2.5/3.5 s interpolation error and heading error REPORTED (not a pass/fail) |
| K3 | frame/lateral sign | `taniteval/adapters/navsim.py::verify_frame` (axis + y-sign guards) on the logged human futures: status OK with ≥ 8 turning windows |
| K4 | seam transparency | a seam carrying the devkit CV's own poses for all 220 tokens, scored through the seam agent, reproduces the official `constant_velocity_agent` run's per-token rows **exactly** |
| K5 | ego mutation (undeclared) | every UNDECLARED EgoStatus field randomised → A1/A3 trajectories **byte-identical**; A2 byte-identical under randomisation of ALL ego fields |
| K6 | mutation can fail | a DECLARED field mutated (v0 × 1.5) → A1 trajectories CHANGE on ≥ 90 % of scenes |
| K7 | packer == trainer | on a real PhysicalAI window (`v2ep-eval6-256x640cyl-REF`), my 10-raw-frame packing is byte-identical to the trainer's own `V3Dataset` window and gives the identical `traj` |
| K8 | empty-set guard | per arm: log `successful == 220`, `failed == 0`, CSV 220 valid token rows, agent call log = seam declaration (204 seam + 16 stand-in, or 220 for A4/K4) |
| K9 | command order | on the logged windows the NavSim argmax 0/2 agree in sign with the human path's lateral displacement at 4 s (left ⇒ +y) on ≥ 80 % of turning windows |

A failed K1/K4/K5/K7/K8 voids the affected arms' numbers (a harness defect outranks every result).

## 7. Stamps

**Tier T1-family, loop OPEN for stage 1** (the agent is queried once; the plan is executed by an
LQR + kinematic bicycle; PI ruling 2026-09-02). Stage 2 = the same single query on a PRE-RENDERED
perturbed start: its loop status is **UNRULED** under the 2026-09-02 vocabulary. ⚠️ Background
vehicles in the two-stage scorer are **IDM-REACTIVE**, not log-replay: `run_pdm_score.py:77-79,
132-134` instantiates `cfg.traffic_agents_policy.reactive` (= `NavsimIDMTrafficAgents`) for BOTH
stages. ⛔ Never called closed loop. Estimator: `{status: UNAVAILABLE, reason: 7 logs < RG-14
floor 8, n}` — warmup never carries a CI. Route leak: `UNVERIFIED` (stream E3). Protocol label:
`EPDMS_v2_warmup_two_stage` — **stage-2 rows only** for the camera arms.

## 8. Four metric families

Our four-family instruments need the model's path AND a GT future on the SAME scene. Stage-2
scenes carry no GT future (`num_future_frames = 0`; synthetic starts have no human who drove
them); stage-1 scenes carry one but have no frames. ⇒ for the camera arms every family is
**UNAVAILABLE with reason + n = 0**, emitted per family through the adapter. A4 (frames-blind)
runs stage 1, so its families are computable on n = 16 (reported as the blind arm's, never A1's).
Unblock: the stage-1 original camera frames (§0).

## 9. Amendment 1 — made BEFORE any scoring run (2026-09-19)

A 2-scene PIPELINE smoke (A1 + A2 on the first two stage-2 tokens, trajectories only — **no
scorer was run and no score exists**) showed that the heading rule in §3 (tangent held below
0.1 m/s) turns the cm-level slot-offset noise of a near-stopped plan into heading swings of
±0.7 rad, and that `np.unwrap` over held values could drift to −6.5 rad. Amended: the tangent is
held below **1.0 m/s** (a 0.5 s interval then moves ≥ 0.5 m, bounding a 5 cm jitter's tangent
error near 0.1 rad), and each new heading takes the branch nearest the previous one (no multi-turn
drift). Knot reproduction (K2's pass condition) is unaffected — the spline is unchanged; only the
heading column moved. The same smoke also showed A2 (ST frames, v0 withheld) emitting a
near-standstill plan on an 11.3 m/s scene — recorded here as an OBSERVATION, not a reason to
change A2, whose definition stands as registered.
