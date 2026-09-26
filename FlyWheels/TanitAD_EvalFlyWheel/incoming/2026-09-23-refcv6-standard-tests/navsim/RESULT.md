# RESULT — refcv6 on the NavSim suite (warmup_two_stage · navhard_two_stage · navtest v1.1)

**Stream** EvalFlyWheel (run from the Master Mind session) · **2026-09-23/24, resumed 2026-09-26** ·
**Model** refcv6-r101-s0 (training on Thor; step 32,561 at 2026-09-26T07:51Z; ends ≈ 2026-09-27 17:30Z)
· **Checkpoints**: the kit's `ckpt_step1000.pt` (md5 `7c3ad3c1…`) = ⛔ **PIPELINE VALIDATION ONLY**
(SPEC §6); `ckpt_5000.pt` (md5 `8a1e4da0dc6561353b28e3947146751e`) = the first **RESULT** reading;
`ckpt_30000.pt` and the FINAL `ckpt.pt` = running / armed (§9) · **Pre-registered** `SPEC.md`, git
blob `ab7f769a…`, staged 2026-09-23T22:08:44Z before any refcv6 score; amendments A1–A4 each staged
before the scores they touch (`raw/SPEC_PREREG_HASH.txt`).

Every number is **MEASURED by this stream** unless marked; its artifact is named beside it. Stamps on
every NavSim row: tier **T1-family**; stage-1 loop **OPEN**; stage 2 **UNRULED**; v2 background
**IDM-reactive**, v1 **non-reactive (logged)**; ⛔ never closed loop; **zero-shot** (PhysicalAI-AV B1 →
nuPlan cameras, 3-camera stitch); the **device + precision** of every refcv6 row is stated beside it
(amendment A3: one device per split).

## 0. Headline (status 2026-09-26 ~10:00Z — updated as checkpoints land)

**PROVEN.** (1) The NavSim harness still reads its known values on ALL THREE splits through this
package's own drivers — warmup CV/STOP/ECHO vs E2, navhard CV vs W7 (KH-nav), navtest STOP vs W3
(KH-navtest) — every cell |Δ| 0.0. (2) refcv6's NavSim inputs are explicit functions with literal /
analytic tests and regression arms that go RED (68 tests + KL bit-exact against the trainer-exact
loader). (3) **refcv6 @ step 5,000 — the first RESULT-grade NavSim reading — FAILS all three committed
bars: navhard official two-stage EPDMS 0.1512 vs STOP 0.2985 (paired −0.147 [−0.179, −0.114], 17×
the seed floor), navtest PDMS 46.48 vs STOP 61.82 (paired −15.3 [−18.3, −12.8]), and warmup
S2-EPDMS-u 0.3966 vs STOP 0.5212 (−0.125; seed floor 0.016).** It beats constant velocity on navhard
(+0.036 [+0.005, +0.074]) and navtest (+25.8), ties it on warmup, ties ECHO on navhard; it never
stops. Vision is the one input lever beyond inference noise (warmup A1 − BLIND +0.032). (4) **The measured lever is LONGITUDINAL**: the drivable-area multiplier carries the largest
single-term ceiling (navhard stage 2 +0.170, navtest +11.1), and on navtest the deficit sits at 2–5
m/s, where the selected plan travels 1.21× the human's 4 s distance and 55.7 % of the plans that
overshoot the human by > 2 m are at-fault collisions — speed bias +0.68 m/s (four families).
(5) The max-speed input is INERT at step 5,000 (200/200 navtest plans bit-identical with it
withheld; its oracle definition moves 8/200, inside the seed floor) — the ARGMAX ceiling filter the
run's config declares is not built (§6.7).

**NOT PROVEN / NOT YET MEASURED.** The mid-run (30,000) and FINAL readings — running / armed,
unattended (§9). The
precision floor KP (CPU fp32 vs CUDA bf16) — waits for the GPU gate, which another session holds.
Nothing here is closed loop, and every reading is zero-shot (PhysicalAI-AV → nuPlan cameras; the
NavSim camera sits ~0.57 m higher than the training rigs).

**Next lever (Rule Zero, named — model-side, therefore the Master Mind's / Arch's):** lead-aware
longitudinal behaviour at 2–8 m/s (distance keeping), then drivable-area compliance on turns
(navhard RIGHT: DAC0 55 %). The eval-side zero-training lever — switching on the declared
parameter-free speed-ceiling filter at eval — would NOT reach it (the collisions happen far below
any posted limit) and changes the evaluated model, so it is a PI / Arch decision, not taken here.

## 1. Deliverable 1 — the harness still reads its known values: ✅ REPRODUCED, bit for bit

E2's two model-free controls re-scored on warmup_two_stage through THIS package's driver
(`code/score_arm6.py`: E1's current wrapper `navsim_win.py` sha256 `97cf493a…` with the loader patch,
E2's seam agent imported, `PYTHONHASHSEED=1`, the C: devkit `0a380a9` + E1's patches):

| control | S2-EPDMS-u (mine = E2) | official stage 1 / stage 2 / two-stage | per-token cells |
|---|---|---|---|
| CV_official | **0.39713167136274696** = 0.39713167136274696 | 0.4602892129691729 / 0.3341286472565139 / **0.1853562745165113** | 220 tokens × every column, **max \|Δ\| = 0.0** |
| STOP_zero | **0.5212469877807624** = 0.5212469877807624 | 0.5777242796695368 / 0.5212469877807625 / **0.3009023137456225** | 220 × every column, **max \|Δ\| = 0.0** |

Count guards PASS (220 successful / 0 failed / 220 valid rows; STOP's seam: 220 agent calls). Rule
committed before the comparison: cell |Δ| ≤ 1e-9, statistics ≤ 1e-12 → **HARNESS_REPRODUCED**
(`raw/HARNESS_REPRO.json`, `raw/harness_repro/`). ⚠️ The wrapper differs from the copy E2's STOP run
used (`c40d18e1…`) only in the RAM guard's sustain (3 → 60 samples) and a pre-aggregation dump; the
bit-identity above is the evidence that neither changed a score. The C: devkit also carries E1's
IDM degenerate-path patch (2026-09-20); inert on warmup by the same evidence.
**navhard floors**: W7's banked model-free runs re-read through this package's statistics reproduce
their published values exactly — CV **0.11481608441648** [0.0825, 0.1450], STOP **0.2985**
[0.2737, 0.3253], ECHO **0.1429** [0.1168, 0.1698], 76 log clusters, 225 mapping keys, each
`reproduces_official: OK` (`raw/floors/navhard_two_stage/floors_summary_check.json`).

**The harness controls, per split, through THIS package's own drivers:**

| split | control | result |
|---|---|---|
| warmup | CV, STOP re-scored vs E2 | ✅ REPRODUCED, every cell max \|Δ\| 0.0 (above) |
| warmup | ECHO re-scored vs E2 | ✅ REPRODUCED, S2-EPDMS-u 0.4287471809626652 = E2, every cell max \|Δ\| 0.0 (`raw/controls/KH_ECHO_warmup.json`) |
| navhard | **KH-nav**: CV re-scored (5,912 tokens) vs W7's banked CV | ✅ REPRODUCED, 20 columns × 5,912 rows max \|Δ\| 0.0; official two-stage EPDMS 0.11481608441648 = W7 (\|Δ\| 0.0) (`raw/floors/navhard_two_stage/FLOORS_PROVENANCE.json`) — completed unattended 2026-09-24T02:56Z after two E1 RAM-guard aborts |
| navtest | **KH-navtest**: STOP re-scored (12,146 tokens) through `score_navtest6.py --official STOP` vs W3's banked STOP | ✅ REPRODUCED 2026-09-26: 12,146 tokens × 8 numeric columns max \|Δ\| 0.0; PDMS 61.82023 = W3; count guard PASS 12,146 / 0 / 12,146 (`raw/controls/KH_navtest_STOP.json`, `raw/harness_repro_navtest/`) |

## 2. Deliverable 2 — refcv6's inputs from NavSim, each an explicit, tested function

| input | construction (code) | test (arm that must go RED) |
|---|---|---|
| **frames** | `frames416.py`: E2's verbatim 3-camera stitch (CAM_L0/F0/R0 → one virtual camera at CAM_F0's boresight, rolled level; pinhole + Brown distortion per camera; largest-cos camera wins; unseen rays black) re-pointed at `trunk_shapes.FRAME_416x1024` = 416 × 1024, `f_ref` 488.92398517830253, **CYLINDRICAL** (column linear in azimuth, hfov 120.0000°, vfov 46.0921°) — the frame recorded in the corpus's own v2ep files (`frame: {416, 1024, 488.92398517830253, cylindrical}`, read from the eval-139 view). Pure-rotation reprojection. Observed fraction 0.880 (E2's 256×640: 0.893) | KG-416 azimuth-linear (8.5e-8 rad, float32 table); **pinhole arm RED** (92.6° field, non-linear); KB-256 re-pointing reproduces the DataFlyWheel bank **bit-exactly** (6/6 + 3/3); KT tamper REFUSED |
| **history timing** | refcv6 reads 10 raw frames at **10 Hz** (8 rows × 3-frame D-015 stacks, t0−0.9 … t0). MEASURED on the eval-139 corpus: dt = displacement / speed = **0.1007 s**, 201 frames per clip, `n_stack` 3; the run's `ego_history` stamp `dt 0.1`. NavSim holds **2 Hz** (t0−1.5, −1.0, −0.5, 0). **ST (primary)**: every slot ← the t0 frame; **NT (sensitivity)**: nearest in time (E2's `slot_sources`, imported) | refusal of non-2 Hz times |
| **nav** | NavSim `driving_command` one-hot → the v7 token row: left→`left`(1), straight→`follow`(0), right→`right`(2), unknown→`follow`(0); E2's `nav_index_from_command` (refuses a non-one-hot). ⚠️ semantics differ (v7: junction turn from the ego future; NavSim: route geometry) | literal map, non-one-hot refused, NAVOFF ignores the command |
| **v0** | `hypot(vx, vy)` at t0 | KI: moves the plan |
| **ego history** | 8 × (x, y, yaw, v) at t0−0.7 … t0, LINEAR in time between the declared 2 Hz states t0−1.0 / −0.5 / 0 (yaw unwrapped); the model's own `ego_channels_from_poses` (backward differences, dt 0.1) then reads each 0.5 s segment's mean accel / yaw rate | constant accel reads **2.0 m/s²** (literal); yaw rate −0.2 rad/s; ±π unwrap; **wrong-grid arm RED** (reads 10 m/s²) |
| **max speed** | the posted limit (`speed_limit_mps`, never the constant-25 `max_speed` placeholder) of the lane the ego occupies at t0 (`export_speed_limits.py`: containing lane within 60° heading, else nearest ≤ 2 m; global pose = the devkit's own `ego2global` + pyquaternion yaw, cross-checked 120/120 exact), fed RAW with `valid = 1`; the model's containing-window ladder {30, 50, 100, 120} km/h is applied ONCE, model-side. **Fallback: `valid = 0` = the all-zero "no ceiling known" one-hot** — ⚠️ OUT OF DISTRIBUTION for this run (train sidecar 4,572/4,572 valid). ⛔ The channel was TRAINED on the ego-future ORACLE (max realised speed over [t0+2, t0+6] s, §6.6); the map limit is its deployable stand-in, and SPEC §12 prices the gap with a PRIVILEGED diagnostic arm (`R6_VMAXORACLE`, `code/vmax_oracle.py`) | literal bins (25 mph→50, 35 mph→100, 15 mph→30), unknown = all-zero, VMAXOFF ignores the map |
| **BEV lift geometry** | `rig6.py`: the stitch's virtual camera in refcv6's rig frame (rotation = the stitch's own M; centre = CAM_F0's mount; **height = F0.z − road plane**: NavSim's ego origin sits **0.3557 m above the road** — median vehicle-cuboid bottom over 74/76 logs, per-log −0.50 … −0.15; `lidar2ego` asserted identity) → camera height **~1.88 m**, pitch 1.1–2.3° down (MEASURED per scene: warmup 1.861–1.891 m, navhard 1.844–1.929 m) — ⚠️ **OUTSIDE the training rigs' 1.203–1.687 m (median 1.312 m; run `config.json` `agent_rig_camera`)**: the lift is geometrically right for NavSim, but the trunk sees a viewpoint ~0.57 m higher than its median training rig — part of what "zero-shot" means here; `build_lift_geometry` with the TRAINER's parameters (stride 16, heights 0/0.5/1.5/2.5 m, `GRID_DEFAULT`, bottom 43 rows unobserved) | analytic road-point row (1e-6 px); **road-offset-ignored arm RED** (> 5 rows); non-identity lidar2ego REFUSED |
| **output** | `out["traj"]` 8 knots (0.5, 1, 1.5, 2, 3, 4, 5, 6 s) → E2's `knots_to_navsim` (imported): C² not-a-knot spline in time through the origin, evaluated at 0.5 … 4.0 s; heading = tangent, held below 1 m/s | CV line exact; circle-arc position ≤ 5 cm & heading ≤ 0.01 rad; **wrong-knot-times arm RED** |

**Coverage of the max-speed map input** (per scene; `raw/inputs/speed_limits_*.json`):

| split | LV | PIT | BOS | SG | all |
|---|---|---|---|---|---|
| warmup (220) | — | **136/136** (all 25 mph) | 0/64 | 0/20 | 136/220 = 61.8 % |
| navhard (5,912) | **786/786** | **2,357/2,358** | 72/1,448 (5.0 %) | 0/1,320 | 3,215/5,912 = 54.4 % |
| navtest (12,146) | **4,064/4,064** | **2,437/2,437** | 157/3,731 (4.2 %) | 0/1,914 | 6,658/12,146 = 54.8 % |

**The model itself** (`refcv6_bridge.load_refcv6`): rebuilt by the landed eval's
`taniteval/tools/refcv3_arm.load_model` (trainer parser + `_pin_trainer_cfg`, `param_breakdown`
cross-check, perception branch rebuilt) — **strict load, 0 missing / 0 unexpected**; recorded argv edits:
`--trunk-compile` dropped (no Triton), two rebuild-time Thor files → md5-verified local copies.
⛔ **Plus one fix `refcv3_arm` needs for refcv6 — found by control KL, not by inspection:** the trainer
reads the anchor artifact BEFORE pinning the config, so the file's declared units (`alat`) reach the
decoder; `refcv3_arm` does not, and rebuilt the decoder as `kappa` (a lateral-acceleration vocabulary
rolled as curvature: anchor bank off by up to 108 m). `load_refcv6` replays the trainer's order and
REFUSES a build whose units differ. **KL: against the battery stream's trainer-exact loader, the same
NavSim scene, inputs, lift, precision and seed give BIT-IDENTICAL plans (3/3, max |Δ| 0.0)**
(`raw/controls/KL_loader_crosscheck.json`). Everything produced before the fix is VOID and quarantined
(`raw/VOID_wrong_anchor_units/`; SPEC amendment A2). CPU
fp32 (bf16 is emulated on this CPU: 98 s vs 9 s/scene MEASURED); the exact-duplicate dedup is
**bit-identical** to the trunk's native path (KD: 4/4 scenes, max |Δ| 0.0, same selections; 5× cheaper
on a static window). Per-scene inference seed (common random numbers across arms); **K0: same seed ⇒
bit-identical plan**. **KI: frames, nav, v0 and the ego history each move the plan; the max-speed
one-hot reaches the tactical decoder's logits** (and, at step 1,000, no plan — §3.1). Tests: **68
green** (`tests/`: 59 CPU-light incl. the SPEC §12 oracle with its wrong-window arm RED, re-run
2026-09-24 00:10Z; 9 on the real checkpoint incl. the navtest-bank tamper arm, re-run 01:09Z after
the bridge edits, 831 s) + **KL** (`tests/test_loader_crosscheck.py`, bit-exact vs the battery
stream's trainer-exact loader).

## 3. Deliverable 3 — refcv6 on NavSim with the control set

⛔ **§3.1 and §3.2 are the kit's step-1,000 checkpoint (1,000 of 50,400 steps) = PIPELINE VALIDATION
by SPEC §6 — not refcv6's NavSim performance; no bar is evaluated on them.** They prove the path.
**§3.3 onward are RESULT readings** (milestone checkpoints, SPEC §5 bars evaluated as written).

### 3.1 warmup_two_stage @ step 1,000 — the whole path, every arm, every control (PIPELINE VALIDATION)

**Bridge** (`raw/bridge_warmup_s1000/`): 6 arms × 220 tokens, CPU fp32, exact dedup, per-scene
common random numbers; 204 refcv6 rows + 16 declared CV stand-ins per camera arm (warmup stage 1
has no original frames on this box), 220 own rows for R6_BLIND; 0 missing. **Scorer**
(`raw/scores_warmup_s1000/`): every arm 220 successful / 0 failed / 220 valid rows, agent calls ==
seam declaration (count guards PASS). **Controls read their known values on this very run:**
ECHO re-scored through this package reproduces E2's banked ECHO **bit for bit** (S2-EPDMS-u
0.4287471809626652 = 0.4287471809626652, every cell max |Δ| 0.0, all three official rows 0.0;
`raw/controls/KH_ECHO_warmup.json`) — the third model-free control after CV and STOP (§1); and two
seams with bit-identical plans (R6_A1, R6_VMAXOFF — see below) score **cell-identically** (max |Δ|
0.0; `raw/controls/KDET_scorer_identical_seams_warmup_s1000.json`).

| arm (step 1,000) | S2-EPDMS-u | official two-stage | stop frac | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R6_A1 | 0.3477 | n/a (stage-1 stand-in) | 0.010 | 0.679 | 0.627 | 0.738 | 0.941 | 0.850 | 0.662 | 0.578 | 0.995 | 0.627 |
| R6_A1_s1 (inference seed 1) | 0.3628 | n/a | 0.005 | 0.686 | 0.657 | 0.770 | 0.941 | 0.853 | 0.667 | 0.574 | 0.995 | 0.608 |
| R6_A1NT (nearest-time frames) | 0.3451 | n/a | 0.010 | 0.689 | 0.627 | 0.740 | 0.941 | 0.850 | 0.676 | 0.574 | 0.995 | 0.637 |
| R6_BLIND (frames-blind) | 0.3208 | 0.1007 | 0.255 | 0.652 | 0.564 | 0.632 | 0.961 | 0.676 | 0.657 | 0.480 | 0.926 | 0.618 |
| R6_NAVOFF | 0.3427 | n/a | 0.005 | 0.694 | 0.618 | 0.725 | 0.941 | 0.847 | 0.667 | 0.569 | 0.995 | 0.637 |
| R6_VMAXOFF | 0.3477 | n/a | 0.010 | 0.679 | 0.627 | 0.738 | 0.941 | 0.850 | 0.662 | 0.578 | 0.995 | 0.627 |
| CV_official (floor) | 0.3971 | 0.1854 | — | 0.681 | 0.716 | 0.811 | 0.946 | 0.718 | 0.681 | 0.564 | 0.995 | 0.588 |
| STOP_zero (floor) | **0.5212** | 0.3009 | 1.000 | 0.926 | 0.926 | 0.983 | 0.980 | 0.359 | 0.926 | 0.593 | 0.662 | 0.255 |
| ECHO_ha0_ext (floor) | 0.4287 | 0.2248 | — | 0.765 | 0.686 | 0.777 | 0.961 | 0.584 | 0.755 | 0.529 | 0.995 | 0.853 |

**Device on every refcv6 row: CPU fp32** (read from each seam's own manifest into
`raw/summary_warmup_s1000.json`; the GPU gate never opened, §7); floors: model-free, devkit scorer.
S2-EPDMS-u over the identical 204 stage-2 tokens (E2 SPEC §2); sub-metrics = stage-2 means. ⛔ No
interval on warmup (7 logs < the 8-cluster floor). Context, not a comparison: E2's fully trained
refcv4b A1 read 0.4670 on the same tokens (E2 `raw/scores_summary.json`).

| pair (step 1,000) | ΔS2-EPDMS-u | W/T/L (scenes) | beyond the seed floor (0.0151)? |
|---|---|---|---|
| A1 − CV | −0.0494 | 69/98/37 | yes |
| A1 − STOP | −0.1735 | 71/33/100 | yes |
| A1 − ECHO | −0.0810 | 71/77/56 | yes |
| **seed floor** A1 − A1_s1 | −0.0151 | 38/125/41 | — |
| vision: A1 − BLIND | +0.0269 | 65/102/37 | **yes** |
| nav: A1 − NAVOFF | +0.0050 | 47/131/26 | no |
| max speed: A1 − VMAXOFF | **0.0000** | **0/204/0** | no — identical plans |
| time construction: A1 − A1NT | +0.0026 | 34/135/35 | no |

**The same levers at the PLAN level, before any scorer** (`raw/controls/plan_deltas_warmup_s1000.json`;
4 s endpoint distance to R6_A1's plan, same scenes, same seeds): seed replicate median **0.653 m**
(p90 2.80; same anchor 67.6 %) · BLIND 5.75 m (same anchor 10.8 %) · NAVOFF 0.17 m (89.7 %) ·
A1NT 0.06 m (97.1 %) · **VMAXOFF 0.000 m — 204/204 plans bit-identical, including the 126 scenes
that have a map limit** (§6.7: the max-speed one-hot reaches the tactical logits but no selection,
and the parameter-free ceiling filter is not built). Of the four levers only VISION moves the plan
beyond inference noise at step 1,000.

**The SPEC §5 ladder, run anyway** (`raw/decomposition_warmup_s1000.json`, W7's `decompose.py`):
118/204 stage-2 scenes score 0; the ATTRIBUTABLE (unique) zeros are DAC 33, NC 18, TLC 7, DDC 1
(co-occurring: DAC 76, NC 64, DDC 43, TLC 12); the single-term ceilings rank **DAC +0.124** ≫ NC
+0.053 > EP +0.033 — at step 1,000 the model's dominant failure is leaving the drivable area, while
it already makes MORE progress than CV (EP 0.850 vs 0.718). By command: STRAIGHT (193 scenes) 0.324
vs STOP 0.498; LEFT (11) 0.473 vs 0.659.

**What this proves and does not.** Proven: the whole warmup path — 416 frames, inputs, lift, model,
seam, official scorer, count guards, statistics, ladder — runs end to end on the corrected build,
reads the three model-free controls bit-exactly, and is deterministic where it must be (K0; K-DET).
**Not** proven: anything about refcv6's driving — step 1,000 sits below CV, ECHO and STOP, as a
2 %-trained model is expected to. Four-family metrics: warmup stage 2 has no GT future ⇒
LONGITUDINAL / LATERAL / TACTICAL **UNAVAILABLE (n = 0, synthetic starts)**; STRATEGIC not
applicable (strategic layer off; NavSim scores no strategic decision) — SPEC §8.

### 3.2 navhard_two_stage (step 1,000, R6_A1 on all 5,912 tokens) — PIPELINE VALIDATION

Completed unattended 2026-09-24 (bridge 5,912/5,912 own rows, **0 stand-ins**, 0 missing — the 450
stage-1 scenes run on their ORIGINAL frames; CPU fp32; scorer 5,912 successful / 0 failed / 5,912
valid rows, seam calls 5,912; reviewed 2026-09-26). Official two-stage EPDMS **0.0752** [0.0567,
0.0937] vs CV 0.1148, ECHO 0.1429, STOP 0.2985 — every paired log-cluster interval below zero
(`raw/summary_navhard_s1000.json`, `raw/tables_navhard_s1000.txt`). ⛔ Pipeline validation: it proves
the defined camera-arm two-stage path end to end, nothing about driving. The navhard **harness
control KH-nav passed**: this package's CV re-score equals W7's banked CV cell for cell (max |Δ| 0.0
over 20 columns × 5,912 rows) and to the digit on the official EPDMS (0.11481608441648, |Δ| 0.0)
(`raw/floors/navhard_two_stage/FLOORS_PROVENANCE.json → _KH_nav_check`).

### 3.3 Step 5,000 — the FIRST RESULT-GRADE NavSim reading of refcv6 (verified 2026-09-26)

`ckpt_5000.pt`, md5 `8a1e4da0dc6561353b28e3947146751e` (Thor `md5sum` == dev box, fetched read-only
2026-09-24T04:11Z; `raw/milestones/runner.log`). Run UNATTENDED 2026-09-24 by `milestone_waiter.sh`;
**reviewed 2026-09-26 before any number entered this file**: count guards, stand-ins, device and
precision stamps (amendment A3), the seed replicate, and the harness control on the same split —
each below. Stamps: tier **T1-family**; stage-1 loop **OPEN**; stage 2 **UNRULED**; v2 background
**IDM-reactive**, v1 **non-reactive (logged)**; ⛔ never closed loop; **zero-shot**.

**navhard_two_stage — device CUDA, precision as trained (bf16 + NHWC trunk), BOTH arms.** CUDA
controls re-measured first: **K0 PASS** (same seed ⇒ bit-identical plan on CUDA); **KD FAIL**
(exact dedup vs native: same selections 4/4, plans differ by 2.16 cm — bf16 kernels depend on the
batch composition) ⇒ the runner dropped the dedup lever and ran the trunk's native path, as designed
(`raw/milestones/step5000/KD_exact_dedup_cuda.json`, `cuda_controls.log`). Bridge: both arms
5,912/5,912 own rows, 0 stand-ins, 0 missing; scorer: both arms 5,912 / 0 / 5,912, seam calls 5,912.

| navhard @ 5,000 | device | official two-stage EPDMS [log-cluster 95 %] | S2-EPDMS-u | stop frac | NC | DAC | DDC | EP | TTC |
|---|---|---|---|---|---|---|---|---|---|
| **R6_A1** | CUDA as-trained | **0.1512** [0.1214, 0.1842] | 0.3506 | 0.000 | 0.820 | 0.630 | 0.802 | 0.821 | 0.792 |
| R6_A1_s1 (inference seed 1) | CUDA as-trained | 0.1427 [0.1152, 0.1704] | 0.3479 | 0.000 | 0.818 | 0.628 | 0.802 | 0.821 | 0.789 |
| CV (floor, W7) | model-free | 0.1148 [0.0825, 0.1450] | 0.3389 | — | 0.834 | 0.569 | 0.735 | 0.701 | 0.815 |
| ECHO (floor, W7) | model-free | 0.1429 [0.1168, 0.1698] | 0.3413 | — | 0.818 | 0.534 | 0.679 | 0.646 | 0.791 |
| **STOP (floor, W7)** | model-free | **0.2985** [0.2737, 0.3253] | 0.4828 | 1.000 | 0.956 | 0.863 | 0.951 | 0.384 | 0.944 |

| paired (same 5,912 tokens, 76 log clusters, B 2000) | Δ official EPDMS | paired 95 % interval | W/T/L (scenes) |
|---|---|---|---|
| A1 − CV | **+0.0364** | [+0.0047, +0.0742] — separated | 1,884 / 2,598 / 980 |
| A1 − ECHO | +0.0083 | [−0.0220, +0.0420] — not separated | 1,970 / 2,325 / 1,167 |
| **A1 − STOP** | **−0.1473** | [−0.1794, −0.1137] — separated, below | 2,112 / 1,017 / 2,333 |
| **seed floor** A1 − A1_s1 | +0.0086 | [−0.0096, +0.0283] | 1,169 / 3,197 / 1,096 |

Sub-metrics are stage-2 means (S2); the official aggregate is the devkit's `extended_pdm_score_combined`
(`raw/milestones/step5000/summary_navhard.json`; estimator `paired_navsim_log_cluster_bootstrap`,
`taniteval/adapters/navsim_ci.py`, question: "another draw of LOGS?" — the seed floor is read
separately). Plan level: the seed replicate keeps the same anchor on 81.8 % of scenes, 4 s endpoint
median 0.50 m (`plan_deltas_navhard.json`).

**BAR-R6-H1 (committed, SPEC §5): FAILED.** refcv6 @ 5,000 beats constant velocity and ties the
model-free ECHO, but a do-nothing plan beats it by 0.147 EPDMS with an interval that excludes zero,
17× the seed floor. It **never stops** (stop fraction 0.000 vs STOP's 1.000). ⇒ SPEC §5 FAIL branch —
decomposed below, next lever named in §0.

**navtest (v1.1) — device CPU, precision fp32, exact dedup (KD bit-identical on CPU).** Bridge
12,146/12,146 own rows, 0 stand-ins; scorer 12,146 / 0 / 12,146, seam calls 12,146. Harness control
on this split: **KH-navtest REPRODUCED** (W3's STOP re-scored through this driver, every cell |Δ| 0.0,
§1). Seed floor on navtest (200-token diagnostic, same device): A1 − A1_s1 +1.26 PDMS [−3.17, +5.78] (§4).

| navtest @ 5,000 (all 12,146 tokens, 136 logs) | device | PDMS ×100 [log-cluster 95 %] | NC | DAC | EP | TTC | C | zeroed (NC or DAC) |
|---|---|---|---|---|---|---|---|---|
| **R6_A1** | CPU fp32 | **46.48** [44.30, 48.36] | 75.51 | 73.25 | 47.26 | 61.18 | 99.98 | 5,574 |
| CV (W3 banked) | model-free | 20.65 [19.20, 22.22] | 68.02 | 57.84 | 19.44 | 50.03 | 100.00 | 8,130 |
| **STOP (W3 banked)** | model-free | **61.82** [60.70, 63.08] | 97.40 | 96.53 | 31.00 | 96.40 | 69.44 | 732 |
| refcv4b A1 (W3, context) | CPU | 58.97 | 91.35 | 73.46 | 52.37 | 82.54 | 98.02 | 3,828 |
| HUMAN (W3, log replay) | — | 94.55 | 100.00 | 100.00 | 86.96 | 100.00 | 99.90 | 0 |

Paired on the same 12,146 tokens: **A1 − STOP −15.34** [−18.29, −12.77] (W/T/L 5,594 / 484 / 6,068);
A1 − CV **+25.83** [+23.44, +28.10]; A1 − refcv4b −12.49 [−16.11, −9.11]
(`raw/milestones/step5000/summary_navtest.json`; W2's registered log-cluster bootstrap; drive-level
sensitivity also separated). **BAR-R6-T1: FAILED.** Stretch references (not bars): Ego-Status MLP
65.6 / 66.4, DiffusionDrive 88.1 (camera + LiDAR, in-domain; §4).

**Where it loses — the SPEC §5 ladder (run automatically, W7's `decompose.py` imported).**
* **navhard:** 2,896/5,462 stage-2 and 234/450 stage-1 scenes score 0; the ATTRIBUTABLE (unique)
  zeros are **DAC 1,374** (stage 2) and **153** (stage 1), then NC 494 / 32, DDC 263 / 8; single-term
  ceilings rank **DAC +0.170** (stage 2) / **+0.263** (stage 1) ≫ DDC +0.057, NC +0.053. By command
  (stage 2, vs STOP): RIGHT −0.209 (DAC0 55 %), LEFT −0.111, STRAIGHT −0.080.
* **navtest:** 5,574/12,146 tokens zeroed — unique zeros DAC 2,790, NC 2,325; single-term ceilings
  **DAC +11.12**, TTC +4.38, NC +3.43 PDMS points (formula self-check vs the devkit `score`: 3.3e-16).
  By v0 band the deficit vs STOP is concentrated at **2–5 m/s (−38.4; NC0 37.0 %)** and 5–8 m/s
  (−21.0); refcv6 BEATS STOP at 0–2 (+2.9), 8–12 (+3.9) and ≥ 12 m/s (+15.8)
  (`decomposition_navtest.json`).
* **The mechanism behind the NC zeros, measured without a model or a scorer** (plan vs the logged
  human future, `raw/milestones/step5000/travel_vs_human_navtest.json`): at 2–5 m/s the selected plan
  travels a median **1.21×** the human's 4 s distance; on the 2,047 of 3,458 tokens where it travels
  > 2 m further than the human, **55.7 % are at-fault collisions**. The model keeps its speed where
  the human was slowing behind traffic — a **longitudinal** (distance-keeping) failure, the same
  family CLAUDE.md measured as 88.7 % of the programme's oracle gap. The FOUR FAMILIES agree:
  speed bias **+0.68 m/s** (navtest) / +0.62 (navhard stage 1), i.e. faster than the human.
  **navhard shows the same band** (W7's speed bands vs STOP, `decomposition_navhard.json`): the
  deficit peaks at **2–5 m/s** (stage 2 −0.191, n = 1,747; stage 1 −0.184, n = 128) and turns into
  a WIN at ≥ 12 m/s on stage 2 (+0.180, n = 144) — two splits, two harnesses, one mechanism.

**The four metric families (SPEC §8)** — where a logged human future exists: navhard stage 1
(n = 450) and navtest (n = 12,146); refcv6 vs CV on the same windows (`families_navhard.json`,
`families_navtest.json`; episode-cluster intervals in the files):

| family (step 5,000) | navhard stage 1: refcv6 / CV | navtest: refcv6 / CV |
|---|---|---|
| LONGITUDINAL speed MAE (m/s) · bias | 0.86 · +0.62 / 1.02 · +0.16 | 0.95 · +0.68 / 1.28 · +0.18 |
| LONGITUDINAL speed within 1 m/s | 71.5 % / 64.4 % | 67.7 % / 54.8 % |
| LATERAL heading MAE (°) · curvature MAE (1/m) · cross-track MAE (m) | 7.70 · 0.0223 · 0.92 / 11.88 · 0.0275 · 1.49 | 5.26 · 0.0164 · 0.67 / 8.48 · 0.0200 · 1.08 |
| TACTICAL lateral decision acc · κ | 0.724 · 0.564 / 0.420 · 0.0 | 0.829 · 0.664 / 0.619 · 0.0 |
| TACTICAL longitudinal decision acc · κ | 0.569 · 0.313 / 0.413 · 0.0 | 0.534 · 0.309 / 0.329 · 0.0 |
| TACTICAL goal point error (m) · bearing MAE (°) | 4.80 · 7.96 / 6.32 · 12.28 | 4.62 · 5.20 / 6.75 · 8.08 |
| STRATEGIC | UNAVAILABLE (strategic layer off; NavSim scores no strategic decision), n = 450 | UNAVAILABLE, n = 12,146 |

Warmup and navhard stage 2 carry no GT future (synthetic starts): LONGITUDINAL / LATERAL / TACTICAL
UNAVAILABLE there, n = 0.

**warmup_two_stage @ 5,000 — device CPU, precision fp32, exact dedup, all six arms.** The unattended
run of 09-24 left this split EMPTY (the runner's GPU gate passed at 04:11Z, the bridge's own gate then
waited 900 s while a sibling held the card and refused; nothing fell back). Fixed 2026-09-26 in the
runner (a refused CUDA bridge that wrote no row re-runs on CPU) and **re-run 2026-09-26T08:18–09:52Z**:
every arm 220 / 0 / 220 with seam calls == declaration (204 refcv6 + 16 declared CV stand-ins on the
camera arms — warmup stage 1 has no original frames here; R6_BLIND 220 own); one E1 RAM-guard abort
(R6_A1_s1, another session's two full-suite pytest runs held ~9.5 GB) retried and PASSED. Harness
control on this split: CV / STOP / ECHO reproduced bit-exactly (§1).

| warmup @ 5,000 | device | S2-EPDMS-u | stop frac | NC | DAC | DDC | EP | TTC |
|---|---|---|---|---|---|---|---|---|
| **R6_A1** | CPU fp32 | **0.3966** | 0.000 | 0.755 | 0.691 | 0.846 | 0.797 | 0.716 |
| R6_A1_s1 (seed 1) | CPU fp32 | 0.4123 | 0.000 | 0.760 | 0.706 | 0.853 | 0.802 | 0.721 |
| R6_BLIND (frames-blind; own stage 1: official 0.1191) | CPU fp32 | 0.3645 | 0.000 | 0.657 | 0.662 | 0.784 | 0.773 | 0.647 |
| R6_NAVOFF | CPU fp32 | 0.4004 | 0.000 | 0.750 | 0.676 | 0.846 | 0.799 | 0.711 |
| R6_VMAXOFF | CPU fp32 | 0.3980 | 0.000 | 0.755 | 0.696 | 0.848 | 0.798 | 0.716 |
| R6_A1NT | CPU fp32 | 0.3938 | 0.000 | 0.755 | 0.686 | 0.843 | 0.797 | 0.716 |
| CV (floor) | model-free | 0.3971 | — | 0.681 | 0.716 | 0.811 | 0.718 | 0.681 |
| **STOP (floor)** | model-free | **0.5212** | 1.000 | 0.926 | 0.926 | 0.983 | 0.359 | 0.926 |
| ECHO (floor) | model-free | 0.4287 | — | 0.765 | 0.686 | 0.777 | 0.584 | 0.755 |

Seed floor |S2(A1) − S2(A1_s1)| = **0.0157**. A1 − STOP **−0.1247** (W/T/L 79/32/93), A1 − ECHO −0.0322,
A1 − CV −0.0006 (inside the seed floor). **BAR-R6-W1: FAILED** (no interval by design: 7 logs < 8).
Lever reads vs the seed floor: vision (A1 − BLIND) **+0.0320 — beyond it**; nav (A1 − NAVOFF)
−0.0039, max speed (A1 − VMAXOFF) −0.0015 (2/201/1 scenes), time construction (A1 − A1NT) +0.0028 —
all inside it. Plan level (`plan_deltas_warmup.json`): seed replicate 4 s endpoint median 0.45 m
(same anchor 88.2 %); BLIND 1.98 m; NAVOFF 0.39 m; VMAXOFF 0.99 of plans bit-identical. Ladder:
102/204 stage-2 scenes score 0; unique zeros DAC 36, NC 16; single-term ceilings **DAC +0.135** >
EP +0.050 > NC +0.048 (`decomposition_warmup.json`) — the same ranking as navhard and navtest.

## 4. Deliverable 4 — navtest v1 PDMS

**Data: ALL ON THIS BOX — nothing to download.** Probed twice (the receipt AND the disk):
the 32 camera shards `openscene_sensor_test_camera_{0..31}.tgz` read **32/32 COMPLETE_VERIFIED**
(sha256 == HF ETag, `…/2026-09-19-navhard-download/raw/receipt_navtest_camera.json`) and the disk
holds 32 files summing to exactly **127,882,665,618 B**; the metric cache
(`D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest`) carries **12,146** metadata rows
(W3's count record PASS, re-counted here); 147 test-log pickles; W3's export
(`…/w3_navtest_v1/inputs/navtest_inputs.json.gz`, 12,146 tokens). The harness is **W3's**, imported,
not rebuilt: NAVSIM **v1.1 @ `3e8291b`**, which reproduces the leaderboard's CV **20.6517** to 4 dp
and scores STOP **61.8202**, HUMAN **94.5514** on all 12,146 tokens (W3 `RESULT.md` §3, its banked
per-token CSVs — the floors `code/parse_navtest6.py` reads, restricted to the arm's tokens).

**What refcv6 needs that did not exist: its 416 × 1024 frames.** W3's bank is 256 × 640. The 416 bank
is built by **W3's own builder**, imported by path with exactly two constants re-pointed (the output
shape, and the stitch → `frames416.E2BF` configured for `FRAME_416x1024`; `code/build_navtest416.py`),
at `D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank`. Per shard: extract F0/L0/R0 only,
stitch each unique frame once, sha256[:16] per token. Wall per shard **258 s → 1,524 s** as the box
filled up (stitch 0.39–0.75 s/frame under contention; `shard_NN.DONE.json`). ✅ **Complete: 32/32
shards DONE (2026-09-24), every token's stacked frames re-hashed against its index sha256[:16] on
every load (K1/KT; the tamper arm is REFUSED).**

**When it runs.** Step 1,000: `R6_A1` on W3's **200-token subset** (`A1_sub200_tokens.json`; its
tokens span **all 32 shards**, 1–20 per shard, so the subset cannot start before the bank is
complete) — PIPELINE VALIDATION. Each milestone: all 12,146 tokens through `milestone_waiter.sh`
phase 2. Scoring is cheap (W3: 1,051–1,222 s per arm); the bridge is the cost (~1.6 s/scene CPU fp32
⇒ ~5.4 h per full pass on CPU).

**The references (PUBLISHED, from banked primaries — and why they are not like-for-like).**
DiffusionDrive **88.1 PDMS** on navtest — arXiv 2411.15139, Table 1 p. 6
(`DiffusionDrive (Ours) C & L ResNet-34 … 98.2 96.2 94.7 100 82.2 88.1`), library key `2411.15139`,
sha256 `6ad4f8a379494eeb…`, re-read from the PDF here. ⚠️ Its input is **Camera AND LiDAR** ("C & L")
and it is trained **in-domain on navtrain**; refcv6 is camera-only and **zero-shot** from
PhysicalAI-AV. Ego-Status MLP **65.6** (NAVSIM, arXiv 2406.15349, Tab. 1 p. 7) / **66.4 ± 0.9**
(Tab. 3 p. 9, v1.1 leaderboard, 3 seeds), library key `2406.15349`, sha256 `d3bc66d321cfcecc…`,
both cells re-read from the PDF here — a **blind** agent (ego status only, no camera). These are
**stretch reference lines, not bars** (SPEC §5).

**Status (2026-09-26): RUN — navtest v1 PDMS exists for refcv6.**
* **Step 5,000, all 12,146 tokens: PDMS 46.48** [44.30, 48.36] (CPU fp32) vs STOP 61.82, CV 20.65 —
  **BAR-R6-T1 FAILED** (§3.3 has the table, the ladder and the four families).
* Step 1,000 (PIPELINE VALIDATION, W3's 200-token subset, CUDA as-trained): PDMS 33.37 [26.78,
  39.11] vs STOP 62.58, CV 21.82 on the same 200 tokens (`raw/navtest_s1000_sub200/`).
* **SPEC §12 max-speed diagnostics** (200 tokens; census groups: 99 no map limit, 49 map = oracle bin,
  49 map above the oracle, 3 below): step 1,000 — A1 − VMAXOFF **0.000** (200/200 ties), ORACLE − A1
  **−0.07** (1 token moved). **Step 5,000 (CPU fp32, `raw/milestones/step5000_navtest_diag/`):**
  R6_A1 **49.06** [42.73, 55.36] on the 200 tokens; **seed floor** A1 − A1_s1 **+1.26** [−3.17,
  +5.78] (same anchor on 84 % of scenes); **A1 − VMAXOFF 0.000 — 200/200 plans bit-identical**;
  **ORACLE − A1 +0.16** [−0.67, +1.04] (8/200 plans differ; +1.46 on the 49 "map above oracle"
  tokens, −0.39 on the 99 "no map" ones) — **inside the seed floor**. ⇒ At 5,000 the max-speed
  channel is inert with the deployable map input and near-inert with its own training definition:
  the §6.6 train/deploy mismatch costs nothing measurable yet, and the missing ceiling filter (§6.7)
  is why the map input cannot act.

## 5. Deliverable 5 — the one-command runner

```
python code/run_navsim_refcv6.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_5000.pt      # a local checkpoint
python code/run_navsim_refcv6.py --fetch 5000                                     # scp from Thor, md5, run
python code/run_navsim_refcv6.py --fetch 10000 --splits warmup --device cuda --gpu-wait-s 7200
bash   code/milestone_waiter.sh 5000                                              # wait for it, then all of the above
```

Each step is its own process, gated on the **artifact** of the step before (never an exit code):
(0) `--fetch N` — `ssh -n … md5sum` ON Thor + `scp`, read-only, local md5 must equal Thor's, recorded
in the kit's `MD5SUMS` (candidates: `runs/…/ckpt_N.pt`, `snapshots/…/ckpt_N.pt`, `…/ckpt_stepN.pt`);
(1) the step is read from the checkpoint → label **PIPELINE-VALIDATION** below 5,000, **RESULT** from
5,000 (SPEC §6); (2) `run_bridge6.py` per split (resumable rows, seams + declared-input manifests;
arms: warmup all six, navhard `R6_A1` + `R6_A1_s1`, navtest `R6_A1`); (3) the official scorer per arm
through the RAM-gated retrying queue — each refused unless its count guards read PASS; (4)
`parse6.py` / `parse_navtest6.py` against the floors, `families6.py` where a human future exists;
(5) `MILESTONE_SUMMARY.json` + `bars6.py` → `BARS.json` (SPEC §5 as written; `NOT_EVALUATED` below
5,000). **Device**: `auto` waits up to `--gpu-wait-s` for the GPU gate (memory.used < 4,300 MiB, no
other python on the card, ≥ 8 GB free RAM), else CPU fp32; one device per split, recorded; on CUDA
K0 and KD are **re-measured first** and a KD miss drops the dedup lever. `milestone_waiter.sh` polls
Thor read-only (`stat`, size stable over 120 s — `torch.save` may still be writing), runs the
two-stage splits, stages the outputs, then waits (≤ 12 h) for the 416 navtest bank to reach 32/32
shards and runs navtest.

**Validated how, stated exactly.** The runner ran end-to-end on the step-1,000 checkpoint (warmup)
on 2026-09-23: every artifact gate fired, the re-run's CSV was identical (max |Δ| 0.0) — but that run
used the mis-built decoder (§2, amendment A2), so it is **plumbing evidence only**
(`raw/VOID_wrong_anchor_units/step1000_runner_check/`). The runner imports nothing of the loader
itself (it calls `run_bridge6.py`, which calls the fixed `load_refcv6`), so the fix reaches it
unchanged; its first run on the corrected build is the **ckpt_5000 milestone**, armed
(`raw/milestones/waiter_5000.log`). Thor was at step **2,621** at 23:45Z (its own `metrics.jsonl`,
scp'd read-only) → ckpt_5000 expected ≈ 04:00–04:15Z at the measured ~6.6 s/step.

## 6. Findings that are not about NavSim (escalated in `COMMS.md`)

1. ⛔ **The live refcv6-r101-s0 run does NOT equalize the bottom 43 rows in its trunk, although its
   argv and its `config.json` say it does.** `_pin_trainer_cfg` sets
   `cfg.core.encoder.trunk_equalize_bottom_rows` as an ad-hoc attribute (`refc_v3_train.py:381`),
   then rebuilds the encoder config with `dataclasses.replace` for `--image-hw` (`:459`), which
   carries only declared fields — `CNNEncoderConfig` declares none by that name (positive control:
   `trunk_compile: bool` is declared). MEASURED on the model rebuilt from the run's own argv through
   the trainer's own pin path: `TimmResNetTrunk.cfg.equalize_bottom_rows = 0`, `normalise()` zeroes
   nothing. Files byte-identical at the run commit `287d72e` and at `fe5872f`. The live `train.log`
   carries no equalize line (0 of 138); `config.json`'s `"equalize_bottom_rows": 43` is the
   perception stamp, read from `args`. The LIFT bank does get the 43-row mask. ⇒ the C26 mitigation
   (a rig-identifying black strip, 0.899 decodable on eval-139) is not applied to the image the trunk
   sees. **Owner: Master Mind / Arch; whether the live run continues is the PI's call.** This stream
   evaluates the model as built and says so.
2. ⚠️ `taniteval/tools/refcv3_arm.rebuild_perception_branch` builds the eval-time
   `LiftGeometryBank` WITHOUT the 43-row observed mask the trainer passes (a train/eval mismatch in
   exactly the C26 rows) — relevant to the sibling stream's refcv6 loader.
3. ⚠️ The pushed tip `fe5872f` lacks eval tooling that D:'s working tree has (`taniteval/adapters/
   navsim_ci.py`, `taniteval/taniteval/bench/`, the `log_names` clustering in
   `taniteval/adapters/navsim.py`, the W3 / W7 / stage-1-extract package code) — imported here from
   D: by path. A clean-tree gate on the tip would fail.
4. ⚠️ E2's `score_arm.py` navhard profile points at an incomplete metric cache (no metadata CSV);
   the complete one is `C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage`.
5. ⛔ **`refcv3_arm.load_model` rebuilds refcv6's decoder with the WRONG anchor units** (`kappa`
   for an `alat` vocabulary) — found by control KL, fixed here, SPEC amendment A2 (§2 above).
6. ⛔ **The max-speed channel was trained on an ego-future ORACLE and NavSim can only give it a map
   limit — on navtest the two carry the same semantics for 25.0 % of tokens.** Run `config.json`:
   `max_speed_onehot_v6.provenance = "ego-future (oracle INPUT, ~1.7555 bits)"` (the max of the
   ego's own realised speed over [t0+2, t0+6] s, in a CONTAINING window — an upper AND a lower
   bound). MEASURED here, label-free (`raw/inputs/vmax_oracle_navtest.json`, SPEC §12): the
   training definition on the navtest logs gives bins 30 / 50 / 100 km/h = **75.74 / 23.50 /
   0.75 %** (training: 38.08 / 34.84 / 22.18 / 4.90 %); where a map limit exists (6,623 tokens) its
   bin is **above** the oracle's on **52.45 %**, equal on 45.81 %, below on 1.74 %; 5,461 tokens have
   no limit (the all-zero row, never trained). **Owner: Master Mind / Arch.**
7. ⛔ **Two of refcv6's three declared selection seams are not built in the live run.** Its
   `config.json` `selection_inputs` declares the parameter-free nav-compliance predicate and the
   max-speed **ARGMAX FILTER**; the rebuilt model has `decoder.speed_ceiling_filter = False` and
   `graft_nav_compliance = False` (no trainer flag sets either; the declaration is a static string
   emitted whenever v6 is on). Behaviourally, at step 1,000 **the max-speed input changes nothing in
   the plan: 204/204 warmup plans bit-identical between R6_A1 and R6_VMAXOFF (126 with a map
   limit)**, same anchor every time (`raw/controls/plan_deltas_warmup_s1000.json`). The filter is
   parameter-free and argmax-only, so it could be switched on at eval without retraining — a PI /
   Arch decision, not taken here. How much it would matter is measurable without running it: at
   step 1,000 the SELECTED warmup plan exceeds its map bin's ceiling on **3 of 126** scenes that have
   a limit (planned max speed median 6.63 m/s, p90 11.66, max 14.11 vs the 50 km/h = 13.89 m/s
   ceiling) — so on warmup, today, it would barely bite. **Owner: Master Mind / Arch.**

## 7. Blocked, and what unblocks it (status 2026-09-26)

| blocker | what it blocks | what unblocks it | owner |
|---|---|---|---|
| **The run is not finished** — the headline is the FINAL checkpoint | the headline reading | Thor finishing (step 32,561 at 07:51Z; the Master Mind measured 6.78 s/step all-in ⇒ ≈ 2026-09-27 17:30Z, `summary.json` with `"done": true`); `milestone_waiter.sh final` is ARMED and fetches `ckpt.pt` read-only (md5 before / after / dev box) and runs all three splits unattended | Thor (training) |
| **GPU gate** — held on 2026-09-26 by another session's REFe NavSim job (PID 6636, `refe_navtest_seam.py`); on 09-24 by the sibling battery (whose own gate self-deadlocked, escalation 10) | speed only: every split falls back to CPU fp32 (1.6–2.6 s/scene) after a 15-min wait; KP waits for the gate | the card free + RAM ≥ 8 GB — the runner then switches to CUDA as-trained on its own (after K0/KD) | box scheduling |
| **Host RAM** (other sessions' REFe teacher shards, ~1.2 GB each) | E1's RAM guard aborts scorers below 3 GB (3 aborts on 09-23/24) | mitigated: RAM-gated retrying scorer queue; the runner's CPU bridge waits for ≥ 6 GB | box scheduling |
| **PI decisions** | (a) is the map posted limit the admissible max-speed INPUT (the recorded ruling speaks of LABELS; escalation 5), given the train/deploy mismatch (§6.6)? (b) switch on the declared, parameter-free speed-ceiling ARGMAX FILTER at eval (§6.7) — measured: it would bite on 3/126 warmup scenes at step 1,000 and not at all in the 2–5 m/s band where refcv6 loses; (c) an admissibility clause against stop-gaming for the bars (SPEC §5); (d) whether the live run continues with the trunk-equalization defect (§6.1) | the PI | PI |
| **Model-side lever** (Rule Zero) | beating STOP | lead-aware longitudinal behaviour at 2–8 m/s and drivable-area compliance on turns (§0, §3.3) — an architecture / training decision | Master Mind / Arch |
| **Integration** | the tip `fe5872f` lacks D:'s eval tooling (§6.3); `refcv3_arm` needs the units fix and the 43-row lift mask (§6.2, §6.5) | landing them | Master Mind / taniteval owner |

Resolved since 09-24: the navtest 416 bank (32/32 shards); KH-nav and KH-navtest (both REPRODUCED);
the RESULT-grade checkpoint (ckpt_5000 read).

## 8. Manifest (2026-09-26)

**Landing.** Per the Master Mind (2026-09-26) this stream never runs git: finished files are copied
to the D: package (`code/stage_to_repo.py`, copy-only, content-verified) and listed, batch by batch,
in `LANDING_READY.txt` at the package root; the Master Mind lands them. The landing gate keeps a
file LOCAL when it is over 20 MB or carries a canonical UUID (the devkit scorer logs carry random
`thread_id` UUIDs — not clip ids — so every `score_*.log` stays local; the `*.counts.json` beside
each carries its parsed guard numbers). CSV / JSONL rows carry NavSim's public 16-hex benchmark
tokens (not PhysicalAI clip ids).

| artifact | where | single location? |
|---|---|---|
| SPEC / PLAN / RESULT / COMMS, `code/` (22 scripts), `tests/` (5 suites) | repo: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/` (via LANDING_READY) + ev6 working tree | no |
| harness controls (warmup CV/STOP/ECHO, KH-nav, KH-navtest), KD/KL/KDET/KDEC, plan deltas, runner check | repo: `…/navsim/raw/controls/`, `raw/harness_repro*/`, `raw/floors/` | no |
| step-1,000 pipeline validation (warmup 6 arms, navhard, navtest sub200 + diagnostics) | repo: `…/navsim/raw/{bridge,scores}_*_s1000/`, `raw/summary_*_s1000.json`, `raw/navtest_s1000_sub200/` | no |
| step-5,000 reading (navhard 2 arms, navtest full + 4 diagnostic arms; warmup re-running) | repo: `…/navsim/raw/milestones/step5000*/` | no |
| scorer logs (`score_*.log`, `r6*.log`) | ev6 + D: package only (UUIDs ⇒ never landed) | ⚠️ local only |
| seam `.npz` files under `raw/milestones/*/bridge_*/` | ev6 only (rebuildable from the landed `rows_*.jsonl`) | ⚠️ single location |
| 416 frame banks: warmup / navhard s1 / navhard s2 (10.4 GB) | `C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/` | ⚠️ single location (rebuildable: `code/frames416.py`, `code/build_navhard_frames.sh`) |
| 416 navtest bank (32 shards) | `D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank` | ⚠️ single location (rebuildable: `code/build_navtest416.py`) |
| checkpoints `ckpt_step1000.pt`, `ckpt_5000.pt` (+ 30000, final as fetched) with `MD5SUMS` | `D:/refcv6_eval_kit/ckpt/` (copies of Thor's, md5-verified) | copies |
| NavSim aux (anchors, extrinsics, max-speed sidecar) | `D:/refcv6_eval_kit/navsim_aux/` (md5 vs Thor in `MD5SUMS.thor`) | copies |
| devkit scratch | `C:/Users/Admin/navsim/exp/e6*`, `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/r6*` | scratch |

## 9. Where this stopped, and what runs unattended

**Hand-off state 2026-09-26 ~08:40Z** (resumed 07:48Z after the weekly-limit stop of 2026-09-24
02:38Z; everything the stop left running had finished and was reviewed, §3.3 / COMMS D20).

| lane | script → log | does | done-marker |
|---|---|---|---|
| results | `code/campaign_0926.sh` → `raw/campaign_0926.log` | warmup @ 5,000 (re-run, CPU after the gate wait) → `milestone_waiter.sh 30000` (fetch read-only + md5; warmup → navtest + 200-token diagnostics → navhard) → `raw/milestones/waiter_30000.log` | `ZZCAMPAIGN0926DONEZZ` |
| precision | `code/kp_lane.sh` → `raw/kp_lane.log` | KP (step 1,000 warmup, CUDA vs CPU) then KP-navtest (step 5,000, 200 tokens) — only when the GPU gate opens (≤ 24 h) | `ZZKPLANEDONEZZ` |
| final | `code/milestone_waiter.sh final` → `raw/milestones/waiter_final.log` | polls Thor every 10 min (read-only `cat summary.json`); after `"done": true` fetches `ckpt.pt` (md5 before / after / dev box) and runs all three splits + diagnostics | `ZZMILESTONEfinalDONEZZ` |

Every unattended output is COPIED to the D: package (copy-only, no git) but is **not** declared
landing-ready until an agent has reviewed it: count guards (`*.counts.json` PASS, successful ==
expected, failed 0, seam calls == declaration), stand-ins (0 on navhard/navtest), device + precision
per split (`seam_*.manifest.json`), the seed replicate, and the harness control on the split — then
`python code/stage_to_repo.py --only <globs> --landing "<heading>"` and a message to the Master Mind.
