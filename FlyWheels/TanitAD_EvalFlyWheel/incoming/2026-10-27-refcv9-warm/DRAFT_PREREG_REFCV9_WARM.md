# DRAFT_PREREG_REFCV9_WARM -- refcv9, warm-started from refcv8 (Master Mind, 2026-10-27; DRAFT -- sealed only after the code review and the 50-step smoke)

PI decisions in force (2026-10-27, `Project Steering/PI_DECISION_QUEUE.md`): warm start, smoke first; inputs unchanged from refcv8 INCLUDING the
future-derived speed band (disclosed: the band is the ego's own max speed 2-6 s ahead, leak 0.224 vs the 0.05 bar -- every refcv9 number states it);
the camera-blind and ego-only-floor controls are BINDING; `launch_gate.py` may gain G-LIVE liveness rules for `w_r9_*` and nothing else.

## 1. The arm REFCV9-W (one arm, one training seed)
* **Init:** refcv8 EMA-48,852 (`ckpt_ema_48852.pt`, md5 b4364029b68f29a33eeb4aebbbbd002b). Same corpus, data, batch 16 and argv as refcv8 except below.
* **Frozen:** perception (image backbone, BEV lift, box head), by `--r9-train-only planner,tactical,map` (FrozenStateGuard + G-DVB).
* **Trained:** three literal module sets (`map_only_ft.TRAIN_ONLY_SETS`, a17de18, MEASURED on the real build): `planner` = `core.decoder` minus
  the 5 subtrees the build declares frozen (178 tensors, 15,254,091 values); `tactical` = 11 modules outside `core.decoder` (tactical trunk,
  tactical decoder, lat / lon heads, behaviour gate, nav-to-tactical, ...; 113 tensors, 5,476,306 values); together with `map` 309 tensors,
  20,835,429 values; `map` = `_map_hires.{refine, near, near_refine}` (18 tensors, 105,032 values;
  PI 2026-10-27: "include it"), with a precision-targeted term `--w-map-fp` (expected own-map sidewalk + edge probability on GT-drivable cells
  of the near-field corridor, x 0-20 m, |y| < 3 m). MEASURED motivation: MAPFT's converged CE did not move the map (`H-R9-MAPFT-1`); own sidewalk
  flags are 25 % precise at 0-5 m and 63 % of them sit on GT-drivable cells. ⚠️ An earlier draft line read "186 tensors; 15,470,833 values" for
  planner + tactical: that was planner + map (204 tensors), and the tactical set was not in it.
* **Map-term weight (fixed from the smoke, never from eval):** the smoke runs with a provisional `--w-map-fp 1.0`; then
  `w = 0.1 x map_hires_loss / map_fp_loss` at the smoke's FIRST logged step, rounded to one significant figure (the term starts at a tenth of the
  map CE's scale).
* **Warm-start schedule hygiene:** `--r8-alloc-emit-start 0` (refcv8 is past its 1,000-step emit delay; a warm start must not hide its allocated
  candidates); refcv8's GT-constraint teacher-forcing ratio is HELD at its end-of-run value, never restarted at 1.0:
  `--r8-tf-start 0.25 --r8-tf-end 0.25` (refcv8's schedule reads 0.25 at step 48,851; required by the profile).
* **The one training change (measured motivation):** `--r9-alloc-force-schedule` -- the GT hypothesis is no longer always forced into the 32 allocated
  candidates during training (`H-R9-ALLOC-ORACLE-1`: refcv8 never trains on a pool that lacks the right class, but at inference a wrong decel call fills
  the pool with braking candidates; `H-R9-L2B-LOCUS-1`: on 98 % of false-brake windows a GOOD cruise candidate is in the fan and the selector misses it).
  Schedule: forced with probability 1.0 -> 0.0 linearly over the first half of the run, 0.0 after (predicted-only pools, as at inference):
  `--r9-alloc-force-schedule linear:1.0:0.0:0:<floor(N/2)>`, the end step checked against `--steps` by the profile.
* **Budget:** N from the 50-step smoke by the MAPFT rule form in exact rational arithmetic:
  `N = min(25000, floor(0.85 x (172800 - smoke_wall_s - RESERVE_S) / s_step))`, 172,800 s = the PI's 48 Thor-h; RESERVE_S = 1.5 x the MEASURED cost of
  4 in-run evals + 1 save (from the MAPFT run log, stated in the kit); **N < 4000 -> STOP** (the second, predicted-only half of the schedule would be
  under 2,000 steps), report s_step to the PI. `--warmup 200 --lr 1e-4 --eval-every max(500, N // 4) --ema-decay round(1 - 10 / N, 5)`; scored
  weights = EMA at step N.
* **Stage 2 (dev box, 0 Thor-h):** the R9-PF progress head (`REFCV9_INMODEL_DESIGN.md`, 70,299 params, trunk frozen) fitted on the clean calibration
  clips (L1C 300 + FRESH2 GROW), with the decel decision PREREG_FRESH2 confirmed: FUSE (`H-R9-FRESH2-1`, F-1 PASS +0.0643 [+0.0423, +0.0847] on
  kinematic GT; G-1 FAIL). Disclosed: raw -a0 alone read 0.7341 there, above FUSE's 0.6981 -- the decision's gain is ego dynamics, which is why
  W4 / W5 below are binding. Written into the checkpoint. ⚠️ The tactical trunk trains in stage 1, so `z_tac` changes: the LH head is REFIT on
  refcv9's own `z_tac` (same clips, same fit protocol as FRESH2); D2 is built on a0 and is unaffected. FRESH2's AUCs are refcv8's and are not
  carried over -- stage 2 re-reads F-1 on the CONFIRM windows with refcv9's LH before the decision is written in.

## 2. Controls (a failed control voids the bars)
* **K-FROZEN:** every changed tensor (raw + EMA) is in the declared planner / tactical / map sets, named per set; perception bit-equal to refcv8
  (`assert_map_only_finetune.py`, generalised to the r9 sets).
* **K-REF-ANTICHEAT (beside, measured before launch):** the camera-blind and ego-only-floor instruments are first run on refcv8 EMA-48,852 + H2cn
  (`2026-10-27-refcv9-anticheat/`). If refcv8 already fails W4 or W5, that is reported to the PI before refcv9 fires: the warm start inherits it.
* **K-BASE:** the battery re-scores refcv8 EMA-48,852 + H2cn and reproduces its banked registered table (`REGISTERED_PF_TABLE.sha256`).
* **CAMERA-BLIND (PI, binding):** REFCV9-W with each window's camera tokens replaced by those of a window from ANOTHER episode (one fixed derangement,
  seed 0), every ego input unchanged.
* **EGO-ONLY FLOOR (PI, binding):** a ridge regression from the ego inputs as fed (v0, ego state at t0, the speed band) to the 6 s trajectory, fitted
  on train windows only, lambda by episode-grouped CV on train (`H-R9-SCALAR-FLOOR-1`'s instrument).
* **ATTRIBUTION (beside, run second when Thor is free):** REFCV9-CONT -- the same trainable set and N with the schedule held at 1.0 (today's pools):
  separates the pool change from "more planner training". Without it a REFCV9-W gain is reported as "warm start + pool change", not as the pool lever.

## 3. Bars (T1 self-action OPEN LOOP on logged eval139 frames -- not driving performance; S6 nav-compliant windows; paired episode-cluster bootstrap
B 2000 seed 0 over EPISODES; inference seeds 0 AND 1 must both hold; reference = refcv8 EMA-48,852 + the shipped H2cn stage)
* **W1 false decels:** FD on GT-cruise windows separated LOWER.
* **W2 missed decels:** MD not separated worse.
* **W3 four families:** no registered row of the SHIP-R8 battery (LONGITUDINAL, LATERAL, TACTICAL, `ade_m`; STRATEGIC n = 0, stated) separated worse.
* **W4 camera-blind (binding):** REFCV9-W separated BETTER than CAMERA-BLIND on `ade_m` and on each family's primary row.
* **W5 ego-only floor (binding):** REFCV9-W separated BETTER than the EGO-ONLY FLOOR on `ade_m` and on each family's primary row.
* **W6 map (the map is trained in this run):** PREREG_MAPPATH's M1 and M2 on refcv9's OWN rasters (soft cost on the progress-matched fan) --
  sidewalk and edge contact separated below BASE; and the corridor precision at 0-5 m above refcv8's 0.251 (`nearfield_prec`, beside the bar).
* PASS = W1-W6. Attribution between the planner change and the map change: W6 is read on refcv9's rasters with refcv8's planner trace held fixed
  (as MAPFT's K-PLAN-ID did), so the map gain is separable from the planner gain. Beside, never a bar: the four families on FRESH2 CONFIRM windows.

## 4. Committed next steps
* **PASS:** REFCV9-W + stage 2 is the refcv9 deliverable candidate; REFCV9-CONT attributes it; a ship request to the PI follows the four-family read.
* **FAIL on W4 or W5:** the planner does not use the camera beyond the ego inputs -- reported first, before any other row, and no ship request.
* **FAIL on W1 only:** the pool change did not move the selection; next = selection trained on predicted-only pools from step 0 (schedule `const:0.0`).
* **FAIL on W3:** name the row; the warm start costs something the reference keeps.

## 5. Open before sealing
Decided by the PI 2026-10-27: Thor cap **48 h**; the `refcv9-warm` profile may be added; the map term joins this run. Landed: the R9-PF head +
`--r9-alloc-force-schedule` + liveness rules (690f84b); `--r9-train-only planner,map`, `--w-map-fp`, the `refcv9-warm` profile + 2 RED arms
(aeb5446); the `tactical` set + the emit-start / teacher-forcing / schedule argv rules (a17de18); R9W-BUDGET and R9W-MAP-FP-WEIGHT closed
open-only (both fixed by the rules above, the token's argv sha256 pins them). K-REF-ANTICHEAT: W5 PASS on refcv8 (3d4cac2), W4 rolling.
Before sealing: the 50-step smoke's `s_step`, N and `w`.
