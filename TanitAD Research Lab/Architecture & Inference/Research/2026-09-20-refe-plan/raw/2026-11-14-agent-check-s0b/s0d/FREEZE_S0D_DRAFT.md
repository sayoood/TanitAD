# FREEZE_S0D_DRAFT: S0-D "R-sim" (the agent LIST through NAVSIM's own scorer), built and smoked, NOT run (2026-11-14)

Status: DRAFT for the MM to review and freeze. No gain, capture or precision number exists: the smoke printed counts and timings plus the
R-sim-true identity control only; a code-path self-test on 1 log / 6 sets wrote its result into scratch and was deleted unread.

## What it measures
NAVSIM's at-fault NC and TTC (the real `PDMScorer`), run on a modified agent list. A proposal is vetoed if nc == 0 or ttc == 0 under that list.
Pick = P6's canonical pick among survivors (all 64 vetoed -> P6's pick). Gain, capture, G-FID, bootstrap = the S0-B code (`s0b_readout.py`, unchanged,
imported from the parent hold). Flags are stored per arm as nc==0 and ttc==0 (packbits, `s0d_flags.npz`, same layout as `s0b_flags.npz`).

## Arms (a modification of the FUTURE tracked objects before NAVSIM builds the observation; ego, map, proposals, PDM-Closed untouched)
| arm | definition |
|---|---|
| R-sim-true | identity: the very same track objects. Control: flags == truth on every proposal, raw nc/ttc == fit pack exactly, per-set gain == oracle exactly |
| R-sim-t0 | every dynamic agent present at iteration 0 sits at its t0 box (pose, size, heading) at all 11 detection frames, velocity (0, 0). Agents absent at t0 do not exist |
| R-sim-cv | every dynamic agent present at t0 moves from its t0 box by its t0 (vx, vy) (nuPlan Agent.velocity = lidar_box vx, vy, GLOBAL frame), heading and size fixed, velocity field (vx, vy) |
| R-sim-t0v | (secondary, MM-added) pose and box frozen at t0 exactly as R-sim-t0, but each agent keeps its TRUE t0 velocity on the object, so `is_track_stopped` classifies it as on the real scene. Separates pose from the stopped-track classification; a head predicting a moving/stopped attribute from one frame |
| C-SIM-NONE | all dynamic agents removed at every frame (statics kept). Control: its NC flags are a subset of R-sim-true's per proposal (stricter than the per-set count) |
Dynamic = type in NAVSIM's AGENT_TYPES (vehicle, pedestrian, bicycle). Statics (cones, barriers, signs, generic objects) are NEVER modified in any arm: they are the
true statics, interpolated by NAVSIM exactly as before.

## Outcome row (mechanical, held-out column decides, point estimate decides)
Order INVALID -> GO -> STOPFLAG -> MOTION -> NULL. INVALID if R-sim-true != truth (flags, raw nc/ttc or gain) or C-SIM-NONE violates monotonicity; S0D-GO if c(R-sim-t0) >= 0.25;
**S0D-STOPFLAG** if c(t0) < 0.25 and c(R-sim-t0v) >= 0.25; S0D-MOTION if c(t0) < 0.25 and c(t0v) < 0.25 and c(cv) >= 0.50; else S0D-NULL. c = gain / O_NC+TTC on the same sets. CI: paired log-cluster bootstrap, 10,000 draws, seed 20260927.

## Hook (file:line)
- NAVSIM navsim @ 3e8291b `navsim/planning/metric_caching/metric_cache_processor.py`: `_interpolate_gt_observation` reads the GT only through `scenario.database_interval`
  (:105) and `scenario.get_tracked_objects_at_iteration(iteration=...)` for gt_indices 0,10,..,100 (:110-113); statics and agents are split at :133 (AGENT_TYPES);
  a one-sample track is passed through untouched at :165-166.
- Ours: `s0d_hook.py` `HookedScenario.get_tracked_objects_at_iteration` (:98) is that scenario for ONE call only; `s0d_rsim.py:135` passes it to
  `proc._interpolate_gt_observation` and nothing else. PDM-Closed's `_get_planner_inputs` (base_parts) gets the REAL scenario.
- Proof of identity: `test_s0d.py::test_identity_hook_is_bit_identical_to_unhooked_navsim` (all 51 steps' polygon WKB + unique objects, synthetic scenario through NAVSIM's real
  interpolation); and on 2 REAL held-out tokens (smoke): hooked R-sim-true == unhooked NAVSIM observation scores, 0 mismatches; flags == truth, raw nc/ttc max abs diff 0.0;
  my split scorer (simulate once, score per arm) == `PT.score_plans`, 0 mismatches.

## Does the scorer read agent velocity? YES, in one place
`pdm_scorer_utils.get_collision_type` (navsim .../scoring/pdm_scorer_utils.py:45 `is_track_stopped`): an agent with speed <= 0.05 m/s is a STOPPED_TRACK, and a collision with it
while ego moves is at-fault whatever the geometry. It reads the FIRST-appearance object of the track (the t=0 interpolated one, `unique_objects`). TTC does NOT read agent
velocity (it uses the ego proposal's speed); the track heading (`unique_objects[token].box.center.heading`, t0 heading) is read by `is_agent_ahead/behind`.
Consequence: in R-sim-t0 every agent is a stopped track, so rear and lateral hits count as at-fault. That is NAVSIM's behaviour for a perceived-stationary agent, not an ill-definition,
but it means the t0 arm tests pose AND the stopped-track classification together (see HANDOFF, MM decision 1).

## Run (MM; dev box, one process, CUDA off, venv driverl-eval; nothing keyed; resumable per log under <run>/parts)
```
set CUDA_VISIBLE_DEVICES=-1 ; TMP=TEMP=TMPDIR=C:/Users/Admin/s0d/work/tmp (create it)
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe E:/Projects/TanitAD-artifacts/scorer_gym_20261107/agent_check_s0b/s0d/test_s0d.py            (13 tests OK)
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe .../s0d/test_s0d.py --mutations                                                              (12 RED)
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe .../s0d/s0d_rsim.py --out <run_dir>                    held-out column (1,343 sets, 13 logs)
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe .../s0d/s0d_rsim.py --out <run_dir> --all             all 3,958 covered sets (47 logs; held-out column inside)
```
It runs far longer than 4 min: launch detached (WMI, CreateFlags 0x08000200, `< NUL`), poll the log for `[k/13]`. Outputs: `s0d_result.json`, `S0D_RESULT.md`, `s0d_flags.npz`, `parts/`.

## ESTIMATED cost (from the 2-token smoke; n = 2, so a range)
- per set (re-smoked with 5 arms): 2.1 s warm, 4.2 s mean; (earlier 4-arm smoke: 1.9 / 4.5); each arm adds about 0.2 s. 1.9 s warm (PDM-Closed 0.4, simulation 0.25, 4 x (observation 0.02-0.14 + scorer 0.12-0.14)), 4.5 s mean including the cold first token of a log (scenario build 2.1 s, PDM-Closed 2.7 s).
- held-out column: **0.8 to 1.6 CPU-h**; all-covered: about 2.1 to 5 CPU-h. Peak RSS **468 MB** (smoke) / **537 MB** (6-set self-test through assembly): modest, the fit pack dominates.
- Cold costs are per log in practice (13 or 47 logs), so the true held-out figure is probably near the low end plus ~13 x 5 s. Not measured.

## sha256 (MEASURED on the hold's files at the time of writing)
| file | sha256 |
|---|---|
| `s0d/s0d_rsim.py` | `66d787153424efef526972f790ea7f06a928eabfdbe574efa71f6cb69320d39e` |
| `s0d/s0d_hook.py` | `b2d953f6d9aae22e1f47d987658ea8301c013ad32a96e9c9db020365e58af345` |
| `s0d/test_s0d.py` | `b18550591fa21af95d0bc995f03a9d49998cec692529676b0f4bd2222d49fb40` |
Unchanged S0-B pins that this run imports (from `FREEZE_S0B.md`): `s0b_readout.py f08e2484...`, `s0b_run.py c9d9c6cc...`, `s0b_raster.py 1d2022c2...`, `ref/sel_common.py 50727023...`,
`ref/refe/*` (verbatim tip exports), `covered_sets.txt` 4e61d655..., `found_logs.txt` 06cdec34...; the fit pack sha prefix `e58653c5` is asserted at run start.
