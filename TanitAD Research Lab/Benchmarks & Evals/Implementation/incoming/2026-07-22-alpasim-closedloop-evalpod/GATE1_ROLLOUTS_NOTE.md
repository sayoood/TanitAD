# Gate-1 on-policy rollout collection — REF-C-base closed-loop, junction scenes — MEASURED 2026-07-23

On-policy AlpaSim rollout data for **Gate-1 (closed-loop-aware training: RoAD/CAT-K/DAgger)**, whose
data is on-policy rollouts. Collected REF-C-base closed-loop over the **15 junction scenes**
(7 intersection + 8 roundabout — where the junction failure lives) and logged, per rollout, the three
signals Gate-1 needs. Non-committal prep: the fine-tune method + Sayed's Gate-1 go come later — this
de-risks it and characterizes the failure regardless.

## ⚠️ FRAMING (mandatory)
**WITHIN-SIM RELATIVE, on NuRec reconstructions, ~3.2× OOD** (RUN_RECIPE §13: REF-C's open-loop ADE on
these reconstructions is 1.47 = 3.1× its 0.47 real-footage ADE, *before* any closed-loop shift). The
divergence **PATTERN** is the deliverable and is trustworthy; the **absolute metres are within-sim, not
real-world**, and bundle the reconstruction-OOD gap with the closed-loop covariate shift. 480×854, canon
`f_eff=266.1 OK` (input geometry validated). All numbers **MEASURED** (`gate1_summary.json`) unless tagged.

## TL;DR
- **Collected (MEASURED):** 15 junction rollouts, **675 on-policy steps**, REF-C-base. pass **3/15**,
  offroad **11/15**, at-fault collision **5/15**. Every artifact staged (below) or indexed pod-side.
- **The divergence is a compounding covariate-shift signature, NOT an execution/tracking failure.**
  Executed cross-track to the reference corridor grows **0.16 m → 9.92 m** (first→last third) while
  **near-horizon plan-tracking stays tight (plan-vs-executed @0.5 s = 0.49 m throughout)**. The **plan
  leads** the departure (plan-to-corridor late **12.98 m** > executed late **9.92 m**). The ego faithfully
  executes a plan that progressively leaves the corridor as the closed-loop input drifts off-distribution.
- **Refines the brief's premise.** "Plan on-road but the ego departs *from its plan*" is **not** what the
  data shows: the ego does **not** depart from its own near-term plan (0.49 m). The failure is **on-policy
  PLAN degradation** executed faithfully — which is exactly the lever Gate-1 (retrain the planner on the
  on-policy distribution with expert recovery targets) fixes.
- **Sufficiency: SUFFICIENT TO PROTOTYPE / de-risk a closed-loop-aware fine-tune** (pipeline proven,
  recovery targets collected, on-policy frames preserved), **not yet sufficient to train a robust policy**
  (n + the OOD confound — see Sufficiency).

## What was run (MEASURED)
- **Model:** REF-C base (`/root/models/refc-base-30k/ckpt.pt`, step 29999, 128 anchors, md5 8f10d6…).
- **Scenes:** the 15 junctions from the balanced v4.2 suite (`scaled_suite_labels.json`): intersection
  `00169207 0810968d 41c06176 59cb0598 69efe005 780ece49 8b04d54e`; roundabout `3cc29c99 471f2484 6dcd2117
  bc843fa0 fd3a49fa adb72a39 c3d4065e d3267951`. Reused sceneset `986fec83…` (no re-download).
- **Topology:** the validated M2/§12 bare pipeline (renderer :6011 native f-theta → controller :6007 →
  physics :6006 → REF-C driver :6789), driver run with **`--log-preds`** for per-step logging; rollouts
  **serialized** (`n_concurrent_rollouts=1`) for clean per-session logs. 50 sim-steps @5 Hz, 3 s force-GT
  warmup, `f_eff=266.1 OK` (canonicalization correct → REF-C input trustworthy; the OOD is the recon gap,
  not preprocessing). Pod left CLEAN (services down, GPU 0 MiB, `gpu_lock` released).

## Dataset format (the collected Gate-1 data)
**`gate1_rollouts.jsonl`** — 675 lines, one per (scene, step); the compact training-ready record. Fields
map to the three requested signals:
| signal (brief) | field(s) | meaning |
|---|---|---|
| **1. visited state sequence** (on-policy dist. open-loop omits) | `x,y,yaw,speed,t_rel_s` | the closed-loop pose+speed REF-C actually reaches |
| **2a. executed + aligned GT** | `exec_xte_m` | executed → nearest GT-corridor vertex (=AlpaSim `dist_to_gt`, validated) |
| **2b. recovery signal** | `recovery_rig_fwd_left`, `ref_lookahead_rig[4]` | vector to the corridor + GT **expert path 0.5/1/1.5/2 s ahead**, rig frame = the **DAgger/CAT-K label** |
| **3. plan-vs-executed divergence** | `plan_rig[4]`, `plan_exec_dev_m[4]`, `plan_churn_m`, `plan_mean_xte_m` | REF-C's plan, its deviation from realized motion per horizon, replan churn, plan→corridor |

**`gate1_summary.json`** — per-scene + per-category + overall aggregates, temporal split, departure onset,
and cross-validation vs the parquet.
**On-policy observation frames** (the images REF-C saw at each step — the heaviest tensor) are **preserved
pod-side** in each `rollout.asl` (protobuf message log, ~11 MB/rollout, re-readable via
`eval.asl_loader.load_scenario_eval_input_from_asl` / `alpasim_utils.logs.async_read_pb_log`), indexed by
`rollout_id` in `gate1_summary.json[*].rollout_id`. Per convention, frames stay pod-side; a fine-tune
extracts them and pairs each with the collected recovery target.

## The measured divergence pattern (⭐ the deliverable)

**Compounding covariate shift (overall, n=15, 675 steps):**
| quantity | early (first ⅓) | late (last ⅓) | growth |
|---|---|---|---|
| executed cross-track to corridor (`exec_xte`) | **0.16 m** | **9.92 m** | 62× |
| plan → corridor (`plan_mean_xte`) | 0.57 m | 12.98 m | 23× |
| near-horizon plan-vs-executed @0.5 s | **0.49 m (flat)** | — | tracking stays tight |
| plan churn (‖plan_i − plan_{i+1}‖) | 0.74 m | — | moderate |

- **Departure onset** in **14/15** scenes (only `adb72a39` never departs — and it PASSES), median **5.0 s**
  (≈2 s after the 3 s warmup). `exec_xte_start = 0.16 m` confirms the ego begins on-corridor (warmup +
  world-frame alignment validated; `exec_xte` reproduces AlpaSim's own `dist_to_gt_trajectory` to ≈0.1 m,
  e.g. 41c06176 24.75 vs 24.63, 59cb0598 4.11 vs 4.10).
- **Mechanism (MEASURED):** the ego executes its near-term plan faithfully (0.49 m) while **both** the plan
  and the executed path leave the reference corridor together and compound super-linearly; the **plan leads**
  (points further off than the ego currently is). ⇒ **on-policy planning degradation under closed-loop
  covariate shift**, executed faithfully — not a controller/tracking failure.

**Two junction failure modes (MEASURED):**
| category | n | pass | at-fault collision | offroad | exec_xte late |
|---|---|---|---|---|---|
| **intersection** | 7 | 1/7 | **0.714 (5/7)** | 0.714 | 9.29 m |
| **roundabout** | 8 | 2/8 | **0.0 (0/8)** | 0.75 | 10.47 m |
- **Intersections fail via collision AND departure** (5/7 at-fault — crossing traffic REF-C never yields to;
  `69efe005`/`59cb0598` collide while ≈on-corridor → an agent-interaction gap distinct from covariate shift).
- **Roundabouts fail via PURE corridor departure** (0 at-fault collisions; the ego drifts out of the circular
  corridor). The clean covariate-shift signature lives here (e.g. `c3d4065e` 0.11 m → 24.6 m).

## Sufficiency for a closed-loop-aware fine-tune (the brief's question)
**Verdict: SUFFICIENT to PROTOTYPE / de-risk; not yet sufficient to train a robust policy.**
- ✅ **Right lever confirmed.** The failure is on-policy plan degradation, exactly what RoAD/CAT-K/DAgger
  target; a tracking/MPC fix would miss it (tracking is already tight).
- ✅ **Recovery labels in hand.** `ref_lookahead_rig` gives the expert action from each off-policy visited
  state (rig frame, REF-C's output space) — a drop-in DAgger/CAT-K target. Concrete example (00169207 s0):
  expert 15.1 m fwd @2 s vs REF-C plan 18.7 m → REF-C over-reaches; the label corrects it.
- ✅ **On-policy observations preserved** (`.asl`, indexed) to pair obs→label.
- ✅ **Pipeline proven + scalable** (`scaled_master.sh` runs any driver over any suite; `gate1_run.sh`
  adds `--log-preds`; `gate1_postprocess.py` emits the dataset).
- ⚠️ **Scale:** 675 steps / 15 scenes de-risks the loop; a robust fine-tune needs more scenes (roundabouts
  are ~2.5 % of the pool — pipeline to classify more is staged: `select_suite.py`/`kf_batch.py`).
- ⚠️ **OOD confound (load-bearing):** the visited states are ~3.2× OOD reconstructions, so a fine-tune on
  them partly teaches reconstruction-OOD robustness, **not pure closed-loop recovery**. Ideal Gate-1 data
  is on-policy at REF-C's real-footage distribution (or with the recon gap closed). **Within-sim data is
  valid for prototyping the objective + loop; a real Gate-1 read needs the OOD gap addressed.**
- ⚠️ **Intersection collision mode** needs an added collision/agent-cost signal (crossing traffic); the
  route-following recovery target addresses the departure mode, not collision avoidance directly.

## Caveats / honest gaps
- **`plan_churn_m` = AlpaSim's `plan_deviation`** (consecutive-plan disagreement, near-horizon weighted;
  confirmed from `eval/scorers/plan_deviation.py`) — **NOT** executed-vs-plan tracking. Tracking is our own
  `plan_exec_dev_short` (0.49 m). (Caught before asserting; the misread would have been a C3/C4 slip.)
- **Plan-vs-drivable-area (on/off-road) check NOT shipped as decision-grade.** Replicating AlpaSim's
  `offroad` lane-polygon logic reproduced its per-step `offroad` on clean scenes (1.0) but only ~0.6 on the
  complex failing junctions — below a bar to quote a per-step "plan on-road" fraction (a C5 risk). We report
  the **executed** offroad from AlpaSim's own metric (validated) + the route-departure; the exact plan
  on/off-road check (feed the plan through `eval.scorers.offroad` directly, not a re-implementation) is the
  cheap refinement. This is why the note says "plan leaves the *corridor/route*" (measured) and stops short
  of a quoted "plan off the *road*" fraction.
- n=15 junctions, 1 rollout/scene; REF-C diffusion is stochastic run-to-run (±~0.08 score, §15). 480×854.

## Deliverable manifest
| artifact | where | status |
|---|---|---|
| `gate1_rollouts.jsonl` (675 steps) | repo (staged) | ⭐ the on-policy Gate-1 dataset (visited states + recovery targets + divergence) |
| `gate1_summary.json` | repo (staged) | per-scene/category/overall aggregates + xval |
| `GATE1_ROLLOUTS_NOTE.md` (this) | repo (staged) | write-up + sufficiency verdict |
| `gate1_run.sh` | repo (staged) · pod `/workspace/` | run harness (`vs_suite_run.sh` + `--log-preds`) |
| `gate1_master.sh` | repo (staged) · pod `/workspace/` | autonomous renderer+run driver |
| `gate1_postprocess.py` | repo (staged) · pod `/workspace/` | GT-align + divergence + recovery-target emitter |
| junction config | pod `/workspace/gate1_junc/` | 15-scene config (sceneset 986fec83, agg=true) |
| 15 rollouts (`rollout.asl` ~11 MB ea = **on-policy frames**, `metrics.parquet`, mp4) | **pod only** `/workspace/gate1_junc/rollouts/clipgt-*/<rollout_id>/` | indexed by `rollout_id`; regenerable |
| REF-C base ckpt | **pod only** `/root/models/refc-base-30k/ckpt.pt` | pre-existing (REGISTRY §4) |

**Pod left CLEAN:** all services (incl. renderer) stopped by explicit port/PID, GPU 0 MiB, `gpu_lock`
released (FREE), no deletions — rollouts + preds preserved.

## Reproduction
```
gpu_lock.sh acquire gate1-rollouts
setsid bash /workspace/gate1_master.sh   # renderer + REF-C base over /workspace/gate1_junc, --log-preds
CUDA_VISIBLE_DEVICES="" .venv/bin/python /workspace/gate1_postprocess.py   # -> gate1_rollouts.jsonl + summary
```
