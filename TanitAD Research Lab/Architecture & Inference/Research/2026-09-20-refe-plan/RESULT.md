# Stage 0 — the released DriveZero teacher runs closed-loop on our dev box

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`
**Date:** 2026-09-20 · **Box:** dev box, RTX 4060 (8.6 GB) · **Author:** Research Lab agent
**Scope (PI 2026-09-20):** REFe is reproduced on **THEIR data** (nuPlan / NAVSIM). Nothing here
touches the TanitAD corpus. Anchor table = DriveZero's own DINOv3 ViT-S row, **93.88 PDMS**.

## What this stage had to prove

That we can run *their* released artifact in *their* environment and read *their* numbers —
before a single line of REFe is written. It is the cheapest possible falsifier of the whole plan:
if the published checkpoint cannot be made to drive here, every later stage is speculation.

## MEASURED

| fact | value | artifact |
|---|---|---|
| DriveRL teacher parameter count | **5,702,413** | `raw/dz_budget.txt` (measured from `checkpoint_2400.pt`, not quoted) |
| critic output channels | **9** (decomposed) | same |
| action head | **Beta**, 2 dims (jerk_long, lat_command) | same |
| renderer validated on real data | **149 frames, 990×990 @ 5 fps** | `media/teacher_mini_485e78d3d4035b52.mp4` |
| per-step policy outputs persisted | jerk_long, lat_command, raw Beta sample, accel/steer control, goal_points, road_graph | `DriveRLActionTrajectory` in every `SimulationHistorySample` |
| closed-loop score, val14_nr, 8 mini scenarios | **97.19**, `successful=8 failed=0` | `raw/stage0_teacher_run.log`, aggregator parquet |

⚠️ **Evidence class.** The parameter count and the render are **MEASURED (ours)**. The 93.88 anchor
is **PUBLISHED (cited)**. No number in this file is INHERITED.

## The run, and what its decomposition says about the teacher

**MEASURED 2026-09-20 12:17, run #5:** `RESULT task=val14_nr score=97.19 successful=8 failed=0 total=8`.
nuPlan **mini**, `one_of_each_scenario_type`, limit 8, non-reactive background, no TTS, sequential
worker, RTX 4060. Wall clock **15m12s** for 8 scenarios (~110 s each incl. metric computation).

⛔ **This is NOT comparable to their published 93.61.** Different tokens (mini ≠ val14), n = 8, one
scenario per type, single inference seed, no replicate. It is a **harness-health number**: it says
the released policy drives and scores, not how well. No interval is quoted because none is earned —
and on a stochastic planner a separated interval would still answer only *"would another draw of
scenarios say this?"*, which is not the question anyone wants answered here.

### Per-scenario, from the aggregator parquet (their own metric engine, not our arithmetic)

| scenario type | score | progress along expert route | speed-limit compliance |
|---|---|---|---|
| stopping_at_stop_sign_with_lead | **1.0000** | 1.000 | 1.000 |
| following_lane_with_lead | 0.9996 | 1.000 | 0.998 |
| near_multiple_vehicles | 0.9992 | 1.000 | 0.997 |
| accelerating_at_traffic_light_without_lead | 0.9974 | 1.000 | 0.990 |
| starting_protected_noncross_turn | 0.9923 | 1.000 | 0.969 |
| starting_left_turn | 0.9661 | 0.956 | 0.919 |
| starting_unprotected_cross_turn | 0.9461 | 0.934 | 0.867 |
| **changing_lane_to_left** | **0.8746** | **0.613** | 0.982 |
| **final_score** | **0.9719** | 0.938 | 0.965 |

⭐ **The informative part is what is CONSTANT.** Across all 8 scenarios these read exactly **1.0**:
`no_ego_at_fault_collisions`, `drivable_area_compliance`, `driving_direction_compliance`,
`ego_is_comfortable`, `ego_is_making_progress`. Every point the teacher loses comes from just two
continuous terms — **progress along the expert route (0.938)** and **speed-limit compliance (0.965)**.

⇒ On this sample the released teacher is **safe, legal and comfortable**, and every point it loses
comes from progress and speed-limit compliance.

⛔ **CORRECTION, same session, before this left the package — "and slow" was WRONG.** I wrote that
from the suite mean (progress 0.938) and it does not survive the per-scenario census
(`raw/stall_census_m-nr-n.txt`, `code/stall_census.py`). Measured path length, ego vs the human
expert, over all eight:

| scenario type | ego / expert path | stalled | ends at |
|---|---|---|---|
| changing_lane_to_left | **0.617** | 36 % | **0.03 m/s** |
| starting_unprotected_cross_turn | 0.910 | – | 4.50 m/s |
| starting_left_turn | 0.965 | – | 10.02 m/s |
| near_multiple_vehicles | 1.068 | – | 5.05 m/s |
| accelerating_at_traffic_light_without_lead | 1.083 | 16 % | 13.06 m/s |
| stopping_at_stop_sign_with_lead | 1.091 | 86 % | 0.00 m/s |
| following_lane_with_lead | 1.107 | 9 % | 13.26 m/s |
| starting_protected_noncross_turn | 1.187 | – | 8.11 m/s |

**The teacher travels FURTHER than the human expert in four of eight scenarios**, and ≥ 0.91 in six.
It is not a slow policy. **The suite's entire progress deficit is one scenario.**

⭐ **Root-cause class of my error: a suite MEAN read as a property of the policy, when it was one
outlier.** The harness clamps per-scenario progress at 1.0, so the four scenarios that over-travel
contribute exactly 1.0 and cannot offset the 0.613 — the mean is structurally unable to show that
the distribution is one bad case and seven good ones. Same family as the `df` / cgroup / `step_s`
traps: **a true number quoted at the wrong scope.** The fix was not more care, it was computing the
per-item quantity, which cost one CPU-only probe.

⭐ **And the census DISCRIMINATES, which the mean also could not.** Stalling is common and usually
**correct**: `stopping_at_stop_sign_with_lead` stalls 86 % of its steps and scores **1.0000** — it is
a stop-sign scenario; `accelerating_at_traffic_light_without_lead` and `following_lane_with_lead`
stall early (last stalled step 23 and 13) and then reach 13 m/s. **Exactly one scenario stalls and
never resumes with a clear road**, and that WAS the finding in `FINDING_TEACHER_STALL.md`.

⛔⛔ **RETRACTED the same day, 13:1x, by its own queued control.** Re-run with **reactive (IDM)**
background agents instead of log replay, that same scenario drives **59.8 m with 0 % stalled steps
and scores 97.60**, against 39.5 m / 36 % / 87.46 non-reactive. **The stall is a non-reactive replay
artifact, not a policy defect:** in log replay the background vehicles execute their recorded paths
regardless of the ego, so the teacher was driven into a 0.5 m lead gap that an IDM agent never
creates. Full account in `FINDING_TEACHER_STALL.md` §0 and in `Project Steering/RETRACTION_LOG.md`.

⚠️ **The census table above remains valid as a NON-REACTIVE measurement** and is not withdrawn — but
it must carry its protocol. Under the reactive protocol the worst path ratio is instead
`following_lane_with_lead` at **0.783**, and only 3 of 8 scenarios stall rather than 4
(`raw/stall_census_m-r-n.txt`). ⇒ **a per-scenario census is a statement about a PROTOCOL, not about
a policy**, which is the same scope error one level down from the mean-vs-outlier correction above.
⚠️ **Do not read this as a defect in their teacher.** n = 1 per type. The correct use of this table
is to pick where DZ-10/DZ-11 should look first (lane changes and unprotected turns), not to grade
anyone. It is also the pre-registered reason to expect **goal augmentation** to matter: a goal
further along an alternative route is precisely the intervention that moves a progress term.

## The five faults between "clone the repo" and "it drives"

Each run got exactly one layer deeper. All five are environment faults, none is a defect in their
code, and every one of them is a trap that would have cost the next person the same hours.

1. **TLS proxy** — `uv pip install` dies `UnknownIssuer`; `--native-tls` fixes it. `curl` needs
   `--ssl-no-revoke`.
2. **Windows MAX_PATH on the clone** — "Filename too long" in the scratchpad path. Re-cloned at
   `C:/Users/Admin/dz` with `core.longpaths=true`.
3. **`fcntl` is Unix-only** — their preflight imports it. A no-op `flock` shim in the venv's
   `site-packages` (warns once, refuses off-Windows) → `PREFLIGHT_OK`.
4. **D: is exFAT and cannot be reached by link.** `mklink /J` from C: into D: succeeds, reports
   `isdir=True`, and lists **EMPTY** to `os.listdir`, Python `glob`, `cmd dir` and bash `ls` alike —
   the devkit then said *"No log files found!"* and simulated 0 scenarios. exFAT holds no symlinks
   either. **Fix pattern: make the tool point at a plain directory on D:** (same-volume rename is
   instant), never at a link. Their runner only needs `$DRIVERL_EVAL_DB_LINK_ROOT/driverl_val14`
   to *exist*, so a plain dir satisfies it.
   ⚠️ A second costume of the same trap: `MSYS_NO_PATHCONV=1` exported for a whole run makes native
   Python resolve `C:\c\Users\…`. Scope it to the single `cmd.exe` line that needs it.
5. **Windows MAX_PATH again, at the simulation-log write** — and this one is the interesting fault,
   because it presents as a *successful* run that aborts near the end.

## Fault 5 in full, because it will recur on every split we add

Run #4 simulated **5 of 8** scenarios, wrote **4** simulation logs, and then died:

```
FileNotFoundError: [Errno 2] No such file or directory:
'C:\Users\Admin\dz\out\mini\notts_nr\mini-nr-l8-notts\closed_loop_nonreactive_agents_driverl_val14_nr
 \simulation_log\DriveRLNuPlanPlanner\accelerating_at_traffic_light_without_lead
 \2021.05.12.23.36.44_veh-35_01133_01535\99ca544752f255ad\99ca544752f255ad.msgpack.xz'
RuntimeError: Simulation failed
```

**MEASURED: that path is 263 characters; the usable limit is 259.** `FileNotFoundError` is a
misleading messenger — the directory exists and the parent was created successfully; only the
*file* open exceeded MAX_PATH. Exactly one scenario type was over budget, so four scenarios wrote
their logs normally and the run looked healthy until `exit_on_failure=true` (THEIR setting, kept)
aborted the suite.

The devkit composes a **fixed** tail under the output dir
(`simulation_log_callback.py:147` → `output/planner/scenario_type/log_name/token/token.msgpack.xz`):

| segment | chars | worst case |
|---|---|---|
| `closed_loop_nonreactive_agents_driverl_val14_nr` | 47 | **55** (`test14_random_nr` in the six-task suite) |
| `simulation_log/DriveRLNuPlanPlanner` | 35 | 35 |
| scenario_type | 42 (the one that failed) | **54** (`starting_straight_traffic_light_intersection_traversal`) |
| log_name | 38 | 38 |
| token twice + `.msgpack.xz` | 45 | 45 |
| **tail total** | 211 | **231** |

⇒ **the prefix budget is 259 − 231 = 28 characters.** The old prefix was **52** and could never have
held it. Output now lives at `C:/dzo/<run_id≤7>` = **14**, leaving 14 characters of headroom for the
full six-task suite. This is why the run id is `m-nr-n` and not `mini-nr-l8-notts`.

⚠️ **The general form, and the reason this is written down:** on Windows, an artifact's path budget
belongs to the **consumer that composes the deepest path**, not to the directory you chose. Pricing
it from the root you created reads exactly like an answer and is off by the length of a tail you
never looked at. Same family as the `df` / cgroup / `step_s` scope traps.

## What the video shows

`media/teacher_mini_485e78d3d4035b52.mp4` — `following_lane_with_lead`, token `485e78d3d4035b52`,
log `2021.06.07.12.54.00_veh-35_01843_02314`, 149 frames.

Drawn from the simulation log alone: map lanes, **the road graph the policy itself consumed**
(cyan, read back out of `debug_info.model_input.road_graph` — not our reconstruction of the map),
other agents, ego, the teacher's plan (green), the human expert log (dashed), mission goal and the
policy's own goal anchors. The overlay carries the real per-step outputs, including the raw Beta
sample before it becomes accel/steer.

⚠️ **What it does NOT show, stated because its absence is informative:** reward and value
components. Those live in the DriveRL sidecar, and a nuPlan `SimulationLog` does not carry them —
the overlay says so on every frame rather than leaving a blank. Value-guided TTS fields appear only
when `DRIVERL_EVAL_TTS_ENABLED=1` (DZ-10).

## What this does NOT establish

* **Not a DriveZero Table 3 number.** Mini tokens are not val14/test14 tokens; 8 scenarios are not
  a benchmark. No number here is comparable to a published row, and none is quoted as one.
* **Not evidence about REFe.** This is their teacher, privileged inputs, no camera encoder.
* **No interval.** 8 scenarios, one inference seed, no replicate. Under the programme's own rule a
  separated CI would still answer only *"would another draw of scenarios say this?"* — and we have
  not earned even that.

## DZ-10 concluded the same day — the TTS sweep, and what it found instead

| | N=1 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| suite score | 97.19 | **97.30** | 97.28 | 97.12 | **97.03** |
| switch rate | – | 2.01 % | 2.68 % | 4.78 % | 5.20 % |

⛔ **Their monotone gain (+0.11 → +0.56) does NOT reproduce on this sample**; our curve peaks at
N = 8 and falls thereafter. The full mechanism — a critic whose value gaps (~0.005) are far below its
own 0.03 switch margin, so selection is dominated by estimation error that **systematically
over-values braking** — is in `FINDING_TEACHER_STALL.md` §8–§12, with the pool-vs-selection control
that isolates it.

⚠️ **Not a claim that TTS hurts DriveZero.** Eight scenarios, one seed, six of nine score terms
saturated at 1.0. The mechanism numbers (1,193 steps/arm) are well powered; the score deltas are not.
The clean test is DZ-11, where those terms have real headroom.
## Next levers, in order (rewritten 2026-09-20 — the earlier list had gone stale)

⚠️ **What the previous version of this list said, and why it was wrong.** It put the TTS sweep at
(2) and "Stage 1 goal augmentation, then Stage 2" at (3). **DZ-10 concluded the same day** and its
finding is in §"DZ-10 concluded" above; **Stage 1 is done** (`STAGE1_GOAL_AUGMENTATION.md`); and
**Stage 2's scorer half is built and gated** (`REFE_MODEL.md` §8.1–8.2). A next-lever list is the
part of a document a reader acts on, so a stale one is worse than none.

1. **DZ-11 headline suite** on test + val — still the only run that produces numbers comparable to
   their table. **Gated on the extraction**, which is running with per-member CRC: test downloaded
   and verified at 95,919,476,643 B, 1,349 databases, 162.3 GB uncompressed; val downloading.
   Guards already proven by mutation: `code/run_headline_suite.sh`.
2. **Finish the scorer target bank and retrain with real coverage.** The first wiring ran at
   **14.6 %** coverage because the bank was 47 frames against 1,964 trajectory tuples. The builder
   now takes `--frame-offset` and the loader reads a directory of shards, so parallel passes
   densify it — ⚠️ **held back on memory, not on design**: 6.7 GB free of 31.8 GB while the
   extraction and a sibling stream are live.
3. **Read `assign_d` per candidate.** The banked targets belong to CANDIDATE trajectories and are
   attached to the model's nearest proposal, so that distance bounds the whole supervision. Its
   floor is set by the candidate set, not by model quality — `teacher` and `lat±2` should converge,
   `over-curb` and `stopped` should not. A pooled mean cannot tell those apart.
4. **Stage 3 training at scale** — needs the A40. ⛔ **PI signal not yet sent**: the dev box is
   still producing the bank and the local validation, and there is nothing for a pod to do until
   they are done.
5. **Stage 4 TanitVFM**, unchanged, per `REFE_PLAN.md`.

⛔ **Not on this list, deliberately:** online scoring of the student's own 64 proposals, which is
DriveZero's actual method. It is a full calculator rollout per proposal per optimisation step and
this rig cannot pay it. The precomputed basis is a stated difference, not an oversight, and any
scorer number must be reported with it.
