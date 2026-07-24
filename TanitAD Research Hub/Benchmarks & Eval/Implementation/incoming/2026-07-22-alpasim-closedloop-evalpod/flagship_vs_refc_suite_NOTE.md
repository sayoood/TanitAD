# Flagship v1 vs REF-C base — PAIRED closed-loop suite (n=12) — MEASURED 2026-07-23

**On NuRec reconstructions (AlpaSim, tanitad-eval A40).** Resolves the n=1 caveat from RUN_RECIPE §14.
Raw: `flagship_vs_refc_suite_results.json` (per-scene rollout UUIDs) · `vs_flag_results-summary.json`
+ `vs_refc_results-summary.json` (runtime aggregates) · scripts `vs_suite_master.sh`,`vs_suite_run.sh`,`vs_aggregate.py`.

## 🎯 VERDICT — the n=1 was a LUCKY SCENE. Flagship v1 does NOT beat REF-C closed-loop; REF-C base statistically WINS.

On the SAME 12 scenes (one rollout each, both models fed the identical NuRec renders, f-theta canon
verified per run), **REF-C base beats flagship v1**: pass **8/12 vs 2/12**, mean score **0.496 vs 0.066**.
The n=1 scene (`01d503d4`, RUN_RECIPE §14, where flagship passed and all REF-C collided) is **NOT in this
suite** — so this is a clean out-of-sample test, and it **reverses** the n=1 read. The n=1 "flagship beats
REF-C" was the **C5 pattern** (a scalar off a scene-dependent result) the coordinator flagged.

## ⚠️ FRAMING (mandatory) — WITHIN-SIM RELATIVE, not absolute
Both models see the **same NuRec-reconstruction input**, which is **~3.2× more OOD** than REF-C's real-footage
training (open-loop ADE **1.47 on these reconstructions** vs 0.4728 on real PhysicalAI val, RUN_RECIPE §13).
The paired design controls for scene + reconstruction fidelity, so the flagship−REF-C delta isolates the
**PLANNER** (WM+tactical-policy vs open-loop anchored-diffusion). It does **NOT** give either model's
real-world closed-loop rate. Everything below is **"on NuRec reconstructions."**

## Method (MEASURED)
- Reused the staged AlpaSim setup (RUN_RECIPE §10–14); did **not** rebuild. Renderer serving the 12-scene
  sceneset on :6011; each model = its own clean logdir + controller/driver/physics; runtime one rollout/scene.
- **flagship v1** = `flagship_v1_driver.py` (tactical-policy path: `encode_window → strategic_policy →
  tactical_policy["waypoints"]{5,10,15,20}`, NO future actions), ckpt `/root/models/flagship-30k/ckpt.pt`
  step **29999** (= HF `Sayood/tanitad-flagship-4b-speedjerk`; strict load 0/0, smoke-verified).
- **REF-C base** = `refc_driver.py`, ckpt `/root/models/refc-base-30k/ckpt.pt` step 29999 (104.2M, 128 anchors).
- **Scenes (n=12, exact, no truncation):** the `public_2601` 26.04 subset used by the prior REF-C suite —
  `00040136, 000525f6, 000548db, 00064c58, 0009402a, 00097de1, 000a3a34, 000a74ae, 000e95f7, 000ff49d,
  0010ce77, 001564ce` (clipgt-…). **480×854**, f-theta canon **f_eff=265.7 (flag) / 265.6 (refc), both OK ≈266**.
- Both share `run_uuid a74ede75…` (config-inherited run_metadata); **per-rollout `rollout_id` is the unique ID**
  (in the JSON per_scene).

## Aggregate (n=12, on NuRec reconstructions, 480×854)
| metric | **flagship v1** | **REF-C base** | flag − refc |
|---|---|---|---|
| at-fault collision rate | 0.167 (2/12) | 0.167 (2/12) | **0.000 (TIED)** |
| offroad rate | **0.667 (8/12)** | 0.167 (2/12) | **+0.500 (worse)** |
| pass rate | **0.167 (2/12)** | **0.667 (8/12)** | **−0.500** |
| mean score | **0.066** | **0.496** | **−0.430** |
| mean dist-to-GT (m) | 1.805 | 1.874 | −0.069 (tied) |
| mean plan_deviation | **1.125** | 0.342 | +0.783 (3.3× wider) |
| mean dist_traveled (m) | 71.3 | 115.2 | −44 (departs earlier) |

## Paired flagship − REF-C (the clean within-sim signal)
- **mean score delta −0.430**, scene-cluster **boot95 [−0.646, −0.215]** (excludes 0 → flagship worse).
- mean dist-to-GT delta −0.069, boot95 **[−0.923, +0.752]** (spans 0 → **tied** on GT-deviation).
- **pass (McNemar): REF-C passes 6 scenes flagship fails; flagship passes 0 scenes REF-C fails** (both_pass 2,
  both_fail 4) → two-sided **p=0.031**.
- **score sign test: REF-C strictly better on 8/12, flagship on 0/12** (4 ties) → **p=0.008**.
- **at-fault collision (McNemar): 1 vs 1** (both_collide 1, neither 9) → **p=1.0 (TIED)**.

## Why (MEASURED mechanism, not just a story)
Flagship v1's tactical policy is a **high-deviation planner** (plan_dev 1.12 vs REF-C 0.34; both drivers share
the identical rig→world trajectory conversion — only `policy.plan()` differs, so this is the model, not the
adapter). It **swings wide**. On the one wide-highway n=1 scene that wide swerve *avoided* a collision (looked
great, n=1). Across a representative suite the same behavior **drives it off-road 8/12** (its failure mode is
**offroad, not collision**). Collisions are tied — flagship's n=1 collision-avoidance **does not generalize**;
it trades collisions for road departures. Flagship's tactical head is **deterministic** in eval, so this is a
stable outcome (not sampling noise).

## Caveats / threats to validity (honest)
1. **WITHIN-SIM RELATIVE, ~3.2× OOD** — not a real-world number for either model.
2. **Resolution (residual confound) — NOW RESOLVED.** Suite is **480×854**; the n=1 flagship *pass* was native
   **1080×1920**. The native-1080×1920 paired re-run (`flagship_vs_refc_native1080_NOTE.md`, RUN_RECIPE §16)
   **confirms the delta HOLDS**: mean-score delta **−0.295 boot95 [−0.494, −0.117]** (still excludes 0), sign
   test 7–0 for REF-C (p=0.016), flagship passes 0 scenes REF-C fails. Flagship does *modestly* better at full
   res (offroad 8→6/12, pass 2→3/12, deficit shrinks 30 %) → a **second-order** resolution sensitivity that
   slightly overstated flagship's deficit at 854, but **does not reverse the conclusion**. **Model, not
   environment.**
3. **n=12, wide binomial CIs** (a 2/12 or 8/12 rate is broad). The **mean-score paired delta (CI excludes 0)**
   and the **McNemar/sign tests** are the load-bearing signals, not the raw rates.
4. **REF-C base absolute varies run-to-run:** this fresh run **0.496 / 8-pass** vs the prior staged suite
   **0.345 / 6-pass** (§12) — likely REF-C's **2-step diffusion sampling** (unseeded) + render state.
   The **paired within-session delta (−0.43)** is far larger than this variance, and flagship passes **0**
   scenes REF-C fails, so the conclusion is robust to REF-C's noise.
5. Both are **open-loop-trained** planners (neither closed-loop-trained); flagship's differentiator is its
   higher deviation, not a closed-loop-training gap.

## Deliverable manifest
| artifact | where | status |
|---|---|---|
| `flagship_vs_refc_suite_results.json` | repo incoming (staged) | ⭐ combined paired result + per-scene UUIDs (MEASURED) |
| `flagship_vs_refc_suite_NOTE.md` (this) | repo incoming (staged) | the write-up |
| `vs_flag_results-summary.json` / `vs_refc_results-summary.json` | repo incoming (staged) | raw runtime aggregates (12 rollouts each) |
| `vs_suite_master.sh` / `vs_suite_run.sh` / `vs_aggregate.py` | repo incoming (staged) · pod `/workspace/` | reproducible orchestrator + paired stats |
| runtime rollouts (`.asl`, metrics) | **pod only** `/workspace/vs_flag`,`/workspace/vs_refc/rollouts/` | regenerable via the scripts |

**Pod left CLEAN (MEASURED):** master killed all services incl. renderer, `gpu_lock released by flagship-vs-refc`,
GPU 0 MiB / compute-procs [], lock state=FREE. **No orphan.**

## ESCALATE (integration — do not let this sit)
1. **Correct the live state.** The n=1 "flagship v1 beats REF-C closed-loop" headline (LOOP_STATE / chat /
   LEADERBOARD §5.5) must be updated: **at n=12 REF-C base beats flagship v1 (pass 8/12 vs 2/12, score
   0.496 vs 0.066); collisions tied; n=1 was a lucky wide-highway scene.** Append a **C5** RETRACTION_LOG entry.
2. **Decision needed:** run the **native-1080×1920 paired re-run** (caveat #2) before this becomes a
   registry headline? It is the one unresolved confound between "flagship is a worse closed-loop planner"
   and "flagship is resolution-sensitive." Cheapest discriminating experiment; needs a GPU-lock go.
