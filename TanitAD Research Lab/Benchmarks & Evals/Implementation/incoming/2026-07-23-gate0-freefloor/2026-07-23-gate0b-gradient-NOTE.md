# Gate-0b — does the per-denoise GRADIENT NUDGE (cost-guided diffusion proper) collapse REF-C's junction off-road? — MEASURED 2026-07-23

**Follow-up to Gate-0.** Gate-0 found the free *selection* floor does NOT collapse off-road because it can
only pick among the fixed 128 FPS anchors and ~21 % of junction off-road moments have **no on-road anchor**
(the anchor-vocabulary ceiling). Gate-0b tests the **stronger, still-training-free** rung the coordinator
called for: the **per-denoise-step gradient nudge** — cost-guided *diffusion* proper — which **synthesizes
trajectories OUTSIDE the fixed anchor set** and can therefore escape that ceiling.

## ⚠️ FRAMING (mandatory)
WITHIN-SIM RELATIVE, on NuRec reconstructions (~3.2× OOD, RUN_RECIPE §13). Relative GRAD-vs-OFF ranking
trustworthy; absolute rates not real-world. Host = eval pod `tanitad-eval` (A40, free); pod1/2/3 untouched.
Lock `gate0b-gradient` (released at end). Control = same-session `gate0_off` (deterministic; reproduces the
baseline exactly, Gate-0).

## The gradient-nudge floor (still no training)
REF-C denoises `x = anchors + offset` over 2 truncated-diffusion steps. Gate-0b **replicates that exact
decode in the driver** (byte-identical when off) and, **after each denoise refinement AND for 4 extra
projection iters**, applies one gradient-descent step of the off-road energy
`E(x)=½·dist(x, lane_union)²`:  `x ← x + η·(q − x)`, where `q` = nearest point on the drivable lane union
(map frame; ego pose maps rig↔map). η=0.5, 4 extra iters. Then the SAME cost-guided selection + safety clamp
+ collision term (λ=5, μ=1.0, clamp 0.75 m) pick among the nudged anchors. Because the nudge moves `x` off
the anchor manifold, the final plan is NOT limited to the 128 anchors — the escape the selection floor lacked.
Implementation: `refc_floor_driver.py --floor grad --grad-eta 0.5 --grad-iters 4 --mu 1.0`.

## ⚠️ GRADIENT VALIDATED FIRST (same discipline that caught the frame bug) — `gate0b_gradient_validation.json`
1. **REPLICATION EXACT:** `_decode_nudged(η=0)` reproduces `model.forward(steps=2)` byte-for-byte
   (max|Δtraj|=0.0, max|Δconf|=0.0) — the nudge is a clean superset, no replication drift.
2. **GRADIENT TOWARD-DRIVABLE:** a KNOWN off-road trajectory (cost 12.34 m) is driven **monotonically to
   0.05 m** over 8 nudges — correct sign, and it reaches an on-road config that is NOT an anchor (the ceiling
   escape, demonstrated).
3. **COLLISION SIGN:** with μ=2 the selector avoids an anchor with an agent on it (μ=0 indifferent).
VERDICT: GRADIENT TRUSTWORTHY → proceed.

## 🎯 RESULT — GRAD (gradient-nudge floor) vs OFF (plain REF-C), REF-C-base, n=37 paired
Aggregator labels GRAD=`floor_on`, OFF control=`floor_off`. Control = same-session `gate0_off` (deterministic).
| category | n | OFF offroad | **GRAD offroad** | **ΔOFFROAD** | OFF→GRAD pass | OFF→GRAD caf | OFF→GRAD score |
|---|---|---|---|---|---|---|---|
| **intersection** | 7 | 0.71 | **0.71** | **+0.00** [0,0] | 1→1 | 0.71→**0.43** | 0.063→0.063 |
| **roundabout** | 8 | 0.75 | **0.75** | **+0.00** [0,0] | 2→2 | 0.00→0.00 | 0.101→0.100 |
| traffic_light | 6 | 0.17 | 0.17 | +0.00 [0,0] | 4→4 | 0.33→0.33 | 0.337→0.353 |
| highway | 8 | 0.38 | 0.25 | −0.125 [−0.375,0] | 3→3 | 0.38→0.38 | 0.221→0.221 |
| straight_other | 8 | 0.38 | 0.38 | +0.00 [0,0] | 5→5 | 0.12→0.12 | 0.430→0.431 |
| **JUNCTION (int+rbt)** | 15 | 0.73 | **0.73** | **+0.00** [0,0] | 3→3 | 0.33→**0.20** | 0.083→0.083 |
| **OVERALL** | 37 | 0.49 | **0.46** | **−0.027** [−0.081, 0] | 15→15 | 0.30→**0.24** | 0.229→0.232 |

Every junction scene has **identical** off-road status OFF→GRAD (not one of the 15 changed). Overall off-road
improves marginally (0.49→0.46, CI touches 0) — entirely from **highway** (0.38→0.25). At-fault collisions
DOWN (0.30→0.24; intersection 0.71→0.43) via the μ=1 collision term.

## MECHANISM — the nudge makes every PLAN on-road, yet the EGO still leaves the road at junctions
MEASURED (`gate0b_floor_GRAD.jsonl`, 1665 drives): with the nudge, **`base_off` and `sel_off` are ~0 in
EVERY category** (0 drives with a planned trajectory >0.5 m off-road — cf. Gate-0 selection's 96). The
validation already proved the nudge drives a known off-road trajectory to 0.05 m (escaping the fixed anchor
set). **So the planned trajectory is on-road at junctions — and the closed-loop junction off-road rate is
STILL 0.73, every scene identical.** The off-road excursions therefore come from the EGO's *executed* path
(the controller tracking on-road waypoints, transient junction-box crossing, covariate drift), NOT from the
planner's ability to *represent* an on-road trajectory. **Inference-time plan shaping — selection OR
gradient-synthesis — cannot fix a closed-loop/execution failure.**

## GRADIENT NUDGE vs the Gate-0 SELECTION floor — strictly better-behaved (deploy this one)
| | OFF (baseline) | Gate-0 SELECTION (on) | **Gate-0b GRADIENT (grad)** |
|---|---|---|---|
| junction offroad | 0.73 | 0.80 (WORSE) | **0.73 (no harm)** |
| roundabout offroad | 0.75 | 0.88 (broke fd3a49fa) | **0.75 (no break)** |
| overall offroad | 0.49 | 0.49 | **0.46** |
| overall plan_deviation | 0.33 | **0.76** (wild picks) | **0.34** (near-model) |
| intersection caf | 0.71 | 0.43 | 0.43 |
The gradient nudge keeps plans close to the model's (plan-dev 0.34 vs selection's 0.76), avoids the wrong-lane
roundabout break selection caused, and gives the same collision benefit — so as a **deployed safety floor it
dominates selection**. It just doesn't move junction off-road.

## VERDICT — ❌ NOT A PASS. Junction off-road SURVIVES the proper cost-guided-diffusion floor → Gate-1 (report plainly)
**Junction off-road does NOT collapse** (intersection 0.71→0.71, roundabout 0.75→0.75, junction 0.73→0.73,
ΔOFFROAD +0.000, every scene identical). The pre-registered PASS condition is **not met**. Crucially, this is
now a **stronger** result than Gate-0: the gradient nudge PROVABLY escapes the anchor-vocabulary ceiling
(validation 12.34→0.05 m; live `base_off`≈0 at junctions) yet junction off-road is unchanged — so the residual
is **NOT** a plan-representability limit. **Both inference-time levers (rungs 1-2: selection AND gradient
synthesis) are ruled out for junction off-road; the failure is closed-loop/execution.** Per the verdict doc,
**Gate-1 is warranted** — closed-loop-aware training (RoaD/CAT-K or analytic-gradient through the diff-WM), the
only family that shapes the EXECUTED trajectory, not just the plan. Junction-box-aware cost + kinematically
feasible projection are secondary refinements but unlikely to close a closed-loop gap on their own.

**Ship anyway:** the gradient-nudge floor is a free, strictly-better-behaved safety override (lower plan-dev
than selection, no roundabout regression, −6 pts at-fault collision, marginal off-road gain on highway).

## Caveats
1. WITHIN-SIM RELATIVE / ~3.2× OOD — relative GRAD-vs-OFF only. 480×854. NuRec reconstructions.
2. n per category 6-8 -> wide CIs; OVERALL + JUNCTION deltas are the powered signals.
3. Control = same-session gate0_off (deterministic; Gate-0 showed it reproduces the baseline exactly).
4. GRAD carries the collision term (μ=1.0) per the coordinator; off-road is the primary reading.
5. AlpaSim `offroad` = not-fully-in-a-lane AND touching-road-edge; the floor targets the same lane geometry.

## Deliverable manifest
| artifact | where | status |
|---|---|---|
| `gate0b_gradient_validation.json` | repo (staged) · pod | replication + gradient-sign + collision-sign (MEASURED) |
| `gate0b_gradient_results.json` | repo (staged) · pod | per-category GRAD-vs-OFF paired (MEASURED) |
| `refc_floor_driver.py` (adds `--floor grad`) | repo (staged) · pod | the gradient-nudge floor |
| `gate0b_run.sh`,`gate0b_master.sh`,`gate0b_validate.py` | repo (staged) · pod | run + validation harness |
| `gate0b_floor_GRAD.jsonl` (per-drive nudge diagnostics) | repo (staged) · pod | mechanism |
| rollouts (38), USDZs (38) | **pod only** `/workspace/gate0b_grad`, sceneset 986fec83 | regenerable |

**Pod left CLEAN:** services killed by port, `gpu_lock` released, GPU idle.
