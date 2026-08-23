# Flagship v1 vs REF-C base — NATIVE 1080×1920 paired re-run — MEASURED 2026-07-23

**Resolves the resolution/environment confound flagged in the 480×854 suite** (Sayed's "data vs
environment vs model?" — resolution is the *environment* axis). Same 12 scenes, same
`vs_suite_master_1080.sh`→`vs_suite_run.sh`, same drivers/ckpts — **only the camera render resolution
changed 480×854 → native 1080×1920** (the resolution the n=1 flagship *pass* used). Both models'
canonicalization self-checked **f_eff=266.0 (== F_REF) OK** at native res (vs 265.6/265.7 at 854; native
`sx=sy=1.0`, no intrinsics down-scaling). Raw: `flagship_vs_refc_native1080_results.json` (per-scene UUIDs)
+ `vs_{flag,refc}_1080_results-summary.json`. Lock `vs-native1080` released; pod clean.

## 🎯 ANSWER — the delta HOLDS at native res. **Model, not environment.**
REF-C base still **decisively beats** flagship v1 closed-loop at full resolution. Flagship does modestly
*better* at native res (a **minor** resolution sensitivity — the 854 suite slightly overstated its deficit),
but nowhere near closing the gap, and the direction/significance never flip. **Flagship v1's tactical head is
the worse closed-loop planner, resolution-robustly. The n=1 win was a lucky scene, not a resolution effect.**

## Side-by-side (n=12, one rollout/scene, on NuRec reconstructions)
| metric | **854 flag** | **854 refc** | **1080 flag** | **1080 refc** | Δ@854 | Δ@1080 |
|---|---|---|---|---|---|---|
| at-fault collision | 0.167 | 0.167 | 0.250 | 0.250 | 0.000 | 0.000 (TIED both) |
| offroad | 0.667 (8/12) | 0.167 | **0.500 (6/12)** | 0.167 | +0.500 | +0.333 |
| pass rate | 2/12 | 8/12 | **3/12** | 7/12 | −0.500 | −0.333 |
| mean score | 0.066 | 0.496 | **0.115** | 0.410 | **−0.430** | **−0.295** |
| mean dist-to-GT (m) | 1.805 | 1.874 | 1.957 | 1.751 | −0.069 | +0.206 |

## Paired flagship − REF-C (the clean within-sim signal), 854 vs native
| paired stat | **480×854** | **native 1080×1920** |
|---|---|---|
| mean score delta | −0.430, boot95 **[−0.646, −0.215]** | **−0.295, boot95 [−0.494, −0.117]** (excludes 0) |
| score sign test (flag vs refc better) | 0 vs 8, **p=0.008** | 0 vs 7 (5 ties), **p=0.016** |
| pass McNemar (flag>refc / refc>flag) | 0 / 6, **p=0.031** | 0 / 4 (both_pass 3, both_fail 5), p=0.125 |
| at-fault-collision McNemar | 1 / 1, p=1.0 (tied) | 1 / 1, p=1.0 (tied) |
| dist-to-GT delta | −0.069 [−0.923, 0.752] (tied) | +0.206 [−0.473, 0.919] (tied) |

## Reading (honest, both directions committed)
- **The score delta HOLDS** (−0.430 → −0.295; **both CIs exclude 0**; sign test significant at both res).
  Flagship passes **0** scenes REF-C fails, at **both** resolutions. → the primary conclusion is
  **resolution-robust**: flagship's WM+tactical-policy is the worse closed-loop planner here, not an
  artifact of the 854 downscale.
- **Minor resolution sensitivity, in flagship's favor** (partially confirms the "854 understated it"
  hypothesis, but only weakly): at native res flagship's offroad drops 8→6/12, pass rises 2→3/12, mean
  score 0.066→0.115, and the paired deficit shrinks 30 % (−0.430→−0.295). So full res helps flagship's
  frozen encoder a little — but the residual gap is still large and significant. Resolution is a
  **second-order modifier, not the explanation.**
- **The pass-McNemar drops below significance at native (p=0.125)** only because there are fewer discordant
  pairs (4 vs 6) — both_fail rose to 5 as native's harder-rendered scenes failed both models. The
  **continuous score signal (mean-delta CI + sign test) stays significant**; at n=12 the binary McNemar is
  the underpowered test, not the load-bearing one. Do not over-read the p=0.125.
- **REF-C shifts slightly too** (score 0.496→0.410, pass 8→7/12) — within its known run-to-run diffusion
  variance (854-fresh 0.496 / §12-prior 0.345 / native 0.410, all ~0.35–0.50). The gap narrows from both
  sides but never closes.
- **At-fault collision is TIED at both resolutions** (2/12→3/12 each) — flagship's n=1 collision-avoidance
  never generalized; the difference is entirely flagship's offroad departures (high plan-deviation swerve).

## ⚠️ Framing (unchanged, mandatory)
WITHIN-SIM RELATIVE, **not** a real-world rate. Both models see the SAME NuRec-reconstruction input, ~3.2×
more OOD than REF-C's real-footage training (open-loop ADE ~1.5 on these reconstructions vs 0.4728 on real
PhysicalAI val, §13). The paired delta isolates the **planner** (WM+tactical-policy vs open-loop diffusion);
it does not give either model's real-world closed-loop rate. Now confirmed at BOTH 854 and native res.

## Deliverable manifest
| artifact | where | status |
|---|---|---|
| `flagship_vs_refc_native1080_results.json` | repo incoming (staged) | ⭐ native paired result + per-scene UUIDs (MEASURED) |
| `flagship_vs_refc_native1080_NOTE.md` (this) | repo incoming (staged) | side-by-side + verdict |
| `vs_flag_1080_results-summary.json` / `vs_refc_1080_results-summary.json` | repo incoming (staged) | raw native runtime aggregates |
| `vs_suite_master_1080.sh` | repo incoming (staged) · pod `/workspace/` | native re-run orchestrator (res flip only) |
| `vs_aggregate.py` (now res-parametrized) | repo incoming (staged) · pod | shared paired-stats script |
| native rollouts (`.asl`, metrics) | **pod only** `/workspace/vs_{flag,refc}_1080/rollouts/` | regenerable via the scripts |

**Pod left CLEAN (MEASURED):** master killed all services incl. renderer, `gpu_lock released by
vs-native1080`, GPU 0 MiB / compute-procs [], lock state=FREE. **No orphan.**

## ESCALATE (integration)
The two paired suites (854 + native) together **settle** the flagship-vs-REF-C closed-loop question for the
registry/leaderboard: **REF-C base > flagship v1 closed-loop on NuRec reconstructions, at both resolutions
(mean-score delta −0.43 / −0.30, both CIs exclude 0); collisions tied; the n=1 flagship pass was a lucky
scene (C5), not resolution.** Recommend: fold into MODEL_REGISTRY / LEADERBOARD §5.5 + the C5
RETRACTION_LOG entry (correcting the n=1 headline). Remaining unknowns are now only the OOD/sim2real axis
(§13, the ~3.2× reconstruction gap) — a real-footage closed-loop harness, not a resolution re-run.
