# Q1 — Community closed-loop benchmarks: what they measure, what they cost us, and which ONE to adopt

**Date:** 2026-08-03 · **Stream:** Benchmarks & Eval (research) · **Author:** benchmarks/render research agent
**Evidence classes:** `MEASURED (ours + artifact path)` · `PUBLISHED (cited)` · `INHERITED (another of our
docs, NOT re-verified)` · `ESTIMATED` · `HYPOTHESIS`.
**Scope note:** I own this directory. I did **not** edit the renderer stream's files
(`stack/experiments/alpasim-gsplat/`, `stack/experiments/nurec-gsplat/`), REF-C's code or the planner
research. Every integration item that touches those is written as an escalation, not a patch.

---

## 0. The answer in five sentences

1. **Adopt AlpaSim / the AlpaSim E2E Closed-Loop Challenge 2026 first.** It is the only candidate that is
   genuinely closed-loop, sensor-in-the-loop, **on our own camera rig and our own data distribution**, has
   a live public leaderboard in 2026, and is already ~80 % integrated here (MEASURED: n=12 and n=37 suites
   have run, driver adapters exist and self-check their focal length).
2. **Its score cannot diagnose anything.** MEASURED from our own run artifact
   (`REFC_suite_base_results.json:score_criteria`): `pass = collision_at_fault == 0 AND offroad == 0`, and
   `score = progress_score = min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1.0)`. It is *progress gated by
   catastrophe*. It contains **no** heading, curvature, yaw-rate, manoeuvre or route term.
3. **That is exactly why it combines with our four-family panel instead of replacing it** — layer 1 buys
   external comparability, layer 2 buys diagnosis, and the composition gap between them is measurable, not
   rhetorical (§4).
4. **Geometry is not the blocker people assume.** The computed matrix (`bench_geometry.json`, produced by
   `bench_geometry_check.py` in this directory) says v5f's 120° cylinder is a **3.00× down-sample of one
   single camera** on the PhysicalAI/NuRec rig with **zero stitch error**, and is **exact in CARLA too**
   because CARLA lets us declare co-located cameras. The rigs where geometry *does* bite are the log-replay
   ones (NAVSIM/nuScenes): 3 cameras, physically separated optical centres, **~18–21 target-pixel stitch
   error at 5 m** — i.e. worst precisely in the near field where cut-ins live.
5. **The real blocker for v5f is not geometry, it is that v5f has no usable checkpoint** (MEASURED: step
   2,300 and degrading, `Project Steering/Reports/2026-08-03-1300-program-report.md` line 20/23). The
   0-GPU work — the v5f driver adapter and the cylindrical resample — should be written now so the eval is
   one command on the day a checkpoint exists.

---

## 1. The field, benchmark by benchmark

Columns: **CL?** = genuinely closed-loop (the policy's action changes the next observation) vs a
log-replay/pseudo proxy. **Interval?** = does the benchmark itself publish an uncertainty.

| benchmark | CL? | observation the policy gets | score | interval? |
|---|---|---|---|---|
| **AlpaSim** (NVIDIA, NuRec) | **YES** — neural-reconstruction sensor sim, ego pose integrates | rendered camera(s) from a NuRec 3DGS reconstruction of a real 20 s clip | `progress_score` gated on `collision_at_fault==0 ∧ offroad==0` (MEASURED, our artifact) | schema has `*_std`, **all `None` in our runs**; NVIDIA publishes **0.73 ± 0.01** over 910 scenarios (PUBLISHED) |
| **NAVSIM v2** | **NO — pseudo-simulation** | 8 cameras @1920×1080 + merged LiDAR, 2 Hz, ≤3 history frames; outputs a 4 s trajectory (PUBLISHED) | **EPDMS** = multiplicative {NC, DAC, DDC∈{0,½,1}, TLC∈{0,1}} × weighted mean {EP 5, TTC 5, LK 2, HC 2, EC 2}; two-stage aggregation with a Gaussian kernel over follow-up scenes (PUBLISHED, navsim `docs/metrics.md`) | **no** ("No uncertainty intervals are reported in this documentation") |
| **nuPlan** (CLS-NR / CLS-R) | **YES** for planning | ⛔ **tracks + map, no renderer** — a vision-only policy cannot be run | 0–100; rule violations + **human-similarity terms (longitudinal velocity error, longitudinal stop-position error, lateral position error)** + dynamics/comfort + progress (PUBLISHED) | **no** |
| **Bench2Drive** (CARLA) | **YES** | whatever the agent declares (CARLA sensors) | DS = RC × IP; SR; **per-skill Skill Score** over 44 interactive scenarios; 220 routes ≈150 m (PUBLISHED) | **no**, but claims lower σ than the official protocol and argues single-seed is admissible (PUBLISHED) |
| **CARLA Leaderboard 2.0** | **YES** | agent-declared; ≤8 RGB cameras (competition) / 4 (qualifier), ≤2 LiDAR, ≤4 radar (PUBLISHED) | DS = RC × IP; penalties 0.50 pedestrian / 0.60 vehicle / 0.65 static / 0.70 red light / 0.80 stop sign (PUBLISHED) | **no**; ~5 DS seed-to-seed spread is *INHERITED* from our own 2026-07-16 note and is **not re-verified — do not let it decide a GPU-day** |
| **HUGSIM** | **YES** (3DGS) | rendered camera; 70+ sequences from KITTI-360/Waymo/nuScenes/PandaSet, 400+ scenarios (PUBLISHED) | driving-score family in the CARLA lineage | not established here |
| **NeuroNCAP** | **YES** (NeRF) | rendered camera, nuScenes rig; safety-critical staged collisions (PUBLISHED) | NCAP-style collision-avoidance score | not established here |
| 2026 arrivals worth naming | — | HiDrive (2605.09972), Bench2Drive-Robust (2605.18059), DriveE2E, MDrive, Fail2Drive, Safe2Drive — all CARLA-lineage or replay | — | — |

**The load-bearing external result the field itself publishes** (PUBLISHED, arXiv 2605.00066, and already in
our own `2026-07-16-benchmark-ecosystem-and-metric-suite.md`): across 15 methods, **ADE/FDE show no reliable
correlation with closed-loop Driving Score**, and even NAVSIM's aggregate PDM score correlates
non-monotonically with Bench2Drive DS **with ranking inversions** (fully paired subset n=8). ⇒ **One
benchmark is a weak claim.** Any adoption plan must name a second one, which is why §5 does.

---

## 2. Genuinely closed-loop, or a proxy? — the distinction that decides the value

- **AlpaSim, Bench2Drive, CARLA LB2.0, HUGSIM, NeuroNCAP: genuinely closed-loop.** Error compounds.
- **NAVSIM v2 is not**, and says so: it is *pseudo-simulation* — a two-stage open-loop protocol where 3DGS
  synthesises counterfactual views **after** a policy deviation, reported as **R² ≈ 0.8** against true
  closed-loop vs ≈0.7 for the best pure open-loop metric (PUBLISHED, 2506.04218). Better than open-loop,
  explicitly not a substitute.
- **nuPlan is closed-loop but not *sensor* closed-loop.** ⚠️ Absence claim, two probes at different
  locations: (i) the nuPlan closed-loop protocol scores a *planner* over tracks and a map — no renderer is
  part of the loop (PUBLISHED, 2106.11810); (ii) NAVSIM/OpenScene exist precisely to redistribute nuPlan's
  **sensor blobs** for camera policies, a separate artifact from the simulator (PUBLISHED, navsim docs).
  ⇒ For TanitAD, nuPlan is evaluable only by bolting a perception stack in front of it, which measures the
  perception stack.

---

## 3. Input / sensor contract and geometry — computed, not asserted

`bench_geometry_check.py --selftest` → 6/6; the matrix below is `bench_geometry.json`, both in this
directory. Method: (T1) union-azimuth coverage of the target span; (T2) worst-case angular sampling —
for a pinhole d*x*/d*θ* = f·sec²θ ≥ f, so the binding constraint collapses to `f_src ≥ W_target / HFOV_rad`;
(T3) cylindrical stitch parallax ≈ (b/r)·(W/HFOV_rad) target-pixels at range r.

Targets: **v5f** = 120° cylindrical 256×640 (model sees the 176×624 sub-frame) → **305.58 px/rad**
(MEASURED, `V5_FLAGSHIP_DEEP_REVIEW.md:22`, from the run's own `config.json`).
**v1 / REF-C** = 256×256 square canon at **F_REF = 266.0 px** → 266 px/rad, HFOV **51.4°** *derived*
(MEASURED: the live driver self-check reads `f_eff 265.6–266.014 == F_REF`).

| rig | target | verdict | src/tgt sampling | cams needed | stitch err @5 m |
|---|---|---|---|---|---|
| **physicalai_av_nurec** (AlpaSim) | **v5f** | **FEASIBLE_RESAMPLE** | **3.00×** | **1** | **0.0 px** |
| **physicalai_av_nurec** | **v1/REF-C** | **FEASIBLE_RESAMPLE** | 3.45× | 1 | **0.0 px** |
| carla_declared_trio (co-located) | v5f | **FEASIBLE_RESAMPLE** | 2.27× | 3 | **0.0 px** |
| carla_declared_trio (co-located) | v1/REF-C | **FEASIBLE_RESAMPLE** | 2.60× | 1 | **0.0 px** |
| navsim_nuplan_8cam | v5f | FEASIBLE_WITH_SEAM | 5.03× | 3 | **21.4 px** *(estimated inputs)* |
| navsim_nuplan_8cam | v1/REF-C | FEASIBLE_WITH_SEAM | 5.78× | 1 | 18.6 px *(estimated inputs)* |
| nuscenes_front_trio (NeuroNCAP/HUGSIM) | v5f | FEASIBLE_WITH_SEAM | 3.74× | 3 | 18.3 px *(estimated inputs)* |
| nuscenes_front_trio | v1/REF-C | FEASIBLE_WITH_SEAM | 4.29× | 1 | 16.0 px *(estimated inputs)* |

**Three conclusions, and one honest limit.**

1. **Resolution is never the problem.** Every candidate rig over-samples our targets by 2.3–5.8×, so every
   re-projection is a *down*-sample (band-limit, then decimate) — the safe direction. The old worry
   "a benchmark whose camera rig we cannot match" is, on the sampling axis, **not the binding constraint**.
2. **Optical-centre separation is the problem.** A cylinder is a single-viewpoint projection. On NAVSIM /
   nuScenes it must be stitched from 3 physically separated cameras, giving **~18–21 px of parallax error
   at 5 m on a 640-px raster (≈3 % of frame width)** — concentrated in the near field, i.e. exactly where
   lead vehicles, cut-ins and VRUs are. That is a distribution v5f never trained on, and it is a
   *geometric* defect, not a domain-shift hand-wave.
3. **On AlpaSim/NuRec there is no stitch at all** (one 120° f-θ camera → one 120° cylinder), and **on CARLA
   there is no stitch error either** because we declare the rig and may co-locate the three pinholes at one
   (x,y,z). Those are the two feasible-without-retrain benchmarks.
4. ⚠️ **Limit, stated:** the NAVSIM/nuScenes rows use **ESTIMATED** per-camera HFOV (64°/70°), yaws and
   baselines — not read from a calibration file. They are flagged `estimated_inputs=true` in the JSON. The
   *ordering* (0 px vs ~20 px) is robust to any plausible correction; the exact pixel counts are not.
   Falsifier is one file read: parse `nuplan` / `nuScenes` calibration and re-run.
5. ⛔ **Geometry feasible ≠ distribution matched.** Nothing above models exposure, ISP, weather, city or
   sensor noise. Our own measurement of that gap on reconstructions is **3.21×** (REF-C open-loop ADE
   1.5157 on NuRec renders vs 0.4728 on real footage — MEASURED, `REFC_openloop_diagnostic.json`), and a
   CARLA-rendered frame is further from our training distribution than a NuRec render of our own clip.

---

## 4. ⭐ Could each score have detected what our four families detected?

The PI's test case, MEASURED, from `TanitAD Research Hub/Evaluation/Videos/alpasim-openloop-thor-2026-08-03/README.md`
(paired episode-cluster bootstrap over 9 disjoint segments of one clip; the estimator and its unit are
named on the page):

- On junction scene `7c72937c`, **heading error (+0.0048 [−0.0536, +0.0647]) and curvature error
  (−0.0210 [−0.0676, +0.0088]) did NOT separate, while yaw-rate error (+0.0126 [+0.0043, +0.0212]) and
  lateral ADE (+0.1631 [+0.1082, +0.2249]) DID.** Four lateral metrics, two verdicts — a single lateral
  number would have been a coin flip, and ADE cannot arbitrate it.
- On the objects-vs-empty contrast with control drift removed, **REF-C's ADE separated (−0.0206
  [−0.0371, −0.0061]) while its target-speed error and lateral ADE did not** — i.e. ADE moved where the
  families said nothing, and (in the junction case above) the families moved where ADE could not look.
  *(The PI's phrasing "ADE saw nothing where four lateral metrics separated cleanly" is INHERITED from the
  brief; the MEASURED instances above are the closest artifacts I could quote, and they establish the same
  conclusion — the families are not redundant with ADE in either direction.)*

Now judge each score against that:

| score | can it see a heading/curvature/yaw split? | can it see target-speed error? | can it see a manoeuvre or route decision? | verdict |
|---|---|---|---|---|
| **AlpaSim `progress_score` + pass** | **No.** No orientation term of any kind. | Only through `progress_rel` — an *integrated* proxy that is blind to the sign and shape of the error. | **No.** | **would have detected nothing** in the case above |
| AlpaSim's **metric bundle** (shipped alongside the score, MEASURED keys) | partially — `min_distance_to_lane_boundary_m`, `wrong_lane`, `plan_deviation`, `dist_to_gt_trajectory` are lateral-*position* terms | `min_distance_to_obstacle_m` is a real headway/TTC substrate; `min_ade@{0.5,1.0,2.5,5.0}s(gt)` are ADE at four horizons | no manoeuvre/route term | **useful raw material**, not a score |
| **EPDMS** | LK is lane-keeping *compliance*; HC/EC are comfort thresholds — pass/fail, not error magnitude | EP + TTC, again integrated | no | would not separate the case above |
| **DS = RC × IP** (CARLA / Bench2Drive) | **No** — it counts catastrophes | no | **Bench2Drive's per-skill Skill Score is the closest community analogue of our TACTICAL family** | least sensitive of all to smooth-but-wrong paths |
| **nuPlan CLS** | it is the **only** community score with explicit human-similarity terms — longitudinal velocity error, longitudinal stop-position error, lateral position error | **yes, explicitly** | no | closest in composition — and it is **the one we cannot feed** |

> ⭐ **The finding worth carrying:** *the community score whose composition most resembles our four families
> (nuPlan CLS) is the one our vision-only models cannot be run on; the one we can run on today (AlpaSim) has
> the thinnest composition of all — progress gated by catastrophe.* The complementarity is therefore not a
> convenient story, it is structural: **no adoptable community score can replace the four-family panel, and
> the panel cannot buy external comparability.** Publish both or publish neither.

---

## 5. RECOMMENDATION

### 5.1 Adopt ONE first: **AlpaSim / the AlpaSim E2E Closed-Loop Challenge 2026**

| criterion | AlpaSim | why it beats the alternatives |
|---|---|---|
| genuinely closed-loop | ✅ | NAVSIM is a proxy; nuPlan has no renderer |
| geometry for **v5f** | ✅ 1 camera, 3.00× down-sample, **0 px stitch** | NAVSIM/nuScenes need a 3-camera stitch at ~20 px near-field error |
| geometry for **REF-C / v1** | ✅ already **live-verified**: `f_eff 266.014 == F_REF 266.0` | nothing else is verified end-to-end |
| distribution | ✅ **the same corpus family we train on** (PhysicalAI-AV) | every other benchmark is also a domain transfer |
| external comparability | ✅ **live leaderboard**: live 2026-06-15, rules freeze **2026-09-15**, public leaderboard closes **2026-10-31**, two tracks (PAI-AV, nuPlan), Docker submissions run by organiser workers on **private held-out** scenes (PUBLISHED) | CARLA LB2.0 is the only other live leaderboard, at much higher integration cost |
| published reference **with an interval** | ✅ **0.73 ± 0.01** over 910 PhysicalAI-AV-NuRec scenarios (PUBLISHED) | no other candidate publishes an interval at all |
| integration cost for us | ✅ **lowest** — the suite has already run at n=12 and n=37, adapters exist, Apache-2.0 simulator | see §5.4 |

⭐ **Non-obvious consequence of the challenge's mechanics:** the challenge's *organiser-managed workers* run
our container on their scenes. ⇒ **The NGC-gated NRE renderer — the program's "one real hole" (MEASURED,
`ALPASIM_STATE.md` row 14, three probes) — does NOT block a leaderboard submission.** It blocks local
iteration only, and local iteration already has a substitute: our own gsplat backend on Thor.
⚠️ Two things must be checked before this is treated as settled (falsifier F1, §6): the container spec, and
whether the challenge's score is the same `score_criteria` our artifacts show.

### 5.2 Second benchmark — **Bench2Drive (CARLA)**, adopted only as a *cross-check*, not a headline

Because of the published ranking inversions (2605.00066), a single benchmark is a weak claim. Bench2Drive
is the right second one: closed-loop, per-skill resolution (the only community analogue of our TACTICAL
family), 220 short routes with one safety-critical scenario each, and — decisively — **we declare the
sensor rig, so the 120° cylinder can be synthesised exactly from co-located cameras (0 px stitch)**.
Cost is dominated by CARLA-on-pod + the agent wrapper, not by data.

### 5.3 Explicitly DEFER, with the reason

| benchmark | defer because |
|---|---|
| **NAVSIM v2** | log-replay proxy (R²≈0.8, not closed-loop) **+** 300–450 GB sensor blobs **+** CC-BY-NC-SA / nuPlan **non-commercial** licence **+** a 3-camera stitch at ~21 px near-field error. Three independent costs, one proxy metric. |
| **nuPlan** | not evaluable by a vision-only policy (§2). Revisit only if TanitAD ever ships a perception front-end. |
| **NeuroNCAP / HUGSIM** | scientifically the most interesting (safety-critical, neural rendering) but they inherit the nuScenes rig's parallax stitch and a second reconstruction domain on top of our own. Revisit after AlpaSim is producing a leaderboard number. |
| **CARLA LB2.0 (official)** | subsumed by Bench2Drive for our purposes; the official leaderboard's long routes have the worst variance-per-GPU-hour. |

### 5.4 How to COMBINE it with the four-family panel — the two-layer report card

**Design.** One rollout set, two layers, never separated:

- **Layer 1 — comparability (external):** AlpaSim `pass` rate and `progress_score`, on a *frozen*
  scene list, reported **with our estimator bolted on**: a **scene-cluster bootstrap** over scenes
  (paired when two arms run the same scenes), because AlpaSim's own aggregation emits `*_std: None`
  (MEASURED). This is the number that goes next to NVIDIA's 0.73 ± 0.01 — and only ever with the scene
  list and rollout length stated.
- **Layer 2 — diagnosis (internal):** the four families computed from the **same** rollout records —
  `cl_metrics.py` already produces them unchanged for open- and closed-loop (MEASURED, provenance block of
  the open-loop README).
- **Layer 1.5 — the free bridge:** AlpaSim's own metric bundle already contains
  `min_distance_to_obstacle_m` (→ LONGITUDINAL distance-keeping), `min_distance_to_lane_boundary_m` +
  `wrong_lane` (→ LATERAL position), and `min_ade@{0.5,1.0,2.5,5.0}s(gt)` (→ the ADE row). Mapping these
  into the panel costs one adapter function and makes the two layers *commensurable on the same rollout*.

**Binding rules for the combined artifact** (they inherit from CLAUDE.md, they are not new policy):
1. Never publish layer 1 without layer 2 — an eval that reports a community scalar alone is exactly the
   ADE-only failure the four-family rule forbids, wearing a leaderboard badge.
2. Never publish layer 2 alone as an *external* claim — it has no community denominator.
3. Every AlpaSim number carries the **reconstruction-OOD control** (REF-C 3.21×, `REFC_openloop_diagnostic.json`).
   Levels are within-sim; orderings survive. This is the hard link to Q2: **if the render work lowers 3.21×,
   the leaderboard number gets more trustworthy, and that is a measurable, pre-registerable prediction.**
4. Report per family; never pool layer 1 and layer 2 into one composite.

### 5.5 Integration plan — per model, with the geometry obstacle stated

**REF-C base (104.2 M) — geometry: SOLVED, verified live.**
| step | state | cost |
|---|---|---|
| driver adapter `refc_driver.py` (f-θ canon + gRPC, model-agnostic) | ✅ exists, `f_eff 265.6–266.0 == F_REF` MEASURED | 0 |
| suite run at n=12 / n=37 | ✅ has run | ~GPU-hours only |
| re-run on the frozen challenge scene list | ⛔ needs an A40 with either NRE **or** our gsplat backend | 1 pod-day ESTIMATED |
| container for the leaderboard | ⛔ not started — Docker spec unread | 1–2 days ESTIMATED |

**v5f — geometry: FEASIBLE (1 camera, 3.00× down-sample, 0 px stitch); the blockers are elsewhere.**
| step | state | cost |
|---|---|---|
| **cylindrical resample in the driver** — f-θ equidistant 1920 px @120° (916.7 px/rad) → cylinder 640 px @120° (305.6 px/rad), then the 176×624 crop | ⛔ **not written. This is 0-GPU work available today** | ~0.5 day |
| an `f_eff`-style self-check for the cylinder (the invariant that already saved REF-C) | ⛔ not written; must assert the resampled px/rad within 1 % of 305.58 | folded in |
| a usable checkpoint | 🔴 **v5f is step 2,300 and degrading (1.0182 vs ~0.31 at steps 1800–2000)** — MEASURED, program report 2026-08-03 13:00 | PI decision, not ours |
| suite run | blocked on the above | — |

**flagship v1 (256 px square)** — adapter exists (`flagship_v1_driver.py`, reuses `RefCDriver`); it has
driven closed-loop and passed a scene REF-C crashed (n=1, directional). Its route head is a **circular nav
echo** and must never be quoted as strategic skill.

### 5.6 The 12-week shape (priority order, not a dependency chain)

1. **Now, 0-GPU:** v5f cylindrical-resample adapter + self-check; the layer-1.5 mapping function;
   read the challenge's metric spec and container requirements.
2. **Next GPU window:** re-run REF-C base + flagship v1 on a frozen scene list with our gsplat backend,
   emit the two-layer report card, and re-measure the 3.21× OOD control after the Q2 render fixes.
3. **Before 2026-09-15 (rules freeze):** decide submission; a submission needs only the container.
4. **In parallel, cheap:** Bench2Drive agent wrapper with a co-located 3-pinhole rig (0 px stitch).

---

## 6. Falsifiers — pre-registered, both outcomes committed

| # | claim | falsified if | cost to test |
|---|---|---|---|
| **F1** | "Our AlpaSim score is comparable to NVIDIA's 0.73 ± 0.01" | the challenge's published metric spec differs from our artifact's `score_criteria` (`progress_score = min(clamp(progress_clipped_rel,0,1)/0.8,1.0)`, gated on `collision_at_fault==0 ∧ offroad==0`), **or** the scene list / rollout length differs | 1 doc read + 1 diff |
| **F2** | "v5f needs no retrain to run on AlpaSim" | the cylindrical resample's measured `px/rad` deviates >1 % from 305.58, **or** v5f's open-loop ADE on renders exceeds its real-footage ADE by more than REF-C's 3.21× | one open-loop pass, no training |
| **F3** | "The community score is complementary, not redundant" | across ≥8 arms, AlpaSim `progress_score` correlates with **every** one of the four family vectors at \|ρ\| > 0.9 — then the panel adds nothing on this benchmark and we should say so | re-analysis of banked rollouts, 0 GPU |
| **F4** | "Bench2Drive is a cheap second benchmark" | v5f's score on CARLA frames degrades relative to its NuRec numbers by **more** than the 3.21× reconstruction-OOD factor — then it measures rendering domain, not driving | one CARLA open-loop pass before any closed-loop spend |
| **F5** | "Geometry is feasible on NAVSIM with a stitch" | reading the real nuPlan calibration moves the stitch error below ~5 target px (then NAVSIM re-enters contention) **or** above ~40 px (then it is out permanently, not merely deferred) | one calibration-file read |

---

## 7. What I did NOT establish

- **The challenge's exact scoring formula and container spec.** The public overview page names a "capability
  score" and a "safety metric" and points at a metrics doc I did not retrieve. Our `score_criteria` is
  MEASURED from *our* runs of stock AlpaSim; that the challenge uses the same one is **HYPOTHESIS** (F1).
- **NAVSIM / nuScenes calibration numbers** — ESTIMATED, flagged in the JSON (F5).
- **The ~5 DS CARLA seed variance** — INHERITED from our 2026-07-16 note, not re-verified. It decides nothing here.
- **Whether NVIDIA's 0.73 ± 0.01 is over scenes, seeds or rollouts.** Until that is read, it is a published
  number, not a comparable one.
- **No new closed-loop run was performed in this stream.** Everything quoted is from banked artifacts.

## 8. Sources

PUBLISHED, external: [navsim metrics](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md) ·
[NAVSIM (arXiv 2406.15349)](https://arxiv.org/html/2406.15349v1) ·
[Pseudo-Simulation / NAVSIM v2 (arXiv 2506.04218)](https://arxiv.org/pdf/2506.04218) ·
[navsim splits/sizes](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md) ·
[nuPlan (arXiv 2106.11810)](https://arxiv.org/pdf/2106.11810) ·
[nuPlan-R (arXiv 2511.10403)](https://arxiv.org/abs/2511.10403) ·
[Bench2Drive (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/017761f94a1cd66d01c041aff85492c4-Abstract-Datasets_and_Benchmarks_Track.html) ·
[CARLA Leaderboard 2.0 evaluation](https://leaderboard.carla.org/evaluation_v2_0/) ·
[CARLA Leaderboard 2.0 get started](https://leaderboard.carla.org/get_started_v2_0/) ·
[open-loop vs closed-loop correlation (arXiv 2605.00066)](https://arxiv.org/pdf/2605.00066) ·
[HUGSIM (arXiv 2412.01718)](https://arxiv.org/abs/2412.01718) ·
[NeuroNCAP (arXiv 2404.07762)](https://arxiv.org/html/2404.07762v2) ·
[Alpamayo 1.5 / AlpaSim + NuRec](https://huggingface.co/blog/drmapavone/nvidia-alpamayo-1-5) ·
[Alpamayo 2](https://huggingface.co/blog/nvidia/nvidia-alpamayo-2) ·
[AlpaSim E2E Closed-Loop Challenge 2026](https://nvidia-alpasime2eclosedloopchallenge2026.hf.space/) ·
[PhysicalAI-AV-NuRec dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) ·
[HiDrive (arXiv 2605.09972)](https://arxiv.org/abs/2605.09972) ·
[Bench2Drive-Robust (arXiv 2605.18059)](https://arxiv.org/abs/2605.18059)

MEASURED, ours (repo paths): `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-22-alpasim-closedloop-evalpod/REFC_suite_base_results.json` ·
`…/REFC_openloop_diagnostic.json` · `…/2026-07-26-alpasim-consolidation/ALPASIM_STATE.md` ·
`TanitAD Research Hub/Evaluation/Videos/alpasim-openloop-thor-2026-08-03/README.md` ·
`Project Steering/MODEL_REGISTRY.md` §4.4 · `Project Steering/V5_FLAGSHIP_DEEP_REVIEW.md` ·
`Project Steering/Reports/2026-08-03-1300-program-report.md` · and this directory's
`bench_geometry_check.py` / `bench_geometry.json`.
