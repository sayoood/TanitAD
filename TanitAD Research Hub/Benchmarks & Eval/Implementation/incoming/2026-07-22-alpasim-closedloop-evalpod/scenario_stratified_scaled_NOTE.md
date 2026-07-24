# Scenario-stratified flagship-v1 vs REF-C-base — BALANCED SCALED suite (n=37) — MEASURED 2026-07-23

The reusable **v4.2 closed-loop benchmark**: 5 scene categories, ~balanced n, both current ckpts.
Raw: `scenario_stratified_scaled_results.json` (per-scene + per-category + paired) · keyframes
`scaled_keyframes/*.jpg` (38) · labels `scaled_suite_labels.json` · pipeline
`kf_batch.py`/`select_suite.py`/`scaled_wizard_gen.sh`/`scaled_master.sh`/`scaled_aggregate2.py`.

## ⚠️ FRAMING (mandatory)
WITHIN-SIM RELATIVE, on NuRec reconstructions (~3.2× OOD, RUN_RECIPE §13). The per-category **ranking**
is trustworthy; the **absolute rates are not real-world.** 480×854 (ranking is resolution-robust, §16).

## Method
- **Labeling (goal 1):** no scene-type field exists in NuRec/public_2604 metadata → classified by direct
  multimodal keyframe inspection. **356 candidate keyframes** downloaded (2 diverse batches from the 1606-scene
  `public_2604` pool, mp4→ffmpeg keyframe→PIL montage grids), all auditable. **Roundabouts are rare (~2.5%)** —
  found **9** and verified each at full res (yield signs / central islands / circular flow); traffic-lights and
  the rest are common.
- **Suite (goal 2):** balanced **~8/category = 38 scenes** (roundabout 8, highway 8, straight/other 8,
  traffic-light 7, intersection 7). Below the ~15/cat stretch target but **real n for every category** incl. the
  two previously-0 categories (roundabout, traffic-light) — the coordinator's stated priority.
- **Runs (goal 3):** flagship-v1 + REF-C-base, 480×854, 1 rollout/scene, canon f_eff=266.1 OK. 1 scene
  (`0580c069`, traffic-light) failed a route sanity check (ego 53 m off-route) in the runtime → aggregation
  was skipped; **recovered n=37** by scoring directly from the per-rollout `metrics.parquet`
  (`scaled_aggregate2.py`; scoring **validated** — reproduces the known single-scene flagship 0.699 exactly).

## 🎯 RESULT — per category (n=37, on NuRec reconstructions, 480×854)
| category | n | flag pass | refc pass | flag score | refc score | **ΔScore (flag−refc)** | flag offroad |
|---|---|---|---|---|---|---|---|
| **roundabout** | 8 | 2/8 | 2/8 | 0.103 | 0.101 | **+0.002 (TIE)** | 0.62 |
| **highway** | 8 | 2/8 | 3/8 | 0.147 | 0.221 | −0.074 (tie, CI spans 0) | 0.38 |
| **intersection** | 7 | **0/7** | 1/7 | 0.000 | 0.063 | −0.063 (both fail) | 0.86 |
| **traffic_light** | 6 | 2/6 | 4/6 | 0.113 | 0.337 | −0.224 | 0.50 |
| **straight_other** | 8 | 3/8 | 5/8 | 0.157 | 0.430 | −0.272 | 0.25 |
| **OVERALL** | 37 | **9/37** | **15/37** | **0.106** | **0.229** | **−0.123** [−0.208, −0.041] | 0.51 |

Overall paired: ΔScore **−0.123, boot95 [−0.208, −0.041]** (excludes 0); score sign test **13–4 REF-C (p=0.049)**;
pass McNemar 8–2 (p=0.109).

## VERDICT — REF-C still wins overall, but the advantage is SMALLER and CATEGORY-SPECIFIC
1. **REF-C wins overall** (ΔScore −0.123, CI excludes 0, sign test p=0.049) — but **far less than the skewed
   12-scene suite's −0.43**. That earlier gap was **inflated** because that suite was 8/12 straight/urban —
   REF-C's single best category.
2. **Where REF-C's edge lives: straight_other (−0.272) and traffic_light (−0.224).**
3. **Where they TIE: roundabout (+0.002) and highway (−0.074, CI spans 0).** ⭐ The newly-measured **roundabout**
   category is a dead heat — flagship's WM+tactical policy is NOT worse than REF-C on roundabouts.
4. **Where BOTH collapse: intersections** (flagship **0/7**, REF-C 1/7) — neither open-loop-trained planner
   handles uncontrolled junctions; flagship goes offroad 6/7.
5. **Flagship's deficit is offroad at COMPLEX JUNCTIONS** (offroad: intersection 0.86, roundabout 0.62,
   traffic-light 0.50) and least on straight roads (0.25). Its plan-deviation is 0.91 vs REF-C 0.33 — the wide-swerve
   behaviour again, punished most where the drivable corridor is narrow/branching.

**Net refinement of the prior read:** not "REF-C beats flagship everywhere off-highway" — rather **REF-C's
advantage is concentrated in straight + signalized driving; on roundabouts and highways they tie; at
intersections both fail.** The architecture gap is real but modest and geometry-dependent.

## Caveats
1. **n per category 6–8** → wide binomial CIs; per-category paired tests are mostly p≈1.0 (small n + ties). The
   **OVERALL delta (CI excludes 0, sign p=0.049)** is the powered signal; per-category ΔScore is **directional**.
2. WITHIN-SIM OOD (relative only). Single mid-clip keyframe labels (a scene's later geometry can differ; traffic-light
   needs the signal in-frame). REF-C diffusion is stochastic run-to-run (±~0.08 score); flagship deterministic.
3. 1 scene dropped (`0580c069`, route-sanity fail) → traffic-light n=6. Recovered scoring, not AlpaSim's
   `results-summary.json` (which the runtime skipped) — validated against a known single-scene score.

## Reusability (the v4.2 benchmark)
`suite_clips.txt` (38 clips) + `scaled_suite_labels.json` + `scaled_wizard_gen.sh` regenerate the exact suite;
`scaled_master.sh` runs any driver over it; `scaled_aggregate2.py` produces the per-category paired table.
⚠️ For v4.2: set `eval.allow_aggregation_with_failed_rollouts: true` in the eval-config **before** the run so a
single route-sanity failure doesn't skip aggregation (the trap that cost a recovery here).

## Deliverable manifest
| artifact | where | status |
|---|---|---|
| `scenario_stratified_scaled_results.json` | repo (staged) | ⭐ per-category paired benchmark (n=37) |
| `scenario_stratified_scaled_NOTE.md` (this) | repo (staged) | write-up |
| `scaled_keyframes/*.jpg` (38) + `scaled_suite_labels.json` | repo (staged) | auditable labels |
| `kf_batch.py`,`select_suite.py`,`scaled_wizard_gen.sh`,`scaled_master.sh`,`scaled_aggregate2.py` | repo · pod | reusable pipeline |
| rollouts (37×2), USDZs (38, ~57 GB) | **pod only** `/workspace/scaled_{refc,flag}`,`.../all-usdzs` | regenerable via the pipeline |

**Pod left CLEAN:** all services incl. renderer killed, `gpu_lock released`, GPU idle, lock FREE.

## ESCALATE / not-done
- **Goal 4 (frame-canon caching fix) — DEFERRED.** The driver re-canonicalizes the 24-frame history every step
  (~46 ms/step @854, §alpasim_realtime). Fix = canon each frame once on arrival in `submit_image_observation`,
  cache `[3,256,256]`, stack the last 8 in `plan`. It needs a cached-vs-uncached rollout equality check first
  (the coordinator's condition) — a ~20 % suite speedup, but it does not affect these results, so I prioritized
  the benchmark. Sketch + the verification gate are the next cheap win.
- **Intersections (0/7 and 1/7)** are the joint failure — the highest-value target for BOTH architectures.
- Optional: grow roundabout/traffic-light toward ~15/cat (classify more of the 1606 pool; pipeline staged).
