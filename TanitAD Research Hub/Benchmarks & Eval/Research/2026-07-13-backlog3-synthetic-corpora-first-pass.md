# Backlog #3 — robustness suite, first pass on the ungated synthetic corpora (no simulator)

**Agent:** Benchmarks & Eval (Sayed-directed, pod-independent — "cheaper first pass while the pods are
busy"). **Date:** 2026-07-13. **Hardware:** dev box + local RTX 4060, CPU-only (numpy + tarfile).
**Cost:** $0. **Pods:** none touched (pod/pod2/pod3 and their caches untouched; downloads went to
`C:\Users\Admin\tanitad-data\cosmos_bench3`). **Intake:**
`Implementation/incoming/2026-07-13-cosmos-robustness-first-pass/`.

## TL;DR

- The custom robustness suite (LAL / TMS / OKRI / CNCE / LOPS) now has its **first numbers on real
  downloaded data** — 13 Cosmos-Drive-Dreams clips — with the pipeline validated end-to-end. This
  closes the "available NOW, no simulator" half of backlog #3 for the corpus that can actually feed it.
- **Key acquisition finding:** the metric suite is **pixel-free**, so the first pass needs only the
  small per-clip RDS-HQ annotation tars (ego pose + 3D object boxes + intrinsics, ~15 MB/clip) — **not
  the 43 GB video shards**. Cheaper than the backlog assumed.
- **Headline data-only number: OKRI** (occluded kinetic-risk integral) — median **21.1**, range
  **0.06 … 268** over the 13 clips, non-trivial on 12/13. It scales as expected with ego KE × blind-spot
  exposure (fast clips near occluders score 200+, a 4.6 m/s crawl scores 0.06).
- **LOPS pipeline validated**: data-only LOPS is the honest **0.0** baseline (no world-model estimate);
  a perfect-perception oracle (wm = gt + N(0,0.3)) returns mean **0.844** vs the analytic **0.8325** —
  the exact constant the 2026-07-09 SC-01 audit recomputed. The occlusion → hidden-agent path is wired
  correctly on real geometry.
- **Corpus #2, WorldModel-Synthetic-Scenarios, is video + VLM-caption ONLY** (no ego pose, no boxes).
  It **cannot** feed the geometric suite data-only — that path needs a perception/pose model (same
  model-dependency that blocks the closed loop). Confirmed by direct HF probe; documented as a gap.
- **Gaps found:** TMS is noise-dominated on pose-derived jerk; LAL-v1 positivity is a jerk-noise
  artifact (re-confirms the H15 fragility finding); LAL-v2's free-cruise-start assumption doesn't hold
  for arbitrary logged clips; CNCE needs real latency/params; LOPS/collisions need a model. Details §5.

## 1. The contract — what each metric needs vs what the data can supply (P8)

`ScenarioTelemetry` (`stack/tanitad/eval/metrics.py` L103-141) wants per-step ego kinematics, occlusion
geometry, a world-model hidden-agent estimate, and the hidden-agent ground truth. Split by provenance:

| Metric | Needs | Data-only from Cosmos annotations? |
|---|---|---|
| **OKRI** | `ego_v`, `dist_to_blind_spot`, `ego_mass` | **Yes** — ego pose + 3D-box occlusion geometry. Headline. |
| **TMS** | `ego_jerk`, `steer_rate` | **Yes** (logged ego) — but jerk is noise-amplified (§5). |
| **LAL-v1** | `hazard_los_flag`, `ego_jerk` | **Yes** (logged ego) — but jerk-threshold fires on noise (§5). |
| **LAL-v2** | `ego_v`, `hazard_los_flag` | **Yes** (logged ego) — free-cruise-start assumption often violated (§5). |
| **LOPS** | `is_occluded_flag`, **`wm_hazard_xy`**, `gt_hazard_xy` | **No** — `wm_hazard_xy` is a *model* latent estimate → 0.0 baseline. GT track + occlusion are data-only (oracle validates). |
| **CNCE** | `ego_v`, **`latency_ms`**, **`params_billions`**, `collisions` | **No** — latency/params are model+hardware; stubbed & labelled. |

All five instruments measure the **logged synthetic ego trajectory**, i.e. the corpus's data-collection
driver — **not** a TanitAD checkpoint. This is a pipeline-validation + corpus-baseline pass, exactly
what backlog #3 asks for. No number here is a model claim.

## 2. Corpora acquired

### 2a. Cosmos-Drive-Dreams — the corpus that feeds the suite (CC-BY-4.0, ungated)
`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`, RDS-HQ format, 5,843 clips. Per clip, on
disk (all small; anonymous download over the HF tree API + `truststore` TLS):
- `vehicle_pose/<clip>.tar` (~0.6 MB) — 297 per-frame 4×4 ego-to-world → ego kinematics.
- `all_object_info/<clip>.tar` (1–31 MB) — 297 per-frame JSON of every object:
  `object_to_world` (4×4), `object_lwh`, `object_is_moving`, `object_type` → 3D dynamic-agent tracks.
- `pinhole_intrinsic/<clip>.tar` (~20 KB) — per-camera `[fx,fy,cx,cy,W,H]` (front_wide = 120° HFOV).

Sampled **13 clips** spread across object-richness (19–246 objects/clip). Hazards auto-selected: 12
Automobile + 1 Heavy_truck, all with an occlusion → reveal profile.

### 2b. WorldModel-Synthetic-Scenarios — video + caption only (OpenMDW-1.1, ungated)
`nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios`, 264k clips. Families present are
exactly the long-tail we want (`pedestrian`, `emergency`, `nudging`, `lanechange`, `weather_degradation`).
**But** each clip dir holds only `video/*.mp4` (7 cameras, up to 665 MB each) + `description/*.json`,
and the description JSONs are **qwen2.5-7B natural-language captions** (`framerate`, `nb_frames`,
`t2w_windows[].qwen2p5_7b_caption`) — **no ego pose, no 3D boxes, no telemetry** (probe saved in
`worldmodel_structure.json`). This confirms the 2026-07-09 DataEng caveat. Consequence: the geometric
suite cannot consume this corpus data-only; extracting telemetry from its pixels needs a
perception/pose model — the same model-dependency as the closed-loop path. It stays a **scene-diversity**
source, not a first-pass metric source.

## 3. First numbers (13 clips, data-only unless flagged)

| Metric | min | median | max | reading |
|---|---|---|---|---|
| **OKRI** (lower safer) | 0.06 | **21.1** | 268 | KE into blind spots; scales v²×occlusion. **Headline.** |
| **TMS** (→1 smooth) | 0.010 | 0.039 | 0.073 | Noise-dominated (see §5); 0.117 median with 5-tap smoothing. |
| **LAL-v1 s** (>0 proactive) | −0.4 | 2.9 | 8.6 | Positive on 11/13 — **jerk-noise artifact**, not anticipation (§5). |
| **LAL-v2 s** (>0 lead) | −4.2 | — | 6.9 | Fired on **6/13**; rest = `LAL_NO_REACTION` (no sustained decel). |
| **CNCE** (higher better) | 629 | 1.8e3 | 4.96e3 | **Stub-driven** (constant latency/params) → preliminary only. |
| **LOPS** (→1 tracks) | 0.0 | 0.0 | 0.0 | Honest data-only baseline (no model estimate). |
| **LOPS oracle** | 0.828 | 0.837 | 0.881 | Pipeline check → 0.844 mean ≈ analytic 0.8325. |

Supporting: `frac_occluded` median 0.37 (range 0.06–0.89), `frac_los` median 0.16, ego speed 4.6–36 m/s.
Per-clip rows in `results.json`; aggregates/diagnostics in `diagnostics.json`.

**OKRI reads sensibly.** Highest: clip `12567db9` (v̄ 32.7 m/s) → 268, and `0a48e742` (v̄ 16.7 m/s,
77% occluded) → 207 — high KE carried while a blind spot is near. Lowest: `25cf666b` (v̄ 4.6 m/s) →
0.06 — a slow crawl carries almost no occluded kinetic risk. This is the metric behaving as designed as
a **data-only robustness signal**, and it is the number to carry forward.

## 4. Pipeline validation (end-to-end, no simulator)

- **Contract intact in the venv:** `test_metrics.py` + `test_metric_dynamics.py` +
  `test_scenario_suite_wiring.py` + `test_work_zone_phantom.py` → **40 passed**.
- **Existing oracle wiring still discriminative:** `scenario_suite_dryrun.py` → all 5 checks pass
  (world_model closure_incursion 0.0 vs reactive 20.8; OKRI 14.8 vs 120.6).
- **Occlusion geometry fires on real data:** 13/13 clips produce a hazard with occluded frames; the
  bird's-eye occlusion test yields realistic exposure (6–89% of frames).
- **LOPS end-to-end verified against analytic ground truth:** oracle mean 0.844 vs 0.8325 (|diff|
  0.011) — the σ=0.3 constant from the 2026-07-09 SC-01 audit, now reproduced on real Cosmos occlusion
  frames. The `is_occluded → wm vs gt` path is correct.

## 5. Gaps found (what the first pass exposes)

1. **TMS is noise-dominated on pose-derived jerk.** Mean ∫|jerk| dt = **41.2** (steering ∫ ≈ 0 — these
   are near-straight highway clips), so `TMS = 1/(1+α·∫|jerk|)` collapses toward 0 (median 0.039).
   Double finite-differencing 10 Hz synthetic poses amplifies quantization noise. A 5-tap smoothing of
   accel/steer lifts the median to 0.117. **Action:** TMS on logged pose data needs a smoothed/native
   jerk source or recalibrated α/β (the α=1, β=1.5 defaults were set for clean CARLA control signals).
2. **LAL-v1 positivity is a jerk-noise artifact.** With mean |jerk| ~4 m/s³, the −1.5 m/s³ brake trigger
   fires early on noise, inflating `t_LoS − t_brake` → false "anticipation" on 11/13 clips. This
   **re-confirms the H15 LAL-v1 fragility** that motivated LAL-v2; LAL-v1 should not be read on
   pose-derived jerk without denoising.
3. **LAL-v2's free-cruise-start assumption often doesn't hold.** Its reference is the max speed over the
   first 30% of the clip; arbitrary logged clips may start mid-manoeuvre → `LAL_NO_REACTION` on 7/13.
   Honest, but means LAL-v2 wants **clip segmentation to a free-cruise onset** before it is comparable
   across a corpus (trivial in the scripted CARLA scenarios; a preprocessing step for logged data).
4. **CNCE needs real latency + active-param count.** Stubbed at the TanitAD-4B envelope (18 ms, 4.0 B) →
   CNCE just tracks distance travelled here. Real numbers come from a checkpoint rollout with measured
   inference latency. Preliminary only.
5. **LOPS and `collisions` need a model.** Data-only LOPS is structurally 0.0 (no latent estimate);
   `collisions` = 0 on these non-interventional logs. Both are closed-loop / model-behavior quantities.
6. **Closure-incursion (H9) is orthogonal here.** The STATE's "closure-incursion detector reads 0" is a
   *live-CARLA SC-01 extraction* gap — on the design oracle it is discriminative (0.0 vs 20.8, §4).
   Generic Cosmos clips have no scripted lane closure, so this signal is neither computed nor claimed in
   this pass; it stays on the scripted-scenario / CARLA-on-pod path.
7. **Occlusion model is a bird's-eye approximation.** Center-bearing occlusion with a half-diagonal
   footprint against 3D boxes — good enough to *exercise* the suite, not a perception-grade LiDAR/depth
   occlusion oracle. Fine for a first pass; flag before any comparative claim.

## 6. Done-now vs still blocked

**Done now (this pass):** metric pipeline validated end-to-end on real ungated data; first OKRI numbers
(data-only, headline); LAL-v1/-v2 + TMS characterized on the corpus with their failure modes quantified;
LOPS occlusion path validated against analytic ground truth via the oracle; the acquisition recipe
(annotation-only, no video shards) established; WorldModel-Synthetic ruled out as a data-only geometric
source (documented).

**Still blocked on CARLA-on-pod / a checkpoint (D-014, W31-32):** real LOPS (needs `wm_hazard_xy` from a
world-model rollout); real CNCE (measured latency/params); `collisions` and closure-incursion on live
telemetry; ≥3-seed CI-separated *comparative* claims (reactive vs world-model) — this pass is
single-source, single logged policy, so no "beats baseline" claim is made.

## 7. Reproduction

```
# venv C:\Users\Admin\venvs\tanitad (numpy 2.5.1, huggingface_hub 1.22, truststore)
cd "…/Benchmarks & Eval/Implementation/incoming/2026-07-13-cosmos-robustness-first-pass"
python acquire_cosmos_sample.py --out C:/Users/Admin/tanitad-data/cosmos_bench3 --n 12
python cosmos_telemetry.py --root C:/Users/Admin/tanitad-data/cosmos_bench3 --out results.json
```

Artifacts committed with this note: `cosmos_telemetry.py`, `acquire_cosmos_sample.py`, `results.json`,
`diagnostics.json`, `acquire_manifest.json`, `worldmodel_structure.json`, `INTAKE.md`.
