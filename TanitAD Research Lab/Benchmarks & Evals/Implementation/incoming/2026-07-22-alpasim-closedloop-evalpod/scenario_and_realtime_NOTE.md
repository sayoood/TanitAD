# Scenario-stratified closed-loop + AlpaSim real-time on the A40 — MEASURED 2026-07-23

Two Sayed asks. Lock `scenario-eval`/`vs-native1080` released, pod clean. ⚠️ Everything closed-loop is
**WITHIN-SIM RELATIVE on NuRec reconstructions** (~3.2× OOD, RUN_RECIPE §13) — relative ranking trustworthy,
absolute rates not real-world.

---

## 1. Scenario-STRATIFIED flagship v1 vs REF-C base (`scenario_stratified_results.json`)

**Labeling method:** no scene-type field exists in NuRec/public_2601 metadata (only a telemetry CSV; USDZ has
calib not semantics). Cosmos-Reason1-7B **is** cached on the pod but needs a separate transformers/qwen2.5-vl
env build → I classified by **direct multimodal inspection of one mid-clip keyframe/scene** (staged in
`keyframes/`, auditable). Equivalent for this n; a full VLM pass is the scale-up path.

**Scale — HONEST:** target was ~15-25 scenes/category; **achieved = the existing 12-scene suite**, which is
too small AND skewed to fill 5 categories: **highway 3, intersection 1, straight/other 8, roundabout 0,
traffic-light 0.** This *confirms* 12 is too few. Scaling needs ~75-125 new scene downloads (~1.5 GB USDZ
each) + closed-loop for both models (multi-hour) → **ESCALATED**; the download→keyframe→classify pipeline is
staged (`kf_download.sh`, note the **`HF_HUB_DISABLE_XET=1`** fix — the HF Xet backend errors on this dataset).

**Per-category (n=12, 480×854):**
| category | n | flag pass | refc pass | flag mean score | refc mean score | flag offroad | refc offroad |
|---|---|---|---|---|---|---|---|
| **highway** | 3 | 1/3 | 1/3 | 0.106 | 0.141 | 1/3 | 1/3 |
| **intersection** | 1 | 0/1 | 1/1 | 0.0 | 1.0 | 1/1 | 0/1 |
| **straight/other** | 8 | **1/8** | **6/8** | **0.060** | **0.566** | **6/8** | **1/8** |
| roundabout | 0 | — | — | — | — | — | — |
| traffic-light | 0 | — | — | — | — | — | — |

**HEADLINE (the actionable read):** flagship v1's deficit is **NOT uniform** — it is **concentrated on
straight/urban/rural roads** (pass 1/8 vs 6/8; flagship **offroad 6/8**), while **on HIGHWAY the two TIE**
(identical scene-by-scene: both pass 0009402a, both collide on 00097de1, both offroad on 000e95f7). Flagship's
wide-swerve WM+tactical policy survives wide highway lanes but **drives off narrower urban roads**. This is
consistent with the n=1 scene where flagship *beat* REF-C — that was a **highway**. So the aggregate
"−0.30/−0.43 score delta" really means **"REF-C wins decisively off-highway, ties on-highway."** The fix target
is flagship's **road-keeping on non-highway geometry**, not its highway behaviour.
⚠️ n per category is tiny (3/1/8) → DIRECTIONAL, not powered; roundabout + traffic-light UNMEASURED.

---

## 2. AlpaSim REAL-TIME on the A40 (`alpasim_realtime_a40.json`)

Control rate is **5 Hz** (200 ms/step). Isolated ticks from `rt_model_iso.py` (no contention); in-situ loop
from the instrumented `rt_driver.py` (480×854); native RT from RUN_RECIPE §14.

| | 480×854 | native 1080×1920 |
|---|---|---|
| **real-time factor** | **~0.75-0.98×** (loop 205 ms median / 265 mean) | **0.29×** (§14) |
| effective Hz | 3.8-4.9 | 1.45 |
| model GPU forward (flagship) | ~90 ms | ~90 ms (res-independent) |
| model GPU forward (REF-C) | ~18 ms | ~18-40 ms |
| frame-canon CPU (steady, 24-frame deque) | ~46 ms | **~475 ms** |
| driver plan total (canon+model) | ~100-109 ms | ~213-500 ms |

**Verdict:** **NEAR real-time at 480×854 (~0.8-1.0×), SUB-real-time at native (0.29×).** The renderer (NuRec
gsplat) is the dominant fixed cost and scales ~5× 854→native, driving the RT drop — so **renderer-bound at
native, confirmed.** BUT three corrections to the ~3 Hz/0.3× estimate:
1. The 0.3× holds at **native**; at **854 it is ~0.8-1.0×** (near the 5 Hz control) — resolution matters ~5×.
2. The flagship model tick is **~90 ms, not 28 ms** (28 ms ≈ REF-C's lighter diffusion). Flagship caps ~11 Hz
   on the model alone, not 35 Hz.
3. **Hidden fixable cost:** the driver **re-canonicalizes the WHOLE frame history (up to 24 frames) every
   step** → **~475 ms of CPU** at native (dwarfing the model). Caching canon'd frames → driver plan drops to
   model-only (~30-90 ms). A software win, independent of the GPU.

⚠️ In-situ numbers are confounded by GPU contention (renderer+model share the A40), gRPC concurrency
(driver `max_workers=8` → in-situ model 396 ms > loop 205 ms, they overlap), and renderer scene-load stalls
(p90 multi-second). The **isolated ticks are clean**; the loop_gap is the honest effective throughput. A clean
render-vs-physics split was not obtained (constant-driver control was scene-load-noisy) — render+physics+IPC
reported as one bucket, render-dominated.

---

## Deliverable manifest
| artifact | where | status |
|---|---|---|
| `scenario_stratified_results.json` | repo incoming (staged) | ⭐ per-category flag-vs-REF-C (n=12) |
| `alpasim_realtime_a40.json` | repo incoming (staged) | ⭐ real-time decomposition + verdict |
| `scenario_and_realtime_NOTE.md` (this) | repo incoming (staged) | write-up |
| `keyframes/*.jpg` (12) | repo incoming (staged) | classification evidence (auditable) |
| `rt_model_iso.py`,`rt_driver.py`,`rt_run.sh`,`rt_timing*.json`,`rt_model_iso.json` | repo (staged) · pod | RT harness + raw |
| `kf_download.sh` | repo (staged) · pod | keyframe pipeline (Xet-disabled) — the scale-up tool |
| scene mp4s + rt_iso/rt_const rollouts | **pod only** `/workspace/kf_dl`,`/workspace/rt_*` | regenerable |

## ESCALATE
1. **Scenario scale-up** to 15-25/category (75-125 scenes, both models) = a multi-hour compute budget; pipeline
   staged. The n=12 read (highway-tie / off-highway-REF-C-wins) is directional and worth confirming at scale,
   especially **roundabout + traffic-light (0 scenes now)**.
2. **Cheap flagship win:** cache canon'd frames in the driver (kill the ~475 ms/step redundant CPU re-canon).
3. VLM labeling (Cosmos-Reason1-7B, cached) needs a transformers/qwen2.5-vl env — stand up if the scale-up
   proceeds (direct inspection sufficed for n=12).
