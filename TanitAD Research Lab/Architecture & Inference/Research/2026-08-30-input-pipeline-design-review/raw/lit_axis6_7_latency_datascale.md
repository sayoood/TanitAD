# Input-pipeline design review — literature, axes 6 (cycle time / latency) and 7 (data scale + diversity)

Stream: literature. Date 2026-08-30. All primaries banked to `TanitAD Research Lab/Library/`
(`kb_add.py --verify` → **238 entries, 0 orphans, 0 problems**).

**Evidence classes used:** MEASURED (ours + artifact) · PUBLISHED (primary, banked) ·
PUBLISHED-SECONDARY · INHERITED · ESTIMATED · HYPOTHESIS.

---

## 0. THE HEADLINE CORRECTION — the brief's axis-7 premise is wrong

> Brief said: *"Single geography/domain family (it is a synthetic-plus-real NVIDIA AV dataset)"*.

**PUBLISHED (NVIDIA dataset card, `huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles`):**

| fact | value |
|---|---|
| total | **1,700 hours**, **306,152 clips**, 20 s each |
| frame rate | **30 fps** cameras |
| geography | **25 countries, 2,500+ cities** (US 155,360 clips · Germany 43,900 · France 10,364) |
| weather | clear, rain, snow, fog |
| road type | highways, urban, residential, rural |
| traffic | none / light / medium / heavy |
| time of day | daytime, nighttime |
| surface | dry, wet, snow/ice |
| provenance | **real-world sensor data** from planned collection drives (NOT synthetic) |
| sensors | 7 cameras all clips; LiDAR 298,326 clips; radar (≤10) 160,761 clips |

**Derived (ESTIMATED, arithmetic shown):** parity set 2,376 clips × 20 s = **13.2 h = 0.78 %** of the
corpus. B1 at 26 h = ~4,680 clips = **1.53 %**. We read **1 of 7 cameras**.

⇒ **Diversity is not a corpus property we lack — it is a sampling decision we made.**
Axis 7 is therefore not "our dataset is small and narrow"; it is **"our ingest reads 1.5 % of an
already-diverse 1,700 h corpus"**. That reframes every cost estimate below: the expensive part is
the epcache build, not data acquisition.

---

# AXIS 6 — CYCLE TIME / LATENCY

## Q1. What cadence do published E2E stacks target and achieve?

All numbers PUBLISHED (primary, banked), GPU named as the paper names it.

| system | params | GFLOPs | latency | FPS | GPU | source |
|---|---|---|---|---|---|---|
| UniAD (full) | **125.0 M** | **1709** | 555.6 ms | **1.8** | Tesla A100 | 2212.10156 Tab. 13 |
| UniAD det-only | 65.9 M | 1324 | — | 4.2 | A100 | 2212.10156 Tab. 13 |
| VAD-Base | — | — | **224.3 ms** | **4.5** | RTX 3090 | 2303.12077 Tab. 1 |
| VAD-Tiny | — | — | **59.5 ms** | **16.8** | RTX 3090 | 2303.12077 Tab. 1 |
| PARA-Drive | — | — | — | **2.77× UniAD** (≈5 FPS) | A100 | paradrive-cvpr2024 §4 |
| SparseDrive-S | 85.9 M | 192 | — | **9.0** | RTX 4090 | 2405.19620 Tab. 3 |
| SparseDrive-B | 104.7 M | 787 | — | **7.3** | RTX 4090 | 2405.19620 Tab. 3 |
| TransFuser | 0.36 B | — | **118 ms** | 8.5 | RTX 4090 | 2402.13243 Tab. 15 |
| VAD (same bench) | 0.36 B | — | 118 ms | 8.5 | RTX 4090 | 2402.13243 Tab. 15 |
| GenAD (ECCV24) | 0.38 B | — | 121 ms | 8.3 | RTX 4090 | 2402.13243 Tab. 15 |
| **VADv2** | **0.40 B** | — | **125 ms** | **8.0** | RTX 4090 | 2402.13243 Tab. 15 |
| DiffusionDrive | ResNet-34 | — | 22 ms | **45** | RTX 4090 | 2411.15139 abs. |
| DriveVLA-W0 backbone | ~7 B VLA | — | 118 / 240 ms | 4.2 / 8.5 | — | 2510.12796 §B.1 |
| DriveVLA-W0 query expert | — | — | **~74 ms** | 13.5 | — | 2510.12796 §B.1 |
| **Alpamayo-R1 (NVIDIA)** | 0.5–10 B | — | **99 ms ON-VEHICLE** | **~10** | on-vehicle | 2511.00088 abs. |

**VERIFIED against the brief's guesses:** UniAD 1.8 FPS ✅ · VAD 4.5/16.8 FPS ✅ ·
PARA-Drive speedup = **2.77×, not a large absolute number** — the paper only claims *"near real-time
speed"* and only after switching to an R50-tiny backbone, and says *"our model can be **potentially**
optimized for embedded devices for real-time deployment"* (i.e. it was never run on embedded).

**World-model family — this is the number that reframes our 10 Hz claim:**
- **GAIA-1 (Wayve, 6.5 B world model), verbatim:** *"the autoregressive generation process, while
  highly effective, does **not yet run at real-time**"* (2309.17080 §7).
- Vista, GenAD, DriveDreamer, WoVoGen: all report generation at **2 Hz training/eval rates** and
  none reports a real-time inference figure (2405.17398 Tab. 1). Probed twice in Vista
  (`inference|generation speed|real-time` and `A100.*sampl`) — **no generation-speed figure is
  published**. UNKNOWN, not absent-by-my-search.

**VERDICT Q1: CONSISTENT — in fact ambitious.** 10 Hz would place us at the **fast end** of the
published E2E band (0.4 B-class models cluster at **8–8.5 FPS on an RTX 4090**), and **far above**
the entire generative world-model family, which is not real-time at all. The honest sentence is:
*"10 Hz is at the top of what published 0.3–0.4 B driving models achieve on a desktop GPU, and the
world-model family it belongs to does not run in real time at all."*

## Q2. Minimum cadence for closed-loop control — ⭐ THE MEASURED CURVE

**This is the "delay X costs Y driving score" number the PI asked for.**

**PUBLISHED (primary, banked): Ji et al., arXiv 2601.07393, "Software-Hardware Co-optimization for
Modular E2E AV Paradigm"** (Jan 2026 preprint — ⚠️ NOT peer-reviewed). Method: a **Real-Time
Synchronous (RTS)** CARLA-Leaderboard harness that skips `n = max(0, int(t_i/Δt) − 1)` control
updates to emulate inference latency. UniAD baseline, CARLA 0.9.15, **220 routes × ~150 m**, diverse
weather, 8× RTX 4090 + TensorRT 10.7.

**Table 2 — Driving Score vs inference rate (UniAD):**

| FPS | DS_rt | DE_rt | DC_rt (comfort) |
|---|---|---|---|
| 1 | 32.85 | 139.81 | 0.42 |
| 1.25 | 32.04 | 141.38 | 0.40 |
| 1.67 | 34.17 | 126.89 | 0.42 |
| 2 | 33.50 | 136.69 | 0.36 |
| 3 | 35.59 | 135.45 | 0.42 |
| 5 | 33.74 | 133.40 | 0.42 |
| 8 | 36.70 | 135.57 | 0.38 |
| **10** | **37.55** | 137.44 | **0.38** |
| 14 | 37.15 | 134.28 | 0.35 |
| 18 | 38.01 | 137.55 | 0.36 |
| **20** | **39.53** | 132.03 | **0.28** |
| **24** | **40.63** | 138.21 | 0.33 |
| 28 | 39.68 | 136.24 | 0.31 |
| 30 | 40.15 | 135.90 | 0.31 |

Verbatim: *"the Driving Score shows a significant upward trend from 1 FPS to 20 FPS, with the Driving
Score improving by **20.33 %**"*; *"When the frame rate reaches approximately **20-24 FPS**,
performance improvements tend to plateau; further increasing inference speed not only yields limited
marginal benefits but may also result in a slight decline"*; *"comfort ... drops by **33.33 %** at
20 FPS"*.

**⇒ WHAT 10 Hz COSTS US (ESTIMATED from the table above):**
- vs 20 FPS: 37.55 → 39.53 = **−5.0 % DS**
- vs the 24 FPS optimum: 37.55 → 40.63 = **−7.6 % DS**
- vs 1 FPS: **+14.3 % DS** — i.e. we are already 2/3 of the way up the curve
- **comfort is BETTER at 10 Hz: DC 0.38 vs 0.28 at 20 Hz = +36 %.**

Second finding from the same paper, and it matters more than the mean: **latency *variance* costs
more than latency *mean*.** Under real (not fixed) latency, `UniAD_rt` @2.31 FPS scored DS 30.69 and
`VAD_rt` @9.52 FPS scored 31.92 — a **−24.62 %** drop against the Bench2Drive baseline of 42.35 —
because *"UniAD exhibits a pronounced long-tail inference latency distribution"*, traced to a
collision-optimisation submodule with *"per-invocation latency exceeding 150 ms"* that fires
**exactly when obstacles appear ahead**. Removing it (`baseline − occ`) **raised** DS_rt by 16.94 %.

Supporting, all PUBLISHED primary:
- **CARLA Leaderboard runs at 20 Hz** — VERIFIED from the evaluator source itself:
  `frame_rate = 20.0  # in Hz`, `fixed_delta_seconds = 1.0 / self.frame_rate`
  (`carla-simulator/leaderboard/leaderboard/leaderboard_evaluator.py`, `_setup_simulation`).
  The brief's "20 Hz?" is correct.
- **Streaming Perception (Li/Wang/Ramanan ECCV 2020, 2005.10420)** — the canonical measured
  accuracy-vs-latency instrument: *"Standard metrics like object-detection average precision (AP)
  dramatically drop (**from 38.0 to 6.2**)"* when the same detector is scored in streaming rather
  than offline mode, on Argoverse-HD at 30 FPS. Also the methodological rule we should adopt:
  *"our evaluation is **hardware dependent** — the same algorithm on different hardware may yield
  different streaming performance. Such **hardware-in-the-loop testing** is commonplace in control
  systems and arguably vital."*
- **comma2k19 (1812.05752)**: openpilot's road camera logs at **20 Hz**; IMU at 100 Hz.
  (I could not reach a primary stating openpilot's *control* loop rate — `docs/SAFETY.md`,
  `docs/CARS.md` and `cereal/services.py` were all probed; SAFETY.md gives only an actuation-limit
  in seconds, services.py 404s at that path. **openpilot's 100 Hz control rate is UNVERIFIED — do
  not quote it.**)

**VERDICT Q2: WEAK-BY-DESIGN, and now quantified.** 10 Hz is **not** below the minimum for
closed-loop control; it sits at **~92 % of the achievable driving score** on the only measured
latency→DS curve I could find, and it is **better on comfort** than the 20–24 Hz optimum. The
sharper risk for us is **not the mean rate but the tail** — the paper's largest effect is a −24.6 %
DS collapse caused by latency spikes, not by a low average.

## Q3. Is 10 Hz plausible for 336.5 M params on a Jetson?

**⭐ We have already MEASURED this — for the tiny arm.** INHERITED by me from the claims register
(`Project Steering/GOALS_AND_CLAIMS.md` H-DEPLOY-2/3/6/7), artifacts
`products/P6-TanitDeploy/2026-08-23-v7tiny-baseline-profile/raw/*.json`, n=50/arm:

| Thor, v7-tiny, batch 1 | tick | implied Hz |
|---|---|---|
| fp32 eager | **12.529 ms** (12.369 / 12.519 across processes, ±4 %) | 80 |
| fp16 autocast | 15.214 ms (**1.21× SLOWER** — H-DEPLOY-3 REFUTED) | 66 |
| fp32 + CUDA-graph replay | **7.864 ms** (1.66×, bit-identical) | 127 |
| **TF32 + CUDA-graph replay** | **6.103 ms** (2.05× vs eager) | **164** |

And the shape of the cost: **`t(B) = 7.45 + 5.05·B` ms, R² = 0.99999** — i.e. at batch 1 **60 % of
the tick is a fixed launch/serialisation term, not arithmetic.** That is why fp16 and TF32 both
*lose* eagerly and TF32 *wins* 22 % only under graph replay.

⇒ For v7-tiny the 10 Hz target has **16× headroom**. ⚠️ **v7-tiny is NOT the 336.5 M scaled arm** —
quoting 6.10 ms for the scaled arm would be exactly the scope error CLAUDE.md warns about. But the
launch-bound fit is the useful prior: **a larger model grows the 5.05·B arithmetic term and leaves
the 7.45 ms fixed term alone**, so a 336.5 M arm should cost far *less* than proportionally at
batch 1. **Work item: re-measure `t(B)` for the scaled arm under graph replay before any claim.**

Published embedded evidence:
- **Alpamayo-R1 (NVIDIA, 2511.00088)** — the closest possible comparable: *"On-vehicle road tests
  confirm real-time performance (**99 ms latency**) and successful urban deployment"*, for a
  0.5–10 B VLA that emits **64 waypoints at 10 Hz over 6.4 s**. **NVIDIA's own shipped AV model runs
  at ~10 Hz.** Our 10 Hz operative rate is *exactly* theirs.
- **NVIDIA Jetson benchmarks page (vendor, PUBLISHED-SECONDARY)**: for **AGX Thor** only
  LLM/VLM token rates are published (Qwen2.5-VL 3B: 71.7 tok/s @ concurrency 1; JetPack 7.0,
  TensorRT 10.13); AGX Orin has MLPerf ResNet (0.64 ms single-stream). **No ViT-scale vision-encoder
  latency for Thor is published anywhere I could reach.** Probed the vendor page and swept the
  238-paper library for `Jetson|AGX Orin|Orin` — **zero hits**. This is a genuine gap in the
  literature, so our own Thor profile is the only admissible source.

**VERDICT Q3: CONSISTENT (MEASURED) for v7-tiny; UNKNOWN for the 336.5 M arm.** The claim is
plausible, is matched by NVIDIA's own deployed model at the same 10 Hz, and the one thing that would
settle it — a Thor `t(B)` sweep for the scaled arm — is a half-day of work we have already tooled.

## Q4. Does the field separate MODEL rate from CONTROL rate?

**Yes, and the regulator does too.** All PUBLISHED primary:

- **Alpamayo-R1 (2511.00088), verbatim:** *"the downstream **low-level vehicle controllers**
  typically smooth trajectory outputs to ensure consistent and stable execution on-vehicle"* — the
  model emits a 6.4 s / 10 Hz trajectory; a separate controller executes it. This is why they adopt
  a unicycle-dynamics action representation rather than raw waypoints.
- **InterFuser (2207.14024):** architecture is explicitly *"a **safety controller** which utilizes
  the interpretable intermediate features to constrain the low-level control within the safe set"* —
  network output → controller, two different stages.
- **UNECE / WP.29 ADS regulatory text** (banked `unece-r157`), ¶21–23, verbatim:
  > *"Driving consists of three categories of functions: **strategic, tactical, and operational**.
  > ... **Strategic** functions include activities such as determining a trip destination that do not
  > involve vehicle dynamic control. ... The **tactical** level involves manoeuvring the vehicle in
  > traffic ... **Tactical functions generally occur over a period of seconds.** ... **Operational**
  > functions include ... lateral vehicle motion control (steering) and longitudinal vehicle motion
  > control ... This operational effort involves **split-second reactions**, such as making
  > micro-corrections while driving."*

  ⇒ **Our hz_op 10.0 / tactical 2.0 / strategic 0.5 is a direct instantiation of the regulator's own
  three-level decomposition**, and our tactical rate (0.5 s) is *faster* than the regulation's
  characterisation ("a period of seconds"). This is a citation worth putting in the paper.
- ⚠️ I could **not** verify a numeric perception-to-actuation latency requirement in ISO/UNECE.
  Probed the banked ADS text for `response time|latency|reaction|0.35|seconds` — the only timing
  requirements found are **data-recording** intervals (−7 to +7 s around a triggering event) and
  occurrence-reporting windows, not a control-latency ceiling. **No regulatory latency number
  exists to cite. Do not invent one.**

**VERDICT Q4: CONSISTENT.** A 10 Hz planner feeding a faster low-level controller is the field
standard *and* the regulatory framing. **Cost to change: zero.** The only thing to add is to *say so
explicitly* in our docs — right now "hz_op 10.0" reads like a system rate rather than a planner rate.

---

# AXIS 7 — DATA SCALE AND DIVERSITY

## Q5. The field's data scale — and the ratio that matters

**Front-view HOURS (PUBLISHED primary, GenAD/OpenDV-2K Tab. 1, arXiv 2403.09630):**

| dataset | hours | front-view frames | countries | cities |
|---|---|---|---|---|
| Cityscapes | 0.5 | 25 k | 3 | 50 |
| KITTI | 1.4 | 15 k | 1 | 1 |
| Argoverse 2 (perception) | 4.2 | 300 k | 1 | 6 |
| Talk2Car | 4.7 | — | 2 | 2 |
| **nuScenes** | **5.5** | 241 k | 2 | 2 |
| **Waymo Open (perception)** | **11** | 390 k | 1 | 3 |
| Honda-HAD / HDD-Cause | 32 | 1.2 M | 1 | — |
| Honda-HDD-Action | 104 | 1.1 M | 1 | — |
| nuPlan (perception subset) | 120 | 4.0 M | 2 | 4 |
| ONCE | 144 | 7 M | 1 | — |
| OpenDV-YouTube | 1 747 | 60.2 M | ~40 | ~244 |
| **OpenDV-2K** | **2 059** | **65.1 M** | ~40 | ~244 |

Other primaries: **nuScenes 1000 scenes × 20 s = 5.5 h**, Boston + Singapore (1903.11027) ·
**nuPlan 1500 h**, 4 cities — Boston/Pittsburgh/Las Vegas/Singapore (2106.11810 §3; note the brief's
"~1200 h" is wrong, the paper says 1500 h) · **Waymo Open Motion 574 h**, >100 k scenes × 20 s @
10 Hz, 6 cities, 1750 km roadway (2104.10133 Tab. 1) · Lyft L5 1118 h, 1 city · Argoverse-1 320 h ·
**BDD100K 100 k videos × 40 s ≈ 1 111 h**, 720p/30 fps, >50 k rides, 6 weather × 6 scene × 3
times-of-day (1805.04687) · **ZOD**: 14 European countries, 705 km², 100 k Frames + 1 473 20-s
Sequences (2305.02008) · **comma2k19 33 h** — on **one 20 km stretch of CA-280** (1812.05752).

**World-model / generative family (PUBLISHED primary, Vista Tab. 1, arXiv 2405.17398):**

| model | hours | Hz | resolution |
|---|---|---|---|
| DriveDreamer | **5** | 12 | 128×192 |
| Drive-WM | **5** | 2 | 192×384 |
| WoVoGen | **5** | 2 | 256×448 |
| ADriver-I | 300 | 2 | 256×512 |
| GenAD | 2 000 | 2 | 256×448 |
| **GAIA-1** | **4 700** | 25 | 288×512 |
| **Vista** | **1 740** | 10 | 576×1024 |

GAIA-1 detail (2309.17080 §3–4): *"4,700 hours at 25Hz ... collected in **London, UK** between 2019
and 2023 ... approximately **420 M unique images**"*; 400 h held-out val; image tokenizer 0.3 B +
world model **6.5 B**, trained on **32× A100 80GB**. Vista: 1 735 h filtered OpenDV-YouTube +
nuScenes, **128× A100 for 20 k iters ≈ 8 days**.

**Waymo scaling-laws corpus (2506.08228 Tab. 1):** 59.8 M run segments · 373 B agents ·
**447 thousand hours** · 5.6 M miles · 541 M training examples.

### ⭐ WHERE WE SIT (ESTIMATED, arithmetic shown)

| reference | hours | our 26 h vs it |
|---|---|---|
| Waymo scaling-laws corpus | 447 000 | **1 / 17 192** |
| GAIA-1 | 4 700 | **1 / 181** |
| OpenDV-2K | 2 059 | **1 / 79** |
| Vista | 1 740 | **1 / 67** |
| **PhysicalAI-AV (our own corpus)** | **1 700** | **1 / 65 — we hold the other 98.5 %** |
| BDD100K | 1 111 | 1 / 43 |
| ONCE | 144 | 1 / 5.5 |
| comma2k19 (shipping product) | 33 | **0.79×** |
| **WOD-E2E (Waymo's own E2E benchmark)** | **12** | **2.2× LARGER** |
| Waymo Open perception | 11 | **2.4× LARGER** |
| nuScenes | 5.5 | **4.7× LARGER** |
| Drive-WM / DriveDreamer / WoVoGen | 5 | **5.2× LARGER** |
| Argoverse 2 perception | 4.2 | **6.2× LARGER** |

**VERDICT Q5: BEHIND the world-model family by 67–181×, and behind Waymo's scaling study by
~17 000× — but AHEAD of every standard AV perception benchmark by 2.4–6.2×, and ahead of the
published small driving world models by 5.2×.** The defensible sentence: *"26 h is 1/67 of the
smallest 'large' driving world model and 5× the largest 'small' one."*

**Cost to change (ESTIMATED, using the programme's own DE-C152 numbers):** going 26 h → 100 h means
~18 000 episodes instead of ~4 680. At the **corrected** v2ep consumer cost of **34.0 MB/ep**
(`v2_dataset.py` reads `*.v2ep.pt`, encoded — *not* the 117 MB raw `ep_*.pt`), that is **~612 GB**.
Build wall-clock scales ~3.8× from the re-timed B1 build (3.4–4.6 h) ⇒ **~13–18 h**, and PNG encode
is 70 % of it. ⚠️ The codec is load-bearing, not a speed knob: `v2_dataset.py:325` and
`slice_v2_cache.py` **refuse to sub-frame a lossy cache**, so a lossy build cannot later be
re-geometried. **This is the single cheapest lever on this axis** — the data is already ours.

## Q6. ⭐ THE CRUX — at what scale does a world model predict rather than regress to the mean?

Three primary curves, and they say something more precise than "more data".

**(a) DINO-WM (2411.04983, App. A.4.1) — the architecture family closest to ours (frozen encoder +
latent prediction + planning), PushT, MEASURED-in-paper:**

| dataset size (trajectories) | planning **SR** | SSIM | LPIPS |
|---|---|---|---|
| n = 200 | **0.08** | 0.949 | 0.056 |
| n = 1 000 | **0.48** | 0.973 | 0.013 |
| n = 5 000 | **0.72** | 0.981 | 0.007 |
| n = 10 000 | **0.88** | 0.984 | 0.006 |
| n = 18 500 | **0.92** | 0.987 | 0.005 |

**⇒ THIS IS THE MOST IMPORTANT TABLE IN THIS REPORT.** Over a 92× data increase, *prediction
fidelity barely moves* (SSIM 0.949 → 0.987, +4 %) while *control competence moves 11.5×*
(SR 0.08 → 0.92). **A world model whose prediction metrics look fine can be worthless for control,
and only more data closes that gap.** That is a published, measured instance of **exactly our
symptom** — the S-curve reproduced 97.9 % open-loop and ~0–5 % closed-loop.

**Where we sit (ESTIMATED):** 2,376 episodes × ~199 steps ≈ **n ≈ 2 400 of ~200-step trajectories**;
B1 ≈ n ≈ 4 700. On DINO-WM's own curve that is the **n = 1 000 → 5 000 band = SR 0.48–0.72** — i.e.
*precisely the regime where prediction has already saturated and control has not*. Our measured
open-loop/closed-loop dissociation is what that regime is *predicted* to look like.
⚠️ HYPOTHESIS, not proof: PushT is a 2-D manipulation task, not driving. But it is the only
published data-scaling curve for a frozen-encoder latent world model with a *behaviour* readout.

**(b) DriveVLA-W0 (2510.12796, Tab. 3) — in-house driving data, 70 k / 700 k / 70 M frames:**

| model | 70 k ADE / Coll% | 700 k | 70 M |
|---|---|---|---|
| TransFuser-50M | 2.5893 / 0.0894 | 1.7464 / 0.0563 | 1.2627 / 0.0472 |
| TransFuser-7B | 2.5757 / 0.0839 | 2.1391 / 0.0710 | 1.2244 / 0.0539 |
| VLA (VQ) action-only | 2.8520 / 0.0982 | 1.5424 / 0.0565 | 1.4829 / 0.0488 |
| **+ World Model** | 2.7482 / 0.0956 | 1.5985 / 0.0520 | **1.0563 (−28.8 %) / 0.0392 (−19.7 %)** |
| VLA (ViT) action-only | 3.1524 / 0.0950 | 1.4202 / 0.0462 | 1.1051 / 0.0359 |
| **+ World Model** | 2.5268 (−19.9 %) / 0.0834 | 1.3436 / 0.0513 | 1.0640 / **0.0302 (−15.9 %)** |

Verbatim: *"baseline models that rely on **sparse action supervision quickly show performance
saturation**, our DriveVLA-W0 models demonstrate sustained improvement ... **action-only supervision
cannot replicate the qualitative advantage provided by our dense world model objective**."*
**Our ~936 k (10 Hz) – 2.8 M (30 fps) frames sit between their 700 k and 70 M points.**

**(c) TransFuser++ (2306.07957, Tab. 5) — a controlled CARLA E2E dataset-size ablation on held-out
towns:** 185 k frames → DS 54 ± 1 / RC 92 ± 5; **555 k frames → DS 60 ± 6 / RC 98 ± 1**.
**3× data = +6 Driving Score (+11 %), at 3× training cost.**

**(d) The macro law (Waymo, 2506.08228):** cross-entropy is a power law in compute; **closed-loop**
metrics improve with scaling too (not just open-loop); *"optimal scaling requires increasing the
model size **1.5× as fast as the dataset size**"*; and the striking one — *"at the same training
compute budget, an **optimal LLM is ~50 times larger than an optimal motion forecasting model** ...
models need more training data to capture less common driving modes."*
⇒ **For driving, the literature says spend on DATA, not parameters.** Our 336.5 M is, by this
reading, already generous relative to 26 h.

**VERDICT Q6: no published threshold exists at which a driving world model "starts predicting rather
than regressing" — the transition is continuous, and I found no paper claiming a knee.** What the
literature *does* give us is better: the DINO-WM curve says the dissociation we measured is the
*expected* signature of the n ≈ 1–5 k regime, and DriveVLA-W0 says dense visual supervision is what
makes additional data pay. **Our architecture choice is right; our data volume is the binding
constraint on the behaviour readout, not on the prediction readout.**

## Q7. Diversity vs volume — measured evidence, and it is strong

**(a) ⭐ Codevilla et al. ICCV 2019 (1904.08980) — MORE DATA WAS WORSE.** CARLA100 = 100 h of expert
demonstrations; models trained on **2, 10, 50, 100 h**, 4 seeds. Verbatim:
> *"Our best results on most of the scenarios were obtained by using **only 10 hours** of training
> data, in particular on the 'Dense Traffic' tasks and novel conditions such as New Weather and New
> Town."* … *"Due to biases in the data, the results may get either **saturated or worse with
> increasing amounts of training data**."* … *"this can result in performance degrading as more data
> is collected, because **the diversity of the dataset does not grow fast enough compared to the main
> mode of demonstrations**. This phenomenon was not clearly measured before."*

The named mechanism is the **inertia problem from causal confusion**, whose failure rate *rises*
with data — and their fix was a **speed-prediction branch** (CILRS). ⚠️ **That is our own
speed-channel result** (REF-A 3.73 → 0.83 m fwd_ade, speed R² 0.61 → 0.965, INHERITED from memory),
found independently. It is also a published cousin of our **action echo**: a model that has learned
the dominant mode instead of the causal structure. See also banked `1905.11979` (Causal Confusion in
Imitation Learning) for the mechanism.

**(b) ⭐ WOD-E2E (2510.26125) — Waymo, who hold 447 000 h, built a 12-hour benchmark by curating for
rarity.** Verbatim: *"WOD-E2E contains 4,021 driving segments (**approximately 12 hours**),
specifically curated for challenging long-tail scenarios that are rare in daily life with an
**occurring frequency of less than 0.03 %**."*
**ESTIMATED from that number:** 12 h of <0.03 %-frequency events implies **~40 000 h of random
driving** would be needed to accumulate them by volume alone. **Curation beat volume by ~3 300×.**

**(c) Diversity axes that exist in the corpora:** BDD100K explicitly argues *"models trained on
existing datasets tend to overfit specific domain characteristics"* and ships 6 weather × 6 scene ×
3 time-of-day tags; ZOD spans 14 European countries / 705 km²; OpenDV-2K ~40 countries / ~244
cities. ⚠️ **I did NOT find a paper that ablates DIVERSITY at FIXED HOURS** — probed BDD100K, ZOD,
GenAD and the E2E ablation papers. The closest is (a), which ablates *volume* and attributes the
failure to diversity, and (b), which holds hours fixed and varies *rarity*. **Marking this UNKNOWN
rather than absent; a fixed-hours diversity ablation is a gap in the literature and would be a
publishable experiment for us.**

**VERDICT Q7: CONSISTENT with strong measured support — diversity/curation beats raw hours, and our
corpus already has the diversity.** Combined with §0, the conclusion is sharp: **the highest-value
axis-7 action is not "get more hours", it is "sample the 1,700 h corpus for coverage rather than
convenience"** — stratify the next build over the card's country / weather / time-of-day / traffic /
surface tags instead of taking the next N clips. **Cost: a selection script, not a data campaign.**
⚠️ It breaks parity (`e438721ae894` / skip-hash `f09e44db`), so it must be a **new named corpus arm**
evaluated alongside, never a re-selection of the existing one.

## Q8. Is 26 h defensible for a world-model DYNAMICS claim?

**YES — with the claim stated narrowly. Evidence:**

- **The published driving world-model floor is 5 hours**: DriveDreamer, Drive-WM and WoVoGen all
  published at **5 h** (Vista Tab. 1, PUBLISHED). We have **5.2× that**.
- **DINO-WM (2411.04983)** — the strongest architectural precedent for a *dynamics* claim — trains
  per-environment on **1 000–20 000 trajectories** (Rope 1 000×20 steps; Granular 1 000×20;
  Wall 1 920×50; PointMaze 2 000×100; Reacher 3 000×100; PushT 18 500; PushObj 20 000). Our 2 376
  episodes × ~199 steps is **inside that band and longer per trajectory than most of it**.
- **DreamerV3 (2301.04104)** establishes world-model results at **Atari100k = 400 k steps** and
  proprio benchmarks at 500 k steps — small-data world models are a normal, publishable object.
- **WOD-E2E is 12 h** and Waymo published a benchmark on it.

**What 26 h supports (defensible):** a *controlled, parity-held comparison of architectures on
latent dynamics prediction and its decodability* — rank ordering between arms, ablations, and
dissociation measurements (T0 diagnostics).
**What 26 h does NOT support (and we must not claim):** a driving-*performance* claim, a
generalisation claim to unseen geographies, or any statement that a negative result is about the
architecture rather than the data regime. DINO-WM's own curve puts us at SR ≈ 0.48–0.72 — **a weak
closed-loop number at our data scale is the expected outcome, not an architecture verdict.**

**VERDICT Q8: DEFENSIBLE for the narrow claim, INADMISSIBLE for a driving claim.** The one sentence
to put in the paper: *"26 h is 5.2× the published driving world-model floor and 2.2× Waymo's own
end-to-end benchmark, which supports a controlled dynamics comparison; it is 1/67 of Vista and
1/17 000 of Waymo's scaling corpus, which is why we make no driving-performance claim."*

---

## Cross-cutting: what this costs US to change

| lever | cost | expected return | evidence |
|---|---|---|---|
| **Say "planner rate" not "system rate"** in docs | zero | removes a false criticism | 2511.00088, unece-r157 |
| **Measure `t(B)` on Thor for the 336.5 M arm** under graph+TF32 | ~half a day, tooling exists | settles Q3 outright | H-DEPLOY-3/7 artifacts |
| **Add a latency-TAIL metric** (p99 tick, not mean) | small | the largest measured effect in 2601.07393 (−24.6 % DS) | 2601.07393 §4.1 |
| **Stratified re-sample of PhysicalAI over the card's tags** at fixed hours | a selection script | the only *measured* lever that beats volume | 1904.08980, 2510.26125 |
| **26 h → 100 h** (~18 000 eps) | ~612 GB, ~13–18 h build, PNG codec mandatory | moves us from DINO-WM n≈2.4 k to n≈18 k (SR 0.48→0.92 in *their* units) | DE-C152, 2411.04983 |
| Raise 10 Hz → 20 Hz | not needed | +5 % DS, **−26 % comfort** | 2601.07393 Tab. 2 |

## Things I could NOT establish (probed twice; do not treat as absence)

1. **openpilot's control-loop rate** — `docs/SAFETY.md`, `docs/CARS.md`, `cereal/services.py` all
   probed; the "100 Hz control / 20 Hz model" figure is **UNVERIFIED**. comma2k19 confirms the
   camera at 20 Hz only.
2. **Jetson Thor ViT-scale latency in the literature** — vendor page has LLM/VLM tokens only;
   a sweep of all 238 banked papers for `Jetson|AGX Orin|Orin` returned **zero hits**.
3. **Generation speed for Vista / GenAD / any driving world model** — not reported; only GAIA-1
   states, qualitatively, that it is not real-time.
4. **A fixed-hours diversity ablation** — none found; genuine literature gap.
5. **A numeric regulatory perception-to-actuation latency ceiling** — the banked ADS/R157 text has
   none.

## Banked this session (all `--verify` clean)

New: `2005.10420` · `2405.17398` · `2403.09630` · `1912.04838` · `1805.04687` · `2305.02008` ·
`1812.05752` · `2207.14024` · `2601.07393` · `2506.08228` · `2306.07957`.
Re-cited (already banked): `2212.10156` · `2303.12077` · `2402.13243` · `2405.19620` · `2411.15139` ·
`paradrive-cvpr2024` · `2511.00088` · `2510.12796` · `2309.17080` · `2411.04983` · `2301.04104` ·
`1904.08980` · `1905.11979` · `2510.26125` · `1903.11027` · `2106.11810` · `2104.10133` ·
`2301.00493` · `unece-r157`.

⚠️ **Tooling note for the orchestrator:** `kb_add.py` failed 3× with a lockfile traceback
(`os.open(..., O_CREAT|O_EXCL)` at `kb_add.py:115`) when other streams banked concurrently. It is
**lock contention, not a network error**, and it exits with a raw traceback rather than a retry.
Worth a retry-with-backoff.
