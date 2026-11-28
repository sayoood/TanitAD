# Input-pipeline design review — literature stream, axes 2 (FOV / camera count) and 5 (temporal context)

Date 2026-08-30 · branch `agent/arch-inf-20260803` · all primaries banked in `TanitAD Research Lab/Library/`
(`kb_add.py --verify` → **199 entries, 0 orphans, 0 problems**).

Evidence classes: MEASURED (ours) · PUBLISHED (primary, banked+cited) · PUBLISHED-SECONDARY (inadmissible)
· INHERITED · ESTIMATED · HYPOTHESIS.

⚠️ **Two search-summary claims were FALSIFIED by reading the primaries.** Recorded here because they are the
`PUBLISHED-SECONDARY` failure mode the standard warns about:
1. A search summary asserted a **UniAD front-only-vs-surround ablation ("1.23 m / 0.33 % vs 1.03 m / 0.31 %")**.
   **It does not exist.** Two probes of the UniAD primary (arXiv HTML v2 + the banked PDF) find **no camera-count
   or FOV ablation at all**. The number appears to have been synthesised by the summariser. It must not be quoted.
2. A search summary and a WebFetch both rendered NAVSIM's history as **"1.5 s at 22 Hz"**. The primary says
   **"1.5s at 2Hz"** (p5). 22 Hz is an artifact of the adjacent phrase "a reduced frequency of 2Hz".

---

## OUR SETUP (INHERITED from the brief; the stack-construction facts below are MEASURED in our code)

- Single front camera, **120.0000° HFOV × 45.4556° VFOV** (cylindrical in azimuth, rectilinear in elevation),
  256×640. No surround/rear/LiDAR/HD-map/route.
- (a) Encoder: 3 consecutive frames channel-stacked to 9 channels = **0.2 s** inside one encoder forward.
- (b) World model: `--window 6` at dt 0.1 s ⇒ the window spans **0.5 s**.
- ⇒ **Total raw-pixel history = 0.7 s** (earliest frame t0−0.2, latest t0+0.5; 8 distinct frames at 0.1 s).
- ⭐ **Horizon = 60 steps × 0.1 s = 6.0 s**, NOT 2.0 s. `stack/tanitad/models/v6.py:148-156`, "THE BINDING
  HORIZON SPEC (PI 2026-08-11)": `PLAN_STEPS = 60`, `OP_BAND_S = (0.0, 2.0)`, `TAC_BAND_S = (2.0, 6.0)`.
  **2.0 s is only the OPERATIVE BAND**; the old 20-step/2.0 s value is superseded.
  *(Corrected mid-review by the measurement stream; this changes Q8 materially — see there.)*
- Rates hz_op 10.0 / hz_tac 2.0 / hz_str 0.5.

**MEASURED (ours)** `stack/tanitad/data/comma2k19.py:633-641` — `stack_frames` stacks strictly **consecutive**
frames (`parts = [vid_u8[i : T-(n-1)+i] for i in range(n_stack)]`); there is **no stride parameter**. Its
docstring states the design intent verbatim: *"D-015: n_stack=3 at 10 Hz -> the encoder sees [t-200ms, t-100ms, t]
in one 9-channel input, making acceleration/curvature observable per input."*
`stack/tanitad/data/v2_dataset.py:36-38` — `T_out = len(poses) - (n_stack-1)`, i.e. the per-episode warm-up cost
is exactly `n_stack-1` rows.

---
---

# AXIS 2 — FIELD OF VIEW / CAMERA COUNT

## Q1. The field's standard rig, and the MEASURED cost of front-only vs surround

### ⭐ The headline: the strongest current planning benchmark is effectively FRONT-ONLY.

**PUBLISHED (NAVSIM, arXiv 2406.15349, NeurIPS 2024 D&B; banked)** — the dataset offers 8 cameras, but the
**default agent input is a forward-facing stitch**:

> "TransFuser [12], which uses three cropped and downscaled forward-facing cameras, concatenated into a
> 1024 × 256 image" (§4.2, p7)

and Table 2's caption: *"Its camera FOV is around 140°."*

**PUBLISHED (Hydra-MDP, arXiv 2406.06978; banked)** — the **winner of the CVPR 2024 NAVSIM challenge** uses the
same shape: *"the front-view image is concatenated with the center-cropped front-left-view and front-right-view
images, yielding an input resolution of 256 × 1024 by default"* (§3.2).

⇒ **The reference input of the benchmark AND of its winner is a ~140° forward-facing strip at 256×1024.**
Our 120° / 256×640 is the same object at 0.63× the width. **This is a major point in TanitAD's favour and the
brief's hypothesis is CONFIRMED.**

### The controlled FOV ablation — MEASURED, in-architecture, single paper

**PUBLISHED — NAVSIM Table 2 (TransFuser ablations, `navtest`), read from the banked PDF p8:**

| Config | Setting | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | **PDMS↑** |
|---|---|---|---|---|---|---|---|
| A1/A2/A3 | **default ~140°, 3 cams** (3 seeds) | 98.0/97.7/97.9 | 91.3/92.8/93.0 | 94.2/92.8/93.1 | 100 | 78.1/79.2/79.3 | **83.3 / 84.0 / 84.4** |
| **C1** | **Camera FOV 60° (1 camera)** | 96.7 | 90.2 | 90.9 | 100 | 75.8 | **80.3** |
| C2 | 160° (3 cameras) | 97.6 | 91.4 | 92.7 | 100 | 78.1 | **82.8** |
| C3 | 240° (5 cameras) | 97.8 | 92.5 | 93.0 | 100 | 79.2 | **84.1** |

Seed std is **±0.56 PDMS** (stated in the paper), default seed mean **83.9**.

The paper's own reading (p8–9):
> "only considering the front camera (C1) with a 60° FOV leads to a small drop in almost all subscores, compared
> to our default setting of three cropped and concatenated images with a FOV of 140°. However, expanding the FOV
> with additional cameras does not result in substantially improved scores."

**Arithmetic.** 60° → 160° = **+2.5 PDMS** over 100° ⇒ ≈ **+0.025 PDMS/degree** in that range.
160° → 240° = +1.3 PDMS, and 240° (84.1) is **inside** the default 3-seed range (83.3–84.4) ⇒ **beyond ~140–160°
the curve is flat within seed noise.** Whole 60°→240° sweep spans 3.8 PDMS, of which ±1.1 is 2σ seed noise.

**ESTIMATED for us:** 120° is 60° above C1 ⇒ ≈ 80.3 + 1.5 = **~81.8**, i.e. **~2 PDMS below the 140° default**.
Because the curve is concave and 120° is much nearer 140° than 60°, the true gap is likely **≤1 PDMS, i.e. within
~2σ of seed noise.** (Linear interpolation on a saturating curve under-estimates the value at 120°, so this is the
conservative direction.)

### The architectural comparison — surround + video buys nothing here

**PUBLISHED — NAVSIM Table 1 (`navtest`), banked PDF p7:**

| Method | Ego | Image | LiDAR | Video | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | **PDMS↑** |
|---|---|---|---|---|---|---|---|---|---|---|
| Constant Velocity | ✓ | | | | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP | ✓ | | | | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| **LTF** (front ~140°, **camera-only, single frame**) | ✓ | ✓ | | | 97.4 | 92.8 | 92.4 | 100 | 79.0 | **83.8** |
| TransFuser (front ~140° + LiDAR) | ✓ | ✓ | ✓ | | 97.7 | 92.8 | 92.8 | 100 | 79.2 | **84.0** |
| **UniAD** (**8 surround cams, 4 temporal frames**) | ✓ | ✓ | | ✓ | 97.8 | 91.9 | 92.9 | 100 | 78.8 | **83.4** |
| **PARA-Drive** (8 surround, 4 temporal) | ✓ | ✓ | | ✓ | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | **84.0** |
| Human | | | | | 100 | 100 | 100 | 99.9 | 87.5 | **94.8** |

**PUBLISHED (same paper, p7):** *"Both UniAD and PARA-Drive use a BEVFormer backbone [31], which encodes the eight
surround-view 1920 × 1080 camera images over four temporal frames"* — and the authors' interpretation:
> "we suspect that surround-view cameras used by UniAD and PARA-Drive, and LiDAR input of TransFuser, are less
> important than the wide-angle front camera which is the only input of LTF."

⇒ **Front-only, camera-only, SINGLE-FRAME (LTF, 83.8) ≥ 8-camera surround with 4-frame video (UniAD, 83.4)**, and
is within seed noise of the best sensor agent (84.0). The full 8-camera surround rig buys **≤0.2 PDMS**, i.e.
**nothing measurable**, on this benchmark.

### ⚠️ The caveat that must travel with every one of these numbers

The NAVSIM authors state the mechanism themselves (p7): *"Due to the definition of at-fault collisions, which
discard certain rear-collisions into the ego vehicle, we suspect that surround-view cameras … are less important."*
**The metric structurally discounts exactly the events rear/side sensing would catch.** Two further limits:
NAVSIM is **non-reactive** — *"no environmental feedback is provided to the driving agent"* over the h = 4 s horizon
(p4) — so it is a pseudo-closed-loop score, not true closed-loop; and h = 4 s bounds how far a lane-change decision
is exercised.

⇒ The correct claim is **"front-only is not the bottleneck for at-fault-forward safety on a 4 s non-reactive
horizon"**, *not* "surround view is unnecessary."

### Why the absence of a UniAD/VAD camera ablation matters less than it looks

**PUBLISHED (arXiv 2312.03031, "Is Ego Status All You Need…", CVPR 2024; banked)** — Ego-MLP, using **only ego
status and no sensor input at all**, is *"on par with state-of-the-art methods in terms of existing L2 distance and
collision rate metrics"* on nuScenes open-loop planning. ⇒ **The nuScenes L2/collision metric cannot resolve a
camera-configuration question** — it is saturated by ego status. Any nuScenes-based front-vs-surround comparison,
had one existed, would have been uninformative. This is why NAVSIM's PDMS ablation is the admissible instrument.

**VERDICT Q1 — CONSISTENT (strongly).** The field's *benchmark* rig is 6–8 cameras; the field's *agent input* on
the strongest planning benchmark is a ~140° forward stitch, and the measured penalty for narrowing further is
small and saturating. Our 120° sits close to the default, not near the 60° floor.

---

## Q2. Which manoeuvres does front-only structurally fail?

**VERDICT — UNKNOWN (no direct measurement found).** Probed ≥2 ways: the ablation/results tables of NAVSIM,
UniAD, VAD, SparseDrive, Hydra-MDP and PARA-Drive, plus targeted queries for manoeuvre-decomposed collision
rates. **No paper found decomposes collision rate or planning error by manoeuvre for front-only vs surround.**
This is a genuine gap in the literature, not merely in our search — but per the standard I state it as UNKNOWN,
not as absence.

The closest admissible anchor is **regulatory, not empirical**:

**PUBLISHED — draft UN GTR on Automated Driving Systems, ECE/TRANS/WP.29/2026/139 (WP.29 199th session,
June 2026; banked).** Its OEDR analysis (§2.1.3, Table 1, p72–73) enumerates the events an ADS must detect and
respond to. Those whose **initiation** lies outside a 120° forward FOV:
*"Cutting in" · "Cutting out" · "Changing lanes" · "Encroaching adjacent vehicle" · "Entering roadway" ·
"Riding in adjacent lane" (cyclists) · "Static/moving in adjacent lane"*.
Also listed: *"Encroaching opposite vehicle"*, *"Lead vehicle decelerating/stopped/accelerating"* — these **are**
inside our FOV.

⚠️ **BANKING DEFECT FOUND — flag for the Master Mind.** The library key **`unece-r157`**, filename
`unece-r157_UNECE-R157-WP-29-ALKS-regulation.pdf`, is **NOT UN Regulation No. 157 (ALKS)**. Page 1 identifies it as
**ECE/TRANS/WP.29/2026/139, "Proposal for a new United Nations Global Technical Regulation on Automated Driving
Systems"** — a different document, from a different instrument family, dated June 2026. The PDF is a legitimate
primary and is the *better* source for OEDR, but **any citation of "R157" resolving to this key is a
misattribution.** Recommend re-keying. *(Caught only by reading p1 — `--verify` passes, because the bytes match
the hash; this is the "a tool reporting success is not evidence its output is right" class, with the
identity rather than the content wrong.)*

⇒ **HYPOTHESIS (ours, untested):** the front-only deficit concentrates in *merge/lane-change initiation* and
*unprotected turns with cross traffic*, and is largely absent in car-following and lead-vehicle response. NAVSIM's
DAC and EP subscores are where C1 loses most (DAC −1.2 to −2.8, EP −2.3 to −3.5 vs the default seeds) while NC
(no at-fault collision) loses only 1.0–1.3 — **weakly consistent** with "front-only degrades path/progress choice
more than forward collision avoidance." Not decisive; the subscore decomposition is not a manoeuvre decomposition.

---

## Q3. Comma.ai / openpilot and the front-camera-only counter-examples

**PARTIAL — the strongest evidence here is academic, not comma.**

**⭐ PUBLISHED (Vista, arXiv 2405.17398, NeurIPS 2024; banked)** — a leading generalizable **driving world model**,
and it is **front-facing monocular**: 10 Hz, **576×1024**, generating K = 25 frames (§3.1, Table 1, p2). This is a
direct architectural precedent for TanitAD's claim class: a front-mono camera is accepted by the field as an
adequate input for a *world-model dynamics* result.

**PUBLISHED (GAIA-1, arXiv 2309.17080; banked)** — world model 6.5 B params, image tokenizer at **288×512**,
trained on *"4,700 hours at 25Hz of proprietary driving data collected in London"*. **Camera count is not stated
in the primary** (2 probes) — I do **not** assert front-only for GAIA-1.

**comma.ai — WEAK / PUBLISHED-SECONDARY, and a genuine counter-signal.** The comma four product page states
*"Dual-cam 360° vision and a narrow cam to see far-away objects"*. That is **marketing copy, not a spec**: no FOV
in degrees, no resolution, no primary. Two attempts to recover the openpilot model's input spec from the repo
(`docs/CARS.md`, `selfdrive/modeld/models/README.md`) failed (404 / wrong document). **I could not obtain a
primary openpilot number and therefore quote none.**
⚠️ **But the direction is honest and against us:** the shipped stack that was historically the canonical
"front-camera-only works" example is **adding cameras and claiming 360°**. Treat "comma proves front-only is
enough" as **no longer supportable** without a primary.

⚠️ **CILRS (arXiv 1904.08980, banked) is front-only but is a CARLA behaviour-cloning study**, not evidence about
real-world FOV sufficiency; TransFuser (2205.15997) is front+LiDAR. Neither adds a measured FOV number.

**VERDICT Q3 — CONSISTENT for the world-model claim (Vista is a direct precedent); WEAK as a "shipped stacks are
front-only" argument, which the comma four hardware actively undercuts.**

---

## Q4. The honest question — does single-front-120° invalidate our claims?

The two failure modes must be separated, and they land differently.

| claim class | verdict | basis |
|---|---|---|
| **"Cannot learn dynamics"** | **FALSE — CONSISTENT** | Vista (PUBLISHED) is a front-mono 576×1024 driving world model at 10 Hz predicting 2.5 s. A front-mono camera is an accepted input for a world-model dynamics result. Our 120°/256×640 is the same class at lower resolution. |
| **"Cannot plan safely"** | **TRUE — WEAK-BY-DESIGN** | The draft UN GTR OEDR list (PUBLISHED) requires response to cut-in/cut-out/lane-change/adjacent-lane encroachment, whose initiation is outside 120°. No amount of 120° front data makes a blind-spot check possible. |
| **"Competitive open-loop planning on a forward-facing benchmark"** | **CONSISTENT** | NAVSIM Table 1: front-only camera-only LTF 83.8 ≥ surround UniAD 83.4; Table 2: 60°→140° costs ~3.6 PDMS, beyond 160° flat. |

**Which of TanitAD's claims each affects:**
- ✅ **Unaffected:** world-model / latent-dynamics claims; the encoder-representation claims; T0 WM diagnostics;
  the operative (10 Hz) planner's forward car-following and lateral-tracking behaviour; **the hierarchy thesis
  itself** (nothing about hierarchy requires 360°).
- ⚠️ **Bounded, must be stated:** tactical-manoeuvre claims involving **lane change and merge** — the classifier
  can be trained (labels may use privileged signals per the binding rule) but at **inference, vision-only at 120°,
  the adjacent-lane-behind state is not observable**. A lane-change decision head that scores well is either
  (a) reading the ego's own future (a leak of the C6/nav-echo family) or (b) deciding without the information a
  safe lane change requires. **This is a leak-audit item, not merely a scoping caveat.**
- ⛔ **Not claimable at all:** any *deployment-grade safety* or *collision-rate* claim, and any ADS-certification
  framing. Front-only 120° cannot satisfy the OEDR set.
- ⚠️ **The VERTICAL field is a second, separate limit and it is easy to forget.** 45.4556° VFOV bounds what the
  rig can see of **traffic lights, overhead gantries and signage** — the strategic tier's natural inputs. This is
  *independent of* the horizontal argument, and none of the NAVSIM evidence above speaks to it: NAVSIM's ablation
  varies **horizontal** FOV only (60°/160°/240°). ⇒ **The "NAVSIM says front-only is fine" defence covers HFOV
  and does NOT transfer to VFOV.** Quoting it for the vertical limit would be a scope error of the `df` family.

**Recommended framing for the paper (HYPOTHESIS, for PI decision):** state the FOV as a *deliberate research
scoping* with the NAVSIM ablation as the justification (front ~140° is the benchmark's own default and its
winner's input; 60°→140° is worth ~3.6 PDMS; >160° is flat), and pre-commit that **no safety/collision claim is
made**, only dynamics + relative planning quality. **The field will accept this** — it is exactly what NAVSIM,
Hydra-MDP and Vista do. What it will *not* accept is a collision-rate or safety claim on this rig.

---
---

# AXIS 5 — TEMPORAL CONTEXT

## Q5. How much history do published stacks use, at what stride?

All rows PUBLISHED (primary, banked) unless noted.

| system | history | stride / rate | frames | horizon | source |
|---|---|---|---|---|---|
| **BEVFormer** | **1.5 s** | 0.5 s (2 Hz) | 4 | perception | 2203.17270 Tab.7/8 (default = 4 frames) |
| **UniAD** | **1.5 s** | 0.5 s (2 Hz) | 4 | 3 s plan | via NAVSIM p7: *"eight surround-view … over four temporal frames"* |
| **VAD** | same (BEVFormer backbone) | 0.5 s | 4 | 3 s plan | 2303.12077: *"Given multi-frame and multi-view image input"*, follows BEVFormer |
| **BEVDet4D** | **0.5 s** | 0.5 s | **2** | perception | 2203.17054 |
| **NAVSIM** | **1.5 s** | 0.5 s (2 Hz) | 4 (1 + 3 past, optional) | **h = 4 s** | 2406.15349 p5, p4 |
| **Vista** (world model) | **0.2 s** | 0.1 s (10 Hz) | **3** | 2.5 s (25 fr) | 2405.17398 §3.1 |
| **GAIA-1** (world model) | window T = 26 @ 6.25 Hz = **4.16 s total** (ctx+pred) | 0.16 s | 26 | sliding window | 2309.17080 |
| nuScenes prediction | **2 s** | 2 Hz | — | **6 s** | devkit task def. |
| Argoverse 2 MF | **5 s** | 0.1 s (10 Hz) | 50 | **6 s** | 2301.00493 |
| Waymo Open Motion | **1.1 s** (10 hist + 1 current) | 0.1 s (10 Hz) | 11 | **8 s** | 2104.10133 |
| **TanitAD** | **0.7 s** (0.5 window + 0.2 stack) | 0.1 s (10 Hz) | **8** | **6.0 s** (op band 0–2, tac band 2–6) | MEASURED (`v6.py:148-156`, `comma2k19.py:633`) |

**PUBLISHED verbatim, Argoverse 2 (p6):** *"Each scenario includes a local vector map and 11 s (10 Hz) of
trajectory data … The first 5 s of each scenario is denoted as the observed window, while the subsequent 6 s is
denoted as the forecasted horizon."*
**PUBLISHED verbatim, WOMD:** *"forecasting horizon is 8 seconds into the future"* (p3); baselines *"encode a
1-second history of observed state"* (p6); examples are *"9.1 second windows"* (p12) ⇒ 1.1 s + 8 s.
**PUBLISHED verbatim, nuScenes:** *"Up to two seconds of past history can be used"*; *"The predictions are
6-seconds long and sampled at 2 hertz."*

## Q6. ⭐ THE KEY COMPARISON — benchmark conventions vs our 0.6–0.7 s

**Two different answers, and conflating them is the trap.**

**(i) On ABSOLUTE history span: BEHIND.** Every camera-based perception/planning stack in the table converges on
**1.5 s** (BEVFormer, UniAD, VAD, NAVSIM — all 4 frames at 2 Hz). We have **0.7 s**, i.e. **47 %** of the field's
convention. Motion-forecasting benchmarks are further ahead still (2 s / 5 s).

**(ii) On the history:horizon RATIO: CONSISTENT — we are NOT an outlier. Waymo Motion sits essentially where
we do.** *(Recomputed at the corrected 6.0 s horizon.)*

| system | history : horizon | ratio |
|---|---|---|
| Vista | 0.2 : 2.5 | **0.080** |
| **TanitAD** | **0.7 : 6.0** | **0.117** |
| **Waymo Open Motion** | **1.1 : 8** | **0.1375** |
| nuScenes prediction | 2 : 6 | **0.333** |
| NAVSIM | 1.5 : 4 | **0.375** |
| Argoverse 2 | 5 : 6 | **0.833** |

**PUBLISHED, verified from the primary (2104.10133):** WOMD's *"forecasting horizon is 8 seconds"* (p3), baselines
*"encode a 1-second history"* (p6), examples are *"9.1 second windows"* (p12) ⇒ 1.1 s + 8 s ⇒ **0.1375**.

⇒ **Our 0.117 is within 15 % of the largest motion-forecasting benchmark's own convention, and above Vista's.**
Stated plainly: **on the history:horizon ratio TanitAD is NOT an outlier and NOT behind.** The brief hypothesised
a BEHIND finding here; **the primaries refute it.**

⚠️ **But the corrected horizon makes the ABSOLUTE finding worse, not better.** Predicting **6.0 s** from **0.7 s**
is an **8.6:1 extrapolation** — the most aggressive ratio in the table except Vista's (12.5:1, and Vista predicts
*video*, not *decisions*). Where Waymo's 0.1375 is reached with **1.1 s** of history, ours is reached with 0.7 s.
**The ratio is fine because our horizon is long, not because our history is adequate.** Both findings stand and
they are different: *ratio CONSISTENT, absolute span BEHIND.*

⚠️ **A units caveat that cuts in our favour, stated for fairness.** Argoverse/Waymo/nuScenes histories are
**tracked agent state** (positions, velocities, headings of *other* agents) — that channel carries no scene
context, so a long history is the only way to infer intent. Ours is **raw pixel/latent** history, where each single
frame already carries scene context. The two are not the same unit and 5 s-of-tracks ≠ 5 s-of-pixels. **But the
part that does transfer is exactly the part we are short on: other-agent intent inference** (see Q8).

⚠️ **Argoverse 2's own designers treat SHORT history as the harder task, not the better one** (p6):
*"Future Argoverse releases could continue to increase the problem difficulty by reducing observation windows and
increasing forecasting horizons."* Short history is a difficulty knob — which means our 0.7 s is a
*self-imposed handicap*, not a neutral choice.

**VERDICT Q6 — split: CONSISTENT on ratio, BEHIND on absolute span (0.7 s vs the field's 1.5 s convention).**

## Q7. ⭐ The history-length ablation — the number most wanted

**PUBLISHED — BEVFormer Table 7 (arXiv 2203.17270, banked PDF p18), nuScenes val, "NDS of models using different
frame numbers during training".** nuScenes keyframes are 2 Hz ⇒ frame spacing 0.5 s.

| #Frames | history span | NDS↑ | ΔNDS | mAVE↓ | ΔmAVE | cumulative % of total NDS gain |
|---|---|---|---|---|---|---|
| 1 | 0.0 s | 0.448 | — | 0.802 | — | 0 % |
| 2 | 0.5 s | 0.490 | **+0.042** | 0.467 | **−0.335** | **61 %** |
| 3 | 1.0 s | 0.510 | +0.020 | 0.423 | −0.044 | **90 %** |
| 4 | 1.5 s | 0.517 | +0.007 | 0.394 | −0.029 | 100 % |
| 5 | 2.0 s | 0.517 | **+0.000** | 0.387 | −0.007 | 100 % (**saturated**) |

**Measured delta per second of history:** 0→0.5 s = **+0.084 NDS/s**; 0.5→1.0 s = +0.040 NDS/s;
1.0→1.5 s = +0.014 NDS/s; 1.5→2.0 s = **0.000 NDS/s**. The curve is steeply concave and **flat by 1.5 s**.

**Where we sit (ESTIMATED, interpolation on our 0.7 s):** NDS ≈ 0.498, i.e. **~72 % of the achievable temporal
gain captured, ~28 % left on the table.** Because the curve is concave, linear interpolation under-estimates:
the true figure is likely **72–80 %**. ⚠️ Two scope caveats: this is a **perception** metric (NDS/mAVE), not
planning; and their spacing is 0.5 s vs our 0.1 s, so this maps span-to-span, not frame-to-frame.

**⚠️ The important counter-result — history depth is NOT a law, it trades against fusion quality.**
**PUBLISHED (BEVDet4D, arXiv 2203.17054, banked):** *"reduces the velocity error by 62.9% from 0.909 mAVE to
0.337 mAVE"* using **two adjacent frames**, and the paper notes *"the velocity precision of BEVFormer is achieved
by fusing features from multiple adjacent frames (i.e. 4 frames in total)"*. **BEVDet4D's 2-frame mAVE (0.337)
beats BEVFormer's 4-frame mAVE (0.394 val / 0.378 test).** ⇒ A well-designed 2-frame fusion beats a mediocre
4-frame one. Do **not** read Table 7 as "more history is always better" — read it as "the *first* 0.5–1.0 s is
worth a lot, and how you fuse it matters at least as much as how much of it you have."

**Second ablation, same paper (Table 8, p18):** ego-motion alignment of history BEV features is worth
**51.0 → 51.7 NDS**. ⇒ *aligning* history is worth ~as much as the 4th frame. Relevant to us: our window elements
are in a moving ego frame.

**VERDICT Q7 — BEHIND, quantified: we capture ~72–80 % of the measured temporal benefit; ~20–28 % is available,
and it is reachable at zero compute (below).**

## Q8. What history is needed to observe what? Is 0.7 s enough for a **6.0 s** prediction?

| quantity | frames / span needed | our 0.7 s @ 10 Hz (8 frames) | which band needs it |
|---|---|---|---|
| position | 1 | ✅ | op |
| velocity | 2 | ✅ | op |
| acceleration | 3 | ✅ | op |
| jerk | 4 | ✅ | op |
| **lead-vehicle braking onset / intent** | ~1–2 s | ⚠️ **marginal** | **tac (2–6 s)** |
| **lane-change / merge initiation by another agent** | ~1–2 s | ⚠️ **marginal-to-insufficient** | **tac (2–6 s)** |
| **traffic-light phase change** | seconds | ❌ (also: no TL feature in our corpus, and 45.46° VFOV bounds overhead visibility) | str |

⭐ **The corrected horizon aligns the deficit exactly with the band structure.** Everything our 0.7 s history
*does* observe (position → jerk) is what the **operative band (0.0–2.0 s)** needs. Everything it *does not*
observe (intent onset) is what the **tactical band (2.0–6.0 s)** needs. The programme's binding horizon spec
therefore places 4 of its 6 seconds in a band whose dominant unknown — what other agents are about to do — is
the one quantity our history length is too short to resolve.

**⭐ PUBLISHED — Vista states our exact D-015 rationale, independently** (2405.17398 §3.1, p3):
> "motion of instances in the scene: position, velocity, and acceleration. Since velocity and acceleration are the
> first- and second-order derivative of position respectively, these priors can be entirely derived by using three
> consecutive frames for conditioning."

That is **verbatim the reasoning in our own `stack_frames` docstring** (*"n_stack=3 at 10 Hz -> … making
acceleration/curvature observable per input"*). **Convergent, independent validation of the 3-frame stack.**
⇒ **For the derivative-observability job, 0.2 s / 3 frames is PUBLISHED-adequate and our encoder is CONSISTENT.**

**But note what Vista's argument does NOT cover:** derivatives of *our* motion and of *currently visible* agents.
It says nothing about *intent onset*, which is where the field's 1.5 s lives. **The deficit is specifically in the
tactical/strategic tiers of our hierarchy — exactly the tiers the programme's thesis rests on.**

**⭐ PUBLISHED — GAIA-1 makes the trade explicitly, and in the direction we should copy** (2309.17080 §2.3):
> "25Hz to 6.25Hz. This allows the world model to reason over longer periods without leading to intractable
> sequence lengths."

⇒ **At a fixed sequence-length budget, the field's answer is to spend it on SPAN, not on DENSITY.** We currently
spend ours on density (10 Hz over 0.7 s). GAIA-1 spends 26 tokens over 4.16 s; BEVFormer spends 4 over 1.5 s.

**VERDICT Q8 — CONSISTENT for derivatives (Vista-validated, and sufficient for the whole operative band);
BEHIND for intent, and — with the corrected 6.0 s horizon — the deficit maps **precisely** onto the tactical band
(2.0–6.0 s), which is 4 of the 6 predicted seconds and the tier the programme's hierarchy thesis rests on.**

⚠️ **Falsifiable consequence, worth pre-registering:** if 0.7 s of history is the binding constraint, then error
should grow *disproportionately* past the 2.0 s band boundary — i.e. the tactical-band degradation should exceed
what a smooth extrapolation of operative-band error predicts. **A longer-history arm should improve the tactical
band far more than the operative band.** If Option C/D moves the operative band and not the tactical band, the
history hypothesis is wrong and the tactical deficit is something else (capacity, supervision, or the FOV limit of
axis 2). That is a cheap, decisive read, and it is free (see costs below).

---
---

# WHAT IT WOULD COST US TO CHANGE

## Axis 2 (FOV) — not changeable

**MEASURED constraint (INHERITED):** PhysicalAI-AV ships one front-wide 120° rig; there is no surround data.
Widening FOV is a **corpus** change, not a config change. **Cost: unbounded (new corpus).**
⇒ **Do not attempt.** Manage it as a *claims* boundary (Q4), not an engineering item. Zero-cost mitigations:
(1) state the scoping with the NAVSIM ablation as justification; (2) **audit the lane-change tactical head for the
"deciding without the information" leak** — that is a real, cheap, high-value work item.

## Axis 5 (history) — three options; two are FREE

Baseline: encoder forwards per training sample = W = 6 on 9-channel inputs; WM sequence length 6.

### Option A — extend the window at 10 Hz to 1.5 s (W = 6 → 16). **COST IS UNCERTAIN — do not reject on my figure.**

⚠️ **I had this wrong before the horizon correction and am flagging the retraction rather than quietly editing.**
With `PLAN_STEPS = 60`, the WM's sequence is **context 6 + rollout 60 = 66 steps**, not 6. So:

| component | 6 → 16 context | multiplier |
|---|---|---|
| **encoder forwards** (context only — the rollout is in latent space) | 6 → 16 | **2.67×** |
| **WM sequence** (linear/recurrent) | 66 → 76 steps | **1.15×** |
| WM sequence (if attention is full over the whole 66) | 66² → 76² | **1.33×** |

⇒ **Total is bounded between ~1.15× (WM-rollout-dominated) and ~2.67× (encoder-dominated)** and I **cannot
narrow it without measuring the encoder/WM split** — which I have not done and which is not my axis.
My earlier "2.4–2.7×, not affordable" assumed a 6-step WM and is **withdrawn**: with a 60-step rollout the
encoder is a much smaller share, and Option A may well be affordable.

- **MEASURED anchor (CLAUDE.md):** the live v6F run is **26.47 s/step** on Thor ⇒ somewhere in
  **~30–71 s/step**; a 30 k-step run goes ~220 h → **~250–590 h**. The low end is viable, the high end is not.
- ⇒ **WORK ITEM for the architecture stream:** profile the per-step encoder-vs-WM split. It is one measurement
  and it decides whether Option A is on the table at all. *(Root-cause class: pricing a change against the wrong
  denominator — the same family as C152, "price what the consumer actually reads".)*
- Option A remains **strictly worse than C/D on cost** (which are free) and is only worth revisiting if C/D
  measure well and more span is still wanted.

### Option B — dilate the WM window stride (0.1 s → 0.3 s, span 0.5 s → 1.5 s). **Free in FLOPs, but architectural.**
- Encoder forwards unchanged (6); WM sequence length unchanged. **Compute delta ≈ 0.**
- ⚠️ **But the WM's context stride and its rollout stride are the same variable** at hz_op 10.0. Decoupling them is
  a real (if contained) architecture change, and it changes what the WM's one-step transition *means*.
- ⚠️ Also perturbs the parity key if it changes window selection — must be checked against the canonical corpus rule.

### ⭐ Option C — widen the encoder's 3-frame stack stride. **RECOMMENDED. Zero FLOPs, ~5 lines.**
Change the stack from (t−0.2, t−0.1, t) to **(t−0.6, t−0.3, t)**, i.e. stride s = 3.
- **Same 9 channels, same encoder, same window, same WM.** Compute delta = **0**. Params delta = **0**.
- **Total raw-pixel span 0.7 s → 1.1 s** — landing where BEVFormer captures **~90–95 %** of its temporal gain
  (vs our current ~72–80 %).
- **MEASURED cost, from our own loader:** `v2_dataset.py:36-38` gives `T_out = len(poses) − (n_stack−1)`; with a
  stride the warm-up becomes `(n_stack−1)·s` rows. Episodes are ~199 rows.
  - now (s=1): 2 rows lost = **1.0 %**
  - s=3: 6 rows lost = **3.0 %** ⇒ over 2376 episodes, ~9.5 k of ~473 k windows. **Negligible.**
  - s=6: 12 rows = 6.0 %
- **Code change:** `stack/tanitad/data/comma2k19.py:640` —
  `parts = [vid_u8[i : T-(n-1)+i] …]` → `parts = [vid_u8[i*s : T-(n-1)*s + i*s] …]`, plus threading `s` through
  `build_episode` / `v2_dataset` / the cache builder. **No cache rebuild needed** — the v2 cache stores whole
  encoded episodes, so reading further back is free. *(⚠️ verify against `scripts/v2_compressed.py` before
  committing to "no rebuild" — that is the C152 consumer-loader rule and I have checked only the reader.)*
- ⚠️ **Risk, stated:** widening the stack loses the tight 0.1 s baseline *within* one encoder forward. Mitigated
  because the 10 Hz window still supplies 0.1 s resolution *across* elements — but this is a **HYPOTHESIS and must
  be measured**, not assumed.
- ⭐ **Supporting MEASURED-in-our-code argument:** `v2_dataset.py:133` already records that *"with a 3-frame stack
  consecutive latents share 2/3 of their input"* (the H-RANK-8 note, 2026-08-23). At 0.1 s spacing inside a 10 Hz
  window our stack is **largely redundant with the window itself**. Widening the stride de-duplicates it and buys
  span **for free** — this is the GAIA-1 trade (span over density) applied to our exact redundancy.

### Option D — asymmetric/log-spaced stack, e.g. (t−0.5, t−0.1, t). **Best of both; slightly more code.**
Keeps a 0.1 s baseline for instantaneous acceleration **and** a 0.5 s baseline for intent; total span 1.0 s.
Requires per-index offsets rather than a uniform stride (~5 lines more than C). Compute delta = **0**.

### Recommended pre-registered ladder (fits the `TanitAD_ValidateAIDesign` v7-tiny pattern)
| arm | stack offsets | total span | compute |
|---|---|---|---|
| **control** (must reproduce the banked number) | (t−0.2, t−0.1, t) | 0.7 s | 1.0× |
| **C** | (t−0.6, t−0.3, t) | 1.1 s | **1.0×** |
| **D** | (t−0.5, t−0.1, t) | 1.0 s | **1.0×** |
| **deliberate regression** | (t, t, t) — stack collapsed | 0.5 s | 1.0× |

The regression arm is the control that must read a **known worse** value; if it does not, the probe is not
measuring history at all. Read-out must include the **four metric families**, not ADE.

⭐ **Report every family SPLIT BY BAND (0–2 s operative vs 2–6 s tactical), not pooled over the 6 s horizon.**
That split *is* the hypothesis test from Q8: history length should move the **tactical band** far more than the
operative band. Pooling the two would average the effect away and is exactly the "single composite hides the
trade-off" failure the four-families rule exists to prevent. Expect the **longitudinal** family (lead-vehicle
intent) to move first, **tactical** second.

---

# SUMMARY OF VERDICTS

| # | question | verdict |
|---|---|---|
| 1 | standard rig / front-vs-surround cost | **CONSISTENT (strongly)** — NAVSIM's default agent input is a ~140° 3-cam forward stitch; the challenge winner too. 60°→140° = ~3.6 PDMS; >160° flat. Front-only LTF 83.8 ≥ surround UniAD 83.4. |
| 2 | manoeuvres front-only fails | **UNKNOWN** — no manoeuvre-decomposed measurement exists in the literature (≥2 probes). Regulatory OEDR list is the qualitative anchor. |
| 3 | comma/openpilot counter-example | **CONSISTENT via Vista** (front-mono world model, PUBLISHED). **WEAK via comma** — no primary obtained, and comma four now advertises 360°. |
| 4 | does 120° invalidate our claims | **Dynamics: NO (CONSISTENT). Planning-safety: YES (WEAK-BY-DESIGN).** Lane-change tactical claims need a leak audit. |
| 5 | published history lengths | table above; field convention **1.5 s / 4 frames @ 2 Hz** for camera stacks. |
| 6 | conventions vs our 0.7 s | **CONSISTENT on ratio** — at the corrected 6.0 s horizon our **0.117** ≈ Waymo Motion's **0.1375**; **not an outlier**. **BEHIND on absolute span** (0.7 s vs the camera-stack convention of 1.5 s, and vs Waymo's 1.1 s). |
| 7 | history-length ablation | **BEHIND, quantified** — BEVFormer Tab.7: +0.084 NDS/s for the first 0.5 s, saturating at 1.5 s; we capture **~72–80 %** of the gain. Counter-result: BEVDet4D's 2 frames beat BEVFormer's 4. |
| 8 | what history observes what | **CONSISTENT for derivatives** (Vista independently states our 3-frame rationale) and sufficient for the whole **operative** band; **BEHIND for intent**, which maps precisely onto the **tactical band (2.0–6.0 s)** — 4 of the 6 predicted seconds. |

**Single highest-value action:** Option C/D — widen the encoder stack's temporal stride. **Zero FLOPs, zero params,
~5 lines, ~3 % window loss**, moves us from ~72–80 % to ~90–95 % of the field's measured temporal benefit.
The programme's own H-RANK-8 note already documents the redundancy that makes it free.

**Banking:** all cited primaries banked and indexed; `kb_add.py --verify` → **199 entries, 0 orphans, 0 problems**.
**One defect found:** library key `unece-r157` holds ECE/TRANS/WP.29/2026/139 (draft UN GTR on ADS), **not**
UN R157 — re-key before anyone cites it as R157.
