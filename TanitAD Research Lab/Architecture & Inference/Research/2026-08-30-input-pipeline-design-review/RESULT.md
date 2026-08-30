# Input-pipeline design review — is TanitAD's DATA SETUP consistent with the field, or weak by design?

**Date:** 2026-08-30 · **Stream:** Research Lab special topic (Architecture & Inference) · **Commissioned by:** the PI
**PI ask, verbatim intent:** *"All our investigations in the last weeks about our WM were about its ARCHITECTURE and TRAINING MECHANISM, which is correct. I want you to deeply review in the literature of self-driving cars whether our SETUP regarding data, image resolution, encoder structure, feature patches, past-image feeding, dealing with distortion in end-to-end models, cycle time, and other setup aspects of feeding the data, is CONSISTENT with the field — or do we have any WEAKNESSES BY DESIGN?"*

**Companion artifact:** `PHASE1_OUR_SETUP.md` beside this file — our setup re-derived from source with `file:line`, and the corrections to the PI's own specification. **Read it for any "OURS" cell you want to check.**

**Evidence classes.** `MEASURED-SOURCE` (read out of our source this session) · `MEASURED-COMPUTE` (computed this session from formulas *in* our source; scripts in `raw/`) · `MEASURED-REGISTRY` · `PUBLISHED-PRIMARY (banked)` (read from a Library-banked PDF, sha-verified) · `PUBLISHED-SECONDARY` (**inadmissible** — flagged, never used to decide) · `INHERITED` · `ESTIMATED` · `HYPOTHESIS` · `UNKNOWN`.

---

## 0. The answer, in six sentences

1. **The projection — the axis the PI most suspected — is our strongest, not our weakest.** Cylindrical is *less* distorting than the pinhole rectification the field assumes (local anisotropy **1.084×** vs **2.000×**; horizontal sampling spread **1.000×** vs **4.000×**), it makes a camera yaw an *exact* pixel translation, and it is affine to within **0.021 px inside every 16-px patch**. Trained natively, a panoramic projection *beat* pinhole on matched scenes by **+6.2 % rel**.
2. **Resolution and patch size are fine, and one number settles it:** our **5.333 px/deg** horizontal is **identical to TransFuser's to four significant figures**, and the current NAVSIM SOTA (DriveVLA-W0, 93.0 PDMS) runs *below* us at ~4.0 px/deg.
3. **Single-camera 120° front is fine for everything except a safety claim** — NAVSIM's own controlled ablation moves **only +2.5 PDMS from 60°/1cam to 160°/3cam and is flat beyond**, and front-only single-frame LTF (**83.8**) beats 8-camera 4-frame UniAD (**83.4**).
4. **⭐ The real weaknesses are all in the parts nobody looks at: no augmentation of any kind on the supervised path, no recovery/perturbation augmentation, no ego-motion compensation of stacked frames, no frame-drop guard, no spatial anti-aliasing, an unfiltered ~3× temporal decimation, and a rig-correlated black region CONFIRMED LIVE on 73.25 % of the train cache.** Most are cheap; three are free; one is on a clock.
5. **⏱ If I change one thing, it is a DECISION and it is time-critical: fix the B1 build geometry before B1 builds.** The rig-correlated black region is **not a risk — it is live on every v6/v7 arm we have**, including the 336.5 M config E (`v2_subframe: null`, three independent sources), on **73.25 %** of the parity train cache. The fix is measured (`subframe_observability "176x624": {A: 1.0, B: 1.0}`) and it is a *geometry* choice, which is free at build time and a full retrain afterwards. **B1 is validated, prerequisites closed on Thor, and not yet built** — this is the only moment the choice is cheap, and rig B is the majority there too (2,723 / 4,719). See §4.0.
6. The best *experiment* to run alongside it is the **frame-stack ladder** (§4.2) — ~5 lines, 0 FLOPs — but it is a discriminating test, **not** a fix: our own `E-DEC-6` measured that collapsing the stack to `[f_t, f_t, f_t]` does not hurt ego decodability and slightly *improves* it, so the stack may simply be inert.

---

## 1. The scorecard

| # | AXIS | OURS (measured, `file:line`) | FIELD (published, banked) | VERDICT | COST TO FIX |
|---|---|---|---|---|---|
| 1 | **Input resolution** | 256×640 = **5.333 px/deg** horizontal, **5.632 px/deg** vertical. `train_v6_staged.py:6613-6614`; `calib.py:1244-1248`; `raw/proj_measure.py` | TransFuser 704×160 @132° = **5.333**; DriveVLA-W0 (NAVSIM SOTA 93.0 PDMS) **~4.0**; Hydra-MDP/DiffusionDrive 1024×256 ~7–8.5; UniAD 1600×900 24.8 | **CONSISTENT** horizontally · **BEHIND 1.82×** vertically | vertical: cache rebuild (§4.1) |
| 2 | **Small-object detectability** | pedestrian **17.3 px** @30 m, 10.4 px @50 m; traffic-light lamp **2.5 px** @30 m, 1.5 px @50 m | Zhang et al. (banked): 30–50 px → **~50 % miss**. BEVDet's 704×256 gives 31.6 px @30 m — only **1.8×** better and itself in the hard band; UniAD 71.7 px (4.1×) | **BEHIND**, but the *field* is barely better | resolution rebuild; or accept + scope claims |
| 3 | **FOV / camera count** | single front, **120.0000° H × 45.4556° V**, cylindrical. `calib.py:149-165` | **NAVSIM Table 2**: 60°/1cam **80.3** → 160°/3cam **82.8** → 240°/5cam **84.1** (seed σ ±0.56); *"expanding the FOV … does not result in substantially improved scores."* Table 1: front-only LTF **83.8** ≥ 8-cam 4-frame UniAD **83.4** | **CONSISTENT** for WM/dynamics/tactical · **WEAK-BY-DESIGN** for any safety claim | **not changeable** (corpus ships one rig) — manage as a claims boundary |
| 4 | **Projection / distortion** ⭐ | cylindrical, **linear in azimuth**, rectilinear in elevation. Local anisotropy **1.084×**; horizontal sampling spread **1.000×** exact; per-patch departure from affine **≤0.021 px**. `calib.py:906-935`; `raw/proj_measure2.py` | **0 of 8** canonical E2E stacks (UniAD/VAD/VADv2/SparseDrive/Hydra-MDP/NAVSIM/PARA-Drive/DiffusionDrive) even *discuss* projection; nuScenes devkit is `K@X`/z, strict pinhole. **Trans4PASS Tab. 4c**: trained natively, panoramic **beats** pinhole **+6.2 % rel** | **CONSISTENT — and ahead of the convention** | nothing to fix |
| 5 | **Patch size / tokenisation** | patch 16 → **16×40 = 640 tokens**. `config.py:43-46` | 16 is the modal choice (GAIA-1 576 tokens, DrivingGPT 576). ViT Tab. 6: L/32→L/16 = **+1.9 pp for ~4× FLOPs** (gain shrinks with scale) | **CONSISTENT** | p8 = 5.13× FLOPs, no cache rebuild |
| 6 | **Spatial pooling before the head** | **arm-dependent**: default **4×4 = 30°/bin**; `rdw8` line **4×8 = 15°/bin**; both `state_dim` 2048. `train_v6_staged.py:6636-6638`; `readout.py:78`; `MODEL_REGISTRY.md:3879` | **TransFuser pools to a single 512-d vector** — and it is the architecture DiffusionDrive (88.1) and Hydra-MDP (91.0) are built on. VAD's planner never reads the 200×200 BEV. Readout width is **uncorrelated with score** across the field | **CONSISTENT** — the axis is mis-framed | 4×8→8×16 ≈ free; **but see §3.6** |
| 7 | **Temporal context** ⭐ | encoder **0.2 s** (3 frames stacked); total distinct span **0.7 s** / 8 frames; horizon **6.0 s**. `comma2k19.py:633-641`; `v6.py:152-154` | Camera convention **1.5 s @2 Hz** (BEVFormer→UniAD/VAD/NAVSIM). **BEVFormer Tab. 7**: NDS 0.448→0.490→0.510→**0.517**, flat by 1.5 s | ratio 0.117 **CONSISTENT** (Waymo 0.1375) · absolute **BEHIND** (~72–80 % of gain) | ⭐ **FREE — ~5 lines, 0 FLOPs** (§4.2) |
| 8 | **Ego-motion compensation of history** ⭐ | **ABSENT.** `stack_frames` is a pure `torch.cat` of raw pixel tensors. `comma2k19.py:640-641` | **BEVDet4D**: ego-aligning the history is worth **51.0 → 51.7 NDS** — about as much as the 4th history frame | **WEAK-BY-DESIGN** | moderate — needs per-frame pose in the loader |
| 9 | **Augmentation (photometric/geometric)** ⭐ | **NONE on the supervised path.** Primitives exist (`v6.py:3521-3627`) but `w_t2_contrast` default **0.0** (`train_v6_staged.py:314`) and they only ever feed a contrastive projector head | BEVDet's image+BEV augmentation is a headline ablation; every planner stack augments | **WEAK-BY-DESIGN** | low — the code is already written and tested |
| 10 | **Recovery / viewpoint perturbation** ⭐⭐ | **ABSENT after 3 probes.** No ChauffeurNet/DAgger-style off-nominal synthesis on real data | ChauffeurNet: without perturbation the policy **cannot recover from drift** | **WEAK-BY-DESIGN — and it matches our headline failure** | high on real data (needs a re-render); see §3.10 |
| 11 | **Anti-aliasing** | **ABSENT.** `grid_sample(mode="bilinear")` at a **3.0×** decimation, no prefilter. `calib.py:990-991`. The *only* `antialias=True` in the stack is on a target-side path that is **off by default** | standard practice is to prefilter or use an area/antialiased resize | **WEAK-BY-DESIGN, second-order** — **1.11–1.31×** excess Δframe on realistic spectra (§3.11) | one code change + **cache rebuild (breaks parity)** |
| 12 | **Rig-correlated invalid region** ⭐⭐⭐ | **CONFIRMED LIVE.** `v2_subframe: null` on **every** v6/v7 arm incl. the 336.5 M config E (3 independent sources); rig B = **73.25 %** of the train cache at **8.89 %** masked (`parity_manifest.json` `rig_observability`). The rig-clean fix is measured — `subframe_observability "176x624": {A: 1.0, B: 1.0}` | Darcet et al.: ViTs **repurpose** low-information tokens into registers that *"discard spatial information"*; removing that recovers **+131.6 %** on LOST | **WEAK-BY-DESIGN — and active** | ⏱ **free at B1 build time; a retrain afterwards** (§4.0) |
| 13 | **Frame-drop / irregular dt** | **NO GUARD.** dt is a constant 0.1 (`train_v6_staged.py:7015`); ingest resamples onto a synthetic uniform grid via `searchsorted`, which **silently duplicates** a frame across a gap while pose/action are `np.interp`-smoothed across it. `physicalai.py:711-719` | — | **WEAK-BY-DESIGN (silent)** | very low — an assertion on real timestamp spacing |
| 14 | **Normalisation** | bare `/255.0`, fixed, whole 9-channel block. `_contract.py:51-56` | ImageNet constants are used **only** for the frozen DINOv3 teacher (`train_v6_staged.py:971-973`) | **CONSISTENT** (correct for from-scratch) | none |
| 15 | **Codec / colour** | **PNG lossless**, RGB, confirmed on the parity corpus. `parity_manifest.json:2514-2536` (`"codec": "png"`) | lossy is the norm; lossless is *better* | **AHEAD** | none |
| 16 | **Label-frame alignment** | newest frame of the stack, `poses[n_stack-1:]`, pinned by `tests/test_v2_dataset.py:93-95` | — | **CONSISTENT** | none |
| 17 | **Encoder recipe actually in use** ⭐ | **SPLIT, and I got this wrong first time.** v6F config E **does** use `--vit5-encoder` + 4 registers (`v6F-SW-30k.config.json:254-255`). The **v7-tiny ladder does not** (`champ30k`, `rdw8p30k`: plain `ViTEncoder`, learned APE, no registers, no RoPE) | Darcet et al.: **4 register tokens**, no measured downside. FishRoPE: angular PE +3.2 % rel mAP | **CONSISTENT at scale · WEAK-BY-DESIGN on the ladder** | **free** — one flag (changes param count ⇒ a declared arm) |
| 18 | **Cycle time / latency** | Thor end-to-end **60.30 ms p50** inside a 100 ms budget (v1/v5f, real weights); A40 18.75 ms; 4060 15.07 ms. ⛔ **zero** measurements for any v6/v7 arm (3 probes) | UniAD ~2 FPS class; the field targets 10–20 Hz | **CONSISTENT** as an engineering target · **UNKNOWN** for the current line | hours — and the ONNX blocker is historical (§3.18) |
| 20 | **Temporal decimation** ➕ | ~30.2 fps → 10 Hz by `searchsorted` nearest-index, **no temporal averaging** (`physicalai.py:711-719`) — a ~3× decimation mirroring the spatial one | — | **suspected WEAK-BY-DESIGN, UNMEASURED** | same class as row 11 |
| 19 | **Data scale / diversity** | *(§3.19)* | *(§3.19)* | *(§3.19)* | |

---

## 2. What I checked of the PI's own specification, and corrected

Full detail in `PHASE1_OUR_SETUP.md` §0. The three that change a conclusion:

| PI's belief | corrected fact | why it matters |
|---|---|---|
| "prediction horizon 20 steps at dt 0.1 = **2.0 s** dense" | **`PLAN_STEPS = 60` → `HORIZON_S = 6.0 s`** (`v6.py:152-154`). 2.0 s is only `OP_BAND_S`; the source comment reads *"Emission k scales **20 → 60**"* | our history:horizon ratio is **0.7 : 6.0 = 0.117**, not 0.3. The ratio turns out **fine** (Waymo 0.1375) — but 0.7 s is now supporting an **8.6:1 extrapolation** |
| "readout pools to **4 × 8**" | **arm-dependent**; the *default* is **4 × 4** (`--readout-grid-w` defaults to `None` → `gw = grid`) | 30°/bin vs 15°/bin. A blanket statement about "our readout" is unsafe |
| "rig A ~**100 %** observed" | rig A is **0.0017 % masked** too — `calib.py:1178-1180` says the 120° request over-runs some clips | the guard needs a *population*, not one clip |

➕ **Not in the spec and load-bearing:** VFOV is **45.4556°** and is **rectilinear, not cylindrical** (`calib.py:154-157`) — this is the fact that makes §4.1 computable, and it is *established from source*, not assumed.

---

## 3. The axes that matter, with the numbers

### 3.4 Projection — the priority axis, and it comes back clean

The PI asked me to treat this as the priority and said it was the axis he least understood. Two independent lines of work — my own computation from `calib.py`'s formulas, and a literature sweep of 10 banked primaries — converge on the same answer: **cylindrical is the better choice at 120°, and the distortion-aware-vision literature's problem is not ours.**

**Three quantities, kept distinct** (conflating them is easy and I nearly did):

| quantity | ours | pinhole at the same 120° field | equirect @±60° (the 360° papers' regime) |
|---|---|---|---|
| **local shape anisotropy** (aspect distortion) = `sec ε` | **1.000 → 1.084**, *azimuth-independent* | 1.000 → **2.000**, varies with azimuth | 2.000 |
| **horizontal sampling-rate spread** | **1.000 exact** | **4.000** at the edge | — |
| **vertical sampling-rate spread** = `sec² ε` | 1.000 → **1.176** | — | — |

⭐ **The mechanism that settles it:** in a cylindrical frame a camera **yaw is exactly a horizontal pixel translation** (`Δx = f·Δθ`). Translation equivariance in azimuth — which the distortion literature says non-pinhole projections *destroy* — is **restored** by our choice. In pinhole the same yaw is a homography and a point at 60° moves 4.0× further than one at 0°.

**The one thing pinhole does better** is keep world straight lines straight, and I measured what we pay for it. Globally the bow is real — worst case **35.54 px (13.88 % of frame H)** for an overhead gantry at 15 m, and **7.75 px** for the adjacent lane marking. **But a ViT never sees a global line.** Within one 16-px patch the worst departure from affine across every world line I tested is **0.021 px**, and analytically `d²u/dφ² = 0` *exactly*. The projection changes the *arrangement* of patches, not their content.

**What the field does** `PUBLISHED-PRIMARY (banked)`: a full-text sweep of UniAD, VAD, VADv2, SparseDrive, Hydra-MDP, NAVSIM, PARA-Drive and DiffusionDrive for `cylindric|equirect|fisheye|panoram|distortion|rectif|undistort` returns **0,0,0,0,0,0,0,1** hits (the one is "rectified *flow*"). **No canonical E2E stack discusses projection at all** — they inherit the pinhole dataset convention silently.

**The cleanest experiment that exists**, Trans4PASS Tab. 4c vs Tab. 2, same fold, same architecture: Trans4PASS-S native pinhole **50.20** → **native panoramic 53.31 = +6.2 % rel**, from 1,413 panoramic images against 70,496 pinhole. **Trained natively, panoramic wins.** (The much-quoted Cityscapes→DensePASS collapse, −38.6 pts, is a *transfer* penalty confounded by city/camera/weather — we never pay it, and I do not use it.)

**Resolution economy, the concrete win:** at equal 640 px width a pinhole rectification of 120° needs `f = 184.75`, so its centre would carry only **60.5 %** of our angular resolution. To *match* our centre resolution pinhole needs **1,059 px** = **1.68× sequence length, 2.81× attention FLOPs** — for the same forward field and a **4.0×-oversampled** periphery.

⚠️ **The honest counter-evidence, and its scope.** One primary points the other way: on KITTI-360, rectified beats distortion-aware-on-raw by **12–31 % relative mAP in the 20–30 m bin** across BEVFormer/PETR/BEVDet. Its three scope limits are stated in the paper itself and none applies to us cleanly — the "distortion-aware" arms are pinhole-*designed* architectures on **ImageNet-pretrained** backbones (rectification restores exactly what they assume; **we have no pretrained backbone**), and their input is 180°+ Kannala–Brandt with a **3–5×** angular-extent ratio versus our **1.0×**. But *long range is where the LONGITUDINAL family lives*, and this is the one place the literature says a non-pinhole projection actually cost somebody accuracy. **Flagged, not dismissed.**

⛔ **One claim chased and rejected:** a search summary asserted *"cylindrical rectification improves over rectilinear undistortion by nearly 4 mAP"*. Both candidate primaries were banked and full-text searched; **the number is in neither.** `PUBLISHED-SECONDARY, primary not located ⇒ INADMISSIBLE.` It was the most quotable-looking number in the pass and it dissolved on contact with the PDF.

**VERDICT: CONSISTENT, and ahead of the convention.** The PI's suspicion is not supported. The cost we do pay is not distortion — it is that **no benchmark shares our geometry, so every external comparison carries an unbounded term**, and no pretrained backbone matches our input.

### 3.6 Spatial pooling — the axis is mis-framed, and that is the finding

The register row calls 4 azimuth bins "2.1–7.8× too coarse for BEV localisation". Against a BEV grid, yes: our 15°/bin is **15.7× coarser** than a 0.5 m cell at 30 m. But that is not the comparison the field makes.

**TransFuser pools everything to a single 512-d vector before its waypoint GRU** — a 1×1 readout — and it is the architecture **DiffusionDrive (88.1 PDMS)** and **Hydra-MDP (91.0)** are built on. VAD's planner never touches its 200×200 BEV at all; it reads ego queries. **Readout width across the field is uncorrelated with score**, and our 4×8 is ~10× *finer* than TransFuser's planner readout.

⭐ **What the coarse-readout stacks share is not token count — it is that their bottleneck sits downstream of dense auxiliary supervision** (TransFuser: depth, semantics, BEV-semantics and detection heads). **That, not the grid, is the design question worth asking about our 4×8.**

⚠️ **And the literature here is structurally blind.** Every "coarse is free" result is measured on detection mAP or open-loop ADE/minADE — all **T0-equivalent**, all unable to see a decision error. The one safety metric available moved **~9× more** than the displacement metric on the same change (VAD-Base→Tiny: L2 +8.3 % but **collision +73 %**). **Nobody has published readout granularity vs collision/tactical quality with everything else fixed.**

⭐⭐ **BUT WE HAVE ALREADY MEASURED IT OURSELVES, AND I ALMOST SHIPPED THIS AXIS AS "UNKNOWN".** `GOALS_AND_CLAIMS.md` carries three settled entries that make our internal evidence *ahead of* the published field here:

- **`E-DEC-1` — SUPPORTED, 12/12 episodes.** The 16×40 → 4×4 pooling **discards roughly half** of what the encoder learned: `enc − z_op` on speed **+0.2117 (t 7.86)**, on `d_ego` **+0.1691 (t 12.38)**. The encoder beats the raw-pixel floor decisively; `z_op` *after* the readout does not. **The readout is a real, measured bottleneck.**
- **`E-DEC-3` — SUPPORTED**, frozen encoder, no retraining, every column PCA'd to k=128 so it is not a dimension race, constant control reads exactly 0.0000. The azimuth-resolution curve on **speed**: 4 bins (30°) **+0.1721** → 8 (15°) +0.2193 → 10 (12°) +0.2320 → **20 bins (6°) +0.3147** → 40 bins +0.2271 → full 16×40 grid +0.1938. ⇒ **the incumbent 4 bins is on the wrong side of a knee; the optimum is 10–20 bins, worth +0.143 R² on speed and +0.164 on `d_ego` — and *finer than 20 hurts*.**
- ⛔ **`E-DEC-3b` — REFUTED, and it is the one that stops this becoming a recommendation.** The E-DEC-3 optimum **reverses on 2 of 4 environment targets**: widening 4→10/20 bins **helps** `n_agents` (+0.0830, t 7.68, 24/24) but **hurts** `lead_gap_m` (−0.0277 / −0.0509, t −11.6 / −11.7, **0/24**) and `nearest_bearing` (−0.0195 / −0.0683). The mechanism is legible: a scalar depending on **one region straight ahead** is *diluted* by more bins, while a scene-wide **count** needs the spread. The register's own verdict: *"the E-DEC-3 optimum (20 bins) is an EGO-TARGET optimum and must NOT be shipped as 'the' readout geometry."*

⇒ **VERDICT REVISED: not "UNKNOWN" — MEASURED, and the answer is that there is no single right granularity.** The readout grid is a **per-target trade**, and the honest design response is not a finer grid but a readout that does not force one resolution on every target. **What the coarse-readout stacks in the field share is that their bottleneck sits downstream of dense auxiliary supervision** (TransFuser: depth, semantics, BEV-semantics, detection heads) — **that, not the grid, is the design question worth asking about our 4×8.**

### 3.7 Temporal context — the ratio is fine, the absolute is not, and the fix is free

**The curve the review needed**, BEVFormer Tab. 7 `PUBLISHED-PRIMARY (banked)`, nuScenes val, 0.5 s spacing:

| frames | span | NDS | mAVE | cumulative % of the achievable gain |
|---|---|---|---|---|
| 1 | 0.0 s | 0.448 | 0.802 | 0 % |
| 2 | 0.5 s | 0.490 | 0.467 | **61 %** |
| 3 | 1.0 s | 0.510 | 0.423 | **90 %** |
| 4 | 1.5 s | 0.517 | 0.394 | 100 % |
| 5 | 2.0 s | 0.517 | 0.387 | saturated |

**+0.084 NDS per second for the first 0.5 s, then +0.040, then +0.014, then flat.** Our **0.7 s** ⇒ `ESTIMATED` **~72–80 %** of the achievable temporal gain captured.

On the **ratio** we are not an outlier: TanitAD **0.117** sits between Vista (0.080) and Waymo (**0.1375**, verified from the primary). On **absolute span we are BEHIND**: 0.7 s against the field's 1.5 s camera convention — and after the horizon correction, 0.7 s of evidence is supporting a **6.0 s** prediction, an **8.6:1 extrapolation**.

⭐ **The band mapping is what makes this diagnostic rather than cosmetic.** Everything 0.7 s can observe (position → velocity → acceleration → jerk) is what the **operative band (0–2 s)** needs. Everything it *misses* — intent onset, braking-lead, gap-acceptance, which develop over ~1–2 s — is what the **tactical band (2–6 s)** needs. That is **4 of our 6 predicted seconds, and the tier the hierarchy thesis rests on.**

**Two counterweights I am keeping visible.** BEVDet4D reaches mAVE **0.337** with 2 frames, beating BEVFormer's 4-frame **0.394** — *more history is not a law; fusion quality matters at least as much.* And the copycat literature measures that **longer observation histories make the copycat/inertia problem worse** — directly relevant to a programme whose measured failure is an action echo. ⇒ **The stack-widening below must be run as a pre-registered ladder with a deliberate-regression arm, not adopted.**

### 3.8 Ego-motion compensation — the cheapest concrete BEHIND

`stack_frames` (`comma2k19.py:640-641`) is a pure `torch.cat` of time-shifted views. The three frames carry **independent camera pose** and nothing removes it. `bev_raster.py:86` even notes a builder *"should ego-compensate via egomotion"* — phrased as an unimplemented recommendation.

**BEVDet4D measures the value of exactly this fix: 51.0 → 51.7 NDS**, about what the 4th history frame is worth. `encoder.py:7-9` states the stack exists *"so ego-motion consequence is visible to the encoder"* — which is a coherent design intent (let the model infer ego-motion from pixel shift), and it is **the opposite of the field's choice**. It is defensible, it is untested, and it should be an arm.

### 3.9–3.10 Augmentation, and the one that matches our headline failure

**Augmentation exists in our code and never runs.** `photometric_jitter_window`, `lane_mirror_window`, `time_reverse_window` (`v6.py:3521-3627`) are fully written and tested; their sole consumer is `t2_contrastive_loss`, gated on `w_t2_contrast` whose default is **0.0** at `train_v6_staged.py:314` and in every stage preset. Even when enabled they augment a *contrastive view* feeding a separate projector — **the primary supervised batch is never jittered or flipped.** No arm exists in which the trajectory/action losses see an augmented frame.

⚠️ **A probe-methodology note worth recording**, because it nearly produced a wrong absence claim: the first scoped Grep over `data/` returned **zero** augmentation hits — the code lives in `models/v6.py`. A library-name probe (`torchvision|albumentations|…`) also returned zero, *correctly*, which would have **confirmed the wrong answer**. Only widening the term set surfaced it. This is `CLAUDE.md`'s "absence at one location is not absence" rule catching a live case.

⭐⭐ **Recovery augmentation is genuinely ABSENT (3 probes), and this is the finding that connects to everything else.** ChauffeurNet's measured result is that without synthesised off-nominal states with corrected labels, **the policy cannot recover from drift**. Our programme's headline failure is exactly that shape: **S-curve reproduction 97.9 % open-loop, ~0–5 % closed-loop.** The three near-misses in our tree (`metadrive_frontcam.py:154-359` action perturbation inside the simulator; `heldout_gate.py:548-557` a `(-8, 0, +8)°` warp at **eval** time) are none of them this.

⚠️ **Honest scoping:** on real video, ChauffeurNet-style perturbation needs a re-render we cannot do on PhysicalAI, so this is **not a cheap fix** — it is the strongest argument for the AlpaSim/NuRec path, and it should be stated as such rather than filed as a to-do.

### 3.11 Anti-aliasing — real, and smaller than I first reported

`grid_sample(mode="bilinear")` has a fixed 2-source-pixel support and **cannot antialias** (the op has no `antialias` parameter). It runs at a **3.0× horizontal / 2.84× vertical** decimation from a 1920×1080 native frame (`FThetaIntrinsics` defaults `width=1920, height=1080`, `calib.py:349-350`, agreeing with the "1080p clip" comment at `physicalai.py:524` — two probes).

Control-verified (DC reads exactly 0.00000; a sub-Nyquist sine agrees to 0.1 % between arms): a structure at **1.5× the output Nyquist** passes through at its **full 0.45 amplitude** when the correct output is **0.15** — a **3.0× spurious response** whose phase depends on the sampling offset.

⚠️ **I over-claimed the consequence and am retracting it inside this document.** My first frame-to-frame panel used a white-noise-textured synthetic frame and reported **2.98×** excess. That is not a camera frame. Repeated across `1/f^α` spectra with a lens/OLPF roll-off:

| frame spectrum | % power above output Nyquist | excess Δframe |
|---|---|---|
| **CONTROL** — band-limited below Nyquist | 0.00 % | **1.02×** ✅ (must be 1.00) |
| natural, 1/f² + lens MTF 0.20 | 0.10 % | **1.11×** |
| natural, 1/f² + lens MTF 0.35 | 1.85 % | **1.31×** |
| white-ish, 1/f⁰·⁸ (my first, unrealistic, arm) | 76.75 % | 4.16× |

**The defensible figure is 1.11–1.31×**, i.e. 10–30 % of the per-frame delta the world model is trained to predict is sampling phase rather than scene. The *mechanism* matters — aliased content is unpredictable, so its loss-optimal prediction is its conditional **mean**, which is a route from the input to a regress-to-the-mean failure — but at 1.1–1.3× it is a **contributor, not the cause**.

⛔ **A third panel (excess vs thin-feature width) is DISCARDED, not reported**: it returned values below 1.0× and was non-monotonic, both impossible for an aliasing measure, and the cause was commensurability between my feature spacing and the decimation ratio. **The per-structure claim is UNMEASURED and the thin-structure question stays open.**

➕ **AND THE SAME OPERATOR RUNS IN TIME, WHICH I HAD NOT NOTICED.** `physicalai.py:711-719` builds a synthetic uniform `t_query` and resolves it with `np.searchsorted` — **nearest-index point sampling with no temporal averaging.** The source is ~**30.2 fps** (`ESTIMATED`: `physicalai.py:524`'s *"604-frame"* clip ÷ 20 s; ⚠️ the adapter declares **no** source-fps constant, three probes, so this is inferred from one comment). ⇒ **a ~3× TEMPORAL decimation with no filter, exactly mirroring the ~3× spatial one.** Two of every three source frames are discarded rather than averaged, so fast motion aliases in time the way thin structure aliases in space. I have **not** measured the magnitude of this one and am not claiming it — but the mechanism is identical, it compounds with §3.11, and it is invisible to every probe the programme currently runs. **Work item, not a finding.**

### 3.12 The rig-correlated invalid region — CONFIRMED LIVE, and the headline of this review

`calib.py:1244-1246`: **8.897 %** of the 256×640 frame is unobserved on rig B, **0.0017 %** on rig A, over n = 3,000 — and rig B is **2,188 / 3,000 = 72.9 %** of the corpus. `cylindrical_rectify` makes those pixels **honest black zeros**, which removes the *fabrication* but not the *asymmetry*: a rig-correlated constant region is a **free rig label**.

Mechanism (certain): rig B has `cy≈755` on a 1080-row sensor, so only **324** native px lie below the principal point against 755 above; the ray fan needs 127.6 output px below centre, so **the mask is a bottom band, not a symmetric ring**. In token terms `ESTIMATED` **56–100 of 640 tokens** (1.4–2.5 of 16 token rows) carry constant black on 72.9 % of clips. *(The exact per-clip shape needs the f-theta poly, which is not on this box; the 40-row depth bound comes from the repo's own measurement over all 3,000 clips.)*

**Why a ViT will not simply ignore it.** Darcet et al. `PUBLISHED-PRIMARY (banked)`: ViTs develop high-norm artifact tokens *"in low-informative background areas, that are repurposed for internal computations"*, in *"patches similar to their neighbors"*, where the model *"recycle[s] the corresponding tokens to aggregate global image information **while discarding spatial information**."* A constant padded region is the maximal case of "similar to its neighbours". Their prevalence is 2.37 % of tokens; **ours would be ~9 % by construction.** Removing the effect is worth **+131.6 %** on LOST (VOC07 11.7 → 27.1) — the task in that paper that depends on the *spatial* structure of the feature map, which is precisely what a latent world model reads.

⚠️ **Fairness caveat from the same paper:** artifacts *"only appear after a sufficiently long training of a sufficiently big transformer."* Ours is from-scratch and small. **UNKNOWN whether the mechanism is active — and it is cheap to test** (histogram output-token L2 norms; check whether high-norm mass sits inside the rig-B mask).

⭐⭐ **AND IT IS NOT A RISK — IT IS LIVE. CONFIRMED THIS SESSION, THREE INDEPENDENT ROUTES.**

`MEASURED-SOURCE`. **Every v6 and v7 arm trains on the full 256×640 frame with `v2_subframe: null`** — the rig-clean sub-frame is *not* passed anywhere in the current line:

| arm | trainer | frame | `--v2-subframe` | source |
|---|---|---|---|---|
| `flagship-v5f-w120-30k` | `train_flagship_v4.py` | **176×624** | ✅ **passed** | `runs.d/flagship-v5f-w120-30k.env:10` (verbatim from `/proc/19412/cmdline`) |
| **v6F "config E", 336,542,025 params** | `train_v6_staged.py` | 256×640 | ❌ **`null`** | `v6F-SW-30k.config.json:300`; independently `RESTART_v6F_SW.sh:30` (argv from `/proc/25477`) |
| `rdw8p30k`, `champ30k`, and today's T1 arms | `train_v6_staged.py` | 256×640 | ❌ **`null`** | `t1_nonparity40_rdw8p30k.json` (`model_frame` adopted *from the checkpoint*); `champ30k_config.json` |

**It was live for exactly one arm family (v5f) and was dropped at the v5→v6 transition** — because in `train_v6_staged.py` the flag moves the **data** and not the **model**, so passing it against a 256×640 encoder now dies at the first forward (`tests/test_v6_st_launch_fixes.py:210-224`). The legal exit is `--frame-h 176 --frame-w 624`, i.e. **a different model and a retrain**.

And the manifest states both halves of the trade in one place — `parity_manifest.json`, corpus `…-w120-256x640cyl`: `rig_observability` rig A n=642 mean **0.9999861572**, rig **B n=1758 mean 0.9110825422**, `fully_observed_by_all: false`; and `subframe_observability` **`"176x624": {A: 1.0, B: 1.0}`**. **73.25 % of the train cache (1,758 / 2,400) is rig B.** The fix exists, is measured, is one geometry change — **and is not being made.**

⚠️⚠️ **AND I HAVE TO RETRACT A CLAIM I MADE EARLIER IN THIS SAME REVIEW.** I wrote that *"every recorded arm runs the plain `ViTEncoder`: no registers, no RoPE"*, on three probes — Grep, PowerShell `Select-String`, and a direct read — **all three against `MODEL_REGISTRY.md` and the steering docs.** They agreed, and they were all looking in the wrong place: **`MODEL_REGISTRY.md` has no row for the v6F arm at all**, and the arm's own `config.json` shows **`--vit5-encoder` with 4 register tokens** (`v6F-SW-30k.config.json:254-255`). ⇒ **Corrected:**

- **v6F config E (336.5 M) DOES use the ViT-5 encoder with 4 registers** — so the Darcet mitigation *is* in place on the scaled arm, and the FishRoPE "angular PE for free" argument *does* apply to it.
- **The v7-tiny ladder does NOT** (`champ30k`, `rdw8p30k`: plain `ViTEncoder`) — and that is the ladder every recent decision has been made on.

*(Class: C133 — three probes, one vantage. The lesson is not "probe more" but "probe a different KIND of source": all three of mine read documentation, and the answer was in a run artifact.)*

### 3.13 Frame drops — a silent one

`physicalai.py:711-719` resamples onto a **synthetic uniform grid** (`np.linspace`) and resolves it with `searchsorted`. If source frames are dropped, this **silently emits a duplicated frame** at nominal 0.1 s spacing, while pose and action are `np.interp`-smoothed **across** the gap. The gap is therefore invisible on both sides. The single `np.diff` in the file (`:626`) runs on the *synthetic* grid, not on real timestamps. **No spacing validation exists** (3 probes). Cost to fix: an assertion on real `t_frames` spacing — very low.


### 3.18 Cycle time — what we can and cannot say

`MEASURED-REGISTRY`, and it is the *only* hardware number the repo carries for the current line: arm `rdw8p30k` ran **30,000 steps at batch 8 in 29,758 s on a Jetson Thor** (`MODEL_REGISTRY.md` §13.0, `summary.json done: true`) = **0.992 s/step ≈ 8.1 windows/s**.

⛔ **That is a TRAINING figure for a 19.3 M-param arm and must never be quoted as an inference latency.** Forward-only inference is a small fraction of a training step (no backward, no optimiser), and the scaled arm is ~17× larger — the two errors point in opposite directions and neither is bounded by this number.

⚠️ **I first wrote "no inference latency exists anywhere" — that is WRONG and is retracted.** Substantial end-to-end measurements exist for the *earlier* architectures. `MEASURED-SOURCE`, `…/2026-08-03-thor-batch9-engine/thor_d6_tick_intent_K20.json`, **Jetson Thor, real weights, full decision tick** (encoder + strategic + tactical + 9-candidate fan + `step_readout` + SE(2) + scoring), 60 windows / 12 episodes, 100 ms budget:

| configuration | p50 | p95 | p99 |
|---|---:|---:|---:|
| A fp32 eager, serialised fan | 764.06 ms | 768.44 | 769.84 |
| B bf16 engine, caller not fixed | 371.97 ms | 380.09 | 382.46 |
| D bf16 eager batched, no engine | 204.41 ms | 205.71 | 206.30 |
| ⭐ **C bf16 + dynamic 1–9 engine, BATCHED fan** | **60.30 ms** | **63.08** | **65.28** |

**6.169× from B to C, and inside the 100 ms budget** — so **16.6 Hz** is demonstrated on the deployment target for that architecture. Per-stage on Thor at the deployed 176×624 (`thor_c2_real_weights.json`): encoder fp32 196.86 ms → **bf16 30.23 ms** (6.512×); TRT-fp16 predictor engine 1.177 ms at batch 1, 1.294 ms at batch 9 (**0.144 ms/candidate**). On the A40 proxy the optimised composed tick is **18.75 ms (53.3 Hz)**; on the dev-box RTX 4060, 15.07 ms.

⛔ **BUT THE FINDING SURVIVES IN A SHARPER FORM: not one of those numbers is a v6 or v7 arm.** Every banked latency is **v1 at 256×256** or **v5f at 176×624**, and all are dated ≤ 2026-08-03 (three probes: Grep over the Architecture & Inference tree, PowerShell `Select-String` over both v7 research directories, and a registry sweep — the v7 streams return nothing but two false positives). ⇒ **The 336.5 M config-E arm and the entire v7 line have zero measured inference cost, at any geometry, on any hardware.** `hz_op = 10.0` (`v6.py:3706`, asserted consistent with `dt` at `:4198-4199`) is for those arms a **design contract, not a measurement**.

➕ **And one blocker I cited from a code comment is now historical — corrected.** `readout.py:90-95` still narrates `AdaptiveAvgPool2d` as blocking the encoder ONNX export at 176×624 and therefore blocking backlog item O2. The same file implements the fix at `:97-110`, and `thor_c5_encoder_export.json` **measured it working**: export OK at opset 17, `has_adaptive_pool: false`, ORT-vs-eager rel-err **1.77e-05**, verdict *"✅ O2-pre CLOSED"*. **Better still, the v7 geometry never meets the blocker at all**: at 16×40 tokens both the default 4×4 and `rdw8`'s 4×8 satisfy `exact_pool` (`readout.py:87`), so both take the plain `AvgPool2d` path that every opset exports. The blocker was specific to 11×39 → 4×4 — **the geometry that is no longer in use.** *(The TensorRT **encoder** engine, O2 proper, is still `not started` per `PRODUCTION_READINESS.md:149`.)*

⇒ **VERDICT: `CONSISTENT` as an engineering target — 60.3 ms end-to-end on Thor is a real, banked, inside-budget number — but `UNKNOWN` for everything we are currently building.** Closing it is cheap and there is no longer an export blocker in the way.

### 3.19 Data scale — and a second correction to the commission's numbers

`MEASURED-SOURCE`, from `stack/tanitad/data/parity_manifest.json`, with the arithmetic shown. An epcache episode is `[199, 9, 256, 640]`, i.e. 199 *stacked* steps; since `T_out = T_raw − (n_stack−1)` that is **201 raw frames at 10 Hz = 20.1 s**:

| corpus | clips / episodes | hours |
|---|---|---|
| full discovered export | 3,000 | **16.75 h** |
| **parity TRAIN** (`physicalai-train-e438721ae894`, 24 decode failures) | **2,376** | **13.27 h** |
| parity VAL (`physicalai-val-0c5f7dac3b11`) | 600 | 3.35 h |
| v2 `256x640cyl` train sibling (0 skips, complete build) | 2,400 | 13.40 h |
| the larger `physicalai-v2bal-…` corpus (`v2_dataset.py:6-7`) | ~9,000 | **~50 h** |

⚠️ **A correction I nearly published, and then withdrew.** I first flagged the commission's "26 h (B1)" as unreproducible, because no corpus in `parity_manifest.json` gives 26 h. **That was a one-vantage absence claim and it was wrong.** `GOALS_AND_CLAIMS.md` `D-CORPUS-B1` records the PI directive of 2026-08-29: **B1 is the Alpamayo-labelled set — 4,719 clips**, a *different* corpus from the parity one. **4,719 × 20.1 s = 26.35 h ✓** — the commission's figure is exact. *(Class: the C133 family — absence found at one location, asserted as absence. The register was the second probe I should have run first.)*

**The corrected data-scale picture, and it is two domains, not one:**

| corpus | clips | hours | role |
|---|---|---|---|
| parity train `physicalai-train-e438721ae894` | 2,376 | **13.27 h** | the sacred corpus — every banked arm to date |
| **B1** (Alpamayo-labelled, `D-CORPUS-B1`) | **4,719** → **4,713** built | **26.35 h** | the scaled-v7 training corpus, **building now** |

⛔ **`D-CORPUS-B1` declares these a COMPARABILITY BOUNDARY** — arms trained on B1 are a **new domain** and may never be same-data-compared to parity arms. Two B1 facts intersect this review directly:
- **rig B is the majority again** (2,723 B / 1,996 A), and the directive states verbatim that *"every epcache build must crop around per-clip cy"* — so §3.12's rig question **travels to B1** rather than being retired by it.
- 6 of the 4,719 clips were caught inside the **deployed val40** and are excluded via `--exclude-parity-overlap` (4,713 enter the cache).

---

## 4. The changes worth making, priced

The PI asked for a ranking by (expected effect on our results) × (cost to change), and for **one** to change first.

### 4.0 ⏱⭐⭐ THE ONE TO CHANGE FIRST — and it is a decision on a clock, not a code change

**ESCALATION, and it is the headline of this report.** Three facts landed together and they only matter jointly:

1. **The rig shortcut is live, not hypothetical.** `v2_subframe: null` on the 336.5 M config E, on `rdw8p30k`, on `champ30k`, and on every arm evaluated today — established from the run's own `config.json`, from `/proc` argv of the live trainer, and from the geometry each checkpoint carries into eval. **73.25 %** of the parity train cache is rig B, at **8.89 %** masked.
2. **The fix is measured and it is a GEOMETRY choice.** `parity_manifest.json` records `subframe_observability "176x624": {A: 1.0, B: 1.0}` — zero masking, both rigs, all clips. But in `train_v6_staged.py` the sub-frame flag moves the *data* and not the *model*, so applying it to an existing arm means `--frame-h 176 --frame-w 624`: **a different model and a full retrain.** At *build* time it costs nothing.
3. ⏱ **B1 is validated and NOT YET BUILT.** Manifest sha256 `5feda062a72a32ad`, 4,719 clips, all four joins clean, prerequisites closed on Thor 2026-08-30 ~07:30, ~3.4 h at 6 shards — **waiting on the GPU.** Rig B is the majority there too (**2,723 A-vs-B split 2,723/1,996**), and `D-CORPUS-B1` already states that *"every epcache build must crop around per-clip cy."*

⇒ **The geometry of the corpus the next four consumers (v7f, refcv3, refav1, refd) will train on is being fixed in the next few hours, and it is the last cheap moment to decide whether they inherit a rig-identifying black region.** After the build it is a 3.4–4.6 h rebuild plus a parity break plus a retrain per consumer.

**What I recommend, and what I explicitly do not.** I am **not** recommending "build B1 at 176×624" — §4.1 shows that crop costs real near-field vertical field (an overhead traffic light leaves frame at 12.2 m instead of 8.4 m), and that is a genuine trade the PI should make, not me. What I recommend is that **the decision be made deliberately, now, with the three options priced**:

| option | rig mask | cost at build time | cost after the build |
|---|---|---|---|
| **build as-is, 256×640** | rig-correlated, 8.89 % on 57.7 % of B1 clips | zero | the shortcut is baked into four consumer lines |
| **build at 176×624** | **0.000000 %, both rigs** | zero | loses 13.3° of elevation and 3° of azimuth |
| ⭐ **build 256×640 AND bank the per-clip validity masks as a sidecar** | unchanged | ~zero | **keeps both options open** — the mask can then be fed to the model explicitly, or used to slice later, without a rebuild |

**The third option is the one I would take if it were mine**, because it is the only one that does not foreclose the other two, and because `observed_report` (`calib.py:1109`) already computes exactly that mask *"without decoding anything"*. **But it has to be decided before the build starts.**

**And before spending a retrain on option 2, run the free test** in §4.3 — the token-norm histogram — which decides whether the mechanism that makes the mask harmful is even active at our scale.

### 4.1 A correction to a recommendation that arrived from the literature stream

One stream proposed *"crop elevation 256 → 176 rows, +45 % vertical px/deg for free."* **A crop cannot add resolution.** `MEASURED-COMPUTE`, `raw/geom_options.py`:

| option | HFOV | VFOV | px/deg x | px/deg y | tokens | FLOPs | what it is |
|---|---|---|---|---|---|---|---|
| current 256×640 @ f=305.577 | 120.00° | 45.46° | 5.333 | 5.632 | 640 | 1.00× | — |
| **D1** crop 176×624 (rig-clean) | 117.00° | 32.13° | **5.333** | **5.478** | 429 | **0.67×** | a **flag** (`--v2-subframe`) |
| **D2** rebuild 256×640 @ f=444.75 | 82.45° | 32.11° | **7.762** | **7.972** | 640 | 1.00× | a **cache rebuild** |

D1 buys **0 % resolution**; it buys 33 % fewer tokens and the rig-clean guarantee. The +41 % vertical resolution is D2, and D2 costs a rebuild **and 37.6° of horizontal field**. Resolution and field trade through **one** `f_ref`; there is no free axis.

**And the crop is not free either.** With a 1.5 m camera and a 5 m overhead light (`ESTIMATED` heights, stated): cropping to 176 rows makes an overhead traffic light **leave frame at 12.2 m instead of 8.4 m**, and pushes the nearest visible road from **3.6 m to 5.2 m** — it bites exactly where a stop-line decision is made. **This must not be sold as a free lunch.**

Where the vertical deficit is real: we spend **45.46°** of elevation where BEVDet spends ~25°, which is the *entire* explanation of our **1.82×** vertical-resolution deficit. Horizontally we are level with the field.

### 4.2 ⭐ THE EXPERIMENT TO RUN ALONGSIDE IT — widen the encoder's frame stack

**What:** `(t−0.2, t−0.1, t)` → `(t−0.6, t−0.3, t)`. One stride parameter in `stack_frames` (`comma2k19.py:640`): `vid_u8[i : T-(n-1)+i]` becomes `vid_u8[i*s : T-(n-1)*s + i*s]`.

**Cost:** ~5 lines. **0 FLOPs, 0 parameters, no parity break, no checkpoint-shape change.** The only data cost is the longer warm-up: `(n_stack−1)·s` = 6 rows of ~199 ≈ **3 % of windows**.

⚠️ **The "no rebuild" half is path-dependent and I have to say which path.** The **v2 cache** (`*.v2ep.pt`, what the v7 trainer reads via `--v2-cache`) stores **individual encoded frames** and stacks them at *load* time in `_decode_stacked` (`v2_dataset.py:143-149`), with full `poses`/`actions` in the payload and the `[n_stack-1:]` alignment applied at load (`:375-377`) — **so on that path the stride genuinely is a loader change.** The older `_epcache` path stores frames **pre-stacked** as `[199, 9, H, W]`; there, changing the stack is a rebuild. `GOALS_AND_CLAIMS.md` `E-DEC-6` records exactly this: the `nstack1` arm *"crashed on Thor because the cache has 9 frame channels baked in, so `n_stack=1` is a cache REBUILD not a flag."* **Quote the free cost only for the v2 path.**

⏱ **AND THERE IS A TIMING WINDOW.** `D-CORPUS-B1` records that the **B1 build's prerequisites closed on Thor 2026-08-30 ~07:30** and it runs ~3.4 h at 6 shards. If the stride is to be a *build-time* choice for the new corpus rather than a loader flag, **this is the moment to decide it.** Escalating rather than assuming.

**Expected effect:** total distinct span **0.7 s → 1.1 s**, moving us from `ESTIMATED` ~72–80 % to ~90–95 % of the temporal benefit BEVFormer measured (Tab. 7: NDS 0.448 → 0.490 → 0.510 → 0.517, flat by 1.5 s).

**Why this one, over the other candidates:**
1. It is the only item where the **effect is supported by a published curve** and the **cost is provably zero**. Every other candidate trades something real.
2. ⭐ **Our own code already argues for it.** `v2_dataset.py:133` (H-RANK-8) records that *"with a 3-frame stack consecutive latents share 2/3 of their input"* — at 0.1 s spacing inside a 10 Hz window the stack is **largely redundant with the window**. Widening de-duplicates it and buys span for nothing. This is exactly GAIA-1's stated trade — *"25 Hz to 6.25 Hz … allows the world model to reason over longer periods without leading to intractable sequence lengths"* — applied to our own measured redundancy. **At a fixed sequence budget the field spends it on span; we currently spend ours on density.**
3. It targets the **tactical band (2–6 s)**, which is 4 of our 6 predicted seconds and the tier the hierarchy thesis rests on.

**⛔⛔ THE STRONGEST COUNTER-ARGUMENT IS OUR OWN, IT IS MEASURED, AND IT NEARLY KILLS THIS RECOMMENDATION.**

`GOALS_AND_CLAIMS.md` **`E-DEC-6` — REFUTED and DROPPED.** Tested with **zero training**: feed the same frozen `rdw8` encoder a **collapsed stack `[f_t, f_t, f_t]`** — identical tensor shape and weights, every inter-frame difference exactly zero. If the stack's temporal content were load-bearing, ego decodability would collapse. **It does not:** speed **+0.2830 → +0.3115 (t +2.37, 10/12 — it slightly *improves*)**; `d_ego` +0.3601 → +0.3558 (not separable). The register's conclusion: *"ego is read from **SINGLE-FRAME APPEARANCE** (motion blur, and scene type correlating with speed), not from inter-frame motion."*

⇒ **My proposed "deliberate regression `(t,t,t)`" arm has effectively already been run, and it did not regress.** The current 0.1 s-spaced stack is carrying approximately nothing between its frames.

**Two readings survive that fact, and they predict opposite outcomes:**
- **(a) the stack mechanism is simply unused** — widening the stride buys nothing either, and this recommendation is dead;
- **(b) the baseline is too short to produce usable inter-frame signal at all** — 0.1 s of ego motion is ~4.7 cm of translation, largely below the aliasing floor measured in §3.11 — and 0.3 s baselines would put real parallax into the same tensor for free.

**I cannot separate (a) from (b) from banked evidence, and I am not going to pretend otherwise.** What makes the arm still worth running is that it costs essentially nothing and it *discriminates between them* — and that E-DEC-6's probe measured **ego decodability from a frozen encoder under a linear map**, which is a different question from **tactical-band prediction**, the thing BEVFormer's curve is about and the thing this change targets. ⚠️ Per the standing rule, a negative from a linear probe on ego targets is **not** a negative about temporal information in general.

**⛔ It must therefore be run as a pre-registered ladder with both outcomes committed in advance, not adopted.** The copycat literature independently measures that *longer* histories can make the action-echo problem **worse** — and an action echo is our headline failure — so the downside is real in both directions. Arms: **control** `(t−0.2,t−0.1,t)` (must reproduce the banked number) · **C** `(t−0.6,t−0.3,t)` · **D** asymmetric `(t−0.5,t−0.1,t)` (keeps a tight acceleration baseline *and* an intent baseline) · **collapsed** `(t,t,t)` (must reproduce E-DEC-6's +0.3115, which makes it a **harness control**, not just a regression arm). Read out on the **four families split by band (0–2 s vs 2–6 s), never pooled** — that split *is* the hypothesis test, and pooling averages it away.

### 4.3 The free test that gates §4.0 — and the register tokens on the tiny ladder

Both are flags. Both address §3.12 from opposite ends: `--v2-subframe 176x624` **removes** the rig-correlated black region (0.000000 % masked on both rigs, over all 3,000 clips); `--vit5-encoder` gives the ViT **4 register tokens** so that any low-information region that remains has somewhere to put global computation other than spatial patch tokens.

**Cost:** each changes the model geometry or the parameter count, so each is a **declared arm with a retrain**, not a hot-swap. `--v2-subframe` also cuts tokens 33 % (0.67× FLOPs), so that retrain is *cheaper* than the baseline. Note the §4.1 caveat: the crop costs near-field overhead structure.

**Before spending that, run the free diagnostic the literature implies:** histogram the output-token L2 norms of an existing checkpoint and check whether high-norm mass sits inside the rig-B mask. Darcet et al. state the artifact *"only appear[s] after a sufficiently long training of a sufficiently big transformer"* — ours is from-scratch and small, so the mechanism may simply not be active. **That test is hours, not GPU-days, and it should gate the retrain.**

### 4.4 The ranking

| rank | change | expected effect | cost | why here |
|---|---|---|---|---|
| ⏱ **0** | **decide B1's build geometry / bank the validity masks (§4.0)** | **high — it fixes what four consumer lines inherit** | **~zero NOW; a rebuild + 4 retrains later** | the only item on a clock, and the only one that forecloses options if missed |
| **1** | **the frame-stack ladder (§4.2)** | **discriminating** — published curve says high; our own `E-DEC-6` says possibly zero | **~zero** (v2 path) | the cheapest experiment that separates two live explanations; valuable whichever way it reads |
| **2** | token-norm diagnostic for the rig shortcut | diagnostic | ~zero (hours) | gates #3, and may retire it |
| **3** | rig-clean geometry on the v7-tiny ladder + `--vit5-encoder` **on the ladder** | medium–high **if** #2 fires | one retrain, **0.67× FLOPs** | ⚠️ the scaled v6F arm **already has** the ViT-5 registers; it is the **tiny ladder** — where every recent decision is made — that has neither registers nor RoPE |
| **3b** | **measure inference latency for any v6/v7 arm** | closes a total blank | hours; the ONNX blocker is historical and the v7 geometry tiles exactly | the whole current line has zero measured forward cost on the deployment target |
| **4** | frame-drop assertion on real timestamps | unknown, possibly zero | very low | a silent corruption with no detector is worth a guard regardless |
| **5** | ego-motion compensation of the stack | medium — BEVDet4D 51.0 → 51.7 NDS | moderate; needs per-frame pose in the loader | our design intent is deliberately the *opposite*; make it an arm, not a fix |
| **6** | enable the existing augmentation on the supervised path | unknown | low — the code is written and tested | but see the copycat caveat |
| **7** | anti-alias prefilter before `grid_sample` | **1.11–1.31×** on the wrong signal | code change + **cache rebuild that breaks parity** | real but second-order; do it *with* the next rebuild, never *for* it |
| **8** | ⛔ **readout grid change — DO NOT SHIP** | **MEASURED and it is a per-target trade** | — | `E-DEC-3` finds a 10–20-bin optimum on ego targets (+0.143 R² on speed) but `E-DEC-3b` finds it **reverses** on `lead_gap_m` and `nearest_bearing` (0/24). The register's own verdict forbids shipping it as "the" geometry. The live question is **dense auxiliary supervision upstream of the bottleneck**, not the grid |
| — | resolution increase | fixes the far-field deficit | 5.13× FLOPs and/or a rebuild | the field is only ~1.8× better than us; this is not where our gap is |
| — | recovery / perturbation augmentation | **high** (ChauffeurNet) | **cannot be done on PhysicalAI** — needs a re-render | the strongest argument for the AlpaSim / NuRec path |

---

## 5. What I could not establish, and what would close it

| # | open item | status | how to close |
|---|---|---|---|
| ~~1~~ | whether the live/scaled arms pass `--v2-subframe` | ✅ **CLOSED — they do not.** Three independent sources (§3.12) | done |
| 2 | whether the **Darcet artifact mechanism is active at our scale** | **OPEN — and it is the gate on §4.0 option 2** | the token-norm histogram in §4.3; hours, not GPU-days |
| 3 | the **exact rig-B mask shape** in token terms | **OPEN** — my 56–100/640 is `ESTIMATED`; needs the per-clip f-theta poly | one pod-side `observed_report` sweep (it needs no decode) |
| 4 | **per-structure aliasing cost** | **OPEN** — my own instrument was invalid and I discarded it (§3.11) | a corrected panel with a non-commensurate feature layout |
| 5 | **readout granularity vs tactical quality** | ✅ **partly closed by our own `E-DEC-3`/`E-DEC-3b`** (§3.6) — measured, and it is a per-target trade | the remaining gap is *published* work; ours is ahead |
| 6 | eval-time preprocessing parity with training | **OPEN** — out of scope this pass | audit `taniteval/`; a mismatch would silently bias every open-loop number |
| 7 | native mp4 **resolution and frame rate** per clip | **OPEN** — `1920×1080` is a dataclass default + a comment; ~30.2 fps is inferred from one comment and the adapter declares no fps constant | one `ffprobe` on a real clip — closes both, and both feed §3.11 |
| 8 | **inference latency for any v6/v7 arm** | ✅ **CLOSED as a question, OPEN as work** — none exists (§3.18), and the ONNX blocker that used to prevent it is historical | one forward benchmark; the v7 geometry tiles exactly so nothing blocks it |
| 9 | ⚠️ **`MODEL_REGISTRY.md` has no row for the 336.5 M config-E arm** | **OPEN — escalated** | the param count (336,542,025), geometry and ViT-5 flag live only in run artifacts and `RETRACTION_LOG.md`. Two superseded figures are still in circulation (336,559,305 and 336,575,049). Per the source-of-truth rule this arm is currently **unquotable from the registry** |

⚠️ **One banking defect found and deliberately NOT fixed.** Library key **`unece-r157`** is **not** UN R157/ALKS — page 1 identifies it as ECE/TRANS/WP.29/2026/139, a *different* instrument (June 2026). The PDF is a legitimate primary and is arguably the better OEDR source; the **identity** is wrong, and `--verify` passes because the *bytes* match, which hashing cannot catch. I did not re-key it because two streams were still writing to the Library concurrently and `kb_add` has a known concurrent-write race. **ESCALATED: this needs a re-key, and any existing citation of "R157" resolving to this key is a misattribution.**

---

## 6. Honest summary of what this review does and does not establish

**What it establishes.** Our input pipeline is **not** the crippling weakness the commission hypothesised, and the axis most suspected — projection — is the one we are most clearly *right* about, on numbers rather than argument. Resolution, patch size, tokenisation, FOV, codec, normalisation and label alignment are all consistent with, or ahead of, the field.

**What it also establishes.** There is a cluster of real, unglamorous weaknesses that no architecture review would have surfaced: **no augmentation on the supervised path, no recovery augmentation, no ego-motion compensation of stacked frames, no frame-drop guard, no spatial anti-aliasing, an unfiltered ~3× temporal decimation, a rig-correlated shortcut confirmed live on every current arm, and — on the tiny ladder every recent decision is made on — an encoder without the registers the scaled arm already has.** Individually each is small. Collectively they are the difference between a pipeline that has been *designed* and one that has been *assembled*.

**What it does not establish.** No causal link from any of these to the programme's measured failures. Two candidates (aliasing, the rig shortcut) have plausible mechanisms that would push a world model toward the marginal; I quantified the first at **1.11–1.31×** and the second is **UNKNOWN pending a free test**. Neither is demonstrated. **The action I recommend first (§4.0) is recommended because it is on a clock and currently free, not because it is proven to fix anything.**

**What I got wrong in this session and corrected before shipping** — five things, which is the honest measure of how much of this was not knowable from one vantage:
1. An aliasing headline of **"≈3×"** that was an artefact of a white-noise test image — retracted to **1.11–1.31×** (§3.11).
2. A per-structure aliasing panel whose controls read impossible values — **discarded**, not reported (§3.11).
3. **"No registry arm uses the ViT-5 encoder"** — wrong; three probes all read *documentation*, and the answer was in a run artifact. The scaled arm **does** have registers (§3.12).
4. **"The 26 h B1 figure does not reproduce"** — wrong; B1 is a different corpus (4,719 clips × 20 s = 26.35 h) and the commission's number was exact (§3.19).
5. **"No inference latency exists"** — wrong; **60.30 ms p50 end-to-end on Thor** is banked. The correct, narrower finding is that none exists for any **v6/v7** arm (§3.18).

And two more from the streams: a **"+45 % vertical resolution for free"** that confused a crop with a rebuild (§4.1), and a **"nearly 4 mAP"** projection number that was in neither candidate primary and was refused as inadmissible (§3.4).

⚠️ **The pattern across all seven is one thing: a claim true in one scope, asserted in another.** That is the `df` / `free` / `step_s` family the programme already names, and the only defences that actually worked here were a control that had to read a known value, and re-deriving the arithmetic from a second *kind* of source.

**Scope limits of this review itself, stated rather than left implicit.**
- **Axes 6 and 7 carry no external literature comparison.** The stream commissioned for published latency and data-scale figures was still running when this shipped, so §3.18 and §3.19 rest on **our own artifacts** (registry, run configs, manifests) plus one comparative note. The *field* columns for "what latency do published stacks achieve" and "what data scale does the world-model family use" are **UNFILLED** — treat those two verdicts as provisional.
- **No GPU work was done.** Every number here is read from source, computed from our own formulas, or read from a banked artifact. The RTX 4060 was checked free and not used; Thor was not touched (it was holding the B1 build).
- **The literature is cited from 237 Library-banked primaries** (`kb_add.py --verify`: 237 entries, 0 orphans, 0 problems, run at ship time). Where a stream could only reach a secondary, it is labelled `PUBLISHED-SECONDARY` and is **not** used to decide anything.
