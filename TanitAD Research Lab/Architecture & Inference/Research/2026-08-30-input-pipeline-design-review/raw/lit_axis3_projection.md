# Axis 3 — PROJECTION AND DISTORTION: literature findings

Stream: literature research for the PI-commissioned TanitAD input-pipeline design review.
Date 2026-08-30. All papers cited below are **banked as primaries** in
`TanitAD Research Lab/Library/` (`--verify`: **199 entries, 0 orphans, 0 problems**).

## Our setup (INHERITED from the brief — not re-derived here)

Cylindrical, 256(H) x 640(W), `f_ref` 305.5774907364391, column linear in azimuth.
From-scratch ViT, patch 16, learned absolute PE (+ 2D axial RoPE in ViT-5). Source is
PhysicalAI-AV `camera_front_wide_120fov` (f-theta). Two rigs; rig B ~9.05 % of the output
frame invalid/padded.

## Papers banked this pass (10)

| key | what it buys us |
|---|---|
| `1912.09390` Tangent Images (CVPR'20) | the projection-transfer penalty AND the angular-resolution-mismatch curve |
| `2203.01452` Trans4PASS (CVPR'22) | **the clean same-scene pinhole-vs-panoramic experiment** + DPE ablation |
| `2207.11860` Trans4PASS+ | extended distortion-aware ViT |
| `1905.01489` WoodScape (ICCV'19) | the argument against naive rectification |
| `2012.02124` Rashed WACV'21 | fisheye box representations (NOT a rectification comparison — see §0) |
| `2604.10391` FishRoPE | **projection-aware RoPE vs 2D axial RoPE, measured** |
| `2603.27818` Mixed pinhole+fisheye BEV bench | **the counter-evidence: rectified beats native at range** |
| `2309.16588` ViT Need Registers (ICLR'24) | what a ViT does with low-information patches |
| `2004.07780` Geirhos Shortcut Learning | frame for the rig-identity mask shortcut |
| (`1903.11027` nuScenes, already banked) | the field's projection convention |

## §0 — One claim I could NOT source, and am therefore NOT using

A search-engine summary asserted *"cylindrical rectification improves over rectilinear
undistortion by nearly 4 mAP points"*. I banked and full-text-searched the two papers it
could plausibly have come from (`2012.02124` Rashed WACV'21; FisheyeDetNet `2404.13443`)
and the number is in **neither** — `2012.02124` contains exactly one occurrence of the word
"FOV" and no rectification comparison at all; it is about bounding-box *representations*
(oriented / ellipse / curved / polygon), not projections. **Evidence class:
PUBLISHED-SECONDARY, primary not located ⇒ INADMISSIBLE.** Do not quote it.
The nearest admissible statement is qualitative, from FisheyeDetNet: *"Rectilinear
correction suffers from loss of Field of View (FoV) and sampling issues, Piece-wise linear
with artifacts at transition areas and massive bleeding in the image, and Cylindrical as a
quasi-linear correction, offers a practical trade-off."* (PUBLISHED, primary, qualitative).

---

# Q1 — What projection does the field actually feed?

**Answer: rectified pinhole, universally, and mostly without saying so.**

**MEASURED (by me, over the banked primaries).** I full-text-searched the banked PDFs of
UniAD (`2212.10156`), VAD (`2303.12077`), VADv2 (`2402.13243`), SparseDrive (`2405.19620`),
Hydra-MDP (`2406.06978`), NAVSIM (`2406.15349`), PARA-Drive (`paradrive-cvpr2024`) and
DiffusionDrive (`2411.15139`) for `cylindric|equirect|fisheye|panoram|distortion|rectif|undistort`.
**Occurrences: 0, 0, 0, 0, 0, 0, 0 and 1** (the single DiffusionDrive hit is "rectified flow",
not image rectification). None of the eight canonical E2E stacks *discusses* image projection
at all. They inherit the dataset's convention silently.

**PUBLISHED (primary, nuScenes devkit source).** `nuscenes-devkit`
`python-sdk/nuscenes/utils/geometry_utils.py::view_points` is a pure `K @ X` followed by
division by `z` — **no distortion coefficients, no lens model of any kind**. Every
nuScenes-derived stack (UniAD, VAD/VADv2, SparseDrive, GenAD, DriveWorld, PARA-Drive) and
every NAVSIM/nuPlan-derived stack (Hydra-MDP, DiffusionDrive, DriveSuprim, GoalFlow)
therefore operates on a strict pinhole model.

**PUBLISHED (primary, `2603.27818`, 2026).** *"Major autonomous driving benchmarks
(nuScenes, Waymo) exclusively use pinhole cameras."* GAIA-2 (`2503.20523`) generates
448x960 conventional multi-camera streams; Vista trains on nuScenes + OpenDV YouTube
dashcam footage — rectilinear in both cases.

**Does ANY published driving stack train on cylindrical camera input? — Absence claim,
THREE probe formulations:**
1. WebSearch `"autonomous driving perception cylindrical projection camera image input
   end-to-end planner 2024 2025"` → zero camera-cylindrical hits; only LiDAR cylindrical voxels.
2. arXiv API `abs:"cylindrical projection" AND abs:"driving"` → **1 result**, `2007.10323`
   Pillar-based Object Detection — **LiDAR point clouds**, not camera.
3. arXiv API `all:"cylindrical image" AND all:"autonomous driving"` → **0 results**
   (`totalResults` 0).

⇒ **In the AV camera domain, "cylindrical" appears only in the LiDAR literature**
(Cylinder3D / PolarNet / pillar-based multi-view), where it is uncontroversial. Panoramic
*camera* work exists but sits in the panoramic-segmentation community (PASS, Trans4PASS,
DensePASS, 360BEV) and is essentially disjoint from the E2E-driving literature. The
adjacent AV work that touches non-pinhole camera geometry is the **fisheye** line
(WoodScape, OmniDet, FisheyeDetNet, F2BEV, FishBEV, FishRoPE, `2603.27818`) — and that
line is about *surround/near-field parking*, not forward planning.

**Verdict for TanitAD: UNUSUAL-BY-DESIGN, not BEHIND.** We are outside the convention, and
the cost of that is real but specific: **no pretrained backbone, no published baseline, and
no benchmark shares our input geometry.** It is not a correctness defect. Note the honest
consequence though — every external number we might want to compare against (nuScenes ADE,
NAVSIM PDMS, WoodScape mAP) was produced on a different projection, so cross-corpus
comparison carries an uncontrolled term we cannot bound.

---

# Q2 — The ViT + non-pinhole domain gap: what is MEASURED?

## (a) The clean experiment — same scenes, same classes, only the projection changes

This is the number the review needs, and it exists.
**PUBLISHED (primary): Trans4PASS `2203.01452` Table 2** — Stanford2D3D **Pinhole (SPin)**
vs Stanford2D3D **Panoramic (SPan)**, *same building, same 13 classes, fold-1*, all models
trained on pinhole and evaluated on both:

| network | SPin mIoU | SPan mIoU | gap |
|---|---|---|---|
| Fast-SCNN (CNN) | 41.71 | 26.86 | −14.85 (−35.6 % rel) |
| SwiftNet R-18 (CNN) | 42.28 | 34.95 | −7.33 (−17.3 %) |
| DANet R-50 (CNN) | 43.33 | 37.76 | −5.57 (−12.9 %) |
| DANet R-101 (CNN) | 40.09 | 31.81 | −8.28 (−20.7 %) |
| Trans4Trans-T (ViT/PVT) | 41.28 | 24.45 | −16.83 (−40.8 %) |
| Trans4Trans-S (ViT/PVT) | 44.47 | 23.11 | −21.36 (−48.0 %) |
| **Trans4PASS-T (distortion-aware)** | 49.05 | 46.08 | **−2.97 (−6.1 %)** |
| **Trans4PASS-S (distortion-aware)** | 50.20 | 48.34 | **−1.86 (−3.7 %)** |

⇒ **Projection-naive networks lose 13–48 % relative going pinhole → equirectangular on
matched scenes. A distortion-aware architecture cuts that to 3.7–6.1 %** — i.e. the fix
recovers roughly 75–90 % of the gap.

**⚠️ The caveat that changes the verdict for us:** every row above is *trained on pinhole,
tested on panoramic*. It measures a **transfer** penalty. **TanitAD trains natively on
cylindrical and therefore never pays it.**

The larger, more-quoted number — Cityscapes → DensePASS, **SegFormer-B2 81.0 → 42.4 mIoU,
−38.6 points** (Trans4PASS Table 1) — is **confounded** (different cities, cameras, weather,
label distribution *and* projection) and should not be quoted as a projection cost. Table 2
is the admissible one.

## (b) The number that actually applies to a from-scratch model

**PUBLISHED (primary): Trans4PASS Table 4c vs Table 2**, same fold, same architecture:

| arm | mIoU |
|---|---|
| Trans4PASS-S trained on pinhole, tested pinhole | 50.20 |
| Trans4PASS-S trained on pinhole, tested panoramic | 48.34 |
| **Trans4PASS-S trained on panoramic (supervised), tested panoramic** | **53.31** |
| DANet trained pinhole/tested pinhole | 43.33 |
| **DANet trained on panoramic (supervised)** | **44.15** |

⇒ **When trained natively, the panoramic projection BEATS the pinhole one: +3.11 mIoU
(+6.2 % rel) for Trans4PASS-S and +0.82 (+1.9 % rel) for DANet, on the same building and
the same classes** — and it does so from **1,413** panoramic images against **70,496**
pinhole ones. The projection-change penalty is a *transfer* penalty, not an intrinsic one.

⚠️ Fairness: the panoramic image also carries a wider FOV, so this conflates projection with
information content. That conflation is exactly TanitAD's situation (we chose cylindrical
*in order* to keep 120°), so it is the right comparison for us — but it is not a pure
projection isolation, and I will not claim it is.

## (c) Positional encodings specifically

**PUBLISHED (primary): FishRoPE `2604.10391` Tables 1–2**, identical DINOv2-B backbone,
only the position encoding changes:

| task | 2D axial RoPE | projection-aware RoPE | delta |
|---|---|---|---|
| WoodScape 2D det, mAP@0.5 | 52.5 | **54.2** | +1.7 (+3.2 % rel) |
| SynWoodScapes BEV seg, mIoU | 61.4 | **65.1** | +3.7 (+6.0 % rel) |

Mechanism, quoted: *"a fixed pixel offset near the principal point subtends approximately
3–5x greater angular extent than the same offset at the image periphery"* (190° Kannala–Brandt),
so *"tokens at equal pixel separation but different radial positions correspond to markedly
different spatial relationships, yet Cartesian encodings assign them identical positional
structure."* θ-only ablation gives 53.6/64.7; adding φ adds +0.6 mAP / +0.4 mIoU.

**⭐ This is the finding that most favours TanitAD, and it is precise.** FishRoPE's fix is
to make attention operate on **angular** separation instead of pixel separation. **In a
cylindrical projection the column axis is already linear in azimuth, so pixel separation
IS angular separation along x — our ViT-5 axial RoPE on the column axis is exactly
FishRoPE's θ component, obtained for free from the projection.** The paper itself notes it
*"naturally reduces to the standard formulation under pinhole geometry"*; the cylindrical
case is the other geometry where the standard formulation is already correct.
The residual is the row axis only, and their own ablation bounds that at **+0.6 mAP /
+0.4 mIoU** — before accounting for the fact that our elevation range is ±22.7° rather than
their ±95°, which shrinks it further.

## (d) Convolutional inductive bias

**PUBLISHED (primary, mechanism): `2603.27818`** — non-pinhole projections create
*"non-uniform spatial sampling, which violates the translational equivariance of standard
CNNs."* This is a statement about **CNNs**. ViTs have no built-in translation equivariance
to break — they learn position from the PE — so the mechanism is weaker for our architecture.

⚠️ **But the empirical evidence on "are transformers more robust to projection change" is
genuinely mixed and I will not overclaim.** Trans4PASS **Table 1** (outdoor) supports it —
CNN gaps −47 to −52, transformer gaps −38.6 to −43.6, i.e. *"previous transformers reduce
the mIoU gap from ~50 % of CNN-based counterparts to ~40 %"*. Trans4PASS **Table 2**
(indoor, the cleaner experiment) **contradicts it** — the two PVT transformers post the
*worst* gaps in the table (−40.8 %, −48.0 %) while three CNNs sit at −12.9 to −20.7 %.
⇒ **UNKNOWN.** "ViTs handle distortion better than CNNs" is not established.

## (e) Pretrained-backbone transfer

**PUBLISHED (primary): Tangent Images `1912.09390` Table 3** — perspective-trained
FCN-ResNet101 evaluated on spherical data via tangent images, no fine-tuning: mAcc **−2.6 %**,
mIOU **−7.5 %** (all folds; per-fold up to −17.2 % mAcc / −10.6 % mIOU). Text: *"we preserve
about 97 % of perspective network accuracy and 93 % of the mean IOU."*
That is the **best-case** transfer penalty, because tangent images are locally planar by
construction. Table 4 shows tangent-image transfer with **no** fine-tuning (70.8 mAcc /
54.9 mIOU at level 8) **beating** a spherical-native-trained prior method (55.1 / 47.1).

**Verdict on Q2 for TanitAD: CONSISTENT, and the exposure is smaller than it looks.**
The 13–48 % figures are transfer penalties we do not pay. The one lever the literature says
is worth paying for on non-pinhole input — a projection-aware positional encoding — we
already have on the axis that matters (azimuth), by construction. **The residual, bounded by
FishRoPE's own ablation, is ≤ +0.6 mAP-equivalent on the row axis.**

---

# Q3 — Position-dependent appearance: is it benign or malign?

**The brief's framing contains one geometric error worth correcting, and one worry that is
correct.**

## The error: "a car is SHEARED differently at different azimuths" — FALSE for cylindrical

**ESTIMATED (exact closed-form geometry; inputs INHERITED, derivation one line).**
For a cylinder of radius `f`, a point at (azimuth θ, elevation φ) maps to `x = f·θ`,
`y = f·tan φ`. Against arc length on the unit sphere the local magnifications are
`dx/dl_θ = f·sec φ` and `dy/dl_φ = f·sec²φ`. The **local shape anisotropy is their ratio,
`sec φ` — a function of ELEVATION ONLY. It is exactly independent of azimuth.**

| | local anisotropy | sampling-rate spread |
|---|---|---|
| **our cylindrical** (rows `f·tan φ`) | **1.000 → 1.084**, azimuth-**independent** | horizontal **1.000** (exact); vertical 1.176 |
| our cylindrical if rows were `f·φ` | 1.000 → 1.095, azimuth-independent | horizontal 1.000; vertical 1.000 |
| **pinhole rectification of the same 120°** | 1.000 → **2.000**, varies **with azimuth** | horizontal **4.000** |

⇒ **The same car at the left edge and at the centre, at the same elevation, has an
identical local shape in our frame** (to within the elevation term, ≤1.084x over the whole
±22.7° of image height). The pinhole alternative is the one that shears with azimuth —
by 2.0x at the edge — and oversamples the edge by 4.0x.

**⭐ Stronger still: a camera yaw is EXACTLY a horizontal pixel translation in a cylindrical
image** (`Δx = f·Δθ`). Translation equivariance in azimuth — the property `2603.27818` says
non-pinhole projections destroy for CNNs — **is exactly restored, not destroyed, by our
choice.** In a pinhole image the same yaw is a homography and a point at 60° moves 4.0x
further in pixels than one at 0°. For a driving model whose dominant nuisance transform is
ego yaw, this is a real and unusual advantage. (This is the same property that makes
cylindrical the standard panorama-stitching surface.)

## The worry that is correct: straight world lines bow

In a cylindrical projection, vertical world lines project to **exactly straight image
columns** (all their points share one azimuth) and the horizon (φ=0) is straight — but every
other straight line bows, with curvature depending on elevation. **Lane markings, road edges
and curbs are exactly the "other" case.** A model estimating lateral offset, heading and
path curvature is reading geometry the projection has bent, and the bending varies with
distance-ahead (elevation).

**Is the cost measured anywhere? Partly.** Trans4PASS's DPE ablation is the closest
isolation: **PUBLISHED (primary), Table 3 group 1 + text** — replacing DPT's DePatch with
their distortion-aware Deformable Patch Embedding, everything else fixed, adds
**+3.01 mIoU on pinhole (Cityscapes) and +9.39 mIoU on panoramic (DensePASS)**. The same
architectural change is worth **3.1x more on the distorted projection.** That is the
literature's best measurement of "the patch grid does not match the geometry" as a cost
term, and it says the cost is real and patch-embedding-shaped.
(I do **not** rely on their Table 6 mIoU row — under text extraction it reports DPT and DPE
both at 45.89, inconsistent with Table 3's 36.50 for DPT, so it is a layout-parse artifact.)

**Is it benign or malign?** The literature supports **benign-but-not-free**, and I found no
paper that measures a *data-requirement multiplier* for position-conditioned appearance —
probed as "distortion-aware ViT", "panoramic semantic segmentation", "fisheye positional
encoding" and via the panoramic-survey `2606.27745`; the field measures *accuracy* deltas,
never *sample-efficiency* deltas. **UNKNOWN, and it is a genuine gap in the literature, not
just in my search.**

**Verdict on Q3: CONSISTENT, with one specific soft spot.** Cylindrical is the projection
that *minimises* position-dependent appearance at 120° — the anisotropy is azimuth-invariant
and ≤1.084x, versus 2.0x and azimuth-varying for the pinhole alternative. The residual cost
is that ground-plane straight lines are curved, which lands precisely on the LATERAL metric
family (curvature error, heading, cross-track).

---

# Q4 — The case FOR cylindrical, stated as strongly as the evidence allows

This is the sub-question where TanitAD comes out best, and the case is quantitative.

**1. Pinhole rectification at 120° is genuinely wasteful — exact numbers.**
*ESTIMATED (exact geometry; `f_ref`, W, H INHERITED).*
`az_max = 320/305.5775 = 1.04720 rad = 60.00°`, i.e. **exactly 120° hFOV**, matching the
rig's own name `camera_front_wide_120fov` — an independent cross-check that the cylindrical
reading of `f_ref` is right.
- At **equal image width (640 px)**, a pinhole rectification needs `f = 320/tan60° = 184.75`
  — **the image centre would carry only 60.5 % of the angular resolution we have now.**
  Cylindrical buys **1.654x more pixels-per-degree at the centre**, which is where the lead
  vehicle and the road ahead are.
- To **match** our present centre resolution, a pinhole frame would need to be **1,059 px
  wide (1.654x)** → at patch 16, **67 tokens/row instead of 40**, i.e. **1.68x the ViT
  sequence length and 2.81x the attention FLOPs** — for the same forward FOV and *less*
  useful information at the edges.
- The pinhole edge is oversampled **4.0x** relative to its centre (corner ≈ **4.70x**), so
  most of those extra pixels carry no new scene content.

**2. The field's own practitioners endorse it.** PUBLISHED (primary, qualitative),
FisheyeDetNet: *"Rectilinear correction suffers from loss of Field of View (FoV) and
sampling issues, Piece-wise linear with artifacts at transition areas and massive bleeding
in the image, and **Cylindrical as a quasi-linear correction, offers a practical
trade-off**."* WoodScape (`1905.01489`) is built on the same premise: *"Rectifying fisheye
images often leads to a loss of field of view and resampling distortion."*

**3. Wide FOV beats pinhole once the network is adapted to the format — MEASURED.**
Tangent Images Table 3, all folds, after 10 epochs on the spherical format: mAcc
**+7.2 %**, mIOU **+5.6 %** over the *same network's* perspective performance, with the
authors attributing it to *"the greater FOV provided by the 360° image."* Trans4PASS Table
4c says the same thing on the supervised arm (§Q2b: +6.2 % rel).

**4. Uniform angular resolution matters, and mismatch is expensive.**
Tangent Images **Figure 6** (network trained at a level-8-equivalent angular resolution,
tested at levels 5–10, each level a factor 2):

| angular-res mismatch | mAcc | vs matched |
|---|---|---|
| 1x (matched, L8) | 59.6 | — |
| 2x (L9) | 56.4 | −5.4 % |
| 2x (L7) | 44.1 | −26.0 % |
| 4x (L10) | 37.8 | −36.6 % |
| 4x (L6) | 21.3 | −64.3 % |

A **4x** angular-resolution spread costs **37–64 % relative mAcc**. A 120° pinhole
rectification embeds **exactly a 4.0x angular-resolution spread inside every single frame**;
cylindrical holds it at **1.000x in azimuth by construction.**
⚠️ **Fairness — this is an upper bound, not our expected cost.** Figure 6 measures a
*train/test* mismatch across whole images; a network that sees the same within-image
gradient in every sample can learn position-conditioned filters. I am using it as a
directional argument, not as a predicted delta, and it should not be quoted as one.

**5. The honest counter-evidence, which I am not going to bury.**
**PUBLISHED (primary): `2603.27818` Table II**, KITTI-360, mAP by range, rectified-images
training vs distortion-aware models on raw fisheye:

| model | 0–10 m | 10–20 m | 20–30 m | 30–40 m | 40–50 m |
|---|---|---|---|---|---|
| BEVFormer | −4.4 % | −22.7 % | −31.0 % | −15.7 % | −36.5 % |
| PETR | **+1.3 %** | −5.8 % | −16.5 % | −6.8 % | −0.7 % |
| BEVDet | −2.7 % | −5.0 % | −12.0 % | −29.8 % | −36.8 % |

Their own conclusion: *"Training with rectified images excels at nearly all distances over
distortion-aware models... Beyond 10 m, rectified models consistently outperform their
distortion-aware counterparts across all architectures, with the performance gap widening at
longer ranges."*

**Why I still judge this weak evidence against us — three scope limits, all stated in the
paper itself:** (i) every "distortion-aware" arm is a **pinhole-designed architecture** with
a polar-BEV patch, on **ImageNet-pretrained** backbones — rectification restores exactly the
geometry those backbones and architectures assume, so the result largely measures
*pretrain/architecture match*, and **TanitAD has no pretrained backbone to match**;
(ii) their input is **180°+ Kannala–Brandt fisheye**, where the angular-extent ratio is
**3–5x** (FishRoPE's measurement of the same lens class) — ours is a **120° cylindrical**
with an azimuthal ratio of **1.0x**; (iii) the same table's near-field bin has the best
architecture (PETR) **winning** on the distorted input. It is nonetheless the one result in
this pass that points the other way, and it points at **long range**, which is where a
planner's speed and headway decisions live.

**Verdict on Q4: CONSISTENT, and the choice is defensible on numbers.** At 120°,
cylindrical is the *better* projection on resolution economy (1.654x centre pixels-per-degree
at equal width, or 2.81x attention FLOPs saved at equal centre resolution), on shape
constancy (1.084x vs 2.0x anisotropy), and on yaw equivariance (exact vs homography). The
published AV practitioners' own framing calls it *"a practical trade-off"*. This is the
axis where TanitAD is **ahead of the convention, not behind it** — provided we say so with
the caveat that no published driving stack has validated it end-to-end.

---

# Q5 — The ~9 % invalid/padded region on rig B

**No projection-specific literature exists on masked regions in ViT inputs** — probed as
"invalid pixels ViT", "padding shortcut vision transformer", "masked region border artifact"
and across the fisheye line, where the phenomenon is *named* (FisheyeYOLO: rectification
produces *"non-rectangular image due to invalid pixels"*) but never *measured*.
**ABSENT — and this is the one place where TanitAD is genuinely operating without cover.**
Two adjacent literatures apply, and both say the same thing.

## (a) A ViT will not ignore a constant region — it will repurpose it

**PUBLISHED (primary): Darcet et al. `2309.16588` (ICLR'24).** The mechanism, quoted:
artifacts are *"high-norm tokens appearing during inference primarily in low-informative
background areas of images, that are repurposed for internal computations"*; they appear
*"in patches similar to their neighbors, meaning patches that convey little additional
information"*; and *"the model learns to recognize patches containing little useful
information, and recycle the corresponding tokens to aggregate global image information
while discarding spatial information."*

**A constant padded region is the maximal case of "similar to its neighbours".** Our ~9 %
invalid region is, by construction, the most attractive real estate in the frame for this
mechanism.

Measured consequences (their Table 1, linear probing on individual patch tokens, DINOv2-g):

| probe | normal patches | outlier patches | [CLS] |
|---|---|---|---|
| ImageNet-1k | 65.8 | **69.0** | 86.0 |
| Aircraft | 17.1 | **79.1** | 87.3 |
| Stanford Cars | 10.8 | **85.2** | 91.5 |
| CUB | 18.6 | **84.9** | 91.3 |

i.e. those tokens stop carrying local content and start carrying global content. Prevalence:
**2.37 %** of DINOv2-g tokens exceed norm 150. Downstream cost of leaving it unfixed —
LOST unsupervised object discovery on DeiT-III, the task that depends most on the *spatial*
structure of the feature map: VOC07 **11.7 → 27.1**, VOC12 **13.1 → 32.7**, COCO20k
**10.7 → 25.1** with registers — **+132 % to +150 % relative**. Dense linear probes move
less: DINOv2 ADE20k **46.6 → 47.9**, NYUd rmse **0.378 → 0.366**.

**⚠️ The fairness caveat, from the same paper: the artifacts** *"only appear after a
sufficiently long training of a sufficiently big transformer."* Our ViT is from-scratch and
small. **Whether the mechanism is active in our regime is UNKNOWN and cheaply testable**
(histogram the output token L2 norms and check whether the high-norm mass sits inside the
rig-B invalid mask). The fix is also cheap and, per their Table 2a and Figure 8, **carries
no measured downside**: append 4 learnable register tokens; *"one register is sufficient to
remove artefacts"*, more helps ImageNet, and dense tasks have an optimum near 4.

## (b) The mask is a perfect rig label — the shortcut risk

**PUBLISHED (primary): Geirhos et al. `2004.07780`.** Shortcut learning is the failure where
a model latches onto an unintended cue that is predictive in-distribution and fails when the
cue's correlation breaks.

**ESTIMATED (mechanism, not measured on our data).** The invalid region is a *deterministic
function of rig identity* — one convolutional patch-embed filter on the mask boundary reads
rig ID at essentially 100 % accuracy, and the learned absolute PE makes the location free.
**This is only a defect if rig identity correlates with something the model is asked to
predict** (geography, season, time of day, vehicle dynamics, speed distribution, label
prior). If it does, the model gets a free grouping variable and will use it, and the failure
surfaces only OOD — which for us means at deployment on a third rig or a rebalanced split.
The programme has already been bitten by exactly this class twice (the nav-echo bijection
scoring 1.0000; REF-A I-JEPA's ~80 % val-in-train leak).

**Verdict on Q5: WEAK-BY-DESIGN, and it is the cheapest thing on this list to fix.**
Three concrete items, in cost order: (1) **add 4 register tokens** to the ViT — free, no
measured downside, removes the strongest reason a ViT would colonise the invalid region;
(2) **feed the validity mask explicitly** (an extra input channel or an attention mask that
zeroes invalid tokens) so the region is *declared* invalid rather than *inferred* — this is
the standard treatment and it converts a latent cue into a known one; (3) **measure the
shortcut before assuming it is benign** — train a linear probe from the encoder's features
to rig identity, and, more decisively, check whether rig ID is predictive of any target
(speed, situation class, ADE) in the corpus. If rig ID is uninformative about the targets,
the shortcut has nothing to exploit and items 1–2 suffice.

---

# The single biggest projection-related risk to TanitAD

**It is not the projection. It is the ~9 % invalid region interacting with a from-scratch
ViT that has learned absolute position embeddings — and the number that makes it a risk is
131.6 %.**

That is the measured improvement on LOST object discovery (VOC07 11.7 → 27.1; VOC12 +149.6 %;
COCO20k +134.6 %, Darcet et al. `2309.16588`) from removing exactly one thing: a ViT
repurposing its **low-information patch tokens** into global-computation registers that
*"discard spatial information"*. Our rig-B frames hand such a ViT a constant, perfectly
localisable, 9 %-of-frame block of the lowest-information tokens obtainable — and a learned
absolute PE that makes finding them free. LOST is the task in that paper that depends on the
*spatial* structure of the feature map, which is precisely what a latent world model reads.
The same paper's dense linear probes move only +1.3 mIoU, so the plausible range is wide —
but the mechanism is documented, our input is its worst case, and **we have never measured
whether it is happening.**

By contrast, the two things the review might have expected to be the risk are, on the
evidence, **not**:
- **The cylindrical choice itself is defensible on numbers** — 1.654x more centre
  pixels-per-degree than a pinhole rectification at equal width, local shape anisotropy
  1.084x versus 2.0x, exact yaw-translation equivariance, and native-trained panoramic
  beating native-trained pinhole by +6.2 % rel on matched scenes (Trans4PASS `2203.01452`).
- **The ViT PE mismatch that FishRoPE fixes for +1.7 mAP / +3.7 mIoU does not apply to our
  azimuth axis** — our column axis is already linear in angle, so we have FishRoPE's θ
  component for free; the residual (row axis) is bounded by their own ablation at
  **+0.6 mAP / +0.4 mIoU**.

The genuine second-order risk, worth one line: **long range**. `2603.27818` Table II is the
one primary pointing the other way, showing rectified input beating distortion-aware input
by **12–31 % relative mAP in the 20–30 m bin** across all three architectures. Its scope
limits are real (180° fisheye, pinhole-pretrained backbones, pinhole-designed heads — none
of which we share), but long range is where our LONGITUDINAL family lives, and it is the one
place the literature says a non-pinhole projection has actually cost somebody accuracy.

---

## Deliverable manifest

| artifact | location |
|---|---|
| this report | `…/scratchpad/lit_axis3_projection.md` (scratchpad) + returned in full in the agent message |
| 10 banked primaries | `TanitAD Research Lab/Library/papers/` (repo, working tree) |
| library index rows | `TanitAD Research Lab/Library/library.json` + generated `LIBRARY.md` (repo) |
| verification | `python tools/kb_add.py --verify` → **199 entries, 0 orphans, 0 problems** |
| extracted paper text (working files) | `…/scratchpad/txt/*.txt` (scratchpad, disposable) |

No commits, no pushes made.
