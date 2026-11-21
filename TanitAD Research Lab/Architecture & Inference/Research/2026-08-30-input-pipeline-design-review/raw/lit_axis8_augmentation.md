# AXIS 8 — "Everything else that is weak by design": augmentation, photometric handling,
# ego-motion compensation, frame drops, label alignment, normalisation

Literature stream for the TanitAD input-pipeline design review, 2026-08-30.
Every number below is either MEASURED-BY-THE-CITED-PAPER (I re-read it out of the banked PDF —
marked **PUBLISHED (primary, verified from banked PDF)**) or read off a repo config
(**PUBLISHED (source code, verbatim)**). Anything I could not verify in the primary is marked
PUBLISHED-SECONDARY and is **inadmissible for the registry or the paper**.

Our own setup is INHERITED from the brief throughout; I did not re-derive it and I flag
every place where the measurement stream must confirm presence/absence.

---

## 0. THE ONE-PARAGRAPH VERDICT

The single strongest finding on this axis is **not** about colour jitter. It is that the
programme's measured signature — open-loop reproduction ~97.9 %, closed-loop ~0–5 %, action
channel correlating with realised motion at r = 0.9988 — is a **named, characterised,
extensively-measured failure mode in the imitation-learning literature** (causal
misidentification / the copycat problem / the inertia problem), and that **every published fix
for it lives in the input pipeline**: drop the confounded input, perturb the state distribution,
or remove the nuisance information adversarially. The literature further measures that the
open-loop metric moves the *wrong way* when the shortcut is closed — so our own open-loop
numbers cannot be used to select the fix. Three primaries put hard numbers on it
(§7). The augmentation questions proper (§1–§5) are real but second-order next to this.

Second strongest: **BEVDet4D measured that adding an UNCOMPENSATED second frame is worse than
using no second frame at all** (§4). Our 9-channel stack is uncompensated by construction.

Third: **temporal consistency of augmentation across the frames of a clip is worth +11.5 top-1
points in video representation learning** (§1c). If we ever add photometric augmentation, doing
it per-frame inside our 3-frame stack is the specific way to make it harmful.

---

## 1. DATA AUGMENTATION IN E2E / BEV DRIVING — WHAT THE FIELD DOES AND MEASURES

### 1a. The cleanest augmentation ablation in AV: BEVDet Table 4

**PUBLISHED (primary, verified from banked PDF `2112.11790`, Table 4 + §4.3 text, p.11–12).**
BEVDet-Tiny, nuScenes val, fixed 20-epoch schedule. IDA = image-view-space augmentation,
BDA = BEV-space augmentation, BE = BEV encoder. They report **both** the best-during-training
and the final-epoch score, precisely to expose overfitting.

| ID | IDA | BDA | BE | mAP-best | NDS-best | mAP-final | NDS-final |
|----|-----|-----|----|----------|----------|-----------|-----------|
| A | – | – | ✓ | 0.230 (e4) | 0.310 | 0.174 (−5.6 %) | 0.283 |
| B | ✓ | – | ✓ | 0.205 (e10) | 0.308 | 0.178 (−2.7 %) | 0.303 |
| C | – | ✓ | ✓ | 0.262 (e11) | 0.357 | 0.236 (−2.6 %) | 0.348 |
| D | ✓ | ✓ | ✓ | **0.316 (e17)** | **0.393** | **0.312 (−0.4 %)** | **0.392** |
| E | – | – | – | 0.231 (e10) | 0.307 | 0.215 (−1.6 %) | 0.306 |
| F | ✓ | – | – | 0.276 (e14) | 0.347 | 0.269 (−0.7 %) | 0.345 |
| G | – | ✓ | – | 0.253 (e12) | 0.345 | 0.224 (−2.9 %) | 0.337 |
| H | ✓ | ✓ | – | 0.299 (e20) | 0.373 | 0.299 (−0.0 %) | 0.373 |

Headline numbers, quoted from the paper's own text:
- **No augmentation at all saturates at epoch 4 and then overfits: 23.0 % → 17.4 % (−5.6).**
- **Full augmentation: +8.6 mAP at the peak, and the end-of-training degradation collapses
  from −5.6 % to −0.4 %** (a ~14× smaller overfitting gap). On the final-epoch number the gap
  is **+13.8 mAP (17.4 → 31.2)**.
- BDA alone beats the no-aug baseline **by +3.2 mAP at the peak**.
- ⚠️ **IDA ALONE IS WORSE THAN NOTHING when the BEV encoder is present (20.5 vs 23.0).**
  Verbatim: *"IDA has a negative impact on the performance when BDA is absent but has a positive
  impact on the contrary."* Without the BEV encoder the sign flips (F 27.6 > G 25.3).

What the operations actually are (**PUBLISHED, primary, verified**):
- **IDA = random flipping, scaling (s ∈ [W/1600 − 0.06, W/1600 + 0.11]), rotating (±5.4°), cropping.**
- **BDA = random flipping, rotating (±22.5°), scaling ([0.95, 1.05]).**
- ⭐ **Neither contains any photometric augmentation.** BEVDet's whole augmentation budget is geometric.

**Scope warning I must state, because it is exactly the trap CLAUDE.md warns about:** BDA is a
BEV-space operation. **TanitAD has no BEV space.** The half of this result that transfers to a
front-camera latent world model is IDA — and IDA in isolation was *negative* in the arm that had
a strong downstream spatial encoder. So this table is **not** a licence to bolt image-space
augmentation onto our stack and expect +8.6.

**Verdict for us:** the "does no-augmentation cost you?" question is **answered YES in AV, and
the mechanism is overfitting, not invariance** — but the transferable operation set is small.
**BEHIND, with a caveat.**

### 1b. What UniAD / VAD / BEVFormer actually apply (repo configs, not prose)

**PUBLISHED (source code, verbatim from the repos).**

| stack | normalisation | photometric aug at train | geometric aug at train | flip |
|---|---|---|---|---|
| BEVFormer (`bevformer_base.py`) | `mean=[103.530,116.280,123.675]`, `std=[1.0,1.0,1.0]`, `to_rgb=False` | **`PhotoMetricDistortionMultiViewImage` YES** | none in base cfg | **no** |
| UniAD (`base_track_map.py`) | identical to BEVFormer | **`PhotoMetricDistortionMultiViewImage` YES** | none | **no** |
| VAD (`VAD_base_e2e.py`) | `mean=[123.675,116.28,103.53]`, `std=[58.395,57.12,57.375]`, `to_rgb=True` | **`PhotoMetricDistortionMultiViewImage` YES** | `RandomScaleImageMultiViewImage scales=[0.8]` (also at test) | **no** |
| BEVDet | (see above) | **none** | IDA + BDA | **yes** (detection task) |

`PhotoMetricDistortionMultiViewImage` defaults (**verified from
`projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`**):
`brightness_delta=32`, `contrast_range=(0.5,1.5)`, `saturation_range=(0.5,1.5)`, `hue_delta=18`,
each applied with p=0.5. It is removed from the test pipeline in all three.

⭐ **Three findings fall straight out of this table:**
1. **The field's E2E planners DO apply photometric jitter, and it is the ONLY augmentation two of
   the three apply.** If we apply none (to confirm), we are behind the *universal* practice, not
   an optional one.
2. **Nobody in the E2E/planning group flips.** See §1d.
3. **BEVFormer and UniAD use `std = 1.0` — i.e. mean-subtraction only, no variance
   normalisation, on the 0–255 range, with the Caffe BGR ImageNet mean.** That is inherited from
   the pretrained ResNet-101-DCN checkpoint, not chosen. VAD uses proper torchvision ImageNet
   mean/std. ⇒ **Fixed dataset-level constants in every case; nobody normalises per-frame or
   per-sequence.** (§3)

### 1c. ⭐ Temporally-consistent augmentation — the finding that most directly threatens our 9-channel stack

**PUBLISHED (primary, verified from banked PDF `2008.03800`, CVRL, Table 9 p.8 + §1 text).**
Kinetics-600, linear eval on frozen R3D-50:

| temporal aug | spatial aug | **temporal consistency** | top-1 | top-5 |
|---|---|---|---|---|
| ✓ | – | – | 33.0 | 57.3 |
| – | ✓ | – | 40.9 | 66.6 |
| ✓ | ✓ | **✗** | **52.3** | 76.0 |
| ✓ | ✓ | **✓** | **63.8** | 85.2 |

Verbatim: *"Adding temporal consistency further improves the top-1 accuracy to 63.8 % by a large
margin of 11.5 % over 52.3 %."* And the mechanism, verbatim from §1: *"Simply applying spatial
augmentation independently to video frames actually hurts the learning because it breaks the
natural motion along the time dimension."*
Consistent parameters are: crop size ratio + aspect ratio, horizontal-flip decision, colour-jitter
decision, greyscale decision — drawn **once per clip**.

⚠️ **And the field's own AV code violates this.** `PhotoMetricDistortionMultiViewImage` draws its
random parameters **independently inside the per-image loop** (verified in the BEVFormer source:
`for img in imgs:` then `if random.randint(2): delta = random.uniform(...)`). For BEVFormer's
6 simultaneous camera views that is defensible-ish; for its *previous-frame* inputs it is not,
and no paper in this set reports a controlled ablation of it. So:
- the **mechanism** is MEASURED (CVRL);
- the **AV-specific magnitude** is **UNKNOWN** — nobody has run it.

**Verdict for us:** our encoder eats **3 consecutive frames channel-stacked into one patch-embed
conv**. That is the *most* temporal-consistency-sensitive structure possible: an independent
brightness delta per channel-group is literally an injected fake optical-flow/illumination-change
signal inside a single token. **If we add photometric augmentation, drawing it once per WINDOW is
not an optimisation, it is a correctness requirement.** If we currently have none, we are
**BEHIND** — but the cheap fix is also the one with a known failure mode if done naively.

### 1d. ⭐ Horizontal flip in driving — does the field flip?

**PUBLISHED (source code + primary).** Answer, with the mechanism:

- **Detection/BEV stacks flip** (BEVDet IDA and BDA both contain random flipping) — because the
  label is a set of 3D boxes, and a box mirrors trivially. No steering label exists to invert.
- **The E2E planners in this set do NOT flip.** BEVFormer, UniAD and VAD all have no flip
  transform at train time and `flip=False` in `MultiScaleFlipAug3D` at test time.
- ChauffeurNet does the *adjacent* thing rather than a flip: **PUBLISHED (primary, verified,
  `1812.03079` §6.1)** — *"the vertical-axis of the top-down coordinate system for each training
  example is randomly oriented within a range of Δ = ±25° of our agent's current heading, in order
  to avoid a bias for driving along the vertical axis"*, and Δ=0 at inference. That is a
  **rotation** augmentation whose motivation is removing a directional prior — the closest thing
  in the literature to a "don't memorise the frame orientation" argument.

I found **no** primary that flips a driving image and negates the steering/lateral label and
reports the measured delta. (Probed two ways: repo configs across three E2E stacks, and the
augmentation sections of BEVDet/ChauffeurNet/Codevilla.) I therefore record it as
**UNKNOWN — and the reason the field avoids it is structural, not an oversight**: a mirrored
scene has the wrong driving side, mirrored text/signage, mirrored roundabout topology and
mirrored overtaking convention. For a *detector* none of that matters; for a *planner* all of it
does. For a **world model predicting future scene content** it also matters, because the mirrored
world is not a physically reachable world in a right-hand-traffic corpus.

**Verdict for us: CONSISTENT (correctly absent).** Do not add flip. If the measurement stream
finds we *do* flip, that is a defect, not a feature — and it is a defect that would be invisible
in ADE while corrupting exactly the tactical/strategic families the four-family rule exists to
protect.

### 1e. Do driving world models augment at all?

**UNKNOWN — and I want to flag this as a genuine gap rather than paper over it.** An arXiv
abstract search for driving-world-model ∧ data-augmentation returned **0 results**; a second probe
(world-model ∧ counterfactual ∧ driving) hit an HTTP 429 and I did not retry it within budget.
What I *can* say from the adjacent evidence:
- CVRL (§1c) establishes that video models are augmented, and that consistency is what makes it work.
- The photometric-augmentation configs above are for *discriminative* perception heads. A
  **generative/predictive** objective has an argument the discriminative one does not: if you
  photometrically jitter the input but ask the model to predict the *un-jittered* future, you are
  asking it to invert the augmentation; if you jitter input and target consistently, you have
  changed the dynamics it must model. **HYPOTHESIS**, untested, and worth a pre-registered arm.

**Verdict: UNKNOWN.** This is the one sub-question where I would not let a reviewer quote me.

### 1f. The honest counterweight — two primaries that say augmentation is NOT the lever

Both verified from banked PDFs. I am including these because the axis brief expects me to find a
defect, and the discipline is to report the evidence that cuts the other way.

1. **Codevilla et al. 2019 (`1904.08980`, CILRS), verbatim:** *"Training controllers on this
   dataset, we found that **augmentation was not as crucial as reported by previous works**. The
   only regularization we found important for performance was using a 50 % dropout rate after the
   last convolutional layer."*
   ⚠️ **Read the scope before quoting this.** Their *data collection* already contains the
   augmentation: **PUBLISHED (primary, verified)** — *"For each sensor we record data in three
   positions: aligned with the car center, rotated 30 degrees to the left and rotated 30 degrees
   to the right"*, plus steering noise injection during collection. So the sentence means
   *"appearance augmentation was not crucial **on top of** viewpoint augmentation"*. It is not
   evidence against viewpoint augmentation; it is evidence that viewpoint augmentation is the
   part that mattered.
   (Also: they store frames as **PNG** — lossless — like us.)
2. **Kong/Dong et al. 2023 (`2303.11040`, nuScenes-C/KITTI-C/Waymo-C), verbatim:** *"we further
   try several data augmentation strategies... The experiments validate that **they can hardly
   improve corruption robustness**, leaving robustness enhancement of 3D object detection an open
   problem."*
   ⚠️ **Scope again:** the camera-side augmentations they tried were **Mixup and CutMix** — generic
   mixing augmentations, *not* photometric augmentation matched to the photometric corruption. This
   is not evidence that brightness jitter fails to buy brightness robustness. Quoting it as such
   would be the `df`-trap in augmentation costume.

---

## 2. ⭐ VIEWPOINT / TRAJECTORY AUGMENTATION — THE CLASSIC IL FIX, AND THE ONE THAT MATCHES OUR FAILURE

### 2a. The theory: DAgger's compounding-error bound

**PUBLISHED (primary, verified from banked PDF `1011.0686`, §1), verbatim:**
> *"a classifier that makes a mistake with probability ε under the distribution of
> states/observations encountered by the expert can make as many as **T²ε** mistakes in
> expectation over T-steps under the distribution of states the classifier itself induces...
> Intuitively this is because as soon as the learner makes a mistake, it may encounter completely
> different observations than those under expert demonstration, leading to a compounding of
> errors."*
DAgger's contribution is a guarantee **linear** in T instead of quadratic.

**Applied to us (ESTIMATED, arithmetic on our own stated horizon):** we roll 20 steps
(2.0 s @ 10 Hz). T² / T = **20×**. A bound of that shape is the *right order of magnitude* for a
97.9 % → ~5 % collapse, and it says the gap is **structural to off-policy training**, not a
capacity or a tuning problem. It also says the fix must change the **state distribution seen in
training** — which is an input-pipeline change by definition.

### 2b. ChauffeurNet — "without perturbation the car cannot recover", with the number

**PUBLISHED (primary, verified from banked PDF `1812.03079`, Fig. 7 row 2, p.14).** Closed-loop
simulation, 20 scenarios, agent placed at varying lateral offset + heading error on a curved road:

| model | Recovers | Gets Stuck |
|---|---|---|
| **M0 — imitation + past-motion dropout, NO perturbation** | **0 %** | **100 %** |
| M1 — M0 + trajectory perturbation | 50 % | 50 % |
| M2 — M1 + environment losses | 50 % | 50 % |
| M3 — M2 with less imitation | **100 %** | 0 % |
| M4 — M2 + imitation dropout | **100 %** | 0 % |

Verbatim: *"the contrast between the baseline model M0 which is **not able to recover in any of
the situations** and the models M3 and M4 which handle all deviations well."*

And from the abstract, verbatim: *"standard behavior cloning is insufficient for handling complex
driving scenarios, even when we leverage a perception system for preprocessing the input and a
controller for executing the output on the car: **30 million examples are still not enough**."*
(Body: ~26 M examples ≈ **60 days of continuous driving**.)

⭐ **This is the direct rebuttal to "more data / more training will close our closed-loop gap."**
Waymo measured that it does not, at 60 days of driving.

Their augmentation mechanism (**PUBLISHED, primary**): *"add a uniform perturbation to the SDV's
current pose (its rear axle's midpoint coordinate and orientation) and fit a new smooth trajectory
that brings the SDV back to the original target location."* ⚠️ **This is easy for ChauffeurNet
because its input is a RENDERED top-down raster — they can re-render from any pose. It is exactly
what we cannot do with a real camera log.** See §2d.

### 2c. NVIDIA PilotNet — the recipe that DOES work on a real camera log

**PUBLISHED (primary, verified from banked PDF `1604.07316`, §5.2 + Fig. 1 caption).** This is the
most directly actionable item in the whole axis, because it solves the camera problem:

- *"we augment the data by adding artificial shifts and rotations to teach the network how to
  recover from a poor position or orientation. The magnitude of these perturbations is chosen
  randomly from a normal distribution... zero mean, and the standard deviation is **twice the
  standard deviation that we measured with human drivers**."*
- *"Additional shifts between the cameras and all rotations are simulated by **viewpoint
  transformation** of the image from the nearest camera. Precise viewpoint transformation requires
  3D scene knowledge which we don't have. We therefore **approximate the transformation by assuming
  all points below the horizon are on flat ground and all points above the horizon are infinitely
  far away.** This works fine for flat terrain but it introduces distortions for objects that stick
  above the ground, such as cars, poles, trees, and buildings. **Fortunately these distortions
  don't pose a big problem for network training.**"*
- Label correction, verbatim: *"**The steering label for transformed images is adjusted to one that
  would steer the vehicle back to the desired location and orientation in two seconds.**"*

⭐⭐ Three things line up for us almost suspiciously well:
1. We have **one** camera and no 3D scene — PilotNet's exact situation, and their answer is a
   flat-ground homography, with an explicit measured statement that the artefacts are tolerable.
2. Our episode build already reads **`camera_intrinsics` and `sensor_extrinsics`** (the 5-feature
   layer), so the homography is computable from data we already load.
3. **PilotNet's label-correction horizon is 2 seconds. Our prediction horizon is 2.0 s.** The
   "re-aim at the logged trajectory over the horizon" rule maps onto our target tensor directly.

⚠️ The one thing that must be re-derived rather than copied: **our frames are a CYLINDRICAL remap,
not a pinhole projection.** The warp must be composed in the un-remapped pinhole frame (or derived
for the cylindrical model) — applying a pinhole homography to a cylindrical image is the same class
of error as the FOV mistake already in CLAUDE.md (92.6° vs the true 120°).

### 2d. DART and the modern (NVS) route

- **DART (`1703.09327`), PUBLISHED (primary, abstract verified):** inject noise into the
  *supervisor* during collection so the demonstrations cover the learner's error states.
  *"On the grasping in clutter task, DART obtains on average a **62 % performance increase over
  Behavior Cloning**."* ⚠️ Robot grasping, not driving; and **it requires collecting with noise —
  it cannot be retro-fitted to a logged corpus.** Not applicable to PhysicalAI.
- Codevilla's CARLA collection used the same idea: **PUBLISHED-SECONDARY** (read from ar5iv, not
  re-verified in the PDF) — steering perturbations for 20 % of training time, p≈0.1/s, duration
  0.5–2 s, triangular impulse, intensity 0.15. Treat as illustrative only.
- Wang et al. MPV (`1905.06937`), **PUBLISHED (primary, verified)**: *"Inspired by DAgger, we
  randomly add noise to the expert's action every 30 s to augment the data with examples of error
  recovery. **The noisy control actions along the following seven frames are removed when saving
  the data for training to avoid imitating the noisy behavior.**"* — note the second half; it is
  the detail people get wrong.
- **Neural rendering / 3DGS is the modern way to do §2b on real logs.** NeuRAD (`2311.15260`,
  banked) reconstructs real AV logs and does novel-view synthesis, and names training-data
  augmentation as a use. **I did NOT find a primary measuring a closed-loop policy gain from
  NVS-based viewpoint augmentation** within budget. **UNKNOWN.**

### 2e. Verdict on axis 2

**Does the field consider this mandatory for closed-loop competence? YES — and it is the only
intervention in this entire literature review with a categorical measured result (0 % → 100 %).**

**Verdict for us: WEAK BY DESIGN, and this is the finding of the axis.** If the measurement stream
confirms we have no viewpoint/trajectory augmentation, then our closed-loop collapse is the
*expected* outcome of the training distribution we built, is predicted by a 2011 theorem, and was
measured to be uncloseable by data volume in 2018. **Cost to change:** the PilotNet homography is
a ~1–2 day implementation on data we already load, needs zero new GPU for the warp itself, and can
be validated on the banked window dumps before any training run. That is an unusually cheap fix
for a first-order defect.

---

## 3. PHOTOMETRIC / EXPOSURE HANDLING AND NORMALISATION

**What the field does — PUBLISHED (source code, verbatim), from §1b:** **fixed dataset-level
constants, always.** Nobody in BEVFormer/UniAD/VAD normalises per-frame or per-sequence.
BEVFormer/UniAD subtract a fixed BGR mean with `std=1.0` (mean-subtraction only, 0–255 range);
VAD uses fixed ImageNet mean **and** std with `to_rgb=True`. The constants are inherited from the
pretrained backbone in all three cases.

⭐ **This is the sub-question where "from-scratch" actually changes the answer.** The convention
those three follow is *"use the constants your pretrained encoder was trained with"*. **We have no
pretrained encoder in the main line, so that convention supplies no constant at all** — the correct
choice for us is our **own corpus statistics**, and there is a specific reason to compute them
carefully (see the padded-pixel item in §8).

**Is per-frame normalisation harmful to a world model?** No primary found either way; treating this
as **UNKNOWN**, but with a strong structural argument: a world model's job includes predicting
*brightness change* (entering a tunnel, cresting into sun glare). **Per-frame normalisation removes
exactly the signal that distinguishes "the scene got darker" from "nothing happened", and it does
so INCONSISTENTLY ACROSS THE 3 FRAMES OF OUR STACK** — which is the CVRL failure mode (§1c) arriving
through the normalisation door instead of the augmentation door. **HYPOTHESIS**, but it is a cheap
and very high-value thing to check.

**Does the field measure photometric robustness? Yes, and camera models fail badly.**
**PUBLISHED (partly primary-verified, `2303.11040`).** nuScenes-C, 27 corruption types.
Camera-only mAP under corruption (PUBLISHED-SECONDARY: these specific cells came from the ar5iv
render, not the banked PDF — **re-verify before quoting in a report**):

| | clean | Snow | Fog | Motion blur |
|---|---|---|---|---|
| FCOS3D | 23.86 | 2.01 | 13.53 | 10.19 |
| PGD | 23.19 | 2.30 | 12.83 | 9.64 |
| DETR3D | 34.71 | 5.08 | 27.89 | 11.06 |
| BEVFormer | 41.65 | 5.73 | 32.76 | 19.79 |

Primary-verified conclusion, verbatim: *"Camera-only models are more easily affected by common
corruptions, demonstrating the indispensability of LiDAR point clouds for reliable 3D detection or
the necessity of developing more robust camera-only models."* Note their snow/rain simulation
itself is photometric: *"we add a 30 %-opacity gray mask layer, and reduce the brightness by 30 %"*
— and that alone takes BEVFormer from 41.65 to 5.73.

⭐ **The camera-specific numbers — PUBLISHED (primary, verified from banked PDF `2304.06719`,
RoboBEV, Table 11 p.20).** BEVFormer-base on nuScenes-C, per-corruption, absolute:

| corruption | NDS ↑ | mAP ↑ | mAVE ↓ | Δ NDS |
|---|---|---|---|---|
| **Clean** | 0.5174 | 0.4164 | 0.3941 | — |
| Brightness | 0.4184 | 0.3312 | 0.7686 | **−19.1 %** |
| Fog | 0.4069 | 0.3141 | 0.7798 | −21.4 % |
| Color Quant | 0.3509 | 0.2393 | 0.8079 | −32.2 % |
| Camera Crash | 0.3154 | 0.1545 | 0.7865 | −39.0 % |
| **Frame Lost** | **0.3017** | **0.1307** | 0.7364 | **−41.7 %** (mAP −68.6 %) |
| Motion Blur | 0.2695 | 0.1531 | 0.9334 | −47.9 % |
| **Low Light** | **0.2515** | **0.1394** | **1.0322** | **−51.4 %** (mAP −66.5 %) |
| Snow | 0.1857 | 0.0739 | 1.0880 | −64.1 % |

⭐ **Low light is the second-worst corruption in the whole suite (−51.4 % NDS), and plain
brightness still costs −19.1 %.** ⚠️ And note which family collapses: **mAVE goes 0.3941 → 1.0322
under low light (×2.6)** — the *longitudinal* family, the one CLAUDE.md records as **88.7 % of our
oracle gap**. Photometric fragility does not show up as a uniform degradation; it lands
disproportionately on velocity.

**Verdict: normalisation choice — UNKNOWN for us until measured, but there is no field convention
we are violating (there is no convention for from-scratch), and the from-scratch case argues for
our own corpus statistics over ImageNet's. Photometric robustness — the field measures it and
camera-only models are catastrophically weak (−51.4 % NDS in low light); we have no evidence either
way about ours. BEHIND on measurement, not necessarily on design.**

---

## 4. ⭐ EGO-MOTION COMPENSATION IN TEMPORAL FUSION — THE SECOND-BIGGEST FINDING

**PUBLISHED (primary, verified from banked PDF `2203.17054`, Table 3 p.7 + §4.3.1 text).**
BEVDet4D, nuScenes val, built up config by config from the single-frame BEVDet-Tiny baseline:

| cfg | Align | Target | Extra BEV enc | vel-weight | time-aug | mAP↑ | NDS↑ | mATE↓ | mAVE↓ |
|---|---|---|---|---|---|---|---|---|---|
| BEVDet (single frame) | – | speed | – | 0.2 | – | 0.312 | 0.392 | 0.691 | 0.909 |
| **A — concat prev frame, NO alignment** | **–** | speed | – | 0.2 | – | **0.296** | **0.376** | **0.711** | **1.544** |
| B | T | speed | – | 0.2 | – | 0.321 | 0.393 | 0.672 | 1.186 |
| C | T | offset | – | 0.2 | – | 0.320 | 0.440 | 0.697 | 0.479 |
| D | T | offset | ✓ | 0.2 | – | 0.323 | 0.449 | 0.680 | 0.479 |
| E | T | offset | ✓ | 1.0 | – | 0.322 | 0.452 | 0.685 | 0.435 |
| F | **R&T** | offset | ✓ | 1.0 | – | 0.321 | 0.461 | 0.681 | 0.376 |
| G | R&T | offset | ✓ | 1.0 | **✓** | **0.340** | **0.481** | 0.660 | **0.328** |

⭐⭐ **The headline, verbatim from §4.3.1:** *"Directly concatenate the current frame feature with
the previous one in configuration Tab. 3 (A), the overall performance **drops from 39.2 % NDS to
37.6 % NDS by −1.6 %**. This modification degrades the models' performance, especially on the
translation and the velocity aspects. We conjecture that, **due to the ego-motion, the positional
shift of the same static object between the two candidate features will confuse the following
modules' judgment on the object position.**"*

Read that again: **adding a second temporal frame WITHOUT ego-motion compensation was worse than
having no temporal input at all** — mAP −1.6, NDS −1.6, and velocity error **+70 % worse**
(0.909 → 1.544).

The rest of the ladder quantifies the compensation itself:
- **Translation-only alignment (A→B): mATE −5.4 % to 0.672, mAVE −23.2 % (1.544 → 1.186).**
- **Adding rotation to the alignment (E→F): mAVE −13.6 % (0.435 → 0.376), NDS +0.9.** Verbatim:
  *"a precise align operation can help increase the precision of velocity prediction."*
- Where you fuse matters enormously (their Table 5): fusing **after** the main BEV encoder collapses
  back to baseline (NDS 0.394 vs 0.453 for early fusion). ⇒ **Early fusion is right; our
  patch-embed-level stack is on the correct side of that axis.** We are only missing the alignment.
- Their Table 4 quantifies the price of the alignment's own interpolation: at 0.8 m BEV resolution,
  aligning after rather than within the view transformer costs **mAVE 0.479 → 0.499**. Small, but
  real — *warping is not free.*

**Scope, stated honestly:** BEVDet4D aligns in **BEV space**, where the warp is an exact rigid
transform of a metric grid. We would be warping in **image space**, where the correct warp needs
depth. The transferable claim is the **sign and the mechanism** (uncompensated temporal stacking
injects ego-motion as a nuisance that the model must undo, and it can make things net-negative),
**not** the magnitude.

⭐ **A second, separate finding hiding in row G — and it is ours to steal.** Row G's only change is
`Aug.` = *"the augmentation in time dimension when selecting the adjacent frame"*, and it is worth
**+2.0 NDS, +1.9 mAP, mAVE −12.8 %** — the largest single step in the back half of the table. What
it actually is (**PUBLISHED, primary, verified**): *"During the training process, we conduct data
augmentation by randomly sampling time intervals within [3T, 27T]"* where T ≈ 0.083 s, i.e.
**dt jitter over 0.25 s – 2.25 s**, with the optimal test interval found to be **≈15T ≈ 1.25 s**
(their Fig. 4 sweeps it and NDS moves over ~44–47 across the range).

⚠️⚠️ **Set that against our stack: t−200 ms, t−100 ms, t.** BEVDet4D searched the temporal aperture
explicitly, found ~1.25 s optimal for extracting motion, and measured ~3 NDS of spread across the
choice. **Our entire temporal aperture is 200 ms — an order of magnitude shorter than the setting
they measured as optimal.** Different task (they need measurable BEV parallax for velocity; we are
predicting scene content), so the number does not transfer — but **"has anyone searched our frame
spacing?" is now a first-class question with published precedent that the answer matters.**

**Verdict: WEAK BY DESIGN (uncompensated stacking), and UNKNOWN-but-suspicious (200 ms aperture,
fixed dt).** Cost to change: dt jitter is nearly free (a dataloader index change + a dt input
channel). Ego-motion compensation in image space is *not* free and needs depth or a flat-ground
approximation — but note it is **the same homography** as the PilotNet warp in §2c, so the two
items share one implementation.

---

## 5. FRAME DROPS / IRREGULAR TIMESTAMPS

**PUBLISHED (primary, verified from banked PDF `2506.05780`, Zoox, Table 1).** Production AV
company, mid-fusion perspective-view detector, F1 by class:

| exp | validation set | model | F1 Cyc | F1 Car | F1 Ped |
|---|---|---|---|---|---|
| 1a | perfectly synchronized | baseline | 32.1 | 52.8 | 28.2 |
| 1b | perfectly synchronized | + timestamp feature & staleness aug | 30.8 | 52.4 | 28.5 |
| **2a** | **camera staleness 100 ms** | **baseline** | **14.1** | **36.6** | **6.3** |
| **2b** | **camera staleness 100 ms** | **+ timestamp feature & staleness aug** | **32.1** | **52.1** | **26.8** |
| 3 | dropout camera at inference | baseline | 30.9 | 52.9 | 30.1 |

⭐ **A 100 ms camera staleness costs the synchronisation-assuming baseline −56 % / −31 % / −78 %
of F1, and is almost entirely recovered by two input-pipeline changes at ~zero cost on
synchronised data.** The two changes are exactly what the axis brief asked about:
1. **a per-point timestamp-offset feature** — i.e. **dt as an explicit input**;
2. **augmenting the staleness distribution during training** — jitter the query timestamp
   uniformly over ±0.1 s and re-fetch the nearest frame, mixing stale and clean at a ratio `P_S`.
Verbatim: *"We find that our method is robust and doesn't require exact replication of the time
difference profile on-vehicle, only reasonable t_max_J and P_S values."*

Corroborating, and much more violent (adversarial setting, so read it as an upper bound):
**PUBLISHED (primary, verified from banked PDF `2507.09095`, DEJAVU)** — MVXNet on KITTI, a
**1-frame LiDAR delay** takes car mAP from **84.1 → 9.7 (−88.5 %)**, pedestrian 87.6 → 34.2,
cyclist 73.2 → 32.9. On BEVFusion, a 1-frame *camera* delay costs −7.4 % car mAP, a 1-frame LiDAR
delay −65.5 %, and **both delayed by one frame: 88.8 → 9.6 (−89.2 %)**. Tracking (MMF-JDT): a
3-frame camera delay drops MOTA by **73 %**.
⚠️ Scope: the catastrophic numbers are cross-modal geometric misregistration. **The honest
camera-only analogue is the −7.4 %.**

⭐ **And frame drops are a benchmarked, named corruption with a camera-only number.**
**PUBLISHED (primary, verified from banked PDF `2304.06719`, RoboBEV, Table 11 + §3.2/§8.1).**
RoboBEV's eight corruptions include **Frame Lost** (*"we randomly drop each camera with an
identical probability at every single input frame"*, `p_drop = (severity − 1)/6`) and **Camera
Crash** (*"images from certain viewpoints are continuously lost"*).
**BEVFormer-base — a model that HAS temporal fusion — loses NDS 0.5174 → 0.3017 (−41.7 %) and
mAP 0.4164 → 0.1307 (−68.6 %) under Frame Lost**, and 0.3154 NDS under Camera Crash.
Verbatim: *"We find models that utilize temporal information (e.g., BEVFormer) can output the
prediction of a lost frame. However, these models still struggle to make predictions under Camera
Crash, where consecutive frames of one camera are lost."*
⇒ **Having a temporal model does not confer frame-drop robustness. It has to be trained for.**

nuScenes-C also carries **Temporal Misalignment** as a first-class corruption type — implemented
(verified, primary) as *"We keep the input of one modality the same as that at the previous
timestamp"*, with stuck-frame severities {2,4,6,8,10}.

**What breaks if dt is assumed constant but is not:** the model learns a fixed
pixels-per-frame → metres-per-second mapping. When dt changes, the same visual displacement means
a different speed, and there is no input that tells it so. That is a *silent* error: it does not
crash, it biases the longitudinal family — the family that CLAUDE.md already records as
**88.7 % of our oracle gap**.

**Verdict: WEAK BY DESIGN if we assume constant dt (to confirm).** Cost to change: **dt as an
input channel + dt jitter in the sampler is one of the cheapest interventions in this whole
review**, and Zoox measured it as free on clean data.

---

## 6. LABEL-FRAME ALIGNMENT / TEMPORAL OFFSET

No primary directly measures "the cost of a one-frame label misalignment in E2E driving" — I
probed three ways (temporal-misalignment ∧ driving; the corruption benchmarks; the sensor-staleness
literature) and the closest evidence is §5, which is about *sensor-to-sensor* rather than
*sensor-to-label* offset. **UNKNOWN as a directly-measured quantity.** But three primaries make it
a first-class risk rather than a theoretical one:

1. **Zoox states the convention explicitly because it is not obvious (PUBLISHED, primary,
   verified):** *"The camera timestamp T_C represents **when the first line stops exposing**"*
   (rolling shutter, 5–15 ms exposure, 25 µs row time), and the LiDAR sweep timestamp is defined as
   *"the 'end' of each sweep (the timestamp of the latest LiDAR point)"*. ⇒ **A production stack
   writes down which instant its timestamp names. If ours does not, the convention is undefined,
   and an undefined convention is one that two pieces of code will disagree about.**
2. **nuScenes-C's "Motion Compensation" corruption (PUBLISHED, primary, verified)** is literally
   *"We add Gaussian noises to the rotation and translation matrices of the vehicle's ego pose"*,
   severities {0.02…0.10} rotation / {0.002…0.010} translation — and it is among the most damaging
   corruptions in the taxonomy (TransFusion falls to 9.01 % mAP under it). ⇒ **Ego-pose error is a
   measured first-order failure driver.**
3. **The magnitude argument (ESTIMATED, our own arithmetic):** one frame = 100 ms. At 10 m/s that
   is **1.0 m of along-track offset**. Our deployed v1 fwd ADE is 0.452 m. **A one-frame label slip
   would be roughly twice our headline error.** It would also be *invisible* in training loss —
   the model simply learns a shifted mapping and fits it well.

**Related and directly on point: `causal confusion` literature (§7) is the reason a one-frame slip
is not merely additive noise** — a label that leads the frame by one step makes the *previous*
action an even better predictor of the label, i.e. it **widens the copycat shortcut**.

**Verdict: UNKNOWN — and this is the cheapest audit in the whole review.** Cost to change: zero if
correct, one line if not. See the go-check list.

---

## 7. ⭐⭐ CAUSAL CONFUSION / THE ACTION ECHO — THE MOST IMPORTANT SECTION

### 7a. The named phenomenon

**PUBLISHED (primary, verified from banked PDF `1905.11979`, de Haan/Jayaraman/Levine, NeurIPS
2019).** Abstract, verbatim: *"it leads to a counter-intuitive 'causal misidentification'
phenomenon: **access to more information can yield worse performance**."* And §3, verbatim:
*"Causal misidentification occurs commonly in natural imitation learning settings, **especially
when the imitator's inputs include history information**."*
And, critically for how we should read our own numbers: *"in Pong, CONFOUNDED produces **lower
validation loss** than ORIGINAL on held-out demonstration samples, but produces **lower rewards**
when actually used for control."*

### 7b. The driving number — history helps open-loop, halves closed-loop

**PUBLISHED (primary, verified from the banked PRIMARY `1905.06937` Table II, not from de Haan's
reproduction of it).** Wang, Devin, Cai, Krähenbühl, Darrell, MPV-Nets, IROS 2019. GTA-V,
8 unseen locations, 10 rollouts of ~800 steps each:

| view | map | **ego history** | perplexity ↓ (open-loop) | distance ↑ | interventions/100 m ↓ | collisions/100 m ↓ |
|---|---|---|---|---|---|---|
| travel | – | – | 0.989 | **268.95** | **1.30 ± 0.78** | 3.38 ± 2.55 |
| north | – | – | 0.969 | 235.85 | 1.70 ± 1.14 | **2.46 ± 2.18** |
| travel | ✓ | – | 1.010 | 218.94 | 2.99 ± 4.49 | 4.12 ± 4.42 |
| north | ✓ | – | 0.975 | 213.79 | 1.92 ± 3.37 | 3.46 ± 4.02 |
| **travel** | – | **✓** | **0.834 (best)** | **144.92** | 2.94 ± 1.79 | **6.49 ± 5.72** |
| **north** | – | **✓** | **0.853** | **96.70 (worst)** | 2.56 ± 0.98 | **6.60 ± 6.32** |

⭐ **Adding ego-motion history: open-loop perplexity improves 15.7 % (0.989 → 0.834) while
closed-loop distance falls 46 % (268.95 → 144.92) and collisions roughly double (3.38 → 6.49).**
Verbatim: *"Including the agent's history performs much better in the off-policy perplexity
evaluation but fails at on-policy driving."* The paper's own design conclusion: *"For all remaining
experiments we use a direction-of-travel oriented plan view **without a map or history**."*

### 7c. The copycat problem and its diagnostic

**PUBLISHED (primary, verified from banked PDF `2010.14876`, Wen et al., NeurIPS 2020).** Abstract:
*"the imitator learns to cheat by predicting the expert's previous action, rather than the next
action."*

⚠️ **CORRECTION TO THE BRIEF.** The brief said this paper "MEASURES that longer observation
histories make the copycat problem WORSE". That is **not what their table shows** and I will not
report it. Their Table 2 (partially-observed MuJoCo, cumulative reward): **BC-OH (H=2) BEATS
BC-SO (H=1) in five of six environments** (e.g. PO-Ant 1750 vs 1300; PO-HalfCheetah 820 vs −38).
What they actually establish is sharper and more useful:

**History introduces a shortcut whose severity is MEASURABLE and inversely correlated with
closed-loop reward, and closing the shortcut — while KEEPING the history — is what unlocks the
benefit.** PO-Hopper: BC-SO 275, BC-OH 293, **Ours 1086**, expert 1780.

⭐⭐ **And here is the number that should reframe how the programme reads its own evals.**
Their Table 4 (Hopper) reports the *open-loop* next-action test MSE, and Table 2 the closed-loop
reward, for the same three arms:

| arm | open-loop next-action MSE ×10⁻³ ↓ | closed-loop PO-Hopper reward ↑ |
|---|---|---|
| BC-OH (the copycat) | **0.89 (best)** | 293 ± 83 |
| Ours w/o IB (TCA only) | 9.48 (**10.7× worse**) | **683 ± 132** |
| Ours w/o IB & TCA | 14.62 | (worse than TCA) |
| Ours (full) | — | **1086 ± 262** |

**The arm that is 10.7× WORSE at the open-loop metric is 2.3× BETTER at the closed-loop task.**
⇒ **An open-loop score cannot rank arms once the copycat shortcut is available.** This is the
literature's independent confirmation of the programme's own EVAL_DOCTRINE T0/T1 split, and it
argues that during a de-confounding intervention the T0 number is expected to get *worse* and that
must be pre-registered as an acceptable outcome, or the intervention will be rejected by its own
gate.

⭐ **The cheap diagnostic — runnable on our banked dumps with ZERO GPU.** Their §4:
> Train a two-layer MLP to predict a_t from (a_{t−1},…,a_{t−k}), k=9. Score held-out trajectories
> from (i) the EXPERT and (ii) the LEARNED policy. **If the learner's actions are MORE predictable
> from their own past than the expert's are, you have the copycat problem.**

Their Table 1 (MSE, lower ⇒ more copycat) — every environment, learner far more predictable:

| | Ant ×10⁻² | Hopper ×10⁻³ | Humanoid ×10⁻¹ | Reacher ×10⁻⁵ | Walker2d ×10⁻² | HalfCheetah ×10⁻² |
|---|---|---|---|---|---|---|
| expert | 6.91 ± 0.21 | 8.60 ± 1.09 | 6.93 ± 0.32 | 1.46 ± 0.37 | 2.47 ± 0.07 | 9.81 ± 0.33 |
| BC-OH | 0.66 ± 0.04 | 1.07 ± 0.16 | 0.18 ± 0.01 | 0.32 ± 0.05 | 0.46 ± 0.02 | 2.97 ± 0.15 |
| ratio | 10.5× | 8.0× | **38.5×** | 4.6× | 5.4× | 3.3× |

And their Fig. 3: the action-predictability ratio is **inversely correlated with normalised
reward** across all six environments. Their Table 3 gives the leak measure directly: predicting
a_{t−1} from the embedding e_t scores 0.38×10⁻³ for BC-OH versus 5.00×10⁻³ from a_t alone —
**the BC-OH embedding carries ~13× more information about the previous action than the current
action itself does.**

### 7d. The inertia problem — and "more data makes it worse"

**PUBLISHED (primary, verified from banked PDF `1904.08980`, Codevilla et al., ICCV 2019).**
Verbatim: *"we identify a typical failure mode due to a subtle dataset bias: the inertia problem.
When the ego vehicle is stopped (e.g., at a red traffic light), the probability it stays static is
indeed overwhelming in the training data. This creates a **spurious correlation between low speed
and no acceleration**, inducing excessive stopping and difficult restarting in the imitative
policy."*
And, verbatim from the figure captions: *"Due to biases in the data, the results may get either
**saturated or worse** with increasing amounts of training data"*; *"the inertia problem becomes
**more prominent with more data**"*; *"a case of worse performance when changing the amount of
training data from 50 to 100 hours."* Their partial mitigation is a **speed-prediction auxiliary
branch**, and they are explicit it is *"not a final solution."*

⚠️ **This one is uncomfortably specific to us.** CLAUDE.md already records: *"0/881 accelerate"*,
*"no arm beats hold-v0 at cruising"*, and *"the 5-way maneuver softmax mixes LAT+LON"*. **A policy
that never accelerates and cannot beat hold-velocity is the textbook description of the inertia
problem**, and Codevilla's diagnosis — a spurious low-speed ⇒ no-acceleration correlation in the
data — is an *input-pipeline / data-distribution* diagnosis, not an architecture one.

### 7e. The three published families of fix (all input-pipeline)

1. **Remove the confounded input.** MPV's own conclusion (§7b): drop the ego history.
   ChauffeurNet's **past-motion dropout**, verbatim (`1812.03079` §4.2): *"Since the past motion
   history during training is from an expert demonstration, the net can learn to 'cheat' by just
   extrapolating from the past rather than finding the underlying causes of the behavior. **During
   closed-loop inference, this breaks down because the past history is from the net's own past
   predictions.**... we introduce a dropout on the past pose history, where **for 50 % of the
   examples, we keep only the current position** (u₀,v₀)."*
   ⚠️ **Necessary but NOT sufficient:** ChauffeurNet's M0 *already has* past-motion dropout and
   still recovers in **0 %** of the perturbation scenarios. Dropout closes the cheat; only
   perturbation teaches recovery. **You need both.**
2. **Perturb the state distribution** — §2.
3. **Remove the nuisance information adversarially** — Wen et al.'s TCA + information bottleneck,
   which keeps the history and strips only the previous-action information *not shared with* the
   target. Most surgical, most expensive, and the one that makes the open-loop metric worse.

### 7f. Verdict

**WEAK BY DESIGN, with high confidence, and it is a data/input problem not an architecture
problem.** Our r = 0.9988 between the action channel and realised motion is a *quantified* measure
of exactly the nuisance correlate these three papers name. Cost to change: past-motion dropout is
**a handful of lines in the dataloader**; the copycat diagnostic is **zero GPU on banked dumps**;
the adversarial fix is a research arm. Do the first two before the third.

---

## 8. OTHER THINGS WEAK BY DESIGN IN AN INPUT PIPELINE

### 8a. Lossy vs lossless codec — a point *against* our PNG, not for it

**PUBLISHED (primary, verified from banked PDF `1604.04004`, Dodge & Karam), verbatim:**
> *"The networks are surprisingly resilient to JPEG and JPEG2000 compression distortions. It is
> only at very low quality levels (**quality parameter less than 10** for JPEG and PSNR less than
> 30 for JPEG2000) that the performance begins to decrease."*
> *"In initial experiments, we found that the accuracy of the networks does **not significantly
> decrease between quality levels 100 to 20**."*
(They swept JPEG quality 2–20 in steps of 2, LibJPEG, on VGG16 / VGG-CNN-S / Caffe-Reference /
GoogLeNet. Their broader conclusion: *"The networks are more sensitive to changes in blur and
noise compared with compression and contrast."*)

⇒ **MEASURED: JPEG at q ≥ 20 costs essentially nothing.** So **PNG buys us no measurable
representational advantage** over a q≥20 JPEG cache, and the storage/throughput we pay for it is
justified by an *operational* property, not an accuracy one — namely the property already recorded
in CLAUDE.md as DE-C152: `v2_dataset.py:325` and `slice_v2_cache.py` **refuse to sub-frame a lossy
cache**, so only a PNG cache can be re-sliced to another geometry without a rebuild. **That is the
real and correct argument for PNG; "JPEG hurts the features" is not, and should not be used.**
⚠️ **HYPOTHESIS, untested:** a *generative/predictive* objective may be more codec-sensitive than
the classification objective Dodge & Karam measured, because it must model the artefacts rather
than be invariant to them. Nobody has measured this that I found.

### 8b. The ~9 % invalid/padded pixels on rig B — my largest independent concern

INHERITED (from the brief): rig B ends with ~9 % invalid/padded pixels after the cylindrical remap.
**HYPOTHESIS** (mine; no literature needed, this is a first-principles defect class):
- If normalisation statistics are computed over **all** pixels including pad, the mean/std are
  contaminated by a constant, **and contaminated by a DIFFERENT amount for rig A vs rig B** —
  which makes the normalisation itself a rig-identifying signal.
- After mean-subtraction, a pad value of 0 becomes a large constant negative — a very high-contrast
  synthetic edge at a **fixed image location**, which a from-scratch ViT with learned positional
  embeddings can trivially latch onto as a rig indicator, and a rig indicator correlates with
  everything else that differs between the two rigs.
- Patches that are **entirely** pad still consume tokens, still receive gradient, and (if the
  reconstruction/prediction loss covers them) are trivially predictable — which **inflates any
  pixel-space loss or PSNR-like metric by a constant that differs per rig.**

This is the same family as the `df` / cgroup / `step_s` traps: a quantity computed over the wrong
support, read as an answer.

### 8c. Derived-constant drift

CLAUDE.md already carries the `HORIZON = round(6.0 * 10.0 / STRIDE)` case. The generalisation for
this axis: **any constant in the input pipeline that is derived from another (window length,
horizon, frame spacing, dt, patch count) silently changes the experiment when its input changes**,
and a "reproduction" then compares two different experiments. Every such constant should be
**printed into the run's config.json and asserted at load**.

---

## 9. GO CHECK WHETHER TANITAD DOES X — for the measurement stream

Ordered by (expected defect severity × cheapness to check). Each is a presence/absence question
answerable from source.

**Tier 1 — a defect here would be first-order, and the check is minutes**
1. **Label/pose temporal alignment.** Which instant does the pose/action at index *t* name — the
   frame's exposure start, exposure midpoint, or the *next* control tick? Trace it from the
   PhysicalAI `egomotion` reader through `physicalai.py`'s episode build into the target tensor.
   Look for any `[1:]`, `[:-1]`, `shift`, or `roll` on either the frame index or the pose index.
   **A one-frame slip = 100 ms ≈ 1.0 m at 10 m/s ≈ 2× our headline ADE, and is invisible in loss.**
2. **Is there ANY viewpoint / trajectory perturbation augmentation?** (§2) Grep the dataloader and
   trainer for `perturb`, `jitter`, `shift`, `homography`, `warp`, `recover`. Expected answer:
   none. If none, that is the axis's headline finding.
3. **Is the action / past-motion channel ever dropped out?** (§7e) Grep for `dropout` applied to
   the *action* or *past-pose* input specifically, not to features. Expected: none.
4. **Is dt assumed constant?** Is there any dt / timestamp input channel, or any handling of a
   dropped frame, or does the sampler just take indices `[i-2, i-1, i]`? (§5)
   Sub-question: **does the PhysicalAI corpus actually have uniform 10 Hz frames, or are there
   gaps?** If the sampler indexes positionally rather than by timestamp, a real gap silently
   becomes a wrong dt. Check whether the episode build records per-frame timestamps at all — and
   if it does not, that absence is itself the finding.

**Tier 2 — cheap, and the answer changes the design**
5. **What exactly is the normalisation?** Fixed constants or per-frame/per-batch statistics? If
   fixed — computed over our corpus or copied from ImageNet? **Are the statistics computed over
   VALID pixels only, or do the ~9 % pad pixels enter them?** (§3, §8b)
6. **Is there a validity mask for the padded region?** Does it reach the loss? Does it reach
   attention? Are all-pad patches counted in any reported metric? (§8b)
7. **Is there ANY photometric augmentation?** If yes — **is it drawn once per WINDOW or once per
   FRAME?** Per-frame inside our 3-frame channel stack is the CVRL failure mode. (§1c)
8. **Is there a horizontal flip anywhere?** If yes, what happens to the lateral/steering target and
   to the nav/goal signal? Expected and desired answer: no flip. (§1d)
9. **Is the 3-frame stack ego-motion compensated?** Expected: no (raw pixel stack). Confirm, and
   confirm nothing downstream assumes it is. (§4)

**Tier 3 — measurement, not inspection**
10. **Run the copycat diagnostic (§7c). Zero GPU.** For each banked arm, fit a small MLP predicting
    a_t from (a_{t−1}…a_{t−9}) and compare held-out MSE on **our rollouts** vs **the logged expert
    trajectories**. If ours is materially lower, the copycat problem is confirmed *numerically*,
    not just by the r = 0.9988 correlation. Carry the CLAUDE.md probe hygiene: a **constant-only
    control** that must read the no-information value exactly, and print **n** and **d**.
11. **What is the actual temporal aperture, and has anyone swept it?** We stack over 200 ms.
    BEVDet4D swept 0.25 s–2.25 s and found ~1.25 s optimal with ~3 NDS of spread. (§4)
12. **Is the corpus's speed/stop distribution measured?** Codevilla's inertia problem is a
    *distributional* fact about the training data (§7d). Given our `0/881 accelerate` and
    `no arm beats hold-v0`, the histogram of (speed, acceleration) pairs in the parity corpus is
    a 20-minute, zero-GPU measurement that may explain a lot.

---

## 10. SUMMARY TABLE

| # | sub-question | what the field does | measured delta (primary) | verdict for us | cost to change |
|---|---|---|---|---|---|
| 1a | augmentation at all | universal in BEV/AV | BEVDet: **+8.6 mAP peak, −5.6 %→−0.4 % overfit gap** | BEHIND (w/ caveat: BDA half doesn't transfer) | medium |
| 1b | photometric jitter | BEVFormer/UniAD/VAD **all** apply it | no isolated ablation published | BEHIND on practice, UNKNOWN on magnitude | low |
| 1c | temporal consistency | CVRL fixes params per clip | **+11.5 top-1** (52.3→63.8) | **critical constraint** for our 9-ch stack | low |
| 1d | horizontal flip | detectors yes, **planners no** | — | **CONSISTENT (correctly absent)** | — |
| 1e | world models augment? | — | — | **UNKNOWN** | — |
| 2 | viewpoint/traj augmentation | mandatory for closed loop | **ChauffeurNet 0 % → 100 % recovery**; DAgger **T²ε vs Tε**; 30 M examples not enough | **WEAK BY DESIGN — headline** | 1–2 d (PilotNet homography) |
| 3 | normalisation / exposure | **fixed constants, never per-frame** | **BEVFormer NDS −51.4 % in low light, −19.1 % under brightness; mAVE ×2.6** | UNKNOWN for us; no convention violated | low |
| 4 | ego-motion compensation | BEV stacks warp before fusing | **uncompensated is WORSE than no 2nd frame** (NDS −1.6, mAVE +70 %); R&T align −13.6 % mAVE | **WEAK BY DESIGN** | medium–high (shares §2 impl) |
| 5 | frame drops / dt | timestamp feature + staleness aug; Frame Lost is a benchmarked corruption | **Zoox: 100 ms staleness F1 −31…−78 %, fully recovered at ~0 cost** · **RoboBEV: Frame Lost −41.7 % NDS on a temporal model** | **WEAK BY DESIGN if dt fixed** | **low — best ratio in the review** |
| 6 | label-frame alignment | production stacks define the instant explicitly | (no direct primary) 1 frame ≈ 1.0 m ≈ 2× our ADE | **UNKNOWN — cheapest audit** | ~0 |
| 7 | causal confusion / action echo | named, characterised, 3 fix families | **MPV: perplexity −15.7 % while distance −46 %, collisions ×1.9**; **Wen: 10.7× worse open-loop = 2.3× better closed-loop** | **WEAK BY DESIGN — most important** | low (dropout, diagnostic) |
| 8a | lossy codec | — | **JPEG q≥20 ≈ free** | PNG justified operationally, not by accuracy | — |
| 8b | padded pixels | — | (ours to measure) | **HYPOTHESIS — rig-identifying leak** | low |

---

## 11. BANKED PRIMARIES (all via `tools/kb_add.py --tag input-pipeline`)

`2112.11790` BEVDet · `2203.17054` BEVDet4D · `1812.03079` ChauffeurNet · `1905.11979` Causal
Confusion · `2010.14876` Copycat · `1604.07316` PilotNet · `1011.0686` DAgger · `1904.08980`
Codevilla BC limitations · `1905.06937` MPV-Nets · `2312.03031` BEV-Planner · `2305.10430` AD-MLP ·
`2008.03800` CVRL · `1604.04004` Dodge & Karam · `2507.09095` DEJAVU · `2506.05780` Zoox staleness ·
`2303.11040` nuScenes-C · `2304.06719` RoboBEV · `1703.09327` DART · `2311.15260` NeuRAD.
`python tools/kb_add.py --verify` → **237 entries, 0 orphans, 0 problems.**

### ⚠️ INSTRUMENT DEFECT FOUND IN `tools/kb_add.py` — report this

`2304.06719` first banked as a **truncated 4,194,304-byte (exactly 4 MiB) file** after a tool
timeout cut the download mid-stream. The file kept a valid `%PDF-1.5` header, so:
- `kb_add.py` printed a normal `banked …` success line;
- a re-run printed **`already banked`** and did NOT re-download;
- **`--verify` reported `0 problem(s)`** — because it re-hashes the file it banked, so a truncated
  download hashes consistently and passes;
- **PyMuPDF opened it and reported `PAGES 0`.** It was unreadable.

⇒ **`--verify` proves the bytes have not CHANGED; it does not prove the bytes are a PAPER.** This is
the same family as the already-recorded case where the tool "filed 11 papers under the query
string" and reported success. **Suggested fix:** after download, assert `fitz.open(path)` yields
`len(doc) > 0` and a non-trivial character count, and record the page count in `library.json` so
`--verify` can re-check it. Until then, **any agent banking a paper should open it once and confirm
a page count before citing it.** I did, which is the only reason this was caught.
(Fixed here by deleting the file and re-banking: 25,648,418 bytes, 27 pages.)
