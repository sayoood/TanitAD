# IDM v3 — Literature Review: metric scale, camera geometry, rotation, acceleration, action pseudo-labelling

**Date:** 2026-07-27
**Author:** literature-review subagent (Architecture & Inference)
**Scope:** citations only. No experiments run, no pods touched, no code modified.
**Deliverable:** this file. See §7 for the channel × method table the caller asked for.

---

## 0. How to read this document

**Evidence classes** (per `Project Steering/AGENT_OPERATING_STANDARD.md`):

| Class | Meaning here |
|---|---|
| `PUBLISHED (cited)` | I fetched the paper's own abstract/body page and verified title, authors, year, venue, arXiv id and the specific mechanism claim. |
| `PUBLISHED (metadata verified, mechanism from secondary source)` | Bibliographic record verified, but the mechanism sentence came from a search snippet / project page rather than the paper body. |
| `UNVERIFIED` | I could not confirm it from a primary source in this pass. **Do not quote it in a decision document without re-verification.** |
| `INHERITED (task brief)` | Our own measured numbers, taken from the brief that commissioned this review. **I did not re-verify them against the artifact.** |

**Our numbers as given to me** — `INHERITED (task brief), n=4195 held-out windows`:
speed pooled R² +0.865 with 56.6 % of MSE a per-clip level bias, gain 0.830 (≈17 % shrinkage toward training mean);
yaw-rate pooled R² +0.105 (PhysicalAI +0.904, comma2k19 +0.011, comma traced to corrupt labels);
long_accel R² −0.240, frozen-latent linear probe −0.095 on clean labels;
steer R² +0.742; 0.86 M > 2.90 M > 19.98 M in parameter count vs quality; ridge probe ≈ trained transformer on all four channels; model is never told intrinsics, extrinsics, focal length, camera height or FOV.

**The one-line orientation.** The literature splits our four channels along exactly the line classical geometry predicts: **rotation is metrically observable from a calibrated monocular image pair; translation is not** (§4). Our yaw-rate failure is therefore *not* the same kind of problem as our speed failure — yaw-rate on PhysicalAI already works (+0.904) and only fails where labels are corrupt, whereas speed is failing for a reason the field has an entire sub-literature about and has *solved* in the closest analogue (metric monocular depth across unknown cameras, §2). Treat them as two separate work items.

---

## 1. Scale resolution in monocular ego-motion / visual odometry

### 1.0 Why scale is unobservable at all

**Nistér, D. (2004). "An Efficient Solution to the Five-Point Relative Pose Problem." *IEEE TPAMI* 26(6):756–770.**
`PUBLISHED (metadata verified: TPAMI 26(6) 756–770, 2004; mechanism from secondary source)` — no arXiv id (pre-dates arXiv adoption in this area). The five-point algorithm recovers the essential matrix from five correspondences in two *calibrated* views, decomposing to a rotation **R** and a translation **direction** **t̂**; the magnitude ‖t‖ is not in the essential matrix at all. This is the formal statement that a calibrated two-view monocular measurement fixes rotation completely and translation only up to a positive scalar.
**Relevance to TanitAD IDM:** this is the *reason* our speed channel hedges and our yaw channel does not. No amount of capacity fixes an unobservable quantity — consistent with our finding that 19.98 M is worse than 0.86 M. The 0.830 gain is what an MMSE regressor does when the target is genuinely underdetermined by the input.

**Scaramuzza, D. & Fraundorfer, F. (2011). "Visual Odometry Part I: The First 30 Years and Fundamentals." *IEEE Robotics & Automation Magazine* 18(4):80–92.**
`PUBLISHED (metadata verified: IEEE RAM 18(4) 80–92, Dec 2011; mechanism from secondary source)` — the canonical tutorial. Establishes the standard taxonomy of scale recovery in monocular VO and notes that nonholonomic vehicle constraints permit absolute scale recovery when the vehicle turns.
**Relevance to TanitAD IDM:** the correct first citation when we have to justify to a reviewer why a monocular IDM cannot be expected to read metres without a prior. Part II (matching/robustness) is a companion article.

**Longuet-Higgins, H. C. & Prazdny, K. (1980). "The interpretation of a moving retinal image." *Proc. R. Soc. Lond. B* 208:385–397. DOI 10.1098/rspb.1980.0057.**
`PUBLISHED (metadata verified; mechanism partly from secondary source)` — the classical decomposition of the optic-flow field into a rotational component and a translational component, where the **rotational component is independent of scene depth** and the translational component is scaled by inverse depth. (The abstract as fetched emphasises the surface-gradient/shear result; the depth-independence of the rotational term is the standard reading of this decomposition — `UNVERIFIED` that this exact phrasing appears in the paper body.)
**Relevance to TanitAD IDM:** the mechanistic reason yaw-rate is learnable from pixels and speed is not. Yaw-rate maps to a depth-free, purely angular flow component; speed maps to a term that is multiplied by unknown 1/Z. This is the single best sentence to put in the v3 design doc.

### 1.1 Known camera height / ground-plane constraint — the dominant driving-specific fix

**Song, S. & Chandraker, M. (2014). "Robust Scale Estimation in Real-Time Monocular SFM for Autonomous Driving." *CVPR 2014*, pp. 1566–1573.**
`PUBLISHED (metadata verified via dblp + CVF open access; mechanism from CVF abstract)` — corrects scale drift with a cue-combination framework for ground-plane estimation, fusing sparse features, dense inter-frame stereo and object detection; reported accuracy comparable to stereo over long driving sequences.
**Relevance to TanitAD IDM:** the reference implementation of "known camera height ⇒ metric scale" for exactly our domain (forward camera on a car). Its cue-fusion idea maps onto our two-rig problem: the per-clip cue is different for rig A and rig B.

**Zhou, D., Dai, Y. & Li, H. (2019/2021). "Ground Plane based Absolute Scale Estimation for Monocular Visual Odometry."** arXiv:1903.00912.
`PUBLISHED (arXiv id verified by search hit; mechanism from title/snippet)` `UNVERIFIED` on venue and author list — I saw the ar5iv listing but did not fetch the abstract page.
**Relevance to TanitAD IDM:** listed for completeness of the ground-plane family; verify before quoting.

**Wang, R. et al. (2021). "Accurate and Robust Scale Recovery for Monocular Visual Odometry Based on Plane Geometry." *ICRA 2021*.** arXiv:2101.05995.
`PUBLISHED (arXiv id + ICRA 2021 verified; mechanism from abstract snippet)` `UNVERIFIED` on full author list. Assumes a **constant camera height above the ground**, extracts high-quality ground points, aggregates them over a local sliding window, and solves a RANSAC least-squares problem for the scale. Reports SOTA translation error on KITTI while keeping rotation error competitive.
**Relevance to TanitAD IDM:** the cleanest modern statement of the recipe. Note the reported asymmetry — the method moves *translation* error and leaves *rotation* essentially alone, exactly the split we see.

**Wagstaff, B. & Kelly, J. (2021). "Self-Supervised Scale Recovery for Monocular Depth and Egomotion Estimation." *IROS 2021*.** arXiv:2009.03787.
`PUBLISHED (cited)` — introduces a scale-recovery **loss** that enforces consistency between a *known* camera height and the camera height implied by the network's own depth prediction, converting up-to-scale depth+egomotion into metric depth+egomotion.
**Relevance to TanitAD IDM:** this is the version that is a *loss term*, not a post-hoc geometric solve — the form that would drop into our training loop most easily, **if** we had per-clip camera height. We currently do not feed it.

**Xue, F., Zhuo, G., Huang, Z., Fu, W., Wu, Z. & Ang, M. H. (2020). "Toward Hierarchical Self-Supervised Monocular Absolute Depth Estimation for Autonomous Driving Applications." *IROS 2020*, pp. 2330–2337 (code: TJ-IPLab/DNet).**
`PUBLISHED (metadata verified via IROS page range + official repo; mechanism from repo/secondary)` `UNVERIFIED` on exact author-order. Uses dense geometrical constraints and camera height to obtain **absolute** (metric) depth self-supervised.
**Relevance to TanitAD IDM:** same family; "DNet" is the commonly cited shorthand.

### 1.2 Velocity / speed priors as supervision — the closest match to what we can actually do

**Guizilini, V., Ambrus, R., Pillai, S., Raventos, A. & Gaidon, A. (2020). "3D Packing for Self-Supervised Monocular Depth Estimation" (PackNet-SfM). *CVPR 2020* (oral).** arXiv:1905.02693.
`PUBLISHED (cited)` for bibliographic record and for the existence of a velocity-supervised variant — the paper's own supplementary reports an **"M+v"** (monocular + velocity supervision) setting and states the method "is able to recover metrically accurate scale purely from monocular imagery," in contrast to methods that "require ground-truth scaling at test-time." The mechanism, per the project's own documentation, is a **weak velocity supervision loss on the magnitude of the pose network's predicted translation**, using instantaneous vehicle velocity available at training time only; no velocity or GT depth is needed at test time. `UNVERIFIED`: exact loss formula and the numeric scale-aware depth results — the CVF PDF returned HTTP 403 to my fetcher.
**Relevance to TanitAD IDM: this is the single most transferable result in this document.** It says: if you have a speed signal *at training time*, you can put a loss on translation magnitude and the network becomes metric — without needing it at inference. We *have* per-clip speed labels (that is what we are regressing). The lesson is not "add velocity supervision" (we already regress speed) but the converse: **PackNet only got metric scale because velocity was supervised; it did not emerge.** Nothing in our setup gives the network the extra information PackNet's velocity loss injects — the loss alone cannot create observability, so PackNet works because the *training distribution* of camera geometry is fixed. On our two-rig + comma corpus it is not.

**Li, R., Wang, S., Long, Z. & Gu, D. (2018). "UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning." *ICRA 2018*.** arXiv:1709.06841.
`PUBLISHED (cited)` — trains with **stereo pairs** (whose known baseline supplies metric scale) and tests monocular. Explicitly lists "absolute scale recovery" as one of its two salient features.
**Relevance to TanitAD IDM:** the "learn the scale from a metric training signal, deploy monocular" template. Our comma2k19 and PhysicalAI corpora both have metric ego-state, so this template is available — but see §1.3 for why the learned scale does not transfer across rigs.

**Yang, N., von Stumberg, L., Wang, R. & Cremers, D. (2020). "D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry." *CVPR 2020*.** arXiv:2003.01060.
`PUBLISHED (cited)` for record; `UNVERIFIED` for the metric-scale mechanism — the abstract describes self-supervised depth trained **on stereo videos** plus predictive brightness transformation and learned photometric uncertainty, but does not state the scale mechanism explicitly. The standard reading is that stereo training gives the depth network metric scale which then scales the VO.
**Relevance to TanitAD IDM:** note the **learned per-pixel uncertainty** used to weight residuals. That is directly applicable to our shrinkage problem: a heteroscedastic head lets the model say "I cannot see the metre here" instead of silently regressing to the mean.

### 1.3 Learned scale and its failure mode

**Guizilini, V., Vasiljevic, I., Chen, D., Ambrus, R. & Gaidon, A. (2023). "Towards Zero-Shot Scale-Aware Monocular Depth Estimation" (ZeroDepth). *arXiv:2306.17253*.**
`PUBLISHED (cited)` — abstract states the problem in the exact terms of our failure: *"Monocular depth estimation is scale-ambiguous, and thus requires scale supervision to produce metric predictions. Even so, the resulting models will be geometry-specific, with learned scales that cannot be directly transferred across domains."* Fixes it with (i) **input-level geometric embeddings** that let the network learn a scale prior over objects, and (ii) encoder/decoder decoupling via a variational latent conditioned on single-frame information.
**Relevance to TanitAD IDM: this is the diagnosis of our 0.830 gain, stated by someone else, in print.** We trained a scale on a mixture of two PhysicalAI rigs and comma2k19; "learned scales cannot be directly transferred across domains" is precisely the per-clip level bias we measured. ZeroDepth's fix — inject camera geometry at the *input* level — is our §2 recommendation.

**Bian, J.-W., Li, Z., Wang, N., Zhan, H., Shen, C., Cheng, M.-M. & Reid, I. (2019). "Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video" (SC-SfMLearner). *NeurIPS 2019*.** arXiv:1908.10553.
`PUBLISHED (cited)` — identifies that networks "output scale-inconsistent results over different samples, i.e. the ego-motion network cannot provide full camera trajectories over a long video sequence because of the **per-frame scale ambiguity**," and fixes it with a **geometry consistency loss** that forces depth predictions of adjacent frames to agree, yielding globally scale-*consistent* (not scale-*correct*) trajectories.
**Relevance to TanitAD IDM:** important distinction for us. SC-SfMLearner buys **consistency**, not **metricity** — it removes the per-*frame* jitter but not the per-*clip* level offset. Our 56.6 %-of-MSE failure is a per-**clip** level bias, so a consistency loss alone would not fix it. Worth stating explicitly in the v3 design so nobody proposes it as the answer.

### 1.4 Scale drift and its correction

**Strasdat, H., Montiel, J. M. M. & Davison, A. J. (2010). "Scale Drift-Aware Large Scale Monocular SLAM." *Robotics: Science and Systems VI*, Zaragoza.**
`PUBLISHED (metadata verified via dblp + author PDF; mechanism from secondary source)` — the paper that made pose-graph optimisation over **Sim(3)** (similarity transforms, i.e. rotation + translation + *scale*) the standard way to absorb scale drift at loop closure; adopted by essentially every later monocular SLAM system.
**Relevance to TanitAD IDM:** mostly a *negative* relevance and worth saying so. Sim(3) loop closure needs a loop; we label short independent clips offline. It tells us the field's own answer to drift is unavailable to us, which strengthens the case for the geometry-conditioning route instead.

### 1.5 IMU / wheel-odometry fusion — the "just measure it" answer

**Martinelli, A. (2012). "Vision and IMU Data Fusion: Closed-Form Solutions for Attitude, Speed, Absolute Scale, and Bias Determination." *IEEE Transactions on Robotics* 28(1):44–60.**
`PUBLISHED (metadata verified: T-RO 28(1) 44–60, 2012; mechanism from secondary source)` — analytically derives the **observable modes** of a monocular-camera + 3-axis-accelerometer + 3-axis-gyro assembly over a short time interval, showing that **absolute scale, speed and attitude become observable** (and biases determinable) once inertial data is added. There is a companion IJCV article, "Closed-Form Solution of Visual-Inertial Structure from Motion."
**Relevance to TanitAD IDM: the formal proof that our missing information is exactly what an IMU supplies.** If a clip carries IMU, speed and long_accel stop being ill-posed. comma2k19 ships a 9-axis IMU and raw GNSS (§3, Schafer et al.); PhysicalAI-AV ships `egomotion`. This is the highest-leverage *label-side* fix — not a modelling fix.

**Qin, T., Li, P. & Shen, S. (2018). "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator." *IEEE T-RO*.** arXiv:1708.03852.
`PUBLISHED (cited)` — opens by stating a camera + low-cost IMU "forms the **minimum sensor suite for metric** six-DoF state estimation," and that the lack of direct distance measurement makes initialisation the hard part; solves it with a robust initialisation procedure plus tightly-coupled optimisation over pre-integrated IMU measurements.
**Relevance to TanitAD IDM:** the practical citation for "camera alone is below the minimum sensor suite for metric state." Useful when justifying why an image-only speed regressor has a floor.

### 1.6 Nonholonomic constraints — scale from turning

**Scaramuzza, D., Fraundorfer, F., Pollefeys, M. & Siegwart, R. (2009). "Absolute Scale in Structure from Motion from a Single Vehicle Mounted Camera by Exploiting Nonholonomic Constraints." *ICCV 2009*, pp. 1413–1419. DOI 10.1109/ICCV.2009.5459294.** (Journal extension: "1-Point-RANSAC Structure from Motion for Vehicle-Mounted Cameras by Exploiting Non-holonomic Constraints," *IJCV* 2011.)
`PUBLISHED (metadata verified; mechanism from secondary source)` — for a camera mounted **off the vehicle's instantaneous centre of rotation** on a wheeled (nonholonomic, locally circular-arc) vehicle, absolute scale can be computed automatically whenever the vehicle turns, without knowing baselines or scene dimensions.
**Relevance to TanitAD IDM:** a genuinely interesting free lunch for us, with a sharp caveat. It requires knowing the camera's **offset from the rear axle** — i.e. extrinsics, which we do not feed — and it degenerates on straight driving, which is most of comma2k19's 280-highway corpus. Worth knowing exists; not a first implementation.

---

## 2. Intrinsics-aware / camera-conditioned networks

> **This section is the closest solved analogue to our problem.** Metric monocular depth across unknown cameras is formally the same ill-posedness as our speed channel — a per-image metre that the pixels do not determine — and the field went from "impossible" to "zero-shot solved" between 2019 and 2024 by one move: **tell the network the camera.**

### 2.1 Coordinate-embedding conditioning (concatenate a camera-derived tensor)

**Facil, J. M., Ummenhofer, B., Zhou, H., Montesano, L., Brox, T. & Civera, J. (2019). "CAM-Convs: Camera-Aware Multi-Scale Convolutions for Single-View Depth." *CVPR 2019*.** arXiv:1904.02028.
`PUBLISHED (cited)` — opens with the exact statement of our disease: *"Single-view depth estimation suffers from the problem that a network trained on images from one camera does not generalize to images taken with a different camera model."* Fix: a convolution that takes the camera parameters into account, so the network can learn calibration-specific patterns. `UNVERIFIED`: the specific channel construction (centred-coordinate maps normalised by focal length, plus a FOV map, concatenated at multiple scales) — the abstract page did not carry the mechanism detail.
**Relevance to TanitAD IDM:** the canonical citation for "concatenate camera-derived coordinate channels." Cheapest possible retrofit to our IDM if we ever go back to pixels — but see the caveat in §7: we consume **frozen latents**, and CAM-Convs conditions the *encoder*.

**Wang, W., Hu, Y. & Scherer, S. (2020). "TartanVO: A Generalizable Learning-based VO." *CoRL 2020*.** arXiv:2011.00359.
`PUBLISHED (cited)` — **the caller asked me to get this right, so here it is in full.** Two mechanisms:
- **Up-to-scale loss.** The paper states: *"Knowing that the scale ambiguity only affects the translation **T**, we design a new loss function for **T** and keep the loss for rotation **R** unchanged."* The loss normalises predicted and ground-truth translation to unit vectors independently, then compares directions, while rotation is compared with an un-normalised L2:
  `L_norm = ‖ T̂/max(‖T̂‖,ε) − T/max(‖T‖,ε) ‖ + ‖ R̂ − R ‖`
  The rationale given is that models trained at a fixed camera height learn **spurious scale cues** that fail on a different setup.
- **Intrinsics Layer (IL).** A tensor `K_c ∈ ℝ^{2×H×W}` is **concatenated to the optical flow** before the pose network, built from the pixel index grids `X_ind, Y_ind` as
  `K_x = (X_ind − o_x)/f_x`, `K_y = (Y_ind − o_y)/f_y`.
  i.e. normalised bearing coordinates. Trained only on synthetic TartanAir, a single TartanVO model transfers to KITTI and EuRoC with no fine-tuning.
**Relevance to TanitAD IDM: this is the most directly copyable paper in the review.** It (a) states our exact rotation/translation asymmetry as a design principle, (b) gives a two-channel intrinsics tensor that is trivial to compute from PhysicalAI's `camera_intrinsics` and comma's f≈910, and (c) warns in so many words that **a fixed-camera-height training set teaches a spurious scale cue** — which is our two-rig, two-corpus situation. The `(u−cx)/fx` normalisation is also the correct per-rig fix for the cy≈543 vs cy≈755 rig split recorded in program memory.

### 2.2 Canonical camera space / focal-length normalisation (warp the data into one virtual camera)

**Yin, W., Zhang, C., Chen, H., Cai, Z., Yu, G., Wang, K., Chen, X. & Shen, C. (2023). "Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image." *ICCV 2023*.** arXiv:2307.10984.
`PUBLISHED (cited)` — abstract: SOTA metric depth methods *"can only handle a single camera model and are unable to perform mixed-data training due to the metric ambiguity,"* while zero-shot methods learn affine-invariant depth and *"cannot recover real-world metrics."* Fix: a **canonical camera space transformation module** that resolves the ambiguity and plugs into existing models; enables stable training over 8 M+ images from thousands of camera models.
The transformation itself (`PUBLISHED (mechanism from the project's own documentation / secondary source)`): pick a canonical focal `f_c`; then either
- **CSTM_label** — rescale the depth label by `ω_d = f_c/f`, i.e. `D_c = ω_d · D*`, and undo at inference `D = D_c/ω_d`; or
- **CSTM_image** — resize the image by `ω_r = f_c/f`.
General form: `Ẑ = f̂_x · Z / f_x`.
**Relevance to TanitAD IDM: the arithmetic here transfers to our speed channel almost verbatim.** For a translating camera, the observed image motion scales as `f · v / Z`. Two clips with the same `v` but different `f` (PhysicalAI rig A/B vs comma's 910 px) produce different flow magnitudes, and a geometry-blind regressor must average over them — which is a *manufactured* per-clip level bias on top of the genuine one. **The `f_c/f` rescaling is a one-line normalisation we can apply to our targets or our window, and it costs nothing.** Cheapest scale experiment available to us.

**Hu, M., Yin, W., Zhang, C., Cai, Z., Long, X., Wang, K., Chen, H., Yu, G., Shen, C. & Shen, S. (2024). "Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation." *IEEE TPAMI* 46(12):10579–10596.** arXiv:2404.15506.
`PUBLISHED (cited)` — restates and scales the result: *"the key to a zero-shot single-view model lies in resolving the metric ambiguity from various camera models and large-scale data training"*; same canonical camera space transformation module, now trained over 16 M images from thousands of camera models.
**Relevance to TanitAD IDM:** the strongest single sentence to justify the v3 architecture decision. Two independent ingredients — resolve camera ambiguity **and** train at scale. We have neither; we should buy the cheap one (camera conditioning) before the expensive one (more data).

**Guo, Y., Garg, S., Miangoleh, S. M. H., Huang, X. & Ren, L. (2025). "Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera." *CVPR 2025*.** arXiv:2501.02464.
`PUBLISHED (cited)` — projects everything into a unified **Equi-Rectangular Projection** representation with pitch-aware conversion, FOV alignment and multi-resolution augmentation; trained only on perspective images, generalises to fisheye and 360°, up to 50 % improvement on large-FOV datasets.
**Relevance to TanitAD IDM:** the "resample into one canonical geometry" alternative to "condition on the geometry." Relevant if we ever ingest a fisheye or a Cosmos f-theta clip (program memory notes cosmos f-theta calibration was unrecoverable for BEV overlays).

### 2.3 Predict the camera instead of being told it (self-prompting / single-image calibration)

**Piccinelli, L., Yang, Y.-H., Sakaridis, C., Segu, M., Li, S., Van Gool, L. & Yu, F. (2024). "UniDepth: Universal Monocular Metric Depth Estimation." *CVPR 2024*.** arXiv:2403.18913.
`PUBLISHED (cited)` — predicts metric 3D points *"at inference time without any additional information"* using a **self-promptable camera module** that predicts a dense camera representation to condition the depth features, a **pseudo-spherical output representation** that disentangles camera from depth, and a **geometric invariance loss** on camera-prompted depth features.
**Relevance to TanitAD IDM:** the fallback when intrinsics are unavailable — which is exactly the YouTube/OpenDV expansion the program has been scoping. Predict the camera, condition on the prediction. Note the *disentangling* idea: separate "what the camera is" from "what the motion is," which is the structure our IDM currently lacks entirely.

**Bochkovskii, A., Delaunoy, A., Germain, H., Santos, M., Zhou, Y., Richter, S. R. & Koltun, V. (2025). "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second." *ICLR 2025*.** arXiv:2410.02073.
`PUBLISHED (cited)` — metric depth *"with absolute scale, without relying on the availability of metadata such as camera intrinsics,"* explicitly claiming **state-of-the-art focal length estimation from a single image** as an enabling contribution.
**Relevance to TanitAD IDM:** if we need per-clip focal for unlabelled video, this is an off-the-shelf estimator. Pairs with GeoCalib below.

**Veicht, A., Sarlin, P.-E., Lindenberger, P. & Pollefeys, M. (2024). "GeoCalib: Learning Single-image Calibration with Geometric Optimization." *ECCV 2024*.** arXiv:2409.06704.
`PUBLISHED (cited)` — estimates intrinsics (focal length) **and** extrinsics (gravity direction) from one image by embedding a geometric optimisation inside the network; the internal optimisation yields **uncertainties** that flag failure cases.
**Relevance to TanitAD IDM:** already on the program's radar — there is a `2026-07-25-geocalib` incoming directory. The gravity direction is the piece we should not overlook: it gives camera **pitch/roll relative to the road**, which is half of the extrinsics that a ground-plane scale method needs, and it comes with an uncertainty we can gate on.

**Jin, L., Zhang, J., Hold-Geoffroy, Y., Wang, O., Matzen, K., Sticha, M. & Fouhey, D. F. (2023). "Perspective Fields for Single Image Camera Calibration." *CVPR 2023*.** arXiv:2212.03239.
`PUBLISHED (cited)` — represents calibration as a **per-pixel field** of an up-vector and a latitude value, making minimal camera-model assumptions and staying invariant/equivariant to cropping, warping and rotation; convertible to conventional calibration parameters.
**Relevance to TanitAD IDM:** the crop-invariance matters concretely. Program memory records that a geometric-centre crop is ~215 px wrong for PhysicalAI rig B; a per-pixel perspective field survives that crop where a scalar focal does not.

### 2.4 Geometric / ray embeddings at the input

**Vasiljevic, I., Guizilini, V., Ambrus, R., Pillai, S., Burgard, W., Shakhnarovich, G. & Gaidon, A. (2020). "Neural Ray Surfaces for Self-Supervised Learning of Depth and Ego-motion." *3DV 2020* (oral).** arXiv:2008.06630.
`PUBLISHED (metadata verified; mechanism from project page + abstract snippet)` — replaces the assumed pinhole model with a learned, fully differentiable **per-pixel projection-ray surface** (after Grossberg & Nayar's generic camera model), learnable end-to-end from raw video, generalising depth/ego-motion learning to all central cameras including fisheye and catadioptric.
**Relevance to TanitAD IDM:** the most general form of camera conditioning — learn the ray field per camera rather than parameterise it. Over-engineered for two rigs plus comma; the right reference if we ingest heterogeneous internet video.

**ZeroDepth** (Guizilini et al. 2023, arXiv:2306.17253) — see §1.3. Its "**input-level geometric embeddings**" are the ray/intrinsics-conditioning of this subsection, applied specifically to make *scale* transferable.

### 2.5 Extrinsics conditioning

**Guizilini, V., Vasiljevic, I., Ambrus, R., Shakhnarovich, G. & Gaidon, A. (2022). "Full Surround Monodepth from Multiple Cameras." *IEEE RA-L*.** arXiv:2104.00152.
`PUBLISHED (metadata verified; mechanism from project page/abstract snippet)` — first self-supervised learning of **scale-aware and scale-consistent** depth in a wide-baseline multi-camera rig, using multi-camera spatio-temporal contexts, **pose consistency constraints across cameras with known extrinsics**, and self-occlusion photometric masking; validated on DDAD and nuScenes.
**Relevance to TanitAD IDM: the clearest published case of extrinsics buying metric scale.** The known inter-camera baseline is what supplies the metre — same trick as stereo, generalised to a surround rig. PhysicalAI ships `sensor_extrinsics` (program memory: read since D-016 R1 by `physicalai.py:153-154`), so the input exists in our corpus and our IDM simply ignores it.

**Tian, S., Wulfe, B., Sargent, K., Liu, K., Zakharov, S., Guizilini, V. & Wu, J. (2024). "View-Invariant Policy Learning via Zero-Shot Novel View Synthesis." *CoRL 2024*.** arXiv:2409.03685.
`PUBLISHED (cited)` — treats **observational viewpoint** as an explicit axis of generalisation for visuomotor policies and attacks it with novel-view-synthesis data augmentation (VISTA); policies trained this way are more robust to out-of-distribution camera viewpoints in sim and real.
**Relevance to TanitAD IDM:** the *augmentation* alternative to *conditioning*. If we cannot get clean extrinsics, synthesising rig-B-like views from rig-A clips (or vice versa) is a legitimate second option. Also the honest answer to "does anyone condition a policy on extrinsics": the manipulation literature mostly **augments away** viewpoint rather than conditioning on it.

> **Gap flagged.** I did **not** find a canonical, well-cited paper that conditions a *driving policy or dynamics model* on camera **extrinsics** as an explicit input. Search surfaced candidates including "AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models" (arXiv:2603.05868) and "Learning to Act Robustly with View-Invariant Latent Actions" (arXiv:2601.02994) — **both `UNVERIFIED`; I did not fetch either abstract, and given their 2026 arXiv ids they should be checked before citing.** The strongest *verified* extrinsics-conditioning evidence in this document is FSM (above), which is a depth paper, not a policy paper.

### 2.6 Two-stage relative → metric

**Bhat, S. F., Birkl, R., Wofk, D., Wonka, P. & Müller, M. (2023). "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth." arXiv:2302.12288.**
`PUBLISHED (cited)` — pre-train for **relative** depth across twelve datasets for generalisation, then fine-tune for **metric** depth on two, with a lightweight per-domain "metric bins" head and a latent classifier that routes an input to the right head at inference.
**Relevance to TanitAD IDM: the architectural pattern that matches our situation most closely.** A shared scale-free backbone plus a small, cheap, **per-domain metric head**, with automatic domain routing. Read "domain" as "rig": one head for PhysicalAI rig A, one for rig B, one for comma. That is a *small* change to our 0.86 M model and it directly targets the per-clip level bias — and the latent classifier means we do not even need a rig label at inference.

**Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J. & Zhao, H. (2024). "Depth Anything V2." *NeurIPS 2024*.** arXiv:2406.09414.
`PUBLISHED (cited)` — synthetic-only labelled training, a scaled-up teacher, and distillation through large-scale pseudo-labelled real images; metric models are obtained by **fine-tuning the generalist with metric depth labels**. `UNVERIFIED`: any intrinsics handling — the abstract does not mention camera intrinsics at all.
**Relevance to TanitAD IDM:** note what it does *not* do. Depth Anything V2 is the strongest relative-depth model of its generation and it still needs a separate metric fine-tune per domain to produce metres. Metricity is not something scale buys you for free.

---

## 3. Self-supervised VO / depth and cross-camera generalisation — what each does about (a) scale and (b) intrinsics

**Zhou, T., Brown, M., Snavely, N. & Lowe, D. G. (2017). "Unsupervised Learning of Depth and Ego-Motion from Video" (SfMLearner). *CVPR 2017*.** arXiv:1704.07813.
`PUBLISHED (cited)` — jointly trains depth and pose networks with **view synthesis as the only supervisory signal**.
- **(a) Scale:** up to scale. Verified from the paper body: depth is evaluated after a **per-image median scaling** `ŝ = median(D_gt)/median(D_pred)`; pose is evaluated after *"first optimiz[ing] the scaling factor for the predictions made by each method to best align with the ground truth"* before computing ATE.
- **(b) Intrinsics:** **required and known.** The warp is `p_s ∼ K T̂_{t→s} D̂_t(p_t) K^{-1} p_t`, and the authors state the limitation themselves: *"our framework assumes the camera intrinsics are given, which forbids the use of random Internet videos with unknown camera types/calibration."*
**Relevance to TanitAD IDM:** two things. First, **the field's own headline numbers for monocular depth/pose are median-scaled** — a per-clip level correction applied at evaluation. Our 56.6 %-of-MSE per-clip level bias is *the same quantity the field routinely divides out*, which means (i) we should report a median-scaled/per-clip-debiased R² alongside the raw one so we are comparable, and (ii) if the downstream consumer of our pseudo-labels only needs *relative* speed, the bias may be far less harmful than the raw MSE suggests. Second, SfMLearner names our exact blocker for the YouTube expansion.

**Godard, C., Mac Aodha, O., Firman, M. & Brostow, G. (2019). "Digging Into Self-Supervised Monocular Depth Estimation" (Monodepth2). *ICCV 2019*.** arXiv:1806.01260.
`PUBLISHED (cited)` — three contributions: a **minimum reprojection loss** for occlusions, **full-resolution multi-scale sampling** to kill artefacts, and an **auto-masking loss** that discards pixels violating the camera-motion assumption.
- **(a) Scale:** up to scale (median-scaled at evaluation, following the SfMLearner protocol). `UNVERIFIED` from the abstract; standard in the KITTI eigen-split protocol.
- **(b) Intrinsics:** known/fixed KITTI intrinsics. `UNVERIFIED` from the abstract.
**Relevance to TanitAD IDM:** the **auto-mask** is the transferable idea, not the depth. It drops pixels where the static-scene assumption fails — most importantly, **frames where the camera is not moving**. That is a direct, cheap remedy for our comma2k19 standstill label corruption (§ below): mask the windows instead of fitting them.

**Wang, S., Clark, R., Wen, H. & Trigoni, N. (2017). "DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks." *ICRA 2017*, pp. 2043–2050. DOI 10.1109/ICRA.2017.7989236.**
`PUBLISHED (metadata verified via IEEE/ACM DL + Heriot-Watt portal)` — **no arXiv id found; do not cite one.** (Beware: several unrelated arXiv preprints share the "DeepVO" name. The journal extension is Wang et al., *IJRR* 2018, "End-to-end, sequence-to-sequence probabilistic visual odometry through deep neural networks.") RCNN regresses pose directly from image sequences.
- **(a) Scale:** trained supervised on KITTI ground-truth poses, so it regresses *metric* pose — but the scale is learned from a single fixed camera/vehicle configuration and there is no mechanism making it transfer. `UNVERIFIED` — mechanism inferred, not read from the paper.
- **(b) Intrinsics:** not an input. `UNVERIFIED`.
**Relevance to TanitAD IDM: DeepVO is architecturally the closest published system to our IDM — supervised, end-to-end, image-sequence → metric pose, no geometry input — and it is exactly the design that ZeroDepth/TartanVO later showed does not transfer across cameras.** Our result is the predicted one.

**Li, R. et al. (2018). UnDeepVO** — see §1.2. **(a)** metric, via stereo training. **(b)** intrinsics not conditioned.

**Zhan, H., Weerasekera, C. S., Bian, J., Reid, I. (2020). "Visual Odometry Revisited: What Should Be Learnt?" (DF-VO). *ICRA 2020*.** arXiv:1909.09803.
`PUBLISHED (cited)` — hybrid: learn single-view **depth** and two-view **optical flow** as intermediates, then run classical epipolar geometry + PnP frame-to-frame. Abstract states the system *"does not suffer from the scale-drift issue being aided by a scale consistent single-view depth CNN."*
- **(a) Scale:** resolved by the (scale-consistent) depth network, not by the pose regressor.
- **(b) Intrinsics:** required — PnP and epipolar geometry are calibrated operations.
**Relevance to TanitAD IDM:** the strongest *architectural* argument in this review for a hybrid IDM. DF-VO's finding is that the network should predict **geometric intermediates** (depth, flow) and let a calibrated solver produce the metric motion. Applied to us: predict flow/depth from latents, then solve for `v` with known `f` and camera height — instead of regressing `v` end-to-end and hoping.

**Teed, Z. & Deng, J. (2021). "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras." *NeurIPS 2021*.** arXiv:2108.10869.
`PUBLISHED (cited)` — recurrent iterative updates of camera pose and per-pixel depth through a **Dense Bundle Adjustment layer**; trained on monocular video but able to consume stereo or RGB-D at test time for better performance.
- **(a) Scale:** monocular mode is up to scale; metric scale comes from the stereo/RGB-D test-time input. The abstract does not discuss scale — `UNVERIFIED` for the precise claim, but the stereo/RGB-D sentence is the tell.
- **(b) Intrinsics:** required by the DBA layer (it optimises reprojection). `UNVERIFIED` from the abstract.
**Relevance to TanitAD IDM:** DROID-SLAM is the standard tool for generating *pseudo-ground-truth trajectories* on unlabelled video. If we expand to YouTube, this is a candidate label source — but its monocular output is up-to-scale, so it would give us **direction and yaw-rate for free and speed only up to an unknown per-clip factor.** That is a very precise statement of what a YouTube expansion can and cannot buy us.

**Teed, Z., Lipson, L. & Deng, J. (2023). "Deep Patch Visual Odometry" (DPVO). *NeurIPS 2023*.** arXiv:2208.04726.
`PUBLISHED (cited)` — sparse **patch** tracking with a recurrent update operator and differentiable bundle adjustment; outperforms prior work at ⅓ the memory and 3× the speed of the previous learned SOTA.
- **(a)/(b):** same as DROID-SLAM — up-to-scale monocular, calibrated BA. `UNVERIFIED`.
**Relevance to TanitAD IDM:** the practical (cheap) version of DROID-SLAM for bulk pseudo-labelling. Same caveat on scale.

**Wang, W., Hu, Y. & Scherer, S. (2020). TartanVO** — see §2.1. **(a)** explicitly refuses scale (up-to-scale loss); **(b)** explicitly conditions on intrinsics (IL). **This is the only system in this section that gets both design decisions right for our setting.**

**Schafer, H., Santana, E., Haden, A. & Biasini, R. (2018). "A Commute in Data: The comma2k19 Dataset." arXiv:1812.05752.**
`PUBLISHED (metadata verified via arXiv listing; contents from arXiv/GitHub description)` — 33 h / 2019 one-minute segments on 20 km of CA-280, from comma EONs carrying a road-facing camera, phone GPS, thermometers, a **9-axis IMU**, raw GNSS, and full CAN via a grey panda. Ships **pose (position + orientation) in a global frame** from a tightly-coupled INS/GNSS/Vision optimiser built on the Laika GNSS library.
**Relevance to TanitAD IDM:** two operational facts. (1) comma2k19 carries **CAN** — which means true wheel-speed and true steering angle exist in the corpus and are a better label source than differentiating a fused pose. (2) The heading we are currently deriving as `arctan2` of ENU velocity is a *derived* quantity that is undefined at standstill; the dataset's own orientation estimate is a better primary.

**comma.ai `calib_challenge` (comma.ai, GitHub, public).**
`PUBLISHED (verified from the official repo README)` — task: predict direction of travel (pitch and yaw, in camera frame, radians) per frame from dashcam video; 10 videos, 1 min each, 20 fps; **"You can estimate the focal length to be 910 pixels"**; labels generated by a neural network and confirmed with SLAM; **"errors for frames where the car speed is less than 4 m/s [are] ignored."**
**Relevance to TanitAD IDM: this independently confirms our comma yaw-rate diagnosis, from comma themselves.** comma's own benchmark **discards all frames below 4 m/s** because direction-of-travel is ill-defined at low speed. Our comma yaw-rate R² of +0.011 is being computed over exactly the frames comma excludes by design. **Recommended immediate action: adopt a speed gate (comma's own threshold is 4 m/s) before re-scoring comma yaw-rate.** It also confirms f≈910 px, matching the brief.

---

## 4. Yaw-rate / rotation vs translation

**The decomposition itself** is stated in three independent places verified above:
- **Nistér 2004** — the essential matrix contains **R** exactly and **t** only as a direction (§1.0). `PUBLISHED`
- **TartanVO** — *"the scale ambiguity only affects the translation **T**… we keep the loss for rotation **R** unchanged."* (§2.1) `PUBLISHED (cited)`
- **Longuet-Higgins & Prazdny 1980** — the rotational optic-flow component is depth-independent; the translational one is scaled by inverse depth (§1.0). `PUBLISHED (metadata)`, mechanism phrasing `UNVERIFIED`

**Lee, S. H. & Civera, J. (2021). "Rotation-Only Bundle Adjustment." *CVPR 2021*.** arXiv:2011.11724.
`PUBLISHED (cited)` — key result stated in the abstract: *"when two calibrated cameras observe five or more of the same points, their relative rotation can be recovered **independently of the translation**"*; generalised across multiple views to give a bundle adjustment for rotation alone, with *"complete immunity to inaccurate translations and structure."*
**Relevance to TanitAD IDM: the sharpest published statement that rotation is separable and immune to the scale problem.** It licenses an architectural decision: **give yaw-rate its own head and its own loss, decoupled from speed**, rather than a shared 4-channel regression head where a badly-conditioned speed target can contaminate the rotation gradient. That is a concrete v3 change.

### Rotation/translation error conventions in the benchmarks

**Geiger, A., Lenz, P. & Urtasun, R. (2012). "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite." *CVPR 2012*.** (Journal: Geiger, Lenz, Stiller, Urtasun, *IJRR* 2013.)
`PUBLISHED (metadata verified; metric definition verified from the official KITTI odometry benchmark page)` — the odometry metric reports **translation error in percent** and **rotation error in degrees per metre**, computed over **all subsequences of length 100…800 m** and averaged.
**Relevance to TanitAD IDM:** the field reports translation and rotation as **separate, differently-normalised** numbers and has done since 2012. Our single pooled R² per channel is already the right instinct; the KITTI convention is the citation that justifies never combining them into one headline figure.

**Sturm, J., Engelhard, N., Endres, F., Burgard, W. & Cremers, D. (2012). "A Benchmark for the Evaluation of RGB-D SLAM Systems." *IROS 2012*, pp. 573–580. DOI 10.1109/IROS.2012.6385773.**
`PUBLISHED (metadata verified; RPE rot/trans split from secondary source)` — defines **Relative Pose Error (RPE)** and **Absolute Trajectory Error (ATE)**, with RPE conventionally reported split into translational and rotational drift components.
**Relevance to TanitAD IDM:** the source for the RPE definition if we ever report our pseudo-labels as trajectories rather than per-channel regressions.

### Learning yaw-rate / steering from video directly

**Bojarski, M. et al. (2016). "End to End Learning for Self-Driving Cars" (PilotNet). arXiv:1604.07316.**
`PUBLISHED (arXiv id verified; contents from NVIDIA's own technical blog + abstract)` — a CNN maps raw pixels from a single front-facing camera **directly to steering commands**, trained on ~72 h of driving; learns useful road features without explicit labels.
**Relevance to TanitAD IDM:** the existence proof that **steering is directly regressible from a single forward camera** — consistent with our steer R² +0.742 being our second-best channel. Steering is a lateral/angular quantity and inherits rotation's good observability.

**Xu, H., Gao, Y., Yu, F. & Darrell (2017). "End-to-end Learning of Driving Models from Large-scale Video Datasets." *CVPR 2017*.** arXiv:1612.01079.
`PUBLISHED (cited)` — an FCN-LSTM that predicts *"a distribution over future vehicle **egomotion** from instantaneous monocular camera observations and previous vehicle state,"* learned from large-scale crowd-sourced video, with segmentation as a privileged side task.
**Relevance to TanitAD IDM:** note the two design choices we do not make. (1) It predicts a **distribution**, not a point estimate — the natural antidote to MMSE shrinkage. (2) It conditions on **previous vehicle state**, i.e. it is given the speed rather than asked to infer it. Both are cheap and both are directly available to us.

> **Gap flagged.** I found **no paper that isolates yaw-rate regression from monocular video as a task and reports its accuracy**, separate from full 6-DoF pose or from steering-angle imitation. The nearest things are the comma calib challenge (direction of travel: pitch+yaw, MSE, speed-gated at 4 m/s) and the rotation half of KITTI odometry. `UNVERIFIED` that such a paper does not exist — this is an absence found at a handful of query formulations, not a proof.

---

## 5. Acceleration from video

> **Short answer from the literature: acceleration is a genuinely second-order quantity, it is weak in the signal, and both biological and computer-vision systems handle it badly. Our −0.240 is not anomalous.**

**Werkhoven, P., Snippe, H. P. & Toet, A. (1992). "Visual processing of optic acceleration." *Vision Research* 32:2313–2329. PMID 1288008.**
`PUBLISHED (metadata verified via PubMed/ScienceDirect; findings from secondary source)` — measures human modulation thresholds for temporal modulations of speed and direction, and presents evidence that **human detection of velocity modulation is not directly based on the acceleration signal** (the temporal derivative of the velocity modulation).
**Relevance to TanitAD IDM:** the canonical psychophysics citation that the visual system does not compute acceleration directly, even where the information is present. Establishes that this is a property of the *signal*, not of our architecture.

**Related psychophysics** `PUBLISHED (metadata verified via PubMed listings; findings from secondary source)`: "Perceptual and oculomotor evidence of limitations on processing accelerating motion" (PMID 14765954); "Visual Acceleration Perception for Simple and Complex Motion Patterns" (PMID 26901879); time-to-passage work finding observers rely on **initial velocity and size** rather than acceleration. The consistent finding across this literature: observers estimate time-to-contact largely **independently of acceleration**.
**Relevance to TanitAD IDM:** three independent lines converge on "second-order optical motion is near-threshold." A representation trained on video with no acceleration supervision has little reason to encode it.

**Zhang, Y., Pintea, S. L. & van Gemert, J. C. (2017). "Video Acceleration Magnification." *CVPR 2017*.** arXiv:1704.04186.
`PUBLISHED (cited)` — observes that *"large motions are linear on the temporal scale of the small changes; small changes deviate from this linearity"*, and links **temporal second-order derivative filtering** to spatial acceleration magnification, magnifying acceleration precisely by ignoring linear motion.
**Relevance to TanitAD IDM:** the CV-side confirmation. Acceleration in video is a residual after linear motion is removed, and it is small enough that an entire method exists just to **amplify** it. Also a constructive hint: an explicit second-difference operator on the latent sequence (or on the model's own speed prediction) is a more promising path than asking a regressor to read acceleration off raw features.

**Kampelmühler, M., Müller, M. G. & Feichtenhofer, C. (2018). "Camera-based vehicle velocity estimation from monocular video." *CVWW 2018*.** arXiv:1802.07094.
`PUBLISHED (cited)` — winning entry of the CVPR 2017 vehicle velocity estimation challenge. **Light-weight trajectory-based features fed to an MLP outperform depth and motion cues extracted from deep ConvNets**, especially at long range where disparity and flow estimators degrade. Reports **1.12 m/s** average error vs **~0.71 m/s** for a LiDAR+radar ground-truth system.
**Relevance to TanitAD IDM: two direct hits.** (1) A small MLP on geometric trajectory features beat deep ConvNet features on a *velocity* task — the identical shape of our "0.86 M beats 19.98 M, and a ridge probe matches the transformer" result. **Capacity is not the lever in this literature either; the input representation is.** (2) The best published monocular velocity error on that benchmark is ~1.12 m/s against a ~0.71 m/s sensor-fusion reference — a useful sanity anchor for what monocular speed accuracy is worth chasing.

**"Estimating motion of constant acceleration from image sequences" (IEEE conference publication, IEEE Xplore document 201646).**
`UNVERIFIED` — I have the IEEE document id from a search hit only; I did not verify authors, year or venue. Listed because a classical treatment of constant-acceleration motion estimation from image sequences appears to exist and would be worth someone's ten minutes. **Do not cite as-is.**

**General principle (no citation attached):** `UNVERIFIED` — differentiating a noisy estimated quantity twice amplifies high-frequency noise, so acceleration derived from a noisy speed estimate is dominated by noise unless the trajectory is smoothed first. This is standard numerical-analysis knowledge; I did not locate a specific paper making the argument in the ego-motion setting. Flagging rather than dressing it up as a citation.

> **Verdict for §5:** the literature contains no positive result for directly regressing ego-vehicle longitudinal acceleration from monocular video, and three independent lines of evidence for why it is hard. Our linear-probe reading of **−0.095 on clean labels** should be read as **"acceleration is essentially absent from the frozen representation,"** which agrees with everything above. Chasing it as a fourth regression head is, on this evidence, the lowest-expected-value of our four channels.

---

## 6. Inverse dynamics models / action pseudo-labelling from video

**Torabi, F., Warnell, G. & Stone, P. (2018). "Behavioral Cloning from Observation" (BCO). *IJCAI 2018*.** arXiv:1805.01954.
`PUBLISHED (cited)` for record; `UNVERIFIED` on the IDM's architecture details, which are in the body not the abstract. Two phases: the agent first gathers **self-supervised experience** (state transitions with known actions), uses it to learn an inverse dynamics model, then applies that model to infer the actions in state-only expert demonstrations and behaviour-clones them.
**Relevance to TanitAD IDM:** the origin of the pattern we are implementing. Note its action units are the agent's own — the IDM is trained on *the same embodiment* it will label. We are doing something harder: labelling across two camera rigs and two vehicles.

**Baker, B., Akkaya, I., Zhokhov, P., Huizinga, J., Tang, J., Ecoffet, A., Houghton, B., Sampedro, R. & Clune, J. (2022). "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos." *NeurIPS 2022*.** arXiv:2206.11795.
`PUBLISHED (cited)` — verified details from the paper body:
- IDM trained on **1962 hours** of labelled contractor data (paper notes **as few as 100 hours** suffices).
- **Explicitly non-causal**: *"can look at both past and future frames to infer actions"*, using context t−2 … t+2 over a window of **128 consecutive frames** at 128×128.
- ~**0.5 B trainable weights**; 3-D convolution → ResNet stack → residual unmasked attention.
- Accuracy: **90.6 % keypress accuracy, 0.97 R² for mouse movements** on held-out data.
- Used to pseudo-label ~**70 k hours** of internet video (filtered from 270 k h).
- Core claim: *"The inverse dynamics task is much easier and thus requires far less data than the behavioral cloning task."*
**Relevance to TanitAD IDM: the direct comparison and it is uncomfortable.** VPT reached **0.97 R²** on its continuous channel with a **0.5 B-parameter non-causal IDM on 1962 h of labels**; our best continuous channel is 0.865 with a shrinkage bias, at 0.86 M params. **But the reason is not capacity** (we measured 19.98 M as worse). The relevant difference is that Minecraft mouse movement is an **on-screen, scale-free, unit-defined** quantity — the action *is* the observed image motion, in pixels — whereas our speed is a **metric world quantity the image does not determine**. VPT is the strongest possible evidence that IDMs work superbly **when the action is observable in the image**, which is exactly the condition our yaw/steer channels satisfy and our speed/accel channels do not.

**Schmidt, D. & Jiang, M. (2024). "Learning to Act without Actions" (LAPO). *ICLR 2024* (spotlight).** arXiv:2312.10812.
`PUBLISHED (cited)` — recovers **latent** action information, and thereby latent-action policies, world models and inverse dynamics models, purely from video; first method to recover the *structure* of the true action space from observed dynamics alone. Latent-action policies then fine-tune rapidly into expert policies with a small action-labelled dataset or with online rewards.
**Relevance to TanitAD IDM:** the "give up on units" branch. If metric speed is unrecoverable, a *latent* action that captures the controllable degrees of freedom, plus a **small calibrated head** mapping latent → m/s, is a legitimate architecture — and it puts the metric problem in one tiny, easily-recalibrated module instead of spreading it through the encoder.

**Bruce, J., Dennis, M., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., Aytar, Y., Bechtle, S., Behbahani, F., Chan, S., Heess, N., Gonzalez, L., Osindero, S., Ozair, S., Reed, S., Zhang, J., Zolna, K., Clune, J., de Freitas, N., Singh, S. & Rocktäschel, T. (2024). "Genie: Generative Interactive Environments." *ICML 2024* (best paper).** arXiv:2402.15391.
`PUBLISHED (cited)` — 11 B-parameter model trained **unsupervised from unlabelled internet videos**; three components: spatiotemporal video tokenizer, autoregressive dynamics model, and a **latent action model** that yields frame-by-frame controllability with no ground-truth actions. The learned latent action space transfers to imitating behaviours from unseen videos.
**Relevance to TanitAD IDM:** the scaled version of the LAPO idea. Genie's latent actions are **discrete and unitless by construction** — which is precisely how it dodges the problem we are stuck on. Confirms the branch is real but does not solve metricity; it declines it.

**Ye, S., Jang, J., Jeon, B., Joo, S., Yang, J., Peng, B., Mandlekar, A., Tan, R., Chao, Y.-W., Lin, B. Y., Liden, L., Lee, K., Gao, J., Zettlemoyer, L., Fox, D. & Seo, M. (2025). "Latent Action Pretraining from Videos" (LAPA). *ICLR 2025*.** arXiv:2410.11758.
`PUBLISHED (cited)` — three stages: VQ-VAE-based action quantisation learns **discrete latent actions between image frames**; a latent VLA is pretrained to predict them from observations + task text; finally the VLA is **fine-tuned on small-scale robot data to map latent → real robot actions.**
**Relevance to TanitAD IDM: this is the cleanest published answer to "what do they do about action scale/units."** They do not solve it in the pretraining — they push the entire units problem into a **small supervised fine-tune at the end**. For us: pretrain the IDM on scale-free targets (direction, yaw-rate, normalised flow) across all corpora, then fit a **tiny per-rig calibration head** on the metric labels we do have. This is the same shape as ZoeDepth's per-domain metric head (§2.6), arrived at from the RL side.

**Assran, M. et al. (2025). "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv:2506.09985.**
`PUBLISHED (arXiv id + core claims verified via arXiv listing/secondary; full author list UNVERIFIED)` — action-free JEPA pretraining on >1 M hours of internet video, then **action-conditioned post-training (V-JEPA 2-AC) on <62 hours of unlabelled robot video** from DROID, deployed zero-shot on Franka arms for pick-and-place via image-goal planning, with no environment-specific data or reward.
**Relevance to TanitAD IDM:** the strongest recent evidence for the split we should adopt — a big action-free representation plus a **small, cheap action-conditioned head**. Also the closest published analogue to the TanitAD 4-brain latent world model as a whole.

### Driving-specific IDM work

**Yang, J., Chitta, K., Gao, S., Chen, L., Shao, Y., Jia, X., Li, H., Geiger, A., Yue, X. & Chen, L. (2025). "ReSim: Reliable World Simulation for Autonomous Driving." *NeurIPS 2025* (Spotlight).** arXiv:2506.09981.
`PUBLISHED (cited)` for title/authors/venue/abstract. **The IDM usage is `PUBLISHED (mechanism from secondary source — the paper's own HTML/GitHub, not the abstract)`:** ReSim uses an **inverse dynamics model that estimates the ego trajectory from a video clip**, both to convert predicted videos into executable trajectories and to *measure action controllability* — the IDM transforms the model's action-conditioned prediction into an estimated trajectory, then scores L2 distance to ground truth.
**Relevance to TanitAD IDM: the only clearly-verified driving IDM in the current literature, and it is used as an evaluation instrument, not a pseudo-labeller.** Directly relevant to the program's closed-loop evaluation work as well as to v3. Worth reading their IDM section in full — they are solving our exact estimation problem for a different purpose, on the same kind of data.

**Yang, J., Gao, S., Qiu, Y., Chen, L., Li, T., Dai, B., Chitta, K., Wu, P., Zeng, J., Luo, P., Zhang, J., Geiger, A., Qiao, Y. & Li, H. (2024). "GenAD: Generalized Predictive Model for Autonomous Driving." *CVPR 2024* (Highlight).** arXiv:2403.09630. Dataset: **OpenDV-2K / OpenDV-YouTube**, 2059 h total (1747 h YouTube + 312 h public), 244 cities / 40 countries.
`PUBLISHED (cited)` for the paper; **`UNVERIFIED` for how the ego-action labels were produced.** The dataset carries two annotation types — frame **contexts** (text) and **ego-driver commands** — and the command *"is designed to correlate future predictions with ego actions."* I could not determine whether commands are model-predicted, VLM-generated or heuristic: the CVF supplementary PDF returned HTTP 403 and the OpenDV README defers to `opendv/utils/cmd2caption.py`.
**Relevance to TanitAD IDM: this is the most important open question in this document for the YouTube expansion, and it is unresolved.** GenAD is the reference project for putting action conditioning on 1700 h of unlabelled YouTube driving video. **Recommended follow-up: read `OpenDriveLab/DriveAGI/opendv/utils/cmd2caption.py` and the CVF supplementary directly.** Note the strong signal in what *is* visible: their command vocabulary is a small set of **discrete categories mapped to natural language**, not metric m/s — consistent with everyone else declining metric action units on internet video.

**Gao, S., Yang, J., Chen, L., Chitta, K., Qiu, Y., Geiger, A., Zhang, J. & Li, H. (2024). "Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability." *NeurIPS 2024*.** arXiv:2405.17398.
`PUBLISHED (cited)` — conditions on *"a versatile set of controls from high-level intentions (command, goal point) to low-level maneuvers (**trajectory, angle, and speed**)."* Uses the model itself to build a generalizable reward for action evaluation "without accessing the ground truth actions." Does not mention action pseudo-labelling in the abstract.
**Relevance to TanitAD IDM:** confirms the field's action space for driving world models is a **layered** one (intention → trajectory → angle/speed), which is congruent with the TanitAD strategic/tactical/operative hierarchy. Note that **speed appears as a low-level control** there too.

**Hu, A., Russell, L., Yeo, H., Murez, Z., Fedoseev, G., Kendall, A., Shotton, J. & Corrado, G. (2023). "GAIA-1: A Generative World Model for Autonomous Driving." arXiv:2309.17080.** (Wayve.)
`PUBLISHED (cited)` — video + text + **action** inputs, tokenised and modelled as next-token prediction, with fine-grained control over ego-vehicle behaviour. No pseudo-labelling mechanism described in the abstract.
**Relevance to TanitAD IDM:** the reference driving world model with genuine action conditioning, on a proprietary fleet corpus **where the actions came from the vehicle**. The contrast with GenAD/OpenDV is the whole point: fleet data has real actions; internet data needs an IDM.

**Pan, J., Zhou, C., Gladkova, M., Khan, Q. & Cremers, D. (2023). "Robust Autonomous Vehicle Pursuit without Expert Steering Labels." *IEEE RA-L*.** arXiv:2308.08380.
`PUBLISHED (metadata verified via arXiv + IEEE Xplore; mechanism from abstract snippet)` — instead of expert steering labels, uses a **classical controller as an offline label generation tool**; a CNN localises the target vehicle from a single RGB image plus velocities and an MLP regresses throttle and steering.
**Relevance to TanitAD IDM:** a different and cheaper answer to the same problem class — when actions are missing, **generate them with a model you trust** (here a controller; for us, a calibrated geometric solver) rather than learning them end-to-end. Note it feeds **velocity as an input**, not as an output.

---

## 7. The deliverable table — which literature method attacks which of our failing channels

**Legend for the last column:** *Yes* = works on top of frozen latents in an offline non-causal IDM as-is. *Partly* = works but needs either an extra input we can obtain, or a change to targets/heads/eval rather than to the encoder. *No* = requires access to pixels, to the encoder, to a calibrated geometric solve, or to sensors we would have to plumb.

| Our channel & failure | Literature methods that attack it | What it needs as input | Applies to a FROZEN-latent offline IDM? |
|---|---|---|---|
| **speed / metric scale**<br>R² +0.865, but 56.6 % of MSE is a per-clip level bias; gain 0.830 (~17 % shrinkage) | **A — Canonical focal normalisation.** Metric3D / Metric3Dv2 (`f_c/f` rescale of target or window), Depth Any Camera (ERP + FOV alignment) | per-clip focal length `f` (PhysicalAI `camera_intrinsics`; comma ≈910 px) | **Yes** — it is a rescale of *targets/windows*, not of the encoder. **Cheapest experiment we have.** |
| | **B — Input-level camera/geometry conditioning.** ZeroDepth geometric embeddings; TartanVO Intrinsics Layer `((u−cx)/fx, (v−cy)/fy)`; CAM-Convs; Neural Ray Surfaces | intrinsics (fx,fy,cx,cy), ideally per-clip | **Partly** — the IL tensor can be pooled/appended to the *latent* window as extra IDM-head channels (cheap, no encoder change). Conditioning the *encoder*, as CAM-Convs/NRS do, requires unfreezing. |
| | **C — Per-domain metric head + routing.** ZoeDepth metric-bins + latent classifier; LAPA/LAPO small latent→units calibration head | a domain/rig label at train time; **none at inference** (latent classifier routes) | **Yes** — a per-rig head on the frozen latent is a small change to our 0.86 M model and targets the per-clip level bias directly. |
| | **D — Known camera height / ground plane.** Song & Chandraker 2014; Wang et al. ICRA 2021; Wagstaff & Kelly IROS 2021; Xue et al. (DNet) IROS 2020 | camera height above road + extrinsics/pitch (GeoCalib gravity can supply pitch) | **No / Partly** — the classical solves need pixels and a ground-plane fit. The Wagstaff–Kelly *loss* form is portable only if we also predict depth, which we do not. |
| | **E — Velocity supervision on translation magnitude.** PackNet-SfM "M+v"; UnDeepVO (stereo baseline); D3VO | metric velocity at training time only | **Partly** — we already supervise speed. The lesson is diagnostic, not prescriptive: a loss cannot create observability, so pair E with A/B/C or it just re-learns a rig-specific scale. |
| | **F — Fuse an inertial/odometric measurement.** Martinelli T-RO 2012 (scale+speed observable with IMU); VINS-Mono ("minimum sensor suite for metric"); comma2k19 CAN wheel speed | IMU and/or CAN alongside video | **No** as a model change — but **the highest-leverage *label-side* fix.** Changes what we can pseudo-label, not how the IDM reads latents. |
| | **G — Predict a distribution / calibrate, don't shrink.** Xu et al. CVPR 2017 (distribution over egomotion); DORN ordinal binning; D3VO learned uncertainty; SfMLearner-style median scaling at eval | none beyond a head/loss/eval change | **Yes** — ordinal/quantile head + heteroscedastic uncertainty + reporting a per-clip-debiased R² alongside raw. Does not fix observability; **does** stop the model silently hedging and makes the residual bias measurable. |
| | **H — Nonholonomic constraint (scale from turning).** Scaramuzza et al. ICCV 2009 / IJCV 2011 | camera offset from rear axle (extrinsics); needs turns | **No** — degenerate on straight highway driving, which is most of comma2k19. |
| **yaw_rate / rotation**<br>pooled R² +0.105 (PhysicalAI **+0.904**, comma **+0.011**, comma labels corrupt) | **I — Fix the labels first.** comma2k19's own fused INS/GNSS/Vision **orientation** and CAN steering (Schafer et al. 2018), instead of `arctan2` of ENU velocity | corpus fields we already have | **Yes** — pure label-side. |
| | **J — Speed-gate the metric.** comma's own `calib_challenge` **ignores all frames below 4 m/s** because direction of travel is undefined at standstill | a speed threshold | **Yes** — a one-line eval change; **do this before re-scoring comma yaw-rate.** |
| | **K — Auto-masking of frames violating the motion assumption.** Monodepth2 auto-mask | none | **Yes** — mask degenerate windows in training as well as eval. |
| | **L — Separate the rotation head and its loss from translation.** TartanVO up-to-scale loss (*"scale ambiguity only affects the translation"*); Rotation-Only BA (rotation recoverable independently of translation); Nistér 2004 | none | **Yes** — architectural, cheap, and well-supported. Stops an ill-posed speed target from contaminating the rotation gradient. |
| | **M — Report rotation and translation separately, in their own units.** KITTI odometry (% for translation, deg/m for rotation); TUM RPE rot/trans split | none | **Yes** — reporting convention only. |
| | *(note)* Yaw-rate needs **no** scale method at all. **None of A–H apply here** and applying them would be wasted effort. | | |
| **long_accel**<br>R² −0.240; frozen-latent linear probe −0.095 on clean labels | **N — Accept it is second-order and near-threshold.** Werkhoven et al. 1992; time-to-contact psychophysics; Video Acceleration Magnification (CVPR 2017) | — | **Yes** as a scoping decision. On this evidence, **descope direct accel regression.** |
| | **O — Derive it, don't regress it.** Second-difference of a *smoothed* speed estimate, i.e. explicit temporal differencing rather than a learned head (Video Acceleration Magnification's second-derivative filtering is the closest published analogue) | a good speed estimate first | **Partly** — strictly downstream of fixing speed; noise amplification must be controlled (smoothing/regularised differentiation). `UNVERIFIED` as an established recipe in this domain. |
| | **P — Measure it instead.** IMU accelerometer (Martinelli: this is exactly what inertial data adds); comma2k19 9-axis IMU | IMU stream | **No** as a model change; **yes** as the correct label source. |
| | **Q — Feed it as an input, not an output.** Xu et al. CVPR 2017 conditions on previous vehicle state; Pan et al. 2023 feeds velocity in | prior ego-state | **Yes** if the downstream consumer permits it. |
| **steer**<br>R² +0.742 (our second-best) | **R — Direct steering regression from a forward camera is established.** PilotNet (Bojarski et al. 2016); Xu et al. CVPR 2017 | none | **Yes** — this channel is working roughly as the literature predicts. |
| | **S — Same rotation-side treatments as yaw.** L (separate head), K (auto-mask), J (speed-gate), plus intrinsics conditioning B for cross-rig transfer | intrinsics for B | **Yes** — steer is an angular/lateral quantity and inherits rotation's good observability; treat it with the yaw toolkit, **not** the scale toolkit. |

### The split the caller asked for, stated plainly

- **Methods that attack SCALE (speed, and by inheritance long_accel):** A canonical focal normalisation · B input-level camera/geometry conditioning · C per-domain metric head · D known camera height / ground plane · E velocity supervision on translation magnitude · F IMU/CAN fusion · G distributional / calibrated heads · H nonholonomic turning constraint.
- **Methods that attack ROTATION (yaw_rate, steer):** I label correction from the corpus's own fused orientation/CAN · J speed-gating (comma's 4 m/s) · K auto-masking degenerate frames · L architectural separation of the rotation head and loss · M separate rot/trans reporting units.
- **The two sets barely overlap, and that is the finding.** Rotation is already metrically observable (Nistér; TartanVO; Rotation-Only BA) — our yaw failure is a **data/labels/eval-protocol** failure, fixable this week at near-zero compute. Scale is genuinely unobservable from a geometry-blind monocular window — our speed failure is an **information** failure, fixable only by adding camera geometry (A/B/C) or a metric sensor (F). **Do not spend scale machinery on yaw, and do not expect label fixes to move speed.**

---

## 8. Register of gaps and UNVERIFIED items

Carried forward explicitly so nobody quotes these as settled.

1. **GenAD / OpenDV ego-command provenance — `UNVERIFIED`, and it is the most consequential gap here.** CVF supplementary returned HTTP 403; the OpenDV README points at `opendv/utils/cmd2caption.py`. Whether those 1747 h of YouTube commands are model-pseudo-labelled or heuristic decides how much of our YouTube plan is de-risked by prior art. **Assign a follow-up.**
2. **Extrinsics-conditioned *policy/dynamics* work — gap.** Verified extrinsics conditioning exists only on the depth side (FSM). Candidates seen but not fetched: arXiv:2603.05868 (AnyCamVLA), arXiv:2601.02994 (View-Invariant Latent Actions), arXiv:2601.09605 (Sim2real viewpoint-robust policies). All `UNVERIFIED`.
3. **A paper isolating yaw-rate-from-video as a task with reported accuracy — not found.** Absence probed at several query formulations only; per the operating standard, that is not proof of absence.
4. **PackNet's exact velocity-supervision loss formula and its scale-aware numbers — `UNVERIFIED`** (CVF PDF 403). The *existence* of the "M+v" setting and the metric-scale claim are verified from the paper's own supplementary.
5. **DeepVO has no verified arXiv id.** Cite the ICRA 2017 DOI (10.1109/ICRA.2017.7989236). Several unrelated preprints share the name — do not attach one.
6. **"Estimating motion of constant acceleration from image sequences" (IEEE Xplore doc 201646) — `UNVERIFIED` on all metadata.** Do not cite as-is.
7. **Longuet-Higgins & Prazdny's depth-independence-of-rotation phrasing — `UNVERIFIED` in the paper body.** The bibliographic record and the general decomposition are verified; the exact sentence is the standard textbook reading. If it must be load-bearing, cite Nistér 2004 or TartanVO instead, both of which state the rotation/translation asymmetry verbatim and were verified.
8. **Monodepth2 / DROID-SLAM / DPVO intrinsics-and-scale handling — `UNVERIFIED` from abstracts.** Stated per convention (median scaling; calibrated BA). Verify before putting in a decision doc.
9. **The noise-amplification-of-double-differentiation argument (§5) has no citation.** It is standard numerical analysis, presented as such.
10. **Author lists marked `UNVERIFIED`:** Zhou/Dai/Li arXiv:1903.00912; Wang et al. arXiv:2101.05995; Xue et al. IROS 2020 (order); Assran et al. arXiv:2506.09985 (full list).
