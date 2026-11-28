# Camera-based situation detection — why our number is low, and the plan to fix it

**Written 2026-08-02 on the PI's directive:** *"If the ego is approaching an intersection, you will
see it in the image without any ego motion or ego data showing this. So ego can never beat the image
semantic detection. So I need a professional plan to boost the result of the camera based situation
detection including the right architecture and training data strategy."*

**The PI is right, and the reason is more damaging than the framing suggests.** This document
establishes the actual bottleneck first, because every architecture choice downstream depends on it.

---

## 0. ⛔ THE BOTTLENECK IS THE LABEL, NOT THE CAMERA AND NOT THE ARCHITECTURE

**MEASURED — `stack/tanitad/data/situations.py:29`, read directly:**

> *"INPUT is the ego trajectory ONLY: `P = [x, y, yaw, v]` at 10 Hz"*

Both detectors are pure ego-trajectory functions:

| label | how it is actually produced |
|---|---|
| `lane_change` | `detect_lane_change(K)` — two opposite-sign **yaw-rate lobes** (`LC_LOBE_DEG 1.5`) with **net heading change < 8°** (`LC_DPSI_MAX`) |
| `intersection` | `detect_intersection(K)` — **heading-change** based, with an optional cross-traffic assist |

⇒ **A frame is labelled `intersection` if and only if the EGO TURNED THERE.**

### What that does to each head

- **`head_ego` is close to a tautology.** It receives `[x, y, yaw, v]`; the label is a deterministic
  function of `[x, y, yaw, v]`. Its 0.08858 CV-AP is not evidence that ego kinematics *perceive*
  anything — it is a partially-memorised decoder of the labelling rule. This is a stronger statement
  than C60 (which says the ego trace encodes the driver's already-executed reaction). C60 says the
  baseline is *optimistic*; this says it is **measuring the label generator**.

- **`head_img` is being asked to predict the wrong thing.** A 4-way junction the car drives
  **straight through** is labelled **NEGATIVE** while the intersection fills the image. The camera
  head is therefore penalised for correctly seeing an intersection. Its ceiling is bounded by
  `P(ego turns | intersection visible)`, not by what the camera can resolve.

⭐ **CONSEQUENCE: the entire sitclf comparison cannot answer the PI's question.** The question is
*"is there an intersection ahead"* (a **scene** property). The labels encode *"did the ego execute a
manoeuvre"* (an **ego** property). Where the two diverge — exactly the cases that matter for
planning — the vision target is unlearnable **by construction**.

⛔ **No architecture change can fix this. Fix the label first or the next number will be as
uninformative as this one.**

### A second, separate defect found in the same file

`situations.py:19-22` still carries a docstring asserting
*"⛔ VISION ADDS NOTHING OVER EGO STATE"* with superseded numbers (img-only **0.0376**). The v2
result MEASURED camera-only at **0.04869 = 2.1× the shuffle null**, with the null landing at 0.02342
against a base rate of ~0.0227. ⇒ **that docstring is a stale absence-claim living in the label
generator itself** and must be struck (RETRACTION_LOG class: stale absence-claim, the C-class the
operating standard's rule 2 exists for).

---

## 1. LABELS — build scene-truth, not ego-event truth

**Target definition (new):** for each frame, *"is there a junction / intersection in the ego's
forward path within D metres, regardless of what the ego does"*, plus the analogous
*"is a lane change geometrically available/underway"*.

### ⛔ The obvious route is CLOSED, and this is already settled

**Map-matching to OpenStreetMap is impossible on our corpus.** `egomotion` carries **no lat/lon and
no GNSS** — coordinates are clip-local metres (CLAUDE.md, settled at five independent probes).
PhysicalAI-AV also ships **no map, lane graph, junction annotation, roundabout label or
traffic-light feature** — the dataset card says verbatim *"we do not include open maps data"*, and
`obstacle.offline`'s enum over 87,481 cuboids is **10 classes, all dynamic agents**.

⇒ Scene-truth labels must be **created**, not looked up. Three routes, ranked:

### ⭐ Route A (RECOMMENDED) — VLM pseudo-labelling with a human-calibrated gold set

We have already **verified access** (2026-07-20 byte-pull, token `Sayood`): **Cosmos-Reason1-7B and
Reason2-32B are UNGATED**, OpenMDW-1.1, commercial-use OK. This is the cheapest path to a
*scene-defined* label at corpus scale.

1. **Gold set first — 300–500 frames hand-labelled** by us, stratified over
   junction/no-junction × turn/straight × day/night/weather. ⛔ **Without the gold set the VLM's
   agreement with the ego-derived label would be mistaken for accuracy.**
2. Prompt the VLM per frame for a **structured** answer: `junction_ahead {none,<30m,30-80m,>80m}`,
   `junction_type {4-way,T,merge,roundabout}`, `lanes_left/right {0,1,2+}`.
3. **Report VLM-vs-gold precision/recall per stratum**, not a pooled accuracy. Publish the
   confusion matrix. A pseudo-label with an unmeasured error rate is not a label.
4. **Only then** pseudo-label the 2,376-episode parity corpus.

⚠️ Pseudo-labels inherit the VLM's blind spots (night, occlusion). Every downstream number must
carry `label_source = cosmos-reason{1,2}` and its gold-set error rate.

### Route B — transfer from a mapped corpus

**nuScenes** ships a real map API with drivable area, lane dividers and **junction polygons**;
Waymo/Argoverse-2 ship lane graphs. Train the situation head there with true scene labels, then
evaluate zero-shot on PhysicalAI and fine-tune on the Route-A gold set.
⭐ This gives a **genuine external validity check** our programme currently lacks entirely, and it
directly serves the "additional cameras" decision because those corpora are multi-camera.
⚠️ Cost: a new ingest path. ⚠️ Domain gap is real and must be measured, not assumed.

### Route C — geometric self-supervision (weakest, but free)

Cross-traffic tracks from `obstacle.offline` (present on **97.44 %** of the corpus, 10 dynamic
classes) give a *proxy*: sustained laterally-crossing agents imply a junction. Useful as an
**auxiliary target**, not as ground truth — it is silent on empty junctions.

---

## 2. ARCHITECTURE — five changes, in expected-value order

### ⭐ 2.1 Stop reading frozen world-model latents

The current head reads v1's latent, which is trained to **predict 2 s of ego motion**. Static scene
semantics that do not help that objective are exactly what such a bottleneck discards.

⚠️ **We already have the evidence that this is an extraction failure, not an absence:** under a
**linear ridge probe the camera reaches 0.836 of ego, where the neural head puts it at 0.549.**
A *linear* probe beating the *neural* head means the signal is present in the features and the head
is the problem. ⇒ **the first fix is the head/encoder path, and it is cheap.**

⇒ Use a **semantically pretrained encoder**: DINOv2 (already in-house, REF-A), SigLIP, or
**V-JEPA 2** (which also serves backlog B5). Frozen first; LoRA / last-N-block unfreeze second.

### 2.2 Make it temporal — an approach is a process, not a frame

An intersection *approaches*: looming, road-edge divergence, the cross-street gap opening. A
single-frame classifier cannot use any of that. ⇒ a **short causal temporal transformer** over
~2–4 s of frames (strided, e.g. 8 frames at 2 Hz). ⛔ Causal only — no future frames.

### 2.3 Resolution and field of view

At 256 px a junction 60–80 m ahead occupies very few pixels; the cue is destroyed before the head
sees it. Our v5 move to **176×624 at 120° HFOV cylindrical** is the right direction — a wide FOV is
also what makes cross-streets visible at all. Recommend the situation head train at **v5 geometry**,
and ablate resolution explicitly (256 vs 176×624) since this is a *measurable* claim.

### 2.4 Replace binary classification with a distance/geometry target

Predict **distance-to-junction** (regression) and **junction type** (multi-class) rather than a
1.7 %/2.8 % binary. A dense, ordinal target gives far more gradient per frame and is what a planner
actually needs. Keep the binary as a derived threshold for comparability with the current number.

### 2.5 Imbalance handling, stated explicitly

Base rates are `lane_change` **0.01726**, `intersection` **0.02816**. Use focal loss + class-balanced
sampling; keep **AP** as the metric (accuracy is meaningless here) and **always publish the shuffle
control** — it is what made the v2 table interpretable.

---

## 3. WHAT THIS MEANS FOR THE "ADDITIONAL CAMERAS" DECISION

⛔ **The current sitclf result cannot support that decision**, for the reason in §0: it measures an
ego-defined target. Feeding it into a camera-count choice would be the same error class as the
`nonav_route_beats_majority` gate — **adjudicating a hardware decision on a label bug**.

⇒ **Sequence: fix labels (§1) → rebuild the head (§2) → *then* run the camera-count ablation.**
The ablation itself is only meaningful on a corpus with real side/rear coverage (Route B), because
PhysicalAI-AV front-wide cannot answer "what would a second camera add".

⚠️ Also carry the **two-rig** finding into any multi-camera work: AV front-wide has **two distinct
rigs** (cy ≈ 543 rig A / cy ≈ 755 rig B) and a geometric-centre crop is ~215 px wrong for rig B.
Per-clip `cy` is required or the vertical framing is inconsistent across training frames.

---

## 4. THE LADDER — cheapest discriminating step first

| # | step | cost | what it decides | falsified if |
|---|---|---|---|---|
| **L0** | ⭐ **Re-score the EXISTING camera head against a 300–500-frame hand-labelled SCENE-truth gold set** | **~0 GPU**, hours of labelling | Whether the low AP is the label or the model. **This is the single highest-value action in this document.** | If camera AP is *also* low on scene-truth, the label is not the bottleneck and §2 becomes primary |
| L1 | Swap frozen-WM latents → DINOv2/V-JEPA2 features, same labels, same head | <0.5 GPU-day | Extraction vs representation (the ridge-vs-neural gap) | No gain ⇒ the WM latent was not the limiter |
| L2 | Add the temporal window (§2.2) | ~0.5 GPU-day | Whether approach dynamics carry the signal | Flat ⇒ single-frame semantics dominate |
| L3 | VLM pseudo-label the parity corpus (Route A) after gold calibration | ~1–2 GPU-days | Scale-up of scene-truth | VLM-vs-gold error too high per stratum |
| L4 | nuScenes transfer (Route B) + distance/type targets | ~1 engineer-week | External validity + the camera-count question | Domain gap too large to transfer |

⭐ **L0 requires no GPU and no new model.** It is executable immediately and it is the step that
tells us whether anything else in this plan is worth doing.

---

## 5. Pre-registered outcomes for L0 (both committed in advance)

| outcome | reading | consequence |
|---|---|---|
| **Camera AP rises substantially on scene-truth** vs the ego-derived label | The label was the bottleneck, as §0 argues | The v2 sitclf conclusion is **superseded**; proceed L1→L3; the "ego beats camera" line is retired |
| **Camera AP stays ~2× null on scene-truth** | The label is NOT the bottleneck | §0's argument is **falsified** and must be logged as a retraction; the limiter is the encoder/head and L1/L2 become primary |
| Gold set too small to separate | Underpowered | Report as underpowered; enlarge the gold set. ⛔ Do **not** pick the convenient reading |

---

## Evidence class

| claim | class |
|---|---|
| labels are derived from ego trajectory only | **MEASURED (ours)** — `situations.py:29`, `detect_lane_change`, `detect_intersection` read directly 2026-08-02 |
| camera-only 0.04869 = 2.1× null; ego 0.08858; ridge 0.836 vs neural 0.549 of ego | **MEASURED (ours)** — `SITCLF_V2_RESULT.md`, 5-fold CV |
| base rates 0.01726 / 0.02816 | **MEASURED (ours)** |
| no map/lane-graph/GNSS in PhysicalAI-AV | **MEASURED + PUBLISHED** — five probes + the dataset card |
| Cosmos-Reason1-7B / Reason2-32B ungated, commercial-OK | **MEASURED (ours)** — byte-pull 2026-07-20 |
| nuScenes/Waymo carry junction polygons / lane graphs | **PUBLISHED** — ⚠️ not re-verified this session |
| "the label is the bottleneck" | **HYPOTHESIS** — L0 is designed to falsify it |
