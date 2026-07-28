# Asset inventory, PART 2 — the workstreams my first pass MISSED

**PI 2026-07-29: "I'm missing the achievements about hierarchical planning and reasoning, the IDM
performance, the data sets, the VLM curation pipeline, AlpaSim, closed loop optimization, attention
based tactical usage of side cameras, strategic routing…"**

⛔ **He is right, and the omission was systematic, not random.** `ASSET_INVENTORY.md` inventoried the
things I had personally touched this session — the flagship/REF arms, TanitEval, parity, the gate —
and silently treated *"what I worked on"* as *"what the programme has"*. **Seven whole workstreams,
several of them the programme's actual thesis, were left out.** This file is the correction; the two
should be read together.

---

## 1. ⭐ IDM — a working uncalibrated inverse dynamics model

`…/Architecture & Inference/…/incoming/2026-07-2{2,4,6,7}-idm-*`

| channel | held-out R² | note |
|---|---|---|
| **speed** | **+0.865** pooled | 56.6 % of MSE is a per-clip level bias; gain 0.830 (~17 % shrinkage) |
| **yaw-rate** | **+0.904 on PhysicalAI** | pooled +0.105 only because comma2k19 (+0.011) has corrupt labels |
| **steer** | **+0.742** | |
| long_accel | −0.240 | label mismatch, see below |

⭐⭐ **The model is never told intrinsics, extrinsics, focal length, camera height or FOV.** That is
the property that makes action-free YouTube video usable at all.

⭐ **Smaller is better, measured: 0.86 M > 2.90 M > 19.98 M** in quality-vs-parameters, and a **ridge
probe ≈ the trained transformer** on all four channels. Capacity is not the constraint.

⭐ **Every channel's ceiling is diagnosed, not guessed** (`IDM_DIAGNOSIS.md`):
- **speed** — monocular scale ambiguity, *not* capacity: a linear probe on frozen z reaches R² 0.772
  vs the trained 2.9 M head's 0.775.
- **yaw-rate** — a comma2k19 **label** defect: 0.30 % of frames physically impossible (|ω| up to
  **15.5 rad/s at v ≈ 0**); PhysicalAI is clean.
- **steer** — **redundant by construction**: `corr(steer, yaw/v) = 0.9865`; the label *is*
  `atan(2.9·κ)` and carries zero information beyond (yaw_rate, speed).
- **long_accel** — the label is not the quantity the video shows: r = **0.434** with pose-derived dv/dt.

✅ **Downstream ablation verdict: GO** — IDM pre-training is **CI-separated from the random-init floor
on all 8 seeds across both domains**.

## 2. ⭐ Strategic routing — the lane graph exists and is machine-readable

`…/incoming/2026-07-26-4brain-gates/GATE_RESULTS.md`, Gate-1, 51 scenes, **MEASURED**:

- **12,030 successor edges over 11,877 lanes, 0 dangling ids**; **11,048 `PREVIOUS_LANE` rows**.
- ⇒ **`succ(lane)` connectivity is readable** — the substrate a strategic planner needs.

⭐ This is the direct answer to the settled finding that **PhysicalAI-AV contains no map, lane graph
or route signal**: the strategic brain's topology has to come from AlpaSim, and Gate-1 proves it is
actually there and traversable.

## 3. Hierarchical planning — the measurement apparatus, built deliberately first

`…/incoming/2026-07-26-4brain-{dominance-program,preconditions,gates,s3}`

The commission was *"prove the dominance of the 4-brain architecture"*, and the stream's own framing
is the valuable part: **"the hierarchy thesis is not on trial — the measurement apparatus is."**
Independently measured confounds each make a hierarchy effect **undetectable rather than absent** —
which is why the programme built gates and problem specs (`STRATEGIC_TACTICAL_PROBLEM_SPEC.md`,
`DATA_STRATEGY.md`) before spending GPU on a dominance claim. Zero GPU consumed.

## 4. ⭐ Attention-based tactical use of side cameras (H2)

`…/incoming/2026-07-25-h2-sensor-attention`

Formulation: **⟨SITUATION⟩ → ⟨TACTICAL OPTION⟩ → ⟨SENSOR REQUEST⟩** — the model asks for the camera
its chosen manoeuvre needs. H2 was the portfolio's least-developed constitutional hypothesis (DoA
15 %, previously *"nothing MEASURED on our stack"*).

⭐ **Its best feature is that it pre-empts its own failure mode:** labelling *"activate left camera"*
by a rule over something the model can already see would reproduce **exactly** the route-target
circularity that made `nonav_route_beats_majority` void by construction. The stream names that trap
**before** building the label — the lesson from one workstream transferring into another.

## 5. AlpaSim / TanitResim — a closed-loop simulator, consolidated

`…/incoming/2026-07-{19,22,26}-alpasim-*`, `2026-07-24-tanitresim-productionization`

- **Simulator source**: 12 packages (controller, driver, eval, grpc, physics, plugins, runtime,
  tools, trafficsim, utils, utils_rs, wizard), upstream `NVlabs/alpasim @ 55814289…`.
- **NRE / NuRec renderer** packaged hermetically (py3.11 + torch 2.7.0+cu128).
- **NuRec scene reconstructions** — 2.89 TB, gated, **with embedded HD map**.
- ✅ `verify_imports.py bad=0` — **MEASURED**, the bare-run setup actually imports.

## 6. ⭐ Pseudo-simulation — a closed-form negative *and* a validated positive

`…/incoming/2026-07-27-pseudo-simulation/PSEUDO_SIMULATION.md`

⛔ **The lateral grid axis is dead, and dead in closed form**: the flat-road warp's relative
displacement error is **exactly `height_above_road / h_cam`**, independent of depth.
⭐ Made concrete: at |dlat| = 2.0 m, depth 15 m — road surface −35.47 px ✅ but **a sedan roof moves
1.18 px, 3.3 % of the truth**. And **exactly 50.0 % of the frame has no ground-plane preimage at
all**, so the model cannot represent the upper half of a driving frame even in principle.

✅ **The instrument is proven non-vacuous: the identical test PASSES on the yaw arm at max error
0.0 px** over 30 (h_cam, pitch) conditions. ⇒ **the yaw axis of pseudo-simulation is validated and
usable**; only the lateral axis is refused. That distinction is what makes the two-sided PSS /
recovery metrics (now the v5 held-out gate's primary) trustworthy.

## 7. VLM curation pipeline

`…/Data Engineering/…/incoming/2026-07-2{0,1}-vlm-*`, `2026-07-26-h2-label-v2`,
`2026-07-27-percandidate-labels`

- **Production semantic labels** over the pod3 val build, **all 80 episodes**.
- ⭐ **The manifest is `t-major` by design** — a run stopped early has covered *every* episode at a
  coarser time stride rather than the first N episodes; the train manifest is shuffled so any prefix
  is a uniform subsample. **Partial output is still a valid sample**, which is real engineering
  judgement, not an accident.
- **Human audit sheet** (`audit_val.tsv`) stratified to over-sample eventful geometries, placing
  `SHIPPED_geometry` beside `passB_geometry_CONTAMINATED` so contamination is visible to the auditor.
- **Cosmos Reason1 vs Reason2** head-to-head staged; Cosmos gating verified by byte-pull.

## 8. Datasets — ours, published and balanced

- **TanitDataSet-C** — **90 episodes, 15.93 GB, pushed GATED to HF** (`2026-07-25-tanitdataset-hf-push`).
- **v2 corpus, 50 h maneuver-balanced** — hits the balance target **exactly**: **turns 14.25 % →
  28.0 %** (L/R balanced), parked fraction 1.4 %, over the **9,000 clips (50 h)** the design needs.
  Feasibility MEASURED from local egomotion alone.
- **Parity corpus** — 2,376 episodes, skip-hash `f09e44db`, cryptographic manifest (Part 1 §3).
- **v2 compressed caches** — 80 GB train + 20 GB val at w120/256×640cyl.

## 9. Closed-loop optimization

Covered in Part 1 §4 (E1c: peak cross-track **38.944 → 3.042 m, −92 %**; λ_replay as a calibrated
trade-off curve). **Add:** the guardrail itself is now a *decided* instrument — the PI chose to gate
on the **lateral p90** (2026-07-29), and the full re-adjudication is in `GA_P90_READJUDICATION.md`.

---

## 10. Documentation / cleanliness — same verdict as Part 1

Every stream above has a dated `…/incoming/<date>-<topic>/` folder with its own `MANIFEST.md` or
result doc, raw JSON, and the producing script. Working tree clean · **0 unpushed** · `stack`
**1,586 passed / 12 skipped** · `origin/main` fast-forwarded to the branch tip 2026-07-29.

⚠️ **What Part 1 got wrong is worth keeping visible:** an inventory assembled from what the author
recently touched will systematically under-report a programme this wide. **The hub's `incoming/`
tree is the authoritative index of what exists — not any one agent's memory of it.**
