# Decision brief: fine-tune refc on NAVSIM navtrain? (`H-NT-DECIDE-1`)

*For the PI. Research Lab (Architecture & Inference), 2026-10-21, Berlin. Design only: nothing has run.*

**The ask.** Authorise one diagnostic fine-tune:
* Start from θ_DEC (md5 `dad4cb5b…`; navtrain sample 76.36 / 75.68; navtest about 72.8, ESTIMATED from θ_0.5's 72.78).
* Train on NAVSIM navtrain with the **refc architecture unchanged** and the **front 120° stitch only**. This is the same input the NavSim agent already feeds: L0/F0/R0 re-stitched into one virtual camera.
* The run answers whether the 16-point gap to REFe (88.82 / 89.15) is **data**.

## Cost

| item | Thor (MEASURED basis) | A40 |
|---|---|---|
| engineering (loader, labels, 3D-box joins, controls, gate profile) | 3–4 agent-days, **0 GPU** | same |
| data: 31 `navtrain_current` shards, 295 GB; the metadata is already on Thor | ~6.5–8 h CPU wall (12.8 MB/s MEASURED) | ~1.2 h pull + ~3 h stitch on a **new pod** (the old one refuses ssh; it had no volume) |
| **A1 training**: 2 epochs = 12,478 steps at batch 16 | **31–34 GPU-h** (8.94–9.83 s/step MEASURED on refcv8; upper bound) | **20–55 A40-h** (UNMEASURED ratio; a 200-step timing smoke first) |
| reads: 10 snapshots on the held-out sample + one navtest read each for A0 and A1 | ~6.4 GPU-h | |
| **total** | **≈ 38–42 exclusive GPU-h (≈ 1.7 Thor-days); ~5–6 calendar days to the verdict** | plus pod spend (not priced) |
| replicate A1r (only if P lands within ±1.0 of a line) | +34–37 GPU-h | |

Disk peaks at about 80 of the 169 GB free on Thor: the shards are streamed, and the PNG bank is 53.8 GB (MEASURED per-frame size). The GPU must be **exclusive**: it read 95 % busy with other agents' jobs today.

## What each outcome means for refcv9 (committed in `PREREG_NT_DECIDE.md` §4)

| A1 navtest PDMS (mean of inference seeds) | verdict | refcv9 |
|---|---|---|
| **≥ 85.0** (or 84.5, see decision 2) | DATA | Keep the refc trunk. Train on or co-train with navtrain, then add the PDM-scorer selector (+3.8 on REFe). |
| **75–85** | BOTH | The split is named by the method-matched data share D_M = (A1 − A0) / (84.46 − A0):<br>• **D_M ≥ 0.8** (≈ A1 ≥ 82.1 if A0 = 72.8): port the scorer before any backbone change;<br>• **D_M < 0.5** (≈ A1 < 78.6): adopt REFe's backbone and multi-camera input;<br>• **between:** the optional camera proxy A2 or the zero-training REFe camera-masking probe decides first. |
| **≤ 75.0**, learning curve flat | SENSORS/BACKBONE | refcv9 adopts DINOv3 ViT-L and multiple cameras. This is an architecture change and needs your approval. |
| < line, curve still rising | UNDERTRAINED | Extend to 4 epochs (+31–34 GPU-h). Needs your compute. |

**Most likely outcome: BOTH, near 84** (ESTIMATED).
* PUBLISHED: the camera-only, front-wide **LTF** (ResNet-34) scores **83.8** on navtest after navtrain training (83.5 ± 0.6 over seeds; NAVSIM paper, banked).
* MEASURED: REFe **without its learned scorer** scores **84.46**.
* About 4.5 of REFe's ~89 points come from its learned scorer, and 1.2 more from its route fix. REFe also has a route-derived goal input. This experiment adds none of these.

## Decisions needed from you

1. **Compute:** about 1.7 exclusive Thor-days, **or** a new A40 pod (spend, plus a timing smoke).
2. **The DATA line.** Your 85.0 stands by default. Alternatively choose **84.5**, the method-matched REFe line. The choice is recorded before the run.
3. **The two navtest reads** (θ_DEC and the chosen A1). The θ_DEC read overlaps D-R8-WISE-NAVTEST.
4. Optional: the camera proxy A2 (+36–39 GPU-h, confounded), and/or the zero-training REFe camera-masking probe.

## Risks

* **The 85 line sits above both comparators.** LTF reads 83.8 and REFe without its scorer reads 84.46, so "DATA" requires refc to beat both.
* **Selection is on a Las Vegas-only sample.** θ_DEC was chosen on that same sample, which biases it upward. Only navtest decides.
* **Training on 2 Hz data.** Targets are interpolated, and the ST frames carry no motion. This matches inference exactly, but it caps what refc can learn.
* **Perception supervision changes.** The truck/bus/trailer classes get no positives on nuPlan, and the map head is off (no SAM3 GT). A1 is a **diagnostic, not a PhysicalAI deliverable**; PhysicalAI forgetting is reported beside.
* **Multi-camera refc is impossible without an architecture change.** The trunk call has no camera axis, there is one canonical frame, and the lift is one camera per clip. The only camera arm is a wider stitch, which is confounded.

Details: `FEASIBILITY_AUDIT.md` (sources, `file:line`) · `DESIGN_AND_COSTING.md` · `PREREG_NT_DECIDE.md`.
