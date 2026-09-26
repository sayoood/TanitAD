# How long will REFe take to train, at the paper's data scale, including the augmented data?

**PI question, 2026-09-21.** *"How long will the training take with the same amount of data
including the usage of the augmented data."*

**Short answer: at the paper's full scale, on ONE A40, 123–133 days. That is not a job for one
card.** *(Published as 99–128 before the sixth review; see the correction in §3.)* Five arms below cost between 6 days and 133 days; the one I recommend costs **13 days** and
is the cheapest arm that produces a number comparable to a published row.

Every figure carries its evidence class. ⛔ **One term is ESTIMATED and it is the multiplier
between this dev box and an A40** — the first job on the pod must replace it, and it takes three
minutes.

⭐ **2026-09-23 -- WHAT THE POD RUN ACTUALLY IS (PI decisions: ViT-L, goal augmentation, ONE pod,
and an epoch = one pass over SCENES).** Arm C's budget below (navtrain only, ViT-L, 25 epochs) is
the run: **103,039 scenes x 25 = 2.58 M sample-passes -> ~900–975 A40-h ≈ 37–41 days** at FP32
on one A40. The goal-augmented targets do NOT enlarge the epoch: each scene is visited once per
epoch and its augmented twin takes turns with the logged intent across epochs (`train.py
--epoch-unit scenes`, R24). Counting epochs over the ~176 K stored targets instead would have been
1.71x the paper's budget (~64–70 days) -- the paper counts its data in scenes (Table A12).

---

## 1 · What "the same amount of data, including the augmented data" is

PUBLISHED, Table A12 p. 30, quoted exactly:

| | |
|---|---|
| Training data | **100K navtrain + 237K SimScale scenes** |
| Epochs | 25 |
| Batch size | 256 |
| GPUs | 16 × H20 |
| Training time | **38 h** |
| Training precision | **FP32** |
| Optimizer / LR | AdamW, 2 × 10⁻⁴, cosine annealing to zero (no warmup), weight decay 0.01 |
| Gradient clipping | global norm 1.0 |

**SimScale is the augmented half** — 237K out-of-distribution scenes on top of the 100K real
navtrain frames. So "the same amount of data including the augmented data" = **337,000 samples ×
25 epochs = 8,425,000 sample-passes**, and their own budget for it was **16 × 38 = 608 GPU-hours**.

⚠️ **Their 608 GPU-hours is not a target we can read our runtime off.** It is an H20 number for
*their* implementation. It is used below only as a cross-check, and the cross-check does not close
— see §5, which I state rather than hide.

---

## 2 · The measured basis

MEASURED 2026-09-21 on the dev box's RTX 4060 (8.00 GiB), REFe forward + backward + optimiser step,
512 × 960, FP32, batch 1, median of 5 reps after 2 warmups, `torch.cuda.synchronize()` around the
timer. Artifact: `raw/2026-09-21-camera-scaling/cam_scaling.txt`.

| backbone | 1 cam | 2 cam | 4 cam | peak GiB @ 4 cam |
|---|---|---|---|---|
| ViT-S | 0.127 | 0.243 | **0.492** | 1.93 |
| ViT-B | 0.279 | 0.554 | **1.139** | 3.69 |
| ViT-L | 0.788 | 1.630 | *(does not fit)* | 9.49 |

**The camera scaling is LINEAR, and that is now measured rather than argued:** ×3.874 at ViT-S and
×4.087 at ViT-B against an analytic prediction of ×4.00. The frozen trunk is ~99 % of the work and
each camera is one more image through it, so this is the expected law — the point is that it has
now been *tested* at two backbone sizes.

⇒ **ViT-L at 4 cameras = 3.217–3.372 s/sample.** ⚠️ **CORRECTED 2026-09-21 — the first version of
this file published 3.05–3.26 and its LOWER BOUND WAS WRONG.**

The sixth review proposed squaring the per-doubling ratio (r² = 2.0685² = 4.279 ⇒ 3.372) instead of
doubling it (2r = 4.137 ⇒ 3.260). **Both of us can be tested**, because at ViT-S and ViT-B all three
camera counts are MEASURED:

| backbone | r = t(2)/t(1) | 2r | r² | **MEASURED t(4)/t(1)** | closer |
|---|---|---|---|---|---|
| ViT-S | 1.913 | 3.827 | 3.661 | **3.874** | **2r** (err 0.047 vs 0.213) |
| ViT-B | 1.986 | 3.971 | 3.943 | **4.082** | **2r** (err 0.111 vs 0.140) |

⇒ **r² is refuted as a predictor** — it under-shoots both measured points, in the one place the
question can be settled. But the review was right that my lower bound was indefensible: the
measured multiplier **grows with backbone size** (3.874 → 4.082), so ViT-L cannot be below ViT-B's
4.082. The defensible band is therefore bounded below by ViT-B's measured multiplier and above by
the reviewer's r², and it **brackets both published numbers**:
`0.788 × (4.082 … 4.279)` = **3.217–3.372 s/sample**.

### ⛔ Why the ViT-L 4-camera row is blank, and why that matters

It **ran** — and produced **27.074 s/step at 9.49 GiB peak on an 8 GiB card**, a 34× outlier against
the 2-camera 1.630 s. That is the PCIe bus, not the GPU. The probe refused to quote it.

This reproduces exactly what Review 5 diagnosed: the previously banked 4-camera seconds were
**host-memory fallback thrash**, and the earlier "×15.25 compute scaling" read off them was an
artifact of paging. A timing probe that does not assert residency will publish the bus speed and
call it compute. *(Same family as reading `df` on a pod or `free` on Thor: a probe answering a
different question than the one asked.)*

---

## 3 · The answer, per arm

Sample-passes × s/sample ÷ 3600 = 4060-hours; ÷ the A40 multiplier = A40-hours.

| # | arm | backbone | samples × epochs | 4060-hours | **A40-hours** | **days, 1 A40** | evidence |
|---|---|---|---|---|---|---|---|
| **D** | **full paper scale, incl. SimScale** | ViT-L | 337K × 25 | 7,529–7,891 | **2,942–3,188** | **123–133** | ViT-L 4-cam EXTRAPOLATED along a measured law |
| C | navtrain only (no SimScale) | ViT-L | 100K × 25 | 2,234–2,341 | 873–946 | 36–39 | same |
| **B** | **full paper scale, incl. SimScale** | **ViT-B** | 337K × 25 | 2,666 | **1,042–1,077** | **43–45** | 4-cam step time **MEASURED** |
| **A** | **navtrain only** | **ViT-B** | 100K × 25 | 791 | **309–320** | **13** | 4-cam step time **MEASURED** |
| A− | navtrain only | ViT-S | 100K × 25 | 342 | 134–138 | **6** | 4-cam step time **MEASURED** |

**On 8 × A40, arm D falls to 15–17 days.** Scaling is near-linear for this workload — the batch is
already split 16 ways in the paper — so cards trade against days almost exactly.

### ⚠️ The ESTIMATED term, named — and CORRECTED

The A40/4060 multiplier is **2.475–2.559×, ESTIMATED, not measured.** Its basis (PUBLISHED spec):
FP32 37.4 vs 15.11 TFLOPS = **2.475×**; memory bandwidth 696 vs 272 GB/s = **2.559×**; TF32 tensor
74.8 vs 30.2 TFLOPS = **2.477×**. Three independent ratios within 3.4 % of each other is why a band
is quotable at all — but they are spec sheets, and none of them is REFe.

⛔ **The first version of this file quoted "2.48–3.0×", and the 3.0 was supported by NONE of those
three sources.** It was an optimistic round number carried over from the superseded one-camera
estimate, and it shortened every arm by ~20 %. Arm D was published as **99–128 days**; on the
programme's own evidence it is **123–133**. *(Caught by the sixth review. Class: a number with no
source surviving because it sat next to three that had one.)*

⇒ **First job on the pod: re-run `cam_scaling.py` there.** It takes three minutes and replaces the
only estimated term in this document. Until then every A40 column is a projection.

---

## 4 · What I recommend, and why it is not arm D

⭐ **Arm A (ViT-B, navtrain only, 13 days), then decide.** Reasons, in order:

1. **Its step time is MEASURED at four cameras**, not extrapolated. Arm D's is not, and cannot be
   on this box.
2. **The paper itself sanctions the backbone sweep.** Table A13 (§B.2) reports DriveVFM ViT-S, ViT-B
   and ViT-L, all trained on navtrain **only, without any SimScale data**, all other settings held
   fixed. So arm A is not a scoped-down improvisation — it reproduces a row the paper publishes,
   which is the correct anchor and the one this programme's "reproduce on THEIR data first" rule
   asks for.
3. **It answers the question arm D cannot answer any faster.** If REFe-with-DINOv3 does not track
   DriveVFM at ViT-B on navtrain, spending 100 days at ViT-L on 337K samples will not fix it.
4. ⛔ **Arm D's 123–133 days on one card is far longer than the programme's decision horizon.** If the
   PI wants arm D, the right question is *how many cards*, not *how long*.

⚠️ **What arm A does NOT buy:** a number comparable to DriveZero's headline 94.8 PDMS, which is a
ViT-L + SimScale result. It buys a comparison against the paper's own ViT-B row.

---

## 5 · The cross-check that does NOT close, stated plainly

⛔ **THIS SECTION USED THE WRONG DATASHEET COLUMN, AND THE ERROR FLATTERED US.** It converted the
H20 at **TF32 ~148 TFLOPS**, which is NVIDIA's **SPARSE** figure. The dense figure is ~74, and the
paper's stated precision is **FP32** (~44). Using the sparse column inflated the H20's assumed
speed, which shrank the apparent gap between their implementation and ours.

| H20 basis (PUBLISHED) | H20/4060 | their 608 GPU-h as 4060-h | **REFe slower by** |
|---|---|---|---|
| TF32 **sparse** 148 *(what I used — wrong)* | 4.90× | 2,980 | 2.5–2.6× |
| TF32 **dense** 74 | 2.45× | 1,490 | **5.1–5.3×** |
| FP32 44 *(the paper's stated precision)* | 2.91× | 1,770 | **4.3–4.5×** |

⇒ **REFe is roughly 4.3–5.3× slower per sample than DriveZero's implementation**, not 2.4–2.6×.
⚠️ **EVIDENCE CLASS: ESTIMATED / PUBLISHED-SECONDARY.** These are datasheet figures I could not
re-verify from a primary source offline, and the H20 row is the weakest of them. The *conclusion*
(our implementation is several times less efficient than theirs) is robust across all three rows;
the *multiple* is not. I am not going to explain the gap away. Candidate causes, none yet tested: no fused/flash attention path, no
`channels_last`, our register-compression module sitting at width 1024, per-step Python overhead,
and the fact that a 4060 at batch 1 is a bad operating point for a ViT (the paper's per-GPU batch
was 16). **The pod re-timing in §3 is also the experiment that starts closing this**, because a
larger resident batch is exactly what the dev box cannot provide.

⚠️ It is a *cost* gap, not a *correctness* gap — nothing here says the model is wrong. But it means
every hour figure above is REFe's hour figure, and REFe could plausibly get ~2× cheaper with
engineering that has not been done.

---

## 6 · Two departures that would change these numbers, and are the PI's call

| lever | effect | what it costs |
|---|---|---|
| **bf16 / AMP autocast** | roughly halves both time and memory ⇒ arm D **50–64 days**, arm A **6–7 days** | ⛔ Table A12 says **Training precision FP32**. This is a declared departure from the paper and must be recorded as one. |
| **larger resident batch on 48 GB** | better GPU utilisation than batch 1; the A40 fits ~batch 4 at ViT-L × 4 cameras | none — this is just using the card. It is also what `--accum` exists for: `--batch 4 --accum 64` reproduces the paper's effective 256. ⛔ **And `--epochs` was BROKEN under `--accum` until 2026-09-21** — it derived steps from the MICRO-batch, so this exact recipe would have run **1,601 epochs, not 25**, and every wall-clock figure built on it was 64× optimistic. Fixed and pinned: the trainer now prints the realised epoch count and it must equal `--epochs`. |

⚠️ **`--accum` is not optional at any batch this hardware can hold.** WTA gives gradient to exactly
one proposal per sample, so batch 4 touches at most 4 of 64 proposals per step. MEASURED at batch 2:
the winner was 1/2 on *every* logged step. Accumulation restores the gradient coverage their batch
of 256 had; it does not restore the forward statistics, and that distinction is in the flag's help
text.

---

## 7 · Data, which is a separate clock

Arm A needs navtrain's four camera channels on the pod. Front-camera-only nuPlan was MEASURED at
50.7 GB (mini) / ~73 GB (test) / ~99 GB (val); **four channels is ~4× that**, and the ranged
fetcher (`code/fetch_front_camera.py --cameras CAM_F0,CAM_L0,CAM_R0,CAM_B0 --container`) pulls
exactly those channels — MEASURED at **23.39 GB = 48.1 % of a 48.63 GB archive**, against 12.2 %
for one camera.

⛔ **Pull it on the pod, not here.** The dev-box uplink is 1.2 MB/s and its downlink is a few MB/s;
the same pull is ~1.2 h on a pod. ⚠️ **Containers are mandatory** wherever the volume has large
allocation units — MEASURED 5.0× inflation on D:'s exFAT 1 MiB clusters — so check the pod
volume's allocation unit before assuming it is a D:-only problem.
