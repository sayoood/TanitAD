# RESULT — WP-G: our agents are SUB-PATCH, and it may explain a defect we blamed on the objective

**2026-09-05 · TanitAD_TrainingFlyWheel · Tier T0 · Evidence class: MEASURED (agent geometry,
89,423 instances) + ESTIMATED (pixel footprint, formula and approximation named below).**

---

## 0. Why this ran before any teacher was loaded

WP-G asks whether a strong backbone can separate motorcycle from bicycle inside `rider`.
⛔ **No backbone can classify what the sensor did not resolve** — so the pixels are asked first,
and they can close the question for free.

⚠️ **And the scoring question is separate and harder:** we cannot measure a teacher's
motorcycle-vs-bicycle accuracy, because those labels are exactly what the corpus lacks. WP-G proper
must therefore be an **unsupervised separability test with a known-separable control pair**. This
script decides whether that is worth attempting at all.

## 1. The footprints

Cylindrical projection, `f_ref` 305.577, 256×640. `px = f_ref · size / range`.
⚠️ **ESTIMATE**: the angular footprint, not a measurement of the rendered crop. ⛔ The pinhole
formula is wrong on this projection (CLAUDE.md FOV trap); the column is linear in azimuth.

| class | n | range p50 | px w × h (p50) | ≥32 px tall |
|---|---|---|---|---|
| automobile | 70,958 | 59.6 m | 9 × 20 | 31.1 % |
| person | 14,214 | 43.2 m | **5 × 12** | 10.6 % |
| **rider** | 1,215 | 48.7 m | **5 × 11** | **14.2 %** |
| bus | 1,431 | 91.9 m | 10 × 39 | 67.5 % |

⭐⭐ **A median rider spans 0.8 of one ViT patch vertically and 0.4 horizontally** (DINOv2/v3 and
CLIP use 14–16 px patches). **It does not fill a single patch.** Motorcycle-vs-bicycle is not
recoverable from that, by any teacher.

## 2. ⭐ But the path is SCOPED, not closed — and the scoping is the safety-relevant half

| rider at | px | patches tall |
|---|---|---|
| 10 m | 24 × 52 | **3.7** |
| 13 m | 19 × 40 | **2.9** |
| 20 m | 12 × 26 | 1.9 |
| 30 m | 8 × 17 | 1.2 |
| 49 m (median) | 5 × 11 | 0.8 |

**14.2 % of riders are ≥32 px tall**, and those are the NEAR ones. ⭐ **A rider at 13 m matters
more than one at 49 m** — the resolvable subset is the subset that can hurt you. ⇒ WP-G proper is
worth running **scoped to the near band, and the scope must be stated in every number it produces.**

## 3. ⭐⭐⭐ THE CROSS-CONNECTION — a defect we may have mis-attributed

P4-3c records champ30k's decodability profile as an **objective** problem. Read it against the
footprints:

| target | result | footprint |
|---|---|---|
| `lead_gap_m` | **+0.1273 separated** | the LEAD car — large and near |
| `ego_speed` | **+0.1823 separated** | egomotion — needs no resolution at all |
| `vru_ahead` | **0.4544 at chance** | VRU = person/rider = **~5 × 11 px** |
| `left_occupied` | **0.5240 at chance** | small lateral agents |

⇒ **Everything the trunk decodes is either large-and-near or ego-derived. Everything it fails is a
small distant agent.** That is exactly the pattern a RESOLUTION limit produces, and it is
indistinguishable — from the decodability numbers alone — from the objective failure it was
recorded as.

⚠️ **I am not claiming the objective is fine.** I am claiming the two explanations are **confounded
in the current evidence**, and that a resolution limit is the cheaper hypothesis nobody tested.
⭐ **The discriminating experiment is cheap:** re-run the decodability battery on the NEAR subset
only (agents < 20 m). If `vru_ahead` separates there and not overall, the objective is not the
problem — the input is.

## 4. Consequences

* ⛔ **Fine-grained semantic distillation (motorcycle vs bicycle) is not viable corpus-wide.** It
  may be viable in the near band; nothing else.
* ⚠️ **This weakens the v6 relational latent further**, from a different direction than WP-F: edges
  between agents the encoder cannot resolve carry the encoder's uncertainty, not the scene's
  structure.
* ⭐ **It raises a question above all of this**: if the bottleneck is input resolution, then
  **higher-resolution or multi-crop input is a larger lever than any reasoning architecture we
  have discussed today** — and it is testable before anything is built.

⚠️ **Caveat, stated:** the footprint is computed from the cuboid `w`/`l` plus an assumed 1.7 m
rider height, on the angular approximation above. A 2× error would still leave the median rider
under 2 patches. The conclusion is robust to that; the exact percentages are not.
