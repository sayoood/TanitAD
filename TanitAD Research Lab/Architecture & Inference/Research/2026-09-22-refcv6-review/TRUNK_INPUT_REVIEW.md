# refcv6 — INDEPENDENT ADVERSARIAL REVIEW: the TRUNK and the INPUT PIPELINE

**Reviewer:** Arch+Inf adversarial stream, 2026-09-22. **Dimension:** trunk + input pipeline only
(diffusion, tactical/nav, perception/data and training/guards are four sibling streams).
**Repo HEAD at start:** `37645fcc61b158a327e6597e64e0abb64dcbc46d`, branch `agent/arch-inf-20260803`.
**Box:** dev box, CPU only (`CUDA_VISIBLE_DEVICES=""`), `OMP_NUM_THREADS=6`, torch 2.11.0+cu128,
timm 1.0.29, `tanitad` imported from `D:\Projects\TanitAD\stack\tanitad\__init__.py` (asserted).

**Authority read first:** `Project Steering/SPEC_REFCV6_V2.md` §2, §10.1–10.3, §11 R1, §12;
`Project Steering/ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md` classes A and B.

⚠️ **Method note on evidence.** Every number below is `MEASURED` on this box with the artifact
path given, or it is marked otherwise. Where an absence is claimed, a same-breath control that
must read non-zero is named. Where a guard is relied on, the regression it should catch was
CONSTRUCTED and confirmed to go RED (or the guard is reported as unproven).

⚠️ **Shared directory.** `raw/` also received `p1_box_path_filter.json` and `recipe_probe.json`,
which this stream did **not** write — a sibling reviewer is using the same package folder. Every
artifact of this stream is prefixed `q1`,`q1b`,`q1c`,`q1d`,`q2`,`q2b` or `trunk_` and listed in the manifest.

---

## VERDICT AT A GLANCE

| # | question | answer |
|---|---|---|
| 1 | Is the checkpoint's ImageNet mean/std actually applied? Channel order? Value range? | ✅ **YES, verified end to end on real pixels.** RGB and `[0,1]` are correct and declared at four sites. **Class A does NOT bite refcv6.** |
| 2 | Any positional/geometric tensor with no checkpoint counterpart? Does the lift consume real geometry? | ✅ **No orphan tensor; the lift genuinely reads per-clip extrinsics** (controls confirm). ⛔ But the lift's **`observed` mask option is dead in the refcv6 path** — see F2. |
| 3 | Is `f_ref` expressed at the resolution the network sees? Hard-coded 640/160/40/20? | ✅ **Exact at 416×1024.** ⛔ **But `frame_for_model` mis-derives `f_ref` for the two declared rig-clean geometries** — class B3, latent — see F3. |
| 4 | 0.1172 °/px vs NAVSIM/DD's 0.1367 | ✅ **Ours re-derived exactly** under the cylindrical rule. ⚠️ **Theirs is an AVERAGE over a pinhole mosaic and its 140° is INHERITED** — the two numbers are not the same kind of quantity — see F9. |
| 5 | K=3 shared weights, fused after the trunk? Ego history past-only, with a guard that can go RED? | ✅ **Yes on both, measured at the stem and proven by 3 constructed mutations.** ⛔ But `fuse_identity_init`'s bit-identity **fails in TRAIN mode** (BatchNorm) — see F6, F7, F8. |
| 6 | Channel counts from `feature_info`, both backbones? | ✅ **Yes — 256/512 and 1024/2048, cross-checked against a real forward.** See F8. |

⭐ **The three findings that would change what happens next:** **F1** (a bimodal unobserved black strip
on 74/139 clips, the C26 signal the programme already refused once), **F3** (a latent class-B3
`f_ref` error that activates the moment F1 is mitigated the obvious way), and **F6** (the K=1 vs
K=3 comparison is confounded at step 0 by a BatchNorm batch, fixable at zero cost).

---

## ⛔ F1 — THE 416×1024 GEOMETRY REINTRODUCES A **BIMODAL UNOBSERVED BLACK STRIP** — THE SHAPE `calib.py` ITSELF NAMES AS RETRACTION CLASS **C26** — ON **74 OF 139** CLIPS

**Evidence class: MEASURED** (ours).
Artifacts: `raw/q1c_zero_border.json`, `raw/q1d_rig_black_strip.json`;
instruments `code/q1c_zero_border_census.py`, `code/q1d_rig_black_strip_census.py`.
Corpus: `D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl` — the gated eval-139 cache
`SPEC_REFCV6_V2.md` §12.3 names, and the one §10.6's pre-pod validation runs on.

Census over **all 139 clips**, 2 decoded frames each, counting rows that are **exactly zero in all
three channels across the full 1024-px width**:

| | |
|---|---|
| clips with **≥1 fully-black row** | **74 / 139 (53.2 %)** |
| clips with **0** | **65 / 139** |
| black rows when present | **27–37, mean 31.1** of 416 (**7.5 %** of the frame height) |
| where | **bottom: 74 clips · top: 0 clips** |
| zero-pixel fraction | mean **0.0578**, median **0.0919**, max **0.1863** |

The histogram is **cleanly bimodal** — 65 clips at exactly 0, then a tight cluster at 27–37, with
nothing in between (0 clips at 1–26). That is the shape of retraction class **C26**, stated
verbatim in the code that defines the programme's answer to it:

> `stack/tanitad/data/calib.py:1031-1036` — *"the deployed crop replicate-PADS rows that fall
> outside the sensor — 0.0017 % on rig A, 8.897 % on rig B (n = 3,000) … `cylindrical_rectify`
> removes the FABRICATION … but NOT the asymmetry: a **rig-correlated BLACK region is still a
> rig-correlated signal, and this model eats shortcuts**."*

The programme's fix was `PHYSICALAI_RIG_CLEAN_176x624` / `_128x576` (`calib.py:1250-1264`) — *"the
LARGEST centred sub-rectangle … whose observed mask is EXACTLY 0.000000 for every clip of BOTH
rigs"*. **416 × 1024 is not such a frame**, and `SPEC_REFCV6_V2.md` §12.3 says so without drawing
the consequence: it reports **VFOV 46.0921° vs the 256×640 reference 45.4556°** and calls the
excess a benefit — *"416 slightly **exceeds** today's field"*. The excess is precisely the region
one rig does not observe.

⭐ **Why this is not cosmetic, and why it is a TRUNK/INPUT finding rather than a data one.**
After the (correctly applied) ImageNet normalisation, an exactly-zero pixel is **not** zero to the
network: it is `(0 − mean)/std` = **−2.1179 / −2.0357 / −1.8044** — a large-magnitude constant,
measured at `raw/q1_normalisation.json` (`global_min −2.117904`). A 7.5 %-of-frame constant block present on some clips and absent on others is exactly the shape
C26 warns about, and it sits in the **NEAR FIELD**: measured in F2's near-field table, every
affected BEV cell falls in **x = 3.75 – 9.75 m**, against a valid span of 2.75 – 59.75 m.

⛔ **THE "RIG-CORRELATED" HALF IS INHERITED, NOT MEASURED HERE — say it that way.** What is
MEASURED is a **bimodal** black strip on 74/139 clips. Attributing it to the two front-wide rigs
requires the per-clip **f-theta intrinsics** (`cy ~543` / `~755`), which are **not on this box**, so
I could not close it. The nearest proxy available — the mount **extrinsics** — does **not** cleanly
separate the two groups (`raw/trunk_q1g_label_vs_mount.json`: camera height mean **1.3175 m**
without a strip vs **1.4044 m** with, ranges 1.2618–1.4799 and 1.2131–1.6622, heavily overlapping;
50 vs 70 distinct chunks). ⚠️ That neither confirms nor refutes the rig attribution — extrinsics and
intrinsics are different tables, and an unobserved region is an **intrinsics** property. **What
would settle it:** `calib.observed_report(intr, FRAME_416x1024)` per clip against the corpus's
`camera_intrinsics`, which is exact from the ray map and needs no decode. ⭐ **The bimodality is
what matters operationally either way:** whatever names the two groups, 53 % of clips carry a
constant block that the other 47 % do not.

⚠️ **Scope, stated honestly.** This is MEASURED on the **eval-139** cache. The corpus rebuild at
416 × 1024 has not happened (it waits on SAM3), so the train-side fraction is **NOT ESTABLISHED**;
the parity-corpus rig split (812 A / 2,188 B over 3,000 clips, `calib.py:1251-1253`, INHERITED)
suggests it will not be smaller. What would settle it: run `code/q1d_rig_black_strip_census.py`
against the corpus cache when it exists, or — with zero decode — `calib.observed_report(intr,
FRAME_416x1024)` per clip, which is exact from the ray map.

⚠️ **What this finding does NOT say.** It does not say 416 × 1024 is the wrong geometry. The PI
ruled it on 2026-09-17 and the azimuth argument behind it is unaffected. It says the **height**
decision carries a cost the ruling's own cost table does not list, that the cost is known to this
programme by name, and that it has a cheap mitigation (below) which is currently **not applied**.

---

## ⛔ F1c — THE DISCRIMINATING EXPERIMENT FOR F1: THE GROUP SPLIT IS **~98 % DECODABLE FROM AN 8×16 THUMBNAIL**, AND A **CROP LADDER SHOWS THE STRIP CARRIES PART OF IT**

**Evidence class: MEASURED** (ours). Artifacts `raw/trunk_q1f_rig_decodability.json`,
`raw/trunk_q1h_rawfloor_ladder.json`, `raw/trunk_q1i_strip_sufficiency.json`; instruments
`code/trunk_q1f_rig_decodability.py`, `code/trunk_q1h_rawfloor_crop_ladder.py`,
`code/trunk_q1i_strip_sufficiency.py`.

F1 establishes a bimodal strip; F2b establishes that it reaches the planner's feature. Neither shows
anything **uses** it. `CLAUDE.md` RULE ZERO says a diagnosis is not a deliverable, so the cheapest
discriminating experiment was run and then re-run when the first instrument turned out to be the
wrong one.

### First pass, and why it did not settle anything

Logistic probe on the trunk's pooled stride-32 feature predicting the group label
(`n_rows_fully_black > 0`), with a matched crop as the intervention. **C selected on the FIT half
only** by inner 4-fold CV, scored on the held-out half, 8 random splits. **n = 139** (74 / 65),
**d = 512** — n ≪ d, i.e. underpowered by construction and stated as such.

| arm | balanced accuracy |
|---|---|
| **CONSTANT-ONLY control** | **0.5000 ± 0.0000** — the no-information value, exactly |
| **labels permuted** | **0.4800 ± 0.0445** — chance |
| trunk `pooled`, FULL | 0.8720 ± 0.0291 |
| trunk `pooled`, NOTOP (strip kept) | 0.8783 ± 0.0410 |
| trunk `pooled`, NOBOTTOM (strip removed) | 0.8483 ± 0.0496 |
| ⛔ **RAW-PIXEL FLOOR** (8×16 average pool, d = 384, **no network**) | **0.9837 ± 0.0169** |

Both mandated controls pass exactly. ⛔ **But the RAW-PIXEL FLOOR BEATS EVERY TRUNK ARM.** By
`CLAUDE.md`'s own probe rule — *"a learned representation that does not beat raw input has added
nothing"* — the trunk's pooled feature is **the wrong instrument for this question**, and its arms
cannot resolve it. ⚠️ A second flaw was mine: `NOBOTTOM` removed **32** rows while the strip is
**27–37**, so **20 of the 74 positives kept 1–5 black rows** and the intervention was incomplete.

### Second pass — the sensitive instrument, and a crop ladder deep enough to work

Same probe, on the **raw-pixel floor** (free: pooling, no forward), with the crop run as a **ladder**
and a matched top-crop at every rung. n = 139, d = 384, 10 splits.

| rows removed | **BOTTOM** (strip removed) | **TOP** (strip kept) |
|---|---|---|
| 0 | **0.9786 ± 0.0181** | — |
| 32 | 0.9417 ± 0.0328 | 0.9688 ± 0.0232 |
| 48 | 0.9412 ± 0.0267 | 0.9705 ± 0.0191 |
| **64** (past the 37-row max) | **0.9051 ± 0.0261** | **0.9776 ± 0.0122** |
| **96** | **0.8854 ± 0.0268** | **0.9836 ± 0.0171** |

Controls: **CONSTANT-ONLY 0.5000 ± 0.0000** exactly; **labels permuted 0.5062 ± 0.0307**.

⇒ **The bottom crop degrades decodability monotonically (0.979 → 0.885) while the matched top crop
does not (0.969 → 0.984).** At k = 64 — past the deepest measured strip — the gap is **0.0725**,
about 3× the larger sd; at k = 96 it is **0.0982**. ⇒ **the strip carries part of the group
identity.** ⛔ **But decodability does not collapse to chance: 0.885 with 96 bottom rows gone.** The
two groups also differ in **scene content** — unsurprising for a rig/mount property, since rigs
track collection campaigns (50 vs 70 distinct chunks, `raw/trunk_q1g_label_vs_mount.json`).

⚠️ **One confound the ladder cannot remove, stated rather than glossed:** the two crops are matched
on **size**, not on **information** — the bottom of a forward camera is near road, the top is sky —
so part of the bottom/top asymmetry is expected on content grounds alone.

### Third pass — the sufficiency control that removes that confound

Take only the **65 clips with NO strip** (one homogeneous group, same campaigns, same scenes), paint
an artificial 31-row black bottom strip on a **random half**, and probe from raw pixels. The label is
then, by construction, *"I painted this one"* and nothing else.

| arm (n = 65, d = 384, 10 splits) | balanced accuracy |
|---|---|
| ⛔ **PAINTED STRIP — the fabricated label** | **0.8987 ± 0.0462** (min 0.8125) |
| **CONTROL — the SAME random labels, UNPAINTED images** | **0.5041 ± 0.0351** — chance |
| **CONTROL — painted images, labels permuted** | **0.5117 ± 0.0540** — chance |
| **CONTROL — constant only** | **0.5000 ± 0.0000** — exactly the no-information value |

⇒ ⛔ **A 31-row black strip is BY ITSELF enough to make an otherwise arbitrary group ~90 %
linearly identifiable from an 8 × 16 thumbnail**, with all three controls at chance. The strip is
not merely present, not merely correlated — it is a **usable feature at essentially zero capacity**,
and **0.899 is a LOWER bound**: the 8 × 16 pool dilutes 31 black rows into a 52-row bin, while a
conv stem at stride 2 reads them directly. ⇒ **F1's shortcut concern is no longer a hypothesis.**

⭐ **Taken together the three passes give a clean verdict.** *Necessity:* removing the strip costs
decodability (0.979 → 0.885) while an equal top crop does not (0.969 → 0.984) — so it carries part
of the real signal. *Sufficiency:* the strip alone manufactures a 0.899-decodable label out of
nothing, against three chance controls. *Residual:* the two real groups stay ~0.885 separable with
96 bottom rows gone, so scene content carries the rest and cropping the strip does **not** make the
groups indistinguishable — it removes the **free, constant, zero-capacity** route to telling them
apart.

**Recommendation is unchanged either way, and that is the practical point:** cropping the strip costs
nothing and removes one legible, constant, group-correlated feature from the input regardless of how
the attribution resolves. **What would settle the attribution completely:** build the label from
**optics** rather than from pixels — `calib.observed_report(intr, FRAME_416x1024)` per clip against
the corpus `camera_intrinsics` (not on this box) — which also removes the mild circularity in the
present label (on FULL and NOTOP the probe can read the very thing it is labelled by; `NOBOTTOM`
and the painted arm are the arms free of that).

---

## ⛔ F2 — THE LIFT'S `observed` MASK IS WIRED, DOCUMENTED, AND **NEVER PASSED** IN THE refcv6 PATH, SO BEV CELLS THAT LAND IN F1's BLACK STRIP ARE MARKED **VALID**

**Evidence class: MEASURED** (ours). Artifacts `raw/q2b_lift_unobserved.json`,
`raw/trunk_q2c_persistent_zero.json`; instruments `code/q2b_lift_vs_unobserved.py`,
`code/trunk_q2c_persistent_zero_mask.py`.
⚠️ **Read the magnitude section before acting on this: the code defect is real, the measured BEV
cost is SMALL, and F1 — not F2 — is the lever with the larger effect.**

`stack/tanitad/models/bev_lift.py:32-35` states the requirement in the module's own docstring:

> *"Validity (per height): IN FRONT of the camera … **optionally AND an observed-pixel mask of the
> cache frame (rig B leaves ~8.9 % of the 256 x 640 frame black, `calib.py:1245-1248`)**."*

and `build_lift_geometry(..., observed: Tensor | None = None)` implements it at
`bev_lift.py:150, 168-177`.

The refcv6 call site passes **four** keyword arguments and `observed` is not one of them:

```
stack/tanitad/models/refcv6_perception_branch.py:271-272
    g = build_lift_geometry(e, frame=self.frame, stride=self.stride,
                            heights_m=self.heights_m, grid=self.grid_spec)
```

and that is the only constructor of lift geometry in the training path
(`stack/scripts/refc_v3_train.py:6275-6276` builds `LiftGeometryBank(_ptable, frame=_pframe,
stride=int(_pcfg.stride))` — no mask argument exists on the bank either, so an operator cannot
supply one).

**Same-breath control for the absence:** the grep that returns the two `build_lift_geometry` call
sites (`refcv6_perception_branch.py:62` import, `:271` call) returns non-zero on the same
invocation that returns zero `observed=` occurrences outside `bev_lift.py` and `stack/tests/`.

⇒ **Consequence.** A BEV cell whose ray lands in the unobserved strip passes `in_front ∧ in_hfov ∧
in_rows` and is marked `valid=True`. It is then filled by `F.grid_sample` from trunk features
computed over the constant −2.12/−2.04/−1.80 block, and multiplied in as real evidence
(`bev_lift.py:262-268`). Meanwhile the module's **designed** handling for "no camera saw this
cell" — the learned `unobserved` embedding at `bev_lift.py:241, 269-270` — fires only where **no
height is valid**, i.e. never for these cells. So the one mechanism built to say *"unknown"* is
bypassed exactly where the answer is unknown.

⭐ **This is the same shape as advisory class C ("a bottleneck that was not a bottleneck") and
class G (a filter whose semantics are its own producer's, not its consumer's): the validity test
answers *"is this cell inside the declared frame?"* while the consumer needs *"did a sensor see
it?"*.** At 256 × 640 those two questions differed by ~8.9 % on rig B; at 416 × 1024, F1 measures
the disagreement at up to 18.6 % of the frame.

### F2's MAGNITUDE, measured — and it is small

⛔ **A correctness gap is not automatically a large effect, and this one is not.** Measured over all
139 clips with the dataset's own per-clip `sensor_extrinsics`
(`…/2026-09-06-refcv4b-landing/raw/extrinsics141.json`, 139/139 covered, 0 missing):

| | |
|---|---|
| BEV cells the lift fills at 416 × 1024 | **6,835.6 / 7,680 = 89.00 %** (min 88.70, max 89.52) |
| clips with any cell sampling the strip | **74 / 139** — exactly F1's set |
| cells touching the strip, per affected clip | mean **13.0**, max **49** |

Because a "fully-black ROW" test misses the curved boundary's partial rows, this was tightened with
a **persistent-zero** mask (a pixel exactly `(0,0,0)` in **all 12** sampled frames of a clip — scene
content does not hold that across 12 frames), with the row test as a lower rail and a one-frame
pixel test as an upper rail. On the five worst clips:

| | |
|---|---|
| persistent-zero pixels | **10.0 – 11.1 %** of the frame |
| BEV samples marked valid inside it | **51 – 71** of ~27,150 = **0.19 – 0.26 %** |
| BEV cells touching it | **51 – 70** of 7,680 |
| ordering LOWER ≤ MEASURED ≤ UPPER | holds on **7 / 7** |
| **CONTROL** — two strip-free clips | **0 / 0 / 0** on all three estimators |

**And the option demonstrably changes the answer**, which is the control that proves the seam is
live rather than notional: passing `observed=` for the worst clip moves valid samples
**27,148 → 25,383 (−6.5 %)** and valid cells 6,816 → 6,784.

⚠️ **But the affected cells are not distributed — they are all in the NEAR FIELD, and the near
field is where this programme's largest known gap lives.** MEASURED on the worst clip
(`raw/trunk_q2e_which_bev_rows.json`):

| | |
|---|---|
| BEV grid span | x = **0.25 – 59.75 m** ahead (120 rows × 0.5 m) |
| nearest cell the lift can fill at all | **x = 2.75 m** (nearer cells are behind the camera, `in_front = False`) |
| all valid cells | x 2.75 – 59.75 m, **mean 33.3 m** |
| **the 70 cells touching the strip** | **x = 3.75 – 9.75 m, mean 5.46 m** |

⇒ every affected cell sits in the **3.75 – 9.75 m band** — the headway / time-gap / TTC band that
`CLAUDE.md`'s binding four-family rule singles out (*"88.7 % of our oracle gap is longitudinal"*).
⇒ **Honest ranking: fix F2 because it is one keyword, because "valid" currently means the wrong
thing, and because the 0.2 % is concentrated exactly where the LONGITUDINAL family reads — not
because 0.2 % of BEV samples will move ADE.** The larger channel is F1 itself — 10–11 % of every
IMAGE on 53 % of clips is a constant block, and it reaches the planner (F2b) as well as the map
head.

### F2b — and the strip reaches the PLANNER, not only perception

The planner's scene vector is `pooled = s32.mean(dim=(2,3))` (`timm_trunk.py:590`), a **global**
spatial mean over the 13 × 32 stride-32 map, so the strip's ~1 of 13 rows enters it. Measured by
**intervention** on six clips that have **no** strip (blacking their bottom 31 rows), in eval mode
so BatchNorm cannot confound (see F6). Artifact `raw/trunk_q1e_strip_planner.json`:

| arm | relative move of the 512-d `pooled` |
|---|---|
| **INTERVENTION** — bottom 31 rows blacked | **0.0561** (0.0495 – 0.0636) |
| **C1** — equal-area strip at the TOP (observed sky) | 0.0543 |
| **C2** — equal-area VERTICAL strip at the left edge | 0.0677 |
| **C3** — null, same frame twice | **0.0 exactly** |
| **reference** — distance between two DIFFERENT clips | **0.3132** (0.2307 – 0.4230, 15 pairs) |

⇒ the strip moves the planner's input by **17.9 % of the typical between-clip distance** — a real
channel, not a rounding error. ⚠️ **But the matched controls say the magnitude is NOT special:**
blacking an equal area anywhere costs about the same. So the measurement establishes the *channel*
and its *scale*; it does **not** establish that a network exploits it. That is F1's HYPOTHESIS
(*"this model eats shortcuts"*, `calib.py:1036`), and the discriminating experiment for it is F1c
below.

**The fix is one keyword.** `build_lift_geometry` already accepts the mask, and the mask is exactly
computable per clip with no decode via `calib.cylindrical_grid(intr, intr.height, intr.width,
frame)[1]` — the same call `cylindrical_rectify` makes when it builds the pixels
(`calib.py:989`), so there is no second spelling of the geometry.

---

## ⛔ F3 — CLASS B3, LATENT: `frame_for_model` DERIVES `f_ref` FROM THE **WIDTH**, WHICH IS WRONG FOR THE TWO DECLARED **RIG-CLEAN** GEOMETRIES — AND THOSE ARE EXACTLY THE FRAMES F1 POINTS A READER TOWARD

**Evidence class: MEASURED** (ours). Artifact `raw/q2_lift_geometry.json`; instrument
`code/q2_lift_geometry_probe.py`.

The lift's frame in training comes from `refc_v3_train.py:6265`:
`_pframe = _perc.frame_for_model(model)`, which is

```
stack/tanitad/models/refcv6_perception_branch.py:470-473
    def frame_for_model(model):
        h, w = model.cfg.core.encoder.image_hw()
        return frame_for_width(int(w), int(h))
```

and `trunk_shapes.frame_for_width` (`trunk_shapes.py:73-82`) scales
`f_ref = 305.5774907364391 * width / 640`. That is correct **only for frames obtained by
RESAMPLING at a new width**. The two rig-clean frames are **centred pixel SLICES** of the 256×640
frame and therefore keep the parent's `f_ref` exactly (`calib.py:1256-1264`,
`calib.centred_subframe` at `:1062-1086`).

Cross-check of every geometry declared in `refc_v3_train._agent_cam_frames()`
(`refc_v3_train.py:1316-1346`) against what `frame_for_model` produces for the same `(h, w)`:

| geometry | declared `f_ref` | `frame_for_model` | Δ | declared HFOV | derived HFOV | max sampling error |
|---|---|---|---|---|---|---|
| 256 × 640 | 305.5774907 | 305.5774907 | **0.0** | 120.0000° | 120.0000° | 0 px |
| 256 × 1024 | 488.9239852 | 488.9239852 | **0.0** | 120.0000° | 120.0000° | 0 px |
| **416 × 1024** (the PI's geometry) | 488.9239852 | 488.9239852 | **0.0** | 120.0000° | 120.0000° | 0 px |
| **176 × 624** (rig-clean) | 305.5774907 | **297.9380535** | **−7.639** | **117.0°** | **120.0°** | **≈ 7.2 px at the edge** |
| **128 × 576** (rig-clean, strict) | 305.5774907 | **275.0197417** | **−30.558** | **108.0°** | **120.0°** | **≈ 28.8 px at the edge** |

**2 of 5** declared geometries mismatch; **3 of 5** agree exactly (the same-breath positive control
— the probe is not reading nothing).

The failure is the advisory's class B3 *verbatim* — *"correct intrinsics applied at the wrong
resolution"*, which on REFe produced **31.34° instead of 62.85°**. Here it manufactures a **120°
field for a 108° frame**: a rig point at true azimuth 54° would be looked up 28.8 px away from the
pixel that shows it, i.e. **1.8 stride-16 feature cells** — and it is silent, because the lift
consumes `f_ref` + projection only (see F4) and nothing downstream can tell.

⚠️ **This is LATENT, not active.** refcv6 is pinned at 416 × 1024 by `SPEC_REFCV6_V2.md` §12, where
the derivation is exact. ⛔ **But it becomes active the moment anyone acts on F1**, because the
programme's own answer to a rig-correlated black strip is precisely `PHYSICALAI_RIG_CLEAN_176x624`
/ `_128x576`. The two findings interlock: **the mitigation for F1 walks straight into F3.**

⭐ **The fix is to stop deriving and start looking up.** `_agent_cam_frames()` is already the table
that declares each geometry's `f_ref`, and `refc_v3_train.py:1461-1472` already **REFUSES** an
undeclared geometry with an explicit message about the cylindrical/pinhole trap. `frame_for_model`
should read that same table and refuse, rather than re-deriving a second spelling of a camera
constant — which is the defect `trunk_shapes.py:85-91` and `refc_v3_train.py:1337-1345` both go out
of their way to prevent one module over.

---

## ✅ F4 — CLASS A DOES **NOT** BITE refcv6: THE IMAGENET STATISTICS ARE REALLY APPLIED, ON RGB, IN `[0,1]`, EXACTLY ONCE

**Evidence class: MEASURED** (ours). Artifacts `raw/q1_normalisation.json`,
`raw/q1b_real_cache.json`; instruments `code/q1_normalisation_probe.py`,
`code/q1b_real_cache_endtoend.py`.

**The declared statistics.** `timm`'s `resnet34.a1_in1k` `default_cfg` reads
`mean (0.485, 0.456, 0.406)`, `std (0.229, 0.224, 0.225)`. `timm_trunk.IMAGENET_MEAN/STD`
(`stack/tanitad/models/timm_trunk.py:118-119`) are **identical** (asserted in the probe, not
eyeballed).

**Applied, and applied once.** `TimmResNetTrunk.normalise` (`timm_trunk.py:538-548`) is reached
from `forward_features` (`:575-576`); `norm_calls` after one forward = **1**. The only opt-out is
the explicit `already_normalised=True`, which **no production caller passes** — a repo-wide grep
returns it in `timm_trunk.py` itself and in `stack/tests/test_refcv6_trunk.py:176` only, with the
non-zero same-breath control being the 29 `def`s the same grep family finds in that module.

**Measured at the pretrained stem, not inferred.** A forward hook on the backbone's **own**
`conv1` (`in_channels = 3`, so the shared-weight path really presents 3 channels) records, on a
**real** 416 × 1024 PNG frame from the gated eval cache:

| | per-channel mean | per-channel std |
|---|---|---|
| **what the stem actually received** | **−0.595159 / −0.611796 / −0.493789** | **0.548452 / 0.594059 / 0.685980** |
| independently derived, IF normalised (same crop) | −0.595160 / −0.611795 / −0.493789 | 0.548458 / 0.594065 / 0.685987 |
| independently derived, if **NOT** normalised | +0.252193 / +0.253250 / +0.239354 | 0.164140 / 0.189577 / 0.216962 |

Agreement to 6 dp with the normalised prediction, and nowhere near the un-normalised one.
`normalise()` also matches hand arithmetic with **max abs diff exactly 0.0**, and the mean/std
buffers are **tiled per frame** (`(mean,)*K`, `timm_trunk.py:497-504`), so history frame 2 gets its
own ImageNet triple rather than sharing one.

**The deliberate regression goes RED.** `imagenet_norm=False` reads `norm_calls = 0`, stem mean
`0.4965/0.4951/0.4969` (raw `[0,1]`), and displaces the stride-16 features by
**rel L2 0.6985 / cos 0.7677**. So the probe discriminates; it is not measuring a tautology.
Two more traps, each constructed and each moving the features:
`[0,255]` input → rel L2 **0.6987**; channel-reversed (BGR) input → rel L2 **0.7268 / cos 0.7375**.
⚠️ These displacements are on a **uniform-random** input and are NOT comparable to REFe's 0.562 /
0.194 on natural images — they establish *discrimination*, not *magnitude*.

**Channel order is RGB, declared at four sites, none of them OpenCV.**
1. PyAV video decode: `vframe.to_ndarray(format="rgb24")` — `stack/scripts/v2_compressed.py:88`.
2. Encode: `tvio.encode_png` / `encode_jpeg` — `v2_compressed.py:143-144` (torchvision is
   RGB-native).
3. Decode on the build-verify path: `mode=tvio.ImageReadMode.RGB` — `v2_compressed.py:185`.
4. Decode in the **trainer's** loader: `mode=tvio.ImageReadMode.RGB` —
   `stack/tanitad/data/v2_dataset.py:137` and `:144`.
`cv2` does not appear anywhere in `stack/tanitad/data/` or in the corpus builder; the same-breath
control shows `import cv2` present in `stack/scripts/geom_sanity.py` and `ph0_pilot.py`, so the
grep was live.

**Value range is `[0,1]`.** Decode yields `uint8 [0,255]`; the trainer's single ingest point
`frames_to_device` (`refc_v3_train.py:3088-3117`) converts with a **0-dim device-tensor** divisor,
measured here on a real frame to give `min 0.0 / max 1.0`.

⚠️ **One thing worth the PI's attention, and it is NOT a bug.** On real corpus frames the
per-channel raw means are **0.252 / 0.253 / 0.239** against ImageNet's **0.485 / 0.456 / 0.406**,
so the normalised input sits about **one ImageNet σ below zero** (measured stem mean ≈ −0.6 on the
crop, ≈ −1.0 over the full frame) with std ≈ 0.55–0.96 rather than ≈ 1. The corpus is genuinely
darker and lower-contrast than ImageNet. ⛔ The correct response is **not** to re-fit the
statistics — the checkpoint's declared operating point is the one its weights were trained at, and
substituting a corpus-fitted one is the *"self-consistent but not theirs"* failure the advisory
names. It is stated here so that a later "the prior is not helping" observation has this measured
fact beside it rather than being re-derived as a discovery.

---

## ✅ F5 — CLASS B1/B2 DO NOT BITE: THERE IS NO ORPHAN POSITIONAL TENSOR, AND THE LIFT REALLY CONSUMES PER-CLIP GEOMETRY

**Evidence class: MEASURED** (ours). Artifact `raw/q2_lift_geometry.json`.

**B1 (a tensor with no checkpoint counterpart) — run by the advisory's own method, not argued.**
REFe's orphan was found by parameter arithmetic against the checkpoint. The same arithmetic was run
here against the real downloaded `model.safetensors` in the HF cache
(`code/trunk_q2d_orphan_tensor_arithmetic.py`, `raw/trunk_q2d_orphan_arithmetic.json`):

| | `resnet34.a1_in1k` | `resnet101.a1_in1k` |
|---|---|---|
| checkpoint tensors | 218 | 626 |
| backbone tensors (params + buffers) | 180 | 520 |
| **backbone tensors with NO checkpoint key** | **0** | **0** |
| **CONTROL** — the dropped classifier found in the unmatched set | ✅ `fc.weight`, `fc.bias` | ✅ `fc.weight`, `fc.bias` |
| backbone params | **21,284,672** | **42,500,160** |
| refcv6 params ON TOP (`fuse16`/`fuse32`) | **983,808** | **15,731,712** |
| those are identity-initialised | ✅ | ✅ |

**Zero orphans on both backbones**, with a same-breath control proving the comparison really read
the checkpoint (it correctly reports `fc.*` as present in the file and absent from the
`features_only` backbone). The two backbone counts reproduce ERRATUM-1's **21.285 M** and
**42.500 M** exactly, and the fusion costs reproduce `timm_trunk.py:189-193`'s **0.98 M** / **15.7 M**.
Backbone weights are separately verified against the real download by `_assert_pretrained_loaded` /
`_PINNED_STATS` (`timm_trunk.py:129-137`, `:452-453`), keyed on `|conv1.w|.sum()` = 1279.8538 /
1402.7781. ⇒ **class B1 has nothing to find here.**

**B2 (does the "geometry" consume geometry, or an index?).** Built the lift geometry three times
at `FRAME_416x1024`, stride 16:

| perturbation | max |Δrow| | max |Δcol| | mean |Δrow| |
|---|---|---|---|
| mount height −0.25 m | **21.84 px** | **52.11 px** | 12.46 px |
| pitch `qx` +0.02 | **20.63 px** | 6.56 px | 15.19 px |
| **CONTROL — same pose twice** | **0.0000** | **0.0000** | **0.0000** (grids bit-identical: `torch.equal` True) |

⇒ the lift is a genuine projection through the clip's own `sensor_extrinsics`, not an index map.
`SPEC`'s *"parameter-free geometry"* is **accurate**: the learned parts are a 1×1 conv and one
`unobserved` embedding (`bev_lift.py:240-242`); everything else is `rig_to_cam` +
`project_cam_to_frame` in float64.

⚠️ **Stated precisely, because the exact scope matters for F3:** the lift consumes the **per-clip
extrinsics** and the **frame's `f_ref` + projection**. It does **not** consume the per-clip f-theta
**intrinsics** — those were consumed once, at cache build time, by `calib.cylindrical_rectify`
(`calib.py:960-998`). That is the right design, and it is also precisely why a wrong `f_ref`
(F3) is undetectable downstream: `f_ref` is the *only* intrinsic left in the loop.

**The feature-coordinate convention holds for the timm trunks too — MEASURED, not inherited.**
`bev_lift.REFC_FEATURE_CENTRE_OFFSET_PX = 0.0` (`bev_lift.py:71`) is derived and pinned for
REF-C's own `ResNetEncoder`; refcv6 uses a `timm` ResNet, so it was re-measured by **impulse
forward** through the real backbones (one lit pixel, find the peak feature column):

| backbone | stride 16 | stride 32 |
|---|---|---|
| `resnet34.a1_in1k` | offset **0 px** at all 4 impulse columns | offset **0 px** at all 4 |
| `resnet101.a1_in1k` | offset **0 px** at all 4 | offset **0 px** at all 4 |

16 independent measurements, all 0. The constant is correct for both arms.

---

## ⛔ F6 — `fuse_identity_init`'s "BIT-IDENTICAL" CLAIM IS TRUE IN **EVAL** AND FALSE IN **TRAIN**, WHICH IS THE ONLY REGIME THE K=1 vs K=3 COMPARISON RUNS IN

**Evidence class: MEASURED** (ours). Artifact `raw/trunk_q5b_guard_bn.json`;
instrument `code/trunk_q5b_guard_and_bn.py`.

`timm_trunk.py:300-306` states the claim that makes image history an attributable lever:

> *"⛔ `fuse_identity_init` makes history REMOVABLE. With it, a freshly built K-frame trunk emits
> **bit-identically** what the single-frame trunk emits … So the first training step starts from
> today's model, and **any later difference is LEARNED rather than an artefact of having added
> parameters**. Without it, '3 frames beat 1 frame' would be confounded with 'a randomly-initialised
> 1x1 conv was inserted'."*

Built a K=3 and a K=1 trunk from the same ImageNet weights and compared the K=3 output against the
K=1 output **on the newest frame**, in three regimes:

| regime | s16 max abs diff | s32 max abs diff | bit-identical? | backbone BNs in training mode |
|---|---|---|---|---|
| **eval** | **0.0** | **0.0** | YES | 0 / 0 |
| **train** (the DEFAULT: `TimmTrunkConfig.frozen_bn = False`) | **23.5636** | **8.8612** | ⛔ **NO** | 36 / 36 |
| **train + `frozen_bn=True`** | **0.0** | **0.0** | YES | 0 / 0 |

The third row is the **discriminating control**: freezing BatchNorm in train mode restores exact
bit-identity, which isolates the mechanism to BatchNorm and rules out the fusion arithmetic.

**The mechanism, and it is already documented one knob over.** The shared-weight path folds K into
the batch — `per = x.reshape(b, k, 3, …).reshape(b * k, 3, …)`, `timm_trunk.py:583-586` — so in
train mode every BatchNorm normalises over **B·K** images instead of **B**, and the extra 2B are
history frames at ~10 Hz, i.e. highly correlated with the newest. The same module already MEASURED
this exact mechanism for the chunking lever:

> `timm_trunk.py:178-182` — *"chunking alone changes BN and MEASURED a **−40 % shift in `ga_trunk`**
> on resnet34 — a config that fits but is a **DIFFERENT ARM**. With BN frozen, chunked and unchunked
> agree to 7.2e-6."*

⇒ **The K=1 vs K=3 comparison SPEC §10.3 sets up is confounded at step 0 by exactly the variable the
chunking lever was refused for.** It is the `--v2` conflation shape: two levers (history, and a K×
BatchNorm batch) on one axis.

⚠️ **Precisely scoped.** The docstring is not wrong, it is **unqualified**: it holds in eval and
under `frozen_bn`, and `frozen_bn` defaults **False** (`timm_trunk.py:182`), reaching the trunk
through `refc.py:1453` `trunk_frozen_bn` (also default False). ⭐ There is a second-order hazard in
the same place: the memory-levered arm **must** set `frozen_bn` (`timm_trunk.py:524-530` refuses
`chunk_ckpt` without it), and that arm is the one that fits on the 8 GB dev box
(`SPEC_REFCV6_V2.md` §12.5 records resnet101 OOM-ing at batch 1). So a dev-box rehearsal and a pod
run can differ in this variable without anyone choosing it.

⭐ **The cheapest fix is already in the codebase: run BOTH K arms with `--trunk-frozen-bn`** (and,
for an ImageNet-vs-random knockout, the `recalibrate_bn_` procedure at `timm_trunk.py:647` that was
written for exactly this class of two-variable comparison). Measured cost: zero — the frozen-BN row
above is bit-identical, not approximately so.

---

## OK F7 — THE EGO-HISTORY FUTURE-READ GUARD **CAN GO RED**, PROVEN BY CONSTRUCTION — WITH ONE NAMED BLIND SPOT

**Evidence class: MEASURED** (ours). Artifact `raw/trunk_q5b_guard_bn.json`.

SPEC §10.3: *"⛔ Nothing from the future may enter it — enforced by a test that fails if a future
index is read."* The test is
`stack/tests/test_refcv6_tactical.py:1243 test_a_FUTURE_index_can_NEVER_enter_the_ego_history`.
It is **not skipped** (`pytest -k "FUTURE or ego_poses"` → **12 passed, 0 skipped**), and
`stack/tests/test_refcv6_trunk.py:441` already ships its own paired `..._GOES_RED` arm for the
encoder. The model-level guard had **no** such arm, so three mutations were constructed and run
against its exact assertion:

| mutation | plan bit-identical under a +1e4 future corruption? | guard |
|---|---|---|
| **baseline** (unmutated) | yes, max abs diff **0.0** | green |
| **MUT1** — the ENCODER ignores `n_past` | **yes**, 0.0 | ⚠️ **stays green — blind spot** |
| **MUT2** — the CHANNEL BUILDER ignores `n_past` | **no**, **0.2953** | **RED** |
| **MUT3** — a CENTRED difference at the last past step | **no**, **0.3991** | **RED** |

MUT3 is the *specific* leak `ego_history.py:39-42` names (*"a centred difference at the last past
step would read `v[n_past]` — the first FUTURE sample"*), and it goes red. The guard is alive.

**The blind spot is real and benign, and it should be written down rather than discovered.** The
model calls `ego_channels_from_poses(ego_poses, n_past)` and then `self.ego_hist(ch)` **with no
`n_past`** (`stack/tanitad/refs/refc.py:4265-4269`). So the single load-bearing slice is
`ego_history.py:92` (`past = poses[:, :n]`); the encoder's own slice (`ego_history.py:176`) is a
no-op on an already-sliced `[B, n_past, 3]`. The module docstring's design argument
(`ego_history.py:17-26` — *"the encoder does not take 'the past'; it takes the whole sequence and
the number of leading steps that are past"*) describes a property the **production call site does
not exercise**. ⛔ That matters if the channel builder is ever inlined or bypassed: the encoder's
defence would then be the only one, and it has never been exercised by the model-level guard.

**Same-breath liveness control:** perturbing the PAST ego (`poses[:, :n_past] += 3.0`) moves the
plan by **0.0425**, so the ego channel genuinely reaches the output and the bit-identity above is
not the trivial result of a dead path.

**And the analytic control passes.** `ego_channels_from_poses` on a hand-built ladder
(v = 0,1,3,6,10,**99**; n_past = 4) returns accel **[0, 10, 20, 30]** — exactly the backward
differences — where a centred difference at the last past step would have returned **35.0**, and
the future value 99 appears nowhere. That is an **analytic target**, not a re-run of the producer's
own arithmetic.

⚠️ One latent skip hazard, not currently biting: `needs_ego_hist`'s predicate
(`test_refcv6_tactical.py:1186-1196`) is `try: … except Exception: return False`, so **any** build
failure would turn the whole admissibility block into a silent skip rather than a failure. It did
not skip here (12 passed), but a bare `except Exception` guarding an admissibility test is the
advisory's class F shape.

---

## OK F8 — Q6: CHANNEL COUNTS COME FROM `feature_info`, AND BOTH BACKBONES BUILD AT 416 × 1024

**Evidence class: MEASURED** (ours). Artifact `raw/trunk_q5q6_history_channels.json`;
instrument `code/trunk_q5q6_history_and_channels.py`.

| backbone | `feature_info` s16 / s32 | **actual forward** s16 / s32 | shapes at 416×1024 | `TrunkSpec` agrees |
|---|---|---|---|---|
| `resnet34.a1_in1k` | **256 / 512** | **256 / 512** | **26×64 = 1664** / **13×32 = 416** | yes |
| `resnet101.a1_in1k` | **1024 / 2048** | **1024 / 2048** | **26×64 = 1664** / **13×32 = 416** | yes |

The forward's tensor shapes are an **independent** derivation of the same quantity (running the
network, not re-reading `feature_info`), and they agree. `SPEC_REFCV6_V2.md` §10.2's table and
ERRATUM-1's `1024 / 2048` and `256 / 512` both reproduce exactly. `TrunkSpec.from_timm`
(`trunk_shapes.py:173-197`) refuses a model with no `feature_info` rather than guessing
(`:186-192`), and `TimmResNetTrunk.__init__` refuses anything that is not the `[16, 32]` reduction
pair (`timm_trunk.py:470-475`). The token counts match §11 R1's **1664 / 416** exactly, which is
also §12.2's own proof that the cost table was already computed at 416.

**Q5, image history — MEASURED, not read.** A forward hook on the backbone's stem during one
K=3 trunk forward with B=2 records: **1** stem call, **input channels {3}**, **input batch {6}**
(= B·K), **1 distinct stem weight tensor**. So it is genuinely *K passes of a 3-channel ImageNet
stem through ONE set of weights, folded into the batch* — **not** a 9-channel inflated stem. Fusion
is `TemporalFuse` at **both** strides, after the backbone (`timm_trunk.py:487-493`), `concat1x1`,
and the default `mode` is `"shared"`. The `inflate` arm is present as the registered knockout and
reproduces the 3-channel response on a repeated-frame input to **2.86e-4** (fp32), as its docstring
claims. SPEC §10.3 is implemented as written — with the F6 qualification about BatchNorm.

---

## OK / ⚠️ F9 — Q4: THE °/px COMPARISON RE-DERIVED, WITH THE PROJECTION STATED FIRST

**Evidence class: MEASURED (ours)** for our side; **INHERITED** for theirs.
Artifact `raw/trunk_q4_camera_resolution.json`; instrument `code/trunk_q4_camera_resolution.py`.

**Ours is CYLINDRICAL**, so the column is linear in azimuth and `az_max = (W/2)/f_ref` — the
pinhole formula is inadmissible here (`CLAUDE.md` traps; `refc_v3_train.py:1465-1473` refuses an
undeclared frame for exactly this reason).

| | HFOV | °/px | edge/centre density | stride-16 °/col | stride-32 °/col |
|---|---|---|---|---|---|
| 256 × 640 | 120.0000° | 0.187500 | **1.0 (exact)** | 3.00 | 6.00 |
| 256 × 1024 | 120.0000° | **0.117187** | **1.0** | 1.875 | 3.75 |
| **416 × 1024** | **120.0000°** | **0.117187** | **1.0** | **1.875** | **3.75** |

Two independent derivations of °/px — `HFOV/W` and `degrees(1/f_ref)` — agree to < 1e-12.
**SPEC §10.1's `0.1172 °/px` reproduces exactly** (0.117187 → 0.1172 at 4 dp), and the cache's own
manifest reports `deg_per_col 0.11718749999999999` independently. The `f_ref` is a
*horizontal* quantity on a cylindrical frame, so 256×1024 and 416×1024 have the **same** azimuth
resolution; only the vertical field changed.

**Theirs: `0.1367 °/px` is exactly `140 / 1024` — an AVERAGE over a PINHOLE mosaic.**
⚠️ **Provenance:** a repo-wide search finds `0.1367` **only** in `SPEC_REFCV6_V2.md:146` and a
different column of `PI_DECISION_QUEUE.md:596`; **nothing in the record derives it**, and the `~140°`
it comes from carries a tilde in SPEC itself. Same shape as §11's `42.7 %`. The banked primary
(`TanitAD Research Lab/Library/papers/2411.15139_DiffusionDrive-Truncated-Diffusion-Model-for-End-to-End-Auto.pdf`)
records the geometry as *"three cropped forward cams concatenated as a 1024x256 image"* but the
library note carries no FOV figure, so **the 140° is INHERITED** and I did not re-verify it.

⛔ **The two numbers are not the same kind of quantity.** Ours is a *uniform* density
(`projection_density_report` returns `edge_local_density_vs_center = 1.0` **exactly** for a
cylindrical frame, `calib.py:1019-1020`). Theirs is an *average* over a projection whose density is
not uniform: read as one 140° pinhole (`x = f·tanθ`, f ≈ 186.35 px), the centre spends
**0.3075 °/px** and the edge **0.0360 °/px** — an **8.55×** spread. Read as three stitched crops, the
density is piecewise and depends on a stitch that is not in our record.

⭐ **The one statement that needs no assumption about their optics:** both images are **1024 px
wide**; ours covers **120.0000°**, theirs **~140°**; so we spend **1.1667×** more pixels per degree
on average. That is the defensible form of SPEC §10.1's claim, and it equals the ratio SPEC quotes.
Anything sharper — *"we are finer where the road ahead is"* — is plausible but **NOT ESTABLISHED**;
what would settle it is the NAVSIM camera table (intrinsics + crop boxes of `cam_f0/l0/r0`) read
from the primary.

---

## OK / ⚠️ F10 — Q3b: THREE GEOMETRY LITERALS SURVIVE AS DEFAULTS. TWO FAIL LOUD; ONE IS STALE AND SILENT

**Evidence class: MEASURED** (ours). Artifact `raw/trunk_q3_geometry_literals.json`;
instrument `code/trunk_q3_geometry_literals.py`.

SPEC §10.1: *"⛔ Nothing in the code may hard-code 640 / 160 / 40 / 20."* A grep across the four
refcv6 trunk/lift modules finds three surviving literals, all as **defaults**. Each was reached and
exercised rather than judged by reading:

| # | site | value | reached with a 416×1024 build? |
|---|---|---|---|
| 1 | `bev_lift.py:236` `BEVLift(feat_hw=(16, 40))` | `(16, 40)` | ⛔ **LOUD** — `ValueError: feature map is 26x64, the lift geometry was built for 16x40` |
| 2 | `timm_trunk.py:922` `build_timm_trunk(image_hw=(256, 640))` | `(256, 640)` | overridden at the only production call site (`refc.py:1442`, `image_hw=cfg.image_hw()`) |
| 3 | `timm_trunk.py:206` `TimmTrunkConfig.image_hw` | **`(256, 1024)`** | ⚠️ **STALE** (SPEC §12 pins 416×1024 since 2026-09-17) and **SILENT** |

⚠️ **(3) is the one worth fixing.** With the default config and a real 416×1024 input the forward
**succeeds**, while the trunk's *declared* `s16_shape` reads **(16, 64)** against an actual
**(26, 64)** and `grid_shape` reads **(8, 32)** against **(13, 32)**. Nothing raises. The declared
number is what `build_perception_branch` reads to size the lift
(`refcv6_perception_branch.py:466-467`, `image_hw=tuple(enc.s16_shape)`) and what the run record
stamps (`refc_v3_train.py:6281`, `"fmap_s16_hw"`), so a `config.json` could carry a wrong geometry.

⭐ **Two live guards close the gap end to end, and they are the reason this is a defect-in-waiting
rather than an active defect.** (a) `refc_v3_train.py:6518-6524` asserts the encoder geometry
against the **episodes' own frame shape** — *"geometry is asserted against the EPISODES, not against
the flag"* — and refuses with the exact `--image-hw` to pass. (b) The `feat_hw` mismatch above then
raises anyway. ⚠️ I did **not** mutation-prove (a); it is READ, at that file:line. What would settle
it: launch the trainer against the 416×1024 cache with `--image-hw 256 640` and confirm the
`SystemExit`. ⚠️ And `BEVLift(feat_hw=None)` disables check (b) entirely — measured: it accepted a
26×64 map while built for 16×40 without complaint.

**Guard suites run, with skip reasons requested:** `test_refcv6_geometry_agnostic.py`,
`test_frame_416x1024.py`, `test_refcv6_trunk.py` → **67 passed, 0 skipped**. `test_frame_416x1024.py`
carries its own mutation audit in SPEC §12.4 (drop the trainer row → 1 red; re-type `f_ref` → 3 red),
and this review independently confirms both halves it pins: `_agent_cam_frames()[(416,1024)]` **is**
`trunk_shapes.FRAME_416x1024` (identity, not equality) and the cache's `f_ref` matches it bit-exactly.

---

## ⚠️ F11 — SPEC §10.2 CALLS `resnet34` *"DiffusionDrive's own"*, BUT DiffusionDrive APPLIES **NO** IMAGENET MEAN/STD

**Evidence class: INHERITED** — I did not re-read the released code. Source: this programme's own
banked note on the primary, `TanitAD Research Lab/Library/library.json` (entry `2411.15139`,
`papers/2411.15139_DiffusionDrive-Truncated-Diffusion-Model-for-End-to-End-Auto.pdf`), tagged
*"resnet-trunk review 2026-09-15 (verified vs released code)"*:

> *"`image_encoder` lr_mult 0.5 → 3e-4, 3-ep warm-up + cosine (`transfuser_config.py:123-127`,
> `transfuser_agent.py:170-175`); **ToTensor [0,1] with NO ImageNet mean/std**
> (`transfuser_features.py:73`); single frame …"*

`SPEC_REFCV6_V2.md` §2's `normalisation` row mandates *"ImageNet mean/std on the RGB input"* under a
heading that reads *"trained as the papers do"*, and §10.2 designates `resnet34.a1_in1k` as
*"the comparison (run 2) — DiffusionDrive's own"*. F4 confirms refcv6 **does** apply the
normalisation. So on this axis **our `resnet34` arm is not DD's recipe**, and a head-to-head against
DD's published **88.1 PDMS** compares preprocessing as well as architecture.

⭐ **This is not an argument for dropping the normalisation.** Applying a checkpoint's declared
operating point is correct, and `E-SEED-2` measured the cost of omitting it on our own rig. The
finding is that §10.2's comparison arm should **say** it differs here, or carry a third
`no-normalisation` cell if the intent is to reproduce DD. ⚠️ Note also the advisory's warning read
in this direction: *"any backbone comparison measures your preprocessing instead of their encoder"*.
What would settle it to PRIMARY: open `transfuser_features.py:73` in the released DiffusionDrive
repo (not on this box) or the banked PDF's supplementary A.

---

## WHAT I COULD NOT ESTABLISH — named, with what would settle each

⛔ Per `CLAUDE.md` RULE ZERO a refutation is a waypoint, so each open item names the next lever and
says whether it was run or is blocked.

1. **The train-side fraction of F1's strip.** MEASURED on the eval-139 cache only; the 416 × 1024
   **corpus** cache does not exist yet (it waits on SAM3, SPEC §7/§12.3), so the training-side
   number is **NOT ESTABLISHED**. **Blocked on data.** **Unblocker:** re-run
   `code/q1d_rig_black_strip_census.py` on the corpus cache when built — or, **today and with zero
   decode**, `calib.observed_report(intr, FRAME_416x1024)` per clip, which is exact from the ray
   map. I could not run that here: the corpus's per-clip `camera_intrinsics` are not on this box and
   `intrinsics_for_clip` needs the PhysicalAI root.
2. **Whether the two groups ARE the two rigs.** **NOT ESTABLISHED** (F1's INHERITED half). The
   mount extrinsics do not separate them cleanly. **Blocked on the same intrinsics table.** ⭐ This
   is now largely moot for the decision: F1c shows the strip is a usable shortcut whatever names the
   groups.
3. **NAVSIM/DiffusionDrive's true angular density (F9).** The `~140°` is **INHERITED** from SPEC
   §10.1 and nothing in the repo derives its `0.1367`. **Unblocker:** the NAVSIM camera table
   (intrinsics + crop boxes for `cam_f0 / l0 / r0`) from the primary; the PDF is banked at
   `TanitAD Research Lab/Library/papers/2411.15139_DiffusionDrive-Truncated-Diffusion-Model-for-End-to-End-Auto.pdf`.
4. **DiffusionDrive's preprocessing (F11) is INHERITED** from this programme's own banked note, not
   re-read by me. **Unblocker:** `transfuser_features.py:73` in the released repo (not on this box),
   or the banked PDF's supplementary A.
5. **The trainer's data-vs-config geometry guard (F10a) is READ, not mutation-proven.**
   **Unblocker:** launch `refc_v3_train.py` against the 416 × 1024 cache with `--image-hw 256 640`
   and confirm the `SystemExit` at `refc_v3_train.py:6518-6524`. Not run here: it needs a full
   trainer launch, outside a read-only review's blast radius.
6. **Whether F6's BatchNorm confound moves a TRAINED K=1 vs K=3 result.** MEASURED at step 0 only.
   **Unblocker:** the two arms with `--trunk-frozen-bn`. ⭐ The mitigation costs zero (bit-identical,
   not approximately), so the cheaper decision is to apply it rather than to measure the confound.

⚠️ **One thing I deliberately did not do:** re-fit the normalisation statistics to our corpus. F4
measures that our frames sit ≈1 ImageNet σ below the checkpoint's operating point. Substituting a
corpus-fitted mean/std is the advisory's *"self-consistent but not theirs"* failure and would make
every backbone comparison measure our preprocessing.

⚠️ **Two errors of my own, logged because the method demands it.** (a) I first read
`fuse_identity_init` as broken from a train-mode comparison before adding the frozen-BN control —
the claim is *unqualified*, not wrong, and the control is what distinguishes those. (b) My first
F1c intervention used the trunk's pooled feature, which its own raw-pixel floor beat 0.984 vs 0.872,
and a 32-row crop against a 37-row strip; both were re-run on the correct instrument at correct
depth. Neither wrong version was quoted as a result.

---

## RECOMMENDATIONS, ranked by measured effect

| # | action | cost | why |
|---|---|---|---|
| **1** | **Decide the 416-row bottom strip explicitly** — crop to a height both groups fully observe, mask it, or accept it *in writing* with the fraction stated. | a crop, or a rebuild | F1 + F1c: 74/139 clips, 10–11 % of the frame, a constant at −2.12/−2.04/−1.80, reaching the planner at 17.9 % of the between-clip scale — and **a painted strip alone makes an arbitrary label 0.899-decodable from an 8×16 thumbnail against three chance controls**. The programme already refused this exact signal once (C26). |
| **2** | ⛔ **Do NOT reach for `PHYSICALAI_RIG_CLEAN_176x624` / `_128x576` until F3 is fixed** — `frame_for_model` would give them `f_ref` 297.94 / 275.02 instead of 305.577, i.e. a 120° field for a 117° / 108° frame. | — | F3 is class B3, latent *today* and active *the moment recommendation 1 is acted on the obvious way*. |
| **3** | **Make `frame_for_model` read `_agent_cam_frames()` and REFUSE an undeclared geometry** instead of re-deriving `f_ref` from the width. | ~10 lines | Removes the second spelling of a camera constant — the discipline `trunk_shapes.py:85-91` and `refc_v3_train.py:1337-1345` already enforce elsewhere. |
| **4** | **Pass `observed=` into `build_lift_geometry` from `LiftGeometryBank`.** | one keyword + a mask on the bank | F2: "valid" currently answers *"is this cell inside the declared frame?"* when the consumer needs *"did a sensor see it?"*. 0.2 % of samples — but **all of them in the 3.75–9.75 m headway band**, and it re-enables the `unobserved` embedding that exists for exactly this. |
| **5** | **Run both K arms with `--trunk-frozen-bn`**, and report `bn_training_count()` in the run record. | zero | F6: otherwise K=1 vs K=3 differ at step 0 by a K× BatchNorm batch — the variable `chunk_ckpt` is refused without. |
| **6** | **Update `TimmTrunkConfig.image_hw` `(256,1024)` → `(416,1024)`**, or make it required. | one line | F10: a stale default that mis-declares `s16_shape` / `grid_shape` **silently**, and those are what the run record stamps. |
| **7** | **Add a paired `..._GOES_RED` arm for the MODEL-level ego guard**, mutating `ego_channels_from_poses`. MUT2 / MUT3 in `code/trunk_q5b_guard_and_bn.py` lift straight in. | ~20 lines | F7: the guard is alive — proven here — but nothing in the suite proves it, and its documented coverage (the encoder) is the half it is blind to. |
| **8** | **State in SPEC §10.2 that our `resnet34` arm applies ImageNet mean/std and DD does not**, or add a third cell. | a sentence | F11: otherwise a head-to-head against DD's 88.1 PDMS compares preprocessing too. |

---

## DELIVERABLE MANIFEST

⚠️ **This package folder is SHARED with sibling reviewers**; only the files below belong to this
stream. Everything is **STAGED, NOT COMMITTED, NOT PUSHED**, on branch `agent/arch-inf-20260803`.
Root: `D:\Projects\TanitAD\TanitAD Research Lab\Architecture & Inference\Research\2026-09-22-refcv6-review\`

**Report** — `TRUNK_INPUT_REVIEW.md` (this document).

**Instruments** (`code\`)

| file | what it measures |
|---|---|
| `q1_normalisation_probe.py` | ImageNet mean/std at the stem; no-norm regression + value-range + BGR traps |
| `q1b_real_cache_endtoend.py` | the same end to end on a real 416×1024 PNG episode; geometry identity checks |
| `q1c_zero_border_census.py` | where the exactly-zero pixels are (5 clips, per-row / per-column) |
| `q1d_rig_black_strip_census.py` | the black-strip census over all 139 clips |
| `trunk_q1e_strip_reaches_planner.py` | intervention: does the strip move `pooled`? 3 controls + a reference scale |
| `trunk_q1f_rig_decodability.py` | first decodability pass (trunk feature); found its own raw-pixel floor beats it |
| `trunk_q1g_label_vs_mount.json`†| whether the strip label separates by mount pose (it does not cleanly) |
| `trunk_q1h_rawfloor_crop_ladder.py` | the crop ladder on the raw-pixel floor, with matched top crops |
| `trunk_q1i_strip_sufficiency.py` | the **sufficiency** control: a painted strip on one homogeneous group |
| `q2_lift_geometry_probe.py` | lift reads extrinsics (B2); `f_ref` per declared geometry (B3); impulse feature-centre offset |
| `q2b_lift_vs_unobserved.py` | BEV cells marked valid inside the unobserved strip, 139 clips |
| `trunk_q2c_persistent_zero_mask.py` | tightens that with a persistent-zero mask + both rails |
| `trunk_q2d_orphan_tensor_arithmetic.py` | class-B1 parameter arithmetic vs the real checkpoint |
| `trunk_q3_geometry_literals.py` | reachability of the three surviving geometry literals |
| `trunk_q4_camera_resolution.py` | °/px re-derived under the cylindrical rule; the pinhole caveat |
| `trunk_q5q6_history_and_channels.py` | K-pass / shared-weight / fusion measurement; channel counts, both backbones |
| `trunk_q5b_guard_and_bn.py` | 3-mutation audit of the model-level ego guard; BN vs `fuse_identity_init` |

† `trunk_q1g` was produced inline; only its JSON is banked.

**Raw output** (`raw\`) — `q1_normalisation.json` · `q1b_real_cache.json` · `q1c_zero_border.json` ·
`q1d_rig_black_strip.json` · `trunk_q1e_strip_planner.json` · `trunk_q1f_rig_decodability.json` ·
`trunk_q1g_label_vs_mount.json` · `trunk_q1h_rawfloor_ladder.json` ·
`trunk_q1i_strip_sufficiency.json` · `q2_lift_geometry.json` · `q2b_lift_unobserved.json` ·
`trunk_q2c_persistent_zero.json` · `trunk_q2d_orphan_arithmetic.json` ·
`trunk_q2e_which_bev_rows.json` · `trunk_q3_geometry_literals.json` ·
`trunk_q4_camera_resolution.json` · `trunk_q5q6_history_channels.json` · `trunk_q5b_guard_bn.json`
(+ matching `.err` files; all empty except `trunk_q1h.err` / `trunk_q1i.err`, which carry sklearn
`ConvergenceWarning`s only — **not** failures; both processes exited 0.)

**Read-only inputs touched** (nothing written)
- `D:\Projects\TanitAD-artifacts\v2ep-eval139-416x1024cyl\` — 139 `.v2ep.pt` + `manifest.json`
- `…\Research\2026-09-06-refcv4b-landing\raw\extrinsics141.json`
- `~\.cache\huggingface\hub\models--timm--resnet34.a1_in1k`, `…--resnet101.a1_in1k`

**Guard suites run** — `test_refcv6_geometry_agnostic.py`, `test_frame_416x1024.py`,
`test_refcv6_trunk.py`: **67 passed, 0 skipped**. `test_refcv6_tactical.py -k "FUTURE or ego_poses"`
plus the trunk ego tests: **12 passed, 0 skipped**.

⚠️ **Staging note, reported rather than undone** (`CLAUDE.md`: *"if you over-stage, report it,
don't unstage"*): the FIRST staging call used `git add -- <code/> <raw/>` on the **shared** package
directory, so sibling reviewers' in-progress files in those two folders were swept into the index
alongside this stream's. No file was modified; only the index was touched. Not from this stream:
`code/arm_bev_controls.py`, `code/arm_c2_force.py`, `code/build_refcv6_probe_model.py`,
`code/diag_*.py`, `code/p1_*.py` – `p4_*.py`, `code/probe_*.py`,
`code/q1_nav_reach_three_consumers.py`, `code/q1b_nav_every_denoising_pass.py`,
`code/q2_tactical_kv_set.py`, `code/q3_strategic_heads_built.py`,
`code/q3b_route_loss_reaches_trunk.py`, and their `raw/*.json`. Every later staging named this
stream's files explicitly, and each was verified by a shape-asserted blob comparison
(`git ls-files --stage` vs `git hash-object`, both required to be 40 chars).

⚠️ **Contention observed:** `.git/index.lock` was held by a sibling for ~20 s mid-session; staging
was retried with backoff rather than the lock being removed (`CLAUDE.md`: never unlink a lock while
git is alive). One `git add` silently left a stale index blob and was caught **only** by the blob
comparison, not by its exit code — the documented trap, reproduced.

**No commit, no push.** Nothing outside this package directory was modified.
