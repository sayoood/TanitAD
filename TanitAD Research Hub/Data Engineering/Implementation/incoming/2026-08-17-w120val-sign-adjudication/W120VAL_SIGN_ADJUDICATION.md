# The `w120val` sign leg, adjudicated — and the confound closes on the INSTRUMENT, not the corpus

**Package owner:** arch-inf agent, 2026-08-17 · branch `agent/arch-inf-20260803`
**Closes:** `…/2026-08-16-sam3-concept-reliability/SAM3_CONCEPT_RELIABILITY.md` §4.1 — *"the remaining
uncontrolled variable is the CORPUS … Neither number transfers."*
**Gates:** `…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md` §3.
**Pre-registration:** `PREREG.md`, written and staged **before a single crop was rendered**.

---

## 0. TL;DR

| question | answer | class |
|---|---|---|
| **Is `w120val`'s `traffic sign` channel ~⅔ garbage, as G1's number has been read?** | ⛔ **NO.** Uniform draw, n=64 over 56 clips: precision **0.852 [0.759, 0.927]**; G1's *"no sign at all"* subclass **6/64 = 9.4 % [3.0, 17.4]**, not ~71 %. | **MEASURED** (§2) |
| **Then do the corpora differ, as the reliability study left open?** | ⛔ **NO — and they are not even two corpora.** The pilot-50 G1 measured is a **strict SUBSET** of the 596-clip `w120val` leg (overlap 50/50). On G1's OWN clips under G1's OWN max-area rule, precision is **0.867 [0.733, 0.967]** and the empty-box rate is **1/37 = 2.7 %**. | **MEASURED** (§1, §3) |
| **So was G1 wrong?** | ⚠️ **NO — G1 read its own crops correctly.** Re-reading G1's **exact banked JPEGs** blind reproduces it and then some: **48/54 = 88.9 % [77.6, 98.2]** contain no traffic sign at all, against G1's own ~71 %. **The adjudicator is not the variable.** | **MEASURED** (§4) |
| ⭐ **What IS the variable, then?** | **G1's CROPPER.** Of its 54 tiles, **0 are a tight crop of the box they are attributed to**, **45 are padded to a ~96 px floor**, **5 are the ENTIRE 640×256 native frame**, and **none carries a box outline**. The median tile is **4.05×** the area of a tight crop of its box. ⇒ the reader is shown a patch of scene and never told which pixels the detector claimed. | **MEASURED** (§5) |
| **Verdict on the pre-registered question** | **Outcome 2** — *G1's reading does not generalise to the detector*, with one **amendment I did not pre-register and must name** (§6). | — |

⛔ **ESCALATION 1 — `NEXT_4472_BUILD_INPUTS.md` §3 AND the reliability study §4.1 BOTH need a
correction, and it is not the one they expect.** Both say G1 measured **`w120val` (600 clips)**. The
primary sources say **the 50-clip pilot** (§1). Both frame the open variable as **the corpus**. It is
not: it is the rendering, and the study's hypothesis-B test could not have found it (§5.1). The
honest sentence is now *"G1's crops were unreadable by construction"*, not *"not on aug120"*.

⛔ **ESCALATION 2 — the sign channel is RELEASED for the 4,472 build as a presence flag at 0.5, and
NOT released for anything else.** §7. Nothing here reopens the KIND or TEXT questions; the two
highest-scoring false positives in the whole sample are a **dashboard `30` roundel (0.927)** and a
**commercial retail-park hoarding (0.778)**, and no threshold removes either.

---

## 1. ⛔ THE SCOPE ERROR, ESTABLISHED BEFORE THE EXPERIMENT

`SAM3_CONCEPT_RELIABILITY.md` §4.1 and `NEXT_4472_BUILD_INPUTS.md` §3 both attribute G1's number to
**`w120val` (600 clips, 4,048 sign detections)**. The primary sources say something narrower:

| source | verbatim |
|---|---|
| `Project Steering/G1_SIGN_OCR_GRADING_SHEET.md:5` | *"These are the **31 non-empty OCR texts** from the **50-clip pilot** — the pre-registered sample."* |
| `Project Steering/G1_RESULT.md:4-5` | *"for each of the 31 pre-registered pilot OCR texts, the sign was cropped from **the pilot videos** using SAM3's boxes"* |
| `Project Steering/G1_RESULT.md:33-35` | *"**Production** banked 4 048 'traffic sign' detections on the 600-clip set; **if** this false-positive character generalises …"* |

⇒ G1 measured **31 crops from 30 clips of the 50-clip pilot**. The `4 048` sentence is a **scale
statement with an explicit hedge** — G1 scoped itself correctly, and the scope was lost downstream.

**And the structural fact that dissolves the corpus hypothesis before any picture is looked at**
(MEASURED, `raw/records_index_*.json`, `code/w1_pull_records.py`):

| leg | HF path | clips | `traffic sign` | over clips | min banked score | frame |
|---|---|---|---|---|---|---|
| **w120val (production)** | `Sayood/tanitad-ph0` → `ph0_prod4/sam3/sam3.json` | **596** | **4 048** | 440 | **0.5000** | 448×179 |
| **pilot-50 (G1's own)** | `Sayood/tanitad-ph0` → `ph0_pilot50/sam3/sam3.json` | **50** | **292** | 37 | **0.5002** | 448×179 |

⭐ **The pilot-50 is a strict SUBSET of the production leg — overlap 50/50, pilot-only 0** — and all
30 distinct G1 clip prefixes resolve in both. Both are the same
`physicalai-val-0c5f7dac3b11-w120-256x640cyl` corpus. ⚠️ `596`, not 600: four of the 600 val clips
carry no SAM3 record. **The "600-clip" figure in three documents is off by four.**

**Box geometry is indistinguishable across all three corpora** (`raw/score_distribution_*.json`,
`raw/maxarea_mechanism.json`) — median `traffic sign` box area **68.9 px² (w120val) / 70.9 (pilot-50)
/ 70.9 (aug120)**, and the fraction of the class within 0.05 of the vendor threshold is
**16.6 % / 16.4 % / 16.9 %**. There is no geometric or score-distributional difference between the
corpora to carry a 17× reliability difference.

---

## 2. ⭐ ARM A — the brief's ask: `w120val`, uniform draw, n=64

**Protocol.** Uniform at random within `traffic sign` over the whole 4,048-detection population, seed
`20260817`; **BLIND** (each cell carried only its index); rendered by the reliability study's own
`crop_cell()` imported unchanged — 6×-box context window from the native 640-px frame, **box outlined
in gold**. Precision, CI, bands and sweep computed by **`r5_precision.py`, run unchanged**. Intervals
are the **episode-cluster bootstrap over CLIPS** (`taniteval/ci.py`, 2 000 draws) — ⛔ never
`overlapping_holdout_se`, never a binomial.

| | n | ✅ | ❌ | ❓ | **precision** | clips |
|---|---|---|---|---|---|---|
| **resolvable only** | 54 | 46 | 8 | — | **0.852 [0.759, 0.927]** | 51 |
| **❓ counted wrong** | 64 | 46 | 8 | 10 | **0.719 [0.612, 0.825]** | 56 |

**Head to head with the numbers this was run to compare against:**

| quantity | **w120val (this arm)** | `aug120` (the study) | G1 (the pilot, as rendered by G1) |
|---|---|---|---|
| precision, resolvable | **0.852 [0.759, 0.927]**, n=64 | 0.880 [0.795, 0.958], n=64 | — |
| **"no sign at all" rate** | **6/64 = 9.4 % [3.0, 17.4]** | 4/96 = **4.2 %** | ~22/31 = **~71 %** |
| `unclear` rate | **15.6 %** | 21.9 % | — |

⇒ **`w120val` is statistically indistinguishable from `aug120`** (CIs overlap heavily) and nowhere
near G1's ~⅔.

### 2.1 The eight false positives — and the two that matter

| # | score | box px² | subclass | what it is |
|---|---|---|---|---|
| #3 | 0.532 | 46.0 | **EMPTY** | blank building facade — a real blue sign is in frame, **outside** the box |
| #25 | 0.539 | 59.9 | **EMPTY** | bare ground/hedge between two poles at dusk |
| #8 | 0.625 | 17.5 | **EMPTY** | uniform grey, night fog, no mount of any kind |
| #18 | 0.629 | 14.8 | **EMPTY** | featureless patch in front of foliage |
| #1 | 0.630 | 197.8 | **EMPTY** | plain beige wall — again a real `P` sign in frame, **outside** the box |
| #53 | 0.668 | 22.5 | **EMPTY** | the uniform red-lit underside of an overpass |
| **#46** | **0.778** | 2310.9 | **real object** | a **commercial retail-park hoarding** — `SANT BOI · PARC COMERCIAL` with a direction arrow, over a `149,-` price panel |
| **#6** | **0.927** | 927.4 | **real object** | ⭐ a perfectly legible **`30` speed-limit roundel on the ego vehicle's own hood/dash band**, on a **motorway** behind a truck, where a 30 limit is impossible ⇒ an in-cabin display or a burned-in overlay |

⭐ **The two highest-scoring false positives in the entire sample are the two that are NOT empty
boxes** — and both are objects a KIND-blind consumer would swallow. This reproduces the reliability
study's §5 mechanism **exactly and independently, on a different corpus**: *"the sign errors that
would corrupt supervision are sign-shaped non-traffic-signs, and no threshold separates them."*
Here the score ordering is even more hostile: every empty box scores **≤ 0.668**, and the two real
misclassifications score **0.778 and 0.927**.

⚠️ **#6 is a judgement call and is recorded as one.** The opposite reading — *it is a speed-limit sign
face, therefore `correct`* — is defensible. It is scored `wrong` because a channel asking *"is there a
traffic sign in this scene"* is corrupted by a dashboard graphic. Flipping it alone moves precision
0.852 → 0.870 and changes nothing else. `#46` is likewise borderline (a retail-park direction board
is roadside guidance in some jurisdictions) and is recorded as borderline rather than hidden.

### 2.2 ⚠️ The `unclear` rate is a fact about the INPUT, and the escalation is reported

**30 of 64 cells could not be called from the 320-px contact-sheet cell.** Every one was re-rendered
at a **600-px cell in BOTH sanctioned renderings** (the 6× context window with the gold box, and G1's
tight 4× LANCZOS crop) and looked at again — `crops/zoomA1..zoomA4_{context,g1tight}.jpg`. **20 of the
30 resolved.** ⚠️ Escalation can only move a verdict toward `correct` or leave it `unclear`; more
pixels never manufacture a false positive. Reporting it is what makes **15.6 %** a property of the
input rather than of how hard I looked. Median box area tracks it exactly: **78.0 px² for correct,
59.9 for wrong, 23.5 for unclear** — the unresolvable cells are the *small* ones, as on `aug120`.

---

## 3. ⭐ ARM B — G1's OWN clips, G1's OWN rule, one variable changed

Because the pilot-50 is a subset of `w120val` (§1), G1's protocol can be re-run on G1's own material.
**CENSUS** of all 37 pilot clips carrying a sign; selection = **G1's largest-area `traffic sign` per
clip**; both renderings viewed per detection before the verdict was fixed. Joined by the study's own
`r8_g1_verdict_join.py`, run unchanged.

| | n | ✅ | ❌ | ❓ | **precision (resolvable)** | **"no sign at all"** |
|---|---|---|---|---|---|---|
| **arm B — G1's clips + G1's selection, box OUTLINED** | 37 | 26 | 4 | 7 | **0.867 [0.733, 0.967]** · 30 clips | **1/37 = 2.7 % [0.0, 8.1]** |
| **arm C — G1's clips + G1's selection + G1's OWN RENDERED TILES** | 54 | 1 | 48 | 5 | — | **48/54 = 88.9 % [77.6, 98.2]** |

⛔ **Same corpus. Same selection rule. Same detector. Same vendor threshold. 2.7 % versus 88.9 %.**
The only thing that changed is how the box was shown to the reader.

**The four arm-B errors, and none of them is a hallucination on emptiness:**

- **#2007** — a **Greek pharmacy shop sign**, `ΦΑΡΜΑΚΕΙΟ` down an illuminated green fascia.
  ⭐ `G1_SIGN_OCR_GRADING_SHEET.md` row 22 is `ΦΑΡΜΑΚΕΙΟ` on clip `15958430-1119`: **the VLM read this
  same shop sign, and G1 correctly refused it as unverifiable.** Same class as the study's #207.
- **#2029** — a **traffic light**: an orange arrow aspect inside a dark signal housing. Cross-class
  confusion *inside* the traffic-infrastructure family — the study's #1001 in reverse.
- **#2023** — a black **X over text on a green-lit office building** — commercial building signage.
  ⚠️ the least certain of the four.
- **#2034** — the **only empty box in the arm**: a featureless dark band above the road at night.

---

## 4. ARM C — G1's own tiles, re-read blind: G1 IS REPRODUCED

The 54 crops G1 read are still banked (`Sayood/tanitad-ph0-aug120` → `g1_evidence/crops/row01…54.jpg`,
0.2 MB). They were **shuffled under seed `20260817` and re-indexed `3000+k`** so G1's row ordering —
and, through the grading sheet, its claimed OCR text and clip — could not reach the eye. Every cell
that was not callable at contact-sheet size was re-rendered at 620 px (`crops/zoomC_1.jpg`,
`crops/zoomC_2.jpg`); **#3001 was promoted from `unclear` to `correct` by that step**, so the
escalation demonstrably works in the detector's favour.

**Result: 1 correct · 48 no-sign-at-all · 5 unclear.** G1 reported ~22 of 31 rows (~71 %); per tile a
second reader gets **88.9 %**. ⇒ **G1's reading of its own evidence is REPRODUCED, and is if anything
CONSERVATIVE.** ⛔ Scored **charitably**: G1's rule was *"sign/**light**"*, so `wrong` here means no
traffic infrastructure **of either kind** is visible — which can only move the number in G1's favour.

The 48 are exactly G1's own list: plain **sky/cloud** (8), uniform **wall/facade** (13), **foliage**
(6), night **darkness** with no discernible object (15) — and **6 tiles that are essentially the whole
scene**. A tight crop of a box cannot be the whole scene unless the crop is not of the box.

---

## 5. ⭐ WHAT G1'S CROPS ACTUALLY WERE — the measurement that closes the case

`code/w6_g1_crop_forensics.py` matched every banked tile against the top-2 largest
`traffic sign`/`traffic light` boxes in its clip's record — G1's own stated selection rule
(`raw/g1_crop_forensics.json`). Geometry only; no scores, no verdicts.

| statistic | value |
|---|---|
| tiles that are the **WHOLE 640×256 NATIVE FRAME** | **5 / 54** |
| tiles consistent with a **pure 4× tight crop** of their box | **0 / 54** |
| tiles **padded to a ~96 px floor** | **45 / 54** |
| tile area ÷ (4× box area) — *1.0 = the tile IS the box* | min **1.27** · **median 4.05** · max **316** |
| tiles carrying a **box outline** | **0 / 54** (visible in `crops/w120sign_C_g1tiles_*.jpg`) |

And the explanation that would have been comfortable is **REFUTED**: there are **no frame-spanning
sign boxes on any leg** (`raw/maxarea_mechanism.json`). The largest `traffic sign` box anywhere in
4,048 + 292 + 538 detections is **7 364 px² = 9.2 % of the frame**, and **zero clips** on **any** leg
have a max-area pick ≥ 25 % of the frame. The whole-frame tiles are a **crop failure**, not a
pathological detection.

⇒ **The mechanism.** A genuine 8×9-px sign, placed unmarked inside a ≥24×24-px window of street
scene, upscaled 4× into a blurry tile, with **no indication of which pixels the detector claimed**,
reads as *"foliage"*, *"a wall"*, *"sky"* — and in five cases the tile is the entire frame, which is
**unadjudicable by construction**. G1's subclass-1 count measures that, faithfully.

⚠️ **The cropper's source is not banked** — no G1 crop script exists in the repo (searched:
`g1sheet`, `row%02d`, `g1_evidence`, `LANCZOS`). So the *per-tile alignment* cannot be verified from
source, only characterised from the artifacts. The five whole-frame tiles, the 96-px floor and the
0/54 tight-crop match are properties of the banked JPEGs themselves and stand regardless.

### 5.1 ⛔ Why the reliability study could not have found this

`SAM3_CONCEPT_RELIABILITY.md` §4.1 tested *"hypothesis B — the RENDERING"* and **REFUTED** it. That
test was sound for what it tested and **could not reach this**: `r7_g1_reconcile.py:143-146` crops
**exactly the box** (`Image.crop((x0,y0,x1,y1))`) and then upscales, so in the reimplementation the
sign **fills the tile**. The real cropper pads to a floor and draws no outline. ⇒ **the study
re-implemented what G1's renderer was believed to do, and the defect lives in the difference.**
⭐ **This is the C79 lesson one level up: an arm that reimplements the thing under test cannot see a
defect in the original.** The artifacts were on HF the whole time; re-reading them cost 0.2 MB.

---

## 6. The pre-registered verdict — and the amendment I owe

`PREREG.md` §3, evaluated mechanically by `code/w7_join_and_verdict.py`:

- **E_val = 0.0938** (≤ 0.15 ✓) and **P_val = 0.852 [0.759, 0.927]** overlaps `aug120`'s
  [0.795, 0.958] ✓ ⇒ **outcome 2 triggered**; outcome 1 **not** triggered (E_val ≥ 0.40 ✗,
  P_hi < 0.70 ✗).

> **REPORTED OUTCOME — 2: G1's original reading does not transfer to the DETECTOR. `w120val` looks
> like `aug120`, and the reliability study's picture generalises to the val side.**

⚠️ **THE AMENDMENT, STATED RATHER THAN BURIED.** `PREREG.md` §3 also binds *"if arms A/B/C disagree,
outcome 3 is the answer regardless"*, and **arm C does disagree** (88.9 % vs 2.7 %). I am **not**
invoking outcome 3, and the reason must be auditable:

1. **A and B measure the same estimand — the detector — and they agree**: 0.852 [0.759, 0.927] and
   0.867 [0.733, 0.967]; empty-box 9.4 % and 2.7 %.
2. **Arm C measures a different estimand** — whether *G1's rendered tiles* show signs — and its
   divergence has an **independently MEASURED cause** (§5) that is not the detector, the corpus or
   the selection.
3. ⚠️ **My own pre-registration was incomplete here, and I will not pretend otherwise.** It offered
   two readings of a high arm-C rate — *"corpora or selection"* or *"the adjudicator"* — and the truth
   is a **third it did not anticipate: the rendering pipeline**. Arm B is what excludes the first two;
   §5 is what establishes the third. **A reader who holds me to the literal clause gets outcome 3, and
   every number needed to do so is in this document.**

---

## 7. Consequence for the 4,472-clip build

| | verdict | why |
|---|---|---|
| `traffic sign` as a **per-clip PRESENCE flag at 0.5** | ✅ **RELEASED on the val side too** | 0.852 [0.759, 0.927] on `w120val` (n=64/56 clips), 0.867 [0.733, 0.967] on the pilot census. `NEXT_4472_BUILD_INPUTS.md` §3's blocking sentence no longer has a corpus to point at |
| `traffic sign` as **per-detection supervision** | 🟨 **≥ 0.70 if used at all, and say so in the record** | sweep: 0.50 → 0.852, **0.70 → 0.920** retaining **1 878/4 048 = 46.4 %**. ⚠️ **NOT tuned** — 8 false positives, and the band structure is **non-monotone** (0.60–0.70 → 0.789, 0.70–0.80 → 0.923, **0.90+ → 0.800**) |
| `traffic sign` as evidence of a sign's **KIND or TEXT** | ⛔ **STILL FORBIDDEN, and nothing here softens it** | the G1 text gate is CLOSED at 0/31; `goal_evidence` is already retired; and §2.1's two worst errors are a **dashboard roundel at 0.927** and a **commercial hoarding at 0.778** |
| a **threshold** as the fix for the sign channel | ⛔ **NO** | every empty box scores ≤ 0.668 while both real misclassifications score 0.778 / 0.927. **A score cut removes the harmless errors and keeps the harmful ones.** A KIND check is the lever; §6.1 of the reliability study stands unchanged |

⭐ **The durable lesson is not about SAM3.** It is that **`g1_evidence/crops/` was banked, and nobody
re-opened it for three days** while two packages argued about which corpus was to blame. The
adjudication protocol that fixed it — **draw the box, show context, and escalate what you cannot
resolve** — is the difference between 2.7 % and 88.9 % on identical detections.

---

## 8. What this package does NOT say

1. ⛔ **No recall, for any concept, on any leg.** Precision only — same limit as the study.
2. ⛔ **Nothing about the other six concepts on `w120val`.** Only `traffic sign` was adjudicated
   (`PREREG.md` §2 delta 3). `car`/`truck`/`bus`/`pedestrian`/`traffic light`/`cyclist` on this leg
   remain **unmeasured**, and their `aug120` numbers must keep carrying that corpus.
3. ⛔ **`raw/precision_A_w120val.json` contains an EMPTY `car_cyclist_confusion` block** (`0 cyclist
   detections`). `r5_precision.py` was run unchanged and its cyclist test filters records on a
   `liveness` key the production engine predates, so it saw nothing. **That block is an artefact of
   running the script unchanged and is NOT a measurement — do not cite it.**
4. ⛔ **`raw/g1_headtohead_B.json` carries HARDCODED `aug120` prose**, because `r8_g1_verdict_join.py`
   was run unchanged: `g1_subclass1_empty_boxes: 0`, a `g1_reference.corpus` string that is itself the
   scope error §1 corrects, and a `verdict{}` narrative belonging to the earlier study. **Only its
   computed numbers are mine.** The subclass counts for arms A/B/C are in
   `raw/w120val_sign_verdict.json`.
5. ⚠️ **Single adjudicator for arms A and B; no inter-rater number.** Arm C *is* an inter-rater
   measurement against G1 and it agrees. Verdicts are banked per index so a second reader can be run
   without re-rendering.
6. ⚠️ **Two verdicts are judgement calls** (#6 the dashboard roundel, #46 the retail hoarding; and
   #2023 in arm B). Each is recorded with its counter-reading. Flipping all three moves arm A to
   0.887 and arm B to 0.897 — i.e. **the conclusion is not load-bearing on them**.

---

## 9. Deliverable manifest

⛔ **STAGED, NOT COMMITTED, NOT PUSHED**, branch `agent/arch-inf-20260803`. Every row verified with
`git ls-files --cached` (new files) and re-verified at end of turn.

All paths under
`TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-17-w120val-sign-adjudication/`:

| artifact | path | what it is |
|---|---|---|
| **pre-registration** | `PREREG.md` | banked **before** any crop was rendered; §0 carries the scope correction, §3 the numeric decision rule |
| this report | `W120VAL_SIGN_ADJUDICATION.md` | — |
| record pull, both legs | `code/w1_pull_records.py` → `raw/records_index_{w120val600,pilot50}.json` | 596/50 clips, 4 048/292 signs, min score 0.5000/0.5002 |
| score distributions | `code/w2_score_dist.py` → `raw/score_distribution_{w120val600,pilot50}.json` | reuses `r2_score_dist.py` with ONE declared substitution |
| sampling + BLIND sheets, arms A/B | `code/w3_sample_and_render.py` → `raw/adjudication_sample_{A_w120val,B_pilot50}.json` | frames from the **pipeline's own bridge** |
| escalation render | `code/w3z_zoom.py` | 600-px cells, both sanctioned renderings |
| arm C sample | `code/w4_g1_tiles.py` → `raw/adjudication_sample_C_g1tiles.json` | G1's 54 tiles, shuffled + re-indexed |
| **the verdicts** | `raw/verdicts_A_w120val.json`, `raw/verdicts_B_pilot50.json`, `raw/verdicts_C_g1tiles.json` | 155 human calls with per-index mechanism notes |
| max-area mechanism | `code/w5_maxarea_mechanism.py` → `raw/maxarea_mechanism.json` | ⛔ my own "giant boxes" hypothesis, **REFUTED** |
| ⭐ **crop forensics** | `code/w6_g1_crop_forensics.py` → `raw/g1_crop_forensics.json` | §5 — **the measurement that closes the case** |
| join + pre-registered decision | `code/w7_join_and_verdict.py` → `raw/w120val_sign_verdict.json`, `raw/precision_A_w120val.json`, `raw/g1_headtohead_B.json` | wraps `r5_precision.py` and `r8_g1_verdict_join.py`, both **run unchanged** |
| evidence banking | `code/w8_bank_crops.py` | PNG → JPEG q=90 **4:4:4** |
| **contact sheets + escalations** | `crops/*.jpg` — 24 files, 4.5 MiB | ⚠️ **the ONLY durable copy of the rendered evidence**; the scratchpad PNGs are session-local. The verdicts were fixed on the PNGs; every sheet re-renders deterministically from the banked sample JSON |

### Read, not modified

`Project Steering/G1_RESULT.md`, `G1_SIGN_OCR_GRADING_SHEET.md`, `MODEL_REGISTRY.md`;
`…/2026-08-16-sam3-concept-reliability/**` (imported, **not one byte changed** — its record stands as
filed); `…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md`; `stack/scripts/ph0_sam3.py`,
`ph0_pilot.py`, `v2_to_pilot.py`; `taniteval/taniteval/ci.py`.

### Far side — nothing written

**Pulled only:** `Sayood/tanitad-ph0` → `ph0_prod4/sam3/sam3.json`, `ph0_pilot50/sam3/sam3.json`;
`Sayood/tanitad-ph0-aug120` → `g1_evidence/crops/*.jpg` (54), `w120val_600/clips.json`,
one `fused_w120val` record; `Sayood/tanitad-physicalai-w120-256x640cyl` →
`physicalai-val-.../<clip>.v2ep.pt` for the **90 clips** the two arms touch (~3.2 GB, HF cache).
**No HF write, no pod, no GPU.**

### Suite — **3 763 passed · 0 failed · 7 skipped · 2 xfailed** (426.8 s)

⚠️ **No file under `stack/` was created or modified by this package** — it imports
`v2_to_pilot`/`ph0_pilot` and reads `ph0_sam3.py`. `git status --short -- stack/scripts/` shows only
sibling agents' entries (`refc_train.py`, `refc_dump_latents.py`, `refc_obj_dk_error.py`).
⇒ **the delta against the briefed baseline (3 750 / 0 / 7 / 2) is +13 PASSED and ZERO failures, and
those +13 are sibling agents' new tests, not mine.** Green.

⛔ **AND A TRAP WORTH BANKING: the literal invocation `cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6
pytest -q` DOES NOT RUN THIS SUITE ON THIS BOX.** Bare `pytest` resolves to
`C:\Users\Admin\AppData\Local\hermes\hermes-agent\venv\Scripts\pytest` — an unrelated agent venv
with no torch — and reports **`10 skipped, 191 errors in 7.61 s`, "191 errors during collection",
while exiting 0**. That reads exactly like a catastrophic regression and is an **interpreter**
problem: `which python` gives `C:\Python314\python.exe`, not the project venv. ⇒ **the runnable form
is `PYTHONUTF8=1 OMP_NUM_THREADS=6 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q`**,
which is what produced the number above. ⚠️ This is the same family as the brief's own warning about
`CUDA_VISIBLE_DEVICES=""` — **an uncontrolled environment variable that manufactures a false failure
whose arithmetic looks plausible** — and it is worth adding to the brief so the next agent does not
report a 191-error regression that never happened.

---

## 10. Escalations — decisions, not documentation

1. ⛔ **`NEXT_4472_BUILD_INPUTS.md` §3 needs a THIRD correction.** It was already corrected once from
   *"G1 was wrong"* to *"not on aug120"*. Both are now superseded: G1's ⅔ is a property of **G1's
   crop rendering**, the corpus row in its table should read **`pilot-50`, a subset of `w120val`**,
   and the *"open work item, 0 GPU, ~2 h"* line is **CLOSED by this package**. Owner: the
   aug120-fusion package.
2. ⛔ **`SAM3_CONCEPT_RELIABILITY.md` §4.1 / ESCALATION 2 / §8.2 should be annotated, not rewritten.**
   Its hypothesis-B refutation is **correct for what it tested** and **does not reach G1's real
   renderer** (§5.1). ⚠️ I have **not touched that package** per the brief; this needs a one-line
   pointer from it to here. Owner: whoever holds it.
3. ⚠️ **`RETRACTION_LOG.md` entry warranted, and I did not write it** — that file carries another
   agent's STAGED changes and editing it would sweep their work into mine (CLAUDE.md's git-hygiene
   rule). **Proposed class:** *an INSTRUMENT DEFECT read as a property of the thing measured* — G1's
   cropper padded, never outlined the box, and emitted the whole frame 5 times in 54; the resulting
   *"⅔ contain no sign"* travelled through two packages as a fact about SAM3, and the corrective
   evidence sat un-reopened on HF the whole time. ⭐ **Sibling of the `df` / Thor `free` / cgroup
   `usage_in_bytes` family already in CLAUDE.md — a probe that reports the wrong thing, read as an
   answer — with the extra twist that the FIX for it was itself a reimplementation (§5.1).**
4. ⚠️ **The "600-clip" figure is 596** in `G1_RESULT.md`, `SAM3_CONCEPT_RELIABILITY.md` §4.1 and
   `NEXT_4472_BUILD_INPUTS.md` §3. Four val clips carry no SAM3 record.
5. ⭐ **Adopt the outlined-box rule as protocol.** Any future visual adjudication in this programme
   **draws the claimed box and shows context**. This package measures the cost of not doing it at
   **2.7 % → 88.9 %** on identical detections.
