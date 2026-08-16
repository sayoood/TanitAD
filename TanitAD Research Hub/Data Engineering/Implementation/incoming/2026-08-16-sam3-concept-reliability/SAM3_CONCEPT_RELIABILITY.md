# How reliable is each SAM3 concept? — a LABELLED precision study, and the threshold decision

**Package owner:** arch-inf agent, 2026-08-16 · branch `agent/arch-inf-20260803`
**Gates:** the 4,472-clip Alpamayo build (`…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md` §3).
**Corpus:** the 83 `sam3_backfill` records the FIXED engine produced (2,496 detections, 0 errors),
HF `Sayood/tanitad-ph0-aug120` → `sam3_backfill/`. The 32 still-stale C77 records are excluded
everywhere — mixing their zeros into a distribution would put un-run clips in as measurements.

---

## 0. TL;DR — and the two things that must be decided by a human

| question | answer | class |
|---|---|---|
| **Are sub-threshold detections retained?** | ⛔ **NO. They are discarded at inference**, inside the vendor's `_forward_grounding` (`keep = out_probs > self.confidence_threshold`, default **0.5**), before our code ever sees them. Our own `min_score` (default 0.0) sits DOWNSTREAM and therefore never binds. | **MEASURED**, two independent ways (§1) |
| ⇒ **is a threshold sweep offline?** | **UPWARD yes** (pure re-analysis on banked scores — this document does it). **DOWNWARD no** — anything below 0.5 requires re-detection. | **MEASURED** |
| **Is `traffic sign` two-thirds garbage, as G1 found?** | ⛔ **NOT on this corpus.** 96 adjudicated sign detections: precision **0.88** (uniform draw) / **0.93** (G1's own max-area rule), and G1's own *"no sign at all"* subclass occurs at **4/96 = 4.2 %**, not ~71 %. | **MEASURED** (§3, §4.1) |
| **Is `car`↔`cyclist` confusion real?** | ⛔ **The named observation is REFUTED.** On `8f5df500` the `car 0.72/0.83` boxes are on **two distant cars**; the same cyclist is boxed `cyclist 0.85` eight frames later. `cyclist` is **thin, not broken** — 6/7 resolvable correct. | **MEASURED** (§4.2) |
| **Should the 0.5 threshold move?** | **NO, not corpus-wide.** Only `traffic sign` has enough false positives (6) to support any statement, and even there the recommendation is conditional on the consumer. Every other class has **0–1** false positives in its whole adjudicated sample. | **MEASURED + a refusal to over-fit** (§5) |

⛔ **ESCALATION 1 — `goal_evidence: grounded` is NOT rescued by this result, and the reason is not
SAM3.** `ph1_fuse.py:469-474` grounds a `route_to` goal on `sum(1 for t in tracks if t["concept"]
== "traffic sign") > 0` — **any sign-like track, anywhere in the clip, of any KIND, at any score**.
This study says that predicate's *inputs* are ~90 % real signs. It says nothing about whether the
**navigation** sign the VLM claims to have read exists, and G1 has already closed that gate at
**0/31**. ⇒ **`grounded` must become `not_computable` (or be renamed to what it measures: "a
sign-like object is present in this clip") regardless of anything here.** Decision needed. §6.1.

⛔ **ESCALATION 2 — G1's ⅔ figure is NOT transferable, and neither is mine.** I tested both protocol
differences I could control (max-area selection; tight 4× LANCZOS crop) and **both are REFUTED** as
the cause. The remaining uncontrolled variable is the **corpus** — G1 measured `w120val` (600 clips,
4,048 sign detections), this study measured `aug120` (83 clips, 538). **The val-side sign channel
needs the same adjudication before its labels are trusted.** Named work item, §4.1.

⚠️ **A number in the brief is wrong and should be corrected at source:** `traffic sign` is **538**
of the 2,496 detections, not 508. Two independent sources agree (`…/2026-08-16-sam3-dtype-fix/raw/
census_after.json`, and this package's own recount from the 83 raw records).

---

## 1. ⛔ THE FACT THAT DECIDED THE WHOLE SHAPE OF THIS STUDY

**Sub-threshold detections were discarded at inference and are NOT in the records.** Established
two ways, deliberately independent, because the answer decides whether a threshold study costs a
GPU day or an afternoon.

**Probe 1 — the emitting code and the vendor source.** `ph0_sam3.build_processor` constructs
`Sam3Processor(model)` and passes no threshold, so the vendor default is in force. The vendor's
`sam3/model/sam3_image_processor.py` is:

```python
def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
...
def _forward_grounding(self, state):          # set_text_prompt returns this
    out_probs = (out_probs * presence_score).squeeze(-1)
    keep = out_probs > self.confidence_threshold      # ⛔ HERE
    out_probs = out_probs[keep]; out_masks = out_masks[keep]; out_bbox = out_bbox[keep]
    ...
    state["scores"] = out_probs
```

`ph0_sam3._score` reads `out["scores"]` — i.e. the ALREADY-FILTERED array — and only then applies
our `min_score`. ⚠️ **Our filter's `0.0` default is honest but inert.** Its docstring
(*"a threshold picked before the score distribution is known is a decision dressed as a default"*)
describes exactly the right principle and the vendor made that decision for us one layer up.

**Probe 2 — the banked data.** Over all **2,496** detections in the 83 fixed-engine records, the
**minimum banked score is 0.5000** (`raw/records_index.json`), with exactly **one** detection
rounding to 0.5 and per-concept minima 0.500 / 0.501 / 0.502 / 0.524 / 0.527. A strict `>` filter
at 0.5 predicts precisely this; an unfiltered distribution could not produce it.

⇒ **Everything in §2–§5 is an UPWARD re-analysis of banked scores — zero GPU.** A downward sweep is
specified but **not run**, per the brief (§7).

---

## 2. Score distributions per concept (n = 2,496 over 83 clips; `raw/score_distribution.json`)

| concept | n | clips | p10 | median | p90 | **< 0.55** | < 0.60 | ≥ 0.90 | median box area |
|---|---|---|---|---|---|---|---|---|---|
| `car` | 1260 | 70 | 0.532 | 0.743 | 0.918 | 14.3 % | 23.2 % | 14.8 % | 188 px² |
| `traffic sign` | 538 | 62 | 0.530 | 0.703 | 0.882 | 16.9 % | 29.6 % | 5.4 % | 71 px² |
| `traffic light` | 385 | 29 | 0.533 | 0.629 | 0.799 | 16.9 % | **37.1 %** | 1.8 % | **34 px²** |
| `pedestrian` | 180 | 33 | 0.535 | 0.688 | 0.835 | 12.2 % | 26.7 % | 3.3 % | 120 px² |
| `truck` | 97 | 28 | 0.530 | 0.707 | 0.940 | 16.5 % | 34.0 % | 21.6 % | 647 px² |
| `bus` | 26 | 12 | 0.618 | **0.880** | 0.949 | **7.7 %** | 7.7 % | **46.2 %** | 712 px² |
| `cyclist` | 10 | 7 | 0.548 | 0.707 | 0.858 | 10.0 % | 20.0 % | **0.0 %** | 376 px² |

**Mass near the boundary.** Every agent class puts **12–17 %** of its detections in `[0.50, 0.55)` —
which is the C79 observation (~7 % of detections flipping across the threshold on a re-encode)
seen from the distribution side, and it is consistent: half the near-boundary mass moving is what a
1.4/255 photometric perturbation would do. **`bus` is the one class that is not fragile** (7.7 %
below 0.55, 46 % above 0.90) and `traffic light` is the most fragile (37 % below 0.60, and the
smallest boxes in the vocabulary).

⚠️ **A distribution is not a reliability measurement and must not be read as one.** A tight
high-scoring cluster is equally consistent with a confident-and-right detector and a
confident-and-wrong one. That is §3's job, and only §3's.

**The liveness control behaves as designed** (its whole point is to sit far from the boundary):
`road` fired on **83/83** clips, `sky` on **81/83** — the two zeros are the known underpasses
(`24b6948f`, `a6b2719b`), which is why `is_live` is `any`, not `all`.

---

## 3. ⭐ THE LABELLED CHECK — precision per concept

**Protocol.** 244 detections, stratified by concept, uniform at random within concept, seed
`20260816`; `bus` (26) and `cyclist` (10) taken as a **CENSUS**, because a class whose whole
population is 10 must not be described by a sample of 6. Rendered as contact sheets from the same
frames SAM3 scored (obtained by calling `ph0_pilot.sample_clip_frames` itself, `t0_s=8.0`), cropped
at the video's native 640 px with a 6×-box context window.

⛔ **BLIND BY CONSTRUCTION — this is what makes §5 non-circular.** Each cell carried **only an
index**. Score, clip, frame and box were withheld until every verdict was fixed, then joined by
`r5_precision.py`. One concept per sheet, because the concept *is* the claim under test. A sheet
labelled `0.94` would have bought my agreement, and the threshold recommendation is derived *from*
the score↔correctness relation — so letting the score reach the eye would have made it
self-fulfilling.

**Intervals are the episode-cluster bootstrap over CLIPS** (`taniteval/ci.py`, 2000 draws) — never a
binomial and ⛔ never `overlapping_holdout_se`. Detections cluster hard in clips (one bad pole gives
several bad boxes across several frames), so the clip is the independent unit.

| concept | pop. | sampling | n | ✅ | ❌ | ❓ | **precision (resolvable)** | precision (❓ counted wrong) |
|---|---|---|---|---|---|---|---|---|
| `traffic sign` | 538 | SAMPLE | 64 | 44 | **6** | 14 | **0.880** [0.795, 0.958] · 33 clips | 0.688 [0.552, 0.800] |
| `car` | 1260 | SAMPLE | 48 | 39 | **1** | 8 | **0.975** [0.919, 1.000] · 28 clips | 0.813 [0.694, 0.917] |
| `traffic light` | 385 | SAMPLE | 32 | 21 | **0** | 11 | **1.000** [1.000, 1.000] · 12 clips | 0.656 [0.447, 0.852] |
| `pedestrian` | 180 | SAMPLE | 32 | 16 | **1** | 15 | **0.941** [0.786, 1.000] · 7 clips | 0.500 [0.261, 0.647] |
| `truck` | 97 | SAMPLE | 32 | 24 | **0** | 8 | **1.000** [1.000, 1.000] · 10 clips | 0.750 [0.516, 0.935] |
| `bus` | 26 | **CENSUS** | 26 | 21 | **0** | 5 | **1.000** [1.000, 1.000] · 9 clips | 0.808 [0.571, 0.963] |
| `cyclist` | 10 | **CENSUS** | 10 | 6 | **1** | 3 | **0.857** [0.667, 1.000] · 6 clips | 0.600 [0.417, 0.857] |

⛔ **THIS MEASURES PRECISION. IT DOES NOT AND CANNOT MEASURE RECALL** — that needs exhaustive
per-frame ground truth for every concept, which this corpus does not have. **A class can score
precision 1.00 here and still miss most of its instances**, and `cyclist` (10 detections in 83
clips) is exactly the shape where that matters. §4.2 shows a concrete miss. **Any downstream use
that needs "all the agents in this scene" is unsupported by this document.**

### 3.1 ⚠️ `unclear` is a finding about the INPUT, not about my diligence

**64 of 244 (26 %) could not be resolved at any magnification.** The rate tracks box size almost
perfectly: `traffic light` (34 px² median) 34 % unclear, `pedestrian` (120 px²) **47 %**, `bus`
(712 px²) 19 %. The frames SAM3 scores are **448×179**, so the median traffic light is about
**6×6 pixels**.

⇒ **The two precision columns BRACKET the truth and both are reported.** Quoting only the
resolvable column would be the optimistic read; quoting only the pessimistic one asserts that an
unreadable box is a wrong box, which is not established. ⭐ **The durable consequence is not a
number — it is that `pedestrian` and `traffic light` are operating below the resolution at which
their own output can be audited.** Neither a human nor, plausibly, the detector can be relied on
at 6×6 px, and no threshold fixes that. **Raising the SAM3 input resolution is a separate, larger,
and probably more valuable lever than any threshold.**

### 3.2 The nine false positives, by mechanism (`raw/adjudication_verdicts.json`)

| mechanism | n | examples |
|---|---|---|
| **empty box** — sky, foliage, blank facade | 4 | `#154` sky/foliage · `#179`, `#209` blank facade · `#185` brick strip |
| **a real object of the wrong class** | 4 | `#207` a pharmacy **red cross** on a wall · `#210` an **advertising hoarding** · `#1001` a **green traffic light** scored as `traffic sign` · `#113` a **doorway** scored as `pedestrian` |
| **a real agent of a neighbouring class** | 1 | `#77` a **pedestrian** scored as `cyclist` |
| **empty road surface** | 1 | `#50` `car` on a strip of road (a cyclist was in frame, and was *not* the boxed object) |

⭐ **The dominant sign failure is NOT hallucination on emptiness — it is SIGN-SHAPED THINGS THAT ARE
NOT TRAFFIC SIGNS.** A pharmacy cross, an illuminated shop sign, an advertising panel and a traffic
light are all high-contrast bordered rectangles/discs mounted at sign height. That failure is
**invisible to a score threshold** (`#207` scores **0.807**, the highest of the six) and invisible
to a KIND-blind consumer — which is exactly what `ph1_fuse.py`'s `goal_evidence` is. §6.1.

---

## 4. The three concerns on the table, settled

### 4.1 `traffic sign` vs G1's *"two thirds contained no sign"* — ⚠️ RECONCILED ONLY PARTLY, AND SAID SO

`Project Steering/G1_RESULT.md` (MEASURED, 2026-08-14) reports **~22 of 31 crops with no sign at
all**. This study reports **4/96 = 4.2 %** on the same subclass. Both are MEASURED. ⛔ A ~17×
disagreement is not settled by preferring the newer number, so I reproduced G1's protocol on this
corpus and separated its two components (`r7_g1_reconcile.py`, `raw/g1_reconcile.json`).

| difference tested | result |
|---|---|
| **A — the SELECTION.** G1 cropped *"the largest-area sign/light detection"* per clip; I drew uniformly. If sign false positives were preferentially LARGE, a max-area pick would concentrate them. | ⛔ **REFUTED.** G1's own max-area rule on this corpus (n=32, 32 clips) gives precision **0.926** [0.815, 1.000] — **no worse** than the uniform draw's 0.880 — and **ZERO** empty boxes. The mechanism runs the *other* way: median box area is **35.6 px² for false positives vs 74.8 px² for true ones**, so sign FPs here are SMALL and a max-area pick actively avoids them. |
| **B — the RENDERING.** G1 used a tight box crop, 4× LANCZOS, from the 448-px bridge; I use a 6×-box context window from native 640. | ⛔ **REFUTED.** The same 32 detections rendered both ways are not harder to read under G1's protocol — and on `#1000` G1's protocol was strictly **BETTER** (a tall thin priority sign that my context window shrank to a bollard). Both sheets are banked. |

⇒ ⚠️ **UNRESOLVED, stated as such: the remaining uncontrolled variable is the CORPUS.** G1 measured
`w120val` (600 clips, 4,048 sign detections); this study measured `aug120` (83 clips, 538).
**Neither number transfers.** G1's ⅔ must not be quoted as a property of SAM3's `traffic sign`
class in general — which is how `NEXT_4472_BUILD_INPUTS.md` §3 currently reads it — and this
study's 0.88–0.93 must not be quoted for `w120val`. **Work item: run this same adjudication on the
w120val sign leg** (offline, no GPU, ~2 h) before any val-side sign label is trusted.

### 4.2 `car` ↔ `cyclist` — ⛔ THE NAMED OBSERVATION IS REFUTED, and the real defect is different

The observation (correctly filed as *VISUAL / SINGLE FRAME — not a measurement*) was that on
`8f5df500` a mid-road cyclist carries boxes reading `car 0.72 / 0.83`. **Three tests:**

1. **Shared-box test, whole corpus** (`r5_precision.py`). For all **10** `cyclist` detections, the
   best IoU with any `car` detection on the SAME frame is **0.0** — not one object carries both
   labels. ⚠️ **This alone settles nothing** and I nearly reported that it did: the hypothesis is
   that `cyclist` stays SILENT while `car` fires on the rider, which produces exactly zero overlap
   because only one box is ever emitted. Recording a null from a test that cannot see the effect is
   the C13/C14 failure class.
2. **The named frame, looked at** (`r6_cyclist_probe.py`, `crops/cyclist_probe_8f5df500.jpg`,
   rendered from the **pipeline's own bridge bytes**). ⛔ **The `car 0.72` and `car 0.83` boxes are
   on TWO DISTANT CARS on the road ahead, to the LEFT of the cyclist.** The cyclist at run-frame 8
   carries **no box at all**. At run-frame 16 the same rider is boxed **`cyclist 0.85`**, tightly,
   with the two cars separately boxed `car 0.79 / 0.74`. ⇒ **Not confusion. A MISS at distance,
   recovered when the rider is closer.**
3. **The reverse direction, looked at.** Zero of the **48** adjudicated `car` crops contained a
   bicycle or a rider.

⇒ **`cyclist` is THIN, not BROKEN.** Its adjudicated precision is 6/7 resolvable (the one error is
a pedestrian, `#77`). ⚠️ **But the real limitation is RECALL, and it is visible in the same clip:**
the rider is present across the run frames and `cyclist` fires on **2 of 6** (`raw/cyclist_probe.json`
+ the record's own per-frame counts). **MEASURED for one tracked instance; it is not a corpus recall
number and must not be quoted as one** — but it is enough to say that `cyclist 10 across 83 clips`
should be read as *"a rare class that is also under-detected at distance"*, and that **no tactical
slot may assume `cyclist` enumerates the cyclists present.**

### 4.3 The `confidence_threshold=0.5` default — a vendor decision, now measured

It is confirmed a vendor default that nobody in this programme chose (§1), it is applied
irreversibly at inference (§1), and **12–17 % of every agent class sits within 0.05 of it** (§2),
which is the mechanism behind C79's re-encode flips. §5 decides what to do.

---

## 5. Threshold recommendation, per concept — mostly "DO NOT MOVE IT"

The score↔correctness relation, computed only after the verdicts were fixed (`raw/precision.json`):

| concept | scores of the FALSE POSITIVES | precision by band (resolvable) | **recommendation** |
|---|---|---|---|
| `traffic sign` | 0.516 · 0.552 · 0.567 · 0.619 · 0.675 · **0.807** | .50–.60 **0.769** · .60–.70 **0.714** · .70–.80 1.00 · .80–.90 0.938 · .90+ 1.00 | **CONDITIONAL — see below** |
| `car` | 0.694 (one) | ≥0.70 all 1.00 | ⛔ **KEEP 0.5.** One FP in 48. |
| `traffic light` | *(none)* | 1.00 in every band | ⛔ **KEEP 0.5.** |
| `truck` | *(none)* | 1.00 in every band | ⛔ **KEEP 0.5.** |
| `bus` | *(none)* | 1.00 in every band | ⛔ **KEEP 0.5.** |
| `pedestrian` | 0.530 (one) | ≥0.55 all 1.00 | ⛔ **KEEP 0.5.** One FP in 32. |
| `cyclist` | 0.551 (one) | ≥0.60 all 1.00 | ⛔ **NO RECOMMENDATION POSSIBLE.** |

**`traffic sign`, the only class with enough events to say anything.** Moving 0.50 → 0.70 raises
sample precision **0.880 → 0.967** and retains **274/538 (50.9 %)** of the class. But the trade is
uniform and unflattering: **~3 true signs are discarded per false one removed**, at every step of
the sweep. And ⚠️ **the band structure is not monotone** — 0.70–0.80 scores 1.00 while 0.80–0.90
scores 0.938, because the worst false positive (`#207`, a pharmacy cross) sits at **0.807**. ⇒

- **For a per-clip PRESENCE signal** (which is all any current consumer uses): **KEEP 0.5.**
  Precision 0.880 [0.795, 0.958] is adequate, and halving the detections buys nothing a presence
  flag can spend.
- **For a per-detection supervision channel** (agent slots, sign-anchored goals): **≥ 0.70**, and
  say in the record that it was applied.
- ⚠️ **Do not read the 0.70 as tuned.** It rests on **6** false positives. It is a defensible
  operating point, not a fitted parameter, and the honest reason to prefer it is §3.2's mechanism —
  five of the six FPs are low-scoring blank-region errors — not the sweep's shape.

⛔ **AND THE THRESHOLD IS THE WRONG LEVER FOR THE FAILURE THAT MATTERS.** The sign errors that would
corrupt supervision are **sign-shaped non-traffic-signs** (pharmacy cross 0.807, advertising panel,
traffic light) and **no threshold separates them** — they are confidently, correctly detected as
"a sign-like object". Fixing that needs a **KIND** check, not a score cut. §6.

⛔ **For `cyclist` I explicitly refuse a recommendation.** The whole population is 10 and there is
one false positive. Fitting 0.60 to that single event is exactly the "threshold fitted to a handful
of frames" the brief forbids, and it would discard 2 of 10 detections of the class we most need
more of.

---

## 6. Consequence for the 4,472-clip build

### 6.1 ⛔ `goal_evidence: grounded` — emit `not_computable`, and NOT because of SAM3

`stack/scripts/ph1_fuse.py:469-474`:

```python
n_sign_frames = sum(1 for t in tracks if t["concept"] == "traffic sign")
cor["goal_evidence"] = {..., "verdict": ("grounded" if ev is not None and n_sign_frames > 0
                                         else "provisional"), ...}
```

The predicate is **KIND-blind, FRAME-blind and THRESHOLD-blind**: any `traffic sign` track anywhere
in the clip at any score ≥ 0.5 grounds the claim. This study establishes that those tracks are
**~88–93 % real signs** — so the *inputs* are fine. **The predicate is still wrong**, for reasons
this study does not repair:

1. It never checks the sign **KIND**. `S2_STRATEGIC_GAP.md` §4.3 (MEASURED) — the evidence sign is
   not even `nav` on **24/31**: speed 15, other 6, yield 2, stop 1. A give-way triangle currently
   grounds a `route_to <place>` claim.
2. **G1 is CLOSED at 0/31**: the sign *texts* are unverifiable at 448 px, several with the shape of
   VLM priors. The claim `grounded` asserts is about a text nobody can read.
3. It never checks that the sign is on the **frame the VLM grounded on**.

⇒ **`goal_evidence` must emit `not_computable` at 4,472 scale, with the reason named**, or be
renamed to the far weaker thing it measures. ⚠️ This is the PI's own live lesson applied one layer
along: `PI_REVIEW_FINDINGS.md` found ≈78 % of `LANE_TARGET` wrong and the fix was **to stop
emitting it**; the root cause there was *"an absent measurement rendered as a confident negative"*,
and `grounded` is the same shape — a weak measurement rendered as a confident positive. **PI/owner
decision required. `ph1_fuse.py` is owned by the lane-change agent this session; I have not
touched it.**

### 6.2 What may carry supervision at 22× scale

| concept | verdict for the 4,472 build | why |
|---|---|---|
| `bus` · `truck` | ✅ **TRUSTWORTHY** at 0.5 | 0 false positives in 58 adjudicated (26 of them a CENSUS); high, well-separated scores |
| `car` | ✅ **TRUSTWORTHY** at 0.5 | 0.975 [0.919, 1.000], n=48 over 28 clips |
| `traffic sign` | 🟨 **PRESENCE ONLY** at 0.5; **≥0.70** if it becomes per-detection supervision | 0.880 [0.795, 0.958] — but ⛔ **never as evidence of a sign's KIND or TEXT** (§6.1) |
| `cyclist` | 🟨 **PRECISION OK, RECALL NOT ESTABLISHED** | 6/7 resolvable correct, but a MEASURED miss at distance on the one clip examined. ⛔ **No slot may treat it as an enumeration of the cyclists present** |
| `pedestrian` · `traffic light` | ⛔ **`not_computable` for any per-detection claim** | 0–1 false positives, but **47 % / 34 % of their detections are unauditable at 448×179** (median box 120 px² / **34 px²**). We cannot certify what we cannot see, and *an unreliable label is worse than an absent one*. Admissible only as a **per-clip presence flag**, where the resolvable precision (0.94 / 1.00) is what applies |

### 6.3 ⭐ ONE ENGINEERING CHANGE WORTH MAKING BEFORE THE BIG BUILD — it permanently removes this gate

**Lower the vendor threshold at inference and bank the sub-threshold tail.** The forward pass is
already paid for; `keep = out_probs > 0.5` throws the tail away for free, and that is the *only*
reason this study could not sweep downward. One line in `ph0_sam3.build_processor`:

```python
Sam3Processor(model, confidence_threshold=0.25)      # bank the tail; filter downstream
```

Our own `min_score` then becomes the real, auditable knob it was always documented to be, and
**every future threshold question is offline re-analysis instead of 26 GPU-hours of re-detection.**

⚠️ **Size it first, do not just do it.** The record carries `rle_rows` per detection, so an unknown
number of extra detections is an unknown size multiplier. **Pilot on 5 clips, measure the detection
count and the record bytes, and if the growth is large, bank sub-0.5 detections with `score` +
`box_xyxy` only (no mask/RLE)** — which answers every threshold question at a bounded cost.

---

## 7. The re-detection job, SPECIFIED AND NOT RUN (per the brief)

Thor is running the 30k v6F S-W (PID 25477) and is the only compute; free-Colab's T4 budget is
spent. **Nothing below was executed.**

| job | scope | cost | when it is needed |
|---|---|---|---|
| **D1 — downward sweep pilot** | 5 clips, `confidence_threshold=0.25`, `--n 5` | **~2 min T4** (~21 s/clip, MEASURED `SAM3_DTYPE_FIX.md` §3b.1) | to size §6.3's record growth before adopting it |
| **D2 — downward sweep, aug120** | the 83 fixed-engine clips at 0.25 | **~29 min T4** | only if a threshold BELOW 0.5 is ever wanted. **This study gives no reason to want one** |
| **D3 — finish the corpus** | the 32 still-stale C77 clips | **~11 min T4**, `code/backfill2.py` unchanged (`SAM3_DTYPE_FIX.md` §4.1) | ⚠️ **independent of this study and already outstanding** |
| **D4 — the 4,472 build's SAM3 leg** | 4,472 clips | **~26 h T4** at 21 s/clip, threshold-independent | the build itself. §6.3 is free *if applied at the same time* and costs a full re-run if applied after |

⭐ **§6.3 is only free if it lands BEFORE D4.** Applying it afterwards means re-running 26 GPU-hours.

**Offline (no GPU), and the one I recommend next:** the w120val sign adjudication of §4.1 —
~2 h of the same pipeline in this package, and it is what makes G1's number and mine commensurable.

---

## 8. What this study does not say

1. ⛔ **No recall, for any concept.** Precision only. §4.2 gives one MEASURED per-instance miss;
   that is an existence proof, not a rate.
2. ⛔ **Nothing about `w120val`** (600 clips, 4,048 sign detections) — §4.1.
3. ⛔ **Nothing about the 32 stale C77 clips**, excluded by design.
4. ⚠️ **Single adjudicator, no second reader.** There is no inter-rater agreement number here. The
   verdicts are banked per index with the sheets, so a second reader can be run against them
   without re-rendering.
5. ⚠️ **The crops come from the pre-bridged HF videos, not the pipeline's own re-bridge, for 63 of
   64 clips.** This is admissible **for adjudication** and was MEASURED, not assumed
   (`r3_encode_equivalence.py`): identical frame count and geometry, photometric Δ **1.36/255**
   whole-frame and **2.22/255 inside the banked boxes**, and frame-aligned (same-index Δ 1.36 vs
   next-index Δ 9.95, a 7.3× separation on 6 of 7 clips; the 7th is a near-static scene with no
   boxes). ⛔ **It remains INADMISSIBLE for comparing detection COUNTS** — that is C79 and it stands.
   The one named frame in §4.2 was rendered from the pipeline's own bridge anyway.

---

## 9. Deliverable manifest

⛔ Every row resolved with `git ls-files --cached` / `--stage` before filing (C78 + the 2026-08-16
sharpening). **STAGED, NOT COMMITTED, NOT PUSHED**, branch `agent/arch-inf-20260803`.

### This package (all in `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-16-sam3-concept-reliability/`)

| artifact | path | what it is |
|---|---|---|
| this report | `SAM3_CONCEPT_RELIABILITY.md` | — |
| record pull + the §1 answer | `code/r1_pull_records.py` → `raw/records_index.json` | per-clip index; **corpus minimum score 0.5** |
| score distributions | `code/r2_score_dist.py` → `raw/score_distribution.json` | §2, full histograms in 0.02 bins |
| encode-admissibility measurement | `code/r3_encode_equivalence.py` → `raw/encode_equivalence.json` | §8.5 |
| sampling + BLIND sheets | `code/r4_sample_and_render.py` → `raw/adjudication_sample.json` | 244 detections, index→metadata |
| **the verdicts** | `raw/adjudication_verdicts.json` | 244 human calls + the mechanism notes |
| precision + cyclist test | `code/r5_precision.py` → `raw/precision.json` | §3, §4.2, §5 |
| the named-frame probe | `code/r6_cyclist_probe.py` → `raw/cyclist_probe.json`, `crops/cyclist_probe_8f5df500.jpg` | §4.2 — **the figure that refutes the confusion claim** |
| G1 reconciliation render | `code/r7_g1_reconcile.py` → `raw/g1_reconcile_sample.json` | §4.1, both protocols |
| G1 reconciliation verdicts + join | `raw/g1_reconcile_verdicts.json`, `code/r8_g1_verdict_join.py` → `raw/g1_reconcile.json` | §4.1 |
| contact sheets (16 + 4) + the probe figure | `crops/*.jpg` — 21 files, 6.0 MiB | ⚠️ **`crops/` is the ONLY copy of the rendered evidence** — the scratchpad PNGs are session-local. ⚠️ **The adjudication was performed on the PNG originals**; these are JPEG q=90, 4:4:4 (no chroma subsampling), 23.2 → 6.0 MiB, so the repo copy is a faithful-but-not-bit-identical record. Every script re-renders the PNGs deterministically from the banked sample JSON if a bit-exact re-read is ever needed |

### Read, not modified

`stack/scripts/ph0_sam3.py` (I own it and **changed nothing** — verified `git diff` empty; its 33
tests pass in isolation), `stack/scripts/ph1_fuse.py` and `Project Steering/G1_RESULT.md`
(read-only; owned by others), `taniteval/taniteval/ci.py` (imported for the estimator).

### Far side — nothing written

This package **pulled only**: `Sayood/tanitad-ph0-aug120` → `sam3_backfill/*.json` (115) and
`bridged_w120train_2400/videos/*.mp4` (64). **No HF write, no pod, no GPU.**

### Suite

⚠️ **I modified no code under `stack/`**, so this package's suite delta is **zero by construction**
(`git diff --stat -- stack/scripts/ph0_sam3.py` empty). I deliberately did **not** quote a full-suite
number: `ph1_fuse.py`, `s2_labels.py`, `train_v6_staged.py` and two test files are being edited by
live sibling agents right now, so any total I measured would be their delta, not mine. The one file
I own was run in isolation: `tests/test_ph0_sam3.py` → **33 passed in 2.24 s**.

---

## 10. Escalations — decisions, not documentation

1. ⛔ **`goal_evidence: grounded` must be changed or retired before the 4,472 build** (§6.1). It is
   not a SAM3 reliability problem and this study does not fix it. Owner: whoever owns
   `ph1_fuse.py`; PI call on whether the token survives at all.
2. ⛔ **`NEXT_4472_BUILD_INPUTS.md` §3 cites G1's ⅔ as a property of the SAM3 sign class.** That
   reading is not supported on `aug120` (§4.1) and the sentence should be corrected to name the
   corpus. Owner: the aug120-fusion package.
3. ⚠️ **The w120val sign leg has never been adjudicated** and G1's number cannot stand in for it.
   Offline, ~2 h, no GPU, same code as this package.
4. ⭐ **§6.3 (bank the sub-threshold tail) must be decided BEFORE the 4,472 SAM3 pass**, or it costs
   a 26-GPU-hour re-run instead of nothing.
5. ⚠️ **`traffic sign` count is 538, not 508** — fix at source in the brief and anywhere it
   propagated.
6. ⚠️ **A `RETRACTION_LOG.md` entry is warranted and I did not write it**, because that file
   currently carries another agent's STAGED changes and editing it would risk sweeping their work
   into mine. **Proposed class, for whoever holds the log:** *a number MEASURED on one corpus,
   quoted as a property of the DETECTOR.* G1's ⅔ was measured on `w120val` and travelled into
   `NEXT_4472_BUILD_INPUTS.md` §3 and `S2_STRATEGIC_GAP.md` §4.5 as *"the known sign-class
   reliability flag"* — i.e. as a fact about SAM3's `traffic sign` class, which it is not. ⭐ **This
   is the same family as the `df` / Thor `free` / cgroup `usage_in_bytes` traps already in
   `CLAUDE.md`: a measurement whose SCOPE is narrower than the claim it gets used for.** G1 itself
   scoped the finding correctly and hedged it (*"if this false-positive character generalises"*);
   the scope was lost in the citations downstream, which is where the rule needs to bite.
