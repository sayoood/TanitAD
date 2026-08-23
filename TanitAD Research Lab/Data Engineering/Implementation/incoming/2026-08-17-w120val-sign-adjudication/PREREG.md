# PRE-REGISTRATION — the `w120val` sign adjudication

**Written and banked BEFORE a single crop was rendered or looked at.** Every rule below is
committed in advance, with **both outcomes named**, because the brief forbids letting the answer
drift toward whichever release is more convenient for the 4,472-clip build.

**Owner:** arch-inf agent, 2026-08-17 · branch `agent/arch-inf-20260803`
**Gates:** `…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md` §3.
**Closes (or fails to close):** `…/2026-08-16-sam3-concept-reliability/SAM3_CONCEPT_RELIABILITY.md`
§4.1 — *"the remaining uncontrolled variable is the CORPUS"*.

---

## 0. ⛔ A SCOPE CORRECTION ESTABLISHED BEFORE THE EXPERIMENT, FROM PRIMARY SOURCES

The reliability study §4.1 and `NEXT_4472_BUILD_INPUTS.md` §3 both state that G1 measured
**`w120val` (600 clips, 4,048 sign detections)**. **The primary sources say something narrower:**

| source | verbatim |
|---|---|
| `Project Steering/G1_SIGN_OCR_GRADING_SHEET.md:5` | *"These are the **31 non-empty OCR texts** from the **50-clip pilot** — the pre-registered sample."* |
| `Project Steering/G1_RESULT.md:4-5` | *"for each of the 31 pre-registered pilot OCR texts, the sign was cropped from **the pilot videos** using SAM3's boxes"* |
| `Project Steering/G1_RESULT.md:33-35` | *"**Production** banked 4 048 'traffic sign' detections on the 600-clip set; **if** this false-positive character generalises …"* — a SCALE sentence and an explicit hedge, not the corpus of the measurement |

⇒ **G1's ~⅔ was measured on 31 crops from 30 clips of the 50-clip pilot**, not on the 596-clip
production leg. G1 scoped itself correctly; the scope was lost downstream.

**And the structural fact that reframes the whole confound (MEASURED, `raw/records_index_*.json`,
this package's own pull):**

| leg | HF path | clips | `traffic sign` det | min banked score |
|---|---|---|---|---|
| **w120val (production)** | `Sayood/tanitad-ph0` → `ph0_prod4/sam3/sam3.json` | **596** | **4 048** over 440 clips | **0.5000** |
| **pilot-50 (G1's own)** | `Sayood/tanitad-ph0` → `ph0_pilot50/sam3/sam3.json` | **50** | **292** over 37 clips | **0.5002** |

⭐ **The pilot-50 is a strict SUBSET of the production leg — overlap 50/50, pilot-only 0** — and all
30 distinct G1 clip prefixes resolve inside both. Both legs are the same `physicalai-val-
0c5f7dac3b11-w120-256x640cyl` corpus, same bridge, same `frame_wh [448, 179]`, same vendor
threshold. **So "G1's corpus" and "w120val" are not two corpora; the second contains the first.**

⚠️ That does **not** pre-decide the answer. `aug120` (the reliability study's corpus) is still a
different split — its clips come from `physicalai-train-e438721ae894-w120-256x640cyl` — so a
train/val difference remains testable. But it does mean this package can do something better than
compare two corpora: it can **re-run G1's protocol on G1's own clips**, and put the
adjudicator, the selection and the rendering each on their own axis.

---

## 1. The three arms, all pre-committed

| arm | corpus | selection | rendering | n | what it isolates |
|---|---|---|---|---|---|
| **A — the brief's ask** | **w120val**, 596 clips / 4 048 sign det | **uniform at random** within `traffic sign`, seed `20260817` | this study's 6×-box context window from native 640 | **64** | the headline: is the val-side sign channel like `aug120` (0.880) or like G1 (~⅓)? |
| **B — G1's rule, G1's clips** | **pilot-50**, 292 sign det over 37 clips | **G1's**: the LARGEST-AREA `traffic sign` detection per clip — **CENSUS**, all 37 | **both** protocols per detection (G1's tight 4× LANCZOS from the 448 bridge, and the context window) | **37** | selection × rendering, on the corpus G1 actually measured |
| **C — G1's own banked tiles, re-read** | G1's **exact 54 crops**, `Sayood/tanitad-ph0-aug120` → `g1_evidence/crops/row01…row54.jpg` | G1's | G1's — **the identical JPEG bytes G1 read** | **54** | ⭐ the ADJUDICATOR, and nothing else. Same corpus, same selection, same rendering, same pixels |

**Arm A is the deliverable.** B and C serve the corpus verdict and are explicitly secondary; if the
turn is cut short, A is banked first and alone.

---

## 2. Protocol — deliberately NOT a new one

⛔ **No second adjudication protocol is written.** The study's own harness is reused:

- **`r5_precision.py` is run UNCHANGED** for the point estimate, the CI, the bands and the sweep.
- **`crop_cell()` is imported from `r4_sample_and_render.py`**, so arm A's cells are rendered by the
  same function, same 6× context, same BICUBIC, same 320-px cell, same gold box.
- The G1-tight rendering in arm B is the code path from **`r7_g1_reconcile.py`**, unchanged
  (tight box crop → 4× LANCZOS → letterboxed into the same cell).
- **`r8_g1_verdict_join.py` is run UNCHANGED** for the head-to-head and the box-area mechanism.
- Intervals: **`taniteval.ci.episode_cluster_bootstrap` over CLIPS**, 2 000 draws — ⛔ never
  `overlapping_holdout_se`, never a binomial.

**Declared PROTOCOL DELTAS, all three of them, stated here in advance:**

1. ⚠️ **The frames come from the pipeline's OWN bridge, not from pre-bridged HF videos.** No
   w120val videos are banked, so each sampled clip's `<clip>.v2ep.pt` is pulled from
   `Sayood/tanitad-physicalai-w120-256x640cyl/physicalai-val-0c5f7dac3b11-w120-256x640cyl/` and
   bridged with **`stack/scripts/v2_to_pilot.py`'s own `decode_full_episode` → `stacked_to_rgb` →
   `write_mp4`**, then read with `ph0_pilot.sample_clip_frames(t0_s=8.0)` — the same call
   `ph0_sam3.py:1472` makes. **This is STRICTLY STRONGER than the study's arrangement**, which used
   pre-bridged videos and had to measure encode-equivalence to justify it (its §8.5). It removes
   that caveat rather than adding one.
2. ⚠️ **The staleness gate differs because the corpus has no stale subset.** `r2_score_dist.load()`
   drops records with no `liveness` block (the aug120 C77 filter). The production engine predates
   the liveness probe, so that gate would drop **100 %** of these records. It is replaced by
   "the record has frames", and the justification is measured, not assumed: **`n_errors` is 0 on
   both legs**, so there is no stale subset to exclude.
3. ⚠️ **Arm A adjudicates `traffic sign` only.** The other six concepts are out of scope for this
   package and **no claim will be made about them on `w120val`**. Reason: each distinct clip costs
   a ~36 MB episode pull, and the brief's question is the sign leg.

**Blindness — unchanged and non-negotiable.** Every rendered cell carries **only an integer index**.
Score, clip id, frame index and box geometry are withheld in the sample JSON and joined **after**
every verdict is fixed. Verdict vocabulary is exactly `correct` / `wrong` / `unclear`, the three
`r5_precision.py` asserts. `unclear` is **never** a soft `wrong`; precision is reported **both**
ways (resolvable-only, and unclear-counted-as-wrong) because those bracket the truth.

**The G1 subclass is scored separately, as G1 defined it.** `G1_RESULT.md:17` — *"no sign visible in
the crop at all — sky, foliage, building walls, clouds"*. A `wrong` cell containing a real,
salient, non-traffic-sign object (shop sign, pharmacy cross, advertising panel, traffic light) is
**NOT** in this subclass. That distinction is the whole disagreement and is recorded per index.

---

## 3. ⛔ THE DECISION RULE — committed now, in numbers

Let, on **arm A** (w120val, n=64):

- **P_val** = precision on resolvable cells, episode-cluster bootstrap over clips;
- **E_val** = the **"no sign at all" subclass rate over ALL 64 adjudicated cells** (not just
  resolvable) — this is the quantity directly comparable to G1's **~22/31 ≈ 0.71** and to the
  reliability study's **4/96 ≈ 0.042**.

| # | outcome I will report | pre-committed trigger |
|---|---|---|
| **1** | **THE CORPORA GENUINELY DIFFER.** Every sign number carries its corpus forever; the val-side sign channel is not covered by the `aug120` result and the 4,472 build inherits a real constraint. | **E_val ≥ 0.40** (in G1's neighbourhood, not `aug120`'s) **OR** the upper end of P_val's 95 % CI **< 0.70** (clearly worse than `aug120`'s [0.795, 0.958]) |
| **2** | **G1's ORIGINAL READING DOES NOT GENERALISE, and the reliability study's picture holds on the val side too.** The 4,472 build is released from this particular constraint (⚠️ **only this one** — §6.1's KIND-blindness and the G1 text gate are untouched by any outcome here). | **E_val ≤ 0.15** **AND** P_val's 95 % CI **overlaps** [0.795, 0.958] |
| **3** | **NEITHER CLEANLY.** I say so, name every variable still uncontrolled, and refuse a release. | anything else — including a wide CI, an `unclear` rate high enough that the resolvable arm is a minority of the sample, or A and B disagreeing with each other |

⚠️ **If arms A/B/C disagree, outcome 3 is the answer regardless of what arm A alone says.** A
convenient headline that its own replications contradict is not a result.

**Secondary, pre-committed, and NOT allowed to override the primary rule:**

- **Arm C is the sharpest instrument and I say so in advance.** If C's empty-box rate lands near
  **0.71**, G1's read is reproduced on G1's own pixels and the disagreement is about *corpora or
  selection*. If C lands near **0.04–0.15**, then the disagreement is about **the adjudicator**, and
  the honest conclusion is that **G1's ⅔ was a reading error on evidence that is still on disk** —
  which I commit to reporting **even though it is the more embarrassing finding for this
  programme's own record**, and to filing as a `RETRACTION_LOG.md` class.
- **A threshold recommendation is the LAST priority and may be omitted.** It is admissible only if
  arm A yields **≥ 5 false positives** — the reliability study fitted 0.70 to six and warned
  explicitly that it "is not tuned". Below 5, I state that no threshold statement is supportable.

## 4. What this package will NOT claim, whatever comes out

1. ⛔ **No recall, for any concept, on any leg.** Precision only. Same limit as the study.
2. ⛔ **Nothing about the other six concepts on `w120val`** (delta 3 above).
3. ⛔ **Nothing that re-opens the sign KIND or TEXT questions.** `goal_evidence` is already retired
   and the G1 text gate is CLOSED at 0/31; a good precision number here does not reopen either.
4. ⛔ **No edit to `…/2026-08-16-sam3-concept-reliability/`** — its record stands as filed; this is a
   NEW package that extends it.
