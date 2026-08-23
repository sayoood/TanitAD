# Tactical label strategy — the three-tier waterfall, and the scale it can reach

**Date:** 2026-08-18 · **PI priorities, verbatim:** (1) use Alpamayo maximally — it is good and
already available for the augmented set, but it does not carry all the information; (2) use semantic
information from a VLM via very precise, detailed prompts and rich input (past/future); (3) close
remaining gaps with ego information, **only in very clear situations without large interpretation**.

This document turns that ordering into an assignment of **axes to tiers**, a precedence rule, an
admissibility bar per tier, and the **measured scale** each tier can reach.

---

## 0. THE ONE INSIGHT THAT ORGANISES EVERYTHING

The three tiers are **not three opinions about the same label**. They are **good at different
axes**, and the screening measured which:

| axis | what it needs | who can supply it |
|---|---|---|
| **LATERAL** (`a_tac_lat`) | a lane-topological fact — keep / change / shift / traverse a junction | ⭐ **Alpamayo** — it emits exactly this typing |
| **LONGITUDINAL** (`a_tac_lon`) | a **REASON** — FOLLOW vs BRAKE_TO vs YIELD_MERGE vs CREEP | ⛔ **not Alpamayo** (it emits magnitude); ⭐ **VLM** |
| both, in trivial cases | a kinematic fact with no semantic content | **ego**, and only there |

⇒ The waterfall is **per axis**, not per clip. A clip can be Tier-1 lateral and Tier-2 longitudinal
simultaneously, and usually is.

---

## 1. TIER 1 — ALPAMAYO, USED TO ITS LIMIT

**Owns: the LATERAL axis.** Mapping (all `MEASURED`, n = 4,729):

| Alpamayo LANE | → `a_tac_lat` | n |
|---|---|---:|
| Lane Keep | `LANE_KEEP` | 4,035 |
| Left / Right Lane Change | `LANE_CHANGE_L` / `_R` | 22 / 82 |
| Slightly Shift Left / Right | `NUDGE_L` / `NUDGE_R` | 69 / 31 |
| **Turn Left / Turn Right** | ⭐ **`TURN_L` / `TURN_R`** — *newly representable in v6.1* | 85 / 101 |
| *(no source)* | `ABORT_LC` | 0 — never emitted by this leg |

⇒ **v6.1 raises Alpamayo's lateral coverage from 96.06 % to 100 % of parseable rows.** That is the
entire practical payoff of the TURN tokens, and it is why they were worth adding.

**What Tier 1 must NOT be asked for:**
* **The longitudinal axis.** Its values are magnitude-typed (*Gentle Deceleration*); ours are
  reason-typed. *"Gentle Deceleration"* cannot distinguish FOLLOW from BRAKE_TO. Every value maps to
  `REASON_REQUIRED` and **the leg casts no longitudinal vote**. Restoring the old substring-over-`cot`
  path would be a lottery, not a label.
* **The 304 `Stop` rows' lateral value.** They emit one axis; the others are NOT-APPLICABLE.
* **Accuracy claims.** PhysicalAI-AV is listed as Alpamayo *training* data, overlap UNRESOLVED, and
  the labels are ONE draw at temperature 0.6. Tier 1 is admissible for **coverage**, and its
  correctness is what Tier 2 and the visual review are for.

---

## 2. TIER 2 — VLM, AND ITS ONE REAL JOB

**Owns: the LONGITUDINAL axis** — i.e. the **reason**. This is not a fallback for Tier 1; it is the
only tier that can produce the quantity at all.

### 2.1 Why a VLM and not ego kinematics

FOLLOW, BRAKE_TO, YIELD_MERGE, CREEP and HOLD are **all consistent with the same deceleration
profile**. What separates them is *what is in the scene*: a lead vehicle, a stop line, a merging
agent, an occlusion. That is semantic content in the image, and ego kinematics cannot recover it —
this is exactly the `long_accel` result (`STREAM D`: unrecoverable from frozen latents, 17 arms).

### 2.2 Input design — rich, and deliberately including the future

⭐ **The future is admissible and should be used.** `CLAUDE.md`'s binding rule is *labels may use
ego; inference is vision-only*. A labeller that sees only the past must **guess** whether the
deceleration resolves into a stop, a follow, or a yield — the future resolves it directly.

Per clip, the VLM receives:
1. **Frames spanning −1.0 s → +6.0 s**, with the 2–6 s decision band marked (the geometry the
   existing review sheet already renders).
2. **The lead track**, from our obstacle join (2,308 eps / 433,040 frames / 12.1 M boxes) — gap,
   time-gap, closing rate at t₀. This converts *"is there a lead"* from a perception question the
   VLM might get wrong into a **given**.
3. **Ego kinematics** as numbers: v(t₀), v(t₀+2 s), the speed profile over the band.
4. ⚠️ **Alpamayo's magnitude + `cot` as a PRIOR, presented as such** — not as an answer. It is a
   teacher's opinion, and telling the VLM otherwise imports Alpamayo's errors as anchors.

### 2.3 Prompt design — three rules that decide whether this works

1. **Forced choice over OUR six tokens, with an explicit `ABSTAIN`.** Never free text; never a
   token outside `TACTICAL_LON_ACTIONS`. `ABSTAIN` must be as easy to choose as any label, or the
   model will pick the plausible one — that is how the old substring lottery happened.
2. **Evidence before verdict.** Require the referent first (*"which object or road feature is the
   ego responding to?"*), then the token. A reason label whose referent is unnameable is not a
   reason label, and this makes that failure visible instead of silent.
3. **One axis per call.** Asking for lateral and longitudinal in one response reintroduces, in the
   prompt, exactly the mixing the factored vocabulary exists to remove.

### 2.4 The bar Tier 2 must clear before its output is used anywhere

⛔ **A VLM leg that has not been scored against human adjudication is a second unvalidated teacher,
not a validation of the first.** The gate: run it on the **42 clips already selected and rendered**
in the 2026-08-16 visual review, whose strata are chosen to include the hardest cases, and report
agreement per stratum against the human verdicts. **The review is currently 0/42 adjudicated** — so
the human pass is on the critical path of the VLM tier, not parallel to it.

---

## 3. TIER 3 — EGO, WITH A HARD CEILING

**Owns: nothing by default.** It fires only where the mapping is a **function of measured
quantities with no semantic content**. Proposed admissible rules — each must be *derivable without
naming an object*:

| token | rule | why it is interpretation-free |
|---|---|---|
| `HOLD` | v < 0.5 m/s for the **whole** band **and** at both endpoints | "stopped and stays stopped" — no referent needed |
| `CRUISE` | \|Δv\| < 0.5 m/s over the band **and** no lead within 60 m (from the join) | absence of a referent is itself the condition |
| `LANE_KEEP` | lateral offset and heading both inside their straight-driving bands | a geometric fact |

⛔ **Ego may NEVER decide:** FOLLOW vs BRAKE_TO vs YIELD_MERGE vs CREEP (all need a referent);
`TURN_L`/`TURN_R` vs a curved `LANE_KEEP` (needs junction topology, which the corpus does not carry);
`ABORT_LC` (needs the intent that was abandoned).

⚠️ **The trap this tier must avoid.** The programme has already measured what happens when an ego
signal is admitted too freely: `vtarget_guarded` is *"hindsight EGO geometry — what speed did this
driver settle at, not what speed is permitted here"*, and on ego inputs **nothing beats repeating
v₀'s band (0.4066 free vs 0.2465 for the trained classifier)**. An ego-derived label that merely
restates the ego's own future is an **echo**, and echoes train models to predict themselves.
⇒ Every Tier-3 rule must pass one test before adoption: *would a model that only echoes v₀ satisfy
it?* If yes, the rule is void.

---

## 4. PRECEDENCE, PROVENANCE, AND CONFLICT

**Precedence** (per axis, highest first): **Tier 1 → Tier 2 → Tier 3 → abstain.**
⛔ **Abstention is a legitimate terminal state.** `NO_CLAIM` beats a coerced label; the whole point
of the 186 unrepresentable clips being *declared* rather than *coerced* was this.

**Every label carries its provenance leg** — `alpamayo` / `vlm` / `ego` / `fused` — and, when fused,
which legs agreed. The existing fusion already records this; the rule is that **no label is ever
anonymous**, so any later retraction has a blast radius that can be computed rather than guessed.

⚠️ **Agreement between VLM and ego is NOT corroboration** if the VLM was shown the ego kinematics
(§2.2 item 3) — that is the **ECHO** the review sheet already flags as its `B_TWO_OF_THREE` caveat.
⇒ The VLM's longitudinal call must be recorded **both** with and without the ego numbers in the
prompt, on at least the 42-clip review subset, so the echo can be measured rather than assumed away.

---

## 5. THE SCALE — what each tier can actually reach today

| population | n | source |
|---|---:|---|
| Alpamayo-labelled clips | **4,729** | `MEASURED`, taxonomy artifact |
| — of which lateral is parseable | 4,425 | 4,729 − 304 one-axis `Stop` rows |
| — of which lateral is representable in **v6.0** | 4,239 | 4,425 − 186 TURN |
| — of which lateral is representable in **v6.1** | ⭐ **4,425 (100 %)** | with TURN_L/TURN_R |
| Clips with **our w120 video** (aug120) | **201** | `MEASURED` |
| — with Alpamayo | **201 / 201** | `fused_aug120_summary.json` |
| — inside the parity train corpus | **201 / 201 (100 %)** | set identity, `MEASURED` |
| — with SAM3 perception | 86 (115 missing) | `AUG120_SAM3_STAGE_GAP` |
| — corroborated / conflicting today | 88 / 10 | fusion summary |
| Parity train corpus | **2,376 episodes** | key `physicalai-train-e438721ae894` |
| Obstacle join (lead tracks for Tier 2/3) | 2,308 eps · 433,040 frames · 12.1 M boxes | `MEASURED` |
| Visual review sheet, rendered | 42 of a 98-clip pool | **0 / 42 adjudicated** |

### 5.1 ⭐ MEASUREMENT #1 — RUN, AND IT REVERSES THE PLAN'S EMPHASIS

**`MEASURED` 2026-08-18, this session:**

```
|Alpamayo ∩ parity-train selection| = 201
   as % of Alpamayo (4,729):            4.25 %
   as % of the training selection:      8.38 %
   Alpamayo clips NOT in training:      4,528
   training episodes with NO Alpamayo:  2,199
```

⭐⭐ **And the 201 are SET-IDENTICAL to the 201 aug120 clips** — verified as set equality, not as
equal cardinality (two counts matching is not the same claim, and this programme has retracted that
confusion before). Sanity check on the denominator: the selection carries **2,400** rows and the
canonical corpus is **2,376** — the difference is exactly the 24-clip skip-hash `f09e44db`.

⛔ **THE TWO CEILINGS ARE ONE CEILING.** I had written §5.1 expecting *"Tier 1 is not video-bound,
so it reaches further than Tier 2"*. That is **false**. The set of clips Alpamayo labels **inside our
training corpus** and the set for which we hold w120 video are **the same 201 clips**. There is no
population that Tier 1 can reach and Tier 2 cannot.

⇒ **91.6 % of the training corpus (2,199 of 2,400 episodes) has NO tactical label from ANY tier
today**, and no reordering of the three tiers changes that by a single clip.

### 5.2 WHAT THIS DOES TO EACH PRIORITY

| PI priority | verdict after Measurement #1 |
|---|---|
| **1. Use Alpamayo maximally** | ⭐ **Correct, and now precisely bounded.** It is *"already available for our augmented data set"* — and for **exactly** that set, 201 clips. Squeeze it fully there (v6.1 takes its lateral coverage to 100 % of parseable rows), but it cannot be the corpus-scale answer. |
| **2. VLM with rich past/future input** | ⭐⭐ **Promoted from gap-filler to the main line.** It is not merely the only source for the longitudinal *reason* on 201 clips — it is the only path to labelling the other **2,199** episodes on **both** axes. Its ceiling is w120 video, not model capability. |
| **3. Ego, only in clear cases** | **Unchanged in scope, but it is the only tier that scales for free today** — it needs no video and no teacher. That makes its discipline *more* important, not less: it is the tier most tempted to overreach, and §3's echo test is what stops it. |

### 5.3 THE LEVER, NAMED

**The w120 extraction for the training corpus is the binding lever on tactical-label scale.** It is
CPU-only, already planned, and deferred to Thor post-30k. Everything else in this plan is bounded by
it. Two sub-questions it raises, both of which should be answered before the job runs:

1. **Do the other 2,199 training episodes need Alpamayo at all, or only video?** If Tier 2 is the
   labeller, video alone suffices and no dependency on NVIDIA's teacher remains — which also removes
   the unresolved training-data-overlap contamination from 91.6 % of the corpus.
2. **What is the per-episode cost of w120 extraction × 2,199?** Unmeasured. It sets the schedule and
   should be measured on ~10 episodes before the full job is committed.

⚠️ **A correction to §5's earlier framing, kept visible rather than silently edited:** the sentence
*"Tier 1 is not video-bound"* is true of Alpamayo's labels in the abstract and **false in practice
for our corpus**, because the overlap happens to be exactly the video-bearing set. The abstract
property was never the operative one.

## 6. THE ORDER OF WORK THIS IMPLIES

1. ✅ **Measurement #1 — DONE (§5.1): 201, set-identical to aug120.** The two ceilings are one.
2. **Adjudicate the 42-clip review sheet.** It is rendered and waiting; it is the only thing that can
   turn Tier 1 from *representable* into *validated*, and it gates Tier 2's admissibility bar.
3. **Rebuild the lateral labels under v6.1** — CPU, closes the 186-clip hole, no retrain
   (`STAGE_MAY_INTRODUCE` legitimises the widened head at the S-T boundary).
4. **Prototype Tier 2 on those same 42 clips**, both with and without ego numbers in the prompt, and
   report agreement per stratum + the echo delta.
5. ⭐ **w120 extraction post-30k — now the BINDING LEVER on the whole plan**, not merely Tier 2's
   ceiling: it is what turns 8.38 % corpus coverage into something larger. Measure the
   per-episode cost on ~10 episodes first.
6. **Tier 3 rules** — implement only after (2), so the "clear situations" are chosen against
   adjudicated evidence rather than intuition.
