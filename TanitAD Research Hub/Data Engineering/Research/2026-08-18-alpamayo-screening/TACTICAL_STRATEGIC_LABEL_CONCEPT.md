# CONCEPT — tactical & strategic labels from the Alpamayo-augmented corpus
### For approval. Binding for **v6F**, **REF-A v1** and **REF-C v3**.

**Date:** 2026-08-18 · **Status:** ✅ **APPROVED by the PI, 2026-08-18** — items 1–4 and 6 as
written; item 5 **amended**: Qwen3.5-9B **in thinking mode** (no Qwen3-VL A/B). Strategic
labelling is **combined into the same pass** (§8). Nothing is implemented yet.

**PI direction:** *use the Alpamayo-augmented data for post-training the tactical and strategic
layers; it also increases data diversity. Priority 1 map Alpamayo's labels to ours; priority 2
leverage a VLM (Qwen 3.5 9B) for semantic information; priority 3 use ego data to extract
constraints or quantified goals (e.g. detect a stop, emit "stop at xx").*

---

## 0. ⭐ THE REFRAME, AND WHY IT IS CORRECT

I had treated the 4,528 Alpamayo clips outside our parity corpus as unusable. **That was wrong, and
the ladder already says why.** From `train_v6_staged.py`, verbatim:

> `STAGE_INVALIDATES["S-T"] = ()` — *"trains layer_tac + planner on a **FROZEN S-W trunk**; the
> trunk's inputs (pixels) are unmoved, so S-W's certificate still applies verbatim"*

⇒ **Parity is a property of the WORLD stage, not of the tactical one.** S-W must stay on the
canonical 2,376 episodes (`physicalai-train-e438721ae894`, skip-hash `f09e44db`) because that is
what makes the world-model comparison attributable. S-T and S-S train *on top of a frozen trunk* —
they need **labels and diversity**, and a larger corpus there **invalidates nothing**.

**This is not a workaround. It is the staged ladder being used as designed**, and it turns the
Alpamayo corpus from a 201-clip footnote into the primary tactical/strategic training set.

### 0.1 ⛔ ONE THING MUST BE EXCLUDED FIRST — a real leak

`MEASURED` this session: **|Alpamayo ∩ val40| = 6 clips.**

```
01acc9de-8bb9-465e-a664-3098ecd92276   026ef99a-2283-4c08-83e3-c0c3b4bae865
030011f7-dfb9-40cf-817f-cb7e8a87b0db   07f7b41f-b007-48d1-905e-b3385e8e897f
09759d8c-b66c-4672-bad2-c05c54ea9ded   (+1)
```

Training on the full 4,729 would put **6 of our 40 evaluation clips into training**. ⇒ The training
population is **4,723**, and the exclusion list is a committed artifact, not a runtime filter — the
same discipline as `dump_exclusions.json`. Every number below uses 4,723.

### 0.2 The one hard dependency, stated before the concept rests on it

The frozen trunk consumes **pixels**. The 4,723 clips have Alpamayo *text* but **no w120 video on
our side**. ⇒ **The w120 extraction job is not a nice-to-have; it is the precondition for this
entire concept.** It is CPU-only, already scoped, deferred to Thor post-30k. Its **per-episode cost
is UNMEASURED** — measure on ~10 episodes before committing 4,723.

---

## 1. ⭐ THE ORGANISING PRINCIPLE — three tiers, three DIFFERENT KINDS of output

The reason the previous draft needed a precedence rule and a conflict policy is that it had three
tiers competing to fill **the same field**. They should not. Each tier produces a **different kind of
thing**, and together they compose one label:

| tier | produces | type | example |
|---|---|---|---|
| **1 · Alpamayo** | the **action CLASS**, lateral | categorical, our vocabulary | `LANE_CHANGE_L` |
| **2 · VLM** | the **REASON** (longitudinal action) **+ the goal TOKEN + its REFERENT** | categorical + a named object | `BRAKE_TO`, referent = *"stop line at the junction"* → `STOP_POINT` |
| **3 · Ego** | the **goal ARGUMENTS** | physical units (m, s, m/s) | `within_m = 23.4`, `by_time_s = 3.1` |

⇒ **No two tiers write the same field. There is no voting, no echo between tiers, and precedence is
needed only *within* a field.** This is what makes the design honest rather than a fusion heuristic.

### 1.1 ⭐⭐ AND THE COMPOSITION RULE THAT MAKES TIER 3 SAFE

> **Ego QUANTIFIES what the VLM has NAMED. Neither is a label alone.**

*"The VLM says there is a stop line ahead; ego says the ego reached v ≈ 0 at 23.4 m."* → `STOP_POINT
(within_m = 23.4)`.

This directly answers the standing trap. `vtarget_guarded` was rejected because it is *"hindsight
EGO geometry — what speed did this driver settle at, not what speed is permitted here"*, and on ego
inputs **nothing beats repeating v₀'s band (0.4066 free vs 0.2465 trained)**. An ego quantity gated
on a **named external referent** is no longer the driver's idiosyncrasy — it is a measurement of a
scene feature the VLM independently asserted exists. ⇒ **Tier 3 may never fire where Tier 2
abstained.** That single rule is what keeps ego out of echo territory.

---

## 2. PRIORITY 1 — THE ALPAMAYO → OURS MAPPING (validated proposal)

`MEASURED` over 4,729 clips; counts are corpus-wide (the 6 leak clips are excluded at training time,
not from this taxonomy).

### 2.1 LATERAL — Alpamayo LANE axis → `a_tac_lat`. **ADOPT.**

| Alpamayo LANE | → `a_tac_lat` (v6.1) | n | confidence |
|---|---|---:|---|
| Lane Keep | `LANE_KEEP` | 4,035 | direct |
| Left Lane Change | `LANE_CHANGE_L` | 22 | direct |
| Right Lane Change | `LANE_CHANGE_R` | 82 | direct |
| Slightly Shift Left | `NUDGE_L` | 69 | direct |
| Slightly Shift Right | `NUDGE_R` | 31 | direct |
| **Turn Left** | **`TURN_L`** | 85 | ⭐ v6.1 |
| **Turn Right** | **`TURN_R`** | 101 | ⭐ v6.1 |
| *(unparsed — the 304 `Stop` rows)* | `NOT_APPLICABLE` | 304 | ⛔ never imputed |
| *(no Alpamayo source)* | `ABORT_LC` | 0 | Tier 2 only |

**Coverage: 4,425 / 4,425 parseable rows = 100 %** with v6.1. This is the tier's whole job and it
does it completely.

### 2.2 LONGITUDINAL — Alpamayo LONGITUDINAL axis → **NOT a label. A PRIOR.**

The type mismatch is structural: theirs is magnitude (*Gentle Deceleration*), ours is reason
(`FOLLOW` / `BRAKE_TO` / `YIELD_MERGE`). **Proposal: carry it as a typed prior into Tier 2, never as
an `a_tac_lon` value**, and use it as a **consistency check** on the VLM's answer:

| Alpamayo magnitude | admissible `a_tac_lon` after Tier 2 | inadmissible |
|---|---|---|
| Strong / Gentle Deceleration | `BRAKE_TO`, `FOLLOW`, `YIELD_MERGE`, `CREEP` | `CRUISE` |
| Maintain Speed | `CRUISE`, `FOLLOW` | `BRAKE_TO`, `HOLD` |
| Gentle / Strong Acceleration | `CRUISE`, `FOLLOW` | `BRAKE_TO`, `HOLD`, `CREEP` |
| Stop | `HOLD`, `BRAKE_TO` | `CRUISE` |

⇒ A VLM answer outside the admissible set is a **flagged disagreement**, routed to review rather
than silently accepted or silently dropped. This is how Alpamayo is used *maximally* on an axis it
cannot label.

### 2.3 The LATERAL (steering-magnitude) axis → **do not map.** Use as validation

κ in the `(a, κ)` action space already carries steering magnitude continuously. Free check: does the
planned/labelled κ agree in **sign and band** with *Go Straight / Steer / Sharp Steer*? Disagreement
is a label-quality signal, not a label.

### 2.4 STRATEGIC — what Alpamayo can and cannot give

Alpamayo emits no route axis. **Proposal: derive the strategic route token from the LANE axis plus
ego geometry** — `TURN_L`/`TURN_R` over a 5–20 s window is a route-level fact — and treat the 186
turn clips as the **seed set for strategic supervision on the augmented corpus**. ⚠️ This is the
weakest part of the concept and I flag it as such: 186 clips is thin for a route head, and §5 names
the measurement that would size it properly.

---

## 3. PRIORITY 2 — THE VLM (proposal)

### 3.1 Model — the PI's choice, verified

`MEASURED` from the HF API this session:

| repo | params | architecture | vision | gated |
|---|---:|---|---|---|
| ⭐ **`Qwen/Qwen3.5-9B`** | **9,653,104,368** | `Qwen3_5ForConditionalGeneration` | ✅ `vision_config` | **no** |
| `Qwen/Qwen3-VL-8B-Instruct` | 8,767,123,696 | `Qwen3VLForConditionalGeneration` | ✅ | no |
| `Qwen/Qwen3-VL-8B-Thinking` | 8,767,123,696 | `Qwen3VLForConditionalGeneration` | ✅ | no |

**Qwen3.5-9B is multimodal and ungated — the PI's choice is available and correct.** I would run the
**Thinking** variant of Qwen3-VL-8B as the A/B comparator on the 42-clip adjudicated subset, because
reason-assignment is exactly a reasoning task; the winner takes the corpus. That is one cheap
comparison, not a second programme.

### 3.2 What it is asked for — three fields, never the action magnitude

1. `a_tac_lon` ∈ `TACTICAL_LON_ACTIONS` ∪ `{ABSTAIN}`
2. **the referent** — a named object or road feature (*"the lead vehicle"*, *"the stop line"*,
   *"the merging car on the right"*), free text, **required** whenever the answer is not `ABSTAIN`
3. `g_tac` token ∈ `TACTICAL_GOAL_TOKENS` ∪ `{ABSTAIN}` — the goal the action serves

### 3.3 Input — rich, past **and** future

`CLAUDE.md` binds *labels may use ego; inference is vision-only*, so the future is admissible for a
labeller and should be used: a model seeing only the past must **guess** whether a deceleration
resolves into a stop, a follow, or a yield.

| # | input | why |
|---|---|---|
| 1 | frames −1.0 s → +6.0 s, decision band 2–6 s marked | the geometry the review sheet already renders |
| 2 | the **lead track** (gap, time-gap, closing rate) from our obstacle join | turns *"is there a lead"* from a perception question into a **given** |
| 3 | ego speed profile over the band | disambiguates magnitude |
| 4 | Alpamayo magnitude + `cot`, **labelled as a prior** | a teacher's opinion, never presented as the answer |

### 3.4 Prompt rules — the four that decide whether this works

1. **Forced choice over OUR tokens, with `ABSTAIN` as easy to pick as any label.** The old substring
   path was a lottery precisely because abstention was not on offer.
2. **Referent before verdict.** A reason whose referent cannot be named is not a reason. This makes
   the failure visible instead of silent.
3. **One axis per call.** Asking both axes in one response reintroduces, in the prompt, the very
   mixing the factored vocabulary exists to remove.
4. ⛔ **Two passes, with and without input #3 (ego numbers), on the adjudicated subset.** Agreement
   between a VLM that was shown ego and an ego-derived label is an **ECHO**, not corroboration — the
   review sheet already flags this as its `B_TWO_OF_THREE` caveat. The delta between the two passes
   **measures** the echo instead of assuming it away.

### 3.5 The admissibility bar — before one label is used

Run on the **42 clips already selected and rendered** in the 2026-08-16 visual review (strata chosen
to include the hardest cases), and report agreement per stratum against human verdicts. ⛔ **That
review is 0/42 adjudicated, so the human pass is ON the critical path**, not parallel to it. A VLM
leg scored against nothing is a second unvalidated teacher, not a validation of the first.

---

## 4. PRIORITY 3 — EGO → QUANTIFIED GOALS (the PI's *"stop at xx"*)

⭐ **This is the strongest part of the concept, because v6 already has the machinery.**
`TACTICAL_GOAL_TOKENS` carry **typed argument slots in physical units** —
`CONSTRAINT_SLOTS = (within_m, by_time_s, at_arc_m, hold_for_s)` and
`GOAL_ARG_NAMES = (arg0..arg3, *CONSTRAINT_SLOTS)`, *"PHYSICAL UNITS (m, s, m/s) — no free text at
inference"*. Ego measurement fills exactly those slots.

| VLM named (Tier 2) | ego measures (Tier 3) | emitted goal |
|---|---|---|
| a stop line / a red light | arc length to where v < 0.5 m/s, and the time to it | `STOP_POINT(within_m=…, by_time_s=…)` |
| a lead vehicle | gap and time-gap at t₀, closing rate (from the join) | `GAP_TARGET(arg0=gap_m, arg1=time_gap_s)` |
| open road, no referent | the sustained speed band over the window | `SPEED_BAND(arg0=v_lo, arg1=v_hi)` |
| a merging vehicle | time to the conflict point | `YIELD_AT(by_time_s=…)` |

**Three hard rules:**
1. ⛔ **Tier 3 never fires where Tier 2 abstained** (§1.1). No referent ⇒ no quantified goal.
2. ⛔ **Ego never assigns an action CLASS.** It cannot separate `FOLLOW` from `BRAKE_TO`; only the
   referent does.
3. ⛔ **Every candidate rule must pass the echo test before adoption:** *would a model that only
   echoes v₀ satisfy it?* If yes, the rule is void. `SPEED_BAND` is the one most at risk and should
   be admitted last, if at all.

---

## 5. SCALE, AND WHAT IT COSTS

| population | n | note |
|---|---:|---|
| Alpamayo clips | 4,729 | `MEASURED` |
| ⛔ minus val40 leak | **−6** | exclusion list, committed |
| **training-eligible** | **4,723** | |
| — already with our w120 video | 201 | = the aug120 set, `MEASURED` set-identical |
| — **needing w120 extraction** | **4,522** | ⭐ the precondition |
| lateral labels available immediately (Tier 1, v6.1) | ~4,419 of 4,723 parseable | text only — no video needed to *derive*, video needed to *train* |
| longitudinal labels available today | **201 max** | VLM needs frames |
| S-W parity corpus (untouched) | 2,376 | `physicalai-train-e438721ae894` |

**Diversity gain, which is the PI's stated motive:** the tactical/strategic training population goes
from **201 → 4,723 clips (23.5×)**, on a corpus whose lateral labels are already 100 % representable
under v6.1.

**Unmeasured, and blocking:** per-episode w120 extraction cost; VLM throughput per clip; the
**strategic** label yield from §2.4. ⇒ Measure extraction on ~10 episodes and VLM on the 42-clip
subset **before** committing the full job.

---

## 6. HOW THIS APPLIES TO ALL THREE MODELS

| model | what it consumes | why the concept fits unchanged |
|---|---|---|
| **v6F** | S-T/S-S post-training on the augmented corpus, frozen S-W trunk | `STAGE_INVALIDATES["S-T"] = ()`; the world certificate survives verbatim |
| **REF-A v1** | the same labels via its tactical/strategic brains | it imports `TACTICAL_LAT_ACTIONS` from `v6.py` — one vocabulary, no adaptation |
| **REF-C v3** | the same, through `tac.N_LAT`/`N_LON` | already factored; already reads the shared vocabulary |

**One vocabulary, one label set, three consumers.** That is only true because the vocabulary lives in
`v6.py` and is imported rather than copied — pinned by an identity test.

---

## 7. ⏳ WHAT I NEED APPROVED

1. **The reframe** — tactical/strategic post-training on the Alpamayo-augmented corpus, S-W parity
   untouched.
2. **The composition rule** — Alpamayo gives the class, the VLM gives the reason and referent, ego
   gives the arguments; **ego never fires where the VLM abstained**.
3. **The 6-clip val40 exclusion** as a committed artifact.
4. **v6.1 adoption** (`TURN_L`/`TURN_R`) at the S-T boundary — prepared, inert at default, awaiting
   your word.
5. **Qwen3.5-9B as the labeller**, with Qwen3-VL-8B-Thinking as a one-off A/B on the 42-clip subset.
6. **Sequencing:** adjudicate 42 → measure extraction cost on 10 → VLM A/B → full extraction → label
   → S-T/S-S post-training.

⚠️ **The honest weak points, so approval is informed:** the strategic label story (§2.4) rests on 186
turn clips and is the thinnest part; the Alpamayo/PhysicalAI training-overlap remains UNRESOLVED, so
Alpamayo is a teacher and not ground truth; and the whole concept is gated on an extraction cost
nobody has measured.

---

## 8. ✅ APPROVED, AMENDED, AND COMBINED WITH THE STRATEGIC PIPELINE

### 8.1 The decision

| # | item | verdict |
|---|---|---|
| 1 | tactical/strategic post-training on the Alpamayo-augmented corpus, S-W parity untouched | ✅ approved |
| 2 | composition rule — class / reason+referent / arguments; ego never fires where the VLM abstained | ✅ approved |
| 3 | the 6-clip val40 exclusion, committed | ✅ approved — artifact banked (`alpamayo_val40_exclusions.json`) |
| 4 | v6.1 (`TURN_L`/`TURN_R`) at the S-T boundary | ✅ approved |
| 5 | the labeller | ⚠️ **AMENDED — Qwen3.5-9B in THINKING MODE; the Qwen3-VL-8B A/B is dropped** |
| 6 | sequencing | ✅ approved, revised by §8.3 |

### 8.2 Thinking mode — verified, with one caveat that matters for our task

`MEASURED` from the model's own card and tokenizer config: **Qwen3.5 runs in thinking mode BY
DEFAULT** — *"Qwen3.5 models operate in thinking mode by default, generating thinking content
signified by `<think>…</think>` before producing the final response"*. The chat template carries
`enable_thinking`, `<think>`, `</think>`. ⇒ The PI was right, and **no A/B is needed**: the thinking
model *is* the model. Item 5's comparator is dropped, which removes a whole experiment.

⚠️ **But the card's recommended thinking-mode sampling is wrong for THIS task.** It advises
`temperature=1.0, presence_penalty=1.5` for general tasks. A **presence penalty pushes away from
tokens already produced** — across a corpus of forced-choice labels drawn from six recurring tokens,
that biases *against the frequent classes*, which is precisely the distribution we must not distort
(`LANE_KEEP` is 85 % of the lateral axis). ⇒ **Proposal:**

* **thinking** free-form at the card's *precise* profile — `temperature=0.6, top_p=0.95, top_k=20,
  presence_penalty=0.0`;
* **the verdict token constrained by logit masking to the allowed set** (`TACTICAL_LON_ACTIONS` ∪
  `{ABSTAIN}`), so sampling parameters cannot move the label at all — only the reasoning;
* context ≥ **128 K**, per the card's own advice for thinking mode.

### 8.3 ⭐ COMBINING WITH THE STRATEGIC PIPELINE — and it INVERTS the tier order

The PI is right that the strategic pipeline is already reviewed and optimised, and reading it
changes this concept. From `s2_derive.py` and `S2_STRATEGIC_GAP.md`, all `MEASURED` on aug120:

* the VLM's TURN **precision is 100 % (19/19)** but its **recall is 17/33 left and 2/29 right** —
  badly side-biased;
* **28 of its 31 `ROUTE_TO` claims sit on plain turn geometry**; G1 is **CLOSED at 0/31**, so
  `ROUTE_TO` is unsupervisable and is remapped to geometry or abstained, never guessed;
* the PI adjudicated 18 of 19 `LANE_TARGET` labels and called **14 wrong**;
* Engine A (`route_from_future_v3` over the hindsight ego path) **covers every clip**;
* the S2 verdict, verbatim: ***"geometry decides, the VLM corroborates"***.

⇒ ⭐⭐ **The tier order is NOT global. It follows the TYPE of the quantity being labelled:**

| layer | the label IS… | primary | corroborator |
|---|---|---|---|
| **TACTICAL lateral** | a lane-topological fact | **Alpamayo** | ego geometry |
| **TACTICAL longitudinal** | a **semantic reason** (why) | **VLM** | Alpamayo magnitude as an admissible-set prior |
| **TACTICAL goal args** | a **measurement** | **ego** | — (gated on a VLM referent) |
| **STRATEGIC `g_str`/`a_str`** | **future-path geometry** | ⭐ **ego geometry (Engine A)** | **VLM**, demoted |

**Geometry is primary where the label IS geometry; the VLM is primary where the label is semantics.**
That is one principle, it is measured rather than aesthetic, and it explains both layers without a
special case.

### 8.4 ⭐ The consequence that changes the schedule: STRATEGIC NEEDS NO VIDEO

Engine A derives `g_str`/`a_str` from the **hindsight ego path** — i.e. from **egomotion**, not
pixels. ⇒ **Strategic labels for all 4,723 clips are derivable without the w120 extraction.** The
extraction gates the *tactical* reason (VLM needs frames); it does **not** gate the strategic layer.

Two consequences:
1. **The strategic label build can start immediately**, on CPU, in parallel with everything else —
   it is not behind the 4,522-clip video job.
2. **The two passes share one clip list and one provenance schema** but consume different inputs
   (egomotion vs frames), so "combined" means *one pipeline, two input paths*, not one model doing
   both. `s2_derive.py` is already the single home of the S2 mapping with three consumers and zero
   drift — the Alpamayo corpus becomes its fourth.

⛔ **The one measurement this rests on:** egomotion availability for the 4,723. The label chunks are
**not local** (probed this session), so it is a bounded pull from the dataset's label chunks — far
smaller than camera chunks. **Run this before anything else**: it decides whether the strategic half
is CPU-immediate or itself gated on a download.

### 8.5 Revised sequencing

| # | step | gate | needs |
|---|---|---|---|
| 1 | egomotion availability for the 4,723 | — | bounded pull |
| 2 | **strategic labels via `s2_derive.py` (Engine A primary)** | step 1 | CPU only |
| 3 | adjudicate the 42-clip visual review | — | human, PI |
| 4 | w120 extraction cost on ~10 episodes | — | CPU |
| 5 | VLM tactical-reason prototype on the 42 clips, two passes (± ego numbers) | step 3 | 1 GPU |
| 6 | full w120 extraction (4,522) | step 4 | CPU, Thor post-30k |
| 7 | tactical labels at corpus scale | steps 5, 6 | GPU |
| 8 | S-T / S-S post-training, all three models | steps 2, 7 | Thor post-30k |

⇒ Steps 1–3 can start **now** and none of them touches the live 30k run.
