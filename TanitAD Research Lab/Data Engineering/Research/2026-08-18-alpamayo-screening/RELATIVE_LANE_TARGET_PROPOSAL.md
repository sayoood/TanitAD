# Recovering the topology-blocked strategic tokens with a RELATIVE lane target

**2026-08-18 · PI proposal:** *the critical strategic tokens can be extracted by a VLM with the right
prompt and conditioned past + future video; keep the target lane simple in a first step —
**current lane / to the left / to the right**.*

**My assessment: this recovers 4 of the 5 blocked tokens, and it is a better idea than a workaround
— it removes the exact failure mode that killed the geometric gate.** One token (`ROUTE_TO`) does
not follow and should stay gated. Below: why it works, what it recovers, what it cannot, and the
pre-registered test it must pass — because the last VLM attempt at strategic labels failed, and we
know how.

---

## 1. WHY THE RELATIVE ENCODING IS THE RIGHT MOVE (three reasons, not one)

**(a) It changes the question from absolute topology to relative vision.**
*"Which lane index, of how many?"* needs a lane graph — the corpus has none (*"we do not include
open maps data"*). *"Current, left, or right?"* needs only the ability to see **lane markings and
the ego's position between them**. That is a visual question, and it is exactly what a VLM is for.

**(b) Past + future video converts INTENT into OUTCOME.**
`LANE_TARGET` failed as an *intent* label because intent is unobservable. With the future in frame,
the labeller is no longer guessing: *if the ego is one lane to the left at t+6 s, the target was
left.* `CLAUDE.md` permits this — *labels may use ego; inference is vision-only* — and it is the same
device that makes the tactical **reason** answerable.

**(c) ⭐ AND IT FIXES THE PRECISE DEFECT THAT KILLED THE GEOMETRIC GATE.**
The retired gate died because **15 of 19 firings had a lateral offset FULLY EXPLAINED by
constant-curvature road following** — ego geometry cannot distinguish *"changed lane"* from *"the
road curved"*. **A VLM watching the lane markings can**: crossing a marking is visible; following a
curve is visible; they look nothing alike. ⇒ This is not the VLM substituting for geometry on
geometry's own turf — it is the VLM supplying **the one observable geometry structurally lacks**.

That third point is why I think the proposal is right rather than merely convenient.

---

## 2. WHAT IT RECOVERS — the cascade

Define one primitive, observed from video:

> **`lane_target_rel` ∈ {LEFT (+1), CURRENT (0), RIGHT (−1)}** — the lane the ego occupies at the
> end of the strategic window, relative to the lane it occupies at t₀, measured against lane
> markings.

Sign convention is **already declared** in `s2_derive.py`: *"+1 = left, −1 = right (matches the ego
frame's +y = left)"*. ⭐ **No vocabulary change is needed** — `LANE_TARGET` already exists as a
token, and the ±1 direction arg is an existing, declared convention. It needs a *filler*, not a
schema.

| token | recovered? | how it follows |
|---|---|---|
| ⭐ **`LANE_TARGET`** | ✅ **directly** | `LANE_TARGET(arg0 = lane_target_rel)`, emitted only when `≠ CURRENT` |
| ⭐ **`PREPARE_LANE_CHANGE`** | ✅ **deterministically** | its own rule already says *"only when the ROUTE requires a lane the ego is not in"* — with a relative target that condition is simply `lane_target_rel ≠ CURRENT`. It stops needing a route. |
| ⭐ **`HOLD_CORRIDOR`** | ✅ **as the complement** | `lane_target_rel == CURRENT` **and** no pending exit ⇒ hold. This also gives the token its first real definition: not *"stay in the corridor"* (undefined without a map) but *"the target lane is the current lane"* |
| ⚠️ **`PREPARE_EXIT`** | ⚠️ **with one more predicate** | needs *"an exit exists ahead"* (visual: sign, off-ramp geometry) **and** `lane_target_rel` pointing at it. Rarer, harder, and the class most likely to be under-powered |
| ⛔ **`ROUTE_TO`** | ⛔ **NO — and it should stay gated** | `ROUTE_TO` names a **navigation destination**. No destination exists anywhere in the corpus; a VLM asked for one would invent it. G1 closed at 0/31 stands, and this proposal does not touch it. |

**⇒ 4 of 5 recovered, from one primitive.** That is the leverage: three tokens fall out of a single
well-posed visual question, and the fourth needs one additional predicate.

---

## 3. ⛔ THE BAR — because the last VLM strategic attempt failed, and we know how

`MEASURED` on aug120, the pipeline that was demoted: TURN precision **100 % (19/19)**, recall
**17/33 left and 2/29 right**; **28 of 31 `ROUTE_TO` on plain turn geometry**; the PI adjudicated 18
of 19 `LANE_TARGET` and called **14 wrong**.

⚠️ **That negative does NOT automatically transfer**, and saying so is not special pleading — the
configuration differs in three ways that each bear on the failure:

| | the demoted pipeline | this proposal |
|---|---|---|
| future frames | not supplied | ⭐ **supplied** — turns intent into outcome |
| the question | one absolute `goal_kind` field | a **relative** 3-way forced choice |
| justification | none required | ⭐ **a named visual referent required** (which marking, which side) |

⇒ The prior result is evidence about *that* configuration. But it sets the bar, and it names the
failure mode to watch: **side bias** (2/29 right is not noise) and **over-claiming**.

### 3.1 Pre-registered test — before one label is used

* **Substrate:** the 42 rendered clips of the 2026-08-16 review, plus **all aug120 clips carrying a
  lane event**, adjudicated by the PI.
* ⭐ **Report per-side, never pooled.** A pooled accuracy would have hidden 2/29 behind 17/33. The
  gate is **per-class recall ≥ 0.6 on BOTH left and right**, and a left/right asymmetry ratio > 2.0
  is a **FAIL regardless of the mean**.
* **Positive control:** the unambiguous lane changes already adjudicated — if the VLM misses those,
  nothing else matters.
* **Negative control (the one that killed the geometric gate):** clips with **large lateral offset
  fully explained by road curvature**. The VLM must return `CURRENT` on them. This is the
  discriminating test, and it is the reason to prefer a VLM here at all — so it must be *measured*,
  not assumed.
* **Abstention is a first-class outcome**, and its rate is reported. Occlusion, no visible markings,
  night, or a construction zone are legitimate `ABSTAIN`s — a labeller that never abstains on those
  is guessing.

## 4. PROMPT SHAPE

One call, one question, evidence first:

1. *"At t₀, between which lane markings is the ego? Name them (e.g. solid right edge, dashed left)."*
2. *"At t+6 s, between which markings is the ego?"*
3. *"Did the ego CROSS a lane marking, or did the road curve? Say which you observed."* ← the
   discriminator against the retired gate's failure
4. **Verdict:** `LEFT` / `CURRENT` / `RIGHT` / `ABSTAIN`, constrained by logit masking.
5. *(only if not CURRENT)* *"Is there an exit or off-ramp ahead on that side?"* → feeds
   `PREPARE_EXIT`.

Qwen3.5-9B in thinking mode (approved): steps 1–3 land in the `<think>` block, the verdict is
constrained. Two passes ± ego numbers, per the standing echo rule.

## 5. WHAT I WOULD STILL NOT CLAIM

* `ROUTE_TO` stays gated. §2.
* `PREPARE_EXIT` is the weakest of the four and may fail its own power check — report it separately
  rather than folding it into a lane-target success.
* This makes the tokens **supervisable**, not **correct**. Correctness is what §3's adjudication
  measures, and it has not run.
* The **absolute** lane index remains unavailable, so anything needing *"lane 2 of 3"* is untouched.
  The relative encoding is a deliberate first step, exactly as the PI framed it — and it should be
  recorded as such so a later reader does not mistake it for full lane topology.

---

## 6. ⭐ PI ADDENDUM — `TURN_LEFT`/`TURN_RIGHT` from Alpamayo's turn labels

**PI:** *turn left and turn right can be extracted from the future turn labels of Alpamayo.*

**Yes — and the value is not where it first appears.** `TURN_LEFT`/`TURN_RIGHT` were already in the
✅ geometry-derivable column (§1 of the audit), so this is not a *new* token. What it adds is an
**independent second leg**, and that is worth more here than a new source would be.

### 6.1 ⛔ First, the constraint I verified before agreeing

The per-clip Alpamayo record carries exactly five fields:

```
clip_id · cot · lane · lateral · longitudinal          ← NO time fields
```

⇒ **The turn label is a CLIP-LEVEL fact with no timestamp.** It says *"this clip contains a turn,
and which way"*. It cannot say **when**, so it cannot by itself be placed at a strategic horizon
(g_str is 8–30 s) or disambiguate two turns in one clip.

### 6.2 ⇒ The composition, which is the same principle the concept already uses

| leg | supplies | for `TURN_LEFT`/`TURN_RIGHT` |
|---|---|---|
| **Alpamayo** | the **TYPE and DIRECTION** | *"a turn, to the left"* |
| **ego geometry** | the **TIMING and the ARGS** | *when* the heading change occurs → `at_arc_m`, `by_time_s` |

Neither is the label alone — exactly as *"ego quantifies what the VLM has named"*, with Alpamayo in
the naming role. This keeps the concept's one principle intact instead of adding a special case.

### 6.3 ⭐ The real prize: an independent RECALL check on 186 clips

The demoted VLM's strategic failure was **recall, and asymmetrically so — 17/33 left, 2/29 right**.
Alpamayo's 85 left + 101 right turns are an **independent** assertion (it read the video; our engine
read the ego path), so agreement is genuine corroboration and **not an echo**. Two things fall out
for free:

1. **A measured recall check on `route_from_future_v3`**: of the 186 clips Alpamayo calls a turn, how
   many does the geometry engine also call a turn? Disagreement in either direction is an instrument
   finding, and it costs one join.
2. **A right/left symmetry check** on our own engine — the failure mode we already know to watch for.

⚠️ **Caveat that must travel with it:** Alpamayo is a **teacher, not ground truth** (PhysicalAI-AV is
listed as its training data, overlap UNRESOLVED, one draw at temperature 0.6). So a disagreement
localises a *discrepancy*, and adjudication decides which leg was wrong — it does not automatically
convict the geometry engine.
