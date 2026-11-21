# ⛔ RETRACTION — "the extractor takes the obstacle's side" (same day)

**Retracted by** DataFlyWheel, 2026-08-30, hours after publishing it to the PI in the
D-LAT-AGREE inspection report. **Two errors, one of them a classic.**

## The claim

*"54 of the 216 hard contradictions are an extractor bug: `alpamayo_side` picked up where an
OBSTACLE was rather than which way the car went, e.g. 'Nudge left to pass the parked car on the
right' extracted as `right`. No judgment needed, I will fix the side field to read the manoeuvre
verb."*

## Error 1 — WRONG MECHANISM. It is not our extractor at all.

`alpamayo_side` does not come from the CoT prose. `AlpamayoClip.lateral`
(`alpamayo_records.py:90`) reads Alpamayo's own **structured** `meta_action["lateral"]` field and
normalises it. MEASURED on the four cases I published:

| clip | `meta_action.lateral` | its own CoT says |
|---|---|---|
| `002646e7` | **Steer Right** | "Nudge **left** to pass the parked car on the right" |
| `00c15800` | **Steer Left** | "Nudge **right** to increase clearance to the parked car on the left" |
| `0618670a` | **Sharp Steer Left** | "**Turn right** into the parking lot" |
| `00f3bda1` | **Go Straight** | "**Lane change to the left**" |

⇒ **our parse is FAITHFUL; ALPAMAYO CONTRADICTS ITSELF.** (Its `lane` axis reads "Lane Keep" on
all four, including the explicit lane change.) There was no bug of ours to fix.

## Error 2 — SELECTION BIAS. I proved the text reliable using cases chosen for matching.

I filtered the contradictions for *"the CoT verb equals the geometry"*, found 54, and concluded the
CoT was the trustworthy source. **That is circular: the filter IS the conclusion.**

MEASURED without the filter, on all 994 clips where Alpamayo supplies BOTH a structured side and a
manoeuvre verb:

| source | agrees with geometry |
|---|---|
| `meta_action.lateral` (structured) | **593 / 994 = 59.7 %** |
| CoT manoeuvre verb (prose) | **178 / 994 = 17.9 %** |
| meta right & CoT wrong | **479** |
| CoT right & meta wrong | **64** |

⇒ **the fix I proposed — switch the fusion source to the CoT verb — would have made the labels
dramatically WORSE**, replacing a 59.7 % signal with a 17.9 % one across the corpus.

## ROOT-CAUSE CLASS — C84: A SUBSET SELECTED ON THE PROPERTY BEING TESTED

Recognition signal: **the filter that produced the sample names the same quantity the conclusion is
about.** Here the sample was "cases where prose agrees with geometry" and the claim was "prose
agrees with geometry". The number can only come out one way.
⇒ **STANDING RULE: a claim about which of two sources is more reliable MUST be measured on every
record where BOTH make a claim — never on a subset defined by their agreement.** The unbiased
denominator is the whole point; a hand-picked panel is an illustration, never evidence of a rate.

⚠️ **This is the same family as the CLAUDE.md ridge-probe failures** (tuning on the data being
scored) in a qualitative costume, and it survived because the hand-picked panels were *genuinely
convincing to look at*. Four of them are in the published report and each one is real; what was
false was the rate I generalised from them.

## What actually stands

* The 54 clips remain cases where the CoT and geometry agree against `meta_action`. That is a real
  observation about those clips, and about Alpamayo's internal inconsistency.
* It is **not** grounds to prefer the prose in general — the opposite is true.
* **Open question for the PI (raised, not decided):** should a clip whose two Alpamayo outputs
  contradict each other carry the corroboration flag at all? A source that disagrees with itself is
  arguably not a second opinion. That is a policy call, not a measurement.

## Where the correction landed

* The published report was **re-issued** with the section rewritten and the correction stated in
  place — the PI was forming judgment from it, so it could not wait.
* `RESULT.md` §1's "extractor" framing is superseded by this file.
