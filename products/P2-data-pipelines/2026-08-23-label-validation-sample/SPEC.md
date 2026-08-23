# SPEC — label-extraction validation sample (E-LABELVAL-1)

`Owner: TanitAD_DataFlyWheel · Opened 2026-08-23 on PI request: "give me a visual
report on a small sample of the current label-extraction performance — key frame,
past and future, the labels our pipeline extracts, and your judgement. We need to
validate it to scale and finish the data set."`

⚠️ **Honest provenance: this SPEC was written alongside the work, not before it.**
The request was for a visual inspection, and the pre-registration discipline
(§3 work-package schema) is applied retrospectively here. Where a threshold was
chosen AFTER seeing data, it is named as such below.

## Question

Is the strategic/tactical label extraction good enough to scale to the full
corpus, and if not, what specifically must be fixed first?

## Method

1. Join extracted labels to clips that have BOTH camera frames and ego poses in
   a local episode cache. The join runs through `episode_id_legacy`, which the
   clip index warns collides — ambiguous ids are REFUSED on BOTH sides, never
   resolved by taking the first match.
2. For every joined clip, compute geometric evidence from ego poses ALONE
   (heading excursion, onset time, speed profile, stop detection).
3. Compare each label family against that evidence; render frames + bird's-eye
   trajectory so a human can adjudicate what geometry cannot settle.

## Success criteria (both outcomes, committed)

| outcome | reading |
|---|---|
| **A — scale now** | every token class ≥ 90 % agreement with independent geometry, no systematic one-directional defect |
| **B — fix first** | any class materially below that, or any defect that is systematic rather than sporadic ⇒ name the guard that removes it, implement it, and re-measure before scaling |

**Outcome B fired.** See RESULT.md.

## Admissibility rules carried

- The geometric check is a SECOND OPINION from ego poses. It cannot see lanes,
  signs or traffic ⇒ a disagreement is a flag for review, never proof of error.
- A STRATEGIC label is judged on the STRATEGIC horizon. (This was got wrong in
  the first pass and is the subject of a retraction below.)
- The sample is NON-PARITY and a convenience sample; no number here is
  cross-arm comparable and class frequencies say nothing about the corpus.

## Thresholds, and when each was chosen

| threshold | value | when |
|---|---|---|
| turn gate `TURN_DEG` | 25° | BEFORE — inherited from `refb_labels`' existing turn gate, not tuned here |
| acceleration gate | +2.0 m/s | AFTER seeing the inverted clip; chosen to separate it from the two correctly-labelled resume-cruise clips |
| "was moving" gate | 3.0 m/s | AFTER; excludes clips already stopped at the anchor |
