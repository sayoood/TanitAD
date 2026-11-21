# PRE-REGISTRATION — PH0: VLM strategic-labeling pilot (three arms, 50 clips)

**Written 2026-08-11 (0-GPU design work; the run itself stays SEQUENCED behind
v5.8f core validation — registry §1.14 release row + T1 rows + P8/P9 — per the
PI's directive in VLM_STRATEGIC_LABELING.md §7. This document binds the pilot's
gates and both outcomes BEFORE any arm runs.)**

## What runs

Three arms, identical 50-clip pilot set, identical prompts/schema, on pod4:

| arm | model | precision/quant | why it is in |
|---|---|---|---|
| A | `Qwen/Qwen3.5-9B` | bf16 | the workhorse ("qwen 3.5 at least" — PI) |
| B | `Qwen/Qwen3.5-27B-FP8` | FP8 | the bigger-quality arm on the same lineage |
| C | `google/gemma-4-31B-it-qat-w4a16-ct` | QAT w4a16 (~17 GB) | survey benchmark leader (MMMU-Pro 76.9) with NO published OCRBench — exactly the unknown PH0's sign-OCR gate measures (PI: "Add it as third PH0 arm") |

**Pilot set (fixed before any run):** 50 clips sampled from the augmentation
set stratified by road class (the card's coverage table), seed 0, clip_ids
listed in `ph0_clips.json` committed WITH the sampler line that produced them.
Same clips for all arms — paired comparison, never pooled.

**Per clip, per arm:** Engine A geometric summary (deterministic, shared) +
two-pass VLM protocol (extract → self-verify CONFIRM/RETRACT) + fusion gate,
producing the v0 schema row (VLM_STRATEGIC_LABELING.md §4).

## Measured quantities (all three arms, same instruments)

1. **Sign-OCR precision** on the human-check sample: every `signs[]` row with
   `text_ocr` non-empty from a 100-field stratified sample, graded
   correct/incorrect by the PI (or a frame-side-by-side sheet the PI eyeballs).
2. **Schema compliance rate**: fraction of clips yielding a
   schema-valid JSON row with all required fields on pass 1 (before any retry).
3. **Strategic-action geometric-consistency rate**: fraction of emitted
   strategic actions that pass Engine A's fusion gate (baseline row, NO
   threshold — PH0 measures it, PH1 decides with it).
4. **Abstention honesty**: fraction of clips with no legible nav signage where
   the arm correctly abstains (`goal.kind != route_to`) — hallucinated cities
   are the failure this measures.
5. **Wall-clock s/clip** (median + p90) and peak VRAM — the PH1 budget input.
6. **Video-template check** (mandatory, per arm, BEFORE the 50 clips): one
   smoke clip through the arm's official chat template with video input;
   an arm that cannot ingest video at all FAILS PH0 outright (survey claims
   are PUBLISHED, not MEASURED, until this passes).

## Pre-registered gates and BOTH outcomes

- **G1 (sign-OCR):** precision ≥ 0.9 on the checked sample.
  - PASS → the arm is OCR-qualified for PH1's sign/nav fields.
  - FAIL → the arm's `signs[].text_ocr` and every `route_to` goal derived from
    signage are EXCLUDED from PH1 for that arm (lane-level/corridor goals may
    still qualify via G3); a FAIL on all three arms ⇒ PH1 drops signage-text
    fields entirely and the strategic goal degrades to hindsight-geometry
    corridor intent — the honest ceiling stands, no re-prompting rescue runs
    without a new prereg.
- **G2 (schema):** compliance ≥ 0.9 on pass 1.
  - FAIL for an arm ⇒ that arm needs constrained decoding (grammar/JSON mode)
    in PH1 — a cost line, not a disqualifier.
- **G3 (consistency, baseline only):** the rate is REPORTED with its CI; no
  pass/fail. It becomes PH1's monitored quantity.
- **Model selection rule (bound now):** the PH1 model = the arm that passes G1
  ∧ G2 with the best sign-OCR precision; wall-clock breaks ties at <2 pp OCR
  difference; the PI signs off on the spend using the measured s/clip × 4,7xx,
  not the design guess.

## Discipline

- n=50 pilot: report exact counts (x/n) with Wilson intervals, never bare
  percentages.
- Prompts + schema + sampler are FROZEN at run start and hash-stamped into
  every row's `_provenance`; any prompt edit during the pilot restarts the
  pilot (iteration is PH0's PURPOSE, but each iteration is a fresh 50-clip
  pass, not a cherry-picked re-run of failures).
- Labels may use future/ego (offline hindsight is the design); nothing from
  this pipeline is ever an inference-time input to a driving model — targets
  and eval strata only.
- The goal/situation disjointness rule travels into the schema: `strategic.*`
  fields carry their derivation source; any field derivable only from the
  scenario classification is inadmissible as g_str supervision.

- [ ] ph0_clips.json sampled+committed · [ ] video-template smoke ×3 ·
  [ ] 50-clip run ×3 · [ ] PI check sheet · [ ] gates graded · [ ] PH1 decision
