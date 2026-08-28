# RESULT — Independent references for deliberate-lane-offset labels + what the literature does with UNTIMED language labels

`Research Lab daily pass 2026-08-28, Data Engineering. Literature-only (0 GPU).
Seed: (a) D-DATA-GTAC-b — CORRIDOR_OFFSET is contested 2-vs-1 and uncalibratable
from ego geometry alone; (b) D-LABEL-GT caveat (1) — 87.2 % of v7 CoT tokens are
untimed; whether they may supervise time-banded heads is an OPEN PI decision.
All library keys verified banked (kb_add --verify clean, see raw/).`

## Findings first

**F1 — No published pipeline labels "deliberate lane offset" from an ego-trajectory
excursion threshold. Every one we found uses a reference EXTERNAL to the ego
trajectory, and the deliberate ones also require a CAUSE.** [PUBLISHED, banked]

| pipeline | reference for lateral-offset semantics | deliberateness criterion |
|---|---|---|
| Alpamayo-R1 CoC (lib `2511.00088`, Tab. 1–2, pp. 11–12) | **lane lines** (perception fact): "In-lane nudge" = offset with NO line crossing; "Out-of-lane nudge (straddle avoidance)" = brief intentional crossing + return | a decision-relevant trigger — "increase clearance around a blockage/hazard", causally supported by evidence in the history window; lane/laneline attributes (type, usable width) recorded as critical components |
| NAVSIM / nuPlan family (lib `2406.15349`) | HD-map lanes: drivable-area compliance and the v2 lane-keeping term are computed against map geometry, never against the ego track itself | n/a (metric, not label) — but the reference-independence principle is identical |
| CoVLA (lib `2408.10845`, §3.1.4) | rule-based captions from vehicle motion + detected objects (speed, accel, curvature, lead vehicle, traffic lights) — notably lane position is NOT captioned, because their sensor set has no lane reference | n/a — they scope the vocabulary to what their references support |

The two contested TanitAD derivers (raw `|peak_lat| >= 1.0 m` and curvature-relative)
are both functions of the SAME ego trajectory — they cannot calibrate each other even
in principle, which the 2-vs-1 conflict already demonstrates empirically (both fire at
~63–66 % base rate; register D-DATA-GTAC-b). The literature pattern says the missing
ingredient is not a better threshold but a second, independent signal class:
**lane boundary (perception) + causal trigger (agent/hazard)**.

**F2 — On a map-free corpus the published reference instrument is monocular lane
detection; current measured quality: CULane F1 81.43 (paper) / 81.11–81.55
(released models).** CLRerNet (lib `2305.08366`). [PUBLISHED] PhysicalAI-AV has no
map layer (settled, CLAUDE.md feature table), so the NavSim-style map reference is
unavailable — the camera-derived lane boundary is the only independent lateral
reference our corpus supports; the causal-trigger half is already in-house
(`obstacle.offline`, 3D agent tracks on 97.44 % of the corpus [MEASURED, registry]).

**F3 — Untimed language CAN supervise, but the literature never lets it touch a
time-localized head directly; three graded mechanisms, all measured:** [PUBLISHED, banked]

1. **MIL over candidate windows** — MIL-NCE (lib `1912.06430`): with ~50 % of
   HowTo100M clip–narration pairs misaligned, softmax-MIL over ~5 candidate
   clip–text pairs learns video features from scratch that beat self-supervised and
   many fully-supervised baselines. Untimed text supervises a POOL of windows, never
   one band.
2. **Alignability estimation before use** — TAN (lib `2204.02968`): on 10 h of
   annotated HowTo100M, **only 30 % of narration sentences are visually alignable at
   all, and only 15 % are naturally well-aligned**; TAN predicts per-sentence
   (alignable?, window) and trains the fine-grained head only on the top-α fraction
   (α swept 0.25–0.75). The default assumption for untimed text is that MOST of it
   cannot be temporally grounded.
3. **Video-level labels → temporal heads via MIL, at a known cost** — P-MIL (lib
   `2305.17861`, THUMOS14 Tab. 1): the weak-label SOTA reaches mAP@0.5 **40.0 vs
   58.6** for the fully-supervised SOTA (RefactorNet) — i.e. video-level supervision
   recovers ≈ 2018-era fully-supervised localization (TAL-Net 42.8) but keeps a
   ~19-point gap at strict IoU. Supervising a banded head from untimed labels is
   possible and measurably worse than timed supervision — never free.

**F4 — The "timed" alternative has a measured ceiling too:** BDD-X (lib
`1807.11546`), the classic time-aligned explanation dataset, reports inter-annotator
temporal IoU **0.63 (SD 0.21)** on doubly-annotated videos. [PUBLISHED] Even human
timing agrees only loosely — a calibration anchor for how much precision a "timed"
CoT band could ever claim.

**F5 — The strongest pattern is Alpamayo-R1's own fix: don't inherit untimed text —
re-anchor at labeling time.** CoC traces are anchored to "the first action taken by
the ego immediately after the critical reasoning moment", with causal locality (all
evidence inside the observed history window, explicitly to prevent referencing
unobservable future events) and labeling only on clips containing an explicit
decision. [PUBLISHED, lib `2511.00088` §4.1–4.2] Our v7 CoT tokens come from the
generic-CoT generation AR1 §2.4 criticizes, not from CoC-style anchored annotation —
the untimedness is a property of the SOURCE, not repairable downstream by any
discounting scheme the literature offers.

## What this changes for TanitAD (≤3 recommendations)

1. **CORRIDOR_OFFSET calibration = lane-boundary reference + causal trigger, not a
   third threshold.** Concretely: run a monocular lane detector (CLRerNet-class,
   research-OK) on a ~100-clip sample, compute lane-relative offset, and emit
   CORRIDOR_OFFSET only where (offset vs lane center is sustained) AND an
   `obstacle.offline` agent occupies the corridor ahead (the CoC "blockage/hazard"
   trigger). The two live derivers become testable against this reference instead of
   against each other; the registered three-armed cross-agreement experiment
   (MM-E2/D-DATA-GTAC-b) stays as-designed and gains a fourth, independent arm.
   0-GPU-blocked today; dev-box GPU after 12:00 suffices for a sample run.
2. **For the open PI decision on untimed CoT tokens, the literature-backed options
   are exactly two** (recommend presenting only these): (i) CLIP-level supervision —
   untimed tokens supervise a clip-level multi-label head (MIL if band attribution is
   wanted as an OUTPUT), never a banded loss directly; expected cost is the P-MIL
   weak-vs-full gap. (ii) TAN-style alignability gating — a cheap grounding pass
   estimates per-token (alignable?, band); only the top-α alignable fraction (TAN's
   measured 30 % is the prior) enters banded training, the rest stays clip-level.
   Blanket band-assignment of untimed tokens (e.g. "all CoT → TACTICAL [2,6]s") has
   no published support and TAN's 30 %/15 % numbers argue it would be mostly wrong.
3. **Adopt the CoC anchoring rule for every FUTURE language-label generation run**
   (any VLM captioning we commission): closed decision vocabulary, anchor to the
   first action after the decision moment, evidence restricted to the history
   window, label only decision-bearing clips. This is upstream prevention — it makes
   the untimed problem structurally impossible for new data, the same way parity
   pinning made re-selection impossible.

## Searches that came up empty
- "Deliberate lane offset" / "corridor deviation" as a *dataset label* with a
  published calibration protocol: nothing beyond the CoC nudge taxonomy (2 search
  rounds; queries in raw/search_log.md). SHRP2/naturalistic-driving lane-position
  literature is not openly downloadable and was not pursued past titles.
- A published discount FACTOR (scalar loss weight) for untimed text on temporal
  heads: none found — the field discounts by MECHANISM (MIL / gating / clip-level),
  not by scalar weighting.

## Evidence table
| # | claim | class | source |
|---|---|---|---|
| F1 | CoC nudge labels = lane-line semantics + causal trigger | PUBLISHED | lib `2511.00088` pp. 11–12 |
| F2 | CULane F1 81.43 / released 81.11–81.55 | PUBLISHED | lib `2305.08366` |
| F3.1 | ~50 % HowTo100M pairs misaligned; MIL over ~5 candidates works | PUBLISHED | lib `1912.06430` |
| F3.2 | 30 % alignable / 15 % well-aligned | PUBLISHED | lib `2204.02968` |
| F3.3 | THUMOS14 mAP@0.5: weak 40.0 vs full 58.6 | PUBLISHED | lib `2305.17861` Tab. 1 |
| F4 | BDD-X inter-annotator temporal IoU 0.63 (SD 0.21) | PUBLISHED | lib `1807.11546` |
| — | obstacle.offline covers 97.44 % of corpus | MEASURED (inherited from registry, not re-verified today) | MODEL_REGISTRY / CLAUDE.md pinned table |
| — | both derivers fire ~63–66 %, conflict 2-vs-1 | MEASURED | GOALS_AND_CLAIMS D-DATA-GTAC-b |
