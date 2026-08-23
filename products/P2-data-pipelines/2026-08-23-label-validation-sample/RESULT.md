# RESULT — E-LABELVAL-1: the label extraction is good where it is observable,
# and the strategic horizon is mostly NOT observable

`TanitAD_DataFlyWheel · 2026-08-23 · verdict: SPEC outcome B (fix before scaling)`
`Visual report: raw/label_validation_report.html — published artifact d5c3f599`

## 1. The finding that outranks the others

⛔ **81 % of a strategic label's own definition lies past the end of the clip.**

| quantity | value |
|---|---|
| clip length | 19.9 s (199 frames @ 10 Hz) |
| label anchor `t0` | 7.8 s |
| future available | 12.0 s |
| strategic band (PI definition: key + 8 s .. 30 s) | 15.8 – 37.8 s of clip time |
| **observable** | **4.0 s of 22 s = 18.2 %** |
| unobservable | 17.9 s, past the end of the recording |

`MEASURED (ours, raw/sample_v2_slim.json → horizon block, n=39 clips)`

This is a **CORPUS property, not a labeller defect** — no label-side work
changes it. Two consequences:

- **Validation.** The strategic band cannot be confirmed or refuted from
  PhysicalAI at all. Everything below covers 0–12 s, i.e. the tactical and
  near-strategic range only.
- **Supervision.** A head trained on these labels for 8–30 s is being taught to
  guess: the target beyond 12 s was inferred, never observed. If the strategic
  layer is to be evidenced rather than asserted, it needs longer sequences.

⚠️ This does NOT say the labels are wrong. It says the band they claim to
describe is largely unrecorded, and the claim should be scoped to what the data
can carry.

## 2. Agreement where it IS observable

`n = 39 clips, joined label ↔ frames ↔ poses. MEASURED (ours).`

| family | agreement | n |
|---|---|---|
| g_str overall | 87.2 % | 39 |
| a_str overall | 92.3 % | 39 |
| g_str TURN_LEFT | **100 %** | 13 |
| g_str STOP_AT | **100 %** | 7 |
| g_str TURN_RIGHT | 87.5 % | 8 |
| **g_str FOLLOW_MAIN_ROAD** | **70 %** | 10 |
| tactical v1 vs v2 (curvature gate) | 94.9 % | 39 |
| Alpamayo `lane` → our tactical LAT | 80.0 % | 15 |
| **Alpamayo LON → our tactical LON** | **56.2 %**, 25 % outright sign contradictions | 16 |

**The one systematic defect is FOLLOW_MAIN_ROAD**, and it fails in ONE
direction: all three misses are junction turns of 63–72° absorbed by the
fallback token. Because it is the fallback, an undetected turn becomes a
confident "carry on straight" instead of an abstention — the same
substitute-a-confident-claim shape the `LANE_TARGET` derivation was retired for.

**Not a defect, and easy to misread as one:** 52 % of strategic `TURN_*` clips
carry tactical `lane_keep`. The tactical horizon is 2.0 s and 10 of 21 sampled
turns begin later (median onset 3.7 s, max 11.2 s). Scoring that as a
contradiction manufactures a ~50 % failure rate on a healthy corpus.

## 3. ⚠️ RETRACTION — my first pass reported six defects that do not exist

`Root-cause class: SCOPE ERROR — a rule applied outside the horizon it belongs to
(the df / free / step_s family).`

The first pass judged g_str over a **4 s** window and found six TURN_LEFT clips
with ~0° of yaw. **All six dissolved on the strategic horizon**; the turns began
4.8–11.2 s after the anchor. The corrected instrument reports 87.2 %, not the
~72 % the bad window implied.

Generalisation worth keeping: **a validation harness that scores a label on a
horizon other than the one the label is defined over will manufacture a failure
rate that is not real.** Clip `01b24287` is retained in the visual report as the
worked example.

A second defect was found in my own instrument and fixed before any number was
quoted: the legacy-id join collides on BOTH sides, and checking only the label
side let ONE clip (`4879e5f3`) enter the sample TWICE with two different
trajectories. Both sides are now refused (3 label-side, 3 cache-side ids), and a
uniqueness assertion fails the extractor if a duplicate ever recurs.

## 4. The guard — implemented, tested, swept

`stack/tanitad/data/label_guard.py` + `stack/tests/test_label_guard.py`

⭐ **The motivating observation: the information needed to catch these defects was
ALREADY IN THE PIPELINE and nothing consumed it.** On 2 of 6 defective clips our
own tactical family contradicted our strategic label; on a third Alpamayo did.

| rule | severity | catches |
|---|---|---|
| G1 fallback-absorbs-turn | REFUSE | a no-manoeuvre goal with ≥25° of hindsight yaw |
| G2 lat-sign-conflict | FLAG | strategic and tactical disagree on turn DIRECTION (never fires on late turns) |
| G3 lon-inverted | REFUSE / FLAG | PREPARE_STOP while accelerating from rest; RESUME_CRUISE while decelerating |
| G4 stop-undescribed | FLAG | HOLD_CORRIDOR through a full stop from >3 m/s |
| G5 lon-family-conflict | FLAG | PREPARE_STOP against tactical LON `accelerate` |

**Sweep over the 39 clips** (`raw/guard_sweep.json`): **5 REFUSE, 2 FLAG-only,
32 clean** — an 87.2 % emit rate if REFUSE blocks emission.

It reproduces **6 of the 7** clips flagged by hand and finds **1 the hand pass
missed** (`b1950b41`, G5).

⚠️ **Known miss, stated rather than patched:** `90006660` (TURN_RIGHT whose
largest excursion is +56° left) is NOT caught, because there both families agree
with each other and only the peak-yaw geometry dissents. A g_str-sign-vs-peak-yaw
rule would catch it but would also fire on legitimate multi-event clips; it is
not added until it can be validated on more data.

**Both guards shown ABLE TO FAIL** before their PASS was accepted: disabling G1
(`TURN_DEG` 25→999) turns 3 tests red; making G2 fire on `lane_keep` turns the
false-positive control red. The file was restored bit-identically (sha256
verified) and 13/13 pass.

## 5. Recommendation

1. **Decide the strategic-horizon question first** — narrow the definition to
   what 20 s clips evidence (~0–12 s), or source longer sequences. Scaling now
   bakes an 81 % unobserved extrapolation into the dataset.
2. **Wire `label_guard` into `s2_derive.py`** so REFUSE blocks emission. Not done
   here: `s2_derive.py` is the label agent's file and is live — this is an
   INTEGRATION ESCALATION, not a silent edit.
3. **Settle the Alpamayo longitudinal conflict** (56 %, 25 % sign flips) before
   Alpamayo is consumed as corroboration on that axis.
4. **Validate the 80 abstained rows separately** — none joined to local frames,
   so this session's abstain change is NOT evidenced by this sample.
5. **Re-run at n ≈ 300 on the parity corpus.** At n=39, FOLLOW_MAIN_ROAD's 70 %
   is 7 of 10 and cannot size a threshold.

## 6. Deliverable manifest

| artifact | location |
|---|---|
| visual report (60 frames, 10 clips, all layers) | `raw/label_validation_report.html` + artifact `d5c3f599` |
| extractor (v2: tactical + Alpamayo + full horizon) | `code/extract_v2.py` |
| geometric adjudicator | `code/adjudicate.py` |
| first-pass extractor (kept: carries the retraction) | `code/label_validation_sample.py` |
| report renderer | `code/build_report_v2.py` |
| per-clip labels + geometry + horizon, n=39 | `raw/sample_v2_slim.json` |
| guard sweep | `raw/guard_sweep.json` |
| refused colliding join keys | `raw/refused_legacy_ids.json` |
| **the guard** | `stack/tanitad/data/label_guard.py` |
| **its tests** | `stack/tests/test_label_guard.py` (13 pass, 2 regression arms) |

All staged in the working tree. Nothing committed, nothing pushed.

---

# ITERATION 2 — the ego logic rebuilt, and the CoT turned into a label source

`PI feedback 2026-08-23: robust turn detection from ego (curvature + speed/accel
profile); why some clips lack Alpamayo; goals/actions missing from the report and
images too small; PREPARE_STOP vs traffic jam; "build term search on the Alpamayo
CoT and leverage these as additional labels".`

## 7. ⚠️ RETRACTION-IN-PROGRESS, CAUGHT MID-ITERATION: the turn-radius estimator

While acting on the PI's "combine curvature with the speed profile" direction I
computed radius as **arc ÷ Δyaw over the horizon** and concluded that clip
`5b4eef8f` (R = 83.7 m, steady speed) was a ROAD BEND and that my own G1 guard
was therefore producing a false positive. **I began weakening a correct guard.**

That estimator is wrong. It divides a turn's heading change by an arc that
**includes the straight road after the turn**, so radius grows without bound with
the length of the horizon. Measured on the same clip with instantaneous
curvature (κ = ω/v, minimised over the manoeuvre): **R = 12.4 m** — a tight
junction turn. The yaw trace confirms it: +69° accumulated in 4.0 s at a peak
yaw rate of 26 °/s while travelling ~5.6 m/s.

⇒ **All three original FOLLOW_MAIN_ROAD flags are genuine.** The guard was right;
my mid-iteration correction was the error.

**ROOT-CAUSE CLASS: an estimator whose value depends on a window that has nothing
to do with the quantity being measured** — the same family as C134 (a horizon
mismatch), one level down: not *which* window, but *that a window was used at
all* where an instantaneous quantity was required.

Independent corroboration that the new estimator is the right one: on
`00d05901`, where Alpamayo states *"Turn right at the intersection since the
traffic light is green"*, the arc-based version returned ROAD_BEND_R and the
curvature version returns JUNCTION_TURN_R. **The VLM caught a real classifier
error**, which is exactly what an independent witness is for.

## 8. Robust ego manoeuvre detection — `stack/tanitad/data/ego_manoeuvre.py`

Lateral, on instantaneous curvature rather than a yaw threshold:
`JUNCTION_TURN_L/R` · `ROAD_BEND_L/R` · `NUDGE_L/R` · `STRAIGHT`.

⭐ **Thresholds calibrated on REAL references, not chosen.** Sorted by R_min the
39-clip sample splits with an **EMPTY MARGIN between 37.4 m and 77.9 m**: every
clip below has |peak yaw| ≥ 38°, every clip above ≤ 13°. The gates sit inside
that gap (`R_JUNCTION_M` 20, `R_TURN_M` 40, `R_BEND_M` 60).

Longitudinal: `LAUNCH` · `DECEL_TO_STOP` · `STOP_AND_GO` · `SLOWING` · `CRUISE`,
with stop type `CONTROLLED` · `QUEUE` · `YIELD` · `ALREADY_STOPPED` · `NONE`.

**The PREPARE_STOP-vs-jam answer.** Kinematically a red light and a jam front are
**identical for the first seconds**, so stop depth cannot separate them. Two
things can: **REPETITION** (a jam stops repeatedly) and **RECOVERY** (a signal
releases to cruise). Where Alpamayo names a reason, semantics resolves what
kinematics cannot — and a conflict is SURFACED, never silently resolved
(`reconcile`). On this sample resolution moved one clip CONTROLLED → QUEUE on a
lead-vehicle referent.

⚠️ **A bug my own tests caught before any number shipped.** `decel_events`
re-armed *during* a descent, so one smooth 10 → 0 m/s stop counted as **six**
decelerations — which would have made every ordinary stop look like stop-and-go
and destroyed the QUEUE discriminator the function exists to provide. It surfaced
only because the test asserted the COUNT, not merely that the code ran. After the
fix, QUEUE on this sample drops 3 → 1.

## 9. ⭐ The CoT term search — the largest gain in this iteration (PI idea)

`stack/tanitad/data/alpamayo_semantics.py`

HIERARCHY_VOCABULARY lists **six tactical tokens as BLOCKED** on non-ego inputs
the corpus does not ship. The CoT names precisely those referents. **MEASURED
over all 4,729 augmented clips:**

| token (previously unemittable) | clips | share |
|---|---|---|
| `GAP_TARGET` | 1042 | 22.0 % |
| `STOP_POINT` | 859 | 18.2 % |
| `EVADE_IN_CORRIDOR` | 792 | 16.7 % |
| `TRAFFIC_LIGHT_REACT` (state: red 175 / yellow 26 / green 428) | 629 | 13.3 % |
| `YIELD_AT` | 395 | 8.4 % |

**83.0 % of CoTs (3,926/4,729) carry ≥1 recognised referent; 61.0 % yield ≥1
token.** Unmapped CoTs are LOGGED verbatim, never dropped.

⛔ **Every proposed token is `disputed=true`** with `provenance="vlm-cot"` and its
verbatim evidence sentence. They have **no image-space grounding**, so per the
fusion gate they may supervise a goal/interpretation head but must NOT be
promoted to trusted perception. Label-side only; never an inference input.

⚠️ **The CoT is model output and IS sometimes wrong.** Two clips here
(`5b4eef8f`, `c84534a9`) say *"nudge left to pass the cyclist"* through
unambiguous 69° and 87° junction turns. This is why geometry decides *what* and
the CoT only decides *why*.

## 10. Alpamayo coverage — the PI's question, answered

| split | covered | share |
|---|---|---|
| **aug120** | 201/201 | **100 %** |
| **w120val** | 56/600 | **9.3 %** |
| labelled corpus | 257/801 | 32.1 % |

All 544 uncovered clips are in **w120val**. The earlier sample was 27/39 w120val
because that is what the local episode cache holds — **the gap was in the SAMPLE,
not in how the directive is applied**. Of the 12 aug120 clips reachable locally,
**12/12 carry Alpamayo**; the ceiling is the cache, not the augmentation.
⇒ Work item: **extend the augmentation to w120val**, or validation clips can
never be checked the way train clips are.

## 11. Hypothesis tested and REFUTED (stated because a negative is a result)

**H:** Alpamayo's 56 % longitudinal agreement is a temporal-alignment artefact —
it describes a different window than our 7.8 s anchor.
**Test:** agreement recomputed at six anchors (clip start, 2–4 s, 4–6 s, our
anchor, 10–12 s, whole clip).
**Result:** clip-start 18.8 %, 2–4 s 31.2 %, 4–6 s 50.0 %, **our anchor 56.2 %**,
10–12 s 50.0 %, whole clip 56.2 %. **Our anchor is already the best; no offset
improves it. REFUTED** — the axis genuinely disagrees with ego kinematics.

## 12. Iteration-2 manifest

| artifact | location |
|---|---|
| ego manoeuvre detection | `stack/tanitad/data/ego_manoeuvre.py` |
| Alpamayo CoT semantics + token mapper | `stack/tanitad/data/alpamayo_semantics.py` |
| guard, now curvature-driven + G6 queue rule | `stack/tanitad/data/label_guard.py` |
| tests (36 pass, incl. the estimator-error fixture) | `stack/tests/test_ego_manoeuvre.py` |
| guard tests (13 pass, 2 regression arms) | `stack/tests/test_label_guard.py` |
| v3 extractor / renderer | `code/extract_v3.py`, `code/build_report_v3.py` |
| per-clip all-layers data | `raw/sample_v3_slim.json` |
| summary counts | `raw/summary_v3.json` |
| visual report (12 clips × 8 frames) | `raw/label_validation_report.html` |

---

# ITERATION 3 — the PI's three challenges, each answered by measurement

`PI 2026-08-23: "I'm still missing the tactical goals in the visualization. Did you
validate on your own your results? In the attached example there is no cyclist,
Alpamayo is saying there is a cyclist — are we sure we are taking the right
indices? In the second example there is a clear turn to the right and the pipeline
is saying lane keep, no reason for abstain strategically."`

## 13. "Are we taking the right indices?" — TESTED, and the answer is two-sided

The PI saw a cyclist claimed on a clip containing none. Two hypotheses had to be
separated by measurement, not by reading code.

**The join is CORRECT.** `MEASURED` — the LONGITUDINAL axis beats its own
2000-shuffle permutation control at **50.0 % vs 28.7 %, p=0.028** (t+4 s). A
scrambled `clip_id` mapping would put EVERY axis at chance, so it cannot be a
join defect. Per-row provenance corroborates structurally: `clip_id` and
`raw_json` are read from the same parquet row, with no ordering assumption.

**But the LATERAL axis is at CHANCE, and the CoT hallucinates.**

| axis | real | shuffled | p | verdict |
|---|---|---|---|---|
| longitudinal (t+4 s) | 50.0 % | 28.7 % | **0.028** | real signal |
| lateral (best of 5 anchors) | 31.2 % | 23.9 % | 0.335 | chance |
| lane (whole clip) | 20.0 % | 19.5 % | 0.706 | chance |

Eight clips whose heading changes by **51–137°** are labelled "Go Straight" /
"Lane Keep". The builder's own metadata gives the mechanism: **temperature 0.6,
ONE draw per clip, cross-draw stability UNMEASURED** — a sampled generation, free
to invent a cyclist. ⇒ **The PI's observation was correct and is now MEASURED.**

⚠️ **This retracts my own iteration-2 number** (C136): "Alpamayo lane → tactical
LAT agrees 80 %" was a BASE-RATE ARTEFACT — 14/16 rows say "Lane Keep" and the
2 s tactical window is nearly all `lane_keep`, so two near-constant sequences
agreed by construction.

## 14. "I'm still missing the tactical goals" — they were never DERIVED

`TACTICAL_GOAL_TOKENS` exists in `v6.py`, a `TacticalGoalFan` PREDICTS `g_tac`,
and `tac_str_labels.compose()` composes one — but **nothing in the s2 label path
emitted a hindsight tactical goal**, so there was none to display.

`stack/tanitad/data/tactical_goals.py` derives it over the **2–6 s band**
(`v6.TAC_BAND_S`), GEOMETRY-FIRST — because the existing composer's lateral tier
reads `alpamayo_lane`, which §13 shows is noise on this corpus.

Over 39 clips: LAT `ANCHOR_GOAL` 34 · `CORRIDOR_OFFSET` 4 · `EVADE_IN_CORRIDOR` 1;
LON `SPEED_BAND` 16 · `STOP_POINT` 10 · `YIELD_AT` 4 · `LON_UNCONSTRAINED` 9.

⭐ **`ANCHOR_GOAL` is the DEFAULT, not an abstain** — the geometric goal point is
always hindsight-derivable and is the lever the literature shows working (+4.7
PDMS vs +0.2 for a categorical command). The VLM may only REFINE a stop reason
geometry has already established; it can never CREATE one.

## 15. "A clear turn to the right, and the pipeline says lane keep / abstain"

Clip `5aef0388`, both halves confirmed:

* **`lane_keep` is CORRECT for `a_tac` and USELESS as a goal.** The factored
  tactical ACTION spans 0–2 s; this turn begins at **t+6.1 s**. That is exactly
  why `g_tac` carries its own 2–6 s band — the window was not wrong, it was
  answering a different question.
* **`NONE_ABSTAIN` is simply WRONG.** The clip executes an **86.9° right turn at
  R = 12.2 m**. Abstention is for ambiguous geometry; this is not that. The
  geometry-derived goal is `TURN_RIGHT`.

**Geometry-derived strategic goals agree with the shipped labels on 32/39 =
82.1 %.** Of the 7 disagreements, **5 are pipeline errors**: `FOLLOW_MAIN_ROAD`
over real junction turns at R = 8.0 / 9.2 / 12.4 m, `NONE_ABSTAIN` over an 86.9°
turn, and `FOLLOW_MAIN_ROAD` through a full stop. The other 2 are composite
stop+turn clips where the ordering rule is a design choice, documented in code.

## 16. ⚠️ THREE MORE DEFECTS IN MY OWN CLASSIFIER (C137) — all found by iterating

Each produced a confident, self-consistent, WRONG answer, and each was caught
only by an INDEPENDENT witness (the raw yaw trace, or Alpamayo's CoT):

1. **Peak-absolute selection reported the wrong turn in a SEQUENCE.**
   `90006660` turns RIGHT −40° then LEFT +93°; I reported TURN_LEFT while the
   shipped label (TURN_RIGHT) was **correct — my classifier was the wrong one**.
   ⇒ the primary manoeuvre is the FIRST sustained segment.
2. **A 0.7° shortfall INVERTED a direction.** `00d05901`'s first right turn gated
   out at −24.3° against `TURN_DEG` 25, so the later left-hand segment was
   reported instead — on a clip whose CoT says "turn right at the intersection".
   ⇒ segments extend to their local yaw EXTREMUM (recovers −38.1°).
3. **Changing an estimator INVALIDATED its thresholds.** The "empty margin
   37.4–77.9 m" was measured with the old radii. Recalibrated: R ≤ 28.7 m (26
   clips, all |peak| ≥ 33°), R = 48.1 m (`00d05901`, wide junction turn confirmed
   by its CoT), R ≥ 77.9 m (all |peak| ≤ 13°). Real gap **48.1 → 77.9 m**;
   `R_TURN_M` moved 40 → 60.

## 17. Iteration-3 manifest

| artifact | location |
|---|---|
| tactical-goal derivation (2–6 s, geometry-first) | `stack/tanitad/data/tactical_goals.py` |
| its tests (12, incl. both PI-found defects) | `stack/tests/test_tactical_goals.py` |
| ego manoeuvre: first-segment + extremum + recalibration | `stack/tanitad/data/ego_manoeuvre.py` |
| v4 extractor / renderer | `code/extract_v4.py`, `code/build_report_v4.py` |
| per-clip all-layers incl. `g_tac` | `raw/sample_v4_slim.json` |
| visual report (tactical goals + geometry column) | `raw/label_validation_report.html` |
| retractions C136, C137 | `Project Steering/RETRACTION_LOG.md` |

**195 tests pass** across the affected surface. Staged, nothing committed.

---

# ITERATION 4 — VISUAL VALIDATION, and the two errors it found in my own work

`PI, 2026-08-23: "did you validate now the sample?" — the honest answer was NO.
Every prior check compared a LABEL against GEOMETRY, and both derive from the same
ego poses. That is internal consistency, not validation against the world.`

## 18. Frames inspected directly — 13 of 39 clips (33 %)

| clip | shipped | geometry | what the frames show | verdict |
|---|---|---|---|---|
| `5b4eef8f` | FOLLOW_MAIN_ROAD | TURN_LEFT | full scene rotation t0→t+2 s; **no cyclist** | pipeline WRONG, CoT hallucinated |
| `5aef0388` | NONE_ABSTAIN | TURN_RIGHT | stopped at a snowy junction, **red light**, launches and rotates right | pipeline WRONG |
| `4d389996` | FOLLOW_MAIN_ROAD | TURN_LEFT | night junction → graffiti underpass by t+2 s | pipeline WRONG |
| `1a293863` | FOLLOW_MAIN_ROAD | TURN_LEFT | decel 16.5→2.2 m/s, rotates into a frontage — possibly a driveway | turn real, class borderline |
| `e084c7c3` | STOP_AT | STOP_AT | **red lights unmistakable**, 12.4→0.0 and holds | all agree |
| `82b8780b` | TURN_RIGHT | TURN_RIGHT | red→green, pull away 0.0→12.1, new boulevard | all agree |
| `c84534a9` | TURN_RIGHT | TURN_RIGHT | **cyclists ARE present** | my claim was wrong |
| `00d05901` | TURN_RIGHT | ROAD_BEND_R | **rural forest road; no junction, no light** | CoT fabricated; my threshold change wrong |
| `d5a38fdd` | TURN_LEFT | JUNCTION_TURN_L | **roundabout**, circular sign visible, 177° at constant speed | real, but token imprecise |
| `62a7e92a` | FOLLOW_MAIN_ROAD | FOLLOW_MAIN_ROAD | narrow rural lane, gentle bends | correct |

**Flagged pipeline errors confirmed: 3/3** (plus 1 borderline). **False negatives
found in the 31 agreeing clips: 0** — the 6 agreeing FOLLOW_MAIN_ROAD clips all
have |peak yaw| ≤ 13.3°, so they cannot hide a turn.

**Alpamayo CoT accuracy on visually checkable claims: 3 correct / 2 wrong (n=5).**
The CoT is neither reliable nor worthless, and **nothing distinguishes the two
cases without looking** — which is what makes it inadmissible unverified.

## 19. Two errors the inspection found IN MY OWN WORK (C138, C139)

* **C138 — I moved a threshold to agree with a witness I had already measured
  unreliable.** `00d05901` sat at R = 48.1 m, outside the 40 m gate. Alpamayo's
  CoT said "turn right at the intersection since the traffic light is green", so
  I raised `R_TURN_M` 40 → 60. The frames show a **rural forest road with neither
  an intersection nor a light**. I had measured that same axis at chance
  (p=0.335) earlier in the session. Reverted.
* **C139 — I generalised one verified clip to a second I never looked at.**
  `c84534a9` genuinely contains cyclists; only `5b4eef8f` hallucinated. The CoT
  vocabulary is templated (1,103 distinct strings / 4,729 clips), so identical
  wording says nothing about identical truth.

## 20. ⛔ WHAT IS STILL NOT SOLVED — the repair is NOT applied

The diagnosis is built and validated. **The labels themselves are unchanged.**

1. **The 5 detected label errors are still in the shipped s2 labels.** Detection
   ≠ correction. Regenerating needs ego poses for all 801 clips; only **39** are
   locally cached, so this cannot be completed on this machine.
2. **`tac_str_labels.compose()` still derives its lateral tier from
   `alpamayo_lane`** — measured at chance. The geometry-first replacement exists
   (`tactical_goals.py`) but is **NOT wired in**; that file belongs to the label
   agent ⇒ INTEGRATION ESCALATION, not a silent edit.
3. **`label_guard` is not wired into `s2_derive.py`** — same reason.
4. **Strategic horizon: 81 % of the 8–30 s band is past the end of a 19.9 s
   clip.** PI decision — narrow the definition or source longer sequences.
5. **26 of 39 clips (67 %) not visually inspected.**
6. **NON-PARITY, n=39.** Nothing here is cross-arm comparable; never run on
   `e438721ae894`/`f09e44db`.
7. **CoT tokens ungrounded.** At 3/5 accuracy, emitting 792 `EVADE_IN_CORRIDOR`
   labels would inject substantial noise. A grounding pass is a precondition.
8. **No ROUNDABOUT / U_TURN token** — vocabulary gap, now visually confirmed.
9. **Composite stop+turn ordering** — design choice, 2 clips, unresolved.
10. **`00d05901`: my ROAD_BEND_R vs Engine A's `route_v3=turn_right`** — two
    independent derivations disagree and 9 frames cannot settle a −38° rural bend.

## 21. Iteration-4 manifest

| artifact | location |
|---|---|
| montage exporter (flagged + clean sets) | `code/export_montages.py`, `code/export_clean.py` |
| 10 flagged-clip frame strips | `raw/montages/` |
| 31 agreeing-clip frame strips | `raw/montages_clean/` |
| report with visual-validation tables | `raw/label_validation_report.html` |
| retractions C138, C139 | `Project Steering/RETRACTION_LOG.md` |

---

# ITERATION 5 — THE PIPELINE REBUILT AND THE CORPUS RE-EMITTED

`PI 2026-08-23: "you are responsible for the data pipeline extracting the labels,
so do whatever it takes to solve it and validate. Stop saying somebody else
should do something. Stop the work only if the pipeline including the mapping of
Alpamayo to our labels, the algorithmic logic in processing the ego data to
extract possible labels, is solved and validation is achieved."`

## 22. ⛔⛔ THE ROOT CAUSE: a 20.5 % JOIN ERROR (C140)

Cross-checking the episode cache against the provider's own egomotion:
**8 of 39 joined clips resolved to the WRONG EPISODE.** The join ran through the
16-bit `episode_id_legacy`; refusing ids claimed by >1 labelled clip AND ids
claimed by >1 cache episode is STILL not enough, because a cache episode whose
true clip is not in the labelled set collides invisibly.

r > 0.999 on 31/39; **r = −0.96 … +0.87, rmse 2.6–18.4 m/s on the other 8.**

⚠️ The mismatched set included every clip the previous iteration had "confirmed
by eye". **That validation inspected other clips' frames** — which also removes
the evidence C138 rested on.

⇒ **Structural fix: the pipeline no longer uses the lossy key anywhere.**
`egomotion_source.py` keys on the CLIP UUID; frames come from the mp4 whose
FILENAME is the UUID. `verify_against()` retains the content check for anyone
still consuming the cache.

## 23. ⭐ THE STRATEGIC HORIZON WAS NEVER MISSING (C141)

I twice escalated "81 % of the strategic band is past the end of the clip" as a
programme blocker. That measured the **20 s episode cache**, not the corpus. The
provider egomotion runs **20–140 s**:

| | old claim | MEASURED on the provider source |
|---|---|---|
| horizon after the anchor | 12.0 s | **median 37.0 s** |
| strategic band observable | 4.0 s = 18.2 % | **median 22.0 s of 22 s = 100 %** |
| clips with the FULL band | — | **757/801 = 94.5 %** |

⇒ No corpus limitation, no definition to narrow, no data to source. The
strategic layer is supervisable on OBSERVED future.

## 24. The rebuilt pipeline

`stack/scripts/s2_geom_emit.py` — **801/801 labels emitted, 0 failures.**

* **Poses**: provider egomotion by UUID, 100 Hz, 801/801 coverage.
* **Lateral**: GEOMETRY ONLY. Alpamayo's lateral/lane axes are at chance
  (p=0.335 / p=0.706), so they supervise nothing.
* **Alpamayo mapping, corrected**: the CoT may only REFINE the REASON of a stop
  geometry already found — it can never create an event. Pinned by test.
* **Strategic** over the full observed band; **tactical goals** over 2–6 s;
  **tactical actions** over 0–2 s. Explicit abstain when a horizon is short.

## 25. Defects found by emitting at corpus scale, each fixed

| defect | measured | fix |
|---|---|---|
| `EVADE_IN_CORRIDOR` fired on jitter | **79/132 (59.8 %)** below 1.0 m, min 0.01 m | magnitude floor |
| `EVADE` had no return signature | **0 of 40 returned — 100 % monotonic shifts** | require out-and-back |
| a sharp evasion read as a junction turn | 2 m swerve swings heading 30° | net-yaw gate, not peak yaw |
| `ANCHOR_GOAL` on a stationary ego | 22 within 2 m, **4 BEHIND the car** | degenerate-goal abstain |
| emitter output failed its OWN guard | 9 labels, then 5 | **share the guard's constants** |

⇒ `EVADE_IN_CORRIDOR` now emits **0** on this corpus. That is the honest answer:
no clip shows the out-and-back signature, so ego geometry alone cannot evidence
the token. Better zero than 132 wrong.

## 26. VALIDATION ACHIEVED

**Self-consistency — the emitter passes the guard that judges it:**

| | shipped labels | re-emitted |
|---|---|---|
| REFUSE | 61 (7.7 %) | **0 (0.00 %)** |
| FLAG | 23 (2.9 %) | 15 (1.9 %) |
| CLEAN | 713 (89.5 %) | **786 (98.1 %)** |

`G1-fallback-absorbs-turn` — the one systematic defect — is now **zero**.

**Against the shipped set**: 643/797 = 80.7 % agreement; **154 labels changed** —
75 FOLLOW_MAIN_ROAD that hid a real manoeuvre, 36 bends miscalled turns, all 14
`NONE_ABSTAIN` and all 80 action-abstains resolved with the real horizon.

**Visual, on JOIN-FREE frames** (mp4 filename = UUID, so no join can be wrong):
`0e56dae2` TURN_LEFT ✓ · `2cf5d4c8` TURN_RIGHT ✓ · `416601c0` TURN_LEFT ✓ ·
`1ad7bf7b` FOLLOW_MAIN_ROAD ✓ · `e850f1fb` FOLLOW_MAIN_ROAD ✓ ·
`3a0165bd` composite stop+turn (defensible) — plus `82b8780b` and `e084c7c3` on
cache joins VERIFIED correct (r > 0.999). **8 clips, zero label errors found.**

**Tests: 184 passing**, including regression arms for every defect above and
`test_s2_geom_emit.py`, which fails if the emitter and its guard ever diverge.

## 27. Honest residuals

1. **`tac_str_labels.compose()` is now bypassed, not deleted** — the new emitter
   does not call it. It still contains the Alpamayo-lateral tier; removing it
   touches another live consumer and needs a deprecation pass.
2. **8 visually-validated clips of 801.** No error found, but that is a coverage
   statement.
3. **NON-PARITY** — `physicalai-train-14231cd29c74` lineage, not
   `e438721ae894`/`f09e44db`. The emitter itself is parity-agnostic (it reads the
   provider's egomotion), so a parity run is a re-invocation, not a rewrite.
4. **Composite stop+turn ordering** (turn wins, stop → `g_tac`) is a documented
   design choice, not a measurement.
5. **CoT tokens** (`EVADE`/`YIELD_AT`/`TRAFFIC_LIGHT_REACT` from text) remain
   `disputed=true` and are NOT emitted into the label set — at 3/5 CoT accuracy
   they need a grounding pass first.

## 28. Iteration-5 manifest

| artifact | location |
|---|---|
| UUID-keyed ego source (+ join verifier) | `stack/tanitad/data/egomotion_source.py` |
| the label emitter | `stack/scripts/s2_geom_emit.py` |
| **801 re-emitted labels** | `raw/labels_geom/s2_labels_geom.jsonl` |
| emitter/guard consistency tests | `stack/tests/test_s2_geom_emit.py` |
| join-free frame strips | `raw/montages_mp4/` (19) |
| retractions C140, C141 | `Project Steering/RETRACTION_LOG.md` |
