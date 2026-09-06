# PI VIDEO REVIEW — refcv4b reel, 2026-09-06

⛔ **WHY THIS FILE EXISTS.** For a full day the programme answered these observations
**by number** — "#12 is a systematic error", "#7 maps to P15" — while the observations
themselves lived **only in a session transcript and not in git**. Agents were briefed
against them, work was prioritised by them, and a register row cited them, all against
an unversioned source. A composing agent caught it on 2026-09-07. ⇒ Banked here so a
reference to "#9" resolves to something a reader can open.

⚠️ **PROVENANCE, stated because it limits what this file is.** Transcribed by the
Master Mind from the PI's message of 2026-09-06 in session `tanitad-6b`. It is a
**faithful transcription, not a verbatim quote of a document** — the PI wrote prose,
and the numbering below is the numbering the programme has been using. ⛔ Where his
intent is ambiguous the ambiguity is preserved rather than resolved. If the PI
disputes any item, **his reading wins over this file without argument.**

⚠️ **A NUMBERING ERROR ALREADY PROPAGATED FROM THE UNVERSIONED SOURCE.** The Master
Mind briefed an agent that observation **#7** (speed overshoot) maps to piece **P15**
(max speed). It does not: **P15 is bound to observation #10.** That mis-mapping exists
nowhere in the repo and could only have come from the transcript being re-read from
memory — which is exactly the failure this file closes.

---

## The twelve observations, with status

| # | the PI's observation | status 2026-09-07 |
|---|---|---|
| **1** | Strategic and tactical goals are missing their **constraints** — the model must emit a goal *and* estimate its constraints, such as position and time. | ⛔ OPEN. No goal carries an estimated constraint today. |
| **2** | The selector picks a trajectory that **cuts road markings** while better candidates are present in the fan. | ⭐ PARTLY ADDRESSED by **P14** (sampler ranks its own fan): a >2× better trajectory is in the fan on **41.09 %** of windows, ceiling ADE 0.4728 → 0.1914 m. ⚠️ But **98.9 % of that gain is ALONG-TRACK** and curvature is not separated — it is a longitudinal lever, **not a steering fix**. |
| **3** | The overlay claims the model does not read the nav token at inference — **"is it true?"** | ⭐ ANSWERED: **the nav token IS consumed.** ⚠️ But its provenance is `ego-future` on 4,719/4,719 records, so on this corpus it is an **oracle**, not the production nav command. |
| **4** | The model does not follow nav commands (the roundabout example). | ⛔ OPEN, and entangled with #3: an oracle-provenance nav input cannot settle whether the model follows a *production* command. |
| **5** | The GT trajectory and the fan gradient are **not smooth**. | ⛔ OPEN. Not addressed by anything composed for refcv5-v2. |
| **6** | Trajectory selection **jumps between consecutive frames**. | ⛔ OPEN. No temporal-consistency term exists. |
| **7** | **Speed overshooting.** | ⛔ OPEN. ⚠️ NOT P15 — see the numbering note above. The output-scored `frac_over` in `stack/tanitad/eval/constraints.py` makes it *measurable*; nothing yet makes it *smaller*. |
| **8** | Not avoiding collisions; not keeping distance to infrastructure. | ⭐ HALF-MOVES under **P1-IN** (agent conditioning) only. ⛔ Infrastructure clearance remains **unmeasurable** — it is overhead (+1.68 m) and the join drops `z`. |
| **9** | Trajectories are not smooth, despite the claim that they are smooth **"by construction"**. | ⛔ OPEN, and the claim is refuted: REF-C is **~84× worse on curvature than a plan that never steers** (0.02737 vs 2.30973), a finding that survives the honest raw-input floor. |
| **10** | **Target speed as a tactical goal — extend the dataset with a MAX SPEED constraint.** The model must adapt speed to the situation and reach max speed when allowed. | ⭐ IN PROGRESS = **P15**. PI ruled 2026-09-06: use the speed band's upper bound, **quantized to legal steps**, as an INPUT. ⚠️ Its raw form is the ego's future speed (corr 0.941 with `v0`, but 51.5 % of clips differ by >1 m/s, p95 +6.12 m/s) ⇒ a **train/deploy mismatch**, not a leak, and quantization is the mitigation. |
| **11** | Not braking for traffic lights. | ⛔ OPEN as a *behaviour*. ⭐ But the LABEL question is settled: the colour **is observed** — RED stops at **39.3 %** vs GREEN **2.1 %**, a **18.58×** ratio against a pre-committed 2.0× bar. ⚠️ Presence is only partially grounded (182/805, **0 contradicted**) and **no independent channel carries the colour**. |
| **12** | **Lane changes never activate — "check if there is a systematic error."** | ⭐⭐ ANSWERED: **YES, and it is structural.** The v7 vocabulary declares **8** lateral actions; the emitter reaches **5**. `LANE_CHANGE_L`, `LANE_CHANGE_R` and `ABORT_LC` are unreachable by construction (`s2_geom_emit_v7.py`, three assignments at :446/:449/:452). ⛔ And the text-side supply ceiling is **105 clips ≈ 2.2 %** — a CORPUS fact, so repairing the emitter makes the class **expressible**, not **supplied**. |

---

## ⛔ The honest scorecard

**Of the twelve, refcv5-v2 as composed moves ONE (#2, and only longitudinally), answers
TWO as questions (#3, #12), half-moves ONE under P1-IN (#8), and has ONE in progress
(#10).** Smoothness (#5, #9), frame-to-frame jitter (#6), speed overshoot (#7), traffic-light
braking (#11), goal constraints (#1) and nav following (#4) **will not move.**

⚠️ That is not a reason to delay the run — it is the number to hold against it when it
lands, so a separated ADE delta is not mistaken for having addressed this review.

⭐ **Two of the twelve turned out to be questions with clean answers rather than defects
to fix (#3, #12), and both answers were structural** — an oracle-provenance input, and a
vocabulary the emitter cannot reach. Neither would have been found by tuning.


---

## ⛔ ADDENDUM 2026-09-07 — THE "~84×" IN ROW #9 IS UNDER CHALLENGE AND MUST NOT BE REQUOTED

Row **#9** above cites *"REF-C is ~84× worse on curvature than a plan that never
steers (0.02737 vs 2.30973)"*. ⛔ **That figure did NOT reproduce.** The P1 gate
arm measured the same contrast at **1.0×** on its corpus.

⚠️ **That is not yet a refutation** — different corpus (B1 EVAL split 104/35),
different checkpoint, 500 steps. Two numbers from different surfaces disagreeing
is the *scope-error* family, not automatically an error in either.

⭐ **BUT A CONCRETE MECHANISM IS NAMED, AND IT WOULD BE LARGER THAN THIS ROW.**
The v3 horizon set is **`[5, 10, 15, 20, 30, 40, 50, 60]`** — **not uniform** —
while `_seq_geometry` divides by a single `dt`. A rate computed by dividing a
non-uniform sequence by one constant time step is wrong wherever the spacing
changes, and **curvature, yaw-rate and heading-rate are all rate metrics**. ⇒ if
this holds, it touches **every curvature number in the programme**, not this row.
Same defect class as an anchor column that declares no units — the arithmetic is
fine and the quantity is not what its name says. The gate agent reported every
rate metric on the **uniform 2 s prefix only**, deliberately.

⛔ **UNTIL RE-DERIVED WITH ITS HORIZON AND GRID STAMPED, DO NOT QUOTE THE 84×** —
including in `REFCV5_MISSING_PIECES_PLAN.md` §8.2 and §6, which carry it, and in
any briefing. ⚠️ The Master Mind quoted it to the PI several times on 2026-09-06
without a horizon stamp; those statements inherit this caveat.

⭐ **WHAT DOES NOT CHANGE:** row #9's *status* stays **OPEN** either way. The claim
"trajectories are smooth by construction" is not established by anything, and the
composed refcv5-v2 arm contains no smoothness lever. The magnitude is in doubt;
the gap is not.

**Registered as `D-CURV-84X-UNREPRODUCED`.**
