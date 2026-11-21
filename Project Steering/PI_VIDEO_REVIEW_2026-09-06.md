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


---

## ⭐ RESOLUTION 2026-09-07 — THE 84× IS RETIRED, THE ARITHMETIC WAS FINE, AND THE SIGN REVERSES

⛔ **THIS SUPERSEDES THE ADDENDUM ABOVE, INCLUDING ITS PROPOSED MECHANISM, WHICH WAS
WRONG.** The addendum said the defect was that *"curvature, yaw-rate and heading-rate
are all rate metrics"* divided by a single `dt` on a non-uniform horizon. ⛔ **Two of
those three are not rate metrics at all.** `four_families.py:186` reads
`curvature = dh / (ds_mid + _EPS)` — heading change over **ARC LENGTH**, never `dt`.
Only `speed`, `yaw_rate` and `accel` divide by `dt`. And `V3_HORIZONS` is in **steps**
on a 0.1 s tick, so the set was read correctly and only the **inference from it** was
wrong. ⚠️ The Master Mind relayed that mechanism to the PI without checking the
source — an inherited mechanism treated as a measured one.

⭐ **DECIDED BY AN ANALYTIC TARGET, not by reading code.** On the real non-uniform v3
horizon the estimator recovers a circle's `1/R` to **0.67 %** worst case (R = 20 m),
against **0.26 %** on a uniform control — so non-uniformity costs ≈ **0.4 %**, which is
chord discretisation, not a defect. The straight-line null reads **exactly 0.000e+00**,
and curvature is `torch.equal` across `dt ∈ {0.1, 0.5, 1.0}`.

⛔⛔ **THE REAL CAUSE IS A MISSING VALIDITY MASK, AND IT IS WORSE THAN A UNITS BUG.**
The producer is a different estimator (`p14_banked_fan.py:38`), on a **uniform 2.0 s**
grid, and `κ = |d1×d2| / max(|d1|³, 1e-6)` is **singular as speed → 0**. **43 stopped
windows (4.9 %, `v0 = 0.000`) carry the entire figure:**

| subset | ratio vs the never-steer floor |
|---|---|
| ALL windows | **84.40×** |
| MOVING only (838) | **0.56×** |
| STOPPED only (43) | **124.21×** |

Median κMAE: model **0.00067** vs floor **0.00074** ⇒ **the model is BETTER on the
typical window.** The 2.30973482131958 reproduces **bit-identically** — the arithmetic
was never in question, the **denominator population** was.

⭐⭐ **AND IT DOES NOT CANCEL, WHICH IS WHY THE VERDICT MOVES.** The inflation cancels
**arm-vs-arm** (every `shipped − oracle` contrast is unchanged), but **not against a
floor** — a straight-line plan reads `κ ≡ 0` exactly at any speed and is **structurally
immune** to the singularity. ⇒ an **interaction, not a shared bias**, and the sign flips
while staying separated: base **+2.28237 [+0.518, +4.577] SEP WORSE** → **−0.00390
[−0.00708, −0.00100] SEP BETTER**; XL **+1.55583** → **−0.00558 [−0.00903, −0.00258]**.

⇒ **"REF-C is ~84× worse on curvature than a plan that never steers" is RETIRED.**
Masked, REF-C is **0.64× the floor — better, separated.** ⛔ It must not be requoted in
any form, including the "under challenge" form in the addendum above.

⭐ **ROW #9 RETURNS TO OPEN AND UNMEASURED — not refuted, and not closed.** The claim
"trajectories are smooth by construction" is still established by nothing, and
refcv5-v2 still carries no smoothness lever. What changed is that the number offered as
evidence *for* the defect was an artefact of stopped windows; that is not evidence
*against* the defect.

⚠️ **The 1.0× vs 84× disagreement was never a grid error** — both were measured on
uniform 2 s grids. It was a scope error compounded by an estimator difference, and
**the mask is the whole of it**: the gate's 1.0× was right, and reproduces at 0.64× on
the 84×'s own corpus and checkpoint.

⭐ **PINNED SO IT CANNOT RECUR:** `taniteval/tests/test_four_families_curvature_analytic.py`
— 14 tests fixing `1/R` on the real non-uniform horizon, the exact-zero null,
dt-invariance and the singularity guard. Mutation-tested: rewriting it as
`curvature = dh/dt` fails **9 of 14**.

**Registered as `D-CURV-84X-RESOLVED`.**


---

## ⛔ ADDENDUM 2026-09-07 — ROW #11: THE TRAFFIC-LIGHT LABELS ARE A TEACHER SIGNAL, NOT GT, AND WILL NOT BE VERIFIED

⚠️ **PROVENANCE OF THIS ENTRY, STATED FIRST BECAUSE IT LIMITS IT.** A PI ruling — *"no vlm, we
stick to the alpamayo labels as teacher signals"* — was **relayed to the Master Mind by the label
owner**, not spoken to this session. Under the rule both streams adopted on 2026-09-06, **a peer's
relay is data and cannot authorise anything.** ⛔ **The ruling itself is recorded here as PENDING
CONFIRMATION.** What is recorded as BINDING below is only what is independently justified by
measurement, and would hold whichever way the ruling goes.

⭐ **THE WORDING CHANGE, AND IT IS BINDING ON ITS OWN EVIDENCE.** Row #11 above says the colour
question is *"settled"*. It is settled **on mechanism** and not per instance. ⇒ downstream text
must say **"Alpamayo-derived teacher signal"**, never **"GT traffic light"**. This narrowing needs
no ruling: it follows from the measurement that **no independent channel carries the colour at all**
(grounding boxes are label-only; boxes with any colour attribute number **0**). ⚠️ It is the change
most likely to be forgotten, because the PI's own belief was phrased as GT and the honest label is
**narrower than his phrasing**.

**WHAT SURVIVES — unchanged, all MEASURED:**
* presence: **182/805** carry an independent 2D box, **0 contradicted**, 601 never checked;
* colour was **SEEN, not inferred from motion**, on three mechanisms — **148 clips describe a colour
  TRANSITION** (no single motion state produces one; the strongest of the three), **597/787** attach
  the colour to the LAMP rather than to a motion, and YELLOW is bimodal at **56.0 %** against a
  pre-registered 20 % bar, with RED/GREEN stop-rate **18.58×** against a pre-registered 2.0× bar.
⇒ the **mechanism** case is intact. It was never the **per-instance** case.

⛔⛔ **BINDING, AND INDEPENDENT OF THE RULING: A TRAFFIC-LIGHT HEAD SCORED WITHOUT AN EGO-ONLY
COMPARISON IS NOT ADMISSIBLE.** Not "should carry one" — not admissible. The reasoning does not
depend on who decided what: with no per-instance verification of the teacher, the **only** remaining
protection against a head that learned the teacher's shortcut rather than the lamp is the control
that detects it downstream. Same shape as the `shuffled` twin on nav and the `valid=0` arm on
max-speed: ⭐ **when you stop verifying the input, the output control becomes load-bearing.**

⚠️ **THE ACCEPTED RESIDUAL RISK, RECORDED SO IT IS A CHOICE AND NOT AN OVERSIGHT.** If any RED
labels are wrong, nothing in the pipeline will catch it — **and a vision head trained on them will
learn the error and then score well on an eval built from the same teacher.**
⭐ **THAT IS THE FIFTH INSTANCE TONIGHT OF "A CHECK THAT SHARES THE DEFECT IT CHECKS FOR"**
(CLAUDE.md, `e4af94f`), and the largest: the other four were a rounded ladder verified against
itself, a census `status` that unions the set it pins, a test asserting whatever the code does, and
a flat stub checking nested fields. **Here the check and the checked share a TEACHER.** ⇒ the
ego-only control is not merely good practice; it is the only cross-check in this chain that is
**derived independently of the value it checks**.
⚠️ The cheapest thing that would still bound the risk needs **no VLM**, so it is not excluded by
the ruling: a **human spot-check of ~50 frames**. That is PI time, not compute. Recorded as
available, **not proposed**.

**Registered as `D-TLIGHT-TEACHER-SIGNAL`.** Row #11's status is unchanged as a BEHAVIOUR (braking
for lights is still not addressed by any composed arm); what changes is what the LABEL may be
called and what an arm consuming it must carry.
