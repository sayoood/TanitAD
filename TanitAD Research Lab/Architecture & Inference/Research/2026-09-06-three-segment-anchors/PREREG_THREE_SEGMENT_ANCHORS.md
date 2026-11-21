# PREREG — D-TRISEG: a three-segment anchor family, default OFF

**Author:** Arch+Inference, three-segment anchor agent · **Date:** 2026-09-06
**Status:** PRE-REGISTERED — written BEFORE any number about a three-segment arm
existed in this turn.
**Registers as:** `D-TRISEG-1` … `D-TRISEG-8` in `Project Steering/GOALS_AND_CLAIMS.md`.
**Predecessor:** `.../Research/2026-09-06-two-segment-anchors/` (commits `5e7001a`,
`8c9d01c`, `f23c3ef`), whose `RESULT.md` §10.2 wrote the bar this document executes.

⛔ **BOTH OUTCOMES ARE COMMITTED FOR EVERY BAR.** A bar that is missed is reported
**FAILED, as written**. RULE ZERO: a failure is not the end of the turn — the next
lever is named and run.

⛔ **PI CONSTRAINT HONOURED:** *"don't generate labels again."* Nothing here
rebuilds, regenerates or re-emits any label set. The anchor vocabulary is a
**candidate set**, and the target it is scored against is **geometric** — the
mechanism is unchanged from D-TWOSEG §1 and is not re-argued here.

---

## 0. THE INHERITED BAR, VERBATIM — and why it had to be written out

The predecessor deliberately did **not** ship the three-segment family, because
choosing it after seeing its exploratory numbers would be post-hoc selection. It
left this, in `RESULT.md` §10.2:

> **PRE-REGISTERED BAR for that arm, written now:** a 3-segment family PASSES if
> it beats the shipped 2-segment family on lane-change supply **and** costs less
> curvature on its own repicks, with the turn control still ≤ 0.01 m and R1/R2-style
> degenerate arms still reading exact zeros.

⭐ **That bar is REAL and it is BINDING. It is also AMBIGUOUS in four places, and
this document resolves each one WITHOUT weakening it** — where a clause admits a
harder and an easier reading, the **harder** reading is taken and both are
reported. **I am stating plainly that I had to write this: §0.1 lists what was
missing.**

### 0.1 The four ambiguities, and the resolution taken

| # | ambiguity | resolution — and it is the HARDER reading |
|---|---|---|
| **A1** | **The schedule `(t₁, t₂)` is not fixed.** The sentence names no split points, and the exploratory table offers seven. | ⛔ Fixed by **RULE S** (§2) — a stated rule, derived from the integrator's kinematics and the shipped family's own inherited grid, **not** from the exploratory ranking. |
| **A2** | **"the shipped 2-segment family"** could mean the **six** candidates that actually shipped, or the **two**-candidate `t_split = 2.0` pair the exploratory table used as its equal-cost reference. | ⛔ **The SIX.** The test family is **two** candidates, so this asks 2 to beat 6 — the harder reading. The equal-cost 2-vs-2 comparison is **reported beside it**, never as the verdict. |
| **A3** | **"costs less curvature on its own repicks"** compares two arms on **two different window sets** (each arm's own repicks), which can be moved by the sets rather than by the geometry. | ⛔ Decided **as written** (own-repicks), because that is what the sentence says. A **common-repick** read — the windows BOTH arms repick, where the two are the same object — is **reported beside it** as the robustness check. If the two disagree, the disagreement is the headline. |
| **A4** | **"R1/R2-style degenerate arms"** names no three-segment degeneracies, and the three-segment family has **more** of them than the two-segment one. | ⛔ **Four** are committed (§3, B4), not two, and each carries its own expectation **plus the reason the expectation is structural**. |

### 0.2 What the inherited sentence does NOT cover, and is added here

The inherited sentence is a **capability** bar. It says nothing about parity,
about the integrator limit, about the 2 s grid, about kinematic admissibility, or
about the artifact declaring what its extra columns mean. Those are **inherited
non-negotiables of this programme**, they are added as **B1, B2, B5, B6, B7, B8**,
and **the arm does not ship if any of them fails**, whatever the capability bar says.

---

## 1. The extension, defined before any number

A **three-segment** candidate holds `+a_lat` on `[0, t₁)`, `−a_lat` on `[t₁, t₂)`,
and **`0` thereafter**, with `a_lon` constant throughout. Everything else is the
incumbent's: the same integrator (`tanitad.models.kinematic.rollout_unicycle`),
the same `kappa = clamp(a_lat / max(v0, alat_v_floor)², ±kappa_cap)` map, the same
`alat_v_floor = 4.0`, `kappa_cap = 0.12`, `dt = 0.1`, and the same 8 slots
`[5,10,15,20,30,40,50,60]`.

**Parameterisation — a FOURTH control column.** `controls` becomes `[N, 4]`:

| col | name | units |
|---|---|---|
| 0 | `a_lon_ms2` | m/s² |
| 1 | `a_lat_ms2` | m/s² |
| 2 | `t1_s` | **s**, the time the lateral sign flips `+ → −` |
| 3 | `t2_s` | **s**, the time the lateral channel goes to **zero** |

⭐ **THE NESTING IS EXACT, AND IT IS WHAT MAKES THE OFF PATH PROVABLE.**

* `t₂ ≥ horizon_s` ⇒ the third segment is empty ⇒ **the two-segment candidate**
  with `t_split = t₁`.
* `t₁ ≥ horizon_s` (and hence `t₂ ≥ horizon_s`) ⇒ the sign is `+1` at every tick
  ⇒ **the incumbent constant candidate.**

⇒ the three-segment integrator **contains** both predecessors as special cases,
so the default-OFF claim is proved **by comparison**, not asserted (B2).

⛔ **WHY THIS IS A NEW SCHEDULE AND NOT A THIRD COLUMN'S VALUE.** A `[N, 3]`
file means `two_segment_alat_flip`. A `[N, 4]` file means
`three_segment_alat_pulse`. **An artifact that carries four columns and declares
nothing is REFUSED** (B8) — the 396 g / 0.31 g units incident, one further column
to the right. ⛔ **And there is NO CLI override for the schedule**, because —
unlike `control_units`, where the live `anchors.pt` is a real legacy file that
must keep loading — **no legacy four-column file exists anywhere.** A permitted
guess would not rescue a real artifact; it would **invent** one.

---

## 2. RULE S — how the schedule is fixed, stated before any number

⛔ **The exploratory sweep in the predecessor's `RESULT.md` §10.2 MAY NOT CHOOSE
THE SCHEDULE.** I have read that table — it is in the document I was handed, so
the honest guard is not a claim that I did not see it, but a rule whose output
can be **checked against** it (§2.5).

### S1 — `t₁` is INHERITED, not re-chosen
`t₁` is drawn from the **shipped two-segment split grid `{2.0, 3.0, 4.0}` s**,
which was pre-registered in D-TWOSEG and fixed from
`.../2026-09-06-selection-quality/raw/out_p1h_two_segment.json` (commit `56dc078`)
— **a probe that predates every three-segment number.** The constraint that
produced it is inherited with it: **every split at `t ≥ 2.0` s**, so the 2 s
structural zero (B5) survives.

### S2 — `t₂` is DERIVED from the integrator, not swept
Under `a_lon = 0` the incumbent's map gives a **constant** `|κ|` and hence a
constant yaw rate `ω = κ·v`. Heading returns to its initial value **iff the
`−` segment has the same duration as the `+` segment**:

```
t₂ − t₁ = t₁      ⟹      t₂ = 2·t₁
```

That is `anchor_twoseg.net_yaw_zero_split_s` **generalised — and generalising it
is precisely what the third segment is for.**

⚠️ **The predecessor's own caveat, quoted and SCOPED, not ignored.** That helper
carries a measured warning that the net-yaw-zero condition is *not* where the
two-segment gain lives (a single split at `3.0 = horizon/2` recovers **0.0304** of
the **0.1645** three splits recover), because a real lane change is executed on a
**curving** road. ⛔ **That warning is a fact about the TWO-segment family, where
net-yaw-zero FORCES the counter-steer to consume the entire remaining horizon —
which `RESULT.md` §9 identified as the defect (72.04 % of the degradation in the
3–6 s tail).** The third segment is exactly what decouples *"return the heading"*
from *"spend the horizon"*, so the same physics yields a different, non-degenerate
schedule here. ⭐ **This is a PREDICTION of the rule and it is allowed to be
wrong.** If a curving-road corpus punishes it, the bar FAILS as written.

### S3 — drop any schedule whose third segment is empty
`t₂ ≥ horizon_s` makes the candidate a **two-segment candidate already in the
shipped bank**, so it is a duplicate, not a new family member:

| `t₁` | `t₂ = 2·t₁` | verdict |
|---|---|---|
| **2.0** | **4.0** | ⭐ **KEPT** — third segment spans 4.0–6.0 s |
| 3.0 | 6.0 | ⛔ dropped — `= horizon`, IS the shipped 2-seg `t_split = 3.0` |
| 4.0 | 8.0 | ⛔ dropped — `> horizon`, IS the shipped 2-seg `t_split = 4.0` |

⇒ **exactly one schedule survives: `t₁ = 2.0 s`, `t₂ = 4.0 s`.**

### S4 — magnitudes are INHERITED
`a_lat ∈ {−0.75, +0.75}` m/s², `a_lon = 0`. MEASURED at commit `56dc078`: eight
magnitudes buy **+0.0009 m** over two. **The split point is the lever; the
magnitude is not.** ⭐ Both values are members of the incumbent's **own** `a_lat`
grid `{0, ±0.75, ±1.5, ±2.25, ±3.0}` (MEASURED, §5), so the extension introduces
no new control alphabet — only a new **schedule** over the existing one.

### ⇒ THE FAMILY IS FIXED BEFORE ANY NUMBER
**2 candidates:** `(a_lon = 0, a_lat = ±0.75, t₁ = 2.0 s, t₂ = 4.0 s)`.
Bank cost **2 / 117 = +1.71 %** — **a third** of the two-segment family's +5.13 %.

### 2.5 The check that RULE S is not the exploratory table in disguise
⭐ **The rule lands on `2.0 → 4.0`. In the predecessor's exploratory table that
row is the ARGMAX OF NOTHING** — it is beaten on lane-change gain by `2.0 → 3.5`
and `1.5 → 3.0`, on all-window gain by `2.0 → 3.0`, on curvature cost by
`2.0 → 3.0` and `2.5 → 4.0`, and on cross-track by `2.0 → 3.5`. ⛔ **A rule that
selects a row which is best on no column cannot be that table's optimum wearing a
justification.** Had I been selecting post-hoc I would have taken `2.0 → 3.0`, and
the brief that sent me here named exactly that row as the one not to take.

⚠️ The exploratory rows are **re-derived from scratch here and reported in full as
CONTEXT** (§7 of the RESULT). ⛔ **None of them may move the shipped default**, and
the verdict is computed only for the RULE S family.

### 2.6 What happens on PASS — pre-committed, so the ship is not a choice made later
On PASS the family is delivered as a **builder option** (`--three-segment`),
**default OFF**, exactly as the two-segment family was; the default bank is
**unchanged**. ⛔ **Which family the composed refcv5 arm carries is a decision for
the composed arm's own pre-registration, not this one.** This document does not
swap a default.

---

## 3. The bars — both outcomes committed

### ⭐ B0 — THE INHERITED CAPABILITY BAR, made operational
Let `A0 = base117`, `A1 = A0 + the shipped 6 two-segment` (123), `A2 = A0 + the 2
RULE S three-segment` (119). "LC supply gain" is the drop in best-in-fan 6 s ADE
against the recorded ego path, on **strict lane-change** windows, vs `A0`.

* **B0a — beats the shipped family on lane-change supply.**
  **PASS** `gain_LC(A2) > gain_LC(A1)`, strictly. **FAIL** `≤`.
* **B0b — costs less curvature on its own repicks.** For arm `X`, let
  `R_X = {LC windows where X's argmin index ≥ 117}` and
  `Δcurv_X = mean curv-MAE(X picked | R_X) − mean curv-MAE(A0 picked | R_X)`.
  **PASS** `Δcurv_{A2} < Δcurv_{A1}`. **FAIL** `≥`.
  ⚠️ The **common-repick** read (`R_{A1} ∩ R_{A2}`) is reported beside it; a
  disagreement between the two is reported as the headline, not smoothed over.
* **B0c — the turn control holds.** `|gain_TURN(A2)| ≤ 0.01 m`.
  **FAIL** otherwise ⇒ the scorer, not the geometry, is doing the work.
* **B0d — the degenerate arms read exact zeros** (B4 below).

⛔ **B0 PASSES only if all four clauses pass.** A three-of-four is a FAIL.

### B1 — DEFAULT-OFF PARITY. ⛔ Hard gate.
The builder with `--three-segment` **absent** must emit a bank whose `anchors` and
`controls` are **sha256-identical** to refcv4b's live bank
(`anchors 51f930dc…a66df`, `controls b072f4c0…89664`, the values independently
recorded in `restamp_refcv4b_anchors.py` **and** re-derivable from
`ckpt_40284_FINAL.pt`), and whose `controls` is `[117, 2]` — **not** `[117, 3]`
and **not** `[117, 4]`. With `--two-segment` but **not** `--three-segment` it must
emit the shipped `[123, 3]` bank, byte-for-byte.
* **FAIL** any difference ⇒ the extension is withdrawn until it is default-inert.

### B2 — THE LIMIT IS BIT-IDENTICAL, PROVED BY COMPARISON, WITH A MUTATION CONTROL. ⛔ Hard gate.
Over ≥ 8 speeds spanning 0–36 m/s and the full 60-tick roll:
* **L1** 117 incumbent `[N,2]` controls widened to `[N,4]` with `t₁ = t₂ = horizon_s`,
  rolled through the four-column integrator, must be `torch.equal` to the
  incumbent 2-column roll.
* **L2** the shipped 123 `[N,3]` bank widened to `[N,4]` with `t₂ = horizon_s` must
  be `torch.equal` to the 3-column two-segment roll.
* **L3 — MUTATION CONTROL.** The same 117 controls rolled with the RULE S
  schedule `(2.0, 4.0)` must **differ materially**: max abs difference **> 1.0 m**,
  and the measured value is reported.
* **PASS** L1 and L2 exactly equal **and** L3 differs. **FAIL** any bit differs in
  L1/L2, **or** L3 does not differ — an equality that cannot fail proves nothing.

### B3 — SUPPLY (the capability bar in its own right). Pre-registered threshold.
On the 6 s grid over the 141-clip eval corpus at stride 1:
* **PASS** `A2` improves the strict lane-change supply ceiling by **≥ 0.10 m**
  AND moves the **TURN control** by **≤ 0.01 m** (either direction).
* **FAIL** otherwise. ⚠️ Threshold **inherited unchanged** from D-TWOSEG B3, not
  re-chosen; a threshold re-set after seeing a number is not a threshold.

### B4 — DELIBERATE REGRESSION, four arms. ⛔ If the gate cannot FAIL these, a PASS means nothing.
Each expectation is committed here, with the **reason it is structural**.
⭐ The two apparatus facts the expectations rest on were established **before this
document was finished**, by `raw/p0_bank_facts.py` reading only the incumbent
checkpoint: the bank contains `(a_lon, a_lat) = (0, 0)` **at index 67**, and its
`a_lat` grid is **symmetric with 0 of 117 rows missing their `(a_lon, −a_lat)`
mirror**.

| arm | schedule | why it is structural | committed expectation |
|---|---|---|---|
| **R1** | `t₁ = t₂ = 6.0` | sign is `+1` at every tick ⇒ the two candidates are **duplicates of existing constant arcs** | **gain = 0.0000000000 EXACTLY**, 0/0 windows improved |
| **R2** | `t₁ = 0.0, t₂ = 6.0` | sign is `−1` at every tick ⇒ constant arcs of the **opposite sign**, present because the grid is mirror-complete | **gain = 0.0000000000 EXACTLY**, 0/0 improved |
| **R4** | `t₁ = t₂ = 0.0` | sign is `0` at every tick ⇒ straight lines at `a_lon = 0` ⇒ **candidate 67**, already in the bank | **gain = 0.0000000000 EXACTLY**, 0/0 improved |
| **R3** *(the GRADING arm)* | `t₁ = 2.0, t₂ = 6.0` | third segment empty ⇒ this **IS** the two-segment `t_split = 2.0` pair | must reproduce that pair's LC gain **BIT-IDENTICALLY** (`==`, not a tolerance) |

* **FAIL** if R1, R2 or R4 reads a non-zero gain (the scorer credits the wrong
  thing), **or** if R3 is not exactly equal (the two families are not nested).
⭐ R3 is strictly stronger than the predecessor's `±0.005` grading arm: the
geometries are the *same object*, so anything but exact equality is a defect.

### B5 — STRUCTURAL ZERO ON THE 2 s GRID. Control at a KNOWN value.
`t₁ = 2.0 s` and slot 3 **is** 2.0 s, so no RULE S candidate can differ from its
constant counterpart at slots 0–3. On the 2 s grid the extension must change every
one of the four families by **EXACTLY 0.000000** and **0** windows may repick.
* **FAIL** any non-zero ⇒ the roll is wrong somewhere.
⛔ **This is also the honest scope statement: a capability gained here CANNOT
appear in `ade_0_2s`.** Reported as a limitation, not buried.

### B6 — KINEMATIC ADMISSIBILITY + VACUITY GATE.
* Every new candidate's peak `|a| = sqrt(a_lon² + a_lat_realised²)` must sit
  **inside** the incumbent envelope, with **0** violations of `μ = 0.7` and
  `kappa_cap` never reached (a clamped candidate is a different candidate than the
  one declared). The **realised** `a_lat` is scored, never the declared one.
* ⛔ **THE MANOEUVRE RATE IS REPORTED BESIDE THE FEASIBILITY NUMBER.** A
  zero-violation result bought by declining the manoeuvre is not a safety result
  (MEASURED precedent: a `turn_left` recall of exactly 0.0000).
* ⛔ **AND THE VACUITY GATE ON THE INCUMBENT'S CURVATURE IS RE-REPORTED**: the
  straight-line floor sits beside every LATERAL number, because MEASURED the
  incumbent's "good" curvature on lane-change windows is **+2.1 % over a plan that
  never steers**.
* **PASS** 0 violations **AND** a non-zero lane-change manoeuvre rate.

### B7 — SUPERVISION RATE.
The fraction of strict lane-change windows whose geometric `a_star` lands on a
three-segment candidate, on the extended bank.
* **PASS** **≥ 20 %** (threshold **inherited unchanged** from D-TWOSEG B7), with
  the all-window rate reported beside it — a family capturing most windows would be
  a vocabulary takeover, not a fix.
* **FAIL** < 20 % ⇒ the candidates exist but the training signal rarely points there.
⛔ **B7 measures SUPERVISION, not EXECUTION** (§6).

### B8 — THE ARTIFACT DECLARES ITS SCHEDULE, AND THERE IS NO OVERRIDE. ⛔ Hard gate.
* `read_anchor_artifact` must raise **`AnchorScheduleMissing`** on a `[N, 4]`
  `controls` that declares no `control_schedule`, and **`AnchorScheduleConflict`**
  when a declared schedule's column count disagrees with the tensor's.
* ⛔ **No keyword, flag or environment variable may suppress either**, and a test
  must assert that no such escape exists. **No legacy four-column file exists**, so
  a permitted guess could not rescue a real artifact — it could only invent one.
* **PASS** both refusals fire and no override path exists. **FAIL** otherwise.

---

## 4. Estimator and evidence discipline

* ⭐ **THE VARIANCE, NAMED CORRECTLY — AND THEREFORE NO INTERVAL.** Every number
  here is **model-free and deterministic** given the banked dump and the recorded
  poses: no training draw (`H-ESTIM-SEED-1` does not apply — no arm is trained), no
  inference sampling (no planner samples — nothing is planned), no episode
  resampling (**every** scoreable window is scored, not a subsample). ⇒ **there is
  no population being estimated, so NO CI IS QUOTED.** ⛔ Manufacturing an interval
  here would be an interval that answers no question at all.
* Every fraction carries its **n** and its **corpus**; a percentage alone is
  inadmissible.
* **Four families, never pooled, never ADE alone**, and ⛔ **in BOTH directions** —
  what the lever buys **and** what it pays. LATERAL is read on curvature MAE **with
  the straight-line floor beside it**.
* ⛔ **A SUPPLY ceiling may never be quoted beside an achievement**
  (`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`).
* ⚠️ **Attribution guard.** No refinement effect is attributed here; nothing in
  this work touches the refinement path. The α = 0.25 output-smoothing credit
  belongs to undoing the decoder's refinement and is not this lever's.
* ⚠️ 6 s-grid curvature numbers are **not** comparable to the 2 s re-rolled arm
  floors `ha0` 0.006841 / `os` 0.008024. Different grid, different object.
* Numbers are re-derived here from the checkpoint and the dump. Where a predecessor
  number is repeated it is stamped **INHERITED** and named as not re-verified.

---

## 5. What is BLOCKED, named in advance

1. ⛔ **Executed lane-change rate of a trained model.** refcv4b has **117 logits**
   and structurally cannot rank a 118th–119th candidate. **Unblocked by** a retrain
   — a tiny-rig v7 arm (~17 min on Thor) or an A40 slot. ⛔ **The A40 is FREE and
   RESERVED for the composed refcv5 arm and is NOT taken by this work.**
2. ⛔ **The tactical head's three dead lane-change classes.** A **different head**
   from the one this work feeds; its repair **is** a label change and is
   **forbidden by the PI**. Blocked on a PI decision, not on compute.
3. ⛔ **The consumer.** `refc.py::AnchoredDiffusionDecoder.roll_bank` refuses a
   `[N, 4]` bank on its shape checks — loudly, which is correct. `refc.py` is a
   sibling's file this turn, so `anchor_twoseg.roll_bank` is extended as the
   **drop-in reference + test** and the decoder-side read of columns 2–3 is a
   **named hand-off with an exact diff**, not an omission.

---

## 6. Scope of the claim, stated before the numbers

⛔ **WHAT CHANGES IS THE SUPERVISION, NOT THE BEHAVIOUR — AND THIS BELONGS IN THE
HEADLINE, NOT A FOOTNOTE.** refcv4b has **117 logits**; this extension makes the
bank 119. **The model structurally cannot select these candidates.** What moves is
where the geometric `a_star` **points** during training. A trained execution rate
requires a retrain and is **blocked** (§5.1). Nothing here may be read as a
behavioural result.

⛔ **THE DOMINANT SELECTION DEFECT IS LONGITUDINAL, AND THIS IS NOT A FIX FOR IT.**
*(INHERITED, commit `a3b232b`, not re-verified here:* **4,350 / 24,114** *windows
carry the wrong longitudinal manoeuvre while the fan held a correct candidate, and
an* `a_lon` *oracle recovers* **74.2 %** *of the regret against* `a_lat`'s
**20.9 %**.*)* ⇒ **a lateral vocabulary fix reaches at most the ~20.9 % lateral
share, and within it only the lane-change sub-population** (**6.97 %** of windows,
INHERITED from D-TWOSEG and re-derived here). **It is not presented as anything
larger.**

⛔ **NO LABEL GENERATION** (PI-forbidden), **no Alpamayo re-ask**, and the nav
command is an **INPUT simulating the vehicle's nav system**, never a training
signal — the phrases *"oracle nav"* and *"deployment gap"* are not used.

---

## 7. Pre-committed reporting

Whatever the outcome, the RESULT reports: the inherited bar verbatim and the fact
that it had to be written out; **PASS or FAIL per bar as written**; the OFF path's
bit-identity **with its mutation control**; the four families **in both
directions**; feasibility **with the manoeuvre rate and the vacuity gate beside
it**; the full re-derived exploratory sweep as **context that may not move the
default**; and one line answering whether the three-segment schedule beats the
shipped two-segment pair on its own pre-registered terms and what share of the
selection defect it can possibly touch.
