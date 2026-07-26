# PRE-REGISTRATION — the Rung 1 PLANNER arm, and the 30 owed control rows

**Written 2026-07-27 (Europe/Berlin; pods log UTC), BEFORE any number in this folder existed.**
This file is **not edited after the fact**; every deviation is recorded as a numbered amendment in
`RUNG1_PLANNER_AND_CONTROLS.md` §A.

**Host:** pod2 only (A40). `MEASURED` idle before launch: `0 MiB, 0 %`; Python **3.11.10**; disk
verified with a **real 2 GB `dd` write** (2,097,152,000 bytes in full, 4.0 GB/s), never `df`; `/root`
holding 350 MB of which `/root/tbr1` is 143 MB. ⛔ **pod1** (v2corpus training), **pod3**
(situation-classifier build) and the **eval pod** (trafficsim + wheelbase) are never connected to.
Kill by **explicit PID** only.

**Estimator, everywhere in this folder:** **paired episode-cluster bootstrap** — `taniteval/ci.py`,
**B = 2000, seed 0**, resampling unit = **episode cluster** (S1: AlpaSim scene). ⛔
`overlapping_holdout_se` / `_jack` appears nowhere; it biases the interval **1.107–3.100×** *and*
the point estimate (**−6.67 %…+11.69 %**, up to ×−4.15 with sign flips).

**Evidence stamps:** class `MEASURED` / `PUBLISHED` / `INHERITED` / `ESTIMATED` / `HYPOTHESIS`
**and** tier `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE` on every number.

🔒 PhysicalAI-AV is gated-confidential: **no clip UUID and no raw content** appears in this folder.
(⚠️ Exception noted in advance: the S1 slice is **AlpaSim**, not PhysicalAI, and its scene ids are
already published in `2026-07-26-4brain-gates/s1_decision_points.json`. Even so, no scene id is
reproduced here — rows are referenced by index.)

**Priority order, fixed here so a killed agent still yields value:**
**(1)** the 30 owed control rows — they can only *remove* results, so they go first;
**(2)** the Rung 1 planner arm; **(3)** the downstream chains. Bank each stage before the next.

---

# PART A — THE 30 OWED CONTROL ROWS

## A.0 The frozen list is frozen

The list is `2026-07-26-control-readjudication/raw/frozen_list.json` — **75 rows**, of which the
predecessor re-adjudicated 45 and left **30 owed**. The 30 are exactly the four families the
predecessor recorded as *"power available to me: none"* and could not re-power because pod2 held the
600-episode val and was training:

| family | rows | frozen-list indices (1-based, as published in §1.1) |
|---|---:|---|
| `S1-BLINDvsMAJORITY` | 3 | 23, 24, 25 |
| `DEAD-CONTROL` | 3 | 49, 50, 51 |
| `CE-CONTROL` | 14 | 52 – 65 |
| `CHANCE-BASELINE` | 10 | 66 – 75 |
| **total** | **30** | |

⛔ **Nothing is added to this list.** Any control-type null discovered during this work goes into a
**clearly-marked SECOND list** (§A.6) and is ranked *owed*, never merged into the 30.

## A.1 Why these rows are the dangerous half

For a **firewall / leakage / shuffle / circularity / dead / chance** control the **null is the
DESIRED verdict**. So *"not separated"* is **not a refuted leak — it is a leak we did not have the
power to see.** The predecessor's re-adjudication of the other 45 found **4 of 12** re-adjudicable
verdicts flipping *"control passed" → "control FAILED"*, **3 with sign flips**. The prior here is
therefore **not** that these hold.

## A.2 The method, fixed in advance — five binding rules

1. **M9 — read the WHOLE artifact, not the matching node.** For every one of the 30 rows, the entire
   source JSON is enumerated key-by-key and every sibling node that measures the same or a
   higher-powered version of the same contrast is listed. *(This is how 61 of 75 rows were missed:
   the S3 firewall's answer sat one JSON key away — `.non_parity` beside `.parity` — and a
   `separated:false` sweep can never reach the second.)*
2. **M8 — every row gets an MDE stated against the effect it exists to catch.** `MDE` = the row's own
   95 % half-width; an effect must **exceed** it to separate. Where the metric is **bounded**, the
   maximum attainable effect is also computed, and `MDE / max_possible_effect` is a **proof**, not a
   projection. *(The killer case: MDE 0.1985 against a real leak of +0.0711 — 2.8× too blunt. Another
   at 222 % of the maximum physically possible effect.)*
3. **M4 — use the packaged instrument.** `taniteval/taniteval/blind_baseline.py` and
   `taniteval/taniteval/ci.py` are used as shipped. ⛔ **No firewall or interval is re-implemented** —
   two independent re-implementations produced the nulls that were overturned (R-E).
4. **M5 — trace the downstream chain for every flip.** If a control fails or is void, the claims that
   must be withdrawn are **named**, with the file and line that carries them.
5. **M6 — do not project.** Point estimates move a **median 75 %, max 649 %**, with sign flips, and
   the harvest's `would_flip` scored **4/8**. Any row I cannot measure is reported as **owed with a
   cost**, never as a forecast.

## A.3 ⭐ The verdict vocabulary — fixed BEFORE any row was re-scored

Every one of the 30 rows terminates in exactly one of these. **Three of the six are failing
verdicts**, and two of those (`CONTROL FAILS`, `VOID-BY-MISMATCH`) *remove* program results.

| verdict | condition | what it means |
|---|---|---|
| ✅ `HOLDS (POWERED)` | re-measured at higher n, **still not separated**, **and** `MDE` < the effect it guards against | the control did its job and we now know it could have fired |
| 🔴 `CONTROL FAILS` | re-measured at higher n and **separated** | **the bad outcome.** A leak/effect that should have been absent is present ⇒ the downstream claim is withdrawn by name |
| ⛔ `VOID` | `MDE ≥ max_possible_effect` ⇒ `can_fire = false` | **zero evidence at any observed value.** Neither confirmed nor refuted. Only a larger n can ever give it content |
| ⛔ `VOID-BY-MISMATCH` | the published interval is computed on a **different statistic** than the published point estimate | the interval does not interval its own headline ⇒ the verdict is uninterpretable as published |
| ⚠️ `UNDER-POWERED (OWED)` | `MDE` > the effect it guards against but `< max_possible_effect` | not void, but **it was never run at a resolution that could see the thing** |
| ⚠️ `OWED — NOT RE-POWERABLE HERE` | re-powering needs data/compute outside this task's mandate | ranked, costed, and handed on. **Explicitly not "checked"** |

**FAILING VALUES, and the proof they can fire.** A verdict rule that cannot return a bad value is a
comment (C13). Registered here:
* `CONTROL FAILS` **fired 4/12 times** in the predecessor run on real rows — the rule discriminates,
  and I am running the same rule.
* `VOID` fired once (`firewall.H`, MDE 222 % of max) — attainable.
* The **null direction is attainable too**: 8 of the 12 predecessor rows returned *holds*.
* ⭐ **Harness validation is required in BOTH directions on every instrument I run** (§A.5): a
  fidelity check that reproduces a committed number, **and** a deliberately failing input that MUST
  separate. An instrument that passes only the fidelity direction is not validated.

## A.4 Per-family plan, and what would make it fail

| family | what I will do | ⛔ what makes it return a FAILING value |
|---|---|---|
| `S1-BLINDvsMAJORITY` (3) | n is bounded by 20 AlpaSim scenes and cannot grow here. So: (a) **reproduce** `S1_RESULTS.json` from its own driver; (b) recompute `blind − majority` **paired on the HEADLINE blind statistic**; (c) compute the **never-intervalled `blind − chance`** contrast — the leak-relevant one; (d) emit `mde` / `max_possible_effect` / `can_fire` per §6 of the predecessor's report | `blind − chance` **separated** ⇒ 🔴 CONTROL FAILS ⇒ the S1 admissibility sentence is withdrawn on *evidence*, not only on a power argument. `can_fire = false` ⇒ ⛔ VOID. Headline/interval statistic mismatch ⇒ ⛔ VOID-BY-MISMATCH |
| `DEAD-CONTROL` (3) | re-power on pod2 from n = 12/40 to the largest episode count the instrument supports, **same script, same conditions, no code change** | the dead perturbation's paired effect becoming **separated** ⇒ 🔴 CONTROL FAILS ⇒ the envelope claim it licenses is withdrawn |
| `CE-CONTROL` (14) | re-power from n = 40 val episodes on pod2 if and only if the cached-feature pipeline and the v4 checkpoint are both present; otherwise MDE-adjudicate and rank owed with a measured cost | the counterfactual-equal arm becoming **separated from as-trained** ⇒ 🔴 CONTROL FAILS ⇒ the "the measured loss is the selector's" attribution is withdrawn |
| `CHANCE-BASELINE` (10) | **n = 322 is already the largest available** and is limited by *labelled clips*, not episodes — the ×3.4 episode-shrinkage argument does not transfer. Adjudicate by MDE + max-possible-effect and by M9 sibling search | `MDE ≥ max_possible_effect` ⇒ ⛔ VOID |

## A.5 The harness self-tests, registered in advance

| instrument | fidelity direction (must reproduce) | ⛔ failing direction (must fire) |
|---|---|---|
| `s1_slice.py` re-run | every field of the committed `S1_RESULTS.json` reproduces exactly | an **echo** target (target copied into the feature row) must drive `acc_blind → 1.0` and the verdict to REFUSED |
| `taniteval.blind_baseline` (packaged) | on the S1 data it must land within a stated tolerance of the S1 driver's own blind accuracy | on an echo context it must return `CIRCULAR`; on a random context it must return `CLEAN` |
| `taniteval.ci.paired_episode_cluster_bootstrap` | reproduces the committed `blind_vs_majority_paired` intervals to 4 dp | on `a = b` it must return a not-separated interval containing 0; on `a = b + 1` it must separate |
| the MDE / `can_fire` emitter | on `firewall.H` it must return `can_fire = false` (the one case where truth is already known) | on a control with MDE far below the guarded effect it must return `can_fire = true` |

## A.6 The second list

Per BOOST §7.1 nothing may be added to the primary list after a result is seen. Any control-type null
I discover goes to a **second list**, clearly marked, ranked *owed*, and it may **not** change any
verdict in Part A.

---

# PART B — THE RUNG 1 PLANNER ARM (`wp_to_control`)

## B.0 What is being scored, and against what

Rung 1 swept the **action-filter** axis and CONFIRMED. It explicitly did **not** run the ladder's
**R1 PLANNER** row, and recorded a prediction for it instead (`TBLIND_RUNG1.md` §7.2), which I quote
verbatim as the thing to be scored:

> *"v1's tactical planner will land between 6 and 12 s of deployable `T_blind` — comfortably above
> the 2.5 s kinematic-inverse baseline and at or below the 11.6 s filter result — because its
> advantage over the inverse is a lower longitudinal gain, which is the same lever measured here. If
> it lands BELOW 6.4 s, the planner is adding a failure mode the gain argument does not explain and
> that is the finding."*

Its stated reasoning: `closedloop.wp_to_control` accelerates with `(v_target − v)/SPEED_TC`,
`SPEED_TC = 0.5 s` (`closedloop.py:156,180`) instead of the inverse's `(v − v_prev)/0.1 s` — a **5×
lower longitudinal gain** — and an EMA at β = 0.8 at 10 Hz has time constant `DT/(1−β) = 0.5 s`,
**numerically `SPEED_TC`**, delivering **64 steps (6.4 s)**.

## B.1 ⭐ The mechanism is already known and it shapes the arm

`MEASURED`, `TBLIND_RUNG1.md` §4.2/§4.3, and it is **not** drift: the own acceleration command is a
**near-zero-mean, clamp-saturating oscillation** — mean |accel| **2.058 m/s²** over the first 0.5 s
against **0.539** for the same inverse fed *true* motion (3.8×), at the ±3 clamp **46.4 %** of the
time against **0.53 %** (87×), bias/amplitude **0.0162**. **Steering is innocent** (`steer_clip`
moved nothing at any setting; `accel_clip` moved everything).

⚠️ **And the textbook fix is catastrophic here:** reducing the action-update frequency gives **9
steps against a 25-step baseline**, because **holding a *sample* of a zero-mean oscillation removes
its cancellation**. **Amplitude-shrinking helps; sample-and-hold destroys.**

**Consequences for the design, adopted before running:**
* the arm is scored on **amplitude statistics as well as `T_blind`** — mean |accel|, clamp-saturation
  fraction, sign-flip rate, and bias/amplitude — because the prediction is a *gain* argument and a
  `T_blind` in range with unchanged amplitude would confirm the number while refuting the reason;
* ⛔ **no sample-and-hold / cadence knob is placed on the planner arm.** `wp_to_control` at a reduced
  re-plan cadence is exactly the `every` family that was measured to be destructive, and running it
  as a "fix" would repeat a refuted intervention;
* the planner's **steering** channel is expected to be a null lever, so no steering variant is run.

## B.2 The arms — fixed here, no arm added after a result is seen

| arm | eligible? | what it is |
|---|---|---|
| `a_planner` / `b_planner` | ✅ **ELIGIBLE** — sets the verdict | the **PRIMARY**. `w_look = tactical_policy(win_s, strategic_policy(win_s, nav=follow)["ctx"])["waypoints"][5]`, then `steer, accel = wp_to_control(w_look, v)`. `v` is tracked by the controller's own integration `v ← v + accel·DT`, exactly `closedloop.closed_loop_rollout`. **The `v0` action channel is held CONSTANT**, matching every Rung 1 arm, so the contrast against `own_kinematic` is a pure action-source contrast |
| `a_planner_vupd` | ✅ ELIGIBLE (reported, not primary) | as above **but** `v0 = v_tracked / SPEED_SCALE`, which is what `closedloop.build_action` actually deploys. Separated from the primary because Rung 1 measured the *decoded*-speed version to be catastrophic (`de@2s` 1.82 → 23.94) and the two must not be conflated |
| `a_planner_vdec` | ✅ ELIGIBLE (reported, not primary) | as primary but `v` fed to `wp_to_control` is the model's **decoded** speed rather than the controller's integrated one — isolates which speed estimate the controller is standing on |
| `a_planner_gtlook` | ⛔ **DIAGNOSTIC — privileged, may never be quoted as deployable** | `wp_to_control` fed the **TRUE** 0.5 s lookahead waypoint, transformed into the current imagined ego frame. The planner's analogue of `gtkin`: it exonerates or convicts the **controller** independently of the tactical head |
| anchors `a/b_…__own__roSTR`, `a/b_…__hold__roSTR` | ⛔ anchors | re-rolled to gate window-set identity against **both** committed dumps |

**Comparators.** Every `T_blind` uses a comparator matched in the **readout** (`str`, k = 20) **and**
in the **action source** — i.e. the `frozen_last` arm driven by the *same* planner. Without that the
number is not a `T_blind`.

## B.3 ⚠️ The confound, named before the result

The planner changes **two** things at once versus the kinematic inverse: the **controller gains**
(§B.0) **and** the **source of the intent** (the trained tactical head vs the step readout's own
decoded Δpose). §7.2's prediction is framed on the gains alone. This is stated as a limitation and is
attacked two ways, both registered here:
* `ema0.8` (**64 steps**, `INHERITED-MEASURED` from Rung 1) is the *gain-lowered kinematic inverse* —
  the same 0.5 s time constant with the **same** intent source. It is the reference the prediction
  itself names, so `planner` vs `ema0.8` is the gains-matched contrast;
* `planner_gtlook` holds the controller fixed and replaces the intent with a perfect one.

## B.4 ⭐ The verdict buckets — fixed before any number existed

Primary statistic: **deployable `T_blind`** in steps, `t_blind()` imported verbatim from the ladder's
`tb_rung0.py` (not re-implemented), paired episode-cluster bootstrap, matched comparator.

| bucket | rule |
|---|---|
| ✅ **PREDICTION CONFIRMED** | `a_planner` `T_blind` ∈ **[60, 120] steps** (6.0 – 12.0 s) |
| ⚠️ **PREDICTION EXCEEDED** | `T_blind` > 120 steps — the planner beats the whole zero-training filter axis |
| ⛔ **PREDICTION REFUTED — LOW** | `T_blind` < 60 steps. §7.2's own words apply: *"the planner is adding a failure mode the gain argument does not explain and that is the finding"* |
| — | **`T_blind` < 64 steps** additionally refutes the *specific* `ema0.8`-equivalence reasoning, and is reported separately from the [60,120] bucket boundary so the two are never merged |

⛔ **THE FAILING VALUE, and it is reachable.** The primary statistic's failing value is **1 step**
(0.1 s), returned when the first evaluable horizon already fails — not 0. In Rung 1 **6 of 18
eligible arms landed in the failing bucket** and four of them returned **9 steps**, far *below* the
25-step baseline. A planner arm that destabilises the loop will land there and the rule will say so.
**A `T_blind` below the 25-step `own_kinematic` baseline is an available outcome and would be
reported as the headline.**

**Mechanism sub-verdict** (separate from the number, per §B.1):

| | rule |
|---|---|
| ✅ gain argument **CONFIRMED** | `T_blind` ∈ [60,120] **and** the planner's mean \|accel\| over the first 0.5 s is **materially below** `own`'s 2.058 m/s² **and** its clamp-saturation is materially below 46.4 % |
| ⛔ gain argument **REFUTED** | the number lands in range **while the amplitude statistics do not move** — the right answer for the wrong reason, and it is reported as a refutation |

## B.5 ⛔ The gates — nothing below them is read until they pass

1. **Window-set identity, BLOCKING.** 599 windows, 596 episode clusters, `eid` and `t0` ordering
   identical to **both** committed dumps; the four anchor arms must reproduce their committed dense
   `de` within **1e-4 m** (Rung 0b measured 3.05e-05 m between two encode passes).
2. **Fidelity against the ladder, BLOCKING on levels.** `de@2s` and `ade_0_2s` for `own` and `hold`
   must reproduce (1.8165 / 0.6718 and 0.8710 / 0.3351). The **integer** `T_blind` reproduction
   (25 / 115) is declared **reported, non-blocking** in advance — a step count is a threshold
   crossing — exactly as Rung 1 declared it, so the latitude cannot be claimed post-hoc.
3. **Plumbing self-test, BOTH directions, at full K.** ⭐ The planner code path must be proved to be
   (a) *not a no-op*: the planner arm must differ from the `own` arm and from the `hold` arm by a
   non-trivial margin; and (b) *actually the deployed controller*: `wp_to_control` is **imported from
   `taniteval.closedloop`**, never copied, and a CPU fixture must show that feeding it the model's
   own decoded 0.5 s displacement reproduces `closed_loop_rollout`'s own first control exactly.
   ⛔ **An arm identical to `own` or to `hold` fails the gate and the run is void.**
4. **Diagnostic vacuity audit.** Any statistic whose failing value is unreachable is **not emitted**.

## B.6 M10 — the tier split is KEPT

The **`T_blind` number** is an extension against a *frozen percept*. The **capability claim** is
separate and rests only on the two comparator-free statistics, which no readout or filter mismatch
can enter because the constant-velocity floor is pure kinematics:

* **beats-CV** — number of horizons where the arm is separated-better than constant velocity, with
  the contiguous interval in seconds;
* **`T_useful@1m`** — largest horizon with mean `de` below 1 m, contiguous from step 1 (failing value
  **0.0**, reachable).

**Both are reported explicitly for the planner arm, and a capability CONFIRM requires beats-CV > 0
with a separated interval, or `T_useful@1m` > the 1.4 s baseline.** A `T_blind` gain with neither is
reported as a metric move and **not** as a capability.

## B.7 What is NOT claimed, registered in advance

* ⛔ This is **not** a safety result. PhysicalAI-AV ships no map, lane graph or agent boxes. Drift only.
* ⛔ One arm (v1 `flagship4b-speedjerk-30k` @ 29999), one readout (`str`, k = 20), one corpus.
* ⛔ Everything past 0.4 s is extrapolation of the operative readout; the `str` readout was calibrated
  at k = 20 and every number beyond it inherits that limit.
* ⛔ `a_hold0` is a CONSTANT and this corpus is mostly near-constant-action driving; the hold ceiling
  partly reflects that.
* ⛔ The window set is **episode-initial** (596 of 599 windows at `t0 = 0`) and runs ~6–12 % low in
  absolute level (`INHERITED-MEASURED`). All contrasts are paired on identical windows.

---

# PART C — REFERENCE VALUES I MAY QUOTE, AND ONE I MAY NOT

| quantity | value | class |
|---|---|---|
| v1 `ade_0_2s`, 40-ep canonical deployment | **0.4271** | `PUBLISHED` (registry §1.2) |
| v1 `ade_0_2s`, 600-ep deployment (a **different** deployment) | **0.4108** | `PUBLISHED` |
| ⛔ `~0.452` | **NOT QUOTABLE** — the deprecated `heldout` split-mean | — |
| `T_blind` own \| `str`, matched | 25 steps [2.5, 3.9] | `INHERITED-MEASURED` — **re-verified by my own gate before use** |
| `T_blind` hold \| `str`, matched | 115 steps [11.5, 17.4] | same |
| `T_blind` `ema0.8` | 64 steps | `INHERITED-MEASURED`, Rung 1 §5 |
| `T_blind` `blend0.75` (Rung 1 best eligible) | 116 steps [11.6, 17.2] | `INHERITED-MEASURED`, Rung 1 §6.1 |

---

# PART D — DELIVERABLES AND THE STAGING RULE

`git add` only. ⛔ **No `git commit`. No `git push`. No branch switch.** Every artifact is copied into
the repo working tree and staged; anything that exists in only one place is declared in the manifest.
`RETRACTION_LOG.md` is **not edited** — rows are drafted with root-cause classes and handed over,
because that file is shared and siblings are staging into it this session.
