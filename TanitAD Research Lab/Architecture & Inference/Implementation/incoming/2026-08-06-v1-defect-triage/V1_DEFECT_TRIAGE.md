# flagship v1 — three observed defects, what the numbers say, and what to do about them

**Raised by Sayed, 2026-08-06, from watching v1 drive:**

1. *"Its planning longitudinal velocity and acceleration wrong and thus the distance ahead"*
2. *"It's not considering any jerk or something comparable, so its trajectory is jumping
   sometimes between the frames"*
3. *"The tactical manoeuvre is toggling the whole time"*

All three are real. Two of them now have numbers; the third has an instrument built and a
run pending. **Evidence class MEASURED (ours)** unless marked otherwise — artifacts listed
at the end.

---

## 0. The numbers, first

MEASURED on 39 paired PhysicalAI OOD-val clips at a 2.0 s / 20-waypoint dense 0.1 s grid,
against the same human ground truth, with Alpamayo 2 Super on the identical clips as an
external reference (`comparison/implied_controls.json`, `comparison/a2_four_families.json`):

| quantity | **flagship v1** | Alpamayo 2 Super | **human** |
|---|---|---|---|
| speed bias (m/s) | **+0.4245** | +0.0569 | 0 |
| along-track bias @2 s (m) | **+0.8176** | +0.1132 | 0 |
| implied accel RMS (m/s²) | **4.1656** | 1.027 | **0.8048** |
| ↳ × the human | **5.18×** | 1.27× | 1.00× |
| implied accel bias (m/s²) | **+0.7160** | +0.1076 | 0 |
| **intra-plan jerk RMS (m/s³)** | **64.2966** | 1.7908 | **1.7975** |
| ↳ × the human | **35.8×** | 0.92× | 1.00× |
| entry (launch) transient (m/s²) | 1.5367 | 1.4039 | 0.4249 *(instrument floor)* |
| curvature MAE (1/m) | 0.007551 | 0.009162 | — |
| heading MAE (deg) | 0.78 | 0.68 | — |

**Two readings jump out.**

⭐ **The jerk number is the observation.** 64.30 against a human floor of 1.80 — **35.8×** —
and Alpamayo, through the identical instrument, sits at 0.92× the human. This is *inside a
single 2 s plan*, so the roughness is not only a frame-to-frame effect. Nothing in the
programme has ever measured it.

⭐ **The lateral channel is not the problem.** We *beat* a 34.3 B six-camera model on
curvature MAE and match it on heading and yaw-rate. Every one of these defects is
longitudinal or temporal.

⚠️ Jerk is a third difference of waypoints and so amplifies discretisation noise — but the
human's own path and Alpamayo's go through the *same* instrument and land at 1.80 and 1.79.
The floor is ~1.8. 64.3 is not noise.

---

## 1. Would the unicycle action space fix the longitudinal problem?

**Yes for three of the four sub-problems, structurally — and no for the fourth on its own.**

The mechanism matters, so take the defect apart:

| sub-defect | measured | does `(accel, curvature)` + integration fix it? |
|---|---|---|
| **launch transient** — first step inconsistent with the ego's real `v0` | 1.5367 vs 0.4249 floor | ⭐ **BY CONSTRUCTION.** `v` is a *state* initialised from the true `v0` and carried forward; the first displacement is `v0·dt` exactly. A free-waypoint head can place waypoint 1 at any speed it likes, and does. This sub-defect becomes **unrepresentable**. |
| **accel magnitude** — 5.18× the human | 4.1656 vs 0.8048 | ⭐ **YES.** Acceleration becomes the head's *output*, bounded by construction (softsign to ±9.8 m/s²) and penalisable with one term on a decision variable — instead of being a second difference of an unconstrained output where the gradient is diffuse. |
| **jerk** — 35.8× the human | 64.2966 vs 1.7975 | ⭐ **YES, and this is the biggest structural win.** Jerk becomes a **first** difference of the output. Today it is a **third** difference of waypoints, which is exactly why 64.3 is reachable without anything noticing. |
| **speed / distance bias** — +0.42 m/s, +0.82 m | +0.4245 / +0.8176 | ⚠️ **NOT BY ITSELF.** An integrated head can still systematically choose too-high accel. |

### Why the bias may nonetheless follow

The along-track error is almost exactly the speed bias integrated: `0.4245 m/s × 2 s = 0.849 m`
against a measured `+0.8176 m`. So there is one longitudinal defect, not two.

And the accel statistics say the arm is **oscillating**, not merely offset: RMS 4.1656 with a
bias of +0.7160 implies a spread of `√(4.1656² − 0.7160²) ≈ 4.10 m/s²` — the *variation* is
5.7× the *bias*. **HYPOTHESIS (to be tested, not assumed): a head that cannot thrash has far
less room to be systematically wrong, so suppressing the oscillation may carry most of the
bias with it.** If it does not, one explicit bias term closes the rest.

### ⛔ The attribution limit, stated because it bounds the claim

Alpamayo's accel RMS is **1.27× the human — better than ours, but not 1.0×**, and it uses the
unicycle action space *plus* a 32 B backbone *plus* a flow-matching diffusion head. **We do
not know how much of its 1.27× is the action space.** So the honest prediction is
*"large improvement on magnitude and jerk, direction certain, size unknown"* — not
*"we will land at 1.27×"*. The pre-registration below is written to be falsifiable either way.

### Pre-registration (extends the one in `ALPAMAYO2_SUPER_ANALYSIS.md` §13)

An arm identical to `flagship-v1arch-v2bal-30k` except that the operative head emits
`(accel, curvature)` and integrates through `rollout_unicycle`, at **matched steps**, read
through `taniteval.four_families` — **never through ADE**, which is blind to all of this:

* **PASS** — implied accel RMS ≤ **2.0×** human (≤ 1.6 m/s²) **AND** jerk RMS ≤ **3×** human
  (≤ 5.4 m/s³) **AND** `speed_bias_mps` < **+0.15** **AND** LATERAL does not regress
  (curvature MAE ≤ 0.0090).
* **PARTIAL (the interesting outcome)** — accel/jerk hit target but `speed_bias` stays above
  +0.30 ⇒ the oscillation and the bias are **independent**, and the bias is a loss or label
  problem, not a parameterisation problem. That is a finding, not a failure.
* **FAIL** — accel RMS stays above 3× human, or lateral regresses ⇒ the action space is not
  the lever.

---

## 2. Jerk / "jumping between the frames" — cheap fixes, ranked

These are **two different defects** and the fixes differ. Separating them is the first job:

* **(a) intra-plan roughness** — MEASURED, 64.30 vs 1.80. Confirmed.
* **(b) inter-frame replan disagreement** — the plan at `t+0.8 s` disagreeing with the plan at
  `t` over their overlap. **No instrument existed.** One is now built
  (`tools/temporal_stability.py`); the run is pending.

⭐ **Why (b) is a clean measurement:** the ground truth is *one* trajectory — the human's
future from `t+Δ` is literally a suffix of its future from `t`. **The GT floor is zero by
construction.** Any value the arm scores is pure self-inconsistency: no baseline to argue
about, no estimator to get wrong.

| # | fix | cost | what it does | risk |
|---|---|---|---|---|
| **1** | **Jerk barrier on the existing head** — `λ·mean(relu(\|jerk\| − human_p90))` on the predicted waypoints | ⭐ **~5 lines, no architecture change, today** | attacks (a) directly | ⚠️ Use a **barrier**, not shrinkage. A plain `λ·jerk²` also penalises legitimate sharp manoeuvres and will flatten emergency braking. Same pattern as `kamm_circle_violation`, already in the stack. |
| **2** | **Unicycle head** (§1) | medium — new head + integrate | makes jerk a *first* difference, so #1 becomes well-conditioned instead of a third-difference penalty on noise | the retrain |
| **3** | **Inference-time EMA over the control sequence** | ⭐ **free, no retraining, 1 line** | attacks (b) | ⛔ **Only legal once #2 lands.** EMA-ing *positions* shortcuts corners and distorts geometry; EMA-ing *controls* and re-integrating cannot. **This is a third independent reason to adopt the unicycle: it makes the free fix legal.** |
| **4** | **Previous-plan conditioning** — feed the last plan's shifted tail, predict a residual | one extra input channel | attacks (b) at train time | ⚠️ lock-in: a sluggish reaction to a new hazard is worse than a jump. Cap the residual; condition the *controls* only, never the perception. |
| **5** | **Temporal-consistency loss** — two windows Δ apart in one batch, transform plan A into B's frame, penalise overlap disagreement | one extra forward pass + dataloader change | the principled fix for (b); optimises exactly the metric now built | most expensive of the five |

**Recommended order: #1 now (it is nearly free and (a) is confirmed at 35.8×), then #2, then
#3 for free.** #5 only if the pending (b) measurement shows the residual is large after #1–#3.

---

## 3. Manoeuvre toggling — diagnose the *cause*, because it is structural

Two measurements bear on it:

* The declared manoeuvre is only **weakly coupled to the path actually driven** — κ **0.3432**
  at gate 0.15, and *falling* to 0.1159 at finer gates
  (`comparison/a2_gate_audit.json`). The head is close to **free-running**: toggling costs it
  nothing, because nothing downstream depends on it consistently.
* The head is **one 5-way softmax** over `[lane_keep, turn_left, turn_right, accelerate,
  brake_stop]`, which **mixes the lateral and longitudinal decisions**.

⭐ **That mixing is the structural cause.** In a decelerating left turn, `turn_left` **and**
`brake_stop` are *both true*. They are not alternatives, but the softmax forces them to
compete, so the argmax flips on noise between two classes that should both be firing. **No
amount of smoothing fixes a head that is being asked an ill-posed question.**

| # | fix | cost | note |
|---|---|---|---|
| **1** ⭐ | **Factorise into independent longitudinal × lateral heads** — exactly what Alpamayo does (`Longitudinal: Gentle Deceleration. / Lateral: Steer Left. / Lane: Lane Keep.`) | **~30 lines + a label-script change** — the 5 existing classes map onto the product space, and `refb_labels` already derives them from ego dynamics, so this is re-deriving labels, **not new annotation** | removes the competition entirely. **Top recommendation.** Also adds a severity axis (`Steer` vs `Sharp Steer`), which is where Alpamayo's declaration carries fine information ours discards. |
| **2** | **Hysteresis at inference** — switch only when the challenger's logit beats the incumbent by margin τ for n consecutive windows | ⭐ **free, 3 lines, today** | It *hides* the toggle rather than fixing it — but it is also the **discriminating experiment**: if a small τ removes most toggles, the logits are near-tied and #1 will fix it; if it does not, the head is genuinely unstable and #3 is needed. Run it as a diagnostic first. |
| **3** | **Coherence loss** — penalise disagreement between the declared manoeuvre and the manoeuvre the rolled-out trajectory actually executes | medium | gives toggling a cost. Directly attacks the κ 0.34 free-running finding. |
| **4** | Temporal EMA on the logits | free | same class as #2; prefer #2, which is interpretable. |

⛔ **Measure the toggle rate before fixing it.** `tools/temporal_stability.py` emits
`maneuver_toggle_rate` and `maneuver_mean_dwell_windows` over consecutive windows within an
episode. Without that number no fix can be shown to have worked, and #2 would silently
"succeed" by construction.

---

## 4. What this adds up to

**One architectural change addresses all three observations**, which is unusual enough to
state plainly:

* emitting **controls** instead of waypoints fixes the launch transient by construction,
  makes accel and jerk directly penalisable, and legalises free inference-time smoothing;
* **factorising the manoeuvre head** removes the ill-posed competition that makes the
  tactical layer toggle.

Both are borrowed from a system we measured on our own clips, and both are cheap relative to
a retrain we are going to do anyway.

⚠️ **The order matters.** Do the **jerk barrier (§2 #1)** and the **hysteresis diagnostic
(§3 #2)** first — they are nearly free, they run against the *current* checkpoint, and they
tell you how much of each defect is parameterisation versus loss. Committing to a new head
before that measurement is spending a retrain to learn something two cheap experiments
would have told you.

---

**Artifacts.** `Benchmarks & Eval/Research/2026-08-05-alpamayo2-super/comparison/` —
`implied_controls.json`, `a2_four_families.json`, `a2_gate_audit.json`.
Instruments — `stack/tanitad/models/kinematic.py` (`rollout_unicycle`,
`unicycle_controls_from_path`, `entry_speed_mismatch`), `tools/temporal_stability.py`.

---

## 5. ⭐ RESULTS — both pending runs landed, and they change the ORDER of the fixes

Full detail in `results/`. Headlines, **MEASURED 2026-08-06 on the idle A40**:

**`results/TEMPORAL_STABILITY_RESULT.md`** — 6,834 windows / 6,794 consecutive pairs, 40
episodes, **stride 1 so consecutive windows are 0.1 s apart**:

| | flagship v1 | GT floor |
|---|---|---|
| replan shift mean / p90 / max (m) | 0.0947 / 0.2022 / **1.0722** | **0.0** |
| **replan accel jump mean (m/s²)** | **1.1021** | **0.0001** |
| intra-plan jerk RMS (m/s³) | **52.2148** | **1.7066** (30.6×) |
| manoeuvre toggle rate / mean dwell | **0.1759** / **5.53 windows = 0.55 s** | — |

⭐ **Observation (2) is a CONTROL-space defect, not a position-space one.** In position the
replan is nearly fine — 9.5 cm mean over 0.1 s. In control it is not: the commanded
acceleration at the **same absolute instant** is revised by **1.1021 m/s² every 0.1 s**, and
the human's *entire* accel RMS is 0.8048. A small position shift hides a large acceleration
change — which is what a passenger feels and what no position metric can see.

⭐ **Observation (3) is literal.** Mean dwell **0.55 s** — the declared manoeuvre changes about
twice a second.

**`results/GATE_RERUN_RESULT.md`** — 880-window gate-swept panel: `verdict_stable = true`,
κ range [0.2038, 0.5787]. The published coherence verdict holds; the magnitude spans 2.8× and
is not quotable without its gate. `kappa_turn_subset = 0.2005`, *on* the threshold.

⇒ **Revised order of action.** The jerk barrier and the control-EMA move up, because the
defect is now located precisely: it is in the **commanded acceleration**, both within a plan
(30.6× the human) and across plans (1.1 m/s² per frame). Both are structurally invisible to a
free-waypoint head and directly addressable once the head emits controls — which is a fourth
independent argument for the unicycle, and the strongest one.

**Open.** Episode-cluster bootstrap over the 40 episodes for the temporal numbers (pairs within
an episode are strongly dependent, so no CI is quoted rather than an optimistic one); REF-B /
REF-C / v2corpus panels still carry unswept κ and now say so in their own output.
