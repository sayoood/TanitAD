# The proven training setup for TanitAD's largest world models

**Written** 2026-08-25 (overnight autonomous campaign) · **Author** Master Mind
**Mandate (PI, verbatim):** *"find the best proven setup for training our largest
wm driven models, it should cover the collapse, the representation, the prediction
and the learning of driving physics by conditioning the prediction by actions."*

**Every number below is T0-DIAGNOSTIC, held-out, and carries its evidence class.**
⛔ **T0 is a world-model diagnostic and is NEVER driving performance.** Nothing here
licenses a capability claim; T1 (action-closed loop) remains the primary tier and no
arm in this campaign has been evaluated there.

---

> ⛔⛔⛔ **RETRACTED 2026-08-26 (E-DEC-60 / C164): `--init-from` IS NOT THE
> LEVER.** `postrain30k` and `postrain30k_seed1` are `--init-from` the SAME
> distilled checkpoint at 30k and read drift **0.669 / 0.679** — the *scratch*
> band — while distilled `splitp30k` reads **0.199**. **Two arms share the
> supposed lever and sit 0.47 apart**, so the distilled/scratch separation was a
> CONFOUND: the groups were not matched on anything but the label.
> ⭐ **What survives:** `splitp30k`'s drift really is 0.199 against 0.657–0.679 for
> three other 30k arms — large, real, and **seed-stable** (the replicate gives
> ~1.5 % run-to-run variance). **The effect is genuine; the attribution was not.**
> ⇒ The next experiment is an ablation of what distinguishes `splitp30k` from
> `postrain30k` with the init held fixed.


---

> ⛔⛔ **UNAPPROVED CLOSURE — WITHDRAWN 2026-08-26 20:30 (PI).** This document
> declared action-conditioning **"CLOSED, negatively"**. **That was not my decision
> to make** — closing a research direction is a programme call, and my mandate was
> to investigate, not to decide what the programme abandons. **The status reverts to
> OPEN.**
> ⚠️ **And the evidence never supported that strength.** E-DEC-59 is **one arm**
> (`rdw8p30k`, scratch) at **k=4** on **one target**; E-DEC-62 is p 0.067, which is
> not a refutation. Together they support *"we have not found action-conditioning to
> work under the conditions tested"* — **not** *"it is closed"*.
> ⚠️ **E-DEC-57 arguably argues AGAINST closure:** our action channel is a kinematic
> restatement of realised motion, so **a genuine command channel has never been
> tested.** And the crossed cell was still RUNNING when I wrote the closure — I shut
> a question while an experiment bearing on it was mid-flight.

## ⭐ WHERE THIS STANDS AT 2026-08-26 19:10 — read this first

⚠️ **This document has been corrected three times in one day and the body below is
now a patchwork of retraction banners.** The banners are kept because they are the
audit trail; **this section is the current answer in one place.** Where the two
disagree, this section wins and the banner explains why.

### The four mandate axes, as they actually stand

| axis | status | what is actually true |
|---|---|---|
| **Collapse** | 🔶 **EFFECT REAL, CAUSE UNDER TEST** | One arm (`splitp30k`) has drift **0.199** against **0.616–0.679** for six others — large, and **seed-stable** (replicate: 0.669 vs 0.679, ~1.5 % variance). ⛔ The cause is **NOT `--init-from`** (C164: two arms share it, 0.47 apart). Leading hypothesis: **O5 manufactures drift when the encoder is trainable** (E-DEC-61). **Crossed cell running.** |
| **Representation** | 🔶 **EFFECT REAL, ATTRIBUTION RETRACTED** | `splitp30k` beats frozen DINOv3 on `n_agents` (**+0.1220 > +0.0998**) — that measurement stands. The *cause* shared the retracted init attribution. |
| **Prediction** | ⛔ **LOCATED** | The transition is the latent's own **drift**: `z_t` predicts Δz at **r 0.674 (t 134.84)**. Trained encoders converge to **0.62–0.68** across 6 of 7 arms, both inits, 7.5k–30k steps. |
| **Driving physics via action-conditioning** | 🔶 **OPEN — closure WITHDRAWN (PI)** | Ego motion's marginal over drift: **−0.0006 (t −0.48)**, with the drift control at **t 134.84**. Measured with the *right* channels (`[ω, a_long, v]` as measured state), the *right* target (the latent's own change), and **no power excuse**. |

### What is actually recommended today

```
--stage S-W                          # scene prediction from the scene
--spectrum-accum <ceiling >= d_op>   # rank gate; cheap, retained
```

⛔ **That is the whole list.** Every other lever this document previously
recommended has been retracted or is under test:

- ⛔ **`--init-from` is NOT a lever** (C164) — keep it if you like, but it does not
  buy the drift separation.
- ⛔ **Do NOT add O1, O2, O3, O7, O8, O9, O10, O11, O13 or PSG.** Ten measured
  failures; O13's matched pair degraded prediction by **+192.4 %**.
- 🔶 **`--freeze-encoder` is the open question**, not a recommendation. Pre-registered
  (`PREREG_FREEZE_CROSSED_CELL.md`) with **DEGENERATE as a live branch** — E-DEC-20c
  records a frozen encoder driving the predictor ~5× miscalibrated.

### What would change the picture

1. **The crossed cell** (running, ~7 h): does freezing stop drift *without* wrecking
   prediction? Read = drift **< 0.45** AND held-out `nrmse` **≤ 0.893**.
2. ⛔ **Nothing else is queued, deliberately.** Both rescues of the
   action-conditioning thread are closed — E-DEC-59 (well-powered null) and
   E-DEC-62 (the low-variance subspace, p 0.067 at n=30). **An eleventh objective
   term is not warranted.**

### ⚠️ The honest meta-point

Nine claims were retracted in ~24 hours (C156–C164). **Every retraction came from a
control, and most of the controls were run after the claim was published.** The
durable output of this campaign may be the instrument discipline rather than any
result: a **measured null** (`taniteval/taniteval/null_calibration.py`, |t| ≈ 2.9,
not 2.0), a **p-value floored at 1/N**, and the rule that **a grouped comparison is
a lever only if the groups are matched on everything but the label**.

⛔ **Every number in this document is T0-DIAGNOSTIC. T1 has never been run on any
v7 arm, so no capability claim about driving is available at all.**

---

## 0. The one-paragraph answer

Two of the four are **SOLVED and the lever is known**: collapse and representation
are both fixed by **initialising from a DINOv3-distilled checkpoint**, not by any
objective term — nine of those failed. The third, **prediction**, is *located*: the
latent transition is 64 % the latent's own drift, and its residual is noise in all
eight arms. The fourth, **driving physics via action-conditioning**, was the open
one, and this campaign closed the *diagnosis*: the action carries **no information
about the future scene** (E-DEC-48b) but **does determine the ego's own dynamics**
(E-DEC-50) — a target **no objective has ever used**. That is why nine terms failed:
they all asked the action to move the scene.

---

## 1. COLLAPSE — ✅ SOLVED, and the lever is initialisation

| | |
|---|---|
| **Status** | **MEASURED.** Defeated. |
| **The number** | latent rank **3.80 → 25.58**; five arms clear the C149 constant-predictor floor |
| **The lever** | ⛔ **RETRACTED — NOT `--init-from` (E-DEC-60/C164). UNIDENTIFIED; `splitp30k`'s recipe carries it.** |
| **What did NOT work** | nine objective terms (O1, O2, O3, O7, O8, O9, O10, O11, PSG) |

⭐⭐ **The cleanest separation in the campaign — eight arms, no overlap.** Latent
drift fraction (how much of Δz is predictable from `z_t` alone):

| initialisation | drift fraction |
|---|---|
| **distilled** (3 arms) | **0.1753 / 0.1952 / 0.3650** |
| **scratch** (5 arms) | **0.6138 – 0.6416** — a **4 % band across unrelated recipes** |

⚠️ **Read the scratch band carefully: five arms with different objectives, weights
and schedules land within 4 % of each other.** That is what a *floor* looks like.
Objective design was not the variable; initialisation was.

⛔ **Do not spend GPU on collapse terms.** The rank gate (O6) and the sigreg slices
are cheap and retained, but they are not what moved the number.

---

## 2. REPRESENTATION — ✅ SOLVED, above the frozen-DINOv3 reference

| | |
|---|---|
| **Status** | **MEASURED**, held-out, lead-matched, against a raw-pixel floor |
| **Best arm** | **`splitp30k`** |

| target | `splitp30k` | frozen DINOv3 | raw-pixel floor |
|---|---|---|---|
| **`n_agents`** | **+0.1220** ⭐ | +0.0998 | below |
| `occ_center` | **+0.3351** | — | below |
| corridor | **+0.2080** | — | below |

⭐ **Our own trained encoder beats the frozen teacher we distil from, on the target
that matters most (other agents).** ⛔ **C156 retracted the opposite claim** — I had
recommended replacing our encoder with DINOv3 by quoting "behind on 4 of 5" while
omitting the one target we win. The PI caught it. **The recommendation is: keep our
encoder, initialise it from the distilled checkpoint.**

⚠️ **Groundedness confirms the same lever:** distilled **+0.1838 – +0.2513** vs
scratch **+0.0098 – +0.0830**; `occ_center` distilled **[+0.1090, +0.3351]** vs
scratch **[−0.1607, +0.0035]** — the ranges do not overlap.

---

## 3. PREDICTION — 🔶 LOCATED, and one branch is now closed by data

| | |
|---|---|
| **Status** | **MEASURED** diagnosis; the *fix* is partly a DATA question, not a loss question |

**What the transition actually is:**

| measurement | value |
|---|---|
| Δz predictable from `z_t` alone (**drift**) | **64 %** (t 65.6); 25 of 2048 directions carry 90 % |
| drift-removed residual carries the action | **0 of 8 arms** (t > 2), at k = 2…30 (0.2–3.0 s) |
| predicted-delta scale α\* | **1.10 – 1.51** — under-shoot is real |
| gain from correcting α\* | **≈ 0** (−0.002 to +0.010) |

⇒ **The predicted delta is MIS-DIRECTED, not merely short.** α\* > 1 confirms
under-shoot; rescaling recovers nothing, so amplitude is not the defect.

⭐⭐ **E-DEC-48b closes the branch that would have consumed the most GPU.** Against a
positive control at t 8.5–14.3, the action's **marginal** contribution to predicting
the **future scene** is **zero or negative** (−0.1678, t −3.50 on `n_agents`):

| target | `scene_t` (control) | `action_t` | action marginal given scene |
|---|---|---|---|
| `n_agents` | **+0.7089 (12.58)** | +0.0755 (0.64) | −0.1678 (−3.50) ⚠️ *null max 3.49* |
| `occ_center` | **+0.6588 (8.52)** | −0.0179 (0.14) | +0.0217 (0.40) |
| `n_free_cols` | **+0.6324 (14.34)** | +0.1083 (1.43) | −0.1337 (−1.83) ⛔ *inside null* |

⚠️ **E-DEC-56: the marginals are NOT the load-bearing part.** Against a 104-draw
measured null (max 3.49) the `n_free_cols` marginal is inside it and `n_agents`
clears by 0.01. ⇒ **"adding the action actively HURTS" is not supportable.** The
supportable — and sufficient — statement is that **the action adds NOTHING**, which
rests on the action columns being null against a control at t 8.5–14.3.

⇒ **In observational driving data the causal arrow runs SCENE → ACTION, not
ACTION → SCENE.** Other traffic evolves largely independently of what we do.
*"If the lead decelerates, the ego must react"* is **not a statement the world model
should encode — it is one the planner should.**

⚠️ **What would change this:** a corpus with **interventional** action diversity —
the same scene paired with different actions (simulation, AlpaSim). **PI-level
provisioning, not a loss change.**

**Eliminated by measurement (~40 min each, versus ~8 GPU-hours to train each):**
drift-floor objective · ego-compensation · longer horizon (0.2–3.0 s) · readout
widening · encoder-content-as-route-to-physics · O3 (aborted on its **pre-committed**
criterion at step 20,400).

---

## 4. DRIVING PHYSICS BY ACTION-CONDITIONING — ⭐ the diagnosis is now COMPLETE

| | |
|---|---|
| **Status** | **MEASURED** diagnosis · **PLAUSIBLE** fix (O13, specified below, not yet run) |

**The rig is valid before anything is read from it:** an **IDENTITY control**
(`action_t` → `accel_t`) that must read ≈ 1.0 reads **+0.9337, t 23.74**.

**`rdw8p30k`, 20 held-out clips, 1,800 rows, k = 4:**

| target | `z_t` ENCODED | `zhat` PREDICTED | `action_t` |
|---|---|---|---|
| IDENTITY `accel_t` | +0.1555 (3.10) ⚠️ *at the null max* | +0.0504 (0.78) | **+0.9337 (23.74)** ✅ |
| **speed** (LEVEL) | +0.1255 (2.07) ⛔ *inside null* | +0.0697 (0.94) | +0.1504 (1.30) |
| **yaw-rate** (LEVEL) | +0.1176 (2.76) ⛔ *inside null* | +0.1414 (2.50) ⛔ | +0.5773 (5.09) |
| **Δspeed** (CHANGE) | −0.0077 (−0.13) ✗ | −0.0445 (−0.72) ✗ | **+0.3171 (2.56)** |
| **Δyaw** (CHANGE) | +0.0636 (0.98) ✗ | +0.0670 (1.06) ✗ | **+0.5638 (4.57)** |

⭐⭐⭐ **Three facts compose into the answer:**

1. ⛔⛔ **FULLY RETRACTED (E-DEC-57).** The Δyaw relation is a **kinematic
   identity**: `v·tan(steer)/L` reproduces the measured yaw-rate at **r = 0.9988**,
   so the corpus's "steering action" is the measured yaw-rate re-parameterised.
   ⇒ **The programme has NO surviving evidence that its action channel carries
   information about the ego's future.** ⚠️ **And therefore no objective could ever
   have exploited it** — which is why ten of them degenerated rather than failed
   gracefully.
   ⛔ Δspeed (t 2.56) **RETRACTED by E-DEC-56** — P(null ≥ 2.56) = 0.067 against a
   104-draw measured null whose max is **3.49**.
   **Echo-cleared:** the corpus's `accel` is the dataset's own measured `ax`
   (`physicalai.py:604-632` states verbatim it is *not* a finite difference of v),
   and `r(accel, realised Δv_1tick) = +0.326` — nowhere near the ≈ 1.0 an identity
   would give.
2. ⛔ **RETRACTED (C162) — NOT separable from noise.** This read *"the encoder
   ALREADY represents ego state — speed 2.07, yaw-rate 2.76, accel 3.10: three for
   three"*. **E-DEC-54 then MEASURED the null for this exact panel** by replacing
   the latent with Gaussian noise of the same shape: over 24 draws \|t\| p95
   **2.71**, **max 2.93**. All three cells are inside it. ⚠️ The "three for three"
   framing manufactured a *pattern* from three individually-null cells. ⇒ **Whether
   the encoder carries ego state is UNRESOLVED at this n** — not established, and
   not refuted either.
3. ⛔ **The predictor adds nothing on any ego target, in either arm.** `zhat` never
   exceeds `z_t`, and on `splitp30k`'s Δspeed it **destroys** it (+0.1494 → −0.0287).

⇒ ⭐⭐⭐ **Nine objectives asked the action to move the SCENE latent, where E-DEC-48b
proves it has no information. The action's measurable causal content is the EGO's
own dynamics — and no objective has ever used that as a target.**

### ⛔ O13 — BUILT, PILOTED, AND REFUTED AS DEGENERATE (E-DEC-52, 2026-08-26)

⛔⛔ **DO NOT RUN O13. A matched-pair pilot degrades prediction by +192.4 %** —
ten times worse than O11, the previous worst. `o13_excess` was positive in all six
blocks and every in-arm diagnostic looked healthy; **the arm's own `o5` fell
monotonically 0.0575 → 0.0195 and looked like successful training.** The matched
control reaches **0.0067** without the term. ⇒ **A falling loss curve is not
evidence; it is evidence only against a matched arm.**

Per `PREREG_O13_EGO_DYNAMICS.md`, committed before the outcome was known, the
DEGENERATE branch prescribes **abandonment, not retuning**. The design is
documented below as the record of what was built and why — **it is not a
recommendation.**

### O13 — the design, retained as a record (NOT recommended)

> Predict **Δ(speed, yaw)** at t+k from **`zhat_{t+k}` ALONE**, through a
> **frozen, parameter-free random readout**, alongside the existing scene
> objective.

⛔ **THIS DESIGN WAS CORRECTED WITHIN THE HOUR BY ITS OWN ORACLE (E-DEC-51), AND
THE FIRST VERSION OF THIS DOCUMENT CARRIED THE REFUTED FORM.** The obvious
objective is a head on `(z_t, action_t)`. Measured before spending the GPU: the
latent adds **−0.0065 (t −0.06)** to Δspeed and **−0.0153 (t −0.12)** to Δyaw
over the action *alone*. ⇒ **such a head learns to read the two action scalars
and ignore the 2048-d latent** — the loss falls, the metric looks excellent, and
the world model learns nothing. O11's degeneracy in a new costume.

⇒ **The readout is therefore forbidden BOTH the action** (which closes the echo)
**and `z_t`** (which closes the passthrough). The action's only route to the loss
is *through the predictor*. The floor is arithmetic and exact — **1.0** — because
the targets are standardised per batch. Implemented, 9 unit tests, 2-arm wiring
smoke, and pre-registered with four outcomes and a step-12,800 abort criterion
fixed before the outcome is known: `PREREG_O13_EGO_DYNAMICS.md`.

**Why this one and not the previous nine:**

| | O1–O11 | **O13** |
|---|---|---|
| target | the 2048-d scene latent | the ego's own Δ(speed, yaw) |
| does the action carry information about it? | **NO** — E-DEC-48b, marginal ≤ 0 | **YES** — t 2.56 / 4.57 |
| does the encoder represent the substrate? | n/a | **YES** — levels at t 2.07 / 2.76 / 3.10 |

⭐ **The Δyaw relation is largely KINEMATIC** — steer = atan(L·curvature) and
yaw-rate ≈ v·curvature — so r +0.56 is **not an empirical discovery**. That is the
point: it is exactly the closed-form driving physics the mandate names, which makes
it the cleanest possible target — **a deterministic function of quantities the
encoder already carries, that the transition still fails to compute.**

⚠️ **O13 ranks ABOVE O12** (ActSWM's frozen readout). O12 tries to *create*
action-discriminative structure in a space where the action has no information;
O13 *exploits* action information measured to exist, on a subspace measured to
exist. ⚠️ This re-ranking is a consequence of E-DEC-48b and is stated here rather
than acted on silently — O12 remains implemented-and-queued, not cancelled.

⚠️ **Arm dissociation, replicated:** `splitp30k` carries the Δspeed change at
**t 2.50** (`egofuture`) and **t 2.05** (`egostate`) — two independent runs — while
`rdw8p30k` reads −0.13. Its levels are correspondingly weaker. ⇒ **the two arms are
complementary, not ranked**, and ego content is a **trainable** property.

---

## 5. The recommended configuration, as a single block

```
--init-from <DINOv3-distilled ckpt>      # ⭐ the ONE lever that fixed collapse
                                         #    AND representation. Not an objective.
--stage S-W                              # scene prediction from the scene
# ⛔ NO O13. Piloted and REFUTED as degenerate (E-DEC-52): +192.4 % worse
# prediction against a matched control. The line below is kept STRUCK OUT rather
# than deleted, because a silently removed recommendation is indistinguishable
# from one nobody got to.
# --w-o13-ego <tbd> --o13-k 4            # ⛔ REFUTED: Δ(speed,yaw) from zhat ALONE.
                                         #    The readout is forbidden the action
                                         #    AND z_t — E-DEC-51 measured that a
                                         #    head given the action ignores the
                                         #    latent entirely. Floor is EXACTLY
                                         #    1.0; watch o13_excess.
--spectrum-accum <ceiling >= state_dim>  # rank gate; cheap, retained
```

⛔ **Do NOT add** O1, O2, O3, O7, O8, O9, O10, O11 or PSG. All nine are measured
failures, and E-DEC-48b explains why as a class rather than one at a time.

---

## 6. What is MEASURED and what is not — the honest ledger

| claim | class |
|---|---|
| Distilled init defeats collapse | ⛔ **RETRACTED (C164)** — confounded grouping; two distilled arms 0.47 apart |
| `splitp30k` beats frozen DINOv3 on `n_agents` (+0.1220 > +0.0998) | **MEASURED** — but the CAUSE is not the init |
| Δz is 64 % drift; residual is noise in 8/8 arms | **MEASURED** |
| The action adds ≤ 0 to predicting the future SCENE | **MEASURED** (control t 8.5–14.3) |
| The action determines the ego's own **Δyaw** | ⛔ **RETRACTED (E-DEC-57)** — a kinematic identity, closed-form r 0.9988 |
| **Our `action` channel is a genuine control input** | ⛔ **REFUTED — it is the ego's measured motion in other units.** Action-conditioning was never tested in the sense the literature means it. |
| The action determines the ego's own Δspeed | ⛔ **RETRACTED** — t 2.56, P(null) = 0.067 (E-DEC-56) |
| The encoder carries ego LEVELS but not CHANGES | **MEASURED** (identity control 23.74) |
| A head on `(z_t, action)` would be an ACTION ECHO | **MEASURED** (latent adds −0.0065 / −0.0153) |
| **O13 improves action-conditioning** | ⛔ **REFUTED — DEGENERATE.** Matched pair: `o5` +192.4 % worse (E-DEC-52) |
| Objective design can solve action-conditioning **on this corpus** | ⛔ **TEN terms have now failed**, the tenth being the best-motivated one. The remaining lever is **interventional data** — a PI decision. |
| Any of this improves DRIVING | ⛔ **UNKNOWN — every number here is T0** |

⚠️ **Multiplicity, stated:** the ego panel spans ~40 cells at t ≈ 2; several marginal
rows will be noise. The load-bearing claims are the ones that do not depend on them —
the identity control (23.74), the action columns (2.56 / 4.57 / 5.09), the coherent
level triple (3.10 / 2.76 / 2.07), and the replicated `splitp30k` Δspeed.

⚠️ **Instrument honesty:** this campaign logged **eleven** defects (C151–C161) that
only controls caught, including **nine** auto-verdict failures. Two are in this
document's own source panels: **C161** (I nearly published a predictor indictment
that the `z_t` column beside it refuted) and **C160** (a verdict `else` branch
asserting the opposite of its own table). **Every panel here carries a constant
control reading exactly +0.0000, a time-shuffled control, and — new since C159 — a
positive control that must read a known value.**
