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
| **The lever** | **`--init-from <DINOv3-distilled ckpt>`** — a single config change |
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

| target | `scene_t` (control) | `action_t` | **action marginal given scene** |
|---|---|---|---|
| `n_agents` | **+0.7089 (12.58)** | +0.0755 (0.64) | **−0.1678 (−3.50)** |
| `occ_center` | **+0.6588 (8.52)** | −0.0179 (0.14) | +0.0217 (0.40) |
| `n_free_cols` | **+0.6324 (14.34)** | +0.1083 (1.43) | **−0.1337 (−1.83)** |

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
| IDENTITY `accel_t` | +0.1555 (3.10) | +0.0504 (0.78) | **+0.9337 (23.74)** ✅ |
| **speed** (LEVEL) | **+0.1255 (2.07)** ✅ | +0.0697 (0.94) | +0.1504 (1.30) |
| **yaw-rate** (LEVEL) | **+0.1176 (2.76)** ✅ | +0.1414 (2.50) ✅ | +0.5773 (5.09) |
| **Δspeed** (CHANGE) | −0.0077 (−0.13) ✗ | −0.0445 (−0.72) ✗ | **+0.3171 (2.56)** |
| **Δyaw** (CHANGE) | +0.0636 (0.98) ✗ | +0.0670 (1.06) ✗ | **+0.5638 (4.57)** |

⭐⭐⭐ **Three facts compose into the answer:**

1. **The action DOES determine the ego's own future** — Δspeed t 2.56, Δyaw t 4.57.
   **Echo-cleared:** the corpus's `accel` is the dataset's own measured `ax`
   (`physicalai.py:604-632` states verbatim it is *not* a finite difference of v),
   and `r(accel, realised Δv_1tick) = +0.326` — nowhere near the ≈ 1.0 an identity
   would give.
2. **The encoder ALREADY represents ego state** — speed 2.07, yaw-rate 2.76, accel
   3.10: **three for three**, a coherent pattern rather than one marginal row.
3. ⛔ **The predictor adds nothing on any ego target, in either arm.** `zhat` never
   exceeds `z_t`, and on `splitp30k`'s Δspeed it **destroys** it (+0.1494 → −0.0287).

⇒ ⭐⭐⭐ **Nine objectives asked the action to move the SCENE latent, where E-DEC-48b
proves it has no information. The action's measurable causal content is the EGO's
own dynamics — and no objective has ever used that as a target.**

### O13 — the ego-dynamics objective (the recommended next arm)

> Predict **Δ(speed, yaw)** at t+k from `(z_t, action_t)` with a small supervised
> head, alongside the existing scene objective.

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
--w-o13-ego <tbd>                        # ⭐ NEW: Δ(speed,yaw) from (z_t, action)
                                         #    the ONLY action target with measured
                                         #    information behind it
--spectrum-accum <ceiling >= state_dim>  # rank gate; cheap, retained
```

⛔ **Do NOT add** O1, O2, O3, O7, O8, O9, O10, O11 or PSG. All nine are measured
failures, and E-DEC-48b explains why as a class rather than one at a time.

---

## 6. What is MEASURED and what is not — the honest ledger

| claim | class |
|---|---|
| Distilled init defeats collapse (drift 0.175–0.365 vs 0.614–0.642) | **MEASURED** |
| Distilled init drives representation (`n_agents` +0.1220 > DINOv3 +0.0998) | **MEASURED** |
| Δz is 64 % drift; residual is noise in 8/8 arms | **MEASURED** |
| The action adds ≤ 0 to predicting the future SCENE | **MEASURED** (control t 8.5–14.3) |
| The action determines the ego's own Δspeed / Δyaw | **MEASURED** (t 2.56 / 4.57) |
| The encoder carries ego LEVELS but not CHANGES | **MEASURED** (identity control 23.74) |
| **O13 will improve action-conditioning** | ⚠️ **PLAUSIBLE — not yet run** |
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
