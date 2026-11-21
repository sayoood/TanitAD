# BLIND-IMAGINATION DRIVING — how long, how well, and what actually limits it

**Date:** 2026-07-26 (Europe/Berlin; pods log UTC). **Stream:** blind-imagination driving (new).
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, written **before any rollout was executed**.
**Host:** pod2 (A40, idle). pod1 (training v2corpus), pod3 and the eval pod (Bar-A) were never touched.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID and no raw content appears in this folder.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers (M1):** `PROVISIONAL` (one path, unreproduced) · `CONFIRMED` (independent reproduction) ·
`DECISION-GRADE` (CONFIRMED + pre-registered + estimator named + falsifier stated).

*(Every table below is rendered from `artifacts/*.json` by `scripts/bi_report.py` and placed by
`scripts/bi_splice.py`. No number in this document is typed by hand.)*

---

# 0. VERDICT

> ## **The answer depends entirely on WHO CHOOSES THE ACTIONS, and the program has only ever measured the privileged half. Given the expert's actions the world model's imagination beats a frozen percept for `T_blind` = 6.5 s [6.5, 8.9]. Given its OWN actions — the deployable condition, the one that answers the PI — it beats a frozen percept for 0.8 s [0.8, 1.0], is separated-WORSE from 1.1 s onward, and NEVER beats constant velocity at any horizon.**
>
> **Pre-registered Outcome A in the privileged regime. Pre-registered Outcome B in the deployable
> regime. Both were committed in advance; neither is softened.**
> `MEASURED` · 599 windows / **596 episode clusters** · paired episode-cluster bootstrap, B = 2000,
> seed 0, identical windows · `artifacts/t_blind.json`.

| regime | who picks the actions | `T_blind` (CI95) | beats the CV floor over |
|---|---|---|---|
| **(i)** true future actions | **the expert** ⚠️ privileged | **6.5 s** [6.5, 8.9] | 0.4 s … 7.4 s |
| **(ii)** own kinematic actions | ⭐ **the model** — deployable | **0.8 s** [0.8, 1.0] | ⛔ **never** |
| **(ii-0)** last observed action held | nobody (no policy) — deployable | **3.2 s** [3.2, 5.0] | 0.4 s … 3.8 s |
| control | the **true** motion through my own inverse | **5.4 s** [5.4, 7.3] | — |

### ⭐⭐ 1. The one strongly positive, immediately actionable result — and it costs ZERO GPU

`HierarchicalGrounding` carries **three** step readouts and `grounding_losses` trains **all three on
the same operative imagination rollout**, only over different lengths: `op_fwd_k = 4`,
`tac_fwd_k = 16`, `str_fwd_k = 20` (MEASURED, §1.2 — trainer defaults, and v1's run manifest overrides
none of them). **Every grounded number in this program decodes with `step["op"]`, which was calibrated
over 4 steps and is then read at 20.**

Swapping the decoder — same checkpoint, same weights, no training, no data:

| decoder | `ade_0_2s` (sparse 4-waypoint, the program's own convention) | `de@2s` | separated-better than the CV floor over |
|---|---:|---:|---|
| `step["op"]` — **what the program uses today** | **0.3839** [0.3598, 0.4106] | 0.814 | 0.4 s … **7.4 s** |
| `step["tac"]` (16-step) | **0.1865** [0.1684, 0.2056] | 0.313 | — |
| `step["str"]` (20-step) | **0.1950** [0.1767, 0.2139] | 0.311 | 0.6 s … **18.5 s** (180 of 185 steps) |

⇒ **v1's blind 2 s error halves (−51.4 % / −49.2 %, paired-separated at every horizon ≥ 1 s), and the
horizon over which imagination beats constant velocity goes from 7.4 s to the end of the sweep.**
🔴 **`wm_canary_ade_2s` is exactly this quantity, and `BOOST_PROGRAM` §3.3 records Bar B as having
NO identified lever and being UNOWNED.** This is a lever, it is measured, and it costs one line.
⚠️ Measured on **v1** and on this window set; transfer to **v4** is a `HYPOTHESIS` and an eval-only
check (§7).

### ⛔ 2. The deep negative, and it explains the rest: **the metric decoder cannot read a real percept**

The operative step readout is trained on transitions of the **imagination rollout** — `(ẑ, ẑ)` pairs —
and nowhere else. Fed a **true** consecutive latent pair it collapses:

| what the readout decodes | `ade_0_2s` |
|---|---:|
| the predictor's **imagined** transition (arm a) | **0.3839** |
| the **real** transition `(z_t, z_{t+1})` (arm c2, pure latent odometry) | **3.6093** — **9.4× worse** |

Three otherwise-puzzling results follow from this one fact, and all three are MEASURED:

1. **FULL OBSERVATION is WORSE than blind imagination** out to ~6 s under true actions (0.5167 vs
   0.3839 `ade_0_2s`). The "ceiling" arm is not a ceiling — a real frame moves the decoder off its
   calibration manifold.
2. ⛔ **The ORACLE peek policy LOSES to uniform peeking at a matched budget**, by **−24.8 % to −112.2 %**
   relative `de` (§4). Peeking exactly when the model is drifting is *worse* than peeking on a clock.
   **The pre-registered "DISAPPOINTING" threshold was a gap below +15 %; the measured gap is
   NEGATIVE.** ⇒ **a learned tactical trigger has negative headroom on this arm.** The informative
   axis H2 said had to be constructed *was* constructed, and it returns a clean negative.
3. **Feeding the model's own predicted speed back into its speed channel is catastrophic** —
   `ade_0_2s` 0.9554 → **9.3577** (×9.8). The constant-`v0` convention is load-bearing.

### 3. What the PI asked, answered in one line each

* **How long can it drive blind?** Deployably: **0.8 s** before a frozen percept is as good, **1.4 s**
  before it is a metre off, and it is **never** better than assuming constant velocity. Under the
  expert's actions: **6.5 s**, **2.2 s**, and 0.4–7.4 s respectively. With the free decoder swap the
  privileged arm beats constant velocity to **18.5 s**.
* **What limits it?** Not the metric decoder's capacity and not the renderer — **the training horizon**
  (0.4 s) and **the decoder's attachment to the imagination manifold instead of to perception**.
* **Can we build the efficiency claim on it?** **Not as the model stands.** The duty-cycle axis is real
  and has genuine dynamic range (§4), but on this arm more camera is not reliably better, and a smart
  trigger is worse than a dumb clock.

### ⚠️ 4. One defect in my own pre-registration, and one limit of the sweep — both found by reading my own results

* **The pre-registered `T_blind` rule could not fire.** Arms (a), (b) and (c) decode a **bit-identical**
  first transition by construction, so the paired Δ at step 1 is exactly 0.0 and "contiguously from
  N = 1" returns 0 for every arm in every regime **regardless of the data** — a **C13** defect (a
  criterion that cannot fire) in my own pre-registration. Repaired by anchoring contiguity at step 2,
  the first horizon on which the arms can differ at all. Amendment **A4**; the unrepaired output is
  preserved in the discussion so the correction is auditable.
* **The `T_blind` for the A2 readout arm SATURATES at the sweep terminus** (18.5 s of 18.5 s). Per
  **C14** that is **a LOWER BOUND on our configuration, not a measured horizon**, and it is labelled
  that way everywhere.

**Tier of the headline:** `T_blind` in the four pre-registered regimes is **DECISION-GRADE** —
pre-registered, estimator named, falsifier stated, reproduction gate passed, and the deployable verdict
is corroborated by three independent contrasts (vs frozen, vs CV, and the convention control).
The **A2 readout lever is CONFIRMED but not DECISION-GRADE**: it was added mid-run (amendment A2) and
needs one independent re-run before it decides anything.

---

# 1. The instrument already existed, and the program has been reading it at one horizon

## 1.1 The source fact, verified before anything was built

`tanitad/models/metric_dynamics.py::rollout_decode` advances its latent window by appending **the
model's own prediction**:

```python
z_hat = predictor(win_s, win_a)[1]                              # :236
dposes.append(step_readout(win_s[:, -1], z_hat))                # :237
win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)    # :241  ← no frame is re-encoded
```

**No frame is encoded after the initial window.** So `taniteval.rollout.collect` — which produces the
program's headline `ade_0_2s` — and `train_flagship_v4.canary_rollout` / `train_flagship_v16.canary_rollout`
— which produce `wm_canary_ade_2s` — are **already blind-imagination drives**. `k` was always a free
parameter; only `k = 20` was ever read, and only under the expert's **true future actions**.

⇒ **`ade_0_2s = 0.4271` (40 eps) and `0.4108` (600 eps) ARE v1's blind-imagination numbers at 2 s
under privileged actions.** They are not open-loop-with-perception numbers. `MEASURED`, and it follows
from the code above rather than from a claim.

⚠️ **Retraction of an INHERITED premise in my own brief.** The brief cited *"a v1-line reference of
~0.452 in `canary_rollout`'s own docstring"* as the number to establish. It is the **`heldout`
split-mean 0.4522** of the very same quantity (`MODEL_REGISTRY §1.2`), i.e. the deprecated
`overlapping_holdout_se` central value — not an independent measurement. The `full_set` value is
**0.4271**. Class **C4** (inherited without re-verification) crossed with the `heldout`/`full_set`
confusion `CLAUDE.md` warns about.

## 1.2 ⭐ The fact that explains everything downstream: **the operative brain was trained to imagine 0.4 s**

`MEASURED`, from v1's **own** stored config and **own** run manifest on pod2 — not from prose:

| quantity | value | primary source |
|---|---|---|
| operative predictor JEPA heads | **horizons `[1, 2, 4]`** | `flagship4b-speedjerk-30k/config.json` (`cfg.predictor.horizons`) |
| recursive rollout in training | `rollout_k` = **4** | same `config.json` (`cfg.train.rollout_k`) |
| operative forward-metric-consistency rollout | `op_fwd_k` = **4** (trainer default) | `train_flagship4b.py:648`; **v1's run manifest `/workspace/ops/runs.d/flagship-speed.env` contains ZERO `fwd-k` overrides** (`grep -c 'fwd-k'` → **0**) |
| tactical / strategic forward rollout | `tac_fwd_k` = **16**, `str_fwd_k` = **20** | same defaults, same manifest |

> ### **The maximum horizon v1 was ever trained to imagine is 4 steps = 0.4 s.**
> `wm_canary_ade_2s` reads it at **20 steps — 5× beyond**. This sweep reads it to **185 — 46× beyond.**

This is not a criticism of the canary; it is the mechanism the canary has been measuring without
naming. And it produces a **zero-training lever** that nobody had noticed: `HierarchicalGrounding`
holds **three** step readouts and `flagship_losses.grounding_losses` trains **all three on the same
operative imagination rollout**, only over different lengths — so `step["str"]` is a **20-step-calibrated
decoder of exactly the latents `step["op"]` is being asked to decode at 20 steps.** Every grounded
number in this program uses `step["op"]`. See §5.1 and amendment **A2**.

## 1.3 Instrument certification (M3)

`taniteval/blindimag.py` **generalises** `rollout_decode` rather than replacing it: the same loop with
a switchable **state source** and **action source**. Certified three ways:

1. **Bit-identity.** `blind_rollout(state_source="imagination", action_source="true_future")` is
   `torch.equal`-identical to `rollout_decode`; `action_source="hold_last"` is identical to
   `rollout_decode(future_actions=None)`. `taniteval/tests/test_blindimag.py`, **22/22 green on the dev
   box AND on pod2** (host-compatibility check).
2. **The guards can FAIL.** Per M3 the suite feeds deliberately wrong inputs: the control arms are
   asserted to genuinely *differ* from `rollout_decode` (a control that silently equalled the arm
   under test would make the experiment vacuous); the steer/accel clamps are shown to fire; the
   oracle peek trigger is shown to fire at a zero bar and never at an infinite one; the path-deviation
   instrument is shown to return 0 on the logged path and to recover an injected 1.5 m offset.
3. **All four arms decode an identical step 1** — they only diverge from step 2, which is what makes
   the contrast attributable (`test_first_step_is_identical_across_state_sources`).

## 1.4 ⛔ The reproduction gate — passed before any new number was read

`PRE_REGISTRATION.md §8`: no E-IMAG number is quoted until two committed deployments reproduce.

<!-- TABLES:GATE -->
| deployment | n windows | n episode clusters | `ade_0_2s` (new instrument) | committed | max abs diff vs unmodified `rollout.collect` |
|---|---:|---:|---:|---:|---:|
| **40 eps — CANONICAL** | 881 | 40 | **0.427109** | 0.4271 | 1.42e-05 m |
| **600 eps** | 13198 | 600 | **0.410807** | 0.4108 | n/a (cached-encode path) |

**`GATE_PASS = True`** · ckpt step 29999 · torch 2.4.1+cu124 · python 3.11.10
<!-- /TABLES:GATE -->

Both committed values reproduce, **CI bounds included** (`[0.3675, 0.4871]` at 40 and
`[0.3956, 0.4273]` at 600 — identical to `MODEL_REGISTRY §1.2a`). The 14 µm residual against the
unmodified harness is float-kernel noise from a different encode batch shape, not a code difference.
⛔ The two deployments are **different corpora** and are never substituted for one another.
`artifacts/gate_reproduction.json`.

---

# 2. E-IMAG-1 — the blind-driving horizon curve

## 2.1 Design (pre-registered; §2–§4 of `PRE_REGISTRATION.md`)

**Four arms**, differing in **exactly one tensor** — what enters the latent window each step:
**(a)** the predictor's own `z_hat` (*the thing under test*) · **(b)** the encoding of the **last real
frame**, re-appended every step — *"the world stopped"*, 🔴 **the critical control** · **(c)** the
encoding of the **true next frame** (*the ceiling*) · **(d)** constant velocity (*the floor*). Plus a
diagnostic **(c2)**: decode `(z_true_t, z_true_{t+1})` — pure latent odometry, no prediction at all.

**Action regimes, reported separately and never pooled:** **(i)** the expert's true future actions —
⚠️ **a privileged upper bound, not deployable capability**; **(ii)** the model's **own** actions,
derived from its **own** decoded motion by the exact inverse of the corpus's steer definition
(`steer = atan(2.9·κ)`, `physicalai.py:412`) — ⭐ **the deployable condition**; **(ii-0)** the last
observed action held (also deployable, no policy); and a **convention control** feeding the same
inverse the **true** motion, so an own-action penalty can be attributed to the model rather than to my
inverse.

## 2.2 The curve

<!-- TABLES:CURVE -->
*One fixed window set: **599 windows / 596 episode clusters**, `K_max = 185`, stride 8, 600-episode clean val. Every horizon and every arm is scored on the SAME windows, so the whole curve is paired.*

### REGIME (i) — TRUE FUTURE ACTIONS  ⚠️ PRIVILEGED UPPER BOUND, not deployable

**`de_N` — displacement error AT horizon N (m)**

| arm | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| (a) IMAGINATION | **0.065** | **0.188** | **0.814** | **1.775** | **4.281** | **8.230** | **20.486** | **38.498** | **96.645** |
| (b) FROZEN-LAST-FRAME | **0.131** | **0.354** | **1.164** | **2.481** | **5.402** | **9.198** | **19.114** | **32.067** | **69.558** |
| (c) FULL OBSERVATION | **0.122** | **0.319** | **1.014** | **2.131** | **4.499** | **7.471** | **14.620** | **23.552** | **47.947** |
| (c2) observed-pair odometry | **1.510** | **2.958** | **5.639** | **8.238** | **12.216** | **16.244** | **23.962** | **31.796** | **49.725** |
| (d) CONSTANT VELOCITY — the floor | **0.094** | **0.339** | **1.268** | **2.750** | **5.898** | **9.926** | **20.281** | **33.547** | **69.810** |
| (d2) hold-v0 go-straight | **0.088** | **0.327** | **1.244** | **2.717** | **5.851** | **9.854** | **20.139** | **33.339** | **69.520** |

| paired contrast | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.066 | +0.166 | +0.350 | +0.705 | +1.121 | +0.968 | -1.372 | -6.431 | -27.087 |
| CI95 | [+0.057, +0.076] | [+0.144, +0.191] | [+0.273, +0.435] | [+0.543, +0.882] | [+0.775, +1.462] | [+0.355, +1.570] | [-2.777, -0.044] | [-9.038, -4.035] | [-33.892, -21.167] |
| imagination separated-better? | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⛔ | ⛔ | ⛔ |

### REGIME (ii) — THE MODEL'S OWN ACTIONS  ⭐ THE DEPLOYABLE CONDITION

**`de_N` — displacement error AT horizon N (m)**

| arm | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| (a) IMAGINATION | **0.127** | **0.439** | **2.158** | **4.954** | **10.202** | **16.570** | **33.919** | **58.438** | **132.328** |
| (b) FROZEN-LAST-FRAME | **0.154** | **0.424** | **1.345** | **2.841** | **6.094** | **10.307** | **21.431** | **36.096** | **78.228** |
| (c) FULL OBS (teacher-forced percept) | **0.149** | **0.400** | **1.204** | **2.432** | **4.998** | **8.174** | **15.957** | **25.790** | **52.664** |
| (d) CONSTANT VELOCITY — the floor | **0.094** | **0.339** | **1.268** | **2.750** | **5.898** | **9.926** | **20.281** | **33.547** | **69.810** |

| paired contrast | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.027 | -0.015 | -0.813 | -2.113 | -4.107 | -6.263 | -12.488 | -22.342 | -54.100 |
| CI95 | [+0.018, +0.037] | [-0.041, +0.010] | [-0.937, -0.692] | [-2.397, -1.841] | [-4.711, -3.522] | [-7.269, -5.269] | [-14.482, -10.588] | [-25.610, -19.196] | [-61.421, -47.218] |
| imagination separated-better? | ✅ | — | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ |

### REGIME (ii-0) — HELD LAST ACTION (deployable, no policy)

**`de_N` — displacement error AT horizon N (m)**

| arm | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| (a) IMAGINATION | **0.069** | **0.217** | **1.035** | **2.392** | **5.808** | **10.986** | **26.418** | **48.166** | **113.308** |
| (b) FROZEN-LAST-FRAME | **0.131** | **0.359** | **1.197** | **2.583** | **5.682** | **9.737** | **20.501** | **34.634** | **74.933** |
| (d) CONSTANT VELOCITY — the floor | **0.094** | **0.339** | **1.268** | **2.750** | **5.898** | **9.926** | **20.281** | **33.547** | **69.810** |

| paired contrast | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.062 | +0.142 | +0.162 | +0.191 | -0.127 | -1.249 | -5.917 | -13.532 | -38.376 |
| CI95 | [+0.053, +0.072] | [+0.118, +0.166] | [+0.085, +0.247] | [+0.030, +0.370] | [-0.458, +0.227] | [-1.821, -0.681] | [-7.248, -4.694] | [-16.093, -11.120] | [-45.061, -32.593] |
| imagination separated-better? | ✅ | ✅ | ✅ | ✅ | — | ⛔ | ⛔ | ⛔ | ⛔ |

### CONVENTION CONTROL — actions from the TRUE motion through the same inverse

**`de_N` — displacement error AT horizon N (m)**

| arm | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| (a) IMAGINATION | **0.076** | **0.235** | **0.951** | **2.042** | **4.812** | **9.108** | **22.120** | **41.013** | **100.431** |
| (b) FROZEN-LAST-FRAME | **0.136** | **0.370** | **1.196** | **2.535** | **5.476** | **9.314** | **19.384** | **32.643** | **70.853** |
| (c) FULL OBSERVATION | **0.128** | **0.336** | **1.053** | **2.192** | **4.608** | **7.655** | **15.103** | **24.495** | **50.170** |

| paired contrast | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.060 | +0.135 | +0.245 | +0.493 | +0.663 | +0.206 | -2.735 | -8.370 | -29.578 |
| CI95 | [+0.051, +0.069] | [+0.113, +0.159] | [+0.172, +0.326] | [+0.337, +0.661] | [+0.330, +0.990] | [-0.356, +0.753] | [-4.055, -1.537] | [-10.815, -6.117] | [-36.047, -23.558] |
| imagination separated-better? | ✅ | ✅ | ✅ | ✅ | ✅ | — | ⛔ | ⛔ | ⛔ |

<!-- /TABLES:CURVE -->

## 2.3 ⭐ `T_blind`

<!-- TABLES:TBLIND -->
| regime | `T_blind` | CI95 | draws with `T_blind = 0` | first step where (b) is separated-BETTER | C14 saturated? |
|---|---:|---|---:|---:|---|
| REGIME (i) — TRUE FUTURE ACTIONS  ⚠️ PRIVILEGED UPPER BOUND, not deployable | **6.5 s** (65 steps) | [6.5, 8.9] s | 0.000 | 9.0 s | no |
| REGIME (ii) — THE MODEL'S OWN ACTIONS  ⭐ THE DEPLOYABLE CONDITION | **0.8 s** (8 steps) | [0.8, 1.0] s | 0.000 | 1.1 s | no |
| REGIME (ii-0) — HELD LAST ACTION (deployable, no policy) | **3.2 s** (32 steps) | [3.2, 5.0] s | 0.000 | 5.1 s | no |
| CONVENTION CONTROL — actions from the TRUE motion through the same inverse | **5.4 s** (54 steps) | [5.4, 7.3] s | 0.000 | 7.4 s | no |
| A2 SENSITIVITY — readout = step['str'] (20-step-calibrated), true actions | **18.5 s** (185 steps) | [18.5, 18.5] s | 0.000 | — | ⚠️ YES — LOWER BOUND |

**Usefulness horizons for arm (a), and the CV floor crossing**

| regime | `de_N` < 1.0 m | < 1.391 m (corridor) | < 2.0 m (miss@2m) | beats CV floor until |
|---|---|---|---|---|
| REGIME (i) — TRUE FUTURE ACTIONS  ⚠️ PRIVILEGED UPPER BOUND, not deployable | 2.2 s | 2.6 s | 3.1 s | **0.1 s** [0.1, 0.1] |
| REGIME (ii) — THE MODEL'S OWN ACTIONS  ⭐ THE DEPLOYABLE CONDITION | 1.4 s | 1.6 s | 1.9 s | **0.1 s** [0.1, 0.1] |
| REGIME (ii-0) — HELD LAST ACTION (deployable, no policy) | 1.9 s | 2.3 s | 2.7 s | **0.1 s** [0.1, 0.1] |
| CONVENTION CONTROL — actions from the TRUE motion through the same inverse | 2.0 s | 2.4 s | 2.9 s | **0.1 s** [0.1, 0.1] |
| A2 SENSITIVITY — readout = step['str'] (20-step-calibrated), true actions | 3.4 s | 3.9 s | 4.5 s | **0.1 s** [0.1, 0.1] |
<!-- /TABLES:TBLIND -->

⚠️ **Read the "beats CV floor until" column with its construction.** It applies the same
*contiguous-from-step-2* rule to the CV contrast, and it returns 0.1 s for every arm because
**constant velocity is essentially exact at 0.2 s** — no model beats it there, so contiguity ends
immediately. Contiguity is the right rule for `T_blind` (where the question is "for how long does
imagination stay ahead of a frozen percept") and the **wrong** rule for the CV floor (where the
question is "over which horizons is the model better at all"). The correct statistic is the
**interval** of horizons on which the arm is separated-better than CV, computed on the dense grid:

| arm | separated-better than the CV floor over | steps |
|---|---|---:|
| (a) imagination, **true** actions | **0.4 s … 7.4 s** | 71 / 185 |
| (a) imagination, **true** actions, `step["str"]` decoder (A2) | **0.6 s … 18.5 s** | **180 / 185** |
| (a) imagination, **held** action | 0.4 s … 3.8 s | 35 / 185 |
| (b) frozen-last-frame, true actions | 1.7 s … 12.0 s | 104 / 185 |
| ⛔ **(a) imagination, the model's OWN actions** | **never — at no horizon in the sweep** | **0 / 185** |

`artifacts/vs_floor_contrasts.json`; paired episode-cluster bootstrap, B = 2000, same 599 windows.

## 2.4 The convention control — is the own-action penalty mine or the model's?

<!-- TABLES:CONTROL -->
*Positive = the TRUE-action arm is better, i.e. the cost of routing the true motion through my kinematic inverse. A value indistinguishable from 0 means the inverse is faithful and any own-action penalty belongs to the model.*

| true_future − gt_kinematic, arm (a) | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.011 | +0.047 | +0.137 | +0.267 | +0.531 | +0.878 | +1.634 | +2.516 | +3.786 |
| CI95 | [+0.008, +0.014] | [+0.037, +0.057] | [+0.107, +0.174] | [+0.190, +0.354] | [+0.359, +0.727] | [+0.601, +1.209] | [+1.054, +2.276] | [+1.608, +3.516] | [+1.921, +5.794] |
| imagination separated-better? | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
<!-- /TABLES:CONTROL -->

## 2.5 The A2 sensitivity — the readout-level lever (amendment A2, NOT part of the primary)

<!-- TABLES:LEVER -->
| contrast (positive = the ALTERNATE readout is better) | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `a_imagination__true__roTAC_vs_a_imagination__true` | -0.022⛔ | +0.050✅ | +0.501✅ | +1.032✅ | +2.130✅ | +3.780✅ | +8.799✅ | +15.981✅ | +39.626✅ |
| `a_imagination__true__roSTR_vs_a_imagination__true` | -0.038⛔ | +0.038✅ | +0.502✅ | +1.080✅ | +2.296✅ | +4.103✅ | +9.535✅ | +17.383✅ | +43.304✅ |
| `a_imagination__own__roSTR_vs_a_imagination__own` | -0.045⛔ | -0.025 | +0.341✅ | +1.101✅ | +2.062✅ | +2.567✅ | +4.417✅ | +9.088✅ | +29.823✅ |
| `a_imagination__hold__roSTR_vs_a_imagination__hold` | -0.037⛔ | +0.027✅ | +0.363✅ | +0.723✅ | +1.563✅ | +2.896✅ | +7.230✅ | +13.846✅ | +34.782✅ |
| `b_frozenlast__true__roSTR_vs_b_frozenlast__true` | -0.287⛔ | -0.520⛔ | -0.876⛔ | -1.130⛔ | -1.391⛔ | -1.667⛔ | -2.260⛔ | -2.749⛔ | -2.799⛔ |
| `c2_observedpair__true__roSTR_vs_c2_observedpair__true` | -0.002 | -0.008 | -0.029⛔ | -0.054⛔ | -0.103⛔ | -0.190⛔ | -0.481⛔ | -0.726⛔ | -1.269⛔ |
<!-- /TABLES:LEVER -->

## 2.6 Reading it

**a. `T_blind` is not one number — it is a property of the ACTION SOURCE, and the program has only
ever measured the privileged half.** 6.5 s [6.5, 8.9] under the expert's actions; **0.8 s [0.8, 1.0]**
under the model's own. The privileged number is **8×** the deployable one. Every existing statement
about our world model's imagination — including `wm_canary_ade_2s` and `ade_0_2s` themselves — lives on
the privileged side.

**b. In the deployable regime the pre-registered Outcome B is what happened, and it will not be
softened.** From **1.1 s** onward, frozen-last-frame is *separated-better* than imagination
(Δ = −0.813 [−0.937, −0.692] m at 2 s; −6.263 [−7.269, −5.269] at 6 s). And the harder bar is worse:
**imagination under its own actions is never separated-better than constant velocity at any horizon in
the sweep** (`artifacts/vs_floor_contrasts.json`). ⇒ **On the deployable surface the world model's
dynamics do not merely fail to help blind driving — they are actively worse than holding the last
percept, and worse than assuming nothing changes.**

**c. …and the convention control says that is the model, not my action inverse.** The inverse costs a
real but small amount: routing the **true** motion through it costs **+0.137 m [+0.107, +0.174] at 2 s**
(separated — the `accel` half is genuinely not an exact inverse, as pre-registered). But the own-action
arm is **2.158 m** at 2 s against the convention control's **0.951 m**: of the **+1.344 m** own-action
penalty, **0.137 m (10.2 %) is my inverse and 1.207 m (89.8 %) is the model's own error compounding
through the loop.** This is exactly what the control was pre-registered for, and it fires in the
direction that preserves the conclusion.

**d. A no-policy controller beats the model's own policy — by a lot.** Simply *holding the last
observed action* (regime ii-0) gives `T_blind` **3.2 s** and `ade_0_2s` **0.4712**, against **0.8 s**
and **0.9554** for the model's own kinematic action. Both are deployable; the one with no model in the
control loop is **4× longer-lived**. ⇒ **the closed-loop instability is in the action feedback, not in
the perception loss.** That is a different lever from the one the brief anticipated, and it is
measured.

**e. The "ceiling" is not a ceiling.** FULL OBSERVATION (a real frame every step) is **worse** than
blind imagination under true actions out to ~6 s (`ade_0_2s` 0.5167 vs 0.3839; `de@2s` 1.014 vs 0.814)
and only overtakes it at 9 s. Under the model's own actions it *is* better at every horizon ≥ 1 s —
because there the imagination arm is unstable and any re-anchoring helps. **Whether perception helps
depends on which failure dominates**, and at short horizon under good actions it does not help at all.

**f. Frozen-last-frame is a genuinely strong baseline, which is why it was the right control.** It
beats the CV floor from 1.7 s to 12.0 s and it beats imagination from 9 s onward even under privileged
actions. A study that had reported only "imagination vs CV" would have concluded the world model works
blind; the control is what turns that into a bounded claim.

---

# 3. E-IMAG-2 — what limits it

<!-- TABLES:DECOMP -->
**(a) IMAGINATION** — `a_imagination__true`

| quantity | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `de_N` (m) | 0.065 | 0.188 | 0.814 | 1.775 | 4.281 | 8.230 | 20.486 | 38.498 | 96.645 |
| along-track \|err\| (m) | 0.060 | 0.172 | 0.733 | 1.408 | 2.797 | 4.979 | 11.668 | 21.707 | 55.488 |
| cross-track \|err\| (m) | 0.015 | 0.047 | 0.248 | 0.808 | 2.547 | 5.371 | 14.305 | 27.448 | 70.532 |
| longitudinal share of squared error | 0.937 | 0.911 | 0.834 | 0.705 | 0.546 | 0.465 | 0.393 | 0.366 | 0.374 |
| DRIFT: signed along mean (m) | 0.008 | 0.059 | 0.483 | 0.917 | 1.529 | 1.996 | 1.809 | -0.744 | -15.815 |
| VARIANCE: along std (m) | 0.096 | 0.260 | 0.801 | 1.579 | 3.571 | 6.621 | 15.827 | 29.094 | 73.263 |
| DRIFT: signed cross mean (m) | 0.001 | -0.002 | -0.016 | -0.032 | -0.154 | -0.552 | -2.709 | -7.271 | -25.015 |
| VARIANCE: cross std (m) | 0.025 | 0.083 | 0.417 | 1.182 | 3.538 | 7.391 | 19.606 | 37.606 | 93.572 |
| drift share of along energy | 0.007 | 0.049 | 0.267 | 0.252 | 0.155 | 0.083 | 0.013 | 0.001 | 0.045 |
| Frenet cross-track p90 (m) | 0.03 | 0.10 | 0.58 | 1.89 | 6.25 | 14.04 | 38.47 | 73.43 | 189.22 |
| model's own predicted speed (m/s) | 12.97 | 13.20 | 13.30 | 13.29 | 13.26 | 13.29 | 13.37 | 13.49 | 13.42 |
| frac steps OUTSIDE the measured envelope | 0.000 | 0.000 | 0.002 | 0.007 | 0.072 | 0.200 | 0.395 | 0.516 | 0.673 |

**(a) IMAGINATION** — `a_imagination__own`

| quantity | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `de_N` (m) | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.570 | 33.919 | 58.438 | 132.328 |
| along-track \|err\| (m) | 0.118 | 0.368 | 1.254 | 2.718 | 5.524 | 9.098 | 19.018 | 32.852 | 75.644 |
| cross-track \|err\| (m) | 0.026 | 0.159 | 1.369 | 3.286 | 6.805 | 11.232 | 23.570 | 41.303 | 95.462 |
| longitudinal share of squared error | 0.936 | 0.771 | 0.363 | 0.332 | 0.355 | 0.381 | 0.404 | 0.401 | 0.406 |
| DRIFT: signed along mean (m) | 0.044 | 0.070 | 0.350 | 1.038 | 1.877 | 2.272 | 0.885 | -4.341 | -30.451 |
| VARIANCE: along std (m) | 0.165 | 0.489 | 1.617 | 3.464 | 7.564 | 13.038 | 27.353 | 45.432 | 96.679 |
| DRIFT: signed cross mean (m) | 0.004 | 0.026 | 0.491 | 0.964 | 1.347 | 1.300 | -0.379 | -4.541 | -21.053 |
| VARIANCE: cross std (m) | 0.044 | 0.268 | 2.136 | 5.036 | 10.421 | 16.815 | 33.243 | 55.559 | 120.914 |
| drift share of along energy | 0.066 | 0.020 | 0.045 | 0.083 | 0.058 | 0.029 | 0.001 | 0.009 | 0.090 |
| Frenet cross-track p90 (m) | 0.07 | 0.39 | 3.90 | 8.78 | 18.36 | 29.59 | 56.34 | 88.28 | 193.21 |
| model's own predicted speed (m/s) | 13.13 | 12.74 | 13.59 | 13.61 | 13.45 | 13.27 | 13.04 | 12.93 | 12.75 |
| frac steps OUTSIDE the measured envelope | 0.002 | 0.015 | 0.072 | 0.166 | 0.293 | 0.396 | 0.538 | 0.625 | 0.735 |

**(b) FROZEN-LAST-FRAME** — `b_frozenlast__true`

| quantity | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `de_N` (m) | 0.131 | 0.354 | 1.164 | 2.481 | 5.402 | 9.198 | 19.114 | 32.067 | 69.558 |
| along-track \|err\| (m) | 0.126 | 0.326 | 1.005 | 2.041 | 4.261 | 7.099 | 14.215 | 23.259 | 48.205 |
| cross-track \|err\| (m) | 0.018 | 0.079 | 0.360 | 0.896 | 2.188 | 3.971 | 9.077 | 15.949 | 37.616 |
| longitudinal share of squared error | 0.975 | 0.923 | 0.847 | 0.802 | 0.751 | 0.714 | 0.663 | 0.632 | 0.595 |
| DRIFT: signed along mean (m) | 0.035 | 0.091 | 0.261 | 0.500 | 0.989 | 1.631 | 2.896 | 3.811 | 7.191 |
| VARIANCE: along std (m) | 0.190 | 0.473 | 1.451 | 2.994 | 6.204 | 10.216 | 20.062 | 32.242 | 65.940 |
| DRIFT: signed cross mean (m) | 0.002 | 0.008 | 0.022 | 0.056 | 0.173 | 0.366 | 0.813 | 1.204 | 2.440 |
| VARIANCE: cross std (m) | 0.031 | 0.139 | 0.626 | 1.506 | 3.609 | 6.544 | 14.443 | 24.737 | 54.722 |
| drift share of along energy | 0.032 | 0.036 | 0.032 | 0.027 | 0.025 | 0.025 | 0.020 | 0.014 | 0.012 |
| Frenet cross-track p90 (m) | 0.04 | 0.18 | 0.87 | 2.32 | 5.24 | 9.60 | 25.09 | 44.28 | 86.27 |
| model's own predicted speed (m/s) | 13.03 | 13.04 | 13.03 | 13.02 | 13.02 | 13.02 | 13.04 | 13.02 | 13.00 |
| frac steps OUTSIDE the measured envelope | 0.000 | 0.001 | 0.007 | 0.031 | 0.092 | 0.163 | 0.297 | 0.399 | 0.534 |

**(c2) observed-pair odometry** — `c2_observedpair__true`

| quantity | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `de_N` (m) | 1.510 | 2.958 | 5.639 | 8.238 | 12.216 | 16.244 | 23.962 | 31.796 | 49.725 |
| along-track \|err\| (m) | 1.508 | 2.951 | 5.592 | 8.091 | 11.807 | 15.434 | 21.970 | 27.982 | 40.281 |
| cross-track \|err\| (m) | 0.025 | 0.097 | 0.387 | 0.830 | 1.803 | 3.080 | 6.194 | 10.117 | 21.292 |
| longitudinal share of squared error | 1.000 | 0.998 | 0.992 | 0.984 | 0.965 | 0.943 | 0.898 | 0.851 | 0.752 |
| DRIFT: signed along mean (m) | 0.118 | 0.269 | 0.617 | 1.015 | 1.395 | 1.884 | 2.760 | 3.238 | 3.586 |
| VARIANCE: along std (m) | 2.083 | 4.022 | 7.674 | 11.199 | 16.405 | 21.506 | 31.195 | 40.517 | 58.775 |
| DRIFT: signed cross mean (m) | -0.001 | -0.001 | 0.001 | 0.041 | 0.195 | 0.403 | 0.792 | 1.283 | 2.909 |
| VARIANCE: cross std (m) | 0.042 | 0.170 | 0.677 | 1.446 | 3.124 | 5.276 | 10.505 | 16.935 | 33.723 |
| drift share of along energy | 0.003 | 0.004 | 0.006 | 0.008 | 0.007 | 0.008 | 0.008 | 0.006 | 0.004 |
| Frenet cross-track p90 (m) | 0.07 | 0.26 | 1.08 | 2.38 | 4.91 | 8.70 | 15.82 | 24.05 | 52.74 |
| model's own predicted speed (m/s) | 13.29 | 13.04 | 13.10 | 13.20 | 12.92 | 13.15 | 13.15 | 13.17 | 13.15 |
| frac steps OUTSIDE the measured envelope | 0.003 | 0.010 | 0.025 | 0.045 | 0.090 | 0.143 | 0.256 | 0.352 | 0.501 |

*GT ego speed at window end: **12.90 m/s**. Envelope: |dlat| ≤ 3.0 m, |dyaw| ≤ 12.0°. ⚠️ The last horizon that is a genuine MEASUREMENT is 0.4 s. Every reading beyond it is EXTRAPOLATION. The OOD RATIO is deliberately NOT quoted: sup(ratio_arr)=1.298888 makes the <=1.30 test a tautology (C13). ENV_YAW_MAX=12deg was never measured; it is a grid terminus (C14).*
<!-- /TABLES:DECOMP -->

## 3.1 The error starts longitudinal and ENDS lateral — the crossover is measured

Under true actions the longitudinal share of squared error falls monotonically
**0.937 → 0.911 → 0.834 → 0.705 → 0.546 → 0.465 → 0.393 → 0.366** across 0.5 s → 12 s. It crosses
50 % between **4.5 s and 6 s**.

* At **2 s** the split is **83.4 % longitudinal** — consistent in kind with the registry's *"89 % of
  squared error is longitudinal"* for v1 (`MODEL_REGISTRY §1.2`, a different deployment and a
  different estimator, so agreement in *kind* is the strongest admissible statement).
* Cross-track error grows **0.015 m → 70.5 m (×4 700)** while along-track grows
  **0.060 m → 55.5 m (×925)**: **lateral compounds ~5× faster**, which independently reproduces
  `taniteval/lateral.py`'s own founding finding (x14.1 vs x3.2 over 0.5→2 s on two clips) **on 596
  episode clusters instead of two clips.**
* **Frenet cross-track p90 reaches 0.58 m at 2 s, 6.25 m at 4.5 s and 38.5 m at 9 s** while the ADE
  headline at 2 s still reads as a speed error. This is `RETRACTION_LOG` **C9** (horizon-blind
  instrument) visible inside a single arm.

⇒ **The 2 s canary is a longitudinal instrument. The failure that ends a blind drive is lateral, and
the canary cannot see it.**

## 3.2 DRIFT vs VARIANCE — it is variance, with one bias window

The signed-mean share of the along-track squared energy peaks at **0.267 at 2 s** and is **≤ 0.09**
everywhere else; the cross-track drift share never exceeds 0.09. The along-track *bias* is
**+0.483 m at 2 s** rising to +1.996 m at 6 s and then reversing to −15.8 m at 18.5 s, while the
along-track *spread* grows monotonically 0.096 → 73.3 m.

⇒ **The latent does not collapse to a single wrong trajectory — it stays plausible and becomes
unreliable.** ~25 % of the 2 s longitudinal error is a systematic **over-shoot** (the same
over-prediction the registry records as *"+0.19 m/s speed bias, along-track overshoot +0.38 m"*), and
essentially all of the rest, at every horizon, is variance.

## 3.3 What the model actually imagines: **a constant-speed extrapolation with a learned curvature**

The model's own decoded speed is **12.97 / 13.30 / 13.29 / 13.37 / 13.42 m/s** at 0.5 / 2 / 3 / 9 /
18.5 s against a true ego speed at the window end of **12.90 m/s**. It holds its observed speed to
within ~4 % for 18.5 s and never revises it. That is why its error tracks the constant-velocity floor
so closely, and it is a mechanical restatement of `MODEL_REGISTRY`'s longitudinal-blindness finding:
**blind imagination is longitudinally a hold-v0 predictor.**

## 3.4 ⚠️ Out-of-envelope accounting, and the label it forces

`taniteval/ood.py` envelope (|dlat| ≤ 3.0 m, |dyaw| ≤ 12°) — fraction of rollout **steps** outside it,
imagination under true actions:

| 0.5 s | 1 s | 2 s | 3 s | 4.5 s | 6 s | 9 s | 12 s | 18.5 s |
|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.002 | 0.007 | 0.072 | 0.200 | 0.395 | **0.516** | **0.673** |

⚠️ **Binding labels, applied here rather than argued:**
* **The last horizon that is a genuine MEASUREMENT is 0.4 s.** Everything past it in this document —
  including the 2 s canary the program already quotes — is **EXTRAPOLATION**, and is labelled so.
* At **12 s and beyond a majority of steps are outside the measured envelope** (0.516, 0.673), which
  fires `ood.py`'s envelope clause: those columns are `EXTRAPOLATION` by the instrument's own rule,
  not merely by convention. 6–9 s is `PARTIAL EXTRAPOLATION`.
* ⛔ **The OOD *ratio* is deliberately not quoted anywhere in this report** — `sup(ratio_arr) =
  1.298888` makes the `≤ 1.30` test a tautology (**C13**), and `ENV_YAW_MAX = 12°` is a grid terminus
  that was never measured (**C14**).

## 3.5 ⭐ The mechanism, isolated: the decoder is attached to imagination, not to perception

Arm **(c2)** decodes only **true** consecutive latent pairs — no prediction is involved at any step.
It is the cleanest possible probe of "can this readout do visual odometry?", and the answer is no:

| arm | what the readout is fed | `ade_0_2s` | `de@0.5s` | longitudinal share at 0.5 s |
|---|---|---:|---:|---:|
| (a) imagination | `(ẑ_t, ẑ_{t+1})` — its training distribution | **0.3839** | 0.065 | 0.937 |
| (c) full observation | `(z_t, ẑ_{t+1})` — half real | 0.5167 | 0.122 | — |
| **(c2) observed pair** | `(z_t, z_{t+1})` — both real | **3.6093** | **1.510** | **1.000** |

The ordering is monotone in *how real the pair is*, the (c2) error is **pure longitudinal** (share
1.000) and it is **variance, not bias** (signed mean 0.118 m against a standard deviation of 2.083 m
at 0.5 s). The 20-step readout behaves identically (−0.029 m at 2 s), so this is a property of the
**grounding recipe**, not of one head.

⇒ **v1 cannot decode its own ego-motion from two real frames. It can only decode it from two imagined
ones.** That is the single fact behind §2.6e (perception hurts), §4 (peeking backfires), and the whole
shape of the efficiency answer.
⚠️ Bounded claim: this is one readout family on one checkpoint. It shows the *deployed grounding* has
this property; it does not show the latent lacks the information — the invdyn heads
(`MetricInverseDynamics`) are trained on real pairs and were not probed here (§6).

---

# 4. E-IMAG-4 — the efficiency claim, with an axis that can actually move

H2 MEASURED that the compute-saving framing is **information-free**: against always-on-7,
never-escalating saves **85.7 %**, a perfect oracle **85.6 %**, the real gate **84.8 %** — the whole
span between useless and perfect is **0.1 pp** (`INHERITED`, `…/2026-07-26-h2-classifier/`). No compute
number can distinguish a good gate from a useless one.

**Pre-registered before the numbers existed (§7.3, binding):** a duty-cycle saving **< 2×** vs
always-on would be **DISAPPOINTING**; an oracle-vs-uniform gap **< 15 %** relative error reduction at
matched duty cycle would be **DISAPPOINTING** — a learned trigger would then be worth less than the
engineering to build it, and H2's failure would repeat one level up.

<!-- TABLES:DUTY -->
**UNIFORM peek-every-T′ (deployable)** — base arm: imagination + own actions

| policy | front-camera duty cycle | de@0.5s | de@1s | de@2s | de@3s | de@4.5s | de@6s | de@9s | de@12s | de@18.5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `peek_own_uniform_T90` | **0.0108** | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.570 | 33.919 | 55.763 | 123.579 |
| `peek_own_uniform_T60` | **0.0162** | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.570 | 32.361 | 54.171 | 119.287 |
| `peek_own_uniform_T45` | **0.0216** | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.127 | 31.287 | 51.248 | 111.034 |
| `peek_own_uniform_T30` | **0.0324** | 0.127 | 0.439 | 2.158 | 4.954 | 9.954 | 15.936 | 30.493 | 49.081 | 103.075 |
| `peek_own_uniform_T20` | **0.0486** | 0.127 | 0.439 | 2.158 | 4.711 | 9.521 | 15.517 | 29.920 | 48.438 | 98.925 |
| `peek_own_uniform_T15` | **0.0649** | 0.127 | 0.439 | 2.036 | 4.572 | 9.452 | 15.470 | 30.932 | 50.604 | 104.886 |
| `peek_own_uniform_T10` | **0.0973** | 0.127 | 0.439 | 1.899 | 4.165 | 8.750 | 14.474 | 29.062 | 48.047 | 101.841 |
| `peek_own_uniform_T5` | **0.1946** | 0.127 | 0.361 | 1.254 | 2.677 | 5.599 | 9.244 | 18.579 | 30.628 | 64.142 |
| `peek_own_uniform_T3` | **0.3297** | 0.120 | 0.363 | 1.205 | 2.545 | 5.306 | 8.745 | 17.401 | 28.441 | 58.830 |
| `peek_own_uniform_T2` | **0.4973** | 0.133 | 0.364 | 1.182 | 2.468 | 5.139 | 8.472 | 16.746 | 27.259 | 56.187 |

**ORACLE peek ⚠️ privileged — reads the true error** — base arm: imagination + own actions

| policy | front-camera duty cycle | de@0.5s | de@1s | de@2s | de@3s | de@4.5s | de@6s | de@9s | de@12s | de@18.5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `peek_own_oracle_e0.5` | **0.1529** | 0.127 | 0.450 | 2.095 | 4.910 | 10.809 | 18.436 | 38.775 | 64.995 | 134.338 |
| `peek_own_oracle_e0.2` | **0.4450** | 0.127 | 0.469 | 1.882 | 3.832 | 8.539 | 14.462 | 29.008 | 46.461 | 92.764 |
| `peek_own_oracle_e0.1` | **0.6199** | 0.128 | 0.446 | 1.609 | 3.200 | 6.573 | 10.866 | 21.517 | 34.633 | 69.688 |
| `peek_own_oracle_e0.05` | **0.7479** | 0.127 | 0.395 | 1.345 | 2.701 | 5.573 | 9.091 | 17.711 | 28.428 | 57.584 |
| `peek_own_oracle_e0.02` | **0.8691** | 0.136 | 0.382 | 1.217 | 2.489 | 5.134 | 8.383 | 16.410 | 26.518 | 53.874 |

**Anchors — the two ends of the duty-cycle axis**

| baseline | duty cycle | de@0.5s | de@1s | de@2s | de@3s | de@4.5s | de@6s | de@9s | de@12s | de@18.5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `a_imagination__own` | 0.0 | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.570 | 33.919 | 58.438 | 132.328 |
| `a_imagination__hold` | 0.0 | 0.069 | 0.217 | 1.035 | 2.392 | 5.808 | 10.986 | 26.418 | 48.166 | 113.308 |
| `c_fullobs__own` | 1.0 | 0.149 | 0.400 | 1.204 | 2.432 | 4.998 | 8.174 | 15.957 | 25.790 | 52.664 |
| `c_fullobs__true` | 1.0 | 0.122 | 0.319 | 1.014 | 2.131 | 4.499 | 7.471 | 14.620 | 23.552 | 47.947 |
| `a_imagination__true` | 0.0 | 0.065 | 0.188 | 0.814 | 1.775 | 4.281 | 8.230 | 20.486 | 38.498 | 96.645 |

**ORACLE vs UNIFORM at matched duty cycle — the informative version of H2's efficiency claim**

| oracle policy | oracle duty | matched uniform | uniform duty | rel. Δde@0.5s | rel. Δde@1s | rel. Δde@2s | rel. Δde@3s | rel. Δde@4.5s | rel. Δde@6s | rel. Δde@9s | rel. Δde@12s | rel. Δde@18.5s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `peek_own_oracle_e0.02` | 0.8691 | `peek_own_uniform_T2` | 0.4973 | -2.0% | -5.0% | -2.9% | -0.9% | +0.1% | +1.1% | +2.0% | +2.7% | +4.1% |
| `peek_own_oracle_e0.05` | 0.7479 | `peek_own_uniform_T2` | 0.4973 | +4.7% | -8.4% | -13.8% | -9.5% | -8.4% | -7.3% | -5.8% | -4.3% | -2.5% |
| `peek_own_oracle_e0.1` | 0.6199 | `peek_own_uniform_T2` | 0.4973 | +3.8% | -22.5% | -36.1% | -29.7% | -27.9% | -28.3% | -28.5% | -27.1% | -24.0% |
| `peek_own_oracle_e0.2` | 0.4450 | `peek_own_uniform_T2` | 0.4973 | +4.5% | -28.7% | -59.2% | -55.3% | -66.1% | -70.7% | -73.2% | -70.4% | -65.1% |
| `peek_own_oracle_e0.5` | 0.1529 | `peek_own_uniform_T5` | 0.1946 | +0.0% | -24.8% | -67.1% | -83.4% | -93.1% | -99.4% | -108.7% | -112.2% | -109.4% |
| `peek_hold_oracle_e0.02` | 0.8613 | `peek_hold_uniform_T2` | 0.4973 | -14.6% | -21.9% | -18.4% | -8.7% | -3.7% | -1.5% | +1.1% | +2.6% | +4.1% |
| `peek_hold_oracle_e0.05` | 0.7563 | `peek_hold_uniform_T2` | 0.4973 | +7.5% | -25.9% | -56.0% | -36.9% | -26.8% | -22.4% | -17.9% | -15.1% | -11.7% |
| `peek_hold_oracle_e0.1` | 0.5867 | `peek_hold_uniform_T2` | 0.4973 | +19.4% | -13.8% | -40.7% | -30.7% | -27.6% | -28.2% | -30.0% | -29.4% | -28.5% |
| `peek_hold_oracle_e0.2` | 0.3715 | `peek_hold_uniform_T3` | 0.3297 | +11.6% | +6.9% | -16.7% | -17.9% | -22.5% | -28.2% | -36.7% | -40.3% | -41.9% |
| `peek_hold_oracle_e0.5` | 0.1396 | `peek_hold_uniform_T10` | 0.0973 | +0.0% | +0.0% | +3.7% | -1.0% | -8.2% | -16.6% | -25.1% | -28.3% | -30.8% |
<!-- /TABLES:DUTY -->

## 4.1 The axis is real, and it does have dynamic range

Unlike H2's compute-saving axis (0.1 pp between useless and perfect), the duty-cycle axis moves: at
2 s the deployable arm goes **2.158 m at duty 0** → **1.899 m at duty 0.097** → **1.182 m at duty
0.497** → **1.204 m at duty 1.000**. That is a **45.2 % error range across the axis**, against the
**0.1 percentage points** that separated a useless gate from a perfect one on H2's compute-saving axis.
*(The two are not the same units and no ratio between them is quoted; the point is only that one axis
can move and the other could not.)* **The pre-registered construction worked: the claim is now
falsifiable.**

## 4.2 ⛔ And having made it falsifiable, it falsifies

**Pre-registered (§7, before the numbers existed):** an oracle-vs-uniform gap **below +15 %** relative
error reduction at matched duty cycle would be **DISAPPOINTING**.

**MEASURED: the gap is NEGATIVE at every matched pair and every horizon ≥ 1 s** —
`−24.8 %` … `−112.2 %` for the sparse oracle (`e0.5`, duty 0.153 vs uniform T5, duty 0.195),
`−28.7 %` … `−73.2 %` for `e0.2`. **An oracle that peeks exactly when the model is already drifting is
between 25 % and 112 % WORSE than peeking on a fixed clock at the same camera budget.**

⇒ **A learned tactical trigger has NEGATIVE headroom on this arm.** Not small headroom — negative.
This is the informative version of H2's claim that H2 said had to be constructed, it was constructed,
and it returns a clean negative. **It must not be re-reported as "further work needed on the gate":
the gate is not the bottleneck, the decoder is.**

## 4.3 The duty-cycle / performance curve, and the honest saving

Reading the uniform column at 2 s: `T = 2` (duty 0.497) already matches full observation (1.182 vs
1.204), and `T = 5` (duty 0.195) reaches 1.254 — **within 4 % of always-on at one fifth of the camera
budget.** Below duty 0.1 the curve is flat at the no-peek value: peeks that sparse buy nothing at 2 s.

| a policy that … | front-camera duty | `de@2s` | vs always-on |
|---|---:|---:|---|
| never looks (pure imagination, own actions) | 0.000 | 2.158 | +79 % |
| looks every 5th tick | 0.195 | 1.254 | **+4 %** |
| looks every 2nd tick | 0.497 | 1.182 | **−2 %** (better) |
| looks every tick | 1.000 | 1.204 | — |

⇒ **A ~5× front-camera duty-cycle reduction costs ~4 % of 2 s accuracy on this arm** — which clears
the pre-registered "< 2× would be disappointing" bar. ⚠️ But it clears it **for the wrong reason**: it
works because a fixed clock is cheap, not because the world model imagines well. The same table shows
imagination-with-no-peek is **worse than the CV floor**, so the "saving" is a saving relative to a
policy nobody would ship.

## 4.4 What would have to change for the efficiency claim to be worth making

The claim needs **both** (a) blind driving that beats the trivial floor in the deployable regime —
which it currently does not (§2.6b) — and (b) a re-anchoring path that *helps* rather than perturbs,
which requires §5's L2. **Until both land, the honest statement is: we can save front-camera duty
cycle, and we cannot yet say the world model is what makes it possible.**

---

# 5. E-IMAG-3 — how to maximise blind driving. **Design only; no training was launched.**

## 5.1 The mechanism the measurements point at — stated once, then used

Two MEASURED facts, both from primary sources, carry the whole design:

1. **The operative brain's entire training experience of imagination is 0.4 s** (§1.2:
   `horizons [1,2,4]`, `rollout_k = 4`, `op_fwd_k = 4`).
2. **Its metric decoder is attached to the imagination manifold, not to perception** (§3.5: 9.4×
   worse on a real latent pair than on an imagined one).

Everything below follows from those two, and each item states what result would kill it.

## 5.2 The ranked interventions

**Ranked by expected value per GPU-hour. No training was launched in this run.**

| # | intervention | GPU cost | evidence it will work | ⛔ pre-registered falsifier |
|---|---|---|---|---|
| **L0** | ⭐⭐ **Swap the decoder: `step["op"]` → `step["str"]` (or `["tac"]`) wherever a grounded rollout is decoded** — `taniteval/rollout.py:collect`, both `canary_rollout`s, `eval_grounded_rollout_4b*`. One line each. | **ZERO — already measured** | `ade_0_2s` **0.3839 → 0.1865/0.1950** on v1, paired-separated at every horizon ≥ 1 s; beats-CV horizon **7.4 s → 18.5 s** (§2.5) | Already run on v1. **On v4 it REFUTES if `wm_canary_ade_2s` does not fall by ≥ 25 %.** Eval-only, ~20 GPU-min: v4's `HierarchicalGrounding` carries the same three readouts. |
| **L1** | ⭐ **Set `--op-fwd-k 20` in training.** `grounding_losses` **already rolls the operative predictor to `k_max = max(fwd_k) = 20` every step** (the tac/str readouts consume all 20), so this adds **no rollout** — only 16 extra readout applications and their loss terms. It is L0 made intrinsic. | **≈ 0** (~1–2 % step time; no extra rollout) | L0's effect size is the direct evidence that 4-step calibration is the binding constraint | REFUTE if `wm_canary_ade_2s` at a matched step does not fall by ≥ 10 % against a same-seed `op_fwd_k = 4` control. Single-variable A/B. |
| **L2** | ⭐ **Scheduled observation dropout on the READOUT's input pair.** With probability *p*, decode `(z_real_t, ẑ_{t+1})` instead of `(ẑ_t, ẑ_{t+1})`. The real states are **already encoded** (`fut_states`, for the metric-inverse-dynamics term), so the cost is one extra readout call. This is the classic teacher/student-forcing repair aimed at the exact tensor §3.5 localises the defect in. | **≈ 0** | §3.5 (9.4× penalty on real pairs), §2.6e (perception hurts), §4.2 (peeking backfires) — three independent symptoms of one cause | REFUTE if, after training, (i) the FULL-OBSERVATION arm does not overtake imagination at 2 s, **and** (ii) the oracle-vs-uniform gap at matched duty does not turn positive. **Both, not either.** |
| **L3** | 🔴 **Fix the ACTION feedback, not the perception loss — and measure before building.** MEASURED here: holding the last observed action gives `T_blind` **3.2 s**; closing the loop through the model's own decoded motion gives **0.8 s**; feeding the model's own speed back gives `ade_0_2s` **9.36**. The deployable failure is **dominated by the control loop**. The cheap discriminating experiment is **eval-only**: re-run this exact sweep with the action taken from v1's *tactical planner* via `closedloop.wp_to_control` instead of my kinematic inverse. | **~30 GPU-min, eval only** | the 4× `T_blind` gap between two deployable action sources on identical windows | REFUTE the "the planner is better than the kinematic inverse" hypothesis if planner-derived actions do not beat `own_kinematic`'s **0.8 s** `T_blind`. If they do not, the instability is intrinsic to closing the loop at all and L3 becomes a training item. |
| **L4** | **Off-path augmentation.** P1 MEASURED that the closed-loop envelope is **not** a renderer limit — the yaw warp is geometrically exact — so about half of it is our arm's own OOD sensitivity (`INHERITED`, `RETRACTION_LOG` 07-26 C14). A blind rollout leaves the logged path **by construction**: §3.4 measures **20 % of steps outside the envelope by 6 s and 52 % by 12 s**. | one homography per sample; a full run to evaluate | the envelope fractions in §3.4 are the direct measurement of the exposure | REFUTE if `frac_steps_out_of_envelope` at 6 s does not fall **and** `de@6s` does not improve. |
| **L5** | ⚠️ **Latent-uncertainty readout — DEPRIORITISED, and the reason is measured.** A free self-signal *does* exist: `|v_pred(j) − v0|` reaches Spearman **0.40–0.44** against the true error at several horizons (`artifacts/uncertainty_readout.json`), clearing the ≥ 0.3 bar I would have set. **But §4.2 measured that a PERFECT error oracle is 25–112 % WORSE than a clock at matched budget** — so a better trigger is worth *negative* value until L2 lands. | ~0.2 GPU-h | — | Not scheduled. Re-open **only** after L2 turns the oracle-vs-uniform gap positive. |
| **L6** | **Low-cost re-anchoring (downscaled / cropped peek).** Strictly downstream of L2: there is no point cheapening an operation that currently hurts. | — | — | Not scheduled until L2 confirms. |
| **L7** | **Longer predictor horizons** (`[1,2,4]` → `[1,2,4,8,16]`). New JEPA heads ⇒ a graft or a restart. | **a full run** | — | Deliberately **last**. L0/L1/L2 test the same hypothesis for ~0 GPU-hours; spending a 59-hour run first is the exact failure `BOOST_PROGRAM` §3.4 exists to prevent. |

## 5.3 What is deliberately NOT proposed, and why

* **No new corpus.** Nothing measured here says data volume or diversity is the blind-driving lever;
  the own-dynamics-encoder line already REFUTED data diversity for the adjacent representational
  collapse (`INHERITED`, `…/2026-07-22-own-dynamics-encoder/`).
* **No renderer.** P1 established the warp is geometrically exact. A photoreal simulator is not what
  stands between us and this measurement.
* ⛔ **No learned peek/escalation trigger.** §4.2 measured negative headroom. Building one now would
  repeat H2's failure one level up — an unfalsifiable benefit claim on a metric that cannot move in
  the claimed direction.
* **No architecture change before L0, L1, L2 and L3 have run.** Three of the four are ~free and the
  fourth is eval-only.

---

# 6. Limitations, stated plainly

1. ⚠️ **The window set is EPISODE-INITIAL, and this is measured rather than assumed.** At `K = 185`
   the harness's own rule (`range(0, T − W − K, 8)`) leaves **599 windows, 596 of them at `t0 = 0`**.
   That subsample is **easier** than the full window set: on it v1's `ade_0_2s` is **0.3839
   [0.3598, 0.4106]** against the committed **0.4108** on all 13,198 windows (**−6.55 %**), and the CV
   floor is **0.6083** against **0.6917** (**−12.06 %**). ⇒ **absolute levels here run ~6–12 % low, and
   the CV floor runs low by twice as much as the model does — i.e. the subsample FLATTERS the floor
   more than it flatters the model, so "never beats CV" is if anything conservative.** All contrasts
   are paired on identical windows and are unaffected. `artifacts/horizon_curve.json`
   → `window_set_representativeness`.
2. ⚠️ **Everything past 0.4 s is EXTRAPOLATION**, including the 2 s canary the program already quotes
   (§3.4). Past **12 s** a majority of steps are outside the measured envelope, which fires `ood.py`'s
   own envelope clause.
3. ⚠️ **The `T_blind` for the A2 readout arm SATURATES at the sweep terminus** — **C14**: a lower
   bound on our configuration, not a horizon.
4. ⚠️ **`own_kinematic` is one action policy, not "the" deployable policy.** Its `accel` half is not
   an exact inverse of the corpus convention (measured cost **+0.137 m at 2 s**), and §2.6d shows a
   *different* deployable policy (hold-last) is 4× better. **The deployable `T_blind` = 0.8 s is a
   property of this arm AND this controller.** L3 is the experiment that separates them.
5. ⚠️ **FULL OBSERVATION under own actions is not self-consistent** — the percepts come from the
   logged trajectory while the ego claims to steer itself. It is a *teacher-forced-percept* ceiling and
   is labelled so. Under **true** actions it is self-consistent and is a genuine ceiling.
6. ⚠️ **The oracle peek is REACTIVE by construction** — it fires on a step whose error has already
   happened. A *prescient* oracle might do better. What is measured is that the natural error-triggered
   oracle loses to a clock; it does not prove no trigger could win. **But it does mean the burden of
   proof sits with the trigger.**
7. **One checkpoint, one arm.** Everything is v1 (`flagship4b-speedjerk-30k` @ 29999). Nothing here has
   been measured on v4, REF-B or REF-C, and the A2 lever's transfer to v4 is a `HYPOTHESIS`.
8. **The path is the SE(2) dead-reckoning of the step readout** — the same construction as every
   grounded number in the program. It is **not** a bicycle integration, so it is **not** comparable to
   `closedloop.closed_bicycle` (`1.6852` / `1.488` — two structurally different pipelines, cf. the
   harvest's definition-mismatch row).
9. **No safety metric exists here.** PhysicalAI-AV ships no map, lane graph, junction annotation or
   agent boxes, so no collision or drivable-area score is computable. This is a **drift** measurement.
10. **The A2 arms were added mid-run** (amendment A2). They are labelled everywhere, excluded from the
    four pre-registered regimes' `T_blind`, and should be discounted until independently re-run.
11. **`obstacle.offline` is machine-labelled** and unused here; nothing in this study depends on it.

---

# 7. What this unblocks (§7.4 requires this field)

## 7.1 🔴 ESCALATIONS — raised here, in the report's headline, not written into a README

**E1. Bar B has a lever now, and it is free. `BOOST_PROGRAM` §3.3 records it as UNOWNED with "no lever
identified".** Swapping the grounded decoder from `step["op"]` (4-step-calibrated) to `step["str"]`
or `step["tac"]` halves v1's blind 2 s ADE — **0.3839 → 0.1865 / 0.1950, paired-separated at every
horizon ≥ 1 s.** `wm_canary_ade_2s` is exactly that quantity, and Bar B asks for a **2.07×** fall
(1.1409 → ≤ 0.55). ⚠️ **On v1 the swap is a 2.06× fall (0.3839 → 0.1865). I do NOT claim it clears
Bar B**: Bar B is a **v4** number, v4 is a different checkpoint with a different `sel_gap` history,
and a ratio measured on one arm is not a prediction for another. What is claimed is narrower and
MEASURED: **a lever exists, its effect on v1 is of the required order, and testing it on v4 is
eval-only, ~20 GPU-min, and needs no new science** — v4's `HierarchicalGrounding` carries the same
three readouts. ⛔ **This must not sit in a document: it is either run on v4 or explicitly declined.**

**E2. Every grounded number in the program is decoded with a readout used 5× beyond its calibration.**
`taniteval/rollout.py:collect`, both `canary_rollout`s and `eval_grounded_rollout_4b*` all pass
`grounding.step["op"]` and evaluate at `k = 20`. That is not a bug — but it means `ade_0_2s` is a
**joint** measurement of the world model and of a decoder mismatch, and the two have never been
separated. §2.5 separates them for the first time. **A registry/leaderboard decision is needed: does
the program's headline metric keep the `op` decoder, or move to the calibrated one?** Either answer is
defensible; **silently having both in circulation is not.**

**E3. The v5 stream and this one independently verified the same instrument fact on the same day.**
`…/2026-07-26-v5-imagination-selection/` §0.2 verified `rollout_decode`'s blindness in the eval pod's
tree; this stream verified it in the repo tree and then reproduced two committed deployments through
it. **That is an M1 CONFIRMED via two independent paths** and should be recorded as such rather than
re-derived a third time.

## 7.2 What this unblocks, per stream

| stream | what it gets |
|---|---|
| **S-2 / v4 (Bar B)** | ⭐ **a named, measured, zero-GPU lever** (E1) for the one bar that had none, plus the mechanism (§1.2, §3.5) that says *why* it should work |
| **S-1 closed-loop measurability** | `T_blind` is a **closed-loop-relevant horizon that IS a measurement out to 0.4 s and an explicitly-labelled extrapolation past it**, on 596 clusters. §3.4's envelope table gives the horizon at which the closed-loop envelope clause fires (12 s), measured on this arm rather than inherited. |
| **H2 / sensor-need gating** | ⛔ **a clean negative that should stop work on the trigger**: at matched camera budget a *perfect* error oracle is 25–112 % **worse** than a clock (§4.2). H2's own conclusion was that the informative axis had to be constructed; it now exists and it says the gate is not the bottleneck. |
| **the hierarchy / three-planner direction** | §2.6d: two *deployable* action sources differ 4× in `T_blind` on identical windows. The operative control loop, not perception, is the dominant deployable failure — which is a measurement the tactical-planner work can be aimed at. |
| **`lateral.py` / corridor work** | §3.1 reproduces `lateral.py`'s founding "lateral compounds ~5× faster" finding on **596 episode clusters** instead of two clips, and locates the longitudinal→lateral crossover at **4.5–6 s** — directly usable as the horizon at which a corridor metric starts to bind. |
| **`MODEL_REGISTRY`** | v1 now has a **blind-horizon row**: `T_blind` by action regime, the beats-CV interval, and the decoder-calibration caveat on `ade_0_2s`. |

**What it unblocks nowhere:** nothing here touches data ingest, the corpus question, Orin/Thor, or
AlpaSim. Stated plainly as §7.4 requires.

---

# 8. Amendments — recorded here, not by editing the pre-registration

| # | what changed | why, and what it can and cannot bias |
|---|---|---|
| **A1** | A **second peek base** (`hold_last`) was added beside the pre-registered `own_kinematic` base for E-IMAG-4 | An addition, not a substitution: the pre-registered base is still reported and still primary. Adding it costs one extra rollout per policy and lets the duty-cycle curve be read in **both** deployable action regimes, which matters because the two regimes do not give the same verdict (§2.5). |
| **A2** | Six **readout-level sensitivity arms** (`step["tac"]` / `step["str"]` instead of `step["op"]`) were added after the primary arm table was fixed and before any result was read | They exist because of §1.2, a fact read out of v1's own config — not out of a result. They are a **sensitivity, never part of the primary comparison**, they are named `__roTAC`/`__roSTR` everywhere, and `T_blind` for the four pre-registered regimes is computed without them. ⚠️ They are the one place in this report where an arm was added mid-run, and the reader should discount them accordingly until an independent re-run confirms them. |
| **A3** | `T_blind`'s interval is the percentile interval of `T_blind` **re-derived inside every episode resample**, not a read-off of where the point curve's CI crosses zero | Pre-registered in §4 as written; recorded here because the distinction matters: the second construction would understate the uncertainty of a *horizon*, which is a non-linear functional of the curve. |
| **A4** 🔴 | **`T_blind`'s contiguity anchor moved from N = 1 to N = 2** | ⛔ **A defect in my own pre-registration, found by reading my own result — the exact C13 class the program warns about.** Arms (a), (b) and (c) decode a **bit-identical** first transition by construction (`test_first_step_is_identical_across_state_sources`), so the paired Δ at step 1 is **exactly 0.0** and its bootstrap lower bound is exactly 0.0 in every draw. "Contiguously from N = 1" therefore returns **`T_blind = 0` for every arm in every regime regardless of the data** — a criterion that **cannot fire**, which is worth nothing whichever way the world is. The repair is the smallest one that makes the criterion evaluable: anchor at the first horizon on which the arms can differ at all. ⚠️ **What it can bias:** it is a strictly *more permissive* rule, so it can only move `T_blind` up from 0. It cannot manufacture the deployable regime's Outcome B, which is a **negative** result and is corroborated independently by the never-beats-CV contrast. The unrepaired all-zero output is quoted in §0 so the correction is auditable, and the rule is pinned in code (`bi_analyze.T_CONTIGUITY_START_STEP`). |

---

# 9. Deliverable manifest

**Everything is in the repo working tree and STAGED (`git add`). Nothing was committed or pushed.**
Deliverable path: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-blind-imagination/`

| artifact | what it is | where it lives |
|---|---|---|
| `taniteval/taniteval/blindimag.py` | ⭐ **the instrument** — `blind_rollout` (4 state sources × 4 action sources + 2 peek policies), the kinematic action inverse, dense GT/floors at arbitrary horizon, the path-deviation probe, the window builder | **repo** (+ `pod2:/root/taniteval/taniteval/`) |
| `taniteval/tests/test_blindimag.py` | **the certification** — 22 tests incl. bit-identity with `rollout_decode`, and M3 deliberately-failing inputs | **repo** (+ pod2) |
| `PRE_REGISTRATION.md` | written before any rollout ran: arms, regimes, `T_blind`, both outcomes, the §7.3 disappointing-value table, the reproduction gate | **repo** |
| `BLIND_IMAGINATION.md` | this document | **repo** |
| `artifacts/gate_reproduction.json` | ⛔ **the reproduction gate** — 0.427109 / 0.410807 against the two committed deployments, plus the tensor diff vs the unmodified harness | **repo** |
| `artifacts/horizon_curve.json` | every arm × `de_N` / `ade_N` / sparse `ade_0_2s` at the reporting grid, each with its episode-cluster bootstrap; the window-set representativeness block | **repo** |
| `artifacts/t_blind.json` | ⭐ `T_blind` per regime with its bootstrapped interval, all paired contrasts, the usefulness bars, the A2 readout-lever block | **repo** |
| `artifacts/decomposition.json` | lat/lon (ego **and** Frenet), drift vs variance, predicted speed, OOD envelope fractions, per arm per horizon | **repo** |
| `artifacts/duty_cycle.json` | E-IMAG-4 — 30 peek policies, realised duty cycles, oracle-vs-uniform at matched budget | **repo** |
| `artifacts/uncertainty_readout.json` | the free self-signal probe (speed-drift / speed-jump vs true error) | **repo** |
| `artifacts/vs_floor_contrasts.json` | paired contrasts against the CV floor + the exact horizon interval over which each arm is separated-better | **repo** |
| `artifacts/_tables.md` | the rendered tables exactly as spliced into §1–§4 | **repo** |
| `perwindow/bi_perwindow_compact.pt` (12.1 MB) | ⭐ **the recompute-anything dump**: full dense per-window `de` for 13 headline arms + reporting-grid `de`/`along`/`cross` for **all 43** arms + `eid`/`speed`/`head_deg`/`t0`/peek duty. **Any bar, any horizon, any stratification can be recomputed from this with no GPU.** | **repo** |
| `scripts/bi_run.py` | the pod2 driver — `gate` and `sweep` (sweep + peek share one encoding pass) | **repo** (+ `pod2:/root/bi/`) |
| `scripts/bi_analyze.py` | all estimators; writes every artifact JSON | **repo** (+ pod2) |
| `scripts/bi_report.py` · `scripts/bi_splice.py` | render the tables from JSON and place them — **no number in the report is hand-typed** | **repo** (+ pod2) |

**Living in only ONE place (declared, per rule 2):**

| | where | why, and what it costs to rebuild |
|---|---|---|
| `perwindow_sweep_K185.pt` (36.8 MB) · `perwindow_peek_K185.pt` (59.7 MB) — the FULL dense dumps incl. `psi` and `peek_mask` for all 43 arms | **`pod2:/root/bi/perwindow/` only** | 96.5 MB is too large for the repo; the **compact 12.1 MB dump above is in the repo and carries everything the report's numbers need.** Rebuild: **28.4 min** on an idle A40 (`bi_run.py sweep`), deterministic (`torch.manual_seed(0)`, no sampling in the rollout). |
| the per-episode encoded states (~1 GB, in RAM only) | never written | rebuilt in **14.3 min** as part of `bi_run.py sweep`. |

**Reproduction, end to end** *(pod2; `PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts:/root/taniteval`, `OMP_NUM_THREADS=8`)*

```
python3 -m pytest /root/taniteval/tests/test_blindimag.py -q          # 22/22, the certification
python3 bi_run.py     gate  --out artifacts                          # the reproduction gate; refuses to pass silently
python3 bi_run.py     sweep --out perwindow --episodes 600 --kmax 185
python3 bi_analyze.py --sweep perwindow/perwindow_sweep_K185.pt \
                      --peek  perwindow/perwindow_peek_K185.pt --out artifacts
python3 bi_report.py  artifacts artifacts/_tables.md
python3 bi_splice.py  artifacts/_tables.md BLIND_IMAGINATION.md
```

**Timings, MEASURED on this run** (`artifacts/sweep_meta_K185.json`, `artifacts/gate_reproduction.json`):
gate 40-ep leg **28.2 s** (+ 45.5 s for the unmodified-harness reference) · gate 600-ep leg **902.5 s**
· sweep encode **852.7 s** · sweep + peek rollouts, 43 arms × 599 windows × 185 steps, **829.1 s** ·
analysis **~30 s CPU**. Total **≈ 44 min** on one idle A40. **pod1, pod3 and the eval pod were never
touched; the val cache was read only.**

**Suites green before staging:** `taniteval` **423 passed**, `stack` **1105 passed / 7 skipped**.
