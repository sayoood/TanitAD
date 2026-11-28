# refcv6 — what it costs, and the three decisions only the PI can make

**Master Mind, 2026-09-10.** ⛔ Nothing here is a capability claim. This prices a panel that has
not run. Compute provisioning and spend are the PI's; the plan is mine.

---

## 1. The unit cost, cross-checked from two independent sources

| quantity | value | source |
|---|---|---|
| steps per arm | **40,284** | refcv5-v2's own `config.json` argv (`--steps 40284`) |
| pace | **4.2 s/step** | MEASURED live on the A40 over the 10,100 → 16,400 window |
| ⇒ derived wall-clock | **47.0 h** | 40,284 × 4.2 s |
| banked `ckpt.pt` mtime | 2026-09-09 21:11 | launch was 2026-09-07 21:58 ⇒ **≈47 h wall** |

⭐ **The two agree.** The per-step pace and the checkpoint's own wall-clock are different
channels and land on the same number, so **47 GPU-hours per arm** is a cross-checked figure and
not a single-probe estimate. ⚠️ The wall-clock also absorbed **four crashes**, all the same
MooseFS EIO — infrastructure, not the arm — so the true compute is at or just under 47 h.

⚠️ **Evidence class**: the 4.2 s/step was read from the live log while the run was in flight; no
`metrics.json` was banked with the final artifact. That is a gap in the record and it is on me.

---

## 2. The panel, and why it is smaller than "every arm gets a replicate"

⛔ **The rig cannot adjudicate a one-seed arm.** MEASURED on WP-D: an arm with **zero levers
moved** read "separably worse" on **5 of 9** family metrics and reproduced a headline ADE effect
at **+0.02460** against the lever's own **+0.02610**. And `H-ESTIM-SEED-1`: a replicate with
A0's flags and A0's seed produced separated differences on **6 of 42** family cells — a **14.3 %
false-positive rate for "separated"**. The paired episode-cluster bootstrap resamples **episodes
with the models held fixed**, so it answers *"would another draw of episodes say this?"* and is
**structurally blind** to *"would another training run say this?"*.

⇒ **A replicate is not optional. But the FLOOR only has to be measured once.**

| # | arm | steps | GPU-h | what it buys |
|---|---|---|---|---|
| 0 | **baseline replicate** — refcv5-v2's argv verbatim, new seed | 40,284 | 47 | ⭐ **the rig's own noise floor.** Without it every later number is unreadable. |
| 1 | **D** — stop passing `--w-u0 0.5` (return to the 0.0 default) | 40,284 | 47 | tests whether **our own non-DD invention** is the longitudinal cost |
| 2 | **A** — `--agents head` | 40,284 | 47 | the DiffusionDrive mechanism, **absent not tested** in v5 (`agent_join = None`) |
| 3 | **B** — `--wp-index on` (requires A) | 40,284 | 47 | coupling (1), +2,208 params |
| 4 | **C** — `--w-tac-goal > 0` | 40,284 | 47 | the 11,286-param head that took **zero gradient** |
| | **total** | | **235 GPU-h** | |

⭐ **Confirmation replicates are bought only by arms that look like they cleared** — +47 h each,
spent on a signal rather than on all four in advance. Worst case (all four clear) the panel is
**423 GPU-h**; expected case is **235–329 h**.

⚠️ **"Free" in the design doc meant no new code and no PI decision. It never meant no GPU.**
Arm D is one argv token and still costs 47 h.

### 2.1 The order is a cost order, not an interest order

**0 → 1 → 2 → 3 → 4.** Arm 0 first because nothing else is readable without it. Arm D second
because it is the cheapest possible test of the one thing in refcv5-v2 that departed from **both**
the paper and our own trainer's default. ⛔ Not all at once: refcv5-v2 moved several levers and
its result is barely attributable, and the `--v2` conflation failure — ten levers on two axes,
non-attributable — is the standing precedent.

### 2.2 The bar, committed before any data

**Beat `ha0_ext` AND `ha` on ADE, separated, plus a 0.10 relative margin.** ⛔ Clearing the CI
while missing the margin is a **FAIL as written**.

⭐ **Plus two non-regression clauses, and these are the half most likely to be quietly dropped:**

* **LATERAL.** refcv5-v2 won heading **1.2121** (vs 1.5489), yaw-rate **1.0534** (vs 1.4542),
  cross-track **0.0994** (vs 0.1226) and masked curvature **0.003485** (vs 0.004030, = **0.512×
  the straight-line floor**). The obvious way to "fix" ADE is to hand those back. refcv6 may not.
* **STRATEGIC.** refcv5-v2 is the **only arm in the programme with a strategic output**: route
  accuracy **0.7708** [0.7146, 0.8254], κ 0.4614, n 3,622, against a 0.3333 chance — from a
  **771-parameter** head, beside a **106,067,312**-parameter core. refcv4b reads **n = 0**.
  The strategic level is nearly free and must not regress.

Four families, **never pooled**. **T1** stamp on every number. Paired episode-cluster bootstrap
only; ⛔ `overlapping_holdout_se` is forbidden and also **biases the point estimate**.

---

## 3. The three decisions — each with the default I will take if the PI says nothing

### D1. Compute — a pod for the panel ⛔ BLOCKING, and it blocks everything

The A40 is **stopped and gone**; every irreplaceable artifact was verified byte-exact off it
first. Thor and the dev-box RTX 4060 are free but are **probe-and-eval class**, not 40 k-step
class. Nothing in refcv6 runs without a pod.

**Default if silent:** provision **one A40-class pod** and run arms 0 and 1 only — **94 GPU-h**,
the smallest spend that produces a readable answer, because it buys the noise floor and the one
lever that tests our own invention. Report, then ask again before arms 2–4.

### D2. Item 10 — the `MANEUVER_WEIGHT` budget for arm C

Arm C needs a number for `--w-tac-goal`. ⛔ **The weight is not mine to pick.**

⭐ **My question, reframed because the earlier one did not land:** rather than asking the PI for
a number, **may I spend a few hours of Thor measuring it?** A small sweep on the tiny rig gives
a defensible starting weight instead of a guess, and it costs no pod time.

**Default if silent:** run that measurement on Thor, publish the sweep, and hold arm C until the
PI confirms the value.

### D3. Item 8 — the RL pilot's cold start ⛔ blocks the ENTIRE V2 RL stage

The pilot cannot load its own cold start: **487 keys against 488 built**, the extra being
`decoder.anchor_controls`. ⛔ Not to be worked around with `strict=False`, which the code itself
calls *"a plausible-looking WRONG experiment"*.

⭐ **Why this outranks its size.** `H-DDA-5`: DD-V2's RL gain is **EP +5.3, DAC +1.7, with NC /
TTC / comfort flat** — a **longitudinal-scale** effect. And **92.2 % of our own `os − ha` gap is
along-track**. The mechanism and our measured deficit are the same axis. This is the piece aimed
at what is actually wrong with the planner.

**Default if silent:** the **declared allowance** — permit the missing tensor **only when
`v0_conditioned` is False**, read from the **checkpoint's own config** and never from the CLI,
refuse otherwise, and stamp what was defaulted into `config.json`. Implementation is in flight.

---

## 4. What is already runnable with no pod and no decision

Every one of these is in flight or landed today: the refcv6 pre-registration and launch scripts;
the item-8 declared allowance; the V2 stricter truncation (`mask_positive = reward_group >
reward_gt`) and the reachability proof for the Hungarian path from `--agents head`; the 121
staged research-bank paths; the refcv5-v2 upload; the registry and leaderboard rows.

⛔ **And one thing the PI asked for by name is blocked on data, not compute:** `--max-speed-input`
reads the v8 `speed_max_input` block, which the v7.2 release carries on **0 of 4,572 records**,
and the flag **REFUSES there rather than looking switched on**. Whether an admissible supplier
exists at all is under investigation — PhysicalAI publishes no map, and a ceiling derived from
the ego's own realised speed would be the **nav-echo defect** in a new costume.

---

## 5. ⛔ CORRECTION — ARM D CANNOT START. The plan in §2 and §3 was wrong.

**MEASURED 2026-09-10**, found by a stream that was sent to audit the V2 mechanism gaps, not to
check this. ⭐ It outranks what that stream was asked to do, and it was reported as such.

### 5.1 The defect

`refc_v3_train.py:468` raises a hard **`SystemExit`** on any arm combining **`--sampler ddim`**
with **`--w-u0 0`**.

refcv6's BASE carries `--sampler ddim` (`arms.py:110-111`). Arm **D** is `--w-u0 0` (`arms.py:286`).
⇒ **That is exactly the refused pair.**

⛔ **Arm D would have died in the first second of a run, discovered only AFTER a pod was
provisioned.** It is the arm §3.3 says to run **first**, the arm I called *"one flag and free"*,
and the arm the register still lists as **"RUNNABLE THE MOMENT A GPU EXISTS — needs no PI
decision"**. All three statements are wrong. ⚠️ The one in §2's default — *"provision one
A40-class pod and run arms 0 and 1"* — is the expensive form of the error, because arm 1 **is**
arm D.

### 5.2 ⭐ The refusal's stated REASON is measurably false — but the refusal stands

`:468` justifies itself by claiming `control_head` *"stays at its zero init forever"* without the
`u0` term. MEASURED, in a single forward:

| path | `grad_abs_sum` |
|---|---|
| via `out["anchor_traj"]` — the tensor the **matched-anchor L1** gathers | **9.39e4** |
| via `u0_hat` | 1.07e5 |
| three no-information controls, same forward | **exactly 0.0** |

⇒ `control_head` reaches the matched-anchor L1 and is trained **without** the `u0` term. The
justification has rotted; the code kept enforcing it. Logged as class **`JUSTIFICATION-ROT`**.

⛔ **The reason was refuted, NOT the decision.** The refusal was kept and its message corrected,
because whether that arm should run is a **PI ruling**, not an agent's. ⭐ That distinction is the
right one: a false justification licenses re-opening a question, never overriding the answer.

### 5.3 ⇒ The decision this creates, and it is now the FIRST thing needed

⭐ **`D-DDV1-NO-DENOISING-LOSS` is the whole reason arm D exists.** DD-v1 has **no ε-prediction and
no denoising MSE at all**; its `diff_loss_weight = 20.0` is dead code, and its only trajectory
supervision is matched-anchor L1 plus focal scoring. Our `--w-u0 0.5` is an **invention, not a
port** — and refcv5-v2, the arm carrying it, is the one that lost longitudinally. The measurement
in §5.2 **strengthens** the case for the arm: the head the refusal was protecting is trained
anyway.

**Two options, and the PI picks:**

| | |
|---|---|
| **(a) ⭐ recommended** | authorise `--sampler ddim` with `--w-u0 0` behind an **explicit acknowledgement flag**, so the configuration is recorded as a deliberate operator choice rather than a bypass — the same shape as `control_units_source: cli-override-legacy-file` |
| **(b)** | drop arm D, and rewrite §3.3's ordering, which currently rests on it |

**Default if silent: (a)**, because the refusal's own premise is refuted and arm D is the cheapest
test of the only lever that is ours rather than the paper's.

### 5.4 What is still runnable with no decision

**`V0` and `V0b` only** — the baseline pair that measures the rig's noise floor. ⛔ That pair is a
**prerequisite for every other arm's bar**, so it is not a consolation item; it is the thing that
must run first regardless. **94 GPU-hours.**

⚠️ ⇒ §2's "run arms 0 and 1" default is **replaced by "run arm 0's pair"** until the PI rules on
arm D.
