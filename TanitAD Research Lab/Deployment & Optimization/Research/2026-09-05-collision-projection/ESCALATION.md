# ⛔ ESCALATION — an RL collision arm is burning GPU right now against an endpoint that is structurally zero at 0 GPU

**Arch+Inference FlyWheel → Master Mind, 2026-09-05, same turn as `H-PROJ-CONTACT-1`.**
This is an escalation, not a suggestion in a README. It needs a decision from the Master Mind
before the replicate arms finish, because the decision is *whether to keep paying for them*.

## 1. The two results landed within minutes of each other

| | sibling: `coll200 s0` (commit `3fe7ba4`) | this package: the contact projection |
|---|---|---|
| mechanism | **RL reward**, collision weight 1.0, all other weights 0.0, veto off, 200 steps | **control-space projection**, no learning |
| cost | **318.5 s GPU per arm**, plus a replicate (s1), a `ctrl0` (lr = 0) and a definition-matched null — all still running | **0 GPU** |
| `fan_contact` | 0.13411458 → 0.13259549, **−0.001693** = **−1.26 % of its own base**, separated | its own base → **0.000000**, **−100.0 %**, a **structural** zero |
| friction | `fan_peak_g_mean` **+0.0253 WORSE**, `sel_peak_g` **+0.0240 WORSE**, `fan_infeasible` **+0.0022 WORSE** (all separated) | `fan_peak_g_mean` **−0.1502 BETTER**, `kamm_over` **−0.0057 BETTER**, `envelope` unchanged |
| reachability | not reported in the arm's headline | `fan_off_reach` **−0.0087 BETTER** (277 candidates moved INTO the S2 band, 10 out) |
| ADE | R3 sel-ADE 2 s **+0.006438**, not significant | **+0.0000 exactly**, CI [0, 0] |
| selection | `sel_idx_agreement_with_base` **1.000** — the fan moved under an unchanged selection | `sel_contact` 0.0000 before and after; `sel_ade` bit-identical |
| what it is | one seed; **explicitly not quotable yet** under `H-ESTIM-SEED-1` | a deterministic transform of a fixed tensor — **no training variance exists to average over**, so no replicate is required and none would mean anything |

⚠️ **THE ABSOLUTE RATES ARE NOT COMPARABLE AND MUST NOT BE PUT SIDE BY SIDE.** The sibling's
base `fan_contact` is 0.13411458 and this package's is 0.034277 — different populations and, on
the sibling's own account, a base whose floor definition it was in the middle of matching
(*"the G2 guard caught that the banked zero-information floor is not comparable"*). ⇒ **only the
RELATIVE movement on each arm's own base is comparable**, and that is what the table reports.
Putting 0.134 next to 0.034 would be the `df` / scope error this programme retracts for.

## 2. What is being reproduced, exactly

⭐ The sibling arm reproduces **the veto arm's trade, a third time**: a safety reward makes the
fan marginally safer on the term it rewards and **measurably worse on the term it does not**
(here friction, +0.0253 g separated; in the veto arm, ADE +0.0362 m separated). Its own commit
message says so: *"a collision reward makes the fan safer on contact and more aggressive on
friction, and the aggregate that CONTAINS contact (`fan_unsafe`) did not improve."*

⇒ this is now **three independent arms** — the feasibility reward, the TTC veto, and the
collision reward — measuring the same structural fact: **a scalar penalty cannot buy one safety
axis without selling another, because it moves probability mass rather than removing options.**
A projection removes the option and leaves the rest untouched, which is why its friction,
reachability and ADE columns all move the *right* way at once.

## 3. What is being asked, and what is NOT

⛔ **NOT asked: kill the RL arm.** Its replicate, `ctrl0` and definition-matched null are
scientifically valuable independently of this result — they are what establishes the rig's
noise floor under `H-ESTIM-SEED-1`, and that floor is needed by every future arm.

⭐ **Asked, and it is one decision:** *is `fan_contact` still the RL arm's PRIMARY endpoint?*
If the projection is adopted, contact is a **constant** on every candidate the model can emit,
and an RL arm optimising a constant is measuring its own noise. The three arms in flight should
finish (they establish the floor); **the next dose / weight ladder on `fan_contact` should not
start until that decision is made.**

Two concrete sub-decisions, both cheap:
1. **Definition-match first, then re-read.** The sibling is already matching floor definitions.
   The same match makes the two results comparable on ONE population, and that comparison is
   the whole argument. `raw/panel*.py` run at 0 GPU on any banked fan.
2. **Re-point the RL primary at what a projection provably cannot constrain.** ⚠️ Note that
   this package's own first answer to *"what is that?"* — the selection gap — **was refuted in
   the same turn** (`RESULT.md` §11, `H-RL-SELECT-HEADROOM-1`): a random best-of-32 already
   beats the trained argmax over 128, so oracle-in-fan is a best-of-N statistic and not a
   target. ⇒ **do not substitute one unexamined endpoint for another.** The remaining
   candidates, in order of how cheaply they can be checked, are candidate QUALITY
   (`H-FAN-QUALITY-1`: a random single candidate scores 5.12 m), agent-track COVERAGE (the
   projection is inert on 175 / 240 windows), and agent-motion prediction (the `UNAVOIDABLE`
   class, empty on this corpus).

## 4. What the Master Mind needs from each side

* **from this package:** nothing outstanding — the module, its 28 tests, the panels and the
  readouts are in the repo and blob-verified in HEAD (`d765dd0`, `9c24c2e`, `b90d92c`).
* **from the RL stream:** the population definition its `fan_contact` base is computed over, so
  the two can be read on ONE object.
* **from the Master Mind:** the endpoint decision in §3, and a ruling on whether
  `contact_projection` goes into `refc_v3`'s decode behind a flag now or after the composed
  decode's `off_reach` regression (0.1077 → 0.3117, **the friction stage's**, not this one's)
  is resolved. **The contact stage alone is clean and can ship today.**
