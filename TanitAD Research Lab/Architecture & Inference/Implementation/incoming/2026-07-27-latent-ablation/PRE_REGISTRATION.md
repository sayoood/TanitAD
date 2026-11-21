# PRE-REGISTRATION — the LATENT ABLATION. Is blind driving imagined semantics, or action integration?

**Written 2026-07-27 ~02:50 Europe/Berlin (00:50 UTC), BEFORE any number in this folder existed.**
This file is **not edited**. Every deviation is an amendment in `LATENT_ABLATION.md` §8.

**Question (the PI's, verbatim):** *is blind driving actually prediction from the world model's
imagined semantics, or is it kinematic integration of the action channel?*

**Arm:** v1 = `flagship4b-speedjerk-30k` @ step 29999 (`MODEL_REGISTRY.md` §1.2), the calibrated
`str` (k = 20) readout, the SAME 599 windows / 596 episode clusters as Rung 0 / Rung 1.
**Estimator, everywhere:** paired episode-cluster bootstrap — `taniteval/ci.py`, **B = 2000, seed 0**,
unit = **episode cluster**. `overlapping_holdout_se` appears nowhere in this folder.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID, no raw content.

---

## 1. What the architecture makes separable, and what it does NOT — stated before running

Read out of `taniteval/blindimag.py` (`blind_rollout`, lines 485–583) and `build_windows`
(lines 799–807), **not** from any prose:

1. ⭐ **The `v0` action channel carries the TRUE observed speed and is held CONSTANT for the whole
   rollout.** `build_windows` broadcasts `pose_last.v / SPEED_SCALE` across the window *and the entire
   future*; `_pack_action` carries channels ≥ 2 through unchanged; every arm here runs
   `update_speed_channel = False`. **This scalar survives every latent ablation below, by
   construction.** That is precisely what makes these ablations a valid test of the PI's dichotomy:
   the "action channel" the integrator hypothesis names is left fully intact while the latent is
   destroyed.
2. ⚠️ **With `own_kinematic` the fed steer/accel are a FUNCTION of the latent** — the action is the
   kinematic inverse of `dpose = step_readout(win_s[:,-1], z_hat)`. So at α < 1 a latent ablation
   *also* perturbs the action. This is not a defect of the design, it is the architecture: outside
   the constant `v0` and the hold-last blend there is no action channel independent of the latent.
   **α = 1.0 isolates it** — there the fed action is the last OBSERVED action for every arm and is
   bit-identical across all state sources, so any difference is purely the latent. α = 1.0 is
   therefore registered as the **UNCONFOUNDED** row and is adjudicated alongside α = 0.25.
3. ⛔ **`T_blind`'s comparator IS the frozen-latent arm.** `tb_rung0.t_blind(de_a, de_b)` is "the
   largest horizon at which (a) is separated-better than (b), contiguously from N = 2", and every
   published `T_blind` in this program uses **b = `frozen_last`**. Consequences fixed here in
   advance:
   * `T_blind(FROZEN vs FROZEN)` is **1 step by construction** — the rule's failing floor, returned
     for any pair of identical arms. It will be emitted as **`VACUOUS: null`** and **never
     adjudicated**. A number that cannot be anything else is not a result.
   * For SHUFFLED / MEAN / ZERO, `T_blind(X vs FROZEN@α)` **is** well defined and is exactly the
     PI's question: *does a corrupted latent still stay ahead of a frozen percept?*

---

## 2. The arms — fixed here, no arm added after a result is seen

**State sources** (the latent channel; what is appended to `win_s` at each step):

| tag | `state_source` | what the predictor's context becomes | status |
|---|---|---|---|
| **INTACT** | `imagination` | the model's own predicted latent `z_hat` | exists |
| **FROZEN** | `frozen_last` | the last REAL percept, held constant | exists |
| **SHUFFLED** | `shuffled` *(new)* | at each step, the imagined latent of a **different window**, via a seeded per-step **derangement** of the batch. Per-step marginal is the SAME multiset; correspondence destroyed. | new |
| **SHUF-REAL** | `shuffled_obs` *(new)* | at each step, a **real observed** latent from a different window (`obs_states` deranged). Marginal = the real-latent marginal. | new |
| **MEAN** | `mean_latent` *(new)* | the batch mean of the last real percept, broadcast — marginally central, zero per-window information | new |
| **ZERO** | `zero_latent` *(new)* | all zeros — the strongest, deliberately off-distribution | new |

**Action damping** α, on `own_kinematic|blend=α` (`a_fed = (1−α)·a_own + α·a_hold0`):

| α | model's share of the steer/accel command | why it is in the design |
|---:|---:|---|
| 0.00 | 100 % | the undamped own-action reference (Rung 0's 25-step arm) |
| **0.25** | **75 %** | ⭐ **the genuinely model-driven point** — where the arm first beats constant velocity. **PRIMARY.** |
| 0.75 | 25 % | the brief's second point; near the hold-last ceiling |
| **1.00** | **0 %** | ⭐ **UNCONFOUNDED** — fed action identical across all state sources. **CO-PRIMARY.** |

⇒ 6 state sources × 4 α. INTACT and FROZEN at all four α already exist in the committed Rung-1 dump
(`a_/b_imagination__own__roSTR`, `a_/b_blend0.25`, `a_/b_blend0.75`, `a_/b_..hold..`) and are
**re-rolled as identity anchors**; the four new state sources are the new compute.

---

## 3. The statistics

**PRIMARY (adjudicating): `de@2s`** — mean dense path deviation at step 20, comparator-free.
**CO-PRIMARY (adjudicating, must AGREE): `T_blind`**, in the PI's own 20 % form.
**Reported for every arm:** `T_blind`, `de@2s`, `de@6s`, `ade_0_2s`, **beats-CV** (steps + interval),
**`T_useful@1m`**, and `R_X` across the horizon grid (5, 10, 20, 30, 45, 60, 90, 120, 185).

**M10 tier split is kept:** the `T_blind` number and the *capability* claim (beats-CV, `T_useful@1m`)
are reported in separate tiers. A `T_blind` move with beats-CV at 0 is a metric move, not a
capability.

Definitions fixed now:

```
R_X(α)     = ( de@2s[X, α] - de@2s[INTACT, α] ) / de@2s[INTACT, α]      # fractional degradation
cost_X(α)  = 1 - T_blind(X vs FROZEN@α) / T_blind(INTACT vs FROZEN@α)   # the PI's form
DESTRUCTIVE = { SHUFFLED, SHUF-REAL, MEAN, ZERO }                       # FROZEN is reported apart:
                                                                        # a stale REAL latent is not
                                                                        # a destroyed one
```

---

## 4. ⛔ THE ADJUDICATION RULE — both outcomes committed, and a mandatory PARTIAL

Evaluated at **α = 0.25 (primary)** and **α = 1.00 (unconfounded)**. Both must be reported; the
verdict sentence quotes α = 1.00 as the attributable reading and α = 0.25 as the deployable one.

| bucket | PRIMARY (`de@2s`) | CO-PRIMARY (`T_blind`) |
|---|---|---|
| **INTEGRATOR** | `max_X R_X < 0.20` — even the most destructive ablation costs < 20 % | `max_X cost_X < 0.20` |
| **SEMANTIC** | `min_X R_X ≥ 1.00` (≥ 2× worse) **and separated** for every X | `min_X cost_X > 0.80` |
| **PARTIAL** | anything else | anything else |

> ⛔ **If PRIMARY and CO-PRIMARY disagree, the verdict is PARTIAL. Mandatory, no discretion.**
> This exists so that two adjudicating statistics cannot become a choice of which to quote.

**What I will do in PARTIAL** — fixed now, so it cannot be invented to fit:
1. **Refuse a headline INTEGRATOR/SEMANTIC verdict.** State PARTIAL in the verdict block.
2. Report the measured share `R_X` / `cost_X` **with its interval** at both α and across the grid,
   and say plainly how much of the horizon the latent is and is not carrying.
3. Quote **α = 1.00** as the attributable number, because at α < 1 the ablation also perturbs the
   action and a PARTIAL there is not attributable to the latent alone.
4. Name the **cheapest discriminating follow-up** that would move it out of PARTIAL.
5. State explicitly which of the PI's two sentences the number supports **and by how much** — a
   PARTIAL is not permission to decline to answer.

---

## 5. ⚠️ THE C13 CHECK — the failing values, and proof in advance that they can fire

*Three vacuous diagnostics have shipped in this program. This is the check that this one is not a
fourth.* For each bucket, a measurable outcome that produces it, and evidence it is attainable **on
this pipeline and these windows**:

| bucket | the outcome that returns it | attainable? |
|---|---|---|
| **INTEGRATOR** | `de@2s[ZERO] ≈ de@2s[INTACT]` — destroying the latent entirely changes nothing | ✅ It is exactly what four MEASURED results predict (the true-`v0` integrator beating every latent probe 24.8×/15.4×/8.3×/1.7×; the second latent slot adding ~nothing). |
| **SEMANTIC** | `de@2s[ZERO] ≥ 2 × de@2s[INTACT]` | ✅ **Demonstrated on these very windows**: Rung 1's `own_vupd` arm has `de@2s` **23.9351 vs 1.8165 = 13.2×** (`rung1_interventions.json`). A ≥ 2× degradation is a value this pipeline demonstrably produces. |
| **PARTIAL** | any intermediate, or PRIMARY/CO-PRIMARY disagreement | ✅ trivially |
| **`T_blind`'s own failing value** | 1 step (0.1 s), returned when the first evaluable horizon already fails | ✅ pinned in `tb_rung0.t_contiguous`; it is what `T_blind(FROZEN vs FROZEN)` returns, which is why that cell is emitted as VACUOUS rather than as a number |

⛔ **Emitted as an artifact, declared here in advance:** a `diagnostic_vacuity_audit` block listing,
for every diagnostic in this folder, whether both a passing and a failing value are attainable. Any
diagnostic that cannot fail is **not emitted at all** rather than emitted and caveated.

---

## 6. ⛔ GATES — nothing below them is read until they pass

**G1 — window-set identity.** 599 windows, 596 episode clusters, `eid` and `t0` ordering identical to
the committed Rung-1 dump; the **six** anchors (`a_/b_imagination__own__roSTR`, `a_/b_blend0.25`,
`a_/b_blend0.75`) reproduce their dense `de` within **1e-4 m** (Rung 0b measured 3.05e-05 m between
two encode passes; that is what the tolerance is for). **Blocking.**

**G2 — plumbing self-test, BOTH directions.** A silently no-op state source produces a flat,
confident, wrong table.
* *fidelity*: `shuffled` with an **identity permutation** must be **BIT-IDENTICAL** to `imagination`
  (max |Δ| = 0.0). Same for `shuffled_obs` with identity vs `full_obs`.
* *anti-no-op*: **every** new state source must move the path — no arm may be identical to INTACT.
**Blocking.**

**G3 — fidelity to the committed headline.** `T_blind` / `de@2s` / `ade_0_2s` / `T_useful@1m` for the
INTACT and FROZEN arms at all four α must reproduce `TBLIND_RUNG1.md` §3. The **level** agreement is
blocking; the **`T_blind` integer** is reported non-blocking (a step count is a threshold crossing and
these come from a second encode pass).

**G4 — CPU test pin.** Every new state source gets a test in `taniteval/tests/test_blindimag.py`,
including the identity self-test and the anti-no-op assertion. `pytest -q` on the full `taniteval`
suite must stay green. **Blocking.**

---

## 7. The second probe — do the imagined latents drift to a fixed point?

Corroborative, **never adjudicating**. Measured on the INTACT arm at each α, per rollout step:

`||z_{j+1} − z_j|| / ||z_j||` · `||z_j − z_0||` · `cos(z_j, z_0)` · `||z_j||`, where `z_0` is the last
REAL percept.

| reading | criterion, fixed now |
|---|---|
| **FIXED POINT** | relative step size at step 40 < **5 %** of its step-1 value, **and** `||z_j − z_0||` plateaus (|Δ| over steps 100→185 < 5 % of its value at 100) |
| **NOT A FIXED POINT** | otherwise |

**A latent that stops moving is not predicting anything**, and that would independently corroborate
the integrator reading. Its failing value is reachable in both directions (a diverging latent gives a
growing step size; the statistic is a ratio and is unbounded above).

---

## 8. Priority order (a killed agent must still yield value) and what is NOT claimed

1. ⭐ **STAGE A — ZERO GPU, bankable immediately.** FROZEN at α = 0.25 (and 0, 0.75, 1.0) is
   **already measured** and sits in the committed Rung-1 dump as `b_blend0.25` etc. It alone answers
   half the PI's question and needs no new compute.
2. SHUFFLED + ZERO at α = 0.25 and α = 1.00.
3. MEAN, SHUF-REAL; α = 0.75 and α = 0.
4. The fixed-point probe.

**Deliberately NOT claimed, whatever the numbers say:**
* ⛔ One arm (v1), one action policy (`own_kinematic` + blend), one readout (`str`). Nothing on v4,
  REF-B or REF-C.
* ⛔ Not a safety result. PhysicalAI-AV ships no map, lane graph or agent boxes. Drift only.
* ⛔ The window set is EPISODE-INITIAL (596 of 599 at `t0 = 0`) and runs ~6–12 % low in absolute
  level (`INHERITED-MEASURED`). All contrasts are paired on identical windows.
* ⛔ Everything past 0.4 s is extrapolation for the `op` readout and past 2.0 s for `str`.
* ⛔ **A latent ablation cannot prove the latent is empty in general — only that it is not carrying
  THIS rollout's metric path.** The distinction is kept in the verdict wording.
