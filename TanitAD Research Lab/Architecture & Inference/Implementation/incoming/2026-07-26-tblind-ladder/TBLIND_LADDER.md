# THE PLAN TO EXTEND `T_blind` — pre-registered, adjudicated, ranked

**Date:** 2026-07-26 (Europe/Berlin; pods log UTC). **Stream:** blind-imagination driving.
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, written **before any number here existed**
(its mtime precedes every file in `artifacts/`). It is **not edited**; every deviation is an
amendment in §9.
**Host:** Rungs 0 and 2 are **zero-GPU on the dev box**. Rungs 0b and 1 ran on **pod2** (A40, verified
idle before launch: `0 MiB, 0 %`, no trainer). ⛔ pod1 (training), pod3 (control re-adjudication) and
the eval pod (situation-semantics) were **never touched**; the val cache was read only.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID and no raw content appears in this folder.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not
re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` · `CONFIRMED` · `DECISION-GRADE` (CONFIRMED + pre-registered + estimator
named + falsifier stated).
**Estimator, everywhere:** **paired episode-cluster bootstrap** — `taniteval/ci.py`, **B = 2000,
seed 0**, resampling unit = **episode cluster**, identical windows for every arm.
`overlapping_holdout_se` appears nowhere.

---

# 0. VERDICT

> ## ⭐ **RUNG 0 = CONFIRM. Decoding the blind rollout at the horizon it is read at MORE THAN TRIPLES the deployable `T_blind`: 0.8 s → 2.5 s [2.5, 3.9], and it FLIPS the sign of the program's own published contrast — at 2 s under the model's own actions, imagination goes from separated-WORSE than a frozen percept (−0.813 m) to separated-BETTER (+0.413 m). The horizon mismatch IS a primary cause, and the training fix is justified.**
>
> ## ⚠️ **AND THE QUALIFICATION IS PART OF THE HEADLINE, NOT A FOOTNOTE: this is an extension relative to a FROZEN PERCEPT ONLY. The deployable arm still NEVER beats constant velocity (0 / 185 steps, unchanged), and its absolute usefulness horizon barely moves (1.4 s → 1.4 s at the 1 m bar, 1.9 s → 2.1 s at 2 m). The world model now demonstrably adds something blind; it still does not add enough to beat assuming nothing changes.**
>
> ## ⭐⭐ **THE TWO LEVERS ARE ROUGHLY MULTIPLICATIVE, and that is the plan's core. Readout alone 0.8 → 2.5 s (3.1×). Action loop alone 0.8 → 3.2 s (4.0×). BOTH: 0.8 → 11.5 s [11.5, 17.4] — 14.4×, against 12.5× predicted by multiplying them. `MEASURED`, on identical windows, with matched comparators.**
>
> ## 🔴 **RUNG 1 IS THE ESCALATION: on v4's own checkpoint the swap moves `wm_canary_ade_2s` 1.1409 → 0.5446 (`tac`) / 0.5521 (`str`) — a 2.09× fall against Bar B's 2.074× requirement. ⚠️ It is a DECODER change, not a model change; the interval [0.4348, 0.6744] straddles the bar; and the two calibrated readouts land on OPPOSITE sides of it. It must NOT be reported as v4 passing Bar B. A registry decision is owed.**

| what the PI asked | the answer, `MEASURED` |
|---|---|
| *Do we have a plan to extend `T_blind`?* | **Yes — §7, eight rungs with costs and pre-registered falsifiers. The top rung costs ZERO and is already measured: +1.7 s of deployable `T_blind` for one line of code.** |
| *Which lever is primary?* | **Both are, and the answer depends on the metric — so all three readings are given.** On **`T_blind`**: action loop **+24 steps**, readout **+17 steps** (59 % / 41 %). On **`de@2s` level**: action loop **83.5 %**, readout **25.4 %**. On **absolute capability** (beats-CV, `T_useful`): action loop decisive, readout **~0**. **The action loop wins on all three — but the readout is a far bigger share of the horizon than of the level, which is exactly why the earlier level-only attribution under-rated it.** |
| *How long can it drive blind, today, deployably?* | **2.5 s [2.5, 3.9] before a frozen percept is as good** (with the calibrated readout, up from 0.8 s), **1.4 s before it is a metre off**, and **never** better than constant velocity. |
| *Best fully deployable configuration we have?* | **held-last action + the 20-step readout: `T_blind` = 11.5 s [11.5, 17.4], `de@2s` = 0.6718 m, `ade_0_2s` = 0.3351.** It beats the **PRIVILEGED as-shipped** arm at 2 s, paired-separated (Δ = +0.1418 m [+0.0743, +0.2058]). **No model in the control loop, no training, one line.** ⚠️ It is an *ablation that localises the fault*, not a shippable policy. |

**Tier.** `T_blind(own, calibrated readout) = 2.5 s` is **DECISION-GRADE**: pre-registered with buckets
fixed in advance, primary statistic named in advance, matched comparator **built** rather than
substituted, estimator named, falsifier stated **and demonstrated reachable**, validated in both
directions, and gated on window-set identity. ⚠️ The broader claim *"the horizon lever extends
deployable blind-driving capability"* is capped at **PROVISIONAL** by my own pre-registered conflict
rule — the primary says yes and two comparator-free statistics say no (§2.4). Rung 1's canary
movement is **CONFIRMED**; Rung 1's **bar clearance is NOT established**.

---

# 1. Pre-registration, and the two defects I had to handle before adjudicating

`PRE_REGISTRATION.md` fixed, before any computation: the primary statistic, the comparator problem,
the three outcome buckets, the tie rule, the C13 check, the Rung-1 bars and the Rung-2 attribution
metric.

## 1.1 ⛔ What the zero-GPU dump could NOT do — declared in advance, not discovered late

The brief asked for the **tactical (k=16) and strategic (k=20)** readouts. The committed dump holds
dense `de` for `a_imagination__own__roSTR` but **not** for `a_imagination__own__roTAC`, and — the
binding gap — **not for `b_frozenlast__own__roSTR`, the readout-matched frozen-last control.**
`bi_analyze.REGIMES["A2_readout_str_own_actions"]` declares `{"b": None}` and `analyse_sweep`
`continue`s on it, which is *why* `t_blind.json` carries five regimes and the deployable-calibrated
one is not among them. `MEASURED`, by loading both the script and the JSON. So Rung 0 ran the
zero-GPU version on `str`, and the four missing arms became **Rung 0b** (§3), a 17-minute pod job.

## 1.2 ⚠️ THE C13 CHECK — my rule's failing value, and proof it can return it

The brief required this because the blind-imagination agent's own pre-registration carried a **C13**
defect: a contiguity rule anchored at N = 1 that **could not fire**. Amendment **A4** moved the anchor
to N = 2. I re-derived the rule from its written specification rather than importing it, and stated
its failing value *before* computing:

* **`t_contiguous` returns 1 step (0.1 s) — never 0 — when the first evaluable horizon already fails.**
  That is the failing value.
* **The result that produces it, named in advance:** the `str` readout is worse than `op` at short
  horizon under own actions (measured here as **−0.0449 m [−0.0549, −0.0350] at 0.5 s**, separated).
  If that early penalty puts the arm behind its control at step 2, the rule returns 0.1 s.
* ⭐ **It did exactly that on the unmatched contrast** (`T_blind = 1 step`, `frac_draws_at_floor =
  1.000`), and **did not** on the matched one (25 steps, floor-frac 0.000). **The rule fires in both
  directions on real data.**

🔴 **And a residual C13 in the committed artifact, found by re-deriving rather than trusting:**
`t_blind.json` prints `frac_draws_T_blind_is_zero = 0.000` for **all five** regimes. Under A4 the
rule's minimum return is **1**, so that counter is **structurally guaranteed to be 0** and carries no
information whichever way the world is. The informative statistic is
`frac_draws_T_blind == 1 step`, which this stream emits instead. `artifacts/rung0_validation.json`.

## 1.3 ⛔ Validation in BOTH directions — run before any new number was read

| direction | test | required | result |
|---|---|---|---|
| **fidelity** | re-derive the 4 pre-registered `T_blind` + the A2 arm from the committed dump | 65 / 8 / 32 / 54 / 185 steps **and** their CIs | ✅ **all 5 exact, CIs exact** |
| **fidelity** | re-derive `ade_0_2s` for 8 committed arms | 0.3839 / 0.1865 / 0.1950 / 3.6093 / 0.9554 / 0.4712 / 0.5167 / 0.6083 | ✅ **all 8, max \|Δ\| < 5e-4** |
| **deliberate failure** | `de_a ≡ de_b` (Δ ≡ 0 everywhere) | must return **1 step** | ✅ 1 |
| **deliberate failure** | arms **swapped** (a known-worse arm fed as "imagination") | must return **1 step** | ✅ 1 |
| **deliberate failure** | a synthetic arm that wins **only from step 40** | must return **1 step** — contiguity must not be manufacturable | ✅ 1 |
| **positive control** | a uniformly-better synthetic arm | must **saturate** at 185, so the rule is not always-failing | ✅ 185 |

**`GATE_PASS = True`, `ALL_PASS = True`.** Only then was anything new computed.

---

# 2. ⭐ RUNG 0 — the adjudication

`MEASURED` · **599 windows / 596 episode clusters** · v1 = `flagship4b-speedjerk-30k` @ 29999 ·
`artifacts/rung0_tblind_deployable.json`, `artifacts/rung0c_matched_tblind.json`.

## 2.1 The zero-GPU primary returned the floor — and my own sensitivity showed it inadmissible

| contrast | `T_blind` | CI95 |
|---|---:|---|
| `a_imagination__own` vs `b_frozenlast__own` — committed | 8 steps (0.8 s) | [0.8, 1.0] |
| `a_imagination__own__roSTR` vs **`b_frozenlast__own` (op)** — P1, the zero-GPU primary | **1 step (0.1 s)** | [0.1, 0.1] |

⚠️ **P1's comparator carries the `op` readout, because the matched one had never been rolled out.**
**P1-S**, pre-registered for exactly this, measures what that substitution does in the **privileged**
regime where both comparators exist:

| privileged regime | comparator | `T_blind` |
|---|---|---:|
| `a__true__roSTR` vs `b__true__roSTR` — **matched** (committed) | str | **185 steps (18.5 s)** |
| `a__true__roSTR` vs `b__true` — **unmatched**, the same substitution P1 makes | op | **1 step (0.1 s)** |

⇒ **The comparator mismatch alone drives `T_blind` from the maximum to the floor.** My
pre-registration argued the substitution was a *conservative lower bound* from the measured sign of
the b-arm readout effect. The sign is right (`−0.287 … −2.799 m`, separated at every horizon); the
**magnitude makes the bound vacuous**. ⛔ **P1 is reported as computed and then SET ASIDE. It is not
evidence.** Rather than adjudicate on it, the missing arms were built (§3).

## 2.2 ⭐⭐ The matched adjudication — CONFIRM

Every row: **both** arms decoded with the **same** readout, so the arms differ in exactly one tensor.

| regime (matched comparator) | `T_blind` | CI95 | frozen percept overtakes at | paired Δ at 2 s (+ = imagination better) |
|---|---:|---|---:|---|
| **own \| `op` (k=4)** — as shipped | **8 steps (0.8 s)** | [0.8, 1.0] | **1.1 s** | ⛔ **−0.8133 [−0.9374, −0.6915]** |
| ⭐ **own \| `tac` (k=16)** — NEW | **21 steps (2.1 s)** | [2.1, 2.9] | 3.0 s | ✅ **+0.1825 [+0.0456, +0.3245]** |
| ⭐ **own \| `str` (k=20)** — NEW | **25 steps (2.5 s)** | **[2.5, 3.9]** | **4.0 s** | ✅ **+0.4130 [+0.2651, +0.5673]** |
| hold \| `op` | 32 steps (3.2 s) | [3.2, 5.0] | 5.1 s | ✅ +0.1618 [+0.0846, +0.2470] |
| ⭐ **hold \| `str`** — NEW | **115 steps (11.5 s)** | **[11.5, 17.4]** | 17.5 s | ✅ **+1.3785 [+1.2503, +1.5122]** |
| true \| `op` ⚠️ privileged | 65 steps (6.5 s) | [6.5, 8.9] | 9.0 s | ✅ +0.3505 [+0.2728, +0.4351] |
| true \| `str` ⚠️ privileged | 185 steps (18.5 s) ⚠️ **C14 saturated — a LOWER BOUND** | [18.5, 18.5] | never | ✅ +1.7286 [+1.6064, +1.8624] |

**Pre-registered buckets applied mechanically: `str` → 25 steps → CONFIRM (≥ 16). `tac` → 21 steps →
CONFIRM.** Baseline 8 steps.

## 2.3 🔴 A published headline of the sibling stream is decoder-conditional, and this corrects it

`BLIND_IMAGINATION.md` §0/§2.6b states, as its deployable headline, that imagination under the
model's own actions *"is separated-WORSE from 1.1 s onward"* — with the paired Δ at 2 s quoted as
**−0.813 [−0.937, −0.692] m**.

**That number is reproduced here exactly — and it is a property of the 4-step decoder, not of the
world model.** With a readout calibrated at the horizon it is read at, the same arm on the same
windows is **separated-BETTER at 2 s (+0.4130 [+0.2651, +0.5673])** and the frozen percept does not
overtake until **4.0 s** instead of 1.1 s. ⇒ **The sign of the program's deployable blind-driving
contrast flips with the decoder.** `MEASURED`, matched comparators, window-set identity gate passed.
This is escalation **E-2** (§8).

## 2.4 ⚠️ What did NOT move — reported in the same breath, per the pre-registered conflict rule

Two statistics need **no control arm at all**, so no readout mismatch can enter them. The CV floor is
pure kinematics and **has no readout**.

| deployable arm | `T_useful` < 1.0 m | < 1.391 m | < 2.0 m | separated-better than the CV floor |
|---|---:|---:|---:|---|
| own \| `op` | **1.4 s** | 1.6 s | 1.9 s | ⛔ **never — 0 / 185** |
| own \| `str` | **1.4 s** | 1.7 s | 2.1 s | ⛔ **never — 0 / 185** |
| hold \| `op` | 1.9 s | 2.3 s | 2.7 s | 0.4 – 3.8 s (35 / 185) |
| hold \| `str` | 2.3 s | 2.7 s | 3.2 s | **0.6 – 8.8 s (83 / 185)** |

⇒ **The pre-registered conflict rule fires.** The primary says the lever extends the horizon by 3.1×;
the two comparator-free statistics say the deployable arm's absolute capability is essentially
unchanged. **Neither is promoted over the other:** the honest statement is that the calibrated readout
makes the world model's dynamics **genuinely useful relative to a frozen percept** and **still not
competitive with constant velocity**. The combined capability claim is capped at **PROVISIONAL**;
the `T_blind` measurement itself is DECISION-GRADE.

## 2.5 The level effect on the deployable arm

| metric, own actions | `op` (k=4) | `str` (k=20) | paired Δ (+ = `str` better) | separated? |
|---|---:|---:|---|---|
| `de@0.5s` | **0.127** | 0.172 | ⛔ **−0.0449 [−0.0549, −0.0350]** — the swap is **WORSE** here | ✅ |
| `de@2s` | 2.158 | **1.817** | +0.3414 [+0.2083, +0.4802] | ✅ |
| `de@6s` | 16.570 | **14.003** | +2.5671 [+1.6622, +3.4951] | ✅ |
| `ade_0_2s` | 0.9554 | **0.8710** | +0.0844 [+0.0327, +0.1394] (**−8.8 %**) | ✅ |

`artifacts/rung0_own_readout_short_horizon.json`. **The short-horizon regression is real and is
reported**: a 20-step-calibrated decoder is worse at 0.5 s than a 4-step one, which is exactly what
"calibrated at a horizon" should mean, and it is why the unmatched contrast collapsed.

---

# 3. RUNG 0b/0c — the matched comparators, built because P1 was inadmissible

Four arms that had never been rolled out were built on pod2 with `bi_run.stage_sweep` **reused
verbatim** — only the arm list was replaced (`tb_rung0b_matched_arms.py`):
`b_frozenlast__own__roSTR`, `b_frozenlast__hold__roSTR`, `a_imagination__own__roTAC`,
`b_frozenlast__own__roTAC`.

⛔ **WINDOW-SET IDENTITY GATE.** The new arms are only poolable with the committed dense dump if the
window set is identical, so **two arms that already exist in it were re-rolled as anchors**:

| check | result |
|---|---|
| windows | **599 new vs 599 committed** ✅ |
| `eid` ordering identical | ✅ |
| `t0` ordering identical | ✅ |
| anchor `a_imagination__own` dense `de` | **max \|Δ\| = 3.05e-05 m** ✅ (float-kernel noise from a separate encode pass; tol 1e-4) |
| anchor `b_frozenlast__own` dense `de` | **max \|Δ\| = 3.05e-05 m** ✅ |

**`GATE_PASS = True`.** Timings, `MEASURED` (`artifacts/rung0b_sweep_meta_K185.json`): encode
**868.3 s**, rollout of 6 arms × 599 windows × 185 steps **103.4 s** — **~17 min** total on the idle
A40. **The encode pass dominates and is RAM-only, so it is repaid by every future pod rung; that is
why R1 and R5 are costed at ~20–30 GPU-min rather than ~2.**

---

# 4. 🔴 RUNG 1 — v4, and the decision the PI is owed

`MEASURED` on **pod2**, eval-only, **~2 GPU-min of rollout**: `flagship-v4-fromscratch-30k` @ **step
29999** — the checkpoint whose own `metrics.json` carries `canary_ade@2s = 1.1409059762954712`.
881 windows / **40 episode clusters**, the canonical gate deployment.
`artifacts/rung1_v4_readout_swap.json`, log `artifacts/rung1_v4_run.log`.

## 4.1 Both gates first

| gate | result |
|---|---|
| **reproduction** — the `op` readout must reproduce the committed v4 canary within ±0.01 m | **1.140906 vs 1.1409059762954712, `abs_diff = 0.000000`** ✅ **exact** |
| **deliberately failing input** — a randomly re-initialised readout must be much worse | **16.113 (14.1× worse)** ✅ — the harness genuinely reads the `level` argument, so three near-equal numbers could not have been a silent no-op |

## 4.2 The swap on v4

| readout | trained rollout length | `wm_canary_ade_2s` | CI95 | fall vs `op` | ≤ 0.55 ? |
|---|---:|---:|---|---:|---|
| `step["op"]` — **what the gate used** | 4 steps | **1.140906** | [0.9935, 1.3025] | 1.000× | ❌ |
| **`step["tac"]`** | 16 steps | **0.544631** | **[0.4348, 0.6744]** | **2.0948×** | ⚠️ **point yes** |
| `step["str"]` | 20 steps | **0.552132** | [0.4466, 0.6772] | 2.0664× | ❌ **point no** |

Paired vs `op`: `tac` **+0.5963 [+0.4730, +0.7360]** ✅ separated (p = 1.0000);
`str` **+0.5888 [+0.4655, +0.7280]** ✅ separated. **Bar B requires 2.0744×; measured 2.0948×.**

## 4.3 ⛔ What may and may not be said about that

* ✅ **MAY:** *the swap MOVES v4's canary, decisively and paired-separated — a **2.09×** fall, the same
  ratio v1 showed (0.3839 → 0.1865 = **2.058×**). **Two independent checkpoints, same effect size.***
  That is the **CONFIRMED** transfer Rung 1 was asked to settle, and it is a real finding: the
  readout-calibration penalty is a ~2.07× property of the *recipe*, not of one arm.
* ⛔ **MAY NOT:** *"v4 clears Bar B."* Three reasons, each sufficient:
  1. **The margin is noise.** `tac` clears by **0.0054 m = 1.0 %**; its 95 % interval **[0.4348,
     0.6744]** reaches **23 % above** the bar.
  2. **The two calibrated readouts STRADDLE the bar** (0.5446 under, 0.5521 over). A bar that one
     sibling decoder clears and the other misses by 0.4 % is not being cleared by the model.
  3. 🔴 **It is a DECODER change, not a model change.** The checkpoint is bit-identical. Bar B was set
     on the `op`-decoded quantity and the whole historical series (`v1 0.4271`, `v4.2 0.7222`, the
     `≤ 0.55` bar itself) is `op`-decoded. **Swapping the decoder to clear a bar is metric
     redefinition.**

⛔ **v4's RESTART decision must not be re-opened on this number.**

---

# 5. RUNG 2 — the attribution that sizes the levers

`MEASURED`, zero-GPU · `artifacts/rung2_decomposition.json` (level) and
`artifacts/rung0c_matched_tblind.json` (horizon) · the **2 × 3 factorial on arm (a)**, all six cells on
identical windows.

## 5.1 On the HORIZON — `T_blind`, matched comparators

| | `op` (k=4) | `str` (k=20) | readout gain |
|---|---:|---:|---:|
| **own actions** (deployable) | **8 steps** | **25 steps** | **+17 (3.1×)** |
| **held last action** (deployable) | **32 steps** | **115 steps** | **+83 (3.6×)** |
| **true actions** ⚠️ privileged | 65 steps | 185 ⚠️ saturated | ≥ +120 |
| **action gain (own → hold)** | **+24 (4.0×)** | **+90 (4.6×)** | |

> ### ⭐⭐ **The two levers are roughly MULTIPLICATIVE. 8 → 25 (readout, 3.1×) → 32 (action, 4.0×) → 115 with BOTH (14.4×), against 12.5× predicted by multiplying them.** This is the single strongest argument in the plan for doing **both**, and it is measured rather than modelled.

## 5.2 On the LEVEL — `de@2s`, comparator-free

| action source | `op` | `str` | readout effect | relative |
|---|---:|---:|---|---:|
| own kinematic — deployable | **2.1580** | **1.8165** | +0.341 [+0.208, +0.480] ✅ | −15.8 % |
| held last action — deployable | **1.0349** | **0.6718** | +0.363 [+0.316, +0.414] ✅ | −35.1 % |
| true actions — ⚠️ privileged | **0.8136** | **0.3112** | +0.502 [+0.459, +0.546] ✅ | −61.8 % |

**Interaction:** `true − own` = +0.161 [+0.034, +0.279] ✅ separated; `hold − own` = +0.022
[−0.107, +0.142] ⛔ not separated. ⇒ **On the LEVEL the readout gain is roughly CONSTANT (~0.34–0.50 m)
regardless of who picks the actions** — the monotone *relative* picture comes from base levels
differing 2.7×, not from the lever switching off. **On the HORIZON it compounds instead.** Both are
true; they are different questions and are reported as such.

## 5.3 The gap decomposition the brief asked for

Deployable **2.1580 m** → privileged **0.8136 m** at 2 s: gap **1.3444 m**.

| fix | `de@2s` | gap closed | `ade_0_2s` | gap closed | deployable `T_blind` |
|---|---:|---:|---:|---:|---:|
| **(a) fix the READOUT horizon only** | 1.8165 | **25.4 %** | 0.8710 | 14.8 % | **0.8 → 2.5 s** |
| **(b) fix the ACTIONS only** (held-last — the ablation that already existed) | 1.0349 | **83.5 %** | 0.4712 | 84.7 % | **0.8 → 3.2 s** |
| ⭐ **(c) BOTH** | **0.6718** | **110.5 %** | **0.3351** | **108.5 %** | ⭐ **0.8 → 11.5 s** |
| (d) privileged actions + fixed readout | 0.3112 | 137.4 % | 0.1950 | 133.1 % | ≥ 18.5 s (saturated) |

**Policy penalty (own − hold) = 1.123 m (`op`) / 1.145 m (`str`) at 2 s = 83.5 % / 76.0 % of the
own→true gap**, separated in both.

> ### ⭐ **(c) is fully DEPLOYABLE — no model in the control loop, no training, one line — and at 2 s it BEATS the PRIVILEGED as-shipped arm, paired-separated: `de@2s` Δ = +0.1418 m [+0.0743, +0.2058], `ade_0_2s` Δ = +0.0488 [+0.0192, +0.0774].**

⚠️ **The advantage is a 2 s statement:** at **6 s** the same contrast is **+0.1405 [−0.5003, +0.7896]
— NOT separated.** So (c) *matches* rather than beats the privileged arm at longer horizon.
⚠️ **And "held last action" is a no-policy controller, not a planner.** It is an *ablation that
localises the fault*, not a shippable driving policy. What it proves is that **the fault is in the
action feedback, not in the perception loss.**

---

# 6. What the plan is aimed at, in one paragraph

The deployable blind-driving deficit has **two** binding causes and they compound. **(1) The operative
brain was only ever trained to imagine 0.4 s, and every grounded number is decoded 5× beyond that
calibration** — fixing the *read* alone triples the deployable horizon and flips the sign of the
program's own deployable contrast, for zero GPU. **(2) The control loop compounds the model's own
action error** — a controller that merely stops compounding is worth 4× more `T_blind` than the model
choosing. Doing both is worth **14.4×**. The third cause — the decoder's attachment to the imagination
manifold rather than to perception — is untouched by either and is the reason the arm still loses to
constant velocity.

---

# 7. ⭐⭐ THE LADDER

**Ranking criterion, stated so it can be argued with: expected DEPLOYABLE `T_blind` seconds first,
GPU cost second.** Every row states what result would **kill** it.

| # | rung | GPU cost | expected deployable gain, and its evidence class | ⛔ pre-registered falsifier |
|---|---|---|---|---|
| **R0** | ⭐⭐ **ADOPT THE CALIBRATED READOUT wherever a grounded rollout is decoded** — `taniteval/rollout.py:collect`, both `canary_rollout`s, `eval_grounded_rollout_4b*`. One line each. **Already measured; this is an adoption decision, not an experiment.** | **ZERO** | **+1.7 s** deployable `T_blind` (0.8 → 2.5 s [2.5, 3.9]) and a **sign flip** on the 2 s frozen-percept contrast. `MEASURED`, matched comparators, 596 clusters. On v4 it also moves the canary **2.09×**. | Not an experiment — **but it is BLOCKED on the registry decision in §8 E-1**, because adopting it silently would put two values of `wm_canary_ade_2s` into circulation for the same checkpoint. ⚠️ Adopt it **with** a re-derived Bar B, or not at all. |
| **R1** | 🔴 **Re-run the sweep with the action from v1's TACTICAL PLANNER** (`closedloop.wp_to_control`) instead of the kinematic inverse, plus a **rate-limited** variant — with the **calibrated** readout throughout. The cheapest discriminating experiment on the co-primary lever. | **~30 GPU-min, eval-only** (the 600-ep encode pass is 14.4 min and dominates) | **up to +9.0 s** — `MEASURED` ceiling, not modelled: the `hold \| str` cell already reaches **11.5 s** on identical windows. Only the mechanism is open. | **REFUTE** the planner hypothesis if planner-derived actions do not beat `own_kinematic`'s matched **2.5 s** `T_blind`. If they do not, closing the loop is intrinsically unstable at this scale and R3 becomes the only action-side path. |
| **R2** | ⭐ **Train the operative readout AT the horizon it is read at** — `--op-fwd-k 20`. `grounding_losses` **already** rolls the operative predictor to `k_max = max(fwd_k) = 20` every step, so this adds **no rollout**, only 16 readout applications and their loss terms. R0 made intrinsic. | **≈ 0** (~1–2 % step time) | `HYPOTHESIS` that it exceeds R0 — R0's eval-only gain is the direct evidence that 4-step calibration binds, but training at 20 could also *cost* short-horizon accuracy (R0 already loses 0.045 m at 0.5 s). | **REFUTE** if `wm_canary_ade_2s` at a matched step does not fall ≥ 10 % against a same-seed `op_fwd_k = 4` control **decoded with the same readout** (otherwise the control is not a control). ⚠️ **Second bar this stream adds:** if `de@0.5s` regresses by > 50 %, the fix trades the horizon we can already buy for free against accuracy we cannot. |
| **R3** | ⭐ **Action-channel scheduled sampling / student forcing.** With probability *p* the predictor is fed its OWN decoded action instead of the logged one, ramped over training. The classic teacher-forcing repair, aimed at the exact tensor §5.3 localises the fault in. | ⚠️ **~1–3 % step time** (the rollout and the readout already run; the extra work is the kinematic inverse per step — **not costed as zero**) **+ one full run (~59 h)** | **up to +9.0 s**, same measured ceiling as R1 but *intrinsic* rather than via a hand-built controller | **REFUTE** if, at a matched step against a same-seed `p = 0` control, deployable `T_blind` does not exceed **3.2 s** (the held-last no-policy value — a learned fix that cannot beat "hold the last action" is not a fix). ⚠️ Gated behind **R1**: if the planner already wins, R3's target narrows to the planner's own divergence. |
| **R4** | ⭐ **Train the step readout on REAL latent pairs, not only imagined ones** — with probability *p*, decode `(z_real_t, ẑ_{t+1})`. The real states are **already encoded** for the metric-inverse-dynamics term, so the cost is one extra readout call. Attacks the **9.4× manifold gap** — the one cause R0–R3 leave untouched, and the reason the arm still loses to constant velocity. | **≈ 0** + one run | `ESTIMATED` — no direct `T_blind` measurement exists, but three independent symptoms share this one cause (9.4× on real pairs; perception hurts; peeking backfires), so one fix could move all three | **REFUTE** if, after training, (i) the FULL-OBSERVATION arm does not overtake imagination at 2 s **and** (ii) the oracle-vs-uniform gap at matched duty does not turn positive. **Both, not either.** |
| **R5** | **Re-measure TRIGGER HEADROOM with the calibrated decoder.** ⚠️ It is currently **negative** (a perfect error oracle is **25–112 % worse** than a fixed clock at matched camera budget) — but that was measured with the `op` decoder, and **every peek arm in the committed dump is `op`-decoded.** §2.3 has now shown one headline of that same run to be decoder-conditional, so this **must** be re-measured, not inherited. | **~20 GPU-min** (encode 14.4 min dominates; 6 policies × 2 bases add ~2 min) | re-opens or closes a whole workstream | **CLOSE the trigger line for good** if the oracle-vs-uniform gap is still **< +15 %** at matched duty with the calibrated decoder — the blind-imagination stream's own pre-registered DISAPPOINTING bar, re-applied. **Do not build a learned trigger before this returns positive.** |
| **R6** | **Off-path augmentation** (one homography per sample). A blind rollout leaves the logged path **by construction**: `INHERITED-MEASURED` **20 %** of steps outside the measured envelope by 6 s, **52 %** by 12 s — and R0/R1 push the usable horizon *into* that region, so the exposure grows as the plan succeeds. | one homography per sample + a full run | `ESTIMATED` — the exposure is measured, no intervention has been run against it | **REFUTE** if `frac_steps_out_of_envelope` at 6 s does not fall **and** `de@6s` does not improve. |
| **R7** | **Longer predictor horizons** (`[1,2,4]` → `[1,2,4,8,16]`). New JEPA heads ⇒ a graft or a restart. | **a full run (~59 h)** | — | Deliberately **last**. R0 costs zero and buys 3.1×; R2/R3/R4 test overlapping hypotheses for ~0 marginal cost. Spending a 59-hour run first is the exact failure `BOOST_PROGRAM` §3.4 exists to prevent. |

## 7.1 What is deliberately NOT on the ladder

* ⛔ **A learned peek/escalation trigger** before **R5** returns positive. Negative headroom,
  `INHERITED-MEASURED`.
* ⛔ **Presenting the decoder swap as a Bar-B fix.** §4.3.
* ⛔ **A new corpus, and a renderer.** Nothing measured here implicates data volume or photorealism in
  blind driving.
* ⛔ **Any architecture change before R0–R4.** One is free, one is eval-only, two are ~free marginal.

## 7.2 The re-aiming, in one line

> **Before this stream the deployable answer was "imagination is worse than a frozen percept from
> 1.1 s and never beats constant velocity". After it: with a decoder read at its own calibration the
> deployable arm is BETTER than a frozen percept out to 2.5 s and the frozen percept does not overtake
> until 4.0 s — for zero GPU — and fixing the action loop as well is measured to be worth 11.5 s.
> What has NOT changed is the constant-velocity floor, and R4 is the rung aimed at it.**

---

# 8. 🔴 ESCALATIONS — raised in the headline, not written into a README

**E-1. A REGISTRY DECISION IS OWED, and R0 is blocked on it.** `wm_canary_ade_2s` now has **two**
values for the same v4 checkpoint — **1.1409** (`op`, 4-step-calibrated) and **0.5446** (`tac`) — and
both are correct measurements of differently-decoded quantities. **Does the program's headline metric
keep the `op` decoder, or move to the calibrated one; and if it moves, is Bar B re-derived on the new
decoder?** Either answer is defensible; **having both in circulation is not**, and a 1 %-margin "pass"
produced by changing the decoder is precisely how a bar stops meaning anything. This is the
blind-imagination stream's escalation **E2**, now unavoidable because it has a number attached.

**E-2. A published deployable headline is decoder-conditional and needs a correction, not a
retraction.** `BLIND_IMAGINATION.md` §0/§2.6b's *"imagination under its own actions is separated-WORSE
from 1.1 s onward, Δ = −0.813 m at 2 s"* reproduces exactly and is **a property of the 4-step
decoder**: with a matched calibrated readout the same arm on the same windows is **separated-BETTER at
2 s (+0.413 m)** and is not overtaken until **4.0 s**. The stream's Outcome-B verdict *does* survive on
the constant-velocity contrast (still 0/185), so the conclusion is narrowed rather than withdrawn —
but the sentence as written is no longer quotable without the decoder named.

**E-3. Two retraction-log rows are owed** (§9.2). They are drafted rather than filed, because
`RETRACTION_LOG.md` is a shared append-only steering document and other agents are writing this
session.

## 8.1 What this unblocks, per stream

| stream | what it gets |
|---|---|
| ⭐ **the three-planner / hierarchy direction** | the action loop is sized as a **co-primary** lever (+24 steps of `T_blind` alone, +90 with the calibrated readout), and **R1 is a 30-GPU-min eval-only experiment aimed straight at it** with a pre-registered bar (must beat 2.5 s). |
| 🔴 **`MODEL_REGISTRY` / the gate card** | E-1. Until decided, two correct values circulate for the same checkpoint. |
| **S-2 / v4 (Bar B)** | the v1 lever **transfers to v4 at the same ~2.07× ratio**, CONFIRMED with an exact reproduction gate — **but it lands ON the bar with a straddling interval and it is a decoder change.** The honest input to the restart decision: *this does not make v4's world model healthy.* |
| **S-1 closed-loop measurability** | a deployable configuration with `T_blind` **11.5 s [11.5, 17.4]** and `de@2s` beating the privileged as-shipped arm — a concrete reference point that needs no planner. |
| **H2 / sensor-need gating** | ⛔ still closed, but the reason is now **provisional**: the negative headroom was measured with the `op` decoder, and §2.3 shows that decoder can flip a sign. **R5** is the 20-GPU-min re-measurement that must precede any trigger work. |
| **the blind-imagination stream** | its amendment **A2** asked for one independent re-run of the readout lever before it decides anything. It now has **two** — a second checkpoint (v4, exact gate) and the matched-comparator build its own analysis could not do. The lever is **CONFIRMED**. |

**What it unblocks nowhere:** nothing here touches data ingest, the corpus question, Orin/Thor, or
AlpaSim.

---

# 9. Amendments and owed retractions

## 9.1 Amendments to my own pre-registration — recorded here, not by editing it

| # | what changed | why, and what it can and cannot bias |
|---|---|---|
| **B1** | **The primary contrast's comparator was BUILT (Rung 0b) rather than substituted.** The pre-registration named P1 (unmatched) primary and P1-S as its sensitivity. | P1-S fired: the substitution is destructive, not conservative, so P1 could not decide. Building the matched arm is the **least-assumption** repair — it removes the confound instead of arguing about its size. ⚠️ **What it can bias:** the new arms come from a second encode pass, so they are not bit-identical to the committed ones; the **window-set identity gate** (§3) bounds that at **3.05e-05 m** and the buckets were **not** touched. |
| **B2** | `roTAC` under own actions was **added** to Rung 0b beside the pre-registered `roSTR`. | The brief asked for both readouts; only `str` existed at zero GPU. It is reported beside `str`, never instead of it, and both land in the same bucket (21 and 25 steps), so no selection is possible. |
| **B3** | The **conflict rule** (§2 of the pre-registration) was applied to a *primary-positive / secondary-null* split, which is not literally "disagree in direction". | Applied in the **stricter** direction: the combined capability claim is capped at PROVISIONAL rather than promoted. Stated so a reader can disagree with the call and still see both numbers. |

## 9.2 🔴 Retraction-log rows this stream owes

⛔ **Escalated, not filed** — `Project Steering/RETRACTION_LOG.md` is a shared append-only document and
other agents are writing this session. Both are **root-cause-class** entries, per operating-standard
rule 4.

| date | what is withdrawn | root-cause class | the correction |
|---|---|---|---|
| 07-26 | `t_blind.json`'s `frac_draws_T_blind_is_zero = 0.000`, printed for **all five** regimes and read as "no bootstrap draw collapsed" | **C13 — a diagnostic that cannot fire**, *and a RECURRENCE: it is the residue of the very C13 that amendment **A4** was written to repair* | Under A4 the rule's minimum return is **1**, so a zero-counter is **structurally impossible**. **BINDING LESSON: when you repair a criterion, re-check every DIAGNOSTIC derived from its old range — the fix moves the floor and the old counter silently becomes a tautology.** Replaced by `frac_draws_T_blind == 1 step`. |
| 07-26 | **my own** pre-registered claim that substituting an `op`-decoded frozen-last control is a **conservative LOWER BOUND** on `T_blind` | **C-BOUND — a bound whose SIGN was verified but whose MAGNITUDE was not** | The sign was right; the magnitude makes it worthless — in the privileged regime the same substitution moves `T_blind` from **185 steps to 1**. **BINDING LESSON: a directional bound is admissible only once its MAGNITUDE is bounded too, or "conservative" licenses quoting a floor value as evidence.** Caught by the pre-registered sensitivity and repaired by building the missing arms. |

---

# 10. Limitations, stated plainly

1. ⚠️ **`own_kinematic` is one action policy, not "the" deployable policy.** Its `accel` half is not an
   exact inverse of the corpus convention (`INHERITED-MEASURED`: +0.137 m at 2 s), and a *different*
   deployable policy is 4× better. **The deployable numbers are a property of this arm AND this
   controller.** R1 separates them.
2. ⚠️ **The window set is EPISODE-INITIAL** (596 of 599 windows at `t0 = 0`) and runs ~6–12 % low in
   absolute level (`INHERITED-MEASURED`). **The CV floor runs low by twice as much as the model does**,
   so "never beats CV" is if anything conservative. All contrasts are paired on identical windows.
3. ⚠️ **Everything past 0.4 s is EXTRAPOLATION** — including the 2 s canary the program quotes. Past
   12 s a majority of steps are outside the measured envelope. **The 11.5 s `hold | str` `T_blind` sits
   deep in that region and is labelled accordingly.**
4. ⚠️ **`true | str` saturates at the sweep terminus** — **C14**: a LOWER BOUND on our configuration,
   not a horizon.
5. ⚠️ **Rung 1 is 40 episode clusters.** Its CIs are wide ([0.43, 0.67]); that width is *why* §4.3
   refuses the bar-clearance claim.
6. **`tac` and `str` differ by 0.0075 m on v4 and 4 steps of `T_blind` on v1** — inside their
   intervals. Nothing here supports a preference between them; the point is that **both** beat 4-step.
7. **One arm per rung.** v1 for Rungs 0/0b/0c/2, v4 for Rung 1. Nothing on REF-B or REF-C.
8. **No safety metric exists here** — PhysicalAI-AV ships no map, lane graph or agent boxes. This is a
   **drift** measurement, not a collision one.
9. **The new matched arms come from a second encode pass**, bounded at 3.05e-05 m by the identity
   gate — not bit-identical, and the gate is the only thing that makes them poolable.

---

# 11. Deliverable manifest

**Everything is in the repo working tree and STAGED (`git add`). Nothing was committed or pushed.**
Path: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-tblind-ladder/`

| artifact | what it is | where it lives |
|---|---|---|
| `PRE_REGISTRATION.md` | written **before** any number existed: primary statistic, the comparator problem declared in advance, the three buckets, the tie rule, the C13 check with its failing value, the Rung-1 bars, the Rung-2 attribution metric | **repo** |
| `TBLIND_LADDER.md` | this document | **repo** |
| `scripts/tb_rung0.py` | ⭐ Rung 0 — re-derives the `T_blind` rule from spec, runs the **fidelity gate** (5 `T_blind` + 8 `ade_0_2s`) and **four deliberately-failing inputs**, then P1 / P2 / P1-S | **repo** |
| `scripts/tb_rung0b_matched_arms.py` | Rung 0b driver — `bi_run.stage_sweep` reused verbatim with the four missing matched arms + two anchors | **repo** · `pod2:/root/tbl/` |
| `scripts/tb_rung0c_matched_verdict.py` | ⭐ Rung 0c — the window-set identity gate and the matched-comparator adjudication | **repo** |
| `scripts/tb_rung1_v4.py` | ⭐ Rung 1 — self-contained v4 loader, per-window canary at any readout level, the reproduction gate and the random-readout failing control | **repo** · `pod2:/root/tbl/` |
| `scripts/tb_rung2_decomp.py` | ⭐ Rung 2 — the 2 × 3 factorial, main effects + interaction, gap attribution, `T_useful`, headline contrasts | **repo** |
| `artifacts/rung0_validation.json` | ⛔ the both-direction validation | **repo** |
| `artifacts/rung0_tblind_deployable.json` | Rung 0 — P1, P2, P1-S, arm levels, the mechanical verdict | **repo** |
| `artifacts/rung0_own_readout_short_horizon.json` | the 0.5 s regression and the `ade_0_2s` paired test | **repo** |
| `artifacts/rung0c_matched_tblind.json` | ⭐ **the headline** — matched `T_blind` for 7 regimes + the identity gate | **repo** |
| `artifacts/rung1_v4_readout_swap.json` | ⭐ Rung 1 — v4's canary at `op`/`tac`/`str` with intervals, both gates, the verdict | **repo** (pulled off pod2) |
| `artifacts/rung1_v4_run.log` · `artifacts/rung0b_run.log` | the pod2 run logs, incl. the parity checks and all gates | **repo** (pulled off pod2) |
| `artifacts/rung0b_sweep_meta_K185.json` | the Rung-0b run manifest: arms, parity block, timings | **repo** (pulled off pod2) |
| `artifacts/rung2_decomposition.json` | Rung 2 — factorial, interaction, gap attribution, `T_useful` | **repo** |
| `perwindow/perwindow_matched_K185.pt` (2.7 MB) | ⭐ **dense per-window `de` [599 × 185] for the 4 new matched arms + the 2 anchors** — any bar, horizon or stratification recomputable with no GPU | **repo** · `pod2:/root/tbl/perwindow/` |

**Living in only ONE place (declared, per rule 2):** the full `perwindow_sweep_K185.pt` (13.8 MB, with
`psi` and `pred_speed`) exists only at `pod2:/root/tbl/perwindow/`. **The 2.7 MB dense-`de` compaction
in the repo carries everything every number in this report needs**; the full dump rebuilds in ~17 min
on an idle A40 with `tb_rung0b_matched_arms.py` (deterministic, `torch.manual_seed(0)`, no sampling).

**Suites:** no file under `stack/` or `taniteval/` was modified — every deliverable is a new file in
the Research Hub folder — so neither suite's behaviour can change. `taniteval` re-run for
confirmation: **423 passed**.

---

# 12. Reproduction

```
# Rungs 0 and 2 — ZERO GPU
python scripts/tb_rung0.py        --dump ../2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt --out artifacts
python scripts/tb_rung2_decomp.py --dump ../2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt --out artifacts

# Rung 0b — pod2, ~17 min: build the four missing readout-matched controls
PYTHONPATH=/root/bi:/root/TanitAD/stack:/root/TanitAD/stack/scripts:/root/taniteval OMP_NUM_THREADS=8 \
python3 scripts/tb_rung0b_matched_arms.py --out perwindow --episodes 600 --kmax 185

# Rung 0c — ZERO GPU: the identity gate + the matched adjudication
python scripts/tb_rung0c_matched_verdict.py --new <rung0b>/perwindow_sweep_K185.pt \
  --committed ../2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt --out artifacts

# Rung 1 — pod2, eval-only, ~2 GPU-min of rollout
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts:/root/taniteval \
python3 scripts/tb_rung1_v4.py \
  --ckpt /workspace/experiments/flagship-v4-fromscratch/ckpt.pt \
  --val-cache /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
  --out artifacts/rung1_v4_readout_swap.json
```
