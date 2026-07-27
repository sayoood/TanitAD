# flagship-v5 retrain — PREPARATION CARD (not yet a launch card)

**Written 2026-07-27 on the PI's instruction: *"do what necessary to prepare v4 retrain after validated
improvements."*** The conditional is the whole design. **This document does not authorise a run.** It
states what is validated, what is not, what must land first, and the bars — all fixed **before** any
checkpoint exists, because a threshold chosen after seeing a number is the forking-paths failure
`GATE_PROTOCOL` §0.3 forbids.

**Restart budget: 0 of 2 used.** This would spend one.

---

## 0. Why the previous run failed — the four mechanical causes, all now fixed or fixable

| # | cause | MEASURED | status |
|---|---|---|---|
| 1 | **No held-out early-stop signal** | **~29.5 GPU-h — half the run — spent training past the best checkpoint**; held-out selection was separated-*worse* while every training term improved | ✅ M4 row-writer fix landed; **mid-run gate specified in §4** |
| 2 | **The four diagnostics that would have caught it were discarded** | computed **601×** and thrown away by the row-writer | ✅ fixed |
| 3 | **No grounding instrument at all** | all three v4 logs carry **no `g_*` key**; the real-vs-imagined gap is *unmeasurable* on v4 | 🔄 being added (~20 GPU-min) |
| 4 | **The selector's target is nearly noise** | trained against "closest to the single realised future"; an **in-sample** re-scorer with zero generalisation gap bottoms out at **0.4907** vs v1's 0.4271 | ⛔ **T3 REFUTED — the composite gave the same answer; selector change struck** |

⚠️ **And a fifth, which is not a bug but a wrong axis:** the run was gated on `ade_0_2s`. **Two
independent lines now say that is the wrong target** — the **ADE-optimal pick collides 4.7× more often
than the rule-optimal pick** (3.36 % vs 0.71 %, separated), and published **L2/ADE vs closed-loop
Driving Score is ρ = −0.36, p = 0.43** while **Ego Progress is ρ = 0.83**.

---

## 1. VALIDATED — changes with measured evidence, admissible for the run

| change | evidence | cost |
|---|---|---|
| **Reachability clamp on the offset head** | 72.08 % of candidates removed, **oracle survives 100 %**, paired **Δ exactly 0.0000**, **3.58× cheaper**. Anchors are **bitwise identical to real human windows** (256/256) — **the vocabulary is fine; the unbounded offset head emits 171.5 km/h** (p99 159.6 vs a val GT max of 132.4) | free |
| **Vision enters at rank ≈ 16** | monotone dose–response **3.659× → 3.685× (k16) → 3.000× (k64) → 2.116× → 1.59× (k2048)**; degradation begins at k=64; 16 PCs carry **97.0 %** of variance. **Replicated by two independent streams**, all ten arms selecting r=16 | free |
| **Calibrated readout selection (C8)** | `op` for lead ≤ 0.5–0.8 s, `str`/`tac` beyond — **Pareto-dominant** over the flat swap (0.8597 vs 0.8710 at identical `T_blind`). ⚠️ **Fitting the switch to ADE costs 3.1× of `T_blind` to buy 0.7 % of ADE** — do not fit it to ADE | free |
| **`imagine_candidates`** | the old `imagine_probes` returned **32 tokens invariant to candidate count** — no candidate axis existed, making E-V5-1's failure over-determined | ~0, paid for by the clamp |
| **Emit the grounding instrument** | without `g_*` keys v4 cannot be diagnosed at all | ~5 lines |
| **Emit the four selection diagnostics** | already computed every step | log-only |

## 2. NOT VALIDATED — must not enter the run on current evidence

- **C1 (inverse-dynamics consistency on imagined transitions)** — X1 scoped it to **k ≤ 2 only**: the head is mis-attached at 0.1–0.2 s but **at or above the frozen-latent ceiling from 0.4 s** (at k=20 the head *beats* every probe by 19.4 %). **A rollout-quality guard is mandatory** — PlaNet Fig. 7 is on record as counter-evidence that overshooting *hurt* their RSSM.
- **C3 (teach the encoder to perceive ego-motion)** — X1 calls this *the real item*, and it is not started. A **one-scalar `v0` integrator beats every latent probe by up to 24.8×**, and the second latent slot adds ~nothing.
- **λ/τ prior strength** — the curve **was not measured**; we have **two points on the soft side** and cannot locate the optimum.
- **The action filter** — ⚠️ at the headline α = 0.75 the model supplies only 25 % of the command and **pure hold-last (0 % model) already reaches 11.5 s with a *better* `de@2s`.** The genuinely model-driven point is **α = 0.25 → 8.5 s**. This is an **inference-time** filter and does not belong in a training card at all.
- **R3 / scheduled sampling** — **condemned**: the free filter recovers **101.1 %** of its lever, and no method in its family attacks an oscillation.

## 3. GATING LIST — what must land before this becomes a launch card

1. ~~**T3's verdict**~~ — ⛔ **LANDED 2026-07-27: REFUTE on both pre-registered legs.** `BCE_RULE − CE_CONTROL` on **PDMS-lite** (the pre-registered primary, not ADE) is **+0.0002 [−0.0025, +0.0031], not separated**; at-fault collision **+0.0000 [−0.0035, +0.0035]**. ⭐ **And neither fitted arm beats doing nothing** — `AS_TRAINED` holds the best PDMS-lite (0.6100), the **lowest collision rate** (0.0361) *and* the best ADE. **Bar A refuted a re-scoring lever on ADE, the standing objection was that ADE is the wrong axis, we changed the axis to the composite — and the answer did not change.** ⇒ **THE SELECTOR CHANGE IS NOT VALIDATED AND IS STRUCK FROM §1.** ⚠️ Read with the composite's resolution: random scores 0.3968 and the three trained arms occupy **0.6096–0.6100 — 0.2 % of the distance to random**; DAC is missing and **comfort is a literal constant on v4's fan (100.0000 % violation over 1,708,288 candidates)**, so two PDMS terms carry no information. The refutation is sound; the instrument's selector-vs-selector resolution is weak, and both are stated.
2. **v4's grounding instrument** (running) — without it the run is undiagnosable, exactly like the last one.
3. **The clamp's zero-change property on v4's own fan** (running) — it is proven on REF-C-XL, **not** on v4, and turning it on correctly broke two v4 tests.
4. **The latent-ablation verdict** (running) — if blind driving is kinematic integration rather than imagined semantics, the world-model objectives need re-aiming before a GPU-week.
5. **The pseudo-simulation arm panel** (running) — v4-30k is currently **indistinguishable from constant velocity** and **worse than never steering on recovery**. A retrain must state what it expects to move on *that* instrument.

## 4. THE BARS — registered now, before any checkpoint exists

**Primary: the map-free composite, NOT `ade_0_2s`.** ADE is reported as a **diagnostic only**, for the reasons in §0.

**Mid-run held-out gate (this is the fix for cause #1):** probe the **deployable surface** on held-out
episodes at a fixed step cadence; **stop the run when the held-out primary is separated-worse for two
consecutive probes.** This converts a 59-hour loss into ~20 hours.

**Kill secondaries carried forward verbatim** from the pre-registered v4 card — no threshold below is
new: `wm_canary_ade_2s ≤ 0.55` · `miss_at_2m ≤ 0.10` · `seam_norm_ratio_max ≤ 1.0` ·
`encoder_touching_levers ≤ 2`.
⚠️ **`speed_benefit_recovered_frac` and `deploy_tick_p99_ms` have NO EMITTER** — they were recorded
`null` last time, and the run was called a *"formal 8-metric gate"* when it was a **6-metric gate**.
**Either build the emitters or strike them from the card. Do not carry an unmeasurable criterion.**
⚠️ **`nonav_route_beats_majority` remains VOID BY CONSTRUCTION** → INSTRUMENT-FAIL, never MODEL-FAIL,
and **must be printed** in the verdict.

**Horizon:** register **K = 60 (6.0 s) primary, K = 70 hard maximum**. Above K=70 the junction stratum
falls below 200 clusters on the maximum-possible corpus, so **HP-2 is permanently unmeasurable there.**
⚠️ **Every sequential closed-loop number carries EXTRAPOLATION *and* lateral infidelity.** Use
**pseudo-simulation** for the closed-loop reads — it is **0.00 % out-of-envelope**.

**Estimator:** paired episode-cluster bootstrap (`taniteval/ci.py`, B=2000), unit = **episode cluster**.
`overlapping_holdout_se` is refused. **A 40-episode "not separated" is UNPOWERED, not refuted** — use
the 600-episode deployment and state the power.

## 5. What is deliberately NOT changed

- **λ_plan stays 1.0** — it is a **gradient scale, not a loss weight**, and `lambda_plan == 1` is a
  documented **no-op**. It was never a lever.
- **The anchor vocabulary** — bitwise identical to real human windows. **Clamp the refinement instead.**
- **`closedloop.py`'s 2.7 → 2.9** — it would move every future closed-loop number and break
  comparability with published suites. **PI call, documented at the constant.**
- **Parity** — `physicalai-train-e438721ae894`, skip-hash `f09e44db`. Anything that re-selects episodes
  is refused.

## 6. Status

**NOT LAUNCHABLE YET.** Five gating items are in flight (§3). When they land, this document becomes a
launch card by adding the measured bar values — **and the PI approves the spend, not an agent.**
