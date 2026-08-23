# E1d RESULT — OUTCOME B, and the α-frontier says something stronger than "no good α"

**MEASURED 2026-07-28**, pod3, wall 3838.5 s. Artifact: `pod3:/workspace/e1c/e1d_alpha_result.json`.
Estimator: **paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000) over the **44 held-out
episodes** (43 clusters at K=185). `overlapping_holdout_se` used nowhere. Pre-registration:
`PRE_REGISTRATION_E1D.md`, committed **before** any number existed.

## 0. The control passed — everything below is quotable

α=1.00 (step code 100) is bit-identical to `delta_step04000.pt` and **reproduces E1c frontier row
4000 exactly**: `dep_overall −0.4274`, `dep_junction −0.4270`, `ade2s +0.1947 [+0.1415, +0.2522]`.
Same evaluator, same `evaluate_point` / `render_verdict`, same strata, same multiplicity correction.

## 1. Verdict: BOUND (pre-registered OUTCOME B)

0 of 9 α satisfy all six conditions. P1∧P2 fire at 2/9 (α = 0.85, 1.00); **the guardrails held at 0/9.**

| α | `dep_overall` Δ [lo, hi] | sep | `dep_junction` Δ [lo, hi] | sep | open-loop ADE@2s Δ [lo, hi] |
|---|---|---|---|---|---|
| 0.10 | **+0.0386** [−0.0387, +0.1154] | – | −0.0874 [−0.2343, +0.0000] | – | +0.0116 [+0.004, +0.019] |
| 0.20 | **+0.1107** [+0.0382, +0.1840] | 🟥 **worse** | −0.1784 [−0.3775, −0.0072] | ✅ | +0.0308 [+0.018, +0.044] |
| 0.30 | **+0.1387** [+0.0725, +0.2064] | 🟥 **worse** | −0.1072 [−0.3126, +0.0072] | – | +0.0538 [+0.036, +0.073] |
| 0.40 | **+0.1492** [+0.0821, +0.2166] | 🟥 **worse** | −0.1000 [−0.2991, +0.0090] | – | +0.0786 [+0.059, +0.100] |
| 0.50 | **+0.1199** [+0.0587, +0.1849] | 🟥 **worse** | −0.0919 [−0.2685, +0.0099] | – | +0.0939 [+0.071, +0.120] |
| 0.60 | **+0.0759** [+0.0063, +0.1436] | 🟥 **worse** | −0.1739 [−0.3892, −0.0108] | ✅ | +0.1105 [+0.083, +0.142] |
| 0.70 | +0.0093 [−0.0685, +0.0881] | – | −0.2234 [−0.4865, −0.0252] | ✅ | +0.1153 [+0.081, +0.152] |
| 0.85 | **−0.1648** [−0.2576, −0.0745] | ✅ better | −0.3090 [−0.5973, −0.0505] | ✅ | +0.1295 [+0.090, +0.175] |
| 1.00 | **−0.4274** [−0.5161, −0.3378] | ✅ better | −0.4270 [−0.6838, −0.1648] | ✅ | +0.1947 [+0.141, +0.252] |

Base: `dep_overall` 0.5877 [0.5107, 0.6622], `dep_junction` 0.8414, open-loop ADE@2s 0.4747.

## 2. ⭐ THE MECHANISM: THE PATH CROSSES A REGION WORSE THAN **BOTH** ENDPOINTS

`dep_overall` is **SEPARATED-WORSE at five consecutive interior points** (α = 0.20, 0.30, 0.40,
0.50, 0.60), peaking at α = 0.40. This is not noise — each CI excludes zero on the paired
episode-cluster bootstrap.

⇒ **Linear weight-space interpolation between REF-C base and the CL-SFT is not monotone; it passes
through a barrier.** The two solutions are **NOT linearly mode-connected** for corridor departure.

⚠️ **That is precisely the PRECONDITION WiSE-FT relies on** — Wortsman et al.'s interpolation
dominates early stopping *when the fine-tune stays in the base's basin*. **We imported the remedy
without checking its precondition, and the precondition is violated here.** The published result is
not wrong; it does not apply. *(Logged as a class — see `RETRACTION_LOG.md` C52.)*

## 3. ⭐ AND THE PRIMARY'S TWO COMPONENTS BEHAVE COMPLETELY DIFFERENTLY

This is the part that changes the next design, and it is invisible in E1c's frontier because there
α and training-time are confounded.

- **`dep_junction` is better (negative) at EVERY α, and never separated-worse anywhere.** It is
  separated-better already at **α = 0.20**, where the open-loop cost is only **+0.0308** — **6.5 %**
  of base ADE, versus **+41 %** at the endpoint.
- **`dep_overall` is separated-WORSE through the middle** and only turns good past α ≈ 0.72.
- **Open-loop ADE is paid immediately and near-linearly**: separated-higher at **every** α,
  including α = 0.10 (+0.0116 [+0.004, +0.019]).

⇒ **The CL-SFT buys JUNCTION recovery cheaply and monotonically, and pays for OVERALL-CORRIDOR
recovery expensively and non-monotonically.** Chasing overall corridor departure is what drags the
model out of the base's basin; the junction benefit does not require leaving it.

## 4. What this closes, and what it opens

**CLOSED — per the pre-registration, and not reopened:**
- "Train longer" — ruled out by E1c's plateau (8 points inside ±0.02, no trend).
- "Pick a better point on the base→FT segment" — ruled out here, and by a *stronger* argument than
  "we didn't find one": the useful region exists only at the endpoint where the open-loop cost is
  maximal, and the interior is separated-worse.
- **A finer α grid is NOT run.** It was pre-committed as inadmissible, and the barrier makes it
  pointless.

**OPEN — the next D-A step, now with a data-driven target:** the lever is the **objective**, as E2a
already indicated. The specific design E1d motivates is to **target junction recovery explicitly and
constrain the open-loop head toward base on non-failure windows** (a basin-preserving penalty), so
the fine-tune is not pulled across the barrier in the first place. This is a *proposal derived from
measurement*, not yet an experiment; it will be pre-registered with both outcomes before any GPU.

## 5. Honest bounds

- **n = 43 episode clusters at K=185**, 6 junction windows. The junction CIs are wide
  (e.g. α=0.20: [−0.3775, −0.0072]); "separated" there rests on a narrow margin and should not be
  quoted as a precise effect size.
- α is a **deployment** knob, not a training result; a winning α would still owe a from-scratch
  confirmation before entering any headline. None won, so the point is moot.
- The K=20 block is reported and **NON-DECIDING** by design.
