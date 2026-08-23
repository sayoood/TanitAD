---
name: TanitAD_ValidateAIDesign
description: Validate any model/design change on the v7-tiny ladder before it earns compute — pre-registered spec, deliberate-regression arm, rank AND decodability gates, controls that must read known values. Use for any architecture, loss, or training-knob change.
---

Validate a design change so it cannot produce a false positive. This skill encodes
the corrections of 2026-08-22, where four estimator bugs, one λ confound and one
statistic inversion each produced a confident wrong number.

**Read first:** `Project Steering/TANITAD_PROGRAMME.md` (§3 schema, §6 quality),
`Project Steering/GOALS_AND_CLAIMS.md` (the hypothesis must have an ID),
`Project Steering/VOCABULARY.md`.

## 1. SPEC before compute (refuse to launch without it)

Create `<field>/<YYYY-MM-DD>-<slug>/SPEC.md` carrying:

```yaml
hypothesis: H-RANK-7          # MUST already exist in GOALS_AND_CLAIMS.md
one_variable: o5_k            # the ONLY difference between arms
held_constant: [lambda, seed, corpus, steps, batch, window]
success: "<criterion with a CI, committed in advance>"
failure: "<the outcome that refutes it, committed in advance>"
controls: [constant_only, raw_input_floor, deliberate_regression]
splits: {fit: "...", val: "carved from FIT only", test: "scored, never tuned on"}
```

⛔ **Preflight refuses the run if**: arms differ in more than `one_variable`;
a control is missing; the hypothesis ID is unknown; any hyper-parameter would be
selected on the scored split.

⚠️ **The one-variable check is not bureaucratic.** MEASURED 2026-08-22: a row-bank
arm changed `n` AND silently multiplied the effective λ by ~n/24 (the Epps-Pulley
statistic is not n-normalised). Two sweeps were invalidated. Diff the actual
launch commands, not the intent.

## 2. Run on the tiny rig first

`v7-tiny` = v6's REAL trainer at ~19 M params, parity corpus, **~17 min/arm on
Thor**, ~29 min on the dev box. Never validate a design on a full-scale run.

Include a **deliberate-regression arm** that re-introduces the defect the gate is
supposed to catch. If the gate does not FAIL that arm, a PASS on the fixed arm
means nothing.

## 3. Gates IN ORDER — an arm earns the next only by clearing the previous

| gate | criterion | reference |
|---|---|---|
| **G-RANK** | `participation_ratio` (σ², NOT effective_rank) ≥ **8.56** | frozen DINOv3, MEASURED on our frames |
| **G-DECODE** | ego probe (speed/yaw/yaw-rate/d_ego) beats BOTH the raw-pixel floor AND the constant control; detection AP > `prior` and > `pixel`, paired | — |
| **G-DRIVE** | T1, four metric families, paired episode-cluster bootstrap | registry rows |

⛔ **Rank is NECESSARY, NOT SUFFICIENT** (C131): flagship v1-era had the highest
rank measured and no environment interpretation — it was conditioned on future GT
points and fails closed-loop. **Never pass an arm on rank alone.**

⛔ **Use participation (σ²), never effective_rank (σ)** (C132): the two disagree by
up to 141×, and effective_rank PASSES a representation with 55 % of its energy in
one direction. Collapse is an ENERGY question.

⚠️ Quote **val-side** participation for representation claims. The gate's pooled
reading comes from the O4-weighted TRAIN stream and runs high (~5.5 vs ~3.4 on the
same model, H-RANK-9); it is cross-arm comparable but is not a val-side measure.

## 4. Every panel carries its controls

- **constant-only control** → must read the no-information value EXACTLY
- **raw-input floor** (pixels) → a learned representation below it added nothing
- **printed n and d** → `n ≪ d` is underpowered BY CONSTRUCTION, not a negative
- fit every hyper-parameter (λ, PCA basis) on the FIT split only

⚠️ A negative from a LINEAR probe is not a negative about learnability. State the
function class, or use a nonlinear probe with a time-shuffled control.

## 5. Verify by CONTENT, never by exit code

A run "succeeded" only if its artifact exists and is non-trivial. MEASURED: chains
printed `TRAIN-0` for arms with 0-byte logs and no checkpoint; a rewrite reported
"rewritten 0" because its search matched nothing. Assert on bytes.

## 6. Close the loop (same turn)

Update `GOALS_AND_CLAIMS.md` with the hypothesis status + evidence path; write
`RESULT.md` with evidence class and tier stamps; bank raw JSON under `raw/`;
add a registry row if a model version resulted; if the procedure is reusable,
draft a skill.
