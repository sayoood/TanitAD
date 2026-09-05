# SPEC (PRE-REGISTERED, banked before any arm ran) — A FEASIBILITY-AWARE DECODE for refcv3

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-feasible-decode/SPEC.md`
Architecture & Inference FlyWheel · 2026-09-05 · dev-box, **ZERO GPU** ·
⛔ `tanitad-refcv3` (refcv4b live) and Thor never touched.
Executes `Project Steering/Decisions/2026-09-05-mm-decisions.md` **§M23.5** — *"the real work
item is a FEASIBILITY-AWARE DECODE"* — and the scoping in
`…/2026-09-05-kinematic-gate/SUCCESSOR_FEASIBILITY_AWARE_DECODE.md`.

Evidence classes: **MEASURED** (ours + artifact path) · **INHERITED** (another WP, not re-run) ·
**HYPOTHESIS**. Tiers: **T0** = readout on an emitted fan, never a driving claim ·
**T1** = self-action OPEN loop (PI ruling 2026-09-02).

⛔ **Nothing below may be edited after the first arm runs.** Both outcomes are committed here.

---

## 0. The two things this package changes, and the one it does not

| | |
|---|---|
| **P1** | `progress` — the ONE reward term that ranks envelope-violating candidates HIGHER (ρ with `peak_g` **+0.2900**, `ttc_below` **+0.4697**, `contact` **+0.3286**; INHERITED from `…/veto-only-fan-safety/raw/reward_envelope_rank.json`). |
| **P2** | the **decode**: make an envelope-violating path *unrepresentable*, not merely penalised. |
| ⛔ **not this** | the top-2 **kinematic gate**. That is a sibling stream (`…/2026-09-05-kinematic-gate/`) and this package must not duplicate or pre-empt it. Its own MEASURED result — a selection rule closes **0.00 %** of the fan-level 8.56× (`gate_share_of_gap.json`, a STRUCTURAL ZERO) — is the argument for P2, and is cited as INHERITED. |

---

## 1. ⭐ THE STRUCTURAL FACT THAT DICTATES P1'S SHAPE (stated before measuring)

The published ρ is a **per-window Spearman across the 128 candidates of one fan**.
`progress = along / ref` where `ref = max(v0·H, min_ref)` is **constant within a window**.
Division by a positive per-window constant is a **monotone** transform, so

> **ρ(progress, peak_g) ≡ ρ(along, peak_g), exactly.**

⇒ **Any "fix" that is a per-window monotone function of `along` alone changes ρ by exactly
zero** (up to ties). Re-normalising, re-scaling, changing `min_ref`, changing the reference to a
feasible-speed reference — **all of them are null operations on this metric.** A fix must
therefore change *what progress is a function of*.

⚠️ This is stated in advance because it is the trap this P1 exists to avoid: three of the
obvious "reshape progress" proposals are provably no-ops, and each would have produced a
confident table showing ρ unchanged and been read as "the fix did not work".

**HYPOTHESIS (H-PROG-SAT-1), tested in P1:** the additive `feasibility` weight is a 270×-worse
lever *because the term is SATURATED*: `feasibility = exp(-ex)·motion_gate` with
`ex = (|a|/4 − 1)+ + (|κ|/0.2 − 1)+`, and on a fan at 4.11 g the exceedance is large enough that
`exp(-ex) ≈ 0` for most candidates, leaving no ordering to amplify. **Committed prediction:** the
inter-quartile range of `feasibility` over a window's own 128 candidates is **< 0.05** for the
median window. If it is NOT, this hypothesis is refuted and reported as refuted.

## 2. P1 — the fix, its committed criterion, and its controls

### 2.1 The lever

`progress_v2(traj, ctx) = min(along / ref, 1.0) · w(ex)`, floored at `-1`, where

* `min(·, 1.0)` — **saturating at "kept the current speed"**. Exceeding `v0` is no longer
  rewarded; the term rewards *not falling behind*, not *accelerating*.
* `w(ex) = 1 / (1 + ex/ex_s)` — a **multiplicative, NON-saturating** envelope discount on the
  same exceedance `ex` the `feasibility` component uses. `ex_s` is calibrated to the observed
  exceedance distribution (§2.3), **not** chosen for the answer.

⛔ **Why this is not "raise the feasibility weight" again.** That lever is ADDITIVE and competes
with `progress` inside a weighted sum, where it can be out-voted; MEASURED, ×4 moves ρ(envelope)
by 0.001. This one is MULTIPLICATIVE *inside the offending term*, so it removes the term's own
pathology rather than trying to outvote it — and the arithmetic already says that is the 270×
better surface.

⚠️ **The honest cost, named in advance:** the earlier per-window-reference decision (2026-08-29)
moved AWAY from saturation because saturation compresses ranking among the fastest candidates.
This SPEC deliberately re-introduces it at 1.0, because the same measurement now says ranking
among the fastest candidates is what *rewards violation*. The two decisions optimise different
things and the trade is stated, not hidden.

### 2.2 Committed criteria — reported as written whatever they read

| id | criterion | reading |
|---|---|---|
| **P1-C1** (the brief's) | **ρ(progress_v2, `peak_g`) < +0.05**, per-window Spearman, episode-cluster bootstrap over the 121 episodes, `n_boot = 4000`, `seed = 11` — the SAME estimator and seed as the number it replaces | PASS / FAIL |
| **P1-C2** (progress not destroyed) | on **straight** windows (median \|lateral\| of the fan's 2 s endpoint < 1 m), the mean along-track displacement of the **argmax-`progress_v2`** candidate is **≥ 0.95 ×** that of the argmax-`progress_v1` candidate **and ≥ the human's own** along-track displacement on the same windows | PASS / FAIL |
| **P1-C3** (reported, not gating) | ρ(progress_v2, `ttc_below`) and ρ(progress_v2, `contact`), and ρ of the **composed DEFAULT reward with progress_v2 substituted** against `envelope` / `peak_g` | reported |

⛔ **BOTH must pass.** A term that clears C1 by refusing to move is a FAILURE, and C2 is the
instrument that says so. If C1 passes and C2 fails, the verdict is FAIL and the successor is a
different shape of `w`, executed in the same run.

### 2.3 Calibration, and the controls that must read known values

* `ex_s` := the **median non-zero exceedance** over the banked fan (a distributional fact of the
  object, computed once, printed in the artifact, and **not** swept for the answer). Committed
  before the ρ is read.
* **C-SELF** — ρ(progress_v2, progress_v2) must read **+1.0000 exactly**.
* **C-CONST** — a constant term must read **UNDEFINED** on every window (Spearman of a constant
  is undefined), not 0.
* **C-RAND** — a uniform-random score must read the probe's own bias floor; INHERITED reference
  **−0.0138**.
* ⭐ **C-OBJECT (RETRACTION #30)** — the artifact must NAME the tensor it ranked and prove it is
  the emitted fan: assert `fan2.shape == (240, 128, 5, 2)`, print the npz **md5**, and reproduce
  the independently banked `emitted.peak_g` **4.1131** and `envelope` **0.8877** from
  `bank_vs_fan_feasibility.json` to 1e-4 / 1e-6. ⛔ *A control that checks only the arithmetic and
  not the object is exactly how #30 passed while interpolating a tensor that exists nowhere in
  the decode.*

---

## 3. P2 — the feasibility-aware decode

### 3.1 The mechanism: a CONTROL-SPACE PROJECTION, exact in the scorer's own geometry

`score_paths` differentiates the **5-point, 0.5 s** prefix with `rewards.kinematics`. The
projection therefore inverts *that exact map*, clamps, and re-integrates:

```
speed[k]  = |p[k+1]-p[k]| / dt        heading[k] = atan2(p[k+1]-p[k])
accel[k]  = (speed[k+1]-speed[k])/dt  yaw[k]     = wrap(heading[k+1]-heading[k])/dt
v_mid[k]  = max(speed[k], 0.5)        kappa[k]   = yaw[k]/v_mid[k]   lat[k] = v_mid[k]*yaw[k]
```
Forward, sequentially in k (so `v_mid[k]` is already determined when step k is clamped):

1. clamp `accel[k]` to `[-a_max, a_max]`, `a_max = 4.0`;
2. clamp `lat[k]` to `±v_mid[k]^2 · kappa_max`, `kappa_max = 0.2`;
3. **radially** project `(accel[k], lat[k])` onto the friction disc of radius `R = mu·g`;
4. `speed[k+1] = speed[k] + accel[k]·dt`; `heading[k+1] = heading[k] + (lat[k]/v_mid[k])·dt`;
5. `p[k+1] = p[k] + speed[k]·dt·(cos heading[k], sin heading[k])`.

⭐ **Because steps 1–3 are the exact inverse-then-forward of the scorer's own finite differences,
`envelope` and `kamm_over` are ZERO BY CONSTRUCTION — an identity, not an estimate.** No seed and
no bootstrap can change a structural zero (`H-ECHO-4` precedent), and the artifact asserts it
rather than reporting it.

**Arms:** `mu ∈ {off, 2.0, 1.5, 1.0, 0.7, 0.5, 0.4, 0.3}` with the box always on; `off` = the
emitted path unchanged. Headline arm = **`mu = 0.7`** (`MU_KAMM`, the scorer's own friction
circle). A second variant `proj+entry` additionally clamps `speed[0]` to `v0 ± a_max·dt`, which
`envelope` cannot see but a real vehicle can; reported beside, never merged.

### 3.2 ⛔ THE SCOPE ERROR THIS SPEC REFUSES TO REPEAT

The sibling's λ-frontier (`displacement_frontier.json`, 400 w) sweeps
`path(λ) = bank + λ(fan − bank)` — an **isotropic shrink of the whole displacement**. Its
committed decision rule reads on "is there a knee". ⚠️ **That instrument answers a question about
the SHRINK family and cannot answer one about the PROJECTION family**: a shrink removes
displacement everywhere, a projection removes only the infeasible *component* of the control
profile and keeps the rest. Quoting the λ verdict against a projection would be the
`df`/`free`/`step_s` scope error in a new costume.

⇒ **The comparison is run on the SAME npz and the SAME windows**
(`kingate_bank_drawA_v1_selscore.npz`, md5 `9bda7715a6571186935243b242c2ab74`, 400 w × 128 cand)
so that the two frontiers are read at **matched `fan_peak_g`**, which is the only fair axis.

### 3.3 Committed criteria for P2

| id | criterion | reading |
|---|---|---|
| **P2-C1** | at `mu = 0.7`, fan `envelope` **= 0.0000 exactly** and `kamm_over` **= 0.0000 exactly** (structural zeros, asserted) | PASS / FAIL |
| **P2-C2** | at **matched `fan_peak_g`**, the projection's `oracle_ade_m` penalty is **< 0.5 ×** the λ-shrink's on the same windows | PASS / FAIL |
| **P2-C3** | the fraction of the **8.56×** `peak_g` gap closed at `mu = 0.7`, reported as a number with the residual named | reported |
| **P2-C4** | `off_reach` must **not** rise by more than **+0.05** absolute at the headline `mu` — ⛔ the failure mode the successor doc named in advance (trading one failure for the other and calling it a win) | PASS / FAIL |

⛔ **If P2-C2 FAILS**, the verdict is that the friction cost is intrinsic to this vocabulary at
this displacement, and the successor executed **in the same run** is the third row of the
successor doc's own decision table: a **v0-conditioned anchor vocabulary**, scoped with the
measured numbers rather than proposed.

### 3.4 Controls

* **C-OFF (deliberate regression / disabled lever)** — `mu = off` must reproduce the input paths
  **bit-identically** (`max |Δ| == 0.0` exactly). ⛔ This is the arm that would expose a
  projection that silently rewrites a path it was asked to leave alone.
* **C-ROUNDTRIP (OBJECT, RET #30)** — projecting a path that is **already feasible** must return
  it to `< 1e-5` m, and the artifact must print WHICH tensor it round-tripped (shape + md5 of the
  source npz), not merely that the arithmetic closed.
* **C-FLOOR** — the `ha0` constant-velocity floor must project to itself and read
  `envelope = kamm_over = 0` before and after.
* **C-KNOWN** — the frozen anchor bank's own rates must reproduce
  `bank_vs_fan_feasibility.json`'s `bank` block (a DIFFERENT script, a DIFFERENT window draw).

---

## 4. P3 — the proof at T1, four families, and the two floors

**Instrument:** `taniteval/tools/paired_openloop.py`, paired **episode-cluster bootstrap**
(`taniteval/ci.py`), `n_boot = 2000`, `seed = 0`, floor arm `ha0`.
⛔ **`overlapping_holdout_se` is never used.**
**Object:** the banked base dump `refcv3_40284_dump` (141 clips / 4823 windows), with the driven
arm `os` replaced by its projection. All four families — LONGITUDINAL, LATERAL, TACTICAL,
STRATEGIC — reported **per family, never pooled**; ADE alone is INCOMPLETE.

### ⛔ The floors, and an honest statement of which one applies

The brief binds two floors: a **seed replicate** (`H-ESTIM-SEED-1`) and **`ctrl0` with `lr = 0`**
as the only admissible zero-lever floor. Both exist because **AdamW normalises by the gradient's
own scale**, so a control that zeroes a *loss* still moves every tensor.

⭐ **This lever takes ZERO gradient steps.** There is no optimiser, no update, and no seed in the
lever: the arms differ by a deterministic geometric function of the emitted path. The floors are
therefore answered by a **strictly stronger** control, and it is MEASURED rather than asserted:

* **F1 — bit-identity of the disabled lever.** `mu = off` on the same dump must produce
  `max |Δ| == 0.0` against base on **every** window of **every** family ⇒ the run-to-run noise
  floor of this rig is **exactly zero**, which no `ctrl0` can improve on.
* **F2 — inference determinism.** The projection is re-run under a different RNG seed and must
  return bit-identical output.
* ⚠️ **Scope, stated plainly:** F1/F2 retire the *training-variance* question for THIS arm only.
  Any future arm that **fine-tunes** the decoder under the projection re-acquires the
  `H-ESTIM-SEED-1` obligation in full, and this SPEC does not discharge it in advance.

### Committed criterion

| id | criterion |
|---|---|
| **P3-C1** | the driven path's `envelope` and `kamm_over` rates over all 4823 windows read **0.0000 exactly** |
| **P3-C2** | `ade_0_2s` regression vs base is **≤ +0.0362 m** — the V2-faithful RL arm's MEASURED failure — and its paired CI is reported whether or not it separates |
| **P3-C3** | all four families reported; a family that refuses is diagnosed as **refusal vs `defect: true`** and named as a work item, never silently dropped |

## 5. What would make this package VOID

1. C-OBJECT or C-ROUNDTRIP failing ⇒ the projection is not operating on the object claimed.
2. C-OFF not bit-identical ⇒ the disabled lever is not disabled; every delta is uninterpretable.
3. The `n` of any reported family falling below **10 episodes** ⇒ the episode-cluster bootstrap
   cannot resample enough clusters and the interval is not decision-grade.

## 6. Register rows this package will write (in the same turn)

`D-PROG-RANK-FIX-1` (P1's verdict), `H-PROG-SAT-1` (the saturation hypothesis),
`D-FEASDEC-1` (the projection's structural zero), `D-FEASDEC-FRONTIER-1` (projection vs shrink at
matched `peak_g`), `H-FEASDEC-VOCAB-1` (the vocabulary successor, only if P2-C2 fails).
