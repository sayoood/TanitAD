# PREREG — P4 / P13 / P14, the three unowned refcv5 pieces

**Author:** P4/P13/P14 validation agent · **Date:** 2026-09-06 · **Status:** REGISTERED
**Governing plan:** `Project Steering/REFCV5_MISSING_PIECES_PLAN.md` §2 (the six-item validation contract)
**Authority:** PI directive 2026-09-06 — *"validate the improvements to be sure they will bring benefit …
use the WHOLE tactical and strategic vocab."*

⛔ **THIS FILE IS WRITTEN BEFORE THE NUMBERS EXIST.** Every bar below is committed with **both
outcomes named**. A number produced after this file may not move a bar; if a bar turns out to be the
wrong question, the correction is a NEW pre-registration, not an edit here.

⚠️ **Scope stamp that travels with every result in this package.** The rig is the **banked fan**
(P14, zero GPU) and **unit-level mutation** (P13, P4-plumbing). A tiny-rig or banked-fan PASS is
**entry to the composed arm, not a published result** (plan §2.1).

---

## 0. What each piece is, in one line, MEASURED

| piece | the defect, measured this session | artifact |
|---|---|---|
| **P4** | `STRATEGIC_GOAL_TOKENS_V7` (8) + `STRATEGIC_ACTION_TOKENS_V7` (7) are minted on every clip and **trained on by nothing** | `raw/p4_label_coverage.txt` |
| **P13** | `--sampler-train-t-max` is assigned to `core.decoder.sampler_train_t_max` and **read by no consumer**; `_sample()` starts at `cfg.sampler_infer_t = 8` in **training and inference alike** | `raw/p13_dead_flag.txt` |
| **P14** | the emitted fan is ranked by a score the sampler never saw; `sampler_ranks_the_fan = bool(self.sel.refined)` and `refc_v3_train.py` **has no `--sel-refined` / `--sel-score-emitted` flag at all** | `raw/p14_flag_absence.txt` |

---

## 1. P14 — the sampler must rank the fan it emitted

### 1.1 The bar (banked fan, ZERO GPU)

The banked fan carries, per window, the **whole emitted fan** and the **shipped ranking score**. That
makes the question answerable without a roll: *how much of the shipped selection's error is
recoverable by ranking the same fan better?* That quantity is the **CEILING on P14** — no wiring can
buy more than it, and if the ceiling is small the piece does not earn its place.

⭐ **PRIMARY BAR — P14-B1 (the ceiling).** Over the banked windows, with the **paired
episode-cluster bootstrap** over episodes:

* **PASS** if `ADE(shipped selection) − ADE(fan-best)` is **≥ 0.05 m** with a **95 % paired CI
  excluding zero**, i.e. there is a real, recoverable ranking loss worth wiring for.
* **FAIL** if the gap is **< 0.05 m** or its CI straddles zero — the fan is already being ranked
  about as well as it can be, and P14 is cosmetic.

⭐ **SECOND BAR — P14-B2 (the defect is RANKING, not FAN QUALITY).**
* **PASS** if the fan-best trajectory beats the shipped selection on **> 25 % of windows by > 2×**
  (the shape the plan quotes at 41.09 % for refcv5).
* **FAIL** otherwise.

⛔ **P14-B3 — the VACUITY GATE.** A ranking win bought by never manoeuvring is not a win. The
**manoeuvre rate** (share of windows whose selected trajectory has |lateral displacement| > 1 m at
the terminal slot) is reported **beside every P14 number**, for the shipped ranking and for every
re-ranking. **FAIL** if a re-ranking's manoeuvre rate falls below **50 % of the shipped rate**.

### 1.2 The deliberate-regression arm (P14-R)

⛔ **`rank_shuffled`** — rank the fan by a **permuted** score (the shipped scores, shuffled across
candidates within each window). A gate that cannot see this is blind.
* **The gate is VALID** only if `rank_shuffled` is measurably **WORSE than shipped** with a
  separated paired CI.
* **If `rank_shuffled` is NOT separated from shipped, the whole P14 measurement is VOID** and is
  reported as VOID, because the instrument cannot distinguish a ranking from noise.

### 1.3 Controls that must read a KNOWN value

| control | the value it MUST read | why |
|---|---|---|
| `rank_shipped` | **exactly** the dump's own `sel` indices, `n_mismatch = 0` | proves my re-ranking harness reproduces the shipped decision before it is allowed to change it |
| `rank_oracle` | the per-window **minimum** fan ADE, by construction | the ceiling; any re-ranking above it is a harness bug |
| `cv` (constant-velocity) | the **raw-input floor** — a plan that uses no learned ranking at all | a learned ranking that does not beat CV has added nothing |
| `rank_constant` (always anchor 0) | the **no-information** value | a constant selector; the no-skill reference |
| `n`, `d` | printed on every table | `n` = windows, `d` = fan width |

### 1.4 Four metric families (never pooled)

| family | metric on the banked fan |
|---|---|
| **ADE** (accompanying, not the result) | mean displacement error at the banked slots |
| **LONGITUDINAL** | along-track error + terminal speed error vs `v_target` where `vt_valid` |
| **LATERAL** | ⛔ **curvature MAE, with the STRAIGHT-LINE floor beside it** + cross-track |
| **TACTICAL** | selected-vs-fan-best anchor agreement, and the manoeuvre-class rate (the vacuity gate) |
| **STRATEGIC** | reported **N/A with its reason and its n** — the banked fan carries no route/goal label |

### 1.5 The variance the interval answers

⛔ The paired episode-cluster bootstrap over the banked episodes answers **"would another draw of
EPISODES say this?"** — and **nothing else**. It is structurally blind to training variance
(replicate false-positive rate **6/42 = 14.3 %**) and, on a sampling planner, to inference variance
(refav1 seed floor ≈ 0.30 m ADE). **Both arms of every P14 contrast are the SAME checkpoint and the
SAME fan**, so training and inference variance are **held at exactly zero by construction** — this
is the one contrast where the episode bootstrap is the whole question. Stated so it is not
generalised to a contrast where it is not.

---

## 2. P13 — DD's `t ~ U[0, t_max)` training draw

### 2.1 The bar (unit-level; the tiny-rig arm is the follow-on)

⭐ **P13-B1 — the flag must acquire a consumer, provably.**
* **PASS** if, with the draw wired, the timesteps the training pass actually consumes are
  **distributed over `[0, t_max)`** — asserted by a **mutation test** that (a) reads the realised
  timesteps, (b) shows `t_max = 50` and `t_max = 1` produce **different** distributions, and (c)
  shows the current code produces the **constant 8** for both.
* **FAIL** if the realised timesteps are constant under a changed `t_max` — that is the dead flag
  surviving the fix.

⭐ **P13-B2 — the noising must be DD's, not merely random.**
* **PASS** if `sigma(t)` from the wired draw matches `DDIMSchedule.sqrt_one_minus_abar(t)` to
  float32 epsilon at every drawn `t`, and `t = 8` reproduces the published **0.0316**.
* **FAIL** on any mismatch.

⛔ **P13-B3 — BENEFIT IS NOT ASSUMED.** Paper-faithfulness is **not** the bar. The training-time
draw enters the composed arm **only** if a tiny-rig arm (`--sampler-train-t-max 50` vs the
current constant `t = 8`) improves the **u0 control-space loss** and does **not** degrade curvature
MAE, both with separated paired CIs **and a replicate arm**. ⛔ **Until that arm runs, P13 is
reported UNVALIDATED-FOR-BENEFIT even if B1 and B2 pass** — B1/B2 establish only that the mechanism
exists and is DD's.

### 2.2 The deliberate-regression arm (P13-R)

⛔ **`t_max = 1`** — a draw that can only produce `t = 0`, i.e. **no noise at all**. The x0 target
is then trivially the input and the loss must **collapse toward zero** while the sampler learns
nothing. **The gate is VALID only if this arm is distinguishable from `t_max = 50`.** If a
zero-noise draw looks the same as DD's, the instrument is blind.

### 2.3 Controls

| control | MUST read |
|---|---|
| `sigma(0)` | `sqrt(1 - abar_0)` = the schedule's first entry, **not** 0.0 |
| `sigma(8)` | **0.0316** (the published truncation sigma) |
| `t_max = 50` mean drawn t | **≈ 24.5** (= (50−1)/2) within Monte-Carlo error, `n` printed |
| current code | **constant 8**, variance **exactly 0.0** |

---

## 3. P4 — the 15-token strategic vocabulary

### 3.1 What is being validated, and what is NOT

⛔ **P4's validation is in two parts and they must not be confused.** Part A is a **corpus fact** —
what the labels can support — and it is measurable at zero GPU. Part B is a **capability claim** and
needs a tiny-rig arm. **Part A gates Part B**: a class with no support cannot be validated by any
arm, and reporting a head's pooled accuracy over an unsupportable vocabulary is the vacuity failure
this contract exists to stop.

### 3.2 Part A bars (corpus, zero GPU)

⭐ **P4-A1 — per-class support.** Report **per-class recall support**, never pooled accuracy, with
the **majority-class control at its known value**.
* **PASS** (the vocabulary is trainable as written) if **≥ 6 of 8** goal classes and **≥ 5 of 7**
  action classes carry **≥ 1 % support**.
* **FAIL** otherwise — and on FAIL the deliverable is the **restricted vocabulary** that IS
  trainable plus the named source of the missing classes, not a head over dead classes.

⭐ **P4-A2 — the supervisable fraction.** Report the share of the usable horizon carrying strategic
GT. **No bar** — this is a measured ceiling that travels with every P4 number.

⭐ **P4-A3 — label quality.** Report the **grounded share** for the strategic tokens specifically.
* **PASS** if the strategic tokens' provenance is **geometry** (i.e. the 175-grounded / 604-disputed
  traffic-light problem does **not** apply to them).
* **FAIL** if any strategic token is `vlm-cot` / `disputed` — such a class must be filtered out
  before it is supervised (plan §5: *"never wire a label without a quality filter"*).

⛔ **P4-A4 — THE ECHO GATE (the one that can void the whole head).** The flagship route head was an
**exact bijection of the nav we feed it (369/369) and scored 1.0000** — an echo of its own input read
as skill. The strategic goal token is derived from ego-future geometry and the nav command is a
**legitimate inference input**, so the two can coincide.
* **VOID** — the head may not be fed `nav_command` — if `nav → g_str` determinism is **≥ 0.999**.
* **CONDITIONAL** — a nav-fed arm requires a **nav-ablated control** before any number is quotable —
  if mutual information is **> 50 %** of `H(token)`.
* **CLEAR** — nav may be an input without a special control — otherwise.

### 3.3 Part B bar (tiny rig) — committed now, run when compute frees

* **PASS** if the vision+`v0` strategic head's **macro per-class recall over the SUPPORTED classes**
  exceeds the **majority-class control's macro recall (exactly `1/K`)** with a separated CI **and a
  replicate arm**, and the composed arm's curvature MAE does not degrade.
* **FAIL** otherwise. ⛔ Pooled accuracy is **not** admissible: a constant `FOLLOW_ROUTE` predictor
  scores high pooled accuracy while its macro recall is exactly `1/K`.

### 3.4 The deliberate-regression arm (P4-R)

⛔ **`labels_shuffled`** — the same head trained against **permuted** strategic labels. It must land
at the **no-information macro recall (`1/K`)**. If a head trained on shuffled labels scores above
chance, the evaluation is leaking and every P4 number is VOID.

### 3.5 Controls

| control | MUST read |
|---|---|
| majority-class constant | pooled acc = the majority share; **macro recall exactly `1/K`** |
| uniform random | macro recall `1/K` |
| parser control (`a_tac.lat` present) | **> 0** on the same rows — a zero strategic count must not be a claim about my parser |
| vocabulary control | all 15 tokens present in `vocab_v7.py` source |
| `n`, `d` | printed on every table |

---

## 4. Composition rule

⛔ A piece enters `refcv5-cap-b1-v72-40k` **only** on a PASS of its bars above. A FAIL is banked with
its **next lever named and run** (RULE ZERO) — a FAIL does not end the piece. A piece the PI wants
carried anyway is stamped **UNVALIDATED — carried on the PI's authority**, never silently included.
