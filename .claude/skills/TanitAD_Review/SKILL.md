---
name: TanitAD_Review
description: The configured TanitAD review — checks every quoted number against its artifact, evidence class and tier; audits estimator discipline; verifies the claims register was updated; flags vocabulary drift. Use before shipping any report, result, or PR.
---

Review work against the programme's own rules. The goal is **maximizing durable
assets**, not finding mistakes — so a finding must name the rule it violates and
the cheapest fix, never just an objection.

**Read first:** `Project Steering/TANITAD_PROGRAMME.md`,
`GOALS_AND_CLAIMS.md`, `VOCABULARY.md`, `QUALITY_SYSTEM.md`, and the ROOT-CAUSE
CLASSES (not the whole text) of `RETRACTION_LOG.md`.

**Scope:** a work package dir, a diff/PR, or "programme".

## 1. Quotable-number audit (the two-key rule)

Every number in a report needs BOTH keys or it is inadmissible:
1. an **artifact path** (raw JSON/log under `raw/`), and
2. an **evidence class** (MEASURED / PUBLISHED / INHERITED / ESTIMATED /
   HYPOTHESIS) **+ a tier stamp** (T0 / T1 / T2).

- `MODEL_REGISTRY.md` and raw eval JSON are the ONLY quotable sources. A number
  copied from prose, a summary, or a changelog is a finding.
- ⛔ **T0 is never "driving performance"** (C131: `ade_0_2s` is
  `wm_fidelity_ade_2s`; open-loop 0.4271 → closed-loop 1.7318).
- Never quote a learning-curve exponent without its fit window, R² and n.
- Never quote an interval without its estimator. `overlapping_holdout_se` is
  deprecated and biases the POINT ESTIMATE, not just the interval.

## 2. Estimator discipline

- **Name the STATISTIC, not just the quantity.** σ vs σ² inverted a gate verdict
  by 141× (C132). "rank" is not a statistic; `participation_ratio (σ²)` is.
- Hyper-parameters (λ, PCA basis, thresholds) fit on the FIT split only — never
  on the scored split.
- Selection on a **point estimate** hides the real effect; select on the CI bound.
- **n and d printed.** `n ≪ d` is underpowered by construction.
- Thresholds calibrated on **REAL references with demonstrated task-relevance**,
  never on synthetic populations alone (`O6_RANK_FLOOR = 64` came from an α=2
  power-law that no real representation approaches).
- A criterion that **cannot RULE** at the configured settings must say so loudly
  — a populated-looking gate report that always returns INCONCLUSIVE is worse
  than no gate.

## 3. Controls

Every panel: a **constant-only control** reading the no-information value, a
**raw-input floor**, and — where a guard is claimed — evidence the guard **can
fail** (a deliberate-regression arm). No controls ⇒ the panel is unreadable, and
that is the finding.

## 4. Register & vocabulary

- Did every asserted/refuted claim update `GOALS_AND_CLAIMS.md` in the same turn?
  An un-registered claim is a finding.
- Do terms match `VOCABULARY.md`? Flag drift and deprecated terms.
- Is any work-package following the §3 schema? Missing SPEC.md with
  both-outcomes is a finding for anything that consumed compute.

## 5. False-positive taxonomy (check BEFORE reporting upward)

Match each candidate finding against the known root-cause classes; if it matches,
the class's specific control is required before the finding may be asserted:

| class | signature |
|---|---|
| scope error | a counter aggregating the wrong scope (`df` on pods, `free` in a container, `step_s` per-trainer, cgroup usage) |
| tuned-on-scored-data | λ/basis/threshold selected on the split being scored |
| statistic ambiguity | two normalisations sharing one name |
| echo test | an output that is a bijection of its own input |
| content-vs-existence | success asserted from exit code, file count, or a name |
| absence at one location | "X does not exist" from a single probe |

## 6. Output

Findings ranked most-severe first. Each: the rule violated, the evidence, the
cheapest fix. Then explicitly list **what is SOUND** — the review's purpose is to
certify assets, not only to find defects. If nothing survives verification, say
so plainly rather than manufacturing findings.
