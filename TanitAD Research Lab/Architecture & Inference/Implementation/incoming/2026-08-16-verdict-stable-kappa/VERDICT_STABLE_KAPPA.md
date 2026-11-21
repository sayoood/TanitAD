# `verdict_stable` tested a threshold the programme does not publish — FIXED

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **GPU used:** none (0 GPU; banked
artifacts + CPU test suites only)

**One line:** `hierarchy._gate_sensitivity` certified verdict stability against a private
**κ ≥ 0.2**, while the verdict word the programme actually publishes uses the **0.1 / 0.4** ladder.
The two disagreed on the programme's only gate-swept panel: the field said `verdict_stable = true`
while the published verdict **does** flip across the sweep. Both call sites now read one imported
constant, and three tests pin it.

---

## 0. Evidence classes

| claim | class |
|---|---|
| the code's two thresholds and their `file:line` | **MEASURED** (source, quoted below) |
| the published ladder is 0.1 / 0.4 | **MEASURED** (four code sites + registry/paper quotes, below) |
| the 880-window sweep flips SUBSTANTIAL → WEAK | **MEASURED** — recomputed by me from `hier_v1arch_gateswept.json.xz` |
| the B-block flip enumeration (14 panels, 2 flips) | **MEASURED** — scan over every banked `hier*.json[.xz]` |
| suite green | **MEASURED** — this box, today, `PYTHONUTF8=1`, `OMP_NUM_THREADS=6` |

⚠️ **No interval is quoted for any κ in this document, and that is not an omission.** Cohen's κ
carries **no CI in this instrument by construction** — `hierarchy.agree()` says so in-line
(*"kappa is the honest read on a straight-dominated corpus and carries no interval by
construction"*). The agreement RATE beside it does carry an episode-cluster-bootstrap interval;
κ does not. Per the estimator rule, I state the absence rather than manufacture one.

⚠️ **Path correction to the brief.** The module is `taniteval/taniteval/hierarchy.py` and its tests
are `taniteval/tests/`, **not** `stack/taniteval/`. `taniteval/` is a top-level sibling of `stack/`.
The two suites **cannot be collected together** (duplicate basename `test_corridor.py`); they must
be run as separate pytest invocations.

---

## 1. The defect, from source

Two kappa thresholds existed in `hierarchy.py`, both bare `0.2`, both deciding a verdict:

| site | code (before) | what it decides |
|---|---|---|
| `taniteval/taniteval/hierarchy.py:262` | `verdicts = {bool(k >= 0.2) for k in ks}` | **`verdict_stable`** |
| `taniteval/taniteval/hierarchy.py:1140` | `coherent = bool(mt["kappa"] is not None and mt["kappa"] >= 0.2)` | `thesis_read.B_layers_mutually_consistent.verdict` |

and the panel's own note (`:278-282`) asserted the rule as if it were the programme's:

> `"verdict = kappa >= 0.2 (the coherence threshold this panel uses)"`

*(A sweep of every numeric literal in the file confirms these are the only two **kappa**
thresholds. The other `0.2`s — `:433`, `:1034` — are the legacy holdout `val_frac`, and `:105` is
`DT = 0.1`.)*

## 2. Which ladder is authoritative — the published one, 0.1 / 0.4

**The 0.1 / 0.4 ladder is authoritative.** It is the source of every coherence adjective the
programme has printed:

| source | evidence |
|---|---|
| `taniteval/taniteval/four_families.py:888-890` | `maneuver_consistency_verdict` — `< 0.1` DECORATIVE, `< 0.4` WEAK, `≥ 0.4` SUBSTANTIAL |
| `Project Steering/MODEL_REGISTRY.md:1111` | *"κ **0.6033** (SUBSTANTIAL)"* — and the registry is the **only quotable source** for model facts |
| `Paper/TANITAD_PAPER.md` (v1arch four-family block) | *"TACTICAL — κ 0.6033 (substantial…)"* |
| `Project Steering/V5_FLAGSHIP_DEEP_REVIEW.md:74` | *"0.253 **WEAK** \| 0.0072 **DECORATIVE**"* |
| `stack/scripts/v5_guard.py:216-217` | the **v5 GPU-spend guard** cuts at the same 0.1 / 0.4 |

**And 0.2 is published nowhere.** `Project Steering/EVAL_DOCTRINE.md` contains **no κ ladder at
all** (grepped: zero matches), so there is no competing authority. The only doc mentioning 0.2 is
`RETRACTION_LOG.md:3270`, and it describes the instrument (*"the panel's **own** 0.2 coherence
threshold"*) rather than publishing a band.

⇒ **The code was wrong, not the docs.** Fixed the code.

### ⭐ It was already producing two words for one number

`hier_v1-lf19.json` emits κ **0.253** as **"SUPPORTED — … cohere"**, while
`V5_FLAGSHIP_DEEP_REVIEW.md:74` publishes the **same κ 0.253** as **WEAK**. One measurement, two
contradictory adjectives, both shipped. This is the drift the single source removes — and note the
repair direction: on the ladder 0.253 is WEAK, so the fixed code now **agrees** with the published
doc instead of contradicting it.

---

## 3. ⚠️ THE FLIP — one published verdict changes. ESCALATED, not silently re-published.

### 3a. `verdict_stable = true` → **false** (PUBLISHED)

Arm **`flagship-v1arch-v2bal-30k`**, step 29999, **880 windows / 40 OOD-val q90 episodes**
(`…/2026-08-06-v1-defect-triage/results/hier_v1arch_gateswept.json.xz`). Recomputed by me with the
fixed instrument:

| gate (rad) | κ | published ladder | old rule (κ ≥ 0.2) |
|---|---|---|---|
| 0.15 (published gate) | 0.5787 | SUBSTANTIAL | ✅ |
| 0.10 | 0.5715 | SUBSTANTIAL | ✅ |
| 0.06 | 0.4796 | SUBSTANTIAL | ✅ |
| 0.04 | 0.4075 | SUBSTANTIAL | ✅ |
| 0.02 | 0.3065 | **WEAK** | ✅ |
| 0.01 | 0.2038 | **WEAK** | ✅ |

- **as banked / as published:** `verdict_stable = true`, `kappa_range [0.2038, 0.5787]`
- **with the fixed instrument:** `verdict_stable = **false**`, `verdicts_across_sweep = ["SUBSTANTIAL", "WEAK"]`
- **n = 880 windows / 40 episodes. CI: none — κ carries no interval in this instrument (see §0).**

**Where the `true` is published, and which sentences are affected:**

| document | line | statement |
|---|---|---|
| `Project Steering/RETRACTION_LOG.md` | 3268-3271 | *"`verdict_stable = true` — κ stays at or above the panel's own 0.2 coherence threshold at **every** swept gate"* |
| `…/2026-08-06-v1-defect-triage/results/GATE_RERUN_RESULT.md` | 30 | *"`kappa_range = [0.2038, 0.5787]`, `verdict_stable = true`"* |
| `…/2026-08-06-v1-defect-triage/V1_DEFECT_TRIAGE.md` | 209 | *"`verdict_stable = true`"* |

⭐ **What survives, and it is the load-bearing half.** The RETRACTION_LOG's *substantive* conclusion
— *"the published coherence call was NOT an artifact and is not retracted"* — **stands**. The
verdict is SUBSTANTIAL at **0.15 and 0.10**, which is the gate move R-2026-08-06-yawgate was
actually about, and `V6F_PLANNER_DESIGN.md:564` re-reads at 0.10. **No live decision moves.**

⛔ **What fails is the stability CERTIFICATE over the full swept range**, and the RETRACTION_LOG
sentence quoted above is now doubly wrong: it asserts stability *and* names the wrong ladder while
doing it. Its own next paragraph already warned the magnitude *"does not travel"*; the corrected
reading is that the **verdict does not travel to the fine end either** — SUBSTANTIAL down to 0.04,
WEAK at 0.02 and 0.01.

**⇒ ESCALATION (PI / registry owner):** `RETRACTION_LOG.md:3268-3271`, `GATE_RERUN_RESULT.md:30`
and `V1_DEFECT_TRIAGE.md:209` need their `verdict_stable = true` restated as
*"SUBSTANTIAL at 0.15–0.04, WEAK at 0.02–0.01; stable across the gates the programme actually
uses (0.15, 0.10), not across the full sweep."* **I have not edited those three documents** — a
published retraction entry is not an agent's to rewrite. They are listed here so the correction is
visible rather than discovered in an audit.

### 3b. `B_layers_mutually_consistent.verdict` — 2 of 14 panels (EMITTED, never published)

Scanned **every** banked hierarchy panel (`hier*.json[.xz]`, 14 with a κ):

| κ | band | old word | new word | n | panels |
|---|---|---|---|---|---|
| 0.5787 – 0.6938 (9 panels) | SUBSTANTIAL | SUPPORTED | SUPPORTED | 265–881 | unchanged |
| **0.253** | **WEAK** | **SUPPORTED** | **WEAK** | **418 win / 19 ep** | `…/2026-08-02-four-family-panel/hier_v1-lf19.json` + its `stack/experiments/pod-rescue-20260802/` duplicate |
| 0.0072, 0.0217 (3 panels) | DECORATIVE | WEAK | WEAK | 418–881 | word unchanged; band now named |

**This block is emitted but consumed by nothing and quoted by no document** — verified at source:
`four_families.py:866`, `runner.py:366` and `report.py:224` all read only the
`A_conditioning_helps_conditioned_layer` block. So this is **a repair, not a re-publication**: the
one flip moves κ 0.253 from "SUPPORTED … cohere" to **WEAK**, which is exactly what
`V5_FLAGSHIP_DEEP_REVIEW.md:74` already publishes for that number.

---

## 4. The fix — one constant, imported, never restated

**`taniteval/taniteval/four_families.py`** (the module that publishes the word) now owns it:

- `KAPPA_VERDICT_LADDER = ((0.1, "DECORATIVE"), (0.4, "WEAK"), (inf, "SUBSTANTIAL"))`
  — docstring cites `MODEL_REGISTRY.md:1111`, the paper, `V5_FLAGSHIP_DEEP_REVIEW.md:74` and
  `v5_guard.py:216-217`, plus the defect that earned it.
- `kappa_band(k)` → the short band NAME (the **comparison key** for stability); `None` for an
  uncomputable/NaN κ, so "not measurable" can never be read as "unrelated".
- `kappa_verdict(k)` → the **published string**, byte-identical to what banked artifacts carry
  (the long DECORATIVE gloss included), so panels stay comparable.
- `tactical()` call site refactored to call them; also emits `maneuver_consistency_band`.

**`taniteval/taniteval/hierarchy.py`** imports them (`hierarchy → four_families` is acyclic and
import-safe; the reverse is not, because `hierarchy` pulls the whole stack at import time):

- `:262` `verdicts = {kappa_band(k) for k in ks}` — `verdict_stable` now tests the shipped verdict.
- `:1140` `coherent = (kappa_band(...) == KAPPA_VERDICT_LADDER[-1][1])`.
- New output fields so a reader sees the flip without recomputing: `verdicts_across_sweep`,
  `verdict_ladder`, per-gate `maneuver_consistency_band` / `maneuver_consistency_verdict`, and
  `B_layers_mutually_consistent.maneuver_vs_trajectory_kappa_band`.
- `_verdict_note` now names the ladder **and warns that a pre-2026-08-16 `verdict_stable` is not a
  statement about the published verdict** — recompute it from `per_gate` before quoting.

## 5. The tests that stop it recurring

`taniteval/tests/test_kappa_ladder_single_source.py` (new, 24 cases):

1. `test_ladder_is_exactly_the_published_bands` — pins 0.1 / 0.4 / SUBSTANTIAL.
2. 12 parametrised boundary cases, incl. the exact published κs (0.0072, 0.253, 0.5787, 0.6033).
3. `test_uncomputable_kappa_is_none_not_a_band` — `None`/NaN stay `None`.
4. `test_published_verdict_strings_are_byte_identical_to_banked_artifacts`.
5. ⭐ `test_hierarchy_imports_the_ladder_rather_than_restating_it` — asserts **object identity**
   (`is`), because an *equal copy* is precisely what drifts.
6. ⭐ `test_no_bare_kappa_threshold_survives_in_hierarchy` — **AST-based** regression guard over
   `_gate_sensitivity` and `_thesis`: no comparison to the retired `0.2`, and each must call
   `kappa_band`.
7. `test_gate_sensitivity_stability_is_evaluated_on_the_ladder` — the real 880-window sweep as a
   fixture: unstable on the ladder, "stable" under the retired rule.
8. `test_v5_guard_in_stack_still_agrees_with_the_canonical_ladder` — **source-level drift detector**
   for `stack/scripts/v5_guard.py`, which re-implements the ladder because it runs on a pod where
   `taniteval` is not importable. It cannot import the constant, so it is pinned by source.

`taniteval/tests/test_gate_sensitivity.py::test_stable_agreement_reports_stable` was **tightened,
not loosened**: it asserted `min(kappa_range) >= 0.2` (the retired private cut); it now asserts
`verdicts_across_sweep == ["SUBSTANTIAL"]`. Its two load-bearing siblings
(`test_verdict_instability_is_reported`, `test_every_swept_gate_is_present_and_ordered`) were
**measured to keep their outcomes under the new ladder before any edit** — no assertion was
weakened to make anything pass.

### ⚠️ A trap hit while writing the guard, worth logging

The first regression guard was a **regex over the module source** — and it failed on **my own
comments explaining that `kappa >= 0.2` was retired**. The searched token appeared in the prose
documenting it. This is the `CLAUDE.md` polling-monitor trap in a new costume (*a filter containing
the pattern it searches for matches its own echo*). Worse, a regex keyed on the identifier `kappa`
would have **missed the original defect entirely**, because `:262` read `bool(k >= 0.2)` — the
identifier is `k`. ⇒ the guard is an **AST walk**, which has no comments or docstring bodies to
trip over and matches the *structure* rather than the *spelling*.

## 6. Suite status — green

Run on this box, `PYTHONUTF8=1`, `OMP_NUM_THREADS=6`, CPU only:

| suite | before my change | after |
|---|---|---|
| `taniteval/tests` | 1018 passed, 0 failed | **1042 passed, 0 failed** (+24 new) |
| `stack/tests` | 2839 passed / 17 skipped / 2 xfailed | **2919 passed / 17 skipped / 2 xfailed, 0 failed** |

⚠️ **Two notes on the baseline.** (a) The brief's *"2,810 passed"* is `stack/tests` at an older
commit; the invariant (**0 failed / 17 skipped / 2 xfailed**) holds. (b) `stack/tests` grew
2839 → 2919 **during this task** — sibling agents committed (`HEAD` moved to `b12c190`, new
`test_v6_*` files landing 01:23–01:46). I touched nothing under `stack/`; that delta is theirs.

---

## 7. Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| single-sourced ladder + `kappa_band` / `kappa_verdict` | `taniteval/taniteval/four_families.py` | modified, **staged** |
| `verdict_stable` + `_thesis` on the published ladder | `taniteval/taniteval/hierarchy.py` | modified, **staged** |
| pinning + drift tests (new) | `taniteval/tests/test_kappa_ladder_single_source.py` | new, **staged** |
| tightened gate test | `taniteval/tests/test_gate_sensitivity.py` | modified, **staged** |
| this writeup | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-verdict-stable-kappa/VERDICT_STABLE_KAPPA.md` | new, **staged** |

**Nothing committed, nothing pushed.** ⚠️ The index also contains **other agents' staged work**
(`stack/scripts/e_wc2_sigma_star.py`, `stack/tests/test_e_wc2_sigma_star.py`,
`taniteval/tools/eval_four_families.py`, `taniteval/tests/test_eval_four_families_tool.py`,
`stack/scripts/refc_dump_latents.py`) — per `CLAUDE.md`, any commit must use an explicit pathspec
or declare the foreign entries.

## 8. Escalations

1. ⛔ **Three documents publish `verdict_stable = true` for the v1arch panel and are now stale**
   (§3a): `RETRACTION_LOG.md:3268-3271`, `GATE_RERUN_RESULT.md:30`, `V1_DEFECT_TRIAGE.md:209`.
   The substantive claim survives; the stability certificate does not. **Not edited by me** —
   a retraction entry needs its owner. Correct wording proposed in §3a.
2. ⚠️ **Every pre-2026-08-16 `verdict_stable` in a banked artifact is uninterpretable as published
   verdict stability.** Only one panel was ever gate-swept, so the blast radius is that single
   file, but the field name is unchanged — hence the in-artifact warning added to `_verdict_note`.
3. ⚠️ **`kappa_turn_subset` is still gate-dependent and still NOT swept** (inherited from the
   2026-08-15 re-read, re-confirmed at source here). The v1arch panel's `kappa_turn_subset = 0.2005`
   over **217 turn-active windows** sits on the retired 0.2 line and, on the published ladder, sits
   **just inside WEAK** — it cannot be re-read at another gate even on the swept panel. Work item:
   extend `_gate_sensitivity` to sweep the turn subset. **Not done here** — it changes what the
   panel measures, which is a scope decision, not a defect fix.
4. ⚠️ **`stack/scripts/v5_guard.py` and `…/2026-08-05-alpamayo2-super/tools/a2_four_families.py`
   still carry their own copies** of the ladder (both currently correct). `v5_guard` is now covered
   by a source-level drift test; the Alpamayo tool is a frozen research artifact and was left alone.
