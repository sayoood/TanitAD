# PRE-REGISTRATION — D-VT1: the leak-guarded target-speed label, and F1

**Date:** 2026-08-04 (Europe/Berlin) · **Stream:** Architecture & Inference ·
**Branch:** `agent/arch-inf-20260803` · **GPU-days spent: 0. No training launched.**
**Parent escalation:** `Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md` §9.1 —
*"the blocker is a leak-guarded label"*.

**Estimator for every interval below:** `taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`,
unit = val episode, B = 2000. ⛔ `overlapping_holdout_se` is never called: it biases the POINT
estimate bidirectionally, up to a sign flip on paired deltas.

---

## 0. WHAT IS AND IS NOT PRE-REGISTERED — stated first, because the honest scope is narrow

| block | status when this file was written | why |
|---|---|---|
| **§3 — the F1 readout probe** (`vt_f1_readout_probe.py`) | ⛔ **NOT YET RUN.** `raw/vt_f1_readout_probe.json` **does not exist** at the moment this file is hashed. | This is the block the pre-registration protects. Both outcomes are committed in §3.3 before any number is visible. |
| **§4 — the retrain arm** | **NOT RUN and NOT REQUESTED IN THIS BRIEF.** A GPU-day is the PI's call. | §4 specifies the cheapest discriminating arm so the decision can be made on a costed proposal rather than a vibe. |
| **§1–§2 — the label design and its leak audit** | ⚠️ **ALREADY RUN when this file was written.** `raw/vt_labels_val40.json` and `raw/vt_leak_audit.json` existed. | ⛔ **They are therefore EXPLORATORY, not pre-registered, and are labelled as such everywhere.** Saying otherwise would be the retro-fit this programme keeps retracting. What protects them instead is that the guard is a **mechanical** property (index disjointness, unit-tested) rather than a threshold anyone could have tuned. |

⚠️ The same applies to `raw/vt_four_families.json` and `raw/vt_irreducible.json` — both existed
before this file. EXPLORATORY.

---

## 1. The label — fixed, and already implemented

`stack/tanitad/lake/vtarget.py::vtarget_guarded`, guard `VT_GUARD_STEPS = 20`.

* read window `v[l + 21 : min(l + 200, T)]` = **[t + 2.1 s, t + 20 s]**
* scored set `{l … l + 20}` = **[t, t + 2.0 s]** — the decoder's `horizons = (5,10,15,20)`,
  `lead_source.K_MAX = 20`, and the manoeuvre label `dv = v(t+2 s) − v(t)`
* **disjoint by index arithmetic**, asserted per window at build time and pinned by
  `stack/tests/test_flagship_v15.py::test_vtarget_guarded_read_window_is_DISJOINT_from_the_scored_horizon`
* fallback on an invalid window is the **raw** `v[l]`, never the smoothed `vs[l]` (a zero-phase
  smoother reads 0.5 s past `t`), and `valid=False` routes it to the DROPPED token

**The guard constant is DERIVED, not chosen.** Raising any scored horizon must raise it; the test
pins the coupling so it cannot drift silently.

---

## 2. The admissibility check, applied — and its verdict fixed in advance of §3

The PI's binding question: *could this have been computed from the thing being measured, or from
the situation classifier's output?*

| signal | side | verdict |
|---|---|---|
| `vtarget_guarded` as a **LABEL** | offline | ✅ **ADMISSIBLE.** PI ruling 2026-08-03: labels may use ego, other agents, maps, future poses. |
| `vtarget_guarded` **SUPPLIED at inference** | inference | ⛔ **INADMISSIBLE** — see §2.1. |
| a **PREDICTED** `v̂_target` from image + `v0` | inference | ✅ admissible **iff** its inputs are inference-legal and disjoint from the situation classifier's output. |
| the situation classifier's posterior / argmax / embedding in the goal path | inference | ⛔ **INADMISSIBLE**, unconditionally (PI 2026-08-03). |

### 2.1 ⛔ Registered BEFORE §3: the verdict on the supplied label does not depend on §3

The excision is **necessary and not sufficient**, and the residual was measured in §2's exploratory
audit. Whatever §3 returns, **a supplied target speed is not made admissible by the horizon guard**.
This is registered here so the §3 result cannot be read as re-opening it.

---

## 3. THE PRE-REGISTERED BLOCK — the F1 readout probe

### 3.1 Substrate and arms — fixed

`…/2026-08-03-dtac1-tactical-head/dtac1_substrate_refc-base-30k.pt`: REF-C-base step 29999's frozen
`pooled` (704-d), `v0`, and (lat, lon, man5) for **1364 windows / 39 val episodes**. The join to the
pose track is asserted bit-exact at runtime.

| arm | head input | legal at inference? |
|---|---|---|
| **A** `img` | `pooled` — the SHIPPED head | ✅ |
| **B** `img+v0` | **F1**, `--tactical-speed-input`, +384 params | ✅ |
| **C** `img+v0+vt_pred` | + a target speed predicted inside the training fold | ✅ |
| **D** `img+v0+vt_label` | + the SUPPLIED guarded label | ⛔ ceiling only |

Head shape = the real one, `Linear(d,384)-ReLU-Linear(384,5)`; 3 seeds (0,1,2), mean logits;
a multinomial-logistic linear probe reported alongside as variance-free corroboration.
Leave-one-episode-out, 39 folds. `vt_pred`'s ridge is fit **inside** the training fold.

### 3.2 Primary and secondary readouts — fixed

* **PRIMARY: `B − A` on 5-way accuracy**, paired episode-cluster bootstrap, CI excludes 0.
* **CO-PRIMARY: `B − A` on longitudinal-class recall** (`accelerate` ∪ `brake_stop`), same estimator.
* Secondary, all reported, none decisive: macro-recall, macro-F1, `C − A`, `D − A`, per-class
  precision **and** recall **with both denominators**, and the speed-stratified table.

⚠️ **Reported with both denominators, always.** A previous report claimed *"brake_stop 0.026 → 0.503,
a free win"* on the wrong denominator, with no precision — precision had in fact fallen
0.2340 → 0.1711 on 380 fires against 153 true. That shape is banned.

### 3.3 ⛔ BOTH OUTCOMES, COMMITTED NOW

| outcome | what I will write |
|---|---|
| **B − A separated ABOVE 0** | F1 is a real readout-level lever. It is **still not a launch authority**: the probe is optimistic (frozen trunk, no `ego_dropout`), so the honest claim is *"the speed channel carries decision-relevant information the frozen image embedding lacks"*, and §4's retrain is the cheapest way to price it. |
| **B − A CI includes 0** | F1 does **not** separate at the readout level. I will write that plainly, will **not** re-cut the metric to find a win, and will state that a retrain is then a **worse** bet than it looked — the +384 params buy nothing the frozen fan can already see. The hypothesis that survives is that the speed channel needs a **co-adapting trunk**, and that hypothesis is *not* evidence for the arm. |
| **B − A separated BELOW 0** | F1 HURTS at the readout level. Reported as such; the first hypothesis is the extra input column diluting a 704-d embedding under a fixed step budget, and the linear probe is the check. |
| **D ≫ C** | the lever's value lives in information a predictor cannot recover ⇒ the deployable form is weak and the ceiling is not reachable. This is a NEGATIVE result for the whole VTARGET input direction and will be reported as one. |
| **C ≈ D** | the predicted goal recovers the ceiling ⇒ the deployable path is viable and §4 is worth a GPU-day. |

⛔ **No metric will be added, removed or re-cut after the numbers are seen.** Anything computed
afterwards is labelled POST-HOC in the results document.

---

## 4. The cheapest discriminating retrain arm — specified, NOT requested

**One axis.** `--tactical-speed-input` alone, on the otherwise-unchanged shipped 5-way head.
⛔ **Not coupled to `--factored-maneuver`** — that coupling was a real defect, removed 2026-08-03,
and reintroducing it would leave F1 estimable only as `full − f2only` where the arms also differ in
the head itself.

* **arm:** `refc_f1only_config()` (already in `refc.py`), REF-C-base scale, parity corpus
  `physicalai-train-e438721ae894`, skip-hash `f09e44db`, identical seed/schedule to `refc-base-30k`.
* **control:** `refc-base-30k` itself — already trained, so the A/B costs **one** run, not two.
* **capacity delta:** +384 params (+0.00037 %), pinned by
  `tests/test_refc_tactical.py::test_f1only_is_not_a_capacity_change`.
* **read:** the same PRIMARY/CO-PRIMARY as §3.2, on the canonical val grid.
* **first knob if it lands weak:** `ego_dropout` (0.5 today — the head is trained without the
  channel on half its samples, then evaluated with it).

**A GPU-day is the PI's decision. This section is a costed proposal, not a launch.**

---

## 5. Content pin

`git hash-object` of this file is recorded in `raw/prereg_pin.json` together with the
**non-existence** of `raw/vt_f1_readout_probe.json` at pin time, and re-verified in the results
document.
