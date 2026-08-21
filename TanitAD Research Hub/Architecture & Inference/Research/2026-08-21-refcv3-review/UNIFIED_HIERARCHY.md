# The unified hierarchy — measured across four arms, extracted, and audited against REF-C v3

`MEASURED (ours, 2026-08-21)` — every number below comes from **building the
arms and counting**, not from a docstring. **T0 throughout.**
PI instruction: *"use the same and best hierarchy architecture in all our
designs"*.

---

## 1. The four arms, measured

| component | **v6** | REF-A v1 | REF-D | **REF-C v3** |
|---|---|---|---|---|
| **tactical** | **5,767,981** | 54,587,392 (+ pool 4,198,400) | 54,587,392 (+ pool 4,198,400) | **1,980,646** |
| ↳ its **own predictor** | ⭐ `predictor_tac` **3,809,792** | ⛔ none | ⛔ none | ⛔ none |
| **strategic** | **4,152,993** | 1,911,040 | ⛔ **NOT BUILT** | **195** (+ cond 66,816) |
| ↳ its **own predictor** | ⭐ `predictor_str` **3,481,856** | ⛔ none | ⛔ none | ⛔ none |
| shared vocabulary tables | ⭐ `vocab_tac/str/a_lat/a_lon/a_str` | imports the tuples | imports the tuples | ⛔ **none** |
| goal representation | tokens **+** geometric | tokens | tokens | geometric only |
| arm total | 87,893,449 (default cfg) | 143,858,464 | 142,332,174 | 62,930,419 |

### 1.1 Two findings that are defects, not preferences

⛔ **REF-D's strategic layer is configured but never built.** `RefDConfig` carries
`str_dt`, `str_steps`, `w_future_str`, `strategic_cfg` — and
`RefD.named_children()` has **no `strategic`**, with **zero state_dict keys
containing "str"**. Its design doc claims *"3 rates … operative 0.2×30, tactical
0.6×10, strategic 1.5×4"*. ⚠️ **I wrote that design doc.** The three-rate claim is
not implemented.

⛔ **REF-C v3's strategic layer is a 195-parameter linear head.** It has the
*shape* of a hierarchy without the *machinery*: no predictor at either rung, and
a strategic "layer" smaller than a single attention head.

### 1.2 ⭐ v6 is alone on the axis that matters

**Only v6 gives each layer its OWN predictor** — the PI's three-planner directive
(*"each predicting via imagination; strategic gets its OWN predictor on a
strategy-only latent subspace"*). That is 7.29 M of the 9.92 M, and it is the
whole difference between *a hierarchy that imagines* and *a stack of heads*.

⇒ **v6's rung is the reference.** REF-A v1's 54.6 M tactical block is bigger but
is a plain transformer with no predictor; size is not the axis.

## 2. What was BUILT — `tanitad/models/hierarchy.py`

One rung, four consumers:

```
adapter    Linear(d_in, hidden) → GELU → Linear(hidden, d_layer) → LayerNorm
cond       GoalConditioner(vocab_ABOVE)      — omitted at the top rung
predictor  FTac(d_layer, d_goal=…)           ⭐ THE LAYER'S OWN IMAGINATION
goal_head  GoalHead(vocab_own, d_layer, d_cond=…)
act_heads  GoalHead(vocab_act, d_layer) × N  — factored, never a joint softmax
```

⛔⛔ **ADDITIVE ONLY — v6 is training under tensor-strict resume.** The module
does not touch `V6Stack`; it re-composes the **same component classes** in the
same order with the same dims. `test_v6_itself_is_UNTOUCHED_by_this_module` pins
**87,893,449 params / 405 state_dict keys**.

⭐ **And it is PROVEN, not asserted.** `assert_matches_v6` compares
component-by-component against the live stack:

| component | rung | v6 |
|---|---|---|
| tactical.adapter | 1,312,768 | 1,312,768 |
| tactical.predictor | 3,809,792 | 3,809,792 |
| tactical.cond | 2,816 | 2,816 |
| tactical.goal_head | 236,817 | 236,817 |
| tactical.act_head[0,1] | 202,894 ×2 | 202,894 ×2 |
| strategic.adapter | 394,496 | 394,496 |
| strategic.predictor | 3,481,856 | 3,481,856 |
| strategic.goal_head | 139,283 | 139,283 |
| strategic.act_head[0] | 137,358 | 137,358 |

**8 tests pass**, including `test_assert_matches_v6_CAN_FAIL` (a guard that
cannot fail is not a guard) and `test_vocabulary_is_SHARED_not_copied` (`id()`
identity — the "second vocabulary" failure the programme paid for once).

⚠️ **The caller keeps the gradient policy.** The rung does not detach: v6 uses
`_cut()`, REF-C v3 uses `cons_detach`. Hiding a detach inside a shared component
would make both callers' policies invisible at their call sites.

## 3. ⭐ Audit — validated improvements vs REF-C v3

### 3.1 Present ✅

| lever | evidence | in v3 |
|---|---|---|
| **factored lat(3)/lon(3)**, never the 5-way | the 5-way *"provably destroys the longitudinal decision"* | ✅ |
| **candidate-independent selection** (`GoalDistanceScorer`) | SEL-1 refused a learned re-scorer (winner's curse); error-rank FALLS with N | ✅ |
| **no learned fan re-scorer** | v1.2 NOT separated across 47 arms | ✅ excluded |
| **no MPC/CEM** | C101: 35.8 % worse than CV at T1 | ✅ excluded |
| **no ego into any goal head** (E11) | pinned by test + `goal_provenance` **intervention** audit (gradient probes are blind — C120) | ✅ |
| **predicted goal, not supplied route** (E12) | PI 2026-08-03 | ✅ |
| **zero-init FiLM, bit-inert at init** | preflight `pooled_rel_move 0.0` | ✅ |
| **speed as input** (`v0` channel) | REF-A 3.73 → 0.83 fwd-ADE | ✅ (15 refs in v3, 86 in core) |
| **ego-dropout anti-shortcut** | `35956b2` | ✅ (core) |
| **6 s horizon, 8 slots** | PI binding horizon spec | ✅ |
| **anchors 128** | *"the knee is anchor count, not encoder scale"* | ✅ |

### 3.2 Missing and APPLICABLE ⛔ — these are the build items

| lever | why it applies | cost |
|---|---|---|
| ⭐ **per-layer OWN predictor** | the three-planner directive; v3 has none at either rung | tactical 3.81 M + strategic 3.48 M |
| ⭐ **shared vocabulary** | REF-A v1 + REF-D both import it; v3 declares its own | ~0 (import) |
| ⭐ **strategic layer with real capacity** | 195 params is not a layer | 4.15 M |
| ⭐ **D-008 scale ≥ 250 M** | standing PI decision; v3 is 4× under | `base_width` 64 → 124 ⇒ core 215.6 M |

⇒ **Rebuilt v3 ≈ 215.6 M core + 9.92 M hierarchy + decoder ≈ 228 M**, or
**≈ 262 M** with the full `refc_xl_config`. The hierarchy cost is **constant
across the size ladder** (2.05 M → 2.17 M measured), so scaling and hierarchy are
independent decisions.

### 3.3 Missing but NOT applicable — stated so it is not "fixed" by mistake

| lever | why it does NOT transfer |
|---|---|
| **SIGReg** (v6: 26 refs; v3 and `refc.py`: **0**) | SIGReg prevents collapse of a **self-predicted latent**. REF-C is supervised end-to-end on trajectory — there is no self-target to collapse. ⚠️ Adding it would regularise a latent that has an external anchor already. |
| **ViT5 encoder** (v6: 4; v3: 0) | REF-C's encoder is the **ResNet/BEV lineage from DiffusionDrive**, and the anchor decoder cross-attends its 8×8×F map. Swapping the encoder is a different experiment, not a hierarchy fix. |
| **`masked_cells` (O3)** | operates on the readout cell grid, which REF-C does not have |
| **IDM / aux ego-motion** | REF-A has it; **v6 does not either**. It grounds a self-supervised latent; REF-C's trajectory supervision already grounds ego motion directly. |

## 4. ⚠️ What this costs, and the decision that is the PI's

Adopting §3.2 **voids the current `PREREG_REFC_V3.md`** — the pinned arm-delta
changes, and the registered cost (~7–9 h A40/run) rises with the core.

⭐ **Amending a pre-registration BEFORE any read is legitimate; after a read it is
not.** Nothing has been run. **So if it changes, it changes now.**

**Two coherent packages:**

| | scope | total | prereg |
|---|---|---|---|
| **A — hierarchy only** | swap `PhiTac`+195-param head → the shared rungs; import the vocabulary | ~72.8 M | amend the delta |
| **B — hierarchy + D-008** | A, plus `base_width` 124 | ~228 M (262 M at `refc_xl_config`) | amend delta **and** cost |

⚠️ **B is the one that tests the programme's thesis.** `D-008` ties scale to *"a
scale where hierarchy is expressible"*, and REF-C v3 exists to show the hierarchy
buys something. Running it at 62.9 M with a 195-parameter strategic layer risks a
null that says nothing — the arm would be too small for the claim it is testing.

## 5. Follow-ups this opened

1. ⛔ **REF-D needs its strategic layer built** — the shared rung is now the
   obvious way, and its config already carries `str_dt` / `w_future_str`.
2. **REF-A v1's tactical block (54.6 M, no predictor)** should be reviewed
   against the rung — bigger, but without the imagination the directive requires.
3. **v6 should eventually consume `HierarchyRung` itself**, so the shared
   component has no privileged copy. ⛔ **NOT while S-W is training** — it is a
   state_dict-neutral refactor but not a zero-risk one, and the resume is
   tensor-strict.

## 6. What I did NOT verify

* **No forward-pass equivalence** — `assert_matches_v6` checks **parameter
  counts component-by-component**, not activations. Identical counts with a
  transposed dim would pass. A tensor-level equivalence test is the stronger
  check and is not written.
* **v3 was not rebuilt with the rungs wired in.** §3.2's ~228 M is arithmetic
  over measured parts, not a built model.
* **The `refc_xl_config` 251,932,584** figure is quoted from `refc.py`'s
  docstring, not re-measured.
