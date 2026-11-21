# refcv6 perception supervision — the four MEASURED defects, FIXED, PROVEN and EVALUATED

**Author:** senior implementer, Architecture & Inference, 2026-09-23.
**Mandate (PI):** *"Fix all findings and evaluate them."*
**Source of the findings:** `…/2026-09-22-refcv6-review/PERCEPTION_DATA_REVIEW.md` (D-1…D-4).
**Repo:** `D:/Projects/TanitAD`. ⛔ **STAGED, NEVER COMMITTED, NEVER PUSHED.**

⛔ **LINE DISCIPLINE.** Every corpus number below carries its line. **v7-B1** =
`physicalai-b1-w120-256x640cyl` (4,566 joined clips / 875,657 frames / 28,958,699 boxes) —
the line refcv6 trains. **parity** = `physicalai-train-e438721ae894` (2,308 clips /
12,122,129 boxes). **eval-139** = the 139 B1 eval clips; the SAM3 map GT on this box is
**135** of them. Clip ids appear only as `sha256(clip_id)[:12]`.

---

## 0. Verdict in one table

| # | finding | fixed? | mutation | measured effect | changes an existing arm? |
|---|---|---|---|---|---|
| **D-1** | BOX loss applied **no visibility filter** | ✅ at the root, in `box3d_set_loss`, **applied BEFORE `match_slots`** | **CAUGHT** ×2 (`M1`, `M2`) | **22,685,275 of 27,350,533 query slots (82.94 %) removed** — supervision for boxes the camera cannot see; **13,696,011 of them (50.08 %) were BEHIND THE EGO** | ⚠️ **YES — BEHAVIOUR CHANGE, default ON.** `--no-box3d-visible-filter` is the regression arm |
| **D-2** | query budget spent half its slots behind the ego | ✅ **by the same call, ordered before the budget** | **CAUGHT** (`M3`, which reverts only the ORDER) | **+76,395 in-field ∩ decode-box boxes recovered (+1.638 %)**, class-biased **toward VRUs**: person **+4.63 %**, rider **+2.45 %**, automobile **+0.60 %**, animal/other_vehicle **0 %** | same change as D-1 |
| **D-3** | MAP `seen` is a **clip-lifetime** mask | ✅ `map_loss_row(..., lift_valid=)`, input emitted by the branch | **CAUGHT** ×2 (`M4`, `M5`) | **2,170,570 of 19,647,460 supervised cells = 11.048 % removed** (eval-139, 135 clips, 2,825 frames) — reproduces the reviewer's independent P7 figure **to the digit** | ⚠️ **YES when the caller passes the mask.** The module default is `None` = today's behaviour; the trainer patch turns it on with `--no-map-lift-valid-mask` as the regression arm |
| **D-4** | the corpus-line guard **could not go red** on the error it names; and D-1 invalidates the banked vector | ✅ expectation **derived from the arm's join**; artifact now carries **both populations** | **CAUGHT** ×2 (`M6`, `M7`) | per-class shift **1.746×** (`other_vehicle`) … **0.663×** (`stroller`), **2.63× end to end**; the visible vector is a **new, producer-built** artifact | ⚠️ **Opt-in.** `target_population` defaults to `raw_join`, so **every existing arm loads a bit-identical vector** (digests `c3937558f59299e7` / `bde3aa19dfd0e59a` unmoved) |
| extra | `map_soft_ce` takes **no class weight** (97.05 % of label mass in 3 of 9 classes) | ✅ mechanism added, **default `None`** | **CAUGHT** (`M8`) | no vector banked — the only census is **eval-139**, and shipping it for a v7-B1 train arm would be the scope error this whole package is about. **NAMED BLOCKER** | ❌ no — `class_weight=None` is bit-identical |

**Mutation harness: 8 arms, 8 CAUGHT, 0 escaped, 0 overbroad, 0 invalid**, with the test
file green **before** the run and green **after restore**, and every mutated file restored
bit-identical (`raw/mutation_perception_fixes.json`).

**Regression: 948 passed / 0 failed** over **all 34 test files that import any changed
module**, at the final code state. The full `stack/` suite reads **8,686 passed / 15
failed**; **14 of the 15 are a sibling's unstaged in-flight edits and the 15th is my own
edit-during-a-31-minute-run artifact** — every one attributed, with controls, in §7. ⭐ §7
also carries a **suite-wide blocker found and solved here**: `pytest -q` cannot be green
from either cwd until `PYTHONPATH` gains the repo's `taniteval/` directory.

---

## 1. D-1 — the BOX loss now filters, and it filters BEFORE the budget

### The fix

`stack/tanitad/models/box3d_head.py::box3d_set_loss` gained
`visible_filter: bool = True`. When on it calls `refc_agents.visible_target_filter` — **the
one existing spelling of the predicate, imported, never re-implemented** — and does so
**before** `agent_slots.match_slots`. That ordering is the whole of D-2 as well; see §2.

Three properties that are not incidental:

1. **It is the SAME filter the v6 seam applies.** MEASURED at HEAD:
   `refc_agents.py:828` `filter_visible: bool = True`, applied at `:849`, and stamped into
   the loss row at `:884`. ⚠️ *"and always has"* would be a history claim I did not probe —
   what I checked is the current default. The refcv6 path was the outlier, not the
   innovation.
2. **The `zh_mask` follows the filtered validity**, so a removed row cannot carry a 3-D
   height target into the z/h loop.
3. **A supplied `match=` with the filter on is REFUSED.** A match built over unfiltered
   targets would re-admit exactly the boxes the filter removed and `n` would then describe
   a different set from the loss — a scope error wearing a green test. Two existing tests
   were updated to say `visible_filter=False` **explicitly**, which states their scope
   rather than assuming it.

### The proof

`stack/tests/test_refcv6_perception_supervision_fixes.py`, expectations **literal or
analytic**:

* **three boxes decided before any code runs** — A `(+20, 0)` visible, B `(−20, 0)` behind
  the ego, C `(+120, 0)` at 2× the decode box. Filtered: `target_prefilter` **3**,
  `target_visible` **1**, `dropped_not_visible` **2**, `matched` **1**. Unfiltered: 3/3/0/3.
* **the survivor is named** — `valid == [True, False, False]` and the matched column is
  `[0]`, because a count of 1 does not prove it kept the *right* one.
* **the boundary is 60° and 60 m exactly** — five points, two of them 1.1 m apart on the
  same ray, with opposite verdicts; and the field-only filter keeps a point the
  decode-box filter drops, which is how we know *which* cut fired.
* ⭐ **the DEFAULT is asserted with no keyword at all**, on both `box3d_set_loss` and the
  trainer's entry point `box3d_loss_row`. This test exists **because the mutation harness
  demanded it**: arm `M1` (flip the default back to `False`) **ESCAPED on its first run**,
  since every other D-1 test names the flag explicitly and is blind to the default by
  construction. The trainer names no flag, so the default *is* the behaviour of every
  refcv6 arm.

### Mutations

| arm | reintroduced defect | verdict |
|---|---|---|
| `M1` | `visible_filter: bool = True` → `False` — the literal pre-2026-09-23 behaviour | **CAUGHT** (only by the no-keyword test; the explicit-keyword tests stay green, and that is recorded) |
| `M2` | `if visible_filter:` → `if False:` — declared, stamped, plumbed nowhere (the `occ_from_geometry` class) | **CAUGHT** |

### The measured effect (v7-B1, whole join, `raw/box_supervision_recovery.json`)

| quantity | value |
|---|---|
| A — query slots the defect consumed | **27,350,533** |
| A — of which on boxes **behind the ego** | **13,696,011 = 50.076 %** |
| B — visible boxes actually supervised before | **4,665,258** |
| C — visible boxes supervised after | **4,741,653** |
| **REMOVED (A − B)** — supervision for the unobservable | **22,685,275 = 82.94 % of A** |
| **RECOVERED (C − B)** — see §2 | **+76,395 = +1.638 % of B** |

⛔ **These are two different quantities and are never summed.** The fix mostly *removes*;
the *recovery* is D-2's half.

**Controls (all four pass):** the census reproduces the reviewer's independently written
`p2_b1_join_filter_census.json` on **all four** corpus totals (28,958,699 / 11,639,984 /
4,741,807 / 14,490,476); an **exact-code** arm ran the real `box3d_set_loss` in both
configurations on **250** sampled frames and agreed with this pass's arithmetic on
**250/250**; the **analytic** arm found **25** frames already carrying ≥100 visible boxes
and `after == 100` on all 25; the **null** arm (no field cut) reads recovery **exactly 0**.

---

## 2. D-2 — the budget now ranks the in-field set

### The fix, and why it is one line

`match_slots` keeps the `n_queries` **nearest** by `√(cx²+cy²)`. That policy is correct
*after* a field cut and destructive without one — a car 5 m **behind** outranks a car 40 m
**ahead**. The fix is therefore not a change to the budget at all; it is *what set the
budget is handed*. `box3d_set_loss` filters first, so the ranking runs over the in-field ∩
decode-box targets.

`tgt_raw` is kept in the function **only** so the deliberate-regression arm can name the old
ordering; nothing else reads it.

### The proof

* **analytic, five boxes.** Two queries; targets at `(−5, 0)`, `(−6, 0)`, `(+40, 0)`.
  Unfiltered the matcher keeps `{0, 1}` — both behind the ego — and the 40 m car ahead gets
  **zero** supervision. Filtered it keeps `{2}`. Both matched sets are asserted by index.
* ⭐ **the discriminating control**: with three targets at 10/20/30 m and two queries — all
  already visible — filtered and unfiltered must give **identical** counts *and a
  bit-identical total*. If the fix had also moved the budget policy this would fail, and the
  effect would no longer be attributable to the filter.

### Mutation

`M3` replaces exactly one line — `match_slots(pred, tgt)` → `match_slots(pred, tgt_raw)` —
so the filter still runs and only the ORDERING reverts. **CAUGHT**, with the D-1 and D-3
tests staying green (i.e. the arm is specific).

### The measured effect, and ⚠️ a RECONCILIATION the reader needs

**+76,395 in-field ∩ decode-box boxes recovered = +1.638 %.**

⚠️ **The review reports 625,379 = 27.18 %, and both numbers are right.** They answer
different questions, and this pass measured the review's version with its own code to prove
the relationship rather than assert it:

| scope | set | denominator | lost / recovered |
|---|---|---|---|
| review P6 | **in field (azimuth only)** | over-budget lines | **625,379 = 27.18 %** |
| here | **in field ∩ decode box** (what `visible_target_filter` keeps, i.e. what the head can EXPRESS) | whole corpus | **76,395 = 1.638 %** |

Recomputing P6's azimuth-only arm here gives **45,592** over-budget lines, **2,300,748**
available, **625,379** lost — **identical to P6 on all three**. The gap is real and
explainable: on a crowded frame most in-field boxes are beyond 60 m, so the **decode box**
removes them whether or not the budget does. Quoting 27.18 % as the supervision this fix
recovers would be a scope error of the same family as everything else in this package.

### ⭐ The recovery IS class-biased — and in the opposite direction from P6's loss table

| class | supervised before | after | recovered | ratio |
|---|---|---|---|---|
| **person** | 1,140,653 | 1,193,476 | **+52,823** | **1.0463** |
| **rider** | 138,529 | 141,927 | **+3,398** | **1.0245** |
| bus | 28,407 | 28,692 | +285 | 1.0100 |
| automobile | 3,200,014 | 3,219,249 | +19,235 | 1.0060 |
| heavy_truck | 63,967 | 64,278 | +311 | 1.0049 |
| trailer | 40,288 | 40,451 | +163 | 1.0040 |
| protruding_object | 26,281 | 26,366 | +85 | 1.0032 |
| stroller | 11,032 | 11,054 | +22 | 1.0020 |
| other_vehicle | 8,988 | 8,988 | **0** | 1.0000 |
| animal | 2,479 | 2,479 | **0** | 1.0000 |

(+73 more on the out-of-vocabulary `train_or_tram_car`, which carries geometry supervision
but no class label — a declared behaviour, `_out_of_vocabulary._handling`.)

⚠️ P6's *loss* table was headed by **bus 55.98 %** and **heavy_truck 43.85 %**; the
*recovery* is headed by **person** and **rider**. Not a contradiction — the same scope
difference: large vehicles are visible far away, so most of what they lost was outside the
decode box and is not recoverable supervision for this head. Inside the decode box, crowded
frames are crowded with **pedestrians**, and they are what the fix gives back.

---

## 3. D-3 — the MAP head stops being supervised on cells the camera never reached

### The fix

`refcv6_perception_branch`:

* `PerceptionBranch.forward` now emits `out["map_valid"] = map_valid_from_lift(valid)`
  — `[B, X, Y]` bool. ⭐ **This is the lift's OWN predicate**, `valid.any(dim=1)`, the exact
  one `BEVLift.forward` uses to decide where to substitute its learned `unobserved`
  embedding. Emitting it costs one reduction and zero parameters; re-deriving it at the loss
  site would be a second spelling of a projection test.
* `map_loss_row(..., lift_valid=None)` narrows `seen` to `seen & lift_valid`, applies the
  **same** mask to `map_metrics`, and reports **both** counts plus their difference:
  `n_map_cells`, `n_map_cells_seen`, `n_map_cells_unobserved`.

⚠️ **The module default is `None`** — i.e. today's behaviour — so nothing moves until a
caller passes the mask. The trainer patch (§6) passes it by default with
`--no-map-lift-valid-mask` as the named regression.

### The proof

* **literal counts.** 4×4 grid, 16 cells seen, 6 lift-valid ⇒ `n_map_cells` **6**,
  `n_map_cells_seen` **16**, `n_map_cells_unobserved` **10**.
* **separated by construction, not by a threshold.** The logits are perfect on the valid
  cells and maximally wrong elsewhere, so `map_acc` reads **1.0** masked and **6/16**
  unmasked, and the losses differ by three orders of magnitude.
* ⭐ **the mask is checked against a REAL `BEVLift`'s own output**, not against a
  re-derivation: a cell carries the `unobserved` vector **iff** `map_valid_from_lift` is
  False there.
* **the 590 is recomputed from the grid spec** — a cell whose azimuth exceeds 60° is
  outside the rig's only camera at every instant, and over the 120×64 grid there are exactly
  **590** of them (7,680 cells total). The review measured 590 and predicted 590.
* float / mis-shaped masks are refused (the same rule `map_soft_ce`'s `seen` carries).

### Mutations

| arm | reintroduced defect | verdict |
|---|---|---|
| `M4` | `m_seen = seen & lift_valid…` → `m_seen = seen` | **CAUGHT** |
| `M5` | the branch stops emitting `map_valid` (the fix's input disappears) | **CAUGHT** |

### The measured effect (eval-139 · 135 clips · 2,825 frames · `raw/map_cells_recovered.json`)

| | |
|---|---|
| supervised cells **before** | **19,647,460** |
| supervised **after** | **17,476,890** |
| **removed as unobserved** | **2,170,570 = 11.048 %** |
| per clip | min **6.100 %** · p50 **11.110 %** · p90 **12.121 %** · max **15.506 %** |
| lift-valid cells per clip | mean **89.003 %** of 7,680 |

⭐ **Measured THROUGH the fixed code's own reported counts**, and it reproduces the
reviewer's independently written P7 probe **to every digit** (`11.04758579…`, `19,647,460`,
and a bit-identical `lift_valid` mean of `0.8900250771604938`). Agreement across two authors
and two code paths is a cross-check; re-running P7's own derivation would have measured
determinism.

**Controls (all four pass):** **analytic** — zero lift-valid cells outside 60°, over every
clip; **discriminating, both ends** — an all-TRUE mask removes exactly **0** and an all-FALSE
mask removes exactly **19,647,460**, so the measured value sits strictly between two arms
that could each have come out; **loss-side** — `n_map_cells_seen − n_map_cells ==
n_map_cells_unobserved` on every batch, from the row's own fields.

⚠️ **11.048 % is a LOWER bound** on non-instantaneously-observable supervision. It prices
geometry only — azimuth, elevation, range. A cell 55 m ahead behind a truck is lift-valid
and still unobserved, and nothing here can see that without the images.
⚠️ **The lift's `valid` is a projection through the frame**, so this figure must be
**recomputed at 416×1024** before it is quoted there. The `seen`/label side is in metres and
carries across; the mask side does not.

---

## 4. D-4 — the guard can now go red, and the vector follows the filter

### 4a. The guard: two independent sources

`CLS_WEIGHT_CHOICES` selected the artifact **and** the expectation from the same key, so the
arm's actual corpus was never an input. MEASURED by the review (`raw/p5_cls_weight_guard.json`):
`guard_can_go_red: true`, **`guard_blocks_the_operator_error: false`**.

**Fix:** `agent_slots.corpus_line_for_join(join_path)` derives the line from the **arm's own
join file**, via a declared `JOIN_CORPUS_LINES` table. It returns **`None`** for a join it
does not name — ⛔ **never a guess** — and `strict=True` refuses. The trainer patch then
requires the derived line and the key's declared line to **agree**.

**Proof:** the arm that P5 measured as `LOADED` is rerun — a B1 arm
(`--agent-join …/b1_train_plus_eval_agents.jsonl.xz`) asking for `train2400` — and now
raises `SystemExit`. ⭐ With a **same-breath control**: the correct pairing must still load,
or the "refusal" is only a broken loader. The mirror-image error on a parity arm is tested too.

**Mutation `M6`:** `base = Path(join_path).name` → a constant that matches nothing, i.e. no
derivation from the arm — the historical state exactly. **CAUGHT.**

### 4b. The vector: fixing D-1 invalidates it, so both landed together

The banked B1 vector is inverse-frequency over the **raw** join. That is correct only while
the loss sees every box. **Fix:** the artifact now carries **both** populations, selected at
load time by the arm's actual filter state:

```
load_cls_class_weight(name, expect_corpus_line=…, target_population=…)
   raw_join                → weights_inv_freq_mean1          (DEFAULT, unmoved)
   in_field_and_decode_box → weights_inv_freq_mean1_visible  (NEW)
```

⛔ **One artifact, two vectors — deliberately.** Two *files* selected by one flag would
reproduce the D-4 defect one level up. An artifact that carries no vector for the requested
population is **REFUSED**, never served the other one.

**Rebuilt with a real producer:** `stack/scripts/build_cls_weight_artifact.py`. ⛔ The
previous artifact stamped `_producer: "qland/work/pbox/build_cls_weight_artifact.py"` — a
scratch path that **no longer exists on this box** (probed: `qland/` is absent from the repo
and the working tree). A banked artifact whose producer is gone can only be hand-edited,
which is `RETR-2026-09-22-SELF-ATTESTING-DIGEST` waiting to happen again. The producer is
now in the repo, and **no code path in it writes a digest by hand** —
`agent_slots.cls_weight_digest` is the one recipe and `load_cls_class_weight` re-derives it
on every load.

⭐ **The producer run reproduced the banked RAW vector EXACTLY** — digest
**`c3937558f59299e7`**, `_control_reconstruction_max_abs_dev` **4.96 × 10⁻⁷**, identical to
the reviewer's independently computed control. That is what makes the visible column
like-for-like rather than a different normalisation. Per-class counts reproduce P6's
`post_filter_counts` **class for class**.

### The measured per-class shift (v7-B1, 28,958,699 boxes, 64.5 s)

| class | raw n | visible n | w raw | w visible | ratio |
|---|---|---|---|---|---|
| **other_vehicle** | 78,231 | 8,988 | 0.877236 | 1.531726 | **1.7461** |
| **heavy_truck** | 553,977 | 64,278 | 0.123881 | 0.214181 | **1.7289** |
| bus | 205,566 | 28,692 | 0.333845 | 0.479826 | 1.4373 |
| trailer | 282,911 | 40,451 | 0.242575 | 0.340342 | 1.4030 |
| automobile | 21,515,941 | 3,219,256 | 0.003190 | 0.004277 | 1.3408 |
| animal | 11,785 | 2,479 | 5.823257 | 5.553512 | 0.9537 |
| person | 5,487,965 | 1,193,623 | 0.012505 | 0.011534 | 0.9224 |
| rider | 641,740 | 141,927 | 0.106939 | 0.097002 | 0.9071 |
| protruding_object | 114,855 | 26,366 | 0.597511 | 0.522156 | 0.8739 |
| **stroller** | 36,522 | 11,054 | 1.879062 | 1.245446 | **0.6628** |

**End to end 2.63×** (1.7461 / 0.6628). Imbalance **1825.7 : 1 → 1298.6 : 1**. Visible
in-vocabulary boxes **4,737,114** (+4,693 out-of-vocabulary `train_or_tram_car` = 4,741,807,
which is P2's in-field ∩ decode-box total exactly). New digest **`c9dbc340acf7d27e`**.

**Mutation `M7`:** the visible request silently resolves to the RAW key — the new defect the
review predicted, and it is **silent**. **CAUGHT.**

### ⚠️ Named blocker

`agent_cls_weights_train2400.json` (**parity**) carries **no** visible-population vector, so
a **parity** arm that turns the filter on is **REFUSED** rather than served the raw one. That
is declared in the file. Unblock: rerun the producer against `train2400_agents.jsonl.xz`
(~70 s CPU). Not done here because refcv6 trains the B1 line and banking a parity vector
nobody asked for is a second decision.

---

## 5. The cheap extra — `map_soft_ce` can now be class-weighted

MEASURED by the review on **eval-139**: label mass on seen cells is `sidewalk/verge`
**37.14 %**, `drivable` **34.00 %**, `seen-no-map-class` **25.90 %** ⇒ **97.05 % in three of
nine classes**, against `arrow/text` **0.067 %** and `hatched area` **0.063 %**.

`map_soft_ce(..., class_weight=None)` — **default off, bit-identical**. The weighted form is
`Σ_cells Σ_c w_c p_c (−log q_c) / Σ_cells Σ_c w_c p_c`, i.e. **the denominator follows the
weights**, exactly as `slot_set_loss`'s `cls` term does. ⭐ At `w = ones` the weight mass
**is** the seen-cell count, so a uniform vector is **bit-identical** to passing nothing —
the identity control, pinned to `abs=1e-12`.

Proven against a **hand-computed** weighted cross-entropy: two cells, three classes, every
number recombined in plain Python — never by calling the function again.

**Mutation `M8`** breaks the denominator rule. ⭐ **The identity control cannot see it**
(both denominators equal `n` at `w = ones`) — which is precisely why the hand-computed
reference exists. **CAUGHT** by that test, with the identity control correctly staying green.

⛔ **NO VECTOR IS BANKED, and that is the honest answer, not an omission.** The only census
that exists is over **eval-139**; shipping it for a **v7-B1 train** arm would be the scope
error this entire package is about. **Blocker:** a map census on the v7-B1 train release
(the SPEC dates its production to ~22 Sep). The mechanism is in place and costs nothing
until a vector exists.

---

## 6. The `refc_v3_train.py` patch — NOT APPLIED BY ME

⛔ I do not own `stack/scripts/refc_v3_train.py` and did not touch it.
**`code/refc_v3_train.PATCH.md` carries six hunks as exact BEFORE/AFTER whole-line blocks.**

⚠️ **The D-1/D-2 fix is ALREADY LIVE without the patch**, because `box3d_loss_row` now
defaults to `visible_filter=True` and the trainer calls it with no keyword. The patch gives
the operator a *name* for the choice (and puts it in argv, hence in `config.json`).
⚠️ **The D-3 fix has NO caller without the patch** — `map_loss_row`'s `lift_valid` defaults
to `None`.
⛔⛔ **A SOURCE-ADJACENCY BUDGET BINDS HUNKS 1 AND 2, AND MY FIRST DRAFT BROKE IT.**
`test_map_iou_drivable.py:127-131` asserts a **character** distance in the trainer's source:
`SRC.index('extra["map_iou_drivable"]') − SRC.index('extra["n_map_cells"] = …') < 2500`, so
the IoU is computed in the same block as the loss from the same tensors. **MEASURED: current
`j − i` = 1,877, headroom 623**; my first draft spent ≈ 685 and would have reddened a test
that has nothing to do with these fixes. The patch is rewritten to put Hunk 1's additions
**before** the marker (which costs zero headroom, since `i` and `j` shift together) and to
keep Hunk 2's comment to one line. ⭐ **Verified by applying hunks 1–2 to a scratch COPY**:
patched `j − i` = **2,070**, headroom **430**, `ast.parse` clean, and the real trainer's
`sha256[:16]` **`40ed2ee52601efa8` before and after** — untouched.

⛔⛔ **Hunks 4 + 5 + 6 must land together**, or a filtered arm can train on raw-join
frequencies. With none of them applied that combination is at least **declared** (the stamp
says `raw_join` while `_visible_filter` is True) rather than silent — but it is still wrong,
and the interim safe arm is `--agent-cls-weight off`.

The patch deliberately does **not** add a `CLS_WEIGHT_CHOICES` key, so
`test_cls_weight_stamp.py::test_every_choice_maps_to_an_artifact_that_declares_the_expected_line`
stays green untouched.

---

## 7. Test evidence

```
stack/tests/test_refcv6_perception_supervision_fixes.py     23 passed   (the new proofs)
mutate_perception_fixes.py                                   8 arms, 8 CAUGHT, 0 escaped
                                                             0 overbroad, 0 invalid
                                                             baseline green BEFORE and
                                                             AFTER restore
```

Two existing tests were edited, both to **state a scope** rather than to make a new default
pass — each now says `visible_filter=False` where it supplies its own `match`, which the
loss refuses to pair with the filter:

* `test_refcv6_perception.py::test_no_zh_labels_is_the_2d_loss_exactly`
* `test_refcv6_perception_realdata.py::test_the_3d_join_reaches_the_height_targets`

⚠️ `agent_cls_weights_b1.json` was **rebuilt**, not hand-edited. Its RAW digest is unchanged
(`c3937558f59299e7`), so `test_cls_weight_stamp.py` passes untouched.

### Regression run

⭐ **Every test file that imports any of the five changed modules** — derived by grep, **34
files** — run at the **final** code state from the canonical cwd (`stack/`, the `pyproject`
rootdir):

```
948 passed, 5 skipped, 1 warning in 226.64s
```

The 5 skips are all env-gated and pre-existing (`$TANITAD_AGENT_JOIN3D` ×2,
`TANITAD_JOIN3D`/`TANITAD_JOIN2D` ×3) — the skip *reasons* were read, not just the count.

### ⭐ A suite-wide blocker found and SOLVED while verifying — `taniteval` needs its own path entry

⛔ **`pytest -q` cannot be green from EITHER cwd today, and the cause is the documented
`taniteval` namespace shadow.** MEASURED, both directions:

| cwd | `tests/test_b1_agent_join_3d.py` | `tests/test_refcv6_tactical.py` |
|---|---|---|
| repo root | **13 failed** (`ImportError: cannot import name 'lead_source'`) | passes |
| `stack/` | 25 passed, 3 skipped | **6 failed** (`ModuleNotFoundError: No module named 'taniteval'`) |

The real package is **`taniteval/taniteval/`** (it holds `lead_source.py`, `ci.py`). The
OUTER `taniteval/` has **no `__init__.py`**, so from the repo root `import taniteval`
resolves to `_NamespacePath(['D:\\Projects\\TanitAD\\taniteval'])` — the wrong directory,
which has no `lead_source` — and from `stack/` it does not resolve at all.

⭐ **THE FIX IS ONE PATH ENTRY, and it is MEASURED, not proposed:**
```
PYTHONPATH="D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval"
```
resolves `taniteval -> D:\Projects\TanitAD\taniteval\taniteval\__init__.py`, and the two
files above then run together **118 passed, 3 skipped, 0 failed**. ⇒ The project's standing
instruction `PYTHONPATH=D:/Projects/TanitAD/stack` is **incomplete for the test suite**.

⚠️⚠️ **BUT DO NOT ADOPT IT GLOBALLY ON MY SAY-SO — IT IS NOT A FREE CHANGE, AND I DID NOT
FINISH MEASURING IT.** A full-suite run under the corrected path was **still in flight at
hand-off** and had already produced **`EEEE`/`F` at 18 %** — runtime errors the incomplete
path does *not* produce. Collection is identical either way (**501 files** both), so the
difference is runtime, not discovery; a plausible mechanism is that `taniteval/` carries its
own `conftest.py` and a `tests/` package that can collide once that directory is importable.
⇒ **MEASURED: the entry fixes `test_b1_agent_join_3d.py` + `test_refcv6_tactical.py`.
UNMEASURED: its net effect on the other 499 files.** Finish that run before changing any
brief, cron or runbook. The claim that stands unconditionally is the diagnosis — *the outer
`taniteval/` shadows the real `taniteval/taniteval/`, and `pytest -q` is green from neither
cwd today* — not the remedy.

⛔ **Not mine** — two probes: none of my five changed modules or my test file mentions
`taniteval` at all (control: `torch` reads 20 in `box3d_head.py`, so the files were read),
and the failing import lives in files I did not touch.

### The full `stack/` suite, and the attribution of every failure

```
8,686 passed · 15 failed · 55 skipped · 2 xfailed   in 1889 s (0:31:29)
```

⛔ **Every one of the 15 is accounted for, and 14 are not mine:**

| n | failure | cause | mine? |
|---|---|---|---|
| **1** | `test_refcv6_perception_supervision_fixes.py::test_D1_THE_DEFAULT_IS_FILTERED…` — `KeyError: 'box3d_visible_filter'` | ⚠️ **MY ARTIFACT, AND A REAL LESSON: I EDITED SOURCE DURING A 31-MINUTE RUN.** The run had imported `refcv6_perception_branch` *before* I added `row["box3d_visible_filter"]`, then executed the *newer* test body. **Green in isolation, in the 34-file set, and in the mutation baseline, all at the final state.** | yes — a stale-import artifact, **not** a defect |
| **11** | 9 × `test_built_heads_receive_gradient.py` + 3 × `test_route_loss_respects_strategic_bypass.py` — `ValueError: refcv5 WP-4 / refcv6 F9: the anchor vocabulary DECLARES v0_conditioned=True but anchor_controls is all zeros` | a **sibling's UNSTAGED edit**: that refusal string appears **0 times in `HEAD:stack/tanitad/models/refcv6_diffusion.py` and 1 time in the worktree** | no |
| **2** | `test_tac_goal_trainer_flag.py` ×2 — `AttributeError: 'DecoderConfig' object has no attribute 'bev_coupling_provenance'` | the same stream: `bev_coupling_provenance` lives in `refc.py` / `refc_v3_train.py`, **both modified in the worktree**; **0 occurrences in any file I touched** | no |

⭐ **The discriminating control for all 14:** none of
`test_built_heads_receive_gradient.py`, `test_route_loss_respects_strategic_bypass.py` or
`test_tac_goal_trainer_flag.py` mentions `w_box3d` or `w_map` **even once**, so none of them
constructs a perception branch and none can reach `box3d_loss_row` or `map_loss_row`. My
changes are not on their path.

⚠️ **Re-run with the corrected `PYTHONPATH` before quoting a suite verdict** — the 31-minute
run above used the incomplete one, so its 55 skips and its `test_refcv6_tactical` /
`test_b1_agent_join_3d` results are cwd artifacts, not facts about the code.

---

## 8. Behaviour-change summary — the binding question, answered per fix

| fix | opt-in or behaviour change? | what a reader must know |
|---|---|---|
| **D-1 + D-2** (box visibility filter) | ⚠️ **BEHAVIOUR CHANGE, default ON** | Any arm trained after this lands supervises a different target set: **82.94 % fewer query slots**, all of them previously spent on boxes the camera cannot see. The regression arm is `visible_filter=False` / `--no-box3d-visible-filter`. Reported here rather than hidden behind an opt-in because the v6 seam has always filtered and a monocular head trained on `cx < 0` boxes is not a defensible default. **See the banked-arm note directly below — my first draft said "no such arm exists" and that was WRONG.** |
| **D-3** (map lift mask) | ⚠️ **BEHAVIOUR CHANGE ONLY WHEN THE CALLER PASSES IT** | Module default `lift_valid=None` = today. The trainer patch turns it on by default; `--no-map-lift-valid-mask` is the regression arm. ⛔ The **0.3412 no-information drivable-IoU floor** was measured on the UNMASKED cell set — requote it against the masked set before using it as a gate on a masked arm. |
| **D-4** (guard + vector) | ✅ **OPT-IN, and bit-identical by default** | `target_population` defaults to `raw_join`. Both banked digests are unmoved. The stricter corpus-line check only fires when the trainer patch supplies a derived line. |
| **map class weight** | ✅ **no change** | `class_weight=None` is the default and is bit-identical. |
| `box3d_set_loss` return | ⚠️ **additive keys** | `n` gains `target_prefilter` / `target_visible` / `dropped_not_visible`; the row gains `box3d_visible_filter` (1.0/0.0) and, for the map, `n_map_cells_seen` / `n_map_cells_unobserved`. New keys in `metrics.jsonl`; no existing key changes meaning. |
| `PerceptionBranch.forward` | ⚠️ **additive key** | `out["map_valid"]`. Zero parameters, no `state_dict` change, no RNG draw. |

### ⛔ WHICH BANKED ARMS THIS TOUCHES — and a self-correction

My first draft of this table asserted *"no banked refcv6 box-head result is invalidated —
there is none"*, from **one probe** (`grep -c "w_box3d" MODEL_REGISTRY.md` → **0**, with a
same-breath control reading 158 for `refcv`). ⛔ **A second probe refuted it**, which is why
the rule exists: scanning every repo `config.json` for `w_box3d > 0` finds **two**:

| run record | steps | `n_target` (banked) | `n_map_cells` (banked) |
|---|---|---|---|
| `…/2026-09-17-refcv6-bev-tactical/raw/live_run` | **10** | 101 · 94 · 53 | 30,625 · **30,720** · 29,733 |
| `…/2026-09-17-refcv6-perception-training/raw/live_run` | **3** | 56 · 64 · 35 | 12,119 · **15,360** · 15,360 |

⭐ **Both are wiring/smoke runs** — 10 and 3 steps — and neither reports `box3d_ap`, a map
IoU, or any capability number. **So no capability claim is invalidated.** But their
`metrics.jsonl` rows **are no longer comparable with a post-fix rerun**: `box3d_n_target`
will fall to the visible subset, and `n_map_cells` — which reads **30,720 = 4 × 7,680
exactly**, i.e. *every* cell of a 4-window batch labelled `seen`, including the 590 the rig
can never see — will fall by roughly 11 %. ⇒ **Rerun those two wiring runs rather than
diffing against them.** Their `--agent-join` is `b1eval_agents_3d.jsonl.xz`, which
`JOIN_CORPUS_LINES` names, so the D-4 derivation resolves for them with no operator input.

⚠️ The registry grep returning 0 is itself correct and is its own small finding: **two
refcv6 perception runs exist with no registry row.**

---

## 9. What I did NOT fix, and why

1. **D-5 — the 3-D join is eval-only.** The `n_z == 0` on the 4,427-clip train split is a
   **missing label**, not a code defect; the builder is parameterised and priced at ≈42 min
   CPU. It writes a multi-hundred-MB corpus artifact, which is the DataFlyWheel's lane.
   **Unblocked, named, not mine.**
2. **The eval z/h proof still does not run by default** (`$TANITAD_AGENT_JOIN3D` gates it).
   Pointing CI at the banked join is a CI change, not a module one.
3. **F8 skip bias** (147 clips = 2.89 %) — untested correlation; needs a sha12-keyed join to
   `clip_index.parquet`.
4. **Occlusion.** Every "visible" number here, mine and the artifacts', is **azimuth (+
   range + elevation) only**. No inter-agent or hood occlusion is priced anywhere, so
   **40.20 %** and **11.048 %** are both **upper bounds on what the camera actually sees**.
5. **Whether any of this moves a metric.** ⛔ **Unmeasured, and it must be said plainly.**
   These are fixes to the *supervision*; no arm was trained and no `box3d_ap` or map IoU was
   measured. The next lever is a v7-tiny A/B — filtered vs unfiltered, same seed, plus a
   **replicate arm**, because a separated CI on a one-seed pair is necessary and not
   sufficient (`H-ESTIM-SEED-1`). That needs GPU the PI has not authorised for this stream.

---

## 10. DELIVERABLE MANIFEST

All paths relative to `D:/Projects/TanitAD/`. **Staged, never committed, never pushed.**

### Code — fixes (repo, `stack/`)

| path | change |
|---|---|
| `stack/tanitad/models/box3d_head.py` | `box3d_set_loss(visible_filter=True, visible_ranges=, visible_half_angle_rad=)`; filter applied **before** `match_slots`; `zh_mask` follows; `match=` + filter REFUSED; `n` gains three counts; `_visible_filter` in the return |
| `stack/tanitad/models/refcv6_perception_branch.py` | new `map_valid_from_lift`; `map_loss_row(lift_valid=)` + two new counts + masked `map_metrics`; `box3d_loss_row(visible_filter=True)` + `box3d_visible_filter` in the row; `forward` emits `out["map_valid"]` |
| `stack/tanitad/models/agent_slots.py` | `TARGET_POPULATION_RAW/VISIBLE`, `_CLS_WEIGHT_KEYS_*`, `JOIN_CORPUS_LINES`, `corpus_line_for_join`; `load_cls_class_weight(target_population=)` + population in the stamp |
| `stack/tanitad/models/bev_encoder.py` | `map_soft_ce(class_weight=None)` with a weight-following denominator |
| `stack/tanitad/refs/refc_agents.py` | **docstring only** — the parity/v7-B1 citation split, with both lines' figures named (the 41.06 %/15.67 % pair is PARITY's) |
| `stack/tanitad/data/agent_cls_weights_b1.json` | **REBUILT by the producer**; RAW vector + digest unmoved, `target_population`, `counts_visible`, `weights_inv_freq_mean1_visible`, `_self_digest_sha256_of_weights_visible`, `ratio_visible_over_raw`, in-repo `_producer` |
| `stack/tanitad/data/agent_cls_weights_train2400.json` | `target_population: "raw_join"` declared + the named blocker note. Weights and digest unmoved |
| `stack/scripts/build_cls_weight_artifact.py` | **NEW** — the producer the old artifact's `_producer` field pointed at a now-deleted scratch path for |

### Tests (repo, `stack/tests/`)

| path | change |
|---|---|
| `stack/tests/test_refcv6_perception_supervision_fixes.py` | **NEW**, 23 tests, all literal/analytic |
| `stack/tests/test_refcv6_perception.py` | one test now names `visible_filter=False` beside its supplied `match` |
| `stack/tests/test_refcv6_perception_realdata.py` | same, in `test_the_3d_join_reaches_the_height_targets` |
| `stack/tests/test_cls_weight_stamp.py` · `test_cls_class_weight.py` · `test_occ_from_geometry.py` · `test_occ_knob_is_stamped.py` | **UNCHANGED BY ME — staged because they were NOT IN HEAD** (see the git note below). They pin `agent_slots.py`, which I own; leaving them stranded beside a staged module is a half-landed feature |

### Research package (repo)

`TanitAD Research Lab/Architecture & Inference/Research/2026-09-23-refcv6-fixes/`

| path | what it is |
|---|---|
| `PERCEPTION_FIXES.md` | this document |
| `code/refc_v3_train.PATCH.md` | the six-hunk trainer patch, exact BEFORE/AFTER |
| `code/mutate_perception_fixes.py` | the 8-arm deliberate-regression harness |
| `code/eval_box_supervision_recovery.py` | D-1/D-2 effect over the whole v7-B1 join |
| `code/eval_map_cells_recovered.py` | D-3 effect through `map_loss_row`'s own counts |
| `raw/mutation_perception_fixes.json` | MEASURED — 8/8 CAUGHT, 70 s |
| `raw/box_supervision_recovery.json` | MEASURED — 875,657 frames / 28,958,699 boxes |
| `raw/map_cells_recovered.json` | MEASURED — 135 clips / 2,825 frames, 9.5 s |

⚠️ **This package directory is SHARED with a sibling stream.** `code/eval_fixes.py`,
`code/mutation_harness.py`, `raw/eval_fixes.json`, `raw/mutation_results.json` and
`raw/_prefix_tmp/` are **theirs**; I neither read, wrote nor overwrote them. My files are the
eight listed above.

### ⚠️ Two git facts found while staging — both are findings, not housekeeping

1. ⛔⛔ **THE WHOLE H-BOXCLS-1 CLASS-WEIGHT FEATURE WAS STRANDED — code, artifacts AND
   tests. MEASURED by positive assertion against `HEAD`** (`git cat-file -e` /
   `git show HEAD:<path> | grep -c`), with a same-breath control:

   | token in `agent_slots.py` | HEAD | worktree |
   |---|---|---|
   | `load_cls_class_weight` | **0** | 1 |
   | `cls_weight_digest` | **0** | 4 |
   | `CLS_WEIGHTS_B1` | **0** | 1 |
   | `CORPUS_LINE_B1` | **0** | 6 |
   | `OCC_HALF_ANGLE_RAD` | **0** | 3 |
   | `occ_logit_from_centre` | **0** | 5 |
   | *control* `match_slots` | **4** | 4 |

   …and **NOT-IN-HEAD**: `agent_cls_weights_b1.json`, `agent_cls_weights_train2400.json`
   (both untracked, `git check-ignore` exit 1 — not ignored), `test_cls_weight_stamp.py`,
   `test_cls_class_weight.py`, `test_occ_from_geometry.py`, `test_occ_knob_is_stamped.py`.

   ⇒ **The vectors a refcv6 arm loads, the loader that verifies their digest, the
   `occ_from_geometry` identity, and all four of their test files existed on ONE DISK.**
   Staging `agent_slots.py` carried the module half; I staged the four test files and both
   artifacts explicitly so the feature is not left half-landed. **Reported, not unstaged.**
   *(The "finish before you start" rule, and the largest instance of it I have seen: 334
   inserted lines in one module of which only ~120 are mine.)*
2. ⛔ **`box3d_head.py`'s INDEX blob was STALE — the 2026-09-07 trap, live.** Three blobs,
   all 40 chars asserted first: `HEAD d1da1b37` (497 lines), **`index 3ece522a` (528)**,
   worktree (530 pre-edit). The index carried a sibling's `box3d_match_rows` refactor but
   **not** the two-line `cls_class_weight` addition that was in the worktree — `grep -c
   cls_class_weight` reads **0 / 0 / 2** across HEAD / index / pre-edit worktree.
   ⇒ Staging my worktree version also carries that sibling's 2 stranded lines.
   **Reported, not unstaged** (the standing rule). Descendant check run before staging: all
   **9** lines that differ from HEAD are lines legitimately *replaced* (the signature, a
   continuation, the `out["n"]` line I changed, and the AP refactor); **zero** unexplained
   removals.

### Read-only inputs, not modified, not copied into the repo

`D:/Projects/TanitAD-artifacts/a40-rescue/b1_train_plus_eval_agents.jsonl.xz` ·
`D:/Projects/TanitAD-artifacts/sam3-maps-eval/` (135 `*.sam3mapgt.npz`) ·
`D:/Projects/TanitAD-artifacts/refcv5v2_final/extrinsics141.json` ·
`…/2026-09-22-refcv6-review/raw/p2,p5,p6,p7*.json` (read for cross-checks only).

### Escalations — these need a decision, not a doc

1. **Apply the trainer patch**, hunks 4+5+6 together (§6). Until then the D-3 fix has no
   caller and the D-4 population can only default.
2. **The default is now FILTERED.** If a refcv6 box arm is already queued, it will train a
   different target set than its SPEC assumed. Confirm that is intended, or launch it with
   `--no-box3d-visible-filter` **and** `--agent-cls-weight off`.
3. **The 0.3412 drivable-IoU floor must be requoted** on the masked cell set before it gates
   a masked arm.
4. **A parity-line visible vector** is ~70 s of CPU and is currently a refusal (§4).
5. **No metric was moved.** The A/B that would price these fixes needs GPU and a replicate
   arm (§9.5).
6. ⚠️ **A latent coupling I left documented rather than mechanised.** The filter's range cut
   is `SlotDecodeRanges()` — the `GRID_DEFAULT` box — while the decoder's own decode ranges
   are a *constructor argument*. Today they cannot disagree, because
   `PerceptionBranch.__init__` builds `Box3DSlotDecoder` **without** `ranges=`, so both take
   the same default; and `box3d_set_loss(visible_ranges=…)` exists for the day one of them
   moves. ⛔ But nothing *asserts* the two are equal, and a decoder built with custom ranges
   would be filtered against a box it does not decode — a scope error of exactly the family
   this package is about. The durable fix is for the loss to read the ranges off the
   decoder; that needs a signature the trainer also passes, so it is named here rather than
   half-done.
7. **Two refcv6 perception runs have no `MODEL_REGISTRY.md` row** (§8). They are 10-step and
   3-step wiring runs, so no result is at stake — but the registry is the only quotable
   source for a model fact, and an arm that ran and is not in it is a hole in that guarantee.
8. ⭐ **`pytest -q` IS GREEN FROM NEITHER CWD TODAY**, and the programme's rule *"`pytest -q`
   must stay green before any commit"* is therefore **unsatisfiable as written** (§7). The
   diagnosis is certain: the outer `taniteval/` (no `__init__.py`) shadows the real
   `taniteval/taniteval/`. ⚠️ The remedy — adding `;D:/Projects/TanitAD/taniteval` to
   `PYTHONPATH` — is **measured on two files and UNMEASURED on the other 499**, and a
   full-suite run under it was still in flight at hand-off showing new runtime errors.
   **Finish that run before changing any brief, cron or runbook.**
9. **Rerun those two wiring runs after the trainer patch** rather than diffing against their
   banked `box3d_n_target` / `n_map_cells`, which the fixes move by construction.

<!-- REFCV6-PERCEPTION-FIXES-2026-09-23 -->
