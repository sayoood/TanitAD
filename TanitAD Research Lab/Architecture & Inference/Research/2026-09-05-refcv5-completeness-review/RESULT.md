# refcv5 completeness review — the readiness claims hold; the GUARD THAT PROTECTS THEM DOES NOT

**Stream:** Architecture & Inference (independent review half) · **2026-09-06** · branch `agent/arch-inf-20260803`
**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refcv5-completeness-review/`
**Role:** ⛔ ADVERSARIAL. Every claim below was re-measured by me from the artifact, never inherited.
**Compute:** CPU-only, dev box. No GPU slot taken (Thor: 3 live eval arms; 4060 saturated).
**Environment:** an ISOLATED mirror `C:/Users/Admin/tanitad-review-20260906/` — deliberately *not*
the shared `tanitad-wt`, which is resynced mid-run and would have invalidated these measurements.

---

## The one line

⭐ **Every readiness claim I was asked to attack SURVIVED re-measurement.** The parity labels really
are 100.000 %, the anchors really do declare `alat`, the geometry really is bit-identical, and 2,400
vs 2,376 is **not** a parity violation. ⛔ **But the mechanism built to stop a family-shaped hole from
being published is BLIND to the defect its own writer emits** — proven by mutation, with both
controls — and the criteria census shows the consequence: the STRATEGIC nav-compliance family is
**present in 0 of 39 in-scope artifacts**.

⇒ **If refcv5 started training tomorrow morning it would train a model with no sampler and no agent
head, stamp `config.json` with seven capabilities it does not have, and be evaluated by a panel that
cannot fail on the one family it was built to protect.**

---

## 1. The readiness claims, each RE-MEASURED

| # | claim (INHERITED) | my measurement | verdict |
|---|---|---|---|
| 1 | parity labels **7.917 % → 100.000 %** | trainer's own arithmetic (`hit/len(eps)` over `stable_episode_id`): **2,400/2,400 = 100.000 %**, md5 `0d45b8d1344aa669c9d39e1ade2832fe` | ✅ **PASS** |
| 2 | parity is sacred — **2,400 vs 2,376** | reconciled from `parity_manifest.json` via `parity.manifest_entry()`; **not a violation**; see §2 | ✅ **PASS (reconciled)** |
| 3 | anchors: units `alat` declared | `control_units='alat'`, `controls_columns=['a_lon_ms2','a_lat_ms2']` **in the file**; Kamm control reproduces **0.306 g / 0-of-117** vs the misreading's **396.3 g / 104-of-117** | ✅ **PASS** |
| 4 | geometry **max abs diff = 0.0** | vs BOTH live copies (md5 `297f6f1db52f6a56094846b0d7f71ed9`): `anchors` and `controls` **max abs delta = 0.0**; negative control (1e-6 perturbation) reads `9.54e-07` ⇒ non-vacuous | ✅ **PASS** |
| 5 | a units-stripped file is **REFUSED** | deliberate regression → **`AnchorUnitsMissing`**; converse control: untouched file **not** refused | ✅ **PASS (guard proven able to fail)** |
| 6 | 85 GB cache on Thor, digest MATCH | ⛔ **not re-verifiable from here** — Thor holds 3 live eval arms and I took no slot. The *digest* half I did verify locally: the parity clip list recomputes to `e61a04553df5b9d5…` = the committed manifest | ⚠️ **INCONCLUSIVE (Thor half)** |
| 7 | preflight **14 PASS / 1 FAIL / 1 INCONCLUSIVE** | my run: **11 PASS / 1 FAIL / 4 INCONCLUSIVE** — the 3 extra are inputs I lack (`--v2-cache`, `--agent-join`, `--smoke`), not disagreements. **The FAIL is the same one** | ✅ **PASS (consistent)** |
| 8 | the FAIL and the INCONCLUSIVE, named | **FAIL** = `every weighted term has a gradient`: `agent_w_ground` grad **1.164e-10** DEAD vs `agent_w_project` **1.462e-03** LIVE. **INCONCLUSIVE** = `a 2-step run states every weight` (`--smoke` not run) | ✅ **named** |
| 9 | strategic band **2,203/2,400 = 91.79 %**, perception absent on 2,199 | sidecar confirms; ⚠️ **but the absence encoding defeats the sidecar's own instruction** — §3.7 | ⚠️ **PARTIAL** |
| 10 | "does any consumer parse the absent layer?" — claim: no | **TRUE today**: `V7Label`'s parsed surface carries none of the perception fields. ⚠️ but see §3.7 — the encoding will mislead the first consumer that does | ✅ **PASS, with a work item** |
| 11 | turn-suppression gap **~1.27 % (ESTIMATED)** | arithmetic reproduces; **the denominator is wrong**: 30/2,400 = 1.25 % of the CORPUS but **30/293 = 10.24 % of the TURN class** — **8.2x** | ⛔ **FAIL as stated** |

---

## 2. ⭐ The 2,400-vs-2,376 reconciliation — NOT a violation, and here is the arithmetic

**MEASURED** from `stack/tanitad/data/parity_manifest.json`:

```
discovered 3000  -  val 600   =  2400 train CLIPS
2400 clips       -  24 skips  =  2376 raw EPISODES   <- the CLAUDE.md invariant number
the V2/w120 cache built ALL 2400 clips (skip_count 0) =  2400 v2 EPISODES
```

Two **registered** corpora, not one re-selection:

| corpus key | `episode_count` | `skip_count` | what it is |
|---|---|---|---|
| `physicalai-train-e438721ae894` | **2376** | 24 | the RAW epcache — `parity.PARITY_TRAIN_EPISODES` |
| `physicalai-train-e438721ae894-w120-256x640cyl` | **2400** | 0 | the V2 cache refcv5 trains on |

`clip_id_sha256_ordered == clip_id_sha256_sorted`, so skip index *i* is sorted position *i*. Naming
the 24 clips positively rather than asserting an absence: `c0af6273-…`, `c4803512-…`, `c50711e4-…`,
`c512cadb-…`, `c5168845-…` (+19) — and **24/24 are labelled by the new parity blob**.

⇒ ✅ **No escalation. Nothing re-selects episodes** — the v2 corpus is separately registered with its
own membership proof and its digest MATCHES the committed one (I recomputed it: `e61a04553df5b9d5…`).

⚠️ **One statement is owed in every cross-arm table:** refcv5 trains **2,400** episodes; every
raw-epcache arm trained **2,376**. The delta is **24 episodes = 1.00 %**, and those 24 are in
refcv5's training set and in no raw-epcache arm's.

⭐ `refcv5_preflight.py` already prints this reconciliation as its own control — *"the manifest names
2376 episodes after the 24-clip skip"*. The **Architecture `RESULT.md` never names 2,376 at all**;
the DE sidecar does name it, but says *"the trainer corpus is 2,376 EPISODES"*, which is **wrong for
refcv5** — its trainer corpus is the 2,400-episode v2 cache. *(Class: true-but-wrong-for-the-reader.)*

---

## 3. FINDINGS, most severe first

### 3.1 ⛔⛔ The STRATEGIC defect gate is BLIND to the defect its own writer emits — PROVEN BY MUTATION

`refcv3_arm.py` classifies a `TypeError` from our own module as a **DEFECT** (not a refusal) and
collects it *"so a driver can exit non-zero instead of publishing a family-shaped hole"*. It writes:

```python
# taniteval/tools/refcv3_arm.py:2633
ref.setdefault("_defects", []).append({...})     # `ref` becomes rec["refcv3"]  (:2445, :2712)
```

The driver reads:

```python
# taniteval/tools/run_hierarchy_panel.py:71
defects = rec.get("_defects") or []              # TOP LEVEL
```

**Mutation proof** (`raw/prove_defect_blind.out`), three arms, both controls behaving as required:

| arm | record shape | `record_ok()` |
|---|---|---|
| **A** | the defect where the writer **actually puts it** (`rec["refcv3"]["_defects"]`) | ⛔ **`True` — "parses, no defects"** |
| **B** *(control)* | the same defect at `rec["_defects"]` | ✅ `False` — caught |
| **C** *(control)* | genuinely clean | ✅ `True` |

⇒ **The gate cannot fire on the defect the writer emits.** And it fails by the *same mechanism* as
the bug it was written to prevent — a path/scope mismatch, the sibling of `os.path.exists(<dict>)`.

⛔ **All three guards over it are blind, and two are string matches rather than behaviour:**

* `run_hierarchy_panel.record_ok` — wrong scope (above);
* `panel_preflight.check_defect_classifier` — asserts the literal `'ref.setdefault("_defects", [])' in src`, i.e. that the **writer's text exists**. It never checks that a reader reads that path;
* `stack/tests/test_navcomp_labels_shape.py::test_the_record_marks_the_block_and_collects_it_at_the_top_level` — **the test's NAME asserts "at the top level"; its assertion is the same source string, which puts it one level down.** The property is in the title and not in the test;
* `refcv3_arm.py` itself never exits non-zero on `_defects` (its only `sys.exit`s are argument checks).

⭐ **The measured consequence.** `tools/criteria_check.py --all taniteval/results/` (registry v2.7.0),
87 artifacts read → **39 in scope**:

```
nav-COMPLIANCE — the emitted BEHAVIOUR follows the route command   0 present,  1 refused, 38 MISSING
nav-compliance CONTROL: paired drop under nav-SHUFFLE              0 present,  1 refused, 38 MISSING
nav-compliance CONTROL: paired drop under nav-ZERO                 0 present,  1 refused, 38 MISSING
```

**The STRATEGIC behavioural family has never once been produced, and no gate ever stopped a panel
over it.** *(Corpus totals: 493 silent omissions, 61 work items, 7 UNSTAMPED tiers, 20 UNKNOWN scope.)*

**Cheapest fix (one line + one test):**

```python
defects = (rec.get("_defects") or []) + ((rec.get("refcv3") or {}).get("_defects") or [])
```

plus a **behavioural** regression test that plants the defect **at the writer's path** and asserts
`record_ok() is False` — replacing the two source-string assertions, which cannot distinguish a
fixed reader from a broken one.

### 3.2 ⛔ SEVEN phantom seam keys — the dead-flag defect is not one flag, it is a class

The sibling found `--sampler ddim` stamps a mechanism that does not exist. **I swept the whole
argv→`config.json` path**: `_seam_stamp` AST-walked, every `getattr(core.decoder, "<name>", …)`
checked against `DecoderConfig`'s real dataclass fields.

| stamped key | a real `DecoderConfig` field? |
|---|---|
| `feasible_decode` · `feasible_mu` · `feasible_entry` · `feasible_a_max` · `feasible_kappa_max` · `feasible_prefix_slots` | ✅ **REAL** (6) — Stage 0's wiring is genuine |
| `sampler` · `sampler_space` · `sampler_steps` · `sampler_infer_t` · `sampler_groups` · `cross_agent` · `control_norm` | ⛔ **PHANTOM** (7) — no such field |

`DecoderConfig` is **not frozen**, so `setattr(dc, "sampler", "STAMPED_BUT_UNREAD")` is *accepted* for
all seven: the trainer's pin creates an ad-hoc attribute, `_seam_stamp` reads it back, and
`config.json` records a **truthful-looking provenance line for a mechanism with no code behind it**.
Converse control: `feasible_decode` IS a declared field, so the sweep is not vacuous.

**Cheapest fix:** `@dataclass(frozen=True)` on `DecoderConfig` (or stamp only `dataclasses.fields()`),
so an unknown seam raises at pin time instead of being stamped. Ship it with the sweep as its test.

### 3.3 ⛔ WP-4 / WP-6 model-side absence — CONFIRMED independently, and NO LAUNCH GATE SEES IT

Re-measured, not inherited: `DecoderConfig(sampler='ddim')` → **`TypeError`**; `refc.py` read
(154,975 chars, **control: 54 `def `** — the file *was* read) contains `control_head` **0**,
`sampler` **0**, `cross_agent` **0**, `agent_head` **0**, `u0` **0**. My own pytest run:
**15 failed, 23 passed** across `test_refc_sampler.py` + `test_refc_v3_refcv5_wiring.py`
(`raw/pytest_refcv5_seams.out`).

⚠️ **My own two runs disagreed (14+1 skipped, then 15 twice) and I chased it rather than reporting
the convenient number.** The 15th is `test_pinned_against_diffusers_where_available`, which *skips*
when `diffusers` is absent and *failed* here because `diffusers` IS installed in this venv. It is a
finding in its own right — §3.3b — not a WP-4 failure, and not flakiness.

⛔ **The gap they did not close:** `refcv5_preflight.py` has **no check that the seam's module
exists**, and **nothing runs the model's own tests before a launch**. The 14 red tests *are* the
acceptance criteria and no gate consults them. A launch today passes every structural check.

**Cheapest fix:** two preflight rows — `--sampler ddim` ⇒ assert
`getattr(model.core.decoder, "control_head", None) is not None`; `--agents head` ⇒ assert
`model.core.agent_head is not None` — each with a converse control (seams off ⇒ absent, not refused).

### 3.3b ⛔ The ONE independent check on WP-4's schedule math can never pass — a tolerance below its own dtype

The sibling reports the schedule math as the sound half of WP-4: *"`refc_sampler.py`, 14 passing
tests … its load-bearing identity is pinned"*. **The pin that makes it "independent" has never
certified anything.** `refc_sampler.assert_matches_diffusers` exists precisely because *"a
re-implementation that is never checked is just an unverified copy"* — and MEASURED here:

| quantity | value |
|---|---|
| our `alphas_cumprod` dtype | **float64** |
| `diffusers` `alphas_cumprod` dtype | **float32** |
| the pin's measured disagreement | **1.689e-07** |
| the pin's declared `atol` | **1e-10** |
| float32 eps | **1.192e-07** ⇒ the error is **1.4x float32 eps** |

⇒ The comparison is a **float64 table against a float32 table at a tolerance ~1,000x below float32's
representable precision**. It therefore has exactly two outcomes and neither is a certification:
**SKIP** where `diffusers` is absent (which is how it has been read as "passing"), or **FAIL** where
it is present. ⭐ Same family as §3.1 — a control whose verdict is decided by its configuration
rather than by the thing it measures.

**Cheapest fix:** compare in a common dtype (cast diffusers' table to float64) and set `atol` to a
value reachable in the *lower* of the two precisions (~1e-6), documenting the dtype in the message.
Then a pass is evidence and a failure is a real disagreement.

### 3.4 ⛔ The turn-suppression bound is quoted against the wrong denominator — 8.2x

MEASURED on the blob: `TURN_L` **154** + `TURN_R` **139** = **293 turns of 2,400 = 12.21 %**.

| statement | value |
|---|---|
| as published | ~30 of 2,400 labels = **1.25 % of the CORPUS** |
| what it actually is | 30 of 293 turns = **10.24 % of the TURN CLASS** |
| ratio | **8.2x** |

The over-calls are *all* in the turn class by construction, so the corpus denominator is the wrong
one. ⚠️ **This matters precisely because turn recall has already collapsed to 0.0000 on another arm**:
a ~10 % label-noise rate concentrated in the minority class is a different object from "1.27 %".
**Cheapest fix:** publish both denominators and the turn family's `n` (293) beside them.

### 3.5 ⛔ "n = 2,400 instead of n = 190" is a CLIP count; the CE is per WINDOW — the real figure is 23.30 %

`v7_labels.window_in_band` admits a window only when `|t_now − t0| ≤ (hi−lo)/2`. With
`bands.tactical_s = [2.0, 6.0]` and `t0_s = 8.0` **on all 2,400 records** (one label per clip), the
admitted NOW is **[6.0, 10.0] s** of a ~20 s episode.

MEASURED on the **real** local parity v2 cache through the trainer's own dataset arithmetic
(controls: `window_in_band(t0=8.0)` **True**, `at t=0.0` **False**):

```
windows per episode 176.0     in-band per episode 41.0
TOTAL 16,893   SUPERVISED 3,936   =  23.30 %
=> 76.70 % of windows carry IGNORE_ID (-100) on tac_lat / tac_lon
```

⚠️ Scope, stated honestly: 96 of 2,400 clips are local (4.0 %). The figure is **structural, not
sampled** — every record carries the same `t0_s` and the same band, and episode length varies only
198–200 frames.

⭐ **The trainer already knows this distinction and applies it to the OTHER join.**
`enable_agent_join` computes *"NOW-frame coverage over THIS dataset's own window index … Clip
coverage is not supervision coverage"* (`refc_v3_train.py:885-888, 921-922`). The **v7 label join
computes only clip coverage** (`hit/len(eps)`) and floors on that.

✅ The `IGNORE_ID` design is **sound** — it never clamps an unlabelled window to a neutral class, and
the half-width is derived from the record's own bands rather than hardcoded (the `HORIZON` trap
handled correctly). The defect is only in **which number is published**.
**Cheapest fix:** compute the same NOW-frame coverage for `v7_by_sid`, stamp it, and quote *it* as
the supervision `n`.

### 3.6 ⛔ The eval-side label join has NO coverage floor

Exactly one floor exists in the trainer — `refc_v3_train.py:2632`, `if frac < 0.5: raise SystemExit`
— on the **train** side. The `--eval-labels` block (`:2694-2701`) builds `e_ds.v7_by_sid` and
**computes no coverage and applies no floor**. An eval-label blob covering ~0 % of the eval cache is
accepted silently, and the in-training TACTICAL/STRATEGIC families would then be computed over almost
no labelled windows while the run looked labelled. **This is the same hole the train-side floor exists
to close, on the half that produces the numbers.**
**Cheapest fix:** lift the identical four lines onto the eval side.

### 3.7 ⚠️ The perception layer encodes ABSENCE as a truthy dict of nulls — defeating its own sidecar

The sidecar instructs: *"A consumer MUST mask this layer rather than read its absence as a negative."*
MEASURED on the blob — the natural mask (`if rec.get(field)`) **cannot do that for two of three fields**:

| field | absent-record encoding | truthy on |
|---|---|---|
| `cot_tokens` | `None` | **201 / 2,400** ✅ honest |
| `cot_source` | `{"chain_of_causation": null, "components_analysis": null, …}` — a populated dict of nulls | ⛔ **2,400 / 2,400** |
| `alpamayo` | a populated dict (`{"anchor_offset_s": -2.9, "box…"}`) | ⛔ **2,400 / 2,400** |

Declared perception coverage is **201 / 2,400 (0.08375)**. ✅ **No current consumer is exposed** —
`V7Label`'s parsed surface carries none of these. But Stage 2/3 contemplates the layer, and the first
consumer to follow the sidecar's instruction literally will read 2,400 as annotated.
**Cheapest fix:** emit `cot_source: null` / `alpamayo: null` when absent, or add
`perception_present: bool`. *(Class: content-vs-existence.)*

### 3.8 ⚠️ `turn_suppression` is listed in a layer stamped `coverage: 1.0` but is present on 8 / 2,400

`meta.json` puts `turn_suppression` in `layers.kinematic.fields`, whose declared `coverage` is **1.0**
(`n_records: 2400`). MEASURED: the field is present on **8** records (0.33 %).
**Cheapest fix:** give it a per-field coverage, or move it out of the flat 100 % field list.

### 3.9 ⚠️ A live escalation in the Architecture `RESULT.md` is already CLOSED

Escalation #2 says `N_QUERIES_DEFAULT = 16` is *"REFUTED and still shipped"*, with a second spelling
`getattr(a, "n_slot_queries", 16)` at `train_v6_staged.py:5313` requiring a two-site fix. **MEASURED
now:** `N_QUERIES_DEFAULT == 100`; `train_v6_staged.py:5320` reads
`n_slot_queries=int(getattr(a, "n_slot_queries", N_QUERIES_DEFAULT))` and imports the constant at
`:127`; `refcv5_preflight` reports *"Second spellings found: none"*.
⇒ **The escalation is stale and should be struck**, or a reader will spend a turn re-fixing it.

### 3.10 ⚠️ The anchor `gate` block cannot certify its own transfer

The new artifact carries a `gate` block whose own `_scope` says *"a re-emitted file reproduces the
grid, not the measurement — re-run the gate before quoting it for a new file"*, while the report
argues no re-validation is owed. **The argument is CORRECT** — I verified max abs delta = 0.0 on both
tensors (§1 row 4). But the gate block records only the **old file's** sha (`e86cf507…`), so a reader
cannot verify the transfer from the artifact alone.
**Cheapest fix:** put `anchors_sha256` / `controls_sha256` **inside** the `gate` block, so the
transfer is a one-line positive assertion in the file rather than a claim in prose.

---

## 4. What is SOUND — certified, not merely un-refuted

* **The parity label build.** 2,400/2,400 = 100.000 % under the trainer's own arithmetic, digest MATCH, md5 `0d45b8d1…`. The same-breath control (shipped blob → 190/2,400 = 0.0792) reproduces the trainer's own comment value.
* **The anchor artifact.** Units declared **in the file**, per-column; geometry bit-identical to the live bank with a non-vacuous negative control; the units-stripped regression **REFUSED** and the converse control **not** refused. The Kamm control reproduces CLAUDE.md's documented trap to three digits.
* **The 2,400/2,376 reconciliation** (§2) — two registered corpora, no re-selection.
* **`tactical_class_ids` / `window_in_band`.** `IGNORE_ID` for out-of-band windows, never a neutral clamp; half-width derived from the record's own bands — the derived-constant trap handled correctly.
* **The `agent_w_ground` FAIL is real and correctly reported**, with a live/dead same-breath control.
* **Stage 0's `feasible_*` wiring is genuine** — 6 real dataclass fields, and that is exactly what makes the 7 phantoms diagnosable.
* **`refcv5_preflight`'s anchor check** now routes through the real resolver with a deliberate-regression control; I re-ran it and it fired.

---

## 5. ⛔ What would still be broken if refcv5 started training tomorrow morning

**It would train a model with no sampler and no agent head (§3.3), write a `config.json` asserting
seven capabilities that have no code behind them (§3.2), supervise 23.30 % of windows while the
record says 100 % (§3.5), and be scored by a panel whose defect gate cannot fire (§3.1) on a
STRATEGIC family that has never been produced in 39 artifacts.**

Ranked work items:

| # | work item | cost |
|---|---|---|
| 1 | fix `record_ok`'s scope + a **behavioural** regression test at the writer's path | one line + one test |
| 2 | freeze `DecoderConfig` (or stamp only declared fields) so a phantom seam raises | small |
| 3 | preflight rows asserting the seam's **module exists**, with converse controls | small |
| 4 | run `test_refc_sampler.py` / `test_refc_v3_refcv5_wiring.py` as a **launch gate** | small |
| 4b | fix the `diffusers` pin's dtype + tolerance so it can actually certify (§3.3b) | small |
| 5 | NOW-frame coverage for `v7_by_sid`; publish it as the supervision `n` | small |
| 6 | the 0.50 floor on the **eval** label join | four lines |
| 7 | republish the turn bound against the turn-class denominator (10.24 %, n = 293) | doc |
| 8 | honest absence encoding for `cot_source` / `alpamayo` | small |
| 9 | strike the stale `N_QUERIES_DEFAULT` escalation; fix the DE sidecar's "2,376" sentence | doc |
| 10 | tensor hashes inside the anchor `gate` block | doc |

⚠️ **Not blockers I can close:** the GPU (PI), and the Thor-side 85 GB digest (I took no slot).

---

## Deliverable manifest

All in the repo, branch `agent/arch-inf-20260803`, staged. Nothing lives only on a mirror.

| path | what |
|---|---|
| `…/2026-09-05-refcv5-completeness-review/RESULT.md` | this review |
| `…/raw/verify_coverage.out` | the coverage re-measurement (trainer arithmetic) |
| `…/raw/verify_2376.out` | the 2,400-vs-2,376 reconciliation |
| `…/raw/verify_anchors.out` | units, Kamm control, units-stripped refusal + converse control |
| `…/raw/verify_geom.out` | max abs delta = 0.0 vs both live banks, with negative control |
| `…/raw/prove_defect_blind.out` | ⭐ the mutation proof that the defect gate is blind |
| `…/raw/sweep_phantom_stamps.out` | ⭐ the 7 phantom seam keys, AST-swept |
| `…/raw/inband_windows.out` | the 23.30 % supervised-window measurement |
| `…/raw/turns_and_windows.out` | the turn denominator and label granularity |
| `…/raw/cot_and_window.out` | the perception-absence encoding |
| `…/raw/preflight_REVIEWER.json` | my own preflight run |
| `…/raw/criteria_census.out` | `tools/criteria_check.py --all taniteval/results/` |
| `…/raw/pytest_refcv5_seams.out` | my own run of the 15 red WP-4 tests |
| `…/raw/diffusers_pin_rootcause.out` | ⭐ the float64-vs-float32 / `atol=1e-10` root cause (§3.3b) |
| `…/code/*.py` | every probe above, runnable |

**Evidence class:** MEASURED (ours) throughout, except §1 row 6, stamped INCONCLUSIVE.
**Tier:** these are readiness / structural measurements, not driving results — **no T-tier is claimed
for any number here**, and none may be quoted as driving performance.
