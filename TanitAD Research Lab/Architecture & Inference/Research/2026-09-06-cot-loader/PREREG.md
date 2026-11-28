# PREREG — D-COT-LOADER: the silent `{}`, the dead boolean, the 5.7x docstring

**Date** 2026-09-06 (Europe/Berlin) · **Stream** Architecture & Inference ·
**Branch** `agent/arch-inf-20260803` · **Author** CoT-loader agent
**Written BEFORE any number below exists.** Every assertion has both outcomes
committed. GPU budget: **0** — no arm, no training, the A40 is not approached.

**Denominator for every fraction in this package:** the **4,729 distinct clips**
of `Sayood/tanitad-alpamayo2-augmentation`, local mirror
`C:/Users/Admin/tanitad-data/alpamayo/records.parquet`, md5
**`9f13474723b880eec7fcc09a7be478d8`** (verified on this machine before writing
this file). ⛔ **NOT** the 2,376-episode parity corpus, **NOT** B1's 4,572.

⛔ **NO EVAL TIER IS STAMPED AND NO FOUR-FAMILY TABLE IS REPORTED, DELIBERATELY.**
No model produces a trajectory anywhere in this package; it is a loader/label
audit. Stamping T0/T1 would be a category error. The four-family rule binds the
arm that ever consumes these labels, and no such arm is proposed here.

---

## PR-1 — D1: `alpamayo_records._load()` must FAIL LOUDLY, and the loud path must be REACHED

**The defect (INHERITED, `…/2026-09-06-speed-limit-source/RESULT.md` §7.4,
re-verified here by reading the source):** `_load()` answers a missing source
parquet with `return {}`. Downstream, *"no speed limit found"* and *"the records
file was not there"* are indistinguishable.

| id | pre-registered assertion | PASS | FAIL |
|---|---|---|---|
| **PR-1a** | With the records path pointed at a path that does not exist, `_load()` raises a **named** exception carrying the path. | raises `AlpamayoRecordsUnavailable`, message contains the path | returns any container, or raises an unnamed/`KeyError`-class error |
| **PR-1b** | ⭐ **The other side.** With the path pointed at the real parquet, the loader works normally and returns **exactly 4,729** clips. | 4,729 clips, no exception | any other count, or an exception |
| **PR-1c** | ⭐ **WIRING, not correctness.** The loud path is **reached** through every PUBLIC entry point, not only by calling `_load()` directly: `available()`, `get()`, `coverage()`, and the sibling module's `alpamayo_structured.coverage()`. | all four raise | any one returns an empty container |
| **PR-1d** | ⛔ **Deliberate-regression arm.** One checker, two loaders: the **pre-fix** body (`if not exists: return {}`) must be judged **NOT LOUD**, the post-fix body **LOUD**. | old=False, new=True | any other pair — a checker that passes both, or fails both, has measured nothing |
| **PR-1e** | The explicit opt-out (`TANITAD_ALPAMAYO_RECORDS_OPTIONAL=1`) is **stamped, not silent**: it warns, and `coverage()` reports `records_available: False` with the path it tried. | warns AND stamps | returns `{"clips": 0}` with no marker |

⚠️ **Why PR-1c is the assertion that matters.** *Correctness and wiring are
different claims, and only the first is usually tested.* A guard that raises
when called, reached from one call path of four, is how a defect survives a
green suite. PR-1a is the correctness claim; **PR-1c is the wiring claim.**

⚠️ **Why PR-1b is not padding.** A guard that refuses everything passes PR-1a
trivially and destroys the pipeline. Both sides are asserted or neither is
evidence.

## PR-2 — D2: capture the speed-limit VALUE, its UNIT, and its HEDGING state

**The defect:** `CotTokens.speed_limit` is a `bool`; `_SPEED_LIMIT` has no
capturing group; `goals_from_cot` emits nothing for it. Every posted value the
CoT contains is extracted and discarded.

| id | pre-registered assertion | PASS | FAIL |
|---|---|---|---|
| **PR-2a** | A new `speed_limit_reading(text)` reproduces the sibling's READ/HEDGED/NEGATED classification **clip-for-clip** when fed the sibling's own per-clip/per-task blobs: **57 READ / 66 HEDGED / 26 NEGATED** clips of 4,729, and the READ **clip-id set is identical**. | all three counts match AND the id sets are equal | any count differs, or one id differs |
| **PR-2b** | ⛔ **The units are carried explicitly and the unit-less third is REFUSED a metric value.** Every reading with no stated unit has `speed_limit_unit is None`, `speed_limit_unit_missing is True`, and `speed_limit_ms is None`. No reading is silently defaulted to km/h. | 0 unit-less readings carry an m/s value | any unit-less reading carries one |
| **PR-2c** | ⛔ **CONTROLS AT KNOWN VALUES.** (i) empty/`None` text → no reading, `speed_limit_value is None` — the no-information value, **exactly**; (ii) a negated sentence → state `NEGATED`, no value; (iii) a hedged language-prior sentence → state `HEDGED`, no value; (iv) a plain unhedged sign sentence with a unit → state `READ`, the value and unit exactly as written. | all four at their known values | any one off |
| **PR-2d** | ⛔ **Deliberate-regression arm for the capture.** The pre-fix regex `\bspeed limit\b` must yield **0 captured values** under the same harness, and the new one **> 0**. | old=0, new>0 | old>0 (harness is not reading the regex) or new=0 |
| **PR-2e** | ⭐ **The `35.1 % (18/57)` figure is internally inconsistent and is RE-DERIVED here.** The pre-registered check is arithmetic, not empirical: `18/57 = 31.58 %`, so a report cannot carry both. I commit in advance to publishing the measured clip-level fraction **and** the row-level fraction, each with its own `n`, whatever they turn out to be. | both fractions published with their `n` | a bare percentage is published |
| **PR-2f** | ⛔ **No token is emitted and no inference input is created.** `goals_from_cot()` output is **byte-identical** before and after this change across all 4,729 CoTs. | 0 of 4,729 differ | any differ |

⛔ **Scope, committed in advance:** capture and bank. **No max-speed head, no
inference channel, no vocabulary token.** The extract-vs-supply decision is the
PI's and is open; the labels are ego-coupled (median ratio limit/ego **1.01**,
grounded **0 of 57**, INHERITED from the sibling) and would not be admissible
even if the decision were mine.

## PR-3 — D3: the docstring under-counts by 5.7x

| id | pre-registered assertion | PASS | FAIL |
|---|---|---|---|
| **PR-3a** | `cot_tokens_v7.py`'s published yield `speed limit  41  0.9 %` is the **`meta_action` task alone**; the corpus-wide phrase count is **235/4,729**. Both are re-measured here from the parquet, not copied. | `meta_action`-only = 41 **and** corpus-wide = 235 | either differs — then the correction itself is wrong and is not made |
| **PR-3b** | ⭐ The corrected text **states its task scope**, so the same under-count cannot recur silently. | the scope appears in the corrected lines | it does not |

⚠️ **PR-3a is a genuine two-sided gate.** If the re-measurement does not
reproduce 41 for `meta_action`, the diagnosis ("a true measurement quoted
outside its scope") is wrong and no edit is made — the docstring would then be
wrong for a different reason and would need a different correction.

## PR-4 — the sibling-loader sweep, and the banked-result exposure

| id | pre-registered assertion | PASS | FAIL |
|---|---|---|---|
| **PR-4a** | Every function in `alpamayo_records.py` that can return an empty container on a **missing or unparseable input** is enumerated, and for each one it is stated whether the empty is (i) the defect, (ii) a legitimate per-clip absence, or (iii) an unfixed silent swallow that is now **counted and stamped**. | every one classified | any left unclassified |
| **PR-4b** | ⛔ **Whether any BANKED result could have been produced through the silent path is checked positively**, by opening banked artifacts and asserting they carry non-empty Alpamayo content — never by assuming they are fine because they look complete. | each named artifact carries a positive marker | reported INCONCLUSIVE with the reason |

## What would make me report INCONCLUSIVE

* ⛔ A file that cannot be **read** (not merely "grepped to zero"). The G: mount
  flaps and has produced correct-size all-NUL files; an absence found through a
  single mechanism is not an absence. Every absence claim here carries a second
  mechanism with a **non-zero control**.
* A count that a second mechanism does not reproduce.

## What I will NOT do

* ⛔ No GPU. **The A40 is reserved for `refcv5-cap-b1-v72-40k` and is not
  approached.** No VQA re-ask is started (10.8 h / 24.1 h — an open PI decision).
* ⛔ No edit to `refc.py`, `refc_v3*.py`, `refcv3_arm.py`, `taniteval/ci.py`,
  `train_v6_staged.py`, `v6.py`, `predictor.py`, `goal_point.py`, `v7_labels.py`,
  `stack/tanitad/rl/`, `refav1_lon_cost.py`, `build_b1_agent_join.py`,
  `CLAUDE.md`, or the paper.
* ⛔ No commit to `main`, no push. Staged only.
