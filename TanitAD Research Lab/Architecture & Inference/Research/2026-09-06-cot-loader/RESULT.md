# D-COT-LOADER — the silent `{}` now RAISES and the loud path is PROVEN REACHED; the dead boolean banks 57 values with their units; the docstring's under-count is corrected and re-measured at 5.3×, not 5.7×; and the new counters found 10 grounding rows the programme had been dropping in silence

**Date** 2026-09-06 (Europe/Berlin) · **Stream** Architecture & Inference ·
**Branch** `agent/arch-inf-20260803` · **Evidence class MEASURED unless stamped
otherwise** · **GPU-days spent: 0.** No model trained, no arm scored, **the A40
was not approached** (it is reserved for `refcv5-cap-b1-v72-40k`), no pod
touched, **no HuggingFace pull** (the source parquet was already on local disk,
so no quota was consumed).

**Pre-registered before any number below existed:** `PREREG.md` in this
directory (PR-1a…e, PR-2a…f, PR-3a…b, PR-4a…b), with both outcomes committed.

⛔ **NO EVAL TIER IS STAMPED AND NO FOUR-FAMILY TABLE IS REPORTED — DELIBERATELY.**
Nothing here produces a trajectory; this is a loader and label-capture audit, so
T0/T1 does not apply and ADE / longitudinal / lateral / tactical / strategic have
nothing to be computed over. Stamping a tier would be a category error. **The
four-family rule binds the arm that ever consumes these labels, and no such arm
is proposed** — §3 explains why none may be.

**Corpus denominator for every fraction below: the 4,729 distinct clips of
`Sayood/tanitad-alpamayo2-augmentation`**, local mirror
`C:/Users/Admin/tanitad-data/alpamayo/records.parquet`, md5
**`9f13474723b880eec7fcc09a7be478d8`** (re-verified on this machine before the
pre-registration was written). ⛔ **NOT** the 2,376-episode parity corpus, **NOT**
B1's 4,572. 23,644 rows, 12 columns, five tasks.

---

## 0. THE ANSWER, in nine lines

1. ⭐ **D1 IS FIXED AND THE FIX IS MUTATION-PROVEN IN BOTH DIRECTIONS, ON THE
   SHIPPED BYTES.** `alpamayo_records._load()` now raises
   `AlpamayoRecordsUnavailable` instead of `return {}`. On the shipped file
   (md5 `cf1f5f7d…`): **19/19 green**. With the defect re-introduced as **one
   line** and nothing else changed: **12 failed / 7 passed** — and **all nine
   reachability arms are among the twelve.** ⚠️ Re-run after the file was edited
   twice more, because a proof against a file you then edit is a proof about a
   file that no longer exists (§1.3).
2. ⭐ **THE PATH IS PROVEN *REACHED*, not merely proven correct.** Nine public
   entry points across three modules are parametrised and each must raise.
   *Correctness and wiring are different claims;* the mutation run is the
   evidence that this suite tests the second one.
3. ⭐ **D2 IS FIXED AND VALIDATED AGAINST AN INDEPENDENT IMPLEMENTATION.** The
   capture reproduces the sibling's classifier **clip-for-clip**: **57 READ /
   66 HEDGED / 26 NEGATED**, id sets **identical**, values **identical**.
4. ⛔ **THE BRIEF'S UNIT FIGURE IS ARITHMETICALLY WRONG AND IS CORRECTED HERE.**
   *"35.1 % (18/57)"* cannot both hold: **18/57 = 31.58 %**. The two real
   fractions are **18/57 = 31.58 % (clip level)** and **32/77 = 41.56 % (row
   level)**. **35.1 % matches neither.** Both are now published with their `n`.
5. ⛔ **ZERO of the 57 unit-less readings carries a metric value**, by
   construction: `speed_limit_ms` is `None` whenever the unit is missing, and a
   test asserts it. 1 km/h = 1/3.6 m/s, 1 mph = 0.44704 m/s — **1.609344×**.
6. ⛔ **NOTHING REACHES INFERENCE, AND THAT IS PROVEN, NOT PROMISED.**
   `goals_from_cot()` output is **byte-identical over all 4,729 clips**
   (md5 `01d9cf407cde33fd45d23dcd9095e4cc`, 639,553 B, before **and** after).
7. ⭐ **D3'S DIAGNOSIS IS CONFIRMED EXACTLY, AND ITS CORPUS FIGURE IS ITSELF A
   SCOPE ERROR.** `41` reproduces exactly as the `meta_action` CoT field alone.
   The corpus-wide figure is **219 STATED (4.63 %)**, not 235 — the published
   235 adds **16 clips where the phrase appears only in the QUESTION PUT TO THE
   MODEL**. **219 + 16 = 235, resolved to the clip.** Being *asked* about a
   speed limit is not the corpus *stating* one.
8. ⇒ ⭐ **NO BANKED RESULT IS EXPOSED.** Six banked label artifacts opened and
   checked **positively**: 4 CLEAN with real Alpamayo content, 2 not
   Alpamayo-dependent at all, **0 exposed, 0 unreadable**.
9. ⛔⛔ **AND THE INSTRUMENT FOUND A SECOND LIVE DEFECT — INCLUDING ONE OF MY
   OWN CLAIMS.** Surfacing the swallow counters showed `box_json_failed = 10`
   on the canonical corpus: **10 grounding rows have been silently dropped since
   this module was written**, all with the same truncation, **10 of 10
   recoverable** (control: a wrong prefix recovers 0). ⚠️ My first draft of §1.5
   said *"all four swallow counters read 0"* — **it was wrong**, because I had
   read only the one counter `coverage()` happens to expose. **Absence found at
   ONE location is not absence**, committed by me, inside the instrument built
   to prevent it, and caught by that instrument the moment it was wired (see 1.6).

---

## 1. ⛔ D1 — the loader that returned `{}`

### 1.1 The defect, re-verified from source

`stack/tanitad/data/alpamayo_records.py::_load()` opened with

```python
if not os.path.exists(RECORDS):
    return {}
```

`RECORDS` is a **hard-coded local path**. On any machine without it — a pod, an
off-Drive clone, a fresh checkout — **every CoT token, every box and every
`meta_action` vanished with no error**, and downstream *"this corpus states no
speed limit"* and *"the records file was not there"* were **indistinguishable**.

⭐ **This is the night's dominant failure class in a loader costume** — the same
shape as a search tool reporting *"no matches"* for files it could not open, a
`grep -c` returning 0 from an unreadable file, and a committer exiting 0 having
committed nothing. **In every one the empty result reads as an answer.**

### 1.2 The fix

| change | why |
|---|---|
| `AlpamayoRecordsUnavailable(RuntimeError)` raised on a missing source | an empty result must be **INCONCLUSIVE, never ABSENT** |
| the same exception when the file **exists but cannot be parsed** | *present-but-garbage* and *absent* are the same non-answer |
| the same exception when a readable file yields **zero clips** | a readable file that produces nothing is still an unusable source |
| `RECORDS_ENV` (`TANITAD_ALPAMAYO_RECORDS`), read at **call** time | a pod/clone/test can repoint it without editing code |
| `RECORDS_OPTIONAL_ENV=1` → **stamped** empty: warns, and `load_report()` / `coverage()` report `records_available: False` with the path tried | ⛔ a **silent** opt-out would simply reinstate the defect |
| `coverage()` carries `records_available` / `records_path` on **every** return | `clips: 0` can never again be read as *"the corpus is empty"* |
| per-reason swallow counters (`row_json_failed`, `auto_labeling_json_failed`, `box_json_failed`, `meta_action_unparsed`) exposed by `load_report()` | a per-row `except: continue` is the same defect at row scale |
| ⭐ any nonzero swallow count **WARNS** at the end of a successful load | ⛔ a counter nothing reads is still a silent swallow — see §1.6, where wiring this found a live defect |

⚠️ **No tolerance was invented.** The counters are **counted and stamped**, not
thresholded — picking a "how many dropped rows is too many" number here would be
the uncalibrated-threshold defect. The only hard refusal is the total one, and
⭐ **a warning is not a threshold**: it names what was dropped and leaves the
admissibility call to the caller. The clean-source arm asserts the loader is
**silent on a clean file** — a loader that warns on every load trains its callers
to ignore the warning, which ends in the same place as not warning at all.

### 1.3 ⭐ THE MUTATION PROOF — BOTH SIDES, RUN LIVE

`code/mutate_records.py` re-introduces **exactly one line** (`return {}` ahead of
the raise) and nothing else — a whole-file revert would have failed on missing
scaffolding and proven nothing about the **defect**.

⚠️ **RUN TWICE — and the second run is the one that counts**, because the file
gained two more edits (the swallow warning and the §1.6 docstring correction)
after the first. **A proof run against a file you then edit is a proof about a
file that no longer exists**, so it was re-run against the shipped bytes:

| state of `alpamayo_records.py` | md5 | result |
|---|---|---|
| ⭐ **SHIPPED, fixed** | `cf1f5f7d87b86c9706fb6e0dda99d1a1` | ⭐ **19 passed** |
| ⭐ **SHIPPED, defect re-introduced (1 line)** | `fe3af15ef1bf10e1b18df3f242d6e80d` | ⛔ **12 failed, 7 passed** |
| ⭐ **SHIPPED, restored** | `cf1f5f7d87b86c9706fb6e0dda99d1a1` (identical to the pre-mutation backup) | ⭐ **19 passed** |
| *(earlier run, before those two edits)* | `82dba13f…` / `73617a39…` / `82dba13f…` | 17 passed / **12 failed, 5 passed** / 17 passed |

⭐ **The same TWELVE fail in both runs** — including all nine reachability arms —
and the passing count moves 5 → 7 for the obvious reason: the two swallow tests
added in between both use a source that is PRESENT, so the mutation cannot
touch them. **A cross-check that also confirms the guard code itself never
moved:** the whole mutated code path (`path = records_path()` through
`read_parquet`) is **byte-identical between the two runs, 1,615 characters
each**.

⭐ **The seven that still pass under the mutation are exactly the seven that
SHOULD** — the three present-source arms, the clean-source silence arm, the
dropped-row arm, the static route census, and the stamped opt-out (which
returns before the mutated line). **A guard that refused everything would have
failed every one of them**; this one fails none. That is what makes the PASS
evidence rather than a tautology.

### 1.4 ⭐ WIRING, NOT CORRECTNESS — the nine routes

*A guard that works when called and is called from one path of several is how a
52 %-dead training budget survived a green suite.* So the route list **is** the
test, and the mutation run shows **all nine fail** when the defect returns.

| entry point | what it used to answer on a missing source |
|---|---|
| `alpamayo_records.available()` | `set()` |
| `alpamayo_records.get()` | `None` |
| `alpamayo_records.coverage()` | `{"clips": 0}` |
| `alpamayo_structured.coverage()` | a dict of zeros |
| `alpamayo_structured.motion_segments()` | `[]` |
| `alpamayo_structured.critical_component()` | `None` |
| `alpamayo_fusion.cot_text()` | `""` |
| `alpamayo_fusion.ground_tokens()` | every token un-grounded |
| `alpamayo_fusion.lateral_concordance()` | a null concordance |

⚠️ **Every one of those is a plausible negative finding.** `cot_text()`
returning `""` is the purest form: an extractor fed `""` produces no tokens, and
the pipeline records *"this clip says nothing"*.

⭐ **The worst concrete case is a corpus builder, not a single clip.**
`stack/scripts/s2_run_corpus.py` selects its work as
`sorted(AR.available() & ES.available())` and then writes a `summary.json`
carrying `alpamayo_lateral_agreement` / `cot_tokens_grounded_by_box` as
`"0/0"`. Under the silent path that run **completed, wrote `"done": true`, and
published a summary saying the corpus agrees on nothing** — a finished-looking
artifact built on a file that was never opened. It now fails at selection.

A tenth test (`test_pr1c_entry_point_list_covers_every_public_caller`) counts the
`AR.get(` / `AR._load(` call sites in the sibling modules and requires the
parametrised list to keep up, so a **new** consumer cannot be added without this
test noticing. It asserts the source read is non-blank first — *a blank read is
not a zero.*

### 1.5 ⛔ PR-4a — the sibling-loader sweep, every empty return classified

AST census over the file, **with the pre-fix file as a control**: 9 suspects
before, **8 after**, and the census **discriminates** (`raw/sibling_sweep.json`).
⚠️ **The census is an enumeration aid, not a guard** — *a census once read 0
suspects on both the fixed and the broken trainer.* The classification is mine:

| site | returns | class | disposition |
|---|---|---|---|
| `_load` missing source | `{}` | ⛔ **THE DEFECT** | **FIXED — raises** |
| `_load` unreadable / zero-clip | *(new)* | same family | **FIXED — raises** |
| `_load` opt-out branch | `{}` | escape hatch | **STAMPED + warned**, and only on an explicit `"1"` |
| `coverage()` `{"clips": 0}` | `{k: 0}` | ⛔ silent empty | **FIXED** — now carries `records_available` / `records_path` on every return |
| `_load` `except: continue` (row `raw_json`) | drops a row | ⚠️ **silent swallow at row scale** | **COUNTED** (`row_json_failed`), not thresholded |
| `_load` `except: a = {}` (`auto_labeling`) | drops a field | same | **COUNTED** |
| `_load` `except: arr = []` (boxes) | drops boxes | same, and the worst of the three: an unparsed box is indistinguishable from *"the question was never asked"* | **COUNTED** |
| `_parse_meta` → `{}` | unparseable `meta_action` | same | **COUNTED** (`meta_action_unparsed`) |
| `lateral` / `longitudinal` / `lane` → `None` | axis absent for **this clip** | ✅ legitimate per-clip absence | unchanged |
| `vqa_answer` → `None` | no matching question | ✅ legitimate | unchanged |
| `get(clip_id)` → `None` | clip not in corpus | ✅ legitimate — **module**-level absence is now loud, **clip**-level absence is real | unchanged |
| `grounded()` → `False` on empty boxes | ✅ already documented: *grounding can CONFIRM a token, never REFUTE one* | unchanged, and the box counter now lets a consumer tell a parse failure from a question never asked |

⛔⛔ **AND THE FIRST DRAFT OF THIS PARAGRAPH WAS WRONG — IT IS KEPT AS THE
RETRACTION IT IS.** I first wrote *"MEASURED on the real corpus: all four
swallow counters read 0 … the counters are instrumentation for a future
corruption, not a live defect."* **False.** I had read `row_json_failed` — the
one counter `coverage()` happens to expose — and generalised it to all four.
**MEASURED, the full `load_report()`:**

```
rows 23644 · clips 4729 · row_json_failed 0 · auto_labeling_json_failed 0
meta_action_unparsed 0 · box_json_failed 10        <- NOT zero
```

⚠️ **Absence found at ONE location is not absence** — the operating standard's
rule, broken by me, *inside the instrument built to enforce it*, and caught by
that instrument the moment it was wired to warn. See §1.6.

⭐ **ESCALATED, NOT EDITED (they are not mine this session):** `alpamayo_fusion.py`
and `alpamayo_structured.py` each contain their own empty-container returns
(`cot_text` → `""`, `motion_segments` → `[]`). **They are now covered by the raise
at the module boundary**, which is the correct place to fix it, but their *own*
per-clip empties remain legitimate and were not touched. No diff is proposed
because none is needed — see §6.

### 1.6 ⛔⛔ WHAT THE COUNTER FOUND: 10 GROUNDING ROWS SILENTLY DROPPED, ALL RECOVERABLE

**MEASURED over all 4,728 `grounding_via_vqa` rows** (`raw/box_drop_probe.json`,
`raw/box_recovery.json`):

| | rows | of 4,728 |
|---|---|---|
| box payload parses | **4,411** | 93.30 % |
| ⛔ **box payload UNPARSEABLE — dropped by `except: arr = []`** | **10** | **0.212 %** |
| genuinely carries no box | **307** | 6.49 % |

4,411 + 10 + 307 = 4,728. ⭐ **All ten fail identically**: the payload has lost
its leading `[{"bbox_2d": ` and begins mid-object, e.g.
`[325, 653, 491, 756], "label": "Bus"}]`.

⭐ **Recoverable, and measured as such rather than asserted**: prepending that
prefix parses **10 of 10**; ⛔ **a deliberately WRONG prefix recovers 0 of 10**
— the control that makes the 10/10 mean something. Recovered labels: Truck 3 ·
car 2 · lead vehicle 2 · Bus 1 · Car 1 · Bike with rider 1 — **real boxed
vehicles, not noise.**

⚠️ **WHY THIS IS NOT COSMETIC, AND WHY IT CORRECTS A CLAIM IN THE MODULE'S OWN
DOCSTRING.** `alpamayo_records.py` states *"4,411 clips carry ONE distinct
question; 318 carry none"*. **10 of that 318 are a PARSE FAILURE, not an
absence** — and the module's own governing rule is that *grounding can CONFIRM a
token and can NEVER REFUTE one, because a missing box overwhelmingly means the
question was not asked.* A row that failed to parse is **indistinguishable from
that**, so the drop silently converted *perception we hold* into *perception we
were never offered*, on exactly the clips where a vehicle **was** boxed. The
docstring is corrected to **4,411 / 10 / 307** with the shape and the recipe.

⛔ **NO REPAIR IS SHIPPED.** Patching a truncated generative output is a
corpus-owner decision, not a loader's; inventing one would manufacture labels.
The shape, the recipe and the control are banked and escalated (§6.6); the fix
belongs upstream in the export.

⭐ **This is the argument for the counters, made by the counters.** The
`except: arr = []` branch had been dropping these ten since the module was
written, and nothing in the programme could see it. **The residual I named as a
latent hole was live within minutes of being instrumented.**

### 1.7 ⭐ PR-4b — could a BANKED result have come through the silent path?

⛔ **Checked positively, never by "it looks complete"** — an artifact built on an
empty dict *does* look complete, which is the whole defect. Each file was opened
and required to carry a marker only a non-empty augmentation could produce
(`raw/banked_exposure.json`).

| banked artifact | n scanned | positive Alpamayo evidence | verdict |
|---|---|---|---|
| `…/2026-08-24-label-extraction-overnight/raw/s2_labels_v7.jsonl.gz` | 4,001 | 4,000 | ⭐ **CLEAN** |
| `…/2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz` | 4,001 | 4,000 | ⭐ **CLEAN** |
| `…/2026-09-04-v72-label-release/raw/s2_labels_v7.2_eval.jsonl.gz` | 147 | 147 | ⭐ **CLEAN** |
| `products/…/2026-08-23-label-validation-sample/raw/labels_v7/s2_labels_v7.jsonl` | 801 | 257 | ⭐ **CLEAN** |
| `…/2026-08-16-s2-v1-labels/labels/s2_labels_aug120.jsonl` | 201 | 0 | **NOT ALPAMAYO-DEPENDENT** |
| `…/2026-08-23-s2-abstain-v3/labels_v3/s2_labels_aug120.jsonl` | 201 | 0 | **NOT ALPAMAYO-DEPENDENT** |

⚠️ **The two zeros are not exposure and the distinction is load-bearing.** Their
schema is `clip_id · t0_s · g_str · a_str · disjointness · valid_window_s ·
_provenance` — the **geometry-only** s2 v1/v3 emitter, which never reads the
augmentation. A file with no Alpamayo-derived key was never at risk; a file with
the key and no content would have been. **That is the discrimination the check
exists to make**, and reporting the raw zero as "exposed" would have been the
same error in the opposite direction.

⚠️ **The 257/801 is also not a gap.** `s2_geom_emit_v7.py` writes
`"cot_tokens": …as_dict() if cot else None`, so `null` means *this clip's `cot`
was empty*. Under the silent path **all 801** would have been `null`; **257 are
not**, which is positive proof the augmentation was loaded when that file was
built. **One positive is enough to prove the source was live; no number of nulls
would have been enough to prove it was not.**

**0 files unreadable.** ⛔ Had any been, it would be reported **INCONCLUSIVE**,
never clean — a 0 from a file that could not be read is indistinguishable from a
real zero.

---

## 2. ⛔ D2 — the dead boolean now carries the value, the unit, and the hedge

### 2.1 The defect

`CotTokens.speed_limit` was a `bool`, assigned at one line and **read nowhere**;
`goals_from_cot` emitted no token for it; and `_SPEED_LIMIT =
re.compile(r"\bspeed limit\b")` **has no capturing group**. Every posted value
the CoT contains was matched and thrown away at the regex.

### 2.2 The capture, and PR-2a — validated against an INDEPENDENT implementation

`speed_limit_reading(text)` returns a `SpeedLimitReading` carrying
`state` · `value` · `unit` · `unit_missing` · `value_ms` · `text`. Five flat
fields were added to `CotTokens` (so the banked label record carries them), and
`extract()` calls the parser on the **RAW** `cot` — the classifier does its own
negation and hedging, and stripping first would hide the state it exists to
record.

**MEASURED over 4,729 clips, against `…/2026-09-06-speed-limit-source/code/classify_limits.py`,
which was written independently:**

| class | this module | sibling | agreement |
|---|---|---|---|
| **READ** (value off a sign, unhedged, un-negated) | **57** (1.21 %) | 57 | ⭐ **clip-id sets IDENTICAL, values IDENTICAL** |
| HEDGED (language prior) | **66** (1.40 %) | 66 | identical count |
| NEGATED | **26** (0.55 %) | 26 | identical count |

**READ clip-ROWS = 77** over 57 clips — `auto_labeling` 24 · `vqa` 23 ·
`trajectory` 15 · `meta_action` 15. The tasks corroborate each other on the same
clip.

⚠️ **One definitional quirk is INHERITED DELIBERATELY and stated rather than
silently "fixed":** the sibling's HEDGE pattern contains `not visible`, and
HEDGED is tested before NEGATED, so *"the speed limit sign is not visible"*
classifies **HEDGED**. Changing that would break clip-for-clip agreement with the
banked 57/66/26. It is pinned by a control that **predicts** the HEDGED verdict,
plus a separate control (*"There is no speed limit sign in view"*) that proves
the NEGATED branch is **reachable** — *an unreachable branch has measured nothing,
exactly like a guard that is never called.*

### 2.3 ⛔ PR-2e — THE UNIT FRACTION IN THE BRIEF IS ARITHMETICALLY IMPOSSIBLE

The brief and `…/2026-09-06-speed-limit-source/RESULT.md` §0.5 both state
**"35.1 % (18/57) state NO UNIT"**. ⛔ **18/57 = 31.58 %.** The two halves cannot
both be right, and **neither real fraction is 35.1 %**:

| level | no unit | n | fraction |
|---|---|---|---|
| ⭐ **clip level** (one reading per clip, first task wins) | **18** | **57** | **31.58 %** |
| **row level** (all contributing clip-rows) | **32** | **77** | **41.56 %** |

⭐ **This is exactly the trap the same report warns about** — *a percentage alone
is inadmissible, because two different fractions round to the same one.* Here
the number rounds to a **third** value that belongs to neither. **The `18` is
correct; the `35.1 %` is not.** Both fractions are published above with their
own `n`, and neither is quotable without its level.

**Units on the 57 clip-level readings:** `km/h` **37** · `mph` **2** ·
⛔ **none stated 18**. (Row level: 42 · 3 · 32 of 77.)

### 2.4 ⛔ PR-2b — a number without its unit is REFUSED a metric value

`speed_limit_ms` is `None` whenever `unit_missing` is `True`. **MEASURED: 0 of
the 57 unit-less readings carries an m/s value.**

⭐ **Why this is a construction and not a convention.** 1 mph / (1 km/h) =
**1.609344**, asserted on the unrounded constants. 70 km/h = 19.4444 m/s;
70 mph = 31.2928 m/s. **Same family as the `anchors.pt` control-units trap**,
where a correct formula under the wrong unit produced 396 g of lateral
acceleration and looked exactly like an answer.

### 2.5 The 57 readings, banked

Full table with clip, task, value, unit and m/s: `raw/validation_stdout.txt`;
machine-readable under `readings` in `raw/speedlimit_validation.json`. Values are
all legal posted-limit round numbers — 10, 15, 20, 30, 35, 40, 50, 60, 70, 80,
100, 120 — with **30 the mode**. Representative rows:

```
  132d381e auto_labeling      30 kph   8.3333    | A 30 km/h speed limit sign
  d5ecf705 trajectory         35 mph   15.6464   | a 35 mph speed limit sign
  2844b027 meta_action        80 NONE  -         | an 80 speed limit sign
  7833a334 trajectory        120 NONE  -         | gantry signs display 120
  3d459173 meta_action        70 NONE  -         | speed limit signs indicate 70
```

⚠️ The `NONE` rows are the point: they carry a value **and no m/s**, so a
consumer cannot use them without deciding the unit, which is what the flag is
for.

### 2.6 ⛔ PR-2f — NOTHING REACHES INFERENCE, and it is PROVEN

`goals_from_cot()` was banked over **all 4,729 clips**, twice — from `c.cot` and
from the richer `alpamayo_fusion.cot_text()` — **before** and **after** the
change:

| | md5 | bytes |
|---|---|---|
| before | `01d9cf407cde33fd45d23dcd9095e4cc` | 639,553 |
| after | `01d9cf407cde33fd45d23dcd9095e4cc` | 639,553 |

**Byte-identical.** No token was minted, no vocabulary changed, no inference
channel exists. A parametrised test asserts no emitted token contains `SPEED` or
`LIMIT` for any of the control texts.

⛔ **This is the whole scope, and it is not timidity.** The PI decision on
extract-vs-supply is **open**; coverage is 1.21 %; the grounded share is
**0 of 57**; and the readings are **ego-coupled** (median ratio limit/ego 1.01
— INHERITED, `…/2026-09-06-speed-limit-source/RESULT.md` §3). **Capture and bank
is the mandate.**

---

## 3. ⚠️ D3 — the docstring, corrected, WITH ITS TASK SCOPE

### 3.1 The diagnosis is confirmed EXACTLY

`cot_tokens_v7.py` published `speed limit  41  0.9 %` under the header
*"MEASURED YIELDS over the 4,729-clip corpus"*.

**MEASURED here, n = 4,729:**

| scope | `speed limit` |
|---|---|
| the `cot` field | **41** |
| the `meta_action` task | **41** |
| all five tasks (text only) | **219** |
| all five tasks + the QUESTION column | 235 |

⭐ **The `41` reproduces exactly**, so the diagnosis — *a true measurement quoted
outside its scope, the `df` / `step_s` family* — is **confirmed**, and the edit
was made. ⚠️ Per the pre-registration, had 41 **not** reproduced, no edit would
have been made.

⭐ **The whole table is one field.** Seven of nine probed rows reproduce exactly
on `c.cot` (`yield` 400, `parked` 419, `pedestrian` 278, `oncoming` 142, `ramp`
26, `open door` 0, `speed limit` 41). The two that did not are **pattern**
differences, not scope errors, and both were resolved to the row: `traffic light`
**638** is the module's own `_LIGHT` (`traffic light|stop light|signal`), and
`merge` **45** is the loose stem `\bmerg`. **The table was right; only its scope
line was wrong.**

### 3.2 ⛔ AND THE "CORPUS FIGURE" IS ITSELF A SCOPE ERROR — 219, not 235

`probe_speedlimit.py` searches `raw_json_text + "  ||Q|| " + question`;
`classify_limits.py` searches the text alone. **MEASURED (`raw/scope_235.json`),
resolved to the clip:**

| | clips | of 4,729 |
|---|---|---|
| ⭐ phrase in the model's own TEXT (**Alpamayo STATES it**) | **219** | **4.63 %** |
| phrase in the QUESTION put to the model (**Alpamayo was ASKED**) | 40 | 0.85 % |
| question-only, never mentioned in any answer | **16** | — |
| union — the published **235** | **235** | 4.97 % |

**219 + 16 = 235**, exactly. ⛔ **Being asked about a speed limit is not the
corpus stating one**, so **219 is the number a label pipeline can act on**, and
41 → 219 is a **5.3×** under-count, not 5.7×.

⭐ **CONTROL: my regex and the probe's return 219 = 219 on the same text**, so the
gap is the question column and not a pattern difference. *A discrepancy is only
resolved when a control isolates the one thing that differs.*

### 3.3 The correction, before → after

**BEFORE** (`cot_tokens_v7.py` docstring):

```
`meta_action` is NOT reachable locally — using it needs the source parquet. All
extraction here is therefore from the CoT sentence alone.

⭐ MEASURED YIELDS over the 4,729-clip corpus (this is what each token can
actually be populated from):
    …
    speed limit           41  0.9 %
```

**AFTER** (excerpt — full text in the file):

```
⛔ RETRACTED 2026-08-23 (C142). The sentence "…`meta_action` is NOT reachable
locally…" was WRONG and is kept here only so it cannot come back: the source
parquet IS local … and reads all FIVE tasks … over 23,644 rows / 4,729 clips.

⛔⛔ THE YIELD TABLE BELOW IS SCOPED TO ONE TEXT FIELD, AND ITS SCOPE WAS NOT
STATED — WHICH MADE ITS SPEED-LIMIT ROW READ 5.3x SMALLER THAN THE CORPUS. …
every row below is the **`cot` field of the `meta_action` task ALONE** …
⇒ Read every row as "of the `meta_action` CoT sentence", never as "of
everything we hold". Same family as the `df` / Thor `free` / `step_s` traps.

⚠️ AND 219 IS THE COUNT OF CLIPS WHERE ALPAMAYO *STATES* THE PHRASE. A further
40 carry it only in the QUESTION PUT TO THE MODEL; 219 + 16 question-only =
the 235 a sibling report published as the corpus figure.

⭐ MEASURED YIELDS, **`meta_action` CoT sentence**, n = 4,729 clips …
    speed limit           41  0.9 %   ⛔ THIS FIELD ONLY. Corpus-wide 219
                                      STATED (4.63 %) / 235 stated-or-asked;
                                      **57 carry a VALUE** — see
                                      `speed_limit_reading`.
```

⭐ **A second stale-absence claim was fixed in the same edit**, unprompted: the
file still asserted *"`meta_action` is NOT reachable locally"* — retracted as
C142 in `alpamayo_records.py`'s own docstring **on 2026-08-23** and never
propagated here. **Exactly the rot the feature-count test was pinned for**, in a
neighbouring file. It is kept as a struck-through quote so it cannot return.

⚠️ **`open door 0` was also corrected**, because a bare `0` in a table whose
scope was wrong is an absence claim: it is 0 in this field and **3 across all
five tasks** — *rare, not absent.*

---

## 4. ⛔ THE PRE-REGISTERED GATES, SCORED

| id | assertion | verdict |
|---|---|---|
| PR-1a | missing source raises a named error carrying path / env / md5 | ⭐ **PASS** |
| PR-1a′ | an unreadable file raises the same way | ⭐ **PASS** |
| PR-1b | a present source is **not** refused; toy parquet loads; real corpus = **4,729** | ⭐ **PASS** |
| PR-1c | the loud path is **reached** from all 9 public entry points | ⭐ **PASS** (and all 9 fail under mutation) |
| PR-1d | one checker, two loaders, opposite verdicts | ⭐ **PASS** (old `False`, new `True`) |
| PR-1e | opt-out is warned + stamped, and requires an explicit `"1"` | ⭐ **PASS** |
| PR-2a | 57 / 66 / 26 with **identical clip-id sets and values** vs the sibling | ⭐ **PASS** |
| PR-2b | 0 unit-less readings carry an m/s value | ⭐ **PASS** |
| PR-2c | 10 controls at known values; all four states reachable | ⭐ **PASS** |
| PR-2d | pre-fix regex captures **0**, current captures **57** | ⭐ **PASS** |
| PR-2e | both fractions published with their `n` | ⭐ **PASS** — and the brief's 35.1 % **refuted** |
| PR-2f | `goals_from_cot` byte-identical over 4,729 clips | ⭐ **PASS** |
| PR-3a | `meta_action` alone = **41**; corpus-wide = **235** | ⛔ **PARTIAL — and the PRE-REGISTRATION WAS MIS-SPECIFIED.** The 41 reproduced **exactly**, which is the half that gates the edit. The **235 did NOT** — it is **219** — so as written this gate FAILED. See the note below. |
| PR-3b | the corrected text states its task scope | ⭐ **PASS** |
| PR-4a | every empty-container return classified | ⭐ **PASS** (12 sites, table §1.5) |
| PR-4b | banked artifacts checked **positively** | ⭐ **PASS** — 0 exposed, 0 unreadable |
| ⭐ **beyond the prereg** | a nonzero swallow count WARNS; a CLEAN source stays silent | ⭐ **PASS both sides** — and it immediately found `box_json_failed = 10` on the live corpus (§1.6) |

⛔⚠️ **PR-3a IS SCORED AS A PARTIAL, NOT A PASS, AND THE FAULT IS IN MY PRE-REGISTRATION.** I wrote the gate as *"`meta_action`-only = 41 AND corpus-wide = 235"*, with **FAIL = either differs**. The second conjunct was **not a prediction about the artifact I was correcting** — it was an INHERITED number from a sibling report, folded into my own gate as though I had measured it. That is the wrong shape for a pre-registration: a gate should bind the claim I am making, not import someone else's.
⭐ **The edit was still made, and here is the reasoning, stated so it can be disagreed with:** the gate's PURPOSE was to stop me correcting a docstring on a mis-diagnosis, and that purpose is served entirely by the `41`, which reproduced exactly. The `235` half turned out to be a defect in the *sibling's* number, fully explained and control-verified (219 + 16 = 235, two regexes agreeing at 219 = 219 on the same text). ⚠️ **A reader who thinks the gate should have blocked the edit is entitled to that view** — which is exactly why it is written here rather than quietly re-scored as a PASS.

**Tests: 19 + 29 = 48 new, all green. Full `stack/tests` suite: see §4.1.**

---

### 4.1 ⚠️ THE FULL `stack/tests` SUITE — WHAT I ACTUALLY HAVE, AND WHAT I DO NOT

⛔ **I do not have a full-suite green, and I am not going to imply that I do.**

**What ran, and what it says:**

| scope | result |
|---|---|
| ⭐ `test_alpamayo_records_loud_absence.py` (new) | **19 passed**, on the shipped md5, plus the two-sided mutation run |
| ⭐ `test_cot_speed_limit_capture.py` (new) | **29 passed** |
| ⭐ **all 15 test files in the COMPUTED transitive importer closure** | ⭐ **15/15 files, 217 tests, 0 failures** — per-file table below |
| ⛔ the whole `stack/tests` suite | **NOT COMPLETED — reached 29 % and was killed by me. INCONCLUSIVE, not green.** |

**Why the full suite did not finish, MEASURED rather than assumed.** It ran ~50 minutes and stopped
advancing. ⚠️ **Before calling it hung I sampled it** — *verify before alarming*: **CPU moved
838.9 s → 840.9 s across 45 s of wall clock (4.4 %) with the log not growing.** ⇒ **STARVED, NOT
HUNG**, which is a different diagnosis with a different response.

⭐ **A process listing named the competitors, and they are all OTHER AGENTS' JOBS:**

| pid | CPU | RSS | what it is |
|---|---|---|---|
| 25436 | 1,427 s | 1.1 GB | a sibling's `build_b1_agent_join.py --pose-source reconstruct` |
| 30204 | 201 s | 2.9 GB | a sibling's `taniteval/tools/refcv3_arm.py` eval |
| 25372 | — | 3.4 GB | ⭐ **another agent running `pytest stack/tests` itself** |
| 2724 | — | 0.6 GB | a fourth agent's `pytest tests/` |

⛔ **I killed none of them.** `build_b1_agent_join.py` is explicitly outside my ownership and the
rest are other agents' production work. I killed **my own** suite instead, because it was consuming
the contention budget of the checks that could still produce evidence, and producing none itself.
⭐ **And a sibling is already running the same suite**, so the full-suite answer is in flight on this
box regardless — it simply is not mine to report.

⚠️ **The partial run showed ~5 failures in its first 29 %, and I CANNOT ATTRIBUTE THEM.** pytest
prints failure names only in its end summary, which the kill discarded. ⛔ **So I will not claim
they are pre-existing, and I will not claim they are mine.** Instead I bounded the question the
other way round, which is answerable:

⭐ **THE BLAST RADIUS IS COMPUTED, NOT ASSUMED.** A fixed-point import scan over all **956** `stack/`
Python files (**0 unreadable** — asserted, because a scan that could not open a file would report a
clean zero) gives the **transitive importer closure** of the two modules I changed:

* **10 non-test modules:** `alpamayo_records` · `cot_tokens_v7` · `alpamayo_fusion` ·
  `alpamayo_structured` · `s2_geom_emit_v7` · `s2_run_corpus` · `build_parity_v7geom_labels` ·
  `refa_v1_profile` · `refa_v1_train` · `refav1_loader`
* **15 test files** — listed in the per-file table below.

⛔ **A test outside that closure cannot execute either changed module, so it cannot be affected by
this change.** ⇒ **every test that CAN be affected was run, individually, and the result is in the
table.** Two further facts point the same way: `goals_from_cot` is **byte-identical over all 4,729
clips**, and **no `cot_tokens` key set is pinned anywhere** (checked in `test_v7_labels.py`,
`test_v7_wiring.py`, `test_vocab_v7.py`), so the five added fields cannot break a schema assertion.

⚠️ **This is a bounded claim and I am not inflating it: "nothing in my blast radius fails" is NOT
"the suite is green".** The unattributed failures remain unattributed.

⛔ **CONSEQUENCE, STATED PLAINLY: under `CLAUDE.md`'s rule that `pytest -q` must be green before a
commit, this work is STAGED AND BLOB-VERIFIED, NOT COMMITTED.** The blocker is named and it is not
mine to clear: the mount is saturated by a sibling's job. **Re-running the full suite once that job
finishes is the one outstanding action on this package**, and it needs no decision from anyone.

**Per-file, every pre-existing test that imports `alpamayo_records` or
`cot_tokens_v7` (run individually, because the mount could not carry a batch):**

| test file | result |
|---|---|
| `test_alpamayo_records_loud_absence.py` | ⭐ **19 passed (NEW)** |
| `test_cot_speed_limit_capture.py` | ⭐ **29 passed (NEW)** |
| `test_alpamayo_fusion_sides.py` | ⭐ **14 passed in 41.91s** |
| `test_alpamayo_lateral_agree.py` | ⭐ **15 passed in 39.64s** |
| `test_alpamayo_lateral_side.py` | ⭐ **13 passed in 45.71s** |
| `test_cot_corridor_offset.py` | ⭐ **12 passed in 47.55s** |
| `test_cot_lane_change.py` | ⭐ **7 passed in 46.41s** |
| `test_cot_negation.py` | ⭐ **10 passed in 55.32s** |
| `test_vocab_v7.py` | ⭐ **33 passed in 164.79s (0:02:44)** |
| `test_refc_v3_nav_from_v7.py` | ⭐ **14 passed in 73.05s (0:01:13)** |
| `test_tac_loss_logging.py` | ⭐ **8 passed in 61.71s (0:01:01)** |
| `test_refav1_loader.py` | ⭐ **8 passed in 42.43s** |
| `test_refav1_loader_labels.py` | ⭐ **9 passed in 44.59s** |
| `test_refa_v1_speed_channel.py` | ⭐ **11 passed in 51.84s** |
| `test_refav1_arm.py` | ⭐ **15 passed in 114.41s (0:01:54)** |


## 5. ⚠️ ONE LINE — CAN A MISSING INPUT STILL LOOK LIKE A NEGATIVE FINDING IN THIS MODULE?

⭐ **No — not at the module boundary, not on any of the nine public routes, and
no longer at row scale either: every silent-swallow path now RAISES or WARNS, and
both were mutation-proven in each direction. The one thing that can still read
like a negative finding is a `grounded()` False on a clip whose single box row was
dropped — and that is now COUNTED, WARNED, and quantified at exactly 10 of 4,728
(§1.6), so it is a known and named quantity rather than an invisible one.**

⭐ **RULE ZERO, honoured rather than cited.** My first answer to this question was
*"YES, in one remaining place — a per-ROW `json.loads` failure, but all four
counters read 0, so it is latent."* That answer named the next lever and stopped.
**The cheapest experiment was to wire the counters to a warning and run the
loader once — minutes, zero GPU — so it was run in the same turn.** It refuted
my own "all four read 0" and surfaced **10 grounding rows the programme had been
dropping since this module was written, 10 of 10 recoverable.** ⛔ A turn that had
ended at *"logged as the next lever"* would have shipped a diagnosis and called it
a product.

⚠️ **What genuinely remains, with its blocker named:** the ten truncated payloads
are **not repaired**, because repairing a truncated generative output is a
corpus-owner decision and inventing a patch would manufacture labels. That is a
**PI / DataFlyWheel decision** (§6.6), not an engineering gap, and everything
around it — the shape, the recipe, the control, the corrected docstring — is
done.

---

## 6. 🔴 ESCALATIONS

1. **→ the author of `…/2026-09-06-speed-limit-source/RESULT.md` (and anyone
   quoting it).** ⛔ **§0.5 / §2's "35.1 % (18/57)" is arithmetically
   impossible.** The clip-level fraction is **18/57 = 31.58 %**; the row-level is
   **32/77 = 41.56 %**; **35.1 % is neither.** The `18` is right. **Exact
   correction:** replace *"35.1 % OF THE READINGS CARRY NO UNIT — 18/57"* with
   *"31.58 % of the clip-level readings carry no unit — 18/57; at row level
   41.56 % — 32/77."* ⚠️ I did not edit that file: it is the sibling's.
2. **→ the same author.** ⛔ **§0.2 / §2's corpus figure `235/4,729` conflates
   what Alpamayo STATES with what it was ASKED.** **219 state it; 16 more carry
   the phrase only in the question; 219 + 16 = 235.** **Exact correction:**
   *"the phrase appears on 219/4,729 clips (4.63 %) in the model's own text, and
   on 235 if the sampled VQA question text is counted as corpus content."* The
   under-count factor is then **5.3×**, not 5.7×. Instrument:
   `code/scope_235.py`, `raw/scope_235.json`.
3. **→ DataFlyWheel / v7 label owners.** The five new `CotTokens` fields land in
   the banked label record via `s2_geom_emit_v7.py:975` **the next time labels
   are rebuilt**. They are **purely additive** — `goals_from_cot` is byte-identical
   — so no existing arm's recipe changes and no rebuild is required by this work.
   ⛔ I did not rebuild any label blob and did not touch `v7_labels.py`.
4. **→ whoever owns `alpamayo_fusion.py` / `alpamayo_structured.py`.** **No diff
   is proposed and none is needed** — their empty returns are now covered by the
   raise at the module boundary, and their per-clip empties are legitimate. This
   is recorded so the absence of a diff is a **decision**, not an omission.
5. **→ PI (unchanged, not mine).** Extract-vs-supply, and whether to authorise
   the 10.8 h / 24.1 h VQA re-ask. ⛔ **I did not start it.**
6. ⭐⭐ **→ DataFlyWheel / corpus owner — NEW, and the most actionable item here.**
   **10 of 4,728 `grounding_via_vqa` box payloads are TRUNCATED and have been
   silently dropped since this module was written** (0.212 %); **10 of 10 are
   recoverable** by prepending the lost `[{"bbox_2d": ` prefix, and a deliberately
   wrong prefix recovers 0. Affected clips and payloads:
   `579d79f2` Bus · `e3a44ea9` Truck · `9f3c098b` Car · `5786a9a5` Bike with rider ·
   `a11dd078` car · `f4740120` lead vehicle · `4a92e409` Truck · `038f1d03` car ·
   `f388678b` lead vehicle · `b2497cc3` Truck.
   ⛔ **I did not repair them** — patching a truncated generative output is a
   corpus decision, not a loader's. **The right fix is upstream in the export**;
   if that is not possible, a repair must be pre-registered as a corpus change
   with its own control, not slipped into a loader. Instruments:
   `code/box_drop_probe.py` · `raw/box_drop_probe.json` · `raw/box_recovery.json`.
   ⚠️ **`alpamayo_records.py`'s docstring is corrected in the same commit** — its
   *"318 carry none"* is really **10 parse-failures + 307 genuine**.

---

## 7. What I did NOT do

* ⛔ No GPU of any kind. **The A40 was not approached** — it is reserved for
  `refcv5-cap-b1-v72-40k`. No pod, no Thor, no dev-box CUDA job. No HF pull, no
  spend, no quota consumed.
* ⛔ **No VQA re-ask started** — an open PI decision.
* ⛔ **No max-speed head, no inference channel, no vocabulary token**, and
  `goals_from_cot` is proven byte-identical.
* ⛔ Did not edit `refc.py`, `refc_v3*.py`, `refcv3_arm.py`, `taniteval/ci.py`,
  `train_v6_staged.py`, `v6.py`, `predictor.py`, `goal_point.py`,
  `v7_labels.py`, `stack/tanitad/rl/`, `refav1_lon_cost.py`,
  `build_b1_agent_join.py`, `constraints.py`, `CLAUDE.md`, or the paper.
* ⛔ Did not edit the sibling's `…/2026-09-06-speed-limit-source/` package —
  its two corrections are escalated with exact replacement text in §6.
* ⛔ Did not open any camera frame, so I make no claim about whether an
  individual reading is a real sign, a dashboard, or a hallucination.
* I did not re-derive the ego-coupling or grounding results; they are cited
  **INHERITED** with their path.

---

## 8. Deliverable manifest

| artifact | path | state |
|---|---|---|
| pre-registration | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-cot-loader/PREREG.md` | repo, **staged** |
| this document | `…/2026-09-06-cot-loader/RESULT.md` | repo, **staged** |
| ⭐ **the loud-failure fix** | `stack/tanitad/data/alpamayo_records.py` | repo, **staged** |
| ⭐ **the value capture + corrected docstring** | `stack/tanitad/data/cot_tokens_v7.py` | repo, **staged** |
| ⭐ **17 loud-failure / reachability / mutation tests** | `stack/tests/test_alpamayo_records_loud_absence.py` | repo, **staged** |
| ⭐ **29 capture / units / no-token tests** | `stack/tests/test_cot_speed_limit_capture.py` | repo, **staged** |
| the deliberate-regression mutator | `…/2026-09-06-cot-loader/code/mutate_records.py` | repo, **staged** |
| PR-2 / PR-3 validation harness | `…/code/validate_speedlimit.py` · `raw/speedlimit_validation.json` · `raw/validation_stdout.txt` | repo, **staged** |
| the 219-vs-235 resolver | `…/code/scope_235.py` · `raw/scope_235.json` | repo, **staged** |
| PR-2f goals control | `…/code/bank_goals.py` | repo, **staged** |
| PR-4a sibling census (+ pre-fix control) | `…/code/sibling_sweep.py` · `raw/sibling_sweep.json` | repo, **staged** |
| PR-4b banked-exposure check | `…/code/banked_exposure.py` · `raw/banked_exposure.json` | repo, **staged** |
| ⭐ **the 10 dropped grounding rows + recovery control** | `…/code/box_drop_probe.py` · `raw/box_drop_probe.json` · `raw/box_recovery.json` | repo, **staged** |

**Source data (NOT banked — 24.8 MB, already on local disk and on HF):**
`C:/Users/Admin/tanitad-data/alpamayo/records.parquet`, md5
`9f13474723b880eec7fcc09a7be478d8`. Every script above regenerates its JSON from
it. **Nothing lives only on a pod, only in a worktree, or only in this agent's
context.**
