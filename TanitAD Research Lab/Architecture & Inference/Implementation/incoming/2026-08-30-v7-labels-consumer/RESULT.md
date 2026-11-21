# RESULT — `v7_labels.py`, the ONE B1 label consumer

**Date:** 2026-08-30 · **Owner:** TanitAD_TrainingFlyWheel · **Spec:**
`Project Steering/SPEC_V7_LABELS_CONSUMER.md` · **Unblocks** v7f, refc_v3,
refa_v1, refd. **Evidence class:** MEASURED (ours) — every number below is
**re-derived by the module** from the released blob, not copied from the spec.

**Blob identity, verified independently before a byte was read:**
`.../release/tanitad-v7-training-corpus/labels/s2_labels_v7.jsonl.gz`,
md5 **`ee44875916ae7c0ac002c6716b9658ea`**, **4,719** records, `s2-geom-v7`,
vocab `v7`. ⚠️ Five other copies exist under three roots with differing md5s; one
in a research `incoming/` directory (`e22acf70…`) is a stale schema.

---

## 0. The three binding conditions — implemented and pinned

| condition | status |
|---|---|
| **Oracle nav unreachable by default, opt-in STAMPS the config** | ✅ `oracle_nav()` checks the **manifest**, not a bare argument, so the permission and the recorded config are the same object and cannot disagree |
| **`a_tac` stays factored (lat ⟂ lon)** | ✅ two heads; `flatten_tactical_actions()` exists only to **raise**, pinned by a deliberate-regression test |
| **Full-width heads, masked LOSS, mask ≡ absence** | ⚠️ implemented — **and the assertion FAILS on the released blob. That is a real defect, see §2** |

## 1. Corpus counts, re-derived by the code

| head | width | present | masked | counts |
|---|---|---|---|---|
| `tac_lat` | 8 | **5** | LANE_CHANGE_L·LANE_CHANGE_R (absent) · ABORT_LC (both) | LANE_KEEP 3057 · NUDGE_R 613 · NUDGE_L 502 · TURN_L 280 · TURN_R 267 |
| `tac_lon` | 8 | **7** | YIELD_MERGE (both) | CRUISE 1287 · ACCELERATE 1038 · ADAPT_SPEED_FOR_CURVE 1018 · BRAKE_TO 927 · FOLLOW 175 · CREEP 141 · HOLD 133 |
| `str_action` | 7 | **5** | PREPARE_EXIT · PREPARE_LANE_CHANGE (both) | HOLD_MAIN_ROAD 2468 (**52.3 %**) · PREPARE_TURN_R 624 · PREPARE_TURN_L 585 · RESUME_CRUISE 573 · PREPARE_STOP 469 |
| `str_goal` | 8 | **4** | the 4 EXIT_/LANE_CHANGE_ route tokens (both) | FOLLOW_ROUTE 3041 · TURN_RIGHT 624 · TURN_LEFT 585 · STOP_AT 469 |

The presence counts (5 / 7 / 5 / 4) reproduce the spec's §3 table exactly.

## 2. ⛔ THREE MEASURED DIVERGENCES FROM THE SPEC — none invented, none dropped

### (a) The vocab mask is INCOMPLETE — two dead logits, via a NAME COLLISION

The spec asserts *"the 8 `NOT_YET_EXTRACTABLE` tokens are exactly the classes with
zero occurrences, per head"*. **They are not.** The mask contains
`LANE_CHANGE_L_FOLLOW_ROUTE` / `_R_FOLLOW_ROUTE` — **strategic goal** tokens — but
not `LANE_CHANGE_L` / `LANE_CHANGE_R`, the **tactical lat action** tokens. The two
families differ by a suffix and the arithmetic conflated them.

⇒ Both tactical tokens are **absent from the corpus AND unmasked** = two dead
logits on `tac_lat`: present in the softmax, never trained, degrading calibration.

⭐ **ROOT CAUSE ESTABLISHED FROM SOURCE (Master Mind, verified independently here):
it is an EXTRACTOR LIMIT, not a corpus accident.** There is no `LANE_CHANGE` branch
anywhere — `ego_manoeuvre.py:97` declares `lateral_class` over
`JUNCTION_TURN_L/R | ROAD_BEND_L/R | NUDGE_L/R | STRAIGHT`, and
`s2_geom_emit_v7.py` (`:446`/`:449`/`:452`) emits only NUDGE, TURN or LANE_KEEP.
That is the PI's 2026-08-16 lane-change ruling applied consistently: with no lane
reference, lateral offset alone is not evidence of a lane change, so **NUDGE is the
honest label rather than a lost one**, and the two tokens genuinely belong in
`NOT_YET_EXTRACTABLE`.

⛔ **CONSUMER-VISIBLE CONSEQUENCE — `NUDGE_L/R` IS A SUPERSET, and `LANE_KEEP` is
too.** The nudge test is `abs(lat) >= NUDGE_LAT_M` with `NUDGE_LAT_M = 1.0` m and
**no upper bound** (`ego_manoeuvre.py:82`, `:318`), while a lane is 3.2–3.7 m — so
a lane change satisfies it by construction. ⚠️ **The same line also requires
`abs(peak_yaw) >= 5.0°`**, so a *gentle* lane change fails the nudge test and lands
in `LANE_KEEP`. ⇒ lane changes are absorbed into **NUDGE or LANE_KEEP depending on
yaw**, not reliably into either. Do not read `NUDGE_L` as "a small lateral
correction", nor `LANE_KEEP` as "stayed in lane". *(This refinement is mine: the
`peak_yaw >= 5.0` conjunct means the absorption is split across two classes, not
concentrated in NUDGE.)*

**Handled without silently patching the vocab** (which other consumers share):
`assert_mask_matches_presence()` **fails loudly** — a test pins that failure so it
cannot be forgotten — while `effective_mask()` gives the trainer
`vocab ∪ empirically-absent` with per-token provenance (`vocab` / `absent` /
`both`), so no arm carries a dead logit meanwhile. ⭐ **`vocab_v7.NOT_YET_EXTRACTABLE`
needs the two tactical tokens added; that is a vocab change and not mine to make
unilaterally.** When it lands, invert the pinning test to an equality assertion.

### (b) `oracle: true` is on 4,190 of 4,719 — a guard on the FLAG leaks 11.2 %

| | n |
|---|---|
| `provenance == "ego-future"` | **4,719 (100 %)** |
| `oracle: true` present | **4,190** |
| `oracle` key **absent** | **529** — all `NAV_FOLLOW_ROAD`, all still `ego-future`, carrying a `reason` (*"curve, not a turn — nav does not command a turn"*) |

⇒ The spec's *"oracle:true, 100 %"* is true of the **provenance** and false of the
**flag**. `if rec["oracle"]` raises `KeyError`; `rec.get("oracle") is True` lets
**529 ego-future records through as non-oracle**. `is_oracle_nav()` keys on
provenance. ⭐ The module's guard was already safe because it gates on the
**manifest** rather than per-record — the design survived a fact it did not know.

### (c) `disputed` / `time_basis` / `t_nominal_s` / `agree` DO exist

The spec reports zero hits for the first three and places `agree` at the top of
`alpamayo`. All four exist one level deeper than its search reached:

* `disputed` / `time_basis` / `t_nominal_s` — **per goal**, inside
  `g_tac.goals.<TOKEN>`. **2,343 records (49.6 %) carry at least one disputed goal.**
* `agree` — in `alpamayo.lateral.agree` (2575 true / 1841 false / 303 null) and
  `alpamayo.longitudinal.agree` (2995 true / 1718 false / 6 null).

⇒ **Following the spec would have discarded a disputed-flag on half the corpus**,
which the D-LABEL-GT conditions require be respected. All four are surfaced under
`.audit`.

## 3. Audit passthrough (never training inputs)

`turn_suppression` non-null **68** · `unassigned_manoeuvres` non-empty **54** ·
`tac_anchor` present **4,719/4,719** · `t0_s` = 8.0 on all records ·
bands constant (`operative [0,2]`, `tactical [2,6]`, `strategic [8,30]`).

⚠️ The spec warns `t0_s` and band edges are per-record and must not be assumed
constant. **Measured: they ARE constant in this blob.** The module still reads
them per-record — the advice is right even where the variance is currently zero,
and a future blob may differ.

## 4. Skew weighting

`class_weights(labels, head, scheme="inverse")` computed **from the loaded split**,
never hardcoded (the derived-constant trap). Masked classes get weight **0.0** —
they contribute no loss, so any other value misstates the objective.

## 5. Deliverable manifest

| artifact | location |
|---|---|
| module | `stack/tanitad/data/v7_labels.py` (repo, staged) |
| tests | `stack/tests/test_v7_labels.py` — **21 passed**, 4 against the real blob (repo, staged) |
| corpus counts | `C:/Users/Admin/tanitad-data/v7_labels_corpus_counts.json` (local; copied into the research dir) |
| this RESULT | research dir (repo, staged) |

⛔ **Nothing pushed; nothing on `main`.**

**Open, and NOT mine to close unilaterally:** `vocab_v7.NOT_YET_EXTRACTABLE`
should gain `LANE_CHANGE_L` and `LANE_CHANGE_R` (§2a). Until then `effective_mask()`
keeps trainers safe and the pinning test keeps the defect visible.
