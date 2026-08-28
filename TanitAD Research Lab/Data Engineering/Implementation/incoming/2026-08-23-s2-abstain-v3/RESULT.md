# S2 labels v3 — the 80 reassigned `a_str` targets are DECLINED, not guessed

`TanitAD_DataFlyWheel · 2026-08-23 · P1 (tactical/strategic label pipeline) item (c)`

> ⚠️ Filed under `TanitAD Research Lab/` because that is the tree at `c29c659`. The rename to
> `TanitAD Research Lab/` lives UNCOMMITTED in the main checkout; move this package with it.

---

## 0. What this closes

`…/2026-08-16-s2-v1-labels/labels/SUPERSEDED.json` recorded an **open defect in its own
successor**, in its author's words:

> *"`a_str` has NO abstain token … so the 80 removed `PREPARE_LANE_CHANGE` rows were
> REASSIGNED to 71 `HOLD_CORRIDOR` + 9 `REDUCE_TO` rather than declined. `s2_labels` now
> supports a per-family abstain mask … that the NEXT label build should emit for those rows;
> **until it does, those 80 `a_str` targets are UNVERIFIED, not corrected.**"*

That next build had not happened. **MEASURED: `labels_v2` carries `a_str` abstains = 0 of 797.**
The abstain machinery in `scripts/s2_labels.py` — validation (`_check_block`), `has_abstain`,
`abstain_census`, and the optional `g_str_valid`/`a_str_valid` batch keys — is fully
implemented and had **never been exercised by an artifact**.

`labels_v3` emits it. **80 rows declined: 19 aug120 + 61 w120val**, exactly matching the count
the finder documented independently from the loader.

---

## 1. The defect, re-established by content (not by trusting the note)

Joining v1 and v2 on `clip_id`:

| split | `PREPARE_LANE_CHANGE` v1 → v2 | where v2 put them | declined in v3 |
|---|---|---|---|
| aug120 (n=201) | 19 → 0 | `HOLD_CORRIDOR` +18, `REDUCE_TO` +1 | **19** |
| w120val (n=596) | 61 → 0 | `HOLD_CORRIDOR` +53, `REDUCE_TO` +8 | **61** |
| **total (n=797)** | **80 → 0** | **71 + 9** | **80** ✅ matches the documented 80 |

The `71 HOLD_CORRIDOR + 9 REDUCE_TO` split reproduces the SUPERSEDED note exactly, from a
different direction (a per-clip join here; the loader's census there). Two independent
derivations agreeing is the evidence, not either one alone.

## 2. Why THESE rows and not all 797 — the question that decides correctness

`lane_change_requirement()` returns `None` (UNKNOWN) for **801/801** clips, so one could argue
every row is underdetermined and none should be labelled. That argument is wrong, and the
distinction is the whole point:

* On the other **717** rows the geometry showed **no lateral-displacement event at all**, and
  v1 and v2 **agree**. `HOLD_CORRIDOR` there is positively supported by the hindsight path.
* On these **80** the refuted gate **fired** — there *is* lateral displacement. The PI's
  2026-08-16 adjudication is that lateral displacement **cannot distinguish a lane change from
  road curvature in either direction**. Labelling them `HOLD_CORRIDOR` asserts *"the ego did
  NOT change lanes"* — precisely the claim the refuted geometry cannot settle. That is a
  manufactured confident claim substituted for a wrong one, and it is exactly what an abstain
  exists for.

⛔ Nothing was re-derived, re-selected, or re-ordered. `g_str` is untouched. The identity of the
80 is **read off the v1/v2 diff**, never re-inferred from geometry.

## 3. Verification

```
pytest tests/test_s2_abstain_v3.py -q                                   ->  13 passed
pytest tests/test_v6_s2_loss.py tests/test_s2_abstain_v3.py \
       tests/test_ph1_fuse.py tests/test_refc_v3_lan_preflight.py -q    -> 149 passed, 0 failed
```

Through the real loader, both splits, both versions:

| set | `has_abstain` | `abstain_census` |
|---|---|---|
| `labels_v2` aug120 / w120val | `False` / `False` | `{g_str: 0, a_str: 0}` both |
| `labels_v3` aug120 | **`True`** | `{g_str: 0, a_str: 19}` |
| `labels_v3` w120val | **`True`** | `{g_str: 0, a_str: 61}` |

⭐ **The parity guard fired on its own and confirms the corpus is untouched:**
`[parity] s2 label_split=w120val: eval split is DISJOINT from physicalai-train-e438721ae894 —
596 clips, 0 in the train split (checked by per-clip sha256, not by provenance).`

The test suite pins **correctness** (exactly the joined 80; `g_str` never abstains — its
`NONE_ABSTAIN` is a supervised target, a different claim) and **surgicality** (`g_str`
byte-identical to v2 on all 797; every non-declined `a_str` block byte-identical to v2). A
rebuild that quietly moved anything else would be the same class of defect it is fixing.

⚠️ One earlier failure was **not** a regression: `test_the_ex_lane_change_clips_are_route_FOLLOW…`
raised `FileNotFoundError` because my local working copy lacked `review/raw/`. After copying it,
149/149 pass. Recorded so "1 failed" is not read as a defect this package introduced.

## 4. ⛔ ESCALATION — the pointer flip is a PI/Master-Mind decision, not mine

`s2_labels.S2_CANONICAL_LABELS_REL` **still names `labels_v2`**, so nothing consumes v3 yet.
That is deliberate. Flipping it changes **what a live training arm is supervised by**, and
REF-C v3 is that arm. I banked the artifact and its proof; I did not repoint the programme.

**The flip is mechanical once approved** — three coordinated edits, because the path is written
in three places and a test cross-checks them against each other (C81):

1. `stack/scripts/s2_labels.py` — `S2_CANONICAL_LABELS_REL` →
   `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-s2-abstain-v3/labels_v3`
2. `…/2026-08-16-s2-v1-labels/labels/SUPERSEDED.json` — `superseded_by` must resolve to the
   same directory (pinned by `test_v6_s2_loss.py:839`), and a **new** `SUPERSEDED.json` goes
   into `labels_v2` naming v3, so the loader refuses v2 by name the way it already refuses v1.
3. `stack/tests/test_v6_s2_loss.py:801` — the assertion `"labels_v2" in msg` is **name-pinned**
   and will fail; it becomes `labels_v3`.

`tests/test_s2_abstain_v3.py::test_v3_is_not_yet_the_canonical_set` deliberately pins the
CURRENT state and will fail on the flip — by design, so the pointer cannot move without
someone reading this rationale.

**Safety of the flip is not a guess — it is the design.** `s2_labels.py:63-72`: the family
masks are *default-off and provably inert*; `v6_loss_step` reads `s2_valid & <family>_valid`
when present and falls back to `s2_valid` when absent, so a pre-abstain artifact yields a
bit-identical loss. With v3 the only change is that **80 fabricated `a_str` CE targets stop
sending gradient**. Both branches are exercised by
`test_v6_s2_loss.py::test_the_family_masks_are_ABSENT_by_default_and_only_ever_REMOVE` and by
`test_s2_abstain_v3.py::test_v2_emits_no_family_masks_and_v3_does`.

**Recommendation: approve the flip.** The incumbent set trains the strategic action head on 80
targets its own provenance file calls UNVERIFIED, and the replacement's only effect is to stop
supervising them.

## 5. What remains open in this area (named, not buried)

* `a_str` still has **no abstain TOKEN**, and that was refused deliberately —
  `GoalVocabulary` sizes its embedding table from `STRATEGIC_ACTION_TOKENS` and the live v6F
  S-W run resumes tensor-level. The mask is the correct mechanism; this is not a gap.
* The 80 declined rows are **unlabelled, not resolved**. Recovering a real `a_str` for them
  needs a lane context PhysicalAI-AV does not ship (`n_lanes_same_direction`, `ego_lane_idx`,
  `route_lane_idx`, `lane_continues`). That is a data-acquisition item, not a labelling one.
* `labels_v3` covers **aug120 + w120val (797 records)** — *not* the parity train corpus, which
  has no S2 labels at all. Stated so the coverage is not overread.

## 6. Deliverable manifest

⛔ Every row resolved with `git ls-files --cached` after staging (C78).

| artifact | path |
|---|---|
| this report | `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-s2-abstain-v3/RESULT.md` |
| labels v3 — aug120 (201 rec, 19 declined) | `.../2026-08-23-s2-abstain-v3/labels_v3/s2_labels_aug120.jsonl` |
| labels v3 — w120val (596 rec, 61 declined) | `.../2026-08-23-s2-abstain-v3/labels_v3/s2_labels_w120val.jsonl` |
| clip index (carried over unchanged) | `.../2026-08-23-s2-abstain-v3/labels_v3/clip_index.json` |
| abstain provenance (the 80 clip_ids + the v2 tokens they replace) | `.../2026-08-23-s2-abstain-v3/labels_v3/_abstain_provenance.json` |
| the builder | `.../2026-08-23-s2-abstain-v3/code/s2_emit_abstain_v3.py` |
| regression tests (13) | `stack/tests/test_s2_abstain_v3.py` |

**Staged, never committed, never pushed.**
