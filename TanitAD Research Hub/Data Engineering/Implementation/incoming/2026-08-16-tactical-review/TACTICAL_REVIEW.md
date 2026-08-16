# Tactical labels — the three mapping defects are FIXED, the echo vote is RETIRED, and the PI has a visual sheet

**Acts on:** `…/incoming/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md`.
**Deliverable for the PI:** `review/TACTICAL_VISUAL_REVIEW.html` — **40 clips, 240 real camera
frames + 40 metric BEVs, read at the 2.0 s horizon, LAT and LON judged separately.**

> ## ⛔ ESCALATION — TWO THINGS NEED A DECISION, NEITHER IS A CODE CHANGE I MAY MAKE
>
> 1. **The corpus MUST be re-fused before any tactical label is used.** The banked aug120 records
>    still carry the pre-fix tokens. **MEASURED: the fix changes 17.41 % of lateral and 47.76 % of
>    longitudinal labels** (n=201). I did **not** re-fuse — it is escalated separately and now has
>    **four** independent reasons pending (SAM3 backfill · the `_provenance.vlm` mis-stamp · these
>    mapping fixes · the retired vote).
> 2. **`a_tac_lon` has ZERO independent corroboration anywhere in the corpus — 0/201.** This is not
>    a bug I introduced; it is what was always true and the old vote concealed. See §3.

---

## 1. Priority 1 — the three defects, fixed at the MECHANISM

`stack/scripts/ph1_fuse.py`. All three came from **one** mechanism: two ordered tuples of
`(substring, token)` scanned with `sub in text.lower()`. Substring matching **cannot tell a verb
from a verb that contains it, nor a token from a token mentioned in prose.** Both source
vocabularies are CLOSED, so each mapping is now a **TOTAL dict over its closed key set**, and the
lookups **raise `UnmappedActionVerb`** on a key they do not know.

### 1.1 What the replay measures — MEASURED, negative control included

`code/tacfix_m1_before_after.py` replays the **pre-fix** rules (carried verbatim as a negative
control, since they were deleted from the module) and the **post-fix** ones over the *same* inputs
the production fuse saw: the 201 banked fused records and their own v2 source. **It re-fuses
nothing.** Artifacts: `raw/m1_before_after.json`, `raw/m1_before_after_per_clip.jsonl`.

| VLM emission | n | OLD lat | OLD lon | **NEW lat** | **NEW lon** |
|---|---|---|---|---|---|
| `hold_corridor_none` | 136 | LANE_KEEP | ⛔ HOLD | LANE_KEEP | **— (no claim)** |
| `reduce_to_none` | 31 | ⛔ none | ⛔ none | — | **BRAKE_TO** |
| `prepare_lane_change_left` | 28 | LANE_CHANGE_L | — | LANE_CHANGE_L | — |
| `prepare_lane_change_right` | 26 | LANE_CHANGE_R | — | LANE_CHANGE_R | — |
| `hold_corridor_right` | 17 | LANE_KEEP | ⛔ HOLD | LANE_KEEP | **—** |
| `reduce_to_left` | 10 | ⛔ none | ⛔ none | — | **BRAKE_TO** |
| `reduce_to_right` | 8 | ⛔ none | ⛔ none | — | **BRAKE_TO** |
| `prepare_stop_none` | 6 | — | BRAKE_TO | — | BRAKE_TO |
| `hold_corridor_left` | 6 | LANE_KEEP | ⛔ HOLD | LANE_KEEP | **—** |
| `prepare_exit_left` | 1 | — | — | — | — *(declared)* |
| `resume_cruise_none` | 1 | — | CRUISE | — | CRUISE |

**Defect 1 — `reduce_to` mapped to nothing.** The VLM's only deceleration verb: **49/270 = 18.15 %**
of emissions silently dropped. Counting `prepare_exit_left`, **50/270 = 18.52 %** were silent on
*both* axes before; **1/270 = 0.37 %** after, and that one is a **declared** abstain.
✅ Fixed — reproduces `TACTICAL_LABEL_VALIDATION.md` §1.3's 49/270 exactly.

**Defect 2 — `hold_corridor` emitted a LONGITUDINAL token.** A lateral verb matching the LON
substring `"hold"`: **159/270 = 58.89 %** of emissions. ✅ Fixed — it now casts **no** longitudinal
vote. Reproduces the doc's 159/270 exactly.

**Defect 3 — the field was an ACTION wearing a GOAL's name.** `g_tac_lat`/`g_tac_lon` were filled
from `TACTICAL_LAT_ACTIONS`/`TACTICAL_LON_ACTIONS`. The vocabularies are **disjoint** — `LANE_KEEP`
is not in `TACTICAL_GOAL_TOKENS_LAT` (`ANCHOR_GOAL`, `CORRIDOR_OFFSET`, `EVADE_IN_CORRIDOR`,
`LAT_UNCONSTRAINED`). ✅ Fixed by **renaming the field to what it holds**: `a_tac_lat`/`a_tac_lon`.
The `g_tac_*` keys are **not dropped** — they are emitted with `token: null` and an
`unavailable_reason`, because silently deleting them would send the next consumer to the action
field by mistake. Nothing in this fuse derives goal tokens; they need the Alpamayo reason.

### 1.2 ⭐ A FOURTH defect, found while fixing the third — the reasoning could out-vote the axis

The Alpamayo leg ran the same substrings over `json.dumps(alp["meta_action"])[:400]` — **a blob
containing the free-text `cot` rationale**, truncated mid-record. So the *reason* could cast the
*axis's* vote. Replaced by `parse_alpamayo_axes()`, which reads the three labelled lines.

**MEASURED (n=201):** the old Alpamayo longitudinal vote spoke on **145** clips and was decided by
text that is **NOT the Longitudinal axis line** on **5/145 = 3.45 %**. Small, and I will not inflate
it — but every one is a label decided by prose. Verbatim example, `bf9bc0e3…`:

| axis says | old vote (from the blob) | vote from the axis line alone | the `cot` that hijacked it |
|---|---|---|---|
| `Gentle Acceleration` | **BRAKE_TO** | CRUISE | *"Stop to yield to the cross-traffic bus crossing ahead."* |

The Alpamayo **lateral** token changed on **18/201 = 8.96 %** of clips.

### 1.3 What is REPORTED, never edited — ⛔ no v6 vocabulary tuple was touched

| gap | scale | why it is reported, not patched |
|---|---|---|
| `Turn Left`/`Turn Right` have **no** `TACTICAL_LAT_ACTIONS` member | **186/4,729 = 3.93 %** | `NUDGE_L` for a left turn would be a fabrication. Shown on the sheet as its own stratum. |
| `reduce_to` and `prepare_stop` **collapse** onto `BRAKE_TO` | 55/270 emissions | `TACTICAL_LON_ACTIONS` has one deceleration token. |
| Alpamayo's LON axis is **MAGNITUDE**-typed, v6's is **REASON**-typed | 4,729 | Every value maps to `REASON_REQUIRED`; the leg casts no LON vote. §2 shows why this is not conservatism. |

The tuples size embedding tables (`v6.py:3297-3298`) and a shape change breaks the live 30k v6F
strict resume on Thor. **Report, never edit** — honoured.

### 1.4 Tests

`stack/tests/test_ph1_fuse.py` — **44 pass**. Seven new, plus the inverted pin:

- ⚠️ **`test_vocab_tokens_come_from_the_real_lists` had its pin INVERTED, not deleted.** It used to
  assert `vocab["g_tac_lat"]["token"] in TACTICAL_LAT_ACTIONS` — pinning a goal-named field holding
  an action token. The docstring now records why the pin changed; the same content is pinned under
  `a_tac_*`, the goal keys are pinned empty-with-a-reason, and a **negative control** asserts the
  emitted action token is *not* a member of `TACTICAL_GOAL_TOKENS_LAT`.
- `test_no_known_verb_maps_to_nothing_and_unknown_ones_raise` — **the mechanism test the brief
  asked for**: the table is total over `ph0_v2.ACTION_VERBS`; the only verb silent on both axes is
  the one that *declares* it; an unknown verb and an unknown direction both raise.
- `test_reduce_to_the_only_deceleration_verb_now_lands` / `…hold_corridor_is_lateral…` — each
  carries the **pre-fix rules as a negative control**, so the test proves the old mechanism really
  did fail rather than asserting the new one in a vacuum.
- `test_alpamayo_axes_are_parsed_and_the_reasoning_cannot_vote` — an adversarial `cot` saying
  *"Stop and yield … then follow it"* against an axis saying `Maintain Speed`.
- `test_alpamayo_turn_has_no_v6_token_and_is_never_bent_into_a_neighbour`.
- `test_ego_and_vlm_are_ONE_block_and_can_never_corroborate_each_other` + the end-to-end
  `test_two_of_three_can_no_longer_be_satisfied_by_ego_plus_vlm`.

**Full suite — my invocation, quoted:** `cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 pytest -q` →
**3675 passed, 2 failed, 7 skipped, 2 xfailed** (356 s).
⚠️ **The 2 failures are NOT mine**, re-run in isolation to confirm:
`test_runbook_commands.py::test_the_static_closure_matches_the_runtime_one` and
`::test_step_zero_md5_list_is_the_real_import_closure`, both failing on
`Missing: ['s2_labels']` — `train_v6_staged.py` now imports `s2_labels.py` and the runbook's md5
list has not caught up. Both files belong to the **integration agent**. `ph1_fuse` is not in that
closure. (Pass count exceeds the 3658 baseline because three agents are adding tests concurrently.)

---

## 2. ⭐ AN INDEPENDENT MEASUREMENT THAT VINDICATES THE `REASON_REQUIRED` DECISION

Leaving Alpamayo's longitudinal axis unmapped looks over-cautious. It is not — **the magnitude
axis and the reason disagree about direction on the clearest class in the taxonomy.**

**MEASURED**, all 201 aug120 clips with poses, at the 2.0 s horizon:

| Alpamayo LON | n | v(t0) | v(+2.0 s) | **Δv** |
|---|---|---|---|---|
| Gentle Acceleration | 44 | 9.29 | 10.17 | +0.88 |
| Gentle Deceleration | 72 | 10.21 | 9.60 | −0.61 |
| Maintain Speed | 58 | 15.80 | 16.01 | +0.21 |
| Strong Acceleration | 8 | 6.80 | 8.95 | +2.15 |
| Strong Deceleration | 11 | 5.65 | 4.60 | −1.06 |
| ⛔ **Stop** | **8** | **0.51** | **2.95** | **+2.44** |

Every class moves the way its name says — **except `Stop`, which is the one that moves fastest in
the *opposite* direction.** Reading the eight `cot` strings settles it: **`Stop` is a STATE ("is
stopped"), not an ACTION ("is stopping")** — the ego is already stationary at t0 (mean 0.51 m/s) and
launching by the horizon. Two of the eight say verbatim *"Resume speed from stop since the traffic
light turns green."*

⇒ **A naive `Stop → BRAKE_TO` map would be wrong on the majority of these clips.** The magnitude
axis genuinely cannot be typed without the reason. **NEW — not in `TACTICAL_LABEL_VALIDATION.md`.**
It is surfaced on the sheet with a direct request for the PI's adjudication.

---

## 3. Priority 2 — the vote could be carried by a source and its own echo. It no longer can.

**The dependence, MEASURED and STRUCTURAL** (`TACTICAL_LABEL_VALIDATION.md` §1.4):
`_ego_prompt_mode == 'past'` on **201/201**, and the ego block printed into the VLM's prompt carries
`motion` and `turning` — **exactly and only** the two fields the ego voter reads. Signature:
VLM↔ego LAT **κ 0.7608** vs Alpamayo↔VLM **κ 0.1717**.

**Where it is implemented:** `ph1_fuse.emit_vocab`, in a `majority()` helper that counted **voters**,
not **sources**.

### The choice, and why

Of the three options in my brief, I took **collapse the two into one vote** — implemented as
**independence blocks**. `ego` and `vlm` are ONE block; it casts at most one vote, and only when its
members agree (an internal disagreement is a *finding* — the VLM departing from its own prompt — so
it is recorded and the block abstains). Alpamayo is the other block. **With two blocks there is no
"2 of 3" left to satisfy.** `majority()` is **deleted**, not reweighted, and a test asserts it is
gone by name.

I rejected *"require Alpamayo to be one of the two"* because Alpamayo is silent on whole axes, which
would delete labels rather than qualify them; and *"weight them"* because a weight is a knob that
drifts, where a block is structural.

Each emission now carries **`corroborated`** = *≥2 INDEPENDENT blocks spoke and agreed*. ⛔ It can
never be True from {ego, vlm} alone. Ties go to **Alpamayo** (the only leg that saw the full camera
rig and the only one external to this pipeline), with the other block's opinion recorded.

### The consequence on the corpus — MEASURED, n = 201

| | LAT | LON |
|---|---|---|
| old majority **decided** by ≥2 voters all inside {ego, vlm} | **6 (2.99 %)** | 1 (0.50 %) |
| old majority's **margin inflated** by ego+vlm both among the winners | **135 (67.16 %)** | 1 (0.50 %) |
| token **unchanged** by the whole fix | 166 (82.59 %) | 105 (52.24 %) |
| new provenance = `alpamayo` | 183 | **0** |
| new provenance = `ego+vlm` block | 5 | 121 |
| new token `null` (abstain) | 13 | 80 |
| ⭐ **`corroborated` (≥2 independent blocks agreeing)** | **115 (57.21 %)** | ⛔ **0 (0.00 %)** |

**Read the two lateral rows together and state both.** The echo *decided* only 6 labels — I will not
inflate that. But on **135 of 201 clips (67.16 %)** the majority's margin was manufactured by
counting one source twice, so the *confidence* was inflated corpus-wide even where the token was
right.

> ### ⛔ AND THE LONGITUDINAL ROW IS THE REAL FINDING: `a_tac_lon` IS CORROBORATED ON **0 of 201** CLIPS.
>
> Every longitudinal tactical label in the corpus comes from the `ego+vlm` block **alone** — because
> Alpamayo's magnitude axis cannot be typed into v6's reason-typed vocabulary (§1.3, §2). Under the
> old vote this was invisible: the constant `HOLD` and the ego's `CRUISE`/`BRAKE_TO` looked like
> independent voters. **The vote was not hiding a weak signal; it was hiding that there is no second
> opinion at all.** ⇒ `a_tac_lon` is **not trainable** until the Alpamayo *meta-action × reason*
> builder exists (`TACTICAL_LABEL_VALIDATION.md` §5.3, still specification-only).

---

## 4. Priority 3 — the visual review sheet

`review/TACTICAL_VISUAL_REVIEW.html` · **5.03 MB** · generator `code/tacrev_visual_review.py` ·
selection record `raw/tacrev_selection.json`.

**Verified to actually render** (the PI's complaint about the first strategic sheet was *"I dont see
any clips or visual elements"*). Served locally and inspected in-browser:
**280/280 images decoded, 0 broken** — 240 camera frames at their true 640×256 cylindrical geometry
and 40 BEV PNGs; 40 cards; 40 `cot` blocks; 240 leg panels; 120 LAT + 120 LON radios. The
export round-trip was exercised and the test verdict cleared, so the PI inherits a clean sheet.

| requirement | how it is met |
|---|---|
| **Read at 2.0 s** | Frames span `t0−1.0 s → t0+2.0 s` (indices 70/80/85/90/95/100 at 10 Hz); the **t0** and **READ** frames are outlined blue. The horizon is printed in the title, the header and the export JSON. Justified on-sheet by the κ peak (LON 0.188→0.3655, LAT 0.290→0.4694). |
| **LAT and LON separate, never pooled** | Two panels, **two independent verdict controls** per clip. A clip can be right on one axis and wrong on the other and one button could not say so. |
| **Three legs side by side** | Each leg shows its **native** output, the v6 token, and the 3-band projection — so the PI can see *which* leg is wrong. |
| **The dependence is visible** | The VLM and ego panels are drawn with a dashed amber border and a `not independent` superscript, with the κ evidence in the header. |
| **Alpamayo `cot` shown** | Verbatim, in a quoted block on every card. |
| **`Turn Left/Right` gap** | Its own stratum `D_UNREPRESENTABLE`, 4 clips, with `— no token` rendered in red. |
| **Stratified, labelled** | Table below; the stratum is on every card and in the export. |
| **Verdicts export JSON** | Per-axis verdict + note, localStorage-persisted, one-click export. ⚠️ Unreviewed rows export as **`null`** — the S2 lesson that *`null` is UNREVIEWED, not agreement*. |

### The strata — stated, with the pool they were drawn from

| stratum | in pool | shown | what it is |
|---|---|---|---|
| `D_UNREPRESENTABLE` | 4 | **4 (all)** | Alpamayo says `Turn Left/Right` — v6 has no such lateral token |
| `E_STOP_ROW_ONE_AXIS` | 3 | **3 (all)** | a stopped vehicle emits one axis only — includes the §2 `Stop` anomaly |
| `C_THREE_WAY_SPLIT` | 2 | **2 (all)** | all three legs speak, all three differ |
| `B_TWO_OF_THREE` | 63 | 18 | exactly two agree — ⚠️ if the two are VLM+ego, that is the echo |
| `A_UNANIMOUS` | 25 | 12 | all three agree — included so the sheet cannot be accused of showing only failures |
| `F_INSUFFICIENT` | 1 | 1 | fewer than two legs spoke — availability, not disagreement |

Selection is **deterministic** (sorted, no RNG). ⚠️ **Only 2 three-way splits exist in the whole
pool** — a consequence of the fix itself: with the VLM's longitudinal leg no longer a constant and
Alpamayo's LON leg abstaining, far fewer clips have three speaking legs to split.

### ⚠️ The pool bound, stated plainly

The pool is **98 clips**: those with a **local** w120 mp4 *and* an ego npz. Of the 4,729
Alpamayo-labelled clips only **201** have our video at all, and the other **4,528 cannot be reviewed
visually by any means available today** — the missing artifact is our video, not the labels
(`TACTICAL_LABEL_VALIDATION.md` §5.5). Extending the pool from 98 to all 201 needs a **≈275 MB**
CPU-only pull from `Sayood/tanitad-ph0-aug120`; **I did not download it** — 98 clips already covers
every stratum, and three strata are shown in full.

⚠️ **The 3-band projection used for stratification discards severity** (`Sharp Steer` and `Steer`
collapse), so any agreement figure computed on it is an **UPPER BOUND**. It decides which clips are
shown, never what a card claims: every card prints each leg's native opinion.

---

## 5. Evidence classes

| claim | class | n |
|---|---|---|
| `reduce_to` silent on both axes pre-fix, 49/270 = 18.15 % | **MEASURED** (replay) | 270 emissions |
| `hold_corridor` → `HOLD` on 159/270 = 58.89 % | **MEASURED** (replay) | 270 emissions |
| `g_tac_*` held `a_tac` tokens; vocabularies disjoint | **MEASURED** (read from `v6.py:172-178, 217-223`) | — |
| old Alpamayo LON vote decided off-axis on 5/145 = 3.45 % | **MEASURED** (replay) | 201 |
| Alpamayo LAT token changed on 18/201 = 8.96 % | **MEASURED** (replay) | 201 |
| fix changes 17.41 % of LAT and 47.76 % of LON labels | **MEASURED** (replay) | 201 |
| echo **decided** 6/201 LAT; **inflated the margin** on 135/201 | **MEASURED** (replay) | 201 |
| ⛔ `a_tac_lon` corroborated on **0/201** | **MEASURED** (replay) | 201 |
| `a_tac_lat` corroborated on 115/201 = 57.21 % | **MEASURED** (replay) | 201 |
| ⭐ `Stop` is a STATE: v(t0) 0.51 → v(+2 s) 2.95, Δv **+2.44** | **MEASURED** (ego poses) | 8 |
| yaw sign: **positive Δyaw = LEFT** | **MEASURED**, not assumed (Sharp Steer Left +0.1963 / Right −0.5371) | 201 |
| 186 clips have no v6 lateral token for `Turn *` | **MEASURED** (a1 taxonomy) | 4,729 |
| κ peaks at H = 2.0 s on both axes | **MEASURED** (INHERITED from a4 sweep, not re-run here) | 201 |
| VLM↔ego κ 0.7608 vs Alpamayo↔VLM 0.1717 | **MEASURED** (INHERITED from a3, not re-run here) | 193 / 185 |
| correctness of any tactical label vs a human | ⛔ **STILL UNMEASURED — the sheet exists to change this** | 0 |

⚠️ **No interval is quoted in this document.** Every number here is a **count over a complete
enumeration** (201 or 4,729 clips, 270 emissions), not a sample estimate — a CI would be a category
error. When the PI's verdicts come back, the error rate they imply **is** a sample statistic and must
carry a **paired episode-cluster bootstrap** (`taniteval/ci.py`) — ⛔ never `overlapping_holdout_se`.
⚠️ **No eval-tier stamp appears here** because nothing in this document is a model capability claim;
these are **label-derivation** measurements. Any downstream tactical *capability* number is **T1**.

---

## 6. What I did NOT do, stated plainly

1. ⛔ **Did not re-fuse the corpus.** Escalated (headline). The banked records still carry pre-fix
   tokens and the old `g_tac_*` field name.
2. ⛔ **Did not change any v6 vocabulary tuple.** Three real gaps are reported in §1.3.
3. ⛔ **Did not touch** `ph0_sam3.py`, `train_v6_staged.py`, `s2_labels.py`, `taniteval/`,
   `MODEL_REGISTRY.md`, or the two sibling `incoming/` directories.
4. ⚠️ **Did not edit `…/2026-08-16-tactical-labels/code/tac_a3_three_leg_agreement.py`**, which
   copies the pre-fix `LAT_RULES`/`LON_RULES` verbatim. That is deliberate: it is the record of the
   **pre-fix** measurement, and "fixing" it would silently change what it measured.
5. ⚠️ **Did not download** the ~275 MB of remaining aug120 video (§4).
6. ⚠️ **`colab/nb_build.py:240` still says the fusion is "2-of-3 voting"** — now stale. Not my file;
   flagged rather than edited.
7. ⚠️ **UNVERIFIED:** post-fix VLM↔ego longitudinal κ. The 0.0000 was an artifact of the constant
   the mapping bugs created; with the VLM's LON leg live, the LON dependence has **not** been
   re-measured. The block grouping does not depend on it (it rests on the shared *input*, MEASURED
   201/201), but the number should be re-taken after the re-fuse.

---

## 7. Deliverable manifest

⚠️ **Everything is in the repo and staged. Nothing lives only on a pod, only in a worktree, or only
in a scratchpad.**

| artifact | location | only copy? |
|---|---|---|
| the mapping + vote fix | `repo:stack/scripts/ph1_fuse.py` (modified) | no — git |
| tests (44 pass; 7 new + 1 inverted pin) | `repo:stack/tests/test_ph1_fuse.py` (modified) | no — git |
| this document | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-16-tactical-review/TACTICAL_REVIEW.md` | no |
| **⭐ the PI's visual sheet (5.03 MB, 280 images)** | `repo:…/2026-08-16-tactical-review/review/TACTICAL_VISUAL_REVIEW.html` | no |
| sheet generator | `repo:…/2026-08-16-tactical-review/code/tacrev_visual_review.py` | no |
| before/after replay harness | `repo:…/2026-08-16-tactical-review/code/tacfix_m1_before_after.py` | no |
| before/after results | `repo:…/2026-08-16-tactical-review/raw/m1_before_after.json` | no |
| before/after per-clip (201) | `repo:…/2026-08-16-tactical-review/raw/m1_before_after_per_clip.jsonl` | no |
| sheet selection + strata record | `repo:…/2026-08-16-tactical-review/raw/tacrev_selection.json` | no |

**Inputs consumed, NOT copied into the repo** (large, and owned by other streams): the banked fused
aug120 corpus, `merged/ph0_v2.json`, the 201 ego npz and the local mp4 cache — all under the session
scratchpad and HF-backed (`Sayood/tanitad-ph0-aug120`). The tactical label JSONLs
(`a1_…_per_clip.jsonl`) are already in `repo:…/2026-08-16-tactical-labels/raw/`.

**How the PI uses the sheet:** open
`…/2026-08-16-tactical-review/review/TACTICAL_VISUAL_REVIEW.html` in any browser (it is fully
self-contained — no server, no network), judge **LATERAL** and **LONGITUDINAL** separately on each
of the 40 clips, then press **Export verdicts JSON** and return the text. Progress is saved in the
browser, so it can be done in several sittings.
