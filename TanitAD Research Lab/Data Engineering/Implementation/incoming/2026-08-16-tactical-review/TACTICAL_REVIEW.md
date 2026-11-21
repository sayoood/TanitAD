# Tactical labels — the three mapping defects are FIXED, the echo vote is RETIRED, and the PI has a visual sheet

**Acts on:** `…/incoming/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md`.

> ## ⛔ THE SHEET THIS DOCUMENT SHIPPED WAS BUILT AT THE WRONG HORIZON — REBUILT 2026-08-17 (C89)
>
> ~~**Deliverable for the PI:** `review/TACTICAL_VISUAL_REVIEW.html` — **40 clips, 240 real camera
> frames + 40 metric BEVs, read at the 2.0 s horizon, LAT and LON judged separately.**~~
>
> **2.0 s is the SEAM**, where `OP_BAND_S` ends and `TAC_BAND_S` begins — not the tactical band.
> The binding spec is `TAC_BAND_S = (2.0, 6.0)` at `stack/tanitad/models/v6.py:136-140`. It was
> picked as the **argmax of κ** and then described as "the v6 tactical band".
>
> ⭐ **CURRENT DELIVERABLE:** `review/TACTICAL_VISUAL_REVIEW_BAND_2_6S.html` — **42 clips, 294 real
> camera frames + 42 metric BEVs, read OVER the (2.0, 6.0] s band**, LAT and LON judged separately.
> Same design in every other respect, deliberately, so the two passes stay comparable.
> The 2.0 s sheet is **kept beside it, banner-marked SUPERSEDED**, and stores its verdicts under a
> **different localStorage key** so the two can never contaminate each other.
> ⚠️ Every κ in §§2–4 below that is quoted "at 2.0 s" is a **SEAM** value. Band restatement:
> `TACTICAL_LABEL_VALIDATION.md` **§3.1** and `raw/b1_band_agreement.json`.

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

**MEASURED**, all 201 aug120 clips with poses, ⚠️ at the 2.0 s **SEAM** (i.e. over `OP_BAND_S`):

| Alpamayo LON | n | v(t0) | v(+2.0 s) | **Δv** |
|---|---|---|---|---|
| Gentle Acceleration | 44 | 9.29 | 10.17 | +0.88 |
| Gentle Deceleration | 72 | 10.21 | 9.60 | −0.61 |
| Maintain Speed | 58 | 15.80 | 16.01 | +0.21 |
| Strong Acceleration | 8 | 6.80 | 8.95 | +2.15 |
| Strong Deceleration | 11 | 5.65 | 4.60 | −1.06 |
| ⛔ **Stop** | **8** | **0.51** | **2.95** | **+2.44** |

⭐ **THE SAME TABLE OVER `TAC_BAND_S` (2.0, 6.0] — added 2026-08-17, and it changes the finding.**
**MEASURED**, same 201 clips, same poses (`raw/b1_band_agreement_per_clip.jsonl`):

| Alpamayo LON | n | v(t0) | v(+2 s) | v(+6 s) | **net over band** | **mean over band** |
|---|---|---|---|---|---|---|
| Gentle Acceleration | 44 | 9.29 | 10.17 | 10.44 | +0.27 | +0.32 |
| Gentle Deceleration | 72 | 10.21 | 9.60 | 9.25 | −0.36 | −0.32 |
| Maintain Speed | 58 | 15.80 | 16.01 | 16.04 | +0.03 | +0.01 |
| Strong Acceleration | 8 | 6.80 | 8.95 | 10.75 | +1.80 | +1.18 |
| ⛔ **Strong Deceleration** | **11** | 5.65 | 4.60 | 5.76 | **+1.17** | **+0.42** |
| ⛔ **Stop** | **8** | **0.51** | **2.95** | **5.82** | **+2.87** | **+1.82** |

⛔ **`Strong Deceleration` REVERSES SIGN between the seam and the band** (−1.06 → **+1.17** net).
It is the `Stop` pathology a second time: **the braking happens inside the OPERATIVE band and the
recovery happens inside the TACTICAL band**, so a label named for the decision at t0 describes the
*opposite* of what the ego does across 2–6 s. ⇒ **Two of the six longitudinal classes (19/201
clips) now point the wrong way at the band.**

⚠️ **And the mechanism behind the κ collapse is visible here:** over the band, four of six classes
move by **less than 0.5 m/s** on average. Against the production threshold (1.0 m/s) they all fall
into `maintain`, which is exactly why band LON agreement is 0.408 and κ 0.1428 (§3.1 of
`TACTICAL_LABEL_VALIDATION.md`). **The band signal is not merely noisier — for most classes it is
below the decision threshold entirely.**

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

⭐ **CURRENT (band):** `review/TACTICAL_VISUAL_REVIEW_BAND_2_6S.html` · **5.76 MB** · generator
`code/tacrev_visual_review_band.py` · selection record `raw/tacrev_selection_band.json`.
⚠️ **SUPERSEDED (seam):** `review/TACTICAL_VISUAL_REVIEW.html` · 5.03 MB ·
`code/tacrev_visual_review.py` · `raw/tacrev_selection.json` — kept, banner-marked by
`code/mark_seam_sheet_superseded.py`. **Do not adjudicate it.**

**Verified to actually render** (the PI's complaint about the first strategic sheet was *"I dont see
any clips or visual elements"*). Served locally and inspected in-browser.

*Seam sheet (as shipped 2026-08-16):* **280/280 images decoded, 0 broken** — 240 camera frames at
their true 640×256 cylindrical geometry and 40 BEV PNGs; 40 cards; 40 `cot` blocks; 240 leg panels;
120 LAT + 120 LON radios.

⭐ *Band sheet (2026-08-17):* **336/336 images decoded, 0 broken** — **294** camera frames at
640×256 and **42** BEV PNGs; **42** cards, all 7 frames each; 42 `cot` blocks; 42 LAT + 42 LON
verdict groups; **0 render failures**, **0 clips dropped** for a short pose track.
⛔ **The frames were verified to be the RIGHT frames, not merely present:** every embedded JPEG on
the first card is **byte-identical (md5)** to a fresh `ffmpeg` extraction at indices
70/80/100/110/120/130/140. All 98 pool videos carry ≥199 frames, so index 140 (+6.0 s) is real
data, never a clamp.
The export round-trip was exercised — a partially-judged clip exports `lat:"wrong", lon:null`, an
untouched clip exports `lat:null, lon:null` — and the test verdict was **cleared from
localStorage**, so the PI inherits a clean sheet reading **0 / 42 clips judged**.

| requirement | how it is met |
|---|---|
| ~~**Read at 2.0 s**~~ ⛔ **WRONG HORIZON (C89)** | ~~Frames span `t0−1.0 s → t0+2.0 s` (indices 70/80/85/90/95/100 at 10 Hz); the **t0** and **READ** frames are outlined blue. Justified on-sheet by the κ peak (LON 0.188→0.3655, LAT 0.290→0.4694).~~ **The κ peak is not a justification — it is the defect.** A band is a **spec lookup**, never an argmax. |
| ⭐ **Read OVER the band (2.0, 6.0]** *(rebuilt sheet)* | Frames span `t0−1.0 s → t0+6.0 s` (indices **70/80/100/110/120/130/140** at 10 Hz); the five in-band frames are outlined blue and captioned **BAND START / in band / BAND END**. The BEV draws the operative slice thin grey and the **tactical band bold**. The band, its spec source (`v6.py:136-140`), and the ego statistic are printed in the banner, the title, the header and the export JSON. Ego leg = **`mean_band`** (interval, not endpoint) at the **PRODUCTION** thresholds. |
| **LAT and LON separate, never pooled** | Two panels, **two independent verdict controls** per clip. A clip can be right on one axis and wrong on the other and one button could not say so. |
| **Three legs side by side** | Each leg shows its **native** output, the v6 token, and the 3-band projection — so the PI can see *which* leg is wrong. |
| **The dependence is visible** | The VLM and ego panels are drawn with a dashed amber border and a `not independent` superscript, with the κ evidence in the header. |
| **Alpamayo `cot` shown** | Verbatim, in a quoted block on every card. |
| **`Turn Left/Right` gap** | Its own stratum `D_UNREPRESENTABLE`, 4 clips, with `— no token` rendered in red. |
| **Stratified, labelled** | Table below; the stratum is on every card and in the export. |
| **Verdicts export JSON** | Per-axis verdict + note, localStorage-persisted, one-click export. ⚠️ Unreviewed rows export as **`null`** — the S2 lesson that *`null` is UNREVIEWED, not agreement*. |

### The strata — stated, with the pool they were drawn from

Same 98-clip pool both times. ⚠️ **The strata are not identical across the two sheets, and that is
a finding, not a bug:** the ego leg is horizon-dependent, so re-reading it over the band moves
clips between strata.

| stratum | in pool ⚠️ *seam* | shown *seam* | ⭐ in pool *band* | ⭐ shown *band* | what it is |
|---|---|---|---|---|---|
| `D_UNREPRESENTABLE` | 4 | **4 (all)** | 4 | **4 (all)** | Alpamayo says `Turn Left/Right` — v6 has no such lateral token |
| `E_STOP_ROW_ONE_AXIS` | 3 | **3 (all)** | 3 | **3 (all)** | a stopped vehicle emits one axis only — includes the §2 `Stop` anomaly |
| `C_THREE_WAY_SPLIT` | 2 | **2 (all)** | **4** | **4 (all)** | all three legs speak, all three differ |
| `B_TWO_OF_THREE` | 63 | 18 | **67** | 18 | exactly two agree — ⚠️ if the two are VLM+ego, that is the echo |
| `A_UNANIMOUS` | 25 | 12 | **19** | 12 | all three agree — included so the sheet cannot be accused of showing only failures |
| `F_INSUFFICIENT` | 1 | 1 | 1 | 1 | fewer than two legs spoke — availability, not disagreement |
| **total shown** | | **40** | | **42** | |

Selection is **deterministic** (sorted, no RNG) in both. ⭐ **Reading over the band moves the corpus
away from agreement**: unanimity falls **25 → 19** and three-way splits **double, 2 → 4**. That is
the same effect the κ drop reports, seen at the level of individual clips — and it is why the band
sheet shows **4** of the hardest cases where the seam sheet could only find 2.
**37 of the 40 seam clips reappear; on those, the ego leg's own verdict changes on 10 clips
laterally and 9 longitudinally.** ⚠️ D and C are still pool-limited, not quota-limited — the quota
asks for 6 and 12 and the pool holds only 4 each.

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
| ⭐ `Stop` is a STATE: v(t0) 0.51 → v(+2 s) 2.95, Δv **+2.44** ⚠️ seam | **MEASURED** (ego poses) | 8 |
| ⭐ `Stop` over the band: v(+2 s) 2.95 → v(+6 s) **5.82**, net **+2.87** | **MEASURED** (ego poses) | 8 |
| ⛔ `Strong Deceleration` **reverses sign** at the band: seam −1.06 → band net **+1.17** | **MEASURED** (ego poses) | 11 |
| yaw sign: **positive Δyaw = LEFT** | **MEASURED**, not assumed (Sharp Steer Left +0.1963 / Right −0.5371) | 201 |
| yaw sign holds **on the band too** (Sharp Steer L +0.1660 / R −0.1565) | **MEASURED**, re-calibrated on the band quantity | 201 |
| ⭐ **seam→band is ASYMMETRIC by turn direction**: LEFT classes **grow** (`Steer Left` +0.0161→**+0.1082**, 6.72×, n=24; `Sharp Steer Left` 1.96×, n=4), RIGHT classes **shrink** (`Steer Right` −0.0569→−0.0125, 0.22×, n=37; `Sharp Steer Right` 0.45×, n=7) | **MEASURED** (`raw/b1_band_agreement_per_clip.jsonl`) | 193 |
| ⭐ at 0.15 rad, **47 of the 61 ordinary `Steer *` clips (77 %) read `straight`** at the band under the primary `mean_band` (36/61 = 59 % under `net_band`); the two `Sharp` class MEANS clear the threshold but lose 2/4 and 4/7 clips individually | **MEASURED** — this, not "aftermath", is the LAT κ mechanism | 193 |
| ⛔ *second self-correction:* I first wrote "**all** 61 collapse" — a **class-mean fact stated as a per-clip fact**; the true figure is 47/61 | mean-to-member overreach, caught by per-clip re-check | 61 |
| ⛔ *self-corrected same day:* my first reading — *"by 2–6 s the ordinary turns are already resolved / aftermath"* — was generalised from the RIGHT classes only and is **refuted** by the LEFT ones | see `TACTICAL_LABEL_VALIDATION.md` §3.1 | — |
| left turns execute later than right turns (would explain the sign pattern) | ⚠️ **HYPOTHESIS — untested**, must not be quoted as a finding | — |
| 186 clips have no v6 lateral token for `Turn *` | **MEASURED** (a1 taxonomy) | 4,729 |
| ~~κ peaks at H = 2.0 s on both axes~~ ⛔ true but **misused** — 2.0 s is `OP_BAND_S`, not the tactical band (C89) | **MEASURED** (INHERITED from a4 sweep, not re-run here) | 201 |
| ⭐ at `TAC_BAND_S` (2.0, 6.0]: LON κ **0.1428** [0.0540, 0.2250] · LAT κ **0.1777** [0.0658, 0.2953] | **MEASURED** (ours, re-run: `code/tacrev_band_agreement.py`) | 201 / 193 |
| ⭐ paired band−seam Δκ **CI-separated on both axes** (LON −0.1843, LAT −0.1354) | **MEASURED** (paired episode-cluster bootstrap) | 201 / 193 |
| ⛔ no row of `a4_horizon_sweep.json` is the tactical band — all `t0`-anchored | **MEASURED** (read from `tac_a4_horizon_sweep.py:140-148`) | — |
| VLM↔ego κ 0.7608 vs Alpamayo↔VLM 0.1717 | **MEASURED** (INHERITED from a3, not re-run here) | 193 / 185 |
| correctness of any tactical label vs a human | ⛔ **STILL UNMEASURED — the sheet exists to change this** | 0 |

⚠️ ~~**No interval is quoted in this document.**~~ **UPDATED 2026-08-17:** that was true of the
replay counts and stays true of them — every *count* here is a **complete enumeration** (201 or
4,729 clips, 270 emissions), where a CI would be a category error. **The κ rows added today are
different: an agreement coefficient IS a sample statistic**, so each carries an **episode-cluster
bootstrap** over the 201 clips (2000 draws) and the band-vs-seam contrast carries the **paired**
version — ⛔ never `overlapping_holdout_se`. When the PI's verdicts come back, the error rate they
imply is likewise a sample statistic and takes the same paired estimator.
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
| **⭐ the PI's visual sheet — BAND (5.76 MB, 336 images, 42 clips)** | `repo:…/2026-08-16-tactical-review/review/TACTICAL_VISUAL_REVIEW_BAND_2_6S.html` | no |
| band sheet generator | `repo:…/2026-08-16-tactical-review/code/tacrev_visual_review_band.py` | no |
| ⭐ band agreement analysis (the restated κ) | `repo:…/2026-08-16-tactical-review/code/tacrev_band_agreement.py` | no |
| ⭐ band agreement results + per-clip | `repo:…/2026-08-16-tactical-review/raw/b1_band_agreement.json` · `…_per_clip.jsonl` | no |
| ⭐ band sheet selection + strata record | `repo:…/2026-08-16-tactical-review/raw/tacrev_selection_band.json` | no |
| ⚠️ SUPERSEDED seam sheet (5.03 MB, 280 images) — kept, banner-marked | `repo:…/2026-08-16-tactical-review/review/TACTICAL_VISUAL_REVIEW.html` | no |
| seam sheet generator | `repo:…/2026-08-16-tactical-review/code/tacrev_visual_review.py` | no |
| supersede-marker (idempotent) | `repo:…/2026-08-16-tactical-review/code/mark_seam_sheet_superseded.py` | no |
| before/after replay harness | `repo:…/2026-08-16-tactical-review/code/tacfix_m1_before_after.py` | no |
| before/after results | `repo:…/2026-08-16-tactical-review/raw/m1_before_after.json` | no |
| before/after per-clip (201) | `repo:…/2026-08-16-tactical-review/raw/m1_before_after_per_clip.jsonl` | no |
| sheet selection + strata record | `repo:…/2026-08-16-tactical-review/raw/tacrev_selection.json` | no |

**Inputs consumed, NOT copied into the repo** (large, and owned by other streams): the banked fused
aug120 corpus, `merged/ph0_v2.json`, the 201 ego npz and the local mp4 cache — all under the session
scratchpad and HF-backed (`Sayood/tanitad-ph0-aug120`). The tactical label JSONLs
(`a1_…_per_clip.jsonl`) are already in `repo:…/2026-08-16-tactical-labels/raw/`.

**How the PI uses the sheet:** open
`…/2026-08-16-tactical-review/review/TACTICAL_VISUAL_REVIEW_BAND_2_6S.html` in any browser (it is
fully self-contained — no server, no network), judge **LATERAL** and **LONGITUDINAL** separately on
each of the **42** clips, then press **Export verdicts JSON** and return the text. Progress is saved
in the browser, so it can be done in several sittings.

⛔ **Open the `_BAND_2_6S` file, not `TACTICAL_VISUAL_REVIEW.html`.** The latter is the superseded
2.0 s seam sheet; it opens with a red SUPERSEDED banner and its title reads `[SUPERSEDED · seam
2.0 s]`. The two use different storage keys, so judging one never affects the other.

⭐ **The question the band sheet asks is narrower than it looks.** Every card shows **+2.0 s →
+6.0 s**; the label is being judged **for that window**, not for t0. Alpamayo's meta-action is a
decision *at t0*, so a label can be right about t0 and still wrong about 2–6 s. ⚠️ **Where that is
what you see, the note field saying so is the single most valuable thing on the sheet** — it
discriminates *"the band is unlabelable"* from *"Alpamayo is the wrong source for this band"*,
which is the open question §3.1 of `TACTICAL_LABEL_VALIDATION.md` leaves standing.
