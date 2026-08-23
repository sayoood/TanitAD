# COMMS — transfer note: v6.1 vocabulary affects MODEL TENSOR SHAPES

`From: TanitAD_DataFlyWheel · 2026-08-23`
`To: TanitAD Master Mind (owns model design) + TanitAD_TrainingFlyWheel (owns run configs)`
`Trigger: PI 2026-08-23 — "if turning right and left is missing in the vocab, let
add it and inform the master mind/training agent because it is affecting the
tensors of the models, which we will train in the future."`

## 1. What changed, and why you must know before the next run

Two vocabulary tuples in `stack/tanitad/models/v6.py` gained **two tokens each**.
Both tuples **SIZE LIVE EMBEDDING TABLES**, so this is a tensor-shape fact, not a
labelling detail.

| tuple | v6.0 | v6.1 | appended |
|---|---|---|---|
| `TACTICAL_LAT_ACTIONS` | 6 | 8 | `TURN_L`, `TURN_R` |
| `STRATEGIC_ACTION_TOKENS` | 6 | 8 | `PREPARE_TURN_L`, `PREPARE_TURN_R` |

Affected tensors: `GoalVocabulary(...).table.weight` → `(n_tokens, d_goal_embed)`
and `GoalHead.type_head.weight` → `(n_tokens, hidden)`.

⭐ `TACTICAL_LAT_ACTIONS_V61` already existed (prepared 2026-08-18, never
default). **`STRATEGIC_ACTION_TOKENS_V61` is new in this change.**

## 2. Why it was necessary — the labels could not express the truth

MEASURED on clip `0e56dae2` (PI review):

* `g_str = TURN_LEFT` (junction turn, R = 6.4 m) with **`a_str = HOLD_CORRIDOR`**
  — "hold the corridor" as the action for LEAVING it. v6.0 has NO token for
  committing to a junction turn: `PREPARE_EXIT` is a motorway exit,
  `PREPARE_LANE_CHANGE` is lane-relative. **Every turn in the corpus was forced
  onto a token that contradicts it.**
* `a_tac.lat = LANE_KEEP` while traversing an intersection — false by
  construction, since there are no lanes to keep in a junction.

Corpus-scale effect of the fix (801 clips):

* `a_str`: **`PREPARE_TURN_L` 126 + `PREPARE_TURN_R` 114 = 240**; `HOLD_CORRIDOR`
  drops **478 → 313** and now means what it says.
* `a_tac.lat`: **`TURN_L` 73 + `TURN_R` 50 = 123** clips that previously read
  `LANE_KEEP`.

## 3. ⛔ THE SAFETY CONTRACT — APPEND, NEVER INSERT, DEFAULT UNCHANGED

1. **Indices 0–5 keep their exact meaning.** Every existing label, dump,
   artifact and checkpoint stays valid with no re-derivation.
2. **A 6-wide head widens to 8 by PADDING**, not retraining. Two fresh rows.
3. **The default is still v6.0**, so importing `v6` changes nothing. Selecting
   v6.1 is a deliberate act in a run config.
4. ⚠️ **Select BOTH LAYERS TOGETHER.** A strategic layer that can say "prepare
   to turn" while the tactical layer can only say "lane keep" reproduces the
   same contradiction one level down. Pinned by
   `stack/tests/test_vocab_v61_turns.py::test_the_two_layers_must_be_selected_together`.

⚠️ **The live v6F S-W run on Thor holds 6-wide tensors under a tensor-strict
resume contract.** Editing the tuples in place would not disturb the running
process — it would brick its AUTO-RESUME hours later as a silent shape mismatch.
That is why this is a version switch and not an edit. **Nothing about the live
run changes unless someone opts in.**

## 4. What we ask of you

| owner | request |
|---|---|
| **Master Mind** | Accept or reject v6.1 as the design default for the next model generation. The label set is emitted at v6.1 today; if you reject it, the emitter falls back and junction turns will ABSTAIN rather than emit a false `LANE_KEEP`. |
| **Training FlyWheel** | Any new run consuming these labels must select `tactical_lat_actions("v6.1")` AND `strategic_action_tokens("v6.1")`, and size both heads to 8. Record the version in `config.json` — an env/vocab knob that is not in the config is the documented failure mode (two arms differing only in a knob produced byte-identical configs). |
| **Eval FlyWheel** | ⛔ **Representable ≠ scoreable.** `TACTICAL_LAT_UNDERPOWERED` refuses per-class metrics below n = 200; TURN_L/TURN_R are at 73/50 here. They may be emitted and supervised; **any per-class metric on them must be REFUSED for under-power.** |

## 5. Also worth your attention

* **No `ROUNDABOUT` / `U_TURN` token exists.** Confirmed visually (`d5a38fdd` is
  a 177° circulation, `028eff14` 173°) and now COUNTED: **140 of 4,729 clips
  (3.0 %)** mention a roundabout in the Alpamayo CoT. Today every wide rotation
  is absorbed into `TURN_LEFT`/`TURN_RIGHT`. Not fixed here — it is a design
  decision, and a third append would want to land in one batch with any others.
* **The strategic-horizon blocker is withdrawn** (C141). It was an artefact of
  the 20 s episode cache; the provider egomotion gives **median 37.0 s** and
  **94.5 % of clips carry the full 8–30 s band**. Nothing needs redesigning.

## 6. Status

**AWAITING ACCEPT/REJECT.** Per charter §7 a transfer must be accepted or
rejected with a reason; silent non-transfer is a process failure. The change is
staged, not committed, and inert until a run selects v6.1.


---

## 7. ⭐ ADDENDUM — THE LAYER BOUNDARY WAS ALSO WRONG (PI, same review)

**PI:** *"the turning is happening within the 6 seconds horizon, so it is no
strategic action any more, it's tactical. The strategic action in such a
situation should be follow road after turning, which can be extracted."*

The strategic label was derived over `[t0, t0+30 s]` — a window that **contains
the tactical band** — so a manoeuvre the plan already executes was promoted to
the strategic layer. MEASURED on `0e56dae2`, the three bands give three
different answers:

| band | manoeuvre |
|---|---|
| operative 0–2 s | STRAIGHT (approaching) |
| **tactical 2–6 s** | **JUNCTION_TURN_L, +102.6°, R = 6.4 m** |
| **strategic 8–30 s** | NUDGE_R, −11.5°, R = 160.6 m — follows the road |

⇒ `g_str` is now derived over the **strategic band only (8–30 s)**.

**Consequences the model owners must know:**

1. **Strategic turn labels drop 271 → 188** (−31 %). Those 83 clips are turns
   the PLAN executes; they are now tactical, where they belong.
2. ⭐ **`PREPARE_TURN_L/R` changes meaning — for the better.** Scoped to 8–30 s
   it now marks a turn the ego **has NOT STARTED** — a decision to prepare for,
   which is what a strategic action should be. 171 clips carry it.
3. **The plan horizon (6 s) must never overlap the strategic band (8 s+).**
   Pinned by `test_the_strategic_band_never_starts_before_8s`.
4. 22.9 % of strategic turns coexist with a tactical turn — legitimate: those
   clips turn twice, once now and once later.

⚠️ **This is a LABEL SEMANTICS change, not a tensor change** — no shape moves.
But a head trained on the old labels learned "strategic turn" from manoeuvres
that were already under way; a head trained on these learns it from manoeuvres
still ahead. **Any comparison across the two label generations is invalid.**


---

## 8. ⭐ THIRD VOCABULARY APPEND — `ADAPT_SPEED_FOR_CURVE` (PI idea, same review)

**PI:** *"it is better to add another long tactical behaviour which is adapt
speed for curve, we can use this in turning manoeuvres and also sharp curves"* —
clarified: *"by curve I don't mean curve as a term but from turning manoeuvre or
arc detection."*

| tuple | v6.0 | v6.1 | appended |
|---|---|---|---|
| `TACTICAL_GOAL_TOKENS_LON` | 7 | 8 | `ADAPT_SPEED_FOR_CURVE` |

Sizes `self.vocab_tac_lon = GoalVocabulary(TACTICAL_GOAL_TOKENS_LON, ...)`.
Same contract: **append, indices 0-6 unchanged, default v6.0, 7-wide head
widens to 8 by padding.**

**Why the vocabulary needed it.** Slowing for an arc is neither a held
`SPEED_BAND` nor a `STOP_POINT`. Before this token the longitudinal half of
every junction turn was either called SPEED_BAND (false — it is not held) or
abstained away. **113 of 801 clips** now carry it, each paired BY CONSTRUCTION
with a turning manoeuvre in the tactical band.

⚠️ **Two errors of mine on the way, both corrected:**
1. I first gated it on a **deceleration coinciding with curvature**. Wrong
   referent — a turn taken at an already-suitable speed is still speed governed
   by the curve. The trigger is now **arc detection in the tactical band**, and
   `dv_ms` records the actual slowdown so "braked for it" and "was already slow
   enough" remain distinguishable.
2. I then "validated" it against the CoT phrase *"curve ahead"* and reported the
   co-occurrence as chance-level (4 overlaps, p=0.24). That measured the wrong
   thing entirely — the token is about the ARC, not the word.

## 9. Strategic action args — `within_m` / `by_time_s`

**PI:** *"add e.g. the arg prepare turn right in x m and y seconds. This will
make clear that this action is not affecting the current tactical manoeuvre."*

`PREPARE_TURN_L/R` now carries the vocabulary's own uniform constraint slots
(§2): **`within_m`** (arc distance from the anchor to the turn) and
**`by_time_s`** (its onset). A head — and a reader — can now separate a turn
60 m away from one being executed now. No tensor change; args ride the existing
slot layout.

## 10. ⚠️ `ANCHOR_GOAL` DEVIATES FROM THE SPEC — a decision for the Master Mind

`HIERARCHY_VOCABULARY.md` §4 defines the args as `anchor_id ∈ fan vocab,
t_reach_s` — a discrete INDEX. **We emit raw metric `goal_x_m` / `goal_y_m`
instead**, deliberately:

E-AG1 (banked, 881 windows / 40 episodes) measured a K-way `anchor_id`
classifier **near-adequate laterally (1.331 vs a 0.680 floor) and hopeless
longitudinally (13.350 vs 0.895, 14.9x)**, because the 2 s goal point's variance
is **98.8 % longitudinal**. Quantising that axis into one shared categorical is
the 5-way-softmax defect in a new place.

⇒ **Decide: either the fan vocabulary gains a longitudinal axis, or
`ANCHOR_GOAL` stays metric and the spec line is updated.** Recorded here rather
than silently diverged.

## 11. Eval-set policy change

**PI:** *"skip all eval data which does not include Alpamayo data."* The
validation/eval set is now restricted to Alpamayo-covered clips. On the
join-free set that is **13 of 33** clips (20 skipped). Rationale: a clip with no
VLM row can never exercise the semantic path, so including it silently dilutes
every rate computed over the set. ⚠️ Consequence for Eval: coverage is 100 % of
aug120 but only 9.3 % of w120val, so an Alpamayo-only eval set is heavily
aug120-weighted — extending the augmentation to w120val is now on the critical
path for evaluation, not just for labelling.

---

## 12. ⭐⭐ v7 VOCABULARY — IMPLEMENTED (PI redesign, 2026-08-23)

`stack/tanitad/models/vocab_v7.py` · `stack/tanitad/data/cot_tokens_v7.py` ·
`stack/scripts/s2_geom_emit_v7.py` · `stack/tests/test_vocab_v7.py` (27 arms).
**801/801 clips emitted, 0 failures, 0 goal-set violations. 183 tests pass.**

⛔ **NO v6 TUPLE IS EDITED, REORDERED OR TRUNCATED.** v7 introduces NEW NAMES
only, so no existing checkpoint changes shape and the live Thor run is
untouched. This is a bigger change than the v6.1 appends and it is therefore
kept fully parallel rather than layered on them.

**The structural change: layers are ORDINAL, not temporal.**
`manoeuvre[0] -> tactical`, `manoeuvre[1] -> strategic`. A time split assigned a
manoeuvre to whichever band it straddled — `01b24287`'s single +97 deg turn
(t+5.9-11.9 s) crossed the 6 s/8 s boundary and was reported as the route-level
decision.

| tuple | size | note |
|---|---|---|
| `STRATEGIC_GOAL_TOKENS_V7` | 8 | every token carries `_FOLLOW_ROUTE` |
| `STRATEGIC_ACTION_TOKENS_V7` | 8 | `HOLD_MAIN_ROAD` replaces `HOLD_CORRIDOR` |
| `TACTICAL_GOAL_TOKENS_V7` | 18 | **multi-label SET**, no ABSTAIN, no ANCHOR_GOAL |
| `TACTICAL_LAT_ACTIONS_V7` | 8 | + args (`within_m`) |
| `TACTICAL_LON_ACTIONS_V7` | 8 | + `ADAPT_SPEED_FOR_CURVE`, + `ACCELERATE` |
| `NAV_COMMAND_TOKENS` | 3 | ⛔ **a MODEL INPUT — see the oracle warning** |

**What the heads must change.** The tactical goal head becomes **MULTI-LABEL**
(sigmoid per token, not softmax), checked against `TACTICAL_GOAL_EXCLUSIVE`
(17 forbidden pairs). The anchor becomes an **always-present continuous arg
triple**, not a class. That is a head-shape change, not just a width change.

⛔⛔ **THE NAV COMMAND IS ORACLE INFORMATION ON THIS CORPUS.** Our only route
supplier on PhysicalAI is the ego's own recorded future, so a nav command
derived from it and fed back as an input **tells the model the answer**. Every
emitted command carries `provenance="ego-future"` and `oracle=true`.
⇒ `nav-system` provenance is admissible at inference; `ego-future` is
**TRAINING ONLY**, and any eval using it is an ORACLE arm that must be labelled
as such and **never compared against a vision-only arm**. An arm that cannot
state its nav provenance is not evaluable.

⚠️ **Three tokens are defined but the corpus cannot populate them:**
`WAIT_FOR_ONCOMING` **0 clips** (142 mention "oncoming" but the phrasing is
lateral CLEARANCE, never waiting — CLEARANCE 110 / PRESENT 32 / WAIT 0),
`OVERTAKE_VEHICLE` **2** ("overtake" in 13/4,729; the 326 "pass the ..." are
almost all PARKED vehicles = an evasion), `GAP_TARGET` **0** (needs perception).
All are below the n=200 floor ⇒ representable, **not scoreable**.

⚠️ **Class balance:** `FOLLOW_ROUTE` is **549/801 = 68.5 %** of strategic goals.
Needs class weighting or the head learns to always say follow-route.

**Status: AWAITING ACCEPT/REJECT** alongside the v6.1 appends in §1-§11.

---

## 13. v7 SECOND PASS — new tokens, the goal↔action matrix, and two coherence fixes

**New / changed tokens** (all still NEW NAMES; no v6 tuple touched):

| token | change | corpus yield |
|---|---|---|
| `REACT_ON_ONCOMING` | renamed from `WAIT_FOR_ONCOMING`, fires on any "oncoming" | 0 → **142** (3.0 %) |
| `YIELD` | extracted from CoT | **400** (8.5 %) |
| `MERGE` | new; also feeds the `YIELD_MERGE` action | **45** (1.0 %) |
| `TAKE_EXIT_L/R` | from CoT TERMS, not geometry | 54 (exit 17 + ramp 26) |
| `SPEED_BAND` | RESTORED to the tactical goals | 256 clips |
| `OVERTAKE_VEHICLE` | redefined: a SLOWER MOVING vehicle in front | 2 → **45** |
| `EVADE_IN_CORRIDOR` | redefined: STATIC obstacle / VRU | **574** (12.1 %) |
| `ACCELERATE` | new LON action | 189 clips |

⭐ **`GOAL_ADMISSIBLE_LAT` / `GOAL_ADMISSIBLE_LON`** — the goal↔action link the PI
asked for. Every one of the 22 tactical goals maps to the actions that may serve
it, and `action_serves_goals()` writes the verdict into every label.

⚠️ **It found two coherence defects of mine inside minutes:**
1. `FOLLOW_LANE` did not admit `BRAKE_TO` — 69 false mismatches. Braking WHILE
   following a lane is ordinary; the map described an idealised taxonomy.
2. `SPEED_BAND` allowed a 3.0 m/s spread while the `CRUISE` action required
   |dv| < 1.0 — **two thresholds for one fact**, 103 clips affected. Same defect
   family as the emitter/guard constants earlier today.
⇒ longitudinal goal↔action consistency **72.7 % → 94.4 %**. The remaining 45 are
temporal ordering inside the 6 s window and are left VISIBLE, not tuned away.

**`REDUCE_TO_FOLLOW_ROUTE` now has a definition** (it had none and 0 emissions):
a SUSTAINED speed drop across the strategic horizon that no manoeuvre explains —
a slower road class. 90 clips. ⚠️ This is the one token whose meaning I CHOSE
rather than derived; say if you would rather drop it.

⛔ **`meta_action` is NOT reachable locally.** Alpamayo's raw records carry
`answer|box|cot|cot_auto_labeling|meta_action|raw_outputs`, but only
`cot, lane, lateral, longitudinal` were exported per clip. Given the lane and
lateral axes measured AT CHANCE, meta_action should be permutation-tested the
same way before any consumer trusts it.

⚠️ **The open-door case does not exist**: 0 of 4,729 CoTs mention a door. The
evade class is parked vehicles (419), pedestrians (278), cyclists (64).

⚠️ **Thin on the labelled set**: only 257 of 801 clips carry Alpamayo, so
MERGE 1, TAKE_EXIT_R 1, OVERTAKE 2, GAP_TARGET 5, REACT_ON_ONCOMING 6 —
all far below n=200 ⇒ emit and supervise, **refuse per-class metrics**.
Corpus-wide those same tokens are well populated, which is a further reason to
extend the augmentation beyond aug120.
