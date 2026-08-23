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
