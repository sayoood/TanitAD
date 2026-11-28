# The traffic-light chain — the label exists, and it reaches nothing

**Date** 2026-09-06 · **Owner** Architecture & Inference · **Claim ID** `D-TLIGHT-1`
**Branch** `agent/arch-inf-20260803` · **Evidence class** MEASURED unless stamped otherwise
**Tier** — no T0/T1/T2 number is asserted here. This is a **wiring audit**, not an eval.
Nothing below is a driving-performance claim.

---

## 0. The one-line answer

| question | answer |
|---|---|
| Is the GT traffic-light label **reaching training**? | ⛔ **NO** |
| Is it **reaching inference**? | ⛔ **NO** — and therefore it is **not a leak either** |
| Is it **being evaluated**? | ⛔ **NO** |

The label is real, colour-carrying, and banked. It is consumed by **nothing**.

---

## 1. The PI's correction, and the root-cause class

The Master Mind asserted *"PhysicalAI-AV has no traffic-light feature, so there is no
label to learn braking-for-red from."* The PI corrected it: our dataset **does** carry GT
traffic light as part of the tactical goals, with the colour extracted from Alpamayo.

⭐ **Both statements are true of different LAYERS**, which is the programme's own
documented failure mode — *"always state the LAYER, never the bare phrase 'our ingest'"*:

| layer | traffic light? |
|---|---|
| **1 — published PhysicalAI-AV feature set** | **absent.** The card says *"we do not include open maps data"*; `obstacle.offline`'s enum is 10 dynamic-agent classes. |
| **2 — our AUGMENTED v7.x label release** | ⭐ **PRESENT, with the colour.** 779/4,572 records. |
| **3 — supervised training targets** | ⛔ **absent.** This is the finding. |

A **layer-1 fact was quoted to settle a layer-3 question.** `CLAUDE.md` has been corrected
in place (§7) and `stack/tests/test_tactical_label_reach.py` now pins the distinction, the
way `test_physicalai_feature_readset.py` pins the feature counts — that file's stale-count
rot has bitten **four times**, which is the argument for a test rather than a sentence.

⛔ **PI ruling that frames admissibility:** GT colour may **SUPERVISE a vision-reading
head**; a GT colour **fed at inference** is a privileged-channel leak. **What the code
does today is neither** — the label is dropped before either could happen.

---

## 2. The label (link 1) — POSITIVE

**Corpus** `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz`
— schema `s2-geom-v7`, vocab `v7`, **4,572 records, one per clip**, md5
`0ff902130ce76886b8a925eceed9e3a5`; eval sibling 147 records, md5
`aa12c948f062181c3297265b51526ec5`. The clip set is **alpamayo(4,729) ∩ provider
egomotion**; train + eval = **4,719** records combined.

⭐ **The md5 is a genuine cross-check, not a formality.** Two agents reached this blob
through **different channels** — a retry-copy through the G: mount, and a join of the Drive
content-cache blobs against the read-only Drive metadata DB — and produced the **same
md5**. That matters here because the mount was mid-outage and a `tar` copy through it
produced **correct-size files filled entirely with NUL bytes** for the other agent: a
silent-corruption channel that would have made every search read zero while looking healthy.
All local copies used in this report were re-asserted on **content** (0 all-NUL files in a
400-file sample; the blobs parse as gzip JSON with the expected record counts).

**Writer chain** `s2_geom_emit_v7.py` (one JSON line per clip) → `build_release.py` (v7.0
gzip) → `build_v71.py` (metadata-only) → `build_v72.py` (adds `release`, `split`; emits the
two blobs above). The v7.0 bank has the **identical** 805/4,719 traffic-light coverage,
confirming v7.1/v7.2 changed no label content.

**Field** `g_tac.goals.<TOKEN>` — the tactical goal **SET**, a dict keyed by token.

**States / coverage** (fraction of the 4,572-record v7.2 train blob; eval blob in brackets):

| token | coverage | |
|---|---|---|
| `TRAFFIC_LIGHT_REACT_RED` | **376/4,572** | [8/147] |
| `TRAFFIC_LIGHT_REACT_GREEN` | **363/4,572** | [15/147] |
| `TRAFFIC_LIGHT_REACT_YELLOW` | **22/4,572** | [3/147] |
| `TRAFFIC_LIGHT_REACT` (colourless) | **18/4,572** | [0/147] |
| **any `TRAFFIC_LIGHT_*`** | **779/4,572** | [26/147] |

The four counts sum exactly to 779 — no record carries two colours, consistent with
`vocab_v7.TACTICAL_GOAL_EXCLUSIVE`.

**Provenance** `provenance: "vlm-cot"` on **779/779**; `object_kind: "traffic_light"`;
the colour in a `state` field (`red` 376 · `green` 363 · `yellow` 22 · `unknown` 18).
Extracted from the Alpamayo chain-of-thought by `stack/tanitad/data/alpamayo_semantics.py`
and frozen into the vocabulary by `stack/tanitad/models/vocab_v7.py`.

⚠️ **Quality caveats that travel with it, and they are not small:**
* `disputed: true` on **604/779 (77.5 %)**
* `grounded: true` on only **175/779 (22.5 %)**
* `time_basis`: **`untimed` 691/779**, `segment` 88/779. The untimed ones carry
  `t_nominal_s` = **4.0 on every single record** (min = p50 = max = 4.00), stamped
  `t_nominal_provenance: "band-midpoint (PI 2026-08-28)"`. ⛔ **That is a placeholder, not
  a measurement** — 4.0 s is the exact midpoint of the [2.0, 6.0] tactical band. Any
  consumer treating `t_nominal_s` as *when the light was that colour* is reading a constant.

**Real banked record** (clip `00948c8c-e41f-4c37-8253-3a88c57f37c7`), verbatim:

```json
"TRAFFIC_LIGHT_REACT_RED": {
  "by_box_perception": false, "by_named_component": true,
  "corroboration": "component", "disputed": true, "grounded": false,
  "object_kind": "traffic_light", "provenance": "vlm-cot", "state": "red",
  "t_nominal_provenance": "band-midpoint (PI 2026-08-28)",
  "t_nominal_s": 4.0, "time_basis": "untimed"
}
```
with `semantics.stop_reason: "light"` and the CoT *"Stop due to the red traffic light."*

---

## 3. The six-link chain — a POSITIVE verdict per link

| # | link | verdict | the positive evidence |
|---|---|---|---|
| 1 | **label exists** | ✅ **YES** | 779/4,572 above, from the banked blob, with a verbatim record. |
| 2 | **reaches the batch** | ⛔ **NO** | `v7_labels.load_v7_labels` reads `g_tac` and keeps only `g_tac["anchor"]` (→ `V7Label.tac_anchor`). `g_tac["goals"]` goes to `V7Label.audit["goal_flags"]` via `_goal_audit`. The module declares `audit` *"audit-only, NEVER a training input"*. The `V7Label` dataclass has **no** goal-set field: `clip_id, tac_lat, tac_lon, str_action, str_goal, tac_anchor, bands, t0_s, horizon, audit, _oracle`. |
| 3 | **reaches the forward** | ⛔ **NO** | Not tuple drift — **no goal-label channel was ever added.** `V6Stack.forward`'s signature is `(frames, actions, v0, *, own_frames_tac, own_frames_str, nav_token, nav_args)`; `train_v6_staged.py`'s call covers it exactly. `g_tac` is **emitted** by the model (`self.goal_head_tac(...)`), never fed in. *(The hand-maintained tuple that bit twice, `rl/refc_adapter.FORWARD_KEYS`, is now guarded by `rl/channel_guard.py`, which diffs it against the live signature — checked both directions, no discrepancy, and it carries no tactical-goal channel either.)* |
| 4 | **in a weighted loss** | ⛔ **NO** | `grep 'out["g_tac'` in `train_v6_staged.py` → **0 hits**; same-breath control `grep -c 'out["'` → **43 hits**, so the probe reads. The only goal-token loss is **strategic**, `w_s2_goal` default **0.0**, guarded `if w.w_s2_goal:` so the term is never even constructed. refc_v3 *does* carry a `g_tac` loss at weight 0.5 — but `refs/refc_v3.masked_goal_loss` is a **smooth-L1 regression over `[B, K, 4]` = (x, y, heading, speed)**, a goal **point**, with no token slot and therefore no traffic-light slot. |
| 5 | **evaluated** | ⛔ **NO** | `tanitad/eval/scenarios/traffic_light.py` + `metrics.compute_tlc` exist, but the module states in its own docstring that its telemetry is a **synthetic design oracle** and *"NOT a claim about our real model"* — it needs a CARLA/MetaDrive rollout that has never been run on a TanitAD checkpoint. No four-families tactical row scores the traffic light for any banked arm. |
| 6 | **extractable at inference** | ⛔ **NO — untested, because nothing was ever trained.** There is no vision-reading traffic-light head to probe. ⭐ **The echo test is moot and must not be reported as passed**: an echo test asks whether a head is reproducing its own input, and here there is no head and no input. The exemplar bar remains the route head (κ **0.4852**; anti-echo control **0.7155** true vs **0.1701** shuffled, non-overlapping, n=1,311) — versus flagship v1's route head, an exact bijection of its own input that scored **1.0000**. |

⭐ **Link 3 has a nuance worth stating precisely, because it is the good news.** The
**output slot already exists**: `V6Config.tac_vocab_version` defaults to `"v7.0"`, so
`goal_head_tac` is sized on `TACTICAL_GOAL_TOKENS_V7` — **22 wide, with four dedicated
traffic-light logits**. The head can say "red"; nothing ever tells it what red looks like.
⇒ **This is a label-plumbing gap, not an architecture gap.**

---

## 4. ⭐ The mutation result — MEASURED, with its discriminating control

⛔ An AST census once read 0 suspects on **both** the fixed and the broken trainer. So the
verdict above is not left on inspection. **Instrument** `stack/scripts/tactical_label_census.py`
(harness `…/2026-09-06-traffic-light-chain/mutate.py`).

**Mutation** — flip RED↔GREEN on every labelled window: the token key, the `state` field,
`a_tac.serves_goals.goals_checked`, and the CoT referent.
**Same-breath discriminating control** — on the **same 779 records**, change `a_tac.lon`,
a field whose perturbation **must** move the supervised target.

**Positive assertion that the mutation landed** (without it, "0 changed" is
indistinguishable from "the mutation never happened"):

```
RED  count  base= 376 target= 363    <- exact swap
GREEN count base= 363 target= 376    <- exact swap
a_tac.lon=ACCELERATE base= 998 control=1451
```

| mutation | supervised fields changed | audit channel changed |
|---|---|---|
| **TARGET** — red↔green on 739 entries | ⛔ **0 / 4,572** | 739 / 4,572 |
| **CONTROL** — `a_tac.lon` on 779 records | ✅ **779 / 4,572** | 0 / 4,572 |

⇒ **The rig is live** (the control moved 779), and **the colour is invisible to every field
the trainer consumes as supervision.** The two channels are live and **disjoint** — the
target moves only audit, the control moves only supervision — which is the mechanism stated
as an experiment rather than as a reading of the source.

⚠️ **A defect in my own first harness, logged because it is the same family:** the initial
mutation swapped RED↔GREEN with an in-place two-way dict flip, which **re-visits the entry
it just inserted and flips it back** — self-cancelling on exactly the RED records. It
reported **1,115 flips over 779 records** (the tell: more flips than records) while only the
363 GREEN ones had actually moved. Rebuilt as a single-pass dict rebuild; the landed-mutation
assertion above exists so this cannot recur silently.

---

## 5. The tactical-field census

`stack/scripts/tactical_label_census.py <blob>` regenerates all of this. Against the v7.2
train blob:

| tactical field | in blob | surfaced by the loader | in a weighted loss | evaluated |
|---|---|---|---|---|
| `a_tac.lat` | 4,572/4,572 | `V7Label.tac_lat` | tensor lands; **no loss reads it** | no |
| `a_tac.lon` | 4,572/4,572 | `V7Label.tac_lon` | tensor lands; **no loss reads it** | no |
| `a_tac.lat_args` / `lon_args` | 4,572/4,572 | `audit['a_tac_args']` | ⛔ no (audit) | no |
| `a_tac.serves_goals` | 4,572/4,572 | `audit['serves_goals']` | ⛔ no (audit) | no |
| `a_tac.truncated` | 4,572/4,572 | `audit['a_tac_truncated']` | ⛔ no (audit) | no |
| `g_tac.anchor` | 4,572/4,572 | `V7Label.tac_anchor` | the admissible goal signal | partially |
| **`g_tac.goals`** | **4,572/4,572** | ⛔ **`audit['goal_flags']` — AUDIT ONLY** | ⛔ **no** | ⛔ **no** |
| `g_tac.violations` | 4,572/4,572 | `audit['g_tac_violations']` | ⛔ no (audit) | no |

⭐ **The generalisation: it is not the traffic light.** **All 22 tactical goal tokens** are
unsupervised, not just the four light ones — `SPEED_BAND` 4,572/4,572, `FOLLOW_LANE`
3,629/4,572, `CORRIDOR_OFFSET` 860/4,572, `YIELD` 609/4,572, `GAP_TARGET` 368/4,572,
`STOP_POINT` 327/4,572 … all of them. **No head in `v7_labels.HEADS` is sized on
`TACTICAL_GOAL_TOKENS_V7`.** The four supervised heads are `tac_lat` (8), `tac_lon` (8),
`str_action` (7), `str_goal` (8) — **31 classes, none of them a tactical goal.**

⚠️ **Read the census's last column carefully.** Four goal tokens (`TURN_L`, `TURN_R`,
`LANE_CHANGE_L`, `LANE_CHANGE_R`) also exist as `tac_lat` **ACTION** classes. A naive
"supervised: yes" there is true of the **string** and false of the **goal**, and would tell
the reader the goal is supervised. The script prints
`NO (name also a tac_lat ACTION class)` for exactly those rows.

⭐ **The trainer already says this about itself** — this audit did not discover a hidden
bug so much as measure a documented, un-actioned gap. `train_v6_staged.py` writes into every
run record:

> `"tactical_consumer": "NONE — the keys land in the batch; no loss term reads them yet (pre-registered follow-up, not a silent addition)"`

---

## 6. ⛔ The ±2.0 s band — the bigger story

The validation video reads *"v7.2 GT: — outside the v7.2 record's ±2.0 s band"* on many
frames. Measured, and it is worse than "many":

* **One record per clip** — 4,572 records over 4,572 distinct `clip_id`s, max 1 per clip.
* **`t0_s` is a CONSTANT 8.0 on all 4,572 records.** Every clip is anchored at the same instant.
* `bands.tactical_s` = `[2.0, 6.0]` on all 4,572 → a **4-second-wide** window.
* `recording_span_s` p50 **139.7 s**; `available_s` p50 **35.0 s**.

**Where the ±2.0 s actually comes from** — `stack/tanitad/data/refav1_loader.py`, and it is
**derived from the record, never hardcoded**:

```python
tol = (float(band[1]) - float(band[0])) / 2.0        # (6.0 - 2.0)/2 = 2.0
self._lab[x.clip_id] = (lat_id, lon_id, route, x.t0_s - tol, x.t0_s + tol)
```

⇒ the supervised window is **`[t0 - 2.0, t0 + 2.0]` = [6.0 s, 10.0 s]**, centred on the
anchor. A window outside it is set to `IGNORE_ID` (**-100**) and, per the loader's own
comment, *"is NEVER clamped to a neutral class"* — so out-of-band frames are correctly
excluded from the loss rather than silently taught a wrong label.
⚠️ **My first pass placed this window at [t0+2, t0+6]** — reading the band as an interval
*forward* from the anchor rather than as a *width* centred on it. The coverage arithmetic
below is unaffected (it uses the 4.0 s **width** against the span), but the window position
was wrong and is corrected here.

| denominator | covered | **uncovered** |
|---|---|---|
| full recording (`recording_span_s`) | 18,288 s / 508,726.5 s = **3.59 %** | **96.41 %** |
| usable horizon (`available_s`) | 18,288 s / 154,070.1 s = **11.87 %** | **88.13 %** |

*(eval blob: 3.49 % / 11.74 % — the same shape.)*

⇒ **The tactical GT is absent at the scored frame far more often than not, on either
denominator.** ⚠️ Both are reported because the honest answer depends on which frames are
scoreable, and quoting only the 3.59 % would overstate it. Either way this is a
**programme-wide constraint on tactical supervision**, not a traffic-light problem: it
bounds how much tactical signal *any* of the 22 goal tokens could ever provide, and it is
the more likely explanation for weak tactical supervision generally.

---

## 7. The `CLAUDE.md` correction — scoped, and pinned

The passage was **already scoped** in its first clause (*"…in PhysicalAI-AV"*). What made it
usable one layer away was the closing imperative — *"Stop re-asking; the strategic-brain
topology must come from AlpaSim or an external corpus"* — which forecloses the augmented
layer without naming it.

**Edit** (that one paragraph only; before/after verified; marker asserted with a non-zero
same-breath control; length 76,182 → 77,631 chars):

> ⛔⛔ **AND THAT PARAGRAPH IS A LAYER-1 FACT THAT HAS ALREADY BEEN QUOTED TO SETTLE A
> LAYER-3 QUESTION.** … **779/4,572 records carry a GT traffic-light tactical goal WITH THE
> COLOUR ATTACHED** … ⇒ **"PhysicalAI-AV publishes no traffic-light feature" is TRUE; "the
> programme has no traffic-light label" is FALSE.**

**Pin** `stack/tests/test_tactical_label_reach.py` — **10 tests, all passing**. It asserts
the four token **names** (not a count — the readset test's count rotted four times), that
the tokens stay in `TACTICAL_GOAL_NEEDS_PERCEPTION` (⛔ a geometry emitter that invents a
light colour is fabricating perception from ego motion — the sitclf leak in a new costume),
that no supervised head carries traffic-light classes, that `V7Label` exposes no goal-set
field, and that `g_tac.goals` is routed to `audit`. Every failure message names the
documents to update.

⭐ **The guard is mutation-tested, not inspected** — `test_the_reach_detector_actually_fires`
adds a fake traffic-light head and requires the detector to fire, so a detector that can
never fail cannot pass forever and certify nothing.

⚠️ **This test is a GAP PIN, not an approval.** It asserts the *current* wiring, so the day
someone supervises the tactical goal set the build breaks and the docs get updated in the
same commit — rather than the fix landing silently while the docs still describe the gap.

---

## 8. The evidence, and what it does not explain

**Behaviour to explain:** at a red light (clip `4c5264dd`) the longitudinal head emitted
**CREEP p=0.30** — argmax but weak; BRAKE 0.18, HOLD 0.08.

⛔ **This audit does not explain that, and must not be quoted as if it did.** It establishes
that the head was **never told about the light** — a red light and a green light are
*identical* inputs to every supervised objective in the programme. A near-uniform
longitudinal posterior at a signalised intersection is **consistent** with that, but the
audit is a wiring fact and the emission is a T1 behaviour; connecting them needs an arm that
actually trains the signal. ⭐ The falsifiable prediction is available and cheap: **an arm
with the tactical-goal head supervised should move the longitudinal posterior on the
779-record red/green subset, and one without it should not.**

---

## 9. What would close it — cheapest first

1. **Surface the field.** `V7Label` gains a goal-set field and `load_v7_labels` stops
   routing `g_tac.goals` to `audit` alone. ~30 lines in the file this agent owns.
   ⚠️ Then `audit` is no longer the only path, so the leak rules must be re-read for every
   other audit field in the same pass.
2. **Supervise the existing head.** `goal_head_tac` is **already 22-wide with the four
   light slots** at the `v7.0` default. A multi-label BCE over the goal set on
   `out["g_tac"]`, with the mask/weight discipline `v7_labels.assert_mask_matches_presence`
   already implements. ⛔ Pre-register it — `train_v6_staged.py` calls this out as a
   *"pre-registered follow-up, not a silent addition"*, and that is the correct bar.
3. ⛔ **Do not fix the band by widening the tolerance.** One record per clip at a constant
   `t0_s` = 8.0 is a **label-density** limit; a wider window would relabel frames the
   emitter never examined. Closing §6 means emitting records at more anchors.
4. **Then, and only then, the vision-only extraction question** (link 6) becomes askable,
   with the route head's anti-echo protocol as the bar.

⚠️ **Power check before any of this earns compute.** `TRAFFIC_LIGHT_REACT_YELLOW` is
**22/4,572** and already sits in `vocab_v7.TACTICAL_GOAL_UNDERPOWERED` against
`GOAL_MIN_N_FOR_METRIC = 200`. **RED (376) and GREEN (363) clear that bar; YELLOW does
not** and must be reported as representable-but-not-scoreable rather than quietly scored.

---

## 10. Two incidental defects found in passing

* ⛔ **`tactical_goals.derive()` can never emit `TRAFFIC_LIGHT_REACT`.** The token is
  declared in `LON_TOKENS` and the function's closing assert admits it, but no branch ever
  assigns it — a light-caused stop becomes `STOP_POINT` with `{"reason": "light"}`. The
  token is *reachable by the assert and unreachable by the code*, which is how a vocabulary
  entry looks alive while being dead. **Not a bug in the v7.2 blob** (whose light tokens come
  from the CoT path, not this function), but a trap for the next reader of that file.
* ⚠️ **`vocab_v7.TACTICAL_GOAL_UNDERPOWERED` contains `"WAIT_FOR_ONCOMING"`, which is not
  in `TACTICAL_GOAL_TOKENS_V7`** — it was renamed to `REACT_ON_ONCOMING` on PI instruction
  and the underpowered set was not updated. A dead string in a live set; harmless today
  because nothing indexes it, and exactly the kind of drift the frozen-vocab test exists for.
* ⚠️ **Factored-head asymmetry** (`v6.py`, `--goal-factored`, default-off): the **LON** half
  is version-resolved to v7, the **LAT** half is hardcoded to the v6 tuple with no version
  lookup — a v6/v7 mismatch inside one factored pair.
* ⛔ **`vocab_v7.py`'s own coverage numbers do not reconcile with the bank, and must not be
  quoted as coverage.** The docstring says *"measured 638/4,729 CoTs (13.5 %) name a light,
  of which green 432, red 192, yellow 28"*. **432 + 192 + 28 = 652 ≠ 638** — internally
  inconsistent. And they count **CoT mentions**, whereas the banked **emitted goal tokens**
  are 805/4,719 with GREEN 378 / RED 384 / YELLOW 25 / plain 18 — **the red-green ratio
  inverts** (docstring: green ≫ red; bank: red ≳ green). Quote the bank, measured from the
  blob by the census script; the docstring number is a different quantity wearing the same
  units. *(Same family as the stale feature count: a number in prose with nothing pinning it
  to what the pipeline actually emits.)*
* ⭐ **A deliberate, correct refusal worth knowing about, not a defect.**
  `s2_geom_emit_v7.py` **will not** promote a detected traffic-light **box** into a goal —
  box presence goes to `scene`, never to `g_tac`. Its stated reason: *the box proves the
  object is VISIBLE; nothing available proves the ego REACTED.* ⇒ there **is** box-level
  traffic-light perception data in `scene` that a future vision head could use as an input,
  and it is correctly kept out of the goal label.
* ⚠️ **The v7.2 eval split is not fully independent**: 6 of the 40 deployed-val40 clips are
  inside it (**6/147** of eval, **6/40** of val40). `build_v72.py` records this as a loss of
  independence rather than a leak. Any eval quoting the v7.2 eval split against a val40
  arm must carry that caveat.

---

## 11. Provenance

| artifact | where |
|---|---|
| this report | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-traffic-light-chain/TRAFFIC_LIGHT_CHAIN.md` |
| census instrument | `stack/scripts/tactical_label_census.py` |
| the pin (10 tests, passing) | `stack/tests/test_tactical_label_reach.py` |
| mutation harness | `…/2026-09-06-traffic-light-chain/mutate.py` |
| mutation + census output | `…/2026-09-06-traffic-light-chain/raw/` |
| corrected passage | `CLAUDE.md` (the no-map/no-traffic-light paragraph) |
| source blob | `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_{train,eval}.jsonl.gz` |

⚠️ **One INCONCLUSIVE, named rather than dropped.** Whether any *banked* run enabled
`--tac-vocab-version v7.0` could not be settled: 168 `config.json` files enumerate by
metadata, and **not one could be read** — six independent probes (bash `grep`/`head`,
PowerShell `Get-Content -Raw`, the Read tool, `git ls-files`) each failed with
`Invalid request code` / `Unzulässige Funktion` while same-breath controls read 0, i.e. a
mount content-read outage, **not absence**. It does not change the verdict — no trainer
imports `TACTICAL_GOAL_TOKENS_V7`, so the flag cannot supervise the goal set whatever the
configs say — but the run-level question stands open. **To resolve:** hydrate the Drive
files and re-run the census's config probe with a `grep -q "{"` control.
