# D-TACGOAL-1 — the 22-token tactical goal vocabulary, wired

**2026-09-06 · Architecture & Inference · evidence class MEASURED unless stamped
otherwise · no model metric produced — this is a SUPERVISION/EXTRACTION result,
so it carries no T-tier and no ADE.**

The audit found we mint a 22-token tactical goal set on every one of 4,572 clips
and train on **none** of it. This package answers the coverage question first,
then wires the head.

---

## P1 — the 12 % coverage wall: **it is BOTH, and the split is measurable**

⭐ **The answer is not one word.** The geometric half is a WRITER limit and is
nearly free to remove. The CoT half — which is where **every traffic-light and
every lane-change token lives** — is a SOURCE limit *on time*, and regenerating
the writer does not touch it.

### The writer half — MEASURED, and it is a hard writer limit

| fact | value | where |
|---|---|---|
| `t0_s` over the train blob | **8.0 on 4,572 of 4,572** | `s2_labels_v7.2_train.jsonl.gz` |
| records per clip | **max 1, min 1**, 4,572 clips | same |
| the anchor's source | `egomotion_source.RAW_T0_S = 8.0`, a **module constant**; `key_index = round(RAW_T0_S * hz)` | `stack/tanitad/data/egomotion_source.py` |
| the emitter's loop | `for c in sorted(clips): rows.append(emit_one(c))` — one record, one anchor | `stack/scripts/s2_geom_emit_v7.py::main` |
| recording span actually available | **median 139.7 s** (min 20.2, p10 40.7, max 141.3) | `horizon.recording_span_s`, n=4,572 |
| what the emitter READS of it | `ES.load(max_s=RAW_T0_S + LOOKAHEAD_S + 5.0)` = **43 s** | `s2_geom_emit_v7.py::emit_one` |
| what it LABELS | one ±2.0 s band = **4.1 s** | `bands.tactical_s = [2.0, 6.0]` on 4,572/4,572 |

⇒ the writer reads **43 s of a ~140 s recording and labels 4.1 s of that**.
Against the 34.2 s episodes the trainer actually sees, that is the audit's
**6,027 / 50,243 = 12.00 %**; against the source recording it is
**6,027 / 168,405 = 3.58 %**.

⛔ **Every label function is a pure function of `key`.** `manoeuvre_sequence`,
`tactical_actions`, `tactical_goals`, `split_by_band` and `nav_command` all take
`(poses, key, seq, HZ)` — `key` is a **parameter**, not a constant inside them.
Sliding it regenerates the geometric labels at any anchor. And the timed
evidence to do so is already in the records: **3,464 manoeuvres over 2,183 of
4,572 records, `t_start_s` 0.0 → 33.6, `t_end_s` → 35.1.**

**9,106 goal annotations over 9 tokens carry `provenance: "geometry"`** —
`FOLLOW_LANE` 3,629, `SPEED_BAND` 4,572, `STOP_POINT` 327, `TURN_R` 259,
`TURN_L` 275, `YIELD_FOR_TURN_L` 21, `YIELD_FOR_TURN_R` 20, plus 3 strays. Both
currently-trained heads (`a_tac.lat`, `a_tac.lon`) are in this class.

### The source half — and this is the part regeneration cannot fix

**3,472 annotations over 15 tokens carry `provenance: "vlm-cot"`**, including
**all 779 traffic-light** and **all 38 lane-change** tokens. For these the record
carries **no observed time at all**:

* `t_nominal_s` is present on 3,144 goals and its value is **exactly 4.00 on
  every single one**, with `t_nominal_provenance: "band-midpoint (PI
  2026-08-28)"`. It is the tactical band's midpoint restated — **zero
  information about when the event happens**. ⚠️ An earlier reading of mine —
  *"82–95 % of traffic-light tokens carry a time"* — was **wrong for exactly
  this reason** and is retracted here: the field exists, the value is a constant.
* `time_basis` is **`untimed` on 691 of 779** traffic-light annotations and
  **33 of 38** lane-change ones. The remaining `segment` does **not** mean a
  measured time either — it does not correlate with grounding (34 of 88 TL
  `segment` annotations are `grounded: False, corroboration: none`).

⛔ **And there is no per-frame perception signal to time them with.** The
record's `scene` block is **ONE grounding box from ONE question per clip**
(`provenance: alpamayo-grounding-box`), and the traffic-light question was asked
on only **998 of 4,572** clips. ⚠️ So `traffic_light_visible == False` on the
other 3,574 is **NOT-PROBED, not absent** — the classic absence-vs-not-probed
trap, and my own first join miscounted it because the probe matched the *key
name* rather than the *value*. Corrected cross-tab:

| | `tl_visible` True | False |
|---|---|---|
| has a TL token | 175 | 604 |
| no TL token | 692 | 3,101 |

The only other per-frame box source, `obstacle.offline`, has a 10-class enum of
**dynamic agents only** — no traffic light.

⭐ One untapped lead, stated as a lead and not a result: **2,838 of 4,572 CoT
texts (62.1 %) contain an explicit `"<n> s"` timestamp.** Whether those refer to
the token's event is unverified. That is the cheapest path to timing the CoT
half, and it is a Data-Eng work item, not this one.

### The regeneration cost — priced on the CONSUMER's loader

⛔ Priced by opening what the trainer opens, not a file found on disk: the
consumer is `v7_labels.load_v7_labels`, called from `refc_v3_train.py` via
`--v7-labels`, reading the **gzipped JSONL** — and its per-window entry points
(`window_in_band`, `tactical_class_ids`, and now `tactical_goal_targets`)
**already take `t_now_s`**. The consumer is per-window; only the FILE is
per-clip. MEASURED on 60 real clips, dev box, single process, no GPU:

| | median | mean |
|---|---|---|
| `ES.load` | 4.2 ms | 10.2 ms |
| geometry-only recompute at a NEW anchor | **0.3 ms** | 0.4 ms |
| full `emit_one` (pays the CoT/alpamayo layer once) | 4.5 ms | 19.6 ms |

| anchors/clip | stride | wall-clock, 4,572 clips | blob | band coverage of a 34.2 s episode |
|---|---|---|---|---|
| 1 (today) | — | 0.3 min | 2.0 MB | 12.0 % |
| 5 | 6.8 s | 0.4 min | 10 MB | 59.9 % |
| **9** | **3.8 s** | **0.5 min** | **18 MB** | **100 %** |
| 35 | 1.0 s | 1.2 min | 70 MB | 100 % |

⇒ **Full geometric coverage costs well under an hour of engineering and about a
minute of compute.** There is no capacity wall. ⚠️ Two caveats that belong in
the same breath: **58.7 % of a record is clip-CONSTANT text** (`cot_source`,
`scene`, `semantics`, `alpamayo`), so a K-anchor blob must split clip-level from
anchor-level fields or it duplicates that K times; and ⛔ **the bands are a
derived geometry — state `[2.0, 6.0]`, `t0_s` and the resulting half-width for
every cache the change meets**, or a "reproduction" silently becomes a different
experiment, as `HORIZON` 7 → 8 once did.

⭐ **Verdict.** Regenerating the writer is worth more than any head for
`tac_lat` / `tac_lon` / `FOLLOW_LANE` / `SPEED_BAND` / `TURN_*` / `STOP_POINT`
— it takes them from 12 % to 100 % of frames for ~1 minute of compute. It does
**nothing** for the traffic light or the lane change. Those two need a *timing*
signal that does not exist in the corpus today.

---

## P2 — the head, and the one change that made the PI's tokens trainable

`v7_labels.HEADS` is untouched: still the four SOFTMAX heads. The goal set is
**MULTI-LABEL** — MEASURED 2–7 tokens per record, mean **2.751** — so it gets its
own surface, `TAC_GOAL_TOKENS`, and 22 independent sigmoids under BCE. ⛔ Folding
a SET into `HEADS` would make it look like a CHOICE (the 5-way-softmax defect one
layer up) and would silently break `head_mask`, `class_weights` and
`assert_mask_matches_presence`, all of which index a single-label vocabulary.

### ⛔ The trap that nearly sank it: absence is not a negative

For a `geometry` token the emitter is exhaustive, so absence IS a negative. For a
`vlm-cot` token absence means *"the caption stayed silent"*. Supervising those as
negatives would teach the head that ~78 % of the corpus has no traffic light, on
no evidence.

⚠️ And the **frozen declaration disagrees with the blob**:
`TACTICAL_GOAL_NEEDS_PERCEPTION` omits `YIELD` (609), `EVADE_IN_CORRIDOR` (238),
`LANE_CHANGE_L` (23) and `LANE_CHANGE_R` (15) — all CoT-sourced in the data. So
the policy is **derived from the loaded split's own provenance**, exactly as
`effective_mask` derives from presence, and never from the constant.

**But an honest provenance policy alone leaves all 15 CoT tokens with ZERO
supervised negatives** — a BCE head with positives only learns "always 1". That
is a diagnosis, not a product.

⭐⭐ **The lever: ENTAILED negatives from the frozen exclusion table.** If a
record carries `TRAFFIC_LIGHT_REACT_GREEN`, then `..._RED` is FALSE **by
entailment** — `TACTICAL_GOAL_EXCLUSIVE` states the pair, and the emitter
validates every record against it. Reading a negative off that assumes nothing
about caption completeness. Result, MEASURED on the train blob:

| | trainable tokens (pos **and** supervised neg) |
|---|---|
| provenance policy alone | **7 / 22** — no TL, no LC |
| **+ entailed negatives** | **17 / 22** — **all four TL, both LC** |

| token | pos | neg | prevalence | `pos_weight` |
|---|---|---|---|---|
| `TRAFFIC_LIGHT_REACT_RED` | 376 | 403 | 8.22 % | 1.1 |
| `TRAFFIC_LIGHT_REACT_GREEN` | 363 | 416 | 7.94 % | 1.1 |
| `TRAFFIC_LIGHT_REACT_YELLOW` | 22 | 757 | 0.48 % | 34.4 |
| `TRAFFIC_LIGHT_REACT` | 18 | 761 | 0.39 % | 42.3 |
| `LANE_CHANGE_L` | 23 | 290 | 0.50 % | 12.6 |
| `LANE_CHANGE_R` | 15 | 282 | 0.33 % | 18.8 |

⛔ **Five tokens stay MASKED and the reason is named per token**: `SPEED_BAND`
(prevalence **100.0 %** — a constant, so a binary logit carries no information),
`YIELD`, `CORRIDOR_OFFSET`, `GAP_TARGET`, `REACT_ON_ONCOMING` — CoT-backed with no
exclusion partner. An unmasked logit there could only ever be pushed towards 1.

### Class imbalance is reported, never pooled

⛔ `LANE_CHANGE_R` at 15 of 4,572 will be predicted never by any unweighted loss,
and a head that predicts the majority scores 99.67 % on that column.
`per_class_scores` reports **recall, precision, n_pos, n_neg and n_fired per
class** and there is no pooled accuracy anywhere. `majority_control_scores` is
the control that **must read its known value** — recall exactly **0.0** where the
majority is absent, exactly **1.0** where it is present. MEASURED precedent: a
`turn_left` recall of exactly 0.0000 of 11, replicated at both seeds, sitting
beside an apparently-healthy safety number.

---

## The six links, asserted POSITIVELY — `stack/tests/test_tac_goal_wiring.py`

17 tests, CPU only, no pod / no Thor / no GPU. Each assertion is paired with a
same-breath control that must read a different known value.

| link | assertion | control in the same breath |
|---|---|---|
| L1 label in a real record | `RED in label.tac_goals`, `meta[RED]["state"] == "red"` | `GREEN not in` — content, not a truthy container |
| L2 reaches the batch | positive `w=1`; **GREEN `w=1` by entailment** | `GAP_TARGET` `w=0` — if it reads 1 the ignore policy is gone |
| L2b band | out-of-band all-ignored | in-band supervises ≥1 |
| L3 reaches the forward | `out["tac_goal_logits"].shape == (B, 22)` | the pre-existing geometric `g_tac` still there at its own width |
| L3b | `kin3` builds **no** head | same build still emits `lat_logits_tac` |
| L4 loss with non-zero weight | `loss > 0`, `head.weight.grad is not None`, `|grad| > 0` | `core.route_head.weight.grad is None` |
| L4b zero weight | at weight 0.0 grad is a **zeros tensor, not None** | at weight 1.0 the same term moves it |
| L5 evaluated | per-class recall = **exactly 0.0** for a never-fire head with real positives | an all-fire head reads **exactly 1.0** on the same cells |
| L5b | majority control at its known 0.0 / 1.0 | its precision equals the prevalence |
| L6 inference | forward emits from frames + measured `v0` only | the signature exposes **no** goal-target channel — echo impossible by construction |

### ⭐⭐ The mutation, and the control that makes it mean something

Flip `TRAFFIC_LIGHT_REACT_RED` → `..._GREEN` in the label. The supervised-cell
count is asserted **unchanged** (so a loss difference isolates the label), and
both the **loss** and the **gradient** must move. The discriminating control
flips the target on a cell the policy IGNORES — the target tensor genuinely
changes and loss and gradient must not move **at all**. The two differ in exactly
one thing: whether the cell carries evidence.

⚠️ **Mask checked before anything was called dead.** An out-of-band batch reads
loss 0.0 with `n_supervised == 0`; forcing the mask on moves it — the same
discrimination that showed a sibling's "dead" route head was a validity mask
(0.0 → 0.687).

### ⛔ The suite was blind once, and that is why it is trusted now

Two deliberate regressions were run against it:

* **A — drop the head's output from the forward dict** (the `FORWARD_KEYS` drift
  class): **6 tests fail**, correctly.
* **B — reintroduce `if n_sup == 0: return zeros(), 0`** (the 52.2 %-of-budget
  guard): the suite stayed **17/17 GREEN**. It could not see the defect it was
  written to prevent. A missing assertion was added — an all-ignored batch must
  still leave `p.grad` non-None — and regression B now fails **exactly one test**
  with a named message, while the fixed build is back to 17/17.

That is the same lesson as the AST census that read 0 suspects on both the fixed
and the broken trainer: **a guard must be tested by mutation, never by
inspection.**

### Controlled suite comparison

`test_tactical_label_reach.py test_label_vocab_audit.py test_v7_wiring.py
test_refcv3_arm.py test_refcv3_ablations.py test_v7_eval_exclusion.py
test_refc_v3_nav_from_v7.py test_vocab_v7_frozen.py`

* baseline tree: **143 passed, 1 skipped**
* wired tree: **1 failed, 142 passed, 1 skipped** — the single failure is
  `test_v7label_exposes_no_tactical_goal_set_field`, the sibling's own gap-pin,
  whose failure message reads *"That is the gap closing -- update the docs."* It
  fired as designed and has been inverted into a gap-**closed** pin in the same
  commit, with its docstring's stale row corrected.
* after that update: **35 passed** across the three label tests, and the full
  eight-file set is green again.

---

## The two zero-weight defaults — written up, NOT flipped

⛔ Flipping a default silently changes the recipe every banked arm was trained
under. Both are registered as future arms with their cost, and neither is touched.

**`GOAL_POINT_WEIGHT_DEFAULT = 0.0`** — `goal_point` **is** an output key that
receives no gradient: a head that emits without being taught. ⚠️ **A sibling owns
`goal_point.py`; this is an escalation, not an edit.** The literature puts a goal
POINT at **+4.7 PDMS** against +0.2 for a categorical command, so this is
plausibly the largest single un-taken lever in the file. Cost: one training arm;
no new data. ⚠️ Its admissibility is already settled — a *predicted* geometric
goal point is the PI-approved goal signal, and `V7Label.tac_anchor` supplies it.

**`AGENT_WEIGHT_DEFAULT = 0.0`** — every agent term is guarded by `w_agent > 0.0`,
so agent/obstacle GT is skipped entirely. ⛔ **Blocked, and the blocker is data,
not a weight**: turning it on needs the **B1 TRAIN join, which does not exist**.
The EVAL join was built today (139/141 clips). Cost: build the train join first,
then one arm.

⚠️ Note the asymmetry with `--w-tac-goal`, which also defaults to 0.0: that one is
**unguarded**, so at weight 0.0 its head still receives a zeros gradient and
`p.grad is None` still means "never wired". These two are guarded, which is why
they read as structurally absent. The distinction is the whole point of L4b.

---

## Out of scope, per the PI

* The **strategic vocabularies stay unwired**: *"remove the strategic layer in
  the next experiments, feed the nav command to tactical and operative planning,
  solve the driving task, then add the strategic layer."* A sibling is building
  `--no-strategic`.
* The **nav command is an INPUT, never a target** — *"it is an INPUT simulating
  the nav system of the vehicle."* No oracle-nav target, no "deployment gap".

---

## Can the model now be taught to brake for a red light and to change lanes?

**The traffic light: yes, and it is now the strongest of the two.** The colour
reaches a supervised head from vision at 376 RED / 363 GREEN with 403 / 416
entailed negatives and `pos_weight` ≈ 1.1 — a balanced, well-posed problem.
⛔ What still blocks it: the label is **clip-level and untimed** (`t_nominal_s` is
a constant 4.00 on all 3,144), so the head learns *"this clip contains a red-light
reaction"*, **not** *"brake NOW"*, and it is supervised on 12 % of frames. Closing
that needs a timing signal the corpus does not have — the 62.1 % of CoT texts
carrying an explicit `"<n> s"` is the cheapest lead.

**Lane changes: wired, but the ceiling is the label, not the head.** 23 + 15 = 38
positives on 4,572 clips, **100 % CoT-sourced**, and `ego_manoeuvre.py` has **no
lane-change branch at all** — a real lane change is absorbed into `NUDGE` or
`LANE_KEEP` depending on yaw. ⛔ What blocks it: a lane-detector reference. Until
one exists, a lane-change head is a 38-example classifier and the honest
expectation is a low, reportable recall — which is why per-class recall and the
majority control are mandatory in the eval, not optional.

⚠️ Neither claim is a performance claim. **No arm has been trained.** The
deliverable is wired, tested, pre-registered code plus the launch command; the
A40 runs refcv5 to ≈2026-09-08 07:33 UTC and was not touched.
