# INTEGRATION CLOSE — the four escalations left by `06b8782`

**Date** 2026-08-16 · **Branch** `agent/arch-inf-20260803` · **GPU used: none**
(Thor is running the v6F S-W 30k; nothing here touched it).

**Suite** (invocation quoted per C82/C84 — a count without its command is a
report about a shell): `cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 pytest -q`
→ **3689 passed · 0 failed · 7 skipped · 2 xfailed**, exit 0, 374 s. Baseline
was 3658 passed; **+31 = the tests added here** (10 `test_v6_seam_dump.py`,
21 in `test_v6_s2_loss.py`). Per C82 this is quoted as *my change adds N
passing tests and introduces zero NEW failures*, not as a global total — the
tree moves under every measurement with other agents live. See §4.5(b) for one
intermittent test observed once and not attributable.

**Safety baseline, re-measured by me before the first edit and again after the
last:** default `V6Stack(V6Config())` = **87,893,449 params / 405 state_dict
keys** — matches the brief exactly, unchanged throughout. No v6 vocabulary
tuple was touched. `test_v6_gstr_port.py` (38 tests, incl.
`test_default_forward_is_bit_identical_and_emits_no_new_key`) passes.

---

## 0. Headline — what needs a decision

1. ⭐ **The PI's question about the 80 ex-lane-change labels is ANSWERED, and
   the answer needed no new mechanism: all 80 are route-following, by the
   route engine's own token.** Residual abstention is **0**. §2.
2. ⚠️ **I built an `a_str` abstain channel that the corpus does not need.** It
   is default-off, provably inert and tested — but **zero records use it**.
   Keep or revert is a call I am flagging rather than making. §2.5.
3. ⚠️ **`LANE_TARGET`'s SPEC, not just its implementation, is refuted.**
   `HIERARCHY_VOCABULARY.md` §3 derived it from "lateral displacement events";
   that signal provably cannot support it. Fixed in the doc. §4.
4. **F-16's seam instrument can now be run** — but it will return DEGENERATE
   until an S-T checkpoint exists. It needs someone to point it at one. §3.

---

## 1. `--s2-labels` now points at the corrected labels (priority 1) ✅

**The defect.** The corrected label set existed at
`…/2026-08-16-s2-v1-labels/review/labels_v2/` and **nothing consumed it**. The
superseded v1 directory was still what a launch line would pick up, and the
failure is INVISIBLE: same 797 records, same index, same band, every guard in
`load_s2_labels` passes identically. It is simply wrong in 80 rows. (C77's
shape: a well-formed artifact whose payload is the problem.)

**The fix — three parts, so the wrong answer cannot be reached silently:**

| | |
|---|---|
| `s2_labels.S2_CANONICAL_LABELS_REL` | THE one name. `--s2-labels`' own `--help` prints it (imported, never retyped), and the test resolves through it. |
| `…/labels/SUPERSEDED.json` | a marker in the DEAD directory. `load_s2_labels` **refuses** any dir carrying it (and any single `.jsonl` inside it), printing the replacement path. ⛔ The marker lives in the superseded dir, never in the canonical one — "this is current" is a stored verdict that rots; "this one is dead" only becomes more true (C81). |
| cross-check test | the marker's `superseded_by` and the code constant must resolve to ONE directory. C81's rule: where a fact is written twice, audit the copies against each other. |

**MEASURED through `load_s2_labels`, both directories, n=797 each:**

| | v1 (superseded) | v2 (canonical) |
|---|---|---|
| `g_str` | FOLLOW 395 · **LANE_TARGET 80** · TURN_L 137 · TURN_R 113 · STOP_AT 59 · ABSTAIN 13 | FOLLOW **474** · TURN_L 137 · TURN_R 113 · STOP_AT 59 · ABSTAIN 14 |
| `a_str` | HOLD 526 · **P_LC 80** · P_STOP 88 · REDUCE_TO 85 · RESUME 18 | HOLD **597** · P_STOP 88 · REDUCE_TO **94** · RESUME 18 |

`clip_index.json` is **byte-identical** between the two (md5 `ee975cac39c4`) —
only the label records changed, so the join is untouched.

⚠️ **The C83 glob trap was checked, not assumed.** `load_s2_labels` globs
`s2_labels_*.jsonl` over the directory. `review/labels_v2/` contains exactly
`clip_index.json` + the two canonical `.jsonl` under their ORIGINAL names, so
there is no `*_v2.jsonl` duplicate-clip_id hazard; and `SUPERSEDED.json` cannot
collide with the glob or with `INDEX_NAME`.

⚠️ **Safe to do now:** `w_s2_goal` defaults to 0.0 and `V6LossWeights.for_stage`
zeroes it in S-W/S-T, so no live run reads these labels. It **must** be done
before it is ever raised.

---

## 2. The 80 ex-lane-change labels — RESOLVED, not declined (priority 2, per the PI's redirect)

**The PI, verbatim:** *"we need to investigate what are these labels and which
are not classified and give them one by adjusting the approach or confirm their
cases for follow route"*.

### 2.1 The answer

**All 80 are route-following cases, and the route engine had already said so.**

⭐ **MEASURED, `labels/engine_a_*.jsonl` (801 clips, the primary artifact):
`engine_a.route.token == "follow"` with `token_valid: true` for 80/80.**

The removed geometric gate was **overriding its own route engine**. For every
other clip the route token already won — 189 clips carry an `lc_*` event with a
non-`follow` route token and were labelled by the route. The lane-change branch
was the single exception, and removing the exception resolves all 80.

### 2.2 The partition, with counts

| bucket | n | basis |
|---|---|---|
| **route-following** → `FOLLOW_MAIN_ROAD` | **79** | `route.token == follow`, `token_valid` |
| **shadowed longitudinal manoeuvre** → `REDUCE_TO` | **9** (8 inside the 79 + the 1 below) | the longitudinal engine the lane-change branch was suppressing. **Cleanly separated: median `net_dv` +0.27 m/s for the 71 `HOLD_CORRIDOR` vs −3.35 m/s for the 9 `REDUCE_TO`.** The coordinator's "known non-empty" bucket — confirmed, n=9 |
| **genuinely undecidable** → `NONE_ABSTAIN` | **1** | already expressible with the EXISTING `g_str` token |
| **real lane change** | see 2.3 | **not a strategic bucket at all** |

⇒ `a_str` residual abstention: **0**. `g_str` residual: **1**, via a token that
already exists. **No new vocabulary entry was needed and none was added.**

### 2.3 The PI's 4 "correct" clips are NOT flattened — they are re-homed

4 of the 19 reviewed clips were adjudicated **correct** (13 wrong, 2 left null).
They are not forced into `FOLLOW` by an argument from convenience:

`LANE_TARGET` is a **STRATEGIC** goal (8–30 s) whose args are
`lane_offset_idx, deadline_m` — a lane *assignment serving a route*. With
`route.token == follow` there is by construction no route reason for one.
A lane change *observed* inside a route-following segment is a **TACTICAL**
event, and the vocabulary already has the token for it:
`a_tac: LANE_CHANGE_L/R(within_m)` (HIERARCHY_VOCABULARY §4). The strategic
label being `FOLLOW_MAIN_ROAD` does **not** deny the lane change; it puts it at
the layer that can express it.

⚠️ **UNVERIFIED, and it is a question for the PI, not something I resolved:**
the review sheet's section is named `S1_LANE_TARGET`, so "correct" could mean
*"the strategic LANE_TARGET label is right"* **or** *"yes, a lane change
happens here"*. Three of the four carry no note. The fourth (`7a28718f`) reads
*"the vlm ist not readin the sign, follow route in direction X"* — which
describes route-following. **⇒ ESCALATION: these 4 clip ids belong to the
tactical-labels stream as candidate `a_tac` lane changes.** I did not write
into that stream's directory (ownership).

### 2.4 What I did NOT use, and why — the trap the coordinator named

I was briefed that Alpamayo is the strongest leg and that it corroborated the
PI at 2/81. **It corroborates the AGGREGATE and does NOT discriminate PER
CLIP**, and building the rule on it would have inverted the one real signal:

- **Coverage:** Alpamayo covers **23 of the 80** (28.8 %) and 257 of the 797.
  **57 of the 80 (71 %) have no external leg at all.**
- **Per-clip cross-tab (n=17 adjudicated, PI × Alpamayo `lane`):**

  | | Lane Keep | Left Lane Change | Slightly Shift Left |
  |---|---|---|---|
  | PI **correct** (4) | 3 | 0 | 1 |
  | PI **wrong** (13) | 12 | **1** | 0 |

  ⛔ **Alpamayo's single `Left Lane Change` call lands on a clip the PI
  adjudicated WRONG, and 3 of the PI's 4 CORRECT clips are `Lane Keep`.** At
  this n it carries no information about the adjudication.
- **The VLM leg is its own fallback:** `vlm_goal_kind == follow_main_road` on
  the 80 (79/80, `agrees: false` 80/80) looks like corroboration, but the same
  VLM says `follow_main_road` on **48 of 84** missed left turns and **45 of 94**
  missed right turns. It under-calls, so its "follow" is weak evidence.
- **Ego geometry is refuted as a discriminator:** the stream's own 34-feature
  search over the adjudicated set found **nothing separating** — best
  `yaw_lin_r2`, **FWER p = 0.29**, no perfect separation, **n = 18**. The study
  was not underpowered by construction (min attainable two-sided p = 0.00065).

⇒ Had I built a promotion rule on Alpamayo, it would have promoted a PI-wrong
clip and flattened 3 PI-correct ones. **The rule is the route token instead** —
corpus-wide (801/801), independent of the refuted signal, and already landed.

### 2.5 The derivation was ALREADY adjusted — I verified it rather than rebuilding it

Reading the primary artifact before asserting (C82), the fix is in
`stack/scripts/s2_derive.py` §LC, landed in `06b8782`:
`lane_change_requirement()` reads the ROUTE and LANE CONTEXT only and is
**structurally unable to read `lane_change_events`**.

**I re-ran the production derivation myself over all 801 engine_a records, no
VLM symbols, no lane context (the corpus's actual state):**

```
LANE_TARGET emitted            : 0
PREPARE_LANE_CHANGE emitted    : 0
lane_change_requirement.required: None for 801/801   (lane context absent)
g_str: FOLLOW 491 · TURN_L 137 · TURN_R 113 · STOP_AT 59 · NONE_ABSTAIN 1
a_str: HOLD 601 · REDUCE_TO 94 · P_STOP 88 · RESUME 18
```

**Behaviour outside the 80 (the generalisation statement the PI asked for):**
the rule is a PRECEDENCE rule, not a threshold — the route token governs, and a
lane-change token requires a route-serving REQUIREMENT. On the 4,472-clip build
it emits `LANE_TARGET`/`PREPARE_LANE_CHANGE` for **exactly zero** clips until a
lane context (`n_lanes_same_direction`, `ego_lane_idx`, `route_lane_idx`,
`lane_continues`) exists; PhysicalAI-AV ships no map or lane graph, so that is
the honest state. It fits nothing to the 80.

⚠️ **Scale of the over-call, for the record:** 364 of 801 clips (**45.4 %**)
carry an `lc_*` ego-geometry event, against Alpamayo's corpus lane-change rate
of 104/4,425 (**2.35 %**) — the geometric detector over-called ~19×.

### ⚠️ 2.6 The abstain channel I built — it is INERT and I am flagging it

I implemented the per-family abstain mask **before** the PI's redirect arrived.
It is complete, tested, default-off and provably inert:

- a label block may carry `"abstain": true` (and then NO token, no set args);
- the loader emits optional `g_str_valid` / `a_str_valid` batch keys **only if
  some record abstains** — otherwise the batch is the identical seven-key dict
  and the loss is bit-identical (both branches pinned);
- `_s2_family_valid` ANDs the mask, so it can only ever REMOVE supervision;
- MEASURED: an all-False `a_str_valid` sends **zero gradient** to
  `act_head_str.*` while `goal_head_str.*` still trains (isolated on `L["s2"]`,
  not on the total loss — the total legitimately reaches `act_head_str` by
  other terms, and measuring that would have proved nothing).

⛔ **But the corpus needs it for 0 records** (§2.2). Per the PI's "build it only
if bucket 4 is non-empty", the honest status is: **built, unused, 0 consumers.**
This codebase punishes advertised-but-inert machinery. **I recommend keeping it
— `a_str` genuinely has no abstain token and any future uncertain action source
hits this wall — but the decision is the orchestrator's/PI's, and reverting it
is a clean revert of `s2_labels.py` + `s2_goal_loss` + one test block.**

---

## 3. The 60-step plan is now bankable (priority 3) ✅

**The defect.** F-16's probe (`taniteval/taniteval/seam.py`,
`taniteval/tools/seam_probe.py`) is complete, self-tested and validated, and
has produced **zero real-arm numbers** — because the emitted plan existed only
inside `V6Stack.emit`'s return value and the trainer saves only checkpoints.

**The fix.**

- **`taniteval/taniteval/seam_dump.py`** — the producer side, `SEAM_INSTRUMENT.md`
  §8's six lines made into a function with its refusals. It refuses: a missing
  controls/waypoints pair, a fan with no selector (guessing candidate 0 would
  probe a trajectory the planner never proposed), a wrong-length `eid` (it is
  the CI's episode resampling unit — a per-window eid silently narrows every
  interval), a misaligned `gt`, and an all-zero plan.
- **`train_v6_staged.py --dump-seam-plan DIR`** — DEFAULT-OFF, writes one
  `seam_<step>.pt` at the CHECKPOINT-SAVE boundary. ⭐ **Zero extra GPU:**
  `L["out"]["plan"]` is already computed for the step's loss; this is a
  `.detach().cpu()` and a `torch.save`, never a re-run, never per step. The
  whole block is wrapped so a dump failure **prints and continues** — a
  diagnostic must never kill a run (the inverse of the `t1_eval`
  analysis-time-refusal trap).

**Validated end to end, no GPU:** a REAL `V6Stack.emit()` output → the dump →
the REAL `seam_probe.py` CLI as a subprocess → a scored record.
`test_v6_seam_dump.py` (10 tests).

⚠️ **WHERE IT WILL AND WILL NOT PRODUCE A NUMBER.** At **S-W the emission head
is at its zero-init**, every control is exactly (0,0), and the probe correctly
returns **DEGENERATE** — the right answer, not a pass. `seam_dump_from_plan`
refuses to bank that by default. **⇒ this is an S-T-and-later instrument, and
the live v6F run is exactly the refused case.** Stated in the flag's `--help`,
in the module docstring, at the call site, and in `DIAGRAM_CONFORMANCE.md`.

⚠️ **MEASURED and worth knowing:** the DEFAULT `V6Config` emits a fan of 8 with
**no `sel_score`** — the selector is opt-in (`--selector`). A selector-less arm
is refused rather than guessed, so a dump run against such a config banks
nothing and says why.

**⇒ ESCALATION: this needs an S-T checkpoint pointed at it. Nothing else
blocks a real X2 number.**

---

## 4. Stale docs (priority 4) ✅

| site | was | now |
|---|---|---|
| `HIERARCHY_VOCABULARY.md:67` | `LANE_TARGET` ← "lateral displacement events (E4.1 LAT)" | ⛔ derivation **refuted** — and it is the SPEC that was wrong, not only the code. Carries the 34-feature/FWER-0.29/n=18 refutation, the route-demand replacement, the `required=None` 801/801 state, and the note that the token STAYS in the vocabulary (it sizes an embedding table) |
| `S2_STRATEGIC_GAP.md:126` | same derivation in the arg-slot table | ⛔ NOT EMITTED, with the 45.4 % vs 2.35 % over-call figure and the test that pins it |
| `S2_LOSS.md:148` + launch line | v1 census + `--s2-labels <…/labels>` | superseded banner + corrected census + the canonical path; the v1 census is KEPT and marked HISTORICAL (do not rewrite history) |
| `DIAGRAM_CONFORMANCE.md:131` | X2 seam instrument ⬜ NOT BUILT | 🟨 **BUILT, WITH ZERO REAL-ARM NUMBERS** — both halves, plus why S-W cannot supply one |
| `DIAGRAM_CONFORMANCE.md:183` | perception agent slots ⬜ NOT BUILT | 🟨 PARTIAL — module built, **3,207,445 params** (MEASURED by me: default 87,893,449/405 vs slots-on 91,100,894/467), default-off; **remaining: no train-corpus obstacle join, never trained.** "presence ≠ capability" stated explicitly |

Every status carries its evidence class and what would change it.

---

## 4.5 Two suite findings, both reported rather than smoothed over

### (a) A real regression I introduced and the guard caught ✅ FIXED

My first version imported `S2_CANONICAL_LABELS_REL` from `s2_labels` at module
level in `train_v6_staged.py` — purely to print it in `--help`. That added
`s2_labels` to the trainer's **import-time closure**, and
`tests/test_runbook_commands.py` failed, correctly: **the closure is exactly
the set of files that must be FILE-SHIPPED to a pod** (pods have no git
credentials). A help string would have made `s2_labels` mandatory for every
launch and given any pod without it a `ModuleNotFound` at startup.
⇒ reverted to a module-local literal, with a test asserting the two copies are
equal and that the help actually carries the path (C81). `s2_labels` stays a
LAZY import on the paths that use it.

### (b) ⚠️ An INTERMITTENT failure I could not attribute — flagged, not explained away

`test_v6_ladder_edges.py::test_after_init_from_exactly_the_intended_groups_train[S-J]`
failed **once** in a full run and I could not make it fail again:

| run | result |
|---|---|
| full suite (run 2) | **1 failed** |
| `test_v6_ladder_edges.py` alone | 26/26 passed |
| `gstr_port + s2_loss + ladder` | failed **once**, then passed **3/3** |
| `gstr_port + seam_probe + ladder` | passed |

⛔ **I initially called this "a reproducible interaction with my test modules"
on ONE failure and ONE pass. That was wrong** — three consecutive repeats of
the identical command then passed. Logging it because it is the C84 shape:
a theory built on a thin sample, and the tell was that I had not repeated the
experiment before naming a mechanism.

**What is true:** the test is intermittent. It seeds the MODEL (`mk("goal",
seed=61)`) but its batch draws from the GLOBAL RNG, so its state depends on how
much RNG every preceding test consumed — a latent order-dependence that adding
*any* test can shift. **UNVERIFIED whether it pre-dates my change**; I did not
establish that either way, and my change touches neither the ladder, the freeze
map, nor `init_from`. ⇒ **ESCALATION: it needs a seeded batch (the
`sigreg_generator` treatment) and its own investigation — not a retry loop.**

---

## 5. Deliverable manifest

| artifact | where | status |
|---|---|---|
| `stack/scripts/s2_labels.py` | repo | modified, staged |
| `stack/scripts/train_v6_staged.py` | repo | modified, staged |
| `stack/tests/test_v6_s2_loss.py` | repo | modified, staged |
| `stack/tests/test_v6_seam_dump.py` | repo | **new**, staged |
| `taniteval/taniteval/seam_dump.py` | repo | **new**, staged |
| `…/2026-08-16-s2-v1-labels/labels/SUPERSEDED.json` | repo | **new**, staged |
| `HIERARCHY_VOCABULARY.md` · `S2_STRATEGIC_GAP.md` · `S2_LOSS.md` · `DIAGRAM_CONFORMANCE.md` | repo | modified, staged |
| this file | repo | **new**, staged |

**Nothing lives in only one place.** No pod, no worktree, no scratchpad. No
commit, no push, no branch switch (per the operating contract).
