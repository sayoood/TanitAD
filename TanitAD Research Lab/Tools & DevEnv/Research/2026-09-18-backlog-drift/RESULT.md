# BACKLOG.md staleness sweep + `backlog_drift.py` — the pull-list finally has the guard it asked for

**Stream:** Tools & DevEnv FlyWheel · **Date:** 2026-09-18 · **GPU:** 0
**Working tree:** `D:/Projects/TanitAD` @ `d221843` (2026-09-11), branch `agent/arch-inf-20260803`
**Subject:** `Project Steering/BACKLOG.md`, last written `8b694049` (2026-09-07), **37 commits ago**

---

## 0. The one-paragraph answer

`BACKLOG.md` diagnosed its own defect on 2026-09-02 and named the missing mechanism in the same
breath: *"the same discipline `lab_backlog_drift.py` enforces for the Lab's list, **WHICH THIS LIST
HAS NO EQUIVALENT OF**."* It now has one. `tools/backlog_drift.py` reads **117 distinct ids over 133
occurrences** and finds **8 rows that contradict the file's own contract** (`BACKLOG.md:8`, *"Strike
items through when done"*) — 7 declare DONE with an unstruck id, 1 has a cleared blocker. Those 8
are the machine-checkable part. The **larger finding is structural and the machine cannot see it**:
**11 of 11 rows that say "blocked on the PI" are absent from the live `PI_DECISION_QUEUE.md`**, and
the B/E/F sections are addressed to a **fleet that no longer exists** (pod1–pod5), while the
programme's own A13 row — inside this file, dated 2026-09-07 — records Thor + an A40 + the dev-box
4060. ⇒ **A puller reaching for "gated ≠ idle" work today is reading a 2026-08 map.**

**Two findings outrank every individual row verdict**, and both came from probing rather than
reading:

* ⛔⛔ **The file's loudest item — `⛔⛔ BLOCKING INTEGRATION ITEM … Blocks every refcv5 arm and every
  TanitLang attachment` — was fixed 12 days ago** (`bb4479c`, 2026-09-06). Its own control still
  reads correctly, and every symbol it reported as `0` in `HEAD` now reads 13–56. **§2.4.**
* ⛔⛔ **`pytest stack/tests` does not COLLECT** — `Interrupted: 2 errors during collection`,
  `PYTEST_RC=2`, **zero tests run**. F5's "23 standing failures" is superseded: the invariant
  *"`pytest -q` must stay green before any commit"* is currently unverifiable by that command.
  **§5.8.**

⛔ **Per Rule Zero this turn does not stop at the diagnosis.** The instrument is built, tested,
mutation-proven (12/12 RED) and banked; the sweep names the exact edit for every stale row; and
§6 lists the 0-GPU items that are **actually unblocked right now**, which is what the list is for.

---

## 1. Evidence classes used here

| class | meaning in this document |
|---|---|
| **MEASURED** | I ran the probe in this working tree; the artifact path or the command is quoted |
| **INHERITED** | taken from a repo document without re-verification (always labelled) |
| **UNVERIFIABLE** | needs a host, a GPU or a person I cannot reach from the dev box; the required probe is named |

⛔ **A row is `DONE` here only on a positive artifact assertion** — a tracked path, a source symbol,
a registry field, a live test result. *"It looks finished"* appears nowhere. Where a probe returned
nothing I say so as **a claim about the probe**, and I ran a second one, per the operating
standard's *"absence found at ONE location is not absence"*.

⚠️ **Scope caveat, stated rather than discovered.** This sweep is against `d221843` (2026-09-11).
Work that landed between then and today, or that lives only on Thor/the A40, is **outside** it. Every
host-dependent row below is marked `UNVERIFIABLE` for exactly that reason rather than guessed at.

---

## 2. What the instrument found (MEASURED — `raw/backlog_drift_scan.json`)

```
[backlog-drift] rows read     : 117 distinct ids over 133 occurrence(s), 30 section(s)
[backlog-drift] last written  : 8b694049d  2026-09-07  (37 commit(s) to the repo since)
[backlog-drift] verdicts      : CLOSED=4  CLOSED-ELSEWHERE=14  DONE-BUT-OPEN=7  OPEN=91  STALE-BLOCKER=1
[backlog-drift] heading items : 2 (1 without a closure marker)
```

### 2.1 DONE-BUT-OPEN — the row's own body says the work landed, its id is not struck

| row | line | the row's own words | the edit |
|---|---|---|---|
| **A11** | 26 | `DONE 2026-08-18, 0 GPU` | `~~A11~~` |
| **A13** | 32 | `DONE 2026-09-07, 0 GPU` | `~~A13~~` |
| **B4** | 43 | `DONE 2026-08-03, 0 pod GPU-h` | `~~B4~~` |
| **C3** | 55 | `CLOSED 2026-08-16 — blocker CLEARED 2026-08-09` | `~~C3~~` |
| **R1** | 122 | `DONE 2026-09-02 (D-REFAV1-LEAD-BLOCK)` | `~~R1~~` |
| **R2** | 123 | `DONE 2026-09-02 (3a6a48c)` | `~~R2~~` |
| **R4** | 125 | `DONE 2026-09-02 19:26Z` | `~~R4~~` |

⭐ **These cost nothing to fix and everything to leave.** Each already carries its evidence; only the
markup lies. **A11 and A13 are the expensive ones** — both sit in section A, the *"0-GPU, ALWAYS
EXECUTABLE, no excuse to idle"* table, which is precisely where a gated turn looks first.

### 2.2 STALE-BLOCKER — the blocker is struck, the row is not

| row | line | state |
|---|---|---|
| **C2** | 54 | Blocker `~~v2corpus reaching 30 k (~17:00 UTC 2026-07-29)~~` is struck; the row then says **"RE-PROBE 2026-08-16 — the gating DATE passed 18 days ago and the gating FACT is unverified … UNVERIFIABLE, not cleared"**. It is now **51 days** past that date and the re-probe was never done. |

⇒ C2 is correctly **not** closed, but it is also not a pull-able item: it is a **re-probe request
that has aged 33 days since it was written**. It should be restated as such.

### 2.3 CLOSED-ELSEWHERE — 14 rows, reported and deliberately NOT gated on

R8 · R10 · R12 · R13 · R15 · R20 · R22 · R23 · R26 · R27 · R33 · R37 · R39 · R41 are closed by a
**later separate line** (`- **R8 — STRUCK 2026-09-03**`) instead of a struck id. That is a real
convention in the bullet sections and it does announce itself to a reader, so the tool counts it
and never fails on it. ⚠️ **But it is a second spelling of "closed" in one file**, and the A/B/C
tables use the first. Worth unifying when the list is next rewritten; not worth a gate today.

### 2.4 ⛔⛔ THE HEADING ITEM THE ROW PARSER IS BLIND TO — AND IT IS **DONE**, WHICH IS THE BIGGEST STALE ROW IN THE FILE

`L259 ⛔⛔ BLOCKING INTEGRATION ITEM — refcv5's model seams are NOT IN HEAD` has **no id**, so no
row check can see it. The tool counts and prints it rather than skipping it silently — **which is
the only reason it got looked at.**

It declares itself *"0 GPU. **Blocks every refcv5 arm and every TanitLang attachment**"*, and it is
**closed**. Re-measured on `HEAD`'s own blob at this tip, with the item's own control:

| symbol in `stack/tanitad/refs/refc.py` | the item's HEAD column (2026-09-06) | **HEAD today (MEASURED)** |
|---|---:|---:|
| `agent_tok` | 0 | **13** |
| `control_head` | 0 | **14** |
| `cross_agent` | 0 | **16** |
| `time_mlp` | 0 | **6** |
| `sampler` | 0 | **56** |
| `assert_seams_are_built` | **0** | **1** |
| `lan_to_cond` — **the item's own CONTROL, must read non-zero** | 7 | **7** ✓ |
| file size | 155,407 B | **214,793 B** |

`git log -S` dates it: **`bb4479c` (2026-09-06) — "refcv5 WP-4/WP-6: the seam that was missing
between two modules that already existed"**, extended by `835286c` (2026-09-08). ⭐ **The item's
one non-negotiable condition was met exactly**: it required that *"`assert_seams_are_built` lands in
the same commit as the seams it guards — a half-fix is worse than either"*, and `bb4479c` is the
commit that introduced **both**. `stack/tanitad/refs/refc_selector.py` is tracked.

⇒ **The item should be struck.** It has been presenting as a hard blocker on *every refcv5 arm* for
**12 days after it was fixed**, in the loudest typography in the file.

⛔⛔ **AND I ALMOST SHIPPED THE OPPOSITE.** The first draft of this section read *"MEASURED, and it
is still real"* — with the item's own 2026-09-06 numbers copied into it. That is **INHERITED wearing
a MEASURED label**, the exact class the operating standard's rule 1 names, committed inside a sweep
whose entire job is to catch stale numbers. It was caught by running the probe instead of quoting
the table, and by the control (`lan_to_cond` = 7) confirming the blob was genuinely being read.
*The row was stale in the direction nobody checks: it under-reported our own progress.*

---

## 3. ⛔⛔ THE FINDING THE MACHINE CANNOT SEE: "blocked on the PI" blocks nothing, because nobody asked the PI

**MEASURED 2026-09-18, two probes.** `Project Steering/PI_DECISION_QUEUE.md` carries **11 open
items** (numbered 1–11, read from its own headings). I searched it for every distinguishing phrase
from every BACKLOG row that claims a PI gate:

| BACKLOG row | its claimed PI gate | hits in `PI_DECISION_QUEUE.md` |
|---|---|---:|
| C4 | old CPU pod release — *"deletion needs the PI"* | **0** |
| C5 | X2 verdict run (30 pod-days) — *"NOT AUTHORISED without the PI"* | **0** |
| C6 | wheelbase fix — *"decision pending"* | **0** |
| R24 | authorise the **DINOv3 ViT-B/16 pull** | **0** |
| R30 | does `EVAL_DOCTRINE.md` admit refcv3 as T1 | **0** |
| R54 | **D3b** — which margin leads the H-vs-F table | **0** |
| R21 | cut-over for `--action-units steer` | **0** |
| R36 | cut-over for `cost_time_grid="tactical"` | **0** |
| R34 / R35 / R49 | design + cut-over decisions | **0** |

**11 of 11 absent.** Second probe (a repo-wide search for the same phrases in `Project Steering/`)
confirms the decisions were not taken elsewhere either: `wheelbase` resolves only to BACKLOG,
`BOOST_PROGRAM.md`, a feasibility doc and a 2026-09-06 MM-decisions file; `X2 verdict` only to
BACKLOG and two July reports; **R24's premise is independently confirmed still standing** —
`Decisions/2026-09-05-mm-decisions.md:1351` records *"the seed on the box is ViT-L/16, which
`build_trunk_anchor` refuses BY DESIGN against the ViT-B/16 trunk."*

⇒ **This is the operating standard's own rule, broken at scale:** *"Escalate integration, don't
write 'please merge' into a doc"* — an orthogonality instrument once sat unmerged for 10 days
because the request lived in a README nobody re-read. Here **eleven** requests live in a pull-list
nobody re-reads, wearing the words "blocked on the PI" as if that were an escalation. It is not. It
is a note to self.

⭐ **Root-cause class:** identical to the stale-blocker class, one level up — *a gate recorded in the
wrong file is not a gate*. **Recommended (Master Mind's call, not mine): promote C4/C5/C6, R24, R30
and R54 into `PI_DECISION_QUEUE.md` as items 12–17, and reduce their BACKLOG rows to a pointer.**
The cut-over decisions (R21, R34, R35, R36, R49) are arguably Master Mind calls, not PI calls, and
should be labelled as such rather than parked under a PI gate that was never opened.

---

## 4. ⛔ THE SECOND STRUCTURAL FINDING: sections B, E and F address a fleet that is gone

**MEASURED.** The section headers and blocker cells name the 2026-07/08 pod fleet:

* `## B. ONE GPU — executable now (pod3; ⛔ eval stays free for C64-A)` — *"executable now"* is a
  claim about **pod3**.
* `## E. 2026-08-11 NIGHT` opens with **"Running (do not duplicate): pod4 … pod5 …"** — a
  **2026-08-11 process snapshot presented in the present tense**, 38 days old.
* `## F — 2026-08-12 v6 launch pull-list` gates F2/F7/F8 on **"D2 (pod name)"**, a PI decision from
  2026-08-12.

Against that, **BACKLOG's own A13 row (2026-09-07, inside this same file)** states: *"refav1 is NOT
training on Thor — it finished 2026-09-04 … ⇒ **Thor is FREE COMPUTE**"*, and the D-ROLL-1h section
(2026-09-06) references **"the LIVE refcv5 A40 run"**. `Project Steering/LOOP_STATE.md`'s fleet
block still describes *"FOUR A40s + pod1's 8× A6000 blocked"* and was last touched at `a7626d5`.

⇒ **A file that contains both statements is not a pull-list, it is an archive with a pull-list
inside it.** ⚠️ I am **not** asserting pod3/pod4/pod5 are dead — that is an absence claim requiring
a live probe I cannot run from the dev box. The verdict is **UNVERIFIABLE, and the probe is named**:
`python tools/fleet_probe.py` (which exists precisely because *"a grep matching nothing prints
nothing, and printing nothing was scored as health"*). **Run it before pulling anything from B, E or
F.**

---

## 5. Per-row sweep — all 91 open rows

Verdict key: **DONE** (positive artifact, needs striking) · **OPEN** (confirmed live) ·
**STALE-BLOCKER** (the gate cleared, the row was not revisited) · **UNVERIFIABLE** (named probe
required) · **PARTLY STALE** (part of the row's premise has been overtaken).

### 5.1 Section A — 0-GPU

| row | verdict | evidence (MEASURED unless marked) |
|---|---|---|
| A3 | **PARTLY STALE** | The artifact exists: `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-02-v2-clean-val-selector/` is present. The row's own text already reduces it to *"now a PI decision: freeze n=400, or reject B"* ⇒ it is **not 0-GPU work**, it is **an unescalated PI decision** (see §3). Recommend: move to the PI queue, strike here. |
| A4 | **OPEN** | Research lead; no named artifact, no closure marker, id appears once. Nothing has closed it. |
| A5 | **UNVERIFIABLE** | Requires replaying `discover_r0_clips` on **pod2's raw root**. Row itself says the partial answer came *"with NO pod"*. Probe: pod2 access. |
| A6 | **UNVERIFIABLE** | *"Reconcile pod2's tree — 317 modified tracked files"*. Requires pod2. ⚠️ Caveat *"after v5 finishes"* — v5f completed **2026-08-09** (registry §1.8, INHERITED via C3's own closure note), so **the stated caveat has cleared** and only host access blocks it. |
| A7 | **OPEN** | HYPOTHESIS-class lead, no artifact named. |
| A8 | **OPEN** | HYPOTHESIS-class lead, no artifact named. |
| A9 | **UNVERIFIABLE / likely obsolete** | *"#37/#39 are marked completed but were falsified (pod1 still down; pod3 had drifted back)"* — refers to a **task list that is not identified by path** in the row. Two probes found no `#37/#39` task register in `Project Steering/`. ⇒ the row cannot be actioned as written. Recommend: name the file or strike. |
| A10 | **OPEN** | `sc_train.py` is tracked at `…/incoming/2026-07-26-situation-classifier/scripts/sc_train.py` (+ a `gen1_sc_train.py` sibling). Whether *other* pod-only siblings remain is a **host** question ⇒ the repo half is clean, the pod half is UNVERIFIABLE. |
| A14 | **UNVERIFIABLE** | 24 dumps **on Thor**; the row correctly assigns the call to the AlpaSim stream. Probe: Thor. ⚠️ Flagged DONE-BUT-OPEN by an early substring matcher — **false alarm**, the phrase was *"the AlpaSim **closed**-loop stream"*. Fixed before shipping (§7.2). |
| A13b | **OPEN, and the gate is REAL** | The row's own claim is confirmed: `pod_currency_audit.py`'s MSYS fix is landed and `stack/tests/test_pod_currency_audit_guards.py` is tracked. The **ship** has not happened (Thor-side ⇒ UNVERIFIABLE), but the row is correctly open and correctly conditional on *"only when a launch is imminent"*. |
| A13c | **OPEN — confirmed by field enumeration** | `MODEL_REGISTRY.md` §2.4 `refav1-b1-v72-ep3-speed` exists at `:2081` with exactly **7 fields** — `Status · Checkpoint · Reading it · Precision · Corpus · Ego input · Eval`. **No code-provenance field.** The one `30d6d60` mention (`:2090`) is about which stack to use *to read* the checkpoint, **not** which stack trained it. ⇒ the row is precisely, still, correct. |

### 5.2 Section B — "ONE GPU, executable now (pod3)"

⚠️ **The whole section carries the stale host premise of §4.** Per-row:

| row | verdict | evidence |
|---|---|---|
| B1 | **UNVERIFIABLE** | Marked *"IN PROGRESS"* with no date. A run state cannot be read from the repo. Probe: `tools/fleet_probe.py`. ⛔ *"IN PROGRESS"* with no date is the same defect as a blocker with no re-probe. |
| B2 | **OPEN** | No artifact; needs 2–4 GPU-h. |
| B3 | **OPEN, blocker still real** | *"depends on A2's chunk download"*; A2 is struck but its own text says **"⛔ Still NOT applied — needs a human-reference rollout + the chunk download"** ⇒ the dependency genuinely stands. ⭐ A correctly-carried blocker; the file can do this. |
| B5 | **OPEN** | The programme's flagged high-value experiment; no arm banked. |
| B6 / B7 / B8 | **UNVERIFIABLE** | Each points at a bare **task number** (`#31`, `#10`, `#29`) in a register the row does not name, and B6's blocker is *"queued since pod3 freed"*. Two probes found no such numbered register in `Project Steering/`. ⇒ **not actionable as written**; recommend naming the source or striking. |

### 5.3 Section C — GATED

| row | verdict | evidence |
|---|---|---|
| C1 | **UNVERIFIABLE + UNESCALATED** | pod1 console stop/start. Not in the PI queue (§3). |
| C2 | **STALE-BLOCKER** (tool-detected) | See §2.2. The re-probe it asks for is itself 33 days old. |
| C4 / C5 / C6 | **OPEN, but MIS-FILED** | All three say *"needs the PI"*; **none is in `PI_DECISION_QUEUE.md`** (§3). They are not gated, they are unasked. |

### 5.4 Section D — standing / long-horizon

| row | verdict | evidence |
|---|---|---|
| D1–D5 | **OPEN (standing)** | Programme-level goals, correctly long-horizon. ⚠️ All five point at bare task numbers (`#26 #23 #15 #5 #28`) from the same unnamed register as B6–B8. Recommend: re-anchor on `GOALS_AND_CLAIMS.md` ids, which are live and searchable. |

### 5.5 Section E — the 2026-08-11 night list

⚠️ **Header presents a 38-day-old process snapshot in the present tense** (§4).

| row | verdict | evidence |
|---|---|---|
| E1 | **STALE-BLOCKER** | *"unblocks on: nothing — runnable the moment T1 lands"*. **T1 has landed**: `taniteval/tools/t1_eval.py` is tracked, R2 (its review) and R13 (`"ha0": "T1"`) are both struck. ⇒ **the stated blocker cleared and the row was never revisited.** This is a 0-GPU item sitting in the "always executable" band, un-pulled. |
| E2 | **OPEN** | Blocked on *"the release row existing"* — a repo-side condition, cheap to settle. |
| E3 | **STALE-BLOCKER + UNVERIFIABLE** | *"pod5 after T1"* — T1 landed; pod5 is the dead-fleet question. |
| E4 / E5 | **OPEN** | GPU items with intra-programme dependencies (p8c gate). |
| E6 / E8 | **STALE-BLOCKER (premise)** | Both gate on **"v6 GO"**. The programme has since moved through v6 → v7 → refcv3/4b/5 → refcv6 (`Project Steering/PREREG_REFCV6.md` is in the index). ⇒ *"v6 GO"* is no longer a meaningful gate; the rows need re-basing or striking. |
| E7 | **OPEN** | `records.parquet` on pod4 ⇒ host-dependent, but the mapping table itself is 0-GPU. |
| E9 | **OPEN** | Analysis item, no artifact banked. |
| E10 | **OPEN** | `MODEL_REGISTRY.md` mentions Alpamayo **34×** but the row asks for the augmentation counts to have **their own row**; no such row found. Cheap 0-GPU registry work. |

### 5.6 Section F — the v6 launch pull-list (2026-08-12)

⚠️ **Every F row is 0-GPU by design and none has moved in 37 days.** Six are confirmed live by
source probe — these are the cleanest pull-able items in the file:

| row | verdict | evidence (positive assertion) |
|---|---|---|
| F1 | **OPEN — confirmed at source** | `train_v6_staged.py:730` still reads `"reported": ("S1_ade_8_30s", …)`. Not promoted to `required`. |
| F2 | **OPEN — confirmed by index** | `stack/ops/runs.d/v6-SW-30k.env` is **not in `git ls-files`** (second probe: no `v6-SW-30k.env` by basename anywhere). ⚠️ Flagged DONE-BUT-OPEN by the early substring matcher on *"(**done**-marker = off-switch)"* — **false alarm**, fixed (§7.2). |
| F3 | **OPEN** | `stack/scripts/w7_selection_rules.py` **is** tracked; `PREREG_SELECTION_RULE.md` is **not** (two probes). ⇒ half-built, exactly as the row says. |
| F4 | **OPEN — confirmed both ways** | `V6_TRAINER_DESIGN.md:383` still reads `v4.2`; `flagship-v4-fromscratch` has **0 hits** in that file. A ~10-minute fix untouched for 37 days. |
| F5 | ⛔⛔ **SUPERSEDED BY SOMETHING WORSE — the suite does not RUN** | See §5.8. All three stated causes are overtaken: the suite is `Interrupted: 2 errors during collection`. |
| F6 | **OPEN** | `PREREG_E_ENC.md` not in the index (two probes). |
| F7 | **OPEN** | `stack/scripts/v6_preflight.sh` not in the index (two probes). |
| F8 | **UNVERIFIABLE** | *"Verify F on the pod"* — host-dependent by construction. |
| F9 | **OPEN** | No `§1.15` S-W row found in `MODEL_REGISTRY.md` (two probes). ⚠️ But see E6/E8: **if v6/S-W is superseded, F9 is obsolete rather than open** — that is a Master Mind call. |

### 5.7 The R-series (refav1 / refcv3 / v7)

| row | verdict | evidence |
|---|---|---|
| R3 | **STALE-BLOCKER** | *"depends on: C-REFAV1-PLAN-NOGOAL fix landed"* — the dependency **has landed**, so the row is unblocked and was never revisited. ⚠️ Flagged DONE-BUT-OPEN by the early matcher on that very word *"landed"* — **the sharpest false positive of the five**, because the word belongs to the row's **dependency**, i.e. its reason to be open. Fixed (§7.2). |
| R5 | **OPEN — correctly** | Says `DIAGNOSED … fix (save-before-eval) in progress`. ⭐ **The tool deliberately does not treat `DIAGNOSED` as a closure** — Rule Zero in one row: a refutation/diagnosis is a waypoint, not a deliverable. Pinned by a test. |
| R6 | **OPEN — confirmed at source** | `--bptt-truncate` has **0 hits** in `stack/scripts/train_stage_a.py`; repo-wide hits are three *Research* scripts only, never the trainer. |
| R7 | **OPEN** | Needs a `TanitAD_ValidateAIDesign` pre-registration; none found for the tactical CE term. |
| R9 | **OPEN** | Strategic ARG slot encoder; no implementation found. |
| R11 | **PARTLY STALE** | Half is **refuted**: `actdiv_anchored` now has 3 callers (`analyse_units.py`, `cost_surface_probe.py`, `test_actdiv_anchored.py`). Half **stands**: `mm_e19_read.py:441` still shells out to `actdiv_local.py`. ⇒ restate as *"route `mm_e19_read` through the anchored tool"*. |
| R14 | **OPEN (process)** | The anti-stranding CHECK. ⭐ Still live and still earning: see the memory note *"agent worktrees go stale under you"*. |
| R16 | **OPEN — confirmed at source** | The retired **8.56** floor still appears 3× in `.claude/skills/TanitAD_ValidateAIDesign/SKILL.md` (`:62`, `:65`, `:78`). ⚠️ **Read the context before acting**: `:62` is an explicit ⛔ RETIREMENT notice, so this is a *documented* retirement, not a live stale threshold. The row's ask — sweep the instruction docs — is **discharged for this file**; re-scope it to the PREREG templates. |
| R17 | **OPEN, and it is the cheapest refutation in the file** | `PREREG_V7F.md:291` defines rung **R0** with its controls; `:545` requires `raw/ldad_step0.json`; **that file does not exist anywhere in the tree** (two probes). ⭐ **0 GPU, can refute the whole LDAD line before an arm is spent.** |
| R18 | **OPEN — and its guard is LIVE RED** | `stack/scripts/dinov3_fp8_encode_ship.py` contains **no** `guard_corpus_build` and is **not** classified in `NOT_AN_INGEST_DOOR` (0 hits). **MEASURED:** `pytest stack/tests/test_build_parity_guard.py` → **1 failed, 23 passed**, `test_every_derived_corpus_writer_is_gated_or_classified` asserting on `['scripts/build_b1_agent_join.py', 'scripts/dinov3_fp8_encode_ship.py']`. ⛔⛔ **The row names ONE file; the failing assert names TWO.** `build_b1_agent_join.py` is an ungated corpus writer that no backlog row mentions. |
| R19 | **OPEN — confirmed at source** | `_artifact_path_names` at `test_build_parity_guard.py:127`, used at `:188`; **no depth/hop bound** present. |
| R21 | **OPEN — MIS-FILED as PI** | A cut-over decision, absent from the PI queue (§3). ⚠️ False-alarmed by the early matcher on the decision id `D-STEER-INTERFACE-**RESOLVED**` — fixed (§7.2). |
| R24 | **OPEN — premise INDEPENDENTLY CONFIRMED, escalation MISSING** | `Decisions/2026-09-05-mm-decisions.md:1351`: *"the seed on the box is **ViT-L/16**, which `build_trunk_anchor` refuses BY DESIGN against the ViT-B/16 trunk."* And **0 hits in the PI queue** (§3). |
| R25 | **OPEN** | The `-k` filter string `v6\|staged\|parity\|v7_labels\|intrain\|v7_wiring\|eval_exclusion` still appears verbatim in `GOALS_AND_CLAIMS.md:5194`; `stack/tests/test_dinov3_seed.py` is tracked and matches none of those tokens. The **brief template** is the fix, as the row says. |
| R28 | **OPEN — the highest-value refav1 item** | A `2026-09-03-tactical-decoder` package exists and `PREREG_TACTICAL_DECODER.md` is tracked; **no arm result**. The row's own framing (*"a planner cannot search for a turn it is never asked to make"*) is unrefuted. |
| R29 | **OPEN (REFRAMED)** | The row says *"REFRAMED (not struck)"* — ⭐ the tool correctly does **not** read this as a closure, and it is pinned by a test. |
| R30 | **OPEN — MIS-FILED as PI** | Doctrine ruling, 0 hits in the PI queue. ⚠️ False-alarmed by the early matcher on the identifier `roll_**closed**` — fixed (§7.2). |
| R31 | **OPEN** | No per-window dump with episode ids found in the in-training eval path (two probes on `refc_v3_train.py`; the two hits at `:445`/`:4283` are comments). |
| R32 | **OPEN** | `C-REFCV3-EVAL-PRIOR-LEAK` is live in `GOALS_AND_CLAIMS.md:5602`; the 34-vs-40 reconciliation is not recorded. |
| R34 / R35 / R36 | **OPEN — Master Mind calls mis-labelled as PI** | Design + cut-over decisions, 0 hits in the PI queue (§3). |
| R38 | **OPEN (PARTLY STRUCK)** | The row's own label. ⭐ The tool does **not** read `PARTLY STRUCK` as a closure — pinned by a test, and by mutation `M2`. |
| R40 | **OPEN** | Label-pipeline item, owned by the Data FlyWheel. |
| R42 | **OPEN (scope note)** | A standing caveat rather than a work item; consider moving it to `GOALS_AND_CLAIMS.md` where caveats are read. |
| R43–R47 | **OPEN** | Lab-scheduling and doctrine items; none has a closing artifact. **R45 confirmed at source**: the four controls (`COUNTERFACTUAL`, `VALIDATED POSITIVE`, `PASSTHROUGH`, normalisation-state) have **0 hits** in `tools/criteria_check.py` — the criteria registry does not carry them. |
| R48 | **OPEN — confirmed at source** | `trunk_lr_scale` exists in `train_v6_staged.py` (`:8496`, `:8524`, `:8541`, `:8544`) and is **whole-trunk**; `lastk` has **0 hits** anywhere in `stack/`. Exactly as the row states. |
| R49 | **OPEN — confirmed exactly, AST-bounded** | `dry_run` spans `train_v6_staged.py:5804–6069` (266 lines, extent taken from the AST, not a guess). Over that whole body: `torch.manual_seed` **0**, `np.random.seed` **0**, `random.seed` **0**, `seed_everything` **0**; the single `manual_seed` at `:5917` is `torch.Generator().manual_seed(a.seed)` — a **LOCAL** generator. ⇒ the global RNG is never seeded, precisely as the row states. |
| R50 | **OPEN — confirmed at source** | `T1_CHECKLIST.md:166` GATE 6 exists and covers the **trajectory** trivial-profile only; no selection-profile bullet in the following 34 lines. The row's requested second bullet is absent. |
| R51 | **UNVERIFIABLE** | Explicitly needs a **human read** of `taniteval/tools/REFCV3_ARM.md` §2 (the file is tracked). Not a machine question. |
| R52 | **OPEN — confirmed at source** | `_odd_raw_frames` present at `taniteval/tools/refcv3_arm.py:2601`; the even-frame limitation is still in place. |
| R53 | **OPEN — confirmed at source** | `TRIVIAL_IDENTICAL_M = 1e-9` at `refav1_arm.py:2101` (and `atol=1e-9` at `:1880`); not batch-aware. |
| R54 | **OPEN — MIS-FILED as PI** | Explicitly labelled *"PI decision D3b"*; **0 hits in the PI queue** (§3). The most clearly mis-filed of the eleven. |
| R55 / R56 | **OPEN** | Standing constraint + an unreconciled 1.43× discrepancy. **R56 is cheap and should be pulled**: two MEASURED numbers, neither withdrawn — §5.9. |
| R57 | **PARTLY STALE — the file is now TRACKED** | `…/2026-09-03-cost-repair/PROPOSED_REGISTER_ROWS.md` is **12,792 B and IS in `git ls-files`** ⇒ the *stranding* half is discharged. The **ownership** half (*"identify its owner"*) is unanswered. ⚠️ The row records **12,695 B**; the file is now **12,792 B** — it changed after the row was written, which is itself evidence somebody took it on. Restate, don't strike. |
| R58 | **OPEN** | `v`-at-the-model-boundary leak; no `v`-zeroed arm found. |
| R59 | **OPEN — confirmed at source** | `transition_probe.py:387` `run_panel`, and `:399–400` is literally `if name not in feats: continue` — the silent skip, exactly as described. A two-line fix. |
| R60 | **PARTLY REFUTED as written** | *"`transition_probe.py` is wired into nothing"* is **false at this tip**: `gs9_vleak.py` and `stack/tests/test_transition_probe.py` both call it. What **stands** is that `mm_e19_read.py:441` still calls `actdiv_local.py`. ⇒ same correction as R11; **the two rows should be merged**. |
| R61 | **OPEN** | Needs a CI on the RFF floor; point estimates only. |
| R62 | **OPEN — confirmed, three probes** | `k8clip05p30k` has **0 hits** in `MODEL_REGISTRY.md`. Second probe: the name lives in `PREREG_V7F.md`, `V7_LAUNCH_GATE.md`, `GOALS_AND_CLAIMS.md` and `…/2026-09-01-mm-e19-k8-attribution/RESULT.md` — so **the arm is real and the registry row is genuinely missing**, not a typo. Cheap 0-GPU registry work. |

### 5.8 ⛔⛔ F5 IS NOT "23 FAILURES" — `stack/tests` DOES NOT COLLECT, SO **ZERO TESTS RUN**

**MEASURED 2026-09-18, banked at `raw/fullsuite_tail.txt` (both runs, with provenance).**

```
PYTHONPATH=D:/Projects/TanitAD/stack  python -m pytest stack/tests -q
  ERROR stack/tests/test_metric_decode_refusal.py
  ERROR stack/tests/test_refa_v1_dk_hook.py
  !!! Interrupted: 2 errors during collection !!!
  2 errors in 160.06s          PYTEST_RC=2
```

* `test_metric_decode_refusal.py:34` → `cannot import name **UntrainedMetricReadout** from
  tanitad.models.v6`
* `test_refa_v1_dk_hook.py:26` → `cannot import name **DistanceKeepingSpec** from
  tanitad.refs.refa_v1`

⛔ **`Interrupted` means pytest aborted before running a single test.** So F5's three stated causes
(`onnx`, a Windows basename assert, a 20-test order polluter) are all **moot**: two of them are
independently cleared — `onnx` **is** installed at **1.22.0**, and `test_resim.py` +
`test_rig_clean_fix.py` read **87 passed** in isolation — but the third cannot even be asked,
because the suite never reaches it.

⇒ **The `CLAUDE.md` invariant *"`pytest -q` must stay green before any commit"* is currently
UNVERIFIABLE by the stated command**, and has been for at least as long as those two symbols have
been missing. This is exactly the condition `tools/ci_gate.py` exists to fail on (*"fails on failure,
**collection error**, …"*), which suggests the gate is not being run, or is being run on a subset.

⛔⛔ **AND THE RUN ITSELF REPRODUCED TWO NAMED TRAPS — both worth more than the F5 verdict:**

1. **THE FIRST RUN IMPORTED FROM `G:`.** With no `PYTHONPATH`, the venv's **editable `tanitad`
   install resolves to the G: mount** — the traceback named
   `G:\…\stack\tanitad\rl\posttrain.py` — so the run was **not about this working tree at all**, and
   it failed on a *different* symbol (`score_gt_bar`). This is the documented dev-box costume of the
   `PYTHONPATH` trap. ⇒ **every `stack/` test result in this sweep was re-run with
   `PYTHONPATH=D:/Projects/TanitAD/stack`**, and R18's verdict is unchanged under the pin
   (`1 failed, 23 passed`, same assert) — stated so the reader can see the pin was actually checked,
   not merely applied. ⚠️ My own instrument is untouched by this: `tools/backlog_drift.py` and its
   suite import **neither `tanitad` nor `torch`** (0 hits, asserted).
2. **THE BACKGROUND JOB REPORTED "exit code 0" FOR A RUN THAT EXITED 2.** My wrapper ended in a
   `tail`, so the shell's status was `tail`'s. ⭐ **The admissible evidence was the ARTIFACT** — the
   `PYTEST_RC=2` line written into the file — exactly as `CLAUDE.md` says: *assert on the artifact,
   not the status*. Had I read the notification, F5 would have gone into this report as "resolved".

### 5.9 What this sweep did NOT do, said plainly

* It did **not** probe any host (Thor, the A40, pod1–pod5). Every host-dependent row is marked
  `UNVERIFIABLE` with the probe named, never guessed.
* It did **not** edit `BACKLOG.md`. ⛔ *A sweeper that rewrites its own input is not auditable* —
  the edits are specified in §6 for the Master Mind to land.
* R49's probe was **bounded to the head of the function**; treat it as INDICATIVE, not settled.
* Rows resting on unnamed task numbers (A9, B6–B8, D1–D5) are **not actionable as written**, and I
  have not invented a register for them.

---

## 6. The recommended edit set — for the Master Mind to land

**Group 0 — the two that matter most, and neither is a row:**
1. ⛔ **Strike the `BLOCKING INTEGRATION ITEM` (L259)** — closed by `bb4479c`, 2026-09-06 (§2.4). It
   is the file's loudest live blocker and it has been false for 12 days.
2. ⛔ **Raise `stack/tests` not collecting as its own work item** (§5.8) — two missing symbols,
   zero tests running, and a `CLAUDE.md` invariant that cannot currently be checked. This did not
   exist as a backlog row and should.

**Group 1 — strike, evidence already in the row (7 rows, ~5 minutes, zero risk):**
`~~A11~~ ~~A13~~ ~~B4~~ ~~C3~~ ~~R1~~ ~~R2~~ ~~R4~~`. After this, `tools/backlog_drift.py` exits 0
on the DONE-BUT-OPEN class. ⚠️ `tools/tests/test_backlog_drift.py::test_the_live_backlog_is_CURRENTLY_DRIFTED_and_the_gate_says_so`
pins today's state deliberately and **will go red** — that test names itself as the thing to update
**in the same commit as the sweep**. That is the discipline, not a bug.

**Group 2 — restate the stale blockers (5 rows):** C2 (re-probe is 33 days old) · E1 (T1 landed) ·
E3 (T1 landed, host unknown) · E6/E8 (*"v6 GO"* superseded by the refcv6 line) · R3 (dependency
landed).

**Group 3 — escalate, don't park (6 rows):** move C4, C5, C6, R24, R30, R54 into
`PI_DECISION_QUEUE.md`; relabel R21, R34, R35, R36, R49 as Master Mind cut-over calls.

**Group 4 — merge / correct (3 rows):** R11 + R60 merge and lose the refuted half; R57 restate as an
ownership question; R18 **must gain `build_b1_agent_join.py`**, which its own guard names and it
does not.

**Group 5 — re-base the fleet premise:** section headers B/E/F. Run `tools/fleet_probe.py` first.

**Group 6 — the pull-list a gated turn should actually use tonight (0 GPU, all confirmed live):**
**R17** (can refute the LDAD line before any arm is spent) · **F4** (~10 min) · **R59** (two lines) ·
**R62** + **E10** (registry rows) · **R19** · **R56** · **A13c**.

---

## 7. The instrument — `tools/backlog_drift.py`

### 7.1 What it is, and why it is not a second spelling of something we have

| tool | question | key |
|---|---|---|
| `stack/scripts/lab_backlog_drift.py` | *have findings outrun the Lab backlog?* | git history over two finding-files |
| `stack/scripts/backlog_claim_check.py` | *does a row call a file absent that is tracked?* | the git index |
| **`tools/backlog_drift.py`** (new) | ***does a row declare itself done while sitting open?*** | a contradiction between two fields the row itself wrote |

It reuses the Lab tool's conventions deliberately: the `--git-dir/--work-tree` git form (the one that
survives a flaky mount), non-zero exit so it can gate, ASCII-only stdout, and the "git/index is the
source of truth, prose is not" stance. It lives in `tools/` because that is the charter for
agent-facing repo hygiene — stdlib-only, ASCII stdout, tests under `tools/tests/` — and because the
rule it enforces is the `tools/` README's own: **"Absence of evidence is an ALARM, not an all-clear."**

```bash
python tools/backlog_drift.py              # 0 clean · 1 drift · 2 REFUSING
python tools/backlog_drift.py --list-open  # the genuinely-open rows
python tools/backlog_drift.py --json       # machine-readable
```

**The guards, all exiting 2:** missing file · unreadable file · short read (< 2,000 B) · **zero rows
parsed** · zero sections · below `--min-rows` (default 40) · **zero rows read as closed** · **every
row read as closed**. The last two are two-sided on purpose: break the strike regex one way and
every row reads open (loud); break it the other and every row reads closed (**silent**) — only the
silent direction needed a guard, so it has one. It also prints, always: unparsed bullet candidates,
and heading-items it structurally cannot check.

### 7.2 ⛔ THE ANCHORING FIX — five false alarms caught before this shipped

The first matcher looked for done-words as **substrings** and fired on **13 rows, 5 of them wrong**:

| row | the text that fooled it | what it really is |
|---|---|---|
| A14 | *"the AlpaSim **closed**-loop stream"* | a compound word |
| F2 | *"(**done**-marker = off-switch)"* | a compound word |
| R30 | *"so `roll_**closed**` cannot be ported"* | an identifier |
| R21 | *"(D-STEER-INTERFACE-**RESOLVED**)"* | a decision id |
| R3 | *"C-REFAV1-PLAN-NOGOAL fix **landed**"* | **a DEPENDENCY that landed — the row's reason to be OPEN** |

⭐ **Five false alarms in thirteen hits is the gate nobody reads.** The fix is to require the
closure word to be the **FIRST word of an emphasised assertion** (`**DONE 2026-08-18**`), which is
how every genuine closure in the file is actually written. All five strings are now pinned as
**verbatim negative-control fixtures**, because a suite whose negative controls are hypothetical
cannot catch the error that actually happened.

### 7.3 ⭐ A guard that guards nothing looks exactly like a guard from the inside

The tool shipped for two hours with a `NEGATIONS` list — `("partly struck", "not struck",
"reframed", …)` — written for the live rows R38 and R29 and defended as *"defence in depth"*. The
mutation proof deleted it and **the suite stayed GREEN** (`M3`), and removing it moved **0 of 117
live verdicts**. First-word anchoring already rejected both. ⇒ **It was deleted, not kept.** The
rule it earns is in the source where the list used to be: *a check no mutation can turn RED is not
a check*. Its absence is now documented so nobody re-adds it as a precaution.

### 7.4 Two-sided mutation proof — 12/12 RED (`raw/MUTATION_PROOF.json`)

```
baseline: GREEN (rc=0)  32 passed
M1_no_zero_row_guard          RED   M2_substring_done_words       RED
M3_bullet_bold_prefix_dropped RED   M9_no_min_bytes_guard         RED
M10_no_min_rows_floor         RED   M11_no_zero_closed_guard      RED
M12_heading_items_unanchored  RED   M4_gate_never_fails           RED
M5_no_id_aggregation          RED   M6_blocker_includes_item_cell RED
M7_no_all_closed_guard        RED   M8_non_ascii_to_stdout        RED
after restore: GREEN (rc=0)  md5 match=True
ALL MUTATIONS CAUGHT
```

Every mutation is a defect the tool **actually had**, or was one edit from. `run_mutation_proof.py`
refuses a red baseline, refuses an anchor that does not occur exactly once, reads status from
`returncode` (**never through a pipe**), and asserts the restore on the file's **md5**, not on the
write's exit code.

⚠️ **Three mutations survived on the first pass and each exposed a worthless test** — the honest
part of this section:

1. **`M3` survived** → the `NEGATIONS` list was dead code (§7.3). *A finding about the code, not a
   reason to weaken the proof.*
2. **`M9` survived** → `test_a_SHORT_read_REFUSES` asserted only `rc == 2`. With the byte floor
   removed the file still refused — **on the row floor, for a different reason** — and the test
   called that a pass. Now it asserts the byte-floor message and passes `--min-rows 1` to remove the
   rescuer. *Same family as reading an exit code instead of the artifact.*
3. **`M12` survived** → the mutation swapped `.match` for `.search` on a pattern that still carries
   `^`, so it was **inert**. An inert mutation reports a suite as weak when it is fine — the mirror
   of a test that passes on broken code, and just as misleading. Now it mutates the pattern's call
   site instead.

A fourth (`M3`'s replacement) initially passed because the test exercised the **function** while the
mutation hit the **call site** in `parse()` — the right behaviour asserted at the wrong boundary. A
second test now drives the payload through the parser.

---

## 8. Root-cause classes this sweep adds

1. **A gate recorded in the wrong file is not a gate.** 11 of 11 "blocked on the PI" rows are absent
   from the PI queue. Same family as *"escalate integration, don't write 'please merge' into a doc"*,
   scaled up and wearing the word "blocked".
2. **A process snapshot in the present tense rots into a lie.** *"Running (do not duplicate): pod4 …
   pod5 …"* was true on 2026-08-11 and reads as current 38 days later. **Date the tense, or write
   the probe that refreshes it.**
3. **A check no mutation can turn RED is not a check** (§7.3) — and *"defence in depth"* is the phrase
   that protects one from scrutiny.
4. **A blocker with no re-probe date ages into a permanent gate.** C2's *"RE-PROBE 2026-08-16"* was
   never done; the row has now been blocked on an unverified fact for 51 days.
5. ⭐ **A stale row can be stale in the direction that UNDER-reports our own progress, and nobody
   checks that direction.** The `BLOCKING INTEGRATION ITEM` claimed to block every refcv5 arm for 12
   days after it was fixed. We sweep for *"is this already done?"* on rows we want to close; we do
   not sweep for it on rows that are shouting. **The loudest row deserves the probe most**, because
   nobody argues with it.
6. ⛔ **An INHERITED number copied into a MEASURED sentence survives every review that reads the
   sentence.** §2.4's first draft did exactly that, inside a sweep written to catch stale numbers.
   The discriminator was cheap and is the same one this programme keeps re-learning: **run the
   probe, and carry a control that must read a known value** (`lan_to_cond` = 7).

---

## 9. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `tools/backlog_drift.py` | **repo**, staged | the instrument, 12/12 mutation-proven |
| `tools/tests/test_backlog_drift.py` | **repo**, staged | 32 tests, green |
| `…/2026-09-18-backlog-drift/RESULT.md` | **repo**, staged | this document |
| `…/2026-09-18-backlog-drift/raw/run_mutation_proof.py` | **repo**, staged | re-runnable proof |
| `…/2026-09-18-backlog-drift/raw/MUTATION_PROOF.json` | **repo**, staged | its output, ALL MUTATIONS CAUGHT |
| `…/2026-09-18-backlog-drift/raw/backlog_drift_scan.json` | **repo**, staged | the full 117-row machine scan |
| `…/2026-09-18-backlog-drift/raw/backlog_drift_console.txt` | **repo**, staged | the console form incl. `--list-open` |
| `…/2026-09-18-backlog-drift/raw/fullsuite_tail.txt` | **repo**, staged | both `stack/tests` runs (unpinned → G:, and the admissible pinned one), each with its `PYTEST_RC` |

**Nothing is stranded**: every artifact is in the repo working tree and staged. **Nothing was
pushed, committed, or landed** — staging only, per the operating standard. **`BACKLOG.md` was not
edited**; §6 is the edit set for the Master Mind.

---

## ⭐ MASTER MIND REVIEW, 2026-09-18 — the suite finding is CONFIRMED and DATED

Re-run independently before landing.

### ✅ `pytest stack/tests` does not complete — CONFIRMED

```
8365 tests collected, 2 errors in 58.63s
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
PYTEST_RC=2
```

* `stack/tests/test_refa_v1_dk_hook.py` — `ImportError: cannot import name 'DistanceKeepingSpec' from 'tanitad.refs.refa_v1'`
* `stack/tests/test_metric_decode_refusal.py`

⇒ `CLAUDE.md`'s *"the full suite lives at `stack/` — `pytest -q` must stay green before any
commit"* **cannot be satisfied by that command as written.** Two import errors abort a run
of **8,365 otherwise-collectable tests**.

### ⭐ And it is PRE-EXISTING — dated, not inherited

| file | last touched |
|---|---|
| `stack/tanitad/refs/refa_v1.py` | **2026-09-06** (`d290cb9`) |
| `stack/tests/test_refa_v1_dk_hook.py` | **2026-09-10** (`bb030da`) |
| `stack/tests/test_metric_decode_refusal.py` | **2026-09-10** (`bb030da`) |

The 2026-09-17/18 commits touched only `refc_v3_train.py`, `pdm_proxy.py` and four **new**
test files — **none of the three above.** ⇒ the breakage is **8–12 days old** and was not
introduced by this night's work. With `--continue-on-collection-errors` the suite runs.

### ⚠️ One claim in the agent's report is NOT reproduced here

It reported that the first run imported `tanitad` from **G:** (the editable install). In the
off-Drive worktree that does **not** happen:

```
tanitad from: C:\Users\Admin\tanitad-wt-bevtac\stack\tanitad\__init__.py
```

⇒ the G: import is a property of **where the suite was launched from**, not of the suite.
It is the documented editable-install trap and the remedy is the documented one — set
`PYTHONPATH` to the clone and **assert `tanitad.__file__`** — which is why that assertion is
worth keeping in any runbook that launches the suite.

### The consequence worth acting on

⛔ **A green-suite invariant that no runnable command can currently satisfy is a rule that
silently stops binding.** Either the two collection errors are fixed, or the invariant is
restated to name the flag it requires. Both are cheap; leaving it as-is is the expensive
option, because every commit "passes" a check nobody can run.

<!-- BACKLOG-DRIFT-MASTERMIND-REVIEW-2026-09-18 -->
