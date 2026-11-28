# The escalation sentence is now a gate — and writing it found the door nobody had named

**2026-08-18 · closing the C113 escalation · all figures MEASURED (ours), re-derived by
`code/derive_build_doors.py` and by `stack/tests/test_build_parity_guard.py` on every run**

---

## Headline, in the order that matters

1. ⛔ **C113 ended with `"whoever runs that build must call parity.filter_train_clips() first"`.
   That sentence was the defect** — doctrine that runs only if the next operator read the report,
   which is C108 (a drift tool that compared the wrong thing for weeks because the doc saying so
   was never re-read) pre-registered instead of discovered. **It is now `parity.py` §10c
   `guard_corpus_build()`, called by the builds themselves.**
2. ⭐ **The gate is safe to leave ON because of a fact I measured before designing it: the parity
   TRAIN digest set (2 400) and the deployed VAL digest set (40) intersect in ZERO clips.** So the
   canonical train build passes untouched and the gate obstructs *only where an overlap actually
   exists*. A guard that fired on the legitimate case would be switched off within a week; a guard
   that fires unconditionally is not a guard (C107), it is a wall.
3. ⛔ **`v2_compressed.py build` was NOT the only door, and a hand-list would have shipped with two
   holes.** Deriving the population found **`rebuild_pai_rolling.py`, which writes `ep_%05d.pt` via
   `mixing.save_episode` DIRECTLY** — it does not go through `epcache.build_episodes_cached`, so
   gating the obvious writer would have missed it — and **`slice_v2_cache.py`, which emits a whole
   NEW `.v2ep.pt` corpus from an existing one without touching source video, HF, or `--sel`.**
   Neither appears in any runbook description of "the build".
4. ✅ **Six doors gated, and `build_pai_cache.py` is now gated WITHOUT BEING TOUCHED** — because the
   gate went into the writer (`epcache.build_episodes_cached`), not the caller.
5. ⛔ **Two doors genuinely CANNOT be gated, and saying so is the finding.** `epcache_to_pilot.py`
   and `lake/view.py` key on **positional `ep_%05d` / `episode_id`**, not PhysicalAI clip ids
   (`parity.py:101`: *"a different uid space"*). A gate there would ask the oracle a question in the
   wrong vocabulary, **miss every lookup, and print a reassuring `0 contaminated` forever** — the
   `df`-reports-the-cluster trap wearing a guard costume. ⚠️ Stated precisely, because the two were
   handled differently: **`epcache_to_pilot.py` was edited** and now prints the gap at run time;
   **`lake/view.py` was NOT edited** — it carries its reason only in the test's classification map,
   because the lake path already refuses PhysicalAI by construction (`gated-confidential` →
   `PermissionError` in `assemble_lake_record`), so a second notice there would be noise.
6. ⚠️ **The gate changes behaviour for one existing launcher, deliberately.** A re-run of the parity
   **VAL** cache build now **REFUSES** until it declares `--corpus-role val`. That is correct — it
   is the one build that legitimately contains the deployed 40 — and it is the price of a default
   that protects the dangerous direction.

---

## 1. What was measured, and how (nothing here is quoted from C113)

The suite recomputes C113's headline from primary sources on every run: the banked
`alpamayo_clip_ids.txt` intersected with the **committed per-clip digest oracles**, never with a
list of offenders.

| quantity | value | how |
|---|---:|---|
| Alpamayo record set | **4 729** | banked list, read at run time |
| …inside `physicalai-train-e438721ae894` | **201** | `parity.clips_in_parity_train` |
| …inside the **deployed val 40** | **6 (15.0 % of val)** | `parity.clips_in_deployed_val` |
| parity-train digests ∩ deployed-val digests | **0** | `parity_train_clip_digests() & deployed_val_clip_digests()` |
| oracle sizes | **2 400** / **40** | `parity.require_ingest_gate()` |

⭐ **The third row is the load-bearing one and it is the reason the design works.** Because train and
val share no clip, the gate can default to the *supervision* check (§10b) for every undeclared
build without ever obstructing the canonical train corpus. It is pinned by
`test_train_role_does_not_refuse_the_parity_train_corpus`.

🔒 **No third plaintext copy of the ids was created — and that is CHECKED, not asserted.** All 13
files this stream wrote or modified were scanned against the full 4 729-id list: **zero hits**.
Every id the guard or the suite needs is either supplied by the caller or derived from the two
committed digest files. Membership stays exact; enumeration stays impossible. C113's escalation #2
(the two *existing* plaintext lists) is **untouched and still a PI question** — this work neither
adds to it nor resolves it, and it now *depends* on those files as test evidence.

---

## 2. The doors — DERIVED, not hand-listed (C99/C105)

`stack/tests/test_build_parity_guard.py::derive_corpus_writers` walks the AST of every module under
`stack/` and asks a mechanical question: **does an artifact-shaped path flow into a publishing
call?** Artifact shapes are the things the pipeline eats — `*.v2ep.pt`, `ep_%05d.pt`, `videos/*.mp4`,
`ego/*.npz`, `clips.json`. Publishing calls include `replace`/`rename`, because the atomic-publish
idiom writes `.tmp` and renames, so **the rename IS the write**.

Two rules, two jobs:

| rule | what it asks | population | consequence |
|---|---|---:|---|
| **WRITE** | an artifact-shaped path flows into a publishing call (2-hop local dataflow) | **11** | must be **gated** or **classified with a reason** |
| **MENTION** | the module merely NAMES a corpus artifact | **34** | census; asserted only as a superset |

MEASURED by `code/derive_build_doors.py` → `raw/build_doors.json`:
**34 mention · 11 write · 6 gated · 0 unclassified.**

### The verdicts

| module | verdict | note |
|---|---|---|
| `stack/scripts/v2_compressed.py` | **GATED** | the 4 472 build's own entry point |
| `stack/scripts/aug120_pipeline.py` | **GATED** | before `create_repo`, before the first shard pull |
| `stack/scripts/v2_to_pilot.py` | **GATED** | the bridge — reachable *without* a cache build |
| `stack/scripts/slice_v2_cache.py` | **GATED** | ⭐ found by derivation; emits a new corpus |
| `stack/scripts/rebuild_pai_rolling.py` | **GATED** | ⭐ found by derivation; bypasses `epcache` |
| `stack/tanitad/data/epcache.py` | **GATED** | the writer ⇒ covers `build_pai_cache.py` untouched |
| `stack/scripts/epcache_to_pilot.py` | **CANNOT_GATE** | positional uid space; now says so at run time |
| `stack/tanitad/lake/view.py` | **CANNOT_GATE** | keys on `episode_id`; lake refuses PhysicalAI anyway |
| `eval_flagship_v4` · `eval_v58f` · `run_idm_proof` · `emit_situation_labels` | **CONSUMER** | read a corpus, write a report; their exposure is §10's, not §10c's |

⚠️ **THE DERIVER'S BLIND SPOT, PINNED RATHER THAN HIDDEN.** `v2_compressed.build` hands the
`<clip>.v2ep.pt` path to a helper (`build_compressed`) that writes it through a *parameter*. Static
derivation cannot cross that boundary, so this door is caught by the MENTION census and **not** by
the WRITE rule. It is therefore pinned by a **behavioural** test that calls `build()` with a real
contaminated selection parquet and requires a refusal *before the output directory is created* —
not by the deriver. **A limitation that is written down and tested is a limitation; one that is not
is a hole.**

⚠️ **And the deriver's own filter is pinned** (C110 — *"an undercount produced by the instrument's
own filter"*). `test_derivation_still_finds_the_known_doors` fails if any regex edit shrinks the
population below the doors we know exist. A shorter list is not a cleaner codebase.

*(Second, independent probe: a dedicated search agent swept six locations and six naming conventions
and reported the same door set plus the two bypasses above — which is how `slice_v2_cache` and
`rebuild_pai_rolling` were caught before the first derivation rule was tight enough to see them.
Absence found at one location is not absence.)*

---

## 3. The gate — `parity.py` §10c

`guard_corpus_build(clip_ids, *, label, role="", mode="refuse", sanctioned_audit=None)` returns
`(kept, record)`.

| `role` | disqualifying overlap | why |
|---|---|---|
| `""` (undeclared, **default**), `train`, `augmentation` | the **deployed VAL** (§10b) | a corpus being BUILT is presumed to become supervision |
| `val`, `eval` | the **parity TRAIN** split (§10) | the held-out side |
| `audit` | none — **requires a REASON**, stamps `decision_grade: False` | a label census over train clips is legitimate; a silent one is not |

**The undeclared default checks the direction that is dangerous, not the one that is common.** C113:
*"the leak I was sent to close is the LESS dangerous of the two directions."* An operator who
declares nothing gets the check that protects the 40 episodes behind every published open-loop
number.

Three ways past, all explicit, all recorded: `mode="exclude"` (`--exclude-parity-overlap`) filters
and reports; `role="val"` flips the check; `sanctioned_audit="<why>"` waives it and voids
decision-grade. **The refusal message names all three, and a test asserts all three flags actually
exist on `v2_compressed`'s CLI** — a message that sends an operator to a flag that is not there is
worse than no message.

⭐ **Unconditional disclosure.** Both overlap counts are printed and put in the record on **every**
call, in every role, pass or fail — the `s2_labels.parity_contamination` precedent. The 201-clip
aug120 overlap survived for days inside a corpus whose name and provenance both said "independent";
the number nobody asked for is the number that would have said otherwise. `v2_compressed` writes the
record into `_geometry.json`, and `aug120_pipeline` into `parity_ingest_gate.json`.

🔒 Every refusal prints **counts only** — pinned by `test_the_refusal_never_prints_a_clip_id`.

⚠️ **A guard that no-ops without its oracle is C112 wearing a green suite.** `require_ingest_gate()`
runs at each door's start-up and refuses in seconds if the digest files are missing — the `t1_eval`
lesson (an import that fails *after* the rollout destroys the run while the compute is already paid
for), applied to the cheap end.

⭐ **AND THE GATE RUNS BEFORE THE SPEND, WHICH IS PART OF THE GUARD.** C112's own launch-path defect
died *after* paying for a 536 MB download. `test_the_gate_runs_before_the_expensive_step` asserts by
AST that `aug120_pipeline`'s gate line number precedes its `create_repo`, and the `v2_compressed`
test asserts the output directory does not exist after the refusal.

---

## 4. The pin — `stack/tests/test_build_parity_guard.py`

**24 tests.** Every refusal test feeds a **real** contaminated clip id derived at run time from the
oracle, and every one is paired with a **positive control** on a disjoint set — so a guard that
raised unconditionally would fail here rather than look strict.

⭐ **THE SUITE FOUND A DEFECT IN MY OWN GATE WHILE I WAS WRITING IT.** With the audit stamp placed
*after* the disjointness shortcut, `role="audit"` over a set that happened to be clean returned
`decision_grade: True` — **the waiver silently did not apply**, and an artifact built under an
explicit "I am waiving the check" would have been quotable as held-out. A waiver whose effect
depends on the data it waives is not a waiver. Fixed, and the fix is what the test now holds.

### C107 discharged by construction — each guard neutered, the suite required to go red

Each guard was disabled in turn and the suite re-run. **All eight go red** (`raw/neuter_matrix.txt`):

| neutered | result |
|---|---|
| `guard_corpus_build` → no-op | **12 failed**, 12 passed |
| `require_ingest_gate` → no-op | **1 failed**, 23 passed |
| the `v2_compressed.build` gate removed | **2 failed**, 22 passed |
| the `epcache.build_episodes_cached` gate removed | **3 failed**, 21 passed |
| the `aug120_pipeline` gate removed | **3 failed**, 21 passed |
| the `v2_to_pilot` gate removed | **2 failed**, 22 passed |
| the `slice_v2_cache` gate removed | **2 failed**, 22 passed |
| the `rebuild_pai_rolling` gate removed | **2 failed**, 22 passed |

⚠️ **AND THE HARNESS ITSELF TAUGHT SOMETHING.** The first matrix run was killed by a 2-minute tool
timeout **mid-case** and left `aug120_pipeline.py` **neutered on disk** — a `try/finally` restore is
no protection against `SIGKILL`. That is why every gated file was **byte-compared (md5) against a
pre-run backup** afterwards: **all 7 IDENTICAL**. *A restore step whose evidence is "the script has a
finally block" is the same class as "exit codes are not evidence".*

---

## 5. ⛔ ESCALATIONS — decisions, not documentation

1. **The parity VAL cache build now REFUSES without `--corpus-role val`.** `…/2026-07-28-wide-val-build/code/launch_val_build.sh:30`
   runs `v2_compressed.py build --only-clips parity_val_clips.txt` with no role, and the 600-clip val
   selection contains the deployed 40. **This is the intended behaviour** — but any re-run of that
   launcher needs the flag added. I did **not** edit the banked launcher: it is evidence of what was
   run, not a live script. **Whoever re-runs it adds `--corpus-role val`.**
2. **The 4 472 build's exact command is now determined, and it is one flag.** After the chunk-index
   step produces the `--sel` parquet:
   `python scripts/v2_compressed.py build --sel <chunk_index>.parquet --only-clips <4472>.txt --exclude-parity-overlap …`
   The gate drops the 6 deployed-val episodes, records the exclusion in `_geometry.json`, and prints
   the post-filter count so a report cannot quote 4 472 for a 4 466-clip corpus. **Without the flag
   it refuses; there is no path that silently proceeds.** *(`NEXT_4472_BUILD_INPUTS.md` §2 item 4 —
   "a parity decision, in writing, before the build" — is now answerable mechanically, but the
   REMAINING half of that item is still open: whether the output is a separate labelled corpus or an
   extension of the parity set is a **PI decision**, and no guard can make it.)*
3. **Two doors cannot be gated and their exposure is real but bounded.** A **hand-assembled** epcache
   of unknown provenance, bridged through `epcache_to_pilot.py`, carries no checkable parity status.
   The upstream writers are gated while the clip ids still exist, so the gap is only reachable by
   assembling an epcache outside both writers. **Closing it properly needs the epcache to carry its
   clip ids** — a schema change, not a guard, and a separate work item.
4. ⚠️ **A bounded gap I did NOT close, named rather than left to be discovered.** The label stages
   (`ph0_v2` → `ph0_sam3` → `ph1_fuse`) write per-clip `{cid}.json`, which is too generic a shape to
   put in `CORPUS_ARTIFACT` without matching every per-clip JSON in the repo — so they are **not in
   the derived population**. They are safe *by position*: their clip set arrives from the bridge,
   which is now gated, and the consumer that turns those labels into supervision
   (`s2_labels.load_s2_labels`) already carries the previous stream's `role=` refusal and its
   unconditional `parity_contamination()` disclosure. **The residual exposure is the same shape as
   the epcache one: a HAND-ASSEMBLED label set that never passed a bridge.** Closing it properly
   means the label records carrying their own provenance, not another guard.
5. **C113's escalation #2 is untouched and still needs the PI.** `…/2026-08-17-thor-concurrency-pilot/`
   commits 4 729 + 201 raw clip ids in plaintext, which contradicts §9's *"the repo carries only the
   digests"*. This work deliberately added no third copy and **depends on those files as test
   evidence**, so deleting them breaks the pin. Unchanged, and still a decision.

---

## 6. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `parity.py` §10c — `guard_corpus_build`, `require_ingest_gate`, the role table | `repo:stack/tanitad/data/parity.py` | no |
| gate at the v2/w120 build | `repo:stack/scripts/v2_compressed.py` (+3 CLI flags) | no |
| gate at the label orchestrator | `repo:stack/scripts/aug120_pipeline.py` | no |
| gate at the bridge | `repo:stack/scripts/v2_to_pilot.py` (+3 CLI flags) | no |
| gate at the cache→cache re-emit | `repo:stack/scripts/slice_v2_cache.py` (+3 CLI flags) | no |
| gate at the rolling epcache rebuild | `repo:stack/scripts/rebuild_pai_rolling.py` (+2 CLI flags) | no |
| gate at the epcache **writer** (covers `build_pai_cache.py` untouched) | `repo:stack/tanitad/data/epcache.py` | no |
| the stated uid-space gap | `repo:stack/scripts/epcache_to_pilot.py` | no |
| **the pin**, 24 tests + the derivation | `repo:stack/tests/test_build_parity_guard.py` | no |
| the standalone re-derivation runner | `repo:…/incoming/2026-08-18-build-parity-guard/code/derive_build_doors.py` | no |
| derived door census | `repo:…/incoming/2026-08-18-build-parity-guard/raw/build_doors.json` | no |
| the C107 neuter matrix | `repo:…/incoming/2026-08-18-build-parity-guard/raw/neuter_matrix.txt` | no |
| this report | `repo:…/incoming/2026-08-18-build-parity-guard/BUILD_PARITY_GUARD.md` | no |

---

## 7. Suites — and the caveat that must travel with them

⚠️ **THIS FULL-SUITE NUMBER IS A TORN SNAPSHOT AND SAYS SO (C114).** MEASURED while the run was in
flight: a sibling agent's `pytest tests/test_v6_t3_curriculum.py tests/test_v6_s1_multitick.py` was
executing concurrently, and `git status` showed `train_v6_staged.py` / `models/v6.py` and two v6 test
files **modified and staged by another stream**. A full-suite figure taken under those conditions
measures a tree that no single commit ever contained, plus CPU contention. **The attributable
evidence for this work is the targeted pair below, each run in isolation.**

| suite | result | run |
|---|---|---|
| **`stack/tests/test_build_parity_guard.py`** (this work) | **24 passed** | in isolation, twice |
| **`stack/tests/test_eval_contamination.py`** (the previous stream's 17, which §10c must not break) | **17 passed** | in isolation |
| `stack/tests` (full) | **4 084 passed · 0 failed · 7 skipped · 2 xfailed** (582 s) | torn snapshot, see caveat |
| `taniteval/tests` (full) | **1 136 passed** (141 s) | clean |

⭐ **The 3 pre-existing `test_v6_stage_init_introduction.py` failures that C113 escalated are GONE** —
the full run is at **zero failures**. They were another stream's work-in-progress, exactly as C113
said, and that stream has since landed. **Reported because C113's escalation #3 asked for their
owner and the answer is now "resolved", not "still open".**

⚠️ Run separately, with `PYTHONUTF8=1` and `OMP_NUM_THREADS=6` — one invocation over both trees
breaks the rootdir, and `torch`'s ~113 threads/process turn a concurrent run into a false hang.

