# Triage — the old EvalFlyWheel session's unfinished work, against HEAD `37645fc`

**Commissioned by** the Master Mind brief of 2026-09-19 (task 1: *"list only; no big work"*).
**Source** the old session's transcript
`C:\Users\Admin\.claude\projects\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD--claude-worktrees-zen-bose-eb35e9\005edae4-7a2f-474c-852f-070b00449cd0.jsonl`
(11,012,561 B; 738 text messages; its last substantive message is **2026-08-30 18:34Z**, and it is
**blocked on the G: mount**. The last event is an orphaned task notice from 2026-09-02).
Every verdict below is a **positive assertion against HEAD** (`git cat-file -e` / `git show
HEAD:<path>`), never an absence read through a failing channel. ⛔ Nothing was staged from the
stale mirror.

## What the old session was waiting on (its own words, 2026-08-30 18:34Z)

1. *"The three peer corrections (drift framing, blobs-found, roots→MEASURED)"*
2. *"Five drift tests, including the arm where a 7 must read as `EVAL SET MOVED` — and a 5, since drift is two-sided"*
3. *"The repo-wide Hub→Lab sweep across 34 files"* (45 lines, per its 17:32Z message)

## Verdicts

| # | item | what it targets (from the transcript's own tool calls) | state in HEAD `37645fc` (MEASURED) | verdict |
|---|---|---|---|---|
| **a1** | drift framing | `taniteval/taniteval/val40_overlap.py`: `EXPECTED_OVERLAP = 6` plus a *"THE EVAL SET MOVED"* message whenever the intersection is ≠ 6 | ⛔ **`val40_overlap.py` does not exist in HEAD** (`git show HEAD:…` → *path does not exist*). It is also absent from the mirror `C:\Users\Admin\tanitad-det` | **STILL OWED, but moot until its BASE lands.** The base module never landed at all |
| **a2** | blobs-found | `CRITERIA_REGISTRY.json`: re-class the 08-30 md5-pinned label blobs from *NOT FOUND* to *HF-resident* | no `blob` / `HF-resident` / `NOT FOUND` string anywhere in the HEAD registry | **OBSOLETE.** It concerned an 08-30 label pin that the v7.2 / **v8** releases have since superseded (refcv6 reads `s2_labels_v8_eval.jsonl.gz`) |
| **a3** | roots→MEASURED | `CRITERIA_REGISTRY.json` → `artifact_hygiene.eval_corpus_root_union` (roots 141 + 6 = 147, measured by the MM on Thor), `version` → 2.8.1 | ⛔ **the entry does not exist in HEAD**, and neither does its sibling `leak_guards.val40_shared_support`. ⚠️ **VERSION-NUMBER COLLISION:** HEAD's registry is v2.9.0, but its **2.8.0** changelog entry is a *different* change (`hyg.inference_seed`, 2026-09-06). The old session's 2.8.0 (2026-08-30: `val40_shared_support` + `eval_corpus_root_union`) **never landed**, and a later agent minted the same number | **STILL OWED for the refcv3/4b/5-v2-era B1 eval corpus** (147-clip union). **Not applicable to refcv6's clean-124 split**, which lives on different roots |
| **b** | five drift tests | `taniteval/tests/test_val40_overlap.py` (6 → no drift; 7 and 5 → `EVAL SET MOVED`; empty list refused; algorithm change refused) | ⛔ **the test file does not exist in HEAD** or in the mirror | **STILL OWED, together with a1** |
| **c** | Hub→Lab sweep (34 files / 45 lines) | repoint `TanitAD Research Hub/` → `TanitAD Research Lab/` in test files. Harness: one file at a time, keeping a change only if that file's failures **strictly decrease**; the `test_library.py` dual-name guard is skipped by PI instruction | **30 files / 36 lines** still cite it in HEAD (control: **64** test files cite `TanitAD Research Lab`, so the probe reads). Classified below | **PARTIALLY LANDED — 7 lines in 6 live files still owed** |

### c — the 36 remaining lines, classified (MEASURED, `git grep -n HEAD`)

| class | files | lines | verdict |
|---|---|---:|---|
| stale **path constants** in LIVE suite tests: `taniteval/tests/test_clhorizon.py:39`, `test_corridor.py:47`, `test_k1_degeneracy_guard.py:300`, `test_lead_source.py:21`, `test_ood_guard.py:37,40`, `test_ridge_intercept_penalty.py:29` | 6 | 7 | ⛔ **STILL OWED.** These are the lines that matter: a pin pointing at the dead tree **skips rather than fails** (memory note *skip-reasons-are-guard-health*: 22/22 taniteval skips were pins disabled by the rename). Route them through `tanitad/research_tree.py`, which knows both names, rather than hard-coding either one |
| comments that **quote** the directive (`stack/tests/test_build_parity_guard.py:60`, `stack/tests/test_eval_contamination.py:47`, `taniteval/tests/test_c2_published_policy.py:32`) | 3 | 3 | **OBSOLETE** — correct as written |
| `tools/tests/test_library.py` (118, 137, 142, 143, 156, 163) | 1 | 6 | **KEEP** — the deliberate dual-name guard (PI, 08-30: *"skip the `test_library.py` dual-name guard … silencing it would defeat the sweep"*) |
| tests inside **banked** packages `TanitAD Research Lab/**/incoming/**/tests/` (20 files, 2026-07-09 … 2026-08-07) | 20 | 20 | **OBSOLETE unless a suite collects them.** They are historical package records; rewriting a banked artifact falsifies where it ran (CLAUDE.md, D: move section) |

## Two more unlanded pieces the transcript shows, not in the brief

* The **val40 overlap guard itself** (`val40_overlap.py` + its test + the two registry entries) was
  *written and passing* on 08-30 (*"66 `criteria_check` tests still pass"*). It survives **only in
  the transcript's `Write` tool inputs**. The old scratchpad
  `…\005edae4-7a2f-474c-852f-070b00449cd0\scratchpad\` **does not exist on disk** (MEASURED: its
  parent holds six other session dirs, none of them this one), and the mirror does not have it
  either. ⇒ recoverable, but only by extracting it from the JSONL.
* `stack/tanitad/data/deployed_val40_clip_digests.json` **still cites the dead
  `../TanitAD Research Hub/…` path** in `source` / `cross_check_source` in HEAD. The old session had
  repointed it (*"40 digests intact, consumers green (27 passed)"*); that edit never landed.

## Is the val40 guard still worth landing? — measured, and INCONCLUSIVE on one point

| clip set | clips | overlap with the 40 deployed-val40 digests |
|---|---:|---:|
| refcv6 clean-124 (halfA + halfB, `D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-half{A,B}`) | 124 | **0** |
| `v2ep-eval139-256x1024cyl` | 139 | **0** |
| `v2ep-eval6-256x640cyl-REF` (the obvious candidate for a positive control) | 6 | **0** |

⚠️ **This zero is not yet admissible as an absence.** The hashing is right: the same
`sha256(filename-stem)[:12]` joins **35 of the 40** item-19 dump clips onto these caches (see the
S1 note). But **no cache on this box that is known to contain val40 clips could be found**, so no
positive control reads non-zero. ⇒ *"clean-124 shares no clip with val40"* is **INCONCLUSIVE —
probable, not established.** If it holds, the guard's live concern (6 shared clips = val40's
**entire** labelled support) does not touch refcv6. It still governs any comparison of the
**banked** 147-clip-era numbers with a val40 number.

## Recommendation (for the Master Mind; nothing here is started)

1. **Cheap and owed:** the 7 stale path constants in 6 live taniteval tests. Use the old
   harness: per file, keep a change only if failures strictly decrease, route through
   `research_tree.py`, and read skip reasons before and after.
2. **Decide, don't drift:** either land the val40 guard, recovered from the transcript and
   renumbered past **2.9.0** because 2.8.0 is taken, or record it as deliberately dropped. Its drift
   framing (a1) and five tests (b) come with it or not at all.
3. Record the **2.8.0 version collision** in the registry changelog. Two different 2.8.0s is the
   *"a label that implies a dependency"* class with the object swapped for a version number.
