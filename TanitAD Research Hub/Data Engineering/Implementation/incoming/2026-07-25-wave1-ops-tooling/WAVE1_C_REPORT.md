# Wave-1 workstream C — three ops tools, one recurring loss each

**Date:** 2026-07-25 · **Branch:** `agent/benchmarks-eval-20260721` · **Scope:** dev-box only
(no pod, no GPU touched; three trainings were live throughout).

**Deliverables are STAGED-READY IN THE WORKING TREE, NOT COMMITTED** — per the brief the
orchestrator stages and commits (a parallel `git add` corrupts the index). Nothing here
was `git add`ed by me.

---

## 0. What each tool kills

| Tool | The recurring loss it removes | Evidence class of the loss |
|---|---|---|
| `tools/safe_commit.py` | A sibling agent's half-finished work swept into the wrong commit (**twice**: `60265d3` ate the eval tooling, `3d41bd0` ate REF-C v1.2's rescorer); `Keys.txt`/tokens reaching a commit; the phantom `index.lock` after the intermittent `git commit -- <pathspec>` segfault | MEASURED — CLAUDE.md §Git-hygiene, `RETRACTION_LOG` 07-25 (C8) |
| `tools/registry_lint.py` | A retracted headline standing in a `MODEL_REGISTRY.md` **section header for 4 days after its own body refuted it**, plus a second instance that wrapped across a newline and evaded a line-based grep; and hand-transcription drift from the raw eval JSON | MEASURED — `RETRACTION_LOG` 07-25 (C4) |
| `tools/repo_janitor.py` | A **false "these modules are stranded, not in main" claim that reached the PI**, caused by an mtime-sorted `Glob` truncated at 100 over 51 worktrees + ~95 `incoming/` bundles | MEASURED — `RETRACTION_LOG` 07-24 (C8) |

---

## 1. `tools/safe_commit.py`

### Invocation

```bash
python tools/safe_commit.py -p tools/ -p "TanitAD Research Hub/Data Engineering" \
    -m "tools: wave-1 ops tooling"
python tools/safe_commit.py --print-index                  # the check CLAUDE.md demands FIRST
python tools/safe_commit.py -p tools/ -m "..." --dry-run
python tools/safe_commit.py --accept-index -m "..."        # whole index, foreign paths NAMED
```

### The four properties, and why each exists

1. **It can never emit the crashing form.** `git commit -- <pathspec>` is git's *partial-commit*
   path; measured 2026-07-25 it segfaults intermittently on this repo (exit 139 MSYS /
   `0xC0000005` native — so not the shell; fsmonitor already `false`). Three root-cause
   theories were asserted into `CLAUDE.md` and falsified in one session (C8). The tool
   therefore only ever runs `git commit -F <file>`, and a test asserts the exact argv shape
   (`test_the_tool_never_emits_a_pathspec_or_amend`). `--amend` is likewise never passed —
   it re-opens the whole index and defeats every check above it.
2. **Pathspec-free means whole-index, so the index is *declared*.** `--path/-p` accepts files,
   directory prefixes and segment-wise globs; anything staged and undeclared **aborts** and is
   printed with its diffstat. `--accept-index` is CLAUDE.md step 1 made mechanical: it proceeds
   *and appends the foreign paths to the commit message*, so a sibling's work is recorded
   rather than hidden.
   *(Note: `*` deliberately does not cross `/`. Plain `fnmatch` lets `tools/*.py` cover
   `tools/sub/deep.py`, and an over-broad declaration silently re-admits the exact failure
   this guard exists to stop.)*
3. **Secrets: three independent probes** (CLAUDE.md operating-standard 2 — absence at one
   location is not absence): git's own `check-ignore --no-index` verdict, the filename shape,
   and the staged content. Findings are printed **redacted**.
   ⚠️ The brief's literal `hf_[A-Za-z0-9]+` is kept only as an **advisory** tier — it matches
   this repo's own committed `hf_export.py`, `hf_relay`, `hf_repo_state_2026-07-25.json`, so
   blocking on it would make the tool unusable. The **blocking** HF pattern is the real token
   shape (`hf_` + ≥30 alnum), alongside `sk-`, `gh[pousr]_`, `AKIA`, `xox[baprs]-` and PEM
   private-key headers.
4. **The phantom lock.** Confirms no `git` process is alive — matched on the **image name**
   (`tasklist` / `ps -e -o comm=`), never `pgrep -f`, which self-matches the caller — then
   removes `.git/index.lock` and retries (default 3). **HEAD movement, not the exit code, is
   the success signal**: a crash can land the commit and still return 139, so a naive retry
   would double-commit.

Plus: refuses to commit on `main`; refuses an empty index; `--dry-run`; `--print-index`.

### MEASURED self-test — RED

Throwaway repo, my file staged alongside a sibling's WIP:

```
[safe_commit] index  = 3 path(s)
    .gitignore                        <-- NOT DECLARED
    stack/tanitad/sibling_wip.py      <-- NOT DECLARED
    tools/mine.py
[safe_commit] REFUSED (1):
    - 2 staged path(s) are NOT covered by your --path declarations. `git commit` commits
      the WHOLE INDEX, so these WOULD ride along under your message. ...
[safe_commit] their diffstat:
     .gitignore                   | 1 +
     stack/tanitad/sibling_wip.py | 1 +
RC=1                                    (HEAD unmoved)
```

`Keys.txt` force-staged into a repo that git-ignores it — **all three probes fire independently**:

```
[safe_commit] REFUSED (3):
    - SECRET GUARD [ignored-path] Keys.txt: staged despite being git-IGNORED (only 'git add -f' does this)
    - SECRET GUARD [secret-path]  Keys.txt: filename matches a credential glob (Keys.txt)
    - SECRET GUARD [token]        Keys.txt: huggingface token shape: hf_***<37 chars, redacted>
RC=1
```

`grep -c` for the token body across the tool's full stdout+stderr: **0 occurrences.**

### MEASURED self-test — GREEN

`tools/tests/test_safe_commit.py` — **29 falsifiers, 7.7 s, all passing.** Each drives a real
throwaway git repo. Notable ones: `test_siblings_staged_work_aborts_the_commit`,
`test_gitignored_keys_txt_is_refused`, `test_token_is_never_echoed`,
`test_ordinary_hf_filenames_are_not_blocked`, `test_stale_index_lock_is_cleared_and_the_commit_proceeds`,
`test_a_crashed_git_that_still_committed_is_not_retried`, `test_a_true_crash_is_retried`,
`test_the_tool_never_emits_a_pathspec_or_amend`.

---

## 2. `tools/registry_lint.py`

### Invocation

```bash
python tools/registry_lint.py                       # the standard sweep
python tools/registry_lint.py --strict               # body + boilerplate hits also fail
python tools/registry_lint.py --self-test            # 5 red/green falsifiers
python tools/registry_lint.py --json lint.json
```

### CHECK 1 — pointer drift

Syntax (inline, preferred long-term):

```markdown
<!-- src: taniteval/results/driving_flagship-30k.json#headline.ade_0_2s.mean near="full-set" -->
| **1=** | **Flagship v1 (speed+jerk) FINAL** | ... *(full-set 0.4271, boot [0.3675, 0.4871])* |
```

Design points that matter:

- **Tolerance is the coarser of the two written precisions.** `0.4271` tolerates 5e-5 and
  `0.452` tolerates 5e-4, but a prose `0.43746` against a stored `0.4375` **passes**, because
  either side may legitimately have been rounded (this is a real registry pair, §1.4b vs
  `driving_flagship-v16-ab-ft.json`).
- `near="…"` restricts candidates to numbers after a marker, so a row carrying both the
  deprecated split-mean and the full-set value is checked against the right one.
- A missing file, missing field or non-numeric field is **itself a finding** — a dangling
  pointer is drift.

⚠️ **Seeding was done via a SIDECAR (`tools/registry_pointers.jsonl`), not by editing
`Project Steering/MODEL_REGISTRY.md`.** The brief lists `Project Steering/` as a
sibling-owned path this session, and a seeding patch that races a concurrent edit is exactly
the accident `safe_commit` exists to prevent. Sidecar rows are keyed by an **anchor regex that
must match exactly one line** (an ambiguous anchor is reported, because a pointer that silently
relocates is worse than one that is missing). **Migrating a row to the inline form is a
one-line edit and should be done whenever that row is next touched.**

**5 pointers seeded on the highest-traffic rows** — the three-way rank-1 tie in the §6
cross-arm leaderboard (`flagship-30k` 0.4271, `refc-xl-30k` 0.4714, `refc-base-30k` 0.4728),
the §1.4b headline-retraction point estimate, and the §4.1 prose restatement of the XL
full-set number.

### CHECK 2 — the multiline retracted-claim sweep

The document is tokenised into a **whitespace-collapsed stream with a token→line map**, so
markdown line breaks disappear and a claim wrapped across a newline is adjacent in the stream.
Matching allows `--gap` inserted tokens (default 1). A hit whose span touches a `#` header is
an **ERROR**; body prose is a **WARN** (a body legitimately quotes what it retracts). Hits near
a retraction marker, or on a line carrying `<!-- lint-ok: … -->`, are suppressed.

`--rare-max` (default 25) demotes a header hit built entirely from house vocabulary.
MEASURED on the registry (26,459 tokens): the neutral title `### 4.4 REF-C CLOSED-LOOP …`
matches the retracted *"flagship v1 beats REF-C closed-loop"* on ref=122 / c=81 / loop=60 /
closed=32, whereas the real stale headline carries best=23 and program=13. The demotion is
**visible, not silent** — it still prints, and `--strict` makes it fatal.

### MEASURED self-test — the decisive historical regression

Run against the **real pre-correction registry blob** `c5e5d5f:Project Steering/MODEL_REGISTRY.md`
(the last commit carrying the stale header) together with **that same commit's retraction log**
— i.e. using only the information that existed during the four days:

```
=== c5e5d5f registry + c5e5d5f retraction log, DEFAULT gap=1 ===
registry_lint: 1 file(s), 0 pointer(s), 39 retracted claim(s) loaded
[ERROR] Project Steering/MODEL_REGISTRY.md:460: RETRACTED CLAIM IN A SECTION HEADER: "best ade in the program"
RESULT: FAIL (1 error(s), 1 warning(s))          <- RED, on the real header, on day 1

=== same inputs, --gap 0 ===
RESULT: PASS (0 error(s), 1 warning(s))          <- reproduces the 4-day MISS
```

Line 460 of that blob is verbatim:
`### 1.4b flagship-v1.6 — \`flagship-v16-ab-ft\` — ✅ **COMPLETE at 5,999** · ⭐ best ADE in the program`

**Why `--gap` is load-bearing:** the 07-21 log quotes the claim as *"best **in** the program"*;
the header said *"best **ADE** in the program"*. One inserted word is the entire difference,
and a strict n-gram sweep run on 07-21 would have reported clean. This is asserted as a test
(`test_one_inserted_word_does_not_hide_a_claim`).

### MEASURED self-test — RED (pointer drift, on the real registry)

A copy of the live registry with the rank-1 full-set ADE mutated `0.4271 → 0.4420`:

```
[ERROR] Project Steering/MODEL_REGISTRY.md:1470: DRIFT:
        taniteval/results/driving_flagship-30k.json#headline.ade_0_2s.mean = 0.4271
        but the line quotes [0.4420, 0.3675, 0.4871, 0.9437, 0.0602]
RESULT: FAIL (1 error(s), 1 warning(s))
```

### MEASURED self-test — GREEN

```
$ python tools/registry_lint.py --self-test
  [ok] GREEN clean doc (correct number, corrected header): exit 0
  [ok] RED   stale headline WRAPPED ACROSS A NEWLINE: exit 1
  [ok] RED   pointer drift (0.4420 vs JSON 0.4271): exit 1
  [ok] RED   REAL c5e5d5f header vs the 07-21 log wording (needs gap=1): exit 1
  [ok] CTRL  same doc+log at gap=0 -- reproduces the 4-day MISS: exit 0
SELF-TEST: PASS

$ python tools/registry_lint.py                     # the LIVE registry, today
registry_lint: 1 file(s), 5 pointer(s), 42 retracted claim(s) loaded
[warn] ...:1346 header matches retracted-claim vocabulary, but every word is house boilerplate
RESULT: PASS (0 error(s), 1 warning(s))
```

All 5 seeded pointers bind and agree with the committed eval JSON — asserted as a test
(`test_seeded_sidecar_pointers_resolve_against_the_live_repo`), so a stale seed fails CI.

`tools/tests/test_registry_lint.py` — **25 falsifiers, 1.1 s, all passing.**

---

## 3. `tools/repo_janitor.py`

### Invocation

```bash
python tools/repo_janitor.py                                    # report everything
python tools/repo_janitor.py --ledger-out INCOMING_LEDGER.md    # the dated ledger
python tools/repo_janitor.py --fast                             # skip per-worktree status
python tools/repo_janitor.py --delete                           # remove ONLY safe candidates
```

### MEASURED — the live repo, 2026-07-25

```
[worktrees] 51 registered | 16 SAFE-TO-DELETE candidate(s) | 18 hold unique commits
[incoming]  97 bundle(s) | 8 older than 14d | 87 with no filled INTAKE verdict | 49.0 MB total
[future]    0 tracked file(s) with a future mtime | 1 bundle with a future date slug
    slug 2026-07-31 (6d ahead)
    TanitAD Research Hub/Opponent Analyzer/Implementation/incoming/2026-07-31-stationary-lead-scenario
```

Wall-clock: **11.9 s** with the full per-worktree status probe, 8.9 s under `--fast`.

**51 worktrees × mostly-fresh mtimes is the whole mechanism of the 07-24 false claim**, and
**18 of them hold commits the tip does not** — so the sprawl is not merely cosmetic, it is
also real unmerged work that the ledger now makes countable.

### Safety property

Deletion is **opt-in and narrow**. A worktree is a CANDIDATE only if it is `0 commits ahead of
the tip` **and** its working tree is clean; `--delete` never passes `--force`, and it **refuses
to run under `--fast`** because skipping the status probe would mean deleting without a
clean-tree check. Asserted by `test_delete_removes_only_candidates` (holder and dirty worktrees
survive) and `test_delete_refuses_without_a_status_check`.

### Artifacts produced here

- `INCOMING_LEDGER_2026-07-25.md` — 97 bundles, oldest first, with age / files / size /
  INTAKE-verdict state. The future-dated bundle is rendered as `6d AHEAD`.
- `janitor_report_2026-07-25.json` — the same data machine-readable.

`tools/tests/test_repo_janitor.py` — **16 falsifiers, 4.7 s, all passing.**

---

## 4. Suite status

```
$ pytest tools/tests -q                       ->  127 passed in 27.23 s
$ python tools/ci_gate.py --rootdir stack     ->  862 passed, 3 skipped, GATE PASS (72.7 s)
```

`tools/tests` was 55 falsifiers before this work; the three new suites add 70.

---

## 5. Findings surfaced in passing (not acted on — out of scope)

1. **87 of 97 incoming bundles have no filled INTAKE verdict**, 8 of them older than 14 days
   (oldest 17 d). `session_guard --strict` already warns at 3 days; the ledger now names them.
2. **16 worktrees are safe to delete** and **18 hold unmerged commits** vs the tip. The 18 are
   the interesting number — that is stranded work by the AGENT_OPERATING_STANDARD definition,
   and `repo_janitor` only reports it. Deletion of the 16 is a one-command, reviewable action
   (`python tools/repo_janitor.py --delete`) but was **not** performed: destroying anything
   unasked is out of bounds.
3. **One future-dated bundle** (`2026-07-31-…`, 6 days ahead) — the known narrative-clock
   artifact, now visible rather than silently sorting first in every mtime-ordered search.
4. `registry_lint --strict` reports **one body-prose WARN** on the live registry
   (§4.4 header, house-vocabulary match). It is a false positive of the sweep, correctly
   demoted; no registry edit is needed.

---

## 6. Deliverable manifest

Every artifact and where it lives. **Nothing is staged, nothing is committed, nothing is on a
pod or a worktree** — all paths are in the main repo working tree.

| Artifact | Path (repo-relative) | State |
|---|---|---|
| safe_commit tool | `tools/safe_commit.py` | NEW, untracked |
| registry_lint tool | `tools/registry_lint.py` | NEW, untracked |
| repo_janitor tool | `tools/repo_janitor.py` | NEW, untracked |
| seeded drift pointers | `tools/registry_pointers.jsonl` | NEW, untracked |
| safe_commit falsifiers (29) | `tools/tests/test_safe_commit.py` | NEW, untracked |
| registry_lint falsifiers (25) | `tools/tests/test_registry_lint.py` | NEW, untracked |
| repo_janitor falsifiers (16) | `tools/tests/test_repo_janitor.py` | NEW, untracked |
| tools documentation | `tools/README.md` | MODIFIED (3 rows + 3 sections added) |
| this report | `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-25-wave1-ops-tooling/WAVE1_C_REPORT.md` | NEW, untracked |
| incoming ledger | `…/2026-07-25-wave1-ops-tooling/INCOMING_LEDGER_2026-07-25.md` | NEW, untracked |
| janitor JSON | `…/2026-07-25-wave1-ops-tooling/janitor_report_2026-07-25.json` | NEW, untracked |

**No file under `taniteval/`, `stack/`, or `Project Steering/` was touched** (sibling-owned
this session). The registry pointer seeds live in `tools/` for exactly that reason.

## 7. Escalations for the orchestrator

1. **Adopt `safe_commit.py` as the commit path** and add a line to `CLAUDE.md`
   §"Git hygiene" pointing at it. The section currently describes the procedure in prose; the
   tool is that prose, executable. *(I did not edit `CLAUDE.md` — it is steering-owned.)*
2. **Wire `registry_lint.py` into `ci_gate`** or the session-end guard, so a stale headline
   cannot survive a single session, let alone four days.
3. **Migrate the 5 sidecar pointers to inline `<!-- src: … -->` comments** the next time
   `MODEL_REGISTRY.md` is edited; the inline form survives a row being moved, the anchor form
   only survives the anchor text being stable.
