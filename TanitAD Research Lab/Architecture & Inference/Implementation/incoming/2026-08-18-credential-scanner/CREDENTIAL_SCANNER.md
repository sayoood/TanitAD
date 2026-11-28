# The credential gate — built, bound, and measured

**Agent:** Architecture & Inference · **Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803`
**Start HEAD:** `8cf49ab` → repo advanced to `a603936` mid-session (siblings committed; see §9).
**GPU used:** none. **Thor:** not touched.

---

## 0. The headline, and it is a retraction of C117 — not of C111

> ⛔ **C117 states, "MEASURED at three probes … NO CREDENTIAL SCANNER EXISTS."**
> **That absence claim is WRONG.** A credential content-scanner has been in this repo since
> **2026-07-25**, in `tools/safe_commit.py` (`scan_secrets()`, commit `37158f7`) — and **MEASURED
> here, it catches the exact C111 token shape.**

**Why all three probes missed it** — this is the whole lesson, and it is *operating-standard rule 2*
("absence found at ONE location is not absence") failing at three locations that were **the same
location in three costumes**:

| C117's probe | why it was structurally unable to see `scan_secrets` |
|---|---|
| scripts/tests **by name** | the file is called `safe_commit.py`; nothing in its name says "scan" or "secret" |
| grep for `detect-secrets` / `trufflehog` / `gitleaks` | those are **third-party tool names**. A home-grown scanner contains none of them |
| `pre-commit` + `.github/workflows/` + the operating standard | the scanner is a **Python function inside a commit wrapper**, not a hook or a workflow |

All three asked *"has someone installed a scanner product?"*. None asked *"does any code in this repo
match a credential pattern?"* ⇒ **ROOT-CAUSE CLASS: three probes that are one probe.** Diversity of
*location* is not diversity of *question*, and the rule only pays out when the questions differ.

### So what actually caused C111

⛔ **Not a missing scanner. A scanner that NOTHING CALLED — C108's class exactly.**

And this was **already written down 24 days ago**. `…/incoming/2026-07-26-program-harvest/H2_UNUSED_CAPABILITIES.md`
row 1 records `tools/safe_commit.py`, *"imported_by: **Nothing** — only its own
`tools/tests/test_safe_commit.py`"*, *"Not referenced by `CLAUDE.md` at all"*, with the cheapest next
step spelled out (one line in CLAUDE.md's Git-hygiene section) and the evidence class **DECISION-GRADE**.
`PROGRAM_HARVEST.md:742` escalated it to the PI as one of three CLAUDE.md defects.

⇒ **C111 is the invoice for that unmerged one-line edit.** The tool was built to mechanise the exact
section of `CLAUDE.md` that agents follow by hand, the harvest found it orphaned and said so in
decision-grade terms, and the recommendation sat in a report. *That* is the pattern this work has to
break — and it is why the largest share of the effort below went into **what invokes the scanner**,
not into the scanner.

---

## 1. The counterfactual, measured

**MEASURED** — `scratchpad/counterfactual2.py`, 9 planted cases, each a fresh throwaway git repo, each
token **assembled at runtime from fragments** so no credential-shaped literal exists on disk.

| planted case | `safe_commit.scan_secrets` (existing) | `secret_scan --staged` (new) | `secret_scan --tree` (new) |
|---|---|---|---|
| **C111 exact shape** — `hf_…` on **line 11** of a rescued `.log` | **CAUGHT** | CAUGHT | CAUGHT |
| same token in `.json` | CAUGHT | CAUGHT | CAUGHT |
| same token in `.txt` | CAUGHT | CAUGHT | CAUGHT |
| OpenAI `sk-…` | CAUGHT | CAUGHT | CAUGHT |
| GitHub `ghp_…` | CAUGHT | CAUGHT | CAUGHT |
| AWS `AKIA…` | CAUGHT | CAUGHT | CAUGHT |
| PEM private-key header | CAUGHT (by path) | CAUGHT | CAUGHT |
| **generic `api_key = <high-entropy>`** | ⛔ **MISSED** | CAUGHT | CAUGHT |
| **token inside a BINARY blob** | ⛔ **MISSED** | CAUGHT | CAUGHT |
| | **7 / 9** | **9 / 9** | **9 / 9** |

⭐ **The first row is the finding.** Had `safe_commit.py` been used for the C111 commit — or had
anything invoked its scan — **the token would have been refused locally and GitHub's push protection
would never have been involved.**

### The two real gaps, and why the second one matters most

⛔ **`safe_commit` scanned `git diff --cached` — diff TEXT.** For a blob git classifies as binary that
diff is the single line *"Binary files … differ"* and carries **no content at all**, so a credential
inside one was **structurally invisible**. The new scanner reads staged **blobs** through
`git cat-file --batch`. This is not hypothetical for us: the rescue banked `.pt`, `.npz` and archive
payloads alongside the logs.

⚠️ The generic-assignment miss is the brief's own requirement and is now Tier A2 (§3).

---

## 2. What was built

| artifact | what it is |
|---|---|
| `tools/secret_scan.py` | the engine + CLI. Modes: `--staged` · `--tree DIR` · `--tracked` · `--history` · `--install-hook` · `--check-hook`. Stdlib-only, ASCII-clean stdout, OS-agnostic (runs on the pods) |
| `tools/secret_scan.py --install-hook` | writes the `pre-commit` gate into `git rev-parse --git-path hooks` (the **common** dir, so every worktree is covered at once, and `core.hooksPath` is honoured) |
| `stack/tests/test_secret_scan.py` | **72 falsifiers**, pinned both ways |
| `tools/ci_gate.py` | `SUITE_MANIFEST` gains `tests/test_secret_scan.py: 60` |
| `tools/tests/test_safe_commit.py` | one fixture de-literalised (§5, FP #7) |

### ⭐ What invokes it — stated explicitly, because that is the failure being fixed

1. **`.git/hooks/pre-commit` — every `git commit` in this repo and all its worktrees.** Unconditional,
   independent of whether anyone remembers a procedure. It scans the staged **blobs** and refuses on
   any Tier A/A2/B finding. **This is the exact control point C111 passed through.**
   *Bypass requires saying so out loud:* `SECRET_SCAN_SKIP=1`, which prints a line telling you to
   justify it in the commit message.
2. **`safe_commit.py` inherits it for free.** MEASURED: `safe_commit` does **not** pass `--no-verify`,
   so the sanctioned commit path fires the same hook. Its own Tier-A scan stays as a redundant,
   earlier belt — two independent checks, one of them now unavoidable.
3. **`stack/tests/test_secret_scan.py::test_hook_is_installed_and_current`** — the anti-C108 test.
   `pytest -q` green is already a hard `CLAUDE.md` invariant before any commit, and
   `tools/ci_gate.py --rootdir stack` is protocol G-E "before every commit/push". **If the hook is not
   armed on a clone or worktree, the mandated gate goes red and names the one-line fix.**
4. **`ci_gate.SUITE_MANIFEST`** — deleting or shrinking the suite below 60 collected tests fails the
   gate. Single-node tripwires only guard nodes somebody thought to name; the manifest guards the
   module, which is how coverage actually disappears with several agents in one tree.
5. **`--tree DIR` is C111's literal rule** — the cheap scan of an imported tree *before* anything is
   staged. It is the *earliest* control, not the binding one; the hook is what makes forgetting it
   survivable.

⚠️ **Known residual gap, stated rather than papered over:** a **fresh clone** has no hooks until
someone runs `--install-hook`. Item 3 catches it the first time anyone runs the suite, which in this
programme is before any commit — but between clone and first pytest there is a window. Closing it
properly needs `core.hooksPath` committed in-repo (a config change) — **PI's call, escalated in §8.**

---

## 3. Patterns, and the false-positive budget they were designed against

**Tier A — provider shapes (BLOCKING, no override).** `huggingface` · `openai-anthropic` (incl.
`sk-ant-`/`sk-proj-`) · `github-pat` (`ghp_ gho_ ghu_ ghs_ ghr_`) · `github-fine` (`github_pat_`) ·
`aws-akid` (`AKIA`+`ASIA`) · `slack` · `google-api` · `private-key` PEM header.
Each carries a **length floor**: the brief's loose `hf_[A-Za-z0-9]+` matches this repo's own committed
`hf_export.py`, `hf_relay`, `hf_repo_state_*.json`, so the blocking tier demands real token length and
the loose form is **advisory**.

**Tier A2 — generic high-entropy assignment (BLOCKING).** `api_key|secret|token|password|passwd|access_key|auth_token|client_secret|bearer`
followed by `=`/`:` and a value, gated on **all** of: length ≥ 20 · Shannon entropy ≥ 3.4 bits/char ·
placeholder deny-list · **not a 32/40/64 hex digest** (this repo commits thousands of md5/sha256
digests — without this the rule is pure noise) · not a path/URL · **credential alphabet only** ·
not a dotted identifier · text files only.

**Tier B — credential-shaped paths (BLOCKING, path only, file never opened).** Exact names and
extensions: `Keys.txt`, `*.pem/.key/.pfx/.p12`, `.env`, `.env.*`, `id_rsa`, `id_ed25519`, `.netrc`,
`*_token.txt`, `*token.json`, `gotty_url.txt` (`.gitignore` records that a RunPod web-terminal URL
carries a root credential).

**Tier C — advisory.** Loose `hf_*` identifiers. Never dropped silently; always counted.

**Every filter travels with its counts as data** (`filter_rule` in the JSON), carrying the explicit
warning that it is *a filtered view, not a census*.

---

## 4. ⛔ The number that is the deliverable

**MEASURED 2026-08-18 · `python tools/secret_scan.py --tracked` · artifact `raw/tracked_final.json`**

```
scope=tracked
candidates = 6201   files scanned = 6201   bytes = 662,095,924
skipped (COUNTED and NAMED, not dropped): compressed-suffix=518, too-large=7
BLOCKING (0) -- clean
advisory (64)
```

> ### **FALSE POSITIVES OVER THE WHOLE TRACKED REPO: 0**
> 6,201 files · 662 MB · **0 blocking** · 64 advisory (60 loose `hf_*` identifiers + 4 `*.env` run
> manifests), every one of which is *reported* and none of which refuses anything.

Runtime **29 s** (down from 129 s, §6). Skipped files are **named** in the JSON, not merely tallied —
518 compressed/media blobs where a plaintext pattern cannot exist, and 7 tracked `.pt`/tensor dumps
over the 25 MB cap (raise with `--max-bytes` for a targeted check).

---

## 5. Getting to zero: the 7 findings of the first run, each classified

The first whole-repo run produced **7 blocking findings — all 7 artifacts.** Narrowed **with reasons**,
never by widening an exemption:

| # | finding | verdict | narrowing, and why it is safe |
|---|---|---|---|
| 1–2 | `generic-assignment` on `tokens = head.build_tokens(st4` (21 chars, **4.01 bits/char**) in two `incoming/*/code/*.py` | **artifact** — a *code expression* assigned to a variable named `tokens` | value must be **credential alphabet only** (`A-Za-z0-9+/=_-.:`) and not a dotted identifier. Brackets and dotted calls are what separate code from credentials |
| 3–6 | `*.env` on 4 files incl. `stack/ops/runs.d/flagship-v5f-w120-30k.env` | **artifact ×4 of 4** — in this programme `*.env` is the **supervisor run-manifest** convention, not a dotenv secret store | `*.env` → **advisory**; `.env` / `.env.*` proper stay blocking. ⭐ Nothing is lost: these files are still **content-scanned** like everything else. The path tier is a redundant belt; the content scan is the braces |
| 7 | `private-key` at `tools/tests/test_safe_commit.py:159` | **artifact** — a PEM header **literal** in a test fixture | fixed at the source: the fixture is now **assembled at runtime**. ⇒ standing rule: **no credential-shaped literal in any repo file, tests included** |

⚠️ **A gate that refuses a normal deliverable is switched off inside a week (C118).** 4 of these 7
were the programme's own run manifests. Shipping that would have been the `pod_git_drift` 63.6 %-artifact
mistake repeated on a surface where the cost is people disabling the guard.

---

## 6. Four defects in my own build, found by measurement, not review

⭐ Each was caught by a test or a timing run, not by reading the code — recorded because the *class*
recurs.

1. ⛔ **The scanner matched its own filename.** The glob `*secret*` fired on **`tools/secret_scan.py`
   and `stack/tests/test_secret_scan.py`**, turning the repo-wide gate red on the day it was written.
   *This is the polling-monitor self-match trap in the path tier* — a filter matching the thing that
   names it. ⇒ Substring globs (heuristics) do not apply to **source/doc** files, whose content is
   scanned anyway; exact names and extensions (declarations) stay unconditional. `.txt` is
   deliberately **not** exempt — that is `Keys.txt`'s own shape and C111's own file class.
2. ⛔ **The hook refused CLEAN commits on Windows.** `command -v python3` **succeeds** because it finds
   the **Microsoft Store alias stub**, which then prints *"Python wurde nicht gefunden"* and exits 1
   ⇒ **every commit blocked.** *Presence of a binary is not the ability to run it, the same way an
   exit code is not evidence.* ⇒ the hook now probes each candidate by **running** `"$py" -c ""`.
   Caught by `test_the_hook_lets_a_clean_commit_through` — the C95/C97 "rejects-everything guard" half
   of the both-ways pin, which is exactly why that half exists.
3. ⚠️ **`--tree` missed a token inside a `.bin` that `--staged` caught.** The first skip list carried
   `.bin/.npy/.h5/.parquet/.so/.pt` — chosen for **byte volume** rather than for *whether the answer
   could be in there*. Those containers are **uncompressed**: an ASCII token survives verbatim.
   ⇒ narrowed to **entropy-coded formats only**, where a plaintext pattern provably cannot exist.
4. ⛔ **A false ALL-CLEAR, measured for real (§7).** `--history` printed *"files scanned = 0 …
   **BLOCKING (0) -- clean**"* and exited **0** while reading **nothing**. ⇒ `Report.unusable`:
   candidates-without-reads, or any git/read error, now makes the whole report **UNUSABLE** and the
   exit code non-zero, with a banner saying the verdict must not be read as a pass.

⚡ **And a performance defect that was a correctness defect:** the first cut took **106 s** over
661 MB — over `ci_gate`'s 15 s per-test budget, i.e. the repo-wide pin would have been **un-gateable**
and the guard would have been switched off *for being slow* rather than for being wrong. Merging the
patterns into one alternation changed nothing (**still 6 MB/s**: every branch starts with `\b`, so
`re` tries all 8 alternatives at all 661 M positions). ⇒ each pattern is now gated behind its own
**mandatory literal** tested with C-speed `str.__contains__`. **129 s → 29 s.** This drops nothing:
every Tier A pattern provably requires one of its literals.

---

## 7. ⏳ The committed-history audit is **NOT COMPLETE** — and the instrument says so

**Brief item 5** asked whether any *committed* blob still carries a credential pattern.

⛔ **I cannot answer it yet, and I am not going to report the run I have as an answer.**

**MEASURED:** the Google-Drive-backed `G:` volume this repo lives on **dropped mid-session** and has
been flaking since — `git rev-parse` → *"fatal: not a git repository"*, `head -c 60 tools/secret_scan.py`
→ *"Invalid request code"*, `import secret_scan` → `OSError: [Errno 22] Invalid argument`, all while
the directory listing works. One outage exceeded **10 minutes**.

The audit run under that fault returned:

```
scope=history
candidates = 5487   files scanned = 0   bytes = 0
*** SCAN UNUSABLE: 14 read/git error(s) -- the scan is INCOMPLETE
      git cat-file --batch failed (rc=128): fatal: not a git repository
BLOCKING (0) -- clean          <-- exit 1, NOT 0
```

⭐ **This is the instrument working.** Before the §6.4 fix the identical run printed
*"BLOCKING (0) -- clean"* and exited **0** — a perfect all-clear manufactured by reading zero of 5,487
candidate blobs. **A guard that cannot see its subject must not report on it.**

**What is known:** the audit covers **reachable and unreachable** blobs
(`git cat-file --batch-all-objects`), and separates them — a pattern in a **reachable** blob means a
credential is in committed history and is escalated, **never** rewritten by an agent; a pattern in an
**unreachable** blob is debris from an undone commit, which is what C111 records for the reset-away
commit `ab77da96`, and it disappears on `git gc --prune=now`.

⚠️ **AND THE OBVIOUS HEALTH PROBE LIES ABOUT THIS FAULT — worth its own line, because any agent on
this box will hit it.** In this failure mode the volume serves **metadata but not content**:
`ls G:/…/TanitAD/tools/` lists all 19 entries and `ls -la .git/HEAD` returns a valid stat, **while
`head -c 60` on the same file returns *"Invalid request code"* and `git rev-parse` returns *"not a
git repository"***. ⇒ **A directory listing is not evidence that files are readable.** Same family as
`df` on a pod, `free`/`tegrastats` on Thor and `memory.usage_in_bytes` in a cgroup: *a probe that
answers a different question than the one being asked, and therefore looks like an answer.* The
admissible probe is a **real content read** — which is exactly why the wait loop here polls
`git rev-parse HEAD && head -c 100 <file>` and not `ls`.

⇒ **Re-run when the volume is stable:** `python tools/secret_scan.py --history --json history.json`.
Exit 0 = genuinely clean; exit 1 with the UNUSABLE banner = ran under a fault; exit 1 with findings =
**stop and escalate to the PI.**

⚠️ **Partial reassurance that does NOT substitute for the audit:** the **working tree** is clean at
0 blocking over 6,201 tracked files (§4), which includes the entire 117-file C111 rescue package at
`…/incoming/2026-08-18-thor-stranded-rescue/rescued/` — **so C111's in-tree redaction held.** That is
a statement about the tree, not about history.

---

## 8. ⏳ PI actions

1. ⛔ **ROTATE THE C111 TOKEN — still open, still time-sensitive.** C111 records it as **still
   plaintext on Thor** at `~/rq_out/logs/contention.log:11`. It has WRITE access to the `Sayood/` HF
   namespace. Redacting our copy does nothing to the machine's. *Nothing in this work changes that.*
2. **Finish the history audit** (§7) once `G:` is stable — one command, no GPU.
3. **Decide the fresh-clone gap** (§2): commit `core.hooksPath` pointing at an in-repo hooks
   directory, so a clone is armed with no manual step. It is a repo-config change, so it is yours.
4. ⚠️ **The one-line `CLAUDE.md` edit from the 2026-07-26 harvest is still unmade.** Git-hygiene still
   instructs raw `git commit -F`; `tools/safe_commit.py` is still unmentioned. **That unmade edit is
   the mechanism by which C111 happened.** The hook now covers raw `git commit` too, so this is no
   longer load-bearing — but it is still the correct fix and it is still one line.

---

## 9. Notes for whoever reads the index

⚠️ **HEAD moved under this work:** `8cf49ab` → **`a603936`** while the volume was down. Siblings
committed; my files were untouched. *Re-check `git status` before assuming anything about the index —
staging is not a latch.*

⚠️ **`Keys.txt` was never read into an argument, printed, or committed.** It is scanned like any other
file and reported **by path only**; `test_keys_txt_contents_are_never_emitted` plants a live-shaped
token in a `Keys.txt` and asserts the value appears in neither the rendered output nor the JSON.
**No matched value is ever printed anywhere** — findings are `(path, line, pattern name, redacted
length)`, which is how C111 itself was handled, deliberately.

---

## 10. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `tools/secret_scan.py` | repo | **NEW, staged** |
| `stack/tests/test_secret_scan.py` (72 falsifiers) | repo | **NEW, staged** |
| `tools/ci_gate.py` (manifest entry) | repo | **MODIFIED, staged** |
| `tools/tests/test_safe_commit.py` (de-literalised fixture) | repo | **MODIFIED, staged** |
| `tools/README.md` (secret_scan row + section) | repo | **MODIFIED, staged** |
| this report | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-credential-scanner/CREDENTIAL_SCANNER.md` | **NEW, staged** |
| whole-repo scan JSON | `…/2026-08-18-credential-scanner/raw/tracked_final.json` | **NEW, staged** |
| counterfactual panel | `…/2026-08-18-credential-scanner/raw/counterfactual.txt` | **NEW, staged** |
| the installed hook | `.git/hooks/pre-commit` (**not tracked by design**) | **armed on this box**; arm elsewhere with `--install-hook` |

**Not committed, not pushed** — per the operating standard.

### Test evidence

* `stack/tests/test_secret_scan.py` — **71 passed, 1 skipped (72 collected), 9.8 s** (the skip is the opt-in full
  6,201-file scan; run it with `SECRET_SCAN_FULL=1`, reason printed).
* `tools/tests/test_safe_commit.py` — **29 passed**, unchanged by the fixture edit.
* Suites run **separately**, `PYTHONUTF8=1`, not under CPU contention.
* The installed hook **executed against this repo's real index**: `[secret_scan] OK -- N staged blob(s) clean`, exit 0.
