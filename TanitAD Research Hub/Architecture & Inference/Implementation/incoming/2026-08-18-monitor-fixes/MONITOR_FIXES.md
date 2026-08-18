# Two instruments that could not report what they exist to report — repaired

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Base:** `6784455`
**Evidence class:** MEASURED (ours) unless marked otherwise. Artifact paths given per claim.
**Live run:** Thor PID 25477 (`v6F-SW-30k`) untouched — see §4.

---

## 0. Headline

> ✅ **JOB 1 — the step-time monitor now exists.** `train_v6_staged.py` logs a true **marginal**
> `step_s_interval` alongside the unchanged cumulative `step_s`, plus `steps_this_process` as a
> first-class key. A guard (`stack/scripts/step_time_guard.py`) is pinned in **both** directions and
> **demonstrated end-to-end**: on a synthetic 40 s/step catastrophe it returns `TRIP` while the
> cumulative field peaks at **26.875** against a **27.677** threshold — i.e. the old criterion
> reports "safe" on the same data.

> ✅ **JOB 2 — `pod_git_drift.py` can see 45 % more of the box, and says what it cannot see.**
> The filter went from `(.py, .sh)` to 21 suffixes split **source / artifact**, the **inclusion rule
> prints before every count**, and the `NAME_ONLY` downgrade that let `thor_profile.py` escape is
> now `NAME_DRIFT` for unambiguous program-authored source.

> ⛔ **RUNNING IT FOUND A REAL BUG THE OLD TOOL COULD NOT REPORT: plain `xargs` splits on
> whitespace, so every path containing a space was torn into fragments and never hashed — 415 of
> 3,397 files on Thor (12.2 %) vanished silently.** The pre-cap count added for the census
> requirement is what made it visible. Fixed with `tr '\n' '\0' | xargs -0`.

> ⚠️ **AND THE OVER-CORRECTION THE BRIEF WARNED ABOUT WAS REAL, MEASURED, AND CORRECTED.** The first
> widened build promoted **478** basename hits on Thor, of which **304 (63.6 %) were result JSONs
> and logs whose names merely coincide with a repo file.** Restricting the promotion to `source`
> kind takes it to **174 · 0 artifacts** with no loss of the C110 case.

> ⚠️ **TWO CLAIMS OF MY OWN WERE FALSIFIED BY MY OWN TESTS AND ARE CORRECTED BELOW** (§1.3, §1.4).
> Both were written into a docstring first and caught before staging.

---

## 1. JOB 1 — `step_s` cannot be used as a live monitor

### 1.1 What changed in the trainer

`stack/scripts/train_v6_staged.py`, **additive only**:

| key | meaning | status |
|---|---|---|
| `step_s` | elapsed / steps-this-process — **cumulative mean** | ⚠️ **UNCHANGED** in name, formula and value |
| `step_s_interval` | **marginal** s/step since the previous logged row | ⭐ NEW — this is the monitor |
| `steps_this_process` | `step - start_step`, first-class | ⭐ NEW — resets on restart ⇒ segment detection |
| `step_s_note` | now states that it is cumulative and cannot fire | amended prose only |

`step_s` is deliberately not redefined: banked logs, the ~5.3-day ETA arithmetic and the recovery
path in §1.2 all key off it. One `time.time()` read now serves both fields, so first-differencing
`step_s` reconciles **exactly** with `step_s_interval` rather than drifting by the microseconds
between two clock reads.

### 1.2 Banked logs are not lost — the cumulative series is invertible

`elapsed_i = step_s_i · n_i`, so
`r(i) = (step_s_i·n_i − step_s_{i−1}·n_{i−1}) / (n_i − n_{i−1})` recovers the marginal rate
**exactly**, not approximately. `step_time_guard.py` prefers the native field and falls back to this,
so it works on every log the programme already holds. (`n` is read from `steps_this_process`, or
regex'd out of `step_s_note` for older logs — prose is the fallback, never the contract.)

### 1.3 ⛔ The C112 proof, now a regression test rather than a paragraph

Process 6,300 steps in at 26.4749 s/step (elapsed 166,791.9 s) suddenly runs at **40 s/step**
(+51 %) for the ~2 h a pilot lasts:

| instrument | reading | verdict at +5 % |
|---|---|---|
| `step_s` (cumulative) | peaks **26.789** | **under 27.677 ⇒ "safe"** |
| `step_s_interval` (marginal) | **40.00** | **TRIP on the 2nd interval** |

At the *intended* trip rate the cumulative mean **never** arrives — its asymptote is that rate.
Pinned by `test_the_cumulative_mean_needs_hours_to_reach_the_trip_point`, which also measures that
even at 40 s/step it needs **> 6 h** of wall clock.

**Demonstrated end-to-end, not only in unit tests** — the CLI on a synthetic log:

```
VERDICT TRIP · worst 40.0029 s/step (+51.760 %) · 2 consecutive over 27.6774 · exit 1
```

### 1.4 ⚠️ Where the tolerance actually sits — and the claim I got wrong

MEASURED on the live `v6F-SW-30k` log
(`…/incoming/2026-08-18-o2-live-and-ridge-reread/raw/v6F-SW-30k_train_log.jsonl`, 254 logged rows →
252 marginal points, 2 process segments):

| segment | n | median s/step | worst excess |
|---|---:|---:|---|
| era-2 **STEADY** (n > 900) | 110 | 26.3594 | **+2.589 %** over its own median |
| era-2 **WARM-UP** (n ≤ 900) | 17 | 27.1252 | **+3.229 %** over the steady median |

⛔ **The post-resume warm-up is +3.2 % for 17 CONSECUTIVE rows**, so a *persistence* rule does not
exclude it — only knowing where the process restarted does.

⚠️ **RETRACTED, mid-task, by my own test:** I first wrote that **+3 % is a rejects-everything
guard**. It is not. The precise statement:

* **+5 %** → `OK` **with or without** the warm-up exclusion (2.41 pp margin over steady, 1.77 pp
  over warm-up). At this tolerance the exclusion buys robustness; it is **not load-bearing**.
* **+3 %** → `OK` on steady state but **`TRIP` on the warm-up** ⇒ **without the exclusion it fires
  on every resume**; with it, on **0.41 pp** of margin.
* ⇒ **The warm-up exclusion becomes load-bearing at any tolerance ≤ +3.23 %.**

⇒ **+5 % on the marginal rate is the defensible setting** — ~1.9× the measured steady worst case.
`step_s` itself is a passes-everything guard (C97) at *every* tolerance.

### 1.5 ⚠️ A second self-correction: the cross-restart difference looks *healthy*

I expected differencing across a process restart to yield an obviously absurd **negative** rate.
MEASURED: it yields **+26.29 s/step** — an entirely plausible healthy number — while the process is
actually running at **40 s/step**. Both terms of the quotient are negative at a reset (the new
process has less elapsed time *and* fewer steps), so the result is *always* a plausible positive.

⇒ **A wrong number that looks wrong is a bug; a wrong number that looks right is the `df`-on-a-pod
family.** The guard therefore **refuses** across a segment boundary rather than estimating, and a
trip run never spans one. Both pinned.

### 1.6 Not-a-result must not read as a clean result

`INSUFFICIENT` is a verdict distinct from `OK`: too few admissible intervals returns *"this run
proves NOTHING about the step time"*. An auto-derived baseline announces that it **cannot see a
uniformly slow run**. Unusable rows are **counted, not dropped** — a shrinking denominator is how a
monitor quietly stops monitoring.

---

## 2. JOB 2 — `pod_git_drift.py` could only see two file types

### 2.1 The filter, widened and made visible

`SUFFIXES` was `(".py", ".sh")` ⇒ it missed **46 of 102** stranded files (45 %), and the published
"45" was a claim about the filter. Now **21 suffixes**, split by what the judgement requires:

* **SOURCE** (someone wrote it): `.py .sh .md .yaml .yml .toml .cfg .ini .patch .diff .sql .bash`
* **ARTIFACT** (a program emitted it): `.json .jsonl .log .txt .csv .tsv .bak .out .err`

⚠️ Artifacts are **not** noise — C110 kept all 17 stranded run logs as raw measurement transcripts —
but they need a different question asked of them (regenerable? superseded?), so they are **counted
separately** rather than swelling one undifferentiated total.

⛔ **The inclusion rule now prints before any count**, every run, and is stamped into `--json` as
`inclusion_rule` with the warning *"THIS IS A FILTERED VIEW, NOT A CENSUS"*. The final total line
reads `TOTAL HOST-ONLY FILES (within the inclusion rule printed above — NOT a census)`.

### 2.2 ⛔ `NAME_ONLY` was the escape hatch — now split

A same-basename hit becomes **`NAME_DRIFT`** (a finding needing the owner's adjudication) when
**all four** hold: the box file is **SOURCE**; **exactly one** repo file carries the basename; it
sits under a root **we author**; and the name is not on a short ubiquitous-name backstop. Everything
else stays the genuinely weak `NAME_ONLY`. `NAME_DRIFT` is never hidden behind `--show-drifted`.

### 2.3 ⚠️ The over-correction was real — measured on Thor, then corrected

| build | files hashed | HOST_ONLY | DRIFTED | NAME_DRIFT | of which artifacts |
|---|---:|---:|---:|---:|---:|
| widened, first cut | 2,982 | 644 | 98 | **478** | **304 (63.6 %)** |
| + `xargs -0` + source-only promotion | 3,397 | 644 | 114 | **174** | **0** |
| + desktop app-data exclusions (final) | 3,381 | **629** | **114** | **174** | **0** |

Two additional guards kept the widening honest: a **symmetric `MAX_BYTES` cap** (an asymmetric cap
invents `HOST_ONLY` rows out of the tool's own inconsistency), and **CRLF normalisation extended to
the new types** — the 94 %-false-positive problem scales with the suffix set, and docs and JSON are
line-ending-bearing too. The final run reports **782 of 1,070 (73.1 %)** of what a naive comparison
would have called drift as artifacts of that comparison.

⚠️ **Desktop app data:** Thor is a workstation. 9 Thunderbird profile files were being reported as
stranded deliverables; `/snap/`, `/.thunderbird/`, `/.mozilla/`, `/.config/`, `/.local/share/`,
`/.jupyter/`, `/.vscode`, `/.gnupg/`, `/.dbus/` are now excluded on both sides.

⚠️ **A concentration line was added** to every by-directory block:
`HOST_ONLY by directory (629 total, 45% in the top 5 — a vendored or generated tree is ONE fact,
not 285)`. **629 rows reads as a crisis; 629 with its concentration reads as three vendored trees
and a results dump**, which is what it is. The two largest blocks are
`/home/nvidia/nurec-gsplat/results/*` (205, generated) and `/home/nvidia/alpasim/src/*` (~100,
a third-party clone). ⚠️ **Neither is excluded by fiat** — which side is canonical is the owner's
call, and the grouping surfaces it rather than pre-empting it.

### 2.4 ⚠️ C111 in a new costume — the widened filter sweeps in the exhaust

A `.txt`/`.log` filter reaches exactly the file class that carried a **live HF token** to a commit
yesterday. The tool prints paths and digests only — it cannot leak — but it now **flags**
sensitive-looking paths (`Keys.txt`, `.pem`, `id_ed25519`, `*token*`, `.env`, …) as
*"READ BEFORE PULLING, do not bank blind"*. The invariant we hold protects `Keys.txt`; it says
nothing about the exhaust.

### 2.5 Other repairs made while running it

* **An incomplete scan is loud, and its two causes are never merged.** The pre-cap count is computed
  pod-side and emitted as an opaque `ZZ<n>ZZ` marker — **disjoint from the searched token**, per the
  echoed-command trap in `CLAUDE.md` (`ZZ%sZZ` in the command text cannot match `ZZ\d+ZZ`).
* **`-size -{N}c` instead of `-size -2M`.** The latter rounds up to whole MiB and actually caps at
  1 MiB, disagreeing with the repo-side byte cap.
* Exit code now reflects **all** actionable verdicts, not `HOST_ONLY` alone.

---

## 3. Tests

| file | n | what is pinned |
|---|---:|---|
| `stack/tests/test_step_time_guard.py` | 19 | both directions of the trip logic; the C112 arithmetic; exact recovery from banked logs; refusal across restarts; `INSUFFICIENT ≠ OK`; and **the trainer wiring by AST** (the field exists, `step_s` keeps its cumulative definition, the interval window is advanced inside the emission block) |
| `stack/tests/test_pod_git_drift.py` | 22 (12 → 22, **+10 new**) | the widened filter sees `.md`/`.json`/`.log`; source/artifact split; `NAME_DRIFT` for `thor_profile.py`, `NAME_ONLY` for ambiguous, vendored and **artifact** collisions; CRLF on new types; symmetric size cap; inclusion rule as data; sensitive-path flag |

⚠️ One pre-existing assertion was **deliberately inverted**:
`test_repo_index_picks_up_sources_and_skips_noise` asserted `"notes.md" not in idx["by_name"]` — it
pinned the very filter that caused the 45 % miss.

⚠️ The real training loop needs the parity corpus, which the dev box does not hold, so the trainer
emission is pinned **structurally (AST)** rather than left unverified. Stated as a limitation, not
papered over.

---

## 4. The live run was not disturbed

* Default build **unchanged**: **87,893,449 params / 405 state_dict keys** (re-measured, not
  inherited).
* `--stage S-W --dry-run` completes; X3 isolation `pass=True`.
* `pytest tests/test_v6_staged.py tests/test_v6_agent_slots.py tests/test_v6_gstr_port.py
  tests/test_pod_git_drift.py` → **177 passed**; my two files → **41 passed**.
* ⚠️ **The full 223-file suite was launched but is NOT reported here as a gate.** MEASURED while it
  ran: **four other heavy python processes** (5.5 GB / 2.7 GB / 1.4 GB / 0.5 GB resident) belonging
  to concurrent agents. `CLAUDE.md` is explicit that gating on a suite under multi-process CPU
  contention is invalid — *it produced 22 spurious failures from contention alone.* ⇒ The targeted
  suites above are the evidence offered; **a green full suite under contention would have been the
  weaker claim, not the stronger one.**
* Thor PID 25477 `kill -0` **alive** at every check, and **advanced across the scans**:
  step **13250 → 13300** (02:40:18 → 02:55:39 UTC). ⚠️ `step_s` reads **26.4694 at both ends** —
  which is not a stall, it is the converging cumulative mean this whole task is about, and it is a
  live demonstration of why it cannot serve as a monitor.
* ⚠️ **Nothing was shipped to Thor.** The trainer edit reaches the box only at the S-T boundary via
  `stack/scripts/launch_closure_audit.py --ship`; the running process already holds its bytecode.
* The drift scans are read-only (`find` + `sha256sum` over `ssh -n`); nothing was written on Thor.

---

## 5. ⛔ Escalations — these need a decision, not a README

### 5.0 ⛔ COMMIT-BLOCKING — the staged `train_v6_staged.py` carries a SIBLING AGENT'S work, and the index as it stands does not import

A concurrent agent's **T2 contrastive + T5 temporal-consistency** implementation (**+626 lines**)
landed in `stack/scripts/train_v6_staged.py` **during** this task. `git add` stages whole files, so
my ~40-line step-time change and their 626 lines are **one blob and cannot be separated in the
index**. This is the `CLAUDE.md` concurrent-staging hazard, and it has a hard consequence:

| fact | value |
|---|---|
| staged `train_v6_staged.py:117` imports | `T2_AUGMENTATIONS, T2_MANOEUVRE_PRESERVING, T2_MANOEUVRE_REVERSING` from `tanitad.models.v6` |
| occurrences in **HEAD**'s `stack/tanitad/models/v6.py` | **0** |
| occurrences in the **worktree**'s `v6.py` | **4** — but it is **NOT STAGED** (` M`) |

⇒ ⛔ **Committing the index as it stands produces an `ImportError` on every import of the trainer.**

**The fix is one line for whoever commits — stage the sibling's dependency in the SAME commit:**

```
git add -- stack/tanitad/models/v6.py \
           stack/tests/test_v6_t2_contrastive.py stack/tests/test_v6_t5_consistency.py
```

⚠️ **I did not stage them myself**: they are another agent's in-progress deliverables, their two test
files are still untracked, and choosing when their work lands is not my call. **Unstaging
`train_v6_staged.py` was also rejected** — it would strand my own change, which is the failure the
operating standard exists to prevent. Naming it precisely is the correct third option.

⚠️ **Also note HEAD moved under this task**: the brief's base was `6784455`; HEAD is now `6001f45`.

### 5.1 Instrument backlog

1. **`NAME_DRIFT` has 174 open rows on Thor.** This is the class C110 says matters and it has never
   been adjudicated. **A15 (`thor_profile.py`) is one of them and is still not acted on** — which
   side is canonical belongs to that package's owner.
2. **629 `HOST_ONLY` files (341 source / 288 artifact).** ~45 % sit in two trees that are plausibly
   generated (`nurec-gsplat/results`) or third-party (`alpasim/src`). **A decision is needed on
   whether those roots are in scope**, otherwise the nightly checker reports the same 300 rows
   forever and stops being read — which is how the old one was allowed to stay narrow.
3. **The `step_time_guard` is built and demonstrated but is NOT called by anything yet.** It is an
   operator/monitor-side criterion, and that is the correct home — but per C109's own warning, a
   guard that is never invoked is the `pod_git_drift` failure mode in advance. **It should be wired
   into the next concurrency pilot's watchdog and into `supervise_run.sh`'s health check.**
4. **Every abort criterion written before today should be re-read against §1.3.** C112 notes this
   was the *third* gate that week unable to return the answer it existed to give (E4 could not
   report PASS, SEL-1 could not report FUNDED, this could not report ABORT).

---

## 6. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `step_s_interval` + `steps_this_process` emission | `repo:stack/scripts/train_v6_staged.py` (staged) | no |
| Step-time guard + CLI | `repo:stack/scripts/step_time_guard.py` (staged) | no |
| Guard tests (19) | `repo:stack/tests/test_step_time_guard.py` (staged) | no |
| Widened drift detector | `repo:stack/scripts/pod_git_drift.py` (staged) | no |
| Drift tests (22) | `repo:stack/tests/test_pod_git_drift.py` (staged) | no |
| This report | `repo:TanitAD Research Hub/…/incoming/2026-08-18-monitor-fixes/MONITOR_FIXES.md` | no |
| Final Thor drift scan (raw) | `repo:…/2026-08-18-monitor-fixes/drift_thor_after.json` | no |

**Nothing produced by this task lives in only one place.** No commit, no push, no branch switch.
