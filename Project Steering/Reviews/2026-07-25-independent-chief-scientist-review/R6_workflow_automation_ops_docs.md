# R6 — Workflow, Automation, Speed-of-Work & Results-Documentation Review

**Reviewer role:** Independent research-operations / MLOps consultant (read-only).
**Date:** 2026-07-25. **Scope:** where the TanitAD program loses *time* and *reliability*, and the
specific automation that would buy it back. **Method:** read-only pass over `CLAUDE.md`,
`AGENT_OPERATING_STANDARD.md`, `LOOP_STATE.md`, `RETRACTION_LOG.md`, `MODEL_REGISTRY.md`,
`GATE_PROTOCOL.md`, `Project Steering/Reports/`, the Research Hub tree, `tools/`, `stack/scripts/`,
`taniteval/`, `.claude/`, and `git log`. No pods, no compute, no `git add`.

**Evidence classes** (per program standard): `MEASURED` = I read it in the repo this pass, artifact
named. `INHERITED` = quoted from a program doc (RETRACTION_LOG / LOOP_STATE) I did not independently
re-verify. Cost figures are `ESTIMATED` unless a doc states them.

---

## 1. Executive verdict

**This is a guardrails-rich, automation-poor operation.** The program's *scientific* discipline is
genuinely top-decile: a code-enforced gate protocol (`run_gate.py` refuses to decide off a train-log
slope), a decision-grade CI estimator (`taniteval/ci.py`), a `ci_gate.py` suite floor, a
`session_guard.py` stranded-work blocker, `pod_git_drift.py`, and a `RETRACTION_LOG.md` that is a real
self-learning instrument. Very few research programs have this.

But almost every **high-frequency operational surface** is still manual, and each one is
demonstrably lossy:

- The **source-of-truth registry is hand-transcribed from JSON** and drifts — a retracted "best-in-program"
  headline sat in the `MODEL_REGISTRY` header for **4 days after its own body refuted it**
  (`RETRACTION_LOG` 07-25); a derived `LEADERBOARD.md` was *more* correct than the source it derives from.
- The **commit path itself corrupts** — an intermittent (~50%) segfault on pathspec commits cost **4
  successive `CLAUDE.md` rewrites** in one day, and the whole-index default has **twice** swept a sibling
  agent's half-finished code into the wrong commit.
- **Finishing is manual** — the program's own audit calls "good work stranded outside git" its
  *dominant failure mode*; instruments have idled **10–12 days**, and false "stranded" alarms now cost
  investigations too.
- The **live state file the drumbeat reads every ~30 min is a 122 KB run-on** with 19 inline
  "superseded/retired/retracted" fragments; it has **shipped stale instructions twice in one day** (its
  own header says so).
- **Multi-GB moves crawl at ~1 MB/s** with no automated relay; the HF fast-path is repeatedly 403-blocked.
- The loop loses agents to **`ENOTFOUND` disconnects** and once exhausted a weekly budget on a **106-subagent
  burst**.

The throughput cost is not the headline experiments — those are well run — it is the **tax on every
commit, every registry edit, every state-file read, every multi-GB move, and every intake promotion**,
plus a recurring **re-derivation / retraction / rescue** load that a handful of small tools would largely
eliminate. **Grade: B-** (§3). The five cheapest fixes (§4 build-order P1–P5) are all **S/M effort** and
each removes a *recurring, sometimes decision-grade* loss.

---

## 2. Ranked time-sink / reliability-loss table

Ranked by estimated cumulative cost. "Cost" blends wall-clock lost, agent-sessions spent
re-doing/rescuing/re-deriving, and *decision risk* (a wrong number nearly deciding a GPU-day).

| # | Time-sink / reliability loss | Est. cost | Evidence (path / artifact) | Class |
|---|---|---|---|---|
| **1** | **Stranding — "good work outside git."** REF-B v2 arch, the entire TanitEval harness, the pod-ops bundle, TanitResim (486 lines), an atomic-archive bug-fix, an orthogonality instrument (**10 days idle**), LAL-v2 (**believed** unmerged 12 days), `v2_compressed.py` builder (repo-absent, pod-only). | **Tens of agent-days** of latent loss + several real multi-day integration delays. The program calls this its **dominant** failure mode. | `AGENT_OPERATING_STANDARD.md` L9–17 table; `RETRACTION_LOG` LAL-v2 + `v2_compressed` entries; `git 9c69525` "rescue the stranded 486-line strip" | INHERITED + MEASURED |
| **2** | **Stale-doc / retraction propagation in the *source of truth*.** v1.6 "⭐ best ADE" stood in the `MODEL_REGISTRY §1.4b` **header 4 days** after a 07-21 retraction fixed only the body; REF-C strata "printed 16k numbers under a FINAL header"; REF-C selection quoted single-clip as corpus; f1b378 "episode-disjoint" was 78.5%-leaked. Each needed a full agent re-derivation to catch. | **Multiple agent-sessions** to detect+fix; ≥1 "nearly designed the hierarchy away"; a paired bootstrap (B=2000) re-run just to retire one stale headline. | `RETRACTION_LOG` 07-25 C4 header entry; `git e02a78d, 40a20d2, bb3f6a7, 23e4b41` | MEASURED (commits) / INHERITED (cost) |
| **3** | **Measurement re-derivation churn.** The *same* numbers recomputed because a doc/metric went stale: the learning-curve exponent re-fit across **5 windows**; s/step "6.37 vs 10.89 UNRESOLVED for a day"; CI re-computed after the "jackknife" was found 1.28–2.06× too narrow (`CI_RECOMPUTE_2026-07-20.json`, `recompute_ci.py`). | **Repeated agent-hours** re-deriving decision-grade numbers that should be computed once and stamped. | `CLAUDE.md` §exponent/§interval; `GATE_PROTOCOL.md §3`; `RETRACTION_LOG` s/step entry; `taniteval/recompute_ci.py` | MEASURED |
| **4** | **Commit-path corruption + tax.** Intermittent ~50% segfault on pathspec commits → **4 `CLAUDE.md` rewrites** (`e405cd4→2c44ae6→e26e2a7→bdb6ba1`); every crash leaves a stale `index.lock` (phantom "another git process"); the whole-index default **swept sibling work twice** (`60265d3`, `3d41bd0`). Every multi-file commit is now a manual index audit + retry ritual. | **~4 doc rewrites + a per-commit manual tax**, on the busiest surface in the repo (45 commits on 07-20 alone). | `CLAUDE.md` §Git-hygiene; `RETRACTION_LOG` 07-25 C8; commit chain above | MEASURED |
| **5** | **Closed-loop "closure" whipsaw + n=1 headlines.** **5** over-claims of closure in one session, each reopened by a cheap follow-up (canary / planner-headroom / renderer / closed / BOUND-horizon); n=1 REF-C & flagship-v1 closed-loop headlines reached chat/reports before n=12/n=40 reversed them. | Mostly caught same-session (low $) but **high reporting-noise**; each reversal = a re-run + a re-report, some reaching Sayed. | `RETRACTION_LOG` 07-22→07-25 meta-pattern (5 entries) | INHERITED |
| **6** | **~1 MB/s dev-box relay for multi-GB moves + HF-403.** The 22 GB v2-corpus pod3→pod1 move ran at ~1.38 MB/s (00:03Z→02:12Z); the HF fast-path (~118 MB/s) is repeatedly **403 storage-full**; pods can't SSH each other. | **Hours per multi-GB move, recurring**; blocked the formal 8-metric gate + a REF-C arm + ckpt-backup simultaneously. | `LOOP_STATE.md` L58; `CLAUDE.md` trap; `LOOP_STATE.md` L22 HF-403 blocker | MEASURED |
| **7** | **Agent deaths on disconnect + async-completion overclaim.** An `ENOTFOUND` "killed **3 agents mid-report**" (07-25, work recovered); the HF uploader (a child of the Claude session) **died committing 0/17 files** on a restart → repo created but empty, reported as "PUSHED". | Recovered but **re-reported / relaunched detached**; the empty-push is a reliability trap on every async action. | `LOOP_STATE.md` L7; `RETRACTION_LOG` 07-22 TanitDataSet-C entry | MEASURED |
| **8** | **Worktree sprawl.** **39** worktree dirs / **426 MB** under `.claude/worktrees/` back to 07-10; `git worktree list` reports **51** (≥12 orphaned registrations). Directly caused the C8 **Glob-truncation false-stranding** (Glob is mtime-sorted + capped at 100 → main-tree files fell off the list → two false "stranded on worktree" claims). | A false-stranding **investigation** + a Glob-unreliability tax on *every* search + 426 MB on the Drive. | `.claude/worktrees/` (this pass); `RETRACTION_LOG` 07-24 C8 | MEASURED |
| **9** | **`incoming/` never cleared.** **~90** intake bundles across 6 areas, oldest `2026-07-08`; the orthogonality instrument (`2026-07-10-orthogonality-instrument`) is *still* in `incoming/`. No promotion-out step. Includes a **future-dated** `2026-07-31-stationary-lead-scenario` (narrative-clock artifact). | The "escalation lost in a dir nobody re-reads" failure mode (the 10-day idle) + steady navigability rot. | `find … -name incoming` (this pass) | MEASURED |
| **10** | **Live-source churn (YouTube-IDM).** The pipeline was iterated against a rate-limited live source (single→×3→GeoCalib→3 smokes→65-clip) → **hard anti-bot block**, the decision-grade scale-up verdict **not delivered**, ~30 h cooldown. | pod3 time + the verdict lost; correctly **not** worked around (bot-bypass out of bounds). | `RETRACTION_LOG` 07-25 new-class entry; `LOOP_STATE.md` L50 | MEASURED |

**Cross-cutting pattern:** items **1, 2, 8, 9** are one disease — *no automated "is this finished / current /
promoted?" check* — and items **4, 6, 7, 10** are another — *no automated wrapper around the risky
mechanical action* (commit, transfer, async-launch, live-fetch). Both are cheap to close.

---

## 3. Ops-maturity grade: **B-**

| Dimension | Grade | Why |
|---|---|---|
| Scientific / measurement ops | **A-** | Code-enforced gate protocol, decision-grade bootstrap CI, evidence-class discipline, an append-only root-cause retraction log. Genuinely rare. |
| Guardrail *authoring* | **A-** | `ci_gate.py` (suite-floor), `gpu_tripwire.py`, `session_guard.py`, `pod_git_drift.py`, 3 skills. The program writes excellent guardrails. |
| Guardrail *coverage of the hot path* | **C** | The highest-frequency surfaces — commit, registry edit, state-file, transfer, intake promotion — have **no** automation and are demonstrably lossy. |
| Results documentation | **C+** | 836 eval JSONs exist and a pod-side `taniteval` dashboard generator exists — but the *quotable* registry/leaderboard is hand-typed over them and drifts. The plumbing exists on one side and is manual on the other. |
| Loop / orchestration robustness | **C+** | Cron "drumbeat" + skills work, but ScheduleWakeup re-arms "silently break", a 106-agent burst crashed a session, and disconnects kill in-flight agents. |

**Net B-.** The signature is *"excellent at documenting a trap after it bites, weak at removing the
trap."* The RETRACTION_LOG is full of lessons that a linter/wrapper could have *enforced* instead of
*taught*. Moving hot-path guardrails from **C** to **B+** is what lifts the program to a solid **B+/A-**
and is almost entirely **S/M** effort.

---

## 4. Automation proposals (primary deliverable) — prioritized build-order

Each proposal: **what it automates · mechanism · payoff · effort (S/M/L)**. Ordered by
payoff ÷ effort. P1–P5 are the recommended first sprint (all S/M, each kills a *recurring* loss).
Everything is stdlib-first and lives in `tools/` (the existing agent-facing tooling home, no intake
round-trip), consistent with `tools/README.md`.

### P1 — `tools/safe_commit.py` (safe-commit wrapper) — **effort S** — *build first*
- **Automates:** time-sink **#4** (segfault + stale-lock + concurrent-index sweep) and the `Keys.txt`
  invariant.
- **Mechanism:** a single entry point the orchestrator uses instead of raw `git commit`. It (1) runs
  `git diff --cached --name-only` and **prints the staged set + aborts if it spans >1 agent area** unless
  `--allow-foreign` (with the foreign paths named in the message, per `CLAUDE.md`); (2) **hard-refuses if
  `Keys.txt` is staged**; (3) commits pathspec-free with `-F <msgfile>` (the never-crashed path); (4) on
  exit **139 / 0xC0000005**, auto-`rm -f .git/index.lock` (after confirming no live git proc) and
  **retries up to 3×** — the procedure `CLAUDE.md` now prescribes by hand; (5) prints the resulting
  `git show --stat`.
- **Payoff:** removes a recurring per-commit manual ritual on the repo's busiest surface, and makes the
  twice-seen sibling-sweep **structurally impossible**. Prevents the next 4-rewrite doc thrash.
- **Effort S** (~150 lines + a `tools/tests/` falsifier driving a throwaway repo, matching the existing
  test style).

### P2 — `tools/registry_lint.py` (source-of-truth drift gate) — **effort S/M** — *build second*
- **Automates:** time-sink **#2** (stale headline / registry-vs-JSON drift) — the discipline the entire
  program rests on.
- **Mechanism:** two checks, wired into `ci_gate.py` and the nightly job. **(a) Numeric drift:** every
  decision-grade number in `MODEL_REGISTRY.md` carries a machine-checkable source pointer
  (`<!-- src: taniteval/results/driving_flagship-v4.1-10k.json#ade_0_2s tol=1e-3 -->`); the linter
  re-reads the JSON and **fails on mismatch**. **(b) Retracted-claim survival:** parse the "retracted
  claim" column of `RETRACTION_LOG.md`, then **multiline-scan** the registry (esp. section *headers*) for
  those phrases — the exact two failure modes of the 07-25 C4 entry ("header not re-read" + "wrapped
  across a newline evaded line-based grep"). Emits `WARN` with the retraction date.
- **Payoff:** the v1.6-4-day-header and "16k-under-FINAL" drift classes become **impossible to merge**;
  the source of truth stops being out-disciplined by its own derived docs.
- **Effort S/M** (parsing + a tolerance compare; the pointer-annotation backfill is the only M part and
  can be incremental — lint only annotated rows first).

### P3 — `tools/repo_janitor.py` (worktree + intake janitor) — **effort S** — *build third*
- **Automates:** time-sinks **#8** (worktree sprawl / Glob unreliability) and **#9** (`incoming/` never
  cleared).
- **Mechanism:** nightly. (1) `git worktree prune`, then flag on-disk worktree dirs whose HEAD has **no
  unique commits vs `main`/tip** for deletion (the 39-dir / 426 MB backlog). (2) Walk `*/Implementation/incoming/*`
  and **report bundles older than N days with no matching promoted path** (a "still-in-incoming" ledger) —
  turning the silent 10-day-idle failure into a dated list. (3) Flag **future-dated** dirs (the `2026-07-31`
  narrative-clock artifact). Read-only by default; `--apply` for the prune.
- **Payoff:** restores **Glob reliability** (kills the mtime-truncation false-stranding class at the
  root), reclaims disk, and makes unpromoted intake *visible* instead of discovered in an audit.
- **Effort S.**

### P4 — `tools/ckpt_relay.py` + provenance stamper — **effort M** — *build fourth*
- **Automates:** time-sinks **#6** (1 MB/s relay) and **#7** (async-completion overclaim), plus the
  `config.json cache_dirs:null` provenance bug flagged in `LOOP_STATE`.
- **Mechanism:** a push/pull pair run **on the pods**. `push`: md5 → HF at ~118 MB/s → write a **terminal
  sentinel** `UPLOAD_COMPLETE {n}/{n} md5=<...>` **only after** the final commit; and drop a
  **provenance sidecar** (`corpus_key`, `parity_hash`, `git_commit`, `step`, `param_count`) next to every
  archived ckpt so provenance never lives only in a log line. `pull`: fetch → **md5-verify** → fail loud on
  mismatch. Ship a `wait_for_marker(path, token)` helper the drumbeat uses so completion is confirmed by
  the **terminal marker, never the launch log** (the exact C1 async sub-pattern).
- **Payoff:** collapses multi-GB moves from hours to minutes when HF space allows, and eliminates the
  "launched ≠ completed" overclaim class *and* the empty-repo push. (Orthogonal Sayed-side unblock: the
  HF-storage cleanup — noted, not automatable here.)
- **Effort M.**

### P5 — `tools/results_ledger.py` (auto-generated results doc) — **effort M** — *build fifth; the results-documentation centerpiece*
- **Automates:** **results documentation** — how results get recorded, versioned, and surfaced (time-sink
  **#2** at the root, and the manual-transcription drift that feeds it).
- **Mechanism:** walk `taniteval/results/*.json` + `Project Steering/Gates/*.json`, extract a normalized
  row per (arm, step, `ade_0_2s`, `miss@2m`, oracle, CI + **estimator name**, split, harness commit,
  artifact path), and **regenerate** (a) a single sorted `Benchmarks & Eval/RESULTS.md` leaderboard and
  (b) a machine `results.jsonl`. The `MODEL_REGISTRY` then **cites rows by JSON pointer** (feeding P2)
  instead of hand-copying. Runs on every eval-land (hook) + nightly; the pod-side `taniteval` dashboard
  (`nightly.sh` → `report.build()`) becomes its upstream feed rather than a disconnected island.
- **Payoff:** the leaderboard/registry **can no longer drift from JSON — it is generated**; new results
  auto-surface; the "derived doc out-disciplined the source" inversion is fixed structurally. Also enforces
  "never quote an interval without its estimator" at generation time.
- **Effort M.**

---

### Second sprint (P6–P10) — high value, slightly larger or lower-frequency

### P6 — `tools/gate_status.py` (hypothesis / pre-registration tracker) — **effort M**
- **Automates:** the H-series / pre-registered-gate state now scattered across LOOP_STATE, registry and
  `Gates/`.
- **Mechanism:** join each `Gates/*.card.json` (which already holds the pre-registered threshold + both
  outcomes + restart budget) to its verdict JSON → a generated `GATES.md`:
  `registered / NOT_YET / PASS / FAIL / REFUTE_LEVER_FAMILY`, restarts-used, GPU-hours. A **schema check**
  on the card would have caught the 07-25 "the 30k gate would have produced **NO VERDICT**" bug
  (`git 3ff5499`) at registration, not at gate time.
- **Payoff:** pre-registration becomes visible and auto-surfaced; the "both outcomes committed in advance"
  rule gets a home; dead/duplicated gate cards surface.
- **Effort M.**

### P7 — `tools/loop_state_lint.py` + structural split — **effort S**
- **Automates:** the LOOP_STATE liability (time-sink adjacent to **#5/#7**): a 122 KB run-on the drumbeat
  re-reads every ~30 min, with 19 inline superseded/retired fragments; it has "shipped stale instructions
  twice in one day" (its own L3–5).
- **Mechanism:** split into **`LOOP_STATE.md`** (live only: `STANDING DIRECTIVES` + `FLEET` + `ACTIVE
  STREAMS`, hard-capped ~150 lines) and **`LOOP_STATE_ARCHIVE.md`** (append superseded/retired/retracted).
  The linter **fails CI if the live file exceeds the cap or contains a `superseded/RETIRED/RETRACTED`
  block**, and warns if `LAST_UPDATED` is >X h old. Collapse the run-on `LAST_UPDATED` line into a dated
  bullet list.
- **Payoff:** the drumbeat parses a clean, current file → fewer stale-instruction shipments and faster
  per-iteration reads. Directly addresses "is LOOP_STATE an asset or a liability" — it's an asset **only if
  bounded**.
- **Effort S.**

### P8 — Report-cadence consolidation + auto-compile — **effort S**
- **Automates:** report-series sprawl. **MEASURED:** three overlapping series, two **abandoned** —
  `Daily Reports/` last `2026-07-20`, `Progress Reports/` last `2026-W33` (07-20) — and the surviving 3×/day
  `Reports/` has a **gap 07-15 → 07-23**. The 07-24 report itself says "drumbeat firing faster than events".
- **Mechanism:** retire the two dead series; make the existing `program-report` skill the single channel;
  **auto-populate its headline** from P5 (results ledger) + `fleet-status` skill + P6 (gate status) so a
  report is *assembled from generated MEASURED facts*, not re-typed. Move filed cadence to **1×/day +
  event-triggered** (on a gate verdict / arm completion), keeping the lightweight chat drumbeat.
- **Payoff:** less reporting overhead for the same signal, no abandoned channels, and reports inherit
  provenance automatically (kills the C1 "trainer-log-in-a-report" class at the source).
- **Effort S.**

### P9 — `tools/new_brief.py` + submit-time brief/report linter — **effort M**
- **Automates:** the `AGENT_OPERATING_STANDARD` preamble (must be pasted **verbatim** into every brief),
  the priority-order requirement, and the deliverable-manifest — the three rules whose violation *is* the
  stranding failure mode.
- **Mechanism:** `new_brief.py <area>` emits a brief scaffold pre-filled with the preamble, the traps
  block, the parity invariants, a **priority-order** stub, and an empty **deliverable-manifest** table. A
  companion `report_lint.py` (or a `Stop`-hook) **refuses an agent report that lacks a manifest or leaves a
  number without an evidence-class tag**. Pairs with `session_guard.py`, which already blocks on
  uncommitted deliverables.
- **Payoff:** makes the contract structural rather than copy-paste; removes preamble drift; catches "an
  artifact in only ONE place" at report time instead of an audit months later.
- **Effort M.**

### P10 — Drumbeat resilience: cron self-heal + fan-out governor + completion watchdog — **effort M/L**
- **Automates:** loop fragility (time-sinks **#5/#7/#10** at the machinery level): ScheduleWakeup re-arms
  "silently break", the **106-subagent** burst that exhausted the weekly budget, and `ENOTFOUND` agent
  deaths.
- **Mechanism:** (a) a **cron heartbeat** that re-verifies the schedule each fire and re-creates it before
  the 7-day expiry (the memory-noted failure); (b) a **fan-out governor** — a semaphore file capping
  concurrent subagents (e.g. ≤8) the orchestrator checks before every spawn, so "STAGGER never burst"
  is enforced not remembered; (c) agents **bank to `incoming/` incrementally + write a terminal marker**,
  and a watchdog **re-attaches by polling the marker** rather than trusting the notification (the measured
  monitor-delivery-gap lesson) — so a disconnect loses ≤1 step, not a whole agent.
- **Payoff:** the loop stops losing agents to disconnects and *cannot* repeat the budget-exhausting burst;
  turns "monitor looks dead" (a repeat false alarm) into a determinate marker check.
- **Effort M/L** (partly depends on the harness's scheduling primitives).

---

## 5. Results-documentation — how results should get recorded, versioned, surfaced

Consolidated because the task calls it out specifically. Today: **raw JSON (good, 836 files) → hand-typed
registry prose (drifts) → hand-typed reports (drift again, C1)**. Each `→` is a manual transcription and
each is a documented drift source.

Target pipeline (built by P5 + P2 + P6 + P8):

```
 eval_*.py / run_gate.py  ──►  taniteval/results/*.json + Gates/*.json   (raw, immutable, MEASURED)
                                      │
                        P5 results_ledger.py (generated, on eval-land + nightly)
                                      ▼
                RESULTS.md  +  results.jsonl  +  GATES.md   (single generated surfaces)
                                      │  cited by JSON-pointer
                        P2 registry_lint.py  ── fails CI on drift / retracted-headline survival
                                      ▼
                     MODEL_REGISTRY.md (quotable; pointers, not copies)
                                      │
                        P8 program-report (headline auto-filled from the above)
```

**Versioning:** raw JSON is already immutable per run; add a one-line **provenance sidecar** (P4) so every
number is traceable to `git_commit + corpus_key + step`. **Surfacing:** the generated `RESULTS.md` /
`GATES.md` replace the hand-maintained leaderboard; the registry keeps prose *interpretation* but its
*numbers* are generated. This is the single change that most raises the results-documentation grade from
**C+** and retires time-sink **#2** at the root.

---

## 6. What is already good (keep, don't rebuild)

`run_gate.py` (code-enforced gate) · `taniteval/ci.py` (episode-cluster bootstrap) · `ci_gate.py`
(suite-floor + tripwire) · `session_guard.py` (stranded-work blocker — extend it with P2/P3, don't
replace) · `pod_git_drift.py` · the `fleet-status` / `gate-eval` / `program-report` skills · the
`RETRACTION_LOG` root-cause-class practice. The proposals above **feed these**, they don't compete with
them — e.g. P5 feeds P8's skill; P2/P3 extend `session_guard`/`ci_gate`; P4 uses the existing HF path.

---

## Appendix — first-sprint build-order (P1–P5), one line each

1. **P1 `safe_commit.py`** (S) — kills the segfault + sibling-sweep on the busiest surface.
2. **P2 `registry_lint.py`** (S/M) — the source of truth can no longer drift from JSON or carry a retracted headline.
3. **P3 `repo_janitor.py`** (S) — prune 39 worktrees / clear `incoming/`; restores Glob reliability.
4. **P4 `ckpt_relay.py` + provenance** (M) — minutes not hours per move; kills async-completion overclaims.
5. **P5 `results_ledger.py`** (M) — generated RESULTS.md/results.jsonl; registry cites, never copies.

*All read-only findings; nothing staged or committed (concurrent `git add` corrupts the index here).* 
