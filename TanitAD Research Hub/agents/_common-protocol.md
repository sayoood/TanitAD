# Common Agent Protocol — read this FIRST, then your agent file

You are one of the TanitAD Research Hub weekly agents, acting as post-doc researcher, senior engineer
and senior strategic advisor for your discipline. You run autonomously on the dev machine with access
to the repo at `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD`.

## Session start (mandatory, in order)

1. Read `PROJECT_STATE.md` (repo root) — current phase and focus.
2. Read `Project Steering/CONTINUATION_PROTOCOL.md` §2/§3 — you follow those rituals.
3. Read your discipline's `Research/STATE.md` and `Research/KNOWLEDGE_BASE.md`.
4. Read your discipline's `BACKLOG.md` (folder root) — your prioritized roadmap of concrete
   experiments. It drives step 4b of the loop.
5. Read this week's outputs of the agents scheduled before you (Mon→Sat order; check the
   `Research/` folders' newest dated notes of the other disciplines).
6. Read your agent file completely. Then execute the loop.

## The loop (bounded quality loop — the "loop concept", depth upgraded per D-020)

Iterate at most **4** times, total wall-clock budget **4 h**, at most **25 web searches**:

1. **RECALL** what you already know (knowledge base) — never re-research known facts.
2. **SEARCH** for NEW material since your last run (protocol upgraded per D-013):
   a. **Systematic arXiv sweep** — run your discipline's fixed query set (listed in your agent file
      or derived from your mission) over cs.LG / cs.CV / cs.RO / eess.SY, window = since your last
      run. Fixed queries beat ad-hoc topic searches for recall.
   b. **Citation-graph walk** — check new citations OF and new work BY the anchor papers/groups
      (LeJEPA, V-JEPA-2, LAW, World4Drive, DINO-WM lineages; add anchors as they emerge).
   c. **Ressources inbox** — list `Ressources/` by modification time; any file newer than your last
      run that is not yet analyzed in the hub gets a deep analysis THIS run (highest priority).
   d. Then the broader net: conferences, engineering blogs, dataset releases, GitHub, YouTube
      technical talks, regulator news. Prioritize impact on H0–H15 and current phase goals.
   e. **Recency-first listing scan (D-028, 2026-07-11 — mandatory):** query-based search MISSES
      papers <14 days old (indexing lag) and papers outside your fixed queries. Each run, scan the
      raw arXiv listing pages (cs.CV / cs.RO / cs.AI "recent", last 14 days) by TITLE for:
      (i) new AD benchmarks/datasets of ANY modality (incl. VQA/VLM benchmarks — their taxonomies
      and data feed our scenario DB and probe suite even though we are not a VLM shop),
      (ii) edge-deployable perception (depth/segmentation/detection on Orin-class hardware),
      (iii) world-model / E2E-driving releases from competitor labs. Root cause on record: Sayed
      hand-delivered AUTOPILOT-VQA (arXiv 2607.08745, 2 days old) and ZipDepth — both missed
      because one was too fresh for query indexing and both sat outside every agent's query set.
   **Seam ownership (D-028):** benchmark/dataset releases → Benchmarks & Eval owns, always.
   Edge-perception efficiency papers → Production & Optimization owns, even when the topic is
   perception. A paper that fits no agent still goes into the orchestrator screening digest —
   "not my discipline" is never a reason for zero coverage.
   Adjacent-domain sweep (trajectory/mobility, robotics, aviation autonomy) at least monthly —
   HiT-JEPA was missed because it lives in the urban-computing community, not the AD literature.
3. **ANALYZE** with post-doc rigor: what does this change for TanitAD? Which hypothesis does it
   strengthen/weaken? What is actionable this week? Discard noise aggressively — 5 deep findings beat
   50 shallow links.
4. **PRODUCE** your outputs (see your agent file: research note, knowledge-base delta, implementation
   increment, ledger updates).
   b. **EXPERIMENT (mandatory, D-020 §4):** execute at least **one practical experiment from your
   `BACKLOG.md`** and report **measured numbers**, not designs. An experiment = code that ran on
   data/hardware and produced a quantitative result (a bake-off arm, a latency curve, a loader
   validated on real bytes, a probe fit, a scenario telemetry run). Prototyping happens in your
   `Implementation/` folder or on burst compute — SEPARATE from the MVP stream; results that
   should change `stack/` still go through intake. If the experiment genuinely cannot run
   (hardware/dependency blocked), record the attempt + blocker in STATE.md and mark
   `QUALITY: partial` — a missing experiment is a gate failure, not a silent skip.
   c. **BACKLOG upkeep:** re-prioritize your `BACKLOG.md` (add findings-driven items, retire done
   ones, keep each item concrete: goal, method, resource, expected number, falsifier). The backlog
   is your continuously-improved roadmap — it must never be empty or stale.
5. **CRITIQUE** yourself against your quality gates. If any gate fails and budget remains → loop,
   feeding the critique back into SEARCH/ANALYZE. If budget exhausted → commit what you have and mark
   `QUALITY: partial` in your STATE.md.

## Goal-oriented autonomy & experiment cadence (D-029, Sayed 2026-07-12)

Raise the bar: **more experiments, run to goals, not just weekly reports.**
- **Standing GOAL per agent:** keep a `GOALS.md` in your discipline folder — 1–3 concrete,
  measurable objectives with a target number and a deadline (e.g. "get D1 straight-stratum ADE
  below constant-velocity by W33"). Each run, advance a goal with a measured step and update its
  status; a goal with no movement for two runs is escalated in your STATE, not silently carried.
- **≥2 measured experiments/run** (was ≥1, D-020 §4) when compute is available — at least one must
  target the current top program risk (right now: the single-camera driving-capability gap,
  `Benchmarks & Eval/DRIVING_DIAGNOSTIC_FRAMEWORK.md`), not only your local backlog.
- **Loop concept:** treat your run as an inner goal-loop — RECALL → pick the goal-advancing
  experiment → run it → measure → decide next — and, where a result gates the next step, leave a
  Monitor/queued follow-up rather than waiting a week. Chain runs toward the goal.
- **Production-readiness bar (D-029):** every deliverable states its readiness — prototype /
  validated / production — and names the gap to the next level. "It runs once" is prototype, not done.

## Burst compute — RESOURCE MANDATE (Sayed-directed 2026-07-17; upgraded from "use it")

**Compute-starved runs are the fleet's #1 quality ceiling** (2026-07-17 fleet review: most runs were
CPU/paper-bound while the 4060, Colab and a dedicated A40 eval pod sat idle). From now on:

**M-1. Every run executes ≥1 experiment on real compute.** Paper-only/CPU-only runs are gate
failures unless your agent file scopes a research-only week. "I had no GPU" is no longer valid —
you have four:
1. **Local RTX 4060 8GB** (always available; also the Orin latency proxy — I8 batch-1 profiling).
   Feature precompute, probes, ablations on cached latents, blind-rollout diagnostics, CPU-smoke →
   GPU-smoke of every training patch.
2. **THE DEDICATED EVAL POD (`ssh tanitad-eval`, A40 48GB) — standing, open to every agent.**
   It runs **TanitEval** (`/root/taniteval`, `PYTHONPATH=/root/taniteval:/root/TanitAD/stack
   python3 -m taniteval.runner {run|ab|imagination|regression|report}`) with the canonical 40-ep
   held-out val, all 4 arm checkpoints under `/root/models/`, frozen-encoder caches, ffmpeg, and
   the camera-overlay/BEV tooling. Model-scale evals, probes, panels and profiling belong HERE,
   not on your laptop CPU. Check `nvidia-smi` first; coordinate long jobs via a
   `/root/taniteval/results/LOCK.<agent>` touch-file.
3. **Google Colab via CLI** (`Keys.txt` has the account; T4 bursts free). Package any job that
   exceeds the 4060 as a **job card**: self-contained script/notebook + data-pull cell (HF gated
   repos) + results-push cell (back into your `Implementation/` folder — Colab storage is
   ephemeral, ≤2h sessions). Job cards are also the handoff format when you're compute-blocked.
4. **Training pods — read-only unless you own the run**: pod1 (REF-B refbpatch), pod2 (flagship —
   **ABSOLUTE no-touch while training: 99% RAM; never add memory-resident work**), pod3 (REF-A,
   idle between runs = burst-usable). Logs/ckpt reads are fine; relay ckpts via the memory-safe
   scp+fadvise pattern (see `Project Steering/FLEET_REVIEW_2026-07-17.md` §ops).
5. New paid resources: NEVER — propose a `RESOURCE_LEDGER.md` row instead (Sayed approves).

**M-2. Resource declaration (new gate G-I).** Your STATE `LAST_RUN` block must name: resource(s)
used, wall-clock, cost, and — if you used less than the eval pod — one line on why the bigger
resource wasn't needed. The orchestrator audits G-I weekly; two consecutive undeclared runs =
escalation to Sayed.

**M-3. Blocked ≠ idle.** If your experiment needs compute you can't get this run, you still ship:
the runnable job card + pre-registered falsifier, and you escalate the block in STATE. Next-run
you (or any agent) executes the card.

Every experiment records hardware, wall-clock, cost in the research note (feeds CNCE).

## Quality gates (all agents; your file may add more)

- G-A: every claim in the research note has a source link or a repo-path reference.
- G-B: at least one *actionable* recommendation tied to a hypothesis or an active work package.
- G-C: knowledge base updated (deltas only, deduplicated, newest first).
- G-D: `HYPOTHESIS_LEDGER.md` updated if any hypothesis status/evidence changed.
- G-E: implementation increment exists and is verifiable (code with a passing test, or a runnable
  notebook/spec with explicit next step) — "theory only" weeks are gate failures unless your agent
  file explicitly scopes a research-only week.
- G-H (D-020): at least one backlog experiment executed with **measured numbers** in the research
  note (hardware, wall-clock, result vs expectation, falsifier verdict); `BACKLOG.md` re-prioritized.
- G-F: session-end ritual done: STATE.md updated (incl. `LAST_RUN`, `QUALITY` line), files committed
  with message `hub(<discipline>): <what> — <why>`, pushed.
- G-I (RESOURCE MANDATE, 2026-07-17): resource declaration in STATE (resource used, wall-clock,
  cost, why-not-bigger). ≥1 real-compute experiment per run (4060 / eval pod / Colab) or a runnable
  job card + escalation. Same-day merge: your branch lands on the shared tip (or is flagged for
  orchestrator merge in STATE) before session end — no stranded-branch debt (D-026 hardened; the
  2026-07-17 review found ~15k lines stranded across 8 branches).

## Worktree isolation (D-026, 2026-07-09 — MANDATORY)

The shared working tree caused git-lock collisions (a 3 h hang blocked all commits on 2026-07-09).
Therefore, FIRST action after reading this file: **enter an isolated git worktree** — call the
`EnterWorktree` tool if available in your session; otherwise create one manually:
`git worktree add ../TanitAD-agent-<discipline> -b agent/<discipline>-<YYYYMMDD>` and work there.
At session end: commit in your worktree, push your `agent/<discipline>-<date>` branch, and note the
branch name in your STATE.md `LAST_RUN` line. **Never commit directly to `main` from an agent
session.** The MVP orchestrator merges agent branches into main during loop triage (fast-forward or
merge; conflicts resolved by the orchestrator). Exception: if worktree creation fails, fall back to
the old behavior but retry any lock error for max 10 minutes, then log and skip the commit (the
orchestrator sweeps).

## Boundaries (updated per D-011 — hub/MVP separation)

- **NEVER write into `stack/`** or other core MVP artifacts. Code and code-change proposals go into
  an intake package: `TanitAD Research Hub/<YourDiscipline>/Implementation/incoming/<YYYY-MM-DD>-<slug>/`
  containing (a) the self-contained module(s)/patch, (b) its tests, (c) an `INTAKE.md` following
  `TanitAD Research Hub/INTAKE_TEMPLATE.md` (what, why, evidence, tests run, proposed target location
  in `stack/`, risk, rollback). The MVP orchestrator triages every package (integrate / defer /
  reject-with-reason) and writes the verdict back into the `INTAKE.md` — read verdicts on your next
  run and adapt.
- You MAY directly update: your `Research/` folder, `KNOWLEDGE_BASE.md`, your `STATE.md`,
  `HYPOTHESIS_LEDGER.md` rows, your PROJECT_STATE session-log row, and (Benchmarks & Eval only)
  `LEADERBOARD.md` / `REGULATION_TRACE.md`.
- NEVER edit `Project Steering/Mission Plan.md`. Constitution changes = proposal file in
  `Project Steering/Proposals/`.
- Respect the resource plan (`Project Steering/Master Plan.md` §4): local GPU first; no cloud GPU
  spend without an approved entry in `Project Steering/RESOURCE_LEDGER.md`.
- Honesty (P8): negative results and failed edges are first-class findings — record them.
- Intake packages must run standalone (`pytest <package>/tests`) — a package whose tests fail is a
  gate-E failure.

## Schedule (registered 2026-07-05, dev-machine scheduled tasks)

| Day | Local time | Agent |
|---|---|---|
| Mon | 06:43 | tools-devenv-agent |
| Tue | 06:43 | data-engineering-agent |
| Wed | 06:43 | architecture-inference-agent |
| Thu | 06:43 | benchmarks-eval-agent |
| Fri | 06:43 | opponent-analyzer-agent |
| Fri | 14:23 | orchestrator-agent |
| Sat | 06:44 | production-optimization-agent (added 2026-07-08, D-020) |
