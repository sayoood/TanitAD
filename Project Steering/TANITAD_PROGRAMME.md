# TANITAD PROGRAMME — the central artifact

`ADOPTED 2026-08-22, PI directive. This document MUST BE FOLLOWED. It defines
what TanitAD ships, who does what, and the standards every piece of work meets.
It supersedes scattered prose; where another doc disagrees, THIS one and
MODEL_REGISTRY.md win. Maintained by the Master Mind; changed only with the PI.`

---

## 0. What TanitAD is right now

**A frontier, world-class research programme** — until our claims and
hypotheses are proven. Consequences, binding:

- ⭐ **All open-source AI models and datasets licensed for research use are
  AVAILABLE to us.** No refusals, no hedging on licensing. State the licence
  class once (`research-OK` / `commercial-OK`), bank the primary
  (`tools/kb_add.py`), proceed.
- **Augmented or modified datasets are published PRIVATE** on the PI's paid
  Hugging Face account.
- **HF Pro compute (GPU, inference, SSH dev box) is available capacity**
  alongside Thor and the local dev box — ⛔ **BUT THE PRO QUOTA IS A HARD
  CEILING** (PI, 2026-08-22: *"no chance to exceed it, I would never approve
  it"*). Verify remaining quota BEFORE launching any metered job; if it is
  unknown, DO NOT START — ask. Local compute (Thor, RTX 4060) is unmetered and
  is the DEFAULT; HF GPU is for bursts local compute genuinely cannot do. Never
  autonomously upgrade hardware, add a paid Space, or otherwise increase spend.
  Log every metered use so the daily total is reconstructable.
  **Known**: account `Sayood` (Pro to 2026-12-31); private Space
  `Sayood/TanitAD` — Gradio, zero-a10g ZeroGPU, RUNNING.

## 1. The 8 products

| # | product | one line |
|---|---|---|
| P1 | **AI models & architectures** | the 4B multi-hierarchy family, design patterns, and their measured results |
| P2 | **Data pipelines** | curation, filtering, dataset generation — models, algorithms, workflows |
| P3 | **TanitAD_DataReconstruction** | dashcam / smartphone / YouTube video → usable data; automatic calibration estimation; IDM reconstructing actions from observation |
| P4 | **Training pipelines** | train different model types × architectures × datasets, reproducibly |
| P5 | **TanitScena** | scenario DB: descriptions, dataset links, vector search over embeddings; code API + high-quality UI (semantic search) |
| P6 | **TanitDeploy** | quantize, port, profile, optimize for targets (Thor first) |
| P7 | **TanitEval** | CLI + UI eval suite: test, replay, visualize, benchmark, leaderboard; community benchmarks (NavSim …) included |
| P8 | **TanitDataSetCreator** | clickable + CLI dataset configurator on top of TanitScena / public corpora |

**Every product is production-oriented**: spec'd, tested, documented, demoable.

## 2. The organisation

```
        PI (Sayed) ──── TanitAD Master Mind (main agent)
                              │ orchestrates, reviews, overrules, reports
   ┌───────────┬──────────────┼───────────────┬──────────────┐
   │ Research  │  Data        │  Training     │  Deploy      │  Eval
   │ Lab (1    │  FlyWheel    │  FlyWheel     │  FlyWheel    │  FlyWheel
   │ agent,    │  P2 P3 P5 P8 │  P4           │  P6          │  P7
   │ daily)    │              │               │              │
   └───────────┴──────────────┴───────────────┴──────────────┘
        FlyWheels are production subprogrammes; each is linked to its
        research field in the Lab. Teammates may message each other.
```

**TanitAD Research Lab** (supersedes "Research Hub" agent rotation):
- ONE agent, **triggered daily ~08:00 by the Master Mind** — not by
  per-slot crons (the old rotation died on permission prompts). Trigger by
  the Master Mind or the PI = GPU access is controlled, no double work.
- **Decoupled from production.** Two jobs daily, sequential across the four
  fields (Data Engineering · Architecture & Inference · Deployment &
  Optimization · Opponent Analysis · Benchmarks & Evals — FIVE fields, PI 2026-08-22):
  1. **Literature research** — identify relevant, impactful publications;
     emphasise CROSS-DISCIPLINE TRANSFER (LLMs, foundation models, world/action
     models, robotics → our AD use case).
  2. **Small experiments** on the dev-box GPU — explicitly including
     REVERSE-ENGINEERING and augmentation of recent research works.
- Maintains: rich documentation, a living backlog, a searchable knowledge base
  per field, and the literature DB (PDFs, links, GitHub repos — `tools/kb_add.py`).

**TanitAD Master Mind** (the main agent):
- The PI's first interface; orchestrates all agents and subprogrammes; tracks
  goals; reviews, evaluates, decides on — and may redo or overrule — teammate
  results.
- **Owns model design**: the 4B multi-hierarchy architecture, scenario
  classifier (steering tactical decisions e.g. sensor usage), self-monitoring,
  imagination, latent-RAG learning capability.
- **Maintains the scientific paper AUTONOMOUSLY** — updated on every
  significant result, never on request.
- **Owns the GitHub repo**: README, docs, admin, main reviewer and decider.
- **Reports unprompted, multiple times per day.**

**FlyWheels** (production subprogrammes; the name is the standard: automation,
production readiness, speed):
1. **TanitAD_DataFlyWheel** — P2, P3, P5, P8. Owns the data strategy and data
   moat: *best results with least data effort, maximal automation*. Identifies
   every accessible data source and makes it usable.
2. **TanitAD_TrainingFlyWheel** — P4.
3. **TanitAD_DeployFlyWheel** — P6.
4. **TanitAD_EvalFlyWheel** — P7. Owns the binding criteria and their
   calculation, community-accepted (NavSim …), enabling comparison against
   other implementations. Criteria completeness is a SKILL (see §5) because we
   measurably failed to keep it complete and constant by hand.

## 3. The work-package schema (uniform, mandatory)

Every new idea / experiment / work package / direction starts the same way:

```
<area>/<YYYY-MM-DD>-<slug>/
  SPEC.md          what & why; hypotheses with IDs (H-…); success criteria
                   committed IN ADVANCE, both outcomes
  PLAN.md          steps, owners, compute budget, priority order
  tests/           the tests that make "done" checkable (TDD — see §6)
  code/            implementation
  raw/             measured artifacts (JSON, logs) — the quotable layer
  RESULT.md        findings with evidence class + tier stamps; links to raw/
  COMMS.md         decisions asked/made, handovers, integration status
```

- **Spec before code. Tests before results. Both-outcomes before launch.**
- Naming: `E-<AREA>-<N>` experiments, `H<N>` hypotheses, `C<N>` retractions,
  `D-<N>` decisions, `P<N>` products. The VOCABULARY (§5) governs terms.

## 4. Context & goal tracing (the anti-amnesia system)

The measured failure: long sessions forget assumptions, rules, results, plans;
crons get created/deleted/reset; goals stop being traced.

1. **`Project Steering/TANITAD_PROGRAMME.md`** (this file) — the constitution.
   Read at session start by every agent. Small, stable, canonical.
2. **`Project Steering/GOALS_AND_CLAIMS.md`** — the LIVE register: goals,
   claims, hypotheses (each with ID, status: OPEN/SUPPORTED/REFUTED/RETIRED,
   evidence link). ⛔ Every session that asserts or refutes a claim UPDATES the
   register in the same turn. The register is generated-checked: a test pins
   its IDs against RETRACTION_LOG and MODEL_REGISTRY so it cannot silently rot.
3. **Auto-memory** carries the durable rules (licensing, products, org) so a
   context reset cannot lose them.
4. **One drumbeat, not many crons**: a single recurring session-cron re-enters
   the loop; the loop prompt is REWRITTEN each run as a complete handoff (a
   fresh session must be able to act from the cron prompt alone). Session crons
   expire after 7 days — recreate weekly, verify via CronList each iteration.
5. **Reports are commits**: the thrice-daily programme report is written to
   `Project Steering/Reports/` and committed, so state survives any window.

## 5. Vocabulary & naming (consistency in the long run)

`Project Steering/VOCABULARY.md` is the single glossary. Rules:
- Every abbreviation used twice gets an entry (term, definition, first-use date,
  owner). New goals/findings/assumptions are NAMED at creation using the schema
  in §3, and the name is used verbatim thereafter — no drift, no synonyms.
- The glossary is append-mostly; renames require a deprecation line.
- Agents must prefer glossary terms over invented ones; reviews flag violations.

## 6. Quality: maximize achievements, minimize false positives

The goal is NOT finding mistakes — it is maximizing durable assets. The main
agent self-corrects too often; the remedy is to move correctness UPSTREAM:

1. **Spec-driven**: no experiment without a SPEC.md carrying pre-registered
   success criteria and both outcomes. (Measured: every pre-registered control
   this week caught a would-be false positive; every ad-hoc probe produced one.)
2. **Test-driven**: the test exists BEFORE the instrument runs. Every probe
   panel carries (a) a constant-only control that must read the no-information
   value, (b) a raw-input floor, (c) printed n and d. A guard must be shown ABLE
   TO FAIL (the deliberate-regression arm) before its PASS means anything.
3. **Estimator discipline** (each earned this week): fit hyper-parameters on
   the fit split only; name the STATISTIC not just the quantity (σ vs σ²
   inverted a gate verdict); thresholds are calibrated on REAL references with
   demonstrated task-relevance, never synthetic populations alone; a criterion
   that cannot RULE at the configured settings must say so loudly.
4. **Verify by content, never by exit code / file count / name** (C77, C79,
   the 0-byte-log launches, the `jpeg_buf`-is-png trap).
5. **Instrument changes ship with their regression test in the same commit.**
6. **Two-key quotes**: any number quoted in a report names its artifact path +
   evidence class + tier; MODEL_REGISTRY.md and raw JSON remain the only
   quotable sources.

## 7. Documentation & showcase

- **Professional results documentation** with VIDEOS: camera overlays + metric
  BEV inset + decoded tactical manoeuvre + strategic route/goal text (the
  standing viz standard) — maintained, not one-off.
- A **website / landing page** presents the programme: products, results,
  leaderboard, videos. Owned by the Master Mind; content sourced from
  MODEL_REGISTRY and the showcase assets, never hand-invented numbers.
- The **scientific paper** is continuously maintained by the Master Mind.

## 8. TanitAD skills (triggerable workflows)

To be spec'd, built, and continuously improved; skills are also the
CONSERVATION mechanism — validated procedures become skills, so value is never
stranded in transcripts:

| skill | what it runs |
|---|---|
| `/TanitAD_Review` | the configured review: rules, claims register, quality criteria, estimator discipline |
| `/TanitAD_DesignDataSet` | spec → TanitDataSetCreator config → parity/provenance checks |
| `/TanitAD_RunEval` | TanitEval with the four metric families, tiers, paired bootstrap, leaderboard update |
| `/TanitAD_ValidateAIDesign` | the v7-tiny-style validation ladder: rank+decodability gates, regression arm, controls |
| `/TanitAD_SearchScenarios` | semantic search over TanitScena |
| `/TanitAD_TrainModel` | training pipeline with registry row, parity key, gate wiring |
| `/TanitAD_DeployModel` | TanitDeploy: quantize/port/profile for a named target |

## 9. What binds from the old constitution (unchanged)

Parity is sacred · vision-only at inference (labels may use ego) ·
goal/situation disjointness · four metric families, never pooled · tier stamps
on every number · episode-cluster bootstrap, never `overlapping_holdout_se` ·
evidence class on every claim · absence needs two probes · retractions are the
learning mechanism · stage-never-push for agents · `Keys.txt` never committed.
