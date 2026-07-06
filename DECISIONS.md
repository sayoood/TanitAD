# TanitAD — Decision Log (ADR style)

> Append-only. Every entry: ID, date, status, decision, rationale, consequences.
> Statuses: `proposed` → `accepted` | `rejected` | `superseded by D-xxx`.
> Decisions must never contradict `Project Steering/Mission Plan.md` (the constitution).
> Changes to the constitution itself are only proposals → `Project Steering/Proposals/`.

---

## D-001 — Repo layout: runnable code lives in `stack/` (2026-07-05, accepted)

**Decision.** All runnable MVP code lives in a top-level `stack/` folder (Python package `tanitad`),
because the discipline folders ("Benchmarks & Eval", "TanitAD Research Hub") contain spaces and mixed
content and are unsuitable as Python package roots. Discipline folders keep documents, research and
per-area state; `stack/` keeps code, configs, tests, scripts.

**Rationale.** Clean imports, CI-ability, one place to run/test; matches Mission Plan structure without
changing it.

## D-002 — Phase 0 data strategy: sim-first, then real data with free actions (2026-07-05, accepted)

**Decision.** Phase 0 uses a staged data ladder:
- **Stage A (weeks 1–4):** MetaDrive procedural driving sim (BEV first, then front camera). Purpose:
  prove the edge hypotheses cheaply and falsifiably (topology/generalization experiments are impossible
  to run cleanly on real logs).
- **Stage B (weeks 5–10):** comma2k19 (33 h real driving, CAN steering/speed + GNSS — real actions with
  zero annotation cost) as first real dataset; **NVIDIA PhysicalAI-AV** (1 727 h, multi-camera + radar +
  lidar, loader already prototyped in `DataEng/AVDataSetLoader`) as the multi-sensor dataset feeding H2
  (modality steering); nuScenes-mini for OOD probes.

**Rationale.** Follows the validated ALPS-4B transfer program (`ALPS-4B/docs/AD_TRANSFER_RESEARCH.md` §9–10);
maximizes proof-per-GPU-dollar (P5); PhysicalAI-AV is the only open dataset with the camera+radar+lidar
diversity H2 needs. License review of PhysicalAI-AV before public claims is an open task.

## D-003 — Architecture baseline: from-scratch 4B latent world model; frozen-encoder as H4 comparison arm (2026-07-05, accepted)

**Decision.** The main track is the 4B hierarchical latent world model trained from scratch with
SIGReg-only anti-collapse (LeJEPA), residual/delta prediction, change-weighted latent loss, inverse-dynamics
grounding, spatial grid readout, and frozen imagination-calibrated probes for trajectory decode.
A parallel low-cost arm trains the same predictor stack on top of a **frozen DINOv3/DINOv2 encoder (+LoRA)**
to answer H4. Decision between arms is made on measured gate results, not preference.

**Rationale.** Every component of the main track is individually validated in ALPS-4B (assets A1–A10) or
externally (LAW, IDOL, ResWorld, V-JEPA-2-AC). The from-scratch arm is what makes the data-efficiency
claim disruptive; the frozen arm hedges the risk and quantifies the trade-off.

## D-004 — Instrument doctrine I1–I4 is mandatory before any model claim (2026-07-05, accepted)

**Decision.** Every evaluation ships four instrument rows before any model claim:
I1 oracle-decode row (≈1.0 or the harness is broken), I2 batch-consistency check (batch-1 vs batched
encodings match to 1e-4), I3 route/episode-level splits for all probes, I4 persistence baseline
(`imag relative` < 1). These run in CI (`stack/tests/`). No architecture change may be motivated by a
gate that has not passed its instrument rows.

**Rationale.** The ALPS-4B BatchNorm incident: four weeks of results were noise because batch-1 and
batched encodings differed 115 %. Deployment inference is batch-1 streaming; in AD the cost of silent
measurement error is safety, not GPU-hours. (P8: be honest — includes honest instruments.)

## D-005 — Continuation protocol: PROJECT_STATE.md as single session entry point (2026-07-05, accepted)

**Decision.** Cross-session continuity is handled by three artifacts: `PROJECT_STATE.md` (current truth,
updated every session), `DECISIONS.md` (this file), and `Project Steering/CONTINUATION_PROTOCOL.md`
(the ritual). Every discipline folder additionally keeps a `STATE.md` for area-local state.

## D-006 — Research hub agents run as weekly scheduled jobs with an internal quality loop (2026-07-05, accepted)

**Decision.** The five weekly agents (Mon Tools&DevEnv, Tue DataEng, Wed Architecture&Inference,
Thu Benchmarks&Eval, Fri Opponent Analyzer + Orchestrator/Synthesizer) are defined as versioned prompt
files in `TanitAD Research Hub/agents/` and registered as scheduled jobs. Each agent runs an internal
loop (search → analyze → write → self-critique → iterate) until its quality gates are met, then updates
its knowledge base, its `STATE.md`, and commits.

## D-008 — Model scale ≥250 M params; H15 imagination promoted into Phase 0 (2026-07-05, accepted — decided by Sayed)

**Decision.** The Phase 0 main-track model is **TanitAD-4B-M at ~250 M instantiated parameters**
(component budget in `Project Steering/Phase 0 Plan.md` §2.1, updated). The 10–100 M framing of the
initial plan is superseded for the trained stack; the efficiency claim shifts to
"sub-300 M with sparse activation and layered inference rates — 40× smaller than Alpamayo-1-class
VLAs". Additionally, **H15 (imagination in unobserved areas) moves from Phase 1 into Phase 0** as a
first-class training mechanism (sector-masked imagination + latent advection prior + epistemic
gating) with a new gate **D9**.

**Consequences.** RTX 4060 becomes smoke/debug-only for the main model (it fits ~250 M at batch ≤8,
128 px, for pipeline validation only); real Stage-A/B training moves to RunPod A40 (runbook:
`stack/RUNPOD_RUNBOOK.md`). Tactical layer gains a parametric maneuver-horizon predictor now (MoE
upgrade in WP4). Strategic stays deliberately non-parametric (VQ + graph) in Phase 0; the frozen-LLM
bridge (Phase 1) sits outside this budget.

## D-013 — Literature-search protocol upgraded after missed-papers finding (2026-07-06, accepted)

**Decision.** Triggered by Sayed surfacing two directly relevant papers (arXiv 2606.27014 JEPA
generalization theory; 2507.00028 HiT-JEPA) that the kickoff research missed. Upgrades:
(1) every agent's SEARCH step now includes a **systematic arXiv sweep** (fixed query set per
discipline over cs.LG/cs.CV/cs.RO/eess.SY, last-7-days window) plus a **citation-graph walk** from
our anchor papers (LeJEPA, V-JEPA-2, LAW, World4Drive, DINO-WM) — not just topic searches;
(2) the Architecture & Inference agent gains an explicit **theory-watch duty** (JEPA/world-model
theory lineage: Balestriero/LeCun, Klindt, HaoChen-style spectral SSL theory, PKU Yisen Wang group);
(3) **Ressources-inbox rule:** any new file Sayed drops into `Ressources/` is detected (mtime) and
deeply analyzed by the next agent run or MVP session, results filed in the hub;
(4) at every phase boundary the MVP stream runs a dedicated multi-day deep-research pass, not a
single-session sweep. Honest note (P8): a one-session kickoff sweep cannot reach post-doc
literature coverage on 16 hypotheses; the weekly cadence + these rules are the fix, and misses
should keep being reported when found.

## D-012 — NVIDIA PhysicalAI-AV: use now for training/research; license handling deferred (2026-07-06, accepted — decided by Sayed)

**Decision.** PhysicalAI-AV enters the training/eval corpus immediately (urban semantic diversity,
multi-camera, radar/lidar — everything comma2k19 lacks). The license finding (internal-dev-only
clauses, DataEng 2026-07-07 note) is NOT forgotten but parked: revisit before any public claim,
publication, or external demo that used it. Ledger and reports must tag experiments that consumed
PhysicalAI-AV data so the exposure is auditable later. Additionally the Data Engineering agent's
top standing duty becomes an **extensive AV-dataset landscape sweep** (HuggingFace-first: search
`datasets` for driving/AV corpora incl. new 2025/26 releases; then academic mirrors), maintaining
`Data Engineering/Research/DATASET_LANDSCAPE.md` with license/actions/sensors/urban-richness columns.

## D-011 — Hub/MVP separation: agents propose via intake queues, the MVP integrates (2026-07-07, accepted — proposed by Sayed)

**Decision.** Research-hub agents no longer write into `stack/` (or any core MVP artifact). Their
implementation increments land as **self-contained intake packages** in
`TanitAD Research Hub/<Discipline>/Implementation/incoming/<date>-<slug>/` with an `INTAKE.md`
(what/why/tests/target location/risk). The **MVP stream acts as orchestrator**: every MVP session
starts with an intake triage (integrate / defer / reject-with-reason), and the Friday Orchestrator
agent runs the same triage as a safety net. Rejections and integration notes are written back to the
package's `INTAKE.md` and the discipline `STATE.md`, so agents learn from the feedback loop.

**Rationale.** Matches the constitution (agents "identify concrete proposals to our main stream";
the MVP is steered manually). Prevents unreviewed mutation of the production stack, keeps gate/
instrument discipline in one place, and avoids working-tree collisions — all sessions share one
folder on one machine, so git-branch isolation per agent would be operationally fragile here; a
directory-level write boundary is robust.

**Scope notes.** (1) Agents still directly update their own Research folders, knowledge bases,
`HYPOTHESIS_LEDGER.md`, their `STATE.md`, and their PROJECT_STATE session-log row — shared *state*,
not core code. (2) The two pre-D-011 agent contributions inside `stack/` (MetaDrive adapter, shared
`_contract.py` + real-data validation) are grandfathered: reviewed, tested, kept. (3) Small doc-only
improvements outside the hub still go through intake if they touch steering documents.

## D-010 — Sim (MetaDrive) is combined with real data for interaction, not pixels (2026-07-06, accepted)

**Decision.** MetaDrive stays in Phase 0 in a strictly complementary role next to the real-data-first
rule (D-009). Role separation:
- **Real data owns:** representation learning, all public open-loop numbers (D1–D3 on real held-out
  routes), validation sets (always real-only).
- **Sim owns what logs cannot provide:** (a) off-expert action–consequence coverage via perturbed/
  exploration rollouts — comma2k19 only covers the safe expert manifold, but imagine-and-select must
  rank *bad* candidate actions it has never seen the consequences of; (b) scripted occluders for
  object-level D9/LOPS (Ghost Cut-Through); (c) blocked-route topology tasks (D5) and procedural
  simple→complex generalization (D6); (d) collisions/near-misses; (e) closed-loop evaluation (G0.5).
- **Combination:** co-training via `MixedWindowDataset` (`--data mix`), default 80 % real / 20 % sim,
  sim episodes rendered as front-camera RGB in the SAME contract (2-frame stacks @ 256 px), every item
  domain-tagged. **Mandatory bake-off:** real-only vs mixed at matched steps — sim keeps its share only
  if real-data gates do not regress (one lever per run, D-004).

**Prerequisite.** MetaDrive live rollout (supervised source install) + front-camera RGB rendering in
the adapter (currently BEV) + a perturbation-policy episode generator writing `*.pt` episodes
(`tanitad/data/mixing.py: save_episode`). Until then training runs real-only — no time lost.

## D-009 — Real camera data first; toy demoted to CI fixture (2026-07-06, accepted — decided by Sayed)

**Decision.** Phase 0 training starts on **real front-camera data immediately**; no training time is
spent on toy data (it remains only as a CI fixture and instrument-check substrate). Primary source:
**comma2k19 via the ungated HuggingFace mirror** (`commaai/comma2k19`, `raw_data/Chunk_*.zip`) — real
20 fps camera + real CAN actions + GNSS poses, zero annotation. Chunk_1 (8.7 GB ≈ 3.3 h) is the first
training corpus; further chunks stream in as needed. PhysicalAI-AV (gated=auto, token verified) is the
second source and the multi-view/G0.7 corpus. **Consequence for gates:** D1–D4, D8, D9 run on real
data; D5/D6 (topology/generalization) still require MetaDrive closed-loop and stay scheduled after the
supervised source-install. Primary config: `base250cam` (6-ch 2-frame RGB stacks @ 256 px, same 261 M
budget). This supersedes the sim-first Stage-A ladder of D-002.

## D-007 — Phase 0 primary open-loop benchmark: NAVSIM-style trajectory metrics + custom Phase-0 metrics (2026-07-05, accepted)

**Decision.** Phase 0 measures: (a) standard open-loop trajectory metrics (ADE/FDE@1s/2s, and NAVSIM
PDMS-style sub-scores where computable), (b) closed-loop success/route-completion in MetaDrive,
(c) the five custom TanitAD metrics from Deep Think 14 (LAL, TMS, OKRI, CNCE, LOPS) implemented in
`Benchmarks & Eval`, (d) efficiency envelope: params, FLOPs/decision, latency on RTX 4060 (proxy) with
Orin/Thor targets tracked. Full NAVSIM/Bench2Drive leaderboard entries are Phase 1+ goals.

**Rationale.** Recognizable external metrics (no own-defined claims only) + custom metrics that expose
the edges existing KPIs don't cover, per Mission Plan "Benchmarks & Eval".
