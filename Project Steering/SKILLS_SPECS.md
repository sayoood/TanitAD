# TANITAD SKILLS — initial specifications

`Created 2026-08-22 (TANITAD_PROGRAMME.md §8). Skills are the CONSERVATION
mechanism: a validated procedure becomes a skill so its value never strands in a
transcript. Each spec below is buildable as .claude/skills/<name>/SKILL.md.
Build order: Review → ValidateAIDesign → RunEval first (they encode this week's
hard-won discipline), then the rest as their products mature.`

## /TanitAD_Review
**Purpose:** the configured programme review — rules, claims, quality criteria.
**Inputs:** scope (a work package dir, a PR, or "programme").
**Procedure:**
1. Read TANITAD_PROGRAMME.md, GOALS_AND_CLAIMS.md, VOCABULARY.md, RETRACTION_LOG (classes only).
2. Check every quoted number: artifact path + evidence class + tier stamp; registry wins conflicts.
3. Check estimator discipline: statistic named (σ vs σ²), CI + estimator named, n/d printed, controls present (constant + floor), hyper-params fit on fit split only.
4. Check claims register: every asserted/refuted claim has a row updated same-turn.
5. Check vocabulary compliance; flag drift.
6. Output: findings ranked by severity; each finding names the rule it violates.
**Success:** zero unverifiable numbers; register consistent; both-outcomes present for any pre-registration touched.

## /TanitAD_ValidateAIDesign
**Purpose:** the v7-tiny-style validation ladder for any model/design change.
**Inputs:** design delta (flag set or module), budget (arms × steps).
**Procedure:**
1. SPEC.md with pre-registered success criteria, BOTH outcomes, and a DELIBERATE REGRESSION arm (the gate must be able to fail).
2. Tiny rig first (v7-tiny scale, 29 min/arm, parity corpus on Thor).
3. Gates in order: participation ≥ 8.56 (σ², vs frozen-DINOv3 reference) → decodability (ego probe > pixel floor AND constant control; detection AP > prior AND pixel) → only then any driving eval.
4. Every panel: constant control, raw-input floor, printed n and d.
5. Bank raw JSON; update GOALS_AND_CLAIMS.md; registry row if a model version results.
**Success:** verdict with CIs on the pre-registered criteria; no metric invented post-hoc.

## /TanitAD_RunEval
**Purpose:** one command from checkpoint → complete, admissible eval.
**Inputs:** model key(s), tier (T1 primary), episode set (parity-checked).
**Procedure:**
1. Refuse non-parity sets unless explicitly flagged NON-PARITY.
2. Run the FOUR FAMILIES (longitudinal / lateral / tactical / strategic) — never ADE alone; paired episode-cluster bootstrap for two arms.
3. Stamp tier on every number; write raw JSON; update MODEL_REGISTRY row + leaderboard.
4. Emit the viz set: camera overlay + BEV inset + decoded tactical manoeuvre + strategic goal text.
**Success:** registry row complete; a report with a missing family says so per-family with n and reason.

## /TanitAD_DesignDataSet
**Purpose:** spec → dataset, with provenance and parity discipline.
**Inputs:** intent (scenario mix, size, sources), target name.
**Procedure:** query TanitScena (P5) → select sources → licence class check (research-OK is sufficient; augmented output goes PRIVATE on HF) → build via TanitDataSetCreator (P8) config → skip-hash + provenance manifest → episode-count and content checks (verify by content) → register in TanitScena.
**Success:** dataset + manifest + TanitScena entry; parity implications stated explicitly.

## /TanitAD_SearchScenarios
**Purpose:** semantic search over TanitScena.
**Inputs:** natural-language scenario description; filters (source, licence, size).
**Procedure:** embed query → vector search over scenario/dataset embeddings → return scenarios with dataset links, counts, provenance; optionally hand off to /TanitAD_DesignDataSet.
**Blocked on:** P5 vector DB (DataFlyWheel).

## /TanitAD_TrainModel
**Purpose:** launch a training run that cannot repeat this week's failures.
**Inputs:** config/arm, corpus (parity-checked), steps, target machine.
**Procedure:**
1. Preflights: rank-gate CAN-RULE check (spectrum-accum arithmetic), residual-init banner, param budget, X3 isolation, verify-by-import that the target machine runs TODAY'S code (md5 the shipped files; pods/Thor have no git credentials).
2. Wire the gates: pooled spectrum to the gate; participation criterion; val cache.
3. Launch under a supervisor with a DONE-marker discipline; register the run.
4. On completion: auto-run /TanitAD_ValidateAIDesign gates on the checkpoint.
**Success:** a run directory whose config.json carries every knob (incl. env-var knobs), gate verdicts that can RULE, and a registry row.

## /TanitAD_DeployModel
**Purpose:** TanitDeploy pipeline for a named target.
**Inputs:** model key, target (Thor first), precision budget.
**Procedure:** export → quantize (with a paired pre/post eval via /TanitAD_RunEval — a quantization without a paired eval is not a deployment) → port → profile (latency, memory via torch.cuda.max_memory_allocated on Thor — the only admissible probe there) → optimization report.
**Blocked on:** P6 consolidation (DeployFlyWheel).

## /TanitAD_BenchmarkCriteria  (added — the gap Sayed named)
**Purpose:** keep eval criteria COMPLETE and CONSTANT; we measurably failed at this by hand.
**Procedure:** maintain the criteria registry (four families + community benchmarks NavSim/…); diff every eval output against the registry — a missing criterion is a WORK ITEM not an omission; quarterly sweep of community benchmarks for additions; comparisons to other implementations use their published protocol verbatim, cited.
**Owner:** EvalFlyWheel.

## Skill-creation rule (autonomous conservation)
When a procedure has been executed twice successfully OR encodes a
hard-won correction (a C-entry), the Master Mind DRAFTS a skill for it
unprompted and proposes it to the PI. A validated procedure that lives only in
a transcript is treated as STRANDED (same class as an artifact on one disk).
