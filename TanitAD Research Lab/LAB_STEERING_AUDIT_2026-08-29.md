# LAB STEERING AUDIT — 2026-08-29

`Commissioned by the PI's question "did you review their instructions, backlog etc.?" — the
honest answer was no. This audit reads every Research Lab steering file, dates it, checks it
against the two binding references (Project Steering/TANITAD_PROGRAMME.md, the constitution,
and Project Steering/GOALS_AND_CLAIMS.md, the live register), and hands each file a verdict.
⛔ This audit CHANGES NOTHING it audited — the Master Mind and the domain owners decide; the
companion deliverable is the harvested seed list at TanitAD Research Lab/LAB_BACKLOG.md.`

## 0. Method + evidence classes

- **Freshness** = last content write, MEASURED two ways that agree byte-for-byte on every file:
  the worktree mtime (PowerShell listing) and the Google-Drive `modifiedTime` (Drive API).
  `git log --follow -1` was also taken for every file (MEASURED): **every audited file's last
  commit is `0e53d7e` 2026-08-23 (the Hub→Lab migration sweep)** except LAB-RUN-002
  (`27021d5`, 08-28) and LAB-RUN-003 (`91c35a1`, 08-29) — so the -1 commit date is the rename,
  NOT a content update, and content dates below are the mtimes.
- ⚠️ **The G: mount was hard-down (Errno 22 / "Invalid request code", reads failing across the
  whole repo) for most of this audit.** Metadata ops worked; content reads did not. All content
  was therefore read through the **Drive API** (fileIds recorded in the session transcript) —
  same bytes, same sizes as the worktree listing. Deeper git history (`--follow -2`) was
  unavailable during the outage; the two-source mtime agreement stands in for it.
- Threshold: anything whose last content write predates **2026-08-22** (the programme redesign)
  is at minimum **REVIEW-REQUIRED**.
- Quotes below are verbatim from the audited files; where the mount outage prevented `grep -n`
  line numbers, the quote is anchored by its section heading instead.

## 1. Verdict table

Verdicts: **CURRENT** · **CURRENT-op** (operational doc, minor refresh) · **STALE-sup** (superseded
by the named artifact) · **CONTRADICTS** (named register row / constitution clause) ·
**ARCHIVE** (historical record; move or mark). Recommendation is the one-word action for the owner.

### 1.1 Lab top level

| file | last content | verdict | recommendation |
|---|---|---|---|
| `2026-08-29-LAB-RUN-003.md` | 2026-08-29 | **CURRENT** | keep |
| `2026-08-28-LAB-RUN-002.md` | 2026-08-28 | **CURRENT** | keep |
| `2026-08-22-LAB-RUN-001.md` | 2026-08-23 | **CURRENT — with one CONFIRMED defect it exported into the register**: its proposed id "H-EVAL-1" collided with the register's pre-existing H-EVAL-1, and BOTH rows now sit in the live register under the same id (see §2 C-11) | keep; re-register the pseudo-sim row under a fresh id |
| `HYPOTHESIS_LEDGER.md` | 2026-08-23 (sweep); newest dated entries **2026-08-07** | **CONTRADICTS** constitution §4 / GOALS_AND_CLAIMS — see §2 C-2 | PI/MM: demote to historical annex of the register, or subordinate its header |
| `KNOWLEDGE_BASE.md` (router) | 2026-08-18 | **CONTRADICTS** — all 7 routed paths are dead post-migration; counts stale — see §2 C-1 | rewrite as 5-field router |
| `2026-07-08-screening-digest.md` | 2026-07-08 | **ARCHIVE** | archive |
| `2026-07-11-sayed-papers-screening.md` | 2026-07-11 | **ARCHIVE** (two P1 seeds harvested: AUTOPILOT-VQA probe-transfer, ZipDepth) | archive after harvest |

### 1.2 Architecture & Inference

| file | last content | verdict | recommendation |
|---|---|---|---|
| `BACKLOG.md` | 2026-08-03 | **STALE-sup** (flagship/REF-era items; two items now measurement-refuted — §2 C-4) | rewrite around v7r/B1; live items harvested |
| `GOALS.md` | 2026-07-20 | **STALE-sup** (G1/G2 target the retired flagship line, W34/W35 deadlines lapsed; G3 quality-per-FLOP ledger harvested) | rewrite |
| `Research/STATE.md` | 2026-08-03 | **STALE-sup** (old rotation; eval pod terminated) — carries live debt: 7 verdictless intake pkgs incl. the 2026-07-10 orthogonality instrument | harvest debt, then archive-fold |
| `Research/KNOWLEDGE_BASE.md` | 2026-08-18 | **REVIEW-REQUIRED** — findings log, no entries since the redesign (see §3 S-1) | resume appending |
| `ARCHITECTURE_WIRING_COMPARISON.md` | 2026-07-20 (sweep-touched 08-23) | **STALE-sup** by v6/v7 wiring + `V6F_PLANNER_DESIGN.md` / `v7-recipe` work | mark historical |
| `V3_HIERARCHICAL_PLANNING_DESIGN.md` | 2026-07-19 | **STALE-sup** (v3 planning line superseded by v6F planner design + PREREG_D-TAC1) | archive |
| `V3_GOAL_VOCABULARY_V1.md` | 2026-07-20 | **STALE-sup** — goal vocabulary now lives in D-LABEL-GT's canonical v7 label set (`tactical_goals.py`, `g_tac_geom.py`) | archive |
| `V3_FACTORIZED_TACTICAL_HEAD_SPEC.md` | 2026-07-21 | **STALE-sup** — factored head SHIPPED (D-TAC1: lat×lon in `refc.py`, macro-recall 0.8290); spec's job is done | archive as design provenance |
| `V35_DESIGN.md` | 2026-07-21 | **STALE-sup** (v3.5 synthesis line closed by the v6 handover) | archive |
| `V4_FLAGSHIP_DESIGN.md` | 2026-07-21 | **STALE-sup** (180 KB flagship-v4 design; line retired) | archive |
| `IDM_VIDEO_PRETRAIN_DESIGN.md` | 2026-07-23 | **REVIEW-REQUIRED — still live as the P3 seed** (TanitAD_DataReconstruction product); H7 slope still `untested` | keep, re-scope under P3 |

### 1.3 Benchmarks & Evals

| file | last content | verdict | recommendation |
|---|---|---|---|
| `BACKLOG.md` | 2026-08-01 | **STALE-sup** in pipeline items (CTRV-floor landing, eval-pod, CARLA W31-32); NAVSIM entry audit item DONE by LAB-RUN-002/003 portfolio; live items harvested (midpoint-CTRV, high-speed lateral, WP.29, eid normalisation) | rewrite around the benchmark portfolio + estimator closeout |
| `GOALS.md` | 2026-08-01 | **STALE-sup** (G1 substantially met; G2 CARLA-gating reframed by Bench2Drive GO-conditional; G3 WP.29 untouched and harvested) | rewrite |
| `Research/STATE.md` | 2026-08-02 | **STALE-sup**; carries 3 escalations of 2026-08-02 whose closure is unverified (registry §0.3 restatement, v4-gate third-floor re-run, `eid` encoding normalisation) + the intake-stall measurement | harvest, archive-fold |
| `Research/KNOWLEDGE_BASE.md` | 2026-08-18 | **REVIEW-REQUIRED** (no entries since redesign, §3 S-1) | resume appending |
| `TACTICAL_LABEL_STATE_INDEX.md` | 2026-08-18 | **CURRENT-op with caveat** — its §6 open items (registry rows unapplied; `maneuver_acc` GPU re-run; DIR_YAW re-read unfinished) are still open; but D-LABEL-GT (08-28) made the PI-reviewed v7 label set canonical, which re-scopes items about v1/v2 label families | keep; add a D-LABEL-GT banner |
| `TANITEVAL_V2_METRIC_SUITE.md` | 2026-07-21 | **STALE-sup** — largely absorbed into `four_families.py` + criteria machinery (H-EVAL-4) | archive as design provenance |
| `PLANNER_VIZ_CONCEPT.md` | 2026-07-21 | **REVIEW-REQUIRED** — viz standard itself is binding (constitution §7); doc predates the standing overlay scripts | fold into P7 UI spec |

### 1.4 Data Engineering

| file | last content | verdict | recommendation |
|---|---|---|---|
| `BACKLOG.md` | 2026-08-03 | **CONTRADICTS** (the "NEVER to HF, even privately" doctrine — §2 C-3) + **STALE-sup** (flagship-v2 mix levers; no mention of B1/Alpamayo corpus) | fix C-3 line, rewrite around B1; live items harvested |
| `GOALS.md` | 2026-07-20 | **STALE-sup** (G3 targets flagship D1; deadlines lapsed) — G1 owned-tier + G2 IDM remain live substance (P3/P2 products) | rewrite |
| `Research/STATE.md` | 2026-08-02 | **STALE-sup**; live escalations harvested (v2-clean-val PI freeze-or-reject, ZOD access signature, `pool_columns.py` intake) | harvest, archive-fold |
| `Research/KNOWLEDGE_BASE.md` | 2026-08-18 | **REVIEW-REQUIRED** (§3 S-1; also predates D-DATA-ALPA-FULL — any "Alpamayo is limited" reading in it is C142-retracted) | resume appending; sweep for C142 |
| `DATA_STRATEGY_FOR_HIERARCHY.md` | 2026-07-21 | **STALE-sup** — strategy predates the Alpamayo screening line (08-18/19), D-LABEL-GT and D-CORPUS-B1, which now ARE the hierarchy data strategy | mark historical; extract the still-unserved strategic-topology gap (no map/lane/junction in PhysicalAI — pinned fact) |
| `TANITDATASET_V1_STRATEGY.md` | 2026-07-20 | **STALE-sup** (v1 strategy predates B1) — TanitDataSet as a PRODUCT is now P8 | fold into P8 spec |
| `TANITDATASET_TIER_INTEGRATION_2026-07-21.md` | 2026-07-21 | **ARCHIVE** | archive |
| `2026-07-22-tanitdataset-C-build-and-push-stage.md` | 2026-07-22 | **ARCHIVE** (build log) | archive |
| `OWN_DATASET_PLAN.md` | 2026-07-17 | **REVIEW-REQUIRED** — owned-tier plan (ZOD/PandaSet) not refuted, but unranked vs B1; ZOD access still PI-gated | re-rank under B1 |
| `DATA_LAKE_ARCHITECTURE.md` | 2026-07-17 | **REVIEW-REQUIRED** — lake never ran at scale; P2-product relevant | re-scope under P2 |
| `Research/YOUTUBE_DASHCAM_STRATEGY.md` | 2026-07-10 | **REVIEW-REQUIRED — live as the P3 seed** (Y0-Y2 pipeline, falsifiers written) | keep, attach to P3 |
| `Research/DATASET_LANDSCAPE.md` | 2026-07-20 | **REVIEW-REQUIRED** (landscape predates Alpamayo full-augmentation + L2D adapter) | refresh on next DataEng pass |

### 1.5 Deployment & Optimization

| file | last content | verdict | recommendation |
|---|---|---|---|
| `BACKLOG.md` | 2026-08-03 | **STALE-sup in the quant thread** (D-B1-GATE + the 08-29 quant-gate spec supersede O5/B3 framing); A2 eval-cluster review + O2-pre export fix harvested; header still says "Production & Optimization" (old field name) | rewrite; rename header |
| `Research/STATE.md` | 2026-07-20 | **STALE-sup** | archive-fold |
| `Research/KNOWLEDGE_BASE.md` | 2026-08-18 | **REVIEW-REQUIRED** — EMPTY by declaration ("carried no entries through the 2026-08-18 restructure") while runs 002/003 produced exactly its subject matter (quant surveys, TRT #4590) | file the findings |
| `THOR_DEPLOYMENT_RUNBOOK.md` | 2026-08-16 | **CURRENT-op** — newest pre-redesign doc; O-item table partially superseded (see BACKLOG); refresh fleet facts (tanitad-eval terminated) | refresh §6 |
| `THOR_OPTIMISATION_PLAN.md` | 2026-08-02 | **STALE-sup in parts** — flagship-v1/v5f-era engines; quant stages → `quant_gate.py` spec | re-price for v7/B1 |
| `THOR_ACCESS_BRIEF.md` | 2026-08-02 | **CURRENT-op** | keep |
| `PRODUCTION_READINESS.md` | 2026-08-02 | **REVIEW-REQUIRED** (predates P6 TanitDeploy product framing) | fold into P6 spec |
| `ALPASIM_ON_THOR_BLOCKED.md` | 2026-08-02 | **CURRENT-op with one stale line** — the self-correction is good; but "Restarting it [tanitad-eval] restores the whole capability" is dead (host TERMINATED; path now = Vulkan-fixed pod / Bench2Drive prep per LAB-RUN-003) | update the unblock line |
| `ALPASIM_ON_THOR_PATH.md` | 2026-08-02 | **CURRENT-op** (same fleet-refresh caveat) | refresh |
| `FLAGSHIP_V1_INFERENCE_OPTIMIZATION.md` | 2026-07-26 | **ARCHIVE** (flagship-v1 measurements; historical) | archive |
| `_devenv/BACKLOG.md` | 2026-08-03 | **CONTRADICTS** in its "Blocked on Sayed" list (AlpaSim NO-GO / docker-host-only — §2 C-6); live items harvested (hook wiring, rerun-sdk pin, dual-sink tee, TerraZero watch) | fix C-6, rewrite |
| `_devenv/GOALS.md` | 2026-08-03 | **STALE-sup** (fleet + agent model changed; G1 probe targets 4 dead pods) | rewrite |
| `_devenv/Research/STATE.md` | 2026-08-03 | **STALE-sup** | archive-fold |
| `_devenv/Research/KNOWLEDGE_BASE.md` | 2026-08-19 | **REVIEW-REQUIRED** (§3 S-1) | resume appending |

### 1.6 Opponent Analysis

| file | last content | verdict | recommendation |
|---|---|---|---|
| `BACKLOG.md` | 2026-08-03 | **STALE-sup** (eval-pod probes retired by run #5 itself; CARLA framing superseded by the benchmark portfolio); live items harvested (Waabi, Orbis-2 read, W-05 wedge, deltas) | rewrite |
| `SCENARIO_DATABASE.md` | 2026-08-03 | **CURRENT-op as an asset** (scenario DB = P5 TanitScena seed); content predates Alpamayo-R1 | keep; attach to P5 |
| `Research/STATE.md` | 2026-08-02 | **STALE-sup**; the intake-stall systemic finding + Orbis-2/CheckVLA recommendations harvested | harvest, archive-fold |
| `Research/KNOWLEDGE_BASE.md` | 2026-08-18 | **REVIEW-REQUIRED** (§3 S-1) | resume appending |
| `Research/WEAKNESS_CATALOG.md` | 2026-08-03 | **CURRENT-op as an asset** (W-01…W-11; Momenta retraction applied) | keep |
| `Research/OPPONENT_PROFILES.md` | 2026-08-03 | **REVIEW-REQUIRED** (predates Alpamayo-R1 open-weights, GAIA-4, DrivoR) | refresh on next Opp pass |

### 1.7 Lab-nested `Project Steering/` (vestige)

| file | last content | verdict | recommendation |
|---|---|---|---|
| `Project Steering/Research/STATE.md` | 2026-07-20 | **ARCHIVE + CONTRADICTS as quotable** — the W33 synthesis states "flagship-v1 @30k … the FIRST sub-floor arm (floor 0.5005)"; under the corrected estimator this is a **TIE vs CTRV** (register H-EST-4; B&E STATE F2: +0.0993 [−0.026, +0.220] not separated) | archive with a correction banner |
| `Project Steering/Research/KNOWLEDGE_BASE.md` | 2026-08-18 | **ARCHIVE** (1 entry; the field does not exist in the 5-field org) | archive |
| `Project Steering/Research/2026-07-17-external-survey-derivation.md` | 2026-07-18 | **ARCHIVE** (H19-H24 origin doc; ledger cites it — keep bytes, mark historical) | archive |

## 2. Contradiction list (file → binding reference, with quotes)

**C-1 — the KB router routes to a tree that no longer exists.**
`TanitAD Research Lab/KNOWLEDGE_BASE.md` (router table) links
`Opponent Analyzer/Research/KNOWLEDGE_BASE.md`, `Tools&DevEnv/Research/KNOWLEDGE_BASE.md`,
`Benchmarks & Eval/Research/KNOWLEDGE_BASE.md`, `Production & Optimization/Research/…` —
**none of these directories exist** after the 0e53d7e migration (actual: `Opponent Analysis/`,
`Deployment & Optimization/_devenv/`, `Benchmarks & Evals/`, `Deployment & Optimization/`).
Contradicts TANITAD_PROGRAMME.md §2 (five fields, named) and the PI's plural-rename directive
(2026-08-27, "Benchmarks & Evals"). Its "175 entries across 7 areas" also predates all three Lab
runs. Every link a fresh agent would follow from the router 404s.

**C-2 — two files each claim to be the single authority on hypothesis status.**
`HYPOTHESIS_LEDGER.md:1`: *"TanitAD — Hypothesis Ledger (SINGLE SOURCE OF TRUTH for hypothesis
status)"* and `:3` *"This file is the only place a hypothesis status may be quoted from."*
vs `Project Steering/GOALS_AND_CLAIMS.md:1-7` (*"the live register … goals, claims, hypotheses …
⛔ any session that asserts, supports, or refutes a claim UPDATES this file IN THE SAME TURN"*)
and TANITAD_PROGRAMME.md §4.2. The ledger's newest dated evidence is **2026-08-07**; it contains
**zero** of the register's live rows (`grep -c 'H-RANK\|E-DEC'` = 0) while carrying
flagship-era statuses and actions (H1a "Confirmed ≈90 %… UNDER RE-ADJUDICATION", HPP battery).
A fresh context obeying the ledger's header would quote a hypothesis surface 3 weeks behind the
programme's frontier.

**C-3 — the DataEng backlog still carries the retired no-HF doctrine.**
`Data Engineering/BACKLOG.md` §"P0 — next run", item "-2": *"WITHOUT shipping
PhysicalAI-derived data (license doctrine: NEVER to HF, even privately)"* — contradicts
TANITAD_PROGRAMME.md §0: *"Augmented or modified datasets are published PRIVATE on the PI's paid
Hugging Face account"*, and the measured fact that `Sayood/tanitad-alpamayo2-augmentation`
exists on HF (register D-DATA-ALPA-FULL, 2026-08-23). An agent obeying this backlog line would
refuse work the constitution mandates.

**C-4 — two AI backlog items are refuted by measurement and would misdirect GPU-days.**
`Architecture & Inference/BACKLOG.md` item 3(c): *"rollout-k 4→12-20 (close the train/eval
gap)"* and item 2c: *"Extend imagination horizons past 0.4 s"* — contradict register
E-DEC-66 (*"drift 0.6701 … vs incumbent 0.669 — k is INERT on the trainable line"*),
H-PROOF-6d (*"the h≥2 heads are VESTIGIAL"*), H-PROOF-7 (composed h=1 predicts to 6 steps),
and LAB-RUN-003 (all four closest-relative WMs roll one-step models). The v7 recommendation is
the opposite: drop the h≥2 heads.

**C-5 — the whole per-domain goal layer targets a retired model line and a dead cadence.**
All GOALS.md files (AI 07-20, B&E 08-01, DataEng 07-20, _devenv 08-03) carry D-029 weekly-run
mechanics and flagship/REF-arm targets with July/August deadlines (e.g. AI G1: *"flagship @30k …
by W34 (≈2026-07-31)"*). Contradicts TANITAD_PROGRAMME.md §2 (ONE Lab agent, daily, decoupled
from production) and D-CORPUS-B1 (the training corpus and consumers are now v7f / refcv3 /
refav1 / refd on the Alpamayo-labelled 4,719-clip set — no steering file in the Lab tree
mentions B1 or the new consumers).

**C-6 — the devenv backlog's standing blockers were dissolved by later measurement.**
`Deployment & Optimization/_devenv/BACKLOG.md` P1.0: *"AlpaSim … RETIRED … answered NO-GO …
residual ask = infra: a docker-capable GPU host"* and "Blocked on Sayed": *"A docker-capable GPU
host … the ONLY blocker between us and an AlpaSim closed loop"* — contradicted by
`ALPASIM_ON_THOR_BLOCKED.md`'s own 2026-08-02 correction (*"AlpaSim RUNS, and we have run it —
bare on one A40 … acquired without Docker — a bearer-token layer-fetch"*) and by the banked
closed-loop videos (`Benchmarks & Evals/_evaluation/Videos/alpasim-closedloop-*`). The Vulkan
ICD fix (CLAUDE.md, operating standard §2) removed the rendering blocker the same way.

**C-7 — a stale unblock line inside an otherwise-corrected doc.**
`ALPASIM_ON_THOR_BLOCKED.md` correction block: *"the real constraint is … that `tanitad-eval` is
stopped … Restarting it restores the whole capability"* — `tanitad-eval` is TERMINATED (fleet
fact; the val dumps are in-repo, the val40 cache on Thor). The current unblock path is the
Vulkan-fixed pod + the Bench2Drive prep item (LAB-RUN-003).

**C-8 — the B&E backlog's top item survives only in mutated form and points at the wrong fix.**
`Benchmarks & Evals/BACKLOG.md` P0#1: *"Land the CTRV floor … every new gate run still publishes
CV-only verdicts"* — the leaderboard side was re-adjudicated (register H-EST-3: 9 of 25
positions move), but the D1 gate itself STILL adjudicates on a mean-of-split-means
(register D-GATE-D1-EST / H-EST-2, found 2026-08-29, `stack/tanitad/eval/gates.py:249-251`,
OPEN, PI decision). Executing the old backlog item as written would patch floors into an
estimator the register says must be replaced.

**C-9 — a dead-path citation inside the ledger.**
`HYPOTHESIS_LEDGER.md:112` (§1.3 heading) cites
*"TanitAD Research Hub/Project Steering/Research/2026-07-17-external-survey-derivation.md"* —
the `TanitAD Research Hub/` prefix is dead (PI 2026-08-27: all citations to
`TanitAD Research Lab/`). At least one Hub-path citation survived the byte-identical rename.

**C-10 — the W33 synthesis is quotable-wrong on the programme's headline result.**
`Project Steering/Research/STATE.md` (Lab-nested, 2026-07-20): *"flagship-v1 @30k FINAL
ade_0_2s 0.4522 ± 0.031 m = the FIRST sub-floor arm (floor 0.5005 / CTRV 0.523 …)"* — under the
binding estimator the CTRV comparison is a **TIE** (+0.0993 [−0.026, +0.220], not separated;
register H-EST-4, B&E STATE 2026-08-02 F2), and §0.3-style "first arm below every trivial bar"
wording was itself escalated for restatement. Any reader quoting this file re-imports a
retracted claim.

**C-11 — LAB-RUN-001 exported a duplicate hypothesis id into the live register (CONFIRMED
2026-08-29, API read of the live `GOALS_AND_CLAIMS.md`, 267,170 chars).** The register carries
TWO rows named `H-EVAL-1` with unrelated semantics: *"NAVSIM-v2 pseudo-simulation … R² ≥ 0.7 …
OPEN (proposed, Lab run 001)"* AND *"~~TACTICAL/STRATEGIC never instrumented~~ … BOTH
RETRACTED"*. Run 001's own hazard log records renumbering its OTHER ids twice to dodge exactly
this ("my proposed ID was renumbered twice, to H-RANK-17, chosen atomically at insertion
time") — the eval-field id slipped through. Any tooling or prose keyed on `H-EVAL-1` is
ambiguous until the pseudo-sim row is re-registered under a fresh id. (H-DATA-1, H-RANK-17,
H-DEPLOY-1, H-OPP-1 landed cleanly — verified in the same read.)

## 3. Systemic findings (not per-file)

**S-1 — the KNOWLEDGE_BASE findings layer has been bypassed since the redesign.** Every
per-field `Research/KNOWLEDGE_BASE.md` was last written 2026-08-18/19 (the "structured along
the agents" reorg). LAB-RUN-001/002/003 banked 129 Library primaries and wrote per-field
RESULT.md packages but appended **zero** KB entries; the Deployment KB says so itself (*"This
area carried no entries through the 2026-08-18 restructure … file them here"*) and is still
empty after two quant surveys landed in packages. The constitution §2 makes "a searchable
knowledge base per field" a Lab duty. Either the daily run appends its 1-line findings, or the
KB layer should be formally re-declared as generated-from-packages — the current state is the
worst of both (a findings layer that silently stopped).

**S-2 — the intake stage is measured as unstaffed, and the debt predates the redesign.**
Four independent STATE files record the same growth: 19 verdictless packages (07-20) → 26
(08-02, oldest 24 d), plus 7 AI-side (08-03) and 9-11 unmerged `agent/*` branches. Opponent
STATE run #5: *"at 26 unverdicted packages the intake step is not a queue with a backlog, it is
an unstaffed stage … worth a program-level decision (assign a triager, or declare intake
advisory and let agents merge behind tests)"*. That decision is still not on record. The cost
is measured: the CTRV floor sat 18 days in intake while gate verdicts published CV-only.

**S-3 — no Lab steering file knows about D-CORPUS-B1.** The corpus decision that redefines the
training corpus for all four consumers (v7f / refcv3 / refav1 / refd) exists only in the
register. Every corpus-related backlog/goal in the Lab tree still optimises the parity/flagship
mix. The rewrite recommended in §1 should start from B1 and its comparability boundary.

**S-4 — naming drift.** "Benchmarks & Eval" (singular), "Opponent Analyzer", "Tools & DevEnv",
"Production & Optimization" all survive in headers and the router; the binding names are the
five fields of TANITAD_PROGRAMME.md §2 + the plural "Benchmarks & Evals" (PI 2026-08-27).
Byte-count-identical renames hid these; verify renames by CONTENT.

## 4. What this audit did NOT do

- It did not rewrite, archive, or delete any audited file (per the commissioning brief).
- It did not re-verify the closure state of every 2026-08-02/03 escalation harvested into
  LAB_BACKLOG.md — those rows are marked "verify-first" where closure is plausible.
- Domain KBs and the three Opponent catalogs were verdicted on metadata + spot-reads, not
  line-by-line (they are append-only logs/databases, low contradiction risk, high volume).

*Evidence class of this document: MEASURED (file mtimes via two independent listings; git -1
per file; verbatim quotes from API-read content) + INHERITED where a register row is cited as
the comparison baseline (the register is the live source; nothing here re-adjudicates it).*
