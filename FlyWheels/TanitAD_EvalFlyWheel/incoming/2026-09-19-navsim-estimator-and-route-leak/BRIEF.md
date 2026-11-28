# BRIEF E3 — Settle the two NavSim gates that keep every number un-intervalled and un-interpretable

**From** EvalFlyWheel orchestrator, 2026-09-19 (session commissioned by the PI in chat).
**Schema** `Project Steering/TANITAD_PROGRAMME.md` §3. **Package** = this folder.

## Operating rules (binding)

1. STAGE, NEVER PUSH. `git add` every deliverable into the working tree when you finish — EXACT FILE
   PATHS ONLY, never a directory (the shared index holds other agents' staged work). Do NOT
   `git commit`, do NOT `git push`, do NOT switch branches. Verify: a NEW file must appear in
   `git ls-files --stage -- <path>`; a MODIFIED tracked file needs its index blob to equal
   `git hash-object <path>` (both 40 chars, else INCONCLUSIVE). Re-verify at the END of your turn.
2. END WITH A DELIVERABLE MANIFEST — every artifact and WHERE it lives. Mark single-location items.
3. ESCALATE INTEGRATION in your report's headline. Never write "please merge" into a README.
4. QUOTE ONLY PRIMARY SOURCES (registry / raw JSON / banked PDFs by library key / source code at a
   pinned commit with file:line).
5. FAIL LOUD, REPORT HONESTLY. UNVERIFIED beats a guess. Evidence class on EVERY claim: MEASURED ·
   PUBLISHED · PUBLISHED-CODE (repo@sha file:line) · INHERITED · ESTIMATED · HYPOTHESIS.
- Research banking: cite a paper ⇒ bank it with `python tools/kb_add.py … --cited-by "<your RESULT>"`
  (into `TanitAD Research Lab/Library/`; "Research Hub" is DEAD). Verify with `--verify`.
- Keys.txt is git-ignored — never commit, print or copy tokens.

## Environment (MEASURED 2026-09-19)

Repo `D:\Projects\TanitAD` (⛔ never `G:`). TanitAD venv `C:\Users\Admin\venvs\tanitad\Scripts\python.exe`
with `PYTHONPATH=D:/Projects/TanitAD/stack` (assert `tanitad.__file__` is on D:). NavSim devkit
`C:\Users\Admin\navsim\devkit` @ 0a380a9; nuplan-devkit `C:\Users\Admin\navsim\nuplan-devkit`; NavSim venv
`C:\Users\Admin\navsim\venv\Scripts\python.exe` (3.9.25); data `C:\Users\Admin\navsim\data\openscene\…`
(OpenScene TEST metadata + warmup_two_stage). ⛔ GPU occupied by a live training — CPU only; don't touch
PIDs 16996/21724. cp1252 console → `PYTHONIOENCODING=utf-8`. Web: WebFetch/WebSearch (pin commits).

## Why

`products/P7-TanitEval/CRITERIA_REGISTRY.json → benchmarks.navsim` carries two gates that block every
NavSim number from being decision-grade:
- `navsim.estimator_unit` — *UNRESOLVED*: our episode-cluster bootstrap does not transfer (NavSim
  resamples scene tokens, and scenes OVERLAP — docs/splits.md), so the only admissible interval today
  is `{status: UNAVAILABLE}`. Every navhard number would ship without an interval.
- `navsim.route_leak_check` — *UNVERIFIED* whether NavSim's `driving_command` / nuPlan's
  `route_roadblock_ids` is derived from the expert's own future path. Our own refcv4b nav token is
  explicitly *"oracle, provenance ego-future"* (its `config.json`), so the answer decides whether a
  command-conditioned NavSim number measures route-following or an echo (the programme's route-echo
  defect: a route head once scored 1.0000 as a bijection of its own input).

## Part A — the estimator (priority 1)

1. **MEASURE the cluster structure** (zero download): from
   `…\devkit\navsim\planning\script\config\common\train_test_split\scene_filter\{navhard_two_stage,
   navtest,warmup_two_stage,private_test_hard_two_stage}.yaml` (+ the non-scene-filter split yamls) and
   the local OpenScene test metadata. Per split: n tokens/scenes (stage 1 and stage 2 separately for
   two-stage), n distinct `log_name`, n cities/maps, scenes per log (min/median/max), and the measured
   **temporal overlap** of scenes within a log. How do stage-2 synthetic scenes map to stage-1 originals
   and to logs? Print every count; a count of 0 from a file you could not read is not an absence — pair
   each with a read control.
2. **Pre-register the estimator** (in SPEC.md, before any number): cluster unit (e.g. `log_name`,
   justified by the overlap measurement), minimum cluster count (find **RG-14** in the registry /
   `products/P7-TanitEval/RELEASE_GATE.md`), the **paired** form for two arms on identical scenes,
   B = 2000. ⚠️ EPDMS is an aggregate over two stages — read `run_pdm_score.py` (and the two-stage
   aggregation code it calls) and make the resample reproduce the SAME aggregation; a bootstrap of the
   wrong aggregate is precise about the wrong thing (the `overlapping_holdout_se` class).
3. **Implement** it in `taniteval/adapters/navsim.py` or a new `taniteval/adapters/navsim_ci.py`
   (reuse `taniteval/ci.py`), with tests: (a) ANALYTIC targets written as LITERALS — identical per-scene
   scores ⇒ zero-width interval exactly; a two-cluster fixture with known means ⇒ known bootstrap mean;
   (b) a MUTATION that reintroduces scene-token (non-clustered) resampling and must go RED on an
   overlapping fixture (it yields a narrower interval); (c) refuses n_clusters < the floor.
4. **Registry + checker**: move the gate from UNRESOLVED to the settled rule (bump `version`, add a
   `changelog` entry), and update `tools/criteria_check.py` + `tools/tests/test_criteria_check.py` so a
   log-cluster interval with n ≥ floor PASSES and scene-token resampling / episode-cluster CIs / n <
   floor still FAIL. Run the relevant suites (`pytest -q tools/tests`, the taniteval tests — find the
   invocation) and report counts. ⚠️ A skip is not a pass — read skip reasons.

## Part B — the route leak (priority 2)

Determine AT SOURCE whether `driving_command` (and `route_roadblock_ids`) derive from the expert's
future path. At least two INDEPENDENT probes (a second probe = a different mechanism, not the same
command twice):
- (i) the local devkit — where `driving_command` is read or created (`navsim/common/dataclasses.py`,
  scene loaders, feature builders) — PUBLISHED-CODE @0a380a9 with file:line;
- (ii) the code that PRODUCED the metadata pickles — OpenScene's nuPlan conversion (github
  `OpenDriveLab/OpenScene`, pin a commit, raw source) — and nuPlan's route provenance (nuplan-devkit
  locally; the nuPlan DB scenario/route tables);
- (iii) an EMPIRICAL probe on local data: for every warmup / test scene with a future, compare
  `driving_command` with the logged future ego path (heading change / lateral displacement at several
  horizons); report the agreement matrix with n. If the command is a deterministic function of the
  future (e.g. ≥ 99 % agreement with a threshold rule), it is ego-future-derived.
Verdict: LEAK / NO LEAK / PARTIAL, evidence class, and what it means for command-conditioned arms. Put
the machine-readable result where `taniteval/adapters/navsim.py::route_leak_check` can consume it
(keep its API backward compatible).

## Ownership

You exclusively own `taniteval/adapters/navsim.py`, `products/P7-TanitEval/CRITERIA_REGISTRY.json`,
`tools/criteria_check.py`, `tools/tests/test_criteria_check.py` in this session — sibling streams E1/E2
READ navsim.py (keep existing function signatures working), and stream E5 (nuScenes) will hand you a
proposed registry/checker addition via its package; do not wait for it. Do NOT edit `LEADERBOARD.md`
(E4) or `Project Steering/GOALS_AND_CLAIMS.md` — put proposed register rows (H-/D- ids, verbatim text)
in your RESULT.md; the orchestrator lands them.

## Deliverables (this folder)

`SPEC.md` (pre-registered estimator) · `PLAN.md` · `tests/` (or the test files placed beside the code
they test, listed here) · `code/` (probes) · `raw/` (split census JSON, overlap measurement, leak
agreement matrix, pytest output) · `RESULT.md` · `COMMS.md`. Stage exact paths; manifest at the end.
