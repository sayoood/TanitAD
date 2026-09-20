# BRIEF E5 — nuScenes: a protocol-verbatim open-loop planning harness, one command from a number

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
4. QUOTE ONLY PRIMARY SOURCES — banked PDFs by library key + table, or reference code at a PINNED
   commit with file:line (PUBLISHED-CODE).
5. FAIL LOUD, REPORT HONESTLY. UNVERIFIED beats a guess. Evidence class on EVERY claim.
- Research banking: cite a paper ⇒ bank it: `python tools/kb_add.py <arxiv-id> --tag nuscenes --note
  "<finding>" --cited-by "<your RESULT path>"` (into `TanitAD Research Lab/Library/`; "Research Hub" is
  DEAD). Verify with `python tools/kb_add.py --verify`.
- Keys.txt is git-ignored — never commit, print or copy tokens.

## Environment

Repo `D:\Projects\TanitAD` (⛔ never `G:`). TanitAD venv `C:\Users\Admin\venvs\tanitad\Scripts\python.exe`,
`PYTHONPATH=D:/Projects/TanitAD/stack` (assert `tanitad.__file__` is on D:). ⛔ GPU occupied by a live
training — CPU only; don't touch PIDs 16996/21724. cp1252 console → `PYTHONIOENCODING=utf-8`.
Web: WebFetch/WebSearch for reference code (pin commits) and papers.

## Context — why build it, and why it is NOT a criterion

- The EvalFlyWheel charter (`Project Steering/AGENT_CHARTERS.md` §5) makes *"NavSim, nuScenes, and peers"*
  MANDATORY for external comparability; nuScenes open-loop L2/collision is still the most-quoted
  external planning number.
- ⛔ It is **inadmissible as a TanitAD criterion** — register row **H-EVAL-6 SUPPORTED**
  (`products/P7-TanitEval/benchmarks/NUSCENES_PROTOCOL.md`: VAD derives its "high-level command" from the
  GT future at ±2 m — our route-echo defect published as SOTA; the averaging convention alone moves one
  checkpoint 0.72 → 1.22 m (+70 %) and flips the UniAD/VAD ranking; the GT human trajectory scores
  0.36–0.96 % collision) — and the PI-approved portfolio marks it **SKIP claim-bearing** (D-BENCH-PORT
  2026-08-29). It may only ever be a **cited external-comparability row**.
- ⛔ No bytes exist locally and **an agent must not register at nuscenes.org or accept its Terms**
  (`stack/tanitad/data/nuscenes.py` docstring). So the job is: make a nuScenes number one command away
  for the day a human provides the data, and measure exactly what that human step costs.

## Work (priority order)

1. Read `NUSCENES_PROTOCOL.md`, the registry block `products/P7-TanitEval/CRITERIA_REGISTRY.json →
   benchmarks.nuscenes` (tasks detection / tracking / prediction / planning), and
   `stack/tanitad/data/nuscenes.py`.
2. **Read the reference implementations AT SOURCE, pinned** (quote file:line): ST-P3's planning metric,
   UniAD's `planning_metrics` (and its L2/collision reduction), VAD's metric, and AD-MLP's evaluation
   (which re-used ST-P3's). Establish exactly: the L2 reduction per convention ("average over all
   waypoints up to t" vs "at t"), the collision definition (ego box size, BEV occupancy grid resolution
   and extent, which agents count, time alignment), which frames/samples are scored, and any filtering.
3. **Implement `taniteval/adapters/nuscenes_planning.py`**: L2 @ 1/2/3 s and collision @ 1/2/3 s in BOTH
   conventions; every output row carries its convention tag and it must be IMPOSSIBLE by API to emit an
   untagged or mixed number. Artifacts carry `claim_bearing: false` with H-EVAL-6's reason.
4. **Tests** (`taniteval/tests/test_nuscenes_planning.py` or the location the taniteval suite already
   uses — find it): analytic targets written as LITERALS (a plan offset laterally by a constant d from a
   straight GT ⇒ L2 = d exactly at every t in both conventions; a linearly growing error has a closed
   form that DIFFERS between conventions — assert both literals); a MUTATION arm that swaps the
   conventions and must go RED; a collision fixture with a known overlap / known miss; all
   dataset-free. Run the suite and report counts; a skip is not a pass.
5. **Criteria-checker guard (proposal, not an edit):** stream E3 exclusively owns
   `tools/criteria_check.py`, `tools/tests/test_criteria_check.py` and `CRITERIA_REGISTRY.json` this
   session. Write your proposed change (a rule refusing any `claim_bearing: false` nuScenes artifact
   used as a TanitAD criterion, + its test) as a ready-to-apply patch in `code/criteria_guard.patch` and
   describe it in COMMS.md — the orchestrator integrates it after E3 lands.
6. **External rows**: published nuScenes OL planning numbers (UniAD, VAD-Base/Tiny, AD-MLP, BEV-Planner,
   SparseDrive, and others you can bank) with convention, ego-status use and whether the command is
   GT-derived, each from a BANKED primary (library key + table). Write them to
   `raw/nuscenes_external_rows.md` (markdown table) — stream E4 (leaderboard) consumes that file.
7. **The measured blocker, for the PI**: which files a planning eval of OUR model needs (front camera
   keyframes of the 150 val scenes? CAN bus expansion? maps expansion? metadata), and which the
   DataFlyWheel's SAM3 paint reference needs (register row **H-SAM3-FUSION-1**: *"nuScenes val with
   maps"*). Sizes from the official AWS Open Data bucket `motional-nuscenes` via LIST/HEAD only (a
   listing is not a download — read the loader docstring's caution: reachability is not licence
   permission). The steps only a human can do (register, accept ToU). Licence caveat (CC BY-NC-SA 4.0,
   research-only, share-alike) stated.

## Ownership / do NOT

You own the new `taniteval/adapters/nuscenes_planning.py` and its test file(s) and this folder. Do NOT
edit `tools/criteria_check.py`, `tools/tests/test_criteria_check.py`, `CRITERIA_REGISTRY.json`,
`taniteval/adapters/navsim.py` (E3), `LEADERBOARD.md` (E4), or any steering file. Do NOT download
nuScenes bytes, register, or accept terms.

## Deliverables (this folder)

`SPEC.md` (the analytic targets pre-stated) · `PLAN.md` · `code/` (+ `criteria_guard.patch`) · `raw/`
(reference-code extracts with pins, `nuscenes_external_rows.md`, bucket listing sizes, pytest output) ·
`RESULT.md` · `COMMS.md`. Stage exact paths; manifest at the end.
