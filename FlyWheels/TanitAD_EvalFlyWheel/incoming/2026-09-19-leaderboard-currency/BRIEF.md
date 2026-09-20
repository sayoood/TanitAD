# BRIEF E4 — The LEADERBOARD, current for 5 October, with the External Field visible

**From** EvalFlyWheel orchestrator, 2026-09-19 (session commissioned by the PI in chat).
**Schema** `Project Steering/TANITAD_PROGRAMME.md` §3. **Package** = this folder; the page itself is
`Benchmarks & Eval/LEADERBOARD.md`.

## Operating rules (binding)

1. STAGE, NEVER PUSH. `git add` every deliverable into the working tree when you finish — EXACT FILE
   PATHS ONLY, never a directory (the shared index holds other agents' staged work). Do NOT
   `git commit`, do NOT `git push`, do NOT switch branches. Verify: a NEW file must appear in
   `git ls-files --stage -- <path>`; a MODIFIED tracked file needs its index blob to equal
   `git hash-object <path>` (both 40 chars, else INCONCLUSIVE). Re-verify at the END of your turn.
2. END WITH A DELIVERABLE MANIFEST — every artifact and WHERE it lives. Mark single-location items.
3. ESCALATE INTEGRATION in your report's headline. Never write "please merge" into a README.
4. QUOTE ONLY PRIMARY SOURCES. **This page's own rule: only two sources are quotable — the registry
   (`Project Steering/MODEL_REGISTRY.md`) and raw eval JSON.** Never a summary, changelog, weekly report,
   commit message or `PROJECT_STATE.md`. External numbers: banked PDFs by library key + table/figure.
5. FAIL LOUD, REPORT HONESTLY. UNVERIFIED beats a guess. Evidence class on EVERY number.
- Research banking: cite a paper ⇒ bank it with `python tools/kb_add.py <arxiv-id> --tag <topic> --note
  "<finding>" --cited-by "Benchmarks & Eval/LEADERBOARD.md"` (into `TanitAD Research Lab/Library/`;
  "Research Hub" is DEAD). Verify with `--verify`. An unbanked number is PUBLISHED-SECONDARY and stays
  OUT of every comparison cell.
- Keys.txt is git-ignored — never commit, print or copy tokens.

## Context

The EvalFlyWheel OWNS the leaderboard (`Project Steering/AGENT_CHARTERS.md` §5): *"must be kept current
with our own flagship solutions, our reference implementations, AND other known stacks — so our
position is always visible."* It is named in the **2026-10-05** evaluation package (`Project Steering/
Master Plan.md` §2 Phase 2: *"reproducible gate ladder, leaderboard, safety case draft"*). The last
dated amendment is 2026-09-03; rows for refcv4b / refcv5-v2 were added later (commits `75c6520`,
`6b4f85f`). There is **no External Field section**, so the position against the field is invisible.
Repo `D:\Projects\TanitAD` (⛔ never `G:`); read big files in ranges (the page is ~1,700 lines).

## Work (priority order)

1. **Currency audit (table in RESULT.md).** Every arm with an eval result in `MODEL_REGISTRY.md`, every
   raw eval JSON in `taniteval/results/`, and every eval package since 2026-09-03 under
   `TanitAD Research Lab/**/` and `FlyWheels/**/` (e.g. refcv3/refcv4b/refcv5-v2 finals, refav1 T1,
   DD-v2 RL heldout dumps, the refcv6 dev-box W1 four-family chain of 2026-09-19 — find the artifacts;
   `git log --since=2026-09-03 --name-only` helps locate them) vs the rows on the page: arm · registry §
   · raw artifact · on page? · stale/contradicted? ⚠️ A registry row can itself be stale — when the
   raw JSON contradicts it, the raw JSON wins and the conflict is REPORTED, not silently fixed.
2. **Add/refresh missing rows** — only from registry or raw JSON. Every row: tier (T0/T1/T2), **loop**
   (OPEN/CLOSED per the page's §0.8 — PI 2026-09-02: a planner feeding its own predictor is still OPEN),
   corpus, estimator + CI (paired episode-cluster bootstrap; ⛔ never `overlapping_holdout_se`),
   evidence class, artifact path, and the floors it is read against (`ha` / `ha0` / `ha0_ext` where the
   artifact has them). A missing family is `NOT MEASURED — <reason>, n=<n>`, never a blank.
3. **W-18 — the loop column** where it is a pure lookup on the tier (zero GPU). ⛔ Do NOT rewrite the 13
   "closed-loop" mislabel sites listed in §0.8 unless `git log` shows no other stream owns that
   correction; if unowned, list them as a work item in §12 rather than editing silently.
4. **NEW SECTION — "External field: where TanitAD stands"** (place it right after §0, before §1a):
   one sub-table PER PROTOCOL, never merged (the registry's `navsim.cross_protocol` gate):
   - **NavSim v2 — navhard_two_stage EPDMS (OFFICIAL column, harness SHA stated)**: DrivoR 56.3
     (camera-only, +TOAD test-time search), DriveFuture 55.5 (learned), PDM-Closed 56.6 (privileged),
     GTRS-D 45.0, iPad 34.7 (+TOAD 49.8), RAP-DINO 39.6 — INHERITED from
     `TanitAD Research Lab/Opponent Analysis/Research/2026-08-28-navsim-v2-camera-lane/RESULT.md`
     (library keys `2601.05083` DrivoR, `2606.07170` TOAD, `2601.22032` Drive-JEPA): **re-read each
     number from the banked PDF** and record the table number; any you cannot re-read stays
     INHERITED and out of comparison cells. State modality + ego-status use per row (the leaderboard
     server records no modality — the authors' claim, not the server's).
   - **NavSim v1 — navtest PDMS, perception-free front-camera-only ladder** (separate column): LAW 83.8 /
     World4Drive 85.1 / Epona 86.1 / Drive-JEPA 89.0 (registry `benchmarks.navsim.GATE_modality_label.
     comparable_ladder_perception_free_front_only`, from `2601.22032` Table 1).
   - **NavSim v2 — navtest single-stage EPDMS (papers' protocol, NOT comparable to navhard)**:
     Transfuser 76.7 → Hydra-MDP++ 81.4 → DriveSuprim 87.1 → Drive-JEPA 87.8; DiffusionDrive-V2 85.5.
   - **nuScenes open-loop planning** — cited only, ⛔ inadmissible as a TanitAD criterion (H-EVAL-6,
     D-BENCH-PORT): every row carries its averaging convention (ST-P3 vs UniAD) and whether ego status
     is consumed. Stream E5 writes `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/
     raw/nuscenes_external_rows.md` — use it if present when you get there; otherwise build the rows
     from banked primaries yourself.
   - **Bench2Drive / CARLA** (GO-conditional per D-BENCH-PORT) if banked primaries exist.
   - **Our row in each table**: `NOT MEASURED — pending <package>` naming the exact artifact that will
     fill it: streams E1/E2 (`…/2026-09-19-navsim-warmup-reference-epdms/`, `…/2026-09-19-navsim-refcv4b-bridge/`)
     produce **warmup_two_stage** numbers — a DIFFERENT protocol from navhard, so they go in their own
     `EPDMS_v2_warmup_two_stage` row, never in the navhard column; navhard is pending the PI's download OK.
5. **A top-of-page "5 October position" block** (≤ 15 lines + one table): what TanitAD can claim today,
   at which tier and loop, against which floor — and what it cannot claim yet. Honest over flattering:
   the registry's own verdicts (e.g. refcv4b/refcv5-v2 do not beat hold-action at T1) are the headline
   if they are the truth.
6. Update the page header's amendment banner (date, what changed, what was NOT re-measured).

## Editing discipline (the page is shared and long)

Before editing: `git log -5 --format='%h %ad %s' -- "Benchmarks & Eval/LEADERBOARD.md"` and compare the
file with `git show HEAD:"Benchmarks & Eval/LEADERBOARD.md"`; if the worktree differs from HEAD, a
sibling is mid-edit — STOP and report instead of overwriting. Make SURGICAL edits (Edit tool), never a
wholesale rewrite; preserve every existing number unless a raw artifact contradicts it (then record the
correction visibly, §11/§12 style). Re-check `git log` for the file immediately before staging.
You exclusively own `LEADERBOARD.md` this session; do NOT edit `MODEL_REGISTRY.md`,
`GOALS_AND_CLAIMS.md` or `CRITERIA_REGISTRY.json` — list proposed changes to them in RESULT.md.

## Deliverables

`LEADERBOARD.md` (edited, staged) · this folder: `SPEC.md` (what "current" means, the audit scope) ·
`PLAN.md` · `raw/currency_audit.json` (+ `.md`) · `raw/external_field_sources.json` (every external
number → library key, table, page) · `RESULT.md` · `COMMS.md`. Stage exact paths; manifest at the end.
