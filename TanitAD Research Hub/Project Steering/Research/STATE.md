# STATE — Project Steering

LAST_RUN: 2026-07-11 (fourth orchestrator report — weekly synthesis 2026-W31; committed to main)
QUALITY: full (G-S1 met — gate-status table D1–D9 + measured KPI trends: main D1 probe ≈7.5 m vs
**REF-A frozen-DINO ≈14.2 m (confounded)**, step-14k D2 0.917/1.000 / D3 I4 3.05, REF-A curve
1.19→0.465/roll 0.076, main 25.75k/30k @~1000 steps/h, REF-B rev2 −0.124 % budget, 15.07 ms tick;
G-S2 met — proposals constitution-safe. Health check: **hygiene debt CLEARED** (`1285524` — Prod-Opt
int8/windowing + Opponent catalogs committed, tree clean); **Data-Eng no-show REVERSED** (fired 15:14,
semantic-datasets survey); discipline branches still empty 3rd cycle but loop carried all work, 0 loss →
de-escalated. Intake: 2 W30-uncommitted items now committed; none >3 days. Schedule: 9/9 enabled + fired.)

## LAST REPORT
`Project Steering/Progress Reports/2026-W31.md`. **All 3 architecture arms now exist** (main 86 %, REF-A
30k COMPLETE, REF-B built rev1+rev2). **First head-to-head:** REF-A frozen-DINO probes ADE@1s ≈14.2 m vs
main ≈7.5 m — frozen encoder currently WORSE but **confounded** (mean-pool adapter kills DINO spatial
tokens; comma-only vs comma+physicalai) → **NO H4 arbitration yet**. D1 route-resampled protocol now real
code (`e9b2491`). D-028 ADOPTED, D-022 gap resolved, REF-B $40 envelope confirmed in chat, H16 banked.
Proposals: P1 run 30k flagship gate tonight ~22:15 (D1 route-CI + **CV kinematic-floor baseline**), P2
de-confound REF-A (grid-adapter retrain + comma-only main re-run) → read H4 honestly, P3 Sayed unblock
pod3 quota (50→300 GB) + build session-end guardrail. **Still pending 3rd report: the step-30k verdict.**
Open for Sayed: pod3 volume expansion; REF-B training GO; D-021/D-022 defaults hold.

## HANDOFF
Next run: read this STATE, then the six discipline `Research/STATE.md` `QUALITY` lines + `LEADERBOARD.md`
(repo-root `Benchmarks & Eval/`, NOT the hub folder) + `RESOURCE_LEDGER.md` + intake `*/incoming/*/INTAKE.md`
verdicts + `list_scheduled_tasks`. Then write `Project Steering/Progress Reports/YYYY-Www.md`.
