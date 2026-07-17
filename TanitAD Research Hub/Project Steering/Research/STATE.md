# STATE — Project Steering

LAST_RUN: 2026-07-17 (fifth orchestrator report — weekly synthesis 2026-W32; committed to main)
QUALITY: full (G-S1 met — gate-status table D1–D9 + measured KPI trends: **honest kinematic floor
0.056 m@1s** (CTRV, 26,132 anchors) → flagship 6.44 m = ~115× floor; post-reset REF-A **2.14 m plateau**
(frozen-DINO ceiling), flagship **2.34 m@5k** rising (beats REF-A@30k at 1s every stratum); REF-B
**8,750/30k healthy** (ETA ~Jul-23); H15 imagination edge **1.35–2.25 ms = 8.4 % params / 12.7–17 %
tick**; 15.07 ms tick. G-S2 met — proposals constitution-safe. Health: **5/5 weekday agents ran +
committed measured work** (tools/data Jul-15 catch-up, arch/bench/opp Jul-15/17); prod-opt Sat not in
window (next Jul-18). Recurring D-026 debt: unmerged branches, stale main-tip STATE.md, uncommitted
bench/opponent work, monitor blind to pod2/pod3. Intake: verdict write-back skipped chronically; 3
genuinely-untriaged (h15-logging-fidelity, cosmos-robustness, cosmos-DD-loader). Schedule 9/9 enabled+fired.)

## LAST REPORT
`Project Steering/Progress Reports/2026-W32.md`. **The week's pivot = the speed/scale RESET (2026-07-14):**
none of the 3 arms fed `v0` (current ego-speed) as input → each tried to infer absolute speed from vision,
which a frozen encoder can't (speed-probe R² 0.61). Fix = `v0` as 3rd action channel → REF-A **3.73 → 2.14 m**
held-out, speed-decodability **0.61 → 0.965 R²**. All 3 arms **restarted from scratch** with speed+jerk+aux-accel.
This **supersedes the entire W31 pre-reset narrative** (the 14.2-vs-7.5 confounded head-to-head is now void).
**Honest floor shipped** (bench, `e8fca8e`): 0.056 m@1s → neither arm beats the trivial baseline open-loop yet.
Verdict gated on **flagship @30k** (~Jul-19–23). Proposals: **P1** ride flagship→30k, gate vs floor+REF-A
(re-gate D2 post-reset); **P2** commit to closed-loop (open-loop ADE ⊥ DS, CV near-unbeatable) — Sayed
pre-approve the graphics CARLA pod for D4–D6 + real D8; **P3** monitoring+integration bundle (retarget
pod-monitor at all 3 arms, merge stranded branches, build D-026 guardrail).
Open for Sayed: pod-monitor retarget (3 d pending); graphics-pod GO; D-021/D-022 defaults hold; REF-B envelope unused.

## HANDOFF
Next run: read this STATE, then the six discipline `Research/STATE.md` `QUALITY` lines + `LEADERBOARD.md`
(repo-root `Benchmarks & Eval/`, NOT the hub folder) + `RESOURCE_LEDGER.md` + intake `*/incoming/*/INTAKE.md`
verdicts + `list_scheduled_tasks`. Then write `Project Steering/Progress Reports/YYYY-Www.md`.
