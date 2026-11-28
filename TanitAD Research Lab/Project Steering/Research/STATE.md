# STATE — Project Steering

LAST_RUN: 2026-07-20 (sixth orchestrator report — weekly synthesis 2026-W33; branch `agent/tools-devenv-20260717` = the shared tip, main is diverged)
QUALITY: full (G-S1 met — gate-status table D1–D9 + measured KPI trends: **flagship-v1 @30k FINAL
ade_0_2s 0.4522 ± 0.031 m = the FIRST sub-floor arm** (floor 0.5005 / CTRV 0.523 / no-vision ego-status
0.5735 / CV 0.838), n=881/40 held-out eps; causal genuine-prediction **vision-effect +1.325 m CI
[1.04,1.64]**; OOD counterweight comma 0.849 vs floor 0.372 / cosmos 0.583 vs 0.358; REF-C-XL@16k 0.565,
REF-B@10k 0.826, REF-A dyn-in@30k 2.92, flagship-v2@6k 6.18; deploy tick **11.16 ms / 89.6 Hz / 1.59×**;
suite **529✓/2s in 52.16 s** measured by me on the shared tree. G-S2 met — proposals constitution-safe,
`Mission Plan.md` untouched. G-I: resource = local RTX-4060 box + read-only ssh probes of 4 pods,
~1.3 h wall-clock, **$0**; no bigger resource needed — this run reads and synthesizes, it does not train.
Health: **6/6 agents ran, zero QUALITY: partial**, 3 fired 1–2 days late (host-availability catch-up).
**🔴 Live probe found 2 of 4 GPUs idle behind DEAD trainers and the 6-hourly monitor blind to both —
4th recurrence.** D-026 debt at its worst: 9 unmerged branches + ~45 uncommitted files (+1613/−101).
Intake 19/23 verdictless (3rd report). Schedule 9/9 enabled.)

## LAST REPORT
`Project Steering/Progress Reports/2026-W33.md` (covers 2026-07-18 → 2026-07-20; short period — W32
landed Fri 07-17 and this job misfired to Monday, flagged in §0).

**The period's pivot = W32's TOP RISK CLOSED POSITIVE.** flagship-v1 @30k is the first arm below every
trivial bar (0.4522 m), and the genuine-prediction panel proves the anticipation is **read from the
scene, not extrapolated from dynamics** (mean-replacing the scene inverts the advantage). **The honest
counterweight, measured the same period:** it does not transfer — on comma2k19 and cosmos the same ckpt
loses to the kinematic floor, so it is partly distribution-fit, not yet a corpus-general world model.
**H4 is now closed NEGATIVE twice** (frozen DINO lacks generalizable yaw; feeding yaw as input still
lands 2.92 m, 94 % longitudinal). **flagship-v2 was diagnosed and restarted** (mechanism-A but exponent
−0.50 vs v1 −0.84 → grinding to 30k would land ~9× worse for 4 days of A40).

**Proposals:** **P1** restore the fleet + retarget the monitor onto live-arm auto-discovery with a
"no trainer PID while step<total" alarm (HIGH, ~1 h, $0); **P2** make the sub-floor result GENERAL —
hold the v3enc recipe but pre-register the **OOD panel** as the acceptance gate (target: beat the comma
floor on ≥35 % of windows, up from 17.5 %), attack high-speed longitudinal, re-gate D2/D3 post-reset
(HIGH, one 30k run, $0 incremental); **P3** pay the D-026 debt with the just-built `session_guard` —
merge 9 branches, commit 45 files, wire the gate in, clear 19 intake verdicts, refresh `LEADERBOARD.md`
(MED, 2–3 h).

**Open for Sayed:** (1) **the fleet is half-dead right now** — pod1 REF-B @22.6k and pod2 v3enc @1,950,
pod2 root cause is a 98 %-full overlay disk; (2) the **docker/graphics-capable GPU host** — one ask, three
unblocks (AlpaSim closed-loop D5/D6, CARLA pixels, ≥32 GB VLM-labelling slot); (3) four decisions filed
as `proposed` (REF-C DiffusionDrive redesign, v2 restart, milestone-ckpt archiving, v3 hierarchical-
planning direction); (4) the SC-13 dedup vs `agent/opponent-20260715` is still unresolved.

**Deliberate non-actions (P8).** I did not restart the two dead trainers and did not delete pod2's stale
3.36 GB `ckpt.tmp`: both are write actions on shared training infrastructure that this scheduled task does
not authorize, and a naive pod2 restart re-crashes on the same full disk. I escalated with turnkey
recovery steps instead (report §3, P1). I also did not triage the 19 verdictless intake packages — with
two trainers dead and 45 files uncommitted, that budget went to the fleet and the strand debt; it is now
a named proposal (P3) rather than a fourth repeated bullet.

## HANDOFF
Next run (Fri 2026-07-24 12:25 UTC): read this STATE, then the six discipline `Research/STATE.md`
`QUALITY` lines + `LEADERBOARD.md` (repo-root `Benchmarks & Eval/`, NOT the hub folder) +
`RESOURCE_LEDGER.md` + intake `*/incoming/*/INTAKE.md` verdicts + `list_scheduled_tasks` + a **live
read-only fleet probe** (this run's single highest-value input — the committed docs did not contain the
two dead trainers). Then write `Project Steering/Progress Reports/2026-W34.md`.

**Carry-forward checks for W34:**
- Did P1 land? (both trainers running, monitor retargeted, "no trainer PID while step<total" alarm firing)
- Did P3 land? (>2 unmerged branches at W34 = `session_guard` is advisory, not blocking → make it blocking)
- Is `LEADERBOARD.md` refreshed into metric-BEV units with the W33 arm table? (R4)
- Are D2/D3 re-gated on a post-reset ckpt, or still carrying 27k pre-reset evidence? (R6)
- Did the v3enc OOD acceptance gate get pre-registered before the run banked steps? (P2)
