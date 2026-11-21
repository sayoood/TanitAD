# TanitAD programme status — 2026-10-11

*Master Mind report, Sunday 11 October 2026. Published copy: https://claude.ai/artifact/ByC38TkXn8QUnpsUitHTFW (private).*

The token limit stopped agent work for six days. This report gives the cause, the countermeasures
and the compaction fix, then where each stream stands and what to do next, in order.

**Evidence classes:** MEASURED = ours, with an artifact path · INHERITED = another session, not
re-verified · ESTIMATED. **Tiers:** T0 = teacher-forced, diagnostic only · NAVSIM = v1.1 open-loop
pseudo-sim. **No number here is T1 closed-loop.** Thor paths are under `/home/nvidia/`.

## Summary

| Item | State |
|---|---|
| REFe SFT-4 | **ADOPT** +3.04 PDMS [+1.21, +4.87] |
| WP-B conditioning | **FAIL** — deranged inputs cost nothing |
| R1 head arms | **FAIL** — controllability 0.355 vs 0.95 |
| P-MAP map head | **FAIL** — 0 of 4 bars, controls valid |
| WP-RL | **PAUSED** — step 159 / 2000 |
| Thor disk | **5.1 G free** of 937 G |

There is one clear win, REFe SFT-4. It is inherited, open loop and single seed. WP-B and R1 fail
with the same diagnosis: **the refcv8 planner does not obey its conditioning**. Deranging its route
and control inputs costs nothing, and STOP controllability is 0.000. One gated control path attacks
both. Operational blockers bind before scientific ones: no GPU spend, a full disk, and the launch
gate.

## 01 What happened: the token burn

The weekly limit ran out on 5 October at about 09:00 and reset on 11 October at 00:00. Jobs already
running on Thor finished unattended, and section 04 reads their results.
*MEASURED · `Project Steering/TOKEN_BURN_ANALYSIS_2026-10-11.md` (landed 4d93721)*

| Cost driver | Share of cost |
|---|---|
| Turns at context above 300k | 92 % |
| Cache reads | 61 % |
| Idle cache re-writes (1,545 events, 5-min TTL) | 29 % |
| Output tokens | 4 % |
| Subagents (vs 28 % main session) | 72 % |
| Opus | 96 % |

Largest consumers: R1 19.6 % (658k context), WP-B 18.2 % (537k), WP-D 10.6 %, and two Master Mind
sessions at 7.9 % and 7.5 %. The shares overlap, because each row is a different cut of the same
spend.

**Root cause.** Long-lived Opus subagents held 500 to 650k contexts and polled while idle. Every
poll re-read the whole context, and every cache expiry paid to rebuild it. The spend went on
re-reading context, not on producing work.

## 02 Countermeasures P1 to P8

| # | Measure | Status |
|---|---|---|
| P1 | Agents hand off and exit instead of polling | LANDED (4d93721) |
| P2 | Agent context cap of about 200k | LANDED |
| P3 | Sonnet as the default subagent model | ACTIVE at user level; project `.claude/settings.json` lands with this document |
| P4 | One Master Mind session, compacted at about 300k | LANDED, then superseded by the PI's 500k choice (cost in section 03) |
| P5 | Agent hand-backs of at most 300 words | LANDED |
| P6 | `CLAUDE.md` cut from 25.4k to 13.5k tokens, traps moved to `TRAPS.md` | LANDED (4d93721 + 6949265) |
| P7 | Amend "at least 5 parallel streams" to "short-lived, scoped agents" | **NEEDS PI** — it amends a binding PI rule |
| P8 | Daily token meter, `tools/token_meter/token_meter.py --days 2` | LANDED, MANUAL — proposed as a morning-routine step |

Also set at user level: at most 4 concurrent subagents, and spawn depth 1, so subagents cannot spawn
their own subagents.

## 03 The compaction fix

**Symptom.** The session stopped at "Kontextfenster ist voll" and asked for confirmation instead of
compacting on its own.

**Mechanism** *(MEASURED · CLI 2.1.295 source)*. Auto-compact fires at `window − 13,000`, and the
blocking limit defaults to `window − 3,000`. That leaves a runway of only 10k tokens. The CLI checks
the blocking limit before the compact threshold, so one large tool result can jump the whole runway.
The session lands in the blocked state, which shows the confirm dialog, and auto-compact never fires.

| Setting (user-level `~/.claude/settings.json`) | Value | Effect |
|---|---|---|
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | 513000 | Auto-compact fires at about 500k, the PI's threshold |
| `CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE` | 900000 | Blocking starts at 900k, which leaves a runway of about 400k |

The settings are read at session start, so the fix applies to new sessions. The current session
needs one manual compaction.

**Cost** *(ESTIMATED)*. Turns above 300k carried 92 % of the burn. With compaction at 500k instead of
300k, a turn in that band re-reads up to about 1.7 times more context. P1, P2 and P5 keep subagents
small, so this applies to the Master Mind session only. If the meter shows that session dominating
again, lower the window to about 313k.

## 04 Where every stream stands

### WP-B: refcv8 route and control conditioning — FAIL

*MEASURED · T0 · `refcv8_wpb/arms/score_wpb.json` · spec `c952d4b4…e04f` · n = 4,634 windows (2,317
turn, 43 turn episodes) · 2 seeds · paired episode-cluster bootstrap, which answers "another draw of
episodes" only*

**The model ignores its conditioning.** The regression arms T1d and T2d derange the conditioning
inputs, and both fail B_ROUTE and B_CTRL on every seed. Their pooled controllability:

- T1 − T1d: s0 +0.0006 [−0.0010, +0.0023], s1 +0.0012 [−0.0005, +0.0029]. Both are zero.
- T2 − T2d: s0 −0.0141 [−0.0275, −0.0017], s1 −0.0149 [−0.0281, −0.0024]. Deranged inputs score
  slightly better.

| Family | T0 baseline, seed 0 |
|---|---|
| ADE / FDE | scored 2.4689 · turn 3.1989 · straight 1.8857 / FDE 8.7192 |
| Longitudinal | speed MAE@2s 0.3155 · along-track −0.4964 m · distance-keeping UNAVAILABLE (no lead tracks) |
| Lateral | heading 0.2321 rad · cross-track 4.4164 m · curvature 0.0070 · yaw-rate 0.0363 |
| Tactical | turn dir 0.9059 [0.8421, 0.9557] · h15 0.5395 · lat3 0.7864 vs majority 0.6344 · lon 0.5817 vs 0.3962 |
| Controllability | TURN_L 0.274 · TURN_R 0.433 · LANE_KEEP 0.421 · STOP 0.000 · pooled 0.3205 |
| Strategic | N/A, the strategic layer is OFF (PI ruling R5, 2026-09-27) |

**Gaps.** The `p7_smoke` check fails X2 and X4. The S3 no-compile run logged only steps 0 and 1. The
ladder ran only S0, and on `cb8289b` instead of `903dd51`. The scorer wrapper's PYTHONPATH fix is
still owed.

**Next lever.** Make the conditioning non-bypassable: gated control injection that is exactly zero
on straights, plus an explicit controllability loss. This needs GPU.

### R1: refcv8 head arms — FAIL (H3 best)

*MEASURED · T0 · `refcv8_r1/arms/score.json` · SPEC_R1_A2 `23c407b6…`*

H3 improves turns, ΔADE_turn [−0.488, −0.068], and the scored set, ΔADE_scored [−0.219, −0.006]. It
regresses straights, [+0.014, +0.123]. ADE_scored is 2.3102 and 2.2959 on the two seeds, against
2.4244 for H0. Controllability peaks at 0.355 (H5) against a 0.95 bar.

**Next lever.** Gated H3: keep the turn gain, and stop the straight-road regression with the same
zero-on-straights gate as WP-B. One arm plus a replicate seed.

### WP-D: perception probes — P-MAP FAIL, PBOX UNPROVEN

*MEASURED · T0-type offline map eval · `refcv8_wpd/out/pmap_verdict_2026-10-11.json` · prereg
`PREREG_WPD_PROBES.md` `c054190b…` · frozen refcv7-50,400 trunk · 3,000 steps*

| Arm | Lever | Result vs bar |
|---|---|---|
| PM1 | stride-4 features | lane IoU2 20–40 m +0.0182, against +0.03 (replicate floor 0.012) — FAIL |
| PM2 | FPN | edge recall R10 0–20 m **−0.0297**, a regression, against +0.05 — FAIL |
| PM3 | line loss | edge IoU2 0–20 m +0.0025, against +0.03 — FAIL |
| PM4 | all three combined | edge IoU2 0–20 m +0.0145 [0.0088, 0.0200], about half the bar · lane IoU2 20–40 m +0.0235 · edge R10 −0.0304 — FAIL |

All three must-fail controls failed as required: R-s4zero, R-s4shuf, and R-gtshift (lane IoU0
−0.2088). The instrument works, so the misses are real. The PM0r replicate differs from PM0 only in
its seed, yet reads "separated" on 3 keys. That is a second rig confirming H-ESTIM-SEED-1: a
separated interval without a replicate is not a lever effect.

**PBOX** *(single seed · `record.json` per arm)*. PB3 reaches AP 0.2590 against PB0 0.2286, but the
query-shuffle control reaches 0.2527. PB1 fails.

**Next levers, ranked by measured effect:** train PM4 longer, with a replicate seed; fix the FPN
edge-recall regression; fine-tune the trunk. Every head lever moved IoU by about +0.01 to +0.02,
against bars of +0.03 to +0.05, which points to the frozen trunk as the ceiling. All three need GPU.

### Other streams

| Stream | State | Class, tier, source |
|---|---|---|
| REFe SFT-4 | **ADOPT** PDMS 85.61 → 88.65, Δ +3.04 [+1.21, +4.87] on 923 confirmation tokens (43-log cluster bootstrap). Speed MAE 0.857 → 0.749, cross-track 0.325 → 0.276, tactical κ lon 0.513 → 0.563. One seed. The recipe is not read yet, and a 12,146-token confirmation is registered. | INHERITED · NAVSIM · `eval/RESULT_SFT4.md` |
| NavSim R7_VMAXOFF | navtest PDMS 70.20, against 61.82 for STOP. navhard EPDMS 19.87, against 29.85 for STOP: it **loses** to doing nothing. | INHERITED · NAVSIM · `HANDOFF_MASTERMIND_2026-10-11.md` |
| WP-RL | **PAUSED** at step 159 of 2000 since 5 October. Its resident processes caused two host OOMs. | MEASURED · `refcv7_post/rl/PAUSED_BY_MASTER_MIND.txt` |
| D6 P4 drivable gate | **PASS**, but thin: +0.0266 with CI lower bound +0.0002. The P1′ write-up is owed. | INHERITED · `D6_navsim_failure_anatomy/` |
| v9 labels | Pushed to a private HF repo, 18 of 18 files OK, 0 bad | MEASURED · hf_push_v9_thor log |
| refcv7 battery | **UNKNOWN** — the milestone-7 logs are unread. | `2026-09-28-refcv7-standard-tests/` |
| Thor disk | 884 G of 937 G used. Largest: data 571 G, sam3map 38 G, tanit-astra-research 33 G. | MEASURED · df + du |

## 05 Next steps, in priority order

| # | Step | Owner | Blocked on |
|---|---|---|---|
| 1 | Free Thor disk: run `cleanup_thor_A.sh`, then `_B.sh`. Decide separately on tanit-astra-research (33 G). | **PI** | none |
| 2 | Lift or scope the no-GPU-spend ruling for the gated-conditioning arms. Thor is idle. | **PI** | none |
| 3 | WP-B repair: fix p7_smoke X2/X4, land the wrapper fix, rerun S3, then run the ladder on 903dd51 (L1, L2, L5, L4b, L4, L3). | MM | Ladder needs step 2 |
| 4 | Gated-conditioning arm, shared by WP-B and R1 gated H3: one arm plus a replicate seed, under a new prereg. | MM | Steps 1 and 2, plus the launch gate |
| 5 | WP-D: PB3 replicate, a long PM4 run with a replicate, and a trunk fine-tune arm. | WP-D | Steps 1 and 2 |
| 6 | Resume WP-RL, running alone on Thor. | MM | Step 1 |
| 7 | Read the REFe SFT-4 recipe and the 12,146-token confirmation. | REFe | none |
| 8 | Write up D6 P1′, and read the refcv7 battery logs. | MM | none |
| 9 | Open PI decisions 8, 11, 2, 9, 10, plus P7. | **PI** | none |
| 10 | Automate the token meter (P8). | MM | none |

---

Each number above carries its evidence class, eval tier and artifact path. Interval estimates are
paired episode-cluster bootstraps unless stated otherwise. None of them answers whether another
training run or inference run would give the same result.
