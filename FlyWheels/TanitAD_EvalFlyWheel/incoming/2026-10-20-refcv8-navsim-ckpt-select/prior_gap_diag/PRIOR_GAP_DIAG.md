# PRIOR_GAP_DIAG: why refcv8's shipped stage scores 10.4 PDMS below its own prior

**POST-HOC / DESCRIPTIVE / decides nothing.** NavSim v1 open-loop PDM scorer, navtrain selection sample (2,089 tokens, 30 logs, all Las Vegas), refcv8-r101-s0 EMA-48,852 = ONE training seed; inference seeds s0 (primary) / s1 (replicate). PDMS x100. Intervals: paired log-cluster bootstrap (30 clusters, B 2000, seed 7) = "another draw of LOGS", not another training or inference run. Numbers: `PRIOR_GAP_DIAG.json` (n, n_logs, CI per cell). Measured scorer formula: PDMS = NC x DAC x (5 TTC + 2 C + 5 EP)/12 (DDC is reported, not scored).

**Controls.** PRIOR-PRIOR = 0, CI [0,0]; means 63.395 / 62.735 / 73.838 / 67.571 / 15.798 = the brief; formula reproduces every scorer score (2e-16); seam poses == poses the scorer received (0.0); independent pandas re-derivation of 24 headline numbers PASS and goes RED on a mutated JSON. **One control FAILS as written:** token-shuffled pairing does not widen the log-cluster CI (11.8 -> 9.6; that estimator sees only per-log sums, token r(RT,PRIOR) = 0.145); it widens under a token-iid bootstrap (4.6 -> 4.8), so pairing carries token signal. Reported, not hidden.

## Answers
**Q1 Where.** R8_RT - PRIOR = -10.44 [-16.63, -4.80] (s1 -11.10); RT - LEGAL = +0.005 [-0.83, +0.78], so the shipped re-timing is neutral and the gap sits in the model's residual.
* **Command is the cut:** straight n=1444 **-19.64 [-25.4, -14.9]** (130 % of the net gap); left n=483 +2.64 [-7.5, +11.6]; right n=162 **+32.5 [+22.9, +40.7]** (PRIOR 32.1 -> RT 64.6). Band [2,8.33) n=1273 -12.7 [-20.6, -4.5] holds 74 % only because straights live there; >=13.89 has n=11 (UNDERPOWERED).
* **Failure type (exclusive, additive):** RT lost the NC/DAC multiplier on 434 tokens **-18.65 [-22.7, -15.1]** (NC alone 267 tok, -11.47 = 110 % of net); RT fixed one on 215 tokens +9.32; graded loss with multipliers intact -3.81; graded gain +2.70. Per-subscore flip rows (NC/DAC/TTC/EP) overlap (EP is zeroed on every NC=0/DAC=0 token) and must not be added.

**Q2 Plan error.** Not one direction. Of 956 losing tokens: 485 slower than the prior, 323 faster, 71 steer differently, 77 near-identical; vs the human at 4 s (>1 m): 464 slower, 407 faster; the 434 multiplier-losers split 218 faster / 202 slower than the human. A 4 s progress that deviates >~15 % from the prior's loses -19 to -30 per token in BOTH directions (n=1226); progress that agrees (n=863) **gains +11.7 [+6.7, +17.0]**; path deviation is net **+29.1** (n=339; right turns +50.5). RT is farther from the human than the prior (ADE +0.55 m [0.45, 0.67]) and closer in only 34 % of tokens. Lead-agent geometry and the anchor table are not on this machine: "accelerates into the lead" vs "brakes and is hit" is UNVERIFIED.

**Q3 Concentration.** Broad but clustered. 24/30 logs negative (s1 24/30; 15/19 with n>=30); leave-one-log-out -12.1..-9.0; top-5 logs = 64 % of the net gap on 27.5 % of tokens (cluster permutation p = 0.0005). Worst 10 % of tokens = 96 % of the net gap [60 %, 205 %] and 44 % of the gross loss [37 %, 53 %]; the 433 collapse-to-zero tokens (21 %) are 83 % of the gross loss.

## Q4 Ranked levers (share of the -10.44; post-hoc, none pre-registered)
| # | lever | measured | status |
|---|---|---|---|
| 1 | **Command gate**: PRIOR on `straight`, RT on left/right (command is admissible) | straight cell = 130 % of the gap. Hybrid **76.97 = PRIOR +3.13 [+0.48, +5.71]**; s1 +2.91 [+0.16, +5.56]; 6-fold log-grouped hold-out +3.13 (s1 +2.63) | offline-exact; 1 of 3 cells |
| 2 | **Longitudinal cap**: stop RT out-driving the prior's progress | the only measured longitudinal intervention (RT = LEGAL with progress pulled toward the prior): **+4.06 [+2.7, +5.4]** where LEGAL out-drives the prior (n=642; s1 +4.21), **-4.18 [-7.0, -1.9]** where it under-drives (n=633): consistent with the prior's progress being a ceiling (HYPOTHESIS). Fallback when 4 s progress deviates from the prior's: held-out **78.67 = PRIOR +4.84 [+2.60, +7.69]**, s1 +4.20 | gate offline-exact but best of 22 triggers; path-preserving cap NOT scored |
| 3 | Residual scale-down alpha; lateral-only vs longitudinal-only | not computable offline (no navtrain cache/map here) | **needs Thor: 18 arms below** |
| - | per-token best-of-two (not computable at inference) | 85.86 = PRIOR +12.0 [+9.4, +14.6] | ceiling for any selector |

Reading: the learned stage helps where the prior cannot turn (right turns) and where its path adds something, and hurts on straights by perturbing the speed profile.

## Thor queue (not run here; Master Mind schedules). `thor_seams/` + `jobs_prior_gap_levers.jsonl` (18 jobs, ~347 s each, `OMP_NUM_THREADS<=6`, `PYTHONHASHSEED=1`; Thor dirs in the jobs file are placeholders)
The builder reproduces banked RT knots/poses and LEGAL poses to 0.0 (PRIOR poses 7.6e-6). Priority: `R8PATH_PRIORPROG` and `PRIORPATH_R8PROG` = the 2x2 attribution against PRIOR 73.84 / LEGAL 63.39; `R8PATH_MINPROG` (cap) with `R8PATH_MAXPROG` as the must-be-worse control; `ALPHA025/050/075`; `R8PATH_HALFPROG`, `PRIORPATH_RTPROG`; then s1. **Reading rule fixed before scoring:** R8PATH_PRIORPROG >= LEGAL + 5 and PRIORPATH_R8PROG within 2 of PRIOR => loss is longitudinal; PRIORPATH_R8PROG <= PRIOR - 5 and R8PATH_PRIORPROG < LEGAL + 2 => path; MINPROG must beat LEGAL and MAXPROG, else lever 2 is dead.

**Caveats.** Cells are post-hoc cuts and failure classes are outcome-defined; the s0-s1 headline difference (0.66) is the inference-noise scale here; every gate needs a pre-registered re-test on other logs or navtest before it is a result.
