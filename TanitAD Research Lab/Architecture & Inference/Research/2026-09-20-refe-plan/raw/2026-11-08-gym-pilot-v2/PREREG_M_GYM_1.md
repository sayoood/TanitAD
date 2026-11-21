# PREREG M-GYM-1 (written 2026-11-08 BEFORE the 20-state measurement; thresholds are the brief's, unchanged)

## Cost verdict (brief, committed)
Quantity: effective seconds per labelled state at 12 workers = (single-worker seconds to roll out all K=24 first
actions x continuation) / 12. A labelled state needs BOTH agent modes (design 2.4), so the PRIMARY figure is
log + idm summed. Per-mode figures are also reported. The single-worker time per state is the sum of timer buckets
inside one worker (scene+map build, proposals, per-arm agent construction/stepping, planning, controller, scoring);
12-worker extrapolation assumes linear scaling (6 workers are measured, then x0.5, and the measured 6-worker
wall/state is reported next to it so contention is visible).
* GO: <= 26 s/state.  REDUCE: 26-260 s/state.  BLOCKED: > 260 s/state.
Horizon: first action 0.5 s + 8.0 s PDM-Closed continuation = 17 cycles of 0.5 s (brief's literal reading; the
design's 16-replan figure would be 6 % cheaper).
Statistic: mean over the 20 states (also median and max reported). The verdict uses the MEAN.

## Regression arms (must read as known). Log-replay agents unless stated. Policy-level, full 17 cycles.
* always-slowest (min mean speed over the 4 s plan, from the full 24-fan each cycle): KNOWN iff mean q_free over
  cycles where q_free is applicable < 0.6 AND its mean return < the PDM-Closed continuation's on the same states.
* always-fastest (max mean speed each cycle): KNOWN iff (hard-event count + count of lead-present cycles with
  q_gap < 0.7) over the 20 states is strictly greater than PDM-Closed's, in the log or idm mode.
* do-nothing (stationary plan): KNOWN iff mean per-cycle reward <= 0.02 over states where the ego moves at t0.
* oracle (argmax over the 24 first actions of the full-continuation return, per mode): KNOWN iff it has no hard
  event on the 20 states (and by construction return >= every other arm).
* Determinism: same state, same arm, same mode run twice in one process and once in a fresh process: per-cycle
  rewards and q terms bit-identical (==).
A regression arm that does NOT read as known is reported as a finding about the reward/arm, not tuned away.

## Activation
Per term: fraction of (state, arm, mode, cycle) records with q < 1 ("fired"), and for q_free/q_gap also the
fraction where the term is applicable. A term with fired rate < 1 % is flagged as sparse (`D-RL-COLL-SPARSE-1`).
Hard events: rate per mode per episode.
