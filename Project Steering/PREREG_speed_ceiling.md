# PRE-REGISTRATION — how much of the oracle-vs-produced gap is the SPEED channel?

**Written and committed BEFORE the arm ran.** 2026-07-29. Task #44 step 2.

## The question

The oracle-vs-produced `ade_0_2s` gap is **+0.2140 [+0.1602, +0.2759]** (paired, 881 windows /
40 episodes). **Route is already measured at ≤ 2.6 % of it** — fixing its threshold lifted balanced
route accuracy 0.4242 → 0.5493 yet moved paired ADE only +0.0022 [−0.0008, +0.0055].

**What accounts for the remaining ~97 %?** The speed channels are the standing suspect: `vt_band`
exact agreement 0.1725, `tspeed_5s` RMSE 4.4545 m/s (~16 km/h), and the gap is longitudinal
(paired `long_abs_2s` +0.4260 vs `lat_abs_2s` +0.0274).

## Design

A **MIXED** arm, newly supported by `--oracle-channels`: `vt_band` and `vt_speed` taken from the
oracle, **every other channel produced**. Compared against the fully-produced arm on the **same 881
windows** (eid + ground truth verified identical), **paired episode-cluster bootstrap**, B=2000,
seed 0. `overlapping_holdout_se` used nowhere. Host: pod3 or eval (both idle) — **v5 is untouched.**

⛔ **This arm is NOT a deployable number.** It is fed a future-derived quantity by construction. It
exists to ATTRIBUTE the gap, never to score the model. The harness records
`oracle_channels_substituted` in the run so no artifact can be mistaken for a deployable one.

## Both outcomes, committed in advance

- **OUTCOME A — speed carries the gap.** Paired Δ vs the produced arm is **negative and separated**,
  and recovers a large share of the 0.2140. ⇒ `tspeed_5s` is confirmed as the binding constraint and
  is worth real training effort. The ceiling this measures is the **maximum** such effort could buy.
- **OUTCOME B — speed does NOT carry the gap.** Δ is not separated, or recovers only a small share.
  ⇒ **neither route nor speed explains the oracle advantage**, and the goal-head framing is wrong:
  the advantage would then have to live in the *interaction* of the channels, or in `route_graded`'s
  continuous value rather than the discrete class, or the oracle is helping through a path we have
  not modelled. That would be a genuinely surprising and important negative — it would mean the
  programme should stop investing in per-channel goal quality altogether.

⚠️ **Neither outcome is a failure.** Outcome B redirects more sharply than A does.

⚠️ **Route's lesson binds:** a channel-metric improvement is not evidence of a trajectory
improvement. Only the paired ADE delta speaks here — and this design measures exactly that.

## Pre-committed reporting

The paired Δ on `ade_0_2s` is the primary. Reported alongside it, whatever they show:
`fde@2s`, the Frenet split (`long_abs_2s`, `lat_abs_2s`), and the share of the 0.2140 recovered.
