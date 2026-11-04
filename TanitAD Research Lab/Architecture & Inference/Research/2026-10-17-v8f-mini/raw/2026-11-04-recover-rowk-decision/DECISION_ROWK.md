# PI decision item: the next lever after addendum 3 row K (RECOVER-NEXT, 2026-11-04)

**Why this exists.** RAPR-P01's read (landed `05f9dea`, `raw/2026-11-04-recover-next-raprp01/RESULTS.md`) matched addendum 3 row K:
the guards fail at both seeds, so the one-offset-per-window recovery recipe is closed at the laws and doses measured (bank law at
p 0.3 and 0.1, a-priori law at p 0.3 and 0.1). Row K commits the next arm to a PI decision between (a) the level-from-image
auxiliary and (b) a per-tick deviation recipe, with the Master Mind bringing both with costs. Neither starts without the PI's
decision and box-2 go.

## What any next lever has to reach (ESTIMATED, post hoc, `d2_closure_need.py` / `.json` in this folder)

Over the five recovery arms already read (R, RTL, RP01, RAPR, RAPR-P01; T1-kin), the composite's D2 row on the M1 crawl share
(XC − P10) is close to a line in the arm's own M1 move (X − C0): slope 1.097 / 1.146, r 0.997 / 0.932, n 5 (s0 / s1). Holding the
interval half-width at its mean over the five arms, the composite stops being separated-worse on M1 when the arm closes about
**0.68 (s0) / 0.60 (s1) of the M1 excess**, against the largest closure measured so far, RAPR's 0.4447 / 0.3175 (an
extrapolation of ×1.54 / ×1.88 beyond the fitted range). Bringing the composite's point to zero would need about 1.04 / 0.95,
outside any range we can quote. Not a registered rule: it ranks levers, it decides nothing.

The only mechanism measured near that range is the per-tick logged speed (HLVL, §7.35.2 of report v37, zero-training oracle):
share closed 0.599 to 0.633. One logged speed held per window (LEVELPROBE 1 and 2, §7.35.4) closes none of it and harms.

## The two options

| | (a) level-from-image auxiliary | (b) per-tick deviation recipe |
|---|---|---|
| what | a head regressing the logged speed from the current frame tokens, fed back to O as a feature at every tick (card row 4; architecture change) | the recovery fine-tune offsets the ego history by a deviation that changes from tick to tick, not one constant per window (a new trainer lever, its own gate entry); inference unchanged |
| proposed form | a head on the current frame tokens, loss on the logged speed, its output appended to O's ego input | **(b1, preferred)** empirical: per-tick own-minus-logged trajectories over the history window from the same TRAIN own-state bank (it already stores the own pose history `[n, hmax, 3]`); (b2) a-priori ramp law as the fallback |
| evidence for | the image carries some speed information: probe on the 608 eval windows with v0 < 6 m/s, RMSE 1.416 m/s, R² 0.489 | the per-tick oracle closes 0.599 to 0.633; §7.36.5 [H]: a constant offset teaches O to plan faster after any slow history, which is what RAPR's longitudinal harm looks like |
| evidence against | on the 313 M1 windows of that probe R² 0.117, and 85.1 % of the M1 windows (v0 ≥ 6 m/s) have no saved features, so it is untested where the excess lives; one held logged level per window HARMS at 4 of 4 cells (LEVELPROBE 1), and under the frame-freeze control a single-frame regressor returns a held level | untested; RAPR shows that a larger deviation buys closure with longitudinal harm, and a per-tick law may do the same |
| inference inputs | vision plus measured ego history (admissible; the head reads frames, not the situation classifier) | unchanged (training-only augmentation) |
| engineering (dev-box CPU, 0 GPU) | about 1 to 1.5 days: head, loss, feedback path, trainer flags with gate entries, G-LIVE gradient checks, a reader rollout mode that feeds the predicted speed per tick, and a frame-freeze reading for a frame-derived input | about 3 to 4 hours: `--recover-dev-shape {constant, pertick}` in `v8f_recover.py` (per-tick table from the bank, per-frame offsets, pose history re-integrated), trainer flag and gate entry, wiring test with a non-zero gradient and a zero control, mutation proof that `constant` reproduces today's law bit for bit, reader `ARM_SPEC` with a known answer, chain tags |
| box 2 | 2 runs, at least 1.3 h (more if the head trains from scratch) | 2 runs, about 1.3 h (MEASURED for RAPR-P01: 78 min including both gates) |
| read (dev-box CPU) | about 30 min plus the new rollout mode | about 30 min (MEASURED for RAPR-P01: 20:10 to 20:40) |
| P(counted DOMINANT), ESTIMATED | about 0.04 | about 0.10 |

## Recommendation

**(b), form b1.** It targets the one mechanism measured near the needed closure, reuses the whole recovery rig (trainer, gate,
reader, chain) and costs about a quarter of (a)'s engineering. Before any run, an addendum 4 freezes its law, its argv (RAPR's
frozen argv with the law and shape tokens changed, checked by token count), its outcome rows and their next arms; the read is the
card's section 3 unchanged. If (b) fails its guards, (a) stays available with this evidence attached.

What this does NOT do: it starts nothing. Implementation of (b) begins on the PI's choice; the box-2 runs need a separate go.
