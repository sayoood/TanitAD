# NEXT_ARM_PLAN - REPAIR-D successor, developed on navtrain before any navtest read (draft, 1 page; nothing run)

Source: `ANATOMY_25.md` (25 navtest residual tokens, descriptive, seen-on-navtest). Rule Zero: a repair that stalls at 25 of 116 is a waypoint; this is the cheapest next lever, not a new hypothesis.

## What the anatomy points to (MEASURED, ours, builder-side T1; navtest-seen, so it ranks levers and fixes nothing)
1. **Two-sided oscillation, not the cap (largest term).** 16 of 20 non-guard tokens flip the return-vector sign across iterations while the executed path follows the shift (15 of 20 at >= 0.85). A single-direction shift cannot satisfy both boundaries at all 41 steps. More iterations help only 1 to 4 tokens (what-if, descriptive).
2. **Guard-blocked, 5 of 25 (NC / TTC / C).** These need a different certificate, not a different shift: the DAC = 1 candidate exists. Options: (a) accept DAC = 1 candidates whose NC/TTC/C loss is within a pre-registered tolerance, (b) search the neighbouring candidate set for one that is DAC = 1 and not worse, (c) leave them as the honest residual.
3. Step size: a 0.20 m margin certifies 8 of 25 at cap 10 in the what-if, which is a hint that overshoot damps the ping-pong; it may not be quoted or used to set the margin.

## Levers, in priority order
- **L1: two-sided shift.** Replace "shift along the single worst corner's return vector" by a lateral offset ramp chosen from the per-step feasible lateral interval (left and right boundary clearance of the vehicle footprint over the 41 simulated steps), with a damped step (half the correction on a sign flip) and an infeasibility flag when the interval is empty (corridor narrower than the footprint plus margin). Cost: CPU only, the same scorer, label pipeline unchanged.
- **L2: margin/cap schedule** (margin 0.05 / 0.10 / 0.20 m x cap 5 / 10 / 20) as a pure parameter sweep; cheap, and the anatomy says it is secondary to L1.
- **L3: certificate tolerance** for the (b) class; needs a PI-visible tolerance because it trades NC/TTC/C against DAC.

## Navtrain development set (no navtest token read to set any parameter)
Existing sets cannot discriminate: the TF-N4 navtrain dev set (`PK/raw/2026-11-11-teacher-fidelity/n4/`) repaired 30/30 and the 469-token set has 5 RJ5.R failures. Candidates for a set with unrepaired REPAIR-D failures, best first:
1. **The RJ5 navtrain re-roll** under `F:/tanitad_scratch/rj5nt/run_local/rows` (another job, still running: batch 0 of 17 on 2026-11-13 09:41, 198 logs). Once it has rolled enough batches, label with P5-R (lambda 3.0), score tracked, run REPAIR-D at A1 settings; the failures that remain are the dev set. INFERRED from the navtest rate (25/12,110 = 0.2 %) that ~2.5k+ tokens are needed per 5 failures; the oscillation class is rarer on navtrain (30/30 repaired), so expect the set to be small. Read the job's own manifest before use; do not touch its files.
2. Other banked navtrain arms' form-R labels (the six arms of the 469-token set, 30 failing instances, all repaired by REPAIR-D): usable as a regression set, not as a discriminating one.
3. **Synthetic stress set (cheap, recommended in addition):** take navtrain tokens, shift the teacher target laterally by a random 0.1 to 0.5 m toward a boundary to manufacture two-sided cases with a known feasible answer; the lever must recover them. This is a construction, label it so; it tests L1's mechanism, not the navtest rate.

## Experiment before any navtest read (pre-register, both outcomes committed)
Arms on the navtrain dev set: A1 REPAIR-D (control, must reproduce the banked repair count), L1, L1 + L2 sweep, a deliberate regression (sign-flipped shift, must fail) and a zero-shift identity (must be bit-identical). Metric: repaired count among DAC-0 tokens and NC/DDC/TTC/C not worse, paired log-cluster bootstrap on tokens (same estimator as TF-N4), and per-token cause class from `n4_repair_trace.py`. Bar to commit: L1 repairs >= 1 more token than A1 on the dev set with 0 certificate violations; otherwise L1 is dropped and L3 goes to the PI. Parameters (damping, margin, cap) are fixed on navtrain only; the 25 navtest tokens are read once, after the freeze. Pre-flight per Rule: reachability (flag -> loss) and launch-gate apply to any later training use; this is CPU label construction only.
