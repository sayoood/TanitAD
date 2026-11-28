# PREREG_MAPSEL AMENDMENT 1 -- DRAFT, NOT APPLIED, NOT SEALED (needs the Master Mind's decision)

Trigger: the sealed K_HUMAN gate (PREREG_MAPSEL.md sec. 5, sha256 `e9be326e692a6402e70bf79c98097770a5cec24259f27d1d6a01d179c64a3bc9`) FAILED. The experiment STOPPED as written: no GPU lock was taken, no roll, no per-candidate map feature exists.

## Measurement (MEASURED, `result/K_HUMAN.json`, `result/K_HUMAN_alt_class_sets_DESCRIPTIVE.json`; 0 GPU; S6-compliant windows with a GT map, n = 3,198 / 136 episodes)
* HUMAN logged path (8 slots, parity with the banked S6 `g` 1.5e-5 m), same sampler, FP5 footprint, GT raster, sealed class set {0 nocls, 5 edge, 7 sidewalk} as non-drivable: **off-drivable 0.0926** (gate 0.02). Edge-contact 0.1485, lane-crossing 0.1986 (human floors for T2 / T3).
* Geometry is NOT the cause: a +-1 window shift of the GT raster does not lower the rate (0.1049 at shifts 0 and -1; 0.1199 at +1; 0.29 at -6); centre-only 0.044; every FP5 point type contributes alike.
* Cause = the GT raster's definition of "not drivable", by class (full population, share of windows with any violation of that class): **nocls (code 0) 0.0735** (72 % of violating cells; 8 of 136 episodes have >= 50 % of windows hit; the hits are in the near field, 2 % of SEEN samples at x < 10 m and 0 beyond = unlabelled holes), **edge (code 5) 0.0291**, **sidewalk (code 7) 0.0125**.
  Class sets: {5,7} 0.0291; {0,7} 0.0785; {7} 0.0125 (<= 0.02); {0} 0.0735; {5} 0.0291.

## Proposal (needs approval; the Master Mind may instead choose to stop the line)
**T1 "off-drivable" := a SEEN footprint point on a SIDEWALK cell (GT code 7); code 0 (nocls) is UNKNOWN like 255; edge (code 5) stays a Tier-2 concept.** OWN side kept symmetric: OWN non-drivable := `P^(sidewalk) >= 0.5`
(OWN-CAL: `pmin` := min over points of `1 - P^(sidewalk)`, tau re-fixed from the human paths the same way). Consequences: (1) `mapsel_roll.py` banks one more plane (`psidewalk_u8`), `mapsel_lib.masks_*`, `mapsel_score.py` and the unit tests change, so the code pins and the seal are re-issued;
(2) K_HUMAN reads 0.0125 on the full population (descriptive, already measured) and remains a STOP gate at 2 %; (3) the human edge/lane floors (0.149 / 0.199) stay in the report: Tier 2 / 3 "avoid" / "prefer" are soft by construction (a hard filter would reject 15-20 % of human paths) -- they remain lexicographic filters with fallbacks.
**Why this is not a goalpost move:** K_HUMAN is a control of the INSTRUMENT (the footprint/raster semantics), not an effect criterion; no candidate, arm or outcome number has been computed; the amended class set is disclosed here with the number that motivated it, and the gate stays at the sealed 2 %.
Rejected: relaxing the gate to ~10 % (moves the goalpost), dropping the 8-9 hole episodes (shrinks the population and selects on the instrument).

## Cost
Code + tests + dry-run + re-seal ~ 1.5 h of agent time; the GPU queue position is unchanged (nothing taken); the roll ~ 2,200-3,600 s once the lock is free.
