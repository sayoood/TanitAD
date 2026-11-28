# VERDICT: PREREAD_V8F_SPEEDSRC = DIFFUSE at both seeds (Master Mind, 2026-11-10)

Card frozen 16:15:45 (sha256 card ed4583bf, script e38730c4; pre-registered in 2664e690), full read 16:17:18-16:23:39, `out/SPEEDSRC.json` and the per-window `out/speedsrc_PARITY_DAGOACC_s{0,1}.npz`. Tier: **T1-kin first tick**, eval-141, an oracle diagnostic, **never driving performance**. Estimator: episode-cluster bootstrap, B 2,000, seed 0 ("another draw of episodes" only); in-minus-complement is an unpaired two-group difference; 17 levels, no multiplicity correction (both seeds are the guard). STRATEGIC: **PAUSED** (PI item 25). All numbers MEASURED.

Controls (both arms): TOP1 = dump `ol` 0.0, GT future = dump 0.0, `v0` = dump 0.0, IDENT = TOP1 0.0, OWN-ALL = IDENT 0.0; checkpoint sha256 67265ec1 (s0, = the SUMS pin), 90b01474 (s1); windows read 2,351 / 139 episodes, 11,702 present speed cells.

## The committed rule, applied verbatim
A stratum qualifies when it has >= 50 windows, >= 0.30 of the squared speed-error mass, mass share >= 2x window share, and an in-minus-complement interval above 0. **No stratum qualifies at either seed. Outcome: DIFFUSE.** Committed trigger: perception capacity (the LoRA arm L, resolution, data scale).

| stratum (s0 / s1) | windows | window share | mass share | ratio | in - complement (m/s)^2 | GT-SPEED-ONLY tac ADE gain (m) |
|---|---|---|---|---|---|---|
| regime STOP | 132 | 0.056 | 0.154 / 0.142 | 2.75 / 2.54 | +14.43 [+8.52, +21.66] / +12.41 [+6.56, +19.55] | -1.13 / -1.00 |
| regime DECEL | 487 | 0.207 | 0.352 / 0.361 | 1.70 / 1.74 | +6.26 [+3.38, +9.68] / +6.58 [+3.54, +9.99] | -0.52 / -0.58 |
| regime ACCEL | 594 | 0.253 | 0.259 / 0.260 | 1.02 / 1.03 | n.s. | -0.41 / -0.42 |
| regime CRUISE | 1,138 | 0.484 | 0.236 / 0.237 | 0.49 | -7.27 / -7.11 (lower) | -0.21 / -0.21 |
| goal STOP_POINT | 117 | 0.050 | 0.200 / 0.197 | 4.01 / 3.95 | +23.36 [+13.23, +33.82] / +22.67 [+13.35, +32.30] | -1.80 / -1.55 |
| goal TL | 155 | 0.066 | 0.070 / 0.063 | 1.06 / 0.95 | n.s. | -0.26 / -0.24 |
| lead (any bin) | 235-1,290 | | | 0.81-1.25 | n.s. | -0.28 to -0.60 |
| curvature HI | 314 | 0.134 | 0.191 / 0.183 | 1.43 / 1.37 | +3.62 [+0.70, +6.68] / +3.05 [+0.18, +6.10] | -0.43 / -0.56 |
| speed at t0 (any bin) | 333-1,085 | | | 0.90-1.02 | n.s. | -0.17 to -0.47 |

Mean per-window speed MSE 7.25 / 7.13 (m/s)^2. Per cell (s0): `lon_v_target_ms` 10.33, `v_end_ms` 9.85, `v_lo_ms` 6.99, `v_hi_ms` 6.12, `v_a_ms` 2.81.

## Beside, no bar (a post-hoc pattern; it is NOT promoted)
* The error leans toward **slowing down and stopping**: goal STOP_POINT holds ~4x its window share of the mass and fails only the 0.30 mass bar (0.20); regimes STOP and DECEL lean the same way (STOP + DECEL: 26 % of windows, ~0.50 of the mass, a union formed after the read). Lead agents, traffic lights and speed level carry no excess. Curvature HI leans mildly (1.4x).
* Where T's speed cells are wrong at stops, a perfect speed code buys most: -1.80 / -1.55 m tactical ADE in STOP_POINT windows against about -0.2 to -0.4 m elsewhere.
* Not a result about lead agents or traffic lights: their strata read flat here, at this tier and on this base.

## What follows (Rule Zero)
1. **The committed trigger (perception capacity):** LoRA arm L (H-V8F-LORA-1) is the lever in flight. MEASURED on the pod at this read: L_s0 step 7,950 of 19,164 after 188,331 s (23.7 s per step), so about 74 h to the end of s0 (ESTIMATED from the mean step time), then L_s1. Its read adds the CONSORC and SPEEDSRC decompositions on L_s0 / L_s1. Resolution and data scale need compute the PI has not authorised (named blockers).
2. **The cheapest lever still open, CPU, zero training:** `PREREAD_V8F_STOPDEC` (being built, pre-registered as a new question about the pattern above): in STOP_POINT windows, does T's own decision see the stop while its speed cells miss it? DECISION-SEES -> a coupling card (tie T's speed cells to its own stop decision in training); DECISION-MISSES -> a stop-cue perception card.
