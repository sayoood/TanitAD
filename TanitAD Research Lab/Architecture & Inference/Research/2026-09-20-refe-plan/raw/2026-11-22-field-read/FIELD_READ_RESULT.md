# Field-loss arm F vs its control C on P7: full-navtest read (MM, 2026-11-22)

**Card.** `PK/eval/PREREG_AUX_FIELD_LOSS_P7.md`. F trains P7 for 2,000 steps with `--w-field 2.0`. C trains the same init, bank, steps and seed 0 with `--w-field 0.0`.

**How the read was done.**
- Read chain: `field_read_chain.sh`, mode `full`, finished 2026-11-22 00:43Z, rc 0. Files here: `VERDICT.txt`, `field_read.json`.
- Tier: T1, NAVSIM v1.1 PDMS. This is the open-loop pseudo-simulation of the executed, LQR-tracked plan.
- Population: all 12,146 navtest tokens in 136 logs.
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927. It answers *another draw of logs* only.
  - Inference variance does not apply: the planner is deterministic, asserted by the self-read control.
  - Training variance is the replicate pair C-seed / F-seed, which has not run yet. So no statement here is a lever claim.
- Controls, all PASS: CTRL-SET, CTRL-CSV (F and C, max |row − CSV| ≤ 7.8e-11), CTRL-SEAM, CTRL-ID (F and C).

## Levels (MEASURED)

Mean over 12,146 tokens, ×100, from `refe_stage2_p7field_{f,c}.csv`, sha256 `60674017…` (F) and `3d3d2936…` (C):

| arm | PDMS | NC | DAC | DDC | TTC | EP | C |
|---|---|---|---|---|---|---|---|
| F | **91.1508** | 98.6498 | 98.0817 | 96.9043 | 94.9942 | 86.9820 | 100.0000 |
| C | **91.2352** | 98.8762 | 97.8758 | 96.4803 | 95.2248 | 87.0811 | 99.9918 |

Both arms are at P7's level (91.0724). The best REFe stays C1 at 92.5548. All of them are below the goal of 94.55.

## Endpoints, F − C

| | point | 95 % interval |
|---|---|---|
| **E1** PDMS (tracked; the decision endpoint) | **−0.0844** | [−0.3771, +0.1955] |
| E2 DAC-zero share, pp | −0.2058 | [−0.4413, +0.0400] |
| E3 zeroed-token share, pp | −0.0082 | [−0.3012, +0.3016] |
| M1, C3 tokens (n = 1,058 in 123 logs; failure-enriched, so these are NOT rates): DAC-compliant share of the 64-proposal fan, pp | **+9.4474** | [8.0722, 10.8551] |
| M1 raw, `navsim_dac_violation(mode='raw')` | +7.8686 | [6.9990, 8.7370] |
| M2 pick dDAC on C3, simulated | +1.5123 | [−0.1015, 2.9204] |

Sub-scores, F − C:

| sub-score | tracked | transparent (no tracker) |
|---|---|---|
| PDMS (E1) | −0.0844 [−0.3771, +0.1955] | **−0.7275 [−1.0912, −0.3597]** |
| NC | −0.2264 [−0.4411, −0.0297] | −0.0329 [−0.1594, +0.0897] |
| DAC | +0.2058 [−0.0400, +0.4413] | −0.5516 [−0.9209, −0.1744] |
| DDC | +0.4240 [0.1898, 0.6782] | +0.3252 [0.0794, 0.6041] |
| TTC | −0.2305 [−0.5570, +0.0912] | +0.1482 [−0.1569, +0.4507] |
| EP | −0.0991 [−0.3810, +0.1703] | −0.6551 [−1.0220, −0.2928] |
| C | +0.0082 [0.0000, 0.0279] | −1.6549 [−2.8931, −0.4300] |

In the transparent frame (the picked plan scored without the tracker), F is separated worse than C on PDMS, DAC, EP and comfort. The tracked frame is the card's decision frame, so this is reported, not gated.

Adverse checks:
- A1 (EP): clear.
- A2 (along-track ratio 0.9995 [0.9977, 1.0012]): clear.
- A3 (fan spread ratio 0.9806 [0.9757, 0.9855], bar hi < 0.9): clear.
- K4 (E1 upper bound < 0): **not fired**.

## The four families (each on its own, never pooled; F − C in the tracked frame)

- **LONGITUDINAL:** 22 components, none separated adverse.
  - `time_gap_min_s` (+0.0197 s [0.0014, 0.0420]) and `min_ttc_s` (+0.2021 s [0.1069, 0.3047]) are read with higher = safer.
  - The inherited `fam_direction()` has no rule for these two and defaults to lower = better. Under that rule both would read adverse, so the correction flips the guard's reading on them.
- **LATERAL:** three components separated adverse:
  - yaw-rate MAE +0.0269 deg/s [0.0030, 0.0519];
  - curvature MAE +0.00014 1/m [0.00005, 0.00024];
  - last-pose heading error.
  - Heading MAE and cross-track MAE do not separate. The longitudinal/lateral guard therefore reads **FAIL**.
- **TACTICAL:** command adherence was computed. Selection quality F − C is −0.0844 [−0.3771, +0.1955]. A random pick from F's fan beats a random pick from C's fan by +8.8441 [7.8316, 9.9431]. So F's fan is better, and the scorer does not turn that into a better pick.
- **STRATEGIC:** reported in full (`out/strategic_family_F_C.json`). It does not gate (PI ruling 2026-11-16).

## Verdict and the committed next step

- **NOT PROVEN** under the card's rule: E1's interval contains 0, and neither the REFUTED clause nor the TRADE clause holds.
- The lateral guard failure would block ADOPT even if E1 separated.
- The card's "Outcomes committed in advance" table gives the NOT PROVEN branch as: *the replicate, C-seed first; more logs are not the answer; if the DAC-zero share did not move while the raw-plan fan cleanliness (M1) did, route to the scorer arm.*
- Both conditions of that last clause hold here:
  - The DAC-zero share did not move.
  - M1 moved by +9.4 pp.
- So the field loss makes the proposals cleaner, but the scorer's pick does not take advantage of it.

**Executed (pod, 2026-11-22 00:47:57Z):** `FIELD_REP_DECISION.txt` (here) starts with RUN. The chain `after_f3.sh` was started in mode `adopt_cs`:
1. It keeps the running C-seed arm.
2. Then the SCORER-FT weight rule and SCORER-FT Stage 1 (C, then T).
3. Then F-seed.
4. Then the C1 seed-1 replicate.

Only K4 would have dropped the replicates, and K4 did not fire.

**Next-arm candidate (not registered):** SCORER-FT on F's base rather than C1's. F's fan is cleaner, and the pick is the bottleneck. It needs its own card after the SCORER-FT C/T read.
