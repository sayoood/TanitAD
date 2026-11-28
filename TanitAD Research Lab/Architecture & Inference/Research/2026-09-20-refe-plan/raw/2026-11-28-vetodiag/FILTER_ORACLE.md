# FILTER-ORACLE: perfect per-proposal checker ceilings on C1's navtest table (MM, 2026-11-28)

**Status:** DESCRIPTIVE. Tier T1, on the banked harness-exact table: `full_C1_s*of2.jsonl`, where `tr` holds the tracked scores and `tp` the planned, untracked scores. Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, over 136 logs. This decides no adoption; it decides what to ask the PI.

**Rule:** on EVERY token, pick k = argmax of C1's agg32 over the proposals the checker passes, and score k with `tr`. If no proposal passes, keep the ship pick. The checks pass (argmax 12,146/12,146; mean 92.5548).

| checker | switched (better / worse) | Delta PDMS [95 % CI] | Delta EP | Delta DAC | Delta NC | realisable? |
|---|---|---|---|---|---|---|
| true tracked DAC (`tr.dac > 0`) | 199 (198 / 0) | **+1.532 [1.147, 1.974]** | +1.445 | +1.638 | ~0 | if an HD map **and** the tracker simulation (the TRK dump's `trk20`) are admissible at inference |
| true tracked NC (`tr.nc > 0`) | 140 (138 / 0) | +0.808 [0.567, 1.061] | +1.104 | -0.016 | | NO: NC scores the logged future agents |
| both | 336 (336 / 0) | +2.340 [1.927, 2.791] | +2.548 | +1.630 | | NO (reproduces VETODIAG's ceiling) |
| **planned-proposal DAC (`tp.dac > 0`)** | 309 (142 / 76) | **+0.538 [0.314, 0.781]** | +0.509 [0.288, 0.751] | +0.642 [0.417, 0.891] | **-0.066 [-0.150, -0.008]** | yes, with an HD map; NC separated worse means guard G would fail as is |

**Implications:**
- DAC accounts for 198 of the 336 zeroed picks.
- A map-based DAC check is information that C1's scorer and the SAFE-HEAD heads do not have:
  - C1's scorer: AUC 0.763 (VETODIAG);
  - the SAFE-HEAD heads: AUC 0.766 to 0.769 (HEADDIAG).
- Checking the **tracked** trajectory closes most of the gap from +0.54 to +1.53.
- **The PI decision:** is an HD map an admissible input at inference on the NAVSIM track? If it is not, the admissible form is a **vision-predicted drivable area**.
