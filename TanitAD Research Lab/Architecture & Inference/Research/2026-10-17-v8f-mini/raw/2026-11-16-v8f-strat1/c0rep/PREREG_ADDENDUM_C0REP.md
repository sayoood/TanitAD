# PREREG addendum C0REP: the STRAT1 same-seed training floor (written 2026-11-17 ~19:45Z, before either arm trains)

**Authority.**

- **The lever:** `PREREG_ADDENDUM_S1D.md`, "Next step per label". For PARTIAL / REMAINS, the next lever is a STRAT1 TAC-LAT floor from at least 2 same-seed replicate pairs, i.e. C0 replicates at s10 and s11.
- **The PI's go:** "2: queue s10 ans s11" (2026-11-17).

**Arms.** Two new trainings, called `C0_s10r` and `C0_s11r`.

- Each is byte-identical in argv to the banked `C0_s10` / `C0_s11`: the same checker (`strat1_check.py` on tree `new4` = a4357e79, seed keys `s10` / `s11`), inputs, 3,000 steps, init and seed.
- Only the run directory differs: `/root/strat1/runs_c0rep/C0_s10`, `/root/strat1/runs_c0rep/C0_s11`.
- Each arm needs its own gate token and post check, exactly as the N1 s10/s11 chain.
- The trainer with DAgger is expected to be non-deterministic at a fixed seed (HYPOTHESIS; the A_s0 retrain on 2026-11-13 differed in 260 of 284 tensors on another rig). This run measures it.

**Reads (same reader, constants, windows and estimator as the SEEDS / N1S10 / S1D reads).**

- T1-kin, 141 episodes / 2,399 windows per arm.
- Paired episode-cluster bootstrap, B 2,000, seed 0.
- The 24 driving cells, plus STRATEGIC route following and route setting as printed by those reads.

1. **Same-seed pairs:** `C0_s10r − C0_s10` and `C0_s11r − C0_s11`, on every cell (delta, CI, separated).
2. **Floor:** `F_rep(cell) = max(|C0_s10r − C0_s10|, |C0_s11r − C0_s11|)`, from the full-set point deltas. The floor of the re-read is `F2(cell) = max(F(cell), F_rep(cell))`, where `F(cell) = |C0_rep − C0_s0|` is the card's floor.
3. **Re-read:** COST4 and GAIN4 for S1 − C0, N1 − C0 and S1d − C0 (and S1d − S1), at seeds {0, 1, 10, 11}. Use the card's count rule exactly, with F2 in place of F. Nothing else changes.

**Labels (TAC-LAT is the deciding cell; every cell is printed).**

| label | rule | meaning, and the next step |
|---|---|---|
| **COST-HOLDS** | TAC-LAT is still COST4_S1 under F2 | S1's TAC-LAT cost is beyond same-seed training noise. The S1D label is re-read under F2 (CURED / PARTIAL / REMAINS as written). If it reads REMAINS, the next lever is the N2 card's attribution with the floor in hand. |
| **COST-WITHIN-FLOOR** | TAC-LAT is COST4_S1 under F, not under F2 | S1's TAC-LAT cost is not distinguishable from same-seed training noise. S1 has no measured driving cost on that cell. The S1D question (does the detach cure it) is not decidable and is closed as such: S1 stays the route-text arm. |
| **DETERMINISTIC-AT-SEED** | both same-seed pairs read exactly 0.0 on all 24 cells | The pod trainer is bit-deterministic at a fixed seed, so a same-seed replicate cannot measure training noise. F_rep = 0, F2 = F, and COST4 stands as read. The floor lever then needs a different-seed design, a new addendum. |

**Controls.**

- K-RE: every recomputed pair that exists in a banked result equals it exactly.
- K1: the self-pair reads exactly 0.0.
- K-ARGV: each replicate's argv equals its banked arm's argv token for token, after the run root `/root/strat1/runs_c0rep` is replaced by `/root/strat1/runs` (the `--out` path is the only intended difference). On a mismatch the pair is left out before any number is read.

**Evidence class and tier.** MEASURED, T1-kin. The interval answers "another draw of episodes". The floor answers "another training run at the same seed", with 2 pairs: a weak floor, stated as such.
