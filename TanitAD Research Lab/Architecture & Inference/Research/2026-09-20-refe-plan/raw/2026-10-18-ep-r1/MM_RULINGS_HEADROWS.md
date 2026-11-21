# REFE head rows: Master Mind rulings (2026-10-19 ~05:45, BEFORE any dev read and before the navtest look)

No dev outcome has been opened for either arm, and the shared navtest look is unspent. The fit-side out-of-fold (OOF) numbers below are SELECTION-tier reads from `epr1_result.json` / `r1_result.json`. They are the reads the registrations define as fit outputs, not dev outcomes.

## A. The open questions from SRTTC-DEV (`RESULT_SRDEV.md`)

**R1 (GN1): a RED navtest C2 makes that reading VOID-C2.** The verdict is never PASS. Its numbers are reported, and the look counts as SPENT. There is no re-read.

**R2 (G2): dev C2 stays strict, with no tolerance (D7 stands).**
- Adding a tolerance now would move a bar after the arms were built.
- Disclosed risk: a chance RED burns the one dev read.

**R3 (G3): a VOID dev read CONSUMES that arm's one dev read.** The arm does not reach the navtest look. Its successor is a NEW registration.

**R4 (G4): A2.6 cannot be reached at f_max = 100 %.** There, the D-DATA branch reads as FLAT, because no larger nested size exists. Disclosed.

**R5: SR-TTC's C2 needs a gate token, exactly as EP-R1's `--stage c2` does.**
- `--no-gate-stage-ack` is NOT accepted. A shuffled-label C2 fit is a fit, and the PI's launch-gate rule (2026-09-26) binds it.
- Build a C2 stage for SR-TTC as a NEW file (the landed `srttc_r1_gate.py` stays frozen), with its own G-DVB entry and a deliberate-regression arm.

**R6: `confirm_read.py read` is THE navtest read command.** `refe_head_confirm.py read` alone skips navtest C2 and families6, so it must not be used for the look.

## B. EP-R1 at f = 10 %: TWO registered rows fire at the fit, so the dev gate is NOT read

- **E1 (grid edge):** c\* = lr 1e-3, anchor 0.1, `c_star_at_grid_edge: true` (pooled OOF concordance 0.9808).
- **D-LEAK** (PREREG_EPR1 section 8: "OOF unsafe-pick rate of the new pick exceeds P2's with the paired interval wholly above 0 on both seeds, **whatever the dev gain**"). The pick-level OOF read at c\* (row 7; paired log-cluster bootstrap, 10,000, seed 20260927):

| seed | dPDMS x100 vs P2 | d unsafe-pick rate |
|---|---|---|
| 0 | **-1.145 [-1.473, -0.840]** | **+0.0102 [+0.0065, +0.0146]** |
| 1 | **-1.174 [-1.503, -0.858]** | **+0.0101 [+0.0065, +0.0143]** |

- **Reading.** The conditional EP row ranks safe contenders better (concordance 0.98). But with violators masked out of its target it extrapolates on them, and the pick moves into unsafe proposals that the NC/DAC rows do not catch. This is the same mechanism as `D-REFE-RL1-HEADWISE`: NC is lost there although every safety head is the base scorer's (`H-REFE-RL1-NCRES`). Two arms now show it. **Raising EP preference is gated by the resolution of the safety rows.**
- **The dev gate is NOT read**, by E1's own text and by D-LEAK. EP-R1 does not reach the navtest look in this form.

**R7. Next arm: EP-R1ub, ONE new registration covering both named successors** (E1 → EP-R1b, the lr extension; D-LEAK → EP-R1u, the unsafe-proposal anchor).
- **Grid (9 configs):** lr in {1e-3, 3e-3, 1e-2} x u in {0.03, 0.1, 0.3}. Anchor fixed at c\*'s 0.1. Same epochs, folds, seeds, fit logs and f = 10 %. u is the weight of an unsafe-proposal anchor with target 0, as in the D-LEAK row.
- **Selection (fixed now, fit-side OOF only):** c\* = argmax of the mean over both seeds of the OOF pick-level dPDMS, SUBJECT TO the OOF d-unsafe-rate upper CI <= 0 on both seeds.
  - Concordance alone is NOT the criterion: at f = 10 % it rose to 0.98 while the pick got worse.
- **If no config is feasible:** the EP-R1 line closes at the fit (D-LEAK confirmed). The next lever is the safety rows' resolution: a pre-registered NC/DAC-row read on the same bank, tied to `H-REFE-RL1-NCRES`.
- **If a config is feasible:** the EP-R1 dev gate is read ONCE on it, with every other PREREG_EPR1 rule unchanged.

**R8. SR-TTC follows its registration as written.**
- Its c\* is also at the lr maximum (lr 1e-3, anchor 0.3, pooled OOF 0.70007). SR-TTC registered no edge rule, so none is added: disclosed only.
- Its dev read proceeds on R1's c\* once the R5 C2 stage exists, read once with `srttc_dev.py`. Under R3, a VOID consumes it.

**R9. The shared look (D1) still waits for both dev gates, or for a formal abandonment.** EP-R1 stays in the look only through EP-R1ub. If EP-R1ub has no feasible config, write the formal abandonment file (`refe_head_confirm.read_abandon` format), and the look may open for SR-TTC alone.
