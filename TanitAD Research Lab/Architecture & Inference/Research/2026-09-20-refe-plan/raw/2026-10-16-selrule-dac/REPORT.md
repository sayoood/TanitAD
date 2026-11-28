# SELRULE-DAC fit: re-weighting the scorer's OWN DAC/NC heads cannot reach the zero tail. The DAC errors are CONFIDENT.

**Master Mind, 2026-10-15 ~23:50 Berlin.** The fit was built and run by a Sonnet agent. The MM verified it:
- `MANIFEST.sha256`: 13/13 OK;
- `FIT.json` sha256 `44d6d2c1…`;
- `test_selrule.py`: 67 checks, including 4 deliberate-regression arms that go RED.

The prereg is `eval/PREREG_SELRULE_DAC.md`, IN FORCE before any number. Amendment 1 (A1–A7) is in `PREREG_SELRULE_DAC.amended.md`. It was written before any pick was computed, and its sections 1–5 are byte-identical to the original.

## ⚠️ What the fit is (A1, a premise correction the MM accepts)
- The RL-val dumps come from **u400** (SFT-4 arm B, the RL-S1 base: `PREREG_RL_S1` OPEN-BASE "DECIDED: A") on the **pre-P2** fan. They do **not** come from P2. My prereg assumed P2, and no P2 RL-val dump exists.
- ⇒ the τ chosen here **transfers** to P2 as an assumption. The navtest confirm is therefore a **transfer read**, and it is labelled as one.

## MEASURED (u400 scorer, RL-val selection tier: 3,000 sets / 75 navtrain-held-out logs; true NAVSIM v1.1 PDMS of the pick by table lookup; paired log-cluster bootstrap, 10,000 resamples, seed 20260927; ×100)
Reference points: B = **90.84** (88 zero picks: 68 with DAC = 0, 20 with NC = 0); best-of-64 = 98.39; random = 77.47. The B-reproduction checks:
- B reproduces `rl_val_eval.run` exactly (Δ 0.0).
- It reproduces the trainer's own step-0 row to 1.3e-6 when the two are paired by rule (`FIT_CROSSCHECK.json`).
- ⚠️ The `banked_trainer_row` field inside `FIT.json` pairs the lane-rule B with the navsim_v1 trainer value, which shows a spurious 0.209. That is a labelling error in that one field. FIT.json is write-once, so the crosscheck file carries the correction.

| rule | ΔPDMS vs B [95 %] | picks changed | EP | TTC | DAC |
|---|---|---|---|---|---|
| V(0.5), V(0.6) | +0.000 | 0 | 0 | 0 | 0 |
| V(0.7) | −0.004 [−0.011, +0.000] | 1 | −0.01 | 0 | 0 |
| **V(0.8), CHOSEN** | **+0.021 [−0.001, +0.067]** | **3** | 0.00 | +0.03 | +0.03 |
| V(0.9) | +0.058 [−0.019, +0.162] | 24 | −0.03 | +0.10 | +0.10 |
| K(2) | −0.129 [−0.285, +0.042] | 754 | −0.60 | +0.20 | +0.10 |
| K(3) | −0.267 [−0.483, −0.040] | 1,163 | −1.05 | +0.30 | +0.13 |
| K(5) | −0.512 [−0.773, −0.234] | 1,568 | −1.60 | +0.27 | +0.13 |
| K(8) | −0.571 [−0.906, −0.234] | 1,808 | −1.97 | +0.37 | +0.23 |
| *O-DAC (oracle)* | *+2.129 [+1.453, +2.877]* | 882 | +2.43 | +0.20 | +1.80 |
| *O-NC (oracle)* | *+0.751 [+0.452, +1.081]* | 795 | +1.24 | +0.03 | +0.13 |
| *O-both (oracle)* | *+2.675 [+1.946, +3.453]* | 1,436 | +3.56 | +0.03 | +1.80 |

**Why V(0.8) and not V(0.9).** The committed tie rule (A4): the tied set is the candidates within 0.05 of the best mean, which is V(0.8) and V(0.9), and the tie goes to the lower τ. ⛔ Neither is separated from 0.

**The mechanism (`FIT_DIAGNOSTIC_VETO_REACH.json`).**
- A veto can only move a pick whose own p̂_DAC is below τ.
- Of B's 68 picks with true DAC = 0, the scorer's p̂_DAC has a **median of 0.93** (IQR 0.89–0.98). Only 5 sit below 0.8, and 19 below 0.9.
- ⇒ **the drivable-area failures are CONFIDENT errors.** No threshold reaches them.
- Sharpening the gates (K) moves 750–1,800 picks and pays for small TTC and DAC gains with 0.6–2.0 points of EP.

## Reading (RULE ZERO: the next lever, with its result or its blocker)
1. **Re-weighting is null on RL-val.** The fit outcome is reported as it fell: the chosen rule is not separated from B.
2. **The lever is the DAC head's REPRESENTATION, not its weighting.** A perfect DAC head is worth **+2.13** (28 % of the best-of-64 headroom), and a perfect NC head +0.75 (10 %). This ranks P6, whose trunk is trained on PDM labels including DAC, and P7.1 (a better DAC head) as the zero-tail levers. DAC comes first by 2.8×.
3. **The confirm (the ONE navtest read) still runs, unchanged, as a transfer read.**
   - It runs when `raw/2026-10-16-p2-full-table/P2FULL_DONE` appears. It costs ~10 min of CPU.
   - Command: `confirm.py --expect-fit-sha256 44d6d2c1…`.
   - Its oracle controls give the **navtest** size of the DAC-head prize on P2 itself.
   - What the MM expects is NOT PROVEN. That is a reading of the fit, not a banked prediction.
4. **The cheapest experiment that could still make selection work is a better DAC head, not a better rule.**
   - On cached P2 trunk features, that is a CPU/8 GB job, which is P7.1's own first step.
   - ⛔ **Blocker:** no cached P2 trunk-feature set exists on the dev box for RL-val or navtest. P6's trunk is the scheduled representational test (queued behind R0c).

## Families
- **LONGITUDINAL:** EP, TTC.
- **LATERAL:** DAC, DDC (in `FIT.json` `curve.*.subscore_delta_vs_B_x100`, paired CIs).
- **TACTICAL** (selection quality): random 77.47 / actual 90.84 / best-of-64 98.39.
- **STRATEGIC:** PAUSED (PI 2026-09-19).
