# SPEC_WPB_LADDER — AMENDMENT A1 (Master Mind, 2026-10-04 ~23:52 Europe/Berlin)

Registered before any ladder arm has run: no tiny-rung number exists. It adds two OPTIONAL-lever rungs to
`SPEC_WPB_LADDER.md` (sha256 `e1b6aff7…`). Every clause of the base SPEC and its §10 applies unchanged: the rig, the
seeds, the floor F, the reading order, and "no arm is read before its rung has finished".

## Why
* **L4b.** The inherited 4-way speed encoder discards 44 % of N2's information: 70 and 80 km/h fall into the 100 bin on
  20.8 % of train rows. MEASURED, WP-B `raw/x3_leak.json`.
* **L5.** D6 P1' (MEASURED, all controls PASS): on navhard DAC-zero scenes the fan holds a fully clean candidate 0.890 of
  the time. The imitation-only selector picks one 0.065 of the time (random 0.320). X1 carries no drivable term.

## Arms (run after the base SPEC's arms, in this order)

| arm | = | role |
|---|---|---|
| **V-R8-E8** | V-R8 + `--r8-speed-enc8` | L4b treatment |
| **V-R8-E8-roll** | V-R8-E8 + `--r8-roll-speed-input` | L4b deliberate regression |
| **V-R8-DRV** | V-R8 + `--r8-critic-drivable` + `--w-r8-drivable w`. w is the value WP-B declares in its L2 hand-over, fixed before this arm runs and never tuned on a ladder number. | L5 treatment |
| **V-R8-DRV-roll** | V-R8-DRV + `--r8-roll-targets map` (the critic's label and the map target rolled by one row) | L5 deliberate regression |

The floor F for both rungs is the L1 floor: the larger of |V0r − V0| and |V-R8r − V-R8|, per metric. No new replicate arm.

## Measures (sampler seeds 0 and 1, eval139, per family, never pooled)
* L4b: speed MAE over 2–6 s, and ceiling compliance of the emitted plan (the share of windows above the fed ceiling).
* L5: the **off-drivable rate of the emitted plan**: the share of windows in which any in-range footprint corner (NavSim
  DAC geometry, t 0.5–4.0 s) lies on a SAM3 `map_fine` cell that is not drivable. Unseen and out-of-range corners carry
  no evidence. It is reported beside the base SPEC's route measures.

## Bars (literals; the R ≥ 2 rule of §10.1 applies, because both are optional levers)
* **B-E8 (L4b):** on BOTH sampler seeds:
  * the speed MAE 2–6 s of V-R8-E8 is better than V-R8's, separated, with R ≥ 2;
  * no family is worse beyond F;
  * V-R8-E8-roll's speed MAE gain over V-R8 is ≤ F.
* **B-DRV (L5):** on BOTH sampler seeds:
  * V-R8-DRV's off-drivable rate is lower than V-R8's, separated, with R ≥ 2;
  * turn direction-correct and heading-15 are each no worse than V-R8 by more than F;
  * V-R8-DRV-roll's off-drivable gain over V-R8 is ≤ F.
* A failed regression arm makes its rung NOT QUOTABLE.

## Reading
* **B-E8 clears** ⇒ `--r8-speed-enc8` enters the refcv8 launch argv.
* **B-DRV clears** ⇒ `--r8-critic-drivable` enters the launch argv, and refcv8's NavSim rows report the critic ON.
* **Either fails** ⇒ it stays out, and the RESULT names the next lever:
  * for L4b, the speed-profile term of the listwise target;
  * for L5, D6 P4's inference-time gate if P4 passes, else WP-D's map levers.
