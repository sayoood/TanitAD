# MM ruling on Amendment 3 section 4: "that quantity" is ONE quantity common to both chassis. Frozen before any OPTB3 read data

**Status:** ruled by the Master Mind (MM) on 2026-11-19, before any number of the OPTB3 read exists. The read chain (`F:/TanitAD-scratch/optb3_read/`) was launched at 11:45:35Z and was still waiting at its first memory gate, with no evidence, plan or read file written (`out/state/` empty).

## The ambiguity
`AMEND_OPTB3_TARGET.md` (sha256 cb4f943f) section 4, TGT-GAIN:
> At BOTH chassis, FTT15 is BETTER on at least one deciding quantity and not WORSE on the other. On that quantity, FTT15 - FTT125 has its CI wholly below 0 at both chassis. MECH-RAISED holds.

The read builder's first version took "that quantity" per chassis: the attributed quantity could differ between chassis.

## The ruling
* "On that quantity ... at both chassis" names one quantity, and the condition is stated at both chassis. So **TGT-GAIN needs ONE deciding quantity** (`acc_mae` or `yaw_mae`, band 0-2 s) on which, at BOTH chassis:
  - FTT15 is BETTER, and
  - FTT15 - FTT125 has its CI wholly below 0.
  It also needs MECH-RAISED and FTT15 not WORSE on the other quantity, as written.
* If both chassis would read TGT-GAIN only on different quantities, both read **TGT-GAIN-UNATTRIBUTED**. That row's condition, "FTT15 - FTT125 is not below 0 at a chassis", holds on every quantity common to both.
* This is the stricter of the two readings and the less favourable one. The per-chassis reading is printed beside and decides nothing.
* Nothing else in the card changes: bars, floors, controls, the other rows, and CHASSIS-DISAGREE.

## Code
* `optb3_decide.py` gains `apply_mm_rule_common_quantity()`, applied to the per-chassis labels before the combined outcome. The output JSON carries the common-quantity sets and the per-chassis reading beside.
* `test_optb3_decide.py` gains `test_mm_rule_common_quantity`: different quantities downgrade, a common quantity stays TGT-GAIN, a single-chassis GAIN is untouched. 5 of 5 tests pass.
* Decide script sha256 after the change: 2a06f6da23e1028df7d4fcc57dcc2dba0f2ae4adbe66081953c5a579d961149d.
