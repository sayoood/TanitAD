# MM ruling: block 3 step count for BL_s0 (2026-11-17 ~21:40Z)

**Finding (MEASURED by the block-3 build agent, checker imported read-only from the pod's `study/`).** C-STEPS requires total steps = 4 x S from the arm's own dry record. For BL_s0, a B-set arm, that is 4 x 4,791 = 19,164. The checker therefore refuses BL_s0 at 61,568.

**What the card says.** `CARD_DATASCALE_TRAIN.md` (frozen, PI go 2026-11-16) is explicit:
- line 75: BL_s0 = "the same total as UE_s0 (`--steps`)";
- line 87: "UE_s0 (4 union epochs) is read against BL_s0 (B set at the same step count)";
- line 195: Δ_ep = BL_s0 − B_s0, "fixed data, about 3.2x steps".

The checker's rule contradicts the card for BL. It is right only for UE.

**Ruling: the checker implements the card, so the checker is fixed. No bar, arm or dose changes.**
1. UE_s0: total steps = 4 x S_U from U_s0's dry record (S_U = 15,392, so 61,568).
2. BL_s0: total steps = UE_s0's total. The target is read from the same U_s0 dry record, not from the B record, and must equal 61,568.
3. Regression arms that must read REFUSED:
   - BL_s0 at 19,164;
   - BL_s0 at 61,569;
   - UE_s0 at 61,569;
   - a BL argv whose target is computed from the B record.
4. The checker's sha256 changes, and block-3 tokens bind the new sha. Block 1 and 2 tokens are spent; their arms are trained and banked. They are not re-minted, and a fresh G-SUITE run is not needed for a study-checker change unless the checker is a G-SUITE file. The build agent states which case holds.
5. The change, its diff and the regression evidence are banked with the block-3 build. The fix lands before the first block-3 mint.
