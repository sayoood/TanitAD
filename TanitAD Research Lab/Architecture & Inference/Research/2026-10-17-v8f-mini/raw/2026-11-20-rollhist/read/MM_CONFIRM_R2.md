# MM confirmation of the R2 reading choices (2026-11-21, about 01:50Z UTC; before any RH or B0 bank, run or number)

These are readings of the frozen text of Amendment 2 (sha256 15004dc9), not changes to it.

1. **Straddle flip on every named cell in every pair (P1, P2, P4): CONFIRMED.** Item 3 is a counterfactual on the name. However many cells it tests, it fires only when the name changes.
2. **G1 carries no G-T0 condition: CONFIRMED.** The card's PARTIAL row has none. G-T0 failing at both seeds is HARMS, which comes first by precedence.
3. **G2 ignores P4: CONFIRMED.** G2 covers both REPAIRS rows (P4 failing and P4 holding).
4. **P2 "at both seeds" is read from the pooled pair families, with each cell needing both seeds on one side: CONFIRMED.** That is the card's cell rule in 3.2: "both seed means agree".
5. **SEED-DISAGREE from opposite-sign seeds beyond the floor stays, placed after HARMS: CONFIRMED.** It is the first clause of the card's SEED-DISAGREE row, in the card's precedence.
6. **Open, work item:** a one-seed (PROVISIONAL) read prints STRATEGIC as NOT-DECIDABLE until a stage-A tick-2 floor exists. The source is the UE_s0.A and UE_s1.A R0C10XHb20 dumps (c-EXEC plus the refs chain), read by the frozen ADD66 functions. The MM wires it after `ZZ ROLLHIST-REFS DONE ZZ`. The two-seed read of record takes its STRATEGIC floor from the RH and B0 seeds and does not need this.
