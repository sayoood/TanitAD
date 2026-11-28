# DATA-SCALE read-path pre-test (MM, 2026-11-28): PASS

**Fixture:** the AMENDMENT 1 plumbing read (`plumb35/read`, lever ds_ep; cv5m heads on the PARENT's 35 logs; NULL-CV). It is a plumbing fixture, and **no DATA-SCALE number is read here**.

**Step 1: `run_fam_chain_ds.sh` with SMOKE=50, arms DEP0 and DEP1.** This was its first run in the parameterised form, through `ds_fam_pretest.sh`, from the tip export of 15170090 with CODE.sha256 OK.
- All seven steps returned rc 0: convert, score, families, tactical_s0, tactical_s1, strategic_s0 and strategic_s1.
- Total time 09:4xZ to 09:46Z. The chain wrote `fam.DONE`.

**Step 2: `ds_read.py confirm --table`** on that smoke output (written to `plumb35/confirm_pretest/`).
- rc 0, and `VERDICT_DSEP.json` was written.
- CTRL-T2: max_abs **0.0** for DEP0, DEP1 and C1pick on the 50 scored tokens. Each arm has 12,096 tokens missing, as a smoke should.
- The outcome reads INVALID only because of those missing rows. That is the correct refusal for a 50-token pass.
- All four families are present: LONGITUDINAL_LATERAL, TACTICAL_command, TACTICAL_selection_quality and STRATEGIC.

**Result:** every stage of `ds_read_chain.sh` has now run on real data. The read (bbef7a7a), the families chain and confirm all ran. The decision run on the 185-log arms is no longer the first run of any step.
