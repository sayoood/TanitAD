# MM verification of the BEHAV package (2026-11-07 ~20:45Z)

1. **Freeze hashes.** `sha256sum` of `BEHAVIOUR_SCORECARD.md` = `fac6c7f150eb026d...` and of `PREREG_P6_SCORER_SPEED.md` = `8e8aa3cac3762ae4...`. Both match their `.sha256` files. They are landed before P6's CHAIN_END and before any P6 output has been read on the C3 tokens.
2. **Independent re-run** of `u400_reference.py` (dev box, CPU, numpy 1.26.4), output in `u400_reference.rerun_MM.stdout.txt`:
   - R1 (weighted CRUISE slow share), R2 tie-correct (average ranks), R3 (without the lane factor) and the reproduction gate on P1's banked values all came out **bit-identical** to the agent's run.
   - **The R2 "legacy" block did NOT reproduce:** NC true −0.039 → +0.034, TTC −0.049 → +0.018, DAC −0.039 → +0.012, EP +0.408 → +0.430. Same code, same data. Its argsort-of-argsort ranks break the ties by an order that differs between processes. This is the defect the agent named, now shown directly.
   ⇒ The P1 "true" column is retracted (RETRACTION_LOG 2026-11-07 ~20:50). The tie-correct values replace it.
3. **Tier ruling (MM).** The TanitScenarioFlyWheel closed loop re-renders the camera views (Cosmos on thor3), so `EVAL_DOCTRINE.md` places it at **T2 (re-perception sim)**. The scorecard's interim stamp "T2-class closed-loop sim" is confirmed as **T2**. The brief's "T1" was the Master Mind's own error.
