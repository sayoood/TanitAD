# Amendment to D-V8F-T2-C0C-1 (commit 21e3a976), Master Mind, 2026-11-06

The row says the arm's end-to-end agreement with the banked dump, 4.0e-5 m, is *inside the float32 batch-size noise of 1.1e-5 m*.
4.0e-5 m is larger than 1.1e-5 m, so that sentence is false as written.

What AGENT M MEASURED (`t2v8f/HANDOFF_M.md`, `t2v8f/C0C_ID_RESULT.json`):
* The float32 CPU forward of C0 moves by up to 1.1e-5 m between batch 1 and batch 16. That is about the size of the RAW plan's gap to the banked dump.
* The end-to-end comparison with the dump uses DECLARED tolerances, 1e-4 m on the raw plan and the prior and 1e-3 m on the composite. They were set after
  the batch-size noise was measured. The largest observed difference is 4.0e-5 m, inside the declared tolerance.
* The identity itself rests on the EXACT tiers: the arm equals the live RI tool bit for bit on 12 windows per seed, and the dump's own float32 plan
  and prior through the composite reproduce its plan on 7,197 of 7,197 rows.

Nothing else in the row changes. Root-cause class: an inequality written between two different quantities; `RETRACTION_LOG.md`, 2026-11-06.
