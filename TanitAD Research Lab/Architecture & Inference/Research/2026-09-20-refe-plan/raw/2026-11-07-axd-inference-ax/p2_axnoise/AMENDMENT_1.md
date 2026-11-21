# AMENDMENT_1 to IMPL_P2.md (P2 AXNOISE, stream YAXN, 2026-11-07)

**Status.** Written and sha256-pinned (`AMENDMENT_1.sha256`, and in `CODE_PIN_AXN.json`) BEFORE the full chain evaluated any arm (only the 4-step smoke had started; it was stopped by explicit PID and is re-run on this code). Origin: MM ruling made before any P2 decision arm was evaluated. PREREG_AXD.md (sha256 1d33a9ed...) and IMPL_P2.md are unedited.

## What changes (two things, nothing else)
1. **An added control arm.** AXON_s0 and AXON_s1 are re-composed from YAX's banked K1 decoder tensors (`YAX/run_full/arms/K1_AXON_s0|s1/final.pt`; chosen_X1 = K1 in `select.json`, the recipe the X1 AXON evals used) with `ax_ckpt_tools.py build`, gated like the AXNOISE checkpoints (K1-K5 at build; identity = G-CKPT), and run through `axn_seam.py` (A0, IDT fresh-trunk, IDTm memo, and the IDT gain arms only) on the same 923 tokens, inside the same chain, one process at a time, behind the shared lock with the MemoryError retry. No scoring (the PDMS control stays the banked `X1_AXON_s`). Each AXON checkpoint is deleted after its seam, through a literal path (`axn_common.CKPT_LITERAL`).
2. **The reference of the relative gain clauses.** IMPL_P2.md section 2/3 used u400's banked P1 gain (1.0798 [0.8391, 1.3132]) as "g_dn(IDT)" in clauses (d) and (e). That mixed the fine-tune's effect with the noise's effect. Now, at each seed: **(d)** CI-hi(g_dn(AXN_s)) < CI-lo(g_dn(AXON_s in-run)); **(e)** g_dn(AXN_s) <= 0.5 x g_dn(AXON_s in-run). AXON_s's in-run gain uses the same instrument, tokens (CRUISE n = 90 / 29 logs), estimator and bootstrap seed as AXN_s. Both seeds must pass. u400's banked gain is reported beside it and the pre-amendment verdict is reported as a descriptive line; neither enters the rule.

## What did NOT change
Every threshold: PDMS margin -0.5; (b) g_dn <= 0.55; (c) CI-hi < 0.92; the factor 0.5 in (e); D_lon <= +0.3; CRUISE floor n >= 60 and logs >= 15; REFUTED clauses; the overall combination (CANDIDATE iff both seeds, REFUTED iff both, VOID on any red gate); the estimator; the PDMS control (banked X1_AXON_s). If AXON_s's in-run gain is below 1.08, clause (e)'s bar is stricter than before; that is the point.

## Added gates (any red -> VOID)
For the AXON reference, per seed: G-SEAM, G-IDT, G-MEMO, G-REACH:IDTgain (the same pure functions as for AXN_s), compose K1-K5, identity (G-CKPT), CRUISE token set equal to AXN_s's. Reported, never voiding: whether the re-composed AXON_s plans bit-identically to the banked X1_AXON_s seam (`ref_axon/AXON_s*/repro.json`).
