# MM ruling on EP-RELIAB-PICKS: the head-row line is CLOSED on S0's features (Master Mind, 2026-10-19)

**1. Outcome stands: EP-OVERPREDICTED on both seeds; not VOID.**
- C-NULL fired 1 of 5 (VOID needs 2). The 200-replicate empirical null fires at 0.04, and the real D_M sits 36 SD from the null mean.
- The EP row is off the label scale (within-set slope 0.31), so the registered rank-free mass form decides: D_M = +0.135 [+0.111, +0.164] / +0.133 [+0.110, +0.162].
- Predicted EP gain at the picks is **+0.124**; realised is **-0.011 [-0.016, -0.004]**.
- Amendment 1 (C-POWER made label-independent, sealed before any real-data stage) is ACCEPTED.

**2. The mechanism is now MEASURED, not hypothesised. It replaces R53's withdrawn reading and supports its replacement hypothesis.**
- The EP-led composite (EP-R1's c* row) buys a predicted EP gain that does not exist; realised EP is slightly WORSE.
- To get it, the composite accepts extra safety risk that the safety rows estimate about CORRECTLY (reliability at the picks: NC 0.895, DAC 0.947, TTC 0.925, every CI containing 1).
- The PDMS loss is mostly **DAC (share 0.63 [0.42, 0.92]; dominant in 98.8 % of draws, and under Shapley)**, then NC (0.34). The EP term is not the loss.
- ⇒ The defect is the EP row's OVER-DISPERSION on fast slots (its concordance is about 98 % slot prior, `D-REFE-HEADROW-PROBE-1`), not safety-row overconfidence.
- Scope: S0 f10, selection tier, seen scenes, EP-R1's composite. RL-1 V1's navtest NC loss is a different composite on a different tier, consistent with but not measured by this arm.

**3. ⇒ HEAD-LEVEL LEVERS ON S0's FEATURES ARE CLOSED.** Ten arms agree:
- EP-1 and EP-R1 (EP leak at the fit), EP-R1ub (no feasible anchor), SR-TTC (flat, VOID);
- HEADROW-PROBE (NC / DAC / TTC FEATURE-LIMITED; EP slot prior);
- FASTNEG-CENSUS (DAC fast-band deficit, coverage not supported);
- PICKRULE-AUDIT-NC (no veto recovers), NC-FASTCAL (already calibrated);
- EP-RELIAB-PICKS (the EP gain is not real);
- RL-1 V1's head-wise composite (N-BOTH on navtest).

A real EP gain needs a SCENE-SPECIFIC EP signal, and safer picks need sharper DAC. Both are FEATURE / TRUNK questions.
- **The lever is the trunk:** P6's end-to-end scorer training (A40, read about 25 Oct); R0e-pick (A40, PI grant).
- PRIVGEOM-SAFE (running) says whether DAC's missing information is proximity information beyond the label's own test.

**4. The committed next arm (pessimistic EP): drafted as `DRAFT_PREREG_EP_ADMISSION.md`.**
- Its main value is an ADMISSION GATE: no EP-led change (head, RL, P6 or R0e checkpoint) reaches a navtest look while its EP is over-predicted at its own picks.
- The pessimistic pick rule is secondary; it mostly reverts toward P2.
- It is sealed when a candidate exists.
