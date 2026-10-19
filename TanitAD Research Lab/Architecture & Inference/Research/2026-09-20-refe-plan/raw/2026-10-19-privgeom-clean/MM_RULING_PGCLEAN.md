# MM ruling on the PRIVGEOM-CLEAN draft (Master Mind, 2026-10-19 ~20:50; no label or outcome of the 76 held-out logs has been read)

**DQ1, the primary arm: CENTRED within set (`LGc`). RATIFIED, on MECHANISM, not on magnitude.**
- The question this programme needs answered is whether proximity improves the PICK, and the pick is a ranking WITHIN a set of 64 proposals. A between-set association (one set's proposals being nearer the boundary overall, e.g. a narrow road) cannot change which proposal is picked. It can also carry scenario-level confounds.
- Centring within set removes exactly that part and nothing else (draft §3). The machinery null (C-NULL) permutes within set by construction (`MM_RULING_PRIVGEOM.md`), so the call and its null read the same quantity.
- Both SEEN readings were banked by PRIVGEOM-SAFE before this ruling: +0.0352 centred, +0.0276 uncentred. The choice is NOT made because the centred one is larger: it would be the primary at any magnitude.
- The UNCENTRED arm (`LGu` = SAFE's `drop0`, exactly) is the COMPANION. It is reported beside the primary with the same estimator and rule, and has no authority over the call.

**DQ2, the bars: CONFIRM = lower CI > 0 AND point ≥ +0.02; NULL = upper CI < +0.02. RATIFIED.**
- The NULL bar sits at the same +0.02 materiality line as CONFIRM's point conjunct, so the two calls are disjoint and cover "material" and "immaterial".
- SH5's < +0.01 would make NULL fire 28 % of the time at a true zero (MEASURED, draft §8), a bar the null cannot reliably clear. INCONCLUSIVE covers everything else.

**DQ3, the scope: DAC only. RATIFIED.** NC-CLEAN is a separate prereg on the same extraction, never folded in post hoc.

**DQ4, the plant: sigma 1.25, calibrated to +0.0276 on SEEN arrays. RATIFIED.** C-POWER fires iff the CONFIRM rule holds on both seeds. A C-POWER failure makes the read UNDERPOWERED, never NULL.

**The four families:** LONGITUDINAL and LATERAL are stated per family with the reason and n (draft §), TACTICAL likewise, and STRATEGIC is PAUSED (PI ruling 2026-09-19, ITEM 25). This is a SELECTION-tier label lookup, not a driving result.

**Run order after the seal:**
1. the pre-read gates (C-REACH, C-ID, C-GRAD, A-ADMIT on the 76's real arrays, with no label read);
2. the extraction (about 6.4 CPU-h, 2 workers, RAM-gated, detached);
3. the fit, C-POWER and C-NULL;
4. the call.

A failed pre-read gate STOPS the run before any label is read.
