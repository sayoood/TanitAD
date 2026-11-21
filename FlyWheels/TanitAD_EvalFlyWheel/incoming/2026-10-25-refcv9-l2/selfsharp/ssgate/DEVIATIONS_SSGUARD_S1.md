# SSGUARD seed-1 deviation (Master Mind, 2026-10-26T05:06:47+0100)

The chain's mechanical score (05:04:58) read **INVALID (control failed: ['K-IND'])**. PREREG_SSGUARD_S1.md (68fc8a07, sealed before the
roll) section 3 says verbatim: "(K-IND is a property of the roll code and was measured at seed 0; it is not re-measured.)". The scorer
nevertheless computed K-IND at seed 1 by comparing the SEED-0 oracle trace with the SEED-1 L1 BASE-os bank -- two different inference
draws, which cannot be bit-equal -- and counted it as a control. That is a scorer defect against the sealed text, not a roll defect.

Fix: at seed != 0 the K-IND record moves to `controls_not_remeasured` (value kept, labelled cross-draw); no bar, threshold, arm or other
control changed. Every control the S1 prereg names (K-ID, K-SSID, K-ALL, K-NONE, K-HOOK) passed in the original scoring as well.
The original outputs are kept as `result_guard/*_as_run_0504.*`; the pre-fix scorer is `ssguard_score.pre_kind_fix.py.bak`.
Re-score: PASS (GUARD-ALL) -> under PREREG_SSGUARD_S1 section 4 this is **REPLICATED**.
