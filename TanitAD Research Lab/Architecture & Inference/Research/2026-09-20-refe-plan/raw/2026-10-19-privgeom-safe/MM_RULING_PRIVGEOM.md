# MM ruling on PRIVGEOM-SAFE (Master Mind, 2026-10-19)

**1. The registered calls stand as written.**
- DAC is **INFO-UPSTREAM** on the primary (+0.161 [+0.122, +0.198] / +0.161) and on the nosign companion (+0.079 [+0.056, +0.098] / +0.078).
- NC is **INCONCLUSIVE**: C-POWER did not fire, so NC is underpowered and SH3 is neither supported nor refuted.
- Every control is clean: gate PASS, C-AGREE exact on 316,224 proposals, C-NULL 0/5, 0 sets lost.
- The prereg mapping was applied exactly. Nothing is re-mapped after the data.

**2. The companion's INTERPRETATION is withdrawn to the sign-clean descriptive read (R54).** Re-checked by the Master Mind on `raw/descr/dac_signleak.json`:
- The "nosign" companion still carries the label's own test through the ORDER of its unsigned columns.
- "Some key step has |c_t| < |c_min|" implies DAC-unsafe with precision 1.0. That holds by construction: 0 safe proposals among 47,206 that meet the condition.
- The rule's recall is 0.887.
- That single contrast alone has F-AUC 0.953 [0.946, 0.959].
- ⇒ The companion's +0.079 is not evidence of proximity information BEYOND the label's own test.
- The sign-clean 9-column read is +0.028 [+0.013, +0.043] (centred +0.035). It is post-hoc and descriptive.
- What stands: the DAC row's missing information is recoverable from privileged geometry (primary). Whether it is PROXIMITY information beyond the verdict itself is supported only at the about +0.03 descriptive level.

**3. Consequences.**
- **SH5 (the committed next arm) proceeds, re-sized.** Its power is pre-registered at about +0.03 F-AUC, not +0.16.
- Any privileged target or auxiliary in SH5 must pass a PLANTED admissibility check: can the DAC label be recomputed from it? The check is never done by inspecting column definitions.
- The SH5 build agent was told this on 2026-10-19, before its prereg was sealed.
- **PRIVGEOM-CLEAN** (the agent's proposal) is registered as CONDITIONAL. A sign-clean pre-registered read on the 76 f10 logs this arm did not see would be a genuinely held-out confirmation of the +0.03. It is blocked on their nuPlan DBs; the DB fetch is a PI decision.
  - It is not run on the 41 seen logs: a pre-registration written after this descriptive read would be a forking path.
- **NC:** the first lever is the same 76 logs, under the same block.
- **C-NULL is not centred** (NC −0.017, DAC −0.062). Its non-fire is therefore conservative. Future permutation nulls centre within-set by construction.
- Scope: S0 f10, 41 logs, SELECTION tier, SEEN scenes, P2 in-sample (every arm out-of-fold). It is a diagnostic only; privileged geometry is never an inference input.
