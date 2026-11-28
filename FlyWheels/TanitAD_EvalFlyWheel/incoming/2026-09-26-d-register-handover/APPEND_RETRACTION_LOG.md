<!-- APPEND to `Project Steering/RETRACTION_LOG.md` on the tip da8400b7 -- 1 region(s), copied verbatim from D:'s worktree lines 17029-17049. Land as a MOD on the tip blob; never as a file. -->

<!-- region D: 17029-17049: insert after tip line 17670 (sha256 d2c71836be8f, = D: line 17027) -->
### R25 (2026-09-26) - "the planner's rule is PDM-shaped, so a formula mismatch is NOT the hypothesis": it is v2-shaped, the harness scores v1

**Retracted premise.** SPEC_NAVTEST Amendment 3 (2026-09-26 ~09:52 Berlin) wrote: "The planner's rule is PDM-shaped
(`refe/planner.py` `aggregate`: NC x DAC x DDC x (5 EP + 5 TTC + 4 C)/14 over sigmoids), so a formula mismatch is NOT the
hypothesis." The formula is quoted correctly, but it is NAVSIM **v2**'s EPDMS shape (comfort standing for v2's two
weight-2 terms, driving direction multiplicative), while the harness REFe is scored on is NAVSIM **v1** PDMS
(navsim @ 3e8291b, `pdm_scorer.py:38-42`): NC x DAC x (5 EP + 5 TTC + 2 C)/12, driving direction at weight 0. So there IS
a formula mismatch: DDC gates the pick though v1 ignores it, and comfort weighs 4/14 against v1's 2/12 (1.7x).

**Measured effect (EXPLORATORY, same 200 tokens, `eval/rule_mismatch_diag.py`, 2,000 log-cluster resamples):**
re-selecting on the SAME stored logits with the v1 formula moves the pick's PDMS by +4.79 [0.58, 8.34] after epoch 11,
+0.04 [-3.84, 4.00] after epoch 12 and +1.54 [-0.23, 3.51] after epoch 13; the shipped rule reproduces the stored pick on
100 % of tokens in all three. So the premise was wrong but the conclusion it supported survives: the mismatch is not what
makes REFe selection-bound -- the scorer's within-scene ranking (Spearman 0.09-0.13) is the gap under either rule.

**Fixed:** nothing in the recipe yet. Which benchmark version the selection rule follows is a PI decision, and any rule
change needs a disjoint-token confirmation (SPEC E-6).

**Class:** a formula quoted without its VERSION -- "PDM" names two different benchmark rules (v1 PDMS, v2 EPDMS); the
`df` / units / camera-projection scope family. It survived because the paper's rule and our harness share every
component name, so the formula read as the benchmark's own.
