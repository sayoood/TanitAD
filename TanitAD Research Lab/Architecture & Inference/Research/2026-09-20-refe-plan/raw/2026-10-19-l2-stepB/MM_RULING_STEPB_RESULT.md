# MM ruling after the Step B read (Master Mind, 2026-10-19 ~22:30)

**The verdict is reported as sealed: NOT CLEARED** (worst of the two head seeds). L2-s0 dPDMS +0.239 pts [−0.041, +0.473], L2-s1 +0.194 [−0.095, +0.455]. Both intervals contain 0, so neither head clears L1's O1 bar (≥ +0.30 with the lower end > 0).

**Attribution (descriptive, as sealed): the L2 EP head does NOT improve the search.** L2 − P2HEAD: −0.088 [−0.551, +0.196] and −0.133 [−0.599, +0.167].

Mechanism, MEASURED per family:
- The L2 heads steer the search to SPEED-UPS (23-28 % of outputs vs P2HEAD's 16 %).
- They gain EP (+0.83 / +0.97 pts) and give it back in TTC (−0.25 / −0.50 pts vs P2HEAD's +0.25).
- Their slow-downs cost −1.5 pts, where P2HEAD's slow-downs GAIN +0.65.

That is the EP-RELIAB class again: a sharper EP row over-weights the aggregate the search climbs. READ 1 predicted it (ΔA3 −0.048 / −0.061, both below 0), and Amendment 1 is consistent with it but too thin to be evidence (12-16 qualifying pairs per arm).

**The committed NOT CLEARED arm (L2-BRANCH) is DEPRIORITISED, not cancelled.** The measured mechanism predicts that a stronger EP row trades more TTC. L2-BRANCH is therefore not worth a GPU-day as drafted. It returns only with an aggregate-preserving or TTC-protecting term in its own amendment, and its `launch_gate.py` profile.

**The largest measured effect is the SEARCH ITSELF on P2's unchanged scorer:** P2HEAD +0.327 pts [+0.056, +0.722].
- Seed range 0.216 < 0.327, and every safety family is ≥ −0.25 (dTTC +0.25, dDAC +0.17).
- That meets every O1 condition. But it was the DESCRIPTIVE control (never gated, 1 of 3 arms read), so it is a POST-HOC finding. It is not promoted without its own pre-registration (CLAUDE.md; RULE ZERO "what this does NOT license").

⇒ **New committed arm, SEARCH-P2-CONFIRM** (RULE ZERO: the lever with the largest measured effect):
- The same search, unchanged: K = 5, M = 64, elites 8, warm start at P2's pick, 3 search seeds, the C8 config `299c6e63…`, P2's scorer with NO head substitution.
- READ C1 on **B-REST**: the 905 held-out plan tokens in neither Step A's 400 nor carve B's 400. The same footing as carve B for P2 (no M-PDM token, SEEN-SCENE). Its bundle needs one locked GPU forward plus CPU labels.
- L1's O1 bar VERBATIM, plus a pre-stated replication rule: the confirm is the CONFIRM read alone, never pooled with carve B.
- READ C2 on the UNSEEN STRATUM (S1b's 486 tokens / 123 logs, never seen by P2 or the trunk): the generalisation read. It needs the PI's yes for the frame fetch (about 400 MB). Its own sealed read.
- Before any navtest look: the frozen EP-ADMISSION gate. Its candidate block names "P2 + CEM search" and a bank of SEARCH OUTPUTS, as Step B's ruling decision 9 set.

**A hypothesis is REGISTERED, not claimed:** `H-SEARCH-ADMIT-A3`. A search's gain tracks the scorer's AGGREGATE concordance (A3), not its EP ordering (A1). Across 3 arms post hoc: P2 has A3 0.767 and the best search; the L2 heads have A3 0.719 / 0.706 and worse searches. If it holds, L1's Step A gate admitted on the wrong quantity. It is tested by SEARCH-P2-CONFIRM together with any future scorer arm. n = 3 is not evidence.

Stamps on every number above: MEASURED, SEEN-SCENE, SELECTION tier, label-only (not T1, not navtest), one training seed per head, log-cluster bootstrap ("another draw of LOGS" only). STRATEGIC PAUSED.
