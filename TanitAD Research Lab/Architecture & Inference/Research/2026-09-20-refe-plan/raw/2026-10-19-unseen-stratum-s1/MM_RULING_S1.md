# MM ruling on S1 (Master Mind, 2026-10-19 ~18:00; no U label, no head score, no frame fetched)

**The sampler: reproduce navtrain's OWN filter. Do not approximate it with speed weights.**
- Navtrain's token list is the window rule PLUS a PDMS filter: tokens where a constant-velocity agent scores PDMS > 0.8 are removed, and tokens where the human (logged) trajectory scores PDMS < 0.8 are removed (Dauner et al. 2024 §3.1, quoted in `NAVSIM_PROTOCOL.md:386-391`).
- Step A's population (P2's held-out navtrain plan) was defined by exactly that rule. A U read must use the same population, or A1 is not comparable to the gate it is read against.
- **Admissibility:** the filter uses the logged future and privileged geometry. That is fine at the LABEL / POPULATION stage (PI rule 2026-08-03: labels may use ego and privileged signals).
  - It is applied identically to every head, involves no head output, and is never an input at inference.
  - It is fixed BEFORE any U label of the endpoint is computed, and it is not tuned on any outcome.
- **The speed-weighted draw is WITHDRAWN as the primary sampler.** It closes only a third of the turning gap (MEASURED). It is kept as a reported sensitivity at most.

**S1b (CPU, DB + map only, no frames, no head):**
1. Identity first: on a seeded subset of the 214 navtrain-in-val logs, apply the window rule plus the reproduced filter (CV-agent PDMS > 0.8 out, human PDMS < 0.8 out), using OUR PDM scorer.
   - Report recall AND precision against the known navtrain tokens. Precision should rise from 0.195 toward 1.
   - Give the residual with its reasons: our PDM scorer vs NAVSIM's, the CV-agent definition, the thresholds' inclusivity.
   - A RED arm: thresholds swapped ⇒ precision must collapse.
2. Cost per token. Then apply the frozen filter to the U logs the S3 sample needs (or all 1,129 logs if cheap).
3. Regenerate the S2 / S4 fetch list from the filtered candidates. The seed stays PROVISIONAL until S3 seals it.

The filter's output (which tokens are in) is a POPULATION label, not an endpoint label. Nothing about A1, EP, or any head's score is computed in S1b.
