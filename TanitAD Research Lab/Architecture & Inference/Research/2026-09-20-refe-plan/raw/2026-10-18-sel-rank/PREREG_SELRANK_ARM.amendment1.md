# SR-TTC, Amendment 1 (Master Mind, 2026-10-18): the fit-set size and the S0 cost were wrong in the registered text. Facts only; no bar, gate, outcome or lever changes.

Registered body: `PREREG_SELRANK_ARM.md`, sha256 `cd73c4d4...`. Written before any SR-TTC label, feature or gain is read.

1. **Fit set.** Section 3 reads it as "about 2,733 sets (174,912 = 64 x 2,733), UNVERIFIED". That is wrong.
   - P2 trained on **174,912 served SETS** (`PREREG_P2.md`). These cover **99,495 distinct tokens**.
   - MEASURED from `raw/rl_pool/served_keys.tsv.gz`, reproduced in `raw/2026-10-18-p2-heldout-table/disjointness_proof.json`.
2. **S0 cost** (one P2 forward to cache pre-head features).
   - Section 4's 1.09 GPU-h becomes, at the draft's own 1.44 s/token rate, **about 40 GPU-h for 99,495 tokens**. That is ESTIMATED; the forward is deterministic per token, so one pass per token suffices.
   - The whole arm is therefore far costlier than the ~7 GPU-h the BACKLOG row carried.
3. **Dev gate coverage, MEASURED, `raw/2026-10-18-p2-heldout-table/`.**
   - P2's own held-out table currently covers **603 of the 3,137 gate sets (11 of 24 logs, 333 of 1,705 tokens)**.
   - All its gates pass: disjoint from navtest and from P2's training sets; banked labels reproduced to 0.0 on 768 proposals; GPU dump bit-identical to the landed table 4/4; pre-head features reproduce the logits to 1e-3.
   - The other 13 logs need camera frames that exist only on the pod (1,010 tokens).
   - 11 logs have no nuPlan DB on the dev box (362 tokens).
   - The section 6 dev gate needs the full set or a disclosed re-scoping, and either decision is written as Amendment 2 BEFORE the gate is read.
4. **Open for the PI** (the GPU decision this arm already needed), to be fixed by Amendment 2 before any fit data is read:
   - (a) The full fit set, ~40 GPU-h for S0 plus training.
   - (b) A pre-registered random subsample of fit LOGS, chosen by a fixed seed before reading. For example 10 % is ~4 GPU-h for S0.
   - Under (b), the reach and MDE arithmetic in section 6 is unchanged, because it concerns the navtest confirm. The fit's statistical power is not, and (b) must say so.
