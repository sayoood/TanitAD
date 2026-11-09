# AMENDMENT 1 to PREREG_P6_SELECTORS.md (sha256 d4778ca4...), 2026-11-09, Master Mind

**Written before any selector was fitted on navtrain and before any selector was read on E2 or E3. No bar, split, candidate family or outcome changes.**

**What changes.** The card lists "measured ego speed" as an input. The FIT pack and P6's full-navtest proposal dump carry no ego speed. Both fit and read therefore use one proxy: the median over the 64 proposals of the first 0.5 s chord divided by 0.5 s. It is computed from the model's own output, the same way on navtrain and navtest, and is recorded as `v0_source` in every output.

**Why that and not a join.** A join would need two different ego sources, the target bank at fit time and the NAVSIM export at read time, which risks a feature that differs between fit and read.

**Check, MEASURED on the 1,058 C3 tokens where both exist (`dump_p6.npz`, `ego[:,6]`):** correlation 0.9991, median absolute error 0.083 m/s, 95th percentile 0.276 m/s, maximum 0.605 m/s, mean error +0.010 m/s.

**Implementation choices where the card is silent** are listed as B1 to B9 in `selectors/read_selectors.py`, and are fixed now: among them, family R nominates from the two best candidates on FIT-A, S uses one MLP per head, ranks are tie-correct, and "loses" in outcome S3 means a point difference below 0.
