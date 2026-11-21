# PREREG_MAPPATH AMENDMENT 2 -- the K-PROG tolerance was below float32 storage resolution; the check is split, tolerance unchanged (Master Mind, 2026-10-26 ~04:33 CET)

Sealed BEFORE any MAPPATH pick, rate or bar. Base 34a030ee, Amendment 1 b2d6eb84.

Amendment 1's walker check read max |diff| **7.594e-6 m** against its 1e-6 m tolerance (FAIL), mutation RED as required
(`result/MAPPATH_KPROG.json`). 7.594e-6 is ONE float32 ULP for coordinates in [64, 128) m (2^-17 = 7.63e-6); the fan is stored as
float32 (as the source trace), so a 1e-6 m tolerance against the STORED output was unreachable by construction -- a second
definitional error, disclosed here, not a transform defect being waved through.
**Split (tolerance unchanged):** (i) the walker vs the transform's float64 output (`PH.at_progress` before the cast) on the same seeded
5,000 triples, max |diff| <= 1e-6 m; (ii) the stored float32 value == float32(that float64 output) EXACTLY on the same triples;
(iii) the s + 0.05 mutation must still read > 1e-6 m. PASS iff (i) and (ii); the mapsoft_path run proceeds only on PASS.
