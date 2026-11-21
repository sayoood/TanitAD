# N4-DAC: CERT order rule, written BEFORE any wider-set CERT number was computed (2026-11-12/13, this file's mtime precedes n4_q2.py's first run)
Candidates (arm.form, all from the banked navtrain 469-token / 13-log score rows, tracked, W-TRANSP DB rebuild; RJ8 excluded = bit-identical to RJ5):
ORDER (fixed a priori; extends TF-N2 A1's CERT2 order [N2-R=RJ5.R, N2-C=RJ5.C, N1-C=R0.C, T=R0.T] by the banked arms in increasing distance from RJ5.R):
 1 RJ5.R  2 RJ5.C  3 R0.C  4 RJ3.R  5 RJ3.C  6 R0.R  7 R5HZ.R  8 RJ5H5.R  9 RG8.R  10 R5HZ.C  11 RJ5H5.C  12 RG8.C  13 R0.T  14 RJ3.T  15 RJ5.T
CERT-A: take the first candidate in ORDER whose tracked DAC = 1; none passes -> RJ5.R (the default).   (the CERT2 rule, wider set)
CERT-B: take the first candidate in ORDER with DAC = 1 AND NC = 1 AND DDC = 1; none -> RJ5.R.            (stricter certificate)
CERT-4: CERT-A restricted to the first four candidates [RJ5.R, RJ5.C, R0.C, R0.T]: the navtest CERT2 analogue on this set.
DESCRIPTIVE upper bound (never a bar): ORACLE = max-PDMS candidate among those with DAC=1.
Estimator: paired log-cluster bootstrap B 10,000 seed 20260927 (n4_common.boot); answers "another draw of LOGS" only; navtrain levels are contrast-only.
