# PREREG_V8F_HACC_MC build: what could not be implemented as written, and every interpretation (quoted)

## A. Two controls that CANNOT pass as written (both gate only the READER; no run depends on them)

**C-1 bar.** PREREG: "(plan - P)_along = c.(tau - P)_along at knots 1-12, every tick | max |d| <= 1e-6 m".
The dumped `plan` is a float32 tensor, so |d| is its rounding: half an ulp at |plan| (9.5e-7 m at 16-32 m, 1.9e-6 m at 32-64 m, 3.8e-6 m at 64-128 m).
MEASURED (SMOKE, 24 windows, tool 81aa3e36, `smoke/chain/READ_CONTROLS.json`): max|d| 3.2e-6 .. 3.7e-6 m on every MC-H / MC-F run = **0.497-0.500 ulp32**, plan up to 98 m, 5.6-9.7 % of tick-knots over the bar.
The real eval-141 run reaches the same positions, so the literal bar fails by construction.
The regression arm (HACC table through the same check) reads 1.5-1.7 m and fails as it must.
Proposed amendment (NOT applied): "max over ticks and knots 1-12 of |d| / spacing(float32(|plan|)) <= 1".
Reader: `--c1-bar ulp32 --c1-amendment <id>`; default is the literal bar, which REFUSES.

**C-4 clause 2.** PREREG: "the regression GT := P gives c = 0".
With the fit as written (probe12: lambda* x D2, the tail held at HACC's w) GT := P gives **c = 0.3094 / 0.3489** (the real fit: 0.3071 / 0.3468). The D2 term couples knot 12 to the held tail (HACC w_along at knot 13 = 0.367 / 0.409).
MC-F's c is therefore pinned by continuity with HACC's tail, NOT by the data: data-only (lambda = 0) c = 0.084 / 0.072. HLAT's analogue passes only because there the whole table is free.
c = 0 holds only if the tail is also zeroed (analytic identity). Reader: `--c4-clause2 tail0 --c4-amendment <id>` (a plumbing identity, not evidence of data-dependence). Default is the literal text, which REFUSES.
Consequence for the outcome table: MC-F is "constant = tail-continuation", not a data fit (it is a beside arm).

## B. Interpretations (each quoted; none changes a bar)

1. m30. "3 arms x 2 seeds x m10, plus m30 per arm": m30 runs use the same flags (causal bucket, `--dump-ticks`).
2. F/2. TRAIN bar: "(0.0604 m)" used as printed. KEEP: F/2 = 0.06035 (F = 0.1207); the outcome also lists which row would match at 0.0604 and 0.06037.
3. M1 / M2 are lcdiag probe2's own quantities (`b_frac_fast_pairs_arm_slow`, `b_share_of_excess_on_pairs_where_arm_own_speed_lt_vmin`). "M2: the arm's L2 share on those pairs. HACC's is +0.0202" is an arm-minus-P10 excess (as probe2). C-6b reproduces 0.0536 / 0.0277 and +0.0202 / +0.0072 to 4 dp from the banked bank.
4. Overlapping outcome rows. FIXED-NOKEEP and MECHANISM-REFUTED can both hold. The word is the FIRST row in table order; `matching_rows_in_table_order` lists all.
5. C-3 "HACC's ... ticks: non-zero". HACC's EVAL ticks are not in the matrix. Regression = HACC's and STEEP's tables through the instrument on the MC-H ticks, STEEP's own ticks, and HACC's TRAIN known value (-0.1588 / -0.0726 +-2e-3, n = 1,747: reproduced -0.15876 / -0.07263).
6. C-5 "as T1KIN": C-1(i) <= 1e-9, C-1(ii) <= 1e-4 with unique grid + eid, C-2 floor ADE6 = 2.48699212 +-1e-5 (m30). Plus my reading of "the P10 loop equals T1KIN's banked floor": the m10 `fl` equals the banked HACC causal dump's `fl` <= 1e-9 (HLAT's C-A1 tolerance). It is a same-batch-composition identity: MEASURED batch 24 vs 12 moves `fl` by 6e-4 m (the loop amplifies float32 noise ~1000x). Do not change `--batch` (default 32) or OMP_NUM_THREADS (8).
7. arm - HACC rows use HACC's float64 originals (tar `t1kin_causalbucket_dumps_f64`, sha256_f64_concat verified against the manifest). C-6 uses the float32 bank, as the PREREG words it; the float64 value is reported beside.
8. Launch subset: "v0 < 6 m/s and GT speed at 6 s >= v0 + 1 m/s": v0 = ego speed at t0; GT speed at 6 s = the last 0.2 s GT chord speed.
9. Joint-mask L2 (beside) is probe2 (b) verbatim; HACC's value is reproduced (+0.0018 / +0.0015) before it is read beside an arm.
10. M2b-derived tables: MC-H c = HACC w_along at knot 10 = row 9 of the table ("the replan knot"); STEEP = probe12's `wst` (rows 7..11 = linspace(0, 0.6, 5)).
11. Families: TACTICAL not computed; STRATEGIC PAUSED (PI_DECISION_QUEUE ITEM 25) and not computed; DISTANCE-KEEPING not computed.
