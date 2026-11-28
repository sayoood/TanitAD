# FACTS_HW3 (card PREREG_V8F_RHHW3, frozen 3c85a42b)
Path: the frozen reader headway_read_r2.py takes --seeds 0,1,2 unchanged, so it was used directly (run_hw.sh), one root F:/TanitAD-scratch/rollhist/read/out/t1 holds all three seeds; no copies made, nothing to delete. Only hw3_name.py (F3 + name), levels.py (per-seed arm U), r0_check.py are new.
R0 PASS: reader with --seeds 0,1 equals the frozen two-seed read (out/read/hw) exactly (D, A_lo, A_hi, d_s; cells U, C, F; both forms). Detail R0_CHECK.json.
U, RH minus B0, three seeds (D [interval], per seed s0,s1,s2), F3:
- K (decides): +0.00415 [-0.00166, +0.01068]; +0.00712, -0.00178, +0.00712; F3 0.00890 -> NOT-SEPARATED, no straddle.
- C*: +0.01601 [+0.00699, +0.02660]; +0.01779, +0.01779, +0.01246; F3 0.02491 -> NOT-SEPARATED, straddle on the worse side (beside only).
C and F printed in RESULT_HW3.json / out_3/RESULT_HEADWAY.json. Planted check of the name: PASS (all four names, +0.03 cost = COST-REAL, 0.0 = NOT-SEPARATED).
Name: HW3-NOT-SEPARATED. Evidence weaker than a fresh arm (per-seed points partly seen earlier).
sha256: see sha256sum of RESULT_HW3.json b389bde8, hw3_name.py 7c43f746, out_3/RESULT_HEADWAY.json 7dd15a79, out_r0/RESULT_HEADWAY.json 7bd3d559, R0_CHECK.json fb75a6a1.
