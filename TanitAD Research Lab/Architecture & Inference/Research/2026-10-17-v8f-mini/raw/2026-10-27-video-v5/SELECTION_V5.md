# SELECTION_V5 -- chapter 8 (DAgger) scenes, fixed by a rule BEFORE any chapter-8 frame was rendered

script `select_scenes5.py` (sha256 d3063aaeef21f352); **rule sha256 ee759319dfafa149f8ed96da2d1893944569f9cdf5675399ebda9aac76d000c5** (the RULE text below, UTF-8, as the string `RULE_TEXT` in the script).
Inputs: `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG_s0_m10_causal_heads` and `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG0_s0_m10_causal_heads` (keys g, cl, ws, clip_index, eid; seed 0 only), `inputs/window_strata_eval141.npz` (sha256 55d12d16ac04f843).

RULE (stated on screen):

> delta_w = ADE6(DAG, seed 0) - ADE6(DAG-0, seed 0) of window w, T1-kin, causal, m = 10 (negative: DAG closer to the human's path), over the 2,399 windows in the dump order.
> 1. For each GT stratum (JUNCTION, CURVE, NO-TURN; window_strata_eval141.npz, defined from the human's labels) sort the stratum's windows by (|delta_w|, grid row) ascending and take position (n - 1) // 2: the lower median of |delta|, the typical size of change in that stratum.
> 2. WORST: the window with the largest delta_w over all 2,399 windows (ties: lowest grid row).
> 3. Three DRAWS: numpy.random.default_rng(20261027).choice(n_cand, 3, replace=False) over the windows (dump order) with GT mean speed >= 5 m/s that are not already chosen by 1 or 2.
> No window is swapped or dropped for how it looks; a window where DAG is worse stays in.

Strata sizes: {'JUNCTION': 298, 'CURVE': 140, 'NO-TURN': 1961}; candidates for the draws: 1908; draw positions [1390, 1882, 415].

| # | tag | how chosen | stratum | grid row | clip (first 8) | clip_index | ws (cache step) | GT mean speed (m/s) | delta = ADE6(DAG) - ADE6(DAG-0), s0 (m) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | JUNCTION | median abs(delta) of stratum JUNCTION (n = 298) | JUNCTION | 496 | 37e61793 | 29 | 11 | 5.56 | +0.6526 |
| 2 | CURVE | median abs(delta) of stratum CURVE (n = 140) | CURVE | 944 | 63434e0b | 55 | 35 | 14.06 | -0.3069 |
| 3 | NO-TURN | median abs(delta) of stratum NO-TURN (n = 1961) | NO-TURN | 59 | 10804db8 | 3 | 35 | 4.60 | +0.4664 |
| 4 | WORST | largest delta over all windows | NO-TURN | 1795 | c8a39711 | 105 | 39 | 6.49 | +8.8633 |
| 5 | DRAW1 | default_rng(20261027).choice | NO-TURN | 1715 | b99058dc | 100 | 59 | 9.60 | -0.6531 |
| 6 | DRAW2 | default_rng(20261027).choice | NO-TURN | 2344 | fb8e8200 | 137 | 55 | 22.47 | -2.6123 |
| 7 | DRAW3 | default_rng(20261027).choice | NO-TURN | 526 | 38fe80c1 | 30 | 63 | 14.15 | +0.3004 |

The delta column is printed here for the record; it is the quantity the rule sorts by.  The seed-1 numbers are shown in the video beside, and were not used to choose.
