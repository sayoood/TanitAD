# SELECTION_V4 -- chapter 7 (T1-kin) scenes, fixed before any per-window T1 number was read

script `select_scenes.py` (sha256 abb28103732ae554); reads only g / ws / clip_index of `C:/Users/Admin/t1kc_run/dump_HACC_s0_m10_causal`; no executed track (cl / fl / ol) is loaded.

RULE (stated on screen):
1. the four v3 windows, if their (clip, cache step) is in the T1 grid;
2. three windows drawn with numpy.random.default_rng(20261026).choice(n_cand, 3, replace=False) from the T1-grid windows (dump order: clip_index, then ws) whose GT mean speed is >= 5 m/s, EXCLUDING the four v3 windows. GT mean speed = mean over the 30 knots of |g[k]-g[k-1]|/0.2 s, g[-1] = origin.
No window is picked, swapped or dropped for how the model looks on it.

T1 grid: 2399 windows; GT mean speed >= 5 m/s: 1911; candidates after excluding v3: 1907; draw positions [705, 1153, 1499].

| tag | source | grid row | clip (first 8) | clip_index | ws (cache step) | GT mean speed (m/s) |
|---|---|---|---|---|---|---|
| JUNCTION | v3 | 1131 | 79e30ac2 | 66 | 35 | 7.43 |
| CURVE | v3 | 654 | 45aea78c | 38 | 31 | 11.63 |
| NO-TURN | v3 | 763 | 51c01ba4 | 44 | 59 | 18.50 |
| FAILURE(worst) | v3 | 2158 | ed87040c | 126 | 59 | 11.50 |
| DRAW1 | rng(default_rng(20261026)).choice | 834 | 59c25a0f | 49 | 3 | 36.50 |
| DRAW2 | rng(default_rng(20261026)).choice | 1400 | 9b4347d2 | 82 | 23 | 11.28 |
| DRAW3 | rng(default_rng(20261026)).choice | 1844 | d1d382aa | 108 | 27 | 15.29 |

v3 windows in the T1 grid: 4 of 4. grid row == v3 window id: {'JUNCTION': True, 'CURVE': True, 'NO-TURN': True, 'FAILURE(worst)': True}
