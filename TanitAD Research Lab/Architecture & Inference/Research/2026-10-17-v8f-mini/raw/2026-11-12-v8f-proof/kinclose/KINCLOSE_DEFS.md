# KINCLOSE_DEFS (card PREREG_V8F_PROOF_ADD88, tier T0)

Source files: K = `kinclose.py`, CF = `closeread/closeread_fit.py`, CL = `closeread/closest.py`, card = `PREREG_V8F_PROOF_ADD88.md`.

| item | definition | source |
|---|---|---|
| frame period | `dt_s` of the dump (0.2 s); used for the slope and the ego path | K:132-160 (`make_ctx`), K:163 (`arm`) |
| ego speed v(j) | key `v` of the clip file, 101 frames, written by `ego` into `ego_v_eval.npz`; joined to the dump by clip id | K:411 `load_ego`, K:458 `run_ego`, card 2 |
| ego path s(j) | cumulative sum of dt * (v(j-1) + v(j)) / 2 from frame 0, s(0) = 0 | K:114 `ego_path`, card 2 |
| slope_W(x)(j) | least-squares slope against time over frames j-W+1..j that carry a value (mask) and are >= 0; NaN below 2 points | K:93 `slope_rows`, card 2 |
| frames that carry a read | rows with `valid` and finite gap, p, clos (the rule of `build_rows`) | K:132 `make_ctx`, CF:86 `build_rows` |
| scored frames | rows of `closeread_fit.build_rows(dump, 25, episode map)` per reader seed | K:132 `make_ctx`, CF:86 |
| K_W | slope_W(g + s) - slope_5(s); g = raw gap read, slope_5(s) over all frames j-4..j | K:163 `arm` (kind K), card 2 |
| S_W | slope_W(g) | K:163 `arm` (kind S) |
| E_W | slope_W(s) - slope_5(s), on the frames that carry a read | K:163 `arm` (kind E) |
| B_W | as K_W with the label gap; value only if the label gap is present and finite on all W frames of the window (and j-W+1 >= 0) | K:163 `arm` (kind B) |
| BS_W | as S_W with the label gap, same support as B_W | K:163 `arm` (kind BS) |
| K_W on the B frames | K_W (and B_W) restricted to the rows where B_W has a value; keys `K<W>_onB`, `B<W>_onB` | K:378 `all_arms` |
| cell L25 | `ClosingEstimator.load(closest_L25_s<k>.npz).apply(X)` on the H 25 scored rows | K:190 `cell_est`, CL:63 |
| target | label closing speed at the frame (`lab_closing` of the scored row) | CF:86 `build_rows` |
| pooled r, interval | `cell_summary`: pooled Pearson over reader seeds and episodes; two-level bootstrap (reader seeds, then episodes, episode draw shared), B 2000, percentile 2.5 / 97.5, generator seed 88 | CF:258 `cell_summary`, CF:234 `boot_totals`, K:214 `summarize` |
| paired difference | mean of |error A| minus |error B| on the same frames, same bootstrap, via `pair_summary` | K:227 `paired`, CF:251 |
| mae | pooled mean absolute error (all reader seeds) | K:214 `summarize` |
| err_sd | standard deviation (ddof 0) of est - y, pooled over reader seeds; per seed in `err_sd_per_seed` | K:214 `summarize` |
| err_lag1 | per seed: Pearson r between the error at frame f and at f+1 over consecutive scored frames of one clip | K:205 `lag1` |
| k0 | max abs(B_5 - label closing) <= 1e-6 m/s on scored rows with `lab_run` = 5 (count > 0) | K:260, card 4 |
| k1 | on rows where all of B_25, BS_25 and the three planted paths have a value: paired B_25 - BS_25 point < 0 and upper end < 0; mae of B_25 below that with the path shifted by +10 frames, by -10 frames, and sign-flipped | K:271, card 4 |
| k2 | v finite and >= 0 on frames 0..last scored frame of every scored clip; mean v over scored rows within [1, 40] m/s | K:291, card 4 |
| k3 | per seed frame count equals `controls.t0.frames.H25_seed<k>` (n_tool and n_dump) of the ADD81 control file; pooled cell r, cell mae, per-seed r, pooled n equal `arms.T25` of the ADD81 result within 1e-6 | K:305, card 4 |
| k4 | planted world (`gen_world`, seed 7): signal read gives r(K_25) > r(S_25) + 0.05 and lower mae; noise-only read gives abs(r(K_25) - r(E_25)) <= 0.10 | K:331, K:513, card 4 |
| name 1 | `K25-BETTER-THAN-OUTPUTS` if paired K25-L25 upper end < 0 and r(K_25) >= 0.30 in each seed, else `K25-NOT-BETTER` | K:351 `names_from`, card 3 |
| name 2 | `EGO-TERM-HELPS` (K25-S25 upper < 0) / `HURTS` (lower > 0) / `NOT-SEPARATED` | K:351, card 3 |
| name 3 | `REACHES-THE-REQUIREMENT` (K_25 r lower >= 0.9) / `BELOW-THE-REQUIREMENT` (upper of K_5, K_10, K_25, K_50 all < 0.9) / `REQUIREMENT-NOT-DECIDED` | K:351, card 3 |
| name 4 | `GAP-READ-IS-THE-LIMIT` (B_25 r lower >= 0.9 and K_25 on the B frames upper < 0.9) / `WINDOW-IS-THE-LIMIT` (B_25 r upper < 0.9) / no name | K:351, card 3 |
| result binding | `controls_sha256` in the result = sha256 of the control file; `show` checks it | K:425 `run_read`, K:485 `run_show` |
| failed control | `names: []`, arms and paired values only under `withheld`, nothing printed but control lines | K:425 `run_read` |
| 1.24 m/s | `row147_error_sd_ms`, printed beside every err_sd as a ratio by `show` | K:26, K:485 |
