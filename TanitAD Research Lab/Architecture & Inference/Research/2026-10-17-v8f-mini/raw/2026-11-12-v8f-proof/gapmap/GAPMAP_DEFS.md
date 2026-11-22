# GAPMAP-T0 definitions fixed where the card is open

* **Ties in the base gap.** Rows with equal base gap are pooled into one group first (weight = row count, value = mean label gap); the pool-adjacent-violators pass runs on the groups. Blocks merge when the earlier block value is strictly greater than the later one.
* **Block mean of x.** Mean of the base gap over all rows of the block (row weighted). Block means are strictly increasing, so `np.interp` is valid.
* **Interpolation.** `np.interp(g, block mean of x, block value)`; end values held outside the range; a one-block fit is constant. Rows inside a block are interpolated, not set to the block value (as the card words it).
* **IN.** Fitted on all error rows, evaluated on the same rows by the same interpolation.
* **Folds.** Clips with at least one error row, sorted by name (Python string order); fold = position modulo 5. A fold with no clips is skipped (only on tiny worlds).
* **LIN.** Closed-form least squares line on the fit rows; if the fit rows have no spread in g, slope 0 and the mean label gap.
* **After a map.** The mapped gap replaces the base gap on error rows only; `error_parts` of the imported tool is called again with the same label, output-flag and frame arrays. The error rows are asserted unchanged.
* **Ranges.** By the label gap, over error rows (near below 20 m, middle from 20 to below 40 m, far from 40 m, all), with the imported tool's NEAR_M and MID_M.
* **Bootstrap.** Clips with error rows; one `rng.integers(0, K, size=K)` per draw from `np.random.default_rng(20261122)`, 2,000 draws in order; a clip drawn twice counts twice (all its rows, with s', f' computed once before); rms over the drawn rows; cut recomputed per draw against the identity rms of the same draw; percentiles 2.5 and 97.5 by `np.percentile` (linear). Intervals exist for cut_s and cut_e of IN, CF and LIN on "all".
* **Rounding.** None in the tool; the printout shows 6 decimals; the JSON holds full floats.
* **Name edge.** A lower end exactly at 0.10 counts as "at least 0.10"; an upper end exactly at 0.10 is not "below".
* **c0.** Compares the identity block, computed from `compose` with the imported `error_parts` and `rms`, with the record's `t0` for n and for rms_e, rms_s, rms_f at 1e-6.
* **c1, c2 data.** Seeded planted worlds (30 clips of 60 frames) and 300 seeded series of 2 to 60 points (half with integer x, so ties occur); both generated in the tool on every run.
* **c3.** Reads the diagnostics of the CF and LIN runs actually used for the values (fit rows of a fold hold no row of a clip of that fold; each error row mapped once; each error clip in one fold).
* **c5 time clause.** File mtime is compared with the time the process started (`time.time()` at the top of main); equal or earlier passes.
* **Stage controls output.** `--out` written in `x` mode (refused if present); values under `payload` only when all controls pass, else under `withheld`.
