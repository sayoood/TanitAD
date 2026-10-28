# POSTFREEZE_LOG -- choices and findings made AFTER PREREG_FROZEN.md (sha256 d9af82be..., 2026-10-28T23:11:02+0100). None of them can move a committed verdict.

1. [23:3x] ARM 1 (NAV) was run before ARM 2. Finding made while reading the ARM 1 output (the AUC of the speed-limit channel L alone / L+E exceeded that of the nav token N5+E): the `limit` / `limit_ok` input the trunk reads is NOT a posted speed limit.
   Tip `stack/scripts/build_v8_speed_max_labels.py` lines 16-19 and 138-146 and `stack/scripts/refc_v3_train.py` lines 270-273: it is `g_tac.goals.SPEED_BAND.v_hi_ms` = "the maximum of the ego's OWN REALISED speed over [t0+2 s, +6 s]" (provenance ego-future, oracle, quantised to 4 bins; "bin + v0 recovers R^2 0.9702 of the raw ego future").
   It is therefore a second ego-future channel in the trunk, one value per clip, and the previous probe's diagnostic "N" (nav5 ++ limit) mixed the two.
2. CONSEQUENCE for ARM 2: the committed Z_nn swaps the nav only, so the trunk still reads the ego-future speed ceiling. Added (POST-FREEZE, reported beside the committed Z_nn, NEVER decisive): **Z_nl** = the donor swap of nav AND limit / limit_ok (the donor window's clip's limit). Extracted in the same pass as Z_nn
   (same donors, same order). The committed verdict rule is applied to Z_nn exactly as frozen; if Z_nl disagrees with Z_nn this is stated next to the verdict.

3. [23:37] RAM guard: free RAM fell to 2.1-2.3 GB while this package's own V4 extraction (working set 4.2 GB of mapped output pages, 1200/2185 clips) was running. The 3 GB rule stays for the V4 / Vdiff readers, extraction and eval stages;
   the Z-source probe stages (12,000 x 1,024 fp32 = 49 MB per population, RSS < 0.6 GB) were run with a 1.5 GB floor. No other process was touched.

4. [23:52] Resource sequencing: the Vdiff and V4 PCAs, the Z_nn MLP stage and the eval extraction were first run concurrently; free RAM fell to 1.3-1.8 GB (this package's own jobs) and the 3 GB guard refused the Vdiff lin / pmove stages. The V4 PCA chain was stopped (per-fold files banked: folds 0-1 of P-STOP) and the stages are run sequentially.
   The 3 GB floor now applies to the V4 / Vdiff PCA readers and the extraction / eval stages; the lin / mlp / pair stages (64-d PCA scores, RSS < 0.4 GB) use a 1 GB floor. The eval extraction process that was started at 3.5 GB free was killed at 1.6 GB free and is re-run alone.

5. [23:59] The frozen C2 (donor-episode labels, seed 2) read Z_nn+E / P-STOP fold-mean 0.4543 [0.418, 0.491]: the CI misses 0.5 -> by the frozen text the arm is FAILED-CONTROL (kept as the committed reading in VERDICT.json).
   The bootstrap CI resamples EPISODES with the labels fixed, so it cannot see the spread over donor-label DRAWS. `c2_band.py` repeats C2 with seeds 3..12: Z_nn+E P-STOP 0.5105 sd 0.0226 [0.470, 0.544]; with seed 2 the 11-draw mean is 0.505 (t vs 0.5 below 1) and seed 2 is the minimum of the 11 draws (a 1/11 event).
   POST-FREEZE reading, reported beside the committed one and never replacing it: C2 'passes against its band' if |t| of the 11-draw mean vs 0.5 < 2.58 and the true arm exceeds the band maximum.

6. [00:05] The band criterion of item 5 was tightened/simplified BEFORE any verdict was read from it (the |t| < 2.58 clause flagged a 0.51 null mean on P-MOVE, where the frozen C2 passes anyway, as a failure): band_pass = (11-draw null mean within 0.03 of chance) AND (true arm AUC > the band maximum). Descriptive only.
