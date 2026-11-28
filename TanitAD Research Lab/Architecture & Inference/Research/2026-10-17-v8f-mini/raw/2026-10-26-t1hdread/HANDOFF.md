# HANDOFF: the one real read of PREREG_V8F_T1_HEADS (ae659d49)

State (2026-10-26 05:05): the READER IS DONE and tested; the REAL RUN IS NOT COMPLETE. No outcome row has been computed on real decisive data.
* `C:/Users/Admin/t1hd_run/logs/chain.out` last line: `HACC_s0_m10_causal_heads: waiting for C: >= 2 GB free (now 291 MB; the pagefile is on C:)`. C: has ~670 MB free. The chain cannot start causal s0 until a human or the parent frees >= 1.4 GB on C: (not mine to free).
* History: the first launch ran both seeds in parallel; causal s0 segfaulted (no traceback), causal s1 FINISHED (ticks_HACC_s1_m10_causal_heads.npz, t1v8f_..json, dump_.. all present). The relaunch is sequential and skips an existing output. Remaining: causal s0, logged s0, logged s1.
* I ran only controls K2 / K3 / X1 and timing on the finished s1 causal file (`scale_test.py`): all green, 6 s for 2,399 windows. No S / T row was computed or printed.

## When `chain.out` ends with `ZZT1HD-DONE 4`, run once (about 3-5 min, CPU only):
    cd C:/Users/Admin/t1hdread && ./run_real.sh > read.log 2>&1
It refuses on `ZZT1HD-STOP` and exits `NOT READY` without the DONE line. Do NOT edit `t1hd_read.py` after reading any outcome; a control failure is REFUSED, not loosened.
Output: `T1HD_READ.json` (first keys: `controls_PASS`, `REFUSED`, `OUTCOME`, `decisive_rows`; then controls in detail, `runs`, `dev_t0_regression_arm_effects`, `extra_checks_NOT_in_PREREG_not_gating`).

## What each control must read (computed, not guessed)
* K1: 0 non-zero of ~31 rows per seed (delta, lo, hi exactly 0).
* K2: float32 replica max 0.0 m (<= 1e-4); float64 recompute ~6e-5 m, under the computed bound ~3e-4 m.
* K3: r' == t0 for all 2,399 windows, S and T; distance 0.0 (<= 1e-6).
* K4 (logged runs, tick 0): s0 pos 0.9774 / 2.2965 / 5.8873 m, F1 0.8352 / 0.5062; s1 0.9290 / 2.2037 / 5.8964 m, F1 0.8476 / 0.5135 (JSON values; tolerance 1e-3); n_valid 2322 / 2319 / 2294; F1 cells 2107 / 2240. Smoke per-window already matches (S ckpt 2.3e-5 m, logits 3e-6). If K4 fails read `K4.per_window` first (ckpt diff, argmax equality) to diagnose the dead-reckoned nav / window set.
* K5 (causal s0): mirror - S about +3.5 / +8.6 / +17.7 m, all separated (DEV effect on T0 data).
* K6 (causal s0): donor - self about -0.48 (lat) / -0.30 (lon) F1, both separated (DEV effect on T0 data).
* K7: cl, fl, g bytewise equal to the banked float32 dumps; `sha256_f64_concat.equal` true.
* Extra (not gating): X1 true; `GT_frame_check_EXTRA` tick 1 / 2 medians near the tick-0 median (~0.8-1.1 m).

## After the read
Write nothing into the register from me; the result document and `GOALS_AND_CLAIMS.md` row `H-V8F-T1-HEADS-1` are the Master Mind's. Report: controls first, then per-family OUTCOME in the PREREG's words, then the beside rows (CTRV2 sensitivity included), with `n` and OFF-ROUTE counts.
