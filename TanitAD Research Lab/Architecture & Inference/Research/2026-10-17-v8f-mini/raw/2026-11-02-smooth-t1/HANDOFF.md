# HANDOFF: SMOOTH-T1 AMENDMENT_1 (written 2026-11-02 ~15:50; read before acting)

Package `E:/Projects/TanitAD-artifacts/dagland/smooth_t1/`, scratch `F:/TanitAD-scratch/smooth_t1/out_m16/`. Card `PREREG_FROZEN.md` (sha256 4030d22f...) and `AMENDMENT_1.md` (sha256 717ff177..., frozen 15:41:09, `AMENDMENT_1_SHA.txt`) are not to be edited. Prior handoff kept as `HANDOFF_1535_prior.md`.

## State
* `smooth_t1_chain_a1.sh` (launched 15:41:50 via WMI, log `logs/chain_a1.out`, markers `ZZA1-*`) runs the 4 new `--smooth off` rollouts (16 threads, one at a time: OFFREP_DAGOACC_s0 = Y'', OFF_DAGOACC_s1, OFF_SOUP_s0, OFF_SOUP_s1; Y' of DAG-OACC s0 = K1OFF_DAGOACC_s0 already existed), then `code/a1_post.sh` (bound -> reader -> `RESULTS_A1.md` -> bank rollouts -> manifest). Post log lines: `logs/a1_bound.out`, `a1_read.out`, `a1_results.out`, `a1_bank.out`, `a1_manifest.out`.
* MEASURED already (`raw_a1/NONDET_BOUND.json`, computed before any X row): Y' and Y'' are **bit-identical** (dump and ticks; N = 0 for dump, floor, tick-0, ticks, rows), so every tolerance stays 1e-4. The rig is run-to-run deterministic at 16 threads; the banked Y differs from a fresh `off` run systematically (the floor by more than 1e-2 m), so the ladder's QUOTABLE ON Y' case is the expected one.
* Reader changes 7-10 are in `READER_FIXES.md`. Untested until the chain's post step ran once: `smooth_t1_read_a1.py`, `make_results_a1.py`.

## Exact next step
1. `tail logs/chain_a1.out` (expect `ZZA1-CHAIN-DONE`); if the post step failed, read `logs/a1_*.out`, fix the named line, rerun `bash code/a1_post.sh` (extracted arrays are cached in `raw_a1/arrays/`; delete nothing, move aside).
2. Read `RESULTS_A1.md` (generated; verify the status line, per-arm outcome words, controls table); add a short hand-written "Reading" section only for interpretation, no new numbers.
3. Report the outcome word per arm and seed, the four crawl rows and ADE6 (X - Y' and X - P10), the bound (zero). Registration of H-SMOOTH-T1-1, landing the patched tool (`stack/scripts/v8f_t1_eval.py`, pin at `v8f_t1_speedsub.py:44`), the tests and this package, and the next arm's pre-registration (curvature-bounded decode, KAPPA_MAX 0.2) are the Master Mind's.

## Update 15:48 (before ending; the chain is expected to finish about 15:56)
* K1OFF (Y') vs the banked floor, measured in the extraction meta: `fl` max abs deviation 1.71e-2 m (mean 3.4e-5), identical for Y'' (they are the same bytes); the 1-thread run had 3.76e-3 m. So K1' / K2' / K5' floor against the banked Y are expected to FAIL at 1e-4 and the status to read QUOTABLE ON Y'.
* `MANIFEST.txt` was written once by `code/make_manifest_a1.py` before the chain finished (partial raw_a1); the post step rewrites it at the end (`MANIFEST_before_A1.txt` keeps the pre-amendment one).
* Do not edit `smooth_t1_chain_a1.sh` while the chain runs (bash reads it incrementally).
