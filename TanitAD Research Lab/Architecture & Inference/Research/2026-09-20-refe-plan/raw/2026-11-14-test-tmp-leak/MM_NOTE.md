# refe/test_p6_levers.py: temp dirs that cannot leak (MM, 2026-11-13/14)

## What happened (MEASURED)

* On 2026-11-13, C: on the dev box fell to 0.46 GB. That ENOSPC-failed:
  * a peer session's pinned run (RefCV9 K2DC, 19:57Z);
  * the MM's own test comparison.
* The RefCV9 session found ~35 GB of `tmp*` dirs whose layout belongs to this test.
* The MM's census of `C:/Users/Admin/AppData/Local/Temp` kept only dirs whose top-level layout is a REFe test's and that had had no write for at least 1 h.
  * Result: **102 dirs, 37,777 MB**, created 2026-10-23 00:10 to 2026-11-13 05:52 local.
  * 26 of them had the layout `bank,o_main,o_off,o_on,w_main,w_off,w_on`. The others: `bank,o_off,w_off` 17, `bank,o_on,w_on` 14, `b,bank,g` 12, `bank,o,wb,ws` 11, `bank,o,wb,wb2,ws` 11, `m.pt` 11.
  * All were deleted by literal path. `deleted_leaked_dirs_2026-11-13.log` gives each path and size.

## Why the dirs leaked

* Every check did `td = Path(tempfile.mkdtemp())` ... `shutil.rmtree(td, ignore_errors=True)`.
* An assert between the two skips the `rmtree`.
* On Windows, `rmtree(ignore_errors=True)` gives up **silently** on a file that the failing frame still holds open, typically a `.pt`. The traceback keeps that frame alive until exit.
* Same class as "a zero is a claim about the probe": the cleanup reported nothing, so it looked successful.

## The fix: base = the tip blob `473cf971`

1. **`_mkdtemp(prefix)`.** Every temp dir is registered when it is made. The prefixes `p6lev_` and `p6mut_` make the owner visible in a temp listing.
2. **`_cleanup_tmp()`, registered with `atexit`.** It runs whatever ended the run:
   * clears `sys.last_*`, so the failing frame and its open files are released;
   * runs `gc`, then `rmtree` with 4 retries;
   * names anything it still cannot remove on stderr as `ZZTMP_LEFT`, never dropping it silently.
3. **New check `tmp_cleanup`.** It runs a child process that ends on an assert while a file in its temp dir is still open. It requires:
   * that dir is gone;
   * no `ZZTMP_LEFT`.
4. **New MUTATION.** The `atexit.register` line is removed, and `tmp_cleanup` must go RED.
5. **Mutated runs get `TMPDIR = <the copy dir>`.** A mutated run that leaks then leaks inside the harness's own dir, which the harness removes.

## Verification (MEASURED, dev box, CPU, `CUDA_VISIBLE_DEVICES=-1`, scratch tree = tip export + the fixed file)

* `--check tmp_cleanup`: `CHECK_OK tmp_cleanup 4`. Its temp dir was empty afterwards.
  * On Windows this covers the hard case: the probe's file was open when its run ended.
* Mutated copy (atexit removed): rc 1, `the probe's temp dir SURVIVED its failed run`. The leftover stayed inside the run's own `TMPDIR`.
* Full run, all mutations: `full_run_with_mutations.log`.
  * rc 0 in 697 s, `ZZP6_PASS 205`. The unfixed tip reads 199; the new check adds 4 asserts and the new mutation 2.
  * **18 mutations RED as required**, including the new one.
  * No `ZZTMP_LEFT`.
  * The only entry left in the run's `TMPDIR` is `torchinductor_Admin`: an empty cache dir that torch creates, not one of the test's dirs.

## Not changed, and why

* `refe/p7_ship_list.json` pins this file's old sha256, `9cd1aa98`.
  * That list is the **record of P7's ship**; it is regenerated at the next ship (`p7_ship_list.py`: "regenerate the list at ship time").
  * No test compares the committed list with the tree (MEASURED by grep).
  * The pod's P7 tree is not touched by this land.
* The `refe_vdac_p6` G-SUITE does not run this test, so HC-SHUF's chain is unaffected.
* Other REFe tests still use bare `mkdtemp`. Memory `refe-test-temp-leak` holds the dev-box discipline: run test batches serially, with `TMP`/`TEMP`/`TMPDIR` set to a dir that is deleted after each test.
