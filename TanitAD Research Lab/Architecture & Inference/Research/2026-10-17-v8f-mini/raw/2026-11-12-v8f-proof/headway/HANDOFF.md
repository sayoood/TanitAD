# HANDOFF: HEADWAY reader (card PREREG_V8F_PROOF_ADD25). Tier T1-kin, MEASURED. Nothing staged, committed or pushed.

## State
Done: DEFS, reader with controls h1 to h6, tests (29 pass), real run Q1 and Q2, independent recomputation of U for HIER_s0. Not run: Q3 (arms LS, LSH, LSD not on disk).

## Results (F:/TanitAD-scratch/v8proof/headway/out_q12/VERDICT_HEADWAY.md, RESULT_HEADWAY.json)
* Controls h1 to h6 all PASS (each with its regression form failing as required). 642 lead windows, 56 episodes of 141 (card bars 200 and 20 met).
* Q1 HIER minus FLAT, cell U: D -0.0056, interval [-0.0269, +0.0140], six-seed spread max 0.0463, rule OPEN: DISTANCE-EQUAL.
* Q2 G minus HIER, cell U: D +0.0021, interval [-0.0136, +0.0175], rule OPEN: GRID-DISTANCE-EQUAL.
* Levels of U (mean over six seeds): FLAT 0.1332, G 0.1296, HIER 0.1275; logged driver unrestricted unsafe share 0.1246, contact 0.
* Baseline U above 5 percent of lead windows: the committed next step of card section 5 (distance-rule veto, own card, pod time) applies, whatever the names.
* Independent recomputation (headway_indep.py, indep.log): HIER_s0 U 0.106761565836, difference to the reader 0.0, agrees to 1e-9.

## Reading choices fixed in HEADWAY_DEFS.md before any arm number
Gap source = label join join_eval141.npz (lead_valid, lead_gap), not the ADD20 bank (bank rows end at step 67 and hold leads within 40 m only; windows reach step 97). Join and bank compared on common rows: 1,164 presence mismatches in 9,460 bank rows, max gap difference 1.84 m on 2,489 common finite rows (decides nothing). h4 compares the dump v0 with the bank ego speed (all 642 lead windows checked, max difference 0.0; with a shift of 5 frames 470 windows exceed 0.05).
Q1 spread uses HIER as commanded; with the FLAT spread (max 0.0374) U also reads OPEN.

## Commands (cwd F:/TanitAD-scratch/v8proof/headway; export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8; python = /c/Users/Admin/venvs/tanitad/Scripts/python.exe)
* controls: `headway_read.py controls --t1-root "F:/TanitAD-scratch/v8proof/hiergap/pull,C:/Users/Admin/v8proof_c/pull" --out-dir out_controls`
* Q1/Q2: `headway_read.py read --t1-root "<same>" --pairs HIER:FLAT,G:HIER --seeds 0,1,2,3,4,5 --spread-arm HIER --spread-seeds 0,1,2,3,4,5 --out-dir out_q12`
* Q3 (when LS, LSH, LSD seeds 0 and 1 are pulled under either root): `headway_read.py read --t1-root "<same>" --pairs LS:G,LSH:G,LSD:G --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir out_q3` (today it writes a controls-free note: no pair has both arms; out_q3 holds that note only)
* tests: `python -m pytest -q test_headway_read.py --basetemp=F:/TanitAD-scratch/v8proof/headway/tmp/pt -p no:cacheprovider` (create tmp first)
* independent: `headway_indep.py [RESULT_HEADWAY.json]`
Runtime of each real command: about 1 to 2 minutes, under 3 GB.

## Not verified / limits
Per-pair h3 runs on every pair with data; the Q3 arms were never read. The label join and the bank live on E: and F: (E:/Projects/TanitAD-artifacts/...). The reader imports q3_read.py and hierrep_read.py (torch import). Unsafe-rate cell is dominated by the time-gap bound (tau under 1.0 s) at 8 to 10 m/s.

## Scratch (nothing deleted)
tmp/ (pytest basetemp, under 1 MB), out_controls/ 12 KB, out_q12/ 56 KB, out_q3/ 4 KB, logs out_controls.log, out_q12.log, indep.log. All in F:/TanitAD-scratch/v8proof/headway/.
