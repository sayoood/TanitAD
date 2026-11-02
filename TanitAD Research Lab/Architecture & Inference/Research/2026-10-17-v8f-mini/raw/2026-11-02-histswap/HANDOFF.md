# HANDOFF: HISTSWAP built, frozen and chained; written 2026-11-02 18:03. A fresh agent finalises it after the chain's marker.

Package `E:/Projects/TanitAD-artifacts/dagland/histswap/`. Scratch `F:/TanitAD-scratch/histswap/` (`tree_dag`, `tree_li` carry the tool `stack/scripts/v8f_t1_eval_histswap.py`; `out/` = rollouts, `work/` = reader rig). Nothing is in git (the Master Mind lands).

## State
* **Card frozen 2026-11-02T17:57:30+0100**, sha256 `7e91a7c742a82365bc73d3709cfd789ffb514eff7f27b063d9c80b339412278b` (`FROZEN_SHA.txt` pins the tool `7751673d...`, tests, reader, pre-check, Y files). Tool = FEAS2 tool (`4b66cc86...`) + `code/apply_histswap_patch.py`; tests 8 passed (`logs/pytest_histswap.log`, 3 mutations RED) and 6 passed (`logs/pytest_read_rule.log`).
* **Smoke** 32 windows real DAG-OACC s0 checkpoint, 2 threads: `logs/smoke_histswap.out` = SMOKEH OK (tick 0 and tick-1 executed state equal the off run bit for bit; v0_in is the logged speed; floor equal).
* **Chain launched 2026-11-02T17:58:50+0100** via WMI (`histswap_chain.cmd`, process 40400). It WAITS (polls inside) for `feas2_t1/raw/FEAS3.done|FEAS3.fail` (~18:20), then runs 5 rollouts one at a time at 16 threads (~150 s each: `SREP4_DAGOACC_s0` = KA1 with `--histswap off`, then `HSW_DAGOACC_s0/s1`, `HSW_SOUP_s0/s1`), then `code/histswap_t1_read.py`. Expected rollouts done ~18:35, read done ~18:45.
* **Markers:** `raw/HISTSWAP.done` (success; `raw/HISTSWAP_T1.json`) / `raw/HISTSWAP.fail` (reason inside). Log `logs/chain.out` (`ZZHSW-*`), per-run `logs/run_<tag>.out`, read `logs/read.out`. RESUME = launch `histswap_chain.cmd` again via WMI Win32_Process.Create (finished runs skipped).

## Finalisation (poll the marker with ONE command, never a sleep loop)
1. `.fail`: read the reason. A REFUSED control (KA1 / KS0 / KS1 / KS2 / KF / K3 / KB1) is a result to report, not to patch around; a read crash after the rollouts is a rerun of `code/histswap_t1_read.py` (a reader edit after data is diffed and sha-logged in the results).
2. `RESULTS_HISTSWAP.md` from `raw/HISTSWAP_T1.json` (adapt `../feas2_t1/code/make_results_feas2.py`): the `outcome` block (SUPPORTED / NOT SUPPORTED / MIXED as the card's table decides), the 4 M1 cells (`m1_cell`: X-Y, Y-P10, share closed), all 19 rows for X vs P10, X - Y and Y - P10 per cell, `read_a` / `read_xy`, the M1 per-class split, `swap_bucket_differs / swap_ticks` (header of each `out/HSW_*/t1v8f_*.json`), the card's ESTIMATED 0.60 / 0.25 / 0.15 beside the result. State: X is an ORACLE (absolute rows are not performance); S-POS rows of X are UNVERIFIED; SUPPORTED is leak-confounded (card section 1).
3. Execute the card's committed next step (Rule Zero): SUPPORTED -> the leak-free separating arm design + the recovery fine-tune to the PI for a box-2 go (`DRAFT_RECOVERY_CARD.md`, still a DRAFT); NOT SUPPORTED -> name the next lever from the per-class split of the remaining M1 excess (launch-class share >= 0.5: trigger / encoder path, PI decision; moving-class share >= 0.5: longitudinal mode-commit head, PI go).
4. `python -I code/make_manifest.py`, then hand back <= 300 words.

## For the Master Mind (integration, PI)
Land the package with `feas2_t1` / `kappa_t1`; register `H-HISTSWAP-1` (T1-kin, paired episode-cluster bootstrap B 2,000 seed 0, episodes only). `tests/test_v8f_t1_histswap.py` needs `V8F_T1_FEAS2_TOOL` (`../feas2_t1/code/v8f_t1_eval.py`, sha `4b66cc86...`). The PI's box-2 go for the recovery fine-tune is still open.
