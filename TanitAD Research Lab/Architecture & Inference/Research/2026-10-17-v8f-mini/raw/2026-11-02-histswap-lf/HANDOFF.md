# HANDOFF: HISTSWAP-LF built, frozen, smoked and chained; written 2026-11-02 18:52. A fresh agent finalises it after the chain's marker.

Package `E:/Projects/TanitAD-artifacts/dagland/histswap_lf/`. Scratch `F:/TanitAD-scratch/histswap_lf/` (`tree_dag`, `tree_li` carry the tool `stack/scripts/v8f_t1_eval_histswap_lf.py`; `out/` = rollouts, `work/` = reader rig). Nothing is in git (the Master Mind lands).

## State
* **Card frozen 2026-11-02T18:50:07+0100**, sha256 `d98db98087e9175fb2dfae83cd68f9a39de72bb82be2410399b2a5c4d49e6a1b` (`FROZEN_SHA.txt` pins the tool `6c66f0a5...`, the HISTSWAP base tool `7751673d...`, patch, reader, tests, logs, Y files). Tool = HISTSWAP tool + `code/apply_lf_patch.py`: `--histswap {off,logged,cv_exec,cv_logged}`; `off` / `logged` bit-identical to HISTSWAP's.
  Tests 9 passed (`logs/pytest_histswap_lf.log`, 3 mutations RED; needs `V8F_T1_HISTSWAP_TOOL=../histswap/code/v8f_t1_eval_histswap.py`) and 7 passed (`logs/pytest_read_rule.log`). Smoke 32 windows: `logs/smoke_lf.out` = SMOKELF OK.
* **Chain launched 2026-11-02T18:50:49+0100** via WMI (`histswap_lf_chain.cmd`, process 59380), 16 threads, 9 rollouts one at a time (~2.5-3 min each): `SREP5_DAGOACC_s0` (KA1, `--histswap off`), `HCV_*` x4 (`cv_exec`), `HLV_*` x4 (`cv_logged`), then `code/histswap_lf_read.py`. **Expected rollouts done ~19:15, read done ~19:20.**
* **Markers:** `raw/HLF.done` (success; `raw/HISTSWAP_LF_T1.json`) / `raw/HLF.fail` (reason inside). Logs `logs/chain.out` (`ZZHLF-*`), `logs/run_<tag>.out`, `logs/read.out`. RESUME = launch `histswap_lf_chain.cmd` again via WMI Win32_Process.Create (finished runs are skipped).

## Finalisation (poll the marker with ONE command, never a sleep loop)
1. `.fail`: read the reason. A REFUSED control (KY / KA1 / KC1 / KC2 / KC3 / KF / K3 / KB1) is a result to report, not to patch around; a read crash after the rollouts is a rerun of `code/histswap_lf_read.py` (a reader edit after data is diffed and sha-logged).
2. `RESULTS_HISTSWAP_LF.md` from `raw/HISTSWAP_LF_T1.json` (adapt `../feas2_t1/code/make_results_feas2.py`): `outcome` (DOMINANT / HARMS / MOVED / NO EFFECT per arm and base), `m1_reading` (CLOSES / MIXED / NOT CLOSES per arm; mean share closed; cells), `interpretation` (A_INERTIA / B_LEVEL / C_INTERACTION),
   the 4 M1 cells per arm, all 19 rows for X vs P10, X - Y, Y - P10, `read_a` / `read_xy`, the M1 per-class split, `swap_bucket_differs / swap_ticks`, the descriptive `o_residual_x30` / `P_x30` block (prior anchor versus O's delta), the card's ESTIMATED probabilities beside the result.
   State: HLVL is an ORACLE, its closure is near-mechanical (card section 4), HCV is the discriminating arm; S-POS rows UNVERIFIED; tier T1-kin; estimator = paired episode-cluster bootstrap, episodes only.
3. Execute the card's committed next step (section 5): (A) HCV as a deployable inference lever + PI go for history-trend dropout training; (B) the recovery card (`../histswap/DRAFT_RECOVERY_CARD.md`) to the PI for a box-2 go; (C) HTRND (its own card; a new tool value); HARMS names the non-crawl rows first.
4. `python -I code/make_manifest.py`, then hand back <= 300 words.

## For the Master Mind (integration, PI)
Land the package with `histswap`, `feas2_t1`, `kappa_t1`; register `H-HISTSWAP-LF-1` (T1-kin, paired episode-cluster bootstrap B 2,000 seed 0, episodes only). `histswap` itself is still unfinalised: its read is banked (`../histswap/raw/HISTSWAP_T1.json`, SUPPORTED 4/4, mean share closed 0.939, an ORACLE) but `RESULTS_HISTSWAP.md` is not written.
The box-2 go for any training (recovery fine-tune, trend dropout) is the PI's and is still open; this package starts none.
