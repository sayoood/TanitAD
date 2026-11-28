# HANDOFF: PREREG_V8F_DAG_HACC (2026-10-27). Package: `E:/Projects/TanitAD-artifacts/daghacc/` (lands as `raw/2026-10-27-dag-hacc/`)

## State (all MEASURED, artifact paths relative to the package)
* DONE: C-H1 `ch1/ch1_M2b_s{0,1}.json` PASS (max |dw| = 0.0, lambda* equal); C-H2 `logs/ch2_pytest.log` + `fit/CH2_refusal_DAG_s*.json` PASS; C-X1 (added) `plans/CX1_*.json`.
* DONE: DAG's T0 plans on W_fit (`plans/fitplans_DAG_s*.npz`, 9,582 windows; fit cache read in place from the scratchpad HBLEND package), fits `fit/fit_DAG_s*.json` (lambda* 35309 / 33857, m* 16 / 16), `fit/weights_HACC_DAG_s*.json`.
* DONE: bake `ckpt/M2b_s{0,1}_dag_hacc.pt` (sha in `ckpt/BAKE_REPORT_seed*.json`); C-H3 (w==1 -> max|d| 0.0) and C-H4 (only `hacc_w` differs) PASS.
* DONE: 6 rollouts (`t1/done_*.json`, lanes `logs/lane{0,1}.log`): decisive `DAGHACC_s*_m10_causal_heads`, beside STRICT `DAGHACCSTRICT_*`, beside limit-logged `DAGHACC_*_logged_heads`.
* RUNNING DETACHED (no model turn needed): `post/post_chain.sh` (pack -> Read A rows -> S-POS -> t1kf -> controls -> T0 beside -> build -> tar) then `post/post_chain2.sh` (`READ_DAG_HACC.md`, `MANIFEST_SHA256.txt`).
  Completion: `logs/post_chain.log` ends `ZZPOST-DONE` then `ZZPOST2-DONE`; each stage logs `rc=` there; stage logs `logs/post_*.log`.
* EARLY LOOK (decisive config only, `pt_tmp/readA_test.json`; the chain recomputes it): KEEP (DAG-HACC - DAG ADE6, bar hi <= +0.0604) s0 +0.1264 [+0.0522, +0.2004] FAILS, s1 -0.1047 [-0.1834, -0.0334] holds.
  Per the PREREG table (REFUSED/BLOCKED first, then KEEP) a failed KEEP is **COSTS POSITION** whatever the other rows say.

## If a stage failed (exact re-run commands; Python = `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, `cd` to the package, `OMP_NUM_THREADS=4`)
`post/pack_daghacc.py` | `post/readA_daghacc.py DAGHACC DAGHACCSTRICT raw/readA_DAGHACC.json` | `post/spos_daghacc.py raw/spos_daghacc.json` | `post/spos_daghacc.py raw/spos_repro.json --repro-dag` |
`bash post/run_t1kf_daghacc.sh` | `bash post/run_c1_t1kf_daghacc.sh` | `post/tacdk_daghacc.py` | `bash post/t0_beside.sh` | `post/controls_daghacc.py` | `post/build_daghacc.py` | `post/tar_dumps.py` (only after S-POS ran: it deletes the per-episode dump dirs after a verified tar) | `post/mk_readme.py`.
Launch detached ONLY via WMI with CreateFlags 0x08000200 and a `.cmd` wrapper (see `lane.cmd`, `post_chain.cmd`).

## Then
1. Read `READ_DAG_HACC.md` (outcome, KEEP, controls, Read A per seed, LONGITUDINAL rows, beside). 2. Hand-back to the Master Mind: OUTCOME, LONGITUDINAL rows + KEEP per seed, rows still separated-worse, controls, manifest.
3. Optional cleanup of dirs THIS task created: `tip/` (re-derivable by `git archive 2a416ca7`), `pt_tmp/`. Do not delete anything else.
4. Open items for the Master Mind: bank the HBLEND fit cache (2.8 GB, scratchpad only); the committed next for COSTS POSITION is "as NOT FIXED": OACC's acceleration term in DAG's fine-tune (box 2, PI go, own PREREG).
