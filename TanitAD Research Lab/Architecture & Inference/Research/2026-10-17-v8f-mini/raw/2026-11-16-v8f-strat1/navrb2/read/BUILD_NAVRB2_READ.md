# BUILD_NAVRB2_READ: bank of the four NAVRB2 arms and the read of PREREG_V8F_NAVRB2 (2026-11-18)

Nothing committed, nothing pushed, no pod file deleted. All paths under `F:/TanitAD-scratch/strat1/navrb2/read/` unless named. Card FROZEN (sha256 `e238ba8d...`, checked by a test). The read has NOT run on the real arms: the full launch is the MM's.

## 1. Bank (MEASURED; the arms trained at e71782a3 on the LoRA pod, ARM_DONE 07:08 / 07:24 / 07:39 / 07:54Z)
PRIVATE HF dataset `Sayood/tanitad-v8f-strat1-runs` (the repo of the STRAT1 arms), hub head rev `48098c45`; per arm `<tag>/{ckpt_last.pt, argv.json, train_argv.json, selfcheck.json, gate_token.json, checks.json, summary.json, train_log.jsonl, zerograd_navrb.json, ARM_DONE, BANK_MANIFEST.json}`; the channel's table under `navrb2_inputs/`.
Uploader (light, nice 15, no GPU, no model load, one upload at a time): pod `/root/strat1/tmp/strat1_bank_pod_navrb2.py` via `bank_navrb2.sh`, rc 0, log `bank_navrb2.log`; copies in `code/`.
| arm | ckpt sha256 (pod file = hub LFS = local pull) | hub commit |
|---|---|---|
| N1R_s0 | `35dc943e0c7989656359cb0d1973b9801f6220c71b395f8be5b5a1dd833b7048` | 6de80847 |
| S1R_s0 | `db6a79b08d81249736521708657191f61a6aad462574120c66cca26ca22fbe1d` | 72d44e6b |
| N1R_s1 | `32b0336d9243dfeeb266bed88767c76ca34bf7fbd578cdf64184b5f10d78a03a` | aae6813f |
| S1R_s1 | `803d1e6251daae5b7459a04b758b4b5aa672a8023f6492cbd31b44addcfc1f02` | 5af8c5cc |
Three copies: (1) the pod file hashed by the uploader into BANK_MANIFEST; (2) the hub's LFS sha256 equals it (checked on the pod, `verified: true` for all four and for `navrb2_inputs/navtext_nav_rb.npz` `f4c4ab7a...` and its report); (3) `code/pull_navrb2.py` pulled every file back from the hub into `ckpts/<tag>/` and the local sha256 equals the manifest (all four True, commit e71782a3b48c...). Zero-grad gates on the pod: PASS for all four (`e_nav_rb.weight` a trained match; 253 / 257 parameters with Adam state, `opt-step-max` 3000).

## 2. What the read computes (`navrb2_read.py` + readers; card sections 5 and 6)
* Tree: `tree/` = `git archive e71782a3 stack taniteval` (autocrlf off). Pinned `v8f_t1_eval.py` (180044eb), `v8f_mini_eval.py` (008f48b7), `ci.py`, `v8f_consistency.py`, `four_families.py` are byte-equal to the STRAT1 pins (`tree_problem`); model and data blobs are the NAVRB2 ones. Banked arms are read from the STRAT1 dumps and T0 passes (Z0: flag off is bit-identical).
* **Dumps** (`navrb2_dumps.py`, the UNEDITED otband `run_dumps.py` with a shim): `v8f_t1_eval_rb.py` with `--nav-rb on --nav-rb-variant follow` for the 4 R arms and `zero` for their 4 twins `N1R_s0Z ...` (Z-EVAL / Z-USE). Refuses an arm whose manifest tag, commit or checkpoint sha256 differs.
* **T0** (`navrb2_t0.py`, `navrb2_k4_eval.py`): real / donor / zero x 4 arms (`v8f_mini_eval_rb.py`) and K4 on both S1R arms.
* **5.1 deciding stratum**: RB = state on at NOW, **235 windows / 16 episodes** (re-counted from the table; a mismatch is a control problem); RB-TURN **43 / 6** (the 43 windows of `RESULT_NAVRB.json`, v9 lat label), RB minus TURN 192. **D1** = TAC-LAT accuracy on RB, R - banked, paired, per seed, gain = separated better and |delta| > F2_rb (F2 recomputed on RB: F from C0_rep - C0_s0, F_rep from the two C0REP same-seed pairs). **D2** beside: TURN recall on RB-TURN, false-turn share on RB minus TURN, with n. All 24 cells on RB beside, no bar.
* **5.2 four families on all windows**: LONGITUDINAL, LATERAL, TACTICAL (+ PRIMARY ADE/FDE) 24 cells x {N1R-N1, S1R-S1} x {s0, s1}, cost / gain under **F2** (read from the banked `RESULT_STRAT1_C0REP.json`, 2 pairs). STRATEGIC in full: the frozen `strat1_strategic.py` / `strat1_ctx_read.py` on a view that presents the R arms as S1 / N1 (route following real / donor / zero, RC-A, K7 / K8 / K9, detected info with K4), plus `navrb2_strategic.py` (S_CKPT and RC-A contrasts R - banked, skill difference, on-RB beside, ROUTE-COST at both seeds) and `navrb2_ctx_nofog.py` (PI "Drop fog for now": fog windows unscored with n, fog logit masked to -inf; weather constant recomputed).
* **Section 6 controls**: Z0 / Z1 / Z2 / G-CLOCK-RB are the pod / builder gates (already banked in the gate tokens and `navtext_nav_rb_report.json`; the read re-counts the strata). In the read: **Z-EVAL** (real - zero on the windows with no nav_rb frame on in the item's 16-row history: must be exactly 0.0 on every cell, window by window), **Z-USE** (beside, RB), **K1** (self-pair exactly 0.0), **K-RE** (N1-C0 and S1-C0 at s0 / s1 recomputed with F equal `RESULT_STRAT1.json`: **96 rows compared, 0 mismatches** in the dry run), K9b inside the frozen strategic reader (T1 tick 0 vs T0), K-MUT = the mutation table below.
* **Section 7 outcomes**: per pair `pair_labels` + `outcomes` (RB-NAV-ONLY crosses both pairs); every label prints its next lever.
* Reading choices written before any number (printed into the result, `navrb2_lib.READING`): r1 Z-USE "separated" = any of the 24 cells separated on RB; r2 guard = no cost at either seed under F2 and no ROUTE-COST; r3 "not separated at both seeds" = unread at seed 0 and at seed 1; r4 the T0 donor pass holds the channel at zero beside the donor nav (the evaluator swaps only the nav array, so the card's "channel follows the nav variant" cannot be met at T0; the T1 donor rule is not read); r5 the Z-EVAL mask.
* Every interval: `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0 (answers "another draw of EPISODES"; training noise = F2; inference variance does not apply). Tiers stamped: T1-kin for LONGITUDINAL / LATERAL / TACTICAL, T0 for STRATEGIC. Everything is "with the route supplied along the driven path" (card 3).

## 3. Launch (the MM)
`powershell -NoProfile -Command "$s=([wmiclass]'Win32_ProcessStartup').CreateInstance(); $s.CreateFlags=0x08000200; $d='F:\TanitAD-scratch\strat1\navrb2\read'; $r=([wmiclass]'Win32_Process').Create(\"cmd.exe /c $d\job_navrb2_read.cmd < NUL\",$d,$s); \"pid=$($r.ProcessId) rv=$($r.ReturnValue)\""` (the same call as `launch_wmi.ps1`; `job_navrb2_read.cmd` calls `Git\usr\bin\bash.exe launch_navrb2_read.sh`). Sentinel: last line of `out/navrb2_read.log` = `ZZ NAVRB2 READ DONE ZZ` or `ZZ NAVRB2 READ FAIL ZZ <why>`; file `out/NAVRB2_READ.done`. Result `out/RESULT_NAVRB2.json` + `.md`. Steps: tests, 8 dumps, 14 T0 passes, view, 5 readers; re-runnable (finished dumps and npz are skipped); each heavy step waits on `wait_commit.py 5.5`. **ESTIMATED wall 75-100 min** (the STRAT1 rates: dump 268-437 s, T0 33-40 s; the dry-run readers 2 min).

## 4. Dry runs and tests (MEASURED, this agent)
* Real checkpoints, 6-episode smoke grid, 16 windows: T1 dumps of N1R_s0, N1R_s0Z, S1R_s1 and T0 real / zero / donor of N1R_s0 and K4 of S1R_s0 ran rc 0 (`nav_rb_in` sum 9.0 in the follow dump, 0.0 in the Z twin). The 141-episode finalisation was not exercised (smoke grid).
* Stand-in read (each R arm = a copy of its banked arm): `navrb2_read.py` 111-118 s, RB 235 / 16, RB-TURN 43 / 6, 192, K-RE 96 rows 0 mismatches, R - banked exactly 0.0, label RB-UNREAD on both pairs (as it must when real - zero reads 0); the whole launcher through WMI (`job_navrb2_read_standin.cmd`): tests, view, frozen strategic (S-PARTIAL on the stand-in) and ctx reads, no-fog read (2,331 of 2,399 weather windows scored, 390 fog steps in 15 clips unscored), contrasts, read, `ZZ NAVRB2 READ DONE ZZ`. Copies in `out/dryrun/`.
* **Z-EVAL regression arm** (one value of one window of N1R_s0's zero twin changed): the full read reports `PROBLEM: Z-EVAL N1R_s0 ...` and exits 1 (`out/dryrun/RESULT_NAVRB2_zperturb_ZEVAL_RED.json`).
* `test_navrb2_read.py`: **34 passed** (whole file). `mutate_navrb2.py`: **19 of 19 mutations RED**, each on its named test (`out/MUTATIONS_NAVRB2.json`): Z-EVAL exact-zero dropped, Z-EVAL mask ignored, D1 stratum ignored, RB-CLOSES guard dropped, RB-CLOSES Z-USE dropped, RB-UNREAD one-seed, RB-NAV-ONLY pair condition, RB-SEEDS-DISAGREE, RB-COSTS, K-RE vacuous, history crosses clips, state read 5 rows ahead (the G-CLOCK-RB regression arm), table pin unchecked, bank commit unchecked, zero twin with channel on, T0 donor without the channel variant, view taking banked passes, fog logit unmasked, ROUTE-COST on one seed.

## 5. UNVERIFIED / limits
* The real 141-episode dumps and the 8 real R-arm results do not exist yet; no outcome label is claimed.
* `navrb_geval_identity.py pair|compare` (BUILD_NAVRB2 section 4, real-checkpoint G-EVAL identity at the tool's default batch) was not run here; Z-EVAL in the read covers the zero-history windows through the real evaluators.
* r4: the T0 donor route-following read holds the channel at zero (deviation from "follows the nav variant"; printed in the result).
* The Z-EVAL mask uses `cfg.hmax` = 16 rows (read from the checkpoint): a conservative superset of the rows the model uses.
* `navrb2_ctx_nofog.py`: the weather constant check is the table's own no-fog re-derivation (no published fog-free number exists); the WH nofog amendment file itself was not found on disk, the rule is taken from the MM brief / PI ruling text.
* The e71782a3 tree was exported from the repo mirror (`C:/Users/Admin/tanitad-push`); the mirror may lag the remote (memory: check by `ls-remote`); the checkpoints' gate tokens name the same commit.

## 6. Deliverable manifest
| artifact | where |
|---|---|
| 4 arms (ckpt, argv, gate, summary, zerograd, manifest) | HF private `Sayood/tanitad-v8f-strat1-runs` `<tag>/`; pod `/root/strat1/navrb2/runs/<tag>/`; local `ckpts/<tag>/{ckpt_last.pt, argv.json, BANK_MANIFEST.json}` |
| channel table + report | pod `/root/strat1/inputs/strat1/`; HF `navrb2_inputs/`; local `F:/TanitAD-scratch/strat1/navrb2/table/` |
| bank / pull scripts | pod `/root/strat1/tmp/strat1_bank_pod_navrb2.py`, `bank_navrb2.sh`, `BANK_MANIFEST_<tag>.json`, `bank_navrb2.log`; local `code/` |
| read code, tests, launcher | `navrb2_{lib,dumps,t0,k4_eval,read,strategic,ctx_nofog,view}.py`, `make_standin.py`, `test_navrb2_read.py`, `mutate_navrb2.py`, `launch_navrb2_read.sh`, `job_navrb2_read*.cmd`, `launch_wmi.ps1` (F: only, not staged, nothing in git) |
| read tree (e71782a3 export) | `tree/` (re-creatable from git) |
| dry-run and mutation results | `out/dryrun/`, `out/MUTATIONS_NAVRB2.json` |
| deletions log | `out/DELETIONS_NAVRB2_READ.log` |
| handoff | `HANDOFF.md` |
Integration to escalate: the MM launches the read, then runs TanitAD_BenchmarkCriteria on `RESULT_NAVRB2.json` and updates GOALS_AND_CLAIMS in the same turn; the read code is scratch (F:), the MM lands it if wanted.
