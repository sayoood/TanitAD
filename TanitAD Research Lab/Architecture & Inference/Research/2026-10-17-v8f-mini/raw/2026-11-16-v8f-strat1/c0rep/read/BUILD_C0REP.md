# BUILD_C0REP: bank of the two C0 same-seed replicates, and the read of the frozen addendum (built 2026-11-17, read launched 22:05:58Z)

Evidence class MEASURED (ours, zero training). Tiers: the 24 driving cells are T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; 141 episodes / 2,399 windows per arm); STRATEGIC is T0 at the window's first tick. Estimator: paired episode-cluster bootstrap B 2,000 seed 0, "another draw of EPISODES" only; the floor F_rep answers "another training run at the same seed" with 2 pairs (a weak floor). All paths are under `F:/TanitAD-scratch/strat1/read/c0rep/` unless stated. Rules: `PREREG_ADDENDUM_C0REP.md`, sha256 prefix 2f41cf5c checked (file equals `.sha256`); no rule changed.

## 1. Bank (done, verified)
* Pod chain `strat1_c0rep.sh` ended `ZZ STRAT1 C0REP DONE ZZ`; pod K-ARGV line in the chain log read ok. Both arms trained at commit a4357e79 (gate token of each run).
* `after_c0rep.sh` with SKIP_LAUNCH=1 ran `strat1_bank_pod_c0rep.py` on the pod (CPU and network only, HF_HUB_DISABLE_XET=1, token read in place). Pod dirs `runs_c0rep/C0_s10` and `runs_c0rep/C0_s11` went to the PRIVATE dataset `Sayood/tanitad-v8f-strat1-runs` under the DISTINCT tags `C0_s10r` and `C0_s11r` (hub rev 1a0917ed). Guards in the uploader: destination tag differs from the source name and ends in `r`; destination prefix empty in the repo before any upload; the hub sha256 of every original file read before and after (result `originals_unchanged: true`). The originals were present (`C0_s10/`, `C0_s11/`) and untouched.
* Hub LFS sha256 equals the manifest for every file. ckpt sha256: C0_s10r ae28eb76fe18, C0_s11r 887438234d64 (originals f69c100a5b65, 8d2d5923ac5f: the checkpoints are not byte-equal; whether the cell reads differ is what the read measures).
* Pulled to `../ckpts/C0_s10r`, `../ckpts/C0_s11r` (ckpt_last.pt, argv.json, BANK_MANIFEST.json); local sha256 equals the manifest. Pod bank scripts and manifests stay in the pod's `/root/strat1/tmp/` (`BANK_MANIFEST_C0_s10r.json`, `BANK_MANIFEST_C0_s11r.json`).

## 2. Code (all in `c0rep/`)
* `c0rep_run.py`: wrapper over the frozen readers through `s1d_run.py` / `n1_run.py` / `seeds_run.py` (unedited). Adds `C0_s10r` / `C0_s11r` to ARMS (v9 nav, not in TEXT_ARMS, not in CTX_ARMS) and lets those two tags through the bank-manifest commit check at a4357e79, these tags only.
* `c0rep_read.py`: K-PIN (frozen reader sha256 prefixes and the addendum prefix), K-ARGV (local argv files, run root `/root/strat1/runs_c0rep/` mapped to `/root/strat1/runs/`; a failing pair is left out before any number is read; with fewer than 2 pairs the label is NOT-COMPUTABLE), `f_rep`, `floor2`, `reflag` (the card cost / gain rule under another floor, equal to `obd_read.cell_delta`), `read_under` (count rule >= 3 of 4 seeds with a seed in each init), `c0rep_label` (DETERMINISTIC-AT-SEED needs delta, lo and hi exactly 0.0 on every scorable cell of both pairs, checked first; then COST-HOLDS, COST-WITHIN-FLOOR; a fourth outcome PREMISE-ABSENT is printed when TAC-LAT is not COST4_S1 under F, which the addendum does not name), `strategic_pairs` (frozen `strat1_strategic` functions: route following of replicate and original, paired replicate-minus-original contrast per arc, RC-A route setting with F and F_rep; detected info is not computable for C0, no context heads).
* Pairs recomputed with F (as the banked reads): S1-C0, N1-C0, S1d-C0, S1d-S1, S1-N1 at seeds 0 / 1 / 10 / 11, the two replicate pairs, and C0_s10-C0_s0 / C0_s11-C0_s1. COST4 / GAIN4 under F2 are re-derived from those rows by `reflag` (same bootstrap rows, only the floor changes); the S1D label and the N1 attribution are re-read under F and F2 through the earlier readers' own functions.
* Controls: K1 self-pair 0.0; K-RE against RESULT_STRAT1.json, RESULT_STRAT1_SEEDS.json, RESULT_STRAT1_N1S10.json and RESULT_STRAT1_S1D.json (a file with no compared row fails); K-ARGV; K-REFLAG (re-derivation under F reproduces every recomputed row's cost / gain); K-PIN.
* `strat1_bank_pod_c0rep.py`, `bank_c0rep.sh`, `after_c0rep.sh`: bank, pull, verify. `launch_c0rep_read.sh`, `launch_wmi.ps1`, `job_c0rep_read.cmd`: launch. Order in the launcher: commit headroom (6 GB), unit tests, 2 T1 dumps, 6 T0 passes (real / donor / zero, no K4), reader. Dumps and T0 run on CPU like the dumps they are compared with (device cpu is fixed in the frozen runners; a GPU would move the numerics K-RE checks), so the shared GPU lock is not taken.
* `test_c0rep_read.py`: 28 tests green. Per rule: reflag sides and floor edge, F_rep / F2, COST4 / GAIN4 under F2, five labels, end-to-end label swap with the floor; mutations: count threshold 3 to 4 and 3 to 2 flip, one-init-only never counts, K-ARGV with a changed value / an added lever / a length change / a different run root, pin prefix change, commit widening for the two tags only, manifest naming the original tag refused, and the uploader script itself run against a fake hub (distinct tags only, refuses a non-empty destination, refuses missing originals, refuses destination equal to source, and exits non-zero when an original changes).

## 3. Read
Launched detached through WMI at 22:05:58Z (wrapper cmd PID 32964), log `out/c0rep_read.log`, stdout `out/c0rep_read_stdout.log`. Done line `ZZ STRAT1 C0REP READ DONE ZZ`; fail line `ZZ STRAT1 C0REP READ FAIL ZZ <why>`. Result: `out/RESULT_STRAT1_C0REP.json` and `.md`. ESTIMATED 30 min in total (two dumps about 6 min each, six T0 passes about 35 s each, reader about 6 min).

## 4. What is not exercised
* The real replicate dumps and T0 passes had not finished when this was written; the result is not read.
* Smoke of the reader before the dumps existed (`out/smoke/`): each replicate was aliased to its original's dump. K-PIN, K-ARGV, K1, K-RE (960 rows over four banked files, 0 mismatches), K-REFLAG (576 rows) all passed; the label read DETERMINISTIC-AT-SEED as it must for a bit-identical pair, and the COST4 re-read reproduced the banked S1-C0 TAC-LAT cost at seeds 0 / 10 / 11. The STRATEGIC path was smoked the same way on copies of the original T0 passes (route following, contrast and RC-A all read 0.0 as they must). The smoke reads say nothing about the replicates.
* The smoke aliasing hook (`C0REP_SMOKE_ALIAS=1`) is off in the launcher.
* K-ARGV was checked on the pod (chain log) and is re-checked locally by the reader from the pulled argv files.

## Deletions (own temp, literal paths)
* `F:/TanitAD-scratch/strat1/read/c0rep/out/smoke/t0alias` 36.8 MB (copies of existing T0 passes, re-creatable). The launcher deletes `c0rep/bt` (pytest basetemp) itself. Pod `/root/strat1/runs_c0rep` (167 MB) is kept: its checkpoints are banked and verified, but the run folders are the pod's own and the MM decides.

## Manifest
| artifact | where |
|---|---|
| replicate checkpoints | HF private `Sayood/tanitad-v8f-strat1-runs` `C0_s10r/`, `C0_s11r/`; local `../ckpts/C0_s10r`, `../ckpts/C0_s11r`; pod `/root/strat1/runs_c0rep/C0_s10`, `C0_s11` |
| code, tests, launcher, this note | `c0rep/*` (F: only, none in git, nothing staged) |
| smoke reads | `out/smoke/` (F: only) |
| read result | `out/RESULT_STRAT1_C0REP.json`, `.md` (written when the read ends) |
| dumps / T0 | `../t1/dump_STRAT1-C0_s1{0,1}r_m10_causal_heads`, `../out/t0/eval_STRAT1-C0_s1{0,1}r_*` (F: only; re-creatable from the banked checkpoints) |
