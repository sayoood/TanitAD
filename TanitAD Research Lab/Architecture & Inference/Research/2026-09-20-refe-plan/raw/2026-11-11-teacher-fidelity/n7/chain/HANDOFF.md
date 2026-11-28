# HANDOFF TF-N7 chain (built and tested, NOT launched; no navtest token built, scored or read)

2026-11-13, engineering agent for the REFe MM. Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n7_chain/` (**single copy: not in git, not on HF**). **Integration needed from the MM: freeze the card, build the run tree, ledger n = 50, launch; and land this hold under `PK/raw/2026-11-11-teacher-fidelity/n7/chain/` (the longest file name here is under 60 characters; the full path stays under 200).**

## State
- Chain `n7_nt_build.py` (derive / merge / unch / reach), `n7_nt_chain.py`, `n7_nt_analyze.py`, `n7_make_pins.py`, `build_run_tree.sh`, `launch.cmd.template`, `N7_PINS.template.json`, `test_n7_chain.py` (87 tests, all passed in 4 min incl. the real-token execution test; log `test_run_n7_chain.log`; the first attempt failed because the test imported numpy before BLAS threads = 1, fixed in the test: a 1e-6 plan drift, the same trap as the build, which sets the threads first), `CHAIN_N7.md` (what changed vs N6, file:line), `PREREG_TF_N7.md` (card, MM rulings applied, no `.sha256`), `dryrun_check_N7.log`, `smoke/` (real navtrain smoke, `SMOKE_N7.md`).
- Real navtrain smoke: 20 tokens through the real CLI, **15 of 15 build controls PASS, selection equals the development on 20 of 20** (N7-T3 16, N7-T 19, N7-X 0); every tier executes on a real token; all selector mutations RED on real rows except `nc_drop` (not binding on these tokens, reported in `SMOKE_N7.md`).

## Exact freeze + launch (commands, MM)
```
H=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n7_chain ; R=C:/Users/Admin/n7nt/run
1. bash $H/build_run_tree.sh $R                      # copies of the N6 run tree + n6/ + n7/ + inputs/n6t (140 MB); refuses an existing $R/n7
2. fetch the N6-T plan file from HF (Sayood/tanitad-refe-scorer1p6, evidence/2026-11-13-tf-n6, seek inside the run tar as for N5-A)
   -> $R/inputs/n6t/n6_plans_N6T.npz ; sha256 MUST be 1c79202f25b6747799d7cb6512d4985a7490cfc7619433c969c4d8a67c93a5ee
3. cp $H/PREREG_TF_N7.md $R/n7/PREREG_TF_N7.md       # (the card after YOUR freeze edits) ; CARD=$(sha256sum $R/n7/PREREG_TF_N7.md)
4. ledger: $R/inputs/ledger_K.json is the banked ledger (max n = 49, the TF-N6 entry); re-read the max n, then append
   {"key": "TF-N7:<CARD>:E3", "n": 50, "nominee": "TF-N7", "family": "TF", "set": "E3", ...}   (the git mirror's ledger was not readable here)
5. pins: cp $H/N7_PINS.template.json $R/n7/N7_PINS.json ; PINS = sha256 of it (the template: b1967406eb77e1fae2bc0eba1d28e8f6fff70a26cb2f73a41eddca9da0facb50),
   valid only if every pinned file is byte-identical to this hold's (it was built from the same copies; the 958 N6 pins are byte-identical to N6's).
   If anything differs: N4_RUN=$R python $R/n7/n7_make_pins.py --run-dir $R --tip <sha>  (navsim venv) and use the new sha.
6. guards: C:/Users/Admin/navsim-crun/venv/Scripts/python.exe $R/n7/n7_nt_chain.py --run-dir $R --card $R/n7/PREREG_TF_N7.md --card-sha256 <CARD> \
     --pins-sha256 <PINS> --ledger $R/inputs/ledger_K.json --k 50 --procs 2 --min-free-gb 5 --check-only      # prints ZZN7 GUARDS PASS
7. launch: fill <CARD_SHA256> and <PINS_SHA256> in $H/launch.cmd.template, start it detached through WMI Win32_Process Create (as TF-N6); log $R/n7run/chain.log; the end line is ZZN7_CHAIN DONE
```
Never `--mutate`. A `.FAIL` marker under `$R/n7run/markers/` stops the chain (`ZZN7_CHAIN STOP`); remove it after the cause is fixed and re-issue the same launch (resumable).

## Cost (ESTIMATED from TF-N6 MEASURED stage times + the navtrain smoke)
- Wall **3.8 to 4.6 h**: guards 5 s (+ cache listing) ; reach_real 3 to 5 min ; derive 0.7 to 1.1 h (TF-N6's 0.38 CPU-h + 20 tokens x 102 s MEASURED on navtrain = 0.57 CPU-h; cache-route cost per token not measured) ; build 10 min ; unch 5 min ; score both seams 1.2 to 1.7 h ; read 2 min ; families 55 to 70 min (three arms).
- CPU only, 2 processes, no GPU, no pod. **Disk:** run tree 140 MB + outputs about 0.6 GB; C: had about 5.8 GB free (`--min-free-gb 5`, as TF-N6; F: has about 11 GB, below the default 20).

## UNVERIFIED / caveats
- `inputs/n6t/n6_plans_N6T.npz` is **not on disk** (HF only): the two banked layers of C-N6SAME run for the first time on navtest (synthetic tests cover them).
- Navtest cache-route derive time and the families per-token speed path (`...block._per_window.longitudinal.speed_mae_mps.value`) are not read from a real file; the latter is reported UNAVAILABLE rather than dropped.
- The ledger in the git mirror was not readable (mirror HEAD is old); the tree's copy (max n 49, TF-N6 entry present) was used.
- The pins template inherits the metric-cache listing digest from `N6_PINS.template.json` (same cache, 12,146 files); the guard recomputes it at launch.
- The tier order, the 0.5 floor and every lever parameter were set on the same 17 navtrain tokens (n = 17 on 7 logs); the T4 admissibility as "teacher output" is the PI's decision (the card runs both arms).
- Tests leave their temp dirs (as the TF-N6 tests); the DEV candidate tarball is read in memory.

## Deletion log (CLAUDE.md rule 4)
- Deleted `C:/Users/Admin/n7chain` (the agent's own scratch, 375 MB: scratch run tree 130 MB, `tmp` 217 MB = pytest temp dirs of earlier test runs, dev candidates 20 MB, smoke 8 MB, mini 2 MB), by literal path, at the end; everything worth keeping was copied into this hold first (smoke artifacts, logs, templates). One earlier compound `rm -rf ... find -exec` of the `tmp` part was refused by the permission system mid-run and not retried by any other route.
- The 87-test runs also leave `tempfile.mkdtemp` dirs (a few MB each) under the system temp (`C:/Users/Admin/AppData/Local/Temp/tmp*`); they are not under a literal path of this job and were not removed.
