# DRY_MINT_PLAN: what the MM runs on the pod BEFORE and AT the RA launch (ROLLADD: own states APPENDED, logged rows kept; build 1)

RA = the RH argv plus exactly `--dagger-mode add` and its own `--out`. The RH bank `/root/rollhist/bank/sK` is REUSED (no bank step), so the arm's mint can be run on the real bank before launch. Nothing here touches the GPU, `/root/gpu_queue.lock`, `/root/N1DW_READY` or a process. Pod: `source C:/Users/Admin/lorapod/podenv.sh`; `pssh`, `pscp`; python `/root/venv/bin/python`. NOTHING BELOW HAS BEEN RUN: the build wrote no byte to the pod.

## 0. The MM owns first: RA needs NEW CODE, so a new commit
Land the changed files (`CHANGED_FILES.txt`: `v8f_dagger.py`, `v8f_mini_train.py`, `test_v8f_rolladd.py`, `golden_v8f_dagger_cf27dbb7.py`) as a commit on top of cf27dbb7 on `agent/v8f-rollhist-cdb-20261120`. Then, on the mirror (`C:/Users/Admin/tanitad-push`):
```
bash pod/make_blobs_ra.sh <RA commit> all_blobs_ra.txt            # prints rows, sha256 and the added/changed paths; the form is proven byte-identical to all_blobs_cf27dbb7.txt on cf27dbb7 (3,413 rows, 3fca0151...)
git -c core.autocrlf=false archive --format=tar -o ra_delta.tar <RA commit> -- <the added/changed paths printed above>
bash pod/fill_ra_commit.sh <RA commit> all_blobs_ra.txt ra_delta.tar   # fills __RA_COMMIT__, the blob-list sha16 and the delta sha16; refuses on a bad commit or a placeholder left; writes SHA256_RA.txt
```
The launch-gate tests (`launch_gate.py` profile fingerprints can move: FLAG_CHECKS gained `dagger_mode`; memory "Pin edit moves profile fingerprints"), the pinned suite and the land guard are the MM's. I ran none of them.

## A. Ship and verify (CPU, seconds)
```
pscp ra_env.sh ra_tree_prep.sh ra_gsuite.sh ra_gate_record.py rh_ra_chain.sh ra_arm.sh chk_arm_ra.sh ra_argv.py rh_ra_marker_watch.sh launch_ra.sh all_blobs_ra.txt ra_delta.tar "$POD_TARGET:/root/rollhist/"
pscp dscale10k_check_ra.py "$POD_TARGET:/root/rollhist/study/"
pssh 'cd /root/rollhist && sha256sum -c SHA256_RA.txt'   # (study/ file: check its line by hand) ; for f in ra_env.sh ra_tree_prep.sh ra_gsuite.sh rh_ra_chain.sh ra_arm.sh chk_arm_ra.sh rh_ra_marker_watch.sh launch_ra.sh; do bash -n $f && echo ok $f; done
```
`dscale10k_check_ra.py` is `python patch_ra.py dscale10k_check_rl.py dscale10k_check_ra.py` (the anchors refuse on any other source; the RL checker is sha256 6f7e07f5...).

## B. Tree and G-SUITE
```
pssh 'bash /root/rollhist/ra_tree_prep.sh'     # expect: ZZ RA TREE OK ZZ  (tree_cf27 must exist: rl_tree_prep.sh ran; it is blob-checked first, then tree_ra = tree_cf27 + ra_delta.tar, blob by blob against all_blobs_ra.txt)
pssh 'bash /root/rollhist/ra_gsuite.sh'        # CPU, about 36 min, when no training runs on the pod (the RL suite needed no lock); expect ZZ RA GSUITE DONE ZZ; the chain keeps a pass (gs_passed)
```

## C. Render and dry mint on the real RH bank
```
pssh 'cd /root/rollhist && /root/venv/bin/python ra_argv.py render; echo rc=$?'      # ZZ RA-RENDER PASS; runs/RA_s0, RA_s1 argv.json = RH's + `--dagger-mode add` + the new --out (+2 tokens, bank sha256 = RH's)
pssh 'mkdir -p /root/rollhist/dry_ra && cd /root/rollhist && DRY_GD=/root/rollhist/dry_ra/dry_gate_s0 DRY_SC=/root/rollhist/runs/RH_s0/selfcheck/selfcheck.json RH_READOUT=causal_g1x bash chk_arm_ra.sh mint RA_s0 > dry_ra/mint_s0.log 2>&1; echo rc=$?; tail -3 dry_ra/mint_s0.log'
```
Expect rc=0, `TOKEN-WRITTEN RA_s0`, `checks.json` all pass (C-ARGV included; G-SUITE reads `gsuite_result_ra.json`). **What this does not prove:** G-DAGGER reads RH's selfcheck (swap mode), so the add read-back is NOT exercised. Only `--selfcheck-only` on the RA argv does that (about 2 min, GPU; `ra_arm.sh` runs it before any training step, under the lock). There `ra_gate_record.py` requires: the argv names `--dagger-mode add`; the built levers carry `dagger_mode add`; the dagger probe read `rows_appended > 0`, `grad_diff_all > 0` and `control_p0_grad_diff` exactly 0.0.

## D. Mutant arms the checker must refuse (dev box, CPU; the build ran them: `mutants/checker_result.txt`, `test_log_ra_checker.txt`)
(1) a second changed flag, (2) a bank other than RH's (`bank_k`, a changed sha256), (3) `--dagger-mode swap` under the RA name, (4) the mode flag missing, (5) dose 0.3. Each must give a non-empty C-ARGV problem list. Also `test_ra_gate_record.py` (planted files; M1-M13 must refuse; `test_log_ra_gate_record.txt`).

## E. Leave the pod as found
`pssh 'rm -r /root/rollhist/dry_ra'` (own temp, by literal path); log it.

## F. Launch (MM, only after the card is frozen and the PI go with the cost)
`bash /root/rollhist/launch_ra.sh` (refuses while an RA, RL or RH chain runs; marker watcher first, then the chain, which waits on `/root/gpu_queue.lock` and holds it to the end; the `/root/N1DW_READY` protocol is `rh_rl_chain.sh` / `rh_rl_marker_watch.sh` unchanged, names aside). Last log line `ZZ ROLLHIST-RA DONE ZZ`. Then the read: `RH_AKIND=RA` (`F:/TanitAD-scratch/rollhist/read/rh_read_ra_chain.cmd`, MECH2 = `mech_ra.py`).
