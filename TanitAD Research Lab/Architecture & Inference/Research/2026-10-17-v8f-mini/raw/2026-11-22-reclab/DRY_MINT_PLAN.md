# DRY_MINT_PLAN: what the MM runs on the pod BEFORE and AT the RL launch (the speed-consistent label lever on own states; build 2, replaces the pose-recovery label)

RL = the RH argv plus exactly `--dagger-label speed_v1` and its own `--out`. The RH bank `/root/rollhist/bank/sK` is REUSED (no bank step, so the arm's mint can be run on the real bank before launch). Nothing here touches the GPU, `/root/gpu_queue.lock`, a marker or a process.
Pod: `source C:/Users/Admin/lorapod/podenv.sh`; `pssh`, `pscp`; python `/root/venv/bin/python`.

## 0. Blocker the MM owns first: RL needs NEW CODE, so a new commit
The flag lives in `stack/tanitad/data/v8f_recovery.py` (new), `v8f_dagger.py`, `stack/scripts/v8f_mini_train.py` and three test files (list: `CHANGED_FILES.txt`), built on **97fe0a56**, not the arch-inf tip (FACTS section 0). The pod tree (`tree_cdb`) is 97fe0a56 and is blob-pinned (`all_blobs_97fe0a56.txt`, `COMMIT=` in `rh_env.sh`, `COMMIT =` in `rh_gate_record.py`). So before any RL step: land the change as a commit on the rollhist branch, build the new blob list, repoint the pins as `repoint_97fe.sh` did, ship the tree, **re-run G-SUITE at the new commit** (`gsuite_result_rh.json` binds to the commit) and the pilot record, and run the launch-gate tests (FLAG_CHECKS gained `dagger_label`; `launch_gate.py` profile fingerprints may move: memory "Pin edit moves profile fingerprints"). I did none of this.

## A. Install and verify (CPU, seconds)
```
pscp rh_rl_chain.sh rl_arm.sh chk_arm_rl.sh rl_argv.py rh_rl_marker_watch.sh launch_rl.sh "$POD_TARGET:/root/rollhist/"
pscp dscale10k_check_rl.py "$POD_TARGET:/root/rollhist/study/"
pssh 'cd /root/rollhist && sha256sum rh_rl_chain.sh rl_arm.sh chk_arm_rl.sh rl_argv.py rh_rl_marker_watch.sh launch_rl.sh study/dscale10k_check_rl.py'    # equal to POD_FILES_SHA256.txt
pssh 'cd /root/rollhist && for f in rh_rl_chain.sh rl_arm.sh chk_arm_rl.sh rh_rl_marker_watch.sh launch_rl.sh; do bash -n $f && echo ok $f; done'
```
`dscale10k_check_rl.py` is produced by `python patch_rl.py dscale10k_check_rhk.py dscale10k_check_rl.py` (anchors refuse on any other source). The untouched files (RH/B0, RHp3, RHK checkers, `rh_env.sh` after the repoint, stock pair check) are not replaced.

## B. Render and dry mint on the real RH bank (the arm's own mint, before the launch)
```
pssh 'cd /root/rollhist && /root/venv/bin/python rl_argv.py render; echo rc=$?'
```
Expect `ZZ RL-RENDER PASS`, `runs/RL_s0/argv.json` and `RL_s1` written, each = RH's argv + `--dagger-label speed_v1` + the new `--out` (differs from RH in exactly `--dagger-label`, `--out`; +2 tokens; bank sha256 = RH's).
```
pssh 'mkdir -p /root/rollhist/dry_rl && cd /root/rollhist && DRY_GD=/root/rollhist/dry_rl/dry_gate_s0 DRY_SC=/root/rollhist/runs/RH_s0/selfcheck/selfcheck.json RH_READOUT=causal_g1x bash chk_arm_rl.sh mint RL_s0 > dry_rl/mint_s0.log 2>&1; echo rc=$?; tail -3 dry_rl/mint_s0.log'
```
Expect rc=0, `TOKEN-WRITTEN RL_s0`, `checks.json` all pass, `C-ARGV` pass. **What this does not prove:** G-DAGGER reads RH's selfcheck (human label), so the label lever read-back is NOT exercised; only the trainer's `--selfcheck-only` on the RL argv does that (about 2 min, GPU; `rl_arm.sh` runs it before any training step). Expect there: `info.dagger.speed_label` with `n_labelled`, `n_emitted`, `n_eligible` == `n_eligible_after` (the eligible index is unchanged), the `census` (per tick ok / relabelled / identity / refused by reason, p50 / p95 of |v_own - v_h[0]| and |delta[30]|, relabelled share), `wall_s` (REAL_SHARD_CENSUS.json measured it), `levers.dagger_label == "speed_v1"`, `dagger_probe.rl_label_rows_changed > 0`.

## C. Mutant arms the checker must refuse (run on the dev box, CPU)
(1) a second changed flag, (2) a bank other than RH's (`bank_k`, or a changed sha256), (3) `--dagger-label human` under the RL name, (4) the label flag missing, (5) dose 0.3. Each must give a non-empty `C-ARGV` problem list.

## D. Leave the pod as found
`pssh 'rm -r /root/rollhist/dry_rl'` (own temp, by literal path), logged in the handoff.

## E. Launch (MM, only after the card/Amendment 9 is frozen and the PI go with the cost)
`bash /root/rollhist/launch_rl.sh` (marker watcher, then the chain, which waits on `/root/gpu_queue.lock` and holds it to the end; the `N1DW_READY` protocol is the RHp3 chain's, unchanged). Last log line `ZZ ROLLHIST-RL DONE ZZ`. Then the read: `RH_AKIND=RL`.
