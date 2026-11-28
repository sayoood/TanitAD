# DRY_MINT_PLAN: what the MM runs on the pod BEFORE and AT the RHK launch

Why: Amendments 4 to 6 (three pre-train refusals) came from paths only a pod run exercises. RHK has one more property: **its bank does not exist before the chain builds it (about 1.9 h per seed)**, so the arm's own mint cannot be run on the real K bank before the launch. This plan therefore has two phases. Phase A is everything that can be proved before the bank exists. Phase B is the arm's mint on the real K bank: it is either the chain's own mint at the arm step (a refusal costs about 2 min and a relaunch: the bank's done-marker is kept), or a dry mint after a bank-only run (`RHK_BANK_ONLY=1`, below).

Pod access: `source C:/Users/Admin/lorapod/podenv.sh`, then `pssh '<cmd>'` and `pscp`. Python on the pod: `/root/venv/bin/python`. All paths below are on the pod. Nothing here touches the GPU, the lock, a marker or a process; the pod is running other sessions' jobs.

## A. Before the bank exists (CPU, seconds)

### A0. Install and verify (MM copies; nothing else moves)
```
pscp  rh_k_chain.sh rhk_arm.sh chk_arm_rhk.sh rhk_env.sh rhk_argv.py rhk_pair.py rhk_bank_check.py rhk_gate_record.py rh_k_marker_watch.sh launch_k.sh  "$POD_TARGET:/root/rollhist/"
pscp  dscale10k_check_rhk.py  "$POD_TARGET:/root/rollhist/study/"
pssh 'cd /root/rollhist && sha256sum rh_k_chain.sh rhk_arm.sh chk_arm_rhk.sh rhk_env.sh rhk_argv.py rhk_pair.py rhk_bank_check.py rhk_gate_record.py rh_k_marker_watch.sh launch_k.sh study/dscale10k_check_rhk.py'   # equal to POD_FILES_SHA256.txt
pssh 'cd /root/rollhist && for f in rh_k_chain.sh rhk_arm.sh chk_arm_rhk.sh rhk_env.sh rh_k_marker_watch.sh launch_k.sh; do bash -n $f && echo "bash -n ok $f"; done'
pssh 'cd /root/rollhist && sha256sum study/dscale10k_check_rh.py study/dscale10k_check_rhp3.py rh_env.sh rh_gate_record.py rh_pilot_check.py tree_cdb/stack/scripts/rollhist_pair_check.py'
```
Expect: sha256 equal to the file; six `bash -n ok`; and the **untouched** files still `69ff1a0f...` (RH/B0 checker), `dcee7cc1...` (RHp3 checker), `530c8205...` (rh_env.sh), `84701957...` (rh_gate_record.py), `02a5d26f...` (rh_pilot_check.py), `0dc30d9e...` (stock pair check). None of them is replaced.

### A1. The readout-K flags and the environment (no python)
```
pssh 'cd /root/rollhist && RH_READOUT=causal_g0x bash -c ". ./rhk_env.sh; rh_readout_flags; echo \"$BANK_RO_FLAGS | $DECL_TOKEN\""; cat RH_READOUT.chosen; cat RHK_READOUT.chosen'
pssh 'cd /root/rollhist && RH_READOUT=causal_g1x bash -c ". ./rhk_env.sh; rh_readout_flags"; echo rc=$?'
```
Expect: `--readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2 | causal_g0x`; `RH_READOUT.chosen` still `causal_g1x` (not touched); `RHK_READOUT.chosen` `causal_g0x`; the second call refuses with `ZZ ROLLHIST-K FAIL ZZ RH_READOUT must be causal_g0x` and rc=1. (If `RHK_READOUT.chosen` is created here it is the chain's own file; leaving it is correct.)

### A2. The generator accepts exactly K's flags and records what the trainer will read back (CPU, imports only)
```
pssh 'cd /root/rollhist && export PYTHONPATH=tree_cdb/stack:tree_cdb/stack/scripts:tree_cdb && /root/venv/bin/python - <<EOF
import argparse, v8f_dagger_bank as G
from tanitad.data.v8f_dagger import readout_token, parse_readout_token
a = argparse.Namespace(readout="causal_g", acc_gain=0.0, path_exec=True, kappa_max="0.2")
g = G.generator_record(a); print(g, readout_token(g["readout"], g["acc_gain"], g["path_exec"]), parse_readout_token("causal_g0x"))
EOF'
```
Expect `{"readout": "causal_g", "acc_gain": 0.0, "path_exec": True, "kappa_max": "0.2"} causal_g0x causal_g0x`. (`generator_record` is the function whose output the bank header, the trainer's declaration check and `rhk_argv.py` compare.) This does not prove the bank is rolled under gain 0: only the bank step and `rhk_bank_check.py` (b: stored acceleration = 0, with the vacuity control) prove that.

### A3. The argv builder and the stock pair check on the real RH argv with a placeholder bank (no pod write)
`rhk_argv.py render` needs `bank_k/` and refuses without it (that refusal is the expected Phase-A reading):
```
pssh 'cd /root/rollhist && /root/venv/bin/python rhk_argv.py render; echo rc=$?'
```
Expect `PROBLEM s0 no /root/rollhist/bank_k/s0/bank.json: the readout-K bank is not built yet`, `ZZ RHK-RENDER FAIL`, rc=1, and **no** `runs/RHK_s*/argv.json` written. The argv builder, the pair wrapper and the gate record are proved on the real RH/B0 argv files by the dev-box tests (`pytest_log_k.txt`), with a SYNTHETIC K bank sha.

### What Phase A cannot prove
G-DAGGER at the K bank (needs the trainer's `--selfcheck-only` on the GPU), C-BANK / C-RH-BANKGEN on a real K bank (the bank does not exist), K-B5 (the re-anchoring control) under gain 0 (a bank-generator control run for the first time at this gain: if it fails, the merge refuses and the chain stops at the bank step with nothing trained), and that K's bank coverage is >= 0.5 (RH's was 0.7969 at s0; the K bank may differ).

## B. The arm's mint on the real K bank
**Route 1 (default, no extra pod time): the chain's own mint.** `launch_k.sh` runs bank_k_s0, its check, the render, then `rhk_arm.sh RHK_s0`, whose first lines are the pair check, the trainer's `--selfcheck-only` (about 2 min) and the two mints. Any refusal there stops the chain before a training step; `done_k/bank_k_s0.done` stays, so a relaunch skips the 1.9 h bank. Read `runs/RHK_s0/gate/checks.json` in full the first time it is written (21 checks, the same names as RH's, problems empty, `C-RH-PAIR.readout_token` and `C-RH-BANKGEN.readout_token` both `causal_g0x`).

**Route 2 (a dry mint on the real bank before any arm; costs one extra queue wait): `RHK_BANK_ONLY=1`.** `RHK_BANK_ONLY=1 RH_READOUT=causal_g0x nohup setsid bash /root/rollhist/rh_k_chain.sh ...` builds and checks both K banks, renders the argv, runs NO arm and exits `ZZ ROLLHIST-K DONE ZZ (banks only: no arm was run)`. Then, per seed, with RH's finished selfcheck rewritten to carry the K token as the SYNTHETIC `DRY_SC`:
```
pssh 'mkdir -p /root/rollhist/dry_k && cd /root/rollhist && sed "s/\"causal_g1x\"/\"causal_g0x\"/g" runs/RH_s0/selfcheck/selfcheck.json > dry_k/sc_s0.json && RH_READOUT=causal_g0x DRY_GD=/root/rollhist/dry_k/dry_gate_s0 DRY_SC=/root/rollhist/dry_k/sc_s0.json bash chk_arm_rhk.sh mint RHK_s0 > dry_k/mint_s0.log 2>&1; echo rc=$?; tail -n 3 dry_k/mint_s0.log'
```
Expect `rc=0` and `TOKEN-WRITTEN RHK_s0`; then read `dry_k/dry_gate_s0/checks.json`: 21 checks, all pass, same names as `runs/RH_s0/gate/checks.json`, `C-ARGV` pass, `C-RH-PAIR.readout_token` `causal_g0x`, `C-RH-BANKGEN` pass (the real K bank header against the argv). **What this does not prove:** G-DAGGER reads a SYNTHETIC selfcheck (RH's, p = 0.1 as RHK is, with only the lever token rewritten), so it proves the checker's read path and not that the trainer builds the K bank; the arm's own `--selfcheck-only` reads that first. Then relaunch without `RHK_BANK_ONLY` (the banks are skipped).

## C. Leave the pod as found, log the deletion
Own temporary data, by literal path, after the dry mint is read (Route 2 only): `pssh 'du -sk /root/rollhist/dry_k; rm -rf /root/rollhist/dry_k/dry_gate_s0 /root/rollhist/dry_k/dry_gate_s1 /root/rollhist/dry_k/sc_s0.json /root/rollhist/dry_k/sc_s1.json /root/rollhist/dry_k/mint_s0.log /root/rollhist/dry_k/mint_s1.log; rmdir /root/rollhist/dry_k'`. Log path and size in the handoff. After the chain ends, the two K banks (about 2 GB each, MEASURED for the C* banks) are re-creatable only by 1.9 h of GPU, so they are kept until the read has landed and the PI says otherwise (CLAUDE.md rule 2: never delete the only copy).

## D. Launch (MM, after Amendment 8 is frozen and the second PI go with the cost)
`bash /root/rollhist/launch_k.sh` (marker watcher, then `RH_READOUT=causal_g0x rh_k_chain.sh`, which waits on `/root/gpu_queue.lock` and holds it to its end). The last line of `/root/rollhist/rh_k_chain.log` is `ZZ ROLLHIST-K DONE ZZ` or `ZZ ROLLHIST-K FAIL ZZ <why>` (`BLOCKED coverage ...` when a K bank covers < 0.5). Then `read/rh_read_k_chain.cmd` on the dev box (WMI launch), only after both RHK arms show `ARM_DONE`.
