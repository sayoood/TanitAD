# HANDOFF_B2 (ROLLHIST, phase B2 prep) 2026-11-20
State: pod tree `/root/rollhist/tree` = 2bda9e1a, 3,458 of 3,458 blobs verified, 0 bad. Scripts staged in `/root/rollhist/` (copies in `F:/TanitAD-scratch/rollhist/build/b2/`). NOTHING started; no pytest, GPU or model on the pod. Disk 102 GB free (gate 60).
Inputs found, read only: UE_s0.A / UE_s1.A `ckpt_last.pt` under `/root/dscale10k/runs/UE_sK/A/` (sha256 match 2fd4b747.., f13df8ce..); templates `/root/dscale10k/argv/U_sK.B.tmpl.json`; union data under `/root/dscale10k/merged/` and `/root/data/cache_union_dscale`. G-SUITE base = `/root/strat1/junit/base`.

Chain `rh_chain.sh` (one process, log `/root/rollhist/rh_chain.log`, markers `done/*.done`): flock fd 9 -> tree/inputs/disk -> render + pair -> G-SUITE -> pilot (checks a-d, refuses) -> bank s0 + re-check -> RH_s0 -> B0_s0 -> bank s1 -> RH_s1 -> B0_s1. Last line `ZZ ROLLHIST DONE ZZ` or `ZZ ROLLHIST FAIL ZZ <why>`.
Readout = ONE parameter `RH_READOUT=causal_g1x|causal_g1`, no default (refused if unset; fixed by the first launch in `RH_READOUT.chosen`).

LAUNCH (MM; the watcher exits 2 if no chain process exists within ~2 min, so start both together):
1. `source C:/Users/Admin/lorapod/podenv.sh; pssh 'nohup setsid bash /root/rollhist/rh_marker_watch.sh < /dev/null > /dev/null 2>&1 &'` (sets /root/N1DW_READY; removes it when the chain holds the lock)
2. `pssh 'RH_READOUT=causal_g1x nohup setsid bash /root/rollhist/rh_chain.sh < /dev/null > /dev/null 2>&1 &'`
3. `bash C:/Users/Admin/actsens_launch/wait_rollhist.sh` (exit 0 DONE, 1 FAIL, 3 timeout)
Resume after a FAIL: fix, relaunch the same line (same RH_READOUT); finished steps are skipped, a partial arm is moved aside.

Both readouts render. causal_g1x passes the stock pair check. **causal_g1 is REJECTED by the stock `rollhist_pair_check.py`** (it hard-codes causal_g1x); `render_argv_rh.py` patches DECL in process (committed code untouched) and records the stock rejection. Check (b), the 1e-6 acceleration identity, is asserted for causal_g1x only (stored pose = executed knots only under --path-exec); for causal_g1 it is reported.

Open for the MM:
1. Gate record: `rh_gate_record.py` mints a CHAIN-LOCAL record (G-SUITE, pilot, bank check, pair, trainer selfcheck of the arm's own argv, tree). It is not a dscale10k_check / launch_gate token: that checker knows only DSCALE10K arm names. Say if you want a study-side RH kind first.
2. Zero-gradient gate = chainlib `post` (252 registered tensors, all with Adam state, all moved vs init); `zerograd_gate.py` runs as a report only.
3. Pilot (a): the defaults pilot of the new tree vs the unpatched DSCALE10K tree, 0.0 diff; on GPU run-to-run noise a second unpatched pilot sets the floor.
4. Banks (size unknown) stay on /root after the arms; the chain fails, not deletes, if free disk < 60 GB before the s1 bank. Sweep them once the checkpoints are banked on HF.
5. After the chain pull `runs/{RH,B0}_sK/train/ckpt_last.pt`; K-CLIP in `runs/*/kclip_*.json`, selfcheck in `runs/*/selfcheck/`; eval reads are dev-box side.
6. Untested on the pod: pilot_check, gate_record, rh_arm paths (compiled, argparse-checked, bash -n only). The first pilot is the first real run; expect a possible path fix.
Temp: local tars and tree export deleted; pod tarball deleted (DELETIONS.log, /root/rollhist/deletions.log).
