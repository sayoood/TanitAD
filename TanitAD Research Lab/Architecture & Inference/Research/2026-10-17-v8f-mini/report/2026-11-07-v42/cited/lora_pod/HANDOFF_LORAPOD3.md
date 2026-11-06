# HANDOFF_LORAPOD3 (pod clock 23:10 UTC, dev-box clock 00:10, 2026-11-07)

Pod address: `C:/Users/Admin/lorapod/pod_direct.txt`, helpers `podenv.sh` (`pssh`, `pscp`). No address in this file or in `lp3/`. All new files: `E:/Projects/TanitAD-artifacts/dagland/motion_c/lora_pod/lp3/` (nothing staged); deployed to pod `/root/ops/` (sha256 list `/root/ops/DEPLOYED_SHA256_LP3.txt`, equal to local).

## 1. L's argv, seed 0 (`lp3/l_argv_s0.json`, sha16 `ef77b73af0e9e67a`, `l_argv_diff.json`)
MEASURED: B' s0's argv, only `--out` changed (`/root/runs/lora/L_s0`), plus `--encoder lora --lora-r 32 --lora-alpha 32 --lora-targets q,v --frames-root /root/data/v2ep/train --accum 16 --enc-ckpt`. No `--amp`. Machine diff: 10 checks, ok. Two choices of mine: frames root is `.../v2ep/train` (FrameStore reads `<root>/<clip>.v2ep.pt`; `/root/data/v2ep` holds only `train/` and `eval/`); `--accum 16` (4 windows, 64 unique frames per outer chunk; every encoder call is the builder's 8 frames). Neither argv names `--ckpt-every`: both save `ckpt_last.pt` every 500 steps. L logs every 50 steps.

## 2. Timing
* **PRELIMINARY (Master Mind request), RUNNING since 23:03 UTC**, niced, L's exact argv + the trainer's own `--max-clips 192` (first 192 clips of the partial cache manifest; all have cache and v2ep files), beside the frame build and the encoder. Step 1 took 70.9 s (warm-up). Expected end about 23:30 UTC. Result: `/root/runs/lora/timing_prelim_b64/PRELIM_SUMMARY.json` (median s/step 6-20, peak memory, 19,164-step projection ESTIMATED, CPU contention record: load average, cgroup throttling, GPU snapshots, caveats) and marker `DONE` in `/root/logs/chain3_prelim.log`. Read the contention note with it: the build uses 13 workers on a 15.3-CPU quota.
* **Chain 3 re-times** with the full clip set (`timing_b64`): at once beside B' if the median GPU utilisation over 10 minutes (B' past step 300) is under 70 %, else when B' ends.

## 3. Checkpoint sidecar (running, pid in `/root/logs/bank_ckpts.lock`)
`bank_ckpts.py` -> PRIVATE HF dataset repo `Sayood/tanitad-lorapod-ckpts` (created, private verified). Copies `ckpt_last.pt` of B' s0, s1 and L when the step is a multiple of 1,000 (and the final one of a done run); stable-size wait, no `.tmp`, copy checked unchanged; remote size and sha256 streamed back (LFS id cross-checked). Real-hub test MEASURED: 12 MB LFS file and tiny file verified. Left on the hub: `_selftest/` (3 small files; not deleted). Ledger `/root/logs/ckpt_bank_ledger.jsonl`, log `/root/logs/bank_ckpts.log`, staged copies `/root/ckpt_bank/`.

## 4. Chain 3 (RUNNING, waits at `B`; `/root/ops/chain_lorapod3.sh`, log `/root/logs/chain3.log`)
Markers: `START, A-ARGV, A-INPUTS, A-HELPERS (done) | B-CHAIN2-LAUNCHED, C-UTIL (UNDER|FULL), D-TIMING-START, D-TIMING, E-GATE-START, E-GATE-L, E-TOKEN-L, F-BESIDE-DECISION, F-BPRIME-DONE, F-TOKEN-VERIFIED, F-L-SUPERVISOR, L-LAUNCHED, END`. `WARN` lines: `B-CHAIN2-STOP-SEEN`, `D-TIMING-FAILED` (the chain goes on). STOPs: `ZZ CHAIN3 STOP <NAME> ZZ` (A-TREE, A-ARGV-DIFF, A-NO-DECISION, A-DECISION-SHA, A-ENCODER-SHA, C-BPRIME-STALLED, C-UTIL-NO-SAMPLES, E-TREE, E-GPU-BUSY, E-GATE-RUN, E-GATE-L, E-TOKEN, F-TOKEN, F-L-REFUSED, F-L-NOSTART, F-L-NOT-ON-GPU, *-BPRIME-STALLED, *-GPU-NOT-FREE). Beside rule: L concurrent step <= 1.15 x L alone (reference = chain 2's smoke projection, ESTIMATED, flagged) and B' slowed <= 30 %; numbers recorded either way (`state/f_beside.json`). Token route = chain 2's (in-process gate, chain-written, decision sha `91ab7b83`, encoder sha `dcb2e451`); `launch_gate.py` untouched.
Evidence: `chain3_dry_run_markers.txt` (12 scenarios, GPU steps stubbed: ok_full, ok_under, beside, beside_slow, chain2_stop, timing_fail, and the STOPs gate_fail, bprime_dead, token_tamper, refuse, nodecision, bad_decision_sha), `pytest_lp3_tools.txt` (12 pass, single-fault arms for argv, gate, token, B' state, utilisation, timing, beside, sidecar).

## 5. Progress commands
`tail -n 5 /root/logs/chain3.log; cat /root/logs/chain3_prelim.log; tail -n 3 /root/logs/ckpt_bank_ledger.jsonl; tail -n 1 /root/runs/bprime/M2b_s{0,1}/train_log.jsonl /root/runs/lora/L_s0/train_log.jsonl`
Restart after a STOP: `setsid nohup bash /root/ops/chain_lorapod3.sh < /dev/null > /root/logs/chain3.stdout 2>&1 &` (steps with `state/done_<A..F>` are skipped; state dir `/root/runs/lora/state`).

## Not done / gaps
* No GPU run of the real gate, token or timing at the full clip set yet (stubbed in the dry run; the preliminary run exercises the real timing driver and trainer path). The L gate's reader was built on AR-2 run 3's real L-path selfcheck and M2b's banked gate; a first real L gate can still show a field I did not expect (it would STOP at `E-GATE-L` with the failing check named).
* The 70 % rule in sequential mode measures s0 alone (marker says so).

## Exact next step
Read `PRELIM_SUMMARY.json` (about 23:30 UTC) for the PI's venue choice. Then watch `chain3.log` from about 01:15 UTC for `B-CHAIN2-LAUNCHED` and `C-UTIL`.
