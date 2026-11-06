# HANDOFF_LORAPOD2 (dev-box clock 23:18, pod clock 22:18 UTC, 2026-11-06)

Pod address: `C:/Users/Admin/lorapod/pod_direct.txt`, helpers `podenv.sh` (`pssh`, `pscp`). No address in this file or in `lora_pod/lp2/`.

## State
The chain is RUNNING, detached, waiting at `A-WAIT` (pid 568659, parent init; markers in `/root/logs/chain2.log`; kill only by that explicit pid). It polls `/root/logs/cache_pod.log` every 60 s for `ZZ CACHEPOD DONE ZZ`.
Cache at 22:17 UTC: 1,050 / 4,713 encoded, build 1,850 / 4,713. ETA ESTIMATED: build done about 23:50, cache done 00:40 to 01:05 UTC. Then: smoke about 15 min, two gates about 10 min, B' launch about 01:15 to 01:40 UTC.

## Deliverables (all under `E:/Projects/TanitAD-artifacts/dagland/motion_c/lora_pod/lp2/`; nothing staged)
1. `bprime_argv_s0.json` (sha16 `a2b68127d72210f0`), `bprime_argv_s1.json` (`c226fba964961dc5`), `bprime_argv_diff.json`. MEASURED: banked M2b argv (4 repo sources agree, recorded sha `1225d98ff87cbe17` / `38fb08854f5bdb33` reproduces) vs B': 8 of 34 tokens differ, all paths (cache root, native_tracks beside it, join, p10, v9/lead/speed, out dir); zero other differences; the two seeds differ only in seed and out. Path map: `cache_full` to `/root/data/cache_pod`, run root to `/root/runs/bprime`.
   * **Disclosed:** the 4,572-clip v9 release (md5 `83e30bd3...`), `lead_train4572.npz` and the speed sidecar are NOT on the dev box (box 2 only). They are read only to build the join. The banked join (sha256 `978f41bf...`, equal to box 2's per HGATE / SPEEDCAL) is shipped to `/root/runs/bprime/join_train.npz`; the argv still names the v9/lead/speed paths, which do not exist on the pod and are never opened (`v8f_mini_train.build_data`, and a real gate run on the partial cache got past the join load).
2. `smoke_argv.json`: AR-2 run 3's recorded smoke argv with pod paths (sha16 `ea090921abb39d49`), frames `/root/data/v2ep/train`, cache reference `cache_pod`, `--max-clips 4`, 200 steps, batch 1 (the later `--batch 1` wins, as AR-2 recorded).
3. `tools/chain_lorapod2.sh` (also `C:/Users/Admin/lorapod/chain_lorapod2.sh`, pod `/root/ops/`), `lp2_tools.py` (readers, token), `lp2_run_trainer.sh`, `lp2_detach_mutant.py`; deployed sha256 equal local (`/root/ops/DEPLOYED_SHA256.txt`).
4. Evidence: `pytest_lp2_tools.txt` (31 pass: every reader runs on real banked artifacts, then 30 single-fault arms go RED), `chain_dry_run_markers.txt` (12 scenarios, every GPU step stubbed, real readers: happy concurrent, happy sequential, and 10 STOPs, each with its named marker), `preflight/`.

## Markers (`/root/logs/chain2.log`)
`START, A-TREE, A-INPUTS, A-WAIT, A-CACHE-DONE, A-COUNTS | B-SMOKE-START, B-SMOKE-OK (s/step, peak, identity), B-MUTANT-RED | C-GATE-s0, C-GATE-s1, C-LAUNCHGATE-NO-PROFILE, C-DECISION, C-TOKEN-s0/s1 | D-TOKEN, D-S0-START, D-S0-SOLO, D-S1-START, D-PAIR-PROBE, D-MODE, LAUNCHED, END`. Failures: `ZZ CHAIN2 STOP <STEP-NAME> ZZ`. A re-run skips steps with `/root/runs/lora_pod/state/done_<A|B|C|D>`.

## BLOCKER for the Master Mind (headline)
**`stack/scripts/launch_gate.py` has no profile for `v8f_mini_train.py`** (every profile binds `refc_v3_train.py`; probe in `state/c_profile_probe.json` once C runs). Amendment 4 item 5 asks for a `launch_gate.py` PASS token, which it cannot mint, and the file must not be edited. The chain therefore runs the trainer's in-process gate for s0 and s1 (read against M2b's banked gate: 4,512 clips, 306,566 windows, train and holdout sha equal), then STOPS at `C-NO-LAUNCHGATE-PROFILE`, GPU idle after the smoke, unless you do ONE of:
* record the v8f-mini precedent (in-process gate plus chain-written token, as the orc chain) in `/root/ops/DECISION_INPROCESS_TOKEN.txt` (any text naming your decision; its sha256 goes into both tokens). Do it before the cache ends and the chain proceeds by itself; after a STOP, start the chain again (command below);
* or add a reviewed v8f-mini profile and run `launch_gate.py` yourself (the chain does not drive it; it stops with `C-LAUNCHGATE-PROFILE-EXISTS`).

## Measured early (preflight, partial cache, outputs in `preflight/`)
* Control smoke, 8 steps, 4 declared clips: training-path identity probe rel L2 **0.0**, bitwise 1.0, on this GPU against `cache_pod`; 1.20 s/step median (CPU shared with the encode), peak 1.44 GiB; frozen DINOv3 and prior gradients exactly 0.0.
* Detach mutant on the real weights: RED at the trainer's G-LIVE (`cached_grad_step`: "element 0 of tensors does not require grad"). The reader requires that signature, so an unrelated crash does not count as RED.
* `check-cache` on the real partial cache: eval 141 equals box 2's clip set and n_frames; every present train row passes; only the count and clip-set checks fail (incomplete).

## Not done / gaps
* B' checkpoints are not copied to HF (not allowed in this task; B' keeps the banked `--ckpt-every 500`). L's argv, gate and token are not prepared.
* Concurrent vs sequential is decided live by `D-PAIR-PROBE` (rule: aggregate steps/s of the pair at least 1.15 x s0 alone and GPU memory under 19,000 MiB). Banked M2b ran 15,741 s per seed on box 2 (MEASURED); the pod figure is unmeasured.

## Commands
* Progress: `tail -n 5 /root/logs/chain2.log; tail -n 1 /root/runs/bprime/M2b_s{0,1}/train_log.jsonl; tail -n 1 /root/logs/cache_pod.log`
* Restart after a STOP: `timeout 30 ssh <pod> 'setsid nohup bash /root/ops/chain_lorapod2.sh < /dev/null > /root/logs/chain2.stdout 2>&1 &'`
* Stop a run: supervisor pid in `/root/ops/sup/bprime-s<n>.sup.pid`, trainer pid in `/root/ops/sup/heartbeats/bprime-s<n>.json`; kill by explicit pid only.

## Exact next step
Master Mind decides the token route (above). Then read `/root/logs/chain2.log` at about 01:00 UTC; on `LAUNCHED`, record the mode and the first s/step, and put B' evals (eval cache is `cache_pod/eval`) on the next agent's list.
