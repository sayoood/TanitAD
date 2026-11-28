# HANDOFF_DATASCALE (2026-11-15, written ~18:32 pod time)

## State
* SCOPE frozen: `F:/TanitAD-scratch/datascale/SCOPE_DATASCALE.md`, sha256 `b558ee81f995f8f7...` (full value in `SCOPE_SHA256.txt`; pod copy matches).
  Pool under the scope = **133,929 clips** (not 306k; NVIDIA test/val + our corpora excluded). Pilot = first 200 of the seeded stratified order (seed 20261115) + 3 controls.
* Pilot launched detached on the pod: `bash /root/datascale/ds_run.sh` (pid in `pilot.log`). Phases: stage (done, 3,952 clips/h, 203/203 ok) -> build_ctl -> encode_ctl (done)
  -> build_pilot (running, 13 workers) -> encode_pilot -> join_pilot (obstacle join, best effort) -> labelprobe (best effort) -> ds_report.
* CONTROL GATE PASSED (3 clips already in our corpus, re-run through the new route): mp4 sha256 = ledger, v2ep sha256 = ledger (bitwise), pool4x10/pool8x20 bitwise equal (frac 1.0, rel-L2 0.0)
  to `/root/data/cache_pod`, kinematics equal, keys equal; intrinsics max abs diff 0.0 vs `intrinsics_dev.csv`.
* Resumable: re-running the same launch command skips finished work (ledgers `ledger_stage.jsonl`, `ledger_pilot.jsonl`, `cache_pilot/train/manifest.jsonl`).

## Read the result (from the dev box)
    source C:/Users/Admin/lorapod/podenv.sh
    pssh 'tail -5 /root/datascale/pilot.log'
DONE: last line `ZZ DATASCALE-PILOT DONE ZZ encoded=<n>`. FAIL: `ZZ DATASCALE-PILOT FAIL ZZ <why>` (then read `/root/datascale/phase_<name>.out`, `build_pilot.log`, `ledger_pilot.jsonl.fail`).
Expected wall: build about 7 min, encode about 10 min (2.76 s/clip earlier), join a few min. If `pilot.log` has no new line for 30 min and `ps` shows no ds_ process, relaunch:
    pssh 'cd /root/datascale; nohup setsid bash /root/datascale/ds_run.sh < /dev/null > /root/datascale/pilot.stdout 2>&1 &'
Then:
    pssh 'cat /root/datascale/PILOT_REPORT.json' > F:/TanitAD-scratch/datascale/PILOT_REPORT.json
    python F:/TanitAD-scratch/datascale/ds_cost.py F:/TanitAD-scratch/datascale/PILOT_REPORT.json > F:/TanitAD-scratch/datascale/COST_DATASCALE.md
Check the cost table, the label-probe result (`labelprobe.json`: can `s2_geom_emit_v7.emit_one` run with no Alpamayo record?) and the join (`join_pilot.jsonl`, `phase_join_pilot.out`) and add them to the hand-back.

## Temp to delete after the report is banked (literal paths, log each in LOOP_STATE)
Pod: `/root/datascale/stage/` (mp4 + members, ~2-3 GB), `/root/datascale/v2ep_pilot/` and `v2ep_ctl/` (~7 GB; re-creatable by the same command, keep only if the PI wants the pilot corpus),
`/root/datascale/ctl_ref/` (3 copies of cache rows, source `/root/data/cache_pod` still exists), `/root/datascale/hfc/`. KEEP: `cache_pilot/` (the pilot cache, ~4 GB), ledgers, `PILOT_REPORT.json`, `meta/`, `selection/`.
Dev box: nothing large in `F:/TanitAD-scratch/datascale/`.

## Open items
* Full encode NOT approved. PI ruling needed: admit NVIDIA `val` clips (our own corpus already contains 1,213 train + 27 eval val-split clips) to lift the 133,929 ceiling toward 306k.
* VLM-CoT labels cannot be built for new clips (only 4,729 corpus clips + 1,364 pool clips with NVIDIA ood_reasoning event text). Needs a VLM pass = compute + PI decision.
* `navigation/navigation.parquet` (16.3 M rows, nav_text per clip/timestamp) exists in the NVIDIA repo; its coverage of pool clips is not measured: flag to the Master Mind.
