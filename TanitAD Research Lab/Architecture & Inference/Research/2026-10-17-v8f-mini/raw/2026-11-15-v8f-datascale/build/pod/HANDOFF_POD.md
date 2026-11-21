# HANDOFF DSCALE10K pod launch kit (pod agent, 2026-11-16 ~22:10Z)

## State
* Steps 1-4 done (details in the hand-back). Step 5 launched: `/root/dscale10k/pilot.sh` (detached, log `/root/dscale10k/pilot.log`). It waits for `ZZ STRAT1 S10 DONE|FAIL ZZ` in `/root/strat1/logs/s10.log` and GPU memory.used < 3500 MiB (STRAT1 was on arm 2 of the S10 chain at 21:06Z), then runs `probe/pilot_run.py`: reach probe (U) + G-HYG records (B, U), trainer `--selfcheck-only` (B, U), 2 x 600-step pilots (`pilot/pB`, `pilot/pU`), 200-clip DAgger bank timing (`pilot/bank200`), writes `/root/dscale10k/PILOT_REPORT.json`, ends with `ZZ DSCALE10K PILOT DONE ZZ` or `... FAIL ZZ <why>`. It does NOT start block 1.
* Check: `tail -3 /root/dscale10k/pilot.log; python -c "import json;d=json.load(open('/root/dscale10k/PILOT_REPORT.json'));print(d['status'],d.get('projection_block1'))"`.

## Pod layout (`/root/dscale10k/`)
`tree/` code of record cdbff971 (3,400 blobs verified), `study/` build tools (merge + check PATCHED on the pod, see PATCH_*.diff), `in/` HF labels (sha256 verified) + `manifest_10k.jsonl`, `merged/` union + R release + lead/speed/tracks/join/manifest/merge_record.json, `argv/` B_s0/U_s0 stage A and B argv (pod paths), `dry/` dry records, `probe/` probe + pilot scripts, `realchk.json` real-data check results. Union view: `/root/data/cache_union_dscale/train/` (14,512 symlinks).

## Next step (MM)
1. Read PILOT_REPORT.json: the 10 h stop rule on `projection_block1.block1_wall_h`; U/B s/step ratio vs 1.5 (pool4x10 copy trigger).
2. Open items before `mint` (not done here): G-CLOCK residuals, G-EVAL, G-CKPT, the stage-B `dagger` reach term, C-INIT, C-CACHE, C-SESS, the evidence JSON assembly. G-SUITE: the tree manifest sha of `tree/` equals `/root/strat1/base` (7ccc204d...), whose full-suite junit is `/root/strat1/junit/base/` (14,851 tests, 431 failed, 206 errors, 1,691 skipped = the baseline itself, MEASURED here by parsing the xml).
3. Delete when done (literal paths): `/root/dscale10k/merged/v9_labels_R_s0.npz` (2.0 GB, re-creatable), the union view, `pilot/` checkpoints.
