# HANDOFF_DATASCALE10K (2026-11-15, launched ~19:01 pod time)

## State
* PI go: "Yes, encode 10k" (train split only). N, cache format, selection, venue unchanged. Selection = first 10,000 clips of the frozen scope order
  (`/root/datascale/selection/clips_10k.txt`, sha256 prefix 31aca42af5eab6c1; pilot 200 = prefix, asserted).
* Full run launched detached on the LoRA pod: `bash /root/datascale/ds_run10k.sh` (driver `ds_stream.py`; wrappers `ds_stage_w.py ds_build_w.py ds_encode_w.py`
  call ds_stage / ds_build / cache_pod UNEDITED). Smoke (20 clips, positions 200-219, one HF upload + LFS sha256 check): PASSED 18:58, 170 s wall, 0 failures.
* 42 shards (windows): `s000_pilot200` (banked, 200 clips from cache_pilot), `s001_smoke20` (banked, 20), `s002..s041` (250 each, last 30 clips; 9,780 new clips).
* HF repo (private dataset): `Sayood/tanitad-datascale-10k`. Per shard: `cache/<shard>.cache.tar` (train/clips/*.pt + manifest.jsonl + pod_provenance.jsonl + meta.json),
  `members/<shard>.members.tar` (egomotion + obstacle parquet per clip, ledgers, `join.jsonl.gz` obstacle join). At the end `manifest/` (corpus manifest, order, scope, banked, fails).
* DEVIATION (needed): the obstacle join runs PER WINDOW before that window's v2ep is deleted (the join reads poses from the v2ep; after deletion it cannot run). Output is banked in each members tar.
* A shard is banked only when the hub's LFS sha256 == the local tar sha256; then local tar + clip files are deleted.

## Is it done / failed
    source C:/Users/Admin/lorapod/podenv.sh; pssh 'tail -3 /root/datascale/run10k.log'
* DONE: `ZZ DATASCALE-10K DONE ZZ banked=<n> failed=<m> expected=10000`. FAIL: `ZZ DATASCALE-10K FAIL ZZ <why>`.
* `PROGRESS` line every 10 min (staged/built/encoded/banked/failed/free_disk_gb/enc clips per h). Expected ~7.4-8 h wall (encode-bound, 1,354 clips/h MEASURED pilot).
* Per-clip failures: `/root/datascale/fails10k.jsonl` (a failed clip is skipped, not retried). Window detail: `/root/datascale/ledgers10k/<shard>/`.

## Resume after a kill (re-run the same command; markers in `state10k/`, ledgers resume, banked shards skipped)
    pssh 'cd /root/datascale; nohup setsid bash /root/datascale/ds_run10k.sh < /dev/null > /root/datascale/run10k.stdout 2>&1 & disown'
Kill only by PID of `ds_stream.py` and its `ds_stage_w/ds_build_w/ds_encode_w` children (never pgrep -f patterns that match your own ssh). Disk guard: staging pauses below 60 GB free (`PAUSE` lines).
If the log has no new line for 30 min and no ds_stream.py process exists, relaunch.

## For the Master Mind to land
* `/root/datascale/corpus_manifest_10k.jsonl` (clip, cache_sha256, shard, shard_sha256, members_sha256) -> repo (also uploaded to HF `manifest/`). Pull it with pscp when DONE.
* `/root/datascale/banked10k.jsonl`, `deletions10k.log`, `fails10k.jsonl`, `ledgers10k/`.

## Temp-delete list (all by literal path; the driver already deletes its own window temp and logs each in `/root/datascale/deletions10k.log`)
* Automatic per window: `w/<shard>/stage/r0` (mp4 after v2ep verified), v2ep files (after cache md5 verified), tars + whole `w/<shard>` (after banked).
* Kept on purpose: `/root/datascale/keep10k/<shard>/` (egomotion/obstacle members + join, ~0.9 MB/clip + join), `ledgers10k/`, `cache_pilot/` (pilot cache, banked in s000; deletable by the MM, ~4 GB).
* After DONE: `/root/datascale/cache_ctl`, `/root/datascale/stage` leftovers, `/root/datascale/hfc`, `/root/datascale/meta` (118 MB, keep until the MM lands) are the pilot's leftovers.
* Not touched: `/root/q5_416`, `/root/data`, `/root/identity_pod`, lp3_sampler.py, bank_ckpts.py, jupyter-lab.

## Open (unchanged from HANDOFF_DATASCALE.md)
VLM-CoT labels for new clips not built (needs PI decision); `navigation.parquet` coverage unmeasured; val-clip admission ruling pending.
