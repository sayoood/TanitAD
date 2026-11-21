# BUILD_CKPT120_FIX3_POD: tree new11, G-SUITE kit (running), chain and queue wrapper staged

Built 2026-11-17 for commit aedf8716989494464277e50ec4decbbf1a2e65dc (branch `agent/v8f-ckpt120-20261117`; fix3 = `stack/scripts/v8f_mini_train.py` modified, `stack/tests/test_v8f_ckpt120_dagger_nav_s.py` new). Nothing committed or pushed, no GPU touched, `/root/gpu_queue.lock`, DSCALE block 2, the sidecars and `/root/dscale10k/` were not touched. Evidence class: MEASURED (commands run on the pod this session).

## 1. Tree `/root/strat1/new11`
`cp -a new10` plus the two fix3 files exported with `git -C <mirror> -c core.autocrlf=false show aedf8716:<path>` (0 CR bytes). `new10` untouched (H1 record).
| file | sha256 | git blob |
|---|---|---|
| stack/scripts/v8f_mini_train.py | 2c25f2310a67f29aa0407e8c52e8b1ce723b6ec096519c377a25feac9ee73561 | aabd5452790b0264e8b5b019d1234f0d0b9866d4 |
| stack/tests/test_v8f_ckpt120_dagger_nav_s.py | a6585601ace957affa532eb41ec2b27fd93f62302185966ea7ab9b2166677099 | ec0411f169dedb4a70e6d45f2287b317a766c0ef |
Blob of each equals `rev-parse aedf8716:<path>`; the pod file equals the local export by sha256 and `git hash-object`. List `/root/strat1/ckpt120_new11.sha256`: 20 entries (the 19 with the trainer's new hash, plus the new test); `sha256sum -c` in new11 = 20 OK. The trainer file equals `files/` in this folder (cmp).

## 2. G-SUITE: `/root/strat1/ckpt120_gsuite_fix3.sh` (local `pod/ckpt120_gsuite_fix3.sh`), RUNNING
Started 15:56Z, CPU only (`CUDA_VISIBLE_DEVICES` empty), `nohup setsid`. Changes against `ckpt120_gsuite.sh` (diffed, all textual): commit aedf8716 (:14), `TREE=$R/new11` (:10), list `ckpt120_new11.sha256` (:21), expected diff 20 entries (:24), junit `junit/new11`, scratch `tmp/new11suite`, log `logs/ckpt120_gsuite_fix3.log`, compare log `logs/suite_compare_ckpt120_fix3.log`, result `gsuite_result_ckpt120_fix3.json` (:56), sentinel `ZZ CKPT120 GSUITE FIX3 DONE ZZ` / `ZZ CKPT120 GSUITE FIX3 FAIL ZZ <why>`. Unchanged: base junit `junit/new6`, base commit a4357e79, shards from `shards_like_base.py` (767 files; the new test file went to shard 3), two waves of 4 shards, 2 threads, start only under 30 GB total_rss, watchdog 52 GB.
Status at hand-back: wave 0 (shards 0-3) running, container total_rss 13 GB at start, 19.4 GB at last read (limit 70.9 GB). **UNVERIFIED until the sentinel.** Check: `tail -3 /root/strat1/logs/ckpt120_gsuite_fix3.log`.

## 3. Chain `/root/strat1/strat1_ckpt120_fix3.sh` (local `pod/strat1_ckpt120_fix3.sh`), `bash -n` OK locally and on the pod
Textual changes against `strat1_ckpt120.sh`: `COMMIT` aedf8716 (:15), `TREE=$R/new11` (:14), `SUITE=gsuite_result_ckpt120_fix3.json` (:16), the sentinel grep on `logs/ckpt120_gsuite_fix3.log` for the FIX3 DONE line (:19), list `ckpt120_new11.sha256` and 20-entry diff (:21-23), verify logs `*_fix3.log`.
* Skip of finished arms (already in the chain; kept and tightened, :36): an arm is skipped when `runs/<tag>/ARM_DONE` exists AND `runs/<tag>/train/ckpt_last.pt` is non-empty. On the pod `runs/H1_s0` and `runs/H1_s1` both hold `ARM_DONE` (`ZZ ARM ... DONE ZZ`) and a `train/ckpt_last.pt`, so both are skipped. The banked-checkpoint sha256 is NOT compared by the chain (UNVERIFIED here; the HF bank of those two arms was verified by the earlier session, not re-read by me).
* H2_s0 refused-gate move aside (:29, guarded: only if `gate/` has no token and the target is absent). I also performed it now on the pod: `runs/H2_s0/gate` is `runs/H2_s0/gate.refused_213250ce` (its `gate.log` holds the `ZZDVB-REFUSED` text with `nav_s`); `runs/H2_s0` now holds `argv.json`, that folder, nothing else. Mint runs fresh (it only skips when `gate/gate_token.json` exists).
* Every token is minted by `ckpt120_check.py mint` on aedf8716 (`--commit` in COMMON). H1 tokens at 213250ce stay in their folders and are not consulted for skipped arms.
* Dry render of the seven arms through the checker on new11 (`pod/ckpt120_argvs_dry_fix3.json`, no GPU): kind problems [], data-binding problems [], no missing CKPT120 flag, argv sha256 prefixes equal to the 213250ce render (H1_s0 8b9549b5acee, H2_s0 8482fd663cb0, H2_s1 03a84b19c656); the H2_s0 argv names `--nav-strat on`, its npz, sha256 and variant. The mint path appends `--selfcheck-only` to those tokens (`ckpt120_check.py:407`), and the trainer calls `dagger_probe` before the `if a.selfcheck_only:` exit (`v8f_mini_train.py:3962` before :4068), so the G-DVB dagger probe on `nav_s` is reachable on new11. This is a reachability read, not a real-data run: the first H2_s0 mint is the first real-data run of the fix (UNVERIFIED).

## 4. Queue wrapper `/root/strat1/queue_ckpt120_fix3.sh` (local `pod/`), prepared, NOT run
Waits (sleep 300, up to 12 h) for `ZZ CKPT120 GSUITE FIX3 DONE ZZ` in `logs/ckpt120_gsuite_fix3.log`, exits `ZZ CKPT120 FIX3 QUEUE FAIL ZZ <why>` on a FIX3 FAIL line, then `exec flock /root/gpu_queue.lock bash /root/strat1/strat1_ckpt120_fix3.sh` (queues behind DSCALE block 2).
**Start line for the MM:**
`nohup setsid bash /root/strat1/queue_ckpt120_fix3.sh < /dev/null > /root/strat1/logs/ckpt120_fix3.log 2>&1 &`
(Last line of that log: `ZZ STRAT1 CKPT120 DONE ZZ`, or a FAIL line. A STOP file at `/root/strat1/chains/STOP` ends the chain between arms.) Arms left to run: H2_s0, H2_s1, H1_rep, R1_s0, R2_s0.

## 5. Open items
1. G-SUITE FIX3 result pending (section 2); if it FAILs the queue wrapper reports it and does not start the chain.
2. The chain skip does not re-hash the H1 checkpoints (section 3).
3. K-REACH-e INCONCLUSIVE handling is as in the earlier chain.
4. Deletions (own temp, literal paths): pod `/root/strat1/tmp/fix3_v8f_mini_train.py` (373,823 B) and `/root/strat1/tmp/fix3_test_nav_s.py` (8,123 B), both copies of the commit blobs; local `F:/TanitAD-scratch/strat1/ckpt120fix3/exp/` (381,946 B git export), and my local copies of the three 213250ce scripts in `pod/` (originals remain in `ckpt120build/pod/` and on the pod). All accepted, none refused. Kept: `pod/` fix3 scripts, `tmp/render11.py` on the pod.

## 6. Manifest
| artifact | where |
|---|---|
| tree new11, list | pod `/root/strat1/new11`, `/root/strat1/ckpt120_new11.sha256` |
| G-SUITE script / log / result | pod `/root/strat1/ckpt120_gsuite_fix3.sh`, `logs/ckpt120_gsuite_fix3.log`, `gsuite_result_ckpt120_fix3.json` (pending); script also local `F:/TanitAD-scratch/strat1/ckpt120fix3/pod/` |
| chain, queue wrapper | pod `/root/strat1/strat1_ckpt120_fix3.sh`, `/root/strat1/queue_ckpt120_fix3.sh`; also local `.../ckpt120fix3/pod/` |
| dry argvs | local `.../ckpt120fix3/pod/ckpt120_argvs_dry_fix3.json` (also pod `tmp/`) |
| refused gate | pod `/root/strat1/runs/H2_s0/gate.refused_213250ce` (one place) |
