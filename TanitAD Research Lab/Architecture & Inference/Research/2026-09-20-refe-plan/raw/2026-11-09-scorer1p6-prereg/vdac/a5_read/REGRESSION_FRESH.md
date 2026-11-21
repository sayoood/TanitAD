# REGRESSION_FRESH (UTC 2026-11-13; tip e262e2cc; no HB/HC file opened or created)

## Real-data regression identity (fresh run)

**Result: reader v2 re-run on the real S1a inputs: 213 leaf values compared, 0 mismatches vs the frozen v1 `s1a_read.json` (PREREG 7.4 bar met). Verdict INCOMPLETE on both. Tables markdown byte-identical to the banked v1 and banked v2 tables (sha256 4a8cc690...).**

### Inputs (all sha256-verified by the reader against the manifest's FIXED values, and by me independently)
| key | path (under F:/tanitad_scratch/a5read/) | sha256 | source |
|---|---|---|---|
| export | pulled/ha/export/h48_proposals_subscores.npz | 14560686490a2edbb0a9611d69f1285f84b396bad1c28138cf192964b2809345 | HF vdac/export (LFS sha256 equal) |
| export_json | pulled/ha/export/h48_proposals_subscores.npz.json | 0f3fbeed3c8a21d08d574e2c68c94ac38ea55e0e42ea5fb8187015b2b7678802 | HF (not LFS; manifest value) |
| head_s0_a | pulled/ha/ha-s0/head_pred_update10000.npz | 6b9ac2bc1bc5989e7904e1c1b1d12eb6e4c2356fe7ad5658cc6fecc5196c03eb | HF vdac/ha-s0 (LFS equal) |
| head_s0_b | pulled/ha/ha-s0/head_pred_h48x.npz | 64d2f71be45154bd36d71d3955a9151023d6f10365f9bac5376c6e3ce02aac3b | HF (LFS equal) |
| head_shuf_a | pulled/ha/ha-shuf/head_pred_update2000.npz | 5cb8c49255de67b8073bfc45cb75f2356c91bdc381cb222cfe608d9d79fb5beb | HF (LFS equal) |
| head_shuf_b | pulled/ha/ha-shuf/head_pred_h48x.npz | 4ee7816cc7a19cacd8a13101f51e6b7ae151dfcc90f2000edcf4b2c627161ba3 | HF (LFS equal) |
| head_init0_a | pulled/ha/stage0/head_pred_step0.npz | 2ebe4b540b163e2638532a71d0a006a35e7ce9e44ee0c0cb1bce8fd10903f94a | HF evidence/2026-11-12-vdac-stage0/stage0.tar (member stage0/head_pred_step0.npz) |
| head_init0_b | pulled/ha/stage0/fwd/head_init0_b.npz | 8b556eb23c842732ec91837781f7181c1f90406f3a1aa2316e2e9493bb542ed4 | same tar (stage0/fwd/head_init0_b.npz) |
| h48_logs | code/<PK>/raw/2026-11-09-scorer1p6-prereg/vdac/s1/h48/h48_logs.txt | 2ae13935a3ead7f30b015f33b0bebc7a35e70eb2a4194161f8cbb66ac89b8da8 | git export e262e2cc |
| in_sample | code/<PK>/raw/2026-11-12-vdac-stage0-read/in_sample_iou.json | 1547bd30ea6fba32f2a499da186b41766c1daa04ce23a542e0df804b7e1d6f9c | repo copy (MM decision) |
All 9 HA rows of `manifest_a5_hb.json` (incl. `hb_init_b` = head_init0_b) read OK against their fixed sha256 (checked with hashlib). The stage0.tar (HF, 79,431,680 B) sha256 was 423c055d4e40c7761eb0965a3bd29adac45c795857c1f057901aca8f74faff4c; I did not cross-check it against an LFS value (the two heads inside match the manifest's fixed values, which is the check that matters).
Code (byte-exact, `git -c core.autocrlf=false archive e262e2cc`): eval/s1_read_v2.py 796aea21...09f4 (= PREREG 7.1); frozen v1 s1a_read/s1_read.py 58aa3a77...718e (= PREREG 7.1); the 5 cards verified by the reader against frozen shas. Two banked files (`s1f_read_v2.json`, `_tables.md`) were added to the code export from `git show` for comparison.

### Commands (Git-bash, CPU only; interpreter C:/Users/Admin/venvs/tanitad/Scripts/python.exe, Python 3.13.5)
```
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8
export S1_READ_V1="F:/tanitad_scratch/a5read/code/<PK>/raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/s1_read.py"
python <code>/eval/s1_read_v2.py --manifest evidence/fresh/inputs_manifest_s1a_fresh.json --out evidence/fresh/s1a_run
python <code>/raw/2026-11-12-vdac-a5-build1/regression_identity.py evidence/fresh/s1a_run/s1_read_v2.json <code>/raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/s1a_read.json
```
`inputs_manifest_s1a_fresh.json` (sha256 d63ad2b6...8abb) = the banked `inputs_manifest_v2_run.json` with only paths re-pointed (cards_dir, files); every fixed sha256 unchanged. Wall: reader `wall_s` 17.6 s; the run took ~2 min elapsed on the shared box (other sessions' CPU jobs); peak working set of the process ~1.2 GB.

### Comparison (banked `regression_identity.py`)
* fresh v2 vs v1 `s1a_read.json`: `compared leaf values: 213`, `mismatches: 0`, audit equal except paths (all 13 keys True), verdict v2 INCOMPLETE | v1 INCOMPLETE. v2-only keys `frozen_v1_reader_sha256`, `paired`, `per_arm` (as banked). `evidence/fresh/regression_identity_fresh.txt`.
* fresh v2 vs the banked v2 json (`s1a_regression/s1_read_v2.json`): 0 mismatches, no v2-only keys (identical keyset).
* `reader_sha256` in the fresh json = 796aea21... ; `manifest_sha256` d63ad2b6... (differs from banked, expected: paths).

### Extra (not required): Stage 0 S1-F read re-run on real HA inputs incl. both init heads
Manifest `manifest_s1f_fresh.json` (sha256 ba95842b...c998) = `manifest_s1f_v2.json` re-pointed, with `in_sample` = the repo copy (sha 1547bd30...). Result: verdict **S1-WEAK** (in-sample IoU 0.7864 < 0.85). Vs banked `s1f_read_v2.json`: 315 leaf values, 1 "mismatch" = `/stage0_vs_reader` present in the banked file and absent in the fresh reader output; that key is not a reader output (a Stage 0 tool adds it after the read; UNVERIFIED which tool, not investigated). All reader-produced values equal, and the tables markdown is byte-identical (`cmp`). So the repo `in_sample_iou.json` copy reproduces the banked S1-F read exactly. `evidence/fresh/regression_identity_s1f_vs_banked.txt`.

### in_sample_iou.json: the two hashes
Stage 0 manifest named dev-box copy sha 5ab0c72c... (gone, unreadable); repo copy 1547bd30.... Value equality: the read with the repo copy reproduces the banked S1-F read (in-sample IoU 0.7864227917638612 in the banked `s1f_read_v2.json` per the freeze note; the fresh run's verdict text reads 0.7864). Byte-level reason for the hash difference: UNVERIFIED (the gone copy cannot be diffed).

### HA-side precheck of manifest_a5_hb.json
The frozen reader has NO dry/validate mode (argparse: --manifest, --out, --n-boot; `load_real` hashes every file then reads). Running it on `manifest_a5_hb.json` would fail on the PENDING HB rows. What was done instead: hashlib verification of all 9 HA-side rows (above), the 5 cards and h48_logs by the reader's own `load_real` in the two runs. Not done: PREREG 7.5 mask-sign precheck (needs HB files).

### Limits
* HB/HC files: none opened, none created. Stage 0 heads came from HF, not the pod.
* Interpreter/numpy for the banked S1a run unknown (UNVERIFIED); results are exact-equal anyway.
