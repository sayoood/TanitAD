# LEVER3 MM handoff (2026-11-28 ~14:50Z)

Card `PREREG_V8F_LEVER3.md` FROZEN sha256 4c019b01... (PI "Yes, build and run", 2026-11-28).

## Running (detached, WMI CreateFlags 0x08000200)
* `build/run_tenk.cmd` -> `l3_build.py tenk --prefetch 2`, tars staged on `E:/TanitAD-scratch/lever3/dl`, clips to `C:/Users/Admin/lever3/cache_union/train/clips`. Log `build/tenk.log`; per shard `build/state_tenk/<shard>.done.json`; end: `state_tenk/TENK.done` + `build/tenk.rc`. Pace about 10 MB/s in total, so it ends around 20:30Z. Resumable per shard: relaunch the same .cmd.
* `build/run_corpus.cmd` -> `l3_build.py corpus` from `F:/TanitAD-scratch/speedread/cache/train/clips` (read-only, not ours). End: `state_corpus/CORPUS.done` + `build/corpus.rc`. Resumable per clip.
* A session waiter (`build/wait_build.sh`) exits when both rc files exist.

## Done and verified
* SHARDS.json: 42 tars, 208,760,043,520 B, hub LFS sha256 each. The tenk job checks each tar's sha and deletes it after its repack (logged in `DELETIONS.log`).
* Control b1, per clip, in `repack_bytes` (the self-test catches a planted byte, a dtype change and a missing key).
* p10: `E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/p10/p10_prior.pt` (14a062a9). Equal bitwise to the prior inside UE_s0/A, UE_s1/A and UE_s2/A (`p10id/*.json`, b5).
* Init checkpoints `C:/Users/Admin/lever3/init/UE_s{0,1}/A/ckpt_last.pt`, hub sha verified (2fd4b747, f13df8ce).
* Labels inputs: `labels/in/corpus/` (lead d4336a44, speed bab41adf, v9 83e30bd3 + manifest.json 7faeec88; first three from box 2 read-only) and `labels/in/tenk/labels/` (hub sha verified).
* Lean merge tool `build/dscale10k_merge_lean.py` (d13538e1, from 89d40399 by `mk_merge_lean.py`; it changes memory use only). The union v9 npz it wrote is md5 9881b170 / sha256 c630dab0: control b2 for v9 PASSES (byte-identical to the pod's). The first run stopped because the corpus v9 needs its `.manifest.json` beside it (now copied).
* Tree: `C:/Users/Admin/lever3/tree_cdb` = git archive 97fe0a56 stack+taniteval (RH's tree_cdb).

## Update ~15:25Z
* **The card of record is v2:** `PREREG_V8F_LEVER3_V2.md`, sha256 707c5225. v1's b2 FAILED for native_tracks_union.pt: its sha differs from the pod's at equal size, because torch.save writes a machine-dependent serialization_id (`build/TRACKS_RESAVE_PROBE2.json`). v2 checks that file by content (`build/tracks_content_check.py`; PASS, `build/B2_TRACKS_CONTENT.json`).
* **Labels done in `C:/Users/Admin/lever3/merged`:** v9 9881b170, lead 73693175, speed fd808a26, all equal to the pod's; tracks content-equal.
* **The corpus repack is DONE** (4,512 clips, `state_corpus/CORPUS.done`). The 10k part is at 7 of 42 shards at about 7.3 min per shard, with an ETA near 19:20Z.

## Next, in order
1. (done) Labels re-merged; b2 read as above.
2. After both repacks: `python -I build/l3_manifest.py` writes `cache_union/train/manifest.jsonl` + meta.json and `build/MANIFEST_RECORD.json`. It must PASS, with 14,512 rows and 0 n_frames mismatches.
3. Kit agent (Sonnet, running): `lever3/kit/` holds b3, the bank kit, the gate kit, arm_run and l3_chain. It reports back with REPORT.md / HANDOFF.md.
4. Then: banks s0 and s1 (CPU), mints, then the D3K_s0, D9K_s0, D3K_s1 and D9K_s1 arms under the GPU lock. Next, an amendment naming the eval cache sha before any rollout, then the reads.
5. Delete at the end, by literal path: `labels/in/corpus/v9_labels_train.npz` (a copy of strat1's), `E:/TanitAD-scratch/lever3/dl` (empty after the build), and `PREREG_V8F_LEVER3_DRAFT.md`.
