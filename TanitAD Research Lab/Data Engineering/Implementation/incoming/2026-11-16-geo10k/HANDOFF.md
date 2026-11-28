# GEO10K handoff (2026-11-16)

## State
* Geometry-side labels for the 10,000 clips of `Sayood/tanitad-datascale-10k` are BUILT on the pod under `/root/geo10k/out/` (tree = `git archive cdbff971 stack/` in `/root/geo10k/tree`; code in `/root/geo10k/code`, copies in the repo staging folder).
* Controls pass (see `CONTROL_RESULTS.md`): poses 20/20 bitwise, clock 19/20 bitwise (1 ulp), v9 ctlA 0 fields above 1e-9, wrong-egomotion arm RED, speed 20/20, context 4,659/4,659.
* Bank step: `chain5.sh` -> `code/g8_bank.py` uploads to HF `labels/` and verifies sha256. Final marker in `/root/geo10k/chain5.log`: `ZZ GEO10K DONE ZZ` or `ZZ GEO10K FAIL ZZ <why>`.
  Check: `source C:/Users/Admin/lorapod/podenv.sh; pssh 'tail -5 /root/geo10k/chain5.log'` (output lines `[bank] OK <file> <bytes> sha256 <prefix>`).

## Files (pod `/root/geo10k/out/` -> HF `labels/`)
v9_labels_new.npz (+ .manifest.json) -> v9_labels_datascale10k.npz; lead_new10k.npz; speed_sidecar_new10k.jsonl; native_tracks_new10k.pt; clock_new10k.jsonl; new10k_v2manifest.pt; clips_datascale10k.parquet (context); geo10k_clip_manifest.jsonl.gz; geo10k_files.json (file, sha256, bytes).

## Next step (Master Mind)
1. Register the md5 of `v9_labels_datascale10k.npz` (value in its manifest `npz_md5`) in `ACCEPTED_TRAIN_MD5` (`v8f_mini_data.py:67`), or merge the rows into a release and register that md5; the trainer refuses unknown md5s.
2. Point the 10k run at: `--v9-train` that file, `--lead-train` lead file, `--speed-train` speed sidecar, `--tracks` native tracks, `--cache-dir` the 10k cache with its `manifest.jsonl`.
3. Decide on the VLM goal bits (0 of 10,000 clips have them; a VLM pass needs a PI decision and re-downloaded frames).
4. Open: a known-value control for the lead pass; image-derived context fields (weather, surface, visibility, illumination) need frames re-download plus a GPU pass; v10 band labels for the 10k (builder on the arch-inf tip) not built.

## Pod temp to delete after landing (by literal path)
`/root/geo10k/stack_cdbff971.tar` (156 MB), `/root/geo10k/ref/` (about 0.3 GB, HF copies), `/root/geo10k/out/build_*` (small), `/root/geo10k/inputs/ts` (10,020 small parquets), `/root/geo10k/ego_flat` (symlinks only). Keep `/root/geo10k/out/*.npz|pt|jsonl` until the HF verify is read.
