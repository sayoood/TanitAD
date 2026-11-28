# IMPL_CHOICES_DECN: every choice the card leaves open, fixed BEFORE any DECN held-out row is read (agent, dev box, 2026-11-09)
Card `../../PREREG_V8F_DECN.md`, frozen prefix sha256 `cfa5bb1fdda36a77efd6bf203232dbf5cf0d15f08a2189ad908e2436509619e5` (not edited). Tier **T0**: a readout diagnostic, no world-model training, no interval.
What had been read before this file: shapes and file lists; the DECINFO result files (already banked); one exploratory comparison of the cache field `v` against the native-track speed on the 982 TUNE windows (targets only, no latent, no R2): 15,712 of 15,712 values equal, mean absolute difference 0.0 (re-run as control 1 below). No DECN latent and no DECN R2 had been read.

## 1. Objects
* Held-out set: DECINFO's, unchanged, built by `decinfo_lib.load_tune` / `build_split` (17 episodes, 238 windows, 16 frames). Fit TUNE side: 53 episodes, 744 windows. No SCORE window is opened (the loader asserts it).
* Fit sets: **N53** = the 744 TUNE fit windows. **N194** = N53 + the 141 clips of `wmact/train_stream/clips141.json` (their order). **N1053** = N194 + 859 clips of `clips_extra.json`.
* Train windows per clip: the 17 starts of `collect_latents_train.window_starts` (the trainer's `D.train_windows`). Only windows whose 16 frames are all real (`frame_ok` all true, the rule the 1,976 eval windows obey) are kept: **13 of 17 per clip** (the 141-clip run shows 13 for every clip). So N194 has 744 + 1,833 = 2,577 windows and N1053 up to 744 + 13,000 = 13,744 windows; counts of samples and episodes are printed per row.
* Fit episode ids of train clips are the clip ids; the early-stopping part of the MLP is `decinfo_lib.es_ids_of` applied to ALL fit episode ids (sha256 `plantest3-mlp-es|<id>` first byte below 51, about 20 %), so N53's early-stopping episodes are those of DECINFO.

## 2. Clip selection (`select_extra.py`, output `clips_extra.json` + `.sha256`, written before any pull)
* Population: the manifest `wmact/train_stream/box2_meta/manifest.jsonl` (4,572 clips); drop the 60 clips `--exclude 540:600` of the sha12 order (the 4,512 the trainer used, as `select_train_clips.py`), drop the 141 of `clips141.json`, drop every eval clip (the 141 file names in `sfresh_inputs/cache/eval/clips` plus the clip ids of `join_eval141.npz`); the script asserts the overlap with the train manifest is empty.
* Rank: `sha256(clip_id + "decn-v1")` as a hex string, ascending (plain concatenation, no separator); the first 859. md5, n_frames, bytes copied from the manifest.
* Stream order: the 141 first (N194 survives a stopped run), then the 859 in rank order. N1053 means all 1,000 clips; a stopped run reports its actual episode count and its outcome is UNDETERMINED.

## 3. Targets
* Yaw rate at frame k = the clip's `ch1` (yaw-mode omega from the native track) at the window's frame, exactly the `act[:,:,1]` the eval windows use (`ClipWins.item`, `V8FMiniWindows.item`).
* Speed: eval windows keep DECINFO's source (native track column 3 at raw frame 2 x (t - 15 + k)). **Train clips use the clip's logged speed `v` of the cache file** at cache frame (t - 15 + k). Control 1 ties the two: on the 982 TUNE windows, the eval clips' own cache `v` against the eval target (R2 at or above 0.99, mean absolute difference at or below 0.10 m/s); a failure stops stage (a) before any pull with a marker and the read is INVALID. Also logged, deciding nothing: per train clip, `v` against the native-track speed at the same frames (maximum absolute difference).

## 4. Inputs and storage
* R0 = `wm.z1` (128-D) of M2b s0 and s1; R2 = pooled adapter token (256-D) of M2b s0 and s1; R1 = `wm.z1` of every ladder checkpoint found by `decinfo_lib.ladder_models()` at stage (a) start (tags saved to `run1/ladder_tags.json`; a tag outside `FROZEN_LADDER` is marked "added after the freeze"). All through `decinfo_lib.load_any` and `vector_inputs`, so z1 equals DECINFO's (the TUNE z1 is asserted against `wmact/latents.pt` as DECINFO did, max abs difference below 1e-4).
* Scratch: `F:/TanitAD-scratch/decn_scratch/run1/`: `stream/cache/train/clips/_current.pt` (one clip, overwritten each time; deleted by the stage after use), `lat/<clip>.npz` (float32 inputs and targets of that clip's kept windows, written atomically; present means done: resumable), `tune_inputs.npz`, `control1.json`, `ladder_tags.json`, `pytest` basetemp. Nothing else is kept from a clip.
* Pull: read-only sftp `get` of one file (the `collect_latents_train.pull` pattern), md5 against the manifest (control 6; mismatch after 3 attempts stops everything with a marker, no stage b). A pull error (network) is retried 3 times with 20 s pauses per clip; after that stage (a) stops with `STREAM_STOPPED.txt` and stage (b) runs on the clips that finished. No command is run on box 2 and nothing is written there.

## 5. Readers (as DECINFO, vector rungs)
* **Ridge**: lambda = 1e-3 times the sample count, intercept unpenalised, features not standardised, targets divided by the fit standard deviation, float64: the same closed form as `plantest3_lib.ridge_fit`. For memory at N1053 it is accumulated in chunks of 20,000 rows (`ridge_pred_chunked`); a test requires agreement with `ridge_fit` within 1e-8 of the predictions' scale.
* **MLP**: `diag_mlp.train` unchanged (2 x 256, GELU, dropout 0.2, AdamW 1e-3, wd 1e-2, batch 512, patience 12, seed 0, early stopping on the 20 % fit-episode part, held-out read once). **Epoch cap: 120 at N53 (DECINFO's value, so N53 reproduces) and 60 at N194 and N1053** (steps per epoch grow 5x to 18x; the cap is stated in every row; `epochs` and `hit_cap` stored).
* One frame (m = 1: 16 samples per window) and four frames (m = 4: frames t-3..t concatenated, 13 samples per window). One model per target.
* The deciding value of an (input, N, target) cell = the maximum over the four readers {ridge, MLP} x {m = 1, m = 4} (optimistic, stated).

## 6. Controls (runtime in the units, pytest `test_decn_controls.py`)
1. **Target tie**: section 3. Stored in `control1.json` and in `RESULT_DECN.md`.
2. **Reproduction**: at N53, R0 ridge m = 1, speed and yaw, both M2b seeds, against `decinfo/runs/R0_M2b_s*.json` (`ridge`, m = 1) within 0.01. Stored in the N53 R0 unit files. A pytest repeats it on the stored `wmact/latents.pt` z1; mutation: targets shifted by one frame leave the tolerance.
3. **Shuffled targets**: whole target tracks permuted among all fit windows with `derange_tracks` (no window keeps a track of its own episode, asserted), held-out true; one run per N, input, reader, m and target; INVALID if any reads above 0.05. Mutation: no permutation reads above the bar.
4. **Planted signal**: `planted_check` form (target = X w / std + 0.3 N(0,1), w seed 0) with the chunked ridge on each unit's real fit and held-out inputs, per m; at or above 0.80 and zeroed at or below 0.02. The "at N1053" bar is read in the N1053 units; N53 and N194 units carry it too. Test on synthetic inputs of N1053's shape (about 220,000 x 128), mutations as DECINFO.
5. **Split**: `split_guard` on the TUNE ids; the train clip ids are disjoint from the held-out episodes and from every eval clip (asserted before use); no SCORE window; the held-out block (window positions, speed and yaw targets) has one sha256 for N53, N194 and N1053 (stored per unit, report asserts equality). Mutation: a held-out id in the fit list makes the guard raise, and a changed held-out target changes the digest.
6. **Clip integrity**: md5 per pulled clip against the manifest; test: a corrupted byte is detected and a mismatch raises.

## 7. Outcome rules (mechanical, per target; `report_decn.py`)
* Cell value V(input, seed, N) = deciding value of section 5. Deciding inputs: R0 and R2 at M2b s0 and s1. Bars: speed 0.70, yaw rate 0.50.
* **N-LIMITED** iff some input X in {R0, R2} has V(X, s, N1053) at or above the bar for BOTH s0 and s1.
* **RISES** iff not N-LIMITED and some input X has rise(X, s) = V(X, s, N1053) - V(X, s, N53) at or above 0.10 for BOTH s0 and s1.
* **FLAT** otherwise (for every input at least one seed has a rise below 0.10). The three outcomes partition the cases. The rise is also printed per reader (same reader at N1053 and N53) next to the cell-value rise; if the two forms give different outcomes the report says so, and the cell-value form decides (it is the card's "best over the readers per input, N and target").
* All outcomes need controls 1 to 6 valid and the N1053 cells complete (1,000 clips); otherwise the outcome line reads UNDETERMINED with the reason.

## 8. Compute
CPU only (`CUDA_VISIBLE_DEVICES=-1`, `NVIDIA_VISIBLE_DEVICES=-1`), torch threads at most 4, one process; the GPU lock file is not touched; `wait_ram` (6 GB free) before every clip and every unit. Unit order in stage (b): R0 and R2 of M2b s0, then s1 (N53, N194, N1053 each), then R1 of each ladder tag. Units are skipped if their file exists. Dev-box drive letters: scratch on F: (1 KB allocation units).
