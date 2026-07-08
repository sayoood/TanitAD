# Cosmos-Drive-Dreams: real repo layout (pod probe, 2026-07-08 — loop iter 8)

Probed `nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams` before the first real-clip
verification. **The loader's video-discovery assumption does not match the repo:**

- Root = RDS-HQ conditioning/annotation folders (`3d_*`, `all_object_info`, `captions`,
  `vehicle_pose` (per-clip 0.6 MB `.tar`s, 5,843 clips), `pose`, `ftheta_intrinsic`,
  `pinhole_intrinsic`, `lidar_raw`, `car_mask_coarse`) — **no per-clip video files**.
- The rendered synthetic videos live in `cosmos_synthetic/single_view/` as
  **`generation.tar.gz.part-000..N` shards of ~43 GB each** (+ `caption.tar.gz`). Single-clip
  retrieval is not possible; ingestion = download shard part-000 (fits the pod volume), stream
  gunzip/tar from the first part, extract a bounded clip set, pair with the per-clip
  `vehicle_pose/*.tar` by clip id.

**Consequences:**
1. `discover_clips()` stays valid for LOCAL extracted layouts — the fix is an extraction step, not
   a loader rewrite: `scripts/cosmos_verify.py` to grow a `--from-shard` mode (stream-extract N
   clips from part-000, unpack matching vehicle_pose tars, then discover+verify as designed).
2. `verify_real_clip` still pending real bytes (honest status unchanged); scheduled for the next
   loop iteration on the pod.
3. Data card: note that cosmos ingestion cost is shard-granular (~43 GB per slice of the corpus),
   pod-side only; local machine gets derived episodes via the epcache, never shards.

Handoff: DataEng agent (Tuesday) owns the data card update; the loop executes the shard extraction.

## UPDATE (same day, loop iter 9): verify_real_clip PASSED on real bytes

Shard part-000 stream-extraction works end-to-end: 60 mp4s (weather variants in filenames:
Rainy/Night/Foggy/Morning/Sunny), 60/60 pose tars paired after the base-id fix (videos carry
`_<variant>_<Weather>` suffixes; pose tars do not). Real-clip stats: speed 4.8 m/s, |steer| ~0,
|accel| <= 1.08, **A8 = 0.109** (consequence-dominant, ~2x comma2k19). Saved:
`/workspace/data/cosmos/verify_real_clip.json`.
**Open question for the data card (P8):** T = 39 frames per episode — shorter than nominal
20 s @ 10 Hz; determine source fps / pose-stream rate before cosmos enters training windows
(window 8 + horizon 16 needs T > 24, so usable, but the temporal semantics must be confirmed).

## RESOLVED (loop iteration, 2026-07-08 evening): temporal semantics + a chunk-pairing bug

Measured on the extracted bytes + cross-checked against the HF card and toolkit docs:

- **Videos are 121-frame chunks at the 30 Hz RDS-HQ label rate** (~4.03 s each). The mp4
  container reports 24 fps — that is a muxing artifact (ffmpeg default), NOT the frame spacing.
  The loader's `SRC_FPS=30`, stride-3, dt=0.1 s assumptions are **correct**.
- **Clips are 10 s with ~300 poses at 30 Hz** (297–300 pose files per clip measured; matches
  "5,843 10-second clips" on the dataset card and RDS-HQ's documented 30-FPS streams).
- **T=39 fully explained:** 121 frames → min(121, 300) → stride 3 → 41 → −2 (D-015 stacking) = 39. ✔
- **BUG found and fixed (real action corruption):** chunk i of a clip renders label frames
  [i·121, i·121+121), and **~half our extracted videos are chunk 1** (`_1_Rainy` ×9, `_1_Snowy`
  ×5, …) — the loader paired every video with poses `[:121]`, giving chunk-1 videos chunk-0's
  actions. Fix in `stack/tanitad/data/cosmos_drive.py`: `_chunk_of()` parsing, pose offset
  `chunk*CHUNK_FRAMES` with an insufficient-poses guard (skip, not crash), chunk in
  `_episode_id` (no more episode-id collisions between chunks of the same clip+weather).
  Tests: accelerating-trajectory fixture proves chunk 1 reads poses [121:242] (speed
  discriminates the segment); 141 tests green. Found BEFORE cosmos entered any training mix.
- **Verdict: cosmos is cleared for the D-010 mix** (window 8 + horizon 16 < T=39), pending the
  usual epcache build. Speeds/A8 previously reported by `verify_real_clip` on chunk-0 clips are
  valid; chunk-1 numbers from before the fix are void.
