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
