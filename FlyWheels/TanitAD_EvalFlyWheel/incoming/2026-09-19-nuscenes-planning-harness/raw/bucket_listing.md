# nuScenes — official AWS Open Data bucket, LIST + HEAD ONLY (no object bytes fetched)

**MEASURED 2026-09-19** (W6/E5). `GET https://motional-nuscenes.s3.amazonaws.com/?list-type=2&max-keys=1000`
returned `KeyCount 90`, `IsTruncated false` (`x-amz-bucket-region: ap-northeast-1`), i.e. the whole bucket in one page.
Every size below is the listing's `<Size>`; five of them were re-read by an independent `HEAD`
(`Content-Length`) — `raw/bucket_head_crosscheck_2026-09-19.txt` — and agreed exactly.
⛔ A listing is not a download and reachability is not licence permission: the Terms of Use are the
operative instrument and a HUMAN accepts them first (`stack/tanitad/data/nuscenes.py`).

| object | bytes | GB | needed for |
|---|---:|---:|---|
| `public/v1.0/v1.0-trainval_meta.tgz` | 461,678,030 | 0.46 | metadata (13 JSON tables) — GT, sample sets, occupancy, intrinsics, v0 |
| `public/v1.0/v1.0-mini.tgz` | 4,168,148,189 | 4.17 | 10-scene mini (all sensors + sweeps) — smoke test + SAM3 pilot |
| `public/v1.0/nuScenes-map-expansion-v1.3.zip` | 398,535,531 | 0.40 | vector map layers — SAM3 paint reference; map-compliance metrics |
| `public/v1.0/nuScenes-map-expansion-v1.2.zip` | 17,136,555 | 0.02 | ⛔ REFUSED by the current devkit (map_api.py:99-101 needs >= v1.3) |
| `public/v1.0/can_bus.zip` | 780,974,697 | 0.78 | NOT NEEDED — the ego-status channel (label source only) |
| `…trainval{01..10}_keyframes.tgz` (10) | 44,902,690,772 | 44.90 | the `samples/` tree (CAM_FRONT + LIDAR_TOP keyframes) |
| `…trainval{01..10}_blobs_camera.tgz` (10) | 177,288,110,316 | 177.29 | camera SWEEPS (12 Hz) — only for a 10 Hz-history arm |

## Tiers (what a given job actually needs)

| tier | objects | bytes | GB |
|---|---|---:|---:|
| pilot (smoke + SAM3 pilot) | mini + map v1.3 | 4,566,683,720 | 4.57 |
| planning eval, full val | metadata + 10 keyframe parts | 45,364,368,802 | 45.36 |
| + SAM3 paint reference | same + map v1.3 | 45,762,904,333 | 45.76 |
| 10 Hz-history arm (instead of keyframes) | metadata + 10 camera-sweep parts | 177,749,788,346 | 177.75 |

⚠️ The keyframe parts are per-PART, not per-scene: which of the 10 parts hold the 150 val scenes is
NOT derivable from the listing or the metadata (no part id in `sample_data.filename`), so the
conservative requirement is all ten. Re-check after the first part lands: `tar -tzf` names its members.

Cross-check: the keyframe total reproduces the 2026-07-26 figure in `stack/tanitad/data/nuscenes.py`
(`44,902,690,772 B`) exactly (44,902,690,772), as do `v1.0-trainval_meta.tgz` and `can_bus.zip`.
The loader's map-expansion line is STALE: it names v1.2 (17,136,555 B), which the current devkit refuses.
