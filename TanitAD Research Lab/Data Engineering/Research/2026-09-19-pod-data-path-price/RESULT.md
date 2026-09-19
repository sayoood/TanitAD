# E20 input: how the 416×1024 refcv6 data reaches a pod — **build it ON the pod; this box uploads at 1.2 MB/s**

**Date** 2026-09-19 · **Owner** DataFlyWheel · **Asked by** Master Mind · **0 GPU**
⛔ **Preparation only.** Nothing was provisioned, pushed, created or changed. The one upload was
195 MB of random bytes to a speed-test sink. HF was read through metadata calls only.

## The answer

| route | over the network | rate | hours | HF $ |
|---|---|---|---|---|
| **(a)** private HF push, pod pulls | 380.6 GB up, then 380.6 GB down | up **1.14–1.23 MB/s** (MEASURED); down ~118 MB/s (INHERITED) | ⛔ **86–92 h of upload**, plus a 10.4 h build here (blocked while A8 runs) and a 0.9 h pull | $0 today: private 176.9 GB, + 6.5 GB of SAM3 maps still to publish, + 380.6 GB = 564.0 of 1,000 GB. ⛔ **19.4 GB over (≈ $0.35/mo) if the 455 GB public 256×640 repo ever goes private** |
| **(b)** direct, dev box → pod | 380.6 GB | 1.14–1.23 MB/s | ⛔ **86–92 h**, with a pod online the whole time | $0 |
| ⭐ **(c)** build on the pod from the HF corpus | **76.2 GB down**, 0.3 GB up | ~118 MB/s (INHERITED); still 0.7 h at 30 MB/s | **0.2 h download + 2–8 h build** (4.0 h at 16 vCPU, ESTIMATED) | **$0** — the sources are already in the private corpus |

⇒ **(c).** It moves a fifth of the bytes, none of them through this box's uplink, and adds nothing to
either HF storage pool. The data also ends up on the pod's own volume, so a dead arm restarts
without a re-transfer (C6: there is no `--resume`).

## Measured today

### The uplink: 1.19 MB/s, and the link is the cap

Payload: `os.urandom` bytes, so a proxy cannot compress them and they carry nothing. Endpoint:
Cloudflare's public speed-test upload sink. ⚠️ **This is a neutral third-party endpoint, not one we
control.** No pod is running, and writing an HF repo is outside this brief. So it measures the box's
uplink, not the path to HF or to a pod specifically. 195 MB sent, 13:49–13:52 local
(`raw/uplink_probe.json`).

| trial | MB/s |
|---|---|
| one stream, 25 MB ×3 | **1.144 · 1.187 · 1.237** |
| one stream, 10 MB ×2 | 1.298 · 1.244 |
| **4 parallel streams**, 25 MB each | **1.230 aggregate** (0.31–0.40 per stream) |

Parallel streams buy nothing, so the link itself is the limit (≈ 9.6 Mbit/s), not a per-connection
window. **A second, independent probe agrees:** the 2026-07 dev-box relay moved 3.26 GB in 47 min,
which is **1.16 MB/s** (INHERITED, from the pod-transfer memory). D:'s 18.7 MB/s read rate (MEASURED
earlier today) is 15× faster, so the disk is not what binds.

### The size: 370.3 GB for the TRAIN split, not 386.5

Three things change the pre-registration's figure:

1. **The pod trains on 4,572 clips, not 4,713.** B1 (4,713 clips, membership digest reproduced) is
   **exactly 4,572 v8-train + 141 v8-eval**: 0 overlap and 0 left over. The eval clips are already
   built: the 139 at 416×1024, and the clean-124 halves.
2. **The per-episode size is now measured at the ruled geometry.** The 139 episodes of
   `v2ep-eval139-416x1024cyl` average **83.02 MB** (SD 14.11, range 52.0–136.3). They are PNG, with
   201 stored frames (every 3rd camera frame) at 413 KB per frame. The old 82.00 MB was 408×1024.
3. **The 139 are not a random draw of the train split.** Their source mp4s average 14.64 MB against
   the train split's 13.01 MB.

| model (fit on the 139; applied to the 4,572's MEASURED covariates) | train total | 95 % CI (bootstrap over the 139) | leave-one-out RMSE per episode |
|---|---|---|---|
| mean × N | 379.5 GB | 368.8–390.4 | 14.2 MB |
| per stored frame | 380.7 GB | 370.0–391.7 | 14.1 MB |
| ⭐ **linear in source mp4 bytes** (R² 0.62) | **370.3 GB** | **363.8–377.0** | **8.9 MB** |
| proportional to source bytes (the 1.121× rescale the C4 note suggested) | 337.4 GB | 311.1–366.8 | ⛔ 40.7 MB — the wrong model |

⭐ **Cache size is not proportional to source size.** The fit has a **64.8 MB intercept** and a slope
of 1.25 cache bytes per source byte, so rescaling by the source-size ratio under-estimates by about
33 GB. ⚠️ The interval covers coefficient uncertainty only. A train/eval difference the covariates
cannot see is caught only by building a random train sample (see *Blocked*).

⇒ **The pod's cache = 370.3 (train, ESTIMATED from the model) + 10.35 (clean-124 halves, MEASURED) = 380.6 GB.**

### What else the trainer reads: A8's argv on this box, the refcv6 configuration

| input | eval side (local, measured) | train side (what the pod needs) |
|---|---|---|
| `--v2-cache` / `--eval-cache` | clean-124 halves, 10.35 GB | 370.3 GB, to build |
| `--map-gt-root` (SAM3 GT) | 0.299 GB for 135 clips (2.2 MB/clip) | 9.98 GB projected. **2,443 of 4,572** train clips are on HF today (2,398 gt + 45 flagged); 2,136 clips remain, ≈ 85 h at 25 clips/h (INHERITED) |
| `--agent-join` / `--join3d` | 12.7 MB (139 clips) | ⛔ **does not exist.** Not in D: artifacts, not in the C: caches, and HF holds only the raw `agents/obstacle_offline` (2.35 GB). Its builders read the cache (`--pose-source v2ep --eps-dir`), so on route (c) they run on the pod after the build. Their run time is unmeasured |
| `--agent-rig-extrinsics` | `extrinsics141.json`, 0.1 MB | ⛔ none for the train clips. Source: HF `calibration/sensor_extrinsics.parquet` (1.65 MB). Builder: `stack/scripts/build_rig_extrinsics_table.py` |
| `--v7-labels` | v8 eval, 89 KB | v8 train, 2.69 MB (on HF) |

⭐ **The cache build reads no map.** Neither the builder (`build_v2ep_wide.py`) nor its resampler
(`v2_compressed.py`) contains a single reference to SAM3 or semantic maps. ⇒ The maps gate
**training**, not the build, so route (c) can build the whole cache before SAM3 finishes.

## Which subset first

| tier | what | size | by (c) | by upload from this box |
|---|---|---|---|---|
| **0** | the clean-124 halves + eval aux (maps, 3-D join, extrinsics, labels) | **10.67 GB** (MEASURED) | rebuild the 139 (2.0 GB of mp4, ≈ 7 min at 16 workers) and re-split with A0's rule. The aux (0.32 GB) comes from this box in ≈ 4.6 min | 2.6 h |
| **1** | the first **2 of 8 train shards** (a shard is 1/8 of the train split in sorted-sha12 order, ≈ 571 clips) | **94.9 GB** (92.4 cache + 2.5 maps), ESTIMATED | ≈ 58 min at 16 workers | 22 h |

* **Tier 0** feeds the pod's first job (§3.4: a 200-step rate measurement), the wiring smoke and C8
  (T1 at 416×1024). ⭐ **It is also the positive control for the pod build.** This box's
  `MANIFEST.json` carries a sha256 for each of the 139 episodes. A pod rebuild that reproduces them
  proves the pod's builder byte-for-byte before 4,572 clips are trusted to it. If the bytes differ
  (library versions), fall back to decoded-pixel equality. Upload the 10.67 GB (2.6 h) only if that
  also fails.
* **Tier 1 is two shards because the clean-124 cannot show the read path.** At batch 20 over 4,572
  clips, the loader misses its LRU about 98.6 % of the time (lru 64; `v2_dataset.py:260`) and reads a
  whole episode on every miss (`torch.load`, `:343`). That is ≈ **1.6 GB of file reads per step**
  (ESTIMATED). The 10.4 GB clean-124 sits in page cache after one pass, so a rate job on it measures
  compute, not the corpus-scale read path. Two shards (≈ 95 GB) exceed the page cache of any pod with
  less than ~90 GB of RAM. For scale: refcv5-v2's 4.0 s/step on an A40 (INHERITED) at 256×640 came to
  ≈ 170 MB/s of the same reads.
* ⛔ **Shards cannot start the real arm early.** The loader lists `*.v2ep.pt` once, when it starts
  (`stack/tanitad/data/v2_dataset.py:398`), and caches that list. A run started on two shards trains
  on two shards for its whole life, and with no `--resume` (C6), picking up more shards means
  starting over. The training arm waits for all 4,572 clips — or it is a different, smaller
  experiment and says so.

## For the pod request (E20)

1. **Disk.** A (c) build peaks at ≈ **458 GB** (cache 381.8 + sources 65.9 + maps 10.0) and settles at
   ≈ **391 GB** before checkpoints. The last measured pod `/workspace` MooseFS quota was ~466 GB
   (INHERITED, 2026-07), and `df` hides it. ⇒ **Ask for ≥ 600 GB**, or delete each source mp4 once it
   is built; verify with a `dd` write test.
2. **The build needs no GPU.** 2–8 h of CPU, ESTIMATED from this box's 48.9 worker-seconds per clip on
   P-cores; server vCPUs may be up to ~2× slower. A CPU pod on the same network volume keeps those
   hours off a GPU bill.
3. **Read access to the private corpus from the pod**, with the token read in place as today. The
   fetcher's HF path checks every mp4 against its LFS sha256.
4. **Measure in the first minutes:** the HF → pod rate, the build rate and per-worker RAM (both
   INHERITED or ESTIMATED here), and the volume's read rate under the 200-step job.
5. ⛔ **The ordering trap stays the PI's decision.** Route (a) plus the 256×640 repo going private puts
   private storage at 1,019.4 GB, over 1 TB, which is billed and so forbidden. Route (c) adds nothing
   to either pool.

## Blocked — and what unblocks it

| check | why not today | unblocks when |
|---|---|---|
| **Build a random sample of ~48 train clips at 416×1024** here (~7 min at 6 workers). It is the only test of the 370.3 GB against train/eval differences the covariates miss | ⛔ **A8** (5,000 steps, **no `--resume`**) is live on this box reading D:, NAVSIM caching is running, and only **6.4 GB of RAM is free**. A build here risks a run that cannot resume | A8 finishes |
| pod-side rates | there is no pod (the PI has parked it until the dev-box checklist is done) | the pod's first job |
| the train-side 3-D agent join and extrinsics table | their builders read the cache, so they can only run after the build | step 2 of route (c) |

## Evidence

`raw/inventory.json` is read-only: sizes, parquet footers and HF metadata, with sha12 only (no clip
ids). The other outputs are `raw/size_model.json`, `raw/uplink_probe.json` and `raw/pricing.json`,
and `code/` holds all four scripts. The INHERITED constants are listed, with their sources, at the top
of `code/price_paths.py`. HF prices are PUBLISHED: fetched 2026-09-19 in the C4 package and not
re-fetched here.

```
python code/pod_data_inventory.py raw/inventory.json
python code/size_model.py raw/inventory.json raw/size_model.json
python code/uplink_probe.py raw/uplink_probe.json
python code/price_paths.py raw/inventory.json raw/size_model.json raw/uplink_probe.json raw/pricing.json
```

🔒 Clip ids appear only as sha12. A scan of every file here found no clip id, 8-hex prefix or 12-hex
tail, against a control (`GOALS_AND_CLAIMS.md` at the tip) that reads 38 (`raw/leak_scan.txt`).
⚠️ **The first build did not pass, and the reason is a trap worth knowing.** `raw/inventory.json`
stored per-clip sizes as raw byte counts. An 8-digit number *is* an 8-hex token, and one mp4's byte
count equalled the prefix of a clip in the 306,152-id index; about 0.3 such collisions are expected
across 4,572 counts. The gate refused the file. Per-clip sizes are now stored in MB to 6 decimals,
which is exact to the byte and can never form an 8-hex token. The rebuilt `raw/pricing.json` is
identical to the first one, field for field.
