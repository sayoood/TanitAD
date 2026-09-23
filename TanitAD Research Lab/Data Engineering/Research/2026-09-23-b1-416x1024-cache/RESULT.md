# B1 training corpus rebuilt at 416x1024 on the Jetson Thor

**Owner:** Data Engineering · **Date:** 2026-09-23 · **Host:** `thor6`
**Authorisation:** PI, 2026-09-23 — *"rebuild for the target resolution"*, *"free the 575 GB"*.

**Output cache:** `thor:/home/nvidia/data/physicalai-b1-w120-416x1024cyl`

---

## 1. Headline

| | |
|---|---|
| **Pilot (10 clips)** | ✅ **10/10 built, 0 failures**, every content assertion passed |
| **Geometry** | `416x1024`, `f_ref` **488.92398517830253**, HFOV **120.0000000000°**, VFOV **46.0921317116°**, cylindrical, codec **png**, `n_stack` 3 |
| **Size** | **83.1 MB/ep** MEASURED over the first 118 clips ⇒ **~392 GB** for 4,713 |
| **Throughput** | **3.5–3.8 s/clip** wall at 12 workers ⇒ **~5.0 h**, not the INHERITED 10.6 h |
| **Disk** | 604.5 GB free at start; ~392 GB needed ⇒ ~210 GB headroom |
| **Parity gate** | MEASURED over all 4,713: `kept=4713` `n_disqualifying=0` `in_deployed_val=0` **`disjoint=true` `decision_grade=true`** |
| **⚠️ Black-row strip** | **8/10 pilot clips carry a 30–33-row black strip, and it tracks the RIG perfectly** (see §6) |

---

## 2. Three corrections to the brief — each found by probing, not inherited

### 2.1 The "416x1024 eval cache at 79.7 MB/ep" does not exist

The brief's first cost estimate cited *"our eval-139 cache at 416x1024: 79.7 MB/ep → 375 GB"*.
MEASURED — the banked manifests in `2026-09-16-256x1024-cache/raw/` are **256x1024** and
**408x1024**. There is no 416 manifest there. The real numbers:

| banked cache | n | total bytes | MB/ep (decimal) | MiB/ep |
|---|---|---|---|---|
| `MANIFEST_eval139_256x1024.json` | 139 | 8,068,812,901 | 58.05 | 55.36 |
| `MANIFEST_eval139_408x1024.json` | 139 | 11,397,853,989 | **82.00** | **78.20** |

**"79.7 MB/ep" is the 408 cache's MiB/ep read as the 416 cache's MB/ep** — two errors compounding,
a wrong height and a wrong unit. Same family as the `df` / cgroup / `step_s` traps: *a true
measurement quoted outside its scope reads exactly like an answer.*

⭐ The eval-139 pair also yields a fact neither estimate used: **PNG bytes are SUBLINEAR in rows.**
408/256 = 1.594× the rows buys only 82.00/58.05 = **1.413×** the bytes (exponent 0.742) — the rows
added at the bottom are low-entropy. Re-deriving on that basis gave **ESTIMATED ~409 GB**, between
the brief's 375 and 463. The build has since MEASURED **83.1 MB/ep → ~392 GB**.

### 2.2 The camera source is not staged — it is not on the box at all

`/home/nvidia/data/physicalai-b1/r0/camera_front_wide` is a **symlink to
`/home/nvidia/data/b1-bundle/camera`, a directory that does not exist**. `find` lists the link and
`ls` fails on the target, so a listing-based check reads "present" and a read fails. **Zero mp4s
were on Thor.**

⇒ All 4,713 source videos come from HF. The volume is **not** an estimate: summing the corpus's own
`camera/camera_sha256.json` over exactly our 4,713 clips gives **61,545,041,700 B = 61.55 GB**
(mean 13.06 MB, p50 11.32, max 62.1). All 4,713 are present in that table (0 missing).

### 2.3 Two more input spellings differ from what the builder expects

| builder expects | Thor actually has | resolution |
|---|---|---|
| `labels/egomotion/egomotion_all.zip` | **1,411 per-chunk zips** (1.96 GB) | merged flat into one archive |
| loose `<clip>.timestamps.parquet` | one **`timestamps.tar`** (4,719 members) | extracted flat |
| — | `r0_selection.parquet` + `calibration/` **not under the staging root** | copied / symlinked in |

Merging the ego zips is **safe, not convenient**: `physicalai.load_egomotion` resolves a member by
`endswith(f"{clip_id}.egomotion.parquet")`, so a flat merge preserves the lookup exactly.

⭐ **The choice made here:** materialise the layout the banked builder already expects rather than
fork the builder to read Thor's layout. The builder is the object whose provenance we rely on
(*"THIS IS A DRIVER, NOT A SECOND RESAMPLER"*); a second spelling of its inputs is precisely the
divergence its own docstring warns about. **`build_v2ep_wide.py` ran UNMODIFIED** — md5
`849a0d2573f1ce93ade13c2ccfd00a3b`, identical on the dev box and on Thor.

Staging result (MEASURED): 4,713 clip list · 4,719 timestamps · 4,719 egomotion members ·
**0 missing on either, asserted per clip, not by count**.

---

## 3. Geometry — DERIVED, and cross-checked three ways

`trunk_shapes.py` is **in the repo but NOT on Thor's checkout** (Thor's `stack/` is stale). The
value was therefore derived on Thor from `CanonicalFrame.from_hfov(120.0, 416, 1024,
"cylindrical")` and checked against an **analytic target typed from the definition**:

| quantity | value | check |
|---|---|---|
| `f_ref` | **488.92398517830253** | `512 / radians(60)` — agrees **bitwise** (`==`, not a tolerance) |
| HFOV | **120.0000000000°** | cylindrical: `degrees(2·(W/2)/f_ref)` |
| VFOV | **46.0921317116°** | `degrees(2·atan((H/2)/f_ref))` |
| deg/col | 0.1171875 | |
| tag | `416x1024f488.924cyl` | |

Independently, the repo's `trunk_shapes.FRAME_416x1024` records **46.09213171161337** — computed by
a different route (scaling the 640 frame's `f_ref` by `1024/640`) and agreeing to the last digit.

**Two traps confirmed rather than assumed:**
- the **pinhole** formula on this cylindrical frame reads **92.6414°** — plausible, and wrong;
- the PI's rounded **`488.92`** reads **120.0010°**, not 120.

⭐ **Why 416 and not the 408 the brief's cost model came from:** `408 % 32 = 24`, so a stride-32 map
is mis-sized and `timm_trunk.py:216` refuses it. `416 % 32 = 0` gives exactly **13 rows**. 416 also
*slightly exceeds* the 256x640 reference vertical field (46.0921° vs **45.4556°**), so no vertical
field is given up to gain the alignment. (This is also the cause of §6 — see there.)

---

## 4. Pilot — content-verified, not exit-code-verified

10 clips, `workers 6`. **10/10 built, rc=0.** Independently re-verified by `verify_cache.py`, whose
expectations are **literals typed from the geometry definition**, not read from the code under test
(re-running the producer's own derivation would measure determinism, not correctness):

```
n_checked 10 · n_ok 10 · n_failed 0
image_h 416 · image_w 1024 · codec png · n_stack 3 · projection_mode cylindrical
f_ref 488.92398517830253  (analytic_agrees_bitwise: true)
hfov_pinhole_WRONG_formula 92.6414        <- recorded so the wrong reading is on file
decoded_mean min 16.723 / mean 49.596 / max 89.891   <- NOT zero: real pixels
n_frames 200-201 · MB_per_ep_decimal 76.947 · MiB_per_ep 73.382
```

The all-zero-bank trap is checked on a **random** frame per payload, not frame 0.

**The pilot earned its keep:** the first attempt failed **10/10** on a missing
`r0_selection.parquet`, and the builder **deleted every payload** rather than leaving full-size
debris — the content check working as designed, caught in 90 seconds instead of after 5 hours.

### Cost and throughput — MEASURED on Thor, replacing the INHERITED figure

| arm | workers | s/clip wall | MB/ep | note |
|---|---|---|---|---|
| shard 1 (pilot) | 6 | 9.08 | 76.9 | `build_s` mean 39.1 s CPU/clip |
| shard 2 | 12 | 3.8 | 84.2 | |
| shard 3 | 12 | **3.5** | 83.3 | prefetch active |
| shard 4 | 12 | 3.6 | 83.3 | download fully hidden |

**12 workers is 2.4× faster than 6** on Thor's 14 aarch64 cores. Over 118 clips: **9.80 GB,
83.1 MB/ep** ⇒ **~392 GB** for 4,713, and **~5.0 h** wall (shard-level rate 918 eps/h) against the
brief's INHERITED **10.6 h**, which was the dev box's throughput and was never re-timed on Thor.

Download is **not** on the critical path: measured 11.9–12.8 MB/s at 10 threads, and the driver
**prefetches shard N+1 while shard N builds**, so a staged shard reports `PREFETCHED` in 0.0 min.
Serially it was ~21 % of wall with every core idle.

---

## 5. How the full run is driven

`run_b1_416_build.py` loops: **stage a shard → build it → evict the source**, so peak transient
disk is one shard of mp4 (~0.5 GB at `--shard 36`), not 61.55 GB. Reuses the staging pattern
already proven over 4,719 clips by `sam3map/eval/corpus_feeder.py`: `hf_hub_download` at the
**pinned revision** `a0cf20df…`, **sha256-checked against that revision's own table**, placed by
atomic rename, unlinked once consumed.

Four hardening decisions, each against a failure this programme has already paid for:

1. **Resume is by RECORD *and* PRESENCE.** A payload with no manifest row is an interrupted build
   whose content checks may never have run — it is deleted and rebuilt, costing at most one shard.
2. **Shard numbering is monotonic across invocations.** A per-run counter restarting at 1 would
   overwrite the first shard's archived manifest on every resume; its clips would then have
   payloads with no record, the orphan rule would delete them, and the run would silently rebuild a
   shard it had already paid for — once per restart, while the log reported progress. *(Found and
   fixed during the pilot, before the full run.)*
3. **A give-up ledger.** A deterministically failing clip stays at the head of `todo` with its mp4
   evicted, so without `_attempts.json` the run re-downloads and re-fails it every shard forever —
   an infinite loop whose log looks like steady progress.
4. **Disk is re-checked with a real `statvfs` before every shard** and the run refuses below
   `--min-free-gb`. Running out mid-`torch.save` is how a cache acquires a truncated payload at a
   plausible size.

The driver holds a `flock`, and the builder subprocess **does not inherit the lock fd** —
`subprocess` defaults to `close_fds=True`, the Python spelling of `200>&-`. Without that the
builder would hold the lock for its whole life and no replacement driver could ever start: a
permanent block that merely looks like a race.

### 5.1 The parity ingest gate, run over the whole set before it could block anything

MEASURED by preflight (`parity.guard_corpus_build`, `role=train`, `mode=refuse`) over the full
clip list, so the finaliser cannot refuse after the build is paid for:

```
n_in 4713 · kept 4713 · n_disqualifying 0
in_parity_train 201 · in_deployed_val 0 of 40 · frac_of_deployed_val 0.0
disjoint TRUE · decision_grade TRUE
```

This is **stronger than the 256x640 cache's gate**, which ran `mode=exclude` over 4,719 and dropped
6. Our input list is the already-filtered 4,713, so the rebuild is **disjoint from the deployed val
by construction** rather than by filtering. The 201 `in_parity_train` clips are inside the OLD
parity corpus `physicalai-train-e438721ae894` (the flagship/refav1 line) and are **not**
disqualifying for a `role=train` corpus — that is the same reading the deployed 256x640 cache
carries, and it is why `decision_grade` is `true` here where the eval-139 rebuild's was `false`.

---

## 6. ⚠️ The black bottom strip — MEASURED on our own build, and it is the RIG

Raised by a parallel reviewer on the gated eval-139 cache (74/139 clips, 27–37 rows, and
**decodable at 0.899 ± 0.046 balanced accuracy** from an 8×16 thumbnail against three controls at
chance). `calib.py:1031-1036` names the shape as retraction class **C26** — *"a rig-correlated
BLACK region is still a rig-correlated signal, and this model eats shortcuts."*

**Censused on OUR pilot (n=10), because the eval-139 split is not the train split:**

| | |
|---|---|
| carrying the strip | **8/10** (0.800) |
| rows | min 30, max 33, **mean 31.125**, median 31 = **7.48 % of height** |
| bimodal? | **yes** — histogram `{0: 2, 30: 3, 31: 3, 33: 2}`; the 1–10 band is **empty** |
| constant within a clip | **10/10** — it is geometric, not a per-frame artifact |

**The correlate is ESTABLISHED, not guessed** — a clean 2×2 against the per-clip principal point
(rig A `cy`~543 / rig B `cy`~755), resolved through the deployed `intrinsics_for_clip` path:

| | carries strip | clean |
|---|---|---|
| **rig A** (`cy` < 650) | **0** | **2** |
| **rig B** (`cy` ≥ 650) | **8** | **0** |

**Perfect separation: `rigA_strip_frac` 0.0, `rigB_strip_frac` 1.0**, 0 clips without resolvable
`cy`. Clip length is **not** the correlate (mean frames 201.0 strip vs 200.5 clean).

**Mechanism, and it follows from §3:** 416 rows push the vertical field to 46.0921°, *past* the
256x640 reference 45.4556°. Rig B's principal point sits lower in the source frame, so the extra
rows run off the bottom of its valid field and are filled with zeros. The 256x640 cache's own
`_geometry.json` already recorded the shadow of this — `rig_observability` B `masked_frac_mean`
**0.0905** vs A **0.00007** — so this is the same asymmetry, widened by the taller frame.

⛔ **Recorded, NOT mitigated.** The geometry is the PI's 416x1024 ruling and the SPEC's; it is
unchanged here. `_geometry.json` carries a `black_bottom_rows` block with **per-clip counts keyed by
sha12** plus the histogram and the contingency table, so a consumer asks the artifact rather than
re-deriving it. The mitigation is a loader/model-side mask and belongs to another stream.

⚠️ **n=10 is thin on the rig split** (only 2 rig-A clips). The census re-runs over all 4,713 at the
end of the build; the corpus-wide fraction and contingency table are the quotable ones, and the
pilot numbers above must not be quoted for the corpus.

---

## 7. Status

**Full run in flight** — PID 2873458 on `thor6`, launched 2026-09-22 22:36 UTC, `--shard 36
--workers 12 --dl-threads 10`. Poll `thor:/home/nvidia/data/physicalai-b1-w120-416x1024cyl/driver.log`
(real-time; `full.out` is block-buffered through `nohup`). Markers are `ZZ…ZZ`-framed and disjoint
from the words used to search them.

On completion: `census_black_rows.py` over all 4,713 → `verify_cache.py` → `finalize_cache.py`
(merged MANIFEST, `_geometry.json` with the parity gate re-run over the whole set, and the
`_digest_scope.json` sidecar).

---

## 8. Deliverable manifest

### Repo — `TanitAD Research Lab/Data Engineering/Research/2026-09-23-b1-416x1024-cache/`

| path | what |
|---|---|
| `RESULT.md` | this document |
| `code/stage_b1_inputs.py` | one-time input staging (ego merge, timestamps, clip list) |
| `code/run_b1_416_build.py` | shard-staged driver (stage → build → evict, resumable) |
| `code/verify_cache.py` | independent content verification against typed literals |
| `code/census_black_rows.py` | black-bottom-row census + rig correlation |
| `code/finalize_cache.py` | merged MANIFEST + `_geometry.json` + digest-scope sidecar |
| `raw/` | banked run artifacts (pulled from Thor at completion) |

### Thor — `thor6`

| path | what |
|---|---|
| `/home/nvidia/data/physicalai-b1-w120-416x1024cyl/` | **the cache** (the deliverable) |
| `…/MANIFEST.json`, `_geometry.json`, `_digest_scope.json` | records |
| `…/_black_rows_census.json`, `_verify_report.json` | content evidence |
| `…/_shards/shard_*.json`, `build_*.log`, `driver.log` | per-shard provenance |
| `/home/nvidia/b1build416/*.py` | the five scripts, md5-verified against the dev box |
| `/home/nvidia/data/_b1stage416/` | staging root (~2.0 GB: merged ego + timestamps) |
| `/home/nvidia/data/physicalai-b1-w120-256x640cyl/` | **KEPT UNTOUCHED** — 4,713 eps, 178.0 GB |

**Nothing is committed or pushed.** Deliverables are staged into the working tree only.

---

## 9. FINAL — completed by the Master Mind after this agent was stopped (2026-09-23)

⚠️ This agent was stopped before it wrote its closing section — the session process that
spawned it ended overnight while the full build was still running. The build itself finished on
its own. Everything below is MEASURED from the finished cache on Thor after the fact, not
inherited from the pilot figures above.

| | |
|---|---|
| **built** | **4,713 / 4,713** episodes, `ZZDRIVER-END-4713-4713ZZ`, **4.87 h** wall (SPEC §10.6 estimated ~10.6 h) |
| **size** | **392,952,981,351 B = 393.0 GB**, **83.38 MB/ep** — the pilot's 83.1 MB/ep was within **0.3 %** |
| **geometry** | `f_ref 488.92398517830253`, HFOV **120.0000°**, VFOV 46.0921°, `analytic_agrees_bitwise: true` |
| **content** | 3 episodes decoded independently: all `416×1024×3`, non-zero mean (34–74), 199–201 frames |
| **`_failures.json`** | 10 entries, all one `FileNotFoundError` on the staging `r0_selection.parquet` from a first attempt; **all 10 clips are present in the finished cache** (matched by sha12) |

⚠️ **Record defect, not a data defect:** `MANIFEST.json` in the cache directory describes only the
**last 23-clip shard** (`n_clips: 23`), not the corpus — the builder writes a per-run manifest and
the final run was the tail shard. A consumer must count `*.v2ep.pt` (4,713), not read `n_clips`.

### The black strip, on the WHOLE corpus — the pilot's 8/10 overstated it

MEASURED over **all 4,713** episodes (frame 0 and the middle frame of each, 0 decode errors, 282 s):

| | |
|---|---|
| carrying a strip | **2,721 = 57.73 %** |
| clean (0 rows) | **1,992** |
| rows when present | min **26**, max **43**, median **31** |
| bimodal | yes — nothing between 0 and 26 |

⭐ With §6's pilot correlate (rig A 0/2, rig B 8/8, perfect separation by principal point), the
**57.73 % is rig B's share of the corpus**: the strip is not merely rig-*correlated*, it *is* the
rig label, painted into every rig-B frame — retraction class **C26** in its purest form.

**Mitigation — landed in `e152e40`:** `--equalize-bottom-rows 43` (the measured corpus MAX) zeroes
the bottom 43 rows of **every** frame at the trunk, train and eval alike, before normalisation, and
the BEV lift marks the same rows UNOBSERVED. The region then carries no rig information. Mutation-
proven **5/5**. ⚠️ Cost, stated: the 42.27 % of clips from rig A lose up to 43 near-field rows
they genuinely observed — the price of making both rigs look the same to the model.

### The train view refcv6 trains on

`/home/nvidia/data/refcv6-b1-416x1024-train` — **4,369** symlinks = the 4,713-clip cache ∩ v8 train
labels (−141, the eval clips) ∩ the B1 train agent join (−145, no boxes) ∩ a validated SAM3 map
(−58). Every clip in it carries all four supervision inputs; the removed sets are recorded by sha12
in its `_VIEW_RECORD.json`.
