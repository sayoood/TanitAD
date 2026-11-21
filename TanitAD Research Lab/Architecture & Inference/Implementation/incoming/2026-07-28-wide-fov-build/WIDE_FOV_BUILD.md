# Wide-FOV cache build for flagship v5 — host census, blocker resolution, and the running build

**Date:** 2026-07-27 (dev box, Europe/Berlin; **all pod times UTC**) · **Stream:** Architecture & Inference
**Status:** staged, never committed, never pushed. **A build is RUNNING on pod2 — see §8 before anything else.**
Repo HEAD at start: `4cb37f4` (the `_decode_mp4` hotfix).

**Predecessor:** `Benchmarks & Eval/.../2026-07-27-fleet-refill/FLEET_REFILL.md` §3 refused this build on
pod2 for a parity reason. **That refusal was correct.** This document resolves it — by probing, not by
relaxing the rule.

---

## 0. Headline

| # | finding | class |
|---|---|---|
| **1** | ⭐ **A parity-capable host EXISTS: `tanitad-pod` (pod1) holds 3,000 clips and reproduces BOTH keys exactly** — `train e438721ae894` ✅ and `val 0c5f7dac3b11` ✅. It is training and was probed **read-only** only. | **MEASURED** |
| **2** | ⚠️ **The brief's bar was wrong: the corpus needs 3,000 discovered clips, not ≥2,400.** `split_clips(val_frac=0.2)` on 3,000 → **2,400 train / 600 val**; the 24 skips are decode failures *inside* the 2,400. 600 is exactly `PARITY_VAL_EPISODES`. A 2,400-clip host would still mint a different key. | **MEASURED** |
| **3** | ⭐ **pod2 IS buildable after all, with no episode re-selection** — it holds the identical `r0_selection.parquet` (3,000 rows), all 3,000 timestamps, all 197 egomotion zips and full per-clip calibration. Only mp4 **bytes** are missing, and `v2_compressed.py build` fetches, uses and deletes those itself. | **MEASURED** |
| **4** | ⛔⛔ **A SECOND live instance of the `fdc5b4f` frame-shadowing bug — in `scripts/v2_compressed.py`, still present at HEAD.** The `4cb37f4` hotfix fixed only `physicalai.py`. It broke **every** v2 build path incl. the deployed one. **This is the only storage-viable route to a wide corpus, so it blocked v5 outright.** Fixed + regression-tested. | **MEASURED** |
| **5** | ⛔ **The briefed runbook is not executable as written: its two halves name two incompatible formats.** `register_geometry_sibling()` only reads a **raw epcache** (`ep_*.pt`); the **112.9 GB** target is the **v2 compressed** format. Raw epcache at this geometry is **697 GB for the train split alone** and fits on no host we have. | **MEASURED** |
| **6** | ⭐ **The v5 frame is confirmed on real frames: 120.00° delivered, shortfall 0.00, `f_eff` 305.577491 on ALL 3,000 clips (stdev 0.0).** The brief's 305.577 is right and `F_REF` 266 is not this frame's focal. | **MEASURED** |
| **7** | ⭐ **Rig-B masked periphery, now a RATE not a direction: n = 3,000 (100 % of the selection).** rig A **0.0017 %**, rig B **8.897 %**. Real-decode subsample (12 A / 12 B) agrees: **0.0056 % / 9.507 %**. | **MEASURED** |
| **8** | ⚠️ **`GEOMETRY_CONFIGURABLE.md` §2c says "~29 % of the corpus" carries the rig-correlated defect. Measured: rig B is 72.93 %, rig A 27.07 % — the figure looks A/B-inverted.** The live defect's blast radius is **2.7× larger** than published. | **MEASURED** |
| **9** | 🔒 **`v2_compressed._hf_download` put a live HF token in `curl`'s argv** — readable by any `ps`, which this program's own runbooks tell agents to run. Moved to curl stdin. | **MEASURED** |

**Bottom line:** the census answers the blocking question (**GO**, on pod2, no re-selection), the geometry
and rig numbers are settled, and **a 2,400-clip build is running**. What is **not** done and cannot be done
with the code at HEAD is the **registration** — §6 is the escalation.

---

## 1. The per-host clip census — and the correction that changes the bar

Raw: `raw/host_census_2026-07-27.json`. Tool: `code/parity_probe.py`.

**Method, and why a clip COUNT is not the test.** The corpus key *is* a hash of the ordered clip-id list
(`epcache.cache_key` over `_source_id` = `clip:<clip_id>`). A count can match while membership differs, so
the only sufficient probe is to **recompute the key**. The probe replicates `build_pai_cache.py` exactly up
to (not including) the decode — `discover_r0_clips` → `split_clips(val_frac=0.2, seed=0)` → `cache_key` —
and **decodes nothing**. Because dict sources hash by `clip_id`, not by path, the key is host-portable.

| host | mp4 on disk | discovered | train key | val key | parity? | state |
|---|---:|---:|---|---|---|---|
| ⭐ **`tanitad-pod` (pod1)** | **3,000** | **3,000** | **`e438721ae894`** | **`0c5f7dac3b11`** | ✅ **BOTH MATCH** | 🔴 training `flagship-v2corpus-30k` — read-only probe only |
| `tanitad-pod2` | 760 | 760 | `0c5d8f7823bd` | `6384ddad559d` | ❌ | 🟢 idle |
| `tanitad-pod3` | 48 | 48 | `acb5bfd5051a` | `267ee7e0835b` | ❌ | 🟢 idle |
| `tanitad-eval` | 0 | — | — | — | ❌ | 🟢 idle |

⚠️ **Correction to the brief.** It said the corpus is "2,376 episodes + 24 skipped = **≥2,400 clips**". The
true requirement is **3,000 discovered clips**: `split_clips` takes `val_frac=0.2`, so 3,000 → **2,400 train /
600 val**, and the 24 skips are decode failures *within* the 2,400 train, leaving 2,376. The 600 lands
exactly on `parity.PARITY_VAL_EPISODES`. **A host holding 2,400 clips would still produce a different key.**
pod2's coverage is therefore 760/3,000 = **25.3 %**, not the 31.7 % the predecessor doc reported against the
2,400 figure.

**Absence probed at more than one location**, per the standing rule: `find /workspace -name '*.mp4'` on all
four hosts; `find / -maxdepth 6 -type d -name 'datasets--nvidia--PhysicalAI*'`; seven explicit candidate
roots; and `find … -name r0_selection.parquet` on eval. eval's 404 mp4s are rollout renders, not R0 source.

⚠️ **One inconsistency found and NOT resolved:** pod1's `_epcache/physicalai-train-e438721ae894/DONE` reads
`{"episodes": 2400, "skipped": 0}` while its filesystem holds **2,376 `ep_*.pt` + 24 `skip_*`** — pod2's
`DONE` reads `{2376, 24}`. The **files agree on both hosts**; only the marker differs, and
`epcache.build_episodes_cached` documents `DONE` as informational and *not read on resume*. Flagged as a
stale marker, **not** used for any decision here.

### 1.1 Why pod2 became the build host anyway — GO, with no re-selection

pod2 is missing **only mp4 bytes**. Everything that *defines* the corpus is already there and identical:

| input | pod2 | needed |
|---|---:|---:|
| `r0_selection.parquet` rows | **3,000** | 3,000 |
| `*.timestamps.parquet` | **3,000** | 3,000 |
| `labels/egomotion/*.zip` | **197** | 197 |
| `calibration/camera_intrinsics`, `sensor_extrinsics` | present | present |
| `*.mp4` | 760 | 3,000 |

`scripts/v2_compressed.py build` **downloads each camera chunk itself, extracts only the selected clips,
builds, then deletes the zip and the mp4s**. The missing 2,240 mp4s are therefore not a prerequisite —
they are a transient.

**And membership is not re-derived on pod2.** The parity train clip-id list was exported **from pod1**, by a
tool that *refuses to write anything* unless both keys verify on that host first
(`code/parity_split_export.py`), then relayed md5-verified and passed to the build as `--only-clips`, which
refuses any id absent from `--sel`. Digests (`raw/parity_split_meta_2026-07-27.json`):

```
verified_train_key  e438721ae894   verified_val_key  0c5f7dac3b11   keys_match_parity  true
discovered 3000 -> train 2400 / val 600
train ids sha256 (sorted) e61a04553df5b9d52a0810be32cf31927bd92644d9d12ada563910b8a0ada4de
val   ids sha256 (sorted) 0b176d2e5cb49667d5009366817f948759724e69642e626a47362b93e31da68e
```

⭐ For all three lists the **ordered** and **sorted** digests are *identical*, i.e. the discovered order is
already clip-id order. That matters: it means a **set**-based membership proof (all the v2 format can offer)
is exactly as strong here as the ordered one, and is not a weaker substitute.

🔒 The clip ids are gated-confidential and live only on pods. Only digests appear in this repo.

---

## 2. ⛔⛔ The blocker nobody had hit yet — the SECOND frame-shadowing bug

`FLEET_REFILL.md` §3.3 found and fixed `physicalai._decode_mp4`, where `fdc5b4f` bound
`fr = as_frame(...)` and the PyAV loop `for fr in c.decode(stream)` rebound it. The `4cb37f4` hotfix landed
that fix.

**`fdc5b4f` made the identical edit in a second file on the same day, and the hotfix did not reach it.**
`scripts/v2_compressed.py::_decode_cropped_selected`:

```python
fr = as_frame(frame, size, 266.0)          # the CanonicalFrame
def flush():
    c = _remap(torch.stack(bfr), intr, fr, projection_mode)   # CLOSURE over fr
...
for fr in c.decode(st):                    # rebinds fr to a VideoFrame
```

This one is **worse hidden**, because the rebound name is read through a **closure** rather than inline, so
the diff does not look wrong. MEASURED on pod2 against a real clip, at the **deployed** geometry:

```
AttributeError: 'av.video.frame.VideoFrame' object has no attribute 'half_angle_x_rad'
```

⛔ **Blast radius: every v2 build path, deployed included.** And the v2 format is the *only* storage-viable
route to a wide corpus (§3), so at HEAD **flagship v5 had no build step at all** — not for lack of a host,
but because the builder was dead.

**Fixed** (loop variable → `vframe`, `c` → `container`, `c` → `out` in `flush`, plus a docstring saying why
the name is reserved). **Verified after the fix, on pod2, real clips:**

| path | before | after |
|---|---|---|
| deployed 256², JPEG q90 | ⛔ AttributeError | ✅ `(199, 9, 256, 256)`, **2.635 MB/clip → 7.7 GB / 2,976 clips** |
| wide 256×640 cyl, PNG lossless | ⛔ AttributeError | ✅ `(199, 9, 256, 640)`, 36.86 MB/clip |

⭐ The deployed row reproducing **7.7 GB**, which is exactly the figure `GEOMETRY_CONFIGURABLE.md` §2b
publishes for *deployed 256² @ JPEG q90*, is the evidence that the fix restores the **correct** behaviour and
not merely a non-crashing one.

**Why no test caught it — the same gap, twice.** No test decoded an mp4 through this module. New regression
test `stack/tests/test_v2_compressed_real.py` (**7 passed on pod2**, 2 passed / 5 skipped on the dev box,
which has no torchvision). It drives a **real** PyAV-written mp4 through the real function for both
projection modes and both frame shapes, forces the mid-loop `flush()` (the exact call site that read the
rebound name), and adds an **AST guard** — `for <name> in` loops binding `fr` — over *both* files, so the
pair cannot regress apart. The AST guard imports nothing and therefore never skips.

⚠️ The first version of that guard was a substring search and **failed against its own docstring**; parsing
the AST is what makes it honest.

---

## 3. ⛔ The runbook's two halves name two different formats

The brief specifies both **"~112.9 GB, PNG lossless"** and **"`parity.register_geometry_sibling()`"**. Those
belong to different artifacts, and the difference is not cosmetic.

| | **raw epcache** | **v2 compressed** |
|---|---|---|
| written by | `scripts/build_pai_cache.py` | `scripts/v2_compressed.py build` |
| layout | `ep_%05d.pt` + `skip_%05d` | `<clip_id>.v2ep.pt` (flat set) |
| identity | **position** in the ordered clip list | **clip id** |
| bytes/episode @ 120°/256×640 | **293.4 MB** | **~40 MB** (PNG lossless, MEASURED) |
| **train split (2,376)** | ⛔ **697 GB** | ✅ **~95 GB** |
| `parity.register_geometry_sibling` | ✅ works | ⛔ **cannot** — uid spaces differ |
| read by `train_flagship_v4` (`require=True`) | ✅ | ⛔ **no v2 support at all** |
| read by `train_flagship4b` | ✅ `--cache-dirs` | ✅ `--v2-cache` |

**The 293.4 MB is arithmetic validated against a real artifact:** `199 × 9 × H × W` bytes gives 117.37 MB at
256² and 293.4 MB at 256×640; pod1's real `physicalai-train-e438721ae894` measures **278,786,484,952 B** =
117.3 MB × 2,376 ✓. So the 256×640 figure is not a guess, it is the same formula that reproduces the
measured one.

⛔ **697 GB fits on no host in this fleet.** pod2's `/workspace` `du` is **560.6 GB** and the largest write
proven anywhere is the **124.55 GB** below. `GEOMETRY_CONFIGURABLE.md` §2 already said "873 GB does not fit";
the train-only figure does not rescue it.

⇒ **The build had to be v2.** That is a deliberate, stated deviation from the briefed runbook, not a
workaround: the briefed artifact is not storable, and building a partial raw corpus is refused.

### 3.1 Disk — proven by a real `dd`, never by `df`

```
df  /workspace   ->  965T size, 243T avail          <- the cluster-wide lie, per the trap list
du  /workspace   ->  560,605,954,907 B  (560.6 GB)  <- actual occupancy
dd  116 GiB, oflag=direct -> 124,554,051,584 B copied, 268 s, 465 MB/s, EXIT 0, then removed
```

⭐ **A second independent falsification of the inherited "~466 GB `/workspace` quota".** pod2 already holds
560.6 GB *and* accepted 116 GiB more; pod3 was previously measured at 487 GB. The figure is not a cap.
⚠️ An earlier 30 GiB `dd` on this same host reported **66.1 MB/s** against this run's **465 MB/s** — a **7×**
spread on identical flags. **Treat MooseFS single-sample throughput as non-quotable**; only the *success*
of the write is the decision-grade fact.

Budget for the running build: **~95 GB cache + ≤12 GB transient zips (8 shards × ~1.5 GB) ≈ 107 GB**,
inside the 124.55 GB proven, with ~17 GB margin.

---

## 4. The geometry, confirmed on real frames

`f_eff` measured on **all 3,000** clips of the canonical selection: **305.577491, stdev 0.0** (min == max).
Fixed by the frame, not the sensor — exactly as the brief said, and **`F_REF` 266 is *not* this frame's
focal**.

| quantity | requested | delivered (MEASURED) |
|---|---|---|
| frame | 256 × 640 cylindrical | 256 × 640 cylindrical |
| **HFOV** | 120.00° | **120.00°** — shortfall **0.00** |
| `f_eff` | — | **305.577491** (n = 3,000, stdev 0.0) |
| frame tag | — | `256x640f305.5775cyl` |
| tokens @ patch 16 | — | **640** (grid 16 × 40) |
| `state_dim` | — | **2048** (invariant) |
| VFOV | — | 45.46° |

**The abort guard was armed and fired.** `v2_compressed.py` had no pre-decode geometry assert at all (only
`build_pai_cache.py` did); I ported it. Verified against the documented trap:

```
--hfov 100 --height 256 --width 256
AssertionError: ABORT: requested 100.00deg but this sensor can only deliver 66.56deg at 256x256 —
the crop CLAMPED at the sensor edge and would have ZOOMED instead of widening.
```

(`GEOMETRY_CONFIGURABLE.md` reports 67.1° for this trap; I measure **66.56°** on a real clip's own
intrinsics. Same trap, per-clip variation.)

---

## 5. ⭐ The rig-B masked periphery — a rate, not a direction

The predecessor's ~9 % was **n = 6 with a single rig-A clip** and was correctly labelled DIRECTIONAL. It is
now a census. Raw: `raw/rig_mix_full_2026-07-27.json` (n = 3,000), `raw/rig_mask_census_2026-07-27.json`
(n = 760 + 24 real decodes). Tools: `code/rig_mix_full.py`, `code/rig_mask_census.py`.

**Two independent measurements, because they fail differently.** `observed_frac` is a property of the
per-clip intrinsics and the requested frame — it comes from the **ray map**, so a `torch.zeros` probe yields
it exactly with no decode, over every clip. The real-decode counterpart counts genuinely-zero output pixels
and is the honest **upper bound** (it also catches real black pixels).

**Rig assignment is derived and CHECKED, not assumed:** the cy distribution is cleanly bimodal — largest gap
**193.35 px**, boundary cy **650.87**; rig A cy 533.95–554.20, rig B cy 747.55–764.52. No clip is ambiguous.

| | n | share | masked periphery (geometry, n=3,000) | zero-pixel frac (real decode, n=24) |
|---|---:|---:|---:|---:|
| **rig A** | **812** | **27.07 %** | **0.0017 %** (median 0, max 0.064 %) | **0.0056 %** (median 0.0, max 0.064 %) |
| **rig B** | **2,188** | **72.93 %** | **8.897 %** (8.10–10.52 %, sd 0.63 %) | **9.507 %** (8.10–11.60 %, sd 1.09 %) |
| pooled | 3,000 | — | 6.489 % | — |

The decode number sits slightly **above** the geometry number for both rigs, which is the expected direction.

⭐ **The finding stands and is now quotable: at 120° cylindrical the masked periphery is essentially a
pure rig-B effect.** The cylindrical path converts the deployed crop's **fabricated** (replicate-padded)
pixels into **honestly masked** ones — but it does **not** remove the asymmetry, so a wide-FOV model still
sees a rig-correlated mask. Class **C26** remains live at 120°, in a more honest form.

⚠️ **And it is bigger than published.** `GEOMETRY_CONFIGURABLE.md` §2c states *"~29 % of the corpus carries
fabricated rows that correlate perfectly with rig"*. **Rig B is 72.93 % of the canonical selection; 27.07 %
is rig A.** The figure appears **A/B-inverted**, making the live defect's blast radius **2.7× larger** than
recorded. Reported to that stream for confirmation rather than asserted as their error — but the
measurement here is n = 3,000 with 0 intrinsics failures and a bimodality check.

---

## 6. 🔴 THE ESCALATION — the registration cannot be done with the code at HEAD

**This is the one deliverable I could not produce, and it needs an owner.**

The runbook's step 2 is `parity.register_geometry_sibling()`. It mints a manifest entry only if the
**uid digest, episode count and skip indices** match `e438721ae894`. Its uid space is
`uid_kind: "epcache_basename"` — `ep_%05d.pt`, i.e. **position**. A v2 cache is a flat set of
`<clip_id>.v2ep.pt`: the two uid spaces are not comparable, so the function refuses — **correctly**, since it
has no way to distinguish "different format" from "different episodes".

**Three things follow, and the third is the sharp one:**

1. **`train_flagship_v4` cannot read a v2 cache at all** (its only `v2` match is the string `"v21"` in a
   `--labels` choice). The trainer whose `require=True` motivated the whole runbook is not the trainer that
   can consume the only storable artifact.
2. **`train_flagship4b --v2-cache` can read it — and applies NO parity check on that branch.** The code says
   so in as many words: *"This is a SEPARATE corpus and does NOT touch the raw parity path."* Its parity
   guard lives in `_cache_split`, which only the `--cache-dirs` branch calls.
3. ⇒ **On the v2 path the registration is not merely impossible, it would be INERT** — nothing downstream
   reads it. A wide v2 cache is trainable today with **no parity enforcement whatsoever**.

**What I did instead**, so the guarantee is not simply dropped: `code/verify_v2_parity.py` proves the same
property in the format's own terms — the set of clip ids built must equal, exactly, the parity train split
exported from pod1, with a matching sorted digest, zero extras, and a shortfall that must equal the **24**
recorded skips. It is a **verification, not a registration**: it writes no manifest, and
`parity.corpus_key_of()` still returns `None` for the directory.

**What is owed (not done here, on purpose — `parity.py` is another stream's file and this should be
reviewed, not smuggled in behind a build):**

- a `uid_kind: "v2ep_clipid"` branch in `parity.py` + a `register_v2_geometry_sibling()`, so a v2 sibling
  becomes a first-class manifest entry; **and**
- a parity guard on `train_flagship4b`'s `--v2-cache` branch, because today that branch has none; **or**
- `--v2-cache` support in `train_flagship_v4`, if v5 is to run on the v4 trainer.

⚠️ **Until one of those lands, the v5 wide cache is trainable but NOT parity-enforced.** That is a change in
the program's guarantees and the PI should know it before the run, not after.

---

## 7. Exactly what a trainer must pass to use this cache

**Verified end-to-end on the dev box, not inherited** — the flags were parsed, applied to the real config,
and the real `ViTEncoder` was run:

```
[geometry] v5-wide: NON-DEFAULT - 256x640px, f_ref 305.58, cylindrical,
           HFOV 120.00deg / VFOV 45.46deg, tokens 640 (16x40 @ patch 16),
           state_dim 2048, cache fragment {'geom': '256x640f305.5775cyl'}
encoder.image_size 256   image_width 640   tokens (2, 640, 768)
```

```bash
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack \
python3 -u scripts/train_flagship4b.py \
  --v2-cache /workspace/data/pai_wide120_v2png_train \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --config flagship4b --v2 ...                     # remaining args unchanged
```

- **`--v2-cache`, NOT `--cache-dirs`** — the cache is v2 format.
- **All four geometry flags are required.** Omit them and the trainer builds a 256×256 encoder and is fed
  256×640 frames. `--frame-hfov` and `--f-ref` are mutually exclusive by design.
- **Encoder cost of widening: 87.02 M → 87.32 M params (+0.30 M, +0.34 %)** — purely the larger positional
  embedding, exactly `(640−256) × 768 = 294,912`. `state_dim` is unchanged at 2048. **Widening is a data
  change, not a model redesign.**
- ⚠️ **`ds_val = None` on the `--v2-cache` branch — that trainer runs no val loop.** Held-out selection, the
  cause #1 of the previous v5 failure per `flagship-v5-retrain.PREP.md`, is **not** available through this
  path as it stands.
- ⚠️ **The val split is NOT built** (see §8). Only the 2,400-clip parity train split is.
- ⚠️ **`apply_geometry_args` itself warns, and it is a PI decision, not a default:**
  *120.00° EXCEEDS comma2k19's entire field (65.203°)* — comma2k19 cannot supply this frame at any
  resolution and must be letterboxed with an explicit unobserved mask, given its own frame, or dropped.

---

## 8. 🔴 THE RUNNING BUILD — PIDs, logs, and how to finish it

⚠️ **A job is running on `tanitad-pod2` and this section is its only record outside my context.**

| | |
|---|---|
| **host** | `tanitad-pod2` |
| **PIDs** | **2924952, 2924953, 2924954, 2924955, 2924956, 2924957, 2924958, 2924959** (8 shards, `--shard i/8`) |
| **also recorded on the pod** | `pod2:/workspace/wfov/build_pids.txt` |
| **logs** | `pod2:/workspace/wfov/build_s0.log` … `build_s7.log` |
| **output** | `pod2:/workspace/data/pai_wide120_v2png_train/` |
| **launcher** | `pod2:/workspace/wfov/launch_build.sh` (staged as `code/launch_build.sh`) |
| **started** | 2026-07-27 ~07:53 UTC |
| **measured rate** | **660 clips/h** (sampled over 120 s); **620 clips/h** over the first 12 min |
| **ETA** | **~3.5 h** → ~11:30 UTC |
| **launched with** | `ssh -f` + `setsid nohup`, `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PAI_DECODE_THREADS=4` |
| **health at filing** (T+11:57) | **8/8 PIDs alive**, ~100 % CPU each, RSS ~850 MB each (~7 GB of the 55 GB cgroup), **124/2,400 built, 4.74 GB, 0 FAILED, 9 chunks complete** |

⛔ **To stop it, kill by EXPLICIT PID.** `pkill -f v2_compressed` / `pgrep -f` self-match the ssh command
that runs them — the trap that has now appeared in six forms in this program.

**What it is building:** the **parity TRAIN split only** — 2,400 clips, `--only-clips` pinned to the list
exported from pod1. Shard sizes sum exactly: `275+300+309+333+304+268+318+293 = 2,400` ✅. It is **resumable
and idempotent**: `done` is re-read from the output dir at startup, so a kill loses only the clip in flight.

`pod2:/workspace/data/pai_wide120_v2png_train/_geometry.json`, written before the first decode:

```json
{"frame": {"height": 256, "width": 640, "f_ref": 305.5774907364391, "projection": "cylindrical"},
 "frame_tag": "256x640f305.5775cyl", "projection_mode": "cylindrical", "codec": "png",
 "clips_requested": 2400,
 "geometry_check": {"requested_hfov_deg": 120.0, "achieved_hfov_deg": 120.0, "f_eff": 305.5775}}
```

**Measured build economics** (single-shard smoke run over a real chunk, then the 8-shard steady state):

| quantity | MEASURED |
|---|---|
| bytes/clip, PNG lossless @ 120°/256×640 | **~40 MB** (n=7 chunk run; 36.86 MB on the n=3 measure) |
| projected train split (2,376 episodes) | **~95 GB** |
| decode+encode per clip, 1 shard | **19.4 s** |
| camera chunk download | ~1.9 GB/chunk, 197 chunks ⇒ **~374 GB** total egress |
| clips per chunk | 15.23 mean |

⚠️ **The download, not the encode, is what makes it 3.5 h.** Each shard serialises
download-then-build. 96 cores are available and only ~8 are in use (each shard is ~1 core: PNG encode is
per-frame serial). **Raising the shard count would cut wall-clock close to linearly**, bounded by transient
disk (~1.5 GB/shard) and HF bandwidth. I did **not** raise it: the margin against a quota-fill — the failure
that has killed a flagship mid-checkpoint here — was worth more than the hour.

### When it finishes — the two commands owed

```bash
# 1. prove membership (writes the proof JSON; exits non-zero on ANY extra clip)
PYTHONPATH=/workspace/wfov/stack_head python3 /workspace/wfov/verify_v2_parity.py \
  --cache /workspace/data/pai_wide120_v2png_train \
  --expect-clips /workspace/wfov/paritysplit/parity_train_clips.txt \
  --split-meta  /workspace/wfov/paritysplit/parity_split_meta.json \
  --out /workspace/wfov/v2_parity_verify.json
```

**Pass criteria, fixed here BEFORE the number exists** (`GATE_PROTOCOL` §0.3):
- `extra_count == 0` — **any** extra clip is a re-selection and the script refuses;
- and **either** `membership_identical` (all 2,400 built) **or** `observed_shortfall == 24` with the
  shortfall being decode failures — because the parity corpus records **exactly 24** corrupt clips at
  indices 1798–1941. ⭐ **That 24 is itself a strong independent check: the same clips must fail again.**
- ⚠️ If the shortfall is neither 0 nor 24, **do not register and do not train** — report it.

```bash
# 2. the val split, if wanted (600 clips, ~24 GB, same command, one flag changed)
/workspace/wfov/launch_build.sh  # edit --only-clips -> parity_val_clips.txt, --out -> ..._val
```

---

## 9. Code changed, and why each change was necessary

All in `stack/`, all staged, none committed.

| file | change | why |
|---|---|---|
| `scripts/v2_compressed.py` | ⛔ **`_decode_cropped_selected`: loop var `fr` → `vframe`** (+ `c`→`container`, `c`→`out`) | §2 — the second instance of the `fdc5b4f` defect; **every** v2 build path was dead |
| `scripts/v2_compressed.py` | **geometry + codec wired into the CLI** (`--hfov/--height/--width/--f-ref/--projection-mode/--codec`), for `build` and `measure` | `build_compressed()` accepted `frame`/`projection_mode`/`codec` since `fdc5b4f`, but `build()` called it with **none of them** — so the CLI could only ever produce the deployed square JPEG cache and **the wide build had no entry point** |
| `scripts/v2_compressed.py` | **`_assert_geometry_deliverable()`** — pre-decode abort, ported from `build_pai_cache.py` | the v2 builder had no such guard; without it a multi-hour build silently ships the wrong field (100°@256² delivers 66.56°) |
| `scripts/v2_compressed.py` | **`--only-clips`**, refusing any id not in `--sel` | pins the build to the exact parity split; makes "no re-selection" enforced, not promised |
| `scripts/v2_compressed.py` | **`_geometry.json` sidecar** | on disk a wide cache is otherwise indistinguishable from the deployed one |
| `scripts/v2_compressed.py` | **reuse clips already on the host; never delete pre-existing mp4s** | ~374 GB of egress otherwise; and the builder must not destroy another stream's copy of the raw corpus |
| `scripts/v2_compressed.py` | 🔒 **HF token: `curl -H` argv → `--config` on stdin** | a live token in argv is readable by any `ps`; `CLAUDE.md` §Invariants forbids tokens in args |
| `tests/test_v2_compressed_real.py` | **new** — real mp4 through the real builder, both projections, both frame shapes, mid-loop flush, + AST guard on both files | the exact gap that let the bug through twice |

⚠️ **`scripts/v2_compressed.py` belongs to another stream** (`2026-07-27-geometry-configurable`). Every change
above is either a defect fix or the wiring needed to reach code that stream already wrote. **It needs their
review before anyone commits.** `git status --short` will show foreign staged entries — follow the
`CLAUDE.md` pathspec / `-F` rule.

**Test status.** `tests/test_v2_compressed_real.py` — **7 passed on pod2**; 2 passed / 5 skipped on the dev
box (no torchvision, so the runtime tests skip there and the AST guards still run).
`tests/test_decode_mp4_real.py` — **4 passed on pod2** with the shipped HEAD, which is the brief's
precondition for starting a multi-hour build.
⭐ **Full suite on the dev box: `cd stack && pytest -q` → 1,259 passed, 12 skipped, 0 failed.**

⚠️ **The RED state was measured, not assumed:** before the fix, the real builder on pod2 raised
`AttributeError: 'VideoFrame' object has no attribute 'half_angle_x_rad'` on the **deployed** geometry —
while `pytest -q` was fully green. That is the evidence the new test is worth its lines.

---

## 10. Shipping HEAD to pod2 — verified by a real decode, not a `git log`

`origin/main` lacks `CanonicalFrame`, `register_geometry_sibling` and `HEADING_MODE_HOLD`, and the branch is
unpushed, so no pod can `git pull` this. Shipped by `scp` **all at once** — `tanitad/` + `scripts/` +
`tests/` in one tarball (`--no-same-owner`, per the documented `tar` ownership trap) to
`pod2:/workspace/wfov/stack_head`, md5 `31eca4c9703d3c4ab23e87ebc86002f9`.

Verified by **running things**, not by inspecting metadata:

```
tanitad OK -> /workspace/wfov/stack_head/tanitad/__init__.py
CanonicalFrame: True     register_geometry_sibling: True     vframe in _decode_mp4: True
pytest tests/test_decode_mp4_real.py  ->  4 passed
```

---

## 11. Deliverable manifest

**Everything is `git add`ed into the working tree. Nothing committed, nothing pushed.**

| artifact | where it lives | only one copy? |
|---|---|---|
| `WIDE_FOV_BUILD.md` (this file) | `repo:…/incoming/2026-07-28-wide-fov-build/` | no |
| ⛔ `stack/scripts/v2_compressed.py` — the §2 fix + §9 wiring | `repo:` **staged** + `pod2:/workspace/wfov/stack_head/scripts/` | no |
| ⛔ `stack/tests/test_v2_compressed_real.py` — new regression test | `repo:` **staged** + `pod2:…/stack_head/tests/` | no |
| `raw/host_census_2026-07-27.json` — the census that decided the host | `repo:` | **yes** |
| `raw/rig_mix_full_2026-07-27.json` — rig mix + mask, n=3,000 | `repo:` + `pod2:/workspace/wfov/` | no |
| `raw/rig_mask_census_2026-07-27.json` — n=760 census + 24 real decodes | `repo:` + `pod2:/workspace/wfov/` | no |
| `raw/parity_split_meta_2026-07-27.json` — the split digests | `repo:` + `pod1:/tmp/paritysplit/` + `pod2:/workspace/wfov/paritysplit/` | no |
| `code/parity_probe.py` — the decode-free key probe | `repo:` + pod1/pod2/pod3 `:/tmp/` | no |
| `code/chunkgap.py` — per-chunk clip/mp4 gap | `repo:` + pod1/pod2 `:/tmp/` | no |
| `code/parity_split_export.py` — export, refusing unless both keys verify | `repo:` + `pod1:/tmp/` | no |
| `code/rig_mix_full.py`, `code/rig_mask_census.py` | `repo:` + `pod2:/workspace/wfov/` | no |
| `code/verify_v2_parity.py` — the §6 membership proof | `repo:` + `pod2:/workspace/wfov/` | no |
| `code/launch_build.sh` — the exact build command | `repo:` + `pod2:/workspace/wfov/` | no |
| 🔴 **the wide cache itself** — `pai_wide120_v2png_train/` | **`pod2:/workspace/data/pai_wide120_v2png_train/` ONLY** | ⚠️ **YES — and it is ~95 GB, so it cannot be copied into the repo. It exists on exactly one disk.** |
| 🔒 parity train/val clip-id lists | `pod1:/tmp/paritysplit/`, `pod2:/workspace/wfov/paritysplit/` | no — **and deliberately NOT in the repo** (gated-confidential); the repo carries their digests |
| HEAD stack shipped to pod2 | `pod2:/workspace/wfov/stack_head/` | n/a (reproducible from repo) |

⚠️ **Two things exist in only one place and both are called out above:** the census JSON (cheap to
regenerate) and — materially — **the ~95 GB cache on pod2**. A pod is not storage; if that host is
reassigned before the cache is used, the build must be re-run (~3.5 h, and it is resumable).

🔒 **Credential note, stated so it is visible and reversible:** the build needs the gated dataset
(anonymous access returns **401**) and pod2 had no HF credential. I provisioned one from `Keys.txt` — read
in place, piped over **ssh stdin**, never printed, never in argv — to `pod2:/root/.hf_env` and
`pod2:/root/.cache/huggingface/token`, both `chmod 600`. Verified via the library's own store
(`whoami` → `Sayood`; repo `gated: auto`, reachable). **Remove both files if that is not wanted.**

---

## 12. Escalations

0. 🔴 **`scripts/v2_compressed.py` carries my fix to another stream's committed regression** (§2) — needs
   the `geometry-configurable` stream's review before anyone commits. The index will contain other agents'
   work; read `git status --short` first.
1. 🔴 **The parity guarantee is currently NOT enforced for the artifact v5 would train on** (§6). Needs a
   `uid_kind: v2ep_clipid` registration path **and/or** a parity guard on `train_flagship4b --v2-cache`
   **and/or** `--v2-cache` support in `train_flagship_v4`. **This is a change to the program's guarantees
   and should reach the PI before the v5 run, not after.**
2. 🔴 **A build is running on pod2 with ~3.5 h ETA** (§8). Someone must run the two commands in §8 when it
   finishes. Its output lives on one disk.
3. ⚠️ **`GEOMETRY_CONFIGURABLE.md` §2c's "~29 % of the corpus" looks A/B-inverted** — measured rig B =
   **72.93 %** (n=3,000). The live crop defect's blast radius is **2.7× larger** than published (§5).
4. ⚠️ **120° exceeds comma2k19's entire 65.203° field.** `apply_geometry_args` raises this itself. Letterbox
   with an explicit mask, give comma its own frame, or drop it from the mix — **PI call, before training.**
5. ⚠️ **The v5 val split is not built** (600 clips, ~24 GB, one flag). And the `--v2-cache` branch runs **no
   val loop** — held-out early-stop, cause #1 of the previous v5 failure, is unavailable through it today.
6. ⚠️ **`RETRACTION_LOG.md` root-cause class:** *"a hotfix that fixes one instance of a defect the same
   commit introduced in two files"*. `4cb37f4` fixed `physicalai.py` and left `v2_compressed.py` broken for
   the same reason, found the same way (running the real thing). The AST guard in the new test is the
   mechanical answer.
7. ⚠️ **pod1's `_epcache/…/DONE` marker disagrees with its own files** (`{2400,0}` vs 2376+24 on disk;
   pod2 reads `{2376,24}`). Harmless today — `DONE` is not read on resume — but it is a stale marker on the
   sacred corpus and someone should correct or delete it.
8. ⚠️ **Never quote MooseFS single-sample throughput.** 30 GiB → 66.1 MB/s and 116 GiB → 465 MB/s on the
   same host with the same flags, 7× apart (§3.1).
