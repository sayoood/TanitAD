# Input geometry is now CONFIGURABLE — wide-FOV enablement for v5

**Date:** 2026-07-27 · **Stream:** Architecture & Inference · **Status:** staged, not committed
**Scope:** the *mechanism*. ⛔ **No default changed and no geometry was chosen here.**
Which frame v5 trains on belongs to `…/incoming/2026-07-27-fov-crop-audit/` and the PI.

> **PI directive (verbatim, 2026-07-27):** *"proceed with the adjustment of the setup for 100 or
> 120 degree from the av dataset … we will consider all new trainings with the larger hfov after a
> small validation, the v5 should train with the adapted setup."*

---

## 0. Headline — read this first

| # | finding | class |
|---|---|---|
| **1** | ⭐ **Selection parity SURVIVES a geometry change. It is a RE-CACHE, not a re-selection.** Same episodes, same `ep_%05d.pt` uids, same 24 skip indices, same `T` — only the pixels differ. | **MEASURED** |
| **2** | ⚠️ **But the rebuild is still BLOCKED until a new corpus key is registered.** The parity guard matches on the *directory name*; a re-cropped cache reads as NON-PARITY and `train_flagship_v4` (`require=True`) **refuses to train on it**. A one-line registration path is now provided that *proves* selection survived. | **MEASURED** |
| **3** | ⛔ **A 100° frame is IMPOSSIBLE at 256×256 — it silently delivers 67.1°.** The f-theta crop needs a 1595 px side and clamps at the sensor's 1080 px height, so it **zooms instead of widening**. Widening *requires* more columns. | **MEASURED** |
| **4** | **The rebuild is DECODE-bound, not geometry-bound.** Remap grows 1.26–2.82×, but remap is ~9–12 % of the work: whole-corpus wall-clock grows only **1.00 → 1.23×**. | **MEASURED** |
| **5** | ~~**Storage may not fit.**~~ ⭐ **SOLVED — and it needed no compromise.** The compressed cache puts the **widest** candidate (120°, 256×640) at **17.3 GB lossy / 112.9 GB LOSSLESS**, against today's deployed **349 GB raw**. **Tripling the field of view SHRINKS the corpus 3× with zero pixel loss.** Storage is no longer a constraint on any candidate. | **MEASURED** |
| **6** | **`state_dim` is INVARIANT to input geometry (2048 at every candidate).** Widening is a data + encoder change, not a whole-model redesign. ⭐ **Confirmed identical to the FOV-audit stream's adaptive-4×4 route, and verified end-to-end by the encoder-tokenization stream** (256×640 → 640 tokens → state `[2, 2048]`) — see §8b. | **MEASURED** |
| **7** | ⚠️ **A pre-existing silent-collision hole is now closed:** build params carried `size` but **never `f_ref`**. ⭐ **Independently found by the FOV-audit stream** (`F_REF` 266 vs 133 → identical key); their exact reproduction is now a regression test. | **MEASURED** |
| **8** | ⛔ **A LIVE DEFECT IN TODAY'S TRAINING INPUT, reproduced: the deployed crop pads 11.2 % of rows on rig B and 0 % on rig A** — ⛔ **CORRECTED 2026-07-28 — THE “~29 %” WAS A/B-INVERTED AND THE BLAST RADIUS IS 2.7× LARGER: RIG B IS 72.93 % OF THE CORPUS, NOT ~29 %.** *(The 29.1 % figure is the fraction of clips with cy < 650 — i.e. rig A — and was read as the affected side.)* Measured at n = 3,000 clips, 0 failures: pad rate **rig A 0.0017 % / rig B 8.897 %**, with a 12/12 real-decode subsample agreeing at 0.0056 % / 9.507 %. **A wide *crop* inherits it (120° → 19.9 %); the cylindrical path removes the fabrication entirely** (mask, not replicate-pad) and cuts rig B to **0.69 % at 100°**. ⚠️ **A rig-correlated MASK is still a rig-correlated signal** — the clean fix is a vertical field both rigs fully observe, which is expressible but deliberately NOT chosen. Retraction class C26. | **MEASURED** |
| **9** | ⚠️ **The inherited "~466 GB `/workspace` quota" is FALSIFIED as a hard cap.** pod3 currently holds **487 GB** and writes at 415 MB/s. The true ceiling is a RunPod console fact, not discoverable from inside the pod — and after finding #5 it no longer gates anything. | **MEASURED** |

**Validation:** fidelity + deliberate-failure suite **90/90 pass, BIT-EXACT**
(`fidelity_2026-07-27.json`). New regression suite **53/53**.
`pytest -q`: `stack/` **1200 passed, 7 skipped → 1253 passed, 7 skipped** (+53, the new file);
`taniteval/` **559 passed → 559 passed** (untouched). *(Final counts in §8.)*

---

## 1. ⭐ Selection vs cache — the verdict, with its code evidence

**VERDICT: changing the crop/resolution invalidates ONLY the feature/episode CACHE. The episode
SELECTION is untouched. The coordinator's inherited reading is CORRECT — and it is now MEASURED,
not inferred.** One material addition follows in §1.4.

Raw evidence: `selection_verdict_2026-07-27.json`. Executed against the real local PhysicalAI root
(500 clips, non-parity key `14231cd29c74` — used to *exercise the chain*, never to build a corpus).

### 1.1 Nothing that decides WHICH episodes exist can see the geometry

MEASURED by introspection, printed into the artifact:

```
discover_r0_clips(root: str | Path) -> list[dict]
split_clips(clips: list[dict], val_frac: float = 0.2, seed: int = 0) -> tuple[...]
```

Neither takes `size`, `frame`, `f_ref`, `projection` or `projection_mode`. The ordered source list
is fully determined before any geometry is consulted. `build_pai_cache.py` calls them in exactly
that order, then hands the result to `build_episodes_cached`.

### 1.2 Episode identity is a POSITION, not a pixel

`epcache.build_episodes_cached` writes `ep_%05d.pt` where the index is the position of the clip in
the ordered source list, and `skip_%05d` when `build_one` raises. `parity.py` records
`uid_kind: "epcache_basename"` — identity **is** that basename. Geometry can change what is inside
the file; it cannot change which slots exist.

MEASURED end-to-end (`identity_under_geometry_change`), building the same 10 sources at 256×256 and
at 256×640 with a builder that fails at fixed indices:

| | canonical 256² | wide 256×640 |
|---|---|---|
| episode uids | identical set | identical set |
| skip indices | `[3, 7]` | `[3, 7]` |
| episode count | 10 | 10 |
| `T` per episode | 5 | 5 |
| frame H×W | 256×256 | **256×640** ← the only difference |

### 1.3 ⭐ The strongest evidence: the canonical path reproduced a REAL on-disk key

Running the new code with the canonical frame over the real local clip list produced cache key
**`14231cd29c74`** — which is the *actual name of the existing local cache directory*
(`_epcache/physicalai-train-14231cd29c74`), built months ago by the pre-refactor code.
`geometry_build_params(canonical) == {}` is therefore not a claim; it round-tripped against a real
artifact. The eight candidate geometries produced eight distinct keys, all re-derivable:

| candidate | frame tag | params fragment | train cache key |
|---|---|---|---|
| deployed | `256x256f266pin` | `{}` | **`14231cd29c74`** (= the real dir) |
| 100° 256² | `256x256f107.4048pin` | `{"geom": …}` | `2ed046e007ac` |
| 100° 256×640 | `256x640f268.5119pin` | `{"geom": …}` | `c5f922dde3f0` |
| 100° 256×640 cyl | `256x640f366.693cyl` | `{"geom": …, "projection_mode": "cylindrical"}` | `d4adc870c25d` |

### 1.4 ⚠️ The addition the "re-cache" framing hides — and its fix

`parity.corpus_key_of()` **substring-matches registered keys against the directory path**. A
re-cropped build lives in `physicalai-train-<newkey>`, so:

- `corpus_key_of(...)` → `None` → the dir reads **NON-PARITY**;
- `assert_parity_corpus(..., require=True)` → **`ParityViolation`, refuses to train**;
- `train_flagship_v4` has always hard-required parity. **v5 cannot start on a re-cropped cache
  until its key is registered.**

MEASURED: `parity_guard.recropped_dir_is_recognised == False`.

**Provided fix — `parity.register_geometry_sibling()`** (`stack/tanitad/data/parity.py`). It mints
a manifest entry for the new key **only if** the observed uid digest, episode count *and* skip
indices match the parity entry exactly. A re-cache passes; a re-selection is refused with
*"geometry may change PIXELS, never MEMBERSHIP"*. The registration is therefore itself the proof
that §1's verdict held on the real rebuild. Both branches are tested.

**Operational consequence for the v5 schedule:** rebuild → `register_geometry_sibling` →
commit the `parity_manifest.json` diff → train. Step 2 is seconds; without it the trainer refuses.

### 1.5 What is NOT claimed

- The 24 skips are **decode/IO failures** (corrupt clips at indices 1798–1941). Geometry cannot
  reach them: the crop/resample is total on any successfully decoded tensor, and the skip is
  written by `except Exception` around `build_one`. The *mechanism* is MEASURED (§1.2); that the
  same 24 real clips fail again is **INFERRED** and will be confirmed by the digest check in
  `register_geometry_sibling` on the real rebuild. **If it ever differs, that function refuses —
  loudly — rather than letting a changed corpus through.**
- The 2376-episode parity build itself was **not** re-run here (gated data, pods busy).

---

## 2. Rebuild cost — MEASURED

Raw: `rebuild_cost_2026-07-27.json`. 6 real clips (3 rig A, 3 rig B, all with per-clip
calibration), 48 frames each, native 1080×1920. **No GPU is involved anywhere**, so the WDDM
host-RAM-spill artefact cannot inflate any figure on this page.

**Two frame counts, and conflating them understates the build by ~3×** — MEASURED on 5 real clips:
a clip is **605 frames** (20.13–20.17 s at 30.0 fps), *all decoded*, then resampled to **199 stored**
frames at 10 Hz. Storage cross-checked exactly: a real cached episode is **117.384 MB** =
199 × 9 × 256 × 256 bytes.

| candidate | HFOV asked → **got** | tokens @p16 | native crop | observed | remap ×dep | peak float/batch | s/ep | **h @16 workers** | **cache GB** |
|---|---|---|---|---|---|---|---|---|---|
| deployed 256² | 51.4 → **51.4** | 256 | 830×830 | 1.000 | 1.00× | 199 MB | 9.3 | **0.48** | **349** |
| 100° 256² pin | 100 → ⛔ **67.1** | 256 | 1080×1080 *(clamped)* | 1.000 | 2.82× | 336 MB | 11.4 | 0.59 | 349 |
| 100° 256×640 pin | 100 → **100.0** | 640 | 823×1594 | 1.000 | 2.00× | 378 MB | 10.5 | 0.54 | 873 |
| 100° 256×640 **cyl** | 100 → **100.0** | 640 | full-frame resample | 0.996 | 1.40× | 597 MB | 9.8 | 0.50 | 873 |
| 120° 256×640 pin | 120 → **120.0** | 640 | 1080×1899 *(V clamped)* | 1.000 | 2.64× | 591 MB | 11.2 | 0.58 | 873 |
| 120° 256×640 **cyl** | 120 → **120.0** | 640 | full-frame resample | 0.954 | 1.38× | 597 MB | 9.7 | 0.50 | 873 |
| 120° 384×960 **cyl** | 120 → **120.0** | 1440 | full-frame resample | 0.954 | 1.74× | 597 MB | 10.1 | 0.52 | **1965** |
| 120° 384² **cyl** | 120 → **120.0** | 576 | full-frame resample | **0.619** | 1.26× | 597 MB | 9.6 | 0.49 | 786 |

**Stated extrapolation.** `work/ep = (decode s/frame + remap s/frame) × 605`;
`storage/ep = bytes_per_stacked_frame × 199`; whole corpus = `× (2376 + 600)`, ÷ workers.
Decode measured **0.0085–0.0134 s/frame** across two runs on this box (the spread is contention
from concurrent test suites — the ratios, not the absolute wall-clock, are the portable numbers).
A pod should substitute its own decode throughput; the **remap ×** column is machine-independent.

**Reading the table:**

- ⛔ **100° at 256×256 is a trap.** Requested 100°, delivered **67.1°** — a 32.9° shortfall, because
  the crop clamps at the sensor's height and *zooms*. Pinned as a permanent regression test, and
  `build_pai_cache.py` now **aborts** on it before a multi-hour decode instead of shipping it.
  This is the "height-bound → silent zoom" failure the D-016 R1 notes warned about, now armed.
- **120° at 256×640 pinhole** also clamps *vertically* (crop 1080 of a wanted ~1119). Horizontal
  120° is delivered; the vertical field is truncated ~3 %.
- **Wall-clock is a non-issue.** Worst case 0.59 h vs 0.48 h at 16 workers (**1.23×**).
- ⚠️ **Storage is the issue.** `/workspace` is under a **~466 GB MooseFS quota that `df` does not
  show** (project trap list). The deployed 349 GB already sits close to it; **873 GB does not fit**
  and 1965 GB is far out. **This is a scheduling/провision input the PI needs, and it is the
  binding constraint — not compute.** Mitigations, not chosen here: the JPEG-compressed v2 cache
  format (`v2_compressed.py`), a bigger volume, or building train/val on separate pods.
- **Peak RAM per build worker rises 199 MB → ~600 MB** (`PAI_DECODE_BATCH=24`). COMPUTED from the
  measured crop, not sampled: the crop path upcasts only the crop, the cylindrical path
  `grid_sample`s the full native frame. ⚠️ Sampled RSS deltas were **not quotable** (7–182 MB for
  arithmetically similar work — allocator caching) and are labelled as such in the JSON.
- **`observed_frac`** is how much of the output frame the sensor actually covers. `0.619` for a
  square 120° frame is 38 % black — direct evidence that *square + wide wastes pixels*. Offered as
  an instrument; **the audit decides**.

---

## 2b. ⭐ STORAGE IS SOLVED — and lossless is enough

Raw: `jpeg_cache_2026-07-27.json` (3 real clips × 4 geometries × 6 codecs, PIL backend)
+ a torchvision cross-check on pod3 at the deployed geometry.

**The v2 compressed cache is now non-square-capable** (it was the blocker I flagged): the payload
carries `image_h`/`image_w`/`frame`/`projection_mode`/`codec`, the manifest is bumped to **v3** with
per-clip `image_h`/`image_w`, and a payload predating the change is square by construction so the
scalar fallback is *exact, not a guess*. A **lossless `codec="png"`** path was added alongside JPEG —
same container, same loader (the decoder is chosen from the stored codec), so the two are A/B-able
without a format fork.

**Corpus size, train + val (2976 episodes), MEASURED:**

| geometry | raw (today's format) | **PNG lossless** | JPEG q95 | JPEG q90 | JPEG q75 |
|---|---:|---:|---:|---:|---:|
| deployed 256² @ 51.4° | 349 GB | **44.8 GB** | 11.1 | 7.7 | 4.6 |
| 100° 256×640 cyl | 873 GB | **117.4 GB** | 25.9 | 17.7 | 10.2 |
| **120° 256×640 cyl** | 873 GB | **112.9 GB** | 25.3 | **17.3** | 10.0 |
| 120° 384×960 cyl | 1965 GB | **221.9 GB** | 47.7 | 32.4 | 18.6 |

⭐ **The decisive line: 120° at 256×640, LOSSLESS, is 112.9 GB — a third of the 349 GB we occupy
today at 51.4°.** The wide rebuild does not need a provisioning decision, does not need lossy
compression, and does not need the fidelity argument at all. **The provisioning request to the PI
is withdrawn.**

**Where the win comes from — decomposed so it cannot be over-claimed.** Two independent factors:

1. **Un-stacking: ~2.97×, lossless, and nothing to do with the codec.** The raw epcache stores the
   D-015 3-frame stack `[T, 9, H, W]`, i.e. **every frame three times**. v2 stores `[T+2, 3, H, W]`
   and re-stacks at load (bit-identically — already validated in-repo).
2. **The codec on top:** PNG ~2.6×, JPEG q90 ~15×, on the unstacked frames.

**The cost, on the axis it actually lands.** ⚠️ **Correcting a framing in the brief:** the
*rebuild* is mp4-decode-bound, but the codec's cost is **not** paid there — it is paid by the
**training dataloader**, once per window served, forever. Both measured separately
(torchvision, pod3, 256²):

| | encode (build, once) | decode (dataloader, every window) | fidelity |
|---|---:|---:|---|
| JPEG q90 | **1.27 ms/frame** | **2.96 ms/frame** | PSNR 39.6 dB, max abs err 68/255 |
| JPEG q95 | 2.69 ms/frame | 2.68 ms/frame | PSNR 41.8 dB, max abs err 68/255 |
| **PNG lossless** | **70.3 ms/frame** | **1.60 ms/frame** | **max abs err 0 — bit-exact** |

- **PNG decodes FASTER than JPEG** (1.60 vs 2.96 ms/frame). A trainer window (`window=8`,
  `n_stack=3` → 10 raw frames) costs **~16 ms** on one worker; with the standard multi-worker
  prefetch this is far below the step time. **The dataloader argument against lossless does not
  survive measurement.**
- PNG's cost is **build-side**: 70 ms/frame × 201 frames ≈ **14 s/episode**, roughly **3× the
  5.1 s mp4 decode**. Whole corpus ≈ **1.2 h at 16 workers** (vs 0.48 h raw). A one-off.
- **WebP lossless is ruled out on build cost**: ~280–324 ms/frame (~60 s/episode) for ~10 % better
  compression than PNG. Measured, not assumed.
- JPEG's **max abs error is 68/255 on a real frame even at q95** — a large single-pixel excursion.
  Since lossless already fits, **there is no reason to accept it**, and no fidelity/ADE experiment
  is needed to justify the choice.

⛔ **Not chosen here.** Codec and quality remain the PI's/audit's call; the point is that **every
option now fits**, so storage stops constraining the geometry decision.

---

## 2c. ⛔ A LIVE DEFECT IN TODAY'S TRAINING INPUT — rig-asymmetric fabricated pixels

Raw: `route_and_rig_2026-07-27.json`. Independently reported by the FOV-audit stream at 11.3 %;
**reproduced here at 11.21 % on 10 real clips** (rig A 0.00 %, rig B 11.21 %, `asymmetry_confirmed`).

The principal-point-centred crop is the D-016 R1 *fix* for the two-rig split — but for rig B
(cy ≈ 755) the box spills ~90 px past the native bottom edge and those rows are **replicate-padded**.
Rig A (cy ≈ 543) spills nothing. ⛔ **CORRECTED 2026-07-28 — A/B-INVERTED; rig B is 72.93 %, so this is 2.7× larger than written (n = 3,000, pad rate rig A 0.0017 % / rig B 8.897 %).** So **72.93 % of the corpus carries fabricated rows that correlate
perfectly with rig**, and replicate padding *looks like real road* — exactly the shortcut this model
eats (`v0` alone moved the imagined decode ×93.7).

| frame | rig A | rig B | fabricated pixels? |
|---|---:|---:|---|
| **deployed 256² (today)** | 0.00 % | **10.94 %** padded rows | **YES — replicate-padded** |
| 100° 256×640 *crop* | 0.00 % | 10.54 % padded rows | **YES — inherits it** |
| 120° 256×640 *crop* | 0.28 % | **19.91 %** padded rows | **YES — nearly doubles it** |
| **100° 256×640 cylindrical** | 0.00 % | **0.69 %** unobserved | **NO — explicit mask** |
| **120° 256×640 cylindrical** | 0.00 % | 8.91 % unobserved | **NO — explicit mask** |

**Two conclusions, and only the first is mine to draw:**

1. **MECHANISM (mine):** a wide rebuild via the *crop* path would bake this artefact in **at larger
   scale**; the cylindrical path removes the *fabrication* entirely — unobserved pixels become an
   explicit mask with a reported `observed_frac` — and at 100° removes almost all of the asymmetry
   too. Both the defect and the cylindrical behaviour are now pinned as permanent tests
   (`test_the_DEPLOYED_crop_is_rig_ASYMMETRIC_and_fabricates_pixels`,
   `test_the_CYLINDRICAL_path_removes_the_FABRICATED_pixel_asymmetry`).
2. **WHETHER IT MATTERS TO ADE (not mine):** the FOV audit owns that. ⚠️ But note it is **not only a
   v5 question** — it is in every number the program has produced since D-016 R1, so it belongs in
   the retraction/known-limitations record regardless of which geometry wins.

⚠️ **A mask is not free either.** A rig-correlated *black* region is still a rig-correlated signal.
The honest ranking is: fabricated-and-invisible (today) → masked-and-visible (cylindrical) →
absent (a frame whose vertical extent both rigs fully observe). The third is achievable — it is a
vertical-FOV choice — and I have made it expressible, not chosen.

---

## 3. The full list of geometry / square assumptions found

Every hit for `grid_hw`, `image_size`, `**0.5`, `sqrt`, and token-grid reshapes.

### 3.1 Fixed — the flagship training path is fully non-square

| location | assumption | resolution |
|---|---|---|
| `models/encoder.py:44-46` | `assert image_size % patch_size`; single `grid_hw`; `n_tokens = grid_hw**2` | `grid_h`/`grid_w`/`grid_shape`; `n_tokens = grid_h*grid_w`; **`grid_hw` now RAISES on a non-square grid** rather than lying |
| `models/encoder.py` forward | no input-shape check | input must match the declared geometry, else `ValueError` |
| `models/encoder.py` `pos` | checkpoint-shaped `[1, 256, D]`, no transfer path | `resize_pos_embed()` / `adapt_pos_embed_()` (§4b) |
| `config.EncoderConfig` | one `image_size` | `+ image_width: int\|None = None`, `image_hw()`, `token_grid()`, `is_square` |
| `config.ReadoutConfig` | square `grid` only | `+ grid_w: int\|None = None` |
| `config.StackConfig` | no geometry field | `+ geometry: dict\|None = None` |
| `models/readout.py:24-34` | `hw = int(n_tokens**0.5)`; `assert hw*hw == n_tokens`; square reshape | `token_grid=(rows, cols)` + `grid_w`; **`out_dim` stays `grid*grid_w*d_readout`** — the geometry firewall |
| `models/imagination.py` `sector_mask` | `vis[b, g, g]` | accepts `(rows, cols)` |
| `models/imagination.py` `advect` | square reshape + single `grid_hw-1` normaliser | per-axis reshape and per-axis normalisation (one cell of dx stays one cell) |
| `models/imagination.py` `ImaginationField` | scalar `grid_hw` | scalar **or** `(rows, cols)` |
| `models/fourbrain.py:412,456` | passes `n_tokens` / `grid_hw` | passes `grid_shape` and `readout.grid_w` |
| `train/train_worldmodel.py:342,412` | `sector_mask(..., encoder.grid_hw)` | `encoder.grid_shape` |
| `scripts/finetune_traj.py:263` | same | `encoder.grid_shape` |
| `refs/refb.py:381` | square readout | `token_grid=` + `grid_w=` |
| `models/dynamics_encoder.py` | `DynEncConfig.image_size` only; square readout | `+ image_width`; readout gets `token_grid` |
| `data/calib.py` (~10 fns) | `size:int=256, f_ref=F_REF` threaded as defaults | `CanonicalFrame` + `as_frame()`; every fn keeps its old signature with `frame=` optional |
| `data/calib.py` `ftheta_crop_*` | square crop only | `ftheta_crop_size_hw` / `ftheta_crop_box_hw` (rectangle); square path bit-exact |
| `data/calib.py` `focal_crop_resize` | square crop | rectangular + `last_clamped` |
| `data/calib.py` `pinhole_rectify*` | square canvas | frame-aware, incl. cylindrical rays |
| `data/physicalai.py` `_decode_mp4` / `build_episode` | `size` scalar | `frame=` + `projection_mode=` |
| `data/comma2k19.py` `_decode_video` / `build_episode` | `size` scalar | `frame=` + `geometry_mode=` |
| `scripts/build_pai_cache.py` | hardcoded params; canonical-only f_eff assert | geometry flags; **generalised assert that the requested field is DELIVERED** |
| `scripts/train_flagship4b.py` | no geometry input | geometry flags + consistency guard |
| **cache `params`** | carried `size`, **never `f_ref`** — silent collision | `geometry_params()`; empty for canonical, keyed otherwise |

### 3.2 Found and deliberately NOT fixed — with reasons

| location | assumption | why left, and what it costs |
|---|---|---|
| `refs/refc.py:391,594-608,643` | `image_size % 32 != 0 → raise`; a **duplicate** `advect`/`ImaginationField` with square `grid_hw` | REF-C is a **reference arm**, not the v5 flagship path. Its CNN would tolerate a rectangle but its config and its private `advect` copy would not. **Cost to fix ≈ the `imagination.py` change, ~30 lines.** Flag: REF-C cannot run wide until then. |
| `replay/arms.py:342` | `ToyTokenizer` raises `"n_tokens must be square"` | Test/demo tokenizer only, explicitly "NOT a DINO substitute". Real `DinoTokenizer` is unaffected. |
| `data/v2_dataset.py:360` | `shape = (T, 3*n_stack, S, S)` — one scalar `image_size` per clip | The v2 **compressed-cache format itself** stores a scalar. Fixing needs a format/manifest version bump (H *and* W per clip). ⚠️ **This blocks the JPEG-cache mitigation for the 873 GB storage problem** — see §2. Flagged for the storage decision. |
| `lake/schema.py:230,312` · `lake/proof.py:70` | `image_size = frames.shape[-1]` (width only) | Lake byte-proof records/compares one scalar; on a wide frame it would record `640` and compare against `256`. Not on the v5 training path; needs an `image_h`/`image_w` schema bump. |
| `CORPUS_META` × 4 (`comma2k19`, `physicalai`, `cosmos_drive`, `l2d`) | `"image_size": 256, "f_eff_px": 266.0` literals | The **I7 task-identity fingerprint** — deliberately a frozen contract. Changing it re-keys probe compatibility across corpora. Must be updated *as a declared change* alongside the default flip, not silently. **This is the single most likely place for a stale default to survive.** |
| `data/metadrive_frontcam.py`, `data/l2d.py` | square `[T, 3n, S, S]` builders | Non-flagship corpora; same one-line `frame=` treatment applies when needed. |
| `eval/gates.py`, `eval/spectral.py`, `models/sigreg.py` | `**0.5` / `sqrt` | **False positives** — statistics (std-err, matrix blocks), no geometry. |

### 3.3 Note for the encoder/tokenization sibling (`2026-07-27-encoder-tokenization/`)

- The non-square path introduces **no** assumption that blocks foveated or non-uniform tokenization.
  `ViTEncoder` keeps a **uniform** `Conv2d` patchifier and a **learned per-token** positional
  embedding sized `n_tokens` — no factorised or separable 2-D encoding was introduced, so a
  variable-density token set remains open. The one constraint added: `H % patch == 0` and
  `W % patch == 0`, checked independently per axis.
- **`SpatialGridReadout` is the geometry firewall** — that stream verified it end-to-end
  (256×640 → 640 tokens → state `[2, 2048]`), matching this stream's measurement. It is also the
  piece a Perceiver-style latent resampler would replace: `state_dim` is already
  geometry-invariant, so that swap moves nothing downstream.
  ⚠️ **The rank-16 rationale is NOT cited here.** It was struck from the PREP card's VALIDATED table
  (every rung an unpaired point estimate; the "peak at k=16" is **+0.00085 [−0.02204, +0.02299]**,
  a CI ~27× wider than the effect, and the image-only ladder is flat to 5 d.p.). **Nothing in this
  change set depends on it**, and an earlier draft of this section referenced it as a possible
  motivation — that reference is retracted. A latent-resampler swap must be justified on its own
  measurement.
- **Tiling rule, re-derived:** exact pooling needs `token_rows % grid == 0` and
  `token_cols % grid_w == 0`, i.e. at the deployed `patch 16 × grid 4` a **multiple of 64** per axis.
  Two corrections, both measured and pinned in
  `test_the_multiple_of_64_tiling_rule_and_its_TWO_corrections`: **448 DOES satisfy it**
  (448 = 7×64 → 28 cols, 28 % 4 = 0), and **`state_dim` does not break when it fails** — the
  converged readout falls back to adaptive pooling and still returns 2048. The cost of a
  non-multiple width (e.g. 672 → 42 cols) is **uneven pooling bins**, not a shape failure.
  `assert_geometry_consistent` now prints a warning for it.
- **I built the cylindrical projection; I did not make it the tokenization grid.** `cylindrical_rays()`
  exposes the per-output-pixel ray directions, which is exactly what a projection-aware token grid
  needs — measure rather than re-derive.

---

## 4. The cylindrical / equidistant-azimuth projection

`calib.cylindrical_rays` · `ftheta_project_rays` · `cylindrical_grid` · `cylindrical_rectify` ·
`projection_density_report`. Available for `pinhole_rectify` too (rectilinear corpora).

**Definition.** Output column ↦ azimuth `phi = (u − (W−1)/2)/f_ref`, ray `(sin φ, y_n, cos φ)` with
`y_n = (v − (H−1)/2)/f_ref` — equidistant in azimuth, pinhole per column vertically. Rays are
pushed through the clip's real f-theta polynomial and `grid_sample`d back.

**Rationale, COMPUTED from the definitions (not assumed), reproduced by the test suite:**

| half-angle | cumulative radius `tan t / t` | local px/radian at edge `1/cos²t` | cylindrical |
|---|---|---|---|
| 25.70° (today) | 1.0730 | 1.2316 | 1.0 |
| 50.00° (100° frame) | 1.3656 | **2.4203** | 1.0 |
| 60.00° (120° frame) | 1.6540 | **4.0000** | 1.0 |

A pinhole canvas at a 50° half-angle packs **2.42× more pixels per radian at the edge than at the
centre** — the new pixels the PI is buying would land disproportionately on smeared periphery.
Cylindrical spends them uniformly and keeps verticals straight. ⚠️ **This is a geometric fact, not
a claim that cylindrical wins.** That is the audit's experiment; measured here only that it is
**cheaper** (remap 1.38–1.40× vs 2.00–2.82× for the wide crop).

### 4a. ⚠️ Provenance — there is exactly ONE cylindrical implementation, and it is this one

A note reached me that *"`calib.py` already has the option — if you added a second one, reconcile
them."* **Checked, not assumed:** `git show HEAD:stack/tanitad/data/calib.py | grep -ci cylindric`
returns **0**. Cylindrical did not exist before this change set; the encoder-tokenization stream
measured its 448–512-token result **against this (staged, uncommitted) implementation**. **There is
no second implementation and nothing to reconcile** — the drift hazard does not apply here.

That stream's result is a genuine argument for the projection and it is theirs, not mine:
cylindrical reaches ~96.5–110.3° at **448–512 tokens**, where pinhole needs **640** for ~100.5°.
It also relieves storage by ~20 % versus 640 wide (§2b), though after §2b storage no longer binds.

**🔒 The two-rig fix is preserved BY CONSTRUCTION, not by care.** The output-centre ray is the
boresight `(0,0,1)`, and `ftheta_project_ray` maps the boresight to exactly the clip's own
`(cx, cy)`. Rig A (cy≈543) and rig B (cy≈755) therefore both land the optical axis at the output
centre — the same guarantee `ftheta_crop_resize(center="principal")` gives. Tested for both rigs to
< 1 px. A corpus-median intrinsic is **REFUSED** (`require_per_clip=True`): unlike a crop there is
no meaningful "geometric-centre" fallback for a full ray fan, and its `cy` is a rig-B value.
The rectangular crop path also keeps the fix — tested at the canonical *and* a 100° wide frame.

### 4b. Warm-starting a checkpoint into a new geometry

⭐ **The positional embedding is the ONE checkpoint-shaped tensor a geometry change touches.**
A 16×16-trained checkpoint cannot `load_state_dict(strict=True)` into a 16×40 encoder at all —
verified, it raises. `encoder.resize_pos_embed()` (bicubic, the standard ViT resolution-transfer
recipe) and `encoder.adapt_pos_embed_(state, encoder)` close that; a same-geometry load is a
byte-identical no-op. Everything else in the encoder is shape-agnostic (strided `Conv2d`,
attention/MLP over an arbitrary token count, `LayerNorm` over `d_model`).

⚠️ **DECLARED BIAS, one-directional** — carried in the docstring and repeated here because it will
decide how the PI's "small validation" is read: weights trained at 51.4°/256 px evaluated at
another geometry sit under a train/test shape shift **that can only hurt the new shape**. A
wider/higher-res arm that **wins** through a resample is strong evidence; one that merely **ties**
is weak evidence and must not be read as *"resolution does not help"* — it may be reading the
handicap. **Any adopted geometry should be RETRAINED, not resampled.**

### 4c. ⚠️ Coordination — the FOV-audit stream's local shim is superseded

`…/2026-07-27-fov-crop-audit/scripts/shape_shim.py` monkey-patches exactly the three call sites
this change fixes properly (`ViTEncoder.__init__` grid, the `pos` tensor, `SpatialGridReadout`'s
square assert + pool). That was the right call for measuring against frozen v1 weights **without**
touching `stack/`. Now that the in-repo path exists, **the shim should be retired once that stream's
current measurements land** — two implementations of the same relaxation is exactly how the
geometries drift apart. One difference worth noting: the shim uses an **adaptive** pool to (4,4);
`SpatialGridReadout` uses an **exact** `AvgPool2d((th//grid, tw//grid))` and asserts divisibility,
so it refuses a grid it cannot pool evenly instead of silently resampling.

---

## 5. Per-corpus policy — all three options are expressible, none is chosen

⛔ **comma2k19's entire horizontal field is `2·atan(582/910) = 65.2027°`**
(`calib.COMMA2K19_MAX_HFOV_DEG`, now a constant so it is quotable from code, with a test).
It cannot supply 100–120° at any resolution, focal or projection.

| option | how it is expressed | status |
|---|---|---|
| **(a) per-corpus geometry** — PhysicalAI wide, comma at its own 65.2° | two `CanonicalFrame`s → two cache keys → `build_episode(..., frame=…)` per corpus | expressible, tested |
| **(b) letterbox** — comma on the wide frame, periphery **explicitly masked** | `comma2k19` `geometry_mode="rectify"` → `pinhole_rectify`, which reports `last_observed_frac` and an honest black band | expressible, tested. ⚠️ **Known hazard, not mine to weigh:** a constant black border is a **corpus-identifiable shortcut**, and this model demonstrably eats shortcuts (`v0` alone moves the imagined decode ×93.7 — v5 PREP item 7). The mask is *honest*, which is not the same as *harmless*. |
| **(c) drop comma** | trainer/mix decision; no code needed | available |

⚠️ Under the default `focal_crop` mode a frame wider than the sensor does **not** widen the field —
the crop clamps and the image zooms. `focal_crop_resize.last_clamped` now records this, and the
test asserts both behaviours so the difference is not a matter of opinion. The CLI prints a loud
warning whenever a frame exceeds comma's ceiling.

---

## 6. ⭐ Exactly what command changes the geometry

**One flag set, shared by the cache builder and the trainer** (`tanitad.geometry.add_geometry_args`).
All defaults reproduce the deployed frame; nothing changes until a flag is passed.

```bash
# 1) rebuild the corpus at the chosen geometry (example: 120 deg, 256x640, cylindrical)
PYTHONPATH=/workspace/TanitAD/stack python scripts/build_pai_cache.py \
    --root /workspace/data/physicalai \
    --frame-h 256 --frame-w 640 --frame-hfov 120 \
    --projection cylindrical --projection-mode cylindrical
#   -> aborts BEFORE the decode if the sensor cannot deliver the requested field
#   -> writes _epcache/physicalai-train-<newkey>

# 2) register the new key as a GEOMETRY SIBLING (proves selection parity survived)
python - <<'PY'
from tanitad.data import parity
from tanitad.data.calib import CanonicalFrame
ent = parity.register_geometry_sibling(
    "/workspace/data/physicalai/_epcache/physicalai-train-<newkey>",
    new_key="physicalai-train-<newkey>",
    geometry=CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical").to_dict())
PY
#   -> REFUSES unless uid digest + count + skip indices match e438721ae894 exactly
#   -> add the entry to stack/tanitad/data/parity_manifest.json and COMMIT the diff

# 3) train v5 on it — the SAME flags, or the trainer refuses a half-applied change
PYTHONPATH=/workspace/TanitAD/stack python scripts/train_flagship4b.py \
    --config flagship4b --out /workspace/models/v5 ... \
    --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical
```

Every run prints one provenance line:

```
[geometry] train_flagship4b/flagship4b: NON-DEFAULT - 256x640px, f_ref 305.58, cylindrical,
  HFOV 120.00deg / VFOV 45.46deg, tokens 640 (16x40 @ patch 16), state_dim 2048,
  cache fragment {'geom': '256x640f305.5775cyl'}
[geometry] WARNING: this frame (120.00deg) EXCEEDS comma2k19's entire field (65.203deg) ...
```

**To flip the DEFAULT** (the separate, declared change after the audit + PI validation): call
`geometry.apply_frame(cfg, frame)` inside `flagship4b_config()`. That is the *only* sanctioned
mutation — it writes `cfg.geometry` **and** `cfg.encoder.image_size`/`image_width` together, and
`assert_geometry_consistent()` refuses any config where only one of them moved.
⛔ **I did not make that change.**

---

## 7. Both-directions validation

**Direction 1 — fidelity.** `measure_fidelity.py` imports the pre-refactor `calib.py` from
`git show HEAD:` as a standalone module and compares outputs. **Tolerance claimed: BIT-EXACT**
(`torch.equal` on uint8 output; exact equality on integer crop boxes and float `f_eff`) — justified
because the canonical branches evaluate the *same arithmetic expressions*, not merely equivalent
ones. Anything weaker would mean "no default changed" is false.

Covered: both rigs + the median fallback; sizes 256/224/128; both centerings; native and
half-resolution decode; `focal_crop_resize`, `pinhole_rectify`, `ftheta_undistort`,
`ftheta_feff_report`, `pinhole_geometry_report`; and the explicit-`CANONICAL_256`-frame spelling of
each. **90 checks, 0 failures.**

**Direction 2 — deliberate failure.** 11 inputs that MUST be rejected, all verified to raise: a
frame *and* non-default legacy scalars together; an unknown projection; a degenerate frame; a
non-positive `f_ref`; `frame.size` on a non-square frame; `cylindrical_rectify` on the corpus-median
intrinsic; an encoder input mismatching the declared geometry; `grid_hw` on a non-square grid; a
frame not divisible by the patch size; and a **half-applied geometry in both directions**.

---

## 8. Test counts — before and after

| suite | BEFORE (captured before the first edit) | AFTER |
|---|---|---|
| `stack/` | **1200 passed, 7 skipped** (139.4 s) | **1253 passed, 7 skipped** (103.9 s) — +53, all from the new file |
| `taniteval/` | **559 passed** (115.4 s) | **559 passed** (69.7 s) — untouched |

No pre-existing test changed behaviour, was modified, or was skipped to make room.

---

## 8b. ⭐ THE ROUTE CONFLICT — resolved by measurement, no rework needed

**The FOV-audit stream's claim, verbatim:** *"a non-square input need NOT change `state_dim`.
Adaptive-4×4 pooling keeps 2048 and leaves every downstream head loadable (measured; bit-identical
at the deployed shape). Only `encoder.pos` breaks. The `grid_w` route a sibling stream is building
is the expensive one."*

**Verdict: (a) — the two routes are the SAME operation, and we agree.** Their conclusion is correct.
Their reading of my implementation is not: `grid_w` is an **opt-in knob that is `None` by default**,
not the route. Re-derived against this implementation (`route_and_rig_2026-07-27.json`):

| token grid | `AvgPool2d((gh/4, gw/4))` vs `AdaptiveAvgPool2d((4,4))` | max abs diff |
|---|---|---:|
| 16×16 (**deployed**) | **bit-identical** | 0.0 |
| 16×40 (256×640) | **bit-identical** | 0.0 |
| 24×60 (384×960) | equal to float32 summation noise | 3.0e-08 |
| 24×24 (384×384) | equal to float32 summation noise | 6.0e-08 |
| 16×42 (non-tiling) | exact pooling undefined; **adaptive works** | — |

- **This implementation's DEFAULT gives `state_dim` 2048 at every candidate** — `[2048, 2048, 2048,
  2048]`, measured. That is their route, reached by a different spelling.
- **`grid_w` is the expensive thing they are warning about, and they are right to.** Setting it to
  10 gives `4×10×128 = 5120` and makes every downstream checkpoint unloadable. It is `None` by
  default and now carries that warning in its docstring and in `ReadoutConfig`.
- **Convergence, so this cannot drift again:** `SpatialGridReadout` now uses the **exact kernel
  where the grid tiles** (keeping the deployed path byte-for-byte) and **falls back to adaptive
  where it does not** — the union of both routes, replacing my hard divisibility assert. Pinned by
  `test_the_two_pooling_ROUTES_are_the_same_operation`.
- **"Only `encoder.pos` breaks" is confirmed** — and §4b already provides `resize_pos_embed()` /
  `adapt_pos_embed_()` for it, with the one-directional bias declared.

⇒ **v5 uses the default `(grid=4, grid_w=None)` readout.** No rework, nothing retired, one framing
correction on each side.

---

## 9. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `GEOMETRY_CONFIGURABLE.md` (this doc) | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-geometry-configurable/` | no — staged |
| `measure_fidelity.py` | same dir | no — staged |
| `measure_rebuild_cost.py` | same dir | no — staged |
| `measure_jpeg_cache.py` + `jpeg_cache_2026-07-27.json` | same dir | no — staged |
| `measure_route_and_rig.py` + `route_and_rig_2026-07-27.json` | same dir | no — staged |
| `fidelity_2026-07-27.json` (90 checks) | same dir | no — staged |
| `selection_verdict_2026-07-27.json` | same dir | no — staged |
| `rebuild_cost_2026-07-27.json` | same dir | no — staged |
| `tanitad/geometry.py` (**new**) | `repo:stack/tanitad/geometry.py` | no — staged |
| `tests/test_geometry_configurable.py` (**new**, 53 tests) | `repo:stack/tests/` | no — staged |
| `calib.py` — `CanonicalFrame`, rect f-theta, cylindrical | `repo:stack/tanitad/data/calib.py` | no — staged |
| `parity.py` — `register_geometry_sibling` | `repo:stack/tanitad/data/parity.py` | no — staged |
| `v2_dataset.py` + `v2_compressed.py` — non-square payload/manifest (v3) + lossless codec | `repo:stack/…` | no — staged |
| `refc.py` — non-square REF-C, per-axis stride check, duplicate `advect` retired | `repo:stack/tanitad/refs/refc.py` | no — staged |
| `config.py` · `encoder.py` · `readout.py` · `imagination.py` · `fourbrain.py` · `dynamics_encoder.py` · `refb.py` · `physicalai.py` · `comma2k19.py` · `train_worldmodel.py` · `finetune_traj.py` · `build_pai_cache.py` · `train_flagship4b.py` | `repo:stack/…` | no — staged |

**Nothing lives in only one place. Nothing is on a pod. Nothing was committed or pushed.**
No PhysicalAI clip UUID appears in any artifact (clips are index + sha1 prefix).
`Keys.txt` verified absent from the index.

### ⚠️ SHARED-INDEX DISCLOSURE (CLAUDE.md §Git hygiene)

The index contains **another stream's work**, and whoever commits must know:

| file | contains | note |
|---|---|---|
| `stack/tanitad/data/comma2k19.py` | **BOTH** the IDM-v3 stream's standstill-heading repair (`HEADING_MODE_*`, `hold_heading_through_standstill`, hunks at ~L45/159/171/178) **AND** my geometry modes (hunks at ~L219/236) | I edited *on top of* their working-tree version; `git diff` is empty, so nothing was lost. |
| `stack/tests/test_comma2k19.py` | **entirely the IDM-v3 stream's** | ⚠️ **Already staged before I began — I did NOT `git add` it.** |

Everything else under `stack/` in the index is mine. Per CLAUDE.md, `git commit -- <pathspec>`
**segfaults on this repo**, so the admissible route is a pathspec-free `git commit -F <msgfile>`
*after* listing the index — and the commit message must name the IDM-v3 files above rather than
pretend they are part of this change.

### Files I own vs the siblings
I own the **training-path edits** listed above. `2026-07-27-fov-crop-audit/` and
`2026-07-27-encoder-tokenization/` write to their own directories. **I touched no file in either.**
⚠️ If the encoder/tokenization sibling changes `models/encoder.py` or `models/readout.py`, we
collide — those two are mine in this change set (see §3.3 for the interfaces I deliberately left open).

---

## 10. Escalation — what needs a decision, not a doc

1. ✅ ~~**STORAGE is the binding constraint on v5.**~~ **RESOLVED — no provisioning decision is
   needed.** The v2 format's scalar `image_size` (the blocker) is unblocked, and the compressed
   cache puts the widest candidate at **112.9 GB LOSSLESS** vs today's **349 GB raw** (§2b).
   ⚠️ Two inherited premises died on the way: the *"~466 GB quota"* is falsified (pod3 holds
   **487 GB**, writing at 415 MB/s), and *"873 GB → ~700 GB still overruns"* is moot at 113 GB.
   **What remains is a format decision, not a purchase:** adopt the compressed cache for the
   rebuild. Recommended framing for the PI — *lossless PNG, because it fits and costs nothing in
   fidelity*; JPEG only if the ~3× build-time saving matters more than bit-exactness.
2. **A corpus-key registration step must enter the v5 runbook** (§1.4). Without it the trainer
   refuses the new cache — a silent schedule stall at the worst moment.
3. **`CORPUS_META` × 4 and the lake schema must move WITH the default flip** (§3.2). They are the
   most likely place for a stale default to survive; they are named here so the flip is a checklist,
   not a memory test.
4. ✅ ~~**REF-C cannot run wide.**~~ **DONE.** `CNNEncoderConfig` gained `image_width` +
   `grid_shape`; the stride-32 check is now **per axis** (it checked one number, so a 256×640 frame
   would have passed while the other axis was silently mis-sized); and REF-C's **duplicate `advect`
   copy is retired** in favour of the shared grid-general one (re-exported, so importers are
   unaffected). 40/40 REF-C tests pass. **Remaining decision: whether REF-C follows v5's geometry
   or stays at 256²** — that is a comparison-design call, not an engineering blocker.

5. ✅ **The `CORPUS_META` × 4 + lake-schema flip checklist is now MECHANICAL**
   (`assert_geometry_literals_consistent`): flip the frame without moving them and it FAILS, naming
   every file. Both directions tested. ⚠️ It also surfaced that
   `lake.schema.LakeRecord.image_size` is a **scalar that cannot express a non-square frame at
   all** — a schema bump (`image_h`/`image_w`) is still owed before the lake ingests wide episodes.

6. ⛔ **The rig-asymmetric fabricated padding (§2c) is in every number since D-016 R1**, not just
   in v5's future. It belongs in the retraction / known-limitations record whichever geometry wins.

**Unblocks:** `Project Steering/Gates/flagship-v5-retrain.PREP.md` **item 7**. Also unblocks the FOV
audit (its candidates are now executable rather than hypothetical) and gives the
encoder/tokenization stream a real non-square substrate to measure on.
