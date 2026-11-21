# refav1 fp8 cache completion — clip `16d325e9-dbfe-439c-a0ed-4ff8850ad2e2` rebuilt, shipped, linked

**Data Engineering FlyWheel · 2026-09-02 23:05 Berlin (21:05Z) · agent branch `agent/arch-inf-20260803`**
**Closes H-EPOCH-2 gate G3** (`Project Steering/GOALS_AND_CLAIMS.md` ~L1722: *"the fp8 cache holds all 4,572
v7.2 train clips — today it holds 4,571"*). **The refav1 train split on Thor now holds exactly 4,572 `.pt` — the
same clip set refcv3 trains on (4,572 train / 141 eval / 4,713 parent, intersection 0).** MEASURED, `raw/ship_verify.log`.

⛔ **Not done here (by the coordinator's instruction at ~23:20 Berlin, shared index corrupt): nothing is `git add`-ed.**
The Master Mind stages the paths listed in §9. This agent ran **no** `git add` at any time (its git use is listed in §10).

---

## 1. Headline

| question | answer | evidence class |
|---|---|---|
| Is the source v2ep sound? | **Yes.** 201/201 PNG frames decode through the corpus reader (`v2_dataset._decode_stacked`, codec `png`) → `[199, 9, 256, 640]` u8; no zero frame; per-frame mean 47.2–76.5, std 32.5–54.2; the builder's decode of every 2nd frame is **byte-identical** to the reader's (max abs diff 0). md5 `cc919d4964f6382aeefd44be2751966b` on Thor, on the manual pull, and on the builder's own re-pull. | MEASURED — `raw/decode_check.json`, `raw/verify_artifacts.json` |
| Was the old cache entry corrupt, and how? | **Torn scp push**, not a bad encode: Thor held **61,624,320 B** of the correct **66,193,268 B** (= 94.03 frames of 655,360 B; no zip central directory, `BadZipFile`), timestamp Sep 1 03:17. The original run's log shows the push thread dying on `scp` exit 1 mid-batch at episode ~432 (this clip 6th of 16 files; the 5 before it landed complete, the 10 after were rebuilt on resume). | MEASURED — `raw/ship_b1_crash_excerpt.log`, Thor `zipfile` header probe (§4) |
| Why was it never rebuilt? | The resume rule `done[e] < 10_000_000` counts **any file ≥ 10 MB** as built, so a 61.6 MB torn file passed as done forever; the pre-fix `scp()` had no retry and `_push_verify` no exception box. The 12-sample content check of the build could not see a single bad file (the register already says so). | MEASURED — builder source + log |
| Rebuilt through the same builder? | **Yes.** `stack/scripts/dinov3_fp8_encode_ship.py --only <clip>` runs the encode path factored into `encode_episode()` — the identical code the batch pipeline runs (DINOv3 ViT-L/16, HF `facebook/dinov3-vitl16-pretrain-lvd1689m`, bf16, ImageNet mean/std, no resize, every 2nd raw frame, CLS + 4 registers dropped, fp32 → `float8_e4m3fn`). Run on **CPU** (the 4060 was at 100 % / 7.7 GB under another agent's `run_arm.py --arm A_prime`), 16 threads, **834.5 s** encode. | MEASURED — `raw/rebuild_only.log` |
| Does it match what the GPU build would have produced? | The Sep-1 kept-local **GPU-bf16** artifact of this very clip (from the original run's failed batch, `C:/Users/Admin/refav1_probe/ship/`, 66,193,268 B, zip intact) agrees with the CPU rebuild to **66.9 % identical fp8 codes, rel-L2 3.26 %, per-token cosine mean 0.99949 (min 0.9886)** — the same order as the fp8 quantisation error itself (rel-MSE 1.06e-3 CPU-vs-GPU vs 7.07e-4 fp8-vs-fp16). Differing codes: median 0.73 e4m3 ulp. mean\|x\| 0.20606 (rebuild) vs 0.20619 (Sep-1) vs 0.20626 (fp16). | MEASURED — `raw/compare_rebuild.json` |
| Gate? | **Per-episode bound PASS on all three targets** (rel-MSE increase fp16→fp8: speed −0.0134, yaw-rate −0.0034, accel +0.0237; bound < 0.05). The default 5-episode gate re-run reproduces the banked JSON number-for-number (PASS). See §6 for the even-n median caveat. | MEASURED — `raw/fp8_l2_gate_extra.json`, `raw/fp8_l2_gate_regress.json` |
| Does the consumer accept it? | **Yes.** `RefAV1Windows` at the live config (op_window 4, op_steps 30, str_dt 3.0, ext 2) opens it: `T_cache 101 = ceil(201/2)`, **37 windows** (register: 36.9/episode), batch `feats [4,4,640,1024]` finite, `future_feats [4,30,640,1024]`, `str_ext_targets [4,2,640,1024]`, v0 ≈ 10.5 m/s. | MEASURED — `raw/loader_check.json` |
| Shipped? | **Yes**, via the builder's new **atomic** push (`.pt.part` → size-verify → `mv`): Thor `dinov3-b1-fp8-w120-256x640cyl/16d325e9….pt` 66,193,268 B, md5 **`049f41b12488a35e24db2d8f118fd492`** = local; symlinked into `refav1-fp8-train/` (md5 through the symlink identical). Counts after: **train 4,572 / eval 141 / parent 4,713 / `.part` 0**. The torn original is kept as `….pt.torn-61624320` (md5 `848d62b56aa8d7f3bae4bc5b13c13c0f`; no `*.pt` glob matches it). Live trainer PID 2833296 untouched and alive. | MEASURED — `raw/ship_verify.log`, `raw/md5_local.txt` |

## 2. What `index.json` is, and why it was not edited

* `refav1-fp8-train/` is a **symlink split** into the parent cache: 4,572 `*.pt` symlinks + `index.json → ../dinov3-b1-fp8-w120-256x640cyl/index.json`.
* The **loader never reads `index.json`** — `RefAV1Windows` enumerates `cache_dir.glob("*.pt")` (`refav1_loader.py`, `__init__`) and refuses a mis-gridded entry by `ceil(T_ep/2)`.
* `refa_v1_train.verify_cache()` reads it for **geometry only** (`n_tokens`/`d_enc`/`hfov_deg` vs `DINOV3_GEOMETRY`); `taniteval/tools/refav1_arm.py:749` copies its non-`episodes` fields into the eval manifest.
* Its `episodes` list (4,713, the parent set) **already contained this clip** — verified on Thor. Nothing to change.

## 3. Builder facts (read end to end; MEASURED from source + the HF snapshot on the dev box)

| item | value |
|---|---|
| encoder | `facebook/dinov3-vitl16-pretrain-lvd1689m`, HF snapshot `ea8dc2863c51be0a264bab82070e3e8836b02d51`, `model.safetensors` sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`; 24 layers, d 1024, patch 16, **4 register tokens** ⇒ `special_dropped = 5` |
| input | the v2ep's PNG frames (`codec == "png"` asserted), decoded with `torchvision.io.decode_png` → **RGB [3, 256, 640] u8**. The "9 channels" of the corpus reader are its `n_stack = 3` channel-stack; **the encoder consumes single RGB frames, not stacks** |
| preprocessing | `/255`, ImageNet mean/std from the processor config (0.485/0.456/0.406 · 0.229/0.224/0.225), **no resize / no crop** — 256×640 cylindrical is fed at native size ⇒ 16×40 = **640 patch tokens** (the processor's 224×224 default is bypassed by calling the model on `pixel_values` directly) |
| grid | every 2nd raw frame, `range(0, T, 2)` ⇒ `T_c = ceil(T_ep/2)` = 101 here (T_ep 201) |
| output | patch tokens only `h[:, 5:, :]` → fp32 `[T_c, 640, 1024]`, asserted finite, shape, `mean|x| > 1e-3`, saved as `torch.float8_e4m3fn` (1 B/elem ⇒ 655,360 B/frame + 1,908 B overhead: every healthy size on Thor is `T_c × 655,360 + 1,908` — 96/100/101/104/105 frames ⇒ 62,916,468 / 65,537,908 / 66,193,268 / 68,159,348 / 68,814,708 B; counts 9 / 343 / 4,317 / 42 / 1) |
| index.json | `{"episodes": [...4,713], "grid": "0.2s (every 2nd frame)", "dtype": "float8_e4m3fn", "geometry": {"n_tokens": 640, "d_enc": 1024, "hfov_deg": 120.0}, "model": MID, "special_dropped": 5, "gate": "fp8_l2_gate.json PASS 2026-09-01"}` |
| dtype on Thor | neighbour `002646e7…pt` header (mmap, Thor venv torch 2.13.0+cu130): `float8_e4m3fn (101, 640, 1024)` — the rebuilt entry is `float8_e4m3fn [101, 640, 1024]`, 66,193,268 B, same as 4,317 siblings |

## 4. The corruption, precisely (MEASURED)

```
Thor  16d325e9….pt   61,624,320 B  Sep 1 03:17   PK\x03\x04 header, tail = raw payload (no PK\x05\x06)  → zipfile: BadZipFile
local ship/16d325e9….pt 66,193,268 B Sep 1 03:17  zip intact; data/0 = 66,191,360 B = 101×640×1024  (the file scp was sending)
ship_b1.log (line ~30): Thread-213 (_push_verify) scp … 16 files … exit 1   ← this clip 6th in the list
                        Thread-214 (_pull)        scp … exit 1              ← the LAN dropped for both threads
                        main: torch.load of a half-pulled v2ep → crash → restart (resume)
```
On resume the 61.6 MB file satisfied `size ≥ 10 MB ⇒ done`. The five files ahead of it in the failed list are complete on Thor (66,193,268 B, all in the train split); their local copies in `ship/` are stale leftovers (harmless).

## 5. Rebuild run (MEASURED)

```
[b1] --only: 1 episode(s) to rebuild on cpu/bf16 (resume scan bypassed)
[b1] 16d325e9-dbfe-439c-a0ed-4ff8850ad2e2 T=101 mean|x|=0.2063 834.5s
[b1] DONE 1 built in 14.0 min (index.json untouched)      EXIT=0
```
outputs (local, `C:/Users/Admin/refav1_probe/rebuild/`): `16d325e9….pt` fp8 66,193,268 B md5 `049f41b1…`; `16d325e9….fp16.pt` 132,384,663 B md5 `458483ae7800395cb5627b438490042e` (the gate's fp16 arm; local only, too large to bank).
⚠️ CPU bf16 on the dev box (i9-12900F, no AVX512-BF16/AMX) is a reference-speed path: 834 s for one episode vs ~2.9 s on the 4060; a single bf16 `[5160×1024]@[1024×4096]` matmul micro-benchmark did not finish in 120 s. Use `--dtype fp32` for any future CPU rebuild of more than a handful of episodes (the fp32-vs-bf16 difference is below the fp8 quantisation step; quantify it against a GPU artifact as done in §1 if it matters).

## 6. Gate detail (MEASURED, `raw/fp8_l2_gate_extra.json`)

Probe fit on the 19 FIT episodes exactly as banked (PCA-128 basis fit-split-only, λ by leave-episode-out CV, n = 1,918, d = 128); the rebuilt episode scored as an **additional VAL** episode (n = 101), never in the fit.

| target | fp16 MSE | fp8 MSE | rel Δ (this episode) | per-episode bound |
|---|---|---|---|---|
| speed | 30.3018 | 29.8947 | **−0.0134** | < 0.05 ok |
| yaw_rate | 0.0066 | 0.0066 | **−0.0034** | < 0.05 ok |
| accel | 1.2442 | 1.2736 | **+0.0237** | < 0.05 ok |

Default 5-episode gate re-run: PASS, every number identical to the banked `fp8_l2_gate.json` (speed −0.0005/+0.0069, yaw +0.0083/+0.0344, accel −0.0060/+0.0213; the only JSON difference is the added `n_episodes` key).

⚠️ **Pooled 6-episode re-evaluation depends on the even-n median convention.** The frozen implementation's `sorted(rel)[n//2]` is the exact median for the banked odd n = 5 but the *upper* median for n = 6: on accel it reads **+0.0128 (FAIL vs < 0.01)** while the standard median reads **+0.0034 (PASS)**; sorted accel deltas `[−0.0192, −0.0137, −0.0060, +0.0128, +0.0213, +0.0237]` — the +0.0128 that trips the upper-median was already in the banked set. The rebuilt episode is the new accel max (+0.0237 < 0.05). **Verdict: the entry is within the pre-stated per-episode bound on every target; the pooled FAIL is a convention artifact, reported, not hidden.** The script now emits both conventions in the extra block; the default path is unchanged.

## 7. What would make the corruption recur, and what changed

| mechanism | status |
|---|---|
| A push that dies mid-file leaves a partial under the **final name** | **FIXED** in `dinov3_fp8_encode_ship.py`: files travel as `<ep>.pt.part`, are size-verified, then `mv`-ed in one ssh; a torn transfer never carries the final name, so the resume scan (final names only) rebuilds it. Exercised on this ship (`parts_left=0`). |
| `scp()` with no retry; `_push_verify` with no exception box | fixed 2026-09-01 03:22 (`_run`, 3 tries) — that version is the only one ever committed (`8f302e9`); the pre-fix code survives only in the log's line numbers |
| Resume rule `size ≥ 10 MB ⇒ done` | **unchanged** — safe *only because* of the atomic rename; a stronger check (expected size `T_c×655,360+1,908`, or the mmap header shape) remains the definitive per-file test. Recommend the full-set header scan the register already prescribes before any launch. |
| Content check on a 12-sample draw | cannot see a per-file defect (the register's own lesson); the shipped split was full-scanned (4,712/4,713 loadable) — that scan is what found this clip |
| ⚠️ **Live-run hazard created by this ship** | The incumbent run enumerated 4,571 episodes at init and keeps that list; **the new symlink is picked up at the next loader init** — i.e. any supervisor relaunch (`--resume`) of the incumbent would silently train on 4,572 episodes with a different seeded window order from that step on. The intended consumer is the H-EPOCH-2 clean-epoch restart. If the incumbent must stay bit-reproducible across a relaunch, unlink `refav1-fp8-train/16d325e9….pt` until the restart. |

## 8. Instrument changes (on disk in the mirror AND copied to the repo; sha256 read-back verified)

* `stack/scripts/dinov3_fp8_encode_ship.py` — `argparse` CLI; `--only CLIP…` (bypasses the resume scan, keeps the local copy, never rewrites `index.json`), `--device`, `--dtype {bf16,fp32}`, `--threads`, `--work`, `--keep-fp16`, `--no-ship`; encode path factored into `load_encoder()` + `encode_episode()` (batch pipeline and `--only` share it); **atomic `.part`→`mv` push**; `FINAL PUSH FAILED` now exits 1. Default batch behaviour otherwise unchanged. `py_compile` OK; no test imports it.
* `stack/scripts/fp8_l2_gate.py` — module-level run moved into `main()` (importable); `--cache/--eps/--out`; `--extra-fp16 + --extra-ep` scores one extra episode as an additional val episode under the same fit-split probe and reports the per-episode bound + the pooled gate under both median conventions; `--out` is mandatory with `--extra-*` so the banked JSON is never overwritten. Default run reproduces the banked JSON number-for-number.

## 9. Deliverable manifest (⛔ NOT staged — coordinator paused git; Master Mind to `git add` these)

| artifact | location | only one place? |
|---|---|---|
| this package: `RESULT.md`, `raw/{verify_artifacts.py,.json, decode_check.py,.json, compare_rebuild.py,.json, loader_check.py,.json, fp8_l2_gate_extra.json,.log, fp8_l2_gate_regress.json, rebuild_only.log, ship_verify.log, ship_b1_crash_excerpt.log, md5_local.txt, MD5SUMS.txt}` | `repo:TanitAD Research Lab/Data Engineering/Research/2026-09-02-refav1-cache-completion/` (+ originals in `C:/Users/Admin/refav1_probe/rebuild/`) | no (repo + local) |
| builder change | `repo:stack/scripts/dinov3_fp8_encode_ship.py` (+ mirror `C:/Users/Admin/tanitad-wt/stack/scripts/`) | no |
| gate change | `repo:stack/scripts/fp8_l2_gate.py` (+ mirror) | no |
| rebuilt fp8 entry `16d325e9….pt` (md5 `049f41b1…`) | `tanitad-thor-wifi:/home/nvidia/data/dinov3-b1-fp8-w120-256x640cyl/` + symlink in `…/refav1-fp8-train/` + local `C:/Users/Admin/refav1_probe/rebuild/` | no (Thor + dev box) |
| fp16 field `16d325e9….fp16.pt` (132 MB, md5 `458483ae…`) | `C:/Users/Admin/refav1_probe/rebuild/` | **YES — dev box only** (re-derivable: `--only … --keep-fp16`) |
| torn original (evidence) | `tanitad-thor-wifi:/home/nvidia/data/dinov3-b1-fp8-w120-256x640cyl/16d325e9….pt.torn-61624320` | **YES — Thor only** (deliberately not copied: it is the defect, 61 MB) |
| Sep-1 GPU-bf16 artifact of the same clip (comparison reference) | `C:/Users/Admin/refav1_probe/ship/16d325e9….pt` (md5 `72df09fafc49782e4a7995e8219fc256`) | **YES — dev box only** |

## 10. Git commands this agent ran (for the index-corruption attribution; none wrote the index)

22:39–22:44 Berlin: `git hash-object <5 files>` ×2 trees, `git --no-pager diff --stat HEAD -- <3 paths>` (timed out on G: at 2 min, killed), `git --no-pager log -n 70 --format=…` (no path), then a **sequential** loop of 70 × `git rev-parse -q --verify <commit>:stack/scripts/dinov3_fp8_encode_ship.py` + one `git ls-files --stage <path>` (background task, exit 0, finished before 22:50). 22:57: `git --no-pager diff --stat -- <2 paths>`. Mirror: `git rev-parse --is-inside-work-tree` (not a repo). **No `git add`, no `git status`, no commit, nothing after 22:57, never ~30 concurrent processes.**

## 11. Register / integration items for the Master Mind

1. `GOALS_AND_CLAIMS.md` H-EPOCH-2 **gate G3 → satisfied** (4,572 = refcv3's set), citing this RESULT.md; D-REFAV1-LAUNCHED's "train 4,571 (after the corrupt exclusion)" is now historical.
2. Decide the §7 live-run hazard (leave the symlink for the restart vs unlink until then).
3. Stage §9 after the index rebuild.
