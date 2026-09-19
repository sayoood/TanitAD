# E17 / C3 — the 4,713-clip source transfer, priced: **zero network bytes. The frames are already on the dev box.**

**Date** 2026-09-19 · **Owner** DataFlyWheel · **Asked by** Master Mind (task 2 of 3, PI-approved)
**Answers** `PREREG_REFCV6_DEVBOX_PREPARATION.md` §3.3 **C3** and checklist item **E17**.
**Compute** 0 GPU. Nothing was downloaded: every HF comparison uses repo **metadata** only.

## The answer

| | |
|---|---|
| **where they live now** | `C:\Users\Admin\tanitad-data\physicalai\camera\camera_front_wide_120fov\` on **this dev box** (C:, NTFS) — 4,719 mp4s plus 4,719 per-clip timestamp parquets, file dates **2026-08-23 → 2026-08-30**. That the HF camera upload was made from this copy is **INFERRED** (dates plus byte-identical content), not traced. Durable copy: private HF `Sayood/tanitad-v7-training-corpus` `camera/`. |
| **bytes** | **61,545,041,700 B** for the 4,713 train clips · 61,616,613,650 B for all 4,719 (= `MANIFEST.json` `camera.bytes_total` exactly) · 71,571,950 B for the 6 the parity gate drops |
| **route** | **local**. Stage from C:, write the cache to D:. No network. |
| **rate** | local read **≥ 826.5 MB/s** — read + sha256 of all 61.6 GB in **74.6 s** |
| **hours** | **0 h of transfer.** Optional staging copy C: → D: **≈ 1–11 min**, or **0 bytes** with NTFS hardlinks staged on C: |

Evidence: **MEASURED (ours)** · `raw/verify_local_source.json`, `raw/verify_sidecars.json`.
The staging-copy minutes are **ESTIMATED**: 61.6 GB at the measured 826.5 MB/s read, bounded
below by an **unmeasured** D: write rate floored at 100 MB/s.

## ⛔ This corrects a premise in the pre-registration

Constant 13 reads *"There is no train corpus and no source-frame corpus on this box"*, and C3
builds on it: *"an unpriced data transfer is"* the blocker. The `du`/`df` behind it was
correct **for `D:\Projects\TanitAD-artifacts`**, which is the only place it looked.
⇒ **Retraction class C2 — absence from a single probe.** A second-location search
(`find` over `C:\Users\Admin` and `D:\`) found the corpus on C: in one pass.
⇒ **C3's transfer leg is void.** The rebuild is still blocked by **C2 (the SAM3 corpus)**,
not by data movement. The Master Mind owns the pre-registration; the amendment is flagged to
it rather than edited here.

## Why "identical" is proven, not assumed

The local files were checked against an **independent** source — the size and LFS sha256 that
HF publishes per file (revision `0ddee95d6a59`) — not against anything derived from them.

| check | result |
|---|---|
| local mp4 ↔ HF `camera/*.mp4`, by clip id | **4,719 / 4,719**, 0 local-only, 0 HF-only |
| byte length | **0** mismatches |
| **sha256 of every byte** (61.6 GB) | **0** mismatches |
| the 6 dropped clips | derived as the corpus ∩ `deployed_val40_clip_digests.json` → **6** |
| ⭐ **positive control**: the remaining 4,713 against the B1 membership digest pinned in `parity_manifest.json` (`physicalai-b1-w120-256x640cyl`) | **`e8bfb98e…` reproduced exactly** — a wrong exclusion set cannot pass this |
| timestamps + egomotion cover all 4,713 | **yes** (4,719 parquets; 4,800 egomotion parquets) |
| `egomotion/egomotion_alpamayo.tar` (1,967,093,760 B) | **sha256 = HF** |
| `timestamps/timestamps.tar` (53,155,840 B) | **sha256 = HF** |
| `physicalai_front_wide_intrinsics.csv` | present (731,043 B); local-only **by design** — the fetcher requires it to be placed |

⇒ **Every input `fetch_corpus_clips.py` pulls from HF already exists on this box, byte-identical.**

## The fallback, if the local copy is ever lost

| route | rate | hours for 61.545 GB | class |
|---|---|---|---|
| **HF pull, serial** | per request **1.192 s** + **9.46 MB/s** per stream (fit R² 0.839, n 139) | **3.37 h** (95 % 3.29–3.46) + ~3.6 min of sidecars | **MEASURED** basis · `raw/fetch_fit.json` from the 2026-09-16 receipt |
| HF pull, aggregate-rate projection | 5.345 MB/s | 3.20 h | ⚠️ under-prices: corpus clips (13.06 MB mean) are **smaller** than the sample's (14.64 MB), so per-request overhead weighs more |
| HF pull, k parallel streams | — | serial ÷ k **only if** the link carries k × 9.46 MB/s | ⛔ **UNMEASURED** — the link ceiling was never measured; the 2026-09-16 "0.8 h at 4×" is this assumption |
| **Thor** | — | **not a route** | Thor evicts source mp4s after use and holds only the derived 256×640 cache, which cannot feed a 408×1024 build (INHERITED · 2026-09-16 package) |
| upstream NVIDIA range reads | not measured | — | fallback of the fallback |

## What the rebuild still needs — none of it is a transfer

1. ⛔ **C2 — the SAM3 corpus.** Unchanged; it blocks the rebuild.
2. **A staging root in the fetcher's layout** (`<root>/r0/camera_front_wide/<clip>.mp4` +
   `.timestamps.parquet`, `<root>/labels/egomotion/egomotion_all.zip`,
   `<root>/calibration/…csv`). `fetch_corpus_clips.py` only knows the HF route. Two options,
   neither built here: **(a)** a local-mirror mode — NTFS hardlinks on C: cost 0 bytes, but
   D: is **exFAT** and cannot hold links; **(b)** a copy to D:, ~1–11 min.
3. ⚠️ **Move the two sidecar tars out of the mirror first.** They sit in
   `C:\Users\Admin\tanitad-wt\_s2build\…`, the mirror that has deleted repo-absent files on
   resync. 2.02 GB to `D:\Projects\TanitAD-artifacts\` is seconds; a re-fetch is ~3.6 min.
4. **Disk.** The cache goes to **D:** (1,329.6 GB free). C: has 170.2 GB free, less than the
   386.5 GB cache. MEASURED · `Get-Volume`.
5. **The rebuild's own write demand is ~10 MB/s** — 446 clips/h × 82.0 MB/clip, from the
   2026-09-16 package's 18.7 min / 139 clips at 408×1024 (**INHERITED**, measured there;
   `raw/build408.log`). No disk here is the bottleneck; the build is CPU-bound.

## Reproduce

```
python code/verify_local_source.py raw/verify_local_source.json
python code/fit_fetch_receipt.py <2026-09-16 package>/raw/fetch_receipt_eval139.json 61545041700 4713 raw/fetch_fit.json
```

🔒 Outputs carry sha12 only. A scan of every file in this package over 4,725 known clip ids,
their 8-hex prefixes and their 12-hex tails found **0** hits, against a control that reads 7.
