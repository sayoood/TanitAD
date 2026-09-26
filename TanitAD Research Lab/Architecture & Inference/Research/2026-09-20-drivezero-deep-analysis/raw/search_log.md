# search log — DriveZero deep analysis (2026-09-20, PI request)

| # | probe | result | note |
|---|---|---|---|
| 1 | ⭐ **V-1 / LI19-1 re-find** — `rg "DriveZero\|2609.06055"` across the Research Lab | **12 files** (LIBRARY, 3 KNOWLEDGE_BASEs, LAB_BACKLOG, TRACKS, OPPONENT_CLAIMS_REGISTER, DrivoR analysis, LAB-RUN-015/016, LEDGER_B11, one search log) | ⛔ **DriveZero is NOT new to us** — banked and number-verified on 2026-09-19. This package is a **deep read of a held paper**, not a discovery. ⚠️ A plain `grep -rli` over the same tree **timed out at 120 s**; ripgrep returned in seconds — noted so the next pass does not read the timeout as an absence |
| 2 | Library: banked PDF present? | ✅ `2609.06055_DriveZero-End-to-End-Driving-Beyond-Human-Demonstrations.pdf` | **32 pages, 104,780 chars** |
| 3 | ⭐ **local pypdf full-text read** (FS19-9 — summariser numbers are not primary numbers) | all sections + appendices A.1–A.5, B.1–B.4 | every number in RESULT.md §1–§3 comes from **this** read, not from a fetch |
| 4 | in-PDF: `navtest` / `navhard` stamps | both present and explicit | ⭐ unlike Drive-HWM (same day), DriveZero **stamps its own splits** ⇒ 57.1 navhard stands in our table |
| 5 | in-PDF: **`limitation`** (case-insensitive) | ⛔ **0 hits in 32 pages** | **W1** — a paper with no limitations section |
| 6 | in-PDF: DriveVFM GPU count / hours | ⛔ **not stated** | **W8** — the expensive half of the system has no published bill |
| 7 | web: `DriveZero arxiv 2609.06055 autonomous driving NAVSIM zero-shot` | 9 | confirmed identity + headline numbers |
| 8 | web: **successors** — `DriveZero Xiaomi follow-up successor "DriveRL" OR "DriveVFM" citing October November 2026` | 9 | ⛔ **EMPTY for successors at one probe.** Hits were the paper's own components, its GitHub, and Xiaomi product pages. Published 2026-09 ⇒ too recent. **Named as a single-probe absence; re-probe later, do not harden** |
| 9 | web: rivals — `"GigaPixel" ... OR "ZTRS" ...` | 9 | ⭐ **GigaPixel `2606.19641`** (50.1 navhard) and **ZTRS `2510.24108`** (48.1) located and **banked** — the zero-human-demo family |
| 10 | fetch `xiaomiautol3.github.io/DriveZero` | read | code link present; ⛔ no license, no weights discussion on the page |
| 11 | fetch `github.com/XiaomiAutoL3/DriveZero` | read | **DriveRL inference code + checkpoints released 2026-09-15**; **DriveZero and DriveVFM are TODO** |
| 12 | fetch the repo root `README.md` (raw) | read | TODO checklist confirmed; ⛔ **no explicit license statement located** |
| 13 | fetch `DriveRL/README.md` (raw) | read | Python 3.11, `nuplan-devkit`, CUDA 12.8, filters `driverl_val14 / test14_hard / test14_random`; ⛔ **no license, no hardware spec, no expected scores** in the README itself |
| 14 | repo: is nuPlan data local? | **maps yes** (`data/nuplan-maps/`), **DB/sensors no** | the hard gate on any direct reproduction |
| 15 | repo: our own comparison numbers (`MODEL_REGISTRY.md`, `PROGRAM_OVERVIEW.md`) | read | 263 M deployed · **60.3 ms p50 / 63.1 ms p95 on Thor** · closed loop 1.488 m [1.329, 1.647] · `alpasim_runtime` unfinished ⇒ **no collision/offroad score** |
| 16 | arithmetic: `code/dz_budget.py` | ✅ **2 internal consistency checks pass** (2,048×96 = stated global batch; 2,048×110 = stated transitions/rank) | the DERIVED scale figures. ⭐ The paper **never states driving hours**, so ours is a reconstruction and is labelled DERIVED, not MEASURED-by-the-authors |

## Named empties / gaps

- **E-DZ-SUCC** *(new)* — no successor to DriveZero located at **one** probe. Single-probe absence.
- **E-DZ-LICENSE** *(new)* — no license statement located in the project page, the repo README, or the DriveRL README (**3 probes**). ⇒ treat as all-rights-reserved; **PI/legal**, not a Lab call.
- **DriveVFM training cost** — not stated by the authors (W8).
- **Student latency / FLOPs** — not reported by the authors (W7). ⭐ This is where our Thor measurement is the differentiator.

## Addendum 2026-09-20 — the encoder's "driving hours" question (PI follow-up)

| # | probe | result |
|---|---|---|
| 17 | in-PDF: DriveVFM param count / patch size / width / depth | ⛔ **NONE of the four stated.** Only the names ViT-S / ViT-B / ViT-L appear; **no `/14` or `/16` patch notation anywhere in 32 pages**. The only three param counts in the paper are teacher 5.7 M and student 338.46 M / 18.58 M |
| 18 | in-PDF: mixture ratios / per-source counts / any figure in hours for DriveVFM | ⛔ **EMPTY.** *"identical mixture ratios"* is asserted (Table A13) and the ratios are **never given**; no per-source image counts; **no figure in hours anywhere in the paper** |
| 19 | arithmetic `code/dz_encoder.py` | frozen student part **319.88 M**; standard ViT-L computed at **304.15 M** ⇒ **15.73 M residual the paper does not explain**. Trainable 18.58 M ≈ LoRA **3.15 M** + heads **15.43 M** |
| 20 | web: `OpenDV-2K dataset hours driving video nuPlan dataset hours total size` | **RELAYED (secondary, not read at primary):** OpenDV-2K **2,059 h** (1,747 h YouTube + 312 h public), ~3 TB raw / ~24 TB processed. nuPlan **~1,200–1,282 h of logs** — ⚠️ **but only ~94–128 h carry CAMERA/sensor data** |

⭐⭐ **The trap this probe avoided.** DriveVFM trains on **images**. Quoting nuPlan's headline **1,200 h** as part of the encoder's driving data would be *true about nuPlan and wrong about DriveVFM* — the camera-bearing subset is **~94–128 h**. ⇒ the driving pool available to DriveVFM is of order **~2,200 h** (OpenDV-2K dominant), against three **non-driving** sources (LAION-2B, ImageNet-21K, SA-1B) that are orders of magnitude larger.

⛔ **Therefore "driving hours for the encoder" is NOT RECOVERABLE**, and arguably not the right unit: with unstated mixture ratios, the only defensible figure is the **DERIVED 1,638,400,000 image samples drawn** (800 K iters × batch 2,048), of which an unknown fraction is driving. Stated as a named gap rather than estimated.
