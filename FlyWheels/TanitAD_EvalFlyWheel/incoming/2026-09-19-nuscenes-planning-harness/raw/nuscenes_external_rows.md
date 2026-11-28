# nuScenes open-loop planning — published rows, each with its protocol

**For W4 / E4 (`LEADERBOARD.md`, `published_results.json`).** Built 2026-09-20 by W6/E5.
Every number below was **re-read from the banked PDF** named in its row (library key · table · page),
not copied from a summary. Evidence class **PUBLISHED** unless the row says otherwise.

> ⛔ **These rows are cited external comparability ONLY.** nuScenes open-loop planning is
> inadmissible as a TanitAD criterion (`H-EVAL-6` SUPPORTED) and SKIP claim-bearing
> (`D-BENCH-PORT`). ⛔ **A row may only be compared with rows carrying the SAME protocol tag.**
> Cross-tag comparison is the error this file exists to prevent: the averaging convention alone
> moves one VAD checkpoint **0.72 → 1.22 m (+70 %)** and **flips** the UniAD/VAD ranking.

## The protocol tags

| tag | L2/collision reduction | valid samples | collision grid | pedestrians in the map |
|---|---|---|---|---|
| `uniad-noavg` | value **AT** t (UniAD `nuscenes_e2e_dataset.py:1043-1044 @609ee08`) | all 6,019 (missing steps scored 0, sample still counted) | axis-aligned ego box, 0.5 m | **excluded** |
| `stp3-temavg` | mean **UP TO** t (ST-P3 `evaluate.py:166 @69aabef`; VAD, AD-MLP) | 5,119 (VAD `fut_valid`) / 4,819 (ST-P3: also drops the first 2 frames per scene) | axis-aligned ego box, 0.5 m | included |
| `bevplanner` | ST-P3 lineage + **0.1 m** raster, estimated yaw, trajectory-level collision (`Σ𝕀 > 0`), 5,119 samples, adds **CCR** | 5,119 | oriented-ish, 0.1 m | included |
| `paradrive-std` | **oriented** ego box, 0.1 m grid, first frame removed, + map compliance and the 686-frame targeted subset | — | oriented, 0.1 m | included |
| `sparsedrive` | L2 per VAD (`stp3-temavg`); collision re-implemented box-vs-box with estimated yaw | 5,119 | polygon vs polygon | included |

⚠️ **Ego status**: the *paper* rows of UniAD and VAD are presented as ego-status-free, but
BEV-Planner (`2312.03031`, Table 1, p.5) states that the **officially released checkpoints use ego
status in the BEV module** (its ID-2 and ID-5 rows are labelled "Official"). The column below says
what each paper claims, with that caveat carried.
⚠️ **Command**: every row listed here consumes the **GT-derived** high-level command (the final
lateral offset of the GT future thresholded at ±2 m) unless stated — VAD
`vad_nuscenes_converter.py:452-457`, UniAD `trajectory_api.py:272-280`, ST-P3
`NuscenesData.py:525-530`, SparseDrive `models/motion/decoder.py:167-173` (all MEASURED-from-source
at the pinned commits). That is our route-echo defect, published as SOTA.

## `uniad-noavg` (L2 AT the timestep)

| method | ego status | cmd | L2 1s | 2s | 3s | **Avg** | Col 1s | 2s | 3s | **Avg** | source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NMP (LiDAR) | — | ? | — | — | 2.31 | — | — | — | 1.92 | — | `2212.10156` T7 p7 |
| SA-NMP (LiDAR) | — | ? | — | — | 2.05 | — | — | — | 1.59 | — | `2212.10156` T7 p7 |
| FF (LiDAR) | — | ? | 0.55 | 1.20 | 2.54 | 1.43 | 0.06 | 0.17 | 1.07 | 0.43 | `2212.10156` T7 p7 |
| EO (LiDAR) | — | ? | 0.67 | 1.36 | 2.78 | 1.60 | 0.04 | 0.09 | 0.88 | 0.33 | `2212.10156` T7 p7 |
| ST-P3 | not claimed | GT | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 | `2212.10156` T7 p7 (ST-P3's own numbers, ⚠️ TemAvg — see the mixing note) |
| **UniAD** | not claimed (official ckpt: ego in BEV) | GT | 0.48 | 0.96 | 1.65 | **1.03** | 0.05 | 0.17 | 0.71 | **0.31** | `2212.10156` T7 p7 |
| VAD (re-scored by PARA-Drive) | no | GT | 0.50 | 1.02 | 1.68 | 1.07 | 0.02 | 0.28 | 0.85 | 0.38 | `paradrive-cvpr2024` T8 p8 |
| PARA-Drive (re-scored) | no | ? | 0.40 | 0.77 | 1.31 | **0.83** | 0.07 | 0.25 | 0.60 | 0.30 | `paradrive-cvpr2024` T8 p8 |
| ⚠️ **GT human trajectory** | — | — | — | — | — | — | 0.35 | 0.38 | 0.35 | **0.36** | `paradrive-cvpr2024` T8 p8 |

## `stp3-temavg` (L2 averaged UP TO the timestep)

| method | ego status | cmd | L2 1s | 2s | 3s | **Avg** | Col 1s | 2s | 3s | **Avg** | source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ST-P3 ⚠️ tail-GT defect | not claimed | GT | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 | `2207.07601` T3 p13 (Avg computed by later papers; ST-P3's own table has no Avg column) |
| VAD-Tiny | no (‡ absent) | GT | 0.46 | 0.76 | 1.12 | 0.78 | 0.21 | 0.35 | 0.58 | 0.38 | `2303.12077` T1 p6 |
| **VAD-Base** | no (‡ absent) | GT | 0.41 | 0.70 | 1.05 | **0.72** | 0.07 | 0.17 | 0.41 | **0.22** | `2303.12077` T1 p6 |
| VAD-Tiny ‡ | **yes** | GT | 0.20 | 0.38 | 0.65 | 0.41 | 0.10 | 0.12 | 0.27 | 0.16 | `2303.12077` T1 p6 |
| VAD-Base ‡ | **yes** | GT | 0.17 | 0.34 | 0.60 | **0.37** | 0.07 | 0.10 | 0.24 | **0.14** | `2303.12077` T1 p6 |
| UniAD (re-scored by PARA-Drive) | no | GT | 0.48 | 0.74 | 1.07 | 0.76 | 0.12 | 0.13 | 0.28 | 0.17 | `paradrive-cvpr2024` T8 p8 |
| PARA-Drive (re-scored) | no | ? | 0.25 | 0.46 | 0.74 | **0.48** | 0.14 | 0.23 | 0.39 | 0.25 | `paradrive-cvpr2024` T8 p8 |
| AD-MLP: trajectory only | **yes (no perception at all)** | no | 0.53 | 0.91 | 1.48 | 0.97 | 0.17 | 0.46 | 0.83 | 0.49 | `2305.10430` T1 p3 |
| AD-MLP: + velocity | **yes** | no | 0.33 | 0.48 | 0.66 | 0.49 | 0.21 | 0.29 | 0.40 | 0.30 | `2305.10430` T1 p3 |
| AD-MLP: + acceleration | **yes** | no | 0.24 | 0.32 | 0.49 | 0.35 | 0.18 | 0.22 | 0.28 | 0.23 | `2305.10430` T1 p3 |
| **AD-MLP: + command** | **yes** | GT | 0.20 | 0.26 | 0.41 | **0.29** | 0.17 | 0.18 | 0.24 | **0.19** | `2305.10430` T1 p3 (n = 4,819 stated in §3.3) |
| ⚠️ **GT human trajectory** | — | — | — | — | — | — | 1.02 | 0.96 | 0.91 | **0.96** | `paradrive-cvpr2024` T8 p8 |

⚠️ AD-MLP's released checkpoint is separately reported as **GT-leaky** (PARA-Drive fn. 5, p.8: the
released model "is trained with GT data leakage"); PARA-Drive re-implemented it rather than using it.

## `sparsedrive` (VAD's L2, re-implemented collision)

| method | ego status | cmd | L2 1s | 2s | 3s | **Avg** | Col 1s | 2s | 3s | **Avg** | source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UniAD † | not claimed | GT | 0.45 | 0.70 | 1.04 | 0.73 | 0.62 | 0.58 | 0.63 | 0.61 | `2405.19620` T2b p8 |
| VAD † | yes | GT | 0.41 | 0.70 | 1.05 | 0.72 | 0.03 | 0.19 | 0.43 | 0.21 | `2405.19620` T2b p8 |
| SparseDrive-S | predicted ego state (not GT) | GT | 0.29 | 0.58 | 0.96 | **0.61** | 0.01 | 0.05 | 0.18 | **0.08** | `2405.19620` T2b p8 |
| SparseDrive-B | predicted ego state (not GT) | GT | 0.29 | 0.55 | 0.91 | **0.58** | 0.01 | 0.02 | 0.13 | **0.06** | `2405.19620` T2b p8 |
| OccNet | ? | ? | 1.29 | 2.13 | 2.99 | 2.14 | 0.21 | 0.59 | 1.37 | 0.72 | `2411.15139` T7 p8 |
| **DiffusionDrive** | follows SparseDrive | GT | 0.27 | 0.54 | 0.90 | **0.57** | 0.03 | 0.05 | 0.16 | **0.08** | `2411.15139` T7 p8 ("metric calculation follows ST-P3", built on SparseDrive) |

† reproduced by the SparseDrive authors with the official checkpoints. ⚠️ UniAD's collision jumps to
0.61 here because SparseDrive re-adds the pedestrians UniAD's own protocol excludes.

## `bevplanner` (0.1 m raster, trajectory-level collision, + CCR)

| # | method | ego in BEV | ego in planner | L2 1/2/3 s | **Avg** | Col 1/2/3 s | **Avg** | **CCR Avg** | ckpt |
|---|---|---|---|---|---:|---|---:|---:|---|
| 0 | ST-P3 † | ✗ | ✗ | 1.59/2.64/3.73 | 2.65 | 0.69/3.62/8.39 | 4.23 | 8.37 | Official |
| 1 | UniAD | ✗ | ✗ | 0.59/1.01/1.48 | **1.03** | 0.16/0.51/1.64 | 0.77 | 1.93 | Reproduce |
| 2 | UniAD | ✓ | ✗ | 0.35/0.63/0.99 | 0.66 | 0.16/0.43/1.27 | 0.62 | 1.72 | **Official** |
| 3 | UniAD | ✓ | ✓ | 0.20/0.42/0.75 | 0.46 | 0.02/0.25/0.84 | 0.37 | 1.59 | Reproduce |
| 4 | VAD-Base | ✗ | ✗ | 0.69/1.22/1.83 | **1.25** | 0.06/0.68/2.52 | 1.09 | 3.82 | Reproduce |
| 5 | VAD-Base | ✓ | ✗ | 0.41/0.70/1.06 | 0.72 | 0.04/0.43/1.15 | 0.54 | 2.72 | **Official** |
| 6 | VAD-Base | ✓ | ✓ | 0.17/0.34/0.60 | 0.37 | 0.04/0.27/0.67 | 0.33 | 2.47 | Official |
| 7 | **GoStright** (constant straight) | — | ✓ | 0.38/0.79/1.33 | **0.83** | 0.15/0.60/2.50 | 1.08 | 8.62 | — |
| 8 | **Ego-MLP** (ego status only) | — | ✓ | 0.15/0.32/0.59 | **0.35** | 0.00/0.27/0.85 | 0.37 | 2.93 | — |
| 9 | BEV-Planner* | ✗ | ✗ | 0.27/0.54/0.90 | 0.57 | 0.04/0.35/1.80 | 0.73 | 3.98 | — |
| 10 | BEV-Planner | ✗ | ✗ | 0.30/0.52/0.83 | **0.55** | 0.10/0.37/1.30 | 0.59 | 4.26 | — |
| 11 | BEV-Planner+ | ✓ | ✗ | 0.28/0.42/0.68 | 0.46 | 0.04/0.37/1.07 | 0.49 | 4.21 | — |
| 12 | BEV-Planner++ | ✓ | ✓ | 0.16/0.32/0.57 | 0.35 | 0.00/0.29/0.73 | 0.34 | 3.16 | — |

Source: `2312.03031` Table 1, p.5. † ST-P3's row is flagged by the authors for the tail-sample GT
defect. *Spelling "GoStright" is the paper's.
⭐ **Rows 7 and 8 are the floors that matter**: a hard-coded straight line (0.83) beats UniAD
without ego status (1.03), and an MLP with **no perception at all** (0.35) beats UniAD-with-ego
(0.46) and ties BEV-Planner++ (0.35).

## `paradrive-std` (oriented box, 0.1 m, first frame removed, + map compliance)

| scenario | method | ego states | Col Ave_all | **L2 Ave_all** | Offroad % | Offlane % |
|---|---|---|---:|---:|---:|---:|
| val | UniAD | No | 0.40 | 0.8317 | 0.91 | 1.74 |
| val | VAD | No | 0.30 | 0.7830 | 1.03 | 1.93 |
| val | PARA-Drive | No | 0.17 | **0.5574** | 0.12 | 0.83 |
| val | **AD-MLP** | **Yes** | 0.20 | **0.5568** | **1.21** | **2.45** |
| val | PARA-Drive+ | Yes | 0.13 | 0.4939 | 0.11 | 0.78 |
| targeted (686) | UniAD | No | 0.15 | 0.9935 | — | — |
| targeted (686) | VAD | No | 0.34 | 1.0840 | — | — |
| targeted (686) | PARA-Drive | No | **0.14** | 0.9082 | — | — |
| targeted (686) | **AD-MLP** | **Yes** | **0.94** | 0.9360 | — | — |
| targeted (686) | PARA-Drive+ | Yes | 0.05 | 0.7018 | — | — |

Source: `paradrive-cvpr2024` Table 6, p.8 (the 686-frame targeted subset = frames whose command is
not "keep forward", §4.1 p.5). ⭐ On full val the **perception-free** AD-MLP (0.5568) ties the full
perception stack PARA-Drive (0.5574); on the 686 non-straight frames its collision rate is
**6.7× worse** (0.94 vs 0.14). The benchmark cannot tell them apart; a corrected one can.

Instrument-quality rows from the same paper (Table 2, p.5), which is why no collision number under
the legacy protocols is quotable: with the axis-aligned box on the 0.5 m grid the **GT trajectory**
scores **0.384** (UniAD methodology); adding the ego's orientation → 0.32; adding the finer
discretisation → **0.00**.

## Cross-convention mixing in the published tables (read before quoting any of them)

MEASURED by re-reading the banked PDFs: **UniAD Table 7**, **VAD Table 1** and **AD-MLP Table 1**
each place UniAD's at-t numbers and ST-P3's / VAD's averaged-up-to-t numbers in ONE column, and
AD-MLP's table additionally carries VAD's **ego-status** rows (0.17/0.34/0.60) beside
ego-status-free ones. AD-MLP states the provenance plainly — *"Results in the table except for our
method are collected from VAD"* — which is exactly how a convention mismatch propagates. Our
harness makes the mix impossible by construction: every row carries `(reduction, pipeline)` and a
result object refuses two tags.

## Our own row

**NOT MEASURED — pending data.** Once the PI's download lands, `python -m taniteval.bench
nuscenes_ol …` produces one run per convention. Our row will be **vision-only, command-free** and
therefore comparable ONLY with the `Ego in BEV ✗ / Ego in Planner ✗` rows (UniAD 1.03 ·
VAD-Base 1.25 · BEV-Planner 0.55 · ST-P3 2.65 in the `bevplanner` tag) — and it will carry the
observed-mask stamp, because our 120°-trained model sees **43 %** of its own frame on nuScenes'
~64.6° CAM_FRONT.
