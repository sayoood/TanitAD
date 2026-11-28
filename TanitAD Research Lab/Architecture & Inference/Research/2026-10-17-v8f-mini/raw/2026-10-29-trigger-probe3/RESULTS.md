# RESULTS -- trigger probe 3: does a FINER image grid (8x20 vs 4x10) carry the go/stay trigger?  (H-CRAWL-06 precursor, continued)

**STATUS: FINAL, 2026-10-29 ~06:20 (Master Mind finalisation of the agent's interim RESULTS; every `raw/CHAIN_*.done` marker present, `tables_g.py` / `tables_pf.py` re-run, `MANIFEST.txt` 118 files, pytest 16 passed).**
Zero training of the planner, dev-box CPU. Card frozen first: `PREREG_FROZEN.md` (sha256 `f73b8a5a...5728`, 2026-10-29T03:37:08+01:00, `FROZEN_SHA.txt`).
**Evidence class: MEASURED** (artifacts in `raw/`; every number is printed by `tables_g.py` -> `raw/tables3.txt`, or `tables_pf.py` -> `raw/tables_pf.txt`, or sits in `raw/VERDICT.json`). **Tier: T0-class** probe of representations on logged inputs, NOT driving performance.
**Estimator:** clip(episode)-cluster bootstrap, B = 2,000, seed 0, on the **mean of per-fold AUCs** (pooled beside), paired on the same windows. It answers "another draw of EPISODES"; it is blind to training variance (H-ESTIM-SEED-1) and to the probe-fit draw (inner-CV C, MLP init). No model is trained here except the probes; the encoder is the one frozen DINOv3-L/16.
Same 12,000 + 12,000 TRAIN windows, folds and bootstrap resamples as probes 1 and 2 (windows_train sha256 asserted `a9a57b32...5783ec`; E arm and V+E are probe 1's banked out-of-fold scores).

## 1. Headline
* **Verdict by the committed rule (PREREG 4): INCONCLUSIVE.** Not because any cell points toward a gain: **all four decision cells are negative, and in none does the CI's upper bound reach +0.01** (largest upper bound +0.0023). The verdict cannot be "DOES NOT HELP" because the rule needs all four cells with passing controls, and both P-MOVE cells are excluded: the P-MOVE **G8+E label-shuffle control C1 reads 0.5120 [0.5017, 0.5226]**, which misses 0.5 by 0.0017 (section 3).
* **HELPS is unreachable in every cell**, valid or excluded: dAUC(G8+E - G4+E) **P-STOP linear -0.042 [-0.060, -0.022]**, **P-STOP MLP -0.022 [-0.049, +0.002]**, P-MOVE linear -0.006 [-0.013, +0.000] (excluded), P-MOVE MLP -0.030 [-0.042, -0.019] (excluded).
* **The finer grid is worse with a flat readout**, separated on P-STOP linear (and on the excluded P-MOVE MLP cell). The content mutation (C5b) shows it is the fine content itself that costs: the same 8x20 shape filled with the nearest-neighbour upsample of G4 scores 0.641 against real G8+E 0.609 (G8+E - G8mut+E -0.032 [-0.048, -0.016], P-STOP). Under one L2 penalty, 4x more decorrelated features add variance faster than signal.
* **The readout, not the resolution, is the bottleneck.** The per-token PCA-16 readout is below probe 1's whole-vector PCA-64 even at 4x10: G4+E - V+E -0.030 [-0.049, -0.010] (P-STOP), -0.014 [-0.020, -0.008] (P-MOVE). The best post-freeze flat readouts (section 6) only come back to V+E: they never beat it, at either grid.
* **The green-light failure is not a resolution problem under this readout:** on P-STOP TL_GREEN windows the images still HURT (G4+E - E -0.116 [-0.191, -0.005]), and the 8x20 grid changes nothing there (G8+E - G4+E -0.000 [-0.072, +0.073]).

## 2. Decision cells (TRAIN; fold-mean AUC [95 % CI]; dAUC paired)
| cell | G4+E | G8+E | dAUC(G8+E - G4+E) | meets HELPS (>= +0.03, CI > 0) | < 0.01 | controls |
|---|---|---|---|---|---|---|
| P-STOP linear | 0.650 [0.618, 0.684] | 0.609 [0.574, 0.644] | **-0.042 [-0.060, -0.022]** | no | yes | PASS |
| P-STOP MLP | 0.631 [0.597, 0.666] | 0.609 [0.572, 0.645] | **-0.022 [-0.049, +0.002]** | no | yes | PASS |
| P-MOVE linear | 0.771 [0.757, 0.785] | 0.765 [0.751, 0.779] | -0.006 [-0.013, +0.000] | no | yes | **FAILED-CONTROL** (G8+E C1) |
| P-MOVE MLP | 0.760 [0.746, 0.775] | 0.730 [0.713, 0.745] | -0.030 [-0.042, -0.019] | no | yes | **excluded with the population** (its own MLP C1 passes; see section 3) |

Context arms, linear fold-mean: P-STOP E 0.626, V+E 0.680, G4 0.632, G8 0.591; P-MOVE E 0.732, V+E 0.785, G4 0.682, G8 0.695. MLP fold-mean: P-STOP E 0.628, V+E 0.681, G4 0.607, G8 0.593; P-MOVE E 0.771, V+E 0.786, G4 0.678, G8 0.665.
Inner-CV C: P-STOP picked the grid minimum 1e-4 in 4 of 5 folds for G4, G8, G4+E and G8+E (the optimum on the edge of the frozen grid); P-MOVE picked 1e-3 in all folds (interior). The post-freeze extended grid (section 6) tests the edge: PF:G8+E 0.603 vs frozen 0.609 (P-STOP), so the edge did not hide a gain.

## 3. Controls (PREREG 6)
* **C4 G4 == mean2x2(G8): PASS, exact.** 64 rows checked at extraction (32 per population), max abs diff 0.0 against the independent loop implementation; pytest (16 passed, `raw/pytest_output.txt`) repeats it on random data and shows the check fails for a wrong pooling.
* **Alignment/continuity gate: PASS.** All 24,000 rows: relative L2 of G4 vs probe 1's cached pool4x10: median 0.0303, p99.9 0.0317, max 0.0319 (P-STOP) / 0.0303, 0.0315, 0.0319 (P-MOVE). No row above 0.15.
* **P-STOP: all PASS.** C1 shuffle G8+E 0.4988 [0.4882, 0.5088], G4+E 0.4953 [0.4857, 0.5052], paired null +0.0035 [-0.0065, +0.0135]; C3 leak 1.0000 (both); C2 donor labels: G4+E seed 2 0.5285 [0.4857, 0.5692] covers 0.5; G8+E seed 2 0.5426 [0.5012, 0.5839] missed 0.5, so the pre-registered seeds 3-6 band ran: band mean **0.5120**, inside [0.47, 0.53] -> PASS. MLP C1: G4 0.5003 [0.4890, 0.5110], G8 0.5031 PASS.
* **P-MOVE: G8+E C1 FAILS.** C1 shuffle **G8+E 0.5120 [0.5017, 0.5226]** (misses 0.5 by 0.0017); G4+E 0.5054 [0.4950, 0.5158] PASS; paired null +0.0066 [-0.0049, +0.0179] PASS; C2 G4+E 0.5071 [0.4866, 0.5276], G8+E 0.5007 [0.4825, 0.5193] PASS; C3 1.0000 PASS; MLP C1 G4 0.4971 [0.4872, 0.5075], G8 0.5050 [0.4950, 0.5149] PASS.
  **Reading, disclosed:** PREREG 6 says a failed control excludes "the arm it guards". `tables_g.py` applies the population's linear control state to BOTH P-MOVE cells; under the narrower reading the P-MOVE MLP cell (its own MLP C1 passes) would stand at -0.030 [-0.042, -0.019]. **The verdict is INCONCLUSIVE under both readings**, because the P-MOVE linear cell is excluded either way. The C1 miss is small (a single label permutation over 2,568 features); PREREG 6 pins no band for C1, so none is applied after the fact.
* **C5(a) PASS** (pytest: the 4x10 file raises at the G8 entry point). **C5(b) content mutation PASS** (the numbers change, `|diff| > 1e-4`: G8mut+E 0.641 [0.607, 0.675] vs G8+E 0.609 [0.574, 0.644], P-STOP).
* **C6 planted known answer PASS** (pytest: planted +s/-s on two adjacent 8x20 tokens: G8+E reads > 0.9, G4+E 0.5 +- CI, dAUC CI excludes 0; null: dAUC CI covers 0). **C7 PASS:** controls run for every source name (the probe-1 `if src == "V"` trap is closed).

## 4. Beside, never decisive
**Strata (TRAIN, linear, fold-mean; `raw/tables3.txt` STRATA):**

| pop | stratum | n / clips | E | V+E | G4+E | G8+E | G4+E - E | G8+E - G4+E |
|---|---|---|---|---|---|---|---|---|
| P-STOP | TL_RED | 4986 / 256 | 0.609 | 0.708 | 0.678 | 0.627 | +0.068 [+0.013, +0.119] | -0.051 [-0.086, -0.016] |
| P-STOP | TL_GREEN | 1377 / 123 | 0.594 | 0.536 | 0.478 | 0.477 | -0.116 [-0.191, -0.005] | -0.000 [-0.072, +0.073] |
| P-STOP | NO_TL | 5348 / 451 | 0.621 | 0.682 | 0.664 | 0.626 | +0.044 [-0.007, +0.087] | -0.038 [-0.067, -0.009] |
| P-STOP | lead_depart | 1741 / 310 | 0.663 | 0.644 | 0.593 | 0.589 | -0.070 [-0.168, +0.011] | -0.004 [-0.053, +0.052] |
| P-MOVE | TL_RED | 1233 / 302 | 0.876 | 0.879 | 0.870 | 0.842 | -0.006 [-0.034, +0.019] | -0.028 [-0.042, -0.013] |
| P-MOVE | TL_GREEN | 1168 / 221 | 0.729 | 0.735 | 0.725 | 0.704 | -0.004 [-0.058, +0.056] | -0.021 [-0.059, +0.016] |
| P-MOVE | NO_TL | 9430 / 1587 | 0.704 | 0.772 | 0.760 | 0.758 | +0.056 [+0.041, +0.072] | -0.002 [-0.010, +0.005] |
| P-MOVE | lead_depart | 2010 / 641 | 0.739 | 0.749 | 0.725 | 0.723 | -0.013 [-0.050, +0.022] | -0.002 [-0.019, +0.015] |

TL_YELLOW is too small in both populations (n 164 / 86). The TL strata are clip-level VLM labels (`vlm-cot`), untimed: coarse.
* The pattern of probe 1 holds at both grids: the image features help where a red light makes the human stay, and **hurt where a green light makes the human go** (P-STOP TL_GREEN, GO share 0.886). A flat readout of pooled tokens appears to learn "a traffic light is in view => stay" without the colour. On lead departure the images add nothing at either grid.
* **Spatial readout map** (`readout_heatmap_G8.png`, descriptive, not an attribution): the G8+E weight is spread, not localised: top-1 token share 0.0191 (P-STOP) / 0.0192 (P-MOVE), top-10 % tokens share 0.217 / 0.195, lower-half share 0.472 / 0.542.
* **Eval-141, no refit** (small: P-STOP n 101 windows / 19 episodes; P-MOVE n 507 / 67): G8+E - G4+E linear -0.058 [-0.165, +0.032] / -0.001 [-0.033, +0.035]; MLP +0.120 [-0.090, +0.297] / +0.005 [-0.051, +0.060]. No eval-141 interval excludes 0 for the decision contrast.

## 5. What it implies
* **For v8F's perception lever:** pooling the frozen DINOv3 field at 4x10 instead of 8x20 is not where the go/stay trigger is lost, for a flat per-token linear or 64-unit-MLP readout. More tokens under a flat readout cost AUC (C5b shows the fine content itself costs, through variance). The readout is the binding term: at equal grids a whole-vector low-rank readout beats per-token PCA-16.
* **A 16x40 box-2 probe under the same flat readout is NOT worth the 1-1.5 h of box-2 time:** it would answer "does a flat readout drown in tokens", which this probe answered. It becomes worth running only with a readout that can use fine tokens, under its own pre-registration.
* **Next lever (Rule Zero), already running:** trigger probe 4 (`dagland/trigger_probe4/`, frozen 2026-10-29T04:48:11+01:00, sha256 `a8f5d785...327f`) reads the same S8 / S4 tokens through a learned **attention** readout and a **max-pool** readout, with 3 readout seeds per arm, against V+E. Its chain started when this probe's markers appeared.
* **The green-light result names the specific gap:** whatever reads the image must resolve a small object's state (light colour), which a pooled-token flat readout cannot. If probe 4's attention readout also fails on TL_GREEN, the gap is in the encoder field itself (pooling before the readout destroys it), and the lever moves to a crop / detector input of the traffic light or a finer encoder field, which is an architecture decision for the PI.

## 6. POST-FREEZE sensitivity arms (NOT in PREREG_FROZEN.md; cannot move a verdict; `postfreeze_g.py`, `raw/tables_pf.txt`)
Motivation: the grid-edge C and one C shared by the 8 ego numbers and 640 / 2,560 image features. Extended grid {1e-6..10}. Fold-mean AUC (pooled beside in the file):

| arm | P-STOP | P-MOVE |
|---|---|---|
| E | 0.626 | 0.732 |
| V+E (probe 1) | 0.680 | 0.785 |
| PF:G4+E / PF:G8+E | 0.650 / 0.603 | 0.771 / 0.765 |
| PF:G4k4+E / PF:G8k4+E (first 4 components per token) | 0.676 / 0.656 | 0.767 / 0.768 |
| PF:G4w64+E / PF:G8w64+E (two-stage PCA-64) | 0.668 / 0.660 | 0.776 / 0.779 |
| PF:G4+Ex64 / PF:G8+Ex64 (E replicated 64x) | 0.680 / 0.669 | 0.780 / 0.785 |

Paired 8x20 - 4x10 at each readout: P-STOP -0.0468 [-0.0663, -0.0290], -0.0197 [-0.0341, -0.0037] (k4), -0.0080 [-0.0202, +0.0042] (w64), -0.0105 [-0.0198, -0.0013] (x64); P-MOVE -0.0062 [-0.0125, +0.0002], +0.0011 [-0.0043, +0.0058] (k4), +0.0031 [-0.0014, +0.0074] (w64), **+0.0047 [+0.0008, +0.0085]** (x64, the only separated positive, far below +0.03). The best flat readouts only reach V+E (PF:G8+Ex64 0.785 = V+E 0.785, P-MOVE); none exceeds it.
The w64 arms chose C = 1e-6 (near-centroid); their pooled P-STOP AUC is 0.49 because per-fold intercepts are not comparable at that C, which is why fold-mean is primary.

## 7. Post-freeze choices (cannot move a verdict) and deviations
1. **RAM guard:** the box had 2.2 GB available while the extraction ran (its output memmap pushed this process to a 6.3 GB working set; the PI's other processes held about 20 GB). The PCA step was therefore rewritten to stream sufficient statistics instead of holding the 192k-token subsample (RSS < 1 GB, identical value, pytest `test_token_stats_pca_equals_direct_pca`), and its RAM floor was 1.5 GB instead of 3 GB for that streaming step only. Every probe-fit arm waited for >= 3 GB free. The extraction (a memmap writer) itself started at 5.4 GB free; by the end of the fill the box showed 0.36 GB free / 0.44 GB available and about 6,000 page-ins per second: it finished (rc 0) and released the memory, but it squeezed the box for about 5 minutes. Later steps use plain buffered reads (`NpyRows`) instead of mmap.
2. The PCA is an exact covariance eigendecomposition of the token subsample (the brief allowed IncrementalPCA): 16 token positions per window, about 153k tokens per outer fold, 192,000 for the all-rows fit, for each grid; explained variance of the 16 components 0.570 (G8) / 0.643 (G4) of token variance (P-STOP, all rows).
3. Chains ran side by side, detached by WMI, using up to about 12 BLAS threads (above the 4-thread house rule; disclosed by the agent). The first chain launched from the tool session was stopped and restarted; `lin_pstop` exited rc 127 once (03:57) and was re-run to rc 0 (resumable; one partly finished arm recomputed).
4. The grid {1e-4..1} replaced probe 1's {1e-3..10} (PREREG 3); the post-freeze extended grid checks it.
5. Finalisation by the Master Mind: `tables_g.py` and `tables_pf.py` re-run after the last marker (outputs `raw/tables_g_final.out`, `raw/tables_pf_final.out`); `make_manifest.py` (118 files, `MANIFEST.txt`, `raw/INPUTS_SHA256.json`); pytest 16 passed. The interim text of sections 1-5 was rewritten from `raw/tables3.txt`; no number was taken from the interim text.

## 8. Provenance and what is UNVERIFIED
* Cache: `F:/TanitAD-scratch/speedread/cache/train/clips` (4,572 clips; T == manifest n_frames asserted for every clip read). UNVERIFIED byte-for-byte against box 2's `D:\tanitad\cache_full` (as in probes 1-2).
* `G8_<pop>.npy` fp16 on F: is lossless w.r.t. the fp8 cache; sha256 in `raw/G8_sha256.txt` and `MANIFEST.txt`. The G8 / S8 / S4 arrays (about 9.5 GB) stay on F: (`F:/TanitAD-scratch/trigger_probe3_scratch/`).
* The G4 used here is the mean of the fp8-derived G8 tokens, NOT the cached pool4x10 (which is only the continuity reference), so G4+E differs from probe 1's V+E in readout, not in encoder.
* Everything here is one fixed encoder and one fixed per-token readout family; 16x40 is not tested; the TL strata are clip-level VLM labels.
