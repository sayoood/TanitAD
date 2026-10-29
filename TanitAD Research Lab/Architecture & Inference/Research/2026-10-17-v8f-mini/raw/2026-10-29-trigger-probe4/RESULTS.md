# RESULTS -- trigger probe 4: does a SPATIAL readout (attention / max pooling over tokens) recover the go/stay trigger from frozen DINOv3 fine tokens where flat readouts did not?

**STATUS: FINAL, 2026-10-29 ~06:50 (Master Mind finalisation from `raw/tables4.txt`, `raw/VERDICT.json` and `raw/results_strata4_<pop>.json`; the chain `chain4.sh` ran 05:26-05:35 after the five probe-3 markers appeared, every stage rc 0, `raw/CHAIN4.done`).**
The design is frozen in `PREREG_FROZEN.md` (sha256 `a8f5d785...327f`, 2026-10-29T04:48:11+01:00, `FROZEN_SHA.txt`), before any real-label fit; one post-freeze edit (`POSTFREEZE_CHANGES.md`) adds a recording-only input check and changes no design element.
**Evidence class: MEASURED. Tier: T0-class probe of representations on logged inputs, NOT driving performance.**
Estimator: episode-cluster bootstrap, B = 2,000, seed 0, on the mean of per-fold AUCs (pooled beside), paired on the same windows; it answers another draw of EPISODES. The training variance of the readout is answered by its companion, the 3-seed spread (H-ESTIM-SEED-1).
Same 12,000 + 12,000 TRAIN windows, folds and resamples as probes 1-3; tokens = probe 3's per-token PCA-16 S8 (8x20) / S4 (4x10), fold-proper (fit f for outer fold f). Input check: probe 3's banked G4+E / G8+E re-score to probe 3's own fold-mean AUC with abs diff 0.0, both populations.

## Verdict (committed rule, PREREG 3): **DOES NOT HELP**
All four decision dAUCs are below 0.01, all controls pass, no cell is missing.

| cell | dAUC (fold-mean) [CI] | 2 x seed spread | controls | status |
|---|---|---|---|---|
| A P-STOP: G8att+E - V+E | **-0.018 [-0.039, +0.002]** | 0.021 | PASS | OK, < 0.01 |
| A P-MOVE: G8att+E - V+E | **-0.004 [-0.010, +0.002]** | 0.007 | PASS | OK, < 0.01 |
| B P-STOP: G8att+E - G4att+E | **-0.001 [-0.016, +0.014]** | 0.053 | PASS | OK, < 0.01 |
| B P-MOVE: G8att+E - G4att+E | **+0.002 [-0.003, +0.007]** | 0.007 | PASS | OK, < 0.01 |

* **SPATIAL READOUT HELPS: no.** The attention readout over 160 fine tokens comes back to probe 1's whole-vector PCA-64 readout (V+E) and does not pass it, in either population.
* **RESOLUTION HELPS UNDER A SPATIAL READOUT: no.** 8x20 vs 4x10 under the same attention readout is within +-0.002 of zero on P-STOP and P-MOVE.

## Arm table (fold-mean AUC [CI]; readout arms: 3-seed logit mean, then per-seed fold-mean AUCs and spread)
| arm | P-STOP | P-MOVE |
|---|---|---|
| E | 0.626 [0.613, 0.640] | 0.732 [0.719, 0.745] |
| V+E (probe 1, whole-vector PCA-64) | 0.680 [0.650, 0.710] | 0.785 [0.771, 0.798] |
| P3:G4+E / P3:G8+E (flat per-token, probe 3) | 0.650 / 0.609 | 0.771 / 0.765 |
| G8att+E | 0.662 [0.632, 0.691]; seeds 0.654 / 0.6615 / 0.6511, spread 0.0104 | 0.781 [0.768, 0.795]; seeds 0.7739 / 0.7759 / 0.7724, spread 0.0035 |
| G4att+E | 0.663 [0.633, 0.692]; seeds 0.6662 / 0.6613 / 0.6398, spread 0.0264 | 0.779 [0.766, 0.793]; seeds 0.7744 / 0.771 / 0.7709, spread 0.0035 |
| G8max+E | 0.678 [0.648, 0.706]; spread 0.0047 | 0.786 [0.773, 0.798]; spread 0.0044 |
| G4max+E | 0.669 [0.640, 0.697]; spread 0.0019 | 0.779 [0.766, 0.793]; spread 0.0022 |

Beside: G8att+E - E +0.036 [+0.005, +0.066] (P-STOP), +0.049 [+0.038, +0.062] (P-MOVE); the attention readout recovers what the flat per-token readout lost at 8x20 (G8att+E - P3:G8+E +0.053 [+0.029, +0.077] P-STOP, +0.016 [+0.009, +0.024] P-MOVE), and stops at V+E. G8max+E - V+E -0.002 [-0.024, +0.019] / +0.001 [-0.005, +0.007]; G8max+E - G4max+E +0.009 [-0.006, +0.025] / +0.006 [+0.002, +0.011] (separated on P-MOVE, far below +0.03).

## Controls (all PASS)
| control | P-STOP | P-MOVE |
|---|---|---|
| C1 label shuffle G8att+E | 0.5083 [0.4976, 0.5183] | 0.4950 [0.4846, 0.5057] |
| C1 label shuffle G4att+E | 0.5041 [0.4934, 0.5152] | 0.4932 [0.4831, 0.5041] |
| C2 donor labels (seed 2) G8att+E / G4att+E | 0.5093 [0.4698, 0.5488] / 0.4814 [0.4378, 0.5250] | 0.4974 [0.4767, 0.5170] / 0.5058 [0.4850, 0.5282] |
| paired null G8att+E - G4att+E (shuffled) | +0.0042 [-0.0053, +0.0136] | +0.0018 [-0.0099, +0.0123] |

Synthetic known answers (`raw/pytest_output.txt`, through the real training code): a signal planted in token (6,10) is read at AUC 0.973, with 0.951 and 0.880 of two queries' attention mass on that token; replacing the softmax by uniform attention drops it to 0.568 (the attention path is the live carrier); R-MAX reads a planted amplitude at 0.758; the null reads 0.516; a G8 request with the S4 file raises. The "some query >= 0.5" reading of the brief's ">= 50 % of attention mass" was written into the PREREG before the freeze (the four-query mean is 0.46).

## Beside (descriptive, never decisive)
**Strata (TRAIN, fold-mean, paired; `raw/results_strata4_<pop>.json`):**

| pop | stratum | n / clips | E | V+E | G8att+E | G8att+E - E | G8att+E - V+E |
|---|---|---|---|---|---|---|---|
| P-STOP | TL_RED | 4986 / 256 | 0.609 | 0.708 | 0.686 | +0.077 [+0.019, +0.128] | -0.022 [-0.063, +0.019] |
| P-STOP | TL_GREEN | 1377 / 123 | 0.594 | 0.536 | 0.558 | -0.036 [-0.110, +0.045] | +0.022 [-0.061, +0.081] |
| P-STOP | NO_TL | 5348 / 451 | 0.621 | 0.682 | 0.668 | +0.048 [+0.002, +0.087] | -0.014 [-0.040, +0.014] |
| P-STOP | lead_depart | 1741 / 310 | 0.663 | 0.644 | 0.633 | -0.030 [-0.115, +0.034] | -0.011 [-0.068, +0.038] |
| P-MOVE | TL_RED | 1233 / 302 | 0.876 | 0.879 | 0.874 | -0.002 [-0.026, +0.022] | -0.005 [-0.020, +0.009] |
| P-MOVE | TL_GREEN | 1168 / 221 | 0.729 | 0.735 | 0.737 | +0.008 [-0.042, +0.062] | +0.002 [-0.025, +0.030] |
| P-MOVE | NO_TL | 9430 / 1587 | 0.704 | 0.772 | 0.768 | +0.064 [+0.050, +0.078] | -0.004 [-0.011, +0.004] |
| P-MOVE | lead_depart | 2010 / 641 | 0.739 | 0.749 | 0.745 | +0.006 [-0.024, +0.037] | -0.004 [-0.019, +0.011] |

* The green-light gap stays: on P-STOP TL_GREEN windows no image readout beats ego alone (G8att+E - E -0.036 [-0.110, +0.045]); the flat readouts were below E there (probe 3: G4+E - E -0.116). The image signal that exists is red-light and no-light context, and none of it reaches lead departure.
* **Attention maps** (`raw/attn_maps_<pop>_G8att.png`, `maps4_<pop>_descriptors.json`): the learned attention is close to uniform: normalised entropy 0.988 (P-STOP, GO and STAY alike), largest token share 0.015 against a uniform 1/160 = 0.006, lower-half share 0.60. On the planted known answer the same readout puts 0.95 on one token. **No single location of the frozen field carries a trigger the readout could find.**
* **Eval-141, no refit** (small): P-STOP n 101 / 19 episodes: G8att+E - E -0.052 [-0.193, +0.132]; P-MOVE n 507 / 67: +0.034 [-0.034, +0.098]; G8att+E - G4att+E -0.020 / +0.005, both CIs cover 0.

## What it implies (with probes 1-3)
* **The frozen single-frame DINOv3-L/16 field caps the go/stay trigger at about V+E**: five readout families (whole-vector PCA-64, per-token PCA-16 linear and MLP, two-stage low-rank, learned attention, max-pool) and two grids (4x10, 8x20) all land at or below V+E (P-STOP 0.680, P-MOVE 0.785), against ego alone 0.626 / 0.732. Frame differences carried nothing (probe 2, NO-MOTION). The readout lever and the 8x20 resolution lever are both exhausted on this field.
* **A 16x40 re-encode on box 2 is not supported by this evidence:** neither 8x20 nor a readout able to localise shows any gain, and the attention map is near uniform. What remains untested sits upstream of the pooled cache: the encoder at native patch resolution on a crop of the traffic-light / lead region, a different or fine-tuned encoder, or a dedicated traffic-light-state input. Each is a perception architecture change (and GPU work), so each is a PI decision.
* **Limits (PREREG 7):** one attention and one max readout over a 1024 -> 16 unsupervised per-token compression; the 3-seed logit mean is compared against a single-fit V+E; nothing here speaks for 1024-d tokens, larger heads, 16x40 or native resolution.

## Provenance
* Package `E:/Projects/TanitAD-artifacts/dagland/trigger_probe4/` (code `p4_lib.py`, `p4_run.py`, `chain4.sh`; tests `test_trigger_probe4.py`, 21 passed before the run, re-run in the chain: `raw/pytest_output_final.txt`); `MANIFEST.txt` with sha256.
* Held-out attention weights: `F:/TanitAD-scratch/trigger_probe4_scratch/` (single copy on F:).
* The TRAIN cache behind S8 / S4 is UNVERIFIED byte-for-byte against box 2 (as in probes 1-3).
