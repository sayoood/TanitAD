*Target `NOT_T_seen` at the **frame** level. Held-out = the label's CONFIRM chunks: **50,119 frames**, **1,642 positives** (base rate **0.03276**), **101 of 322 clips** positive. TRAIN = 31,032 frames / 836 positives / 198 clips.*

**Fidelity check (direction 1): `all_match = True`** — this loader reproduces every one of H2's published substrate counts exactly (train_rows 31032, train_pos 836, heldout_rows 50119, heldout_pos 1642, heldout_clips 322, heldout_pos_clips 101, train_clips 198).

### The ladder — held-out, RIDGE (closed form)

| representation | role | dim | AP | AP / base | AUROC | ΔAP vs chance | CI95 | above chance? | selected λ |
|---|---|---|---|---|---|---|---|---|---|
| `ego_t` | ⭐ POSITIVE CONTROL (2 ego channels) | 2 | 0.11798 [0.07667, 0.16974] | **3.6012×** | 0.7630 | +0.07194 | [+0.04715, +0.13166] | ✅ **YES** | 0.1 |
| `ego_win` | ⭐ POSITIVE CONTROL (ego, head's window) | 16 | 0.11987 [0.08006, 0.17042] | **3.6589×** | 0.7706 | +0.07383 | [+0.04935, +0.13300] | ✅ **YES** | 0.1 |
| `img_t_SHUFFLED` | ⭐ NEGATIVE CONTROL (features permuted) | 2048 | 0.03296 [0.02513, 0.04140] | **1.0061×** | 0.4972 | -0.01308 | [-0.01730, +0.00641] | no | 0.1 |
| `img_t` | ⭐⭐ **PRIMARY** — frozen 2048-d state at t | 2048 | 0.06196 [0.04439, 0.08301] | **1.8913×** | 0.6989 | +0.01592 | [+0.00737, +0.04746] | ✅ **YES** | 3.16e+06 |
| `img_win_mean` | frozen state, 8-step window mean | 2048 | 0.04057 [0.02979, 0.05788] | **1.2384×** | 0.5887 | -0.00547 | [-0.00994, +0.02287] | no | 10 |
| `img_win_flat` | the head's exact input, read linearly | 16384 | 0.04002 [0.02943, 0.05459] | **1.2215×** | 0.5835 | -0.00602 | [-0.00975, +0.01995] | no | 100 |
| `img_pca16` | low-rank image, k=16 | 16 | 0.06212 [0.04449, 0.08351] | **1.896×** | 0.6981 | +0.01608 | [+0.00759, +0.04852] | ✅ **YES** | 3.16e+06 |
| `img_pca64` | low-rank image, k=64 | 64 | 0.06197 [0.04439, 0.08300] | **1.8914×** | 0.6989 | +0.01593 | [+0.00737, +0.04746] | ✅ **YES** | 3.16e+06 |
| `img_pca256` | low-rank image, k=256 | 256 | 0.06196 [0.04439, 0.08301] | **1.8913×** | 0.6989 | +0.01592 | [+0.00737, +0.04746] | ✅ **YES** | 3.16e+06 |
| `ego_win+img_pca16` | ego + low-rank image, k=16 | 32 | 0.12073 [0.08201, 0.17167] | **3.685×** | 0.7878 | +0.07469 | [+0.04966, +0.13269] | ✅ **YES** | 0.1 |
| `ego_win+img_pca64` | ego + low-rank image, k=64 | 80 | 0.09828 [0.06564, 0.14129] | **2.9999×** | 0.7530 | +0.05224 | [+0.03361, +0.10414] | ✅ **YES** | 0.1 |
| `ego_win+img_pca256` | ego + low-rank image, k=256 | 272 | 0.06931 [0.04702, 0.10396] | **2.1156×** | 0.7026 | +0.02327 | [+0.01289, +0.06756] | ✅ **YES** | 31.6 |
| `constant` | the chance arm itself | 0 | 0.04604 [0.02377, 0.05181] | **1.4053×** | 0.5000 | +0.00000 | [+0.00000, +0.00000] | no | — |

### The second reader — LOGISTIC (LBFGS), same split, same selection rule

| representation | AP | AP / base | ΔAP vs chance | CI95 | above chance? |
|---|---|---|---|---|---|
| `ego_t` | 0.11519 | **3.5161×** | +0.06915 | [+0.04527, +0.12827] | ✅ **YES** |
| `ego_win` | 0.11234 | **3.4288×** | +0.06630 | [+0.04495, +0.12199] | ✅ **YES** |
| `img_t_SHUFFLED` | 0.03335 | **1.0181×** | -0.01269 | [-0.01688, +0.00670] | no |
| `img_t` | 0.05867 | **1.7909×** | +0.01263 | [+0.00536, +0.04279] | ✅ **YES** |
| `img_win_mean` | 0.04128 | **1.26×** | -0.00476 | [-0.00849, +0.02337] | no |
| `img_win_flat` | 0.06347 | **1.9373×** | +0.01743 | [+0.00935, +0.04903] | ✅ **YES** |
| `img_pca16` | 0.06002 | **1.8319×** | +0.01398 | [+0.00619, +0.04584] | ✅ **YES** |
| `img_pca64` | 0.05868 | **1.791×** | +0.01264 | [+0.00536, +0.04283] | ✅ **YES** |
| `img_pca256` | 0.05868 | **1.791×** | +0.01264 | [+0.00536, +0.04279] | ✅ **YES** |
| `ego_win+img_pca16` | 0.11660 | **3.5589×** | +0.07056 | [+0.04587, +0.12938] | ✅ **YES** |
| `ego_win+img_pca64` | 0.08208 | **2.5053×** | +0.03604 | [+0.02224, +0.08116] | ✅ **YES** |
| `ego_win+img_pca256` | 0.06610 | **2.0176×** | +0.02006 | [+0.01153, +0.05904] | ✅ **YES** |

### ⭐ The comparison that is the finding — SAME target, SAME split, SAME estimator

| reader | params | AP | AP / base | ΔAP vs chance | CI95 | above chance? |
|---|---|---|---|---|---|---|
| `head_ego` — H2's attention head (INHERITED) | 415 k | 0.12263 | 3.74× | +0.07659 | [+0.05055, +0.13529] | ✅ **YES** |
| `head_img_ego` — H2's attention head (INHERITED) | 2.17 M | 0.05205 | 1.59× | +0.00601 | [-0.00040, +0.03947] | **no** |
| `head_img` — H2's attention head (INHERITED) | 2.17 M | 0.04914 | 1.5× | +0.00310 | [-0.00291, +0.04284] | **no** |
| ⭐ **LINEAR RIDGE probe on the frozen 2048-d state** (ours) | **2,049** | **0.06196** | **1.8913×** | **+0.01592** | [+0.00737, +0.04746] | ✅ **YES** |
| ⭐ LINEAR LOGISTIC probe, same input (ours) | 2,049 | 0.05867 | 1.7909× | +0.01263 | [+0.00536, +0.04279] | ✅ **YES** |

### Amendment A1 — the chance arm, both conventions

*A fully-tied constant score is ranked by ROW ORDER under a stable sort, so its full-sample AP is **0.04604** against an analytic base rate of **0.032762** — while inside a bootstrap draw `_draws` randomises the clip order, giving a median of **0.033268**. A uniform random ranker gives **0.032848**. The POINT estimate is therefore deflated under H2's convention; **the CI, which is what decides separation, is computed on the randomised draws and is unaffected.** Both are reported below.*

| representation | ΔAP vs chance (`const`, H2's convention) | CI95 | ΔAP vs chance (`rand`, unbiased) | CI95 | above chance (both)? |
|---|---|---|---|---|---|
| `ego_t` | +0.07194 | [+0.04715, +0.13166] | +0.08513 | [+0.04879, +0.13079] | ✅ **YES** |
| `ego_win` | +0.07383 | [+0.04935, +0.13300] | +0.08703 | [+0.05166, +0.13210] | ✅ **YES** |
| `img_t_SHUFFLED` | -0.01308 | [-0.01730, +0.00641] | +0.00011 | [-0.00202, +0.00243] | no |
| `img_t` | +0.01592 | [+0.00737, +0.04746] | +0.02911 | [+0.01592, +0.04639] | ✅ **YES** |
| `img_win_mean` | -0.00547 | [-0.00994, +0.02287] | +0.00772 | [+0.00097, +0.02204] | no |
| `img_win_flat` | -0.00602 | [-0.00975, +0.01995] | +0.00717 | [+0.00085, +0.01932] | no |
| `img_pca16` | +0.01608 | [+0.00759, +0.04852] | +0.02927 | [+0.01598, +0.04768] | ✅ **YES** |
| `img_pca64` | +0.01593 | [+0.00737, +0.04746] | +0.02912 | [+0.01592, +0.04639] | ✅ **YES** |
| `img_pca256` | +0.01592 | [+0.00737, +0.04746] | +0.02911 | [+0.01592, +0.04639] | ✅ **YES** |
| `ego_win+img_pca16` | +0.07469 | [+0.04966, +0.13269] | +0.08788 | [+0.05275, +0.13484] | ✅ **YES** |
| `ego_win+img_pca64` | +0.05224 | [+0.03361, +0.10414] | +0.06543 | [+0.03656, +0.10446] | ✅ **YES** |
| `ego_win+img_pca256` | +0.02327 | [+0.01289, +0.06756] | +0.03646 | [+0.01891, +0.06843] | ✅ **YES** |
| `constant` | +0.00000 | [+0.00000, +0.00000] | +0.01319 | [-0.00607, +0.01714] | no |

*PCA on the TRAIN rows of the standardised state: top-1 component carries **19.8 %** of the variance, 16 components **96.99 %**, 64 **99.94 %**, 256 **100.0000 %**. The frozen state is **extremely low-rank** — which is itself part of the answer.*
