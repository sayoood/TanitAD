# PREREG_WEATHERPROBE: the zero-training attention-pooling probe for NOT-DETECTED(weather) (STRAT1 card section 6, "Detected info")

Written 2026-11-16 BEFORE any probe was fitted or scored. Frozen card: commit 434bb876, `PREREG_V8F_STRAT1.md` section 6. Result being followed up: `RESULT_STRAT1.md` / `read/out/STRAT1_CTX.json` (weather: S1_s0 +0.0463 [-0.0108, +0.1014] not separated; S1_s1 +0.0717 [+0.0254, +0.1204] separated; the card needs both seeds, so NOT-DETECTED(weather)). Tier: T0 (a representation read at the window's NOW frame). Evidence class of every number to come: MEASURED (ours).

## 1. Question

Does weather live in S1's frozen adapter output (the 40 tokens x 4 history frames) when a learned attention pool reads it, where the shipped head (a token-mean followed by an MLP) did not separate it at both seeds? The adapter is frozen, so a probe that detects shows the information is present and the pooling was the limit; a probe that does not detect points at the representation.

## 2. Probe architecture (exactly the card's)

* Input per window (NOW frame t): the cached 4x10 pooled DINO-class tokens `pool4x10` of frames t-3 .. t, each [40, 1024] (float8_e4m3fn cache, cast to float32). Frozen S1 adapter of the arm's checkpoint: `LayerNorm(1024) -> Linear(1024, 256)`, plus the checkpoint's `pos[40]` (token position) and `time[4]` (frame slot, frame t-3 = slot 0, NOW = slot 3). Result: 160 tokens of width 256 per window. No parameter of the adapter, `pos` or `time` is trained. No ego, nav, label or metadata input of any kind (vision only).
* Probe: `LayerNorm(256)` -> `MultiheadAttention(256, 4 heads)` with ONE learned query (a [1, 256] parameter, N(0, 0.02^2)) over all 160 tokens (attention pooling over all 40 tokens and the 4 frames) -> `LayerNorm(256)` -> `Linear(256, 4)` (weather vocabulary: clear, fog, rain, snow). Cross-entropy loss, unweighted (the scored quantity is accuracy against the majority baseline).
* Trained parameters about 0.27 M. Nothing else moves.

## 3. Training set: TRAIN clips only

* The 4,572 train clips of `strat_labels_table.npz` (sha256 8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff) read from the train cache `/root/data/cache_pod/train`. The 141 eval clips are never loaded by the fit.
* Windows: per train clip, 8 NOW frames t_k = round(linspace(3, n_frames - 1, 8)), unique. Label row = `train_off[clip] + t` (the same row convention as the eval reader, `eval_off + t`). A window is kept when `train_flag == 0` and `train_weather_valid` at that row. No other filter, no class re-balancing.
* **Disjointness check (must read 0):** `|set(train_clip) & set(eval_clip)|` from the table, AND the same intersection of the clip file names actually read for the fit against the 141 eval clip names, AND the same against the eval grid's `order`. The report prints all three; any non-zero value stops the run and is a FAIL.
* A clip whose cache frame count differs from `train_n` is skipped and listed (count reported).

## 4. Fit budget, fixed before any eval number

* AdamW, lr 1e-3, weight decay 0.01, cosine decay to 0, 100 warm-up steps, batch 256, probe seed 0 (initialisation and batch order), float32 probe, deterministic data order from seed 0. No dropout, no augmentation.
* **The one hyper-parameter, the number of epochs E in {2, 4, 8}, is chosen on a TRAIN-only split:** the train clips are split by clip (seed 0 permutation, 80 % fit / 20 % held out). The probe is fit on the 80 % at each E; the E with the highest held-out (accuracy minus majority-class accuracy, windows pooled) is taken (ties: the smaller E). The final probe is then re-fit on ALL train windows with that E, once per adapter. E is chosen separately for S1_s0 and S1_s1 (each adapter is a different representation) and recorded. The eval set is never used to choose E, the architecture, or anything else.
* Eval is scored once per adapter per arm after the final fit.

## 5. Scoring: the same as `strat1_ctx_read.py`

* Eval windows: the eval grid of record (`eval_grid_windows.json`, sha256 659c3b65e364ea5de87a4e17d076eba5645df95cdcf39d8abc08caab1b2d594c, pinned in `strat1_lib.PINS`), 2,399 windows over the 141 eval episodes, NOW frame t of the grid, read from the dev-box eval cache `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval` (PRIMARY: the very cache the T0 read of record used, so probe and shipped head see the same bytes). The pod's own eval cache was re-encoded on the pod (content_md5 differs for every clip, verified before any fit) and is scored BESIDE as an encoder-source sensitivity, never as the bar. The train cache is the pod's (the one the arms trained on). Labels: `label_rows` of `strat1_ctx_read.py` (flag != 0 and invalid masked to -1; weather has no twilight mask). Constant baseline: the majority class of the table (clear, class 0), accuracy 0.8574 over the 2,399 windows as in the read.
* Statistic: accuracy of the probe minus accuracy of the constant, on the same windows, paired episode-cluster bootstrap over the 141 eval episodes, B 2,000, seed 0 (`taniteval.ci.paired_episode_cluster_bootstrap`, the read tree copy). "Another draw of EPISODES" only: it does not answer a training-run or inference-seed question, and a replicate is the sufficient form (H-ESTIM-SEED-1). The probe's own fit is one seed per adapter (seed 0); the two S1 seeds are the two replicates the bar reads.
* Beside (no bar): balanced accuracy, per-class recall, confusion, and the shipped head's own numbers from `STRAT1_CTX.json` for the same arm.

## 6. The bar

* **DETECTS(weather) by the probe** at an S1 seed = probe accuracy minus constant > 0 and separated, AND that seed's regression arm (section 7) reads NOT DETECTS.
* **Label: DETECTS(weather)-by-probe only when both S1_s0 and S1_s1 are DETECTS (as the read's rule). Otherwise NOT-DETECTED-by-probe.** A detect at one seed only is reported as such and counts as NOT-DETECTED-by-probe (the card's rule needs both seeds).
* Bars do not change after any number is seen.

## 7. Controls (each must read its value or the probe is INSTRUMENT-FAIL and nothing is quoted)

1. **Deliberate-regression arm R1 (labels shuffled):** the same probe, same E, same seed, trained on train labels permuted between train CLIPS (a seed-0 permutation of the clips: each clip's windows take the labels of the clip at the permuted index, taking rows by position modulo that clip's window count). Scored on the real eval labels. Must read NOT DETECTS (accuracy minus constant not > 0 separated) at each adapter. Run for S1_s0 and S1_s1.
2. **Random-adapter arm R2:** the same probe on an adapter whose LayerNorm/Linear/pos/time are re-initialised from seed 0 (`nn.Linear` default init, `pos`/`time` N(0, 0.02^2)), E as the real arm of S1_s0, one fit. Reported beside (random projections of the same features can carry weather too, so this is NOT a bar: it measures how much of any detection is the trained adapter versus the cache features). The cached DINO-class features carry the signal by construction; this arm is what makes that visible.
3. **Instrument identity:** the stored ctx head of each checkpoint (`ctx_head`, GELU MLP on the token mean of NOW) applied by the probe script's own adapter code to the eval windows must reproduce the T0 `ctx_logits` weather argmax of `eval_STRAT1-S1_s{0,1}_real.npz` on at least 99 % of the 2,399 windows (fp16 vs fp32 tolerance), or the extraction code is wrong and the run is INSTRUMENT-FAIL.
4. **Constant predictor** reads exactly the constant baseline (delta 0).

## 8. Outcomes and their next levers (from the card, no new ones)

| outcome | next |
|---|---|
| DETECTS at both seeds, R1 NOT DETECTS | the next arm is that head: an attention-pooling weather head on the adapter tokens (S1 plus this head; a new card, a PI go for training) |
| NOT-DETECTED-by-probe (either or both seeds) | the representation lever: the 8x20 grid (`pool8x20`, 160 tokens per frame, cached) or the 416x1024 encode (a new card, a PI go) |
| R1 reads DETECTS, a control fails | INSTRUMENT-FAIL: no probe number is quoted; the fault is fixed and the run repeated |

Per RULE ZERO a NOT-DETECTED result is reported as written and the representation-lever card is drafted in the same turn.

## 9. Compute and disk

* Fit and eval run on the LoRA pod GPU (inference-only token extraction plus a 0.27 M-parameter probe; one process, under 2 GB of GPU memory, the train windows stay in host RAM as float8). Forbidden paths untouched: `/root/ops/lp3_sampler.py`, `/root/ops/bank_ckpts.py`, `/root/data/cache_10k`, `/root/datascale`. Scoring (numpy only, B 2,000) runs on the dev box CPU. Temporary pod data goes under `/root/wprobe/` and is deleted at the end.
