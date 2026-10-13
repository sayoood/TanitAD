# Untrained-head floor on all 1,058 C3 tokens (Amendment A2.4, MEASURED 2026-10-13, before any S1 arm)

An **instrument property**, published before any S1 head trains. It changes no number in `eval/PREREG_AUX_DRIVABLE_S1.md`.
Tier T0-like (perception on teacher-forced frames, never driving performance). Estimator: paired log-cluster bootstrap over **123 logs**,
10,000 resamples, seed 20260927. It answers "another draw of logs" only.

## Inputs, each verified

| input | evidence |
|---|---|
| ctx: the planner's fp32 path, u400 (`model_sft4_B_u400.pt`, md5 `8e382eeb…`) | **1,058 / 1,058 tokens bit-identical** to the C3 table's scorer logits: 473,984 values, max \|Δ\| 0.0 (`c3_full_ctx_identity.json`). The cache fill passed the identity check on every 100-token chunk (`c3_ctx_fill.log`) |
| heads: the trainer's own initial heads for seeds 0, 1, 2 (`raw/2026-10-13-aux-p3-floor/make_init_heads.py`; feed on and off share one head per seed) | sha256 in `init_heads_sha256.txt`. The head-only pass re-checked the identity on all 1,058 tokens: PASS (`identity.json`, `run.json`: 289 s on CPU, 0 trunk forwards) |
| labels | the navtest store, `store_sha256` `985a3bc7…` |
| u400 DAC-zero tokens | 416 (`floor_instrument.json`) |

The outputs `head_pred_H2-s{0,1,2}.npz` (11 MB each) stay on the dev box in `D:/Projects/TanitAD-artifacts/aux_navtest_forward/p3_floor_full/heads/`; their sha256 are in `head_pred_sha256.txt`.

## Result

| initial head | P3 pooled AUC [95 % CI] | P3 within-set mean | P1 IoU [0, 40) m | P1 IoU by band 0–20 / 20–40 / 40–72 m | P2 near-boundary SDF MAE |
|---|---|---|---|---|---|
| seed 0 | **0.659** [0.623, 0.694] | 0.491 | 0.184 [0.173, 0.195] | 0.261 / 0.077 / 0.032 | 2.27 m |
| seed 1 | **0.563** [0.516, 0.605] | 0.486 | 0.273 [0.253, 0.298] | 0.297 / 0.243 / 0.259 | 3.29 m |
| seed 2 | **0.698** [0.654, 0.737] | 0.567 | 0.134 [0.118, 0.150] | 0.094 / 0.169 / 0.133 | 2.28 m |
| u400 scorer, same proposals | **0.787** [0.731, 0.831] | 0.656 | – | – | – |

The P3 mark is max(0.90, scorer + 0.05) = 0.90. The K2 bar is P1 IoU on [0, 40) m ≥ 0.80.

## What it says

1. **The 32-token warning was a one-log artefact.** On 32 tokens from one log, the seed-0 head read 0.8245 against the scorer's 0.8005. On 1,058 tokens over 123 logs it reads **0.659**, and every initial head is **below the scorer, with separated intervals** (head − scorer: −0.128 [−0.184, −0.068], −0.224 [−0.287, −0.160], −0.089 [−0.129, −0.054]).
   - ⇒ An S1 head must climb **0.20 to 0.34 pooled AUC above its own start** to reach 0.90. The bar is not covered by a position prior.
2. **A position prior exists, and it is seed-dependent.**
   - Every pooled interval excludes 0.5, and the seeds span 0.56–0.70.
   - **Within a token the initial heads are at chance** (0.49–0.57). The pooled excess comes from ranking between tokens.
   - That is why A2 pairs each arm with **its own** initial head rather than a pooled floor.
3. **P1 and K2.** Initial IoU on [0, 40) m is 0.13–0.27 against K2's 0.80. K2 is a fit bar, not a prior bar.

## How this was run (reproducible)

`raw/2026-10-13-aux-p3-floor/README.md` "Full floor": a head-only pass of `eval/aux_navtest_forward.py` on the cached ctx (`--device cpu --no-gpu-lock`), then
`eval/aux_s1_readout.py perception` with the three initial heads as `--pred` (`floor_instrument.json`). Logs: `p3_floor_full_heads.log`, `p3_floor_full_readout.log`.
⚠️ A trained head and its floor must be read on the **same device**: CPU and GPU head passes differ by up to 1.3e-6 in logit (`raw/2026-10-13-aux-p3-floor/README.md`). These floors are CPU passes.
