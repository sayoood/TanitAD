# v8F-mini hierarchy attention: what S, T and O look at (arm M2b, seeds 0 and 1)

Date 2026-10-25. CPU only on the dev box; box 2 was READ ONLY (scp of checkpoints, eval cache clips, small inputs; nothing run there, nothing written there).
Tier **T0 open-loop** everywhere (the true past is fed, nothing the model outputs feeds back). Qualitative figures and aggregate statistics; this is **not an eval** and reports no result row.
All paths below are relative to `OUT = C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz/out`.

## 0. Clip-selection rule (declared and executed BEFORE any render; `select_clips.py` -> `selection.json`)

Seed 0, banked own-code ADE6 (`eval_M2b_s0-kind-own.npz['ade6']`, 141 episodes, 2,399 windows), strata from `window_strata_eval141.npz`:
per stratum the **lower-median** window (sort by (ade6, window index), position (n-1)//2); plus the **single worst** window of all 2,399; the **whole clip** of each is rendered; nothing swapped after viewing.

| row | stratum (n windows) | eval window | clip | t | own-code ADE6 seed 0 / seed 1 (banked) |
|---|---|---|---|---|---|
| JUNCTION median | JUNCTION (298) | #1131 | 79e30ac2 | 7.0 s | 2.66 / 1.58 m |
| CURVE median | CURVE (140) | #654 | 45aea78c | 6.2 s | 2.72 / 5.37 m |
| NO-TURN median | NO-TURN (1,961) | #763 | 51c01ba4 | 11.8 s | 1.39 / 2.15 m |
| FAILURE (worst of 2,399) | CURVE | #2158 | ed87040c | 11.8 s | 16.95 / 16.90 m |

The four windows fall in four different clips, so no substitution rule was needed. The failure case stays in the video as it is.

## 1. Method

* **Instrument** (`attn_extract.py`). Each Reader (`read_s`, `read_t`, `read_o`) is a learned-query `nn.TransformerDecoder` (norm_first); its cross-attention reads `mem = [cond tokens ; scene tokens]`.
  S cond = (ego, nav, lim) 3 tokens; T = (ego, nav, lim, ck) 4; O = (ego, nav, lim, ck, code, age) 6; scene = 4 history frames x 40 tokens, index `h*40 + r*10 + c`, h = 3 is the Reader's tick frame. Depths S 1, T 1, O 2; 4 heads; 4 queries.
  **Tap mode `exact` (default):** the original `multihead_attn.forward` runs UNCHANGED (need_weights=False, SDPA path) and the weights are recomputed beside it from the same query/key/projection and padding mask. So the hooked forward is bit-identical to the unhooked one. The alternative `need_weights` mode (torch's own weights) is kept as a cross-check. Every layer and head is stored (`work/stats/chunks`, 141 files).
  **Plotted aggregation (declared): mean over the 4 heads, the 4 queries and the Reader's layers** (S, T one layer; O two). Per-layer numbers are in `fig_attention_stats.json` (`per_layer_cond_mass`).
* **Token geometry** (measured from code, not assumed). `pull_encode_pooled.py:259` pools the 16x40 DINOv3 patch field with `reshape(T,4,4,10,4,D).mean((2,4))`, so token (r, c) is the **64x64 px block** y in [64r, 64r+64), x in [64c, 64c+64) of the 256x640 cylindrical frame; row 0 = top (sky / far), row 3 = bottom (bonnet / near). Cache step j = native frame 2j (`dinov3_fp8_encode_ship.py:117`).
* **Sample-and-hold.** Each level is drawn on the frame it last ticked at (S at t - a_s, T at t - a_t, O at t). Banked eval windows use the eval's own cycling ages (`a_t = i%3`, `a_s = a_t + (i//3)%5`). The **video** evaluates every cache step t = 3..70 with a physical schedule instead (S ticks every 5 steps, T every 2.5 steps; `viz_data.py`), so its per-step numbers differ from the banked window numbers (see 4.4). Both lie in the trained age range.
* **Data path.** Reuses the eval's own loader (`v8f_mini_data`: grid, join, `ClipStore`, `V8FMiniWindows`), the box-2 `taniteval` modules (copied to `deps/taniteval_b2`, read-only), the checkpoints M2b_s0 (sha256 prefix 601cb435 matches the given one) and M2b_s1 (prefix 6104fb15 matches). The 141-clip eval cache (2.8 GB) was **streamed clip by clip** (scp, process, delete), so the disk footprint stayed ~0.5 GB.
* **Statistics.** All 2,399 windows, both seeds. CI = **episode-cluster bootstrap** (141 episodes resampled, window-weighted mean, 2,000 draws, 95 % percentile). It answers "would another draw of EPISODES say this?" only; "would another TRAINING RUN say this?" is answered by the second seed, shown beside every number (the paired bootstrap is blind to training variance, `H-ESTIM-SEED-1`).

## 2. Controls (all MEASURED; `controls.json`, `align_control.json`, `control_f64.json`)

| # | control | result |
|---|---|---|
| 1a | hooked vs unhooked `traj60`, all 2,399 windows x 2 seeds | **max abs diff = 0.0 exactly** (exact mode; bit-identical). The `need_weights` mode gives 1.526e-5 = 2^-16 = one float32 ulp at |x| in [128, 256) m, i.e. just above the 1e-5 bar for that alternative path, which is why the exact mode is the one used. |
| 1b | vs the BANKED eval, same windows | mean ADE6 **2.175306 (s0) / 2.054566 (s1)** equals the banked mean to 1e-8. Per-window ADE6 max abs diff **1.19e-5 m (s0), 2.77e-5 m (s1)**; `pred` (traj30) max abs diff 4.6e-5 / 9.9e-5 m; 408 / 379 windows differ above 1e-5 in `pred`. The banked eval ran on box-2 GPU, batch 64; this is CPU float32, batch 24; the differences are float32 round-off (1e-6 relative). UNVERIFIED which kernel causes it; it is 5 orders below any plotted effect. |
| 1c | attention rows sum to 1 | min 0.9999993, max 1.0000006 (all levels/layers). |
| 1d | explicit weights vs torch's `need_weights` weights | max abs diff 1.8e-5 (s0) / 1.1e-5 (s1) in float32. Float64 recomputation of the same captured inputs: **both float32 paths are 9.8e-6 (s0) / 1.35e-5 (s1) from the float64 reference, identically**, because the logits are huge (max abs logit 183 / 158) and float32 round-off in a near-one-hot softmax shows at 1e-5. Round-off, not a bug. |
| 2 | **frame alignment** (`align_control.py`) | For every cache step j (101 per clip, 404 frames over the 4 clips) the v2ep frame 2j was **re-encoded on CPU with DINOv3-L** (same recipe as the shipping encoder) and pooled to 4x10. Mean per-token cosine to the cached tokens of step j = **0.9993-0.9995** (min 0.9990); the best of the neighbours j-2, j-1, j+1, j+2 = 0.963-0.984. **404 / 404 frames have step j as the argmax.** So the pixels overlaid at step j are the pixels step j's tokens came from, and an off-by-one frame would have been caught. Overlay frame (`refav1-eval141-256rb`) vs the BANKED eval v2ep frame (`refav1-eval141/eps`): **PSNR >= 48.6 dB, max abs 3/255** on every frame. |
| 3 | **model randomisation** (Adebayo 2018; `stats_pass.py`, seed 0, 2,399 windows) | One Reader re-initialised (fresh `nn.TransformerDecoder`, seed 1234); the downstream levels' inputs are frozen at the trained values so that nothing else changes. Spearman rho(trained map, randomised map) over the 160 scene tokens, mean [95 % CI over episodes]: **S +0.084 [0.057, 0.112]; T +0.028 [-0.004, 0.062]; O -0.087 [-0.115, -0.058]**; tick-frame 40 tokens +0.087 / +0.019 / -0.099; windows with rho > 0.5: 2.7 % / 3.9 % / 0.8 %. **The other two levels' attention changed by max abs 0.0 exactly in all three randomisations.** Randomising O's Reader moves `traj60` by up to 53.9 m (so the Reader carries the function); for S and T the trajectory is unchanged BY DESIGN (downstream frozen), so that is no evidence about function loss for them. The maps are therefore not input-independent artifacts. The test says nothing about faithfulness. |

What was NOT done: the mp4 -> cylindrical re-projection (the raw stage mp4s were not re-projected; the overlay is the encoder's own input frame, so no downscale step exists on our side, and control 2 covers the token side); a cascading (progressive) randomisation; a data-randomisation (label-permutation) control.

## 3. Deliverables

* `v8f_hierarchy_attention.mp4`: **80 s, 1920x1080, H.264 yuv420p, 4 fps (0.8x real time), 8.0 MB.** Title card, then 4 clips (JUNCTION, CURVE, NO-TURN, FAILURE; 68 steps each) with a card before each. Per frame: top row = the camera frame three times with the S, T, O heat overlays (S purple, T blue, O orange ramps, each on the frame it last ticked; % on the top-3 cells = share of ALL of that level's attention); bottom = BEV (GT 6 s path, O's own-code prediction, S's 3 checkpoints, and "T0 ORACLE (GT code), diagnostic only" dashed), T's code (lat 5-way, lon 7-way with GT box, top-4 goals, target speed / lead gap / turn distance / turn heading change vs GT), per-level attention-by-source bars with the uniform-attention tick, and an ADE6-vs-time strip with the selected eval window marked. Every frame carries arm, tier, "qualitative" and the selection rule.
* `fig_attention_levels.png`: 4 rows (the 4 banked windows) x S/T/O overlays + BEV.
* `fig_attention_stats.png` + `fig_attention_stats.json` (+ `controls.json`): panels A-I, both seeds, CIs.
* `attn_extract.py` (instrument), `stats_pass.py`, `analyze_stats.py`, `plot_stats.py`, `viz_data.py`, `viz_lib.py`, `render_video.py`, `select_clips.py`, `align_control.py`, `control_f64.py`.

## 4. Findings (MEASURED, all 2,399 windows, artifacts `fig_attention_stats.json` / `controls.json`)

### 4.1 The levels differ in WHAT they read and how sharply: reproduced in both seeds
| | seed 0: cond-token share of attention | seed 1 |
|---|---|---|
| S | 3.4 % [2.3, 4.7] | 0.0 % [0.0, 0.0] |
| T | 0.6 % [0.3, 1.0] | 0.1 % [0.0, 0.3] |
| O | **76.5 % [75.9, 77.2]** | **93.4 % [92.2, 94.4]** |

S and T put 96.6-100 % of their attention on the scene tokens; **O puts most of its attention on its own 6 cond tokens** and 6.6-23.5 % on the scene. Effective number of scene tokens attended (exp of the entropy, of 160; uniform = 160): **S 6.4 [6.1, 6.7] / 5.3, T 8.2 [8.0, 8.4] / 7.7, O 14.9 [12.9, 17.2] / 15.7** (s0 / s1). Per (head, query) row the attention is nearly one-hot: S mean max weight 0.85 / 0.83 (62 % / 53 % of rows above 0.9), T 0.71 / 0.74, O layer 0 0.94 / 0.92.
Paired level contrasts (episode-cluster CI, both seeds same sign and separated from 0): cond share O-T and O-S (huge), effective tokens T-S (+1.8 / +2.4), O-T (+6.7 / +8.0), tick-frame share T-S (-0.18 / -0.33).

### 4.2 What O reads among its cond tokens is NOT reproducible across seeds
Seed 0 O: age 28.7 %, lim 27.3 %, nav 11.0 %, ck 7.3 %, ego 2.1 %, code 0.2 %. Seed 1 O: **ck 59.0 %**, age 24.0 %, ego 5.4 %, nav 3.3 %, lim 0.7 %, code 0.9 %. O's second layer puts 43 % on the scene in seed 0 (layer-1 cond share 56.6 %; layer 0: 96.5 %) but only 6 % in seed 1 (layer 1: 93.9 %; layer 0: 92.8 %). Both seeds place ~0 % on the `code` token (T's tactical code) as a KEY (0.2 % / 0.9 %). Whether O's output depends on T's code is a separate question, tested by the banked TAC arms and NOT re-checked here; see limits: cond.mean is added to every query, so attention mass on a cond key understates its use. A quarter of O's attention sits on the `age` token (2 numbers, a_t and a_s): UNVERIFIED hypothesis that it acts as an attention sink.

### 4.3 Where in the image: "centre ahead, horizon and road"; finer differences between levels are seed-specific
* Near rows are ignored: **row 3 (bonnet / near) carries 0.5-0.7 % (S) and 0.5-1.5 % (T)** of scene attention in both seeds; O 8.7 % (s0) / 3.2 % (s1).
* The horizon row dominates S (53 % / 59 %) and O (60 % / 52 %); T spreads over far / horizon / road (s0 34 / 29 / 36 %; s1 6 / 52 / 40 %).
* Columns: O puts 79 % (s0) / 76 % (s1) on columns 4-5 (straight ahead). Mean tick-frame maps: S s0 cells (1,4) 32.6 % + (1,5) 29.9 %; T s0 (1,5) 31.8 %, (0,4) 14.6 %; O s0 (1,4) 40.4 % (percent of that frame's attention).
* **Seed-specific:** S s0 puts 6.4 % of its tick-frame attention on the single top-right corner token (row 0, col 9), s1 0.0 %; far-row share (row 0) of S is 24 % (s0) vs 1.5 % (s1); the O-T far-row contrast flips sign between seeds (-0.32 vs +0.12). The mean tick-frame maps correlate across levels within seed 0 (Spearman over 40 cells: S~T 0.63, T~O 0.59, S~O 0.70) about as strongly as across seeds within a level (S 0.72, T 0.60, O 0.69): **on the mean maps, a level-specific spatial focus cannot be told apart from seed variation.** (Spearman over 40 cells is dominated by the ~0 cells; weak statistic.)
* Recency: the tick frame carries 68 % / 80 % (S), 50 % / 47 % (T), 70 % / 49 % (O) of scene attention (uniform 25 %); the three older frames still take 20-53 %. T uses history most in both seeds (50 % / 53 % on older frames); O does so in seed 1 only (51 %).
* Stratum (means [CI], effective tokens of O): JUNCTION 20.9 [17.9, 24.6] / 30.2 [23.8, 36.4], CURVE 9.2 [8.1, 10.6] / 7.0 [5.5, 9.0], NO-TURN 14.4 [12.2, 16.9] / 14.1 [11.9, 16.5] (s0 / s1): **O attends more broadly at junctions and more narrowly in curves, both seeds** (non-overlapping intervals vs NO-TURN; unpaired comparison). T's effective tokens do not differ clearly by stratum (7.3-9.4).

### 4.4 The four clips (qualitative; n = 1 per row, no inference)
* In the four banked windows S and T put their top cells mostly on the centre-ahead horizon / road cells (the failure case also puts cells to the left and right of the road ahead); O's attention on the tick frame is 11-24 % of its total. In the JUNCTION window T's top cell (21 %) lies on the traffic light.
* The failure case (window #2158, t = 11.8 s): T predicts **TURN_L (lat class 1) while the GT label is LANE_KEEP (class 0)**, in both seeds; the S checkpoints bend left, O's path curves left, the GT path goes ahead / right. With the video schedule: own-code ADE6 **15.95 m (s0) / 16.09 m (s1)**, with the GT code fed to O (T0 oracle, diagnostic only) **6.41 m (s0) / 10.25 m (s1)**. So the wrong tactical code accounts for ~9.5 m of the 16 m in seed 0, and in seed 1 even the correct code leaves 10 m.
* Age sensitivity, single-window anecdote: JUNCTION window #1131 reads own-code ADE6 **2.66 m banked (a_S = 2) vs 1.48 m under the video schedule (a_S = 0)** for seed 0; CURVE #654 seed 1: 5.37 m both ways. Do not read the video's per-step ADE as the banked number.

## 5. Limits (read before quoting)
1. **Attention is not an explanation.** Randomisation (control 3) shows the maps depend on the trained weights; it does not show they are faithful. Each Reader also adds `cond.mean` to every query (`Reader.forward`), so ego / nav / limit reach the output without any attention on their key; ~0 % attention on a cond key means little for S and T.
2. Logits reach |183|, so most (head, query) rows are one-hot: a mean map over heads is a union of a few hard picks (4 heads x 25 %), not a smooth saliency. Per-head maps are in the chunks, not plotted.
3. A DINOv3 token carries global context (ViT), so a heat cell is not "the content of that 64x64 px block"; the cache is fp8; the grid is 4x10 (coarse).
4. T0 only: open loop, the true past fed. No T1 / closed-loop claim. One arm (M2b), two seeds, one clip per row.
5. Episode CIs do not cover training variance; the seeds disagree on cond-token usage and on far-row attention (4.2, 4.3), so those are not findings about the architecture.
6. No STRATEGIC-family eval number is reported (the PI pause of 2026-09-19 stands); S appears only as a diagnostic of its Reader's attention and its 3 checkpoints in the BEV.
7. Video ages follow a physical tick schedule; banked windows use the eval's cycled ages (4.4).
8. C: dropped to ~196 MB free at one moment (our footprint ~0.5 GB; another process), which killed one ffmpeg run and overwrote the first mp4 with a stub; the render was rerun (atomic `.tmp.mp4` + rename now). The deliverable mp4 is the complete 80 s file.

## 6. Reproduce (dev box, python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `CUDA_VISIBLE_DEVICES=-1`, `PYTHONPATH=<TREE>/stack`)
```
python select_clips.py                      # selection.json (needs work/banked eval npz)
python stats_pass.py                        # streams the 141 clips from box 2, ~2 min, writes work/stats/chunks
python analyze_stats.py && python plot_stats.py
python viz_data.py --seed M2b_s0 [--seed M2b_s1]   # needs the 4 clips in work/cache_viz
python align_control.py --clips 79e30ac2-c619-4d61-98a3-1007037b4d05,45aea78c-7762-4e46-aef1-e9e22602b8fd,51c01ba4-9000-4370-8abb-9fd22b397d05,ed87040c-cf3c-4fe4-9a95-50c1114f0d38
python render_video.py keyfig ; python render_video.py video --workers 4 --fps 4
```
Scripts hard-code the scratchpad paths (`SCRATCH` in `attn_extract.py` / `render_video.py`); relocate before landing.

## 7. Deliverable manifest
| artifact | dev-box path | box-2 path | note |
|---|---|---|---|
| video | `OUT/v8f_hierarchy_attention.mp4` (8.0 MB) | none | **only on the dev-box scratchpad** |
| key-frame figure | `OUT/fig_attention_levels.png` | none | only here |
| stats figure + data | `OUT/fig_attention_stats.png`, `.json`, `controls.json` | none | only here |
| controls | `OUT/align_control.json` (+4 npz), `control_f64.json` | none | only here |
| instrument + pipeline | `OUT/*.py`, `OUT/deps/taniteval_b2/` (copy of box-2 `taniteval` modules) | sources read from `D:/tanitad/repo/taniteval/taniteval/` | only here |
| per-step video data | `OUT/data/M2b_s{0,1}__<clip>.npz` (8), `merged_summary.npz` | none | only here |
| full per-window attention (all heads, layers) | `work/stats/chunks/*.npz` (141 files, 211 MB) | none | **not in OUT** (size); regenerable in ~2 min with `stats_pass.py` |
| inputs | `work/{ckpt,inputs,banked,cache_viz}` | `D:/tanitad/runs/v8f_mini_full/...`, `D:/tanitad/cache_full/...` | copies; originals untouched on box 2 |
