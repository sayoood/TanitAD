# L2 SAFE-HEAD feature/label cache: data contract (v1, 2026-11-27)

Cache root `F:/safehead_cache/` (nothing in the repo; the Master Mind lands the code under this folder's repo path `PK/raw/2026-11-27-safehead-data/`).
Checkpoint: **C1** `model_final.pt` sha256 `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4` (= `Project Steering/MODEL_REGISTRY.md:5660`), built by `REFePlanner` exactly as the banked C1 navtest read built it (nav_mode `command`, per-sample calib, yaw shim with the table of the tokens' split, rule `navsim_v1`, A7 repair on, goal_fix `pdm_route`).

## Files
* `out/<log>.npz`, one per log (fixed zip timestamps: identical content gives identical bytes). `out/<log>.merge.json` = its sha256, size, dropped tokens.
* `index.csv`: `token,log,split,file,sha256`, one row per token; `sha256` is of the log's npz. `split` is `train` or `heldout` (the 6 held-out logs are from `splits/sft_heldout_logs.txt`; keep them out of fitting, use them for selection/eval).
* `plan/PLAN.json`: the selected logs (split, source archive, token list); `plan/DOWNLOAD_FILES.md`: the exact public files.

## Arrays in `out/<log>.npz` (n = tokens of that log; K = 64 proposals)
| key | dtype / shape | meaning |
|---|---|---|
| `token` | `<U16` [n] | nuPlan lidar_pc token (hex), the navtrain.yaml token |
| `log`, `split` | 0-d str | log name; `train` / `heldout` |
| `props` | float32 [n,64,20,3] | C1's 64 proposals as the model emits them: ego frame (x fwd, y left), 0.2 s grid, (x, y, heading); this is what `score_q_mlp` flattens |
| `props8` | float32 [n,64,8,3] | the EXECUTED plans on NAVSIM's 0.5 s grid (A7 last-heading repair, then `to_navsim`): the poses the labels were computed on and the harness scores |
| `s` | float32 [n,64,256] | the exact input of `score_head` (`nn.Linear(256,6)`, `model.py:563`): output of the 4th `CrossBlock` of `score_dec`, one forward pre-hook on `model.score_head`. float32 holds the captured tensor exactly |
| `logits` | float32 [n,64,6] | C1's own head output on `s` (order NC, DAC, EP, TTC, C, DDC); `sigmoid` for probabilities |
| `pick` | int64 [n] | C1's own selection: argmax of the planner's `navsim_v1` aggregate `NC x DAC x (5 EP + 5 TTC + 2 C) / 12` over `sigmoid(logits)` |
| `tr_nc tr_dac tr_ddc tr_ttc tr_ep tr_c tr_pdms` | float32 [n,64] | NAVSIM v1.1 PDM labels of each proposal, LQR-TRACKED (`PDMSimulator.simulate_proposals`), via `navsim_pdm_targets.score_plans` (the programme's labeller); EP/PDMS pairwise against PDM-Closed |
| `meta` | 0-d str (JSON) | ckpt path+sha256, rule, repair flag, goal_fix, yaw table path+sha256, tool, captured `s` dtype |

Token order inside a file = dump order; tokens without a label (scenario or memory failure that survived the retry) are DROPPED and listed in `merge.json` (`dropped_no_label`), never filled.
`ego` (7-vector) and `cmd` (NAVSIM driving_command one-hot, from `p6dry/hf/logs/commands.json`) are in the dump stage (`dump/<log>.dump.npz`) only.

## Reading the label/feature pair
`np.load(f)`; `s[i]` [64,256] and `tr_*[i]` [64] belong to token `token[i]`. A head that replaces rows 0, 1, 3, 5 of `score_head` (NC, DAC, TTC, DDC) reads `s` and is scored against `tr_nc, tr_dac, tr_ttc, tr_ddc`; the EP row (2) stays C1's.
`CTRL-LOGIT` shows `s @ W.T + b` (C1's head weights) reproduces `logits`, so a probe fitted on `s` starts from C1's head exactly.

## Caveats (state them in any report built on this cache)
* Evidence class MEASURED (ours, `ctrl/ctrl_results.json`); labels are a REBUILT metric cache, not NAVSIM's file: PDMS mean |d| 0.0008 vs NAVSIM's real cache on navtest (PREREG_SFT4.md:19-22, INHERITED), and on 57 navtest picks here NC/DAC/TTC/C/DDC agree exactly while EP differs on 6/57 (max 0.0108).
* C1 was trained on navtrain: these are train-distribution tokens (fit data for a head, not an unbiased read of C1's navtrain skill). The `heldout` logs are held out from the HEAD fit, not from C1's own training. Tier: T0-style single-step open-loop planning labels, NAVSIM v1.1 pseudo-simulation, not a closed-loop claim.
* The per-log token counts are small (log slices of 4 to 381 tokens); the cluster for any bootstrap is the LOG (35 logs, 6 held-out).
