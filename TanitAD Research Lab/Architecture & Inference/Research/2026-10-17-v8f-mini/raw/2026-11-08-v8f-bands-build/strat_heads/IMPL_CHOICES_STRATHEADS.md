# IMPL_CHOICES_STRATHEADS (Stream C, 2026-11-08): the strategic heads on the frozen ladder world model

Nothing was trained on real data. Tests ran on CPU with synthetic clips of the real schemas and the frozen first ladder model `F:/TanitAD-scratch/hwm_bands/ckpt/R35_s0.pt`. Evidence classes: MEASURED (this folder; test log below), ESTIMATED, HYPOTHESIS, UNVERIFIED.

## 1. What is built
| file | role |
|---|---|
| `strat_heads.py` | library: schema, `CtxHead`, `CkptHead`, `GstrHead`, `FrozenWM`, losses, label and clip loaders, sampler, source scans, vision-only and goal-path probes |
| `train_strat_heads.py` | trainer: lever read-back self-check, token, training, evaluation (STRATEGIC PAUSED banner), per-clip `stok` dump |
| `act_extract.py` | box-2 sidecar per clip: `act` [Te,2], `v` [Te], `nav` [T,Dn] (native pose track + label join) |
| `synth_clips.py`, `test_strat_heads.py`, `run_tests.cmd` | synthetic clips and 8 tests |
| `box2_argvs.txt` | box-2 argvs (sidecars, self-check, train, dump) |

## 2. Design
* **World model** (`FrozenWM`): `WMAux` rebuilt from the checkpoint's `cfg`, `wm.*` weights loaded strictly, `requires_grad` False, eval, every call under `no_grad`. K = 1 only (window-1 encoders; refused otherwise). L3 state of any frame = `encoders[1](encoders[0](proj(zmean)))`; at the tick frames (0, 15, 30, ...) it equals `HJEPAStack.encode` (max abs diff 1.1e-6 for the state, 0.0 for the macro-action, MEASURED, selfcheck `chain_vs_encode`, tolerance 1e-4). L3 step = 0.2 x 3 x 5 = 3.0 s, read from the built strides; horizons 6..30 s are rolled states k = 2..10 (`k_index` buffer).
* **head_ctx** (vision only): input = `zmean` at NOW and the masked mean over frames NOW, -1 s, -2 s, -5 s (`HIST_OFFSETS`); LayerNorm, two-layer MLP, five linear outputs (country 25, illumination 4, weather 4, surface 3, visibility 3). CE per attribute with per-attribute masks (label -1 = unknown). Class balance: country weight (n/n_c)^1.0 (`--country-w-pow`), the others ^0.5 (`--other-w-pow`), each normalised to mean 1 over present classes. Glare is folded into day as in `merge_full.py`. Period of day is not a head (the brief lists five attributes); it stays a label column.
* **head_ckpt**: shared MLP on [LN(rolled state k), LN(L3 state NOW), horizon embedding] -> (x, y, heading) in units of (10 m/s x horizon, 10 m/s x horizon, 1 rad). Masked L1 per horizon (wrapped heading); mask = stream A `ck_valid` AND (logged mode) the logged actions reach the horizon. Rollout actions: `logged` = the clip's actions from NOW (150 steps = 30 s), `zero` = neutral zero action; `--ck-act-mix` (0.5) is the share of zero samples per batch. **Only `zero` exists at inference, so only `zero` goes into `stok`**; `logged` is a T0 diagnostic (saved as `ck_logged_diagnostic`). Rolled states past k = 6 (18 s) are outside the latent-trained reach (SPEC section 2): supervised by geometry only; nothing here claims they are accurate latents.
* **head_gstr**: MLP on [LN(L3 state NOW), nav, softmax posteriors of head_ctx **detached**] -> 8-way token CE (class-weighted ^0.5), `by_time_s` and `within_m` (Huber, masked to non-FOLLOW tokens with finite args), two context-derived constraints (speed margin fraction, headway margin in s; Huber). Reads no tactical code, no ckpt output, no ego quantity, no situation-classifier output.
* **Constraint labels** (`constraint_rule`): a fixed table over the GT context labels (for example dark +0.10 speed and +0.5 s headway; snow +0.20 and +1.0 s; caps 0.4 and 3.0 s). HYPOTHESIS; the values are a design item for the PI (SPEC section 3). The head learns them from the ctx posteriors, so this term checks the plumbing and the context edge; it adds no information the context head does not already hold.
* **Labels**: context and goal columns from the release frames parquet (`merge_full.py` schema: `weather`, `surface`, `visibility`, `illumination` or folded `illum_img`, `g_str_*`); checkpoints from stream A `strategic_geom_<split>.npz` (keys `clip_ids`, `clip__off`, `clip__n`, `ck_xyh` [N,9,3], `ck_valid` [N,9], read from `band_labels/build_strategic_geom.py`), else the release `ckpt_*` columns. `SCHEMA.md` did not exist when this was written: if its keys differ, only `Labels.__init__` changes.
* **Trainer**: AdamW over head parameters only, linear warm-up then cosine to 10 %, grad-norm clip 1.0, one loss per head, total = w_ctx L_ctx + w_ck L_ck + w_g L_g (all 1.0). Resumable (`--resume`; `heads_last.pt` every 500 steps, atomic rename).

## 3. Controls (all run in `--selfcheck-only`, before step 1; any failure exits 2)
| control | reading (synthetic batch, MEASURED) |
|---|---|
| declared levers read back from the BUILT objects (widths, class counts, horizons, `k_index`, strides, step_s, act mix, hist offsets, batch, lr, wd, seed, loss weights, country weights, WM trainable count, optimizer-in-WM count) | equal, else REFUSED. Test: `args.ctx_hidden` changed after build -> exit 2 |
| gradient matrix: each loss alone, abs-sum of grads on ctx / ckpt / gstr / WM | own head > 0 (ctx 161.7, ckpt 400.6, gstr 249.1); every other entry exactly 0.0, including gstr loss -> ctx (the detach) and every loss -> WM |
| detach mutation (`ctx_for_gstr` without `.detach()`) | RED |
| WM bit-equal | sha256 of the WM state before and after one head step identical; after the 60-step smoke identical (`train_done.json`). Mutation: an optimizer over a copy's WM parameters changes the sha, so the detector is alive |
| head_ctx vision only | source scan of its own `forward` (no ego, nav, act, limit, pose, speed, situation, ck or label names), key-guard reads only {zhist, hist_ok}, output bit-identical when every other batch key changes, output moves with `zhist`. Mutations RED: ego input named; ego read under a computed key name (caught by the invariance probe) |
| head_gstr goal path | source scan (no situation, sitclf, tactical, ego or label names), key-guard reads only {nav}, bit-identical under change of other keys. Mutation: `batch["situation"]` read -> RED |
| head_ckpt source scan | no ego, nav or label names |
| edges live (PI rule 1) | gstr output changes under ctx swap (1.5e-3), ctx cut (1.3e-2), nav swap (3.5e-2); mutation: ctx columns of gstr zeroed -> RED |
| context-swap control at eval | gstr run with another episode's ctx posteriors; plain and swap numbers written side by side, plus the mean L1 change of token probabilities (0.080 on the smoke) |
| token | `SELFCHECK_TOKEN.json` = argv sha16 (argv minus `--token`, `--selfcheck-only`) + selfcheck sha256 + WM sha256 + code sha256; train refuses with no token, a token for another argv, or another code sha. Test: `--w-g 2.0` with the old token -> exit 2 |

This token is the trainer's own. It is **not** a `launch_gate.py` PASS token: the MM registers the trainer with the gate (the brief forbids editing `launch_gate.py` here).

## 4. Evaluation output
`eval_strat_heads.json` and a printed block that opens with `STRATEGIC PAUSED (PI item 25)`: tier T0, never a pass. Per ctx attribute: n frames, n clips, macro-F1 over classes with support (episode-cluster bootstrap over clips, `taniteval/ci.py` `stat_cluster_bootstrap`, estimator name written), accuracy, support, confusion (true x predicted). Per horizon and mode (`zero`, `logged`): n, head xy error, constant-velocity xy error (x = v0 t, y = 0, heading 0; `v0` from the sidecar, used only here), their paired difference with CI, heading errors. gstr: plain and ctx-swap token macro-F1, `by_time_s` MAE, `within_m` MAE, constraint MAEs, swap effect.
Smoke numbers (60 steps, 12 synthetic clips) are plumbing evidence only and are quoted nowhere as results.

## 5. `stok` dump (streams B and D)
Per clip `<dump>/<split>/<clip>.npz`: `stok` f32 [T, 78] (rows with `ok` False are zero), `stok_ok` [T], `stok_dim` = 78, `stok_keys` = JSON list of [name, start, stop]: `ctx_country` 0-25, `ctx_illumination` 25-29, `ctx_weather` 29-33, `ctx_surface` 33-36, `ctx_visibility` 36-39, `ck_zero` 39-66 (9 horizons x (x, y, heading), normalised by 10 m/s x horizon and 1 rad), `g_tok_p` 66-74 (softmax), `g_by_time_norm` 74-75, `g_within_norm` 75-76, `g_constraints_norm` 76-78; `meta` JSON (heads sha256, WM sha256, normalisers, tier). Everything is produced under `no_grad`: detached. Streams B and D take the width from `stok_dim`.
**Train-clip rows are in-sample for these heads.** A downstream stream trained on them sees posteriors cleaner than at test time. Eval-clip rows are held out. The lever (K-fold heads) is in the PREREG draft.

## 6. UNVERIFIED and open
1. `act` comes from the native track (`ego_channels_native`); the WM was trained on the cache file's `a`. `act_extract.py --check-cache` writes the max difference to `ACT_CHECK.json`; not run (no cache or tracks on this box). Until run, the logged-action checkpoints are UNVERIFIED as T0 diagnostics.
2. nav is the label join's `nav` row; its width is taken from the file (`--nav-dim`, default 8, truncates or zero-pads). Read the join's width before the first run and set `--nav-dim`.
3. The constraint table is a HYPOTHESIS.
4. Only 4 of 8 strategic tokens occur in the labels (SPEC F9): the `EXIT_*` and `LANE_CHANGE_*` rows of the token confusion stay empty; macro-F1 is over classes with support.
5. L3 states at non-tick frames are the pointwise lift of that frame's z1 (window-1 encoders), the same function the tick states use; the WM's L3 encoder was trained on tick frames only. Cheap check in the PREREG.
6. The release frames path and stream A's final keys are not fixed yet (`<RELEASE>`, `<BANDS>` in the argvs).
7. RAM: 4,572 train clips x 101 x 256 x 4 B is about 0.5 GB plus the parquet frame (about 0.5 M rows): ESTIMATED, fine on box 2.

## 7. Projected time (ESTIMATED from MEASURED unit costs)
Smoke on 2 CPU threads, batch 128, synthetic clips: 0.0123 s batch assembly + 0.193 s forward and backward = 0.205 s/step (MEASURED over 20 steps). 6,000 steps = 21 min. Label load (parquet groupby), zcache load, self-check (the 8 tests, which include one self-check and a 60-step run, take 46 s, MEASURED), evaluation (141 clips, stride 5) and the dump (4,713 clips, one 101-frame batch each, two rollouts) add an ESTIMATED 25 to 35 min. Total ESTIMATED 50 to 60 min on CPU; no GPU is needed and none is touched.

## 8. Tests (MEASURED)
`run_tests.cmd` -> `8 passed in 46.21s` (CPU, 2 threads): vocabulary pin, constraint rule, source-scan detection, chain equals `HJEPAStack.encode`, self-check + token + train + eval + dump + token refusals, lever-mismatch refusal, ctx-mutation probe, WM frozen. The log lives in scratch (`F:/TanitAD-scratch/hwm_bands/c/tests.log`); this line is its copy.

## 9. Integration items for the Master Mind
1. Register `train_strat_heads.py` with `launch_gate.py` (own token today).
2. Ship `strat_heads.py`, `train_strat_heads.py`, `act_extract.py` to `D:\tanitad\strat\`.
3. Fix `<RELEASE>` and `<BANDS>` and the WM checkpoint (R35_s0 or the ladder winner).
4. Streams B and D read `stok` by path; until a real dump exists they use synthetic tokens of width 78.
