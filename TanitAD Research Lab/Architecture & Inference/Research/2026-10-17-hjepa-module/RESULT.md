# RESULT -- stream S3: the H-JEPA module (`stack/tanitad/models/hjepa.py`)

**Date:** 2026-10-17. **Plan:** `Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §4, §7-S3.
**Primary:** H-JEPA, arXiv 2610.06805 (`lib:2610.06805`). Implemented from the paper only; no code or repo was downloaded.

## Headline (integration escalation)

* **Needs landing by the orchestrating session.** Three NEW files exist in the working tree, UNTRACKED and NOT staged (session
  amendment: no `git add`): `stack/tanitad/models/hjepa.py`, `stack/tests/test_hjepa.py`, and this package.
  The working tree at `E:\Projects\TanitAD` is the stale HEAD `37645fcc`; land through `qland`, not by committing from this tree.
* **Default-off by construction:** `grep -rl hjepa stack` finds only the two new files. The module imports nothing from `tanitad`.
* **No trainer wiring exists.** Wiring is stream S5 and is gated on the PI go (§10 D-A). The open items are in the last section.

## What is built (all in `hjepa.py`)

| piece | paper | notes |
|---|---|---|
| `SIGReg` | App. K, K.3, Eq. 9 | Epps-Pulley ECF statistic, M random unit directions, 17-knot trapezoid on [0,3] with the weights doubled for the even integrand, Gaussian window, grouping (T,N,D), `n` prefactor. `reduce_fn` / `torch.distributed` path all-reduces the real and imaginary ECFs and passes `n_local·W` as the prefactor. `null_floor()` returns the analytic expectation on Gaussian data (1.0525). |
| `IDMHead` | S4.3, Eq. 8, Prop. 3 | MLP on `[z_t, z_{t+1}]`; loss = MSE over `d_a`, mean over t and batch; target `sg[·]`. |
| `LevelEncoder`, `MacroActionEncoder` | Eq. 1; App. C.1/C.2 | 2-layer MLP over the (window of the) lower latent; macro-action pooled from `s` action embeddings, `mlp` (DROID) or `transformer` (single-scene) mode, final LayerNorm. |
| `CausalPredictor` | Eq. 2; App. C | pre-LN causal transformer, per-block AdaLN (shift, scale, gate) from the action embedding. |
| `HJEPAStack` | S2.1-2.2 | `encode`, `loss`, `rollout`, `lift` (Eq. 5), `goal_cost`, `score_candidates`, `plan_topdown` (Eqs. 4-6), `action_sensitivity`, `fit_action_prior`, `param_groups`, `built_levers`, `parameter_counts`. |
| free functions | Eq. 10 and PREREG | `rho_within_episode`, `donor_indices`, `monotonicity` (BAR-HJ-COST statistic), `param_grad_report`, `dead_parameters` (the G-LIVE building block). |

Sizes (MEASURED, `parameter_counts()`, z1 width 256, action dim 10, level 2 width 128, macro-action 8): **5.8 M** for the WM levels at
predictor widths 256/192 (depth 4/4), **15.0 M** at DROID-scale 384/256 (depth 6/4). The plan's "≈ 6-12 M (E)" is consistent.
A full `loss()+backward()` at B=8, T1=13 takes 0.46 s on CPU for the WM alone (trunk excluded).

## Design choices and every deviation (UNVERIFIED = the paper or its text extraction does not settle it)

1. **Prediction loss is the per-dimension MEAN** (`pred_reduction="mean"`; `"sum"` offered). Eq. 2 written literally is a sum over D. Fig. 10's magnitudes (0.01-0.2 on SIGReg-whitened latents of width 64-384) are only consistent with a mean. UNVERIFIED against the released code. The β and γ values quoted from the paper presuppose this choice.
2. **AdaLN init = "identity gate"** (gate bias 1, shift/scale ≈ 0, modulation weights N(0, 0.02²)), **not DiT adaLN-Zero.** The paper says only "adaptive LayerNorm". MEASURED here (`test_zero_init_adaln_is_glive_unsafe…`): under adaLN-Zero the gates are 0 at step 0, so every block weight has an exactly-zero gradient and G-LIVE would be RED. `adaln_init="zero"` stays selectable.
3. **Attention is hand-written on `scaled_dot_product_attention`**, not `nn.MultiheadAttention`, so the grad and no-grad paths are numerically identical (the torch fast path is not). Widths, depth and heads are configuration (paper: width 192, 4 layers, 8 heads; DROID 384/12 layers).
4. **Level-1 encoder is external.** `z¹` is an input. The paper's ViT + BatchNorm projector is not reproduced. A learned `A^(1)` is optional (`act_embed_dim`); the default is the identity ("Level 1 consumes raw actions", C.2). Level-1 IDM regresses the RAW action, level ≥ 2 regresses the detached macro-action (Eq. 8).
5. **A latent is a vector per frame.** The plan's "16-32 tokens × 256 d" must be flattened or pooled by the chassis (open item 1).
6. **Windows:** `w ≥ 1` is supported in `encode` and `lift`; **planning supports `w = 1` only** (`initial_latents` raises `NotImplementedError` otherwise). The paper uses `w = 1` throughout.
7. **Sequences longer than the context** are processed in consecutive chunks of `c` positions (the first position of each chunk attends only to itself, as at planning time). The paper trains on exactly `c+1` states per level.
8. **Macro-action `transformer` mode ends in a LayerNorm.** The paper states the final LayerNorm for the MLP variant only. UNVERIFIED.
9. **SIGReg window width = 1** (`w(t)=exp(-t²/2)`). Consistent with the paper's floor of 1.04-1.37 (ours: analytic 1.0525), but the width is UNVERIFIED. **Table 11's absolute numbers were not reproduced:** the extracted table columns are ambiguous (0.349154 vs 1.372619 for one shared matrix). What is pinned is the property the table proves: exact invariance to sharding.
10. **Distributed gradient convention (derived, then MEASURED).** The all-reduce's backward is the identity, so that DDP's mean over ranks of the per-rank gradients equals the single-device gradient of the global statistic. A real 2-process gloo run confirms `grad_rank = W × grad_reference_shard` and value equality to 1e-12.
11. **`goal_cost`** averages the squared distance over the last `k` states (`k=2` at level 1, `1` above; App. E). Sum vs mean is UNVERIFIED.
12. **Planner not implemented from App. E:** the cross-level action cost (weight 0.2; disabled on DROID and on FourRoom/AntMaze), upper-level Gaussian action noise (0.01; Push-T/Cube only), and the replanning schedule with decreasing horizons (we plan open loop, one plan, as on DROID). Implemented: per-level AdamW on the actions only (model parameters never receive a gradient), at most `n_iters` (default 90) with early stopping, candidate 0 at the prior mean, per-level box bounds and fitted percentile box, `project_fn` for non-box constraints, `init_actions` (refinement of an existing plan), per-level `lr` and `num_candidates`, K-subgoal tracking with `lam`.
13. **Extras beyond the brief:** `monotonicity` (the BAR-HJ-COST statistic), `param_groups` (level 1 at `lr`, upper levels at 0.2×, AdamW wd 1e-3, paper E4), `built_levers` (the G-DVB reader), `fit_action_prior`, `dead_parameters`.

## Findings worth keeping

* **SIGReg's "floor" is an expectation, not a per-batch value.** On one fixed sample (1,024 episodes × D=8) the statistic read **1.36** against the analytic 1.05. At D=32 it is within 30 %. Any gate threshold built on "≈ floor" needs ±30 % at small D. The test uses D=32 and ±30 %.
* **A first-pass test gap was found by mutation, not by inspection.** The mutation "prediction target not shifted" survived the first 44 tests (every identity-predictor control is shift-invariant). `test_pred_loss_target_is_the_next_state_not_the_current_one` now pins it, using data generated by the stub's own dynamics (loss < 1e-20) with time-shuffled and time-reversed targets that must read > 1e-2.
* **One real bug caught while writing tests:** `plan_topdown` returned the best iterate of an UNPROJECTED seed when `project_fn` was set (an infeasible plan could win). The seed is now projected. Test: `test_plan_topdown_respects_box_bounds…`.

## Tests

`pytest -q stack/tests/test_hjepa.py` (python `C:/Users/Admin/venvs/tanitad`, `OMP_NUM_THREADS=6`, CPU): **45 passed**, 10-51 s wall depending on host load (typical 17-28 s; the box is shared). They include a real 2-process gloo run (~3-6 s).

| group | tests | what is pinned |
|---|---|---|
| SIGReg | 10 | analytic floor (n = 32/256/1024); exact linear growth in n on collapsed data; scale and rank collapse detected; **(T,N,D) is blind to background collapse (Prop. 2)**; float64 sharding invariance for W = 1,2,4,8 (≤ 1e-12) plus the naive per-shard average that is NOT invariant; generator reproducibility; finite gradient incl. bf16; gloo 2-process value and gradient |
| Prop. 1 / Prop. 3 | 3 | collapsed encoder + identity predictor: `L_pred = 0` exactly (L1) / < 1e-12 (L2) while SIGReg ≈ healthy; trained IDM on collapsed latents reads **0.95-1.06 (floor 1)**, the same head on informative latents < 0.1 |
| structure | 8 | Eq. 1 indexing for (s=2, w=1) and (s=3, w=2) incl. macro-action window offset; loud failure on short or mismatched clips; macro encoder shape and LayerNorm (both modes); causal mask; action conditioning at init; loss terms and weights; target = next state; chunking |
| reachability and training | 10 | every parameter of every level live (4 variants: macro mode × `A^(1)`), unused control reads **exactly 0.0**; γ=0 makes exactly the IDM heads dead while γ>0 makes none; adaLN-Zero shown G-LIVE-unsafe; level-2 loss alone reaches `z¹` (E4); IDM target detached; training reduces pred and IDM loss |
| diagnostics | 6 | self-swap **0.0 exactly**; identity predictor **0.0**; linear-action stub > 0.05; real predictor > 0; per-channel isolation (a deaf channel reads exactly 0); level-2 macro-action space; donor permutation; Eq. 10 ρ (0 at collapse; ρ + between-share = 1 to 1e-12; ragged form agrees); monotonicity |
| planner / scorer | 6 (+2 misc: `built_levers`, action prior) | single-level toy: cost falls > 1000×, replayed plan reaches the goal; two-level subgoal tracking (H₁ = K·s + w − 1); box bounds hit the analytic optimum (cost 8.0); `project_fn`; seed refinement never worse; early stop and mode restore; candidate ranking via the lifted goal distance; misaligned horizon raises |

**Mutation check** (`code/mutation_check.py`, restores the source in `finally`): **13 of 13** deliberate regressions turn the suite RED, including drop-IDM,
prefactor n→1/n, ignore world size, all-reduce backward halved, off-by-one macro window, no-op action swap, planner that never steps, undetached IDM
target, non-causal attention, unshifted target, SIGReg with time in the sample axis, dropped action-SIGReg, and off-by-one macro pooling.

**Nothing existing broke:** `pytest -q stack/tests -k "hierarchy or refd or predictor"` → **106 passed, 1 skipped, 8,608 deselected, 22 collection errors** (311 s).
All 22 errors are PRE-EXISTING and unrelated: 20 × `ImportError: cannot import name 'kinematic_prior'`, 1 × `'slot_presence'`, 1 × missing helper `test_map_hires_wiring`.
They come from this stale tree (HEAD `37645fcc`, whose `scripts/refc_v3_train.py` needs modules that do not exist at that commit). They are not caused by this work and need the landed tip.

## Open integration items for the refcv chassis (S5) -- NOT done here

Chassis references are from this tree (`stack/tanitad/refs/refc.py`, `refc_v3.py` at HEAD `37645fcc`), UNVERIFIED against the tip `e8cadf2`.

1. **Where `z¹` comes from.**
   * Candidate: `pooled_seq [B, W, F]` (`refc.py:4036`, `F = encoder.feat_dim`, `W = cfg.window` = 8), the per-frame pooled feature. It is the hierarchy hook's input (`hierarchy_hook(pooled_seq, ctx)`).
   * `pooled_seq` is HISTORY (past W frames). H-JEPA needs a sequence that continues into the future for the level-1 targets and for the +6 s goal target (12 level-1 steps at 0.5 s).
   * Therefore the future frames need extra trunk forwards. Run them under `no_grad` and pass `detach_target=True` (paper K.1: detachment changes nothing).
   * That is the cost G-CLOCK must measure (PLAN §4.1 expects +10-15 % step).
   * If z¹ is a token set (16-32 × 256), decide flatten vs pool first. The module treats a latent as one vector.
2. **Action blocks.** Level-1 `a_seq[:, t]` = the 5×0.1 s (a, κ) block flattened (`action_dim` = 10), in normalised units; the paper regresses unit-variance actions (Prop. 3 floor = 1). The (a, κ) channels must be normalised per channel, or the IDM floor is not 1.
3. **Trainer flags a `v8f` profile needs.** Every flag must be named in the arm's argv, because `default=0.0` is how a wired mechanism silently does nothing (CLAUDE.md "unreachable from its caller").
   * Weights and dims: `--hj-levels`, `--hj-beta1/2`, `--hj-gamma1/2`, `--hj-act-sigreg`, `--hj-lambda-wm`, `--hj-ramp-steps`, `--hj-sigreg-m`.
   * Structure: `--hj-stride2`, `--hj-macro-dim`, `--hj-pred-width/depth/heads`, `--hj-context1/2`.
   * Optimisation and targets: `--hj-upper-lr-mult`, `--hj-detach-target`, `--hj-adaln-init`.
   * Planning and scoring (β/γ rungs): `--hj-score-gate-init`, `--hj-refine-steps`, `--hj-goal-head`.
4. **G-DVB** (`tanitad/train/declared_vs_built.py`, `register(dest, kind, check)`).
   * `loss`-kind entries for β, γ, `act_sigreg` and λ: the weight the loss reads is `stack.built_levers()[l]`, and the IDM head or SIGReg must be live iff the weight is > 0.
   * `built`-kind entries for strides, dims, context and macro mode: read from `built_levers()`, which reads built modules and never the config that built them.
   * A flag without an entry is refused by the gate itself.
5. **G-LIVE.**
   * Per-group gradient must be non-zero after one real-data backward: use `param_grad_report` / `dead_parameters` per level, with an exactly-zero control parameter.
   * Keep `adaln_init="identity_gate"`; adaLN-Zero is RED by construction at step 0.
   * Add `check_untrained_params.py` as the post-training gate (the Adam-state probe from CLAUDE.md).
6. **G-EVAL / G-CKPT.** Prior buffers (`prior_*_l`) are persistent and round-trip; the SIGReg buffers are non-persistent by design. The eval loader must rebuild `HJEPAStack` from the SAME `LevelConfig` list, so store it in the checkpoint's config.
7. **DDP.** Call `stack.loss` on each rank with its shard and let DDP average. `SIGReg` auto-broadcasts the directions from rank 0 and all-reduces the ECFs. Use equal shards and drop the last partial batch. Never wrap the loss in an extra `/W`.
8. **Goal latent ĝ²** (PLAN §4.3) is NOT in this module. Its head, the "inputs ⊆ {trunk features, nav token, turn distance}" signature test and the classifier-logit probe belong to S5.
9. **Level-1 learning rate.** The paper uses 5e-5 at level 1 and 1e-5 above. Use `param_groups(lr, 0.2)` for the stack, and the chassis's own groups for the trunk.
10. **BAR-HJ-SENS wiring.** `action_sensitivity(z_start, actions, level, ks, channels, episode_ids, seed=0)` implements the PREREG definition. Name the channel groups for (a, steer, v), and pass `episode_ids` so every donor comes from another episode.

## Deliverable manifest

| artifact | where | stranded? |
|---|---|---|
| `stack/tanitad/models/hjepa.py` (module, 1,136 lines CRLF) | `repo:stack/tanitad/models/hjepa.py` | **only in this working tree, untracked, not staged** |
| `stack/tests/test_hjepa.py` (45 tests) | `repo:stack/tests/test_hjepa.py` | **only in this working tree, untracked, not staged** |
| this report | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-hjepa-module/RESULT.md` | **only in this working tree, untracked, not staged** |
| mutation script | `repo:…/2026-10-17-hjepa-module/code/mutation_check.py` | **only in this working tree, untracked, not staged** |

Nothing lives on a pod or in a worktree. All four are one `qland` away from the repo.
