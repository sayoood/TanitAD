# MOTION_DEFS: what was built for the level-1 predictor levers (2026-11-14)

Base: `F:/TanitAD-scratch/v8proof/spatial/cellpred/tree_patch` (the CELLPRED result: flat and cell level-1 predictors, `--wm-l1-pred`). CPU only, nothing trained, no git, no pod.
Hooks (paths relative to `stack/`): `ResidualPredictor`, `LevelConfig.pred_residual / roll_k / roll_weight`, `HJEPAStack._roll_embedded`, `.roll_pairs`, `._roll_loss`, the term in `HJEPAStack.loss`
(`tanitad/models/hjepa.py`); config fields, validation, `WMAux` build, `probe_pred`, `built_levers`, train-term keys (`tanitad/models/v8f_mini.py`); four flags, `FLAG_CHECKS`, `make_config`,
`INIT_CKPT_MATCH`, `bad_wm_state`, G-LIVE expected terms, checkpoint cfg (`scripts/v8f_mini_train.py`); tests `tests/test_v8f_motion_levers.py`.

## Reading choices (each: it moves no threshold, bar or committed outcome of the brief)

1. **The wrapper is a module, built only when the lever is on.** `ResidualPredictor(inner)` returns `z_ctx + inner(z_ctx, a_ctx)`; the state-dict keys of the level-1 predictor then read `predictors.0.inner.*`.
   Off = no wrapper, the keys are those of the base tree (checked by m1 and by a test). The wrapper holds `latent_dim`, `action_dim`, `context` as plain attributes and passes every other attribute (`in_proj`, `n_cells`, `cell_dim`, ...) through to `inner`.
2. **Zero-init = the LAST linear layer of `inner.out` (weight and bias).** Done after the predictor is built, so no random draw is added or skipped: every other tensor equals the unwrapped model's (tested).
   Consequence: at step 0 the gradient reaches only that layer; the trunk gets gradient from step 1 on (G-LIVE accumulates over 3 steps, so its "every group non-zero" rule holds).
3. **The existing `CausalPredictor(residual=True)` option is not used.** It exists for the flat kind only and has no zero-init; the wrapper covers both kinds the same way.
4. **`probe_pred` reads the kind through the wrapper, and the behavioural cross-cell probe uses a copy of the trunk with a fixed non-zero last layer when the lever is on** (otherwise a zero-initialised last layer makes every cell
   output 0 and the c4 coupling probe of the previous lever would read "no coupling" at init). The probe reads the built module's class and weights, not the flag. The parameter count is that of `inner` (the wrapper adds none).
5. **Roll loss geometry.** Starts `s in range(0, T - K, 2)`; `T` = the number of level-1 frames given to the stack. In the trainer that is `hmax` (the WM sees all `hmax` cached frames: checked by the trainer runs), so the config refuses
   `K > hmax - 1` (16 for WM arms by default: K <= 15). The stack refuses at loss time `K > T - 1` as well. Per start one call of the shared rollout core with batch `B`; stacking all starts on the batch axis was tried first and
   moved the result by 1.7e-6 against `HJEPAStack.rollout` (bar 1e-6), so every start is its own call (m4 then reads 0.0). Cost: S x K sequential predictor calls (m7).
6. **ONE rollout function.** `HJEPAStack.rollout` and the roll loss both call `_roll_embedded(pred, z, e, context)` (embedded actions). `_pred_loss` receives EMBEDDED actions (`enc.act[0] = act_embed(raw)`), so the roll loss takes embedded
   actions and `rollout` embeds raw ones itself: m4 compares `roll_pairs(z, act_embed(raw))` with `rollout(1, z0, raw, context=<predictor context>)`. The context is `pred.context` (3 for every level-1 predictor here); it grows 1 -> 3 and then slides.
7. **Targets are `z[:, s+1 : s+K+1].detach()` always** (the lever has no un-detached form). The start state `z[:, s]` keeps its graph, and so does the action embedding.
8. **Reduction.** Squared error, mean over the width when `pred_reduction == "mean"` (sum otherwise, as `_pred_loss`), then the mean over starts, steps and batch (every element has equal weight). The term enters level 1 as `level_loss + roll_weight * roll`
   (the line the m5 mutation deletes). `out["roll/L1"]` is returned unweighted.
9. **Refusals.** `(K == 0 and W == 0.0) or (K >= 2 and W > 0 and finite)` in `V8FMiniConfig`, in `LevelConfig` use (HJEPAStack) and in the gate (`bad_wm_state`); `K <= hmax - 1`; levers with wm off; levels >= 2 may not set them; a residual with a level-1
   predictor override is refused.
10. **Diagnostics flag.** `wm_l1_diag on` adds `wm_pred_l1` (the level-1 one-step loss, detached), `wm_copy_l1` (copy `z_t` against `z_{t+1}`, same reduction), and with K > 0 `wm_roll_l1` and `wm_roll_copy_l1` (copy of `z_s` against the rollout targets).
    All under `no_grad`. Off = no key. Separately, with K > 0 the trainer always logs one extra term `wm_roll` (= roll/L1, detached) so that G-LIVE can require it (a declared rollout term that never appears is a launch blocker); that key exists only when K > 0.
11. **Checkpoint cfg** records `wm_l1_residual`, `wm_l1_roll_k`, `w_wm_roll`, `wm_l1_diag` only when they differ from their defaults, so default checkpoints keep their key set. `INIT_CKPT_MATCH` gets residual, K and W (an absent key = the default);
    this follows `wm_beta / wm_gamma`: a fine-tune start must have the same lever values. `wm_l1_diag` is not matched (it changes no weights).
12. **Gate reads.** `wm_l1_residual` = "on" iff the BUILT level-1 predictor module is a `ResidualPredictor`; `wm_l1_roll_k` and `w_wm_roll` = the values of `stack.cfg[0]`, which is what `HJEPAStack.loss` reads. A lever declared but not built, built but not
    declared, or a wrong expectation in `gate_extra` (string compare of the built value) is refused. The m6 check shows the three rc values.
13. **Not done / not measured:** nothing on real data, no pod-size model through the trainer (the m2 to m5 checks use the real predictor sizes of arms G and P, CPU; the trainer checks use the tiny synthetic model, d 32).
