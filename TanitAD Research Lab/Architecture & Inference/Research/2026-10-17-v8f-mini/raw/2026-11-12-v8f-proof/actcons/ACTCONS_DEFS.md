# ACTCONS_DEFS: what `--w-act-cons W` / `--act-cons-max M` does, and every decision with its file:line

Card: `F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD37.md` section 2 (lever), section 5 (controls a1 to a3 built here; a4 is the pod gate). Tree: `F:/TanitAD-scratch/v8proof/actcons/work/tree_patch` (base copy `work/tree_base` = the tree of LEADCOND, trainer sha256 `85035ca4...` verified on both copies, all 12 pinned files of `leadcond/ARMS_LEADCOND.txt` equal). Line numbers are of tree_patch. Evidence class of this file: MEASURED by reading source, plus the checks in `checks/`. Tier of any later read: T0 (world-model diagnostics on logged frames), family read at T1-kin.

## The lever in one paragraph

A TRAINING-ONLY term. It adds no module, no parameter, no state-dict key and no config field, so the model, the checkpoint config and every inference output are the base tree's. The trainer calls `act_cons.act_cons_term` after `model.loss` and adds `W * term` to the total (the same place and style as RULEOWN). Default `--w-act-cons 0.0`: the block is not entered (no tensor op, no random draw, no log key, no selfcheck key, no summary key).

## The term (file `stack/tanitad/models/act_cons.py`, new, 123 lines)

* Start frames: `valid_starts(F, 10) = range(0, F - 10, 2)` (`act_cons.py:30`), the start rule of the existing rollout loss (`hjepa.py:966`, `starts = list(range(0, t_ - k, 2))`). With the proof window F = hmax = 16: starts 0, 2, 4. The rule applies even when the arm has no rollout loss (`--wm-l1-roll-k` is not needed).
* Rollouts: the stack's own `HJEPAStack.roll_pairs(z, e, 10)` (`hjepa.py:956`), the rollout core of the rollout loss and of the read tool (`imag_dump.py`'s `control_c2` compares it with the tool's rollout: <= 1e-5). The context is the predictor's own (`hjepa.py:965`). Reference rollout under `torch.no_grad()` (`act_cons.py:111`); offset rollout with gradient (`:112`). Start state = `wm.z1(adapt.detach())` computed inside `torch.no_grad()` (`:98-99`): the start state and everything before it receive exactly 0.0.
* Offset: one d per sample, `draw_offsets(seed, step, n, max)` (`act_cons.py:35`): `U(-max, +max)` from a DEDICATED `torch.Generator` seeded `(seed * 1_000_003 + step) XOR 0x41435443` and created fresh each step. The global generator is not touched (test `test_offsets_dedicated_generator_range_and_global_rng_untouched` reads the global state before and after); a resumed run draws the same offsets. Constant over the 10 steps and shared by the 3 start frames of the sample. It enters as `d / WM_ACC_SCALE` on the acceleration channel of the scaled level-1 action (`act_cons.py:109-112`), yaw unchanged. `WM_ACC_SCALE = 2.0` is `v8f_mini.py:206`, used at `WMAux.scale_actions`, `v8f_mini.py:957` (`act[..., 0] / WM_ACC_SCALE`). The action of transition j to j+1 is `scale_actions(act)[:, :-1]`, as in `WMAux.forward`.
* Head reads: `functional_call(head, {n: p.detach()}, x)` (`act_cons.py:86,113`): the lead head's parameters get no gradient from this term; its own supervised loss in the same total keeps its gradient (the term never touches `wm.lead_head`'s `.grad`; check a2). Reads: steps 1..5 (T = 1.0 s) and 6..10 (T = 2.0 s) = the 5 states ending at step 5 and at step 10, gap = output[1] x `LEAD_SUP_GAP_SCALE_M` (50.0, `v8f_mini.py:172`), the same reads as `imag_dump.py` `head_reads_rolled`.
* Term: `mean over valid (sample, start) pairs and the two times of | gap_offset - gap_reference + 0.5 d T^2 |` in metres (`act_cons.py:120`; `tot / (2 n_valid)`). No valid pair: `torch.zeros(())` with no graph, `n_valid` 0.
* Valid (`valid_mask`, `act_cons.py:73`): label presence > 0.5, finite gap with `0 <= gap <= 60`, `frame_ok` at frame s, `wm_speed[:, s] >= max * 2.0`. Batch keys read: `lead_pres`, `lead_gapm`, `wm_speed`, `frame_ok`, `act` (all already in the lead-sup batch).
* Precision: computed in float32 with autocast off, on the device of `adapt` (the term creates every tensor from its inputs; `d` is moved to that device).
* Rows with no valid start are dropped before the rollouts (`z[rows]`), so the cost is two rollouts of the valid rows only; invalid pairs inside a kept row are multiplied by 0.

## Trainer wiring (`stack/scripts/v8f_mini_train.py`, changed)

| item | where |
|---|---|
| import `AC` | `:49` |
| G-DVB entries `w_act_cons`, `act_cons_max` (kind `data:act_cons`) | `:130` |
| flags (defaults 0.0 and 1.5) | `:300`, `:305` |
| `bad_act_cons` (declared == built record; W 0 with a non-default max refused; needs lead head, grid, acc_yaw, lead_hist 5, window >= 11 frames, no dropout; refuses `--ego-perturb-p`, `--dagger-p`, `--recover-p` != 0 and `--encoder lora`) | `:1265`, called from `declared_vs_built` `:1310` |
| built record `info["act_cons"]` (read from the model: head width, start set, reads, units) | `:3271` via `AC.build_record` (`act_cons.py:64`) |
| live probe `act_cons_probe` (on `copy.deepcopy(model).cpu()`) | `:2049`, called `:3377`, recorded as `selfcheck.json["act_cons_probe"]` only when on `:3432` |
| selfcheck `levers.w_act_cons`, `levers.act_cons_max` (only when on) | `:3404-3405` |
| the loop block: offsets, term, `T["total"] = tot_ac + a.w_act_cons * ac_r["term"]`, `T["act_cons"]`, `T["act_cons_n"]`, not-applied guard `ZZACT-CONS-NOT-APPLIEDZZ` | `:3614-3625` |
| run totals, summary keys `w_act_cons`, `act_cons_max`, `act_cons_n_valid`, `act_cons_term_mean` (only when on) | `:3450`, `:3691` |

Train-log keys (only when on): `act_cons` (the per-step term, metres) and `act_cons_n` (valid (sample, start) pairs of that step). The existing keys and their values are unchanged.

The probe reports: `term`, `n_valid`, `n_rows`, `starts`, `woken`, `grad_norm_predictor` (> 0), `grad_norm_action_embed` (> 0), `grad_norm_lead_head` (exactly 0.0), `grad_norm_state_projection` (exactly 0.0), `grad_abs_sum_outside_predictor_and_embed` (exactly 0.0: every other parameter incl. adapter, encoder-path, stack heads), the all-invalid control (`control_all_invalid_term` exactly 0.0, no graph) and `pass`. A violation is `ZZDVB-REFUSEDZZ act-cons`. On a fresh adaLN-zero predictor the action input has no path yet (zero gates); then the same read is made on a copy whose all-zero predictor tensors are woken with small noise and `woken` is recorded `true` (it is `false` on the tiny synthetic model and on any fine-tune from a trained checkpoint). LESSON of LEADCOND honoured: the probe batch is on the CPU, so it runs on a CPU copy; the term inside the training step takes the device of `adapt`.

## Admissibility on a fine-tune (`--init-ckpt`)

Nothing in the model or the checkpoint changes, so a checkpoint whose config lacks any new field loads exactly as for LFT: no new key, no `INIT_CKPT_MATCH` entry, no `NEW_LEVER_KEY_PREFIXES` entry, `fresh_keys` empty (measured, `checks/a4_smoke_finetune.json`: `init_ckpt_fresh_keys` null, checkpoint cfg equal to the base checkpoint's cfg, state keys equal, strict load into a model built from that cfg, and the unchanged `v8f_t1_eval.py` loads and runs it).

## Departures from the card (each named)

1. **`frame_ok` at frame s is part of "valid".** The card lists a label lead and a speed bound. The batch holds clamped copies of frame 0 before a clip starts, with real-looking labels, masked in the dataset only by `frame_ok` (`v8f_mini_data.py:670` comment); such a frame has no label lead at that time, so it is not valid. On the proof windows (all 16 frames real unless a clip is shorter) it removes nothing; on padded windows it removes the padded starts.
2. **"Between 0 and 60 m" is read as inclusive** (`0 <= gap <= 60`).
3. **One offset per sample is shared by its three start frames** (the card says one offset per sample, constant over the 10 steps; it does not say per start). The valid pairs are averaged over (sample, start) as the card says "mean over valid samples and the two times": a sample with two valid starts counts twice (the card's own "for each start frame s of the start set" reading); `act_cons_n` counts pairs, not samples (`n_rows` in the probe counts samples).
4. **The term is not inside `model.loss`.** It lives in the trainer next to the RULEOWN term, so the model, the checkpoint config and the key sets are exactly the base tree's. Consequence: the existing `live_check` (G-LIVE over `model.loss`) does not contain it; the term has its own probe (`act_cons_probe`) and its own G-DVB clause.
5. **Under `--amp` (a pod run in bf16) the term is computed in float32** (autocast off), from `out["adapt"]`; the WM's own `z1` of that step is not re-used (the term recomputes `wm.z1` from the same adapter tokens, no gradient): one extra projection per step.
6. **Refusals beyond the card:** `--lead-hist` other than 5 (the 1.0 s and 2.0 s reads are 5-state windows), `--wm-l1-act speed`, window shorter than 11 frames, predictor dropout, `--ego-perturb-p / --dagger-p / --recover-p` > 0 (they rewrite the logged actions the rollouts read), `--encoder lora`. None applies to LA / LA01.
7. a4 (pod gate on real data) is not run here (no pod). What the gate will see: `levers.w_act_cons`, `levers.act_cons_max`, `act_cons_probe`, the log keys above.

## Not verified here

No real data, no pod, no real checkpoint: the tiny synthetic world only (n_valid 11 of 24 start pairs in the probe batch there). Real-data statistics of the term (its magnitude against the base predictor, the share of valid pairs, the gradient norms) are first seen in the pod selfcheck of LA_s0; on the tiny model the gradient norm on the predictor is large (18,918) because the model is untrained, which says nothing about real arms.
