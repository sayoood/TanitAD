# LEADCOND_DEFS: what `--lead-cond {off,on}` does, and every decision with its file:line

Tree: `F:/TanitAD-scratch/v8proof/leadcond/work/tree_patch` (base copy `work/tree_base` = the tree of RULEOWNL1, `ruletrain3/work/tree_patch`; trainer `2ad71c9f...` verified before the work). Line numbers are of tree_patch. Tier of any later read: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react). Evidence class of this file: MEASURED by reading source, plus the checks in `checks/`.

## The lever in one paragraph

`--lead-cond on` (needs `--lead-sup on`) builds one module, `lead_cond = Linear(2, d)`, weight and bias zero. At the current frame the LEADSUP head's outputs are turned into `u = (p, p * g)`, DETACHED, and `lead_cond(u)` is ADDED to the ego conditioning token of T and of O. Zero init means the step-0 forward of a model loaded from a lead-sup checkpoint is the base model's, bit for bit. Default `off`: no module, no state-dict key, no log key, no `levers` key, no selfcheck key, no extra random draw.

## Features u (decision 1)

* Definition (`v8f_mini.py:1106` `V8FMini._lead_u(b, adapt, last)`): the head reads `wm.z1` (`WMAux.z1`) of the `lead_hist` adapted frames `last - lead_hist + 1 .. last`, exactly the window `lead_sup_terms` reads at its end frame (`LeadSupHead.forward` of those frames gives one row, end frame `last`). `p = sigmoid(logit)`; `g` = the head's gap output in the unit it is trained in (gap / 50 m, `LEAD_SUP_GAP_SCALE_M`) clamped to `[0, LEAD_COND_GAP_MAX]` with `LEAD_COND_GAP_MAX = 60 / 50 = 1.2` (`v8f_mini.py:182`; the label range of `v8f_mini_data.lead_sup_label_arrays` is 0 to 60 m). `u = (p, p * g)`, float32. The closing-speed output (column 2) is not read.
* Detached: `o = head(...)[:, 0].detach().float()` (`v8f_mini.py:1117`). The plan loss reaches neither the head nor the trunk through this path (c3 reads the gradient on the head's parameters exactly 0.0).
* Only frames up to `t` influence u: `torch.gather` takes the `lead_hist` frames ending at `last` (`v8f_mini.py:1114-1116`); a frame after `last` is never gathered. `last` is the current frame of the batch (the plan's tick, `Fn - 1`); in training and in inference the batch ends there.
* Real frames only: u is zeros where any of the `lead_hist` frames is not a real frame (`frame_ok`, `v8f_mini.py:1119-1120`), because the head is trained only on windows whose frames are all real (`lead_sup_terms`).

## Insertion point (decision 2)

The conditioning of both T and O enters through the ego token (`e_ego` of the standardised ego features). The lever adds to that token, in the one form the programme already uses (`e_level`, `e_lead` of the level / lead heads: `ego_tok + e(...)`, zero init: `x + 0.0 == x`; an appended token would change `cond.mean(1)`, the query and the softmax keys, so it could not be an identity at step 0):

* T: `run_t`, `v8f_mini.py:1152-1153` (`ego_tok = ego_tok + self.lead_cond(self._lead_u(b, adapt, <current frame>))`), the cond stack then takes `ego_tok` (`:1154`).
* O: `run_o`, `v8f_mini.py:1177-1178` (`ego_tok = ego_tok + self.lead_cond(self._lead_u(b, adapt, i))`; `i` = O's tick frame = the current frame).
* S (strategic, switched off by the PI ruling of 2026-09-19) is not touched; it has no plan output of this kind. T and O each read the token once, so "every plan output, operative and tactical" is two one-line additions to the same token, and `run_t` / `run_o` keep their signatures (callers and probes unchanged). u is computed in both calls from `adapt` and `last`; it is the same value.
* Order of addition in `run_o`: after `e_level` and `e_lead` (when those levers are on), before the cond stack. Neither lever is part of this arm.

## Build and identity (decisions 2 and 4)

* Field `V8FMiniConfig.lead_cond` (`v8f_mini.py:249`), validation: value in `LEAD_COND_MODES`; `on` without `lead_sup on` raises "a lever that is declared and does nothing" (`:371-374`; the trainer exits rc 1).
* Module built LAST in `V8FMini.__init__` (`v8f_mini.py:1043-1049`). The default `nn.Linear` init draws random numbers, so the global RNG state is saved before and restored after (`torch.get_rng_state` / `set_rng_state`); no other module's init and no later draw moves. Then weight and bias are zeroed.
* Checkpoint cfg: `lead_cond` is written into a checkpoint's cfg only when it differs from the default (`LEADCOND_CFG_DEFAULTS`, `v8f_mini.py:184`; trainer `v8f_mini_train.py:3366`), exactly like the LEADSUP fields, so a flag-off checkpoint keeps its cfg key set. An absent key reads as off when loading (`v8f_mini_train.py:756`). `lead_cond` is not in `INIT_CKPT_MATCH`: a fine-tune moves it on purpose.
* `--init-ckpt` from a lead-sup checkpoint without the lever: `NEW_LEVER_KEY_PREFIXES` includes `LEAD_COND_KEY_PREFIXES = ("lead_cond.",)` (`v8f_mini.py:181,191`), so `load_init_ckpt` loads strictly except the two fresh keys `lead_cond.weight` and `lead_cond.bias`; any other missing key, and any unexpected key, raises (`v8f_mini_train.py:~770-780`, unchanged code). `bad_init_ckpt` (`:723-727`) requires, when lead_cond is the only declared new-key lever, that the fresh keys are exactly those two. The record (`fresh_keys`, `declared_new_keys`) is in `selfcheck.json` `levers.init_ckpt` and in the checkpoint.

## Inference path (decision 3)

`forward_infer` calls `_forward`, which calls `run_t` and `run_o`; both compute u from `adapt` (the adapter tokens of the frames in the batch) and `frame_ok`. `v8f_t1_eval.py` calls `forward_infer` as is (unchanged file, same sha256), builds the model from the checkpoint cfg (`V8FMiniConfig(**cfg)`, so `lead_cond: on` rebuilds the module) and loads strictly. The level-1 state `WMAux.z1` is a per-cell projection of the adapter tokens (`v8f_mini.py:z1`): it reads no action, no speed and no ego input in any WM mode, so no combination needs refusing; the lead head exists only with `wm_state grid` (config refusal, unchanged). Inputs of u are image tokens and `frame_ok` only (vision only); c4 perturbs actions, speed and every ego input and reads u unchanged, exactly.

## Gate (decision 5)

| item | where |
|---|---|
| flag, choices `off`/`on`, default `off` | `v8f_mini_train.py:170` |
| G-DVB entry (`FLAG_CHECKS["lead_cond"]`) | `v8f_mini_train.py:76` |
| declared equals built: module exists iff on, `built_levers()["lead_cond"]` equals the declaration, Linear(2, d), zero init at a fresh start, `lead_sup on` required | `bad_lead_cond`, `v8f_mini_train.py:1108-1129`, called from `declared_vs_built` `:1266` |
| built levers (key present only when on) | `v8f_mini.py:1531-1533` -> selfcheck string `levers.lead_cond=on` |
| trainable group `LEADCOND` | `v8f_mini.py:1465` |
| make_config | `v8f_mini_train.py:392` |
| live probe `lead_cond_probe` (G-LIVE, real-data probe batch) | `v8f_mini_train.py:1941`, called `:3261`, recorded as `selfcheck.json["lead_cond_probe"]` only when on (`:3311`) |

The probe reports: `n_rows` (rows with a real lead window), mean and std of p and of g over those rows, `u_abs_max`, `grad_norm_weight` (> 0 required), `grad_norm_bias` (> 0), the control `control_zero_u_grad_norm_weight` (u forced to zeros: exactly 0.0 required, with the bias gradient still > 0 so the probe can read non-zero), and `grad_lead_head_through_u_max_abs_diff` (the plan loss's gradient on the lead head with and without the path: exactly 0.0 required). A violation is `ZZDVB-REFUSEDZZ lead-cond`.

## Interaction with other levers

* `--w-rule-own W --rule-own-form {sq,l1}`: independent (the term acts on the plan after the forward; the token only changes the forward). c7 trains the pair.
* Fine-tune (`--init-ckpt ... --steps N --lr 1e-4 --warmup 100`): c7.
* `--lead-sup-detach on`: allowed; u is detached either way.

## Limits

* Real data was not read: nothing ran on a real cache, the pod or box 2. On the tiny synthetic world p is near 0.5 and g near 0 (the head is barely trained); the probe numbers there say the wiring reaches, not what the real arms will see.
* u at the current frame feeds T although T's own scene tick is `age_t` frames earlier; T's plan is delivered now and the batch holds those frames, so this is allowed by the time rule (frames up to t), and it is the same u for O.
