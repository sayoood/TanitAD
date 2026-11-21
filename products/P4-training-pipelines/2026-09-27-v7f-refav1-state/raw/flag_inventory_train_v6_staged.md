# G-DVB flag inventory — `train_v6_staged.py`

Static AST inventory (no import). **Class = name-pattern PROPOSAL; NO_READ_FOUND = candidate, not proof.** Stack files scanned for forwarded reads: 1126.

**232 flags** · status {'READ': 232} · proposed class {'RUNTIME': 20, 'UNCLASSIFIED': 91, 'MODEL': 60, 'OPTIM': 23, 'LOSS': 28, 'DATA': 10}

## ⛔ NO_READ_FOUND — parsed, never read anywhere in the stack (declared-but-inert CANDIDATES)

| flag | default | proposed class | line |
|---|---|---|---|
| (none) | | | |

## Every flag

| flag | dest | default | class | reads (trainer / stack) | status |
|---|---|---|---|---|---|
| `--allow-any-labels` | `allow_any_labels` | `False` | DATA | 3 / 2 | READ |
| `--goal-multilabel` | `goal_multilabel` | `None` | DATA | 1 / 3 | READ |
| `--nav-labels` | `nav_labels` | `None` | DATA | 6 / 0 | READ |
| `--obs-monitor-window` | `obs_monitor_window` | `256` | DATA | 1 / 0 | READ |
| `--psg-labels` | `psg_labels` | `None` | DATA | 4 / 0 | READ |
| `--s2-labels` | `s2_labels` | `None` | DATA | 9 / 0 | READ |
| `--v2-cache` | `v2_cache` | `[]` | DATA | 8 / 42 | READ |
| `--v2-lru` | `v2_lru` | `64` | DATA | 0 / 20 | READ |
| `--v2-val-cache` | `v2_val_cache` | `[]` | DATA | 1 / 20 | READ |
| `--window` | `window` | `6` | DATA | 26 / 439 | READ |
| `--anchor-axis-w` | `anchor_axis_w` | `<expr> list(ANCHOR_AXIS_W_DEFAULT)` | LOSS | 4 / 2 | READ |
| `--frame-w` | `frame_w` | `640` | LOSS | 5 / 8 | READ |
| `--lambda-plan` | `lambda_plan` | `None` | LOSS | 9 / 6 | READ |
| `--o3-block-w` | `o3_block_w` | `2` | LOSS | 2 / 0 | READ |
| `--readout-grid-w` | `readout_grid_w` | `None` | LOSS | 1 / 0 | READ |
| `--w-anchor` | `w_anchor` | `0.0` | LOSS | 5 / 19 | READ |
| `--w-o10-psg` | `w_o10_psg` | `0.0` | LOSS | 5 / 0 | READ |
| `--w-o11-cf` | `w_o11_cf` | `0.0` | LOSS | 1 / 0 | READ |
| `--w-o13-ego` | `w_o13_ego` | `0.0` | LOSS | 1 / 0 | READ |
| `--w-o14` | `w_o14` | `0.0` | LOSS | 3 / 0 | READ |
| `--w-o1-ctrl` | `w_o1_ctrl` | `1.0` | LOSS | 1 / 0 | READ |
| `--w-o1-fact` | `w_o1_fact` | `1.0` | LOSS | 1 / 0 | READ |
| `--w-o1-scene` | `w_o1_scene` | `0.3` | LOSS | 1 / 0 | READ |
| `--w-o2` | `w_o2` | `1.0` | LOSS | 1 / 0 | READ |
| `--w-o3` | `w_o3` | `1.0` | LOSS | 1 / 0 | READ |
| `--w-o5` | `w_o5` | `1.0` | LOSS | 1 / 3 | READ |
| `--w-o6` | `w_o6` | `0.1` | LOSS | 1 / 0 | READ |
| `--w-o7-distill` | `w_o7_distill` | `0.0` | LOSS | 4 / 1 | READ |
| `--w-o8-pixel` | `w_o8_pixel` | `0.0` | LOSS | 4 / 1 | READ |
| `--w-o9-ema` | `w_o9_ema` | `0.0` | LOSS | 4 / 0 | READ |
| `--w-s1` | `w_s1` | `1.0` | LOSS | 1 / 0 | READ |
| `--w-s1-multi` | `w_s1_multi` | `0.0` | LOSS | 9 / 6 | READ |
| `--w-s2-goal` | `w_s2_goal` | `0.0` | LOSS | 8 / 7 | READ |
| `--w-select` | `w_select` | `0.0` | LOSS | 8 / 7 | READ |
| `--w-t1` | `w_t1` | `1.0` | LOSS | 2 / 0 | READ |
| `--w-t2-contrast` | `w_t2_contrast` | `0.0` | LOSS | 4 / 1 | READ |
| `--w-t5-consist` | `w_t5_consist` | `0.0` | LOSS | 6 / 1 | READ |
| `--w-trunk-anchor` | `w_trunk_anchor` | `0.0` | LOSS | 2 / 2 | READ |
| `--anchor-goal` | `anchor_goal` | `none` | MODEL | 3 / 7 | READ |
| `--anchor-objective` | `anchor_objective` | `metric` | MODEL | 3 / 2 | READ |
| `--anchor-table` | `anchor_table` | `None` | MODEL | 4 / 1 | READ |
| `--cond-param` | `cond_param` | `<expr> COND_INCUMBENT` | MODEL | 2 / 0 | READ |
| `--i-know-this-is-the-control-arm` | `control_arm_ack` | `None` | MODEL | 2 / 0 | READ |
| `--diffusion-sigma-k` | `diffusion_sigma_k` | `0.1` | MODEL | 1 / 2 | READ |
| `--enc-depth` | `enc_depth` | `8` | MODEL | 1 / 1 | READ |
| `--enc-dim` | `enc_dim` | `384` | MODEL | 1 / 1 | READ |
| `--enc-heads` | `enc_heads` | `6` | MODEL | 1 / 1 | READ |
| `--fallback-roll-k` | `fallback_roll_k` | `10` | MODEL | 1 / 5 | READ |
| `--frame-h` | `frame_h` | `256` | MODEL | 5 / 10 | READ |
| `--frame-hfov` | `frame_hfov` | `120.0` | MODEL | 0 / 8 | READ |
| `--freeze-encoder` | `freeze_encoder` | `None` | MODEL | 1 / 0 | READ |
| `--freeze-readout` | `freeze_readout` | `None` | MODEL | 1 / 0 | READ |
| `--horizons` | `horizons` | `[1, 2, 4]` | MODEL | 3 / 165 | READ |
| `--in-channels` | `in_channels` | `9` | MODEL | 6 / 183 | READ |
| `--init-encoder-from/--enc-init-from` | `init_encoder_from` | `None` | MODEL | 6 / 2 | READ |
| `--init-from` | `init_from` | `None` | MODEL | 8 / 0 | READ |
| `--max-horizon` | `max_horizon` | `None` | MODEL | 3 / 80 | READ |
| `--mpc-roll-k` | `mpc_roll_k` | `0` | MODEL | 1 / 7 | READ |
| `--mpc-topk` | `mpc_topk` | `2` | MODEL | 1 / 4 | READ |
| `--n-agent-slots` | `n_agent_slots` | `8` | MODEL | 1 / 5 | READ |
| `--n-anchors` | `n_anchors` | `256` | MODEL | 1 / 47 | READ |
| `--n-candidates` | `n_candidates` | `8` | MODEL | 1 / 34 | READ |
| `--n-lat-bins` | `n_lat_bins` | `16` | MODEL | 1 / 8 | READ |
| `--n-registers` | `n_registers` | `4` | MODEL | 1 / 9 | READ |
| `--n-slot-queries` | `n_slot_queries` | `<expr> N_QUERIES_DEFAULT` | MODEL | 1 / 8 | READ |
| `--nav-cond` | `nav_cond` | `None` | MODEL | 2 / 2 | READ |
| `--newest-frame-only` | `newest_frame_only` | `None` | MODEL | 1 / 2 | READ |
| `--no-isolate-uplink` | `no_isolate_uplink` | `None` | MODEL | 3 / 0 | READ |
| `--o11-k` | `o11_k` | `6` | MODEL | 2 / 0 | READ |
| `--o13-k` | `o13_k` | `4` | MODEL | 2 / 0 | READ |
| `--o14-k` | `o14_k` | `4` | MODEL | 2 / 0 | READ |
| `--o14-shuffle-targets` | `o14_shuffle_targets` | `None` | MODEL | 1 / 0 | READ |
| `--o1-detach-encoder` | `o1_detach_encoder` | `None` | MODEL | 3 / 1 | READ |
| `--o1-k` | `o1_k` | `10` | MODEL | 5 / 0 | READ |
| `--o5-k` | `o5_k` | `20` | MODEL | 9 / 0 | READ |
| `--o5-target` | `o5_target` | `live` | MODEL | 2 / 2 | READ |
| `--o5-target-crop` | `o5_target_crop` | `0.0` | MODEL | 3 / 2 | READ |
| `--o9-neighbour-k` | `o9_neighbour_k` | `0` | MODEL | 1 / 0 | READ |
| `--obs-monitor-dims` | `obs_monitor_dims` | `32` | MODEL | 1 / 0 | READ |
| `--patch` | `patch` | `16` | MODEL | 1 / 15 | READ |
| `--per-layer-encoders` | `per_layer_encoders` | `None` | MODEL | 2 / 0 | READ |
| `--pred-depth` | `pred_depth` | `6` | MODEL | 1 / 0 | READ |
| `--pred-dim` | `pred_dim` | `768` | MODEL | 1 / 0 | READ |
| `--pred-heads` | `pred_heads` | `12` | MODEL | 1 / 0 | READ |
| `--proposals` | `proposals` | `query` | MODEL | 3 / 5 | READ |
| `--readout-dim` | `readout_dim` | `128` | MODEL | 1 / 0 | READ |
| `--readout-grid` | `readout_grid` | `4` | MODEL | 1 / 4 | READ |
| `--s1-multi-k` | `s1_multi_k` | `2` | MODEL | 8 / 0 | READ |
| `--sigreg-free-dims` | `sigreg_free_dims` | `0` | MODEL | 2 / 4 | READ |
| `--slot-depth` | `slot_depth` | `3` | MODEL | 1 / 2 | READ |
| `--slot-heads` | `slot_heads` | `8` | MODEL | 1 / 4 | READ |
| `--t2-positive` | `t2_positive` | `photometric` | MODEL | 2 / 0 | READ |
| `--tac-goal-cond` | `tac_goal_cond` | `None` | MODEL | 2 / 17 | READ |
| `--tac-vocab-version` | `tac_vocab_version` | `v7.0` | MODEL | 1 / 39 | READ |
| `--trunk-anchor-model` | `trunk_anchor_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | MODEL | 1 / 0 | READ |
| `--uplink` | `uplink` | `stopgrad` | MODEL | 1 / 9 | READ |
| `--v2-subframe` | `v2_subframe` | `None` | MODEL | 1 / 9 | READ |
| `--vit5-encoder` | `vit5_encoder` | `None` | MODEL | 1 / 1 | READ |
| `--allow-eval-clips-in-train` | `allow_eval_clips_in_train` | `False` | OPTIM | 2 / 4 | READ |
| `--batch` | `batch` | `16` | OPTIM | 9 / 167 | READ |
| `--clip` | `clip` | `1.0` | OPTIM | 2 / 125 | READ |
| `--dry-batch` | `dry_batch` | `2` | OPTIM | 4 / 0 | READ |
| `--ema-decay` | `ema_decay` | `0.996` | OPTIM | 8 / 11 | READ |
| `--ema-decay-end` | `ema_decay_end` | `None` | OPTIM | 4 / 5 | READ |
| `--ema-decay-ramp` | `ema_decay_ramp` | `off` | OPTIM | 5 / 1 | READ |
| `--ema-decay-start` | `ema_decay_start` | `0.99` | OPTIM | 4 / 1 | READ |
| `--enc-grad-checkpoint` | `enc_grad_checkpoint` | `auto` | OPTIM | 1 / 5 | READ |
| `--eps-per-batch` | `eps_per_batch` | `4` | OPTIM | 4 / 8 | READ |
| `--exclude-eval-clips` | `exclude_eval_clips` | `auto` | OPTIM | 2 / 1 | READ |
| `--grad-checkpoint` | `grad_checkpoint` | `None` | OPTIM | 2 / 19 | READ |
| `--lr` | `lr` | `0.0001` | OPTIM | 4 / 76 | READ |
| `--mpc-lr` | `mpc_lr` | `0.05` | OPTIM | 1 / 3 | READ |
| `--o1-stopgrad-factual` | `o1_stopgrad_factual` | `None` | OPTIM | 3 / 0 | READ |
| `--rollout-grad-checkpoint` | `rollout_grad_checkpoint` | `auto` | OPTIM | 1 / 4 | READ |
| `--sigreg-accum` | `sigreg_accum` | `1` | OPTIM | 5 / 0 | READ |
| `--spectrum-accum` | `spectrum_accum` | `1` | OPTIM | 6 / 0 | READ |
| `--steps` | `steps` | `30000` | OPTIM | 14 / 153 | READ |
| `--t3-warmup-frac` | `t3_warmup_frac` | `0.5` | OPTIM | 2 / 0 | READ |
| `--trunk-lr-scale` | `trunk_lr_scale` | `1.0` | OPTIM | 3 / 1 | READ |
| `--trunk-lr-warmup-steps` | `trunk_lr_warmup_steps` | `0` | OPTIM | 3 / 1 | READ |
| `--wd` | `wd` | `0.05` | OPTIM | 2 / 7 | READ |
| `--allow-discarded-weights` | `allow_discarded_weights` | `None` | RUNTIME | 2 / 2 | READ |
| `--allow-inconclusive-gate` | `allow_inconclusive_gate` | `None` | RUNTIME | 3 / 3 | READ |
| `--allow-unreached` | `allow_unreached` | `[]` | RUNTIME | 2 / 2 | READ |
| `--device` | `device` | `cuda` | RUNTIME | 28 / 658 | READ |
| `--dry-k` | `dry_k` | `12` | RUNTIME | 3 / 0 | READ |
| `--dry-run` | `dry_run` | `None` | RUNTIME | 8 / 11 | READ |
| `--dry-steps` | `dry_steps` | `2` | RUNTIME | 1 / 1 | READ |
| `--log-every` | `log_every` | `50` | RUNTIME | 1 / 34 | READ |
| `--no-step-ckpts` | `no_step_ckpts` | `None` | RUNTIME | 1 / 1 | READ |
| `--o13-seed` | `o13_seed` | `1300` | RUNTIME | 2 / 0 | READ |
| `--obs-monitor-every` | `obs_monitor_every` | `0` | RUNTIME | 2 / 1 | READ |
| `--out` | `out` | `None` | RUNTIME | 5 / 702 | READ |
| `--print-launch` | `print_launch` | `None` | RUNTIME | 1 / 1 | READ |
| `--psg-eval-every` | `psg_eval_every` | `3` | RUNTIME | 3 / 0 | READ |
| `--refuse-unreached` | `refuse_unreached` | `None` | RUNTIME | 2 / 2 | READ |
| `--resume` | `resume` | `auto` | RUNTIME | 1 / 5 | READ |
| `--save-every` | `save_every` | `1000` | RUNTIME | 1 / 32 | READ |
| `--seed` | `seed` | `0` | RUNTIME | 13 / 213 | READ |
| `--spectrum-every` | `spectrum_every` | `200` | RUNTIME | 3 / 0 | READ |
| `--stage` | `stage` | `None` | RUNTIME | 71 / 59 | READ |
| `--a-max` | `a_max` | `<expr> A_MAX` | UNCLASSIFIED | 1 / 24 | READ |
| `--adapter-hidden` | `adapter_hidden` | `512` | UNCLASSIFIED | 1 / 9 | READ |
| `--agent-slots` | `agent_slots` | `None` | UNCLASSIFIED | 4 / 13 | READ |
| `--bptt-truncate` | `bptt_truncate` | `0` | UNCLASSIFIED | 3 / 11 | READ |
| `--d-goal-embed` | `d_goal_embed` | `128` | UNCLASSIFIED | 1 / 58 | READ |
| `--d-str` | `d_str` | `256` | UNCLASSIFIED | 4 / 38 | READ |
| `--d-t2-hidden` | `d_t2_hidden` | `256` | UNCLASSIFIED | 1 / 4 | READ |
| `--d-t2-proj` | `d_t2_proj` | `128` | UNCLASSIFIED | 1 / 3 | READ |
| `--d-tac` | `d_tac` | `512` | UNCLASSIFIED | 3 / 65 | READ |
| `--daccel` | `daccel` | `<expr> DACCEL_DEFAULT` | UNCLASSIFIED | 2 / 10 | READ |
| `--diffusion-hidden` | `diffusion_hidden` | `256` | UNCLASSIFIED | 1 / 2 | READ |
| `--diffusion-noise-rho` | `diffusion_noise_rho` | `0.9` | UNCLASSIFIED | 1 / 5 | READ |
| `--diffusion-sigma-a` | `diffusion_sigma_a` | `2.0` | UNCLASSIFIED | 1 / 2 | READ |
| `--diffusion-steps` | `diffusion_steps` | `4` | UNCLASSIFIED | 1 / 27 | READ |
| `--dkappa` | `dkappa` | `<expr> DKAPPA_DEFAULT` | UNCLASSIFIED | 2 / 10 | READ |
| `--domain-max-amp` | `domain_max_amp` | `<expr> DOMAIN_MIX_MAX_AMPLIFICATION` | UNCLASSIFIED | 2 / 0 | READ |
| `--domain-min-stratum` | `domain_min_stratum` | `<expr> DOMAIN_MIX_MIN_STRATUM_EPISODES` | UNCLASSIFIED | 2 / 0 | READ |
| `--domain-strata` | `domain_strata` | `` | UNCLASSIFIED | 3 / 0 | READ |
| `--domain-tau` | `domain_tau` | `1.0` | UNCLASSIFIED | 4 / 0 | READ |
| `--dt` | `dt` | `0.1` | UNCLASSIFIED | 8 / 140 | READ |
| `--dump-seam-plan` | `dump_seam_plan` | `None` | UNCLASSIFIED | 4 / 3 | READ |
| `--dump-seam-plan-degenerate` | `dump_seam_plan_degenerate` | `None` | UNCLASSIFIED | 1 / 0 | READ |
| `--expect-n-trainable` | `expect_n_trainable` | `None` | UNCLASSIFIED | 1 / 0 | READ |
| `--f-blocks` | `f_blocks` | `3` | UNCLASSIFIED | 1 / 6 | READ |
| `--f-hidden-str` | `f_hidden_str` | `512` | UNCLASSIFIED | 1 / 2 | READ |
| `--f-hidden-tac` | `f_hidden_tac` | `512` | UNCLASSIFIED | 1 / 3 | READ |
| `--fallback-calibration` | `fallback_calibration` | `None` | UNCLASSIFIED | 8 / 1 | READ |
| `--fallback-trigger` | `fallback_trigger` | `None` | UNCLASSIFIED | 3 / 3 | READ |
| `--force-rerun` | `force_rerun` | `None` | UNCLASSIFIED | 1 / 0 | READ |
| `--gate-off-reason` | `gate_off_reason` | `` | UNCLASSIFIED | 3 / 3 | READ |
| `--gate-probes` | `gate_probes` | `None` | UNCLASSIFIED | 5 / 0 | READ |
| `--goal-cat-args` | `goal_cat_args` | `None` | UNCLASSIFIED | 2 / 5 | READ |
| `--goal-factored` | `goal_factored` | `None` | UNCLASSIFIED | 1 / 4 | READ |
| `--kappa-max` | `kappa_max` | `<expr> KAPPA_MAX` | UNCLASSIFIED | 1 / 36 | READ |
| `--mpc-refine` | `mpc_refine` | `None` | UNCLASSIFIED | 2 / 5 | READ |
| `--mpc-steps` | `mpc_steps` | `3` | UNCLASSIFIED | 1 / 3 | READ |
| `--mpc-w-consist` | `mpc_w_consist` | `0.0` | UNCLASSIFIED | 1 / 8 | READ |
| `--mpc-w-goal` | `mpc_w_goal` | `1.0` | UNCLASSIFIED | 1 / 4 | READ |
| `--mpc-w-kin` | `mpc_w_kin` | `0.1` | UNCLASSIFIED | 1 / 3 | READ |
| `--nav-semantics` | `nav_semantics` | `t0_constant` | UNCLASSIFIED | 1 / 0 | READ |
| `--no-amp` | `no_amp` | `None` | UNCLASSIFIED | 1 / 18 | READ |
| `--no-isolate-interp` | `no_isolate_interp` | `None` | UNCLASSIFIED | 3 / 0 | READ |
| `--no-isolate-planner` | `no_isolate_planner` | `None` | UNCLASSIFIED | 4 / 0 | READ |
| `--o11-negs` | `o11_negs` | `1` | UNCLASSIFIED | 2 / 0 | READ |
| `--o11-tau` | `o11_tau` | `1.0` | UNCLASSIFIED | 2 / 0 | READ |
| `--o14-mode` | `o14_mode` | `fut` | UNCLASSIFIED | 2 / 0 | READ |
| `--o2-tau-s` | `o2_tau_s` | `2.0` | UNCLASSIFIED | 2 / 0 | READ |
| `--o3-band-rows` | `o3_band_rows` | `0` | UNCLASSIFIED | 2 / 0 | READ |
| `--o3-block-h` | `o3_block_h` | `2` | UNCLASSIFIED | 2 / 0 | READ |
| `--o3-blocks` | `o3_blocks` | `2` | UNCLASSIFIED | 2 / 0 | READ |
| `--o3-mode` | `o3_mode` | `action` | UNCLASSIFIED | 2 / 0 | READ |
| `--o4-alpha` | `o4_alpha` | `1.0` | UNCLASSIFIED | 6 / 0 | READ |
| `--o4-floor` | `o4_floor` | `0.25` | UNCLASSIFIED | 1 / 0 | READ |
| `--o5-form` | `o5_form` | `l1` | UNCLASSIFIED | 2 / 0 | READ |
| `--o5-mode` | `o5_mode` | `uniform` | UNCLASSIFIED | 2 / 0 | READ |
| `--o6-innovation` | `o6_innovation` | `None` | UNCLASSIFIED | 2 / 3 | READ |
| `--o6-innovation-shuffle` | `o6_innovation_shuffle` | `None` | UNCLASSIFIED | 2 / 3 | READ |
| `--o7-model` | `o7_model` | `<expr> O7_DEFAULT_MODEL` | UNCLASSIFIED | 1 / 0 | READ |
| `--o9-mask-frac` | `o9_mask_frac` | `0.5` | UNCLASSIFIED | 1 / 0 | READ |
| `--o9-momentum` | `o9_momentum` | `0.996` | UNCLASSIFIED | 1 / 0 | READ |
| `--param-budget` | `param_budget` | `300000000` | UNCLASSIFIED | 1 / 3 | READ |
| `--plan-steps` | `plan_steps` | `<expr> PLAN_STEPS` | UNCLASSIFIED | 11 / 85 | READ |
| `--plan-wta-eps` | `plan_wta_eps` | `0.0` | UNCLASSIFIED | 3 / 6 | READ |
| `--pred-modern` | `pred_modern` | `None` | UNCLASSIFIED | 1 / 0 | READ |
| `--i-know-this-arm-predates-nav` | `predates_nav` | `None` | UNCLASSIFIED | 1 / 1 | READ |
| `--prev-gate` | `prev_gate` | `None` | UNCLASSIFIED | 3 / 0 | READ |
| `--projection` | `projection` | `cylindrical` | UNCLASSIFIED | 0 / 80 | READ |
| `--psg-enc-only` | `psg_enc_only` | `None` | UNCLASSIFIED | 1 / 0 | READ |
| `--rand-daccel-max` | `rand_daccel_max` | `3.0` | UNCLASSIFIED | 2 / 2 | READ |
| `--rand-dkappa-max` | `rand_dkappa_max` | `0.05` | UNCLASSIFIED | 2 / 2 | READ |
| `--require-parity` | `require_parity` | `True` | UNCLASSIFIED | 0 / 19 | READ |
| `--no-require-parity` | `require_parity` | `None` | UNCLASSIFIED | 0 / 19 | READ |
| `--selector` | `selector` | `none` | UNCLASSIFIED | 12 / 31 | READ |
| `--selector-mlp-hidden` | `selector_mlp_hidden` | `256` | UNCLASSIFIED | 1 / 6 | READ |
| `--selector-tau-m` | `selector_tau_m` | `1.0` | UNCLASSIFIED | 1 / 6 | READ |
| `--sigreg-slices` | `sigreg_slices` | `512` | UNCLASSIFIED | 1 / 4 | READ |
| `--sigreg-subspaces` | `sigreg_subspaces` | `1` | UNCLASSIFIED | 1 / 1 | READ |
| `--slot-hidden` | `slot_hidden` | `256` | UNCLASSIFIED | 1 / 4 | READ |
| `--slot-src` | `slot_src` | `cells` | UNCLASSIFIED | 3 / 6 | READ |
| `--spectrum-ci-reps` | `spectrum_ci_reps` | `0` | UNCLASSIFIED | 6 / 0 | READ |
| `--t2-contrastive` | `t2_contrastive` | `None` | UNCLASSIFIED | 4 / 2 | READ |
| `--t2-negative` | `t2_negative` | `lane_mirror` | UNCLASSIFIED | 2 / 0 | READ |
| `--t2-tau` | `t2_tau` | `0.1` | UNCLASSIFIED | 1 / 1 | READ |
| `--t3-alpha-end` | `t3_alpha_end` | `1.0` | UNCLASSIFIED | 4 / 0 | READ |
| `--t3-alpha-start` | `t3_alpha_start` | `-1.0` | UNCLASSIFIED | 4 / 0 | READ |
| `--t3-floor` | `t3_floor` | `0.25` | UNCLASSIFIED | 3 / 0 | READ |
| `--t3-scores` | `t3_scores` | `` | UNCLASSIFIED | 3 / 0 | READ |
| `--t5-lag` | `t5_lag` | `0` | UNCLASSIFIED | 3 / 0 | READ |
| `--t5-pairs` | `t5_pairs` | `None` | UNCLASSIFIED | 4 / 0 | READ |
| `--t5-w-kappa` | `t5_w_kappa` | `1.0` | UNCLASSIFIED | 2 / 0 | READ |
| `--x4-spectrum-layers` | `x4_spectrum_layers` | `tac,str` | UNCLASSIFIED | 1 / 1 | READ |
