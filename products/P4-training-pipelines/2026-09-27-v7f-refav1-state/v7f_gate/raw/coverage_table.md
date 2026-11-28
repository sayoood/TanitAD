| flag | dest | default | inventory class | DVB kind | status | static read sites | CPU consumers |
|---|---|---|---|---|---|---|---|
| `--v2-val-cache` | `v2_val_cache` | `[]` | DATA | data | **REFUSED-BY-TRAINER** | T:train | train |
| `--a-max` | `a_max` | `4.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--adapter-hidden` | `adapter_hidden` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--agent-slots` | `agent_slots` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--anchor-goal` | `anchor_goal` | `none` | MODEL | built | **COVERED-CPU** | T:_term_precondition, T:build_stack_from_args, T:preflight | _term_precondition, build_stack_from_args, preflight |
| `--anchor-table` | `anchor_table` | `None` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args |
| `--d-goal-embed` | `d_goal_embed` | `128` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--d-str` | `d_str` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--d-t2-hidden` | `d_t2_hidden` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--d-t2-proj` | `d_t2_proj` | `128` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--d-tac` | `d_tac` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-hidden` | `diffusion_hidden` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-noise-rho` | `diffusion_noise_rho` | `0.9` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-sigma-a` | `diffusion_sigma_a` | `2.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-sigma-k` | `diffusion_sigma_k` | `0.1` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-steps` | `diffusion_steps` | `4` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--dt` | `dt` | `0.1` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight, T:train | build_stack_from_args, train |
| `--ema-decay` | `ema_decay` | `0.996` | OPTIM | built | **COVERED-CPU** | T:_ema_tau_record, T:build_stack_from_args, T:dry_run, T:train | build_stack_from_args |
| `--enc-depth` | `enc_depth` | `8` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--enc-dim` | `enc_dim` | `384` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--enc-grad-checkpoint` | `enc_grad_checkpoint` | `auto` | OPTIM | built | **COVERED-CPU** | T:build_stack_from_args | resolve_gc |
| `--enc-heads` | `enc_heads` | `6` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--f-blocks` | `f_blocks` | `3` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--f-hidden-str` | `f_hidden_str` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--f-hidden-tac` | `f_hidden_tac` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--fallback-calibration` | `fallback_calibration` | `None` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--fallback-roll-k` | `fallback_roll_k` | `10` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--fallback-trigger` | `fallback_trigger` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--frame-h` | `frame_h` | `256` | MODEL | built | **COVERED-CPU** | T:_preflight_subframe, T:build_stack_from_args, T:subframe_desync, geo | frame_from_args, build_stack_from_args |
| `--frame-w` | `frame_w` | `640` | LOSS | built | **COVERED-CPU** | T:_preflight_subframe, T:build_stack_from_args, T:subframe_desync, geo | frame_from_args, build_stack_from_args |
| `--goal-cat-args` | `goal_cat_args` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args |
| `--goal-factored` | `goal_factored` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:_preflight_tac_label_all, T:_term_precondition, T:build_stack_from_a | _preflight_tac_label_all, _term_precondition, build_stack_fr |
| `--goal-multilabel` | `goal_multilabel` | `False` | DATA | built | **COVERED-CPU** | T:_preflight_tac_label_all, T:_term_precondition, T:build_stack_from_a | _preflight_tac_label_all, _term_precondition, build_stack_fr |
| `--horizons` | `horizons` | `[1, 2, 4]` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--in-channels` | `in_channels` | `9` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--kappa-max` | `kappa_max` | `0.2` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--max-speed-input-v6` | `max_speed_input_v6` | `False` | (merge) | built | **COVERED-CPU** | T:_preflight_r1_r4, T:assert_r1r4_record, T:build_stack_from_args, T:r | _preflight_r1_r4, assert_r1r4_record, build_stack_from_args, |
| `--mpc-lr` | `mpc_lr` | `0.05` | OPTIM | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-refine` | `mpc_refine` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--mpc-roll-k` | `mpc_roll_k` | `0` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-steps` | `mpc_steps` | `3` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-topk` | `mpc_topk` | `2` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-w-consist` | `mpc_w_consist` | `0.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-w-goal` | `mpc_w_goal` | `1.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-w-kin` | `mpc_w_kin` | `0.1` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-agent-slots` | `n_agent_slots` | `8` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-anchors` | `n_anchors` | `256` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-candidates` | `n_candidates` | `8` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-lat-bins` | `n_lat_bins` | `16` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-registers` | `n_registers` | `4` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-slot-queries` | `n_slot_queries` | `100` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--nav-cond` | `nav_cond` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--newest-frame-only` | `newest_frame_only` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, train_v58f_unicycle_head:build_train_episodes | build_stack_from_args |
| `--no-isolate-interp` | `no_isolate_interp` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--no-isolate-planner` | `no_isolate_planner` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:_preflight_tac_label_all, T:build_stack_from_args, T:preflight | _preflight_tac_label_all, build_stack_from_args, preflight |
| `--no-isolate-uplink` | `no_isolate_uplink` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--o5-target` | `o5_target` | `live` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--param-budget` | `param_budget` | `300000000` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--patch` | `patch` | `16` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--per-layer-encoders` | `per_layer_encoders` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args |
| `--plan-steps` | `plan_steps` | `60` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--plan-vmax-cap` | `plan_vmax_cap` | `False` | (merge) | built | **COVERED-CPU** | T:_preflight_r1_r4, T:assert_r1r4_record, T:build_stack_from_args, T:d | _preflight_r1_r4, assert_r1r4_record, build_stack_from_args, |
| `--plan-wta-eps` | `plan_wta_eps` | `0.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--pred-depth` | `pred_depth` | `6` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--pred-dim` | `pred_dim` | `768` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--pred-heads` | `pred_heads` | `12` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--pred-modern` | `pred_modern` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--proposals` | `proposals` | `query` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--readout-dim` | `readout_dim` | `128` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--readout-grid` | `readout_grid` | `4` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--readout-grid-w` | `readout_grid_w` | `None` | LOSS | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--selector` | `selector` | `none` | UNCLASSIFIED | built | **COVERED-CPU** | T:_term_precondition, T:build_stack_from_args, T:preflight | _term_precondition, build_stack_from_args, preflight |
| `--selector-mlp-hidden` | `selector_mlp_hidden` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--selector-tau-m` | `selector_tau_m` | `1.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--sigreg-free-dims` | `sigreg_free_dims` | `0` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--sigreg-slices` | `sigreg_slices` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--sigreg-subspaces` | `sigreg_subspaces` | `1` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--slot-depth` | `slot_depth` | `3` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--slot-heads` | `slot_heads` | `8` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--slot-hidden` | `slot_hidden` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--slot-src` | `slot_src` | `cells` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--strategic-off` | `strategic_off` | `False` | (merge) | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--t2-contrastive` | `t2_contrastive` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:_term_precondition, T:build_stack_from_args, T:preflight | _term_precondition, build_stack_from_args, preflight |
| `--t2-tau` | `t2_tau` | `0.1` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--tac-goal-cond` | `tac_goal_cond` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--tac-op-cond` | `tac_op_cond` | `off` | (merge) | built | **COVERED-CPU** | T:_preflight_r1_r4, T:build_stack_from_args, T:r1r4_config_block | _preflight_r1_r4, build_stack_from_args, r1r4_config_block |
| `--tac-vocab-version` | `tac_vocab_version` | `v7.0` | MODEL | built | **COVERED-CPU** | T:_preflight_tac_label_all, T:build_stack_from_args | _preflight_tac_label_all, build_stack_from_args |
| `--uplink` | `uplink` | `stopgrad` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--vit5-encoder` | `vit5_encoder` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--w-anchor` | `w_anchor` | `0.0` | LOSS | loss | **COVERED-CPU** | T:_preflight_r1_r4, T:_weights_from_args, T:preflight | _preflight_r1_r4, _weights_from_args, preflight |
| `--w-o14` | `w_o14` | `0.0` | LOSS | loss | **COVERED-CPU** | T:_weights_from_args, T:build_stack_from_args, T:train | _weights_from_args, build_stack_from_args, train |
| `--w-select` | `w_select` | `0.0` | LOSS | loss | **COVERED-CPU** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-t2-contrast` | `w_t2_contrast` | `0.0` | LOSS | loss | **COVERED-CPU** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-tac-label-all` | `w_tac_label_all` | `0.0` | (merge) | loss | **COVERED-CPU** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--window` | `window` | `6` | DATA | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--anchor-axis-w` | `anchor_axis_w` | `[1.0, 1.0]` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--anchor-objective` | `anchor_objective` | `metric` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--bptt-truncate` | `bptt_truncate` | `0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--cond-param` | `cond_param` | `steer_accel_v` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--daccel` | `daccel` | `2.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--dkappa` | `dkappa` | `0.02` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--freeze-encoder` | `freeze_encoder` | `False` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--freeze-readout` | `freeze_readout` | `False` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--grad-checkpoint` | `grad_checkpoint` | `False` | OPTIM | elsewhere | **COVERED-CPU (named guard)** | T:resolve_gc | resolve_gc |
| `--lambda-plan` | `lambda_plan` | `None` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:preflight, T:resolve_lambda_plan, T:v6_weight_specs | resolve_lambda_plan, v6_weight_specs |
| `--o11-k` | `o11_k` | `6` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o11-negs` | `o11_negs` | `1` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o11-tau` | `o11_tau` | `1.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o13-k` | `o13_k` | `4` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o13-seed` | `o13_seed` | `1300` | RUNTIME | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o14-k` | `o14_k` | `4` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run |
| `--o14-mode` | `o14_mode` | `fut` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run |
| `--o1-detach-encoder` | `o1_detach_encoder` | `False` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--o1-k` | `o1_k` | `10` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o1-stopgrad-factual` | `o1_stopgrad_factual` | `False` | OPTIM | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--o2-tau-s` | `o2_tau_s` | `2.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-band-rows` | `o3_band_rows` | `0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-block-h` | `o3_block_h` | `2` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-block-w` | `o3_block_w` | `2` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-blocks` | `o3_blocks` | `2` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-mode` | `o3_mode` | `action` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o5-form` | `o5_form` | `l1` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o5-k` | `o5_k` | `20` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--o5-mode` | `o5_mode` | `uniform` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o5-target-crop` | `o5_target_crop` | `0.0` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:build_stack_from_args, T:train | build_stack_from_args, train |
| `--o6-innovation` | `o6_innovation` | `False` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o6-innovation-shuffle` | `o6_innovation_shuffle` | `False` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--rand-daccel-max` | `rand_daccel_max` | `3.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--rand-dkappa-max` | `rand_dkappa_max` | `0.05` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--rollout-grad-checkpoint` | `rollout_grad_checkpoint` | `auto` | OPTIM | elsewhere | **COVERED-CPU (named guard)** | T:train | resolve_gc |
| `--s1-multi-k` | `s1_multi_k` | `2` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | preflight |
| `--sigreg-accum` | `sigreg_accum` | `1` | OPTIM | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--stage` | `stage` | `None` | RUNTIME | elsewhere | **COVERED-CPU (named guard)** | T:_declared_freeze_preflight, T:_preflight_effective_weights, T:_prefl | _declared_freeze_preflight, _preflight_effective_weights, _p |
| `--t2-negative` | `t2_negative` | `lane_mirror` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--t2-positive` | `t2_positive` | `photometric` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--t5-lag` | `t5_lag` | `0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:preflight, T:train | preflight, train |
| `--t5-pairs` | `t5_pairs` | `False` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:_term_precondition, T:preflight, T:train | _term_precondition, train |
| `--t5-w-kappa` | `t5_w_kappa` | `1.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--w-o10-psg` | `w_o10_psg` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:train, T:v6_weight_specs | train, v6_weight_specs |
| `--w-o11-cf` | `w_o11_cf` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o13-ego` | `w_o13_ego` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o1-ctrl` | `w_o1_ctrl` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o1-fact` | `w_o1_fact` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o1-scene` | `w_o1_scene` | `0.3` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o2` | `w_o2` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o3` | `w_o3` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o5` | `w_o5` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o6` | `w_o6` | `0.1` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o7-distill` | `w_o7_distill` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--w-o8-pixel` | `w_o8_pixel` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--w-o9-ema` | `w_o9_ema` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--w-s1` | `w_s1` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-s1-multi` | `w_s1_multi` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-s2-goal` | `w_s2_goal` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-t1` | `w_t1` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args, T:preflight | _weights_from_args |
| `--w-t5-consist` | `w_t5_consist` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-trunk-anchor` | `w_trunk_anchor` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:assert_trunk_anchor_preflight, T:build_trunk_anchor | assert_trunk_anchor_preflight, build_trunk_anchor |
| `--o14-shuffle-targets` | `o14_shuffle_targets` | `False` | MODEL | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--o7-model` | `o7_model` | `facebook/dinov3-vitl16-p` | UNCLASSIFIED | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--o9-mask-frac` | `o9_mask_frac` | `0.5` | UNCLASSIFIED | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--o9-momentum` | `o9_momentum` | `0.996` | UNCLASSIFIED | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--o9-neighbour-k` | `o9_neighbour_k` | `0` | MODEL | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--psg-enc-only` | `psg_enc_only` | `False` | UNCLASSIFIED | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--psg-eval-every` | `psg_eval_every` | `3` | RUNTIME | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--allow-any-labels` | `allow_any_labels` | `False` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:dry_run, T:preflight, T:train | preflight |
| `--allow-eval-clips-in-train` | `allow_eval_clips_in_train` | `False` | OPTIM | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:_resolve_eval_exclusion | _eval_excl_key, _resolve_eval_exclusion |
| `--cot-negative-sidecar` | `cot_negative_sidecar` | `None` | (merge) | data | **NOT-COVERABLE-ON-CPU (data)** | T:_preflight_tac_label_all, T:tac_label_policy | _preflight_tac_label_all |
| `--domain-strata` | `domain_strata` | `` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:preflight, T:train | preflight, train |
| `--dump-seam-plan` | `dump_seam_plan` | `None` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:_preflight_seam_dump, T:seam_dump_import_error, T:train | seam_dump_import_error, train |
| `--exclude-eval-clips` | `exclude_eval_clips` | `auto` | OPTIM | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:_resolve_eval_exclusion | _eval_excl_key, _resolve_eval_exclusion |
| `--frame-hfov` | `frame_hfov` | `120.0` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | geometry:frame_from_args | frame_from_args |
| `--gate-probes` | `gate_probes` | `None` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--init-encoder-from/--enc-init-from` | `init_encoder_from` | `None` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | T:apply_encoder_seed, T:assert_trunk_anchor_preflight | apply_encoder_seed |
| `--init-from` | `init_from` | `None` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | T:apply_encoder_seed, T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--nav-labels` | `nav_labels` | `None` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:eval_exclusion_roots, T:label_blob_md5_for_vmax, T | _eval_excl_key, eval_exclusion_roots, preflight, train |
| `--prev-gate` | `prev_gate` | `None` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:dry_run, T:train | dry_run, train |
| `--projection` | `projection` | `cylindrical` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | geometry:frame_from_args | frame_from_args |
| `--psg-labels` | `psg_labels` | `None` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:train, T:v6_weight_specs | v6_weight_specs |
| `--require-parity` | `require_parity` | `True` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | train_v58f_unicycle_head:build_train_episodes | - |
| `--no-require-parity` | `require_parity` | `True` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | train_v58f_unicycle_head:build_train_episodes | - |
| `--s2-labels` | `s2_labels` | `None` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:_term_precondition, T:dry_run, T:eval_exclusion_ro | _eval_excl_key, _term_precondition, dry_run, eval_exclusion_ |
| `--speed-max-sidecar-v6` | `speed_max_sidecar_v6` | `None` | (merge) | data | **NOT-COVERABLE-ON-CPU (data)** | T:_preflight_r1_r4, T:dry_run, T:r1r4_config_block, T:train | _preflight_r1_r4, dry_run, r1r4_config_block, train |
| `--t3-scores` | `t3_scores` | `` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:preflight, T:train | preflight, train |
| `--tac-goal-negatives` | `tac_goal_negatives` | `measured` | (merge) | data | **NOT-COVERABLE-ON-CPU (data)** | T:_preflight_tac_label_all, T:tac_label_policy | _preflight_tac_label_all |
| `--trunk-anchor-model` | `trunk_anchor_model` | `facebook/dinov3-vitb16-p` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | T:build_trunk_anchor | - |
| `--v2-cache` | `v2_cache` | `[]` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:_preflight_eval_exclusion, T:_resolve_eval_exclusi | _eval_excl_key, _preflight_eval_exclusion, _resolve_eval_exc |
| `--v2-lru` | `v2_lru` | `64` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | train_v58f_unicycle_head:build_train_episodes | - |
| `--v2-subframe` | `v2_subframe` | `None` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | T:subframe_desync, train_flagship_v4:resolve_v2_frames | resolve_v2_frames, subframe_desync |
| `--batch` | `batch` | `16` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_warn_rank_gate_unrulable, T:train, T:x4_monitor_from_args | _warn_rank_gate_unrulable, train, x4_monitor_from_args |
| `--clip` | `clip` | `1.0` | OPTIM | runtime | **NON-LEVER (runtime)** | T:dry_run, T:train | dry_run, train |
| `--device` | `device` | `cuda` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:_run_config, T:train | _run_config, train |
| `--domain-max-amp` | `domain_max_amp` | `20.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--domain-min-stratum` | `domain_min_stratum` | `8` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--domain-tau` | `domain_tau` | `1.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--dry-batch` | `dry_batch` | `2` | OPTIM | runtime | **NON-LEVER (runtime)** | T:dry_run | dry_run |
| `--dry-k` | `dry_k` | `12` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:dry_run | dry_run |
| `--dry-run` | `dry_run` | `False` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:_preflight_eval_exclusion, T:_preflight_r1_r4, T:_preflight_tac_labe | _preflight_eval_exclusion, _preflight_r1_r4, _preflight_tac_ |
| `--dry-steps` | `dry_steps` | `2` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:dry_run | dry_run |
| `--ema-decay-end` | `ema_decay_end` | `None` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_ema_tau_record, T:build_stack_from_args, T:dry_run, T:train | - |
| `--ema-decay-ramp` | `ema_decay_ramp` | `off` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_ema_tau_record, T:build_stack_from_args, T:dry_run, T:train | build_stack_from_args |
| `--ema-decay-start` | `ema_decay_start` | `0.99` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_ema_tau_record, T:build_stack_from_args, T:dry_run, T:train | - |
| `--eps-per-batch` | `eps_per_batch` | `4` | OPTIM | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--force-rerun` | `force_rerun` | `False` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--log-every` | `log_every` | `50` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--lr` | `lr` | `0.0001` | OPTIM | runtime | **NON-LEVER (runtime)** | T:build_trunk_optimizer | build_trunk_optimizer |
| `--max-horizon` | `max_horizon` | `None` | MODEL | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--nav-semantics` | `nav_semantics` | `t0_constant` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--no-amp` | `no_amp` | `False` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train | - |
| `--no-step-ckpts` | `no_step_ckpts` | `False` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--o4-alpha` | `o4_alpha` | `1.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | train |
| `--o4-floor` | `o4_floor` | `0.25` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--obs-monitor-dims` | `obs_monitor_dims` | `32` | MODEL | runtime | **NON-LEVER (runtime)** | T:build_observer_monitor | - |
| `--obs-monitor-every` | `obs_monitor_every` | `0` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:assert_trunk_anchor_preflight, T:build_observer_monitor | build_observer_monitor |
| `--obs-monitor-window` | `obs_monitor_window` | `256` | DATA | runtime | **NON-LEVER (runtime)** | T:build_observer_monitor | - |
| `--out` | `out` | `None` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--print-launch` | `print_launch` | `False` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:main | main |
| `--refuse-unreached` | `refuse_unreached` | `False` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:dry_run, T:train | dry_run, train |
| `--resume` | `resume` | `auto` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--save-every` | `save_every` | `1000` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--seed` | `seed` | `0` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:build_observer_monitor, T:dry_run, T:train, T:x4_monitor_from_args | dry_run, train |
| `--spectrum-accum` | `spectrum_accum` | `1` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_warn_rank_gate_unrulable, T:train, T:x4_monitor_from_args | _warn_rank_gate_unrulable, train, x4_monitor_from_args |
| `--spectrum-ci-reps` | `spectrum_ci_reps` | `0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train, T:x4_monitor_from_args | train, x4_monitor_from_args |
| `--spectrum-every` | `spectrum_every` | `200` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--steps` | `steps` | `30000` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_ema_tau_record, T:build_lr_scheduler, T:build_stack_from_args, T:dr | build_lr_scheduler, train |
| `--t3-alpha-end` | `t3_alpha_end` | `1.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--t3-alpha-start` | `t3_alpha_start` | `-1.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--t3-floor` | `t3_floor` | `0.25` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | - |
| `--t3-warmup-frac` | `t3_warmup_frac` | `0.5` | OPTIM | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--trunk-lr-scale` | `trunk_lr_scale` | `1.0` | OPTIM | runtime | **NON-LEVER (runtime)** | T:build_trunk_optimizer, T:trunk_lr_factor, T:trunk_lr_split_active | trunk_lr_split_active |
| `--trunk-lr-warmup-steps` | `trunk_lr_warmup_steps` | `0` | OPTIM | runtime | **NON-LEVER (runtime)** | T:build_trunk_optimizer, T:trunk_lr_factor, T:trunk_lr_split_active | trunk_lr_split_active |
| `--wd` | `wd` | `0.05` | OPTIM | runtime | **NON-LEVER (runtime)** | T:build_trunk_optimizer | build_trunk_optimizer |
| `--x4-spectrum-layers` | `x4_spectrum_layers` | `tac,str` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:x4_monitor_from_args | x4_monitor_from_args |
| `--allow-discarded-weights` | `allow_discarded_weights` | `False` | RUNTIME | record | **NON-LEVER (record)** | T:_preflight_effective_weights, T:effective_weights_stamp | _preflight_effective_weights, effective_weights_stamp |
| `--allow-inconclusive-gate` | `allow_inconclusive_gate` | `False` | RUNTIME | record | **NON-LEVER (record)** | T:dry_run, T:preflight, T:train | preflight, train |
| `--allow-unreached` | `allow_unreached` | `[]` | RUNTIME | record | **NON-LEVER (record)** | T:dry_run, T:train | dry_run, train |
| `--i-know-this-is-the-control-arm` | `control_arm_ack` | `False` | MODEL | record | **NON-LEVER (record)** | T:_preflight_r1_r4, T:main, T:preflight | _preflight_r1_r4, main, preflight |
| `--dump-seam-plan-degenerate` | `dump_seam_plan_degenerate` | `False` | UNCLASSIFIED | record | **NON-LEVER (record)** | T:train | - |
| `--expect-n-trainable` | `expect_n_trainable` | `None` | UNCLASSIFIED | record | **NON-LEVER (record)** | T:_declared_freeze_preflight | _declared_freeze_preflight |
| `--gate-off-reason` | `gate_off_reason` | `` | UNCLASSIFIED | record | **NON-LEVER (record)** | T:dry_run, T:preflight, T:train | preflight, train |
| `--i-know-this-arm-predates-nav` | `predates_nav` | `False` | UNCLASSIFIED | record | **NON-LEVER (record)** | T:preflight | - |
