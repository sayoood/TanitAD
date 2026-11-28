# Sources read for the REF-C vs DiffusionDrive audit (2026-09-05)

## Papers (banked primaries — library keys, not re-downloaded)

| key | version read | title | library path / sha256 |
|---|---|---|---|
| `2411.15139` | v3 (2025-04-10), HTML full text `https://arxiv.org/html/2411.15139v3` | DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving | `TanitAD Research Lab/Library/papers/2411.15139_DiffusionDrive-Truncated-Diffusion-Model-for-End-to-End-Auto.pdf` · `6ad4f8a379494eeb099cf7d489b75bec1c0ec3321428bcd0b0ca0f05662b8600` |
| `2512.07745` | v1 (2025-12-08), HTML full text `https://arxiv.org/html/2512.07745v1` | DiffusionDriveV2: Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving | `TanitAD Research Lab/Library/papers/2512.07745_DiffusionDriveV2-Reinforcement-Learning-Constrained-Truncate.pdf` · `076ce47e0b0a323c9bb0072432e7f1219ccdcf645b8ff6db172f1bb9efb85023` |

Paper facts used: v1 — Eq. 4 (anchored truncated forward process), Eq. 6 (loss: L1 on the nearest anchor + BCE on the score), 20 anchors (NAVSIM) / 18 (nuScenes) by k-means, "50/1000" truncation, 2 denoising steps at inference, 2 cascade layers (Table 5), N_infer ∈ {10, 20, 40} (Table 6), diversity score Eq. 3 (74 % vs 11 % vanilla), 88.1 PDMS @ 45 FPS. V2 — Eq. 3 (same forward process), Eq. 5 (Gaussian per-step policy), Eq. 7–8 (intra-anchor GRPO, group-relative advantage), Eq. 10 (truncated advantage: −1 on collision, max(0, A)), scale-adaptive multiplicative noise, γ = 0.8, λ_IL = 0.1, mode selector (coarse-to-fine, margin-rank Eq. 11), 91.2 PDMS / 85.5 EPDMS, Table 3 (PDMS@10 75.3 → 84.4).

## Repositories (read at a pinned commit)

### hustvl/DiffusionDrive @ `9b52ed0ec06b073d82d6f392ab084c7b301c8681` (main, 2025-12-08, "Update README.md")

Read VERBATIM (full file returned): `navsim/agents/diffusiondrive/transfuser_model_v2.py` (22,659 B: `V2TransfuserModel`, `AgentHead`, `DiffMotionPlanningRefinementModule`, `ModulationLayer`, `CustomTransformerDecoderLayer`, `CustomTransformerDecoder`, `TrajectoryHead` incl. `forward_train` / `forward_test`), `navsim/agents/diffusiondrive/modules/multimodal_loss.py` (`LossComputer`, `py_sigmoid_focal_loss`), `navsim/agents/diffusiondrive/modules/blocks.py` (`GridSampleCrossBEVAttention`, `gen_sineembed_for_position`, `bias_init_with_prob`), `navsim/agents/diffusiondrive/transfuser_config.py`, `navsim/agents/diffusiondrive/transfuser_loss.py`, `navsim/agents/diffusiondrive/modules/scheduler.py` (an LR scheduler only — the diffusion scheduler is `diffusers.DDIMScheduler`). Read as summaries: `transfuser_backbone.py` (fusion at 4 scales, GPT blocks, top-down BEV decoder), `transfuser_features.py` (status feature = driving_command(4) + velocity(2) + acceleration(2); 1024 × 256 stitched cameras; 0.25 m/px LiDAR histogram).

Key code facts used: `DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample")`; train `timesteps = randint(0, 50)`, `add_noise` on `norm_odo(anchors)`, clamp [−1, 1]; test `trunc_timesteps = 8`, `step_num = 2`, `roll_timesteps = [10, 0]`, `set_timesteps(1000)`; per-layer `(reg, cls)` heads, loss summed over both cascade layers; `traj_points = poses_reg[...,:2].clone().detach()` between layers; selection `argmax(poses_cls[-1])` on `poses_reg[-1]`; `ModulationLayer`: `x·(1+scale)+shift` from `Mish → Linear(time_embed)`; `CustomTransformerDecoderLayer.forward`: BEV grid-sample attention → agent MHA → ego MHA → FFN → time modulation → heads; `LossComputer`: nearest anchor by `norm(dim=-1).mean(dim=-1).argmin`, focal BCE (γ 2, α 0.25) × 10 + L1 × 8; config `tf_d_model 256, tf_num_head 8, tf_d_ffn 1024, num_bounding_boxes 30, trajectory_weight 12`.

### hustvl/DiffusionDriveV2 @ `1cd12a1e155c34dcc471261835444c8d5587580b` (master, 2025-12-29, "Update evaluation script link in train_eval.md")

Tree: `navsim/agents/diffusiondrivev2/{diffusiondrivev2_model_rl.py (52,504 B), diffusiondrivev2_model_sel.py (72,863 B), diffusiondrivev2_rl_agent.py, diffusiondrivev2_rl_config.py, diffusiondrivev2_sel_agent.py, diffusiondrivev2_sel_config.py, transfuser_backbone.py (19,898 B), modules/blocks.py (7,309 B), …}`; `kmeans_navsim_traj_20.npy` (2,688 B = 20 × 8 × 2 float64 + header) at the repo root.

Read VERBATIM: `diffusiondrivev2_rl_config.py` (identical to v1's config plus `num_groups: int = 4`), `modules/blocks.py` (v1's blocks + `gen_sineembed_for_position_1d` + `GridSampleCrossBEVAttentionScorer`). Read as targeted excerpts: `diffusiondrivev2_model_rl.py` — `DDIMScheduler_with_logprob(num_train_timesteps=1000, steps_offset=1, beta_schedule="scaled_linear", prediction_type="sample")`; RL train `step_num = 10`, test `step_num = 2`, `trunc_timesteps = 8`; multiplicative noise `prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add` with `variance_noise_{horizon,vert} = randn([B, M, 1, 1]) * std_dev_t_mul + 1`; `std_dev_t_mul = clip(std_dev_t, min=0.1)`; Gaussian `log_prob` summed over (points, dims); advantages `(r − mean_G) / (std_G + 1e-4)` over `num_groups`; collision / drivable_area failures → −1; `advantages.clamp(min=0) * (reward_group > reward_gt − 1e-6)`; discount `0.8 ** (step_num − i − 1)`; `il_weight = 0.1` (1.0 when a row has no positive sample); reward via `pdm_score_para(metric_cache, trajectory, SIMULATOR, SCORER)` in a process pool returning NC / DAC / EP / TTC / C / driving-direction sub-scores. `diffusiondrivev2_model_sel.py` — coarse `ScorerTransformerDecoder(…, 1)` with `NC/EP/DAC/TTC/C_head`, fine `ScorerTransformerDecoder(…, 3)`; losses BCE per sub-score + `MarginRankingLoss(margin=0.05)` × 2; candidates = 2-step denoised fan + `add_mul_noise(n_aug=2, std 0.1–0.2)` + a GTRS vocabulary subset; inference argmax coarse → top-32 → fine argmax.

## DiffusionDrive noise magnitudes in metres (computed, `raw/` — diffusers `scaled_linear`: β = linspace(√1e-4, √0.02, 1000)², ᾱ = ∏(1−β); `norm_odo` half-ranges x 28.45 m, y 23 m)

| t | √ᾱ | √(1−ᾱ) | σ_x (m) | σ_y (m) |
|---|---|---|---|---|
| 0 | 0.99995 | 0.0100 | 0.284 | 0.230 |
| 8 (inference start) | 0.99950 | 0.0316 | 0.899 | 0.727 |
| 10 (first DDIM step) | 0.99937 | 0.0354 | 1.006 | 0.813 |
| 49 (max train) | 0.99555 | 0.0943 | 2.682 | 2.168 |

## Our side (worktree = off-Drive clone, md5-verified)

`stack/tanitad/refs/refc.py` `547779023248eb6bb36be940fca73dae` · `stack/tanitad/refs/refc_v3.py` `815fdd9aee19289fe04dd591aa85b28c` · `stack/tanitad/refs/refc_tactical.py` `df0d63ef3b955bd910fceb4365a46d23` · `stack/tanitad/models/kinematic.py` `2b0fa0daa6195f2ba90c1f7e94b55422` · `stack/scripts/build_refc_anchors.py` `d00d4cb2dfbcf896ff8221b262eac314`. The numerical check executed the pod-identical tree `C:\Users\Admin\run_refcv3_viz\repo\stack` (`refc.py` md5 `b33656099fd3ace0e4ce17986aac16e1`); `_decode`, the denoise loop (`# Truncated diffusion: refine …` → `# ---- the RANKED score`) and `CrossAttnLayer` are diff-identical between the two trees (268 differing lines elsewhere, all v4 vocabulary / ego-state additions).
