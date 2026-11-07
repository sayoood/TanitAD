# REPORT_B: how driving and hierarchical world models validate what they imagine (Research Lab agent B, 2026-11-07; written to file by the Master Mind from the agent's hand-back, because the harness blocked the agent's own Write)

Evidence key:
* [PUB k] = read in the banked PDF text (pdftotext -layout; row labels can shift).
* [PUB-SEC] = abstract or index record only.
* [HYP] = the agent's inference.
* "held" = already in `library.json` before this pass.

25 new papers banked into the staging Library `F:/TanitAD-scratch/kbB` (`BANKED_B.txt`).

## Top 5 for our gaps (agent's ranking)
1. **Level-wise probe NMSE with seeds, plus decoded start, goal and predicted latents.** H-JEPA 2610.06805 (held); Clockwork VAE 2102.09532. Closes: which level holds what. Data: speed, acceleration, yaw rate, agent count, lead gap per window. PhysicalAI-AV has all of them (ego motion + cuboids).
2. **Action-consistency inverse probe, counterfactual ego actions, kinematic null.** ATM 2606.09028, iKCE 2607.05966, LAW 2406.08481 (all held). Turns the wrong-sign lead gap and the L1 overshoot into numbers. Needs ego actions only.
3. **A scene decoder on cuboid-built occupancy**, conditioned on the ego trajectory, scored by per-horizon IoU with a Copy&Paste control. OccWorld 2311.16038, MILE 2210.07729, Cam4DOcc 2311.17663. Cuboids can be rasterised; there is no static map layer.
4. **Multi-frame history with a causal-mask check.** DINO-WM 2411.04983 Table 6, DOME 2410.10429 Table 5, Epona 2506.24113 Table 7. Available in PhysicalAI-AV.
5. **Action-diverse windows, and a shared latent for the planner and the next-state head.** DOME resampling, Epona joint training (PDMS 78.1 → 86.2), WoTE trajectory scoring 2504.01941. Count turning vs straight windows first.
* Extra control: Mind the Gap 2607.12547 feeds the TRUE future subgoal to the low level, which separates a high-level fault from a low-level fault.

## 1. Driving latent, occupancy and generative models
| Paper | Decoded / measured | Protocol, controls | Numbers | Action test? | Map to us |
|---|---|---|---|---|---|
| MILE 2210.07729 (held) | Latent decoded to BEV segmentation; sampled futures; driving in imagination | CARLA, new town and weather; futures shown qualitatively | +31 % driving score [PUB abstract] | Plans in imagination, qualitative | BEV decode per level from cuboid occupancy |
| LAW 2406.08481 (held) | Future scene latent from the current latent + ego trajectory; scored only by planning | nuScenes Table 4: no model / latent only / latent + predicted trajectory | Perception-free avg L2 0.71 → 0.68 → 0.61 m, collision 0.41 → 0.37 → 0.30 %; perception-based 0.54 → 0.52 m [PUB Table 4, layout] | Yes, by ablation: the latent prediction depends on actions | The closest design; it never decodes |
| WoTE 2504.01941 (held) | BEV world model predicts the future BEV per candidate trajectory | NAVSIM; future-step ablation | PDMS 84.0 (0, 4 s) vs 85.6 (0, 2, 4 s) [PUB Table 5] | Per trajectory | Trajectory-scoring head on levels 2 and 3 |
| OccWorld 2311.16038 | Occupancy tokens + ego trajectory | mIoU / IoU at 1 / 2 / 3 s; **a Copy&Paste row**; Table 4 | 3 s avg IoU 26.63, mIoU 17.13, L2 1.16 m [PUB abstract]; without ego, mIoU 17.14 → 15.13 [PUB Table 4] | Ego token ablated | Per-horizon table + Copy&Paste control |
| OccSora 2405.20337 | 4D occupancy from a trajectory | The FID table mixes 2D, video and 4D (DiT 12.03, DriveDreamer 14.9, DriveGAN 27.8) | [PUB Table 2] | Qualitative | Cross-modality FID is not valid; do not copy it |
| DOME 2410.10429 | Diffusion occupancy, 64 frames at 2 Hz | Table 3 ablation | mIoU 13.08 → 18.60 → 24.24 (+ trajectory) → 27.00 (+ resampling) → 27.10; +36.0 % mIoU vs baseline [PUB] | Yes. 87 % of nuScenes training scenes go straight and each scene is driven once, so the model ignores the trajectory; the fix is resampling [PUB] | Matches our weakly imagined steering |
| Copilot4D 2311.01017 | LiDAR tokens, discrete diffusion | Chamfer at 1 s / 3 s | > 65 % / > 50 % lower than the prior state of the art [PUB abstract] | Ego pose [PUB-SEC] | Error-vs-horizon curve per level |
| SSR 2409.18341 | 16 navigation-guided tokens, future BEV feature loss | nuScenes vs UniAD | −27.2 % L2, −51.6 % collision [PUB abstract] | Navigation-guided | A tiny token budget can plan, like our 4 × 10 |
| Epona 2506.24113 | Next-frame head + trajectory head on a shared latent | NAVSIM; FVD on NuPlan | Joint training PDMS 78.1 → 86.2 [PUB Table 5] | Same latent | The planner gains from sharing the latent |
| Drive-OccWorld 2408.14197, Cam4DOcc 2311.17663, PreWorld 2502.07309, UnO 2406.08691, OccLLaMA 2409.03272, Doe-1 2412.09627, Iso-Dream 2205.13817, AD-L-JEPA 2501.04969, DriveDreamer 2309.09777 / -2 2403.06845 | Occupancy + flow; per-horizon IoU; weak-label occupancy; separating controllable from uncontrollable dynamics; pixels | Not read this pass | — | — | [PUB-SEC] |
| Think2Drive 2402.16720, World4Drive 2507.00603, Drive-JEPA 2601.22032, AD-LiST-JEPA 2602.12540, GAIA-1/2, GenAD ×2, ACT-Bench 2412.05337 (held) | Policy in imagination; trajectory scoring; JEPA; pixels; action fidelity | Not read this pass | — | ACT-Bench is the dedicated action-fidelity test | [PUB-SEC] |

Diagnostics (held, abstract level, [PUB-SEC]):
* **ATM 2606.09028:** inverse probes on real vs predicted transitions, a 2 × 2 matrix, no planner.
* **iKCE 2607.05966:** the departure of an imagined rollout from a closed-form kinematic null, with a perturbation protocol.
* **Intervention gap 2608.29998:** the return falls as an operator-error diagnostic grows, while reward-prediction error stays flat.
* **WorldLens 2512.10958 / 2605.10858:** the field scores realism, rarely behaviour.

## 2. Hierarchical models and planners
| Paper | Evidence that the high level is slower or more abstract | Evidence that subgoals mean something | Numbers | Map |
|---|---|---|---|---|
| H-JEPA 2610.06805 (held) | Two-layer MLP probes on frozen latents, variance-normalised MSE (1 = the marginal mean), 3 seeds, per level. Distractor and body state fade with depth while agent position stays; stride 2 per level [PUB] | Decoders (visualisation only) for start, goal and planned latents; an ablation separates temporal decomposition from goal representation | Visual AntMaze 18 % → 73 %, with less planner compute [PUB abstract] | The best template: our levels are already 0.2 / 0.4 / 0.8 s |
| HWM 2604.03208 (held) | Hierarchical MPC on a visual world model | High-level subgoals decoded | Pick-and-place 70 % vs 0 % [PUB] | Decode L3's target for L2 |
| Mind the Gap 2607.12547 (held) | A frozen low level with latent subgoals | True future latents given as subgoals test the low level | Hierarchy does not always help [PUB-SEC] | Copy this control |
| Director 2206.04114 | The manager picks a discrete goal every K = 8 steps | A goal decoder renders goals as images (Fig. 1) [PUB] | — | Decoded goal beside the real frame |
| THICK (ICLR 2024) | The high level predicts only at low-level context changes | Changes tie to events (object interactions, boundaries); subgoals for MPC [PUB] | — | Do L3 changes align with braking or a cut-in? |
| Clockwork VAE 2102.09532 | Factors 2 / 4 / 6 / 8; slower levels hold static content, faster levels fast objects [PUB abstract] | — | A larger factor keeps long horizons accurate | A factor sweep |
| HIQL 2307.11949, Drive-HWM 2609.03572, Orbis 2 2607.15898, Jumpy WMs 2602.19634, HM-RNN 1609.01704, VTA 1910.00775, FeUdal 1703.01161, HIRO 1805.08296, Humanoid HWM 2405.18418 | Not read this pass | — | — | [PUB-SEC] |

## 3. Multi-rate and multi-frame history
| Paper | History | Gain | Our test |
|---|---|---|---|
| DINO-WM 2411.04983 (held) | h = 1, 2, 3, with a causal mask | PushT 0.76 / 0.88 / 0.92 with the mask; 0.76 / 0.36 / 0.08 without (the model cheats on future frames). The authors link longer history to velocity, acceleration and momentum [PUB Table 6] | The closest to our gap; run the mask check |
| DOME 2410.10429 | 1 to 4 context frames at 2 Hz | mIoU 12.59 / 20.01 / 20.70 / 20.07 [PUB Table 5; layout] | The biggest jump is 1 → 2; it plateaus by 3 |
| Epona 2506.24113 | 2 vs 5 frames | FVD10 59.85 → 55.46; FVD25 81.58 → 71.28; FVD40 103.70 → 86.76 [PUB Table 7] | The gain grows with the horizon |
| BEVFormer 2203.17270, BEVDet4D 2203.17054 | Previous BEV features | Abstract level only [PUB-SEC] | No rate study |
| Copycat 2010.14876, causal confusion 1905.11979, ego status 2312.03031 | Observation and action history | History lets a policy copy its last action [PUB-SEC] | Keep history on the WM encoder, not on the planner |

No paper found sweeps history spacing (0.2 vs 0.4 vs 0.8 s) against the decodability of acceleration or yaw rate. Our pre-registered test (MRATE) is new on that axis [HYP].

## Manifest (agent B)
* `research_imagination/B/BANKED_B.txt`: 25 keys + sha256 + staging path.
* Banked PDFs and the updated index: `F:/TanitAD-scratch/kbB/TanitAD Research Lab/Library/{papers,library.json,LIBRARY.md}`. Staging only; nothing is staged in git or pushed.
* `--verify` on the staging copy: 767 entries, 0 orphans, 706 problems. The 706 are entries whose PDFs were not exported to staging (767 − 61 = 706); all 61 staged PDFs verify.
* Deleted by the agent (its own scratch, re-creatable): `F:/TanitAD-scratch/kbB/_txt` (6 MB) and `F:/TanitAD-scratch/kbB/_dl` (3 MB).
* Kept: 36 re-exported held PDFs in the staging `papers/`, so `--verify` stays consistent. They can be deleted once the 25 new entries land.
