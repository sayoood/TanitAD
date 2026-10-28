# REVIEW: training data and training stages for the v8F planner

Research Lab agent (Sonnet), 2026-10-28. Direction: PI 2026-10-28 (data choice, data amount, semantic selection, staged SFT / DAgger / RL).
Scope: reading only. No GPU, no box 2, no git write, no kb_add. Cards are in `HYPOTHESES.md`; PDFs in `pdf/`; bank list in `BANKING_MANIFEST.txt`.

**Evidence classes.** [PUB] banked PDF with page or table (key = arXiv id; all PDFs are in `pdf/`). [MI] MEASURED-INTERNAL: a repo path read with `git -C C:/Users/Admin/tanitad-push -c safe.directory=* show efb6223c:<path>`. [INH] INHERITED (Master Mind brief, or another report not re-derived). [EST] ESTIMATED (my arithmetic, basis stated). [HYP] hypothesis.
**Internal short names.** `REG` = `Project Steering/GOALS_AND_CLAIMS.md`; `MREG` = `Project Steering/MODEL_REGISTRY.md`; `RAW` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw`; `DSR` = `.../2026-10-11-refe-data-and-rl/DATA_SELECTION_REVIEW.md` (NAVSIM/REFe, earlier pass). All at efb6223c.
**Tier.** Every v8F effect below is T1-kin (self-action open loop, logged frames) unless marked T0. None is driving performance (EVAL_DOCTRINE).

---

## 0. Bottom line

1. **The one big lever this stack has found is valid on-policy supervision, not data volume or data selection.** DAgger (p = 0.1, 3,000 steps, ~15-19k own-state samples) moved T1-kin ADE6 by -0.5435 / -0.4046 m against DAG-0, 2.2x / 1.7x the 2F bar [MI REG H-V8F-DAGGER-1]. Everything else tried on the data side (half corpus, launch up-weight, perturbation, iteration 2, code mixing) read null or harmful (section 1). The published record agrees on the direction: closed-loop / on-policy SFT beats logged-state BC at equal budget [PUB 2412.05334 p6; 2609.20756 Table I p4].
2. **Data amount is predicted to be a sub-floor lever for this model.** Halving the corpus moved ADE6 by +0.039 m (n.s.) [MI]; the only two-point slope on our data (540 -> 4,512 clips, a different model) is -0.088 m per doubling [MI, EST arithmetic]; point-estimate planners saturate in closed loop at ~8,000 scenes [PUB 2603.15185 p6] and at 256 h [PUB 2504.04338 p8]. Expected gain from doubling: **-0.04 to -0.25 m ADE6, central -0.09 m (0.75 F)** [EST]. The full PhysicalAI-AV corpus is 306,152 clips / 1,700 h / 133 TB; ours is 1.47% of its clips (section 2).
3. **Semantic selection is worth one zero-training look before any run.** The crawl is concentrated: the top 1.4% of eval windows carry 66% / 98% of the L2 excess [MI RAW/2026-10-26-lcdiag/DIAGNOSIS.md]. Whether the same concentration exists on TRAIN, and whether it has a label-nameable address (traffic-light go, lead departure), decides if selection can target it (cards H-DD-FITGAP, H-DD-SEMLOC).
4. **RL is not the next stage for a 7M, frozen-feature, non-reactive-replay planner.** Every working IL+RL recipe needs a reactive simulator or a privileged map teacher (PhysicalAI-AV has no map); naive learned-world-model RL lowers the score (88.1 -> 85.3) [PUB 2511.20325 p7]; our own RL arms in other families failed or traded [MI REG H-DDV2RL-1/-2, D-REFE-RLS1-R0]. The cheapest RL-flavoured experiment is outcome-weighted DAgger draws (H-ST-REGRET), which needs no new infrastructure.
5. **The noise floor decides what can be learned.** Same-recipe, same-init replicates differ by ADE6 +0.4025 / +0.0116 m and launch speed -0.556 m/s [MI REG H-V8F-DAG-ORC-1, H-V8F-LAUNCH-INTENT-1]. All of that variance is produced inside a 3,000-step fine-tune from a fixed init (REP = DAG-OACC argv with `--seed 10/11`, init M2b_s0/s1 [MI PREREG_V8F_DAG_ORC.md:361]). A lever below ~0.3 m ADE6 needs ~3-11 runs per arm to be seen (section 5). Variance reduction (SOUP, EMA; sibling pass `research_crawl/`) comes before any sub-0.3 m data lever.
6. **Eight cards** (`HYPOTHESES.md`): two zero-training deciders, three stage levers that extend DAgger, one RL-lite, one added-corpus lever (PI decision), one reweighting lever. No card has an expected effect >= 2F; the best has ~1 F.

---

## 1. What the programme already did (do not re-propose without a new reason)

| item | what it was | result | source |
|---|---|---|---|
| DAGGER (iter 1) | M2b fine-tuned 3,000 steps, p = 0.1 of draws replaced by own T1-kin states, human label at matched route point | **ROBUST.** ADE6 -0.5435 [-0.7564, -0.3690] / -0.4046 [-0.5496, -0.2782] m; R-T0 +0.0019 / +0.0483 (guard 0.0604) | [MI] REG H-V8F-DAGGER-1 |
| DAGGER2 (iter 2) | bank from DAG's own states, p = 0.1 | **HARMS.** DAG2 - DAG2-0 -0.2247 / -0.1630 m but DAG2 - DAG +0.0426 (s0, worse) / -0.0309 (n.s.); crawl rows unmoved | [MI] REG H-V8F-DAGGER2-1 |
| DAG-ORC | O-RECOVER ego perturbation, **targets unchanged**, on DAG-OACC | **HARMS.** R-ADE +0.1434 m (s0), R-T0 fails both seeds, crawl rows 8.8-33.8% of bar | [MI] REG H-V8F-DAG-ORC-1 |
| LAUNCH-INTENT | loss weight 1.58926 on GT-launch TRAIN samples (f = 0.1619), O's along-track term only | **HARMS / M-LAUNCH NOT MOVED.** launch speed X-Y +0.0348 [-0.0299, +0.0923] m/s; ADE6 sign flips between same-seed pairs; HI dose (W 4.17852) pending PI | [MI] REG H-V8F-LAUNCH-INTENT-1 |
| DSCALE | halve unique clips at fixed steps (seed 0) | **NOT DATA-LIMITED.** T latcons +6.25% n.s.; ADE6 +0.039 m n.s. | [MI] REG H-V8F-DSCALE-1 |
| DSEFF, DATA-SCALE doubling arms | halving costs FLAT more than M2b; doubling | **not run** | [MI] REG H-V8F-DSEFF-1, H-DATA-SCALE-1 |
| MIXO | per-kind code mixing | **FAIL, O learned** | [MI] REG H-V8F-MIXO-1 |
| SPEEDSUB / -T1 | swap O's five speed cells for an ego-only read-out | T0 PASS (screening); **T1-kin NULL**, only hit is the slow-T draw REP s0 | [MI] REG H-V8F-SPEEDSUB-1, -T1-1 |
| FLATDAG | DAgger on both sides | NOT ESTABLISHED; DAgger helps FLAT too (-0.1522 / -0.1905 m) | [MI] REG H-V8F-FLATDAG-1 |
| RL, other families | DDv2 port; RL-S1 scorer; WP-RL | FAILED / killed by K3 / paused | [MI] REG H-DDV2RL-1/-2, D-REFE-RLS1-R0, OPS-THOR-RL-PAUSE |

Facts the cards lean on:
- **Training recipe.** M2b: 19,164 steps x batch 64 = 1.2265M samples = **4.0 epochs** of 306,566 windows; 7,214,495 trainable params; frozen DINOv3-L `pool4x10` tokens [MI MREG s17; RAW/2026-10-27-dagger-run/M2b_s0_dag/summary.json]. DAgger fine-tune: `--init-ckpt M2b_s0 --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --dagger-p 0.1`, 2,273 s = 38 min = 0.63 epoch [MI .../M2b_s0_dag/argv.json, summary.json].
- **Bank (M2b generator, s0).** 613,132 own states; ticks 1-2 only; of 306,566 tick-2 records 165,093 (53.9%) survive, 67,105 lost to clip end, 36,716 to heading > 30 deg, 37,652 to lateral > 3 m; window coverage 0.7556 [MI RAW/2026-10-27-dagger-run/bank_s0/bank.json]. The eval rollout is 6 s with a replan every 10 rows, i.e. plans 0, 1, 2 [MI RAW/2026-10-26-lcdiag/DIAGNOSIS.md "plan 2 (pairs 20-28, t = 4-6 s)"], so ticks 1-2 are exactly the states the eval visits. **Consequence:** the DAgger gain is measured on the rollout it was trained on; T2 (AlpaSim-style re-perception) is not provisioned [INH EVAL_DOCTRINE], so transfer to real closed loop is untested [HYP].
- **Launch statistics.** 49,618 of 306,566 windows are GT-launch (f = 0.16185); **1,891 of 4,512 clips contain one**; max 68 per clip; median v0 2.85 m/s, v(6 s) 6.85 m/s [MI RAW/2026-10-28-launch-intent-flag/f_measure.json]. Independent launch events are therefore on the order of 2k [EST: ~1 event per launch-bearing clip].
- **Where the crawl is.** 88-96% of the L2 curvature excess sits on step pairs where GT >= 5.2 m/s and the model's own car is below; 332 / 220 windows from 61 / 47 episodes; v0 median 5.7 / 4.7 m/s; "those states exist only in the model's own rollouts" [MI RAW/2026-10-26-lcdiag/DIAGNOSIS.md; PREREG_V8F_DAGGER.md section "Why"].
- **Launch-exposure variance from data order is negligible.** Four sampler simulations of the 3,000-step stage give launch share 0.1572-0.1607 (f = 0.1619) [MI f_measure.json `sampler_simulation_3000_steps_batch64`]. Run-to-run launch-speed swings of -0.556 m/s therefore do not come from how many launch windows a run happens to see. A stratified sampler built to cut variance is falsified before it is built, and is not carded.

---

## 2. D. Data amount and scaling

### 2.1 Published measurements

| study | regime | measured | source |
|---|---|---|---|
| Naumann (NVIDIA) | internal, 16-8,192 h, 1 camera, ResNet-18 | open-loop FDE power law to 8,192 h; **1% FDE needs ~4,000 h more, 3% ~29,000 h, 5% ~273,000 h**; ResNet-50 reaches the same FDE with ~3,000 h (-63%); closed-loop **MDBF improves only to 256 h, then plateaus at ~1 km** | [PUB 2504.04338 p7, p8] |
| Zheng (Li Auto) | ~4M demos, >30,000 h, 23 scenario types | ADE Y = 0.6833 X^-0.188, r = -0.963 (open loop); closed loop is **not** a power law; safety / comfort growth slows between 200k and 700k samples; **at constant total volume, doubling a long-tail scenario type: +9.7..16.9%; quadrupling: +22.8..32.9%** (open-loop error) | [PUB 2412.02689 p5, p6] |
| Baniodeh (Waymo) | ~500k h | open- and closed-loop improve as power laws of compute; optimal model size grows 1.5x as fast as data; **10 observed miles of other agents = 2-3 demonstrated miles** | [PUB 2506.08228 p1, p19] |
| BevAD | Bench2Drive, closed loop, 1k-16k scenes | point-estimator planner's growth **decelerates after ~8,000 scenes**; diffusion planner stays linear to 16,000 | [PUB 2603.15185 p6, Fig 4] |
| Codevilla | CARLA, >80 h of driving | diminishing returns after a certain amount of demonstrations, even degradation on unseen environments | [PUB 1904.08980 p2, p3] |
| Bronstein (Waymo) | fleet run segments, imitation planner, 10 seeds | a random 10% subset equals the full set; difficulty-prioritised: **-15% collisions, +14% route adherence at 10% of the data** | [PUB 2212.01375 p1, p6] |
| IINE (Waymo) | >100k miles; Top1 / Top10 / Top50 = hardest 1% / 10% / 50% by a difficulty model (Top10 = 400k train examples) | BC trained on Top10 fails 5.79% on the Top1 eval set vs 9.74% trained on all data | [PUB 2212.11419 p4, p6 Table II] |
| Lin (manipulation) | 40k demos | environment / object diversity matters far more than demonstrations per environment; extra demos saturate | [PUB 2410.18647 p1] |

Reading: open-loop error keeps falling with volume; closed-loop plateaus early; **distribution beats volume once past a threshold**, and a point-estimate regressor saturates first.

### 2.2 What this predicts for 4,512 clips [EST]

- Our corpus: 4,512 clips x 20 s = **25.1 h**; 306,566 heavily overlapping windows. It sits left of every published regime except Codevilla's.
- Three slopes per doubling of clips, on ADE6 (our scale ~1.3-2.2 m [MI MREG s17, REG DAG-REREAD]): DSCALE-implied ~-0.04 m (halving +0.039, n.s.) [MI]; RA-1-implied **-0.088 m** (A1 at 1000 steps, 540 -> 4,512 clips: -0.2690 / -0.2705 m over log2(8.36) = 3.06 doublings) [MI REG H-RA1-FULL-1; arithmetic mine]; Zheng-exponent-implied ~-12% of ADE, ~-0.2 to -0.26 m [PUB 2412.02689 p5; 2^-0.188 = 0.878; not transferable across regimes].
- **Central -0.09 m = 0.75 F (F = 0.1207 m, MREG s17), range -0.04 to -0.25 m.** One doubling cannot be resolved with two seeds (section 5).
- Where returns flatten: closed loop at 256 h / ~8,000 scenes for regressors [PUB 2504.04338 p8; 2603.15185 p6]. v8F-mini is a point-estimate regressor with a residual head, so the flat region is the default expectation [EST]; DSCALE agrees [MI].
- **Mechanism caveat the programme has already measured:** "at full data T is fit-limited, eval ~ train" (the R70 correction inside the DSCALE row) [MI REG H-V8F-DSCALE-1]. If the crawl is also fit-limited, more data cannot help; card H-DD-FITGAP tests exactly that.

### 2.3 The full corpus, and what mining needs

- PhysicalAI-AV: **306,152 clips x 20 s, 1,700 h, 25 countries, 133 TB**, 7 cameras, 30 fps; splits: default train 153,625 / val 90,928 / test 61,599; alpamayo_aligned 276,299 / 28,317 / alpasim 1,536 [PUB-secondary: HF dataset card, fetched 2026-10-28; same totals MEASURED 2026-10-11 in `DSR` section 3.1 [MI]]. The card gives no scenario breakdown (stopped, turning).
- **Label shard sizes, MEASURED 2026-10-28 through the HF metadata API (no data downloaded):** 3,146 chunks each; `labels/egomotion` **128.6 GB**, `labels/egomotion.offline` **4.94 GB**, `labels/obstacle.offline` **158.02 GB**. Launch mining needs ego speed only, so `egomotion.offline` (content UNVERIFIED; ~1.6 MB per chunk) or a subset of `egomotion` (~41 MB per chunk) is enough for a zero-GPU query over all 306k clips. Pixels are only needed for the selected clips.
- Per camera-clip video: 133 TB / (306,152 x 7) = **~62 MB** [EST]. 2,000 front-wide clips ~125 GB.
- Alpamayo-R1's "80,000 hours" is its internal training corpus, not the public set [PUB 2511.00088 p23].
- Ours is **4,512 / 306,152 = 1.47%** of the clips. Disjointness from eval-141 and the 60-clip holdout must be enforced by clip id; session / drive disjointness is UNVERIFIED (needs `clip_index.parquet`, 11 MB, not fetched).

### 2.4 Register note for the Master Mind
`REG D-DATA-EFFICIENCY` states that no published work ablates diversity at fixed hours. Two partial exceptions: Zheng p6 (constant total volume, targeted scenario doubling) in driving, Lin p1 (diversity vs demos per environment) in manipulation. The sentence needs scoping, not deletion.

---

## 3. E. Semantic data selection and curation

### 3.1 Methods and measured effects

| method class | paper | measured effect | regime | read-across |
|---|---|---|---|---|
| semantic dedup in embedding space | SemDeDup [PUB 2303.09540 p1, p4, p10] | embed -> k-means (k = 50,000 for LAION) -> drop within-cluster cosine near-duplicates; **50% of LAION-440M removed, minimal loss, half the training time**; robust to k (Table 1) | web-scale CLIP | our 306k windows are overlapping slices of 4,512 clips: near-duplicate mass is certainly high [EST]; but dropping windows re-selects the corpus (parity) |
| dedup + diversify, then repeat | D4 [PUB 2308.12284 p1] | 20% efficiency gain, up to +2% on 16 tasks (6.7B); **repeating data intelligently beats baseline, repeating random data is worse** | LLM pre-training | we train 4 epochs, i.e. we repeat; which windows repeat matters |
| prune by difficulty, regime-dependent | Sorscher [PUB 2206.14486 p2 Fig 1B] | **keep hard examples when total data is abundant, keep easy when scarce** | CIFAR / ImageNet | our canonical set is scarce (do not prune hard windows); a mined pool of 301k is abundant (hard-first applies there) |
| trajectory-cluster balancing | FlowDrive [PUB 2509.21961 p9 Table 2a] | Val14(R) none **81.91**, scenario-type **80.08**, trajectory-cluster **85.37** | nuPlan closed loop, single run | the largest published sampler effect; label the clusters by what the plan looks like, not by scenario type |
| VQ-VAE trajectory clusters + priority sampling | CAPS [PUB 2503.01650 p5] | Bench2Drive driving score 59.5 -> 66.7, success 48.2 -> 52.9%, completion time 71.4 -> 50.0 s | CARLA, 220 segments, seeds not checked | completion time falls ~30%: the progress channel, which is our crawl |
| hard-first by the model's own metric | MOSAIC [PUB 2604.08366 p6; numbers INH via `DSR` section 1] | up to 1.1 EPDMS above the next best at every budget on Navtrain; logit-entropy "uncertainty" is **negative** | NAVSIM added-data, 460-clip base | added-data from a pool, not re-weighting |
| uncertainty is not criticality | DARB [PUB Prakash2020_DARB_cvlibs p8] | high prediction uncertainty does not correlate with critical scenarios; critical-state sampling + replay buffer: **+16% success**, collisions down an order of magnitude | CARLA NoCrash | agrees with MOSAIC's entropy negative: do not use uncertainty as hardness |
| closed-loop-selected re-weighting | CW-ERM [PUB 2210.02174 p4] | upsampling the scenes that fail closed loop: **up to ~35% fewer collisions** on some metrics; improving side collisions also cut rear collisions; adding rear collisions regressed | nuPlan-like, log replay | the closest published form of our REGRET card |
| difficulty-prioritised subset | Bronstein [PUB 2212.01375 p1]; IINE Table II [PUB 2212.11419 p6] | see section 2.1 | fleet data | strongest controlled "curation beats volume" evidence |
| retrieval of relevant sub-trajectories | STRAP [PUB 2412.15182 p2, p7, p9]; Behavior Retrieval [PUB 2304.08742 p1] | DINOv2 features + dynamic time warping; "beats goal-conditioned methods by 20%" | manipulation | a mining recipe for the added-pool route (query = TRAIN-side crawl windows) |
| data mixtures by DRO | Re-Mix [PUB 2408.14037 p1] | **+38% over uniform**, +32% over human weights | Open X-Embodiment | domain-level weights; our strata could be domains |
| influence / attribution | CUPID [PUB 2506.19121 p1, p3, p7]; TRAK [PUB 2303.14186 p1]; DemInf [PUB 2502.08623 p1]; Demo-SCORE [PUB 2503.03707 p1]; DataMIL [PUB 2505.09603 p1] | CUPID: **<33% curated data matches SOTA** on Transport MH (uses TRAK projections); DemInf +5-10% on RoboMimic; Demo-SCORE +15-35% absolute; DataMIL: naive similarity selection can harm, a surrogate loss on target data avoids rollouts | manipulation, small data | feasible at 7.2M params; no driving evidence [HYP] |
| scenario mining by language | RefAV [PUB 2505.20981 p1] | off-the-shelf VLMs perform poorly; built on HD-map-localised logs | Argoverse 2 | not usable on PhysicalAI-AV (no map) |
| clip selection by explicit decision | Alpamayo-R1 [PUB 2511.00088 p10 Fig 3, p22] | label clips with an explicit driving decision, e.g. "resume at traffic light"; RL data = high model / reward disagreement **plus an equal share of random samples** | PhysicalAI-AV family | confirms the launch event is a first-class curation target; no controlled ablation of the selection is reported |

### 3.2 Which of these target the launch / low-speed / turn crawl

- **Kinematic strata (label side)** are already used (LAUNCH-INTENT) and read null at the loss-weight level. Not re-proposed as-is.
- **Embedding-space clustering over TRAIN** (SemDeDup / FlowDrive / CAPS recipe) is the only family that can find a crawl address the labels do not name; it needs the zero-training concentration read first (H-DD-SEMLOC).
- **Retrieval from the 301k pool** (STRAP / Behavior Retrieval / Alpamayo-R1 clip selection) is the data-amount route that adds independent launch events; a two-stage mine (ego-speed query on `egomotion.offline`, embedding diversity only inside the candidate set) keeps decode and feature cost proportional to the selected clips (H-DD-LAUNCHMIX).
- **Influence** (CUPID / TRAK) is the only family that scores a window by its effect on a closed-loop outcome; cheap at 7.2M params (H-DD-INFLUENCE).
- **We hold the inputs.** Vision features: DINOv3-L `pool4x10` cache. Label-side semantics: v7.x VLM-derived tactical goals (779 of 4,572 records carry a traffic-light goal with colour: RED 376, GREEN 363, YELLOW 22 [MI CLAUDE.md layer table; `RAW/.../s2_labels_v7.2_train.jsonl.gz` per CLAUDE.md]); lead-vehicle tracks `lead_train4572.npz` and `obstacle.offline` (97.44% of the corpus) [MI argv.json; CLAUDE.md]. These are **labels and strata only; none may enter the goal path at inference** (BINDING goal rule).
- **Selection must be made on TRAIN only.** Using eval-141 crawl windows to choose which strata to up-weight is the programme's known "probe tunes on the data it scores" trap; eval-side localisation is a pre-registered confirmation, never the selector.

---

## 4. F. Staged and combined training

### 4.1 Published staging results (small models first)

| stage design | result | source |
|---|---|---|
| BC pre-train -> closed-loop SFT (CAT-K: roll out the policy, pick among the top-K the action nearest GT so visited states stay near GT and the GT label stays valid) | **7M model RMM 0.7702 > 102M model 0.7614**; its own BC base 0.7591 | [PUB 2412.05334 p1, p6] |
| BC base -> on-policy supervision from an RL-trained privileged teacher (OPTED) | LTFv6 (62-356M students) scene score 26.5 -> **41.8**; VaVAM 3.9 -> 37.1; **logged-state BC on the same scenes 12.8 / 25.2**; ~3 orders of magnitude fewer simulator interactions than direct RL. Teacher needs **HD-map + bounding boxes** | [PUB 2609.20756 p1, p4 Table I, p6] |
| IL pre-train -> IL+RL (RAD, 3DGS closed loop) | collision rate **0.229 (IL) / 0.143 (RL) / 0.089 (IL+RL)**; ADD 0.238 / 0.345 / 0.257 | [PUB 2502.13144 p8] |
| BC + SAC (Waymo) | hardest-set failure 9.74% (BC) -> **3.72%**; pure SAC **route progress 77.8%** vs BC 99.0%; large progress rewards degrade | [PUB 2212.11419 p6 Table II] |
| DAgger with critical states + replay buffer (DARB) | +16% success; the on-policy proportion must be controlled to keep generalisation | [PUB Prakash2020_DARB_cvlibs abstract, p5] |
| world-model RL | naive optimistic WM: **88.1 -> 85.3 PDMS**; counterfactual synthesis 89.1; progress reward prevents the "frozen robot" | [PUB 2511.20325 p7, p18] |
| model-based RL in a latent WM | expert-level in CARLA v2 in 3 days on one A6000, **simulator reward**; Raw2Drive needs a privileged WM first | [PUB 2402.16720 p1; 2505.16394 p1] |
| privileged PPO | 300M samples CARLA / 500M nuPlan, one route-completion reward, 8 GPUs | [PUB 2504.17838 p1] |
| DAgger theory | supervised BC cost grows quadratically in horizon T; aggregation of on-policy data restores linear growth | [PUB 1011.0686 p3] |

**Ordering the evidence supports for a sub-300M model:** BC -> on-policy supervised stage with labels that stay valid -> (optional) outcome emphasis -> RL only under an IL anchor and a trusted simulator. Pure RL loses human-likeness (RAD ADD 0.345) or progress (77.8% route progress); progress is precisely our crawl, so an RL stage that fixes progress must be anchored, not free.

### 4.2 Where our stack sits on that ladder

- Stage 1 BC (M2b) and stage 2 on-policy SFT (DAG) exist. **Iteration 2 saturated** (DAG2 - DAG ~0). Dose and label validity are the two untested knobs: p was never raised above 0.1, and 24.3% of tick-2 records (lateral > 3 m or heading > 30 deg) are discarded because the matched human label is invalid there [MI bank.json]. CAT-K and OPTED are both about keeping the label valid; our ORC arm changed states and kept the targets, and harmed [MI REG H-V8F-DAG-ORC-1].
- **Stage 3 RL is blocked, with named reasons:** (a) no map in PhysicalAI-AV, so an OPTED-style privileged teacher cannot be built (`RefAV`-class map tooling also unavailable); (b) the eval is non-reactive replay of logged frames, so a progress reward has no interacting traffic and no re-rendered pixels; AlpaSim / NuRec is the venue OPTED used and T2 is not provisioned [INH]; (c) M2b's own WM aux did not help (H-MINI-WM-1 FAIL, M2b - M1 -0.0713 / +0.0302 n.s. [MI MREG s17]), so there is no validated learned simulator, and naive WM-RL is documented to lower scores. Unblocking: a provisioned closed-loop sim (PI / MM) or a validated WM.
- **Anti-shortcut stages** (past-motion dropout on 50% of samples, ChauffeurNet Sec 4.2 [PUB 1812.03079 p8]; ego-channel cut) are the sibling pass's scope (`research_crawl/`); the programme's next committed anti-shortcut arm is T-EGOCUT inside PREREG_V8F_DAGGER's NO EFFECT row [MI]. Here only the staging point: such a stage belongs late, after on-policy SFT, so it acts on states the model really visits.

### 4.3 Stage variance, mechanism (why every stage lever is hard to read)
- Same init, different seed, 3,000 steps: ADE6 +0.4025 / +0.0116 m; speed cells of one draw ~1 m/s slow [MI REG H-V8F-DAG-ORC-1; dagland/rep_s0_diag/REP_S0_SPEED_BIAS.md, INH]. In BERT fine-tuning both weight init and data order contribute comparably to variance [PUB 2002.06305 p1], and instability is mostly optimisation difficulty plus generalisation differences at equal train loss [PUB 2006.04884 p1]. Whether ours is optimisation (fixable by schedule / averaging) or data-order cannot be told from n = 2; the launch-share simulation (section 1) already excludes composition of launch exposure.

---

## 5. Noise floor and resolvability

Floors [MI]: F_M2b = 0.1207 m (the |s0 - s1| ADE6 spread of the base, MREG s17); REP - CTRL ADE6 +0.4025 / +0.0116 m; launch speed REP - DAG-OACC s0 -0.5560 [-0.6994, -0.3975] m/s against a lever effect of +0.03; crawl bars (2 x max(F, RF0)): yaw-rate 1.1070, curvature 0.0526, L2 0.0136, M1 0.0096.
Per-run SD on ADE6: sigma ~ 0.1 (from F) to ~0.2 (from the REP pairs) m [EST, n = 2 pairs].
Runs per arm for 80% power at 5%: n = 15.7 sigma^2 / delta^2 [EST]:

| true effect delta (m ADE6) | n per arm, sigma 0.1 | n per arm, sigma 0.2 |
|---|---|---|
| 0.5 (DAgger-sized) | 1.6 | 6.3 |
| 0.2415 (2F) | 2.7 | 10.8 |
| 0.1 | 15.7 | 62.8 |
| 0.05 | 62.8 | 251 |

At ~40 min per box-2 run, a 2F effect costs 2-7 h of box 2 per arm; a 0.1 m effect is out of reach until variance falls. This is why sub-floor data levers (cards 5-6) are ranked last and why the two zero-training deciders come first. The paired same-seed design and the `A_rep` replicate pairs the programme uses reduce but do not remove this [MI REG H-V8F-LAUNCH-INTENT-1: separated ADE6 cells flipped sign between pairs].

---

## 6. Parity ledger (kind per idea)

Canonical = the v8F TRAIN list (4,512 clips, 306,566 windows, window-list sha `e144adcd...`, train sha `37f4f9ee...`) and eval-141 / holdout `540:600` (sha `f6a1dee0...`) [MI summary.json]. Registry notes this corpus is not the older parity corpus `physicalai-train-e438721ae894` [MI MREG s17].

| card | kind | parity |
|---|---|---|
| H-DD-FITGAP | read-only | none |
| H-DD-SEMLOC | read-only, clusters fitted on TRAIN | none |
| H-ST-DAGDOSE | resampling fraction within bank | within-corpus |
| H-ST-RECOVERYLABEL | derived labelled states from canonical TRAIN windows | within-corpus (derived, eval excluded by K-B3) |
| H-ST-REGRET | reweighting within bank | within-corpus |
| H-DD-INFLUENCE | analysis, then down-weight | within-corpus; **never drop windows** |
| H-DD-LAUNCHMIX | ADDED corpus, canonical intact, eval untouched | **PI decision**: download size, new corpus kind, matched-steps canonical-only control mandatory |
| H-DD-SEMBAL | reweighting within fixed corpus | within-corpus; the drop variant (SemDeDup as published) **would re-select windows and breaks the window-list sha: PI decision** |

---

## 7. Limits and flags

- INHERITED, not re-derived here: MOSAIC per-budget gains, DriveVLA-W0 Table 7 (nuPlan pre-train swings -6.5..+4.9 PDMS), DriveZero, Gigapixel, DDv2, SimScale (all via `DSR` / `RL_POSTTRAIN_REVIEW`; keys in the manifest tail). OPTED Table I columns were reconstructed from garbled text and cross-checked against the prose on p6.
- Abstract-level reads only: 2406.01544, 2409.16663, 2402.16720, 2505.16394, 2504.17838, 2303.14186, 2502.08623, 2503.03707, 2505.09603, 2408.14037, 2304.08742, 2505.20981.
- HF card numbers come through WebFetch (PUBLISHED-SECONDARY); the label-shard sizes are my own API totals.
- Sigma for section 5 rests on two replicate pairs; treat as an order of magnitude.
- I did not read any box-2 or dev-box artifact; the bank statistics are the repo's `bank.json` header only.
- Siblings: inertia / copycat / variance reduction levers are in `dagland/research_crawl/`; this pass does not duplicate them. A merged ranking is the Master Mind's.
