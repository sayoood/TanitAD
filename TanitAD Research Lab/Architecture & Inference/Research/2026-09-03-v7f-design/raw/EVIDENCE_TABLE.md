<title>raw/EVIDENCE_TABLE — every number v7f's design rests on, with its source and class</title>

# EVIDENCE TABLE — v7f design + pre-registration

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03 · 0 GPU`
`Classes: MEASURED (ours + artifact path) · PUBLISHED-PRIMARY (banked PDF, lib:<key> + table)`
`· INHERITED (another agent/doc, NOT re-verified here) · UNVERIFIED (flagged, never a premise)`
`Paths are repo-relative unless marked. Register rows are cited by id; GOALS_AND_CLAIMS.md line`
`numbers are given where a row was opened directly.`

⛔ **This package produced NO new measurements.** It is a design + pre-registration deliverable; every
number below was read from an existing artifact in this session.

---

## A. OURS — MEASURED

| # | number | what it is | source | used for |
|---|---|---|---|---|
| A1 | **0.005947** | `postrain30k` banked legacy `ratio_action_over_scene` at h=1 | MM-E10 (register); reproduced dev-box, k8 package | the G-ACT bar's base (×10 = 0.0595) |
| A2 | **0.0087 / 0.0138 / 0.0032 / 0.0181** | anchored `rel_to_scene(2σ)` for `postrain30k` / `k8clip05p30k` / `k60clip05p30k` / `rdw8p30k` | `…/Research/2026-09-03-anchored-actdiv-and-transition-probe/RESULT.md` §2.2 | G-ACT calibration: none is material |
| A3 | **F_sep 16.39 / 21.87 / 7.19 / 136.1**, null median 0.86–0.99, **p95 1.01–1.28** | same four arms; 200-permutation label null | idem §2.2 | G-ACT's shuffled-action control expected reading |
| A4 | verdicts **SEPARATED-NONMONOTONE ×3, STRUCTURED-WEAK ×1** | committed verdict table | `…/2026-09-03-anchored-actdiv-and-transition-probe/SPEC.md` §4 | the gate's classes; bar **0.0595 at 2σ** |
| A5 | **0.027338 / 0.014359 / 0.007742 / 0.026363** | the SAME legacy ratio under the **trainer's** lift (`v_last/10`) vs the banked `v_first/30` | idem §2.1 | the lift is load-bearing: 3–5× |
| A6 | `SPEED_SCALE = 30.0` (probes) vs **10.0** (trainer) | `actdiv_thor.py:47`, `actdiv_local.py:56`, `condpath_thor.py:55` vs `train_v6_staged.py:3634-3635`, `v6.py:5340` | `H-LEAK-1` | G-ACT uses the trainer lift |
| A7 | **0.00460 vs 0.00298 (+54 %)** on `k60clip05p30k`; **−5.7 % / −9 %** on k=8-class arms | the lift's effect per arm | `…/2026-09-03-p2-probe-leak-audit/raw/actdiv.json` | why the k=60 attribution reversed |
| A8 | steer/accel spread **1.41× [1.03, 1.78]** (banked 0.83× [0.61, 1.09]); scene spread **1.72×** | k=60 vs the one-variable k8 control at the trained lift | `…/p2-probe-leak-audit/raw/actdiv_pairs.json`; `H-LEAK-1` | k=60 verdict stands; SMAS-1's "response fell 17 %" **withdrawn** |
| A9 | full tuple `[steer, accel, v]` moves ẑ **2.79 % / 2.68 %** vs **0.56 % / 0.51 %** for `[steer, accel]`; factors **4.97 [3.64, 7.13] / 5.25 [4.04, 6.95]** | conditioning response composition | `H-LEAK-5`; `…/p2-probe-leak-audit/raw/actdiv.json` | the response is ~5× speed-channel; P2's sign stands |
| A10 | `postrain30k_freeze` steer/accel ratio **0.00078** (7.2× below the incumbent's **0.00561**); paired factor **0.45 [0.28, 0.72]**; scene spread **0.750 (×3.25)** | the frozen arm's command channel | `D-P2-LEAK-AUDIT` §0 finding 8; `H-LEAK-5` | ⭐ freezing made the action pathway **deader** |
| A11 | endpoint-shuffled control reads r **0.677 / 0.673** against "drift" **0.670 / 0.674** ⇒ **102 % / 100 %** | the drift metric is arithmetic | `H-LEAK-3`; `…/p2-probe-leak-audit/raw/drift.json` (80 clips, 7,680 rows, K=4, band 0:8) | drift leaves the gate; G.1 not adopted |
| A12 | L3 t's are **pseudo-replicated** (~92 % overlap, ÷√24); under true LOCO **no column separates** (\|t\| ≤ 1.5) | `envpred.py:211-217`, `v7tiny_probe.py:180-186`, `:238-241` | `H-LEAK-2`; `raw/l3.json`, `raw/l3_pooled.json` | ⛔ **L3 never admissibly passed or failed** ⇒ v7f does not gate on it as instrumented |
| A13 | GT-future-action leak present and **inert**: `zhat(GT) − zhat(HOLD)` = **0.0000** at K=1, \|t\| ≤ 2.19 at K=3/6 | `envpred.py:184-190`, `actionshuf.py:141-147` | `H-LEAK-4`; `raw/l3.json` | the L3 nulls survive |
| A14 | O11's positive action **is** the target's realised motion (r **0.966–0.999**); degenerate solution named `train_v6_staged.py:1370-1375`; `shuffle_all` d_out **0.51** vs latent control 0.43, `zero_v` **1.64** | O11 breakout re-classified | `H-LEAK-6`; `…/2026-08-24-…/raw/actchan_o11_degenerate.json` | O11 stays at 0 |
| A15 | **r 0.9988** (20 clips) / **0.9664** (129 clips) | `v·tan(steer)/L` vs measured yaw-rate | E-DEC-57; `…/p2-probe-leak-audit/raw/data.json` (R5) | the action is realised motion ⇒ the VO confound on LDAD |
| A16 | E-DEC-49 echo r **0.326** (20 clips) / **0.63 (all frames) / 0.79 (first 100)** on 129 clips | `accel` vs Δv₁ | `D-P2-LEAK-AUDIT` §0; `echocheck.py:58` | scope discipline; quoted with its `n` |
| A17 | `postrain30k_freeze` held-out nrmse **0.9301 vs 0.8115 (+14.6 %)**; drift 0.3905 | the one-variable freeze cell | E-DEC-64; `MODEL_REGISTRY.md` registry:3818-3820, 3826 | DEGENERATE ⇒ v7 trunk stays trainable. ⚠️ the **drift** half is uninformative post-A11 |
| A18 | `splitp30k` `n_agents` **+0.3881** > frozen DINOv3 **+0.2754** > constant 0.0000; `lead_range_m` **−0.1611**, `d_ego` **−0.0399** | L2 content, target-specific | E-DEC-29 / registry §13.0d, registry:3958 | "static kept, dynamic erased" — the Observer-Effect shape in our own data |
| A19 | participation **3.80/3.62 → 25.58/26.96** at parity (`rdw8p30k`) | collapse solved | H-RANK-11; registry:3907 | G-RANK context |
| A20 | ⛔ frozen DINOv3 ViT-L/16 through `spectrum_report`: **5.756** (12 val clips, n1440, d1024), **20.228 ± 0.327** (130-clip, n1440, d1024), **20.516** (n5617) | the 8.56 and 40.77 floors are **not reproduced** | `MODEL_REGISTRY.md` §13.3; `stack/tanitad/models/v6.py:1517-1552` (`O6_PARTICIPATION_REFERENCES`); pinned by `stack/tests/test_participation_floor_provenance.py` | ⛔ **G-RANK re-specified** (registry beats the skill) |
| A21 | participation is comparable only at matched **corpus / episode count / ambient d**; the 3.51× spread is **episode diversity alone** (H-RANK-23) | idem | idem | the matched-reference requirement |
| A22 | v1.6/v1.7 S-curve **0.9785 open / 0.0430 closed**, hold-action **0.0 %**; ADE 0.2849 open / 0.4616 closed; **CV floor 0.5352** | the echo failure | `MODEL_REGISTRY.md` §1.12, registry:1281 | the L4 deliberate-regression arm |
| A23 | v7 first parity T1: ADE **14.52 m [11.55, 17.58]** (trained) / 14.21 (frozen) vs CV floor 0.5352; `emao14_30k` cl/ha **14.069/13.879**; all three **LOSES_TO_HOLDV0**; `copy_detector` CLEAN, echo **0.0000** | the untrained-planner floor at T1 | registry:3784-3802, 3868 | G-DRIVE's starting point; ⚠️ **scope: ~19 M v7-tiny, S-W only, every planner objective at zero** |
| A24 | refav1 step-1000 T1: cl ADE **0.547 [0.405, 0.714]**, ha 0.610, cl_navshuf 0.642, cl_oraclegoal 0.718; `baseline_won_frac` **0.7571**, cem 0.2429 | the first real T1 read | `D-REFAV1-STEP1000-READ`; `…/2026-09-03-refav1-step1000-read/raw/refav1_t1_step1000.json` (388,352 B) | G-DRIVE's `baseline_won_frac` reference |
| A25 | two checkpoints differing by a full recipe: **identical to 1e-9 on 140/140 windows** (max \|Δ\| exactly 0.000000 m) on cl/ha/ol/cl_navshuf | the VOID paired read | `D-REFAV1-PAIRED-READ-VOID`; `…/refav1-step1000-read/raw/refav1_t1_ep2_step1000.json` + `tools/paired_dump_compare.py` | the bit-identity refusal |
| A26 | cl trajectories: **y ≡ 0 and constant speed on 140/140**; hold-action drifts **0.12 m** on straight windows / **0.76 m** on curved; the straight line reads **0.035 / 0.496** | the trivial-profile finding | AMENDMENT TO `D-REFAV1-STEP1000-READ`; `tools/straight_line_probe.py` | the `ha0` control; trivial-profile refusal |
| A27 | open-loop replay of the RECORDED `(a, κ)` misses the human laterally by **0.716 m** on 72 curved windows | `C-REFAV1-KIN-CONTRACT-LAT` | idem | ⛔ no lateral row quotable until repaired |
| A28 | `H-REFAV1-LAT-INSENSITIVE` **REFUTED**: κ-axis F_sep **5,962.49** vs null p95 2.595 (fp32) and **29.91** vs 2.545 (clean epoch); norms scale **1 : 2.500 : 5.005**; cos(m(+L), m(−L)) −0.9956…−0.9999; shuffled-action collapses 27.90 → ≤ 1.83 | the anchored instrument discriminating two bit-identical-driving checkpoints | `…/Research/2026-09-03-anchored-actdiv-refav1/RESULT.md` + `PROPOSED_REGISTER_ROWS.md` | ⭐ the instrument works; the flat plan is a **cost** property |
| A29 | k = 60 diverged at gnorm **2.1e9**, killed at 9,000; the clip-0.5 successor spiked **1.24e10** at 18k | horizon instability | `PREREG_MM_E19_K60_HORIZON.md:298-309`; readiness report §B P4 | R1's regression arm |
| A30 | `--w-o1-* 0` ⇒ the O1 block, and its **six** `rollout_transitions` calls, are never reached | `train_v6_staged.py:3691` (`if w.o1_ctrl or w.o1_fact or w.o1_scene:`); `train_stage_a.py:274-320` | source read this session | ⭐ **BACKLOG R6 does not bind on v7f** |
| A31 | `--freeze-encoder` is all-or-nothing (`train_v6_staged.py:5964-5970`); the optimizer is a single flat AdamW over `trainable` with one `lr` (`:6120`) | no partial-depth unfreeze, no per-group LR | source read this session | D1/D7 need new flags |
| A32 | `--init-from` loads a **whole-stack** dict and REFUSES on `fatal` missing keys | `train_v6_staged.py:7216-7247` | source read this session | the seed-checkpoint route (D1-A) |
| A33 | DINOv3 appears only as O7's frozen teacher (`train_v6_staged.py:931`, `O7_DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"`), REF-A's feature bank (`stack/scripts/dino_precompute.py`), and the fp8 shipper (`stack/scripts/dinov3_fp8_encode_ship.py`) | ⛔ **no DINOv3 weight loader into `ViTEncoder`/`ViT5Encoder`** | **two probes**: `grep -rln dinov3 stack/ tools/` and PowerShell `Select-String -List` over `stack`,`tools`,`taniteval` | E1 — the run's premise is unwired |
| A34 | `Block5` = RMSNorm + QK-Norm + LayerScale + GELU, **no biases on qkv/proj**; joint APE + 2D RoPE + 4 registers | our ViT-5 differs structurally from DINOv3 | `stack/tanitad/models/encoder.py:268-320` | why D1-B (a weight "port") is lossy |
| A35 | `resize_pos_embed` / `adapt_pos_embed_` exist for geometry transfer, with a **declared one-directional bias** (*"any adopted geometry should be RETRAINED, not resampled"*) | the only pretrained-transfer machinery present | `stack/tanitad/models/encoder.py:38-96` | D1-A's pos-embed step |
| A36 | O5/O11 rolls carry `bptt_truncate`; the flag detaches the whole carried WINDOW (refav1's literal `z_.detach()` is a **NO-OP at W = 3**) | `D-V7-WIRING`; `metric_dynamics.py:250-300`; `train_v6_staged.py:3795, 3837, 5096, 6484` | register + source | the horizon design |
| A37 | B1 parity key `physicalai-b1-w120-256x640cyl`, **4,713** clips, `episode_uid_sha256 e8bfb98e06eb…`; the split is **4,572 train + 147 eval**, of which **141 have pixels** | corpus | `D-V7-WIRING`; `D-B1-IS-4713-NOT-4719` | held-constant corpus + the eval split |
| A38 | ⛔ the 141 eval clips with pixels sit **inside** the 4,713-clip train cache; `build_train_episodes`/`build_v2_providers` carry **no exclusion list** | BACKLOG R8, a launch blocker | `D-V7-WIRING` (hazard found at two probes) | E2 |
| A39 | v7.2 label md5s **`0ff90213…` train / `aa12c948…` eval** (canonical, `intrain_eval.V72`); `D-V72-SPLIT`'s md5s are the **pre-schema-fix** blobs | labels | `D-V7-WIRING`; readiness report §F | held-constant labels; a stale register row to fix |
| A40 | the tactical labels **land in the batch but no loss reads them**; strategic ARGS unsupervised | BACKLOG R7 / R9 | `D-V7-WIRING` | §8 of DESIGN — out of v7f's scope |
| A41 | step-stamped checkpoints ARE implemented | `train_v6_staged.py:6354-6387` | readiness report §B note 3 | checkpoint selection has something to select from |
| A42 | the window census depends on `max_horizon`, derived from the stage's live loss horizon; an episode shorter than `window + max_horizon` yields **ZERO** windows | the L-10 window-count confound | `train_v6_staged.py:3188-3214, 3389-3391`; `tanitad/data/_contract.py:120` | held-constant `o5_k` |
| A43 | predictor collapses horizons ≥ 2: `max|h1−h2| = 1.211` vs `max|h2−h4| = 0.0018`; only head 1 is trained | MM-E14 | `V7_RECIPE_AND_SCALEUP.md` §6 | the K ≥ 2 precondition for any predictor-side fix |
| A44 | run-to-run variance on this recipe: `postrain30k` 0.669 vs `postrain30k_seed1` 0.679 (**~1.5 %**) | the programme's only seed estimate | `V7_RECIPE_AND_SCALEUP.md` §4 | every single-seed comparison is read against it |
| A45 | τ-ramp gate **CLOSED NEUTRAL** (drift 0.6936 vs 0.6952, nrmse 0.7408 vs 0.7466, cos 0.7513 vs 0.7524; deltas ~20× smaller than the seed spread) | τ stays fixed at 0.996 | `V7_RECIPE_AND_SCALEUP.md` §5.1 | held constant |
| A46 | `--cond-param omega_accel_v`: marginal null (t −1.29), drift unchanged, held-out nrmse **−14.8 %** (single seed) | D2 ADOPTED | E-DEC-65; `V7_RECIPE_AND_SCALEUP.md` §5.2 | held constant |
| A47 | DINOv3 beats V-JEPA2 **9/9** point estimates, 6/9 paired-separated (`n_agents` +0.360 vs +0.239, Δ +0.121 [0.052, 0.188]) | teacher bake-off | E-DEC-68; `…/incoming/2026-08-27-teacher-bakeoff/RESULT.md` | why DINOv3 and not V-JEPA2 |
| A48 | the paired episode-cluster bootstrap is the decision-grade interval; `overlapping_holdout_se` **biases the point estimate** −6.67 % to +11.69 % (27 dumps, 25 distinct arms), bidirectional, up to ×−4.15 on paired deltas including a **sign flip** | estimator discipline | `CLAUDE.md` "Never quote an interval without its estimator"; `taniteval/ci.py` | every CI in this prereg |
| A49 | the measured null bar for the L-probe family is **\|t\| ≈ 2.9, not 2.0** (104 draws) | null calibration | `taniteval/taniteval/null_calibration.py`; `V7_RECIPE_AND_SCALEUP.md` §7 | any t-based criterion |
| A50 | `m_t` already exists as `delta` under `cfg.residual = True` | `predictor.py:285-289`; pairs at `metric_dynamics.py:279` | `D-M_T-SLOT` | D8, three lines |
| A51 | ⛔ **DEFECT 1 — the ratio bar FLOATS**: the k = 60 arm's `scene_factor` **1.7126** turned a written-down **10×** bar into **17.13×**, *"a 71 % harder test than the one written down, and one that could not be known until after the arm ran"*; sign-blind the other way (a genuinely doubled sensitivity reads **+17 %** and is written up INERT) | the actdiv ratio's denominator is a property of the arm being judged | `PREREG_MM_E19_K60_HORIZON.md:74-84` | ⭐ **G-ACT's denominator is PINNED to a named reference arm** |
| A52 | ⛔ **DEFECT 2 — the outcome set admitted no worsening**; the arm fell **0.48×**, off the table, and an off-table result invites force-fitting onto "INERT" | idem | `PREREG_MM_E19_K60_HORIZON.md:85-89` | ⭐ **G-ACT-WORSE branch pre-committed** |
| A53 | the four mandated co-primaries: action side alone; own `scene_spread` (**73.7 %** of MM-E19's ratio fall); the **h ≥ 2 floor** (~**1e-05** on every arm — k = 8, k = 60, incumbent — horizon-INDEPENDENT); the **training-window count** (`o5_k` also sets `o4_n`: **415,002 → 319,002** = exactly 40 × 2,400 ⇒ the k = 60 arm trained on **76.9 %** of the k = 8 arm's windows) | *"how arms on this statistic are judged from now on"* | `PREREG_MM_E19_K60_HORIZON.md:90-105` | G-ACT reports all four |
| A54 | ⛔ backlog **L-13**: the actdiv probe has **no paired episode-cluster bootstrap** — *"a decision statistic without an interval is a gap, not a virtue"* | the gate instrument lacks an interval | idem | a **precondition**: `actdiv_anchored.py` must emit a clip-level bootstrap before G-ACT is read |

---

## B. PUBLISHED-PRIMARY — banked, read in full by the Research Lab's four SOTA passes

*(all cited via `D-SOTA-PASSES-2026-09-03`; each package's §7 lists what was read and from which table)*

| # | lib key | short | number(s) used | where |
|---|---|---|---|---|
| B1 | `2602.12218` | Observer Effect | linear probe on frozen **ρ 0.91** vs full fine-tune **0.05** vs last-layer **0.65**; kinematic invariant **0.94 → −0.03**; deep blocks B5–B10 δ(l) > 0.10, CKA < 0.2; invasive probe MAPE **140 % → 18 %** | sota-encoder F1/F3 |
| B2 | `2601.03460` | FROST-Drive | frozen VLM 14 B RFS **8.17**, ADE@3 s **1.04**; fine-tuned **8.13 / 1.47**; ImageNet ViT frozen 7.39 → fine-tuned 7.79; width 256 → 7.68 | sota-encoder F1 |
| B3 | `2603.24581` | Latent-WAM | Base full FT **89.3**, Small 86.3, **Small-LoRA 84.7, Base-LoRA 68.5** (Table 5); distillation into the trunk **89.3** vs frozen-feature concat 88.0 vs none 88.3 (Table 4) | sota-encoder F1 |
| B4 | `2606.31232` | Delta-JEPA / LDAD | `Δz` beats concat on all four envs **+4.07 / +1.07 / +12.60 / +0.67** (Table 2); λ sweep {0…1000}, **λ = 0 near-collapse, λ = 50 best** (Fig. 3), λ = 10 main; decode target Reacher raw action **81.33** / Δjoint 80.47 / both 76.40 / **Δfinger 64.93** (Table 3); `Δx` from `Δz` r **0.992** vs LeWM 0.765 (Table 5); success 100 / 81.33 / 89.07 / 79.27 (Table 1); encoder = ViT-Tiny **trained from scratch** (§3) | sota-action F2(i) |
| B5 | `2606.20104` | SMWM | λ env-specific: Two-Room **0.1**, Reacher 5, Push-T **30**, Cube 1; **λ = 0.1 collapses Reacher**; inverse MLP on `[z_t; z_{t+1}]` | sota-action F2(i) |
| B6 | `2602.03604` | EB-JEPA | ablation Table 4: variance α = 0 → 47 ± 3, covariance β = 0 → 46 ± 3, temporal δ = 0 → 61 ± 2, **IDM ω = 0 → 1 ± 1 (collapse)**; K = 8, Pareto k = 4 | sota-action F2(i) |
| B7 | `2607.26712` | ActSWM | Table 5: LeWM H=3 gap **−0.005**, H=32 **0.001**, +rollout **0.000/0.002**; frozen-random readout **0.592**; hinge + readout **0.760**; cosine **0.239 → 0.972** | sota-action F1 |
| B8 | `2608.06706` | Dueling WM / AD-JEPA | separation **1.28 → 0.002**; **val-loss checkpoint selection picks collapsed checkpoints 17/36**; post-hoc centring **−0.002…+0.052 → 0.19–0.52**; `ẑ' = B(z) + [Δ(z,a) − Δ̄(z)]`, K = 16 | sota-action F1/F2(iv) |
| B9 | `2512.24497` | What Drives Success | **K = 2 optimal in sim, K = 6 on DROID**, decline beyond (Fig. 3b); TBPTT through the **last** prediction (§C); Remark 1 Lipschitz/accuracy trade-off; the **one-step** embedding loss is the best success proxy, mean −ρ Push-T 0.86 / Wall 0.81 / Maze 0.57 / Metaworld 0.47 (Tables 13–16); frozen DINOv2/v3 > V-JEPA/2 at ViT-L | sota-drift F2/F4, sota-encoder F2 |
| B10 | `2506.09985` | V-JEPA 2(-AC) | §3.1: **T = 2, "only differentiate the predictor through one recurrent step"**; frozen ViT-g | sota-drift F2 |
| B11 | `2606.07687` | action-relevant latents | inverse-dynamics on frozen features: V-JEPA 2 ViT-L **0.40 → 0.85**, **Web-DINO ViT-L −0.01 → 0.16**, SigLIP2 0.05 → 0.17; k-step ID peaks at **k = 4**; ⭐ **action-conditioning paradox (Table 6): action INTO the trunk drops R² 0.26 → −0.32** | sota-action F3, sota-encoder F4 |
| B12 | `2411.04983` | DINO-WM | Table 2 Push-T: **patch tokens 0.90** vs CLS 0.44 / R3M 0.42 / ResNet 0.20 | sota-encoder F2 |
| B13 | `2507.19468` | DINO-world | Table 4: predictor from scratch **46.9 / 87.1 / 59.4**; pretrained + action blocks 49.4 / 91.1 / 61.6; **pretrained fine-tuned 59.4 / 93.8 / 68.7**; Table 1 mid-term VSPW 47.0 vs copy-last 42.1 vs jointly-trained V-JEPA **7.7** | sota-encoder F2 |
| B14 | `2602.18639` | bisimulation adapter on frozen DINOv2 | PointMaze under LCG shift **0.48 → 0.78**; end-to-end without a pretrained trunk **0.26** vs DINOv2 0.86; iBOT 0.72, SimDINOv2 0.40; 196×384 → 196×32 per-patch MLP | sota-encoder F4 |
| B15 | `2608.24044` | JEPA-x | drift **0.361 → 0.104** (cross-prediction, training-only); **REGRESS 0.373, DISTILL 0.516, SHUFFLE 0.540**; control 53.6 → 78.2 %; `drift(h) = E‖ẑ−z‖²/E‖z_t−z‖²` persistence-normalised | sota-drift F2 |
| B16 | `2607.02403` | ACID | planning-cost IDM on **frozen** WM latents: Le-WM Cube 70 → 74, Reacher 76 → 88, Push-T 96 → 100; λ robust 0.005–0.1 | sota-action F2(ii) — the fallback family |
| B17 | `2608.29434` | point-cloud JEPA WMs | frozen Utonia-WM **46.6** on Push-T vs trained Point-LeWM **83.6** | sota-encoder F1 |
| B18 | `2605.25313` | UWM-JEPA | teacher-forced target ⇒ ‖H1‖/‖H0‖ **≈ 0.03**; counterfactual simulator targets **1.00 ± 0.15** | the diagnosis is ours; the fix needs a simulator |

⚠️ **Caveats that travel with B1–B18** (from the passes' own limits sections): the AD-JEPA post-hoc
numbers are on RePo/TIA and DMC, **not driving**; ActSWM's behaviour with a **frozen** encoder is
**UNVERIFIED** (not shown to fail either); LeWM's Fig. 6 bar values are INHERITED from a figure read;
`2607.09185` is banked but **NOT read and NOT cited** — its `cited_by` record is a false citation until
re-banked.

---

## C. INHERITED — not re-verified in this session

| # | claim | source | how it is used |
|---|---|---|---|
| C1 | config E = **336.5 M** params, ViT-5 768×12 encoder, ModernCausalBlock predictor 1024×12, `o5_k = 60`, `param_budget` 350 | `HANDOVER_TO_LOCAL_2026-08-15.md:76-79` (itself "verified against the run's own config.json, MEASURED 2026-08-14") | D5's baseline; the ≈ 286 M option is **derived** from it by the closed-form 12·d² per block and must be replaced by `config.json` before launch |
| C2 | sub-300 M north star | `ROADMAP.md:24` | D5's constraint |
| C3 | v7-tiny ≈ **19 M**, ~17 min/arm Thor / ~29 min dev box | registry §13 scope note (registry:3798); `TanitAD_ValidateAIDesign` §2 | the ladder's cost baseline — ⛔ **invalid once the trunk is ViT-B**; must be re-measured (D3) |
| C4 | refav1 = frozen DINOv3 fp8 features, patch-token readout | readiness report §D; `V7_LAUNCH_GATE.md:623-628` | context only. ⚠️ **refav1 has NO registry row** (0 hits at 10 patterns, two probes) |
| C5 | the four SOTA passes' own INHERITED rows (O5 target teacher-forced; MM-E19 denominator floats) | `…/2026-09-03-sota-action/RESULT.md` §5 | carried with their class |
| C6 | 2.90 s/step on a 19 M tiny rig at the 6 s cost; config-E OOM at k = 60 before checkpointing | readiness report §H | ⛔ **no day count is quoted for v7f**; the run has **no MEASURED cost** |

---

## D. ⛔ UNVERIFIED — flagged, never used as a premise

| # | item | why it matters | how it would be verified |
|---|---|---|---|
| D1 | **a DINOv3 ViT-S/16 exists on HF** | D3's cheap alternative | the only ids this repo names are `facebook/dinov3-vitl16-pretrain-lvd1689m` (`train_v6_staged.py:931`) and `facebook/dinov3-vitb16-pretrain-lvd1689m` (`dino_precompute.py:32`) |
| D2 | **the DINOv3 → `ViT5Encoder` weight mapping** (option D1-B) | it would be the cheap route | it is not attempted; `Block5`'s structure (A34) makes it lossy |
| D3 | **per-arm wall-clock of a ViT-B trunk on the tiny rig** | the ladder's schedule | `--dry-run --dry-steps` on the 4060, before scheduling |
| D4 | **whether the v7 T1 adapter shares `C-REFAV1-KIN-CONTRACT-LAT`** | it gates every lateral row | replay the recorded `(a, κ)` through the v7 adapter's unicycle against GT |
| D5 | **the join of `--require-parity` against a real B1 cache** | no v7 run exists on B1 | the first `--require-parity` launch; `D-V7-WIRING` marks it UNVERIFIED |
| D6 | **truncation's effect on the k = 60 divergence** | R1's whole question | property tests + a 1-step CPU dry-run only, per `D-V7-WIRING` |
| D7 | **`taniteval/tools/actdiv_anchored.py` on a v7 checkpoint at v7f scale** | the gate instrument | it has been run on four v7-tiny arms and two refav1 checkpoints; never at v7f geometry |

---

## E. Conflicts found and reported (not resolved unilaterally)

| # | conflict | resolution rule applied |
|---|---|---|
| E1 | `TanitAD_ValidateAIDesign` **G-RANK ≥ 8.56** vs `MODEL_REGISTRY.md` §13.3 + `v6.py:1517-1552` (*"neither published number survives"*, *"DO NOT FAIL AN ARM ON THE PARTICIPATION CLAUSE"*) | ⛔ **the registry wins** (CLAUDE.md). G-RANK re-specified as a matched-corpus / matched-episode-count / matched-`d` comparison; reported and escalated, not silently changed |
| E2 | `D-V7-WIRING`'s suggested `--bptt-truncate 15` vs every published gradient depth (1–2) | reported as **E4** in the PREREG; the value is decided by rung **R1**, and 4 is the *recommended default*, not an assertion |
| E3 | readiness report §F: *"a register row says '2 of 147 dropped': conflicts with 4,713 − 4,572 = 141; needs reconciliation"* | v7f uses **141 clips with pixels** for the scored split and names the open reconciliation |
| E4 | readiness decision **G.1** (accept P3 as met by `postrain30k_freeze`) vs `H-LEAK-3` (drift is 100 % arithmetic) | **G.1 not adopted**; drift is a diagnostic and leaves the gate |
| E5 | `D-V72-SPLIT`'s md5s are the pre-schema-fix blobs vs `intrain_eval.V72`'s canonical `0ff90213…` / `aa12c948…` | the canonical pair is used; the stale register row is flagged for the Master Mind |
| E6 | `BINDING_TRAINING_IMPROVEMENTS.md:16-33` (*"There is no epoch … raising steps to reach one epoch is forbidden"*) vs `D-ONE-EPOCH` (PI 2026-09-02) | the newer PI ruling supersedes; the doc is stale (readiness report §A) |
