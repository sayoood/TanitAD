# VDAC on P6: feasibility inventory (Decision A, 2026-11-10; read-only, no GPU, no pod, no navtest truth read)

Abbreviations. **PK** = `E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
**TIP** = the mirror `C:/Users/Admin/tanitad-push`, ref `refs/remotes/origin/agent/arch-inf-20260803` @ `364c8609`
(read with `git show`; the E: tree lags it, see C4). Evidence classes: MEASURED (artifact named), INHERITED (another
doc, not re-verified), ESTIMATED (arithmetic shown), UNVERIFIED.

## 0. Conflicts found (reported, not fixed; C1 and C2 change the design)

| id | conflict | evidence |
|---|---|---|
| **C1** | The memo (`../DECISION_P6_NEXT_LEVERS.md` section 3) and the register row `H-VDAC-P6-1` (TIP `GOALS_AND_CLAIMS.md` line 17283) say *"probes on P6's FROZEN trunk features reached drivable IoU 0.18 to 0.27 ... so this needs a TRAINED head"*. **The 0.18-0.27 are UNTRAINED heads (update 0, `w_aux` 0.0) on u400's context, not P6 probes.** Trained heads on u400's frozen trunk read 0.50 to 0.82. No drivable head has ever been trained or read on P6's trunk (not found in TIP rows H-REFE-AUXDRV-1, H-REFE-S1C-1, H-REFE-AUXVETO-1; `grep -c 968a469d` on TIP `GOALS_AND_CLAIMS.md` = 0). | `PK/raw/2026-10-13-aux-p3-floor/full_1058/run.json` (`"update": 0`, `"w_aux": 0.0` per head); `FLOOR_1058.md` "Inputs" (ctx = `model_sft4_B_u400.pt`) and item 3 (*"K2 is a fit bar, not a prior bar"*); trained reads in section 2 below |
| **C2** | The memo quotes the truth-fed ranking AUC 0.932 as the upper bound and omits the **measured selection-level conversion** on the same 1,058 C3 tokens: a pose-only veto from the TRUE raster gains only **+0.367 [+0.015, +0.638]** PDMS, 14 % of the oracle DAC veto **+2.546 [+2.196, +2.880]**; from deployable rasters −0.788 (IoU 0.675) and +0.114 [−0.019, +0.226] (IoU 0.723). | TIP `PK/raw/2026-10-16-auxveto-read/RESULT.md` section 3; TIP `PK/raw/2026-10-16-s1c-read/out/RESULT.md` section 5 |
| C3 | `Project Steering/MODEL_REGISTRY.md` has **no P6 row**: `grep -c 968a469d` = 0, section 14 ends at 14.1 (the final REFe run). Every P6 fact below is from the checkpoint's own `config.json` or the P6 read package. ESCALATE: the registry needs a P6 row before any report quotes P6's architecture. | `MODEL_REGISTRY.md` lines 5240-5262 |
| C4 | The E: tree lags TIP: `PK/raw/2026-10-16-*` (AUXVETO read, S1c read, S1d scope) are absent on E:, and the E: blobs of `refe/scorer_finetune.py` and `refe/aux_label_store.py` differ from TIP (`model.py`, `planner.py`, `aux_drivable_head.py`, `ckpt_io.py` are identical). All trainer line numbers below are TIP's. | `git hash-object` vs `git rev-parse TIP:<path>`, this session |
| C5 | P6's held-out set: **3,137** sets (`../labels/p6_labels_chain.sh:33`, `EXPECT_HELDOUT`) vs **1,705** sets, 24 logs (`../PREREG_DIV_P6.md` section 2). The 24 logs hold **1,708** frames in the label store (this note, section 3), so 1,705 is probably frames and 3,137 sets with twins: UNVERIFIED. | as named |

## 1. P6 as built (item 1)

| fact | value | source (MEASURED unless marked) |
|---|---|---|
| checkpoint | `model_final.pt`, sha256 `968a469d5b9abdd279af...`, 1,288,514,360 B | `F:/TanitAD_p6read_work/ckpt/run/model_final.pt` (sha256 computed this session) |
| params | **322,078,274 total / 18,998,850 trainable** | same dir, `config.json` `params` |
| backbone | DINOv3 **ViT-L/16** (width 1024, depth 24), frozen, rank-32 LoRA on Q and V | `PK/refe/model.py:67-74, :91`; frozen sha `b0fc2885...` in `config.json` `meta` |
| cameras / resolution | **4 cameras** `CAM_F0, CAM_L0, CAM_R0, CAM_B0`, **512 x 960**, patch 16 => 32 x 60 = 1,920 tokens per camera, **7,680 visual tokens** | `config.json` `meta`; `model.py:68-70, :87-88, :486` |
| ego / nav inputs | 7-D ego kinematics (vx, vy, ax, ay, yaw rate, tire steering angle, speed) + NAVSIM `driving_command` one-hot (`nav_mode=command`), embedded into ONE ego token added to the proposal queries only | `model.py:185, :191-192, :771-772`; `config.json` argv `--nav-mode command` |
| recipe | 25 scene-epochs, batch 4 x 64, lr 2e-4, bf16, online PDM labeller, **un-detached scorer context** (`--scorer-sees-trunk`) | `config.json` argv; `PK/SPEC_P6_PAPER_FAITHFUL.md` section 1 rows "detach", "scorer targets" |
| feature maps a new head can read | (a) **`visual_ctx` [B, 7,680, 256]** = `scene_proj(visual)`, the tensor the scoring decoder cross-attends (`model.py:759, :784-786`); (b) `scene_ctx` [B, 64, 256] (16 registers x 4 cams, `model.py:757-758`), read by the proposal decoder. **There is no BEV feature map in P6**: the trunk emits image tokens plus a PETR-style 3D position code (`model.py:747`). | `model.py` |
| information path of `visual_ctx` | backbone tokens + 3D position only; ego and command enter at `:771-772`, AFTER `visual_ctx` is formed at `:759` => a head on `visual_ctx` has no ego, command or goal input by construction | `model.py:744-772` |
| scorer, per proposal | input = the proposal's own 20 x 3 poses (x, y, yaw), detached (`score_q_mlp(traj.detach().flatten(2))`, `:779`), self-attention across the 64 proposals, cross-attention to `visual_ctx`; **no ego token, no command**; output 6 logits NC, DAC, EP, TTC, C, DDC | `model.py:699-709, :779-788`; column order `../eval/read_scorer1p6.py:57-59` |
| pick rule | argmax of p_NC x p_DAC x (5 p_EP + 5 p_TTC + 2 p_C) / 12 (sigmoid of logits, torch float32, first index on ties) | `../eval/read_scorer1p6.py:89` (A10 text) and `pick_logits` |
| frozen vs trained in a check build | `scorer_finetune.py` freezes the trunk and the proposal generator (proposals bit-identical, docstring l.3-6) and can host the drivable head as a **sibling** reading `sctx = visual_ctx`, own AdamW (`--aux-drv-lr`), saved to `aux_drv.pt`; feed `off` keeps the scorer bit-identical | TIP `PK/refe/scorer_finetune.py` docstring l.51-57, `aux_s1_refusals` l.1069-1082 (scene tap refused on this host) |
| head as built | `DrivableAuxHead`: 40 x 32 BEV queries (2 m cells, x in [-8, 72), y in [-32, 32) m, rear-axle frame) + Fourier cell code, 2 x `CrossBlock(256, 8 heads, ctx 256)`, per cell a mask logit and an SDF `6 m * tanh`; **2,440,450 params** | `PK/refe/aux_drivable_head.py:7-15, :142-171`; n_params in `PK/raw/2026-10-13-aux-p3-floor/full_1058/run.json` |
| flags that exist | `--aux-drv`, `--w-aux-drv`, `--aux-drv-tap`, `--aux-drv-depth`, `--aux-drv-feed`, `--aux-drv-labels`, `--aux-drv-shuffle-seed`, `--aux-drv-lr` | `aux_drivable_head.py:110-118`; TIP `scorer_finetune.py:1589` |

**P6 navtest artifacts an S2 read needs (all exist; shapes read, no values):**
`F:/TanitAD_p6read_work/pass2/proposals_p6final.npz`: `proposals` [12146, 64, 8, 3] (8 poses at 0.5 s), `logits`
[12146, 64, 6], `pick` [12146]; `F:/tanitad_scratch/selq2read/table_p6final_c.npz` (sha256 `3bda2080...`): per-proposal
`pdms` [12146, 64], `sub` [12146, 64, 6], `valid`, `pick`, `logits`, `log` (the truth table PESS reads); E2 table
`../e2/e2_table_p6final.npz`. navtest = 12,146 tokens / **136 logs** (navtest label store manifest).

**P6's gap, MEASURED** (TIP `PK/raw/2026-11-09-p6-navtest-read/gap_to_human/p6_gap_to_human.json`): 91.196 vs 94.551; gate
term 3.040; **384 zeroed tokens; DAC 0 on 2.050 % (249 tokens), NC 0 on 1.120 % (136)**; TTC term 1.527 (TTC 0 on
4.92 %). u400 for comparison: DAC 0 on 3.425 % (416), gate 4.155.

## 2. The October aux work (item 2): what was probed, on which features, and why K2 fired

All on **u400's** trunk (SFT-4 arm B, 400 updates; trunk = the final REFe run's), never on P6. Tier T0-like for perception.
Estimator everywhere: paired log-cluster bootstrap, 10,000 resamples, seed 20260927 ("another draw of logs" only).

| read | head / features | result | source |
|---|---|---|---|
| initial-head floor (A2.4) | 3 UNTRAINED heads on cached u400 `visual_ctx`, 1,058 C3 tokens / 123 logs | IoU [0,40) **0.184 / 0.273 / 0.134**; P3 AUC 0.659 / 0.563 / 0.698 vs scorer 0.787; K2 "fires" on an untrained head | `PK/raw/2026-10-13-aux-p3-floor/full_1058/FLOOR_1058.md` |
| truth-fed ceiling (P3 reference) | the label raster itself, 2 m bilinear | pooled DAC AUC **0.932 [0.918, 0.945]** vs u400 scorer DAC head **0.787 [0.731, 0.831]**; within-set 0.948 vs 0.656 | `PK/raw/2026-10-13-aux-s1-readout/p3_reference.json` |
| A2 probe | head on frozen u400 ctx, trained on **navtest folds** (cross-fold, not deployable), 10,000 updates | held-out IoU **0.675**, in-sample **0.894**; near-boundary SDF MAE 2.24 m | TIP `PK/raw/2026-10-16-scene-tap-a2/A2_RESULT.md` |
| S1 (9 arms) | H1/H2 heads, 400 updates, navtrain labels | H1-s0 IoU C3 0.496; **H2 (3 detached scalars fed to the scorer) NOT PROMOTED** (P3 H2_KILLED 3/3 seeds, PDMS pairs not above the seed floor); H1 identity defect | TIP `PK/raw/2026-10-15-s1-read/S1_READ.md` "Outcome" |
| **S1c** | `H1u10k-s0`: same head, frozen u400 ctx, **10,000 updates x batch 8**, navtrain labels | IoU [0,40) **0.723** [0.687, 0.759] C3, **0.764** [0.734, 0.791] natural navtest, **0.821** [0.769, 0.861] held-out navtrain (24 logs); near-boundary MAE 1.97 m; per city (natural) LV 0.860, BOS 0.713, PIT 0.703, **SG 0.632** => **K2 FIRES by 0.04 to 0.08** (row O2 PARTIAL) | TIP `PK/raw/2026-10-16-s1c-read/out/RESULT.md` sections 2-4 |
| S1d scope | why S1c stops short | not log count (S1c drew 64,146 frames from 1,161 of 1,167 train logs); **composition**: LV is 66 % of the draws | TIP `GOALS_AND_CLAIMS.md` line 16774 (CORRECTION R45) |
| AUXVETO (u400, C3 stratified to navtest, pose-only footprint, margin 0.25 m) | veto rule | V-PRED (0.675) **-0.788 [-1.496, -0.197]** AV-HARM; V-S1C (0.723) **+0.114 [-0.019, +0.226]** AV-CEILING-ONLY; **V-GT +0.367** [+0.015, +0.638]; **V-GT-TRACKED +0.794** [+0.628, +0.970] (reported only); **O-DAC +2.546**; V-GT at margin 0: +0.888 (descriptive) | TIP `PK/raw/2026-10-16-auxveto-read/RESULT.md`; S1c read section 5 |
| PRIVGEOM-SAFE (P2, seen scenes) | privileged geometry + the scorer's DAC row, learned combination | DAC-unsafe pick rate 0.0358 -> **0.0099**; but the features partly re-derive the label (R54), sign-clean +0.028 AUC | TIP `GOALS_AND_CLAIMS.md` line 16747 |

**Why K2 fired**, in one line: the bar (IoU [0,40) >= 0.80 on navtest) was missed by a trained head at 0.723 / 0.764,
mostly on non-LV cities, while held-out navtrain read 0.821. **What the veto read adds, and the memo omits:** IoU is
not the binding constraint. A footprint check needs about 0.25 m boundary accuracy against a ~2 m near-boundary error,
and even the TRUE raster under a hard pose-only veto captures 14 % of the oracle (the tracker alone holds 27.8 % of
NAVSIM's DAC failures, INHERITED `AGREEMENT.json` via `PREREG_AUXVETO.md` section 1).

## 3. The label store `E:/Projects/TanitAD-artifacts/aux_drivable` (MEASURED from the manifests)

| store | keys | logs | coverage | content | sha256 |
|---|---|---|---|---|---|
| `label_store` (navtrain) | **103,139** (bank 103,037 + db 102) | **1,192** | 99.856 % of navtrain.yaml's 103,288 tokens; **149 missing** (`missing_tokens.txt`); 0 failures | by map: LV **67,593**, BOS 12,925, PIT 10,807, SG 11,814 | `4a17db67...` |
| `label_store_navtest` | **12,146** (all from DB) | **136** | 100 % of navtest; 0 missing | LV 4,064, BOS 3,731, PIT 2,437, SG 1,914 | `985a3bc7...` |

- Raster: signed distance, int8, 0.1 m unit, clip +/-12.7 m, **0.5 m, 160 x 128, x in [-8, 72), y in [-32, 32) m,
  rear-axle frame at t0**, positive inside, map radius 100 m (manifest `raster`). Polygons ROADBLOCK + INTERSECTION +
  CARPARK_AREA, NAVSIM's own DAC polygons (`PK/SPEC_AUX_DRIVABLE_HEAD.md` 3.1, INHERITED 100.00 % agreement).
- Leakage (manifest `leakage`, `label_store_navtest/disjoint_vs_train.json`): navtrain/navtest **0 shared tokens, 0 shared logs**.
- QC: the teacher's rank-0 target points lie inside the label on mean 0.99999 of points over 103,037 rows.
- **Agents: none.** The store is drivable-area only (schema `aux-drv-label-v1`); no box, occupancy or agent-future label
  exists anywhere I looked. An agent check needs a new label build (section 5).
- The 24 SFT held-out logs (`PK/splits/sft_heldout_logs.txt`): **LV 15, PIT 4, BOS 4, SG 1 logs**; 1,013 / 195 / 257 /
  243 = **1,708 frames** in the store. SG rests on ONE log, so a city-reweighted held-out read on these 24 alone is weak.
- `run_meta_all.json` argv paths name `D:/...` (the store moved to E:); the manifests' sha256 are what binds.

## 4. How a per-proposal check can enter the pick (item 3)

| entry | what it does | measured prior | admissibility |
|---|---|---|---|
| **veto** (AUXVETO rule) | drop proposals whose footprint leaves the raster by > m; fall back to the pick if all are dropped | u400: GT +0.367, S1c-quality +0.114 (n.s.), A2-quality -0.788 | vision-only (raster + the proposal): admissible |
| **multiplier on p_DAC** (soft veto) | p_DAC' = p_DAC x q, q in [0, 1] from the raster's signed distance along the footprint; NC kept as its own factor | none measured (new) | same as veto |
| learned fusion q = f(raster features) | q calibrated on navtrain DAC labels of P6's proposals, head predictions out of fold | PRIVGEOM-SAFE shows geometry + the row is strong with PRIVILEGED geometry (label re-derivation caveat) | same; the fit must not use P6's in-sample logits (section 5 of DESIGN) |
| scorer input (H2 feed) | 3 detached raster scalars into `score_q_mlp`, scorer fine-tuned | S1 H2 NOT PROMOTED (400-update head); and fine-tunes share P6's blind spot (`H-PESS-P6-1`, PE4) | admissible; weakest prior |
| tracked footprint (any of the above) | the footprint on NAVSIM's LQR-tracked states instead of raw poses | u400 GT: +0.794 vs +0.367 pose-only | ⚠️ needs ego kinematics at inference; P6 already reads them (`model.py:185`), but the tracker's initial state includes a tire steering angle: **PI/MM ruling needed** (`PREREG_AUXVETO.md` section 7 left it open) |

**Information disjointness.** P6 has **no situation classifier**: its inputs are images, ego kinematics and NAVSIM's
`driving_command` (`model.py:185-192`). The head reads `visual_ctx` only, formed before any ego or command path
(`model.py:759` vs `:771`). The check's output replaces nothing but the DAC factor; NC, EP, TTC and C keep P6's heads.
So the goal/command path and the check stay information-disjoint at inference; a G-DVB check must pin it (DESIGN section 6).

## 5. Ceiling arithmetic for P6 (ESTIMATED, no navtest truth read)

Per DAC-zero token, u400's oracle veto recovered 2.546 / 3.425 = 0.743 PDMS points per 1 % of tokens (AUXVETO,
stratified). Assuming P6's DAC-zero tokens recover alike: **O-DAC on P6 ~ 2.050 x 0.743 = 1.52 points.** The u400
capture ratios then project to P6 as: GT pose-only veto 0.144 x 1.52 = **0.22**; GT tracked veto 0.312 x 1.52 = **0.47**;
GT veto at margin 0 (post-hoc) 0.349 x 1.52 = **0.53**; S1c-quality veto 0.045 x 1.52 = **0.07**. The P1 bar is
d > **0.4971**. ⇒ **A DAC-only hard veto is projected below the bar even with a perfect raster.** The design therefore
uses a soft entry and measures its ceiling at zero GPU before any training (DESIGN section 5, PREREG S0). The NC side
(136 tokens, ~0.83 ESTIMATED by the same per-token rate) and TTC (1.527 measured) are agent terms this store cannot reach.

## 6. UNVERIFIED (each a named gap, none guessed)

1. No drivable head has been trained on **P6's** trunk; whether P6's PDM-shaped `visual_ctx` reads the road better or
   worse than u400's is unknown (the S1c recipe's IoU on P6 is the first S1 number).
2. The u400 -> P6 transfer of every ceiling ratio in section 5 (different scorer, DAC-zero share 2.05 % vs 3.43 %).
3. Location and completeness of P6's navtrain label bank (~60,000 train sets + the held-out sets with per-proposal
   PDM sub-scores): `../labels/p6_labels_chain.sh:25` writes `/workspace/data/p6base_labels` **on the pod**; a dev-box
   copy was not found. S0-NT needs it.
4. The held-out count conflict C5.
5. Whether NAVSIM's LQR tracker at inference is admissible (steering-angle input vs the PI's yaw-rate ruling,
   memory `pi-ruling-yaw-rate-not-steer.md`), and its CPU cost for 64 proposals per token (not measured).
6. P6's latency on Thor (not found); the head's cost is FLOP-share only (+0.53 % of the trunk's forward,
   `PK/SPEC_AUX_DRIVABLE_HEAD.md` section 0, analytic, ViT-L 4-camera trunk).
7. Whether `scorer_finetune.py`'s preflight G1-G16 passes with `--aux-drv` on a 6-row P6 head (S1c ran on a 7-row u400
   head; the WIRING_NOTE section 2 already flags the 6-row preflight as untested).
8. The proposal extent: the share of P6 navtest footprint corners beyond x = 72 m (fast tokens) was not computed
   (it reads only proposals, but I left it for the S0 script so it lands with its gate).
9. Agent labels: whether the pod's NAVSIM `metric_cache` holds the per-token agent boxes and logged futures an
   occupancy label would need (location and schema not read).
