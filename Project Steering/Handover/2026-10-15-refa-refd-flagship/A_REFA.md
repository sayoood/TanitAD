# Section A — the REF-A line (refav1) — handover

> **Source of truth for this section:** the landed tip `460508ed2f5e120667859e637ac233c28a48d90b` (2026-10-15 21:54 +0200), read with `git --git-dir=/c/Users/Admin/tanitad-push/.git show <tip>:<path>`. `Project Steering/MODEL_REGISTRY.md` wins over every other doc; where the registry is stale or silent this section says so. Abbreviations: **AI/** = `TanitAD Research Lab/Architecture & Inference/Research/`; **P4S/** = `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/`; **REG** = `Project Steering/MODEL_REGISTRY.md` (line numbers at tip); **GC** = `Project Steering/GOALS_AND_CLAIMS.md`.
> Evidence classes: **M** = MEASURED (artifact path given) · **P** = PUBLISHED · **I** = INHERITED (not re-verified by me) · **E** = ESTIMATED · **H** = HYPOTHESIS. Tiers: **T0** teacher-forced / WM diagnostic · **T1** self-action OPEN loop (PI ruling 2026-09-02: a planner feeding its own predictor is still open loop) · **probe** = representation readout, no tier. Every comparison is a **paired episode-cluster bootstrap, n_boot 2,000**, unless stated. ⛔ **Closed loop has never been run for refav1** (REG:1274). ⛔ **Training-seed variance has never been measured** for refav1 (one run).

---

## 1. Status in one paragraph

**refav1** (`refav1-b1-v72-ep3-speed`) is a 182,459,701-parameter latent world model on **frozen DINOv3 ViT-L/16 fp8 features** with a three-rate FiLM hierarchy and a **test-time iCEM planner**; its one complete run finished at **step 21,109 (one epoch) on 2026-09-04** and the line has had **no training since** (REG:2081-2095). Every zero-training lever has now been tried and **eliminated on this checkpoint**: the shipped planner emits an injected do-nothing plan on **282/282** full-grid windows (`cl − ha0` ADE **+0.0158 [+0.0007, +0.0315]**, separated WORSE; M, T1; REG:2094, 2129-2137); the best tuned configuration `loncomb3` **FAILS-WORSE on the full 141-episode grid at both inference seeds** (`A1 − ha0_ext` **+0.0896 [+0.0117, +0.1756]**; M, T1; `AI/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` §0) and a planner-free blend `damp50` beats it. ⭐ **The single most important fact:** the pre-registered trunk probe found that the **frozen refav1 trunk carries NO trajectory information beyond 8 t0 kinematic features** — under every readout tried (linear/MLP, global/4×10 spatial, 113 or 600 fitting episodes) — while an **MLP on those 8 kinematic features alone (`P10`) is the strongest predictor the programme has measured on this grid: 2 s ADE 0.2229 m, 6 s ADE 2.9257 m on 2,399 eval windows / 141 episodes** (M, T1-equivalent; `AI/2026-09-27-refav1-trunk-probe/RESULT.md` §3). ⇒ The only lever left on this line is **option T: retrain the trunk (adapter + world models) jointly with a trajectory loss** — blocked on a **PI storage/compute decision** (~303 GB fp8 train features, deleted from Thor, or on-the-fly encoding). The PI's R1–R6 code for it is **built and tested but NOT landed** at tip (§5, §6), and **neither the registry nor the claims register records the 2026-09-27 verdicts yet** (§8).

---

## 2. Architecture and recipe

| item | value | evidence |
|---|---|---|
| Encoder | **DINOv3 ViT-L/16**, frozen, d_enc 1024, **640 patch tokens (16×40) at 256×640 cylindrical, 120° HFOV**; CLS + 4 registers dropped; every 2nd frame encoded bf16 → **fp8 (e4m3) cache**; ImageNet mean/std at encode | `stack/tanitad/refs/refa_v1.py:75` (`DINOV3_GEOMETRY`); `stack/scripts/dinov3_fp8_encode_ship.py`; GC `D-REFAV1-CACHE-COMPLETE` (I) |
| Adapter | **WideAdapter, no bottleneck** (`d_state` 1024 ≥ `d_enc`), trained at `adapter_lr_mult 0.1` | `refa_v1.py:714-716`; ckpt `config.json` (M) |
| World models (3 rates, 6 s) | **operative** 0.2 s × 30 (6 layers, 8 heads, window 4) · **tactical** 0.6 s × 10 (64 queries, 4 layers) · **strategic** 3.0 s × 2 in-window + extension to 12 s on its own subspace predictor (str_dim 256); chained **strategic → FiLM → tactical → FiLM → operative**; objective = **predict future PATCH features (L2)**, `w_feat_op 1.0 / tac 0.5 / str 0.25` | `refa_v1.py:706-760`; ckpt `config.json` `cfg` (M) |
| Heads | `lat_head`, `lon_head` (tactical decision, v7.0 vocab), strategic `route_logits`; `w_tac_label 0.1`, `w_str_label 0.1`; **proposal/imitation head `w_aux_head 0.0` — never trained** | `refa_v1.py:1537-1539, 2060-2075`; `config.json` (M) |
| Actions / ego | `a_dim 2` = (a, steer); **`--speed-channel` ON** ⇒ predictor input (a, steer, v_k/30), v_k integrated from the measured v0 (never future GT); a missing v0 **raises** | REG:2093; GC `D-REFAV1-EP3-SPEED` |
| ⚠️ Action units | `actions[:,0]` is a **road-wheel STEER ANGLE** `arctan(L·κ)`, L = 2.9 — **not κ** (≈2.9× its magnitude); banked evals ran `--action-units kappa` (legacy) | REG:2170-2173; RETRACTION_LOG #15 |
| Nav | `nav_inject: True` — nav embedding added to the **tactical ctx** and to the **field predictors' intent**; strategic subspace predictor stays nav-free | `refa_v1.py:905, 1561-1564, 1658-1664` |
| Planner | **iCEM** 300 samples × 30 iters × 30 elites, coloured noise β 2.5, **horizon 10 × 0.2 s = 2.0 s** (`plan_horizon_s 2.0`) while the goal/cost spans 6 s; `plan_level 'tactical'`, `verify_on_operative`; injects `cv / hold_v0 / decel_1.5 …` baselines into its own population; `kappa_max 0.2`, `a_max 4.0` | `stack/tanitad/refs/refa_v1_plan.py:65-93`; `refa_v1.py:782` |
| Cost | goal term (`cos` default; `chord / ccos / ccosh` selectable) + `W_JERK 0.02` + `W_KAPPA 0.05` + `W_VEND 0.10`; turn token commands `GOAL_KAPPA_TURN 0.08` 1/m (R 12.5 m). ⚠️ `W_VEND` is **inert in every arm ever run**: `target_speed` is never passed to `plan()` | `refa_v1.py:119, 433-435`; `RESULT_HARVEST.md` §3 (M) |
| Params | **182,459,701** (checkpoint) | REG:2089 (M) |
| Training recipe (ep3) | `stack/scripts/refa_v1_train.py` · bs 8 · lr 3e-4 · adapter_lr_mult 0.1 · **21,109 steps = one epoch** · `--target-space frozen --detach-aux-targets --bptt-truncate 15 --ema-targets` (decay 0.996 → 0.999) `--precision bf16 --tf32 --skip-nonfinite --speed-channel` · lru 64 · seed 0 · clip 1.0 · torch 2.13.0+cu130 on Thor; **3.47 s/step at step 100 (≈ 20 h/epoch)** | dev-box `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json` (md5 `bd3ebed7…`, read this session, M); GC `D-REFAV1-EP3-SPEED` (M there) |
| Corpus | **B1 / v7.2 labels: 168,910 windows over 4,572 episodes**, labels md5 `0ff90213…` (train) / `aa12c948…` (eval), join 4,572/4,572. ⛔ **NOT the parity corpus `physicalai-train-e438721ae894`** — allowed by PI ruling (GC:3325/5735, I) ⇒ refav1 numbers are not comparable with REF-A dinov2/dyn-in rows | REG:2092 (M) |
| ⚠️ Code provenance | trained from **Thor's older stack** (`refa_v1.py` 1,885 lines vs 3,030 at tip); a strict load into tip code works (`missing_keys: []`), but **a re-run from tip is NOT a replication**; the registry still does not record this (BACKLOG A13c, open) | `Project Steering/Reports/2026-09-15-1445-refcv6-refav1-v7-status.md:128` (I) |
| Checkpoint | `ckpt.pt` 2,122,997,633 B, md5 `1189bc020018c2c67ce03d566c390285`, keys `cfg/model/opt/step`, 407 tensors. Copies: **Thor** `/home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt` (+ `/home/nvidia/refav1_lon/ckpt/ckpt.pt`) and **dev box** `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt` (size verified this session). **No HF copy** | REG:2084-2090; ccos-eval RESULT header (M) |
| Eval harness | `taniteval/tools/refav1_arm.py` (Thor invocation: `taniteval/tools/REFAV1_ARM.md` §6; dev-box invocation: `AI/2026-09-27-refav1-fullgrid-loncomb3/raw/run/run_fullgrid.sh`); paired deltas `taniteval/tools/refav1_paired_delta.py`. Arms: `cl` (planner, T1), `ha` (hold recorded action, T1), `ha0` (constant velocity, T1), `ha0_ext` (hold measured a0 + yaw-rate, T1), `ol` (recorded future actions, **T0**). Planner-free floors: `damp50 = 0.5·ha0 + 0.5·ha0_ext`; `kd_x` (fixed damped kinematic prior — ⚠️ defined only in the trunk-probe code, `AI/2026-09-27-refav1-trunk-probe/code/probe.py`, and the full-grid `raw/analysis/explore/` builders; **not in `stack/` or `taniteval/` at tip** — the unlanded R5 build carries it as `--r5-prior kdx`, and the 30-step and 10-step `kd_x` differ, so name which); `kd_ha0_ext`; `stack/tanitad/models/kinematic_prior.py` holds the refcv-line priors (`ha0_ext_pose`) | REFAV1_ARM.md; full-grid RESULT §0/§4 |
| Eval grids — ⛔ never mix them | **(a) full grid**: v7.2 EVAL split, 141 episodes, stride 40 ⇒ **282 windows**, K = 10 @ 0.2 s — the decision grid; ~2.1 h/arm on the 4060 (M: two arms 09:33–13:48 Berlin). **(b) p4 panel**: 8 turn-dense episodes, stride 16, **40 windows** — ⛔ *not admissible for lateral levers* (full-grid RESULT §4.2). **(c) trunk-probe grid**: the same 141 episodes, **2,399 windows**, 2 s and 6 s horizons | — |
| Data on the dev box (existence verified this session) | eval fp8 cache `C:/Users/Admin/tanitad-data/refav1-eval141/refav1-fp8-eval` (142 entries), episodes `…/refav1-eval141/eps`, labels `C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz`, probe workspace `C:/Users/Admin/refav1_probe/`, full-grid outputs `C:/Users/Admin/refav1_fullgrid/`, R1-VMAX dev copy `C:/Users/Admin/refav1_dev/`, p4 dumps `C:/Users/Admin/refav1_margin/p4out/`. ⚠️ `C:/Users/Admin/refav1_trainprobe/` (600-clip TRAIN features) is **GONE** — regenerable with the banked `code/pull_encode.py` (M: 2,834 s for 600 clips) | — |
| Thor-side | train fp8 cache `/home/nvidia/data/refav1-fp8-train` ⛔ **deleted** (~303 GB to rebuild); eval cache `/home/nvidia/data/refav1-fp8-eval`; v2ep source frames (4,713 episodes) still on Thor | full-grid RESULT §5; `P4S/REFAV1_RETRAIN_DECISION.md` §4 (I) |

---

## 3. Lineage

| run | encoder / key change | status | headline (class, tier, estimator) | verdict |
|---|---|---|---|---|
| `refa-phase0-30k` | frozen DINOv2-B/14, grid adapter, no speed | superseded | ADE@2s 3.726 (heldout split-mean — ⛔ deprecated estimator) | pre-fix baseline; voided by D-A3 |
| `refa-plus-speed-30k` | + `v0` as 3rd action channel | archived | trainer fwd-ADE 3.73 → 0.83, speed-R² 0.61 → 0.965 (trainer metric, not eval) | the isolated proof of the speed fix (D-A3) |
| `refa-4brain-speed-30k` = **REF-A dinov2-4b** | + `--four-brain` | ✅ accepted as reference | ADE@2s **2.1675 [1.9081, 2.4212]** full-set (M, T0, episode-cluster bootstrap; REG:3760); does not beat CV | the frozen-encoder reference (REG §2.1) |
| `refa-ijepa-4brain-speed-15k` | I-JEPA ViT-H/14, 320 eps | diagnostic only | ⛔ canonical val **80 % leaked** — unusable | REG §2.2, R8 |
| `refa-dynin-4brain-30k` = **REF-A dyn-in** | + `[v0, yr0]`, ego-dropout 0.25 | ✅ complete 29,999 | ADE@2s **3.0471 [2.4984, 3.6878]** full-set (M, T0); flagship paired Δ **+2.62 [+2.0945, +3.2570]** | **H4 closed NEGATIVE** (D-A5): capability ceiling, 94.2 % longitudinal error |
| `refav1-b1-v72-1ep-21109` | **new design**: DINOv3-L, WideAdapter, 3 WMs + iCEM | retired at step 1,000 | collapsed ~step 450 (I); relaunched with frozen targets; step-1,000 T1 read = CV straight line on 140/140 (M, 20-clip slice) | ⚠️ the directory name is not its step |
| `refav1-b1-v72-ep2-ema-bf16` | + EMA teacher, bf16/TF32 (6.1× faster) | retired at 6,850 | speed never reached the model (`act.0.weight` [1024, 2]) (M) | GC `D-REFAV1-EP3-SPEED` |
| **`refav1-b1-v72-ep3-speed`** | + `--speed-channel` | ✅ **complete 21,109** (2026-09-04) | `cl` ADE **0.5474 [0.4839, 0.6108]** vs `ha0` 0.5316, `ha` 0.5391, `ha0_ext` 0.5207, `ol` (T0) 0.4237 (M, T1, full grid; REG:2094; ccos-eval RESULT §3a) | **planner = injected baseline 282/282; does not beat doing nothing** |
| "REF-A v2" | trainable DINOv3-distilled encoder | ⛔ **never commissioned, no SPEC** | — | a recommendation only (`products/P4-training-pipelines/V7_FULLSCALE_RECOMMENDATION.md:344-350`, 2026-08-24). ⚠️ In that doc "REF-A v1" means the OLD DINOv2 arm, not refav1 |
| **option T** (refav1 trunk + R5 head retrained jointly) | trajectory loss as a first-class objective | ⏸ **recommended, not launched** | — | §5 item A |

---

## 4. What was measured on refav1 (chronology, 2026-09-02 → 09-27)

All on checkpoint 21,109 unless stated. "Eliminated" = a candidate ruled out; "named" = the next lever the package recorded.

| date · package | question | result (class · tier) | eliminated | named next |
|---|---|---|---|---|
| 09-02 `AI/2026-09-02-refav1-target-space-collapse/` | does the shipped anti-collapse config stop the collapse? | CONFIRMED on every pre-committed read (M, T0, 250 steps/arm, Thor); ⚠️ licenses the *configuration* (target_space + detach), not "frozen targets" alone | — | relaunch (done) |
| 09-02 `…-step-profile/` | where does the step go? | the 30-step operative rollout is ~83 % (bs 1, M) / ~92 % (bs 8, E) of the step; compute-bound; **bf16 3.0× faster** (M); the full step does **not fit an 8 GB card at any batch or precision** | launch/loader overhead | bf16 (adopted) |
| 09-02 `…-ema-inflation/` | does an EMA teacher pin the tactical scale? | **VOID** — dev-box rig could not reproduce the inflation (M, T0) | — | ran on Thor in ep2 |
| 09-03 `…-step1000-read/` | first T1 read (step 1,000, 20 clips) | plan = constant-velocity straight line on **140/140** (M, T1) | — | full-epoch read |
| 09-03 `…-anchored-actdiv-refav1/` | is the WM lateral-insensitive? | `H-REFAV1-LAT-INSENSITIVE` **REFUTED** (M, T0, CPU) | "the WM ignores steering" | cost surface |
| 09-03 `…-refav1-cost-surface/` | decompose the planner cost | goal term negligible against the penalties (M, T0) — ⚠️ its 1.63e-10 figure is the **step-1,000** checkpoint's (RETRACTION_LOG #21) | — | cost scale |
| 09-04 `taniteval/results/RESULT-refav1-21109-openloop.md` | the epoch-end read, full grid | injected baseline **282/282**, κ ≡ 0 on 282/282, **2 distinct plans** in total. Four families, `cl − ha0`: ADE +0.0158 sep. worse; LON speed **+0.0308 [+0.0049, +0.0585]** worse, along **+0.0258** worse; LAT exactly 0.0000 [0,0] ×3; TAC lat 0.0000, TAC lon −0.0035 n.s. `cl − ha`: LON speed **+0.2706** worse; LAT cross **−0.1569** better (a straight line beats a drifting held κ); TAC lon **−0.1277** worse. **STRATEGIC: route head = pure nav echo** (1.0000 true nav / κ −0.1427 shuffled / majority class with nav withheld). **What training DID buy:** WM beats persist-last-field from 1 s (+0.1129) to 6 s (+0.2454), loses at 0.2 s (−0.0442) (M, T0); LON tactical head **0.4326 [0.3546, 0.5177]** vs 0.2979 majority floor under `nav_zero` (M); tactical decoder asks for turns on 13.5 % of windows (M). Distance-keeping: `cl` ≈ `ha0` (n = 87, I via survey) | "early in training" | cost scale |
| 09-04 `AI/2026-09-04-refav1-cost-scale/` | the cost, in units | WM varies along κ by a **median 1.99e-06** (n = 282); float32 ties the κ argmin on 45.4 % of windows; **the binding penalty is JERK** (99.5 % of the median 0.362 penalty) (M; REG:2141-2150) | re-weighting alone | ccos |
| 09-05 `…-refav1-ccos-eval/` | centred cosine on the real code | `ccos` naive ADE **0.9268 [0.7808, 1.0819]**, weight-compensated **0.7116 [0.6291, 0.7973]** vs `ha0_ext` 0.5207 — **worse** (M, T1, full grid; per-arm CIs). ⚠️ REG:2095 still says "IN PROGRESS" — stale | `ccos` alone | make-it-drive |
| 09-05 `…-make-it-drive/` | why does it not turn? | **two gates in series**: the goal head decodes LANE_KEEP on 86.5 % ⇒ κ = 0 on 244/244; under `cos` a correctly decoded turn is refused on **38/38** (M, T1). Lever `--lat-logit-bias` A/B: ADE **+0.8927 worse**, 11.2× its seed replicate (M, T1, `AB_RESULT.md`) | decision rule; "must retrain the WM" | vocabulary |
| 09-05 `…-goal-margin/` | is it the margin? | **it is the VOCABULARY**: only κ ∈ {0, ±0.08} can be sustained while the corpus curves at R 100–1000 m; LANE_KEEP is vocabulary-optimal on 90.6 % of GT-turn windows; the head detects turns (AUC 0.88) and L/R (AUC 0.90) (M) | head re-fit (nav echo / shrink-to-zero) | `--goal-kappa-turn` |
| 09-05 `…-vocab-l3/` | L=3 curvature vocabulary | panel **un-runnable as pre-registered** (deficit inside 6.1× seed noise, 10 windows); vocab implemented and pinned; **the trained head cannot choose among 3 magnitudes** (M) | — | a chooser (never built) |
| 09-05 `…-surface-sweep/` | 59-setting cost screen | `W_JERK` inert on 24/26 candidates; decoded goal directionally right 0.778 [0.625, 0.923] but commits to a turn on only 0.205 of GT-turn windows (M, T1) | most weights | cost geometry |
| 09-05 `…-cost-geometry/` (p4 panel) | the cost's geometry | shipped config = `ha0` **bit for bit on 40/40**; `W_KAPPA` interior optimum **15.11245**: ADE 0.8934 vs `ha0_ext` 0.8772 — parity (`cl − ha0_ext` +0.0162 [−0.1648, +0.1980]); beats `ha0_ext` on turns, loses on straights; **leaves the μ = 0.7 friction circle on 29.6 % of windows at v0 ≥ 2 m/s, peak 3.262 g** — `--kamm-mu 0.7` fixes that at no measured cost; `ccosh` and the seed ladder are **nulls** (M, T1; REG:2096) | candidate set; hold branch | longitudinal |
| 09-05 `…-longitudinal/`, `…-lonshift-t1/` (p4) | LON levers | `a0_shift` cuts LON speed MAE **−0.2283 [−0.3227, −0.1420]** (Thor) / −0.2263 (dev box), 12× its seed floor, replicated on 2 GPUs (M, T1, 8 clusters); `lonshift` vs `ha0_ext` ADE −0.0889 [−0.2049, +0.0137] (parity), FDE separated better, LON still worse | — | full-grid confirmation |
| 09-05/06 `…-close-the-gaps/` | why still under-turning? | the operative cost was **goal-blind** ⇒ a **3.9× under-turn** (0.02066 vs commanded 0.08) (M, T1, p4); goal-conditioned lateral cost (`--w-kappa-by-goal`) implemented, 15 tests; pre-registered `kammshift` **FAILS** (ADE 1.0652 > bar 0.8892; M, `…/raw/refav1_margin_banked_2026-09-27/RESULT_HARVEST.md` §1). ⚠️ the A3/A4 goal-conditioned arms ran on Thor and were **never read back** (UNVERIFIED; GC rows still say RUNNING) | — | — |
| 09-06 `…-distance-keeping/` | can it keep distance? | DK cost term fires on 21/90 lead windows, re-ranks 21/21; moved headway/TTC the right way on an **n = 4 window** smoke (M, T1, plumbing only); the gap is an ORACLE label | — | decoded gap; DK sweep (never run) |
| 09-06 `…-perception-probe/` | is the lead in the latent? | lead gap decodable **R² 0.3632** vs pixels −0.0513 (paired +0.4145 [+0.2018, +0.6120]); **closing rate absent** (every arm's CI spans 0) (M, probe, 42 clusters) | "a perception defect" for the gap | closing-rate representation |
| 09-06 `…-decoded-gap/` | vision-only gap head for DK | **BAR-2 FAILED** 12/21 = 0.5714 vs a committed 0.80 (16/21 after a post-hoc bias correction — still a miss); BAR-3 passes (M) | — | — |
| 09-27 `RESULT_HARVEST.md` (p4) | read the 17 stranded arms | `loncomb3` (= `ccos`, weights 0.02/15.11245/64.297, `--a-sustain-mode a0_shift --jerk-seam a0`) is the best p4 point estimate: ADE 0.7739, LON speed 0.4665; seed floors config-dependent (+0.0015 to +0.1035) (M, T1, 8 clusters) | — | full grid |
| ⭐ 09-27 `AI/2026-09-27-refav1-fullgrid-loncomb3/` | does `loncomb3` beat doing nothing on 141 episodes? | **FAIL-WORSE** at both seeds (`A1/A2 − ha0_ext` **+0.0896 / +0.1515**, separated); `damp50 − ha0_ext` **−0.1576 [−0.2297, −0.0990]**; `b50(A1) − damp50` **+0.0505 [+0.0120, +0.0918]** worse; the inference-seed replicate is itself separated, **+0.0619 [+0.0094, +0.1169]**. Four families vs `damp50`: tied on all LON; loses all LAT (cross **+0.3109**, heading **+1.87°**, yaw-rate **+0.0460**); TAC lat **−0.0780**; **STRATEGIC UNAVAILABLE** (no route label on the planner surface, n = 0). Exploratory: `kd_ha0_ext − ha0_ext` **−0.2282 [−0.2946, −0.1702]**, a stronger planner-free floor (M, T1) | **every zero-training planner lever on 21,109** | a trained trajectory readout |
| ⭐ 09-27 `AI/2026-09-27-refav1-trunk-probe/` | does the frozen trunk carry trajectory info beyond t0 kinematics? | **NEGATIVE** under every readout: `P4 trunk − P2 kin` +0.0042 [+0.0016, +0.0071] (2 s); A3 `P9 trunk_sp_mlp − P10 kin_mlp` **+0.0054 [+0.0016, +0.0093] @ 2 s, +0.0987 [+0.0332, +0.1658] @ 6 s** (worse); validity controls OK. ⭐ `P10` (MLP on 8 kinematic features, fitted on 600 TRAIN episodes): **2 s 0.2229 / 6 s 2.9257 m**; vs the linear readout at 6 s **−0.5024 [−0.6331, −0.3767]**; vs `ha0_ext` at 2 s ADE **−0.3495**, FDE −0.9272, heading −2.92° (worse only on accel MAE +0.104); at 6 s, point estimates `damp50` 3.45, `kd_x` 3.56, `ha0_ext` 5.36 (M, T1-equiv., 2,399 windows / 141 clusters). The only stratum where the trunk helps: 2 s `stop` −0.0690 on 14 windows (reported, not read) | **option H (heads-only)** | **option T** |

PUBLISHED context for the trunk verdict (banked in the Library and cited by the RESULT): an ego-state-only MLP matches perception-based planners on open-loop nuScenes L2 — Zhai et al. 2023 (`2305.10430`); E2E planners given ego status come to rely on it — Li et al., CVPR 2024 (`2312.03031`). ⇒ vision has to earn its place where kinematics fail: the 6 s horizon and event windows (turn onsets, stops, lead braking), not the pooled 2 s mean.

---

## 5. Planned and NOT finished

| # | item | spec / where | how far it got (verified at tip) | blocked by |
|---|---|---|---|---|
| **A** | ⭐ **Option T — retrain the refav1 trunk (adapter + WMs) with the trajectory loss as a first-class objective; R5 head trained jointly; prior = the learned kinematic MLP** | `P4S/REFAV1_RETRAIN_DECISION.md` (FINAL, 2026-09-27); bars committed in its §3 | **Recommendation only.** No run, no PREREG file of its own, no launch-gate profile. ⚠️ The R5 prior the build offers is `{kdx, damp50, cv}`, **not** the P10 MLP the decision names (`P4S/refav1_r1r6/RESULT.md` §0) | **PI**: ~303 GB fp8 train features placed (D: 92 GB free, C: ~70 GB, Thor 96 GB — none fits; I) **or** train on Thor with DINOv3 encoded on the fly from the v2ep cache (per-step cost **UNMEASURED**); a compute slot (Thor runs refcv8 at time of writing — I, from the session's memory index, not re-verified); items B, C, F |
| **B** | **R1/R3/R5/R6 build** — `--strategic-off`, `--vmax-input`, `--w-goal --goal-negatives --cot-negative-sidecar`, `--w-speed-band`, `--w-traj --r5-prior --r5-kappa-source`, refav1 G-DVB registry (48 dests) | `P4S/refav1_r1r6/RESULT.md` (+ landing table §9.1, 11 files) | **Built, tested (511 passed / 10 skipped on its set; 0 new failures), bit-identical when off (6,088 tensors + 310 values) — and NOT LANDED.** Verified this session: `HEAD:stack/tanitad/refs/refa_v1.py` is still the base blob `d504eb9b…` (the build's is `6e0988aa…`); `refav1_traj.py`, `train/declared_vs_built_refav1.py` and all 6 new tests are ABSENT at tip. The sources are banked under `P4S/refav1_r1r6/code/fix/…` | Master Mind landing; PI decisions **D1** κ0 source at inference (`steer_t0` is "unruled at inference" per SPEC_REFCV7 §10) · **D2** cot-absence negatives on v7.2 (17/22 vs 21/22 goal classes) · **D3** SPEED_BAND as target while max speed is an input (partial echo) · **D4** prior default · **D5** max-speed inference source (oracle label today) |
| **C** | **R1-VMAX** — inference hard cap `PlanConfig.v_max` + `refav1_arm.py --vmax-from-labels` | `P4S/LANDING_READY.txt` batch R1-VMAX; sources `P4S/code/fix/{stack/tanitad/refs/refa_v1_plan.py, taniteval/tools/refav1_arm.py, stack/tests/test_refav1_vmax_cap.py}` | **NOT LANDED** (tip `refa_v1_plan.py` blob `ba3a4a67…` vs banked `c46ac16b…`; `test_refav1_vmax_cap.py` absent; 0 `v_max`/`vmax` hits in the model/planner/loader/trainer at tip) | landing; then run C′ |
| **C′** | **`A1cap` — R1 full-grid engineering check** (282 windows; `n_plan_over_limit` must be 0) | `P4S/R1_FULLGRID_CHECK.md` (pre-registered 2026-09-27) | **No result at tip** (no hits for `A1cap` beyond the SPEC) — UNVERIFIED whether it ran on the dev box | item C landed; ~2.1 h on the 4060 |
| **D** | Option H — heads-only retrain on the frozen trunk | `REFAV1_RETRAIN_DECISION.md` §2 | **WITHDRAWN** by the trunk probe's committed NEGATIVE branch | — |
| **E** | **First R5 arm** (trajectory head) — needs its own pre-registration: bars vs the chosen prior AND the other two floors at 2 s and 6 s, four families, a **training-seed** replicate (the head is deterministic) | `P4S/refav1_r1r6/RESULT.md` §8.1 item 1 | **No PREREG exists** | B landed |
| **F** | **refav1 launch-gate profile** (binding since 2026-09-26) | `refav1_r1r6/RESULT.md` §7 (proposal); 40-flag inventory in `P4S/raw/` | `stack/scripts/launch_gate.py:432-438` has profiles `refc`, `refcv6`, `refcv7` only — **no refav1** | Master Mind |
| **G** | **Eval harness wiring** for the new levers: an R5 arm calling `RefAV1.trajectory()`, `speed_max_ms` into `plan()`, pose-past κ0 | `refav1_r1r6/RESULT.md` §8.1 item 2 | not started | B, C |
| **H** | **Distance-keeping weight sweep** with inference-seed replicates on the full 141 panel | `AI/2026-09-06-refav1-distance-keeping/RESULT.md` §0 | queued, never run; the DK planner hook is **missing at tip** (`dk_spec` 0 hits in `refa_v1.py`/`refa_v1_plan.py` vs 19 in `refav1_arm.py`) | the hook; the planner is eliminated as a lever anyway (low priority) |
| **I** | Goal-conditioned lateral cost arms A3/A4 (Thor) | `AI/2026-09-05-refav1-close-the-gaps/` | ran, **never read back** (UNVERIFIED) | Thor access; superseded by the full-grid elimination |
| **J** | "REF-A v2" (trainable DINOv3-distilled encoder) | `V7_FULLSCALE_RECOMMENDATION.md:344-350` | never commissioned | would become the fallback if option T's frozen-encoder premise fails (§9 E1) |
| **K** | Closed-loop (AlpaSim/NuRec) read of refav1 | REG:1274 | never run | a policy worth driving |
| **L** | Register integration of the 09-27 verdicts + A13c code provenance | BACKLOG A13c; `P4S/HANDOFF_TO_MASTER_MIND.md` §2/§4 | **not done** — see §8 | Master Mind |

---

## 6. PI requirements R1–R6 (binding, 2026-09-27) — compliance AT TIP

⚠️ The build in §5 B/C would change R1/R3/R5/R6 to "built, default OFF" — **but none of it is in the tip**, so the tip state is:

| req | status at tip | file:line (tip `460508ed`) |
|---|---|---|
| **R1** max speed = input AND hard cap | ⛔ **ABSENT** | 0 hits for `v_max|speed_max|vmax|max_speed` in `stack/tanitad/refs/refa_v1.py`, `refa_v1_plan.py`, `stack/tanitad/data/refav1_loader.py`, `stack/scripts/refa_v1_train.py`; the 3 hits in `taniteval/tools/refav1_arm.py:1691,1735,1783` are a `speed_check` diagnostic, not a cap |
| **R2** nav → tactical + operative, train + inference | ✅ **MET** (as oracle nav) | `refa_v1.py:905` (`nav_inject: True`), `:1561-1564` (`nav_inj_emb`, `nav_to_ctx`, `nav_to_intent`), `:1658-1664` (nav → tactical ctx and → field-predictor intent). Caveats: nav is the label at inference; the route head is a pure nav echo (§4, 09-04) |
| **R3** ALL tactical labels train the tactical layer | 🟡 **PARTIAL** | only the lat/lon decision labels (+ route) reach a loss: `refa_v1.py:1537-1539` (heads), `:2060-2075` (v7.2 label supervision); the 22 goal families and SPEED_BAND do not |
| **R4** tactical choices condition the operative plan | 🟡 **MET IN FORM, DEFECTIVE IN FUNCTION** | goal token → planning cost + FiLM tactical → operative (`refa_v1.py:38-39` design table, `:1650-1666`); MEASURED: the operative cost was goal-blind (3.9× under-turn, §4) |
| **R5** one combined tactical+operative trajectory to 6 s | ⛔ **ABSENT** | plan horizon **2.0 s**: `refa_v1.py:782` (`plan_horizon_s: float = 2.0`), `refa_v1_plan.py:85` (`horizon: int = 10`); the cost spans 6 s but no 6 s trajectory is emitted |
| **R6** strategic layer OFF | ⛔ **ABSENT — needs an architecture change** | the strategic policy is always in the chain (`refa_v1.py:1656`, `strat = self.strategic_policy(...)`), `nav_to_ctx` is sized from `cfg.strategic_cfg.d_ctx` (`:1562`), and `no_hierarchy` drops the tactical layer too (`:2069`) |

---

## 7. Open RESEARCH topics — ranked by measured effect (Rule Zero: largest lever first)

| rank | topic | largest measured effect (class, tier) | what it implies |
|---|---|---|---|
| 1 | **A learned nonlinear kinematic trajectory readout (P10) as the prior and the bar** | 6 s ADE **−0.5024 [−0.6331, −0.3767]** vs the linear readout; 2 s ADE −0.3495 vs `ha0_ext`; 6 s 2.93 vs `damp50` 3.45 (point estimates) (M, T1-equiv.) | the programme's "vision helps" bar for refav1 / v7F / refc heads, and the best residual prior for R5 — ⚠️ but it is **not in the R5 build's prior menu** (§5 A) |
| 2 | **Planner-free damped floors** (`kd_ha0_ext`, `damp50`) | `kd_ha0_ext − ha0_ext` **−0.2282** (282 windows) / **−0.2638 [−0.3374, −0.2005]** (2,399 windows); `damp50 − ha0_ext` −0.1576 (M, T1) | every older refav1 claim "vs `ha0_ext`" must be restated vs `damp50`/`kd_ha0_ext`; a floor only makes tests stricter |
| 3 | **Option T — does a trunk trained WITH the trajectory objective carry information beyond kinematics?** | **UNMEASURED** (H). The negative it must overturn: frozen trunk `P9 − P10` +0.0987 m at 6 s (M) | the only lever that can make refav1's vision matter; pre-register on event strata + 6 s, not on the pooled 2 s mean |
| 4 | **Data scale vs readout capacity on the FROZEN encoder** | untested at 4,572 clips: the trunk probe fit at most 600 episodes, and inner CV picked the **smallest** image capacity (PCA m = 8) every fold (M) | the cheapest discriminator before buying option T (§9 E1) |
| 5 | Longitudinal planner levers (`a0_shift`, jerk seam) | LON speed **−0.2283** (p4, 12× seed floor) and seam **−0.2360** (p4, 8.5×) (M, T1) — but **tied with `damp50` on LON on the full grid** | eliminated as planner levers on this checkpoint; the LON *signal* (accel persistence) is what P10 learns |
| 6 | Lateral cost geometry (`W_KAPPA`, Kamm cap, goal-conditioned cost) | `W_KAPPA` 15.11 vs 0: ADE −0.4338 on p4 (M) — but on the full grid the tuned lateral is **worse than planning straight** (`A1 − shipped` cross +0.1977) | eliminated; keep `kamm_mu` as a safety constraint for any planner |
| 7 | Distance keeping / closing rate | lead gap decodable R² 0.3632 (+0.4145 over pixels); closing rate absent; decoded-gap BAR-2 0.5714 vs 0.80 (M, probe) | a temporal (multi-frame) representation item for option T |
| 8 | Tactical decoder & vocabulary | LON head 0.4326 vs 0.2979 under `nav_zero`; turn AUC 0.88, L/R AUC 0.90; vocabulary cannot express R 100–1000 m curves (M) | R3/R5 replace the vocabulary bottleneck with a regressed trajectory |
| 9 | Strategic layer | route head = nav echo (M); strategic family unmeasurable on the planner surface (n = 0) | R6 says OFF for now — measure strategic only after driving works |

---

## 8. Open ENGINEERING topics

1. **Land B + C** (14 files, banked under `P4S/refav1_r1r6/code/fix/` and `P4S/code/fix/`) — use `stack/scripts/mm_commit.py` / `scoped_commit.py` with the per-path **descendant** check (CLAUDE.md git traps); the build's LF blobs are listed in `refav1_r1r6/RESULT.md` §9.1. Then run `A1cap` (§5 C′).
2. **Add a `kinmlp` R5 prior** (the P10 MLP, frozen) to `refav1_traj.py`, with a known-value test: zero-init head == P10 output on the 2,399 windows, max |Δ| = 0. Confirm which κ0 P10 consumes (UNVERIFIED: if it is the t0 steer channel, D1 applies).
3. **refav1 launch-gate profile** in `stack/scripts/launch_gate.py` (none today; launch gate is binding).
4. **Trainer defects** (`stack/scripts/refa_v1_train.py`): no `summary.json` done-marker (prints `done:` only at `:800`) — the cause of the **80-relaunch** episode (REG:2114-2123); feature standardiser fit on **one batch** (`:581-582`).
5. **Register/registry are stale for refav1** (Rule: update in the same turn): REG §2.4 has no row for `a0_shift`, turn asymmetry, perception probe, `loncomb3` full grid or the trunk probe; REG:2095 says `ccos` "IN PROGRESS" (delivered and refuted); **GC has no row for the 09-27 full-grid FAIL-WORSE or the trunk-probe NEGATIVE** (grep at tip: 0 hits for `trunk-probe`, `damp50`, `FAIL-WORSE`); several GC rows still read QUEUED/RUNNING for landed or abandoned arms; A13c code provenance unrecorded.
6. **Planner instrument defects** (moot while the planner is eliminated, but they will bite the moment a planner sits on an R5 proposal): `plan_source` labels seed winners `cem`; `planner.baseline_won_frac` under-reports (0.6631 vs true 1.0000; tie-break repairs gated on the LABEL, `refa_v1_plan.py:277-281`, `refa_v1.py:2203` per REG — line numbers from the registry, not re-checked); `target_speed` never passed ⇒ `W_VEND` inert while `stack/tests/test_refa_v1_a_sustain.py:148-150` claims otherwise; `cost_fidelity` (design gate G1) has no caller; DK hook absent at tip; `D-REFAV1-SEED-GOAL-MISMATCH` (goal from the 30-step control, seed truncated to 10 steps — 62/64 token pairs disagree).
7. **`v7_labels._MEASURED_GEOMETRY_TOKENS` is module state** set by the last `load_v7_labels` call — loading eval after train in one process flips the negative policy (`refav1_r1r6/RESULT.md` §8.1 item 8; unfixed).
8. **Storage**: train fp8 features deleted (~303 GB); checkpoint has two copies (Thor, dev box), **no HF copy**.
9. **Instruments missing**: training-seed replicate (never run), closed-loop harness for refav1 (never run), strategic family on the planner surface (no route label), refav1 answers to the 7 frozen-trunk questions (`Project Steering/ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md`, BACKLOG A16 — partially answered by the trunk probe's bit-exact encode check, not formally).

---

## 9. Recommended next 3 experiments

Common bar language (from `REFAV1_RETRAIN_DECISION.md` §3, committed before any retrain data): T1, the 141 EVAL episodes, paired episode-cluster bootstrap, **four families**, **2 s AND 6 s**, GT-defined event strata reported beside the pooled mean; floors **`damp50`, `kd_x`/`kd_ha0_ext`, and the nonlinear kinematic readout `P10` (2 s 0.2229 / 6 s 2.9257)**; a lever effect must exceed **2× the training-seed replicate** (and the inference-seed replicate if anything samples).

**E1 — The frozen-encoder ceiling at full data (dev box, unblocked today).**
* *Hypothesis (H):* the trunk-probe NEGATIVE is a data-scale / readout-capacity artefact — a **trainable** spatial-temporal attention readout over the frozen DINOv3 token field (t0 and ≥1 past frame, region-pooled e.g. 8×20), fitted on **all 4,572 TRAIN clips**, beats P10 refit on the same 4,572 clips.
* *Bar (pre-register first):* `readout − P10` ADE @ 6 s < 0, CI excluding 0, at **2 training seeds**, larger than 2× the seed replicate; plus the shuffled-feature and constant controls the probe used; event strata (turn onset, stop, lead braking) reported.
* *Cost:* streamed encode 4,572 clips at ~4.6–4.7 s/clip ≈ **5.8–6 h on the RTX 4060** (E, from M 2,834 s / 600 clips), pooled cache ~16–64 GB on D: (E); readout training a few GPU-h. Pull read-only from Thor's v2ep sources (never add load to a training box — check Thor first).
* *Why first:* it is the cheapest experiment that decides **what option T must train**: POSITIVE ⇒ the adapter/readout was the bottleneck and option T can keep DINOv3 frozen; NEGATIVE ⇒ frozen DINOv3 at 256×640 carries no 6 s information beyond kinematics at our scale, and option T must train the encoder (the "REF-A v2" direction, §5 J) or the vision lever moves to REF-C's trained encoder.
* *Fallback lever if it fails:* E2 with an unfrozen (or LoRA) encoder, priced first.

**E2 — Option T, first arm (Thor or the A40; needs the PI).**
* *Hypothesis (H):* refav1's adapter + WMs trained jointly with the R5 trajectory loss (`--strategic-off --vmax-input --w-goal --w-traj --r5-prior <kinmlp|kdx>`) produce a 6 s trajectory that beats P10.
* *Bar:* `retrained − P10-style kinematic head (same inputs minus vision, same data)` ADE @ 6 s < 0, CI excluding 0, at two training seeds, > 2× the training-seed replicate; R1: **0** plan exceedances of the clip limit on every window; R6: no strategic parameter updated (bit-identical before/after).
* *Cost:* one epoch ≈ **20 h on Thor** (M: 3.47 s/step × 21,109 for the ep3 recipe; the R5 head's extra cost UNMEASURED) **× 2 seeds**; plus either ~303 GB of fp8 features placed, or on-the-fly DINOv3 encode whose per-step cost must be **measured before launch**.
* *Unblocks:* §5 B + C landed, a refav1 launch-gate PASS, PI decisions D1–D5 and storage/compute, and Thor free (it was training refcv8 at last record — I).
* *Fallback:* if E1 was positive but E2 fails, ship a frozen-encoder R5 head (E1's readout) as refav1's trajectory; if both fail, retire refav1 as a WM-planner reference and record the negative with its scope (one checkpoint, frozen DINOv3, open loop).

**E3 — The kinematic prior inside refav1 (dev box, ~0 GPU, unblocked after landing).**
* *Hypothesis (M-anchored):* swapping the R5 prior from `kdx` to the frozen P10 MLP lowers every R5 arm's floor by ~0.5 m at 6 s (the measured P10 − linear gap), so E2's head only has to learn the vision residual.
* *Bar:* known value first — zero-init R5 head == P10 output on the 2,399 windows, **max |Δ| = 0.0**; then `kinmlp-prior − kdx-prior` (zero-init) ADE @ 6 s reproduces the trunk probe's −0.50 within its CI (an identity check, not a lever claim).
* *Cost:* code + CPU (the P10 weights are re-fit in minutes from the banked `code/probe_te.py`; the TRAIN features must be regenerated — `C:/Users/Admin/refav1_trainprobe/` is gone).
* *Unblocks:* §5 B landed; PI D1 if P10 uses the steer channel at t0.
* *Fallback:* `--r5-prior damp50` (best 6 s FDE of the three built priors: 10.150 vs `kdx` 11.216 — exploratory, M, `refav1_r1r6/RESULT.md` §6).

⚠️ None of E1–E3 is "tune the planner again": the full grid eliminated every zero-training planner lever on 21,109. A planner returns only after a trained trajectory exists, as a re-ranker over its proposals.

---

## 10. Traps specific to this arm (each has already cost time)

1. **Three grids, never mixed** — p4 (40 windows / 8 eps, turn-dense), full (282 / 141), trunk-probe (2,399 / 141). p4 tuning bought a lateral behaviour the lane-keep-dominated full grid punishes (247/282 windows are `lane_keep`).
2. **`ha0_ext` is no longer the strongest floor** — `damp50` and `kd_ha0_ext` beat it, separated. Restate any "beats `ha0_ext`" claim.
3. **The inference-seed floor is config-dependent**: 0.0607 m ADE (p4, `ccos`), +0.0619 (full grid, `loncomb3`), +0.0015 (`best`) to +0.1035 (`combined`) on p4. The "≈0.30 m" figure in CLAUDE.md/REG:3035/3321 comes from `D-REFAV1-DRIVE-AB` (+0.2956 on the *unchanged* windows of one panel, `AI/2026-09-05-refav1-make-it-drive/AB_RESULT.md:92-94`), **not** from `D-REFAV1-SEED-GOAL-MISMATCH` as CLAUDE.md attributes it. Measure the replicate on the arm you claim.
4. **Read the plan's CONTENT, not its label** — `planner.baseline_won_frac` read 0.6631 while the true trivial-plan rate was 1.0000; `plan_source` says `cem` for an injected baseline.
5. **Units**: `actions[:,0]` is a steer angle (×2.9 vs κ); the banked `ol`/`ha` arms ran `--action-units kappa` (`REFAV1_ARM.md` §3b).
6. **Retracted figure still in a 09-27 RESULT**: `loncomb3` RESULT §4.5 quotes "the world model's lateral contribution is 1.63e-10" — that is the **step-1,000** checkpoint's number (RETRACTION_LOG #21); at 21,109 the median is **1.99e-06** and the binding penalty under the shipped cost is **jerk** (REG:2144-2150). The registry wins.
7. **Price the file the consumer opens** — the trainer reads the **~303 GB fp8 feature cache** (4,572 × ~66.2 MB), not the ~155 GB v2ep source cache (the trunk-probe SPEC quoted the latter; corrected in its RESULT §5).
8. **Thor's `/home/nvidia/TanitAD/stack` is stale** (HEAD `30d6d60`, no `refav1_arm.py`) — use a self-contained shipped tree and assert `tanitad.__file__` (`run_fullgrid.sh` does this with a `ZZ…ZZ` marker).
9. **The checkpoint was trained from Thor's 1,885-line `refa_v1.py`** — re-running from tip is not a replication.
10. **Done-marker trap**: `refa_v1_train.py` writes no `summary.json`, so the supervisor relaunched the finished run 80 times and left a `DRIFT_ALARM` red herring; the log's last row (21,100) is not the final step (21,109) — the checkpoint's `step` field is authoritative.
11. **Dev-box commit limit, not RAM**: the trunk probe died on a 63 MiB allocation with 11 GB physical free (Windows commit 86.6 GB exhausted by a concurrent 23 GB test suite) — and its wrapper reported exit 0. Make long jobs resumable per arm and assert on the artifact, not the exit code.
12. **The full training step does not fit the 8 GB 4060** at any batch or precision — training is Thor/A40 only; the 4060 runs evals (~2.1 h/arm, full grid) and probes.
13. **`C:/Users/Admin/refav1_probe/` holds `*_LEAKY_*DO_NOT_READ*` smoke outputs** — never read them as results.
14. **Name collision**: "REF-A v1" in `V7_FULLSCALE_RECOMMENDATION.md` (2026-08-24) is the old DINOv2 arm; refav1 is the DINOv3 WM-planner. The directory `refav1-b1-v72-1ep-21109` holds a **step-1,000** checkpoint.
15. **`/c/lgs/tip_files.txt` is incomplete** (it omits e.g. `taniteval/tools/refav1_arm.py`, which `git cat-file -e` proves present) — the `ls-tree` truncation trap; assert paths positively.

---

## 11. File index

| path (at tip unless marked) | why |
|---|---|
| `Project Steering/MODEL_REGISTRY.md` §2 (lines 1967-2173) | REF-A lineage and refav1 §2.4 (stale after 09-06) |
| `AI/2026-09-27-refav1-trunk-probe/RESULT.md` (+ `SPEC.md`, `code/`, `raw/out_te/`) | ⭐ the frozen-trunk NEGATIVE and the P10 bar |
| `AI/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md` (+ `SPEC.md`, `raw/run/run_fullgrid.sh`, `raw/analysis/pd_fullgrid*.md`) | ⭐ planner eliminated; exact dev-box invocation |
| `P4S/REFAV1_RETRAIN_DECISION.md` | option T vs H, bars, the PI ask |
| `P4S/refav1_r1r6/RESULT.md` (+ `code/fix/…`) | R1/R3/R5/R6 build, landing table, D1–D5 |
| `P4S/R1_FULLGRID_CHECK.md`, `P4S/LANDING_READY.txt`, `P4S/code/fix/…` | R1-VMAX batch + the `A1cap` pre-registration |
| `P4S/STATE_AND_GAPS.md`, `P4S/HANDOFF_TO_MASTER_MIND.md`, `P4S/raw/SURVEY_REFAV1.md` | 2026-09-27 state, gaps, contradictions (survey = I) |
| `taniteval/results/RESULT-refav1-21109-openloop.md` (+ `refav1-21109-openloop*.json`) | the epoch-end four-family read |
| `taniteval/tools/refav1_arm.py`, `taniteval/tools/REFAV1_ARM.md` | the eval adapter and its Thor invocation |
| `taniteval/tools/refav1_paired_delta.py` | the paired four-family delta tool |
| `stack/tanitad/refs/refa_v1.py` | model, config, cost constants, `plan()` (`:2393`) |
| `stack/tanitad/refs/refa_v1_plan.py` | iCEM, `PlanConfig`, `kamm_mu` |
| `stack/tanitad/refs/refav1_lon_cost.py` | longitudinal cost levers |
| `stack/tanitad/data/refav1_loader.py` | windows, labels, nav join |
| `stack/scripts/refa_v1_train.py` | trainer (no done-marker; one-batch standardiser) |
| `stack/scripts/dinov3_fp8_encode_ship.py` | the fp8 encoder/shipper (needed for option T) |
| `stack/tanitad/models/kinematic_prior.py` | the refcv-line kinematic priors (`ha0_ext_pose`); refav1's `kd_x` lives in the trunk-probe `code/` until the R5 build lands |
| `stack/scripts/launch_gate.py` | binding gate — no refav1 profile yet |
| `stack/tests/test_refa_v1*.py`, `stack/tests/test_refav1_*.py` | ~45 pinning tests |
| `AI/2026-09-05-refav1-cost-geometry/RESULT.md` (+ `HANDOFF.md`, `PREREG_COST_GEOMETRY.md`) | W_KAPPA optimum, Kamm cap, nulls |
| `AI/2026-09-05-refav1-make-it-drive/RESULT.md`, `AB_RESULT.md` | two gates; the 11.2× seed-ratio A/B |
| `AI/2026-09-05-refav1-goal-margin/RESULT.md` | the vocabulary diagnosis |
| `AI/2026-09-05-refav1-lonshift-t1/RESULT.md` | `a0_shift`, two-GPU replication |
| `AI/2026-09-05-refav1-close-the-gaps/RESULT.md`, `…/raw/refav1_margin_banked_2026-09-27/RESULT_HARVEST.md` | goal-blind cost; `kammshift` FAIL; `loncomb3` on p4 |
| `AI/2026-09-06-refav1-perception-probe/RESULT.md`, `…-decoded-gap/RESULT.md`, `…-distance-keeping/RESULT.md` | lead gap / closing rate / DK |
| `AI/2026-09-04-refav1-cost-scale/RESULT.md` | the cost in units |
| `AI/2026-09-05-refav1-ccos-eval/RESULT.md` | `ccos` on the full grid |
| `AI/2026-08-18-refa-v1/REFA_V1_DESIGN.md` | design rationale and gates G1–G5 |
| `Project Steering/ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md` | the 7 questions owed for refav1 |
| `Project Steering/BACKLOG.md` (A13c, A16, refav1 pull-list from line 119) | open small items |
| `Project Steering/RETRACTION_LOG.md` (#15, #21, 2026-09-03 entries) | refav1 retractions |
| dev box: `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/{ckpt.pt,config.json}` | the checkpoint + its training record |
| dev box: `C:/Users/Admin/tanitad-data/refav1-eval141/` | eval fp8 cache + episodes |
| Thor: `/home/nvidia/experiments/refav1-b1-v72-ep3-speed/`, `/home/nvidia/data/refav1-fp8-eval` | canonical run dir + eval cache |
