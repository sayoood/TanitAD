# v8F / v8F-mini status digest, 2026-11-12

Sources: mirror tip `origin/agent/arch-inf-20260803`; code at `3e264bbf` (`stack/tanitad/models/v8f_mini.py`, abbreviated m.py); PAPER-v3 = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-10-paper-pdf-v3/tanitad_4b_paper.tex`; GAC = `Project Steering/GOALS_AND_CLAIMS.md`; REG = `MODEL_REGISTRY.md` section 17; LS = `dagland/LOOP_STATE_2026-11-02.md`; RAW = `...2026-10-17-v8f-mini/raw/`. Tiers: T0 = own outputs / teacher-forced on logged frames; T1-kin = self-action open loop on logged frames (own executed motion, causal speed bucket, logged scene). Neither is driving performance; no T2 result decides anything. Strategic family: PAUSED (PI item 25) throughout. All numbers MEASURED unless marked. Two values are "s0 / s1" (training seeds).

**Naming.** "v8F-mini" is the only implemented arm (M1, M2b, adopted base DAG-OACC). A larger flagship v8F as a built model: NOT FOUND (probed REG grep "v8F", and the `stack/tanitad/models/` listing: v8f_mini.py, v8f_beta.py, v8f_lora_encoder.py only).

## 1. Design
| module | trainable params (REG, M1 / M2b) | inputs | output, horizon | loss | code |
|---|---|---|---|---|---|
| DINOv3-L/16 features (frozen), pooled 4x10 tokens x 1024 | frozen | front-wide cylindrical 256x640, 4 frames, 0.2 s grid | 40 tokens/frame | none | cache |
| shared adapter (LayerNorm, Linear 1024 to 256, pos, time) | 275,712 | tokens | scene tokens d=256 | all | m.py:~725 |
| S reader (depth 1, 4 queries, 4 heads, ff 512) | 1,321,737 | scene, ego f8 (yaw-rate based), nav, limit | route checkpoints 3x3 at 30/50/80 m; ~1 Hz | w_ckpt 1.0 | `read_s`, `head_s` |
| T reader (depth 1) | 1,463,868 | scene, ego, nav, limit, held S checkpoints (odometry re-expressed) | typed code, 60-d: 22 goal sigmoids, 5 lat + 7 lon softmaxes, 26 constraints; ~2 Hz; no trajectory head | goal BCE + 0.5(lat CE + lon CE) + 0.5 constraint Huber (partial-label CE) | `run_t`:845 |
| O reader (depth 2) | 2,120,216 | scene, ego, nav, limit, held S ck, held T code, ages | 12 cosine-basis coefficients x 2 axes, 60 steps (6 s at 10 Hz); 10 Hz; last layer zero-init | w_traj x (ADE/3 + 0.5 FDE/6); optional accel term | `run_o`:860 |
| frozen P10 prior (kinematic MLP) | frozen | f8, base | 30x2 path upsampled; plan = up(P10) + B c_res (ramp, scale 2 m) | none | `P10Prior`:310 |
| H-JEPA WM aux (M2b only) | +2,007,874 (totals 5,206,621 / 7,214,495) | adapter tokens, actions | z1/z2/z3 at 0.2/0.4/0.8 s, IDM gamma 100 | w_wm 0.1 | `WMAux`:672 |
| rule penalties (SPEED/STOP/TURN/LEAD hinges) | 0 | plan | vs GT w 0.5, vs own constraints w 0.25 | | `consistency_penalties`:555 |

* Levels exchange only held outputs (sample-and-hold, ages a_T 0..2, a_S up to 6). Scheduled teacher forcing p_gt 1.0 to 0.2 (`p_gt_schedule`:1203). Inference reads a 14-key allow-list through `_KeyGuard` (`forward_infer`:951).
* **Inference selection: none.** One deterministic pass, one plan. No sampling, iCEM or scorer. Top-3 and best-of-3 are oracle diagnostics. Zero-training add-ons: HACC post-map `P + w(O-P)` (buffer `hacc_w`, absent by default, :961) and speed/Kamm projection `project_speed` (:494, called from `v8f_mini_eval.py:347`, not from the model).
* The WM is an auxiliary on the adapter; O reads no WM rollout. z1 is the pooled vision token and measured ego state does not enter the WM latent (GAC H-V8F-PLANTEST3-GATE).

## 2. Implementation, reachability, gate
* Reached from `v8f_mini_train.py`: arms M1/M2b, DAgger (`--dagger-p`), `--w-acc`, `--detach-upper`, `--code-mix`, `--o-interface`, `--wm`, `--goal-pos-weight-stop`, `--lon-class-weight-stop`, `--launch-intent-w`, `--level-head`, `--lead-head`, `--stop-aux` / `--w-stop-aux` / `--stop-aux-extr`. `--wm-basis` exists only on branch commit `f7e790ab` (SLOWWM), not at 3e264bbf.
* **Defaults that leave a lever off (must be in argv):** `--w-acc 0.0`, `--launch-intent-w 0.0`, both stop weights 1.0, `--level-head off`, `--lead-head off`, `--stop-aux off`, `--w-stop-aux 0.0`, `--code-mix whole`, `--o-interface full`, `--detach-upper` unset, `hacc_w` absent.
* `built_levers()` (:1142) reports what was built; the trainer refuses declared-but-unbuilt levers. Launch gate (`launch_gate.py`, `launch_gate_arms.py`): v8f profile pins the P10 prior sha256 (`e70a9ae3...`, steer `af5a29dd...`) with regression arms (missing, zero-weight, trainable prior). Every banked run has `gate_token.json` and `selfcheck.json` (declared_vs_built PASS, no dead params).
* Stop-aux gate items open on 11-12: extr table coverage >= 0.99 (built 4,572/4,572), real G-LIVE probe needs an in-frame rest point (LS 11:20). Tests: 38 stop-aux, 171 across six v8F suites pass (LS). Arm A has not trained.

## 3. Training
| item | value |
|---|---|
| corpus | PhysicalAI-AV, box-2 `D:\tanitad\cache_full`; 4,512 train clips (`--exclude 540:600` of 4,572), 306,566 windows (REG). Not the parity corpus `physicalai-train-e438721ae894` (REG). Hours: NOT FOUND (probed PAPER-v3, REG) |
| pretrain | M1/M2b, 2 seeds, 19,164 steps (steps-mult 4), batch 64, lr end 3e-5, ~3.9 h / 4.4 h per run on box 2 (GPU model NOT FOUND) |
| DAG-OACC | fine-tune from M2b: 3,000 steps, lr 1e-4, warmup 100, p_gt 0.2, DAgger p 0.1, `--w-acc 49.6969` (INHERITED: BAL-NULL argv, `C:/Users/Admin/t2v8f_pulls/stopbal_BAL-NULL_s0/argv.txt`) |
| OTBAND / OBD | 9,582 steps, `--freeze S adapter cond`, `--detach-upper` |
| SLOWWM / ladder | ~0.49 s/step, ~3.35 h per run (LS) |
| eval | eval-141: 141 episodes, 2,399 windows (first-tick reads 139 episodes / 2,365 windows) |
| determinism | box-2 same-seed replicate is bit-identical (so replicates change the seed); dev-box dumps drift ~mm with thread count (rolled track 3.9 mm) |

## 4. Verification
Estimator: paired episode-cluster bootstrap, B 2,000 ("another draw of episodes" only). Baseline P10-yaw-4512 6 s ADE 2.4870 m (T0, REG, `RAW/full/eval/verdict.json`). Seed floors F_M2b 0.1207, F_M1 0.2222 m (H-V8F-SEED-1).
| result (s0 / s1) | value | tier |
|---|---|---|
| M2b 6 s ADE minus P10 | -0.312 / -0.432 m (bar 2F 0.2415) | T0 |
| M2b + HACC: 2 s ADE -0.011/-0.008; 6 s -0.50/-0.47; accel -0.022/-0.040 m/s2; better on every pre-registered row | T0 |
| HACC on own motion, 6 s ADE minus P10 | -0.83 / -0.79 m (past-only bucket -0.77/-0.73; frozen camera -0.73/-0.64) | T1-kin |
| DAgger p 0.1: lat macro-F1 / lon macro-F1 | +0.108/+0.118 / +0.055/+0.081; 6 s ADE -0.54/-0.40 m vs control | T1-kin |
| **DAG-OACC base, 6 s ADE minus P10** | **-1.26 / -1.19 m** | T1-kin |
| per band: DAgger rows 2-6 s ADE -0.82/-0.62; accel term 0-2 s ADE -0.062/-0.079, 0-2 s accel error -0.97/-1.14 m/s2 | T1-kin |
| OB-BAND minus OB-CTL, operative ADE | -0.0168 / -0.0149 m (floor 0.0074); heading +0.031 deg (worse) | T1-kin |
| top-3 tactical contains logged manoeuvre | 90 % (top-1 57-59 %); oracle best-of-3 -0.35..-0.39 m | T1-kin |
| GT tactical code substituted: 6 s ADE | ~2.1 to ~1.0 m | T0 |
Sources: PAPER-v3 Table 1 and Fig. 1; `RAW/2026-11-09-v8f-otband/read/RESULT_FULL.md`. Per-family rows (longitudinal speed/accel/along-track/headway/time-gap; lateral cross-track/heading/yaw-rate/curvature; tactical macro-F1/accuracy) sit in each RESULT json. A separated CI is necessary, not sufficient (H-ESTIM-SEED-1); replicate floors arbitrate.

## 5. What it demonstrably does not do (since 11-01)
| item | measured | bar / floor | source |
|---|---|---|---|
| band loss on tactical 2-6 s | ADE -0.024 / -0.060 m, not separated | tactical replicate floor 0.287 m; GAC outcome BAND-COST | OTBAND RESULT_FULL |
| T stops when it should (p >= 0.5, STOP_POINT windows) | hit 0.103 / 0.128; 0.94 of speed error in misses | DECISION-SEES >= 0.6 | `RAW/2026-11-10-v8f-stopdec/VERDICT_STOPDEC.md` |
| lon decision in GT stop regime | ACCELERATE in 63 / 56 of 132 windows | | same |
| stop-cue AUROC | 0.836 [0.784,0.886] / 0.857 [0.809,0.903]; seeds disagree | CUE-PRESENT >= 0.85 | `RAW/2026-11-10-v8f-stopcue/VERDICT_STOPCUE.md` |
| class weighting, doses 1 and 2 | hit 0.45-0.62, FPR 0.12-0.145 (was 0.005-0.013); AUROC moves -0.006..+0.020; stop-point ADE not improved | hit >= 0.30 and FPR <= 0.05 | `RAW/2026-11-10-v8f-stopbal/dose2/read/VERDICT_DOSE2.md` |
| STOPPERC S0-P probes | V40+E minus T +0.032 / +0.004; RES-ADDS -0.006 (linear) | +0.037 / +0.02, CI > 0 | `RAW/2026-11-11-v8f-stopperc/VERDICT_S0P.md` (FLAT) |
| launch-crawl share | the one unmet row of 19; external progress floor closes ~half even with exact speed | | PAPER-v3 |
| speed / yaw-rate readout from L1 latent | R2 0.271/0.314 and -0.122/0.268 | 0.70 / 0.50 | GAC H-V8F-PLANTEST3-GATE |
| WM at 0.2 s vs "nothing changes" | SKILL +0.0010 / -0.0010 not separated; at 2 s -0.150 / -0.172 | PREDICTS not met | GAC H-V8F-WMACT-1 |
| measured speed in WM (EGOWM) | +0.007..+0.013 | floor 0.0124 | GAC H-V8F-EGOWM-1 |
| T from logged state (TSWAP) on crawl share | +0.0031 / -0.0037 | not separated | GAC H-V8F-TSWAP-1 |
| T2 first read, 48 scenes | adapted encoder: 6 s ADE not separated from unadapted | | PAPER-v3 |
Disagreement: GAC H-V8F-SLOWWM-1 reads "PRE-REGISTERED, NOT RUN", yet `t2v8f_pulls/slowwm_{SLOW,PCA-B}_s{0,1}` are banked (11-12) and LS says SLOW_rep was the last run; the read is not done. GAC labels OTBAND BAND-COST; PAPER-v3 words the same numbers as an operative-band gain.

## 6. Gap list, ranked by measured effect
1. **Tactical constraints, speed kind first.** DAG-OACC tactical GT-ALL minus TOP1 = -1.625 / -1.773 m; constraints 0.872 / 0.892 of it, decision 0.111 / 0.094; speed +2.7 m (T1-kin first tick, GAC H-V8F-CONSORC-1). Interface headroom ~1.1 m at 6 s (T0).
2. **Stop decision.** Forcing GT stop on 70 % of stop windows gives -0.96 / -0.86 m tactical ADE (`RAW/2026-11-11-v8f-stopperc/VERDICT_S0.md`); T covers 6 % / 12 %. Weighting is ruled out (one ROC); frozen vision adds nothing measurable. Arm A: PI go 11-12 20:30, not run.
3. **Run-to-run variance.** Tactical replicate floor 0.287 m; seed floors 0.12-0.22 m; most 2-6 s levers fall below.
4. **Launch/crawl and speed level.** Vision-only level head: R2 0.780, band MAE 1.90 / 1.83 m/s, bias +1.01 / +0.65 (T0, `RAW/2026-11-05-levelhead-cl-readout/RESULT.md`).
5. **Manoeuvre selection.** Oracle selector captures 0.239 / 0.228 of best-of-3 (GAC H-V8F-TOPK-PAIR-1).
6. **Tier.** Every capability number is T0 or T1-kin on logged frames; no T2 verdict.
7. **WM use.** Aux only; planner gates DECODER-WEAK.
8. Lateral: small costs (heading +0.03 deg; forced-stop yaw +2.2 deg/s).

## 7. Non-ego world understanding
| evidence | result | tier / path |
|---|---|---|
| LATSEM2B, agents within 30 m | R2 +0.357 [+0.191,+0.478] / +0.329 [+0.122,+0.485] at L1 only (eval-141 +0.308/+0.231) | T0, `RAW/2026-11-07-v8f-latsem2b-result/RESULT_LATSEM2B.md` |
| same, lead gap | encoded at no level; lead presence, 7-class lon label, turn ahead encoded at every level; no scene imagining | T0 |
| LISTEN swaps | nav reaches the trajectory through T | T1-class, `RAW/2026-10-25-listen/READ.json` |
| stop cue | T AUROC above; probe on T's non-vision inputs alone 0.8525, with vision 0.8615; `limit` removal -0.04 inside CI | T1-kin first tick, VERDICT_S0P / S0 |
| vision-only level head | R2 0.780 [0.685,0.845] / 0.780 [0.692,0.842] | T0 |
Traffic-light, stop-line or lead-vehicle AUROC probes on v8F latents: NOT FOUND (probed GAC grep "V8F" with traffic light / stop line; RAW folder listing). The model's only traffic-light-related target is the STOP_POINT goal bit.

**Labels and where they live.** v7.2 label release: 4,572 clips, 779 with a GT traffic-light goal and colour (CLAUDE.md, MEASURED 2026-09-06). v9 labels: private HF dataset `Sayood/tanitad-refcv8-v9-labels`, rev `130c0b94`, `release/v9_labels_train.npz` md5 `f63ece41...`; box-2 trainer reads `D:\tanitad\data\v9\v9_labels_train.npz` md5 `83e30bd3...` (two md5s: disagreement, unresolved). Lead table (obstacle.offline join) `D:\tanitad\data\v9\lead_train4572.npz`; speed `refcv6_speed_max_v8_train.jsonl`; stop-aux extrinsics `stopaux_extr_join4572.json` (4,572/4,572, sha `77b6e8dc...`, banked in `732c5970`, dev copy under `C:/Users/Admin/stopaux_extr/`). Context labels: `Sayood/tanitad-context-labels-v1` (rev `100f03da`), `Sayood/tanitad-band-labels-v10` (`211730b4`), `Sayood/tanitad-context-parts-v1` (5,659 parts). `stop_hit` = STOP_POINT goal in T's goal vector. Corpus: `Sayood/tanitad-physicalai-w120-256x640cyl`; plantest runs `Sayood/tanitad-v8f-plantest3-runs`. Box-2 cache: 4,572 files, ~20.9 MB each (grids 4x10 and 8x20). Label file sizes: NOT FOUND (probed `stopaux_extr/` listing, GAC).

## 8. Inventory
`C:/Users/Admin/t2v8f_pulls/` (5.3 GB; each dir: argv.txt, ckpt_last.pt, selfcheck.json, summary.json, train_log.jsonl, BANK_MANIFEST.json with sha256).
| group | dirs | ckpt size |
|---|---|---|
| stopbal_BAL-{NULL_s0,NULL_s1,NULL_rep,G,G2,L,L2}_s* | DAG-OACC-recipe fine-tunes | 86,971,285 B (NULL_s0, sha `5169ea95...2fdf`) |
| otband_OB-{CTL,BAND}_*, obd_OBD-* | band-loss fine-tunes | ~87 MB (not each measured) |
| ladder_{C0,R35,R35b,R35H,S35}_*, egowm_*, slowwm_{SLOW,PCA-B}_s0/s1 | WM-only arms | 46,847,094 B (R35H_s0) |
| stage3/4/5_*, renc*, wsplora, gap48 | T2 / LoRA / re-encoder | not measured |
M2b / M1 originals: box 2 `D:\tanitad\runs\v8f_mini_full\<run>\ckpt_last.pt` (REG manifest `RAW/full/MANIFEST.md`). HACC-baked M2b: `E:/Projects/TanitAD-artifacts/v8f-mini-hacc/M2b_s{0,1}_hacc.pt`, 86,957,898 B, sha `71de3c61...` / `32e4809d...`.
Dumps: per-read npz (STOPDEC `stopdec_PARITY_DAGOACC_s{0,1}.npz`: clip_index, ws, eid, has, v0, err_mps, goal_prob, lon_prob, lon_argmax; `stopdec_read.py:240`); T1 `t1v8f_<tag>.json`; OTBAND `read/DUMPS_*.json` with private HF tars (sha256 verified).
Tools (dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`): `v8f_mini_eval.py` (T0), `v8f_t1_eval.py` (T1-kin), `v8f_paired_families.py`, `v8f_mini_eval_{oacc,speedread,speedcal,speedavg,kindx}.py`, `v8f_kscore.py`; per-card readers (`stopdec_read.py`, `report_slowwm.py`, `read_otband.py`). Cost: STOPDEC full read ~5 min (MEASURED); SLOWWM planner read needs >= 8 GB RAM and 6 GB on C:; other CPU hours NOT FOUND. Training: box-2 GPU only (pretrain 3.9-4.4 h MEASURED; 3,000-step fine-tune ~0.5-1 h ESTIMATED).
Open at 11-12 23:30: arm A chain waits on SLOWWM, then NULL_s0, NULL_rep2, A_s0, A_s1 (~2.8-4.1 box-2 h ESTIMATED, LS).
