# RESULT — refcv7 four-family milestone battery (EvalFlyWheel)

**Pre-registration:** `SPEC.md` (sha256 at registration in `raw/SPEC_SHA256_AT_REGISTRATION.txt`, written 05:29 Berlin BEFORE any refcv7 forward; AMENDMENT A1 05:39, before any refcv7 number: surface M = the S2 windows).
**Status of this file:** the pipeline-validation record (step 1,500) and the armed milestone. The step-5,000 section is APPENDED here by `code/chain_milestone.sh` when it finishes (its full artifacts land under `raw/step5000/`).

> ⛔ **Everything computed on the step-1,500 checkpoint is PIPELINE VALIDATION ONLY — not a result** (SPEC §6). No bar is evaluated on it.

## 1. The kit (MEASURED)
* refcv7 config `D:/refcv7_eval_kit/ckpt/config.json` md5 `e6512a01…` == Thor; launch `fec3a0dc`; the `ev7` code tree (`0c444082`) differs from the launch commit in 2 Training-Watch files only (`git diff --raw --abbrev=40`).
* Data: **290/290** eval-side inputs md5-identical to Thor (`D:/refcv7_eval_kit/kit_checks/eval_inputs_md5_compare.json`), so the refcv6 kit's data root is reused; nothing pulled or copied twice.
* Validation checkpoint: rolling `ckpt.pt` at step 1,500, md5 `c35966f799b1c724515ba1281736fd48` (Thor before == after == local; step key 1,500 read locally).
* Milestone file (from source): `ckpt_5000.pt` = `{model, step}`, written by `MILESTONES = (5000, 15000, 20000, 30000)` (`refc_v3_train.py:225`, `:9645-9647`) after the rolling `ckpt.pt` and before the step's eval row; never rotated. The rolling `ckpt.pt` is the fallback in [5,000, 5,500).

## 2. G0 on the validation checkpoint (step 1,500) — PIPELINE VALIDATION ONLY

Source: `raw/g0_step1500/g0.json` (8 inference seeds × the trainer's 128 in-run windows in its 8 fixed batches, micro-batched `2,2,3,3,3,3`; 1,396 s; peak 5.86 GiB), re-judged with zero GPU by `code/g0_rejudge.py` after SPEC AMENDMENT A2 was registered (`raw/g0_step1500/g0_rejudge.log`).

| verdict | value |
|---|---|
| **G0 as registered** | **FAIL** — exactly ONE term out: `eval_box3d_det_ap4_stroller_0_20` in-run 0.11111 vs 0.14286 on all 8 seeds (sd 0); its support is **n_pos = 1** |
| **G0-A2** (the milestone gate, SPEC A2) | **PASS** — 0 reasons; 238 low-support detection keys REPORTED (n < 30), 96 gating |
| strict load / `param_breakdown` / loader G-DVB | 0 missing, 0 unexpected / equal / 0 mismatches (221 registry entries) |
| SMOOTH (179) | median rel dev **3.6e-5** (bar 0.2 %); worst `eval_lon_tac` 0.17 % |
| STOCHASTIC (6) | all inside the 8-seed 99 % PI (`loss`, `traj`, `cascade`, `law`, `sel_v3`, `goal_score_absmean`) |
| MATCHED (29) / MAP10 counts (160) / MAP10 IoU (80) | median 4.6e-4 / 4.4e-5 / all inside abs 0.01 |
| DETECTION gating (96) | median abs dev 4.5e-5 (bar 0.005) |
| COUNT (210) / null keys (98) | exact / null on both sides |
| wrapper clause (P3 `fp32_det`) | **PASS**: max wrapper rel 6.4e-6 vs floor 1.4e-7 → bar 1e-5 held on every term; W1 and W2 both detected. (P1 as-run 2.8e-3 and P2 1.5e-3 are batch-composition numerics, reported, not gating — the refcv6 A2 finding reproduced) |
| **M1** trunk equalisation dropped (the FIX-3 defect; must FAIL) | **detected**, 1 term: `eval_lon_tac` 1.68761 → 1.64529 (rel 2.5 %). ⚠️ Thin power at step 1,500; the equalise call counter read 0 during the arm, so the mutation did take effect |
| M2 map rule `prior_corrected` → `raw` | detected, 40 terms |
| M4 residual prior OFF | detected, 4 terms (`anchor_acc`, `sel_v3`, `law`, `goal_score_absmean`) |
| requires_grad control (trained flags vs the loader's all-False, 2 windows, eps 0) | 23 of 444 terms differ; the 20 the artifact records (`differ_first20`) are all at **≤ 1.4e-6 relative** (slot heads, traj, cascade, law); the other 3 were not recorded (UNVERIFIED). The battery rolls in the loader's state; the difference is stated beside it |

⚠️ **Judge defect found and fixed (SPEC A2 item 2):** the mutation count first included terms already OUT under the unmutated reproduction (M1 read "2 terms", one being the stroller key). Both verdicts now count only terms a mutation MOVES outside.

## 3. VOID gates 5 and 6 (MEASURED, `raw/g0_step1500/void_gates_map.json`)
* **Gate 5** — the `/3` GT inside the old 60 m × ±16 m window equals the `/2` `fine_codes` byte for byte on **137/137 clips, 27,540 frames** (anchored column offset 140); the refcv6 0.5 m → 10 cm hook literal (one class-5 cell → exactly its 25 fine cells, `NO_PREDICTION` everywhere outside the old window, 1000 × 600) PASSES.
* **Gate 6** — perception is seed-invariant: **603/603** MAP10 / DETECTION / MATCHED terms have rel spread exactly 0 over G0's 8 inference seeds (the heads run before the DDIM planner), so ONE perception pass per checkpoint suffices.

## 4. The armed milestone (step 5,000)
* **Waiter:** `code/chain_step5000.sh wait` → `code/waiter_milestone.sh 5000`, launched detached 05:54:52 Berlin (MSYS pid 134504 / Windows pid 40712). Log `D:/refcv7_eval_kit/chain/waiter_5000.log`; pull log `D:/refcv7_eval_kit/chain/pull_5000.log`.
* **It waits for the EVENT:** Thor's `ckpt_5000.pt` present, size/mtime stable ≥ 20 s, and the step-5,000 eval row in `metrics.jsonl` (the trainer saves before that row). It then pulls read-only (Thor md5 before == after == local), reads the `step` key LOCALLY and REFUSES anything but 5,000. Fallback: the rolling `ckpt.pt` while the last step is in [5,000, 5,500). Poll 5 min; 1 min once the file exists.
* **Then** `code/chain_milestone.sh 5000`: GPU lock (held throughout) → G0 (8 seeds, M1/M2/M4, wrapper; STOP unless G0-A2 = PASS) → T1 rolls at inference seeds 0 and 1 on S2 → panels, four families, BAR-R7-1/2/3 → VOID gates 5/6 → perception pass (map 10 cm + box; refcv7 vs refcv6@38k vs positional prior) → BAR-M7-1..4, BAR-B7-1/2 → interim RESULT → refcv6@38k S2 rolls if the pre-stage has not banked them → re-read bars → `RESULT_step5000.json` + section appended here + a new LANDING_READY heading.
* **By hand** (if the waiter died): `sh D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/chain_step5000.sh wait` (or `... run` once `D:/refcv7_eval_kit/ckpt/ckpt_5000.pt` and `D:/refcv7_eval_kit/chain/pull_5000_OK.json` exist).

## 5. Pipeline smoke on the step-1,500 checkpoint (PIPELINE VALIDATION ONLY — no number here is a result)
Banked sanitized under `raw/smoke_step1500/` (work dirs on the dev box: `D:/refcv7_eval_kit/smoke/`, `D:/refcv7_eval_kit/battery/smoke_step1500/`). Every stage of the milestone chain ran once, 06:23–06:58 Berlin, under one hold of the GPU lock:
1. **refcv6@38k perception dump** (82c2331 tree, the refcv6 package's loader), 40 S2 windows: 40/40 maps (9-channel 0.5 m logits), 40/40 box3d slot sets (100 queries).
2. **refcv7 perception pass**, the same 40 windows, THREE arms on every window (refcv7 prior-corrected 10 cm codes, refcv6@38k upsampled, positional prior fitted on 300 TRAIN clips / 60,286 frames, md5-verified): 40/40 map windows, 40/40 box3d packs for both models on refcv7's VIS-1 target blocks.
3. **map/box scorer**: `summarize` + `evaluate_bars_m7` + BAR-B7 bootstrap ran; both box controls read their known values (pooled AP == `detection_metrics.summarise` exactly, per-window census additive). Bars: NOT EVALUATED (validation). BAR-B7-2 exposed an undefined-F1 case (no confident slot at σ ≥ 0.5 at step 1,500), now reported NOT EVALUABLE with the census, never as a number.
4. **T1 battery**, 2 clips × 4 windows, seeds 0 and 1, reusing the step-1,500 G0 artifact (G0-A2 PASS; as registered FAIL, shown beside it): rolls, panels, VOID gates 1–4 PASS at both seeds, four-family analysis, cross-model paired cells, S6, tactical, replicate. BAR-R7-2 read ABSENT (no refcv6@38k dump yet), exactly as designed. The criteria registry check first failed to find `products/P7-TanitEval/CRITERIA_REGISTRY.json` in the ev7 tree; it was added to ev7 from the same commit by `git archive`, and the re-run reads 0 violations.
5. **refcv6@38k baseline harness** (the refcv6 package's `run_battery.py`, 82c2331 tree, 1 clip × 3 windows, no G0, smoke): dump written.

**Cost MEASURED here → SPEC A3:** 4.1 s per window for the full arm set (5.4 h per seed over S2), hence os-only at inference seed 1.

## RESULT — refcv7 step5000 (2026-09-28T14:19:53+0200)

* **G0 as registered: FAIL** — eval_traj [STOCHASTIC] in-run 0.64714 vs seeds mean 0.673517 (sd 0.00212)
* **G0-A2 (the gate, SPEC A2): FAIL**; mutation terms moved: M1 7 (detected), M2 60 (detected), M4 3 (detected); wrapper clause PASS; reasons: ['eval_traj [STOCHASTIC] in-run 0.64714 vs seeds mean 0.673517 (sd 0.00212)']
* **Tier:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed loop, never driving performance. **Estimator:** FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0.
* **Inference-seed floor:** None m ADE (training-seed floor NOT measured).

| bar | verdict | seed 0 | seed 1 |
|---|---|---|---|

| BAR-R7-N1 (NavSim) | not this battery | — | — |

**Four families, paired `os − ha0_ext` (seed 0):** 
* strategic: NOT APPLICABLE (n 0): strategic layer OFF

## 2026-10-04 — WHY step 5,000's G0 failed (EvalFlyWheel; diagnosis, SPEC AMENDMENT A5)

> Every number here is a **G0 loss-replay diagnostic** (the trainer's own in-run eval terms on its 128 fixed windows; `eval_traj` is the oracle-anchor L1), **not a battery number and not a tier claim**. Evidence class: MEASURED from `raw/step5000/g0.json` unless stated.

**The failure.** G0 as registered and G0-A2 FAIL on one term, `eval_traj` [STOCHASTIC]: in-run 0.64714 vs the 8-seed replay mean 0.673517, sd 0.00212 (12 sd). Every other term held (SMOOTH median 5.1e-4, MATCHED 6.4e-4, DETECTION 1.8e-4, COUNT exact, wrapper clause PASS under P3, M1/M2/M4 detected 7/60/3 terms).

**Ruled out from source** (launch blob `refc_v3_train.py` 203b437f = the ev7 copy, md5 0a6fb0d8…; file:line in SPEC A5):
* a step-keyed schedule in the eval path — the loop mutates one model attribute per step, `anchor_withheld_bank` (`:9410-9412`), constant `"fixed"` for this run (`config.json` withheld_bank mode fixed, warmup 0); the LR schedule (`:8969-8971`) is optimiser-only; the tactical-prior EMA is gated on `model.training` (`:4608-4609`); **nothing changes between steps 1,500 and 5,000 in the eval path**;
* EMA / averaged weights — none in the trainer; train/eval mode — `model.eval()` + `no_grad` (`:9658`, `:9672`), F1 random-t only in training (`refc.py:2800`);
* the saved weights — `ckpt.pt` / `ckpt_5000.pt` written at `:9634-9647` BEFORE the eval at `:9656`, same `model`, same step; `train.log`: `ckpt step 5000 -> ckpt.pt` then `[v3:eval] step 5000 … traj 0.6471`; one uninterrupted run (no restart in 1,474 log lines);
* the windows — identical `inrun_eval_perm` sha256; `V3Dataset.__getitem__` draws no random number (`:3808-3937`); `num_workers=0`, `shuffle=False` (`:8908-8911`);
* micro-batching — wrapper P3 max rel 1.5e-6 on `traj`;
* trunk numerics (`--trunk-compile` dropped, Thor vs RTX 4060, bf16) — **quantitatively**: M1 perturbs the trunk enough to move `lat` 1.4 %, `goal_tac` 3.7 %, `lon_tac` 2.6 %, yet moves `eval_traj` only +0.12 % (0.67416 vs seed-0 0.67337); the in-run row matches the replay on those heads to ≤ 0.16 %.

**What remains: the DDIM draw and the estimator built on it.** The eval decode draws `eps = torch.randn_like(x0_n)` (`refc.py:2802`) from the run's training RNG; the replay seeds it. The per-batch seed spread of `traj` (sd 0.0175–0.0333 per batch; 8 batches × 8 seeds) implies an sd of the 128-window mean of **0.00878** if the batches' draws are independent; the 8 seed means have sd **0.00212** (ratio 0.24; χ²₇ lower tail p = 2.8e-4). At step 1,500 the same ratio is 1.30 (sd 0.0153). Under the independence sd the in-run deviation (0.0264) is 3.0 sd.

**The discriminating test, pre-registered in SPEC A5 item 7 BEFORE any of its numbers** (`code/g0_diag_r7.py`, launched detached by `code/orchestrate_final.sh`): 16 more seeds (8..23) + `eps0` / `fp32_s0` / `loaderflags_s0` / `micro_alt_s0` + an `s0` control that must reproduce 0.67337. 24-seed sd ≥ 0.006 ⇒ estimator (A5 is the fix); ≤ 0.004 ⇒ mechanism, localised by the arms; between ⇒ INCONCLUSIVE. **RESULT of the discriminating test (MEASURED 2026-10-04 01:10–02:35 Berlin; `raw/g0diag_step5000/diag.json`, `raw/step5000/g0_A5_posthoc.json`):**

| arm (seed-0 replay unless named; 128 windows, 8 batches) | `eval_traj` | `eval_cascade` | `eval_sel_v3` |
|---|---|---|---|
| in-run (Thor, the run's own draw) | 0.64714 | 4.46348 | 1.85047 |
| `s0` — control, a fresh process through the A5 mmap path | **0.67337** (= G0 seed 0; 1,097/1,098 keys bit-equal, the other `eval_map_hires` rel 7e-9) | 4.71026 | 1.90450 |
| seeds 0..23 (8 from G0 + 16 new): mean / **sd** / min / max | 0.67161 / **0.01072** / **0.63596** / 0.68866 | 4.60576 / 0.08069 | 1.88176 / 0.05502 |
| `eps0` — DDIM draw zeroed | 0.65478 | 4.47285 | 1.82476 |
| `fp32_s0` — trunk fp32 + NCHW, cuDNN TF32 off | 0.67313 (−0.04 % vs `s0`) | 4.71075 | 1.90347 |
| `loaderflags_s0` — every `requires_grad` False | 0.67337 (identical to `s0`) | 4.71026 | 1.90450 |
| `micro_alt_s0` — micro 3,3,3,3,4 (⚠ a different micro split re-maps the eps stream, so this is another DRAW, not a numerics test) | 0.66036 | 4.54096 | 1.87656 |

* **Pre-registered branch reached: 24-seed sd 0.01072 ≥ 0.006 ⇒ the K = 8 sd was a small-sample under-estimate; the mechanism is the ESTIMATOR, and A5 is the fix.** Seeds 0..7 have sd 0.00212, seeds 8..23 sd 0.01308; with 24 seeds the 8 batch draws are independent (row-mean sd 0.01072 vs the independence value 0.01001; off-diagonal/trace +0.15 — the −0.94 seen on 8 seeds was the same sampling accident).
* The in-run value is an ordinary draw of the replay distribution: z = −2.28 on 24 seeds, and **seed 15 (0.63596) lies BELOW it**. `cascade` z −1.76, `loss` −1.51, `sel_v3` −0.57, `goal_score_absmean` −2.58 (below all 24 seeds; inside the A5 interval only through the registered 1 % term — noted, not hidden). These decoder terms move together because they share one draw; the in-run draw was a good one.
* Every other candidate is now measured, not argued: trunk precision 0.04 %, `requires_grad` state 0.0, a noise-free decode (`eps0`) sits between the in-run and the seed mean and does not reproduce it (0.65478 vs 0.64714).
* **Post-hoc A5 re-judge of step 5,000: PASS** (`raw/step5000/g0_A5_posthoc.json`; A5 PI for `eval_traj` [0.63418, 0.70903]; M1 / M2 / M4 still detected: 7 / 60 / 4 terms). ⛔ **POST HOC — A5 was written after the step-5,000 numbers were seen; this is REPORTED and is NOT step 5,000's gate result. Step 5,000's registered verdicts (as registered FAIL, A2 FAIL) stand.** No step-5,000 battery number exists.
* ⚠️ Cost note (MEASURED): an arm took 80–305 s instead of G0's 77 s; the desktop held 3.2–3.6 GB of the 8 GB card, so G0's 5.9 GiB peak spilled into shared memory whenever both were resident. Numbers are unaffected (`s0` bit-reproduced G0); wall time is not.

**SPEC AMENDMENT A5** (registered 2026-10-04T00:58:29+0200, sha256 `5582f092…` in `raw/SPEC_SHA256_AMENDMENT_A5.txt`, before any step-15k/20k/30k/50.4k G0 number): STOCHASTIC class on **K = 24 seeds**, t(0.995, 23) = 2.807, form and 1 % term unchanged — an exact 99 % PI, **stricter in expectation** (2.83 σ vs 3.58 σ); as registered (seeds 0..7) and A2 still reported first; the battery proceeds iff G0-A5 = PASS; A5 fixes an estimator, never a mechanism. Step 5,000's registered verdicts **stand** (FAIL).

## RESULT — refcv7 step30000 (2026-10-04T06:58:23+0200)

* **G0 as registered: FAIL** — eval_agent_det_ap2_bus_40_60 [DETECTION] in-run 0.1 vs seeds mean 0.125 (sd 0); eval_box3d_det_ap2_bus_20_40 [DETECTION] in-run 0.25 vs seeds mean 0.33333 (sd 0); eval_box3d_det_ap2_stroller_0_20 [DETECTION] in-run 0.16667 vs seeds mean 0.14286 (sd 0)
* **G0-A2 (seeds 0..7): FAIL** — eval_tacv6_goal_conf_bce [SMOOTH] in-run 0.15189 vs seeds mean 0.1542 (sd 0)
* **G0-A5 (THE GATE, SPEC A5; 24 inference seeds): FAIL**; mutation terms moved: M1 7 (detected), M2 59 (detected), M4 2 (detected); wrapper clause PASS; reasons: ['eval_tacv6_goal_conf_bce [SMOOTH] in-run 0.15189 vs seeds mean 0.1542 (sd 0)']
* **Tier:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed loop, never driving performance. **Estimator:** FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0.
* **Inference-seed floor:** None m ADE (training-seed floor NOT measured).

| bar | verdict | seed 0 | seed 1 |
|---|---|---|---|

| BAR-R7-N1 (NavSim) | not this battery | — | — |

**Four families, paired `os − ha0_ext` (seed 0):** 
* strategic: NOT APPLICABLE (n 0): strategic layer OFF

## 2026-10-04 G0 DIAGNOSIS AT STEP 30,000 — why `eval_tacv6_goal_conf_bce` is 1.5 % off, the A6 draft, the lock fix (G0-30k diagnosis agent; written ~08:15 Berlin)

**Verdicts that stand (unchanged, registered):** step 30,000 G0 as registered **FAIL**, G0-A2 **FAIL**, G0-A5 **FAIL** (`raw/step30000/g0.json`). Nothing below changes them. No step-30,000 battery number exists.

**The failing term.** `eval_tacv6_goal_conf_bce` [SMOOTH]: in-run 0.15189 vs 0.15420 on all 24 seeds (sd 0), rel **1.52 %** against the 1 % per-term bar (MEASURED, `raw/step30000/g0.json`). The as-registered DETECTION failures are all A2-exempt: support n_pos = 1–3 (REPORTED, `verdict.terms.*.support_n`).

### Mechanism (source + MEASURED; the 30,000-specific cell-level confirmation is ARMED, see below)
1. **The term is discontinuous in the numerics by construction.** Its target is a hard threshold of the model's own output: `correct = (sigmoid(goal_logits) >= 0.5) == goal_y` (`stack/tanitad/refs/refcv6_tactical.py:823-825`), then `BCE(goal_conf, correct)` weighted by `goal_w × class_mask`, divided by the batch's Σw (`:826-833`). On G0's 128 windows only **271 cells are supervised — 18 / 26 / 53 / 23 / 7 / 53 / 36 / 55 per batch** (MEASURED, `tacv6_n_supervised_goal_cells` per batch). One validity logit crossing 0 moves its batch by exactly ±c·w/Σw (`softplus(c) − softplus(−c) = c`), i.e. the mean by ±c/(8·n_b). The whole 30k gap (−0.00231 on the mean) is ONE flip with a confidence logit |c| of 0.13 (batch 4) … 1.02 (batch 7); in batch 4 any flip with |c| > 0.085 alone exceeds the 1 % bar (ESTIMATED from those counts). SPEC §2 names exactly this property as the reason for the DETECTION class ("a threshold … can flip on cross-hardware numerics"); the by-name table put this term in SMOOTH.
2. **The replay cannot reproduce the run's trunk numerics, and the gap grows with training.** The run used `--trunk-compile` (Inductor, bf16, Thor; `config.json` argv). The loader drops it (`refcv7_loader.py:144-146`, no Triton on Windows) and records "module tree and state_dict identical" — true, silent on numerics; the trunk's own docstring MEASURED on Thor that compiled bf16 differs from eager bf16 (`timm_trunk.py:223-226`). MEASURED drift of the replay vs the run, same 128 windows: SMOOTH class median **3.6e-5 (1,500) → 5.1e-4 (5,000) → 1.64e-3 (30,000)** against a 2e-3 bar; the wrapper clause's as-run batch-partition sensitivity **0.28 % → 0.78 % → 5.2 %** (P1 max rel) while fp32 stays ≤ 6.4e-6 (P3) (`raw/*/g0.json` `wrapper_control`). This term: exact at 1,500 (0.33698 = 0.33698), 0.36 % at 5,000, 1.52 % at 30,000 — the confidence head grows more confident (in-run BCE 0.48 → 0.15, `metrics_final_50400.jsonl`), so each flip costs more.
3. **The discriminating measurement (MEASURED, step 5,000, `raw/g0diag_step5000/diag.json`, an arm pre-registered in A5 item 7):** `fp32_s0` — the SAME weights, flags, windows and code, only the trunk in fp32 + NCHW with cuDNN TF32 off — moved this term **0.29403 → 0.29518 (+0.39 %), entirely in ONE batch** (batch 1, n = 26: 0.33497 → 0.34446; the other seven ≤ 0.0003): one flip with c ≈ 0.25. The continuous `eval_tacv6_goal_bce` moved 0.03 %. The in-run value **0.29509 sits 0.03 % from the fp32 arm.** Over the 13 SMOOTH keys with |x| ≥ 0.1 the median |fp32 − s0| = **5.6e-4** ≈ the median |in-run − s0| = **5.2e-4**: the replay's deviation from the run sits at the dev box's own numerics floor. `micro_alt_s0` (partition only) moved nothing (≤ 2.4e-6).

### Ruled out, with file:line (the brief's candidates a–d)
* **(a) state outside the state_dict.** The training loop mutates one model attribute per step, `anchor_withheld_bank` (`refc_v3_train.py:9410-9412`; `fixed`, warmup 0); the tactical-prior EMA is gated on `model.training` (`:4607`); every non-persistent buffer in `stack/tanitad/` is a construction-time constant (`timm_trunk.py:608-615`, `slot_query_select.py:169-170`, `refc.py:1761-1762`, `refcv7_heads.py:168/250-251`, `encoder.py:369-370`, `readout.py:107-110`, `v6.py:2492/5199-5201`); the tactical decoder is a pure function of parameters + (agent tokens, BEV tokens, nav/max-speed/v0/a0) (`refcv6_tactical.py:500-582`). The `requires_grad` flag state moves no tactical term on the dev box (`requires_grad_control`, 23/444 terms at ≤ 8.6e-7 rel, none `tacv6_*`).
* **(b) a training-time target.** The class mask and pos_weight are launch-time constants from the train split (`refc_v3_train.py:8490-8496`), read by the loader from the stamp (`refcv7_loader.py:472-491`); `tacv6_n_supervised_goal_cells` (Σ w·mask) reproduces EXACTLY at 1,500 / 5,000 / 30,000 (33.875). A static constant would have shown at 1,500, where the term reproduced exactly.
* **(c) a build departure.** Static ones (graft flags via the trainer's own pin, `refcv7_loader.py:360-362`; G-DVB 0 mismatches; `param_breakdown` equal) would show at 1,500. The one departure whose effect GROWS with training is the dropped `--trunk-compile` — mechanism 2.
* **(d) the milestone file.** `ckpt_30000.pt` is `model.state_dict()` written at `refc_v3_train.py:9645-9647`, after `ckpt.pt` (`:9634-9644`) and BEFORE the eval (`:9656`), same object, same step; strict load 0 missing / 0 unexpected. Key sets: the in-run row and the replay carry the same 1,098 keys at all three steps (the in-run's 11 extra are EXCLUDED `*_calib_*` + `eval_step`).

### The 30,000-specific confirmation — ARMED, outcomes committed before any number
* **CPU (no GPU, waits for ≥ 6 GB free commit and no chain marker; gives up 11:26 if never started):** `code/g0_cells_cpu_r7.py` → `D:/refcv7_eval_kit/battery/g0cells_step30000_cpu/cells.json`: G0's 128 windows with the trunk in fp32 on CPU, every batch's tactical cells captured; then `code/g0_cells_analyse.py` decomposes each batch's bf16-replay − fp32 difference and the in-run − fp32 difference into validity flips.
* **GPU (first stage of the armed waiter, ~25 min):** `g0_refcv7.py --seeds 0` on `ckpt_30000.pt` (seed 0 must reproduce the banked seed 0 within rel 1e-5, else REFUSED) + the `fp32_s0` arm + both captures → `code/g0_rejudge_a6.py` → the POST-HOC A6 re-judge, banked to `raw/g0a6supp_step30000/`.
* **Committed now:** CONFIRMED if the fp32 arm's difference from the bf16 replay is a sum of single-cell flips of near-threshold cells (residual at the continuous term's own floor) and the in-run value lies inside the A6 flip interval; NOT CONFIRMED if the fp32 arm moves the term by less than its continuous floor with no flip AND no near-threshold cell can carry ±0.0185 on the 8-batch sum — then the gap is UNEXPLAINED by numerics and the next lever is a seed-0 replay **on Thor with `--trunk-compile` kept** (the loader supports a Thor kit, `refcv7_loader.py:95-101`; Thor compute is the Master Mind's call).

### The fix: SPEC AMENDMENT A6 — DRAFTED, NOT REGISTERED (`raw/AMENDMENT_A6_DRAFT.md`, sha256 in `raw/AMENDMENT_A6_DRAFT.sha256`, written before any step-50,400 number and before any step-30,000 cell/fp32 number)
* **THRESHOLD_TARGET class** (by name): `eval_tacv6_goal_conf_bce`, judged by a **flip interval** — every supervised cell whose replay validity logit is within 3·δ of 0 (δ = the largest validity-logit move the numerics-only `fp32_s0` arm produced on that checkpoint) may flip; the in-run value must lie in the reachable interval ± the SMOOTH tolerance; two controls (cells reproduce the trainer's per-batch values; the interval centre equals the logged replay) or OUT.
* **SMOOTH floor:** tolerance max(registered, 3·φ_k), φ_k = |fp32_s0 − s0| measured on that checkpoint; class-median bar max(0.2 %, 3 × median φ/|x|). 3 = SPEC §2's wrapper-clause convention. Mutations judged with the same tolerances; **M1 undetected ⇒ VOID** (A6 can lose power, never invent it). Every rescued term is named.
* **Power check at 5,000 (POST HOC, partial — no cells existed there):** with the φ floors applied, M1 / M2 / M4 are still detected with the same counts (7 / 60 / 4); the THRESHOLD term fails closed for lack of cells, as designed.
* **Code** (`code/g0_refcv7.py`): every 24-seed G0 now runs `fp32_s0` + both captures (+~1 eval pass) and writes `verdict_A6` beside `verdict_A5`; **the gate switches to A6 only if `raw/SPEC_SHA256_AMENDMENT_A6.txt` was written BEFORE that G0 started** (`a6_registration`). Unregistered ⇒ A5 gates, A6 is reported as DRAFT. Re-judging the banked 1,500 / 5,000 / 30,000 artifacts with the modified judge reproduces every registered verdict, reason list, median and mutation count exactly (A5/A2/as-registered are untouched). Tests: `code/test_g0_a6.py` (literals, a brute-force flip enumeration, the REAL `tactical_behaviour_losses`, and the deliberate regression of a flip-SIGN error this author actually made first), `code/test_g0_main_a6_flow.py` (the real `main()` end to end on fakes: A6 DRAFT → A5 gates; registered-before-start → A6 gates; `--no-a6` = the old flow).
* ⚠️ **Why register it before 12:00 if at all:** under A5 alone the step-50,400 G0 is likely to fail the same way — the same discontinuous term, and the SMOOTH median at 82 % of its bar and rising.

### The lock gate (the 03:35 `ZZCHAINNOLOCK50400ZZ`) — FIXED and tested
`nvidia-smi` timed out once (60 s) under memory pressure; `subprocess.TimeoutExpired` escaped `gpu_gate.gate()` and `gpu_lock.acquire()`, the lock tool wrote no `--out`, the chain read a missing file and stopped (`D:/refcv7_eval_kit/chain/chain_50400.log`). Now a failed probe is a WAIT (fail closed, `gpu_gate.PROBE_ERRORS`, `probe_ok` False) retried every poll; `gpu_lock.py` always writes `--out` and releases a lock it created if anything unexpected raises; `with_gpu_lock.py --child-timeout-s`. `code/test_gpu_lock_retry.py`: 3 timeouts then success → acquired on poll 4; never acquires while the probe fails; never breaks a held lock; the deliberate regression (`PROBE_ERRORS = ()`) reproduces the 03:35 crash.

### Armed (nothing started on the GPU; the NavSim runner holds the lock)
`code/waiter_final.sh 50400 129600 120 disk` (detached; log `D:/refcv7_eval_kit/chain/waiter_final_50400.log`): GO iff the lock file is absent AND free disk on D: ≥ 10 GB AND free commit ≥ 6 GB (`code/final_waiter.py`, fail closed), polled every 120 s for ≤ 36 h → the step-30,000 A6 supplement + post-hoc re-judge → GO re-checked → `chain_milestone.sh 50400`.
⛔ **Named blocker: D: has 7.7 GB free (< 10 GB)** — MEASURED 08:03; until it reaches 10 GB the waiter will not start, by the brief's rule. One-command lever (the Master Mind's call): relaunch it with mode `auto` (`ram` when the disk rule fails): G0's batches then stay in RAM (`REFCV7_G0_BATCH_CACHE=ram`, the same tensors; the disk batch-cache path never runs) and GO needs ≥ 12 GB free commit instead.

### Exploratory (NOT a registered read; the registered check is `raw/PREREG_SEED_GROUP_CHECK_50400.md` on step 50,400)
Step 30,000 `eval_traj` seed means: seeds 0–7 sd **0.00399**, seeds 8–23 sd **0.00715**, one-sided F(15, 7) = **3.21, p = 0.063**; Levene p = 0.20 (MEASURED, `raw/step30000/g0.json`). The step-5,000 pattern (F 38.1) does not recur at this strength; it neither confirms nor refutes reading (A) or (B) — the registered test is the step-50,400 one.
* **Addendum 08:13 (operational, no outcome changed):** the CPU job first met its 6 GB commit bar at 08:11:52 and was stopped by its author before it allocated anything, because free commit was MEASURED swinging **1.57 → 5.86 GB within two 2-min polls** (another stream's ~4.3 GB burst) — a 6 GB bar does not protect that stream from a ~3 GB job. Relaunched with a **10 GB** bar, no start deadline (≤ 36 h), and it also waits while any battery GPU job (`refcv7-milestone-*`, `refcv7-g0*`) holds the lock. Follow-up `code/post_cells.sh` runs the flip decomposition and banks to `raw/g0cells_step30000_cpu/` when it ends.

## RESULT — refcv7 step50400 (2026-10-04T14:49:14+0200)

* **G0 as registered: FAIL** — eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0)
* **G0-A2 (seeds 0..7): PASS**
* **G0-A6 (THE GATE, SPEC A6; 24 inference seeds): FAIL**; mutation terms moved: M1 22 (detected), M2 62 (detected), M4 2 (detected); wrapper clause PASS; reasons: ['eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0)', 'eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0)', 'eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0)', 'eval_agent_det_ap2_stroller_0_20 [DETECTION] in-run 0.33333 vs seeds mean 0.25 (sd 0)', 'eval_agent_det_ap2_stroller_all [DETECTION] in-run 0.16667 vs seeds mean 0.125 (sd 0)']
* **G0-A5 (24 seeds; reported beside the A6 gate): PASS**
* **G0-A6 (REGISTERED; measured numerics floor): FAIL** — eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0); floor-rescued terms: []; threshold-target terms: eval_tacv6_goal_conf_bce in-run 0.14591 interval [0.14224843595836026, 0.18356920461913012] OK
* **Tier:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed loop, never driving performance. **Estimator:** FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0.
* **Inference-seed floor:** None m ADE (training-seed floor NOT measured).

| bar | verdict | seed 0 | seed 1 |
|---|---|---|---|

| BAR-R7-N1 (NavSim) | not this battery | — | — |

**Four families, paired `os − ha0_ext` (seed 0):** 
* strategic: NOT APPLICABLE (n 0): strategic layer OFF

## X8 CLOSURE (2026-10-11) -- the 50,400 four-family battery RAN after the G0 text PASS; its banked section follows verbatim

* **Why this block exists.** The section above dated 2026-10-04T14:49:14+0200 is the AS-REGISTERED run (G0 FAIL as coded). The Master Mind ruled 2026-10-04 (GOALS_AND_CLAIMS tip 235156d, RETRACTION_LOG R31) that G0-A6 as registered in the text is PASS and the coded FAIL is a judge defect (the A2 low-support tuple omitted A6). The battery then ran to completion under the corrected judge on 2026-10-05; nothing in RESULT.md recorded it until now. **MEASURED, artifact: `raw/step50400_a6text/RESULT_step50400_a6text.json` (section text: `raw/step50400_a6text/RESULT_SECTION_step50400_a6text.md`; the as-registered record stays in `raw/step50400/RESULT_step50400.json`).**
* **Judge status for FUTURE checkpoints.** `run_battery_r7.g0_text_override` re-judges on content and is pinned by `battery/code/test_battery_g0_text_override.py` (deliberate-regression and mutation-count arms). Re-run 2026-10-11: **105 passed** (`raw/fanctl_x8/pytest_g0.out`). No gap found, so **no A7 is drafted**. Two records stay carried, as the section says.
* **Tier / estimator / CI scope for everything in the next section:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); FULL-SET mean; paired episode-cluster bootstrap n_boot 2000 seed 0. The CI answers ONLY "would another draw of EPISODES say this?". It does NOT answer training-seed variance (single training seed; floor unmeasured) and the inference-seed floor is 0.0006 m ADE.
* **Every BAR-R7-1 / M7 / B7-2 FAILED line below is a committed-bar result, reported as written. The continuation levers are named in the fan-control block that follows the verbatim section.**

## RESULT — refcv7 step50400_a6text (2026-10-05T02:09:30+0200)

* **G0 as registered: FAIL** — eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0)
* **G0-A2 (seeds 0..7): PASS**
* **G0-A6 (THE GATE, SPEC A6; 24 inference seeds): FAIL**; mutation terms moved: M1 22 (detected), M2 62 (detected), M4 2 (detected); wrapper clause PASS; reasons: ['eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0)', 'eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0)', 'eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0)', 'eval_agent_det_ap2_stroller_0_20 [DETECTION] in-run 0.33333 vs seeds mean 0.25 (sd 0)', 'eval_agent_det_ap2_stroller_all [DETECTION] in-run 0.16667 vs seeds mean 0.125 (sd 0)']
* **G0-A5 (24 seeds; reported beside the A6 gate): PASS**
* **G0-A6 (REGISTERED; measured numerics floor): FAIL** — eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0); floor-rescued terms: []; threshold-target terms: eval_tacv6_goal_conf_bce in-run 0.14591 interval [0.14224843595836026, 0.18356920461913012] OK
* ⭐ **G0 RECORD (gate source: REGISTERED A6 TEXT, corrected judge (Master Mind ruling 2026-10-04)):** **G0-A6 as registered (text): PASS, computed by the corrected judge after the coded verdict was read; the criterion is unchanged** · **G0-A6 as coded: FAIL (judge defect: the A2 low-support tuple omitted A6)** (`g0_A6_text.json`, sha256 f3c7de52db67…; re-verified by the judge in use)
* **Tier:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed loop, never driving performance. **Estimator:** FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0.
* **Inference-seed floor:** 0.0006 m ADE (training-seed floor NOT measured).

| bar | verdict | seed 0 | seed 1 |
|---|---|---|---|
| BAR-R7-1 | FAILED | -0.0102 [-0.0260, +0.0067] (n 4754/139) | -0.0096 [-0.0253, +0.0069] (n 4754/139) |
| BAR-R7-2 | PENDING (the refcv6@38k S2 rolls run LAST in the milestone chain; SPEC A4) | ABSENT | ABSENT |
| BAR-R7-3 | PASS (single training seed; training-seed floor unmeasured) | -0.9804 [-1.2940, -0.6815] SEP (n 3668/139) | -0.9796 [-1.2855, -0.6842] SEP (n 3668/139) |
| BAR-M7-1 | FAILED | (per band in perc_score.json) | — |
| BAR-M7-2 | FAILED | (per band in perc_score.json) | — |
| BAR-M7-3 | FAILED | (per band in perc_score.json) | — |
| BAR-M7-4 | FAILED | (per band in perc_score.json) | — |
| BAR-B7-1 | PASS (single training seed; no replicate floor measured for boxes) | +0.1051 [+0.0796, +0.1289] SEP (n 4532/139) | — |
| BAR-B7-2 | FAILED | -0.2204 [-0.2448, -0.1885] SEP (n 4532/139) | — |

| BAR-R7-N1 (NavSim) | not this battery | — | — |

**Four families, paired `os − ha0_ext` (seed 0):** 
* ADE: ade_m -0.0102 [-0.0260, +0.0067] (n 4754/139); fde_m -0.0360 [-0.0759, +0.0054] (n 4754/139)
* longitudinal: LON_speed_mae_mps +0.0040 [-0.0102, +0.0198] (n 4754/139); LON_along_mae_m -0.0086 [-0.0225, +0.0057] (n 4754/139); LON_accel_mae_mps2 +0.0534 [+0.0329, +0.0753] SEP (n 4754/139)
* lateral: LAT_cross_mae_m -0.0049 [-0.0158, +0.0059] (n 4754/139); LAT_heading_mae_deg -0.0961 [-0.2042, +0.0146] (n 4509/139); LAT_yaw_rate_mae_radps_valid -0.0047 [-0.0075, -0.0021] SEP (n 4473/139)
* tactical: TAC_traj_lat_correct +0.0206 [+0.0103, +0.0306] SEP (n 4754/139); TAC_traj_lon_correct -0.0133 [-0.0288, +0.0006] (n 4754/139)

**Family levels, seed 0 (T1; FULL-SET mean; intervals in `analysis_s0.json`):**

| arm | speed MAE m/s | target-speed acc | along MAE m | min headway m / time gap s / min TTC s (n, n_closing) | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | tac lat / lon (traj) |
|---|---|---|---|---|---|---|---|---|---|
| os | 0.259 | 0.853 (±0.5 m/s) | 0.226 | 27.95 / 4.13 / 25.1 (n 1303, closing 482) | 1.35 | 0.0037 | 1.08 | 0.103 | 0.962 / 0.831 |
| os_filteroff | 0.259 | 0.853 (±0.5 m/s) | 0.226 | 27.95 / 4.13 / 25.1 (n 1303, closing 482) | 1.35 | 0.0037 | 1.08 | 0.103 | 0.962 / 0.831 |
| ha0_ext | 0.255 | 0.866 (±0.5 m/s) | 0.235 | 28.02 / 4.20 / 25.5 (n 1285, closing 446) | 1.45 | 0.0038 | 1.36 | 0.108 | 0.941 / 0.844 |
| ha | 0.255 | 0.866 (±0.5 m/s) | 0.235 | 28.05 / 4.21 / 25.6 (n 1291, closing 447) | 1.57 | 0.0041 | 1.47 | 0.124 | 0.937 / 0.844 |
| ha0 | 0.489 | 0.705 (±0.5 m/s) | 0.472 | 27.45 / 4.12 / 24.1 (n 1315, closing 499) | 2.82 | 0.0069 | 2.41 | 0.317 | 0.864 / 0.757 |
* TACTICAL lat decision, v6_behaviour_decoder: acc 0.7401, kappa 0.45663136165485746, n in band 1081 (label clock: the refcv7 loader's TRUE clip clock, `enable_clip_clock` + G3)
* TACTICAL lat decision, z_tac_v7_heads: acc 0.7299, kappa 0.3461702577189327, n in band 1081 (label clock: the refcv7 loader's TRUE clip clock, `enable_clip_clock` + G3)
* TACTICAL lon decision, v6_behaviour_decoder: acc 0.481, kappa 0.3184823321971728, n in band 1081 (label clock: the refcv7 loader's TRUE clip clock, `enable_clip_clock` + G3)
* TACTICAL lon decision, z_tac_v7_heads: acc 0.519, kappa 0.3631659980581842, n in band 1081 (label clock: the refcv7 loader's TRUE clip clock, `enable_clip_clock` + G3)
* TACTICAL goal selection (22 tokens): 2 scoreable classes (n_pos ≥ 200); per-class AUROC / AP / P / R in `tactical_v6_s0.json`
* strategic: NOT APPLICABLE (n 0): the strategic layer is OFF (--no-strategic); its route loss is gated off in training, so the route head is untrained
* ⚠️ baselines missing: ['b_refcv6_38k_s0', 'b_refcv6_38k_s1']

## D-REFCV7-FANCEIL -- fan random-pick control, refcv7-r101-s0 at 50,400 (2026-10-11, EvalFlyWheel)

* **Tests behind the judge statement above:** `pytest -q` over test_g0_a7, test_g0_a7_reports, test_g0_main_a7_flow, test_battery_g0_text_override, test_g0_a6, test_g0_a6_lowsupport, test_g0_main_a6_flow, test_r7_a7_reporting, test_g0_a5 (all in `battery/code/`) = **105 passed in 22.42 s**.
* **What was measured (MEASURED, T0 DIAGNOSTIC).** A fresh fan roll of ckpt_50400 (md5 b418d0fc4a92a6848c246a6a7c50207b), 4,754 windows, 139 episodes (the battery window set `s2_windows.json`), 117 candidates per window, inference seed 0, wall 1,959.9 s. Producer: `code/fan_roll_r7.py` (wrapper over `run_battery_r7.roll_one`, full=False). Bank: `D:/refcv7_eval_kit/fanctl/s0/dump/fan_extras.npz` sha256 ff2d0f86b1e3a3f6ff6979b27d8dd5d7a819d1386117b2e971f60668f99ffcb8 (NOT in the repo; 21957189 bytes, on the D: kit only). `traj_equals_sidecar_nav_true` = true: the captured fan reproduces the banked selected plan exactly (max |fan_sel - traj4| = 0.0 m). Analysis: `code/fan_control_x8.py` -> `raw/fanctl_x8/ctl_refcv7_50400_fan_s0.json`.
* **Metric.** ADE over the 4 saved waypoints (mean per-candidate L2 over 4 waypoints). Oracle-in-fan = per-window MIN over candidates (uses GT, optimistic, unreachable: the minimum of 117 noisy draws). **Tier:** the oracle and every difference containing it are T0 (GT-conditioned diagnostics, never driving performance); `selected` alone is the T1 open-loop pick. **Estimator:** FULL-SET mean; episode-cluster bootstrap (single arm) and PAIRED episode-cluster bootstrap (differences), `taniteval/taniteval/ci.py`, n_boot 2000, seed 0, n 4754 windows / 139 episodes. **CI scope:** ONLY "would another draw of EPISODES say this?". It does NOT answer training-seed or inference-seed variance; the seed-1 replicate below is the inference-seed check.
* **Cross-check.** fan-selected 0.2772 vs the banked battery `os` 0.2784 (seed-0 dump) and 0.2790 (seed-1 dump): paired delta -0.0012 [-0.0036, +0.0012] not separated (RNG differs between full=False and full=True, so only means are comparable).

| quantity (ADE 4wp, m) | value [95 % episode-cluster CI] |
|---|---|
| selected (shipped selector) | 0.2772 [0.2527, 0.3051] |
| oracle-in-fan (T0) | 0.1143 [0.1057, 0.1257] |
| uniform-random pick, analytic expectation | 1.5501 [1.5081, 1.5905] |
| sel_gap = selected - oracle (paired) | +0.1630 [+0.1446, +0.1825] SEP |
| random - selected (paired, analytic) | +1.2728 [+1.2201, +1.3232] SEP |
| oracle - random (paired) | -1.4358 [-1.4765, -1.3942] SEP |
| 12 explicit uniform draws (seeds 20261011..20261022) | mean 1.5557, range [1.5367, 1.5748]; random - selected range [+1.2595, +1.2976], **12 of 12 separated positive, 0 negative** |
| skill fraction (random - selected)/(random - oracle), POINT estimate | **0.8865** (0 = chance, 1 = oracle) |
| windows where selected > 2x oracle | **44.17 %** (uniform random: 99.64 %) |
| windows where selected beats the random expectation | 97.83 % ; distinct sel_idx used: 64 of 117 |

* **Reading, with the straw man named.** Uniform random over 117 candidates is the weakest null (the fan spans every anchor, so it includes plans for the wrong manoeuvre). Against it the selector has real skill (0.8865). The informative nulls are tighter, all PAIRED on the same windows:

| informative null | ADE | null - selected [CI] |
|---|---|---|
| random among top-2 by anchor_logits | 0.3010 | +0.0238 [+0.0179, +0.0297] SEP |
| random among top-4 / top-8 / top-16 / top-32 by anchor_logits | 0.3686 / 0.4678 / 0.6316 / 0.8985 | +0.0914 / +0.1905 / +0.3544 / +0.6213, all SEP |
| random among top-2 by refined_logits | 0.2994 | +0.0222 [+0.0179, +0.0264] SEP |
| random among top-2 by sel_score | 0.2983 | +0.0211 [+0.0167, +0.0254] SEP |
| argmax anchor_logits | 0.2726 | -0.0046 [-0.0102, +0.0009] not separated |
| argmax refined_logits | 0.2732 | -0.0040 [-0.0069, -0.0016] SEP (argmax is 4 mm BETTER) |
| hold-action plan `ha` | 0.3010 | +0.0238 [+0.0062, +0.0405] SEP |

  The selector beats a random choice among its own top-2 by only about 0.02 m, is not better than a plain argmax of the logits (the refined-logit argmax is 0.004 m better and separated), and beats the hold-action plan by +0.0238 m. (0.3010 and +0.0238 appear twice only because `ha` and the anchor_logits top-2 happen to agree at 4 dp; the intervals differ, so they are different quantities.) The `ha` here is the sidecar hold-action plan, NOT `ha0_ext` of BAR-R7-1, and the metric is the 4-waypoint ADE, not `ade_m`; do not read +0.0238 against BAR-R7-1 (-0.0102, not separated) as a contradiction without a matched-metric recompute (UNVERIFIED).

### D-REFCV7-FANCEIL -- continuation (2026-10-11, EvalFlyWheel): conflict, headroom, and one inference-only lever

**1. DOC-VS-RAW CONFLICT -- ESCALATED.** GOALS_AND_CLAIMS.md line 12229 (row D-REFCV7-FANCEIL) and MODEL_REGISTRY.md section 15.3 (the row at line 5300 at tip 37645fcc) label *oracle-in-fan 0.1914 m, selected 0.4728 m, gap +0.2813 [+0.2127, +0.3543], selected more than 2x worse in 41.09 %* as **refcv7 at 50.4k**. Those are the **refc-base-30k** numbers, not refcv7:
- REPRODUCED from raw (`raw/fanctl_x8/ctl_repro_refc-base-30k.json`, pt sha256 f7edc65b..., 128 anchors, 881 windows, 40 episodes): selected 0.4728, oracle 0.1914, sel_gap +0.2813 [+0.2127, +0.3543], frac_sel_2x_worse 0.4109. Identical to four digits, i.e. the same arm.
- INDEPENDENT corroboration: MODEL_REGISTRY line 3747 publishes 0.4728 [0.3835, 0.5699] as the registered REF-C-base ADE@2s headline (`taniteval/results/driving_refc-base-30k.json`). The refc-xl-30k fan reads 0.4714 / 0.1640 / +0.3075 (`ctl_repro_refc-xl-30k.json`, 256 anchors).
- True refcv7-r101-s0 at 50,400 (4754 windows, 139 episodes, 117 candidates, ADE over 4 waypoints): selected 0.2772, oracle 0.1143, sel_gap +0.1630 [0.1446, 0.1825], selected beyond 2x oracle 44.17 % (section above).
- The same registry row says the random-pick control is "not yet run". It has now run (section above). **Registry and claim row need the number swapped and the control status updated; this agent does not edit Project Steering.**
- NOT comparable across rows: the old-pt panel was built by `run_oldpt` on the older arms stored fans over 881 windows; its random expectation is 14.5 m against 1.55 m for refcv7, so the candidate-space geometry differs. Metric-horizon equality of the two panels is UNVERIFIED; do not pool them.

**2. HEADROOM -- what a reranker restricted to the model own top-K could win (T0 DIAGNOSTIC, uses GT).**
Instrument `code/fan_headroom_x8.py`, output `raw/fanctl_x8/ctl_refcv7_50400_headroom_s0.json`. Selected = 0.2772. Oracle among the top-K by the model score vs oracle among a random K-subset (8 draws, same windows). Controls read as required: K=1 equals the argmax pick (anchor 0.2726, refined 0.2732, both equal to the banked argmax rows) and the K=117 random-subset oracle equals the full oracle 0.1143.

| K | random-K oracle | top-K by anchor_logits | top-K by refined_logits | selected minus that oracle (refined), paired CI |
|---|---|---|---|---|
| 1 | 1.5534 | 0.2726 | 0.2732 | +0.0040 [+0.0016, +0.0069] SEP |
| 2 | 1.0939 | 0.2159 | 0.2203 | +0.0569 [+0.0512, +0.0628] SEP |
| 4 | 0.7349 | 0.1698 | 0.1737 | +0.1035 [+0.0935, +0.1137] SEP |
| 8 | 0.4831 | 0.1440 | 0.1462 | +0.1310 [+0.1182, +0.1442] SEP |
| 16 | 0.3130 | 0.1258 | 0.1275 | +0.1497 [+0.1341, +0.1664] SEP |
| 32 | 0.2115 | 0.1182 | 0.1172 | +0.1600 [+0.1422, +0.1790] SEP |
| 64 | 0.1501 | 0.1149 | 0.1146 | +0.1627 [+0.1444, +0.1819] SEP |

Reading: the model own ranking is informative (top-4 contains a 0.170 m plan where four random plans contain 0.735 m), so the fan is a real proposal set and not a lottery. That makes the gap a **selection** problem, not a proposal problem. But each of those oracle cells peeks at GT per window, so the table is a **ceiling** for a reranker, not an achievable number. Estimator: FULL-SET mean, paired episode-cluster bootstrap n_boot 2000 seed 0. The CI answers only "another draw of EPISODES?".

**3. NEXT LEVER, RUN IN THE SAME TURN -- an inference-only pick rule (post-hoc EXPLORATORY, not a registered claim).**
Instrument `code/fan_mbr_x8.py`, output `raw/fanctl_x8/ctl_refcv7_50400_mbr_s0.json`. Rule: minimum-Bayes-risk pick inside the model own top-K, `argmin_i sum_j p_j d(i,j)` with `p = softmax(score/T)` over the top-K and `d` the 4-waypoint L2 between two fan members. It uses **no GT, no ego state and no new network**, so the vision-only rule is untouched and it is admissible at inference. GT enters only the scoring.
Forking-paths guard: 36-cell grid (3 scores x K 4/8/16/32 x T 0.5/1/2); the cell is CHOSEN on episode-half A (even rank of sorted episode ids, 2398 windows) and REPORTED on half B (2356 windows, the other episodes). Chosen on A: `refined_logits`, K=32, T=1.0.

| set | selected | argmax refined | MBR | selected minus MBR (paired) | argmax minus MBR (paired) |
|---|---|---|---|---|---|
| half B (held out) | 0.2845 | 0.2810 | 0.2715 | +0.0130 [+0.0076, +0.0187] SEP | +0.0095 [+0.0050, +0.0138] SEP |
| all windows (reference only) | 0.2772 | 0.2732 | 0.2657 | +0.0115 [+0.0079, +0.0156] SEP | +0.0075 [+0.0046, +0.0106] SEP |

- **It is real but small.** On the held-out half MBR beats the registered pick by 13 mm (4.6 % of 0.2845) and beats the plain refined-logit argmax by 9.5 mm. That recovers about 8 % of the 0.1630 selected-to-oracle gap (0.0130 / 0.1630), so it is NOT the closing lever for FANCEIL. By the headroom table, a rule must exploit information beyond the model own scores to get near 0.17 m.
- **Variance honesty.** Single checkpoint, single inference seed (the seed-1 replicate below), n = 139 episodes. The CI answers only "another draw of EPISODES?". The 36-cell grid is exploratory: the half split controls selection bias within this fan, but half B belongs to the same 139 episodes, so it is not a fresh corpus. Promotion needs its own pre-registration and a fresh episode set.
- **Tier.** The pick is T1 open loop (the model fan, self-action, no GT in the rule). Whether it helps CLOSED loop is **not measured** and cannot be read from this table; the planner feeding its own rollout is the open-vs-closed ruling territory.
- **Four families.** This control scores the 4-waypoint ADE family (longitudinal/lateral not separated here). The four-family numbers for the registered selected pick are in the banked 50,400 section above; none was recomputed for the MBR pick, so the MBR gain is an ADE-family statement only. Per-family MBR scoring is a named open item.

### D-REFCV7-FANCEIL -- inference-seed replicate (seed 1) and what it settles (2026-10-11, EvalFlyWheel)

**Question this block answers (named, because one interval carries three):** *would another INFERENCE run say this?*
The standard CI (paired episode-cluster bootstrap, `taniteval/taniteval/ci.py`, n_boot 2000, seed 0) answers only
*another draw of EPISODES?*; the **training-seed floor is UNMEASURED** (H-ESTIM-SEED-1, tiny-rig `A0b_replicate`
false-positive rate 14.3 %). Two inference seeds (0, 1) on ONE checkpoint (refcv7 step 50,400, ckpt md5
b418d0fc4a92a6848c246a6a7c50207b) is **n = 2: a lower bound on inference noise, not a variance estimate.**
Tier: the fan oracle rows are **T0** (they use GT); `selected` and the MBR pick are **T1 open-loop picks**; the
closed-loop effect of the MBR pick is **NOT measured**. Estimator: FULL-SET mean, 4-waypoint mean L2 (`_de_all`).

| metric (m, 2 s grid, 4 wp) | seed 0 | seed 1 | s0 - s1 (paired) |
|---|---|---|---|
| selected | 0.2772 [0.2527, 0.3051] | 0.2790 [0.2542, 0.3068] | -0.0018 [-0.0041, +0.0005] not separated |
| oracle-in-fan (117 cand., GT) | 0.1143 | 0.1145 | -0.0003 [-0.0015, +0.0010] not separated |
| random expectation | 1.5501 | 1.5501 | -0.0001 |
| sel_gap (selected - oracle) | +0.1630 [0.1446, 0.1825] | +0.1645 [0.1454, 0.1847] | -0.0015 [-0.0041, +0.0009] not separated |
| skill fraction | 0.8865 | 0.8854 | |
| windows with selected > 2x oracle | 44.17 % | 44.95 % | |
| argmax(refined_logits) - selected | -0.0040 [-0.0069, -0.0016] SEP | -0.0057 [-0.0092, -0.0027] SEP | |
| ha - selected | +0.0238 [0.0062, 0.0405] | +0.0220 [0.0043, 0.0388] | |

Artifacts (all MEASURED): `raw/fanctl_x8/ctl_refcv7_50400_fan_s0.json`, `..._fan_s1.json`,
`..._headroom_s{0,1}.json`, `..._mbr_s{0,1}.json`, `..._seedcmp.json`; instrument `code/fan_seedcmp_x8.py`.
Same windows both seeds (window order asserted identical; n = 4,754 windows, 139 episodes).

**Verdict.** The FANCEIL finding for refcv7 -- the fan CONTAINS a plan ~0.114 m from GT and the selector lands at
~0.278 m, a **+0.163 m selection gap and skill fraction ~0.886** -- is **not an inference-seed artefact**: the
seed-to-seed change in `selected` is 1.8 mm (CI straddles 0) against a 163 mm gap. This says nothing about the
TRAINING-seed floor and nothing about fresh episodes.

**What varies and what does not (MEASURED, `ctl_refcv7_50400_seedcmp.json`).**
* `anchor_logits` are **identical across inference seeds** (max abs diff 0.0, argmax agreement 1.0): the proposal
  distribution is deterministic.
* `refined_logits` and the fan trajectories are **stochastic**: refined_logits differ by up to 4.04, argmax
  agreement 0.9011; selected-index agreement 89.0 %; mean L2 between the two selected plans 0.0827 m
  [0.0796, 0.0860]; mean L2 between corresponding fan candidates 0.0803 m; only 2.27 % of windows have a bitwise
  identical fan.
* Headroom (selected minus refined-oracle at top-K) is stable: +0.0040 / +0.0057 at K = 1,
  +0.0569 / +0.0582 at K = 2, +0.1627 / +0.1642 at K = 64 (s0 / s1). The gap is a SELECTION problem, not a
  proposal problem, on both rolls.

**Consistency with the banked battery dumps (and one open question).**
* The s1 fan roll reproduces the banked battery seed-1 `os` arm **exactly**: paired delta 0.0 [0.0, 0.0], banked
  mean 0.279. The s0 fan roll differs from banked `dump_s0` by -0.0012 (and from `dump_s1` by +0.0006
  [-0.0019, +0.0031], not separated).
* MEASURED from `raw/step50400_a6text/roll_s{0,1}.json`: the banked seed-0 roll used the **full** arm set
  (SPEC 3.2) and the banked seed-1 roll the **os + model-free controls** set (SPEC A3); the fan roller
  (`code/fan_roll_r7.py`) calls `roll_one(..., full=False)` for both seeds. So seed 1 was rolled with the SAME arm
  set as the banked dump and seed 0 was not. That the differing arm set (hence differing RNG consumption) is the
  CAUSE of the s0 offset is **UNVERIFIED** -- consistent with, not isolated by, this evidence. Do not read the
  s1 exact match as a reproducibility guarantee beyond the measured identity.

**The inference-only lever replicates (MBR re-rank of the model own top-K; post-hoc, exploratory, NOT a registered claim).**
Rule: `argmin_i sum_j p_j d(i,j)` over the top-K, `p = softmax(score / T)`; no GT, no ego state, no new network,
so it is admissible at inference. 36-cell grid (3 scores x K{4,8,16,32} x T{0.5,1,2}); the cell is chosen on
episode-half A and reported on half B (same 139 episodes -- NOT fresh data). Gain = selected - MBR (positive =
MBR closer to GT), paired episode-cluster bootstrap on half B.

| cell | seed | gain on half B | argmax(refined) - MBR |
|---|---|---|---|
| own cell: refined_logits, K 32, T 1 | s0 | +0.0130 [+0.0076, +0.0187] | +0.0095 |
| own cell: sel_score, K 32, T 2 | s1 | +0.0173 [+0.0067, +0.0282] | +0.0103 |
| s0 cell FIXED on the s1 fan (`--fixed-cell refined_logits,32,1`) | s1 | +0.0136 [+0.0073, +0.0202] | +0.0083 [+0.0036, +0.0129] |

All-window reference for the fixed-cell s1 gain (NOT a held-out number): +0.0118 [+0.0081, +0.0161].
* The MBR gain (13 to 17 mm) is ~7x the s0-vs-s1 difference in `selected` (1.8 mm): it is not inference noise.
* The cell chosen on A is **not stable across seeds** in score or T (K = 32 is); the lever does not depend on
  which near-tied cell wins, as the fixed-cell replicate shows.
* Size check: MBR closes ~13 to 17 mm of a 163 mm gap (~8 to 10 %). It is a small, real, free gain -- not the
  fix for the selection problem. `argmax(refined_logits)` already beats `selected` by 4 to 6 mm (SEP on both
  rolls), i.e. part of the MBR gain is simply "do not use the head selector".
* Closed-loop effect: NOT MEASURED. Per-family (four-family) scoring of the MBR pick: NOT YET DONE.

**Instruments added / changed this segment.** `code/fan_seedcmp_x8.py` (new; paired s0-vs-s1 comparison, asserts
identical window order, stamps sha256 of both fans). `code/fan_mbr_x8.py` gained `--fixed-cell score,K,T`
(writes `cell_source`); the default path re-run on s0 reproduces the original numbers exactly
(`ctl_refcv7_50400_mbr_s0_recheck.json`). Off-repo, on `D:/refcv7_eval_kit/fanctl/`: `s0/dump/fan_extras.npz`
sha256 ff2d0f86b1e3a3f6ff6979b27d8dd5d7a819d1386117b2e971f60668f99ffcb8,
`s1/dump/fan_extras.npz` sha256 81be3b94abcba89b4a74581b97c039b8fb037c2b8b3b1e67c4baff13b7d3ef4c (both MEASURED 2026-10-11).

**Incident (logged for the class): a self-aliased copy truncated 10 files to 0 bytes.** `D:/Projects/TanitAD` and
`E:\Projects\TanitAD` are ONE tree; `cp` between the two spellings opens the destination for write and truncates
the very file it is reading. Detected by `wc -c` / `samefile`; recovered by replaying the generating commands
(`fan_control_x8.py` rebuilt to 314 lines and its re-run equalled the banked JSON on 1,223 leaves), re-copying
8 `raw/fanctl_x8` files from `D:/refcv7_eval_kit/fanctl/*` through the E: spelling only, and replaying
`HANDOFF.md`. A targeted `find -type f -size 0` over this package found none left. Rule: never copy between the
two spellings; verify any copy by byte count or hash. (`D:/refcv7_eval_kit` is a SEPARATE tree and safe.)

**Doc-vs-raw conflict, restated (escalated; I do not edit `Project Steering/*`).** `GOALS_AND_CLAIMS.md` line
12229 and `MODEL_REGISTRY.md` section 15.3 (row ~5300) label 0.1914 / 0.4728 / +0.2813 [+0.2127, +0.3543] /
41.09 % as refcv7; they are **refc-base-30k** numbers (`ctl_repro_refc-base-30k.json`; `MODEL_REGISTRY` line ~3747
publishes 0.4728 as REF-C-base ADE@2s). True refcv7 (this block, raw JSON): selected 0.2772, oracle 0.1143,
sel_gap +0.1630, 44.17 %. The registry row still says the random-pick control is "not yet run".

**Next levers (RULE ZERO) -- executed above or named with their blocker.**
1. Inference-seed replicate -- EXECUTED (this block). 2. MBR pick -- EXECUTED, replicates.
3. Closed-loop T1 test of the MBR pick -- BLOCKED on a pre-registration with both outcomes committed and on
   fresh (non-A/B-half) episodes; unblock = Master Mind registers it. 4. Four-family scoring of the MBR pick --
   UNBLOCKED, 0 GPU (needs the per-window picks written out; a small code change in `fan_mbr_x8.py`): next EvalFlyWheel item.
5. A trained reranker over the fan -- BLOCKED on PI authorisation (new network + training compute).
6. Training-seed replicate of refcv7 (the missing floor) -- BLOCKED on compute and a PI decision.
7. Fix the FANCEIL numbers in GOALS_AND_CLAIMS / MODEL_REGISTRY -- owed by the Master Mind.
8. Land this UNTRACKED package (`2026-09-28-refcv7-standard-tests/`) -- owed by the single committer; the ckpt,
   both `fan_extras.npz` and the chain logs live only on `D:/refcv7_eval_kit`.
